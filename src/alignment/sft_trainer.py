from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.alignment.synthetic_math import target_text
from src.utils.io import append_jsonl, ensure_dir, read_jsonl, save_json


@dataclass
class SFTConfig:
    model_name_or_path: str
    train_file: str
    val_file: str | None
    output_dir: str
    learning_rate: float = 2e-5
    num_train_epochs: float = 1.0
    max_steps: int = -1
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_length: int = 512
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    bf16: bool = False
    fp16: bool = False
    trust_remote_code: bool = True
    logging_steps: int = 50


class MathSFTDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer, max_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        prompt = row["prompt"].strip() + "\n"
        text = prompt + target_text(row["answer"])
        enc = self.tokenizer(text, truncation=True, max_length=self.max_length, padding=False)
        prompt_enc = self.tokenizer(prompt, truncation=True, max_length=self.max_length, padding=False)
        labels = enc["input_ids"].copy()
        labels[: min(len(prompt_enc["input_ids"]), len(labels))] = [-100] * min(len(prompt_enc["input_ids"]), len(labels))
        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate(features: list[dict[str, torch.Tensor]], pad_id: int) -> dict[str, torch.Tensor]:
    max_len = max(x["input_ids"].numel() for x in features)
    batch = {}
    for key in ["input_ids", "attention_mask", "labels"]:
        fill = -100 if key == "labels" else (0 if key == "attention_mask" else pad_id)
        batch[key] = torch.stack(
            [
                torch.nn.functional.pad(x[key], (0, max_len - x[key].numel()), value=fill)
                for x in features
            ]
        )
    return batch


def train_sft(config: SFTConfig) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ensure_dir(config.output_dir)
    save_json(config.__dict__, f"{config.output_dir}/config.json")
    dtype = torch.bfloat16 if config.bf16 else torch.float16 if config.fp16 else torch.float32
    model_path = config.model_name_or_path
    candidate_checkpoint = __import__("pathlib").Path(model_path) / "checkpoint"
    if candidate_checkpoint.exists():
        model_path = str(candidate_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=config.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=config.trust_remote_code,
        low_cpu_mem_usage=True,
    )
    if config.use_lora and hasattr(model, "peft_config"):
        if hasattr(model, "enable_adapter_layers"):
            model.enable_adapter_layers()
        for name, param in model.named_parameters():
            param.requires_grad_("lora_" in name or "modules_to_save" in name)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Continuing existing PEFT adapter with {trainable} trainable parameters.", flush=True)
    elif config.use_lora:
        from peft import LoraConfig, get_peft_model

        lora = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_rows = read_jsonl(config.train_file)
    loader = DataLoader(
        MathSFTDataset(train_rows, tokenizer, config.max_length),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda xs: collate(xs, tokenizer.pad_token_id),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    global_step = 0
    model.train()
    opt.zero_grad(set_to_none=True)
    max_steps = config.max_steps if config.max_steps and config.max_steps > 0 else int(len(loader) * config.num_train_epochs)
    progress = tqdm(total=max_steps, desc="sft")
    while global_step < max_steps:
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / config.gradient_accumulation_steps
            loss.backward()
            if (global_step + 1) % config.gradient_accumulation_steps == 0:
                opt.step()
                opt.zero_grad(set_to_none=True)
            loss_value = float(loss.detach().cpu())
            append_jsonl([{"step": global_step, "loss": loss_value}], f"{config.output_dir}/train_metrics.jsonl")
            if global_step % max(config.logging_steps, 1) == 0:
                print(f"sft step={global_step} loss={loss_value:.6f}", flush=True)
            global_step += 1
            progress.update(1)
            if global_step >= max_steps:
                break
    progress.close()
    model.save_pretrained(f"{config.output_dir}/checkpoint")
    tokenizer.save_pretrained(f"{config.output_dir}/checkpoint")
    final_metrics = {"final_step": global_step}
    save_json(final_metrics, f"{config.output_dir}/metrics.json")
    save_json(final_metrics, f"{config.output_dir}/eval_metrics.json")
