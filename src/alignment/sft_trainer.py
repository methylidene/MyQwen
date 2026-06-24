from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.alignment.training_core import AlignmentConfig, BaseAlignmentTrainer, OptimizerConfig, TrainingMetrics
from src.data import DatasetLoadConfig, DatasetRegistry, PromptFormatter, ReasoningExample
from src.models.backend import ModelBackend, ModelBackendRegistry, ModelInputs, ModelLoadConfig
from src.utils.io import append_jsonl, ensure_dir, save_json


@dataclass
class SFTConfig:
    model_name_or_path: str
    output_dir: str
    train_file: str | None = None
    val_file: str | None = None
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    num_train_epochs: float = 1.0
    max_steps: int = -1
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_length: int = 512
    use_lora: bool = True
    backend_name: str = "huggingface"
    model_revision: str | None = None
    custom_factory_name: str | None = None
    device: str = "cuda"
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: list[str] | None = None
    dtype: str = "auto"
    bf16: bool = False
    fp16: bool = False
    gradient_checkpointing: bool = False
    trust_remote_code: bool = True
    logging_steps: int = 50
    checkpoint_interval: int = 0
    checkpoint_keep: int = 2
    resume_from_checkpoint: str | None = None
    dataset_name: str = "synthetic_arithmetic"
    dataset_split: str = "train"
    validation_split: str = "val"
    dataset_config_name: str | None = None
    dataset_revision: str | None = None
    dataset_cache_dir: str | None = None
    max_samples: int | None = None
    dataset_seed: int = 42
    dataset_shuffle: bool = False
    system_prompt: str | None = None
    final_answer_format: str = "<answer>{answer}</answer>"
    use_chat_template: bool = False
    deterministic_smoke: bool = False


class MathSFTDataset(Dataset):
    """Tokenized SFT examples with prompt tokens masked out of the loss."""

    def __init__(self, rows: list[ReasoningExample], tokenizer: Any, max_length: int, formatter: PromptFormatter, use_chat_template: bool):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.formatter = formatter
        self.use_chat_template = use_chat_template

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        prompt, completion = self.formatter.sft_text(row, chat=self.use_chat_template)
        prompt = prompt.strip() + "\n"
        encoded = self.tokenizer(prompt + completion, truncation=True, max_length=self.max_length, padding=False)
        prompt_encoded = self.tokenizer(prompt, truncation=True, max_length=self.max_length, padding=False)
        labels = list(encoded["input_ids"])
        prompt_length = min(len(prompt_encoded["input_ids"]), len(labels))
        labels[:prompt_length] = [-100] * prompt_length
        return {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate(features: list[dict[str, torch.Tensor]], pad_id: int) -> dict[str, torch.Tensor]:
    max_len = max(x["input_ids"].numel() for x in features)
    result: dict[str, torch.Tensor] = {}
    for key, fill in (("input_ids", pad_id), ("attention_mask", 0), ("labels", -100)):
        result[key] = torch.stack([
            torch.nn.functional.pad(item[key], (0, max_len - item[key].numel()), value=fill)
            for item in features
        ])
    return result


class SFTTrainerEngine(BaseAlignmentTrainer):
    """Self-hosted SFT engine using ModelBackend and DatasetRegistry."""

    def __init__(self, backend: ModelBackend, config: SFTConfig, output_dir: str | Path) -> None:
        alignment = AlignmentConfig(
            seed=config.dataset_seed,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            gradient_checkpointing=config.gradient_checkpointing,
            checkpoint_interval=config.checkpoint_interval,
            checkpoint_keep=config.checkpoint_keep,
            resume_from_checkpoint=config.resume_from_checkpoint,
            deterministic_smoke=config.deterministic_smoke,
        )
        super().__init__(backend, OptimizerConfig(config.learning_rate, config.weight_decay), alignment, output_dir)
        self.config = config
        self.output_dir = Path(output_dir)

    def _run_validation(self, loader: DataLoader) -> dict[str, float]:
        self.backend.eval()
        total_loss, total_tokens = 0.0, 0
        with torch.no_grad():
            for batch in loader:
                batch = {name: value.to(self.backend.device) for name, value in batch.items()}
                output = self.backend.forward(ModelInputs(**batch, use_cache=False))
                tokens = int((batch["labels"] != -100).sum().item())
                total_loss += float(output.loss.detach().cpu()) * max(tokens, 1)
                total_tokens += tokens
        self.backend.train()
        return {"validation_loss": total_loss / max(total_tokens, 1), "validation_tokens": total_tokens}

    def train(self, train_loader: DataLoader, validation_loader: DataLoader | None = None) -> TrainerState:
        self.maybe_resume()
        self.backend.train()
        self.optimizer.zero_grad(set_to_none=True)
        updates_target = self.config.max_steps if self.config.max_steps > 0 else max(1, int(len(train_loader) * self.config.num_train_epochs / self.config.gradient_accumulation_steps))
        progress = tqdm(total=updates_target, initial=self.state.global_step, desc="sft")

        def apply_update(raw_loss: torch.Tensor, tokens: int) -> bool:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.state.global_step += 1
            self.state.optimizer_steps += 1
            metric = TrainingMetrics(step=self.state.global_step, loss=float(raw_loss.detach().cpu()), train_tokens=tokens)
            append_jsonl([metric.to_dict()], self.output_dir / "train_metrics.jsonl")
            if self.state.global_step % max(self.config.logging_steps, 1) == 0:
                print(f"sft step={self.state.global_step} loss={metric.loss:.6f} tokens={tokens}", flush=True)
            if self.config.checkpoint_interval and self.state.global_step % self.config.checkpoint_interval == 0:
                self.checkpoint(asdict(self.config))
            progress.update(1)
            return self.state.global_step >= updates_target

        while self.state.global_step < updates_target:
            last_loss: torch.Tensor | None = None
            last_tokens = 0
            for batch in train_loader:
                batch = {name: value.to(self.backend.device) for name, value in batch.items()}
                output = self.backend.forward(ModelInputs(**batch, use_cache=False))
                if output.loss is None:
                    raise RuntimeError("SFT backend did not return a loss for labels.")
                last_loss = output.loss
                (last_loss / self.config.gradient_accumulation_steps).backward()
                last_tokens = int((batch["labels"] != -100).sum().item())
                self.state.train_tokens += last_tokens
                self.state.micro_step += 1
                if self.state.micro_step % self.config.gradient_accumulation_steps == 0 and apply_update(last_loss, last_tokens):
                    break
            if self.state.global_step >= updates_target:
                break
            # Preserve tail gradients rather than dropping a short final accumulation.
            if last_loss is not None and self.state.micro_step % self.config.gradient_accumulation_steps:
                if apply_update(last_loss, last_tokens):
                    break
            self.state.epoch += 1
        progress.close()
        if validation_loader is not None:
            save_json(self._run_validation(validation_loader), self.output_dir / "eval_metrics.json")
        return self.state


def _load_backend(config: SFTConfig) -> ModelBackend:
    if config.bf16:
        config.dtype = "bf16"
    if config.fp16:
        config.dtype = "fp16"
    model_path = config.model_name_or_path
    if config.resume_from_checkpoint:
        model_path = str(Path(config.resume_from_checkpoint) / "model")
    elif (Path(model_path) / "checkpoint").exists():
        model_path = str(Path(model_path) / "checkpoint")
    return ModelBackendRegistry.from_config(ModelLoadConfig(
        model_name_or_path=model_path,
        revision=config.model_revision,
        backend_name=config.backend_name,
        custom_factory_name=config.custom_factory_name,
        device=config.device,
        dtype=config.dtype,
        trust_remote_code=config.trust_remote_code,
        use_lora=config.use_lora,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        lora_target_modules=config.lora_target_modules,
    ))


def _load_examples(config: SFTConfig, source_path: str | None, split: str, purpose: str) -> list[ReasoningExample]:
    examples, _, _ = DatasetRegistry.load(DatasetLoadConfig(
        dataset_name=config.dataset_name,
        split=split,
        config_name=config.dataset_config_name,
        revision=config.dataset_revision,
        source_path=source_path,
        cache_dir=config.dataset_cache_dir,
        max_samples=config.max_samples,
        shuffle=config.dataset_shuffle if purpose == "train" else False,
        seed=config.dataset_seed,
        purpose=purpose,
    ))
    return examples


def train_sft(config: SFTConfig) -> None:
    """Backward-compatible SFT entry point backed by :class:`SFTTrainerEngine`."""
    ensure_dir(config.output_dir)
    backend = _load_backend(config)
    formatter = PromptFormatter(config.system_prompt, config.final_answer_format)
    train_examples = _load_examples(config, config.train_file, config.dataset_split, "train")
    tokenizer = backend.tokenizer
    train_loader = DataLoader(
        MathSFTDataset(train_examples, tokenizer, config.max_length, formatter, config.use_chat_template),
        batch_size=config.batch_size,
        shuffle=not config.deterministic_smoke,
        collate_fn=lambda rows: collate(rows, tokenizer.pad_token_id),
    )
    validation_loader = None
    if config.val_file:
        validation_examples = _load_examples(config, config.val_file, config.validation_split, "eval")
        validation_loader = DataLoader(
            MathSFTDataset(validation_examples, tokenizer, config.max_length, formatter, config.use_chat_template),
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=lambda rows: collate(rows, tokenizer.pad_token_id),
        )
    save_json(asdict(config), Path(config.output_dir) / "config.json")
    engine = SFTTrainerEngine(backend, config, config.output_dir)
    state = engine.train(train_loader, validation_loader)
    backend.save_pretrained(Path(config.output_dir) / "checkpoint")  # Legacy checkpoint path.
    engine.checkpoint(asdict(config))
    save_json({"final_step": state.global_step, "train_tokens": state.train_tokens}, Path(config.output_dir) / "metrics.json")
