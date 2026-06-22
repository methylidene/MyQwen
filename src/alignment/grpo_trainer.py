from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from tqdm import tqdm

from src.alignment.kl_utils import approximate_kl, sequence_logprobs
from src.alignment.rewards import rule_based_reward
from src.utils.io import append_jsonl, ensure_dir, read_jsonl, save_json


@dataclass
class GRPOConfig:
    model_name_or_path: str
    train_file: str
    val_file: str | None
    output_dir: str
    group_size: int = 4
    max_steps: int = 1000
    beta_kl: float = 0.02
    learning_rate: float = 5e-6
    temperature: float = 0.8
    top_p: float = 0.95
    max_new_tokens: int = 128
    max_prompt_length: int = 384
    batch_size: int = 1
    clip_eps: float = 0.2
    use_lora: bool = True
    trust_remote_code: bool = True
    dtype: str = "bf16"
    bf16: bool = False
    fp16: bool = False
    logging_steps: int = 10


def _dtype(name: str):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def train_grpo(config: GRPOConfig) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # 1. 加载配置
    ensure_dir(config.output_dir)
    save_json(config.__dict__, f"{config.output_dir}/config.json")
    if config.bf16:
        config.dtype = "bf16"
    if config.fp16:
        config.dtype = "fp16"
    model_path = config.model_name_or_path
    candidate_checkpoint = __import__("pathlib").Path(model_path) / "checkpoint"
    if candidate_checkpoint.exists():
        model_path = str(candidate_checkpoint)
    
    # 2. 加载tokenizer和模型
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=config.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=_dtype(config.dtype),
        trust_remote_code=config.trust_remote_code,
        low_cpu_mem_usage=True,
    )
    ref = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=_dtype(config.dtype),
        trust_remote_code=config.trust_remote_code,
        low_cpu_mem_usage=True,
    )

    # 3. 设置lora和optimizer
    if config.use_lora and hasattr(model, "peft_config"):
        if hasattr(model, "enable_adapter_layers"):
            model.enable_adapter_layers()
        for name, param in model.named_parameters():
            param.requires_grad_("lora_" in name or "modules_to_save" in name)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Continuing existing PEFT adapter with {trainable} trainable parameters.", flush=True)
    elif config.use_lora:
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(
            model,
            LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"),
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    ref.to(device).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # 4.读取训练数据
    rows = read_jsonl(config.train_file)
    progress = tqdm(range(config.max_steps), desc="grpo")
    model.train()

    # 5.每一步是一道题
    for step in progress:
        row = rows[step % len(rows)]
        prompt = row["prompt"]
        enc = tokenizer([prompt] * config.group_size, return_tensors="pt", padding=True, truncation=True, max_length=config.max_prompt_length).to(device)
        with torch.no_grad():
            # 模型生成Group Size个回答
            generated = model.generate(
                **enc,
                do_sample=True,     #需要随机采样
                temperature=config.temperature, #控制采样的随机程度，值越大越随机，值越小越确定（作用于最后一层的softmax函数，温度越高烫越平）
                top_p=config.top_p, # top_p越高，候选词采样范围越大，用来兜底不至于产生废话
                max_new_tokens=config.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
        prompt_len = enc["input_ids"].shape[-1]
        responses = tokenizer.batch_decode(generated[:, prompt_len:], skip_special_tokens=True)#只保留需要的答案部分
        rewards = [rule_based_reward(x, row["answer"]) for x in responses]
        reward_t = torch.tensor([x["total_reward"] for x in rewards], dtype=torch.float32, device=device)
        adv = (reward_t - reward_t.mean()) / (reward_t.std(unbiased=False) + 1e-6)

        labels = generated[:, 1:].contiguous()
        attn = (generated != tokenizer.pad_token_id).long()
        mask = attn[:, 1:].float()
        # Only optimize generated response tokens; prompt tokens are conditioning context.
        mask[:, : max(prompt_len - 1, 0)] = 0.0
        out = model(input_ids=generated[:, :-1], attention_mask=attn[:, :-1])
        with torch.no_grad():
            ref_out = ref(input_ids=generated[:, :-1], attention_mask=attn[:, :-1])
        logp = sequence_logprobs(out.logits, labels, mask)
        ref_logp = sequence_logprobs(ref_out.logits, labels, mask)
        seq_logp = logp.sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)
        kl = approximate_kl(logp, ref_logp, mask)
        policy_loss = -(adv.detach() * seq_logp).mean()
        kl_loss = config.beta_kl * kl
        loss = policy_loss + kl_loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        metrics = {
            "step": step,
            "loss": float(loss.detach().cpu()),
            "policy_loss": float(policy_loss.detach().cpu()),
            "kl_loss": float(kl_loss.detach().cpu()),
            "kl": float(kl.detach().cpu()),
            "reward_mean": float(reward_t.mean().detach().cpu()),
            "reward_std": float(reward_t.std(unbiased=False).detach().cpu()),
            "accuracy": sum(x["accuracy"] for x in rewards) / len(rewards),
            "format_pass_rate": sum(x["format_pass"] for x in rewards) / len(rewards),
            "invalid_rate": sum(x["invalid"] for x in rewards) / len(rewards),
            "avg_response_length": sum(len(x.split()) for x in responses) / len(responses),
        }
        metrics["response_length"] = metrics["avg_response_length"]
        append_jsonl([metrics], f"{config.output_dir}/train_metrics.jsonl")
        if step % max(config.logging_steps, 1) == 0:
            print(
                "grpo "
                f"step={step} loss={metrics['loss']:.6f} policy_loss={metrics['policy_loss']:.6f} "
                f"kl_loss={metrics['kl_loss']:.6f} kl={metrics['kl']:.6f} "
                f"reward_mean={metrics['reward_mean']:.6f} reward_std={metrics['reward_std']:.6f} "
                f"accuracy={metrics['accuracy']:.4f} format_pass_rate={metrics['format_pass_rate']:.4f} "
                f"invalid_rate={metrics['invalid_rate']:.4f} avg_response_length={metrics['response_length']:.2f}",
                flush=True,
            )
        sample_rows = [
            {"step": step, "prompt": prompt, "response": r, "gold_answer": row["answer"], "reward": rb}
            for r, rb in zip(responses, rewards)
        ]
        append_jsonl(sample_rows, f"{config.output_dir}/sampled_responses.jsonl")
        append_jsonl(sample_rows, f"{config.output_dir}/samples.jsonl")
    model.save_pretrained(f"{config.output_dir}/checkpoint")
    tokenizer.save_pretrained(f"{config.output_dir}/checkpoint")
    save_json({"final_step": config.max_steps}, f"{config.output_dir}/metrics.json")
