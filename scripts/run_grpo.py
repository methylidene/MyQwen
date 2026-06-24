#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import yaml

from src.alignment.grpo_trainer import GRPOConfig, train_grpo


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config")
    p.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-3B")
    p.add_argument("--train_file")
    p.add_argument("--val_file")
    p.add_argument("--dataset_name", default=None)
    p.add_argument("--dataset_split", default=None)
    p.add_argument("--dataset_config_name", default=None)
    p.add_argument("--dataset_revision", default=None)
    p.add_argument("--dataset_cache_dir", default=None)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--dataset_seed", type=int, default=None)
    p.add_argument("--dataset_shuffle", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--system_prompt", default=None)
    p.add_argument("--final_answer_format", default=None)
    p.add_argument("--use_chat_template", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--group_size", type=int)
    p.add_argument("--max_steps", type=int)
    p.add_argument("--max_generated_completion_tokens", type=int)
    p.add_argument("--beta_kl", type=float)
    p.add_argument("--learning_rate", type=float)
    p.add_argument("--temperature", type=float)
    p.add_argument("--top_p", type=float)
    p.add_argument("--max_new_tokens", type=int)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--forward_micro_batch_size", type=int)
    p.add_argument("--dtype", default=None)
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction)
    p.add_argument("--fp16", action=argparse.BooleanOptionalAction)
    p.add_argument("--logging_steps", type=int)
    p.add_argument("--weight_decay", type=float)
    p.add_argument("--clip_eps", type=float)
    p.add_argument("--entropy_coef", type=float)
    p.add_argument("--advantage_epsilon", type=float)
    p.add_argument("--use_reference_policy", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--checkpoint_interval", type=int)
    p.add_argument("--checkpoint_keep", type=int)
    p.add_argument("--resume_from_checkpoint", default=None)
    p.add_argument("--deterministic_smoke", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--backend_name", default=None)
    p.add_argument("--custom_factory_name", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--lora_target_modules", nargs="+", default=None)
    p.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction)
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8")) if args.config else {}
    for k, v in vars(args).items():
        if k != "config" and v is not None:
            cfg[k] = v
    train_grpo(GRPOConfig(**cfg))


if __name__ == "__main__":
    main()
