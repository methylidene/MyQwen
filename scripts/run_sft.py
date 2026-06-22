#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import yaml

from src.alignment.sft_trainer import SFTConfig, train_sft


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config")
    p.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--train_file", required=True)
    p.add_argument("--val_file")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--learning_rate", type=float)
    p.add_argument("--num_train_epochs", type=float)
    p.add_argument("--max_steps", type=int)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--gradient_accumulation_steps", type=int)
    p.add_argument("--max_length", type=int)
    p.add_argument("--use_lora", action=argparse.BooleanOptionalAction)
    p.add_argument("--lora_r", type=int)
    p.add_argument("--lora_alpha", type=int)
    p.add_argument("--lora_dropout", type=float)
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction)
    p.add_argument("--fp16", action=argparse.BooleanOptionalAction)
    p.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction)
    p.add_argument("--logging_steps", type=int)
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8")) if args.config else {}
    for k, v in vars(args).items():
        if k != "config" and v is not None:
            cfg[k] = v
    train_sft(SFTConfig(**cfg))


if __name__ == "__main__":
    main()
