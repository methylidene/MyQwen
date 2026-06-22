from __future__ import annotations

import argparse


def add_generation_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=["auto", "fp32", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--use_cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache_window", type=int, default=-1)
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    return parser


def torch_dtype(dtype: str):
    import torch

    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp32":
        return torch.float32
    return "auto"


def peak_gpu_memory_mb(device: str) -> float:
    import torch

    if "cuda" in str(device) and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0
