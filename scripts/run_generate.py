#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse

from src.inference.generation_utils import add_generation_args
from src.inference.kv_cache_generator import KVCacheGenerator
from src.utils.io import save_json, write_jsonl
from src.utils.seed import set_seed


def main() -> None:
    p = add_generation_args(argparse.ArgumentParser())
    p.add_argument("--prompt", default="Solve 12 + 7. Use <reasoning> and <answer> tags.")
    p.add_argument("--backend_name", default="huggingface")
    p.add_argument("--custom_factory_name", default=None)
    args = p.parse_args()
    set_seed(args.seed)
    gen = KVCacheGenerator.from_pretrained(args.model_name_or_path, args.device, args.dtype, args.trust_remote_code, args.backend_name, args.custom_factory_name)
    results = gen.generate([args.prompt], args.max_new_tokens, args.use_cache, args.cache_window)
    save_json(vars(args), f"{args.output_dir}/config.json")
    write_jsonl([r.__dict__ for r in results], f"{args.output_dir}/outputs/results.jsonl")
    save_json({"num_prompts": len(results), "total_latency": sum(r.total_latency for r in results)}, f"{args.output_dir}/metrics.json")


if __name__ == "__main__":
    main()
