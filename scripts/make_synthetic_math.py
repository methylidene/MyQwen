#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from pathlib import Path

from src.alignment.synthetic_math import generate_dataset
from src.utils.io import save_json, write_jsonl


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_train", type=int, default=10000)
    p.add_argument("--num_val", type=int, default=1000)
    p.add_argument("--num_test", type=int, default=1000)
    p.add_argument("--easy_ratio", type=float, default=0.4)
    p.add_argument("--medium_ratio", type=float, default=0.4)
    p.add_argument("--hard_ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--profile", choices=["target_v1", "target_v2", "target_v3"], default="target_v1")
    args = p.parse_args()
    data = generate_dataset(
        num_train=args.num_train,
        num_val=args.num_val,
        num_test=args.num_test,
        easy_ratio=args.easy_ratio,
        medium_ratio=args.medium_ratio,
        hard_ratio=args.hard_ratio,
        seed=args.seed,
        profile=args.profile,
    )
    out = Path(args.output_dir)
    write_jsonl(data["train"], out / "train.jsonl")
    write_jsonl(data["val"], out / "val.jsonl")
    write_jsonl(data["test"], out / "test.jsonl")
    save_json(vars(args), out / "config.json")
    save_json({k: len(v) for k, v in data.items()}, out / "metrics.json")


if __name__ == "__main__":
    main()
