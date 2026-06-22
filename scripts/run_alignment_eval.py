#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from pathlib import Path

from src.alignment.eval_math import evaluate_with_generator
from src.inference.kv_cache_generator import KVCacheGenerator
from src.utils.io import read_jsonl, save_json, write_csv, write_jsonl


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--test_file", required=True)
    p.add_argument("--checkpoint_dirs", nargs="+", required=True)
    p.add_argument("--output_dir", default="outputs/alignment_eval")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()
    samples = read_jsonl(args.test_file)
    summary_rows = []
    report_lines = ["# Alignment Evaluation Report", ""]
    for ckpt in args.checkpoint_dirs:
        name = Path(ckpt).name
        model_path = str(Path(ckpt) / "checkpoint") if (Path(ckpt) / "checkpoint").exists() else ckpt
        gen = KVCacheGenerator.from_pretrained(model_path, args.device, args.dtype, args.trust_remote_code)
        preds, metrics = evaluate_with_generator(gen, samples, args.max_new_tokens)
        exp_dir = Path(args.output_dir) / name
        save_json(vars(args) | {"experiment_name": name}, exp_dir / "config.json")
        write_jsonl(preds, exp_dir / "predictions.jsonl")
        save_json(metrics, exp_dir / "metrics_by_difficulty.json")
        for diff, row in metrics.items():
            summary_rows.append({"experiment": name, "difficulty": diff, **row})
        report_lines += [f"## {name}", "", "```json", __import__("json").dumps(metrics, indent=2), "```", ""]
    write_csv(summary_rows, Path(args.output_dir) / "summary.csv")
    save_json({"experiments": summary_rows}, Path(args.output_dir) / "metrics.json")
    report_lines += [
        "## Analysis Prompts",
        "Compare SFT-only vs SFT-continued for accuracy and format gains, compare GRPO group sizes for reward/KL behavior, and inspect invalid-rate decreases and policy drift.",
    ]
    (Path(args.output_dir) / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
