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
from src.evaluation.evaluator import write_evaluation_outputs
from src.data import DatasetLoadConfig, DatasetRegistry, PromptFormatter
from src.inference.kv_cache_generator import KVCacheGenerator
from src.utils.io import save_json, write_bilingual_readme, write_csv


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--test_file")
    p.add_argument("--dataset_name", default="synthetic_arithmetic")
    p.add_argument("--split", default="test")
    p.add_argument("--dataset_config_name", default=None)
    p.add_argument("--dataset_revision", default=None)
    p.add_argument("--dataset_cache_dir", default=None)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--dataset_seed", type=int, default=42)
    p.add_argument("--dataset_shuffle", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--system_prompt", default=None)
    p.add_argument("--final_answer_format", default="<answer>{answer}</answer>")
    p.add_argument("--checkpoint_dirs", nargs="+", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--backend_name", default="huggingface")
    p.add_argument("--custom_factory_name", default=None)
    args = p.parse_args()
    samples, fingerprint, _ = DatasetRegistry.load(
        DatasetLoadConfig(
            dataset_name=args.dataset_name,
            split=args.split,
            config_name=args.dataset_config_name,
            revision=args.dataset_revision,
            source_path=args.test_file,
            cache_dir=args.dataset_cache_dir,
            max_samples=args.max_samples,
            shuffle=args.dataset_shuffle,
            seed=args.dataset_seed,
            purpose="eval",
        )
    )
    formatter = PromptFormatter(args.system_prompt, args.final_answer_format)
    output_root = Path(args.output_dir)
    write_bilingual_readme(
        output_root,
        title="Alignment Evaluation Batch",
        english="Unified evaluation batch for multiple checkpoints. Per-checkpoint outputs live in child directories; report.md remains English-only.",
        chinese="这是多个 checkpoint 的统一评测批次目录。各 checkpoint 的产物位于子目录中；report.md 保持英文输出。",
        preserve_existing=False,
    )
    summary_rows = []
    report_lines = ["# Alignment Evaluation Report", ""]
    for ckpt in args.checkpoint_dirs:
        name = Path(ckpt).name
        model_path = str(Path(ckpt) / "checkpoint") if (Path(ckpt) / "checkpoint").exists() else ckpt
        gen = KVCacheGenerator.from_pretrained(model_path, args.device, args.dtype, args.trust_remote_code, args.backend_name, args.custom_factory_name)
        preds, metrics = evaluate_with_generator(gen, samples, args.max_new_tokens, formatter)
        exp_dir = output_root / name
        config_payload = vars(args) | {"experiment_name": name, "dataset_fingerprint": fingerprint.to_dict()}
        save_json(config_payload, exp_dir / "config.json")
        unified = write_evaluation_outputs(exp_dir, preds, ckpt, config_payload)
        for diff, row in unified["groups"]["difficulty"].items():
            summary_rows.append({"experiment": name, "difficulty": diff, **{key: value for key, value in row.items() if not isinstance(value, (dict, list))}})
        report_lines += [f"## {name}", "", "```json", __import__("json").dumps(unified["overall"], indent=2), "```", ""]
    write_csv(summary_rows, output_root / "summary.csv")
    save_json({"experiments": summary_rows}, output_root / "metrics.json")
    report_lines += [
        "## Analysis Prompts",
        "Compare SFT-only vs SFT-continued for accuracy and format gains, compare GRPO group sizes for reward/KL behavior, and inspect invalid-rate decreases and policy drift.",
    ]
    (output_root / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
