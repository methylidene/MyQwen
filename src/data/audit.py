"""CLI for auditable, deterministic reasoning dataset inspection."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.io import save_json, write_jsonl

from .adapters import DatasetRegistry
from .schemas import DatasetLoadConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a normalized reasoning dataset split.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--config-name", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--source-path", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config = DatasetLoadConfig(
        dataset_name=args.dataset,
        split=args.split,
        config_name=args.config_name,
        revision=args.revision,
        source_path=args.source_path,
        cache_dir=args.cache_dir,
        max_samples=args.max_samples,
        shuffle=args.shuffle,
        seed=args.seed,
        purpose="audit",
        strict=False,
    )
    examples, fingerprint, report = DatasetRegistry.load(config)
    output = Path(args.output_dir)
    manifest = {
        "dataset_name": config.dataset_name,
        "config_name": config.config_name,
        "split": config.split,
        "revision": config.revision,
        "source_path": config.source_path,
        "seed": config.seed,
        "shuffle": config.shuffle,
        "max_samples": config.max_samples,
        "fingerprint": fingerprint.to_dict(),
    }
    save_json(manifest, output / "dataset_manifest.json")
    save_json(report.to_dict(), output / "audit_report.json")
    write_jsonl([item.to_dict() for item in examples[: min(20, len(examples))]], output / "sample_preview.jsonl")
    write_jsonl(report.invalid_examples, output / "invalid_examples.jsonl")
    save_json({"duplicate_uids": report.duplicate_uids}, output / "duplicate_report.json")


if __name__ == "__main__":
    main()
