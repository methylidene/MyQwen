"""Compare unified or legacy evaluation result directories."""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from pathlib import Path
from typing import Any

from src.utils.io import save_json, write_csv

from .evaluator import aggregate_rows, load_predictions_compatible


def bootstrap_ci(values: list[float], samples: int, seed: int) -> tuple[float, float] | None:
    if not values or samples <= 0:
        return None
    rng = random.Random(seed)
    means = [sum(rng.choices(values, k=len(values))) / len(values) for _ in range(samples)]
    means.sort()
    return means[int(0.025 * (len(means) - 1))], means[int(0.975 * (len(means) - 1))]


def load_run(path: str | Path) -> tuple[str, dict[str, Any], list[dict[str, Any]] | None]:
    root = Path(path)
    prediction = root / "predictions.jsonl"
    if prediction.exists():
        rows = load_predictions_compatible(prediction)
        return root.name, aggregate_rows(rows), rows
    metrics = root / "metrics.json"
    if metrics.exists():
        data = json.loads(metrics.read_text(encoding="utf-8"))
        if "overall" in data:
            return root.name, data["overall"], None
        if "accuracy" in data:
            return root.name, data, None
    summary = root / "summary.csv"
    if summary.exists():
        with summary.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        return root.name, {"legacy_summary_rows": rows, "n": len(rows)}, None
    raise FileNotFoundError(f"No predictions.jsonl, metrics.json or summary.csv in {root}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    records = []
    report = ["# Evaluation Comparison", "", "## Runs", ""]
    seed_values: dict[str, list[float]] = {}
    for run in args.runs:
        name, metric, rows = load_run(run)
        record = {"run": name, **{key: value for key, value in metric.items() if not isinstance(value, (dict, list))}}
        records.append(record)
        if metric.get("accuracy") is not None:
            seed_values.setdefault(name.split("-seed", 1)[0], []).append(float(metric["accuracy"]))
        report.extend((f"## {name}", "```json", json.dumps(metric, indent=2), "```", ""))
    aggregates = {}
    for name, values in seed_values.items():
        if len(values) > 1:
            aggregates[name] = {"n_seeds": len(values), "accuracy_mean": statistics.mean(values), "accuracy_std": statistics.stdev(values), "accuracy_bootstrap_ci": bootstrap_ci(values, args.bootstrap_samples, args.seed)}
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    write_csv(records, destination / "summary.csv")
    save_json({"runs": records, "multi_seed": aggregates}, destination / "metrics.json")
    (destination / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
