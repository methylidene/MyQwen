from __future__ import annotations

from collections import defaultdict
from typing import Any


def compare_token_outputs(reference: list[int], candidate: list[int]) -> dict[str, Any]:
    first_diff = None
    for i, (a, b) in enumerate(zip(reference, candidate)):
        if a != b:
            first_diff = {"index": i, "reference_token": a, "candidate_token": b}
            break
    if first_diff is None and len(reference) != len(candidate):
        first_diff = {
            "index": min(len(reference), len(candidate)),
            "reference_token": reference[min(len(reference), len(candidate))] if len(reference) > len(candidate) else None,
            "candidate_token": candidate[min(len(reference), len(candidate))] if len(candidate) > len(reference) else None,
        }
    return {
        "exact_match": reference == candidate,
        "first_difference": first_diff,
        "reference_length": len(reference),
        "candidate_length": len(candidate),
    }


def summarize_ablation(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["mode"]].append(row)
    summary = []
    for mode, items in groups.items():
        n = max(len(items), 1)
        summary.append(
            {
                "mode": mode,
                "num_prompts": len(items),
                "avg_total_latency": sum(x["total_latency"] for x in items) / n,
                "avg_latency_per_token": sum(x["avg_latency_per_token"] for x in items) / n,
                "avg_tokens_per_second": sum(x["tokens_per_second"] for x in items) / n,
                "output_match_rate": sum(bool(x.get("exact_output_match_with_no_cache", False)) for x in items) / n,
                "avg_cache_seq_len": sum(x.get("avg_cache_seq_len", 0.0) for x in items) / n,
                "max_cache_seq_len": max(x.get("max_cache_seq_len", 0) for x in items),
                "cache_rebuild_count": sum(x.get("cache_rebuild_count", 0) for x in items),
            }
        )
    return summary


def write_ablation_report(config: dict[str, Any], summary: list[dict[str, Any]], output_path) -> None:
    from pathlib import Path

    lines = [
        "# KV Cache Ablation Report",
        "",
        "## Experiment Config",
        "```json",
        __import__("json").dumps(config, ensure_ascii=False, indent=2),
        "```",
        "",
        f"Prompt count: {config.get('num_prompts', 'unknown')}",
        "",
        "## Speed And Consistency",
        "",
        "| mode | avg total latency | tok/s | match rate | avg cache len | max cache len | rebuilds |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['mode']} | {row['avg_total_latency']:.4f} | {row['avg_tokens_per_second']:.2f} | "
            f"{row['output_match_rate']:.3f} | {row['avg_cache_seq_len']:.1f} | {row['max_cache_seq_len']} | {row['cache_rebuild_count']} |"
        )
    lines += [
        "",
        "## Analysis",
        "",
        "Full cache avoids recomputing attention over the whole prefix at every decode step, so decode latency usually drops as contexts grow.",
        "Window cache bounds memory and cache length, but trimming or rebuilding removes long-range context from attention and can change greedy outputs.",
    ]
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
