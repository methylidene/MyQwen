from __future__ import annotations

from collections import defaultdict
from typing import Any

from .rewards import rule_based_reward


def aggregate_predictions(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[row.get("difficulty", "unknown")].append(row)
    metrics = {}
    for diff, items in buckets.items():
        n = max(len(items), 1)
        metrics[diff] = {
            "accuracy": sum(x["reward"]["accuracy"] for x in items) / n,
            "reward": sum(x["reward"]["total_reward"] for x in items) / n,
            "format_pass_rate": sum(x["reward"]["format_pass"] for x in items) / n,
            "invalid_rate": sum(x["reward"]["invalid"] for x in items) / n,
            "kl": sum(x.get("kl", 0.0) for x in items) / n,
            "avg_response_length": sum(len(x.get("response", "").split()) for x in items) / n,
        }
    return metrics


def evaluate_with_generator(generator, samples: list[dict[str, Any]], max_new_tokens: int = 128) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prompts = [x["prompt"] for x in samples]
    results = generator.generate(prompts, max_new_tokens=max_new_tokens, use_cache=True)
    predictions = []
    for sample, result in zip(samples, results):
        reward = rule_based_reward(result.generated_text, sample["answer"])
        predictions.append(
            {
                "id": sample["id"],
                "difficulty": sample["difficulty"],
                "prompt": sample["prompt"],
                "gold_answer": sample["answer"],
                "response": result.generated_text,
                "reward": reward,
                "kl": 0.0,
            }
        )
    return predictions, aggregate_predictions(predictions)
