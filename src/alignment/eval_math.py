"""Backward-compatible facade over the unified training-independent evaluator."""
from __future__ import annotations

from typing import Any

from src.data import PromptFormatter, ReasoningExample
from src.evaluation.evaluator import EvaluationGenerationConfig, Evaluator, KVGenerationBackend, metrics_by_group


def aggregate_predictions(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return the legacy difficulty map from canonical evaluator predictions."""
    return metrics_by_group(rows)["groups"]["difficulty"]


def evaluate_with_generator(
    generator: Any,
    samples: list[ReasoningExample],
    max_new_tokens: int = 128,
    formatter: PromptFormatter | None = None,
    batch_size: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    formatter = formatter or PromptFormatter()
    evaluator = Evaluator(KVGenerationBackend(generator), formatter)
    predictions = evaluator.evaluate(samples, EvaluationGenerationConfig(max_new_tokens=max_new_tokens, batch_size=batch_size))
    return predictions, aggregate_predictions(predictions)
