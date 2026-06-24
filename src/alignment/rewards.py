from __future__ import annotations

import re
from typing import Any

from src.data.answers import AnswerExtractor


REASONING_RE = re.compile(r"<reasoning>.*?</reasoning>", re.IGNORECASE | re.DOTALL)


def extract_answer(output: str) -> str | None:
    """Compatibility wrapper for tagged model-output extraction."""
    return AnswerExtractor.extract_tagged(output)


def normalize_number(text: str | None):
    """Compatibility wrapper for numeric answer normalization."""
    return AnswerExtractor.normalize_number(text)


def rule_based_reward(output: str, gold_answer: str) -> dict[str, Any]:
    parsed = extract_answer(output)
    pred_num = normalize_number(parsed)
    gold_num = normalize_number(gold_answer)
    format_ok = parsed is not None and pred_num is not None
    correct = bool(format_ok and gold_num is not None and pred_num == gold_num)
    reasoning_ok = bool(REASONING_RE.search(output or ""))
    breakdown = {
        "correctness_reward": 1.0 if correct else 0.0,
        "format_reward": 0.2 if format_ok else 0.0,
        "reasoning_reward": 0.1 if reasoning_ok else 0.0,
        "invalid_penalty": 0.0 if format_ok else -0.5,
        "accuracy": 1.0 if correct else 0.0,
        "format_pass": 1.0 if format_ok else 0.0,
        "invalid": 0.0 if format_ok else 1.0,
        "parsed_answer": parsed,
    }
    breakdown["total_reward"] = (
        breakdown["correctness_reward"]
        + breakdown["format_reward"]
        + breakdown["reasoning_reward"]
        + breakdown["invalid_penalty"]
    )
    return breakdown
