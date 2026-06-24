"""Shared answer extraction independent from reward calculation."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation


class AnswerExtractor:
    """Extract final answers from supported reasoning-data and model formats."""

    _tagged_answer = re.compile(r"<answer>\s*([^<]+?)\s*</answer>", re.IGNORECASE | re.DOTALL)
    _gsm8k_answer = re.compile(r"####\s*([^\n\r]+)")

    @classmethod
    def extract_tagged(cls, text: str | None) -> str | None:
        match = cls._tagged_answer.search(text or "")
        return match.group(1).strip() if match else None

    @classmethod
    def extract_gsm8k_final_answer(cls, solution: str | None) -> str | None:
        matches = cls._gsm8k_answer.findall(solution or "")
        return matches[-1].strip() if matches else None

    @classmethod
    def normalize_number(cls, text: str | None) -> Decimal | None:
        if text is None or not str(text).strip():
            return None
        try:
            return Decimal(str(text).strip().replace(",", ""))
        except (InvalidOperation, ValueError):
            return None
