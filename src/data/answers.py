"""Shared answer extraction independent from reward calculation."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation


class AnswerExtractor:
    """Extract final answers from supported reasoning-data and model formats."""

    _tagged_answer = re.compile(r"<answer>\s*([^<]+?)\s*</answer>", re.IGNORECASE | re.DOTALL)
    _gsm8k_answer = re.compile(r"####\s*([^\n\r]+)")
    _bare_number = re.compile(r"^[+-]?(?:\d+(?:,\d{3})*|\d+)(?:\.\d+)?%?$")
    _number_token = re.compile(r"[+-]?(?:\d+(?:,\d{3})*|\d+)(?:\.\d+)?%?")

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
        raw = str(text).strip()
        direct = raw.removeprefix("$").strip()
        parsed = cls._decimal_from_token(direct)
        if parsed is not None:
            return parsed
        tokens = cls._number_token.findall(raw)
        if len(tokens) != 1:
            return None
        return cls._decimal_from_token(tokens[0])

    @classmethod
    def is_bare_number(cls, text: str | None) -> bool:
        if text is None:
            return False
        return bool(cls._bare_number.fullmatch(str(text).strip()))

    @classmethod
    def _decimal_from_token(cls, token: str) -> Decimal | None:
        cleaned = token.strip().replace(",", "")
        if cleaned.endswith("%"):
            cleaned = cleaned[:-1]
        try:
            return Decimal(cleaned)
        except (InvalidOperation, ValueError):
            return None
