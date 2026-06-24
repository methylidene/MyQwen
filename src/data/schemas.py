"""Schema and validation primitives for reasoning datasets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ReasoningExample:
    """Normalized example used by SFT, GRPO and evaluation."""

    uid: str
    dataset_name: str
    split: str
    question: str
    reference_answer: str
    reference_solution: str
    difficulty: str | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetLoadConfig:
    """Portable dataset request with explicit split and reproducibility settings."""

    dataset_name: str
    split: str
    config_name: str | None = None
    revision: str | None = None
    source_path: str | None = None
    cache_dir: str | None = None
    max_samples: int | None = None
    shuffle: bool = False
    seed: int = 42
    purpose: str = "train"
    strict: bool = True


@dataclass(frozen=True)
class DatasetFingerprint:
    """Stable digest of normalized content and load provenance."""

    value: str
    num_examples: int
    dataset_name: str
    split: str
    revision: str | None

    @classmethod
    def from_examples(cls, config: DatasetLoadConfig, examples: list[ReasoningExample]) -> "DatasetFingerprint":
        payload = {
            "dataset_name": config.dataset_name,
            "config_name": config.config_name,
            "split": config.split,
            "revision": config.revision,
            "examples": [
                {
                    "uid": item.uid,
                    "question": item.question,
                    "reference_answer": item.reference_answer,
                    "reference_solution": item.reference_solution,
                    "difficulty": item.difficulty,
                }
                for item in examples
            ],
        }
        encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return cls(hashlib.sha256(encoded).hexdigest(), len(examples), config.dataset_name, config.split, config.revision)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetValidationReport:
    """Audit result for malformed content, duplicate UIDs and split violations."""

    total_examples: int = 0
    valid_examples: int = 0
    invalid_examples: list[dict[str, Any]] = field(default_factory=list)
    duplicate_uids: dict[str, int] = field(default_factory=dict)
    empty_questions: int = 0
    empty_answers: int = 0
    unparseable_answers: int = 0
    split_mismatches: int = 0

    @property
    def is_valid(self) -> bool:
        return not self.invalid_examples and not self.duplicate_uids

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_examples": self.total_examples,
            "valid_examples": self.valid_examples,
            "invalid_count": len(self.invalid_examples),
            "duplicate_uid_count": len(self.duplicate_uids),
            "duplicate_uids": self.duplicate_uids,
            "empty_questions": self.empty_questions,
            "empty_answers": self.empty_answers,
            "unparseable_answers": self.unparseable_answers,
            "split_mismatches": self.split_mismatches,
            "is_valid": self.is_valid,
        }
