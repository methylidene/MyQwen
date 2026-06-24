"""Dataset adapters, registry and validation for normalized reasoning data."""

from __future__ import annotations

import hashlib
import json
import random
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import Any, ClassVar

from src.utils.io import read_jsonl, write_jsonl

from .answers import AnswerExtractor
from .schemas import DatasetFingerprint, DatasetLoadConfig, DatasetValidationReport, ReasoningExample


def stable_uid(dataset_name: str, split: str, source_id: str, question: str) -> str:
    """Produce a stable UID that does not depend on record ordering."""
    payload = "\x1f".join((dataset_name, split, str(source_id), question.strip()))
    return f"{dataset_name}-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]}"


class DatasetAdapter(ABC):
    """Normalize one external dataset into :class:`ReasoningExample` records."""

    dataset_name: str

    @abstractmethod
    def load(self, config: DatasetLoadConfig) -> list[ReasoningExample]:
        """Load and normalize a single explicit split."""


class SyntheticArithmeticAdapter(DatasetAdapter):
    """Read legacy synthetic arithmetic JSONL without changing its generation CLI."""

    dataset_name = "synthetic_arithmetic"

    def load(self, config: DatasetLoadConfig) -> list[ReasoningExample]:
        if not config.source_path:
            raise ValueError("SyntheticArithmeticAdapter requires source_path (for example data/synthetic_math/train.jsonl).")
        rows = read_jsonl(config.source_path)
        examples: list[ReasoningExample] = []
        for index, row in enumerate(rows):
            source_id = str(row.get("uid") or row.get("id") or index)
            question = str(row.get("question") if row.get("question") is not None else row.get("prompt", "")).strip()
            answer = str(row.get("reference_answer") if row.get("reference_answer") is not None else row.get("answer", "")).strip()
            solution = str(row.get("reference_solution") if row.get("reference_solution") is not None else "").strip()
            if not solution and answer:
                solution = f"<reasoning>We compute the arithmetic carefully.</reasoning>\n<answer>{answer}</answer>"
            source_split_value = row.get("split")
            if source_split_value is None:
                source_split_value = source_id.split("-", 1)[0] if "-" in source_id else config.split
            source_split = str(source_split_value)
            metadata = dict(row.get("metadata") or {})
            metadata.update({"source_id": source_id, "source_split": source_split, "source_path": config.source_path})
            examples.append(
                ReasoningExample(
                    uid=stable_uid(self.dataset_name, config.split, source_id, question),
                    dataset_name=self.dataset_name,
                    split=config.split,
                    question=question,
                    reference_answer=answer,
                    reference_solution=solution,
                    difficulty=row.get("difficulty"),
                    metadata=metadata,
                )
            )
        return examples


class GSM8KAdapter(DatasetAdapter):
    """Normalize ``openai/gsm8k`` main while preserving its original solution."""

    dataset_name = "gsm8k"

    def load(self, config: DatasetLoadConfig) -> list[ReasoningExample]:
        if config.split not in {"train", "test"}:
            raise ValueError("GSM8K exposes only 'train' and 'test' splits.")
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError("GSM8K requires the optional 'datasets' package. Install it with: pip install datasets") from exc
        dataset = load_dataset(
            "openai/gsm8k",
            config.config_name or "main",
            split=config.split,
            revision=config.revision,
            cache_dir=config.cache_dir,
        )
        examples: list[ReasoningExample] = []
        for index, row in enumerate(dataset):
            question = str(row.get("question", "")).strip()
            solution = str(row.get("answer", "")).strip()
            final_answer = AnswerExtractor.extract_gsm8k_final_answer(solution)
            metadata = {
                "source_id": str(index),
                "dataset_config": config.config_name or "main",
                "revision": config.revision,
                "raw_question": row.get("question"),
                "raw_answer": row.get("answer"),
            }
            examples.append(
                ReasoningExample(
                    uid=stable_uid(self.dataset_name, config.split, str(index), question),
                    dataset_name=self.dataset_name,
                    split=config.split,
                    question=question,
                    reference_answer=final_answer or "",
                    reference_solution=solution,
                    difficulty=None,
                    metadata=metadata,
                )
            )
        return examples


class DatasetRegistry:
    """Resolve adapters and apply loading, split checks, caching and validation."""

    _adapters: ClassVar[dict[str, DatasetAdapter]] = {}

    @classmethod
    def register(cls, adapter: DatasetAdapter) -> None:
        if not adapter.dataset_name:
            raise ValueError("Dataset adapter must declare a dataset_name.")
        cls._adapters[adapter.dataset_name] = adapter

    @classmethod
    def get(cls, name: str) -> DatasetAdapter:
        try:
            return cls._adapters[name]
        except KeyError as exc:
            raise KeyError(f"Unknown dataset '{name}'. Available datasets: {', '.join(sorted(cls._adapters))}.") from exc

    @classmethod
    def load(cls, config: DatasetLoadConfig) -> tuple[list[ReasoningExample], DatasetFingerprint, DatasetValidationReport]:
        cls._validate_purpose(config)
        examples = cls._load_cached_or_adapter(config)
        if config.shuffle:
            examples = list(examples)
            random.Random(config.seed).shuffle(examples)
        if config.max_samples is not None:
            if config.max_samples < 0:
                raise ValueError("max_samples must be non-negative.")
            examples = examples[: config.max_samples]
        report = validate_examples(examples, expected_split=config.split)
        if config.strict and not report.is_valid:
            raise ValueError(f"Dataset validation failed: {json.dumps(report.to_dict(), sort_keys=True)}")
        return examples, DatasetFingerprint.from_examples(config, examples), report

    @classmethod
    def _validate_purpose(cls, config: DatasetLoadConfig) -> None:
        if config.purpose == "eval" and config.split == "train":
            raise ValueError("Evaluation cannot use the train split. Choose test or validation explicitly.")
        if config.purpose == "train" and config.split == "test":
            raise ValueError("Training cannot use the test split. Choose train or validation explicitly.")

    @classmethod
    def _cache_path(cls, config: DatasetLoadConfig) -> Path | None:
        if not config.cache_dir:
            return None
        source = Path(config.source_path).resolve().as_posix() if config.source_path else "hf"
        token = hashlib.sha256(f"{config.dataset_name}|{config.config_name}|{config.split}|{config.revision}|{source}".encode()).hexdigest()[:16]
        return Path(config.cache_dir) / "normalized" / f"{config.dataset_name}-{config.split}-{token}.jsonl"

    @classmethod
    def _load_cached_or_adapter(cls, config: DatasetLoadConfig) -> list[ReasoningExample]:
        cache_path = cls._cache_path(config)
        if cache_path and cache_path.exists():
            return [ReasoningExample(**row) for row in read_jsonl(cache_path)]
        examples = cls.get(config.dataset_name).load(config)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            write_jsonl([item.to_dict() for item in examples], cache_path)
        return examples


def validate_examples(examples: list[ReasoningExample], *, expected_split: str) -> DatasetValidationReport:
    """Validate normalized records without inventing missing answers."""
    report = DatasetValidationReport(total_examples=len(examples))
    counts = Counter(item.uid for item in examples)
    report.duplicate_uids = {uid: count for uid, count in counts.items() if count > 1}
    for item in examples:
        reasons: list[str] = []
        if not item.question.strip():
            report.empty_questions += 1
            reasons.append("empty_question")
        if not item.reference_answer.strip():
            report.empty_answers += 1
            reasons.append("empty_answer")
        elif AnswerExtractor.normalize_number(item.reference_answer) is None:
            report.unparseable_answers += 1
            reasons.append("unparseable_answer")
        if item.split != expected_split or item.metadata.get("source_split", expected_split) != expected_split:
            report.split_mismatches += 1
            reasons.append("split_mismatch")
        if counts[item.uid] > 1:
            reasons.append("duplicate_uid")
        if reasons:
            report.invalid_examples.append({"example": item.to_dict(), "reasons": reasons})
        else:
            report.valid_examples += 1
    return report


DatasetRegistry.register(SyntheticArithmeticAdapter())
DatasetRegistry.register(GSM8KAdapter())
