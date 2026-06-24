from __future__ import annotations

import json
import sys
import types

import pytest

from src.data import (
    AnswerExtractor,
    DatasetLoadConfig,
    DatasetRegistry,
    GSM8KAdapter,
    PromptFormatter,
    ReasoningExample,
    SyntheticArithmeticAdapter,
    stable_uid,
)


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def synthetic_rows():
    return [
        {
            "id": f"train-{index:06d}",
            "difficulty": "easy",
            "prompt": f"Compute {index} + 1.",
            "answer": str(index + 1),
            "metadata": {"expr": f"{index} + 1"},
        }
        for index in range(8)
    ]


def load_synthetic(path, **overrides):
    values = {
        "dataset_name": "synthetic_arithmetic",
        "split": "train",
        "source_path": str(path),
        "purpose": "train",
    }
    values.update(overrides)
    return DatasetRegistry.load(DatasetLoadConfig(**values))


def fake_gsm8k(monkeypatch):
    calls = []

    def load_dataset(name, config_name, *, split, revision, cache_dir):
        calls.append((name, config_name, split, revision, cache_dir))
        return [{"question": "How many?", "answer": "Reasoning steps.\n#### 1,234"}]

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=load_dataset))
    return calls


def test_gsm8k_final_answer_extraction_and_schema(monkeypatch):
    calls = fake_gsm8k(monkeypatch)
    examples, _, report = DatasetRegistry.load(
        DatasetLoadConfig(dataset_name="gsm8k", split="train", config_name="main", purpose="train")
    )
    assert AnswerExtractor.extract_gsm8k_final_answer("x\n#### 1,234") == "1,234"
    assert examples[0].reference_answer == "1,234"
    assert examples[0].reference_solution.endswith("#### 1,234")
    assert examples[0].metadata["raw_answer"].endswith("#### 1,234")
    assert calls == [("openai/gsm8k", "main", "train", None, None)]
    assert report.is_valid
    assert set(examples[0].to_dict()) == {
        "uid", "dataset_name", "split", "question", "reference_answer", "reference_solution", "difficulty", "metadata"
    }


def test_synthetic_and_gsm8k_share_schema(tmp_path, monkeypatch):
    source = tmp_path / "train.jsonl"
    write_jsonl(source, synthetic_rows()[:1])
    synthetic, _, _ = load_synthetic(source)
    fake_gsm8k(monkeypatch)
    gsm8k, _, _ = DatasetRegistry.load(DatasetLoadConfig(dataset_name="gsm8k", split="test", purpose="eval"))
    assert set(synthetic[0].to_dict()) == set(gsm8k[0].to_dict())


def test_split_leakage_is_rejected(tmp_path):
    source = tmp_path / "train.jsonl"
    write_jsonl(source, synthetic_rows())
    with pytest.raises(ValueError, match="Evaluation cannot use the train split"):
        DatasetRegistry.load(DatasetLoadConfig("synthetic_arithmetic", "train", source_path=str(source), purpose="eval"))
    with pytest.raises(ValueError, match="Training cannot use the test split"):
        DatasetRegistry.load(DatasetLoadConfig("synthetic_arithmetic", "test", source_path=str(source), purpose="train"))


def test_source_split_mismatch_is_rejected_even_when_purpose_is_train(tmp_path):
    source = tmp_path / "test.jsonl"
    rows = synthetic_rows()[:1]
    rows[0]["id"] = "test-000000"
    write_jsonl(source, rows)
    with pytest.raises(ValueError, match="split_mismatches"):
        load_synthetic(source, strict=True)


def test_deterministic_shuffle_and_stable_uid(tmp_path):
    source = tmp_path / "train.jsonl"
    write_jsonl(source, synthetic_rows())
    first, first_fp, _ = load_synthetic(source, shuffle=True, seed=19)
    second, second_fp, _ = load_synthetic(source, shuffle=True, seed=19)
    third, _, _ = load_synthetic(source, shuffle=True, seed=20)
    assert [row.uid for row in first] == [row.uid for row in second]
    assert [row.uid for row in first] != [row.uid for row in third]
    assert first_fp.value == second_fp.value
    assert stable_uid("gsm8k", "train", "4", "Question?") == stable_uid("gsm8k", "train", "4", "Question?")


def test_grpo_prompt_never_contains_reference_answer():
    example = ReasoningExample(
        uid="example",
        dataset_name="synthetic_arithmetic",
        split="train",
        question="What is 4 plus 5?",
        reference_answer="9",
        reference_solution="4 + 5 = 9",
        difficulty="easy",
    )
    formatter = PromptFormatter(system_prompt="Solve carefully.")
    prompt = formatter.grpo_prompt(example, chat=True)
    assert "reference_solution" not in prompt
    assert "4 + 5 = 9" not in prompt
    assert "<answer>9</answer>" not in prompt


def test_malformed_and_duplicate_samples_are_audited(tmp_path):
    source = tmp_path / "train.jsonl"
    write_jsonl(
        source,
        [
            {"id": "train-000001", "prompt": "", "answer": ""},
            {"id": "train-000002", "prompt": "Compute.", "answer": "not-a-number"},
            {"id": "train-000002", "prompt": "Compute.", "answer": "not-a-number"},
        ],
    )
    examples, _, report = load_synthetic(source, strict=False)
    assert len(examples) == 3
    assert not report.is_valid
    assert report.empty_questions == 1
    assert report.empty_answers == 1
    assert report.unparseable_answers == 2
    assert report.duplicate_uids
    with pytest.raises(ValueError, match="Dataset validation failed"):
        load_synthetic(source, strict=True)


def test_gsm8k_canonical_jsonl_source_is_supported(tmp_path):
    source = tmp_path / "gsm8k_test.jsonl"
    source.write_text(json.dumps({
        "uid": "gsm8k-local-1", "dataset_name": "gsm8k", "split": "test", "question": "1+1?",
        "reference_answer": "2", "reference_solution": "#### 2", "difficulty": None, "metadata": {"raw_answer": "#### 2"},
    }) + "\n", encoding="utf-8")
    examples, _, report = DatasetRegistry.load(DatasetLoadConfig(
        dataset_name="gsm8k", split="test", source_path=str(source), purpose="eval",
    ))
    assert examples[0].uid == "gsm8k-local-1"
    assert report.is_valid


def test_adapter_classes_are_registered():
    assert isinstance(DatasetRegistry.get("synthetic_arithmetic"), SyntheticArithmeticAdapter)
    assert isinstance(DatasetRegistry.get("gsm8k"), GSM8KAdapter)
