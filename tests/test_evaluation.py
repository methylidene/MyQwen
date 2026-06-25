from __future__ import annotations

import json
from types import SimpleNamespace

from src.data import PromptFormatter, ReasoningExample
from src.evaluation.evaluator import (
    EvaluationGenerationConfig,
    Evaluator,
    load_predictions_compatible,
    metrics_by_group,
    pass_at_k,
    write_evaluation_outputs,
)


class MockGenerationBackend:
    def __init__(self):
        self.call = 0
        self.batch_sizes = []

    def generate(self, prompts, config):
        self.batch_sizes.append(len(prompts))
        outputs = [
            "<reasoning>1+1 is 2</reasoning><answer>2</answer>",
            "<reasoning>bad arithmetic</reasoning><answer>9</answer>",
        ]
        text = outputs[self.call % len(outputs)]
        self.call += 1
        return [
            SimpleNamespace(
                generated_text=text,
                token_ids=[1, 2, 3, 4],
                generated_token_ids=[3, 4],
                total_latency=0.02,
                mode="full-cache",
                prefill_latency=0.005,
                cache_rebuild_count=0,
                avg_cache_seq_len=3.0,
                max_cache_seq_len=3,
            )
            for _ in prompts
        ]


def example():
    return ReasoningExample(
        uid="eval-1", dataset_name="synthetic_arithmetic", split="test", question="1+1?",
        reference_answer="2", reference_solution="", difficulty="easy", metadata={"expr": "1 + 1"},
    )


def numbered_example(index: int):
    return ReasoningExample(
        uid=f"eval-{index}",
        dataset_name="synthetic_arithmetic",
        split="test",
        question=f"{index}+1?",
        reference_answer="2",
        reference_solution="",
        difficulty="easy",
        metadata={"expr": "1 + 1"},
    )


def test_tiny_evaluation_outputs_and_pass_at_k(tmp_path):
    evaluator = Evaluator(MockGenerationBackend(), PromptFormatter(), checkpoint_id="tiny", seed=7)
    predictions = evaluator.evaluate([example()], EvaluationGenerationConfig(max_new_tokens=8, num_generations=2, seed=7))
    required = {
        "uid", "question", "reference_answer", "raw_completion", "extracted_answer", "normalized_answer",
        "correctness", "format_validity", "answer_parse_validity", "strict_format_validity", "invalid_reason",
        "reward_breakdown", "prompt_tokens", "completion_tokens",
        "latency_seconds", "generation_config", "checkpoint_id",
    }
    assert required <= set(predictions[0])
    assert pass_at_k(predictions, 1)["value"] == 0.5
    assert pass_at_k(predictions, 2)["value"] == 1.0
    grouped = metrics_by_group(predictions)
    assert grouped["groups"]["difficulty"]["easy"]["n"] == 2
    write_evaluation_outputs(tmp_path, predictions, tmp_path, {"tiny": True})
    for name in ("README.md", "predictions.jsonl", "metrics.json", "metrics_by_group.json", "summary.csv", "error_cases.jsonl", "report.md", "figures"):
        assert (tmp_path / name).exists()
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "## English" in readme
    assert "## 中文" in readme
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "## Error Analysis" in report


def test_evaluator_respects_batch_size():
    backend = MockGenerationBackend()
    evaluator = Evaluator(backend, PromptFormatter(), checkpoint_id="tiny", seed=7)

    predictions = evaluator.evaluate(
        [numbered_example(i) for i in range(5)],
        EvaluationGenerationConfig(max_new_tokens=8, batch_size=2),
    )

    assert len(predictions) == 5
    assert backend.batch_sizes == [2, 2, 1]


def test_alignment_eval_checkpoint_output_names_are_unique():
    from scripts.run_alignment_eval import checkpoint_output_name

    paths = [
        "outputs/v02_qwen25_3b/exp01/SFT-only/20260624T163829Z-aec32e45/checkpoints",
        "outputs/v02_qwen25_3b/exp01/GRPO-G4-250/20260624T164140Z-1d37d145/checkpoints",
        "outputs/v02_qwen25_3b/exp01/GRPO-G8-250/20260624T165149Z-e6592f88/checkpoints",
    ]
    names = [checkpoint_output_name(path) for path in paths]

    assert names == [
        "SFT-only__20260624T163829Z-aec32e45",
        "GRPO-G4-250__20260624T164140Z-1d37d145",
        "GRPO-G8-250__20260624T165149Z-e6592f88",
    ]


def test_error_taxonomy_is_structural_not_keyword_claims():
    evaluator = Evaluator(MockGenerationBackend(), PromptFormatter())
    sample = example()
    verifier = evaluator.verifier
    assert verifier.verify("no tag", "2").invalid_reason == "no_final_answer"
    assert verifier.verify("<answer>two</answer>", "2").parser_error
    assert verifier.verify("<answer>2</answer><answer>3</answer>", "2").multiple_conflicting_answers
    currency = verifier.verify("<answer>$2</answer>", "2")
    assert currency.correctness
    assert currency.format_valid
    assert not currency.strict_format_valid


def test_legacy_prediction_reader(tmp_path):
    path = tmp_path / "legacy.jsonl"
    path.write_text(json.dumps({"id": "old-1", "difficulty": "easy", "prompt": "1+1", "gold_answer": "2", "response": "<answer>2</answer>", "reward": {"accuracy": 1.0, "format_pass": 1.0, "total_reward": 1.2}}) + "\n", encoding="utf-8")
    rows = load_predictions_compatible(path)
    assert rows[0]["uid"] == "old-1"
    assert rows[0]["correctness"]
    assert rows[0]["completion_tokens"] > 0
