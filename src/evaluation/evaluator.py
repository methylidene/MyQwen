"""Training-decoupled reasoning evaluator and structured prediction metrics."""
from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import torch

from src.alignment.rewards import rule_based_reward
from src.data import AnswerExtractor, PromptFormatter, ReasoningExample
from src.utils.io import save_json, write_bilingual_readme, write_csv, write_jsonl


@dataclass(frozen=True)
class EvaluationGenerationConfig:
    max_new_tokens: int = 128
    num_generations: int = 1
    batch_size: int = 1
    use_cache: bool = True
    cache_window: int | None = None
    seed: int | None = None


@dataclass(frozen=True)
class VerificationResult:
    extracted_answer: str | None
    normalized_answer: str | None
    correctness: bool
    format_valid: bool
    strict_format_valid: bool
    invalid_reason: str | None
    parser_error: bool
    multiple_conflicting_answers: bool


class AnswerVerifier:
    """Numeric final-answer verification with explicit parser outcomes."""

    def verify(self, completion: str, reference_answer: str) -> VerificationResult:
        import re
        tags = re.findall(r"<answer>\s*([^<]+?)\s*</answer>", completion or "", flags=re.IGNORECASE | re.DOTALL)
        values = [item.strip() for item in tags if item.strip()]
        extracted = values[-1] if values else None
        conflicting = len(set(values)) > 1
        normalized = AnswerExtractor.normalize_number(extracted)
        reference = AnswerExtractor.normalize_number(reference_answer)
        strict_format = AnswerExtractor.is_bare_number(extracted)
        if not values:
            return VerificationResult(None, None, False, False, False, "no_final_answer", False, False)
        if conflicting:
            return VerificationResult(extracted, str(normalized) if normalized is not None else None, False, False, strict_format, "multiple_conflicting_answers", normalized is None, True)
        if normalized is None:
            return VerificationResult(extracted, None, False, False, False, "parser_error", True, False)
        if reference is None:
            return VerificationResult(extracted, str(normalized), False, True, strict_format, "reference_parser_error", True, False)
        return VerificationResult(extracted, str(normalized), normalized == reference, True, strict_format, None, False, False)


class GenerationBackend(Protocol):
    """Evaluator-side generation protocol returning one result per prompt."""

    def generate(self, prompts: list[str], config: EvaluationGenerationConfig) -> list[Any]:
        ...


class KVGenerationBackend:
    """Adapter for the explicit KV-cache generator without trainer coupling."""

    def __init__(self, generator: Any) -> None:
        self.generator = generator

    def generate(self, prompts: list[str], config: EvaluationGenerationConfig) -> list[Any]:
        return self.generator.generate(
            prompts,
            config.max_new_tokens,
            config.use_cache,
            config.cache_window,
            batch_size=config.batch_size,
        )


@dataclass
class Evaluator:
    generation_backend: GenerationBackend
    formatter: PromptFormatter
    verifier: AnswerVerifier = field(default_factory=AnswerVerifier)
    checkpoint_id: str = "unknown"
    seed: int | None = None

    def evaluate(self, examples: list[ReasoningExample], config: EvaluationGenerationConfig) -> list[dict[str, Any]]:
        if config.num_generations < 1:
            raise ValueError("num_generations must be at least one.")
        if config.batch_size < 1:
            raise ValueError("batch_size must be at least one.")
        prompts = [self.formatter.grpo_prompt(item) for item in examples]
        predictions: list[dict[str, Any]] = []
        for generation_index in range(config.num_generations):
            for start in range(0, len(examples), config.batch_size):
                batch_examples = examples[start : start + config.batch_size]
                batch_prompts = prompts[start : start + config.batch_size]
                results = self.generation_backend.generate(batch_prompts, config)
                if len(results) != len(batch_examples):
                    raise ValueError("GenerationBackend returned a result count different from prompt count.")
                for example, prompt, result in zip(batch_examples, batch_prompts, results):
                    predictions.append(self._prediction(example, prompt, result, config, generation_index))
        return predictions

    def _prediction(self, example: ReasoningExample, prompt: str, result: Any, config: EvaluationGenerationConfig, generation_index: int) -> dict[str, Any]:
        completion = str(result.generated_text)
        verification = self.verifier.verify(completion, example.reference_answer)
        reward = rule_based_reward(completion, example.reference_answer)
        prompt_tokens = max(0, len(getattr(result, "token_ids", [])) - len(getattr(result, "generated_token_ids", [])))
        completion_tokens = len(getattr(result, "generated_token_ids", []))
        truncated = completion_tokens >= config.max_new_tokens and not completion.rstrip().endswith("</answer>")
        category = classify_error(example, completion, verification, truncated, completion_tokens)
        return {
            "uid": example.uid,
            "dataset": example.dataset_name,
            "split": example.split,
            "difficulty": example.difficulty or "unknown",
            "problem_type": str(example.metadata.get("problem_type") or ("arithmetic" if "expr" in example.metadata else "unknown")),
            "answer_type": "numeric",
            "question": example.question,
            "reference_answer": example.reference_answer,
            "raw_completion": completion,
            "extracted_answer": verification.extracted_answer,
            "normalized_answer": verification.normalized_answer,
            "correctness": verification.correctness,
            "format_validity": verification.format_valid,
            "answer_parse_validity": verification.format_valid,
            "strict_format_validity": verification.strict_format_valid,
            "invalid_reason": verification.invalid_reason,
            "parser_error": verification.parser_error,
            "reward_breakdown": reward,
            "prompt": prompt,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "latency_seconds": float(getattr(result, "total_latency", 0.0)),
            "tokens_per_second": completion_tokens / max(float(getattr(result, "total_latency", 0.0)), 1e-9),
            "truncated": truncated,
            "generation_config": asdict(config),
            "generation_index": generation_index,
            "checkpoint_id": self.checkpoint_id,
            "seed": self.seed,
            "cache_metrics": {
                "mode": getattr(result, "mode", "unknown"),
                "prefill_latency": float(getattr(result, "prefill_latency", 0.0)),
                "cache_rebuild_count": int(getattr(result, "cache_rebuild_count", 0)),
                "avg_cache_seq_len": float(getattr(result, "avg_cache_seq_len", 0.0)),
                "max_cache_seq_len": int(getattr(result, "max_cache_seq_len", 0)),
            },
            "error_category": category,
        }


def classify_error(example: ReasoningExample, completion: str, verification: VerificationResult, truncated: bool, completion_tokens: int) -> str:
    """Conservative taxonomy; no textual heuristic claims exact reasoning causes."""
    if verification.multiple_conflicting_answers:
        return "multiple_conflicting_answers"
    if truncated:
        return "truncation"
    if verification.invalid_reason == "no_final_answer":
        return "no_final_answer"
    if verification.parser_error:
        return "parser_error"
    if not verification.format_valid:
        return "format_failure"
    if verification.correctness:
        if completion_tokens <= 8 and "<reasoning>" in completion:
            return "correct_short_reasoning"
        if completion_tokens > 32 and "<reasoning>" in completion:
            return "correct_long_reasoning"
        return "correct"
    if "expr" in example.metadata:
        return "arithmetic_error"
    # The text alone cannot reliably identify a reasoning failure.
    return "unknown"


def pass_at_k(rows: list[dict[str, Any]], k: int) -> dict[str, Any]:
    """Unbiased pass@k estimate from actual independent generated samples only."""
    if k < 1:
        raise ValueError("k must be at least one.")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["uid"]].append(row)
    estimates = []
    for samples in grouped.values():
        n = len(samples)
        if n < k:
            continue
        correct = sum(bool(sample["correctness"]) for sample in samples)
        value = 1.0 - math.comb(n - correct, k) / math.comb(n, k) if n - correct >= k else 1.0
        estimates.append(value)
    return {"value": sum(estimates) / len(estimates) if estimates else None, "n_prompts": len(estimates), "k": k}


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def aggregate_rows(rows: list[dict[str, Any]], rollout_tokens: int | None = None) -> dict[str, Any]:
    n = len(rows)
    rewards = [row["reward_breakdown"] for row in rows]
    component_names = sorted({key for reward in rewards for key, value in reward.items() if isinstance(value, (int, float))})
    completion = [float(row["completion_tokens"]) for row in rows]
    latencies = [float(row["latency_seconds"]) for row in rows]
    metric: dict[str, Any] = {
        "n": n,
        "accuracy": sum(bool(row["correctness"]) for row in rows) / n if n else 0.0,
        "pass_at_1": pass_at_k(rows, 1),
        "format_pass_rate": sum(bool(row["format_validity"]) for row in rows) / n if n else 0.0,
        "strict_format_pass_rate": sum(bool(row.get("strict_format_validity", row["format_validity"])) for row in rows) / n if n else 0.0,
        "invalid_rate": sum(not bool(row["format_validity"]) for row in rows) / n if n else 0.0,
        "parser_error_rate": sum(bool(row["parser_error"]) for row in rows) / n if n else 0.0,
        "average_reward": sum(float(reward.get("total_reward", 0.0)) for reward in rewards) / n if n else 0.0,
        "average_completion_tokens": sum(completion) / n if n else 0.0,
        "median_completion_tokens": _median(completion),
        "truncated_completion_rate": sum(bool(row["truncated"]) for row in rows) / n if n else 0.0,
        "average_latency_seconds": sum(latencies) / n if n else 0.0,
        "tokens_per_second": sum(float(row["completion_tokens"]) for row in rows) / sum(latencies) if sum(latencies) > 0 else None,
        "peak_vram_mb": torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0,
        "cache_rebuild_count": sum(row["cache_metrics"]["cache_rebuild_count"] for row in rows),
        "average_cache_seq_len": sum(row["cache_metrics"]["avg_cache_seq_len"] for row in rows) / n if n else 0.0,
    }
    metric["reward_components"] = {name: sum(float(reward.get(name, 0.0)) for reward in rewards) / n if n else 0.0 for name in component_names}
    for k in sorted({len(samples) for samples in _group_by_uid(rows).values()}):
        if k > 1:
            metric[f"pass_at_{k}"] = pass_at_k(rows, k)
    if rollout_tokens is not None and rollout_tokens > 0:
        metric["correctness_per_million_rollout_tokens"] = sum(bool(row["correctness"]) for row in rows) * 1_000_000 / rollout_tokens
        metric["rollout_tokens"] = rollout_tokens
    else:
        metric["correctness_per_million_rollout_tokens"] = None
        metric["rollout_tokens"] = None
    return metric


def _group_by_uid(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["uid"]].append(row)
    return grouped


def metrics_by_group(rows: list[dict[str, Any]], rollout_tokens: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"overall": aggregate_rows(rows, rollout_tokens), "groups": {}}
    for field in ("dataset", "split", "difficulty", "problem_type", "answer_type", "checkpoint_id", "seed"):
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            buckets[str(row.get(field, "unknown"))].append(row)
        result["groups"][field] = {name: aggregate_rows(items, rollout_tokens) for name, items in buckets.items()}
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        length = int(row["completion_tokens"])
        label = "0-8" if length <= 8 else "9-32" if length <= 32 else "33+"
        buckets[label].append(row)
    result["groups"]["completion_length_bucket"] = {name: aggregate_rows(items, rollout_tokens) for name, items in buckets.items()}
    return result


def training_rollout_tokens(checkpoint: str | Path) -> int | None:
    path = Path(checkpoint)
    for candidate in (path, *path.parents):
        summary = candidate / "summary.json"
        if summary.exists():
            try:
                value = json.loads(summary.read_text(encoding="utf-8")).get("generated_completion_tokens")
                return int(value) if value is not None else None
            except (OSError, ValueError, TypeError):
                return None
    return None


def write_evaluation_outputs(output_dir: str | Path, predictions: list[dict[str, Any]], checkpoint: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    write_bilingual_readme(
        destination,
        title="Evaluation Outputs",
        english="Evaluation artifact directory containing predictions, metrics, grouped summaries, error cases, figures and an English report.md.",
        chinese="这是评测产物目录，包含预测、指标、分组汇总、错误样例、图表，以及英文 report.md。",
        preserve_existing=False,
    )
    (destination / "figures").mkdir(exist_ok=True)
    rollout_tokens = training_rollout_tokens(checkpoint)
    grouped = metrics_by_group(predictions, rollout_tokens)
    write_jsonl(predictions, destination / "predictions.jsonl")
    write_jsonl([row for row in predictions if row["error_category"] not in {"correct", "correct_short_reasoning", "correct_long_reasoning"}], destination / "error_cases.jsonl")
    save_json(grouped["overall"], destination / "metrics.json")
    save_json(grouped, destination / "metrics_by_group.json")
    csv_rows = []
    for dimension, values in grouped["groups"].items():
        for group, metric in values.items():
            csv_rows.append({"group_dimension": dimension, "group": group, **_flat_metric(metric)})
    write_csv(csv_rows, destination / "summary.csv")
    _write_report(destination / "report.md", config, grouped, predictions, rollout_tokens)
    return grouped


def _flat_metric(metric: dict[str, Any]) -> dict[str, Any]:
    result = {key: value for key, value in metric.items() if not isinstance(value, (dict, list))}
    result.update({f"reward_{key}": value for key, value in metric.get("reward_components", {}).items()})
    pass_one = metric.get("pass_at_1", {})
    result["pass_at_1"] = pass_one.get("value") if isinstance(pass_one, dict) else pass_one
    return result


def _write_report(path: Path, config: dict[str, Any], grouped: dict[str, Any], predictions: list[dict[str, Any]], rollout_tokens: int | None) -> None:
    overall = grouped["overall"]
    errors: dict[str, int] = defaultdict(int)
    for row in predictions:
        errors[row["error_category"]] += 1
    lines = [
        "# Evaluation Report", "", "## Experiment Summary", f"Samples: {overall['n']}", "",
        "## Configuration", "```json", json.dumps(config, indent=2, ensure_ascii=False), "```", "",
        "## Main Results", f"Accuracy: {overall['accuracy']:.4f}", f"Pass@1: {overall['pass_at_1']['value']}",
        f"Format pass rate: {overall['format_pass_rate']:.4f}", f"Strict format pass rate: {overall['strict_format_pass_rate']:.4f}",
        f"Invalid rate: {overall['invalid_rate']:.4f}", "",
        "## Efficiency", f"Completion tokens (mean/median): {overall['average_completion_tokens']:.2f}/{overall['median_completion_tokens']:.2f}",
        f"Latency (mean): {overall['average_latency_seconds']:.4f}s", f"Tokens/s: {overall['tokens_per_second']:.2f}",
        f"Peak VRAM: {overall['peak_vram_mb']:.2f} MiB", f"Training rollout tokens: {rollout_tokens if rollout_tokens is not None else 'unavailable'}", "",
        "## Reward Diagnostics", "```json", json.dumps(overall['reward_components'], indent=2), "```", "",
        "## Error Analysis", "```json", json.dumps(dict(errors), indent=2), "```", "",
        "## Reproducibility Information", "Prediction rows include checkpoint id, seed and resolved generation settings.", "",
        "## Limitations", "Error labels are structural. Except for explicit format/parser/truncation/conflicting-answer cases and synthetic expression errors, reasoning causes are reported as unknown rather than inferred from keywords.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_predictions_compatible(path: str | Path) -> list[dict[str, Any]]:
    """Read canonical predictions or normalize the repository's legacy rows."""
    from src.utils.io import read_jsonl
    rows = read_jsonl(path)
    converted = []
    verifier = AnswerVerifier()
    for row in rows:
        if "uid" in row:
            converted.append(row)
            continue
        completion = row.get("raw_completion", row.get("response", ""))
        reference = row.get("reference_answer", row.get("gold_answer", ""))
        verified = verifier.verify(completion, reference)
        reward = row.get("reward", rule_based_reward(completion, reference))
        converted.append({
            "uid": row.get("id", "legacy-unknown"), "dataset": row.get("dataset", "unknown"), "split": row.get("split", "unknown"),
            "difficulty": row.get("difficulty", "unknown"), "problem_type": "unknown", "answer_type": "numeric", "question": row.get("prompt", ""),
            "reference_answer": reference, "raw_completion": completion, "extracted_answer": verified.extracted_answer,
            "normalized_answer": verified.normalized_answer, "correctness": bool(reward.get("accuracy", verified.correctness)),
            "format_validity": bool(reward.get("format_pass", verified.format_valid)), "invalid_reason": verified.invalid_reason,
            "answer_parse_validity": verified.format_valid, "strict_format_validity": verified.strict_format_valid,
            "parser_error": verified.parser_error, "reward_breakdown": reward, "prompt": row.get("prompt", ""), "prompt_tokens": 0,
            "completion_tokens": len(str(completion).split()), "latency_seconds": 0.0, "tokens_per_second": 0.0, "truncated": False,
            "generation_config": {}, "generation_index": 0, "checkpoint_id": row.get("checkpoint_id", "legacy"), "seed": row.get("seed"),
            "cache_metrics": {"mode": "unknown", "prefill_latency": 0.0, "cache_rebuild_count": 0, "avg_cache_seq_len": 0.0, "max_cache_seq_len": 0},
            "error_category": "unknown" if not verified.correctness else "correct",
        })
    return converted
