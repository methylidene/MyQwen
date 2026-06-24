"""Validated, serializable experiment configuration loaded from one YAML file."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    model_name_or_path: str
    revision: str | None = None
    backend_name: str = "huggingface"
    custom_factory_name: str | None = None
    device: str = "cuda"
    dtype: str = "bf16"
    trust_remote_code: bool = True
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: list[str] | None = None


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    split: str
    config_name: str | None = None
    revision: str | None = None
    source_path: str | None = None
    cache_dir: str | None = None
    max_samples: int | None = None
    shuffle: bool = False


@dataclass(frozen=True)
class GenerationConfig:
    seed: int | None
    temperature: float = 0.8
    top_p: float = 0.95
    max_new_tokens: int = 128
    max_prompt_length: int = 384
    system_prompt: str | None = None
    final_answer_format: str = "<answer>{answer}</answer>"
    use_chat_template: bool = False


@dataclass(frozen=True)
class SFTConfig:
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    num_train_epochs: float = 1.0
    max_steps: int = -1
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_length: int = 512
    gradient_checkpointing: bool = False
    checkpoint_interval: int = 0
    checkpoint_keep: int = 2


@dataclass(frozen=True)
class GRPOConfig:
    max_steps: int = 1000
    max_generated_completion_tokens: int | None = None
    group_size: int = 4
    learning_rate: float = 5e-6
    weight_decay: float = 0.0
    beta_kl: float = 0.02
    clip_eps: float = 0.2
    entropy_coef: float = 0.0
    advantage_epsilon: float = 1e-6
    forward_micro_batch_size: int = 4
    use_reference_policy: bool = True
    gradient_checkpointing: bool = False
    checkpoint_interval: int = 0
    checkpoint_keep: int = 2


@dataclass(frozen=True)
class EvaluationConfig:
    checkpoint_dirs: list[str] = field(default_factory=list)
    max_new_tokens: int = 128


@dataclass(frozen=True)
class LoggingConfig:
    output_root: str = "outputs/experiments"
    experiment_dir: str | None = None
    stage_name: str | None = None
    run_id: str | None = None
    resume_from: str | None = None
    allow_existing: bool = False


@dataclass(frozen=True)
class BudgetConfig:
    max_steps: int | None = None
    max_wall_time_seconds: float | None = None
    record_peak_vram: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    task: str
    smoke: bool
    seed: int | None
    data_seed: int | None
    model: ModelConfig
    dataset: DatasetConfig
    generation: GenerationConfig
    sft: SFTConfig = field(default_factory=SFTConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)

    def validate(self) -> None:
        if self.task not in {"sft", "grpo", "evaluation", "zero_shot"}:
            raise ValueError("task must be one of: sft, grpo, evaluation, zero_shot.")
        if not self.smoke and any(value is None for value in (self.seed, self.data_seed, self.generation.seed)):
            raise ValueError("Formal experiments require seed, data_seed and generation.seed.")
        if self.task in {"evaluation", "zero_shot"} and self.dataset.split == "train":
            raise ValueError("Evaluation/zero-shot experiments cannot use the train split.")
        if self.task == "grpo" and self.grpo.group_size < 2:
            raise ValueError("GRPO group_size must be at least 2.")
        if self.generation.temperature <= 0:
            raise ValueError("generation.temperature must be positive.")
        max_generated_completion_tokens = self.grpo.max_generated_completion_tokens
        if max_generated_completion_tokens is not None and (
            isinstance(max_generated_completion_tokens, bool)
            or not isinstance(max_generated_completion_tokens, int)
            or max_generated_completion_tokens <= 0
        ):
            raise ValueError("GRPO max_generated_completion_tokens must be positive when provided.")
        if self.grpo.beta_kl < 0 or self.grpo.clip_eps < 0:
            raise ValueError("GRPO beta_kl and clip_eps must be non-negative.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def config_hash(self) -> str:
        payload = json.dumps(_semantic_dict(self.to_dict()), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _strict_dataclass(cls, values: dict[str, Any]):
    allowed = {item.name for item in fields(cls)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {', '.join(sorted(unknown))}")
    return cls(**values)


def experiment_from_dict(values: dict[str, Any]) -> ExperimentConfig:
    values = dict(values)
    nested = {
        "model": ModelConfig,
        "dataset": DatasetConfig,
        "generation": GenerationConfig,
        "sft": SFTConfig,
        "grpo": GRPOConfig,
        "evaluation": EvaluationConfig,
        "logging": LoggingConfig,
        "budget": BudgetConfig,
    }
    for name, cls in nested.items():
        values[name] = _strict_dataclass(cls, values.get(name, {}))
    config = _strict_dataclass(ExperimentConfig, values)
    config.validate()
    return config


def apply_overrides(values: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    result = json.loads(json.dumps(values))
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must use dotted.path=value, got '{item}'.")
        path, raw_value = item.split("=", 1)
        value = yaml.safe_load(raw_value)
        target = result
        keys = path.split(".")
        for key in keys[:-1]:
            if key not in target or not isinstance(target[key], dict):
                raise ValueError(f"Unknown override path '{path}'.")
            target = target[key]
        if keys[-1] not in target:
            raise ValueError(f"Unknown override path '{path}'.")
        target[keys[-1]] = value
    return result


def load_experiment_config(path: str | Path, overrides: list[str] | None = None) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}
    if not isinstance(values, dict):
        raise ValueError("Experiment YAML root must be a mapping.")
    # Materialize dataclass defaults before applying CLI overrides, so an
    # override may target a valid section omitted from concise YAML.
    baseline = experiment_from_dict(values).to_dict()
    return experiment_from_dict(apply_overrides(baseline, overrides or []))


def dump_resolved_config(config: ExperimentConfig, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False, allow_unicode=False)


def _semantic_dict(data: dict[str, Any]) -> dict[str, Any]:
    data = json.loads(json.dumps(data))
    logging = data.get("logging", {})
    for key in ("run_id", "resume_from", "allow_existing", "output_root", "experiment_dir", "stage_name"):
        logging.pop(key, None)
    return data
