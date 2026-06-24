"""Shared alignment training contracts, reward/generation adapters and checkpoints."""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

import torch

from src.alignment.rewards import rule_based_reward
from src.models.backend import ModelBackend
from src.utils.io import save_json


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8


@dataclass(frozen=True)
class AlignmentConfig:
    seed: int = 42
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    checkpoint_interval: int = 0
    checkpoint_keep: int = 2
    resume_from_checkpoint: str | None = None
    deterministic_smoke: bool = False


@dataclass
class TrainerState:
    global_step: int = 0
    micro_step: int = 0
    epoch: int = 0
    optimizer_steps: int = 0
    train_tokens: int = 0
    rollout_tokens: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass
class TrainingMetrics:
    step: int
    loss: float
    policy_loss: float | None = None
    kl_loss: float | None = None
    kl: float | None = None
    entropy: float | None = None
    clip_fraction: float | None = None
    reward_mean: float | None = None
    reward_std: float | None = None
    advantage_mean: float | None = None
    advantage_std: float | None = None
    zero_variance_groups: int = 0
    truncated_completions: int = 0
    rollout_tokens: int = 0
    train_tokens: int = 0
    rollout_seconds: float = 0.0
    optimization_seconds: float = 0.0
    reward_components: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key, value in self.reward_components.items():
            data[f"reward_{key}"] = value
        return data


@dataclass
class RolloutBatch:
    prompt: str
    prompt_length: int
    generated_ids: torch.Tensor
    attention_mask: torch.Tensor
    completion_mask: torch.Tensor
    responses: list[str]
    rewards: list[dict[str, Any]]
    old_logprobs: torch.Tensor
    reference_logprobs: torch.Tensor | None
    truncated: torch.Tensor
    rollout_seconds: float


@dataclass
class GRPOBatch(RolloutBatch):
    advantages: torch.Tensor | None = None


class GenerationBackend(Protocol):
    """Generation role used during rollout, independent from model implementation."""

    def generate(self, **kwargs: Any) -> torch.Tensor:
        ...


class ModelGenerationBackend:
    """Adapter exposing a :class:`ModelBackend` as a rollout generator."""

    def __init__(self, model: ModelBackend) -> None:
        self.model = model

    def generate(self, **kwargs: Any) -> torch.Tensor:
        return self.model.generate(**kwargs)


class RewardPipeline(Protocol):
    """Evaluate generated completions without coupling trainers to one reward function."""

    def evaluate(self, responses: list[str], reference_answer: str) -> list[dict[str, Any]]:
        ...


class RuleBasedRewardPipeline:
    """Default reward adapter around the project's existing arithmetic reward."""

    def evaluate(self, responses: list[str], reference_answer: str) -> list[dict[str, Any]]:
        return [rule_based_reward(response, reference_answer) for response in responses]


class CheckpointManager:
    """Persist model, optimizer, trainer state, RNG and resolved configuration."""

    def __init__(self, output_dir: str | Path, keep_last: int = 2) -> None:
        self.output_dir = Path(output_dir)
        self.keep_last = max(1, keep_last)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(self, backend: ModelBackend, optimizer: torch.optim.Optimizer, state: TrainerState, resolved_config: dict[str, Any]) -> Path:
        target = self.output_dir / f"checkpoint-step-{state.global_step:08d}"
        target.mkdir(parents=True, exist_ok=True)
        backend.save_pretrained(target / "model")
        payload = {
            "trainer_state": state.to_dict(),
            "optimizer": optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "python_rng": random.getstate(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        torch.save(payload, target / "trainer_state.pt")
        save_json(resolved_config, target / "resolved_config.json")
        self._rotate()
        return target

    def load(self, checkpoint_dir: str | Path, optimizer: torch.optim.Optimizer) -> TrainerState:
        path = Path(checkpoint_dir) / "trainer_state.pt"
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        optimizer.load_state_dict(payload["optimizer"])
        torch.set_rng_state(payload["torch_rng"])
        random.setstate(payload["python_rng"])
        if torch.cuda.is_available() and payload.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
        return TrainerState(**payload["trainer_state"])

    def latest(self) -> Path | None:
        checkpoints = sorted(self.output_dir.glob("checkpoint-step-*"))
        return checkpoints[-1] if checkpoints else None

    def _rotate(self) -> None:
        checkpoints = sorted(self.output_dir.glob("checkpoint-step-*"))
        for old in checkpoints[:-self.keep_last]:
            import shutil
            shutil.rmtree(old)


class BaseAlignmentTrainer:
    """Common state, optimizer and checkpoint operations for self-hosted trainers."""

    def __init__(self, backend: ModelBackend, optimizer_config: OptimizerConfig, alignment_config: AlignmentConfig, output_dir: str | Path) -> None:
        self.backend = backend
        self.alignment_config = alignment_config
        self.optimizer = torch.optim.AdamW(
            backend.parameters(),
            lr=optimizer_config.learning_rate,
            weight_decay=optimizer_config.weight_decay,
            betas=optimizer_config.betas,
            eps=optimizer_config.eps,
        )
        self.state = TrainerState()
        self.checkpoints = CheckpointManager(output_dir, alignment_config.checkpoint_keep)
        if alignment_config.gradient_checkpointing:
            backend.enable_gradient_checkpointing()

    def maybe_resume(self) -> None:
        if self.alignment_config.resume_from_checkpoint:
            self.state = self.checkpoints.load(self.alignment_config.resume_from_checkpoint, self.optimizer)

    def checkpoint(self, resolved_config: dict[str, Any]) -> Path:
        return self.checkpoints.save(self.backend, self.optimizer, self.state, resolved_config)

    @staticmethod
    def now() -> float:
        return perf_counter()
