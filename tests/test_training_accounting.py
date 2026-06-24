from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.alignment import grpo_trainer
from src.alignment.grpo_trainer import GRPOTrainerEngine, reached_completion_token_budget, train_grpo


class DeterministicTrainEngine:
    def __init__(self, output_dir: Path, *, max_steps: int, completion_token_budget: int | None) -> None:
        self.config = SimpleNamespace(
            max_steps=max_steps,
            max_generated_completion_tokens=completion_token_budget,
            logging_steps=1000,
            checkpoint_interval=0,
        )
        self.state = SimpleNamespace(global_step=0, rollout_tokens=0)
        self.output_dir = output_dir
        self.rollout_calls = 0

    def maybe_resume(self) -> None:
        pass

    def rollout(self, example, formatter):
        self.rollout_calls += 1
        self.state.rollout_tokens += 2
        return SimpleNamespace(prompt="prompt", responses=[], rewards=[])

    def optimize(self, batch):
        self.state.global_step += 1
        return SimpleNamespace(step=self.state.global_step, loss=0.0, reward_mean=0.0, kl=0.0, to_dict=lambda: {})

    def checkpoint(self, config) -> None:
        raise AssertionError("checkpointing is disabled in this fake")


def _example() -> SimpleNamespace:
    return SimpleNamespace(reference_answer="answer")


@pytest.mark.parametrize(
    ("rollout_tokens", "completion_token_budget", "expected"),
    [
        (4095, 4096, False),
        (4096, 4096, True),
        (4096, None, False),
    ],
)
def test_reached_completion_token_budget(
    rollout_tokens: int, completion_token_budget: int | None, expected: bool
) -> None:
    assert reached_completion_token_budget(rollout_tokens, completion_token_budget) is expected


def test_grpo_train_stops_before_another_rollout_when_cumulative_budget_is_reached(tmp_path):
    engine = DeterministicTrainEngine(tmp_path, max_steps=10, completion_token_budget=4)

    stop_reason = GRPOTrainerEngine.train(engine, [_example()], formatter=None)

    assert stop_reason == "completion_token_budget"
    assert engine.state.rollout_tokens == 4
    assert engine.rollout_calls == 2


@pytest.mark.parametrize("completion_token_budget", [None, 99])
def test_grpo_train_honors_max_steps_when_budget_is_unset_or_not_reached(tmp_path, completion_token_budget):
    engine = DeterministicTrainEngine(tmp_path, max_steps=3, completion_token_budget=completion_token_budget)

    stop_reason = GRPOTrainerEngine.train(engine, [_example()], formatter=None)

    assert stop_reason == "max_steps"
    assert engine.state.global_step == 3
    assert engine.rollout_calls == 3


@pytest.mark.parametrize(
    ("completion_token_budget", "stop_reason"),
    [(4, "completion_token_budget"), (None, "max_steps")],
)
def test_train_grpo_writes_final_stop_reason(tmp_path, monkeypatch, completion_token_budget, stop_reason):
    class Policy:
        def save_pretrained(self, destination: Path) -> None:
            destination.mkdir(parents=True, exist_ok=True)

    class Fingerprint:
        def to_dict(self) -> dict:
            return {}

    class Engine:
        def __init__(self, *args, **kwargs) -> None:
            self.state = SimpleNamespace(global_step=2, rollout_tokens=4)

        def train(self, examples, formatter) -> str:
            return "completion_token_budget" if completion_token_budget is not None else "max_steps"

        def checkpoint(self, config) -> None:
            pass

    monkeypatch.setattr(grpo_trainer, "_load_backend", lambda *args, **kwargs: Policy())
    monkeypatch.setattr(grpo_trainer.DatasetRegistry, "load", lambda config: ([_example()], Fingerprint(), None))
    monkeypatch.setattr(grpo_trainer, "GRPOTrainerEngine", Engine)
    config = grpo_trainer.GRPOConfig(
        model_name_or_path="tiny",
        output_dir=str(tmp_path),
        max_generated_completion_tokens=completion_token_budget,
        use_reference_policy=False,
    )

    train_grpo(config)

    assert json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8")) == {
        "final_step": 2,
        "rollout_tokens": 4,
        "stop_reason": stop_reason,
    }


def test_run_grpo_cli_parses_completion_token_budget(tmp_path, monkeypatch):
    module_path = Path(__file__).parents[1] / "scripts" / "run_grpo.py"
    spec = importlib.util.spec_from_file_location("test_run_grpo", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    captured = {}
    monkeypatch.setattr(module, "train_grpo", lambda config: captured.setdefault("config", config))
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_grpo.py", "--output_dir", str(tmp_path), "--max_generated_completion_tokens", "4096"],
    )

    module.main()

    assert captured["config"].max_generated_completion_tokens == 4096
