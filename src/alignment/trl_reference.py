"""Optional TRL reference bridge for small alignment cross-checks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from src.alignment.training_core import RewardPipeline
from src.data import PromptFormatter, ReasoningExample


@dataclass(frozen=True)
class TRLComparisonReport:
    """Comparable small-smoke output from a user-provided TRL execution."""

    loss: float | None
    reward_mean: float | None
    kl: float | None
    samples: list[dict[str, Any]]
    implementation: str = "trl"


class TRLReferenceRunner:
    """Lazy optional runner that keeps prompt/reward/generation semantics shared.

    TRL changes APIs frequently, so callers supply a small ``run_callable``
    that builds the appropriate TRL trainer for their installed version. This
    class supplies normalized prompts, reward adapter and comparison reporting;
    it never becomes the production training path.
    """

    def __init__(self, formatter: PromptFormatter, reward_pipeline: RewardPipeline) -> None:
        self.formatter = formatter
        self.reward_pipeline = reward_pipeline

    @staticmethod
    def require_trl() -> Any:
        try:
            import trl
        except ImportError as exc:
            raise RuntimeError("TRLReferenceRunner requires the optional 'trl' package. Install it with: pip install trl") from exc
        return trl

    def run_smoke(
        self,
        examples: list[ReasoningExample],
        generation_kwargs: dict[str, Any],
        run_callable: Callable[[list[dict[str, Any]], RewardPipeline, dict[str, Any]], dict[str, Any]],
    ) -> TRLComparisonReport:
        """Run a version-specific tiny TRL callable with shared semantics.

        ``run_callable`` must return ``loss``, ``reward_mean``, ``kl`` and
        ``samples``. Its prompts and reward adapter are supplied here so a
        comparison cannot accidentally use different templates or rewards.
        """
        self.require_trl()
        rows = [
            {
                "uid": item.uid,
                "prompt": self.formatter.grpo_prompt(item),
                "reference_answer": item.reference_answer,
            }
            for item in examples
        ]
        result = run_callable(rows, self.reward_pipeline, generation_kwargs)
        return TRLComparisonReport(
            loss=result.get("loss"),
            reward_mean=result.get("reward_mean"),
            kl=result.get("kl"),
            samples=list(result.get("samples", [])),
        )

    @staticmethod
    def difference_report(self_metrics: dict[str, float], trl_metrics: TRLComparisonReport) -> dict[str, float | None]:
        return {
            key: None if getattr(trl_metrics, key, None) is None else self_metrics.get(key, 0.0) - getattr(trl_metrics, key)
            for key in ("loss", "reward_mean", "kl")
        }
