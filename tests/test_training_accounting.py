from __future__ import annotations

import pytest

from src.alignment.grpo_trainer import reached_completion_token_budget


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
