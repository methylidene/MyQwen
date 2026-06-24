"""Token-level log-probability, KL and clipped-policy utilities."""
from __future__ import annotations

import torch


def sequence_logprobs(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return selected next-token log probabilities, zeroed outside ``mask``."""
    logp = torch.log_softmax(logits, dim=-1)
    gathered = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return gathered * mask


def approximate_kl(policy_logprobs: torch.Tensor, ref_logprobs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Non-negative sampled KL estimator over completion tokens only."""
    denom = mask.sum().clamp_min(1.0)
    log_ratio = (ref_logprobs - policy_logprobs) * mask
    per_token_kl = torch.exp(log_ratio) - 1.0 - log_ratio
    return (per_token_kl * mask).sum() / denom


def token_entropy(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean categorical entropy over tokens selected by ``mask``."""
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    return (entropy * mask).sum() / mask.sum().clamp_min(1.0)


def normalize_group_advantages(rewards: torch.Tensor, epsilon: float = 1e-6) -> tuple[torch.Tensor, bool]:
    """Normalize one prompt's rollout group; constant groups receive zero advantage."""
    std = rewards.std(unbiased=False)
    if float(std.detach()) <= epsilon:
        return torch.zeros_like(rewards), True
    return (rewards - rewards.mean()) / std.clamp_min(epsilon), False


def clipped_policy_loss(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PPO-style clipped objective using rollout-policy ``old_logprobs``.

    Per-sequence objectives are length-normalized over completion tokens before
    averaging across group members.
    """
    if clip_eps < 0:
        raise ValueError("clip_eps must be non-negative.")
    ratio = torch.exp(new_logprobs - old_logprobs) * mask + (1.0 - mask)
    unclipped = ratio * advantages.unsqueeze(-1)
    clipped = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * advantages.unsqueeze(-1)
    objective = torch.minimum(unclipped, clipped) * mask
    per_sequence = objective.sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)
    return -per_sequence.mean(), ratio
