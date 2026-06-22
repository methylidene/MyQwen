from __future__ import annotations

import torch


def sequence_logprobs(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    logp = torch.log_softmax(logits, dim=-1)
    gathered = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return gathered * mask


def approximate_kl(policy_logprobs: torch.Tensor, ref_logprobs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum().clamp_min(1.0)
    log_ratio = (ref_logprobs - policy_logprobs) * mask
    per_token_kl = torch.exp(log_ratio) - 1.0 - log_ratio
    return (per_token_kl * mask).sum() / denom
