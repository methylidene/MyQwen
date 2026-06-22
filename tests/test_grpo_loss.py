import torch

from src.alignment.kl_utils import approximate_kl, sequence_logprobs


def test_approximate_kl_masking():
    policy = torch.tensor([[1.0, 2.0, 100.0]])
    ref = torch.tensor([[0.5, 1.5, -100.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    expected = torch.exp(torch.tensor(-0.5)) - 1.0 + 0.5
    assert torch.isclose(approximate_kl(policy, ref, mask), expected)


def test_approximate_kl_is_non_negative_for_signed_logprob_differences():
    policy = torch.tensor([[0.0, 3.0, -1.0]])
    ref = torch.tensor([[1.0, 1.0, -4.0]])
    mask = torch.ones_like(policy)
    assert approximate_kl(policy, ref, mask) >= 0.0


def test_sequence_logprobs_shape():
    logits = torch.randn(2, 3, 5)
    labels = torch.tensor([[1, 2, 3], [0, 4, 1]])
    mask = torch.ones(2, 3)
    out = sequence_logprobs(logits, labels, mask)
    assert out.shape == (2, 3)
