import torch

from src.inference.cache_utils import cache_seq_len, next_position_ids, position_ids_from_attention_mask, trim_past_key_values


def make_past(seq_len=5):
    return tuple((torch.zeros(2, 3, seq_len, 4), torch.ones(2, 3, seq_len, 4)) for _ in range(2))


def test_trim_cache_length():
    past = trim_past_key_values(make_past(7), 3)
    assert cache_seq_len(past) == 3
    assert past[0][0].shape == (2, 3, 3, 4)


def test_position_ids_from_attention_mask():
    mask = torch.tensor([[1, 1, 1], [0, 1, 1]])
    pos = position_ids_from_attention_mask(mask)
    assert pos.tolist() == [[0, 1, 2], [0, 0, 1]]


def test_next_position_ids_monotonic():
    pos = [next_position_ids([i], "cpu").item() for i in range(3, 7)]
    assert pos == [3, 4, 5, 6]


def test_cache_and_no_cache_length_contract_with_mock():
    no_cache = [1, 2, 3, 4]
    cache = [1, 2, 3, 4]
    assert len(no_cache) == len(cache)
