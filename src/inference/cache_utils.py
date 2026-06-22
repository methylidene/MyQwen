from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CacheStats:
    rebuild_count: int = 0
    lengths: list[int] | None = None

    def add_length(self, length: int) -> None:
        if self.lengths is None:
            self.lengths = []
        self.lengths.append(int(length))

    @property
    def avg_cache_seq_len(self) -> float:
        if not self.lengths:
            return 0.0
        return sum(self.lengths) / len(self.lengths)

    @property
    def max_cache_seq_len(self) -> int:
        if not self.lengths:
            return 0
        return max(self.lengths)


def _layer_key_value(layer: Any) -> tuple[Any, Any]:
    if isinstance(layer, (tuple, list)) and len(layer) >= 2:
        return layer[0], layer[1]
    raise TypeError("past_key_values layers must be tuples/lists containing key and value tensors")


def cache_seq_len(past_key_values: Any) -> int:
    if past_key_values is None or len(past_key_values) == 0:
        return 0
    key, _ = _layer_key_value(past_key_values[0])
    return int(key.shape[-2])


def trim_past_key_values(past_key_values: Any, window: int | None) -> Any:
    if past_key_values is None or window is None or window <= 0:
        return past_key_values
    trimmed = []
    for layer in past_key_values:
        key, value = _layer_key_value(layer)
        extras = tuple(layer[2:]) if isinstance(layer, tuple) else []
        trimmed.append((key[..., -window:, :], value[..., -window:, :], *extras))
    return tuple(trimmed)


def position_ids_from_attention_mask(attention_mask):
    pos = attention_mask.long().cumsum(dim=-1) - 1
    return pos.clamp_min(0).masked_fill(attention_mask == 0, 0)


def next_position_ids(context_lengths, device):
    import torch

    return torch.as_tensor(context_lengths, dtype=torch.long, device=device).unsqueeze(-1)
