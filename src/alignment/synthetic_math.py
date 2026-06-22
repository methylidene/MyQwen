from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


@dataclass
class MathSample:
    id: str
    difficulty: str
    prompt: str
    answer: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "difficulty": self.difficulty,
            "prompt": self.prompt,
            "answer": self.answer,
            "metadata": self.metadata,
        }


def target_text(answer: str) -> str:
    return f"<reasoning>We compute the arithmetic carefully.</reasoning>\n<answer>{answer}</answer>"


# Original generators retained for experiment provenance. The target profiles below
# are generated into separate directories and never overwrite data/synthetic_math.
#
# def make_easy(rng: random.Random, sid: str) -> MathSample:
#     a, b = rng.randint(0, 99), rng.randint(0, 99)
#     op = rng.choice(["+", "-"])
#     value = a + b if op == "+" else a - b
#     expr = f"{a} {op} {b}"
#     return MathSample(sid, "easy", f"Solve. Return only reasoning and answer tags.\nProblem: {expr}", str(value), {"expr": expr})
#
# def make_medium(rng: random.Random, sid: str) -> MathSample:
#     a, b, c = rng.randint(2, 50), rng.randint(2, 50), rng.randint(2, 12)
#     if rng.random() < 0.5:
#         value = (a + b) * c - rng.randint(0, 20)
#         expr = f"({a} + {b}) * {c} - {((a + b) * c - value)}"
#     else:
#         value = (a - b) * c + rng.randint(0, 20)
#         expr = f"({a} - {b}) * {c} + {(value - (a - b) * c)}"
#     return MathSample(sid, "medium", f"Solve. Use <reasoning>...</reasoning> and <answer>...</answer>.\nProblem: {expr}", str(value), {"expr": expr})
#
# def make_hard(rng: random.Random, sid: str) -> MathSample:
#     apples = rng.choice([24, 30, 36, 42, 48, 60, 72])
#     denom = rng.choice([2, 3, 4, 6])
#     if apples % denom != 0:
#         apples += denom - apples % denom
#     bought = rng.randint(5, 30)
#     left = apples - apples // denom + bought
#     prompt = (
#         "Solve the word problem. Use <reasoning>...</reasoning> and <answer>...</answer>.\n"
#         f"Problem: A has {apples} apples, gives 1/{denom} away, then buys {bought} more. How many now?"
#     )


def make_easy(rng: random.Random, sid: str, profile: str) -> MathSample:
    """Create a controlled single-operation multiplication distribution."""
    if profile == "target_v1":
        a, b = rng.randint(11, 29), rng.randint(2, 9)
    elif profile == "target_v2":
        a, b = rng.randint(20, 49), rng.randint(2, 6)
    else:  # target_v3
        a, b = rng.randint(12, 39), rng.randint(2, 7)
    value = a * b
    expr = f"{a} * {b}"
    return MathSample(sid, "easy", f"Solve. Return only reasoning and answer tags.\nProblem: {expr}", str(value), {"expr": expr, "profile": profile})


def make_medium(rng: random.Random, sid: str, profile: str) -> MathSample:
    """Use addition-only prompts so this bucket calibrates near ceiling accuracy."""
    if profile == "target_v1":
        a, b = rng.randint(10, 49), rng.randint(10, 49)
    elif profile == "target_v2":
        a, b = rng.randint(0, 19), rng.randint(0, 19)
    else:  # target_v3
        a, b = rng.randint(20, 69), rng.randint(0, 9)
    value = a + b
    expr = f"{a} + {b}"
    return MathSample(sid, "medium", f"Solve. Use <reasoning>...</reasoning> and <answer>...</answer>.\nProblem: {expr}", str(value), {"expr": expr, "profile": profile})


def make_hard(rng: random.Random, sid: str, profile: str) -> MathSample:
    """Keep word-problem reasoning while controlling the fractional-operation mix."""
    if profile == "target_v1":
        denom = rng.choice([2, 2, 2, 3])
        apples = rng.choice([24, 30, 36, 42, 48, 60])
    elif profile == "target_v2":
        denom = rng.choice([2, 2, 3])
        apples = rng.choice([18, 24, 30, 36, 42, 48])
    else:  # target_v3
        denom = rng.choice([2, 3, 3, 4])
        apples = rng.choice([24, 36, 48, 60, 72])
    apples -= apples % denom
    bought = rng.randint(5, 20)
    left = apples - apples // denom + bought
    prompt = (
        "Solve the word problem. Use <reasoning>...</reasoning> and <answer>...</answer>.\n"
        f"Problem: A has {apples} apples, gives 1/{denom} away, then buys {bought} more. How many now?"
    )
    return MathSample(
        sid,
        "hard",
        prompt,
        str(left),
        {"apples": apples, "denom": denom, "bought": bought, "profile": profile},
    )


def generate_split(n: int, ratios: tuple[float, float, float], seed: int, prefix: str, profile: str) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    makers = {"easy": make_easy, "medium": make_medium, "hard": make_hard}
    labels = rng.choices(["easy", "medium", "hard"], weights=ratios, k=n)
    rows = []
    for i, label in enumerate(labels):
        sample = makers[label](rng, f"{prefix}-{i:06d}", profile)
        rows.append(sample.to_dict())
    return rows


def generate_dataset(
    num_train: int,
    num_val: int,
    num_test: int,
    easy_ratio: float = 0.4,
    medium_ratio: float = 0.4,
    hard_ratio: float = 0.2,
    seed: int = 42,
    profile: str = "target_v1",
) -> dict[str, list[dict[str, Any]]]:
    if profile not in {"target_v1", "target_v2", "target_v3"}:
        raise ValueError(f"Unknown profile: {profile}")
    ratios = (easy_ratio, medium_ratio, hard_ratio)
    return {
        "train": generate_split(num_train, ratios, seed, "train", profile),
        "val": generate_split(num_val, ratios, seed + 1, "val", profile),
        "test": generate_split(num_test, ratios, seed + 2, "test", profile),
    }
