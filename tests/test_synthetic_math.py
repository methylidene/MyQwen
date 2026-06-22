from src.alignment.synthetic_math import generate_dataset


def test_generate_dataset_sizes_and_schema():
    data = generate_dataset(4, 2, 2, seed=7)
    assert len(data["train"]) == 4
    row = data["train"][0]
    assert {"id", "difficulty", "prompt", "answer", "metadata"} <= set(row)
    assert row["difficulty"] in {"easy", "medium", "hard"}
