from src.alignment.rewards import extract_answer, rule_based_reward


def test_extract_answer():
    assert extract_answer("<reasoning>x</reasoning><answer>42</answer>") == "42"


def test_rule_reward_correct():
    reward = rule_based_reward("<reasoning>x</reasoning><answer>42</answer>", "42")
    assert reward["accuracy"] == 1.0
    assert reward["format_pass"] == 1.0
    assert reward["total_reward"] > 1.0


def test_rule_reward_invalid():
    reward = rule_based_reward("forty two", "42")
    assert reward["invalid"] == 1.0
    assert reward["total_reward"] < 0.0


def test_rule_reward_currency_answer_is_parseable():
    reward = rule_based_reward("<reasoning>x</reasoning><answer>$42</answer>", "42")
    assert reward["accuracy"] == 1.0
    assert reward["format_pass"] == 1.0
