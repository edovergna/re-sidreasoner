"""A1 No format term — Games dataset. verl entry-point: rule_base_reward."""
from src.grpo.reward_variants import make_rule_base_reward

rule_base_reward = make_rule_base_reward(
    "A1", "data/raw/games/info/Video_Games_5_2016-10-2018-11.txt"
)
