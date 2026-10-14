"""A2 λ=1.0 — Games dataset. verl entry-point: rule_base_reward."""
from src.grpo.reward_variants import make_rule_base_reward

rule_base_reward = make_rule_base_reward(
    "A2", "data/raw/games/info/Video_Games_5_2016-10-2018-11.txt", lam=1.0
)
