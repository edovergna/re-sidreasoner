"""A2 λ=0.2 — Office dataset. verl entry-point: rule_base_reward."""
from src.grpo.reward_variants import make_rule_base_reward

rule_base_reward = make_rule_base_reward(
    "A2", "data/raw/office/info/Office_Products_5_2016-10-2018-11.txt", lam=0.2
)
