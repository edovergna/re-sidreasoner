"""A0 Baseline — Office dataset. verl entry-point: rule_base_reward."""
from src.grpo.reward_variants import make_rule_base_reward

rule_base_reward = make_rule_base_reward(
    "A0", "data/raw/office/info/Office_Products_5_2016-10-2018-11.txt"
)
