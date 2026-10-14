"""
GRPO reward-formula variants for SIDReasoner stage S3.

Each variant is a subclass of BaseRewardComputer that overrides `_score()`.
All utility functions (extract_sid_tokens, extract_solution,
construct_prefix_allowed_hashmap) are copied from the original reward file so
that shim files loaded by verl are fully self-contained and never depend on
a cross-submodule import.

Usage (from a shim file):
    from src.grpo.reward_variants import make_rule_base_reward
    rule_base_reward = make_rule_base_reward("A1", "/abs/path/to/item_info.txt")
"""

from __future__ import annotations

import re
from collections import defaultdict, Counter
from pathlib import Path
from typing import Optional

# Repo root = <repo>/src/grpo/reward_variants.py -> parents[2]. Reward shims are
# loaded by verl via exec_module with cwd set to the SIDReasoner submodule, so a
# relative item-info path in a shim won't resolve against the outer repo. Fall back
# to repo-root-relative resolution when a relative path isn't found under cwd.
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(path: str) -> str:
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        candidate = _REPO_ROOT / p
        if candidate.exists():
            return str(candidate)
    return path

# ---------------------------------------------------------------------------
# Shared utilities (copied verbatim from direct_recommendation_StepRule_Games.py
# to keep shim files self-contained when verl loads them by path).
# ---------------------------------------------------------------------------

_SOLUTION_CLIP_CHARS = 50


def _extract_sid_tokens(s: str) -> list[str]:
    return re.findall(r"<[^>]+>", s)


def _extract_solution(solution_str: str) -> Optional[list[str]]:
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]
    match = re.search(r"</think>\s*(.*)", solution_str, re.DOTALL)
    if match:
        answer_sids = _extract_sid_tokens(match.group(1).strip())[:3]
        if len(answer_sids) == 3:
            return answer_sids
    return None


def _extract_think_text(solution_str: str) -> str:
    """Return the text inside <think>…</think>, or '' if absent."""
    match = re.search(r"<think>(.*?)</think>", solution_str, re.DOTALL)
    return match.group(1) if match else ""


def _construct_prefix_map(item_info_path: str) -> dict:
    item_info_path = _resolve_path(item_info_path)
    sid_pattern = re.compile(r"<[^>]+>")
    prefix_map: dict = defaultdict(set)
    with open(item_info_path) as f:
        for line in f:
            sids = sid_pattern.findall(line.split("\t")[0].strip())
            if len(sids) != 3:
                continue
            a, b, c = sids
            prefix_map[(a,)].add(b)
            prefix_map[(a, b)].add(c)
    return {k: list(v) for k, v in prefix_map.items()}


def _is_valid_sid(sids: list[str], prefix_map: dict) -> bool:
    if len(sids) < 3:
        return False
    a, b, c = sids[:3]
    return (
        (a,) in prefix_map
        and b in prefix_map[(a,)]
        and (a, b) in prefix_map
        and c in prefix_map[(a, b)]
    )


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseRewardComputer:
    """Lazy-loading singleton base; subclasses override `_score()`."""

    def __init__(self, item_info_path: str):
        self.prefix_map = _construct_prefix_map(item_info_path)

    def compute(
        self,
        data_source: str,
        solution_str: str,
        ground_truth: str,
        extra_info: Optional[dict] = None,
    ) -> float:
        answer = _extract_solution(solution_str)
        if answer is None:
            return 0.0
        gt = _extract_sid_tokens(ground_truth)[:3]
        return self._score(answer, gt, solution_str)

    def _score(self, answer: list[str], gt: list[str], solution_str: str) -> float:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# A0 — Baseline
# Change   : R = R_sr + 0.1·R_f  (original formula)
# Hypothesis: Reference; all other variants compared against this.
# ---------------------------------------------------------------------------

class RewardA0Baseline(BaseRewardComputer):
    """
    Change   : R = R_sr + 0.1·R_f  (original formula, λ=0.1)
    Hypothesis: Reference condition. All other variants are compared against
                this baseline. R_sr is hierarchical (0→0.25→0.5→1.0);
                R_f is 1 if the predicted SID is a valid catalogue path.
    """

    def _score(self, answer: list[str], gt: list[str], solution_str: str) -> float:
        r_acc = _hierarchical_acc(answer, gt)
        r_fmt = float(_is_valid_sid(answer, self.prefix_map))
        return r_acc + 0.1 * r_fmt


# ---------------------------------------------------------------------------
# A1 — No format term
# Change   : R = R_sr  (λ = 0, format reward removed)
# Hypothesis: Is the format reward actually helping, or does GRPO+KL already
#             keep outputs valid without the explicit validity bonus?
# ---------------------------------------------------------------------------

class RewardA1NoFormat(BaseRewardComputer):
    """
    Change   : R = R_sr  (λ = 0, format reward removed entirely)
    Hypothesis: Is the format reward actually helping, or does GRPO+KL already
                keep outputs syntactically valid without an explicit bonus?
                If format rate stays high without the bonus, the 0.1 term is
                redundant signal.
    """

    def _score(self, answer: list[str], gt: list[str], solution_str: str) -> float:
        return _hierarchical_acc(answer, gt)


# ---------------------------------------------------------------------------
# A2 — Sweep λ (format-reward weight)
# Change   : λ ∈ {0.05, 0.2, 0.5, 1.0}  (vary the format-reward coefficient)
# Hypothesis: Trade-off between catalogue-validity pressure and pure accuracy.
#             A larger λ forces valid SIDs even when wrong; a smaller λ lets
#             the model prioritise accuracy over format compliance.
# ---------------------------------------------------------------------------

class RewardA2SweepLambda(BaseRewardComputer):
    """
    Change   : R = R_sr + λ·R_f, with λ swept over {0.05, 0.2, 0.5, 1.0}
    Hypothesis: Explores the trade-off between catalogue-validity pressure and
                accuracy. Larger λ forces valid SIDs even when wrong; smaller λ
                lets the model focus on accuracy over format compliance.
    """

    def __init__(self, item_info_path: str, lam: float = 0.1):
        super().__init__(item_info_path)
        self.lam = lam

    def _score(self, answer: list[str], gt: list[str], solution_str: str) -> float:
        r_acc = _hierarchical_acc(answer, gt)
        r_fmt = float(_is_valid_sid(answer, self.prefix_map))
        return r_acc + self.lam * r_fmt


# ---------------------------------------------------------------------------
# A3 — Linear / partial-credit accuracy
# Change   : R_sr = 0.33·[a1=g1] + 0.33·[a1=g1 & a2=g2] + 0.34·[all match]
#            (additive, not multiplicative)
# Implemented weights: cumulative prefix credit is 0, 0.33, 0.66, 1.0.
# This implementation defines the reward used for the reported experiments.
# Hypothesis: Does softer, more linear reward shaping beat the steep ×2
#             doubling? Reduces reward sparsity by rewarding partial matches
#             more evenly across all three SID levels.
# ---------------------------------------------------------------------------

class RewardA3LinearAcc(BaseRewardComputer):
    """
    Change   : R_sr = 0.33·[a1=g1] + 0.33·[a1=g1 & a2=g2] + 0.34·[all match]
               (additive accumulation, not multiplicative doubling)
    Hypothesis: Does softer, more linear reward shaping beat the steep ×2
                doubling of A0? Reducing reward sparsity by giving more uniform
                credit for partial SID matches may produce a smoother gradient
                signal and faster convergence.
    """

    def _score(self, answer: list[str], gt: list[str], solution_str: str) -> float:
        r_acc = 0.0
        if answer[0] == gt[0]:
            r_acc += 0.33
            if answer[1] == gt[1]:
                r_acc += 0.33
                if answer[2] == gt[2]:
                    r_acc += 0.34
        r_fmt = float(_is_valid_sid(answer, self.prefix_map))
        return r_acc + 0.1 * r_fmt





# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _hierarchical_acc(answer: list[str], gt: list[str]) -> float:
    """Original hierarchical accuracy: {0, 0.25, 0.50, 1.0}."""
    score = 0.0
    if answer[0] == gt[0]:
        score = 0.25
        if answer[1] == gt[1]:
            score *= 2  # 0.50
            if answer[2] == gt[2]:
                score *= 2  # 1.00
    return score


def _hierarchical_acc_with_floor(answer: list[str], gt: list[str], floor: float) -> float:
    """Hierarchical accuracy with a non-zero floor on a complete miss."""
    score = _hierarchical_acc(answer, gt)
    return score if score > 0.0 else floor


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_VARIANT_MAP = {
    "A0": RewardA0Baseline,
    "A1": RewardA1NoFormat,
    "A2": RewardA2SweepLambda,
    "A3": RewardA3LinearAcc,
}


def make_rule_base_reward(variant: str, item_info_path: str, **kwargs):
    """
    Return a verl-compatible `rule_base_reward` callable for `variant`.

    Args:
        variant      : One of 'A0'-'A3'.
        item_info_path: Absolute path to the dataset item-info .txt file.
        **kwargs     : Passed to the variant constructor (e.g. lam=0.2 for A2).

    Returns:
        A function with signature
            rule_base_reward(data_source, solution_str, ground_truth,
                             extra_info=None) -> float
        suitable for use as `custom_reward_function.name=rule_base_reward` in verl.
    """
    cls = _VARIANT_MAP.get(variant.upper())
    if cls is None:
        raise ValueError(f"Unknown reward variant '{variant}'. Choose from {list(_VARIANT_MAP)}")

    # Lazy singleton: verl may import this module multiple times per process;
    # compute the prefix map only once.
    _instance: list = []  # mutable closure cell

    def rule_base_reward(data_source, solution_str, ground_truth, extra_info=None):
        if not _instance:
            _instance.append(cls(item_info_path, **kwargs))
        return _instance[0].compute(data_source, solution_str, ground_truth, extra_info)

    return rule_base_reward
