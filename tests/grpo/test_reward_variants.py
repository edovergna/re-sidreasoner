"""
Smoke tests for src/grpo/reward_variants.py.

No GPU, no real data files.  A temporary item-info file is synthesised so that
`construct_prefix_map` has a real (but tiny) catalogue to work with.

Run from repo root:
    python -m pytest tests/grpo/test_reward_variants.py -v
or:
    python tests/grpo/test_reward_variants.py
"""

import os
import sys
import tempfile
import textwrap
import importlib
from pathlib import Path

import pytest


def _find_repo_root(start: Path) -> Path:
    """Walk up to the repo root (sentinel: .gitmodules next to baselines/)."""
    for c in (start, *start.parents):
        if (c / ".gitmodules").is_file() and (c / "baselines").is_dir():
            return c
    raise FileNotFoundError("repo root (.gitmodules + baselines/) not found")


# Make sure the repo root is on sys.path when run directly (depth-independent).
_REPO_ROOT = str(_find_repo_root(Path(__file__).resolve()))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.grpo.reward_variants import (
    make_rule_base_reward,
    _hierarchical_acc,
    _is_valid_sid,
    _construct_prefix_map,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Tiny catalogue: 4 items, 2 root tokens, 2 second-level each.
_CATALOGUE = textwrap.dedent("""\
    <A><B1><C1>\titem1
    <A><B1><C2>\titem2
    <A><B2><C3>\titem3
    <X><Y1><Z1>\titem4
""")

GT  = ["<A>", "<B1>", "<C1>"]   # ground truth
ALT = ["<A>", "<B1>", "<C2>"]   # wrong only at level-3
BAD = ["<A>", "<B2>", "<C1>"]   # wrong at level-3 and invalid (C1 not under B2)
UNK = ["<A>", "<B9>", "<C9>"]   # level-2 unknown


@pytest.fixture(scope="module")
def item_info_path(tmp_path_factory):
    p = tmp_path_factory.mktemp("data") / "item_info.txt"
    p.write_text(_CATALOGUE)
    return str(p)


def _solution(a, b, c, think_len=10):
    think_part = "x" * think_len
    return f"<think>{think_part}</think> {a}{b}{c}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def test_hierarchical_acc_full_match():
    assert _hierarchical_acc(GT, GT) == 1.0


def test_hierarchical_acc_level2():
    assert _hierarchical_acc(["<A>", "<B1>", "<C99>"], GT) == pytest.approx(0.5)


def test_hierarchical_acc_level1():
    assert _hierarchical_acc(["<A>", "<B99>", "<C99>"], GT) == pytest.approx(0.25)


def test_hierarchical_acc_miss():
    assert _hierarchical_acc(["<X>", "<B1>", "<C1>"], GT) == 0.0


def test_is_valid_sid(item_info_path):
    prefix_map = _construct_prefix_map(item_info_path)
    assert _is_valid_sid(GT, prefix_map)
    assert _is_valid_sid(ALT, prefix_map)
    assert not _is_valid_sid(BAD, prefix_map)
    assert not _is_valid_sid(UNK, prefix_map)


# ---------------------------------------------------------------------------
# Per-variant reward value checks
# ---------------------------------------------------------------------------

def _make(variant, item_info_path, **kwargs):
    return make_rule_base_reward(variant, item_info_path, **kwargs)


class TestA0Baseline:
    def test_full_match(self, item_info_path):
        fn = _make("A0", item_info_path)
        r = fn(None, _solution(*GT), " ".join(GT))
        assert r == pytest.approx(1.0 + 0.1)  # acc=1.0, fmt=1

    def test_level2_valid(self, item_info_path):
        fn = _make("A0", item_info_path)
        r = fn(None, _solution(*ALT), " ".join(GT))
        assert r == pytest.approx(0.5 + 0.1)  # acc=0.5, fmt=1 (ALT is valid)

    def test_no_think(self, item_info_path):
        fn = _make("A0", item_info_path)
        r = fn(None, "no think tag here", " ".join(GT))
        assert r == 0.0


class TestA1NoFormat:
    def test_full_match_no_fmt_bonus(self, item_info_path):
        fn = _make("A1", item_info_path)
        r = fn(None, _solution(*GT), " ".join(GT))
        assert r == pytest.approx(1.0)

    def test_level1_only(self, item_info_path):
        fn = _make("A1", item_info_path)
        r = fn(None, _solution(*UNK), " ".join(GT))
        # UNK has same level-0 token <A> as GT
        assert r == pytest.approx(0.25)


class TestA2SweepLambda:
    def test_lambda_zero_same_as_a1(self, item_info_path):
        fn = _make("A2", item_info_path, lam=0.0)
        r = fn(None, _solution(*GT), " ".join(GT))
        assert r == pytest.approx(1.0)

    def test_lambda_sweep(self, item_info_path):
        for lam in (0.05, 0.2, 0.5, 1.0):
            fn = _make("A2", item_info_path, lam=lam)
            r = fn(None, _solution(*GT), " ".join(GT))
            assert r == pytest.approx(1.0 + lam), f"failed for lam={lam}"


class TestA3LinearAcc:
    def test_full_match(self, item_info_path):
        fn = _make("A3", item_info_path)
        r = fn(None, _solution(*GT), " ".join(GT))
        # 0.33 + 0.33 + 0.34 + 0.1*1 = 1.1
        assert r == pytest.approx(0.33 + 0.33 + 0.34 + 0.1)

    def test_level2_match(self, item_info_path):
        fn = _make("A3", item_info_path)
        r = fn(None, _solution(*ALT), " ".join(GT))
        # a1=a1, a2=a2 match; a3 differs → 0.33+0.33 + 0.1*1(ALT valid)
        assert r == pytest.approx(0.33 + 0.33 + 0.1)

    def test_level1_only(self, item_info_path):
        fn = _make("A3", item_info_path)
        r = fn(None, _solution(*["<A>", "<B9>", "<C9>"]), " ".join(GT))
        # only a1 matches, format invalid → 0.33
        assert r == pytest.approx(0.33)





# ---------------------------------------------------------------------------
# Shim import check (no data files needed — just checks the module syntax)
# ---------------------------------------------------------------------------

def test_shim_import_syntax():
    """Each shim must be importable without data-file access."""
    import importlib.util, pathlib
    shim_dir = pathlib.Path(_REPO_ROOT) / "src" / "training" / "reward_shims"
    for shim_file in sorted(shim_dir.glob("a*.py")):
        spec = importlib.util.spec_from_file_location(shim_file.stem, shim_file)
        mod = importlib.util.module_from_spec(spec)
        # We only check that the module defines rule_base_reward at the top level
        # without executing the body (which would try to open data files).
        # We do this by inspecting the source AST, not by loading the module.
        import ast
        src = shim_file.read_text()
        tree = ast.parse(src)
        assigns = [
            node.targets[0].id
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
        ]
        assert "rule_base_reward" in assigns, (
            f"{shim_file.name} does not assign rule_base_reward at module level"
        )


# ---------------------------------------------------------------------------
# make_rule_base_reward factory error handling
# ---------------------------------------------------------------------------

def test_unknown_variant_raises(item_info_path):
    with pytest.raises(ValueError, match="Unknown reward variant"):
        make_rule_base_reward("Z9", item_info_path)


# ---------------------------------------------------------------------------
# Entry point for running without pytest
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import traceback, sys

    passed = failed = 0
    tests = [
        test_hierarchical_acc_full_match,
        test_hierarchical_acc_level2,
        test_hierarchical_acc_level1,
        test_hierarchical_acc_miss,
    ]

    with tempfile.TemporaryDirectory() as tmp:
        info = os.path.join(tmp, "item_info.txt")
        with open(info, "w") as f:
            f.write(_CATALOGUE)

        variant_tests = [
            TestA0Baseline(),
            TestA1NoFormat(),
            TestA2SweepLambda(),
            TestA3LinearAcc(),
        ]

        print("Running helper tests...")
        for fn in tests:
            try:
                fn()
                print(f"  PASS  {fn.__name__}")
                passed += 1
            except Exception:
                print(f"  FAIL  {fn.__name__}")
                traceback.print_exc()
                failed += 1

        print("\nRunning variant tests...")
        for obj in variant_tests:
            cls_name = type(obj).__name__
            for method_name in dir(obj):
                if not method_name.startswith("test_"):
                    continue
                try:
                    getattr(obj, method_name)(info)
                    print(f"  PASS  {cls_name}.{method_name}")
                    passed += 1
                except Exception:
                    print(f"  FAIL  {cls_name}.{method_name}")
                    traceback.print_exc()
                    failed += 1

        print(f"\nResults: {passed} passed, {failed} failed")
        sys.exit(1 if failed else 0)
