"""Smoke test 3: verify eval mode selection and checkpoint inference for every config.

Expectations are derived from each config's `stages` block (taxonomy-agnostic),
not a hard-coded experiment-name map:
    - S3 enabled              → thinking eval, checkpoint suffix "stage3_rl"
    - S2 enabled (no S3)      → thinking eval, "stage2_reasoning_activation/final_checkpoint"
    - otherwise (S1 only)     → beam eval,     "stage1_sft/final_checkpoint"
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from conftest import ALL_CONFIGS, REPO_ROOT


def _load_evaluate_module():
    """Dynamically import scripts/common/eval/evaluate.py to test its functions directly."""
    spec = importlib.util.spec_from_file_location(
        "evaluate",
        REPO_ROOT / "scripts" / "common" / "eval" / "evaluate.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load(config_rel: str) -> dict:
    with (REPO_ROOT / config_rel).open() as f:
        return yaml.safe_load(f)


def _expected(cfg: dict) -> tuple[bool, str]:
    """Return (uses_thinking, checkpoint_path_suffix) implied by the config's stages."""
    stages = cfg["stages"]
    if stages["s3"]["enabled"]:
        return True, "stage3_rl"
    if stages["s2"]["enabled"]:
        return True, "stage2_reasoning_activation/final_checkpoint"
    return False, "stage1_sft/final_checkpoint"


def _run_dry_eval(config_rel: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "scripts/common/eval/evaluate.py", "--config", config_rel, "--dry-run"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )


# --- Unit tests: directly exercise infer_checkpoint() and uses_thinking() ---

@pytest.mark.parametrize("config_rel", ALL_CONFIGS)
def test_uses_thinking_flag(config_rel):
    ev = _load_evaluate_module()
    cfg = _load(config_rel)
    expected, _ = _expected(cfg)
    got = ev.uses_thinking(cfg)
    assert got == expected, (
        f"uses_thinking({config_rel}): expected {expected}, got {got}"
    )


@pytest.mark.parametrize("config_rel", ALL_CONFIGS)
def test_infer_checkpoint_path_suffix(config_rel):
    ev = _load_evaluate_module()
    cfg = _load(config_rel)
    _, expected_suffix = _expected(cfg)
    ckpt = ev.infer_checkpoint(cfg)
    assert str(ckpt).endswith(expected_suffix), (
        f"infer_checkpoint({config_rel}): "
        f"expected suffix '{expected_suffix}', got '{ckpt}'"
    )


# --- Subprocess dry-run tests ---

@pytest.mark.parametrize("config_rel", ALL_CONFIGS)
def test_dryrun_eval_exits_zero(config_rel):
    result = _run_dry_eval(config_rel)
    assert result.returncode == 0, (
        f"evaluate.py --dry-run failed (exit {result.returncode}) for {config_rel}\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )


@pytest.mark.parametrize("config_rel", ALL_CONFIGS)
def test_dryrun_eval_think_mode_line(config_rel):
    result = _run_dry_eval(config_rel)
    out = result.stdout
    expected_thinking, _ = _expected(_load(config_rel))
    if expected_thinking:
        assert "Think mode:  True" in out, (
            f"Expected 'Think mode:  True' for {config_rel}\nOutput:\n{out}"
        )
    else:
        assert "Think mode:  False" in out, (
            f"Expected 'Think mode:  False' for {config_rel}\nOutput:\n{out}"
        )


@pytest.mark.parametrize("config_rel", ALL_CONFIGS)
def test_dryrun_eval_correct_script_invoked(config_rel):
    result = _run_dry_eval(config_rel)
    out = result.stdout
    expected_thinking, _ = _expected(_load(config_rel))
    if expected_thinking:
        # evaluate_Qwen3_think.py must appear (not just the base script)
        assert "evaluate_Qwen3_think.py" in out, (
            f"Expected evaluate_Qwen3_think.py for {config_rel}\nOutput:\n{out}"
        )
    else:
        # evaluate_Qwen3_think.py must NOT appear (discriminates base from think)
        assert "evaluate_Qwen3_think.py" not in out, (
            f"evaluate_Qwen3_think.py wrongly invoked for a non-thinking config: {config_rel}\nOutput:\n{out}"
        )


@pytest.mark.parametrize("config_rel", ALL_CONFIGS)
def test_dryrun_eval_calc_step_always_runs(config_rel):
    result = _run_dry_eval(config_rel)
    assert "calc.py" in result.stdout, (
        f"calc.py step missing from eval output for {config_rel}\nOutput:\n{result.stdout}"
    )


@pytest.mark.parametrize("config_rel", ALL_CONFIGS)
def test_dryrun_eval_checkpoint_path_in_output(config_rel):
    result = _run_dry_eval(config_rel)
    out = result.stdout
    _, expected_suffix = _expected(_load(config_rel))
    assert expected_suffix in out, (
        f"Expected checkpoint suffix '{expected_suffix}' in eval output for {config_rel}\n"
        f"Output:\n{out}"
    )
