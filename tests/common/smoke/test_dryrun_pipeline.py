"""Smoke test 2: verify subprocess command generation via --dry-run for all configured variants."""
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from conftest import ALL_CONFIGS, REPO_ROOT


def _load(config_rel: str) -> dict:
    with (REPO_ROOT / config_rel).open() as f:
        return yaml.safe_load(f)


def _run_dry(config_rel: str) -> str:
    """Run train_pipeline.py --dry-run and return stdout (asserts exit 0)."""
    result = subprocess.run(
        [sys.executable, "scripts/common/train/train_pipeline.py", "--config", config_rel, "--dry-run"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"train_pipeline.py --dry-run failed (exit {result.returncode}) for {config_rel}\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
    return result.stdout


@pytest.mark.parametrize("config_rel", ALL_CONFIGS)
def test_dryrun_exits_zero(config_rel):
    _run_dry(config_rel)


@pytest.mark.parametrize("config_rel", ALL_CONFIGS)
def test_enabled_stages_appear_in_output(config_rel):
    cfg = _load(config_rel)
    out = _run_dry(config_rel)
    stages = cfg["stages"]

    if stages["s1"]["enabled"]:
        assert "sft_Qwen3.py" in out, f"sft_Qwen3.py missing from output for {config_rel}"
    if stages["s2"]["enabled"]:
        assert "sft_reasoning_activation.py" in out, (
            f"sft_reasoning_activation.py missing from output for {config_rel}"
        )
    if stages["s3"]["enabled"]:
        assert "verl.trainer.main_ppo" in out, (
            f"verl.trainer.main_ppo missing from output for {config_rel}"
        )
        assert "algorithm.adv_estimator=grpo" in out, (
            f"algorithm.adv_estimator=grpo missing from output for {config_rel}"
        )


@pytest.mark.parametrize("config_rel", ALL_CONFIGS)
def test_disabled_stages_absent_from_output(config_rel):
    cfg = _load(config_rel)
    out = _run_dry(config_rel)
    stages = cfg["stages"]

    if not stages["s2"]["enabled"]:
        assert "sft_reasoning_activation.py" not in out, (
            f"S2 is disabled but sft_reasoning_activation.py appeared in output for {config_rel}"
        )
    if not stages["s3"]["enabled"]:
        assert "verl.trainer.main_ppo" not in out, (
            f"S3 is disabled but verl.trainer.main_ppo appeared in output for {config_rel}"
        )


@pytest.mark.parametrize("config_rel", ALL_CONFIGS)
def test_backbone_and_category_in_commands(config_rel):
    cfg = _load(config_rel)
    out = _run_dry(config_rel)
    # Reward-only runs initialize from RA, not the unfine-tuned backbone.
    if cfg["stages"]["s1"]["enabled"]:
        expected_model = cfg["model"]["backbone"]
    else:
        expected_model = cfg["stages"]["s3"]["start_checkpoint"]
    assert expected_model in out, (
        f"Expected initialization '{expected_model}' not found for {config_rel}"
    )
    assert cfg["experiment"]["dataset_category"] in out, (
        f"Dataset category '{cfg['experiment']['dataset_category']}' not found in output for {config_rel}"
    )


@pytest.mark.parametrize("config_rel", ALL_CONFIGS)
def test_s3_total_epochs_in_command(config_rel):
    cfg = _load(config_rel)
    if not cfg["stages"]["s3"]["enabled"]:
        pytest.skip("S3 disabled")
    out = _run_dry(config_rel)
    expected = f"trainer.total_epochs={cfg['stages']['s3']['total_epochs']}"
    assert expected in out, (
        f"Expected '{expected}' in S3 command output for {config_rel}\n"
        f"Output snippet (last 500 chars):\n{out[-500:]}"
    )


@pytest.mark.parametrize("config_rel", ALL_CONFIGS)
def test_s3_reward_fn_path_in_command(config_rel):
    cfg = _load(config_rel)
    if not cfg["stages"]["s3"]["enabled"]:
        pytest.skip("S3 disabled")
    out = _run_dry(config_rel)
    # The reward_fn path is resolved to absolute; check the filename at minimum
    reward_fn_name = Path(cfg["paths"]["reward_fn"]).name
    assert reward_fn_name in out, (
        f"Reward function '{reward_fn_name}' not found in S3 command for {config_rel}"
    )


@pytest.mark.parametrize("config_rel", ALL_CONFIGS)
def test_s1_output_dir_includes_stage1_sft(config_rel):
    cfg = _load(config_rel)
    if not cfg["stages"]["s1"]["enabled"]:
        pytest.skip("S1 disabled — no stage1_sft output dir")
    out = _run_dry(config_rel)
    assert "stage1_sft" in out, f"Expected 'stage1_sft' in output dir arg for {config_rel}"


@pytest.mark.parametrize("config_rel", ALL_CONFIGS)
def test_summary_printed_without_write_claim(config_rel):
    out = _run_dry(config_rel)
    assert "[SUMMARY]" in out, (
        f"[SUMMARY] line missing from dry-run output for {config_rel}"
    )
    assert "[SUMMARY] Written to" not in out


@pytest.mark.parametrize("config_rel", ALL_CONFIGS)
def test_paper_reward_runs_stop_at_100_steps(config_rel, monkeypatch):
    """The paper budget must apply to direct CLI runs, not only Slurm jobs."""
    monkeypatch.delenv("GRPO_MAX_STEPS_OVERRIDE", raising=False)
    out = _run_dry(config_rel)
    assert "trainer.total_training_steps=100" in out


def test_explicit_step_override_takes_precedence(monkeypatch):
    """Longer follow-up experiments can override the paper's configured cap."""
    monkeypatch.setenv("GRPO_MAX_STEPS_OVERRIDE", "250")
    out = _run_dry("configs/grpo/a0_baseline/games.yaml")
    assert "trainer.total_training_steps=250" in out
    assert "trainer.total_training_steps=100" not in out
