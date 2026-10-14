"""Smoke test 1: verify all YAML-referenced paths resolve to existing files/directories."""
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from conftest import ALL_CONFIGS, REPO_ROOT

REQUIRED_TOP_KEYS = ["experiment", "model", "stages", "paths"]

ALWAYS_REQUIRED_PATH_KEYS = [
    "train_file",
    "eval_file",
    "test_file",
    "info_file",
    "item_meta",
    "sid_index",
    "llm_items",
    "llm_seqs",
    "general_reasoning",
]

S3_REQUIRED_PATH_KEYS = ["rl_train_parquet", "rl_val_parquet", "reward_fn"]


def _load(config_rel: str) -> dict:
    with (REPO_ROOT / config_rel).open() as f:
        return yaml.safe_load(f)


@pytest.mark.parametrize("config_rel", ALL_CONFIGS)
def test_config_file_exists(config_rel):
    assert (REPO_ROOT / config_rel).exists(), f"Config file missing: {REPO_ROOT / config_rel}"


@pytest.mark.parametrize("config_rel", ALL_CONFIGS)
def test_yaml_top_level_keys(config_rel):
    cfg = _load(config_rel)
    for key in REQUIRED_TOP_KEYS:
        assert key in cfg, f"Missing top-level key '{key}' in {config_rel}"


@pytest.mark.parametrize("config_rel", ALL_CONFIGS)
def test_sid_reasoner_root_is_directory(config_rel):
    cfg = _load(config_rel)
    root = REPO_ROOT / cfg["paths"]["sid_reasoner_root"]
    assert root.is_dir(), f"sid_reasoner_root not a directory: {root}"


@pytest.mark.parametrize("config_rel", ALL_CONFIGS)
def test_always_required_data_files_exist(config_rel):
    cfg = _load(config_rel)
    paths = cfg["paths"]
    missing = []
    for key in ALWAYS_REQUIRED_PATH_KEYS:
        if key not in paths:
            missing.append(f"YAML key absent: {key}")
            continue
        p = REPO_ROOT / paths[key]
        if not p.exists():
            missing.append(f"{key} → {p}")
    assert not missing, f"Missing paths in {config_rel}:\n  " + "\n  ".join(missing)


@pytest.mark.parametrize("config_rel", ALL_CONFIGS)
def test_s3_paths_exist_when_s3_enabled(config_rel):
    cfg = _load(config_rel)
    if not cfg["stages"]["s3"]["enabled"]:
        pytest.skip("S3 disabled — skipping S3-specific path checks")
    paths = cfg["paths"]
    missing = []
    for key in S3_REQUIRED_PATH_KEYS:
        if key not in paths:
            missing.append(f"YAML key absent: {key}")
            continue
        p = REPO_ROOT / paths[key]
        if not p.exists():
            missing.append(f"{key} → {p}")
    assert not missing, f"Missing S3 paths in {config_rel}:\n  " + "\n  ".join(missing)


@pytest.mark.parametrize("config_rel", ALL_CONFIGS)
def test_enabled_stage_has_required_keys(config_rel):
    cfg = _load(config_rel)
    stages = cfg["stages"]
    s1_keys = ["num_epochs", "batch_size", "micro_batch_size", "learning_rate", "nproc_per_node"]
    s2_keys = ["num_epochs", "batch_size", "micro_batch_size", "learning_rate", "nproc_per_node"]
    s3_keys = ["total_epochs", "n_gpus_per_node"]

    if stages["s1"]["enabled"]:
        for k in s1_keys:
            assert k in stages["s1"], f"S1 missing '{k}' in {config_rel}"
    if stages["s2"]["enabled"]:
        for k in s2_keys:
            assert k in stages["s2"], f"S2 missing '{k}' in {config_rel}"
    if stages["s3"]["enabled"]:
        for k in s3_keys:
            assert k in stages["s3"], f"S3 missing '{k}' in {config_rel}"


def test_experiment_names_are_unique():
    names = [_load(c)["experiment"]["name"] for c in ALL_CONFIGS]
    assert len(names) == len(set(names)), f"Duplicate experiment names found: {names}"


@pytest.mark.parametrize("config_rel", ALL_CONFIGS)
def test_total_stage_epochs_matches_declared_total(config_rel):
    cfg = _load(config_rel)
    stages = cfg["stages"]
    if "total_epochs" not in cfg["experiment"]:
        pytest.skip("experiment.total_epochs not declared (e.g. S3-only configs)")
    declared = cfg["experiment"]["total_epochs"]
    s1 = stages["s1"]["num_epochs"] if stages["s1"]["enabled"] else 0
    s2 = stages["s2"]["num_epochs"] if stages["s2"]["enabled"] else 0
    s3 = stages["s3"]["total_epochs"] if stages["s3"]["enabled"] else 0
    assert s1 + s2 + s3 == declared, (
        f"Epoch budget mismatch in {config_rel}: "
        f"S1={s1} + S2={s2} + S3={s3} = {s1+s2+s3}, declared={declared}"
    )
