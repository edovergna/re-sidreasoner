import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import yaml

from baselines.adapters.sid_reasoner import SIDReasonerAdapter

ROOT = Path(__file__).resolve().parents[2]


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dry_training_preserves_existing_summary(tmp_path):
    summary = tmp_path / "pipeline_summary.json"
    summary.write_text('{"real_run": true}')
    config = {"experiment": {"name": "test"}, "paths": {"output_root": str(tmp_path)},
              "stages": {stage: {"enabled": False} for stage in ("s1", "s2", "s3")}}
    SIDReasonerAdapter(config, ROOT, dry_run=True).run_pipeline()
    assert json.loads(summary.read_text()) == {"real_run": True}


def test_dry_eval_resolves_override_without_creating_outputs(tmp_path):
    config = yaml.safe_load((ROOT / "configs/grpo/a0_baseline/games.yaml").read_text())
    output = tmp_path / "new-results"
    config["paths"]["output_root"] = str(output)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/common/eval/evaluate.py"),
         "--config", str(config_path), "--checkpoint", "my-checkpoint", "--dry-run"],
        cwd=tmp_path, capture_output=True, text=True, check=True,
    )
    assert f"--base_model {tmp_path / 'my-checkpoint'}" in result.stdout
    assert not output.exists()


def test_environment_normalization_preserves_data_and_setup(tmp_path):
    module = load_script("normalize_env", "scripts/common/normalize_baseline_env.py")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "example_env.sh").write_text('PYTHON_CMD=python\nTORCHRUN_CMD=torchrun\n')
    (scripts / "setup_env.sh").write_text('echo example setup\n')
    launcher = scripts / "train.sh"
    launcher.write_text('source scripts/example_env.sh\n')
    for directory in ("data", ".venv", "checkpoints"):
        folder = tmp_path / directory
        folder.mkdir()
        (folder / "keep.py").write_text('example\n')
    module.normalize(tmp_path)
    module.normalize(tmp_path)
    assert (scripts / "cluster_env.sh").exists()
    assert (scripts / "setup_env.sh").exists()
    assert launcher.read_text() == 'source scripts/cluster_env.sh\n'
    for directory in ("data", ".venv", "checkpoints"):
        assert (tmp_path / directory / "keep.py").read_text() == 'example\n'
