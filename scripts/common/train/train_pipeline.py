"""
Entry point for ablation training runs.

Usage:
    python scripts/common/train/train_pipeline.py --config configs/grpo/a0_baseline/games.yaml
    python scripts/common/train/train_pipeline.py --config ... --dry-run   # print commands only

Must be run from the repo root so relative paths in the YAML resolve correctly.
The sbatch job sources cluster_env.sh before calling this script, which sets
TORCHRUN_CMD and PYTHON_CMD in the environment.
"""

import argparse
import sys
from pathlib import Path

import yaml


def _repo_root() -> Path:
    """Walk up to the repo root (depth-independent; runs before `src` is importable)."""
    here = Path(__file__).resolve()
    for c in (here, *here.parents):
        if (c / ".gitmodules").is_file() and (c / "baselines").is_dir():
            return c
    raise FileNotFoundError("repo root (.gitmodules + baselines/) not found")


# Allow importing the adapter from the repo root
REPO_ROOT = _repo_root()
sys.path.insert(0, str(REPO_ROOT))
from baselines.adapters.sid_reasoner import SIDReasonerAdapter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a SIDReasoner ablation pipeline.")
    p.add_argument("--config", required=True, type=Path, help="Path to ablation YAML config.")
    p.add_argument("--repo-root", type=Path,
                   default=REPO_ROOT / "baselines" / "repos" / "SIDReasoner",
                   help="Path to the SIDReasoner submodule root.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print subprocess commands without executing them.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    config_path = args.config.resolve()
    if not config_path.exists():
        sys.exit(f"Config not found: {config_path}")

    with config_path.open() as f:
        config = yaml.safe_load(f)

    print(f"[train_pipeline] Experiment: {config['experiment']['name']}")
    print(f"[train_pipeline] Config:     {config_path}")
    print(f"[train_pipeline] Repo root:  {args.repo_root}")
    if args.dry_run:
        print("[train_pipeline] DRY RUN — no commands will execute.\n")

    adapter = SIDReasonerAdapter(
        config=config,
        repo_root=args.repo_root,
        dry_run=args.dry_run,
    )
    final_ckpt = adapter.run_pipeline()

    if not args.dry_run:
        print(f"\n[train_pipeline] Done. Final checkpoint: {final_ckpt}")


if __name__ == "__main__":
    main()
