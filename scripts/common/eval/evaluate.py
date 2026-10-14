"""
Evaluate a trained checkpoint and write metrics to results/.

Usage:
    python scripts/common/eval/evaluate.py --config configs/grpo/a0_baseline/games.yaml
    python scripts/common/eval/evaluate.py --config ... --checkpoint path/to/ckpt  # override ckpt
    python scripts/common/eval/evaluate.py --config ... --dry-run

Selects evaluation mode automatically:
    - alignment_only experiments   → evaluate_Qwen3.py (beam search, no thinking)
    - all other experiments        → evaluate_Qwen3_think.py (vLLM, thinking tokens)
"""

import argparse
import json
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


REPO_ROOT = _repo_root()
sys.path.insert(0, str(REPO_ROOT))
from baselines.adapters.sid_reasoner import SIDReasonerAdapter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a SIDReasoner ablation checkpoint.")
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Override checkpoint path (default: inferred from output_root).")
    p.add_argument("--repo-root", type=Path,
                   default=REPO_ROOT / "baselines" / "repos" / "SIDReasoner")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def infer_checkpoint(config: dict) -> Path:
    """Return the expected final checkpoint path based on enabled stages.

    For S3 (RL) this returns the verl output *root* (stage3_rl); the actual
    evaluable model is resolved/merged at run time by
    SIDReasonerAdapter.resolve_rl_checkpoint, which picks the latest
    global_step_<N>/actor and merges its FSDP shards into a single HF model.
    """
    output_root = Path(config["paths"]["output_root"]).resolve()
    stages = config["stages"]
    if stages["s3"]["enabled"]:
        return output_root / "stage3_rl"
    if stages["s2"]["enabled"]:
        return output_root / "stage2_reasoning_activation" / "final_checkpoint"
    return output_root / "stage1_sft" / "final_checkpoint"


def uses_thinking(config: dict) -> bool:
    stages = config["stages"]
    return stages["s2"]["enabled"] or stages["s3"]["enabled"]


def main() -> None:
    args = parse_args()

    config_path = args.config.resolve()
    if not config_path.exists():
        sys.exit(f"Config not found: {config_path}")

    with config_path.open() as f:
        config = yaml.safe_load(f)

    checkpoint = args.checkpoint.resolve() if args.checkpoint is not None else infer_checkpoint(config)
    output_root = Path(config["paths"]["output_root"]).resolve()
    result_json = output_root / "eval_predictions.json"

    print(f"[evaluate] Experiment:  {config['experiment']['name']}")
    print(f"[evaluate] Checkpoint:  {checkpoint}")
    print(f"[evaluate] Result JSON: {result_json}")
    print(f"[evaluate] Think mode:  {uses_thinking(config)}")

    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)

    adapter = SIDReasonerAdapter(config=config, repo_root=args.repo_root, dry_run=args.dry_run)

    if uses_thinking(config):
        # S3 (RL) writes sharded FSDP checkpoints under stage3_rl/global_step_N/actor.
        # Merge the latest into a single HF model before vLLM/HF can load it.
        if config["stages"]["s3"]["enabled"] and args.checkpoint is None:
            checkpoint = adapter.resolve_rl_checkpoint(checkpoint)
            print(f"[evaluate] Resolved RL checkpoint: {checkpoint}")
        adapter.run_eval_think(checkpoint, result_json)
    else:
        adapter.run_eval_base(checkpoint, result_json)

    adapter.run_calc(result_json)

    if not args.dry_run:
        print(f"[evaluate] Done. Metrics printed above; predictions at {result_json}")


if __name__ == "__main__":
    main()
