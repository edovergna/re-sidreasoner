"""
Adapter for SIDReasoner (baselines/repos/SIDReasoner).

Maps our unified YAML config schema to the CLI arguments expected by each
training stage and evaluation script. All subprocess calls run with
cwd=repo_root so relative imports inside SIDReasoner resolve correctly.

Environment variables TORCHRUN_CMD and PYTHON_CMD are set by
baselines/repos/SIDReasoner/scripts/cluster_env.sh, which the sbatch job
sources before invoking this module. Defaults fall back to plain torchrun/python.
"""

import json
import os
import subprocess
import time
from pathlib import Path


class SIDReasonerAdapter:
    def __init__(self, config: dict, repo_root: Path, dry_run: bool = False):
        self.config = config
        self.repo_root = Path(repo_root).resolve()
        self.dry_run = dry_run
        self.output_root = Path(config["paths"]["output_root"]).resolve()
        self._torchrun = os.environ.get("TORCHRUN_CMD", "torchrun")
        self._python = os.environ.get("PYTHON_CMD", "python")
        self._timings: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run_pipeline(self) -> Path:
        """Orchestrate enabled stages in order S1 → S2 → S3."""
        stages = self.config["stages"]
        ckpt: Path | None = None

        if stages["s1"]["enabled"]:
            ckpt = self._timed("s1", self.run_s1)()
        if stages["s2"]["enabled"]:
            assert ckpt is not None, "S2 requires S1 checkpoint"
            ckpt = self._timed("s2", self.run_s2)(ckpt)
        if stages["s3"]["enabled"]:
            if ckpt is None:
                # Support S3-only runs by pointing at a pre-existing RA checkpoint.
                start = stages["s3"].get("start_checkpoint")
                assert start is not None, (
                    "S3 needs a checkpoint. Either enable S1+S2 or set "
                    "stages.s3.start_checkpoint in the config."
                )
                ckpt = Path(start).resolve()
            ckpt = self._timed("s3", self.run_s3)(ckpt)

        self._write_summary(ckpt)
        return ckpt

    def run_s1(self) -> Path:
        """Stage 1: multi-task SFT alignment (sft_Qwen3.py)."""
        cfg = self.config["stages"]["s1"]
        output_dir = self.output_root / "stage1_sft"
        # Per-source enrichment gating. Absent block => all enabled (preserves the
        # behaviour of the original ablations, which always feed all three files).
        # A disabled source is passed the literal "None", which Fire turns into a
        # Python None so sft_Qwen3.py's `if <path> is not None:` checks skip it.
        # (Omitting the flag is unsafe: the script defaults to hardcoded Office paths.)
        enrich = cfg.get("enrichment", {"items": True, "seqs": True, "general": True})
        cmd = [
            self._torchrun,
            "--nproc_per_node", str(cfg.get("nproc_per_node", 4)),
            "--master_port", "12340",
            "sft_Qwen3.py",
            "--base_model",                  self.config["model"]["backbone"],
            "--train_file",                  self._p("train_file"),
            "--eval_file",                   self._p("eval_file"),
            "--output_dir",                  str(output_dir),
            "--num_epochs",                  str(cfg["num_epochs"]),
            "--batch_size",                  str(cfg.get("batch_size", 1024)),
            "--micro_batch_size",            str(cfg.get("micro_batch_size", 1)),
            "--learning_rate",               str(cfg.get("learning_rate", 3e-4)),
            "--warmup_steps",                str(cfg.get("warmup_steps", 20)),
            "--sid_index_path",              self._p("sid_index"),
            "--item_meta_path",              self._p("item_meta"),
            "--llm_generated_data_path",     self._p("llm_items") if enrich.get("items", True) else "None",
            "--llm_generated_sequence_path", self._p("llm_seqs") if enrich.get("seqs", True) else "None",
            "--general_reasoning_path",      self._p("general_reasoning") if enrich.get("general", True) else "None",
            "--category",                    self.config["experiment"]["dataset_category"],
            "--wandb_project",               "SIDReasoner_Ablations",
            "--wandb_run_name",              f"{self.config['experiment']['name']}_s1",
            "--seed",                        "42",
        ]
        self._run(cmd)
        return output_dir / "final_checkpoint"

    def run_s2(self, s1_checkpoint: Path) -> Path:
        """Stage 2: reasoning-activation fine-tuning (sft_reasoning_activation.py)."""
        cfg = self.config["stages"]["s2"]
        output_dir = self.output_root / "stage2_reasoning_activation"
        cmd = [
            self._torchrun,
            "--nproc_per_node", str(cfg.get("nproc_per_node", 4)),
            "--master_port", "12341",
            "sft_reasoning_activation.py",
            "--base_model",               str(s1_checkpoint),
            "--train_file",               self._p("train_file"),
            "--eval_file",                self._p("eval_file"),
            "--output_dir",               str(output_dir),
            "--num_epochs",               str(cfg["num_epochs"]),
            "--batch_size",               str(cfg.get("batch_size", 1024)),
            "--micro_batch_size",         str(cfg.get("micro_batch_size", 8)),
            "--learning_rate",            str(cfg.get("learning_rate", 1e-5)),
            "--sid_index_path",           self._p("sid_index"),
            "--item_meta_path",           self._p("item_meta"),
            "--reasoning_train_file",     self._p("llm_seqs"),
            "--category",                 self.config["experiment"]["dataset_category"],
            "--wandb_project",            "SIDReasoner_Ablations",
            "--wandb_run_name",           f"{self.config['experiment']['name']}_s2",
            "--seed",                     "42",
        ]
        self._run(cmd)
        return output_dir / "final_checkpoint"

    def run_s3(self, checkpoint: Path) -> Path:
        """Stage 3: GRPO reinforcement learning (verl.trainer.main_ppo)."""
        cfg = self.config["stages"]["s3"]
        output_dir = self.output_root / "stage3_rl"
        cmd = [
            self._python, "-m", "verl.trainer.main_ppo",
            "algorithm.adv_estimator=grpo",
            f"data.train_files={self._p('rl_train_parquet')}",
            f"data.val_files={self._p('rl_val_parquet')}",
            f"actor_rollout_ref.model.path={checkpoint}",
            f"trainer.total_epochs={os.environ.get('GRPO_EPOCHS_OVERRIDE', cfg['total_epochs'])}",
            f"custom_reward_function.path={self._p('reward_fn')}",
            "custom_reward_function.name=rule_base_reward",
            f"trainer.default_local_dir={output_dir}",
            f"trainer.experiment_name={self.config['experiment']['name']}_s3",
            f"trainer.n_gpus_per_node={cfg.get('n_gpus_per_node', 4)}",
            "trainer.nnodes=1",
            "data.train_batch_size=256",
            "data.max_prompt_length=1024",
            "data.max_response_length=1024",
            "data.filter_overlong_prompts=True",
            "data.truncation=error",
            # learning rate sweep
            f"actor_rollout_ref.actor.optim.lr={cfg.get('lr', 5e-7)}",
            "actor_rollout_ref.model.use_remove_padding=True",
            "actor_rollout_ref.actor.ppo_mini_batch_size=256",
            "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8",
            # KL as loss term (True) vs. folded into reward (False)
            f"actor_rollout_ref.actor.use_kl_loss={cfg.get('use_kl_loss', True)}",
            # KL coefficient sweep
            f"actor_rollout_ref.actor.kl_loss_coef={cfg.get('kl_loss_coef', 0.001)}",
            # KL estimator type {kl, mse, low_var_kl}
            f"actor_rollout_ref.actor.kl_loss_type={cfg.get('kl_loss_type', 'low_var_kl')}",
            # entropy bonus (0 = disabled)
            f"actor_rollout_ref.actor.entropy_coeff={cfg.get('entropy_coeff', 0)}",
            "actor_rollout_ref.model.enable_gradient_checkpointing=True",
            "actor_rollout_ref.actor.fsdp_config.param_offload=False",
            "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",
            "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8",
            "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
            "actor_rollout_ref.rollout.name=vllm",
            "actor_rollout_ref.rollout.gpu_memory_utilization=0.8",
            # group size G (samples per prompt)
            f"actor_rollout_ref.rollout.n={cfg.get('rollout_n', 16)}",
            "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8",
            "actor_rollout_ref.ref.fsdp_config.param_offload=True",
            # KL in reward vs. in loss; clip ratio
            f"algorithm.use_kl_in_reward={cfg.get('use_kl_in_reward', False)}",
            f"actor_rollout_ref.actor.clip_ratio={cfg.get('clip_ratio', 0.2)}",
            # advantage normalisation (divide by std or just subtract mean)
            f"algorithm.norm_adv_by_std_in_grpo={cfg.get('norm_adv_by_std', True)}",
            "trainer.critic_warmup=0",
            "trainer.logger=['console','wandb']",
            "trainer.project_name=SIDReasoner_Ablations",
            "trainer.save_freq=100",
            "trainer.test_freq=50",
        ]
        # Configured hard cap on optimizer steps (paper runs: 100). When set, verl
        # stops at this many steps regardless of total_epochs, and treats it as
        # the last step (so a checkpoint + final eval are still produced).
        max_steps = os.environ.get("GRPO_MAX_STEPS_OVERRIDE", cfg.get("max_steps"))
        if max_steps:
            cmd.append(f"trainer.total_training_steps={max_steps}")
        self._run(cmd)
        return output_dir

    # ------------------------------------------------------------------
    # Evaluation helpers (called by scripts/eval/evaluate.py)
    # ------------------------------------------------------------------

    def run_eval_base(self, checkpoint: Path, result_json: Path) -> None:
        """Beam-search evaluation for S1-only conditions (no thinking)."""
        cmd = [
            self._python, "evaluate_Qwen3.py",
            "--base_model",      str(checkpoint),
            "--train_file",      self._p("train_file"),
            "--info_file",       self._p("info_file"),
            "--category",        self.config["experiment"]["dataset_category"],
            "--test_data_path",  self._p("test_file"),
            "--result_json_data", str(result_json),
        ]
        self._run(cmd)

    def run_eval_think(self, checkpoint: Path, result_json: Path) -> None:
        """vLLM-based thinking-mode evaluation for S2/S3 conditions."""
        cmd = [
            self._python, "evaluate_Qwen3_think.py",
            "--base_model",      str(checkpoint),
            "--info_file",       self._p("info_file"),
            "--category",        self.config["experiment"]["dataset_category"],
            "--test_data_path",  self._p("test_file"),
            "--item_file",       self._p("item_meta"),
            "--index_file",      self._p("sid_index"),
            "--result_json_data", str(result_json),
        ]
        self._run(cmd)

    def run_calc(self, result_json: Path) -> None:
        """Compute NDCG/HR metrics from a prediction JSON."""
        cmd = [
            self._python, "calc.py",
            str(result_json),
            self._p("info_file"),
        ]
        self._run(cmd)

    # ------------------------------------------------------------------
    # S3 (RL) checkpoint resolution
    # ------------------------------------------------------------------

    def resolve_rl_checkpoint(self, rl_root: Path) -> Path:
        """Resolve an evaluable HF checkpoint from a verl RL output directory.

        verl/GRPO saves *sharded* FSDP actor weights under
        ``<rl_root>/global_step_<N>/actor``. The evaluation scripts load via
        Hugging Face / vLLM and cannot read those shards directly, so we merge
        the latest checkpoint into a single HF model with
        ``scripts/merge_fsdp_checkpoint.py`` and return that directory.
        """
        rl_root = Path(rl_root)
        if self.dry_run:
            # No real checkpoints exist during a dry run; show intended commands.
            actor = rl_root / "global_step_<N>" / "actor"
            return self.merge_fsdp_checkpoint(actor)
        actor = self.find_latest_rl_actor(rl_root)
        return self.merge_fsdp_checkpoint(actor)

    def find_latest_rl_actor(self, rl_root: Path) -> Path:
        """Return the ``actor`` dir of the highest ``global_step_<N>`` under rl_root."""
        rl_root = Path(rl_root)
        steps = [p for p in rl_root.glob("global_step_*") if p.is_dir()]
        if not steps:
            raise FileNotFoundError(
                f"No global_step_* checkpoints found under {rl_root}. "
                "S3 RL training must complete and save at least one checkpoint "
                "(check trainer.save_freq against the number of training steps)."
            )
        latest = max(steps, key=lambda p: int(p.name.rsplit("_", 1)[-1]))
        actor = latest / "actor"
        if not actor.is_dir():
            raise FileNotFoundError(f"Expected an 'actor' directory under {latest}")
        return actor

    def merge_fsdp_checkpoint(self, actor_dir: Path) -> Path:
        """Merge a verl FSDP actor checkpoint into a single HF model dir; return it."""
        actor_dir = Path(actor_dir)
        merged_dir = actor_dir.parent / "actor_merged"
        if not self.dry_run and (merged_dir / "config.json").exists():
            print(f"[merge] Reusing existing merged checkpoint: {merged_dir}")
            return merged_dir
        cmd = [
            self._python, "scripts/merge_fsdp_checkpoint.py",
            "--checkpoint", str(actor_dir),
            "--output-dir", str(merged_dir),
        ]
        self._run(cmd)
        return merged_dir

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _p(self, key: str) -> str:
        """Resolve a path key from config to an absolute path string."""
        raw = self.config["paths"][key]
        return str(Path(raw).resolve())

    def _run(self, cmd: list) -> None:
        str_cmd = " ".join(str(c) for c in cmd)
        if self.dry_run:
            print(f"[DRY RUN] (cwd={self.repo_root})\n  {str_cmd}\n")
            return
        print(f"[RUN] {str_cmd}")
        subprocess.run([str(c) for c in cmd], cwd=self.repo_root, check=True)

    def _timed(self, stage: str, fn):
        def wrapper(*args, **kwargs):
            t0 = time.time()
            result = fn(*args, **kwargs)
            self._timings[stage] = round(time.time() - t0, 1)
            return result
        return wrapper

    def _write_summary(self, final_checkpoint: Path | None) -> None:
        summary = {
            "experiment": self.config["experiment"]["name"],
            "final_checkpoint": str(final_checkpoint) if final_checkpoint else None,
            "stage_durations_s": self._timings,
            "dry_run": self.dry_run,
        }
        if self.dry_run:
            print(f"[SUMMARY] {json.dumps(summary, indent=2)}")
            return
        out = self.output_root / "pipeline_summary.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2))
        print(f"[SUMMARY] Written to {out}")
