# GRPO reward variants

Six Games runs start from the released Qwen3-1.7B RA checkpoint and train for
100 GRPO steps. SFT and RA are disabled in these configs.

## The baseline reward (A0)

The released reward is a sum of an accuracy term and a format term:

```math
R = R_{\text{sr}} + \lambda \cdot R_{\text{f}}, \qquad \lambda = 0.1
```

The accuracy term is hierarchical (coarse-to-fine), doubling at each matched level:

```math
R_{\text{sr}} =
\begin{cases}
1.00 & \text{if } a_1 = g_1,\ a_2 = g_2,\ a_3 = g_3 \\
0.50 & \text{if } a_1 = g_1,\ a_2 = g_2,\ a_3 \neq g_3 \\
0.25 & \text{if } a_1 = g_1,\ a_2 \neq g_2 \\
0 & \text{otherwise}
\end{cases}
\qquad R_{\text{sr}} \in \{0,\ 0.25,\ 0.50,\ 1.00\}
```

```math
R_{\text{f}} =
\begin{cases}
1 & \text{if } (a_1, a_2, a_3) \text{ is a valid path in the SID catalogue trie} \\
0 & \text{otherwise}
\end{cases}
```

$a = (a_1, a_2, a_3)$ is the predicted SID, $g$ the ground truth. The GRPO advantage normalizes
rewards **within each group** of $G = 16$ rollouts:

```math
\hat{A}_i = \frac{R_i - \text{mean}(R)}{\text{std}(R) + \epsilon}
```

## The experimental axis

Each variant is a **single controlled change** to the reward formula above; all GRPO objective
hyperparameters are held fixed.

| Variant | Reward | Description |
|---------|--------------|-------|
| **A0** baseline | $R = R_{\text{sr}} + 0.1 \cdot R_{\text{f}}$ (original formula). | Reference bar. |
| **A1** no-format | $R = R_{\text{sr}}$ ($\lambda = 0$; format term removed). | Format-bonus ablation. |
| **A2** $\lambda$-sweep | $R = R_{\text{sr}} + \lambda \cdot R_{\text{f}}$, $\lambda \in \{0.05,\ 0.5,\ 1.0\}$. | Format-weight sweep. |
| **A3** linear acc | $R_{\text{sr}} = 0.33 \cdot [a_1 = g_1] + 0.33 \cdot [a_1 = g_1,\ a_2 = g_2] + 0.34 \cdot [\text{all match}]$, plus $0.1 \cdot R_{\text{f}}$. | Prefix-gated additive credit. |

## Results

All six trained checkpoints (A0, A1, A2$\times$3, A3) evaluated with the think-mode ranking eval
(`scripts/common/eval/evaluate.py → run_eval_think`) on the full Games held-out test set
($\sim$6.1k examples, `n_beam=10`, vLLM, `global_step_100` merged actor). Slurm job 23816638.

Results from the reported step-100 evaluation:

### NDCG@k

| Experiment | Reward | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|
| **A0** baseline   | $R_{\text{sr}} + 0.1 \cdot R_{\text{f}}$  | 0.0216 | 0.0380 | 0.0438 | 0.0543 |
| **A1** no-format  | $R_{\text{sr}}$ ($\lambda = 0$)             | 0.0308 | 0.0428 | **0.0486** | **0.0588** |
| **A2** $\lambda = 0.05$ | $+\,0.05 \cdot R_{\text{f}}$         | 0.0213 | 0.0376 | 0.0439 | 0.0544 |
| **A2** $\lambda = 0.5$  | $+\,0.5 \cdot R_{\text{f}}$          | 0.0226 | 0.0379 | 0.0438 | 0.0537 |
| **A2** $\lambda = 1.0$  | $+\,1.0 \cdot R_{\text{f}}$          | 0.0252 | 0.0399 | 0.0464 | 0.0564 |
| **A3** linear acc | linear $R_{\text{sr}} + 0.1 \cdot R_{\text{f}}$ | 0.0290 | 0.0403 | 0.0470 | 0.0571 |

### HR@k

| Experiment | Reward | HR@1 | HR@3 | HR@5 | HR@10 |
|---|---|---|---|---|---|
| **A0** baseline   | $R_{\text{sr}} + 0.1 \cdot R_{\text{f}}$  | 0.0216 | 0.0506 | 0.0648 | 0.0977 |
| **A1** no-format  | $R_{\text{sr}}$ ($\lambda = 0$)             | 0.0308 | 0.0516 | 0.0658 | 0.0972 |
| **A2** $\lambda = 0.05$ | $+\,0.05 \cdot R_{\text{f}}$         | 0.0213 | 0.0506 | 0.0659 | **0.0987** |
| **A2** $\lambda = 0.5$  | $+\,0.5 \cdot R_{\text{f}}$          | 0.0226 | 0.0488 | 0.0632 | 0.0941 |
| **A2** $\lambda = 1.0$  | $+\,1.0 \cdot R_{\text{f}}$          | 0.0252 | 0.0508 | **0.0667** | 0.0978 |
| **A3** linear acc | linear $R_{\text{sr}} + 0.1 \cdot R_{\text{f}}$ | 0.0290 | 0.0487 | 0.0650 | 0.0965 |

## Checkpoints

Checkpoint download links will be added once the submission is public.

The six actors are merged step-100 checkpoints:

*grpo-a0-baseline-games*

*grpo-a1-no-format-games*

*grpo-a2-l005-games*

*grpo-a2-l05-games*

*grpo-a2-l10-games*

*grpo-a3-linear-acc-games*

The optional λ=0.2 configuration has no reported trained checkpoint.

## Reference files

| Area | Path |
|------|------|
| Source code | [`../../src/grpo/`](../../src/grpo/) |
| Configs | [`../../configs/grpo/`](../../configs/grpo/) |
| Jobs | [`../../jobs/grpo/`](../../jobs/grpo/) |
| Training | [`train_pipeline.py`](../../scripts/common/train/train_pipeline.py) |
| Evaluation | [`evaluate.py`](../../scripts/common/eval/evaluate.py) |
| Tests | [`../../tests/grpo/`](../../tests/grpo/) |
| Checkpoints | See [checkpoint names](#checkpoints). |
| Results | Unpopulated |

## Reproduce

The variants start from the RA checkpoint; SFT and RA are disabled in these configs.

**Inspect a single variant without training (omit `--dry-run` to train):**
```bash
python scripts/common/train/train_pipeline.py \
    --config configs/grpo/a1_no_format/games.yaml \
    --dry-run
```

**Run a single variant (Slurm cluster):**
```bash
sbatch --export=ALL,CONFIG=configs/grpo/a1_no_format/games.yaml jobs/grpo/reward_sweep.job
```

**Run the six paper variants sequentially (Slurm cluster):**
```bash
sbatch jobs/grpo/reward_sweep.job
```

The paper configs cap training at **100 optimizer steps**, including direct Python runs.
`GRPO_MAX_STEPS_OVERRIDE` can override the cap for additional experiments. To run the
unreported λ=0.2 condition, pass `CONFIG=configs/grpo/a2_lambda_sweep/games_l02.yaml`.

**Evaluation:** Use `scripts/common/eval/evaluate.py` (think-mode eval, `run_eval_think`) on the merged S3 actor checkpoint.

## Implementation Details

The reward implementation is in the outer repository:

- **Logic:** `src/grpo/reward_variants.py` contains all A0–A3 reward classes.
- **Shims:** `src/grpo/reward_shims/` contains thin files that import from `reward_variants` and expose `rule_base_reward` to the `verl` framework.
- **Adapter:** `baselines/adapters/sid_reasoner.py` was extended to read `stages.s3.start_checkpoint` to inject the RA checkpoint directly into GRPO, and to map GRPO objective knobs from the YAML configs to `verl` arguments.
