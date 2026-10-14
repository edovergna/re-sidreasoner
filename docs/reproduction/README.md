# Reproduction experiments

Released-checkpoint evaluation on Games, Office, and Industrial, plus independent
SFT and RA training with Qwen3-1.7B on Games.

## R1 — Released-checkpoint replication

Released-checkpoint scores, relative changes, and bootstrap p-values:

| Dataset | Metric | Paper | Ckpt | Δ% | p |
|---|---|---|---|---|---|
| Games | R@5 | 0.0710 | 0.0723 | +1.8 | .72 |
| Games | N@5 | 0.0460 | 0.0453 | -1.6 | .74 |
| Games | R@10 | 0.1031 | 0.1055 | +2.3 | .54 |
| Games | N@10 | 0.0563 | 0.0559 | -0.7 | .85 |
| Office | R@5 | 0.1373 | 0.1363 | -0.8 | .81 |
| Office | N@5 | 0.1119 | 0.1106 | -1.2 | .74 |
| Office | R@10 | 0.1648 | 0.1650 | +0.1 | .96 |
| Office | N@10 | 0.1208 | 0.1199 | -0.8 | .82 |
| Industrial | R@5 | 0.1109 | 0.1099 | -0.9 | .83 |
| Industrial | N@5 | 0.0905 | 0.0900 | -0.6 | .89 |
| Industrial | R@10 | 0.1438 | 0.1482 | +3.1 | .40 |
| Industrial | N@10 | 0.1010 | 0.1023 | +1.3 | .75 |

## R2 — Independent early-stage retraining (Games)

Retrain Qwen3-1.7B on the original S1 multi-task alignment data ablation. Here **S1 names
a data setting**, not all three training stages. Evaluation uses greedy reasoning followed
by constrained beam-10 SID decoding; HR@K is Recall@K in this single-target protocol.

| Stage | HR@5 | HR@10 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|
| SFT (raw align) | 0.0215 | 0.0387 | 0.0129 | 0.0184 |
| RA 5-epoch | 0.0321 | 0.0602 | 0.0187 | 0.0277 |

## Where the code lives

The launch scripts are part of the **SIDReasoner submodule**, materialised by
`patches/common/sidreasoner-local.patch` (same pattern as `rq2_components` /
`rq5_diversity`). After applying the patch they sit under
`baselines/repos/SIDReasoner/final_scripts/rq1_reproduction/`. Run from the submodule
root with `$PYTHON_CMD`; source `final_scripts/cluster_env.sh` first.

| Stage | Script (`final_scripts/rq1_reproduction/`) |
|-------|--------------------------------------------|
| 1.7B S1 alignment SFT | `sft_Qwen3_S1_Games.sh` |
| 1.7B S1 reasoning activation (5-ep) | `sft_reasoning_activation_S1_Games.sh` |
| think-mode eval | `evaluate_Qwen3_think_S1.sh` (+ `_bestof1` for beam-1) |

Cold-start RA saves the final epoch with `load_best_model_at_end=False`.

## Reproduce

```bash
bash scripts/common/apply_submodule_patch.sh
cd baselines/repos/SIDReasoner
source final_scripts/cluster_env.sh

sbatch final_scripts/rq1_reproduction/sft_Qwen3_S1_Games.sh                 # SFT  -> HR@10 0.0387
sbatch final_scripts/rq1_reproduction/sft_reasoning_activation_S1_Games.sh  # RA(5-ep) -> HR@10 0.0602
sbatch final_scripts/rq1_reproduction/evaluate_Qwen3_think_S1.sh            # standard greedy think eval
```

Use `evaluate_Qwen3_think_S1.sh` for the reported greedy-thinking evaluation.

> **Environment:** Transformers **4.51.3**, matching the patched lockfile and
> Appendix D. The dependency constraint is `>=4.51.0,<4.52`.

## Independent retraining checkpoints

Checkpoint download links will be added once the submission is public.

*SIDReasoner-extensions/1.7B-S1/SFT*

*SIDReasoner-extensions/1.7B-S1/RA*
