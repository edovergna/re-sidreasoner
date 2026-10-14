# Component experiments

Qwen3-1.7B continuation and prompt ablations on Games, Office, and Industrial.

## Pipeline context

SIDReasoner trains in three stages:

| Stage | Name | Purpose | Checkpoint |
|-------|------|---------|------------|
| S1 | SFT | Teach the model to emit valid Semantic IDs (SIDs). | `sidreasoner_sft_*` |
| S2 | Reasoning-activation (RA) | Generate a rationale inside `<think>…</think>`, followed by the SID. | `sidreasoner_ra_*` |
| S3 | GRPO RL | Reinforce the policy so reasoning leads to correct recommendations. | post-GRPO |

The continuation controls start from the released RA checkpoint. Inference ablations
evaluate both RA and post-GRPO checkpoints.

Backbone: `Qwen/Qwen3-1.7B` (the original paper's setting). Datasets: `Video Games`,
`Office Products`, `Industrial & Scientific`.

## Experiments

### C1 — Supervised continuations

From the RA checkpoint, train two non-RL continuations and compare them to the actual
post-GRPO checkpoint, with deltas measured relative to RA on the same dataset:

- **Extra SFT** — continue supervised fine-tuning from RA for three epochs.
- **Extra RA** — continue reasoning-activation training from RA for ten epochs.
- **RA + RL** — original authors’ released post-GRPO checkpoint.

Both controls select the best epoch by validation loss (Appendix B).
The launcher sets `LOAD_BEST_MODEL_AT_END=True` for extra SFT and extra RA.
Extra SFT retains its epoch checkpoints for diagnostics; the main evaluation uses
its `final_checkpoint`, containing the restored best model. This selection applies to
continuations from an already activated model, not cold-start RA.

Relative percentage changes are measured against the RA checkpoint.

| Dataset | Checkpoint | R@10 | NDCG@10 | ΔR@10 (%) | ΔNDCG@10 (%) |
|---|---|---|---|---|---|
| Games | RA | 0.0876 | 0.0474 | — | — |
| Games | Extra SFT | 0.0511 | 0.0294 | -41.7 | -38.0 |
| Games | Extra RA | 0.0440 | 0.0255 | -49.8 | -46.2 |
| Games | **RA + RL** | **0.1055** | **0.0559** | **+20.4** | **+17.9** |
| Office | RA | 0.1424 | 0.1023 | — | — |
| Office | Extra SFT | 0.1385 | 0.0995 | -2.7 | -2.7 |
| Office | Extra RA | 0.1424 | 0.1028 | +0.0 | +0.5 |
| Office | **RA + RL** | **0.1650** | **0.1199** | **+15.9** | **+17.2** |

Checkpoint names for the accompanying artifact package:

Checkpoint download links will be added once the submission is public.

*Video_Games_extra_sft_Qwen3-1.7B*

*Video_Games_extra_ra_Qwen3-1.7B*

*Office_Products_extra_sft_Qwen3-1.7B*

*Office_Products_extra_ra_Qwen3-1.7B*

### C2 — Pre-RL decoding

Decoding results for the Video Games RA checkpoint:

| Checkpoint | Decoder | R@10 | NDCG@10 |
|---|---|---|---|
| RA | beam10 | 0.0876 | 0.0474 |
| RA | beam20 | 0.0842 | — |
| RA | beam50 | 0.0822 | — |
| Post-GRPO | beam10 | 0.1055 | 0.0559 |

### C3 — Thinking and no-think evaluation

Strict **no-think** ablation: the evaluator removes both the instruction to reason and the
`<think>…</think>` scaffold, then re-runs the *same* checkpoints. Deltas are computed as
**no-think $-$ thinking** from the displayed scores, so a negative value means
thinking was better.

| Dataset | Checkpoint | T R@10 | No-T R@10 | $\Delta$R@10 | T NDCG@10 | No-T NDCG@10 | $\Delta$NDCG@10 |
|---|---|---|---|---|---|---|---|
| Games | RA pre-RL | 0.0876 | 0.1001 | $+0.0125$ | 0.0474 | 0.0503 | $+0.0029$ |
| Games | RL post-GRPO | 0.1055 | 0.0980 | $-0.0075$ | 0.0559 | 0.0515 | $-0.0044$ |
| Office | RA pre-RL | 0.1424 | 0.1473 | $+0.0049$ | 0.1023 | 0.1028 | $+0.0005$ |
| Office | RL post-GRPO | 0.1650 | 0.1247 | $-0.0403$ | 0.1199 | 0.0974 | $-0.0225$ |
| Industrial | RL post-GRPO | 0.1482 | 0.1366 | $-0.0116$ | 0.1023 | 0.0999 | $-0.0024$ |

### C4 — Prompt ablations

On the final post-RL checkpoints, compare four prompt conditions under deterministic
constrained beam search (beam 10):

- **Original thinking** — paper-style "reason step by step, then output the SID".
- **Strict no-think** — scaffold removed, decode the SID directly from the first token.
- **No step-by-step** — keep the prompt but drop the explicit step-by-step instruction.
- **Preference summary** — briefly summarize the user's recent preference pattern, then the SID.

| Dataset | Prompt condition | R@10 | NDCG@10 |
|---|---|---|---|
| Video Games | Original thinking | **0.1055** | 0.0559 |
| Video Games | Strict no-think | 0.0980 | 0.0515 |
| Video Games | No step-by-step | 0.0864 | 0.0489 |
| Video Games | **Preference summary** | **0.1055** | **0.0666** |
| Office Products | Original thinking | 0.1650 | 0.1199 |
| Office Products | Strict no-think | 0.1247 | 0.0974 |
| Office Products | No step-by-step | 0.1603 | **0.1239** |
| Office Products | **Preference summary** | **0.1656** | 0.1207 |
| Industrial | Original thinking | 0.1482 | 0.1023 |
| Industrial | Strict no-think | 0.1366 | 0.0999 |
| Industrial | **No step-by-step** | **0.1489** | **0.1079** |
| Industrial | Preference summary | 0.1436 | 0.1024 |

## Reproduce

The scripts are part of the SIDReasoner submodule (materialised by
`patches/common/sidreasoner-local.patch`). Run from the submodule root after applying
the patch; environment, data, and base checkpoints are assumed present (see the root
README and `scripts/common/data_prep/`).

```bash
bash scripts/common/apply_submodule_patch.sh
cd baselines/repos/SIDReasoner

# C1: train non-RL continuations from RA, then compare RA / extra-SFT / extra-RA / RL
sbatch final_scripts/rq2_components/train_ra_continuations.sh
sbatch final_scripts/rq2_components/evaluate_nonrl_vs_rl.sh

# C2: supplementary pre-RL inference-search sweep
sbatch final_scripts/rq2_components/evaluate_inference_search.sh

# C3: thinking vs strict no-think
sbatch final_scripts/rq2_components/evaluate_reasoning_ablation.sh

# C4: prompt-form ablations (appendix)
sbatch final_scripts/rq2_components/evaluate_prompt_ablations.sh
```

Per-script options (datasets, checkpoints, decoders) and the assumed data/checkpoint
layout are documented in `final_scripts/rq2_components/README.md` inside the submodule.

---
