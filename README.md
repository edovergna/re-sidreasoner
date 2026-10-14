# Beyond Accuracy in SIDReasoner: A Reproducibility Study of Reward, Scale, and Diversity

A reproducibility study and extension of **SIDReasoner** (*Reasoning over Semantic IDs Enhances
Generative Recommendation*). SIDReasoner casts sequential recommendation as **reasoning +
Semantic-ID (SID) generation**: a user's history is encoded as a prompt, the model *thinks*,
then emits a Semantic ID — three nested tokens `(a₁, a₂, a₃)`, coarse-to-fine — naming the next
item. It is trained in three stages and tuned with GRPO reinforcement learning.

We evaluate the authors' released checkpoints, independently retrain the early SFT and
reasoning-activation (RA) stages at 1.7B on Games, and run the full SFT → RA → GRPO
pipeline at 0.6B and 4B. Component and reward ablations branch from the authors' RA
checkpoints. Full independent retraining of the 1.7B pipeline is outside this study's budget.

## Setup

The training jobs target a Slurm cluster with H100 GPUs. Run setup commands from the
repository root. Submit dependent training stages only after the preceding job finishes.

**1. Download the baseline sources.** Anonymous source snapshots are verified against
checksums of the reviewed files and installed under `baselines/repos/` (requires
Python 3 and `curl`):

```bash
python3 scripts/common/download_baselines.py
```

The downloader preserves existing baseline directories. The anonymous URLs serve
source archives, not Git repositories.
After checksum verification, cluster environment scripts are renamed to `cluster_env.sh`
and their references are updated. Patch setup also normalizes existing downloads.

**2. Apply the working-tree patch.** Four of the five suites (`reproduction`, `components`,
`model_size`, `diversity`) run from scripts that live *inside* the SIDReasoner submodule, because
they call the submodule's own `split.py` / `merge.py` / `evaluate_Qwen3_think.py` directly. Rather
than committing our changes into the submodule (which would unpin it from the upstream commit),
we keep them as a single diff, `patches/common/sidreasoner-local.patch`, and lay them down into
the submodule's working tree:

```bash
bash scripts/common/apply_submodule_patch.sh
```

This is **idempotent** (safe to re-run) and creates, among other things, the per-suite launch
scripts under `baselines/repos/SIDReasoner/final_scripts/<rqN_*>/`. The submodule stays pinned to
its upstream commit (`.gitmodules` sets `ignore = dirty`); run this before sourcing the
submodule's `cluster_env.sh`. The patch must be re-applied after any fresh
baseline download.

**3. Build the environment.** Creates the SIDReasoner `uv` env (Python 3.10) inside the
submodule and re-applies the patch first:

```bash
sbatch jobs/common/setup_env.sh            # add --sglang / --megatron / --all for extras
```

**4. Download data and checkpoints.** The wrapper uses `data/raw/<slug>/` and
`data/checkpoints/`. Submodule launchers instead use `data/Amazon/` and `pretrained/`
inside the submodule. These directories are git-ignored. The commands below populate
the wrapper layout only. For native launchers, place data under the SIDReasoner
baseline’s `data/Amazon/` directory and checkpoints under `pretrained/`, or set the
launcher path overrides. Each dataset call extracts the full shared archive;
the baseline Makefile does not filter by category.

```bash
# Datasets (delegates to the SIDReasoner Makefile):
bash scripts/common/data_prep/download_amazon.sh Video_Games
bash scripts/common/data_prep/download_amazon.sh Office_Products
bash scripts/common/data_prep/download_amazon.sh Industrial_and_Scientific

# SASRec uses the same archive; override its unavailable default Drive ID:
make -C baselines/repos/SASRec data DATA_FILE_ID=1etg1e8oStGOjsg1Vr15vFnjlTMUx4Htz

# Author SFT/RA checkpoints from HuggingFace (heyingzhi/sidreasoner_*):
bash scripts/common/data_prep/download_checkpoints.sh

# Or as Slurm jobs (same steps, batched):
sbatch jobs/common/download_amazon.job
sbatch jobs/common/download_checkpoints.job
sbatch jobs/common/download_model.job        # Qwen3 backbone
```

Per-suite run commands (which launch script to `sbatch`, expected metrics) are in each suite's
doc — see the **Docs** column below.

## Training pipeline

| Stage | Name | Purpose | Checkpoint |
|-------|------|---------|------------|
| S1 | SFT | Emit valid Semantic IDs. | `sidreasoner_sft_*` |
| S2 | Reasoning-activation (RA) | Generate a rationale inside `<think>…</think>`, followed by the SID. | `sidreasoner_ra_*` |
| S3 | GRPO RL | Reinforce so reasoning leads to correct recommendations. | post-GRPO |

Backbone: `Qwen/Qwen3-1.7B`. Datasets: Amazon `Video Games`, `Office Products`,
`Industrial & Scientific` (5-core filtered, sliding window 10, chronological 8:1:1 split,
full-item ranking). Primary metrics: Recall@{5,10}, NDCG@{5,10}.

## Research questions / experiment suites

The study is organized into five suites, one per research question. The code tree and `docs/`
mirror these names (see [`CODE-STRUCTURE.md`](CODE-STRUCTURE.md)).

| Suite | RQ | Question | Docs |
|-------|----|----------|------|
| `reproduction` | RQ1 | Can released artifacts recover the reported results, and can partial pre-RL retraining recover the early-stage trends? | [docs](docs/reproduction/README.md) |
| `components` | RQ2 | Are RL and explicit reasoning strictly necessary? | [docs](docs/components/README.md) |
| `grpo` | RQ3 | Is the GRPO reward formulation a sensitive design choice? | [docs](docs/grpo/README.md) |
| `model_size` | RQ4 | How does backbone scale influence behavior? | [docs](docs/model_size/README.md) |
| `diversity` | RQ5 | How do decoding strategies affect item exposure? | [docs](docs/diversity/README.md) |

RQ1–RQ2 are the reproduction analyses (§5); RQ3–RQ5 are extensions (§6).
`HR@K` in code/results is Recall@K here because each test case has one target item.
Pipeline stages 1–3 are distinct from the original paper's S1/S3 *data ablation* labels.

All five suites are written up — start with the **Docs** link in each row above, or see
[`docs/README.md`](docs/README.md) for the documentation index.

## Repository layout

GRPO configs, jobs, and reward code live in this repository. The other four suites
use launchers inside the SIDReasoner submodule, added by the common patch.
Experiment documentation is in `docs/<suite>/`.

| Directory | Contents |
|-----------|----------|
| `baselines/` | Baseline repos (submodules under `repos/`) + uniform `adapters/`. |
| `configs/` | YAML experiment configs (`configs/grpo/<variant>/<dataset>.yaml`). |
| `data/` | `raw/ processed/ enriched/` datasets, SID mappings, checkpoints. |
| `scripts/` | Shared evaluation, training, and data-preparation entry points under `common/`. |
| `src/` | Importable library — `src.common.*`, `src.<suite>.*`. |
| `jobs/` | Slurm job files. |
| `results/` | Metrics + prediction files, mirroring `configs/`. |
| `patches/` | Local submodule modifications. |
| `docs/` | Per-suite documentation (`docs/<suite>/`). |

See [`CODE-STRUCTURE.md`](CODE-STRUCTURE.md) for the full layout and conventions (depth-independent
repo-root resolution, the `model_size` underscore rule).
