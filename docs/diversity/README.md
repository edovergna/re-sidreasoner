# Item-exposure evaluation

Top-10 exposure metrics for SIDReasoner and SASRec on Games. The scripts also
support Office and the SFT/RA checkpoints.

## Item-side exposure diversity

Three exposure metrics are computed over the top-10 lists (added to the modified
`calc.py`):

| Metric | Meaning | Better when |
|--------|---------|-------------|
| `Coverage@10` | fraction of the catalog that appears in any top-10 | higher (broader exposure) |
| `Entropy@10` | entropy of the item-exposure distribution | higher (more evenly spread) |
| `Gini@10` | concentration of exposure among a few items | lower (less concentrated) |

Games results in Figure 3 use means and standard deviations over seeds 42, 43,
and 44. Relative changes reported in the paper:

| Method | Reference | ΔR@10 | ΔCoverage@10 | ΔGini@10 |
|---|---|---|---|---|
| SIDReasoner beam search | SASRec | +14.5% | -2.8% | -5.0% |
| SIDReasoner top-p | SIDReasoner beam search | -38.5% | +20.9% | -11.8% |
| SIDReasoner top-k | SIDReasoner beam search | -40.9% | +20.3% | -9.5% |

Gini includes unexposed catalog entries. Exposure units are SIDs for SIDReasoner
and item IDs for SASRec.

The evaluation launcher now defaults to all stages (`STAGES="sft ra rl"`).
The table builder recomputes every SIDReasoner row from per-seed prediction JSONs,
including RL. Missing predictions fail explicitly; there is no fallback to the old
RL aggregates. SASRec still uses its stored full-catalog baseline aggregates, which
require matching catalog and test-set provenance.

Existing predictions can be rescored without retraining or rerunning generation:
from the patched submodule root, run
`$PYTHON_CMD utils/make_rq5_diversity_tables.py` once the catalog info files and
SFT/RA/RL prediction JSONs for all seeds and decoders are in their expected paths.


### Reproduce

The scripts are part of the SIDReasoner submodule (materialised by
`patches/common/sidreasoner-local.patch`). Run from the submodule root after applying the
patch; environment, data, and base checkpoints are assumed present (see the root README
and `scripts/common/data_prep/`).

```bash
bash scripts/common/apply_submodule_patch.sh
cd baselines/repos/SIDReasoner

# 1. evaluate SFT, RA, and RL stages -> per-run JSON + manifest + summary
sbatch final_scripts/rq5_diversity/evaluate_item_side_diversity.sh

# 2. build the paper tables once the JSONs exist
sbatch final_scripts/rq5_diversity/make_diversity_tables.sh
```

Outputs land under `results/item_side_diversity/` (manifest, summary, per-run JSONs, and
the `*_sft_ra_rl_sasrec_comparison.md` tables). Per-script options and the assumed
data/checkpoint layout are documented in
`final_scripts/rq5_diversity/README.md` inside the submodule.

## Reference files

| Area | Path |
|------|------|
| Launch scripts (submodule) | `baselines/repos/SIDReasoner/final_scripts/rq5_diversity/` (via `patches/common/sidreasoner-local.patch`) |
| Outputs | `results/item_side_diversity/` |
