# Documentation Index

The experiment suites cover reproduction as RQ1–RQ2 (§5) and extensions as RQ3–RQ5 (§6).
Suite names mirror [CODE-STRUCTURE.md](../CODE-STRUCTURE.md).

| Suite | RQ | Research question | Paper location |
|---|---|---|---|
| [reproduction](reproduction/README.md) | RQ1 | Can released artifacts recover the reported results, and can partial pre-RL retraining recover the early-stage trends? | §5, Table 1, Appendix A |
| [components](components/README.md) | RQ2 | Are RL and explicit reasoning strictly necessary? | §5, Table 2, Appendix B / Table 4 |
| [grpo](grpo/README.md) | RQ3 | Is the GRPO reward formulation a sensitive design choice? | §6, Table 3, Appendix C |
| [model_size](model_size/README.md) | RQ4 | How does backbone scale influence behavior? | §6, Figure 2, Appendix D |
| [diversity](diversity/README.md) | RQ5 | How do decoding strategies affect item exposure? | §6, Figure 3 |

Pipeline stages SFT / RA / GRPO are distinct from the original paper's S1/S3
data-ablation labels. `HR@K` equals Recall@K for the single-target test protocol.
