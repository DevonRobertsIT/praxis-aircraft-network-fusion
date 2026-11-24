# Step 2 — Transformer and R-GCN on the coupling cells (intensity target): H2′ and H3′

**Runs:** student's machine, notebooks 05/06 unmodified on 18 cells materialized by `35_materialize_coupling_cell_v3.py` (κ ∈ {0, 0.5, 1} × σ ∈ {0, 120 s} × seeds 0–2). Each cell reproduces the tabular grid's placement exactly, so torch and tabular models are compared on identical test events (1,015 attack-block events, 27 independent 240 s bins, prevalence 0.53–0.61).
**Aggregation:** `36_aggregate_torch_reruns_v3.py --cells-root`, 1,000-rep bin-cluster bootstrap; H2′/H3′ rule: seed-mean ≥ margin and every seed's interval above zero.

## Torch results by cell (mean of 3 seeds)

| σ | κ | Transformer (fused) | R-GCN full | R-GCN ablation | H2′: R-GCN − Transformer [seed range] | H3′: full − ablation [seed range] | max p95 |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 0.777 | 0.580 | 0.588 | −0.197 [−0.224, −0.183] | −0.008 [−0.043, +0.014] | 3.6 ms |
| 0 | 0.5 | 0.748 | 0.623 | 0.632 | −0.125 [−0.152, −0.090] | −0.009 [−0.044, +0.009] | 3.5 ms |
| 0 | 1 | 0.874 | 0.699 | 0.699 | −0.175 [−0.196, −0.139] | +0.001 [−0.018, +0.013] | 3.6 ms |
| 120 | 0 | 0.710 | 0.613 | 0.604 | −0.097 [−0.112, −0.079] | +0.009 [−0.018, +0.036] | 3.6 ms |
| 120 | 0.5 | 0.738 | 0.624 | 0.618 | −0.114 [−0.118, −0.108] | +0.006 [−0.014, +0.028] | 3.5 ms |
| 120 | 1 | 0.894 | 0.716 | 0.719 | −0.178 [−0.208, −0.121] | −0.003 [−0.022, +0.008] | 3.7 ms |

Per-cell bin-cluster intervals are wide (e.g., Transformer at κ = 0: 0.75 [0.55, 0.90]; R-GCN: 0.56 [0.35, 0.78]) — 27 independent bins.

## Decisions

- **H2′ — not supported at any κ.** The R-GCN trails the Transformer by 0.10–0.20 in every cell; all 18 bin-cluster intervals are below zero. At κ = 0 the R-GCN (0.58) sits at prevalence (0.53–0.61): it cannot see window-level intensity at all, because its flow nodes carry 12 per-flow fields from 16 sampled flows out of ~5,000 and its event node carries no cyber aggregates. The only signal it ever learns is the physical coupling (+0.12 from κ = 0 to 1, through the event node's physical features).
- **H3′ — not supported at any κ.** Lag/relation features add −0.009 to +0.009; intervals straddle zero in 17 of 18 cells. Latency passes trivially (p95 ≤ 3.7 ms).

## Same cells, tabular models (seeds 0–2 matched)

| σ | κ | Transformer | cyber LR | cyber RF | physical RF | concat-fused RF | late-fusion RF |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 0.777 | **0.889** | 0.704 | 0.558 | 0.691 | 0.696 |
| 0 | 1 | 0.874 | 0.884 | 0.707 | **0.968** | 0.838 | 0.950 |
| 120 | 0 | 0.710 | **0.856** | 0.692 | 0.574 | 0.696 | 0.685 |
| 120 | 1 | 0.894 | 0.874 | 0.698 | **0.968** | 0.890 | 0.955 |

Coupling gain, AUPRC(κ = 1) − AUPRC(κ = 0), seed-matched: Transformer +0.098 (σ = 0) / +0.183 (σ = 120); R-GCN +0.119 / +0.103; late-fusion RF +0.254 / +0.270; cyber LR −0.005 / +0.019 (as required — cyber-only cannot see coupling).

## What the praxis can now say

1. **On a non-ceiling target, neither neural architecture beats the best tabular model.** At κ = 0 the Transformer (0.78) is 0.11 below a logistic regression on the same window features (0.89); at κ = 1 it is 0.08 below late fusion (0.95) and 0.09 below physical-only RF (0.97). The neural models detect coupling (both gain ≈ 0.10–0.18) but exploit it worse than a probability average of two random forests.
2. **H2 and H3 fail on both targets for a structural reason, not a tuning reason.** The R-GCN's information budget — 16 label-blind flows with 12 fields each — is insufficient to estimate window intensity; the relational machinery has nothing to relate. Any future R-GCN must place the window aggregates on the event node (the input-budget ablation recommended in Tier 3), or the comparison remains uninformative.
3. **Alignment error hurts the cyber pathway and is compensated by coupled physical evidence**, in the neural models exactly as in the tabular ones: Transformer 0.777 → 0.710 at κ = 0 under 120 s jitter, but 0.874 → 0.894 at κ = 1.
4. **Concatenation fusion is the bottleneck across model families.** Transformer (fused features, +0.098) and concat RF (+0.147) both fall well short of late fusion (+0.254); the Chapter 3 architecture should add a late-fusion or stacked head as the reference fusion.
5. **Sample-size floor stands:** 27 independent held-out bins; per-cell intervals ±0.15–0.20; only seed-averaged, cluster-corrected differences are reportable.

## Files

`step2_torch_cells/step2_cell_results_v3.csv` (per cell, all AUPRCs and paired differences with intervals), `step2_cell_aggregate_v3.csv`, `step2_torch_vs_tabular_same_cells_v3.csv`, `step2_torch_vs_tabular_per_cell_v3.csv`, `step2_cells_figure_v3.png`, `raw/` (each cell's notebook result, hypothesis, and latency CSVs plus `placement_v3.csv`).

## Attribution

Aggregation code and this report by Claude (Anthropic) at the student's direction, 21 September 2026; model runs executed by the student with the unmodified v2 notebooks on inputs materialized by the student from the v3 package.
