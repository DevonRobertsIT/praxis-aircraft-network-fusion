# Trained R-GCN vs. Transformer on the intensity target — ten seeds (v3)

**Runs (student's machine, 22–23 Sep 2026):** 20 coupling cells, κ ∈ {0, 1} at σ = 0, seeds 0–9, each materialized by script 35 (seeds 0–4 verified against the tabular grid record; 5–9 are new placements), Transformer via unmodified notebook 05 and the trained R-GCN via the patched notebook 06 (2,000 / 100, event-only variant), seed = 42 + cell seed for both. Paired differences on identical test events (1,015 attack-block events, 27 independent 240 s bins), 1,000-rep bin-cluster bootstrap per cell; seed-level inference with t-intervals over the ten cell means.

## Held-out AUPRC (mean ± SD over 10 seeds)

| κ | Transformer | R-GCN full (trained) | R-GCN ablation (trained) | Event-only MLP (trained) | Tabular reference (5-seed grid) |
|---|---|---|---|---|---|
| 0 | 0.750 ± 0.060 | **0.815 ± 0.040** | 0.795 ± 0.035 | 0.751 ± 0.078 | cyber LR 0.884 · late-fusion RF 0.705 |
| 1 | 0.876 ± 0.029 | **0.905 ± 0.011** | 0.895 ± 0.012 | 0.885 ± 0.031 | late-fusion RF 0.951 · physical RF 0.969 |

Training: full variant stopped after 272–536 epochs, event-only after 118–394; local p95 latency ≤ 3.6 ms.

## Seed-level inference (10 seeds; t-interval; counts of seeds whose bin-cluster CI lies above zero)

| κ | Question | Mean | 95% t-interval | Seeds > 0 | Bin-CIs > 0 | Bin-CIs < 0 | p (two-sided vs 0) |
|---|---|---|---|---|---|---|---|
| 0 | **H2′** R-GCN full − Transformer | **+0.065** | [+0.011, +0.120] | 7 / 10 | 5 | 0 | 0.025 |
| 0 | **Relational** full − event-only MLP | **+0.064** | [+0.022, +0.107] | 8 / 10 | 6 | 0 | 0.008 |
| 0 | H3′ full − ablation | +0.020 | [−0.007, +0.047] | 7 / 10 | 4 | 0 | 0.13 |
| 0 | Event-only MLP − Transformer | +0.001 | [−0.058, +0.061] | 5 / 10 | 2 | 2 | 0.96 |
| 1 | **H2′** R-GCN full − Transformer | **+0.029** | [+0.005, +0.054] | 7 / 10 | 5 | 0 | 0.023 |
| 1 | Relational full − event-only MLP | +0.021 | [−0.002, +0.043] | 8 / 10 | 3 | 0 | 0.073 |
| 1 | H3′ full − ablation | +0.010 | [−0.003, +0.022] | 7 / 10 | 2 | 0 | 0.11 |
| 1 | Event-only MLP − Transformer | +0.009 | [−0.023, +0.041] | 4 / 10 | 3 | 1 | 0.55 |

One-sided tests against the pre-registered margins: H2′ ≥ +0.03 — p = 0.089 (κ = 0), 0.52 (κ = 1); H3′ ≥ +0.05 — p = 0.98 and 1.00.

## Decisions

- **H2′ (relational vs. sequential): the trained R-GCN is better than the Transformer, significantly, at both coupling levels** (+0.065, p = 0.025 at κ = 0; +0.029, p = 0.023 at κ = 1; no seed's cluster interval favours the Transformer). The pre-registered "≥ +0.03" margin is exceeded by both point estimates but not by the interval lower bounds (+0.011, +0.005), so under the original rule H2′ is *directionally supported, margin not established*. Stated plainly for Chapter 5: "The relational model outperformed the sequential model on the graded target; the size of the advantage is between 0.01 and 0.12 at κ = 0 and between 0.01 and 0.05 at κ = 1."
- **Relational structure has real value where the task is purely cyber.** At κ = 0 the full graph beats the same model with all relations removed by +0.064 [+0.022, +0.107] (p = 0.008), and the event-only MLP is indistinguishable from the Transformer (+0.001). The R-GCN's entire advantage over the Transformer at κ = 0 is therefore attributable to the relations (flow neighbourhoods and event-to-event links), which is exactly the claim the praxis wanted to make. At κ = 1 the increment shrinks to +0.021 (p = 0.07) because the physical coupling, which every model sees on the event node, dominates.
- **H3′ (lag/relation weighting) — not supported.** +0.020 and +0.010, far below the +0.05 margin; the unweighted graph carries the value, the lag weighting does not. Replace H3 with the event-only comparison in the praxis.
- **Neither neural model is the best model.** At κ = 0 the regularized logistic regression on window aggregates (0.884) still beats the trained R-GCN (0.815); at κ = 1 tabular late fusion (0.951) beats it. The defensible contribution is "relational neural modelling of flow evidence beats sequential neural modelling on the same inputs", not "the R-GCN is the recommended prioritizer". The gap to the LR at κ = 0 is a further finding: both neural models carry 47 physical features that are pure noise at κ = 0 and appear to pay for it, where the L2-regularized linear model does not — a modality-gating or cyber-only neural variant is the obvious follow-up.
- **Stability.** The trained R-GCN is the least seed-sensitive model on the graded target (SD 0.040 at κ = 0, 0.011 at κ = 1, vs 0.060 / 0.029 for the Transformer).

## What this means for the praxis

1. H2 is rewritten as above and reported with the ten-seed intervals; the original −0.046 / −0.082 results (untrained model) are described in Chapter 4 as superseded and why.
2. H3 is retired; the event-only comparison becomes the relational-value result.
3. Chapter 3 gains: the training regime (full-batch, epochs, patience, early stopping on validation AUPRC), the event-only control, and the seed protocol.
4. Chapter 5 adds the lesson: the earlier model-family ranking was reversed by fixing a 50-step training cap. Model comparisons are only as valid as the weakest training protocol they contain.

## Files

`rgcn_ablation/seeds_0_to_9/`: `rgcn_training_budget_ablation_runs_v3.csv` (per cell, all AUPRCs and paired differences with bin-cluster intervals), `rgcn_training_budget_ablation_aggregate_v3.csv`, `seed_level_inference_v3.csv`, `RGCN_ABLATION_SUMMARY_v3.md`, `rgcn_seeds_figure_v3.png`, `raw/` (per cell: placement, Transformer and R-GCN result tables, hypothesis tables, training histories).

## Attribution

Aggregation, inference, and report by Claude (Anthropic) at the student's direction, 23 September 2026; model runs executed by the student with the unmodified Transformer notebook and the minimally patched R-GCN notebook.

*Correction (manuscript v6): the κ = 1 late-fusion RF reference previously read 0.950, which is the seeds 0–2 mean from the Step 2 same-cell table; the five-seed grid value in this column is 0.951 (review_corrections_v31/coupling_auprc_surface_v3.csv).*
