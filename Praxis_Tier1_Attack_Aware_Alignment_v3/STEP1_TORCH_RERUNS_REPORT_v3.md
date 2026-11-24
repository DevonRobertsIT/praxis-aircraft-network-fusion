# Step 1 — Transformer and R-GCN on the cross-regime alignment (regime target), five seeds

**Runs:** student's machine, notebooks 05/06 unmodified via `run_notebook_code.py`, seeds 42–46 (`TT_RANDOM_STATE`, `RGCN_RANDOM_STATE`), inputs = `alignment_v3_cross_regime` staged by `32_stage_torch_reruns_v3.py`.
**Aggregation:** `36_aggregate_torch_reruns_v3.py`, 1,000-rep bin-cluster bootstrap over the 49 independent 240 s source bins of the 1,828 test events (1,015 positive / 813 negative, all Session 1). Paired differences are computed on identical test events. AUPRC values recomputed from the saved per-event predictions agree with the notebooks' reported values.

## Results

| Seed | Transformer | R-GCN full | R-GCN ablation | H2: R-GCN − Transformer [95% CI] | H3: full − ablation [95% CI] | R-GCN p95 (ms) |
|---|---|---|---|---|---|---|
| 42 | 0.99994 | 0.9499 | 0.9334 | −0.0500 [−0.0918, −0.0266] | +0.0165 [+0.0042, +0.0345] | 3.37 |
| 43 | 0.99999 | 0.9546 | 0.9609 | −0.0454 [−0.0824, −0.0196] | −0.0062 [−0.0168, +0.0010] | 3.30 |
| 44 | 0.99991 | 0.9457 | 0.9657 | −0.0542 [−0.0970, −0.0259] | −0.0200 [−0.0526, +0.0036] | 3.48 |
| 45 | 0.99988 | 0.9628 | 0.9686 | −0.0371 [−0.0711, −0.0158] | −0.0058 [−0.0335, +0.0200] | 3.45 |
| 46 | 0.99989 | 0.9582 | 0.9550 | −0.0417 [−0.0807, −0.0172] | +0.0032 [−0.0181, +0.0261] | 3.44 |
| **mean** | **0.99992** | **0.9542** | **0.9567** | **−0.0457** | **−0.0025** | 3.41 |

Against the tabular models on the same events: cyber RF and fused RF = 1.000 on every seed; Transformer − cyber RF ≈ −0.0001 (identical within rounding); R-GCN − cyber RF = −0.037 to −0.054 with every interval below zero.

## Decisions

- **H2 — not supported.** The R-GCN is *worse* than the Transformer by 0.046 on average, and the bin-cluster interval excludes zero in the negative direction on all five seeds. This is now a seed-robust, cluster-corrected result rather than a single-run point estimate.
- **H3 — not supported.** Lag/relation features add −0.0025 on average (range −0.020 to +0.017); only seed 42's interval is above zero. The latency criterion passes trivially (p95 ≤ 3.5 ms against a 500 ms limit), so H3 fails on predictive gain alone.

## Interpretation for the praxis

1. **This is the ceiling task, and the ranking of model families reflects failure modes, not capability.** The regime label is solved perfectly by a depth-3 tree on `nf3_dst_ip_unique_count`; the Transformer reproduces that ceiling; the R-GCN's flow nodes see only 12 transformed per-flow fields and never see window-level host diversity directly, so it lands at 0.95. The R-GCN falling *below a decision tree* is the clearest statement possible that Chapter 4's H2/H3 comparisons on this target say nothing about relational modeling.
2. **Seed variance is small for the Transformer (±0.00006) and moderate for the R-GCN (±0.009 full, ±0.014 ablation).** Report means with ranges and cluster intervals; single-seed numbers in the current Chapter 4 should be replaced by these.
3. **H2/H3 in their original form should be retired** and the restated H2′/H3′ evaluated on the intensity target through the coupling cells (Step 2), where headroom exists (cyber LR 0.88, RF 0.71 at κ = 0) and the tabular study has already established a detection floor (late fusion, ρ ≥ 0.75).

## Files

`step1_torch_reruns/step1_seed_results_v3.csv` (per-seed AUPRCs, all paired differences with intervals), `step1_summary_v3.json`, `TORCH_RERUNS_SUMMARY_v3.md`, `step1_seed_figure_v3.png`, `raw/` (each seed's notebook result, hypothesis, and latency CSVs); aggregator `code/36_aggregate_torch_reruns_v3.py` (also aggregates Step 2 cells via `--cells-root` once they exist).

## Attribution

Aggregation code and this report by Claude (Anthropic) at the student's direction, 21 September 2026; the model runs were executed by the student with the unmodified v2 notebooks.
