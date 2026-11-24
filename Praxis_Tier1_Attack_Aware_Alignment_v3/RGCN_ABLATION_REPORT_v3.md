# R-GCN ablation — training budget and relational value (v3)

**Runs (student's machine, 22 Sep 2026):** patched notebook `06_rgcn_leakage_safe_v3_trained.ipynb` (13 lines added, 4 replaced: output-folder override, an `event_only_self_loop` variant, variant-aware regression checks) with `RGCN_MAX_EPOCHS=2000`, `RGCN_PATIENCE=100`; everything else — architecture, learning rate, preprocessing, seeds, inputs — identical to the original runs. Eleven runs: the Step 1 regime-target root (seeds 42–46) and six coupling cells (intensity target, κ ∈ {0, 1}, σ = 0, seeds 0–2). Original and trained models are compared on identical test events with 1,000-rep bin-cluster bootstrap intervals (49 bins on the regime target, 27 on the cells).

## Why this ablation replaced the "input-budget" one

Inspection of notebook 06 showed the event node already receives the fused feature bank (physical + cyber window aggregates); the input-budget explanation offered earlier was wrong. The actual limitation was the training loop: full-batch, one optimizer step per epoch, capped at 50 epochs with patience 8 — at most 50 gradient steps, 5.6 s of training — while the Transformer takes ~1,200 mini-batch steps. Training loss was still falling and validation AUPRC still rising at the last epoch of every original run. **Every R-GCN number in the praxis before this ablation came from an untrained model.** Trained runs stopped after 362–550 epochs (full variant), 178–334 (event-only).

## Results (mean over seeds; per-seed values in `results/rgcn_training_budget_ablation_runs_v3.csv`)

| Target | κ | Transformer | R-GCN full, original | **R-GCN full, trained** | R-GCN ablation, trained | Event-only MLP, trained |
|---|---|---|---|---|---|---|
| Regime (5 seeds) | — | 0.9999 | 0.9542 | **0.9996** | 0.9996 | 0.9977 |
| Intensity (3 seeds) | 0 | 0.7768 | 0.5803 | **0.7865** | 0.8032 | 0.7483 |
| Intensity (3 seeds) | 1 | 0.8744 | 0.6993 | **0.9082** | 0.9032 | 0.8933 |

Paired differences on identical events (seed mean; number of seeds whose bin-cluster interval lies entirely above / below zero):

| Question | Regime | Intensity κ = 0 | Intensity κ = 1 |
|---|---|---|---|
| **Budget**: trained full − original full | +0.045 (5 / 0) | +0.206 (3 / 0) | +0.209 (3 / 0) |
| **Relational value**: trained full − event-only MLP | +0.002 (4 / 0) | +0.038 (1 / 0), all seeds positive | +0.015 (0 / 0), all seeds positive |
| **H2 / H2′**: trained full − Transformer | −0.0003 (0 / 1) | +0.010 (1 / 0) | +0.034 (2 / 0) |
| **H3 / H3′**: trained full − trained ablation | 0.000 (1 / 0) | −0.017 (0 / 0) | +0.005 (0 / 0) |
| Event-only MLP − Transformer | −0.002 (0 / 5) | −0.029 (0 / 0) | +0.019 (1 / 1) |

Latency: trained full-variant local p95 3.1–10.8 ms (limit 500 ms).

## Decisions

- **Training budget was the entire original deficit.** +0.045 on the regime target and +0.21 on the intensity target, every seed's interval above zero. The trained R-GCN reaches the ceiling on the regime target (0.9996 vs. 0.9999) and matches or exceeds the Transformer on the intensity target. The −0.046 H2 result reported for Step 1, and the −0.10 to −0.20 H2′ results across the coupling cells, were artifacts of a 50-step training cap.
- **H2 / H2′ — not supported under the pre-registered rule, but the direction has reversed.** Regime: a tie (−0.0003). Intensity κ = 1: +0.034 mean, above the +0.03 margin, with two of three seeds' intervals above zero (the third, seed 1, is −0.010). Intensity κ = 0: +0.010, one of three. The claim "the relational model is inferior to the sequential model" is no longer supportable; the claim "it is better by ≥ 0.03" is not yet supportable either — that is a seed-count question (three seeds at 27 independent bins).
- **Relational value is small and consistently positive.** Full − event-only is positive in every one of the eleven runs (+0.001 to +0.071). At the ceiling it is +0.002 with four of five intervals above zero; on the intensity target it is +0.038 (κ = 0) and +0.015 (κ = 1), inside the intervals. Most of the trained R-GCN's performance is its event-node MLP; the graph adds a modest increment that is largest when no physical coupling exists.
- **H3 / H3′ — not supported.** Lag/relation weighting adds nothing once trained: the unweighted ablation equals or exceeds the full model (κ = 0: −0.017; κ = 1: +0.005; regime: 0.000). The relational *structure* helps a little; the *lag weighting* of that structure does not.
- **Still below the best tabular model at κ = 1**: trained R-GCN 0.908 vs. late-fusion RF 0.950 and physical RF 0.969.

## What this changes in the praxis

1. Chapter 3 must state the training regime (full-batch, step count) and Chapter 4 must be rebuilt from the trained runs; the current R-GCN tables describe an untrained model.
2. H2 is retired as "R-GCN loses"; it becomes "trained R-GCN ties the Transformer at the ceiling and matches or edges it on the graded target; the +0.03 margin is not established with three seeds." Five to ten seeds on the two intensity cells would settle it and are cheap (≈ 5 min per run).
3. H3 is retired outright: lag/relation weighting has no measurable value; the graph structure's value is the event-only comparison, which should replace H3.
4. The training-budget finding itself belongs in Chapter 5 as a lesson for practice: a model comparison is only as valid as the weakest training protocol in it.

## Files

`rgcn_ablation/results/`: `rgcn_training_budget_ablation_runs_v3.csv` (per run, all AUPRCs and paired differences with intervals), `rgcn_training_budget_ablation_aggregate_v3.csv`, `RGCN_ABLATION_SUMMARY_v3.md`, `rgcn_ablation_figure_v3.png`, and the eleven runs' notebook outputs (results, predictions, histories, hypothesis tables, latency; edge and node manifests omitted for size). Notebook and runner in `rgcn_ablation/`; aggregator `code/38_aggregate_rgcn_training_ablation_v3.py`.

## Attribution

Patch, aggregation, and report by Claude (Anthropic) at the student's direction, 22 September 2026; runs executed by the student. The original notebook is the student's v2 code; the patch is confined to the lines listed above.
