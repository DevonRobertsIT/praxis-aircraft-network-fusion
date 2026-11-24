# R-GCN ablation (v3): training budget and relational value

## Why this replaces the "input-budget" ablation
Notebook 06 already gives the event node the fused feature bank (physical + cyber window aggregates); the
input-budget explanation was wrong. The deficit is training: the R-GCN is trained full-batch with ONE optimizer
step per epoch, capped at 50 epochs with patience 8 -> at most 50 gradient steps (5.6 s of training). Training
loss is still falling and validation AUPRC still rising at the last epoch in every run; best_epoch = last epoch
whenever the cap binds. The Transformer takes ~1,200 mini-batch steps.

## What the patched notebook changes (06_rgcn_leakage_safe_v3_trained.ipynb; 13 lines added, 1 replaced)
1. RGCN_OUTPUT_SUBDIR env var -> outputs go to a new folder; original rgcn_leakage_safe_v2 results are untouched.
2. Third graph variant `event_only_self_loop` (enabled with RGCN_EVENT_ONLY_VARIANT=1): the only relation is an
   event self-loop, so the model is an MLP on the event node. Its distance from the full model is the value of
   the relational structure.
3. The causality assertion skips the self-loop relation.
Epochs, patience, learning rate and seed remain env-controlled (unchanged code); the runner sets 2000 / 100.

## Run
1. Copy `06_rgcn_leakage_safe_v3_trained.ipynb` into `$PKG\notebooks\`.
2. In a PowerShell terminal with $PX/$PKG/$DEL/$ROOT/$CELLS defined (see RUN_ME_FIRST.md):
       & "$DEL\rgcn_ablation\run_rgcn_ablation_v3.ps1"
   Runs 6 cells (kappa 0/1, sigma 0, seeds 0-2) and the Step 1 root (5 seeds). Run one cell first if unsure.
3. Zip and send: `<cell>\data\processed\point_event_v2\rgcn_trained_v3\` for the 6 cells and
   `$ROOT\data\processed\point_event_v2\rgcn_trained_v3_seed*\` (predictions, results, history; skip .pt/.png).

## Aggregate (here or on your machine)
    python "$PKG\code\38_aggregate_rgcn_training_ablation_v3.py" --cells-root "$CELLS" --position-bank "$DEL\step2_coupling_study\position_bank_cross_regime_v3.csv" --step1-root "$ROOT" --alignment-dir "$DEL\alignment_v3_cross_regime" --output-dir "$PX\rgcn_ablation_out"
Outputs per run and aggregated: AUPRC of Transformer / original full / trained full / trained ablation / event-only,
and the paired differences  budget (trained - original), relational (full - event-only), H2 (full - Transformer),
H3 (full - ablation), each with bin-cluster 95% intervals.

## How to read the outcome
- trained full >> original full, and full - event_only > 0 with intervals above zero at kappa=1: relational
  modeling is rescued (the graph adds value once the model is trained).
- trained full ~= event_only: the relations add nothing even when trained -> H2/H3 fail for architectural reasons;
  the R-GCN's contribution reduces to its event-node MLP.
- trained full ~= original full: training budget was not the limitation (unlikely given the histories).
