# R-GCN training-budget + relational-value ablation (patched notebook 06, original outputs untouched).
# Run AFTER defining $PX/$PKG/$DEL/$ROOT/$CELLS and copying 06_rgcn_leakage_safe_v3_trained.ipynb into $PKG\notebooks\
# Setting: full-batch training allowed up to 2,000 epochs with patience 100 (vs 50 / 8 originally); adds the
# event-only self-loop variant. Each run trains 3 variants; expect 2-15 min per run on CPU.
$env:RGCN_MAX_EPOCHS = "2000"; $env:RGCN_PATIENCE = "100"; $env:RGCN_EVENT_ONLY_VARIANT = "1"
$NB = "$PKG\notebooks\06_rgcn_leakage_safe_v3_trained.ipynb"
# (a) six coupling cells: kappa 0 and 1 at sigma 0, seeds 0-2 (same seeds the original cell runs used: 42 + seed)
foreach ($K in "0","1") { foreach ($SEED in 0,1,2) {
  $CELL = "$CELLS\cell_k${K}_s0_seed$SEED"; if (-not (Test-Path $CELL)) { Write-Host "missing $CELL"; continue }
  $env:RGCN_RANDOM_STATE = "$(42 + $SEED)"; $env:RGCN_OUTPUT_SUBDIR = "rgcn_trained_v3"
  Set-Location $CELL; Write-Host "=== $CELL"; python "$PKG\code\run_notebook_code.py" $NB --cwd $CELL
} }
# (b) the Step 1 regime-target root, seeds 42-46 (does a trained R-GCN reach the ceiling too?)
foreach ($SEED in 42,43,44,45,46) {
  $env:RGCN_RANDOM_STATE = "$SEED"; $env:RGCN_OUTPUT_SUBDIR = "rgcn_trained_v3_seed$SEED"
  Set-Location $ROOT; Write-Host "=== step1 seed $SEED"; python "$PKG\code\run_notebook_code.py" $NB --cwd $ROOT
}
Write-Host "Done. Zip: each cell's data\processed\point_event_v2\rgcn_trained_v3 and the root's rgcn_trained_v3_seed* folders (skip .pt/.png)."
