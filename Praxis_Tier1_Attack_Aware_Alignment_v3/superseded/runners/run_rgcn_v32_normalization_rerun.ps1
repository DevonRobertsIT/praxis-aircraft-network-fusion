# Required before freezing results: rerun the trained R-GCN (full, ablation, event-only) on the 20 intensity cells under the
# training-only overlap-weight scale (notebook 06 v3.2), with UNCHANGED seeds and training settings, into rgcn_trained_v32.
# Then aggregate with 38_ (--trained-subdir rgcn_trained_v32) and re-validate with 39_ (--trained-rgcn-subdir rgcn_trained_v32).
$NB06 = "$PKG\notebooks\06_rgcn_leakage_safe_v3_trained.ipynb"
$env:RGCN_MAX_EPOCHS = "2000"; $env:RGCN_PATIENCE = "100"; $env:RGCN_EVENT_ONLY_VARIANT = "1"; $env:RGCN_OVERLAP_SCALE_SOURCE = "train"
foreach ($K in "0","1") { foreach ($SEED in 0,1,2,3,4,5,6,7,8,9) {
  $CELL = "$CELLS\cell_k${K}_s0_seed$SEED"; if (-not (Test-Path "$CELL\placement_v3.csv")) { Write-Host "missing $CELL"; continue }
  if (Test-Path "$CELL\data\processed\point_event_v2\rgcn_trained_v32\rgcn_predictions_v2.csv") { continue }
  $env:RGCN_RANDOM_STATE = "$(42 + $SEED)"; $env:RGCN_OUTPUT_SUBDIR = "rgcn_trained_v32"
  Set-Location $CELL; Write-Host "=== v3.2 R-GCN $CELL"; python "$PKG\code\run_notebook_code.py" $NB06 --cwd $CELL
} }
# aggregate (compare against the v3 runs and the Transformer on identical events)
python "$PKG\code\38_aggregate_rgcn_training_ablation_v3.py" --cells-root "$CELLS" --position-bank "$DEL\step2_coupling_study\position_bank_cross_regime_v3.csv" --trained-subdir "rgcn_trained_v32" --output-dir "$PX\rgcn_v32_aggregate"
# XAI on the replacement checkpoints (kappa 0 and 1, seed 0)
foreach ($K in "0","1") {
  python "$PKG\code\39_stage_xai_on_trained_models_v3.py" --source-root "$CELLS\cell_k${K}_s0_seed0" --trained-rgcn-subdir "rgcn_trained_v32" --package-root "$PKG" --staging-root "$PX\xai_intensity_k${K}_seed0_v32"
  & "$PX\xai_intensity_k${K}_seed0_v32\run_xai.ps1" }
# collect
$COL = "$PX\rgcn_v32_results"; if (Test-Path $COL) { Remove-Item -Recurse -Force $COL }
Get-ChildItem "$CELLS" -Directory | ForEach-Object { $s = "$($_.FullName)\data\processed\point_event_v2\rgcn_trained_v32"; if (Test-Path $s) { New-Item -ItemType Directory -Force "$COL\$($_.Name)\rgcn_trained_v32" | Out-Null; Copy-Item "$s\*.csv" "$COL\$($_.Name)\rgcn_trained_v32"; Copy-Item "$s\*.json" "$COL\$($_.Name)\rgcn_trained_v32" } }
Copy-Item -Recurse "$PX\rgcn_v32_aggregate" "$COL\aggregate"
foreach ($K in "0","1") { $x = "$PX\xai_intensity_k${K}_seed0_v32\data\processed\point_event_v2\xai_leakage_safe_v2"; if (Test-Path $x) { Copy-Item -Recurse $x "$COL\xai_k${K}_v32" } }
Compress-Archive -Path "$COL\*" -DestinationPath "$PX\rgcn_v32_results.zip" -Force
