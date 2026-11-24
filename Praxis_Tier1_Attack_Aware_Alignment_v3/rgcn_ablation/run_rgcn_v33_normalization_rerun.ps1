# FINAL PROCEDURE (v3.3): R-GCN intensity rerun under the training-only overlap scale, aggregation, and XAI re-validation.
# Required files (copy from this package's rgcn_ablation\ and code\ folders into the v2 package):
#   $PKG\notebooks\06_rgcn_leakage_safe_v3_trained.ipynb    (notebook 06 v3.3; must contain "overlap_scale_v33")
#   $PKG\notebooks\09_xai_validation_leakage_safe_v33.ipynb (notebook 09 v3.3; reconstructs with the checkpoint's recorded scale)
#   $PKG\code\38_aggregate_rgcn_training_ablation_v3.py and $PKG\code\39_stage_xai_on_trained_models_v3.py
# Requires $PX, $PKG, $DEL, $CELLS. Safe to re-run: cells that already have rgcn_trained_v33 outputs are skipped.
$NB06 = "$PKG\notebooks\06_rgcn_leakage_safe_v3_trained.ipynb"; $NB09 = "$PKG\notebooks\09_xai_validation_leakage_safe_v33.ipynb"
if (-not (Select-String -Path $NB06 -Pattern "overlap_scale_v33" -Quiet)) { throw "Notebook 06 in $PKG\notebooks is not v3.3" }
if (-not (Test-Path $NB09)) { throw "Notebook 09 v3.3 not found at $NB09" }
$env:RGCN_MAX_EPOCHS = "2000"; $env:RGCN_PATIENCE = "100"; $env:RGCN_EVENT_ONLY_VARIANT = "1"; $env:RGCN_OVERLAP_SCALE_SOURCE = "train"; $env:RGCN_OUTPUT_SUBDIR = "rgcn_trained_v33"
foreach ($K in "0","1") { foreach ($SEED in 0,1,2,3,4,5,6,7,8,9) {
  $CELL = "$CELLS\cell_k${K}_s0_seed$SEED"; if (-not (Test-Path "$CELL\placement_v3.csv")) { Write-Host "missing $CELL"; continue }
  if (Test-Path "$CELL\data\processed\point_event_v2\rgcn_trained_v33\rgcn_predictions_v2.csv") { continue }
  $env:RGCN_RANDOM_STATE = "$(42 + $SEED)"; Set-Location $CELL; Write-Host "=== v3.3 R-GCN $CELL"
  python "$PKG\code\run_notebook_code.py" $NB06 --cwd $CELL
} }
python "$PKG\code\38_aggregate_rgcn_training_ablation_v3.py" --cells-root "$CELLS" --position-bank "$DEL\step2_coupling_study\position_bank_cross_regime_v3.csv" --trained-subdir "rgcn_trained_v33" --output-dir "$PX\rgcn_v33_aggregate"
foreach ($K in "0","1") {
  $XR = "$PX\xai_intensity_k${K}_seed0_v33"
  python "$PKG\code\39_stage_xai_on_trained_models_v3.py" --source-root "$CELLS\cell_k${K}_s0_seed0" --trained-rgcn-subdir "rgcn_trained_v33" --package-root "$PKG" --staging-root $XR
  Set-Location $XR; python "$PKG\code\run_notebook_code.py" $NB09 --cwd $XR
}
$COL = "$PX\rgcn_v33_results"; if (Test-Path $COL) { Remove-Item -Recurse -Force $COL }
Get-ChildItem "$CELLS" -Directory | ForEach-Object { $s = "$($_.FullName)\data\processed\point_event_v2\rgcn_trained_v33"; if (Test-Path $s) { New-Item -ItemType Directory -Force "$COL\$($_.Name)\rgcn_trained_v33" | Out-Null; Copy-Item "$s\*.csv" "$COL\$($_.Name)\rgcn_trained_v33"; Copy-Item "$s\*.json" "$COL\$($_.Name)\rgcn_trained_v33" } }
Copy-Item -Recurse "$PX\rgcn_v33_aggregate" "$COL\aggregate"
foreach ($K in "0","1") { $x = "$PX\xai_intensity_k${K}_seed0_v33\data\processed\point_event_v2\xai_leakage_safe_v2"; if (Test-Path $x) { Copy-Item -Recurse $x "$COL\xai_k${K}_v33" } }
Compress-Archive -Path "$COL\*" -DestinationPath "$PX\rgcn_v33_results.zip" -Force
Get-ChildItem "$CELLS" -Directory | ForEach-Object { $j = "$($_.FullName)\data\processed\point_event_v2\rgcn_trained_v33\overlap_scale_v33.json"; if (Test-Path $j) { $o = Get-Content $j | ConvertFrom-Json; "$($_.Name): source=$($o.scale_source) used=$([math]::Round($o.overlap_scale_used_s,3)) train=$([math]::Round($o.training_max_overlap_s,3)) global=$([math]::Round($o.global_max_overlap_s,3))" } }
