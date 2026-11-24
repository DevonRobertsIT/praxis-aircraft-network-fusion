# Close the seed question: materialize kappa 0/1 (sigma 0) cells for seeds 3-9, run the Transformer and the
# TRAINED R-GCN on each (the original untrained R-GCN is not needed). Safe to re-run: finished cells are skipped.
# Requires $PX/$PKG/$DEL/$ROOT/$CELLS defined and the patched notebook in $PKG\notebooks\.
$NB05 = "$PKG\notebooks\05_temporal_transformer_leakage_safe_v2.ipynb"
$NB06 = "$PKG\notebooks\06_rgcn_leakage_safe_v3_trained.ipynb"
foreach ($K in "0","1") { foreach ($SEED in 3,4,5,6,7,8,9) {
  $CELL = "$CELLS\cell_k${K}_s0_seed$SEED"
  if (-not (Test-Path "$CELL\placement_v3.csv")) {
    python "$PKG\code\35_materialize_coupling_cell_v3.py" --alignment-dir "$DEL\alignment_v3_cross_regime" --position-bank "$DEL\step2_coupling_study\position_bank_cross_regime_v3.csv" --position-edges "$DEL\step2_coupling_study\position_bank_edges_cross_regime_v3.csv" --package-root "$PKG" --grid-dir "$DEL\step2_coupling_study" --output-root "$CELLS" --kappa $K --sigma 0 --seed $SEED
    if (-not (Test-Path "$CELL\placement_v3.csv")) { Write-Host "materialization failed for $CELL"; continue }
  }
  Set-Location $CELL
  $OUTS = "$CELL\data\processed\point_event_v2"
  if (-not (Test-Path "$OUTS\transformer_v2\temporal_transformer_predictions_v2.csv")) {
    $env:TT_RANDOM_STATE = "$(42 + $SEED)"; Write-Host "=== Transformer $CELL"
    python "$PKG\code\run_notebook_code.py" $NB05 --cwd $CELL
  }
  if (-not (Test-Path "$OUTS\rgcn_trained_v3\rgcn_predictions_v2.csv")) {
    $env:RGCN_MAX_EPOCHS = "2000"; $env:RGCN_PATIENCE = "100"; $env:RGCN_EVENT_ONLY_VARIANT = "1"
    $env:RGCN_RANDOM_STATE = "$(42 + $SEED)"; $env:RGCN_OUTPUT_SUBDIR = "rgcn_trained_v3"; Write-Host "=== trained R-GCN $CELL"
    python "$PKG\code\run_notebook_code.py" $NB06 --cwd $CELL
  }
} }
# ---- collect seeds 0-9: transformer predictions/results + trained R-GCN csv/json (+ placement)
$COL = "$PX\rgcn_seeds_results"; if (Test-Path $COL) { Remove-Item -Recurse -Force $COL }
foreach ($K in "0","1") { foreach ($SEED in 0,1,2,3,4,5,6,7,8,9) {
  $CELL = "$CELLS\cell_k${K}_s0_seed$SEED"; if (-not (Test-Path "$CELL\placement_v3.csv")) { continue }
  $dst = "$COL\cell_k${K}_s0_seed$SEED"; New-Item -ItemType Directory -Force "$dst\transformer_v2", "$dst\rgcn_trained_v3" | Out-Null
  Copy-Item "$CELL\placement_v3.csv" $dst
  Copy-Item "$CELL\data\processed\point_event_v2\transformer_v2\*.csv" "$dst\transformer_v2"
  Copy-Item "$CELL\data\processed\point_event_v2\rgcn_trained_v3\*.csv" "$dst\rgcn_trained_v3"; Copy-Item "$CELL\data\processed\point_event_v2\rgcn_trained_v3\*.json" "$dst\rgcn_trained_v3"
  Remove-Item "$dst\rgcn_trained_v3\selected_event_flow_edges_v2.csv", "$dst\rgcn_trained_v3\rgcn_node_manifest_v2.csv" -ErrorAction SilentlyContinue
} }
Compress-Archive -Path "$COL\*" -DestinationPath "$PX\rgcn_seeds_results.zip" -Force
Get-ChildItem "$COL" -Recurse -Filter rgcn_predictions_v2.csv | Measure-Object | Select-Object Count   # expect 20
