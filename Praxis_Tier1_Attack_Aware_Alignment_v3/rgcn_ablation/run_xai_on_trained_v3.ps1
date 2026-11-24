# XAI validation on the FINAL checkpoints. Requires $PX/$PKG/$DEL/$ROOT/$CELLS.
# (a) regime target, trained R-GCN seed 42 + Transformer seed 42
python "$PKG\code\39_stage_xai_on_trained_models_v3.py" --source-root "$ROOT" --trained-rgcn-subdir "rgcn_trained_v3_seed42" --transformer-subdir "transformer_v3_seed42" --package-root "$PKG" --staging-root "$PX\xai_regime_trained"
& "$PX\xai_regime_trained\run_xai.ps1"
# (b) intensity target, kappa 0 and kappa 1, seed 0 (trained R-GCN + Transformer of that cell)
foreach ($K in "0","1") {
  python "$PKG\code\39_stage_xai_on_trained_models_v3.py" --source-root "$CELLS\cell_k${K}_s0_seed0" --package-root "$PKG" --staging-root "$PX\xai_intensity_k${K}_seed0"
  & "$PX\xai_intensity_k${K}_seed0\run_xai.ps1"
}
# collect: xai_leakage_safe_v2 folders (csv/json/md + analyst_cards) from the three staging roots
$COL = "$PX\xai_trained_results"; if (Test-Path $COL) { Remove-Item -Recurse -Force $COL }
foreach ($S in "xai_regime_trained","xai_intensity_k0_seed0","xai_intensity_k1_seed0") {
  $srcd = "$PX\$S\data\processed\point_event_v2\xai_leakage_safe_v2"; if (Test-Path $srcd) { Copy-Item -Recurse $srcd "$COL\$S" } }
Compress-Archive -Path "$COL\*" -DestinationPath "$PX\xai_trained_results.zip" -Force
