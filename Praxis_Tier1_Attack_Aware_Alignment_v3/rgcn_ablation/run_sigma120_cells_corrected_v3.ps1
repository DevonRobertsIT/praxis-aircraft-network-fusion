# Re-materialize the sigma = 120 s cells under the corrected (sigma-free) placement and run Transformer + trained R-GCN (v3.2 notebook).
$NB05 = "$PKG\notebooks\05_temporal_transformer_leakage_safe_v2.ipynb"; $NB06 = "$PKG\notebooks\06_rgcn_leakage_safe_v3_trained.ipynb"
foreach ($K in "0","0.5","1.0") { foreach ($SEED in 0,1,2) {
  $CELL = "$CELLS\cell_k${K}_s120_seed$SEED"; if (Test-Path $CELL) { Remove-Item -Recurse -Force $CELL }
  python "$PKG\code\35_materialize_coupling_cell_v3.py" --alignment-dir "$DEL\alignment_v3_cross_regime" --position-bank "$DEL\step2_coupling_study\position_bank_cross_regime_v3.csv" --position-edges "$DEL\step2_coupling_study\position_bank_edges_cross_regime_v3.csv" --package-root "$PKG" --grid-dir "$DEL\review_corrections_v31" --output-root "$CELLS" --kappa $K --sigma 120 --seed $SEED
  Set-Location $CELL
  $env:TT_RANDOM_STATE = "$(42 + $SEED)"; python "$PKG\code\run_notebook_code.py" $NB05 --cwd $CELL
  $env:RGCN_MAX_EPOCHS = "2000"; $env:RGCN_PATIENCE = "100"; $env:RGCN_EVENT_ONLY_VARIANT = "1"; $env:RGCN_RANDOM_STATE = "$(42 + $SEED)"; $env:RGCN_OUTPUT_SUBDIR = "rgcn_trained_v3"
  python "$PKG\code\run_notebook_code.py" $NB06 --cwd $CELL
} }
