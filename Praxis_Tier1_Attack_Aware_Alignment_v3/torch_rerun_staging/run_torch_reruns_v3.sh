#!/bin/bash
# Transformer + R-GCN reruns on the v3 cross-regime alignment, five seeds each. Needs PyTorch.
set -e
ROOT="/home/claude/work/v3_out/torch_staging_root"
PKG="/home/claude/work/v2/Praxis_NF3_Leakage_Safe_Reanalysis_v2"
cd "$ROOT"
for SEED in 42 43 44 45 46; do
  echo "=== Transformer seed $SEED"; TT_RANDOM_STATE=$SEED python3 "$PKG/code/run_notebook_code.py" "$PKG/notebooks/05_temporal_transformer_leakage_safe_v2.ipynb" --cwd "$ROOT"
  echo "=== R-GCN seed $SEED";       RGCN_RANDOM_STATE=$SEED python3 "$PKG/code/run_notebook_code.py" "$PKG/notebooks/06_rgcn_leakage_safe_v2.ipynb" --cwd "$ROOT"
  rm -rf "data/processed/point_event_v2/transformer_v3_seed$SEED" "data/processed/point_event_v2/rgcn_v3_seed$SEED"
  cp -r data/processed/point_event_v2/transformer_v2 "data/processed/point_event_v2/transformer_v3_seed$SEED"
  cp -r data/processed/point_event_v2/rgcn_leakage_safe_v2 "data/processed/point_event_v2/rgcn_v3_seed$SEED"
done
echo "Outputs: data/processed/point_event_v2/{transformer,rgcn}_v3_seed*/ ; aggregate test_auprc across seeds (mean, min, max, bin-cluster CI)."
