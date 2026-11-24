#!/bin/bash
set -o pipefail
PR="/home/claude/work/Project/Jupyter Notebook"; ZIP=/mnt/user-data/uploads/Project.zip; OUT=/home/claude/work/v3_out
cd /home/claude/work/v2/Praxis_NF3_Leakage_Safe_Reanalysis_v2
run() { name=$1; shift; echo "### START $name $(date +%T)"; python3 code/11_nf3_leakage_safe_alignment_v3.py --project-root "$PR" --recovery-zip $ZIP --output-dir $OUT/$name "$@" > $OUT/$name.log 2>&1; echo "### END $name exit=$? $(date +%T)"; }
run alignment_v3_cross_regime --design cross_regime_s2_train_s1_attack_test --windows primary_only
run alignment_v3_within_session --design within_session_stratified --windows primary_only
run alignment_v3_cross_regime_snapshot_basis --design cross_regime_s2_train_s1_attack_test --windows primary_only --time-basis snapshot_time_utc
echo "### ALL DONE $(date +%T)"
