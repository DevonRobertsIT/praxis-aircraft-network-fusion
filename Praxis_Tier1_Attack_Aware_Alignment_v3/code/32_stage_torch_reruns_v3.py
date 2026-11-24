#!/usr/bin/env python3
"""Stage the Transformer (05) and R-GCN (06) reruns on a v3 alignment without editing the notebooks.

Both notebooks resolve PROJECT_ROOT = Path.cwd() and read v2-named files under
data/processed/point_event_v2/{nf3_alignment_v2, modeling_v4}. This script builds a staging root
whose v2-named entries point at the v3 artifacts, writes the v3 split/target file in the v4 schema,
a v4-schema copy of the v3 tabular results (for the notebooks' comparison tables), and a runner
that executes each notebook over five seeds via the TT_RANDOM_STATE / RGCN_RANDOM_STATE variables.
Run the runner on a machine with PyTorch: bash <staging_root>/run_torch_reruns_v3.sh
"""
import argparse, json, os, shutil
from pathlib import Path
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--alignment-dir", type=Path, required=True)
p.add_argument("--tabular-v3-dir", type=Path, required=True)
p.add_argument("--package-root", type=Path, required=True, help="the v2 handoff package (for feature_manifest_v4.json etc.)")
p.add_argument("--staging-root", type=Path, required=True)
p.add_argument("--copy", action="store_true", help="copy files instead of symlinking (for transfer to another machine)")
a = p.parse_args()

pe = a.staging_root / "data/processed/point_event_v2"
al, mo = pe / "nf3_alignment_v2", pe / "modeling_v4"
for d in (al, mo): d.mkdir(parents=True, exist_ok=True)
def put(src: Path, dst: Path):
    if dst.exists() or dst.is_symlink(): dst.unlink()
    (shutil.copy2 if a.copy else os.symlink)(src.resolve(), dst)
links = {
    al / "candidate_point_events_v2_with_context_and_nf3_primary.csv": a.alignment_dir / "candidate_point_events_v3_with_context_and_nf3_primary.csv",
    al / "event_flow_edges_primary_bounded_v2.csv": a.alignment_dir / "event_flow_edges_primary_bounded_v3.csv",
    al / "event_nf3_targets_v2.csv": a.alignment_dir / "event_nf3_targets_v3.csv",
    al / "event_source_block_mapping_v2.csv": a.alignment_dir / "event_source_block_mapping_v3.csv",
    al / "point_event_nf3_alignment_leakage_safe_v2_build_summary.json": a.alignment_dir / "point_event_nf3_alignment_leakage_safe_v3_build_summary.json",
    mo / "feature_manifest_v4.json": a.package_root / "tabular_modeling_v4/feature_manifest_v4.json",
    mo / "model_target_definition_v4.json": a.package_root / "tabular_modeling_v4/model_target_definition_v4.json",
}
for dst, src in links.items():
    assert src.exists(), src; put(src, dst)

# split/target file in the v4 schema with the v3 (cross-regime) target
v4 = pd.read_csv(a.package_root / "tabular_modeling_v4/model_split_assignments_v4.csv")
t = pd.read_csv(a.alignment_dir / "event_nf3_targets_v3.csv"); t = t[t.window_name.eq("primary_120s")][["event_id", "cyber_attack_window_label"]]
s = v4.drop(columns=["cyber_attack_window_label", "model_target"]).merge(t, on="event_id", how="left", validate="one_to_one")
s["model_target"] = s["cyber_attack_window_label"].astype(int)
s = s[["event_id", "aircraft_key", "aircraft_event_time_utc", "cyber_attack_window_label", "model_target", "chronological_split", "aircraft_group_split"]]
s.to_csv(mo / "model_split_assignments_v4.csv", index=False)

# v3 tabular results in the v4 results schema (only for the notebooks' comparison tables)
r3 = pd.read_csv(a.tabular_v3_dir / "tabular_results_v3.csv"); r4 = pd.read_csv(a.package_root / "tabular_modeling_v4/baseline_model_results_v4.csv")
r3["split_strategy"] = "chronological_primary"
out = pd.DataFrame({c: r3[c] if c in r3.columns else pd.NA for c in r4.columns}); out.to_csv(mo / "baseline_model_results_v4.csv", index=False)

nb = a.package_root / "notebooks"; runner = a.staging_root / "run_torch_reruns_v3.sh"
runner.write_text(f"""#!/bin/bash
# Transformer + R-GCN reruns on the v3 cross-regime alignment, five seeds each. Needs PyTorch.
set -e
ROOT="{a.staging_root.resolve()}"
PKG="{a.package_root.resolve()}"
cd "$ROOT"
for SEED in 42 43 44 45 46; do
  echo "=== Transformer seed $SEED"; TT_RANDOM_STATE=$SEED python3 "$PKG/code/run_notebook_code.py" "$PKG/notebooks/05_temporal_transformer_leakage_safe_v2.ipynb" --cwd "$ROOT"
  echo "=== R-GCN seed $SEED";       RGCN_RANDOM_STATE=$SEED python3 "$PKG/code/run_notebook_code.py" "$PKG/notebooks/06_rgcn_leakage_safe_v2.ipynb" --cwd "$ROOT"
  rm -rf "data/processed/point_event_v2/transformer_v3_seed$SEED" "data/processed/point_event_v2/rgcn_v3_seed$SEED"
  cp -r data/processed/point_event_v2/transformer_v2 "data/processed/point_event_v2/transformer_v3_seed$SEED"
  cp -r data/processed/point_event_v2/rgcn_leakage_safe_v2 "data/processed/point_event_v2/rgcn_v3_seed$SEED"
done
echo "Outputs: data/processed/point_event_v2/{{transformer,rgcn}}_v3_seed*/ ; aggregate test_auprc across seeds (mean, min, max, bin-cluster CI)."
""")
runner.chmod(0o755)
print(json.dumps({"staging_root": str(a.staging_root), "links": {str(k.relative_to(a.staging_root)): str(v) for k, v in links.items()},
                  "split_file_positives": int(s.model_target.sum()), "runner": str(runner)}, indent=1))
