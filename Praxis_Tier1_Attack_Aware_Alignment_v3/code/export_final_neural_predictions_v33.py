#!/usr/bin/env python3
"""Export per-event test predictions of every final neural run: 20 intensity cells (Transformer + final R-GCN variants,
plus the original-budget R-GCN where present) and 5 regime-target seeds. All locations are arguments; nothing is hard-coded.

Example (PowerShell, variables as in RUN_ME_FIRST.md):
  python "$PKG\\code\\export_final_neural_predictions_v33.py" --cells-root "$CELLS" --step1-root "$ROOT\\data\\processed\\point_event_v2" `
      --position-bank "$DEL\\step2_coupling_study\\position_bank_cross_regime_v3.csv" --alignment-dir "$DEL\\alignment_v3_cross_regime" `
      --output "$PX\\final_neural_test_predictions_v33.csv"
"""
import argparse, glob, os, re
import numpy as np, pandas as pd
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--cells-root", required=True, help="folder containing cell_k*_s0_seed*/ (each with placement_v3.csv and data/processed/point_event_v2/)")
ap.add_argument("--step1-root", required=True, help="folder containing transformer_v3_seed*/, rgcn_trained_v3_seed*/ and rgcn_v3_seed*/")
ap.add_argument("--position-bank", required=True); ap.add_argument("--alignment-dir", required=True)
ap.add_argument("--trained-subdir", default="rgcn_trained_v33", help="final R-GCN output folder inside each cell")
ap.add_argument("--regime-seeds", default="42,43,44,45,46"); ap.add_argument("--output", default="final_neural_test_predictions_v33.csv")
a = ap.parse_args()
bank = pd.read_csv(a.position_bank); test_frame = bank[bank.source_partition.eq("test")].reset_index(drop=True)
def rg(d, prefix=""):
    r = pd.read_csv(os.path.join(d, "rgcn_predictions_v2.csv")); r = r[r.split.eq("test")].pivot_table(index="event_id", columns="variant", values="probability")
    return r.rename(columns={"full_lag_relation": prefix + "rgcn_full", "lag_relation_ablation": prefix + "rgcn_ablation", "event_only_self_loop": prefix + "rgcn_event_only"})
def tr(d):
    t = pd.read_csv(os.path.join(d, "temporal_transformer_predictions_v2.csv")); t = t[t.split.eq("test")]
    return t.set_index("event_id")[["model_target", "predicted_probability"]].rename(columns={"predicted_probability": "transformer"})
cells = []
for c in sorted(glob.glob(os.path.join(a.cells_root, "cell_k*_s0_seed*"))):
    k, s = re.search(r"cell_k([\d.]+)_s0_seed(\d+)", os.path.basename(c)).groups(); b = os.path.join(c, "data", "processed", "point_event_v2")
    d = tr(os.path.join(b, "transformer_v2")).join(rg(os.path.join(b, a.trained_subdir)), how="inner")
    if os.path.exists(os.path.join(b, "rgcn_leakage_safe_v2", "rgcn_predictions_v2.csv")):
        d = d.join(rg(os.path.join(b, "rgcn_leakage_safe_v2"), "original_budget_")[["original_budget_rgcn_full", "original_budget_rgcn_ablation"]], how="left")
    pl = pd.read_csv(os.path.join(c, "placement_v3.csv")); pl = pl[pl.chronological_split.eq("test")].set_index("event_id")
    d["bin_240s"] = test_frame["position_index"].to_numpy()[pl.loc[d.index, "true_row"].to_numpy()] // 8
    d.insert(0, "seed", int(s)); d.insert(0, "kappa", float(k)); d.insert(0, "target", "intensity"); cells.append(d.reset_index())
m = pd.read_csv(os.path.join(a.alignment_dir, "event_source_block_mapping_v3.csv"), usecols=["event_id", "aligned_event_time_utc"])
m["bin_240s"] = pd.to_datetime(m.aligned_event_time_utc, utc=True).dt.tz_convert(None).to_numpy(dtype="datetime64[ns]").astype(np.int64) // int(240e9); bins = dict(zip(m.event_id, m.bin_240s))
reg = []
for s in (int(x) for x in a.regime_seeds.split(",")):
    b = a.step1_root
    d = tr(os.path.join(b, f"transformer_v3_seed{s}")).join(rg(os.path.join(b, f"rgcn_trained_v3_seed{s}")), how="inner").join(rg(os.path.join(b, f"rgcn_v3_seed{s}"), "original_budget_")[["original_budget_rgcn_full", "original_budget_rgcn_ablation"]], how="left")
    d["bin_240s"] = [bins[e] for e in d.index]; d.insert(0, "seed", s); d.insert(0, "kappa", np.nan); d.insert(0, "target", "regime"); reg.append(d.reset_index())
out = pd.concat(cells + reg, ignore_index=True); out.to_csv(a.output, index=False)
print(f"rows: {len(out)} | intensity cells: {len(cells)} | regime seeds: {len(reg)} -> {a.output}")
