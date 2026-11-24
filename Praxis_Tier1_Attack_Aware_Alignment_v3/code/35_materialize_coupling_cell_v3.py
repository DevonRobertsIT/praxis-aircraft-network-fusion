#!/usr/bin/env python3
"""Materialize a coupling-grid cell as inputs for notebooks 05 (Transformer) and 06 (R-GCN).

Reproduces the placement of 31_coupling_injection_grid_v3.py exactly (same RNG seed formula and
call order), then writes a staging root laid out the way the notebooks expect
(data/processed/point_event_v2/{nf3_alignment_v2, modeling_v4} with v2/v4 file names):
  - candidate_point_events_v2_with_context_and_nf3_primary.csv : attack-block events only, physical
    features from the event, cyber features from the OBSERVED (jittered) position
  - event_flow_edges_primary_bounded_v2.csv : the 16 label-blind nearest flows of the observed position
  - model_split_assignments_v4.csv : model_target = intensity label at the TRUE position
  - feature_manifest_v4.json, model_target_definition_v4.json (copied), baseline_model_results_v4.csv
    (this cell's tabular results from the grid, in the v4 schema, for the notebooks' comparison tables)
  - point_event_nf3_alignment_leakage_safe_v2_build_summary.json (v3 summary + cell metadata)
  - run_cell.ps1 / run_cell.sh : runs 05 then 06 with the cell's seed
Verification: the printed intensity_threshold, train Spearman and test prevalence must equal the
row for (kappa, sigma, seed, control) in coupling_grid_cells_v3.csv.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
_s = importlib.util.spec_from_file_location("grid31", HERE / "31_coupling_injection_grid_v3.py")
grid = importlib.util.module_from_spec(_s); sys.modules["grid31"] = grid; _s.loader.exec_module(grid)
GRID_S, BIN_STEPS, HALF_S = grid.GRID_S, grid.BIN_STEPS, 120.0


def args_() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--alignment-dir", type=Path, required=True)
    p.add_argument("--position-bank", type=Path, required=True)
    p.add_argument("--position-edges", type=Path, required=True)
    p.add_argument("--package-root", type=Path, required=True, help="extracted v2 handoff package")
    p.add_argument("--grid-dir", type=Path, required=True, help="folder with coupling_grid_results_v3.csv / coupling_grid_cells_v3.csv")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--kappa", type=float, required=True)
    p.add_argument("--sigma", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--control", action="store_true")
    return p.parse_args()


def main() -> None:
    a = args_()
    manifest = json.loads((a.package_root / "tabular_modeling_v4/feature_manifest_v4.json").read_text())
    bank = pd.read_csv(a.position_bank)
    pedges = pd.read_csv(a.position_edges)
    fused = pd.read_csv(a.alignment_dir / "candidate_point_events_v3_with_context_and_nf3_primary.csv", low_memory=False)
    ev = pd.read_csv(a.alignment_dir / "event_source_block_mapping_v3.csv", low_memory=False)
    ev = ev[ev["source_block_id"].isin(bank["source_block_id"].unique())].reset_index(drop=True)
    train_mask = ev["chronological_split"].eq("train").to_numpy()
    ev["anomaly_score"] = grid.anomaly_score(ev, train_mask)
    cyb_num = [c for c in manifest["feature_sets"]["cyber"]["numeric"] if c in bank.columns]

    # --- identical position tables and placement as the grid
    P = {}
    for part, g in bank.groupby("source_partition"):
        g = g.reset_index(drop=True)
        P[part] = {"frame": g, "order": np.argsort(g["attack_flow_fraction"].to_numpy(), kind="mergesort"),
                   "block_of": g["source_block_id"].to_numpy(), "pos_in_block": g["position_index"].to_numpy(),
                   "block_size": g.groupby("source_block_id")["position_index"].max().add(1).to_dict(),
                   "row_by_block_pos": {(b, k): i for i, (b, k) in enumerate(zip(g["source_block_id"], g["position_index"]))}}

    def place(part, scores, kappa, sigma, rng, jitter_rng):
        pt = P[part]; nP, nE = len(pt["frame"]), len(scores)
        coupled = rng.random(nE) < kappa
        q = (pd.Series(scores).rank(method="first").to_numpy() - 1) / max(nE - 1, 1)
        true_rows = np.where(coupled, pt["order"][np.round(q * (nP - 1)).astype(int)], rng.integers(0, nP, nE))
        if sigma > 0:
            shift = np.round(jitter_rng.normal(0, sigma, nE) / GRID_S).astype(int)
            obs_rows = np.empty(nE, dtype=int)
            for i, r in enumerate(true_rows):
                b, k = pt["block_of"][r], int(pt["pos_in_block"][r]); k2 = int(np.clip(k + shift[i], 0, pt["block_size"][b] - 1))
                obs_rows[i] = pt["row_by_block_pos"][(b, k2)]
        else:
            obs_rows = true_rows
        return true_rows, obs_rows, coupled

    rng = np.random.default_rng(1000 * a.seed + int(a.kappa * 100) + (7 if a.control else 0))  # v3.1: sigma-free placement stream
    jitter_rng = np.random.default_rng(900_000 + 1000 * a.seed + int(a.kappa * 100) + int(a.sigma))
    parts = {}
    for part in ("train", "validation", "test"):
        e = ev[ev["chronological_split"].eq(part)].reset_index(drop=True)
        sc = e["anomaly_score"].to_numpy().copy()
        if a.control:
            sc = rng.permutation(sc)
        tr, ob, cp = place(part, sc, a.kappa, a.sigma, rng, jitter_rng)
        frame = P[part]["frame"]
        d = pd.DataFrame({"event_id": e["event_id"], "chronological_split": part,
                          "true_row": tr, "obs_row": ob, "coupled": cp.astype(int),
                          "intensity_true": frame["attack_flow_fraction"].to_numpy()[tr],
                          "obs_block": frame["source_block_id"].to_numpy()[ob], "obs_position": frame["position_index"].to_numpy()[ob],
                          "obs_center_utc": frame["center_utc"].to_numpy()[ob], "anomaly_score": e["anomaly_score"]})
        for c in cyb_num:
            d[c] = frame[c].to_numpy()[ob]
        parts[part] = d
    thr = float(np.median(parts["train"]["intensity_true"]))
    for part in parts:
        parts[part]["model_target"] = (parts[part]["intensity_true"] >= thr).astype(int)
    placed = pd.concat(parts.values(), ignore_index=True)
    rho = float(spearmanr(parts["train"]["anomaly_score"], parts["train"]["intensity_true"]).correlation)
    print(json.dumps({"kappa": a.kappa, "sigma": a.sigma, "seed": a.seed, "control": a.control, "intensity_threshold": thr,
                      "train_spearman_score_intensity": rho, "test_prevalence": float(parts["test"]["model_target"].mean()),
                      "events": {k: len(v) for k, v in parts.items()}}, indent=1))
    ref = pd.read_csv(a.grid_dir / "coupling_grid_cells_v3.csv")
    ref = ref[(ref.kappa == a.kappa) & (ref.sigma == a.sigma) & (ref.seed == a.seed) & (ref.control == a.control)]
    if len(ref):
        r = ref.iloc[0]
        ok = abs(r.intensity_threshold - thr) < 1e-12 and abs(r.train_spearman_score_intensity - rho) < 1e-9 and abs(r.test_prevalence - parts["test"]["model_target"].mean()) < 1e-12
        print("matches grid cell record:", bool(ok))
        if not ok:
            raise SystemExit("Materialized placement does not reproduce the grid cell; aborting.")

    # --- staging root
    tag = f"cell_k{a.kappa:g}_s{a.sigma:g}_seed{a.seed}" + ("_control" if a.control else "")
    root = a.output_root / tag
    al, mo = root / "data/processed/point_event_v2/nf3_alignment_v2", root / "data/processed/point_event_v2/modeling_v4"
    for d in (al, mo):
        d.mkdir(parents=True, exist_ok=True)
    fb = fused[fused["event_id"].isin(placed["event_id"])].merge(placed[["event_id", "obs_block", "obs_position", "obs_center_utc", "intensity_true", "model_target"] + cyb_num], on="event_id", suffixes=("_prev", ""))
    fb = fb.drop(columns=[c for c in fb.columns if c.endswith("_prev")])
    fb["source_block_id"] = fb["obs_block"]; fb["aligned_event_time_utc"] = fb["obs_center_utc"]; fb["replay_cycle_index"] = 0
    ct = pd.to_datetime(fb["obs_center_utc"], utc=True)
    fb["window_start_utc"] = ct - pd.Timedelta(seconds=HALF_S); fb["window_end_utc"] = ct + pd.Timedelta(seconds=HALF_S)
    fb["alignment_design"] = f"coupling_cell:{tag}"; fb["alignment_mode"] = "position_bank_rank_coupling"
    fb = fb.drop(columns=["obs_block", "obs_position", "obs_center_utc", "intensity_true", "model_target"])
    fb.to_csv(al / "candidate_point_events_v2_with_context_and_nf3_primary.csv", index=False)

    v4 = pd.read_csv(a.package_root / "tabular_modeling_v4/model_split_assignments_v4.csv")
    sp = v4.drop(columns=["cyber_attack_window_label", "model_target"]).merge(placed[["event_id", "model_target"]], on="event_id", how="inner")
    sp["cyber_attack_window_label"] = sp["model_target"]
    sp = sp[["event_id", "aircraft_key", "aircraft_event_time_utc", "cyber_attack_window_label", "model_target", "chronological_split", "aircraft_group_split"]]
    sp.to_csv(mo / "model_split_assignments_v4.csv", index=False)
    tg = placed[["event_id", "chronological_split", "intensity_true", "coupled", "model_target"]].rename(columns={"model_target": "cyber_attack_window_label", "chronological_split": "source_partition"})
    tg["window_name"] = "primary_120s"; tg["intensity_threshold"] = thr
    tg.to_csv(al / "event_nf3_targets_v2.csv", index=False)

    ed = placed[["event_id", "chronological_split", "obs_block", "obs_position"]].merge(
        pedges, left_on=["obs_block", "obs_position"], right_on=["source_block_id", "position_index"], how="inner")
    ed = ed.drop(columns=["obs_block", "obs_position", "position_index", "source_partition"]).rename(columns={"chronological_split": "source_partition"})
    ed.to_csv(al / "event_flow_edges_primary_bounded_v2.csv", index=False)

    shutil.copy2(a.package_root / "tabular_modeling_v4/feature_manifest_v4.json", mo / "feature_manifest_v4.json")
    shutil.copy2(a.package_root / "tabular_modeling_v4/model_target_definition_v4.json", mo / "model_target_definition_v4.json")
    (mo / "intensity_target_definition_v3.json").write_text(json.dumps({"target": "nf3_attack_flow_fraction at the TRUE position >= training median", "threshold": thr, "cell": tag}, indent=1))
    res = pd.read_csv(a.grid_dir / "coupling_grid_results_v3.csv")
    res = res[(res.kappa == a.kappa) & (res.sigma == a.sigma) & (res.seed == a.seed) & (res.control == a.control) & res.feature_set.isin(["physical", "cyber", "fused"])]
    r4 = pd.read_csv(a.package_root / "tabular_modeling_v4/baseline_model_results_v4.csv")
    out = pd.DataFrame({c: res[c] if c in res.columns else pd.NA for c in r4.columns}); out["split_strategy"] = "chronological_primary"
    out["test_brier_score"] = res["test_brier"].values if "test_brier" in res else pd.NA
    out.to_csv(mo / "baseline_model_results_v4.csv", index=False)
    summ = json.loads((a.alignment_dir / "point_event_nf3_alignment_leakage_safe_v3_build_summary.json").read_text())
    summ["coupling_cell"] = {"tag": tag, "kappa": a.kappa, "sigma": a.sigma, "seed": a.seed, "control": a.control, "intensity_threshold": thr, "train_spearman": rho}
    summ["outputs"].update({"event_rows": int(len(fb)), "primary_materialized_bounded_edge_rows": int(len(ed)),
                            "primary_positive_windows": int(placed.model_target.sum()), "primary_negative_windows": int((placed.model_target == 0).sum()),
                            "primary_prevalence": float(placed.model_target.mean())})
    (al / "point_event_nf3_alignment_leakage_safe_v2_build_summary.json").write_text(json.dumps(summ, indent=2, default=str))
    placed.drop(columns=cyb_num).to_csv(root / "placement_v3.csv", index=False)

    pkg = a.package_root.resolve(); seed = 42 + a.seed
    (root / "run_cell.sh").write_text(f"""#!/bin/bash
set -e; ROOT="{root.resolve()}"; PKG="{pkg}"; cd "$ROOT"
TT_RANDOM_STATE={seed} python3 "$PKG/code/run_notebook_code.py" "$PKG/notebooks/05_temporal_transformer_leakage_safe_v2.ipynb" --cwd "$ROOT"
RGCN_RANDOM_STATE={seed} python3 "$PKG/code/run_notebook_code.py" "$PKG/notebooks/06_rgcn_leakage_safe_v2.ipynb" --cwd "$ROOT"
""")
    (root / "run_cell.ps1").write_text(f"""$ROOT = "{root.resolve()}"; $PKG = "{pkg}"; Set-Location $ROOT
$env:TT_RANDOM_STATE = "{seed}"; python "$PKG\\code\\run_notebook_code.py" "$PKG\\notebooks\\05_temporal_transformer_leakage_safe_v2.ipynb" --cwd "$ROOT"
$env:RGCN_RANDOM_STATE = "{seed}"; python "$PKG\\code\\run_notebook_code.py" "$PKG\\notebooks\\06_rgcn_leakage_safe_v2.ipynb" --cwd "$ROOT"
""")
    man = a.output_root / "cells_manifest.csv"
    pd.DataFrame([{"cell": tag, "kappa": a.kappa, "sigma": a.sigma, "seed": a.seed, "control": a.control, "events": len(fb), "edges": len(ed), "threshold": thr, "train_spearman": rho}]).to_csv(man, mode="a", header=not man.exists(), index=False)
    print("materialized:", root)


if __name__ == "__main__":
    main()
