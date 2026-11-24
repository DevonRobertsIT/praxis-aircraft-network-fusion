#!/usr/bin/env python3
"""Parametric coupling study (tabular level) on the leakage-safe cross-regime layout.

Target   : attack intensity within attack-regime windows - y = 1 if the TRUE position's
           nf3_attack_flow_fraction >= the median over training events' true positions.
Score    : physical anomaly composite - mean over {altitude, ground speed, distance} of the
           train-only z-scored absolute deviation from the aircraft's own prior mean; for
           first-observed aircraft the train-only |z| of the raw value is used instead.
Coupling : within each partition's attack blocks, with probability kappa an event takes the
           position whose intensity rank matches its anomaly rank (exact rank matching, positions
           reusable); otherwise a uniformly random position. Blocks, embargoes, insets and
           partition ownership are never changed. The negative control couples to a SHUFFLED score.
Jitter   : the observed cyber window is the true position shifted by round(N(0, sigma) / 30 s)
           grid steps, clipped inside the same block. The target stays at the true position;
           physical features belong to the event and are never jittered.
Models   : LR and RF exactly as in 20_tabular_baselines_v3.py on physical / cyber / fused sets.
Inference: bin-cluster bootstrap over 240 s true-position bins; paired fused - cyber and
           fused - physical differences; RF impurity-importance share on the six score-bearing
           physical columns; realized Spearman(score, intensity) on train as the coupling check.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score

HERE = Path(__file__).resolve().parent
_s = importlib.util.spec_from_file_location("tab20", HERE / "20_tabular_baselines_v3.py")
tab = importlib.util.module_from_spec(_s); sys.modules["tab20"] = tab; _s.loader.exec_module(tab)

GRID_S = 30.0
BIN_STEPS = 8  # 240 s / 30 s
DEV_COLS = ["altitude_ft_deviation_from_prior_mean", "ground_speed_kt_deviation_from_prior_mean", "distance_miles_deviation_from_prior_mean"]
RAW_COLS = ["altitude_ft", "ground_speed_kt", "distance_miles"]
SCORE_COLS = DEV_COLS + RAW_COLS


def args_() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--alignment-dir", type=Path, required=True)
    p.add_argument("--position-bank", type=Path, required=True)
    p.add_argument("--feature-manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--kappas", default="0,0.1,0.25,0.5,0.75,1.0")
    p.add_argument("--sigmas", default="0,60,120,300")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--control-kappas", default="0.5,1.0")
    p.add_argument("--models", default="logistic_regression,random_forest")
    p.add_argument("--bootstrap", type=int, default=300)
    return p.parse_args()


def auprc(y, s):
    return float(average_precision_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def cluster_boot(stat, bins, rng, reps):
    u, inv = np.unique(bins, return_inverse=True); members = [np.flatnonzero(inv == k) for k in range(len(u))]
    out = np.empty(reps)
    for r in range(reps):
        idx = np.concatenate([members[k] for k in rng.integers(0, len(u), len(u))]); out[r] = stat(idx)
    return out


def anomaly_score(events: pd.DataFrame, train_mask: np.ndarray) -> np.ndarray:
    parts = []
    for dev, raw in zip(DEV_COLS, RAW_COLS):
        d = pd.to_numeric(events[dev], errors="coerce").abs(); r = pd.to_numeric(events[raw], errors="coerce")
        dm, ds = d[train_mask].mean(), d[train_mask].std(ddof=0) or 1.0
        rm, rs = r[train_mask].mean(), r[train_mask].std(ddof=0) or 1.0
        z = (d - dm) / ds
        z = z.where(z.notna(), ((r - rm) / rs).abs())  # first-observed aircraft: |z| of the raw value
        parts.append(z)
    return pd.concat(parts, axis=1).mean(axis=1, skipna=True).fillna(0.0).to_numpy()


def main() -> None:
    a = args_()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(a.feature_manifest.read_text())
    bank = pd.read_csv(a.position_bank)
    ev = pd.read_csv(a.alignment_dir / "event_source_block_mapping_v3.csv", low_memory=False)
    ev = ev[ev["source_block_id"].isin(bank["source_block_id"].unique())].reset_index(drop=True)  # attack-block events only
    if "track_degrees" in ev:
        rad = np.deg2rad(pd.to_numeric(ev["track_degrees"], errors="coerce")); ev["track_sin"], ev["track_cos"] = np.sin(rad), np.cos(rad)
    phys_num = [c for c in manifest["feature_sets"]["physical"]["numeric"] if c in ev.columns]
    phys_cat = [c for c in manifest["feature_sets"]["physical"]["categorical"] if c in ev.columns]
    cyb_num = [c for c in manifest["feature_sets"]["cyber"]["numeric"] if c in bank.columns]
    train_mask = ev["chronological_split"].eq("train").to_numpy()
    ev["anomaly_score"] = anomaly_score(ev, train_mask)
    print(f"attack-block events: {ev.groupby('chronological_split').size().to_dict()}; cyber features {len(cyb_num)}, physical {len(phys_num)}+{len(phys_cat)}", flush=True)

    # per-partition position tables sorted by intensity
    P = {}
    for part, g in bank.groupby("source_partition"):
        g = g.reset_index(drop=True)
        P[part] = {"frame": g, "order": np.argsort(g["attack_flow_fraction"].to_numpy(), kind="mergesort"),
                   "block_of": g["source_block_id"].to_numpy(), "pos_in_block": g["position_index"].to_numpy(),
                   "block_size": g.groupby("source_block_id")["position_index"].max().add(1).to_dict(),
                   "row_by_block_pos": {(b, k): i for i, (b, k) in enumerate(zip(g["source_block_id"], g["position_index"]))}}

    def place(part: str, scores: np.ndarray, kappa: float, sigma: float, rng: np.random.Generator, jitter_rng: np.random.Generator):
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

    kappas = [float(x) for x in a.kappas.split(",")]; sigmas = [float(x) for x in a.sigmas.split(",")]
    cells = [(k, s, seed, False) for s in sigmas for k in kappas for seed in range(a.seeds)]
    cells += [(k, 0.0, seed, True) for k in (float(x) for x in a.control_kappas.split(",")) for seed in range(a.seeds)]
    res_path, cell_path = a.output_dir / "coupling_grid_results_v3.csv", a.output_dir / "coupling_grid_cells_v3.csv"
    done = set()
    if res_path.exists():
        prev = pd.read_csv(res_path); done = set(zip(prev.kappa, prev.sigma, prev.seed, prev.control))
    t0 = time.time()
    for ci, (kappa, sigma, seed, control) in enumerate(cells):
        if (kappa, sigma, seed, control) in done:
            continue
        # v3.1: the placement stream is seeded WITHOUT sigma so that true positions, labels and the target
        # threshold are identical across jitter levels for a given (kappa, seed); jitter has its own stream.
        rng = np.random.default_rng(1000 * seed + int(kappa * 100) + (7 if control else 0))
        jitter_rng = np.random.default_rng(900_000 + 1000 * seed + int(kappa * 100) + int(sigma))
        parts = {}
        for part in ("train", "validation", "test"):
            e = ev[ev["chronological_split"].eq(part)].reset_index(drop=True)
            sc = e["anomaly_score"].to_numpy().copy()
            if control:
                sc = rng.permutation(sc)  # shuffled score: the coupling exists but not with the physical features
            tr, ob, cp = place(part, sc, kappa, sigma, rng, jitter_rng)
            frame = P[part]["frame"]
            d = e[["event_id", "chronological_split"] + phys_num + phys_cat + ["anomaly_score"]].copy()
            d["intensity_true"] = frame["attack_flow_fraction"].to_numpy()[tr]
            d["bin_240s"] = (frame["position_index"].to_numpy()[tr] // BIN_STEPS) + 100_000 * pd.factorize(frame["source_block_id"].to_numpy()[tr])[0]
            d["coupled"] = cp.astype(int)
            for c in cyb_num:
                d[c] = frame[c].to_numpy()[ob]
            parts[part] = d
        thr = float(np.median(parts["train"]["intensity_true"]))
        for part in parts:
            parts[part]["model_target"] = (parts[part]["intensity_true"] >= thr).astype(int)
        tr_, va_, te_ = parts["train"], parts["validation"], parts["test"]
        rho = float(spearmanr(tr_["anomaly_score"], tr_["intensity_true"]).correlation)
        y_te = te_["model_target"].to_numpy(); bins_te = te_["bin_240s"].to_numpy()
        cell_row = {"kappa": kappa, "sigma": sigma, "seed": seed, "control": control, "intensity_threshold": thr,
                    "train_spearman_score_intensity": rho, "test_prevalence": float(y_te.mean()), "test_events": len(te_),
                    "test_independent_bins": int(len(np.unique(bins_te))), "train_coupled_share": float(tr_["coupled"].mean())}
        scores_by = {}
        val_auprc = {}
        pred_rows = []
        rows = []
        for fs, (num, cat) in {"physical": (phys_num, phys_cat), "cyber": (cyb_num, []), "fused": (phys_num + cyb_num, phys_cat)}.items():
            num = [c for c in num if tr_[c].notna().any() and tr_[c].nunique(dropna=True) > 1]
            cat = [c for c in cat if tr_[c].nunique(dropna=True) > 1]
            for m in a.models.split(","):
                est, scale = tab.SPECS[m]
                est = est.set_params(random_state=42 + seed)
                pipe = tab.build(num, cat, est, scale); pipe.fit(tr_[num + cat], tr_["model_target"])
                pos = list(pipe.named_steps["model"].classes_).index(1)
                pv = pipe.predict_proba(va_[num + cat])[:, pos]; pt = pipe.predict_proba(te_[num + cat])[:, pos]
                thr_m = tab.select_threshold(va_["model_target"].to_numpy(), pv)
                scores_by[(fs, m)] = pt
                val_auprc[(fs, m)] = auprc(va_["model_target"].to_numpy(), pv)
                pred_rows.append(pd.DataFrame({"kappa": kappa, "sigma": sigma, "seed": seed, "control": control, "feature_set": fs, "model": m, "event_id": te_["event_id"].to_numpy(), "model_target": y_te, "bin_240s": bins_te, "p": pt}))
                cb = cluster_boot(lambda idx, pt=pt: auprc(y_te[idx], pt[idx]), bins_te, rng, a.bootstrap)
                share = float("nan")
                names = np.asarray(pipe.named_steps["preprocess"].get_feature_names_out(), dtype=str)
                if m == "random_forest":
                    imp = pipe.named_steps["model"].feature_importances_
                elif m == "logistic_regression":
                    imp = np.abs(pipe.named_steps["model"].coef_.ravel())
                else:
                    imp = None
                if imp is not None and imp.sum() > 0 and fs != "cyber":
                    is_score = np.array([any(n.endswith("__" + c) or ("__" + c + "_") in n or n.endswith(c) for c in SCORE_COLS) for n in names])
                    share = float(imp[is_score].sum() / imp.sum())
                phys_share = float("nan")
                if imp is not None and imp.sum() > 0 and fs == "fused":
                    is_phys = np.array([not any(("__" + c) in n for c in cyb_num) for n in names])
                    phys_share = float(imp[is_phys].sum() / imp.sum())
                rows.append({**cell_row, "feature_set": fs, "model": m, "test_auprc": auprc(y_te, pt), "validation_auprc": val_auprc[(fs, m)],
                             "bin_cluster_ci_low": float(np.nanpercentile(cb, 2.5)), "bin_cluster_ci_high": float(np.nanpercentile(cb, 97.5)),
                             "test_f1": float(f1_score(y_te, (pt >= thr_m).astype(int), zero_division=0)), "test_brier": float(brier_score_loss(y_te, pt)),
                             "score_feature_importance_share": share, "physical_block_importance_share": phys_share})
        for m in a.models.split(","):
            late = 0.5 * (scores_by[("physical", m)] + scores_by[("cyber", m)])  # late fusion: mean of unimodal probabilities, no extra fit
            scores_by[("late_fusion_mean", m)] = late
            cb = cluster_boot(lambda idx: auprc(y_te[idx], late[idx]), bins_te, rng, a.bootstrap)
            rows.append({**cell_row, "feature_set": "late_fusion_mean", "model": m, "test_auprc": auprc(y_te, late),
                         "bin_cluster_ci_low": float(np.nanpercentile(cb, 2.5)), "bin_cluster_ci_high": float(np.nanpercentile(cb, 97.5)),
                         "test_f1": float("nan"), "test_brier": float(brier_score_loss(y_te, late)),
                         "score_feature_importance_share": float("nan"), "physical_block_importance_share": float("nan")})
            for other in ("cyber", "physical"):
                sl, so = late, scores_by[(other, m)]
                d = auprc(y_te, sl) - auprc(y_te, so)
                db = cluster_boot(lambda idx: auprc(y_te[idx], sl[idx]) - auprc(y_te[idx], so[idx]), bins_te, rng, a.bootstrap)
                cell_row[f"{m}_late_minus_{other}"] = d
                cell_row[f"{m}_late_minus_{other}_ci_low"] = float(np.nanpercentile(db, 2.5)); cell_row[f"{m}_late_minus_{other}_ci_high"] = float(np.nanpercentile(db, 97.5))
            # H5 comparator: best unimodal model chosen on VALIDATION AUPRC (physical vs cyber, same estimator)
            best_uni = "physical" if val_auprc[("physical", m)] >= val_auprc[("cyber", m)] else "cyber"
            cell_row[f"{m}_best_unimodal_by_validation"] = best_uni
            for fusion, sfz in (("late", late), ("fused", scores_by[("fused", m)])):
                so = scores_by[(best_uni, m)]
                d = auprc(y_te, sfz) - auprc(y_te, so)
                db = cluster_boot(lambda idx, sfz=sfz, so=so: auprc(y_te[idx], sfz[idx]) - auprc(y_te[idx], so[idx]), bins_te, rng, a.bootstrap)
                cell_row[f"{m}_{fusion}_minus_best_unimodal"] = d
                cell_row[f"{m}_{fusion}_minus_best_unimodal_ci_low"] = float(np.nanpercentile(db, 2.5)); cell_row[f"{m}_{fusion}_minus_best_unimodal_ci_high"] = float(np.nanpercentile(db, 97.5))
            for other in ("cyber", "physical"):
                sf, so = scores_by[("fused", m)], scores_by[(other, m)]
                d = auprc(y_te, sf) - auprc(y_te, so)
                db = cluster_boot(lambda idx: auprc(y_te[idx], sf[idx]) - auprc(y_te[idx], so[idx]), bins_te, rng, a.bootstrap)
                cell_row[f"{m}_fused_minus_{other}"] = d
                cell_row[f"{m}_fused_minus_{other}_ci_low"] = float(np.nanpercentile(db, 2.5)); cell_row[f"{m}_fused_minus_{other}_ci_high"] = float(np.nanpercentile(db, 97.5))
        pd.DataFrame([{**r, **{k: v for k, v in cell_row.items() if k not in r}} for r in rows]).to_csv(res_path, mode="a", header=not res_path.exists(), index=False)
        pp = a.output_dir / "coupling_grid_test_predictions_v3.csv"
        pd.concat(pred_rows, ignore_index=True).to_csv(pp, mode="a", header=not pp.exists(), index=False)
        pd.DataFrame([cell_row]).to_csv(cell_path, mode="a", header=not cell_path.exists(), index=False)
        print(f"[{ci + 1}/{len(cells)}] kappa={kappa} sigma={sigma} seed={seed} control={control} rho={rho:+.3f} "
              f"cyberRF={[r['test_auprc'] for r in rows if r['feature_set'] == 'cyber' and r['model'] == 'random_forest'][0]:.3f} "
              f"fused-cyber RF={cell_row.get('random_forest_fused_minus_cyber', float('nan')):+.3f} "
              f"fused-phys RF={cell_row.get('random_forest_fused_minus_physical', float('nan')):+.3f}  ({time.time() - t0:.0f}s)", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
