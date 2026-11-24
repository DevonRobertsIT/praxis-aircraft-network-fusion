#!/usr/bin/env python3
"""Physical / cyber / fused tabular baselines on a v3 (attack-aware) alignment.

Replicates notebook 04 (leakage-safe v4): the same feature sets (read from
tabular_modeling_v4/feature_manifest_v4.json), training-only median imputation with
missingness indicators, one-hot categoricals, StandardScaler(with_mean=False) for
logistic regression only, the same four estimators and hyperparameters, seed 42, and
a validation-F1 threshold. Writes per-model test metrics and test predictions that the
session-baseline / cluster-bootstrap script consumes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

RANDOM_STATE = 42
SPECS = {
    "dummy_prior": (DummyClassifier(strategy="prior", random_state=RANDOM_STATE), False),
    "logistic_regression": (LogisticRegression(class_weight="balanced", max_iter=2_000, solver="liblinear", random_state=RANDOM_STATE), True),
    "decision_tree": (DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, class_weight="balanced", random_state=RANDOM_STATE), False),
    "random_forest": (RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=5, class_weight="balanced_subsample", n_jobs=1, random_state=RANDOM_STATE), False),
}


def args_() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--alignment-dir", type=Path, required=True)
    p.add_argument("--feature-manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--models", default="dummy_prior,logistic_regression,decision_tree,random_forest")
    p.add_argument("--feature-sets", default="physical,cyber,fused")
    p.add_argument("--target-basis-dir", type=Path, default=None, help="optional: score the test partition against this alignment's targets instead (H1 test)")
    return p.parse_args()


def select_threshold(y: np.ndarray, p: np.ndarray) -> float:
    if y.min() == y.max():
        return 0.5
    pr, rc, th = precision_recall_curve(y, p)
    if th.size == 0:
        return 0.5
    f1 = 2 * pr[:-1] * rc[:-1] / np.clip(pr[:-1] + rc[:-1], 1e-12, None)
    return float(th[int(np.argmax(f1))])


def build(numeric: list[str], categorical: list[str], est, scale: bool) -> Pipeline:
    tr = []
    if numeric:
        steps = [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
        if scale:
            steps.append(("scaler", StandardScaler(with_mean=False)))
        tr.append(("numeric", Pipeline(steps), numeric))
    if categorical:
        tr.append(("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("one_hot", OneHotEncoder(handle_unknown="ignore"))]), categorical))
    return Pipeline([("preprocess", ColumnTransformer(tr, remainder="drop", sparse_threshold=0.3, verbose_feature_names_out=True)), ("model", clone(est))])


def main() -> None:
    a = args_()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    (a.output_dir / "models").mkdir(exist_ok=True)
    manifest = json.loads(a.feature_manifest.read_text())
    bank = pd.read_csv(a.alignment_dir / "candidate_point_events_v3_with_context_and_nf3_primary.csv", low_memory=False)
    mapping = pd.read_csv(a.alignment_dir / "event_source_block_mapping_v3.csv", usecols=["event_id", "chronological_split", "partition_replay_cycle_index"])
    targets = pd.read_csv(a.alignment_dir / "event_nf3_targets_v3.csv")
    targets = targets[targets.window_name.eq("primary_120s")][["event_id", "cyber_attack_window_label"]]
    data = bank.merge(mapping, on="event_id").merge(targets, on="event_id", suffixes=("", "_target"))
    data["model_target"] = data["cyber_attack_window_label"].astype(int)
    if "track_degrees" in data:
        rad = np.deg2rad(pd.to_numeric(data["track_degrees"], errors="coerce"))
        data["track_sin"], data["track_cos"] = np.sin(rad), np.cos(rad)
    # optional alternate target basis for the test partition (H1: score snapshot-aligned features on event-time targets)
    if a.target_basis_dir is not None:
        alt = pd.read_csv(a.target_basis_dir / "event_nf3_targets_v3.csv")
        alt = alt[alt.window_name.eq("primary_120s")][["event_id", "cyber_attack_window_label"]].rename(columns={"cyber_attack_window_label": "alt_target"})
        data = data.merge(alt, on="event_id", how="left")
        data.loc[data.chronological_split.eq("test"), "model_target"] = data.loc[data.chronological_split.eq("test"), "alt_target"].astype(int)
    tr, va, te = (data[data.chronological_split.eq(s)].reset_index(drop=True) for s in ("train", "validation", "test"))
    print(f"rows train/val/test: {len(tr)}/{len(va)}/{len(te)}; positives {tr.model_target.sum()}/{va.model_target.sum()}/{te.model_target.sum()}", flush=True)

    rows, preds, rules = [], [], {}
    for fs in a.feature_sets.split(","):
        spec = manifest["feature_sets"][fs]
        numeric = [c for c in spec["numeric"] if c in data.columns]
        categorical = [c for c in spec["categorical"] if c in data.columns]
        # training-only screening: drop columns that are all-missing or constant in train
        numeric = [c for c in numeric if tr[c].notna().any() and tr[c].nunique(dropna=True) > 1]
        categorical = [c for c in categorical if tr[c].nunique(dropna=True) > 1]
        for mname in a.models.split(","):
            est, scale = SPECS[mname]
            pipe = build(numeric, categorical, est, scale)
            pipe.fit(tr[numeric + categorical], tr.model_target)
            pv = pipe.predict_proba(va[numeric + categorical])[:, list(pipe.named_steps["model"].classes_).index(1)]
            pt = pipe.predict_proba(te[numeric + categorical])[:, list(pipe.named_steps["model"].classes_).index(1)]
            thr = select_threshold(va.model_target.to_numpy(), pv)
            yt = te.model_target.to_numpy(); yhat = (pt >= thr).astype(int)
            both = 0 < yt.sum() < len(yt)
            rows.append({
                "feature_set": fs, "model": mname, "numeric_features": len(numeric), "categorical_features": len(categorical),
                "train_rows": len(tr), "validation_rows": len(va), "test_rows": len(te), "test_positives": int(yt.sum()),
                "selected_threshold": thr,
                "test_auprc": float(average_precision_score(yt, pt)) if both else np.nan,
                "test_roc_auc": float(roc_auc_score(yt, pt)) if both else np.nan,
                "test_f1": float(f1_score(yt, yhat, zero_division=0)), "test_precision": float(precision_score(yt, yhat, zero_division=0)),
                "test_recall": float(recall_score(yt, yhat, zero_division=0)), "test_brier_score": float(brier_score_loss(yt, pt)),
                "test_tp": int(((yhat == 1) & (yt == 1)).sum()), "test_fp": int(((yhat == 1) & (yt == 0)).sum()),
                "test_fn": int(((yhat == 0) & (yt == 1)).sum()), "test_tn": int(((yhat == 0) & (yt == 0)).sum()),
            })
            preds.append(pd.DataFrame({"event_id": te.event_id, "feature_set": fs, "model": mname, "model_target": yt, "predicted_probability": pt,
                                       "selected_threshold": thr, "source_block_id": te.source_block_id, "partition_replay_cycle_index": te.partition_replay_cycle_index,
                                       "aligned_event_time_utc": te.aligned_event_time_utc}))
            joblib.dump(pipe, a.output_dir / "models" / f"{fs}__{mname}.joblib")
            if mname == "decision_tree":
                names = pipe.named_steps["preprocess"].get_feature_names_out()
                rules[fs] = export_text(pipe.named_steps["model"], feature_names=list(names), max_depth=3)
            print(f"{fs:9s} {mname:20s} AUPRC={rows[-1]['test_auprc']:.6f} F1={rows[-1]['test_f1']:.4f} Brier={rows[-1]['test_brier_score']:.6f}", flush=True)
    res = pd.DataFrame(rows); res.to_csv(a.output_dir / "tabular_results_v3.csv", index=False)
    pd.concat(preds, ignore_index=True).to_csv(a.output_dir / "tabular_test_predictions_v3.csv", index=False)
    (a.output_dir / "decision_tree_rules_v3.txt").write_text("\n\n".join(f"### {k}\n{v}" for k, v in rules.items()))
    (a.output_dir / "tabular_build_summary_v3.json").write_text(json.dumps({
        "alignment_dir": str(a.alignment_dir), "feature_manifest": str(a.feature_manifest), "random_state": RANDOM_STATE,
        "target_basis_dir": str(a.target_basis_dir) if a.target_basis_dir else None,
        "rows": {"train": len(tr), "validation": len(va), "test": len(te)},
        "positives": {"train": int(tr.model_target.sum()), "validation": int(va.model_target.sum()), "test": int(te.model_target.sum())},
    }, indent=2))


if __name__ == "__main__":
    main()
