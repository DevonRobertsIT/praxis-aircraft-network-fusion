#!/usr/bin/env python3
"""Training-budget and relational-value ablation of the R-GCN (v3).

For each run root (a coupling cell root, or the Step 1 staging root) this script reads, on the SAME test events:
  transformer_v2/                   Transformer predictions (reference)
  rgcn_leakage_safe_v2/             original R-GCN run (<= 50 full-batch steps): full + lag/relation ablation
  <trained-subdir>/ (rgcn_trained_v3) patched notebook run with RGCN_MAX_EPOCHS=2000, RGCN_PATIENCE=100 and the
                                    event-only self-loop variant: full + ablation + event_only_self_loop
and reports AUPRC with bin-cluster intervals plus the paired differences that answer three questions:
  budget      trained full  - original full         does training the model change the verdict?
  relational  trained full  - trained event-only    do event-event and flow-event relations add value at all?
  H2/H2'      trained full  - Transformer           the original hypothesis, now on a trained model
  H3/H3'      trained full  - trained ablation      lag/relation weighting, now on a trained model
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
try:
    import tabulate  # noqa: F401  (required by DataFrame.to_markdown; see requirements-analysis.txt)
except ImportError:  # plain-text fallback so numerical outputs are never lost
    pd.DataFrame.to_markdown = lambda self, *args, **kwargs: self.to_string(index=kwargs.get("index", True))
from sklearn.metrics import average_precision_score

BIN_S, GRID_S = 240.0, 30.0
VARIANTS = {"full_lag_relation": "full", "lag_relation_ablation": "ablation", "event_only_self_loop": "event_only"}


def auprc(y, s):
    return float(average_precision_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def cluster_boot(stat, bins, rng, reps):
    u, inv = np.unique(bins, return_inverse=True); members = [np.flatnonzero(inv == k) for k in range(len(u))]
    out = np.empty(reps)
    for r in range(reps):
        idx = np.concatenate([members[k] for k in rng.integers(0, len(u), len(u))]); out[r] = stat(idx)
    return out


def rgcn_scores(d: Path, prefix: str) -> pd.DataFrame:
    r = pd.read_csv(d / "rgcn_predictions_v2.csv"); r = r[r["split"].eq("test")]
    p = r.pivot_table(index="event_id", columns="variant", values="probability").reset_index()
    return p.rename(columns={k: f"{prefix}_{v}" for k, v in VARIANTS.items()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells-root", type=Path, help="folder with cell_k*_s*_seed*/ roots")
    ap.add_argument("--position-bank", type=Path, help="position bank (bins for cells)")
    ap.add_argument("--step1-root", type=Path, help="Step 1 staging root (regime target)")
    ap.add_argument("--alignment-dir", type=Path, help="v3 alignment (bins for the Step 1 root)")
    ap.add_argument("--trained-subdir", default="rgcn_trained_v3")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--repeats", type=int, default=1000)
    a = ap.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    runs = []  # (label, base_dir, bins_by_event, meta)
    if a.step1_root:
        m = pd.read_csv(a.alignment_dir / "event_source_block_mapping_v3.csv", usecols=["event_id", "aligned_event_time_utc"])
        b = pd.to_datetime(m["aligned_event_time_utc"], utc=True).dt.tz_convert(None).to_numpy(dtype="datetime64[ns]").astype(np.int64) // int(BIN_S * 1e9)
        base = a.step1_root / "data/processed/point_event_v2"
        for td in sorted(glob.glob(str(base / f"{a.trained_subdir}*"))):
            seed = re.search(r"seed(\d+)", td); seed = int(seed.group(1)) if seed else -1
            tr_dir = base / (f"transformer_v3_seed{seed}" if (base / f"transformer_v3_seed{seed}").exists() else "transformer_v2")
            orig = base / (f"rgcn_v3_seed{seed}" if (base / f"rgcn_v3_seed{seed}").exists() else "rgcn_leakage_safe_v2")
            runs.append((f"step1_regime_seed{seed}", tr_dir, orig, Path(td), dict(zip(m["event_id"], b)), {"target": "regime", "kappa": np.nan, "sigma": np.nan, "seed": seed}))
    if a.cells_root:
        bank = pd.read_csv(a.position_bank); test_frame = bank[bank.source_partition.eq("test")].reset_index(drop=True)
        for cell in sorted(glob.glob(str(a.cells_root / "cell_k*"))):
            base = Path(cell) / "data/processed/point_event_v2"
            if not (base / a.trained_subdir / "rgcn_predictions_v2.csv").exists():
                continue
            mm = re.match(r"cell_k([\d.]+)_s([\d.]+)_seed(\d+)", os.path.basename(cell))
            pl = pd.read_csv(Path(cell) / "placement_v3.csv"); pl = pl[pl.chronological_split.eq("test")]
            bins = dict(zip(pl["event_id"], test_frame["position_index"].to_numpy()[pl["true_row"].to_numpy()] // int(BIN_S / GRID_S)))
            runs.append((os.path.basename(cell), base / "transformer_v2", base / "rgcn_leakage_safe_v2", base / a.trained_subdir, bins, {"target": "intensity", "kappa": float(mm.group(1)), "sigma": float(mm.group(2)), "seed": int(mm.group(3))}))
    if not runs:
        raise SystemExit("No trained R-GCN outputs found.")

    rows = []
    for label, tr_dir, orig_dir, trained_dir, bins_by_event, meta in runs:
        t = pd.read_csv(tr_dir / "temporal_transformer_predictions_v2.csv"); t = t[t["split"].eq("test")][["event_id", "model_target", "predicted_probability"]].rename(columns={"predicted_probability": "transformer"})
        d = t.merge(rgcn_scores(trained_dir, "trained"), on="event_id")
        if (orig_dir / "rgcn_predictions_v2.csv").exists():  # original (untrained) run is optional for extra seeds
            d = d.merge(rgcn_scores(orig_dir, "orig"), on="event_id", how="left")
        y = d["model_target"].to_numpy().astype(int); bins = d["event_id"].map(bins_by_event).to_numpy()
        S = {c: d[c].to_numpy() for c in d.columns if c not in ("event_id", "model_target")}
        rec = {"run": label, **meta, "test_events": len(d), "independent_240s_bins": int(len(np.unique(bins)))}
        for k, s in S.items():
            rec[f"auprc_{k}"] = auprc(y, s)
            cb = cluster_boot(lambda idx, s=s: auprc(y[idx], s[idx]), bins, rng, a.repeats)
            rec[f"auprc_{k}_ci_low"], rec[f"auprc_{k}_ci_high"] = float(np.nanpercentile(cb, 2.5)), float(np.nanpercentile(cb, 97.5))
        pairs = [("trained_full", "orig_full", "budget_trained_full_minus_orig_full"), ("trained_full", "trained_event_only", "relational_full_minus_event_only"),
                 ("trained_full", "transformer", "H2_trained_full_minus_transformer"), ("trained_full", "trained_ablation", "H3_trained_full_minus_ablation"),
                 ("trained_event_only", "transformer", "event_only_minus_transformer")]
        for x, z, name in pairs:
            if x in S and z in S:
                rec[name] = auprc(y, S[x]) - auprc(y, S[z])
                cb = cluster_boot(lambda idx: auprc(y[idx], S[x][idx]) - auprc(y[idx], S[z][idx]), bins, rng, a.repeats)
                rec[f"{name}_ci_low"], rec[f"{name}_ci_high"] = float(np.nanpercentile(cb, 2.5)), float(np.nanpercentile(cb, 97.5))
        rr = pd.read_csv(trained_dir / "rgcn_results_v2.csv").set_index("variant")
        for k, v in VARIANTS.items():
            if k in rr.index:
                rec[f"trained_{v}_epochs"] = int(rr.loc[k, "epochs_completed"]); rec[f"trained_{v}_p95_ms"] = float(rr.loc[k, "local_latency_p95_ms"]) if "local_latency_p95_ms" in rr.columns else np.nan
        rows.append(rec)
        print(f"{label}: T={rec['auprc_transformer']:.4f} origFull={rec.get('auprc_orig_full', np.nan):.4f} trainedFull={rec.get('auprc_trained_full', np.nan):.4f} trainedAbl={rec.get('auprc_trained_ablation', np.nan):.4f} eventOnly={rec.get('auprc_trained_event_only', np.nan):.4f} | budget {rec.get('budget_trained_full_minus_orig_full', np.nan):+.4f} relational {rec.get('relational_full_minus_event_only', np.nan):+.4f} H2 {rec.get('H2_trained_full_minus_transformer', np.nan):+.4f} H3 {rec.get('H3_trained_full_minus_ablation', np.nan):+.4f} epochs {rec.get('trained_full_epochs')}", flush=True)
    res = pd.DataFrame(rows); res.to_csv(a.output_dir / "rgcn_training_budget_ablation_runs_v3.csv", index=False)
    res = res.reindex(columns=sorted(set(res.columns), key=list(res.columns).index))  # stable column order
    for c in ("auprc_orig_full", "auprc_orig_full_ci_low", "auprc_orig_full_ci_high", "budget_trained_full_minus_orig_full", "budget_trained_full_minus_orig_full_ci_low", "budget_trained_full_minus_orig_full_ci_high"):
        if c not in res.columns:
            res[c] = np.nan
    keys = ["target", "sigma", "kappa"]
    diffs = [c for c in res.columns if c.startswith(("budget_", "relational_", "H2_", "H3_", "event_only_minus")) and not c.endswith(("_ci_low", "_ci_high"))]
    agg = res.groupby(keys, dropna=False).agg(**{f"{c}_mean": (c, "mean") for c in diffs}, **{f"{c}_seeds_ci_gt0": (c + "_ci_low", lambda s: int((s > 0).sum())) for c in diffs},
                                          **{f"{c}_seeds_ci_lt0": (c + "_ci_high", lambda s: int((s < 0).sum())) for c in diffs},
                                          transformer=("auprc_transformer", "mean"), orig_full=("auprc_orig_full", "mean") if "auprc_orig_full" in res.columns else ("auprc_transformer", "size"), trained_full=("auprc_trained_full", "mean"),
                                          trained_ablation=("auprc_trained_ablation", "mean"), trained_event_only=("auprc_trained_event_only", "mean"), seeds=("seed", "nunique")).reset_index()
    agg.to_csv(a.output_dir / "rgcn_training_budget_ablation_aggregate_v3.csv", index=False)
    md = ["# R-GCN training-budget and relational-value ablation", "",
          agg[keys + ["seeds", "transformer", "orig_full", "trained_full", "trained_ablation", "trained_event_only"]].round(4).to_markdown(index=False), "",
          "Paired differences (mean over seeds; counts of seeds whose bin-cluster 95% CI lies entirely above / below zero):", "",
          agg[keys + [c for c in agg.columns if c.endswith(("_mean", "_seeds_ci_gt0", "_seeds_ci_lt0"))]].round(4).to_markdown(index=False), ""]
    (a.output_dir / "RGCN_ABLATION_SUMMARY_v3.md").write_text("\n".join(md)); print("\n".join(md))


if __name__ == "__main__":
    main()
