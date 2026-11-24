#!/usr/bin/env python3
"""Corrections after external review (v3.1 grid):
  (1) H1'(b): alignment jitter with FROZEN true positions, labels and threshold - paired AUPRC loss
      AUPRC(model, sigma=0) - AUPRC(model, sigma) on identical test events, bin-cluster bootstrap over the
      sigma=0 bins, per seed; seed-level t-interval.
  (2) H5: fusion gain against the BEST UNIMODAL model selected on validation AUPRC (physical vs cyber, same
      estimator), per cell with bin-cluster CI, aggregated over seeds; detection rule as pre-registered.
  (3) Label-freeze audit: number of test labels that differ between sigma=0 and sigma>0 cells (must be 0).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
try:
    import tabulate  # noqa: F401  (required by DataFrame.to_markdown; see requirements-analysis.txt)
except ImportError:  # plain-text fallback so numerical outputs are never lost
    pd.DataFrame.to_markdown = lambda self, *args, **kwargs: self.to_string(index=kwargs.get("index", True))
from scipy import stats
from sklearn.metrics import average_precision_score

p = argparse.ArgumentParser()
p.add_argument("--grid-dir", type=Path, required=True)
p.add_argument("--margin", type=float, default=0.02)
p.add_argument("--repeats", type=int, default=1000)
a = p.parse_args()
rng = np.random.default_rng(42)
pred = pd.read_csv(a.grid_dir / "coupling_grid_test_predictions_v3.csv")
cells = pd.read_csv(a.grid_dir / "coupling_grid_cells_v3.csv")
res = pd.read_csv(a.grid_dir / "coupling_grid_results_v3.csv")
pred = pred[~pred.control]


def auprc(y, s):
    return float(average_precision_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def cluster_boot(stat, bins, reps):
    u, inv = np.unique(bins, return_inverse=True); members = [np.flatnonzero(inv == k) for k in range(len(u))]
    out = np.empty(reps)
    for r in range(reps):
        idx = np.concatenate([members[k] for k in rng.integers(0, len(u), len(u))]); out[r] = stat(idx)
    return out


# ---------------------------------------------------------------- (3) label-freeze audit and (1) paired jitter loss
freeze_rows, loss_rows = [], []
for (kappa, seed), g in pred.groupby(["kappa", "seed"]):
    base = g[g.sigma == 0]
    if base.empty:
        continue
    for sigma, gs in g[g.sigma > 0].groupby("sigma"):
        b0 = base[(base.feature_set == "cyber") & (base.model == "random_forest")].set_index("event_id")
        s0 = gs[(gs.feature_set == "cyber") & (gs.model == "random_forest")].set_index("event_id").reindex(b0.index)
        freeze_rows.append({"kappa": kappa, "seed": seed, "sigma": sigma, "events": len(b0), "labels_differ": int((b0.model_target.to_numpy() != s0.model_target.to_numpy()).sum()), "bins_differ": int((b0.bin_240s.to_numpy() != s0.bin_240s.to_numpy()).sum())})
        for fs, m in [("cyber", "logistic_regression"), ("cyber", "random_forest"), ("late_fusion_mean", "logistic_regression"), ("late_fusion_mean", "random_forest")]:
            x0 = base[(base.feature_set == fs) & (base.model == m)].set_index("event_id"); xs = gs[(gs.feature_set == fs) & (gs.model == m)].set_index("event_id").reindex(x0.index)
            if x0.empty or xs.p.isna().any():
                continue
            y = x0.model_target.to_numpy().astype(int); bins = x0.bin_240s.to_numpy(); p0, ps = x0.p.to_numpy(), xs.p.to_numpy()
            loss = auprc(y, p0) - auprc(y, ps)
            cb = cluster_boot(lambda idx: auprc(y[idx], p0[idx]) - auprc(y[idx], ps[idx]), bins, a.repeats)
            loss_rows.append({"kappa": kappa, "seed": seed, "sigma": sigma, "feature_set": fs, "model": m, "auprc_sigma0": auprc(y, p0), "auprc_sigma": auprc(y, ps), "loss": loss, "ci_low": float(np.nanpercentile(cb, 2.5)), "ci_high": float(np.nanpercentile(cb, 97.5))})
freeze = pd.DataFrame(freeze_rows); loss = pd.DataFrame(loss_rows)
freeze.to_csv(a.grid_dir / "h1prime_label_freeze_audit_v31.csv", index=False); loss.to_csv(a.grid_dir / "h1prime_paired_jitter_loss_runs_v31.csv", index=False)
agg_rows = []
for (kappa, sigma, fs, m), g in loss.groupby(["kappa", "sigma", "feature_set", "model"]):
    d = g["loss"].to_numpy(); n = len(d); t = stats.t.ppf(0.975, n - 1) if n > 1 else np.nan; se = d.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    agg_rows.append({"kappa": kappa, "sigma": sigma, "feature_set": fs, "model": m, "seeds": n, "loss_mean": d.mean(), "t_ci_low": d.mean() - t * se, "t_ci_high": d.mean() + t * se,
                     "seeds_ci_gt0": int((g.ci_low > 0).sum()), "seeds_ci_lt0": int((g.ci_high < 0).sum()), "supported_H1b": bool(d.mean() >= a.margin and (g.ci_low > 0).all())})
loss_agg = pd.DataFrame(agg_rows); loss_agg.to_csv(a.grid_dir / "h1prime_paired_jitter_loss_aggregate_v31.csv", index=False)

# ---------------------------------------------------------------- (2) H5 against best unimodal (validation-selected)
c = cells[~cells.control]
h5_rows = []
for (sigma, kappa), g in c.groupby(["sigma", "kappa"]):
    for m in ("random_forest", "logistic_regression"):
        for fusion in ("late", "fused"):
            col = f"{m}_{fusion}_minus_best_unimodal"
            if col not in g:
                continue
            d = g[col].to_numpy(); lo = g[col + "_ci_low"].to_numpy(); hi = g[col + "_ci_high"].to_numpy()
            h5_rows.append({"sigma": sigma, "kappa": kappa, "model": m, "fusion": fusion, "best_unimodal_by_validation": "|".join(sorted(set(g[f"{m}_best_unimodal_by_validation"]))),
                            "gain_mean": d.mean(), "gain_min": d.min(), "gain_max": d.max(), "seeds_ci_gt0": int((lo > 0).sum()), "seeds_ci_lt0": int((hi < 0).sum()), "seeds": len(g),
                            "detected": bool(d.mean() >= a.margin and (lo > 0).all())})
h5 = pd.DataFrame(h5_rows); h5.to_csv(a.grid_dir / "h5_fusion_vs_best_unimodal_v31.csv", index=False)
# also fused - cyber (operational question) for reference
ref_rows = []
for (sigma, kappa), g in c.groupby(["sigma", "kappa"]):
    for m in ("random_forest", "logistic_regression"):
        for fusion in ("late", "fused"):
            col = f"{m}_{fusion}_minus_cyber"; d = g[col].to_numpy(); lo = g[col + "_ci_low"].to_numpy()
            ref_rows.append({"sigma": sigma, "kappa": kappa, "model": m, "fusion": fusion, "gain_over_cyber_mean": d.mean(), "seeds_ci_gt0": int((lo > 0).sum()), "detected_vs_cyber": bool(d.mean() >= a.margin and (lo > 0).all())})
pd.DataFrame(ref_rows).to_csv(a.grid_dir / "h5_fusion_vs_cyber_reference_v31.csv", index=False)

md = ["# Review corrections (v3.1 grid)", "",
      "## Label-freeze audit (test labels / bins differing between sigma = 0 and sigma > 0, same kappa and seed)", freeze.groupby(["kappa", "sigma"])[["labels_differ", "bins_differ"]].max().reset_index().to_markdown(index=False), "",
      "## H1'(b): paired AUPRC loss with frozen labels (seed mean, t-interval; seeds whose bin-cluster CI > 0)", loss_agg[loss_agg.kappa.isin([0.0, 1.0])].round(4).to_markdown(index=False), "",
      "## H5: fusion minus best unimodal model (selected on validation), sigma = 0", h5[h5.sigma == 0].round(4).to_markdown(index=False), "",
      "## Reference: fusion minus cyber-only (operational question), sigma = 0", pd.DataFrame(ref_rows).query("sigma == 0").round(4).to_markdown(index=False), ""]
(a.grid_dir / "REVIEW_CORRECTIONS_v31.md").write_text("\n".join(md)); print("\n".join(md))
