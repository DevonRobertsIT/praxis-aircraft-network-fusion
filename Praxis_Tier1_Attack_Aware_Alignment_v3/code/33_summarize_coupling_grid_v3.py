#!/usr/bin/env python3
"""Summarize the coupling-injection grid (31_coupling_injection_grid_v3.py outputs)."""
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

p = argparse.ArgumentParser()
p.add_argument("--grid-dir", type=Path, required=True)
p.add_argument("--margin", type=float, default=0.02)
a = p.parse_args()
res = pd.read_csv(a.grid_dir / "coupling_grid_results_v3.csv")
cells = pd.read_csv(a.grid_dir / "coupling_grid_cells_v3.csv")
out = a.grid_dir

# ---------------------------------------------------------------- AUPRC surface (mean over seeds, min/max, mean cluster CI)
main = res[~res.control]
surf = main.groupby(["sigma", "kappa", "feature_set", "model"]).agg(
    auprc_mean=("test_auprc", "mean"), auprc_min=("test_auprc", "min"), auprc_max=("test_auprc", "max"),
    ci_low_mean=("bin_cluster_ci_low", "mean"), ci_high_mean=("bin_cluster_ci_high", "mean"), brier_mean=("test_brier", "mean"),
    score_share_mean=("score_feature_importance_share", "mean"), physical_share_mean=("physical_block_importance_share", "mean"), seeds=("seed", "nunique"),
).reset_index()
surf.to_csv(out / "coupling_auprc_surface_v3.csv", index=False)

# ---------------------------------------------------------------- paired gains over seeds
gain_rows = []
mc = cells[~cells.control]
for (sigma, kappa), g in mc.groupby(["sigma", "kappa"]):
    for m in ("random_forest", "logistic_regression"):
        for fusion in ("fused", "late"):
            for other in ("cyber", "physical"):
                col = f"{m}_{fusion}_minus_{other}"
                if col not in g:
                    continue
                d = g[col].to_numpy(); lo = g[col + "_ci_low"].to_numpy()
                gain_rows.append({"sigma": sigma, "kappa": kappa, "model": m, "fusion": fusion, "versus": other, "gain_mean": float(d.mean()),
                                  "gain_min": float(d.min()), "gain_max": float(d.max()), "cluster_ci_low_mean": float(lo.mean()),
                                  "seeds_ci_excluding_zero": int((lo > 0).sum()), "seeds": len(g),
                                  "detected": bool(d.mean() >= a.margin and (lo > 0).all())})
gains = pd.DataFrame(gain_rows); gains.to_csv(out / "coupling_fusion_gains_v3.csv", index=False)

# ---------------------------------------------------------------- minimum detectable coupling per (sigma, model, fusion, versus)
det = gains[gains.detected].groupby(["sigma", "model", "fusion", "versus"])["kappa"].min().rename("min_detectable_kappa").reset_index()
allk = gains.groupby(["sigma", "model", "fusion", "versus"]).size().reset_index()[["sigma", "model", "fusion", "versus"]]
det = allk.merge(det, how="left"); det["min_detectable_kappa"] = det["min_detectable_kappa"].fillna(np.inf)
det.to_csv(out / "coupling_minimum_detectable_kappa_v3.csv", index=False)

# ---------------------------------------------------------------- alignment-jitter sensitivity (H1')
h1 = surf[surf.feature_set.eq("cyber")].pivot_table(index=["model", "kappa"], columns="sigma", values="auprc_mean")
h1.to_csv(out / "coupling_cyber_auprc_vs_sigma_v3.csv")
base0 = surf[(surf.feature_set == "cyber") & (surf.kappa == 0)].set_index(["model", "sigma"])["auprc_mean"]
h1_rows = []
for m in ("random_forest", "logistic_regression"):
    for s in sorted(surf.sigma.unique()):
        if s == 0:
            continue
        loss = float(base0.get((m, 0.0), np.nan) - base0.get((m, s), np.nan))
        h1_rows.append({"model": m, "sigma": s, "cyber_auprc_loss_vs_sigma0_at_kappa0": loss, "loss_ge_0_02": bool(loss >= 0.02)})
h1_tab = pd.DataFrame(h1_rows); h1_tab.to_csv(out / "coupling_h1prime_alignment_sensitivity_v3.csv", index=False)

# ---------------------------------------------------------------- control
ctrl = cells[cells.control]
ctrl_rows = []
for kappa, g in ctrl.groupby("kappa"):
    for m in ("random_forest", "logistic_regression"):
        for fusion in ("fused", "late"):
            col = f"{m}_{fusion}_minus_cyber"
            if col in g:
                ctrl_rows.append({"kappa": kappa, "model": m, "fusion": fusion, "gain_over_cyber_mean": float(g[col].mean()),
                                  "cluster_ci_low_mean": float(g[col + "_ci_low"].mean()), "spearman_mean": float(g.train_spearman_score_intensity.mean()),
                                  "leak_detected": bool(g[col].mean() >= a.margin and (g[col + "_ci_low"] > 0).all())})
ctrl_tab = pd.DataFrame(ctrl_rows); ctrl_tab.to_csv(out / "coupling_negative_control_v3.csv", index=False)

# ---------------------------------------------------------------- attribution tracking
att = surf[(surf.sigma == 0) & surf.model.eq("random_forest") & surf.feature_set.isin(["physical", "fused"])][["kappa", "feature_set", "score_share_mean", "physical_share_mean"]]
att.to_csv(out / "coupling_attribution_vs_kappa_v3.csv", index=False)
rho = mc.groupby(["kappa"])["train_spearman_score_intensity"].agg(["mean", "min", "max"]).reset_index(); rho.to_csv(out / "coupling_realized_spearman_v3.csv", index=False)

# ---------------------------------------------------------------- markdown summary
def md(df, cols=None, r=3):
    d = df if cols is None else df[cols]
    return d.round(r).to_markdown(index=False)
lines = ["# Coupling-injection grid — summary (tabular level)", "",
         f"Cells: {len(mc)} main + {len(ctrl)} control; seeds per cell: {mc.seed.nunique()}; test events per cell: {int(mc.test_events.iloc[0])}; independent 240 s bins in test: {int(mc.test_independent_bins.median())}; test prevalence (median over cells): {mc.test_prevalence.median():.3f}", "",
         "## Realized coupling (train Spearman of anomaly score vs. true intensity)", md(rho), "",
         "## AUPRC surface at sigma = 0 (mean over seeds)", md(surf[surf.sigma == 0].pivot_table(index=["feature_set", "model"], columns="kappa", values="auprc_mean").reset_index()), "",
         "## Fusion gains at sigma = 0 (mean over seeds; detected = mean >= margin and every seed's bin-cluster CI excludes 0)", md(gains[gains.sigma == 0], ["kappa", "model", "fusion", "versus", "gain_mean", "gain_min", "gain_max", "cluster_ci_low_mean", "seeds_ci_excluding_zero", "detected"]), "",
         "## Minimum detectable coupling strength", md(det), "",
         "## H1' — cyber-only AUPRC vs. alignment jitter at kappa = 0", md(h1_tab), "",
         "## Late-fusion gain over cyber vs. jitter (mean over seeds)", md(gains[(gains.fusion == "late") & (gains.versus == "cyber")].pivot_table(index=["model", "kappa"], columns="sigma", values="gain_mean").reset_index()), "",
         "## Attribution tracking at sigma = 0 (RF importance share)", md(att), "",
         "## Negative control (coupling to a shuffled score)", md(ctrl_tab), ""]
(out / "COUPLING_GRID_SUMMARY_v3.md").write_text("\n".join(lines))
print("\n".join(lines))

# ---------------------------------------------------------------- figures (optional)
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    s0 = surf[surf.sigma == 0]
    for fs, style in (("physical", "-o"), ("cyber", "-s"), ("fused", "-^"), ("late_fusion_mean", "-d")):
        for m, c in (("random_forest", "tab:blue"), ("logistic_regression", "tab:orange")):
            g = s0[(s0.feature_set == fs) & (s0.model == m)].sort_values("kappa")
            axes[0].plot(g.kappa, g.auprc_mean, style, color=c, alpha=0.9 if m == "random_forest" else 0.5, label=f"{fs} {m.split('_')[0]}")
    axes[0].set(xlabel="coupling strength κ", ylabel="held-out AUPRC (mean of 5 seeds)", title="AUPRC vs κ (σ = 0)"); axes[0].legend(fontsize=7, ncol=2)
    for m, c in (("random_forest", "tab:blue"), ("logistic_regression", "tab:orange")):
        g = gains[(gains.sigma == 0) & (gains.model == m) & (gains.fusion == "late") & (gains.versus == "cyber")].sort_values("kappa")
        axes[1].errorbar(g.kappa, g.gain_mean, yerr=[g.gain_mean - g.gain_min, g.gain_max - g.gain_mean], fmt="-o", color=c, label=f"late fusion − cyber ({m.split('_')[0]})")
        g = gains[(gains.sigma == 0) & (gains.model == m) & (gains.fusion == "fused") & (gains.versus == "cyber")].sort_values("kappa")
        axes[1].plot(g.kappa, g.gain_mean, "--", color=c, alpha=0.6, label=f"concat fused − cyber ({m.split('_')[0]})")
    axes[1].axhline(a.margin, color="gray", ls=":"); axes[1].set(xlabel="κ", ylabel="ΔAUPRC", title="Fusion gain vs κ (σ = 0), seed min–max"); axes[1].legend(fontsize=7)
    for kappa, c in ((0.0, "gray"), (0.5, "tab:green"), (1.0, "tab:red")):
        g = surf[(surf.feature_set == "cyber") & (surf.model == "logistic_regression") & (surf.kappa == kappa)].sort_values("sigma")
        axes[2].plot(g.sigma, g.auprc_mean, "-o", color=c, label=f"cyber LR, κ={kappa}")
        g = surf[(surf.feature_set == "late_fusion_mean") & (surf.model == "logistic_regression") & (surf.kappa == kappa)].sort_values("sigma")
        axes[2].plot(g.sigma, g.auprc_mean, "--d", color=c, alpha=0.7, label=f"late fusion LR, κ={kappa}")
    axes[2].set(xlabel="alignment jitter σ (s)", ylabel="AUPRC", title="Alignment sensitivity"); axes[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "coupling_grid_figure_v3.png", dpi=150)
    print("figure written")
except Exception as e:  # matplotlib may be unavailable
    print("figure skipped:", e)
