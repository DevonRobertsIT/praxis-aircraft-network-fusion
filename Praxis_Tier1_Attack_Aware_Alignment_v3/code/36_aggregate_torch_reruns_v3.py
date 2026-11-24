#!/usr/bin/env python3
"""Aggregate Transformer / R-GCN reruns (Step 1 seeds and, when present, Step 2 coupling cells).

Step 1 (--results-root): folders transformer_v3_seed*/ and rgcn_v3_seed*/ produced by notebooks 05/06 on the
cross-regime alignment. For every seed the script recomputes test AUPRC from the saved per-event predictions,
forms paired differences on the SAME test events (R-GCN full - Transformer = H2; full - ablation = H3; each vs.
the tabular cyber RF), and bootstraps them over 240 s source-time bins (the independent unit of this design).
Decision rule per hypothesis: seed-mean >= margin AND every seed's bin-cluster interval excludes zero.

Step 2 (--cells-root): folders cell_k{K}_s{S}_seed{N}/ each holding placement_v3.csv and the two notebook output
folders; the same statistics per cell (H2', H3') plus aggregation over seeds per (kappa, sigma).
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

BIN_S = 240.0
GRID_S = 30.0


def auprc(y, s):
    return float(average_precision_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def cluster_boot(stat, bins, rng, reps):
    u, inv = np.unique(bins, return_inverse=True); members = [np.flatnonzero(inv == k) for k in range(len(u))]
    out = np.empty(reps)
    for r in range(reps):
        idx = np.concatenate([members[k] for k in rng.integers(0, len(u), len(u))]); out[r] = stat(idx)
    return out


def load_test_scores(tr_dir: Path, rg_dir: Path) -> pd.DataFrame:
    t = pd.read_csv(tr_dir / "temporal_transformer_predictions_v2.csv")
    t = t[t["split"].eq("test")][["event_id", "model_target", "predicted_probability"]].rename(columns={"predicted_probability": "transformer"})
    r = pd.read_csv(rg_dir / "rgcn_predictions_v2.csv")
    r = r[r["split"].eq("test")].pivot_table(index="event_id", columns="variant", values="probability").reset_index()
    r = r.rename(columns={"full_lag_relation": "rgcn_full", "lag_relation_ablation": "rgcn_ablation"})
    d = t.merge(r, on="event_id", how="inner")
    if len(d) != len(t):
        raise ValueError(f"test event sets differ between Transformer ({len(t)}) and R-GCN ({len(r)}) in {tr_dir}")
    return d


def paired_stats(d: pd.DataFrame, bins: np.ndarray, rng, reps, extra: dict[str, np.ndarray] | None = None) -> dict:
    y = d["model_target"].to_numpy().astype(int)
    S = {"transformer": d["transformer"].to_numpy(), "rgcn_full": d["rgcn_full"].to_numpy(), "rgcn_ablation": d["rgcn_ablation"].to_numpy()}
    if extra:
        S.update(extra)
    out = {"test_events": len(d), "test_positives": int(y.sum()), "independent_240s_bins": int(len(np.unique(bins)))}
    for k, s in S.items():
        out[f"auprc_{k}"] = auprc(y, s)
        cb = cluster_boot(lambda idx, s=s: auprc(y[idx], s[idx]), bins, rng, reps)
        out[f"auprc_{k}_ci_low"], out[f"auprc_{k}_ci_high"] = float(np.nanpercentile(cb, 2.5)), float(np.nanpercentile(cb, 97.5))
    pairs = [("rgcn_full", "transformer", "H2_rgcn_minus_transformer"), ("rgcn_full", "rgcn_ablation", "H3_full_minus_ablation")]
    if extra:
        pairs += [("rgcn_full", k, f"rgcn_minus_{k}") for k in extra] + [("transformer", k, f"transformer_minus_{k}") for k in extra]
    for a_, b_, name in pairs:
        sa, sb = S[a_], S[b_]
        out[name] = auprc(y, sa) - auprc(y, sb)
        cb = cluster_boot(lambda idx: auprc(y[idx], sa[idx]) - auprc(y[idx], sb[idx]), bins, rng, reps)
        out[f"{name}_ci_low"], out[f"{name}_ci_high"] = float(np.nanpercentile(cb, 2.5)), float(np.nanpercentile(cb, 97.5))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", type=Path, help="folder with transformer_v3_seed*/ and rgcn_v3_seed*/")
    p.add_argument("--alignment-dir", type=Path, help="v3 alignment (for 240 s bins of Step 1 test events)")
    p.add_argument("--tabular-predictions", type=Path, help="tabular_test_predictions_v3.csv for the same alignment (optional)")
    p.add_argument("--cells-root", type=Path, help="folder with cell_k*_s*_seed*/ (optional)")
    p.add_argument("--position-bank", type=Path, help="position bank (needed with --cells-root for bins)")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=1000)
    p.add_argument("--h2-margin", type=float, default=0.03)
    p.add_argument("--h3-margin", type=float, default=0.05)
    p.add_argument("--latency-limit-ms", type=float, default=500.0)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    md = []

    # ------------------------------------------------------------------ Step 1: seeds on the regime target
    if a.results_root:
        m = pd.read_csv(a.alignment_dir / "event_source_block_mapping_v3.csv", usecols=["event_id", "aligned_event_time_utc"])
        m["bin"] = pd.to_datetime(m["aligned_event_time_utc"], utc=True).dt.tz_convert(None).to_numpy(dtype="datetime64[ns]").astype(np.int64) // int(BIN_S * 1e9)
        bins_by_event = dict(zip(m["event_id"], m["bin"]))
        extra_by_event = None
        if a.tabular_predictions and a.tabular_predictions.exists():
            tp = pd.read_csv(a.tabular_predictions)
            extra_by_event = {f"{fs}_rf": tp[(tp.feature_set == fs) & (tp.model == "random_forest")].set_index("event_id")["predicted_probability"] for fs in ("cyber", "fused")}
        rows = []
        for tr_dir in sorted(glob.glob(str(a.results_root / "transformer_v3_seed*"))):
            seed = re.search(r"seed(\d+)", tr_dir).group(1)
            rg_dir = a.results_root / f"rgcn_v3_seed{seed}"
            d = load_test_scores(Path(tr_dir), rg_dir)
            bins = d["event_id"].map(bins_by_event).to_numpy()
            extra = {k: v.reindex(d["event_id"]).to_numpy() for k, v in extra_by_event.items()} if extra_by_event else None
            st = paired_stats(d, bins, rng, a.repeats, extra)
            rr = pd.read_csv(rg_dir / "rgcn_results_v2.csv").set_index("variant")
            st.update({"seed": int(seed), "rgcn_full_local_latency_p95_ms": float(rr.loc["full_lag_relation", "local_latency_p95_ms"]),
                       "reported_transformer_auprc": float(pd.read_csv(Path(tr_dir) / "temporal_transformer_results_v2.csv")["test_auprc"].iloc[0]),
                       "reported_rgcn_full_auprc": float(rr.loc["full_lag_relation", "test_auprc"]), "reported_rgcn_ablation_auprc": float(rr.loc["lag_relation_ablation", "test_auprc"])})
            rows.append(st)
            print(f"seed {seed}: T={st['auprc_transformer']:.5f} R={st['auprc_rgcn_full']:.4f} A={st['auprc_rgcn_ablation']:.4f} H2={st['H2_rgcn_minus_transformer']:+.4f} [{st['H2_rgcn_minus_transformer_ci_low']:+.4f},{st['H2_rgcn_minus_transformer_ci_high']:+.4f}] H3={st['H3_full_minus_ablation']:+.4f} [{st['H3_full_minus_ablation_ci_low']:+.4f},{st['H3_full_minus_ablation_ci_high']:+.4f}]", flush=True)
        seeds = pd.DataFrame(rows).sort_values("seed"); seeds.to_csv(a.output_dir / "step1_seed_results_v3.csv", index=False)
        agg = {}
        for c in [c for c in seeds.columns if c.startswith("auprc_") and not c.endswith(("_ci_low", "_ci_high"))] + ["H2_rgcn_minus_transformer", "H3_full_minus_ablation"] + [c for c in seeds.columns if c.startswith(("rgcn_minus_", "transformer_minus_")) and not c.endswith(("_ci_low", "_ci_high"))]:
            agg[c] = {"mean": float(seeds[c].mean()), "min": float(seeds[c].min()), "max": float(seeds[c].max()), "sd": float(seeds[c].std(ddof=1))}
        h2 = bool(seeds["H2_rgcn_minus_transformer"].mean() >= a.h2_margin and (seeds["H2_rgcn_minus_transformer_ci_low"] > 0).all())
        h3 = bool(seeds["H3_full_minus_ablation"].mean() >= a.h3_margin and (seeds["H3_full_minus_ablation_ci_low"] > 0).all() and (seeds["rgcn_full_local_latency_p95_ms"] <= a.latency_limit_ms).all())
        summary = {"design": "cross_regime_s2_train_s1_attack_test, regime target, 5 seeds", "aggregates": agg,
                   "H2": {"rule": f"seed-mean R-GCN - Transformer >= {a.h2_margin} and all bin-cluster CIs > 0", "seed_mean": agg["H2_rgcn_minus_transformer"]["mean"], "seeds_ci_excluding_zero": int((seeds["H2_rgcn_minus_transformer_ci_low"] > 0).sum()), "supported": h2},
                   "H3": {"rule": f"seed-mean full - ablation >= {a.h3_margin}, all bin-cluster CIs > 0, p95 <= {a.latency_limit_ms} ms", "seed_mean": agg["H3_full_minus_ablation"]["mean"], "seeds_ci_excluding_zero": int((seeds["H3_full_minus_ablation_ci_low"] > 0).sum()), "max_p95_ms": float(seeds["rgcn_full_local_latency_p95_ms"].max()), "supported": h3},
                   "independent_240s_bins": int(seeds["independent_240s_bins"].iloc[0]), "bootstrap_repeats": a.repeats}
        (a.output_dir / "step1_summary_v3.json").write_text(json.dumps(summary, indent=2))
        cols = ["seed", "auprc_transformer", "auprc_rgcn_full", "auprc_rgcn_ablation", "H2_rgcn_minus_transformer", "H2_rgcn_minus_transformer_ci_low", "H2_rgcn_minus_transformer_ci_high", "H3_full_minus_ablation", "H3_full_minus_ablation_ci_low", "H3_full_minus_ablation_ci_high", "rgcn_full_local_latency_p95_ms"]
        md += ["# Step 1 - Transformer / R-GCN on the cross-regime alignment (regime target)", "", seeds[cols].round(4).to_markdown(index=False), "",
               f"H2 (R-GCN - Transformer >= +{a.h2_margin}): seed mean {agg['H2_rgcn_minus_transformer']['mean']:+.4f} [{agg['H2_rgcn_minus_transformer']['min']:+.4f}, {agg['H2_rgcn_minus_transformer']['max']:+.4f}] -> **{'supported' if h2 else 'not supported'}**  ",
               f"H3 (full - ablation >= +{a.h3_margin}, p95 <= {a.latency_limit_ms:.0f} ms): seed mean {agg['H3_full_minus_ablation']['mean']:+.4f} [{agg['H3_full_minus_ablation']['min']:+.4f}, {agg['H3_full_minus_ablation']['max']:+.4f}], max p95 {seeds['rgcn_full_local_latency_p95_ms'].max():.2f} ms -> **{'supported' if h3 else 'not supported'}**", ""]
        if extra_by_event:
            cols2 = ["seed", "auprc_cyber_rf", "auprc_fused_rf", "rgcn_minus_cyber_rf", "rgcn_minus_cyber_rf_ci_low", "rgcn_minus_cyber_rf_ci_high", "transformer_minus_cyber_rf", "transformer_minus_cyber_rf_ci_low", "transformer_minus_cyber_rf_ci_high"]
            md += ["## Against the tabular cyber / fused RF on the same test events", "", seeds[cols2].round(4).to_markdown(index=False), ""]

    # ------------------------------------------------------------------ Step 2: coupling cells (intensity target)
    if a.cells_root and any(Path(a.cells_root).glob("cell_k*")):
        bank = pd.read_csv(a.position_bank)
        frames = {part: g.reset_index(drop=True) for part, g in bank.groupby("source_partition")}
        rows = []
        for cell in sorted(glob.glob(str(a.cells_root / "cell_k*"))):
            tag = os.path.basename(cell); mm = re.match(r"cell_k([\d.]+)_s([\d.]+)_seed(\d+)(_control)?", tag)
            kappa, sigma, seed, control = float(mm.group(1)), float(mm.group(2)), int(mm.group(3)), bool(mm.group(4))
            base = Path(cell) / "data/processed/point_event_v2"
            if not (base / "transformer_v2/temporal_transformer_predictions_v2.csv").exists() or not (base / "rgcn_leakage_safe_v2/rgcn_predictions_v2.csv").exists():
                print(f"{tag}: outputs missing, skipped", flush=True); continue
            d = load_test_scores(base / "transformer_v2", base / "rgcn_leakage_safe_v2")
            pl = pd.read_csv(Path(cell) / "placement_v3.csv"); pl = pl[pl.chronological_split.eq("test")].set_index("event_id")
            true_pos = frames["test"]["position_index"].to_numpy()[pl.loc[d["event_id"], "true_row"].to_numpy()]
            bins = true_pos // int(BIN_S / GRID_S)
            st = paired_stats(d, bins, rng, a.repeats)
            rr = pd.read_csv(base / "rgcn_leakage_safe_v2/rgcn_results_v2.csv").set_index("variant")
            st.update({"cell": tag, "kappa": kappa, "sigma": sigma, "seed": seed, "control": control, "rgcn_full_local_latency_p95_ms": float(rr.loc["full_lag_relation", "local_latency_p95_ms"])})
            rows.append(st); print(f"{tag}: T={st['auprc_transformer']:.4f} R={st['auprc_rgcn_full']:.4f} A={st['auprc_rgcn_ablation']:.4f} H2'={st['H2_rgcn_minus_transformer']:+.4f} H3'={st['H3_full_minus_ablation']:+.4f}", flush=True)
        if rows:
            cells = pd.DataFrame(rows).sort_values(["control", "sigma", "kappa", "seed"]); cells.to_csv(a.output_dir / "step2_cell_results_v3.csv", index=False)
            g = cells[~cells.control].groupby(["sigma", "kappa"])
            agg2 = g.agg(transformer_mean=("auprc_transformer", "mean"), rgcn_full_mean=("auprc_rgcn_full", "mean"), rgcn_ablation_mean=("auprc_rgcn_ablation", "mean"),
                         H2p_mean=("H2_rgcn_minus_transformer", "mean"), H2p_min=("H2_rgcn_minus_transformer", "min"), H2p_max=("H2_rgcn_minus_transformer", "max"),
                         H2p_seeds_ci_gt0=("H2_rgcn_minus_transformer_ci_low", lambda s: int((s > 0).sum())),
                         H3p_mean=("H3_full_minus_ablation", "mean"), H3p_min=("H3_full_minus_ablation", "min"), H3p_max=("H3_full_minus_ablation", "max"),
                         H3p_seeds_ci_gt0=("H3_full_minus_ablation_ci_low", lambda s: int((s > 0).sum())), max_p95_ms=("rgcn_full_local_latency_p95_ms", "max"), seeds=("seed", "nunique")).reset_index()
            agg2["H2p_supported"] = (agg2.H2p_mean >= a.h2_margin) & (agg2.H2p_seeds_ci_gt0 == agg2.seeds)
            agg2["H3p_supported"] = (agg2.H3p_mean >= a.h3_margin) & (agg2.H3p_seeds_ci_gt0 == agg2.seeds) & (agg2.max_p95_ms <= a.latency_limit_ms)
            agg2.to_csv(a.output_dir / "step2_cell_aggregate_v3.csv", index=False)
            md += ["# Step 2 - coupling cells (intensity target)", "", agg2.round(4).to_markdown(index=False), ""]
    (a.output_dir / "TORCH_RERUNS_SUMMARY_v3.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
