#!/usr/bin/env python3
"""Session-indicator / any-flow baselines with a reporting margin, and cluster bootstrap.

Baselines (test partition): `session_indicator` scores an event 1 if its owned NF3 block is in
Session 2; `any_flow` scores 1 if the primary window contains any flow. A model's separability
is reported as "beyond the baselines" only if its AUPRC exceeds the stronger baseline by at
least --margin AND the cluster-bootstrap 95% interval of that margin excludes zero.

Cluster bootstrap: test events are near-duplicates of each other because a partition's source
blocks are replayed many times. The independent unit is a non-overlapping source-time bin of
width 2 x primary half-window (240 s): two events in the same bin share most of their flows.
The script resamples bins (and, secondarily, replay cycles) with replacement, compares the
variance with an event-level bootstrap, and reports the design effect and effective sample
size n_eff = n / DEFF for every model and for the paired fused - cyber differences.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

BIN_S = 240.0


def args_() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--alignment-dir", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--margin", type=float, default=0.02)
    p.add_argument("--repeats", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def auprc(y: np.ndarray, s: np.ndarray) -> float:
    return float(average_precision_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def bootstrap(stat, keys: np.ndarray, rng: np.random.Generator, repeats: int) -> np.ndarray:
    """Resample clusters identified by `keys` with replacement; stat receives an index array."""
    uniq, inv = np.unique(keys, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    out = np.empty(repeats)
    for r in range(repeats):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[k] for k in pick])
        out[r] = stat(idx)
    return out


def main() -> None:
    a = args_()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    pred = pd.read_csv(a.predictions)
    bank = pd.read_csv(a.alignment_dir / "candidate_point_events_v3_with_context_and_nf3_primary.csv", usecols=["event_id", "nf3_flow_count"])
    pred = pred.merge(bank, on="event_id", how="left")
    base = pred.drop_duplicates("event_id").set_index("event_id")
    y = base["model_target"].to_numpy().astype(int)
    session = base["source_block_id"].str.extract(r"^NF3(S\d)_")[0]
    s_ind = (session == "S2").astype(int).to_numpy()
    anyflow = (base["nf3_flow_count"].fillna(0) > 0).astype(int).to_numpy()
    t_ns = pd.to_datetime(base["aligned_event_time_utc"], utc=True).dt.tz_convert(None).to_numpy(dtype="datetime64[ns]").astype(np.int64)
    bins = t_ns // int(BIN_S * 1e9)
    cycles = base["partition_replay_cycle_index"].to_numpy()
    n = len(y)
    baselines = {
        "session_indicator": {"auprc": auprc(y, s_ind), "roc_auc": float(roc_auc_score(y, s_ind)) if 0 < s_ind.sum() < n and 0 < y.sum() < n else float("nan"), "sessions_in_test": "|".join(sorted(session.unique()))},
        "any_flow": {"auprc": auprc(y, anyflow), "flow_covered_events": int(anyflow.sum())},
        "prevalence": {"auprc": float(y.mean())},
    }
    strongest = max(v["auprc"] for k, v in baselines.items() if k != "prevalence" and not np.isnan(v["auprc"]))
    strongest_name = [k for k, v in baselines.items() if v["auprc"] == strongest][0]
    units = {
        "test_events": n, "positives": int(y.sum()), "negatives": int(n - y.sum()),
        "independent_240s_bins_total": int(len(np.unique(bins))), "independent_240s_bins_positive": int(len(np.unique(bins[y == 1]))),
        "independent_240s_bins_negative": int(len(np.unique(bins[y == 0]))), "replay_cycles": int(len(np.unique(cycles))),
    }

    rows, paired = [], []
    scores = {}
    for (fs, m), g in pred.groupby(["feature_set", "model"]):
        g = g.set_index("event_id").loc[base.index]
        s = g["predicted_probability"].to_numpy(); scores[(fs, m)] = s
        point = auprc(y, s)
        st = lambda idx, s=s: auprc(y[idx], s[idx])
        ev = bootstrap(st, np.arange(n), rng, a.repeats)  # event-level (each event its own cluster)
        cb = bootstrap(st, bins, rng, a.repeats)
        cc = bootstrap(st, cycles, rng, a.repeats)
        margin = point - strongest
        mb = bootstrap(lambda idx, s=s: auprc(y[idx], s[idx]) - auprc(y[idx], (s_ind if strongest_name == "session_indicator" else anyflow)[idx]), bins, rng, a.repeats)
        v_ev, v_cb = np.nanvar(ev), np.nanvar(cb)
        deff = float(v_cb / v_ev) if v_ev > 0 else float("nan")
        rows.append({
            "feature_set": fs, "model": m, "test_auprc": point,
            "event_bootstrap_ci_low": float(np.nanpercentile(ev, 2.5)), "event_bootstrap_ci_high": float(np.nanpercentile(ev, 97.5)),
            "bin_cluster_ci_low": float(np.nanpercentile(cb, 2.5)), "bin_cluster_ci_high": float(np.nanpercentile(cb, 97.5)),
            "cycle_cluster_ci_low": float(np.nanpercentile(cc, 2.5)), "cycle_cluster_ci_high": float(np.nanpercentile(cc, 97.5)),
            "design_effect_bins_vs_events": deff, "effective_sample_size": float(n / deff) if deff and deff > 0 else float("nan"),
            "strongest_baseline": strongest_name, "strongest_baseline_auprc": strongest,
            "margin_over_baseline": margin, "margin_bin_cluster_ci_low": float(np.nanpercentile(mb, 2.5)), "margin_bin_cluster_ci_high": float(np.nanpercentile(mb, 97.5)),
            "separability_beyond_baselines": bool(margin >= a.margin and np.nanpercentile(mb, 2.5) > 0),
        })
        print(f"{fs:9s} {m:20s} AUPRC {point:.4f}  event CI [{rows[-1]['event_bootstrap_ci_low']:.4f},{rows[-1]['event_bootstrap_ci_high']:.4f}]  bin-cluster CI [{rows[-1]['bin_cluster_ci_low']:.4f},{rows[-1]['bin_cluster_ci_high']:.4f}]  DEFF {deff:.2f}  n_eff {rows[-1]['effective_sample_size']:.0f}  margin {margin:+.4f} beyond={rows[-1]['separability_beyond_baselines']}", flush=True)
    for m in sorted({m for _, m in scores}):
        if ("fused", m) in scores and ("cyber", m) in scores:
            sf, sc = scores[("fused", m)], scores[("cyber", m)]
            d = auprc(y, sf) - auprc(y, sc)
            db = bootstrap(lambda idx: auprc(y[idx], sf[idx]) - auprc(y[idx], sc[idx]), bins, rng, a.repeats)
            de = bootstrap(lambda idx: auprc(y[idx], sf[idx]) - auprc(y[idx], sc[idx]), np.arange(n), rng, a.repeats)
            paired.append({"model": m, "fused_minus_cyber_auprc": d, "event_ci_low": float(np.nanpercentile(de, 2.5)), "event_ci_high": float(np.nanpercentile(de, 97.5)),
                           "bin_cluster_ci_low": float(np.nanpercentile(db, 2.5)), "bin_cluster_ci_high": float(np.nanpercentile(db, 97.5))})
    res = pd.DataFrame(rows); res.to_csv(a.output_dir / "model_vs_baselines_cluster_bootstrap_v3.csv", index=False)
    pd.DataFrame(paired).to_csv(a.output_dir / "paired_fused_minus_cyber_cluster_bootstrap_v3.csv", index=False)
    summary = {"baselines": baselines, "strongest_baseline": strongest_name, "margin_rule": {"minimum_margin": a.margin, "interval": "bin-cluster bootstrap 95% must exclude 0"},
               "independent_units": units, "bootstrap_repeats": a.repeats, "bin_width_s": BIN_S, "seed": a.seed}
    (a.output_dir / "session_baseline_and_effective_sample_size_v3.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
