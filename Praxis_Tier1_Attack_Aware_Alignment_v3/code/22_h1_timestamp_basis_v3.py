#!/usr/bin/env python3
"""H1 (revised): does aligning cyber windows on snapshot (processing) time instead of
aircraft event time change the target and degrade held-out ranking?

  H1a  the primary attack-window label changes for >= 5% of events
  H1b  the cyber model built on processing-time alignment loses >= 0.02 held-out AUPRC
       when judged against the event-time target (bin-cluster bootstrap interval excludes 0)

Inputs: two v3 alignments of the SAME design and block layout that differ only in
--time-basis, and the test predictions produced by 20_tabular_baselines_v3.py on each
(the processing-time run scored against event-time targets via --target-basis-dir).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

BIN_S = 240.0


def args_() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--event-basis-dir", type=Path, required=True)
    p.add_argument("--snapshot-basis-dir", type=Path, required=True)
    p.add_argument("--event-basis-predictions", type=Path, required=True)
    p.add_argument("--snapshot-basis-predictions", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", default="random_forest")
    p.add_argument("--feature-set", default="cyber")
    p.add_argument("--label-change-threshold", type=float, default=0.05)
    p.add_argument("--auprc-loss-threshold", type=float, default=0.02)
    p.add_argument("--repeats", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_targets(d: Path) -> pd.DataFrame:
    t = pd.read_csv(d / "event_nf3_targets_v3.csv")
    t = t[t.window_name.eq("primary_120s")][["event_id", "source_partition", "cyber_attack_window_label"]]
    m = pd.read_csv(d / "event_source_block_mapping_v3.csv", usecols=["event_id", "event_lag_seconds", "aligned_event_time_utc", "chronological_split"])
    return t.merge(m, on="event_id")


def main() -> None:
    a = args_()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    ev, sn = load_targets(a.event_basis_dir), load_targets(a.snapshot_basis_dir)
    j = ev.merge(sn, on="event_id", suffixes=("_event", "_snapshot"))
    j["label_changed"] = (j.cyber_attack_window_label_event != j.cyber_attack_window_label_snapshot).astype(int)
    j["center_shift_s"] = (pd.to_datetime(j.aligned_event_time_utc_snapshot, utc=True) - pd.to_datetime(j.aligned_event_time_utc_event, utc=True)).dt.total_seconds()
    by_part = j.groupby("chronological_split_event").agg(events=("event_id", "size"), changed=("label_changed", "sum"), mean_abs_lag_s=("event_lag_seconds_event", lambda s: float(np.abs(s).mean())), max_abs_shift_s=("center_shift_s", lambda s: float(np.abs(s).max())))
    by_part["changed_fraction"] = by_part.changed / by_part.events
    overall = {"events": int(len(j)), "changed": int(j.label_changed.sum()), "changed_fraction": float(j.label_changed.mean()),
               "mean_abs_processing_lag_s": float(np.abs(j.event_lag_seconds_event).mean()), "p95_abs_processing_lag_s": float(np.percentile(np.abs(j.event_lag_seconds_event), 95)),
               "max_abs_center_shift_s": float(np.abs(j.center_shift_s).max()), "changed_events_mean_abs_lag_s": float(np.abs(j.loc[j.label_changed.eq(1), "event_lag_seconds_event"]).mean()) if j.label_changed.any() else float("nan")}
    j.to_csv(a.output_dir / "h1_label_change_by_event_v3.csv", index=False)
    by_part.to_csv(a.output_dir / "h1_label_change_by_partition_v3.csv")

    pe = pd.read_csv(a.event_basis_predictions); ps = pd.read_csv(a.snapshot_basis_predictions)
    pe = pe[pe.feature_set.eq(a.feature_set) & pe.model.eq(a.model)].set_index("event_id")
    ps = ps[ps.feature_set.eq(a.feature_set) & ps.model.eq(a.model)].set_index("event_id").loc[pe.index]
    y = pe.model_target.to_numpy().astype(int)  # event-time target (the snapshot run was scored against it via --target-basis-dir)
    assert (ps.model_target.to_numpy().astype(int) == y).all(), "snapshot-basis predictions were not scored against event-time targets"
    se, ss = pe.predicted_probability.to_numpy(), ps.predicted_probability.to_numpy()
    bins = pd.to_datetime(pe.aligned_event_time_utc, utc=True).dt.tz_convert(None).to_numpy(dtype="datetime64[ns]").astype(np.int64) // int(BIN_S * 1e9)
    auprc = lambda yy, s: float(average_precision_score(yy, s)) if 0 < yy.sum() < len(yy) else float("nan")
    loss = auprc(y, se) - auprc(y, ss)
    rng = np.random.default_rng(a.seed)
    uniq, inv = np.unique(bins, return_inverse=True); members = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    draws = np.empty(a.repeats)
    for r in range(a.repeats):
        idx = np.concatenate([members[k] for k in rng.integers(0, len(uniq), len(uniq))])
        draws[r] = auprc(y[idx], se[idx]) - auprc(y[idx], ss[idx])
    lo, hi = float(np.nanpercentile(draws, 2.5)), float(np.nanpercentile(draws, 97.5))
    h1a = overall["changed_fraction"] >= a.label_change_threshold
    h1b = loss >= a.auprc_loss_threshold and lo > 0
    summary = {
        "hypothesis": "H1 (revised): processing-time alignment changes >=5% of targets and costs >=0.02 held-out AUPRC",
        "design": "same block layout; event centers placed on snapshot_time_utc instead of aircraft_event_time_utc",
        "label_change": overall, "label_change_by_partition": by_part.reset_index().to_dict(orient="records"),
        "auprc": {"model": a.model, "feature_set": a.feature_set, "event_basis_auprc": auprc(y, se), "snapshot_basis_auprc_vs_event_target": auprc(y, ss),
                  "loss": loss, "bin_cluster_ci_95": [lo, hi], "repeats": a.repeats},
        "decision": {"H1a_label_change_supported": bool(h1a), "H1b_auprc_loss_supported": bool(h1b), "H1_supported": bool(h1a and h1b)},
    }
    (a.output_dir / "h1_timestamp_basis_summary_v3.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
