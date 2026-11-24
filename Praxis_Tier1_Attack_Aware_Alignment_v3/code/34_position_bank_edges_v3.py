#!/usr/bin/env python3
"""Bounded edges for every position of the position bank.

For each candidate center in position_bank_*_v3.csv, select the 16 label-blind nearest flows of
its primary +/-120 s window with the same v2 function the alignments use (nearest_edge_indices),
and write them in the schema of event_flow_edges_primary_bounded_v3.csv keyed by
(source_block_id, position_index) instead of event_id. 35_materialize_coupling_cell_v3.py joins
these to events by their observed position.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("nf3_v2", HERE / "11_nf3_leakage_safe_alignment.py")
v2 = importlib.util.module_from_spec(_spec); sys.modules["nf3_v2"] = v2; _spec.loader.exec_module(v2)
HALF_S = 120.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--alignment-dir", type=Path, required=True)
    p.add_argument("--position-bank", type=Path, required=True)
    p.add_argument("--recovery-zip", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    bank = pd.read_csv(a.position_bank, usecols=["source_block_id", "source_partition", "position_index", "center_utc"])
    blocks = pd.read_csv(a.alignment_dir / "nf3_source_block_manifest_v3.csv").set_index("source_block_id")

    with ZipFile(a.recovery_zip) as z, z.open(v2.RAW_ENTRY) as h:
        net = pd.read_csv(h, usecols=v2.RAW_COLUMNS, low_memory=False, dtype={"IPV4_SRC_ADDR": "category", "IPV4_DST_ADDR": "category", "Attack": "category"})
    net["source_row_number"] = np.arange(len(net), dtype=np.int64)
    net["flow_start_utc"] = pd.to_datetime(pd.to_numeric(net["FLOW_START_MILLISECONDS"]), unit="ms", utc=True)
    net["flow_end_utc"] = pd.to_datetime(pd.to_numeric(net["FLOW_END_MILLISECONDS"]), unit="ms", utc=True)
    net["flow_duration_s"] = (net["flow_end_utc"] - net["flow_start_utc"]).dt.total_seconds()
    for c in ["L4_SRC_PORT", "L4_DST_PORT", "PROTOCOL"]:
        net[c] = pd.to_numeric(net[c], errors="coerce")
    net = net.sort_values(["flow_start_utc", "flow_end_utc", "source_row_number"], kind="mergesort").reset_index(drop=True)
    net["flow_start_ns"] = net["flow_start_utc"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    net["flow_end_ns"] = net["flow_end_utc"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    net["flow_id"] = "NF3-" + net["source_row_number"].astype(str).str.zfill(12)
    fs, fe = net["flow_start_ns"].to_numpy(), net["flow_end_ns"].to_numpy()
    max_dur_ns = int(net["flow_duration_s"].max() * 1e9) + 1
    half_ns = int(HALF_S * 1e9)

    frames = []
    for bid, g in bank.groupby("source_block_id"):
        b = blocks.loc[bid]
        bs, be = pd.Timestamp(b["block_start_utc"]).value, pd.Timestamp(b["block_end_utc"]).value
        idx = np.flatnonzero((fs >= bs) & (fe <= be)); starts = fs[idx]
        for r in g.itertuples(index=False):
            c = pd.Timestamp(r.center_utc).value; ws, we = c - half_ns, c + half_ns
            left = int(np.searchsorted(starts, ws - max_dur_ns, side="left")); right = int(np.searchsorted(starts, we, side="right"))
            cand = idx[left:right]; matched = cand[fe[cand] >= ws] if cand.size else np.empty(0, dtype=np.int64)
            if not matched.size:
                continue
            sel = v2.nearest_edge_indices(net, matched, c, ws, we, v2.MAX_MATERIALIZED_EDGES_PER_EVENT)
            sf = net.iloc[sel]; ss, se = sf["flow_start_ns"].to_numpy(), sf["flow_end_ns"].to_numpy(); lag = (ss - c) / 1e9
            frames.append(pd.DataFrame({
                "source_block_id": bid, "source_partition": r.source_partition, "position_index": r.position_index,
                "flow_id": sf["flow_id"].to_numpy(), "window_name": "primary_120s", "aligned_event_time_utc": pd.to_datetime(c, utc=True),
                "window_start_utc": pd.to_datetime(ws, utc=True), "window_end_utc": pd.to_datetime(we, utc=True),
                "flow_start_utc": sf["flow_start_utc"].to_numpy(), "flow_end_utc": sf["flow_end_utc"].to_numpy(),
                "signed_flow_start_lag_s": lag, "absolute_flow_start_lag_s": np.abs(lag),
                "overlap_seconds": np.maximum(0, np.minimum(se, we) - np.maximum(ss, ws)) / 1e9,
                "relation_type": np.where(se < c, "pre_event", np.where(ss > c, "post_event", "overlaps_event_time")),
                "source_row_number": sf["source_row_number"].to_numpy(), "protocol": sf["PROTOCOL"].to_numpy(),
                "src_port": sf["L4_SRC_PORT"].to_numpy(), "dst_port": sf["L4_DST_PORT"].to_numpy(),
            }))
        print(f"{bid}: {len(g)} positions", flush=True)
    edges = pd.concat(frames, ignore_index=True)
    edges.to_csv(a.output, index=False)
    print(f"edge rows: {len(edges):,} for {edges.groupby(['source_block_id', 'position_index']).ngroups:,} positions")


if __name__ == "__main__":
    main()
