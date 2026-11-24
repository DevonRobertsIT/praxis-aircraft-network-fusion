#!/usr/bin/env python3
"""Position bank for the coupling study.

For every attack-regime source block of a v3 alignment, place candidate event centers every
GRID_S seconds across the block's safe-center span and aggregate the primary +/-120 s window
with the SAME v2 feature and target functions the alignments use. Each row is a possible
cyber position with its features and its attack intensity (nf3_attack_flow_fraction).
The coupling study then maps aircraft events onto positions without re-running alignment.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("nf3_v2", HERE / "11_nf3_leakage_safe_alignment.py")
v2 = importlib.util.module_from_spec(_spec); sys.modules["nf3_v2"] = v2; _spec.loader.exec_module(v2)

GRID_S = 30.0
HALF_S = 120.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--alignment-dir", type=Path, required=True)
    p.add_argument("--recovery-zip", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    blocks = pd.read_csv(a.alignment_dir / "nf3_source_block_manifest_v3.csv")
    blocks = blocks[blocks["regimes"].eq("attack")].reset_index(drop=True)
    print("attack blocks:", blocks["source_block_id"].tolist(), flush=True)

    with ZipFile(a.recovery_zip) as z, z.open(v2.RAW_ENTRY) as h:
        net = pd.read_csv(h, usecols=v2.RAW_COLUMNS, low_memory=False, dtype={"IPV4_SRC_ADDR": "category", "IPV4_DST_ADDR": "category", "Attack": "category"})
    net["source_row_number"] = np.arange(len(net), dtype=np.int64)
    net["flow_start_utc"] = pd.to_datetime(pd.to_numeric(net["FLOW_START_MILLISECONDS"]), unit="ms", utc=True)
    net["flow_end_utc"] = pd.to_datetime(pd.to_numeric(net["FLOW_END_MILLISECONDS"]), unit="ms", utc=True)
    net["flow_duration_s"] = (net["flow_end_utc"] - net["flow_start_utc"]).dt.total_seconds()
    net["packet_total"] = pd.to_numeric(net["IN_PKTS"], errors="coerce") + pd.to_numeric(net["OUT_PKTS"], errors="coerce")
    net["byte_total"] = pd.to_numeric(net["IN_BYTES"], errors="coerce") + pd.to_numeric(net["OUT_BYTES"], errors="coerce")
    for c in v2.IAT_COLUMNS + ["L4_SRC_PORT", "L4_DST_PORT", "PROTOCOL"]:
        net[c] = pd.to_numeric(net[c], errors="coerce")
    net["binary_label"] = pd.to_numeric(net["Label"]).astype(np.int8)
    net["attack_category"] = net["Attack"]
    net = net.sort_values(["flow_start_utc", "flow_end_utc", "source_row_number"], kind="mergesort").reset_index(drop=True)
    net["flow_start_ns"] = net["flow_start_utc"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    net["flow_end_ns"] = net["flow_end_utc"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    fs, fe = net["flow_start_ns"].to_numpy(), net["flow_end_ns"].to_numpy()
    max_dur_ns = int(net["flow_duration_s"].max() * 1e9) + 1
    half_ns = int(HALF_S * 1e9)

    rows = []
    for b in blocks.itertuples(index=False):
        bs, be = pd.Timestamp(b.block_start_utc).value, pd.Timestamp(b.block_end_utc).value
        idx = np.flatnonzero((fs >= bs) & (fe <= be))          # flows fully contained in the block
        starts = fs[idx]
        s0, s1 = pd.Timestamp(b.safe_center_start_utc).value, pd.Timestamp(b.safe_center_end_utc).value
        centers = np.arange(s0, s1 + 1, int(GRID_S * 1e9))
        for k, c in enumerate(centers):
            ws, we = c - half_ns, c + half_ns
            left = int(np.searchsorted(starts, ws - max_dur_ns, side="left")); right = int(np.searchsorted(starts, we, side="right"))
            cand = idx[left:right]
            matched = cand[fe[cand] >= ws] if cand.size else np.empty(0, dtype=np.int64)
            row = {"source_block_id": b.source_block_id, "source_partition": b.source_partition, "position_index": k,
                   "center_utc": pd.to_datetime(c, utc=True), "bin_240s": int(k // int(2 * HALF_S / GRID_S))}
            row.update(v2.aggregate_features(net, matched, int(c), ws, we))
            t = v2.aggregate_targets(net, matched)
            row.update({"attack_flow_fraction": t["nf3_attack_flow_fraction"], "attack_flow_count": t["nf3_attack_flow_count"],
                        "distinct_attack_category_count": t["nf3_distinct_attack_category_count"], "attack_window_label": t["cyber_attack_window_label"]})
            rows.append(row)
        print(f"{b.source_block_id}: {len(centers)} positions", flush=True)
    bank = pd.DataFrame(rows)
    bank.to_csv(a.output, index=False)
    print(json.dumps({"positions": len(bank), "by_partition": bank.groupby("source_partition").size().to_dict(),
                      "intensity_quantiles_by_partition": bank.groupby("source_partition")["attack_flow_fraction"].quantile([0.1, 0.5, 0.9]).round(4).unstack().to_dict()}, indent=1, default=str))


if __name__ == "__main__":
    main()
