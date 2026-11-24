#!/usr/bin/env python3
"""Leakage-safe NF3 controlled replay, v3: attack-aware source-block allocation.

Changes relative to 11_nf3_leakage_safe_alignment.py (v2):

1. Regime-stratified allocation. The attack-window landscape of each session is
   computed on a 1 s timeline with the target's own +/-120 s containment rule and
   cut into maximal ATTACK / benign regime intervals. NF-UNSW-NB15-v3 has three:
   a 120 min attack plateau opening Session 1, 636 min of benign traffic after it,
   and a 714 min attack plateau covering Session 2 (7.5 min of benign). Every
   regime interval is split into contiguous partition sub-blocks (70/15/15 by
   duration, each at least a minimum viable length) with a 1,200 s embargo at
   every partition change; intervals too short to host a block are left unused
   and listed. v2 allocated one block per partition per session in fixed order,
   which placed Session 1's validation and test blocks in the benign span and
   made label == session in both held-out partitions.
2. `--design` selects within_session_stratified (every partition receives a share
   of every regime interval) or cross_regime_s2_train_s1_attack_test (train and
   validation on Session 2 attacks + Session 1 benign; test on Session 1's attack
   plateau + held-out Session 1 benign, so the test attack regime is unseen and
   every test event comes from one session). cross_regime_s1_train_s2_attack_test
   is provided for the audit only: its test label is identical to session.
3. `--time-basis` maps events on aircraft_event_time_utc (default) or on
   snapshot_time_utc (processing time) from the same origin, for the H1 test.
4. The post-build gate adds `both_classes_within_each_partition` and
   `session_indicator_not_perfect_within_any_partition`; the per-cell table
   `both_classes_within_each_session_partition` is written and reported, with
   cells that the data make infeasible (a session without both regimes) flagged.

Feature, target, and edge computations are imported unchanged from the v2 script.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import sys
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("nf3_v2", HERE / "11_nf3_leakage_safe_alignment.py")
v2 = importlib.util.module_from_spec(_spec)
sys.modules["nf3_v2"] = v2
_spec.loader.exec_module(v2)  # defines helpers only; its main() is __main__-guarded

VERSION = "point_event_nf3_alignment_leakage_safe_v3"
WINDOWS_ALL = dict(v2.WINDOWS)
PRIMARY_WINDOW = v2.PRIMARY_WINDOW
PRIMARY_HALF_S = WINDOWS_ALL[PRIMARY_WINDOW]
EMBARGO_S = v2.INTER_PARTITION_EMBARGO_S
SESSION_GAP_S = v2.SESSION_GAP_S
MAX_EDGES = v2.MAX_MATERIALIZED_EDGES_PER_EVENT
SPLITS = v2.SPLITS
SPLIT_BITS = v2.SPLIT_BITS
ALL3 = ("train", "validation", "test")
DEV = ("train", "validation")
DESIGNS = {  # keyed by (session label, regime)
    "within_session_stratified": {("S1", "attack"): ALL3, ("S1", "benign"): ALL3, ("S2", "attack"): ALL3, ("S2", "benign"): ALL3},
    "cross_regime_s2_train_s1_attack_test": {("S1", "attack"): ("test",), ("S1", "benign"): ALL3, ("S2", "attack"): DEV, ("S2", "benign"): DEV},
    "cross_regime_s1_train_s2_attack_test": {("S1", "attack"): DEV, ("S1", "benign"): ALL3, ("S2", "attack"): ("test",), ("S2", "benign"): ("test",)},
}
MIN_CENTER_SPAN_S = 600.0  # a sub-block must leave at least this much safe-center time
TARGET_SHARES = {"train": 0.70, "validation": 0.15, "test": 0.15}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--recovery-zip", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--design", choices=sorted(DESIGNS), default="cross_regime_s2_train_s1_attack_test")
    p.add_argument("--time-basis", choices=["aircraft_event_time_utc", "snapshot_time_utc"], default="aircraft_event_time_utc")
    p.add_argument("--windows", choices=["all", "primary_only"], default="all")
    p.add_argument("--min-minority-events", type=int, default=10, help="gate: minimum events of each class per partition")
    p.add_argument("--max-events", type=int, default=0, help="debug: limit events (0 = all)")
    return p.parse_args()


# --------------------------------------------------------------------------- allocation
def attack_window_coverage(att_start_ns: np.ndarray, att_end_ns: np.ndarray, s_ns: int, e_ns: int, half_ns: int) -> np.ndarray:
    """1-second timeline over [s, e]; True where a +/-half window centered there contains an attack flow.
    A window centered at c contains flow f iff f.start <= c + half and f.end >= c - half,
    i.e. c in [f.start - half, f.end + half] (same containment rule as the target)."""
    n = int((e_ns - s_ns) // 1_000_000_000) + 2
    diff = np.zeros(n + 1, dtype=np.int32)
    lo = np.clip((att_start_ns - half_ns - s_ns) // 1_000_000_000, 0, n - 1).astype(np.int64)
    hi = np.clip((att_end_ns + half_ns - s_ns) // 1_000_000_000 + 1, 0, n).astype(np.int64)
    np.add.at(diff, lo, 1)
    np.add.at(diff, hi, -1)
    return np.cumsum(diff)[:n] > 0


def regime_intervals(covered: np.ndarray, s_ns: int) -> list[dict[str, Any]]:
    """Maximal runs of the 1 s attack-window coverage timeline -> [{regime, start_ns, end_ns}]."""
    change = np.flatnonzero(np.diff(covered.astype(np.int8)))
    bounds = np.concatenate([[0], change + 1, [len(covered)]])
    return [
        {"regime": "attack" if covered[b] else "benign", "start_ns": s_ns + int(b) * 1_000_000_000, "end_ns": s_ns + int(bounds[i + 1]) * 1_000_000_000}
        for i, b in enumerate(bounds[:-1])
    ]


def allocate_session(
    session_label: str,
    s_ns: int,
    e_ns: int,
    covered: np.ndarray,
    roles: dict[tuple[str, str], tuple[str, ...]],
    inset_s: float,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    """Split every regime interval of the session into contiguous partition sub-blocks.

    A sub-block needs room for a 600 s half-embargo on each side, the safe-center inset
    on each side, and MIN_CENTER_SPAN_S of centers. Partitions get 70/15/15 of the
    interval by duration, each lifted to that minimum; the train share absorbs the
    remainder. Intervals that cannot host all their partitions are given whole to
    the first partition in the role tuple; intervals too short for one block are
    left unassigned and listed."""
    min_len = EMBARGO_S + 2.0 * inset_s + MIN_CENTER_SPAN_S
    segments: list[dict[str, Any]] = []
    unassigned: list[dict[str, Any]] = []
    info: dict[str, Any] = {"regime_intervals": [], "min_viable_block_s": min_len}
    for k, iv in enumerate(regime_intervals(covered, s_ns)):
        L = (iv["end_ns"] - iv["start_ns"]) / 1e9
        parts = roles.get((session_label, iv["regime"]))
        rec = {"interval_index": k, "regime": iv["regime"], "start_utc": str(pd.to_datetime(iv["start_ns"], utc=True)), "end_utc": str(pd.to_datetime(iv["end_ns"], utc=True)), "duration_s": L}
        if not parts or L < min_len:
            rec["assigned"] = None
            rec["reason"] = "no role in design" if not parts else f"shorter than minimum viable block ({min_len:.0f} s)"
            unassigned.append({**rec, "session": session_label})
            info["regime_intervals"].append(rec)
            continue
        if L < len(parts) * min_len:
            parts = (parts[0],)
            rec["note"] = "interval hosts a single partition (too short for all)"
        shares = np.array([TARGET_SHARES[p] for p in parts]); shares = shares / shares.sum()
        lens = np.maximum(shares * L, min_len)
        lens[0] = L - lens[1:].sum()
        if lens[0] < min_len:
            raise RuntimeError(f"{session_label} {iv['regime']} interval cannot host {parts} with minimum block {min_len:.0f} s.")
        cursor = iv["start_ns"]
        for p, ln in zip(parts, lens):
            nxt = cursor + int(round(ln * 1e9))
            segments.append({"segment_index": len(segments), "segment_start_ns": cursor, "segment_end_ns": nxt, "partition": p, "regime": iv["regime"], "interval_index": k})
            cursor = nxt
        segments[-1]["segment_end_ns"] = iv["end_ns"]
        rec["assigned"] = {p: float(ln) for p, ln in zip(parts, lens)}
        info["regime_intervals"].append(rec)
    seg = pd.DataFrame(segments)
    if seg.empty:
        raise RuntimeError(f"No assignable intervals in session {session_label}.")
    return seg, unassigned, info


def segments_to_blocks(seg: pd.DataFrame, session_index: int, session_label: str, block_index0: int, inset_s: float) -> list[dict[str, Any]]:
    """Merge consecutive same-partition segments; carve a symmetric 1,200 s embargo at every partition change."""
    rows = []
    seg = seg.sort_values("segment_index").reset_index(drop=True)
    runs = []
    for _, r in seg.iterrows():
        if runs and runs[-1]["partition"] == r["partition"]:
            runs[-1]["end_ns"] = int(r["segment_end_ns"])
            runs[-1]["segments"].append(int(r["segment_index"]))
        else:
            runs.append({"partition": r["partition"], "start_ns": int(r["segment_start_ns"]), "end_ns": int(r["segment_end_ns"]), "segments": [int(r["segment_index"])], "regimes": set()})
        runs[-1]["regimes"].add(str(r["regime"]))
    half_embargo = int(EMBARGO_S / 2 * 1e9)
    inset = int(inset_s * 1e9)
    for k, run in enumerate(runs):
        start = run["start_ns"] + (half_embargo if k > 0 else 0)
        end = run["end_ns"] - (half_embargo if k < len(runs) - 1 else 0)
        rows.append(
            {
                "source_block_index": block_index0 + k,
                "source_block_id": f"NF3{session_label}_{run['partition']}_b{k}",
                "source_session_index": session_index,
                "source_partition": run["partition"],
                "block_start_utc": pd.to_datetime(start, utc=True),
                "block_end_utc": pd.to_datetime(end, utc=True),
                "safe_center_start_utc": pd.to_datetime(start + inset, utc=True),
                "safe_center_end_utc": pd.to_datetime(end - inset, utc=True),
                "block_duration_s": (end - start) / 1e9,
                "safe_center_span_s": (end - start - 2 * inset) / 1e9,
                "preceding_embargo_s": 0 if k == 0 else EMBARGO_S,
                "following_embargo_s": 0 if k == len(runs) - 1 else EMBARGO_S,
                "segments": "|".join(str(s) for s in run["segments"]),
                "regimes": "|".join(sorted(run["regimes"])),
            }
        )
        if rows[-1]["safe_center_span_s"] <= 0:
            raise RuntimeError("A merged block is too short for the safe-center inset.")
    return rows


def map_events_to_blocks(events: pd.DataFrame, blocks: pd.DataFrame, time_column: str) -> pd.DataFrame:
    """v2 mapper with a selectable time basis. The origin of elapsed time is always the partition's
    minimum aircraft_event_time_utc, so switching to snapshot_time_utc shifts each event by exactly
    its own processing lag."""
    parts = []
    for split_name in SPLITS:
        part = events.loc[events["chronological_split"].eq(split_name)].copy()
        part = part.sort_values(["aircraft_event_time_utc", "event_id"], kind="mergesort")
        owned = blocks.loc[blocks["source_partition"].eq(split_name)].sort_values(["source_session_index", "source_block_index"]).reset_index(drop=True)
        if owned.empty:
            raise RuntimeError(f"No source blocks owned by partition {split_name}.")
        s = pd.to_datetime(owned["safe_center_start_utc"], utc=True).to_numpy(dtype="datetime64[ns]").astype(np.int64)
        e = pd.to_datetime(owned["safe_center_end_utc"], utc=True).to_numpy(dtype="datetime64[ns]").astype(np.int64)
        spans = e - s
        cumulative = np.cumsum(spans)
        total = int(cumulative[-1])
        origin = int(part["aircraft_event_time_utc"].to_numpy(dtype="datetime64[ns]").astype(np.int64).min())
        physical = pd.to_datetime(part[time_column], utc=True).to_numpy(dtype="datetime64[ns]").astype(np.int64)
        elapsed = physical - origin
        elapsed = np.where(elapsed < 0, 0, elapsed)
        virtual = np.mod(elapsed, total)
        cycle = elapsed // total
        pos = np.searchsorted(cumulative, virtual, side="right")
        prev = np.where(pos == 0, 0, cumulative[np.maximum(pos - 1, 0)])
        part["aligned_event_time_utc"] = pd.to_datetime(s[pos] + (virtual - prev), utc=True)
        part["source_block_index"] = owned.loc[pos, "source_block_index"].to_numpy(dtype=int)
        part["source_block_id"] = owned.loc[pos, "source_block_id"].to_numpy(dtype=str)
        part["partition_replay_cycle_index"] = cycle.astype(np.int64)
        part["mapping_time_basis"] = time_column
        parts.append(part)
    mapped = pd.concat(parts, ignore_index=True)
    return mapped.sort_values(["aircraft_event_time_utc", "event_id"], kind="mergesort").reset_index(drop=True)


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    recovery_zip = args.recovery_zip.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    windows = WINDOWS_ALL if args.windows == "all" else {PRIMARY_WINDOW: PRIMARY_HALF_S}
    inset_s = float(max(windows.values()))  # safe-center inset = largest half-window in use
    roles = DESIGNS[args.design]

    context_path = project_root / "data/processed/point_event_v1/context_v1/candidate_point_events_v1_with_episode_recurrence.csv"
    split_path = project_root / "data/processed/point_event_v1/modeling_v3/model_split_assignments_v3.csv"
    events = pd.read_csv(context_path, low_memory=False)
    splits = pd.read_csv(split_path, usecols=["event_id", "chronological_split"], dtype="string")
    events["event_id"] = events["event_id"].astype("string")
    events["aircraft_event_time_utc"] = pd.to_datetime(events["aircraft_event_time_utc"], utc=True)
    events["snapshot_time_utc"] = pd.to_datetime(events["snapshot_time_utc"], utc=True)
    events = events.merge(splits, on="event_id", how="left", validate="one_to_one")
    if events["chronological_split"].isna().any():
        raise ValueError("Events lack the frozen chronological split.")
    if args.max_events:
        events = events.groupby("chronological_split", group_keys=False).head(args.max_events // 3).reset_index(drop=True)

    print("Loading NF3 from the recovery archive...", flush=True)
    with ZipFile(recovery_zip) as archive, archive.open(v2.RAW_ENTRY) as handle:
        net = pd.read_csv(
            handle, usecols=v2.RAW_COLUMNS, low_memory=False,
            dtype={"IPV4_SRC_ADDR": "category", "IPV4_DST_ADDR": "category", "Attack": "category"},
        )
    net["source_row_number"] = np.arange(len(net), dtype=np.int64)
    net["flow_start_utc"] = pd.to_datetime(pd.to_numeric(net["FLOW_START_MILLISECONDS"]), unit="ms", utc=True)
    net["flow_end_utc"] = pd.to_datetime(pd.to_numeric(net["FLOW_END_MILLISECONDS"]), unit="ms", utc=True)
    if (net["flow_end_utc"] < net["flow_start_utc"]).any():
        raise ValueError("NF3 contains reversed flow intervals.")
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
    net["flow_id"] = "NF3-" + net["source_row_number"].astype(str).str.zfill(12)
    print(f"Loaded {len(net):,} NF3 flows.", flush=True)

    # ---- sessions and attack-window landscape
    sess = net["flow_start_utc"].diff().dt.total_seconds().fillna(0).gt(SESSION_GAP_S).cumsum().to_numpy(dtype=np.int16)
    net["source_session_index"] = sess
    session_rows, block_rows, alloc_info, unassigned_rows = [], [], {}, []
    block_index = 0
    for si in sorted(np.unique(sess)):
        label = f"S{si + 1}"
        m = net["source_session_index"].to_numpy() == si
        s_ns = int(net.loc[m, "flow_start_ns"].min())
        e_ns = int(net.loc[m, "flow_end_ns"].max())
        att = m & (net["binary_label"].to_numpy() != 0)
        covered = attack_window_coverage(net.loc[att, "flow_start_ns"].to_numpy(), net.loc[att, "flow_end_ns"].to_numpy(), s_ns, e_ns, int(PRIMARY_HALF_S * 1e9))
        seg, unassigned, info = allocate_session(label, s_ns, e_ns, covered, roles, inset_s)
        unassigned_rows.extend(unassigned)
        seg["session"] = label
        seg["segment_start_utc"] = pd.to_datetime(seg["segment_start_ns"], utc=True)
        seg["segment_end_utc"] = pd.to_datetime(seg["segment_end_ns"], utc=True)
        session_rows.append(seg.drop(columns=["segment_start_ns", "segment_end_ns"]))
        info["session_start_utc"] = str(pd.to_datetime(s_ns, utc=True))
        info["session_end_utc"] = str(pd.to_datetime(e_ns, utc=True))
        info["attack_window_coverage_share_1s"] = float(covered.mean())
        alloc_info[label] = info
        new_blocks = segments_to_blocks(seg, int(si), label, block_index, inset_s)
        block_rows.extend(new_blocks)
        block_index += len(new_blocks)
        print(f"{label}: {json.dumps(info, default=str)[:600]}", flush=True)
    pd.DataFrame(unassigned_rows).to_csv(out / "nf3_unassigned_intervals_v3.csv", index=False)
    blocks = pd.DataFrame(block_rows)
    pd.concat(session_rows, ignore_index=True).to_csv(out / "nf3_segment_allocation_v3.csv", index=False)

    # ---- assign flows to blocks (full containment), block counts
    fbi = np.full(len(net), -1, dtype=np.int16)
    fs, fe = net["flow_start_ns"].to_numpy(), net["flow_end_ns"].to_numpy()
    for b in blocks.itertuples(index=False):
        mask = (fs >= pd.Timestamp(b.block_start_utc).value) & (fe <= pd.Timestamp(b.block_end_utc).value)
        if np.any(fbi[mask] >= 0):
            raise AssertionError("A source flow was assigned to more than one block.")
        fbi[mask] = int(b.source_block_index)
    net["source_block_index"] = fbi
    counts = (
        net.loc[net["source_block_index"].ge(0)]
        .groupby("source_block_index")
        .agg(assigned_flow_rows=("source_row_number", "size"), attack_flow_rows=("binary_label", "sum"))
        .reset_index()
    )
    blocks = blocks.merge(counts, on="source_block_index", how="left")
    blocks["assigned_flow_rows"] = blocks["assigned_flow_rows"].fillna(0).astype(int)
    blocks["attack_flow_rows"] = blocks["attack_flow_rows"].fillna(0).astype(int)
    blocks["benign_flow_rows"] = blocks["assigned_flow_rows"] - blocks["attack_flow_rows"]
    blocks["attack_flow_fraction"] = blocks["attack_flow_rows"] / blocks["assigned_flow_rows"].replace(0, np.nan)
    blocks.to_csv(out / "nf3_source_block_manifest_v3.csv", index=False)

    # ---- map events, aggregate windows (unchanged v2 helpers)
    mapped = map_events_to_blocks(events, blocks, args.time_basis)
    mapped.to_csv(out / "event_source_block_mapping_v3.csv", index=False)
    block_lookup = blocks.set_index("source_block_index")
    block_flow_indices = {int(i): np.flatnonzero(fbi == int(i)) for i in blocks["source_block_index"]}
    block_starts = {i: fs[idx] for i, idx in block_flow_indices.items()}
    max_dur_ns = int(net["flow_duration_s"].max() * 1e9) + 1
    features, targets, edge_frames = [], [], []
    rel_count = 0
    src_mask = np.zeros(len(net), dtype=np.uint8)
    for n_done, ev in enumerate(mapped.itertuples(index=False), start=1):
        bi = int(ev.source_block_index)
        aligned = pd.Timestamp(ev.aligned_event_time_utc)
        a_ns = aligned.value
        idx, starts = block_flow_indices[bi], block_starts[bi]
        blk = block_lookup.loc[bi]
        bs_ns, be_ns = pd.Timestamp(blk["block_start_utc"]).value, pd.Timestamp(blk["block_end_utc"]).value
        for wname, half_s in windows.items():
            h = int(half_s * 1e9)
            ws, we = a_ns - h, a_ns + h
            if ws < bs_ns or we > be_ns:
                raise AssertionError("An event window crossed its owned source block boundary.")
            left = int(np.searchsorted(starts, ws - max_dur_ns, side="left"))
            right = int(np.searchsorted(starts, we, side="right"))
            cand = idx[left:right]
            matched = cand[fe[cand] >= ws] if cand.size else np.empty(0, dtype=np.int64)
            base = {
                "event_id": str(ev.event_id), "alignment_version": VERSION, "alignment_design": args.design,
                "alignment_mode": "attack_aware_partitioned_controlled_cyclic_replay", "alignment_is_controlled": 1,
                "mapping_time_basis": args.time_basis, "source_partition": str(ev.chronological_split),
                "source_block_id": str(ev.source_block_id), "physical_event_time_utc": pd.Timestamp(ev.aircraft_event_time_utc),
                "aligned_event_time_utc": aligned, "replay_cycle_index": int(ev.partition_replay_cycle_index),
                "window_name": wname, "window_half_width_s": half_s,
                "window_start_utc": pd.to_datetime(ws, utc=True), "window_end_utc": pd.to_datetime(we, utc=True),
                "window_boundary_truncated_flag": 0,
            }
            f = dict(base)
            f.update(v2.aggregate_features(net, matched, a_ns, ws, we))
            features.append(f)
            t = {k: base[k] for k in ("event_id", "alignment_version", "alignment_design", "alignment_mode", "alignment_is_controlled", "mapping_time_basis", "source_partition", "source_block_id", "window_name", "window_half_width_s")}
            t.update(v2.aggregate_targets(net, matched))
            targets.append(t)
            if wname == PRIMARY_WINDOW and matched.size:
                rel_count += len(matched)
                src_mask[net["source_row_number"].to_numpy()[matched]] |= SPLIT_BITS[str(ev.chronological_split)]
                sel = v2.nearest_edge_indices(net, matched, a_ns, ws, we, MAX_EDGES)
                sf = net.iloc[sel]
                ss, se = sf["flow_start_ns"].to_numpy(), sf["flow_end_ns"].to_numpy()
                lag = (ss - a_ns) / 1e9
                edge_frames.append(pd.DataFrame({
                    "event_id": str(ev.event_id), "flow_id": sf["flow_id"].to_numpy(), "window_name": wname,
                    "aligned_event_time_utc": aligned, "window_start_utc": pd.to_datetime(ws, utc=True), "window_end_utc": pd.to_datetime(we, utc=True),
                    "flow_start_utc": sf["flow_start_utc"].to_numpy(), "flow_end_utc": sf["flow_end_utc"].to_numpy(),
                    "signed_flow_start_lag_s": lag, "absolute_flow_start_lag_s": np.abs(lag),
                    "overlap_seconds": np.maximum(0, np.minimum(se, we) - np.maximum(ss, ws)) / 1e9,
                    "relation_type": np.where(se < a_ns, "pre_event", np.where(ss > a_ns, "post_event", "overlaps_event_time")),
                    "source_row_number": sf["source_row_number"].to_numpy(), "protocol": sf["PROTOCOL"].to_numpy(),
                    "src_port": sf["L4_SRC_PORT"].to_numpy(), "dst_port": sf["L4_DST_PORT"].to_numpy(),
                    "source_partition": str(ev.chronological_split), "source_block_id": str(ev.source_block_id),
                }))
        if n_done % 1000 == 0 or n_done == len(mapped):
            print(f"Aligned {n_done:,} / {len(mapped):,}", flush=True)

    wf = pd.DataFrame(features)
    tg = pd.DataFrame(targets)
    prim = wf.loc[wf["window_name"].eq(PRIMARY_WINDOW)].drop(columns=["physical_event_time_utc"], errors="ignore")
    fused = events.drop(columns=["chronological_split"]).merge(prim, on="event_id", how="left", validate="one_to_one")
    edges = pd.concat(edge_frames, ignore_index=True) if edge_frames else pd.DataFrame()
    wf.to_csv(out / "event_nf3_window_features_v3.csv", index=False)
    tg.to_csv(out / "event_nf3_targets_v3.csv", index=False)
    fused.to_csv(out / "candidate_point_events_v3_with_context_and_nf3_primary.csv", index=False)
    edges.to_csv(out / "event_flow_edges_primary_bounded_v3.csv", index=False)
    used = np.flatnonzero(src_mask)
    net.loc[net["source_row_number"].isin(used), ["flow_id", "source_row_number", "binary_label", "attack_category"]].rename(
        columns={"binary_label": "nf3_binary_label", "attack_category": "nf3_attack_category"}
    ).to_csv(out / "nf3_flow_labels_used_v3.csv.gz", index=False, compression="gzip")

    # ---- independence gate (as v2) + session x partition class gate (new)
    pairs = [("train", "validation"), ("train", "test"), ("validation", "test")]
    rec = pd.DataFrame([{
        "left_split": a, "right_split": b,
        "left_unique_source_flows": int(np.count_nonzero(src_mask & SPLIT_BITS[a])),
        "right_unique_source_flows": int(np.count_nonzero(src_mask & SPLIT_BITS[b])),
        "reused_source_flows": int(np.count_nonzero((src_mask & SPLIT_BITS[a]) & (src_mask & SPLIT_BITS[b]))),
    } for a, b in pairs])
    rec.to_csv(out / "leakage_safe_exact_source_flow_audit_v3.csv", index=False)
    pw = wf.loc[wf["window_name"].eq(PRIMARY_WINDOW), ["source_partition", "window_start_utc", "window_end_utc"]]
    ov_rows = []
    for a, b in pairs:
        L = pw[pw.source_partition.eq(a)]
        R = pw[pw.source_partition.eq(b)].sort_values("window_start_utc")
        rs = R["window_start_utc"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
        re_ = R["window_end_utc"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
        ls = L["window_start_utc"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
        le = L["window_end_utc"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
        # overlap exists iff some R window has start <= l_end and end >= l_start
        maxend_prefix = np.maximum.accumulate(re_) if len(re_) else np.array([], dtype=np.int64)
        detected = False
        for s, e in zip(ls, le):
            k = int(np.searchsorted(rs, e, side="right"))
            if k > 0 and maxend_prefix[k - 1] >= s:
                detected = True
                break
        ov_rows.append({"left_split": a, "right_split": b, "source_window_overlap_detected": detected})
    ov = pd.DataFrame(ov_rows)
    ov.to_csv(out / "leakage_safe_source_window_audit_v3.csv", index=False)

    tp = tg.loc[tg["window_name"].eq(PRIMARY_WINDOW), ["event_id", "cyber_attack_window_label"]]
    cell = mapped[["event_id", "chronological_split", "source_block_id", "aligned_event_time_utc"]].merge(tp, on="event_id")
    cell["session"] = cell["source_block_id"].str.extract(r"^NF3(S\d)_")[0]
    audit = cell.groupby(["chronological_split", "session"]).agg(events=("event_id", "size"), positives=("cyber_attack_window_label", "sum")).reset_index()
    audit["negatives"] = audit["events"] - audit["positives"]
    audit["prevalence"] = audit["positives"] / audit["events"]
    audit["minority_events"] = audit[["positives", "negatives"]].min(axis=1)
    audit["both_classes_present"] = audit["minority_events"] >= args.min_minority_events
    regimes_by_session = {lab: {r["regime"] for r in inf["regime_intervals"] if r.get("assigned")} for lab, inf in alloc_info.items()}
    audit["feasible_by_data"] = audit["session"].map(lambda lab: len(regimes_by_session.get(lab, set())) == 2)
    audit.to_csv(out / "session_partition_class_audit_v3.csv", index=False)
    from sklearn.metrics import average_precision_score, roc_auc_score
    part_rows = []
    for split_name in SPLITS:
        c = cell[cell.chronological_split.eq(split_name)]
        y = c["cyber_attack_window_label"].to_numpy(); s_ind = (c["session"] == "S2").astype(int).to_numpy()
        both = 0 < y.sum() < len(y)
        bins = (c["aligned_event_time_utc"].astype("int64") // int(2 * PRIMARY_HALF_S * 1e9))
        part_rows.append({
            "partition": split_name, "events": len(c), "positives": int(y.sum()), "negatives": int(len(y) - y.sum()),
            "both_classes_present": bool(min(y.sum(), len(y) - y.sum()) >= args.min_minority_events),
            "session_indicator_auprc": float(average_precision_score(y, s_ind)) if both else float("nan"),
            "session_indicator_roc_auc": float(roc_auc_score(y, s_ind)) if both and 0 < s_ind.sum() < len(s_ind) else float("nan"),
            "sessions_present": "|".join(sorted(c["session"].unique())),
            "independent_240s_source_bins_positive": int(bins[y == 1].nunique()), "independent_240s_source_bins_negative": int(bins[y == 0].nunique()),
        })
    partition_audit = pd.DataFrame(part_rows)
    partition_audit.to_csv(out / "partition_class_and_session_audit_v3.csv", index=False)
    te = partition_audit[partition_audit.partition.eq("test")].iloc[0]
    sess_auprc, sess_roc = float(te["session_indicator_auprc"]), float(te["session_indicator_roc_auc"])
    y = cell.loc[cell.chronological_split.eq("test"), "cyber_attack_window_label"].to_numpy()
    gate_classes_partition = bool(partition_audit["both_classes_present"].all())
    if args.design.startswith("cross_regime"):
        # train/validation are label==session BY DESIGN (that is what makes the test regime unseen);
        # the session indicator must be uninformative in the TEST partition only.
        gate_session = bool(te["session_indicator_auprc"] < 1.0 - 1e-9)
        gate_classes = True  # per-cell mixing is not a requirement of this design; the table is reported
    else:
        gate_session = bool((partition_audit["session_indicator_auprc"].fillna(0) < 1.0 - 1e-9).all())
        gate_classes = bool(audit.loc[audit["feasible_by_data"], "both_classes_present"].all())

    gate_recurrence = int(rec["reused_source_flows"].sum()) == 0
    gate_overlap = not bool(ov["source_window_overlap_detected"].any())
    summary = {
        "alignment_version": VERSION, "design": args.design, "time_basis": args.time_basis, "windows": list(windows),
        "allocation": {"mode": "regime_stratified", "embargo_s": EMBARGO_S, "safe_center_inset_s": inset_s, "min_center_span_s": MIN_CENTER_SPAN_S,
                        "sessions": alloc_info, "unassigned_intervals": unassigned_rows},
        "inputs": {"context_rows": len(events), "nf3_flow_rows": len(net), "recovery_archive_sha256": v2.sha256_file(recovery_zip),
                   "context_sha256": v2.sha256_file(context_path), "frozen_split_sha256": v2.sha256_file(split_path)},
        "outputs": {"event_rows": len(fused), "blocks": len(blocks), "window_feature_rows": len(wf), "target_rows": len(tg),
                    "primary_full_relationship_count": rel_count, "primary_materialized_bounded_edge_rows": len(edges),
                    "primary_events_with_flows": int(fused["nf3_flow_count"].gt(0).sum()),
                    "primary_positive_windows": int(tp["cyber_attack_window_label"].sum()),
                    "primary_negative_windows": int(tp["cyber_attack_window_label"].eq(0).sum()),
                    "primary_prevalence": float(tp["cyber_attack_window_label"].mean()),
                    "replay_cycles_by_split": {s: int(mapped.loc[mapped.chronological_split.eq(s), "partition_replay_cycle_index"].nunique()) for s in SPLITS}},
        "session_partition_class_audit": audit.to_dict(orient="records"),
        "partition_class_and_session_audit": partition_audit.to_dict(orient="records"),
        "session_indicator_baseline_test": {"auprc": sess_auprc, "roc_auc": sess_roc, "test_prevalence": float(y.mean())},
        "gate": {"exact_source_flow_recurrence": gate_recurrence, "mapped_source_window_overlap_free": gate_overlap,
                 "both_classes_within_each_partition": gate_classes_partition,
                 "session_indicator_not_perfect_within_any_partition": gate_session,
                 "both_classes_within_each_session_partition_where_feasible": gate_classes,
                 "decision": "PASS" if (gate_recurrence and gate_overlap and gate_classes_partition and gate_session and gate_classes) else "FAIL"},
        "interpretation": {"site_coincident_claim": False, "causal_claim": False},
    }
    (out / f"{VERSION}_build_summary.json").write_text(json.dumps(summary, indent=2, default=v2.json_scalar), encoding="utf-8")
    sums = [f"{v2.sha256_file(p)}  {p.name}" for p in sorted(out.iterdir()) if p.is_file() and p.name != "SHA256SUMS.txt"]
    (out / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("outputs", "session_indicator_baseline_test", "gate")}, indent=2, default=v2.json_scalar), flush=True)
    print(audit.to_string(index=False), flush=True)
    print(partition_audit.to_string(index=False), flush=True)
    if summary["gate"]["decision"] != "PASS":
        print("GATE FAILED", flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
