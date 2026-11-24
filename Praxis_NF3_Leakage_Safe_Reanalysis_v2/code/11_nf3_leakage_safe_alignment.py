#!/usr/bin/env python3
"""Create a leakage-safe NF3 controlled replay aligned to frozen aircraft splits.

The original aircraft event split assignments are retained. The two active NF3
capture sessions are each divided into disjoint train/validation/test source-time
blocks with a 20-minute embargo between adjacent partitions. Event centers are
kept at least 10 minutes from block boundaries, matching the largest configured
half-window. All aggregate features and targets use every matching flow in the
owned block. A label-blind maximum of 16 nearest edges per event is materialized
for downstream graph modeling; the complete relationship count and exact source-
flow split mask are audited in memory.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import warnings
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
import pandas as pd


VERSION = "point_event_nf3_alignment_leakage_safe_v2"
WINDOWS = {"primary_120s": 120, "sensitivity_300s": 300, "sensitivity_600s": 600}
PRIMARY_WINDOW = "primary_120s"
MAX_HALF_WINDOW_S = max(WINDOWS.values())
INTER_PARTITION_EMBARGO_S = 1_200
SESSION_GAP_S = 6 * 60 * 60
MAX_MATERIALIZED_EDGES_PER_EVENT = 16
SPLITS = ("train", "validation", "test")
SPLIT_FRACTIONS = {"train": 0.70, "validation": 0.15, "test": 0.15}
SPLIT_BITS = {"train": 1, "validation": 2, "test": 4}
BENIGN_VALUES = {"BENIGN", "NORMAL", "0", "FALSE"}

RAW_ENTRY = "Project/Jupyter Notebook/data/Netflow/NF-UNSW-NB15-v3.csv"
RAW_COLUMNS = [
    "FLOW_START_MILLISECONDS",
    "FLOW_END_MILLISECONDS",
    "IPV4_SRC_ADDR",
    "L4_SRC_PORT",
    "IPV4_DST_ADDR",
    "L4_DST_PORT",
    "PROTOCOL",
    "IN_BYTES",
    "IN_PKTS",
    "OUT_BYTES",
    "OUT_PKTS",
    "SRC_TO_DST_IAT_MIN",
    "SRC_TO_DST_IAT_MAX",
    "SRC_TO_DST_IAT_AVG",
    "SRC_TO_DST_IAT_STDDEV",
    "DST_TO_SRC_IAT_MIN",
    "DST_TO_SRC_IAT_MAX",
    "DST_TO_SRC_IAT_AVG",
    "DST_TO_SRC_IAT_STDDEV",
    "Label",
    "Attack",
]
IAT_COLUMNS = [
    "SRC_TO_DST_IAT_MIN",
    "SRC_TO_DST_IAT_MAX",
    "SRC_TO_DST_IAT_AVG",
    "SRC_TO_DST_IAT_STDDEV",
    "DST_TO_SRC_IAT_MIN",
    "DST_TO_SRC_IAT_MAX",
    "DST_TO_SRC_IAT_AVG",
    "DST_TO_SRC_IAT_STDDEV",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--recovery-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def json_scalar(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def finite_stat(values: np.ndarray, method: str) -> float:
    numeric = np.asarray(values, dtype=float)
    numeric = numeric[np.isfinite(numeric)]
    if numeric.size == 0:
        return float("nan")
    if method == "sum":
        return float(numeric.sum())
    if method == "mean":
        return float(numeric.mean())
    if method == "median":
        return float(np.median(numeric))
    if method == "std":
        return float(numeric.std(ddof=0))
    if method == "max":
        return float(numeric.max())
    if method == "min":
        return float(numeric.min())
    raise ValueError(method)


def normalized_iat_name(column: str) -> str:
    return column.lower()


def aggregate_features(
    net: pd.DataFrame,
    matched_indices: np.ndarray,
    event_time_ns: int,
    window_start_ns: int,
    window_end_ns: int,
) -> dict[str, Any]:
    matched = net.iloc[matched_indices]
    count = len(matched)
    feature: dict[str, Any] = {
        "nf3_flow_count": count,
        "nf3_flow_duration_sum_s": 0.0,
        "nf3_flow_duration_mean_s": np.nan,
        "nf3_flow_duration_median_s": np.nan,
        "nf3_flow_duration_std_s": np.nan,
        "nf3_flow_duration_max_s": np.nan,
        "nf3_packet_total_sum": 0.0,
        "nf3_packet_total_mean": np.nan,
        "nf3_packet_total_max": np.nan,
        "nf3_byte_total_sum": 0.0,
        "nf3_byte_total_mean": np.nan,
        "nf3_byte_total_max": np.nan,
        "nf3_protocol_unique_count": 0,
        "nf3_src_ip_unique_count": 0,
        "nf3_dst_ip_unique_count": 0,
        "nf3_src_port_unique_count": 0,
        "nf3_dst_port_unique_count": 0,
        "nf3_pre_event_flow_count": 0,
        "nf3_at_event_flow_count": 0,
        "nf3_post_event_flow_count": 0,
        "nf3_nearest_flow_start_lag_s": np.nan,
        "nf3_first_flow_start_lag_s": np.nan,
        "nf3_last_flow_start_lag_s": np.nan,
        "nf3_total_overlap_duration_s": 0.0,
        "nf3_mean_overlap_duration_s": np.nan,
        "nf3_max_overlap_duration_s": np.nan,
    }
    for column in IAT_COLUMNS:
        name = normalized_iat_name(column)
        feature[f"nf3_iat_{name}_mean"] = np.nan
        feature[f"nf3_iat_{name}_std"] = np.nan
    if count == 0:
        return feature

    durations = matched["flow_duration_s"].to_numpy(dtype=float)
    packets = matched["packet_total"].to_numpy(dtype=float)
    bytes_total = matched["byte_total"].to_numpy(dtype=float)
    feature.update(
        {
            "nf3_flow_duration_sum_s": finite_stat(durations, "sum"),
            "nf3_flow_duration_mean_s": finite_stat(durations, "mean"),
            "nf3_flow_duration_median_s": finite_stat(durations, "median"),
            "nf3_flow_duration_std_s": finite_stat(durations, "std"),
            "nf3_flow_duration_max_s": finite_stat(durations, "max"),
            "nf3_packet_total_sum": finite_stat(packets, "sum"),
            "nf3_packet_total_mean": finite_stat(packets, "mean"),
            "nf3_packet_total_max": finite_stat(packets, "max"),
            "nf3_byte_total_sum": finite_stat(bytes_total, "sum"),
            "nf3_byte_total_mean": finite_stat(bytes_total, "mean"),
            "nf3_byte_total_max": finite_stat(bytes_total, "max"),
            "nf3_protocol_unique_count": int(matched["PROTOCOL"].nunique(dropna=True)),
            "nf3_src_ip_unique_count": int(matched["IPV4_SRC_ADDR"].nunique(dropna=True)),
            "nf3_dst_ip_unique_count": int(matched["IPV4_DST_ADDR"].nunique(dropna=True)),
            "nf3_src_port_unique_count": int(matched["L4_SRC_PORT"].nunique(dropna=True)),
            "nf3_dst_port_unique_count": int(matched["L4_DST_PORT"].nunique(dropna=True)),
        }
    )

    starts = matched["flow_start_ns"].to_numpy(dtype=np.int64)
    ends = matched["flow_end_ns"].to_numpy(dtype=np.int64)
    lags = (starts - event_time_ns) / 1e9
    feature["nf3_pre_event_flow_count"] = int(np.count_nonzero(ends < event_time_ns))
    feature["nf3_at_event_flow_count"] = int(
        np.count_nonzero((starts <= event_time_ns) & (ends >= event_time_ns))
    )
    feature["nf3_post_event_flow_count"] = int(np.count_nonzero(starts > event_time_ns))
    feature["nf3_nearest_flow_start_lag_s"] = float(lags[np.argmin(np.abs(lags))])
    feature["nf3_first_flow_start_lag_s"] = float(lags.min())
    feature["nf3_last_flow_start_lag_s"] = float(lags.max())
    overlap_ns = np.maximum(
        0,
        np.minimum(ends, window_end_ns) - np.maximum(starts, window_start_ns),
    )
    overlap_s = overlap_ns / 1e9
    feature["nf3_total_overlap_duration_s"] = float(overlap_s.sum())
    feature["nf3_mean_overlap_duration_s"] = float(overlap_s.mean())
    feature["nf3_max_overlap_duration_s"] = float(overlap_s.max())
    for column in IAT_COLUMNS:
        name = normalized_iat_name(column)
        values = matched[column].to_numpy(dtype=float)
        feature[f"nf3_iat_{name}_mean"] = finite_stat(values, "mean")
        feature[f"nf3_iat_{name}_std"] = finite_stat(values, "std")
    return feature


def aggregate_targets(net: pd.DataFrame, matched_indices: np.ndarray) -> dict[str, Any]:
    if matched_indices.size == 0:
        return {
            "nf3_binary_label_available_flag": 1,
            "nf3_attack_category_available_flag": 1,
            "cyber_attack_window_label": 0,
            "nf3_attack_flow_count": 0,
            "nf3_benign_flow_count": 0,
            "nf3_unknown_binary_label_count": 0,
            "nf3_attack_flow_fraction": 0.0,
            "nf3_distinct_attack_category_count": 0,
            "nf3_attack_categories_present": None,
            "nf3_attack_category_counts_json": None,
        }
    matched = net.iloc[matched_indices]
    labels = matched["binary_label"].to_numpy(dtype=np.int8)
    attack_mask = labels != 0
    attack_count = int(np.count_nonzero(attack_mask))
    benign_count = int(labels.size - attack_count)
    categories = (
        matched.loc[attack_mask, "attack_category"]
        .astype("string")
        .str.strip()
        .dropna()
    )
    categories = categories.loc[~categories.str.upper().isin(BENIGN_VALUES)]
    counts = categories.value_counts().sort_index()
    count_dict = {str(key): int(value) for key, value in counts.items()}
    return {
        "nf3_binary_label_available_flag": 1,
        "nf3_attack_category_available_flag": 1,
        "cyber_attack_window_label": int(attack_count > 0),
        "nf3_attack_flow_count": attack_count,
        "nf3_benign_flow_count": benign_count,
        "nf3_unknown_binary_label_count": 0,
        "nf3_attack_flow_fraction": float(attack_count / labels.size),
        "nf3_distinct_attack_category_count": len(count_dict),
        "nf3_attack_categories_present": "|".join(count_dict) if count_dict else None,
        "nf3_attack_category_counts_json": json.dumps(count_dict, sort_keys=True) if count_dict else None,
    }


def nearest_edge_indices(
    net: pd.DataFrame,
    matched_indices: np.ndarray,
    event_time_ns: int,
    window_start_ns: int,
    window_end_ns: int,
    maximum: int,
) -> np.ndarray:
    starts = net.iloc[matched_indices]["flow_start_ns"].to_numpy(dtype=np.int64)
    ends = net.iloc[matched_indices]["flow_end_ns"].to_numpy(dtype=np.int64)
    source_rows = net.iloc[matched_indices]["source_row_number"].to_numpy(dtype=np.int64)
    abs_lags = np.abs(starts - event_time_ns)
    overlaps = np.maximum(
        0,
        np.minimum(ends, window_end_ns) - np.maximum(starts, window_start_ns),
    )
    order = np.lexsort((source_rows, -overlaps, abs_lags))
    return matched_indices[order[:maximum]]


def build_source_blocks(net: pd.DataFrame) -> pd.DataFrame:
    sorted_starts = net["flow_start_utc"]
    session_id = sorted_starts.diff().dt.total_seconds().fillna(0).gt(SESSION_GAP_S).cumsum()
    net["source_session_index"] = session_id.to_numpy(dtype=np.int16)
    sessions = (
        net.groupby("source_session_index", sort=True)
        .agg(session_start_utc=("flow_start_utc", "min"), session_end_utc=("flow_end_utc", "max"), flow_rows=("source_row_number", "size"))
        .reset_index()
    )
    if len(sessions) < 2:
        warnings.warn("Only one NF3 active capture session was detected.")

    rows: list[dict[str, Any]] = []
    block_index = 0
    for session in sessions.itertuples(index=False):
        start = pd.Timestamp(session.session_start_utc)
        end = pd.Timestamp(session.session_end_utc)
        total_s = (end - start).total_seconds()
        allocatable_s = total_s - INTER_PARTITION_EMBARGO_S * (len(SPLITS) - 1)
        if allocatable_s <= len(SPLITS) * 2 * MAX_HALF_WINDOW_S:
            raise ValueError("Detected source session is too short for leakage-safe partitioning.")
        cursor = start
        for position, split_name in enumerate(SPLITS):
            duration_s = allocatable_s * SPLIT_FRACTIONS[split_name]
            block_end = cursor + pd.Timedelta(seconds=duration_s)
            safe_start = cursor + pd.Timedelta(seconds=MAX_HALF_WINDOW_S)
            safe_end = block_end - pd.Timedelta(seconds=MAX_HALF_WINDOW_S)
            rows.append(
                {
                    "source_block_index": block_index,
                    "source_block_id": f"NF3S{int(session.source_session_index) + 1}_{split_name}",
                    "source_session_index": int(session.source_session_index),
                    "source_partition": split_name,
                    "block_start_utc": cursor,
                    "block_end_utc": block_end,
                    "safe_center_start_utc": safe_start,
                    "safe_center_end_utc": safe_end,
                    "block_duration_s": duration_s,
                    "safe_center_span_s": (safe_end - safe_start).total_seconds(),
                    "preceding_embargo_s": 0 if position == 0 else INTER_PARTITION_EMBARGO_S,
                    "following_embargo_s": 0 if position == len(SPLITS) - 1 else INTER_PARTITION_EMBARGO_S,
                }
            )
            block_index += 1
            cursor = block_end
            if position < len(SPLITS) - 1:
                cursor += pd.Timedelta(seconds=INTER_PARTITION_EMBARGO_S)
        if abs((cursor - end).total_seconds()) > 1e-6:
            raise AssertionError("Source block durations do not close on the session boundary.")
    return pd.DataFrame(rows)


def map_events_to_blocks(events: pd.DataFrame, blocks: pd.DataFrame) -> pd.DataFrame:
    mapped_parts: list[pd.DataFrame] = []
    for split_name in SPLITS:
        part = events.loc[events["chronological_split"].eq(split_name)].copy()
        part = part.sort_values(["aircraft_event_time_utc", "event_id"], kind="mergesort")
        owned = blocks.loc[blocks["source_partition"].eq(split_name)].sort_values("source_session_index")
        spans_ns = (
            pd.to_datetime(owned["safe_center_end_utc"], utc=True).to_numpy(dtype="datetime64[ns]").astype(np.int64)
            - pd.to_datetime(owned["safe_center_start_utc"], utc=True).to_numpy(dtype="datetime64[ns]").astype(np.int64)
        )
        cumulative = np.cumsum(spans_ns)
        total_span_ns = int(cumulative[-1])
        physical_ns = part["aircraft_event_time_utc"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
        elapsed_ns = physical_ns - int(physical_ns.min())
        virtual_ns = np.mod(elapsed_ns, total_span_ns)
        cycle = elapsed_ns // total_span_ns
        positions = np.searchsorted(cumulative, virtual_ns, side="right")
        previous = np.where(positions == 0, 0, cumulative[positions - 1])
        owned_records = owned.reset_index(drop=True)
        block_starts_ns = (
            pd.to_datetime(owned_records["safe_center_start_utc"], utc=True)
            .to_numpy(dtype="datetime64[ns]")
            .astype(np.int64)
        )
        aligned_ns = block_starts_ns[positions] + (virtual_ns - previous)
        part["aligned_event_time_utc"] = pd.to_datetime(aligned_ns, utc=True)
        part["source_block_index"] = owned_records.loc[positions, "source_block_index"].to_numpy(dtype=int)
        part["source_block_id"] = owned_records.loc[positions, "source_block_id"].to_numpy(dtype=str)
        part["partition_replay_cycle_index"] = cycle.astype(np.int64)
        mapped_parts.append(part)
    mapped = pd.concat(mapped_parts, ignore_index=True)
    return mapped.sort_values(["aircraft_event_time_utc", "event_id"], kind="mergesort").reset_index(drop=True)


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    recovery_zip = args.recovery_zip.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    context_path = (
        project_root
        / "data"
        / "processed"
        / "point_event_v1"
        / "context_v1"
        / "candidate_point_events_v1_with_episode_recurrence.csv"
    )
    split_path = (
        project_root
        / "data"
        / "processed"
        / "point_event_v1"
        / "modeling_v3"
        / "model_split_assignments_v3.csv"
    )
    if not context_path.is_file() or not split_path.is_file() or not recovery_zip.is_file():
        raise FileNotFoundError("Required recovered context, split, or raw NF3 archive is missing.")

    events = pd.read_csv(context_path, low_memory=False)
    split_assignments = pd.read_csv(
        split_path,
        usecols=["event_id", "chronological_split"],
        dtype={"event_id": "string", "chronological_split": "string"},
    )
    events["event_id"] = events["event_id"].astype("string")
    events["aircraft_event_time_utc"] = pd.to_datetime(events["aircraft_event_time_utc"], utc=True, errors="raise")
    events = events.merge(split_assignments, on="event_id", how="left", validate="one_to_one")
    if events["chronological_split"].isna().any():
        raise ValueError("One or more aircraft events lack the frozen chronological split.")
    if set(events["chronological_split"].unique()) != set(SPLITS):
        raise ValueError("Unexpected frozen chronological split labels.")

    print("Loading the complete NF3 source table from the verified recovery archive...", flush=True)
    with ZipFile(recovery_zip) as archive, archive.open(RAW_ENTRY) as handle:
        net = pd.read_csv(handle, usecols=RAW_COLUMNS, low_memory=False)
    net["source_row_number"] = np.arange(len(net), dtype=np.int64)
    net["flow_start_utc"] = pd.to_datetime(
        pd.to_numeric(net["FLOW_START_MILLISECONDS"], errors="raise"), unit="ms", utc=True
    )
    net["flow_end_utc"] = pd.to_datetime(
        pd.to_numeric(net["FLOW_END_MILLISECONDS"], errors="raise"), unit="ms", utc=True
    )
    if (net["flow_end_utc"] < net["flow_start_utc"]).any():
        raise ValueError("NF3 contains reversed flow intervals.")
    net["flow_duration_s"] = (net["flow_end_utc"] - net["flow_start_utc"]).dt.total_seconds()
    net["packet_total"] = pd.to_numeric(net["IN_PKTS"], errors="coerce") + pd.to_numeric(net["OUT_PKTS"], errors="coerce")
    net["byte_total"] = pd.to_numeric(net["IN_BYTES"], errors="coerce") + pd.to_numeric(net["OUT_BYTES"], errors="coerce")
    for column in IAT_COLUMNS + ["L4_SRC_PORT", "L4_DST_PORT", "PROTOCOL"]:
        net[column] = pd.to_numeric(net[column], errors="coerce")
    net["binary_label"] = pd.to_numeric(net["Label"], errors="raise").astype(np.int8)
    net["attack_category"] = net["Attack"].astype("string")
    net = net.sort_values(["flow_start_utc", "flow_end_utc", "source_row_number"], kind="mergesort").reset_index(drop=True)
    net["flow_start_ns"] = net["flow_start_utc"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    net["flow_end_ns"] = net["flow_end_utc"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    net["flow_id"] = net["source_row_number"].map(lambda value: f"NF3-{int(value):012d}")
    print(f"Loaded {len(net):,} valid NF3 flows.", flush=True)

    blocks = build_source_blocks(net)
    flow_block_index = np.full(len(net), -1, dtype=np.int16)
    for block in blocks.itertuples(index=False):
        start_ns = pd.Timestamp(block.block_start_utc).value
        end_ns = pd.Timestamp(block.block_end_utc).value
        mask = (net["flow_start_ns"].to_numpy() >= start_ns) & (net["flow_end_ns"].to_numpy() <= end_ns)
        if np.any(flow_block_index[mask] >= 0):
            raise AssertionError("A source flow was assigned to more than one leakage-safe block.")
        flow_block_index[mask] = int(block.source_block_index)
    net["source_block_index"] = flow_block_index

    block_counts = (
        net.loc[net["source_block_index"].ge(0)]
        .groupby("source_block_index")
        .agg(
            assigned_flow_rows=("source_row_number", "size"),
            attack_flow_rows=("binary_label", "sum"),
            first_assigned_flow_start_utc=("flow_start_utc", "min"),
            last_assigned_flow_end_utc=("flow_end_utc", "max"),
        )
        .reset_index()
    )
    blocks = blocks.merge(block_counts, on="source_block_index", how="left", validate="one_to_one")
    blocks["assigned_flow_rows"] = blocks["assigned_flow_rows"].fillna(0).astype(int)
    blocks["attack_flow_rows"] = blocks["attack_flow_rows"].fillna(0).astype(int)
    blocks["benign_flow_rows"] = blocks["assigned_flow_rows"] - blocks["attack_flow_rows"]
    blocks["attack_flow_fraction"] = blocks["attack_flow_rows"] / blocks["assigned_flow_rows"].replace(0, np.nan)
    blocks.to_csv(out / "nf3_source_block_manifest_v2.csv", index=False)

    mapped = map_events_to_blocks(events, blocks)
    mapped.to_csv(out / "event_source_block_mapping_v2.csv", index=False)
    block_lookup = blocks.set_index("source_block_index")

    block_flow_indices: dict[int, np.ndarray] = {}
    block_start_arrays: dict[int, np.ndarray] = {}
    maximum_flow_duration_ns = int(net["flow_duration_s"].max() * 1e9) + 1
    for block_index in blocks["source_block_index"].astype(int):
        indices = np.flatnonzero(net["source_block_index"].to_numpy() == block_index)
        block_flow_indices[block_index] = indices
        block_start_arrays[block_index] = net.iloc[indices]["flow_start_ns"].to_numpy(dtype=np.int64)

    feature_records: list[dict[str, Any]] = []
    target_records: list[dict[str, Any]] = []
    bounded_edge_frames: list[pd.DataFrame] = []
    primary_relationship_count = 0
    primary_source_mask = np.zeros(len(net), dtype=np.uint8)

    for event_number, event in enumerate(mapped.itertuples(index=False), start=1):
        event_id = str(event.event_id)
        split_name = str(event.chronological_split)
        block_index = int(event.source_block_index)
        block_id = str(event.source_block_id)
        aligned_time = pd.Timestamp(event.aligned_event_time_utc)
        aligned_ns = aligned_time.value
        physical_time = pd.Timestamp(event.aircraft_event_time_utc)
        indices = block_flow_indices[block_index]
        starts = block_start_arrays[block_index]

        for window_name, half_width_s in WINDOWS.items():
            half_width_ns = int(half_width_s * 1e9)
            window_start_ns = aligned_ns - half_width_ns
            window_end_ns = aligned_ns + half_width_ns
            block = block_lookup.loc[block_index]
            if window_start_ns < pd.Timestamp(block["block_start_utc"]).value or window_end_ns > pd.Timestamp(block["block_end_utc"]).value:
                raise AssertionError("A leakage-safe event window crossed its owned source block boundary.")
            left = int(np.searchsorted(starts, window_start_ns - maximum_flow_duration_ns, side="left"))
            right = int(np.searchsorted(starts, window_end_ns, side="right"))
            candidate_indices = indices[left:right]
            if candidate_indices.size:
                candidate_ends = net.iloc[candidate_indices]["flow_end_ns"].to_numpy(dtype=np.int64)
                matched_indices = candidate_indices[candidate_ends >= window_start_ns]
            else:
                matched_indices = np.empty(0, dtype=np.int64)

            base = {
                "event_id": event_id,
                "alignment_version": VERSION,
                "alignment_mode": "partitioned_controlled_cyclic_replay",
                "alignment_is_controlled": 1,
                "source_partition": split_name,
                "source_block_id": block_id,
                "physical_event_time_utc": physical_time,
                "aligned_event_time_utc": aligned_time,
                "alignment_time_shift_s": (aligned_time - physical_time).total_seconds(),
                "replay_cycle_index": int(event.partition_replay_cycle_index),
                "window_name": window_name,
                "window_half_width_s": half_width_s,
                "window_start_utc": pd.to_datetime(window_start_ns, utc=True),
                "window_end_utc": pd.to_datetime(window_end_ns, utc=True),
                "window_boundary_truncated_flag": 0,
            }
            feature = dict(base)
            feature.update(aggregate_features(net, matched_indices, aligned_ns, window_start_ns, window_end_ns))
            feature_records.append(feature)
            target = {key: base[key] for key in [
                "event_id", "alignment_version", "alignment_mode", "alignment_is_controlled",
                "source_partition", "source_block_id", "window_name", "window_half_width_s",
            ]}
            target.update(aggregate_targets(net, matched_indices))
            target_records.append(target)

            if window_name == PRIMARY_WINDOW:
                primary_relationship_count += len(matched_indices)
                if matched_indices.size:
                    primary_source_mask[net.iloc[matched_indices]["source_row_number"].to_numpy(dtype=np.int64)] |= SPLIT_BITS[split_name]
                    selected = nearest_edge_indices(
                        net,
                        matched_indices,
                        aligned_ns,
                        window_start_ns,
                        window_end_ns,
                        MAX_MATERIALIZED_EDGES_PER_EVENT,
                    )
                    selected_frame = net.iloc[selected]
                    selected_starts = selected_frame["flow_start_ns"].to_numpy(dtype=np.int64)
                    selected_ends = selected_frame["flow_end_ns"].to_numpy(dtype=np.int64)
                    signed_lag_s = (selected_starts - aligned_ns) / 1e9
                    overlap_s = np.maximum(
                        0,
                        np.minimum(selected_ends, window_end_ns) - np.maximum(selected_starts, window_start_ns),
                    ) / 1e9
                    relation = np.where(
                        selected_ends < aligned_ns,
                        "pre_event",
                        np.where(selected_starts > aligned_ns, "post_event", "overlaps_event_time"),
                    )
                    bounded_edge_frames.append(
                        pd.DataFrame(
                            {
                                "event_id": event_id,
                                "flow_id": selected_frame["flow_id"].to_numpy(),
                                "window_name": window_name,
                                "aligned_event_time_utc": aligned_time,
                                "window_start_utc": pd.to_datetime(window_start_ns, utc=True),
                                "window_end_utc": pd.to_datetime(window_end_ns, utc=True),
                                "flow_start_utc": selected_frame["flow_start_utc"].to_numpy(),
                                "flow_end_utc": selected_frame["flow_end_utc"].to_numpy(),
                                "signed_flow_start_lag_s": signed_lag_s,
                                "absolute_flow_start_lag_s": np.abs(signed_lag_s),
                                "overlap_seconds": overlap_s,
                                "relation_type": relation,
                                "source_row_number": selected_frame["source_row_number"].to_numpy(dtype=np.int64),
                                "protocol": selected_frame["PROTOCOL"].to_numpy(),
                                "src_port": selected_frame["L4_SRC_PORT"].to_numpy(),
                                "dst_port": selected_frame["L4_DST_PORT"].to_numpy(),
                                "source_partition": split_name,
                                "source_block_id": block_id,
                            }
                        )
                    )
        if event_number % 500 == 0 or event_number == len(mapped):
            print(f"Aligned {event_number:,} / {len(mapped):,} events...", flush=True)

    window_features = pd.DataFrame(feature_records)
    targets = pd.DataFrame(target_records)
    primary_features = window_features.loc[window_features["window_name"].eq(PRIMARY_WINDOW)].copy()
    primary_features = primary_features.drop(columns=["physical_event_time_utc"], errors="ignore")
    context_without_split = events.drop(columns=["chronological_split"], errors="ignore")
    primary_fused = context_without_split.merge(primary_features, on="event_id", how="left", validate="one_to_one")
    bounded_edges = pd.concat(bounded_edge_frames, ignore_index=True) if bounded_edge_frames else pd.DataFrame()

    window_features.to_csv(out / "event_nf3_window_features_v2.csv", index=False)
    targets.to_csv(out / "event_nf3_targets_v2.csv", index=False)
    primary_fused.to_csv(out / "candidate_point_events_v2_with_context_and_nf3_primary.csv", index=False)
    bounded_edges.to_csv(out / "event_flow_edges_primary_bounded_v2.csv", index=False)

    used_source_rows = np.flatnonzero(primary_source_mask)
    label_export = net.loc[
        net["source_row_number"].isin(used_source_rows),
        ["flow_id", "source_row_number", "binary_label", "attack_category"],
    ].rename(columns={"binary_label": "nf3_binary_label", "attack_category": "nf3_attack_category"})
    label_export.to_csv(out / "nf3_flow_labels_used_v2.csv.gz", index=False, compression="gzip")

    pair_rows: list[dict[str, Any]] = []
    for left_name, right_name in (("train", "validation"), ("train", "test"), ("validation", "test")):
        left = (primary_source_mask & SPLIT_BITS[left_name]) > 0
        right = (primary_source_mask & SPLIT_BITS[right_name]) > 0
        pair_rows.append(
            {
                "left_split": left_name,
                "right_split": right_name,
                "left_unique_source_flows": int(np.count_nonzero(left)),
                "right_unique_source_flows": int(np.count_nonzero(right)),
                "reused_source_flows": int(np.count_nonzero(left & right)),
            }
        )
    source_recurrence = pd.DataFrame(pair_rows)
    source_recurrence.to_csv(out / "leakage_safe_exact_source_flow_audit_v2.csv", index=False)

    window_overlap_rows: list[dict[str, Any]] = []
    for left_name, right_name in (("train", "validation"), ("train", "test"), ("validation", "test")):
        left_intervals = window_features.loc[
            window_features["source_partition"].eq(left_name) & window_features["window_name"].eq(PRIMARY_WINDOW),
            ["window_start_utc", "window_end_utc"],
        ]
        right_intervals = window_features.loc[
            window_features["source_partition"].eq(right_name) & window_features["window_name"].eq(PRIMARY_WINDOW),
            ["window_start_utc", "window_end_utc"],
        ]
        overlap_detected = False
        for lrow in left_intervals.itertuples(index=False):
            if ((right_intervals["window_start_utc"] <= lrow.window_end_utc) & (right_intervals["window_end_utc"] >= lrow.window_start_utc)).any():
                overlap_detected = True
                break
        window_overlap_rows.append(
            {"left_split": left_name, "right_split": right_name, "source_window_overlap_detected": overlap_detected}
        )
    source_window_audit = pd.DataFrame(window_overlap_rows)
    source_window_audit.to_csv(out / "leakage_safe_source_window_audit_v2.csv", index=False)

    if source_recurrence["reused_source_flows"].gt(0).any() or source_window_audit["source_window_overlap_detected"].any():
        raise AssertionError("Leakage-safe v2 construction failed its post-build independence gate.")

    target_primary = targets.loc[targets["window_name"].eq(PRIMARY_WINDOW)]
    summary = {
        "alignment_version": VERSION,
        "status": "corrected_leakage_safe_rerun",
        "frozen_aircraft_splits_retained": True,
        "mapping": {
            "source_sessions_detected": int(blocks["source_session_index"].nunique()),
            "source_blocks": len(blocks),
            "inter_partition_embargo_s": INTER_PARTITION_EMBARGO_S,
            "safe_center_inset_s": MAX_HALF_WINDOW_S,
            "windows": WINDOWS,
            "materialized_edges_per_event_maximum": MAX_MATERIALIZED_EDGES_PER_EVENT,
            "full_edge_materialization": False,
            "full_edge_note": (
                "All relationships were used for aggregation and the recurrence mask. Only the 16 "
                "label-blind nearest relationships per event were materialized because the active-session "
                "remapping produces a very large redundant edge table and downstream R-GCN code bounds at 16."
            ),
        },
        "inputs": {
            "context_rows": len(events),
            "nf3_flow_rows": len(net),
            "recovery_archive_sha256": sha256_file(recovery_zip),
            "context_sha256": sha256_file(context_path),
            "frozen_split_sha256": sha256_file(split_path),
        },
        "outputs": {
            "event_rows": len(primary_fused),
            "window_feature_rows": len(window_features),
            "target_rows": len(targets),
            "primary_full_relationship_count": primary_relationship_count,
            "primary_materialized_bounded_edge_rows": len(bounded_edges),
            "primary_events_with_flows": int(primary_fused["nf3_flow_count"].gt(0).sum()),
            "primary_positive_windows": int(target_primary["cyber_attack_window_label"].sum()),
            "primary_negative_windows": int(target_primary["cyber_attack_window_label"].eq(0).sum()),
            "primary_prevalence": float(target_primary["cyber_attack_window_label"].mean()),
        },
        "independence_gate": {
            "exact_source_flow_recurrence": 0,
            "mapped_source_window_overlap": False,
            "decision": "PASS",
        },
        "interpretation": {
            "site_coincident_claim": False,
            "causal_claim": False,
            "statement": (
                "Physical/cyber relationships remain controlled, modeled pairings. Leakage-safe source "
                "partitioning does not establish historical co-occurrence or causation."
            ),
        },
    }
    (out / "point_event_nf3_alignment_leakage_safe_v2_build_summary.json").write_text(
        json.dumps(summary, indent=2, default=json_scalar), encoding="utf-8"
    )

    checksums = []
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            checksums.append(f"{sha256_file(path)}  {path.name}")
    (out / "SHA256SUMS.txt").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    print(json.dumps(summary["outputs"], indent=2), flush=True)
    print("Leakage-safe v2 independence gate: PASS", flush=True)


if __name__ == "__main__":
    main()
