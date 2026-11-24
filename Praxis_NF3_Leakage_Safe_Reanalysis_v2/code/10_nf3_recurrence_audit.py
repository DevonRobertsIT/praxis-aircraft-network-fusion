#!/usr/bin/env python3
"""Audit NF3 source recurrence across the frozen chronological model splits.

This script is intentionally read-only with respect to the frozen v1 artifacts. It
streams the complete primary event-flow edge table, audits exact NF3 flow reuse,
source-time-neighborhood reuse, replay-cycle coverage, and overlap of the mapped
source-time windows, then writes a versioned audit package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable
from zipfile import ZipFile

import numpy as np
import pandas as pd


SPLITS = ("train", "validation", "test")
SPLIT_BITS = {"train": 1, "validation": 2, "test": 4}
PAIRINGS = (("train", "validation"), ("train", "test"), ("validation", "test"))
BIN_MINUTES = (1, 2, 5, 10, 30, 60)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def merge_intervals(starts_ns: np.ndarray, ends_ns: np.ndarray) -> np.ndarray:
    order = np.argsort(starts_ns, kind="mergesort")
    starts = starts_ns[order]
    ends = ends_ns[order]
    merged: list[tuple[int, int]] = []
    for start, end in zip(starts.tolist(), ends.tolist()):
        if not merged or start > merged[-1][1]:
            merged.append((int(start), int(end)))
        elif end > merged[-1][1]:
            merged[-1] = (merged[-1][0], int(end))
    return np.asarray(merged, dtype=np.int64)


def interval_length_ns(intervals: np.ndarray) -> int:
    if intervals.size == 0:
        return 0
    return int(np.sum(intervals[:, 1] - intervals[:, 0], dtype=np.int64))


def intersection_length_ns(left: np.ndarray, right: np.ndarray) -> int:
    i = j = 0
    total = 0
    while i < len(left) and j < len(right):
        start = max(int(left[i, 0]), int(right[j, 0]))
        end = min(int(left[i, 1]), int(right[j, 1]))
        if end > start:
            total += end - start
        if left[i, 1] <= right[j, 1]:
            i += 1
        else:
            j += 1
    return total


def bin_ids(values: pd.Series, minutes: int) -> np.ndarray:
    timestamps = pd.to_datetime(values, utc=True, errors="raise")
    ns = timestamps.to_numpy(dtype="datetime64[ns]").astype(np.int64, copy=False)
    width_ns = int(pd.Timedelta(minutes=minutes).value)
    return np.unique(ns // width_ns)


def pairwise_set_rows(
    sets_by_split: dict[str, set[int]],
    domain: str,
    bin_minutes: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for left, right in PAIRINGS:
        left_set = sets_by_split[left]
        right_set = sets_by_split[right]
        overlap = left_set & right_set
        denominator = min(len(left_set), len(right_set))
        rows.append(
            {
                "domain": domain,
                "bin_minutes": bin_minutes,
                "left_split": left,
                "right_split": right,
                "left_unique_bins": len(left_set),
                "right_unique_bins": len(right_set),
                "overlap_bins": len(overlap),
                "overlap_fraction_of_smaller": (
                    len(overlap) / denominator if denominator else np.nan
                ),
            }
        )
    return rows


def frame_to_markdown(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without the optional tabulate package."""
    columns = [str(column) for column in frame.columns]

    def clean(value: object) -> str:
        if isinstance(value, float):
            text = f"{value:.6g}"
        else:
            text = str(value)
        return text.replace("|", "\\|").replace("\n", " ")

    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for values in frame.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(clean(value) for value in values) + " |")
    return "\n".join(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--recovery-zip", type=Path)
    parser.add_argument("--chunksize", type=int, default=250_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    base = root / "data" / "processed" / "point_event_v1"
    align = base / "nf3_alignment_v1"
    modeling = base / "modeling_v3"
    features_path = align / "candidate_point_events_v1_with_context_and_nf3_primary.csv"
    window_path = align / "event_nf3_window_features_v1.csv"
    edges_path = align / "event_flow_edges_primary_v1.csv"
    labels_path = align / "nf3_flow_labels_v1.csv"
    split_path = modeling / "model_split_assignments_v3.csv"
    summary_path = align / "point_event_nf3_alignment_v1_build_summary.json"

    required = [features_path, window_path, labels_path, split_path, summary_path]
    if args.recovery_zip is None:
        required.append(edges_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required frozen artifacts: {missing}")

    split_frame = pd.read_csv(
        split_path,
        usecols=["event_id", "chronological_split"],
        dtype={"event_id": "string", "chronological_split": "string"},
    )
    if split_frame["event_id"].duplicated().any():
        raise ValueError("Frozen split table contains duplicate event_id values.")
    if set(split_frame["chronological_split"].dropna().unique()) != set(SPLITS):
        raise ValueError("Unexpected chronological split labels.")
    split_lookup = split_frame.set_index("event_id")["chronological_split"]

    primary = pd.read_csv(
        features_path,
        usecols=[
            "event_id",
            "aircraft_event_time_utc",
            "aligned_event_time_utc",
            "replay_cycle_index",
            "window_start_utc",
            "window_end_utc",
            "nf3_flow_count",
        ],
        low_memory=False,
    ).merge(split_frame, on="event_id", how="left", validate="one_to_one")
    if primary["chronological_split"].isna().any():
        raise ValueError("One or more aligned events lack a frozen split assignment.")
    for column in ["aircraft_event_time_utc", "aligned_event_time_utc", "window_start_utc", "window_end_utc"]:
        primary[column] = pd.to_datetime(primary[column], utc=True, errors="raise")

    replay_rows: list[dict[str, object]] = []
    for split_name in SPLITS:
        group = primary.loc[primary["chronological_split"].eq(split_name)]
        cycle_counts = group["replay_cycle_index"].value_counts().sort_index()
        for cycle, count in cycle_counts.items():
            replay_rows.append(
                {
                    "split": split_name,
                    "replay_cycle_index": int(cycle),
                    "event_rows": int(count),
                    "events_with_primary_flows": int(
                        group.loc[group["replay_cycle_index"].eq(cycle), "nf3_flow_count"].gt(0).sum()
                    ),
                }
            )
    replay_frame = pd.DataFrame(replay_rows)
    replay_frame.to_csv(out / "replay_cycle_by_split_v1.csv", index=False)

    center_bin_rows: list[dict[str, object]] = []
    for minutes in BIN_MINUTES:
        sets = {
            split_name: set(
                bin_ids(
                    primary.loc[primary["chronological_split"].eq(split_name), "aligned_event_time_utc"],
                    minutes,
                ).tolist()
            )
            for split_name in SPLITS
        }
        center_bin_rows.extend(pairwise_set_rows(sets, "aligned_event_centers", minutes))
    center_bins = pd.DataFrame(center_bin_rows)
    center_bins.to_csv(out / "aligned_event_center_bin_recurrence_v1.csv", index=False)

    windows = pd.read_csv(
        window_path,
        usecols=["event_id", "window_name", "window_start_utc", "window_end_utc"],
        low_memory=False,
    ).merge(split_frame, on="event_id", how="left", validate="many_to_one")
    if windows["chronological_split"].isna().any():
        raise ValueError("One or more alignment windows lack a frozen split assignment.")
    windows["window_start_utc"] = pd.to_datetime(windows["window_start_utc"], utc=True, errors="raise")
    windows["window_end_utc"] = pd.to_datetime(windows["window_end_utc"], utc=True, errors="raise")

    window_overlap_rows: list[dict[str, object]] = []
    for window_name, window_group in windows.groupby("window_name", sort=True):
        merged_by_split: dict[str, np.ndarray] = {}
        for split_name in SPLITS:
            group = window_group.loc[window_group["chronological_split"].eq(split_name)]
            starts = group["window_start_utc"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
            ends = group["window_end_utc"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
            merged_by_split[split_name] = merge_intervals(starts, ends)
        for left, right in PAIRINGS:
            left_intervals = merged_by_split[left]
            right_intervals = merged_by_split[right]
            left_length = interval_length_ns(left_intervals)
            right_length = interval_length_ns(right_intervals)
            overlap = intersection_length_ns(left_intervals, right_intervals)
            smaller = min(left_length, right_length)
            window_overlap_rows.append(
                {
                    "window_name": window_name,
                    "left_split": left,
                    "right_split": right,
                    "left_union_hours": left_length / 3.6e12,
                    "right_union_hours": right_length / 3.6e12,
                    "overlap_hours": overlap / 3.6e12,
                    "overlap_fraction_of_smaller": overlap / smaller if smaller else np.nan,
                    "left_merged_intervals": len(left_intervals),
                    "right_merged_intervals": len(right_intervals),
                }
            )
    window_overlap = pd.DataFrame(window_overlap_rows)
    window_overlap.to_csv(out / "mapped_source_window_overlap_v1.csv", index=False)

    label_rows = sum(1 for _ in labels_path.open("rb")) - 1
    if label_rows <= 0:
        raise ValueError("The full NF3 flow label table is empty.")
    flow_split_mask = np.zeros(label_rows, dtype=np.uint8)
    edge_rows_by_split = {split_name: 0 for split_name in SPLITS}
    edge_events_by_split = {split_name: set() for split_name in SPLITS}
    edge_time_bin_sets: dict[int, dict[str, set[int]]] = {
        minutes: {split_name: set() for split_name in SPLITS} for minutes in BIN_MINUTES
    }
    edge_time_min = {split_name: None for split_name in SPLITS}
    edge_time_max = {split_name: None for split_name in SPLITS}
    total_edges = 0

    usecols = ["event_id", "source_row_number", "flow_start_utc"]
    archive = None
    edge_handle = None
    if args.recovery_zip is not None:
        archive = ZipFile(args.recovery_zip.resolve())
        edge_handle = archive.open(
            "Project/Jupyter Notebook/data/processed/point_event_v1/nf3_alignment_v1/"
            "event_flow_edges_primary_v1.csv"
        )
        edge_source = edge_handle
    else:
        edge_source = edges_path

    edge_reader = pd.read_csv(
        edge_source,
        usecols=usecols,
        chunksize=args.chunksize,
        low_memory=False,
    )
    for chunk_number, chunk in enumerate(edge_reader, start=1):
        chunk["event_id"] = chunk["event_id"].astype("string")
        chunk["chronological_split"] = chunk["event_id"].map(split_lookup)
        if chunk["chronological_split"].isna().any():
            unknown = chunk.loc[chunk["chronological_split"].isna(), "event_id"].head().tolist()
            raise ValueError(f"Full edge table contains event IDs absent from frozen splits: {unknown}")
        source_rows = pd.to_numeric(chunk["source_row_number"], errors="raise").astype(np.int64)
        if source_rows.min() < 0 or source_rows.max() >= label_rows:
            raise ValueError("Full edge table contains an out-of-range NF3 source row number.")
        chunk["source_row_number"] = source_rows
        chunk["flow_start_utc"] = pd.to_datetime(chunk["flow_start_utc"], utc=True, errors="raise")

        total_edges += len(chunk)
        for split_name in SPLITS:
            part = chunk.loc[chunk["chronological_split"].eq(split_name)]
            if part.empty:
                continue
            edge_rows_by_split[split_name] += len(part)
            edge_events_by_split[split_name].update(part["event_id"].astype(str).unique().tolist())
            unique_source_rows = np.unique(part["source_row_number"].to_numpy(dtype=np.int64))
            flow_split_mask[unique_source_rows] |= SPLIT_BITS[split_name]
            part_min = part["flow_start_utc"].min()
            part_max = part["flow_start_utc"].max()
            if edge_time_min[split_name] is None or part_min < edge_time_min[split_name]:
                edge_time_min[split_name] = part_min
            if edge_time_max[split_name] is None or part_max > edge_time_max[split_name]:
                edge_time_max[split_name] = part_max
            ns = part["flow_start_utc"].to_numpy(dtype="datetime64[ns]").astype(np.int64, copy=False)
            for minutes in BIN_MINUTES:
                width = int(pd.Timedelta(minutes=minutes).value)
                edge_time_bin_sets[minutes][split_name].update(np.unique(ns // width).tolist())
        if chunk_number % 4 == 0:
            print(f"Audited {total_edges:,} full edge rows...", flush=True)

    if edge_handle is not None:
        edge_handle.close()
    if archive is not None:
        archive.close()

    exact_rows: list[dict[str, object]] = []
    for left, right in PAIRINGS:
        left_bit = SPLIT_BITS[left]
        right_bit = SPLIT_BITS[right]
        left_count = int(np.count_nonzero(flow_split_mask & left_bit))
        right_count = int(np.count_nonzero(flow_split_mask & right_bit))
        both = (flow_split_mask & left_bit > 0) & (flow_split_mask & right_bit > 0)
        overlap_count = int(np.count_nonzero(both))
        examples = np.flatnonzero(both)[:10]
        exact_rows.append(
            {
                "left_split": left,
                "right_split": right,
                "left_unique_source_flows": left_count,
                "right_unique_source_flows": right_count,
                "reused_source_flows": overlap_count,
                "overlap_fraction_of_smaller": overlap_count / min(left_count, right_count),
                "example_source_row_numbers": "|".join(str(int(value)) for value in examples),
                "example_flow_ids": "|".join(f"NF3-{int(value):012d}" for value in examples),
            }
        )
    exact_recurrence = pd.DataFrame(exact_rows)
    exact_recurrence.to_csv(out / "exact_nf3_flow_recurrence_v1.csv", index=False)

    mask_counts = pd.Series(flow_split_mask).value_counts().sort_index()
    mask_frame = pd.DataFrame(
        {
            "split_mask": mask_counts.index.astype(int),
            "source_flow_rows": mask_counts.values.astype(int),
        }
    )
    mask_frame["split_membership"] = mask_frame["split_mask"].map(
        lambda value: "|".join(name for name, bit in SPLIT_BITS.items() if int(value) & bit) or "unused"
    )
    mask_frame.to_csv(out / "nf3_flow_split_membership_counts_v1.csv", index=False)

    edge_bin_rows: list[dict[str, object]] = []
    for minutes in BIN_MINUTES:
        edge_bin_rows.extend(
            pairwise_set_rows(edge_time_bin_sets[minutes], "matched_flow_start_times", minutes)
        )
    edge_bins = pd.DataFrame(edge_bin_rows)
    edge_bins.to_csv(out / "matched_flow_source_time_bin_recurrence_v1.csv", index=False)

    edge_summary = pd.DataFrame(
        [
            {
                "split": split_name,
                "edge_rows": edge_rows_by_split[split_name],
                "events_with_edges": len(edge_events_by_split[split_name]),
                "unique_source_flows": int(np.count_nonzero(flow_split_mask & SPLIT_BITS[split_name])),
                "minimum_flow_start_utc": edge_time_min[split_name],
                "maximum_flow_start_utc": edge_time_max[split_name],
            }
            for split_name in SPLITS
        ]
    )
    edge_summary.to_csv(out / "full_edge_audit_summary_v1.csv", index=False)

    direct_reused = int(
        np.count_nonzero(
            ((flow_split_mask & 1) > 0).astype(np.uint8)
            + ((flow_split_mask & 2) > 0).astype(np.uint8)
            + ((flow_split_mask & 4) > 0).astype(np.uint8)
            >= 2
        )
    )
    primary_window_overlap = window_overlap.loc[
        window_overlap["window_name"].eq("primary_120s"), "overlap_hours"
    ]
    gate_failed = direct_reused > 0 or bool(primary_window_overlap.gt(0).any())
    decision = {
        "audit_version": "nf3_source_recurrence_audit_v1",
        "frozen_alignment_version": "point_event_nf3_alignment_v1",
        "frozen_model_split_version": "point_event_tabular_baselines_v3",
        "full_primary_edge_rows_audited": total_edges,
        "nf3_source_flow_rows": label_rows,
        "source_flows_reused_across_two_or_more_splits": direct_reused,
        "primary_source_window_overlap_detected": bool(primary_window_overlap.gt(0).any()),
        "leakage_safe_independence_gate": "FAIL" if gate_failed else "PASS",
        "rerun_required": gate_failed,
        "decision_rule": (
            "FAIL when any original NF3 source flow is reused across frozen chronological splits "
            "or when the mapped primary source-time window unions overlap across splits."
        ),
    }
    (out / "nf3_recurrence_gate_v1.json").write_text(
        json.dumps(decision, indent=2, default=str), encoding="utf-8"
    )

    lines = [
        "# Full NF3 source-block recurrence audit",
        "",
        f"**Gate decision: {decision['leakage_safe_independence_gate']}**",
        "",
        f"The audit streamed all **{total_edges:,}** rows in the frozen primary edge table and "
        f"tracked all **{label_rows:,}** original NF3 source-flow identifiers.",
        "",
        "## Exact source-flow recurrence",
        "",
        frame_to_markdown(exact_recurrence),
        "",
        "## Mapped primary source-window overlap",
        "",
        frame_to_markdown(window_overlap.loc[window_overlap["window_name"].eq("primary_120s")]),
        "",
        "## Interpretation",
        "",
        (
            "The frozen cyclic replay does not provide source-independent train, validation, and test "
            "partitions. A leakage-safe remapping and rerun is therefore required."
            if gate_failed
            else "The frozen mapping passed the preregistered source-independence gate."
        ),
        "",
        "This audit concerns reuse of NF3 source observations and source-time neighborhoods. It does "
        "not change the separate non-causal interpretation boundary of the controlled aircraft–NF3 pairing.",
    ]
    (out / "NF3_SOURCE_RECURRENCE_AUDIT_v1.md").write_text("\n".join(lines), encoding="utf-8")

    checksums = []
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            checksums.append(f"{sha256_file(path)}  {path.name}")
    (out / "SHA256SUMS.txt").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()
