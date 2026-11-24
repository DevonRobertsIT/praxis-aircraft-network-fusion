#!/usr/bin/env python3
"""Rebuild and verify the v2 primary feature bank from completed v2 outputs."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--window-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    context = pd.read_csv(args.context, low_memory=False)
    windows = pd.read_csv(args.window_features, low_memory=False)
    primary = windows.loc[windows["window_name"].eq("primary_120s")].copy()
    primary = primary.drop(columns=["physical_event_time_utc"], errors="ignore")
    fused = context.merge(primary, on="event_id", how="left", validate="one_to_one")
    if len(fused) != 12_185 or fused["source_partition"].isna().any():
        raise AssertionError("Rebuilt v2 feature bank failed row or merge completeness checks.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fused.to_csv(args.output, index=False)
    verify = pd.read_csv(args.output, usecols=["event_id", "source_partition", "nf3_flow_count"])
    if len(verify) != len(fused) or verify["event_id"].nunique() != len(fused):
        raise AssertionError("Serialized v2 feature bank failed round-trip row checks.")
    print(f"rows={len(verify):,}")
    print(f"sha256={sha256_file(args.output)}")


if __name__ == "__main__":
    main()
