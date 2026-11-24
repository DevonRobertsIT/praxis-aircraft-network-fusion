#!/usr/bin/env python3
"""Independently validate the NF3 recurrence audit and leakage-safe rerun."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest(directory: Path) -> tuple[bool, int, list[str]]:
    manifest = directory / "SHA256SUMS.txt"
    failures: list[str] = []
    checked = 0
    for raw_line in manifest.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        expected, relative = raw_line.split(maxsplit=1)
        path = directory / relative.strip()
        checked += 1
        if not path.is_file():
            failures.append(f"missing:{relative.strip()}")
        elif sha256_file(path) != expected.lower():
            failures.append(f"checksum:{relative.strip()}")
    return not failures, checked, failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    audit_dir = args.audit_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    v2 = project_root / "data/processed/point_event_v2"
    alignment = v2 / "nf3_alignment_v2"
    modeling = v2 / "modeling_v4"
    transformer = v2 / "transformer_v2"
    rgcn = v2 / "rgcn_leakage_safe_v2"
    results = v2 / "results_v2"
    controls = v2 / "control_evaluations_v2_final"
    chapter4 = v2 / "chapter4_exports_v2"
    xai = v2 / "xai_leakage_safe_v2"

    checks: list[dict[str, Any]] = []

    def check(name: str, expected: Any, actual: Any, passed: bool, note: str = "") -> None:
        checks.append(
            {
                "check": name,
                "expected": json.dumps(expected, default=str) if not isinstance(expected, str) else expected,
                "actual": json.dumps(actual, default=str) if not isinstance(actual, str) else actual,
                "passed": bool(passed),
                "note": note,
            }
        )

    v1_gate = json.loads((audit_dir / "nf3_recurrence_gate_v1.json").read_text(encoding="utf-8"))
    check(
        "v1_full_edge_rows_audited",
        2_974_178,
        v1_gate["full_primary_edge_rows_audited"],
        int(v1_gate["full_primary_edge_rows_audited"]) == 2_974_178,
    )
    check(
        "v1_all_source_flows_tracked",
        2_365_424,
        v1_gate["nf3_source_flow_rows"],
        int(v1_gate["nf3_source_flow_rows"]) == 2_365_424,
    )
    check(
        "v1_recurrence_gate_detected_failure",
        "FAIL",
        v1_gate["leakage_safe_independence_gate"],
        v1_gate["leakage_safe_independence_gate"] == "FAIL",
    )
    check(
        "v1_reused_source_flows_detected",
        ">0",
        v1_gate["source_flows_reused_across_two_or_more_splits"],
        int(v1_gate["source_flows_reused_across_two_or_more_splits"]) > 0,
    )

    exact = pd.read_csv(alignment / "leakage_safe_exact_source_flow_audit_v2.csv")
    windows = pd.read_csv(alignment / "leakage_safe_source_window_audit_v2.csv")
    block_manifest = pd.read_csv(alignment / "nf3_source_block_manifest_v2.csv")
    check(
        "v2_exact_source_flow_recurrence",
        0,
        int(exact["reused_source_flows"].sum()),
        int(exact["reused_source_flows"].sum()) == 0,
    )
    overlap_values = windows["source_window_overlap_detected"].astype(str).str.lower().eq("true")
    check(
        "v2_source_window_overlap",
        False,
        bool(overlap_values.any()),
        not bool(overlap_values.any()),
    )

    block_manifest["block_start_utc"] = pd.to_datetime(block_manifest["block_start_utc"], utc=True)
    block_manifest["block_end_utc"] = pd.to_datetime(block_manifest["block_end_utc"], utc=True)
    minimum_gap = np.inf
    for _, group in block_manifest.groupby("source_session_index"):
        ordered = group.sort_values("block_start_utc")
        gaps = (
            ordered["block_start_utc"].iloc[1:].reset_index(drop=True)
            - ordered["block_end_utc"].iloc[:-1].reset_index(drop=True)
        ).dt.total_seconds()
        if len(gaps):
            minimum_gap = min(minimum_gap, float(gaps.min()))
    check(
        "v2_interpartition_source_embargo_seconds",
        ">=1200",
        minimum_gap,
        bool(minimum_gap >= 1_200.0),
    )

    features = pd.read_csv(
        alignment / "candidate_point_events_v2_with_context_and_nf3_primary.csv",
        usecols=["event_id", "source_partition", "nf3_flow_count"],
        low_memory=False,
    )
    splits = pd.read_csv(
        modeling / "model_split_assignments_v4.csv",
        usecols=["event_id", "chronological_split", "model_target"],
        low_memory=False,
    )
    joined = splits.merge(features, on="event_id", how="inner", validate="one_to_one")
    check("v2_event_rows", 12_185, len(joined), len(joined) == 12_185)
    check(
        "v2_event_ids_unique",
        True,
        bool(joined["event_id"].is_unique),
        bool(joined["event_id"].is_unique),
    )
    partition_agreement = joined["source_partition"].eq(joined["chronological_split"]).all()
    check("v2_source_partition_matches_model_split", True, bool(partition_agreement), bool(partition_agreement))
    test = joined.loc[joined["chronological_split"].eq("test")]
    check("v2_test_rows", 1_828, len(test), len(test) == 1_828)
    check(
        "v2_test_has_both_classes",
        [0, 1],
        sorted(test["model_target"].unique().tolist()),
        set(test["model_target"].unique()) == {0, 1},
    )

    feature_manifest = json.loads((modeling / "feature_manifest_v4.json").read_text(encoding="utf-8"))
    predictors: set[str] = set()
    for specification in feature_manifest["feature_sets"].values():
        predictors.update(specification.get("numeric", []))
        predictors.update(specification.get("categorical", []))
    forbidden = set(feature_manifest["excluded_target_derived_columns"])
    target_overlap = sorted(predictors & forbidden)
    check("v2_target_derived_predictors", [], target_overlap, not target_overlap)

    model_results = pd.read_csv(modeling / "baseline_model_results_v4.csv")
    check("v2_tabular_model_runs", 14, len(model_results), len(model_results) == 14)
    model_files = list((modeling / "models").glob("*.joblib"))
    check("v2_tabular_model_artifacts", 14, len(model_files), len(model_files) == 14)
    check(
        "v2_transformer_checkpoint",
        True,
        (transformer / "models/fused_temporal_transformer.pt").is_file(),
        (transformer / "models/fused_temporal_transformer.pt").is_file(),
    )
    check(
        "v2_rgcn_checkpoints",
        2,
        len(list((rgcn / "models").glob("*.pt"))),
        len(list((rgcn / "models").glob("*.pt"))) == 2,
    )

    any_flow = pd.read_csv(controls / "any_flow_baseline_evaluation.csv")
    v1_any = any_flow.loc[any_flow["analysis_version"].eq("frozen_v1")].iloc[0]
    v2_any = any_flow.loc[any_flow["analysis_version"].eq("leakage_safe_v2")].iloc[0]
    check("v1_any_flow_auprc", 0.55, float(v1_any["auprc"]), np.isclose(v1_any["auprc"], 0.55))
    check(
        "v2_any_flow_equals_prevalence",
        float(v2_any["prevalence"]),
        float(v2_any["auprc"]),
        np.isclose(v2_any["auprc"], v2_any["prevalence"]),
    )
    balance = pd.read_csv(controls / "matched_flow_covered_balance_diagnostics.csv")
    max_smd = float(balance["after_absolute_smd"].max())
    check("matched_covariate_balance_max_absolute_smd", "<=0.10", max_smd, max_smd <= 0.10)
    matched_metrics = pd.read_csv(controls / "matched_flow_covered_model_evaluation.csv")
    matched_v2 = matched_metrics.loc[
        matched_metrics["analysis_version"].eq("leakage_safe_v2")
        & matched_metrics["model"].eq("cyber_random_forest")
    ].iloc[0]
    check("v2_matched_pairs", ">=500", int(matched_v2["matched_pairs"]), int(matched_v2["matched_pairs"]) >= 500)

    hypothesis = pd.read_csv(results / "consolidated_hypothesis_evaluation_v2.csv")
    support = dict(zip(hypothesis["hypothesis"], hypothesis["supported"].astype(bool)))
    check("v2_hypothesis_decisions", {"H1": True, "H2": False, "H3": False}, support, support == {"H1": True, "H2": False, "H3": False})

    provenance = pd.read_csv(xai / "xai_provenance_audit_v2.csv")
    check(
        "v2_xai_provenance_checks",
        "all pass",
        f"{int(provenance['passed'].sum())}/{len(provenance)}",
        bool(provenance["passed"].astype(bool).all()),
    )
    check(
        "v2_chapter4_table_files",
        32,
        len(list((chapter4 / "tables").glob("*"))),
        len(list((chapter4 / "tables").glob("*"))) == 32,
    )
    check(
        "v2_chapter4_figure_files",
        9,
        len(list((chapter4 / "figures").glob("*.png"))),
        len(list((chapter4 / "figures").glob("*.png"))) == 9,
    )

    manifest_directories = [
        audit_dir,
        alignment,
        modeling,
        transformer,
        rgcn,
        results,
        controls,
        chapter4,
        xai,
    ]
    for directory in manifest_directories:
        passed, checked, failures = validate_manifest(directory)
        check(
            f"sha256_manifest::{directory.name}",
            "all entries valid",
            {"entries": checked, "failures": failures},
            passed,
        )

    frame = pd.DataFrame(checks)
    frame.to_csv(output_dir / "independent_validation_checks_v2.csv", index=False)
    payload = {
        "validation_version": "nf3_leakage_safe_reanalysis_validation_v2",
        "checks": len(frame),
        "passed": int(frame["passed"].sum()),
        "failed": int((~frame["passed"]).sum()),
        "overall_decision": "PASS" if frame["passed"].all() else "FAIL",
        "failed_checks": frame.loc[~frame["passed"], "check"].tolist(),
    }
    (output_dir / "independent_validation_summary_v2.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))
    if not frame["passed"].all():
        print(frame.loc[~frame["passed"]].to_string(index=False))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
