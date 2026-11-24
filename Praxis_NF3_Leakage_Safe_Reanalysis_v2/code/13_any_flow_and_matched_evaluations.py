#!/usr/bin/env python3
"""Finalize any-flow and coarsened-exact matched flow-covered evaluations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


RANDOM_STATE = 20260903
BOOTSTRAP_ITERATIONS = 5_000
CEM_SENSITIVITY = (2, 3, 4, 5, 6)
BALANCE_THRESHOLD = 0.10
MATCH_COLUMNS = ("nf3_flow_count", "nf3_flow_duration_sum_s")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel().tolist()
    both = np.unique(y).size == 2
    return {
        "rows": len(y),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()),
        "auprc": float(average_precision_score(y, p)) if both else np.nan,
        "roc_auc": float(roc_auc_score(y, p)) if both else np.nan,
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if both else np.nan,
        "brier_score": float(brier_score_loss(y, p)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def standardized_mean_difference(values: np.ndarray, target: np.ndarray) -> float:
    positive = values[target == 1]
    negative = values[target == 0]
    pooled = np.sqrt((np.var(positive, ddof=1) + np.var(negative, ddof=1)) / 2.0)
    return float((np.mean(positive) - np.mean(negative)) / pooled) if pooled > 0 else 0.0


def coarsened_exact_matches(
    flow_covered: pd.DataFrame,
    quantiles: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Match classes 1:1 inside joint quantile strata, then minimize distance.

    Coarsened exact matching makes the target classes contribute the same number
    of observations within each joint flow-count/duration stratum.  Optimal
    assignment is used only to select the closest individual pairs inside each
    eligible stratum; it does not determine stratum membership.
    """
    transformed = np.log1p(flow_covered[list(MATCH_COLUMNS)].astype(float).to_numpy())
    means = transformed.mean(axis=0)
    scales = transformed.std(axis=0, ddof=0)
    scales[scales == 0] = 1.0
    standardized = (transformed - means) / scales
    target = flow_covered["model_target"].to_numpy(dtype=int)
    if np.unique(target).size != 2:
        raise ValueError("Flow-covered evaluation requires both target classes.")
    strata = pd.DataFrame(index=flow_covered.index)
    for column_index, column in enumerate(MATCH_COLUMNS):
        strata[f"stratum_{column_index}"] = pd.qcut(
            transformed[:, column_index],
            q=quantiles,
            labels=False,
            duplicates="drop",
        )
    pair_rows: list[dict[str, Any]] = []
    pair_id = 0
    for stratum_key, stratum in strata.groupby(list(strata.columns), observed=True, sort=True):
        positions = stratum.index.to_numpy(dtype=int)
        positive_positions = positions[target[positions] == 1]
        negative_positions = positions[target[positions] == 0]
        if not len(positive_positions) or not len(negative_positions):
            continue
        distances = np.sqrt(
            np.sum(
                (
                    standardized[positive_positions, None, :]
                    - standardized[negative_positions][None, :, :]
                )
                ** 2,
                axis=2,
            )
        )
        positive_assignment, negative_assignment = linear_sum_assignment(distances)
        for positive_index, negative_index in zip(positive_assignment, negative_assignment):
            pair_rows.append(
                {
                    "pair_id": pair_id,
                    "positive_position": int(positive_positions[positive_index]),
                    "negative_position": int(negative_positions[negative_index]),
                    "stratum_flow_count": int(stratum_key[0]),
                    "stratum_flow_duration": int(stratum_key[1]),
                    "standardized_euclidean_distance": float(distances[positive_index, negative_index]),
                }
            )
            pair_id += 1
    pairs = pd.DataFrame(pair_rows)
    if pairs.empty:
        raise ValueError("Coarsened exact matching produced no eligible pairs.")
    return pairs, standardized


def bootstrap_pair_difference(
    matched: pd.DataFrame,
    left_probability: str,
    right_probability: str,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    grouped = [group for _, group in matched.groupby("pair_id", sort=True)]
    if not grouped:
        raise ValueError("No matched pairs were available for paired bootstrap.")
    rng = np.random.default_rng(seed)
    differences: list[float] = []
    left_scores: list[float] = []
    right_scores: list[float] = []
    for _ in range(iterations):
        sampled = rng.integers(0, len(grouped), size=len(grouped))
        draw = pd.concat([grouped[int(index)] for index in sampled], ignore_index=True)
        y = draw["model_target"].to_numpy(dtype=int)
        left = float(average_precision_score(y, draw[left_probability].to_numpy(dtype=float)))
        right = float(average_precision_score(y, draw[right_probability].to_numpy(dtype=float)))
        left_scores.append(left)
        right_scores.append(right)
        differences.append(left - right)
    diff = np.asarray(differences)
    return {
        "bootstrap_iterations": iterations,
        "point_left_auprc": float(average_precision_score(matched["model_target"], matched[left_probability])),
        "point_right_auprc": float(average_precision_score(matched["model_target"], matched[right_probability])),
        "point_difference": float(
            average_precision_score(matched["model_target"], matched[left_probability])
            - average_precision_score(matched["model_target"], matched[right_probability])
        ),
        "difference_ci_lower_95": float(np.quantile(diff, 0.025)),
        "difference_ci_upper_95": float(np.quantile(diff, 0.975)),
        "probability_difference_gt_zero": float(np.mean(diff > 1e-12)),
    }


def bootstrap_event_difference(
    data: pd.DataFrame,
    left_probability: str,
    right_probability: str,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    y = data["model_target"].to_numpy(dtype=int)
    left = data[left_probability].to_numpy(dtype=float)
    right = data[right_probability].to_numpy(dtype=float)
    differences: list[float] = []
    for _ in range(iterations):
        selected = rng.integers(0, len(data), size=len(data))
        if np.unique(y[selected]).size < 2:
            continue
        differences.append(
            float(average_precision_score(y[selected], left[selected]) - average_precision_score(y[selected], right[selected]))
        )
    diff = np.asarray(differences)
    return {
        "bootstrap_iterations": len(diff),
        "point_left_auprc": float(average_precision_score(y, left)),
        "point_right_auprc": float(average_precision_score(y, right)),
        "point_difference": float(average_precision_score(y, left) - average_precision_score(y, right)),
        "difference_ci_lower_95": float(np.quantile(diff, 0.025)),
        "difference_ci_upper_95": float(np.quantile(diff, 0.975)),
        "probability_difference_gt_zero": float(np.mean(diff > 1e-12)),
    }


def load_analysis(
    version: str,
    feature_path: Path,
    split_path: Path,
    prediction_path: Path,
) -> pd.DataFrame:
    features = pd.read_csv(feature_path, usecols=["event_id", *MATCH_COLUMNS], low_memory=False)
    splits = pd.read_csv(
        split_path,
        usecols=["event_id", "model_target", "chronological_split"],
        low_memory=False,
    )
    predictions = pd.read_csv(prediction_path, low_memory=False)
    selected = predictions.loc[
        predictions["split_strategy"].eq("chronological_primary")
        & predictions["model"].eq("random_forest")
        & predictions["feature_set"].isin(["cyber", "fused"]),
        ["event_id", "feature_set", "predicted_probability", "selected_threshold"],
    ].copy()
    wide_probability = selected.pivot(index="event_id", columns="feature_set", values="predicted_probability")
    wide_probability = wide_probability.rename(columns={"cyber": "cyber_rf_probability", "fused": "fused_rf_probability"})
    wide_threshold = selected.pivot(index="event_id", columns="feature_set", values="selected_threshold")
    wide_threshold = wide_threshold.rename(columns={"cyber": "cyber_rf_threshold", "fused": "fused_rf_threshold"})
    data = (
        splits.loc[splits["chronological_split"].eq("test")]
        .merge(features, on="event_id", how="left", validate="one_to_one")
        .merge(wide_probability.reset_index(), on="event_id", how="left", validate="one_to_one")
        .merge(wide_threshold.reset_index(), on="event_id", how="left", validate="one_to_one")
    )
    if data.isna().any().any():
        missing = data.columns[data.isna().any()].tolist()
        raise ValueError(f"{version}: missing values in completed evaluation bank: {missing}")
    data["analysis_version"] = version
    data["any_flow_probability"] = data["nf3_flow_count"].gt(0).astype(float)
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    specifications = {
        "frozen_v1": {
            "features": root / "data/processed/point_event_v1/nf3_alignment_v1/candidate_point_events_v1_with_context_and_nf3_primary.csv",
            "splits": root / "data/processed/point_event_v1/modeling_v3/model_split_assignments_v3.csv",
            "predictions": root / "data/processed/point_event_v1/modeling_v3/baseline_test_predictions_v3.csv",
        },
        "leakage_safe_v2": {
            "features": root / "data/processed/point_event_v2/nf3_alignment_v2/candidate_point_events_v2_with_context_and_nf3_primary.csv",
            "splits": root / "data/processed/point_event_v2/modeling_v4/model_split_assignments_v4.csv",
            "predictions": root / "data/processed/point_event_v2/modeling_v4/baseline_test_predictions_v4.csv",
        },
    }

    any_flow_rows: list[dict[str, Any]] = []
    matched_metric_rows: list[dict[str, Any]] = []
    match_quality_rows: list[dict[str, Any]] = []
    coarsening_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    match_exports: list[pd.DataFrame] = []

    for version, paths in specifications.items():
        data = load_analysis(version, paths["features"], paths["splits"], paths["predictions"])
        any_flow_rows.append(
            {
                "analysis_version": version,
                "evaluation": "any_mapped_flow_present_on_full_test",
                "threshold": 0.5,
                "flow_covered_rows": int(data["nf3_flow_count"].gt(0).sum()),
                **metrics(data["model_target"].to_numpy(), data["any_flow_probability"].to_numpy()),
            }
        )

        full_bootstrap = bootstrap_event_difference(
            data,
            "fused_rf_probability",
            "cyber_rf_probability",
            BOOTSTRAP_ITERATIONS,
            RANDOM_STATE + (0 if version == "frozen_v1" else 1),
        )
        bootstrap_rows.append(
            {
                "analysis_version": version,
                "evaluation": "full_test_fused_rf_minus_cyber_rf",
                "matched_pairs": np.nan,
                **full_bootstrap,
            }
        )

        covered = data.loc[data["nf3_flow_count"].gt(0)].reset_index(drop=True)
        eligible_designs: list[tuple[int, pd.DataFrame, np.ndarray, float, float]] = []
        for quantiles in CEM_SENSITIVITY:
            sensitivity_pairs, sensitivity_standardized = coarsened_exact_matches(covered, quantiles)
            sensitivity_positions = np.r_[
                sensitivity_pairs["positive_position"].to_numpy(dtype=int),
                sensitivity_pairs["negative_position"].to_numpy(dtype=int),
            ]
            sensitivity_target = covered.iloc[sensitivity_positions]["model_target"].to_numpy(dtype=int)
            sensitivity_values = np.log1p(
                covered.iloc[sensitivity_positions][list(MATCH_COLUMNS)].to_numpy(dtype=float)
            )
            flow_count_smd = abs(
                standardized_mean_difference(sensitivity_values[:, 0], sensitivity_target)
            )
            flow_duration_smd = abs(
                standardized_mean_difference(sensitivity_values[:, 1], sensitivity_target)
            )
            coarsening_rows.append(
                {
                    "analysis_version": version,
                    "quantile_bins_per_variable": quantiles,
                    "matched_pairs": len(sensitivity_pairs),
                    "retained_fraction_of_maximum_class_pairs": len(sensitivity_pairs)
                    / int(covered["model_target"].value_counts().min()),
                    "absolute_smd_log_flow_count": flow_count_smd,
                    "absolute_smd_log_flow_duration": flow_duration_smd,
                    "meets_balance_threshold": bool(
                        flow_count_smd <= BALANCE_THRESHOLD and flow_duration_smd <= BALANCE_THRESHOLD
                    ),
                    "distance_median": float(sensitivity_pairs["standardized_euclidean_distance"].median()),
                    "distance_maximum": float(sensitivity_pairs["standardized_euclidean_distance"].max()),
                }
            )
            if flow_count_smd <= BALANCE_THRESHOLD and flow_duration_smd <= BALANCE_THRESHOLD:
                eligible_designs.append(
                    (
                        quantiles,
                        sensitivity_pairs,
                        sensitivity_standardized,
                        flow_count_smd,
                        flow_duration_smd,
                    )
                )

        if not eligible_designs:
            raise ValueError(f"{version}: no CEM design met the {BALANCE_THRESHOLD:.2f} balance threshold.")
        selected_quantiles, retained, standardized, _, _ = max(
            eligible_designs,
            key=lambda design: (len(design[1]), -design[0]),
        )
        if len(retained) < 10:
            raise ValueError(f"{version}: fewer than 10 matched pairs survived the primary CEM design.")
        matched_parts: list[pd.DataFrame] = []
        for row in retained.itertuples(index=False):
            positive = covered.iloc[[int(row.positive_position)]].copy()
            negative = covered.iloc[[int(row.negative_position)]].copy()
            pair = pd.concat([positive, negative], ignore_index=True)
            pair["pair_id"] = int(row.pair_id)
            pair["match_distance"] = float(row.standardized_euclidean_distance)
            pair["stratum_flow_count"] = int(row.stratum_flow_count)
            pair["stratum_flow_duration"] = int(row.stratum_flow_duration)
            matched_parts.append(pair)
        matched = pd.concat(matched_parts, ignore_index=True)
        match_exports.append(matched)

        before_target = covered["model_target"].to_numpy(dtype=int)
        after_target = matched["model_target"].to_numpy(dtype=int)
        for column_index, column in enumerate(MATCH_COLUMNS):
            before_values = np.log1p(covered[column].to_numpy(dtype=float))
            after_values = np.log1p(matched[column].to_numpy(dtype=float))
            match_quality_rows.append(
                {
                    "analysis_version": version,
                    "matching_variable": f"log1p({column})",
                    "before_absolute_smd": abs(standardized_mean_difference(before_values, before_target)),
                    "after_absolute_smd": abs(standardized_mean_difference(after_values, after_target)),
                    "before_rows": len(covered),
                    "after_rows": len(matched),
                    "matched_pairs": len(retained),
                    "match_design": "coarsened_exact_matching",
                    "quantile_bins_per_variable": selected_quantiles,
                }
            )

        model_specs = [
            ("any_flow", "any_flow_probability", 0.5),
            ("cyber_random_forest", "cyber_rf_probability", float(matched["cyber_rf_threshold"].iloc[0])),
            ("fused_random_forest", "fused_rf_probability", float(matched["fused_rf_threshold"].iloc[0])),
        ]
        for model_name, probability_column, threshold in model_specs:
            matched_metric_rows.append(
                {
                    "analysis_version": version,
                    "evaluation": "primary_cem_matched_flow_covered_test",
                    "model": model_name,
                    "matched_pairs": len(retained),
                    "match_design": "coarsened_exact_matching",
                    "quantile_bins_per_variable": selected_quantiles,
                    "threshold": threshold,
                    **metrics(
                        matched["model_target"].to_numpy(),
                        matched[probability_column].to_numpy(),
                        threshold=threshold,
                    ),
                }
            )
        matched_bootstrap = bootstrap_pair_difference(
            matched,
            "fused_rf_probability",
            "cyber_rf_probability",
            BOOTSTRAP_ITERATIONS,
            RANDOM_STATE + 10 + (0 if version == "frozen_v1" else 1),
        )
        bootstrap_rows.append(
            {
                "analysis_version": version,
                "evaluation": "matched_flow_covered_fused_rf_minus_cyber_rf",
                "matched_pairs": len(retained),
                **matched_bootstrap,
            }
        )

    any_flow_frame = pd.DataFrame(any_flow_rows)
    matched_metrics_frame = pd.DataFrame(matched_metric_rows)
    match_quality_frame = pd.DataFrame(match_quality_rows)
    coarsening_frame = pd.DataFrame(coarsening_rows)
    bootstrap_frame = pd.DataFrame(bootstrap_rows)
    matched_events_frame = pd.concat(match_exports, ignore_index=True)

    any_flow_frame.to_csv(out / "any_flow_baseline_evaluation.csv", index=False)
    matched_metrics_frame.to_csv(out / "matched_flow_covered_model_evaluation.csv", index=False)
    match_quality_frame.to_csv(out / "matched_flow_covered_balance_diagnostics.csv", index=False)
    coarsening_frame.to_csv(out / "matched_flow_covered_coarsening_sensitivity.csv", index=False)
    bootstrap_frame.to_csv(out / "random_forest_paired_bootstrap_differences.csv", index=False)
    matched_events_frame.to_csv(out / "matched_flow_covered_event_pairs.csv", index=False)

    v1_any = any_flow_frame.loc[any_flow_frame["analysis_version"].eq("frozen_v1")].iloc[0]
    v2_any = any_flow_frame.loc[any_flow_frame["analysis_version"].eq("leakage_safe_v2")].iloc[0]
    v1_boot = bootstrap_frame.loc[
        bootstrap_frame["analysis_version"].eq("frozen_v1")
        & bootstrap_frame["evaluation"].str.startswith("matched")
    ].iloc[0]
    v2_boot = bootstrap_frame.loc[
        bootstrap_frame["analysis_version"].eq("leakage_safe_v2")
        & bootstrap_frame["evaluation"].str.startswith("matched")
    ].iloc[0]
    report = f"""# Any-flow and matched flow-covered evaluations

## Any-flow baseline

The frozen v1 test set contains {int(v1_any.flow_covered_rows):,} flow-covered events out of {int(v1_any.rows):,}. Predicting attack whenever any mapped flow is present yields AUPRC {v1_any.auprc:.4f}, precision {v1_any.precision:.4f}, recall {v1_any.recall:.4f}, and F1 {v1_any.f1:.4f}. This confirms that flow presence alone is a strong proxy in v1.

Under leakage-safe v2, {int(v2_any.flow_covered_rows):,} of {int(v2_any.rows):,} held-out events are flow-covered. The any-flow baseline yields AUPRC {v2_any.auprc:.4f} against prevalence {v2_any.prevalence:.4f}; it is therefore not a useful discriminator after active source-block remapping.

## Matched flow-covered evaluation

Events were matched 1:1 without replacement using coarsened exact matching on joint quantile strata of log flow count and log total flow duration. Within every eligible stratum, optimal assignment selected the closest individual pairs. From two through six bins per variable, the retained-sample-maximizing design with absolute standardized mean differences no greater than {BALANCE_THRESHOLD:.2f} on both covariates was selected separately for each analysis version. Alternative coarsening levels and full balance diagnostics are exported separately.

For frozen v1, {int(v1_boot.matched_pairs):,} matched pairs remained. Fused-minus-cyber random-forest AUPRC was {v1_boot.point_difference:+.6f} with a 95% paired-bootstrap interval of [{v1_boot.difference_ci_lower_95:+.6f}, {v1_boot.difference_ci_upper_95:+.6f}].

For leakage-safe v2, {int(v2_boot.matched_pairs):,} matched pairs remained. Fused-minus-cyber random-forest AUPRC was {v2_boot.point_difference:+.6f} with a 95% paired-bootstrap interval of [{v2_boot.difference_ci_lower_95:+.6f}, {v2_boot.difference_ci_upper_95:+.6f}].

## Interpretation boundary

These results evaluate discrimination within a controlled cross-domain replay. The matched design reduces imbalance in flow volume and duration; it does not make the independently collected aircraft and NF3 records historically co-occurring or causal.
"""
    (out / "ANY_FLOW_AND_MATCHED_EVALUATION_REPORT.md").write_text(report, encoding="utf-8")

    summary = {
        "analysis_version": "any_flow_and_matched_controls_v2",
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "match_design": "coarsened_exact_matching",
        "candidate_quantile_bins_per_variable": list(CEM_SENSITIVITY),
        "primary_design_selection_rule": "maximum matched pairs subject to both absolute SMDs <= 0.10",
        "balance_threshold": BALANCE_THRESHOLD,
        "matching_variables": list(MATCH_COLUMNS),
        "versions_evaluated": list(specifications),
        "outputs": [path.name for path in sorted(out.iterdir()) if path.is_file()],
    }
    (out / "control_evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    checksums = []
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            checksums.append(f"{sha256_file(path)}  {path.name}")
    (out / "SHA256SUMS.txt").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    print(any_flow_frame.to_string(index=False))
    print(bootstrap_frame.to_string(index=False))


if __name__ == "__main__":
    main()
