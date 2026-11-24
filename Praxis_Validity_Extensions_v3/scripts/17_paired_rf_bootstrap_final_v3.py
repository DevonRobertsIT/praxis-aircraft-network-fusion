from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score


RANDOM_STATE = 42
BOOTSTRAP_REPEATS = int(os.getenv("PRAXIS_BOOTSTRAP_REPEATS", "20000"))

WORKSPACE = Path(__file__).resolve().parents[1]
PROJECT_ROOT = WORKSPACE / "audit_work" / "Project" / "Jupyter Notebook"
BASE_DIR = PROJECT_ROOT / "data" / "processed" / "point_event_v2"
MODELING_DIR = BASE_DIR / "modeling_v4"
CONTROL_DIR = BASE_DIR / "control_evaluations_v2_final"
OUTPUT_DIR = BASE_DIR / "validity_extensions_v3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_IN = MODELING_DIR / "baseline_test_predictions_v4.csv"
MATCHED_PAIRS_IN = CONTROL_DIR / "matched_flow_covered_event_pairs.csv"
THRESHOLDS_IN = MODELING_DIR / "baseline_thresholds_v4.json"

SUMMARY_OUT = OUTPUT_DIR / "paired_rf_bootstrap_final_v3.csv"
SAMPLES_OUT = OUTPUT_DIR / "paired_rf_bootstrap_samples_v3.csv"
MANIFEST_OUT = OUTPUT_DIR / "paired_rf_bootstrap_manifest_v3.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric_values(
    y: np.ndarray,
    cyber_probability: np.ndarray,
    fused_probability: np.ndarray,
    cyber_threshold: float,
    fused_threshold: float,
) -> dict[str, float]:
    cyber_f1 = f1_score(y, cyber_probability >= cyber_threshold, zero_division=0)
    fused_f1 = f1_score(y, fused_probability >= fused_threshold, zero_division=0)
    return {
        "cyber_auprc": float(average_precision_score(y, cyber_probability)),
        "fused_auprc": float(average_precision_score(y, fused_probability)),
        "auprc_difference_fused_minus_cyber": float(
            average_precision_score(y, fused_probability)
            - average_precision_score(y, cyber_probability)
        ),
        "cyber_brier": float(brier_score_loss(y, cyber_probability)),
        "fused_brier": float(brier_score_loss(y, fused_probability)),
        "brier_difference_fused_minus_cyber": float(
            brier_score_loss(y, fused_probability)
            - brier_score_loss(y, cyber_probability)
        ),
        "cyber_f1": float(cyber_f1),
        "fused_f1": float(fused_f1),
        "f1_difference_fused_minus_cyber": float(fused_f1 - cyber_f1),
    }


thresholds = json.loads(THRESHOLDS_IN.read_text(encoding="utf-8"))
cyber_threshold = float(
    thresholds["chronological_primary__cyber__random_forest"]["selected_threshold"]
)
fused_threshold = float(
    thresholds["chronological_primary__fused__random_forest"]["selected_threshold"]
)

predictions = pd.read_csv(PREDICTIONS_IN, low_memory=False)
rf = predictions.loc[
    predictions["split_strategy"].astype(str).eq("chronological_primary")
    & predictions["model"].astype(str).eq("random_forest")
    & predictions["feature_set"].astype(str).isin(["cyber", "fused"]),
    ["event_id", "model_target", "feature_set", "predicted_probability"],
].copy()
full = rf.pivot(
    index=["event_id", "model_target"],
    columns="feature_set",
    values="predicted_probability",
).reset_index()
full.columns.name = None
if len(full) != 1828 or set(full["model_target"].unique()) != {0, 1}:
    raise AssertionError(f"Unexpected full-test RF comparison bank: rows={len(full)}")

matched_all = pd.read_csv(MATCHED_PAIRS_IN, low_memory=False)
matched = matched_all.loc[
    matched_all["analysis_version"].astype(str).eq("leakage_safe_v2")
].copy()
pair_counts = matched.groupby("pair_id").agg(rows=("event_id", "size"), positives=("model_target", "sum"))
if not ((pair_counts["rows"] == 2) & (pair_counts["positives"] == 1)).all():
    raise AssertionError("Matched evaluation does not contain one positive and one negative per pair.")

datasets = {
    "full_test_stratified_event_bootstrap": full.rename(
        columns={"cyber": "cyber_rf_probability", "fused": "fused_rf_probability"}
    ),
    "matched_flow_covered_pair_bootstrap": matched,
}

rng = np.random.default_rng(RANDOM_STATE + 1700)
sample_rows: list[dict[str, Any]] = []
point_estimates: dict[str, dict[str, float]] = {}

for dataset_name, frame in datasets.items():
    y = frame["model_target"].to_numpy(dtype=np.int8)
    cyber = frame["cyber_rf_probability"].to_numpy(dtype=float)
    fused = frame["fused_rf_probability"].to_numpy(dtype=float)
    point_estimates[dataset_name] = metric_values(
        y, cyber, fused, cyber_threshold, fused_threshold
    )
    print(f"Running {BOOTSTRAP_REPEATS:,} paired replicates for {dataset_name} ...")

    if dataset_name.startswith("full_test"):
        positive_positions = np.flatnonzero(y == 1)
        negative_positions = np.flatnonzero(y == 0)
        for repeat in range(BOOTSTRAP_REPEATS):
            positions = np.concatenate(
                [
                    rng.choice(positive_positions, len(positive_positions), replace=True),
                    rng.choice(negative_positions, len(negative_positions), replace=True),
                ]
            )
            values = metric_values(
                y[positions],
                cyber[positions],
                fused[positions],
                cyber_threshold,
                fused_threshold,
            )
            sample_rows.append({"dataset": dataset_name, "repeat": repeat, **values})
    else:
        pairs = sorted(frame["pair_id"].unique().tolist())
        pair_to_positions = {
            pair: np.flatnonzero(frame["pair_id"].to_numpy() == pair) for pair in pairs
        }
        for repeat in range(BOOTSTRAP_REPEATS):
            sampled_pairs = rng.choice(pairs, len(pairs), replace=True)
            positions = np.concatenate([pair_to_positions[int(pair)] for pair in sampled_pairs])
            values = metric_values(
                y[positions],
                cyber[positions],
                fused[positions],
                cyber_threshold,
                fused_threshold,
            )
            sample_rows.append({"dataset": dataset_name, "repeat": repeat, **values})

samples = pd.DataFrame(sample_rows)
samples.to_csv(SAMPLES_OUT, index=False)

summary_rows: list[dict[str, Any]] = []
for dataset_name, group in samples.groupby("dataset", sort=False):
    point = point_estimates[dataset_name]
    frame = datasets[dataset_name]
    unit_count = (
        int(frame["pair_id"].nunique())
        if "pair_id" in frame.columns
        else int(len(frame))
    )
    for metric, direction in [
        ("auprc_difference_fused_minus_cyber", "positive_favors_fused"),
        ("brier_difference_fused_minus_cyber", "negative_favors_fused"),
        ("f1_difference_fused_minus_cyber", "positive_favors_fused"),
    ]:
        values = group[metric].to_numpy(dtype=float)
        summary_rows.append(
            {
                "dataset": dataset_name,
                "resampling_unit": "matched_pair" if "pair_id" in frame.columns else "event_within_class",
                "rows": len(frame),
                "resampling_units": unit_count,
                "positive_rows": int(frame["model_target"].sum()),
                "bootstrap_repeats": BOOTSTRAP_REPEATS,
                "metric": metric,
                "direction": direction,
                "cyber_point_estimate": point[metric.replace("_difference_fused_minus_cyber", "").replace("auprc", "cyber_auprc").replace("brier", "cyber_brier").replace("f1", "cyber_f1")]
                if False else np.nan,
                "fused_point_estimate": np.nan,
                "difference_point_estimate": point[metric],
                "bootstrap_standard_error": float(np.std(values, ddof=1)),
                "ci_95_lower_percentile": float(np.quantile(values, 0.025)),
                "ci_95_upper_percentile": float(np.quantile(values, 0.975)),
                "probability_difference_gt_zero": float(np.mean(values > 0)),
                "probability_difference_lt_zero": float(np.mean(values < 0)),
                "probability_difference_eq_zero": float(np.mean(values == 0)),
            }
        )

summary = pd.DataFrame(summary_rows)
point_columns = {
    "auprc_difference_fused_minus_cyber": ("cyber_auprc", "fused_auprc"),
    "brier_difference_fused_minus_cyber": ("cyber_brier", "fused_brier"),
    "f1_difference_fused_minus_cyber": ("cyber_f1", "fused_f1"),
}
for row_index, row in summary.iterrows():
    cyber_key, fused_key = point_columns[str(row["metric"])]
    point = point_estimates[str(row["dataset"])]
    summary.loc[row_index, "cyber_point_estimate"] = point[cyber_key]
    summary.loc[row_index, "fused_point_estimate"] = point[fused_key]
summary.to_csv(SUMMARY_OUT, index=False)

manifest = {
    "analysis_version": "paired_rf_bootstrap_final_v3",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "random_state": RANDOM_STATE,
    "bootstrap_repeats": BOOTSTRAP_REPEATS,
    "comparison": "fused random forest minus cyber-only random forest",
    "full_test_resampling": (
        "paired event bootstrap stratified by class; positive and negative event counts are "
        "preserved in every replicate"
    ),
    "matched_resampling": (
        "paired bootstrap over matched-pair identifiers; both members of each sampled pair "
        "are retained together"
    ),
    "confidence_interval": "two-sided percentile interval at 2.5% and 97.5%",
    "thresholds": {
        "cyber_random_forest": cyber_threshold,
        "fused_random_forest": fused_threshold,
        "selection_data": "chronological validation partition",
        "selection_policy": "maximize F1; tie by higher recall then lower threshold",
    },
    "inputs": {
        str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
        for path in [PREDICTIONS_IN, MATCHED_PAIRS_IN, THRESHOLDS_IN]
    },
    "interpretation_boundary": (
        "Intervals quantify uncertainty for the frozen leakage-safe controlled-replay test "
        "events. They do not imply causal or site-coincident aircraft-cyber detection."
    ),
}
MANIFEST_OUT.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

print("Final paired RF bootstrap completed.")
print(summary.to_string(index=False))
