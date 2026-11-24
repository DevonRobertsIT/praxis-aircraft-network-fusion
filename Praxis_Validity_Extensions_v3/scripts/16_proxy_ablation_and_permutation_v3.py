from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


RANDOM_STATE = 42
PERMUTATION_REPEATS = int(os.getenv("PRAXIS_PERMUTATION_REPEATS", "1000"))

WORKSPACE = Path(__file__).resolve().parents[1]
PROJECT_ROOT = WORKSPACE / "audit_work" / "Project" / "Jupyter Notebook"
BASE_DIR = PROJECT_ROOT / "data" / "processed" / "point_event_v2"
ALIGNMENT_DIR = BASE_DIR / "nf3_alignment_v2"
MODELING_DIR = BASE_DIR / "modeling_v4"
OUTPUT_DIR = BASE_DIR / "validity_extensions_v3"
MODEL_DIR = OUTPUT_DIR / "models"

FEATURES_IN = ALIGNMENT_DIR / "candidate_point_events_v2_with_context_and_nf3_primary.csv"
SPLITS_IN = MODELING_DIR / "model_split_assignments_v4.csv"
MANIFEST_IN = MODELING_DIR / "feature_manifest_v4.json"
SAVED_PREDICTIONS_IN = MODELING_DIR / "baseline_test_predictions_v4.csv"

RESULTS_OUT = OUTPUT_DIR / "proxy_ablation_random_forest_results_v3.csv"
PREDICTIONS_OUT = OUTPUT_DIR / "proxy_ablation_random_forest_predictions_v3.csv"
PERMUTATION_SAMPLES_OUT = OUTPUT_DIR / "physical_and_cyber_block_permutation_samples_v3.csv"
PERMUTATION_SUMMARY_OUT = OUTPUT_DIR / "physical_and_cyber_block_permutation_summary_v3.csv"
MANIFEST_OUT = OUTPUT_DIR / "proxy_ablation_manifest_v3.json"

for directory in [OUTPUT_DIR, MODEL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True, min_frequency=2)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True, min_frequency=2)


def usable_columns(
    train: pd.DataFrame,
    numeric_candidates: list[str],
    categorical_candidates: list[str],
) -> tuple[list[str], list[str], list[str]]:
    numeric: list[str] = []
    categorical: list[str] = []
    dropped: list[str] = []
    for column in numeric_candidates:
        observed = pd.to_numeric(train[column], errors="coerce").dropna()
        if observed.empty or observed.nunique() <= 1:
            dropped.append(column)
        else:
            numeric.append(column)
    for column in categorical_candidates:
        observed = train[column].astype("string").dropna()
        if observed.empty or observed.nunique() <= 1:
            dropped.append(column)
        else:
            categorical.append(column)
    return numeric, categorical, dropped


def build_rf_pipeline(
    numeric: list[str], categorical: list[str], add_missing_indicators: bool
) -> Pipeline:
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median", add_indicator=add_missing_indicators
                            ),
                        )
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("one_hot", make_one_hot_encoder()),
                    ]
                ),
                categorical,
            )
        )
    if not transformers:
        raise ValueError("No usable features remain.")
    return Pipeline(
        [
            (
                "preprocess",
                ColumnTransformer(
                    transformers=transformers,
                    remainder="drop",
                    sparse_threshold=0.3,
                    verbose_feature_names_out=True,
                ),
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=12,
                    min_samples_leaf=5,
                    class_weight="balanced_subsample",
                    n_jobs=1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def positive_probability(model: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    classes = np.asarray(model.named_steps["model"].classes_)
    column = int(np.flatnonzero(classes == 1)[0])
    return np.asarray(model.predict_proba(frame)[:, column], dtype=float)


def select_f1_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    p = precision[:-1]
    r = recall[:-1]
    denominator = p + r
    f1_values = np.divide(
        2.0 * p * r,
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator > 0,
    )
    best = float(np.nanmax(f1_values))
    candidates = np.flatnonzero(np.isclose(f1_values, best, rtol=1e-12, atol=1e-12))
    candidate_recall = r[candidates]
    candidates = candidates[np.isclose(candidate_recall, np.nanmax(candidate_recall))]
    selected = int(candidates[np.argmin(thresholds[candidates])])
    return {
        "threshold": float(thresholds[selected]),
        "validation_f1": float(f1_values[selected]),
        "validation_precision": float(p[selected]),
        "validation_recall": float(r[selected]),
        "policy": "maximize_validation_f1_tie_higher_recall_then_lower_threshold",
    }


def metric_row(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    predicted = (probabilities >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "auprc": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


@dataclass(frozen=True)
class Scenario:
    feature_set: str
    name: str
    remove: tuple[str, ...] = ()
    add_missing_indicators: bool = True


manifest = json.loads(MANIFEST_IN.read_text(encoding="utf-8"))
feature_sets = manifest["feature_sets"]
features = pd.read_csv(FEATURES_IN, low_memory=False)
splits = pd.read_csv(SPLITS_IN, low_memory=False)
data = features.merge(
    splits[["event_id", "model_target", "chronological_split"]],
    on="event_id",
    how="inner",
    validate="one_to_one",
)
track_radians = np.deg2rad(pd.to_numeric(data["track_degrees"], errors="coerce"))
data["track_sin"] = np.sin(track_radians)
data["track_cos"] = np.cos(track_radians)
data["model_target"] = pd.to_numeric(data["model_target"], errors="raise").astype("int8")
data["chronological_split"] = data["chronological_split"].astype("string")
data["aircraft_event_time_utc"] = pd.to_datetime(
    data["aircraft_event_time_utc"], utc=True, errors="raise"
)
data["event_id"] = data["event_id"].astype("string")
all_numeric = sorted(
    {
        str(column)
        for specification in feature_sets.values()
        for column in specification["numeric"]
    }
)
all_categorical = sorted(
    {
        str(column)
        for specification in feature_sets.values()
        for column in specification["categorical"]
    }
)
for column in all_numeric:
    data[column] = pd.to_numeric(data[column], errors="coerce")
for column in all_categorical:
    data[column] = data[column].astype("string")
data = data.sort_values(
    ["aircraft_event_time_utc", "event_id"], kind="mergesort"
).reset_index(drop=True)

train = data.loc[data["chronological_split"].eq("train")].copy()
validation = data.loc[data["chronological_split"].eq("validation")].copy()
test = data.loc[data["chronological_split"].eq("test")].copy()

flow_count_only = (
    "nf3_flow_count",
    "nf3_pre_event_flow_count",
    "nf3_at_event_flow_count",
    "nf3_post_event_flow_count",
)
flow_presence_volume = flow_count_only + (
    "nf3_flow_duration_sum_s",
    "nf3_packet_total_sum",
    "nf3_packet_total_mean",
    "nf3_packet_total_max",
    "nf3_byte_total_sum",
    "nf3_byte_total_mean",
    "nf3_byte_total_max",
    "nf3_protocol_unique_count",
    "nf3_src_ip_unique_count",
    "nf3_dst_ip_unique_count",
    "nf3_src_port_unique_count",
    "nf3_dst_port_unique_count",
    "nf3_total_overlap_duration_s",
)
explicit_missingness = (
    "stale_event_flag",
    "missing_hex_flag",
    "missing_callsign_flag",
    "window_boundary_truncated_flag",
)
identifier_source = (
    "aircraft_key_source",
    "missing_hex_flag",
    "missing_callsign_flag",
)
acquisition_provenance = (
    "source_type",
    "aircraft_key_source",
    "deduplication_status",
    "event_lag_seconds",
    "stale_event_flag",
    "missing_hex_flag",
    "missing_callsign_flag",
)

scenarios = [
    Scenario("physical", "baseline"),
    Scenario("physical", "no_imputer_missing_indicators", add_missing_indicators=False),
    Scenario("physical", "remove_identifier_source", identifier_source),
    Scenario("physical", "remove_acquisition_provenance", acquisition_provenance),
    Scenario("cyber", "baseline"),
    Scenario("cyber", "no_imputer_missing_indicators", add_missing_indicators=False),
    Scenario("cyber", "remove_flow_count_only", flow_count_only),
    Scenario("cyber", "remove_flow_presence_volume", flow_presence_volume),
    Scenario(
        "cyber",
        "remove_flow_presence_volume_and_missing_indicators",
        flow_presence_volume,
        False,
    ),
    Scenario("fused", "baseline"),
    Scenario("fused", "no_imputer_missing_indicators", add_missing_indicators=False),
    Scenario("fused", "remove_flow_count_only", flow_count_only),
    Scenario("fused", "remove_flow_presence_volume", flow_presence_volume),
    Scenario("fused", "remove_explicit_missingness_flags", explicit_missingness),
    Scenario("fused", "remove_identifier_source", identifier_source),
    Scenario("fused", "remove_acquisition_provenance", acquisition_provenance),
    Scenario(
        "fused",
        "remove_all_proxy_groups",
        tuple(sorted(set(flow_presence_volume + explicit_missingness + acquisition_provenance))),
        False,
    ),
]

result_rows: list[dict[str, Any]] = []
prediction_parts: list[pd.DataFrame] = []
scenario_manifest: list[dict[str, Any]] = []

for scenario_index, scenario in enumerate(scenarios, start=1):
    candidates_numeric = [
        str(c) for c in feature_sets[scenario.feature_set]["numeric"] if c not in set(scenario.remove)
    ]
    candidates_categorical = [
        str(c)
        for c in feature_sets[scenario.feature_set]["categorical"]
        if c not in set(scenario.remove)
    ]
    numeric, categorical, constant_dropped = usable_columns(
        train, candidates_numeric, candidates_categorical
    )
    columns = numeric + categorical
    model = build_rf_pipeline(numeric, categorical, scenario.add_missing_indicators)
    started = time.perf_counter()
    model.fit(train[columns], train["model_target"].to_numpy(dtype=np.int8))
    fit_seconds = time.perf_counter() - started
    validation_probability = positive_probability(model, validation[columns])
    threshold = select_f1_threshold(
        validation["model_target"].to_numpy(dtype=np.int8), validation_probability
    )
    test_probability = positive_probability(model, test[columns])
    metrics = metric_row(
        test["model_target"].to_numpy(dtype=np.int8),
        test_probability,
        float(threshold["threshold"]),
    )
    model_path = MODEL_DIR / f"{scenario.feature_set}__{scenario.name}__random_forest.joblib"
    joblib.dump(model, model_path, compress=3)
    row = {
        "scenario_order": scenario_index,
        "feature_set": scenario.feature_set,
        "scenario": scenario.name,
        "removed_feature_count": len(set(scenario.remove)),
        "removed_features": "|".join(sorted(set(scenario.remove))),
        "imputer_missing_indicators_enabled": int(scenario.add_missing_indicators),
        "usable_numeric_features": len(numeric),
        "usable_categorical_features": len(categorical),
        "training_constant_or_empty_dropped": "|".join(constant_dropped),
        "fit_seconds": fit_seconds,
        "selected_threshold": float(threshold["threshold"]),
        "validation_f1": float(threshold["validation_f1"]),
        **metrics,
        "model_file": model_path.name,
        "model_sha256": sha256_file(model_path),
    }
    result_rows.append(row)
    prediction_parts.append(
        pd.DataFrame(
            {
                "event_id": test["event_id"].astype(str).to_numpy(),
                "model_target": test["model_target"].to_numpy(dtype=np.int8),
                "feature_set": scenario.feature_set,
                "scenario": scenario.name,
                "predicted_probability": test_probability,
                "selected_threshold": float(threshold["threshold"]),
                "predicted_label": (
                    test_probability >= float(threshold["threshold"])
                ).astype(np.int8),
            }
        )
    )
    scenario_manifest.append(
        {
            "feature_set": scenario.feature_set,
            "scenario": scenario.name,
            "removed_features_requested": sorted(set(scenario.remove)),
            "usable_numeric_features": numeric,
            "usable_categorical_features": categorical,
            "training_constant_or_empty_dropped": constant_dropped,
            "imputer_missing_indicators_enabled": scenario.add_missing_indicators,
        }
    )
    print(
        f"[{scenario_index:02d}/{len(scenarios)}] {scenario.feature_set}/{scenario.name}: "
        f"AUPRC={metrics['auprc']:.6f}, Brier={metrics['brier_score']:.6f}"
    )

results = pd.DataFrame(result_rows)
for feature_set in results["feature_set"].unique():
    baseline = results.loc[
        results["feature_set"].eq(feature_set) & results["scenario"].eq("baseline")
    ].iloc[0]
    mask = results["feature_set"].eq(feature_set)
    results.loc[mask, "auprc_change_from_retrained_baseline"] = (
        results.loc[mask, "auprc"] - float(baseline["auprc"])
    )
    results.loc[mask, "brier_change_from_retrained_baseline"] = (
        results.loc[mask, "brier_score"] - float(baseline["brier_score"])
    )
results.to_csv(RESULTS_OUT, index=False)
predictions = pd.concat(prediction_parts, ignore_index=True)
predictions.to_csv(PREDICTIONS_OUT, index=False)

# Regression check against the frozen v2 baseline predictions.
saved_predictions = pd.read_csv(SAVED_PREDICTIONS_IN, low_memory=False)
reproduction_checks: dict[str, float] = {}
for feature_set in ["physical", "cyber", "fused"]:
    frozen = saved_predictions.loc[
        saved_predictions["split_strategy"].astype(str).eq("chronological_primary")
        & saved_predictions["feature_set"].astype(str).eq(feature_set)
        & saved_predictions["model"].astype(str).eq("random_forest"),
        ["event_id", "predicted_probability"],
    ].rename(columns={"predicted_probability": "frozen_probability"})
    reproduced = predictions.loc[
        predictions["feature_set"].eq(feature_set) & predictions["scenario"].eq("baseline"),
        ["event_id", "predicted_probability"],
    ].rename(columns={"predicted_probability": "reproduced_probability"})
    compared = frozen.merge(reproduced, on="event_id", validate="one_to_one")
    reproduction_checks[feature_set] = float(
        np.max(np.abs(compared["frozen_probability"] - compared["reproduced_probability"]))
    )
    if reproduction_checks[feature_set] > 1e-12:
        raise AssertionError(
            f"Retrained {feature_set} RF does not reproduce the frozen model: "
            f"max abs error={reproduction_checks[feature_set]:.12g}"
        )

# Held-out block-permutation tests on the frozen fused RF. Each permutation jointly
# reassigns the entire modality block, preserving within-modality covariance.
fused_model = joblib.load(
    MODELING_DIR / "models" / "chronological_primary__fused__random_forest.joblib"
)
fused_model.named_steps["model"].set_params(n_jobs=1)
fused_numeric = [str(c) for c in feature_sets["fused"]["numeric"]]
fused_categorical = [str(c) for c in feature_sets["fused"]["categorical"]]
fused_columns = fused_numeric + fused_categorical
physical_columns = [
    str(c) for c in feature_sets["physical"]["numeric"] + feature_sets["physical"]["categorical"]
]
cyber_columns = [str(c) for c in feature_sets["cyber"]["numeric"]]

y_test = test["model_target"].to_numpy(dtype=np.int8)
baseline_probability = positive_probability(fused_model, test[fused_columns])
baseline_auprc = float(average_precision_score(y_test, baseline_probability))
baseline_brier = float(brier_score_loss(y_test, baseline_probability))
rng = np.random.default_rng(RANDOM_STATE + 1600)
permutation_rows: list[dict[str, Any]] = []

for block_name, block_columns in [
    ("physical_features", physical_columns),
    ("cyber_features_negative_control", cyber_columns),
]:
    print(f"Running {PERMUTATION_REPEATS:,} joint {block_name} permutations ...")
    for repeat in range(PERMUTATION_REPEATS):
        order = rng.permutation(len(test))
        permuted = test[fused_columns].copy()
        permuted.loc[:, block_columns] = test.iloc[order][block_columns].to_numpy()
        probability = positive_probability(fused_model, permuted)
        auprc = float(average_precision_score(y_test, probability))
        brier = float(brier_score_loss(y_test, probability))
        permutation_rows.append(
            {
                "block": block_name,
                "repeat": repeat,
                "auprc": auprc,
                "auprc_change_from_baseline": auprc - baseline_auprc,
                "brier_score": brier,
                "brier_change_from_baseline": brier - baseline_brier,
            }
        )

permutation_samples = pd.DataFrame(permutation_rows)
permutation_samples.to_csv(PERMUTATION_SAMPLES_OUT, index=False)
summary_rows: list[dict[str, Any]] = []
for block_name, group in permutation_samples.groupby("block", sort=False):
    summary_rows.append(
        {
            "block": block_name,
            "repeats": len(group),
            "baseline_auprc": baseline_auprc,
            "permuted_auprc_mean": float(group["auprc"].mean()),
            "permuted_auprc_p2_5": float(group["auprc"].quantile(0.025)),
            "permuted_auprc_p50": float(group["auprc"].quantile(0.5)),
            "permuted_auprc_p97_5": float(group["auprc"].quantile(0.975)),
            "auprc_change_mean": float(group["auprc_change_from_baseline"].mean()),
            "auprc_change_p2_5": float(group["auprc_change_from_baseline"].quantile(0.025)),
            "auprc_change_p97_5": float(group["auprc_change_from_baseline"].quantile(0.975)),
            "probability_auprc_decreased": float(
                np.mean(group["auprc_change_from_baseline"].to_numpy() < 0)
            ),
            "baseline_brier": baseline_brier,
            "permuted_brier_mean": float(group["brier_score"].mean()),
            "permuted_brier_p2_5": float(group["brier_score"].quantile(0.025)),
            "permuted_brier_p50": float(group["brier_score"].quantile(0.5)),
            "permuted_brier_p97_5": float(group["brier_score"].quantile(0.975)),
            "brier_change_mean": float(group["brier_change_from_baseline"].mean()),
            "brier_change_p2_5": float(group["brier_change_from_baseline"].quantile(0.025)),
            "brier_change_p97_5": float(group["brier_change_from_baseline"].quantile(0.975)),
            "probability_brier_worsened": float(
                np.mean(group["brier_change_from_baseline"].to_numpy() > 0)
            ),
        }
    )
permutation_summary = pd.DataFrame(summary_rows)
permutation_summary.to_csv(PERMUTATION_SUMMARY_OUT, index=False)

payload = {
    "analysis_version": "validity_extensions_v3",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "random_state": RANDOM_STATE,
    "test_rows": len(test),
    "test_positive_rows": int(test["model_target"].sum()),
    "test_prevalence": float(test["model_target"].mean()),
    "random_forest_configuration": {
        "n_estimators": 200,
        "max_depth": 12,
        "min_samples_leaf": 5,
        "class_weight": "balanced_subsample",
        "n_jobs": 1,
        "random_state": RANDOM_STATE,
    },
    "scenario_definitions": scenario_manifest,
    "permutation_test": {
        "repeats_per_block": PERMUTATION_REPEATS,
        "unit": "held-out event",
        "scheme": "joint row reassignment of every raw feature in one modality block",
        "physical_block_feature_count": len(physical_columns),
        "cyber_negative_control_feature_count": len(cyber_columns),
        "within_block_covariance_preserved": True,
        "model_retrained": False,
        "frozen_model": "chronological_primary__fused__random_forest.joblib",
    },
    "baseline_probability_reproduction_max_abs_error": reproduction_checks,
    "inputs": {
        str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
        for path in [FEATURES_IN, SPLITS_IN, MANIFEST_IN, SAVED_PREDICTIONS_IN]
    },
    "interpretation_boundary": (
        "The target is the leakage-safe, controlled NF3 replay window label. Results do not "
        "establish site-coincident or causal aircraft-cyber threat detection."
    ),
}
MANIFEST_OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

print("Proxy ablation and modality-block permutation analysis completed.")
print(results[["feature_set", "scenario", "auprc", "brier_score"]].to_string(index=False))
print(permutation_summary.to_string(index=False))
