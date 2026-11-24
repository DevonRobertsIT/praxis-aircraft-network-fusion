from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


WORKSPACE = Path(__file__).resolve().parents[1]
PROJECT_ROOT = WORKSPACE / "audit_work" / "Project" / "Jupyter Notebook"
OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "point_event_v2"
    / "validity_extensions_v3"
)
ANALYSIS_OUTPUTS = WORKSPACE / "analysis_outputs"
ANALYSIS_OUTPUTS.mkdir(parents=True, exist_ok=True)

REPORT_OUT = ANALYSIS_OUTPUTS / "NF3_PROXY_ABLATION_RF_BOOTSTRAP_LATENCY_REPORT_v3.md"
CHAPTER_TEXT_OUT = OUTPUT_DIR / "chapter4_validity_extension_text_v3.md"
SUMMARY_OUT = OUTPUT_DIR / "validity_extensions_v3_build_summary.json"
CHECKSUMS_OUT = OUTPUT_DIR / "SHA256SUMS.txt"
PACKAGE_OUT = ANALYSIS_OUTPUTS / "Praxis_Validity_Extensions_v3.zip"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fmt(value: float, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


def markdown_table(frame: pd.DataFrame) -> str:
    def clean(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    headers = [clean(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(clean(value) for value in row) + " |")
    return "\n".join(lines)


ablation = pd.read_csv(OUTPUT_DIR / "proxy_ablation_random_forest_results_v3.csv")
permutation = pd.read_csv(OUTPUT_DIR / "physical_and_cyber_block_permutation_summary_v3.csv")
bootstrap = pd.read_csv(OUTPUT_DIR / "paired_rf_bootstrap_final_v3.csv")
latency = pd.read_csv(OUTPUT_DIR / "standardized_latency_summary_v3.csv")
proxy_manifest = json.loads(
    (OUTPUT_DIR / "proxy_ablation_manifest_v3.json").read_text(encoding="utf-8")
)
latency_manifest = json.loads(
    (OUTPUT_DIR / "standardized_latency_manifest_v3.json").read_text(encoding="utf-8")
)

checks: list[dict[str, object]] = []


def record(name: str, expected: object, actual: object, passed: bool) -> None:
    checks.append({"check": name, "expected": expected, "actual": actual, "passed": bool(passed)})


record("proxy_scenario_count", 17, len(ablation), len(ablation) == 17)
record(
    "proxy_feature_sets",
    ["cyber", "fused", "physical"],
    sorted(ablation["feature_set"].unique().tolist()),
    sorted(ablation["feature_set"].unique().tolist()) == ["cyber", "fused", "physical"],
)
max_rf_reproduction_error = max(
    float(value)
    for value in proxy_manifest["baseline_probability_reproduction_max_abs_error"].values()
)
record(
    "frozen_rf_probability_reproduction",
    "<=1e-12",
    max_rf_reproduction_error,
    max_rf_reproduction_error <= 1e-12,
)
record(
    "permutation_blocks_and_repeats",
    "2 blocks x 1000",
    permutation[["block", "repeats"]].to_dict("records"),
    len(permutation) == 2 and bool((permutation["repeats"] == 1000).all()),
)
record(
    "paired_bootstrap_repeats",
    "20000 per dataset",
    sorted(bootstrap["bootstrap_repeats"].unique().tolist()),
    len(bootstrap) == 6 and bool((bootstrap["bootstrap_repeats"] == 20000).all()),
)
record(
    "latency_model_boundary_rows",
    12,
    len(latency),
    len(latency) == 12,
)
record(
    "latency_common_event_count",
    200,
    sorted(latency["events"].unique().tolist()),
    bool((latency["events"] == 200).all()),
)
record(
    "latency_probability_reproduction",
    "<=1e-5",
    float(latency["reproduced_probability_max_abs_error"].max()),
    float(latency["reproduced_probability_max_abs_error"].max()) <= 1e-5,
)
record(
    "all_raw_to_score_p95_under_500ms",
    True,
    bool(
        (
            latency.loc[
                latency["measurement_boundary"].eq("in_memory_raw_to_score"),
                "latency_p95_ms",
            ]
            < 500.0
        ).all()
    ),
    bool(
        (
            latency.loc[
                latency["measurement_boundary"].eq("in_memory_raw_to_score"),
                "latency_p95_ms",
            ]
            < 500.0
        ).all()
    ),
)

if not all(bool(row["passed"]) for row in checks):
    raise AssertionError(f"Validation failure: {[row for row in checks if not row['passed']]}")


def get_ablation(feature_set: str, scenario: str) -> pd.Series:
    selected = ablation.loc[
        ablation["feature_set"].eq(feature_set) & ablation["scenario"].eq(scenario)
    ]
    if len(selected) != 1:
        raise AssertionError(f"Missing ablation result: {feature_set}/{scenario}")
    return selected.iloc[0]


key_ablation_specs = [
    ("physical", "baseline", "Physical RF baseline"),
    ("physical", "remove_identifier_source", "Physical RF: remove identifier/source"),
    ("physical", "remove_acquisition_provenance", "Physical RF: remove acquisition provenance"),
    ("cyber", "baseline", "Cyber RF baseline"),
    ("cyber", "no_imputer_missing_indicators", "Cyber RF: disable imputation indicators"),
    ("cyber", "remove_flow_count_only", "Cyber RF: remove four flow-count fields"),
    ("cyber", "remove_flow_presence_volume", "Cyber RF: remove broad flow presence/volume block"),
    (
        "cyber",
        "remove_flow_presence_volume_and_missing_indicators",
        "Cyber RF: remove flow block + disable imputation indicators",
    ),
    ("fused", "baseline", "Fused RF baseline"),
    ("fused", "remove_identifier_source", "Fused RF: remove identifier/source"),
    ("fused", "remove_acquisition_provenance", "Fused RF: remove acquisition provenance"),
    ("fused", "remove_flow_presence_volume", "Fused RF: remove broad flow presence/volume block"),
    ("fused", "remove_all_proxy_groups", "Fused RF: remove all tested proxy groups"),
]
ablation_table_rows = []
for feature_set, scenario, label in key_ablation_specs:
    row = get_ablation(feature_set, scenario)
    ablation_table_rows.append(
        {
            "Model/control": label,
            "AUPRC": fmt(row["auprc"]),
            "Δ AUPRC": fmt(row["auprc_change_from_retrained_baseline"]),
            "Brier": fmt(row["brier_score"]),
            "Δ Brier": fmt(row["brier_change_from_retrained_baseline"]),
        }
    )
ablation_table = markdown_table(pd.DataFrame(ablation_table_rows))

permutation_table = permutation[
    [
        "block",
        "baseline_auprc",
        "permuted_auprc_mean",
        "permuted_auprc_p2_5",
        "permuted_auprc_p97_5",
        "brier_change_mean",
        "brier_change_p2_5",
        "brier_change_p97_5",
    ]
].copy()
permutation_table.columns = [
    "Permuted block",
    "Baseline AUPRC",
    "Mean AUPRC",
    "AUPRC 2.5%",
    "AUPRC 97.5%",
    "Mean Δ Brier",
    "Δ Brier 2.5%",
    "Δ Brier 97.5%",
]
for column in permutation_table.columns[1:]:
    permutation_table[column] = permutation_table[column].map(fmt)
permutation_table_md = markdown_table(permutation_table)

bootstrap_table_rows = []
for dataset in [
    "full_test_stratified_event_bootstrap",
    "matched_flow_covered_pair_bootstrap",
]:
    for metric in [
        "auprc_difference_fused_minus_cyber",
        "brier_difference_fused_minus_cyber",
        "f1_difference_fused_minus_cyber",
    ]:
        row = bootstrap.loc[
            bootstrap["dataset"].eq(dataset) & bootstrap["metric"].eq(metric)
        ].iloc[0]
        difference = float(row["difference_point_estimate"])
        lower = float(row["ci_95_lower_percentile"])
        upper = float(row["ci_95_upper_percentile"])
        if max(abs(difference), abs(lower), abs(upper)) < 1e-12:
            difference = lower = upper = 0.0
        bootstrap_table_rows.append(
            {
                "Evaluation": "Full test" if dataset.startswith("full") else "Matched flow-covered",
                "Metric (fused − cyber)": metric.split("_difference")[0].upper(),
                "Cyber": fmt(row["cyber_point_estimate"]),
                "Fused": fmt(row["fused_point_estimate"]),
                "Difference": fmt(difference),
                "95% paired bootstrap interval": f"[{fmt(lower)}, {fmt(upper)}]",
            }
        )
bootstrap_table_md = markdown_table(pd.DataFrame(bootstrap_table_rows))

latency_order = {
    "physical_random_forest": 0,
    "cyber_random_forest": 1,
    "fused_random_forest": 2,
    "temporal_transformer": 3,
    "full_rgcn": 4,
    "ablated_rgcn": 5,
}
latency_table = latency.copy()
latency_table["order"] = latency_table["model"].map(latency_order)
latency_table = latency_table.sort_values(["order", "measurement_boundary"])
latency_table = latency_table[
    [
        "model",
        "measurement_boundary",
        "latency_p50_ms",
        "latency_p95_ms",
        "latency_p95_bootstrap_ci_lower_ms",
        "latency_p95_bootstrap_ci_upper_ms",
        "latency_p99_ms",
    ]
]
latency_table.columns = [
    "Model",
    "Boundary",
    "p50 ms",
    "p95 ms",
    "p95 CI low",
    "p95 CI high",
    "p99 ms",
]
for column in latency_table.columns[2:]:
    latency_table[column] = latency_table[column].map(lambda value: fmt(value, 3))
latency_table_md = markdown_table(latency_table)

physical_perm = permutation.loc[permutation["block"].eq("physical_features")].iloc[0]
cyber_perm = permutation.loc[
    permutation["block"].eq("cyber_features_negative_control")
].iloc[0]
full_brier = bootstrap.loc[
    bootstrap["dataset"].eq("full_test_stratified_event_bootstrap")
    & bootstrap["metric"].eq("brier_difference_fused_minus_cyber")
].iloc[0]
matched_brier = bootstrap.loc[
    bootstrap["dataset"].eq("matched_flow_covered_pair_bootstrap")
    & bootstrap["metric"].eq("brier_difference_fused_minus_cyber")
].iloc[0]

report = f"""# NF3 Proxy Ablation, Paired RF Bootstrap, and Standardized Latency Report (v3)

## Completion status

All three requested validity extensions were completed against the frozen leakage-safe v2 data split and artifacts:

1. Random-forest proxy ablations and a held-out physical-feature permutation test.
2. A final paired cyber-only versus fused RF bootstrap comparison.
3. A standardized single-event latency benchmark across physical, cyber, and fused RFs, the temporal Transformer, and both R-GCN variants.

All frozen-model probability reproduction checks passed. The largest random-forest reproduction error was {max_rf_reproduction_error:.3g}; the largest neural-model reproduction error in the latency benchmark was {float(latency['reproduced_probability_max_abs_error'].max()):.3g}.

## 1. Proxy ablations

Each ablation retrained the same 200-tree random forest using the original chronological train/validation/test assignments, fixed seed 42, identical validation-threshold policy, and the same target. A positive Δ Brier is worse.

{ablation_table}

### Interpretation

- Removing the four direct flow-count fields alone did not change the perfect cyber or fused AUPRC.
- Removing the broader flow-presence/volume block reduced cyber AUPRC from 1.000000 to {fmt(get_ablation('cyber', 'remove_flow_presence_volume')['auprc'])} and fused AUPRC from 1.000000 to {fmt(get_ablation('fused', 'remove_flow_presence_volume')['auprc'])}.
- Disabling imputation-generated missingness indicators alone did not reduce ranking performance and improved RF Brier scores. The ceiling result is therefore not explained solely by imputer missingness indicators.
- Removing aircraft identifier/source and acquisition-provenance variables did not reduce fused AUPRC, although calibration shifted slightly. These fields are not necessary for the fused RF's perfect ranking.
- Removing every tested proxy group reduced fused AUPRC to {fmt(get_ablation('fused', 'remove_all_proxy_groups')['auprc'])}. The remaining high score shows that temporal, duration, lag, overlap, and interarrival-time cyber fields still contain substantial target information.

## 2. Held-out modality-block permutation

The frozen fused RF was evaluated over 1,000 joint permutations per modality. Every feature in a modality was reassigned with the same row permutation, preserving within-modality covariance. The cyber permutation is a negative control for test sensitivity.

{permutation_table_md}

Jointly permuting the physical block left AUPRC at 1.000000 in every run. Its mean Brier change was {fmt(physical_perm['brier_change_mean'])}, with a 95% empirical interval of [{fmt(physical_perm['brier_change_p2_5'])}, {fmt(physical_perm['brier_change_p97_5'])}]. By contrast, permuting the cyber block reduced mean AUPRC to {fmt(cyber_perm['permuted_auprc_mean'])} and increased Brier loss by {fmt(cyber_perm['brier_change_mean'])} on average. The sensitivity control therefore worked, while the physical block contributed no measurable held-out discrimination to the frozen fused RF.

This is a predictive dependence test, not a causal test. Joint permutation deliberately breaks cross-modal correspondence and cannot establish that aircraft measurements are intrinsically irrelevant in other datasets or deployment settings.

## 3. Final paired RF bootstrap

The full test comparison used 20,000 paired event bootstraps stratified by class. The matched flow-covered comparison used 20,000 paired resamples of the 531 matched positive-negative pair identifiers, retaining both members of every sampled pair. The reported difference is fused minus cyber-only; lower Brier is better.

{bootstrap_table_md}

Both RFs had perfect AUPRC and F1 on the leakage-safe v2 full test set and matched flow-covered subset, so their discrimination differences are exactly zero apart from floating-point roundoff. Calibration separated the models: on the full test set, fused-minus-cyber Brier was {fmt(full_brier['difference_point_estimate'])}, 95% CI [{fmt(full_brier['ci_95_lower_percentile'])}, {fmt(full_brier['ci_95_upper_percentile'])}]. On matched flow-covered pairs, it was {fmt(matched_brier['difference_point_estimate'])}, 95% CI [{fmt(matched_brier['ci_95_lower_percentile'])}, {fmt(matched_brier['ci_95_upper_percentile'])}]. Both intervals are entirely positive, favoring the cyber-only RF.

The defensible conclusion is that adding physical features did not improve RF ranking or classification and produced worse probability calibration in both paired evaluations.

## 4. Standardized model latency

All models were measured on the same fixed-seed, class-balanced sample of 200 held-out events, with three repetitions per event, 20 warm-up events, CPU-only execution, and one compute thread. The table summarizes each event's median repetition. P95 intervals are percentile bootstraps over event medians.

{latency_table_md}

`model_ready_inference` excludes preprocessing and structural assembly. `in_memory_raw_to_score` begins with required raw records already in memory and includes preprocessing, sequence or local two-hop subgraph assembly, tensor conversion, and model inference. Both boundaries exclude disk I/O, model loading, network I/O, and service orchestration.

All raw-to-score p95 values were below 12 ms and no sampled event exceeded 500 ms. For the R-GCNs, the median local subgraph contained seven event nodes, 48 flow nodes, and 54 edges. These standardized measurements replace the earlier non-comparable timing definitions for cross-family discussion, but they remain steady-state workstation measurements rather than deployment service-level guarantees.

## 5. Implications for Chapters 3–5

- **Chapter 3:** Add the proxy-group definitions, joint modality-block permutation procedure, paired resampling units, two latency boundaries, common event sample, warm-up policy, single-thread CPU constraint, and exclusions from end-to-end timing.
- **Chapter 4:** Report that broad flow presence/volume features materially drive RF performance; identifier/source and missingness indicators do not explain the ceiling result; physical-block permutation has no measurable effect; and cyber-only RF calibration is significantly better than fused RF calibration.
- **Chapter 5:** Revise the fusion conclusion to state that physical data did not add predictive value for this controlled replay target. Treat this as a boundary finding about the present target and linkage design, not as evidence that physical context can never improve operational aircraft-threat assessment.

## Interpretation boundary

The target remains an NF3 label under leakage-safe controlled cross-domain replay. These tests strengthen internal validity for that experimental construct but do not establish site-coincident, causal, or operational hostile-aircraft detection.

## Generative-AI disclosure

OpenAI Codex assisted with analysis-code generation, execution orchestration, statistical summarization, and report drafting. The student remains responsible for verifying the analysis, explaining the code and methods, confirming citations, and approving all submitted text.
"""
REPORT_OUT.write_text(report, encoding="utf-8")

chapter_text = f"""# Chapter 4 Validity-Extension Insert (v3)

## Proxy-variable ablation and modality permutation

To determine whether the random-forest results depended on proxy variables, the cyber-only, physical-only, and fused random forests were retrained after removing prespecified feature groups. All ablations retained the original chronological partitions, random-forest hyperparameters, seed, and validation-only threshold-selection procedure. Removing the four direct flow-count variables did not change cyber-only or fused test AUPRC from 1.000. In contrast, removing a broader block of flow-presence and volume measures reduced cyber-only AUPRC to {fmt(get_ablation('cyber', 'remove_flow_presence_volume')['auprc'])} and fused AUPRC to {fmt(get_ablation('fused', 'remove_flow_presence_volume')['auprc'])}. Removing that block together with imputation-derived missingness indicators yielded cyber-only AUPRC of {fmt(get_ablation('cyber', 'remove_flow_presence_volume_and_missing_indicators')['auprc'])}. Removing all tested proxy groups yielded fused AUPRC of {fmt(get_ablation('fused', 'remove_all_proxy_groups')['auprc'])}.

The narrower missingness and provenance ablations did not explain the ceiling result. Disabling imputation-generated indicators preserved AUPRC of 1.000 for both cyber-only and fused models, and removing aircraft identifier/source or acquisition-provenance fields also preserved fused AUPRC of 1.000. Thus, no single missingness or source-identity cue was necessary for perfect ranking. The larger performance reduction after removal of aggregate flow measures indicates that the RFs relied substantially on cyber flow magnitude and diversity, while the remaining temporal and interarrival-time features retained strong target separability.

A held-out modality-block permutation provided a complementary frozen-model test. Across 1,000 joint permutations of the physical feature block, fused-model AUPRC remained 1.000 and mean Brier change was {fmt(physical_perm['brier_change_mean'])}, 95% empirical interval [{fmt(physical_perm['brier_change_p2_5'])}, {fmt(physical_perm['brier_change_p97_5'])}]. As a sensitivity control, 1,000 joint cyber-block permutations reduced mean AUPRC to {fmt(cyber_perm['permuted_auprc_mean'])} and increased Brier by {fmt(cyber_perm['brier_change_mean'])}. The contrast indicates that the frozen fused RF's held-out predictions depended on cyber rather than physical inputs.

## Paired cyber-only and fused random-forest comparison

The final paired comparison used 20,000 class-stratified event bootstraps for the full test set and 20,000 matched-pair bootstraps for the 531 flow-covered positive-negative pairs. Both models obtained AUPRC and F1 of 1.000 in both evaluations, yielding an AUPRC difference of zero. The full-test fused-minus-cyber Brier difference was {fmt(full_brier['difference_point_estimate'])}, 95% CI [{fmt(full_brier['ci_95_lower_percentile'])}, {fmt(full_brier['ci_95_upper_percentile'])}]. The matched-pair difference was {fmt(matched_brier['difference_point_estimate'])}, 95% CI [{fmt(matched_brier['ci_95_lower_percentile'])}, {fmt(matched_brier['ci_95_upper_percentile'])}]. Because lower Brier loss is better and both intervals were positive, the cyber-only RF was better calibrated. These results do not support a random-forest fusion advantage.

## Standardized latency

Prediction latency was remeasured under a common protocol using the same 200 class-balanced held-out events, three repetitions per event, 20 warm-ups, CPU-only execution, and a single compute thread. The in-memory raw-to-score boundary included preprocessing, sequence or two-hop local-subgraph construction, tensor conversion, and inference, while excluding disk access, model loading, network transfer, and service orchestration. Raw-to-score p95 latency was {fmt(latency.loc[(latency.model == 'temporal_transformer') & (latency.measurement_boundary == 'in_memory_raw_to_score'), 'latency_p95_ms'].iloc[0], 3)} ms for the Transformer, {fmt(latency.loc[(latency.model == 'full_rgcn') & (latency.measurement_boundary == 'in_memory_raw_to_score'), 'latency_p95_ms'].iloc[0], 3)} ms for the full R-GCN, {fmt(latency.loc[(latency.model == 'ablated_rgcn') & (latency.measurement_boundary == 'in_memory_raw_to_score'), 'latency_p95_ms'].iloc[0], 3)} ms for the ablated R-GCN, {fmt(latency.loc[(latency.model == 'cyber_random_forest') & (latency.measurement_boundary == 'in_memory_raw_to_score'), 'latency_p95_ms'].iloc[0], 3)} ms for the cyber-only RF, and {fmt(latency.loc[(latency.model == 'fused_random_forest') & (latency.measurement_boundary == 'in_memory_raw_to_score'), 'latency_p95_ms'].iloc[0], 3)} ms for the fused RF. Every model remained below the 500 ms operational criterion, but these steady-state measurements should not be interpreted as deployment service-level guarantees.

## Validity interpretation

Together, the controls show that leakage-safe partitioning did not eliminate strong NF3 target separability, but they also narrow the substantive interpretation. The fused RF did not gain discrimination from physical features, its calibration was worse than the cyber-only RF, and its predictions were insensitive to joint permutation of the physical block. The evidence therefore supports cyber-window classification under controlled replay, not demonstrated multimodal detection of hostile aircraft or a causal aircraft-network relationship.

## Generative-AI disclosure

OpenAI Codex assisted with analysis-code generation, execution orchestration, statistical summarization, and drafting. The student remains responsible for verification, interpretation, and final submitted language.
"""
CHAPTER_TEXT_OUT.write_text(chapter_text, encoding="utf-8")

build_summary = {
    "analysis_version": "validity_extensions_v3",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "completion": {
        "proxy_ablations_and_physical_permutation": True,
        "paired_rf_bootstrap": True,
        "standardized_model_latency": True,
    },
    "validation_checks": checks,
    "headline_results": {
        "cyber_rf_broad_flow_block_removed_auprc": float(
            get_ablation("cyber", "remove_flow_presence_volume")["auprc"]
        ),
        "fused_rf_broad_flow_block_removed_auprc": float(
            get_ablation("fused", "remove_flow_presence_volume")["auprc"]
        ),
        "fused_rf_all_proxy_groups_removed_auprc": float(
            get_ablation("fused", "remove_all_proxy_groups")["auprc"]
        ),
        "physical_block_permutation_mean_auprc": float(
            physical_perm["permuted_auprc_mean"]
        ),
        "cyber_block_permutation_mean_auprc": float(
            cyber_perm["permuted_auprc_mean"]
        ),
        "full_test_fused_minus_cyber_brier": float(
            full_brier["difference_point_estimate"]
        ),
        "matched_fused_minus_cyber_brier": float(
            matched_brier["difference_point_estimate"]
        ),
        "maximum_raw_to_score_p95_ms": float(
            latency.loc[
                latency["measurement_boundary"].eq("in_memory_raw_to_score"),
                "latency_p95_ms",
            ].max()
        ),
    },
    "report": REPORT_OUT.name,
    "chapter_insert": CHAPTER_TEXT_OUT.name,
}
SUMMARY_OUT.write_text(json.dumps(build_summary, indent=2), encoding="utf-8")

# Freeze checksums after every output has been created.
checksum_lines = []
for path in sorted(OUTPUT_DIR.rglob("*")):
    if path.is_file() and path != CHECKSUMS_OUT:
        checksum_lines.append(
            f"{sha256_file(path)}  {path.relative_to(OUTPUT_DIR).as_posix()}"
        )
CHECKSUMS_OUT.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

# Package every result, fitted ablation model, manifest, checksum, report, and source script.
if PACKAGE_OUT.exists():
    PACKAGE_OUT.unlink()
with zipfile.ZipFile(PACKAGE_OUT, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    archive.write(REPORT_OUT, arcname=REPORT_OUT.name)
    for path in sorted(OUTPUT_DIR.rglob("*")):
        if path.is_file():
            archive.write(path, arcname=f"validity_extensions_v3/{path.relative_to(OUTPUT_DIR)}")
    for script_number in [16, 17, 18, 19]:
        matches = sorted((WORKSPACE / "analysis_scripts").glob(f"{script_number}_*.py"))
        for path in matches:
            archive.write(path, arcname=f"scripts/{path.name}")

shutil.copy2(CHAPTER_TEXT_OUT, ANALYSIS_OUTPUTS / CHAPTER_TEXT_OUT.name)

print("Validity-extension validation and packaging completed.")
print(f"Report:  {REPORT_OUT}")
print(f"Package: {PACKAGE_OUT}")
print(pd.DataFrame(checks).to_string(index=False))
