# NF3 Leakage-Safe Reanalysis v2 — Handoff Guide

Start with `NF3_LEAKAGE_SAFE_REANALYSIS_REPORT_v2.md` for the findings and interpretation.

## Main directories

- `audit_original_v1/`: full audit of the original 2,974,178-row edge table.
- `alignment_leakage_safe_v2/`: corrected event features, targets, source blocks, partition audits, and bounded graph edges.
- `tabular_modeling_v4/`: 14 trained tabular models, predictions, metrics, feature manifests, and figures.
- `transformer_v2/`: trained Transformer, predictions, metrics, attributions, and figures.
- `rgcn_leakage_safe_v2/`: full and ablated R-GCNs, graph manifests, predictions, attributions, and figures.
- `results_v2/`: consolidated H1–H3 results and paired uncertainty analysis.
- `control_evaluations_v2/`: any-flow and coarsened-exact matched evaluations.
- `chapter4_exports_v2/`: numbered Chapter 4 tables and figures, narrative, and HTML report.
- `xai_leakage_safe_v2/`: XAI fidelity, stability, latency, provenance, and analyst cases.
- `validation_v2/`: independent 34-check validation gate.
- `code/`: standalone audit, alignment, controls, and validation scripts.
- `notebooks/`: corrected Notebooks 04–09.

## Recommended review order

1. Read the technical report.
2. Inspect `audit_original_v1/nf3_recurrence_gate_v1.json`.
3. Inspect `alignment_leakage_safe_v2/point_event_nf3_alignment_leakage_safe_v2_build_summary.json`.
4. Inspect `control_evaluations_v2/ANY_FLOW_AND_MATCHED_EVALUATION_REPORT.md`.
5. Inspect `results_v2/consolidated_hypothesis_evaluation_v2.csv`.
6. Inspect `validation_v2/independent_validation_summary_v2.json`.
7. Verify the top-level `SHA256SUMS.txt` before reuse.

## Important boundary

The corrected analysis prevents NF3 source observations and source-time neighborhoods from recurring across model partitions. It does not establish historical co-occurrence or causation between aircraft events and NF3 flows.
