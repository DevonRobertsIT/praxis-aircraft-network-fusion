# NF3 Source-Recurrence Audit and Leakage-Safe Reanalysis

**Analysis date:** September 3, 2026  
**Corrected analysis version:** v2  
**Independent validation decision:** **PASS (34 of 34 checks)**

## Executive conclusion

The original cyclic NF3 replay was not source-independent across the frozen chronological train, validation, and test partitions. A complete audit of all 2,974,178 rows in the original primary event–flow edge table found 478,226 distinct NF3 source flows used in two or more model partitions. The mapped NF3 source-time windows also overlapped across every partition pairing. The original recurrence gate therefore failed.

A leakage-safe correction was built and the analytical pipeline was rerun. The corrected alignment assigns each aircraft partition only to independent, contiguous NF3 source blocks, separated by a 1,200-second embargo and a 600-second safe-center inset. The original aircraft event population and its chronological 70/15/15 split were retained. The corrected gate found:

- zero shared NF3 source-flow identifiers between train, validation, and test;
- no overlap between mapped NF3 source-time window unions; and
- full agreement between each event's NF3 source partition and its model partition.

The correction materially changes the interpretation of the model results. H1 remains supported as a timestamp-fidelity result. H2 is no longer supported because the full R-GCN underperforms the Transformer. H3 remains unsupported because the R-GCN's gain over its lag/relation ablation is smaller than the required margin. Cyber-only and fused tabular models remain perfectly separable in the corrected replay, but fusion provides no incremental AUPRC over cyber-only data.

## 1. Full original NF3 source-block recurrence audit

The audit streamed the complete original edge table rather than relying on samples or the smaller aggregate feature bank.

| Audit quantity | Result |
|---|---:|
| Original primary event–flow edge rows audited | 2,974,178 |
| Original NF3 source flows tracked | 2,365,424 |
| Distinct source flows used in at least two model partitions | 478,226 |
| Primary source-window overlap detected | Yes |
| Original independence gate | **FAIL** |

### Exact source-flow recurrence

| Partition pairing | Left unique flows | Right unique flows | Reused flows | Reuse as fraction of smaller partition |
|---|---:|---:|---:|---:|
| Train–validation | 1,182,024 | 501,063 | 282,222 | 56.32% |
| Train–test | 1,182,024 | 469,730 | 245,114 | 52.18% |
| Validation–test | 501,063 | 469,730 | 105,402 | 22.44% |

### Primary source-time window overlap

| Partition pairing | Overlap hours | Overlap as fraction of smaller window union |
|---|---:|---:|
| Train–validation | 64.714 | 59.55% |
| Train–test | 63.303 | 57.90% |
| Validation–test | 19.989 | 18.39% |

These findings show that chronological partitioning by mapped 2025 aircraft-event time did not isolate the original 2015 NF3 observations. Repeated replay cycles caused the same source observations and source-time neighborhoods to recur in multiple model partitions.

## 2. Leakage-safe NF3 partition and alignment design

The correction detected two active NF3 capture sessions and divided each session into train, validation, and test source blocks. The design has the following safeguards:

1. Aircraft events keep their original chronological train, validation, or test assignment.
2. An event can map only to an NF3 block having the same partition label.
3. Adjacent NF3 source partitions are separated by a 1,200-second embargo.
4. Event centers are kept at least 600 seconds from block boundaries, matching the largest sensitivity half-window.
5. A source flow is assigned only when its full duration is contained within its source block.
6. Original NF3 flow identifiers, source timestamps, block identifiers, and partition labels remain traceable.

### NF3 source blocks

| Session | Partition | Assigned flows | Attack-labeled flows |
|---:|---|---:|---:|
| 1 | Train | 743,309 | 17,462 |
| 1 | Validation | 165,869 | 0 |
| 1 | Test | 166,928 | 0 |
| 2 | Train | 802,762 | 73,123 |
| 2 | Validation | 184,602 | 16,215 |
| 2 | Test | 177,463 | 15,563 |

The primary ±120-second alignment aggregated all 78,620,868 qualifying event–flow relationships. Because materializing that redundant table would be unnecessarily large and the downstream R-GCN already limits each event to 16 label-blind nearest flows, 192,956 bounded edge rows were stored. Aggregate event features and targets were calculated from all qualifying relationships, not only the bounded edge subset.

### Corrected independence gate

| Partition pairing | Reused source flows | Source-window overlap |
|---|---:|---|
| Train–validation | 0 | No |
| Train–test | 0 | No |
| Validation–test | 0 | No |

The corrected source-independence gate therefore **passes**.

## 3. Corrected dataset and split profile

| Split | Events | Positive targets | Prevalence |
|---|---:|---:|---:|
| Train | 8,529 | 5,157 | 60.46% |
| Validation | 1,828 | 931 | 50.93% |
| Test | 1,828 | 913 | 49.95% |
| Total | 12,185 | 7,001 | 57.46% |

The primary window contains mapped flows for 12,070 of 12,185 events (99.06%). All 1,828 corrected test events are flow-covered.

## 4. Pipeline rerun

The corrected artifacts were propagated through the tabular, Transformer, R-GCN, consolidated-results, Chapter 4 export, and XAI-validation stages. Fourteen tabular model artifacts, one Transformer checkpoint, and two R-GCN checkpoints were regenerated.

### Main held-out results

| Model | Held-out AUPRC | Held-out F1 | Brier score |
|---|---:|---:|---:|
| Physical logistic regression | 0.540181 | 0.664228 | 0.249120 |
| Physical random forest | 0.518618 | 0.666180 | 0.250105 |
| Cyber logistic regression | 1.000000 | 1.000000 | 0.000036 |
| Cyber random forest | 1.000000 | 1.000000 | 0.002526 |
| Fused logistic regression | 1.000000 | 1.000000 | 0.000074 |
| Fused random forest | 1.000000 | 1.000000 | 0.005098 |
| Temporal Transformer | 0.999996 | 0.997807 | 0.001543 |
| Full R-GCN | 0.917919 | 0.826087 | 0.138854 |
| Lag/relation-ablated R-GCN | 0.906042 | 0.808845 | 0.168239 |

No target-derived columns or event identifiers appear in the tabular predictor manifests. The perfect cyber and fused scores therefore do not result from directly supplying the target label to the models. They nevertheless require cautious interpretation because the target is constructed from NF3 attack labels and the corrected source-block replay may preserve capture-session and attack-signature regularities.

### H1–H3 after correction

| Hypothesis | Corrected result | 95% interval | Decision |
|---|---:|---:|---|
| H1: event-time MAE reduction ≥ 1.0 s | +1.371686 s | [1.327442, 1.421016] | Supported as timestamp fidelity |
| H2: full R-GCN − Transformer AUPRC ≥ +0.03 | −0.082077 | [−0.093636, −0.070747] | **Not supported** |
| H3: full R-GCN − ablation AUPRC ≥ +0.05 and p95 ≤ 500 ms | +0.011877; 2.410 ms | [0.006082, 0.017616] | **Not supported** |

H3's point difference is positive after the correction, but it remains far below the required +0.05 margin. The local R-GCN latency criterion passes; the predictive-gain criterion does not.

## 5. Any-flow baseline

| Analysis | Test events | Flow-covered events | Test prevalence | Any-flow AUPRC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original frozen v1 | 1,828 | 80 | 2.41% | 0.550000 | 0.550000 | 1.000000 | 0.709677 |
| Leakage-safe v2 | 1,828 | 1,828 | 49.95% | 0.499453 | 0.499453 | 1.000000 | 0.666180 |

In v1, flow presence alone was a strong proxy: every positive test event was flow-covered and only 36 flow-covered events were negative. In v2, every test event is flow-covered, so the any-flow rule is constant and its AUPRC equals prevalence. This eliminates the original missing-flow shortcut, although it does not by itself validate the realism of the corrected replay.

## 6. Matched flow-covered evaluation

Attack and benign events were matched 1:1 without replacement by coarsened exact matching on joint strata of log flow count and log total flow duration. Candidate coarsenings from two through six bins per variable were evaluated. For each analysis version, the design retaining the most pairs while keeping both absolute standardized mean differences at or below 0.10 was selected. Optimal assignment paired the closest observations within eligible strata.

| Analysis | Matched pairs | Flow-count SMD | Flow-duration SMD | Cyber RF AUPRC | Fused RF AUPRC | Fused − cyber AUPRC (95% paired-bootstrap interval) |
|---|---:|---:|---:|---:|---:|---:|
| Original frozen v1 | 21 | 0.0885 | 0.0774 | 0.994048 | 0.990842 | −0.003205 [−0.012894, 0.000000] |
| Leakage-safe v2 | 531 | 0.0804 | 0.0324 | 1.000000 | 1.000000 | 0.000000 [0.000000, 0.000000] |

The corrected matched analysis contains 1,062 events and remains perfectly separable for both cyber and fused random forests. It provides no evidence that aircraft features improve ranking beyond the cyber features. The original matched result is less precise because only 21 pairs satisfy the balance requirement.

For reference, the original full-test fused-minus-cyber random-forest difference was −0.000868 with a 95% interval of −0.004121 to approximately zero. The corrected full-test difference is effectively zero.

## 7. Corrected XAI validation

The XAI stage reproduced saved Transformer and R-GCN predictions with maximum absolute error of approximately 1.1×10⁻¹⁶. All eight provenance checks passed, and five held-out analyst cases were regenerated.

- Removing top-ranked evidence produced a mean absolute probability change of 0.192876, compared with 0.008924 for matched random-feature removal.
- Mean top-10 overlap under small input perturbations was 0.778; mean Spearman rank correlation was 0.802.
- R-GCN relation-weight norm and relation-removal AUPRC effect had Spearman correlation −0.371, reinforcing that weight norm is a diagnostic rather than faithful causal importance.
- Transformer explanation p95 latency was 4.764 ms across 10 samples.
- R-GCN local-subgraph explanation p95 latency was 36.346 ms across 10 samples.
- The provenance audit passed 8 of 8 checks.

## 8. Interpretation and residual limitations

The recurrence problem has been corrected for the v2 experimental design, but the repair does not convert the two independent historical datasets into observed cyber-physical incidents. The following boundaries remain essential:

1. The aircraft and NF3 records were not collected together, and their pairing remains synthetic and non-causal.
2. Only two active NF3 capture sessions are available. Contiguous source blocking can preserve capture-session or attack-period signatures that make the corrected cyber target unusually easy to separate.
3. The sharp change from sparse v1 flow coverage to near-complete v2 coverage shows that results are sensitive to the replay construction.
4. Perfect cyber and fused tabular performance should be described as NF3 window-label separability within a controlled replay, not verified low-altitude aircraft threat detection.
5. Fusion provides no incremental AUPRC over the strongest cyber-only models in either the full corrected test set or the matched corrected sample.
6. The stored corrected edge table is the label-blind 16-edge-per-event graph input; the 78.6 million full relationships were used for aggregation but were not all materialized.
7. End-to-end model latency is still not standardized across model families; current latency definitions differ.

## 9. Reproducibility and integrity

The handoff package contains:

- the complete v1 source-recurrence audit;
- the v2 source-block manifest, event mapping, aggregate features, targets, source-flow audit tables, and bounded edge table;
- all regenerated tabular, Transformer, and R-GCN model artifacts;
- consolidated H1–H3 results and paired bootstrap outputs;
- any-flow and matched flow-covered controls;
- Chapter 4 tables, figures, HTML report, and narrative export;
- XAI fidelity, stability, relation-consistency, latency, provenance, and analyst-case outputs;
- patched executable notebooks and standalone audit/control scripts; and
- SHA-256 manifests.

An independent validation script checked source recurrence, embargo separation, event/split integrity, target-feature exclusion, model artifacts, control balance, hypothesis decisions, XAI provenance, publication exports, and every component checksum manifest. All **34 of 34** checks passed.

## AI-assistance disclosure

OpenAI ChatGPT/Codex assisted with writing and executing the recurrence-audit, leakage-safe alignment, control-evaluation, validation, and reporting code; adapting the existing notebooks to the corrected artifacts; and drafting this technical report. The student remains responsible for reviewing the methods, reproducing the analysis, verifying the outputs and citations, and explaining the code and decisions to the advisor and defense committee.
