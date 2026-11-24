# NF3 Proxy Ablation, Paired RF Bootstrap, and Standardized Latency Report (v3)

## Completion status

All three requested validity extensions were completed against the frozen leakage-safe v2 data split and artifacts:

1. Random-forest proxy ablations and a held-out physical-feature permutation test.
2. A final paired cyber-only versus fused RF bootstrap comparison.
3. A standardized single-event latency benchmark across physical, cyber, and fused RFs, the temporal Transformer, and both R-GCN variants.

All frozen-model probability reproduction checks passed. The largest random-forest reproduction error was 2.22e-16; the largest neural-model reproduction error in the latency benchmark was 1.19e-07.

## 1. Proxy ablations

Each ablation retrained the same 200-tree random forest using the original chronological train/validation/test assignments, fixed seed 42, identical validation-threshold policy, and the same target. A positive Δ Brier is worse.

| Model/control | AUPRC | Δ AUPRC | Brier | Δ Brier |
| --- | --- | --- | --- | --- |
| Physical RF baseline | 0.518618 | 0.000000 | 0.250105 | 0.000000 |
| Physical RF: remove identifier/source | 0.511123 | -0.007495 | 0.250658 | 0.000553 |
| Physical RF: remove acquisition provenance | 0.508826 | -0.009792 | 0.250503 | 0.000397 |
| Cyber RF baseline | 1.000000 | 0.000000 | 0.002526 | 0.000000 |
| Cyber RF: disable imputation indicators | 1.000000 | 0.000000 | 0.001316 | -0.001211 |
| Cyber RF: remove four flow-count fields | 1.000000 | 0.000000 | 0.002704 | 0.000178 |
| Cyber RF: remove broad flow presence/volume block | 0.942087 | -0.057913 | 0.115911 | 0.113384 |
| Cyber RF: remove flow block + disable imputation indicators | 0.936654 | -0.063346 | 0.118047 | 0.115520 |
| Fused RF baseline | 1.000000 | 0.000000 | 0.005098 | 0.000000 |
| Fused RF: remove identifier/source | 1.000000 | 0.000000 | 0.007017 | 0.001918 |
| Fused RF: remove acquisition provenance | 1.000000 | 0.000000 | 0.005364 | 0.000266 |
| Fused RF: remove broad flow presence/volume block | 0.940830 | -0.059170 | 0.119671 | 0.114573 |
| Fused RF: remove all tested proxy groups | 0.937510 | -0.062490 | 0.120520 | 0.115422 |

### Interpretation

- Removing the four direct flow-count fields alone did not change the perfect cyber or fused AUPRC.
- Removing the broader flow-presence/volume block reduced cyber AUPRC from 1.000000 to 0.942087 and fused AUPRC from 1.000000 to 0.940830.
- Disabling imputation-generated missingness indicators alone did not reduce ranking performance and improved RF Brier scores. The ceiling result is therefore not explained solely by imputer missingness indicators.
- Removing aircraft identifier/source and acquisition-provenance variables did not reduce fused AUPRC, although calibration shifted slightly. These fields are not necessary for the fused RF's perfect ranking.
- Removing every tested proxy group reduced fused AUPRC to 0.937510. The remaining high score shows that temporal, duration, lag, overlap, and interarrival-time cyber fields still contain substantial target information.

## 2. Held-out modality-block permutation

The frozen fused RF was evaluated over 1,000 joint permutations per modality. Every feature in a modality was reassigned with the same row permutation, preserving within-modality covariance. The cyber permutation is a negative control for test sensitivity.

| Permuted block | Baseline AUPRC | Mean AUPRC | AUPRC 2.5% | AUPRC 97.5% | Mean Δ Brier | Δ Brier 2.5% | Δ Brier 97.5% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| physical_features | 1.000000 | 1.000000 | 1.000000 | 1.000000 | -0.000006 | -0.000030 | 0.000016 |
| cyber_features_negative_control | 1.000000 | 0.500288 | 0.478387 | 0.523587 | 0.452807 | 0.431276 | 0.473584 |

Jointly permuting the physical block left AUPRC at 1.000000 in every run. Its mean Brier change was -0.000006, with a 95% empirical interval of [-0.000030, 0.000016]. By contrast, permuting the cyber block reduced mean AUPRC to 0.500288 and increased Brier loss by 0.452807 on average. The sensitivity control therefore worked, while the physical block contributed no measurable held-out discrimination to the frozen fused RF.

This is a predictive dependence test, not a causal test. Joint permutation deliberately breaks cross-modal correspondence and cannot establish that aircraft measurements are intrinsically irrelevant in other datasets or deployment settings.

## 3. Final paired RF bootstrap

The full test comparison used 20,000 paired event bootstraps stratified by class. The matched flow-covered comparison used 20,000 paired resamples of the 531 matched positive-negative pair identifiers, retaining both members of every sampled pair. The reported difference is fused minus cyber-only; lower Brier is better.

| Evaluation | Metric (fused − cyber) | Cyber | Fused | Difference | 95% paired bootstrap interval |
| --- | --- | --- | --- | --- | --- |
| Full test | AUPRC | 1.000000 | 1.000000 | 0.000000 | [0.000000, 0.000000] |
| Full test | BRIER | 0.002526 | 0.005098 | 0.002572 | [0.002304, 0.002851] |
| Full test | F1 | 1.000000 | 1.000000 | 0.000000 | [0.000000, 0.000000] |
| Matched flow-covered | AUPRC | 1.000000 | 1.000000 | 0.000000 | [0.000000, 0.000000] |
| Matched flow-covered | BRIER | 0.003053 | 0.006262 | 0.003210 | [0.002896, 0.003531] |
| Matched flow-covered | F1 | 1.000000 | 1.000000 | 0.000000 | [0.000000, 0.000000] |

Both RFs had perfect AUPRC and F1 on the leakage-safe v2 full test set and matched flow-covered subset, so their discrimination differences are exactly zero apart from floating-point roundoff. Calibration separated the models: on the full test set, fused-minus-cyber Brier was 0.002572, 95% CI [0.002304, 0.002851]. On matched flow-covered pairs, it was 0.003210, 95% CI [0.002896, 0.003531]. Both intervals are entirely positive, favoring the cyber-only RF.

The defensible conclusion is that adding physical features did not improve RF ranking or classification and produced worse probability calibration in both paired evaluations.

## 4. Standardized model latency

All models were measured on the same fixed-seed, class-balanced sample of 200 held-out events, with three repetitions per event, 20 warm-up events, CPU-only execution, and one compute thread. The table summarizes each event's median repetition. P95 intervals are percentile bootstraps over event medians.

| Model | Boundary | p50 ms | p95 ms | p95 CI low | p95 CI high | p99 ms |
| --- | --- | --- | --- | --- | --- | --- |
| physical_random_forest | in_memory_raw_to_score | 9.986 | 11.280 | 10.667 | 11.720 | 12.148 |
| physical_random_forest | model_ready_inference | 6.557 | 7.234 | 7.055 | 7.686 | 7.857 |
| cyber_random_forest | in_memory_raw_to_score | 8.649 | 9.799 | 9.345 | 9.984 | 10.699 |
| cyber_random_forest | model_ready_inference | 6.524 | 7.283 | 7.067 | 7.413 | 7.663 |
| fused_random_forest | in_memory_raw_to_score | 10.054 | 10.995 | 10.743 | 11.381 | 11.804 |
| fused_random_forest | model_ready_inference | 6.532 | 7.102 | 6.992 | 7.421 | 7.711 |
| temporal_transformer | in_memory_raw_to_score | 4.164 | 5.049 | 4.770 | 5.502 | 6.795 |
| temporal_transformer | model_ready_inference | 0.593 | 0.894 | 0.823 | 0.993 | 1.112 |
| full_rgcn | in_memory_raw_to_score | 6.963 | 7.891 | 7.741 | 8.095 | 8.155 |
| full_rgcn | model_ready_inference | 0.560 | 0.713 | 0.677 | 0.743 | 0.769 |
| ablated_rgcn | in_memory_raw_to_score | 6.758 | 7.554 | 7.445 | 7.785 | 7.940 |
| ablated_rgcn | model_ready_inference | 0.449 | 0.679 | 0.598 | 0.786 | 0.794 |

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
