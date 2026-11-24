# Chapter 4 Validity-Extension Insert (v3)

## Proxy-variable ablation and modality permutation

To determine whether the random-forest results depended on proxy variables, the cyber-only, physical-only, and fused random forests were retrained after removing prespecified feature groups. All ablations retained the original chronological partitions, random-forest hyperparameters, seed, and validation-only threshold-selection procedure. Removing the four direct flow-count variables did not change cyber-only or fused test AUPRC from 1.000. In contrast, removing a broader block of flow-presence and volume measures reduced cyber-only AUPRC to 0.942087 and fused AUPRC to 0.940830. Removing that block together with imputation-derived missingness indicators yielded cyber-only AUPRC of 0.936654. Removing all tested proxy groups yielded fused AUPRC of 0.937510.

The narrower missingness and provenance ablations did not explain the ceiling result. Disabling imputation-generated indicators preserved AUPRC of 1.000 for both cyber-only and fused models, and removing aircraft identifier/source or acquisition-provenance fields also preserved fused AUPRC of 1.000. Thus, no single missingness or source-identity cue was necessary for perfect ranking. The larger performance reduction after removal of aggregate flow measures indicates that the RFs relied substantially on cyber flow magnitude and diversity, while the remaining temporal and interarrival-time features retained strong target separability.

A held-out modality-block permutation provided a complementary frozen-model test. Across 1,000 joint permutations of the physical feature block, fused-model AUPRC remained 1.000 and mean Brier change was -0.000006, 95% empirical interval [-0.000030, 0.000016]. As a sensitivity control, 1,000 joint cyber-block permutations reduced mean AUPRC to 0.500288 and increased Brier by 0.452807. The contrast indicates that the frozen fused RF's held-out predictions depended on cyber rather than physical inputs.

## Paired cyber-only and fused random-forest comparison

The final paired comparison used 20,000 class-stratified event bootstraps for the full test set and 20,000 matched-pair bootstraps for the 531 flow-covered positive-negative pairs. Both models obtained AUPRC and F1 of 1.000 in both evaluations, yielding an AUPRC difference of zero. The full-test fused-minus-cyber Brier difference was 0.002572, 95% CI [0.002304, 0.002851]. The matched-pair difference was 0.003210, 95% CI [0.002896, 0.003531]. Because lower Brier loss is better and both intervals were positive, the cyber-only RF was better calibrated. These results do not support a random-forest fusion advantage.

## Standardized latency

Prediction latency was remeasured under a common protocol using the same 200 class-balanced held-out events, three repetitions per event, 20 warm-ups, CPU-only execution, and a single compute thread. The in-memory raw-to-score boundary included preprocessing, sequence or two-hop local-subgraph construction, tensor conversion, and inference, while excluding disk access, model loading, network transfer, and service orchestration. Raw-to-score p95 latency was 5.049 ms for the Transformer, 7.891 ms for the full R-GCN, 7.554 ms for the ablated R-GCN, 9.799 ms for the cyber-only RF, and 10.995 ms for the fused RF. Every model remained below the 500 ms operational criterion, but these steady-state measurements should not be interpreted as deployment service-level guarantees.

## Validity interpretation

Together, the controls show that leakage-safe partitioning did not eliminate strong NF3 target separability, but they also narrow the substantive interpretation. The fused RF did not gain discrimination from physical features, its calibration was worse than the cyber-only RF, and its predictions were insensitive to joint permutation of the physical block. The evidence therefore supports cyber-window classification under controlled replay, not demonstrated multimodal detection of hostile aircraft or a causal aircraft-network relationship.

## Generative-AI disclosure

OpenAI Codex assisted with analysis-code generation, execution orchestration, statistical summarization, and drafting. The student remains responsible for verification, interpretation, and final submitted language.
