# Chapter 4 — Results

## 4.1 Evaluation design

The final evaluation used 12,185 candidate low-altitude point events and a common held-out set of 1,828 events, including 913 positive targets (49.95%). Table 4.1 summarizes the frozen evaluation design. All model comparisons used identical held-out event identifiers, and the paired bootstrap preserved the binary target strata. The aircraft and NF-UNSW-NB15-v3 observations were joined through a controlled temporal replay; therefore, these results characterize the analytical pipeline rather than observed site-coincident aircraft/cyber incidents.

## 4.2 RQ1: Timestamp alignment fidelity

Table 4.2 and Figure 4.1 compare aircraft-reported event time with snapshot/processing time. Event-time MAE was 0 seconds by definition relative to the event timestamp, whereas processing-time MAE was 1.372 seconds. The resulting MAE reduction was 1.372 seconds (95% bootstrap CI 1.327 to 1.421). H1 was supported under the frozen requirement of at least 1.0 second. This finding evaluates timestamp fidelity within the source records; it is not evidence of real cross-domain clock synchronization.

## 4.3 RQ2: Comparative predictive performance

Table 4.3 reports validation and held-out performance, while Figure 4.2 displays held-out AUPRC with paired-bootstrap uncertainty. The strongest held-out point estimate was produced by **Fused tabular — decision_tree** with AUPRC 1.0000. The full R-GCN minus temporal Transformer AUPRC difference was -0.0821 (95% CI -0.0936 to -0.0707). H2 was not supported under the required +0.03 AUPRC margin. Figure 4.3 reports each model's frozen p95 latency using the measurement definition retained from its source notebook.

## 4.4 RQ3: Lag/relation contribution and latency

Table 4.4 presents paired AUPRC differences, and Table 4.5 consolidates all hypothesis decisions. The full R-GCN minus lag/relation-ablation difference was +0.0119 (95% CI +0.0061 to +0.0176). Full-model local p95 latency was 2.410 ms against the 500.0 ms limit. Because H3 required both an AUPRC gain of at least 0.05 and latency within the limit, H3 was not supported.

## 4.5 Robustness and sensitivity

Table 4.6 and Figure 4.5 summarize the ±120-, ±300-, and ±600-second alignment windows without retraining the models. Flow coverage ranged from 99.06% to 100.00%, while target prevalence ranged from 57.46% to 58.33%. These changes describe sensitivity of the controlled temporal mapping, not observed changes in operational attack incidence. Table 4.7 and Figure 4.6 summarize frozen-threshold subgroup diagnostics. The subgroup results are descriptive because the available group sizes and class counts were not designed as separate inferential tests.

## 4.6 Explainability results

Table 4.8 and Figures 4.7–4.9 summarize within-method explanation rankings. The highest-ranked Transformer feature was `numeric__nf3_dst_ip_unique_count`. The highest-ranked full R-GCN event-node feature was `numeric__nf3_byte_total_max`, and the highest-ranked full R-GCN relation diagnostic was `episode_previous_event`. Gradient-times-input values and relation-weight norms have different meanings and scales; rankings were therefore interpreted within their originating explanation method rather than as cross-model effect sizes.

## 4.7 Summary of hypothesis outcomes

H1 was supported, H2 was not supported, and H3 was not supported. The results support conclusions about timestamp fidelity and model behavior under the controlled multimodal alignment design. They do not demonstrate that an NF3 flow caused, accompanied, or was produced by a specific aircraft observation near PANC.
