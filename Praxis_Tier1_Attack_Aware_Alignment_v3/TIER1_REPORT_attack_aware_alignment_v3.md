# Tier 1 — Attack-aware, session-confound-free NF3 alignment (v3)

**Run date:** 19 September 2026 · **Environment:** Linux container, 1 CPU, 3 GB RAM, Python 3.12, pandas 3.0.2, scikit-learn 1.8.0 (no PyTorch: Transformer / R-GCN reruns are staged, not executed)

## 1. What was asked and what the data allowed

The task was to choose NF3 source-block boundaries so that every session × partition cell contains both attack and benign windows, add the missing validator check, add a cross-session design, a session-indicator baseline with a reporting margin, a cluster bootstrap with an effective-sample-size estimate, and an empirical replacement for H1.

The first full-scale attempt (1-hour candidate segments, exhaustive assignment search) failed its own gate, and the reason is a property of the benchmark that the praxis must state. At the ±120 s window scale, NF-UNSW-NB15-v3 has **three regimes**, computed with the target's own containment rule on a 1 s timeline:

| Session | Regime | Start (UTC) | Duration |
|---|---|---|---|
| 1 | attack plateau | 2015-01-22 11:49:36 | 119.9 min |
| 1 | benign | 2015-01-22 13:49:30 | 635.9 min |
| 2 | attack (4.7 min) / benign (7.5 min) | 2015-02-18 00:23:27 | 12.2 min, too short for any block |
| 2 | attack plateau | 2015-02-18 00:35:40 | 713.8 min |

There are no short runs at all: attack presence is a step function with hour-scale plateaus. Consequently **Session 2 cannot contribute benign windows to any partition** (7.5 minutes of benign traffic cannot host a block with a 1,200 s embargo and inset), and the literal requirement "both classes in every session × partition cell" is infeasible by data. The allocator was therefore rewritten as **regime-stratified**: every (session, regime) interval is split into contiguous partition sub-blocks with a 1,200 s embargo at every partition change and the safe-center inset, so the only mixed material (Session 1) reaches every partition, and two designs are provided:

- `within_session_stratified` — every partition receives a share of every regime interval (Session 1's plateau is split three ways; Session 2 feeds only positives).
- `cross_regime_s2_train_s1_attack_test` (**primary**) — train/validation on Session 2 attacks + Session 1 benign; test on Session 1's attack plateau + held-out Session 1 benign. The test attack regime is never seen in training and every test event comes from one session, so session identity carries no information about the test label.

Gates now include `both_classes_within_each_partition`, `session_indicator_not_perfect_within_any_partition` (test partition for cross-regime designs), and the per-cell table `both_classes_within_each_session_partition` with data-infeasible cells flagged. All three runs use the primary ±120 s window (inset 120 s) so they share an identical block layout; the ±300/±600 s sensitivity windows are one flag away (`--windows all`, inset 600 s).

## 2. Alignment results

| Quantity | v2 (praxis) | v3 cross-regime (primary) | v3 within-session |
|---|---|---|---|
| Gate decision | PASS (34/34, no session check) | **PASS** (all five gates) | **PASS** |
| Cross-partition source-flow reuse | 0 | 0 | 0 |
| Source blocks | 6 | 6 (S1 attack→test; S1 benign 70/15/15; S2 attack 82/18 train/val) | 9 |
| Positives / negatives | 7,001 / 5,184 | 7,094 / 5,091 | 6,820 / 5,365 |
| Test positives / negatives | 913 / 915 | 1,015 / 813 (all Session 1) | 1,016 / 812 |
| Sessions present in test | S1, S2 | **S1 only** | S1, S2 |
| Session-indicator AUPRC on test | **1.000** | **0.555 = prevalence (uninformative)** | 0.957 |
| Independent 240 s source bins in test (pos / neg) | 28 / 21 | 28 / 21 | 28 / 21 |
| Replay cycles in test | 399 | 367 | 355 |
| Events with flows | 12,070 | 12,098 | 12,088 |

The v2 validator, re-run with the two added checks, reports **34 / 36**: `both_classes_within_each_session_partition` fails on the four degenerate cells (test S1 915/0, test S2 0/913, validation S1 897/0, validation S2 0/931) and `v2_test_session_indicator_not_perfect` fails at 1.000 (`validation_v2_rerun/`).

## 3. Tabular models on the cross-regime design (feature sets and estimators identical to v4)

| Model | AUPRC | F1 | Brier | Margin over strongest baseline | Bin-cluster 95% CI (AUPRC) | DEFF | n_eff |
|---|---|---|---|---|---|---|---|
| Physical LR | 0.578 | 0.714 | 0.250 | +0.023 | [0.43, 0.72] | 18.9 | 97 |
| Physical RF | 0.572 | 0.712 | 0.248 | +0.016 | [0.43, 0.71] | 19.6 | 93 |
| Physical DT | 0.551 | 0.714 | 0.253 | −0.004 | [0.41, 0.69] | 29.8 | 61 |
| Cyber LR | 0.99999 | 0.949 | 0.0089 | +0.445 | [0.9999, 1.0000] | 4.7 | 386 |
| Cyber RF | 1.000 | 0.985 | 0.0043 | +0.445 | [1.000, 1.000] | — | — |
| Cyber DT | 1.000 | 0.983 | 0.0003 | +0.445 | [1.000, 1.000] | — | — |
| Fused LR | 0.99998 | 0.958 | 0.0082 | +0.445 | [0.9999, 1.0000] | 2.4 | 770 |
| Fused RF | 1.000 | 0.998 | 0.0063 | +0.445 | [1.000, 1.000] | — | — |
| Fused DT | 1.000 | 1.000 | 0.0000 | +0.445 | [1.000, 1.000] | — | — |

Baselines on this test set: session indicator 0.555, any-flow 0.555 (all 1,828 events flow-covered), prevalence 0.555. Margin rule: AUPRC must exceed the strongest baseline by ≥ 0.02 with a bin-cluster interval excluding zero. Paired fused − cyber AUPRC: 0.000 for RF and DT, −6.6 × 10⁻⁶ for LR (bin-cluster CI [−2.6 × 10⁻⁵, 8.6 × 10⁻⁷]).

Decision-tree roots (`tabular_v3_cross_regime/decision_tree_rules_v3.txt`): cyber and fused both split first on `nf3_dst_ip_unique_count ≤ 21.5`; physical splits on site recurrence counts and reaches prevalence.

Within-session design (`tabular_v3_within_session/`, `baselines_v3_within_session/`): cyber and fused LR/RF again 1.000; margin over the session indicator +0.043 (passes the 0.02 rule but is small); physical 0.56–0.58 with n_eff ≈ 90–96.

## 4. Revised H1 (timestamp basis)

Same cross-regime block layout, event centers placed on `snapshot_time_utc` instead of `aircraft_event_time_utc` (origin fixed at the partition's first aircraft event, so each event shifts by exactly its own processing lag; mean |lag| 1.372 s, p95 3.0 s).

- Primary label changed for **2 of 12,185 events (0.016%)**; 0 of 1,828 test events.
- Cyber RF trained on processing-time alignment, scored on event-time targets: AUPRC 1.000 vs 1.000 — loss 0.000, bin-cluster CI [−1 × 10⁻¹⁶, 1 × 10⁻¹⁶].
- **H1 (revised): not supported** on both criteria (≥ 5% label change; ≥ 0.02 AUPRC loss).

The original H1 measured a quantity (mean |snapshot − event| = 1.37 s) that has no downstream consequence at a ±120 s window; that is now an empirical statement rather than a definitional one. (Two label changes arise from events sitting within their lag of a replay-cycle boundary, an artifact of the modulo mapping; it is recorded in `h1_v3/h1_label_change_by_event_v3.csv`.)

## 5. What this means for the praxis

1. **The session confound is removed and the perfect cyber result survives.** Under the cross-regime design the test attack regime was never seen in training and the session indicator is uninformative, yet cyber LR/RF/DT still reach AUPRC ≈ 1.000 with `nf3_dst_ip_unique_count` as the root split. Destination-host diversity is a transferable attack-period signature of the IXIA campaign in this benchmark, not a session fingerprint. Chapter 4 can now say this with evidence instead of a caveat.
2. **It is a ceiling task.** A depth-3 tree separates the target perfectly. Any model-family comparison on this target (H2, H3) is a comparison of failure modes, not of capability; the praxis should say so explicitly and re-frame H2/H3 as robustness/latency questions or move them to the Tier 2 coupling study.
3. **The held-out set is small in the way that matters.** 1,828 test events cover 49 independent 240 s source bins (28 attack, 21 benign). Physical-model intervals widen from ±0.03 (event bootstrap) to ±0.14 (bin-cluster bootstrap); design effects are 19–30 and n_eff ≈ 60–100. Report these instead of the event-level intervals.
4. **Fusion adds nothing, still by construction.** Physical features are uninformative for an NF3 label (AUPRC at prevalence; fused − cyber = 0). The Tier 2 injection study remains the only way to give the fusion question content.
5. **Session 2 can only ever supply positives** at this window scale. Any future design must either stratify by regime as done here or redefine the target (finer window, attack-flow fraction, or category).

## 6. Files

- `code/11_nf3_leakage_safe_alignment_v3.py` — regime-stratified alignment; imports the v2 feature/target/edge functions unchanged; `--design`, `--time-basis`, `--windows`.
- `code/20_tabular_baselines_v3.py`, `code/21_session_baseline_and_cluster_bootstrap_v3.py`, `code/22_h1_timestamp_basis_v3.py`, `code/14_validate_reanalysis_patched.py`, `code/run_alignments_v3.sh`.
- `alignment_v3_cross_regime/` — complete primary alignment (block manifest, segment allocation, unassigned intervals, event mapping, window features, targets, fused bank, bounded edges, flow labels, audits, build summary, SHA256SUMS).
- `alignment_v3_within_session_audit/`, `alignment_v3_cross_regime_snapshot_basis_audit/` — manifests, audits and summaries of the two secondary runs.
- `tabular_v3_*`, `baselines_v3_*`, `h1_v3/`, `validation_v2_rerun/`, run logs, `SHA256SUMS_package.txt`.

## 7. Next steps (staged, not run here)

1. Rerun notebooks 05 (Transformer) and 06 (R-GCN) on `alignment_v3_cross_regime` in the PyTorch environment: point them at `candidate_point_events_v3_with_context_and_nf3_primary.csv`, `event_nf3_targets_v3.csv`, `event_flow_edges_primary_bounded_v3.csv`, and `event_source_block_mapping_v3.csv` (splits are unchanged). Repeat over ≥ 5 seeds.
2. Rerun proxy ablations (script 16) and the standardized latency benchmark (script 18) on the v3 artifacts.
3. Add the session-indicator row to Table 4.8 and replace event-level intervals with bin-cluster intervals in Tables 4.4–4.5.
4. Proceed to the Tier 2 parametric coupling study using the cross-regime layout.

## Attribution

Analysis code in `code/` and this report were produced by Claude (Anthropic) at the student's direction on 19 September 2026, building on the student's v2 pipeline (`11_nf3_leakage_safe_alignment.py`, whose feature, target, and edge functions are imported unchanged). The student is responsible for reviewing, re-running, and being able to explain every design decision recorded here.
