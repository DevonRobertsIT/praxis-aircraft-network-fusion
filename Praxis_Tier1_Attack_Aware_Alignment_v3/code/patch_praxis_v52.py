#!/usr/bin/env python3
"""Apply the second external review's corrections to the praxis builder (v5.1 -> v5.2)."""
import json, re
from pathlib import Path
import pandas as pd

W = Path("/home/claude/work"); G = W / "v3_out/coupling_v31"
src = (W / "build_praxis_v51.js").read_text()
h5 = pd.read_csv(G / "h5_fusion_vs_best_unimodal_v31.csv")
ref = pd.read_csv(G / "h5_fusion_vs_cyber_reference_v31.csv")
bs = pd.read_csv(G / "block_size_sensitivity_v31.csv")
lat = h5[(h5.model == "random_forest") & (h5.fusion == "late")]
det = lat[lat.detected].sort_values(["sigma", "kappa"])
zero = lat[lat.sigma == 0]
k05, k075, k1 = (zero[zero.kappa == k].iloc[0] for k in (0.5, 0.75, 1.0))
n_conditions = len(lat)

def rep(old, new, count=1):
    global src
    assert old in src, f"anchor not found: {old[:90]}"
    src = src.replace(old, new, count)
def rx(pattern, new, flags=re.S):
    global src
    new_src, n = re.subn(pattern, lambda m: new, src, count=1, flags=flags)
    assert n == 1, f"regex not found: {pattern[:80]}"
    src = new_src

# ------------------------------------------------------------------ 1. fusion conclusion
det_rows = ", ".join(f'["{r.kappa:g}", "{int(r.sigma)}", "{r.gain_mean:+.3f} [{r.gain_min:+.3f}, {r.gain_max:+.3f}]", "{r.seeds_ci_gt0} / {r.seeds}", "{r.best_unimodal_by_validation}"]' for r in det.itertuples())
rx(r'  body\("Against the H5 comparator, fusion is .*?the injection mechanism does not leak\."\),',
   f'''  body("Against the validation-selected unimodal comparator, at zero jitter no tested coupling level satisfied the stated criterion for a reliable fusion advantage: positive mean gains were observed for late fusion at κ = 0.5 ({k05.gain_mean:+.3f}, {k05.seeds_ci_gt0} of {k05.seeds} seeds' intervals above zero) and κ = 0.75 ({k075.gain_mean:+.3f}, {k075.seeds_ci_gt0} of {k075.seeds}), but not with every seed's interval above zero, and at κ = 1 the physical model alone is best and late fusion trails it ({k1.gain_mean:+.3f}) while concatenation fusion falls further behind. H5 is therefore not supported at zero jitter. Under nonzero alignment jitter the picture differs: {len(det)} of the {n_conditions} tested (κ, σ) conditions meet the saved detection rule against the validation-selected comparator (Table 4.8b). Because H5 was pre-registered over κ at a single jitter level and no correction for the number of conditions was specified, these are reported as exploratory findings, not as acceptance of H5. Against the cyber-only model—the operational reference of an operator adding physical evidence to a cyber pipeline—late fusion is detectable from κ = 0.75 at zero jitter, concatenation fusion only at κ = 1.0, and linear fusion never within the window scale. The shuffled-score control at κ ∈ {{0.5, 1.0}} gives realized correlations of 0.000–0.004 and gains of −0.011 to +0.012 with no detection: the injection mechanism does not leak."),
  caption("Table", "Table 4.8b. Conditions under nonzero jitter in which late-fusion RF meets the detection rule against the validation-selected unimodal comparator (exploratory; mean over 5 seeds with seed range; seeds whose bin-cluster interval lies above zero)."),
  table([["κ", "σ (s)", "Late-fusion − best unimodal", "Seeds CI > 0", "Comparator selected"], {det_rows}], [900, 900, 3000, 1500, 3060]),
  new Paragraph({{ spacing: {{ after: 120 }} }}),''')
rx(r'    \["H5 fusion > best unimodal \(validation-selected\)".*?\],',
   f'''    ["H5 fusion > best unimodal (validation-selected)", "Zero jitter: no κ meets the rule (κ = 0.5 {k05.gain_mean:+.3f} and κ = 0.75 {k075.gain_mean:+.3f} positive but {k05.seeds_ci_gt0}/{k05.seeds} and {k075.seeds_ci_gt0}/{k075.seeds} seeds); {len(det)} nonzero-jitter conditions meet it (exploratory). Reference: late fusion − cyber-only detected from κ = 0.75 at σ = 0", "Not supported as pre-registered; exploratory support under jitter"],''')
rep('''Fused models never exceeded the best unimodal model; relative to the cyber-only model, late fusion became detectable at a rank correlation of about 0.75 with 27 independent held-out windows.''',
    f'''At zero jitter no coupling level produced a reliable fusion advantage over the validation-selected unimodal model, although positive mean gains appeared from a rank correlation of 0.5; under nonzero alignment jitter {len(det)} conditions met the detection rule (exploratory), and relative to the cyber-only model late fusion became detectable at a rank correlation of about 0.75 with 27 non-overlapping held-out windows.''')
rx(r'H5 is not supported: no fused model exceeded the validation-selected best unimodal model at any coupling strength—at high κ the physical model alone is best and late fusion trails it by [0-9.]+\.',
   f'''H5 is not supported as pre-registered: at zero jitter no coupling level met the criterion against the validation-selected unimodal comparator, although late fusion showed positive mean gains of {k05.gain_mean:+.3f} at κ = 0.5 and {k075.gain_mean:+.3f} at κ = 0.75, and at κ = 1 the physical model alone is best. Under nonzero jitter {len(det)} conditions met the rule; these are exploratory.''')
rep('''Third, the fusion question can be given content without coincident data by injecting coupling of known strength, provided the detection floor is stated; here it is ρ ≈ 0.75.''',
    '''Third, the fusion question can be given content without coincident data by injecting coupling of known strength, provided the comparator and the jitter condition are stated with the floor: against the cyber-only model at zero jitter the floor is ρ ≈ 0.75 for late fusion; against the validation-selected unimodal model no zero-jitter level met the rule.''')

# ------------------------------------------------------------------ 2. superseded neural jitter rows
rep('''  caption("Table", "Table 4.11. Transformer and R-GCN (original protocol) on the intensity target, mean of 3 seeds."),''',
    '''  caption("Table", "Table 4.11. Transformer and R-GCN (original training budget) on the intensity target, mean of 3 seeds. The σ = 120 s rows are SUPERSEDED development history: those cells were materialized under the earlier seeding in which jitter also changed the replay (507 of 1,015 test labels differ between the σ = 0 and σ = 120 s cells at κ = 0, seed 0), so they do not isolate a jitter effect and are not used for any conclusion."),''')
rep('''  caption("Figure", "Figure 4.3. Intensity target, original R-GCN protocol: neural and tabular models on identical cells (seeds 0–2) at σ = 0 (left) and σ = 120 s (right)."),''',
    '''  caption("Figure", "Figure 4.3. Intensity target, original R-GCN training budget: neural and tabular models on identical cells (seeds 0–2) at σ = 0 (left) and, superseded, at σ = 120 s (right; confounded replay, development history only)."),''')
rep('''Both neural models detect coupling (+0.10 to +0.18 from κ = 0 to 1) but exploit it less than late fusion (+0.25)."),''',
    '''Both neural models detect coupling (+0.10 to +0.18 from κ = 0 to 1) but exploit it less than late fusion (+0.25). The σ = 120 s cells of this table predate the jitter correction of Section 3.9 and compare different replays; neural jitter effects await re-materialized cells (Appendix A) and no claim rests on them."),''')

# ------------------------------------------------------------------ 3. XAI limitation, consistently
rep('''Explanations are validated for fidelity, stability, provenance, and attribution tracking, not for operator effectiveness, which would require a human-subjects study outside the praxis boundary."),''',
    '''Explanations were validated for fidelity, stability, and provenance on the v2 checkpoints and, for the tabular models, by attribution tracking against injected coupling; the corresponding validation of the final trained Transformer and R-GCN checkpoints is staged but not complete, and no validation involved operators, which would require a human-subjects study outside the praxis boundary."),''')
rep('''This is the first ground-truth test of the explanation layer in the study."),''',
    '''This is the first ground-truth test of the explanation layer in the study, and it applies to the tabular random-forest explanations. Fidelity, stability, and provenance were executed on the v2 checkpoints; for the final trained checkpoints reported in Section 4.7 the same package is staged (script 39) and, for each model supporting an explainability claim, must retain the checkpoint and preprocessing identifiers, the prediction-reproduction check, fidelity against random evidence removal, stability under the specified perturbations, provenance checks, and explanation timing. Until that runs, explainability claims for the final neural models are limited to what Section 4.8 states."),''')
rep('''"A parametric coupling-injection protocol that turns an unanswerable fusion question into a measurable one, with attribution tracking as a ground-truth test of the explanation layer and a stated detection floor.",''',
    '''"A parametric coupling-injection protocol that turns an unanswerable fusion question into a measurable one, with attribution tracking as a ground-truth test of the tabular explanation layer and detection floors stated per comparator and jitter condition.",''')

# ------------------------------------------------------------------ 4. statistical and preprocessing qualifications
b0 = bs[(bs.kappa == 0)]; w = {r.block: r.auprc_ci_width for r in b0.itertuples()}
rep('''Bins are only approximately independent—windows of events in adjacent bins can share flows that straddle the boundary, and replay cycles revisit the same bins—so the bin bootstrap under-corrects slightly and the intervals reported should be read as lower bounds on uncertainty.''',
    f'''Non-overlapping bins are not necessarily independent observations: windows of events in adjacent bins can share flows that straddle the boundary, and replay cycles revisit the same bins. A block-size sensitivity analysis on the saved cyber-LR predictions (κ = 0, σ = 0, five seeds) gave mean 95% interval widths of {w['event-level']:.3f} at the event level, {w['240 s (1 bin)']:.3f} with 240 s bins, {w['480 s (2 bins)']:.3f} with 480 s blocks, and {w['960 s (4 bins)']:.3f} with 960 s blocks: widening the block beyond 240 s does not widen the interval, which supports 240 s as the resampling unit, while the event-level interval is four to five times too narrow. With 7–27 clusters the bootstrap is coarse, and bin intervals are reported as approximate. Two kinds of interval appear in this praxis and answer different questions: bin-cluster intervals describe sampling uncertainty within one replay and one seed; seed-level t-intervals describe variability across replays and initializations. They are never used interchangeably.''')
rep('''All preprocessing—median imputation with missingness indicators, one-hot encoding, and standardization for linear models—is fitted on the training partition only. Thresholds for F1 are chosen on validation."),''',
    '''Tabular and Transformer preprocessing—median imputation with missingness indicators, one-hot encoding, standardization, and the Transformer's time-gap statistics—is fitted on the training partition only, and thresholds for F1 are chosen on validation. One exception existed in the reported R-GCN runs: the scale that normalizes the flow-to-event overlap weight was the maximum overlap across all selected edges rather than across training edges. For the regime-target runs the training-partition maximum (120.297 s) is the global maximum, so the scale and all model inputs are identical under a training-only rule; for the intensity cells the global maximum (119.966 s) came from the test partition and exceeds the training maximum (119.647 s), which changes the overlap weight by at most 0.0003 on a scale of 0.5–1.0. The patched notebook (v3.2) derives the scale from training edges; the intensity cells are scheduled for regeneration under it, and the results reported here carry this qualification."),''')
rep('''**Internal validity.** Source-level partition disjointness, embargoes, insets, training-only preprocessing, target-derived-column exclusion, and shortcut baselines with margins address leakage and shortcuts; the training-protocol correction addresses the largest internal threat found, which was undertraining of one comparator.''',
    '''**Internal validity.** Source-level partition disjointness, embargoes, insets, training-only preprocessing (with the R-GCN overlap-scale exception quantified in Section 3.8.1), target-derived-column exclusion, and shortcut baselines with margins address leakage and shortcuts; the training-budget correction addresses the largest internal threat found, which was an insufficiently trained comparator.''')
for old, new in [("independent 240 s source bins", "non-overlapping 240 s source bins"), ("independent 240 s bins", "non-overlapping 240 s bins"), ("independent 240 s source windows", "non-overlapping 240 s source windows"),
                 ("Independent 240 s bins in test (pos / neg)", "Non-overlapping 240 s bins in test (pos / neg)"), ("independent bins", "resampling bins"), ("independent held-out windows", "non-overlapping held-out windows"),
                 ("independent source bins", "non-overlapping source bins"), ('["Independent bin", "A non-overlapping 240 s interval of source time; events in the same bin share most of their flows and are treated as one cluster."]', '["Resampling bin", "A non-overlapping 240 s interval of source time; events in the same bin share most of their flows and are resampled together. Bins are approximately, not strictly, independent."]'),
                 ("effective samples of 60–100 where nominal samples were 1,828", "effective samples of roughly 60–100 where nominal samples were 1,828")]:
    src = src.replace(old, new)
src = re.sub(r"(\d+) independent (240 s )?bins", lambda m: f"{m.group(1)} non-overlapping {m.group(2) or ''}bins", src)

# ------------------------------------------------------------------ 5. consistency items
rep('''What matters is alignment error at the scale of the window: at 120 s the best cyber model loses 0.02 AUPRC and at 300 s it loses 0.11, and coupled physical evidence compensates for part of that loss.''',
    '''What matters is alignment error at the scale of the window: in the paired comparison on frozen labels the best cyber model loses 0.033 AUPRC at 120 s (the pre-registered rule, which requires every seed's interval above zero, is not met at this level) and 0.129 at 300 s (met), and coupled physical evidence compensates for part of that loss.''')
rep('''    ["H3′ lag/relation weighting ≥ 0.05", "+0.020 and +0.010; unweighted graph equals or exceeds full at κ = 0", "Not supported"],''',
    '''    ["H3′ lag/relation weighting ≥ 0.05", "+0.020 and +0.010 over ten seeds (full 0.815 vs ablation 0.795 at κ = 0); intervals include zero", "Not supported"],''')
rep('''    ["Original H2, H3 (regime target)", "Ceiling task; comparator untrained", "Retired"]], [3000, 4200, 2160]),''',
    '''    ["Original H2, H3 (regime target)", "Ceiling task; comparator insufficiently trained under the original budget", "Retired"]], [3000, 4200, 2160]),''')
for old, new in [("and a final audit revealed that the relational model in all earlier comparisons had never been trained.", "and a final audit revealed that the relational model in all earlier comparisons had been trained for at most 50 gradient steps, far short of convergence."),
                 ("All R-GCN results in earlier versions of this praxis therefore describe an untrained model.", "All R-GCN results in earlier versions of this praxis therefore describe a model that was insufficiently trained under the original budget."),
                 ("**Every R-GCN number in the praxis before this ablation came from an untrained model.**", "**Every R-GCN number in the praxis before this ablation came from a model insufficiently trained under the original budget.**"),
                 ("rather than capability, and no fusion benefit can exist when the cyber signal is already perfect.", "rather than capability, and no fusion benefit can exist when the cyber signal is already perfect."),
                 ("measures failure modes—here, an untrained comparator—rather than capability", "measures failure modes—here, an insufficiently trained comparator—rather than capability"),
                 ("came from allowing one model to train.", "came from allowing one model to train to convergence."),
                 ("that a model comparison is only as valid as the weakest training protocol in it.", "that a model comparison is only as valid as the weakest training protocol in it."),
                 ("came from a model that had taken fewer than 50 gradient steps", "came from a model trained for fewer than 50 gradient steps"),
                 ("was an artifact of a training cap of 50 gradient steps.", "was an artifact of a training budget capped at 50 gradient steps."),
                 ("and three seeds for the untrained comparison;", "and three seeds for the comparison under the original training budget;"),
                 ("and the R-GCN of all earlier comparisons was untrained.", "and the R-GCN of all earlier comparisons was insufficiently trained under its original budget."),
                 ("The earlier verdict that the R-GCN was inferior—reported in every prior version of this praxis—was an artifact of a training cap of 50 gradient steps.", "The earlier verdict that the R-GCN was inferior—reported in every prior version of this praxis—was an artifact of a training budget capped at 50 gradient steps."),
                 ('["Late fusion", "The mean of the physical-only and cyber-only model probabilities, used as a reference fusion that cannot dilute either modality."]', '["Late fusion", "The mean of the physical-only and cyber-only model probabilities, used as a reference fusion that does not concatenate the two feature spaces; it carries no performance guarantee."]'),
                 ("A **late-fusion** reference—the mean of the physical-only and cyber-only probabilities—is added because concatenation can dilute a small set of informative features among many uninformative ones; late fusion cannot.", "A **late-fusion** reference—the mean of the physical-only and cyber-only probabilities—is added because concatenation can dilute a small set of informative features among many uninformative ones; late fusion does not concatenate feature spaces and therefore serves as a reference for that specific failure mode, without any guarantee of better performance.")]:
    src = src.replace(old, new)
assert "untrained" not in src.replace("insufficiently trained", ""), [m.start() for m in re.finditer("untrained", src)]

# Chapter 2 latency reconciliation (student's text): append to the last paragraph of 2.6.11
ch2 = json.loads((W / "chapter2_blocks.json").read_text())
idx = next(i for i, b in enumerate(ch2) if b["t"] == "h3" and b["x"].startswith("2.6.11"))
j = idx + 1
while j + 1 < len(ch2) and ch2[j + 1]["t"] == "p":
    j += 1
if not ch2[j]["x"].endswith("(Section 4.9)."):
    ch2[j]["x"] += " In this study the latency reported for every model family is inference on prepared inputs—precomputed sequences, local subgraphs, or assembled feature vectors—and excludes feature aggregation and alignment; a raw-input-to-score measurement would add those stages and is outside the reported boundary (Section 4.9)."
(W / "chapter2_blocks.json").write_text(json.dumps(ch2, indent=0))

# ------------------------------------------------------------------ 6. reproducibility
rep('''Hardware is disclosed: the alignment, tabular, and coupling analyses ran on a single-core Linux container with 3 GB of memory; all neural-network runs ran on the author's Windows workstation."),''',
    '''The companion package additionally holds the per-event test predictions of the corrected coupling grid (with bin identifiers, so every paired bootstrap can be reconstructed), the final Transformer and R-GCN checkpoints with their serialized preprocessing state for the regime root and every coupling cell, and the collection script that gathers them. Hardware is disclosed: the alignment, tabular, and coupling analyses ran on a single-core Linux container with 3 GB of memory; all neural-network runs ran on the author's Windows workstation."),''')
rep('''  ["Review corrections (3.9, 4.6.3, 4.6.4)", "40_review_corrections_v31.py on the v3.1 grid", "h1prime_paired_jitter_loss_aggregate_v31.csv; h5_fusion_vs_best_unimodal_v31.csv; REVIEW_CORRECTIONS_v31.md"]],''',
    '''  ["Review corrections (3.9, 4.6.3, 4.6.4)", "40_review_corrections_v31.py on the v3.1 grid", "h1prime_paired_jitter_loss_aggregate_v31.csv; h5_fusion_vs_best_unimodal_v31.csv; coupling_grid_test_predictions_v31.csv.gz; block_size_sensitivity_v31.csv; REVIEW_CORRECTIONS_v31.md"],
  ["Final checkpoints and preprocessors (3.13)", "collect_final_checkpoints_v3.ps1", "transformer_v2/models/fused_temporal_transformer.pt and rgcn_trained_v3/models/*.pt per cell and per regime seed, with rgcn_config, thresholds and reference JSONs"],
  ["Pending: corrected σ = 120 s neural cells; final-checkpoint XAI; R-GCN overlap-scale rerun", "35_ (v3.1 seeding), 39_, notebook 06 v3.2", "to be added on completion"]],''')

src = src.replace("Chapters 1–5, Version 5.1 (revised results; external-review corrections applied)", "Chapters 1–5, Version 5.2 (second external-review corrections applied)")
src = src.replace("Roberts_Praxis_Chapters_1_5_v5_1.docx", "Roberts_Praxis_Chapters_1_5_v5_2.docx")
(W / "build_praxis_v52.js").write_text(src)
print("v5.2 builder written; detected jitter conditions:", len(det), "| block widths:", {k: round(v, 3) for k, v in w.items()})
