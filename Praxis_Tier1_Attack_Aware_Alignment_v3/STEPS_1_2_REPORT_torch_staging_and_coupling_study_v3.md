# Steps 1–2 — Torch rerun staging and the parametric coupling study (v3)

**Run date:** 19 September 2026 · Linux container, 1 CPU, 3 GB, scikit-learn 1.8.0 · no PyTorch (Transformer / R-GCN staged, not run)

## Step 1 — closing Tier 1

| Item | Status |
|---|---|
| Tabular closure on the cross-regime alignment (Tables 4.4/4.5/4.8 material with bin-cluster intervals and the session row) | Done — `tabular_v3_cross_regime/`, `baselines_v3_cross_regime/` |
| Transformer (05) and R-GCN (06) reruns, 5 seeds | **Staged** — `torch_rerun_staging/32_stage_torch_reruns_v3.py` builds a root in which both notebooks' hard-coded v2 paths resolve to the v3 artifacts, writes the v3 split/target file in the v4 schema (7,094 positives), and emits `run_torch_reruns_v3.sh` (seeds 42–46 via `TT_RANDOM_STATE` / `RGCN_RANDOM_STATE`, unmodified notebooks through `run_notebook_code.py`). Run the staging script with `--copy` on the PyTorch machine, then the runner. |
| Proxy ablations (16), standardized latency (18) | Not rerun: the ablations are superseded by the target redefinition below; latency needs the torch models. |

Expectation for the torch reruns on the *regime* target: Transformer ≈ 1.0, R-GCN below it, for the input-budget reason already documented. Their real use is on the intensity target and coupling grid below, which is where H2′/H3′ live.

## Step 2 — design as executed

- **Target.** Attack intensity within attack-regime windows: `y = 1` if the true position's `nf3_attack_flow_fraction` ≥ the median over training events' true positions (threshold re-derived per cell; test prevalence 0.53–0.61). Events in benign blocks are out of scope.
- **Position bank.** 1,584 candidate centers every 30 s across the attack blocks (train 1,148 / validation 224 / test 212), each aggregated with the same v2 feature and target functions the alignments use. Intensity p10/p50/p90 ≈ 0.056 / 0.088 / 0.126 (train), 0.055 / 0.096 / 0.133 (test).
- **Anomaly score.** Mean over {altitude, ground speed, distance} of the train-only z-scored |deviation from the aircraft's prior mean|; first-observed aircraft (16.7%) use the train-only |z| of the raw value.
- **Coupling.** Within each partition's attack blocks, with probability κ an event takes the position whose intensity rank equals its anomaly rank; otherwise a uniform random position. Blocks, embargoes, insets and partition ownership untouched. Realized train Spearman(score, intensity) = 0.002 / 0.102 / 0.249 / 0.499 / 0.748 / 1.000 for κ = 0 / 0.1 / 0.25 / 0.5 / 0.75 / 1.
- **Jitter.** Observed cyber window = true position shifted by round(N(0, σ)/30 s) grid steps, clipped inside the block; the target stays at the true position; physical features are never jittered.
- **Models.** LR and RF as in v4 on physical / cyber / concat-fused sets, plus **late fusion** = mean of the two unimodal probabilities (no extra fit). 5 seeds × 6 κ × 4 σ = 120 cells + 10 shuffled-score control cells; 300-rep bin-cluster bootstrap over 27 independent 240 s test bins.
- **Detection rule.** A gain counts as detected when its 5-seed mean ≥ 0.02 and every seed's bin-cluster interval excludes zero.

## Step 2 — results

### AUPRC at σ = 0 (mean of 5 seeds)

| Model | κ = 0 | 0.1 | 0.25 | 0.5 | 0.75 | 1.0 |
|---|---|---|---|---|---|---|
| Cyber LR | 0.884 | 0.879 | 0.872 | 0.890 | 0.884 | 0.884 |
| Cyber RF | 0.706 | 0.700 | 0.719 | 0.699 | 0.710 | 0.708 |
| Physical LR | 0.569 | 0.567 | 0.616 | 0.628 | 0.691 | 0.745 |
| Physical RF | 0.577 | 0.572 | 0.643 | 0.737 | 0.853 | **0.969** |
| Concat-fused LR | 0.880 | 0.872 | 0.866 | 0.885 | 0.882 | 0.891 |
| Concat-fused RF | 0.700 | 0.697 | 0.715 | 0.707 | 0.750 | 0.836 |
| Late-fusion LR | 0.880 | 0.870 | 0.868 | 0.880 | 0.885 | 0.895 |
| Late-fusion RF | 0.705 | 0.704 | 0.747 | 0.786 | 0.879 | **0.951** |

Cyber performance is flat in κ, as it must be (coupling changes which aircraft event sits at a position, not the cyber window). Bin-cluster intervals at κ = 0: cyber LR [0.75, 0.96], cyber RF [0.51, 0.88], physical ≈ [0.40, 0.74].

### Fusion gain over cyber-only (σ = 0, RF)

| κ | Concat-fused − cyber | Late-fusion − cyber | Detected (late) |
|---|---|---|---|
| 0 | −0.007 [seed range −0.030, +0.006] | −0.001 | no |
| 0.1 | −0.003 | +0.003 | no |
| 0.25 | −0.004 | +0.028 (CI low −0.006) | no |
| 0.5 | +0.007 | +0.086 (4 of 5 seeds' CI > 0) | borderline |
| 0.75 | +0.040 | **+0.169** (all seeds) | **yes** |
| 1.0 | +0.128 | **+0.242** (all seeds) | **yes** |

Minimum detectable κ (late fusion vs. cyber, RF): 0.75 at σ = 0, 0.5 at σ = 60–300 s. Concat fusion: 1.0 at σ = 0, 0.75 at σ ≥ 60 s. LR: never at σ ≤ 120 s.

### Alignment-jitter sensitivity (H1′)

Cyber-only loss vs. σ = 0 at κ = 0: LR −0.018 / **+0.023** / **+0.109** at σ = 60 / 120 / 300 s; RF +0.007 / −0.003 / −0.040 (the RF is too weak to be hurt). Late-fusion LR gain over cyber at κ = 1 grows with σ: +0.011 / +0.010 / +0.019 / **+0.061** — coupled physical evidence partially compensates for cyber misalignment.

### Attribution tracking (σ = 0, RF importance share)

| κ | Fused RF: physical-block share | Fused RF: score-feature share | Physical RF: score-feature share |
|---|---|---|---|
| 0 | 0.012 | 0.005 | 0.402 |
| 0.5 | 0.040 | 0.018 | 0.520 |
| 0.75 | 0.133 | 0.064 | 0.588 |
| 1.0 | 0.327 | 0.180 | 0.637 |

### Negative control (shuffled anomaly score)

ρ = 0.000–0.004; gains over cyber −0.011 to +0.012, none detected. The injection mechanism does not leak.

## What the praxis can now say

1. **An honest null exists.** On a non-ceiling target (cyber LR 0.88, RF 0.71), fusion adds nothing when nothing is coupled — and the study shows it *would* have detected coupling above ρ ≈ 0.75 (late fusion, ρ ≈ 0.5 marginally). "Fusion did not help" becomes a calibrated statement with a stated detection floor, not an artifact of the design.
2. **The concatenation architecture, not fusion, is the bottleneck.** At κ = 1 the physical RF alone reaches 0.969 while the concat-fused RF reaches 0.836 — concatenation *destroys* 0.13 of available signal (sqrt feature sub-sampling among 89 columns dilutes 6 informative ones; LR cannot represent |deviation| at all). Late fusion recovers 0.951. Chapter 3 needs a late-fusion or stacked baseline, and H2′/H3′ must be judged against it.
3. **Alignment fidelity matters at the window scale.** The better cyber model loses ≥ 0.02 AUPRC once alignment error reaches the window half-width (120 s) and 0.11 at 300 s. This is the empirical H1 the original hypothesis was reaching for.
4. **The XAI layer now has a passed ground-truth test.** Physical-block attribution rises monotonically with κ (0.012 → 0.327) and stays flat under the shuffled control.
5. **Sample size is the binding constraint.** With 27 independent held-out bins, coupling weaker than ρ ≈ 0.5 is invisible to every tabular model. Report this floor explicitly.

## Restated hypotheses and status

| | Statement | Status (tabular level) |
|---|---|---|
| H1′ | Alignment error ≥ 120 s reduces cyber-only held-out AUPRC by ≥ 0.02 | **Supported** for LR (0.023 at 120 s, 0.109 at 300 s); not for RF |
| H2′ | At κ ≥ 0.5 the R-GCN exceeds the Transformer by ≥ 0.03 AUPRC on the intensity target | Awaits torch reruns (`torch_rerun_staging/`) |
| H3′ | Lag/relation evidence adds ≥ 0.05 AUPRC at κ ≥ 0.5 under 500 ms p95 | Awaits torch reruns |
| Detection floor | Fusion gain is detectable at ρ ≥ 0.75 (late fusion; concat ρ = 1.0) with 27 independent test bins | Established |

## Files

`step2_coupling_study/`: `30_position_bank_v3.py`, `31_coupling_injection_grid_v3.py`, `33_summarize_coupling_grid_v3.py`, `position_bank_cross_regime_v3.csv`, `coupling_grid_results_v3.csv` (1,040 rows), `coupling_grid_cells_v3.csv`, `coupling_auprc_surface_v3.csv`, `coupling_fusion_gains_v3.csv`, `coupling_minimum_detectable_kappa_v3.csv`, `coupling_h1prime_alignment_sensitivity_v3.csv`, `coupling_attribution_vs_kappa_v3.csv`, `coupling_negative_control_v3.csv`, `coupling_realized_spearman_v3.csv`, `COUPLING_GRID_SUMMARY_v3.md`, `coupling_grid_figure_v3.png`, `grid.log`.
`torch_rerun_staging/`: `32_stage_torch_reruns_v3.py`, `run_torch_reruns_v3.sh`, `model_split_assignments_v4.csv` (v3 targets, v4 schema), `baseline_model_results_v4.csv` (v3 tabular results in the v4 schema for the notebooks' comparison tables).

## Attribution

Code and this report were produced by Claude (Anthropic) at the student's direction on 19 September 2026, on top of the student's v2 pipeline; feature, target and edge functions are the student's v2 code imported unchanged. The student is responsible for reviewing, re-running, and explaining every design decision.
