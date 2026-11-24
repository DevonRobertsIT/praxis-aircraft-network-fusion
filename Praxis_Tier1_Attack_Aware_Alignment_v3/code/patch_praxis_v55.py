#!/usr/bin/env python3
"""v5.4 -> v5.5: freeze the R-GCN intensity results on the v3.3 reruns (training-only overlap scale)."""
import re, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

W = Path("/home/claude/work"); A = W / "v3_out/rgcn_v33_aggregate"
src = (W / "build_praxis_v54.js").read_text()
misses = []
def rep(old, new, count=1):
    global src
    if old not in src: misses.append(old[:100]); return
    src = src.replace(old, new, count)
def rx(pattern, new):
    global src
    s2, n = re.subn(pattern, lambda m: new, src, count=1, flags=re.S)
    if n != 1: misses.append("REGEX " + pattern[:90]); return
    src = s2
def sg(x, d=3): return (f"{x:+.{d}f}").replace("-", "−")
def ci(lo, hi, d=3): return f"[{sg(lo, d)}, {sg(hi, d)}]"
def p(x): return f"{x:.3f}" if x < 0.1 else f"{x:.2f}"

r33 = pd.read_csv(A / "rgcn_training_budget_ablation_runs_v3.csv")
inf = pd.read_csv(A / "seed_level_inference_v33.csv")
reg = pd.read_csv(W / "v3_out/rgcn_ablation_aggregate/rgcn_training_budget_ablation_runs_v3.csv"); reg = reg[reg.target.eq("regime")]
def I(k, q): return inf[(inf.kappa == k) & inf.question.str.startswith(q)].iloc[0]
h2_0, h2_1, h4_0, h4_1 = I(0.0, "H2"), I(1.0, "H2"), I(0.0, "H4"), I(1.0, "H4")
emin = int(min(r33.trained_full_epochs.min(), reg.trained_full_epochs.min())); emax = int(max(r33.trained_full_epochs.max(), reg.trained_full_epochs.max()))
pc_min, pc_max = r33.trained_full_p95_ms.min(), r33.trained_full_p95_ms.max(); pr_min, pr_max = reg.trained_full_p95_ms.min(), reg.trained_full_p95_ms.max()

# ---------------- figures
fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
cells3 = r33[r33.seed <= 2]
for k, (d, title, yl) in enumerate([(reg.sort_values("seed"), "Regime target, 5 seeds", (0.93, 1.005)), (cells3[cells3.kappa == 0].sort_values("seed"), "Intensity target, κ = 0, 3 seeds", (0.45, 1.0)), (cells3[cells3.kappa == 1].sort_values("seed"), "Intensity target, κ = 1, 3 seeds", (0.45, 1.0))]):
    x = np.arange(len(d))
    for j, (col, c, lab) in enumerate((("auprc_transformer", "tab:green", "Transformer"), ("auprc_orig_full", "lightgray", "R-GCN full, original budget"), ("auprc_trained_full", "tab:orange", "R-GCN full, trained"), ("auprc_trained_ablation", "tab:red", "R-GCN ablation, trained"), ("auprc_trained_event_only", "tab:purple", "event-only MLP, trained"))):
        ax[k].errorbar(x + (j - 2) * 0.12, d[col], yerr=[d[col] - d[col + "_ci_low"], d[col + "_ci_high"] - d[col]], fmt="o", ms=4, color=c, capsize=2, label=lab)
    ax[k].set(xticks=x, xticklabels=[f"s{int(s)}" for s in d.seed], title=title, ylim=yl, ylabel="held-out AUPRC (bin-cluster 95% CI)" if k == 0 else "")
ax[0].legend(fontsize=7, loc="lower right"); fig.tight_layout(); fig.savefig(A / "fig_4_4_v33.png", dpi=150); plt.close(fig)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for i, k in enumerate([0.0, 1.0]):
    g = r33[r33.kappa == k].sort_values("seed"); data = [g.auprc_transformer, g.auprc_trained_event_only, g.auprc_trained_ablation, g.auprc_trained_full]
    ax[i].boxplot(data, tick_labels=["Transformer", "event-only MLP", "R-GCN ablation", "R-GCN full"], showmeans=True)
    for j, d in enumerate(data): ax[i].scatter(np.full(len(d), j + 1) + np.linspace(-.08, .08, len(d)), d, s=14, color="k", alpha=.6)
    a, b = I(k, "H2"), I(k, "H4")
    ax[i].set(title=f"Intensity target, κ = {k:g}, 10 seeds (trained models)", ylabel="held-out AUPRC" if i == 0 else "")
    ax[i].text(0.02, 0.03, f"R-GCN full − Transformer: {a['mean']:+.3f} [{a.lo:+.3f}, {a.hi:+.3f}]\nfull − event-only: {b['mean']:+.3f} [{b.lo:+.3f}, {b.hi:+.3f}]", transform=ax[i].transAxes, fontsize=8, va="bottom", bbox=dict(fc="white", alpha=.8))
fig.tight_layout(); fig.savefig(A / "fig_4_5_v33.png", dpi=150); plt.close(fig)
rep("fig44: `${W}/v3_out/rgcn_ablation_aggregate/rgcn_ablation_figure_v3.png`", "fig44: `${W}/v3_out/rgcn_v33_aggregate/fig_4_4_v33.png`")
rep("fig45: `${W}/v3_out/rgcn_seeds_aggregate/rgcn_seeds_figure_v3.png`", "fig45: `${W}/v3_out/rgcn_v33_aggregate/fig_4_5_v33.png`")
rep("per-seed AUPRC with bin-cluster intervals for the Transformer, the original and trained R-GCN, the trained ablation, and the event-only MLP.", "per-seed AUPRC with bin-cluster intervals for the Transformer, the R-GCN under its original budget and trained, the trained ablation, and the event-only MLP. Intensity-target R-GCN runs use the training-only overlap scale.")
rep("with seed-level intervals for the relational-versus-sequential and relational-value differences.", "with seed-level intervals for the relational-versus-sequential and relational-value differences. R-GCN runs use the training-only overlap scale.")

# ---------------- Table 4.12 intensity rows
for k, old_full, old_budget in ((0.0, "0.7865", "+0.206"), (1.0, "0.9082", "+0.209")):
    g = cells3[cells3.kappa == k]
    old_row_prefix = f'["Intensity", "{int(k)}", "3", '
    i0 = src.find(old_row_prefix); i1 = src.find("]", i0)
    if i0 < 0: misses.append(f"T4.12 row k={k}"); continue
    new_row = f'["Intensity", "{int(k)}", "3", "{g.auprc_transformer.mean():.4f}", "{g.auprc_orig_full.mean():.4f}", "{g.auprc_trained_full.mean():.4f}", "{g.auprc_trained_ablation.mean():.4f}", "{g.auprc_trained_event_only.mean():.4f}", "{g.budget_trained_full_minus_orig_full.mean():+.3f} ({int((g.budget_trained_full_minus_orig_full_ci_low > 0).sum())} / {int((g.budget_trained_full_minus_orig_full_ci_high < 0).sum())})"'
    src = src[:i0] + new_row + src[i1:]
# ---------------- Table 4.13
for k in (0.0, 1.0):
    g = r33[r33.kappa == k]; ms = lambda c: f"{g[c].mean():.3f} ± {g[c].std():.3f}"
    rx(r'\["' + str(int(k)) + r'", "[0-9.]+ ± [0-9.]+", "\*\*[0-9.]+ ± [0-9.]+\*\*", "[0-9.]+ ± [0-9.]+", "[0-9.]+ ± [0-9.]+",',
       f'["{int(k)}", "{ms("auprc_transformer")}", "**{ms("auprc_trained_full")}**", "{ms("auprc_trained_ablation")}", "{ms("auprc_trained_event_only")}",')
rep("R-GCN values are provisional pending the training-only overlap-normalization rerun (Section 3.8.1); Transformer and tabular values are final.", "R-GCN values from the training-only overlap-scale runs (Section 3.8.1).")
# ---------------- Table 4.14 (regenerated)
lab = {"H2": "H2′: R-GCN full − Transformer", "H4": "H4: full − event-only", "H3": "H3′: full − ablation", "ev": "Event-only − Transformer"}
rows = []
for k in (0.0, 1.0):
    for q in ("H2", "H4", "H3", "ev"):
        r = I(k, q if q != "ev" else "event-only"); d = 4 if (q == "H2" and k == 1.0) else 3
        mean = sg(r["mean"], d); mean = f"**{mean}**" if (q == "H2" or (q == "H4" and k == 0.0)) else mean
        pm = f"{p(r.p_margin)} ({'0.03' if q == 'H2' else '0.05'})" if q in ("H2", "H3") else "—"
        rows.append(f'["{int(k)}", "{lab[q]}", "{mean}", "{ci(r.lo, r.hi)}", "{int(r.pos)} / 10", "{int(r.ci_gt0)} / {int(r.ci_lt0)}", "{p(r.p0)}", "{pm}"]')
rx(r'(table\(\[\["κ", "Question", "Mean", "95% t-interval", "Seeds > 0", "Bin-CIs > 0 / < 0", "p \(vs 0\)", "p \(vs margin\)"\],).*?(\], \[400, 2300, 800, 1500, 900, 1200, 900, 1360\], \{ size: 16 \}\),)',
   'table([["κ", "Question", "Mean", "95% t-interval", "Seeds > 0", "Bin-CIs > 0 / < 0", "p (vs 0)", "p (vs margin)"],\n    ' + ", ".join(rows) + '], [400, 2300, 800, 1500, 900, 1200, 900, 1360], { size: 16 }),')
rep("Rows involving the R-GCN are provisional pending the normalization rerun (Section 3.8.1).", "R-GCN rows from the training-only overlap-scale runs (Section 3.8.1).")
# ---------------- 4.7.3 / abstract / 5.x numbers
H2a, H2b = f"{h2_0['mean']:+.3f}", f"{h2_1['mean']:+.4f}"
rep("at κ = 0 the mean (+0.065) clears the margin", f"at κ = 0 the mean ({H2a}) clears the margin")
rep("at κ = 1 the mean (+0.029) falls just below the margin", f"at κ = 1 the mean ({H2b}) falls just below the margin")
rep("(t-intervals [+0.011, +0.120] and [+0.005, +0.054]; p = 0.025 and 0.023)", f"(t-intervals {ci(h2_0.lo, h2_0.hi)} and {ci(h2_1.lo, h2_1.hi)}; p = {p(h2_0.p0)} and {p(h2_1.p0)})")
rep("restoring the relations yields +0.064 (p = 0.008)", f"restoring the relations yields {h4_0['mean']:+.3f} (p = {p(h4_0.p0)})")
rep("The trained R-GCN is also the most seed-stable model on the graded target (SD 0.011 at κ = 1).", f"The trained graph variants are the most seed-stable models on the graded target (SD {r33[r33.kappa==1].auprc_trained_full.std():.3f} for the full and {r33[r33.kappa==1].auprc_trained_ablation.std():.3f} for the ablated graph at κ = 1, against {r33[r33.kappa==1].auprc_transformer.std():.3f} for the Transformer).")
rep("of 0.065 (seed-level 95% interval 0.011 to 0.120, ten seeds)", f"of {h2_0['mean']:.3f} (seed-level 95% interval {h2_0.lo:.3f} to {h2_0.hi:.3f}, ten seeds)")
rep('"+0.065 (5/10 CIs > 0) at κ = 0; +0.029 (5/10) at κ = 1; three seeds per level favour the Transformer"', f'"{H2a} ({int(h2_0.ci_gt0)}/10 CIs > 0) at κ = 0; {H2b} ({int(h2_1.ci_gt0)}/10) at κ = 1; three seeds per level favour the Transformer"')
rep('"+0.064 [0.022, 0.107] at κ = 0 (p = 0.008); +0.021 [−0.002, 0.043] at κ = 1"', f'"{h4_0["mean"]:+.3f} [{h4_0.lo:.3f}, {h4_0.hi:.3f}] at κ = 0 (p = {p(h4_0.p0)}); {h4_1["mean"]:+.3f} {ci(h4_1.lo, h4_1.hi)} at κ = 1"'.replace("[+", "[").replace(", +", ", "))
src = src.replace("p ≈ 0.02)", f"p = {p(h2_0.p0)} and {p(h2_1.p0)})")
rep("the κ = 1 mean (0.029) is below the 0.03 margin", f"the κ = 1 mean ({h2_1['mean']:.4f}) is below the 0.03 margin")
rep("0.01 to 0.12 at κ = 0 and 0.01 to 0.05 at κ = 1", f"{h2_0.lo:.3f} to {h2_0.hi:.3f} at κ = 0 and {h2_1.lo:.3f} to {h2_1.hi:.3f} at κ = 1")
rep("add 0.064 AUPRC where no physical coupling exists (p = 0.008)", f"add {h4_0['mean']:.3f} AUPRC where no physical coupling exists (p = {p(h4_0.p0)})")
rep("; the pre-specified margin was not established, and the R-GCN figures await the normalization rerun.", "; the pre-specified margin was not established.")
rep("trained runs stopped after 272–536 epochs.", f"trained full-variant runs stopped after {emin}–{emax} epochs.")
rep("Trained runs stopped after 272–536 epochs; local p95 latency remained at 3–11 ms.", f"Trained full-variant runs stopped after {emin}–{emax} epochs; local p95 latency on prepared subgraphs remained at {min(pr_min, pc_min):.0f}–{max(pr_max, pc_max):.0f} ms.")
rep("remained at 3–11 ms p95 across all eleven ablation runs and 3.1–3.6 ms across the twenty ten-seed cells", f"remained at {pr_min:.1f}–{pr_max:.1f} ms p95 across the five regime-target runs and {pc_min:.1f}–{pc_max:.1f} ms across the twenty intensity cells")
# ---------------- Table 4.16 and 4.8 text
X = {}
for kk in ("k0", "k1"):
    f = pd.read_csv(W / f"v33/xai_{kk}_v33/xai_fidelity_results_v2.csv"); f = f[(f.model == "R-GCN") & f.perturbation_type.eq("event_feature_zeroing")]
    g = f.groupby("evidence_set").agg(drop=("auprc_drop", "mean"), pch=("mean_absolute_probability_change", "mean"))
    lat = pd.read_csv(W / f"v33/xai_{kk}_v33/xai_explanation_latency_v2.csv").set_index("model").latency_p95_ms
    X[kk] = (g, lat)
g0, l0 = X["k0"]; g1, l1 = X["k1"]
rep('"1.00 / 1.00", "19.5 ms"]', f'"1.00 / 1.00", "{l0["Temporal Transformer"]:.1f} ms"]')
rep('["Intensity κ = 0", "R-GCN (event features)", "0.036", "0.012", "0.243", "0.114", "met", "1.00 / 1.00", "44.7 ms"]', f'["Intensity κ = 0", "R-GCN (event features)", "{g0.loc["top_ranked","drop"]:.3f}", "{g0.loc["random_matched","drop"]:.3f}", "{g0.loc["top_ranked","pch"]:.3f}", "{g0.loc["random_matched","pch"]:.3f}", "met", "1.00 / 1.00", "{l0["R-GCN"]:.1f} ms"]')
rep('"met (small)", "0.54 / 0.52", ""]', '"met (small)", "0.54 / 0.52–0.53", ""]')
rep('"met", "1.00 / 1.00", "6.9 ms"]', f'"met", "1.00 / 1.00", "{l1["Temporal Transformer"]:.1f} ms"]')
rep('["Intensity κ = 1", "R-GCN (event features)", "0.057", "0.009", "0.223", "0.089", "met", "0.82–1.00 / 1.00", "59.1 ms"]', f'["Intensity κ = 1", "R-GCN (event features)", "{g1.loc["top_ranked","drop"]:.3f}", "{g1.loc["random_matched","drop"]:.3f}", "{g1.loc["top_ranked","pch"]:.3f}", "{g1.loc["random_matched","pch"]:.3f}", "met", "0.82–1.00 / 1.00", "{l1["R-GCN"]:.1f} ms"]')
rep("Criterion = top-10 AUPRC drop exceeds the random-10 drop.", "Criterion = top-10 AUPRC drop exceeds the random-10 drop. Intensity-setting R-GCN rows: replacement checkpoints trained under the training-only overlap scale.")
rep("and the intensity target at κ = 0 and κ = 1 (seed 0, trained R-GCN and Transformer of those cells).", "and the intensity target at κ = 0 and κ = 1 (seed 0, trained R-GCN and Transformer of those cells). For the intensity settings the R-GCN checkpoints are the replacements trained under the training-only overlap scale, and the validation notebook reconstructs each graph with the overlap scale recorded by the checkpoint.")
rep("(|Δp| 0.22–0.51 versus 0.01–0.11)", "(|Δp| 0.22–0.51 versus 0.01–0.12)")
rep("(Spearman +0.03, −0.14, +0.77 across the three runs)", "(Spearman +0.03, −0.14, +0.60 across the three runs)")
rep("is 7–20 ms (Transformer) and 29–59 ms (R-GCN local subgraph with attribution) at p95.", "is 7–11 ms (Transformer) and 23–29 ms (R-GCN local subgraph with attribution) at p95.")
rep("Any R-GCN checkpoint replaced by the normalization rerun must be re-validated with the same package.", "The replacement intensity checkpoints passed the same checks as the checkpoints they replaced (Section 3.12).")
# ---------------- 3.8.1, 3.12, 3.13, 3.14, appendix, versions
diff = r33.merge(pd.read_csv(W / "v3_out/rgcn_seeds_aggregate/rgcn_training_budget_ablation_runs_v3.csv"), on=["kappa", "seed"], suffixes=("", "_old"))
dd = diff.auprc_trained_full - diff.auprc_trained_full_old; de = diff.trained_full_epochs - diff.trained_full_epochs_old
rep("One exception existed in the reported R-GCN runs:", "One exception existed in the R-GCN runs as first reported:")
rx(r"The patched notebook \(v3\.2\) derives the scale from training edges\..*?global maxima coincide\.",
   f"The corrected notebook (v3.3) derives the scale from training-partition edges and records it with every checkpoint; all twenty intensity cells were rerun under it with unchanged seeds and training settings, and every cell reports the training maximum (119.647 s) as the scale used, a maximum weight change of 0.000275. Only the full variant uses the overlap weight: the ablation and event-only variants reproduced exactly, while the full variant's held-out AUPRC changed by {sg(dd.min())} to {sg(dd.max())} per cell (mean {sg(dd.mean(), 4)}) because the small input change altered training trajectories and early stopping (epoch differences {int(de.min())} to +{int(de.max())}). All R-GCN intensity results in Sections 4.7.2–4.8 come from the corrected runs; no hypothesis outcome changed. The random-forest and Transformer runs are unaffected, since their inputs contain no overlap-weight normalization, and the regime-target R-GCN runs are unaffected because the training-partition and global maxima coincide.")
rep("No hypothesis rule was changed after its result was known except as stated here.", "The R-GCN overlap-scale correction was completed on 23 September 2026 in two steps: a first rerun under notebook v3.2 reproduced the earlier results exactly, which revealed that its training mask depended on a column absent from the edge table and had silently fallen back to the global scale; notebook v3.3 derives the mask from the frozen split, records the scale used with every checkpoint, and refuses to run without training edges. The explanation notebook was then amended to reconstruct each graph with the scale recorded by the checkpoint, after its reproduction check correctly rejected the v3.3 checkpoints under the old global rule (errors of 6 × 10⁻⁵ and 1.5 × 10⁻⁵ against a 10⁻⁵ tolerance); a first re-validation attempt executed stale staging folders and was repeated. The v3.2 outputs are retained in the companion package as evidence of the no-op. No hypothesis rule was changed after its result was known except as stated here.")
rep("(with the R-GCN overlap-scale exception quantified in Section 3.8.1)", "(the R-GCN overlap scale, originally global, was corrected to a training-only scale and all affected runs regenerated; Section 3.8.1)")
rep("the v3 package contains 595 files with a package-level checksum list.", "the companion package contains more than 700 files with a package-level checksum list, a companion-package map, and a clean-directory reproduction test.")
rep("A reproducible artifact package (595 files, checksummed)", "A reproducible artifact package (more than 700 files, checksummed, with a clean-directory reproduction test)")
rep('  ["Final checkpoints and preprocessors (3.13)", "collect_final_checkpoints_v3.ps1", "transformer_v2/models/fused_temporal_transformer.pt and rgcn_trained_v3/models/*.pt per cell and per regime seed, with rgcn_config, thresholds and reference JSONs"],\n', "")
rep('"final_checkpoints.zip: 112 checkpoints with config, threshold and reference JSON; fused_event_preprocessor.joblib and flow_preprocessor.joblib per R-GCN run"', '"final_checkpoints.zip: 232 checkpoints and 167 preprocessor files with config, threshold and reference JSON, covering the regime seeds, all cells, and the v3, v3.2 and v3.3 R-GCN runs"')
rep('["Required before freezing: R-GCN intensity rerun under training-only overlap scale (3.8.1)", "run_rgcn_v32_normalization_rerun.ps1 (notebook 06 v3.2, unchanged seeds and budget) -> 38_ aggregation -> 39_ XAI on replacement checkpoints", "rgcn_trained_v32/ per cell; refreshed Tables 4.12–4.14, 4.16 and Figures 4.4–4.5"]',
    '["R-GCN intensity rerun under the training-only overlap scale (3.8.1, 3.12)", "run_rgcn_v33_normalization_rerun.ps1 (notebook 06 v3.3; the v3.2 no-op is retained as evidence) -> 38_ aggregation -> 39_ with notebook 09 v3.3 on the replacement checkpoints", "rgcn_trained_v33/ per cell with overlap_scale_v33.json; rgcn_v33_aggregate/ (runs, seed-level inference, Figures 4.4–4.5); xai_k0_v33, xai_k1_v33"]')
rep('"39_stage_xai_on_trained_models_v3.py + notebook 09 on xai_regime_trained, xai_intensity_k0_seed0, xai_intensity_k1_seed0"', '"39_stage_xai_on_trained_models_v3.py + notebook 09 (regime) and notebook 09 v3.3 (intensity, replacement checkpoints)"')
src = src.replace("Chapters 1–5, Version 5.4 (third external-review corrections; R-GCN intensity results provisional pending normalization rerun)", "Chapters 1–5, Version 5.5 (final corrected results)")
src = src.replace("Roberts_Praxis_Chapters_1_5_v5_4.docx", "Roberts_Praxis_Chapters_1_5_v5_5.docx")
(W / "build_praxis_v55.js").write_text(src)
left = [w for w in ("provisional", "await the normalization", "0.7865", "0.9082", "19.5 ms", "44.7 ms", "59.1 ms", "+0.77", "595 files", "272–536") if w in src]
print("misses:", misses); print("stale strings left:", left)
print(f"H2 k0 {h2_0['mean']:+.4f} [{h2_0.lo:.4f},{h2_0.hi:.4f}] | H2 k1 {h2_1['mean']:+.4f} | H4 k0 {h4_0['mean']:+.4f} p={h4_0.p0:.4f} | epochs {emin}-{emax} | p95 regime {pr_min:.1f}-{pr_max:.1f}, cells {pc_min:.1f}-{pc_max:.1f}")
