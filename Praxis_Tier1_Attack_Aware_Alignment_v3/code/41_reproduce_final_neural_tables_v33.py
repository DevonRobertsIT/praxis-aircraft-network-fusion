#!/usr/bin/env python3
"""Reproduce Tables 4.12 (rows), 4.13 and 4.14 from final_neural_test_predictions_v33.csv alone; optionally compare with saved results."""
import argparse, json, numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score as aps
ap = argparse.ArgumentParser(); ap.add_argument("--predictions", default="final_neural_test_predictions_v33.csv"); ap.add_argument("--reference-inference", default=None); ap.add_argument("--repeats", type=int, default=400); a = ap.parse_args()
P = pd.read_csv(a.predictions); rng = np.random.default_rng(42)
def boot(y, s1, s2, bins):
    u, inv = np.unique(bins, return_inverse=True); mem = [np.flatnonzero(inv == k) for k in range(len(u))]; out = []
    for _ in range(a.repeats):
        i = np.concatenate([mem[k] for k in rng.integers(0, len(u), len(u))]); yy = y[i]
        out.append(aps(yy, s1[i]) - aps(yy, s2[i]) if 0 < yy.sum() < len(yy) else np.nan)
    return np.nanpercentile(out, 2.5), np.nanpercentile(out, 97.5)
pairs = {"H2": ("rgcn_full", "transformer"), "H4": ("rgcn_full", "rgcn_event_only"), "H3": ("rgcn_full", "rgcn_ablation"), "ev": ("rgcn_event_only", "transformer"), "budget": ("rgcn_full", "original_budget_rgcn_full")}
runs = []
for (tgt, k, s), g in P.groupby(["target", "kappa", "seed"], dropna=False):
    y = g.model_target.to_numpy().astype(int); b = g.bin_240s.to_numpy(); r = {"target": tgt, "kappa": k, "seed": s}
    for col in ("transformer", "rgcn_full", "rgcn_ablation", "rgcn_event_only", "original_budget_rgcn_full"):
        r[col] = aps(y, g[col]) if g[col].notna().all() else np.nan
    for q, (x, z) in pairs.items():
        if np.isnan(r.get(x, np.nan)) or np.isnan(r.get(z, np.nan)): continue
        r[q] = r[x] - r[z]
        if tgt == "intensity" and (q != "budget" or s <= 2): r[q + "_lo"], r[q + "_hi"] = boot(y, g[x].to_numpy(), g[z].to_numpy(), b)
    runs.append(r)
R = pd.DataFrame(runs); R.to_csv("reproduced_runs_v33.csv", index=False)
inf = []
for k, g in R[R.target.eq("intensity")].groupby("kappa"):
    for q in ("H2", "H4", "H3", "ev"):
        d = g[q].to_numpy(); n = len(d); se = d.std(ddof=1) / np.sqrt(n); t = stats.t.ppf(0.975, n - 1)
        inf.append({"kappa": k, "q": q, "mean": d.mean(), "sd": d.std(ddof=1), "lo": d.mean() - t * se, "hi": d.mean() + t * se, "pos": int((d > 0).sum()), "ci_gt0": int((g[q + "_lo"] > 0).sum()), "ci_lt0": int((g[q + "_hi"] < 0).sum()), "p0": stats.ttest_1samp(d, 0).pvalue})
I = pd.DataFrame(inf); I.to_csv("reproduced_seed_level_inference_v33.csv", index=False)
T13 = R[R.target.eq("intensity")].groupby("kappa")[["transformer", "rgcn_full", "rgcn_ablation", "rgcn_event_only"]].agg(["mean", "std"]); T13.to_csv("reproduced_table_4_13_v33.csv")
T12 = R[(R.target.eq("intensity") & (R.seed <= 2)) | R.target.eq("regime")].groupby(["target", "kappa"], dropna=False)[["transformer", "original_budget_rgcn_full", "rgcn_full", "rgcn_ablation", "rgcn_event_only", "budget"]].mean(); T12.to_csv("reproduced_table_4_12_v33.csv")
rep = {"deterministic": {}, "bootstrap": {}}
if a.reference_inference:
    ref = pd.read_csv(a.reference_inference); ref["q"] = ref.question.str.extract(r"^(H2|H4|H3|event-only)")[0].replace({"event-only": "ev"})
    m = I.merge(ref, on=["kappa", "q"], suffixes=("", "_ref"))
    rep["deterministic"] = {c: float((m[c] - m[c + "_ref"]).abs().max()) for c in ("mean", "sd", "lo", "hi", "p0")} | {"pos_mismatches": int((m.pos != m.pos_ref).sum())}
    rep["bootstrap"] = {"ci_gt0_count_differences": (m.ci_gt0 - m.ci_gt0_ref).tolist(), "ci_lt0_count_differences": (m.ci_lt0 - m.ci_lt0_ref).tolist()}
print(json.dumps(rep)); print(I.round(4).to_string(index=False)); print(T12.round(4).to_string())
