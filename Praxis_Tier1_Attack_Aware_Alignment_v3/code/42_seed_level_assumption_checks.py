"""Seed-level assumption checks for Table 4.15 (manuscript v6.1, Section 4.7.3).

For each of the eight paired contrasts (four contrasts x kappa in {0, 1}) over the ten final runs:
  - one-sample t-test of the paired seed differences against zero (two-sided), as reported in Table 4.15
  - Shapiro-Wilk test of normality of the ten differences
  - exact sign-flip permutation test over all 2^10 = 1,024 sign assignments (two-sided, |mean|)
  - Wilcoxon signed-rank test (supplementary)
Input : rgcn_ablation/v33_final/rgcn_training_budget_ablation_runs_v3.csv
Output: final_reproduction/seed_level_assumption_checks_v6_1.csv
"""
import itertools, pathlib, sys
import numpy as np, pandas as pd
from scipy import stats

root = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path('.')
runs = pd.read_csv(root / 'rgcn_ablation/v33_final/rgcn_training_budget_ablation_runs_v3.csv')
contrasts = {
    "H2' full - Transformer": 'H2_trained_full_minus_transformer',
    'H4 full - event-only': 'relational_full_minus_event_only',
    "H3' full - ablation": 'H3_trained_full_minus_ablation',
    'event-only - Transformer': 'event_only_minus_transformer',
}
signs = np.array(list(itertools.product([1, -1], repeat=10)))
rows = []
for kappa in (0.0, 1.0):
    cell = runs[runs.kappa == kappa].sort_values('seed')
    assert len(cell) == 10
    for label, col in contrasts.items():
        d = cell[col].to_numpy()
        perm = float((np.abs((signs * d).mean(axis=1)) >= abs(d.mean()) - 1e-15).mean())
        rows.append({'kappa': kappa, 'contrast': label, 'n_seeds': len(d), 'mean_difference': d.mean(),
                     't_test_p_two_sided': stats.ttest_1samp(d, 0).pvalue,
                     'shapiro_wilk_p': stats.shapiro(d).pvalue,
                     'sign_flip_permutation_p_exact': perm,
                     'wilcoxon_signed_rank_p': stats.wilcoxon(d).pvalue})
out = pd.DataFrame(rows)
dest = root / 'final_reproduction/seed_level_assumption_checks_v6_1.csv'
out.to_csv(dest, index=False)
print(out.round(4).to_string(index=False))
print('minimum Shapiro-Wilk p:', round(out.shapiro_wilk_p.min(), 4))
