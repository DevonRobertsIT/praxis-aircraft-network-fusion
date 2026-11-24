# Review corrections (v3.1 grid)

## Label-freeze audit (test labels / bins differing between sigma = 0 and sigma > 0, same kappa and seed)
|   kappa |   sigma |   labels_differ |   bins_differ |
|--------:|--------:|----------------:|--------------:|
|    0    |      60 |               0 |             0 |
|    0    |     120 |               0 |             0 |
|    0    |     300 |               0 |             0 |
|    0.1  |      60 |               0 |             0 |
|    0.1  |     120 |               0 |             0 |
|    0.1  |     300 |               0 |             0 |
|    0.25 |      60 |               0 |             0 |
|    0.25 |     120 |               0 |             0 |
|    0.25 |     300 |               0 |             0 |
|    0.5  |      60 |               0 |             0 |
|    0.5  |     120 |               0 |             0 |
|    0.5  |     300 |               0 |             0 |
|    0.75 |      60 |               0 |             0 |
|    0.75 |     120 |               0 |             0 |
|    0.75 |     300 |               0 |             0 |
|    1    |      60 |               0 |             0 |
|    1    |     120 |               0 |             0 |
|    1    |     300 |               0 |             0 |

## H1'(b): paired AUPRC loss with frozen labels (seed mean, t-interval; seeds whose bin-cluster CI > 0)
|   kappa |   sigma | feature_set   | model               |   seeds |   loss_mean |   t_ci_low |   t_ci_high |   seeds_ci_gt0 |   seeds_ci_lt0 | supported_H1b   |
|--------:|--------:|:--------------|:--------------------|--------:|------------:|-----------:|------------:|---------------:|---------------:|:----------------|
|       0 |      60 | cyber         | logistic_regression |       5 |     -0.0041 |    -0.0321 |      0.0238 |              0 |              0 | False           |
|       0 |      60 | cyber         | random_forest       |       5 |     -0.007  |    -0.0336 |      0.0196 |              0 |              0 | False           |
|       0 |     120 | cyber         | logistic_regression |       5 |      0.033  |     0.0146 |      0.0515 |              1 |              0 | False           |
|       0 |     120 | cyber         | random_forest       |       5 |     -0.0053 |    -0.0423 |      0.0318 |              0 |              0 | False           |
|       0 |     300 | cyber         | logistic_regression |       5 |      0.1295 |     0.0883 |      0.1706 |              5 |              0 | True            |
|       0 |     300 | cyber         | random_forest       |       5 |     -0.0247 |    -0.0571 |      0.0077 |              0 |              0 | False           |
|       1 |      60 | cyber         | logistic_regression |       5 |     -0.0064 |    -0.018  |      0.0051 |              0 |              0 | False           |
|       1 |      60 | cyber         | random_forest       |       5 |     -0.0047 |    -0.0376 |      0.0282 |              0 |              0 | False           |
|       1 |     120 | cyber         | logistic_regression |       5 |      0.0167 |    -0.0052 |      0.0386 |              0 |              0 | False           |
|       1 |     120 | cyber         | random_forest       |       5 |      0.0226 |    -0.0014 |      0.0465 |              0 |              0 | False           |
|       1 |     300 | cyber         | logistic_regression |       5 |      0.1105 |     0.0745 |      0.1465 |              3 |              0 | False           |
|       1 |     300 | cyber         | random_forest       |       5 |     -0.0312 |    -0.0775 |      0.015  |              0 |              0 | False           |

## H5: fusion minus best unimodal model (selected on validation), sigma = 0
|   sigma |   kappa | model               | fusion   | best_unimodal_by_validation   |   gain_mean |   gain_min |   gain_max |   seeds_ci_gt0 |   seeds_ci_lt0 |   seeds | detected   |
|--------:|--------:|:--------------------|:---------|:------------------------------|------------:|-----------:|-----------:|---------------:|---------------:|--------:|:-----------|
|       0 |    0    | random_forest       | late     | cyber                         |     -0.0011 |    -0.0113 |     0.0139 |              0 |              1 |       5 | False      |
|       0 |    0    | random_forest       | fused    | cyber                         |     -0.0067 |    -0.0302 |     0.0064 |              0 |              1 |       5 | False      |
|       0 |    0    | logistic_regression | late     | cyber                         |     -0.004  |    -0.0102 |     0.0053 |              0 |              1 |       5 | False      |
|       0 |    0    | logistic_regression | fused    | cyber                         |     -0.0049 |    -0.0061 |    -0.0027 |              0 |              1 |       5 | False      |
|       0 |    0.1  | random_forest       | late     | cyber                         |      0.0032 |    -0.0107 |     0.0205 |              0 |              0 |       5 | False      |
|       0 |    0.1  | random_forest       | fused    | cyber                         |     -0.0032 |    -0.022  |     0.022  |              0 |              1 |       5 | False      |
|       0 |    0.1  | logistic_regression | late     | cyber                         |     -0.0091 |    -0.0141 |    -0.0027 |              0 |              0 |       5 | False      |
|       0 |    0.1  | logistic_regression | fused    | cyber                         |     -0.007  |    -0.0117 |    -0.0039 |              0 |              3 |       5 | False      |
|       0 |    0.25 | random_forest       | late     | cyber                         |      0.0283 |     0.0083 |     0.0588 |              1 |              0 |       5 | False      |
|       0 |    0.25 | random_forest       | fused    | cyber                         |     -0.0042 |    -0.0219 |     0.032  |              0 |              1 |       5 | False      |
|       0 |    0.25 | logistic_regression | late     | cyber                         |     -0.0039 |    -0.0131 |     0.0041 |              0 |              0 |       5 | False      |
|       0 |    0.25 | logistic_regression | fused    | cyber                         |     -0.0061 |    -0.008  |    -0.0031 |              0 |              3 |       5 | False      |
|       0 |    0.5  | random_forest       | late     | cyber                         |      0.0863 |     0.0747 |     0.0962 |              4 |              0 |       5 | False      |
|       0 |    0.5  | random_forest       | fused    | cyber                         |      0.0074 |    -0.0107 |     0.0347 |              1 |              0 |       5 | False      |
|       0 |    0.5  | logistic_regression | late     | cyber                         |     -0.0102 |    -0.013  |    -0.0029 |              0 |              2 |       5 | False      |
|       0 |    0.5  | logistic_regression | fused    | cyber                         |     -0.0048 |    -0.009  |    -0.0014 |              0 |              1 |       5 | False      |
|       0 |    0.75 | random_forest       | late     | cyber|physical                |      0.1396 |     0.0237 |     0.1844 |              4 |              0 |       5 | False      |
|       0 |    0.75 | random_forest       | fused    | cyber|physical                |      0.0106 |    -0.1227 |     0.0599 |              3 |              1 |       5 | False      |
|       0 |    0.75 | logistic_regression | late     | cyber                         |      0.0013 |    -0.0038 |     0.0041 |              0 |              0 |       5 | False      |
|       0 |    0.75 | logistic_regression | fused    | cyber                         |     -0.0024 |    -0.0076 |     0.0033 |              0 |              0 |       5 | False      |
|       0 |    1    | random_forest       | late     | physical                      |     -0.0179 |    -0.0201 |    -0.0171 |              0 |              0 |       5 | False      |
|       0 |    1    | random_forest       | fused    | physical                      |     -0.1325 |    -0.1414 |    -0.1217 |              0 |              5 |       5 | False      |
|       0 |    1    | logistic_regression | late     | cyber                         |      0.0113 |     0.0113 |     0.0113 |              0 |              0 |       5 | False      |
|       0 |    1    | logistic_regression | fused    | cyber                         |      0.0074 |     0.0074 |     0.0074 |              0 |              0 |       5 | False      |

## Reference: fusion minus cyber-only (operational question), sigma = 0
|   sigma |   kappa | model               | fusion   |   gain_over_cyber_mean |   seeds_ci_gt0 | detected_vs_cyber   |
|--------:|--------:|:--------------------|:---------|-----------------------:|---------------:|:--------------------|
|       0 |    0    | random_forest       | late     |                -0.0011 |              0 | False               |
|       0 |    0    | random_forest       | fused    |                -0.0067 |              0 | False               |
|       0 |    0    | logistic_regression | late     |                -0.004  |              0 | False               |
|       0 |    0    | logistic_regression | fused    |                -0.0049 |              0 | False               |
|       0 |    0.1  | random_forest       | late     |                 0.0032 |              0 | False               |
|       0 |    0.1  | random_forest       | fused    |                -0.0032 |              0 | False               |
|       0 |    0.1  | logistic_regression | late     |                -0.0091 |              0 | False               |
|       0 |    0.1  | logistic_regression | fused    |                -0.007  |              0 | False               |
|       0 |    0.25 | random_forest       | late     |                 0.0283 |              1 | False               |
|       0 |    0.25 | random_forest       | fused    |                -0.0042 |              0 | False               |
|       0 |    0.25 | logistic_regression | late     |                -0.0039 |              0 | False               |
|       0 |    0.25 | logistic_regression | fused    |                -0.0061 |              0 | False               |
|       0 |    0.5  | random_forest       | late     |                 0.0863 |              4 | False               |
|       0 |    0.5  | random_forest       | fused    |                 0.0074 |              1 | False               |
|       0 |    0.5  | logistic_regression | late     |                -0.0102 |              0 | False               |
|       0 |    0.5  | logistic_regression | fused    |                -0.0048 |              0 | False               |
|       0 |    0.75 | random_forest       | late     |                 0.169  |              5 | True                |
|       0 |    0.75 | random_forest       | fused    |                 0.04   |              2 | False               |
|       0 |    0.75 | logistic_regression | late     |                 0.0013 |              0 | False               |
|       0 |    0.75 | logistic_regression | fused    |                -0.0024 |              0 | False               |
|       0 |    1    | random_forest       | late     |                 0.2424 |              5 | True                |
|       0 |    1    | random_forest       | fused    |                 0.1279 |              5 | True                |
|       0 |    1    | logistic_regression | late     |                 0.0113 |              0 | False               |
|       0 |    1    | logistic_regression | fused    |                 0.0074 |              0 | False               |
