# Coupling-injection grid — summary (tabular level)

Cells: 120 main + 10 control; seeds per cell: 5; test events per cell: 1015; independent 240 s bins in test: 27; test prevalence (median over cells): 0.574

## Realized coupling (train Spearman of anomaly score vs. true intensity)
|   kappa |   mean |    min |   max |
|--------:|-------:|-------:|------:|
|    0    |  0.002 | -0.013 | 0.039 |
|    0.1  |  0.101 |  0.077 | 0.124 |
|    0.25 |  0.252 |  0.234 | 0.272 |
|    0.5  |  0.503 |  0.486 | 0.516 |
|    0.75 |  0.743 |  0.727 | 0.763 |
|    1    |  1     |  1     | 1     |

## AUPRC surface at sigma = 0 (mean over seeds)
| feature_set      | model               |   0.0 |   0.1 |   0.25 |   0.5 |   0.75 |   1.0 |
|:-----------------|:--------------------|------:|------:|-------:|------:|-------:|------:|
| cyber            | logistic_regression | 0.884 | 0.879 |  0.872 | 0.89  |  0.884 | 0.884 |
| cyber            | random_forest       | 0.706 | 0.7   |  0.719 | 0.699 |  0.71  | 0.708 |
| fused            | logistic_regression | 0.88  | 0.872 |  0.866 | 0.885 |  0.882 | 0.891 |
| fused            | random_forest       | 0.7   | 0.697 |  0.715 | 0.707 |  0.75  | 0.836 |
| late_fusion_mean | logistic_regression | 0.88  | 0.87  |  0.868 | 0.88  |  0.885 | 0.895 |
| late_fusion_mean | random_forest       | 0.705 | 0.704 |  0.747 | 0.786 |  0.879 | 0.951 |
| physical         | logistic_regression | 0.569 | 0.567 |  0.616 | 0.628 |  0.691 | 0.745 |
| physical         | random_forest       | 0.577 | 0.572 |  0.643 | 0.737 |  0.853 | 0.969 |

## Fusion gains at sigma = 0 (mean over seeds; detected = mean >= margin and every seed's bin-cluster CI excludes 0)
|   kappa | model               | fusion   | versus   |   gain_mean |   gain_min |   gain_max |   cluster_ci_low_mean |   seeds_ci_excluding_zero | detected   |
|--------:|:--------------------|:---------|:---------|------------:|-----------:|-----------:|----------------------:|--------------------------:|:-----------|
|    0    | random_forest       | fused    | cyber    |      -0.007 |     -0.03  |      0.006 |                -0.038 |                         0 | False      |
|    0    | random_forest       | fused    | physical |       0.122 |      0.099 |      0.15  |                -0.008 |                         1 | False      |
|    0    | random_forest       | late     | cyber    |      -0.001 |     -0.011 |      0.014 |                -0.019 |                         0 | False      |
|    0    | random_forest       | late     | physical |       0.128 |      0.106 |      0.142 |                -0     |                         2 | False      |
|    0    | logistic_regression | fused    | cyber    |      -0.005 |     -0.006 |     -0.003 |                -0.016 |                         0 | False      |
|    0    | logistic_regression | fused    | physical |       0.311 |      0.27  |      0.358 |                 0.172 |                         5 | True       |
|    0    | logistic_regression | late     | cyber    |      -0.004 |     -0.01  |      0.005 |                -0.019 |                         0 | False      |
|    0    | logistic_regression | late     | physical |       0.311 |      0.272 |      0.352 |                 0.178 |                         5 | True       |
|    0.1  | random_forest       | fused    | cyber    |      -0.003 |     -0.022 |      0.022 |                -0.037 |                         0 | False      |
|    0.1  | random_forest       | fused    | physical |       0.125 |      0.05  |      0.186 |                -0.01  |                         2 | False      |
|    0.1  | random_forest       | late     | cyber    |       0.003 |     -0.011 |      0.02  |                -0.022 |                         0 | False      |
|    0.1  | random_forest       | late     | physical |       0.132 |      0.079 |      0.203 |                 0.002 |                         2 | False      |
|    0.1  | logistic_regression | fused    | cyber    |      -0.007 |     -0.012 |     -0.004 |                -0.017 |                         0 | False      |
|    0.1  | logistic_regression | fused    | physical |       0.305 |      0.261 |      0.342 |                 0.164 |                         5 | True       |
|    0.1  | logistic_regression | late     | cyber    |      -0.009 |     -0.014 |     -0.003 |                -0.027 |                         0 | False      |
|    0.1  | logistic_regression | late     | physical |       0.303 |      0.251 |      0.346 |                 0.17  |                         5 | True       |
|    0.25 | random_forest       | fused    | cyber    |      -0.004 |     -0.022 |      0.032 |                -0.039 |                         0 | False      |
|    0.25 | random_forest       | fused    | physical |       0.072 |      0.04  |      0.118 |                -0.052 |                         0 | False      |
|    0.25 | random_forest       | late     | cyber    |       0.028 |      0.008 |      0.059 |                -0.007 |                         1 | False      |
|    0.25 | random_forest       | late     | physical |       0.105 |      0.078 |      0.144 |                -0.011 |                         1 | False      |
|    0.25 | logistic_regression | fused    | cyber    |      -0.006 |     -0.008 |     -0.003 |                -0.017 |                         0 | False      |
|    0.25 | logistic_regression | fused    | physical |       0.25  |      0.226 |      0.263 |                 0.12  |                         5 | True       |
|    0.25 | logistic_regression | late     | cyber    |      -0.004 |     -0.013 |      0.004 |                -0.019 |                         0 | False      |
|    0.25 | logistic_regression | late     | physical |       0.252 |      0.229 |      0.269 |                 0.124 |                         5 | True       |
|    0.5  | random_forest       | fused    | cyber    |       0.007 |     -0.011 |      0.035 |                -0.024 |                         1 | False      |
|    0.5  | random_forest       | fused    | physical |      -0.03  |     -0.082 |      0.014 |                -0.151 |                         0 | False      |
|    0.5  | random_forest       | late     | cyber    |       0.086 |      0.075 |      0.096 |                 0.013 |                         4 | False      |
|    0.5  | random_forest       | late     | physical |       0.049 |      0.016 |      0.076 |                -0.041 |                         0 | False      |
|    0.5  | logistic_regression | fused    | cyber    |      -0.005 |     -0.009 |     -0.001 |                -0.017 |                         0 | False      |
|    0.5  | logistic_regression | fused    | physical |       0.257 |      0.203 |      0.301 |                 0.136 |                         5 | True       |
|    0.5  | logistic_regression | late     | cyber    |      -0.01  |     -0.013 |     -0.003 |                -0.03  |                         0 | False      |
|    0.5  | logistic_regression | late     | physical |       0.252 |      0.21  |      0.294 |                 0.128 |                         5 | True       |
|    0.75 | random_forest       | fused    | cyber    |       0.04  |      0.024 |      0.06  |                 0.001 |                         2 | False      |
|    0.75 | random_forest       | fused    | physical |      -0.104 |     -0.123 |     -0.079 |                -0.235 |                         0 | False      |
|    0.75 | random_forest       | late     | cyber    |       0.169 |      0.151 |      0.184 |                 0.061 |                         5 | True       |
|    0.75 | random_forest       | late     | physical |       0.025 |      0.016 |      0.042 |                -0.035 |                         0 | False      |
|    0.75 | logistic_regression | fused    | cyber    |      -0.002 |     -0.008 |      0.003 |                -0.015 |                         0 | False      |
|    0.75 | logistic_regression | fused    | physical |       0.19  |      0.179 |      0.213 |                 0.08  |                         5 | True       |
|    0.75 | logistic_regression | late     | cyber    |       0.001 |     -0.004 |      0.004 |                -0.017 |                         0 | False      |
|    0.75 | logistic_regression | late     | physical |       0.194 |      0.182 |      0.217 |                 0.092 |                         5 | True       |
|    1    | random_forest       | fused    | cyber    |       0.128 |      0.115 |      0.142 |                 0.05  |                         5 | True       |
|    1    | random_forest       | fused    | physical |      -0.132 |     -0.141 |     -0.122 |                -0.263 |                         0 | False      |
|    1    | random_forest       | late     | cyber    |       0.242 |      0.236 |      0.25  |                 0.1   |                         5 | True       |
|    1    | random_forest       | late     | physical |      -0.018 |     -0.02  |     -0.017 |                -0.058 |                         0 | False      |
|    1    | logistic_regression | fused    | cyber    |       0.007 |      0.007 |      0.007 |                -0.012 |                         0 | False      |
|    1    | logistic_regression | fused    | physical |       0.146 |      0.146 |      0.146 |                 0.056 |                         5 | True       |
|    1    | logistic_regression | late     | cyber    |       0.011 |      0.011 |      0.011 |                -0.007 |                         0 | False      |
|    1    | logistic_regression | late     | physical |       0.15  |      0.15  |      0.15  |                 0.063 |                         5 | True       |

## Minimum detectable coupling strength
|   sigma | model               | fusion   | versus   |   min_detectable_kappa |
|--------:|:--------------------|:---------|:---------|-----------------------:|
|       0 | logistic_regression | fused    | cyber    |                 inf    |
|       0 | logistic_regression | fused    | physical |                   0    |
|       0 | logistic_regression | late     | cyber    |                 inf    |
|       0 | logistic_regression | late     | physical |                   0    |
|       0 | random_forest       | fused    | cyber    |                   1    |
|       0 | random_forest       | fused    | physical |                 inf    |
|       0 | random_forest       | late     | cyber    |                   0.75 |
|       0 | random_forest       | late     | physical |                 inf    |
|      60 | logistic_regression | fused    | cyber    |                 inf    |
|      60 | logistic_regression | fused    | physical |                   0    |
|      60 | logistic_regression | late     | cyber    |                 inf    |
|      60 | logistic_regression | late     | physical |                   0    |
|      60 | random_forest       | fused    | cyber    |                   0.75 |
|      60 | random_forest       | fused    | physical |                   0    |
|      60 | random_forest       | late     | cyber    |                   0.5  |
|      60 | random_forest       | late     | physical |                   0    |
|     120 | logistic_regression | fused    | cyber    |                 inf    |
|     120 | logistic_regression | fused    | physical |                   0    |
|     120 | logistic_regression | late     | cyber    |                 inf    |
|     120 | logistic_regression | late     | physical |                   0    |
|     120 | random_forest       | fused    | cyber    |                   0.75 |
|     120 | random_forest       | fused    | physical |                 inf    |
|     120 | random_forest       | late     | cyber    |                   0.5  |
|     120 | random_forest       | late     | physical |                   0    |
|     300 | logistic_regression | fused    | cyber    |                   1    |
|     300 | logistic_regression | fused    | physical |                   0    |
|     300 | logistic_regression | late     | cyber    |                   1    |
|     300 | logistic_regression | late     | physical |                   0    |
|     300 | random_forest       | fused    | cyber    |                   0.75 |
|     300 | random_forest       | fused    | physical |                   0    |
|     300 | random_forest       | late     | cyber    |                   0.5  |
|     300 | random_forest       | late     | physical |                   0    |

## H1' — cyber-only AUPRC vs. alignment jitter at kappa = 0
| model               |   sigma |   cyber_auprc_loss_vs_sigma0_at_kappa0 | loss_ge_0_02   |
|:--------------------|--------:|---------------------------------------:|:---------------|
| random_forest       |      60 |                                 -0.007 | False          |
| random_forest       |     120 |                                 -0.005 | False          |
| random_forest       |     300 |                                 -0.025 | False          |
| logistic_regression |      60 |                                 -0.004 | False          |
| logistic_regression |     120 |                                  0.033 | True           |
| logistic_regression |     300 |                                  0.129 | True           |

## Late-fusion gain over cyber vs. jitter (mean over seeds)
| model               |   kappa |    0.0 |   60.0 |   120.0 |   300.0 |
|:--------------------|--------:|-------:|-------:|--------:|--------:|
| logistic_regression |    0    | -0.004 | -0.003 |  -0.007 |  -0.003 |
| logistic_regression |    0.1  | -0.009 | -0.006 |  -0.01  |  -0.006 |
| logistic_regression |    0.25 | -0.004 | -0.005 |  -0.003 |   0.008 |
| logistic_regression |    0.5  | -0.01  | -0.01  |  -0.01  |  -0.001 |
| logistic_regression |    0.75 |  0.001 |  0.001 |   0.002 |   0.029 |
| logistic_regression |    1    |  0.011 |  0.011 |   0.016 |   0.056 |
| random_forest       |    0    | -0.001 | -0.004 |  -0.001 |  -0.004 |
| random_forest       |    0.1  |  0.003 | -0.002 |   0.002 |  -0.003 |
| random_forest       |    0.25 |  0.028 |  0.021 |   0.026 |   0.019 |
| random_forest       |    0.5  |  0.086 |  0.083 |   0.09  |   0.075 |
| random_forest       |    0.75 |  0.169 |  0.159 |   0.158 |   0.154 |
| random_forest       |    1    |  0.242 |  0.24  |   0.268 |   0.224 |

## Attribution tracking at sigma = 0 (RF importance share)
|   kappa | feature_set   |   score_share_mean |   physical_share_mean |
|--------:|:--------------|-------------------:|----------------------:|
|    0    | fused         |              0.005 |                 0.012 |
|    0    | physical      |              0.402 |               nan     |
|    0.1  | fused         |              0.005 |                 0.012 |
|    0.1  | physical      |              0.407 |               nan     |
|    0.25 | fused         |              0.005 |                 0.013 |
|    0.25 | physical      |              0.441 |               nan     |
|    0.5  | fused         |              0.018 |                 0.04  |
|    0.5  | physical      |              0.52  |               nan     |
|    0.75 | fused         |              0.064 |                 0.133 |
|    0.75 | physical      |              0.588 |               nan     |
|    1    | fused         |              0.18  |                 0.327 |
|    1    | physical      |              0.637 |               nan     |

## Negative control (coupling to a shuffled score)
|   kappa | model               | fusion   |   gain_over_cyber_mean |   cluster_ci_low_mean |   spearman_mean | leak_detected   |
|--------:|:--------------------|:---------|-----------------------:|----------------------:|----------------:|:----------------|
|     0.5 | random_forest       | fused    |                 -0.001 |                -0.027 |           0     | False           |
|     0.5 | random_forest       | late     |                 -0     |                -0.02  |           0     | False           |
|     0.5 | logistic_regression | fused    |                 -0.005 |                -0.014 |           0     | False           |
|     0.5 | logistic_regression | late     |                 -0.005 |                -0.018 |           0     | False           |
|     1   | random_forest       | fused    |                  0.012 |                -0.014 |           0.004 | False           |
|     1   | random_forest       | late     |                 -0.002 |                -0.019 |           0.004 | False           |
|     1   | logistic_regression | fused    |                 -0.007 |                -0.017 |           0.004 | False           |
|     1   | logistic_regression | late     |                 -0.011 |                -0.028 |           0.004 | False           |
