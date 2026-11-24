| Model | Validation AUPRC | Test AUPRC | Test ROC-AUC | Precision | Recall | F1 | Balanced accuracy | Brier score | Threshold | p95 latency (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Fused tabular — decision_tree | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 2.904 |
| Temporal Transformer — fused | 0.9995 | 1.0000 | 1.0000 | 0.9989 | 0.9967 | 0.9978 | 0.9978 | 0.0015 | 0.3675 | 12.820 |
| R-GCN — full lag/relation | 0.9446 | 0.9179 | 0.9081 | 0.7831 | 0.8740 | 0.8261 | 0.8163 | 0.1389 | 0.4395 | 2.410 |
| R-GCN — lag/relation ablation | 0.9256 | 0.9060 | 0.8979 | 0.7335 | 0.9014 | 0.8088 | 0.7873 | 0.1682 | 0.4153 | 1.165 |
