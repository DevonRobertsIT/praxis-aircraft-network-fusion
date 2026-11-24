# Any-flow and matched flow-covered evaluations

## Any-flow baseline

The frozen v1 test set contains 80 flow-covered events out of 1,828. Predicting attack whenever any mapped flow is present yields AUPRC 0.5500, precision 0.5500, recall 1.0000, and F1 0.7097. This confirms that flow presence alone is a strong proxy in v1.

Under leakage-safe v2, 1,828 of 1,828 held-out events are flow-covered. The any-flow baseline yields AUPRC 0.4995 against prevalence 0.4995; it is therefore not a useful discriminator after active source-block remapping.

## Matched flow-covered evaluation

Events were matched 1:1 without replacement using coarsened exact matching on joint quantile strata of log flow count and log total flow duration. Within every eligible stratum, optimal assignment selected the closest individual pairs. From two through six bins per variable, the retained-sample-maximizing design with absolute standardized mean differences no greater than 0.10 on both covariates was selected separately for each analysis version. Alternative coarsening levels and full balance diagnostics are exported separately.

For frozen v1, 21 matched pairs remained. Fused-minus-cyber random-forest AUPRC was -0.003205 with a 95% paired-bootstrap interval of [-0.012894, +0.000000].

For leakage-safe v2, 531 matched pairs remained. Fused-minus-cyber random-forest AUPRC was +0.000000 with a 95% paired-bootstrap interval of [-0.000000, +0.000000].

## Interpretation boundary

These results evaluate discrimination within a controlled cross-domain replay. The matched design reduces imbalance in flow volume and duration; it does not make the independently collected aircraft and NF3 records historically co-occurring or causal.
