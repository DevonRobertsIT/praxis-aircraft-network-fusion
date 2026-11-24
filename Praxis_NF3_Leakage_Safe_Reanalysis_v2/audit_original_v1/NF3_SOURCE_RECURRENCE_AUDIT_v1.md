# Full NF3 source-block recurrence audit

**Gate decision: FAIL**

The audit streamed all **2,974,178** rows in the frozen primary edge table and tracked all **2,365,424** original NF3 source-flow identifiers.

## Exact source-flow recurrence

| left_split | right_split | left_unique_source_flows | right_unique_source_flows | reused_source_flows | overlap_fraction_of_smaller | example_source_row_numbers | example_flow_ids |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | validation | 1182024 | 501063 | 282222 | 0.563247 | 2336\|2337\|2338\|2339\|2340\|2341\|2342\|2343\|2344\|2345 | NF3-000000002336\|NF3-000000002337\|NF3-000000002338\|NF3-000000002339\|NF3-000000002340\|NF3-000000002341\|NF3-000000002342\|NF3-000000002343\|NF3-000000002344\|NF3-000000002345 |
| train | test | 1182024 | 469730 | 245114 | 0.521819 | 139\|140\|141\|142\|143\|144\|145\|146\|147\|148 | NF3-000000000139\|NF3-000000000140\|NF3-000000000141\|NF3-000000000142\|NF3-000000000143\|NF3-000000000144\|NF3-000000000145\|NF3-000000000146\|NF3-000000000147\|NF3-000000000148 |
| validation | test | 501063 | 469730 | 105402 | 0.224388 | 39026\|39027\|39028\|39029\|39030\|39031\|39032\|39033\|39034\|39035 | NF3-000000039026\|NF3-000000039027\|NF3-000000039028\|NF3-000000039029\|NF3-000000039030\|NF3-000000039031\|NF3-000000039032\|NF3-000000039033\|NF3-000000039034\|NF3-000000039035 |

## Mapped primary source-window overlap

| window_name | left_split | right_split | left_union_hours | right_union_hours | overlap_hours | overlap_fraction_of_smaller | left_merged_intervals | right_merged_intervals |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| primary_120s | train | validation | 364.659 | 108.678 | 64.7143 | 0.595468 | 4499 | 1624 |
| primary_120s | train | test | 364.659 | 109.327 | 63.3028 | 0.579025 | 4499 | 1631 |
| primary_120s | validation | test | 108.678 | 109.327 | 19.9886 | 0.183924 | 1624 | 1631 |

## Interpretation

The frozen cyclic replay does not provide source-independent train, validation, and test partitions. A leakage-safe remapping and rerun is therefore required.

This audit concerns reuse of NF3 source observations and source-time neighborhoods. It does not change the separate non-causal interpretation boundary of the controlled aircraft–NF3 pairing.