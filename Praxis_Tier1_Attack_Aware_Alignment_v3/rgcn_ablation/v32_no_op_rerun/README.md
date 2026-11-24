# v3.2 rerun: no-op (documented)
The 20-cell rerun under notebook v3.2 reproduced the v3 results exactly (all variants identical to <1e-6 on identical events).
Cause: v3.2 derived the training mask from a `source_partition` column that the notebook's selected-edge frame does not carry,
so the condition never fired and the global scale was used. Notebook v3.3 derives the mask from the event's partition in the
split table, writes overlap_scale_v33.json (source, train/global maxima, max weight change) beside its outputs, and raises if no
training edges are found. Use run_rgcn_v33_normalization_rerun.ps1; the v3.2 outputs are retained here only as evidence of the no-op.
