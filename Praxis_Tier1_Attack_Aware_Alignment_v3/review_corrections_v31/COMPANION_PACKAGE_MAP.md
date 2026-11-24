# Companion package map (what reproduces what)

| Result | Inputs in this package | Code | Machine |
|---|---|---|---|
| v3 alignment, gates, audits (Tables 3.3-3.4, 4.2-4.4) | Project.zip (original data; NF3 CSV inside), alignment_v3_cross_regime/ | code/11_nf3_leakage_safe_alignment_v3.py (+ v2 code/11_nf3_leakage_safe_alignment.py) | any Python 3.12 |
| Tabular regime results, baselines, cluster bootstrap (Table 4.5) | alignment_v3_cross_regime/, tabular_modeling_v4/feature_manifest_v4.json | code/20_, code/21_ | any |
| H1'(a) timestamp basis (4.5) | alignment_v3_cross_regime_snapshot_basis_audit/ + rerun of 11_ with --time-basis | code/22_ | any |
| Coupling grid, corrected (Tables 4.7-4.10, 4.8b; 4.6.4) | step2_coupling_study/position_bank_cross_regime_v3.csv, position_bank_edges_cross_regime_v3.csv; review_corrections_v31/coupling_grid_test_predictions_v3.csv, coupling_grid_cells_v3.csv, coupling_grid_results_v3.csv | code/30_, 34_, 31_ (v3.1), 33_, 40_ | any (paired bootstrap: 40_ on the exported predictions only) |
| Neural cells, Step 1 seeds (Tables 4.6, 4.11-4.14, Figs 4.1, 4.3-4.5) | cells materialized by code/35_ from the files above; final_checkpoints/final_checkpoints.zip (models, preprocessing, JSON) | notebooks 05_, 06_ (v2 package) and rgcn_ablation/06_rgcn_leakage_safe_v3_trained.ipynb (v3.3, final); code/32_, 35_, 36_, 38_ | PyTorch |
| Explanation validation (Table 4.16) | regime: xai_trained_results/xai_regime_trained/; intensity (final): rgcn_ablation/v33_final/xai/; inputs: final_checkpoints/final_checkpoints.zip (xai_trained_results/xai_intensity_* are historical, v3 checkpoints) | code/39_ + original notebook 09 (regime), rgcn_ablation/09_xai_validation_leakage_safe_v33.ipynb (intensity) | PyTorch |
| SUPERSEDED (v3.2 no-op) | superseded/runners/run_rgcn_v32_normalization_rerun.ps1 | notebook 06 v3.2 | - |

Clean-directory test performed 23 Sep 2026: package extracted to an empty folder; code/40_review_corrections_v31.py rerun on
review_corrections_v31/ alone reproduced h1prime_paired_jitter_loss_aggregate_v31.csv and h5_fusion_vs_best_unimodal_v31.csv
(see CLEAN_DIR_TEST.txt). Notebook-based results were not re-executed in the clean test (no PyTorch in the test environment).

## Final R-GCN intensity results (v5.5)
| Result | Inputs | Code | Machine |
|---|---|---|---|
| Tables 4.12–4.14, Figures 4.4–4.5 (R-GCN intensity) | rgcn_ablation/v33_final/ (runs CSV, seed_level_inference_v33.csv, overlap_scale_records/); final_checkpoints/final_checkpoints.zip (rgcn_trained_v33 checkpoints + preprocessors) | notebook 06 v3.3 (rgcn_ablation/06_rgcn_leakage_safe_v3_trained.ipynb), code/38_, run_rgcn_v33_normalization_rerun.ps1 | PyTorch |
| Table 4.16 intensity R-GCN rows | rgcn_ablation/v33_final/xai/ | notebook 09 v3.3 (rgcn_ablation/09_xai_validation_leakage_safe_v33.ipynb), code/39_ | PyTorch |
| Evidence of the v3.2 no-op | rgcn_ablation/v32_no_op_rerun/ | — | — |

## Final neural predictions and clean-directory test (v5.6)
| Result | Inputs | Code | Machine |
|---|---|---|---|
| Tables 4.12-4.14 reproduced from predictions | final_reproduction/final_neural_test_predictions_v33.csv | code/41_reproduce_final_neural_tables_v33.py (export: code/export_final_neural_predictions_v33.py) | any Python |
| Scope of the clean-directory tests | final_reproduction/CLEAN_DIR_TEST_FINAL.txt | - | - |
| Final model entry point | rgcn_ablation/run_rgcn_v33_normalization_rerun.ps1 (explicit notebook 06 v3.3 and 09 v3.3; _v33 folders only) | - | PyTorch |

## Dependencies
Analysis scripts: `python -m pip install -r requirements-analysis.txt` (tested versions: requirements-analysis-tested.txt).
Notebook reruns: `python -m pip install -r requirements-models.txt` (torch 2.8.0 CPU, IPython).
