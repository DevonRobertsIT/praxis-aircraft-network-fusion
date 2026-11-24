# An Audited Framework for Aligning and Fusing Low-Altitude Aircraft Events with Network Telemetry under Controlled Coupling

Code and notebooks for the Doctor of Engineering praxis by Devon Roberts, School of Engineering and Applied Science, The George Washington University. Final examination: December 2, 2026.

The files are the scripts and notebooks listed in Appendix A of the praxis (Table A-1), kept in the experiment packages they were delivered in. As stated in Appendix A, `NUMBERS_FINAL.json` and the final comparison files are the sources of record.

The reported results come from release `defense-2026-12-02`. Later commits may clarify the documentation but do not change that release.

## Start here

- **See the results without running anything:** in the final package, `NUMBERS_FINAL.json` holds the final hypothesis results and `review_corrections_v31/COMPANION_PACKAGE_MAP.md` lists the files behind each table and figure. The results are CSV, JSON, Markdown and PNG files, which GitHub displays in the browser. Most notebooks are saved without outputs (see [Notes on the notebooks](#notes-on-the-notebooks)).
- **Check the final tables in a few minutes:** see [Quick check](#quick-check).
- **Find the code behind a section or table:** see [Map of the repository](#map-of-the-repository). To jump to a file, press `t` on the repository page and type its name.
- **Rerun the whole study:** see [Full rerun](#full-rerun). It needs the Flightradar24 data, which is not in this repository.

## Repository layout

| Label | Folder | Contents |
|---|---|---|
| v3 | `Praxis_Tier1_Attack_Aware_Alignment_v3/` | Final package: analysis scripts in `code/`, results, frozen predictions, checkpoints and `NUMBERS_FINAL.json` |
| v2 | `Praxis_NF3_Leakage_Safe_Reanalysis_v2/` | v1 audit and v2 design scripts in `code/`; Transformer, R-GCN and explanation notebooks in `notebooks/` |
| ext | `Praxis_Validity_Extensions_v3/` | Standardized latency benchmark |
| nb | `Project/Jupyter Notebook/` | Point-event notebooks 01–03, `config/point_event_v1.yaml` and the Flightradar24 download script |

Paths in the tables below start from these folders.

## Map of the repository

### Final pipeline

Steps are listed in dependency order.

| # | Step | Praxis | Code | Main outputs |
|---|---|---|---|---|
| 1 | Point events and episodes | §3.4 | nb: `01_build_candidate_point_events_v1.ipynb`, `02_add_episode_and_recurrence_features.ipynb` | nb: `data/processed/point_event_v1/context_v1/` † |
| 2 | Alignment (v3, cross-regime) | §3.6.3–3.6.5 | v3: `code/11_nf3_leakage_safe_alignment_v3.py`, run by `code/run_alignments_v3.sh` | v3: `alignment_v3_cross_regime/`: source-block manifest, session and partition audits, build summary |
| 3 | Tabular baselines and cluster bootstrap | §3.8.2, §3.10 | v3: `code/20_tabular_baselines_v3.py`, `code/21_session_baseline_and_cluster_bootstrap_v3.py` | v3: `tabular_v3_cross_regime/`, `baselines_v3_cross_regime/` |
| 4 | Timestamp-basis check (descriptive) | §4.5 | v3: `code/22_h1_timestamp_basis_v3.py`, on the snapshot-basis alignment | v3: `h1_v3/`, `alignment_v3_cross_regime_snapshot_basis_audit/` |
| 5 | Position bank and coupling grid | §3.9 | v3: `code/30_position_bank_v3.py`, `code/34_position_bank_edges_v3.py`, `code/31_coupling_injection_grid_v3.py`, `code/33_summarize_coupling_grid_v3.py` | v3: `step2_coupling_study/`; the corrected grid is in `review_corrections_v31/` |
| 6 | Neural staging and runs | §3.8.3–3.8.5 | v3: `code/32_stage_torch_reruns_v3.py`, `torch_rerun_staging/run_torch_reruns_v3.sh`, `code/35_materialize_coupling_cell_v3.py`; v2: `notebooks/05_temporal_transformer_leakage_safe_v2.ipynb`, `notebooks/06_rgcn_leakage_safe_v2.ipynb` (original training budget) | v3: `step1_torch_reruns/`, `step2_torch_cells/` |
| 7 | Aggregation | §3.10 | v3: `code/36_aggregate_torch_reruns_v3.py`, `code/38_aggregate_rgcn_training_ablation_v3.py` | v3: `step1_torch_reruns/step1_seed_results_v3.csv`, `step2_torch_cells/`, `rgcn_ablation/results/` |
| 8 | Review corrections (H1′ and H5) | §3.9, §4.6.3, §4.6.4 | v3: `code/40_review_corrections_v31.py` | v3: `review_corrections_v31/` |
| 9 | Final R-GCN with the training-only overlap scale | §3.8.1, §3.12 | v3: `rgcn_ablation/run_rgcn_v33_normalization_rerun.ps1`, which runs `rgcn_ablation/06_rgcn_leakage_safe_v3_trained.ipynb` (v3.3); then `code/38_aggregate_rgcn_training_ablation_v3.py` | v3: `rgcn_ablation/v33_final/`: runs, `seed_level_inference_v33.csv`, Figures 4-4 and 4-5, `overlap_scale_records/` |
| 10 | Explanation checks on the final checkpoints | §4.8 | v3: script 39 in `code/`; v2: notebook 09 in `notebooks/` (regime); v3: notebook 09 v3.3 in `rgcn_ablation/` (intensity) | v3: regime explanation outputs; intensity explanation outputs under `rgcn_ablation/v33_final/` |
| 11 | Seed-level assumption checks | §4.7.3 | v3: `code/42_seed_level_assumption_checks.py` | v3: `final_reproduction/seed_level_assumption_checks_v6_1.csv` |
| 12 | Final checkpoints and preprocessors | §3.13 | v3: `rgcn_ablation/collect_final_checkpoints_v3.ps1` | v3: `final_checkpoints/final_checkpoints.zip`: 232 checkpoints and 167 preprocessor files |
| 13 | Final predictions and table reproduction | §3.13, §4.7 | v3: `code/export_final_neural_predictions_v33.py`, `code/41_reproduce_final_neural_tables_v33.py` | v3: `final_reproduction/`: `final_neural_test_predictions_v33.csv`, `reproduced_*.csv`, `CLEAN_DIR_TEST_FINAL.txt` |
| 14 | Standardized latency | §4.9 | ext: `scripts/18_standardized_model_latency_v3.py` | ext: `validity_extensions_v3/standardized_latency_summary_v3.csv`; measured on the frozen v2 split and checkpoints |

† Contains aircraft identifiers and positions, so it is not posted here (see [Data](#data)).

The ten-seed intensity R-GCN results (Table 4-14) use step 9's training-only overlap scale. `review_corrections_v31/COMPANION_PACKAGE_MAP.md` lists which inputs and code reproduce each table and figure.

Saved filenames and older records keep the original table numbers. The praxis's Tables 4-9 to 4-15 were numbered 4.8b to 4.14, and the former Table 4-15 (hypothesis summary) is now Table 4-17. The numbers themselves are unchanged.

### Historical and supporting files

These files are kept as the audit trail. None of them is used for the final estimates.

| Files | Praxis | What they document |
|---|---|---|
| nb: `03_align_point_events_with_nf3.ipynb`; v2: `code/10_nf3_recurrence_audit.py`, `audit_original_v1/` | §3.6.1 | v1 cyclic replay and its recurrence audit |
| v2: `code/11_nf3_leakage_safe_alignment.py`, `alignment_leakage_safe_v2/` | §3.6.2 | v2 source-block design. The v3 alignment script imports its helper functions |
| v3: `code/14_validate_reanalysis_patched.py`, `validation_v2_rerun/` | §3.6.2 | Independent validator, patched to 36 checks |
| v3: `alignment_v3_within_session_audit/`, `tabular_v3_within_session/`, `baselines_v3_within_session/` | §3.6.3 | Within-session comparison design, in which the session indicator stayed predictive |
| v3: `superseded/runners/run_rgcn_v32_normalization_rerun.ps1`, `rgcn_ablation/v32_no_op_rerun/` | §3.8.1 | Notebook 06 v3.2, which kept the global overlap scale because the training mask referenced an edge-table column that was not present |
| v2: notebook 09 and its outputs on the v2 checkpoints | §3.11 | Earlier explanation checks |
| v3: the `intensity_k0_seed0` and `intensity_k1_seed0` explanation folders | §4.8 | Explanation checks on the superseded v3 checkpoints |
| v3: `rgcn_ablation/run_sigma120_cells_corrected_v3.ps1` | — | Optional σ = 120 s neural cells. Not executed for the final evidence; before use it would need notebook 06 v3.3 and the recorded training-only scale |

Other files in the v2 and ext packages are kept as delivered; Appendix A does not cite them.

## Quick check

This rebuilds the final neural-model tables and the coupling-grid corrections from the saved predictions, without retraining any model. It repeats the clean-directory test recorded on September 23, 2026 in `final_reproduction/CLEAN_DIR_TEST_FINAL.txt`.

```bash
cd Praxis_Tier1_Attack_Aware_Alignment_v3

# 1. Check file integrity first (on Windows, use Git Bash)
sha256sum -c --ignore-missing SHA256SUMS_package.txt

# 2. Install the analysis dependencies
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements-analysis.txt

# 3. Final neural tables (Tables 4-13 to 4-15) from the frozen per-event predictions
python code/41_reproduce_final_neural_tables_v33.py \
  --predictions final_reproduction/final_neural_test_predictions_v33.csv \
  --reference-inference rgcn_ablation/v33_final/seed_level_inference_v33.csv

# 4. Coupling-grid corrections (Tables 4-8 to 4-10) from the exported grid predictions
python code/40_review_corrections_v31.py --grid-dir review_corrections_v31 --repeats 500
```

What to expect:

- `--ignore-missing` skips package files that are not posted here. A file edited after packaging shows as `FAILED`. The repository's `.gitattributes` turns off line-ending conversion, so the checksums also hold in a Windows clone.
- Script 41 writes `reproduced_*.csv` to the current folder; the saved copies are in `final_reproduction/`. It prints the largest difference from the saved seed-level inference. In the recorded test, every deterministic quantity (per-run AUPRC, paired differences, seed means, SDs, t-intervals, p-values and seeds positive) matched to within 1.2 × 10⁻¹⁵. Counts of bin-cluster intervals above or below zero can differ by one seed for H3′ and for event-only minus Transformer; that is Monte Carlo variation, and no decision depends on those counts.
- Script 40 rewrites its outputs inside `review_corrections_v31/`. In the recorded test, `h1prime_paired_jitter_loss_aggregate_v31.csv`, `h5_fusion_vs_best_unimodal_v31.csv` and `h1prime_label_freeze_audit_v31.csv` matched with a maximum difference of 0.0. Use `git diff` to compare the rewritten files with the committed ones.

## Full rerun

### Environment

- **Analysis scripts:** tested on Linux with Python 3.12.3. Minimum versions are in `requirements-analysis.txt` and the exact tested versions in `requirements-analysis-tested.txt`.
- **Neural notebooks:** run on Windows 11 with Python 3.13.2 and the CPU build of torch 2.8.0, installed from `requirements-models.txt`. The notebooks run headless through v2: `code/run_notebook_code.py`.
- The reported alignment, tabular and coupling runs used a Linux container with 3 GB of RAM and one CPU core; the neural-model runs used a Windows computer. The hardware differed, so speeds are not comparable across model families.

### Setup

1. Put the four folders under one parent folder, for example `C:\Praxis`.
2. Copy every file in v3: `code/` into v2: `code/`. The v3 scripts import v2 helper functions from their own folder.
3. Copy the two v3.3 notebooks from v3: `rgcn_ablation/` (`06_rgcn_leakage_safe_v3_trained.ipynb` and notebook 09 v3.3) into v2: `notebooks/`.
4. Place the data as described in [Data](#data).

The PowerShell commands use these variables:

```powershell
$PX = "C:\Praxis"
$PKG = "$PX\Praxis_NF3_Leakage_Safe_Reanalysis_v2"
$DEL = "$PX\Praxis_Tier1_Attack_Aware_Alignment_v3"
$ROOT = "$PX\torch_staging_root"      # staging folder for the regime-target neural runs
$CELLS = "$PX\coupling_cells"          # materialized coupling cells
```

### Recorded commands

Every numbered script in the v3 and v2 `code/` folders except 42 lists its arguments with `--help`.

- **Alignment:** `code/run_alignments_v3.sh` holds the three alignment runs (the primary cross-regime design, the within-session audit and the snapshot-basis audit). Set `PR`, `ZIP` and `OUT` at the top to your own folders; the paths there come from the original run environment.
- **Regime-target neural runs:** stage with `code/32_stage_torch_reruns_v3.py`, then run `torch_rerun_staging/run_torch_reruns_v3.sh` (Transformer and R-GCN, seeds 42–46). Set `ROOT` and `PKG` at the top.
- **Intensity cells, final R-GCN and explanation checks:** materialize the cells with `code/35_materialize_coupling_cell_v3.py`, run the Transformer through notebook 05, then run the final entry point, which checks the notebook versions before it runs:

  ```powershell
  & "$DEL\rgcn_ablation\run_rgcn_v33_normalization_rerun.ps1"
  ```

  `rgcn_ablation/README_rgcn_ablation_v3.md` documents the training-budget launcher (`run_rgcn_ablation_v3.ps1`) and the `38_` aggregation command.
- **Per-event prediction export:**

  ```powershell
  python "$PKG\code\export_final_neural_predictions_v33.py" --cells-root "$CELLS" --step1-root "$ROOT\data\processed\point_event_v2" --position-bank "$DEL\step2_coupling_study\position_bank_cross_regime_v3.csv" --alignment-dir "$DEL\alignment_v3_cross_regime" --output "$PX\final_neural_test_predictions_v33.csv"
  ```

- **Checkpoints for archiving:** `& "$DEL\rgcn_ablation\collect_final_checkpoints_v3.ps1"`

### Settings

- Seeds: regime model seeds 42–46; intensity placement index i = 0–9 with model seed 42 + i; five placements per tabular grid cell.
- Grid: κ = 0, 0.1, 0.25, 0.5, 0.75, 1.0 and σ = 0, 60, 120, 300 s. Gaussian shifts are rounded to 30 s and clipped, and true positions and labels stay fixed across σ.
- Thresholds are chosen on validation data. The intensity median comes from the training data.
- All other settings are in Appendix B.1 (Table B-1).

## Data

Neither raw data set is in this repository.

**Flightradar24 (aircraft).** The 2025 archive from the Flightradar24 API (Historic flight positions light, API version 1): 365 daily JSONL files holding 105,120 five-minute snapshots around 61.1743° N, 149.9982° W (Ted Stevens Anchorage International Airport). Flightradar24's terms govern how this data may be shared, so the raw files are not posted, and neither is any table with aircraft identifiers or positions, such as the point-event tables, the event-to-block mapping and the explanation case studies. Committee members can request them from the author.

To rebuild from raw data, use your own API access. nb: `Flightradar_data_5min_12mo.py` downloads the snapshots and reads the token from the `FR24_API_TOKEN` environment variable. Notebook 01 reads the daily files from nb: `data/fr24_raw_5min/`, as set by `raw_data_dir` in `config/point_event_v1.yaml`.

**NF-UNSW-NB15-v3 (network).** 2,365,424 NetFlow records from two 2015 capture sessions, published by The University of Queensland at <https://doi.org/10.48610/6e0eda1>. Download it from there; it is not redistributed here. The alignment scripts read it from inside a zip archive passed as `--recovery-zip`, at the member path `Project/Jupyter Notebook/data/Netflow/NF-UNSW-NB15-v3.csv`, and `--project-root` points to the nb folder. The flow-level extracts (`event_flow_edges_primary_bounded_v*.csv` and `nf3_flow_labels_used_v*.csv.gz` from the alignment scripts, and `selected_event_flow_edges_v2.csv` from notebook 06 v2) are regenerated on a rerun and are not posted.

**Included:** audit summaries, source-block manifests, model results, frozen predictions with resampling-bin identifiers, checkpoints and checksums. The frozen prediction files identify events only by hashed event IDs. The repository's `.gitignore` lists every excluded file pattern.

## What has been verified

From Appendix B.2 of the praxis:

> The recorded clean-directory test verifies saved prediction-based table reproduction, deterministic aggregate values, and package checksums. It does not independently reproduce raw-data alignment, model fitting, or execution of the explanation notebooks.

## Notes on the notebooks

- The v2 and v3 notebooks ran headless through v2: `code/run_notebook_code.py`, which runs the code cells without saving their outputs, so GitHub shows their code only. Their results are the files in the output folders listed in the [Map of the repository](#map-of-the-repository). Notebooks 02 and 03 are also saved without outputs.
- Notebook 01 keeps its saved outputs, except four that listed individual aircraft (sample rows of the snapshot and event tables, and the one duplicate rejection), which were cleared before posting.
- A fresh run on other hardware or library versions may not match digit for digit, which is why the frozen predictions and `NUMBERS_FINAL.json` are the reference.
- The final R-GCN notebook (06, v3.3) and the final intensity explanation notebook (09, v3.3) are in v3: `rgcn_ablation/`. The regime explanation run uses the original notebook 09 in v2: `notebooks/`, because the overlap scale is the same for the regime analysis.
- Script 39 runs notebook 09 v3.3 when that file is in v2: `notebooks/`, and the original notebook 09 otherwise. To repeat the regime run as recorded, stage it before copying v3.3 into that folder.
- Notebooks 04–09 in the nb folder are earlier versions that read the v1 data; the copies in v2: `notebooks/` replace them, so they are not part of this repository.

## Citation

Roberts, D. (2027). *An audited framework for aligning and fusing low-altitude aircraft events with network telemetry under controlled coupling* [Doctor of Engineering praxis, The George Washington University].

Data:

- Flightradar24 AB. (n.d.). *Flightradar24 API: Historic flight positions light* (API Version 1) [Data set]. <https://fr24api.flightradar24.com>
- Luay, M., Layeghy, S., Hosseininoorbin, S., Sarhan, M., Moustafa, N., & Portmann, M. (2025). *NF-UNSW-NB15-v3* [Data set]. The University of Queensland. <https://doi.org/10.48610/6e0eda1>

## License

Code: [TODO: choose a license, for example MIT]. The license covers the code only, not the Flightradar24 or NF-UNSW-NB15-v3 data.
