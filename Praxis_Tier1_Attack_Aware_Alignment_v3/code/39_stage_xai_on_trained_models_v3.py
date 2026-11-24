#!/usr/bin/env python3
"""Stage notebook 09 (XAI validation) against the FINAL trained checkpoints.

Notebook 09 reads a v2-layout root: nf3_alignment_v2 (bank), modeling_v4 (splits, manifest, target definition,
permutation_importance_v4.csv), transformer_v2/models/fused_temporal_transformer.pt and rgcn_leakage_safe_v2/models/
full_lag_relation_rgcn.pt with their predictions, attributions, selected edges and node manifest. This script builds a
staging root for (a) a coupling cell or (b) the Step 1 regime root in which `rgcn_leakage_safe_v2` IS the trained
R-GCN folder, so the XAI package validates the models actually reported. `permutation_importance_v4.csv` (a tabular
regime-target baseline the notebook compares against) is copied from the v2 package and flagged as such.
"""
import argparse, shutil, os
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--source-root", type=Path, required=True, help="cell root (has data/processed/point_event_v2/...) or Step 1 staging root")
p.add_argument("--trained-rgcn-subdir", default="rgcn_trained_v3", help="folder with the trained R-GCN outputs (models/full_lag_relation_rgcn.pt)")
p.add_argument("--transformer-subdir", default="transformer_v2")
p.add_argument("--package-root", type=Path, required=True)
p.add_argument("--staging-root", type=Path, required=True)
a = p.parse_args()

src = a.source_root / "data/processed/point_event_v2"
dst = a.staging_root / "data/processed/point_event_v2"
dst.mkdir(parents=True, exist_ok=True)
def link_or_copy(s: Path, d: Path):
    if d.exists() or d.is_symlink():
        shutil.rmtree(d) if d.is_dir() and not d.is_symlink() else d.unlink()
    try: os.symlink(s.resolve(), d)
    except OSError: shutil.copytree(s, d) if s.is_dir() else shutil.copy2(s, d)
for name in ("nf3_alignment_v2", "modeling_v4"):
    assert (src / name).exists(), src / name; link_or_copy(src / name, dst / name)
link_or_copy(src / a.transformer_subdir, dst / "transformer_v2")
tr = src / a.trained_rgcn_subdir
assert (tr / "models" / "full_lag_relation_rgcn.pt").exists(), f"trained checkpoint missing under {tr}"
link_or_copy(tr, dst / "rgcn_leakage_safe_v2")
perm = dst / "modeling_v4" / "permutation_importance_v4.csv"
if not perm.exists():
    if (dst / "modeling_v4").is_symlink():  # materialize a real folder so we can add the file
        real = Path(os.readlink(dst / "modeling_v4")); (dst / "modeling_v4").unlink(); shutil.copytree(real, dst / "modeling_v4")
    shutil.copy2(a.package_root / "tabular_modeling_v4" / "permutation_importance_v4.csv", perm)
    (dst / "modeling_v4" / "PERMUTATION_IMPORTANCE_IS_REGIME_TARGET_BASELINE.txt").write_text("permutation_importance_v4.csv was copied from the v2 regime-target tabular run; it is a baseline for comparison only.\n")
nb09 = "09_xai_validation_leakage_safe_v33.ipynb" if (a.package_root / "notebooks" / "09_xai_validation_leakage_safe_v33.ipynb").exists() else "09_xai_validation_and_analyst_evidence_leakage_safe_v2.ipynb"
(a.staging_root / "run_xai.ps1").write_text(f"""$ROOT = "{a.staging_root.resolve()}"; $PKG = "{a.package_root.resolve()}"; Set-Location $ROOT
python "$PKG\\code\\run_notebook_code.py" "$PKG\\notebooks\\{nb09}" --cwd "$ROOT"
""")
print("staged:", a.staging_root, "| R-GCN checkpoint:", tr / "models/full_lag_relation_rgcn.pt")
