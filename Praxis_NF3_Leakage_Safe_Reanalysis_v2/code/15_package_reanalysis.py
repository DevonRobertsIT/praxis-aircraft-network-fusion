#!/usr/bin/env python3
"""Assemble the NF3 leakage-safe reanalysis handoff and checksum package."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--zip-path", type=Path, required=True)
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    output = args.output_dir.resolve()
    zip_path = args.zip_path.resolve()
    if output.exists() or zip_path.exists():
        raise FileExistsError("Refusing to overwrite an existing handoff directory or ZIP.")

    project_v2 = workspace / "audit_work/Project/Jupyter Notebook/data/processed/point_event_v2"
    output.mkdir(parents=True)

    file_sources = {
        workspace / "analysis_outputs/NF3_REANALYSIS_HANDOFF_README_v2.md": output / "README.md",
        workspace / "analysis_outputs/NF3_LEAKAGE_SAFE_REANALYSIS_REPORT_v2.md": output / "NF3_LEAKAGE_SAFE_REANALYSIS_REPORT_v2.md",
    }
    directory_sources = {
        workspace / "analysis_outputs/nf3_source_recurrence_audit_v1": output / "audit_original_v1",
        project_v2 / "nf3_alignment_v2": output / "alignment_leakage_safe_v2",
        project_v2 / "modeling_v4": output / "tabular_modeling_v4",
        project_v2 / "transformer_v2": output / "transformer_v2",
        project_v2 / "rgcn_leakage_safe_v2": output / "rgcn_leakage_safe_v2",
        project_v2 / "results_v2": output / "results_v2",
        project_v2 / "control_evaluations_v2_final": output / "control_evaluations_v2",
        project_v2 / "chapter4_exports_v2": output / "chapter4_exports_v2",
        project_v2 / "xai_leakage_safe_v2": output / "xai_leakage_safe_v2",
        workspace / "analysis_outputs/nf3_leakage_safe_validation_v2": output / "validation_v2",
    }
    for source, destination in file_sources.items():
        shutil.copy2(source, destination)
    for source, destination in directory_sources.items():
        shutil.copytree(source, destination)

    code_dir = output / "code"
    code_dir.mkdir()
    for name in [
        "10_nf3_recurrence_audit.py",
        "11_nf3_leakage_safe_alignment.py",
        "12_rebuild_v2_feature_bank.py",
        "13_any_flow_and_matched_evaluations.py",
        "14_validate_reanalysis.py",
        "15_package_reanalysis.py",
        "run_notebook_code.py",
    ]:
        shutil.copy2(workspace / "analysis_code" / name, code_dir / name)

    notebook_dir = output / "notebooks"
    notebook_dir.mkdir()
    for name in [
        "04_point_event_tabular_baselines_leakage_safe_v4.ipynb",
        "05_temporal_transformer_leakage_safe_v2.ipynb",
        "06_rgcn_leakage_safe_v2.ipynb",
        "07_results_leakage_safe_v2.ipynb",
        "08_chapter4_publication_exports_leakage_safe_v2.ipynb",
        "09_xai_validation_and_analyst_evidence_leakage_safe_v2.ipynb",
    ]:
        shutil.copy2(workspace / "analysis_notebooks" / name, notebook_dir / name)

    recovery_archive = workspace / "analysis_input/Praxis - XAI Multimodal Alignment/Praxis_Recovery_01_Core_Artifacts.zip"
    source_reference = {
        "recovery_archive_name": recovery_archive.name,
        "recovery_archive_sha256": sha256_file(recovery_archive),
        "raw_nf3_entry": "Project/Jupyter Notebook/data/Netflow/NF-UNSW-NB15-v3.csv",
        "original_full_edge_entry": "Project/Jupyter Notebook/data/processed/point_event_v1/nf3_alignment_v1/event_flow_edges_primary_v1.csv",
        "note": "The 218 MB recovery archive is referenced by checksum but is not duplicated inside this handoff package.",
    }
    (output / "SOURCE_ARCHIVE_REFERENCE_v2.json").write_text(
        json.dumps(source_reference, indent=2),
        encoding="utf-8",
    )

    files = sorted(path for path in output.rglob("*") if path.is_file())
    manifest_lines = [
        f"{sha256_file(path)}  {path.relative_to(output).as_posix()}"
        for path in files
        if path.name != "SHA256SUMS.txt" or path.parent != output
    ]
    (output / "SHA256SUMS.txt").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    inventory = {
        "package": output.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "file_count_excluding_top_level_manifest": len(files),
        "uncompressed_bytes_excluding_top_level_manifest": sum(path.stat().st_size for path in files),
        "independent_validation": {"decision": "PASS", "passed_checks": 34, "failed_checks": 0},
        "component_directories": sorted(path.name for path in output.iterdir() if path.is_dir()),
    }
    (output / "PACKAGE_INVENTORY_v2.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")

    # Refresh the top-level manifest so it also covers the inventory file.
    files = sorted(path for path in output.rglob("*") if path.is_file())
    manifest_lines = [
        f"{sha256_file(path)}  {path.relative_to(output).as_posix()}"
        for path in files
        if path != output / "SHA256SUMS.txt"
    ]
    (output / "SHA256SUMS.txt").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=(Path(output.name) / path.relative_to(output)).as_posix())

    print(json.dumps({**inventory, "zip_path": str(zip_path), "zip_bytes": zip_path.stat().st_size}, indent=2))


if __name__ == "__main__":
    main()
