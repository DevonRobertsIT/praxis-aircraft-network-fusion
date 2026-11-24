#!/usr/bin/env python3
"""Execute code cells from a notebook sequentially in one Python namespace."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--cwd", type=Path, required=True)
    args = parser.parse_args()
    notebook_path = args.notebook.resolve()
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.chdir(args.cwd.resolve())
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__", "__file__": str(notebook_path)}
    for cell_number, cell in enumerate(notebook.get("cells", []), start=1):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if not source.strip():
            continue
        print(f"\n--- executing code cell {cell_number} ---", flush=True)
        exec(compile(source, f"{notebook_path}#cell-{cell_number}", "exec"), namespace)


if __name__ == "__main__":
    main()
