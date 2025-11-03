#!/usr/bin/env python3
"""Sync example notebooks from examples/ to docs/examples/ for documentation build."""

import shutil
from pathlib import Path

# Paths
examples_dir = Path(__file__).parent.parent / "examples"
docs_examples_dir = Path(__file__).parent / "examples"

# Sync all notebooks
for notebook in examples_dir.glob("*.ipynb"):
    dest = docs_examples_dir / notebook.name
    print(f"Syncing {notebook.name}...")
    shutil.copy2(notebook, dest)

print(f"✓ Synced {len(list(examples_dir.glob('*.ipynb')))} notebooks")
