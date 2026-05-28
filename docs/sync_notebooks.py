#!/usr/bin/env python3
"""Sync example notebooks from examples/ to docs/examples/ for documentation build."""

import shutil
from pathlib import Path

# Paths
examples_dir = Path(__file__).parent.parent / "examples"
docs_examples_dir = Path(__file__).parent / "examples"

docs_examples_dir.mkdir(parents=True, exist_ok=True)

patterns = ("*.ipynb", "*.py")
synced = 0

for pattern in patterns:
    for source in sorted(examples_dir.glob(pattern)):
        dest = docs_examples_dir / source.name
        print(f"Syncing {source.name}...")
        shutil.copy2(source, dest)
        synced += 1

print(f"✓ Synced {synced} example files")
