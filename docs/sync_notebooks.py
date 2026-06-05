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

# Prune stale mirrors: remove any synced-type file in docs/examples/ whose
# source no longer exists in examples/ (e.g. a deleted or renamed example).
# Only *.ipynb / *.py are pruned -- hand-maintained files such as index.md
# are never touched.
pruned = 0
for pattern in patterns:
    for mirror in sorted(docs_examples_dir.glob(pattern)):
        source = examples_dir / mirror.name
        if not source.exists():
            print(f"Pruning stale {mirror.name} (no source in examples/)...")
            mirror.unlink()
            pruned += 1

print(f"✓ Synced {synced} example files")
print(f"✓ Pruned {pruned} stale mirror files")
