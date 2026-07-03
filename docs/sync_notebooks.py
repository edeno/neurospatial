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

# Prune stale mirrors: remove a synced numbered-example file in docs/examples/
# whose source no longer exists in examples/ (e.g. a deleted or renamed
# example). The prune scan is restricted to the numbered-example naming
# convention the sync actually produces (``NN_*.ipynb`` / ``NN_*.py``, e.g.
# ``23_path_progression.ipynb``), so hand-maintained docs-only files -- such as
# index.md, a local conftest.py, or any non-numbered helper -- are never
# candidates for deletion.
prune_patterns = ("[0-9][0-9]_*.ipynb", "[0-9][0-9]_*.py")
pruned = 0
for pattern in prune_patterns:
    for mirror in sorted(docs_examples_dir.glob(pattern)):
        source = examples_dir / mirror.name
        if not source.exists():
            print(f"Pruning stale {mirror.name} (no source in examples/)...")
            mirror.unlink()
            pruned += 1

print(f"✓ Synced {synced} example files")
print(f"✓ Pruned {pruned} stale mirror files")
