#!/usr/bin/env python3
"""Sync example notebooks from examples/ to docs/examples/ for documentation build."""

import re
import shutil
from pathlib import Path

# Paths
examples_dir = Path(__file__).parent.parent / "examples"
docs_examples_dir = Path(__file__).parent / "examples"

docs_examples_dir.mkdir(parents=True, exist_ok=True)


def sync_catalog() -> None:
    """Keep the complete catalog consistent with the notebook introductions."""
    rows = ["| Notebook | Estimated time | Useful background |", "| --- | --- | --- |"]
    for source in sorted(examples_dir.glob("[0-9][0-9]_*.py")):
        text = source.read_text(encoding="utf-8")
        title = re.search(r"^# # (.+)$", text, flags=re.MULTILINE)
        estimates = re.findall(
            r"^# \*\*Estimated time\*\*:[ \t]*([^\n]+)", text, flags=re.MULTILINE
        )
        prerequisites = re.search(
            r"^# \*\*Prerequisites\*\*:[ \t]*(.*(?:\n# [^\n]+)*)",
            text,
            flags=re.MULTILINE,
        )
        if title is None or len(estimates) != 1 or prerequisites is None:
            raise ValueError(
                f"{source.name} needs a title, one **Estimated time**: entry and "
                "**Prerequisites**: links (or None) before syncing the catalog."
            )
        dependencies = re.findall(
            r"\[[^\]]+\]\(([^)]+\.ipynb)\)", prerequisites.group(1)
        )
        dependency_links = (
            ", ".join(f"[{Path(target).name[:2]}]({target})" for target in dependencies)
            or "None"
        )
        notebook = source.with_suffix(".ipynb").name
        rows.append(
            f"| [{source.name[:2]}. {title.group(1)}]({notebook}) | "
            f"{estimates[0].rstrip('.')} | {dependency_links} |"
        )
    readme = examples_dir / "README.md"
    text = readme.read_text(encoding="utf-8")
    start = "<!-- notebook-catalog:start -->"
    end = "<!-- notebook-catalog:end -->"
    before, remainder = text.split(start, 1)
    _, after = remainder.split(end, 1)
    readme.write_text(
        before + start + "\n\n" + "\n".join(rows) + "\n\n" + end + after,
        encoding="utf-8",
    )
    print(f"Synced catalog for {len(rows) - 2} notebooks")


# Validate metadata before publishing any mirrors.
sync_catalog()

# The notebook collection: numbered examples and the private helpers they
# import, such as _style.py. Scripts run from a checkout, such as
# try_annotation.py, are not part of the rendered collection.
patterns = ("[0-9][0-9]_*.ipynb", "[0-9][0-9]_*.py", "_*.py")
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
# ``03_trajectories_and_behavior.ipynb``), so hand-maintained docs-only files -- such as
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
