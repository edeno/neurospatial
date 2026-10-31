"""Warnings point at the scientist's line, however the package is imported."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import neurospatial

_SCRIPT = """
import warnings

import numpy as np

import neurospatial
from neurospatial import Environment

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    Environment.from_samples(np.array([[0.0, 0.0], [1.0, 1.0]]), bin_size=5.0)
print(neurospatial.__file__)
print(caught[0].filename)
"""


def test_warning_points_at_caller_when_imported_through_a_symlink(tmp_path) -> None:
    """A symlinked sys.path entry gives unresolved frame filenames, so the
    package directory must not be resolved either."""
    src = Path(neurospatial.__file__).parent.parent
    link = tmp_path / "src-link"
    try:
        link.symlink_to(src, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available")
    script = tmp_path / "caller.py"
    script.write_text(_SCRIPT)

    result = subprocess.run(
        [sys.executable, str(script)],
        env={**os.environ, "PYTHONPATH": str(link)},
        capture_output=True,
        text=True,
        check=True,
    )

    imported_from, warned_at = result.stdout.split()
    assert Path(imported_from).is_relative_to(link)
    assert Path(warned_at) == script
