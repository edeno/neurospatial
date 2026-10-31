"""Tests for ``neurospatial._typing`` and Environment combine refusals.

Covers:

- ``import neurospatial._typing`` does not drag in pynapple / pynwb.
- ``Environment.combine`` refuses polar charts and non-environments.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from neurospatial import Environment


@pytest.fixture
def polar_env() -> Environment:
    return Environment.from_polar_egocentric(
        distance_range=(0.0, 50.0),
        angle_range=(-np.pi, np.pi),
        distance_bin_size=10.0,
        angle_bin_size=np.pi / 6,
    )


def test_combine_refuses_polar_charts(polar_env: Environment) -> None:
    """A Cartesian union cannot interpret polar columns as Cartesian axes."""
    with pytest.raises(TypeError, match="Cartesian"):
        Environment.combine({"polar": polar_env})


def test_combine_rejects_non_environment() -> None:
    with pytest.raises(TypeError, match="Environment"):
        Environment.combine({"invalid": object()})


def test_typing_import_is_light() -> None:
    """``import neurospatial._typing`` must not import pynapple / pynwb."""
    code = (
        "import sys; import neurospatial._typing; "
        "assert 'pynapple' not in sys.modules, 'pynapple imported'; "
        "assert 'pynwb' not in sys.modules, 'pynwb imported'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
