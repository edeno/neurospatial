"""Shared synthetic fixtures for behavior subsystem tests.

These builders are reused across the binning / dt-correctness regression
tests so that the same out-of-environment trajectory, duplicate-timestamp
arrays, and disconnected environment are not copy-pasted across modules.
"""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Point

from neurospatial import Environment


@pytest.fixture
def grid_env_with_last_bin_region():
    """2D grid environment with a polygon region covering the *last* bin.

    Returns a tuple ``(env, region_name, last_bin)``. The region is built so
    that ``regions_to_mask(env, [region_name])[last_bin]`` is True, which is
    exactly the condition under which a wrapped ``-1`` index (numpy maps ``-1``
    to the last element) would be mis-attributed to the region. The rest of the
    environment is outside the region.
    """
    x = np.linspace(0, 100, 50)
    y = np.linspace(0, 100, 50)
    xx, yy = np.meshgrid(x, y)
    sample_positions = np.column_stack([xx.ravel(), yy.ravel()])
    env = Environment.from_samples(sample_positions, bin_size=5.0)

    last_bin = env.n_bins - 1
    center = env.bin_centers[last_bin]
    env.regions.add("last_region", polygon=Point(*center).buffer(6.0))

    # Sanity: the region really does cover the last bin (so the -1 wrap would
    # otherwise land inside the region).
    from neurospatial.ops.binning import regions_to_mask

    assert regions_to_mask(env, ["last_region"])[last_bin]

    return env, "last_region", last_bin


def duplicate_timestamps(n: int = 11, total: float = 5.0) -> np.ndarray:
    """Return strictly-increasing timestamps with one duplicated element.

    Builds ``np.linspace(0, total, n)`` then overwrites index 5 to equal index
    4, producing a single zero ``dt`` (a duplicate frame). Used to exercise the
    per-sample dt guards.
    """
    times = np.linspace(0.0, total, n)
    times[5] = times[4]
    return times


def nonfinite_timestamps(n: int = 11, total: float = 5.0) -> np.ndarray:
    """Return otherwise-valid timestamps with one ``inf`` element."""
    times = np.linspace(0.0, total, n)
    times[5] = np.inf
    return times
