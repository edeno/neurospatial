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


@pytest.fixture
def out_and_back_track():
    """Out-and-back trajectory on a thin 2D linear-track environment.

    Builds a horizontal corridor (a linear track) with ``home`` and ``reward``
    regions at the two ends, then synthesizes a single out-and-back run: the
    animal travels ``home -> reward`` (outbound), then ``reward -> home``
    (inbound). The midpoint sample sits in neither end region (the turnaround),
    so a correct labeler produces a contiguous ``"outbound"`` stretch, then
    ``"other"`` at the turn, then a contiguous ``"inbound"`` stretch.

    Returns
    -------
    dict
        Keys:

        - ``env`` : the linear-track Environment with ``home`` / ``reward``.
        - ``position_bins`` : bin index per trajectory sample.
        - ``positions`` : (n_samples, 2) continuous trajectory.
        - ``times`` : (n_samples,) timestamps, strictly increasing.
        - ``start_region`` : ``"home"``.
        - ``end_region`` : ``"reward"``.
    """
    # Thin corridor sampled densely along x so binning is well-defined.
    x_samples = np.linspace(0.0, 100.0, 200)
    samples = np.column_stack([x_samples, np.zeros_like(x_samples)])
    env = Environment.from_samples(samples, bin_size=5.0)
    env.regions.add("home", polygon=Point(0.0, 0.0).buffer(8.0))
    env.regions.add("reward", polygon=Point(100.0, 0.0).buffer(8.0))

    # Out-and-back: home -> reward (outbound), reward -> home (inbound).
    x_out = np.linspace(0.0, 100.0, 50)
    x_back = np.linspace(100.0, 0.0, 50)
    x_traj = np.concatenate([x_out, x_back])
    positions = np.column_stack([x_traj, np.zeros_like(x_traj)])
    position_bins = env.bin_at(positions)
    times = np.linspace(0.0, 10.0, len(positions))

    return {
        "env": env,
        "position_bins": position_bins,
        "positions": positions,
        "times": times,
        "start_region": "home",
        "end_region": "reward",
    }


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
