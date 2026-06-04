"""Tests for the v0.6 detect_region_crossings transitional arg-order (Task 1.4e).

Target signature (0.6+): ``(position_bins, times, env, *, region_name, ...)``.
Old signature (deprecated, removed in 0.7):
``(position_bins, times, region_name, env, *, ...)``.

The transitional dispatch must (a) accept the old 4-positional call with a
DeprecationWarning, (b) accept the new ``(..., env, region_name=...)`` form
warning-free, and (c) produce identical crossings either way.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from shapely.geometry import Point

from neurospatial import Environment
from neurospatial.behavior.segmentation import detect_region_crossings


def _make_env_and_trajectory() -> tuple[Environment, np.ndarray, np.ndarray]:
    x = np.linspace(0, 100, 50)
    y = np.linspace(0, 100, 50)
    xx, yy = np.meshgrid(x, y)
    positions = np.column_stack([xx.ravel(), yy.ravel()])
    env = Environment.from_samples(positions, bin_size=5.0)
    env.regions.add("goal", polygon=Point(50.0, 50.0).buffer(10.0))

    traj_x = np.array([10.0, 30.0, 50.0, 70.0, 50.0, 30.0])
    traj_y = np.array([50.0, 50.0, 50.0, 50.0, 50.0, 50.0])
    trajectory = np.column_stack([traj_x, traj_y])
    position_bins = env.bin_at(trajectory)
    times = np.arange(len(trajectory), dtype=float)
    return env, position_bins, times


def test_new_order_is_warning_free() -> None:
    env, position_bins, times = _make_env_and_trajectory()
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        crossings = detect_region_crossings(
            position_bins, times, env, region_name="goal", direction="both"
        )
    assert len(crossings) > 0


def test_old_positional_order_warns() -> None:
    env, position_bins, times = _make_env_and_trajectory()
    with pytest.warns(DeprecationWarning):
        crossings = detect_region_crossings(
            position_bins, times, "goal", env, direction="both"
        )
    assert len(crossings) > 0


def test_old_and_new_order_identical() -> None:
    env, position_bins, times = _make_env_and_trajectory()

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        new = detect_region_crossings(
            position_bins, times, env, region_name="goal", direction="both"
        )

    with pytest.warns(DeprecationWarning):
        old = detect_region_crossings(
            position_bins, times, "goal", env, direction="both"
        )

    assert [(c.time, c.direction, c.bin_index) for c in old] == [
        (c.time, c.direction, c.bin_index) for c in new
    ]


def test_missing_region_name_in_new_order_raises() -> None:
    env, position_bins, times = _make_env_and_trajectory()
    with pytest.raises(TypeError, match="region_name"):
        detect_region_crossings(position_bins, times, env)
