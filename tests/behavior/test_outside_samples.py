"""A sample outside the environment (bin ``-1``) is in no bin.

``-1`` must never index ``bin_centers`` or a per-bin array, where it would
silently stand for the last bin.
"""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import box

from neurospatial import Environment
from neurospatial.behavior.segmentation import (
    detect_goal_directed_runs,
    trajectory_similarity,
)
from neurospatial.behavior.trajectory import compute_home_range
from neurospatial.regions import Regions


@pytest.fixture(scope="module")
def env() -> Environment:
    x, y = np.meshgrid(np.arange(0.0, 30.0), np.arange(0.0, 30.0))
    return Environment.from_samples(np.c_[x.ravel(), y.ravel()], bin_size=3.0)


METHODS = ["jaccard", "correlation", "hausdorff", "dtw"]


@pytest.mark.parametrize("method", METHODS)
def test_similarity_outside_sample_is_not_the_last_bin(env, method):
    reference = np.array([0, 1, 2, 3, 4])
    with_outside = np.array([0, 1, 2, -1, 4])
    with_last_bin = np.array([0, 1, 2, env.n_bins - 1, 4])

    outside = trajectory_similarity(reference, with_outside, env, method=method)
    last_bin = trajectory_similarity(reference, with_last_bin, env, method=method)

    assert outside != last_bin


@pytest.mark.parametrize("method", METHODS)
def test_similarity_ignores_outside_samples(env, method):
    """Adding outside samples to a trajectory leaves its similarity unchanged."""
    first = np.array([0, 1, 2, 3, 4])
    second = np.array([2, 3, 4, 5, 6])
    padded = np.array([-1, 2, 3, 4, 5, 6, -1])
    if method == "correlation":
        # Correlation pairs samples by time, so pad both trajectories alike.
        first = np.array([-1, 0, 1, 2, 3, 4, -1])

    assert trajectory_similarity(
        first, padded, env, method=method
    ) == trajectory_similarity(first[first >= 0], second, env, method=method)


def test_similarity_outside_samples_are_not_shared_places(env):
    """Two trajectories that only share time outside have no overlap."""
    similarity = trajectory_similarity(
        np.array([-1, -1, 5]), np.array([-1, -1, 6]), env, method="jaccard"
    )

    assert similarity == 0.0


def test_similarity_refuses_a_trajectory_with_no_place(env):
    with pytest.raises(ValueError, match="no sample inside the environment"):
        trajectory_similarity(np.array([-1, -1]), np.array([0, 1]), env)


def test_similarity_refuses_an_index_that_is_not_a_bin(env):
    with pytest.raises(ValueError, match="out of range"):
        trajectory_similarity(np.array([env.n_bins]), np.array([0, 1]), env)


@pytest.mark.parametrize("timed", [False, True])
def test_home_range_leaves_out_outside_samples(timed):
    position_bins = np.array([-1] * 50 + [3] * 5 + [4] * 6)
    times = np.arange(len(position_bins)) * 0.1 if timed else None

    home_range = compute_home_range(position_bins, times=times, percentile=100.0)

    np.testing.assert_array_equal(home_range, [4, 3])


def test_home_range_refuses_negative_indices_other_than_minus_one():
    with pytest.raises(ValueError, match="not a bin"):
        compute_home_range(np.array([-2, 3, 4]))


@pytest.fixture(scope="module")
def strip() -> Environment:
    x = np.linspace(0, 100, 200)
    samples = np.vstack(
        [np.c_[x, np.full_like(x, 1.0)], np.c_[x, np.full_like(x, 2.0)]]
    )
    env = Environment.from_samples(samples, bin_size=2.0)
    regions = Regions()
    regions.add("goal", polygon=box(90, -5, 101, 5))
    return env.with_regions(regions)


def _approach(strip):
    positions = np.c_[np.linspace(1, 80, 40), np.full(40, 1.0)]
    return strip.bin_at(positions), np.linspace(0, 10, 40)


def test_goal_directed_run_skips_an_outside_sample_like_a_deleted_row(strip):
    position_bins, times = _approach(strip)
    position_bins[20] = -1
    kept = np.delete(np.arange(len(times)), 20)

    runs = detect_goal_directed_runs(
        position_bins, times, strip, goal_region="goal", min_progress=10.0
    )
    deleted = detect_goal_directed_runs(
        position_bins[kept], times[kept], strip, goal_region="goal", min_progress=10.0
    )

    assert [(run.start_time, run.end_time) for run in runs] == [
        (run.start_time, run.end_time) for run in deleted
    ]
    assert all((run.bins >= 0).all() for run in runs)


def test_goal_directed_run_ends_at_an_outside_sample_beyond_max_gap(strip):
    position_bins, times = _approach(strip)
    position_bins[20] = -1

    runs = detect_goal_directed_runs(
        position_bins,
        times,
        strip,
        goal_region="goal",
        min_progress=10.0,
        max_gap=1.5 * (times[1] - times[0]),
    )

    assert [(run.start_time, run.end_time) for run in runs] == [
        (times[0], times[19]),
        (times[21], times[-1]),
    ]
    assert all((run.bins >= 0).all() for run in runs)


def test_goal_directed_runs_without_a_place_are_empty(strip):
    _, times = _approach(strip)

    runs = detect_goal_directed_runs(
        np.full(len(times), -1), times, strip, goal_region="goal"
    )

    assert runs == []
