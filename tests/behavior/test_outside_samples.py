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
    detect_laps,
    detect_region_crossings,
    detect_runs_between_regions,
    running_direction_labels,
    segment_trials,
    trajectory_similarity,
)
from neurospatial.behavior.trajectory import compute_home_range
from neurospatial.regions import Regions

# Equal one-unit bins: the home range ranks them by occupancy alone.
_HOME_ENV = Environment.from_samples(
    np.arange(1000.0)[:, None], bin_size=1.0, infer_active_bins=False
)


@pytest.fixture(scope="module")
def env() -> Environment:
    x, y = np.meshgrid(np.arange(0.0, 30.0), np.arange(0.0, 30.0))
    return Environment.from_samples(np.c_[x.ravel(), y.ravel()], bin_size=3.0)


METHODS = ["jaccard", "correlation", "hausdorff", "dtw"]


def _env_for(method: str, env: Environment) -> Environment:
    """Correlation compares positions along a track, so it gets a 1-D one."""
    if method == "correlation":
        return Environment.from_samples(np.arange(0.0, 30.0)[:, None], bin_size=3.0)
    return env


@pytest.mark.parametrize("method", METHODS)
def test_similarity_outside_sample_is_not_the_last_bin(env, method):
    env = _env_for(method, env)
    reference = np.array([0, 1, 2, 3, 4])
    with_outside = np.array([0, 1, 2, -1, 4])
    with_last_bin = np.array([0, 1, 2, env.n_bins - 1, 4])

    outside = trajectory_similarity(reference, with_outside, env, method=method)
    last_bin = trajectory_similarity(reference, with_last_bin, env, method=method)

    assert outside != last_bin


@pytest.mark.parametrize("method", METHODS)
def test_similarity_ignores_outside_samples(env, method):
    """Adding outside samples to a trajectory leaves its similarity unchanged."""
    env = _env_for(method, env)
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


@pytest.mark.parametrize(("timed", "expected"), [(False, [4, 3]), (True, [3, 4])])
def test_home_range_leaves_out_outside_samples(timed, expected):
    """Outside samples add nothing; timed dwell is env.occupancy's.

    Counted, bin 4 has six samples and bin 3 five. Timed, each observed
    interval goes to the bin of its first sample: bin 3 holds its five
    intervals (the last one ends in bin 4) and bin 4 five (its last sample
    starts none), so they tie and are listed in ascending order.
    """
    position_bins = np.array([-1] * 50 + [3] * 5 + [4] * 6)
    times = np.arange(len(position_bins)) * 0.1 if timed else None

    home_range = compute_home_range(
        position_bins, _HOME_ENV, times=times, percentile=100.0
    )

    np.testing.assert_array_equal(home_range, expected)


def test_home_range_dwell_matches_occupancy_across_an_excursion():
    """A 0.33 s excursion outside removes the same time from dwell and occupancy."""
    env = Environment.from_samples(
        np.random.default_rng(0).uniform(0, 100, (4000, 2)), bin_size=10.0
    )
    times = np.arange(300) / 30
    positions = np.column_stack([np.linspace(10, 90, 300), np.full(300, 50.0)])
    positions[100:110] = [500.0, 500.0]
    occupancy = env.occupancy(times, positions)
    assert occupancy.sum() == pytest.approx(9.6)
    home_range = compute_home_range(
        env.bin_at(positions), env, times=times, percentile=100.0
    )
    np.testing.assert_array_equal(np.sort(home_range), np.flatnonzero(occupancy))


def test_home_range_refuses_negative_indices_other_than_minus_one():
    with pytest.raises(ValueError, match="not a bin"):
        compute_home_range(np.array([-2, 3, 4]), _HOME_ENV)


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


@pytest.mark.parametrize("max_gap", [1.0, None, "sampling"])
def test_goal_directed_run_ends_at_an_outside_sample(strip, max_gap):
    """As in env.occupancy, an interval touching an outside sample is not observed.

    The approach splits at the outside sample whatever ``max_gap`` is: it is
    not a missing row to bridge but a position outside the environment.
    """
    position_bins, times = _approach(strip)
    position_bins[20] = -1
    if max_gap == "sampling":
        max_gap = 1.5 * (times[1] - times[0])

    runs = detect_goal_directed_runs(
        position_bins,
        times,
        strip,
        goal_region="goal",
        min_progress=10.0,
        max_gap=max_gap,
    )

    np.testing.assert_array_equal(runs, [[times[0], times[19]], [times[21], times[-1]]])


def test_goal_directed_runs_without_a_place_are_empty(strip):
    _, times = _approach(strip)

    with pytest.warns(UserWarning, match="excluded ALL trajectory intervals"):
        runs = detect_goal_directed_runs(
            np.full(len(times), -1),
            times,
            strip,
            goal_region="goal",
            min_progress=10.0,
        )

    assert runs.shape == (0, 2)


@pytest.fixture(scope="module")
def shuttle(strip):
    """Two round trips between "start" (x < 10) and "goal" (x > 90) at 10 Hz.

    Three samples are outside the environment: halfway back on the first
    return (60), halfway out on the second trip (100), and just before that
    trip enters the goal.
    """
    regions = Regions(strip.regions.values())
    regions.add("start", polygon=box(-5, -5, 10, 5))
    env = strip.with_regions(regions)
    trip = np.linspace(1, 99, 41)
    xs = np.r_[trip, trip[::-1][1:], trip[1:], trip[::-1][1:]]
    times = np.arange(xs.size) * 0.1
    bins = env.bin_at(np.c_[xs, np.ones_like(xs)])
    entry = 81 + int(np.argmax(xs[81:] > 90))
    return env, times, bins, np.array([60, 100, entry - 1])


SHUTTLE_CALLS = {
    "detect_region_crossings": lambda env, b, t, **kw: detect_region_crossings(
        b, t, env, region_name="goal", **kw
    ),
    "detect_runs_between_regions": lambda env, b, t, **kw: (
        detect_runs_between_regions(b, t, env, source="goal", target="start", **kw)
    ),
    "running_direction_labels": lambda env, b, t, **kw: running_direction_labels(
        b, t, env, start_region="start", end_regions="goal", **kw
    ),
    "segment_trials": lambda env, b, t, **kw: segment_trials(
        b, t, env, start_region="start", end_regions="goal", **kw
    ),
}


@pytest.mark.parametrize("name", sorted(SHUTTLE_CALLS))
def test_an_outside_sample_ends_the_run_like_a_declared_break(shuttle, name):
    """As in env.occupancy, an interval touching bin -1 is not observed.

    The result equals that of the recording with the outside row deleted and
    a break declared on the row after it, however short the gap would be.
    """
    env, times, bins, rows = shuttle
    outside = bins.copy()
    outside[rows] = -1
    kept = np.delete(np.arange(times.size), rows)
    breaks = np.isin(kept, rows + 1)
    call = SHUTTLE_CALLS[name]
    with_outside = call(env, outside, times)
    with_break = call(env, bins[kept], times[kept], break_before=breaks)
    if name == "running_direction_labels":
        np.testing.assert_array_equal(with_outside[kept], with_break)
    else:
        assert repr(with_outside) == repr(with_break)


def test_an_outside_sample_censors_the_run_it_interrupts(shuttle):
    """The first return no longer reaches "start": it is censored at the outside sample."""
    env, times, bins, rows = shuttle
    k = rows[0]
    outside = bins.copy()
    outside[k] = -1
    runs = detect_runs_between_regions(
        outside, times, env, source="goal", target="start"
    )
    assert [run.outcome for run in runs] == ["censored", "reached"]
    assert runs[0].end_time == pytest.approx(times[k - 1])


def test_an_outside_sample_ends_a_lap():
    """A lap never spans a sample outside the environment."""
    theta = np.linspace(0, 20 * np.pi, 1801)
    positions = np.column_stack([50 + 30 * np.cos(theta), 50 + 30 * np.sin(theta)])
    times = np.linspace(0, 60, 1801)
    env = Environment.from_samples(positions, bin_size=3.0)
    bins = env.bin_at(positions)
    k = 900
    outside = bins.copy()
    outside[k] = -1
    kept = np.delete(np.arange(times.size), k)
    breaks = np.zeros(kept.size, dtype=bool)
    breaks[k] = True
    laps = detect_laps(outside, times, env, method="auto")
    assert laps
    assert repr(laps) == repr(
        detect_laps(bins[kept], times[kept], env, method="auto", break_before=breaks)
    )
    assert not any(lap.start_time < times[k] < lap.end_time for lap in laps)
