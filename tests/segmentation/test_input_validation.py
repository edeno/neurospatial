"""Segmentation detectors refuse malformed trajectories instead of guessing.

A float bin array would be truncated to unrelated bins, a bin past the
environment would raise a bare IndexError or wrap, and unsorted, repeated or
non-finite times would silently reorder or drop events.
"""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import box

from neurospatial import BinIndexOutOfRangeError, Environment, Regions
from neurospatial.behavior.segmentation import (
    detect_goal_directed_runs,
    detect_laps,
    detect_region_crossings,
    detect_runs_between_regions,
    running_direction_labels,
    segment_trials,
)


@pytest.fixture(scope="module")
def env() -> Environment:
    xs = np.linspace(0.0, 100.0, 401)
    env = Environment.from_samples(np.c_[xs, np.zeros_like(xs)], bin_size=2.0)
    regions = Regions(env.regions.values())
    regions.add("start", polygon=box(-5.0, -5.0, 5.0, 5.0))
    regions.add("goal", polygon=box(95.0, -5.0, 105.0, 5.0))
    return env.with_regions(regions)


DETECTORS = {
    "detect_region_crossings": lambda b, t, env: detect_region_crossings(
        b, t, env, region_name="goal"
    ),
    "detect_runs_between_regions": lambda b, t, env: detect_runs_between_regions(
        b, t, env, source="start", target="goal"
    ),
    "segment_trials": lambda b, t, env: segment_trials(
        b, t, env, start_region="start", end_regions=["goal"]
    ),
    "detect_laps": lambda b, t, env: detect_laps(b, t, env),
    "detect_goal_directed_runs": lambda b, t, env: detect_goal_directed_runs(
        b, t, env, goal_region="goal"
    ),
    "running_direction_labels": lambda b, t, env: running_direction_labels(
        b, t, env, start_region="start", end_regions="goal"
    ),
}


def _walk(env):
    x = np.linspace(0.0, 98.0, 50)
    return env.bin_at(np.c_[x, np.zeros_like(x)]), np.arange(50) / 10.0


@pytest.mark.parametrize("name", DETECTORS)
def test_float_bins_are_refused(env, name):
    bins, times = _walk(env)
    with pytest.raises(TypeError, match="integer bin indices"):
        DETECTORS[name](bins.astype(float) + 0.4, times, env)


@pytest.mark.parametrize("name", DETECTORS)
@pytest.mark.parametrize("bad", ["past_last", "below_minus_one"])
def test_bins_outside_the_environment_are_refused(env, name, bad):
    bins, times = _walk(env)
    bins[10] = env.n_bins if bad == "past_last" else -2
    with pytest.raises(BinIndexOutOfRangeError, match="out of range"):
        DETECTORS[name](bins, times, env)


@pytest.mark.parametrize("name", DETECTORS)
@pytest.mark.parametrize("problem", ["unsorted", "repeated", "nan", "inf"])
def test_bad_times_are_refused(env, name, problem):
    bins, times = _walk(env)
    if problem == "unsorted":
        times[[5, 6]] = times[[6, 5]]
    elif problem == "repeated":
        times[6] = times[5]
    else:
        times[5] = np.nan if problem == "nan" else np.inf
    with pytest.raises(ValueError, match="times"):
        DETECTORS[name](bins, times, env)


@pytest.mark.parametrize("name", DETECTORS)
def test_length_mismatch_is_refused(env, name):
    bins, times = _walk(env)
    with pytest.raises(ValueError, match="same length"):
        DETECTORS[name](bins, times[:-1], env)


def test_minus_one_marks_a_missing_sample(env):
    """-1 (outside, from env.bin_at) is accepted and treated as missing."""
    bins, times = _walk(env)
    bins[20] = -1
    (crossing,) = detect_region_crossings(
        bins, times, env, region_name="goal", direction="entry"
    )
    assert crossing.time == times[-2]
