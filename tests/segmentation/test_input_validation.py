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
    env = env.with_metadata(units="cm")
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


def test_unknown_crossing_direction_is_refused(env):
    bins, times = _walk(env)
    with pytest.raises(ValueError, match="direction must be one of"):
        detect_region_crossings(
            bins,
            times,
            env,
            region_name="goal",
            direction="entries",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("min_duration", "max_duration", "match"),
    [
        (-1.0, 10.0, "min_duration must be finite and nonnegative"),
        (np.nan, 10.0, "min_duration must be finite and nonnegative"),
        (5.0, 2.0, "needs max_duration > min_duration"),
        (5.0, 5.0, "needs max_duration > min_duration"),
    ],
)
@pytest.mark.parametrize(
    "name",
    ["detect_runs_between_regions", "segment_trials", "running_direction_labels"],
)
def test_run_and_trial_durations_are_checked(
    env, name, min_duration, max_duration, match
):
    bins, times = _walk(env)
    call = {
        "detect_runs_between_regions": lambda **kw: detect_runs_between_regions(
            bins, times, env, source="start", target="goal", **kw
        ),
        "segment_trials": lambda **kw: segment_trials(
            bins, times, env, start_region="start", end_regions=["goal"], **kw
        ),
        "running_direction_labels": lambda **kw: running_direction_labels(
            bins, times, env, start_region="start", end_regions="goal", **kw
        ),
    }[name]
    with pytest.raises(ValueError, match=match):
        call(min_duration=min_duration, max_duration=max_duration)


@pytest.mark.parametrize(
    ("options", "match"),
    [
        ({"min_speed": np.nan}, "min_speed must be finite and positive"),
        ({"min_speed": 0.0}, "min_speed must be finite and positive"),
        ({"min_speed": 5.0, "min_duration": -1.0}, "min_duration must be finite"),
        ({"min_speed": 5.0, "hysteresis": np.inf}, "hysteresis must be finite"),
    ],
)
def test_segment_by_velocity_parameters_are_checked(options, match):
    from neurospatial.behavior.segmentation import segment_by_velocity

    times = np.arange(20) / 10.0
    positions = np.c_[times * 20.0, np.zeros(20)]
    with pytest.raises(ValueError, match=match):
        segment_by_velocity(times, positions, **options)


def test_one_end_region_name_is_one_region(env):
    """end_regions='goal' means ['goal'], not the characters g, o, a, l."""
    x = np.r_[0.0, 0.0, np.linspace(10.0, 98.0, 45)]
    times = np.arange(x.size) / 10.0
    bins = env.bin_at(np.c_[x, np.zeros_like(x)])
    one = segment_trials(bins, times, env, start_region="start", end_regions="goal")
    listed = segment_trials(
        bins, times, env, start_region="start", end_regions=["goal"]
    )
    assert one == listed
    assert [t.outcome for t in one] == ["reached"]


class TestMinProgressUnits:
    """The default min_progress is 20 cm in the environment's units."""

    @staticmethod
    def _corridor(scale, units):
        xs = np.linspace(0.0, 100.0 * scale, 401)
        env = Environment.from_samples(
            np.c_[xs, np.zeros_like(xs)], bin_size=2.0 * scale
        )
        if units is not None:
            env = env.with_metadata(units=units)
        regions = Regions(env.regions.values())
        goal = box(95.0 * scale, -5.0 * scale, 105.0 * scale, 5.0 * scale)
        regions.add("goal", polygon=goal)
        return env.with_regions(regions)

    @pytest.mark.parametrize(
        ("scale", "units"), [(1.0, "cm"), (10.0, "mm"), (0.01, "m")]
    )
    @pytest.mark.parametrize(("start_cm", "detected"), [(70.0, True), (82.0, False)])
    def test_same_run_in_any_metric_unit(self, scale, units, start_cm, detected):
        # A straight approach from start_cm to 96 cm: 26 cm or 14 cm of
        # progress, on either side of the 20 cm default.
        env = self._corridor(scale, units)
        x = np.linspace(start_cm, 96.0, 20) * scale
        bins = env.bin_at(np.c_[x, np.zeros_like(x)])
        runs = detect_goal_directed_runs(
            bins, np.arange(20) / 10.0, env, goal_region="goal"
        )
        assert runs.shape == ((1, 2) if detected else (0, 2))

    def test_unknown_units_need_an_explicit_min_progress(self):
        env = self._corridor(1.0, None)
        x = np.linspace(70.0, 96.0, 20)
        bins = env.bin_at(np.c_[x, np.zeros_like(x)])
        times = np.arange(20) / 10.0
        with pytest.raises(ValueError, match="needs min_progress"):
            detect_goal_directed_runs(bins, times, env, goal_region="goal")
        runs = detect_goal_directed_runs(
            bins, times, env, goal_region="goal", min_progress=20.0
        )
        assert runs.shape == (1, 2)
