"""Segmentation detectors refuse malformed trajectories instead of guessing.

Positions in the wrong coordinates would be located in unrelated bins, and
unsorted, repeated or non-finite times would silently reorder or drop events.
"""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import box

from neurospatial import Environment, Regions
from neurospatial.behavior.segmentation import (
    detect_goal_directed_runs,
    detect_laps,
    detect_region_crossings,
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
    "detect_region_crossings": lambda env, t, p: detect_region_crossings(
        env, t, p, region_name="goal"
    ),
    "detect_laps": lambda env, t, p: detect_laps(env, t, p),
    "detect_goal_directed_runs": lambda env, t, p: detect_goal_directed_runs(
        env, t, p, goal_region="goal"
    ).windows,
}


def _walk():
    x = np.linspace(0.0, 98.0, 50)
    return np.c_[x, np.zeros_like(x)], np.arange(50) / 10.0


@pytest.mark.parametrize("name", DETECTORS)
def test_positions_in_other_dimensions_are_refused(env, name):
    positions, times = _walk()
    with pytest.raises(ValueError, match="3 coordinate"):
        DETECTORS[name](env, times, np.c_[positions, np.zeros(len(times))])


@pytest.mark.parametrize("name", DETECTORS)
@pytest.mark.parametrize("problem", ["unsorted", "repeated", "nan", "inf"])
def test_bad_times_are_refused(env, name, problem):
    positions, times = _walk()
    if problem == "unsorted":
        times[[5, 6]] = times[[6, 5]]
    elif problem == "repeated":
        times[6] = times[5]
    else:
        times[5] = np.nan if problem == "nan" else np.inf
    with pytest.raises(ValueError, match="times"):
        DETECTORS[name](env, times, positions)


@pytest.mark.parametrize("name", DETECTORS)
def test_length_mismatch_is_refused(env, name):
    positions, times = _walk()
    with pytest.raises(ValueError, match="same length"):
        DETECTORS[name](env, times[:-1], positions)


@pytest.mark.parametrize("name", DETECTORS)
def test_swapped_times_and_positions_are_named(env, name):
    positions, times = _walk()
    with pytest.raises(ValueError, match="did you pass positions before times"):
        DETECTORS[name](env, positions, times)


def test_nan_marks_a_missing_sample(env):
    """A NaN position (lost tracking) is accepted and ends observation."""
    positions, times = _walk()
    positions[20] = np.nan
    (crossing,) = detect_region_crossings(
        env, times, positions, region_name="goal", direction="entry"
    )
    assert crossing.time == times[-2]


def test_unknown_crossing_direction_is_refused(env):
    positions, times = _walk()
    with pytest.raises(ValueError, match="direction must be one of"):
        detect_region_crossings(
            env,
            times,
            positions,
            region_name="goal",
            direction="entries",  # type: ignore[arg-type]
        )


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
        segment_by_velocity(times, positions, **options).windows


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
        runs = detect_goal_directed_runs(
            env, np.arange(20) / 10.0, np.c_[x, np.zeros_like(x)], goal_region="goal"
        ).windows
        assert runs.shape == ((1, 2) if detected else (0, 2))

    def test_unknown_units_need_an_explicit_min_progress(self):
        env = self._corridor(1.0, None)
        x = np.linspace(70.0, 96.0, 20)
        times = np.arange(20) / 10.0
        with pytest.raises(ValueError, match="needs min_progress"):
            detect_goal_directed_runs(
                env, times, np.c_[x, np.zeros_like(x)], goal_region="goal"
            ).windows
        runs = detect_goal_directed_runs(
            env,
            times,
            np.c_[x, np.zeros_like(x)],
            goal_region="goal",
            min_progress=20.0,
        ).windows
        assert runs.shape == (1, 2)
