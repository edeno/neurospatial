"""A declared recording break acts exactly like a gap longer than ``max_gap``.

Two recordings joined on one clock with no time between them are analyzed as
separate recordings when ``break_before`` marks the first sample of the second
one. The reference is the same data with every time from the break on shifted
by a large offset, which the ``max_gap`` gate already splits.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from neurospatial.behavior import (
    compute_goal_directed_metrics,
    compute_home_range,
    detect_goal_directed_runs,
    detect_laps,
    detect_region_crossings,
    detect_region_visits,
    detect_runs_between_regions,
    goal_bias,
    heading_direction_labels,
    instantaneous_goal_alignment,
    running_direction_labels,
    segment_by_velocity,
    segment_trials,
)

BREAK = 1000  # first sample of the second recording in both tracks
OFFSET = 1024.0  # seconds; dyadic, so shifted times and differences are exact
GOAL = np.array([95.0, 5.0])
# Off the track, so a step toward the target is not aligned with the goal.
OFF_TRACK_GOAL = np.array([60.0, 100.0])


@dataclasses.dataclass(frozen=True)
class Joined:
    """Two recordings joined with no time between them, and the reference."""

    times: np.ndarray
    shifted_times: np.ndarray
    positions: np.ndarray
    position_bins: np.ndarray
    env: Any

    @property
    def break_before(self) -> np.ndarray:
        return np.arange(self.times.size) == BREAK


def _join(track) -> Joined:
    # 16 Hz on a dyadic grid, so every time and difference is exact in float64.
    times = np.arange(track.times.size) / 16
    shifted = times + np.where(np.arange(times.size) >= BREAK, OFFSET, 0.0)
    return Joined(times, shifted, track.positions, track.position_bins, track.env)


@pytest.fixture(scope="module")
def pause(continuous_pause_track) -> Joined:
    """A run toward the target that the join moves straight into it."""
    return _join(continuous_pause_track)


@pytest.fixture(scope="module")
def laps(continuous_lap_track) -> Joined:
    """Two stretches of a square-ring lap; the join skips a quarter lap."""
    return _join(continuous_lap_track)


def _plain(value: Any) -> Any:
    """Comparable form of a result: dataclasses become dicts, arrays lists."""
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: _plain(getattr(value, f.name)) for f in dataclasses.fields(value)
        }
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


_TIME_FIELDS = ("time", "start_time", "end_time")


def _shift_reported_times(value: Any, *, after: float) -> Any:
    """Shift reported times at or after the break, as the reference reports them."""
    if isinstance(value, list):
        return [_shift_reported_times(item, after=after) for item in value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        changes = {
            name: getattr(value, name) + OFFSET
            for name in _TIME_FIELDS
            if hasattr(value, name) and getattr(value, name) >= after
        }
        return dataclasses.replace(value, **changes)
    return value


Case = Callable[[Joined, np.ndarray, dict], Any]

PAUSE_CASES: dict[str, Case] = {
    "detect_region_crossings": lambda r, t, kw: detect_region_crossings(
        r.position_bins, t, r.env, region_name="target", **kw
    ),
    "detect_runs_between_regions": lambda r, t, kw: detect_runs_between_regions(
        r.position_bins, t, r.env, source="source", target="target", **kw
    ),
    "segment_by_velocity": lambda r, t, kw: segment_by_velocity(
        t, r.positions, min_speed=5.0, min_duration=0.1, **kw
    ),
    "running_direction_labels": lambda r, t, kw: running_direction_labels(
        r.position_bins,
        t,
        r.env,
        start_region="source",
        end_regions="target",
        min_duration=0.1,
        **kw,
    ),
    "segment_trials": lambda r, t, kw: segment_trials(
        r.position_bins,
        t,
        r.env,
        start_region="source",
        end_regions=["target"],
        min_duration=0.1,
        **kw,
    ),
    "detect_goal_directed_runs": lambda r, t, kw: detect_goal_directed_runs(
        r.position_bins, t, r.env, goal_region="target", min_progress=1.0, **kw
    ),
    "detect_region_visits": lambda r, t, kw: detect_region_visits(
        r.position_bins, t, r.env, region_names=["source", "target"], **kw
    ),
    "heading_direction_labels": lambda r, t, kw: heading_direction_labels(
        t, r.positions, min_speed=5.0, **kw
    ),
    "instantaneous_goal_alignment": lambda r, t, kw: instantaneous_goal_alignment(
        t, r.positions, GOAL, min_speed=5.0, **kw
    ),
    "goal_bias": lambda r, t, kw: goal_bias(
        t, r.positions, OFF_TRACK_GOAL, min_speed=5.0, **kw
    ),
    "compute_goal_directed_metrics": lambda r, t, kw: compute_goal_directed_metrics(
        r.env, t, r.positions, GOAL, metric="geodesic", min_speed=5.0, **kw
    ),
    "compute_home_range": lambda r, t, kw: compute_home_range(
        r.position_bins, times=t, percentile=100.0, **kw
    ),
}

LAP_CASES: dict[str, Case] = {
    f"detect_laps[{method}]": (
        lambda r, t, kw, method=method: detect_laps(
            r.position_bins,
            t,
            r.env,
            method=method,
            start_region="start" if method == "region" else None,
            reference_lap=r.position_bins[160:800] if method == "reference" else None,
            min_overlap=0.5,
            **kw,
        )
    )
    for method in ("auto", "reference", "region")
}

CASES = {name: ("pause", case) for name, case in PAUSE_CASES.items()} | {
    name: ("laps", case) for name, case in LAP_CASES.items()
}


@pytest.fixture
def joined(request, pause, laps) -> Joined:
    return {"pause": pause, "laps": laps}[request.param]


def _cases():
    return [
        pytest.param(track, case, name.split("[")[0], id=name)
        for name, (track, case) in CASES.items()
    ]


@pytest.mark.parametrize(("joined", "case", "name"), _cases(), indirect=["joined"])
def test_declared_break_equals_long_gap(joined, case, name):
    r = joined
    declared = case(r, r.times, {"break_before": r.break_before})
    reference = case(r, r.shifted_times, {})
    undeclared = case(r, r.times, {})

    if isinstance(declared, list):
        declared = _shift_reported_times(declared, after=r.times[BREAK])
    # Wall-clock time to the goal includes the pause in the reference.
    time_to_goal = getattr(declared, "time_to_goal", None)
    if time_to_goal is not None and time_to_goal >= r.times[BREAK] - r.times[0]:
        declared = dataclasses.replace(declared, time_to_goal=time_to_goal + OFFSET)
    np.testing.assert_equal(_plain(declared), _plain(reference))
    if name != "compute_home_range":  # see test_home_range_break_isolates_sample
        # The join is not a recording break unless declared.
        assert _plain(declared) != _plain(undeclared)


@pytest.mark.parametrize(("joined", "case", "name"), _cases(), indirect=["joined"])
def test_no_breaks_matches_omitting_break_before(joined, case, name):
    r = joined
    expected = case(r, r.times, {})
    no_breaks = np.zeros(r.times.size, dtype=bool)
    np.testing.assert_equal(
        _plain(case(r, r.times, {"break_before": no_breaks})), _plain(expected)
    )
    # Entry 0 is not a break: there is no sample before it.
    np.testing.assert_equal(
        _plain(case(r, r.times, {"break_before": np.arange(r.times.size) == 0})),
        _plain(expected),
    )


INVALID = {
    "wrong length": lambda n: np.zeros(n - 1, dtype=bool),
    "not Boolean": lambda n: np.zeros(n, dtype=int),
}


@pytest.mark.parametrize("bad", INVALID.values(), ids=INVALID.keys())
@pytest.mark.parametrize(("joined", "case", "name"), _cases(), indirect=["joined"])
def test_invalid_break_before_names_function(joined, case, name, bad):
    r = joined
    with pytest.raises(ValueError, match=rf"{name}\(\) needs break_before"):
        case(r, r.times, {"break_before": bad(r.times.size)})


def test_trial_spanning_break_ends_failed_at_recording_end(pause):
    r = pause
    kwargs = {"start_region": "source", "end_regions": ["target"], "min_duration": 0.1}
    (joined,) = segment_trials(r.position_bins, r.times, r.env, **kwargs)
    assert joined.success
    (split,) = segment_trials(
        r.position_bins, r.times, r.env, break_before=r.break_before, **kwargs
    )
    assert not split.success
    assert split.end_region is None
    assert split.end_time == r.times[BREAK - 1]


def test_visit_after_break_is_left_censored(pause):
    r = pause
    visits = detect_region_visits(
        r.position_bins, r.times, r.env, region_names=["target"]
    )
    assert [v.entry_observed for v in visits] == [True]
    (censored,) = detect_region_visits(
        r.position_bins,
        r.times,
        r.env,
        region_names=["target"],
        break_before=r.break_before,
    )
    assert censored.start_index == BREAK
    assert not censored.entry_observed


def test_lap_spanning_break_is_dropped(laps):
    r = laps
    kwargs = {"method": "region", "start_region": "start"}
    joined = detect_laps(r.position_bins, r.times, r.env, **kwargs)
    split = detect_laps(
        r.position_bins, r.times, r.env, break_before=r.break_before, **kwargs
    )
    spans = [lap for lap in joined if lap.start_time < r.times[BREAK] <= lap.end_time]
    assert len(spans) == 1
    assert len(split) == len(joined) - 1
    assert all(
        lap.end_time < r.times[BREAK] or lap.start_time >= r.times[BREAK]
        for lap in split
    )


def test_heading_labels_refuse_breaks_with_precomputed_kinematics():
    with pytest.raises(ValueError, match="precomputed speed and heading"):
        heading_direction_labels(
            speed=np.ones(3),
            heading=np.zeros(3),
            min_speed=0.5,
            break_before=np.array([False, True, False]),
        )


def test_home_range_refuses_breaks_without_times():
    with pytest.raises(ValueError, match="break_before without times"):
        compute_home_range(np.array([0, 0, 1]), break_before=np.zeros(3, dtype=bool))


def test_home_range_break_isolates_sample():
    # Uniform sampling gives a run's last sample the same dwell as its
    # interval, so only a sample isolated by a break changes the home range.
    position_bins = np.array([0, 1, 1, 1])
    times = np.arange(4.0)
    assert compute_home_range(
        position_bins, times=times, percentile=100.0, max_gap=None
    ).tolist() == [1, 0]
    isolated = compute_home_range(
        position_bins,
        times=times,
        percentile=100.0,
        max_gap=None,
        break_before=np.array([False, True, False, False]),
    )
    assert isolated.tolist() == [1]
