"""Trials as the runs between visits: from a visit table and from tracking."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neurospatial import Intervals, Regions
from neurospatial.behavior import (
    detect_region_visits,
    detect_trials,
    trials_between_visits,
)
from neurospatial.simulation.mazes import make_w_maze

FRAME_RATE = 30.0
WELL_RADIUS = 5.0


def _visit_table(regions, starts, stops, column="region"):
    return Intervals.from_dataframe(
        pd.DataFrame({"start_time": starts, "stop_time": stops, column: regions})
    )


def test_trials_are_gaps_between_visits():
    visits = _visit_table(
        ["A", "B", "A", "A", "C"],
        [0.0, 5.0, 12.0, 20.0, 30.0],
        [2.0, 8.0, 15.0, 22.0, 33.0],
    )
    trials = trials_between_visits(visits)
    assert trials.start.tolist() == [2.0, 8.0, 15.0, 22.0]
    assert trials.stop.tolist() == [5.0, 12.0, 20.0, 30.0]
    assert trials.start_region.tolist() == ["A", "B", "A", "A"]
    assert trials.end_region.tolist() == ["B", "A", None, "C"]
    assert trials.outcome.tolist() == ["reached", "reached", "returned", "reached"]
    assert trials.interval_ids.tolist() == [0, 1, 2, 3]


def test_user_table_with_port_column():
    by_region = trials_between_visits(_visit_table(["A", "B"], [0.0, 5.0], [2.0, 8.0]))
    by_port = trials_between_visits(
        _visit_table(["A", "B"], [0.0, 5.0], [2.0, 8.0], column="port"), region="port"
    )
    pd.testing.assert_frame_equal(
        by_region.intervals_table(), by_port.intervals_table(), check_like=True
    )
    with pytest.raises(ValueError, match="no field 'region'"):
        trials_between_visits(_visit_table(["A"], [0.0], [1.0], column="port"))


def test_overlapping_visits_refused():
    visits = _visit_table(["A", "B"], [0.0, 1.5], [2.0, 3.0])
    with pytest.raises(ValueError, match=r"visits 0 and 1 overlap"):
        trials_between_visits(visits)


def test_region_values_must_be_names():
    visits = Intervals.from_arrays([0.0, 5.0], [2.0, 8.0], region=[1, 2])
    with pytest.raises(TypeError, match="not a name"):
        trials_between_visits(visits)


def test_fewer_than_two_visits_give_no_trials():
    trials = trials_between_visits(_visit_table(["A"], [0.0], [1.0]))
    assert len(trials) == 0
    assert trials.fields == ("start_region", "end_region", "outcome")


def _w_track_with_sized_wells():
    track = make_w_maze().env_track
    regions = Regions()
    for name, region in track.regions.items():
        regions.add(name, point=region.data, radius=WELL_RADIUS)
    return track.with_regions(regions)


def _run(wells, start, goal, speed):
    """Samples from a rest at ``start`` to arrival at ``goal``, at ``speed``."""
    a, b = wells[start], wells[goal]
    corners = np.array([a, [a[0], 0.0], [b[0], 0.0], b])
    along = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(corners, axis=0), axis=1))]
    distance = np.arange(0.0, along[-1], speed / FRAME_RATE)
    rest = np.repeat(a[None, :], int(FRAME_RATE), axis=0)  # 1 s at the start well
    path = np.column_stack([np.interp(distance, along, corners[:, k]) for k in (0, 1)])
    return np.vstack([rest, path])


@pytest.fixture
def w_track_session():
    """An alternation session: centre, left, centre, right, ... at 30 Hz."""
    env = _w_track_with_sized_wells()
    wells = {
        name: np.asarray(env.regions[name].metadata["center"]) for name in env.regions
    }
    sequence = ["well_2", "well_1", "well_2", "well_3", "well_2", "well_1", "well_2"]
    speeds = np.random.default_rng(3).uniform(25.0, 60.0, len(sequence) - 1)
    parts = [
        _run(wells, a, b, v)
        for a, b, v in zip(sequence[:-1], sequence[1:], speeds, strict=True)
    ]
    parts.append(np.repeat(wells[sequence[-1]][None, :], int(FRAME_RATE), axis=0))
    positions = np.vstack(parts)
    times = np.arange(len(positions)) / FRAME_RATE
    return env, times, positions, sequence, wells


def _inside_any(positions, wells):
    return np.any(
        [np.hypot(*(positions - c).T) < WELL_RADIUS for c in wells.values()], axis=0
    )


def test_multi_start_alternation_in_one_call(w_track_session):
    env, times, positions, sequence, wells = w_track_session
    trials = detect_trials(env, times, positions, regions=list(wells))
    assert trials.start_region.tolist() == sequence[:-1]
    assert trials.end_region.tolist() == sequence[1:]
    assert set(trials.outcome) == {"reached"}
    # Each trial is the run between wells: every sample in it is outside the
    # wells; the arrival sample is inside the end well and the sample before
    # the departure inside the start well (to the bin resolution).
    inside = _inside_any(positions, wells)
    for start, stop in zip(trials.start, trials.stop, strict=True):
        run = (times >= start) & (times < stop)
        assert not inside[run][2:-2].any()
        k = np.searchsorted(times, stop)
        assert inside[k - 2 : k + 3].any()


def test_trials_are_the_gaps_between_detected_visits(w_track_session):
    env, times, positions, _, wells = w_track_session
    visits = detect_region_visits(env, times, positions, regions=list(wells))
    from_visits = trials_between_visits(visits)
    detected = detect_trials(env, times, positions, regions=list(wells))
    np.testing.assert_array_equal(from_visits.start, detected.start)
    np.testing.assert_array_equal(from_visits.stop, detected.stop)
    assert from_visits.end_region.tolist() == detected.end_region.tolist()


def test_no_trial_across_tracking_loss(w_track_session):
    env, times, positions, sequence, wells = w_track_session
    trials = detect_trials(env, times, positions, regions=list(wells))
    lost = positions.copy()
    middle = 0.5 * (trials.start[1] + trials.stop[1])  # during the second run
    gap = (times >= middle) & (times < middle + 0.2)
    lost[gap] = np.nan
    broken = detect_trials(env, times, lost, regions=list(wells))
    assert broken.outcome.tolist().count("censored") == 1
    censored = broken[broken.outcome == "censored"]
    assert censored.start_region.tolist() == [sequence[1]]
    assert censored.stop[0] <= middle
    # The arrival after the loss starts no trial back to before it.
    assert len(broken) == len(trials)


def test_returned_trial(w_track_session):
    env, _, _, _, wells = w_track_session
    out_and_back = np.vstack(
        [
            np.repeat(wells["well_1"][None, :], 30, axis=0),
            wells["well_1"] + np.c_[np.zeros(30), -np.linspace(0, 20, 30)],
            wells["well_1"] + np.c_[np.zeros(30), -np.linspace(20, 0, 30)],
        ]
    )
    t = np.arange(len(out_and_back)) / FRAME_RATE
    trials = detect_trials(env, t, out_and_back, regions=list(wells))
    assert trials.outcome.tolist() == ["returned"]
    assert trials.end_region.tolist() == [None]


def test_point_regions_refused():
    track = make_w_maze().env_track
    times = np.arange(3.0)
    positions = np.zeros((3, 2))
    with pytest.raises(ValueError, match=r"(?s)is a point.*radius="):
        detect_trials(track, times, positions, regions=["well_1", "well_2"])


def test_visits_are_intervals_with_region(w_track_session):
    env, times, positions, sequence, wells = w_track_session
    visits = detect_region_visits(env, times, positions, regions=list(wells))
    assert visits.region.tolist() == sequence
    assert visits.interval_ids.tolist() == list(range(len(sequence)))
    assert visits.analysis_settings["regions"] == list(wells)
