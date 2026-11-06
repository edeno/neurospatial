"""An arrival needs a stop: min_dwell and the running-through warning."""

from __future__ import annotations

import warnings
from itertools import pairwise

import numpy as np
import pytest
from shapely.geometry import box

from neurospatial import Environment, Regions
from neurospatial.behavior import detect_region_visits, detect_trials

FPS = 30.0
SPEED = 40.0  # cm/s
STOP = 3.0  # s


def _wells_in_a_row():
    """Three wells on a line: runs between the outer two pass the middle one."""
    env = Environment.from_polygon(box(0, 0, 200, 40), bin_size=4.0, units="cm")
    centers = {"left": (20.0, 20.0), "middle": (100.0, 20.0), "right": (180.0, 20.0)}
    regions = Regions()
    for name, center in centers.items():
        regions.add(name, point=center, radius=8.0)
    return env.with_regions(regions), centers


def _session(route, centers, *, stop=STOP):
    pieces = [np.tile(centers[route[0]], (int(stop * FPS), 1))]
    for start, end in pairwise(route):
        a, b = np.array(centers[start]), np.array(centers[end])
        n_steps = int(np.linalg.norm(b - a) / SPEED * FPS)
        pieces.append(a + np.outer(np.arange(n_steps) / n_steps, b - a))
        pieces.append(np.tile(b, (int(stop * FPS), 1)))
    positions = np.vstack(pieces)
    positions += np.random.default_rng(0).normal(0.0, 0.3, positions.shape)
    return np.arange(len(positions)) / FPS, positions


ROUTE = ["left", "right"] * 6 + ["left"]
NAMES = ["left", "middle", "right"]


def test_every_entry_is_an_arrival_by_default_and_a_warning_says_so():
    env, centers = _wells_in_a_row()
    times, positions = _session(ROUTE, centers)
    with pytest.warns(UserWarning, match=r"12 of 25 visits crossed their region"):
        trials = detect_trials(env, times, positions, regions=NAMES)
    assert len(trials) == 24  # each of the 12 runs is cut in two at the middle well


def test_min_dwell_keeps_runs_whole_and_passing_a_well_makes_no_visit():
    env, centers = _wells_in_a_row()
    times, positions = _session(ROUTE, centers)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        trials = detect_trials(env, times, positions, regions=NAMES, min_dwell=1.0)
        visits = detect_region_visits(
            env, times, positions, regions=NAMES, min_dwell=1.0
        )
    assert trials.start_region.tolist() == ROUTE[:-1]
    assert trials.end_region.tolist() == ROUTE[1:]
    assert set(trials.outcome) == {"reached"}
    assert "middle" not in set(visits.region)
    assert np.median(trials.durations) == pytest.approx(160.0 / SPEED, abs=0.5)
    assert trials.analysis_settings["min_dwell"] == 1.0


def test_stopping_at_every_well_is_silent_and_min_dwell_changes_nothing():
    env, centers = _wells_in_a_row()
    route = ["left", "middle", "right", "middle"] * 3 + ["left"]
    times, positions = _session(route, centers)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        every_entry = detect_trials(env, times, positions, regions=NAMES)
        stops = detect_trials(env, times, positions, regions=NAMES, min_dwell=1.0)
    assert every_entry.start_region.tolist() == stops.start_region.tolist()
    assert every_entry.end_region.tolist() == route[1:]


def test_a_visit_cut_off_by_the_recording_is_kept():
    """It may have lasted longer than was seen, so it is not called short."""
    env, centers = _wells_in_a_row()
    times, positions = _session(["left", "right"], centers, stop=0.5)
    visits = detect_region_visits(env, times, positions, regions=NAMES, min_dwell=5.0)
    assert visits.region.tolist() == ["left", "right"]
    assert not visits.entry_observed[0]
    assert not visits.exit_observed[1]


@pytest.mark.parametrize("min_dwell", [-1.0, np.nan, np.inf])
def test_min_dwell_must_be_a_duration(min_dwell):
    env, centers = _wells_in_a_row()
    times, positions = _session(["left", "right"], centers)
    with pytest.raises(ValueError, match="min_dwell must be finite"):
        detect_trials(env, times, positions, regions=NAMES, min_dwell=min_dwell)


def _stops_and_passes(n_passes):
    """Runs that stop at every well, plus runs straight past the middle one."""
    env, centers = _wells_in_a_row()
    route = ["left", "middle", "right", "middle"] * 8 + ["left"]
    stopped = _session(route, centers, stop=1.0)[1]
    passing = _session(["left", "right"] * n_passes + ["left"], centers, stop=1.0)[1]
    positions = np.vstack([stopped, passing])
    return env, np.arange(len(positions)) / FPS, positions


def test_a_long_rest_elsewhere_does_not_make_a_stop_read_as_a_crossing():
    """Stops left by the far side stay stops, however long the rest in the arrays."""
    env, centers = _wells_in_a_row()
    route = ["left", "middle", "right", "middle"] * 2 + ["left"]
    _, task = _session(route, centers)
    corner = np.array([100.0, 2.0])
    rest = corner + np.random.default_rng(1).normal(0.0, 0.3, (int(600 * FPS), 2))
    positions = np.vstack([task, rest])
    times = np.arange(len(positions)) / FPS
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        detect_trials(env, times, positions, regions=NAMES)


def test_a_slow_pass_warns_as_well_as_a_fast_one():
    env, centers = _wells_in_a_row()
    _, positions = _session(ROUTE, centers)
    slow = np.repeat(positions, 3, axis=0)  # 13 cm/s, not 40
    with pytest.warns(UserWarning, match="crossed their region"):
        detect_trials(env, np.arange(len(slow)) / FPS, slow, regions=NAMES)


def test_a_band_across_the_arena_warns_though_it_is_long():
    env = Environment.from_polygon(box(0, 0, 200, 200), bin_size=4.0, units="cm")
    regions = Regions()
    regions.add("left", polygon=box(0, 0, 30, 200))
    regions.add("band", polygon=box(92, 0, 108, 200))
    regions.add("right", polygon=box(170, 0, 200, 200))
    env = env.with_regions(regions)
    centers = {"left": (15.0, 100.0), "right": (185.0, 100.0)}
    times, positions = _session(ROUTE, centers)
    with pytest.warns(UserWarning, match="crossed their region"):
        detect_trials(env, times, positions, regions=["left", "band", "right"])


def test_jitter_at_a_regions_edge_does_not_warn():
    env, centers = _wells_in_a_row()
    edge = np.array(centers["middle"]) + np.array([8.0, 0.0])
    sides = 2.0 * ((np.arange(600) // 4) % 2) - 1.0  # 4 samples in, 4 out
    positions = edge + np.outer(sides, (1.0, 0.0))
    positions += np.random.default_rng(2).normal(0.0, 0.3, positions.shape)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        detect_trials(env, np.arange(len(positions)) / FPS, positions, regions=NAMES)


def test_one_detour_does_not_warn_and_a_rare_pass_does_not_either():
    env, times, positions = _stops_and_passes(0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        detect_trials(env, times, positions, regions=NAMES)
    env, times, positions = _stops_and_passes(1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        detect_trials(env, times, positions, regions=NAMES)
    env, times, positions = _stops_and_passes(8)
    with pytest.warns(UserWarning, match="crossed their region"):
        detect_trials(env, times, positions, regions=NAMES)


def test_a_visit_exactly_as_long_as_min_dwell_counts():
    env, centers = _wells_in_a_row()
    inside, outside = centers["middle"], (100.0, 35.0)
    positions = np.array([outside, inside, inside, inside, outside])
    times = np.arange(5.0)
    assert len(detect_region_visits(env, times, positions, regions=NAMES)) == 1
    kept = detect_region_visits(
        env, times, positions, regions=NAMES, min_dwell=3.0, max_gap=None
    )
    assert len(kept) == 1
    dropped = detect_region_visits(
        env, times, positions, regions=NAMES, min_dwell=3.01, max_gap=None
    )
    assert len(dropped) == 0


def test_a_pass_cut_off_by_a_gap_is_no_arrival():
    """Tracking lost inside a well during a run: the trial is censored there."""
    env, centers = _wells_in_a_row()
    times, positions = _session(["left", "right"], centers)
    in_middle = np.flatnonzero(np.abs(positions[:, 0] - 100.0) < 8.0)
    lost = times >= times[in_middle[3]]
    positions = positions.copy()
    positions[lost & (times < times[in_middle[3]] + 2.0)] = np.nan
    trials = detect_trials(env, times, positions, regions=NAMES, min_dwell=1.0)
    assert trials.outcome.tolist()[0] == "censored"
    assert trials.end_region.tolist()[0] is None


def test_visits_of_one_or_two_samples_are_not_judged():
    env, centers = _wells_in_a_row()
    _, positions = _session(ROUTE, centers)
    coarse = positions[::8]  # 3.75 Hz: a pass holds at most two samples
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        detect_trials(env, np.arange(len(coarse)) * 8 / FPS, coarse, regions=NAMES)


def test_choosing_a_min_dwell_turns_the_warning_off():
    env, centers = _wells_in_a_row()
    times, positions = _session(ROUTE, centers)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        detect_trials(env, times, positions, regions=NAMES, min_dwell=0.1)
