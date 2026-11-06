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
