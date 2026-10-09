"""Position-only analyses warn when their gap/epoch gates exclude every interval.

Tracking sampled more coarsely than ``max_gap`` (or epochs on the wrong clock)
leaves no observed interval. An empty or all-NaN result must not look like a
scientific answer, so the shared gate names ``max_gap`` at the user's call site.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from neurospatial import Environment
from neurospatial.behavior.decisions import extract_pre_decision_window
from neurospatial.behavior.navigation import goal_bias
from neurospatial.behavior.segmentation import (
    detect_region_crossings,
    segment_by_velocity,
)
from neurospatial.events import add_positions
from neurospatial.ops.egocentric import heading_from_velocity


@pytest.fixture(scope="module")
def coarse_track():
    """1 Hz back-and-forth track: every interval exceeds the default max_gap."""
    times = np.arange(0.0, 100.0, 1.0)
    x = np.abs((times % 40) - 20) * 4 + 5
    positions = np.column_stack([x, np.full_like(x, 5.0)])
    env = Environment.from_samples(
        np.vstack([positions, positions + np.array([0.0, 5.0])]), bin_size=2.0
    )
    env.regions.add("mid", point=(45.0, 5.0))
    env.regions.buffer("mid", distance=6.0, new_name="mid_area")
    return env, times, positions


CALLS = {
    "detect_region_crossings": lambda env, t, p, **kw: detect_region_crossings(
        env.bin_at(p), t, env, region_name="mid_area", **kw
    ),
    "segment_by_velocity": lambda env, t, p, **kw: segment_by_velocity(
        t, p, 1.0, min_duration=0.0, **kw
    ),
    "heading_from_velocity": lambda env, t, p, **kw: heading_from_velocity(t, p, **kw),
    "goal_bias": lambda env, t, p, **kw: goal_bias(
        t, p, np.array([100.0, 5.0]), min_speed=0.0, **kw
    ),
    "extract_pre_decision_window": lambda env, t, p, **kw: (
        extract_pre_decision_window(t, p, 50.0, 10.0, **kw)
    ),
    "add_positions": lambda env, t, p, **kw: add_positions(
        pd.DataFrame({"timestamp": [10.5, 50.5]}), times=t, positions=p, **kw
    ),
}


@pytest.mark.parametrize("name", sorted(CALLS))
def test_coarse_sampling_warns_at_call_site(coarse_track, name):
    env, times, positions = coarse_track
    with pytest.warns(UserWarning, match=r"excluded ALL.*max_gap=0\.5") as record:
        CALLS[name](env, times, positions)
    messages = [w for w in record if "excluded ALL" in str(w.message)]
    assert len(messages) == 1
    assert "max_gap=None" in str(messages[0].message)
    assert messages[0].filename == __file__


@pytest.mark.parametrize("name", sorted(CALLS))
def test_disabling_the_gap_gate_is_silent(coarse_track, name):
    env, times, positions = coarse_track
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*excluded ALL.*")
        CALLS[name](env, times, positions, max_gap=None)


def test_detect_region_crossings_recovers_crossings_without_gap_gate(coarse_track):
    env, times, positions = coarse_track
    bins = env.bin_at(positions)
    with pytest.warns(UserWarning, match="excluded ALL"):
        assert detect_region_crossings(bins, times, env, region_name="mid_area") == []
    assert (
        len(
            detect_region_crossings(
                bins, times, env, region_name="mid_area", max_gap=None
            )
        )
        > 0
    )


def test_epochs_on_the_wrong_clock_name_epochs(coarse_track):
    _, times, positions = coarse_track
    with pytest.warns(UserWarning, match=r"excluded ALL.*epochs") as record:
        segment_by_velocity(times, positions, 1.0, max_gap=None, epochs=[(1e4, 6e4)])
    message = str(next(w for w in record if "excluded ALL" in str(w.message)).message)
    assert "max_gap=" not in message


def test_partial_exclusion_is_silent():
    times = np.r_[np.arange(0.0, 10.0, 0.1), np.arange(20.0, 30.0, 0.1)]
    positions = np.column_stack([times, np.zeros_like(times)])
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*excluded ALL.*")
        segment_by_velocity(times, positions, 0.5)


def test_fewer_than_two_samples_is_silent():
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*excluded ALL.*")
        segment_by_velocity(np.array([0.0]), np.array([[0.0, 0.0]]), 1.0)
