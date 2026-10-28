"""Position-only analyses warn when their gap/epoch gates exclude every interval.

Tracking sampled more coarsely than ``max_gap`` (or epochs on the wrong clock)
leaves no observed interval. An empty or all-NaN result must not look like a
scientific answer, so the shared gate names ``max_gap`` at the user's call site.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from neurospatial import Environment, Regions
from neurospatial._intervals import DEFAULT_MAX_GAP
from neurospatial.behavior import (
    compute_path_efficiency,
    compute_speed,
    heading_from_velocity,
)
from neurospatial.behavior.decisions import (
    compute_decision_analysis,
    extract_pre_decision_window,
)
from neurospatial.behavior.navigation import (
    compute_goal_directed_metrics,
    goal_bias,
    heading_direction_labels,
    instantaneous_goal_alignment,
)
from neurospatial.behavior.segmentation import (
    Trial,
    detect_region_crossings,
    segment_by_velocity,
)
from neurospatial.behavior.vte import (
    compute_vte_session,
    compute_vte_trial,
    head_sweep_from_positions,
)
from neurospatial.events import add_positions


@pytest.fixture(scope="module")
def coarse_track():
    """0.5 Hz back-and-forth track: every interval exceeds the default max_gap."""
    times = np.arange(0.0, 200.0, 2.0)
    x = np.abs(((times / 2) % 40) - 20) * 4 + 5
    positions = np.column_stack([x, np.full_like(x, 5.0)])
    env = Environment.from_samples(
        np.vstack([positions, positions + np.array([0.0, 5.0])]), bin_size=2.0
    )
    region_builder = Regions(env.regions.values())
    region_builder.add("mid", point=(45.0, 5.0))
    env = env.with_regions(region_builder)
    region_builder = Regions(env.regions.values())
    region_builder.buffer("mid", distance=6.0, new_name="mid_area")
    env = env.with_regions(region_builder)
    return env, times, positions


@pytest.fixture(scope="module")
def nan_bridged_track(coarse_track):
    """10 Hz track with every other row missing, gated at max_gap=0.15 s.

    Every sampling interval (0.1 s) passes max_gap, but every gap across a
    missing row (0.2 s) exceeds it, so no interval is usable.
    """
    env, _, _ = coarse_track
    times = np.arange(0.0, 200.0, 0.1)
    x = np.abs(((times / 2) % 40) - 20) * 4 + 5
    positions = np.column_stack([x, np.full_like(x, 5.0)])
    positions[1::2] = np.nan
    return env, times, positions


def _vte_session(env, times, positions, **kwargs):
    """VTE session with a straight approach and one through a turnaround.

    With every interval excluded, both trials are unmeasured, with a warning.
    """
    trials = [
        Trial(0.0, 39.0, "start", "mid_area", "reached"),
        Trial(35.0, 79.0, "start", "mid_area", "reached"),
    ]
    options = {
        "decision_region": "mid_area",
        "trials": trials,
        "window_duration": 10.0,
        "min_speed": 1.0,
    }
    if kwargs.get("max_gap", DEFAULT_MAX_GAP) is None:
        result = compute_vte_session(env, times, positions, **options, **kwargs)
        assert len(result.trial_results) == 2
        return result
    with pytest.warns(UserWarning, match=r"only 0 observed sample\(s\)") as record:
        result = compute_vte_session(env, times, positions, **options, **kwargs)
    assert sum("only 0 observed" in str(w.message) for w in record) == 2
    assert [t.status for t in result.trial_results] == ["too_few_samples"] * 2
    return result


CALLS = {
    "detect_region_crossings": lambda env, t, p, **kw: detect_region_crossings(
        env.bin_at(p), t, env, region_name="mid_area", **kw
    ),
    "segment_by_velocity": lambda env, t, p, **kw: segment_by_velocity(
        t, p, min_speed=1.0, min_duration=0.0, **kw
    ),
    "heading_from_velocity": lambda env, t, p, **kw: heading_from_velocity(t, p, **kw),
    "goal_bias": lambda env, t, p, **kw: goal_bias(
        t, p, np.array([100.0, 5.0]), min_speed=0.0, **kw
    ),
    "instantaneous_goal_alignment": lambda env, t, p, **kw: (
        instantaneous_goal_alignment(t, p, np.array([100.0, 5.0]), min_speed=0.0, **kw)
    ),
    "compute_goal_directed_metrics": lambda env, t, p, **kw: (
        compute_goal_directed_metrics(
            env, t, p, np.array([100.0, 5.0]), min_speed=1.0, **kw
        )
    ),
    "heading_direction_labels": lambda env, t, p, **kw: heading_direction_labels(
        t, p, min_speed=1.0, **kw
    ),
    "extract_pre_decision_window": lambda env, t, p, **kw: (
        extract_pre_decision_window(t, p, 50.0, window_duration=10.0, **kw)
    ),
    "compute_decision_analysis": lambda env, t, p, **kw: compute_decision_analysis(
        env,
        t,
        p,
        decision_region="mid_area",
        goal_regions=["mid_area"],
        min_speed=1.0,
        **kw,
    ),
    "head_sweep_from_positions": lambda env, t, p, **kw: head_sweep_from_positions(
        t, p, min_speed=1.0, **kw
    ),
    "compute_vte_trial": lambda env, t, p, **kw: compute_vte_trial(
        t, p, 50.0, window_duration=10.0, min_speed=1.0, **kw
    ),
    "compute_vte_session": lambda env, t, p, **kw: _vte_session(env, t, p, **kw),
    "add_positions": lambda env, t, p, **kw: add_positions(
        pd.DataFrame({"timestamp": [10.5, 50.5]}), times=t, positions=p, **kw
    ),
    "compute_speed": lambda env, t, p, **kw: compute_speed(t, p, **kw),
    "compute_path_efficiency": lambda env, t, p, **kw: compute_path_efficiency(
        env, t, p, np.array([45.0, 5.0]), **kw
    ),
}


@pytest.mark.parametrize("name", sorted(CALLS))
def test_coarse_sampling_warns_at_call_site(coarse_track, name):
    env, times, positions = coarse_track
    with pytest.warns(
        UserWarning, match=rf"excluded ALL.*max_gap={DEFAULT_MAX_GAP}"
    ) as record:
        CALLS[name](env, times, positions)
    messages = [w for w in record if "excluded ALL" in str(w.message)]
    assert len(messages) == 1
    assert "max_gap=None" in str(messages[0].message)
    assert messages[0].filename == __file__


@pytest.mark.parametrize("name", sorted(CALLS))
def test_gaps_across_missing_samples_warn_at_call_site(nan_bridged_track, name):
    # The shared warning judges the intervals left after missing samples are
    # bridged, not each sampling interval on its own.
    env, times, positions = nan_bridged_track
    with pytest.warns(UserWarning, match=r"excluded ALL.*max_gap=0\.15") as record:
        CALLS[name](env, times, positions, max_gap=0.15)
    messages = [w for w in record if "excluded ALL" in str(w.message)]
    assert len(messages) == 1
    message = str(messages[0].message)
    assert "gaps across missing samples" in message
    assert "max_gap=None" in message
    assert messages[0].filename == __file__


# The session helper asserts the coarse track's per-trial warnings.
@pytest.mark.parametrize("name", sorted(set(CALLS) - {"compute_vte_session"}))
def test_bridging_the_missing_samples_is_silent(nan_bridged_track, name):
    env, times, positions = nan_bridged_track
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*excluded ALL.*")
        CALLS[name](env, times, positions, max_gap=0.25)


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
        segment_by_velocity(
            times, positions, min_speed=1.0, max_gap=None, epochs=[(1e4, 6e4)]
        )
    message = str(next(w for w in record if "excluded ALL" in str(w.message)).message)
    assert "max_gap=" not in message


def test_partial_exclusion_is_silent():
    times = np.r_[np.arange(0.0, 10.0, 0.1), np.arange(20.0, 30.0, 0.1)]
    positions = np.column_stack([times, np.zeros_like(times)])
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*excluded ALL.*")
        segment_by_velocity(times, positions, min_speed=0.5)


def test_fewer_than_two_samples_is_silent():
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*excluded ALL.*")
        segment_by_velocity(np.array([0.0]), np.array([[0.0, 0.0]]), min_speed=1.0)
