"""Behavior functions use the speed and headings a scientist passes.

Every function that gates on ``min_speed`` or uses the direction of movement
takes ``speed=`` and ``headings=``. Passing what the package would compute
reproduces the default; passing something else (speed from your own
processing, head direction from a sensor) is used as given, except that
samples outside ``max_gap``, ``epochs`` and ``break_before`` stay excluded.
"""

import numpy as np
import pytest

from neurospatial.behavior import (
    compute_goal_directed_metrics,
    compute_pre_decision_metrics,
    compute_speed,
    extract_pre_decision_window,
    goal_bias,
    heading_direction_labels,
    heading_from_velocity,
    instantaneous_goal_alignment,
    pre_decision_heading_stats,
    pre_decision_speed_stats,
    segment_by_velocity,
)
from neurospatial.behavior.vte import compute_vte_trial, head_sweep_from_positions

from ._kinematics_snapshots import load_kinematic_recording
from ._segmentation_snapshots import structural_snapshot
from .test_segmentation_gap_free import assert_snapshot_equal

MIN_SPEED = 5.0


@pytest.fixture(scope="module")
def session(continuous_recording):
    """The kinematic fixture with lost frames and a declared break."""
    r = load_kinematic_recording(continuous_recording)
    positions = r.positions.copy()
    positions[[1200, 1201, 3000]] = np.nan
    break_before = np.zeros(len(r.times), dtype=bool)
    break_before[6000] = True
    support = {"max_gap": 1.0, "break_before": break_before}
    speed = compute_speed(r.times, positions, **support)
    headings = heading_from_velocity(r.times, positions, **support)
    return r, positions, support, speed, headings


DEFAULT_CALLS = {
    "heading_direction_labels": lambda r, p, **kw: heading_direction_labels(
        r.times, p, min_speed=MIN_SPEED, **kw
    ),
    "instantaneous_goal_alignment": lambda r, p, **kw: instantaneous_goal_alignment(
        r.times, p, r.positions[-1], min_speed=MIN_SPEED, **kw
    ),
    "goal_bias": lambda r, p, **kw: goal_bias(
        r.times, p, r.positions[-1], min_speed=MIN_SPEED, **kw
    ),
    "compute_goal_directed_metrics": lambda r, p, **kw: compute_goal_directed_metrics(
        r.env, r.times, p, r.positions[-1], goal_radius=5, **kw
    ),
    "pre_decision_heading_stats": lambda r, p, **kw: pre_decision_heading_stats(
        r.times, p, min_speed=MIN_SPEED, **kw
    ),
    "head_sweep_from_positions": lambda r, p, **kw: head_sweep_from_positions(
        r.times, p, min_speed=MIN_SPEED, **kw
    ),
}
SPEED_ONLY_CALLS = {
    "pre_decision_speed_stats": lambda r, p, **kw: pre_decision_speed_stats(
        r.times, p, **kw
    ),
    "segment_by_velocity": lambda r, p, **kw: segment_by_velocity(
        r.times, p, min_speed=MIN_SPEED, **kw
    ).windows,
}


@pytest.mark.parametrize("name", [*DEFAULT_CALLS, *SPEED_ONLY_CALLS])
def test_passing_the_computed_motion_reproduces_the_default(session, name):
    r, positions, support, speed, headings = session
    call = {**DEFAULT_CALLS, **SPEED_ONLY_CALLS}[name]
    motion = {"speed": speed}
    if name in DEFAULT_CALLS:
        motion["headings"] = headings
    assert_snapshot_equal(
        structural_snapshot(call(r, positions, **support, **motion)),
        structural_snapshot(call(r, positions, **support)),
    )


def test_window_metrics_take_your_motion_at_the_window_samples(session):
    """A pre-decision window reads the arrays you pass at its own samples."""
    r, positions, support, _, _ = session
    rng = np.random.default_rng(0)
    speed = rng.uniform(0.0, 40.0, len(r.times))
    headings = rng.uniform(-np.pi, np.pi, len(r.times))
    entry, duration = 100.0, 2.0
    window_times, window_positions = extract_pre_decision_window(
        r.times, positions, entry, window_duration=duration, **support
    )
    at = np.searchsorted(r.times, window_times)
    mine = {"speed": speed[at], "headings": headings[at]}

    metrics = compute_pre_decision_metrics(
        r.times,
        positions,
        entry,
        window_duration=duration,
        min_speed=MIN_SPEED,
        speed=speed,
        headings=headings,
        **support,
    )
    mean_speed, min_speed = pre_decision_speed_stats(
        window_times, window_positions, speed=mine["speed"], max_gap=1.0
    )
    assert (metrics.mean_speed, metrics.min_observed_speed) == (mean_speed, min_speed)
    _, _, resultant = pre_decision_heading_stats(
        window_times, window_positions, min_speed=MIN_SPEED, max_gap=1.0, **mine
    )
    assert metrics.heading_mean_resultant_length == resultant

    trial = compute_vte_trial(
        r.times,
        positions,
        entry,
        window_duration=duration,
        min_speed=MIN_SPEED,
        speed=speed,
        headings=headings,
        **support,
    )
    assert trial.head_sweep_magnitude == head_sweep_from_positions(
        window_times, window_positions, min_speed=MIN_SPEED, max_gap=1.0, **mine
    )
    assert trial.mean_speed == mean_speed
    # Random headings, not the tracked path's, set the window's statistics.
    default = compute_pre_decision_metrics(
        r.times,
        positions,
        entry,
        window_duration=duration,
        min_speed=MIN_SPEED,
        **support,
    )
    assert metrics.heading_mean_resultant_length < default.heading_mean_resultant_length


def test_sensor_headings_replace_the_direction_of_movement(session):
    """Head direction that always points at the goal scores goal bias 1.

    The tracked path wanders, so its direction of movement does not point at
    the goal; the headings passed are used instead.
    """
    r, positions, support, _, _ = session
    goal = r.positions[-1]
    toward = goal - positions
    headings = np.arctan2(toward[:, 1], toward[:, 0])
    assert goal_bias(r.times, positions, goal, min_speed=MIN_SPEED, **support) < 0.5
    bias = goal_bias(
        r.times, positions, goal, min_speed=MIN_SPEED, headings=headings, **support
    )
    assert bias == pytest.approx(1.0, abs=1e-12)
    # A heading that never changes sweeps no angle.
    constant = np.full(len(r.times), 0.7)
    assert (
        head_sweep_from_positions(
            r.times, positions, min_speed=MIN_SPEED, headings=constant, **support
        )
        == 0.0
    )


def test_your_speed_decides_who_is_moving(session):
    """A stop in the speed you pass removes those samples from every gate."""
    r, positions, support, _, _ = session
    speed = np.full(len(r.times), 20.0)
    stopped = (r.times >= 50.0) & (r.times < 60.0)
    speed[stopped] = 0.0
    alignment = instantaneous_goal_alignment(
        r.times, positions, r.positions[-1], min_speed=MIN_SPEED, speed=speed, **support
    )
    assert np.isnan(alignment[stopped]).all()
    epochs = segment_by_velocity(
        r.times, positions, min_speed=MIN_SPEED, speed=speed, **support
    ).windows
    overlaps = (epochs[:, 0] < 60.0) & (epochs[:, 1] > 50.0)
    assert epochs.size
    assert not overlaps.any()
    labels = heading_direction_labels(
        r.times, positions, min_speed=MIN_SPEED, speed=speed, **support
    )
    assert set(labels[stopped]) <= {"stationary", None}


def test_your_motion_stays_inside_the_time_support(session):
    """Samples outside epochs, lost frames and lone samples stay excluded."""
    r, positions, _, _, _ = session
    speed = np.full(len(r.times), 20.0)
    headings = np.zeros(len(r.times))
    alignment = instantaneous_goal_alignment(
        r.times,
        positions,
        r.positions[-1],
        min_speed=MIN_SPEED,
        speed=speed,
        headings=headings,
        epochs=(10.0, 20.0),
        max_gap=1.0,
    )
    # The sample at 20.0 s ends the last interval inside [10, 20), so it is
    # in the run, as it is for the default.
    outside = (r.times < 10.0) | (r.times > 20.0)
    assert np.isnan(alignment[outside]).all()
    assert np.isnan(alignment[[1200, 1201, 3000]]).all()
    assert np.isfinite(alignment[~outside & np.isfinite(positions).all(axis=1)]).any()


@pytest.mark.parametrize(
    ("name", "value", "match"),
    [
        ("speed", np.ones(3), "one number per sample"),
        ("speed", np.ones((10000, 1)), "one number per sample"),
        ("speed", np.array(["fast"] * 10000), "one number per sample"),
        ("speed", np.r_[-1.0, np.ones(9999)], "finite and >= 0 or NaN"),
        ("speed", np.r_[np.inf, np.ones(9999)], "finite and >= 0 or NaN"),
        ("headings", np.r_[np.inf, np.zeros(9999)], "finite or NaN"),
    ],
)
def test_invalid_motion_is_refused_with_a_fix(session, name, value, match):
    r, positions, support, _, _ = session
    with pytest.raises(ValueError, match=match) as caught:
        goal_bias(
            r.times,
            positions,
            r.positions[-1],
            min_speed=MIN_SPEED,
            **{name: value},
            **support,
        )
    message = str(caught.value)
    assert f"goal_bias() needs {name}" in message or f"every {name}" in message
    assert "Fix:" in message


def test_negative_headings_are_valid():
    """Headings are angles: any finite value, wrapped or unwrapped."""
    times = np.arange(20) / 10
    positions = np.column_stack([10 * times, np.zeros(20)])
    headings = np.full(20, -3 * np.pi / 2)  # North, unwrapped
    labels = heading_direction_labels(
        times, positions, min_speed=1.0, headings=headings
    )
    assert set(labels) == {"90°"}
