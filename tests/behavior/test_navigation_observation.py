"""Navigation headings follow the shared observation and movement rules.

A direction exists only where the animal moves (smoothed speed above zero and
at least ``min_speed``). A missing position (NaN row) ends a run; samples in no
run are not observed, which is different from an observed animal standing
still.
"""

import warnings

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.behavior.navigation import (
    compute_goal_directed_metrics,
    goal_bias,
    heading_direction_labels,
    instantaneous_goal_alignment,
)

EAST = "0\u201345\u00b0"  # the 0-45 degree sector label


def _times(n: int) -> np.ndarray:
    return np.arange(n) / 10


def _east_run(n: int, step: float) -> np.ndarray:
    return np.column_stack([step * np.arange(n, dtype=float), np.zeros(n)])


class TestHeadingDirectionLabels:
    def test_still_animal_is_stationary_at_min_speed_zero(self):
        labels = heading_direction_labels(_times(10), np.zeros((10, 2)), min_speed=0.0)
        assert labels.tolist() == ["stationary"] * 10

    def test_nan_row_splits_runs(self):
        times, positions = _times(20), _east_run(20, 2.0)  # 20 cm/s East
        positions[10] = np.nan
        labels = heading_direction_labels(times, positions, min_speed=1.0)
        before = heading_direction_labels(times[:10], positions[:10], min_speed=1.0)
        after = heading_direction_labels(times[11:], positions[11:], min_speed=1.0)
        assert labels[10] is None
        np.testing.assert_array_equal(labels[:10], before)
        np.testing.assert_array_equal(labels[11:], after)
        assert np.r_[before, after].tolist() == [EAST] * 19

    def test_isolated_sample_is_not_observed(self):
        times, positions = _times(20), _east_run(20, 2.0)
        positions[[9, 11]] = np.nan
        labels = heading_direction_labels(times, positions, min_speed=1.0)
        assert labels[9:12].tolist() == [None, None, None]

    def test_one_sample_is_not_observed(self):
        labels = heading_direction_labels(
            np.array([0.0]), np.array([[0.0, 0.0]]), min_speed=1.0
        )
        assert labels.dtype == object
        assert labels.tolist() == [None]

    def test_empty_input_raises_naming_the_call(self):
        with pytest.raises(ValueError, match="heading_direction_labels"):
            heading_direction_labels(
                np.array([], dtype=float), np.zeros((0, 2)), min_speed=1.0
            )

    def test_precomputed_kinds(self):
        labels = heading_direction_labels(
            speed=np.array([10.0, 1.0, np.nan, 10.0, 0.0]),
            heading=np.array([0.3, 0.3, 0.3, np.nan, np.nan]),
            min_speed=5.0,
        )
        assert labels.tolist() == [EAST, "stationary", None, None, "stationary"]


class TestGoalAlignment:
    def test_never_moving_animal_has_no_goal_bias(self):
        times, positions = _times(10), np.zeros((10, 2))
        goal = np.array([100.0, 0.0])
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            bias = goal_bias(times, positions, goal, min_speed=0.0)
            alignment = instantaneous_goal_alignment(
                times, positions, goal, min_speed=0.0
            )
        assert np.isnan(bias)
        assert np.isnan(alignment).all()

    def test_nan_row_splits_alignment(self):
        times = _times(21)
        positions = np.column_stack([np.linspace(0, 100, 21), np.zeros(21)])
        positions[10] = np.nan
        goal = np.array([200.0, 0.0])
        alignment = instantaneous_goal_alignment(times, positions, goal, min_speed=1.0)
        before = instantaneous_goal_alignment(
            times[:10], positions[:10], goal, min_speed=1.0
        )
        after = instantaneous_goal_alignment(
            times[11:], positions[11:], goal, min_speed=1.0
        )
        assert np.isnan(alignment[10])
        np.testing.assert_array_equal(alignment[:10], before)
        np.testing.assert_array_equal(alignment[11:], after)
        np.testing.assert_allclose(np.r_[before, after], 1.0, rtol=0, atol=1e-12)


def test_goal_metrics_with_missing_rows():
    times = _times(21)
    positions = np.column_stack([np.linspace(0, 100, 21), np.zeros(21)])
    positions[10] = np.nan
    goal = np.array([100.0, 0.0])
    env = Environment.from_samples(
        np.column_stack([np.linspace(0, 100, 101), np.zeros(101)]),
        bin_size=5.0,
        units="cm",
    )
    result = compute_goal_directed_metrics(env, times, positions, goal, min_speed=1.0)
    assert np.isfinite(result.goal_bias)
    assert result.min_distance_to_goal == 0.0

    positions[0] = np.nan
    result = compute_goal_directed_metrics(env, times, positions, goal, min_speed=1.0)
    # The first observed sample is at x = 5.
    assert result.goal_distance_at_start == 95.0


@pytest.mark.parametrize(
    "name",
    [
        "heading_direction_labels",
        "instantaneous_goal_alignment",
        "goal_bias",
        "compute_goal_directed_metrics",
    ],
)
def test_errors_name_the_called_function(name):
    times = np.array([0.0, 0.1, 0.1, 0.2, 0.3])
    positions = np.column_stack([np.arange(5.0), np.zeros(5)])
    goal = np.array([10.0, 0.0])
    env = Environment.from_samples(
        np.column_stack([np.arange(11.0), np.zeros(11)]), bin_size=1.0, units="cm"
    )
    calls = {
        "heading_direction_labels": lambda: heading_direction_labels(
            times, positions, min_speed=1.0
        ),
        "instantaneous_goal_alignment": lambda: instantaneous_goal_alignment(
            times, positions, goal, min_speed=1.0
        ),
        "goal_bias": lambda: goal_bias(times, positions, goal, min_speed=1.0),
        "compute_goal_directed_metrics": lambda: compute_goal_directed_metrics(
            env, times, positions, goal, min_speed=1.0
        ),
    }
    with pytest.raises(ValueError) as error:
        calls[name]()
    message = str(error.value)
    assert message.startswith(name)
    assert "heading_from_velocity" not in message


def test_goal_distances_are_nan_when_nothing_is_observed():
    env = Environment.from_samples(
        np.column_stack([np.linspace(0, 100, 101), np.zeros(101)]),
        bin_size=5.0,
        units="cm",
    )
    positions = np.full((10, 2), np.nan)
    result = compute_goal_directed_metrics(
        env, _times(10), positions, np.array([100.0, 0.0]), min_speed=1.0
    )
    assert np.isnan(result.min_distance_to_goal)
    assert np.isnan(result.goal_distance_at_start)
    assert np.isnan(result.goal_distance_at_end)
    assert np.isnan(result.goal_bias)


def _calls_with(times, positions, goal):
    """Every public caller of the x/y and goal-dimension checks, by name."""
    from neurospatial import Regions
    from neurospatial.behavior import (
        approach_rate,
        compute_decision_analysis,
        compute_pre_decision_metrics,
        compute_vte_session,
        compute_vte_trial,
        goal_direction,
        goal_vector,
        head_sweep_from_positions,
        heading_from_velocity,
        pre_decision_heading_stats,
    )

    env = Environment.from_samples(
        np.column_stack([np.arange(11.0), np.arange(11.0)]), bin_size=1.0, units="cm"
    )
    regions = Regions(env.regions.values())
    regions.add("center", point=(5.0, 5.0))
    regions.add("goal", point=(9.0, 9.0))
    env = env.with_regions(regions)
    return {
        "heading_from_velocity": lambda: heading_from_velocity(times, positions),
        "heading_direction_labels": lambda: heading_direction_labels(
            times, positions, min_speed=1.0
        ),
        "instantaneous_goal_alignment": lambda: instantaneous_goal_alignment(
            times, positions, goal, min_speed=1.0
        ),
        "goal_bias": lambda: goal_bias(times, positions, goal, min_speed=1.0),
        "compute_goal_directed_metrics": lambda: compute_goal_directed_metrics(
            env, times, positions, goal, min_speed=1.0
        ),
        "pre_decision_heading_stats": lambda: pre_decision_heading_stats(
            times, positions, min_speed=1.0
        ),
        "compute_pre_decision_metrics": lambda: compute_pre_decision_metrics(
            times, positions, 0.5, window_duration=0.5, min_speed=1.0
        ),
        "head_sweep_from_positions": lambda: head_sweep_from_positions(
            times, positions, min_speed=1.0
        ),
        "compute_vte_trial": lambda: compute_vte_trial(
            times, positions, 0.5, window_duration=0.5, min_speed=1.0
        ),
        "compute_decision_analysis": lambda: compute_decision_analysis(
            env,
            times,
            positions,
            decision_region="center",
            goal_regions=["goal"],
            min_speed=1.0,
        ),
        "compute_vte_session": lambda: compute_vte_session(
            env,
            times,
            positions,
            decision_region="center",
            trials=[],
            min_speed=1.0,
        ),
        "goal_vector": lambda: goal_vector(positions, goal),
        "goal_direction": lambda: goal_direction(positions, goal),
        "approach_rate": lambda: approach_rate(times, positions, goal),
    }


@pytest.mark.parametrize(
    "name",
    [
        "heading_from_velocity",
        "heading_direction_labels",
        "instantaneous_goal_alignment",
        "goal_bias",
        "compute_goal_directed_metrics",
        "pre_decision_heading_stats",
        "compute_pre_decision_metrics",
        "head_sweep_from_positions",
        "compute_vte_trial",
        "compute_decision_analysis",
        "compute_vte_session",
    ],
)
def test_one_dimensional_positions_error_names_the_call(name):
    times = _times(10)
    positions = np.arange(10.0)  # a linearized track: no x and y
    with pytest.raises(ValueError) as error:
        _calls_with(times, positions, np.array([10.0]))[name]()
    message = str(error.value)
    assert message.startswith(f"{name}() needs positions with x and y columns")
    assert "got shape (10,)" in message
    assert "Why:" in message
    assert "Fix:" in message


@pytest.mark.parametrize(
    "name",
    [
        "goal_vector",
        "goal_direction",
        "instantaneous_goal_alignment",
        "goal_bias",
        "approach_rate",
        "compute_goal_directed_metrics",
    ],
)
def test_goal_dimension_error_names_the_call(name):
    times = _times(10)
    positions = _east_run(10, 1.0)
    goal = np.array([10.0, 0.0, 0.0])
    with pytest.raises(ValueError) as error:
        _calls_with(times, positions, goal)[name]()
    message = str(error.value)
    assert message.startswith(
        f"{name}() needs a goal with one coordinate per position dimension"
    )
    assert "goal has 3 coordinates but positions have 2" in message
    assert "Why:" in message
    assert "Fix: pass goal with shape (2,)" in message
