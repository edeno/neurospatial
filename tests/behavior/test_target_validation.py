"""Analysis functions validate their bin targets like the environment queries.

A coordinate passed where a bin index is expected is refused, not truncated
to an unrelated bin; ``-1`` keeps meaning "no target for this sample" where
the targets are per-sample.
"""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial import BinIndexOutOfRangeError, Environment
from neurospatial.behavior import (
    cost_to_goal,
    distance_to_decision_boundary,
    distance_to_region,
    geodesic_voronoi_labels,
    goal_reward_field,
    graph_turn_sequence,
    path_progress,
    shortest_path_length,
)


@pytest.fixture(scope="module")
def env() -> Environment:
    x, y = np.meshgrid(np.arange(0.0, 30.0), np.arange(0.0, 30.0))
    return Environment.from_samples(np.c_[x.ravel(), y.ravel()], bin_size=3.0)


@pytest.fixture(scope="module")
def position_bins(env) -> np.ndarray:
    return np.arange(10, dtype=np.intp) % env.n_bins


def _calls(env, position_bins, target):
    """Each analysis function with ``target`` as its bin target."""
    per_sample = np.full(len(position_bins), target)
    return {
        "distance_to_region": lambda: distance_to_region(
            position_bins, env, target_bins=target
        ),
        "cost_to_goal": lambda: cost_to_goal(position_bins, env, goal_bins=target),
        "cost_to_goal_weighted": lambda: cost_to_goal(
            position_bins, env, goal_bins=target, cost_map=np.ones(env.n_bins)
        ),
        "path_progress": lambda: path_progress(
            position_bins, env, start_bins=per_sample, goal_bins=per_sample
        ),
        "path_progress_single": lambda: path_progress(
            position_bins, env, start_bins=target, goal_bins=target
        ),
        "geodesic_voronoi_labels": lambda: geodesic_voronoi_labels(
            env, np.r_[target, target]
        ),
        "distance_to_decision_boundary": lambda: distance_to_decision_boundary(
            env, position_bins, np.r_[target, target]
        ),
        "goal_reward_field": lambda: goal_reward_field(env, target),
        "graph_turn_sequence": lambda: graph_turn_sequence(
            env, position_bins, target, 0
        ),
    }


_NAMES = sorted(_calls(None, np.zeros(1), 0))
# Every call whose target is one bin, not a per-sample array.
_SINGLE_TARGET_NAMES = [name for name in _NAMES if name != "path_progress"]


@pytest.mark.parametrize("name", _NAMES)
def test_float_target_points_to_points(env, position_bins, name):
    call = _calls(env, position_bins, 12.7)[name]
    with pytest.raises(TypeError, match=r"integer bin indices[\s\S]*points="):
        call()


@pytest.mark.parametrize("name", _NAMES)
def test_out_of_range_target_raises(env, position_bins, name):
    call = _calls(env, position_bins, env.n_bins + 5)[name]
    with pytest.raises(BinIndexOutOfRangeError):
        call()


@pytest.mark.parametrize("name", ["distance_to_region", "cost_to_goal"])
def test_boolean_target_is_refused(env, position_bins, name):
    with pytest.raises(TypeError, match="booleans"):
        _calls(env, position_bins, True)[name]()


def test_integer_targets_give_the_query_answer(env, position_bins):
    goal = 40
    expected = env.distance_to(bins=goal)[position_bins]
    np.testing.assert_array_equal(
        distance_to_region(position_bins, env, target_bins=np.int64(goal)), expected
    )
    np.testing.assert_array_equal(
        cost_to_goal(position_bins, env, goal_bins=goal), expected
    )


def test_minus_one_still_marks_samples_without_a_target(env, position_bins):
    targets = np.where(np.arange(len(position_bins)) < 5, 40, -1)
    distances = distance_to_region(position_bins, env, target_bins=targets)
    assert np.all(np.isnan(distances[5:]))
    assert np.all(np.isfinite(distances[:5]))
    progress = path_progress(
        position_bins, env, start_bins=targets, goal_bins=targets[::-1]
    )
    assert np.all(np.isnan(progress))  # every sample lacks a start or a goal


@pytest.mark.parametrize("name", _SINGLE_TARGET_NAMES)
@pytest.mark.parametrize("shape", ["scalar", "one_element"])
def test_minus_one_is_never_a_single_target(env, position_bins, name, shape):
    """-1 is bin_at's outside marker; as one target it is refused, not NaN.

    ``env.bin_at([[x, y]])`` returns a one-element array, so a shape-(1,)
    target must be read as one bin, not as a per-sample array.
    """
    target = -1 if shape == "scalar" else np.array([-1])
    with pytest.raises(BinIndexOutOfRangeError, match=r"-1 is the value bin_at gives"):
        _calls(env, position_bins, target)[name]()


@pytest.mark.parametrize("weighted", [False, True])
def test_bin_at_output_of_an_outside_goal_is_refused(env, position_bins, weighted):
    goal = env.bin_at(np.array([[500.0, 500.0]]))  # shape (1,), value -1
    cost_map = np.ones(env.n_bins) if weighted else None
    with pytest.raises(BinIndexOutOfRangeError):
        distance_to_region(position_bins, env, target_bins=goal)
    with pytest.raises(BinIndexOutOfRangeError):
        cost_to_goal(position_bins, env, goal_bins=goal, cost_map=cost_map)


@pytest.mark.parametrize("weighted", [False, True])
def test_one_element_target_equals_the_scalar_target(env, position_bins, weighted):
    cost_map = np.ones(env.n_bins) if weighted else None
    np.testing.assert_array_equal(
        cost_to_goal(position_bins, env, goal_bins=np.array([40]), cost_map=cost_map),
        cost_to_goal(position_bins, env, goal_bins=40, cost_map=cost_map),
    )
    np.testing.assert_array_equal(
        distance_to_region(position_bins, env, target_bins=np.array([40])),
        distance_to_region(position_bins, env, target_bins=40),
    )


def test_per_sample_targets_of_the_wrong_length_raise(env, position_bins):
    with pytest.raises(ValueError, match=r"shape \(3,\), but there are 10 samples"):
        distance_to_region(position_bins, env, target_bins=np.array([1, 2, 3]))


def test_approach_rate_refuses_a_goal_outside(env):
    from neurospatial.behavior import approach_rate

    times = np.arange(5) / 10.0
    positions = np.c_[np.linspace(2.0, 20.0, 5), np.full(5, 5.0)]
    with pytest.raises(ValueError, match="outside the environment"):
        approach_rate(times, positions, [500.0, 3.0], env=env, metric="geodesic")
    rates = approach_rate(times, positions, [27.0, 5.0], env=env, metric="geodesic")
    assert np.all(rates[1:] < 0)  # moving east toward the goal


@pytest.mark.parametrize("bins", [-1, -2, [3, -1]])
def test_queries_refuse_negative_bins(env, bins):
    with pytest.raises(BinIndexOutOfRangeError, match=r"Fix: pass a bin index in \[0,"):
        env.distance_to(bins=bins)


def test_polar_hint_names_a_polar_lookup():
    polar = Environment.from_polar_egocentric(
        distance_range=(0.0, 30.0),
        angle_range=(-np.pi, np.pi),
        distance_bin_size=5.0,
        angle_bin_size=np.pi / 6,
    )
    with pytest.raises(TypeError) as caught:
        distance_to_region(np.arange(3), polar, target_bins=1.5)
    assert "env.nearest_bin(points)" in str(caught.value)
    assert "bin_at" not in str(caught.value)


def test_shortest_path_length_is_the_geodesic_distance_between_points(env):
    start, goal = np.array([1.0, 1.0]), np.array([28.0, 27.0])
    assert shortest_path_length(env, start, goal) == env.distance_between(
        points=[start, goal]
    )


@pytest.mark.parametrize("outside", ["start", "goal"])
def test_shortest_path_length_refuses_points_outside(env, outside):
    """An outside start used to return the distance of the last bin."""
    inside, far = np.array([1.0, 1.0]), np.array([500.0, 500.0])
    start, goal = (far, inside) if outside == "start" else (inside, far)
    with pytest.raises(ValueError, match="outside the environment"):
        shortest_path_length(env, start, goal)
