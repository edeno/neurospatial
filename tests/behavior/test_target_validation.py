"""Analysis functions validate their targets like the environment queries.

A coordinate passed where a bin index is expected is refused, not truncated
to an unrelated bin, and a goal point outside the environment is refused
where a geodesic distance needs it inside.
"""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial import BinIndexOutOfRangeError, Environment
from neurospatial.behavior import (
    distance_to_decision_boundary,
    distance_to_goal,
    geodesic_voronoi_labels,
    goal_reward_field,
    shortest_path_length,
)


@pytest.fixture(scope="module")
def env() -> Environment:
    x, y = np.meshgrid(np.arange(0.0, 30.0), np.arange(0.0, 30.0))
    return Environment.from_samples(np.c_[x.ravel(), y.ravel()], bin_size=3.0)


@pytest.fixture(scope="module")
def position_bins(env) -> np.ndarray:
    return np.arange(10, dtype=np.intp) % env.n_bins


def _bin_calls(env, target):
    """Each analysis function that takes ``target`` as a bin index."""
    return {"goal_reward_field": lambda: goal_reward_field(env, target)}


_BIN_NAMES = sorted(_bin_calls(None, 0))


@pytest.mark.parametrize("name", _BIN_NAMES)
def test_float_target_points_to_points(env, name):
    call = _bin_calls(env, 12.7)[name]
    with pytest.raises(TypeError, match=r"integer bin indices[\s\S]*points="):
        call()


@pytest.mark.parametrize("name", _BIN_NAMES)
def test_out_of_range_target_raises(env, name):
    call = _bin_calls(env, env.n_bins + 5)[name]
    with pytest.raises(BinIndexOutOfRangeError):
        call()


@pytest.mark.parametrize("name", _BIN_NAMES)
def test_boolean_target_is_refused(env, name):
    with pytest.raises(TypeError, match="booleans"):
        _bin_calls(env, True)[name]()


@pytest.mark.parametrize("name", _BIN_NAMES)
@pytest.mark.parametrize("shape", ["scalar", "one_element"])
def test_minus_one_is_never_a_single_target(env, name, shape):
    """-1 is bin_at's outside marker; as one target it is refused, not NaN.

    ``env.bin_at([[x, y]])`` returns a one-element array, so a shape-(1,)
    target must be read as one bin, not as a per-sample array.
    """
    target = -1 if shape == "scalar" else np.array([-1])
    with pytest.raises(BinIndexOutOfRangeError, match=r"-1 is the value bin_at gives"):
        _bin_calls(env, target)[name]()


def _goal_calls(env, position_bins, goal):
    """Each goal function with the goal point ``goal``."""
    positions = env.bin_centers[position_bins]
    times = np.arange(len(positions), dtype=float)
    return {
        "distance_to_goal": lambda: distance_to_goal(env, times, positions, goal=goal),
        "distance_to_goal_weighted": lambda: distance_to_goal(
            env, times, positions, goal=goal, cost_map=np.ones(env.n_bins)
        ),
        "geodesic_voronoi_labels": lambda: geodesic_voronoi_labels(
            env, goals=[goal, env.bin_centers[0]]
        ),
        "distance_to_decision_boundary": lambda: distance_to_decision_boundary(
            env, times, positions, goals=[goal, env.bin_centers[0]]
        ),
    }


_GOAL_NAMES = [
    "distance_to_decision_boundary",
    "distance_to_goal",
    "distance_to_goal_weighted",
    "geodesic_voronoi_labels",
]


@pytest.mark.parametrize("name", _GOAL_NAMES)
def test_goal_point_outside_the_environment_is_refused(env, position_bins, name):
    """A geodesic path cannot end outside the environment."""
    with pytest.raises(ValueError, match="outside the environment"):
        _goal_calls(env, position_bins, np.array([500.0, 500.0]))[name]()


@pytest.mark.parametrize("name", _GOAL_NAMES)
def test_goal_point_of_the_wrong_dimension_is_refused(env, position_bins, name):
    with pytest.raises(ValueError, match=r"one\s+coordinate per environment"):
        _goal_calls(env, position_bins, np.array([5.0, 5.0, 5.0]))[name]()


def test_point_goal_gives_the_query_answer(env, position_bins):
    goal = 40
    # The square arena is convex, so the shortest path inside it between two
    # bin centers is the straight line between them.
    centers = env.bin_centers
    expected = np.linalg.norm(centers[position_bins] - centers[goal], axis=1)
    np.testing.assert_allclose(
        distance_to_goal(
            env,
            np.arange(len(position_bins), dtype=float),
            centers[position_bins],
            goal=centers[goal],
        ),
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


def test_approach_rate_refuses_a_goal_outside(env):
    from neurospatial.behavior import approach_rate

    times = np.arange(5) / 10.0
    positions = np.c_[np.linspace(2.0, 20.0, 5), np.full(5, 5.0)]
    with pytest.raises(ValueError, match="outside the environment"):
        approach_rate(env, times, positions, goal=[500.0, 3.0], metric="geodesic")
    rates = approach_rate(env, times, positions, goal=[27.0, 5.0], metric="geodesic")
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
        goal_reward_field(polar, 1.5)
    assert "(distance, angle) rows, use env.bin_at(points)" in str(caught.value)


def test_shortest_path_length_in_an_open_field_is_the_straight_line(env):
    """The straight line stays inside the arena, so it is the shortest path."""
    start, goal = np.array([1.0, 1.0]), np.array([28.0, 27.0])
    assert shortest_path_length(env, start, goal) == pytest.approx(
        np.hypot(27.0, 26.0), rel=1e-12
    )


@pytest.mark.parametrize("outside", ["start", "goal"])
def test_shortest_path_length_refuses_points_outside(env, outside):
    """An outside start used to return the distance of the last bin."""
    inside, far = np.array([1.0, 1.0]), np.array([500.0, 500.0])
    start, goal = (far, inside) if outside == "start" else (inside, far)
    with pytest.raises(ValueError, match="outside the environment"):
        shortest_path_length(env, start, goal)


@pytest.fixture(scope="module")
def polar() -> Environment:
    return Environment.from_polar_egocentric(
        distance_range=(0.0, 30.0),
        angle_range=(-np.pi, np.pi),
        distance_bin_size=5.0,
        angle_bin_size=np.pi / 6,
    )


def test_shortest_path_length_refuses_polar_environments(polar):
    """Allocentric start/goal positions are not (distance, angle) bin centers."""
    from neurospatial._exceptions import CartesianOnlyError

    with pytest.raises(CartesianOnlyError, match="shortest_path_length"):
        shortest_path_length(polar, np.array([10.0, 1.0]), np.array([20.0, -2.0]))


def test_approach_rate_refuses_polar_environments(polar):
    from neurospatial._exceptions import CartesianOnlyError
    from neurospatial.behavior import approach_rate

    times = np.arange(3) / 10.0
    positions = np.array([[10.0, 0.0], [12.0, 0.0], [14.0, 0.0]])
    with pytest.raises(CartesianOnlyError, match="approach_rate"):
        approach_rate(polar, times, positions, goal=[20.0, 0.0], metric="geodesic")
