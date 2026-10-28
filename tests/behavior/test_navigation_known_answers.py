"""Known-answer tests for the bin-based navigation covariates.

Every expected value here is derived from the geometry of the environment,
not from the code under test.
"""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial import BinIndexOutOfRangeError, Environment
from neurospatial.behavior import cost_to_goal, distance_to_region, path_progress


@pytest.fixture(scope="module")
def corridor() -> Environment:
    """A 1 cm wide, 100 cm long corridor of 1 cm bins centred at 0.5 ... 99.5."""
    x = np.arange(0.5, 100.0, 1.0)
    return Environment.from_samples(np.c_[x, np.zeros_like(x)], bin_size=1.0)


def _bin(env: Environment, x: float) -> int:
    return int(env.bin_at(np.array([[x, 0.0]]))[0])


class TestPathProgress:
    def test_halfway_along_a_corridor_is_one_half(self, corridor):
        start, goal = _bin(corridor, 10.5), _bin(corridor, 90.5)
        position_bins = np.array(
            [start, _bin(corridor, 30.5), _bin(corridor, 50.5), goal]
        )
        for metric in ("geodesic", "euclidean"):
            progress = path_progress(
                position_bins,
                corridor,
                start_bins=start,
                goal_bins=goal,
                metric=metric,
            )
            np.testing.assert_allclose(progress, [0.0, 0.25, 0.5, 1.0])

    def test_start_equal_to_goal_has_no_progress_away_from_it(self, corridor):
        """With start == goal the ratio is 0/0: only the goal bin itself is 1."""
        goal = _bin(corridor, 50.5)
        position_bins = np.array([goal, _bin(corridor, 60.5), -1])
        progress = path_progress(
            position_bins, corridor, start_bins=goal, goal_bins=goal
        )
        np.testing.assert_array_equal(progress, [1.0, np.nan, np.nan])

    def test_float_position_bins_are_refused(self, corridor):
        with pytest.raises(TypeError, match=r"position_bins"):
            path_progress(np.array([0.7, 3.2]), corridor, start_bins=0, goal_bins=5)

    def test_negative_bins_other_than_minus_one_are_refused(self, corridor):
        """-2 would index the second-to-last bin instead of meaning 'outside'."""
        with pytest.raises(BinIndexOutOfRangeError):
            path_progress(np.array([0, -2, 3]), corridor, start_bins=0, goal_bins=5)


@pytest.mark.parametrize(
    "call",
    [
        lambda bins, env: distance_to_region(bins, env, target_bins=5),
        lambda bins, env: cost_to_goal(bins, env, goal_bins=5),
        lambda bins, env: cost_to_goal(
            bins, env, goal_bins=5, cost_map=np.zeros(env.n_bins)
        ),
    ],
    ids=["distance_to_region", "cost_to_goal", "cost_to_goal_weighted"],
)
class TestPositionBinsValidation:
    def test_minus_two_is_refused(self, corridor, call):
        with pytest.raises(BinIndexOutOfRangeError):
            call(np.array([0, -2, 3]), corridor)

    def test_past_the_last_bin_is_refused(self, corridor, call):
        with pytest.raises(BinIndexOutOfRangeError):
            call(np.array([0, corridor.n_bins]), corridor)

    def test_floats_are_refused(self, corridor, call):
        with pytest.raises(TypeError, match=r"position_bins"):
            call(np.array([0.7, 3.2]), corridor)

    def test_minus_one_is_missing(self, corridor, call):
        result = call(np.array([0, -1, 5]), corridor)
        assert np.isnan(result[1])
        np.testing.assert_allclose(result[[0, 2]], [5.0, 0.0])


def _corridor(bin_size: float) -> Environment:
    """A 100 cm corridor of ``bin_size`` cm bins centred at b/2, 3b/2, ..."""
    x = np.arange(bin_size / 2, 100.0, bin_size)
    return Environment.from_samples(np.c_[x, np.zeros_like(x)], bin_size=bin_size)


class TestCostToGoal:
    """Cost is a per-unit-length rate integrated along the path.

    The start bin centre is at 20 - b/2 and the goal bin centre at 100 - b/2,
    80 cm apart for every bin size b.
    """

    @pytest.mark.parametrize("bin_size", [1.0, 2.0, 5.0])
    def test_uniform_cost_scales_the_length_independent_of_bin_size(self, bin_size):
        env = _corridor(bin_size)
        start = _bin(env, 20 - bin_size / 2)
        goal = _bin(env, 100 - bin_size / 2)
        bins = np.array([start, goal])
        ones = np.ones(env.n_bins)

        np.testing.assert_allclose(cost_to_goal(bins, env, goal_bins=goal), [80, 0])
        np.testing.assert_allclose(
            cost_to_goal(bins, env, goal_bins=goal, cost_map=np.zeros(env.n_bins)),
            [80, 0],
        )
        np.testing.assert_allclose(
            cost_to_goal(bins, env, goal_bins=goal, cost_map=ones), [160, 0]
        )
        np.testing.assert_allclose(
            cost_to_goal(bins, env, goal_bins=goal, terrain_difficulty=2 * ones),
            [160, 0],
        )
        np.testing.assert_allclose(
            cost_to_goal(
                bins, env, goal_bins=goal, cost_map=ones, terrain_difficulty=2 * ones
            ),
            [320, 0],
        )

    @pytest.mark.parametrize("bin_size", [1.0, 2.0, 5.0])
    def test_a_punished_stretch_adds_its_length_times_the_cost(self, bin_size):
        """Bins centred in [40, 60) cost 9 extra per cm: 80 + 9 * 20 = 260."""
        env = _corridor(bin_size)
        start = _bin(env, 20 - bin_size / 2)
        goal = _bin(env, 100 - bin_size / 2)
        centers = env.bin_centers[:, 0]
        cost_map = np.where((centers >= 40) & (centers < 60), 9.0, 0.0)

        result = cost_to_goal(np.array([start]), env, goal_bins=goal, cost_map=cost_map)

        np.testing.assert_allclose(result, [260.0])

    @pytest.mark.parametrize(
        ("name", "kind", "match"),
        [
            ("cost_map", "negative", r"cost_map.*negative"),
            ("cost_map", "nan", r"cost_map.*finite"),
            ("cost_map", "short", r"cost_map.*shape"),
            ("terrain_difficulty", "zero", r"terrain_difficulty.*positive"),
            ("terrain_difficulty", "nan", r"terrain_difficulty.*finite"),
            ("terrain_difficulty", "short", r"terrain_difficulty.*shape"),
        ],
    )
    def test_invalid_costs_are_refused(self, corridor, name, kind, match):
        """A negative weight hung Dijkstra; NaN silently gave inf everywhere."""
        values = {
            "negative": np.r_[-1.0, np.zeros(corridor.n_bins - 1)],
            "zero": np.r_[0.0, np.ones(corridor.n_bins - 1)],
            "nan": np.r_[np.nan, np.ones(corridor.n_bins - 1)],
            "short": np.ones(corridor.n_bins - 1),
        }[kind]
        with pytest.raises(ValueError, match=match):
            cost_to_goal(np.array([0, 1]), corridor, goal_bins=5, **{name: values})

    def test_samples_outside_the_environment_are_nan(self, corridor):
        result = cost_to_goal(
            np.array([0, -1]),
            corridor,
            goal_bins=5,
            cost_map=np.ones(corridor.n_bins),
        )
        np.testing.assert_array_equal(result, [10.0, np.nan])
