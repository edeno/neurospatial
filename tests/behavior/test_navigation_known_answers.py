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
