"""Known-answer tests for the bin-based navigation covariates.

Every expected value here is derived from the geometry of the environment,
not from the code under test.
"""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial import BinIndexOutOfRangeError, Environment
from neurospatial.behavior import (
    cost_to_goal,
    distance_to_region,
    graph_turn_sequence,
    path_progress,
)


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


@pytest.fixture(scope="module")
def t_maze() -> Environment:
    """A T-maze track: a 50 cm stem heading North and two 50 cm arms, 5 cm bins."""
    import networkx as nx

    graph = nx.Graph()
    graph.add_node(0, pos=(0.0, -50.0))
    graph.add_node(1, pos=(0.0, 0.0))
    graph.add_node(2, pos=(-50.0, 0.0))
    graph.add_node(3, pos=(50.0, 0.0))
    for u, v in [(0, 1), (1, 2), (1, 3)]:
        graph.add_edge(u, v, distance=50.0)
    return Environment.from_graph(
        graph,
        edge_order=[(0, 1), (1, 2), (1, 3)],
        edge_spacing=0.0,
        bin_size=5.0,
        track_width=5.0,
    )


def _t_maze_run(env: Environment, arm_x: float, rate_hz: float) -> np.ndarray:
    """Bins of a 20 cm/s run up the stem and out to the arm end at ``arm_x``."""
    n = round(50 / 20 * rate_hz)
    stem = np.c_[np.zeros(n), np.linspace(-49.0, -1.0, n)]
    arm = np.c_[np.linspace(0.0, arm_x, n), np.zeros(n)]
    return env.bin_at(np.vstack([stem, arm]))


class TestGraphTurnSequence:
    @pytest.mark.parametrize("rate_hz", [30.0, 500.0])
    @pytest.mark.parametrize(("arm_x", "expected"), [(-49.0, "left"), (49.0, "right")])
    def test_one_turn_at_a_t_junction_at_any_frame_rate(
        self, t_maze, rate_hz, arm_x, expected
    ):
        """Heading North, the West arm is a left turn, the East arm a right."""
        bins = _t_maze_run(t_maze, arm_x, rate_hz)
        assert graph_turn_sequence(t_maze, bins) == expected

    def test_a_straight_oblique_run_has_no_turns(self):
        """A lattice staircase along a straight line is not a sequence of turns."""
        x, y = np.meshgrid(np.arange(0.0, 50.0), np.arange(0.0, 50.0))
        field = Environment.from_samples(np.c_[x.ravel(), y.ravel()], bin_size=2.0)
        t = np.linspace(0, 1, 2000)
        run = np.c_[5 + 30 * t, 5 + 20 * t]
        assert graph_turn_sequence(field, field.bin_at(run)) == ""

    def test_jitter_at_a_bin_border_is_not_a_turn(self):
        """Flicking between two bins, then running straight East: no turn."""
        x, y = np.meshgrid(np.arange(0.0, 50.0), np.arange(0.0, 50.0))
        field = Environment.from_samples(np.c_[x.ravel(), y.ravel()], bin_size=2.0)
        rng = np.random.default_rng(0)
        dwell = np.c_[np.full(200, 10.0), 20.0 + rng.normal(0, 0.3, 200)]
        run = np.c_[np.linspace(10.0, 40.0, 300), 20.0 + rng.normal(0, 0.3, 300)]
        bins = field.bin_at(np.vstack([dwell, run]))
        assert len(np.unique(bins[:200])) == 2  # the dwell does flicker
        assert graph_turn_sequence(field, bins) == ""

    def test_noisy_corner_in_an_open_field_is_one_left_turn(self):
        """East 30 cm, then North 30 cm, with 0.3 cm tracking noise: one left."""
        x, y = np.meshgrid(np.arange(0.0, 50.0), np.arange(0.0, 50.0))
        field = Environment.from_samples(np.c_[x.ravel(), y.ravel()], bin_size=2.0)
        rng = np.random.default_rng(1)
        east = np.c_[np.linspace(5.0, 35.0, 300), np.full(300, 5.0)]
        north = np.c_[np.full(300, 35.0), np.linspace(5.0, 35.0, 300)]
        run = np.vstack([east, north]) + rng.normal(0, 0.3, (600, 2))
        assert graph_turn_sequence(field, field.bin_at(run)) == "left"

    def test_samples_outside_the_environment_are_skipped(self, t_maze):
        """-1 must not read as the last bin and invent a turn there."""
        bins = _t_maze_run(t_maze, -49.0, 30.0)
        bins[::7] = -1
        assert graph_turn_sequence(t_maze, bins) == "left"

    def test_float_bins_are_refused(self, t_maze):
        with pytest.raises(TypeError, match=r"position_bins"):
            graph_turn_sequence(t_maze, np.array([0.0, 1.0, 2.0]))

    def test_out_of_range_bins_are_refused(self, t_maze):
        with pytest.raises(BinIndexOutOfRangeError):
            graph_turn_sequence(t_maze, np.array([0, 1, t_maze.n_bins]))
