"""Tests for single-arm and joint-arm replay on a maze (Wu & Foster 2014)."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.decoding import DecodingResult
from neurospatial.replay import (
    ArmReplayMethod,
    compare_replay_methods,
    detect_arm_replay,
)
from neurospatial.replay.methods import WU_FOSTER_2014

DT = 0.01


@pytest.fixture(scope="module")
def y_maze():
    """Three 40 cm arms in 4 cm bins: center 0-9, left 10-19, right 20-29."""
    graph = nx.Graph()
    for node, xy in {0: (0, 0), 1: (0, -40), 2: (-28, 28), 3: (28, 28)}.items():
        graph.add_node(node, pos=xy)
    graph.add_edges_from([(0, 1), (0, 2), (0, 3)])
    env = Environment.from_graph(
        graph,
        edge_order=[(1, 0), (0, 2), (0, 3)],
        edge_spacing=0.0,
        bin_size=4.0,
        units="cm",
    )
    assert env.n_bins == 30
    arms = {
        "center": np.arange(10)[::-1],
        "left": np.arange(10, 20),
        "right": np.arange(20, 30),
    }
    return env, arms


def _detect(maze, posterior, method=WU_FOSTER_2014):
    env, arms = maze
    return detect_arm_replay(
        env,
        posterior,
        np.arange(len(posterior)) * DT,
        arms=arms,
        arm_replay_method=method,
    )


class TestWuFoster:
    def test_a_path_through_the_junction_is_a_joint_replay(self, y_maze):
        env, arms = y_maze
        path = np.r_[arms["center"][::-1], arms["left"]]
        result = _detect(y_maze, np.eye(env.n_bins)[path])
        assert result.is_replay and result.template == "center+left"
        assert result.score == pytest.approx(1.0)
        assert result.criteria["center+left: same direction"]

    def test_one_arm_alone_is_a_single_arm_replay(self, y_maze):
        env, arms = y_maze
        result = _detect(y_maze, np.eye(env.n_bins)[arms["left"]])
        assert result.is_replay and result.template == "left"
        assert not result.criteria["center: time bins >= 5"]

    def test_arms_replayed_the_wrong_way_round_are_not_joint(self, y_maze):
        """Center outward then left outward: two single-arm replays."""
        env, arms = y_maze
        path = np.r_[arms["center"], arms["left"]]
        result = _detect(y_maze, np.eye(env.n_bins)[path])
        assert not result.criteria["center+left: same direction"]
        assert result.is_replay and result.template == "center,left"

    def test_two_joint_replays_cancel(self, y_maze):
        """Into the junction from the center, then out along left and right."""
        env, arms = y_maze
        posterior = np.zeros((20, env.n_bins))
        posterior[np.arange(10), arms["center"][::-1]] = 1.0
        posterior[np.arange(10, 20), arms["left"]] = 0.5
        posterior[np.arange(10, 20), arms["right"]] = 0.5
        result = _detect(y_maze, posterior)
        assert result.criteria["center+left: |wc| > 0.5"]
        assert result.criteria["center+right: |wc| > 0.5"]
        assert not result.is_replay and result.template is None

    def test_coverage_must_exceed_half_the_arm(self, y_maze):
        env, arms = y_maze
        half = _detect(y_maze, np.eye(env.n_bins)[arms["left"][:5]].repeat(2, axis=0))
        assert half.values["left: coverage > 0.5"] == pytest.approx(0.5)
        assert not half.is_replay

    @pytest.mark.parametrize(("n_rows", "expected"), [(2, False), (3, True)])
    def test_a_subregion_needs_five_time_bins(self, y_maze, n_rows, expected):
        """Smoothing spreads a short sweep by one time bin at each end."""
        env, arms = y_maze
        bins = arms["left"][np.linspace(0, 9, n_rows).round().astype(int)]
        posterior = np.full((n_rows + 6, env.n_bins), 1.0 / env.n_bins)
        posterior[3 : 3 + n_rows] = np.eye(env.n_bins)[bins]
        result = _detect(y_maze, posterior)
        assert result.criteria["left: time bins >= 5"] is expected

    def test_inputs_are_checked(self, y_maze):
        env, arms = y_maze
        posterior = np.eye(env.n_bins)[arms["left"]]
        times = np.arange(10) * DT
        with pytest.raises(ValueError, match="each bin belongs to one arm"):
            detect_arm_replay(
                env,
                posterior,
                times,
                arms={"a": [0, 1, 2], "b": [2, 3]},
                arm_replay_method=WU_FOSTER_2014,
            )
        with pytest.raises(ValueError, match="evenly spaced"):
            detect_arm_replay(
                env,
                posterior,
                times**2,
                arms=arms,
                arm_replay_method=WU_FOSTER_2014,
            )
        with pytest.raises(ValueError, match="one column per bin"):
            detect_arm_replay(
                env,
                posterior[:, :20],
                times,
                arms=arms,
                arm_replay_method=WU_FOSTER_2014,
            )

    def test_compare_classifies_each_event(self, y_maze):
        env, arms = y_maze
        events = [
            DecodingResult(
                posterior=np.eye(env.n_bins)[bins],
                env=env,
                times=np.arange(len(bins)) * DT,
            )
            for bins in (np.r_[arms["center"][::-1], arms["right"]], arms["left"])
        ]
        table = compare_replay_methods(
            env, events, replay_methods=[WU_FOSTER_2014], arms=arms
        )
        assert table["template"].tolist() == ["center+right", "left"]
        with pytest.raises(ValueError, match="need arms="):
            compare_replay_methods(env, events, replay_methods=[WU_FOSTER_2014])

    def test_recipe_matches_the_paper(self):
        method = WU_FOSTER_2014
        assert (method.peak_threshold, method.smoothing_sigma) == (5.0, 0.01)
        assert method.min_time_bins == 5
        assert (method.min_coverage, method.min_abs_correlation) == (0.5, 0.5)
        assert method.joint_min_abs_correlation == 0.5

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"min_coverage": 1.0}, "min_coverage"),
            ({"min_time_bins": 1}, "min_time_bins"),
            ({"smoothing_sigma": -0.01}, "smoothing_sigma"),
        ],
    )
    def test_invalid_methods_are_refused(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            ArmReplayMethod(name="x", reference="y", **kwargs)
