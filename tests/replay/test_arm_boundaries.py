"""Arm replay at the published boundaries: run length and merge gap."""

from __future__ import annotations

import dataclasses

import networkx as nx
import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.replay import detect_arm_replay
from neurospatial.replay.methods import BERNERS_LEE_2021, WU_FOSTER_2014


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
    arms = {
        "center": np.arange(10)[::-1],
        "left": np.arange(10, 20),
        "right": np.arange(20, 30),
    }
    return env, arms


def _detect(maze, posterior, method, dt):
    env, arms = maze
    return detect_arm_replay(
        env,
        posterior,
        np.arange(len(posterior)) * dt,
        arms=arms,
        arm_replay_method=method,
    )


@pytest.mark.parametrize(("n_rows", "long_enough"), [(4, False), (5, True)])
def test_a_run_of_exactly_min_duration_is_long_enough(y_maze, n_rows, long_enough):
    """Wu & Foster (2014): at least 50 ms, five 10 ms time bins, unsmoothed here.

    Each time bin holds half its mass in each of two neighboring bins, so
    the sweep covers more than half of the arm.
    """
    env, arms = y_maze
    method = dataclasses.replace(WU_FOSTER_2014, name="unsmoothed", smoothing_sigma=0.0)
    posterior = np.full((n_rows + 4, env.n_bins), 1.0 / env.n_bins)
    for row in range(n_rows):
        posterior[2 + row] = 0.0
        posterior[2 + row, arms["left"][[2 * row, 2 * row + 1]]] = 0.5
    result = _detect(y_maze, posterior, method, dt=0.01)
    assert result.values["left: duration >= 0.05"] == pytest.approx(0.01 * n_rows)
    assert result.criteria["left: duration >= 0.05"] is long_enough
    assert (result.template == "left") is long_enough


@pytest.mark.parametrize(
    ("empty_rows", "merged"), [(9, True), (10, False), (11, False)]
)
def test_runs_merge_across_a_gap_of_at_most_merge_max_gap(y_maze, empty_rows, merged):
    """Berners-Lee et al. (2021): runs merge when the time between the last
    window of one and the first of the next is at most 50 ms (10 steps of
    5 ms): 9 windows between them make 50 ms, 10 make 55 ms.
    """
    env, arms = y_maze
    method = dataclasses.replace(
        BERNERS_LEE_2021, name="unsmoothed", smoothing_sigma=0.0
    )
    rows = 12 + empty_rows + 12
    posterior = np.full((rows, env.n_bins), 1.0 / env.n_bins)
    first = arms["left"][np.linspace(0, 5, 12).round().astype(int)]
    second = arms["left"][np.linspace(6, 9, 12).round().astype(int)]
    posterior[:12] = np.eye(env.n_bins)[first]
    posterior[12 + empty_rows :] = np.eye(env.n_bins)[second]
    result = _detect(y_maze, posterior, method, dt=0.005)
    coverage = result.values["left: coverage > 0.5"]
    if merged:
        # The mean position ranges over 9 of the arm's 10 bins.
        assert coverage == pytest.approx(0.9)
        assert result.values["left: duration >= 0.055"] == pytest.approx(rows * 0.005)
    else:
        assert coverage <= 0.5
