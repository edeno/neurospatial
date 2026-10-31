"""Route comparison along the environment (geodesic Hausdorff and DTW)."""

from __future__ import annotations

import itertools

import networkx as nx
import numpy as np
import pytest
from shapely.geometry import box
from shapely.ops import unary_union

from neurospatial import Environment
from neurospatial.behavior import trajectory_similarity


@pytest.fixture(scope="module")
def u_maze() -> Environment:
    """Two 10 cm arms 30 cm apart, joined at the bottom."""
    shape = unary_union([box(0, 0, 10, 60), box(40, 0, 50, 60), box(0, 0, 50, 10)])
    return Environment.from_polygon(shape, bin_size=2.0, units="cm")


def _arm_run(env: Environment, x: float) -> np.ndarray:
    """Bins of a run up the arm at ``x``, bottom to top."""
    y = np.linspace(12.0, 58.0, 24)
    return env.bin_at(np.column_stack([np.full_like(y, x), y]))


def _geodesic(env: Environment, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Shortest paths between bins, by networkx: an independent oracle."""
    lengths = dict(
        nx.all_pairs_dijkstra_path_length(env.connectivity, weight="distance")
    )
    return np.array([[lengths[i].get(j, np.inf) for j in b] for i in a])


def _diameter(env: Environment) -> float:
    lengths = nx.all_pairs_dijkstra_path_length(env.connectivity, weight="distance")
    return max(max(row.values()) for _, row in lengths)


def _dtw(cost: np.ndarray) -> float:
    n1, n2 = cost.shape
    total = np.full((n1 + 1, n2 + 1), np.inf)
    total[0, 0] = 0.0
    for i, j in itertools.product(range(1, n1 + 1), range(1, n2 + 1)):
        total[i, j] = cost[i - 1, j - 1] + min(
            total[i - 1, j], total[i, j - 1], total[i - 1, j - 1]
        )
    return float(total[n1, n2])


def test_runs_up_the_two_arms_are_far_apart_along_the_maze(u_maze):
    """30 cm apart in a straight line, about 100 cm apart along the maze."""
    left, right = _arm_run(u_maze, 5.0), _arm_run(u_maze, 45.0)
    for method in ("hausdorff", "dtw"):
        straight = trajectory_similarity(
            left, right, u_maze, method=method, metric="euclidean"
        )
        along = trajectory_similarity(left, right, u_maze, method=method)
        assert along < straight


def test_geodesic_hausdorff_and_dtw_match_networkx(u_maze):
    left, right = _arm_run(u_maze, 5.0), _arm_run(u_maze, 45.0)[::2]
    cost = _geodesic(u_maze, left, right)
    diameter = _diameter(u_maze)
    hausdorff = max(cost.min(axis=1).max(), cost.min(axis=0).max())
    assert trajectory_similarity(
        left, right, u_maze, method="hausdorff"
    ) == pytest.approx(1 - hausdorff / diameter, rel=1e-9)
    dtw = _dtw(cost) / (max(len(left), len(right)) * diameter)
    assert trajectory_similarity(left, right, u_maze, method="dtw") == pytest.approx(
        1 - dtw, rel=1e-9
    )


def test_euclidean_metric_keeps_the_straight_line_score(u_maze):
    left, right = _arm_run(u_maze, 5.0), _arm_run(u_maze, 45.0)
    centers = u_maze.bin_centers
    cost = np.linalg.norm(centers[left][:, None] - centers[right][None], axis=2)
    extent = np.linalg.norm(np.ptp(centers, axis=0))
    hausdorff = max(cost.min(axis=1).max(), cost.min(axis=0).max())
    assert trajectory_similarity(
        left, right, u_maze, method="hausdorff", metric="euclidean"
    ) == pytest.approx(1 - hausdorff / extent, rel=1e-12)


def test_routes_in_parts_that_do_not_connect_score_zero():
    rng = np.random.default_rng(0)
    samples = np.vstack(
        [
            rng.uniform([0, 0], [10, 10], (500, 2)),
            rng.uniform([30, 0], [40, 10], (500, 2)),
        ]
    )
    env = Environment.from_samples(samples, bin_size=2.0, units="cm")
    assert not nx.is_connected(env.connectivity)
    left = env.bin_at(np.array([[3.0, 5.0], [5.0, 5.0]]))
    right = env.bin_at(np.array([[33.0, 5.0], [35.0, 5.0]]))
    for method in ("hausdorff", "dtw"):
        assert trajectory_similarity(left, right, env, method=method) == 0.0


@pytest.mark.parametrize("method", ["jaccard", "correlation"])
def test_metric_is_refused_where_it_has_no_effect(u_maze, method):
    run = _arm_run(u_maze, 5.0)
    with pytest.raises(ValueError, match="no effect"):
        trajectory_similarity(run, run, u_maze, method=method, metric="geodesic")
