from __future__ import annotations

from dataclasses import replace

import networkx as nx
import numpy as np
import pytest
from _tracks import (
    best_fit,
    circle_points,
    cut_region,
    declared_geometry,
    figure8_candidates,
    fit_circle_centre,
    graph_arrays,
    graph_from_points,
    graph_residual,
    loop_centreline,
    parallel_arm_candidates,
    polyline_graph,
    segment_distance,
    star_candidates,
    straight_track,
    x_maze_candidates,
    z_track_candidates,
)
from operations import _graph_env, border_scores, grid_scores, laps

RNG = np.random.default_rng(7)


def rotate(
    points: np.ndarray, degrees: float, shift: tuple[float, float]
) -> np.ndarray:
    a = np.radians(degrees)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return points @ rot.T + np.asarray(shift)


def samples_along(graph: nx.Graph, n: int, noise: float) -> np.ndarray:
    """Points spread along every edge in proportion to its length, plus noise."""
    _, nodes, edges = graph_arrays(graph)
    lengths = np.linalg.norm(nodes[edges[:, 1]] - nodes[edges[:, 0]], axis=1)
    which = RNG.choice(len(edges), n, p=lengths / lengths.sum())
    t = RNG.uniform(0, 1, n)[:, None]
    a, b = nodes[edges[which, 0]], nodes[edges[which, 1]]
    return a + t * (b - a) + RNG.normal(0, noise, (n, 2))


def node_error(fitted: nx.Graph, truth: nx.Graph) -> float:
    """Largest distance from a true node to the nearest fitted node."""
    _, got, _ = graph_arrays(fitted)
    _, want, _ = graph_arrays(truth)
    return float(np.max(np.min(np.linalg.norm(want[:, None] - got[None], axis=2), 1)))


def w_track(degrees: float = 23.0, shift=(40.0, -10.0)) -> nx.Graph:
    local = {
        "base_0": (0, 0),
        "base_1": (40, 0),
        "base_2": (80, 0),
        "arm_0": (0, 80),
        "arm_1": (40, 80),
        "arm_2": (80, 80),
    }
    points = {
        k: rotate(np.array([v], float), degrees, shift)[0] for k, v in local.items()
    }
    edges = [("base_0", "base_1"), ("base_1", "base_2")]
    edges += [(f"base_{i}", f"arm_{i}") for i in range(3)]
    return graph_from_points(points, edges)


def test_segment_distance_is_the_distance_to_the_nearest_segment():
    nodes = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
    edges = np.array([[0, 1], [1, 2]])
    points = np.array([[5.0, 3.0], [12.0, 5.0], [-4.0, 3.0], [10.0, 0.0]])
    np.testing.assert_allclose(segment_distance(points, nodes, edges), [3, 2, 5, 0])


def test_straight_track_keeps_the_stated_length_on_the_samples_axis():
    along = RNG.uniform(-55, 55, 20000)
    local = np.column_stack([along, RNG.normal(0, 1.5, along.size)])
    points = rotate(local, 30.0, (5.0, 7.0))
    graph, fit = straight_track(points, 100.0)
    _, nodes, _ = graph_arrays(graph)
    assert np.linalg.norm(nodes[1] - nodes[0]) == pytest.approx(100.0)
    assert np.degrees(fit["axis_angle_rad"]) % 180 == pytest.approx(30.0, abs=0.5)
    assert nodes.mean(axis=0) == pytest.approx([5.0, 7.0], abs=1.0)
    # The samples run 5 units past each stated end: about 9 % of them.
    assert fit["share_beyond_ends"] == pytest.approx(10 / 110, abs=0.01)


@pytest.mark.parametrize("degrees", [0.0, 23.0, 90.0, 200.0])
def test_w_track_fit_recovers_the_nodes(degrees):
    truth = w_track(degrees)
    points = samples_along(truth, 6000, noise=2.0)
    fitted, _ = best_fit(parallel_arm_candidates(points), points, cap=10.0)
    assert node_error(fitted, truth) < 3.0
    assert graph_residual(fitted, points, None)["median_offset"] < 2.0


def test_figure8_fit_recovers_both_loops():
    corners = {
        "corner_0": (0, 0),
        "corner_1": (70, 0),
        "corner_2": (70, 70),
        "corner_3": (0, 70),
        "bar_0": (0, 35),
        "bar_1": (70, 35),
    }
    edges = [
        ("corner_0", "corner_1"),
        ("corner_1", "bar_1"),
        ("bar_1", "corner_2"),
        ("corner_2", "corner_3"),
        ("corner_3", "bar_0"),
        ("bar_0", "corner_0"),
        ("bar_0", "bar_1"),
    ]
    truth = graph_from_points(corners, edges)
    points = samples_along(truth, 6000, noise=1.5)
    fitted, _ = best_fit(figure8_candidates(points), points, cap=10.0)
    assert node_error(fitted, truth) < 3.0
    assert len(nx.cycle_basis(fitted)) == 2


def test_star_x_and_z_fits_recover_their_layouts():
    star = graph_from_points(
        {"centre": (0, 0)}
        | {
            f"arm_{k}": (90 * np.cos(a), 90 * np.sin(a))
            for k, a in enumerate(np.arange(8) * np.pi / 4)
        },
        [("centre", f"arm_{k}") for k in range(8)],
    )
    points = samples_along(star, 8000, noise=1.5)
    fitted, _ = best_fit(star_candidates(points, 8, np.zeros(2)), points, cap=10.0)
    assert node_error(fitted, star) < 3.0
    x_maze = graph_from_points(
        {
            "junction_0": (-10, 0),
            "junction_1": (10, 0),
            "arm_0": (-45, -35),
            "arm_1": (-45, 35),
            "arm_2": (45, -35),
            "arm_3": (45, 35),
        },
        [
            ("junction_0", "junction_1"),
            ("junction_0", "arm_0"),
            ("junction_0", "arm_1"),
            ("junction_1", "arm_2"),
            ("junction_1", "arm_3"),
        ],
    )
    points = samples_along(x_maze, 6000, noise=1.5)
    fitted, _ = best_fit(x_maze_candidates(points), points, cap=10.0)
    assert node_error(fitted, x_maze) < 3.0
    z_track = polyline_graph(
        np.array([[150.0, 120.0], [0.0, 120.0], [150.0, 0.0], [0.0, 0.0]]), closed=False
    )
    points = samples_along(z_track, 6000, noise=1.5)
    fitted, _ = best_fit(z_track_candidates(points), points, cap=10.0)
    assert node_error(fitted, z_track) < 3.0


def test_loop_centreline_and_circle_fit_recover_a_ring():
    angle = RNG.uniform(0, 2 * np.pi, 20000)
    radius = 50.0 + RNG.normal(0, 3.0, angle.size)
    points = np.column_stack([radius * np.cos(angle) + 40, radius * np.sin(angle) - 5])
    centreline = loop_centreline(points, n_points=72)
    assert np.linalg.norm(centreline - [40, -5], axis=1) == pytest.approx(50.0, abs=1.5)
    centre, fitted = fit_circle_centre(points, scale=3.0)
    assert centre == pytest.approx([40.0, -5.0], abs=0.5)
    assert fitted == pytest.approx(50.0, abs=0.5)


def test_declared_geometry_states_measures_or_refuses_the_width():
    graph, _ = straight_track(np.array([[0.0, 0.0], [100.0, 0.0]]), 100.0)
    points = np.column_stack([RNG.uniform(0, 100, 5000), RNG.normal(0, 2.0, 5000)])
    stated, line = declared_geometry(
        graph,
        points,
        kind="track",
        source="s",
        bin_size=2.0,
        track_width=6.0,
        units="cm",
        name="t",
        placement="p",
    )
    assert stated["declared_track_width"] == 6.0
    assert "(stated)" in line
    assert stated["environment"]().n_bins == 50
    measured, line = declared_geometry(
        graph,
        points,
        kind="track",
        source="s",
        bin_size=2.0,
        track_width=None,
        units="cm",
        name="t",
        placement="p",
    )
    # Twice the 99th percentile of |N(0, 2)|: 2 * 2.576 * 2.
    assert measured["declared_track_width"] == pytest.approx(10.3, abs=0.5)
    assert "measured" in line
    off = points + np.array([0.0, 8.0])
    refused, line = declared_geometry(
        graph,
        off,
        kind="track",
        source="s",
        bin_size=2.0,
        track_width=6.0,
        units="cm",
        name="t",
        placement="p",
    )
    assert refused == {}
    assert line.startswith("environment: inferred")


@pytest.fixture
def loop_recording(simulated_recording):
    """Ten counter-clockwise laps of a 50 cm radius circle, declared as a loop."""
    times = np.arange(0, 200.0, 0.05)
    angle = 2 * np.pi * times / 20.0 + 0.1
    positions = np.column_stack([50 * np.cos(angle), 50 * np.sin(angle)])
    positions += np.random.default_rng(3).normal(0, 1.0, positions.shape)
    graph = polyline_graph(circle_points(np.zeros(2), 50.0, 72), closed=True)
    region = cut_region(
        "start", (50.0, 0.0), (0.0, 1.0), half_length=3.0, half_width=15.0
    )
    entries, _ = declared_geometry(
        graph,
        positions,
        kind="loop",
        source="test",
        bin_size=2.5,
        track_width=10.0,
        units="cm",
        name="loop",
        placement="p",
    )
    return replace(
        simulated_recording,
        times=times,
        positions=positions,
        spike_times=None,
        unit_ids=None,
        geometry={**entries, "lap_start_region": region},
        params={"bin_size": 2.5, "analysis_epochs": np.array([[0.0, 200.0]])},
    )


def test_declared_loop_keeps_every_edge_and_counts_laps_by_region(loop_recording):
    cache: dict = {}
    env = _graph_env(loop_recording, cache)
    assert env is not None
    assert env.is_linearized_track
    # 72 edges of 4.36 cm, two 2.18 cm bins each: the closing edge is kept.
    assert env.n_bins == 144
    outcome = laps(loop_recording, cache)
    # The run starts just past the region: its ten entries (19.6 s to 199.6 s)
    # bound nine complete laps.
    assert outcome.summary["n_laps_region"] == 9
    assert (
        outcome.summary["n_laps_region_counter-clockwise"]
        == outcome.summary["n_laps_region"]
    )


def test_area_scores_are_na_on_a_declared_track(loop_recording):
    assert grid_scores(loop_recording, {}).status == "n/a"
    assert border_scores(loop_recording, {}).status == "n/a"
