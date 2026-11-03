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
    with_positions,
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


def samples_along(graph: nx.Graph, n: int, noise: float, seed: int = 0) -> np.ndarray:
    """Points spread along every edge in proportion to its length, plus noise (seeded)."""
    rng = np.random.default_rng(seed)
    _, nodes, edges = graph_arrays(graph)
    lengths = np.linalg.norm(nodes[edges[:, 1]] - nodes[edges[:, 0]], axis=1)
    which = rng.choice(len(edges), n, p=lengths / lengths.sum())
    t = rng.uniform(0, 1, n)[:, None]
    a, b = nodes[edges[which, 0]], nodes[edges[which, 1]]
    return a + t * (b - a) + rng.normal(0, noise, (n, 2))


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


def _declare(graph, points, **kwargs):
    options = {
        "kind": "track",
        "source": "s",
        "bin_size": 2.5,
        "track_width": 7.0,
        "units": "cm",
        "name": "t",
        "placement": "p",
        "placed_by": "fitted to the samples",
    }
    return declared_geometry(graph, points, **(options | kwargs))


def test_declared_geometry_has_three_kinds_of_width():
    graph, _ = straight_track(np.array([[0.0, 0.0], [100.0, 0.0]]), 100.0)
    points = np.column_stack([RNG.uniform(0, 100, 5000), RNG.normal(0, 2.0, 5000)])
    stated, line = _declare(graph, points, bin_size=2.0, track_width=6.0)
    assert stated["declared_track_width"] == 6.0
    assert stated["declared_track_width_kind"] == "stated"
    assert "(stated)" in line
    assert stated["environment"]().n_bins == 50
    chosen, line = _declare(
        graph, points, track_width=8.0, width_kind="chosen", width_note="why"
    )
    assert chosen["declared_track_width_kind"] == "chosen"
    assert "(chosen: why)" in line
    measured, line = _declare(graph, points, bin_size=2.0, track_width=None)
    # Twice the 99th percentile of |N(0, 2)|: 2 * 2.576 * 2.
    assert measured["declared_track_width"] == pytest.approx(10.3, abs=0.5)
    assert measured["declared_track_width_kind"] == "measured"
    assert "placement tolerance" in line
    assert "by construction" in line
    # The width-independent fit: |N(0, 2)| has a median of 0.674 x 2.
    fit = measured["environment_fit"]
    assert fit["median_abs_lateral"] == pytest.approx(1.35, abs=0.1)
    assert fit["max_edge_bias"] < 0.2


@pytest.fixture(scope="module")
def scattered_w():
    """A W-track sampled with 3 cm of lateral scatter: about a quarter of the
    samples lie beyond the 7 cm width, yet the track is where they are."""
    truth = w_track(0.0, (0.0, 0.0))
    points = samples_along(truth, 8000, noise=3.0)
    return truth, points


def test_lateral_scatter_is_not_refused(scattered_w):
    truth, points = scattered_w
    entries, line = _declare(truth, points)
    assert entries, line
    assert entries["environment_fit"]["share_within_half_width"] < 0.8


@pytest.mark.parametrize(
    ("change", "why"),
    [
        (
            lambda p: 0.88 * (p - np.array([40.0, 40.0])) + np.array([40.0, 40.0]),
            "side",
        ),
        (
            # About the corner node base_0: the far arms move sideways.
            lambda p: rotate(p[None], 6.0, (0.0, 0.0))[0],
            "side",
        ),
    ],
    ids=["scaled_by_0.88", "rotated_6_degrees"],
)
def test_a_subtly_misplaced_layout_is_refused(scattered_w, change, why):
    truth, points = scattered_w
    moved = with_positions(
        truth, {n: change(np.asarray(p, float)) for n, p in truth.nodes(data="pos")}
    )
    entries, line = _declare(moved, points)
    assert entries == {}
    assert line.startswith("environment: inferred")
    assert why in line


def test_a_wrong_stated_length_is_refused_but_the_authors_geometry_is_not(scattered_w):
    truth, points = scattered_w
    entries, line = _declare(truth, points, stated_lengths={"arms": (80.0, 95.0)})
    assert entries == {}
    assert "arms -16% of the stated length" in line
    entries, _ = _declare(
        truth, points, stated_lengths={"arms": (80.0, 95.0)}, placed_by="authors' graph"
    )
    assert entries["declared_from"] == "authors' graph"
    assert entries["environment_fit"]["size_residual_arms"] == pytest.approx(
        -0.158, abs=0.001
    )


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
        placed_by="fitted to the samples",
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


@pytest.fixture(scope="module")
def ring_samples():
    """A 50 cm radius ring sampled with 2 cm scatter; over a 20 degree reward
    area (17 cm) the head leans 4 cm outward."""
    rng = np.random.default_rng(9)
    angle = rng.uniform(0, 2 * np.pi, 20000)
    radius = 50.0 + rng.normal(0, 2.0, angle.size)
    radius[(angle > 0) & (angle < np.radians(20))] += 4.0
    return np.column_stack([radius * np.cos(angle), radius * np.sin(angle)])


def test_a_loop_is_judged_over_stretches_not_polygon_sides(ring_samples):
    loop = polyline_graph(circle_points(np.zeros(2), 50.0, 72), closed=True)
    entries, line = _declare(loop, ring_samples, track_width=None)
    # The reward area's lean is local: a 4.4 cm side shows it, a 50 cm stretch
    # does not, and the loop is kept.
    assert entries, line
    assert entries["environment_fit"]["max_edge_bias"] < 2.5
    # The same circle moved 4 cm sideways is refused.
    moved = polyline_graph(circle_points(np.array([4.0, 0.0]), 50.0, 72), closed=True)
    entries, line = _declare(moved, ring_samples, track_width=None)
    assert entries == {}
    assert "side" in line
