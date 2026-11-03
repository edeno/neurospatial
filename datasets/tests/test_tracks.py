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
    if "track_width" in kwargs:
        kwargs["physical_width"] = kwargs.pop("track_width")
    options = {
        "times": np.arange(len(points)) / 30.0,
        "tracked_point": "test",
        "kind": "track",
        "source": "s",
        "bin_size": 2.5,
        "physical_width": 7.0,
        "units": "cm",
        "name": "t",
        "placement": "p",
        "placed_by": "fitted to the samples",
    }
    return declared_geometry(graph, points, **(options | kwargs))


def test_physical_width_and_placement_tolerance_are_kept_apart():
    graph, _ = straight_track(np.array([[0.0, 0.0], [100.0, 0.0]]), 100.0)
    rng = np.random.default_rng(11)
    points = np.column_stack([rng.uniform(0, 100, 5000), rng.normal(0, 2.0, 5000)])
    # Twice the 99th percentile of |N(0, 2)|: 2 * 2.576 * 2 = 10.3 cm.
    stated, line = _declare(graph, points, bin_size=2.0, track_width=6.0)
    assert stated["declared_physical_width"] == 6.0
    assert stated["declared_physical_width_kind"] == "stated"
    assert stated["declared_track_width"] == pytest.approx(10.3, abs=0.5)
    assert stated["declared_track_width_kind"] == "placement tolerance"
    assert "physical width 6 cm (stated)" in line
    # Beyond 3 cm of the axis: 2 x P(N(0, 2) > 3) = 13.4 %, a tracking measure.
    fit = stated["environment_fit"]
    assert fit["outside_physical_width"] == pytest.approx(0.134, abs=0.02)
    assert fit["share_outside_tolerance"] == pytest.approx(0.01, abs=0.005)
    assert stated["environment"]().n_bins == 50
    assert stated["tracked_point"] == "test"
    chosen, line = _declare(
        graph, points, track_width=12.0, width_kind="chosen", width_note="why"
    )
    assert chosen["declared_physical_width_kind"] == "chosen"
    assert "(chosen: why)" in line
    # The tolerance is never narrower than the physical width.
    assert chosen["declared_track_width"] == 12.0
    unstated, line = _declare(graph, points, bin_size=2.0, track_width=None)
    assert unstated["declared_physical_width"] is None
    assert "physical width not stated" in line
    assert "by construction" in line
    # The width-independent fit: |N(0, 2)| has a median of 0.674 x 2.
    fit = unstated["environment_fit"]
    assert fit["median_abs_lateral"] == pytest.approx(1.35, abs=0.1)
    assert fit["max_edge_bias"] < 0.2


def test_outside_physical_width_is_split_by_speed():
    # Stopped samples lean 5 cm off a 6 cm track, running ones (10 cm/s) stay on it.
    from _geometry import share_by_speed

    times = np.arange(2000) / 10.0
    x = np.concatenate([np.full(1000, 50.0), np.linspace(0, 1000, 1000)])
    y = np.concatenate([np.full(1000, 5.0), np.zeros(1000)])
    positions = np.column_stack([x, y])
    split = share_by_speed(times, positions, np.abs(y) > 3.0, (4.0, 30.0))
    assert split["all"] == pytest.approx(0.5, abs=0.01)
    assert split["0-4"] == pytest.approx(1.0, abs=0.01)
    assert split["4-30"] == pytest.approx(0.0, abs=0.01)


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
    assert "environment" in entries, line
    assert entries["environment_fit"]["outside_physical_width"] > 0.2


def _scaled(graph, factor, degrees=0.0):
    """The graph scaled and rotated about its own centre."""
    _, nodes, _ = graph_arrays(graph)
    centre = nodes.mean(axis=0)
    moved = {
        name: rotate(
            (np.asarray(pos, float) - centre)[None] * factor, degrees, tuple(centre)
        )[0]
        for name, pos in graph.nodes(data="pos")
    }
    return with_positions(graph, moved)


@pytest.mark.parametrize("factor", [0.88, 0.80])
def test_a_scale_error_is_absorbed_by_the_fit_and_caught_only_by_the_stated_size(
    factor,
):
    # The file's scale is off: a stated 80 cm W-track's samples come out
    # `factor` x as large. The pipeline fits the layout to the samples, so
    # the fit follows them (no lateral bias); only the stated size can tell.
    truth = w_track(0.0, (0.0, 0.0))
    points = samples_along(_scaled(truth, factor), 8000, noise=3.0)
    fitted, _ = best_fit(parallel_arm_candidates(points), points, cap=10.0)
    from _tracks import edge_fit, parallel_arm_lengths

    assert edge_fit(fitted, points, tolerance=2.5)["max_edge_bias"] < 1.0
    base, arms = parallel_arm_lengths(fitted)
    entries, line = _declare(
        fitted, points, stated_lengths={"base": (base, 80.0), "arms": (arms, 80.0)}
    )
    if factor == 0.80:  # 20 % off: beyond the 15 % size tolerance
        assert "environment" not in entries
        assert "of the stated length" in line
    else:  # 12 % off: within it, so no check here catches this scale error
        assert "environment" in entries


def test_a_rotated_layout_is_absorbed_by_the_fit():
    # Samples of a W rotated 6 degrees about its centre: the fit starts from
    # the samples and rotates with them, so nothing is misfit (the
    # orientation is not stated).
    truth = w_track(0.0, (0.0, 0.0))
    points = samples_along(_scaled(truth, 1.0, 6.0), 8000, noise=3.0)
    fitted, _ = best_fit(parallel_arm_candidates(points), points, cap=10.0)
    entries, line = _declare(fitted, points)
    assert "environment" in entries, line
    assert entries["environment_fit"]["max_edge_bias"] < 1.0


@pytest.mark.parametrize("length", [85.0, 70.0])
def test_a_track_shorter_than_its_samples_is_misfit(length):
    # Samples along 100 cm; the declared track is shorter. The lateral and
    # support measures see nothing; the reach past the free ends does.
    from _geometry import graph_environment, judge_declared

    rng = np.random.default_rng(4)
    on = np.column_stack([rng.uniform(0, 100, 5000), rng.normal(0, 3.0, 5000)])
    track = polyline_graph(
        np.array([[50 - length / 2, 0.0], [50 + length / 2, 0.0]]), closed=False
    )
    env = graph_environment(track, bin_size=2.5, track_width=60.0, units="cm", name="t")
    judged = judge_declared(env, np.arange(len(on)) / 30.0, on, bin_size=2.5)
    assert judged["fit"]["max_edge_bias"] < 1.0
    assert judged["fit"]["max_unsupported"] == 0.0
    assert any("past its free ends" in reason for reason in judged["misfit"])


def test_a_wrong_stated_length_is_refused_but_the_authors_geometry_is_not(scattered_w):
    truth, points = scattered_w
    entries, line = _declare(truth, points, stated_lengths={"arms": (80.0, 95.0)})
    assert "environment" not in entries
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
        times=times,
        tracked_point="simulated",
        bin_size=2.5,
        physical_width=10.0,
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
    assert "environment" in entries, line
    assert entries["environment_fit"]["max_edge_bias"] < 2.5
    # The same circle moved 4 cm sideways is refused.
    moved = polyline_graph(circle_points(np.array([4.0, 0.0]), 50.0, 72), closed=True)
    entries, line = _declare(moved, ring_samples, track_width=None)
    assert "environment" not in entries
    assert "side" in line


def _judged_line(segments, samples, bin_size=2.5):
    from _geometry import graph_environment, judge_declared

    graph = polyline_graph(np.asarray(segments, dtype=float), closed=False)
    env = graph_environment(
        graph, bin_size=bin_size, track_width=60.0, units="cm", name="t"
    )
    return judge_declared(env, np.arange(len(samples)) / 30.0, samples)


def test_a_short_reward_arm_leaning_off_a_junction_is_reported_not_judged():
    # A 100 cm track with a 15 cm reward arm off its middle (a T); the head
    # leans 3 cm to one side on the arm, more than a 2.5 cm bin. The arm is
    # shorter than a judged stretch (20 bins, 50 cm), so its lean is reported.
    from _geometry import graph_environment, judge_declared

    graph = graph_from_points(
        {"a": (0.0, 0.0), "j": (50.0, 0.0), "b": (100.0, 0.0), "r": (50.0, 15.0)},
        [("a", "j"), ("j", "b"), ("j", "r")],
    )
    env = graph_environment(graph, bin_size=2.5, track_width=60.0, units="cm", name="t")
    rng = np.random.default_rng(11)
    main = np.column_stack([rng.uniform(0, 100, 8000), rng.normal(0, 1.0, 8000)])
    arm = np.column_stack(
        [50.0 + 3.0 + rng.normal(0, 1.0, 1200), rng.uniform(1, 15, 1200)]
    )
    samples = np.vstack([main, arm])
    judged = judge_declared(env, np.arange(len(samples)) / 30.0, samples, bin_size=2.5)
    assert judged["fit"]["max_short_stretch_bias"] == pytest.approx(3.0, abs=0.3)
    assert judged["fit"]["max_edge_bias"] < 1.0
    assert judged["fit"]["judged_stretches"] == 2  # the two 50 cm halves
    assert judged["misfit"] == []


def test_support_does_not_depend_on_how_the_track_is_cut_into_segments():
    # Samples along 8-92 cm of a 0-100 cm track: the head stops 8 cm short of
    # each free end. Drawn as one segment or with 15 cm end segments, the
    # verdict is the same; the ends' unsupported part is reported.
    rng = np.random.default_rng(12)
    samples = np.column_stack([rng.uniform(8, 92, 6000), rng.normal(0, 1.0, 6000)])
    one = _judged_line([[0, 0], [100, 0]], samples)
    cut = _judged_line([[0, 0], [15, 0], [85, 0], [100, 0]], samples)
    assert one["misfit"] == cut["misfit"] == []
    assert one["fit"]["max_unsupported"] == cut["fit"]["max_unsupported"] == 0.0
    for judged in (one, cut):
        assert judged["fit"]["max_end_unsupported"] > 0.5


def test_a_track_with_no_samples_near_it_is_misfit():
    # Every sample 40 cm off the declared line, beyond the 3-bin window of
    # every stretch: a NaN bias is no evidence of fit.
    rng = np.random.default_rng(13)
    samples = np.column_stack(
        [rng.uniform(0, 100, 3000), 40.0 + rng.normal(0, 1, 3000)]
    )
    judged = _judged_line([[0, 0], [100, 0]], samples)
    assert np.isnan(judged["fit"]["max_edge_bias"])
    assert any("no samples near the declared geometry" in r for r in judged["misfit"])


# The whole-track frame offset (``_tracks.frame_offset``): one translation
# fitted to every piece's median perpendicular offset, judged at one bin.
FRAME_BIN = 2.5


def _old_reasons(judged):
    """The misfit reasons the per-stretch, end and size checks give alone."""
    return [r for r in judged["misfit"] if "as a whole" not in r]


@pytest.fixture(scope="module")
def grid_maze():
    """A 4 x 4 grid of 15 cm corridors (6 bins, all shorter than a judged
    stretch), both orientations, no free ends; 2 cm scatter on both sides of
    each corridor and unequal sampling (corridor k gets 100 + 20 k samples)."""
    names = {(i, j): f"{i}{j}" for i in range(4) for j in range(4)}
    graph = graph_from_points(
        {name: (15.0 * i, 15.0 * j) for (i, j), name in names.items()},
        [(names[i, j], names[i + 1, j]) for i in range(3) for j in range(4)]
        + [(names[i, j], names[i, j + 1]) for i in range(4) for j in range(3)],
    )
    _, nodes, edges = graph_arrays(graph)
    rng = np.random.default_rng(21)
    parts = []
    for k, (u, v) in enumerate(edges):
        n = 100 + 20 * k
        a, b = nodes[u], nodes[v]
        normal = np.array([-(b - a)[1], (b - a)[0]]) / 15.0
        parts.append(
            a
            + rng.uniform(0, 1, n)[:, None] * (b - a)
            + rng.normal(0, 2.0, n)[:, None] * normal
        )
    return graph, np.vstack(parts)


def _judged(graph, samples):
    from _geometry import graph_environment, judge_declared

    env = graph_environment(
        graph, bin_size=FRAME_BIN, track_width=20.0, units="cm", name="t"
    )
    return judge_declared(
        env, np.arange(len(samples)) / 30.0, samples, bin_size=FRAME_BIN
    )


def test_a_maze_of_short_corridors_in_place_has_no_frame_offset(grid_maze):
    graph, samples = grid_maze
    judged = _judged(graph, samples)
    fit = judged["fit"]
    assert fit["judged_stretches"] == 0  # no stretch reaches 20 bins
    assert fit["frame_pieces"] == 24
    # 12 corridors along x and 12 along y: sum P_k = diag(12, 12).
    assert fit["frame_min_eigenvalue"] == pytest.approx(12.0)
    assert fit["frame_max_eigenvalue"] == pytest.approx(12.0)
    assert np.linalg.norm(fit["frame_offset"]) < 0.2 * FRAME_BIN
    assert fit["frame_unconstrained"] == []
    # Parallel corridors 15 cm apart: the 7.5 cm window is not wider than
    # half of it, so shifts below 7.5 cm are resolved.
    assert fit["frame_spacing"] == pytest.approx(15.0)
    assert judged["misfit"] == []
    assert judged["unverified"] == []


@pytest.mark.parametrize("axis", [0, 1])
def test_a_maze_of_short_corridors_moved_off_its_samples_is_misfit(grid_maze, axis):
    # Samples 2 bins (5 cm) off the maze along x, then y: the 12 corridors
    # across that axis each see the shift, those along it see none, so the
    # refitted s = (5, 0) or (0, 5). A single step reads less (neighbouring
    # corridors take samples moved off one) but still more than a bin, so
    # both estimators exceed the limit: a misfit.
    graph, samples = grid_maze
    shift = np.zeros(2)
    shift[axis] = 2.0 * FRAME_BIN
    judged = _judged(graph, samples + shift)
    fit = judged["fit"]
    np.testing.assert_allclose(fit["frame_offset"], shift, atol=0.2 * FRAME_BIN)
    assert fit["frame_offset_judged_one_step"] > FRAME_BIN
    assert fit["frame_offset_judged_one_step"] < fit["frame_offset_judged"]
    assert any("as a whole" in r for r in judged["misfit"])
    assert judged["unverified"] == []
    # Negative control: without the frame check nothing fails it (no judged
    # stretch, no free end, no stated size).
    assert _old_reasons(judged) == []


def test_a_frame_offset_the_estimators_disagree_on_is_not_verified(grid_maze):
    # 1.2 bins (3 cm) off along x: the refitted shift recovers it (above the
    # limit), a single step reads under a bin. The check neither passes nor
    # fails the maze; it lists why it is unverified.
    graph, samples = grid_maze
    judged = _judged(graph, samples + np.array([1.2 * FRAME_BIN, 0.0]))
    fit = judged["fit"]
    assert fit["frame_offset"][0] == pytest.approx(1.2 * FRAME_BIN, abs=0.2 * FRAME_BIN)
    assert fit["frame_offset_judged"] > FRAME_BIN > fit["frame_offset_judged_one_step"]
    assert judged["misfit"] == []
    assert len(judged["unverified"]) == 1
    assert "depends on the estimator" in judged["unverified"][0]


def test_a_shift_on_corridors_closer_than_the_window_is_not_verified():
    # A grid of 5 cm corridors at 2.5 cm bins (8 x 8 nodes, 0.5 cm scatter):
    # the 7.5 cm window is wider than half the 5 cm spacing, so a 1.5-bin
    # (3.75 cm) shift lands nearer the neighbouring corridor and reads as a
    # small shift the other way. Without the spacing rule both estimators
    # read under a bin and it matched; now it is not verified.
    from _geometry import graph_environment, judge_declared

    names = {(i, j): f"{i}_{j}" for i in range(8) for j in range(8)}
    graph = graph_from_points(
        {name: (5.0 * i, 5.0 * j) for (i, j), name in names.items()},
        [(names[i, j], names[i + 1, j]) for i in range(7) for j in range(8)]
        + [(names[i, j], names[i, j + 1]) for i in range(8) for j in range(7)],
    )
    _, nodes, edges = graph_arrays(graph)
    rng = np.random.default_rng(22)
    parts = []
    for k, (u, v) in enumerate(edges):
        n = 100 + 20 * k
        a, b = nodes[u], nodes[v]
        normal = np.array([-(b - a)[1], (b - a)[0]]) / 5.0
        parts.append(
            a
            + rng.uniform(0, 1, n)[:, None] * (b - a)
            + rng.normal(0, 0.5, n)[:, None] * normal
        )
    samples = np.vstack(parts) + np.array([1.5 * FRAME_BIN, 0.0])
    env = graph_environment(
        graph, bin_size=FRAME_BIN, track_width=6.0, units="cm", name="t"
    )
    judged = judge_declared(
        env, np.arange(len(samples)) / 30.0, samples, bin_size=FRAME_BIN
    )
    fit = judged["fit"]
    assert fit["frame_spacing"] == pytest.approx(5.0)
    # The aliased reading: under a bin by both estimators.
    assert fit["frame_offset_judged_one_step"] < FRAME_BIN
    assert fit["frame_offset_judged"] < FRAME_BIN
    assert judged["misfit"] == []
    assert any("half the corridor spacing" in r for r in judged["unverified"])


def test_an_unsettled_refit_is_not_verified(grid_maze):
    # 10 cm scatter on 15 cm corridors, moved 2 bins: the refit is still
    # moving after its 50 steps, so its reading confirms nothing.
    graph, _ = grid_maze
    _, nodes, edges = graph_arrays(graph)
    rng = np.random.default_rng(21)
    parts = []
    for k, (u, v) in enumerate(edges):
        n = 100 + 20 * k
        a, b = nodes[u], nodes[v]
        normal = np.array([-(b - a)[1], (b - a)[0]]) / 15.0
        parts.append(
            a
            + rng.uniform(0, 1, n)[:, None] * (b - a)
            + rng.normal(0, 10.0, n)[:, None] * normal
        )
    judged = _judged(graph, np.vstack(parts) + np.array([2.0 * FRAME_BIN, 0.0]))
    assert not judged["fit"]["frame_converged"]
    assert any("did not settle" in r for r in judged["unverified"])


def tilted_lattice(n_cubes: int = 3, side: float = 16.0) -> np.ndarray:
    """Bars of a cubic lattice turned so a body diagonal is vertical: every
    bar 35.26 degrees from horizontal, in three directions. Shape (n, 2, 3)."""
    # Rotation taking (1, 1, 1) / sqrt(3) to +z.
    z = np.ones(3) / np.sqrt(3)
    x = np.array([1.0, -1.0, 0.0]) / np.sqrt(2)
    rotation = np.vstack([x, np.cross(z, x), z])
    ticks = side * np.arange(n_cubes + 1)
    bars = []
    for axis in range(3):
        others = [a for a in range(3) if a != axis]
        for p in ticks:
            for q in ticks:
                start, stop = np.zeros(3), np.zeros(3)
                start[others], stop[others] = (p, q), (p, q)
                stop[axis] = ticks[-1]
                bars.append([rotation @ start, rotation @ stop])
    return np.asarray(bars)


@pytest.fixture(scope="module")
def lattice_samples():
    """Head samples along every segment with 1 cm scatter across the bar and
    a head held 1.5 bins (6 cm at 4 cm bins) above the bars."""
    from _geometry import graph_environment, lattice_graph

    graph = lattice_graph(tilted_lattice(), 3)
    _, nodes, edges = graph_arrays(graph)
    rng = np.random.default_rng(23)
    parts = []
    for u, v in edges:
        a, b = nodes[u], nodes[v]
        d = (b - a) / np.linalg.norm(b - a)
        noise = rng.normal(0, 1.0, (60, 3))
        noise -= (noise @ d)[:, None] * d
        parts.append(a + rng.uniform(0, 1, 60)[:, None] * (b - a) + noise)
    samples = np.vstack(parts) + np.array([0.0, 0.0, 6.0])
    env = graph_environment(graph, bin_size=4.0, track_width=30.0, units="cm", name="l")
    return env, samples


def _judged_lattice(env, samples):
    from _geometry import judge_declared

    return judge_declared(
        env, np.arange(len(samples)) / 50.0, samples, bin_size=4.0, lattice=True
    )


def test_a_tilted_lattice_reports_the_head_height_and_judges_the_horizontal(
    lattice_samples,
):
    env, samples = lattice_samples
    judged = _judged_lattice(env, samples)
    fit = judged["fit"]
    # The head's +6 cm is seen by every bar as its part perpendicular to it,
    # P_k (0, 0, 6), so s = (0, 0, 6): reported, not judged.
    np.testing.assert_allclose(fit["frame_offset"], [0.0, 0.0, 6.0], atol=0.8)
    assert fit["frame_offset_vertical"] == pytest.approx(6.0, abs=0.8)
    assert fit["frame_converged"]
    assert fit["frame_offset_judged"] < 0.2 * 4.0
    # Three bar directions 35.26 degrees from horizontal, equally many:
    # sum P_k = (2/3) n_pieces I.
    assert fit["frame_min_eigenvalue"] == pytest.approx(
        2 / 3 * fit["frame_pieces"], rel=1e-6
    )
    assert judged["misfit"] == []
    # Parallel bars 16 cm apart: the 3-bin (12 cm) window is wider than half
    # of it, so a shift past 8 cm would read as the next bar's; the check
    # passes no verdict even in place.
    assert fit["frame_spacing"] == pytest.approx(16.0)
    assert len(judged["unverified"]) == 1
    assert "half the corridor spacing" in judged["unverified"][0]


def test_known_limit_a_moved_lattice_is_not_verified_rather_than_a_mismatch(
    lattice_samples,
):
    # A known limit, not a requirement: moved 1.5 bins (6 cm) horizontally,
    # the refitted shift recovers it within 0.2 bin, but a single step reads
    # under a bin (samples moved off a bar leave its window or go to a
    # crossing bar), so the check gives "not verified", not a mismatch.
    env, samples = lattice_samples
    moved = _judged_lattice(env, samples + np.array([6.0, 0.0, 0.0]))
    fit = moved["fit"]
    np.testing.assert_allclose(fit["frame_offset"], [6.0, 0.0, 6.0], atol=0.8)
    assert fit["frame_offset_judged"] > 4.0 > fit["frame_offset_judged_one_step"]
    assert moved["misfit"] == []
    assert any("depends on the estimator" in r for r in moved["unverified"])


def test_a_single_straight_track_cannot_read_its_along_track_offset():
    # A straight 100 cm track cut at 15 and 85 cm: three overlapping 50 cm
    # pieces, all along x. Across it each adds 1 (2 with any one left out);
    # along it nothing, so a 4 cm shift along the track is not read.
    rng = np.random.default_rng(25)
    samples = np.column_stack([rng.uniform(0, 100, 4000), rng.normal(0, 2.0, 4000)])
    cut = [[0, 0], [15, 0], [85, 0], [100, 0]]
    judged = _judged_line(cut, samples + np.array([4.0, 0.0]))
    fit = judged["fit"]
    assert fit["frame_pieces"] == 3
    assert fit["frame_min_eigenvalue"] == pytest.approx(0.0, abs=1e-12)
    assert fit["frame_max_eigenvalue"] == pytest.approx(3.0)
    np.testing.assert_allclose(np.abs(fit["frame_unconstrained"]), [[1.0, 0.0]])
    assert fit["frame_judged_unconstrained"] == 1
    assert fit["frame_offset"][0] == 0.0
    assert abs(fit["frame_offset"][1]) < 0.2 * FRAME_BIN
    # Drawn as one segment it is one piece, and no single piece decides: no
    # direction is read.
    one = _judged_line([[0, 0], [100, 0]], samples)["fit"]
    assert one["frame_pieces"] == 1
    assert one["frame_judged_unconstrained"] == 2


def test_one_short_arm_alone_does_not_decide_the_frame_offset():
    # The T of the reward-arm test: only its 15 cm arm crosses x, so x is not
    # read and the arm's 3 cm lean stays a reported local feature; y is read
    # by the two halves.
    from _geometry import graph_environment, judge_declared

    graph = graph_from_points(
        {"a": (0.0, 0.0), "j": (50.0, 0.0), "b": (100.0, 0.0), "r": (50.0, 15.0)},
        [("a", "j"), ("j", "b"), ("j", "r")],
    )
    env = graph_environment(graph, bin_size=2.5, track_width=60.0, units="cm", name="t")
    rng = np.random.default_rng(11)
    main = np.column_stack([rng.uniform(0, 100, 8000), rng.normal(0, 1.0, 8000)])
    arm = np.column_stack(
        [50.0 + 3.0 + rng.normal(0, 1.0, 1200), rng.uniform(1, 15, 1200)]
    )
    samples = np.vstack([main, arm])
    fit = judge_declared(env, np.arange(len(samples)) / 30.0, samples, bin_size=2.5)[
        "fit"
    ]
    assert fit["frame_pieces"] == 3
    np.testing.assert_allclose(np.abs(fit["frame_unconstrained"]), [[1.0, 0.0]])
    assert fit["frame_offset"][0] == 0.0
    # Moved 4 cm in y, the two halves read it, and it survives dropping either.
    moved = judge_declared(
        env,
        np.arange(len(samples)) / 30.0,
        samples + np.array([0.0, 4.0]),
        bin_size=2.5,
    )
    assert moved["fit"]["frame_offset"][1] == pytest.approx(4.0, abs=0.5)
    assert any("as a whole" in r for r in moved["misfit"])
