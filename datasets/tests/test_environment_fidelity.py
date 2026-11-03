from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
from _record import Recording
from scipy.spatial import cKDTree
from studies.environment_fidelity import (
    Apparatus,
    Reference,
    along_track_distance,
    bar_alignment,
    corner_links,
    edge_midpoint_gaps,
    face_adjacent,
    grid_index,
    hole_sizes,
    lattice_reference,
    on_track_shares,
    outline_measures,
    project,
    verdict,
)

from neurospatial import Environment

BIN = 2.5


def _grid_points(x_range, y_range, step=0.5):
    xs = np.arange(*x_range, step)
    ys = np.arange(*y_range, step)
    return np.column_stack([np.repeat(xs, len(ys)), np.tile(ys, len(xs))])


@pytest.fixture(scope="module")
def u_arena():
    # Two 10 cm arms 20 cm apart joined by a 10 cm base: a U, densely sampled.
    samples = np.vstack(
        [
            _grid_points((0, 10), (0, 50)),
            _grid_points((30, 40), (0, 50)),
            _grid_points((10, 30), (0, 10)),
        ]
    )
    return samples, Environment.from_samples(samples, bin_size=BIN)


def test_link_straight_across_the_gap_is_flagged(u_arena):
    samples, env = u_arena
    centers = np.asarray(env.bin_centers)
    pairs = np.asarray(env.edge_pairs)
    left = int(env.bin_at(np.array([[8.75, 41.25]]))[0])
    right = int(env.bin_at(np.array([[31.25, 41.25]]))[0])
    across = np.vstack([pairs, [left, right]])
    gaps = edge_midpoint_gaps(centers, across, cKDTree(samples))
    # The link's midpoint lies in the 20 cm gap, about 10 cm from either arm.
    midpoint = centers[[left, right]].mean(axis=0)
    assert gaps[-1] == pytest.approx(np.min(np.linalg.norm(samples - midpoint, axis=1)))
    assert gaps[-1] > 3 * BIN
    # No real link of the U is far from the samples (the diagonals at its two
    # inner corners cross a corner the samples miss by under a bin).
    assert np.all(gaps[:-1] <= BIN)


def test_open_box_has_no_unsupported_links():
    samples = _grid_points((0, 50), (0, 50))
    env = Environment.from_samples(samples, bin_size=BIN)
    gaps = edge_midpoint_gaps(
        np.asarray(env.bin_centers), np.asarray(env.edge_pairs), cKDTree(samples)
    )
    assert len(gaps) > 0
    assert np.all(gaps <= BIN / 2)


def test_holes_are_enclosed_unvisited_patches():
    active = np.ones((8, 8), dtype=bool)
    active[2:4, 2:5] = False  # enclosed: 6 cells
    active[6, 6] = False  # enclosed: 1 cell
    active[0, 3:5] = False  # touches the edge of the grid: outside, not a hole
    np.testing.assert_array_equal(hole_sizes(active), [6, 1])
    assert hole_sizes(np.ones((5, 5), dtype=bool)).size == 0


def test_corner_links_only_where_the_block_is_empty():
    # Two visited cells touching at one corner, and a 2 x 2 block.
    active = np.zeros((6, 6), dtype=bool)
    active[0, 0] = active[1, 1] = True
    active[3:5, 3:5] = True
    index = np.argwhere(active)
    lookup = {tuple(c): k for k, c in enumerate(index)}
    pairs = np.array(
        [
            [lookup[(0, 0)], lookup[(1, 1)]],  # through a corner only
            [lookup[(3, 3)], lookup[(4, 4)]],  # diagonal inside a full block
            [lookup[(3, 3)], lookup[(3, 4)]],  # orthogonal
        ]
    )
    np.testing.assert_array_equal(
        corner_links(index, pairs, active), [True, False, False]
    )


def test_grid_index_matches_active_mask(u_arena):
    _, env = u_arena
    index = grid_index(env)
    assert np.all(np.asarray(env.active_mask)[tuple(index.T)])
    assert len(np.unique(index, axis=0)) == env.n_bins


def test_along_track_distance_goes_round_the_corner():
    # An L: (0, 0) -> (10, 0) -> (10, 10).
    ref = Reference(
        np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]]),
        np.array([[0, 1], [1, 2]]),
        "L",
    )
    points = np.array([[2.0, 1.0], [11.0, 7.0], [5.0, 0.0]])
    dist, seg, along = project(points, ref)
    np.testing.assert_allclose(dist, [1.0, 1.0, 0.0])
    d = along_track_distance(
        ref, seg[[0, 0]], along[[0, 0]], seg[[1, 2]], along[[1, 2]]
    )
    # 8 to the corner then 7 up; and 3 along the same segment.
    np.testing.assert_allclose(d, [15.0, 3.0])


def test_along_track_distance_matches_networkx_on_a_lattice():
    bars = []
    for axis in range(3):
        others = [a for a in range(3) if a != axis]
        for i in range(3):
            for j in range(3):
                start = np.zeros(3)
                start[others] = [10.0 * i, 10.0 * j]
                stop = start.copy()
                stop[axis] = 20.0
                bars.append([start, stop])
    ref = lattice_reference(np.array(bars), n_cubes=2)
    assert len(ref.nodes) == 27
    assert len(ref.edges) == 54
    graph = nx.Graph()
    for (u, v), length in zip(ref.edges, ref.lengths, strict=True):
        graph.add_edge(int(u), int(v), weight=length)
    corner = int(np.argmin(np.linalg.norm(ref.nodes, axis=1)))
    far = int(np.argmax(np.linalg.norm(ref.nodes, axis=1)))
    # From one corner of the 2 x 2 x 2 lattice to the opposite one: 60 along bars.
    assert nx.shortest_path_length(
        graph, corner, far, weight="weight"
    ) == pytest.approx(60.0)
    points = ref.nodes[[corner, far]] + 0.001
    _, seg, along = project(points, ref)
    d = along_track_distance(ref, seg[:1], along[:1], seg[1:], along[1:])
    assert d[0] == pytest.approx(60.0, abs=0.01)


def test_outline_of_a_well_sampled_square():
    # A 100 cm box whose samples come within 1 cm of each wall.
    samples = _grid_points((1, 99.5), (1, 99.5), step=1.0)
    env = Environment.from_samples(samples, bin_size=BIN)
    out = outline_measures(
        env, samples, Apparatus("open", "test", True, shape="square", size=100.0)
    )
    assert out["stated_area_covered"] > 0.97
    assert out["bin_centres_outside_stated"] < 0.03
    assert out["hull_over_min_rectangle"] > 0.99
    assert out["wall_gap_median"] < BIN


def test_bar_alignment_of_movement_along_and_across_the_bars():
    cube = Reference(
        np.array([[0.0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 10]]),
        np.array([[0, 1], [0, 2], [0, 3]]),
        "three axes",
    )
    times = np.arange(200) * 0.02
    along_x = np.column_stack([30.0 * times, np.zeros(200), np.zeros(200)])
    diagonal = np.repeat(30.0 * times[:, None] / np.sqrt(3), 3, axis=1)

    def share(positions):
        rec = Recording("t", "t", "t", "t", times=times, positions=positions)
        return bar_alignment(rec, cube)

    out = share(along_x)
    assert out["n_axes"] == 3
    assert out["share_within_degrees"] == 1.0
    # A cube diagonal is 54.7 degrees from every axis.
    assert share(diagonal)["share_within_degrees"] == 0.0
    assert 0.1 < out["share_isotropic"] < 0.3


def test_face_adjacent_pairs_are_across_the_bars_not_along_one():
    x = np.array([1.0, 0.0, 0.0])
    foot = np.array(
        [
            [5.0, 0.0, 0.0],  # middle of a segment of the bar along x at y = 0
            [5.0, 10.0, 0.0],  # the parallel bar across the face: adjacent
            [15.0, 0.0, 0.0],  # next segment of the same bar: not adjacent
            [5.0, 0.0, 10.0],  # the parallel bar above: adjacent
        ]
    )
    i, j = face_adjacent(foot, np.repeat(x[None], 4, axis=0), spacing=10.0)
    # (1, 3) are 14.1 apart, across a face diagonal: not adjacent either.
    assert list(zip(i.tolist(), j.tolist(), strict=True)) == [(0, 1), (0, 3)]


def test_on_track_shares_use_half_the_full_width():
    samples = np.array([0.0, 2.0, 3.0, 3.5])
    bins = np.array([0.0, 4.0, 4.5])
    on_samples, on_bins = on_track_shares(samples, bins, width=6.0, bin_size=2.0)
    assert on_samples == pytest.approx(3 / 4)  # within 3 of the centreline
    assert on_bins == pytest.approx(2 / 3)  # within 3 + 1


def _row(major=1, bins_on_track=0.95, ratio=1.0, stated=True):
    return {
        "env": {"major_components": major},
        "reference": {
            "reference_length": 100.0,
            "track_width_stated": stated,
            "bins_on_track": bins_on_track,
            "env_over_track_median": ratio,
        },
    }


def test_verdict_judges_only_stated_facts():
    track = Apparatus("track", "test", True, track_length=100.0, track_width=6.0)
    assert verdict(_row(), track) == ("matches", "")
    status, reason = verdict(_row(major=2, ratio=0.7), track)
    assert status == "mismatch"
    assert "2 parts" in reason
    assert "0.70" in reason
    # An assumed width is reported, not judged.
    assert verdict(_row(bins_on_track=0.5, stated=False), track)[0] == "matches"
    assert verdict(_row(bins_on_track=0.5), track)[0] == "mismatch"
    assert verdict(_row(), Apparatus("track", "test", False))[0] == "not verified"


def test_verdict_tests_loop_closure():
    loop = Apparatus("track", "test", True, notes="1-D loop")

    def row(end_to_end, camera):
        return {
            "env": {"major_components": 1},
            "one_d": {
                "range": [0.0, 200.0],
                "end_to_end_geodesic": end_to_end,
                "camera_distance_between_ends": camera,
            },
        }

    assert verdict(row(200.0, 3.0), loop)[0] == "mismatch"  # a loop cut open
    assert "does not close" in verdict(row(200.0, 60.0), loop)[1]
    assert verdict(row(10.0, 3.0), loop)[0] == "matches"


def _declared_track(positions, *, placed_by, geometry=None, width=6.0, graph=None):
    from _tracks import declared_geometry, straight_track

    if graph is None:
        graph, _ = straight_track(positions, 100.0)
    entries, _ = declared_geometry(
        graph,
        positions,
        kind="track",
        source="test",
        times=np.arange(len(positions)) / 30.0,
        tracked_point="simulated",
        bin_size=2.5,
        physical_width=width,
        units="cm",
        name="track",
        placement="p",
        placed_by=placed_by,
    )
    return Recording(
        dataset="track",
        session="s",
        source="s",
        license="n/a",
        times=np.arange(len(positions)) / 30.0,
        positions=positions,
        position_units="cm",
        geometry={**(geometry or {}), **entries},
        params={"bin_size": 2.5},
    )


def _measure(rec, app):
    from studies.environment_fidelity import measure_declared, reference_for, tracked

    return measure_declared(rec, app, tracked(rec), reference_for(rec, app))


@pytest.fixture(scope="module")
def track_samples():
    """A 100 cm straight track sampled 1 cm about its axis."""
    rng = np.random.default_rng(5)
    return np.column_stack([rng.uniform(0, 100, 8000), rng.normal(0, 1.0, 8000)])


TRACK = Apparatus("track", "test", True, track_length=100.0, track_width=6.0)


def test_declared_track_against_an_independent_reference_can_match(track_samples):
    # The authors' graph (here the true axis) checks a layout fitted to the samples.
    graph = nx.Graph()
    graph.add_node("a", pos=(0.0, 0.0))
    graph.add_node("b", pos=(100.0, 0.0))
    graph.add_edge("a", "b", distance=100.0)
    rec = _declared_track(
        track_samples, placed_by="fitted to the samples", geometry={"graph": graph}
    )
    out = _measure(rec, TRACK)
    assert out["reference_independent"] is True
    assert out["frac_samples_outside"] < 0.01
    assert out["physical_width_kind"] == "stated"
    assert out["track_width_kind"] == "placement tolerance"
    assert out["reference"]["env_over_track_median"] == pytest.approx(1.0, abs=0.05)
    assert out["verdict"] == "matches"


def test_declared_track_without_an_independent_reference_is_not_verified(
    track_samples,
):
    # Placed on the samples' axis, which is also the study's reference.
    out = _measure(
        _declared_track(track_samples, placed_by="samples' principal axis"), TRACK
    )
    assert out["reference_independent"] is False
    assert out["verdict"] == "not verified"
    assert "no independent reference" in out["verdict_reason"]
    # No reference at all (no stated length): not verified either.
    no_length = Apparatus("track", "test", True, track_width=6.0)
    out = _measure(
        _declared_track(track_samples, placed_by="fitted to the samples"), no_length
    )
    assert "reference" not in out
    assert out["verdict"] == "not verified"


def test_head_scatter_past_the_physical_width_is_reported_not_judged(track_samples):
    # The head swings 6 cm (SD) about a 6 cm track: 62 % of samples are
    # beyond the physical width, yet the track is where they are.
    leaning = track_samples.copy()
    leaning[:, 1] = np.random.default_rng(6).normal(0, 6.0, len(leaning))
    graph = nx.Graph()
    graph.add_node("a", pos=(0.0, 0.0))
    graph.add_node("b", pos=(100.0, 0.0))
    graph.add_edge("a", "b", distance=100.0)
    rec = _declared_track(
        leaning, placed_by="fitted to the samples", geometry={"graph": graph}
    )
    out = _measure(rec, TRACK)
    assert out["outside_physical_width"] == pytest.approx(0.62, abs=0.03)
    assert out["frac_samples_outside"] < 0.02  # the tolerance keeps them
    assert out["misfit"] == []
    assert out["verdict"] == "matches"


def test_authors_track_beside_its_samples_is_a_mismatch(track_samples):
    # The authors' graph (not fitted, so the loader keeps it) lies 6 cm to one
    # side of every sample: misfit evidence, whatever the width.
    from _tracks import polyline_graph

    shifted = track_samples + np.array([0.0, 6.0])
    axis = polyline_graph(np.array([[0.0, 0.0], [100.0, 0.0]]), closed=False)
    out = _measure(
        _declared_track(shifted, placed_by="authors' graph", graph=axis), TRACK
    )
    assert out["verdict"] == "mismatch"
    assert "to one side" in out["verdict_reason"]


def test_merged_replaces_only_the_rerun_datasets_and_their_views():
    from studies.environment_fidelity import merged

    previous = [
        {"dataset": "a"},
        {"dataset": "figure8"},
        {"dataset": "figure8_camera"},
        {"dataset": "b"},
    ]
    new = [{"dataset": "figure8", "v": 2}, {"dataset": "figure8_linearized", "v": 2}]
    out = merged(previous, new, ("figure8",))
    assert [row["dataset"] for row in out] == [
        "a",
        "figure8",
        "figure8_linearized",
        "b",
    ]
    assert merged(previous, [{"dataset": "c"}], ("c",))[-1] == {"dataset": "c"}


def test_geodesic_pairs_skip_samples_outside_a_declared_track():
    # An L-shaped track, 6 cm wide; a third of the samples sit 20 cm off it.
    from _geometry import graph_environment
    from _tracks import polyline_graph
    from studies.environment_fidelity import measure_env

    rng = np.random.default_rng(2)
    graph = polyline_graph(
        np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0]]), closed=False
    )
    env = graph_environment(
        graph, bin_size=2.5, track_width=6.0, units="cm", name="L", root=None
    )
    t = rng.uniform(0, 200, 6000)
    on = np.where(
        (t < 100)[:, None],
        np.column_stack([t, np.zeros_like(t)]),
        np.column_stack([np.full_like(t, 100.0), t - 100]),
    )
    off = on[:3000] + np.array([0.0, 20.0])
    samples = np.vstack([on, off])
    out = measure_env(env, samples, np.random.default_rng(0))
    # Pairs on the track: the path along it over the straight line, 1 on one
    # arm and up to sqrt(2) across the corner; never below 1.
    assert out["geodesic_ratio_median"] >= 1.0
    assert out["geodesic_ratio_p95"] <= 1.5
