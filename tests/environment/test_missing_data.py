"""A NaN sample is a missing sample on every layout, never an error.

Per-sample queries give a missing sample bin ``-1``, ``False`` or a NaN value,
and the remaining samples get exactly the answers they get on their own.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
from shapely.geometry import Polygon

from neurospatial import Environment
from neurospatial.layout import create_layout


def _grid() -> Environment:
    positions = np.random.default_rng(0).uniform(0, 30, (2000, 2))
    return Environment.from_samples(positions, bin_size=3.0)


def _holed() -> Environment:
    mask = np.ones((10, 10), dtype=bool)
    mask[3:6, 4:7] = False
    edges = (np.linspace(0, 30, 11), np.linspace(0, 30, 11))
    return Environment.from_grid_mask(mask, grid_edges=edges)


def _hexagonal() -> Environment:
    return Environment.from_layout(
        create_layout(
            "Hexagonal", hexagon_width=4.0, dimension_ranges=[(0, 30), (0, 30)]
        )
    )


def _track() -> Environment:
    graph = nx.Graph()
    for node, pos in enumerate([(0.0, 0.0), (20.0, 0.0), (20.0, 20.0)]):
        graph.add_node(node, pos=pos)
    graph.add_edge(0, 1, distance=20.0)
    graph.add_edge(1, 2, distance=20.0)
    return Environment.from_graph(
        graph, edge_order=[(0, 1), (1, 2)], edge_spacing=0.0, bin_size=2.0
    )


def _mesh() -> Environment:
    return Environment.from_layout(
        create_layout(
            "TriangularMesh",
            boundary_polygon=Polygon([(0, 0), (30, 0), (30, 30), (0, 30)]),
            point_spacing=4.0,
        )
    )


def _polar() -> Environment:
    return Environment.from_polar_egocentric(
        distance_range=(0.0, 30.0),
        angle_range=(-np.pi, np.pi),
        distance_bin_size=5.0,
        angle_bin_size=np.pi / 6,
    )


_CARTESIAN = {
    "grid": _grid,
    "holed": _holed,
    "hexagonal": _hexagonal,
    "track": _track,
    "mesh": _mesh,
}


@pytest.fixture(scope="module", params=sorted(_CARTESIAN))
def cartesian_env(request) -> Environment:
    return _CARTESIAN[request.param]()


@pytest.fixture(scope="module", params=sorted([*_CARTESIAN, "polar"]))
def any_env(request) -> Environment:
    return _polar() if request.param == "polar" else _CARTESIAN[request.param]()


def _with_missing_rows(env: Environment) -> tuple[np.ndarray, np.ndarray]:
    """Points at bin centers with NaN rows interleaved, and the finite rows."""
    finite = np.asarray(env.bin_centers[[0, env.n_bins // 2, env.n_bins - 1]])
    points = np.array(
        [finite[0], [np.nan, finite[0, 1]], finite[1], [np.nan, np.nan], finite[2]]
    )
    return points, finite


_FINITE_ROWS = [0, 2, 4]
_MISSING_ROWS = [1, 3]


def test_bin_at_gives_missing_samples_minus_one(cartesian_env):
    points, finite = _with_missing_rows(cartesian_env)
    bins = cartesian_env.bin_at(points)
    assert (bins[_MISSING_ROWS] == -1).all()
    np.testing.assert_array_equal(bins[_FINITE_ROWS], cartesian_env.bin_at(finite))


def test_contains_gives_missing_samples_false(cartesian_env):
    points, finite = _with_missing_rows(cartesian_env)
    if cartesian_env.is_linearized_track:
        # A track without a declared width has no footprint to test against.
        with pytest.raises(NotImplementedError, match="track footprint"):
            cartesian_env.contains(points)
        return
    inside = cartesian_env.contains(points)
    assert not inside[_MISSING_ROWS].any()
    np.testing.assert_array_equal(inside[_FINITE_ROWS], cartesian_env.contains(finite))


def test_nearest_bin_gives_missing_samples_minus_one_and_nan_distance(any_env):
    points, finite = _with_missing_rows(any_env)
    bins, dist = any_env.nearest_bin(points, return_dist=True)
    expected_bins, expected_dist = any_env.nearest_bin(finite, return_dist=True)
    assert (bins[_MISSING_ROWS] == -1).all()
    assert np.isnan(dist[_MISSING_ROWS]).all()
    np.testing.assert_array_equal(bins[_FINITE_ROWS], expected_bins)
    np.testing.assert_array_equal(dist[_FINITE_ROWS], expected_dist)
    np.testing.assert_array_equal(any_env.nearest_bin(points), bins)


def test_nearest_bin_treats_an_infinite_coordinate_as_outside(any_env):
    point = np.array([[np.inf, 1.0]])
    bins, dist = any_env.nearest_bin(point, return_dist=True)
    assert bins.tolist() == [-1]
    assert dist.tolist() == [np.inf]


def test_interpolate_gives_missing_samples_nan(cartesian_env):
    points, finite = _with_missing_rows(cartesian_env)
    field = np.sin(np.arange(cartesian_env.n_bins) * 0.37)
    values = cartesian_env.interpolate(field, points)
    assert np.isnan(values[_MISSING_ROWS]).all()
    np.testing.assert_array_equal(
        values[_FINITE_ROWS], cartesian_env.interpolate(field, finite)
    )


def test_interpolate_reads_a_missing_bin_value_as_nan(cartesian_env):
    field = np.sin(np.arange(cartesian_env.n_bins) * 0.37)
    field[1] = np.nan
    values = cartesian_env.interpolate(field, cartesian_env.bin_centers)
    np.testing.assert_array_equal(values, field)


def _trajectory(env: Environment) -> tuple[np.ndarray, np.ndarray]:
    """Two samples per bin center along a path, repeated, 30 Hz, no gaps."""
    path = env.path_between(bins=[0, env.n_bins - 1])
    walk = np.resize(np.repeat(np.asarray(path), 2), 60)
    times = np.arange(len(walk)) / 30.0
    return times, np.asarray(env.bin_centers[walk], dtype=float)


def test_occupancy_gives_a_missing_sample_no_time(cartesian_env):
    times, positions = _trajectory(cartesian_env)
    with_missing = positions.copy()
    with_missing[[10, 30]] = np.nan
    occupancy = cartesian_env.occupancy(times, with_missing)

    # The intervals that start at the missing samples are the only time lost.
    dt = np.diff(times)
    np.testing.assert_allclose(occupancy.sum(), dt.sum() - dt[10] - dt[30])
    expected = np.zeros(cartesian_env.n_bins)
    keep = np.setdiff1d(np.arange(len(dt)), [10, 30])
    np.add.at(expected, cartesian_env.bin_at(positions)[keep], dt[keep])
    np.testing.assert_allclose(occupancy, expected)


def test_bin_sequence_marks_or_drops_a_missing_sample(cartesian_env):
    times, positions = _trajectory(cartesian_env)
    positions[10] = np.nan
    bins = cartesian_env.bin_at(positions)
    marked = cartesian_env.bin_sequence(times, positions, dedup=False)
    dropped = cartesian_env.bin_sequence(
        times, positions, dedup=False, outside_value=None
    )
    np.testing.assert_array_equal(marked, bins)
    assert marked[10] == -1
    np.testing.assert_array_equal(dropped, np.delete(bins, 10))


@pytest.mark.parametrize("lag", [1, 2])
def test_transitions_skip_pairs_with_a_missing_or_outside_sample(cartesian_env, lag):
    times, positions = _trajectory(cartesian_env)
    positions[10] = np.nan
    positions[20] = cartesian_env.bin_centers.max(axis=0) + 100.0
    counts = cartesian_env.transitions(
        times=times, positions=positions, lag=lag, normalize=False, allow_teleports=True
    )

    # Count the pairs whose two samples are both inside, by hand.
    bins = cartesian_env.bin_at(positions)
    expected = np.zeros((cartesian_env.n_bins, cartesian_env.n_bins))
    for source, target in zip(bins[:-lag], bins[lag:], strict=True):
        if source >= 0 and target >= 0:
            expected[source, target] += 1
    np.testing.assert_array_equal(counts.toarray(), expected)
    # The missing sample removes the pair it starts and the one it ends.
    assert bins[10] == -1
    assert counts.sum() <= len(bins) - lag - 2


def test_a_precomputed_sequence_with_minus_one_is_still_refused(cartesian_env):
    with pytest.raises(ValueError, match="Invalid bin indices"):
        cartesian_env.transitions(np.array([0, -1, 0], dtype=np.int32))
