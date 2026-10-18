"""Discrete geometry snapshots validate inputs and own their published buffers."""

from __future__ import annotations

import copy
import pickle
import re
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from neurospatial.geometry._binning import BinConnections, SpatialBins
from neurospatial.geometry._capabilities import RectilinearGrid


@pytest.fixture
def bin_arrays():
    """Three bins with distinct measures."""
    return {
        "centers": np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        "sizes": np.array([1.0, 2.0, 3.0]),
        "measure_dim": 2,
    }


@pytest.fixture
def connection_arrays():
    """A noncanonical edge order with a valid zero navigation distance."""
    return {
        "pairs": np.array([[2, 1], [1, 0]], dtype=np.intp),
        "pair_distances": np.array([0.0, 2.0]),
    }


def _assert_names_index(exc, field, index):
    if index is not None:
        assert re.search(rf"{field}(?:\[|\(){index}(?:\]|,|\))", str(exc.value)), str(
            exc.value
        )


@pytest.mark.parametrize(
    ("field", "invalid", "index"),
    [
        ("centers", np.ones(3), None),
        ("centers", np.ones((1, 3, 1)), None),
        ("centers", np.array([[0.0, 0.0], [np.nan, 1.0], [2.0, 2.0]]), 1),
        ("centers", np.array([[0.0, 0.0], [1.0, np.inf], [2.0, 2.0]]), 1),
        ("sizes", np.ones((3, 1)), None),
        ("sizes", np.ones(2), None),
        ("sizes", np.array([1.0, 0.0, 1.0]), 1),
        ("sizes", np.array([1.0, -1.0, 1.0]), 1),
        ("sizes", np.array([1.0, np.nan, 1.0]), 1),
        ("sizes", np.array([1.0, np.inf, 1.0]), 1),
        ("measure_dim", 0, None),
        ("measure_dim", 3, None),
    ],
)
def test_spatial_bins_rejects_invalid_arrays(bin_arrays, field, invalid, index):
    """A malformed coordinate or measure cannot become geometry."""
    bin_arrays[field] = invalid
    with pytest.raises(ValueError, match=field) as exc:
        SpatialBins.from_arrays(**bin_arrays)
    _assert_names_index(exc, field, index)


@pytest.mark.parametrize(
    ("field", "invalid", "index"),
    [
        ("pairs", np.array([0, 1]), None),
        ("pairs", np.zeros((2, 3), dtype=np.intp), None),
        ("pairs", np.array([[2, 1], [-1, 0]]), 1),
        ("pairs", np.array([[2, 1], [0, 0]]), 1),
        ("pairs", np.array([[2, 1], [2, 1]]), 1),
        ("pairs", np.array([[2, 1], [1, 2]]), 1),
        ("pairs", np.array([[2.0, 1.0], [0.5, 0.0]]), 1),
        ("pairs", np.array([[True, False], [False, True]]), None),
        ("pair_distances", np.ones((2, 1)), None),
        ("pair_distances", np.ones(3), None),
        ("pair_distances", np.array([1.0, -1.0]), 1),
        ("pair_distances", np.array([1.0, np.nan]), 1),
        ("pair_distances", np.array([1.0, np.inf]), 1),
    ],
)
def test_bin_connections_rejects_invalid_arrays(
    connection_arrays, field, invalid, index
):
    """A malformed neighbor pair or distance cannot become a connection."""
    connection_arrays[field] = invalid
    with pytest.raises(ValueError, match=field) as exc:
        BinConnections.from_arrays(**connection_arrays)
    _assert_names_index(exc, field, index)


def test_bin_connections_rejects_is_open_of_wrong_shape_or_dtype(connection_arrays):
    for is_open in (np.array([True]), np.array([1, 0])):
        with pytest.raises(ValueError, match="is_open"):
            BinConnections.from_arrays(**connection_arrays, is_open=is_open)


def test_bin_connections_check_bins_names_an_out_of_range_pair(connection_arrays):
    """A pair naming a bin that does not exist is caught against the bin count."""
    connection_arrays["pairs"] = np.array([[2, 1], [3, 0]])
    connections = BinConnections.from_arrays(**connection_arrays)
    with pytest.raises(ValueError, match="pairs") as exc:
        connections.check_bins(3)
    _assert_names_index(exc, "pairs", 1)
    assert "[0, 3)" in str(exc.value)
    connections.check_bins(4)


@pytest.mark.parametrize("constructor", [SpatialBins.from_arrays, SpatialBins])
def test_spatial_bins_owns_read_only_arrays(constructor):
    """Freezing noncontiguous inputs must neither freeze nor retain their bases."""
    bases = {
        "centers": np.arange(24.0).reshape(6, 4),
        "sizes": np.arange(1.0, 7.0),
    }
    inputs = {name: array[::2] for name, array in bases.items()}
    inputs["centers"] = inputs["centers"][:, ::2]
    expected = {name: array.copy() for name, array in inputs.items()}
    bins = constructor(inputs["centers"], inputs["sizes"], 2)

    for name, source in inputs.items():
        published = getattr(bins, name)
        assert not np.shares_memory(published, source), name
        assert source.flags.writeable and bases[name].flags.writeable, name
        source[...] = 0
        np.testing.assert_array_equal(published, expected[name])
        assert not published.flags.writeable, name
        with pytest.raises(ValueError, match="read-only"):
            published.flat[0] = 0


@pytest.mark.parametrize(
    "constructor",
    [
        BinConnections.from_arrays,
        lambda pairs, pair_distances, is_open: BinConnections(
            pairs, pair_distances, is_open
        ),
    ],
)
def test_bin_connections_owns_read_only_arrays(constructor):
    """Freezing noncontiguous inputs must neither freeze nor retain their bases."""
    bases = {
        "all_pairs": np.array([[2, 0], [9, 9], [1, 2], [9, 9]]),
        "all_distances": np.array([2.0, 99.0, 3.0, 99.0]),
        "is_open": np.array([True, True, False, True]),
    }
    inputs = {name: array[::2] for name, array in bases.items()}
    expected = {name: array.copy() for name, array in inputs.items()}
    connections = constructor(*inputs.values())

    for name, source in inputs.items():
        published = getattr(connections, name)
        assert not np.shares_memory(published, source), name
        assert source.flags.writeable and bases[name].flags.writeable, name
        source[...] = 0
        np.testing.assert_array_equal(published, expected[name])
        assert not published.flags.writeable, name
        with pytest.raises(ValueError, match="read-only"):
            published.flat[0] = 0
    np.testing.assert_array_equal(connections.pairs, [[2, 0]])
    np.testing.assert_array_equal(connections.pair_distances, [2.0])
    for name in ("pairs", "pair_distances"):
        assert not getattr(connections, name).flags.writeable, name


def test_bin_connections_preserves_pair_order_and_normalizes_dtypes(
    connection_arrays,
):
    """Flux vectors keep their edge positions, even with reversed orientations."""
    connection_arrays["pairs"] = connection_arrays["pairs"].astype(np.int16)
    connections = BinConnections.from_arrays(**connection_arrays)
    np.testing.assert_array_equal(connections.pairs, [[2, 1], [1, 0]])
    np.testing.assert_array_equal(connections.pair_distances, [0.0, 2.0])
    np.testing.assert_array_equal(connections.is_open, [True, True])
    assert connections.pair_distances.dtype == np.float64
    assert connections.pairs.dtype == np.intp


def test_bin_connections_open_pairs_follow_all_pairs_order():
    """Closed pairs stay listed; the open pairs keep their relative order."""
    connections = BinConnections.from_arrays(
        [[2, 1], [1, 0], [0, 2]], [0.0, 2.0, 3.0], is_open=[True, False, True]
    )
    np.testing.assert_array_equal(connections.all_pairs, [[2, 1], [1, 0], [0, 2]])
    np.testing.assert_array_equal(connections.pairs, [[2, 1], [0, 2]])
    np.testing.assert_array_equal(connections.pair_distances, [0.0, 3.0])


def test_spatial_bins_normalizes_dtypes(bin_arrays):
    bin_arrays["centers"] = bin_arrays["centers"].astype(np.float32)
    bins = SpatialBins.from_arrays(**bin_arrays)
    assert bins.centers.dtype == bins.sizes.dtype == np.float64
    assert (bins.n_bins, bins.n_dims, bins.measure_dim) == (3, 2, 2)


def test_spatial_bins_accepts_an_isolated_track_bin():
    """An embedded track has length measures and can have no neighbor pairs."""
    bins = SpatialBins.from_arrays(centers=[[1.0, 2.0]], sizes=[0.5], measure_dim=1)
    connections = BinConnections.from_arrays(
        pairs=np.empty((0, 2), dtype=np.intp), pair_distances=[]
    )
    connections.check_bins(bins.n_bins)
    assert connections.pairs.shape == (0, 2)
    assert connections.pair_distances.shape == (0,)
    assert (bins.n_bins, bins.n_dims, bins.measure_dim) == (1, 2, 1)


def test_spatial_bins_identity_semantics(bin_arrays):
    """Equal-valued snapshots are distinct hashable objects with frozen fields."""
    bins = SpatialBins.from_arrays(**bin_arrays)
    equal_values = SpatialBins.from_arrays(**bin_arrays)
    assert bins == bins
    assert bins != equal_values
    keyed = {bins: 1, equal_values: 2}
    assert len(keyed) == 2
    assert keyed[bins] == 1
    assert keyed[equal_values] == 2
    with pytest.raises(FrozenInstanceError):
        bins.measure_dim = 1


@pytest.mark.parametrize(
    "restore", [copy.deepcopy, lambda x: pickle.loads(pickle.dumps(x))]
)
def test_spatial_bins_serialization_preserves_read_only_arrays(bin_arrays, restore):
    bins = SpatialBins.from_arrays(**bin_arrays)
    restored = restore(bins)
    assert restored != bins
    for name in ("centers", "sizes"):
        array = getattr(restored, name)
        np.testing.assert_array_equal(array, getattr(bins, name))
        assert not array.flags.writeable, name
        assert not np.shares_memory(array, getattr(bins, name)), name


@pytest.mark.parametrize(
    "restore", [copy.deepcopy, lambda x: pickle.loads(pickle.dumps(x))]
)
def test_bin_connections_serialization_preserves_read_only_arrays(restore):
    connections = BinConnections.from_arrays(
        [[2, 1], [1, 0]], [0.0, 2.0], is_open=[True, False]
    )
    restored = restore(connections)
    assert restored != connections
    for name in ("all_pairs", "all_distances", "is_open", "pairs", "pair_distances"):
        array = getattr(restored, name)
        np.testing.assert_array_equal(array, getattr(connections, name))
        assert not array.flags.writeable, name
        assert not np.shares_memory(array, getattr(connections, name)), name


def test_rectilinear_component_owns_edges_and_active_mask():
    """The cell descriptor cannot retain editable buffers from a layout."""
    x = np.arange(6.0)[::2]
    y = np.arange(6.0)[::2]
    active = np.ones((2, 2), dtype=bool)
    grid = RectilinearGrid(edges=(x, y), shape=(2, 2), active_mask=active)
    x[:] = -1
    y[:] = -1
    active[:] = False
    for edge in grid.edges:
        np.testing.assert_array_equal(edge, [0.0, 2.0, 4.0])
        assert not edge.flags.writeable
    np.testing.assert_array_equal(grid.active_mask, True)
    assert not grid.active_mask.flags.writeable
    assert x.flags.writeable and y.flags.writeable and active.flags.writeable


@pytest.mark.parametrize(
    "restore", [copy.deepcopy, lambda x: pickle.loads(pickle.dumps(x))]
)
def test_rectilinear_component_identity_and_serialization(restore):
    grid = RectilinearGrid(
        edges=(np.arange(3.0),) * 2,
        shape=(2, 2),
        active_mask=np.ones((2, 2), dtype=bool),
    )
    restored = restore(grid)
    assert restored != grid
    assert len({grid, restored}) == 2
    for array in (*restored.edges, restored.active_mask):
        assert not array.flags.writeable
    with pytest.raises(FrozenInstanceError):
        grid.shape = (4,)
