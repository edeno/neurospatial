"""Environment geometry and capabilities are published as one stable snapshot."""

from __future__ import annotations

import copy
import pickle
import warnings
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import networkx as nx
import numpy as np
import pytest
from environment_helpers import (
    built_from,
    face_graph,
    native_layout,
    publish_binning,
    publish_environment,
)

from neurospatial import CartesianOnlyError, Environment
from neurospatial.environment._fields import face_geometry
from neurospatial.geometry._diffusion import (
    FaceGeometry,
)
from neurospatial.layout import create_layout

_BuiltEnvironment = Environment | Environment


def _source_layout(env: _BuiltEnvironment):
    """Build an independent native oracle for the published snapshot."""
    return native_layout(env)


def _query_points(env: _BuiltEnvironment) -> np.ndarray:
    """Probe active centers, nearby boundaries, outside points and missing data."""
    low = np.min(env.bin_centers, axis=0)
    high = np.max(env.bin_centers, axis=0)
    margin = np.maximum(high - low, 1.0) * 0.2
    random = np.random.default_rng(0).uniform(
        low - margin, high + margin, (128, env.n_dims)
    )
    nonfinite = np.tile(env.bin_centers[0], (2, 1))
    nonfinite[0, 0] = np.nan
    nonfinite[1, -1] = np.inf
    return np.concatenate((env.bin_centers, random, low[None], high[None], nonfinite))


def _lookup_with_warnings(lookup, points):
    """Retain existing non-finite lookup warnings as part of parity checks."""
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", RuntimeWarning)
        indices = lookup(points)
    return indices, [(warning.category, str(warning.message)) for warning in captured]


def test_binning_matches_layout(built_environment: _BuiltEnvironment) -> None:
    """Measures and edge-indexed vectors retain the layout's scientific meaning."""
    env = built_environment
    bins = env._binning.bins
    edges = list(env.connectivity.edges(data=True))
    np.testing.assert_array_equal(bins.centers, env.bin_centers)
    np.testing.assert_array_equal(bins.sizes, env.bin_sizes)
    np.testing.assert_array_equal(
        bins.pairs,
        np.array([(u, v) for u, v, _ in edges], dtype=np.intp).reshape(-1, 2),
    )
    np.testing.assert_array_equal(
        bins.pair_distances, [attrs["distance"] for _, _, attrs in edges]
    )
    assert (bins.n_bins, bins.n_dims) == (env.n_bins, env.n_dims)
    assert bins.measure_dim == (1 if env.is_linearized_track else env.n_dims)
    assert env.bin_sizes is not bins.sizes
    for array in (bins.centers, bins.sizes, bins.pairs, bins.pair_distances):
        assert not array.flags.writeable


def test_binning_capabilities_match_layout_protocols(
    built_environment: _BuiltEnvironment,
) -> None:
    """Optional components describe actual grid, track and face capabilities."""
    env = built_environment
    layout = _source_layout(env)
    binning = env._binning
    assert (binning.rectilinear is not None) == hasattr(layout, "rectilinear_grid")
    assert (binning.linearization is not None) == layout.is_linearized_track
    if not hasattr(layout, "bin_shapes"):
        assert binning.shapes is None
    if not hasattr(layout, "outline"):
        assert binning.outline is None
    assert binning.kind == env.layout_type
    if binning.rectilinear is not None:
        expected = layout.rectilinear_grid()
        assert binning.rectilinear.shape == expected.shape
        for actual, edge in zip(binning.rectilinear.edges, expected.edges, strict=True):
            np.testing.assert_array_equal(actual, edge)
            assert not actual.flags.writeable
        np.testing.assert_array_equal(
            binning.rectilinear.active_mask, expected.active_mask
        )
        assert not binning.rectilinear.active_mask.flags.writeable
    if binning.linearization is not None:
        np.testing.assert_array_equal(
            env.to_linear(env.bin_centers), layout.to_linear(env.bin_centers)
        )
        linear = env.to_linear(env.bin_centers)
        np.testing.assert_array_equal(
            env.linear_to_nd(linear), layout.linear_to_nd(linear)
        )


def test_binning_published_once(built_environment: _BuiltEnvironment) -> None:
    """Repeated reads and cache clearing keep one immutable published record."""
    env = built_environment
    published = env._binning
    assert published is env._binning
    assert env._geometry.binning is published
    env.clear_cache()
    assert env._binning is published
    np.testing.assert_array_equal(env.bin_sizes, published.bins.sizes)
    with pytest.raises(AttributeError):
        env._binning = published
    with pytest.raises(FrozenInstanceError):
        published.kind = "different"
    other = _source_layout(env).binning()
    assert other != published
    assert len({published, other}) == 2


def test_locator_preserves_layout_lookup_and_input_arrays(
    built_environment: _BuiltEnvironment,
) -> None:
    """Published lookup agrees at centers, off-grid points and non-finite rows."""
    env = built_environment
    points = _query_points(env)
    untouched = points.copy()
    expected, expected_warnings = _lookup_with_warnings(
        env._binning.locator.locate, points
    )
    actual, actual_warnings = _lookup_with_warnings(env._binning.locator.locate, points)
    np.testing.assert_array_equal(actual, expected)
    assert actual_warnings == expected_warnings
    if not env._is_polar:
        if env.arena.domain_known:
            inside = env.arena.contains(points)
            # Qhull can choose either triangle at a shared edge, depending
            # on preceding queries. Compare the same arena-filtered batch.
            expected = np.full_like(expected, -1)
            expected[inside], expected_warnings = _lookup_with_warnings(
                env._binning.locator.locate, points[inside]
            )
        indices, bin_warnings = _lookup_with_warnings(env.bin_at, points)
        np.testing.assert_array_equal(indices, expected)
        assert bin_warnings == expected_warnings
        if env.is_linearized_track and not env.arena.domain_known:
            with pytest.raises(NotImplementedError, match="track_width"):
                env.contains(points)
        else:
            contains, contains_warnings = _lookup_with_warnings(env.contains, points)
            np.testing.assert_array_equal(contains, expected != -1)
            assert contains_warnings == expected_warnings
    else:
        with pytest.raises(CartesianOnlyError):
            env.bin_at(points)
        with pytest.raises(CartesianOnlyError):
            env.contains(points)
    np.testing.assert_array_equal(points, untouched)
    assert points.flags.writeable
    assert actual.dtype == np.intp
    np.testing.assert_array_equal(actual[-2:], [-1, -1])
    for query in (points[:1], np.empty((0, env.n_dims))):
        np.testing.assert_array_equal(
            env._binning.locator.locate(query), env._binning.locator.locate(query)
        )


def test_locator_preserves_dimension_behavior(
    built_environment: _BuiltEnvironment,
) -> None:
    """The adapter preserves each layout's existing dimension validation."""
    env = built_environment
    query = np.ones((2, env.n_dims + 1))
    try:
        expected = env._binning.locator.locate(query)
    except ValueError as expected_error:
        with pytest.raises(ValueError) as actual_error:
            env._binning.locator.locate(query)
        assert str(actual_error.value) == str(expected_error)
    else:
        np.testing.assert_array_equal(env._binning.locator.locate(query), expected)


def test_faces_match_layout_and_cache_lifetime(
    built_environment: _BuiltEnvironment,
) -> None:
    """Faces share pair order and measures while caching stays on the environment."""
    env = built_environment
    graph, sizes = face_graph(_source_layout(env).binning())
    expected = FaceGeometry.from_graph(graph, sizes)
    faces = face_geometry(env._cache, env._binning)
    assert faces is face_geometry(env._cache, env._binning)
    for actual, reference in zip(faces, expected, strict=True):
        np.testing.assert_array_equal(actual, reference)
        assert not actual.flags.writeable
    np.testing.assert_array_equal(faces.edges, env._binning.bins.pairs)
    np.testing.assert_array_equal(faces.volumes, env.bin_sizes)
    if env.is_linearized_track:
        assert np.any(faces.distance != env._binning.bins.pair_distances)
    assert env._binning.faces.build() is not env._binning.faces.build()

    field = np.arange(env.n_bins, dtype=float)
    before = env.smooth(field, bandwidth=1.0)
    published = env._binning
    env.clear_cache()
    assert env._binning is published
    assert face_geometry(env._cache, env._binning) is not faces
    np.testing.assert_allclose(env.smooth(field, bandwidth=1.0), before)


@pytest.mark.parametrize(
    "restore", [copy.deepcopy, lambda x: pickle.loads(pickle.dumps(x))]
)
def test_serialized_binning_preserves_ownership(
    built_environment: _BuiltEnvironment, restore
) -> None:
    """Copying the full publication retains protected bins and cell descriptors."""
    original = built_environment._binning
    restored = restore(original)
    assert restored != original
    assert restored.bins != original.bins
    for name in ("centers", "sizes", "pairs", "pair_distances"):
        array = getattr(restored.bins, name)
        np.testing.assert_array_equal(array, getattr(original.bins, name))
        assert not array.flags.writeable
        assert not np.shares_memory(array, getattr(original.bins, name))
    if restored.rectilinear is not None:
        for array in (*restored.rectilinear.edges, restored.rectilinear.active_mask):
            assert not array.flags.writeable
    points = built_environment.bin_centers
    np.testing.assert_array_equal(
        restored.locator.locate(points), original.locator.locate(points)
    )


def test_restored_geometry_has_independent_binning_and_cached_faces() -> None:
    """A restored graph has fresh state without modifying an existing value."""
    mask = np.ones((3, 3), dtype=bool)
    edges = (np.arange(4.0),) * 2
    env = Environment.from_grid_mask(mask, grid_edges=edges)
    replacement = Environment.from_grid_mask(
        mask, grid_edges=edges, connect_diagonal_neighbors=False
    )
    old_binning, old_faces = env._binning, face_geometry(env._cache, env._binning)
    restored = Environment.from_dict(replacement.to_dict())
    assert restored._binning is not old_binning
    assert face_geometry(restored._cache, restored._binning) is not old_faces
    assert env._binning is old_binning
    assert face_geometry(env._cache, env._binning) is old_faces
    np.testing.assert_array_equal(
        restored._binning.bins.pairs, list(replacement.connectivity.edges())
    )
    assert len(restored._binning.bins.pairs) < len(old_binning.bins.pairs)


def test_invalid_binning_is_not_published(monkeypatch) -> None:
    """An invalid measure must leave the previous geometry and caches intact."""
    env = built_from(
        lambda: Environment.from_grid_mask(
            np.ones((2, 2), dtype=bool), grid_edges=(np.arange(3.0),) * 2
        )
    )
    geometry, published = env._geometry, env._binning
    kernel = env.compute_kernel(1.0)
    cache, n_cached = env._cache, len(env._cache)
    layout = _source_layout(env)
    monkeypatch.setattr(layout, "bin_sizes", lambda: np.full(env.n_bins, np.nan))
    with pytest.raises(ValueError, match="sizes"):
        layout.binning()
    with pytest.raises(ValueError, match="sizes"):
        publish_environment(layout=layout, layout_type_used=layout._layout_type_tag)
    assert env._geometry is geometry
    assert env._binning is published
    assert env._cache is cache and len(cache) == n_cached
    np.testing.assert_array_equal(env.compute_kernel(1.0), kernel)


def test_faces_reject_changed_pair_order() -> None:
    """Native recipes retain the supplied authoritative pair order."""
    env = built_from(
        lambda: Environment.from_grid_mask(
            np.ones((2, 2), dtype=bool), grid_edges=(np.arange(3.0),) * 2
        )
    )
    reordered = nx.Graph()
    reordered.add_nodes_from(reversed(list(env.connectivity.nodes)))
    reordered.add_edges_from(env.connectivity.edges(data=True))
    assert set(map(frozenset, reordered.edges)) == set(
        map(frozenset, env.connectivity.edges)
    )
    assert list(reordered.edges) != list(env.connectivity.edges)
    layout = _source_layout(env)
    layout.connectivity = reordered
    binning = layout.binning()
    faces = binning.faces.build()
    np.testing.assert_array_equal(faces.edges, binning.bins.pairs)
    expected = face_geometry(env._cache, env._binning)
    by_pair = {frozenset(pair): i for i, pair in enumerate(expected.edges)}
    for i, pair in enumerate(faces.edges):
        assert faces.A[i] == expected.A[by_pair[frozenset(pair)]]


def test_publication_rejects_graph_nodes_without_bins():
    env = Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool), grid_edges=(np.arange(3.0),) * 2
    )
    graph = env.connectivity.copy()
    graph.add_node(4, pos=(10.0, 10.0))
    with pytest.raises(ValueError, match=r"nodes|bins"):
        publish_binning(
            env._binning,
            connectivity=graph,
            dimension_ranges=env.dimension_ranges,
            layout_type=env.layout_type,
            category=env._provenance.category,
            build_params={},
            is_linearized_track=False,
        )


@pytest.mark.nwb
def test_non_grid_nwb_record_retains_faces() -> None:
    """A saved non-grid binning supports smoothing without its original layout."""
    pynwb = pytest.importorskip("pynwb")
    from neurospatial.io.nwb import read_environment, write_environment

    env = Environment.from_layout(
        create_layout(
            "Hexagonal", hexagon_width=4.0, dimension_ranges=[(0, 12), (0, 12)]
        )
    )
    nwb = pynwb.NWBFile(
        "geometry test", "binning-refusal", datetime(2020, 1, 1, tzinfo=timezone.utc)
    )
    write_environment(nwb, env)
    restored = read_environment(nwb)
    assert restored._binning.faces.available
    assert restored._binning.rectilinear is None
    assert restored._binning.linearization is None
    field = np.arange(env.n_bins, dtype=float)
    np.testing.assert_array_equal(
        restored.smooth(field, bandwidth=1.0), env.smooth(field, bandwidth=1.0)
    )


def test_require_rectilinear_message_unchanged() -> None:
    """Linear interpolation still explains how a hexagonal user can proceed."""
    env = Environment.from_layout(
        create_layout(
            "Hexagonal", hexagon_width=4.0, dimension_ranges=[(0, 12), (0, 12)]
        )
    )
    with pytest.raises(NotImplementedError) as exc:
        env.interpolate(np.ones(env.n_bins), env.bin_centers[:1], mode="linear")
    assert str(exc.value) == (
        "interpolate(mode='linear') needs a grid of axis-aligned rectangular cells; this environment's Hexagonal layout does not have one.\n"
        "Why: it reads the cell edges along each axis, which grids built by from_samples, from_grid_mask, from_pixel_mask and from_polygon provide.\n"
        "Fix: use mode='nearest', which works on every layout"
    )
