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

from neurospatial import Environment
from neurospatial._exceptions import EnvironmentNotFittedError
from neurospatial.environment.polar import EgocentricPolarEnvironment
from neurospatial.layout._binning import LayoutFaces, binning_from_layout
from neurospatial.layout._capabilities import (
    RectilinearGridProvider,
    TrackLinearizationProvider,
)
from neurospatial.layout._diffusion import (
    FaceGeometry,
    FiniteVolumeGeometryProvider,
)
from neurospatial.layout.engines.regular_grid import RegularGridLayout

_BuiltEnvironment = Environment | EgocentricPolarEnvironment


def _source_layout(env: _BuiltEnvironment):
    """Build an independent native oracle for the published snapshot."""
    return type(env)._create_layout(env.layout_type, env.layout_parameters)


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
    np.testing.assert_array_equal(bins.sizes, env.layout.bin_sizes())
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
    assert (binning.rectilinear is not None) == isinstance(
        layout, RectilinearGridProvider
    )
    assert (binning.linearization is not None) == isinstance(
        layout, TrackLinearizationProvider
    )
    assert binning.faces.available == isinstance(layout, FiniteVolumeGeometryProvider)
    if not hasattr(layout, "bin_shapes"):
        assert binning.shapes is None
    if not hasattr(layout, "outline"):
        assert binning.outline is None
    assert binning.kind == env.layout._layout_type_tag
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
    other = binning_from_layout(env.layout, env.connectivity)
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
        env.layout.point_to_bin_index, points
    )
    actual, actual_warnings = _lookup_with_warnings(env._binning.locator.locate, points)
    np.testing.assert_array_equal(actual, expected)
    assert actual_warnings == expected_warnings
    if isinstance(env, Environment):
        if env.arena.domain_known:
            inside = env.arena.contains(points)
            expected = expected.copy()
            expected[~inside] = -1
            _, expected_warnings = _lookup_with_warnings(
                env._binning.locator.locate, points[inside]
            )
        indices, bin_warnings = _lookup_with_warnings(env.bin_at, points)
        contains, contains_warnings = _lookup_with_warnings(env.contains, points)
        np.testing.assert_array_equal(indices, expected)
        np.testing.assert_array_equal(contains, expected != -1)
        assert bin_warnings == contains_warnings == expected_warnings
    else:
        assert not hasattr(env, "bin_at")
        assert not hasattr(env, "contains")
    np.testing.assert_array_equal(points, untouched)
    assert points.flags.writeable
    assert actual.dtype == np.intp
    np.testing.assert_array_equal(actual[-2:], [-1, -1])
    for query in (points[:1], np.empty((0, env.n_dims))):
        np.testing.assert_array_equal(
            env._binning.locator.locate(query), env.layout.point_to_bin_index(query)
        )


def test_locator_preserves_dimension_behavior(
    built_environment: _BuiltEnvironment,
) -> None:
    """The adapter preserves each layout's existing dimension validation."""
    env = built_environment
    query = np.ones((2, env.n_dims + 1))
    try:
        expected = env.layout.point_to_bin_index(query)
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
    graph, sizes = _source_layout(env).finite_volume_geometry()
    expected = FaceGeometry.from_graph(graph, sizes)
    faces = env._face_geometry
    assert faces is env._face_geometry
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
    assert env._face_geometry is not faces
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


def test_legacy_environment_pickle_rebuilds_publication(
    built_environment: _BuiltEnvironment,
) -> None:
    """Saved environments predating the snapshot rebuild it and drop stale caches."""
    legacy = copy.deepcopy(built_environment)
    field = np.arange(legacy.n_bins, dtype=float)
    expected_sizes = legacy.bin_sizes.copy()
    expected_smooth = legacy.smooth(field, bandwidth=1.0)
    expected_bins = legacy.layout.point_to_bin_index(legacy.bin_centers)
    legacy.compute_kernel(1.0)
    _ = legacy._face_geometry, legacy._diffusion_geometry
    assert legacy._kernel_cache
    # Old pickles stored a cached property here, rather than a published size array.
    vars(legacy)["bin_sizes"] = np.full(legacy.n_bins, -1.0)
    version = legacy._state_version
    vars(legacy._geometry).pop("binning")

    restored = pickle.loads(pickle.dumps(legacy))

    assert restored._geometry.binning is restored._binning
    assert restored._state_version == version + 1
    assert not restored._kernel_cache
    assert restored._kdtree_cache is None
    assert "bin_sizes" not in vars(restored)
    assert "_versioned_cache___face_geometry" not in vars(restored)
    assert "_versioned_cache___diffusion_geometry" not in vars(restored)
    np.testing.assert_array_equal(restored.bin_sizes, expected_sizes)
    assert not restored.bin_sizes.flags.writeable
    if isinstance(restored, Environment):
        np.testing.assert_array_equal(
            restored.bin_at(restored.bin_centers), expected_bins
        )
    else:
        np.testing.assert_array_equal(
            restored._binning.locator.locate(restored.bin_centers), expected_bins
        )
    np.testing.assert_allclose(restored.smooth(field, bandwidth=1.0), expected_smooth)


def test_legacy_unfitted_pickle_restores_an_unpublished_placeholder() -> None:
    """An unbuilt saved environment must not attempt to publish missing geometry."""
    legacy = Environment(layout=RegularGridLayout())
    assert not legacy._is_fitted
    vars(legacy._geometry).pop("binning")

    restored = pickle.loads(pickle.dumps(legacy))

    assert not restored._is_fitted
    assert restored._geometry.binning is None
    assert restored._state_version == legacy._state_version
    with pytest.raises(EnvironmentNotFittedError):
        restored.bin_at(np.zeros((1, 2)))


def test_restored_geometry_has_independent_binning_and_cached_faces() -> None:
    """A restored graph has fresh state without modifying an existing value."""
    mask = np.ones((3, 3), dtype=bool)
    edges = (np.arange(4.0),) * 2
    env = Environment.from_grid_mask(mask, grid_edges=edges)
    replacement = Environment.from_grid_mask(
        mask, grid_edges=edges, connect_diagonal_neighbors=False
    )
    old_binning, old_faces, version = (
        env._binning,
        env._face_geometry,
        env._state_version,
    )
    restored = Environment.from_dict(replacement.to_dict())
    assert restored._binning is not old_binning
    assert restored._face_geometry is not old_faces
    assert env._binning is old_binning
    assert env._face_geometry is old_faces
    assert env._state_version == version
    np.testing.assert_array_equal(
        restored._binning.bins.pairs, list(replacement.connectivity.edges())
    )
    assert len(restored._binning.bins.pairs) < len(old_binning.bins.pairs)


def test_invalid_binning_is_not_published(monkeypatch) -> None:
    """An invalid measure must leave the previous geometry and caches intact."""
    env = Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool), grid_edges=(np.arange(3.0),) * 2
    )
    geometry, published, version = env._geometry, env._binning, env._state_version
    kernel = env.compute_kernel(1.0)
    cached_kernel = env._kernel_cache[(1.0, "density")]
    layout = _source_layout(env)
    monkeypatch.setattr(layout, "bin_sizes", lambda: np.full(env.n_bins, np.nan))
    with pytest.raises(ValueError, match="sizes"):
        binning_from_layout(layout, layout.connectivity)
    with pytest.raises(TypeError, match=r"owned immutable snapshot|Migrate custom"):
        Environment(layout=layout, layout_type_used=layout._layout_type_tag)
    assert env._geometry is geometry
    assert env._binning is published
    assert env._state_version == version
    assert env._kernel_cache[(1.0, "density")] is cached_kernel
    np.testing.assert_array_equal(env.compute_kernel(1.0), kernel)


def test_faces_reject_changed_pair_order() -> None:
    """Equal undirected edge sets cannot silently reorder a gradient's values."""
    env = Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool), grid_edges=(np.arange(3.0),) * 2
    )
    reordered = nx.Graph()
    reordered.add_nodes_from(reversed(list(env.connectivity.nodes)))
    reordered.add_edges_from(env.connectivity.edges(data=True))
    assert set(map(frozenset, reordered.edges)) == set(
        map(frozenset, env.connectivity.edges)
    )
    assert list(reordered.edges) != list(env.connectivity.edges)
    binning = binning_from_layout(_source_layout(env), reordered)
    with pytest.raises(ValueError, match=r"out of sync with env\.connectivity"):
        binning.faces.build()


def test_faces_reject_isolated_graph_nodes_without_centers(monkeypatch) -> None:
    """Face validation counts spatial bins even when graph edges still align."""
    original = Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool), grid_edges=(np.arange(3.0),) * 2
    )
    layout = _source_layout(original)
    graph, _ = layout.finite_volume_geometry()
    graph.add_node(
        4,
        pos=(10.0, 10.0),
        source_grid_flat_index=4,
        original_grid_nd_index=(2, 0),
    )
    layout.connectivity = graph
    monkeypatch.setattr(layout, "finite_volume_geometry", lambda: (graph, np.ones(5)))
    faces = LayoutFaces(layout, graph)
    assert layout.bin_centers.shape[0] == 4
    assert graph.number_of_nodes() == 5
    with pytest.raises(ValueError, match="does not match the environment's 4 bins"):
        faces.build()


@pytest.mark.nwb
def test_non_grid_nwb_record_retains_faces() -> None:
    """A saved non-grid binning supports smoothing without its original layout."""
    pynwb = pytest.importorskip("pynwb")
    from neurospatial.io.nwb import read_environment, write_environment

    env = Environment.from_layout(
        "Hexagonal", {"hexagon_width": 4.0, "dimension_ranges": [(0, 12), (0, 12)]}
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
        "Hexagonal", {"hexagon_width": 4.0, "dimension_ranges": [(0, 12), (0, 12)]}
    )
    with pytest.raises(NotImplementedError) as exc:
        env.interpolate(np.ones(env.n_bins), env.bin_centers[:1], mode="linear")
    assert str(exc.value) == (
        "interpolate(mode='linear') needs a grid of axis-aligned rectangular cells; this environment's Hexagonal layout does not have one.\n"
        "Why: it reads the cell edges along each axis, which grids built by from_samples, from_grid_mask, from_pixel_mask and from_polygon provide.\n"
        "Fix: use mode='nearest', which works on every layout"
    )
