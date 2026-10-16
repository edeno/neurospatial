"""Explicit bins retain identity, edge orientation and declared capabilities."""

from __future__ import annotations

import copy
import pickle
from dataclasses import replace

import networkx as nx
import numpy as np
import pytest
from shapely.geometry import LineString, box

from neurospatial import Environment
from neurospatial.arena import OpenArena
from neurospatial.binning import (
    Binning,
    FaceGeometry,
    FaceNormals,
    KnownFaces,
    MissingFaces,
    NearestBinLocator,
    PolygonCells,
    RectilinearGrid,
    RectilinearLocator,
    SpatialBins,
)
from neurospatial.geometry._components import Outline
from neurospatial.layout import create_layout
from neurospatial.ops.calculus import divergence, gradient
from neurospatial.regions import Regions


def _nearest(*, faces=None, kind="explicit"):
    centers = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 3.0]])
    bins = SpatialBins.from_arrays(
        centers, [1.0, 2.0, 3.0], 2, [[1, 2], [0, 1]], [7.0, 5.0]
    )
    return Binning(
        kind,
        bins,
        NearestBinLocator(centers, max_distance=0.25),
        MissingFaces("No finite-volume faces were supplied.")
        if faces is None
        else faces,
    )


def _rectilinear():
    centers = np.array([[0.5, 0.5], [0.5, 1.5], [1.5, 0.5], [1.5, 1.5]])
    pairs = np.array([[3, 1], [2, 3], [0, 2]], dtype=np.intp)
    bins = SpatialBins.from_arrays(centers, np.ones(4), 2, pairs, [2.0, 3.0, 4.0])
    edges = (np.arange(3.0),) * 2
    mask = np.ones((2, 2), dtype=bool)
    return Binning(
        "explicit_grid",
        bins,
        RectilinearLocator(edges, (2, 2), mask, None),
        KnownFaces(FaceGeometry(pairs, np.ones(3), np.ones(3), np.ones(4))),
        rectilinear=RectilinearGrid(edges, (2, 2), mask),
    )


def test_nearest_only_bins_keep_explicit_lookup_and_refuse_cell_capabilities():
    source = _nearest()
    env = Environment.from_bins(source, name="recorded", units="cm", frame="world")
    assert (env.name, env.units, env.frame) == ("recorded", "cm", "world")
    np.testing.assert_array_equal(env.bin_centers, source.bins.centers)
    np.testing.assert_array_equal(env.bin_sizes, [1, 2, 3])
    np.testing.assert_array_equal(env.edge_pairs, [[1, 2], [0, 1]])
    np.testing.assert_array_equal(
        env.bin_at([[0, 0], [2, 3], [100, 100], [np.nan, 0]]), [0, 2, -1, -1]
    )
    assert not env.arena.domain_known
    with pytest.raises(NotImplementedError, match=r"faces|finite-volume"):
        env.compute_kernel(1.0)
    with pytest.raises(ValueError, match=r"cell|polygon|clip"):
        env.clip(box(-1, -1, 3, 4))
    with pytest.raises(ValueError, match=r"cell|linear"):
        env.occupancy(
            np.array([0.0, 1.0]),
            np.array([[0.0, 0.0], [2.0, 3.0]]),
            time_allocation="linear",
            max_gap=None,
        )


@pytest.mark.parametrize(
    "restore",
    [
        lambda x: x,
        copy.copy,
        copy.deepcopy,
        lambda x: pickle.loads(pickle.dumps(x)),
        lambda x: Environment.from_dict(x.to_dict()),
    ],
)
def test_arbitrary_pair_order_and_orientation_define_calculus(restore):
    source = _rectilinear()
    env = restore(Environment.from_bins(source))
    np.testing.assert_array_equal(env.edge_pairs, source.bins.pairs)
    assert list(env.connectivity.edges()) != [tuple(pair) for pair in env.edge_pairs]
    assert nx.shortest_path_length(env.connectivity, 0, 1, weight="distance") == 9.0
    np.testing.assert_array_equal(
        gradient(env, np.array([0.0, 1.0, 3.0, 7.0])), [-6, 4, 3]
    )
    np.testing.assert_array_equal(
        divergence(env, np.array([2.0, 5.0, 11.0])), [11, -2, -6, -3]
    )
    attributes = env.get_edge_attributes()
    np.testing.assert_array_equal(attributes[["source", "target"]], env.edge_pairs)
    expected_vectors = (
        env.bin_centers[env.edge_pairs[:, 1]] - env.bin_centers[env.edge_pairs[:, 0]]
    )
    np.testing.assert_array_equal(
        attributes[["vector_dim0", "vector_dim1"]], expected_vectors
    )
    angles = attributes["angle_2d"].to_numpy()
    unit_vectors = expected_vectors / np.linalg.norm(expected_vectors, axis=1)[:, None]
    # +/-pi describe the same direction; signed zero can choose either branch.
    np.testing.assert_allclose(
        np.column_stack((np.cos(angles), np.sin(angles))),
        unit_vectors,
        rtol=0,
        atol=4 * np.finfo(float).eps,
    )
    for u, v, attrs in env.connectivity.edges(data=True):
        np.testing.assert_array_equal(
            attrs["vector"], env.bin_centers[v] - env.bin_centers[u]
        )


@pytest.mark.parametrize("region", [box(0, 0, 2, 2), box(0, 0, 2, 1)])
def test_clip_preserves_declared_navigation_paths_at_unchanged_endpoints(region):
    env = Environment.from_bins(_rectilinear())
    child = env.clip(region)
    parent_bins = child.clip_mapping.parent_bin_indices
    for u, v, attrs in child.connectivity.edges(data=True):
        assert (
            attrs["distance"]
            == env.connectivity.edges[parent_bins[u], parent_bins[v]]["distance"]
        )
    restored = Environment.from_dict(child.to_dict())
    np.testing.assert_array_equal(
        restored._binning.bins.pair_distances, child._binning.bins.pair_distances
    )


def test_clip_refuses_to_guess_navigation_paths_after_moving_endpoints():
    env = Environment.from_bins(_rectilinear())
    with pytest.raises(
        ValueError, match=r"navigation path.*endpoint|endpoint.*navigation path"
    ):
        env.clip(box(0.25, 0, 2, 2))
    np.testing.assert_array_equal(env._binning.bins.pair_distances, [2.0, 3.0, 4.0])


def test_clip_can_remove_declared_paths_without_guessing_new_lengths():
    env = Environment.from_bins(_rectilinear())
    child = env.clip(box(0.3, 0, 0.9, 0.75))
    assert child.n_bins == 1
    assert child.connectivity.number_of_edges() == 0
    np.testing.assert_allclose(child.bin_centers, [[0.6, 0.375]])


def test_clip_updates_euclidean_edges_without_changing_unaffected_declared_paths():
    native = Environment.from_grid_mask(
        np.ones((3, 1), dtype=bool),
        grid_edges=(np.arange(4.0), np.arange(2.0)),
    )
    bins = replace(native._binning.bins, pair_distances=np.array([3.0, 1.0]))
    env = Environment.from_bins(
        replace(
            native._binning, bins=bins, faces=KnownFaces(native._binning.faces.build())
        )
    )
    child = env.clip(box(0, 0, 2.8, 1))
    assert child.connectivity.edges[0, 1]["distance"] == 3.0
    assert child.connectivity.edges[1, 2]["distance"] == pytest.approx(0.9)
    # Finite-volume distances still describe center-to-face flux geometry.
    np.testing.assert_allclose(child._binning.faces.build().distance, [1.0, 0.9])


def test_ingress_and_exports_do_not_alias_explicit_bin_records():
    source = _rectilinear()
    env = Environment.from_bins(source)
    expected = env.bin_centers.copy()
    source.bins.centers.setflags(write=True)
    source.bins.centers[:] = -99
    source.locator.edges[0].setflags(write=True)
    source.locator.edges[0][:] = -99
    source.faces.faces.A.setflags(write=True)
    source.faces.faces.A[:] = 999
    np.testing.assert_array_equal(env.bin_centers, expected)
    np.testing.assert_array_equal(env.bin_at(expected), np.arange(4))
    np.testing.assert_array_equal(gradient(env, expected[:, 0]), [-1, 0, 1])
    pairs = env.edge_pairs
    assert not pairs.flags.writeable
    pairs.shape = (6,)
    np.testing.assert_array_equal(env.edge_pairs, [[3, 1], [2, 3], [0, 2]])


@pytest.mark.parametrize("transform_kind", ["translation", "scale"])
def test_transform_preserves_pair_alignment_with_unequal_edge_lengths(transform_kind):
    from neurospatial.ops.transforms import scale_2d, translate

    source = _nearest()
    bins = source.bins
    bins = SpatialBins.from_arrays(
        bins.centers, bins.sizes, bins.measure_dim, bins.pairs, [3.0, 2.0]
    )
    env = Environment.from_bins(replace(source, bins=bins))
    scale = 1.0 if transform_kind == "translation" else 2.0
    transform = translate(1.0, 2.0) if scale == 1.0 else scale_2d(scale)
    derived = env.apply_transform(transform)
    np.testing.assert_array_equal(derived.edge_pairs, env.edge_pairs)
    for pair, distance in zip(
        derived.edge_pairs, [3.0 * scale, 2.0 * scale], strict=True
    ):
        assert derived.connectivity.edges[tuple(pair)]["distance"] == distance


@pytest.mark.parametrize("source_factory", [_nearest, _rectilinear])
@pytest.mark.parametrize(
    "matrix,scale",
    [
        ([[1, 0, 1.25], [0, 1, -2.5], [0, 0, 1]], 1.0),
        ([[0, -1, 0], [1, 0, 0], [0, 0, 1]], 1.0),
        ([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], 1.0),
        ([[2, 0, 0], [0, 2, 0], [0, 0, 1]], 2.0),
        ([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]], 0.5),
    ],
    ids=["translation", "rotation", "reflection", "expansion", "contraction"],
)
def test_similarity_preserves_declared_navigation_weights(
    source_factory, matrix, scale
):
    from neurospatial.ops.transforms import Affine2D

    source = source_factory()
    env = Environment.from_bins(source)
    derived = env.apply_transform(Affine2D(np.asarray(matrix, dtype=float)))
    np.testing.assert_array_equal(derived.edge_pairs, source.bins.pairs)
    expected = source.bins.pair_distances * scale
    for pair, distance in zip(derived.edge_pairs, expected, strict=True):
        assert derived.connectivity.edges[tuple(pair)]["distance"] == distance
    assert nx.shortest_path_length(derived.connectivity, 0, 1, weight="distance") == (
        nx.shortest_path_length(env.connectivity, 0, 1, weight="distance") * scale
    )
    # These are navigation weights; finite-volume distances have their own units.
    if source_factory is _rectilinear:
        np.testing.assert_array_equal(
            derived._binning.faces.build().distance,
            source.faces.build().distance * scale,
        )
    np.testing.assert_array_equal(
        env._binning.bins.pair_distances, source.bins.pair_distances
    )


@pytest.mark.parametrize("source_factory", [_nearest, _rectilinear])
@pytest.mark.parametrize(
    "matrix",
    [
        [[1, 0.25, 0], [0, 1, 0], [0, 0, 1]],
        [[2, 0, 0], [0, 1, 0], [0, 0, 1]],
    ],
    ids=["shear", "unequal_scaling"],
)
def test_non_similarity_refuses_unknown_navigation_path_geometry(
    source_factory, matrix
):
    from neurospatial.ops.transforms import Affine2D

    env = Environment.from_bins(source_factory())
    with pytest.raises(ValueError, match=r"navigation|path|weight"):
        env.apply_transform(Affine2D(np.asarray(matrix, dtype=float)))


@pytest.mark.parametrize(
    "matrix",
    [
        [[1, 0.25, 0], [0, 1, 0], [0, 0, 1]],
        [[2, 0, 0], [0, 1, 0], [0, 0, 1]],
    ],
    ids=["shear", "unequal_scaling"],
)
def test_non_similarity_keeps_euclidean_navigation_supported(matrix):
    from neurospatial.ops.transforms import Affine2D

    source = _nearest()
    bins = source.bins
    euclidean = np.linalg.norm(
        bins.centers[bins.pairs[:, 1]] - bins.centers[bins.pairs[:, 0]], axis=1
    )
    env = Environment.from_bins(
        replace(source, bins=replace(bins, pair_distances=euclidean))
    )
    derived = env.apply_transform(Affine2D(np.asarray(matrix, dtype=float)))
    centers = derived.bin_centers
    expected = np.linalg.norm(
        centers[derived.edge_pairs[:, 1]] - centers[derived.edge_pairs[:, 0]], axis=1
    )
    for pair, distance in zip(derived.edge_pairs, expected, strict=True):
        assert derived.connectivity.edges[tuple(pair)]["distance"] == distance


def test_euclidean_translation_keeps_existing_center_difference_rounding():
    from neurospatial.ops.transforms import translate

    source = _nearest()
    centers = np.array([[0.1, 0.0], [0.3, 0.0], [0.3, 0.7]])
    pairs = source.bins.pairs
    distances = np.linalg.norm(centers[pairs[:, 1]] - centers[pairs[:, 0]], axis=1)
    source = replace(
        source,
        bins=replace(source.bins, centers=centers, pair_distances=distances),
        locator=NearestBinLocator(centers, max_distance=0.25),
    )
    derived = Environment.from_bins(source).apply_transform(translate(1, 2))
    mapped = derived.bin_centers
    expected = np.linalg.norm(mapped[pairs[:, 1]] - mapped[pairs[:, 0]], axis=1)
    assert not np.array_equal(expected, distances)
    for pair, distance in zip(pairs, expected, strict=True):
        assert derived.connectivity.edges[tuple(pair)]["distance"] == distance


def test_declared_zero_weight_survives_restoration_and_nested_similarities():
    from neurospatial.ops.transforms import scale_2d, translate

    source = _rectilinear()
    weights = np.array([2.0, 1.0, 0.0])
    source = replace(source, bins=replace(source.bins, pair_distances=weights))
    env = Environment.from_dict(Environment.from_bins(source).to_dict())
    derived = env.apply_transform(translate(1, 2))
    derived = pickle.loads(pickle.dumps(derived)).apply_transform(scale_2d(2))
    for pair, distance in zip(source.bins.pairs, weights * 2, strict=True):
        assert derived.connectivity.edges[tuple(pair)]["distance"] == distance


def test_positive_navigation_weight_cannot_underflow_to_zero():
    from neurospatial.ops.transforms import scale_2d

    centers = np.array([[0.0, 0.0], [1.0, 0.0]])
    bins = SpatialBins.from_arrays(centers, [1.0, 1.0], 2, [[0, 1]], [1e-200])
    env = Environment.from_bins(
        Binning(
            "weighted",
            bins,
            NearestBinLocator(centers, max_distance=0.25),
            MissingFaces("No finite-volume faces were supplied."),
        )
    )
    with pytest.raises(ValueError, match=r"navigation|distance|range|representable"):
        env.apply_transform(scale_2d(1e-150))


def test_regions_and_arena_use_the_normal_ownership_contract():
    regions = Regions()
    regions.add("outside", point=(100, 100))
    env = Environment.from_bins(
        _rectilinear(), regions=regions, arena=OpenArena(box(0, 0, 2, 2))
    )
    regions.clear()
    assert list(env.regions) == ["outside"]
    with pytest.raises(TypeError, match="read-only"):
        env.regions.clear()
    assert env.arena.domain_known


@pytest.mark.parametrize("field", ["centers", "sizes", "pairs", "pair_distances"])
def test_mutated_record_arrays_are_revalidated_at_ingress(field):
    source = _nearest()
    value = getattr(source.bins, field)
    value.setflags(write=True)
    value.flat[0] = np.nan if field == "centers" else -1
    with pytest.raises(ValueError, match=field):
        Environment.from_bins(source)


def test_missing_faces_is_explicit_and_delegated_faces_are_refused():
    source = _nearest()
    native = Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool), grid_edges=(np.arange(3.0),) * 2
    )
    for faces in (None, object(), native._binning.faces):
        with pytest.raises(TypeError, match=r"KnownFaces|MissingFaces"):
            Environment.from_bins(replace(source, faces=faces))


@pytest.mark.parametrize("field,value", [("A", -1), ("distance", 0), ("volumes", -1)])
def test_mutated_known_faces_are_revalidated_at_ingress(field, value):
    source = _rectilinear()
    array = getattr(source.faces.faces, field)
    array.setflags(write=True)
    array[0] = value
    with pytest.raises(ValueError, match=r"face|volume"):
        Environment.from_bins(source)


def test_shared_geometry_mismatch_is_not_hidden_by_public_ingress():
    source = _rectilinear()
    bad_edges = (np.array([0.0, 0.95, 2.0]), np.arange(3.0))
    changed = replace(
        source,
        rectilinear=RectilinearGrid(bad_edges, (2, 2), np.ones((2, 2), dtype=bool)),
    )
    with pytest.raises(ValueError, match=r"rectilinear edges"):
        Environment.from_bins(changed)


def test_descriptive_kind_cannot_grant_regular_grid_rebinning():
    env = Environment.from_bins(_nearest(kind="RegularGrid"))
    with pytest.raises(NotImplementedError, match="uniform rectilinear grid"):
        env.rebin(1)


def test_nonuniform_cells_are_supported_without_regular_grid_rebinning():
    edges = (np.array([0.0, 1.0, 3.0]), np.array([0.0, 1.0]))
    centers = np.array([[0.5, 0.5], [2.0, 0.5]])
    mask = np.ones((2, 1), dtype=bool)
    bins = SpatialBins.from_arrays(centers, [1, 2], 2, [[0, 1]], [1.5])
    source = Binning(
        "RegularGrid",
        bins,
        RectilinearLocator(edges, (2, 1), mask, None),
        MissingFaces("No finite-volume faces supplied."),
        rectilinear=RectilinearGrid(edges, (2, 1), mask),
    )
    env = Environment.from_bins(source)
    np.testing.assert_array_equal(env.bin_sizes, [1, 2])
    np.testing.assert_array_equal(env.bin_at([[0.5, 0.5], [2.5, 0.5]]), [0, 1])
    with pytest.raises(NotImplementedError, match="uniform rectilinear grid"):
        env.rebin(1)


def test_nearly_uniform_variable_measures_are_not_reinterpreted_as_native_cells():
    edges = (np.array([0.0, 1.0, 2.00000005]), np.array([0.0, 1.0]))
    centers = np.array([[0.5, 0.5], [0.5 * (edges[0][1] + edges[0][2]), 0.5]])
    mask = np.ones((2, 1), dtype=bool)
    bins = SpatialBins.from_arrays(
        centers, np.diff(edges[0]), 2, [[0, 1]], [centers[1, 0] - centers[0, 0]]
    )
    source = Binning(
        "RegularGrid",
        bins,
        RectilinearLocator(edges, (2, 1), mask, None),
        MissingFaces("No finite-volume faces supplied."),
        rectilinear=RectilinearGrid(edges, (2, 1), mask),
    )
    env = Environment.from_bins(source)
    np.testing.assert_array_equal(env.bin_sizes, np.diff(edges[0]))
    with pytest.raises(NotImplementedError, match=r"uniform|measure"):
        env.rebin(1)


def test_rebin_uniformity_guard_preserves_native_large_offset_rounding():
    source = Environment.from_layout(
        create_layout(
            "RegularGrid",
            bin_size=0.1,
            dimension_ranges=[(1e9, 1e9 + 1.0), (0.0, 1.0)],
            infer_active_bins=False,
        )
    )
    restored = source.rebin(1)
    np.testing.assert_array_equal(restored.bin_centers, source.bin_centers)
    np.testing.assert_array_equal(restored.bin_sizes, source.bin_sizes)


def test_empty_bins_remain_a_valid_explicit_empty_environment():
    centers = np.empty((0, 2))
    bins = SpatialBins.from_arrays(centers, [], 2, np.empty((0, 2), dtype=int), [])
    env = Environment.from_bins(
        Binning("empty", bins, NearestBinLocator(centers), MissingFaces("No bins."))
    )
    assert env.n_bins == 0
    assert env.n_dims == 2
    np.testing.assert_array_equal(env.bin_at([[1, 2], [np.nan, 0]]), [-1, -1])
    assert env.boundary_bins.size == 0


def test_custom_callbacks_are_refused_before_copy_or_lookup():
    class CustomLocator:
        def __deepcopy__(self, memo):
            pytest.fail("custom callback was copied")

        def locate(self, points):
            pytest.fail("custom callback was executed")

    with pytest.raises(TypeError, match=r"owned.*snapshot|supported"):
        Environment.from_bins(replace(_nearest(), locator=CustomLocator()))


@pytest.mark.parametrize("slot", ["mapping", "cells", "clip_mapping"])
def test_clipped_records_refuse_nested_custom_providers_before_copy(slot):
    parent = Environment.from_grid_mask(
        np.ones((1, 2), dtype=bool),
        grid_edges=(np.array([0.0, 1.0]), np.arange(3.0)),
    )
    source = parent.clip(box(0.1, 0.1, 0.9, 1.9))._binning
    mapping = source.locator.mapping
    polygons = source.locator.cells.polygons

    class ExternalMapping:
        n_parent_bins = mapping.n_parent_bins

        @property
        def parent_bin_indices(self):
            return mapping.parent_bin_indices

        def __deepcopy__(self, memo):
            pytest.fail("nested mapping callback was copied")

    class ExternalCells:
        @property
        def polygons(self):
            return polygons

        def __deepcopy__(self, memo):
            pytest.fail("nested cell callback was copied")

    if slot == "mapping":
        source = replace(
            source, locator=replace(source.locator, mapping=ExternalMapping())
        )
    elif slot == "cells":
        source = replace(source, locator=replace(source.locator, cells=ExternalCells()))
    else:
        source = replace(source, clip_mapping=ExternalMapping())
    with pytest.raises(TypeError, match=r"owned.*snapshot|supported"):
        Environment.from_bins(source)


def test_unrelated_polygon_shapes_do_not_grant_exact_cells():
    source = _nearest()
    shapes = PolygonCells(
        (box(-0.5, -0.5, 0.5, 0.5), box(1.5, -0.5, 2.5, 0.5), box(1.5, 2.5, 2.5, 3.5))
    )
    with pytest.raises(ValueError, match=r"polygon.*locator|locator.*polygon"):
        Environment.from_bins(replace(source, shapes=shapes))


@pytest.mark.parametrize(
    "built_environment",
    [
        "grid8",
        "grid4",
        "polygon",
        "mask",
        "pixels",
        "hex",
        "track",
        "mesh",
        "mesh_boundary",
    ],
    indirect=True,
)
def test_explicit_native_data_retains_lookup_measures_boundaries_and_faces(
    built_environment,
):
    source = built_environment
    exported = source._binning
    explicit = replace(exported, faces=KnownFaces(exported.faces.build()))
    env = Environment.from_bins(explicit, arena=source.arena)
    np.testing.assert_array_equal(env.bin_centers, source.bin_centers)
    np.testing.assert_array_equal(env.bin_sizes, source.bin_sizes)
    np.testing.assert_array_equal(env.edge_pairs, source.edge_pairs)
    np.testing.assert_array_equal(
        env.bin_at(source.bin_centers), source.bin_at(source.bin_centers)
    )
    np.testing.assert_array_equal(env.boundary_bins, source.boundary_bins)
    np.testing.assert_array_equal(
        env.get_differential_operator().toarray(),
        source.get_differential_operator().toarray(),
    )


def test_networkx_export_is_editable_and_nested_attributes_are_independent():
    env = Environment.from_bins(_rectilinear())
    exported = env.to_networkx()
    assert not nx.is_frozen(exported)
    exported.graph["caller"] = {"values": [1, 2]}
    exported.nodes[0]["pos"] = (99.0, 99.0)
    exported.remove_node(1)
    assert env.n_bins == 4
    assert "caller" not in env.connectivity.graph
    np.testing.assert_array_equal(env.bin_centers[0], [0.5, 0.5])


@pytest.mark.parametrize("transport", ["dict", "file", "from_bins"])
def test_hex_data_roundtrip_preserves_fft_autocorrelation(transport, tmp_path):
    from neurospatial.encoding.grid import spatial_autocorrelation

    source = Environment.from_layout(
        create_layout(
            "Hexagonal",
            hexagon_width=2.0,
            dimension_ranges=[(0.0, 10.0)] * 2,
            infer_active_bins=False,
        )
    )
    if transport == "dict":
        restored = Environment.from_dict(source.to_dict())
    elif transport == "file":
        path = tmp_path / "hex"
        source.to_file(path)
        restored = Environment.from_file(path)
    else:
        exported = source._binning
        explicit = replace(exported, faces=KnownFaces(exported.faces.build()))
        restored = Environment.from_bins(explicit, arena=source.arena)
    rates = np.arange(source.n_bins, dtype=float)
    assert source.grid_shape == restored.grid_shape == (7, 6)
    np.testing.assert_array_equal(restored.active_mask, source.active_mask)
    np.testing.assert_array_equal(
        spatial_autocorrelation(restored, rates), spatial_autocorrelation(source, rates)
    )


@pytest.mark.parametrize("derivation", ["subset", "transform", "clip", "chain"])
def test_explicit_derived_data_preserves_retained_source_indices(derivation):
    from neurospatial.ops.transforms import Affine2D, translate

    parent = Environment.from_grid_mask(
        np.ones((3, 3), dtype=bool), grid_edges=(np.arange(4.0),) * 2
    )
    keep = np.ones(parent.n_bins, dtype=bool)
    keep[4] = False
    if derivation == "subset":
        source = parent.subset(mask=keep)
    elif derivation == "transform":
        cosine, sine = np.cos(0.2), np.sin(0.2)
        source = parent.apply_transform(
            Affine2D(np.array([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]]))
        )
    elif derivation == "clip":
        source = parent.clip(box(0.2, 0.2, 2.8, 2.8))
    else:
        source = parent.subset(mask=keep).apply_transform(translate(1.0, 2.0))
    exported = source._binning
    explicit = replace(exported, faces=KnownFaces(exported.faces.build()))
    env = Environment.from_bins(explicit, arena=source.arena)
    np.testing.assert_array_equal(env.boundary_bins, source.boundary_bins)
    np.testing.assert_array_equal(
        env.bin_at(source.bin_centers), source.bin_at(source.bin_centers)
    )
    assert [
        env.connectivity.nodes[i]["original_grid_nd_index"] for i in range(env.n_bins)
    ] == [
        source.connectivity.nodes[i]["original_grid_nd_index"]
        for i in range(source.n_bins)
    ]


def test_explicit_pairs_crossing_walls_are_rejected_without_changing_input():
    source = _rectilinear()
    arena = OpenArena(box(0, 0, 2, 2), walls=(LineString([(1, 0), (1, 2)]),))
    expected = source.bins.pairs.copy()
    with pytest.raises(ValueError, match=r"supplied.*pairs|Supplied.*pairs"):
        Environment.from_bins(source, arena=arena)
    np.testing.assert_array_equal(source.bins.pairs, expected)


def test_explicit_outline_is_drawing_geometry_without_physical_wall_evidence():
    source = replace(_rectilinear(), outline=Outline(box(0, 0, 2, 2)))
    env = Environment.from_bins(source)
    assert env.arena.domain_known
    assert not env.arena.physical_footprint_known
    np.testing.assert_array_equal(env.bin_centers, source.bins.centers)
    asserted = Environment.from_bins(source, arena=OpenArena(box(0, 0, 2, 2)))
    assert asserted.arena.physical_footprint_known


@pytest.mark.parametrize("field", ["A", "distance"])
def test_explicit_grid_faces_must_match_cell_geometry(field):
    source = _rectilinear()
    faces = source.faces.build()
    invalid = replace(
        source, faces=KnownFaces(faces._replace(**{field: getattr(faces, field) * 7.0}))
    )
    with pytest.raises(ValueError, match=r"grid faces"):
        Environment.from_bins(invalid)


def _variable_grid_faces(n_dims):
    from itertools import combinations, product

    edges = tuple(
        np.array([0.0, float(d + 1), float(2 * d + 3)]) for d in range(n_dims)
    )
    coordinates = np.array(list(product(range(2), repeat=n_dims)), dtype=np.intp)
    centers = np.column_stack(
        [
            ((edge[:-1] + edge[1:]) / 2)[coordinates[:, d]]
            for d, edge in enumerate(edges)
        ]
    )
    widths = np.column_stack(
        [np.diff(edge)[coordinates[:, d]] for d, edge in enumerate(edges)]
    )
    sizes = np.prod(widths, axis=1)
    pairs = np.array(list(combinations(range(len(centers)), 2)), dtype=np.intp)
    areas = []
    for u, v in pairs:
        moved = np.flatnonzero(coordinates[u] != coordinates[v])
        areas.append(
            np.prod(np.delete(widths[u], moved[0])) if len(moved) == 1 else 0.0
        )
    distances = np.linalg.norm(centers[pairs[:, 1]] - centers[pairs[:, 0]], axis=1)
    bins = SpatialBins.from_arrays(centers, sizes, n_dims, pairs, distances * 3 + 1)
    mask = np.ones((2,) * n_dims, dtype=bool)
    return Binning(
        "variable_grid",
        bins,
        RectilinearLocator(edges, mask.shape, mask, None),
        KnownFaces(FaceGeometry(pairs, np.array(areas), distances, sizes)),
        rectilinear=RectilinearGrid(edges, mask.shape, mask),
    )


@pytest.mark.parametrize("n_dims", [1, 2, 3])
@pytest.mark.parametrize("history", ["native", "subset", "load"])
def test_variable_grid_physical_faces_are_separate_from_navigation_weights(
    n_dims, history
):
    source = _variable_grid_faces(n_dims)
    env = Environment.from_bins(source)
    if history == "subset":
        env = env.subset(mask=np.arange(env.n_bins) != env.n_bins - 1)
    elif history == "load":
        env = Environment.from_dict(env.to_dict())
    exported = env._binning
    np.testing.assert_allclose(env.diffuse(np.ones(env.n_bins), 0.5, mode="average"), 1)
    assert np.all(exported.bins.pair_distances > exported.faces.build().distance)


@pytest.mark.parametrize("n_dims", [1, 2, 3])
@pytest.mark.parametrize("field", ["A", "distance"])
def test_variable_grid_faces_reject_inconsistent_physical_values(n_dims, field):
    source = _variable_grid_faces(n_dims)
    faces = source.faces.build()
    wrong = getattr(faces, field).copy()
    wrong[0] *= 2.0
    with pytest.raises(ValueError, match=r"grid faces"):
        Environment.from_bins(
            replace(source, faces=KnownFaces(faces._replace(**{field: wrong})))
        )


@pytest.mark.parametrize("kind", ["missing_face", "invented_diagonal_face"])
def test_explicit_grid_cannot_omit_a_shared_face_or_invent_a_diagonal_one(kind):
    source = _variable_grid_faces(2)
    faces = source.faces.build()
    areas = faces.A.copy()
    areas[
        np.flatnonzero(areas > 0)[0]
        if kind == "missing_face"
        else np.flatnonzero(areas == 0)[0]
    ] = 0.0 if kind == "missing_face" else 1.0
    with pytest.raises(ValueError, match=r"grid faces.A"):
        Environment.from_bins(
            replace(source, faces=KnownFaces(faces._replace(A=areas)))
        )


@pytest.mark.parametrize("history", ["similarity", "axis_scale", "subset_similarity"])
def test_transformed_explicit_grid_faces_cannot_be_forged_through_wrappers(history):
    from neurospatial.ops.transforms import Affine2D

    source = _variable_grid_faces(2)
    if history == "axis_scale":
        source = replace(
            source,
            bins=replace(source.bins, pair_distances=source.faces.build().distance),
        )
    env = Environment.from_bins(source)
    if history == "axis_scale":
        matrix = np.array([[2.0, 0.0, 100.0], [0.0, 3.0, -100.0], [0.0, 0.0, 1.0]])
    else:
        angle = 0.3
        matrix = np.array(
            [
                [2 * np.cos(angle), -2 * np.sin(angle), 100.0],
                [2 * np.sin(angle), 2 * np.cos(angle), -100.0],
                [0, 0, 1],
            ]
        )
    if history == "subset_similarity":
        env = env.subset(mask=np.arange(env.n_bins) != 3)
    derived = env.apply_transform(Affine2D(matrix))
    source = derived._binning
    Environment.from_bins(source)
    faces = source.faces.build()
    with pytest.raises(ValueError, match=r"grid faces"):
        Environment.from_bins(
            replace(source, faces=KnownFaces(faces._replace(A=faces.A * 7)))
        )


@pytest.mark.parametrize("field", ["A", "distance"])
def test_stored_grid_faces_cannot_bypass_publication_geometry_checks(field):
    env = Environment.from_bins(_variable_grid_faces(2))
    payload = env.to_dict()
    key = f"binning/faces/{field}"
    payload["arrays"][key] = (np.asarray(payload["arrays"][key]) * 7.0).tolist()
    with pytest.raises(ValueError, match=r"grid faces"):
        Environment.from_dict(payload)


def test_legacy_tiny_grid_zero_faces_require_rebuilding():
    env = Environment.from_grid_mask(
        np.ones((2, 1), dtype=bool),
        grid_edges=(np.arange(3.0) * 1e-10, np.arange(2.0) * 1e-10),
    )
    source = env._binning
    faces = source.faces.build()
    with pytest.raises(ValueError, match="Older tiny-grid records"):
        Environment.from_bins(
            replace(
                source,
                faces=KnownFaces(faces._replace(A=np.zeros_like(faces.A))),
                face_normals=None,
            )
        )


@pytest.mark.parametrize("n_dims", [2, 3])
def test_native_quantized_grid_faces_remain_valid_at_large_coordinate_offsets(n_dims):
    edges = tuple(1e6 + 0.1 + np.arange(4.0) * 0.1 for _ in range(n_dims))
    env = Environment.from_grid_mask(
        np.ones((3,) * n_dims, dtype=bool), grid_edges=edges
    )
    source = env._binning
    explicit = Environment.from_bins(
        replace(source, faces=KnownFaces(source.faces.build()))
    )
    np.testing.assert_array_equal(explicit._face_geometry.A, env._face_geometry.A)
    np.testing.assert_array_equal(
        explicit._face_geometry.distance, env._face_geometry.distance
    )


def test_three_dimensional_affine_grid_faces_follow_physical_planes():
    from neurospatial.geometry._locators import TransformedLocator

    source = _variable_grid_faces(3)
    angle = 0.3
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    scales = np.array([2.0, 3.0, 4.0])
    linear = rotation @ np.diag(scales)
    matrix = np.eye(4)
    matrix[:3, :3] = linear
    matrix[:3, 3] = [100, -50, 25]
    centers = source.bins.centers @ linear.T + matrix[:3, 3]
    pairs = source.bins.pairs
    distances = np.linalg.norm(centers[pairs[:, 1]] - centers[pairs[:, 0]], axis=1)
    source_vectors = source.bins.centers[pairs[:, 1]] - source.bins.centers[pairs[:, 0]]
    axes = np.argmax(np.abs(source_vectors), axis=1)
    sizes = source.bins.sizes * np.prod(scales)
    faces = FaceGeometry(
        pairs, source.faces.build().A * np.prod(scales) / scales[axes], distances, sizes
    )
    transformed = replace(
        source,
        bins=replace(source.bins, centers=centers, sizes=sizes),
        locator=TransformedLocator(source.locator, np.linalg.inv(matrix)),
        faces=KnownFaces(faces),
        rectilinear=None,
    )
    env = Environment.from_bins(transformed)
    np.testing.assert_allclose(env.diffuse(np.ones(env.n_bins), 0.5, mode="average"), 1)
    with pytest.raises(ValueError, match=r"grid faces"):
        Environment.from_bins(
            replace(transformed, faces=KnownFaces(faces._replace(A=faces.A * 7)))
        )


@pytest.mark.parametrize("scale", [1e200, 1e-200])
def test_grid_face_validation_preserves_finite_extreme_coordinate_scales(scale):
    centers = np.array([[0.5], [1.5]]) * scale
    sizes = np.full(2, scale)
    pairs = np.array([[0, 1]], dtype=np.intp)
    edges = (np.arange(3.0) * scale,)
    mask = np.ones(2, dtype=bool)
    source = Binning(
        "finite_extreme_grid",
        SpatialBins.from_arrays(centers, sizes, 1, pairs, [scale]),
        RectilinearLocator(edges, (2,), mask, None),
        KnownFaces(FaceGeometry(pairs, np.ones(1), np.array([scale]), sizes)),
        rectilinear=RectilinearGrid(edges, (2,), mask),
    )
    env = Environment.from_bins(source)
    np.testing.assert_array_equal(env._face_geometry.distance, [scale])
    np.testing.assert_array_equal(env.bin_at(centers), [0, 1])


_CELL_FACE_LABELS = {"hex": "hexagon faces", "mesh": "triangle faces"}


def _cell_face_source(kind, history):
    from neurospatial.ops.transforms import Affine2D

    if kind == "hex":
        env = Environment.from_layout(
            create_layout(
                "Hexagonal",
                **{"hexagon_width": 4.0, "dimension_ranges": [(1e6, 1e6 + 12)] * 2}
                if history == "large_offset"
                else {"hexagon_width": 4.0, "dimension_ranges": [(0, 12), (0, 12)]},
            )
        )
    else:
        origin = 1e6 if history == "large_offset" else 0.0
        env = Environment.from_layout(
            create_layout(
                "TriangularMesh",
                boundary_polygon=box(origin, origin, origin + 12, origin + 12),
                point_spacing=4.0,
            )
        )
    if history == "subset":
        env = env.subset(mask=np.arange(env.n_bins) % 3 != 0)
    elif history in {"similarity", "reflection"}:
        angle, scale = 0.3, 2.5
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        linear = (
            scale
            * rotation
            @ (np.diag([-1.0, 1.0]) if history == "reflection" else np.eye(2))
        )
        matrix = np.eye(3)
        matrix[:2, :2] = linear
        matrix[:2, 2] = [100.0, -50.0]
        env = env.apply_transform(Affine2D(matrix))
    source = env._binning
    return env, replace(source, faces=KnownFaces(source.faces.build()))


@pytest.mark.filterwarnings("ignore:Triangular mesh is non-K-orthogonal")
@pytest.mark.parametrize("kind", ["hex", "mesh"])
@pytest.mark.parametrize(
    "history", ["native", "subset", "similarity", "reflection", "large_offset"]
)
def test_explicit_cell_faces_from_native_layouts_round_trip(kind, history):
    env, source = _cell_face_source(kind, history)
    restored = Environment.from_bins(source)
    np.testing.assert_array_equal(restored._face_geometry.A, env._face_geometry.A)
    np.testing.assert_array_equal(
        restored._face_geometry.distance, env._face_geometry.distance
    )


@pytest.mark.parametrize("kind", ["hex", "mesh"])
@pytest.mark.parametrize(
    ("field", "factor"), [("A", 10.0), ("distance", 10.0), ("A", np.sqrt(3.0))]
)
@pytest.mark.parametrize("history", ["native", "similarity"])
def test_explicit_cell_faces_must_match_cell_geometry(kind, field, factor, history):
    _, source = _cell_face_source(kind, history)
    faces = source.faces.build()
    invalid = replace(
        source,
        faces=KnownFaces(faces._replace(**{field: getattr(faces, field) * factor})),
    )
    with pytest.raises(ValueError, match=rf"{_CELL_FACE_LABELS[kind]}\.{field}"):
        Environment.from_bins(invalid)


@pytest.mark.parametrize("kind", ["hex", "mesh"])
@pytest.mark.parametrize("change", ["missing_face", "invented_face"])
def test_explicit_cell_faces_cannot_omit_or_invent_shared_faces(kind, change):
    _, source = _cell_face_source(kind, "native")
    bins = source.bins
    faces = source.faces.build()
    if change == "missing_face":
        areas = faces.A.copy()
        areas[0] = 0.0
        normals = source.face_normals.values.copy()
        normals[0] = 0.0
        invalid = replace(
            source,
            faces=KnownFaces(faces._replace(A=areas)),
            face_normals=FaceNormals(normals),
        )
    else:
        # Link the two farthest-apart bins, which cannot share a cell face.
        separation = np.linalg.norm(
            bins.centers[:, None, :] - bins.centers[None, :, :], axis=-1
        )
        u, v = np.unravel_index(np.argmax(separation), separation.shape)
        pairs = np.vstack([bins.pairs, [[u, v]]]).astype(np.intp)
        distance = separation[u, v]
        invalid = replace(
            source,
            bins=replace(
                bins,
                pairs=pairs,
                pair_distances=np.append(bins.pair_distances, distance),
            ),
            faces=KnownFaces(
                FaceGeometry(
                    pairs,
                    np.append(faces.A, faces.A[0]),
                    np.append(faces.distance, distance),
                    faces.volumes,
                )
            ),
            face_normals=FaceNormals(
                np.vstack(
                    [
                        source.face_normals.values,
                        (bins.centers[v] - bins.centers[u]) / distance,
                    ]
                )
            ),
        )
    # A face without a physical counterpart fails the normal-direction check
    # first; a missing face fails the shared-edge check.
    with pytest.raises(
        ValueError,
        match=rf"{_CELL_FACE_LABELS[kind]}\.A|face normals disagree with the locator",
    ):
        Environment.from_bins(invalid)


def test_hexagon_faces_refuse_affine_maps_that_shear_the_lattice():
    from neurospatial.geometry._locators import TransformedLocator

    _, source = _cell_face_source("hex", "native")
    matrix = np.array([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 1.0]])
    bins = source.bins
    centers = bins.centers @ matrix[:2, :2].T
    distances = np.linalg.norm(
        centers[bins.pairs[:, 1]] - centers[bins.pairs[:, 0]], axis=1
    )
    faces = source.faces.build()
    sheared = replace(
        source,
        bins=SpatialBins.from_arrays(
            centers, bins.sizes * 6.0, 2, bins.pairs, distances
        ),
        locator=TransformedLocator(source.locator, np.linalg.inv(matrix)),
        faces=KnownFaces(
            FaceGeometry(
                bins.pairs, faces.A * np.sqrt(6.0), distances, bins.sizes * 6.0
            )
        ),
        shapes=None,
    )
    with pytest.raises(
        ValueError, match=r"hexagon faces|face normals disagree with the locator"
    ):
        Environment.from_bins(sheared)


@pytest.mark.filterwarnings("ignore:Triangular mesh is non-K-orthogonal")
@pytest.mark.parametrize("kind", ["hex", "mesh"])
def test_stored_cell_faces_cannot_bypass_publication_geometry_checks(kind):
    env, _ = _cell_face_source(kind, "native")
    payload = env.to_dict()
    key = "binning/faces/A"
    payload["arrays"][key] = (np.asarray(payload["arrays"][key]) * 7.0).tolist()
    with pytest.raises(ValueError, match=rf"{_CELL_FACE_LABELS[kind]}\.A"):
        Environment.from_dict(payload)
