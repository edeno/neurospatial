"""Decoded layouts validate shared geometry before publishing bin identity."""

from __future__ import annotations

from dataclasses import replace

import networkx as nx
import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.layout._binning import (
    Binning,
    KnownFaces,
    MissingFaces,
    SpatialBins,
)
from neurospatial.layout._binning_layout import BinningLayout
from neurospatial.layout._capabilities import (
    RectilinearGrid,
    RectilinearGridProvider,
    TrackLinearizationProvider,
)
from neurospatial.layout._construction import binning_from_layout
from neurospatial.layout._diffusion import FaceGeometry, FiniteVolumeGeometryProvider
from neurospatial.layout._locators import HexLocator, RectilinearLocator


def _facade(env, *, binning=None, connectivity=None):
    source = env._binning if binning is None else binning
    if not isinstance(source.faces, (KnownFaces, MissingFaces)):
        source = replace(source, faces=KnownFaces(source.faces.build()))
    return BinningLayout(
        source,
        connectivity=env.connectivity if connectivity is None else connectivity,
        dimension_ranges=env.dimension_ranges,
        layout_type_tag=source.kind,
        layout_type=env.layout.layout_type,
        build_params=env.layout_parameters,
        is_linearized_track=env.is_linearized_track,
        coordinate_kind="polar" if getattr(env, "_POLAR", False) else "cartesian",
    )


def test_facade_exposes_the_decoded_snapshot(built_environment):
    env = built_environment
    facade = _facade(env)
    binning = facade.binning()
    assert binning_from_layout(facade, facade.connectivity) is binning
    assert facade.binning() is binning
    assert facade.layout_type == env.layout.layout_type
    np.testing.assert_array_equal(facade.bin_centers, env.bin_centers)
    np.testing.assert_array_equal(facade.bin_sizes(), env.bin_sizes)
    np.testing.assert_array_equal(
        facade.point_to_bin_index(env.bin_centers), np.arange(env.n_bins)
    )
    assert not isinstance(facade, RectilinearGridProvider)
    assert not isinstance(facade, TrackLinearizationProvider)
    assert not isinstance(facade, FiniteVolumeGeometryProvider)
    if binning.rectilinear is not None:
        component = binning.rectilinear
        assert facade.grid_shape == component.shape
        np.testing.assert_array_equal(facade.active_mask, component.active_mask)
        for actual, expected in zip(facade.grid_edges, component.edges, strict=True):
            np.testing.assert_array_equal(actual, expected)
    elif binning.linearization is not None:
        np.testing.assert_array_equal(
            facade.active_mask, binning.linearization.active_mask
        )
        assert facade.grid_shape == facade.active_mask.shape
    elif isinstance(binning.locator, HexLocator):
        locator = binning.locator
        assert facade.grid_edges == ()
        assert facade.grid_shape == locator.centers_shape
        expected = (locator.flat_to_active >= 0).reshape(locator.centers_shape)
        np.testing.assert_array_equal(facade.active_mask, expected)
        assert np.count_nonzero(facade.active_mask) == env.n_bins
    else:
        assert len(facade.grid_edges) == 0
        assert facade.grid_shape == (env.n_bins,)
        np.testing.assert_array_equal(
            facade.active_mask, np.ones(env.n_bins, dtype=bool)
        )


@pytest.fixture
def grid_env():
    return Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool), grid_edges=(np.arange(3.0),) * 2
    )


@pytest.mark.parametrize(
    "mismatch",
    [
        "node_labels",
        "node_position",
        "distance",
        "face_edges",
        "face_volumes",
        "active_count",
        "grid_edges",
        "grid_shape",
        "locator_order",
        "locator_edges",
    ],
)
def test_facade_rejects_inconsistent_data(grid_env, mismatch):
    env = grid_env
    binning = env._binning
    graph = env.connectivity.copy()
    pattern = ""
    if mismatch == "node_labels":
        graph = nx.relabel_nodes(graph, {0: env.n_bins})
        pattern = "nodes|labels"
    elif mismatch == "node_position":
        graph.nodes[0]["pos"] = (0.51, 0.5)
        pattern = "pos|centers"
    elif mismatch == "distance":
        graph.edges[next(iter(graph.edges))]["distance"] += 0.01
        pattern = "distance"
    elif mismatch.startswith("face_"):
        faces = binning.faces.build()
        if mismatch == "face_edges":
            faces = faces._replace(edges=faces.edges[::-1])
            pattern = "faces.*edges|face.*pairs"
        else:
            faces = faces._replace(volumes=faces.volumes * 2.0)
            pattern = "volumes|sizes"
        binning = replace(binning, faces=KnownFaces(faces))
    elif mismatch in {"active_count", "grid_edges", "grid_shape"}:
        grid = binning.rectilinear
        if mismatch == "active_count":
            mask = grid.active_mask.copy()
            mask.flat[0] = False
            grid = replace(grid, active_mask=mask)
            pattern = "active_mask|active.*count"
        elif mismatch == "grid_shape":
            grid = replace(grid, shape=(1, 4))
            pattern = "shape|edges"
        else:
            edges = [edge.copy() for edge in grid.edges]
            edges[0][1] = 0.95
            grid = replace(grid, edges=tuple(edges))
            pattern = "edges|centers|geometry"
        binning = replace(binning, rectilinear=grid)
    elif mismatch == "locator_order":
        mapping = np.arange(env.n_bins, dtype=np.intp)[::-1]
        binning = replace(
            binning, locator=replace(binning.locator, flat_to_active=mapping)
        )
        pattern = "flat_to_active|locator|bin.*order"
    else:
        edges = [edge.copy() for edge in binning.locator.edges]
        edges[0][1] = 0.95
        binning = replace(binning, locator=replace(binning.locator, edges=tuple(edges)))
        pattern = "edges|centers|geometry"
    with pytest.raises(ValueError, match=pattern):
        _facade(env, binning=binning, connectivity=graph)


@pytest.mark.parametrize("graph_order", ["pair_order", "pair_orientation"])
def test_graph_iteration_order_preserves_authoritative_pairs_and_weights(
    grid_env, graph_order
):
    """Graph iteration cannot change bin IDs or ordered gradient/flux values."""
    from neurospatial.ops.calculus import divergence, gradient

    env = grid_env
    graph = nx.Graph()
    nodes = list(env.connectivity.nodes(data=True))
    graph.add_nodes_from(nodes[::-1] if graph_order == "pair_orientation" else nodes)
    graph.add_edges_from(list(env.connectivity.edges(data=True))[::-1])
    assert list(graph.edges) != list(env.connectivity.edges)
    facade = _facade(env, connectivity=graph)
    bins = facade.binning().bins
    np.testing.assert_array_equal(bins.centers, env.bin_centers)
    np.testing.assert_array_equal(bins.pairs, env.edge_pairs)
    np.testing.assert_array_equal(bins.pair_distances, env._binning.bins.pair_distances)
    for (u, v), distance in zip(bins.pairs, bins.pair_distances, strict=True):
        assert facade.connectivity.edges[u, v]["distance"] == distance
    for actual, expected in zip(
        facade.binning().faces.build(), env._face_geometry, strict=True
    ):
        np.testing.assert_array_equal(actual, expected)

    published = Environment(layout=facade, layout_type_used=facade._layout_type_tag)
    np.testing.assert_array_equal(
        published.bin_at(env.bin_centers), np.arange(env.n_bins)
    )
    np.testing.assert_array_equal(published.edge_pairs, env.edge_pairs)
    attributes = published.get_edge_attributes()
    np.testing.assert_array_equal(
        attributes[["source", "target"]].to_numpy(), published.edge_pairs
    )
    vectors = (
        published.bin_centers[published.edge_pairs[:, 1]]
        - published.bin_centers[published.edge_pairs[:, 0]]
    )
    np.testing.assert_array_equal(np.asarray(attributes["vector"].tolist()), vectors)
    np.testing.assert_array_equal(
        attributes[["vector_dim0", "vector_dim1"]].to_numpy(), vectors
    )
    np.testing.assert_array_equal(attributes["distance"], bins.pair_distances)
    np.testing.assert_array_equal(
        attributes["angle_2d"], np.arctan2(vectors[:, 1], vectors[:, 0])
    )
    field = np.arange(env.n_bins, dtype=float) ** 2
    flux = np.arange(len(env.edge_pairs), dtype=float) + 0.5
    np.testing.assert_array_equal(gradient(published, field), gradient(env, field))
    np.testing.assert_array_equal(divergence(published, flux), divergence(env, flux))
    np.testing.assert_allclose(
        published.compute_kernel(1.0, mode="average"),
        env.compute_kernel(1.0, mode="average"),
    )


@pytest.mark.parametrize("built_environment", ["hex"], indirect=True)
@pytest.mark.parametrize("field", ["centers", "radius"])
def test_facade_rejects_hexagon_geometry_disagreement(built_environment, field):
    env = built_environment
    shapes = env._binning.shapes
    changed = replace(shapes, **{field: getattr(shapes, field) + 0.01})
    with pytest.raises(ValueError, match=field):
        _facade(env, binning=replace(env._binning, shapes=changed))


@pytest.mark.parametrize("built_environment", ["mesh_boundary"], indirect=True)
@pytest.mark.parametrize("field", ["points", "triangles", "boundary"])
def test_facade_rejects_mesh_geometry_disagreement(built_environment, field):
    env = built_environment
    binning = env._binning
    if field == "boundary":
        from shapely.geometry import Polygon

        binning = replace(
            binning,
            outline=replace(
                binning.outline, geometry=Polygon([(0, 0), (12, 0), (12, 12), (0, 12)])
            ),
        )
    else:
        shapes = binning.shapes
        changed = shapes.points + 0.01 if field == "points" else shapes.triangles[:-1]
        binning = replace(binning, shapes=replace(shapes, **{field: changed}))
    with pytest.raises(ValueError, match=field):
        _facade(env, binning=binning)


@pytest.mark.parametrize("built_environment", ["track"], indirect=True)
@pytest.mark.parametrize("field", ["active_mask", "grid_edges", "bin_indices"])
def test_facade_rejects_track_geometry_disagreement(built_environment, field):
    env = built_environment
    binning = env._binning
    component = binning.linearization
    if field == "bin_indices":
        locator = replace(binning.locator, bin_indices=np.arange(env.n_bins)[::-1])
        binning = replace(binning, locator=locator)
    elif field == "active_mask":
        mask = component.active_mask.copy()
        mask[0] = False
        binning = replace(binning, linearization=replace(component, active_mask=mask))
    else:
        edges = [edge.copy() for edge in component.grid_edges]
        edges[0][1] -= 0.01
        binning = replace(
            binning, linearization=replace(component, grid_edges=tuple(edges))
        )
    with pytest.raises(
        ValueError, match=r"active_mask|edges|sizes|centers|bin_indices|locator"
    ):
        _facade(env, binning=binning)


@pytest.mark.parametrize("built_environment", ["track"], indirect=True)
def test_facade_accepts_coherent_nearest_locator_storage_permutation(built_environment):
    env = built_environment
    locator = env._binning.locator
    permuted = replace(
        locator, centers=locator.centers[::-1], bin_indices=locator.bin_indices[::-1]
    )
    facade = _facade(env, binning=replace(env._binning, locator=permuted))
    np.testing.assert_array_equal(
        facade.point_to_bin_index(env.bin_centers), np.arange(env.n_bins)
    )


@pytest.mark.parametrize("built_environment", ["hex", "mesh_boundary"], indirect=True)
def test_facade_accepts_coherent_geometric_bin_permutation(built_environment):
    """Active maps encode bin identity independently of lattice or simplex order."""
    env = built_environment
    original = env._binning
    permutation = np.arange(env.n_bins)[::-1]
    inverse = np.argsort(permutation)
    bins = replace(
        original.bins,
        centers=original.bins.centers[permutation],
        sizes=original.bins.sizes[permutation],
        pairs=inverse[original.bins.pairs],
    )
    graph = nx.relabel_nodes(env.connectivity, dict(enumerate(inverse)), copy=True)
    np.testing.assert_array_equal(np.asarray(list(graph.edges)), bins.pairs)
    mapping_name = (
        "flat_to_active" if original.kind == "Hexagonal" else "simplex_to_active"
    )
    mapping = getattr(original.locator, mapping_name).copy()
    active = mapping >= 0
    mapping[active] = inverse[mapping[active]]
    locator = replace(original.locator, **{mapping_name: mapping})
    shapes = original.shapes
    if original.kind == "Hexagonal":
        shapes = replace(shapes, centers=shapes.centers[permutation])
    else:
        shapes = replace(shapes, triangles=shapes.triangles[permutation])
    faces = original.faces.build()
    faces = KnownFaces(
        faces._replace(edges=inverse[faces.edges], volumes=faces.volumes[permutation])
    )
    binning = replace(original, bins=bins, locator=locator, shapes=shapes, faces=faces)
    facade = _facade(env, binning=binning, connectivity=graph)
    np.testing.assert_array_equal(
        facade.point_to_bin_index(bins.centers), np.arange(env.n_bins)
    )


@pytest.mark.parametrize(
    "built_environment", ["grid8", "polar", "polar_open"], indirect=True
)
@pytest.mark.parametrize(
    "corrupt", [False, True], ids=["original_measures", "doubled_measures"]
)
def test_facade_validates_physical_grid_measures(built_environment, corrupt):
    """Polar sectors use radial areas while Cartesian cells use edge products."""
    env = built_environment
    binning = env._binning
    if corrupt:
        faces = env._face_geometry
        binning = replace(
            binning,
            bins=replace(binning.bins, sizes=binning.bins.sizes * 2.0),
            faces=KnownFaces(faces._replace(volumes=faces.volumes * 2.0)),
        )
        with pytest.raises(ValueError, match=r"sizes|volumes|measures|area"):
            _facade(env, binning=binning)
    else:
        facade = _facade(env)
        np.testing.assert_array_equal(facade.bin_sizes(), env.bin_sizes)


def test_facade_rejects_symmetric_cell_widening_without_measure_change():
    """Unchanged centroids cannot hide a different physical cell area."""
    env = Environment.from_grid_mask(
        np.ones((1, 1), dtype=bool), grid_edges=(np.arange(2.0),) * 2
    )
    original = env._binning
    edges = (np.array([-0.5, 1.5]), original.rectilinear.edges[1])
    binning = replace(
        original,
        locator=replace(original.locator, edges=edges),
        rectilinear=replace(original.rectilinear, edges=edges),
        faces=KnownFaces(env._face_geometry),
    )
    with pytest.raises(ValueError, match=r"sizes|volumes|measures|area"):
        BinningLayout(
            binning,
            connectivity=env.connectivity,
            dimension_ranges=env.dimension_ranges,
            layout_type_tag=original.kind,
            layout_type=env.layout.layout_type,
            build_params=env.layout_parameters,
            is_linearized_track=False,
        )


@pytest.mark.parametrize("case", ["nearly_uniform", "large_offset"])
def test_facade_preserves_native_grid_measure_precision(case):
    """Accepted floating-point grid edges retain native constant cell measures."""
    if case == "nearly_uniform":
        edges = (np.array([0.0, 1.0, 2.0 + 5e-8, 3.0]), np.array([0.0, 1.0]))
    else:
        edges = (1e9 + np.arange(4) * 0.1, 1e9 + np.arange(3) * 0.2)
    env = Environment.from_grid_mask(
        np.ones(tuple(len(edge) - 1 for edge in edges), dtype=bool), grid_edges=edges
    )
    np.testing.assert_array_equal(_facade(env).bin_sizes(), env.bin_sizes)
    loaded = Environment.from_dict(env.to_dict())
    np.testing.assert_array_equal(loaded.bin_sizes, env.bin_sizes)


@pytest.mark.parametrize(
    "corrupt", [False, True], ids=["cell_measures", "constant_measures"]
)
def test_facade_validates_genuinely_nonuniform_cartesian_cells(corrupt):
    edges = (np.array([0.0, 1.0, 3.0]), np.array([0.0, 1.0]))
    mask = np.ones((2, 1), dtype=bool)
    centers = np.array([[0.5, 0.5], [2.0, 0.5]])
    sizes = np.array([1.0, 1.0] if corrupt else [1.0, 2.0])
    pairs = np.array([[0, 1]])
    distances = np.array([1.5])
    binning = Binning(
        "MaskedGrid",
        SpatialBins(centers, sizes, 2, pairs, distances),
        RectilinearLocator(edges, (2, 1), mask, np.arange(2)),
        KnownFaces(FaceGeometry(pairs, np.array([1.0]), distances, sizes)),
        rectilinear=RectilinearGrid(edges, (2, 1), mask),
    )
    graph = nx.Graph()
    for index, center in enumerate(centers):
        graph.add_node(index, pos=tuple(center))
    graph.add_edge(0, 1, distance=1.5)

    def construct():
        return BinningLayout(
            binning,
            connectivity=graph,
            dimension_ranges=[(0.0, 3.0), (0.0, 1.0)],
            layout_type_tag="MaskedGrid",
            layout_type="mask",
            build_params={},
            is_linearized_track=False,
        )

    if corrupt:
        with pytest.raises(ValueError, match=r"sizes|measures"):
            construct()
    else:
        facade = construct()
        np.testing.assert_array_equal(facade.bin_sizes(), [1.0, 2.0])
        np.testing.assert_array_equal(facade.point_to_bin_index(centers), [0, 1])
