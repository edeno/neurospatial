"""Decoded layouts validate shared geometry before publishing bin identity."""

from __future__ import annotations

from dataclasses import replace

import networkx as nx
import numpy as np
import pytest
from environment_helpers import publish_binning

from neurospatial import Environment
from neurospatial.environment._fields import face_geometry
from neurospatial.geometry._binning import (
    Binning,
    KnownFaces,
    MissingFaces,
    SpatialBins,
)
from neurospatial.geometry._capabilities import (
    RectilinearGrid,
)
from neurospatial.geometry._diffusion import FaceGeometry
from neurospatial.geometry._locators import HexLocator, MeshLocator, RectilinearLocator


def _facade(env, *, binning=None):
    source = env._binning if binning is None else binning
    if not isinstance(source.faces, (KnownFaces, MissingFaces)):
        source = replace(source, faces=KnownFaces(source.faces.build()))
    return publish_binning(
        source,
        dimension_ranges=env.dimension_ranges,
        layout_type=source.kind,
        build_params=env.layout_parameters,
        coordinate_kind=env.coordinate_kind,
        arena=env.arena if env.coordinate_kind == "polar" else None,
    )


def test_facade_exposes_the_decoded_snapshot(built_environment):
    env = built_environment
    facade = _facade(env)
    binning = facade._binning
    assert binning is facade._binning
    assert facade._binning is binning
    assert facade.layout_type == env.layout_type
    np.testing.assert_array_equal(facade.bin_centers, env.bin_centers)
    np.testing.assert_array_equal(facade.bin_sizes, env.bin_sizes)
    np.testing.assert_array_equal(
        facade._binning.locator.locate(env.bin_centers), np.arange(env.n_bins)
    )
    assert not hasattr(facade, "rectilinear_grid")
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
    elif isinstance(binning.locator, MeshLocator):
        assert facade.grid_shape == env.grid_shape
        np.testing.assert_array_equal(facade.active_mask, env.active_mask)
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
    if mismatch.startswith("face_"):
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
        _facade(env, binning=binning)


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
        facade._binning.locator.locate(env.bin_centers), np.arange(env.n_bins)
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
    facade = _facade(env, binning=binning)
    np.testing.assert_array_equal(
        facade._binning.locator.locate(bins.centers), np.arange(env.n_bins)
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
        faces = face_geometry(env._cache, env._binning)
        binning = replace(
            binning,
            bins=replace(binning.bins, sizes=binning.bins.sizes * 2.0),
            faces=KnownFaces(faces._replace(volumes=faces.volumes * 2.0)),
        )
        with pytest.raises(ValueError, match=r"sizes|volumes|measures|area"):
            _facade(env, binning=binning)
    else:
        facade = _facade(env)
        np.testing.assert_array_equal(facade.bin_sizes, env.bin_sizes)


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
        faces=KnownFaces(face_geometry(env._cache, env._binning)),
    )
    with pytest.raises(ValueError, match=r"sizes|volumes|measures|area"):
        publish_binning(
            binning,
            dimension_ranges=env.dimension_ranges,
            layout_type=original.kind,
            build_params=env.layout_parameters,
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
    np.testing.assert_array_equal(_facade(env).bin_sizes, env.bin_sizes)
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
        graph.add_node(
            index,
            pos=tuple(center),
            source_grid_flat_index=index,
            original_grid_nd_index=(index,),
        )
    graph.add_edge(0, 1, distance=1.5, edge_id=0, vector=(1.5, 0.0), angle_2d=0.0)

    def construct():
        return publish_binning(
            binning,
            dimension_ranges=[(0.0, 3.0), (0.0, 1.0)],
            layout_type="MaskedGrid",
            build_params={},
        )

    if corrupt:
        with pytest.raises(ValueError, match=r"sizes|measures"):
            construct()
    else:
        facade = construct()
        np.testing.assert_array_equal(facade.bin_sizes, [1.0, 2.0])
        np.testing.assert_array_equal(facade._binning.locator.locate(centers), [0, 1])
