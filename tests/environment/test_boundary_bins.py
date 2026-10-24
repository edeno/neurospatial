"""Boundary bins follow cell adjacency through subsets and affine maps."""

import networkx as nx
import numpy as np
import pytest
import shapely
from shapely.geometry import box
from shapely.ops import unary_union

from neurospatial import Environment
from neurospatial.binning import (
    BinConnections,
    Binning,
    MissingFaces,
    NearestBinLocator,
    SpatialBins,
)
from neurospatial.geometry._polygon import cell_polygons
from neurospatial.layout import create_layout
from neurospatial.ops.transforms import AffineND


def _cells_touching_outline(env: Environment) -> set[int]:
    """Bins whose cell shares a positive-length edge with the cells' outline."""
    cells = np.asarray(cell_polygons(env._binning), dtype=object)
    outline = unary_union(cells).boundary
    shared = shapely.length(shapely.intersection(shapely.boundary(cells), outline))
    return set(np.flatnonzero(shared > 1e-9).tolist())


def _mesh(polygon, spacing: float) -> Environment:
    return Environment.from_layout(
        create_layout("TriangularMesh", boundary_polygon=polygon, point_spacing=spacing)
    )


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(
            lambda: Environment.from_samples(
                np.random.default_rng(0).uniform(0, 30, (5000, 2)), bin_size=3.0
            ),
            id="grid",
        ),
        pytest.param(
            lambda: Environment.from_samples(
                np.random.default_rng(0).uniform(0, 30, (5000, 2)),
                bin_size=3.0,
                layout="Hexagonal",
            ),
            id="hex",
        ),
        pytest.param(lambda: _mesh(box(0, 0, 20, 20), 2.0), id="mesh"),
        pytest.param(lambda: _mesh(box(0, 0, 30, 30), 4.0), id="coarse-mesh"),
        pytest.param(
            lambda: _mesh(box(0, 0, 20, 20).difference(box(8, 8, 12, 12)), 2.0),
            id="mesh-with-hole",
        ),
        pytest.param(
            lambda: Environment.from_polygon(box(0, 0, 20, 20), bin_size=2.0),
            id="polygon",
        ),
        pytest.param(
            lambda: Environment.from_polygon(box(0, 0, 20, 20), bin_size=2.0).clip(
                box(0, 0, 13, 13)
            ),
            id="clipped-cells",
        ),
    ],
)
def test_boundary_bins_are_the_cells_touching_the_outline(build):
    env = build()
    expected = _cells_touching_outline(env)
    assert 0 < len(expected) < env.n_bins
    assert set(env.boundary_bins.tolist()) == expected


def test_mesh_boundary_bins_survive_an_affine_map():
    env = _mesh(box(0, 0, 20, 20), 2.0)
    np.testing.assert_array_equal(
        env.apply_transform(_rotation(0.7)).boundary_bins, env.boundary_bins
    )


def _point_bins(centers: np.ndarray, measure_dim: int) -> Environment:
    pairs = np.column_stack([np.arange(len(centers) - 1), np.arange(1, len(centers))])
    return Environment.from_bins(
        Binning(
            "sites",
            SpatialBins.from_arrays(centers, np.ones(len(centers)), measure_dim),
            BinConnections.from_arrays(
                pairs, np.linalg.norm(np.diff(centers, axis=0), axis=1)
            ),
            NearestBinLocator(centers, max_distance=0.5),
            MissingFaces("Sites lack faces."),
        )
    )


def test_area_bins_without_cell_geometry_refuse_boundary_detection():
    xx, yy = np.meshgrid(np.arange(5.0), np.arange(5.0))
    env = _point_bins(np.column_stack([xx.ravel(), yy.ravel()]), measure_dim=2)
    with pytest.raises(NotImplementedError, match="cell geometry") as info:
        _ = env.boundary_bins
    assert "Fix:" in str(info.value)


def test_length_bins_without_cells_have_their_chain_ends_as_boundary():
    centers = np.column_stack([np.arange(4.0), np.zeros(4)])
    np.testing.assert_array_equal(
        _point_bins(centers, measure_dim=1).boundary_bins, [0, 3]
    )
    np.testing.assert_array_equal(
        _point_bins(centers[:2], measure_dim=1).boundary_bins, [0, 1]
    )


def test_track_boundary_bins_are_its_dead_ends():
    graph = nx.Graph()
    for node, pos in enumerate([(0.0, 0.0), (50.0, 0.0), (100.0, 0.0), (50.0, 50.0)]):
        graph.add_node(node, pos=pos)
    graph.add_edges_from([(0, 1), (1, 2), (1, 3)])
    env = Environment.from_graph(
        graph, edge_order=[(0, 1), (1, 2), (1, 3)], bin_size=5.0
    )
    ends = [
        int(np.argmin(np.linalg.norm(env.bin_centers - pos, axis=1)))
        for pos in [(0.0, 0.0), (100.0, 0.0), (50.0, 50.0)]
    ]
    np.testing.assert_array_equal(env.boundary_bins, np.sort(ends))


def test_combined_grid_and_track_use_each_components_boundary():
    grid = Environment.from_grid_mask(
        np.ones((3, 3), dtype=bool), grid_edges=(np.arange(4.0), np.arange(4.0))
    )
    graph = nx.Graph()
    graph.add_node(0, pos=(20.0, 0.0))
    graph.add_node(1, pos=(40.0, 0.0))
    graph.add_edge(0, 1)
    track = Environment.from_graph(
        graph, edge_order=[(0, 1)], bin_size=5.0
    ).with_track_width(2.0)
    env = Environment.combine({"grid": grid, "track": track})
    expected = [b for b in range(9) if b != 4] + [9, 9 + track.n_bins - 1]
    np.testing.assert_array_equal(env.boundary_bins, expected)

    # Subsets and affine maps keep each component's boundary.
    np.testing.assert_array_equal(
        env.subset(mask=np.ones(env.n_bins, dtype=bool)).boundary_bins, expected
    )
    np.testing.assert_array_equal(
        env.apply_transform(_rotation(0.7)).boundary_bins, expected
    )
    without_end = np.ones(env.n_bins, dtype=bool)
    without_end[-1] = False
    np.testing.assert_array_equal(
        env.subset(mask=without_end).boundary_bins, [*expected[:-1], env.n_bins - 2]
    )


def test_combined_3d_grids_keep_their_boundary_through_derivations():
    def cube(shift: float) -> Environment:
        edges = np.arange(4.0)
        return Environment.from_grid_mask(
            np.ones((3, 3, 3), dtype=bool), grid_edges=(edges + shift, edges, edges)
        )

    env = Environment.combine({"a": cube(0.0), "b": cube(10.0)})
    # Every cell of a 3x3x3 block but its center touches the block's outline.
    expected = [b for b in range(54) if b not in (13, 27 + 13)]
    np.testing.assert_array_equal(env.boundary_bins, expected)
    np.testing.assert_array_equal(
        env.subset(mask=np.ones(env.n_bins, dtype=bool)).boundary_bins, expected
    )
    translation = np.eye(4)
    translation[0, 3] = 5.0
    np.testing.assert_array_equal(
        env.apply_transform(AffineND(translation)).boundary_bins, expected
    )


def _rotation(angle: float) -> AffineND:
    c, s = np.cos(angle), np.sin(angle)
    return AffineND(np.array([[c, -s, 3.0], [s, c, -1.0], [0.0, 0.0, 1.0]]))


def _with_hole(layout: str) -> tuple[Environment, int, np.ndarray]:
    rng = np.random.default_rng(0)
    env = Environment.from_samples(
        rng.uniform(0, 30, (5000, 2)), bin_size=3.0, layout=layout
    )
    center = int(np.argmin(np.linalg.norm(env.bin_centers - 15.0, axis=1)))
    keep = np.ones(env.n_bins, dtype=bool)
    keep[center] = False
    return env, center, keep


def test_removing_an_interior_grid_bin_exposes_its_orthogonal_neighbors():
    env, center, keep = _with_hole("RegularGrid")
    assert center not in env.boundary_bins
    child_of = np.cumsum(keep) - 1
    offsets = np.abs(
        env.bin_centers[env.neighbors(bins=center)] - env.bin_centers[center]
    )
    orthogonal = np.array(env.neighbors(bins=center))[np.min(offsets, axis=1) < 1e-9]
    assert len(orthogonal) == 4
    expected = set(child_of[np.setdiff1d(env.boundary_bins, center)]) | set(
        child_of[orthogonal]
    )
    assert set(env.subset(mask=keep).boundary_bins.tolist()) == expected


@pytest.mark.parametrize("layout", ["RegularGrid", "Hexagonal"])
def test_boundary_bins_do_not_depend_on_an_affine_map(layout):
    env, _, keep = _with_hole(layout)
    subset = env.subset(mask=keep).boundary_bins
    assert len(subset) > len(env.boundary_bins) - 1
    np.testing.assert_array_equal(
        env.apply_transform(_rotation(0.7)).subset(mask=keep).boundary_bins, subset
    )
    np.testing.assert_array_equal(
        env.subset(mask=keep).apply_transform(_rotation(-0.3)).boundary_bins, subset
    )


@pytest.mark.parametrize("derive", ["subset", "native", "transformed"])
def test_removing_an_interior_hexagon_exposes_all_six_neighbors(derive):
    env, center, keep = _with_hole("Hexagonal")
    neighbors = np.array(env.neighbors(bins=center))
    assert len(neighbors) == 6
    assert center not in env.boundary_bins
    child_of = np.cumsum(keep) - 1
    expected = set(child_of[np.setdiff1d(env.boundary_bins, center)]) | set(
        child_of[neighbors]
    )
    if derive == "native":
        # A hexagonal layout built directly with the same hole.
        child = Environment.from_layout(
            create_layout(
                "Hexagonal",
                **{
                    **env.layout_parameters,
                    "positions": env.bin_centers[keep],
                    "infer_active_bins": True,
                    "bin_count_threshold": 0,
                },
            )
        )
        np.testing.assert_array_equal(child.bin_centers, env.bin_centers[keep])
    else:
        child = env.subset(mask=keep)
        if derive == "transformed":
            child = child.apply_transform(_rotation(0.7))
    assert set(child.boundary_bins.tolist()) == expected
