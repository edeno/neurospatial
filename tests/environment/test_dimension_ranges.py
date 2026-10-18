"""``dimension_ranges`` is the extent of the bins' cells, for every layout.

A parent reports its cells' extent and a subset the extent of the cells it
keeps, with no margin, so a subset never reports more than its parent.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from shapely.geometry import Point, Polygon

from neurospatial import Environment
from neurospatial.geometry._polygon import materialize_cells
from neurospatial.layout import create_layout
from neurospatial.ops.transforms import AffineND

_CARTESIAN = [
    "grid8",
    "grid4",
    "polygon",
    "mask",
    "pixels",
    "hex",
    "track",
    "mesh",
    "mesh_boundary",
]


def _keep_first_quarter(env: Environment) -> np.ndarray:
    keep = np.zeros(env.n_bins, dtype=bool)
    keep[: max(1, env.n_bins // 4)] = True
    return keep


def _cell_bounds(env: Environment) -> np.ndarray:
    bounds = np.array([cell.bounds for cell in materialize_cells(env.binning)])
    return np.column_stack((bounds[:, :2].min(axis=0), bounds[:, 2:].max(axis=0)))


@pytest.mark.parametrize("built_environment", ["hex", "mesh"], indirect=True)
def test_subset_ranges_are_the_kept_cells_extent(built_environment):
    child = built_environment.subset(mask=_keep_first_quarter(built_environment))
    np.testing.assert_allclose(
        child.dimension_ranges, _cell_bounds(child), rtol=0, atol=1e-12
    )


def test_subset_of_clipped_polygon_cells_bounds_the_kept_polygons():
    grid = Environment.from_grid_mask(
        np.ones((6, 6), dtype=bool), grid_edges=(np.linspace(0, 12, 7),) * 2
    )
    clipped = grid.clip(Point(6, 6).buffer(5.0))
    child = clipped.subset(mask=_keep_first_quarter(clipped))
    np.testing.assert_array_equal(child.dimension_ranges, _cell_bounds(child))


def _track_cell_ends(env: Environment) -> np.ndarray:
    """Each bin's centerline stretch: its center +- half its length, on its segment."""
    graph = env.binning.linearization.graph
    ends = []
    for center, length in zip(env.bin_centers, env.bin_sizes, strict=True):
        for u, v in graph.edges:
            start = np.asarray(graph.nodes[u]["pos"], dtype=float)
            stop = np.asarray(graph.nodes[v]["pos"], dtype=float)
            direction = (stop - start) / np.linalg.norm(stop - start)
            along = (center - start) @ direction
            off = np.linalg.norm(center - start - along * direction)
            if off < 1e-9 and -1e-9 <= along <= np.linalg.norm(stop - start) + 1e-9:
                ends += [
                    center - length / 2 * direction,
                    center + length / 2 * direction,
                ]
                break
        else:
            raise AssertionError(f"bin center {center} lies on no track segment")
    return np.array(ends)


@pytest.mark.parametrize("built_environment", ["track"], indirect=True)
def test_track_subset_bounds_the_kept_stretches_of_centerline(built_environment):
    child = built_environment.subset(mask=_keep_first_quarter(built_environment))
    ends = _track_cell_ends(child)
    np.testing.assert_allclose(
        child.dimension_ranges,
        np.column_stack((ends.min(axis=0), ends.max(axis=0))),
        rtol=0,
        atol=1e-12,
    )


def _expected_extent(kind: str, env: Environment) -> np.ndarray:
    """The extent of the cells, computed from their geometry."""
    if kind in {"grid8", "grid4", "polygon", "mask", "pixels", "polar", "polar_open"}:
        return _active_cell_bounds(env.binning.rectilinear)
    if kind == "track":
        nodes = np.array([(0, 0), (6, 0), (6, 6), (12, 0)], dtype=float)
        return np.column_stack((nodes.min(axis=0), nodes.max(axis=0)))
    return _cell_bounds(env)


def test_every_layout_reports_the_extent_of_its_cells(built_environment, request):
    kind = request.node.callspec.params["built_environment"]
    np.testing.assert_allclose(
        built_environment.dimension_ranges,
        _expected_extent(kind, built_environment),
        rtol=0,
        atol=1e-12,
    )


def _derived(kind: str, env: Environment) -> dict[str, Environment]:
    """The environment and every derivation that applies to its layout."""
    angle, scale = 0.7, 1.5
    matrix = np.eye(3)
    matrix[:2, :2] = scale * np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    matrix[:2, 2] = (3.0, -2.0)
    derived = {"built": env, "transformed": env.apply_transform(AffineND(matrix))}
    if kind != "track":
        derived["clipped"] = env.clip(Point(6, 6).buffer(4.0))
    if kind in {"grid8", "grid4", "polygon", "mask", "pixels"}:
        derived["rebinned"] = env.rebin(2)
    return derived


def _assert_within(inner, outer, scale: float) -> None:
    inner, outer = np.asarray(inner), np.asarray(outer)
    tolerance = 1e-12 * scale
    assert np.all(inner[:, 0] >= outer[:, 0] - tolerance), (inner, outer)
    assert np.all(inner[:, 1] <= outer[:, 1] + tolerance), (inner, outer)


@pytest.mark.parametrize("built_environment", _CARTESIAN, indirect=True)
@settings(derandomize=True, max_examples=10, deadline=None)
@given(seed=st.integers(min_value=0, max_value=2**32 - 1))
def test_a_subset_never_exceeds_its_parent(built_environment, request, seed):
    kind = request.node.callspec.params["built_environment"]
    rng = np.random.default_rng(seed)
    for name, parent in _derived(kind, built_environment).items():
        ranges = np.asarray(parent.dimension_ranges)
        scale = float(np.max(np.abs(ranges)))
        # Every bin lies inside its environment's ranges.
        _assert_within(
            np.column_stack((parent.bin_centers.min(0), parent.bin_centers.max(0))),
            ranges,
            scale,
        )
        # Keeping every bin reports the parent's own extent ...
        whole = parent.subset(mask=np.ones(parent.n_bins, dtype=bool))
        np.testing.assert_allclose(
            whole.dimension_ranges, ranges, rtol=0, atol=1e-12 * scale, err_msg=name
        )
        # ... and keeping some of them never reports more.
        keep = rng.random(parent.n_bins) < rng.uniform(0.05, 0.95)
        keep[rng.integers(parent.n_bins)] = True
        _assert_within(parent.subset(mask=keep).dimension_ranges, ranges, scale)


def _active_cell_bounds(grid) -> np.ndarray:
    """Bounds of a grid's active cells, from the grid edges and mask."""
    cells = np.argwhere(grid.active_mask)
    return np.array(
        [
            (edge[cells[:, d].min()], edge[cells[:, d].max() + 1])
            for d, edge in enumerate(grid.edges)
        ]
    )


@pytest.mark.parametrize("built_environment", ["grid8", "polygon"], indirect=True)
def test_grid_subset_reports_its_kept_cells_not_the_lattice(built_environment):
    # A grid subset keeps its whole lattice (grid_edges), but its extent is
    # that of the cells it keeps.
    child = built_environment.subset(mask=_keep_first_quarter(built_environment))
    np.testing.assert_array_equal(
        child.dimension_ranges, _active_cell_bounds(child.binning.rectilinear)
    )
    lattice = np.array([(edge[0], edge[-1]) for edge in child.grid_edges])
    assert not np.array_equal(np.asarray(child.dimension_ranges), lattice)


def test_grid_with_an_inactive_border_reports_its_active_cells():
    mask = np.zeros((6, 5), dtype=bool)
    mask[1:4, 2:5] = True
    env = Environment.from_grid_mask(
        mask, grid_edges=(np.arange(7.0) * 2.0, np.arange(6.0) * 2.0)
    )
    np.testing.assert_array_equal(env.dimension_ranges, [(2.0, 8.0), (4.0, 10.0)])


@pytest.mark.parametrize("built_environment", ["mesh"], indirect=True)
def test_mesh_subset_shrinks_to_its_kept_triangles(built_environment):
    parent = built_environment
    corner = np.all(parent.bin_centers < 4.0, axis=1)
    assert 0 < corner.sum() < parent.n_bins
    child = parent.subset(mask=corner)
    shapes = child.binning.shapes
    vertices = shapes.points[shapes.triangles].reshape(-1, 2)
    np.testing.assert_array_equal(
        child.dimension_ranges,
        np.column_stack((vertices.min(axis=0), vertices.max(axis=0))),
    )
    # Every mesh point is retained, but the unused ones do not widen the extent.
    assert np.all(np.asarray(child.dimension_ranges)[:, 1] <= 8.0)
    assert np.all(shapes.points.max(axis=0) == 12.0)


def _clipped_grid() -> Environment:
    grid = Environment.from_grid_mask(
        np.ones((6, 6), dtype=bool), grid_edges=(np.linspace(0, 12, 7),) * 2
    )
    return grid.clip(Point(6, 6).buffer(5.0))


def _hexagons() -> Environment:
    return Environment.from_layout(
        create_layout(
            "Hexagonal", hexagon_width=4.0, dimension_ranges=[(0, 12), (0, 12)]
        )
    )


def _triangles() -> Environment:
    return Environment.from_layout(
        create_layout(
            "TriangularMesh",
            boundary_polygon=Polygon([(0, 0), (12, 0), (12, 12), (0, 12)]),
            point_spacing=4.0,
        )
    )


@pytest.mark.parametrize("build", [_hexagons, _triangles, _clipped_grid])
def test_cell_shapes_select_and_bound_their_own_cells(build):
    env = build()
    keep = np.zeros(env.n_bins, dtype=bool)
    keep[1::3] = True
    kept = env.binning.shapes.subset(keep)
    child = env.subset(mask=keep)
    assert type(kept) is type(child.binning.shapes)
    assert kept.extent() == child.dimension_ranges
    for shapes, owner in ((kept, child), (env.binning.shapes, env)):
        np.testing.assert_allclose(
            shapes.extent(), _cell_bounds(owner), rtol=0, atol=1e-12
        )
    with pytest.raises(ValueError, match="boolean mask"):
        env.binning.shapes.subset(np.flatnonzero(keep))
