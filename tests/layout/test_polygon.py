"""Exact polygon support is independent of conditional numerical geometry."""

from __future__ import annotations

import copy
import pickle
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
import shapely
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box
from shapely.ops import unary_union

from neurospatial import Environment
from neurospatial.geometry._binning import KnownFaces, MissingFaces
from neurospatial.geometry._derive import subset_binning, transform_binning
from neurospatial.geometry._diffusion import FaceGeometry
from neurospatial.geometry._locators import RectilinearLocator, SubsetLocator
from neurospatial.geometry._polygon import (
    cells_from_locator,
    certify_faces,
    clip_binning,
    materialize_cells,
)
from neurospatial.geometry._polygon_cells import (
    ClipMapping,
    ClippedLocator,
    PolygonCells,
    transform_polygon_cells,
)
from neurospatial.ops.transforms import AffineND


def _grid(shape=(2, 1), edges=None, diagonals=False):
    if edges is None:
        edges = tuple(np.arange(size + 1.0) for size in shape)
    return Environment.from_grid_mask(
        np.ones(shape, dtype=bool),
        grid_edges=edges,
        connect_diagonal_neighbors=diagonals,
    )._binning


def test_clip_mapping_owns_strided_indices_and_preserves_protection():
    caller = np.array([0, 99, 0, 99, 1, 99])
    view = caller[::2]
    mapping = ClipMapping(view, 2)
    caller[:] = 0
    np.testing.assert_array_equal(mapping.parent_bin_indices, [0, 0, 1])
    assert caller.flags.writeable and view.flags.writeable
    for restored in (
        mapping,
        copy.deepcopy(mapping),
        pickle.loads(pickle.dumps(mapping)),
    ):
        assert not restored.parent_bin_indices.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            restored.parent_bin_indices[0] = 1
        with pytest.raises(FrozenInstanceError):
            restored.n_parent_bins = 3
    assert mapping != ClipMapping([0, 0, 1], 2)


@pytest.mark.parametrize(
    "indices,count",
    [([0.5], 2), ([True], 2), ([-1], 2), ([2], 2), ([[0]], 2), ([0], -1), ([0], True)],
)
def test_mapping_rejects_invalid_indices(indices, count):
    with pytest.raises(ValueError):
        ClipMapping(indices, count)


@pytest.mark.parametrize(
    "polygons",
    [
        (Point(0, 0),),
        (Polygon(),),
        (Polygon([(0, 0), (1, 1), (1, 0), (0, 1)]),),
        (box(0, 0, 2, 2), box(1, 0, 3, 2)),
        (MultiPolygon([box(0, 0, 1, 1), box(2, 0, 3, 1)]),),
    ],
)
def test_polygon_cells_reject_invalid_partition(polygons):
    with pytest.raises(ValueError):
        PolygonCells(polygons)


def test_polygon_cells_name_the_first_invalid_cell():
    bow_tie = Polygon([(0, 0), (1, 1), (1, 0), (0, 1)])
    infinite = Polygon([(10, 0), (np.inf, 0), (11, 1)])
    with pytest.raises(ValueError, match=r"cells\[1\] is invalid"):
        PolygonCells((box(0, 0, 1, 1), bow_tie, infinite))
    with pytest.raises(ValueError, match=r"cells\[1\] must be nonempty, finite"):
        PolygonCells((box(0, 0, 1, 1), infinite, bow_tie))


def test_polygon_cells_name_the_first_overlapping_pair():
    # Cells sharing edges and corners are a partition; a sliver of overlap,
    # between cells whose bounding boxes touch only nearly, is not.
    grid = tuple(box(i, j, i + 1, j + 1) for i in range(3) for j in range(3))
    PolygonCells(grid)
    sliver = box(2.999, 0, 4, 1)
    with pytest.raises(
        ValueError, match=r"cells\[6\] and cells\[9\] overlap in their interiors"
    ):
        PolygonCells((*grid, sliver))


def test_clipped_locator_preserves_parent_boundary_and_nonfinite_semantics():
    base = RectilinearLocator(
        (np.array([0.0, 4.0]), np.array([0.0, 4.0])), (1, 1), np.ones((1, 1), bool)
    )
    cells = PolygonCells((box(0, 0, 1, 4), box(3, 0, 4, 4)))
    mapping = ClipMapping([0, 0], 1)
    locator = ClippedLocator(base, cells, mapping)
    points = np.array([[0.5, 2], [2, 2], [3.5, 2], [4, 2], [np.nan, 2], [np.inf, 2]])
    untouched = points.copy()
    # (4, 2) lies on the parent grid's outline, which belongs to the last cell.
    np.testing.assert_array_equal(locator.locate(points), [0, -1, 1, 1, -1, -1])
    np.testing.assert_array_equal(points, untouched)
    for restored in (copy.deepcopy(locator), pickle.loads(pickle.dumps(locator))):
        np.testing.assert_array_equal(restored.locate(points), locator.locate(points))
        assert not restored.mapping.parent_bin_indices.flags.writeable
    np.testing.assert_array_equal(
        locator.locate(np.empty((0, 2))), np.empty(0, dtype=int)
    )
    for width in (1, 3):
        with pytest.raises(ValueError, match="shape"):
            locator.locate(np.empty((0, width)))


def test_subset_then_reclip_does_not_resurrect_point_touch_owner():
    parent = _grid((1, 1), (np.array([0.0, 2.0]),) * 2)
    child, _ = clip_binning(parent, MultiPolygon([box(0, 0, 1, 1), box(1, 1, 2, 2)]))
    query = np.array([[1.0, 1.0]])
    np.testing.assert_array_equal(child.locator.locate(query), [0])
    selected = replace(
        child,
        bins=replace(
            child.bins,
            centers=child.bins.centers[1:],
            sizes=child.bins.sizes[1:],
            measure_dims=child.bins.measure_dims[1:],
        ),
        connections=child.connections.select(np.array([-1, 0]))[0],
        locator=SubsetLocator(child.locator, np.array([-1, 0])),
        shapes=child.shapes.subset([False, True]),
        faces=KnownFaces(
            FaceGeometry(
                np.empty((0, 2), dtype=int),
                np.empty(0),
                np.empty(0),
                child.bins.sizes[1:],
            )
        ),
    )
    again, _ = clip_binning(selected, box(0, 0, 2, 2))
    np.testing.assert_array_equal(again.locator.locate(query), [-1])


@pytest.mark.parametrize(
    "built_environment",
    ["grid8", "grid4", "polygon", "mask", "pixels", "hex", "mesh", "mesh_boundary"],
    indirect=True,
)
def test_materialized_native_cells_match_measures_and_centers(built_environment):
    parent = built_environment._binning
    cells = materialize_cells(parent)
    assert len(cells) == parent.bins.n_bins
    np.testing.assert_allclose(
        [polygon.area for polygon in cells], parent.bins.sizes, rtol=1e-12, atol=1e-12
    )
    assert all(
        polygon.contains(Point(center))
        for polygon, center in zip(cells, parent.bins.centers, strict=True)
    )
    assert unary_union(cells).area == pytest.approx(
        sum(polygon.area for polygon in cells), rel=1e-13
    )


@pytest.mark.parametrize("built_environment", ["hex"], indirect=True)
def test_canonical_hex_materialization_has_all_shared_faces(built_environment):
    parent = built_environment._binning
    cells = materialize_cells(parent)
    for u, v in parent.connections.pairs:
        assert cells[u].intersection(cells[v]).area == 0
        assert cells[u].boundary.intersection(
            cells[v].boundary
        ).length == pytest.approx(parent.shapes.radius)


def test_disconnected_intersection_splits_and_shares_owned_data():
    parent = _grid((1, 1), (np.array([0.0, 4.0]),) * 2)
    region = MultiPolygon([box(3, 0, 4, 4), box(0, 0, 1, 4)])
    child, mapping = clip_binning(parent, region)
    assert child.shapes is child.locator.cells
    assert mapping is child.locator.mapping
    np.testing.assert_array_equal(mapping.parent_bin_indices, [0, 0])
    np.testing.assert_array_equal(child.bins.centers, [[0.5, 2], [3.5, 2]])
    np.testing.assert_array_equal(child.bins.sizes, [4, 4])
    assert len(child.connections.pairs) == 0
    assert child.faces.available
    reversed_child, _ = clip_binning(parent, MultiPolygon(list(reversed(region.geoms))))
    assert [p.wkb for p in child.shapes.polygons] == [
        p.wkb for p in reversed_child.shapes.polygons
    ]
    for restored in (copy.deepcopy(child), pickle.loads(pickle.dumps(child))):
        assert restored.shapes is restored.locator.cells
        np.testing.assert_array_equal(
            restored.locator.locate(restored.bins.centers), [0, 1]
        )


@pytest.mark.parametrize("region", [box(4, 0, 5, 4), box(4, 4, 5, 5)])
def test_zero_area_intersections_are_not_bins(region):
    with pytest.raises(ValueError, match="positive-area"):
        clip_binning(_grid((1, 1), (np.array([0.0, 4.0]),) * 2), region)


@pytest.mark.parametrize("width", [2.0**-40, 1e-100])
def test_every_positive_sliver_is_retained(width):
    parent = _grid((1, 1), (np.array([0.0, 4.0]),) * 2)
    child, _ = clip_binning(parent, box(0, 0, width, 4))
    assert child.bins.sizes.tolist() == [4 * width]
    np.testing.assert_array_equal(child.locator.locate(child.bins.centers), [0])


@pytest.mark.parametrize(
    "region",
    [
        box(0, 0, 4, 4).difference(box(1, 1, 3, 3)),
        Polygon([(0, 0), (4, 0), (4, 4), (3, 4), (3, 1), (1, 1), (1, 4), (0, 4)]),
    ],
)
def test_holed_and_nonconvex_cells_keep_exact_geometry_and_interior_anchor(region):
    parent = _grid((1, 1), (np.array([0.0, 4.0]),) * 2)
    child, _ = clip_binning(parent, region)
    assert not region.contains(region.centroid)
    assert child.shapes.polygons[0].equals(region)
    assert region.contains(Point(child.bins.centers[0]))
    assert not child.faces.available
    assert "holed or nonconvex" in child.faces.reason
    np.testing.assert_array_equal(child.locator.locate(child.bins.centers), [0])


def test_rectangular_clip_has_analytic_faces_and_norm_distances():
    child, _ = clip_binning(_grid(), box(0.25, 0, 2, 1))
    assert child.faces.available
    faces = child.build_faces()
    np.testing.assert_array_equal(child.bins.centers, [[0.625, 0.5], [1.5, 0.5]])
    np.testing.assert_array_equal(faces.volumes, [0.75, 1])
    np.testing.assert_array_equal(faces.A, [1])
    np.testing.assert_array_equal(faces.distance, [0.875])


def test_sloping_clip_centroids_refuse_spurious_normal_flux():
    child, _ = clip_binning(_grid(), Polygon([(0, 0), (2, 0), (2, 0.75), (0, 0.25)]))
    np.testing.assert_allclose(child.bins.sizes, [3 / 8, 5 / 8])
    np.testing.assert_allclose(
        child.bins.centers, [[5 / 9, 7 / 36], [23 / 15, 19 / 60]]
    )
    assert not child.faces.available
    assert "orthogonality" in child.faces.reason


def test_multiple_openings_are_not_certified_by_parallel_center_link():
    cells = PolygonCells((box(0, 0, 1, 3), box(1, 0, 2, 3)))
    faces = certify_faces(
        cells,
        np.array([[0.5, 1.5], [1.5, 1.5]]),
        np.array([[0, 1]]),
        walls=(LineString([(1, 1), (1, 2)]),),
    )
    assert not faces.available
    assert "single connected" in faces.reason


def test_full_physical_wall_closes_pair_without_changing_cell_area():
    child, _ = clip_binning(
        _grid(), box(0, 0, 2, 1), walls=(LineString([(1, 0), (1, 1)]),)
    )
    assert len(child.connections.pairs) == 0
    np.testing.assert_array_equal(child.connections.all_pairs, [[0, 1]])
    np.testing.assert_array_equal(child.connections.is_open, [False])
    np.testing.assert_array_equal(child.bins.sizes, [1, 1])
    assert child.faces.available


def test_parent_path_only_faces_stay_zero_and_missing_roles_stay_missing():
    parent = _grid((2, 2), diagonals=True)
    child, _ = clip_binning(parent, box(0, 0, 2, 2))
    np.testing.assert_array_equal(child.connections.pairs, parent.connections.pairs)
    np.testing.assert_array_equal(child.build_faces().A, parent.build_faces().A)
    missing = replace(parent, faces=MissingFaces("Unknown source flux roles"))
    uncertified, _ = clip_binning(missing, box(0, 0, 2, 2))
    np.testing.assert_array_equal(
        uncertified.connections.pairs, parent.connections.pairs
    )
    assert not uncertified.faces.available
    assert "inherited path-only" in uncertified.faces.reason


def test_unlocated_partial_parent_conductance_is_not_promoted():
    parent = _grid()
    faces = parent.build_faces()
    parent = replace(
        parent,
        faces=KnownFaces(
            FaceGeometry(faces.edges, np.array([0.5]), faces.distance, faces.volumes)
        ),
    )
    child, _ = clip_binning(parent, box(0, 0, 2, 1))
    assert not child.faces.available
    assert "unlocated partial" in child.faces.reason


def test_domain_is_intersected_and_repeated_clips_keep_immediate_parent_map():
    parent = _grid((2, 2), (np.arange(3.0) * 2,) * 2)
    region = Polygon(
        [(0.25, 0.15), (3.8, 0.45), (3.45, 3.8), (0.2, 3.2)],
        holes=[[(1, 1), (2.6, 1.1), (2.4, 2.8), (0.9, 2.7)]],
    )
    first, _ = clip_binning(parent, region, domain=box(0, 0, 4, 3.7))
    second_region = MultiPolygon([box(0, 0, 1.9, 4), box(2.1, 0, 4, 4)])
    second, mapping = clip_binning(first, second_region)
    expected = (
        unary_union(materialize_cells(parent))
        .intersection(region)
        .intersection(box(0, 0, 4, 3.7))
        .intersection(second_region)
    )
    actual = unary_union(second.shapes.polygons)
    assert actual.symmetric_difference(expected).area < 1e-13
    assert mapping.n_parent_bins == first.bins.n_bins
    assert second.bins.sizes.sum() == pytest.approx(expected.area)


def test_transform_polygon_cells_maps_shared_vertices_and_keeps_holes():
    cells = PolygonCells((box(0, 0, 1, 1), box(1, 0, 2, 1)))
    matrix = np.array([[1.0, 0.25, 3.0], [0.0, -2.0, -1.0], [0.0, 0.0, 1.0]])
    transformed = transform_polygon_cells(cells, matrix)
    assert transformed.polygons[0].intersection(transformed.polygons[1]).area == 0
    assert transformed.polygons[0].boundary.intersection(
        transformed.polygons[1].boundary
    ).length == pytest.approx(np.hypot(0.25, 2))
    np.testing.assert_allclose([p.area for p in transformed.polygons], [2, 2])
    assert cells.polygons[0].bounds == (0, 0, 1, 1)


@pytest.mark.parametrize(
    "polygon", [Point(0, 0), Polygon(), Polygon([(0, 0), (1, 1), (1, 0), (0, 1)])]
)
def test_clipping_rejects_invalid_geometry_without_repair(polygon):
    with pytest.raises(ValueError):
        clip_binning(_grid(), polygon)


@pytest.mark.parametrize("built_environment", ["hex"], indirect=True)
def test_materialized_wrapped_hex_cells_keep_shared_transformed_vertices(
    built_environment,
):
    parent = built_environment._binning
    angle = 0.3
    matrix = np.array(
        [
            [1.5 * np.cos(angle), -1.5 * np.sin(angle), 10.0],
            [1.5 * np.sin(angle), 1.5 * np.cos(angle), -3.0],
            [0, 0, 1],
        ]
    )
    keep = np.arange(parent.bins.n_bins) % 3 != 1
    expected = transform_polygon_cells(PolygonCells(materialize_cells(parent)), matrix)
    selected = subset_binning(transform_binning(parent, AffineND(matrix)), keep)
    actual = materialize_cells(selected)
    for polygon, wanted in zip(actual, expected.subset(keep).polygons, strict=True):
        assert polygon.equals_exact(wanted, 1e-13)
    for u, v in selected.connections.pairs:
        assert actual[u].intersection(actual[v]).area == 0
        assert actual[u].boundary.intersection(actual[v].boundary).length > 0


@pytest.mark.parametrize("origin", [0.0, 1e3, 1e5])
@pytest.mark.parametrize("angle", [0.0, 0.3, 2.8])
def test_recentred_hexagon_cells_are_the_bins(origin, angle):
    """Hexagons recorded far from the origin keep their cells when recentred.

    Moving tracker coordinates so the arena center is at (0, 0) carries the
    original coordinates' rounding into the small new ones; the cells refused
    with "Canonical hexagon centers disagree with published bin centers".
    """
    from neurospatial.layout import create_layout
    from neurospatial.ops.transforms import AffineND

    env = Environment.from_layout(
        create_layout(
            "Hexagonal",
            hexagon_width=1.0,
            dimension_ranges=[(origin, origin + 30), (origin, origin + 30)],
        )
    )
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    shift = -rotation @ np.full(2, origin + 15.0)
    moved = env.apply_transform(
        AffineND(np.block([[rotation, shift[:, None]], [0, 0, 1]]))
    )
    cells = materialize_cells(moved._binning)
    centroids = np.array([cell.centroid.coords[0] for cell in cells])
    np.testing.assert_allclose(centroids, moved.bin_centers, rtol=0, atol=1e-9)
    np.testing.assert_allclose([cell.area for cell in cells], moved.bin_sizes)


def test_centroid_centers_do_not_depend_on_clip_history():
    parent = _grid((1, 1))
    first, _ = clip_binning(parent, box(0, 0, 0.4, 1))
    sequential, _ = clip_binning(first, box(0, 0, 1, 0.8))
    direct, _ = clip_binning(parent, box(0, 0, 0.4, 0.8))
    np.testing.assert_allclose(
        sequential.bins.centers, [[0.2, 0.4]], rtol=0, atol=1e-15
    )
    np.testing.assert_allclose(
        sequential.bins.centers, direct.bins.centers, rtol=0, atol=1e-15
    )


@pytest.mark.parametrize("origin,scale", [(0.0, 1.0), (1e9, 1.0), (0.0, 1e-8)])
def test_numerical_certificate_uses_local_scales_and_rejects_actual_skew(origin, scale):
    cells = PolygonCells(
        (
            box(origin, origin, origin + scale, origin + scale),
            box(origin + scale, origin, origin + 2 * scale, origin + scale),
        )
    )
    centers = np.array(
        [
            [origin + 0.5 * scale, origin + 0.5 * scale],
            [origin + 1.5 * scale, origin + 0.5 * scale],
        ]
    )
    pairs = np.array([[0, 1]], dtype=np.intp)
    assert certify_faces(cells, centers, pairs).available
    centers[1, 1] += 1e-3 * scale
    assert not certify_faces(cells, centers, pairs).available


def test_numerical_certificate_does_not_hide_resolvable_skew_at_large_origin():
    origin = 1e9
    cells = PolygonCells(
        (
            box(origin, origin, origin + 1, origin + 1),
            box(origin + 1, origin, origin + 2, origin + 1),
        )
    )
    centers = np.array(
        [[origin + 0.5, origin + 0.5], [origin + 1.5, origin + 0.5 + 1e-6]]
    )
    assert not certify_faces(cells, centers, np.array([[0, 1]])).available


@pytest.mark.parametrize("angle", [0.3, 0.7])
def test_numerical_certificate_keeps_rotated_rectangles_at_large_origin(angle):
    matrix = np.array(
        [
            [np.cos(angle), -np.sin(angle), 1e9],
            [np.sin(angle), np.cos(angle), -1e9],
            [0.0, 0.0, 1.0],
        ]
    )
    cells = transform_polygon_cells(
        PolygonCells((box(0, 0, 1, 1), box(1, 0, 2, 1))), matrix
    )
    centers = np.array([[0.5, 0.5], [1.5, 0.5]]) @ matrix[:2, :2].T + matrix[:2, 2]
    assert certify_faces(cells, centers, np.array([[0, 1]])).available


def test_partial_parent_conductance_is_not_hidden_by_large_origin():
    parent = _grid(edges=(np.arange(3.0) + 1e9, np.arange(2.0) + 1e9))
    faces = parent.build_faces()
    parent = replace(
        parent,
        faces=KnownFaces(
            FaceGeometry(
                faces.edges, np.array([1.0 - 1e-5]), faces.distance, faces.volumes
            )
        ),
    )
    child, _ = clip_binning(parent, box(1e9, 1e9, 1e9 + 2, 1e9 + 1))
    assert not child.faces.available
    assert "unlocated partial" in child.faces.reason


@pytest.mark.parametrize("angle", [0.3, 0.37, 0.7])
@pytest.mark.parametrize("offset", [0.0, 1e9])
def test_polygon_transform_nodes_shared_boundary_before_mapping(angle, offset):
    cells = PolygonCells(
        (box(0, 0, 1, 1), Polygon([(1, 0), (2, 0), (2, 1), (1, 1), (1, 0.5)]))
    )
    matrix = np.array(
        [
            [np.cos(angle), -np.sin(angle), offset],
            [np.sin(angle), np.cos(angle), -offset],
            [0, 0, 1.0],
        ]
    )
    transformed = transform_polygon_cells(cells, matrix)
    first, second = transformed.polygons
    assert first.intersection(second).area == 0
    assert first.boundary.intersection(second.boundary).length == pytest.approx(
        1.0, abs=1e-7
    )


@pytest.mark.parametrize(
    "linear",
    [
        np.array([[0.0, -1.0], [1.0, 0.0]]),
        np.diag([-1.0, 1.0]),
        np.array([[1.0, 0.25], [0.0, 1.0]]),
    ],
)
def test_materialize_transformed_grid_from_stored_locator(linear):
    parent = _grid((2, 2))
    matrix = np.eye(3)
    matrix[:2, :2] = linear
    child = transform_binning(parent, AffineND(matrix))
    assert child.rectilinear is None
    expected = transform_polygon_cells(PolygonCells(materialize_cells(parent)), matrix)
    actual = materialize_cells(child)
    assert all(a.equals(b) for a, b in zip(actual, expected.polygons, strict=True))
    assert all(
        a.equals(b)
        for a, b in zip(cells_from_locator(child.locator), actual, strict=True)
    )


def test_domain_intersection_discards_only_zero_area_collection_parts():
    parent = _grid((1, 1), (np.array([0.0, 3.0]), np.array([0.0, 1.0])))
    region = MultiPolygon([box(0, 0, 1, 1), box(2, 0, 3, 1)])
    child, _ = clip_binning(parent, region, domain=box(0, 0, 2, 1))
    assert len(child.shapes.polygons) == 1
    assert child.shapes.polygons[0].equals(box(0, 0, 1, 1))


def test_mapping_refuses_unrepresentable_parent_count_before_integer_cast():
    with pytest.raises(ValueError, match="representable"):
        ClipMapping(np.array([2**63], dtype=np.uint64), 2**64)


SLANTED_ARENA = Polygon([(0, 0), (90, 0), (90, 40), (50, 40), (60, 90), (0, 70)])


@pytest.mark.parametrize(
    "clip",
    [box(-5, -5, 95, 95), box(20, 0, 95, 95), Point(45, 40).buffer(30)],
    ids=["whole", "box", "disk"],
)
def test_clip_keeps_the_arena_a_grid_drops_at_its_outline(clip):
    # A grid drops the cells whose centers lie outside a slanted outline and
    # gives a position in their arena part to the nearest neighboring bin. The
    # clipped cells take that arena, so a clip keeps every position the grid
    # gives a bin, in a piece of that bin, and covers the clipped arena.
    env = Environment.from_polygon(SLANTED_ARENA, bin_size=3.0, units="cm")
    clipped = env.clip(clip)
    points = np.random.default_rng(0).uniform(0, 90, (40_000, 2))
    target = SLANTED_ARENA.intersection(clip)
    points = points[shapely.contains_xy(target, *points.T)]
    grid_bins = env.bin_at(points)
    clipped_bins = clipped.bin_at(points)
    assert np.all(grid_bins >= 0)
    assert np.all(clipped_bins >= 0)
    np.testing.assert_array_equal(
        clipped.clip_mapping.parent_bin_indices[clipped_bins], grid_bins
    )
    area = sum(cell.area for cell in materialize_cells(clipped.binning))
    assert area == pytest.approx(target.area, rel=1e-12)
