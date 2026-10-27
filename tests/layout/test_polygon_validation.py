"""Polygon publication checks current geometry and immediate clip ancestry."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from shapely.affinity import translate
from shapely.geometry import Polygon, box

from neurospatial.geometry._binning import (
    BinConnections,
    Binning,
    KnownFaces,
    MissingFaces,
    SpatialBins,
)
from neurospatial.geometry._coordinates import CartesianSystem
from neurospatial.geometry._derive import subset_binning, transform_binning
from neurospatial.geometry._locators import RectilinearLocator
from neurospatial.geometry._polygon import _representative, certify_faces
from neurospatial.geometry._polygon_cells import (
    ClipMapping,
    ClippedLocator,
    PolygonCells,
    _polygon_coordinate_errors,
    clip_mapping_of,
)
from neurospatial.geometry._validation import validate_components


def _binning(polygons, *, pairs=(), known=False):
    cells = PolygonCells(tuple(polygons))
    lower = np.min([polygon.bounds[:2] for polygon in polygons], axis=0)
    upper = np.max([polygon.bounds[2:] for polygon in polygons], axis=0)
    parent = RectilinearLocator(
        tuple(np.array([lo, hi]) for lo, hi in zip(lower, upper, strict=True)),
        (1, 1),
        np.ones((1, 1), bool),
    )
    mapping = ClipMapping(np.zeros(len(polygons), dtype=int), 1)
    centers = np.array([_representative(polygon) for polygon in polygons])
    edges = np.array(pairs, dtype=int).reshape(-1, 2)
    distances = np.hypot.reduce(centers[edges[:, 1]] - centers[edges[:, 0]], axis=1)
    bins = SpatialBins(centers, np.array([p.area for p in polygons]), 2)
    faces = certify_faces(cells, centers, edges) if known else MissingFaces("fixture")
    return Binning(
        "fixture",
        bins,
        BinConnections.from_arrays(edges, distances),
        ClippedLocator(parent, cells, mapping),
        faces,
        shapes=cells,
    )


def _publish(binning):
    validate_components(binning, CartesianSystem())


def _matrix(angle=0.0, offset=0.0):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, offset], [s, c, -offset], [0.0, 0.0, 1.0]])


def test_origin_zero_positive_sliver_is_resolved_along_its_thin_axis():
    _publish(_binning([box(0, 0, 1e-12, 1)]))


@pytest.mark.parametrize("width", [1e-6, 1e-4])
def test_thin_polygon_transform_refuses_unresolved_large_origin(width):
    source = _binning([box(0, 0, width, 1)])
    transformed = transform_binning(source, SimpleNamespace(A=_matrix(offset=1e9)))
    with pytest.raises(ValueError, match=r"resol|precision|represent"):
        _publish(transformed)


def test_current_polygon_shift_cannot_hide_inside_transform_error_budget():
    width = 1e-6
    polygon = Polygon(
        [
            (0, 0),
            (width, 0),
            (width, 0.5),
            (2 * width, 0.5),
            (2 * width, 1),
            (0, 1),
        ]
    )
    source = _binning([polygon])
    transformed = transform_binning(source, SimpleNamespace(A=_matrix(offset=1e9)))
    corrupted = replace(
        transformed,
        shapes=PolygonCells((translate(transformed.shapes.polygons[0], xoff=1e-5),)),
    )
    with pytest.raises(ValueError, match=r"polygon|resol|precision|represent"):
        _publish(corrupted)


@pytest.mark.parametrize("angle", [0.3, 0.7])
def test_mapped_faces_allow_coordinate_rounding_without_changing_geometry(angle):
    source = _binning([box(0, 0, 1, 1), box(1, 0, 2, 1)], pairs=[(0, 1)], known=True)
    transformed = transform_binning(source, SimpleNamespace(A=_matrix(angle, 1e9)))
    _publish(transformed)
    corrupted = replace(
        transformed,
        faces=replace(
            transformed.faces,
            faces=transformed.faces.faces._replace(
                A=transformed.faces.faces.A + 1e-3,
            ),
        ),
    )
    with pytest.raises(ValueError, match=r"faces\.A"):
        _publish(corrupted)


def _twice_clipped():
    source = _binning([box(0, 0, 1, 4), box(3, 0, 4, 4)])
    cells = PolygonCells((box(0, 0, 1, 2), box(3, 2, 4, 4)))
    mapping = ClipMapping(np.array([0, 1]), 2)
    bins = replace(
        source.bins,
        centers=np.array([[0.5, 1.0], [3.5, 3.0]]),
        sizes=np.array([2.0, 2.0]),
    )
    return replace(
        source,
        bins=bins,
        shapes=cells,
        locator=ClippedLocator(source.locator, cells, mapping),
    )


@pytest.mark.parametrize(
    ("operation", "expected"),
    [("none", [0, 1]), ("subset", [1]), ("transform", [0, 1]), ("both", [1])],
)
def test_clip_mapping_follows_latest_clip_through_wrappers(operation, expected):
    result = _twice_clipped()
    if operation in ("subset", "both"):
        result = subset_binning(result, np.array([False, True]))
    if operation in ("transform", "both"):
        result = transform_binning(result, SimpleNamespace(A=_matrix(0.3)))
    _publish(result)
    mapping = clip_mapping_of(result.locator)
    assert mapping is not None
    assert mapping.n_parent_bins == 2
    np.testing.assert_array_equal(mapping.parent_bin_indices, expected)


def test_clip_mapping_is_none_without_a_clip():
    source = _binning([box(0, 0, 1, 1)])
    assert clip_mapping_of(source.locator.base) is None


def test_stored_zero_face_cannot_opt_out_of_positive_opening_validation():
    source = _binning([box(0, 0, 1, 1), box(1, 0, 2, 1)], pairs=[(0, 1)], known=True)
    _publish(source)
    changed = KnownFaces(source.faces.faces._replace(A=np.array([0.0])))
    with pytest.raises(ValueError, match=r"faces\.A"):
        _publish(replace(source, faces=changed))


def _over_grid(polygons, parents, active):
    """Polygon cells over a 3x1 grid of unit cells with ``active`` kept."""
    cells = PolygonCells(tuple(polygons))
    base = RectilinearLocator(
        (np.arange(4.0), np.array([0.0, 1.0])), (3, 1), np.array(active)[:, None]
    )
    centers = np.array([_representative(polygon) for polygon in polygons])
    return Binning(
        "fixture",
        SpatialBins(centers, np.array([p.area for p in polygons]), 2),
        BinConnections.from_arrays(np.empty((0, 2), int), np.empty(0)),
        ClippedLocator(base, cells, ClipMapping(np.array(parents), sum(active))),
        MissingFaces("fixture"),
        shapes=cells,
    )


def test_a_cell_may_take_unbinned_space_next_to_its_parent():
    # Splitting at walls gives a bin the arena of a grid cell dropped next to
    # it; space another bin holds, or space farther away, is refused.
    _publish(_over_grid([box(0, 0, 1.5, 1)], [0], [True, False, False]))
    with pytest.raises(ValueError, match="outside its source parent"):
        _publish(
            _over_grid(
                [box(0, 0, 1.5, 1), box(1.5, 0, 2, 1)], [0, 1], [True, True, False]
            )
        )
    with pytest.raises(ValueError, match="outside its source parent"):
        _publish(_over_grid([box(0, 0, 2.5, 1)], [0], [True, False, False]))


@pytest.mark.parametrize("start", range(6))
def test_resolvability_does_not_depend_on_where_the_ring_starts(start):
    # Overlay can leave two vertices a rounding error apart at a corner (here
    # where a slanted edge crosses a grid vertex); they are not a feature.
    ring = [(33, 78), (30, 78), (27, 79), (30, 80), (32.99999999999999, 81), (33, 81)]
    _polygon_coordinate_errors(Polygon(ring[start:] + ring[:start]))
