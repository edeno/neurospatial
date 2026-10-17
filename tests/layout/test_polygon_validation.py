"""Polygon publication checks current geometry and immediate clip ancestry."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from shapely.affinity import translate
from shapely.geometry import Polygon, box

from neurospatial.geometry._binning import (
    Binning,
    KnownFaces,
    MissingFaces,
    SpatialBins,
)
from neurospatial.geometry._derive import subset_binning, transform_binning
from neurospatial.geometry._locators import RectilinearLocator
from neurospatial.geometry._polygon import _representative, certify_faces
from neurospatial.geometry._polygon_cells import (
    ClipMapping,
    ClippedLocator,
    PolygonCells,
)
from neurospatial.geometry._validation import _validate_components


def _binning(polygons, *, pairs=(), known=False):
    cells = PolygonCells(tuple(polygons))
    lower = np.min([polygon.bounds[:2] for polygon in polygons], axis=0)
    upper = np.max([polygon.bounds[2:] for polygon in polygons], axis=0)
    parent = RectilinearLocator(
        tuple(np.array([lo, hi]) for lo, hi in zip(lower, upper, strict=True)),
        (1, 1),
        None,
        None,
    )
    mapping = ClipMapping(np.zeros(len(polygons), dtype=int), 1)
    centers = np.array([_representative(polygon) for polygon in polygons])
    edges = np.array(pairs, dtype=int).reshape(-1, 2)
    distances = np.hypot.reduce(centers[edges[:, 1]] - centers[edges[:, 0]], axis=1)
    bins = SpatialBins(
        centers, np.array([p.area for p in polygons]), 2, edges, distances
    )
    faces = certify_faces(cells, centers, edges) if known else MissingFaces("fixture")
    return Binning(
        "fixture",
        bins,
        ClippedLocator(parent, cells, mapping),
        faces,
        shapes=cells,
        clip_mapping=mapping,
    )


def _publish(binning):
    _validate_components(binning, False, "cartesian")


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
        clip_mapping=mapping,
        locator=ClippedLocator(source.locator, cells, mapping),
    )


@pytest.mark.parametrize("operation", ["none", "subset", "transform", "both"])
@pytest.mark.parametrize("corruption", ["missing", "historical", "parent_count"])
def test_current_clip_mapping_matches_latest_locator_through_wrappers(
    operation, corruption
):
    original = _twice_clipped()
    result = original
    if operation in ("subset", "both"):
        result = subset_binning(result, np.array([False, True]))
    if operation in ("transform", "both"):
        result = transform_binning(result, SimpleNamespace(A=_matrix(0.3)))
    _publish(result)
    if corruption == "missing":
        mapping = None
    elif corruption == "historical":
        mapping = ClipMapping(np.zeros(result.bins.n_bins, dtype=int), 1)
    else:
        mapping = ClipMapping(result.clip_mapping.parent_bin_indices, 3)
    with pytest.raises(ValueError, match="clip_mapping"):
        _publish(replace(result, clip_mapping=mapping))


def test_unexpected_clip_mapping_without_clipped_locator_is_rejected():
    source = _binning([box(0, 0, 1, 1)])
    plain = replace(source, locator=source.locator.base, shapes=None)
    with pytest.raises(ValueError, match="clip_mapping"):
        _publish(plain)


def test_stored_zero_face_cannot_opt_out_of_positive_opening_validation():
    source = _binning([box(0, 0, 1, 1), box(1, 0, 2, 1)], pairs=[(0, 1)], known=True)
    _publish(source)
    changed = KnownFaces(source.faces.faces._replace(A=np.array([0.0])))
    with pytest.raises(ValueError, match=r"faces\.A"):
        _publish(replace(source, faces=changed))
