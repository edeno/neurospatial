"""Validate explicit binning and expose it through the normal publication path."""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Literal

import numpy as np

from neurospatial._exceptions import _format_error
from neurospatial.geometry._binning import (
    Binning,
    KnownFaces,
    MissingFaces,
    SpatialBins,
)
from neurospatial.geometry._components import HexagonShapes, TriangleShapes
from neurospatial.geometry._ownership import _check_binning, _snapshot_binning
from neurospatial.geometry._polygon import PolygonCells


def cell_extent(
    binning: Binning, *, shapeless: Literal["none", "centers"]
) -> tuple[tuple[float, float], ...] | None:
    """Bound the retained cells, one ``(low, high)`` per axis.

    Parameters
    ----------
    binning : Binning
        Bins whose cells to bound.
    shapeless : {"none", "centers"}
        For bins without cell shapes, return None ("none": centers and scalar
        measures do not establish a cell extent) or the bounds of the bin
        centers ("centers").

    Returns
    -------
    tuple of (float, float), or None
        Grid extent for grids (full lattice, including inactive cells), the
        outline's bounds when one is stored, otherwise the bounds of the
        cells themselves: polygon cells, hexagons, or the vertices of the
        retained triangles. None for an empty binning.
    """
    if binning.rectilinear is not None:
        return tuple(
            (float(edge[0]), float(edge[-1])) for edge in binning.rectilinear.edges
        )
    if binning.grid_frame is not None:
        corners = binning.grid_frame.corners()
        return tuple((float(axis.min()), float(axis.max())) for axis in corners.T)
    if binning.bins.n_bins == 0:
        return None
    if binning.outline is not None:
        low_x, low_y, high_x, high_y = binning.outline.geometry.bounds
        return ((low_x, high_x), (low_y, high_y))
    shapes = binning.shapes
    if isinstance(shapes, PolygonCells):
        bounds = np.array([polygon.bounds for polygon in shapes.polygons])
        low, high = bounds[:, :2].min(axis=0), bounds[:, 2:].max(axis=0)
    elif isinstance(shapes, HexagonShapes):
        angles = np.arange(6) * np.pi / 3 + np.pi / 2 + shapes.orientation
        offsets = shapes.radius * np.column_stack((np.cos(angles), np.sin(angles)))
        low = shapes.centers.min(axis=0) + offsets.min(axis=0)
        high = shapes.centers.max(axis=0) + offsets.max(axis=0)
    elif isinstance(shapes, TriangleShapes):
        # Only the retained triangles: a subset keeps every mesh point.
        vertices = shapes.points[np.unique(shapes.triangles)]
        low, high = vertices.min(axis=0), vertices.max(axis=0)
    elif shapeless == "centers":
        centers = binning.bins.centers
        low, high = centers.min(axis=0), centers.max(axis=0)
    else:
        return None
    return tuple((float(a), float(b)) for a, b in zip(low, high, strict=True))


def binning_ranges(binning: Binning) -> tuple[tuple[float, float], ...] | None:
    """Return the binning's own ``dimension_ranges``, else its cell extent.

    Parameters
    ----------
    binning : Binning
        Bins whose coordinate ranges to report.

    Returns
    -------
    tuple of (float, float), or None
        ``binning.dimension_ranges`` when the binning records it; otherwise
        the extent of its cells (see :func:`cell_extent`), or None for bins
        without cells.
    """
    if binning.dimension_ranges is not None:
        return binning.dimension_ranges
    return cell_extent(binning, shapeless="none")


def prepare_bins(binning: object) -> Binning:
    """Own and validate explicit Cartesian bins, with no layout builder.

    Parameters
    ----------
    binning : Binning
        Supported data records with explicit lookup and KnownFaces or
        MissingFaces. Neighbor pair order and orientation remain authoritative.

    Returns
    -------
    Binning
        Validated, independently owned bins.

    Raises
    ------
    TypeError
        If a runtime provider or an implicit face source was supplied.
    ValueError
        If records or their shared geometry disagree.
    """
    if type(binning) is not Binning:
        from neurospatial.geometry._adapters import _snapshot_provider

        binning = _snapshot_provider(binning, "binning")
    assert isinstance(binning, Binning)
    binning = _snapshot_binning(binning)
    if type(binning.faces) not in (KnownFaces, MissingFaces):
        raise TypeError(
            _format_error(
                "from_bins() needs the cell faces themselves, but this Binning "
                "holds a recipe that builds them, as env.binning returns.",
                why="Why: from_bins() checks the faces against the bins it is "
                "given and does not run a layout's recipe, which could describe "
                "other bins.",
                fix="build the faces first: from dataclasses import replace; "
                "from neurospatial.binning import KnownFaces; "
                "binning = replace(binning, faces=KnownFaces(binning.faces.build())); "
                "Environment.from_bins(binning, arena=env.arena)",
            )
        )
    _check_binning(binning)
    if not isinstance(binning.kind, str):
        raise TypeError("Binning.kind must be a descriptive string.")
    if isinstance(binning.faces, MissingFaces) and not isinstance(
        binning.faces.reason, str
    ):
        raise TypeError("MissingFaces.reason must be a string.")
    snapshot = copy.deepcopy(binning)
    raw = snapshot.bins
    # Standalone records have writable NumPy headers; revalidate at ingress
    # rather than assuming they remain identical to their constructor inputs.
    bins = SpatialBins.from_arrays(
        raw.centers, raw.sizes, raw.measure_dim, raw.pairs, raw.pair_distances
    )
    faces = (
        KnownFaces(snapshot.faces.faces)
        if isinstance(snapshot.faces, KnownFaces)
        else snapshot.faces
    )
    snapshot = replace(snapshot, bins=bins, faces=faces)
    return snapshot
