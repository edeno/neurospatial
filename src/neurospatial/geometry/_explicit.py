"""Validate explicit binning and expose it through the normal publication path."""

from __future__ import annotations

import copy
import itertools
from dataclasses import replace
from typing import Literal

import numpy as np

from neurospatial._exceptions import _format_error
from neurospatial.geometry._arrays import axis_ranges
from neurospatial.geometry._binning import (
    BinConnections,
    Binning,
    KnownFaces,
    MissingFaces,
    SpatialBins,
    TrackLinearization,
)
from neurospatial.geometry._capabilities import RectilinearGrid
from neurospatial.geometry._codecs import snapshot_provider
from neurospatial.geometry._components import (
    HexagonShapes,
    Outline,
    TriangleShapes,
)
from neurospatial.geometry._grid_frame import FramedGrid
from neurospatial.geometry._ownership import _check_binning, _snapshot_binning
from neurospatial.geometry._polygon_cells import PolygonCells
from neurospatial.geometry._track import track_cell_ends


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
        The extent of the cells themselves: the active cells of a grid (not
        its inactive border), or the cells described by an outline, shapes or
        a track (see :func:`component_extent`). None for an empty binning.
    """
    if binning.bins.n_bins == 0:
        return None
    if binning.rectilinear is not None:
        return _active_grid_extent(binning.rectilinear)
    if binning.grid_frame is not None:
        return _active_framed_extent(binning.grid_frame)
    extent = component_extent(
        outline=binning.outline,
        shapes=binning.shapes,
        linearization=binning.linearization,
    )
    if extent is not None or shapeless == "none":
        return extent
    centers = binning.bins.centers
    return tuple(
        (float(a), float(b))
        for a, b in zip(centers.min(axis=0), centers.max(axis=0), strict=True)
    )


def _active_box(grid: RectilinearGrid) -> list[tuple[float, float]]:
    """Per-axis edges that bound the active cells of a rectangular grid."""
    mask = np.asarray(grid.active_mask, dtype=bool)
    box = []
    for axis, edge in enumerate(grid.edges):
        others = tuple(other for other in range(mask.ndim) if other != axis)
        active = np.flatnonzero(mask.any(axis=others))
        box.append((float(edge[active[0]]), float(edge[active[-1] + 1])))
    return box


def _active_grid_extent(grid: RectilinearGrid) -> tuple[tuple[float, float], ...]:
    """Extent of the active cells of an axis-aligned grid."""
    return tuple(_active_box(grid))


def _active_framed_extent(frame: FramedGrid) -> tuple[tuple[float, float], ...]:
    """Extent of the active cells of a rotated, reflected or sheared grid."""
    grid = frame.source
    mask = np.asarray(grid.active_mask, dtype=bool)
    cells = np.argwhere(mask)
    corners = np.array(list(itertools.product((0, 1), repeat=mask.ndim)))
    source = np.stack(
        [
            np.asarray(edge)[cells[:, axis][:, None] + corners[None, :, axis]]
            for axis, edge in enumerate(grid.edges)
        ],
        axis=-1,
    ).reshape(-1, mask.ndim)
    world = frame.to_world(source)
    return axis_ranges(world.min(axis=0), world.max(axis=0))


def component_extent(
    *,
    outline: Outline | None = None,
    shapes: HexagonShapes | TriangleShapes | PolygonCells | None = None,
    linearization: TrackLinearization | None = None,
) -> tuple[tuple[float, float], ...] | None:
    """Bound the cells that an outline, cell shapes or a track describe.

    Layout engines report this as their ``dimension_ranges``, and derived
    bins report it for the cells they keep, so a parent and its subsets
    measure their extent the same way.

    Parameters
    ----------
    outline : Outline or None, optional
        Boundary of the cells; when given, its bounds are the extent.
    shapes : HexagonShapes, TriangleShapes or PolygonCells, optional
        The cells themselves: hexagon corners, the vertices of the retained
        triangles, or polygon bounds.
    linearization : TrackLinearization or None, optional
        A track: its cells are the stretches of centerline between linear
        bin edges, so the extent is the ends of its active cells.

    Returns
    -------
    tuple of (float, float), or None
        One ``(low, high)`` per axis, or None when nothing describes cells.
    """
    if outline is not None:
        low_x, low_y, high_x, high_y = outline.geometry.bounds
        return ((low_x, high_x), (low_y, high_y))
    if shapes is not None:
        return shapes.extent()
    if linearization is not None:
        ends = track_cell_ends(
            linearization.graph,
            linearization.edge_order,
            linearization.edge_spacing,
            linearization.grid_edges[0],
            linearization.active_mask,
        )
        return axis_ranges(ends.min(axis=(0, 1)), ends.max(axis=(0, 1)))
    return None


def binning_ranges(binning: Binning) -> tuple[tuple[float, float], ...] | None:
    """Return the extent of the binning's cells, else its declared ranges.

    Parameters
    ----------
    binning : Binning
        Bins whose coordinate ranges to report.

    Returns
    -------
    tuple of (float, float), or None
        The extent of the cells (see :func:`cell_extent`). Bins without cells,
        such as explicit point bins, keep ``binning.dimension_ranges``, which
        is None unless declared.
    """
    extent = cell_extent(binning, shapeless="none")
    return binning.dimension_ranges if extent is None else extent


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
        binning = snapshot_provider(binning, "binning")
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
    raw, links = snapshot.bins, snapshot.connections
    # Standalone records have writable NumPy headers; revalidate at ingress
    # rather than assuming they remain identical to their constructor inputs.
    bins = SpatialBins.from_arrays(raw.centers, raw.sizes, raw.measure_dim)
    connections = BinConnections.from_arrays(
        links.all_pairs, links.all_distances, links.is_open
    )
    faces = (
        KnownFaces(snapshot.faces.faces)
        if isinstance(snapshot.faces, KnownFaces)
        else snapshot.faces
    )
    snapshot = replace(snapshot, bins=bins, connections=connections, faces=faces)
    return snapshot
