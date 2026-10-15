"""Validate explicit binning and expose it through the normal publication path."""

from __future__ import annotations

import copy
from dataclasses import replace

import networkx as nx
import numpy as np

from neurospatial.environment._ownership import _check_binning, _snapshot_binning
from neurospatial.layout._binning import (
    Binning,
    KnownFaces,
    MissingFaces,
    SpatialBins,
)
from neurospatial.layout._components import HexagonShapes, TriangleShapes
from neurospatial.layout._graph import _graph_from_binning
from neurospatial.layout._polygon import PolygonCells


def _category(binning: Binning) -> str:
    if binning.linearization is not None:
        return "graph"
    if isinstance(binning.shapes, HexagonShapes):
        return "hexagonal"
    if isinstance(binning.shapes, TriangleShapes):
        return "mesh"
    if isinstance(binning.shapes, PolygonCells) or binning.outline is not None:
        return "polygon"
    return "grid" if binning.rectilinear is not None else "other"


def _plot_ranges(binning: Binning) -> tuple[tuple[float, float], ...] | None:
    """Use actual retained geometry for descriptive plotting bounds only."""
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
        low, high = shapes.points.min(axis=0), shapes.points.max(axis=0)
    else:
        # Centers and scalar measures do not establish a cell extent.
        return None
    return tuple((float(a), float(b)) for a, b in zip(low, high, strict=True))


def prepare_bins(
    binning: object,
) -> tuple[Binning, nx.Graph, tuple[tuple[float, float], ...] | None, str]:
    """Own and validate explicit Cartesian bins, with no layout builder.

    Parameters
    ----------
    binning : Binning
        Supported data records with explicit lookup and KnownFaces or
        MissingFaces. Neighbor pair order and orientation remain authoritative.

    Returns
    -------
    tuple
        Validated data and a synthesized navigation graph. Graph iteration
        order can differ from the retained pair order.

    Raises
    ------
    TypeError
        If a runtime provider or an implicit face source was supplied.
    ValueError
        If records or their shared geometry disagree.
    """
    if type(binning) is not Binning:
        from neurospatial.extensions import _snapshot_provider

        binning = _snapshot_provider(binning, "binning")
    assert isinstance(binning, Binning)
    binning = _snapshot_binning(binning)
    if type(binning.faces) not in (KnownFaces, MissingFaces):
        raise TypeError(
            "from_bins requires explicit KnownFaces or MissingFaces; "
            "materialize valid faces before constructing an environment."
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
    graph = _graph_from_binning(snapshot)
    return snapshot, graph, _plot_ranges(snapshot), _category(snapshot)
