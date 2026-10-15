"""Validate explicit binning and expose it through the normal publication path."""

from __future__ import annotations

import copy
from dataclasses import replace

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from neurospatial.environment._ownership import _check_binning, _snapshot_binning
from neurospatial.layout._binning import (
    Binning,
    KnownFaces,
    Locator,
    MissingFaces,
    SpatialBins,
    TrackLinearization,
)
from neurospatial.layout._components import HexagonShapes, TriangleShapes
from neurospatial.layout._locators import (
    HexLocator,
    RectilinearLocator,
    SubsetLocator,
    TransformedLocator,
)
from neurospatial.layout._polygon import ClippedLocator, PolygonCells


def _source_grid(locator: Locator) -> tuple[tuple[int, ...], NDArray[np.intp]] | None:
    """Recover logical cell IDs in active-bin order without fitting geometry."""
    if isinstance(locator, ClippedLocator):
        # Clipping creates new cells. Immediate-parent ancestry lives in the
        # separate ClipMapping, rather than in logical grid coordinates.
        count = len(locator.cells.polygons)
        return (count,), np.arange(count, dtype=np.intp)
    if isinstance(locator, (TransformedLocator, SubsetLocator)):
        source = _source_grid(locator.base)
        if source is None:
            return None
        shape, indices = source
        if isinstance(locator, SubsetLocator):
            selected = np.flatnonzero(locator.parent_to_child >= 0)
            selected = selected[np.argsort(locator.parent_to_child[selected])]
            indices = indices[selected]
        return shape, indices
    if isinstance(locator, RectilinearLocator):
        shape = locator.shape
        mask = locator.active_mask
        active = (
            np.arange(np.prod(shape), dtype=np.intp)
            if mask is None
            else np.flatnonzero(mask.ravel())
        )
        mapping = locator.flat_to_active
    elif isinstance(locator, HexLocator):
        shape = locator.centers_shape
        mapping = locator.flat_to_active
        active = np.flatnonzero(mapping >= 0)
    else:
        return None
    if mapping is not None:
        active = active[np.argsort(mapping[active])]
    return shape, active.astype(np.intp, copy=False)


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
    graph = nx.Graph()
    for index, center in enumerate(bins.centers):
        graph.add_node(
            index,
            pos=tuple(float(value) for value in center),
            source_grid_flat_index=index,
            original_grid_nd_index=(index,),
        )
    for index, ((a, b), distance) in enumerate(
        zip(bins.pairs, bins.pair_distances, strict=True)
    ):
        # Graph nodes iterate in bin ID order. Keep its vector in that graph
        # orientation; edge_id links it to the explicit pair-array authority.
        u, v = sorted((int(a), int(b)))
        try:
            with np.errstate(over="raise", invalid="raise"):
                vector = bins.centers[v] - bins.centers[u]
        except FloatingPointError as exc:
            raise ValueError(
                "Pair displacement exceeds floating-point range; recenter coordinates."
            ) from exc
        graph.add_edge(
            u,
            v,
            distance=float(distance),
            vector=tuple(float(value) for value in vector),
            edge_id=index,
        )
        if bins.n_dims == 2:
            graph.edges[u, v]["angle_2d"] = float(np.arctan2(vector[1], vector[0]))
    source = _source_grid(snapshot.locator)
    if source is None and isinstance(snapshot.linearization, TrackLinearization):
        mask = snapshot.linearization.active_mask
        source = mask.shape, np.flatnonzero(mask)
    if source is not None:
        shape, indices = source
        for index, flat in enumerate(indices):
            graph.nodes[index].update(
                source_grid_flat_index=int(flat),
                original_grid_nd_index=tuple(
                    int(value) for value in np.unravel_index(flat, shape)
                ),
            )
    return snapshot, graph, _plot_ranges(snapshot), _category(snapshot)
