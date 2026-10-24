"""Project authoritative binning data into standard navigation graph attributes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from neurospatial.geometry._arrays import active_order, get_centers
from neurospatial.geometry._binning import Binning, TrackLinearization
from neurospatial.geometry._coordinates import CartesianSystem, CoordinateSystem
from neurospatial.geometry._locators import (
    HexLocator,
    Locator,
    MeshLocator,
    RectilinearLocator,
    SubsetLocator,
    TransformedLocator,
)
from neurospatial.geometry._polygon_cells import ClippedLocator

_G = TypeVar("_G", bound=nx.Graph)


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
            if not isinstance(locator, SubsetLocator):
                return None
            # A subset of bins without logical cells keeps its parent's bin IDs.
            n_parent = len(locator.parent_to_child)
            source = (n_parent,), np.arange(n_parent, dtype=np.intp)
        shape, indices = source
        if isinstance(locator, SubsetLocator):
            selected = active_order(locator.parent_to_child)
            indices = indices[selected]
        return shape, indices
    if isinstance(locator, RectilinearLocator):
        return locator.shape, np.flatnonzero(locator.active_mask).astype(np.intp)
    if isinstance(locator, HexLocator):
        return locator.centers_shape, active_order(locator.flat_to_active)
    if isinstance(locator, MeshLocator):
        return (len(locator.simplices),), active_order(locator.simplex_to_active)
    return None


def _track_cell_segments(track: TrackLinearization) -> NDArray[np.intp] | None:
    """Return the track-graph edge ID of every linear cell, or None.

    Walks the segments in linear order the way a track layout bins them:
    each segment of positive length fills the cells up to its end, a
    zero-length segment gets one cell, and a positive gap after a segment is
    one inactive cell (ID -1). A segment ends at the cell edge its length
    reaches up to rounding: a rotated or shifted track recomputes its lengths
    from moved node positions, which differ from the stored edges in the last
    bits. Returns None when the stored cell edges do not follow that rule, so
    no segment can be assigned.
    """
    edges = track.grid_edges[0]
    n_cells = len(edges) - 1
    positions = nx.get_node_attributes(track.graph, "pos")
    edge_ids = {
        frozenset((u, v)): int(data["edge_id"])
        for u, v, data in track.graph.edges(data=True)
    }
    order = track.edge_order
    spacing = track.edge_spacing
    gaps = (
        np.full(max(0, len(order) - 1), float(spacing))
        if isinstance(spacing, float)
        else np.asarray(spacing, dtype=float)
    )
    if len(gaps) != max(0, len(order) - 1):
        return None
    segments = np.full(n_cells, -1, dtype=np.intp)
    cell = 0
    cursor = 0.0
    for index, (u, v) in enumerate(order):
        if u not in positions or v not in positions:
            return None
        length = float(
            np.linalg.norm(
                np.asarray(positions[v], dtype=float)
                - np.asarray(positions[u], dtype=float)
            )
        )
        # Rounding in the node positions, relative to their magnitude.
        tolerance = (
            64
            * np.finfo(float).eps
            * max(
                abs(cursor + length),
                *np.abs(np.asarray(positions[u], dtype=float)),
                *np.abs(np.asarray(positions[v], dtype=float)),
            )
        )
        count = (
            1
            if length <= 0.0
            else int(np.searchsorted(edges, cursor + length - tolerance, side="left"))
            - cell
        )
        if count < 1 or cell + count > n_cells:
            return None
        segments[cell : cell + count] = edge_ids.get(frozenset((u, v)), -1)
        cell += count
        cursor += length
        if index < len(order) - 1 and gaps[index] > 0.0:
            cell += 1
            cursor += float(gaps[index])
    return segments if cell == n_cells else None


def graph_from_binning(
    binning: Binning,
    *,
    coordinates: CoordinateSystem = CartesianSystem(),
    graph_class: type[_G] = nx.Graph,
    attributes: Callable[..., dict[str, Any]] = dict,
) -> _G:
    """Build navigation attributes without changing bin or neighbor-pair order.

    Parameters
    ----------
    binning : Binning
        Bins and neighbor pairs to project.
    coordinates : CoordinateSystem, default=CartesianSystem()
        Coordinate system of the bin centers. Its ``cartesian_embedding``
        places the centers for edge vectors: the centers themselves for
        Cartesian bins, and the animal's egocentric plane (x ahead, y left)
        for ``(distance, angle)`` bins, so the angle seam needs no wrapping.
    graph_class : type of networkx.Graph, default=networkx.Graph
        Empty-constructible graph type to fill.
    attributes : callable, default=dict
        Makes each node's and edge's attribute mapping from keyword
        arguments, once; an undirected edge shares one mapping in both
        directions, as ``networkx`` stores it.

    Returns
    -------
    graph : networkx.Graph
        One node per bin, in bin order, and one edge per row of
        ``binning.connections.pairs``. Every node has ``pos`` (its center,
        length ``n_dims``), ``source_grid_flat_index`` and
        ``original_grid_nd_index`` (the bin's cell in its source grid when
        the locator records one, else the bin index). Track bins also carry
        their linear position ``pos_1D`` and track-graph edge
        ``source_edge_id`` (-1 in a gap cell), but only when every active
        linear cell is a bin and the cell edges follow the track's segments;
        otherwise those two attributes are absent.

        Every edge has ``distance`` (the stored pair distance), ``vector``
        (the embedded displacement from the lower to the higher bin index)
        and ``edge_id`` (the edge's row in ``binning.connections.pairs``,
        which lists each neighbor pair once). Edges of 2-D bins also carry
        ``angle_2d``, ``arctan2`` of ``vector`` in radians: an allocentric
        direction (0 = East, +π/2 = North) for Cartesian bins, and an
        egocentric one (0 = ahead, +π/2 = left) for ``(distance, angle)``
        bins.
    """
    bins = binning.bins
    centers = bins.centers
    n_bins = bins.n_bins
    # Track bins are indexed by their linear cells; other bins by the logical
    # cells their locator records, else by their own index.
    flat_indices: list[int] = list(range(n_bins))
    nd_indices: list[tuple[int, ...]] = [(index,) for index in range(n_bins)]
    track = binning.linearization
    source = (
        (track.active_mask.shape, np.flatnonzero(track.active_mask))
        if isinstance(track, TrackLinearization)
        else _source_grid(binning.locator)
    )
    if source is not None:
        shape, indices = source
        flat_indices[: len(indices)] = indices.tolist()
        nd_indices[: len(indices)] = [
            tuple(row)
            for row in np.column_stack(np.unravel_index(indices, shape)).tolist()
        ]
    linear: list[float] | None = None
    if isinstance(track, TrackLinearization):
        flat = np.flatnonzero(track.active_mask)
        segments = _track_cell_segments(track)
        if len(flat) == n_bins and segments is not None:
            linear = get_centers(track.grid_edges[0])[flat].tolist()
            edge_ids = segments[flat].tolist()

    graph: _G = graph_class()
    if linear is None:
        graph._node = {
            index: attributes(
                pos=tuple(center),
                source_grid_flat_index=flat_index,
                original_grid_nd_index=nd_index,
            )
            for index, (center, flat_index, nd_index) in enumerate(
                zip(centers.tolist(), flat_indices, nd_indices, strict=True)
            )
        }
    else:
        graph._node = {
            index: attributes(
                pos=tuple(center),
                source_grid_flat_index=flat_index,
                original_grid_nd_index=nd_index,
                pos_1D=position,
                source_edge_id=edge_id,
            )
            for index, (center, flat_index, nd_index, position, edge_id) in enumerate(
                zip(
                    centers.tolist(),
                    flat_indices,
                    nd_indices,
                    linear,
                    edge_ids,
                    strict=True,
                )
            )
        }

    # Graph nodes iterate in bin ID order. Keep each vector in that graph
    # orientation; edge_id links it to the explicit pair-array authority.
    pairs = binning.connections.pairs
    low = np.minimum(pairs[:, 0], pairs[:, 1])
    high = np.maximum(pairs[:, 0], pairs[:, 1])
    places = coordinates.cartesian_embedding(centers)
    try:
        with np.errstate(over="raise", invalid="raise"):
            vectors = places[high] - places[low]
    except FloatingPointError as exc:
        raise ValueError(
            "Pair displacement exceeds floating-point range; recenter coordinates."
        ) from exc
    columns = [
        low.tolist(),
        high.tolist(),
        binning.connections.pair_distances.tolist(),
        vectors.tolist(),
    ]
    if bins.n_dims == 2:
        columns.append(np.arctan2(vectors[:, 1], vectors[:, 0]).tolist())
    # Neighbor dicts list nodes in bin order, then neighbors in pair order,
    # as Graph.add_nodes_from and add_edges_from would insert them.
    adjacency: dict[int, dict[int, dict[str, Any]]] = {
        index: {} for index in range(n_bins)
    }
    for edge_id, (u, v, distance, vector, *angle) in enumerate(
        zip(*columns, strict=True)
    ):
        data = (
            attributes(
                distance=distance,
                vector=tuple(vector),
                edge_id=edge_id,
                angle_2d=angle[0],
            )
            if angle
            else attributes(distance=distance, vector=tuple(vector), edge_id=edge_id)
        )
        adjacency[u][v] = data
        adjacency[v][u] = data
    graph._adj = adjacency
    return graph
