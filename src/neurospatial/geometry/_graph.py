"""Project authoritative binning data into standard navigation graph attributes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from neurospatial.geometry._arrays import get_centers
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
    elif isinstance(locator, MeshLocator):
        shape = (len(locator.simplices),)
        mapping = locator.simplex_to_active
        active = np.flatnonzero(mapping >= 0)
    else:
        return None
    if mapping is not None:
        active = active[np.argsort(mapping[active])]
    return shape, active.astype(np.intp, copy=False)


def _track_cell_segments(track: TrackLinearization) -> NDArray[np.intp] | None:
    """Return the track-graph edge ID of every linear cell, or None.

    Walks the segments in linear order the way a track layout bins them:
    each segment of positive length fills the cells up to its end, a
    zero-length segment gets one cell, and a positive gap after a segment is
    one inactive cell (ID -1). Returns None when the stored cell edges do
    not follow that rule, so no segment can be assigned.
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
        count = (
            1
            if length <= 0.0
            else int(np.searchsorted(edges, cursor + length, side="left")) - cell
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

    Graph vectors follow graph edge orientation, from the lower to the higher
    bin ID. They are physical displacements between the bin centers placed by
    ``coordinates.cartesian_embedding``: the centers themselves for Cartesian
    bins, and the animal's egocentric plane (x ahead, y left) for
    ``(distance, angle)`` bins, so the angle seam needs no wrapping. Edge IDs identify the original pair-array rows; source-grid IDs
    preserve logical indexing when available. Track bins also carry their
    linear position (``pos_1D``) and track-graph edge (``source_edge_id``).
    Derivations preserving existing graph metadata keep that separate from
    this projection of newly constructed bin data.

    ``graph_class`` is the (empty-constructible) graph type to fill, and
    ``attributes`` makes each node's and edge's attribute mapping from keyword
    arguments, once; an undirected edge shares one mapping in both directions,
    as ``networkx`` stores it.
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
    low = np.minimum(bins.pairs[:, 0], bins.pairs[:, 1])
    high = np.maximum(bins.pairs[:, 0], bins.pairs[:, 1])
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
        bins.pair_distances.tolist(),
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
