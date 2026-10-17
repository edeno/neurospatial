"""Project authoritative binning data into standard navigation graph attributes."""

from __future__ import annotations

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from neurospatial.geometry._arrays import get_centers
from neurospatial.geometry._binning import Binning, TrackLinearization
from neurospatial.geometry._locators import (
    HexLocator,
    Locator,
    MeshLocator,
    RectilinearLocator,
    SubsetLocator,
    TransformedLocator,
)
from neurospatial.geometry._polygon_cells import ClippedLocator


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


def _graph_from_binning(binning: Binning) -> nx.Graph:
    """Build navigation attributes without changing bin or neighbor-pair order.

    Graph vectors follow graph edge orientation, from the lower to the higher
    bin ID. Edge IDs identify the original pair-array rows; source-grid IDs
    preserve logical indexing when available. Track bins also carry their
    linear position (``pos_1D``) and track-graph edge (``source_edge_id``).
    Derivations preserving existing graph metadata keep that separate from
    this projection of newly constructed bin data.
    """
    bins = binning.bins
    centers = bins.centers
    graph = nx.Graph()
    graph.add_nodes_from(
        (
            index,
            {
                "pos": tuple(center),
                "source_grid_flat_index": index,
                "original_grid_nd_index": (index,),
            },
        )
        for index, center in enumerate(centers.tolist())
    )
    # Graph nodes iterate in bin ID order. Keep each vector in that graph
    # orientation; edge_id links it to the explicit pair-array authority.
    low = np.minimum(bins.pairs[:, 0], bins.pairs[:, 1])
    high = np.maximum(bins.pairs[:, 0], bins.pairs[:, 1])
    try:
        with np.errstate(over="raise", invalid="raise"):
            vectors = centers[high] - centers[low]
    except FloatingPointError as exc:
        raise ValueError(
            "Pair displacement exceeds floating-point range; recenter coordinates."
        ) from exc
    distances = bins.pair_distances.tolist()
    vector_rows = vectors.tolist()
    if bins.n_dims == 2:
        angles = np.arctan2(vectors[:, 1], vectors[:, 0]).tolist()
        graph.add_edges_from(
            (
                u,
                v,
                {
                    "distance": distance,
                    "vector": tuple(vector),
                    "edge_id": index,
                    "angle_2d": angle,
                },
            )
            for index, (u, v, distance, vector, angle) in enumerate(
                zip(
                    low.tolist(),
                    high.tolist(),
                    distances,
                    vector_rows,
                    angles,
                    strict=True,
                )
            )
        )
    else:
        graph.add_edges_from(
            (
                u,
                v,
                {"distance": distance, "vector": tuple(vector), "edge_id": index},
            )
            for index, (u, v, distance, vector) in enumerate(
                zip(low.tolist(), high.tolist(), distances, vector_rows, strict=True)
            )
        )
    # Track bins are indexed by their linear cells; other bins by the logical
    # cells their locator records.
    track = binning.linearization
    source = (
        (track.active_mask.shape, np.flatnonzero(track.active_mask))
        if isinstance(track, TrackLinearization)
        else _source_grid(binning.locator)
    )
    if source is not None:
        shape, indices = source
        nd_indices = np.column_stack(np.unravel_index(indices, shape)).tolist()
        for index, (flat, nd_index) in enumerate(
            zip(indices.tolist(), nd_indices, strict=True)
        ):
            graph.nodes[index].update(
                source_grid_flat_index=flat,
                original_grid_nd_index=tuple(nd_index),
            )
    if isinstance(track, TrackLinearization):
        flat = np.flatnonzero(track.active_mask)
        segments = _track_cell_segments(track)
        if len(flat) == bins.n_bins and segments is not None:
            linear = get_centers(track.grid_edges[0])[flat].tolist()
            for index, (position, segment) in enumerate(
                zip(linear, segments[flat].tolist(), strict=True)
            ):
                graph.nodes[index].update(pos_1D=position, source_edge_id=segment)
    return graph
