"""Utility functions for graph-based (linearized track) layouts. 📐

This module provides helper functions specifically for environments where the
spatial layout is defined by a graph structure that is subsequently linearized,
such as an animal's track in an experiment. These functions handle:

- Discretizing a linearized graph track into 1D bins, accounting for edge
  lengths and specified spacing between segments (`_get_graph_bins`).
- Pairing adjacent bins of these binned segments, along each segment and
  across the track-graph nodes where segments meet (`track_pairs`).
- Projecting 1D linearized positions back to their corresponding N-D
  coordinates on the original track graph (`project_1d_to_2d`).
- Finding the appropriate 1D bin index for a given linear position
  (`_find_bin_for_linear_position`).

These utilities are primarily used by the `GraphLayout` engine.
"""

from collections.abc import Sequence
from typing import Any

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from neurospatial.layout.helpers.neighbor_pairs import (
    euclidean_pair_distances,
    order_neighbor_pairs,
)
from neurospatial.layout.helpers.utils import get_centers

Edge = tuple[Any, Any]


def _get_graph_bins(
    graph: nx.Graph,
    edge_order: list[tuple[object, object]],
    edge_spacing: float | Sequence[float],
    bin_size: float,
) -> tuple[
    NDArray[np.float64],
    tuple[NDArray[np.float64], ...],
    NDArray[np.bool_],
    NDArray[np.int_],
]:
    """Discretize each edge of a linearized graph into fixed-length bins, optionally
    inserting inactive "gap" bins between segments.

    Parameters
    ----------
    graph : networkx.Graph
        Must have `pos` attributes on each node: {node: (x, y, …)}.
    edge_order : list of (u, v) tuples
        Ordered edges defining the linear track. Each tuple's nodes must exist in `graph`.
    edge_spacing : float or sequence of floats
        If float: uniform gap length inserted between every consecutive edge.
        If sequence: length must be len(edge_order) - 1, specifying each gap individually.
    bin_size : float
        Desired length of each bin along the edge; must be positive.

    Returns
    -------
    bin_centers_1d : NDArray[np.float64], shape (n_total_bins,)
        1D coordinates of the center of every bin (including gaps).
    bin_edges_1d_tuple : tuple of one NDArray[np.float64], shape (n_total_bins + 1,)
        Unique sorted 1D edge coordinates (matches np.histogramdd format).
    active_mask_1d : NDArray[np.bool_], shape (n_total_bins,)
        True for bins that lie on an actual graph edge, False for gap bins.
    edge_ids_for_active_bins : NDArray[np.int_], shape (n_active_bins,)
        Original edge index (0..len(graph.edges)-1) for each active bin.

    Raises
    ------
    ValueError
        - If `bin_size` is not positive.
        - If `edge_spacing` is a sequence but length != len(edge_order) - 1.
        - If any node in `edge_order` is missing from `graph`.

    """
    # 1) Validate bin_size
    if bin_size <= 0:
        raise ValueError(f"`bin_size` must be a positive float (got {bin_size}).")

    # 2) Validate edge_order contents
    missing_nodes = {
        node for edge in edge_order for node in edge if node not in graph.nodes
    }
    if missing_nodes:
        raise ValueError(f"Nodes {missing_nodes} in edge_order not found in graph.")

    # 3) Build gap array
    n_edges = len(edge_order)
    if isinstance(edge_spacing, (int, float, np.number)):
        gaps = np.full(max(0, n_edges - 1), float(edge_spacing))
    else:
        gaps = np.asarray(edge_spacing, dtype=float)
        if gaps.ndim != 1 or gaps.shape[0] != max(0, n_edges - 1):
            raise ValueError(
                f"`edge_spacing` length must be {max(0, n_edges - 1)} (got {gaps.shape[0]})."
            )
    # If there are zero edges, we return empty structures
    if n_edges == 0:
        return (
            np.zeros((0,), dtype=float),
            (np.zeros((0,), dtype=float),),
            np.zeros((0,), dtype=bool),
            np.zeros((0,), dtype=int),
        )

    # 4) Precompute node positions; raise if any node lacks 'pos'
    node_positions = nx.get_node_attributes(graph, "pos")
    if len(node_positions) < len(graph.nodes):
        raise ValueError("All nodes must have a 'pos' attribute (x,y,…).")

    # 5) Map each (u, v) or (v, u) in graph.edges() to an integer ID
    #    Normalize edges so (u,v) and (v,u) map to the same ID:
    edge_id_map: dict[frozenset, int] = {
        frozenset(edge): idx for idx, edge in enumerate(graph.edges())
    }

    bin_edges: list[float] = []
    active_mask_list: list[bool] = []
    edge_id_list: list[int] = []
    cursor = 0.0

    for idx, (u, v) in enumerate(edge_order):
        pos_u = np.asarray(node_positions[u], dtype=float)
        pos_v = np.asarray(node_positions[v], dtype=float)
        segment_length = float(np.linalg.norm(pos_v - pos_u))
        # If the two nodes coincide, create exactly one bin of zero length
        n_bins = 1 if segment_length <= 0.0 else int(np.ceil(segment_length / bin_size))
        # Create (n_bins + 1) edges from cursor to cursor + segment_length
        edges_segment = np.linspace(
            cursor,
            cursor + segment_length,
            n_bins + 1,
            dtype=float,
        )
        bin_edges.extend(edges_segment[:-1])  # all but last; last becomes next cursor
        # Record edge IDs for each _active_ bin
        mapped_id = edge_id_map.get(frozenset((u, v)), -1)
        edge_id_list.extend([mapped_id] * n_bins)
        active_mask_list.extend([True] * n_bins)

        cursor += segment_length
        # Optionally insert a gap after this edge
        if idx < n_edges - 1:
            gap_len = float(gaps[idx])
            if gap_len > 0.0:
                bin_edges.append(cursor)
                active_mask_list.append(False)
                cursor += gap_len

    # After the loop, append final edge
    bin_edges.append(cursor)

    # 6) Convert to arrays and dedupe
    bin_edges_arr: NDArray[np.float64] = np.array(bin_edges, dtype=np.float64)
    bin_centers_1d = get_centers(bin_edges_arr)

    active_mask_1d = np.asarray(active_mask_list, dtype=np.bool_)
    edge_ids_for_active_bins = np.array(edge_id_list, dtype=int)

    return bin_centers_1d, (bin_edges_arr,), active_mask_1d, edge_ids_for_active_bins


def set_straight_distances(graph: nx.Graph) -> None:
    """Give every edge its straight-line length as ``distance``, in place.

    An edge of a track graph is the straight segment between its nodes'
    ``pos``. The bins are laid out along that segment, while bin centers,
    cell ends and linear positions are measured with ``distance``, so the
    two must agree: a missing ``distance`` is filled in, and one that
    differs is refused.

    Raises
    ------
    ValueError
        If an edge's ``distance`` is not its nodes' straight-line distance.
    """
    positions = nx.get_node_attributes(graph, "pos")
    mismatched = []
    for u, v, data in graph.edges(data=True):
        length = float(
            np.linalg.norm(
                np.asarray(positions[v], dtype=float)
                - np.asarray(positions[u], dtype=float)
            )
        )
        given = data.get("distance")
        if given is not None and not np.isclose(float(given), length):
            mismatched.append(
                f"({u!r}, {v!r}): distance={float(given):g}, {length:g} apart"
            )
        data["distance"] = length
    if mismatched:
        raise ValueError(
            f"{len(mismatched)} track-graph edge(s) have a 'distance' that is not "
            f"the distance between their nodes' 'pos': {'; '.join(mismatched[:3])}"
            f"{'; ...' if len(mismatched) > 3 else ''}. Why: an edge is the "
            "straight segment between its nodes; the bins are laid out along "
            "it and positions are measured along it with 'distance', so the "
            "two lengths must agree. Fix: drop the 'distance' attributes (they "
            "are computed from 'pos'), or move the nodes; a curved track needs "
            "more nodes along the curve."
        )


def track_pairs(
    graph: nx.Graph,
    bin_centers_nd: NDArray[np.float64],
    original_edge_ids: NDArray[np.int_],
    edge_order: list[tuple[object, object]],
) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
    """Pair adjacent bins of a binned track, along and between its segments.

    Bins on the same track segment (original graph edge) are paired with the
    next bin along it, segment by segment in order of first appearance. Then,
    at each track-graph node, the end bins of the segments that meet there
    are paired in ``edge_order`` order. The proposals are put in
    authoritative order by
    :func:`~neurospatial.layout.helpers.neighbor_pairs.order_neighbor_pairs`.

    Parameters
    ----------
    graph : networkx.Graph
        The track graph from which bins were derived.
    bin_centers_nd : NDArray[np.float64], shape (n_active_bins, n_dims)
        n-D coordinates of the centers of active bins.
    original_edge_ids : NDArray[np.int_], shape (n_active_bins,)
        Integer IDs of the original graph edge for each active bin.
    edge_order : list of tuples
        The ordered sequence of edges from the original graph that defines
        the linearized track layout. Used for inter-segment connections.

    Returns
    -------
    pairs : ndarray of intp, shape (n_pairs, 2)
        Pairs ``(lower, higher)`` of bin IDs in authoritative order.
    distances : ndarray of float64, shape (n_pairs,)
        Euclidean distance between the n-D centers of each pair.

    """
    # --- 1. Intra-segment pairs: consecutive bins on the same original edge ---
    _, first_bin, segment = np.unique(
        original_edge_ids, return_index=True, return_inverse=True
    )
    # Rank segments by their first bin so they are visited in track order.
    rank = np.argsort(np.argsort(first_bin))[segment.ravel()]
    by_segment = np.argsort(rank, kind="stable")
    ranks = rank[by_segment]
    same = ranks[1:] == ranks[:-1]
    first = [by_segment[:-1][same]]
    second = [by_segment[1:][same]]
    starts = np.flatnonzero(np.r_[True, ~same])
    ends = np.r_[starts[1:] - 1, len(by_segment) - 1]
    bin_edge_order_arr = np.column_stack((by_segment[starts], by_segment[ends]))

    # --- 2. Inter-segment pairs: end bins of the segments meeting at a node ---
    edge_order_array = np.asarray(edge_order)
    for node_id in graph.nodes:
        connections = bin_edge_order_arr[edge_order_array == node_id]
        first.append(connections[:-1])
        second.append(connections[1:])

    lower = np.concatenate(first).astype(np.intp)
    upper = np.concatenate(second).astype(np.intp)
    return order_neighbor_pairs(
        lower,
        upper,
        euclidean_pair_distances(bin_centers_nd, np.column_stack((lower, upper))),
    )


def _find_bin_for_linear_position(
    linear_positions: float | NDArray[np.float64],
    bin_edges: NDArray[np.float64],
    active_mask: NDArray[np.bool_] | None = None,
) -> int | NDArray[np.int_]:
    """Find the 1D bin index for each given linear position.

    Assigns each position in `linear_positions` to a bin defined by
    `bin_edges_1d`. If `active_mask_1d` is provided, positions falling into
    inactive bins (gaps) can be handled (e.g., by adjusting to a nearby
    active bin or returning an invalid index, depending on implementation details).
    The current implementation maps to bins based on `np.digitize` and handles
    positions on edges that might fall into inactive bins by shifting them to
    the preceding active bin.

    Parameters
    ----------
    linear_positions : float | NDArray[np.float64], shape (n_points,)
        Linear position(s) to map to bin indices.
    bin_edges_1d : NDArray[np.float64], shape (n_total_bins + 1,)
        1D array of sorted bin edge coordinates for the entire linearized track
        (including gaps).
    active_mask_1d : NDArray[np.bool_] | None, shape (n_total_bins,), optional
        Boolean mask indicating which bins in the `bin_edges_1d` definition
        are active (on an edge segment) vs. inactive (in a gap). If provided,
        the function ensures returned indices correspond to active bins or
        handles points in gaps.

    Returns
    -------
    bin_indices : int | NDArray[np.int_], shape (n_points,)
        0-based bin index (relative to the full set of bins defined by
        `bin_edges_1d`) for each input linear position.
        Returns -1 if a position falls outside the range of `bin_edges_1d`.
        If `active_mask_1d` is used, logic attempts to assign to an active bin.

    Raises
    ------
    ValueError
        If `active_mask_1d` (if provided) length doesn't match the number of bins.
        If, after processing, a position maps to an index marked as invalid by
        `active_mask_1d` (indicating an issue with gap handling logic or input).

    """
    was_scalar = np.isscalar(linear_positions)
    linear_positions = np.atleast_1d(np.asarray(linear_positions, dtype=float))
    n_bins = len(bin_edges) - 1
    if n_bins <= 0:
        # No bins defined
        return -1 if was_scalar else np.full(linear_positions.shape, -1, dtype=int)

    if active_mask is not None and len(active_mask) != n_bins:
        raise ValueError(
            f"active_mask length ({len(active_mask)}) must match number of bins ({n_bins}).",
        )

    bin_ind = np.digitize(linear_positions, bin_edges, right=False) - 1
    out_of_bounds_mask = (bin_ind < 0) | (bin_ind >= n_bins)
    bin_ind[out_of_bounds_mask] = -1

    if active_mask is not None:
        invalid_bin_ind = np.nonzero(~active_mask)[0]
    else:
        invalid_bin_ind = np.array([])

    in_gap_ind = np.nonzero(np.isin(bin_ind, invalid_bin_ind))[0]
    is_bin_edge = np.asarray(
        [
            np.any(np.isclose(bin_edges[1:], lin_pos))
            for lin_pos in linear_positions[in_gap_ind]
        ],
        dtype=bool,
    )
    bin_ind[in_gap_ind[is_bin_edge]] = bin_ind[in_gap_ind[is_bin_edge]] - 1

    bin_ind[in_gap_ind[~is_bin_edge]] = -1
    return int(bin_ind[0]) if was_scalar else bin_ind.astype(int)
