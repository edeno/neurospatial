"""Pure projections between a stored linear track and embedded coordinates."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from neurospatial.geometry._locators import _coordinate_roundoff

Edge = tuple[Any, Any]


def _segment_starts(
    graph: nx.Graph, edge_order: Sequence[Edge], edge_spacing: float | Sequence[float]
) -> NDArray[np.float64]:
    """Linear position of the first node of each segment, shape (n_edges,)."""
    lengths = np.array([graph.edges[e]["distance"] for e in edge_order], dtype=float)
    if isinstance(edge_spacing, (int, float, np.number)):
        gaps = np.full(max(0, len(edge_order) - 1), float(edge_spacing))
    else:
        gaps = np.asarray(edge_spacing, dtype=float)
    return np.concatenate([[0.0], np.cumsum(lengths[:-1] + gaps)])


def project_nd_to_1d(
    points: NDArray[np.float64],
    graph: nx.Graph,
    edge_order: Sequence[Edge],
    edge_spacing: float | Sequence[float],
) -> NDArray[np.float64]:
    """Map N-D points to linear positions on their nearest track segment.

    A point as near, within rounding, to several segments (as on the shared
    stretch of an out-and-back track) takes the segment first in
    ``edge_order``, the one holding the lowest-numbered bins, as bin lookup
    takes the lowest-numbered of tied bins. The choice therefore does not
    depend on how a rotation or shift of the track rounds.

    Parameters
    ----------
    points : ndarray of float64, shape (n_points, n_dims)
        Positions in the track's coordinates.
    graph : networkx.Graph
        Track graph with node ``pos`` and edge ``distance`` attributes.
    edge_order : sequence of (node, node)
        Track segments in linear coordinate order.
    edge_spacing : float or sequence of float
        Gaps between successive segments in linear coordinates.

    Returns
    -------
    ndarray of float64, shape (n_points,)
        Linear position of each point's projection, measured from the first
        node of its segment; NaN for a non-finite point.
    """
    query = np.atleast_2d(np.asarray(points, dtype=np.float64))
    order = list(edge_order)
    position = nx.get_node_attributes(graph, "pos")
    # Project along each edge as the graph stores it.
    stored = {frozenset(edge): edge for edge in graph.edges()}
    segments = np.array(
        [[position[node] for node in stored[frozenset(edge)]] for edge in order],
        dtype=np.float64,
    )
    first = segments[:, 0]
    direction = segments[:, 1] - first
    squared = np.sum(direction**2, axis=1)
    finite = np.all(np.isfinite(query), axis=1)
    values = np.where(finite[:, None], query, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        along = np.sum(direction * (values[:, None, :] - first), axis=2) / squared
    # A zero-length segment is its one node.
    along = np.clip(np.where(squared > 0, along, 0.0), 0.0, 1.0)
    projected = first + along[:, :, None] * direction
    distance = np.linalg.norm(values[:, None, :] - projected, axis=2)
    tolerance = _coordinate_roundoff(values, float(np.max(np.abs(segments))))
    tied = distance <= (distance.min(axis=1) + tolerance)[:, None]
    segment = np.argmax(tied, axis=1)
    rows = np.arange(len(query))
    start_nodes = np.array([position[edge[0]] for edge in order], dtype=np.float64)
    linear = _segment_starts(graph, order, edge_spacing)[segment] + np.linalg.norm(
        start_nodes[segment] - projected[rows, segment], axis=1
    )
    linear[~finite] = np.nan
    return np.asarray(linear, dtype=np.float64)


def project_1d_to_2d(
    linear_position: NDArray[np.float64],
    graph: nx.Graph,
    edge_order: list[Edge],
    edge_spacing: float | Sequence[float] = 0.0,
) -> NDArray[np.float64]:
    """Map 1D linear positions back to N-D coordinates on the track graph.

    Projects points from a 1D linearized representation of the track (defined
    by `graph`, `edge_order`, and `edge_spacing`) back to their original
    N-dimensional spatial coordinates.

    Parameters
    ----------
    linear_position : NDArray[np.float64], shape (n_points,)
        1D positions along the linearized track.
    graph : networkx.Graph
        The original graph. Nodes must have a 'pos' attribute (N-D coordinates)
        and edges must have a 'distance' attribute (pre-calculated length).
    edge_order : list[tuple[node, node]]
        Ordered sequence of edges defining the linearization path.
    edge_spacing : float or list[float], optional, default=0.0
        Spacing (gaps) between track segments used during linearization.
        If float, applied uniformly. If list, length must be `len(edge_order) - 1`.

    Returns
    -------
    coords_nd : NDArray[np.float64], shape (n_points, n_dims)
        N-dimensional coordinates corresponding to each input `linear_position`.
        A position on no segment is a row of NaN: one before the start (below
        0), past the end of the last segment, inside a gap between segments,
        or not finite (NaN or ±inf). Segment ends, including both ends of the
        track, are on the track.

    Raises
    ------
    ValueError
        If `linear_position` is not 1D, `edge_spacing` (if a list) has an
        incorrect length, or an edge in `edge_order` has no 'distance'
        attribute.

    """
    linear_position = np.asarray(linear_position, dtype=float)
    if linear_position.ndim == 0:
        linear_position = linear_position.reshape((1,))
    elif linear_position.ndim > 1:
        # Attempt to squeeze if it's like (N,1) or (1,N)
        squeezed_pos = np.squeeze(linear_position)
        if squeezed_pos.ndim == 1:
            linear_position = squeezed_pos
        elif squeezed_pos.ndim == 0 and linear_position.size == 1:  # e.g. from [[5.]]
            linear_position = squeezed_pos.reshape((1,))
        else:  # If it's still not 1D (e.g. (M,N) where M,N > 1)
            raise ValueError("linear_position must be convertible to a 1D array.")
    if linear_position.ndim != 1:
        raise ValueError("linear_position must be a 1D array.")
    n_edges = len(edge_order)

    # Every edge must carry a numeric 'distance' (segment length). An external
    # track graph (built by track_linearization, or by hand) may omit it; without
    # this check the comprehension below raises a bare ``KeyError: 'distance'``
    # deep in an internal helper, naming neither the offending edge nor how to
    # fix it.
    missing = [e for e in edge_order if "distance" not in graph.edges[e]]
    if missing:
        raise ValueError(
            f"{len(missing)} of {n_edges} track-graph edge(s) are missing the "
            f"required 'distance' edge attribute (segment length); the first is "
            f"edge {missing[0]}. Give every edge a numeric 'distance', e.g. "
            f"nx.set_edge_attributes(graph, {{{missing[0]!r}: <length>}}, "
            f"name='distance'), or pass distance=<length> when adding each edge."
        )

    # --- edge lengths & spacing ------------------------------------------------
    edge_lengths = np.array(
        [graph.edges[e]["distance"] for e in edge_order],
        dtype=float,
    )

    if isinstance(edge_spacing, (int, float, np.number)):
        gaps = np.full(max(0, n_edges - 1), float(edge_spacing))
    else:
        gaps = np.asarray(edge_spacing, dtype=float)
        if gaps.size != max(0, n_edges - 1):
            raise ValueError("edge_spacing length must be len(edge_order)-1")

    # cumulative start position of each edge
    cumulative_edge_start_position = np.concatenate(
        [[0.0], np.cumsum(edge_lengths[:-1] + gaps)],
    )  # shape (n_edges,)

    # The last edge starting at or before each position; positions before the
    # first edge start on edge 0 and are rejected below.
    edge_ind = np.clip(
        np.searchsorted(cumulative_edge_start_position, linear_position, side="right")
        - 1,
        0,
        n_edges - 1,
    )
    finite = np.isfinite(linear_position)
    edge_ind[~finite] = 0  # dummy index, overwritten with NaN below

    # param along each chosen edge
    edge_start = cumulative_edge_start_position[edge_ind]
    edge_length = edge_lengths[edge_ind]
    with np.errstate(invalid="ignore"):
        normalized_edge_position = (linear_position - edge_start) / edge_length
        # Rounding of the subtraction, so the ends of each edge stay on it.
        slack = (
            4
            * np.finfo(np.float64).eps
            * (np.abs(linear_position) + np.abs(edge_start) + edge_length)
            / edge_length
        )
        on_track = (
            finite
            & (normalized_edge_position >= -slack)
            & (normalized_edge_position <= 1.0 + slack)
        )
    # Positions before the start, past the end or in a gap are not on the track.
    nan_mask = ~on_track
    normalized_edge_position = np.clip(normalized_edge_position, 0.0, 1.0)

    # gather endpoint coordinates
    node_position_2d = nx.get_node_attributes(graph, "pos")
    start_node_position_2d = np.array(
        [node_position_2d[edge_order[int(i)][0]] for i in edge_ind],
    )
    end_node_position_2d = np.array(
        [node_position_2d[edge_order[int(i)][1]] for i in edge_ind],
    )

    # Linear interpolation between endpoints
    position_2d = (
        1.0 - normalized_edge_position[:, None]
    ) * start_node_position_2d + normalized_edge_position[
        :,
        None,
    ] * end_node_position_2d

    position_2d[nan_mask] = np.nan

    return np.asarray(position_2d, dtype=np.float64)


def track_cell_ends(
    graph: nx.Graph,
    edge_order: Sequence[Edge],
    edge_spacing: float | Sequence[float],
    linear_edges: NDArray[np.float64],
    active_mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Both ends of each active track cell, in the track's coordinates.

    A track cell is the stretch of centerline between two consecutive linear
    bin edges. Each cell lies on one track segment, so both of its ends are
    placed on the segment that holds the cell's middle; a cell ending where
    the next segment starts elsewhere keeps its own segment's end node.

    Parameters
    ----------
    graph : networkx.Graph
        Track graph with node ``pos`` and edge ``distance`` attributes.
    edge_order : sequence of (node, node)
        Track segments in linear coordinate order.
    edge_spacing : float or sequence of float
        Gaps between successive segments in linear coordinates.
    linear_edges : ndarray, shape (n_linear_bins + 1,)
        Linear bin edges, including gap bins.
    active_mask : ndarray of bool, shape (n_linear_bins,)
        Cells that are bins, in linear order.

    Returns
    -------
    ndarray, shape (n_active, 2, n_dims)
        Start and end of each active cell.
    """
    order = list(edge_order)
    lengths = np.array([graph.edges[e]["distance"] for e in order], dtype=float)
    if isinstance(edge_spacing, (int, float, np.number)):
        gaps = np.full(max(0, len(order) - 1), float(edge_spacing))
    else:
        gaps = np.asarray(edge_spacing, dtype=float)
    starts = np.concatenate([[0.0], np.cumsum(lengths[:-1] + gaps)])
    low = np.asarray(linear_edges[:-1], dtype=float)[active_mask]
    high = np.asarray(linear_edges[1:], dtype=float)[active_mask]
    segment = np.clip(
        np.searchsorted(starts, (low + high) / 2, side="right") - 1, 0, len(order) - 1
    )
    positions = nx.get_node_attributes(graph, "pos")
    first = np.array([positions[u] for u, _ in order], dtype=float)[segment]
    last = np.array([positions[v] for _, v in order], dtype=float)[segment]
    ends = np.stack([low, high], axis=1)
    fraction = np.clip(
        (ends - starts[segment, None]) / lengths[segment, None], 0.0, 1.0
    )
    return np.asarray(
        first[:, None] + fraction[..., None] * (last - first)[:, None],
        dtype=np.float64,
    )
