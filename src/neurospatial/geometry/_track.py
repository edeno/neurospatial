"""Pure projection from a stored linear track to embedded coordinates."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import networkx as nx
import numpy as np
from numpy.typing import NDArray

Edge = tuple[Any, Any]


def _project_1d_to_2d(
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
        Positions falling into gaps or beyond track ends are typically
        projected onto the nearest valid track segment endpoint.
        NaNs in `linear_position` propagate to rows of NaNs in output.

    Raises
    ------
    ValueError
        If `linear_position` is not 1D.
        If `edge_spacing` (if a list) has an incorrect length.

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

    edge_ind = (
        np.searchsorted(cumulative_edge_start_position, linear_position, side="right")
        - 1
    )
    edge_ind = np.clip(edge_ind, 0, n_edges - 1)  # clamp to valid edge index

    nan_mask = ~np.isfinite(linear_position).squeeze()
    edge_ind[nan_mask] = 0  # dummy index, will overwrite later

    # param along each chosen edge
    normalized_edge_position = (
        linear_position - cumulative_edge_start_position[edge_ind]
    ) / edge_lengths[edge_ind]
    normalized_edge_position = np.clip(
        normalized_edge_position,
        0.0,
        1.0,
    )  # project extremes onto endpoints

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

    # propagate NaNs from the input
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
