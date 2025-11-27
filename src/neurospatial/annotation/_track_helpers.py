"""Graph building helpers for track graph annotation.

This module provides utility functions for building track graphs from
annotation state, including coordinate transformation and result construction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import networkx as nx
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from neurospatial.environment import Environment
    from neurospatial.transforms import VideoCalibration

from neurospatial.annotation._track_state import TrackBuilderState


def transform_nodes_to_output(
    nodes_px: list[tuple[float, float]],
    calibration: VideoCalibration | None,
) -> list[tuple[float, float]]:
    """Transform node positions from pixels to output coordinates.

    Reuses VideoCalibration.transform_px_to_cm() for consistency
    with annotate_video and other annotation tools.

    Parameters
    ----------
    nodes_px : list of tuple
        Node positions in pixel coordinates as (x, y) tuples.
    calibration : VideoCalibration or None
        Pixel-to-cm transform. If None, returns pixel coordinates unchanged.

    Returns
    -------
    list of tuple
        Node positions in output coordinates (cm if calibrated, else pixels).

    Examples
    --------
    >>> nodes = [(100.0, 200.0), (300.0, 400.0)]
    >>> transform_nodes_to_output(nodes, calibration=None)
    [(100.0, 200.0), (300.0, 400.0)]

    """
    if not nodes_px:
        return []

    if calibration is None:
        return nodes_px

    # Convert to numpy array for transform
    nodes_array = np.array(nodes_px, dtype=np.float64)

    # Apply the calibration transform (includes Y-flip and scaling)
    nodes_cm = calibration.transform_px_to_cm(nodes_array)

    # Convert back to list of tuples
    return [tuple(pos) for pos in nodes_cm]


def build_track_graph_from_positions(
    node_positions: list[tuple[float, float]],
    edges: list[tuple[int, int]],
) -> nx.Graph:
    """Build track graph from node positions and edge list.

    Creates a NetworkX graph with proper node 'pos' attributes and
    edge 'distance'/'edge_id' attributes, compatible with track_linearization.

    Parameters
    ----------
    node_positions : list of tuple
        Node coordinates as (x, y) tuples in output units (cm or pixels).
    edges : list of tuple
        Edge connections as (node_i, node_j) index tuples.

    Returns
    -------
    nx.Graph
        Track graph with node 'pos' and edge 'distance'/'edge_id' attributes.

    Examples
    --------
    >>> positions = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)]
    >>> edges = [(0, 1), (1, 2)]
    >>> graph = build_track_graph_from_positions(positions, edges)
    >>> graph.number_of_nodes()
    3
    >>> graph.number_of_edges()
    2

    """
    try:
        from track_linearization import make_track_graph

        # Use track_linearization's make_track_graph for proper attributes
        return make_track_graph(node_positions=node_positions, edges=edges)
    except ImportError:
        # Fallback: build graph manually if track_linearization not installed
        graph = nx.Graph()

        # Add nodes with positions
        for i, pos in enumerate(node_positions):
            graph.add_node(i, pos=pos)

        # Add edges with distance and edge_id
        for edge_id, (n1, n2) in enumerate(edges):
            pos1 = np.array(node_positions[n1])
            pos2 = np.array(node_positions[n2])
            distance = float(np.linalg.norm(pos2 - pos1))
            graph.add_edge(n1, n2, distance=distance, edge_id=edge_id)

        return graph


class TrackGraphResult(NamedTuple):
    """Result from track graph annotation session.

    Contains the track graph, node positions, edges, and metadata needed
    to create an Environment via `Environment.from_graph()`.

    Attributes
    ----------
    track_graph : nx.Graph or None
        NetworkX graph with node 'pos' and edge 'distance'/'edge_id' attributes.
        Created via track_linearization.make_track_graph() if available.
        None if insufficient nodes/edges for a valid graph.
    node_positions : list of tuple
        Node coordinates in output units (cm if calibrated, else pixels).
    edges : list of tuple
        Edge connections as (node_i, node_j) tuples.
    edge_order : list of tuple
        Ordered edge list for linearization (from infer_edge_layout).
    edge_spacing : NDArray[np.float64]
        Spacing between consecutive edges (from infer_edge_layout).
    node_labels : list of str
        Labels for each node (e.g., "start", "goal", "junction").
    start_node : int or None
        Designated start node for linearization.
    pixel_positions : list of tuple
        Original node coordinates in pixels (before calibration transform).

    Examples
    --------
    >>> result = build_track_graph_result(state, calibration=None)
    >>> if result.track_graph is not None:
    ...     env = result.to_environment(bin_size=2.0)

    """

    track_graph: nx.Graph | None
    node_positions: list[tuple[float, float]]
    edges: list[tuple[int, int]]
    edge_order: list[tuple[int, int]]
    edge_spacing: NDArray[np.float64]
    node_labels: list[str]
    start_node: int | None
    pixel_positions: list[tuple[float, float]]

    def to_environment(
        self,
        bin_size: float,
        edge_spacing: float | list[float] | NDArray[np.float64] | None = None,
        name: str = "",
    ) -> Environment:
        """Create Environment from annotated track graph.

        Parameters
        ----------
        bin_size : float
            Bin size for discretization along the track.
        edge_spacing : float or array-like, optional
            Override inferred edge_spacing. If None, uses self.edge_spacing.
        name : str, optional
            Name for the environment.

        Returns
        -------
        Environment
            Environment with GraphLayout for 1D linearized track.

        Raises
        ------
        ValueError
            If track_graph is None (insufficient nodes/edges).

        Examples
        --------
        >>> env = result.to_environment(bin_size=2.0)
        >>> env = result.to_environment(bin_size=2.0, name="linear_track")

        """
        from neurospatial.environment import Environment

        if self.track_graph is None:
            raise ValueError(
                "Cannot create Environment: no track graph. "
                "Ensure at least 2 nodes and 1 edge were created.",
            )

        # Use provided edge_spacing or fall back to inferred spacing
        spacing = edge_spacing if edge_spacing is not None else self.edge_spacing

        # Convert numpy array to list for type compatibility with from_graph
        if isinstance(spacing, np.ndarray):
            spacing_arg: float | list[float] = spacing.tolist()
        elif isinstance(spacing, (float, int)):
            spacing_arg = float(spacing)
        else:
            spacing_arg = list(spacing)

        return Environment.from_graph(
            graph=self.track_graph,
            edge_order=self.edge_order,
            edge_spacing=spacing_arg,
            bin_size=bin_size,
            name=name,
        )


def build_track_graph_result(
    state: TrackBuilderState,
    calibration: VideoCalibration | None,
) -> TrackGraphResult:
    """Build TrackGraphResult from annotation state.

    Transforms coordinates using calibration and builds the track graph
    using track_linearization.make_track_graph().

    Parameters
    ----------
    state : TrackBuilderState
        Final state from annotation session containing nodes, edges, and labels.
    calibration : VideoCalibration or None
        Pixel-to-cm transform. If None, coordinates stay in pixels.

    Returns
    -------
    TrackGraphResult
        Complete result with track graph and all metadata.

    Notes
    -----
    The track_graph node 'pos' attributes use the SAME coordinate system
    as node_positions (cm if calibrated, else pixels). This ensures
    consistency when using Environment.from_graph().

    If the state has fewer than 2 nodes or no edges, track_graph will be
    None and edge_order/edge_spacing will be empty.

    Examples
    --------
    >>> state = TrackBuilderState()
    >>> state.add_node(0.0, 0.0, label="start")
    0
    >>> state.add_node(100.0, 0.0, label="end")
    1
    >>> state.add_edge(0, 1)
    True
    >>> result = build_track_graph_result(state, calibration=None)
    >>> result.node_positions
    [(0.0, 0.0), (100.0, 0.0)]

    """
    # Store original pixel positions
    pixel_positions = list(state.nodes)

    # Transform to output coordinates (cm if calibrated)
    node_positions = transform_nodes_to_output(state.nodes, calibration)

    # Copy edges from state
    edges = list(state.edges)

    # Copy node labels
    node_labels = list(state.node_labels)

    # Get effective start node (defaults to 0 if not set)
    start_node = state.get_effective_start_node()

    # Build track graph and infer edge layout if we have enough data
    if len(state.nodes) >= 2 and len(state.edges) >= 1:
        track_graph = build_track_graph_from_positions(node_positions, edges)

        # Use manual overrides if set, otherwise auto-infer
        if state.edge_order_override is not None:
            edge_order = list(state.edge_order_override)
        else:
            try:
                from track_linearization import infer_edge_layout

                edge_order, _spacing = infer_edge_layout(
                    track_graph,
                    start_node=start_node,
                )
            except ImportError:
                # Fallback: use edges in creation order
                edge_order = edges

        if state.edge_spacing_override is not None:
            edge_spacing = np.array(state.edge_spacing_override, dtype=np.float64)
        else:
            try:
                from track_linearization import infer_edge_layout

                _order, edge_spacing = infer_edge_layout(
                    track_graph,
                    start_node=start_node,
                )
            except ImportError:
                # Fallback: zero spacing
                edge_spacing = np.zeros(max(0, len(edges) - 1), dtype=np.float64)
    else:
        track_graph = None
        edge_order = []
        edge_spacing = np.array([], dtype=np.float64)

    return TrackGraphResult(
        track_graph=track_graph,
        node_positions=node_positions,
        edges=edges,
        edge_order=edge_order,
        edge_spacing=edge_spacing,
        node_labels=node_labels,
        start_node=start_node,
        pixel_positions=pixel_positions,
    )
