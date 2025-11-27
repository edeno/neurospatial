"""Track graph builder widget and layer setup for napari.

This module provides napari layers and widgets for interactively building
track graphs on video frames. The track graph can be used with
`Environment.from_graph()` to create 1D linearized track environments.

Notes
-----
This module requires napari to be installed. All napari-dependent code
is imported lazily to allow the rest of the annotation module to work
without napari.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    import napari
    from napari.layers import Points, Shapes

    from neurospatial.annotation._track_state import TrackBuilderState


# =============================================================================
# Colorblind-Safe Color Palette
# =============================================================================
# Based on Tab10 colormap, verified for colorblind accessibility
# See: https://jfly.uni-koeln.de/color/

NODE_COLOR: str = "#1f77b4"
"""Blue - Default color for track nodes."""

EDGE_COLOR: str = "#ff7f0e"
"""Orange - Color for track edges."""

START_NODE_COLOR: str = "#2ca02c"
"""Green - Color for the designated start node."""

SELECTED_COLOR: str = "#d62728"
"""Red - Highlight color for selected items."""

PREVIEW_COLOR: str = "#7f7f7f"
"""Gray - Color for edge preview line (dashed)."""


# =============================================================================
# Layer Setup
# =============================================================================


def setup_track_layers(viewer: napari.Viewer) -> tuple[Shapes, Points]:
    """Create napari layers for track graph annotation.

    Creates two layers in the viewer:
    1. Shapes layer for track edges (middle z-order)
    2. Points layer for track nodes (top z-order, interactive)

    The layers are ordered so that nodes are on top and clickable,
    while edges appear below for visual clarity.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance to add layers to.

    Returns
    -------
    edges_layer : napari.layers.Shapes
        Shapes layer for drawing track edges as paths.
    nodes_layer : napari.layers.Points
        Points layer for placing track nodes.

    Examples
    --------
    >>> import napari
    >>> viewer = napari.Viewer()  # doctest: +SKIP
    >>> edges, nodes = setup_track_layers(viewer)  # doctest: +SKIP
    >>> nodes.add([[100, 100], [200, 150]])  # Add waypoints  # doctest: +SKIP
    >>> edges.add_paths([[[100, 100], [200, 150]]])  # Connect them  # doctest: +SKIP
    """
    # Edges layer (middle - below nodes for clickability)
    # Uses Shapes with path type for line segments between nodes
    edges_layer = viewer.add_shapes(
        name="Track Edges",
        shape_type="path",
        edge_color=EDGE_COLOR,
        edge_width=3,
    )

    # Nodes layer (top - interactive, clickable)
    # Points layer for track waypoints
    # Use border_width_is_relative=False for absolute pixel border widths
    nodes_layer = viewer.add_points(
        name="Track Nodes",
        size=15,
        face_color=NODE_COLOR,
        border_color="white",
        border_width=2,
        border_width_is_relative=False,
    )

    return edges_layer, nodes_layer


# =============================================================================
# Constants
# =============================================================================

# Size constants for nodes
DEFAULT_NODE_SIZE: float = 15.0
"""Default size for track nodes in pixels."""

START_NODE_SIZE: float = 20.0
"""Larger size for start node (for visibility)."""

# Click detection threshold
CLICK_THRESHOLD: float = 20.0
"""Distance threshold for detecting clicks on nodes (pixels)."""


# =============================================================================
# Color Utilities
# =============================================================================


def _hex_to_rgba(hex_color: str) -> np.ndarray:
    """Convert hex color string to RGBA array.

    Parameters
    ----------
    hex_color : str
        Hex color string (e.g., "#1f77b4" or "#1f77b4ff").

    Returns
    -------
    np.ndarray
        RGBA array with values in [0, 1], shape (4,).
    """
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 6:
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        a = 1.0
    elif len(hex_color) == 8:
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        a = int(hex_color[6:8], 16) / 255.0
    else:
        raise ValueError(f"Invalid hex color: {hex_color}")
    return np.array([r, g, b, a], dtype=np.float64)


# =============================================================================
# Layer Synchronization
# =============================================================================


def _sync_layers_from_state(
    state: TrackBuilderState,
    nodes_layer: Points,
    edges_layer: Shapes,
) -> None:
    """Synchronize napari layers with TrackBuilderState.

    Updates the nodes and edges layers to reflect the current state.
    The start node is highlighted with a different color and larger size.

    Parameters
    ----------
    state : TrackBuilderState
        The current state containing nodes and edges.
    nodes_layer : napari.layers.Points
        The Points layer for track nodes.
    edges_layer : napari.layers.Shapes
        The Shapes layer for track edges.

    Notes
    -----
    Coordinates are converted from state (x, y) format to napari (row, col)
    format by swapping axes: napari uses (y, x) ordering.
    """
    # Convert state nodes (x, y) to napari (row, col) = (y, x)
    if state.nodes:
        # Stack as (row, col) = (y, x)
        node_data = np.array([(y, x) for x, y in state.nodes], dtype=np.float64)

        # Build per-node colors (as RGBA arrays) and sizes
        n_nodes = len(state.nodes)
        node_rgba = _hex_to_rgba(NODE_COLOR)
        start_rgba = _hex_to_rgba(START_NODE_COLOR)

        # Create color array (n_nodes, 4)
        face_colors = np.tile(node_rgba, (n_nodes, 1))
        sizes = np.full(n_nodes, DEFAULT_NODE_SIZE, dtype=np.float64)

        # Highlight start node
        if state.start_node is not None and 0 <= state.start_node < n_nodes:
            face_colors[state.start_node] = start_rgba
            sizes[state.start_node] = START_NODE_SIZE

        # Update nodes layer
        nodes_layer.data = node_data
        nodes_layer.face_color = face_colors
        nodes_layer.size = sizes
    else:
        # Clear nodes layer
        nodes_layer.data = np.empty((0, 2), dtype=np.float64)

    # Convert edges to paths for shapes layer
    if state.edges and state.nodes:
        paths = []
        for n1, n2 in state.edges:
            if n1 < len(state.nodes) and n2 < len(state.nodes):
                x1, y1 = state.nodes[n1]
                x2, y2 = state.nodes[n2]
                # Convert to napari (row, col) = (y, x)
                path = np.array([[y1, x1], [y2, x2]], dtype=np.float64)
                paths.append(path)

        if paths:
            # Clear and re-add all paths
            edges_layer.data = []
            for path in paths:
                edges_layer.add_paths([path])
        else:
            edges_layer.data = []
    else:
        # Clear edges layer
        edges_layer.data = []


# =============================================================================
# Click Handlers
# =============================================================================


def _handle_click(
    state: TrackBuilderState,
    nodes_layer: Points,
    edges_layer: Shapes,
    position: tuple[float, float],
    threshold: float = CLICK_THRESHOLD,
) -> None:
    """Handle mouse click based on current mode.

    In add_node mode: adds a new node at the click position.
    In delete mode: deletes the nearest node within threshold.
    In add_edge mode: implements two-click edge creation pattern.

    Parameters
    ----------
    state : TrackBuilderState
        The current state to modify.
    nodes_layer : napari.layers.Points
        The Points layer for track nodes.
    edges_layer : napari.layers.Shapes
        The Shapes layer for track edges.
    position : tuple[float, float]
        Click position as (row, col) in napari coordinates.
    threshold : float, optional
        Distance threshold for node detection (default: CLICK_THRESHOLD).
    """
    # Convert napari (row, col) to state (x, y)
    row, col = position
    x, y = col, row  # Swap: x = col, y = row

    if state.mode == "add_node":
        state.add_node(x, y)
        _sync_layers_from_state(state, nodes_layer, edges_layer)

    elif state.mode == "delete":
        nearest_idx = state.find_nearest_node(x, y, threshold)
        if nearest_idx is not None:
            state.delete_node(nearest_idx)
            _sync_layers_from_state(state, nodes_layer, edges_layer)

    elif state.mode == "add_edge":
        # Find nearest node to click position
        nearest_idx = state.find_nearest_node(x, y, threshold)

        if nearest_idx is None:
            # Click away from any node - do nothing
            return

        if state.edge_start_node is None:
            # First click - select start node for edge
            state.edge_start_node = nearest_idx
        else:
            # Second click - create edge
            if state.edge_start_node != nearest_idx:
                # Different node - create edge
                state.add_edge(state.edge_start_node, nearest_idx)
            # Clear edge start regardless (even for self-loop attempt)
            state.edge_start_node = None
            _clear_edge_preview(edges_layer)
            _sync_layers_from_state(state, nodes_layer, edges_layer)


# =============================================================================
# Edge Preview
# =============================================================================

# Track preview shape index for removal
_PREVIEW_INDEX: int | None = None


def _show_edge_preview(
    state: TrackBuilderState,
    edges_layer: Shapes,
    cursor_position: tuple[float, float],
) -> None:
    """Show preview line from edge start node to cursor position.

    Parameters
    ----------
    state : TrackBuilderState
        The current state containing edge_start_node.
    edges_layer : napari.layers.Shapes
        The Shapes layer to add preview to.
    cursor_position : tuple[float, float]
        Current cursor position as (row, col) in napari coordinates.
    """
    global _PREVIEW_INDEX

    if state.edge_start_node is None or state.edge_start_node >= len(state.nodes):
        return

    # Get start node position
    x1, y1 = state.nodes[state.edge_start_node]
    # Convert to napari (row, col)
    start_point = [y1, x1]

    # End point is cursor position (already in napari coords)
    end_point = list(cursor_position)

    # Create preview path
    preview_path = np.array([start_point, end_point], dtype=np.float64)

    # Remove previous preview if exists
    _clear_edge_preview(edges_layer)

    # Add new preview path with preview color
    edges_layer.add_paths([preview_path], edge_color=PREVIEW_COLOR)
    _PREVIEW_INDEX = len(edges_layer.data) - 1


def _clear_edge_preview(edges_layer: Shapes) -> None:
    """Remove the edge preview line from the edges layer.

    Parameters
    ----------
    edges_layer : napari.layers.Shapes
        The Shapes layer containing the preview.
    """
    global _PREVIEW_INDEX

    if len(edges_layer.data) > 0:
        # Remove all data (we'll re-sync from state)
        edges_layer.data = []
    _PREVIEW_INDEX = None


# =============================================================================
# Edge Creation Cancel
# =============================================================================


def _cancel_edge_creation(state: TrackBuilderState) -> None:
    """Cancel in-progress edge creation.

    Parameters
    ----------
    state : TrackBuilderState
        The current state to modify.
    """
    state.edge_start_node = None


# =============================================================================
# Keyboard Shortcuts
# =============================================================================


def _handle_key(
    state: TrackBuilderState,
    key: str,
    modifiers: Sequence[str] | None = None,
    selected_node: int | None = None,
) -> str | None:
    """Handle keyboard shortcut.

    Parameters
    ----------
    state : TrackBuilderState
        The current state to modify.
    key : str
        The key pressed (e.g., "A", "E", "X", "Z", "Escape").
    modifiers : Sequence[str], optional
        Modifier keys held (e.g., ["Control"], ["Control", "Shift"]).
    selected_node : int, optional
        Index of currently selected node (for Shift+S).

    Returns
    -------
    str or None
        Action taken: "undo", "redo", "cancel", "set_start", or None.
    """
    if modifiers is None:
        modifiers = []

    key_upper = key.upper()
    has_control = "Control" in modifiers
    has_shift = "Shift" in modifiers

    # Mode switching shortcuts (A, E, X)
    if key_upper == "A" and not has_control and not has_shift:
        state.mode = "add_node"
        return None

    if key_upper == "E" and not has_control and not has_shift:
        state.mode = "add_edge"
        return None

    if key_upper == "X" and not has_control and not has_shift:
        state.mode = "delete"
        return None

    # Undo/Redo (Ctrl+Z, Ctrl+Shift+Z)
    if key_upper == "Z" and has_control:
        if has_shift:
            state.redo()
            return "redo"
        else:
            state.undo()
            return "undo"

    # Cancel (Escape)
    if key == "Escape":
        _cancel_edge_creation(state)
        return "cancel"

    # Set start node (Shift+S)
    if (
        key_upper == "S"
        and has_shift
        and not has_control
        and selected_node is not None
        and 0 <= selected_node < len(state.nodes)
    ):
        state.set_start_node(selected_node)
        return "set_start"

    return None
