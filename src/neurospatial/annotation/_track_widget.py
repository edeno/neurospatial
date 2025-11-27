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

from typing import TYPE_CHECKING, Any

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


# =============================================================================
# Control Widget
# =============================================================================


def create_track_widget(
    viewer: napari.Viewer,
    edges_layer: Shapes,
    nodes_layer: Points,
    state: TrackBuilderState,
) -> TrackGraphWidget:
    """Create the track graph builder control widget.

    Creates a Qt widget with controls for building track graphs interactively.
    The widget includes:

    - Mode selector (add_node, add_edge, delete)
    - Node and edge list views
    - Node label input
    - Set Start Node button
    - Delete buttons
    - Validation status
    - Save and Close button
    - Help text panel

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance.
    edges_layer : napari.layers.Shapes
        The Shapes layer for track edges.
    nodes_layer : napari.layers.Points
        The Points layer for track nodes.
    state : TrackBuilderState
        The shared state object (modified in place by handlers).

    Returns
    -------
    TrackGraphWidget
        The control widget (QWidget subclass).

    Examples
    --------
    >>> import napari  # doctest: +SKIP
    >>> viewer = napari.Viewer()  # doctest: +SKIP
    >>> edges, nodes = setup_track_layers(viewer)  # doctest: +SKIP
    >>> state = TrackBuilderState()  # doctest: +SKIP
    >>> widget = create_track_widget(viewer, edges, nodes, state)  # doctest: +SKIP
    >>> viewer.window.add_dock_widget(widget)  # doctest: +SKIP
    """
    return TrackGraphWidget(viewer, edges_layer, nodes_layer, state)


class TrackGraphWidget:
    """Control widget for track graph annotation.

    This class wraps a Qt widget and provides the interface for
    interacting with the track graph builder.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance.
    edges_layer : napari.layers.Shapes
        The Shapes layer for track edges.
    nodes_layer : napari.layers.Points
        The Points layer for track nodes.
    state : TrackBuilderState
        The shared state object.
    """

    def __init__(
        self,
        viewer: napari.Viewer,
        edges_layer: Shapes,
        nodes_layer: Points,
        state: TrackBuilderState,
    ) -> None:
        from qtpy.QtWidgets import (
            QComboBox,
            QDoubleSpinBox,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QListWidget,
            QPushButton,
            QVBoxLayout,
            QWidget,
        )

        self._viewer = viewer
        self._edges_layer = edges_layer
        self._nodes_layer = nodes_layer
        self._state = state
        self._saved = False

        # Create main widget
        self._widget = QWidget()
        layout = QVBoxLayout()
        self._widget.setLayout(layout)

        # Title
        title_label = QLabel("<b>Track Graph Builder</b>")
        layout.addWidget(title_label)

        # Help text
        help_text = QLabel(
            "1. Press A → Click to add nodes\n"
            "2. Press E → Click two nodes to connect\n"
            "3. Press X → Click node/edge to delete\n"
            "4. Select node → Shift+S to set as start\n"
            "5. Ctrl+Enter to save\n\n"
            "Shortcuts: A (add) | E (edge) | X (delete) | Ctrl+Z (undo)"
        )
        help_text.setWordWrap(True)
        layout.addWidget(help_text)

        # Mode selector group
        mode_group = QGroupBox("Mode")
        mode_layout = QHBoxLayout()
        mode_group.setLayout(mode_layout)

        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["add_node", "add_edge", "delete"])
        self._mode_combo.setCurrentText(state.mode)
        self._mode_combo.currentTextChanged.connect(self._on_mode_changed)
        mode_layout.addWidget(self._mode_combo)

        layout.addWidget(mode_group)

        # Status label
        self._status_label = QLabel(f"Mode: {state.mode}")
        layout.addWidget(self._status_label)

        # Node list group
        node_group = QGroupBox("Nodes")
        node_layout = QVBoxLayout()
        node_group.setLayout(node_layout)

        self._node_combo = QComboBox()
        node_layout.addWidget(self._node_combo)

        # Node label input
        label_layout = QHBoxLayout()
        self._label_input = QLineEdit()
        self._label_input.setPlaceholderText("Node label...")
        label_layout.addWidget(self._label_input)

        self._apply_label_btn = QPushButton("Apply Label")
        self._apply_label_btn.clicked.connect(self._on_apply_label)
        label_layout.addWidget(self._apply_label_btn)
        node_layout.addLayout(label_layout)

        # Node buttons
        node_btn_layout = QHBoxLayout()

        self._set_start_btn = QPushButton("Set as Start")
        self._set_start_btn.clicked.connect(self._on_set_start)
        node_btn_layout.addWidget(self._set_start_btn)

        self._delete_node_btn = QPushButton("Delete Node")
        self._delete_node_btn.clicked.connect(self._on_delete_node)
        node_btn_layout.addWidget(self._delete_node_btn)

        node_layout.addLayout(node_btn_layout)
        layout.addWidget(node_group)

        # Edge list group
        edge_group = QGroupBox("Edges")
        edge_layout = QVBoxLayout()
        edge_group.setLayout(edge_layout)

        self._edge_combo = QComboBox()
        edge_layout.addWidget(self._edge_combo)

        self._delete_edge_btn = QPushButton("Delete Edge")
        self._delete_edge_btn.clicked.connect(self._on_delete_edge)
        edge_layout.addWidget(self._delete_edge_btn)

        layout.addWidget(edge_group)

        # Edge Order group
        edge_order_group = QGroupBox("Edge Order")
        edge_order_layout = QVBoxLayout()
        edge_order_group.setLayout(edge_order_layout)

        # Edge order list widget
        self._edge_order_list = QListWidget()
        edge_order_layout.addWidget(self._edge_order_list)

        # Move buttons
        move_btn_layout = QHBoxLayout()

        self._move_up_btn = QPushButton("▲ Up")
        self._move_up_btn.clicked.connect(self._on_move_up)
        move_btn_layout.addWidget(self._move_up_btn)

        self._move_down_btn = QPushButton("▼ Down")
        self._move_down_btn.clicked.connect(self._on_move_down)
        move_btn_layout.addWidget(self._move_down_btn)

        edge_order_layout.addLayout(move_btn_layout)

        # Reset to Auto button
        self._reset_to_auto_btn = QPushButton("Reset to Auto")
        self._reset_to_auto_btn.clicked.connect(self._on_reset_to_auto)
        edge_order_layout.addWidget(self._reset_to_auto_btn)

        # Edge spacing input
        spacing_layout = QHBoxLayout()
        spacing_label = QLabel("Edge Spacing:")
        spacing_layout.addWidget(spacing_label)

        self._edge_spacing_spin = QDoubleSpinBox()
        self._edge_spacing_spin.setRange(0.0, 1000.0)
        self._edge_spacing_spin.setDecimals(2)
        self._edge_spacing_spin.setSingleStep(0.1)
        self._edge_spacing_spin.setValue(0.0)
        spacing_layout.addWidget(self._edge_spacing_spin)

        self._apply_spacing_btn = QPushButton("Apply")
        self._apply_spacing_btn.clicked.connect(self._on_apply_spacing)
        spacing_layout.addWidget(self._apply_spacing_btn)

        edge_order_layout.addLayout(spacing_layout)

        layout.addWidget(edge_order_group)

        # Validation status
        self._validation_label = QLabel("")
        layout.addWidget(self._validation_label)

        # Save and Close button
        self._save_btn = QPushButton("Save and Close")
        self._save_btn.clicked.connect(self._on_save)
        layout.addWidget(self._save_btn)

        # Add stretch to push everything up
        layout.addStretch()

        # Initial sync
        self.sync_from_state()

    # -------------------------------------------------------------------------
    # Properties for test access
    # -------------------------------------------------------------------------

    @property
    def mode_selector(self) -> _ModeSelector:
        """Mode selector accessor for tests."""
        return _ModeSelector(self._mode_combo)

    @property
    def node_list(self) -> _NodeList:
        """Node list accessor for tests."""
        return _NodeList(self._node_combo, self._state)

    @property
    def edge_list(self) -> _EdgeList:
        """Edge list accessor for tests."""
        return _EdgeList(self._edge_combo, self._state)

    @property
    def status_label(self) -> _StatusLabel:
        """Status label accessor for tests."""
        return _StatusLabel(self._status_label)

    @property
    def set_start_button(self) -> _Button:
        """Set start button accessor for tests."""
        return _Button(self._set_start_btn)

    @property
    def delete_node_button(self) -> _Button:
        """Delete node button accessor for tests."""
        return _Button(self._delete_node_btn)

    @property
    def delete_edge_button(self) -> _Button:
        """Delete edge button accessor for tests."""
        return _Button(self._delete_edge_btn)

    @property
    def node_label_input(self) -> _LineEdit:
        """Node label input accessor for tests."""
        return _LineEdit(self._label_input)

    @property
    def apply_label_button(self) -> _Button:
        """Apply label button accessor for tests."""
        return _Button(self._apply_label_btn)

    @property
    def edge_order_list(self) -> _EdgeOrderList:
        """Edge order list accessor for tests."""
        return _EdgeOrderList(self._edge_order_list, self._state, self)

    @property
    def move_up_button(self) -> _Button:
        """Move up button accessor for tests."""
        return _Button(self._move_up_btn)

    @property
    def move_down_button(self) -> _Button:
        """Move down button accessor for tests."""
        return _Button(self._move_down_btn)

    @property
    def reset_to_auto_button(self) -> _Button:
        """Reset to auto button accessor for tests."""
        return _Button(self._reset_to_auto_btn)

    @property
    def edge_spacing_input(self) -> _EdgeSpacingInput:
        """Edge spacing input accessor for tests."""
        return _EdgeSpacingInput(self._edge_spacing_spin, self._state)

    @property
    def apply_edge_spacing_button(self) -> _Button:
        """Apply edge spacing button accessor for tests."""
        return _Button(self._apply_spacing_btn)

    # -------------------------------------------------------------------------
    # QWidget compatibility
    # -------------------------------------------------------------------------

    def findChildren(self, type_: type, name: str | None = None):  # noqa: N802
        """Proxy to underlying widget's findChildren."""
        return self._widget.findChildren(type_, name)

    def __getattr__(self, name: str):
        """Proxy attribute access to underlying widget."""
        return getattr(self._widget, name)

    # -------------------------------------------------------------------------
    # Event handlers
    # -------------------------------------------------------------------------

    def _on_mode_changed(self, mode: str) -> None:
        """Handle mode selector change."""
        # Validate mode is a valid TrackGraphMode
        valid_modes = ("add_node", "add_edge", "delete")
        if mode not in valid_modes:
            return  # Ignore invalid modes
        self._state.mode = mode  # type: ignore[assignment]  # QComboBox returns str
        self._update_status()
        _sync_layers_from_state(self._state, self._nodes_layer, self._edges_layer)

    def _on_set_start(self) -> None:
        """Handle Set as Start button click."""
        idx = self._node_combo.currentIndex()
        if idx >= 0 and idx < len(self._state.nodes):
            self._state.set_start_node(idx)
            self.sync_from_state()

    def _on_delete_node(self) -> None:
        """Handle Delete Node button click."""
        idx = self._node_combo.currentIndex()
        if idx >= 0 and idx < len(self._state.nodes):
            self._state.delete_node(idx)
            self.sync_from_state()

    def _on_delete_edge(self) -> None:
        """Handle Delete Edge button click."""
        idx = self._edge_combo.currentIndex()
        if idx >= 0 and idx < len(self._state.edges):
            self._state.delete_edge(idx)
            self.sync_from_state()

    def _on_apply_label(self) -> None:
        """Handle Apply Label button click."""
        idx = self._node_combo.currentIndex()
        if idx >= 0 and idx < len(self._state.nodes):
            label = self._label_input.text()
            # Update label in state
            while len(self._state.node_labels) <= idx:
                self._state.node_labels.append("")
            self._state.node_labels[idx] = label
            self.sync_from_state()

    def _on_save(self) -> None:
        """Handle Save and Close button click."""
        if self.try_save():
            self._viewer.close()

    def _on_move_up(self) -> None:
        """Handle Move Up button click."""
        idx = self._edge_order_list.currentRow()
        if idx > 0:
            self._move_edge_order_item(idx, idx - 1)

    def _on_move_down(self) -> None:
        """Handle Move Down button click."""
        idx = self._edge_order_list.currentRow()
        if idx >= 0 and idx < self._edge_order_list.count() - 1:
            self._move_edge_order_item(idx, idx + 1)

    def _move_edge_order_item(self, from_idx: int, to_idx: int) -> None:
        """Move edge order item from one position to another."""
        # Get current edge order (compute if needed)
        current_order = self._get_current_edge_order()
        if len(current_order) <= 1:
            return

        # Swap items
        item = current_order.pop(from_idx)
        current_order.insert(to_idx, item)

        # Update state with manual override
        self._state.set_edge_order(current_order)
        self.sync_from_state()

    def _get_current_edge_order(self) -> list[tuple[int, int]]:
        """Get current edge order (from override or computed)."""
        if self._state.edge_order_override is not None:
            return list(self._state.edge_order_override)
        return self._state.compute_edge_order()

    def _on_reset_to_auto(self) -> None:
        """Handle Reset to Auto button click."""
        self._state.clear_edge_order()
        self._state.clear_edge_spacing()
        self.sync_from_state()

    def _on_apply_spacing(self) -> None:
        """Handle Apply edge spacing button click."""
        # Get the selected edge index in the order list
        idx = self._edge_order_list.currentRow()
        if idx < 0:
            return

        # Get current edge spacing (n_edges - 1 values)
        current_order = self._get_current_edge_order()
        n_spacing = max(0, len(current_order) - 1)

        if n_spacing == 0:
            return

        # Get current spacing values (from override or computed)
        if self._state.edge_spacing_override is not None:
            spacing = list(self._state.edge_spacing_override)
        else:
            spacing = list(self._state.compute_edge_spacing())

        # Ensure spacing list has correct length
        while len(spacing) < n_spacing:
            spacing.append(0.0)

        # Update spacing for the gap before the selected edge
        # (spacing[i] is the gap between edge[i] and edge[i+1])
        spacing_idx = min(idx, n_spacing - 1)
        if spacing_idx >= 0:
            spacing[spacing_idx] = self._edge_spacing_spin.value()

        self._state.set_edge_spacing(spacing)
        self.sync_from_state()

    # -------------------------------------------------------------------------
    # Public methods
    # -------------------------------------------------------------------------

    def sync_from_state(self) -> None:
        """Synchronize widget UI from current state."""
        # Update mode selector
        self._mode_combo.blockSignals(True)
        self._mode_combo.setCurrentText(self._state.mode)
        self._mode_combo.blockSignals(False)

        # Update status
        self._update_status()

        # Update node list
        self._node_combo.clear()
        for i, _node in enumerate(self._state.nodes):
            label = ""
            if i < len(self._state.node_labels) and self._state.node_labels[i]:
                label = f" ({self._state.node_labels[i]})"
            start_marker = " [START]" if i == self._state.start_node else ""
            self._node_combo.addItem(f"Node {i}{label}{start_marker}")

        # Update edge list
        self._edge_combo.clear()
        for i, edge in enumerate(self._state.edges):
            self._edge_combo.addItem(f"Edge {i}: {edge[0]} → {edge[1]}")

        # Update edge order list
        self._edge_order_list.clear()
        edge_order = self._get_current_edge_order()
        for i, edge in enumerate(edge_order):
            self._edge_order_list.addItem(f"{i + 1}. ({edge[0]}, {edge[1]})")

        # Update validation status
        _is_valid, errors, warnings = self.get_validation_status()
        if errors:
            self._validation_label.setText(f"❌ {errors[0]}")
            self._validation_label.setStyleSheet("color: red;")
        elif warnings:
            self._validation_label.setText(f"⚠️ {warnings[0]}")
            self._validation_label.setStyleSheet("color: orange;")
        else:
            self._validation_label.setText("✓ Valid")
            self._validation_label.setStyleSheet("color: green;")

        # Sync napari layers
        _sync_layers_from_state(self._state, self._nodes_layer, self._edges_layer)

    def _update_status(self) -> None:
        """Update status label with current mode."""
        mode_display = self._state.mode.replace("_", " ").title()
        self._status_label.setText(f"Mode: {mode_display}")

    def get_validation_status(self) -> tuple[bool, list[str], list[str]]:
        """Get current validation status.

        Returns
        -------
        is_valid : bool
            Whether the current state is valid for saving.
        errors : list[str]
            List of error messages (blocking).
        warnings : list[str]
            List of warning messages (non-blocking).
        """
        return self._state.is_valid_for_save()

    def try_save(self) -> bool:
        """Attempt to save the current state.

        Returns
        -------
        bool
            True if save succeeded, False if validation failed.
        """
        from qtpy.QtWidgets import QMessageBox

        is_valid, errors, warnings = self.get_validation_status()

        if not is_valid:
            QMessageBox.warning(
                self._widget,
                "Validation Failed",
                "Cannot save:\n• " + "\n• ".join(errors),
            )
            return False

        # Show warnings but allow save
        if warnings:
            QMessageBox.information(
                self._widget,
                "Warnings",
                "Saved with warnings:\n• " + "\n• ".join(warnings),
            )

        # Mark as saved
        self._saved = True
        return True


# =============================================================================
# Test accessor classes
# =============================================================================


class _ModeSelector:
    """Accessor for mode selector in tests."""

    def __init__(self, combo) -> None:
        self._combo = combo

    @property
    def value(self) -> str:
        return str(self._combo.currentText())

    @value.setter
    def value(self, mode: str) -> None:
        self._combo.setCurrentText(mode)


class _NodeList:
    """Accessor for node list in tests."""

    def __init__(self, combo, state) -> None:
        self._combo = combo
        self._state = state

    @property
    def choices(self) -> list:
        return [self._combo.itemText(i) for i in range(self._combo.count())]

    @property
    def value(self) -> int | None:
        idx = self._combo.currentIndex()
        return idx if idx >= 0 else None

    @value.setter
    def value(self, idx: int) -> None:
        self._combo.setCurrentIndex(idx)


class _EdgeList:
    """Accessor for edge list in tests."""

    def __init__(self, combo, state) -> None:
        self._combo = combo
        self._state = state

    @property
    def choices(self) -> list:
        return [self._combo.itemText(i) for i in range(self._combo.count())]

    @property
    def value(self) -> int | None:
        idx = self._combo.currentIndex()
        return idx if idx >= 0 else None

    @value.setter
    def value(self, idx: int) -> None:
        self._combo.setCurrentIndex(idx)


class _StatusLabel:
    """Accessor for status label in tests."""

    def __init__(self, label) -> None:
        self._label = label

    def text(self) -> str:
        return str(self._label.text())


class _Button:
    """Accessor for buttons in tests."""

    def __init__(self, btn) -> None:
        self._btn = btn

    def click(self) -> None:
        self._btn.click()


class _LineEdit:
    """Accessor for line edit in tests."""

    def __init__(self, edit) -> None:
        self._edit = edit

    @property
    def value(self) -> str:
        return str(self._edit.text())

    @value.setter
    def value(self, text: str) -> None:
        self._edit.setText(text)


class _EdgeOrderList:
    """Accessor for edge order list in tests."""

    def __init__(
        self,
        list_widget: Any,
        state: TrackBuilderState,
        widget: TrackGraphWidget,
    ) -> None:
        self._list = list_widget
        self._state = state
        self._widget = widget

    @property
    def items(self) -> list[tuple[int, int]]:
        """Get current edge order as list of edge tuples."""
        result: list[tuple[int, int]] = self._widget._get_current_edge_order()
        return result

    def select(self, idx: int) -> None:
        """Select item at index."""
        self._list.setCurrentRow(idx)

    def move_item(self, from_idx: int, to_idx: int) -> None:
        """Move item from one position to another."""
        self._widget._move_edge_order_item(from_idx, to_idx)


class _EdgeSpacingInput:
    """Accessor for edge spacing input in tests."""

    def __init__(self, spin, state) -> None:
        self._spin = spin
        self._state = state

    def set_value(self, spacing_idx: int, value: float) -> None:
        """Set spacing value for a specific gap between edges."""
        # Set the spin box value
        self._spin.setValue(value)
