"""Track graph annotation entry point.

This module provides the public API for interactive track graph annotation,
allowing users to build track graphs on video frames using napari.

The output integrates with `Environment.from_graph()` for creating 1D
linearized track environments.

Examples
--------
>>> # Annotate track graph on video frame
>>> from neurospatial.annotation import annotate_track_graph
>>> result = annotate_track_graph("maze.mp4")  # doctest: +SKIP
>>> env = result.to_environment(bin_size=2.0)  # doctest: +SKIP

>>> # With calibration for pixel-to-cm conversion
>>> from neurospatial.ops.transforms import VideoCalibration
>>> result = annotate_track_graph("maze.mp4", calibration=calib)  # doctest: +SKIP
>>> # node_positions now in cm

"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

# Re-export TrackGraphResult from helpers module for public API
from neurospatial.annotation._track_helpers import (
    TrackGraphResult,
    build_track_graph_result,
)
from neurospatial.annotation._track_state import TrackBuilderState
from neurospatial.annotation._track_widget import (
    _clear_edge_preview,
    _handle_click,
    _handle_key,
    _sync_layers_from_state,
    create_track_widget,
    setup_track_layers,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from neurospatial.ops.transforms import VideoCalibration

__all__ = [
    "TrackGraphResult",
    "annotate_track_graph",
]


def annotate_track_graph(
    video_path: str | Path | None = None,
    *,
    image: NDArray[np.uint8] | None = None,
    frame_index: int = 0,
    calibration: VideoCalibration | None = None,
    initial_nodes: NDArray[np.float64] | None = None,
    initial_edges: list[tuple[int, int]] | None = None,
    initial_node_labels: list[str] | None = None,
) -> TrackGraphResult:
    """Launch interactive napari annotation to build a track graph.

    Opens a napari viewer with a video frame or image as background.
    Users can:

    - Click to add nodes (waypoints on the track)
    - Click two nodes to create edges (two-click pattern)
    - Delete nodes/edges as needed
    - Designate start node for linearization

    Parameters
    ----------
    video_path : str or Path, optional
        Path to video file. Extracts frame at frame_index.
        Either video_path or image must be provided.
    image : NDArray, optional
        Background image array (H, W, 3) RGB uint8.
        Use this for static images or pre-loaded frames.
        Either video_path or image must be provided.
    frame_index : int, default=0
        Which frame to extract from video (ignored if image provided).
    calibration : VideoCalibration, optional
        Pixel-to-cm transform. If provided, output coordinates are in cm.
    initial_nodes : NDArray, optional
        Pre-existing node positions to display for editing.
        Shape: (n_nodes, 2) with (x, y) coordinates.
    initial_edges : list of tuple, optional
        Pre-existing edge connections as (node_i, node_j) tuples.
    initial_node_labels : list of str, optional
        Labels for initial nodes (e.g., ["start", "junction", "goal"]).

    Returns
    -------
    TrackGraphResult
        Named tuple with track_graph, node_positions, edges, edge_order,
        edge_spacing, node_labels, start_node, and pixel_positions.

    Raises
    ------
    ValueError
        If neither video_path nor image is provided.
    IndexError
        If frame_index is out of range for the video.
    ImportError
        If napari is not installed.
    FileNotFoundError
        If video_path does not exist.

    Examples
    --------
    >>> # From video file
    >>> from neurospatial.annotation import annotate_track_graph
    >>> result = annotate_track_graph("maze.mp4")  # doctest: +SKIP
    >>> env = result.to_environment(bin_size=2.0)  # doctest: +SKIP

    >>> # From static image
    >>> import numpy as np
    >>> img = np.zeros((480, 640, 3), dtype=np.uint8)
    >>> result = annotate_track_graph(image=img)  # doctest: +SKIP

    >>> # With calibration (convert pixels to cm)
    >>> from neurospatial.ops.transforms import VideoCalibration
    >>> result = annotate_track_graph("maze.mp4", calibration=calib)  # doctest: +SKIP
    >>> # node_positions now in cm

    Notes
    -----
    This function blocks until the napari viewer is closed. The viewer runs
    in the same Python process, and the function returns only after the user
    closes it (via the "Save and Close" button or window close).

    Keyboard Shortcuts
    ^^^^^^^^^^^^^^^^^^
    Uses napari's default Points layer keybindings plus custom additions:

    **napari defaults:**
    - 2: Add mode (click to add nodes)
    - 3: Select mode (click to select, Delete to remove)
    - 4: Pan/Zoom mode

    **Custom additions:**
    - E: Edge mode (click two nodes to connect)
    - Shift+S: Set selected node as start
    - Ctrl+Z: Undo
    - Ctrl+Shift+Z: Redo
    - Escape: Cancel edge creation

    See Also
    --------
    TrackGraphResult : Result container with to_environment() method
    Environment.from_graph : Create Environment from track graph

    """
    # 1. Validate inputs
    if video_path is None and image is None:
        raise ValueError(
            "Either video_path or image must be provided. "
            "Provide a video path to extract a frame, or pass an image array directly.",
        )

    # 2. Import napari (lazy to allow module to work without it)
    try:
        import napari
    except ImportError as e:
        raise ImportError(
            "napari is required for interactive annotation. "
            "Install with: pip install napari[all]",
        ) from e

    # 3. Load background image
    if video_path is not None:
        from neurospatial.animation._video_io import VideoReader

        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        reader = VideoReader(str(video_path))
        if frame_index >= reader.n_frames:
            raise IndexError(
                f"Frame index {frame_index} is out of range for video '{video_path.name}'. "
                f"Video has {reader.n_frames} frames (indices 0-{reader.n_frames - 1}).",
            )
        frame = reader[frame_index]  # (H, W, 3) RGB uint8
    else:
        # image must be non-None since we validated at function start
        assert image is not None  # Help mypy understand the validation
        frame = image

    # 4. Create napari viewer with video frame as bottom layer
    viewer = napari.Viewer(title="Track Graph Builder")
    viewer.add_image(frame, name="video_frame", rgb=True)

    # 5. Initialize state (with optional initial data)
    state = TrackBuilderState()
    if initial_nodes is not None:
        for i, pos in enumerate(initial_nodes):
            label = None
            if initial_node_labels is not None and i < len(initial_node_labels):
                label = initial_node_labels[i]
            state.add_node(float(pos[0]), float(pos[1]), label=label)

    if initial_edges is not None:
        for node1, node2 in initial_edges:
            state.add_edge(node1, node2)

    # 6. Add track graph layers (edges, then nodes)
    edges_layer, nodes_layer = setup_track_layers(viewer)

    # 7. Add control widget
    # TrackGraphWidget is a wrapper class; pass the internal QWidget to napari
    widget = create_track_widget(viewer, edges_layer, nodes_layer, state)
    viewer.window.add_dock_widget(
        widget._widget,
        name="Track Graph Builder",
        area="right",
    )

    # 8. Sync napari layer changes back to state
    # When using napari's default "2" add mode, points are added directly to layer.data
    # We need to sync these back to our state so they persist.

    def _sync_layer_to_state(event=None):
        """Sync napari layer data back to TrackBuilderState."""
        # Get current layer data
        layer_data = nodes_layer.data

        # Convert napari (row, col) back to state (x, y)
        # napari stores as (y, x), we need (x, y)
        new_nodes = [(float(col), float(row)) for row, col in layer_data]
        n_layer = len(new_nodes)
        n_state = len(state.nodes)

        if n_layer > n_state:
            # New nodes added - add them to state
            for i in range(n_state, n_layer):
                x, y = new_nodes[i]
                state.add_node(x, y)
            widget.sync_from_state()
        elif n_layer < n_state:
            # Nodes deleted - rebuild state from layer
            # Clear state and re-add all nodes from layer
            state.nodes.clear()
            state.node_labels.clear()
            state.edges.clear()  # Edges reference deleted nodes
            state.start_node = None
            state.edge_start_node = None
            for x, y in new_nodes:
                state.add_node(x, y)
            widget.sync_from_state()
        elif n_layer == n_state and n_layer > 0:
            # Same count - check if positions changed (node moved)
            positions_changed = False
            for i, (x, y) in enumerate(new_nodes):
                old_x, old_y = state.nodes[i]
                if abs(x - old_x) > 0.01 or abs(y - old_y) > 0.01:
                    state.nodes[i] = (x, y)
                    positions_changed = True
            if positions_changed:
                # Don't call widget.sync_from_state() to avoid feedback loop
                # Just update edges layer to reflect new positions
                _sync_layers_from_state(state, nodes_layer, edges_layer)

    # Connect to layer data change events
    nodes_layer.events.data.connect(_sync_layer_to_state)

    # Sync napari layer mode changes back to state
    def _sync_mode_to_state(event=None):
        """Sync napari layer mode changes to TrackBuilderState."""
        layer_mode = nodes_layer.mode
        # Map napari modes to our state modes
        mode_map = {
            "add": "add_node",
            "select": "delete",  # Select mode is used for delete
            "pan_zoom": state.mode,  # Keep current mode when pan/zoom
        }
        new_mode = mode_map.get(layer_mode, state.mode)
        if new_mode != state.mode:
            state.mode = new_mode
            widget.sync_from_state()

    nodes_layer.events.mode.connect(_sync_mode_to_state)

    # 9. Setup keyboard shortcuts (only for things napari doesn't provide)
    # Work WITH napari's defaults:
    #   - 2 = Add mode (napari default for adding points)
    #   - 3 = Select mode (napari default for selecting/deleting)
    #   - 4 = Pan/Zoom mode (napari default)
    # Only add custom keybindings for functionality napari doesn't provide:
    #   - E = Edge mode (click two nodes to connect)
    #   - Ctrl+Z / Ctrl+Shift+Z = Undo/Redo
    #   - Shift+S = Set start node
    #   - Escape = Cancel edge creation

    # Mouse click handler for edge mode
    @nodes_layer.mouse_drag_callbacks.append
    def on_click(layer, event):
        """Handle mouse clicks for edge creation mode."""
        # Only handle single clicks (not drags) in add_edge mode
        if state.mode != "add_edge":
            return
        # Only respond to click (not drag)
        if event.type != "mouse_press":
            return

        # Get click position in data coordinates
        position = event.position
        _handle_click(state, nodes_layer, edges_layer, position)
        widget.sync_from_state()

        # Update status based on edge creation progress
        if state.edge_start_node is not None:
            viewer.status = f"Edge start: Node {state.edge_start_node} - click another node to connect"
        else:
            viewer.status = "Mode: Edge - click two nodes to connect (Escape to cancel)"

    @nodes_layer.bind_key("e", overwrite=True)
    def set_add_edge_mode(layer):
        """Switch to edge mode - click two nodes to connect them."""
        _handle_key(state, "E")
        # Switch napari layer to pan_zoom mode so clicking doesn't add points
        nodes_layer.mode = "pan_zoom"
        widget.sync_from_state()
        viewer.status = "Mode: Edge - click a node to start, then another to connect"

    @viewer.bind_key("Control-z")
    def undo_action(viewer):
        """Undo last action."""
        result = _handle_key(state, "Z", modifiers=["Control"])
        if result == "undo":
            _sync_layers_from_state(state, nodes_layer, edges_layer)
            widget.sync_from_state()
            viewer.status = "Undo"

    @viewer.bind_key("Control-Shift-z")
    def redo_action(viewer):
        """Redo last undone action."""
        result = _handle_key(state, "Z", modifiers=["Control", "Shift"])
        if result == "redo":
            _sync_layers_from_state(state, nodes_layer, edges_layer)
            widget.sync_from_state()
            viewer.status = "Redo"

    @viewer.bind_key("Shift-s")
    def set_start_node(viewer):
        """Set selected node as start."""
        # Get selected node from layer
        if len(nodes_layer.selected_data) == 1:
            selected_idx = next(iter(nodes_layer.selected_data))
            result = _handle_key(
                state,
                "S",
                modifiers=["Shift"],
                selected_node=selected_idx,
            )
            if result == "set_start":
                _sync_layers_from_state(state, nodes_layer, edges_layer)
                widget.sync_from_state()
                viewer.status = f"Node {selected_idx} set as start"
        else:
            viewer.status = "Select exactly one node first (press 3), then Shift+S"

    @viewer.bind_key("Escape")
    def cancel_edge(viewer):
        """Cancel edge creation in progress."""
        result = _handle_key(state, "Escape")
        if result == "cancel":
            _clear_edge_preview(edges_layer)
            _sync_layers_from_state(state, nodes_layer, edges_layer)
            viewer.status = "Edge creation cancelled"

    # 10. Start in add mode (need to add nodes before connecting them)
    state.mode = "add_node"
    nodes_layer.mode = "add"
    widget.sync_from_state()
    viewer.status = "Mode: Add - click to place nodes (press E to connect them)"

    # 11. Run napari (blocking until viewer closes)
    napari.run()

    # 12. Extract and transform results
    return build_track_graph_result(state, calibration)
