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
>>> from neurospatial.transforms import VideoCalibration
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
    create_track_widget,
    setup_track_layers,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from neurospatial.transforms import VideoCalibration

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
    >>> from neurospatial.transforms import VideoCalibration
    >>> result = annotate_track_graph("maze.mp4", calibration=calib)  # doctest: +SKIP
    >>> # node_positions now in cm

    Notes
    -----
    This function blocks until the napari viewer is closed. The viewer runs
    in the same Python process, and the function returns only after the user
    closes it (via the "Save and Close" button or window close).

    Keyboard Shortcuts
    ^^^^^^^^^^^^^^^^^^
    - A: Switch to add_node mode
    - E: Switch to add_edge mode
    - X: Switch to delete mode
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
            "Provide a video path to extract a frame, or pass an image array directly."
        )

    # 2. Import napari (lazy to allow module to work without it)
    try:
        import napari
    except ImportError as e:
        raise ImportError(
            "napari is required for interactive annotation. "
            "Install with: pip install napari[all]"
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
                f"Video has {reader.n_frames} frames (indices 0-{reader.n_frames - 1})."
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
    widget = create_track_widget(viewer, edges_layer, nodes_layer, state)
    viewer.window.add_dock_widget(widget, name="Track Graph Builder", area="right")

    # 8. Set initial status bar with shortcut reminder
    viewer.status = "Track Graph Mode: A (add) | E (edge) | X (delete) | Ctrl+Z (undo)"

    # 9. Run napari (blocking until viewer closes)
    napari.run()

    # 10. Extract and transform results
    return build_track_graph_result(state, calibration)
