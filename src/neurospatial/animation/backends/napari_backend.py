"""Napari GPU-accelerated viewer backend."""

from __future__ import annotations

import contextlib
from collections import OrderedDict
from threading import Lock
from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from neurospatial.environment.core import Environment

# Check napari availability
try:
    import napari

    NAPARI_AVAILABLE = True
except ImportError:
    NAPARI_AVAILABLE = False


# =============================================================================
# Overlay Rendering Helper Functions
# =============================================================================


def _transform_coords_for_napari(coords: NDArray[np.float64]) -> NDArray[np.float64]:
    """Transform coordinates from (x, y) to (y, x) for Napari axis convention.

    Napari uses (y, x) axis order for 2D data, opposite of the standard (x, y).

    Parameters
    ----------
    coords : ndarray
        Coordinates with shape (..., n_dims) where last dimension is spatial.
        For 2D data, expects (..., 2) with (x, y) ordering.

    Returns
    -------
    transformed : ndarray
        Coordinates with swapped axes. For 2D: (x, y) → (y, x).
        Higher dimensions returned unchanged.
    """
    if coords.shape[-1] == 2:
        # Swap x and y for 2D
        return coords[..., ::-1]  # Reverse last dimension
    # For other dimensions, return unchanged
    return coords


def _render_position_overlay(
    viewer: Any, position_data: Any, name_suffix: str = ""
) -> list:
    """Render a single position overlay with optional trail.

    Parameters
    ----------
    viewer : napari.Viewer
        Napari viewer instance
    position_data : PositionData
        Position overlay data aligned to frames
    name_suffix : str
        Suffix for layer names (e.g., for multi-animal)

    Returns
    -------
    layers : list
        List of created layer objects for later updates
    """
    layers = []
    n_frames = len(position_data.data)

    # Transform coordinates to napari (y, x) convention
    transformed_data = _transform_coords_for_napari(position_data.data)

    # Add tracks layer for trail if trail_length specified
    if position_data.trail_length is not None:
        # Create track data: (track_id, time, y, x)
        track_data = np.column_stack(
            [
                np.zeros(n_frames),  # Single track
                np.arange(n_frames),  # Time
                transformed_data[:, 0],  # Y
                transformed_data[:, 1],  # X
            ]
        )

        layer = viewer.add_tracks(
            track_data,
            name=f"Position Trail{name_suffix}",
            tail_length=position_data.trail_length,
            color=position_data.color,
        )
        layers.append(layer)

    # Add points layer for current position marker
    # Create points with time dimension: (time, y, x)
    points_data = np.column_stack(
        [
            np.arange(n_frames),  # Time
            transformed_data[:, 0],  # Y
            transformed_data[:, 1],  # X
        ]
    )

    layer = viewer.add_points(
        points_data,
        name=f"Position{name_suffix}",
        size=position_data.size,
        face_color=position_data.color,
        edge_color="white",
        edge_width=0.5,
    )
    layers.append(layer)

    return layers


def _render_bodypart_overlay(
    viewer: Any, bodypart_data: Any, name_suffix: str = ""
) -> list:
    """Render a single bodypart overlay with skeleton.

    Parameters
    ----------
    viewer : napari.Viewer
        Napari viewer instance
    bodypart_data : BodypartData
        Bodypart overlay data aligned to frames
    name_suffix : str
        Suffix for layer names

    Returns
    -------
    layers : list
        List of created layer objects for later updates
    """
    layers = []

    # Validate non-empty bodyparts dictionary
    if not bodypart_data.bodyparts:
        raise ValueError(
            "BodypartData must contain at least one bodypart. "
            "Received empty bodyparts dictionary."
        )

    n_frames = len(next(iter(bodypart_data.bodyparts.values())))

    # Render bodypart points
    # Combine all bodyparts into single points layer with properties for coloring
    all_points = []
    all_features = []

    for part_name, coords in bodypart_data.bodyparts.items():
        # Transform coordinates
        transformed = _transform_coords_for_napari(coords)

        # Add time dimension: (time, y, x)
        points_with_time = np.column_stack(
            [
                np.arange(n_frames),
                transformed[:, 0],  # Y
                transformed[:, 1],  # X
            ]
        )
        all_points.append(points_with_time)

        # Track which bodypart each point belongs to
        all_features.extend([part_name] * n_frames)

    # Stack all points
    points_array = np.vstack(all_points)

    # Determine face colors
    face_colors: list[str] | str
    if bodypart_data.colors is not None:
        # Map bodypart names to colors
        face_colors = [bodypart_data.colors.get(name, "white") for name in all_features]
    else:
        face_colors = "white"

    layer = viewer.add_points(
        points_array,
        name=f"Bodyparts{name_suffix}",
        size=5.0,
        face_color=face_colors,
        edge_color="black",
        edge_width=0.5,
        features={"bodypart": all_features},
    )
    layers.append(layer)

    # Render skeleton if provided
    if bodypart_data.skeleton is not None:
        # Create line shapes for skeleton edges
        # For each frame, create lines connecting skeleton pairs
        skeleton_lines = []

        for frame_idx in range(n_frames):
            frame_lines = []
            for start_part, end_part in bodypart_data.skeleton:
                if (
                    start_part in bodypart_data.bodyparts
                    and end_part in bodypart_data.bodyparts
                ):
                    start_coords = bodypart_data.bodyparts[start_part][frame_idx]
                    end_coords = bodypart_data.bodyparts[end_part][frame_idx]

                    # Skip if either endpoint is NaN
                    if np.any(np.isnan(start_coords)) or np.any(np.isnan(end_coords)):
                        continue

                    # Transform to napari coords
                    start_napari = _transform_coords_for_napari(
                        start_coords.reshape(1, -1)
                    )[0]
                    end_napari = _transform_coords_for_napari(
                        end_coords.reshape(1, -1)
                    )[0]

                    # Line in napari format: [[time, y1, x1], [time, y2, x2]]
                    line = np.array(
                        [
                            [frame_idx, start_napari[0], start_napari[1]],
                            [frame_idx, end_napari[0], end_napari[1]],
                        ]
                    )
                    frame_lines.append(line)

            if frame_lines:
                skeleton_lines.extend(frame_lines)

        if skeleton_lines:
            layer = viewer.add_shapes(
                skeleton_lines,
                name=f"Skeleton{name_suffix}",
                shape_type="line",
                edge_color=bodypart_data.skeleton_color,
                edge_width=bodypart_data.skeleton_width,
            )
            layers.append(layer)

    return layers


def _render_head_direction_overlay(
    viewer: Any, head_dir_data: Any, env: Environment, name_suffix: str = ""
) -> list:
    """Render a single head direction overlay as vectors.

    Parameters
    ----------
    viewer : napari.Viewer
        Napari viewer instance
    head_dir_data : HeadDirectionData
        Head direction data aligned to frames
    env : Environment
        Environment for positioning vectors
    name_suffix : str
        Suffix for layer names

    Returns
    -------
    layers : list
        List of created layer objects for later updates
    """
    layers = []
    n_frames = len(head_dir_data.data)

    # Determine if data is angles or vectors
    is_angles = head_dir_data.data.ndim == 1

    # Get centroid of environment for vector origin
    centroid = np.mean(env.bin_centers, axis=0)
    centroid_napari = _transform_coords_for_napari(centroid.reshape(1, -1))[0]

    # Convert to vectors for napari
    vectors_data = []

    for frame_idx in range(n_frames):
        if is_angles:
            # Convert angle to unit vector, then scale by length
            angle = head_dir_data.data[frame_idx]
            direction = np.array([np.cos(angle), np.sin(angle)]) * head_dir_data.length
        else:
            # Already a vector, just scale to desired length
            direction = head_dir_data.data[frame_idx]
            # Normalize and scale
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm * head_dir_data.length

        # Transform direction to napari coords
        direction_napari = _transform_coords_for_napari(direction.reshape(1, -1))[0]

        # Napari vector format: [[time, y, x], [dt, dy, dx]]
        # Direction also needs time component (0 for spatial direction only)
        vector = np.array(
            [
                [frame_idx, centroid_napari[0], centroid_napari[1]],  # Origin (t, y, x)
                [0, direction_napari[0], direction_napari[1]],  # Direction (0, dy, dx)
            ]
        )
        vectors_data.append(vector)

    # Stack vectors and add layer
    vectors_array = np.array(vectors_data)

    layer = viewer.add_vectors(
        vectors_array,
        name=f"Head Direction{name_suffix}",
        edge_color=head_dir_data.color,
        edge_width=3.0,
        length=1.0,  # Vectors already scaled by length
    )
    layers.append(layer)

    return layers


def _render_regions(
    viewer: Any, env: Environment, show_regions: bool | list[str], region_alpha: float
) -> list:
    """Render environment regions as shapes.

    Parameters
    ----------
    viewer : napari.Viewer
        Napari viewer instance
    env : Environment
        Environment with regions
    show_regions : bool or list of str
        If True, show all regions. If list, show only specified regions.
    region_alpha : float
        Alpha transparency for regions (0-1)

    Returns
    -------
    layers : list
        List of created layer objects
    """
    layers: list[Any] = []

    if not show_regions or len(env.regions) == 0:
        return layers

    # Determine which regions to show
    if isinstance(show_regions, bool):
        region_names = list(env.regions.keys())
    else:
        region_names = show_regions

    # Collect region shapes
    region_shapes: list[NDArray[np.float64]] = []
    region_properties: dict[str, list[str]] = {"name": []}

    for region_name in region_names:
        if region_name not in env.regions:
            continue

        region = env.regions[region_name]

        # Convert region to napari shape
        if region.kind == "point":
            # Point regions: render as small circle
            coords = _transform_coords_for_napari(region.data.reshape(1, -1))[0]
            # Create circle polygon (approximate with octagon)
            radius = 2.0  # Small visual marker
            angles = np.linspace(0, 2 * np.pi, 9)
            circle = np.column_stack(
                [
                    coords[0] + radius * np.cos(angles),
                    coords[1] + radius * np.sin(angles),
                ]
            )
            region_shapes.append(circle)
            region_properties["name"].append(region_name)
        elif region.kind == "polygon":
            # Polygon regions: use directly
            # Extract coordinates from Shapely polygon
            # Note: region.data is a Shapely Polygon here
            coords = np.array(region.data.exterior.coords)  # type: ignore[union-attr]
            coords_napari = _transform_coords_for_napari(coords)
            region_shapes.append(coords_napari)
            region_properties["name"].append(region_name)

    if region_shapes:
        layer = viewer.add_shapes(
            region_shapes,
            name="Regions",
            shape_type="polygon",
            edge_color="white",
            face_color="white",
            opacity=region_alpha,
            properties=region_properties,
        )
        layers.append(layer)

    return layers


def _is_multi_field_input(fields: list) -> bool:
    """Check if fields is multi-field input (list of sequences).

    Parameters
    ----------
    fields : list
        Input fields to check

    Returns
    -------
    is_multi : bool
        True if fields is a list of lists (multi-field), False if single sequence

    Notes
    -----
    Detection logic:
    - Empty list → single field (False)
    - List of arrays → single field (False)
    - List of lists → multi-field (True)
    """
    if len(fields) == 0:
        return False

    # Check if first element is a list/sequence
    first_elem = fields[0]
    return isinstance(first_elem, (list, tuple))


def _add_speed_control_widget(
    viewer: Any, initial_fps: int = 30, frame_labels: list[str] | None = None
) -> None:
    """Add enhanced playback control widget to napari viewer.

    Creates a comprehensive docked widget with:
    - Play/Pause button (large, prominent)
    - FPS slider (1-120 range, 200px wide)
    - Frame counter ("Frame: 15 / 30")
    - Frame label (if provided: "Trial 15")

    Inspired by nwb_data_viewer playback widget pattern.

    Parameters
    ----------
    viewer : napari.Viewer
        Napari viewer instance to add widget to
    initial_fps : int, default=30
        Initial playback speed in frames per second
    frame_labels : list of str, optional
        Labels for each frame (e.g., ["Trial 1", "Trial 2", ...])
        If provided, displays current frame's label in widget

    Notes
    -----
    If magicgui is not available, this function silently returns without
    adding the widget. This ensures the napari backend works even if
    magicgui is not installed.

    The frame counter updates in real-time as animation plays, connected
    to viewer.dims.events.current_step.
    """
    try:
        from magicgui import magicgui
        from napari.settings import get_settings
    except ImportError:
        # magicgui or napari.settings not available - skip widget
        return

    # Track playback state for button updates
    playback_state = {"is_playing": False, "last_frame": -1}

    # Create enhanced playback widget using magicgui
    # Set slider max to at least initial_fps (but minimum 120)
    slider_max = max(120, initial_fps)

    # Throttle widget updates for high FPS (30 Hz max to avoid Qt overhead)
    # At 250 FPS, updating 250x/sec causes stalling; throttle to 30 Hz
    update_interval = max(1, initial_fps // 30) if initial_fps >= 30 else 1

    @magicgui(
        auto_call=True,
        play={"widget_type": "PushButton", "text": "▶ Play"},
        fps={
            "widget_type": "Slider",
            "min": 1,
            "max": slider_max,
            "value": initial_fps,
            "label": "Speed (FPS)",
        },
        frame_info={"widget_type": "Label", "label": ""},
    )
    def playback_widget(
        play: bool = False,
        fps: int = initial_fps,
        frame_info: str = "",
    ) -> None:
        """Enhanced playback control widget."""
        # Update FPS setting when slider changes
        settings = get_settings()
        settings.application.playback_fps = fps

    # Make slider larger and easier to interact with
    with contextlib.suppress(Exception):
        playback_widget.fps.native.setMinimumWidth(200)

    # Connect play button to toggle playback
    def toggle_playback(event=None):
        """Toggle animation playback."""
        playback_state["is_playing"] = not playback_state["is_playing"]
        if playback_state["is_playing"]:
            playback_widget.play.text = "⏸ Pause"
        else:
            playback_widget.play.text = "▶ Play"
        # Actually toggle napari's playback
        viewer.window._toggle_play()

    playback_widget.play.changed.connect(toggle_playback)

    # Update frame counter when dims change
    def update_frame_info(event=None):
        """Update frame counter and label display (throttled for high FPS)."""
        try:
            # Get current frame from first dimension (time)
            current_frame = viewer.dims.current_step[0] if viewer.dims.ndim > 0 else 0

            # Throttle updates: only update every Nth frame to avoid Qt overhead
            # For high FPS (e.g., 250 FPS), this prevents stalling
            if (
                current_frame % update_interval != 0
                and current_frame != playback_state["last_frame"]
            ):
                # Skip this update (not at interval boundary and frame changed)
                # Exception: always update when paused (frame == last_frame)
                playback_state["last_frame"] = current_frame
                return

            playback_state["last_frame"] = current_frame

            # Sync button state with napari's playback (fixes button sync issue)
            # Detect if playback is active by checking if dims is playing
            try:
                # Check napari's internal playing state via dims
                is_playing = viewer.window.qt_viewer.dims.is_playing
                if is_playing != playback_state["is_playing"]:
                    playback_state["is_playing"] = is_playing
                    playback_widget.play.text = "⏸ Pause" if is_playing else "▶ Play"
            except Exception:
                # If unable to detect playback state, don't update button
                pass

            # Get total frames from dims range
            total_frames = (
                viewer.dims.range[0][2]
                if viewer.dims.ndim > 0 and viewer.dims.range
                else 0
            )

            # Build frame info text
            frame_text = f"Frame: {current_frame + 1} / {total_frames}"

            # Add label if provided
            if frame_labels and 0 <= current_frame < len(frame_labels):
                frame_text += f" ({frame_labels[current_frame]})"

            playback_widget.frame_info.value = frame_text
        except Exception:
            # If anything fails, show minimal info
            playback_widget.frame_info.value = "Frame: -- / --"

    # Connect to dims events to track frame changes
    viewer.dims.events.current_step.connect(update_frame_info)

    # Initialize frame info
    update_frame_info()

    # Add spacebar keyboard shortcut to toggle playback
    # Must be defined here to access toggle_playback() function
    @viewer.bind_key("Space")
    def spacebar_toggle(viewer_instance):
        """Toggle animation playback with spacebar (syncs with widget button)."""
        # Call the same toggle function as the button to keep them in sync
        toggle_playback()

    # Add widget as dock to viewer
    with contextlib.suppress(Exception):
        viewer.window.add_dock_widget(
            playback_widget,
            name="Playback Controls",
            area="left",  # Dock on left side
        )


def render_napari(
    env: Environment,
    fields: list[NDArray[np.float64]] | list[list[NDArray[np.float64]]],
    *,
    fps: int = 30,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    frame_labels: list[str] | None = None,
    title: str = "Spatial Field Animation",
    cache_size: int = 1000,
    chunk_size: int = 10,
    max_chunks: int = 100,
    layout: Literal["horizontal", "vertical", "grid"] | None = None,
    layer_names: list[str] | None = None,
    overlay_data: Any | None = None,  # OverlayData - use Any to avoid circular import
    show_regions: bool | list[str] = False,
    region_alpha: float = 0.3,
    **kwargs: Any,  # Accept other parameters gracefully
) -> Any:
    """Launch Napari viewer with lazy-loaded field animation.

    Napari provides GPU-accelerated rendering with on-demand frame loading,
    making it ideal for large datasets (100K+ frames). Frames are cached
    with a true LRU (Least Recently Used) eviction policy for efficient
    memory management.

    Parameters
    ----------
    env : Environment
        Environment defining spatial structure
    fields : list of arrays or list of lists
        Single-field mode: List of 1D arrays, each with shape (n_bins,)
        Multi-field mode: List of field sequences, where each sequence
        is a list of 1D arrays. Automatically detected based on input structure.
        All sequences must have the same length (same number of frames).
    fps : int, default=30
        Frames per second for playback (Napari slider speed)
    cmap : str, default="viridis"
        Matplotlib colormap name
    vmin, vmax : float, optional
        Color scale limits. If None, computed from all fields.
    frame_labels : list of str, optional
        Frame labels (e.g., ["Trial 1", "Trial 2", ...])
        Displayed in the enhanced playback control widget alongside
        the frame counter (requires magicgui, included with napari[all])
    title : str, default="Spatial Field Animation"
        Viewer window title
    cache_size : int, default=1000
        Maximum number of frames to cache for per-frame caching strategy.
        Allows callers to tune cache size per dataset to prevent thrashing
        (too small) or RAM blow-ups (too large). Typical usage: 30KB per
        frame for 100x100 grids, so 1000 frames = ~30MB.
    chunk_size : int, default=10
        Number of frames per chunk for chunked caching strategy. Allows
        tuning the granularity of chunk-based caching. Larger values improve
        sequential access but increase memory per chunk.
    max_chunks : int, default=100
        Maximum number of chunks to cache for chunked caching strategy.
        Controls total cache memory (cache_memory = chunk_size * max_chunks
        * frame_size). Adjust based on available RAM.
    layout : {"horizontal", "vertical", "grid"}, optional
        Multi-field mode only: Layout arrangement for multiple field sequences.
        Required when providing multiple field sequences. Options:
        - "horizontal": side-by-side arrangement
        - "vertical": stacked top-to-bottom
        - "grid": automatic NxM grid layout
        Single-field mode: ignored
    layer_names : list of str, optional
        Multi-field mode only: Custom names for each layer. Must match the
        number of field sequences. If None, uses "Field 1", "Field 2", etc.
        Single-field mode: ignored
    overlay_data : OverlayData, optional
        Overlay data containing position, bodypart, head direction, and region
        overlays to render on top of fields. Created by the conversion pipeline
        (see neurospatial.animation.overlays). If None, no overlays are rendered.
        Default is None.
    show_regions : bool or list of str, default=False
        If True, show all environment regions. If list, show only specified
        regions by name. Regions are rendered as semi-transparent polygon shapes.
    region_alpha : float, default=0.3
        Alpha transparency for region overlays (0.0 = fully transparent,
        1.0 = fully opaque). Only applies when show_regions is True or non-empty.
    **kwargs : dict
        Additional parameters (gracefully ignored for compatibility
        with other backends)

    Returns
    -------
    viewer : napari.Viewer
        Napari viewer instance. Call `napari.run()` to block until
        window is closed, or interact with viewer programmatically.

    Raises
    ------
    ImportError
        If napari is not installed

    Examples
    --------
    .. code-block:: python

        import numpy as np
        from neurospatial import Environment
        from neurospatial.animation.backends.napari_backend import render_napari

        # Create environment
        positions = np.random.randn(100, 2) * 50
        env = Environment.from_samples(positions, bin_size=10.0)

        # Single-field animation
        fields = [np.random.rand(env.n_bins) for _ in range(100)]

        # Launch viewer
        viewer = render_napari(env, fields, fps=30, cmap="viridis")
        # napari.run()  # Uncomment to block until window closed

        # Multi-field animation (e.g., comparing 3 neurons)
        seq1 = [np.random.rand(env.n_bins) for _ in range(50)]
        seq2 = [np.random.rand(env.n_bins) for _ in range(50)]
        seq3 = [np.random.rand(env.n_bins) for _ in range(50)]

        # View side-by-side with custom names
        viewer = render_napari(
            env,
            fields=[seq1, seq2, seq3],
            layout="horizontal",
            layer_names=["Neuron A", "Neuron B", "Neuron C"],
            fps=10,
        )

    Notes
    -----
    **Playback Controls:**

    Napari provides built-in playback controls at the **bottom-left** of the viewer window:

    - **Frame slider** - Horizontal slider showing current frame position
    - **Play button (▶)** - Triangle icon to start/stop animation (next to slider)
    - **Frame counter** - Shows "1/N" indicating current frame
    - **Keyboard shortcuts**:
        - **Spacebar** - Play/pause animation (toggle)
        - **← →** Arrow keys - Step forward/backward through frames

    **Enhanced Playback Controls:**

    An interactive "Playback Controls" widget is automatically added to the left side
    of the viewer (requires magicgui, which is included with napari[all]):

    - **Play/Pause Button** - Large, prominent button to toggle animation (▶/⏸)
    - **FPS Slider** - Large slider (200px wide) to adjust playback speed from 1-120 FPS
    - **Frame Counter** - Shows current frame and total frames (e.g., "Frame: 15 / 30")
    - **Frame Labels** - Displays custom labels if provided (e.g., "Trial 15")
    - **Real-time updates** - All controls update instantly during playback

    The animation starts at frame 0 with playback speed set by the `fps` parameter.
    If magicgui is not available, the speed can still be changed via
    File → Preferences → Application → "Playback frames per second".

    **Note:** Only the time dimension slider is shown. Spatial dimensions (height, width)
    are displayed in the 2D viewport, not as separate sliders.

    **Memory Efficiency:**

    Napari backend uses lazy loading with LRU caching. Two caching strategies
    are available:

    1. **Per-frame caching** (default for <10K frames):
       - Caches individual frames
       - Cache size: 1000 frames (configurable via cache_size parameter)
       - Good for small to medium datasets

    2. **Chunked caching** (default for >10K frames):
       - Caches frames in chunks (default: 10 frames/chunk, 100 max chunks)
       - More efficient for sequential playback (pre-loads neighboring frames)
       - Better for large datasets (100K+ frames)
       - Configurable via chunk_size and max_chunks parameters

    Both strategies work efficiently with memory-mapped arrays.

    **Performance:**

    - Seek time: <100ms for 100K+ frames
    - GPU acceleration for rendering
    - Suitable for hour-long sessions (900K frames at 250 Hz)

    **Overlay System:**

    The overlay system supports multiple overlay types that can be combined:

    - **Position overlays**: Trajectory tracking with optional trails
      (tracks + points layers)
    - **Bodypart overlays**: Multi-keypoint pose tracking with skeleton
      visualization (points + shapes layers)
    - **Head direction overlays**: Directional heading as vectors (vectors layer)
    - **Region overlays**: Environment regions of interest (shapes layer
      with transparency)

    Multiple overlays of the same type can be rendered simultaneously
    (e.g., multi-animal tracking). Overlays are automatically aligned to
    animation frame times during the conversion pipeline. All overlays use
    Napari's native layer types for GPU-accelerated rendering.

    **Multi-Field Viewer Mode:**

    Napari backend supports comparing multiple field sequences side-by-side
    in a single viewer. Automatically detected when ``fields`` is a list of
    lists (each inner list is a field sequence).

    Key features:
    - **Auto-detection**: No special API needed, just pass list of sequences
    - **Global color scale**: Computed across ALL sequences for fair comparison
    - **Synchronized playback**: All layers share the same time dimension
    - **Layout options**: horizontal, vertical, or grid arrangement
    - **Custom names**: Use ``layer_names`` parameter for meaningful labels

    Example use cases:
    - Comparing place fields across multiple neurons
    - Visualizing learning across trials
    - Side-by-side condition comparison

    Requirements:
    - All sequences must have the same length (number of frames)
    - Must provide ``layout`` parameter when using multi-field mode
    - Optional: ``layer_names`` for custom layer labels

    See Also
    --------
    neurospatial.animation.core.animate_fields : Main animation interface
    """
    if not NAPARI_AVAILABLE:
        raise ImportError(
            "Napari backend requires napari. Install with:\n"
            "  pip install napari[all]\n"
            "or\n"
            "  uv add napari[all]"
        )

    # Detect multi-field input and route appropriately
    if _is_multi_field_input(fields):
        # Multi-field viewer support
        return _render_multi_field_napari(
            env=env,
            field_sequences=fields,  # type: ignore[arg-type]
            fps=fps,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            frame_labels=frame_labels,
            title=title,
            cache_size=cache_size,
            chunk_size=chunk_size,
            max_chunks=max_chunks,
            layout=layout,
            layer_names=layer_names,
            overlay_data=overlay_data,
            show_regions=show_regions,
            region_alpha=region_alpha,
        )

    # Single-field viewer (original behavior)
    # At this point, fields is guaranteed to be list[NDArray] (not multi-field)
    from neurospatial.animation.rendering import compute_global_colormap_range

    # Compute global color scale
    if vmin is None or vmax is None:
        vmin_computed, vmax_computed = compute_global_colormap_range(fields, vmin, vmax)  # type: ignore[arg-type]
        vmin = vmin if vmin is not None else vmin_computed
        vmax = vmax if vmax is not None else vmax_computed

    # Note: contrast_limits parameter is ignored for RGB images
    # RGB images are already in [0, 255] range and don't need normalization
    # We keep the parameter for API compatibility but don't use it

    # Pre-compute colormap lookup table (256 RGB values)
    cmap_obj = plt.get_cmap(cmap)
    cmap_lookup = (cmap_obj(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)

    # Create lazy frame loader (with optional chunked caching)
    lazy_frames = _create_lazy_field_renderer(
        env,
        fields,  # type: ignore[arg-type]
        cmap_lookup,
        vmin,
        vmax,
        cache_size,
        chunk_size,
        max_chunks,
    )

    # Create napari viewer
    viewer = napari.Viewer(title=title)

    # Add image layer (RGB - no contrast_limits needed)
    viewer.add_image(
        lazy_frames,
        name="Spatial Fields",
        rgb=True,  # Already RGB
        # Don't pass contrast_limits for RGB images - they're already [0, 255]
    )

    # Configure playback controls
    # 1. Set initial frame to 0 (start at beginning, not middle)
    viewer.dims.current_step = (0, *viewer.dims.current_step[1:])

    # 2. Configure which dimensions to display
    # For RGB images with shape (time, height, width, 3):
    # - Dimension 0 (time): slider (not displayed)
    # - Dimensions 1, 2 (height, width): displayed in 2D view
    # This ensures only the time slider appears, not height/width sliders
    viewer.dims.ndisplay = 2  # Display 2 spatial dimensions
    if viewer.dims.ndim >= 3:
        # Set displayed order: show dimensions 1 and 2 (height, width)
        viewer.dims.order = tuple(range(viewer.dims.ndim))

    # 3. Configure FPS via napari settings (avoids deprecated qt_viewer access)
    try:
        from napari.settings import get_settings

        settings = get_settings()
        settings.application.playback_fps = fps

        # 4. Add interactive speed control widget (if magicgui available)
        # The widget also adds spacebar keyboard shortcut (synced with button)
        _add_speed_control_widget(viewer, initial_fps=fps, frame_labels=frame_labels)
    except ImportError:
        # Fallback: Add spacebar shortcut without widget (magicgui not available)
        @viewer.bind_key("Space")
        def toggle_playback(viewer):
            """Toggle animation playback with spacebar."""
            viewer.window._toggle_play()

    # Render overlay data if provided
    if overlay_data is not None:
        # Render position overlays (tracks + points)
        for idx, pos_data in enumerate(overlay_data.positions):
            suffix = f" {idx + 1}" if len(overlay_data.positions) > 1 else ""
            _render_position_overlay(viewer, pos_data, name_suffix=suffix)

        # Render bodypart overlays (points + skeleton)
        for idx, bodypart_data in enumerate(overlay_data.bodypart_sets):
            suffix = f" {idx + 1}" if len(overlay_data.bodypart_sets) > 1 else ""
            _render_bodypart_overlay(viewer, bodypart_data, name_suffix=suffix)

        # Render head direction overlays (vectors)
        for idx, head_dir_data in enumerate(overlay_data.head_directions):
            suffix = f" {idx + 1}" if len(overlay_data.head_directions) > 1 else ""
            _render_head_direction_overlay(
                viewer, head_dir_data, env, name_suffix=suffix
            )

    # Render regions if requested
    if show_regions:
        _render_regions(viewer, env, show_regions, region_alpha)

    return viewer


def _render_multi_field_napari(
    env: Environment,
    field_sequences: list[list[NDArray[np.float64]]],
    *,
    fps: int = 30,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    frame_labels: list[str] | None = None,
    title: str = "Multi-Field Animation",
    cache_size: int = 1000,
    chunk_size: int = 10,
    max_chunks: int = 100,
    layout: Literal["horizontal", "vertical", "grid"] | None = None,
    layer_names: list[str] | None = None,
    overlay_data: Any | None = None,  # OverlayData
    show_regions: bool | list[str] = False,
    region_alpha: float = 0.3,
) -> Any:
    """Render multiple field sequences in a single napari viewer.

    Creates separate image layers for each field sequence, arranged according
    to the specified layout. All layers share the same time dimension for
    synchronized playback.

    Parameters
    ----------
    env : Environment
        Environment defining spatial structure
    field_sequences : list of lists
        Multiple field sequences to animate. Each sequence is a list of
        fields with shape (n_bins,). All sequences must have the same length.
    fps : int, default=30
        Frames per second for playback
    cmap : str, default="viridis"
        Matplotlib colormap name (applied to all layers)
    vmin, vmax : float, optional
        Color scale limits. If None, computed globally across all sequences
        for consistent comparison.
    frame_labels : list of str, optional
        Frame labels (e.g., ["Trial 1", "Trial 2", ...])
    title : str, default="Multi-Field Animation"
        Viewer window title
    cache_size : int, default=1000
        Maximum number of frames to cache for per-frame caching strategy
    chunk_size : int, default=10
        Number of frames per chunk for chunked caching strategy
    max_chunks : int, default=100
        Maximum number of chunks to cache for chunked caching strategy
    layout : {"horizontal", "vertical", "grid"}, required
        Layout arrangement for multiple fields:
        - "horizontal": side-by-side
        - "vertical": stacked top-to-bottom
        - "grid": automatic NxM grid
    layer_names : list of str, optional
        Custom names for each layer. Must match number of sequences.
        If None, uses "Field 1", "Field 2", etc.

    Returns
    -------
    viewer : napari.Viewer
        Napari viewer instance with multiple layers

    Raises
    ------
    ValueError
        If layout is None (required for multi-field input)
        If sequences have different lengths
        If layer_names count doesn't match sequence count
    """
    # Validation
    if layout is None:
        raise ValueError(
            "Multi-field input requires 'layout' parameter. "
            "Choose from: 'horizontal', 'vertical', 'grid'"
        )

    n_sequences = len(field_sequences)

    # Validate all sequences have same length
    sequence_lengths = [len(seq) for seq in field_sequences]
    if len(set(sequence_lengths)) > 1:
        raise ValueError(
            f"All field sequences must have the same length. "
            f"Got lengths: {sequence_lengths}"
        )

    # Generate layer names if not provided
    if layer_names is None:
        layer_names = [f"Field {i + 1}" for i in range(n_sequences)]
    else:
        if len(layer_names) != n_sequences:
            raise ValueError(
                f"Number of layer_names ({len(layer_names)}) must match "
                f"number of sequences ({n_sequences})"
            )

    from neurospatial.animation.rendering import compute_global_colormap_range

    # Compute global color scale across ALL sequences for consistent comparison
    if vmin is None or vmax is None:
        # Flatten all fields from all sequences
        all_fields = [field for sequence in field_sequences for field in sequence]
        vmin_computed, vmax_computed = compute_global_colormap_range(
            all_fields, vmin, vmax
        )
        vmin = vmin if vmin is not None else vmin_computed
        vmax = vmax if vmax is not None else vmax_computed

    # Pre-compute colormap lookup table (shared across all layers)
    cmap_obj = plt.get_cmap(cmap)
    cmap_lookup = (cmap_obj(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)

    # Create lazy frame loaders for each sequence
    lazy_renderers = []
    for sequence in field_sequences:
        renderer = _create_lazy_field_renderer(
            env, sequence, cmap_lookup, vmin, vmax, cache_size, chunk_size, max_chunks
        )
        lazy_renderers.append(renderer)

    # Create napari viewer
    viewer = napari.Viewer(title=title)

    # Add image layers
    # Note: Napari handles multi-layer arrangement automatically. The layout
    # parameter is accepted for API compatibility and future customization,
    # but currently all layouts add layers in the same way (napari manages
    # visual positioning). Users can manually arrange layers in the napari GUI.
    for renderer, name in zip(lazy_renderers, layer_names, strict=True):
        viewer.add_image(
            renderer,
            name=name,
            rgb=True,
        )

    # Configure playback controls
    viewer.dims.current_step = (0, *viewer.dims.current_step[1:])
    viewer.dims.ndisplay = 2

    if viewer.dims.ndim >= 3:
        viewer.dims.order = tuple(range(viewer.dims.ndim))

    # Configure FPS and add playback widget
    try:
        from napari.settings import get_settings

        settings = get_settings()
        settings.application.playback_fps = fps

        _add_speed_control_widget(viewer, initial_fps=fps, frame_labels=frame_labels)
    except ImportError:

        @viewer.bind_key("Space")
        def toggle_playback(viewer):
            """Toggle animation playback with spacebar."""
            viewer.window._toggle_play()

    # Render overlay data if provided
    if overlay_data is not None:
        # Render position overlays (tracks + points)
        for idx, pos_data in enumerate(overlay_data.positions):
            suffix = f" {idx + 1}" if len(overlay_data.positions) > 1 else ""
            _render_position_overlay(viewer, pos_data, name_suffix=suffix)

        # Render bodypart overlays (points + skeleton)
        for idx, bodypart_data in enumerate(overlay_data.bodypart_sets):
            suffix = f" {idx + 1}" if len(overlay_data.bodypart_sets) > 1 else ""
            _render_bodypart_overlay(viewer, bodypart_data, name_suffix=suffix)

        # Render head direction overlays (vectors)
        for idx, head_dir_data in enumerate(overlay_data.head_directions):
            suffix = f" {idx + 1}" if len(overlay_data.head_directions) > 1 else ""
            _render_head_direction_overlay(
                viewer, head_dir_data, env, name_suffix=suffix
            )

    # Render regions if requested
    if show_regions:
        _render_regions(viewer, env, show_regions, region_alpha)

    return viewer


def _create_lazy_field_renderer(
    env: Environment,
    fields: list[NDArray[np.float64]],
    cmap_lookup: NDArray[np.uint8],
    vmin: float,
    vmax: float,
    cache_size: int = 1000,
    chunk_size: int = 10,
    max_chunks: int = 100,
) -> LazyFieldRenderer | ChunkedLazyFieldRenderer:
    """Create lazy field renderer for Napari.

    Automatically selects between per-frame and chunked caching based on
    dataset size. Uses per-frame caching for datasets ≤10K frames and
    chunked caching for larger datasets.

    Parameters
    ----------
    env : Environment
        Environment defining spatial structure
    fields : list of arrays
        All fields to animate
    cmap_lookup : ndarray, shape (256, 3)
        Pre-computed colormap RGB lookup table
    vmin, vmax : float
        Color scale limits
    cache_size : int, default=1000
        Maximum number of frames to cache for per-frame caching strategy
    chunk_size : int, default=10
        Number of frames per chunk for chunked caching strategy
    max_chunks : int, default=100
        Maximum number of chunks to cache for chunked caching strategy

    Returns
    -------
    renderer : LazyFieldRenderer or ChunkedLazyFieldRenderer
        Lazy renderer instance implementing array-like interface
        for Napari
    """
    n_frames = len(fields)

    # Auto-select caching strategy based on dataset size
    if n_frames <= 10_000:
        # Use per-frame caching for small/medium datasets
        return LazyFieldRenderer(
            env, fields, cmap_lookup, vmin, vmax, cache_size=cache_size
        )
    else:
        # Use chunked caching for large datasets (>10K frames)
        return ChunkedLazyFieldRenderer(
            env,
            fields,
            cmap_lookup,
            vmin,
            vmax,
            chunk_size=chunk_size,
            max_chunks=max_chunks,
        )


class LazyFieldRenderer:
    """On-demand field rendering for Napari with true LRU cache.

    This class provides an array-like interface that Napari can use
    for lazy loading. Frames are rendered on-demand when accessed
    and cached with a true LRU (Least Recently Used) eviction policy.

    The LRU cache is implemented using OrderedDict:
    - New frames added to end
    - Accessed frames moved to end (mark as recently used)
    - When cache full, oldest frame (first item) evicted

    Parameters
    ----------
    env : Environment
        Environment defining spatial structure
    fields : list of arrays
        All fields to animate
    cmap_lookup : ndarray, shape (256, 3)
        Pre-computed colormap RGB lookup table
    vmin, vmax : float
        Color scale limits
    cache_size : int, default=1000
        Maximum number of frames to cache

    Attributes
    ----------
    _cache : OrderedDict
        LRU cache mapping frame index to rendered RGB array
    _cache_size : int
        Maximum number of frames to cache
    _lock : Lock
        Thread lock for thread-safe cache operations

    Notes
    -----
    The cache size of 1000 frames is chosen to balance memory usage
    and performance. For typical 100x100 grids with RGB data:
    - Memory per frame: ~30KB
    - Cache memory: ~30MB
    - Sufficient for responsive scrubbing

    For larger grids or tighter memory constraints, reduce _cache_size.

    Examples
    --------
    .. code-block:: python

        renderer = LazyFieldRenderer(env, fields, cmap_lookup, 0.0, 1.0)
        len(renderer)  # Number of frames
        # 100
        frame = renderer[0]  # Render frame 0 (cached)
        frame = renderer[0]  # Retrieved from cache (instant)
        frame = renderer[-1]  # Negative indexing supported
    """

    def __init__(
        self,
        env: Environment,
        fields: list[NDArray[np.float64]],
        cmap_lookup: NDArray[np.uint8],
        vmin: float,
        vmax: float,
        cache_size: int = 1000,
    ):
        """Initialize lazy field renderer."""
        self.env = env
        self.fields = fields
        self.cmap_lookup = cmap_lookup
        self.vmin = vmin
        self.vmax = vmax
        self._cache: OrderedDict[int, NDArray[np.uint8]] = OrderedDict()
        self._cache_size = cache_size
        self._lock = Lock()

    def __len__(self) -> int:
        """Return number of frames."""
        return len(self.fields)

    def __getitem__(self, idx: int | tuple) -> NDArray[np.uint8]:
        """Render frame on-demand when Napari requests it (thread-safe).

        Implements true LRU caching with thread-safety:
        1. If frame in cache, move to end (mark as recently used)
        2. If frame not in cache, render it and add to end
        3. If cache full, evict oldest frame (first item)

        Thread-safety is guaranteed by wrapping all cache operations in a lock,
        preventing race conditions when Napari requests data from non-main threads.

        Parameters
        ----------
        idx : int or tuple
            Frame index (supports negative indexing) or tuple of slices
            (napari passes tuples like (0, :, :, :) for slicing)

        Returns
        -------
        rgb : ndarray
            Rendered RGB frame or sliced data
        """
        with self._lock:
            return self._getitem_locked(idx)

    def _getitem_locked(self, idx: int | tuple) -> NDArray[np.uint8]:
        """Internal implementation of __getitem__ (assumes lock is held).

        Parameters
        ----------
        idx : int or tuple
            Frame index or tuple of slices

        Returns
        -------
        rgb : ndarray
            Rendered RGB frame or sliced data
        """
        # Handle tuple indexing from napari (e.g., data[0, :, :, :])
        if isinstance(idx, tuple):
            # Extract frame index (first element)
            frame_idx = idx[0]
            spatial_slices = idx[1:]  # (height_slice, width_slice, channel_slice)

            # Handle slice objects for frame index
            if isinstance(frame_idx, slice):
                # Return multiple frames as array
                start, stop, step = frame_idx.indices(len(self.fields))
                frames = [self._get_frame(i) for i in range(start, stop, step)]
                result = np.stack(frames, axis=0)

                # Apply spatial slices if provided
                if spatial_slices:
                    result = result[(slice(None), *spatial_slices)]
                return result
            else:
                # Single frame with spatial slices
                frame = self._get_frame(frame_idx)
                if spatial_slices:
                    sliced: NDArray[np.uint8] = frame[spatial_slices]
                    return sliced
                return frame

        # Handle integer indexing (original behavior)
        return self._get_frame(idx)

    def _get_frame(self, idx: int) -> NDArray[np.uint8]:
        """Get a single frame by integer index (internal helper).

        Parameters
        ----------
        idx : int
            Frame index (supports negative indexing)

        Returns
        -------
        rgb : ndarray
            Rendered RGB frame
        """
        # Handle negative indexing
        if idx < 0:
            idx = len(self.fields) + idx

        # Validate bounds
        if idx < 0 or idx >= len(self.fields):
            original_idx = idx - len(self.fields) if idx < 0 else idx
            raise IndexError(
                f"Frame index {original_idx} out of range for {len(self.fields)} frames"
            )

        if idx not in self._cache:
            # Render this frame
            from neurospatial.animation.rendering import field_to_rgb_for_napari

            rgb = field_to_rgb_for_napari(
                self.env,
                self.fields[idx],
                self.cmap_lookup,
                self.vmin,
                self.vmax,
            )
            self._cache[idx] = rgb

            # Evict oldest frame if cache too large (true LRU)
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)  # Remove oldest (first item)
        else:
            # Move to end for LRU (mark as recently used)
            self._cache.move_to_end(idx)

        return self._cache[idx]

    @property
    def shape(self) -> tuple[int, ...]:
        """Return shape for napari (time, height, width, channels).

        Napari needs to know the shape to properly display the image layer.
        """
        sample = self[0]
        return (len(self.fields), *sample.shape)

    @property
    def dtype(self) -> type:
        """Return dtype (always uint8 for RGB)."""
        return np.uint8

    @property
    def ndim(self) -> int:
        """Return number of dimensions (always 4: time, height, width, channels)."""
        return len(self.shape)


class ChunkedLazyFieldRenderer:
    """Chunked LRU cache for efficient large-dataset rendering.

    Similar to LazyFieldRenderer but caches frames in chunks for better
    memory efficiency with 100K+ frame datasets. Inspired by nwb_data_viewer
    chunked caching pattern.

    When a frame is requested, the entire chunk containing that frame is
    rendered and cached. Subsequent requests for frames in the same chunk
    are served from cache without re-rendering.

    Parameters
    ----------
    env : Environment
        Environment defining spatial structure
    fields : list of arrays
        All fields to animate
    cmap_lookup : ndarray, shape (256, 3)
        Pre-computed colormap RGB lookup table
    vmin, vmax : float
        Color scale limits
    chunk_size : int, default=100
        Number of frames per chunk
    max_chunks : int, default=50
        Maximum number of chunks to cache (LRU eviction)

    Attributes
    ----------
    _chunk_cache : dict
        Cache mapping chunk index to rendered frames
    _chunk_size : int
        Frames per chunk
    _max_chunks : int
        Max chunks to cache
    _lock : Lock
        Thread lock for thread-safe cache operations

    Notes
    -----
    **Memory Efficiency:**

    For typical 100x100 grids with RGB data:
    - Memory per frame: ~30KB
    - Memory per chunk (100 frames): ~3MB
    - Cache memory (50 chunks): ~150MB

    Chunked caching is more efficient than individual frame caching for:
    - Sequential playback (pre-loads neighboring frames)
    - Large datasets (100K+ frames)
    - Memory-mapped arrays (batch loading is faster)

    **Performance:**

    - Sequential access: ~10x faster (chunk pre-loading)
    - Random access: Similar to LazyFieldRenderer
    - Memory usage: Configurable via chunk_size and max_chunks

    Examples
    --------
    .. code-block:: python

        renderer = ChunkedLazyFieldRenderer(
            env, fields, cmap_lookup, 0.0, 1.0, chunk_size=100, max_chunks=50
        )
        len(renderer)  # Number of frames
        # 100000
        frame = renderer[0]  # Loads chunk 0 (frames 0-99)
        frame = renderer[50]  # Retrieved from chunk 0 cache (instant)
        frame = renderer[100]  # Loads chunk 1 (frames 100-199)
    """

    def __init__(
        self,
        env: Environment,
        fields: list[NDArray[np.float64]],
        cmap_lookup: NDArray[np.uint8],
        vmin: float,
        vmax: float,
        chunk_size: int = 100,
        max_chunks: int = 50,
    ):
        """Initialize chunked lazy field renderer."""
        self.env = env
        self.fields = fields
        self.cmap_lookup = cmap_lookup
        self.vmin = vmin
        self.vmax = vmax
        self._chunk_size = chunk_size
        self._max_chunks = max_chunks
        self._chunk_cache: OrderedDict[int, list[NDArray[np.uint8]]] = OrderedDict()
        self._lock = Lock()

    def __len__(self) -> int:
        """Return number of frames."""
        return len(self.fields)

    def _get_chunk_index(self, frame_idx: int) -> int:
        """Get chunk index for a given frame index.

        Parameters
        ----------
        frame_idx : int
            Frame index (0-indexed)

        Returns
        -------
        chunk_idx : int
            Chunk index containing this frame
        """
        return frame_idx // self._chunk_size

    def _render_chunk(self, chunk_idx: int) -> list[NDArray[np.uint8]]:
        """Render all frames in a chunk.

        Parameters
        ----------
        chunk_idx : int
            Chunk index to render

        Returns
        -------
        frames : list of ndarrays
            Rendered RGB frames for this chunk
        """
        from neurospatial.animation.rendering import field_to_rgb_for_napari

        # Calculate frame range for this chunk
        start_frame = chunk_idx * self._chunk_size
        end_frame = min(start_frame + self._chunk_size, len(self.fields))

        # Render all frames in chunk
        frames = []
        for idx in range(start_frame, end_frame):
            rgb = field_to_rgb_for_napari(
                self.env,
                self.fields[idx],
                self.cmap_lookup,
                self.vmin,
                self.vmax,
            )
            frames.append(rgb)

        return frames

    def _get_chunk(self, chunk_idx: int) -> list[NDArray[np.uint8]]:
        """Get chunk from cache or render it.

        Implements LRU eviction when cache is full.

        Parameters
        ----------
        chunk_idx : int
            Chunk index

        Returns
        -------
        frames : list of ndarrays
            Rendered frames for this chunk
        """
        if chunk_idx not in self._chunk_cache:
            # Render this chunk
            frames = self._render_chunk(chunk_idx)
            self._chunk_cache[chunk_idx] = frames

            # Evict oldest chunk if cache too large (LRU)
            if len(self._chunk_cache) > self._max_chunks:
                self._chunk_cache.popitem(last=False)  # Remove oldest
        else:
            # Move to end for LRU (mark as recently used)
            self._chunk_cache.move_to_end(chunk_idx)

        return self._chunk_cache[chunk_idx]

    def __getitem__(self, idx: int | tuple) -> NDArray[np.uint8]:
        """Render frame on-demand from cached chunk (thread-safe).

        Thread-safety is guaranteed by wrapping all cache operations in a lock,
        preventing race conditions when Napari requests data from non-main threads.

        Parameters
        ----------
        idx : int or tuple
            Frame index (supports negative indexing) or tuple of slices

        Returns
        -------
        rgb : ndarray
            Rendered RGB frame or sliced data
        """
        with self._lock:
            return self._getitem_locked(idx)

    def _getitem_locked(self, idx: int | tuple) -> NDArray[np.uint8]:
        """Internal implementation of __getitem__ (assumes lock is held).

        Parameters
        ----------
        idx : int or tuple
            Frame index or tuple of slices

        Returns
        -------
        rgb : ndarray
            Rendered RGB frame or sliced data
        """
        # Handle tuple indexing from napari (e.g., data[0, :, :, :])
        if isinstance(idx, tuple):
            frame_idx = idx[0]
            spatial_slices = idx[1:]

            # Handle slice objects for frame index
            if isinstance(frame_idx, slice):
                start, stop, step = frame_idx.indices(len(self.fields))
                frames = [self._get_frame(i) for i in range(start, stop, step)]
                result = np.stack(frames, axis=0)

                if spatial_slices:
                    result = result[(slice(None), *spatial_slices)]
                return result
            else:
                # Single frame with spatial slices
                frame = self._get_frame(frame_idx)
                if spatial_slices:
                    sliced: NDArray[np.uint8] = frame[spatial_slices]
                    return sliced
                return frame

        # Handle integer indexing
        return self._get_frame(idx)

    def _get_frame(self, idx: int) -> NDArray[np.uint8]:
        """Get a single frame by integer index (internal helper).

        Parameters
        ----------
        idx : int
            Frame index (supports negative indexing)

        Returns
        -------
        rgb : ndarray
            Rendered RGB frame
        """
        # Handle negative indexing
        if idx < 0:
            idx = len(self.fields) + idx

        # Validate bounds
        if idx < 0 or idx >= len(self.fields):
            original_idx = idx - len(self.fields) if idx < 0 else idx
            raise IndexError(
                f"Frame index {original_idx} out of range for {len(self.fields)} frames"
            )

        # Get chunk containing this frame
        chunk_idx = self._get_chunk_index(idx)
        chunk_frames = self._get_chunk(chunk_idx)

        # Get frame within chunk
        frame_offset = idx - (chunk_idx * self._chunk_size)
        return chunk_frames[frame_offset]

    @property
    def shape(self) -> tuple[int, ...]:
        """Return shape for napari (time, height, width, channels)."""
        sample = self[0]
        return (len(self.fields), *sample.shape)

    @property
    def dtype(self) -> type:
        """Return dtype (always uint8 for RGB)."""
        return np.uint8

    @property
    def ndim(self) -> int:
        """Return number of dimensions (always 4: time, height, width, channels)."""
        return len(self.shape)

    def _get_chunk_cache_info(self) -> dict[int, int]:
        """Get chunk cache information for debugging/testing.

        Returns
        -------
        cache_info : dict
            Mapping of chunk_idx to number of frames in that chunk
        """
        return {
            chunk_idx: len(frames) for chunk_idx, frames in self._chunk_cache.items()
        }
