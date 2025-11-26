"""Napari GPU-accelerated viewer backend."""

from __future__ import annotations

import contextlib
import os
import warnings
from collections import OrderedDict
from threading import Lock
from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

# Import shared coordinate transforms
from neurospatial.animation.transforms import EnvScale as _EnvScale
from neurospatial.animation.transforms import make_env_scale as _make_env_scale
from neurospatial.animation.transforms import (
    transform_coords_for_napari as _transform_coords_for_napari,
)
from neurospatial.animation.transforms import (
    transform_direction_for_napari as _transform_direction_for_napari,
)

# Re-export for backward compatibility (used by tests)
__all__ = ["_EnvScale", "_make_env_scale"]

# Performance monitoring support (enabled via NAPARI_PERFMON env var)
_PERFMON_ENABLED = bool(os.environ.get("NAPARI_PERFMON"))
if _PERFMON_ENABLED:
    try:
        from napari.utils.perf import add_instant_event, perf_timer
    except ImportError:
        _PERFMON_ENABLED = False
        perf_timer = contextlib.nullcontext
        add_instant_event = lambda *args, **kwargs: None  # noqa: E731
else:
    perf_timer = contextlib.nullcontext
    add_instant_event = lambda *args, **kwargs: None  # noqa: E731

if TYPE_CHECKING:
    import napari
    from napari.layers import Layer

    from neurospatial.animation.overlays import (
        BodypartData,
        HeadDirectionData,
        OverlayData,
        PositionData,
        VideoData,
    )
    from neurospatial.environment.core import Environment

# Check napari availability
try:
    import napari

    NAPARI_AVAILABLE = True
except ImportError:
    napari = None
    NAPARI_AVAILABLE = False

# =============================================================================
# Constants
# =============================================================================

# Overlay rendering constants
POINT_MARKER_RADIUS: float = 2.0
"""Radius for point region markers in napari pixel units."""

POINT_BORDER_WIDTH: float = 0.5
"""Border width for position and bodypart point markers."""

BODYPART_POINT_SIZE: float = 5.0
"""Default size for bodypart point markers."""

SKELETON_DEFAULT_WIDTH: float = 2.0
"""Default edge width for skeleton lines."""

# Region rendering constants
REGION_CIRCLE_SEGMENTS: int = 9
"""Number of line segments to approximate circles for point regions (octagon + close)."""

# Cache constants
DEFAULT_CACHE_SIZE: int = 1000
"""Default maximum number of frames to cache for per-frame caching."""

DEFAULT_CHUNK_SIZE: int = 10
"""Default number of frames per chunk for chunked caching."""

DEFAULT_MAX_CHUNKS: int = 100
"""Default maximum number of chunks to cache."""

CHUNKED_CACHE_THRESHOLD: int = 10_000
"""Frame count threshold above which chunked caching is used instead of per-frame."""

# Playback constants
DEFAULT_FPS: int = 30
"""Default frames per second for playback."""

FPS_SLIDER_MIN: int = 1
"""Minimum FPS value for playback slider."""

FPS_SLIDER_DEFAULT_MAX: int = 120
"""Default maximum FPS value for playback slider."""

WIDGET_UPDATE_TARGET_HZ: int = 30
"""Target update rate for playback widget to avoid Qt overhead at high FPS."""

# Per-viewer warning state metadata key
_TRANSFORM_WARNED_KEY: str = "_transform_fallback_warned"
"""Metadata key for tracking transform fallback warning per viewer."""


# =============================================================================
# Per-Viewer Warning State Management
# =============================================================================


def _check_viewer_warned(viewer: napari.Viewer) -> bool:
    """Check if viewer has already shown transform fallback warning.

    Parameters
    ----------
    viewer : napari.Viewer
        Napari viewer instance.

    Returns
    -------
    bool
        True if viewer has already shown the warning, False otherwise.
    """
    return bool(viewer.metadata.get(_TRANSFORM_WARNED_KEY, False))


def _mark_viewer_warned(viewer: napari.Viewer) -> None:
    """Mark viewer as having shown transform fallback warning.

    Parameters
    ----------
    viewer : napari.Viewer
        Napari viewer instance.
    """
    viewer.metadata[_TRANSFORM_WARNED_KEY] = True


def _transform_coords_with_viewer(
    coords: NDArray[np.float64],
    env: Environment,
    viewer: napari.Viewer,
) -> NDArray[np.float64]:
    """Transform coordinates with per-viewer warning tracking.

    This wrapper around _transform_coords_for_napari manages warning state
    per-viewer, ensuring each viewer receives the fallback warning at most once.

    Parameters
    ----------
    coords : ndarray
        Coordinates in environment (x, y) space.
    env : Environment
        Environment instance for coordinate transformation.
    viewer : napari.Viewer
        Napari viewer instance for warning state tracking.

    Returns
    -------
    ndarray
        Coordinates in napari pixel (row, col) space.
    """
    already_warned = _check_viewer_warned(viewer)
    result = _transform_coords_for_napari(coords, env, suppress_warning=already_warned)
    # Mark as warned if we didn't suppress (warning may have been shown)
    if not already_warned:
        _mark_viewer_warned(viewer)
    return result


def _transform_direction_with_viewer(
    direction: NDArray[np.float64],
    env: Environment,
    viewer: napari.Viewer,
) -> NDArray[np.float64]:
    """Transform direction vectors with per-viewer warning tracking.

    This wrapper around _transform_direction_for_napari manages warning state
    per-viewer, ensuring each viewer receives the fallback warning at most once.

    Parameters
    ----------
    direction : ndarray
        Direction vectors in environment (dx, dy) space.
    env : Environment
        Environment instance for coordinate transformation.
    viewer : napari.Viewer
        Napari viewer instance for warning state tracking.

    Returns
    -------
    ndarray
        Direction vectors in napari pixel (dr, dc) space.
    """
    already_warned = _check_viewer_warned(viewer)
    result = _transform_direction_for_napari(
        direction, env, suppress_warning=already_warned
    )
    # Mark as warned if we didn't suppress (warning may have been shown)
    if not already_warned:
        _mark_viewer_warned(viewer)
    return result


# =============================================================================
# Overlay Rendering Helper Functions
# =============================================================================


def _build_skeleton_vectors(
    bodypart_data: BodypartData,
    env: Environment,
    *,
    dtype: type[np.floating] = np.float32,
) -> tuple[NDArray[np.floating], dict[str, NDArray[np.object_]]]:
    """Precompute napari Vectors data for skeleton edges across all frames.

    This function precomputes all skeleton vectors at initialization time rather
    than computing them per-frame, eliminating the per-frame callback overhead
    that causes playback stalling.

    Parameters
    ----------
    bodypart_data : BodypartData
        Bodypart overlay data containing:
        - bodyparts: dict mapping part names to (n_frames, n_dims) coordinate arrays
        - skeleton: list of (start_part, end_part) tuples defining edges
    env : Environment
        Environment instance for coordinate transformation to napari pixel space.
    dtype : np.dtype, default=np.float32
        Data type for vectors array. Float32 recommended for memory efficiency.

    Returns
    -------
    vectors : ndarray, shape (n_segments, 2, 3)
        Pre-computed skeleton vectors for all frames. Each segment has format:
        [[t, y0, x0], [0, dy, dx]] where:
        - t is frame index
        - (y0, x0) is the start position in napari pixel coordinates
        - (dy, dx) is the direction/displacement vector to the end point
        - n_segments = n_frames * n_valid_edges
        This format follows napari's Vectors layer convention of [position, direction].
    features : dict[str, ndarray]
        Feature arrays parallel to vectors:
        - "edge_name": str array with format "start-end" for each segment

    Notes
    -----
    Skeleton edges with missing bodyparts or NaN coordinates are excluded from
    the output. The time dimension (t) uses frame indices (0, 1, 2, ...) to
    enable napari's native time slicing.

    This approach eliminates the 5.38ms per-frame `layer.data` assignment that
    was blocking the Qt event loop during playback.

    See Also
    --------
    _render_bodypart_overlay : Uses this function for skeleton rendering

    Examples
    --------
    >>> vectors, features = _build_skeleton_vectors(bodypart_data, env)
    >>> viewer.add_vectors(vectors, features=features, ...)
    """
    # Early exit if no skeleton defined
    skeleton = bodypart_data.skeleton
    if skeleton is None or len(skeleton.edges) == 0:
        empty_vectors = np.empty((0, 2, 3), dtype=dtype)
        empty_features: dict[str, NDArray[np.object_]] = {
            "edge_name": np.empty(0, dtype=object)
        }
        return empty_vectors, empty_features

    # Get frame count and validate bodyparts exist
    bodyparts = bodypart_data.bodyparts
    if not bodyparts:
        empty_vectors = np.empty((0, 2, 3), dtype=dtype)
        empty_features = {"edge_name": np.empty(0, dtype=object)}
        return empty_vectors, empty_features

    skeleton_edges = skeleton.edges

    # Pre-compute scale factors for coordinate transformation
    env_scale = _EnvScale.from_env(env)

    # Pre-transform all bodypart coordinates to napari space (once per bodypart)
    napari_coords: dict[str, NDArray[np.float64]] = {}
    for part_name, coords in bodyparts.items():
        napari_coords[part_name] = _transform_coords_for_napari(
            coords, env_scale if env_scale is not None else env
        )

    # Build vectors for all edges using vectorized operations
    # Collect per-edge results for concatenation
    edge_vectors_list: list[NDArray[np.floating]] = []
    edge_names_list: list[NDArray[np.object_]] = []

    for start_part, end_part in skeleton_edges:
        # Skip if either bodypart is missing
        if start_part not in napari_coords or end_part not in napari_coords:
            continue

        edge_name = f"{start_part}-{end_part}"
        start_coords = napari_coords[start_part]  # (n_frames, 2)
        end_coords = napari_coords[end_part]  # (n_frames, 2)

        # Vectorized NaN check: find frames where both endpoints are valid
        # ~np.isnan(...).any(axis=1) gives True for rows with all finite values
        valid_mask = ~np.isnan(start_coords).any(axis=1) & ~np.isnan(end_coords).any(
            axis=1
        )
        valid_frame_indices = np.where(valid_mask)[0]

        n_valid = len(valid_frame_indices)
        if n_valid == 0:
            continue

        # Extract valid coordinates (n_valid, 2)
        valid_start = start_coords[valid_frame_indices]
        valid_end = end_coords[valid_frame_indices]

        # Build vectors array for this edge: (n_valid, 2, 3)
        edge_vectors = np.empty((n_valid, 2, 3), dtype=dtype)

        # Position row: [t, y, x]
        edge_vectors[:, 0, 0] = valid_frame_indices  # frame indices
        edge_vectors[:, 0, 1] = valid_start[:, 0]  # row (y in napari)
        edge_vectors[:, 0, 2] = valid_start[:, 1]  # col (x in napari)

        # Direction row: [dt, dy, dx]
        edge_vectors[:, 1, 0] = 0  # dt = 0 (same time)
        edge_vectors[:, 1, 1] = valid_end[:, 0] - valid_start[:, 0]  # dy
        edge_vectors[:, 1, 2] = valid_end[:, 1] - valid_start[:, 1]  # dx

        edge_vectors_list.append(edge_vectors)

        # Edge names for this edge (all same name)
        edge_names_arr = np.full(n_valid, edge_name, dtype=object)
        edge_names_list.append(edge_names_arr)

    # Concatenate all edges
    if edge_vectors_list:
        vectors = np.concatenate(edge_vectors_list, axis=0)
        edge_names = np.concatenate(edge_names_list)
    else:
        vectors = np.empty((0, 2, 3), dtype=dtype)
        edge_names = np.empty(0, dtype=object)

    features: dict[str, NDArray[np.object_]] = {"edge_name": edge_names}
    return vectors, features


# =============================================================================
# Video Layer Rendering
# =============================================================================


def _add_video_layer(
    viewer: napari.Viewer,
    video_data: VideoData,
    env: Environment,
    n_frames: int,
    name: str = "Video",
) -> Layer:
    """Add video overlay as a streaming Image layer.

    Creates a napari Image layer that displays video frames synchronized with
    the animation. Supports affine transforms to position the video in
    environment coordinates.

    Parameters
    ----------
    viewer : napari.Viewer
        Napari viewer instance.
    video_data : VideoData
        Video overlay data containing frame indices, reader, transform, etc.
    env : Environment
        Environment instance for coordinate transforms.
    n_frames : int
        Total number of animation frames.
    name : str, default="Video"
        Name for the video layer.

    Returns
    -------
    layer : Layer
        The created napari Image layer.

    Notes
    -----
    The video layer uses napari's affine parameter to position the video
    in environment coordinates. Frame updates are handled by the callback
    registered via viewer.dims.events.current_step.

    Z-ordering is controlled by the `z_order` attribute:
    - "below": Video layer is added before field layer (lower in stack)
    - "above": Video layer is added after field layer (higher in stack)

    **Coordinate Transform Chain** (see animation/COORDINATES.md):

    The affine parameter encodes the full transform: video_px → napari_px

    .. code-block:: text

        video_px ──► env_cm ──► napari_px
                 │          │
                 │          └── build_env_to_napari_matrix()
                 │              (y-invert + axis swap + scale)
                 │
                 └── video_data.transform_to_env
                     (y-flip + scale from calibration)

    **Frame Index Mapping**:

    ``video_data.frame_indices[anim_frame]`` maps animation frames to video frames:
    - >= 0: Valid video frame index to display
    - -1: No video frame available (animation time outside video range)

    The callback ``_make_video_frame_callback()`` uses this mapping to update
    ``layer.data`` when the animation frame changes.
    """

    # Build affine transform matrix
    # Chains: video_px → env_cm → napari_px (see COORDINATES.md for details)
    affine = _build_video_napari_affine(video_data, env)

    # Get initial frame
    initial_frame_idx = (
        video_data.frame_indices[0] if len(video_data.frame_indices) > 0 else 0
    )
    if initial_frame_idx >= 0:
        if isinstance(video_data.reader, np.ndarray):
            initial_frame = video_data.reader[initial_frame_idx]
        else:
            # VideoReader interface
            initial_frame = video_data.reader[initial_frame_idx]
    else:
        # No valid frame - create blank
        if isinstance(video_data.reader, np.ndarray):
            h, w = video_data.reader.shape[1:3]
        else:
            # Estimate from env_bounds
            h, w = 64, 64  # fallback
        initial_frame = np.zeros((h, w, 3), dtype=np.uint8)

    # Add the image layer
    layer = viewer.add_image(
        initial_frame,
        name=name,
        rgb=True,
        opacity=video_data.alpha,
        affine=affine,
        blending="translucent",
    )

    # Store video data for frame updates
    layer.metadata["video_data"] = video_data
    layer.metadata["video_frame_indices"] = video_data.frame_indices

    return layer


def _make_video_frame_callback(
    viewer: napari.Viewer,
    video_layers: list[Layer],
) -> None:
    """Register callback to update video layers when animation frame changes.

    Creates and connects a callback to viewer.dims.events.current_step that
    updates each video layer's data based on the current animation frame.

    Parameters
    ----------
    viewer : napari.Viewer
        Napari viewer instance.
    video_layers : list[Layer]
        List of video layers to update. Each layer must have metadata keys
        'video_data' and 'video_frame_indices'.
    """
    if not video_layers:
        return

    def update_video_frames(event: Any | None = None) -> None:
        """Update all video layers to show frame at current animation time."""
        # Get current animation frame from first dimension (time)
        if viewer.dims.ndim == 0:
            return
        current_frame = int(viewer.dims.current_step[0])

        for layer in video_layers:
            video_data = layer.metadata.get("video_data")
            if video_data is None:
                continue

            # Get the video frame for this animation frame
            frame = video_data.get_frame(current_frame)
            if frame is not None:
                # Update layer data in-place
                layer.data = frame

    # Connect callback to dims events
    viewer.dims.events.current_step.connect(update_video_frames)

    # Trigger initial update
    update_video_frames()


def _build_video_napari_affine(
    video_data: VideoData,
    env: Environment,
) -> NDArray[np.float64]:
    """Build affine transform matrix for video positioning in napari.

    Combines the video-to-environment transform with the environment-to-napari
    transform to position video frames correctly in the napari viewer.

    Parameters
    ----------
    video_data : VideoData
        Video overlay data containing transform_to_env and env_bounds.
    env : Environment
        Environment instance for coordinate transforms.

    Returns
    -------
    affine : ndarray of shape (3, 3)
        2D affine transform matrix for napari's Image layer.

    Notes
    -----
    The full transform chain is:
    video_px → (transform_to_env) → env_cm → (env_to_napari) → napari_px

    If transform_to_env is None, uses env_bounds to position the video
    by computing a simple scale+translate transform from video pixel
    coordinates to environment bounds.
    """
    from neurospatial.animation.transforms import EnvScale, build_env_to_napari_matrix

    # Get environment scale factors
    scale = EnvScale.from_env(env)
    if scale is None:
        # Fallback: return identity (video displayed in pixel coords)
        return np.eye(3, dtype=np.float64)

    # Build env→napari matrix
    env_to_napari = build_env_to_napari_matrix(scale)

    # Get video dimensions
    if isinstance(video_data.reader, np.ndarray):
        video_h, video_w = video_data.reader.shape[1:3]
    else:
        # VideoReader provides frame_size_px as (width, height)
        video_w, video_h = video_data.reader.frame_size_px

    # Build video→env matrix based on env_bounds
    xmin, xmax, ymin, ymax = video_data.env_bounds

    if video_data.transform_to_env is not None:
        # User's transform expects (x, y) input, but napari provides (row, col)
        # Need to convert: (row, col) → (x, y) → env_coords
        # where x = col, y = row (swap axes)
        row_col_to_xy = np.array(
            [
                [0.0, 1.0, 0.0],  # x = col
                [1.0, 0.0, 0.0],  # y = row
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        video_to_env = video_data.transform_to_env.A @ row_col_to_xy
    else:
        # Build scale+translate from video pixels to env bounds
        # Video pixel (0,0) is top-left, (video_w, video_h) is bottom-right
        # But in env coords, we typically want y increasing upward

        # Scale factors: map [0, video_w) → [xmin, xmax), [0, video_h) → [ymin, ymax)
        sx = (xmax - xmin) / video_w if video_w > 0 else 1.0
        sy = (ymax - ymin) / video_h if video_h > 0 else 1.0

        # Video y=0 at top → env y=ymax, video y=video_h at bottom → env y=ymin
        # So: env_y = ymax - (video_row / video_h) * (ymax - ymin)
        #           = ymax - video_row * sy
        #           = -sy * video_row + ymax

        # Video x follows same direction as env x
        # env_x = xmin + video_col * sx

        # Matrix form (row, col) → (x, y):
        # [x]   [0,  sx, xmin] [row]
        # [y] = [-sy, 0, ymax] [col]
        # [1]   [0,  0,   1 ] [1  ]
        #
        # But we need (row, col) input format for napari:
        video_to_env = np.array(
            [
                [0.0, sx, xmin],  # x = sx * col + xmin
                [-sy, 0.0, ymax],  # y = -sy * row + ymax
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    # Combine: video_px → env_cm → napari_px
    # Note: env_to_napari operates on (x, y), but napari expects (row, col)
    # The video_to_env converts (row, col) to (x, y)
    # The env_to_napari converts (x, y) to (row, col)
    # Combined: (row_v, col_v) → (x, y) → (row_n, col_n)
    affine: NDArray[np.float64] = env_to_napari @ video_to_env

    return affine


def _render_position_overlay(
    viewer: napari.Viewer,
    position_data: PositionData,
    env: Environment,
    name_suffix: str = "",
) -> list[Layer]:
    """Render a single position overlay with optional trail.

    Parameters
    ----------
    viewer : napari.Viewer
        Napari viewer instance
    position_data : PositionData
        Position overlay data aligned to frames
    env : Environment
        Environment instance for coordinate transformation
    name_suffix : str
        Suffix for layer names (e.g., for multi-animal)

    Returns
    -------
    layers : list of Layer
        List of created napari layer objects for later updates

    Notes
    -----
    Trail coloring requires a property-based colormap approach because napari's
    Tracks layer does not support a direct `color` parameter (unlike add_points
    which accepts face_color). The Tracks layer exclusively uses property-based
    coloring via `color_by` + `colormaps_dict`. See inline comments and
    https://napari.org/stable/api/napari.layers.Tracks.html for details.
    """
    layers: list[Layer] = []
    n_frames = len(position_data.data)

    # Transform coordinates to napari (y, x) convention with Y-axis inversion
    transformed_data = _transform_coords_for_napari(position_data.data, env)

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

        # Napari tracks layer uniform color workaround (REQUIRED - no simpler alternative):
        #
        # Unlike add_points (which accepts face_color directly), the Tracks layer
        # has NO direct `color` parameter. Per napari docs and source code, track
        # coloring is exclusively feature-based:
        #   - color_by: selects which feature column determines vertex colors
        #   - colormap/colormaps_dict: maps feature values to colors
        #
        # To achieve uniform coloring, we:
        # 1. Create a constant feature ("color" = all zeros) for each track point
        # 2. Create a Colormap mapping any value to our desired color
        #    (Colormap requires 2+ control points, so we use identical colors)
        # 3. Set color_by="color" to apply this feature-to-colormap mapping
        #
        # Note: napari 0.5+ uses `features` parameter (not `properties`).
        #
        # See: https://napari.org/stable/api/napari.layers.Tracks.html
        from napari.utils.colormaps import Colormap

        features = {"color": np.zeros(n_frames)}
        custom_colormap = Colormap(
            colors=[position_data.color, position_data.color],
            name=f"trail_color{name_suffix}",
        )
        colormaps_dict = {"color": custom_colormap}

        # Note: We set color_by AFTER layer creation to avoid napari warning.
        # During __init__, napari's data setter resets features to {} before our
        # features are applied. If color_by is passed at init time, the check runs
        # against empty features and warns. Setting color_by after creation avoids this.
        layer = viewer.add_tracks(
            track_data,
            name=f"Position Trail{name_suffix}",
            tail_length=position_data.trail_length,
            features=features,
            colormaps_dict=colormaps_dict,
        )
        layer.color_by = "color"  # Set after features are applied
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
        border_color="white",
        border_width=POINT_BORDER_WIDTH,
    )
    layers.append(layer)

    return layers


def _render_bodypart_overlay(
    viewer: napari.Viewer,
    bodypart_data: BodypartData,
    env: Environment,
    name_suffix: str = "",
) -> list[Layer]:
    """Render a single bodypart overlay with skeleton.

    Parameters
    ----------
    viewer : napari.Viewer
        Napari viewer instance
    bodypart_data : BodypartData
        Bodypart overlay data aligned to frames
    env : Environment
        Environment instance for coordinate transformation
    name_suffix : str
        Suffix for layer names

    Returns
    -------
    layers : list of Layer
        List of created napari layer objects for later updates
    """
    layers: list[Layer] = []

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
        # Transform coordinates with Y-axis inversion
        transformed = _transform_coords_for_napari(coords, env)

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
        size=BODYPART_POINT_SIZE,
        face_color=face_colors,
        border_color="black",
        border_width=POINT_BORDER_WIDTH,
        features={"bodypart": all_features},
    )
    layers.append(layer)

    # Render skeleton if provided using precomputed vectors layer
    # This eliminates per-frame callback overhead (5.38ms per frame) that was
    # blocking the Qt event loop during playback. The vectors layer handles
    # time slicing natively via napari's built-in dims.
    if bodypart_data.skeleton is not None:
        # Precompute all skeleton vectors at initialization (not per-frame)
        vectors_data, vector_features = _build_skeleton_vectors(bodypart_data, env)

        # Only add layer if there are valid skeleton segments
        if vectors_data.size > 0:
            # Get styling from Skeleton object
            skeleton = bodypart_data.skeleton
            skeleton_layer = viewer.add_vectors(
                vectors_data,
                name=f"Skeleton{name_suffix}",
                edge_color=skeleton.edge_color,
                edge_width=skeleton.edge_width,
                vector_style="line",  # Hide arrow heads for skeleton lines
                features=vector_features,
            )
            layers.append(skeleton_layer)

    return layers


def _build_head_direction_tracks(
    head_dir_data: HeadDirectionData,
    env: Environment,
    position_data: PositionData | None = None,
    *,
    dtype: type[np.floating] = np.float32,
) -> NDArray[np.floating]:
    """Build head direction tracks for Tracks layer rendering.

    Creates track data where each frame's direction line is a short track segment.
    Uses napari's Tracks layer which is optimized for time-based filtering.

    Parameters
    ----------
    head_dir_data : HeadDirectionData
        Head direction overlay data with angles or unit vectors.
    env : Environment
        Environment instance for coordinate transformation.
    position_data : PositionData | None, optional
        Head position data. If None, uses environment centroid as origin.
    dtype : type, optional
        Data type for output arrays. Default is np.float32.

    Returns
    -------
    tracks : ndarray of shape (n_valid * 2, 4)
        Track data in format [track_id, t, y, x]. Each frame's line is a
        separate track with 2 points (origin and indicator).
    """
    n_frames = len(head_dir_data.data)
    is_angles = head_dir_data.data.ndim == 1

    # Pre-compute scale factors
    env_scale = _EnvScale.from_env(env)
    transform_target = env_scale if env_scale is not None else env

    # Compute origins
    if position_data is not None:
        origins = position_data.data  # (n_frames, n_dims)
    else:
        centroid = np.mean(env.bin_centers, axis=0)
        origins = np.tile(centroid, (n_frames, 1))

    # Compute direction vectors
    if is_angles:
        angles = head_dir_data.data
        directions = np.column_stack(
            [
                np.cos(angles) * head_dir_data.length,
                np.sin(angles) * head_dir_data.length,
            ]
        )
    else:
        directions = head_dir_data.data.copy()
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        directions = directions / norms * head_dir_data.length

    # Compute indicator positions
    indicator_positions = origins + directions

    # Transform to napari coordinates
    origins_napari = _transform_coords_for_napari(origins, transform_target)
    indicator_napari = _transform_coords_for_napari(
        indicator_positions, transform_target
    )

    # Find valid frames
    valid_origins = ~np.isnan(origins).any(axis=1)
    valid_directions = (
        ~np.isnan(head_dir_data.data).any(axis=-1)
        if head_dir_data.data.ndim > 1
        else ~np.isnan(head_dir_data.data)
    )
    valid_indicators = ~np.isnan(indicator_napari).any(axis=1)
    valid_mask = valid_origins & valid_directions & valid_indicators
    valid_frame_indices = np.where(valid_mask)[0]

    n_valid = len(valid_frame_indices)
    if n_valid == 0:
        return np.empty((0, 4), dtype=dtype)

    # Build tracks array: [track_id, t, y, x]
    # Each frame's line is a separate track with 2 points
    # Points must be sorted by time within each track for napari
    # We put indicator at t+0.5, origin at t, so tail shows origin→indicator
    tracks = np.empty((n_valid * 2, 4), dtype=dtype)

    # Origin points at time t (earlier point)
    tracks[0::2, 0] = valid_frame_indices  # track_id = frame index
    tracks[0::2, 1] = valid_frame_indices  # t = frame index
    tracks[0::2, 2] = origins_napari[valid_frame_indices, 0]  # y
    tracks[0::2, 3] = origins_napari[valid_frame_indices, 1]  # x

    # Indicator points at time t+0.5 (later point - current "head" of track)
    tracks[1::2, 0] = valid_frame_indices  # same track_id
    tracks[1::2, 1] = valid_frame_indices + 0.5  # t + offset
    tracks[1::2, 2] = indicator_napari[valid_frame_indices, 0]  # y
    tracks[1::2, 3] = indicator_napari[valid_frame_indices, 1]  # x

    return tracks


def _render_head_direction_overlay(
    viewer: napari.Viewer,
    head_dir_data: HeadDirectionData,
    env: Environment,
    name_suffix: str = "",
    position_data: PositionData | None = None,
) -> list[Layer]:
    """Render head direction as a line from head position to direction indicator.

    Uses napari's Tracks layer for efficient time-based filtering, achieving
    smooth 30 FPS playback even with 40k+ frames.

    Parameters
    ----------
    viewer : napari.Viewer
        Napari viewer instance
    head_dir_data : HeadDirectionData
        Head direction data aligned to frames. The ``length`` attribute controls
        the line length, and ``width`` controls the line thickness.
    env : Environment
        Environment for coordinate transformation
    name_suffix : str
        Suffix for layer names
    position_data : PositionData | None, optional
        If provided, lines start from position coordinates.
        If None, lines start from environment centroid.
        Default is None.

    Returns
    -------
    layers : list of Layer
        List of created napari layer objects for later updates

    Notes
    -----
    The direction line extends from head position to:
    ``head_position + length * unit_direction``
    """
    layers: list[Layer] = []

    tracks = _build_head_direction_tracks(
        head_dir_data, env, position_data, dtype=np.float32
    )
    if tracks.size > 0:
        layer = viewer.add_tracks(
            tracks,
            name=f"Head Direction{name_suffix}",
            tail_length=0,  # No past tracks
            head_length=1,  # Show current track (origin→indicator)
            tail_width=head_dir_data.width,
            blending="opaque",  # Better visibility against dark backgrounds
        )
        layer.colormap = head_dir_data.color
        layers.append(layer)

    return layers


def _render_regions(
    viewer: napari.Viewer,
    env: Environment,
    show_regions: bool | list[str],
    region_alpha: float,
) -> list[Layer]:
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
    layers : list of Layer
        List of created napari layer objects
    """
    layers: list[Layer] = []

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
            coords = _transform_coords_for_napari(region.data.reshape(1, -1), env)[0]
            # Create circle polygon (approximate with octagon)
            angles = np.linspace(0, 2 * np.pi, REGION_CIRCLE_SEGMENTS)
            circle = np.column_stack(
                [
                    coords[0] + POINT_MARKER_RADIUS * np.cos(angles),
                    coords[1] + POINT_MARKER_RADIUS * np.sin(angles),
                ]
            )
            region_shapes.append(circle)
            region_properties["name"].append(region_name)
        elif region.kind == "polygon":
            # Polygon regions: use directly
            # Extract coordinates from Shapely polygon
            # Note: region.data is a Shapely Polygon here
            coords = np.array(region.data.exterior.coords)  # type: ignore[union-attr]
            coords_napari = _transform_coords_for_napari(coords, env)
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
    - List of lists/tuples → multi-field (True)

    This function only checks the first element for quick detection.
    For robust validation that all elements are consistent, use
    _validate_field_types_consistent() before calling this.
    """
    if len(fields) == 0:
        return False

    # Check if first element is a list/sequence (not ndarray)
    first_elem = fields[0]
    return isinstance(first_elem, (list, tuple))


def _validate_field_types_consistent(fields: list) -> None:
    """Validate that all field elements have consistent types.

    Parameters
    ----------
    fields : list
        Input fields to validate

    Raises
    ------
    ValueError
        If fields contain mixed types (some arrays, some lists/tuples).
        Provides a clear WHAT/WHY/HOW error message.

    Notes
    -----
    This validation should be called before _is_multi_field_input() to ensure
    the detection is reliable. The check ensures ALL elements are either:
    - All numpy arrays (single-field mode)
    - All lists/tuples (multi-field mode)

    Mixed inputs like [list, array, list] will raise an error with guidance.
    """
    if len(fields) == 0:
        return  # Empty is valid (single-field)

    # Determine expected type from first element
    first_elem = fields[0]
    first_is_sequence = isinstance(first_elem, (list, tuple))

    # Check all elements match the first element's type
    mismatched_indices = []
    for i, elem in enumerate(fields):
        elem_is_sequence = isinstance(elem, (list, tuple))
        if elem_is_sequence != first_is_sequence:
            mismatched_indices.append(i)

    if mismatched_indices:
        # Build informative error message
        expected_type = "list/tuple (multi-field)" if first_is_sequence else "array"
        actual_types = []
        for i in mismatched_indices[:3]:  # Show first 3 mismatches
            elem = fields[i]
            actual_type = (
                "list/tuple" if isinstance(elem, (list, tuple)) else type(elem).__name__
            )
            actual_types.append(f"  fields[{i}]: {actual_type}")

        raise ValueError(
            f"WHAT: Inconsistent field types detected - mixed arrays and sequences.\n"
            f"  fields[0] is: {expected_type}\n"
            f"  Mismatched elements:\n" + "\n".join(actual_types) + "\n\n"
            "WHY: Fields must be either:\n"
            "  - Single-field mode: list of 1D arrays [arr1, arr2, ...]\n"
            "  - Multi-field mode: list of sequences [[arr1, arr2], [arr3, arr4]]\n"
            "  Mixed types cannot be processed correctly.\n\n"
            "HOW: Ensure all elements are the same type:\n"
            "  # Single-field: list of arrays\n"
            "  fields = [field_frame0, field_frame1, ...]\n"
            "  # Multi-field: list of lists/tuples\n"
            "  fields = [[seq1_frame0, seq1_frame1], [seq2_frame0, seq2_frame1]]"
        )


def _add_speed_control_widget(
    viewer: napari.Viewer,
    initial_fps: int = DEFAULT_FPS,
    frame_labels: list[str] | None = None,
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
    # Set slider max to at least initial_fps (but minimum FPS_SLIDER_DEFAULT_MAX)
    slider_max = max(FPS_SLIDER_DEFAULT_MAX, initial_fps)

    # Throttle widget updates for high FPS (WIDGET_UPDATE_TARGET_HZ max to avoid Qt overhead)
    # At 250 FPS, updating 250x/sec causes stalling; throttle to target Hz
    update_interval = (
        max(1, initial_fps // WIDGET_UPDATE_TARGET_HZ)
        if initial_fps >= WIDGET_UPDATE_TARGET_HZ
        else 1
    )

    @magicgui(
        auto_call=True,
        play={"widget_type": "PushButton", "text": "▶ Play"},
        fps={
            "widget_type": "Slider",
            "min": FPS_SLIDER_MIN,
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
    def toggle_playback(event: Any | None = None) -> None:
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
    def update_frame_info(event: Any | None = None) -> None:
        """Update frame counter and label display (throttled for high FPS during playback).

        Throttling only applies during playback to prevent Qt overhead at high FPS.
        When scrubbing (not playing), updates are immediate for responsive feedback.
        """
        try:
            # Get current frame from first dimension (time)
            current_frame = viewer.dims.current_step[0] if viewer.dims.ndim > 0 else 0

            # Throttle updates ONLY during playback; always update when scrubbing
            # This ensures responsive feedback during manual navigation
            # For high FPS (e.g., 250 FPS), throttling prevents Qt stalling
            if (
                playback_state["is_playing"]
                and current_frame % update_interval != 0
                and current_frame != playback_state["last_frame"]
            ):
                # Skip this update (not at interval boundary and frame changed)
                playback_state["last_frame"] = current_frame
                return

            playback_state["last_frame"] = current_frame

            # Note: We track our own playback state via playback_state["is_playing"]
            # instead of syncing with napari's internal state. This avoids using
            # the deprecated qt_viewer API. The button state is updated when the
            # user clicks our play/pause button (see play_callback below).

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
    def spacebar_toggle(viewer_instance: napari.Viewer) -> None:
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
    fps: int = DEFAULT_FPS,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    frame_labels: list[str] | None = None,
    title: str = "Spatial Field Animation",
    cache_size: int = DEFAULT_CACHE_SIZE,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    max_chunks: int = DEFAULT_MAX_CHUNKS,
    layout: Literal["horizontal", "vertical", "grid"] | None = None,
    layer_names: list[str] | None = None,
    overlay_data: OverlayData | None = None,
    show_regions: bool | list[str] = False,
    region_alpha: float = 0.3,
    **kwargs: Any,
) -> napari.Viewer:
    """Launch Napari viewer with lazy-loaded field animation.

    Napari provides GPU-accelerated rendering with on-demand frame loading,
    making it ideal for large datasets (100K+ frames). Frames are cached
    with a true LRU (Least Recently Used) eviction policy for efficient
    memory management.

    Parameters
    ----------
    env : Environment
        Environment defining spatial structure.
    fields : list of ndarray of shape (n_bins,) or list of list of ndarray of shape (n_bins,)
        Field data to animate, dtype float64. Two modes:
        - Single-field mode: List of arrays, each with shape (n_bins,)
        - Multi-field mode: List of field sequences, where each sequence
          is a list of 1D arrays. Automatically detected based on input structure.
          All sequences must have the same length (same number of frames).
    fps : int, default=30
        Frames per second for playback (Napari slider speed).
    cmap : str, default="viridis"
        Matplotlib colormap name (e.g., "viridis", "hot", "plasma").
    vmin : float, optional
        Minimum value for color scale normalization. If None, computed from
        all fields using NaN-robust min.
    vmax : float, optional
        Maximum value for color scale normalization. If None, computed from
        all fields using NaN-robust max.
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
        Multi-field mode only: Layout arrangement for multiple field sequences
        using napari's built-in grid mode. Required when providing multiple
        field sequences. Options:
        - "horizontal": side-by-side in a single row (grid.shape = (1, n))
        - "vertical": stacked top-to-bottom in a single column (grid.shape = (n, 1))
        - "grid": automatic NxM grid layout (grid.shape = (-1, -1))
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
        Alpha transparency for region overlays, range [0.0, 1.0] where 0.0 is
        fully transparent and 1.0 is fully opaque. Only applies when show_regions
        is True or a non-empty list.
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
    # Warn about unknown kwargs for easier debugging
    if kwargs:
        unknown_keys = ", ".join(sorted(kwargs.keys()))
        warnings.warn(
            f"render_napari received unknown keyword arguments that will be ignored: "
            f"{unknown_keys}. These may be parameters intended for other backends.",
            UserWarning,
            stacklevel=2,
        )

    if not NAPARI_AVAILABLE:
        raise ImportError(
            "Napari backend requires napari. Install with:\n"
            "  pip install napari[all]\n"
            "or\n"
            "  uv add napari[all]"
        )

    # Validate field types are consistent (all arrays or all lists)
    _validate_field_types_consistent(fields)

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
        def toggle_playback(viewer: napari.Viewer) -> None:
            """Toggle animation playback with spacebar."""
            viewer.window._toggle_play()

    # Render overlay data if provided
    if overlay_data is not None:
        # Render video overlays (z_order="below" first, then field layer exists, then "above")
        n_frames = len(fields)
        video_layers: list[Layer] = []
        for idx, video_data in enumerate(overlay_data.videos):
            suffix = f" {idx + 1}" if len(overlay_data.videos) > 1 else ""
            name = f"Video{suffix}"
            if video_data.z_order == "below":
                # Add video layer below field - it will appear under the field
                # We need to reorder layers after adding
                video_layer = _add_video_layer(
                    viewer, video_data, env, n_frames, name=name
                )
                # Move video layer to bottom of stack (index 0)
                viewer.layers.move(viewer.layers.index(video_layer), 0)
                video_layers.append(video_layer)
            else:
                # Add video layer above field (default position - on top)
                video_layer = _add_video_layer(
                    viewer, video_data, env, n_frames, name=name
                )
                video_layers.append(video_layer)

        # Register callback to update video frames when animation frame changes
        _make_video_frame_callback(viewer, video_layers)

        # Render position overlays (tracks + points)
        for idx, pos_data in enumerate(overlay_data.positions):
            suffix = f" {idx + 1}" if len(overlay_data.positions) > 1 else ""
            _render_position_overlay(viewer, pos_data, env, name_suffix=suffix)

        # Render bodypart overlays (points + skeleton)
        for idx, bodypart_data in enumerate(overlay_data.bodypart_sets):
            suffix = f" {idx + 1}" if len(overlay_data.bodypart_sets) > 1 else ""
            _render_bodypart_overlay(viewer, bodypart_data, env, name_suffix=suffix)

        # Render head direction overlays (vectors)
        # Auto-pair with position overlay when there's exactly one position
        paired_position = (
            overlay_data.positions[0] if len(overlay_data.positions) == 1 else None
        )
        for idx, head_dir_data in enumerate(overlay_data.head_directions):
            suffix = f" {idx + 1}" if len(overlay_data.head_directions) > 1 else ""
            _render_head_direction_overlay(
                viewer,
                head_dir_data,
                env,
                name_suffix=suffix,
                position_data=paired_position,
            )

    # Render regions if requested
    if show_regions:
        _render_regions(viewer, env, show_regions, region_alpha)

    return viewer


def _render_multi_field_napari(
    env: Environment,
    field_sequences: list[list[NDArray[np.float64]]],
    *,
    fps: int = DEFAULT_FPS,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    frame_labels: list[str] | None = None,
    title: str = "Multi-Field Animation",
    cache_size: int = DEFAULT_CACHE_SIZE,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    max_chunks: int = DEFAULT_MAX_CHUNKS,
    layout: Literal["horizontal", "vertical", "grid"] | None = None,
    layer_names: list[str] | None = None,
    overlay_data: OverlayData | None = None,
    show_regions: bool | list[str] = False,
    region_alpha: float = 0.3,
) -> napari.Viewer:
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
        Layout arrangement for multiple fields using napari's grid mode:
        - "horizontal": side-by-side in a single row (grid.shape = (1, n))
        - "vertical": stacked top-to-bottom in a single column (grid.shape = (n, 1))
        - "grid": automatic NxM grid arrangement (grid.shape = (-1, -1))
        Grid mode is automatically enabled when multiple sequences are provided.
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
            "WHAT: Multi-field input requires 'layout' parameter.\n"
            "  Current: layout=None\n"
            "  Available: 'horizontal', 'vertical', 'grid'\n\n"
            "WHY: Multiple field sequences must be arranged spatially for comparison.\n\n"
            "HOW: Specify layout when animating multiple fields:\n"
            "  env.animate_fields(fields, layout='horizontal')  # Side-by-side\n"
            "  env.animate_fields(fields, layout='vertical')    # Stacked\n"
            "  env.animate_fields(fields, layout='grid')        # 2D grid"
        )

    n_sequences = len(field_sequences)

    # Validate all sequences have same length
    sequence_lengths = [len(seq) for seq in field_sequences]
    if len(set(sequence_lengths)) > 1:
        raise ValueError(
            f"WHAT: All field sequences must have the same length.\n"
            f"  Got lengths: {sequence_lengths}\n"
            f"  Expected: {sequence_lengths[0]} frames (from first sequence)\n\n"
            f"WHY: Animation requires synchronized frames across all fields.\n\n"
            f"HOW: Ensure all field sequences have the same number of frames:\n"
            f"  # Truncate to shortest sequence\n"
            f"  min_len = min(len(seq) for seq in field_sequences)\n"
            f"  fields = [[seq[:min_len] for seq in field_sequences]]\n\n"
            f"  # Or pad shorter sequences\n"
            f"  max_len = max(len(seq) for seq in field_sequences)\n"
            f"  fields = [[list(seq) + [seq[-1]] * (max_len - len(seq)) "
            f"for seq in field_sequences]]"
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
    for renderer, name in zip(lazy_renderers, layer_names, strict=True):
        viewer.add_image(
            renderer,
            name=name,
            rgb=True,
        )

    # Configure grid mode based on layout parameter
    # Napari's grid mode arranges layers in a 2D grid for side-by-side viewing
    if n_sequences > 1:
        viewer.grid.enabled = True
        if layout == "horizontal":
            # Side-by-side: single row with n_sequences columns
            viewer.grid.shape = (1, n_sequences)
            viewer.grid.stride = 1
        elif layout == "vertical":
            # Stacked: n_sequences rows with single column
            viewer.grid.shape = (n_sequences, 1)
            # stride=-1 reverses layer order so first layer appears at top
            viewer.grid.stride = -1
        elif layout == "grid":
            # Auto-grid: napari determines optimal arrangement
            # Use -1 for automatic sizing based on number of layers
            viewer.grid.shape = (-1, -1)
            viewer.grid.stride = 1

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
        def toggle_playback(viewer: napari.Viewer) -> None:
            """Toggle animation playback with spacebar."""
            viewer.window._toggle_play()

    # Render overlay data if provided
    if overlay_data is not None:
        # Render video overlays (z_order="below" first, then field layer exists, then "above")
        n_frames = len(field_sequences[0]) if field_sequences else 0
        video_layers: list[Layer] = []
        for idx, video_data in enumerate(overlay_data.videos):
            suffix = f" {idx + 1}" if len(overlay_data.videos) > 1 else ""
            name = f"Video{suffix}"
            if video_data.z_order == "below":
                # Add video layer below field - it will appear under the field
                video_layer = _add_video_layer(
                    viewer, video_data, env, n_frames, name=name
                )
                # Move video layer to bottom of stack (index 0)
                viewer.layers.move(viewer.layers.index(video_layer), 0)
                video_layers.append(video_layer)
            else:
                # Add video layer above field (default position - on top)
                video_layer = _add_video_layer(
                    viewer, video_data, env, n_frames, name=name
                )
                video_layers.append(video_layer)

        # Register callback to update video frames when animation frame changes
        _make_video_frame_callback(viewer, video_layers)

        # Render position overlays (tracks + points)
        for idx, pos_data in enumerate(overlay_data.positions):
            suffix = f" {idx + 1}" if len(overlay_data.positions) > 1 else ""
            _render_position_overlay(viewer, pos_data, env, name_suffix=suffix)

        # Render bodypart overlays (points + skeleton)
        for idx, bodypart_data in enumerate(overlay_data.bodypart_sets):
            suffix = f" {idx + 1}" if len(overlay_data.bodypart_sets) > 1 else ""
            _render_bodypart_overlay(viewer, bodypart_data, env, name_suffix=suffix)

        # Render head direction overlays (vectors)
        # Auto-pair with position overlay when there's exactly one position
        paired_position = (
            overlay_data.positions[0] if len(overlay_data.positions) == 1 else None
        )
        for idx, head_dir_data in enumerate(overlay_data.head_directions):
            suffix = f" {idx + 1}" if len(overlay_data.head_directions) > 1 else ""
            _render_head_direction_overlay(
                viewer,
                head_dir_data,
                env,
                name_suffix=suffix,
                position_data=paired_position,
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
    cache_size: int = DEFAULT_CACHE_SIZE,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    max_chunks: int = DEFAULT_MAX_CHUNKS,
) -> LazyFieldRenderer | ChunkedLazyFieldRenderer:
    """Create lazy field renderer for Napari.

    Automatically selects between per-frame and chunked caching based on
    dataset size. Uses per-frame caching for datasets ≤CHUNKED_CACHE_THRESHOLD
    frames and chunked caching for larger datasets.

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
    cache_size : int, default=DEFAULT_CACHE_SIZE
        Maximum number of frames to cache for per-frame caching strategy
    chunk_size : int, default=DEFAULT_CHUNK_SIZE
        Number of frames per chunk for chunked caching strategy
    max_chunks : int, default=DEFAULT_MAX_CHUNKS
        Maximum number of chunks to cache for chunked caching strategy

    Returns
    -------
    renderer : LazyFieldRenderer or ChunkedLazyFieldRenderer
        Lazy renderer instance implementing array-like interface
        for Napari
    """
    n_frames = len(fields)

    # Auto-select caching strategy based on dataset size
    if n_frames <= CHUNKED_CACHE_THRESHOLD:
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
        Environment defining spatial structure.
    fields : list of ndarray of shape (n_bins,), dtype float64
        All fields to animate. Each array contains field values for one frame.
    cmap_lookup : ndarray of shape (256, 3), dtype uint8
        Pre-computed colormap RGB lookup table with values in range [0, 255].
    vmin : float
        Minimum value for color scale normalization.
    vmax : float
        Maximum value for color scale normalization.
    cache_size : int, default=DEFAULT_CACHE_SIZE
        Maximum number of frames to cache.

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
    The default cache size (DEFAULT_CACHE_SIZE) is chosen to balance memory
    usage and performance. For typical 100x100 grids with RGB data:
    - Memory per frame: ~30KB
    - Cache memory: ~30MB
    - Sufficient for responsive scrubbing

    For larger grids or tighter memory constraints, reduce cache_size.

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

    # Prevent numpy from coercing this array-like into ndarray eagerly
    __array_priority__ = 1000

    def __init__(
        self,
        env: Environment,
        fields: list[NDArray[np.float64]],
        cmap_lookup: NDArray[np.uint8],
        vmin: float,
        vmax: float,
        cache_size: int = DEFAULT_CACHE_SIZE,
    ) -> None:
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
        with perf_timer("LazyFieldRenderer_getitem"), self._lock:
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
            add_instant_event("frame_cache_miss")
            # Render this frame
            from neurospatial.animation.rendering import field_to_rgb_for_napari

            with perf_timer("field_to_rgb_render"):
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
            add_instant_event("frame_cache_hit")
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
    def dtype(self) -> np.dtype[np.uint8]:
        """Return dtype (always uint8 for RGB)."""
        return np.dtype(np.uint8)

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
        Environment defining spatial structure.
    fields : list of ndarray of shape (n_bins,), dtype float64
        All fields to animate. Each array contains field values for one frame.
    cmap_lookup : ndarray of shape (256, 3), dtype uint8
        Pre-computed colormap RGB lookup table with values in range [0, 255].
    vmin : float
        Minimum value for color scale normalization.
    vmax : float
        Maximum value for color scale normalization.
    chunk_size : int, default=DEFAULT_CHUNK_SIZE
        Number of frames per chunk.
    max_chunks : int, default=DEFAULT_MAX_CHUNKS
        Maximum number of chunks to cache (LRU eviction).

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
    - Memory per chunk (DEFAULT_CHUNK_SIZE frames): ~300KB
    - Cache memory (DEFAULT_MAX_CHUNKS chunks): ~30MB

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
            env, fields, cmap_lookup, 0.0, 1.0, chunk_size=10, max_chunks=100
        )
        len(renderer)  # Number of frames
        # 100000
        frame = renderer[0]  # Loads chunk 0 (frames 0-9)
        frame = renderer[5]  # Retrieved from chunk 0 cache (instant)
        frame = renderer[10]  # Loads chunk 1 (frames 10-19)
    """

    # Prevent numpy from coercing this array-like into ndarray eagerly
    __array_priority__ = 1000

    def __init__(
        self,
        env: Environment,
        fields: list[NDArray[np.float64]],
        cmap_lookup: NDArray[np.uint8],
        vmin: float,
        vmax: float,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        max_chunks: int = DEFAULT_MAX_CHUNKS,
    ) -> None:
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
    def dtype(self) -> np.dtype[np.uint8]:
        """Return dtype (always uint8 for RGB)."""
        return np.dtype(np.uint8)

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
