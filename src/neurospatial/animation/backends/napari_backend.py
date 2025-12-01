"""Napari GPU-accelerated viewer backend."""

from __future__ import annotations

import contextlib
import os
import warnings
from collections import OrderedDict
from collections.abc import Callable
from threading import Lock
from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

# Import shared coordinate transforms
from neurospatial.animation.core import MAX_PLAYBACK_FPS
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
        EventData,
        HeadDirectionData,
        OverlayData,
        PositionData,
        TimeSeriesData,
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

WIDGET_UPDATE_TARGET_HZ: int = 10
"""Target update rate for playback widget to avoid Qt overhead at high FPS.

This should be low enough that users can read the frame counter (10 Hz = 100ms
between updates is easily readable), and high enough to feel responsive.
Setting this to 10 Hz instead of 30 Hz significantly reduces Qt overhead from
label updates during playback.
"""

# Per-viewer warning state metadata key
_TRANSFORM_WARNED_KEY: str = "_transform_fallback_warned"
"""Metadata key for tracking transform fallback warning per viewer."""

# Playback controller metadata key
_PLAYBACK_CONTROLLER_KEY: str = "playback_controller"
"""Metadata key for storing PlaybackController in viewer."""


# =============================================================================
# PlaybackController - Central Playback Control
# =============================================================================


class PlaybackController:
    """Central playback controller for frame-skipping-aware animation.

    Centralizes playback control to enable frame skipping when the viewer
    falls behind schedule. This is essential for maintaining smooth playback
    with multiple overlays where frame rendering may exceed the target fps.

    Parameters
    ----------
    viewer : napari.Viewer
        Napari viewer instance to control.
    n_frames : int
        Total number of frames in the animation.
    fps : float
        Target frames per second for playback.
    frame_times : ndarray of shape (n_frames,), optional
        Timestamps for each frame in seconds. If None, frames are assumed
        uniformly spaced at 1/fps intervals.
    allow_frame_skip : bool, default=True
        If True, skip frames when falling behind schedule to maintain
        real-time playback. If False, always advance by exactly 1 frame.

    Attributes
    ----------
    viewer : napari.Viewer
        The napari viewer being controlled.
    n_frames : int
        Total number of frames.
    fps : float
        Target playback rate.
    frame_times : ndarray or None
        Frame timestamps.
    allow_frame_skip : bool
        Whether frame skipping is enabled.
    current_frame : int
        Current frame index (0-based).
    is_playing : bool
        Whether playback is active.
    frames_rendered : int
        Total frames rendered since creation.
    frames_skipped : int
        Total frames skipped since creation.

    Examples
    --------
    >>> import napari
    >>> viewer = napari.Viewer()
    >>> controller = PlaybackController(viewer, n_frames=100, fps=30.0)
    >>> controller.go_to_frame(50)  # Jump to frame 50
    >>> controller.play()  # Start playback
    >>> # ... playback runs via QTimer ...
    >>> controller.pause()  # Stop playback

    Notes
    -----
    The controller uses elapsed time to calculate target frames, enabling
    automatic frame skipping when rendering falls behind. This prevents
    the "slow-motion" effect that occurs when every frame must be rendered.

    When ``allow_frame_skip=True`` (default):
    - ``step()`` calculates target frame from elapsed time
    - If behind schedule, skips directly to target frame
    - Frames between current and target are counted as skipped

    When ``allow_frame_skip=False``:
    - ``step()`` always advances by exactly 1 frame
    - May result in slow-motion playback if rendering is slow
    - Useful for frame-accurate playback (e.g., video export)

    See Also
    --------
    render_napari : Main rendering function that creates PlaybackController
    """

    import time

    def __init__(
        self,
        viewer: napari.Viewer,
        n_frames: int,
        fps: float,
        frame_times: NDArray[np.float64] | None = None,
        allow_frame_skip: bool = True,
    ) -> None:
        """Initialize PlaybackController."""
        self.viewer = viewer
        self.n_frames = n_frames
        self.fps = fps
        self.frame_times = frame_times
        self.allow_frame_skip = allow_frame_skip

        # Internal state
        self._current_frame: int = 0
        self._playing: bool = False
        self._start_time: float | None = None
        self._start_frame: int = 0
        self._callbacks: list[Callable[[int], None]] = []

        # Metrics
        self._frames_rendered: int = 0
        self._frames_skipped: int = 0

    @property
    def current_frame(self) -> int:
        """Current frame index (0-based)."""
        return self._current_frame

    @property
    def is_playing(self) -> bool:
        """Whether playback is currently active."""
        return self._playing

    @property
    def frames_rendered(self) -> int:
        """Total number of frames rendered since creation."""
        return self._frames_rendered

    @property
    def frames_skipped(self) -> int:
        """Total number of frames skipped since creation."""
        return self._frames_skipped

    def go_to_frame(self, frame_idx: int) -> None:
        """Jump to a specific frame.

        Clamps frame index to valid range [0, n_frames-1], updates the
        viewer dims, and notifies all registered callbacks.

        Parameters
        ----------
        frame_idx : int
            Target frame index (0-based). Will be clamped to valid range.
        """

        # Clamp to valid range
        old_frame = self._current_frame
        self._current_frame = max(0, min(frame_idx, self.n_frames - 1))

        # Update viewer
        self.viewer.dims.set_current_step(0, self._current_frame)

        # Track metrics
        self._frames_rendered += 1
        if self._current_frame > old_frame + 1:
            # Skipped frames (jumped more than 1)
            self._frames_skipped += self._current_frame - old_frame - 1

        # Notify callbacks
        for callback in self._callbacks:
            callback(self._current_frame)

    def step(self) -> None:
        """Advance to the next frame, with optional skipping if behind schedule.

        When playing, calculates the target frame based on elapsed time since
        ``play()`` was called. If ``allow_frame_skip`` is True and the target
        frame is ahead of current, skips directly to target. Otherwise,
        advances by exactly 1 frame.

        Does nothing if not currently playing.
        """
        import time

        if not self._playing or self._start_time is None:
            return

        if self.allow_frame_skip:
            # Calculate target frame based on elapsed time
            elapsed = time.perf_counter() - self._start_time
            target_frame = self._start_frame + int(elapsed * self.fps)

            # Skip to target if behind schedule
            if target_frame > self._current_frame:
                self.go_to_frame(target_frame)
        else:
            # No skipping - advance by exactly 1 frame
            self.go_to_frame(self._current_frame + 1)

        # Check for end of animation
        if self._current_frame >= self.n_frames - 1:
            self.pause()

    def play(self) -> None:
        """Start playback.

        Records the current time and frame for elapsed-time-based
        frame skipping calculations in ``step()``.
        """
        import time

        self._playing = True
        self._start_time = time.perf_counter()
        self._start_frame = self._current_frame

    def pause(self) -> None:
        """Pause playback."""
        self._playing = False

    def register_callback(self, callback: Callable[[int], None]) -> None:
        """Register a callback to be notified on frame changes.

        Parameters
        ----------
        callback : Callable[[int], None]
            Function that accepts a single int argument (the new frame index).
            Called whenever ``go_to_frame()`` is invoked.

        Examples
        --------
        >>> def on_frame_change(frame_idx):
        ...     print(f"Now at frame {frame_idx}")
        >>> controller.register_callback(on_frame_change)
        """
        self._callbacks.append(callback)


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
) -> tuple[Layer, bool]:
    """Add video overlay as an Image layer.

    Creates a napari Image layer that displays video frames synchronized with
    the animation. Supports affine transforms to position the video in
    environment coordinates.

    For in-memory video arrays, uses napari's native time dimension for
    efficient playback without per-frame callbacks. For file-based video
    readers, uses the traditional callback approach for streaming.

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
    uses_native_time : bool
        True if the layer uses napari's native time dimension (4D array)
        and does NOT require a frame update callback.
        False if the layer is 3D (single frame) and REQUIRES a callback
        to update ``layer.data`` on each frame change during playback.

    Notes
    -----
    **In-Memory Video Optimization (Phase 2.1)**:

    When the video source is an in-memory np.ndarray, we create a 4D Image
    layer with shape (n_animation_frames, height, width, 3). This allows
    napari to handle frame selection natively via dims[0], eliminating
    per-frame ``layer.data = frame`` overhead (~2-3ms saved per frame).

    The frame_indices mapping is pre-applied by reordering the video array
    to match animation frame order. Missing frames (frame_indices == -1)
    are filled with black/zeros.

    **File-Based Video (Streaming)**:

    When the video source is a VideoReaderProtocol (file-based), we create
    a 3D single-frame layer and use the callback approach for streaming.
    This is necessary because loading all frames into memory may not be
    feasible for large video files.

    **Z-ordering**:

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
    """
    # Build affine transform matrix
    # Chains: video_px → env_cm → napari_px (see COORDINATES.md for details)
    affine = _build_video_napari_affine(video_data, env)

    # Check if video source is in-memory array
    if isinstance(video_data.reader, np.ndarray):
        # In-memory optimization: create 4D array with time dimension
        # Reorder video frames according to frame_indices
        video_array: NDArray[np.uint8] = video_data.reader
        n_anim_frames = len(video_data.frame_indices)
        n_video_frames = len(video_array)
        h, w = video_array.shape[1:3]

        # Pre-allocate 4D array for all animation frames (zeros = black for missing)
        time_indexed_array = np.zeros((n_anim_frames, h, w, 3), dtype=np.uint8)

        # Use vectorized assignment for valid frame indices (faster than loop)
        frame_indices = video_data.frame_indices
        valid_mask = (frame_indices >= 0) & (frame_indices < n_video_frames)
        valid_anim_indices = np.where(valid_mask)[0]
        valid_video_indices = frame_indices[valid_mask]
        time_indexed_array[valid_anim_indices] = video_array[valid_video_indices]

        # Add 4D image layer - napari handles time dimension natively
        layer = viewer.add_image(
            time_indexed_array,
            name=name,
            rgb=True,
            opacity=video_data.alpha,
            affine=affine,
            blending="translucent",
        )

        # Store metadata (for compatibility, though not needed for callbacks)
        layer.metadata["video_data"] = video_data
        layer.metadata["video_frame_indices"] = video_data.frame_indices
        layer.metadata["uses_native_time"] = True

        return layer, True

    else:
        # File-based video: use single-frame layer with callback approach
        # Get initial frame
        initial_frame_idx = (
            video_data.frame_indices[0] if len(video_data.frame_indices) > 0 else 0
        )
        if initial_frame_idx >= 0:
            initial_frame = video_data.reader[initial_frame_idx]
        else:
            # No valid frame - create blank (estimate size from reader)
            try:
                first_frame = video_data.reader[0]
                h, w = first_frame.shape[:2]
            except (IndexError, KeyError):
                h, w = 64, 64  # fallback
            initial_frame = np.zeros((h, w, 3), dtype=np.uint8)

        # Add 3D image layer (single frame)
        layer = viewer.add_image(
            initial_frame,
            name=name,
            rgb=True,
            opacity=video_data.alpha,
            affine=affine,
            blending="translucent",
        )

        # Store video data for frame updates (needed for callback)
        layer.metadata["video_data"] = video_data
        layer.metadata["video_frame_indices"] = video_data.frame_indices
        layer.metadata["uses_native_time"] = False

        return layer, False


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
    scale: tuple[float, float] | None = None,
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
    scale : tuple of (y_scale, x_scale), optional
        Physical units scale to match the Image layer. If provided, overlay
        coordinates will be scaled to match the field visualization.

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

        # Filter out NaN values - napari Tracks layer requires finite data
        valid_mask = np.all(np.isfinite(track_data), axis=1)
        if not np.all(valid_mask):
            track_data = track_data[valid_mask]

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

        # Features must match filtered track data length
        n_valid = len(track_data)
        features = {"color": np.zeros(n_valid)}
        custom_colormap = Colormap(
            colors=[position_data.color, position_data.color],
            name=f"trail_color{name_suffix}",
        )
        colormaps_dict = {"color": custom_colormap}

        # Note: We set color_by AFTER layer creation to avoid napari warning.
        # During __init__, napari's data setter resets features to {} before our
        # features are applied. If color_by is passed at init time, the check runs
        # against empty features and warns. Setting color_by after creation avoids this.
        if n_valid > 0:
            # Tracks data format is [track_id, time, y, x] but track_id is an identifier,
            # not a dimension. Layer ndim is 3 (time, y, x), so scale is (time, y, x)
            tracks_scale = (1.0, scale[0], scale[1]) if scale else None
            layer = viewer.add_tracks(
                track_data,
                name=f"Position Trail{name_suffix}",
                tail_length=position_data.trail_length,
                features=features,
                colormaps_dict=colormaps_dict,
                scale=tracks_scale,
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

    # Filter out NaN values for points layer as well
    valid_points_mask = np.all(np.isfinite(points_data), axis=1)
    if not np.all(valid_points_mask):
        points_data = points_data[valid_points_mask]

    if len(points_data) > 0:
        # Points data is (time, y, x) - scale dimensions 1, 2 (y, x)
        points_scale = (1.0, scale[0], scale[1]) if scale else None
        layer = viewer.add_points(
            points_data,
            name=f"Position{name_suffix}",
            size=position_data.size,
            face_color=position_data.color,
            border_color="white",
            border_width=POINT_BORDER_WIDTH,
            scale=points_scale,
        )
        layers.append(layer)

    return layers


def _render_bodypart_overlay(
    viewer: napari.Viewer,
    bodypart_data: BodypartData,
    env: Environment,
    name_suffix: str = "",
    scale: tuple[float, float] | None = None,
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

    # Points data is (time, y, x) - scale dimensions 1, 2 (y, x)
    points_scale = (1.0, scale[0], scale[1]) if scale else None
    layer = viewer.add_points(
        points_array,
        name=f"Bodyparts{name_suffix}",
        size=BODYPART_POINT_SIZE,
        face_color=face_colors,
        border_color="black",
        border_width=POINT_BORDER_WIDTH,
        features={"bodypart": all_features},
        scale=points_scale,
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
            # Vectors data is (time, y, x, dy, dx) - scale dimensions 1, 2 (y, x)
            vectors_scale = (1.0, scale[0], scale[1]) if scale else None
            skeleton_layer = viewer.add_vectors(
                vectors_data,
                name=f"Skeleton{name_suffix}",
                edge_color=skeleton.edge_color,
                edge_width=skeleton.edge_width,
                vector_style="line",  # Hide arrow heads for skeleton lines
                features=vector_features,
                scale=vectors_scale,
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
    scale: tuple[float, float] | None = None,
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
    scale : tuple of float, optional
        Scale factors (y_scale, x_scale) to map pixel coordinates to physical
        coordinates. If provided, applied to the layer for alignment with
        the field image layer.

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
        # Tracks data format is [track_id, time, y, x] but track_id is an identifier,
        # not a dimension. Layer ndim is 3 (time, y, x), so scale is (time, y, x)
        tracks_scale = (1.0, scale[0], scale[1]) if scale else None
        layer = viewer.add_tracks(
            tracks,
            name=f"Head Direction{name_suffix}",
            tail_length=0,  # No past tracks
            head_length=1,  # Show current track (origin→indicator)
            tail_width=head_dir_data.width,
            blending="opaque",  # Better visibility against dark backgrounds
            scale=tracks_scale,
        )
        layer.colormap = head_dir_data.color
        layers.append(layer)

    return layers


def _render_event_overlay(
    viewer: napari.Viewer,
    event_data: EventData,
    env: Environment,
    name_suffix: str = "",
    scale: tuple[float, float] | None = None,
) -> list[Layer]:
    """Render event overlay using Points layer with dynamic visibility.

    Supports three rendering modes based on decay_frames:
    - **Cumulative mode** (decay_frames=None): Events appear and stay permanently.
      All events up to current frame are visible (spikes accumulate over time).
    - **Instant mode** (decay_frames=0): Events appear only on their exact frame.
      Uses napari's native 3D points format for optimal performance.
    - **Decay mode** (decay_frames > 0): Events visible for N frames then hidden.

    Parameters
    ----------
    viewer : napari.Viewer
        Napari viewer instance.
    event_data : EventData
        Event overlay data aligned to animation frames. Contains:
        - event_positions: Dict mapping event type to (n_events, n_dims) arrays
        - event_frame_indices: Dict mapping event type to frame indices
        - colors, size, decay_frames, border_color, border_width
        - markers: Marker styles (not used - napari renders all events as circles)
    env : Environment
        Environment instance for coordinate transformation.
    name_suffix : str
        Suffix for layer names (e.g., for multiple event overlays).
    scale : tuple of (y_scale, x_scale), optional
        Physical units scale to match the Image layer.

    Returns
    -------
    layers : list of Layer
        List of created napari layer objects.

    Notes
    -----
    **Instant mode optimization (decay_frames=0):**
    Uses napari's native 3D time-dimensional points format (time, y, x). This is
    much faster than callback-based visibility because napari handles time filtering
    internally without per-frame callback overhead.

    **Cumulative/Decay modes:**
    Uses a single Points layer with dynamic visibility via the ``shown`` mask,
    which is updated on each frame change via a callback. This approach handles
    millions of events efficiently (O(1) per frame for cumulative mode using
    searchsorted on pre-sorted indices).

    Performance characteristics:
    - Instant mode: No callback (native napari handling)
    - Cumulative mode: O(log N) per frame update (searchsorted)
    - Decay mode: O(N) per frame update (range comparison)
    """
    layers: list[Layer] = []

    # Collect ALL events across all event types into single arrays
    all_positions: list[NDArray[np.float64]] = []
    all_frame_indices: list[NDArray[np.int_]] = []
    all_colors: list[str] = []

    for event_type, positions in event_data.event_positions.items():
        frame_indices = event_data.event_frame_indices[event_type]
        color = event_data.colors.get(event_type, "#ffffff")

        if len(positions) == 0:
            continue

        # Transform coordinates to napari (y, x) with Y-axis inversion
        transformed = _transform_coords_for_napari(positions, env)

        all_positions.append(transformed)
        all_frame_indices.append(frame_indices)
        all_colors.extend([color] * len(positions))

    if not all_positions:
        return layers

    # Concatenate into single arrays
    positions_array = np.vstack(all_positions)  # (N_total, 2)
    frame_indices_array = np.concatenate(all_frame_indices)  # (N_total,)
    colors_array = np.array(all_colors)  # (N_total,) strings

    # Sort by frame index for efficient per-frame updates
    sort_idx = np.argsort(frame_indices_array)
    positions_array = positions_array[sort_idx]
    frame_indices_array = frame_indices_array[sort_idx]
    colors_array = colors_array[sort_idx]

    n_events = len(positions_array)

    # Convert string colors to RGBA arrays with user-specified opacity
    import matplotlib.colors as mcolors

    opacity = event_data.opacity
    face_colors = np.array([mcolors.to_rgba(c, alpha=opacity) for c in colors_array])

    # Border color: napari needs string or (N, 4) array, not a single RGBA tuple
    # Use string for simplicity - napari will broadcast it
    border_color = event_data.border_color

    # OPTIMIZATION: For instant mode (decay_frames=0), use napari's native 3D
    # time-dimensional points format instead of callback-based visibility.
    # This eliminates per-frame callback overhead entirely.
    if event_data.decay_frames == 0:
        # Create 3D points (time, y, x) - napari handles visibility natively
        points_3d = np.column_stack(
            [
                frame_indices_array,  # Time dimension
                positions_array[:, 0],  # Y
                positions_array[:, 1],  # X
            ]
        )

        # Scale for 3D points: (time_scale, y_scale, x_scale)
        points_scale_3d = (1.0, scale[0], scale[1]) if scale else None

        points_layer = viewer.add_points(
            points_3d,
            name=f"Events{name_suffix}",
            size=event_data.size,
            face_color=face_colors,
            border_color=border_color,
            border_width=event_data.border_width,
            scale=points_scale_3d,
        )
        layers.append(points_layer)
        return layers

    # For cumulative/decay modes, use callback-based visibility
    # Precompute per-frame index ranges for O(events_per_frame) updates
    # frame_starts[f] and frame_stops[f] give the slice of events for frame f
    if n_events > 0:
        max_frame = int(frame_indices_array[-1]) + 1
        # Count events per frame
        counts = np.bincount(frame_indices_array.astype(np.int64), minlength=max_frame)
        # Cumulative sum gives end indices
        frame_stops = np.cumsum(counts)
        # Start indices are shifted stops
        frame_starts = np.empty_like(frame_stops)
        frame_starts[0] = 0
        frame_starts[1:] = frame_stops[:-1]
    else:
        frame_starts = np.array([], dtype=np.int64)
        frame_stops = np.array([], dtype=np.int64)

    # Create Points layer with all events initially hidden via shown mask
    # Using shown mask is ~30x faster than alpha-based visibility (O(N) bool vs O(N*4) float)
    initial_shown = np.zeros(n_events, dtype=bool)

    # Event positions are 2D (y, x) - scale directly matches Image layer's (y, x) scale
    points_scale = scale  # Already (y_scale, x_scale) or None

    points_layer = viewer.add_points(
        positions_array,  # (N, 2) - just y, x (no time dimension)
        name=f"Events{name_suffix}",
        size=event_data.size,
        face_color=face_colors,
        border_color=border_color,
        border_width=event_data.border_width,
        shown=initial_shown,
        scale=points_scale,
    )

    # Store metadata for callback - includes precomputed per-frame ranges
    points_layer.metadata["event_frame_indices"] = frame_indices_array
    points_layer.metadata["event_decay_frames"] = event_data.decay_frames
    points_layer.metadata["event_n_events"] = n_events
    points_layer.metadata["event_frame_starts"] = frame_starts
    points_layer.metadata["event_frame_stops"] = frame_stops
    # Persistent shown mask (updated in-place by callback)
    points_layer.metadata["event_shown_mask"] = initial_shown
    # State tracking for incremental updates
    points_layer.metadata["event_last_frame"] = -1
    points_layer.metadata["event_last_cutoff_idx"] = 0

    # Register frame-change callback
    _register_event_visibility_callback(viewer, points_layer)

    layers.append(points_layer)
    return layers


def _register_event_visibility_callback(
    viewer: napari.Viewer,
    points_layer: napari.layers.Points,
) -> None:
    """Register callback to update event visibility on frame change.

    Parameters
    ----------
    viewer : napari.Viewer
        Napari viewer instance.
    points_layer : napari.layers.Points
        Points layer containing event markers with metadata:
        - event_frame_indices: Sorted array of frame indices for each event
        - event_decay_frames: None (cumulative), 0 (instant), or >0 (decay window)
        - event_n_events: Total number of events
        - event_frame_starts: Precomputed start indices for each frame
        - event_frame_stops: Precomputed stop indices for each frame
        - event_shown_mask: Persistent boolean mask (updated in-place)
        - event_last_frame: Last processed frame index
        - event_last_cutoff_idx: Last cutoff index (for cumulative mode)

    Notes
    -----
    Uses in-place updates to a persistent ``shown`` mask for O(events_per_frame)
    performance instead of O(N_total) per frame. The precomputed frame_starts
    and frame_stops arrays enable direct slicing without scanning all events.

    Visibility modes with optimized update strategies:

    - **Cumulative** (decay_frames=None): Only set newly visible events to True.
      Work is O(new_events) = O(events at current frame).
    - **Instant** (decay_frames=0): Clear previous frame's events, set current.
      Work is O(events_at_prev + events_at_current).
    - **Decay** (decay_frames>0): Update sliding window edges.
      Work is O(events_entering + events_leaving).
    """

    def on_frame_change(event: Any) -> None:
        with perf_timer("event_visibility_callback"):
            # Get current frame from dims slider
            current_frame = int(viewer.dims.current_step[0])

            # Retrieve metadata
            frame_indices = points_layer.metadata["event_frame_indices"]
            decay_frames = points_layer.metadata["event_decay_frames"]
            n_events = points_layer.metadata["event_n_events"]
            frame_starts = points_layer.metadata["event_frame_starts"]
            frame_stops = points_layer.metadata["event_frame_stops"]
            shown = points_layer.metadata["event_shown_mask"]
            last_frame = points_layer.metadata["event_last_frame"]
            last_cutoff_idx = points_layer.metadata["event_last_cutoff_idx"]

            # Handle edge cases
            if n_events == 0:
                return

            n_frames = len(frame_starts)
            mask_changed = False

            # Compute visibility mask based on mode using in-place updates
            if decay_frames is None:
                # Cumulative mode: all events up to current frame
                # Only set newly visible events (those between last_cutoff and new cutoff)
                # Use searchsorted for O(log N) cutoff lookup
                cutoff_idx = np.searchsorted(frame_indices, current_frame, side="right")
                if cutoff_idx > last_cutoff_idx:
                    # Moving forward: set new events to True
                    shown[last_cutoff_idx:cutoff_idx] = True
                    mask_changed = True
                elif cutoff_idx < last_cutoff_idx:
                    # Moving backward: set removed events to False
                    shown[cutoff_idx:last_cutoff_idx] = False
                    mask_changed = True
                # Update state
                points_layer.metadata["event_last_cutoff_idx"] = cutoff_idx

            elif decay_frames == 0:
                # Instant mode: only events on exact frame
                # Clear previous frame's events, set current frame's events
                if last_frame >= 0 and last_frame < n_frames:
                    prev_start = frame_starts[last_frame]
                    prev_stop = frame_stops[last_frame]
                    if prev_stop > prev_start:
                        shown[prev_start:prev_stop] = False
                        mask_changed = True

                if 0 <= current_frame < n_frames:
                    curr_start = frame_starts[current_frame]
                    curr_stop = frame_stops[current_frame]
                    if curr_stop > curr_start:
                        shown[curr_start:curr_stop] = True
                        mask_changed = True

            else:
                # Decay mode: events within window [current - decay, current]
                # Update sliding window: hide events leaving, show events entering
                start_frame = max(0, current_frame - decay_frames)
                prev_start_frame = (
                    max(0, last_frame - decay_frames) if last_frame >= 0 else 0
                )

                if last_frame < 0:
                    # First update: show entire window
                    if start_frame < n_frames:
                        window_start = frame_starts[start_frame]
                        window_stop = frame_stops[min(current_frame, n_frames - 1)]
                        if window_stop > window_start:
                            shown[window_start:window_stop] = True
                            mask_changed = True
                elif current_frame > last_frame:
                    # Moving forward: hide events leaving window, show new events
                    # Events leaving: frames in [prev_start_frame, start_frame)
                    for f in range(prev_start_frame, min(start_frame, n_frames)):
                        if f < n_frames:
                            shown[frame_starts[f] : frame_stops[f]] = False
                            mask_changed = True
                    # Events entering: frames in (last_frame, current_frame]
                    for f in range(last_frame + 1, min(current_frame + 1, n_frames)):
                        if f < n_frames:
                            shown[frame_starts[f] : frame_stops[f]] = True
                            mask_changed = True
                elif current_frame < last_frame:
                    # Moving backward: show events re-entering, hide events leaving
                    # Events leaving: frames in (current_frame, last_frame]
                    for f in range(current_frame + 1, min(last_frame + 1, n_frames)):
                        if f < n_frames:
                            shown[frame_starts[f] : frame_stops[f]] = False
                            mask_changed = True
                    # Events re-entering: frames in [start_frame, prev_start_frame)
                    for f in range(start_frame, min(prev_start_frame, n_frames)):
                        if f < n_frames:
                            shown[frame_starts[f] : frame_stops[f]] = True
                            mask_changed = True

            # Update last_frame state
            points_layer.metadata["event_last_frame"] = current_frame

            # Only update layer if mask changed (napari repaints on shown change)
            if mask_changed:
                with perf_timer("event_shown_update"), points_layer.events.blocker():
                    points_layer.shown = shown
                # Note: refresh() was removed for performance. Setting points_layer.shown
                # triggers napari's internal refresh via EventedList. Explicit refresh()
                # was causing redundant Qt/OpenGL updates during playback (~18ms overhead).
                # If visual glitches occur, uncomment: points_layer.refresh()

    # Connect to dims change event
    viewer.dims.events.current_step.connect(on_frame_change)

    # Trigger initial update
    on_frame_change(None)


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
    initial_fps: int | None = None,
    frame_labels: list[str] | None = None,
    *,
    initial_speed: float = 1.0,
    sample_rate_hz: float = 30.0,
    max_playback_fps: int = MAX_PLAYBACK_FPS,
) -> None:
    """Add enhanced playback control widget to napari viewer.

    Creates a comprehensive docked widget with:
    - Play/Pause button (large, prominent)
    - Speed slider (0.01-4.0× range, shows speed multiplier)
    - Frame counter ("Frame: 15 / 30")
    - Frame label (if provided: "Trial 15")
    - FPS info label ("≈ 12 fps")

    Inspired by nwb_data_viewer playback widget pattern.

    Parameters
    ----------
    viewer : napari.Viewer
        Napari viewer instance to add widget to
    initial_fps : int, optional
        **Deprecated.** Use ``initial_speed`` instead.
        If provided, overrides computed fps from speed/sample_rate_hz.
    frame_labels : list of str, optional
        Labels for each frame (e.g., ["Trial 1", "Trial 2", ...])
        If provided, displays current frame's label in widget
    initial_speed : float, default=1.0
        Initial playback speed multiplier:

        - 1.0: Real-time playback
        - 0.1: 10% speed (slow motion)
        - 2.0: 2× speed (fast forward)
    sample_rate_hz : float, default=30.0
        Data sample rate in Hz. Used to compute actual fps from speed.
        For example, 30 Hz data at speed=0.5 gives 15 fps playback.
    max_playback_fps : int, default=MAX_PLAYBACK_FPS (25)
        Maximum playback fps. Speed slider is capped so that
        ``sample_rate_hz * speed <= max_playback_fps``.

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

    # Compute fps from speed and sample_rate_hz (unless initial_fps is explicitly provided)
    if initial_fps is None:
        computed_fps = int(min(sample_rate_hz * initial_speed, max_playback_fps))
        computed_fps = max(FPS_SLIDER_MIN, computed_fps)  # Clamp to minimum
        fps_value: int = computed_fps
    else:
        fps_value = initial_fps

    # Track playback state for button updates
    playback_state = {"is_playing": False, "last_frame": -1}

    # Throttle widget updates to WIDGET_UPDATE_TARGET_HZ to avoid Qt overhead.
    # At 25 fps with 10 Hz target, this updates every 2-3 frames (readable speed).
    # The previous logic only throttled when fps >= target, which was backwards
    # and resulted in NO throttling at typical playback speeds (fps=25, target=30).
    update_interval = max(1, fps_value // WIDGET_UPDATE_TARGET_HZ)

    # Use simple FPS slider (like old working code) instead of speed slider.
    # The FloatSlider with tiny range (0.01-0.12 for 500Hz data) appears to cause
    # Qt event loop issues that stall napari playback. The FPS slider works reliably.
    slider_max = max(FPS_SLIDER_DEFAULT_MAX, fps_value)

    @magicgui(
        auto_call=True,
        play={"widget_type": "PushButton", "text": "▶ Play"},
        fps={
            "widget_type": "Slider",
            "min": FPS_SLIDER_MIN,
            "max": slider_max,
            "value": fps_value,
            "label": "Speed (FPS)",
        },
        frame_info={"widget_type": "Label", "label": ""},
    )
    def playback_widget(
        play: bool = False,
        fps: int = fps_value,
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


# =============================================================================
# Time Series Dock Widget
# =============================================================================

# Maximum update rate for time series plot (Hz)
# Prevents matplotlib from becoming bottleneck at high napari FPS.
# Lower values reduce update frequency but improve playback smoothness.
# Combined with draw-pending check, this prevents queue buildup.
# Reduced from 30 to 20 Hz to give more headroom for Qt event loop
# and reduce impact of matplotlib draw spikes (profiled at ~12-15ms per draw).
TIMESERIES_MAX_UPDATE_HZ: int = 20

# Time series figure DPI - lower values render faster
# 72 DPI is sufficient for screen display and reduces rendering workload
TIMESERIES_FIGURE_DPI: int = 72


def _add_timeseries_dock(
    viewer: napari.Viewer,
    timeseries_data: list[TimeSeriesData],
    frame_times: NDArray[np.float64],
) -> None:
    """Add time series dock widget to napari viewer.

    Creates a matplotlib figure embedded in a Qt widget, displaying time series
    data as scrolling plots in the right dock area. Updates automatically when
    the viewer's frame slider changes.

    Parameters
    ----------
    viewer : napari.Viewer
        Napari viewer instance to add widget to.
    timeseries_data : list[TimeSeriesData]
        Time series data containers from conversion pipeline.
    frame_times : NDArray[np.float64]
        Animation frame timestamps for synchronization.

    Notes
    -----
    **Update modes**: The dock widget respects the ``update_mode`` setting
    from TimeSeriesData. When multiple overlays have different modes, the
    most restrictive mode is used (priority: manual > on_pause > live).

    - ``"live"`` (default): Updates on every frame change, throttled to 20 Hz.
    - ``"on_pause"``: Only updates when PlaybackController is paused.
      If no controller is available, falls back to live behavior.
    - ``"manual"``: Never auto-updates. Updates only via explicit API call.

    **Throttling**: Widget updates are throttled to TIMESERIES_MAX_UPDATE_HZ
    (20 Hz) in all auto-update modes to prevent matplotlib from becoming
    a performance bottleneck when napari's FPS is higher.

    The matplotlib figure is styled to match napari's dark theme (background
    color #262930, white text/ticks).
    """
    if not timeseries_data:
        return  # No time series to display

    try:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        from qtpy.QtWidgets import QVBoxLayout, QWidget
    except ImportError:
        warnings.warn(
            "Time series dock widget requires Qt backend for matplotlib. "
            "Install with: pip install PyQt5 or pip install PySide2",
            UserWarning,
            stacklevel=2,
        )
        return

    from neurospatial.animation._timeseries import (
        TimeSeriesArtistManager,
        _group_timeseries,
    )

    # Compute figure size based on number of rows
    groups = _group_timeseries(timeseries_data)
    n_rows = len(groups)
    fig_height = max(1.5, 1.2 * n_rows)  # Minimum 1.5 inches, scale with rows

    # Create matplotlib figure ONCE with reduced DPI for faster rendering
    fig = Figure(figsize=(3.5, fig_height), dpi=TIMESERIES_FIGURE_DPI)

    # Create artist manager ONCE (handles all matplotlib setup)
    manager = TimeSeriesArtistManager.create(
        fig=fig,
        timeseries_data=timeseries_data,
        frame_times=frame_times,
        dark_theme=True,
    )

    # Create Qt canvas and widget
    canvas = FigureCanvasQTAgg(fig)
    widget = QWidget()
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(canvas)

    # Add dock widget to viewer's right area
    viewer.window.add_dock_widget(widget, name="Time Series", area="right")

    # Determine effective update_mode from overlays
    # Priority: manual > on_pause > live (use most restrictive if mixed)
    mode_priority = {"manual": 2, "on_pause": 1, "live": 0}
    effective_mode = "live"
    for ts in timeseries_data:
        if mode_priority.get(ts.update_mode, 0) > mode_priority.get(effective_mode, 0):
            effective_mode = ts.update_mode

    # Throttle updates to prevent matplotlib bottleneck at high FPS
    import time

    last_update_time = [0.0]
    min_update_interval = 1.0 / TIMESERIES_MAX_UPDATE_HZ

    # Track whether a draw is pending to prevent queue buildup
    # When draw_idle() is called faster than matplotlib can render,
    # redraws queue up and cause the UI to stall ("sticking" during playback).
    # By skipping updates when a draw is pending, we prevent this backlog.
    draw_pending = [False]

    # Blitting support: cache background for fast updates
    # When blitting is enabled, we only redraw the changed artists (lines, cursors)
    # instead of the full figure, which is significantly faster.
    blit_background: list[Any] = [None]
    use_blitting = [True]  # Will be disabled if blitting fails

    def on_draw_complete(event: Any) -> None:
        """Reset draw_pending flag and capture background for blitting."""
        draw_pending[0] = False
        # Capture background after full draw for subsequent blitting
        if use_blitting[0]:
            try:
                blit_background[0] = canvas.copy_from_bbox(fig.bbox)
            except Exception:
                # Blitting not supported, fall back to full redraws
                use_blitting[0] = False

    # Connect to matplotlib's draw_event to know when rendering is complete
    canvas.mpl_connect("draw_event", on_draw_complete)

    def on_frame_change(event: Any) -> None:
        """Update time series plot when frame changes.

        Throttled and skips updates if previous draw is pending.
        Uses blitting when available for faster updates.
        Respects update_mode setting:
        - 'live': Update on every frame change (throttled to 20 Hz)
        - 'on_pause': Only update when playback is paused
        - 'manual': Never auto-update
        """
        # Manual mode: never auto-update
        if effective_mode == "manual":
            return

        # on_pause mode: only update when playback is paused
        if effective_mode == "on_pause":
            # Try to get PlaybackController from viewer
            controller = getattr(viewer, "playback_controller", None)
            if controller is not None and controller.is_playing:
                return  # Skip updates during playback

        current_time = time.time()

        # Throttle updates during playback (but allow immediate scrubbing response)
        if current_time - last_update_time[0] < min_update_interval:
            return  # Skip update, too soon

        # Skip if previous draw hasn't completed (prevents queue buildup)
        if draw_pending[0]:
            return

        last_update_time[0] = current_time

        # Get current frame index
        frame_idx = viewer.dims.current_step[0] if viewer.dims.ndim > 0 else 0

        # Bounds check
        if frame_idx < 0 or frame_idx >= len(frame_times):
            return

        # Update all artists for this frame
        manager.update(frame_idx, timeseries_data)

        # Mark draw as pending before scheduling
        draw_pending[0] = True

        # Use blitting if available for faster updates
        if use_blitting[0] and blit_background[0] is not None:
            try:
                # Restore background
                canvas.restore_region(blit_background[0])
                # Redraw only the changed artists
                for line in manager.lines.values():
                    if line.axes is not None:
                        line.axes.draw_artist(line)
                for cursor in manager.cursors:
                    if cursor is not None and cursor.axes is not None:
                        cursor.axes.draw_artist(cursor)
                for text in manager.value_texts.values():
                    if text.axes is not None:
                        text.axes.draw_artist(text)
                # Blit the updated region
                canvas.blit(fig.bbox)
                # Manual flush to display immediately
                canvas.flush_events()
                draw_pending[0] = False  # Blitting is synchronous
                return
            except Exception:
                # Fall back to full redraw
                use_blitting[0] = False

        # Fallback: full redraw (draw_idle is non-blocking)
        canvas.draw_idle()

    # Connect to dims events
    viewer.dims.events.current_step.connect(on_frame_change)

    # Initial render at frame 0 (also captures blit background)
    if len(frame_times) > 0:
        manager.update(0, timeseries_data)
        canvas.draw()


def render_napari(
    env: Environment,
    fields: (
        list[NDArray[np.float64]]
        | list[list[NDArray[np.float64]]]
        | NDArray[np.float64]
    ),
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
    scale_bar: bool | Any = False,  # bool | ScaleBarConfig
    speed: float = 1.0,
    sample_rate_hz: float | None = None,
    max_playback_fps: int = MAX_PLAYBACK_FPS,
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
    fields : ndarray of shape (n_frames, n_bins), or list of ndarray of shape (n_bins,), or list of list
        Field data to animate, dtype float64. Three input formats:
        - **Array mode (recommended for large datasets)**: 2D array with shape
          (n_frames, n_bins). Efficient for memory-mapped arrays (memmaps).
        - **Single-field list mode**: List of 1D arrays, each with shape (n_bins,)
        - **Multi-field list mode**: List of field sequences, where each sequence
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
    scale_bar : bool or ScaleBarConfig, default=False
        If True, enable napari's native scale bar. If ScaleBarConfig object,
        configure scale bar appearance. Requires environment to have units set.
    speed : float, default=1.0
        Playback speed multiplier relative to real-time:

        - 1.0: Real-time playback (1 second of data = 1 second viewing)
        - 0.1: 10% speed (slow motion, good for replay analysis)
        - 2.0: 2× speed (fast forward)

        This parameter is passed from ``animate_fields()`` along with
        ``sample_rate_hz`` to enable the interactive speed control widget
        to show speed multipliers instead of raw fps values.
    sample_rate_hz : float, optional
        Data sample rate in Hz (e.g., 30.0 for 30 Hz position tracking, 500.0
        for replay decoding). Computed from ``frame_times`` by ``animate_fields()``
        and used by the speed control widget to convert between speed multiplier
        and playback fps. If None, defaults to computing from fps.
    max_playback_fps : int, default=MAX_PLAYBACK_FPS (25)
        Maximum playback fps for the speed control widget. Higher values may
        exceed display refresh rate. Passed from ``animate_fields()`` to ensure
        consistent speed capping across the API.
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

    # Detect if fields is a 2D array (memmap-friendly path)
    # This skips validation/multi-field detection which require list iteration
    fields_is_array = isinstance(fields, np.ndarray) and fields.ndim == 2

    # Only validate list inputs (array inputs don't need type consistency checks)
    if not fields_is_array:
        # Validate field types are consistent (all arrays or all lists)
        _validate_field_types_consistent(fields)  # type: ignore[arg-type]

    # Detect multi-field input and route appropriately
    # (only possible for list inputs, not arrays)
    # Type narrowing: if not fields_is_array, fields is a list
    if not fields_is_array and _is_multi_field_input(fields):  # type: ignore[arg-type]
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
            scale_bar=scale_bar,
            speed=speed,
            sample_rate_hz=sample_rate_hz,
            max_playback_fps=max_playback_fps,
        )

    # Single-field viewer (original behavior)
    # At this point, fields is either:
    # - 2D array with shape (n_frames, n_bins) if fields_is_array is True
    # - list[NDArray] if fields_is_array is False (not multi-field)
    from neurospatial.animation.rendering import compute_global_colormap_range

    # Compute global color scale
    if vmin is None or vmax is None:
        # Compute n_frames for large dataset detection
        n_frames_for_range = (
            fields.shape[0] if fields_is_array else len(fields)  # type: ignore[union-attr]
        )

        # For very large datasets, use subsampling for faster range estimation
        sample_stride: int | None = None
        large_dataset_threshold = 200_000

        if n_frames_for_range > large_dataset_threshold:
            # Compute sample_stride to sample ~50K frames
            sample_stride = max(1, n_frames_for_range // 50_000)
            warnings.warn(
                f"Estimating colormap range from {n_frames_for_range:,} frames using "
                f"sample_stride={sample_stride} (sampling every {sample_stride}th frame). "
                f"For exact range, pass explicit vmin/vmax parameters.",
                UserWarning,
                stacklevel=2,
            )

        vmin_computed, vmax_computed = compute_global_colormap_range(
            fields,  # type: ignore[arg-type]
            vmin=vmin,
            vmax=vmax,
            sample_stride=sample_stride,
        )
        vmin = vmin if vmin is not None else vmin_computed
        vmax = vmax if vmax is not None else vmax_computed

    # Note: contrast_limits parameter is ignored for RGB images
    # RGB images are already in [0, 255] range and don't need normalization
    # We keep the parameter for API compatibility but don't use it

    # Pre-compute colormap lookup table (256 RGB values)
    cmap_obj = plt.get_cmap(cmap)
    cmap_lookup = (cmap_obj(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)

    # Create lazy frame loader with chunked caching
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

    # Compute scale for napari (converts pixels to environment units)
    # This enables the scale bar to show correct physical units (e.g., "10 cm")
    # Scale is (time, y, x) for shape (time, height, width, channels)
    layer_scale: tuple[float, float, float] | None = None
    if (
        hasattr(env.layout, "grid_shape")
        and env.layout.grid_shape is not None
        and len(env.layout.grid_shape) == 2
        and env.dimension_ranges is not None
    ):
        n_x, n_y = env.layout.grid_shape
        (x_min, x_max), (y_min, y_max) = env.dimension_ranges
        # After transposition: image is (n_y, n_x) so scale is (y_scale, x_scale)
        y_scale = (y_max - y_min) / n_y if n_y > 0 else 1.0
        x_scale = (x_max - x_min) / n_x if n_x > 0 else 1.0
        # Scale for (time, height, width): time=1 (no scaling)
        layer_scale = (1.0, y_scale, x_scale)

    # Add image layer (RGB - no contrast_limits needed)
    viewer.add_image(
        lazy_frames,
        name="Spatial Fields",
        rgb=True,  # Already RGB
        scale=layer_scale,  # Physical units scale (enables correct scale bar)
        multiscale=False,  # Disable pyramid computation for large datasets
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
        _add_speed_control_widget(
            viewer,
            frame_labels=frame_labels,
            initial_speed=speed,
            sample_rate_hz=sample_rate_hz if sample_rate_hz is not None else 30.0,
            max_playback_fps=max_playback_fps,
        )
    except ImportError:
        # Fallback: Add spacebar shortcut without widget (magicgui not available)
        @viewer.bind_key("Space")
        def toggle_playback(viewer: napari.Viewer) -> None:
            """Toggle animation playback with spacebar."""
            viewer.window._toggle_play()

    # Create PlaybackController for centralized playback control
    # Compute n_frames from fields (type ignore needed for union-attr on shape)
    n_frames = (
        fields.shape[0]  # type: ignore[union-attr]
        if fields_is_array
        else len(fields)
    )

    # Extract frame_times from overlay_data if available
    frame_times = overlay_data.frame_times if overlay_data is not None else None

    # Create controller and store as viewer attribute
    # Note: Using object.__setattr__ to bypass pydantic validation on napari.Viewer
    # which doesn't allow arbitrary attribute assignment
    controller = PlaybackController(
        viewer=viewer,
        n_frames=n_frames,
        fps=float(fps),
        frame_times=frame_times,
        allow_frame_skip=True,  # Enable frame skipping by default
    )
    object.__setattr__(viewer, "playback_controller", controller)

    # Render overlay data if provided
    if overlay_data is not None:
        # Extract overlay scale from layer_scale (y_scale, x_scale)
        # This aligns overlays with the scaled field image layer
        overlay_scale = (layer_scale[1], layer_scale[2]) if layer_scale else None

        # Render video overlays (z_order="below" first, then field layer exists, then "above")
        n_frames = len(fields)
        video_layers_needing_callback: list[Layer] = []
        for idx, video_data in enumerate(overlay_data.videos):
            suffix = f" {idx + 1}" if len(overlay_data.videos) > 1 else ""
            name = f"Video{suffix}"
            if video_data.z_order == "below":
                # Add video layer below field - it will appear under the field
                # We need to reorder layers after adding
                video_layer, uses_native_time = _add_video_layer(
                    viewer, video_data, env, n_frames, name=name
                )
                # Move video layer to bottom of stack (index 0)
                viewer.layers.move(viewer.layers.index(video_layer), 0)
                if not uses_native_time:
                    video_layers_needing_callback.append(video_layer)
            else:
                # Add video layer above field (default position - on top)
                video_layer, uses_native_time = _add_video_layer(
                    viewer, video_data, env, n_frames, name=name
                )
                if not uses_native_time:
                    video_layers_needing_callback.append(video_layer)

        # Register callback only for file-based videos (not in-memory with native time)
        _make_video_frame_callback(viewer, video_layers_needing_callback)

        # Render event overlays FIRST (spike events, region crossings, etc.)
        # Events are background context, rendered below position/bodypart overlays
        for idx, event_data in enumerate(overlay_data.events):
            suffix = f" {idx + 1}" if len(overlay_data.events) > 1 else ""
            _render_event_overlay(
                viewer, event_data, env, name_suffix=suffix, scale=overlay_scale
            )

        # Render position overlays (tracks + points) - above events
        for idx, pos_data in enumerate(overlay_data.positions):
            suffix = f" {idx + 1}" if len(overlay_data.positions) > 1 else ""
            _render_position_overlay(
                viewer, pos_data, env, name_suffix=suffix, scale=overlay_scale
            )

        # Render bodypart overlays (points + skeleton)
        for idx, bodypart_data in enumerate(overlay_data.bodypart_sets):
            suffix = f" {idx + 1}" if len(overlay_data.bodypart_sets) > 1 else ""
            _render_bodypart_overlay(
                viewer, bodypart_data, env, name_suffix=suffix, scale=overlay_scale
            )

        # Render head direction overlays (vectors) - on top
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
                scale=overlay_scale,
            )

        # Add time series dock widget if present
        if overlay_data.timeseries and overlay_data.frame_times is not None:
            _add_timeseries_dock(
                viewer=viewer,
                timeseries_data=overlay_data.timeseries,
                frame_times=overlay_data.frame_times,
            )

    # Render regions if requested
    if show_regions:
        _render_regions(viewer, env, show_regions, region_alpha)

    # Configure napari's native scale bar if requested
    if scale_bar:
        from neurospatial.visualization.scale_bar import (
            ScaleBarConfig,
            configure_napari_scale_bar,
        )

        config = scale_bar if isinstance(scale_bar, ScaleBarConfig) else None
        configure_napari_scale_bar(viewer, units=env.units, config=config)

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
    scale_bar: bool | Any = False,  # bool | ScaleBarConfig
    speed: float = 1.0,
    sample_rate_hz: float | None = None,
    max_playback_fps: int = MAX_PLAYBACK_FPS,
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

    # Compute scale for napari (converts pixels to environment units)
    # This enables the scale bar to show correct physical units (e.g., "10 cm")
    layer_scale: tuple[float, float, float] | None = None
    if (
        hasattr(env.layout, "grid_shape")
        and env.layout.grid_shape is not None
        and len(env.layout.grid_shape) == 2
        and env.dimension_ranges is not None
    ):
        n_x, n_y = env.layout.grid_shape
        (x_min, x_max), (y_min, y_max) = env.dimension_ranges
        y_scale = (y_max - y_min) / n_y if n_y > 0 else 1.0
        x_scale = (x_max - x_min) / n_x if n_x > 0 else 1.0
        layer_scale = (1.0, y_scale, x_scale)

    # Add image layers
    for renderer, name in zip(lazy_renderers, layer_names, strict=True):
        viewer.add_image(
            renderer,
            name=name,
            rgb=True,
            scale=layer_scale,  # Physical units scale (enables correct scale bar)
            multiscale=False,  # Disable pyramid computation for large datasets
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

        _add_speed_control_widget(
            viewer,
            frame_labels=frame_labels,
            initial_speed=speed,
            sample_rate_hz=sample_rate_hz if sample_rate_hz is not None else 30.0,
            max_playback_fps=max_playback_fps,
        )
    except ImportError:

        @viewer.bind_key("Space")
        def toggle_playback(viewer: napari.Viewer) -> None:
            """Toggle animation playback with spacebar."""
            viewer.window._toggle_play()

    # Create PlaybackController for centralized playback control
    # Compute n_frames from field sequences
    n_frames = len(field_sequences[0]) if field_sequences else 0

    # Extract frame_times from overlay_data if available
    frame_times = overlay_data.frame_times if overlay_data is not None else None

    # Create controller and store as viewer attribute
    # Note: Using object.__setattr__ to bypass pydantic validation on napari.Viewer
    # which doesn't allow arbitrary attribute assignment
    controller = PlaybackController(
        viewer=viewer,
        n_frames=n_frames,
        fps=float(fps),
        frame_times=frame_times,
        allow_frame_skip=True,  # Enable frame skipping by default
    )
    object.__setattr__(viewer, "playback_controller", controller)

    # Render overlay data if provided
    if overlay_data is not None:
        # Extract overlay scale from layer_scale (y_scale, x_scale)
        # This aligns overlays with the scaled field image layer
        overlay_scale = (layer_scale[1], layer_scale[2]) if layer_scale else None

        # Render video overlays (z_order="below" first, then field layer exists, then "above")
        # n_frames already computed above
        video_layers_needing_callback: list[Layer] = []
        for idx, video_data in enumerate(overlay_data.videos):
            suffix = f" {idx + 1}" if len(overlay_data.videos) > 1 else ""
            name = f"Video{suffix}"
            if video_data.z_order == "below":
                # Add video layer below field - it will appear under the field
                video_layer, uses_native_time = _add_video_layer(
                    viewer, video_data, env, n_frames, name=name
                )
                # Move video layer to bottom of stack (index 0)
                viewer.layers.move(viewer.layers.index(video_layer), 0)
                if not uses_native_time:
                    video_layers_needing_callback.append(video_layer)
            else:
                # Add video layer above field (default position - on top)
                video_layer, uses_native_time = _add_video_layer(
                    viewer, video_data, env, n_frames, name=name
                )
                if not uses_native_time:
                    video_layers_needing_callback.append(video_layer)

        # Register callback only for file-based videos (not in-memory with native time)
        _make_video_frame_callback(viewer, video_layers_needing_callback)

        # Render event overlays FIRST (spike events, region crossings, etc.)
        # Events are background context, rendered below position/bodypart overlays
        for idx, event_data in enumerate(overlay_data.events):
            suffix = f" {idx + 1}" if len(overlay_data.events) > 1 else ""
            _render_event_overlay(
                viewer, event_data, env, name_suffix=suffix, scale=overlay_scale
            )

        # Render position overlays (tracks + points) - above events
        for idx, pos_data in enumerate(overlay_data.positions):
            suffix = f" {idx + 1}" if len(overlay_data.positions) > 1 else ""
            _render_position_overlay(
                viewer, pos_data, env, name_suffix=suffix, scale=overlay_scale
            )

        # Render bodypart overlays (points + skeleton)
        for idx, bodypart_data in enumerate(overlay_data.bodypart_sets):
            suffix = f" {idx + 1}" if len(overlay_data.bodypart_sets) > 1 else ""
            _render_bodypart_overlay(
                viewer, bodypart_data, env, name_suffix=suffix, scale=overlay_scale
            )

        # Render head direction overlays (vectors) - on top
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
                scale=overlay_scale,
            )

        # Add time series dock widget if present
        if overlay_data.timeseries and overlay_data.frame_times is not None:
            _add_timeseries_dock(
                viewer=viewer,
                timeseries_data=overlay_data.timeseries,
                frame_times=overlay_data.frame_times,
            )

    # Render regions if requested
    if show_regions:
        _render_regions(viewer, env, show_regions, region_alpha)

    # Configure napari's native scale bar if requested
    if scale_bar:
        from neurospatial.visualization.scale_bar import (
            ScaleBarConfig,
            configure_napari_scale_bar,
        )

        config = scale_bar if isinstance(scale_bar, ScaleBarConfig) else None
        configure_napari_scale_bar(viewer, units=env.units, config=config)

    return viewer


def _create_lazy_field_renderer(
    env: Environment,
    fields: list[NDArray[np.float64]] | NDArray[np.float64],
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
    fields : list of arrays or 2D array
        All fields to animate. Can be a list of 1D arrays (n_bins,) or a
        2D array (n_frames, n_bins). Array input is recommended for large
        datasets and memory-mapped files.
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
    # Compute n_frames: use shape[0] for arrays, len() for lists
    n_frames = fields.shape[0] if isinstance(fields, np.ndarray) else len(fields)

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
    fields : list of ndarray or 2D ndarray
        All fields to animate. Can be a list of 1D arrays (n_bins,) or a
        2D array (n_frames, n_bins). Array input is recommended for large
        datasets and memory-mapped files.
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
    _fields_is_array : bool
        True if fields is a 2D numpy array, False if list

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
        fields: list[NDArray[np.float64]] | NDArray[np.float64],
        cmap_lookup: NDArray[np.uint8],
        vmin: float,
        vmax: float,
        cache_size: int = DEFAULT_CACHE_SIZE,
    ) -> None:
        """Initialize lazy field renderer."""
        self.env = env
        self.fields = fields
        self._fields_is_array = isinstance(fields, np.ndarray)
        self.cmap_lookup = cmap_lookup
        self.vmin = vmin
        self.vmax = vmax
        self._cache: OrderedDict[int, NDArray[np.uint8]] = OrderedDict()
        self._cache_size = cache_size
        self._lock = Lock()

    def __len__(self) -> int:
        """Return number of frames."""
        # Use isinstance for mypy type narrowing (faster check already done in __init__)
        if isinstance(self.fields, np.ndarray):
            return int(self.fields.shape[0])
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
                start, stop, step = frame_idx.indices(len(self))
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
        n_frames = len(self)
        if idx < 0:
            idx = n_frames + idx

        # Validate bounds
        if idx < 0 or idx >= n_frames:
            original_idx = idx - n_frames if idx < 0 else idx
            raise IndexError(
                f"Frame index {original_idx} out of range for {n_frames} frames"
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
        return (len(self), *sample.shape)

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
    fields : list of ndarray or 2D ndarray
        All fields to animate. Can be a list of 1D arrays (n_bins,) or a
        2D array (n_frames, n_bins). Array input is recommended for large
        datasets and memory-mapped files.
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
    _fields_is_array : bool
        True if fields is a 2D numpy array, False if list

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
        fields: list[NDArray[np.float64]] | NDArray[np.float64],
        cmap_lookup: NDArray[np.uint8],
        vmin: float,
        vmax: float,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        max_chunks: int = DEFAULT_MAX_CHUNKS,
    ) -> None:
        """Initialize chunked lazy field renderer."""
        self.env = env
        self.fields = fields
        self._fields_is_array = isinstance(fields, np.ndarray)
        self.cmap_lookup = cmap_lookup
        self.vmin = vmin
        self.vmax = vmax
        self._chunk_size = chunk_size
        self._max_chunks = max_chunks
        self._chunk_cache: OrderedDict[int, list[NDArray[np.uint8]]] = OrderedDict()
        self._lock = Lock()

    def __len__(self) -> int:
        """Return number of frames."""
        # Use isinstance for mypy type narrowing (faster check already done in __init__)
        if isinstance(self.fields, np.ndarray):
            return int(self.fields.shape[0])
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
        end_frame = min(start_frame + self._chunk_size, len(self))

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
                start, stop, step = frame_idx.indices(len(self))
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
        n_frames = len(self)
        if idx < 0:
            idx = n_frames + idx

        # Validate bounds
        if idx < 0 or idx >= n_frames:
            original_idx = idx - n_frames if idx < 0 else idx
            raise IndexError(
                f"Frame index {original_idx} out of range for {n_frames} frames"
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
        return (len(self), *sample.shape)

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
