"""Napari GPU-accelerated viewer backend."""

from __future__ import annotations

import contextlib
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

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


def _add_speed_control_widget(viewer: Any, initial_fps: int = 30) -> None:
    """Add interactive playback speed control widget to napari viewer.

    Creates a docked widget with an FPS slider that updates playback speed
    in real-time. Requires magicgui for widget creation.

    Parameters
    ----------
    viewer : napari.Viewer
        Napari viewer instance to add widget to
    initial_fps : int, default=30
        Initial playback speed in frames per second

    Notes
    -----
    If magicgui is not available, this function silently returns without
    adding the widget. This ensures the napari backend works even if
    magicgui is not installed.
    """
    try:
        from magicgui import magicgui
        from napari.settings import get_settings
    except ImportError:
        # magicgui or napari.settings not available - skip widget
        return

    # Create speed control widget using magicgui
    @magicgui(
        auto_call=True,
        fps={
            "widget_type": "Slider",
            "min": 1,
            "max": 120,
            "value": initial_fps,
            "label": "Playback Speed (FPS)",
        },
    )
    def speed_control(fps: int = initial_fps) -> None:
        """Update playback speed in real-time."""
        settings = get_settings()
        settings.application.playback_fps = fps

    # Add widget as dock to viewer
    # If docking fails (e.g., no Qt window), silently skip
    with contextlib.suppress(Exception):
        viewer.window.add_dock_widget(
            speed_control,
            name="Playback Speed",
            area="left",  # Dock on left side
        )


def render_napari(
    env: Environment,
    fields: list[NDArray[np.float64]],
    *,
    fps: int = 30,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    frame_labels: list[str] | None = None,
    overlay_trajectory: NDArray[np.float64] | None = None,
    title: str = "Spatial Field Animation",
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
    fields : list of arrays
        Fields to animate, each with shape (n_bins,)
    fps : int, default=30
        Frames per second for playback (Napari slider speed)
    cmap : str, default="viridis"
        Matplotlib colormap name
    vmin, vmax : float, optional
        Color scale limits. If None, computed from all fields.
    frame_labels : list of str, optional
        Frame labels (e.g., ["Trial 1", "Trial 2", ...])
        Note: Not currently displayed in Napari (future enhancement)
    overlay_trajectory : ndarray, optional
        Positions to overlay as trajectory (shape: n_timepoints, n_dims)
        - 2D trajectories: rendered as tracks
        - Higher dimensions: rendered as points
    title : str, default="Spatial Field Animation"
        Viewer window title
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
    ValueError
        If overlay_trajectory has invalid shape (must be 2D)

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.animation.backends.napari_backend import render_napari
    >>>
    >>> # Create environment
    >>> positions = np.random.randn(100, 2) * 50
    >>> env = Environment.from_samples(positions, bin_size=10.0)
    >>>
    >>> # Create field sequence
    >>> fields = [np.random.rand(env.n_bins) for _ in range(100)]
    >>>
    >>> # Launch viewer
    >>> viewer = render_napari(env, fields, fps=30, cmap="viridis")
    >>> # napari.run()  # Uncomment to block until window closed

    Notes
    -----
    **Playback Controls:**

    Napari provides built-in playback controls at the **bottom-left** of the viewer window:

    - **Frame slider** - Horizontal slider showing current frame position
    - **Play button (▶)** - Triangle icon to start/stop animation (next to slider)
    - **Frame counter** - Shows "1/N" indicating current frame
    - **Keyboard shortcuts** - Arrow keys to step forward/backward

    **Playback Speed Control:**

    An interactive "Playback Speed" widget is automatically added to the left side
    of the viewer (requires magicgui, which is included with napari[all]):

    - **FPS Slider** - Drag slider to adjust playback speed from 1-120 FPS in real-time
    - **Current FPS** - Shows current speed setting
    - Updates instantly as you drag the slider

    The animation starts at frame 0 with playback speed set by the `fps` parameter.
    If magicgui is not available, the speed can still be changed via
    File → Preferences → Application → "Playback frames per second".

    **Note:** Only the time dimension slider is shown. Spatial dimensions (height, width)
    are displayed in the 2D viewport, not as separate sliders.

    **Memory Efficiency:**

    Napari backend uses lazy loading with LRU caching:
    - Frames rendered on-demand when requested
    - Cache size: 1000 frames (configurable in LazyFieldRenderer)
    - Oldest frames evicted when cache full
    - Works efficiently with memory-mapped arrays

    **Performance:**

    - Seek time: <100ms for 100K+ frames
    - GPU acceleration for rendering
    - Suitable for hour-long sessions (900K frames at 250 Hz)

    **Trajectory Overlay:**

    2D trajectories are rendered as tracks with temporal dimension.
    Higher-dimensional trajectories fall back to point cloud rendering.

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

    from neurospatial.animation.rendering import compute_global_colormap_range

    # Compute global color scale
    if vmin is None or vmax is None:
        vmin_computed, vmax_computed = compute_global_colormap_range(fields, vmin, vmax)
        vmin = vmin if vmin is not None else vmin_computed
        vmax = vmax if vmax is not None else vmax_computed

    # Note: contrast_limits parameter is ignored for RGB images
    # RGB images are already in [0, 255] range and don't need normalization
    # We keep the parameter for API compatibility but don't use it

    # Pre-compute colormap lookup table (256 RGB values)
    cmap_obj = plt.get_cmap(cmap)
    cmap_lookup = (cmap_obj(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)

    # Create lazy frame loader
    lazy_frames = _create_lazy_field_renderer(env, fields, cmap_lookup, vmin, vmax)

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
        _add_speed_control_widget(viewer, initial_fps=fps)
    except ImportError:
        # Fallback for older napari versions
        pass

    # Add trajectory overlay if provided
    if overlay_trajectory is not None:
        if overlay_trajectory.ndim != 2:
            raise ValueError(
                f"overlay_trajectory must be 2D (n_timepoints, n_dims), "
                f"got shape {overlay_trajectory.shape}"
            )

        # For 2D trajectories, add as tracks
        if overlay_trajectory.shape[1] == 2:
            # Create track data: (track_id, time, y, x)
            n_points = len(overlay_trajectory)
            track_data = np.column_stack(
                [
                    np.zeros(n_points),  # Single track
                    np.arange(n_points),  # Time
                    overlay_trajectory[:, 1],  # Y
                    overlay_trajectory[:, 0],  # X
                ]
            )
            viewer.add_tracks(track_data, name="Trajectory")
        else:
            # Higher dimensional - just plot as points
            viewer.add_points(overlay_trajectory, name="Trajectory", size=2)

    # Frame labels currently not displayed in napari
    # (could be future enhancement with custom widget)

    return viewer


def _create_lazy_field_renderer(
    env: Environment,
    fields: list[NDArray[np.float64]],
    cmap_lookup: NDArray[np.uint8],
    vmin: float,
    vmax: float,
) -> LazyFieldRenderer:
    """Create lazy field renderer for Napari.

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

    Returns
    -------
    renderer : LazyFieldRenderer
        Lazy renderer instance implementing array-like interface
        for Napari
    """
    return LazyFieldRenderer(env, fields, cmap_lookup, vmin, vmax)


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

    Attributes
    ----------
    _cache : OrderedDict
        LRU cache mapping frame index to rendered RGB array
    _cache_size : int
        Maximum number of frames to cache (default: 1000)

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
    >>> renderer = LazyFieldRenderer(env, fields, cmap_lookup, 0.0, 1.0)
    >>> len(renderer)  # Number of frames
    100
    >>> frame = renderer[0]  # Render frame 0 (cached)
    >>> frame = renderer[0]  # Retrieved from cache (instant)
    >>> frame = renderer[-1]  # Negative indexing supported
    """

    def __init__(
        self,
        env: Environment,
        fields: list[NDArray[np.float64]],
        cmap_lookup: NDArray[np.uint8],
        vmin: float,
        vmax: float,
    ):
        """Initialize lazy field renderer."""
        self.env = env
        self.fields = fields
        self.cmap_lookup = cmap_lookup
        self.vmin = vmin
        self.vmax = vmax
        self._cache: OrderedDict[int, NDArray[np.uint8]] = OrderedDict()
        self._cache_size = 1000

    def __len__(self) -> int:
        """Return number of frames."""
        return len(self.fields)

    def __getitem__(self, idx: int | tuple) -> NDArray[np.uint8]:
        """Render frame on-demand when Napari requests it.

        Implements true LRU caching:
        1. If frame in cache, move to end (mark as recently used)
        2. If frame not in cache, render it and add to end
        3. If cache full, evict oldest frame (first item)

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
