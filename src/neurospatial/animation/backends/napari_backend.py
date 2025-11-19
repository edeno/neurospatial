"""Napari GPU-accelerated viewer backend."""

from __future__ import annotations

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
    contrast_limits: tuple[float, float] | None = None,
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
    contrast_limits : tuple of float, optional
        Napari-specific contrast limits (overrides vmin/vmax)
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

    if contrast_limits is None:
        contrast_limits = (vmin, vmax)

    # Pre-compute colormap lookup table (256 RGB values)
    cmap_obj = plt.get_cmap(cmap)
    cmap_lookup = (cmap_obj(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)

    # Create lazy frame loader
    lazy_frames = _create_lazy_field_renderer(env, fields, cmap_lookup, vmin, vmax)

    # Create napari viewer
    viewer = napari.Viewer(title=title)

    # Add image layer
    viewer.add_image(
        lazy_frames,
        name="Spatial Fields",
        rgb=True,  # Already RGB
        contrast_limits=contrast_limits,
    )

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

    def __getitem__(self, idx: int) -> NDArray[np.uint8]:
        """Render frame on-demand when Napari requests it.

        Implements true LRU caching:
        1. If frame in cache, move to end (mark as recently used)
        2. If frame not in cache, render it and add to end
        3. If cache full, evict oldest frame (first item)

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
