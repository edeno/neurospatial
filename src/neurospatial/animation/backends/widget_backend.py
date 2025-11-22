"""Jupyter notebook widget backend."""

from __future__ import annotations

import io
import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from neurospatial.animation._timing import timing

# Module-level logger for fallback diagnostics
_logger = logging.getLogger("neurospatial.animation.backends.widget_backend")

if TYPE_CHECKING:
    from neurospatial.environment.core import Environment

# Check ipywidgets availability
try:
    import ipywidgets
    from IPython.display import display

    IPYWIDGETS_AVAILABLE = True
except ImportError:
    IPYWIDGETS_AVAILABLE = False


def render_field_to_png_bytes_with_overlays(
    env: Environment,
    field: NDArray[np.float64],
    cmap: str,
    vmin: float,
    vmax: float,
    dpi: int,
    frame_idx: int,
    overlay_data: Any | None = None,
    show_regions: bool | list[str] = False,
    region_alpha: float = 0.3,
) -> bytes:
    """Render field with overlays to PNG bytes.

    This function reuses the video backend's overlay rendering logic to
    produce PNG frames with overlays for the widget backend. It creates
    a matplotlib figure, renders the field, adds overlays, and saves to
    PNG bytes.

    Parameters
    ----------
    env : Environment
        Environment defining spatial structure.
    field : ndarray of shape (n_bins,), dtype float64
        Field values to render. Length must match env.n_bins.
    cmap : str
        Matplotlib colormap name (e.g., "viridis", "hot", "plasma").
    vmin : float
        Minimum value for color scale normalization.
    vmax : float
        Maximum value for color scale normalization.
    dpi : int
        Resolution for rendering in dots per inch.
    frame_idx : int
        Current frame index (for extracting overlay data at this time point).
    overlay_data : OverlayData or None, optional
        Overlay data structure containing positions, bodyparts, head directions.
    show_regions : bool or list of str, default=False
        Whether to show regions. If True, show all regions. If list, show
        only specified region names.
    region_alpha : float, default=0.3
        Alpha transparency for region rendering, range [0.0, 1.0] where 0.0 is
        fully transparent and 1.0 is fully opaque.

    Returns
    -------
    png_bytes : bytes
        PNG-encoded image data ready for display in widget.

    Notes
    -----
    This function is designed to work with the LRU cache in the widget backend.
    It uses the same overlay rendering logic as the video backend for consistency.

    The function creates a fresh matplotlib figure for each call to avoid
    state accumulation between frames. This ensures clean rendering but has
    a performance cost compared to artist reuse.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        from neurospatial import Environment
        from neurospatial.animation.overlays import PositionData, OverlayData

        positions = np.random.randn(100, 2) * 50
        env = Environment.from_samples(positions, bin_size=10.0)
        field = np.random.rand(env.n_bins)

        # Create overlay data
        overlay_positions = np.random.rand(20, 2) * 50
        overlay_data = OverlayData(
            positions=[
                PositionData(
                    data=overlay_positions, color="red", size=10.0, trail_length=5
                )
            ]
        )

        # Render frame with overlays
        png_bytes = render_field_to_png_bytes_with_overlays(
            env, field, "viridis", 0, 1, dpi=100, frame_idx=0, overlay_data=overlay_data
        )
        len(png_bytes) > 0
        True
    """
    with timing("render_field_to_png_bytes_with_overlays"):
        # Set Agg backend BEFORE any pyplot imports
        try:
            import matplotlib

            if matplotlib.get_backend().lower() not in (
                "agg",
                "module://matplotlib_inline.backend_inline",
            ):
                matplotlib.use("Agg", force=True)
        except Exception:
            pass

        import matplotlib.pyplot as plt

        # Import overlay rendering function from video backend
        from neurospatial.animation._parallel import _render_all_overlays

        # Create figure and axes
        fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)
        ax.set_axis_off()

        # Render field using environment's plot method
        env.plot_field(
            field,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            colorbar=False,
        )

        # Render overlays if provided
        if overlay_data is not None:
            _render_all_overlays(
                ax, env, frame_idx, overlay_data, show_regions, region_alpha
            )

        # Save to PNG bytes (removed bbox_inches="tight" for consistent dimensions)
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)  # Close figure to free memory
        buf.seek(0)
        png_bytes = buf.read()

        return png_bytes


class PersistentFigureRenderer:
    """Reusable figure renderer for efficient on-demand frame generation.

    Creates a single matplotlib Figure/Axes that is reused across frames,
    updating only the field data rather than recreating the entire figure.
    This can provide 2-5x speedup for cache miss rendering in the widget backend.

    Parameters
    ----------
    env : Environment
        Environment defining spatial structure.
    cmap : str
        Matplotlib colormap name (e.g., "viridis", "hot", "plasma").
    vmin : float
        Minimum value for color scale normalization.
    vmax : float
        Maximum value for color scale normalization.
    dpi : int, default=100
        Resolution for rendering in dots per inch.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        from neurospatial import Environment
        from neurospatial.animation.backends.widget_backend import (
            PersistentFigureRenderer,
        )

        positions = np.random.randn(100, 2) * 50
        env = Environment.from_samples(positions, bin_size=10.0)
        fields = [np.random.rand(env.n_bins) for _ in range(10)]

        renderer = PersistentFigureRenderer(env, "viridis", 0.0, 1.0, dpi=100)
        try:
            for i, field in enumerate(fields):
                png_bytes = renderer.render(field, frame_idx=i)
                # Use png_bytes...
        finally:
            renderer.close()

    Notes
    -----
    **Performance:**

    The persistent figure approach avoids per-frame overhead of:
    - Creating new Figure/Axes objects
    - Setting up axis limits, aspect ratio, colormap
    - Layout/tight_layout calculations

    Instead, only the image data is updated via ``set_data()``, and overlays
    use persistent artists via OverlayArtistManager for efficient updates.

    **Memory:**

    The renderer holds a single matplotlib Figure in memory. Call ``close()``
    when done to release memory.

    **Thread Safety:**

    Not thread-safe. Use one renderer per thread/process.

    See Also
    --------
    render_field_to_png_bytes_with_overlays : Fresh-figure rendering function.
    render_widget : Widget backend using persistent renderer for cache misses.
    """

    def __init__(
        self,
        env: Environment,
        cmap: str,
        vmin: float,
        vmax: float,
        dpi: int = 100,
        raise_on_fallback: bool = False,
    ) -> None:
        """Initialize the persistent figure renderer.

        Parameters
        ----------
        env : Environment
            Environment defining spatial structure
        cmap : str
            Matplotlib colormap name
        vmin : float
            Minimum value for colormap
        vmax : float
            Maximum value for colormap
        dpi : int, default=100
            Resolution for rendering
        raise_on_fallback : bool, default=False
            If True, raise RuntimeError when fallback to full re-render is required
            (e.g., for non-grid layouts). Useful for debugging performance issues.
            If False (default), fallback is logged at DEBUG level and rendering
            continues.
        """
        # Set Agg backend BEFORE any pyplot imports
        try:
            import matplotlib

            if matplotlib.get_backend().lower() not in (
                "agg",
                "module://matplotlib_inline.backend_inline",
            ):
                matplotlib.use("Agg", force=True)
        except Exception:
            pass

        import matplotlib.pyplot as plt
        from matplotlib.collections import QuadMesh

        self._env = env
        self._cmap = cmap
        self._vmin = vmin
        self._vmax = vmax
        self._dpi = dpi
        self._raise_on_fallback = raise_on_fallback
        self._plt = plt  # Store reference to prevent issues with lazy imports
        self._QuadMesh = QuadMesh  # Store class reference for isinstance checks

        # Create persistent figure
        self._fig, self._ax = plt.subplots(figsize=(8, 6), dpi=dpi)
        self._ax.set_axis_off()
        self._mesh: QuadMesh | None = None  # Will hold QuadMesh after first render
        self._is_first_render = True
        self._overlay_manager: Any = None  # Initialized on first render with overlays

    def render(
        self,
        field: NDArray[np.float64],
        frame_idx: int,
        overlay_data: Any | None = None,
        show_regions: bool | list[str] = False,
        region_alpha: float = 0.3,
    ) -> bytes:
        """Render field to PNG bytes, reusing figure.

        Parameters
        ----------
        field : ndarray, shape (n_bins,)
            Field values to render
        frame_idx : int
            Current frame index (for overlays)
        overlay_data : OverlayData, optional
            Overlay data structure containing positions, bodyparts, head directions
        show_regions : bool or list of str, default=False
            Whether to show regions. If True, show all regions. If list, show
            only specified region names.
        region_alpha : float, default=0.3
            Alpha transparency for region rendering (0=transparent, 1=opaque)

        Returns
        -------
        bytes
            PNG image data
        """
        from neurospatial.animation._parallel import OverlayArtistManager

        if self._is_first_render:
            # First render: create image using environment's plot method
            self._env.plot_field(
                field,
                ax=self._ax,
                cmap=self._cmap,
                vmin=self._vmin,
                vmax=self._vmax,
                colorbar=False,
            )
            # Store reference to QuadMesh (from pcolormesh) for efficient updates
            # plot_field uses pcolormesh which creates QuadMesh in ax.collections
            for collection in self._ax.collections:
                if isinstance(collection, self._QuadMesh):
                    self._mesh = collection
                    break

            # Initialize overlay manager if we have overlays
            if overlay_data is not None or show_regions:
                self._overlay_manager = OverlayArtistManager(
                    ax=self._ax,
                    env=self._env,
                    overlay_data=overlay_data,
                    show_regions=show_regions,
                    region_alpha=region_alpha,
                )
                self._overlay_manager.initialize(frame_idx=frame_idx)

            self._is_first_render = False
        else:
            # Subsequent renders: update mesh data using set_array (efficient)
            if self._mesh is not None:
                # Check if layout supports efficient updates
                mesh_data = self._field_to_mesh_array(field)
                if mesh_data is not None:
                    self._mesh.set_array(mesh_data)
                else:
                    # Fallback: re-render completely for non-grid layouts
                    layout_type = getattr(self._env.layout, "layout_type", "unknown")
                    fallback_reason = (
                        f"set_array optimization not supported for layout type "
                        f"'{layout_type}' (is_grid_compatible=False)"
                    )
                    self._handle_fallback(fallback_reason)
                    self._do_full_rerender(
                        field, overlay_data, show_regions, region_alpha, frame_idx
                    )
            else:
                # No mesh found, re-render completely
                fallback_reason = "QuadMesh not found in axes, requiring full re-render"
                self._handle_fallback(fallback_reason)
                self._do_full_rerender(
                    field, overlay_data, show_regions, region_alpha, frame_idx
                )

            # Update overlays using manager
            if self._overlay_manager is not None:
                self._overlay_manager.update_frame(frame_idx)

        # Capture to PNG bytes (removed bbox_inches="tight" for consistent dimensions)
        with timing("PersistentFigureRenderer.render_savefig"):
            buf = io.BytesIO()
            self._fig.savefig(buf, format="png")
            buf.seek(0)
            return buf.read()

    def _handle_fallback(self, reason: str) -> None:
        """Handle fallback to full re-render.

        Logs the fallback at DEBUG level, or raises RuntimeError if
        raise_on_fallback is True.

        Parameters
        ----------
        reason : str
            Description of why fallback is needed.

        Raises
        ------
        RuntimeError
            If raise_on_fallback was set to True in the constructor.
        """
        message = f"PersistentFigureRenderer fallback to full re-render: {reason}"

        if self._raise_on_fallback:
            raise RuntimeError(message)

        _logger.debug(message)

    def _do_full_rerender(
        self,
        field: NDArray[np.float64],
        overlay_data: Any | None,
        show_regions: bool | list[str],
        region_alpha: float,
        frame_idx: int,
    ) -> None:
        """Perform full re-render (fallback path).

        Clears axes and redraws everything from scratch. This is slower
        than the optimized set_array path but works for all layout types.

        Parameters
        ----------
        field : ndarray
            Field values to render.
        overlay_data : OverlayData or None
            Overlay data structure.
        show_regions : bool or list of str
            Region display settings.
        region_alpha : float
            Region transparency.
        frame_idx : int
            Current frame index.
        """
        from neurospatial.animation._parallel import OverlayArtistManager

        self._ax.clear()
        self._ax.set_axis_off()
        self._env.plot_field(
            field,
            ax=self._ax,
            cmap=self._cmap,
            vmin=self._vmin,
            vmax=self._vmax,
            colorbar=False,
        )

        # Find new QuadMesh after re-render
        self._mesh = None
        for collection in self._ax.collections:
            if isinstance(collection, self._QuadMesh):
                self._mesh = collection
                break

        # Reset overlay manager for re-initialized axes
        if overlay_data is not None or show_regions:
            self._overlay_manager = OverlayArtistManager(
                ax=self._ax,
                env=self._env,
                overlay_data=overlay_data,
                show_regions=show_regions,
                region_alpha=region_alpha,
            )
            self._overlay_manager.initialize(frame_idx=frame_idx)

    def _field_to_mesh_array(
        self, field: NDArray[np.float64]
    ) -> NDArray[np.float64] | None:
        """Convert field to flat array suitable for QuadMesh.set_array().

        Parameters
        ----------
        field : ndarray, shape (n_bins,)
            Field values to convert

        Returns
        -------
        ndarray or None
            Flat array for set_array() if grid layout, None otherwise.

        Notes
        -----
        For pcolormesh with shading='flat' (default), the array should have
        shape (nrows * ncols,). The array is flattened in row-major order
        from the transposed grid (grid_data.T.ravel()).
        """
        layout = self._env.layout

        # Only grid-compatible layouts can use set_array optimization
        # Grid-compatible layouts have is_grid_compatible=True (grid, mask, polygon)
        if not getattr(layout, "is_grid_compatible", False):
            return None

        # Check required grid attributes
        if (
            not hasattr(layout, "grid_shape")
            or layout.grid_shape is None
            or not hasattr(layout, "active_mask")
            or layout.active_mask is None
        ):
            return None

        from neurospatial.layout.helpers.utils import map_active_data_to_grid

        # Convert to grid
        grid_data: NDArray[np.float64] = map_active_data_to_grid(
            layout.grid_shape, layout.active_mask, field, fill_value=np.nan
        )

        # Transpose and flatten for pcolormesh set_array
        # pcolormesh expects data in the same shape/order as when first created
        return np.asarray(grid_data.T.ravel(), dtype=np.float64)

    def _field_to_image_data(
        self, field: NDArray[np.float64]
    ) -> NDArray[np.float64] | None:
        """Convert field to 2D image data suitable for set_data().

        .. deprecated::
            Use `_field_to_mesh_array` instead. This method is retained for
            backwards compatibility with code that may use imshow.

        Parameters
        ----------
        field : ndarray, shape (n_bins,)
            Field values to convert

        Returns
        -------
        ndarray or None
            2D array for set_data() if grid layout, None otherwise.
        """
        layout = self._env.layout

        # Only grid-compatible layouts can use set_data optimization
        # Grid-compatible layouts have is_grid_compatible=True (grid, mask, polygon)
        if not getattr(layout, "is_grid_compatible", False):
            return None

        # Check required grid attributes
        if (
            not hasattr(layout, "grid_shape")
            or layout.grid_shape is None
            or not hasattr(layout, "active_mask")
            or layout.active_mask is None
        ):
            return None

        from neurospatial.layout.helpers.utils import map_active_data_to_grid

        # Convert to grid
        grid_data: NDArray[np.float64] = map_active_data_to_grid(
            layout.grid_shape, layout.active_mask, field, fill_value=np.nan
        )

        # Transpose for display (matplotlib pcolormesh expects data.T)
        return np.asarray(grid_data.T, dtype=np.float64)

    def close(self) -> None:
        """Close the figure to free memory.

        Should be called when the renderer is no longer needed.
        """
        self._plt.close(self._fig)


def render_widget(
    env: Environment,
    fields: list[NDArray[np.float64]],
    *,
    fps: int = 30,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    frame_labels: list[str] | None = None,
    dpi: int = 100,
    initial_cache_size: int | None = None,
    cache_limit: int = 1000,
    overlay_data: Any | None = None,
    show_regions: bool | list[str] = False,
    region_alpha: float = 0.3,
    **kwargs: Any,  # Accept other parameters gracefully
) -> None:
    """Create interactive Jupyter widget with slider control.

    Pre-renders a subset of frames for responsive scrubbing, then renders
    remaining frames on-demand with LRU caching. Slider updates are throttled
    to ~30 Hz to prevent UI stutter with large images.

    Parameters
    ----------
    env : Environment
        Environment defining spatial structure.
    fields : list of ndarray of shape (n_bins,), dtype float64
        List of field arrays to animate. Each array contains field values
        for one frame.
    fps : int, default=30
        Frames per second for playback.
    cmap : str, default="viridis"
        Matplotlib colormap name (e.g., "viridis", "hot", "plasma").
    vmin : float, optional
        Minimum value for color scale normalization. If None, computed from
        all fields using NaN-robust min.
    vmax : float, optional
        Maximum value for color scale normalization. If None, computed from
        all fields using NaN-robust max.
    frame_labels : list of str, optional
        Frame labels (e.g., ["Trial 1", "Trial 2", ...]). If None,
        generates default labels "Frame 1", "Frame 2", etc.
    dpi : int, default=100
        Resolution for frame rendering in dots per inch.
    initial_cache_size : int, optional
        Number of frames to pre-render during initialization. If None,
        defaults to min(len(fields), 500). Increase for faster initial
        scrubbing with larger datasets.
    cache_limit : int, default=1000
        Maximum number of frames to keep in LRU cache. Increase for
        larger datasets if memory allows (~30-100 MB per 1000 frames).
    overlay_data : OverlayData or None, optional
        Overlay data structure containing positions, bodyparts, and head
        directions to render on top of spatial fields. If None, no overlays
        are rendered. Use the conversion funnel in `core.py` to create this
        from user-facing overlay dataclasses.
    show_regions : bool or list of str, default=False
        Whether to show regions overlaid on the animation. If True, shows all
        regions defined in the environment. If a list of strings, shows only
        regions with matching names. If False, no regions are displayed.
    region_alpha : float, default=0.3
        Alpha transparency for region rendering, range [0.0, 1.0] where 0.0 is
        fully transparent and 1.0 is fully opaque. Only used when show_regions
        is True or a list of region names.
    **kwargs : dict
        Additional parameters (accepted for backend compatibility, ignored).

    Returns
    -------
    None
        Widget is displayed directly in the notebook output area.
        Does not return a value to prevent duplicate auto-display.

    Raises
    ------
    ImportError
        If ipywidgets is not installed

    Examples
    --------
    .. code-block:: python

        import numpy as np
        from neurospatial import Environment
        from neurospatial.animation.backends.widget_backend import render_widget

        # Create environment
        positions = np.random.randn(100, 2) * 50
        env = Environment.from_samples(positions, bin_size=10.0)

        # Create fields
        fields = [np.random.rand(env.n_bins) for _ in range(20)]

        # In Jupyter notebook:
        render_widget(env, fields, fps=10)
        # Widget displays automatically with play/pause and slider controls

    Notes
    -----
    **Performance Strategy:**
    - Pre-renders configurable number of frames (default: 500) during initialization
    - Remaining frames rendered on-demand with LRU caching
    - Caches raw PNG bytes (not base64 strings) for efficiency
    - Slider updates throttled to ~30 Hz to prevent UI stutter
    - Balances responsiveness (pre-cached frames) with memory efficiency

    **Widget Controls:**
    - Play button: Automatic playback at specified FPS
    - Slider: Manual frame scrubbing (continuous_update=True for smooth scrubbing)
    - Frame counter: Shows current frame and total frames
    - Frame label: Displays custom label if provided

    **Memory Considerations:**
    - LRU cache: configurable size (default: 1000 frames, ~30-100 MB depending on DPI)
    - Caches raw PNG bytes directly (no base64 encoding/decoding overhead)
    - Automatic eviction of least-recently-used frames
    - Works efficiently with very large datasets (100K+ frames)
    """
    if not IPYWIDGETS_AVAILABLE:
        raise ImportError(
            "Widget backend requires ipywidgets. Install with:\n"
            "  pip install ipywidgets\n"
            "or\n"
            "  uv add ipywidgets"
        )

    import time
    from collections import OrderedDict

    from neurospatial.animation.rendering import (
        compute_global_colormap_range,
        render_field_to_png_bytes,
    )

    # Compute global color scale
    vmin, vmax = compute_global_colormap_range(fields, vmin, vmax)

    # LRU cache storing raw PNG bytes (not base64 strings)
    cached_frames: OrderedDict[int, bytes] = OrderedDict()

    def cache_put(i: int, png_bytes: bytes) -> None:
        """Add frame to cache with LRU eviction."""
        cached_frames[i] = png_bytes
        cached_frames.move_to_end(i)
        if len(cached_frames) > cache_limit:
            cached_frames.popitem(last=False)  # Remove oldest

    # Pre-render subset of frames for responsive scrubbing
    if initial_cache_size is None:
        initial_cache_size = min(len(fields), 500)
    else:
        initial_cache_size = min(initial_cache_size, len(fields))
    print(f"Pre-rendering {initial_cache_size} frames for widget...")

    for i in range(initial_cache_size):
        if overlay_data is not None or show_regions:
            # Render with overlays/regions
            png_bytes = render_field_to_png_bytes_with_overlays(
                env,
                fields[i],
                cmap,
                vmin,
                vmax,
                dpi,
                frame_idx=i,
                overlay_data=overlay_data,
                show_regions=show_regions,
                region_alpha=region_alpha,
            )
        else:
            # Render without overlays (backward compatibility)
            png_bytes = render_field_to_png_bytes(env, fields[i], cmap, vmin, vmax, dpi)
        cache_put(i, png_bytes)

    # Generate frame labels
    if frame_labels is None:
        frame_labels = [f"Frame {i + 1}" for i in range(len(fields))]

    # Create persistent figure renderer for efficient on-demand rendering
    # Only create when overlays are present (main optimization target for cache misses)
    persistent_renderer: PersistentFigureRenderer | None = None
    if overlay_data is not None or show_regions:
        persistent_renderer = PersistentFigureRenderer(
            env=env,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            dpi=dpi,
        )

    # On-demand rendering with LRU cache
    def get_frame_bytes(idx: int) -> bytes:
        """Get PNG bytes for frame, using cache or rendering on-demand."""
        if idx in cached_frames:
            # Move to end (mark as recently used)
            cached_frames.move_to_end(idx)
            return cached_frames[idx]
        else:
            # Render on-demand
            if persistent_renderer is not None:
                # Use persistent figure renderer for efficient overlay rendering
                png_bytes = persistent_renderer.render(
                    fields[idx],
                    frame_idx=idx,
                    overlay_data=overlay_data,
                    show_regions=show_regions,
                    region_alpha=region_alpha,
                )
            else:
                # Render without overlays (backward compatibility)
                png_bytes = render_field_to_png_bytes(
                    env, fields[idx], cmap, vmin, vmax, dpi
                )
            cache_put(idx, png_bytes)
            return png_bytes

    # Create persistent Image widget (updated in-place, not re-displayed)
    image_widget = ipywidgets.Image(format="png", width=800)

    # Create persistent HTML widget for frame label
    title_widget = ipywidgets.HTML()

    # Update function that mutates persistent widgets (no display() calls)
    def show_frame(frame_idx: int) -> None:
        """Update frame display by mutating persistent widgets."""
        png_bytes = get_frame_bytes(frame_idx)
        # Set bytes directly to Image widget (no encoding/decoding needed)
        image_widget.value = png_bytes
        # Update title HTML
        title_widget.value = (
            f"<h3 style='text-align: center; margin: 0;'>{frame_labels[frame_idx]}</h3>"
        )

    # Create slider control
    slider = ipywidgets.IntSlider(
        min=0,
        max=len(fields) - 1,
        step=1,
        value=0,
        description="Frame:",
        continuous_update=True,  # Update while dragging for smooth scrubbing
        readout=True,
    )

    # Create play button
    play = ipywidgets.Play(
        interval=int(1000 / max(1, fps)),  # Convert fps to milliseconds
        min=0,
        max=len(fields) - 1,
        step=1,
        value=0,
    )

    # Link play button to slider (JavaScript-level linking for performance)
    # Store reference to prevent garbage collection
    link = ipywidgets.jslink((play, "value"), (slider, "value"))

    # Throttle slider updates to ~30 Hz to avoid decode storms with large images
    _last_update_time = [0.0]  # Use list for mutable capture in closure

    def on_slider_change(change):
        """Update display when slider value changes (throttled to ~30 Hz)."""
        if change["name"] != "value":
            return

        # Throttle updates: skip if less than 1/30 second since last update
        current_time = time.time()
        if current_time - _last_update_time[0] < (1 / 30):
            return  # Skip this update

        _last_update_time[0] = current_time
        show_frame(int(change["new"]))

    slider.observe(on_slider_change, names="value")

    # Initialize with first frame
    show_frame(0)

    # Create container with all widgets
    container = ipywidgets.VBox(
        [
            ipywidgets.HBox([play, slider]),
            title_widget,
            image_widget,
        ]
    )

    # Store link reference on container to prevent garbage collection
    container._links = [link]

    # Display container and return None to prevent auto-display
    display(container)

    return None  # Prevent auto-display (widget already displayed above)
