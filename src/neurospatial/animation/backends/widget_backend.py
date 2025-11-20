"""Jupyter notebook widget backend."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

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
        Environment defining spatial structure
    field : ndarray
        Field values to render
    cmap : str
        Matplotlib colormap name
    vmin, vmax : float
        Color scale limits
    dpi : int
        Resolution for rendering
    frame_idx : int
        Current frame index (for extracting overlay data at this time point)
    overlay_data : OverlayData, optional
        Overlay data structure containing positions, bodyparts, head directions
    show_regions : bool or list of str, default=False
        Whether to show regions. If True, show all regions. If list, show
        only specified region names.
    region_alpha : float, default=0.3
        Alpha transparency for region rendering (0=transparent, 1=opaque)

    Returns
    -------
    png_bytes : bytes
        PNG image data ready for display in widget

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
        # True
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

    # Save to PNG bytes
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)  # Close figure to free memory
    buf.seek(0)
    png_bytes = buf.read()

    return png_bytes


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
        Environment defining spatial structure
    fields : list of arrays
        List of field arrays to animate, each with shape (n_bins,)
    fps : int, default=30
        Frames per second for playback
    cmap : str, default="viridis"
        Matplotlib colormap name
    vmin, vmax : float, optional
        Color scale limits. If None, computed from all fields.
    frame_labels : list of str, optional
        Frame labels (e.g., ["Trial 1", "Trial 2", ...]). If None,
        generates default labels "Frame 1", "Frame 2", etc.
    dpi : int, default=100
        Resolution for frame rendering
    initial_cache_size : int, optional
        Number of frames to pre-render during initialization. If None,
        defaults to min(len(fields), 500). Increase for faster initial
        scrubbing with larger datasets.
    cache_limit : int, default=1000
        Maximum number of frames to keep in LRU cache. Increase for
        larger datasets if memory allows (~30-100 MB per 1000 frames).
    overlay_data : OverlayData, optional
        Overlay data structure containing positions, bodyparts, and head
        directions to render on top of spatial fields. If None, no overlays
        are rendered. Use the conversion funnel in `core.py` to create this
        from user-facing overlay dataclasses.
    show_regions : bool or list of str, default=False
        Whether to show regions overlaid on the animation. If True, shows all
        regions defined in the environment. If a list of strings, shows only
        regions with matching names. If False, no regions are displayed.
    region_alpha : float, default=0.3
        Alpha transparency for region rendering (0=transparent, 1=opaque).
        Only used when show_regions is True or a list of region names.
    **kwargs : dict
        Additional parameters (accepted for backend compatibility, ignored)

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

    # On-demand rendering with LRU cache
    def get_frame_bytes(idx: int) -> bytes:
        """Get PNG bytes for frame, using cache or rendering on-demand."""
        if idx in cached_frames:
            # Move to end (mark as recently used)
            cached_frames.move_to_end(idx)
            return cached_frames[idx]
        else:
            # Render on-demand
            if overlay_data is not None or show_regions:
                # Render with overlays/regions
                png_bytes = render_field_to_png_bytes_with_overlays(
                    env,
                    fields[idx],
                    cmap,
                    vmin,
                    vmax,
                    dpi,
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
