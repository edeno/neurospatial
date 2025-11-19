"""Jupyter notebook widget backend."""

from __future__ import annotations

import base64
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
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.animation.backends.widget_backend import render_widget
    >>>
    >>> # Create environment
    >>> positions = np.random.randn(100, 2) * 50
    >>> env = Environment.from_samples(positions, bin_size=10.0)
    >>>
    >>> # Create fields
    >>> fields = [np.random.rand(env.n_bins) for _ in range(20)]
    >>>
    >>> # In Jupyter notebook:
    >>> render_widget(env, fields, fps=10)
    >>> # Widget displays automatically with play/pause and slider controls

    Notes
    -----
    **Performance Strategy:**
    - Pre-renders first 500 frames during initialization
    - Remaining frames rendered on-demand with LRU caching (1000 frame limit)
    - Slider updates throttled to ~30 Hz to prevent decode storms
    - Balances responsiveness (pre-cached frames) with memory efficiency

    **Widget Controls:**
    - Play button: Automatic playback at specified FPS
    - Slider: Manual frame scrubbing (continuous_update=True for smooth scrubbing)
    - Frame counter: Shows current frame and total frames
    - Frame label: Displays custom label if provided

    **Memory Considerations:**
    - LRU cache: up to 1000 frames (~30-100 MB depending on DPI)
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

    # LRU cache with configurable size
    cache_limit = 1000  # Max frames to cache
    cached_frames: OrderedDict[int, str] = OrderedDict()

    def cache_put(i: int, b64: str) -> None:
        """Add frame to cache with LRU eviction."""
        cached_frames[i] = b64
        cached_frames.move_to_end(i)
        if len(cached_frames) > cache_limit:
            cached_frames.popitem(last=False)  # Remove oldest

    # Pre-render subset of frames for responsive scrubbing
    initial_cache_size = min(len(fields), 500)
    print(f"Pre-rendering {initial_cache_size} frames for widget...")

    for i in range(initial_cache_size):
        png_bytes = render_field_to_png_bytes(env, fields[i], cmap, vmin, vmax, dpi)
        b64 = base64.b64encode(png_bytes).decode("utf-8")
        cache_put(i, b64)

    # Generate frame labels
    if frame_labels is None:
        frame_labels = [f"Frame {i + 1}" for i in range(len(fields))]

    # On-demand rendering with LRU cache
    def get_frame_b64(idx: int) -> str:
        """Get base64-encoded frame, using cache or rendering on-demand."""
        if idx in cached_frames:
            # Move to end (mark as recently used)
            cached_frames.move_to_end(idx)
            return cached_frames[idx]
        else:
            # Render on-demand
            png_bytes = render_field_to_png_bytes(
                env, fields[idx], cmap, vmin, vmax, dpi
            )
            b64 = base64.b64encode(png_bytes).decode("utf-8")
            cache_put(idx, b64)
            return b64

    # Create persistent Image widget (updated in-place, not re-displayed)
    image_widget = ipywidgets.Image(format="png", width=800)

    # Create persistent HTML widget for frame label
    title_widget = ipywidgets.HTML()

    # Update function that mutates persistent widgets (no display() calls)
    def show_frame(frame_idx: int) -> None:
        """Update frame display by mutating persistent widgets."""
        png_bytes = get_frame_b64(frame_idx)
        # Decode base64 back to bytes for Image widget
        image_widget.value = base64.b64decode(png_bytes)
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
