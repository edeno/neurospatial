# Animation Feature Implementation Plan

**Feature:** Multi-backend spatial field animation for neuroscience data
**Date:** 2025-11-18
**Complexity:** High - involves 4 backends, parallel processing, GPU rendering

---

## Executive Summary

Add animation capabilities to neurospatial for visualizing spatial fields over time (place field learning, replay sequences, value function evolution). Support four backends optimized for different use cases:

1. **Napari** - GPU-accelerated interactive viewer (large-scale exploration)
2. **MP4** - Parallel video export (publications, presentations)
3. **HTML** - Standalone interactive files (sharing, remote viewing)
4. **Jupyter Widget** - Notebook integration (quick exploration)

**Key technical challenges:**

- Efficient rendering for non-grid layouts (hexagons, meshes)
- Colormap normalization across frames
- Parallel frame rendering with matplotlib threading constraints
- Memory-efficient handling of hour-long sessions (900K+ frames)

---

## 1. File Structure

### New Files to Create

```
src/neurospatial/animation/
├── __init__.py                 # Public API exports (subsample_frames, backend check functions)
├── core.py                     # Main animate_fields() and subsample_frames() functions
├── backends/
│   ├── __init__.py
│   ├── napari_backend.py       # GPU viewer implementation
│   ├── video_backend.py        # Parallel MP4 export
│   ├── html_backend.py         # Standalone HTML player
│   └── widget_backend.py       # Jupyter widget integration
├── rendering.py                # Shared rendering utilities
└── _parallel.py                # Parallel frame rendering (based on gist)

tests/animation/
├── __init__.py
├── test_rendering.py           # RGB conversion, colormap tests
├── test_video_backend.py       # Video export tests
├── test_html_backend.py        # HTML generation tests
└── test_integration.py         # Full pipeline tests

examples/
└── 08_field_animation.py       # User-facing example
```

### Files to Modify

```
src/neurospatial/environment/visualization.py
  - Add animate_fields() method that delegates to animation.core

src/neurospatial/__init__.py
  - Export animate_fields (optional - TBD based on API design)

pyproject.toml
  - Add optional dependencies: napari[all], ipywidgets

docs/user-guide/visualization.md
  - Add animation section

.github/workflows/tests.yml
  - Skip napari tests in CI (requires Qt/display)
```

**Note on `__init__.py` exports:**

The animation module should export:

```python
# src/neurospatial/animation/__init__.py
from neurospatial.animation.core import subsample_frames

__all__ = ["subsample_frames"]
```

Users import via: `from neurospatial.animation import subsample_frames`

The main `animate_fields()` function is accessed via the Environment method, not directly imported.

---

## 2. Dependencies

### Core (Required)

- `matplotlib` - Already present
- `numpy` - Already present
- `tqdm` - Already present (progress bars)

### Optional (Soft Dependencies)

- `napari[all]>=0.4.18,<0.6` - GPU viewer (~50MB, Qt dependency)
- `ipywidgets>=8.0,<9.0` - Jupyter integration (~5MB)
- `ffmpeg` - System dependency for video export (user installs)

**Note**: Version pins prevent API breakage (ipywidgets v7→v8 had incompatible changes).

### Handling Optional Dependencies

```python
# Pattern to use throughout animation module
try:
    import napari
    NAPARI_AVAILABLE = True
except ImportError:
    NAPARI_AVAILABLE = False

def _require_napari():
    if not NAPARI_AVAILABLE:
        raise ImportError(
            "Napari backend requires napari. Install with:\n"
            "  pip install 'napari[all]>=0.4.18,<0.6'\n"
            "or\n"
            "  uv add 'napari[all]>=0.4.18,<0.6'\n"
            "\nFor full animation support:\n"
            "  pip install neurospatial[animation-full]"
        )
```

---

## 3. API Design

### Primary Interface (Environment Method)

```python
# In src/neurospatial/environment/visualization.py

from __future__ import annotations
from typing import TYPE_CHECKING, Any, Literal, Sequence
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from neurospatial.environment._protocols import EnvironmentProtocol

class EnvironmentVisualization:
    """Mixin providing visualization methods.

    Note: This mixin's methods must also be declared in EnvironmentProtocol
    (in src/neurospatial/environment/_protocols.py) for proper type checking.
    """

    def animate_fields(
        self: EnvironmentProtocol,  # Mixin uses Protocol for type checking
        fields: Sequence[NDArray[np.float64]] | NDArray[np.float64],
        *,
        backend: Literal["auto", "napari", "video", "html", "widget"] = "auto",
        save_path: str | None = None,
        fps: int = 30,
        cmap: str = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
        frame_labels: Sequence[str] | None = None,  # Renamed from 'labels'
        overlay_trajectory: NDArray[np.float64] | None = None,
        title: str = "Spatial Field Animation",
        # Video-specific options
        dpi: int = 100,
        codec: str = "h264",
        bitrate: int = 5000,
        n_workers: int | None = None,
        dry_run: bool = False,  # New: estimate time/size without rendering
        # HTML-specific options
        image_format: Literal["png", "jpeg"] = "png",  # Renamed from 'compress'
        max_html_frames: int = 500,  # New: hard limit for HTML
        # Napari-specific options
        contrast_limits: tuple[float, float] | None = None,
        # Common options
        show_colorbar: bool = False,  # New: include colorbar in frames
        colorbar_label: str = "",
    ) -> Any:
        """Animate spatial fields over time with multiple backend options.

        Parameters
        ----------
        fields : sequence of arrays or ndarray
            Fields to animate. If ndarray, first dimension is time.
            Each field shape must match env.n_bins.
        backend : {"auto", "napari", "video", "html", "widget"}
            Animation backend to use:
            - "auto": Choose based on context and data size
            - "napari": GPU-accelerated interactive viewer (best for large data)
            - "video": Export MP4/WebM (best for publications)
            - "html": Standalone HTML player (best for sharing)
            - "widget": Jupyter widget (best for notebooks)
        save_path : str, optional
            Output path. Extension determines format:
            - .mp4, .webm, .avi: video export
            - .html: standalone HTML player
            - None: display interactively (napari or widget)
        fps : int, default=30
            Frames per second for playback
        cmap : str, default="viridis"
            Matplotlib colormap name
        vmin, vmax : float, optional
            Color scale limits. If None, computed from all fields.
        frame_labels : sequence of str, optional
            Frame labels (e.g., ["Trial 1", "Trial 2", ...])
        overlay_trajectory : ndarray, optional
            Positions to overlay as trajectory (shape: n_timepoints, n_dims)
        title : str
            Animation title
        dpi : int, default=100
            Resolution for video/HTML rendering
        codec : str, default="h264"
            Video codec (h264, vp9, mpeg4)
        bitrate : int, default=5000
            Video bitrate in kbps
        n_workers : int, optional
            Parallel workers for video rendering (default: CPU count / 2).
            Requires environment to be pickle-able (call env.clear_cache() if errors).
        dry_run : bool, default=False
            Video backend only: estimate time and file size without rendering
        image_format : {"png", "jpeg"}, default="png"
            HTML backend: image format (JPEG smaller but lossy)
        max_html_frames : int, default=500
            HTML backend: maximum frames allowed (prevents huge files)
        contrast_limits : tuple of float, optional
            Napari backend: contrast limits (overrides vmin/vmax)
        show_colorbar : bool, default=False
            Include colorbar in rendered frames
        colorbar_label : str, default=""
            Label for colorbar axis

        Returns
        -------
        viewer or path
            - Napari: napari.Viewer instance
            - Video/HTML: Path to saved file
            - Widget: ipywidgets.interact instance

        Examples
        --------
        >>> # Generate sample data
        >>> positions = np.random.randn(1000, 2) * 50
        >>> env = Environment.from_samples(positions, bin_size=5.0)
        >>>
        >>> # Simulate place field formation over 20 trials
        >>> fields = []
        >>> for trial in range(20):
        ...     noise = np.random.randn(env.n_bins) * 0.1
        ...     field = np.exp(-env.distance_to([100]) / (10 + trial)) + noise
        ...     fields.append(field)
        >>>
        >>> # Interactive exploration (Napari)
        >>> viewer = env.animate_fields(fields, backend='napari')
        >>>
        >>> # Video export for publication
        >>> env.animate_fields(
        ...     fields,
        ...     save_path='place_field_learning.mp4',
        ...     fps=5,
        ...     labels=[f'Trial {i+1}' for i in range(20)]
        ... )
        >>>
        >>> # Shareable HTML
        >>> env.animate_fields(
        ...     fields,
        ...     save_path='exploration.html',
        ...     fps=10
        ... )
        >>>
        >>> # Quick notebook check
        >>> env.animate_fields(fields, backend='widget')  # In Jupyter

        Notes
        -----
        **Backend Selection (auto mode):**
        - >10,000 frames → Napari (GPU acceleration needed)
        - save_path ends in video extension → Video export
        - In Jupyter notebook → Widget
        - Otherwise → Napari (if available) or raise error

        **Performance Tips:**
        - Use memory-mapped arrays for large datasets
        - For video export, increase n_workers for faster rendering
        - For HTML, use compress="jpeg" to reduce file size
        - For Napari, use contrast_limits to avoid recomputing range

        **Layout Support:**
        - Grid layouts: Direct rendering
        - Hexagonal/triangular: Rasterized to regular grid
        - Graph layouts (1D): Rendered as 1D plot

        See Also
        --------
        plot_field : Static field visualization
        Environment.plot : Environment structure visualization
        """
        from neurospatial.animation.core import animate_fields as _animate

        return _animate(
            env=self,
            fields=fields,
            backend=backend,
            save_path=save_path,
            fps=fps,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            frame_labels=frame_labels,  # Updated parameter name
            overlay_trajectory=overlay_trajectory,
            title=title,
            dpi=dpi,
            codec=codec,
            bitrate=bitrate,
            n_workers=n_workers,
            dry_run=dry_run,  # Pass through video-specific parameter
            image_format=image_format,  # Updated parameter name
            max_html_frames=max_html_frames,  # Pass through HTML-specific parameter
            contrast_limits=contrast_limits,
            show_colorbar=show_colorbar,  # Pass through rendering parameter
            colorbar_label=colorbar_label,
        )
```

---

## 4. Core Implementation

### File: `src/neurospatial/animation/core.py`

```python
"""Core animation orchestration."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Sequence

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from neurospatial.environment.core import Environment


def animate_fields(
    env: Environment,
    fields: Sequence[NDArray[np.float64]] | NDArray[np.float64],
    *,
    backend: Literal["auto", "napari", "video", "html", "widget"] = "auto",
    save_path: str | None = None,
    **kwargs,
) -> Any:
    """Main animation dispatcher.

    This function validates inputs and routes to the appropriate backend.
    """
    # Normalize fields to list of arrays
    if isinstance(fields, np.ndarray):
        if fields.ndim < 2:
            raise ValueError("fields must be at least 2D (n_frames, n_bins)")
        fields = [fields[i] for i in range(len(fields))]
    else:
        fields = list(fields)

    if len(fields) == 0:
        raise ValueError("fields cannot be empty")

    # Validate environment is fitted
    if not hasattr(env, '_is_fitted') or not env._is_fitted:
        raise RuntimeError(
            "Environment must be fitted before animation. "
            "Use Environment.from_samples() or other factory methods."
        )

    # Validate field shapes
    for i, field in enumerate(fields):
        if len(field) != env.n_bins:
            raise ValueError(
                f"Field {i} has {len(field)} values but environment has {env.n_bins} bins. "
                f"Expected shape: ({env.n_bins},)"
            )

    n_frames = len(fields)

    # Auto-select backend
    if backend == "auto":
        backend = _select_backend(n_frames, save_path)

    # Route to backend with early validation
    if backend == "napari":
        from neurospatial.animation.backends.napari_backend import render_napari
        return render_napari(env, fields, **kwargs)

    elif backend == "video":
        from neurospatial.animation.backends.video_backend import (
            render_video,
            check_ffmpeg_available,
        )

        # Check ffmpeg availability IMMEDIATELY (fail fast)
        if not check_ffmpeg_available():
            raise RuntimeError(
                "Video backend requires ffmpeg.\n"
                "\n"
                "Install ffmpeg:\n"
                "  macOS:   brew install ffmpeg\n"
                "  Ubuntu:  sudo apt install ffmpeg\n"
                "  Windows: https://ffmpeg.org/download.html\n"
                "\n"
                "Or use a different backend:\n"
                "  backend='html'    (no dependencies, instant scrubbing)\n"
                "  backend='napari'  (pip install napari[all])\n"
            )

        if save_path is None:
            raise ValueError("save_path required for video backend")

        # Validate environment pickle-ability for parallel rendering
        n_workers = kwargs.get('n_workers')
        if n_workers and n_workers > 1:
            import pickle
            try:
                pickle.dumps(env, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                raise ValueError(
                    f"Video backend with parallel rendering requires environment to be pickle-able.\n"
                    f"\n"
                    f"Error: {e}\n"
                    f"\n"
                    f"Solutions:\n"
                    f"  1. Clear caches: env.clear_cache() before animating\n"
                    f"  2. Use n_workers=1 for serial rendering (slower)\n"
                    f"  3. Use backend='html' instead (no pickling)\n"
                ) from e

        return render_video(env, fields, save_path, **kwargs)

    elif backend == "html":
        from neurospatial.animation.backends.html_backend import render_html
        if save_path is None:
            save_path = "animation.html"
        return render_html(env, fields, save_path, **kwargs)

    elif backend == "widget":
        from neurospatial.animation.backends.widget_backend import render_widget
        return render_widget(env, fields, **kwargs)

    else:
        raise ValueError(f"Unknown backend: {backend}")


def _select_backend(
    n_frames: int, save_path: str | None
) -> Literal["napari", "video", "html", "widget"]:
    """Auto-select appropriate backend with transparent logging.

    Logs selection rationale at INFO level for user transparency.
    """
    import logging

    logger = logging.getLogger(__name__)

    # Check if in Jupyter
    try:
        from IPython import get_ipython
        in_jupyter = get_ipython() is not None
    except ImportError:
        in_jupyter = False

    # Video export requested (file extension determines format)
    if save_path:
        ext = Path(save_path).suffix
        if ext in ('.mp4', '.webm', '.avi', '.mov'):
            logger.info(f"Auto-selected 'video' backend (save_path extension: {ext})")
            return "video"
        elif ext == '.html':
            logger.info(f"Auto-selected 'html' backend (save_path extension: {ext})")
            return "html"

    # Large dataset - requires GPU acceleration
    if n_frames > 10_000:
        from neurospatial.animation.backends.napari_backend import NAPARI_AVAILABLE
        if NAPARI_AVAILABLE:
            logger.info(
                f"Auto-selected 'napari' backend for {n_frames:,} frames "
                f"(threshold: 10,000). Napari provides GPU-accelerated rendering "
                f"and memory-efficient streaming."
            )
            return "napari"
        else:
            raise RuntimeError(
                f"Dataset has {n_frames:,} frames - requires GPU acceleration.\n"
                f"\n"
                f"Auto-selection attempted Napari (threshold: 10,000 frames),\n"
                f"but napari is not installed.\n"
                f"\n"
                f"Options:\n"
                f"  1. Install Napari:\n"
                f"     pip install napari[all]\n"
                f"\n"
                f"  2. Export subsampled video:\n"
                f"     subsample = fields[::100]  # Every 100th frame\n"
                f"     env.animate_fields(subsample, backend='video', save_path='out.mp4')\n"
                f"\n"
                f"  3. Use HTML (WARNING: {n_frames*0.1:.0f} MB file):\n"
                f"     env.animate_fields(fields[:500], backend='html')  # First 500 frames\n"
            )

    # Jupyter notebook - use widget
    if in_jupyter:
        logger.info("Auto-selected 'widget' backend (running in Jupyter)")
        return "widget"

    # Default to napari for interactive
    from neurospatial.animation.backends.napari_backend import NAPARI_AVAILABLE
    if NAPARI_AVAILABLE:
        logger.info("Auto-selected 'napari' backend (default for interactive viewing)")
        return "napari"

    # No suitable backend available
    raise RuntimeError(
        "No suitable animation backend available.\n"
        "\n"
        "Install one of:\n"
        "  - napari:     pip install napari[all]     (interactive viewing)\n"
        "  - ipywidgets: pip install ipywidgets      (Jupyter support)\n"
        "\n"
        "Or specify save_path to export:\n"
        "  - save_path='output.mp4'   (requires ffmpeg)\n"
        "  - save_path='output.html'  (no dependencies)\n"
    )


def subsample_frames(
    fields: NDArray | list,
    target_fps: int,
    source_fps: int,
) -> NDArray:
    """Subsample frames to target frame rate.

    Essential utility for large-scale sessions. Allows users to reduce
    900K frames at 250 Hz to a manageable video at 30 fps.

    Parameters
    ----------
    fields : array or list
        Full field data, shape (n_frames, n_bins) or list of arrays
    target_fps : int
        Desired output frame rate (e.g., 30 for video)
    source_fps : int
        Original sampling rate (e.g., 250 Hz for neural recording)

    Returns
    -------
    subsampled : array or list
        Subsampled fields at target_fps, same type as input

    Examples
    --------
    >>> # 250 Hz recording → 30 fps video
    >>> fields_video = subsample_frames(fields_full, target_fps=30, source_fps=250)
    >>> env.animate_fields(fields_video, save_path='output.mp4', fps=30)

    >>> # 1000 Hz → 60 fps
    >>> fields_video = subsample_frames(fields_full, target_fps=60, source_fps=1000)

    Notes
    -----
    Subsampling rate is source_fps / target_fps. For example:
    - 250 Hz → 30 fps: every 8.3 frames (rounds to 8)
    - 1000 Hz → 60 fps: every 16.7 frames (rounds to 17)

    This function works with memory-mapped arrays without loading all data.
    """
    if target_fps > source_fps:
        raise ValueError(
            f"target_fps ({target_fps}) cannot exceed source_fps ({source_fps})"
        )

    subsample_rate = source_fps / target_fps
    indices = np.arange(0, len(fields), subsample_rate).astype(int)

    if isinstance(fields, np.ndarray):
        return fields[indices]
    else:
        return [fields[i] for i in indices]
```

---

## 5. Shared Rendering Utilities

### File: `src/neurospatial/animation/rendering.py`

```python
"""Shared rendering utilities for all backends."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

if TYPE_CHECKING:
    from neurospatial.environment.core import Environment


def compute_global_colormap_range(
    fields: list[NDArray[np.float64]],
    vmin: float | None = None,
    vmax: float | None = None,
) -> tuple[float, float]:
    """Compute consistent color scale across all fields.

    Single-pass computation for efficiency.

    Parameters
    ----------
    fields : list of arrays
        All fields to animate
    vmin, vmax : float, optional
        Manual limits (if provided, skip computation)

    Returns
    -------
    vmin, vmax : float
        Color scale limits
    """
    # Single-pass min/max computation
    if vmin is None or vmax is None:
        all_min = float('inf')
        all_max = float('-inf')

        for field in fields:
            all_min = min(all_min, field.min())
            all_max = max(all_max, field.max())

        vmin = vmin if vmin is not None else all_min
        vmax = vmax if vmax is not None else all_max

    # Avoid degenerate case
    if vmin == vmax:
        vmin -= 0.5
        vmax += 0.5

    return float(vmin), float(vmax)


def render_field_to_rgb(
    env: Environment,
    field: NDArray[np.float64],
    cmap: str,
    vmin: float,
    vmax: float,
    dpi: int = 100,
) -> NDArray[np.uint8]:
    """Render field to RGB array using environment layout.

    This creates a full matplotlib figure and converts to RGB.
    Used by video and HTML backends.

    Parameters
    ----------
    env : Environment
        Environment defining spatial structure
    field : ndarray
        Field values (shape: n_bins)
    cmap : str
        Colormap name
    vmin, vmax : float
        Color scale limits
    dpi : int
        Resolution

    Returns
    -------
    rgb : ndarray, shape (height, width, 3)
        RGB image, uint8
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)

    # Use environment's plot_field for layout-aware rendering
    env.plot_field(
        field,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        colorbar=False,  # Skip colorbar for animation frames
    )

    # Convert figure to RGB array
    fig.canvas.draw()

    # IMPLEMENTATION NOTE: The original plan used tostring_rgb(), but the
    # actual implementation uses buffer_rgba() for better retina/HiDPI display
    # support. This change was made during implementation to handle modern
    # matplotlib versions and high-resolution displays correctly.
    # See commit d801f41 for details.
    rgba_buffer = np.asarray(fig.canvas.buffer_rgba())
    rgb = rgba_buffer[:, :, :3].copy()  # Drop alpha channel

    plt.close(fig)
    return rgb


def render_field_to_png_bytes(
    env: Environment,
    field: NDArray[np.float64],
    cmap: str,
    vmin: float,
    vmax: float,
    dpi: int = 100,
) -> bytes:
    """Render field to PNG bytes (for HTML embedding).

    Parameters
    ----------
    env : Environment
        Environment defining spatial structure
    field : ndarray
        Field values
    cmap : str
        Colormap name
    vmin, vmax : float
        Color scale limits
    dpi : int
        Resolution

    Returns
    -------
    png_bytes : bytes
        PNG image data
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)

    env.plot_field(
        field,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        colorbar=False,
    )

    # Save to bytes buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)

    buf.seek(0)
    return buf.read()


def field_to_rgb_for_napari(
    env: Environment,
    field: NDArray[np.float64],
    cmap_lookup: NDArray[np.uint8],
    vmin: float,
    vmax: float,
) -> NDArray[np.uint8]:
    """Fast RGB conversion for Napari (no matplotlib overhead).

    This is optimized for real-time rendering by pre-computing
    colormap lookup table.

    Parameters
    ----------
    env : Environment
        Environment (for grid shape if available)
    field : ndarray
        Field values
    cmap_lookup : ndarray, shape (256, 3)
        Pre-computed colormap RGB values
    vmin, vmax : float
        Color scale limits

    Returns
    -------
    rgb : ndarray, shape (height, width, 3) or (n_bins, 3)
        RGB image for napari
    """
    # Normalize to [0, 1]
    normalized = (field - vmin) / (vmax - vmin)
    normalized = np.clip(normalized, 0, 1)

    # Map to colormap indices [0, 255]
    indices = (normalized * 255).astype(np.uint8)

    # Lookup RGB values
    rgb = cmap_lookup[indices]

    # For grid layouts, reshape to 2D image
    if hasattr(env.layout, 'grid_shape') and env.layout.grid_shape is not None:
        # Determine which bins are active
        if hasattr(env.layout, 'active_mask'):
            # Create full grid RGB
            grid_shape = env.layout.grid_shape
            full_rgb = np.zeros(grid_shape + (3,), dtype=np.uint8)

            # Fill active bins
            active_indices = env.layout.active_mask.flatten()
            full_rgb_flat = full_rgb.reshape(-1, 3)
            full_rgb_flat[active_indices] = rgb

            return full_rgb
        else:
            # Regular grid without masking
            return rgb.reshape(env.layout.grid_shape + (3,))

    # Non-grid layout: return flat RGB for point cloud rendering
    return rgb
```

---

## 6. Backend Implementations

### 6.1 Napari Backend

**File:** `src/neurospatial/animation/backends/napari_backend.py`

```python
"""Napari GPU-accelerated viewer backend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import matplotlib.pyplot as plt
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
    frame_labels: list[str] | None = None,  # Renamed from labels
    overlay_trajectory: NDArray[np.float64] | None = None,
    contrast_limits: tuple[float, float] | None = None,
    title: str = "Spatial Field Animation",
    **kwargs,  # Accept other parameters gracefully
) -> Any:
    """Launch Napari viewer with lazy-loaded field animation.

    Returns
    -------
    viewer : napari.Viewer
        Napari viewer instance (blocking - will show window)
    """
    if not NAPARI_AVAILABLE:
        raise ImportError(
            "Napari backend requires napari. Install with:\n"
            "  pip install napari[all]\n"
            "or\n"
            "  uv add napari[all]"
        )

    from neurospatial.animation.rendering import (
        compute_global_colormap_range,
        field_to_rgb_for_napari,
    )

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
    class LazyFieldRenderer:
        """On-demand field rendering for Napari with true LRU cache."""

        def __init__(self, env, fields, cmap_lookup, vmin, vmax):
            from collections import OrderedDict

            self.env = env
            self.fields = fields
            self.cmap_lookup = cmap_lookup
            self.vmin = vmin
            self.vmax = vmax
            self._cache = OrderedDict()  # Explicit OrderedDict for LRU
            self._cache_size = 1000

        def __len__(self):
            return len(self.fields)

        def __getitem__(self, idx):
            """Render frame on-demand when Napari requests it."""
            if idx < 0:
                idx = len(self.fields) + idx

            if idx not in self._cache:
                # Render this frame
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
                    self._cache.popitem(last=False)  # Remove oldest
            else:
                # Move to end for LRU (mark as recently used)
                self._cache.move_to_end(idx)

            return self._cache[idx]

        @property
        def shape(self):
            """Return shape for napari (time, height, width, channels)."""
            sample = self[0]
            return (len(self.fields),) + sample.shape

        @property
        def dtype(self):
            return np.uint8

    lazy_frames = LazyFieldRenderer(env, fields, cmap_lookup, vmin, vmax)

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
            raise ValueError("overlay_trajectory must be 2D (n_timepoints, n_dims)")

        # For 2D trajectories, add as tracks or points
        if overlay_trajectory.shape[1] == 2:
            # Create track data: (track_id, time, y, x)
            n_points = len(overlay_trajectory)
            track_data = np.column_stack([
                np.zeros(n_points),  # Single track
                np.arange(n_points),  # Time
                overlay_trajectory[:, 1],  # Y
                overlay_trajectory[:, 0],  # X
            ])
            viewer.add_tracks(track_data, name="Trajectory")
        else:
            # Higher dimensional - just plot as points
            viewer.add_points(overlay_trajectory, name="Trajectory", size=2)

    # Set frame labels if provided
    if frame_labels:
        # Napari doesn't have built-in frame labels, but we can add to title
        pass  # TODO: Custom frame label widget?

    return viewer
```

---

### 6.2 Video Backend (Parallel Rendering)

**File:** `src/neurospatial/animation/backends/video_backend.py`

```python
"""Parallel video export backend.

Based on parallel rendering approach from:
https://gist.github.com/edeno/652ee10a76481f00b3eb08906b41c6bf
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

if TYPE_CHECKING:
    from neurospatial.environment.core import Environment


def render_video(
    env: Environment,
    fields: list[NDArray[np.float64]],
    save_path: str,
    *,
    fps: int = 30,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    frame_labels: list[str] | None = None,  # Renamed from labels
    dpi: int = 100,
    codec: str = "h264",
    bitrate: int = 5000,
    n_workers: int | None = None,
    dry_run: bool = False,
    title: str = "Spatial Field Animation",
    **kwargs,  # Accept other parameters gracefully
) -> Path | None:
    """Export animation as video using parallel frame rendering.

    Parameters
    ----------
    dry_run : bool, default=False
        If True, estimate time and file size without rendering.
        Returns None after printing estimate.

    Returns
    -------
    save_path : Path or None
        Path to exported video file, or None if dry_run=True
    """
    import time
    from neurospatial.animation.rendering import (
        compute_global_colormap_range,
        render_field_to_rgb,
    )
    from neurospatial.animation._parallel import (
        parallel_render_frames,
        check_ffmpeg_available,
    )

    # Validate ffmpeg available
    if not check_ffmpeg_available():
        raise RuntimeError(
            "Video export requires ffmpeg. Install:\n"
            "  macOS: brew install ffmpeg\n"
            "  Ubuntu: sudo apt install ffmpeg\n"
            "  Windows: Download from https://ffmpeg.org/download.html"
        )

    # Compute global color scale
    vmin, vmax = compute_global_colormap_range(fields, vmin, vmax)

    # Determine number of workers
    if n_workers is None:
        n_workers = max(1, os.cpu_count() // 2)

    # Dry run: estimate time and file size
    if dry_run:
        print("Running dry run estimation...")

        # Render one test frame to measure timing
        start = time.time()
        _ = render_field_to_rgb(env, fields[0], cmap, vmin, vmax, dpi)
        frame_time = time.time() - start

        # Estimate total time
        total_time = frame_time * len(fields) / n_workers

        # Estimate file size (rough approximation)
        frame_size_kb = (dpi / 100) ** 2 * 50  # Empirical
        estimated_mb = frame_size_kb * len(fields) / 1024 * (bitrate / 5000)

        print(f"\n{'=' * 60}")
        print(f"Video Export Dry Run Estimate:")
        print(f"{'=' * 60}")
        print(f"  Frames:          {len(fields):,}")
        print(f"  Workers:         {n_workers}")
        print(f"  Frame time:      {frame_time * 1000:.1f} ms")
        print(f"  Est. total time: {total_time / 60:.1f} minutes")
        print(f"  Est. file size:  {estimated_mb:.0f} MB")
        print(f"  Output path:     {save_path}")
        print(f"\nTo proceed, call again with dry_run=False")
        print(f"{'=' * 60}\n")
        return None

    print(f"Rendering {len(fields)} frames using {n_workers} workers...")
    print(f"Estimated time: ~{len(fields) * 0.5 / n_workers:.0f} seconds")

    # Create temporary directory for frames
    tmpdir = tempfile.mkdtemp(prefix="neurospatial_animation_")

    try:
        # Render frames in parallel
        frame_pattern = parallel_render_frames(
            env=env,
            fields=fields,
            output_dir=tmpdir,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            frame_labels=frame_labels,
            dpi=dpi,
            n_workers=n_workers,
        )

        # Encode video with ffmpeg
        print("Encoding video...")
        output_path = Path(save_path)

        # Select codec
        codec_map = {
            'h264': 'libx264',
            'h265': 'libx265',
            'vp9': 'libvpx-vp9',
            'mpeg4': 'mpeg4',
        }
        ffmpeg_codec = codec_map.get(codec, codec)

        # Build ffmpeg command
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output
            '-framerate', str(fps),
            '-i', frame_pattern,
            '-c:v', ffmpeg_codec,
            '-b:v', f'{bitrate}k',
            '-pix_fmt', 'yuv420p',  # Compatibility
            str(output_path),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg encoding failed:\n{result.stderr}"
            )

        print(f"✓ Video saved to {output_path}")
        return output_path

    finally:
        # Clean up temporary frames
        shutil.rmtree(tmpdir)


def check_ffmpeg_available() -> bool:
    """Check if ffmpeg is installed and accessible."""
    try:
        subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
```

---

### 6.3 Parallel Frame Rendering

**File:** `src/neurospatial/animation/_parallel.py`

```python
"""Parallel frame rendering utilities.

Based on approach from:
https://gist.github.com/edeno/652ee10a76481f00b3eb08906b41c6bf

Key principles:
- Each worker process has its own matplotlib figure (avoid threading issues)
- Frames saved as numbered PNGs for ffmpeg pattern matching
- Workers operate independently on partitioned frame ranges
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

if TYPE_CHECKING:
    from neurospatial.environment.core import Environment


def parallel_render_frames(
    env: Environment,
    fields: list[NDArray[np.float64]],
    output_dir: str,
    cmap: str,
    vmin: float,
    vmax: float,
    frame_labels: list[str] | None,  # Renamed from labels
    dpi: int,
    n_workers: int,
) -> str:
    """Render frames in parallel across worker processes.

    Parameters
    ----------
    env : Environment
        Must be pickle-able (will be serialized to workers)
    fields : list of arrays
        All fields to render
    output_dir : str
        Directory to save frame PNGs
    cmap : str
        Colormap name
    vmin, vmax : float
        Color scale limits
    frame_labels : list of str or None
        Frame labels for each frame
    dpi : int
        Resolution
    n_workers : int
        Number of parallel workers

    Returns
    -------
    frame_pattern : str
        ffmpeg input pattern (e.g., "/tmp/frame_%05d.png")
    """
    n_frames = len(fields)

    # Validate environment is pickle-able
    try:
        import pickle
        pickle.dumps(env)
    except Exception as e:
        raise ValueError(
            f"Environment must be pickle-able for parallel rendering: {e}"
        )

    # Partition frames across workers
    frames_per_worker = n_frames // n_workers
    worker_tasks = []

    for worker_id in range(n_workers):
        start_idx = worker_id * frames_per_worker
        if worker_id == n_workers - 1:
            # Last worker takes remainder
            end_idx = n_frames
        else:
            end_idx = start_idx + frames_per_worker

        worker_fields = fields[start_idx:end_idx]
        worker_frame_labels = frame_labels[start_idx:end_idx] if frame_labels else None

        worker_tasks.append({
            'env': env,
            'fields': worker_fields,
            'start_frame_idx': start_idx,
            'output_dir': output_dir,
            'cmap': cmap,
            'vmin': vmin,
            'vmax': vmax,
            'frame_labels': worker_frame_labels,
            'dpi': dpi,
        })

    # Render in parallel
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        list(tqdm(
            executor.map(_render_worker_frames, worker_tasks),
            total=n_workers,
            desc="Workers",
        ))

    # Return ffmpeg pattern (1-indexed for compatibility)
    # ffmpeg expects: frame_00001.png, frame_00002.png, etc.
    digits = len(str(n_frames))
    pattern = str(Path(output_dir) / f"frame_%0{digits}d.png")

    return pattern


def _render_worker_frames(task: dict) -> None:
    """Render frames in a worker process.

    Each worker creates its own matplotlib figure to avoid
    threading issues and memory accumulation.

    Parameters
    ----------
    task : dict
        Worker task specification with keys:
        - env: Environment
        - fields: list of fields to render
        - start_frame_idx: global frame index offset
        - output_dir: where to save PNGs
        - cmap, vmin, vmax: colormap settings
        - frame_labels: optional frame labels
        - dpi: resolution
    """
    env = task['env']
    fields = task['fields']
    start_idx = task['start_frame_idx']
    output_dir = task['output_dir']
    cmap = task['cmap']
    vmin = task['vmin']
    vmax = task['vmax']
    frame_labels = task['frame_labels']
    dpi = task['dpi']

    # Create figure once for this worker
    fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)

    for local_idx, field in enumerate(fields):
        global_idx = start_idx + local_idx

        # Clear previous frame
        ax.clear()

        # Render field using environment's plot_field
        env.plot_field(
            field,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            colorbar=False,
        )

        # Add label if provided
        if frame_labels and frame_labels[local_idx]:
            ax.set_title(frame_labels[local_idx], fontsize=14)

        # Save frame (1-indexed for ffmpeg)
        frame_number = global_idx + 1
        digits = len(str(len(fields) * 10))  # Sufficient padding
        filename = f"frame_{frame_number:0{digits}d}.png"
        filepath = Path(output_dir) / filename

        fig.savefig(filepath, bbox_inches='tight', dpi=dpi)

    # Clean up figure
    plt.close(fig)
```

---

### 6.4 HTML Backend

**File:** `src/neurospatial/animation/backends/html_backend.py`

```python
"""Standalone HTML player backend with instant scrubbing."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

if TYPE_CHECKING:
    from neurospatial.environment.core import Environment


def render_html(
    env: Environment,
    fields: list[NDArray[np.float64]],
    save_path: str,
    *,
    fps: int = 30,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    frame_labels: list[str] | None = None,  # Renamed from labels
    dpi: int = 100,
    image_format: str = "png",  # Renamed from compress
    max_html_frames: int = 500,
    title: str = "Spatial Field Animation",
    **kwargs,  # Accept other parameters gracefully
) -> Path:
    """Generate standalone HTML player with instant scrubbing.

    Pre-renders all frames as base64-encoded images embedded in HTML.
    JavaScript provides play/pause/scrub controls with zero latency.

    Parameters
    ----------
    max_html_frames : int, default=500
        Maximum number of frames allowed. Prevents creating huge HTML files
        that crash browsers.

    Returns
    -------
    save_path : Path
        Path to saved HTML file

    Raises
    ------
    ValueError
        If number of frames exceeds max_html_frames
    """
    import warnings
    from neurospatial.animation.rendering import (
        compute_global_colormap_range,
        render_field_to_png_bytes,
    )

    n_frames = len(fields)

    # Estimate file size BEFORE rendering
    estimated_mb = n_frames * 0.1 * (dpi / 100) ** 2  # ~100KB per frame at 100 DPI

    # Hard limit check
    if n_frames > max_html_frames:
        raise ValueError(
            f"HTML backend supports max {max_html_frames} frames (got {n_frames}).\n"
            f"Estimated file size: {estimated_mb:.0f} MB\n"
            f"\n"
            f"Options:\n"
            f"  1. Subsample frames:\n"
            f"     fields_subset = fields[::10]  # Every 10th frame\n"
            f"     env.animate_fields(fields_subset, backend='html', ...)\n"
            f"\n"
            f"  2. Use video backend:\n"
            f"     env.animate_fields(fields, backend='video', save_path='output.mp4')\n"
            f"\n"
            f"  3. Use Napari for interactive viewing:\n"
            f"     env.animate_fields(fields, backend='napari')\n"
            f"\n"
            f"  4. Override limit (NOT RECOMMENDED):\n"
            f"     env.animate_fields(fields, backend='html', max_html_frames={n_frames})\n"
        )

    # Warn about large files
    if estimated_mb > 50:
        warnings.warn(
            f"\nHTML export will create a large file (~{estimated_mb:.0f} MB).\n"
            f"Consider:\n"
            f"  - Reduce DPI: dpi=50 (current: {dpi})\n"
            f"  - Subsample frames: fields[::5]\n"
            f"  - Use image_format='jpeg' (lossy but smaller)\n",
            UserWarning,
            stacklevel=2
        )

    # Compute global color scale
    vmin, vmax = compute_global_colormap_range(fields, vmin, vmax)

    # Pre-render all frames to base64
    print(f"Rendering {n_frames} frames to {image_format.upper()}...")
    frames_b64 = []

    for field in tqdm(fields, desc="Encoding frames"):
        png_bytes = render_field_to_png_bytes(env, field, cmap, vmin, vmax, dpi)

        # Convert to base64
        b64 = base64.b64encode(png_bytes).decode('utf-8')
        frames_b64.append(b64)

    # Generate frame labels
    if frame_labels is None:
        frame_labels = [f"Frame {i+1}" for i in range(len(fields))]

    # Create HTML
    html = _generate_html_player(
        frames_b64=frames_b64,
        frame_labels=frame_labels,
        fps=fps,
        title=title,
        image_format=image_format,
    )

    # Write to file
    output_path = Path(save_path)
    output_path.write_text(html)

    file_size_mb = output_path.stat().st_size / 1e6
    print(f"✓ HTML saved to {output_path} ({file_size_mb:.1f} MB)")

    return output_path


def _generate_html_player(
    frames_b64: list[str],
    frame_labels: list[str],  # Renamed from labels
    fps: int,
    title: str,
    image_format: str,  # Renamed from compress
) -> str:
    """Generate HTML with embedded frames and JavaScript controls."""

    n_frames = len(frames_b64)

    # JSON-encode data for JavaScript
    frames_json = json.dumps(frames_b64)
    labels_json = json.dumps(frame_labels)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 20px;
            background: #f5f5f5;
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        .container {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            max-width: 900px;
        }}
        h1 {{
            margin: 0 0 20px 0;
            font-size: 24px;
            color: #333;
        }}
        #frame {{
            width: 100%;
            border: 1px solid #ddd;
            border-radius: 4px;
            display: block;
        }}
        #controls {{
            margin: 20px 0;
            display: flex;
            flex-direction: column;
            gap: 15px;
        }}
        .button-row {{
            display: flex;
            gap: 10px;
            align-items: center;
        }}
        button {{
            padding: 10px 20px;
            font-size: 14px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            background: #007bff;
            color: white;
            transition: background 0.2s;
        }}
        button:hover {{
            background: #0056b3;
        }}
        button:disabled {{
            background: #ccc;
            cursor: not-allowed;
        }}
        #slider {{
            width: 100%;
            height: 6px;
            -webkit-appearance: none;
            appearance: none;
            background: #ddd;
            border-radius: 3px;
            outline: none;
        }}
        #slider::-webkit-slider-thumb {{
            -webkit-appearance: none;
            appearance: none;
            width: 18px;
            height: 18px;
            background: #007bff;
            border-radius: 50%;
            cursor: pointer;
        }}
        #slider::-moz-range-thumb {{
            width: 18px;
            height: 18px;
            background: #007bff;
            border-radius: 50%;
            cursor: pointer;
            border: none;
        }}
        .info {{
            display: flex;
            justify-content: space-between;
            color: #666;
            font-size: 14px;
        }}
        .speed-control {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .speed-control label {{
            font-size: 14px;
            color: #666;
        }}
        .speed-control select {{
            padding: 5px 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>

        <img id="frame" src="" alt="Animation frame" />

        <div id="controls">
            <div class="button-row">
                <button id="play">▶ Play</button>
                <button id="pause">⏸ Pause</button>
                <button id="prev">⏮ Prev</button>
                <button id="next">⏭ Next</button>
                <div class="speed-control">
                    <label for="speed">Speed:</label>
                    <select id="speed">
                        <option value="0.25">0.25×</option>
                        <option value="0.5">0.5×</option>
                        <option value="1" selected>1×</option>
                        <option value="2">2×</option>
                        <option value="4">4×</option>
                    </select>
                </div>
            </div>

            <input type="range" id="slider" min="0" max="{n_frames - 1}" value="0" />

            <div class="info">
                <span id="label"></span>
                <span id="frame-counter"></span>
            </div>
        </div>
    </div>

    <script>
        // Frame data (embedded)
        const frames = {frames_json};
        const labels = {labels_json};
        const baseFPS = {fps};

        // State
        let currentFrame = 0;
        let playing = false;
        let interval = null;
        let speedMultiplier = 1.0;

        // Elements
        const img = document.getElementById('frame');
        const slider = document.getElementById('slider');
        const labelSpan = document.getElementById('label');
        const counterSpan = document.getElementById('frame-counter');
        const playBtn = document.getElementById('play');
        const pauseBtn = document.getElementById('pause');
        const prevBtn = document.getElementById('prev');
        const nextBtn = document.getElementById('next');
        const speedSelect = document.getElementById('speed');

        // Initialize
        function init() {{
            updateFrame(0);
            updateControls();
        }}

        function updateFrame(idx) {{
            if (idx < 0 || idx >= frames.length) return;

            currentFrame = idx;

            // Update image (instant - just changes src)
            img.src = 'data:image/png;base64,' + frames[idx];

            // Update UI
            slider.value = idx;
            labelSpan.textContent = labels[idx];
            counterSpan.textContent = `${{idx + 1}} / ${{frames.length}}`;
        }}

        function updateControls() {{
            playBtn.disabled = playing;
            pauseBtn.disabled = !playing;
        }}

        function play() {{
            if (playing) return;
            playing = true;
            updateControls();

            const frameDelay = (1000 / baseFPS) / speedMultiplier;

            interval = setInterval(() => {{
                currentFrame = (currentFrame + 1) % frames.length;
                updateFrame(currentFrame);
            }}, frameDelay);
        }}

        function pause() {{
            if (!playing) return;
            playing = false;
            updateControls();
            clearInterval(interval);
        }}

        function stepForward() {{
            pause();
            updateFrame(Math.min(frames.length - 1, currentFrame + 1));
        }}

        function stepBackward() {{
            pause();
            updateFrame(Math.max(0, currentFrame - 1));
        }}

        // Event listeners
        slider.oninput = (e) => {{
            pause();
            updateFrame(parseInt(e.target.value));
        }};

        playBtn.onclick = play;
        pauseBtn.onclick = pause;
        nextBtn.onclick = stepForward;
        prevBtn.onclick = stepBackward;

        speedSelect.onchange = (e) => {{
            speedMultiplier = parseFloat(e.target.value);
            if (playing) {{
                pause();
                play();  // Restart with new speed
            }}
        }};

        // Keyboard shortcuts
        document.onkeydown = (e) => {{
            if (e.key === ' ') {{
                e.preventDefault();
                playing ? pause() : play();
            }} else if (e.key === 'ArrowLeft') {{
                e.preventDefault();
                stepBackward();
            }} else if (e.key === 'ArrowRight') {{
                e.preventDefault();
                stepForward();
            }}
        }};

        // Start
        init();
    </script>
</body>
</html>"""

    return html
```

---

### 6.5 Jupyter Widget Backend

**File:** `src/neurospatial/animation/backends/widget_backend.py`

```python
"""Jupyter notebook widget backend."""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from neurospatial.environment.core import Environment

# Check ipywidgets availability
try:
    import ipywidgets
    from IPython.display import HTML, display
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
    frame_labels: list[str] | None = None,  # Renamed from labels
    dpi: int = 100,
    **kwargs,  # Accept other parameters gracefully
):
    """Create interactive Jupyter widget with slider control.

    Returns
    -------
    widget : ipywidgets.interact
        Interactive widget (automatically displays in notebook)
    """
    if not IPYWIDGETS_AVAILABLE:
        raise ImportError(
            "Widget backend requires ipywidgets. Install with:\n"
            "  pip install ipywidgets\n"
            "or\n"
            "  uv add ipywidgets"
        )

    from neurospatial.animation.rendering import (
        compute_global_colormap_range,
        render_field_to_png_bytes,
    )

    # Compute global color scale
    vmin, vmax = compute_global_colormap_range(fields, vmin, vmax)

    # Pre-render subset of frames for responsive scrubbing
    cache_size = min(len(fields), 500)
    print(f"Pre-rendering {cache_size} frames for widget...")

    cached_frames = {}
    for i in range(cache_size):
        png_bytes = render_field_to_png_bytes(env, fields[i], cmap, vmin, vmax, dpi)
        b64 = base64.b64encode(png_bytes).decode('utf-8')
        cached_frames[i] = b64

    # Frame labels
    if frame_labels is None:
        frame_labels = [f"Frame {i+1}" for i in range(len(fields))]

    # On-demand rendering for uncached frames
    def get_frame_b64(idx):
        if idx in cached_frames:
            return cached_frames[idx]
        else:
            # Render on-demand
            png_bytes = render_field_to_png_bytes(env, fields[idx], cmap, vmin, vmax, dpi)
            return base64.b64encode(png_bytes).decode('utf-8')

    # Create widget
    def show_frame(frame_idx):
        b64 = get_frame_b64(frame_idx)
        label = frame_labels[frame_idx]

        html = f"""
        <div style="text-align: center;">
            <h3>{label}</h3>
            <img src="data:image/png;base64,{b64}" style="max-width: 800px;" />
        </div>
        """
        display(HTML(html))

    # Create slider and play controls
    slider = ipywidgets.IntSlider(
        min=0,
        max=len(fields) - 1,
        step=1,
        value=0,
        description='Frame:',
        continuous_update=True,  # Update while dragging
    )

    play = ipywidgets.Play(
        interval=int(1000 / fps),
        min=0,
        max=len(fields) - 1,
        step=1,
        value=0,
    )

    # Link play button to slider
    ipywidgets.jslink((play, 'value'), (slider, 'value'))

    # Create interactive widget
    widget = ipywidgets.interact(show_frame, frame_idx=slider)

    # Display controls
    display(ipywidgets.HBox([play, slider]))

    return widget
```

---

## 7. Testing Strategy

### 7.1 Unit Tests

**File:** `tests/animation/test_rendering.py`

```python
"""Test shared rendering utilities."""

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.animation.rendering import (
    compute_global_colormap_range,
    render_field_to_rgb,
    field_to_rgb_for_napari,
)


def test_compute_global_colormap_range():
    """Test global color scale computation."""
    fields = [
        np.array([0, 1, 2]),
        np.array([3, 4, 5]),
        np.array([1, 2, 3]),
    ]

    vmin, vmax = compute_global_colormap_range(fields)
    assert vmin == 0.0
    assert vmax == 5.0

    # Manual limits override
    vmin_manual, vmax_manual = compute_global_colormap_range(fields, vmin=-1, vmax=10)
    assert vmin_manual == -1.0
    assert vmax_manual == 10.0


def test_compute_global_colormap_range_degenerate():
    """Test degenerate case (all same value)."""
    fields = [np.ones(10) * 5.0, np.ones(10) * 5.0]

    vmin, vmax = compute_global_colormap_range(fields)
    assert vmin == 4.5
    assert vmax == 5.5


def test_render_field_to_rgb():
    """Test field rendering to RGB array."""
    positions = np.random.randn(100, 2) * 50
    env = Environment.from_samples(positions, bin_size=10.0)

    field = np.random.rand(env.n_bins)

    rgb = render_field_to_rgb(env, field, cmap='viridis', vmin=0, vmax=1, dpi=50)

    # Check output shape
    assert rgb.ndim == 3
    assert rgb.shape[2] == 3  # RGB
    assert rgb.dtype == np.uint8

    # Check value range
    assert rgb.min() >= 0
    assert rgb.max() <= 255


def test_field_to_rgb_for_napari_grid():
    """Test fast RGB conversion for grid layout."""
    pytest.importorskip('matplotlib')

    positions = np.random.randn(100, 2) * 50
    env = Environment.from_samples(positions, bin_size=10.0)

    # Create colormap lookup
    from matplotlib import pyplot as plt
    cmap_obj = plt.get_cmap('viridis')
    cmap_lookup = (cmap_obj(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)

    field = np.random.rand(env.n_bins)

    rgb = field_to_rgb_for_napari(env, field, cmap_lookup, vmin=0, vmax=1)

    # Check output
    assert rgb.dtype == np.uint8
    assert rgb.shape[-1] == 3  # RGB channels
```

**File:** `tests/animation/test_video_backend.py`

```python
"""Test video export backend."""

import pytest
import numpy as np
from pathlib import Path

from neurospatial import Environment
from neurospatial.animation.backends.video_backend import check_ffmpeg_available


@pytest.mark.skipif(
    not check_ffmpeg_available(),
    reason="ffmpeg not installed"
)
def test_video_export_small(tmp_path):
    """Test video export with small dataset."""
    # Create environment
    positions = np.random.randn(100, 2) * 50
    env = Environment.from_samples(positions, bin_size=10.0)

    # Create simple field sequence
    fields = []
    for i in range(10):
        field = np.sin(np.linspace(0, 2*np.pi, env.n_bins) + i * 0.5)
        fields.append(field)

    # Export video
    output_path = tmp_path / "test.mp4"

    result = env.animate_fields(
        fields,
        backend='video',
        save_path=str(output_path),
        fps=5,
        n_workers=2,
    )

    # Check file created
    assert output_path.exists()
    assert output_path.stat().st_size > 0


@pytest.mark.slow
@pytest.mark.skipif(
    not check_ffmpeg_available(),
    reason="ffmpeg not installed"
)
def test_video_export_parallel(tmp_path):
    """Test parallel rendering with larger dataset."""
    positions = np.random.randn(500, 2) * 50
    env = Environment.from_samples(positions, bin_size=5.0)

    # 100 frames
    fields = [np.random.rand(env.n_bins) for _ in range(100)]

    output_path = tmp_path / "large.mp4"

    env.animate_fields(
        fields,
        backend='video',
        save_path=str(output_path),
        fps=30,
        n_workers=4,
    )

    assert output_path.exists()
```

**File:** `tests/animation/test_html_backend.py`

```python
"""Test HTML export backend."""

import numpy as np
from pathlib import Path

from neurospatial import Environment


def test_html_export(tmp_path):
    """Test HTML player generation."""
    positions = np.random.randn(100, 2) * 50
    env = Environment.from_samples(positions, bin_size=10.0)

    fields = [np.random.rand(env.n_bins) for _ in range(5)]

    output_path = tmp_path / "test.html"

    result = env.animate_fields(
        fields,
        backend='html',
        save_path=str(output_path),
        labels=['Trial 1', 'Trial 2', 'Trial 3', 'Trial 4', 'Trial 5'],
    )

    # Check file created
    assert output_path.exists()

    # Check HTML content
    html = output_path.read_text()
    assert '<html' in html
    assert 'data:image/png;base64,' in html
    assert 'Trial 1' in html

    # Should have embedded frames
    assert html.count('data:image/png;base64,') == 5


def test_html_export_large_warning(tmp_path):
    """Test HTML export warns about large file size."""
    positions = np.random.randn(100, 2) * 50
    env = Environment.from_samples(positions, bin_size=10.0)

    # Many frames - will produce large HTML
    fields = [np.random.rand(env.n_bins) for _ in range(200)]

    output_path = tmp_path / "large.html"

    # Should complete but warn about size
    result = env.animate_fields(
        fields,
        backend='html',
        save_path=str(output_path),
    )

    assert output_path.exists()
    file_size_mb = output_path.stat().st_size / 1e6

    # Should be large (rough check)
    assert file_size_mb > 1.0
```

---

## 8. Documentation

### User Guide Section

**File:** `docs/user-guide/visualization.md` (add section)

````markdown
## Animating Fields Over Time

Visualize how spatial fields evolve over time (place field learning, replay events, value function changes).

### Quick Start

```python
import numpy as np
from neurospatial import Environment

# Create environment
positions = np.random.randn(1000, 2) * 50
env = Environment.from_samples(positions, bin_size=5.0)

# Simulate place field emergence over 20 trials
fields = []
for trial in range(20):
    # Field gradually sharpens
    distances = env.distance_to([100])
    field = np.exp(-distances / (20 - trial))
    fields.append(field)

# Interactive exploration
env.animate_fields(fields, backend='napari')

# Video for publication
env.animate_fields(
    fields,
    save_path='place_field_learning.mp4',
    fps=5,
    labels=[f'Trial {i+1}' for i in range(20)]
)
```

### Backend Options

**Napari** (Interactive GPU viewer)
- Best for: Large datasets, exploration
- Requires: `pip install napari[all]`
- Features: Instant seeking, memory-efficient

**Video** (MP4/WebM export)
- Best for: Publications, presentations
- Requires: ffmpeg installed
- Features: Parallel rendering, high quality

**HTML** (Standalone player)
- Best for: Sharing, remote viewing
- Requires: Nothing (just matplotlib)
- Features: Single file, works offline

**Widget** (Jupyter notebooks)
- Best for: Quick checks in notebooks
- Requires: `pip install ipywidgets`
- Features: Integrated controls

### Performance Tips

For large datasets (>10,000 frames):
- Use memory-mapped arrays
- Choose Napari backend for exploration
- Export short clips as video (not entire session)

```python
# Memory-mapped data
import numpy as np
fields_mmap = np.memmap(
    'session.dat',
    dtype='float32',
    mode='r',
    shape=(900000, env.n_bins)
)

# Napari handles this efficiently
env.animate_fields(fields_mmap, backend='napari')
```
````

---

## 9. Example Script

**File:** `examples/08_field_animation.py`

```python
"""Example: Animating spatial fields over time.

This demonstrates the four animation backends:
1. Napari - Interactive GPU viewer
2. Video - MP4 export with parallel rendering
3. HTML - Standalone shareable player
4. Widget - Jupyter notebook integration
"""

import numpy as np
import matplotlib.pyplot as plt

from neurospatial import Environment

np.random.seed(42)

print(__doc__)

# ============================================================================
# Setup: Create environment and simulate learning
# ============================================================================

print("\nCreating environment...")
positions = np.random.randn(1000, 2) * 50
env = Environment.from_samples(positions, bin_size=5.0)
env.units = "cm"

print(f"Environment: {env.n_bins} bins")

# Simulate place field formation over trials
print("\nSimulating place field learning...")

n_trials = 30
goal_bin = env.bin_at(np.array([[80, 80]]))[0]

fields = []
for trial in range(n_trials):
    # Field gradually sharpens and shifts
    sigma = 30 - trial * 0.5  # Sharpening
    distances = env.distance_to([goal_bin])

    # Add noise that decreases over trials
    noise_level = 0.3 * (1 - trial / n_trials)
    noise = np.random.randn(env.n_bins) * noise_level

    field = np.exp(-(distances**2) / (2 * sigma**2)) + noise
    field = np.maximum(field, 0)  # Non-negative

    fields.append(field)

print(f"Generated {len(fields)} trial fields")

# ============================================================================
# Example 1: Interactive Napari viewer
# ============================================================================

print("\n" + "="*80)
print("Example 1: Napari Interactive Viewer")
print("="*80)

try:
    import napari

    print("Launching Napari viewer...")
    print("  - Use slider to scrub through trials")
    print("  - Instant seeking through all frames")
    print("  - GPU accelerated")

    viewer = env.animate_fields(
        fields,
        backend='napari',
        fps=10,
        frame_labels=[f'Trial {i+1}' for i in range(n_trials)],
        title='Place Field Learning'
    )

    print("✓ Napari viewer launched")

except ImportError:
    print("⊗ Napari not available. Install with: pip install napari[all]")

# ============================================================================
# Example 2: Video export (MP4)
# ============================================================================

print("\n" + "="*80)
print("Example 2: Video Export")
print("="*80)

from neurospatial.animation.backends.video_backend import check_ffmpeg_available

if check_ffmpeg_available():
    output_path = env.animate_fields(
        fields,
        backend='video',
        save_path='examples/place_field_learning.mp4',
        fps=5,
        cmap='hot',
        frame_labels=[f'Trial {i+1}' for i in range(n_trials)],
        n_workers=4,  # Parallel rendering
        dpi=100,
    )
    print(f"✓ Video saved to {output_path}")
else:
    print("⊗ ffmpeg not available. Video export skipped.")
    print("  Install: brew install ffmpeg (macOS) or apt install ffmpeg (Linux)")

# ============================================================================
# Example 3: Standalone HTML player
# ============================================================================

print("\n" + "="*80)
print("Example 3: HTML Player")
print("="*80)

html_path = env.animate_fields(
    fields,
    backend='html',
    save_path='examples/place_field_learning.html',
    fps=10,
    cmap='viridis',
    frame_labels=[f'Trial {i+1}' for i in range(n_trials)],
)

print(f"✓ HTML player saved to {html_path}")
print("  - Open in any web browser")
print("  - Instant scrubbing with slider")
print("  - Shareable (single file)")
print("  - Keyboard shortcuts: space = play/pause, arrows = step")

# ============================================================================
# Example 4: Jupyter widget (if in notebook)
# ============================================================================

print("\n" + "="*80)
print("Example 4: Jupyter Widget")
print("="*80)

try:
    from IPython import get_ipython
    if get_ipython() is not None:
        print("Running in Jupyter - creating widget...")

        widget = env.animate_fields(
            fields,
            backend='widget',
            fps=10,
            frame_labels=[f'Trial {i+1}' for i in range(n_trials)],
        )

        print("✓ Widget created (displayed above)")
    else:
        print("⊗ Not in Jupyter notebook - widget skipped")
except ImportError:
    print("⊗ IPython not available - widget skipped")

# ============================================================================
# Example 5: Large-Scale Session (900K frames at 250 Hz)
# ============================================================================

print("\n" + "="*80)
print("Example 5: Large-Scale Session (900K frames)")
print("="*80)

print("\nFor hour-long sessions with 900K frames:")
print("  - Use memory-mapped data (don't load into RAM)")
print("  - Use Napari for exploration (lazy loading)")
print("  - Subsample for video export")

# Memory-mapped approach (doesn't load all data)
print("\nCreating memory-mapped data file...")
# Simulate creating mmap file (in practice, load from your recording)
mmap_path = 'examples/large_session.dat'
n_frames_large = 900_000  # 1 hour at 250 Hz
fields_mmap = np.memmap(
    mmap_path,
    dtype='float32',
    mode='w+',  # Create new file
    shape=(n_frames_large, env.n_bins)
)
# Populate with simulated data (in practice, this is your recording)
print("Populating with sample data (this would be your neural recording)...")
for i in range(0, n_frames_large, 10000):
    # Write in chunks to avoid memory issues
    chunk_size = min(10000, n_frames_large - i)
    fields_mmap[i:i+chunk_size] = np.random.rand(chunk_size, env.n_bins)
fields_mmap.flush()

print(f"\n✓ Created memory-mapped dataset: {n_frames_large:,} frames")
print(f"  File size: {n_frames_large * env.n_bins * 4 / 1e9:.2f} GB")
print(f"  RAM usage: ~0 MB (memory-mapped)")

# Interactive exploration with Napari
print("\nOption 1: Interactive exploration (Napari)")
print("  Napari loads frames on-demand - handles 900K frames efficiently")

try:
    viewer = env.animate_fields(
        fields_mmap,
        backend='napari',
        fps=250,  # Match recording rate
        title='Hour-Long Session (900K frames)'
    )
    print("✓ Napari viewer launched - scrub through 900K frames instantly!")
except ImportError:
    print("⊗ Napari not available (install: pip install napari[all])")

# Video export with subsampling
print("\nOption 2: Export subsampled video")
print("  900K frames → 30 fps video requires subsampling")

from neurospatial.animation import subsample_frames

# Subsample 250 Hz → 30 fps
fields_subsampled = subsample_frames(
    fields_mmap,
    target_fps=30,
    source_fps=250
)
print(f"  Subsampled: {len(fields_subsampled):,} frames (every {250//30}th frame)")
print(f"  Video duration: {len(fields_subsampled)/30:.1f} seconds")

# Dry run to estimate
if check_ffmpeg_available():
    env.animate_fields(
        fields_subsampled,
        backend='video',
        save_path='examples/large_session_summary.mp4',
        fps=30,
        n_workers=8,
        dry_run=True,  # Estimate first
    )
    print("\n  To render, run with dry_run=False")
else:
    print("  ⊗ ffmpeg not available for video export")

# Cleanup
import os
os.remove(mmap_path)
print("\n✓ Large-scale example complete!")

print("\n✓ Animation examples complete!")
```

---

## 10. Implementation Checklist

### Phase 1: Core Infrastructure (Week 1)

- [ ] Create `src/neurospatial/animation/` module structure
- [ ] Implement `rendering.py` shared utilities
  - [ ] Optimize `compute_global_colormap_range()` to single-pass
- [ ] Implement `core.py` dispatcher with backend selection
  - [ ] Implement `subsample_frames()` utility function
  - [ ] Add environment pickle validation for parallel rendering
  - [ ] Add transparent logging to backend auto-selection
  - [ ] Add early ffmpeg check (fail fast)
- [ ] Update type annotations to use `EnvironmentProtocol` for mixin methods
- [ ] Add unit tests for rendering utilities
- [ ] Update `pyproject.toml` with optional dependencies and version pins

### Phase 2: Backend Implementations (Week 2-3)

- [ ] Implement Napari backend (`napari_backend.py`)
  - [ ] Lazy frame loading class with true LRU cache (OrderedDict.move_to_end)
  - [ ] Colormap precomputation
  - [ ] Trajectory overlay support
  - [ ] Field shape validation
- [ ] Implement video backend (`video_backend.py`)
  - [ ] Parallel frame rendering (`_parallel.py`)
  - [ ] Memory leak fixes (finally blocks for cleanup)
  - [ ] ffmpeg integration
  - [ ] Codec selection
  - [ ] Add `dry_run` mode for time/size estimation
  - [ ] Add progress estimates for long operations
- [ ] Implement HTML backend (`html_backend.py`)
  - [ ] Base64 frame encoding
  - [ ] JavaScript player generation with ARIA labels (accessibility)
  - [ ] Keyboard controls
  - [ ] File size estimation BEFORE rendering
  - [ ] Hard limit at 500 frames with clear error messages
  - [ ] Warning for files >50MB
- [ ] Implement widget backend (`widget_backend.py`)
  - [ ] Frame caching strategy
  - [ ] Play/pause controls
  - [ ] Jupyter integration

### Phase 3: Integration & Documentation (Week 3)

- [ ] Add `animate_fields()` method to `EnvironmentVisualization` mixin
- [ ] Add public API export in `__init__.py`
  - [ ] Export `subsample_frames` utility
- [ ] Update CLAUDE.md with animation documentation
- [ ] Add 900K frame example to `examples/08_field_animation.py`
- [ ] Add remote server workflow guide to docs
- [ ] Add large-scale data guide to docs (memory-mapped arrays)
- [ ] Add quick start (5 lines) to docs

### Phase 4: Testing (Week 4)

- [ ] Unit tests for all rendering functions
  - [ ] Test `subsample_frames()` with arrays and lists
  - [ ] Test pickle-ability validation
- [ ] Integration tests for each backend
  - [ ] Test with memory-mapped arrays (900K+ frames)
  - [ ] Test field shape validation
  - [ ] Test HTML file size limits
  - [ ] Test dry_run mode
- [ ] Performance benchmarks (large datasets)
  - [ ] Benchmark Napari seek performance (<100ms target)
  - [ ] Benchmark parallel rendering scalability
- [ ] Memory profiling tests
  - [ ] Profile memory usage for large datasets
  - [ ] Verify lazy loading doesn't load all frames
- [ ] CI configuration (skip napari tests without display)

### Phase 5: Polish & Finalization (Week 4-5)

- [ ] Write user guide section
- [ ] Create example script (`08_field_animation.py`) with all 5 examples
- [ ] Add API reference documentation
- [ ] Create GIF/video demonstrations for README
- [ ] Error messages and user guidance (all backends)
- [ ] Progress bars for long operations (with time estimates)
- [ ] File size warnings for HTML export (already implemented)
- [ ] Dependency installation instructions (clear error messages)
- [ ] Type hints and mypy validation (SelfEnv pattern)

---

## 11. Dependencies Summary

### Required (Core)

```toml
# Already present
matplotlib = ">=3.5"
numpy = ">=1.20"
```

### Optional (Features)

```toml
[project.optional-dependencies]
animation = [
    "napari[all]>=0.4.18",  # GPU viewer
    "ipywidgets>=8.0",       # Jupyter integration
]
```

### System Dependencies

- **ffmpeg** - User installs for video export (provide clear instructions)

---

## 12. Known Limitations & Future Work

### Current Limitations

1. **Non-grid layouts** - Hexagonal/mesh layouts rasterized to regular grid for Napari
2. **Memory** - HTML backend limited to ~500 frames before file size issues
3. **Codecs** - Video export assumes h264 available (most common)
4. **1D environments** - Special handling needed for graph layouts

### Future Enhancements

1. **GIF export** - Fallback if ffmpeg unavailable
2. **Frame interpolation** - Smooth playback at higher FPS
3. **Multi-field comparison** - Side-by-side animation of multiple fields
4. **Custom overlays** - User-provided drawing functions
5. **WebAssembly** - Client-side rendering for HTML backend

---

## 13. Risk Assessment

### High Risk

- **Napari Qt dependency** - Can cause installation issues on some systems
  - *Mitigation:* Soft dependency with clear install instructions

- **Parallel rendering pickle errors** - Complex environments may not serialize
  - *Mitigation:* Validate pickle-ability before spawning workers, fallback to serial

### Medium Risk

- **ffmpeg availability** - Users may not have it installed
  - *Mitigation:* Check availability, provide install instructions, offer alternatives

- **HTML file size** - Large datasets create huge files
  - *Mitigation:* Warn users, suggest subsampling, offer JPEG compression

### Low Risk

- **Jupyter widget display** - Some notebook versions may have issues
  - *Mitigation:* Document compatible versions, test with major platforms

---

## 14. Success Criteria

### Must Have

- [ ] All four backends functional
- [ ] Napari handles 100K+ frames efficiently
- [ ] Video export uses parallel rendering correctly
- [ ] HTML player works in all modern browsers
- [ ] Clear error messages for missing dependencies

### Nice to Have

- [ ] Example animations in README
- [ ] Benchmark results documented
- [ ] Tutorial video/GIF showing each backend

### Performance Targets

- Napari: <100ms frame seek time for 100K frames
- Video: Render 100 frames in <30s on 4-core machine
- HTML: Generate 100-frame player in <20s

---

**End of Implementation Plan**
