"""Shared rendering utilities for all backends."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from neurospatial.environment.core import Environment


__all__ = [
    "compute_global_colormap_range",
    "field_to_rgb_for_napari",
    "render_field_to_png_bytes",
    "render_field_to_rgb",
]


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
    vmin : float
        Minimum value for color scale
    vmax : float
        Maximum value for color scale

    Examples
    --------
    >>> import numpy as np
    >>> fields = [np.array([0, 1, 2]), np.array([3, 4, 5])]
    >>> vmin, vmax = compute_global_colormap_range(fields)
    >>> vmin, vmax
    (0.0, 5.0)
    """
    # Single-pass min/max computation
    if vmin is None or vmax is None:
        all_min = float("inf")
        all_max = float("-inf")

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
    dpi : int, default=100
        Resolution

    Returns
    -------
    rgb : ndarray, shape (height, width, 3)
        RGB image, uint8

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> positions = np.random.randn(100, 2) * 50
    >>> env = Environment.from_samples(positions, bin_size=10.0)
    >>> field = np.random.rand(env.n_bins)
    >>> rgb = render_field_to_rgb(env, field, "viridis", 0, 1, dpi=50)
    >>> rgb.shape  # doctest: +SKIP
    (height, width, 3)
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

    # Get RGBA buffer - handles retina/HiDPI displays automatically
    # Note: buffer_rgba() is available in FigureCanvas implementations
    rgba_buffer = np.asarray(fig.canvas.buffer_rgba())  # type: ignore[attr-defined]

    # Buffer is already shaped correctly as (height, width, 4)
    # Convert RGBA to RGB by dropping alpha channel
    rgb: NDArray[np.uint8] = rgba_buffer[:, :, :3].copy()

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
    dpi : int, default=100
        Resolution

    Returns
    -------
    png_bytes : bytes
        PNG image data

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> positions = np.random.randn(100, 2) * 50
    >>> env = Environment.from_samples(positions, bin_size=10.0)
    >>> field = np.random.rand(env.n_bins)
    >>> png_bytes = render_field_to_png_bytes(env, field, "viridis", 0, 1)
    >>> png_bytes[:8]  # PNG signature
    b'\\x89PNG\\r\\n\\x1a\\n'
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
    fig.savefig(buf, format="png", bbox_inches="tight")
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

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from neurospatial import Environment
    >>> positions = np.random.randn(100, 2) * 50
    >>> env = Environment.from_samples(positions, bin_size=10.0)
    >>> # Pre-compute colormap lookup
    >>> cmap_obj = plt.get_cmap("viridis")
    >>> cmap_lookup = (cmap_obj(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    >>> field = np.random.rand(env.n_bins)
    >>> rgb = field_to_rgb_for_napari(env, field, cmap_lookup, 0, 1)
    >>> rgb.dtype
    dtype('uint8')
    """
    # Normalize to [0, 1]
    normalized = (field - vmin) / (vmax - vmin)
    normalized = np.clip(normalized, 0, 1)

    # Map to colormap indices [0, 255]
    indices = (normalized * 255).astype(np.uint8)

    # Lookup RGB values
    rgb = cmap_lookup[indices]

    # For grid layouts, reshape to 2D image
    if hasattr(env.layout, "grid_shape") and env.layout.grid_shape is not None:
        # Determine which bins are active
        if hasattr(env.layout, "active_mask") and env.layout.active_mask is not None:
            # Create full grid RGB
            grid_shape = env.layout.grid_shape
            full_rgb = np.zeros((*grid_shape, 3), dtype=np.uint8)

            # Fill active bins
            active_indices = env.layout.active_mask.flatten()
            full_rgb_flat = full_rgb.reshape(-1, 3)
            full_rgb_flat[active_indices] = rgb

            return full_rgb
        else:
            # Regular grid without masking
            return rgb.reshape((*env.layout.grid_shape, 3))

    # Non-grid layout: return flat RGB for point cloud rendering
    return rgb
