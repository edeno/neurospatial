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
    "render_field_to_image_bytes",
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
    vmin, vmax : tuple of float
        (vmin, vmax) - Minimum and maximum values for color scale

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


def render_field_to_image_bytes(
    env: Environment,
    field: NDArray[np.float64],
    cmap: str,
    vmin: float,
    vmax: float,
    dpi: int = 100,
    image_format: str = "png",
) -> bytes:
    """Render field to image bytes (PNG or JPEG) for HTML embedding.

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
    image_format : {"png", "jpeg"}, default="png"
        Image format. PNG is lossless, JPEG is smaller but lossy.

    Returns
    -------
    image_bytes : bytes
        PNG or JPEG image data

    Raises
    ------
    ValueError
        If image_format is not "png" or "jpeg"
    ImportError
        If image_format="jpeg" but Pillow is not installed

    Notes
    -----
    Uses default bbox (no "tight") to ensure all frames have identical
    dimensions. This is critical for animations and video encoding.

    For JPEG output, uses quality=85 and optimize=True for good
    compression with minimal quality loss. Note that JPEG is most effective
    for large, high-DPI images; for small images (DPI < 75), PNG may actually
    be smaller due to JPEG header overhead.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> positions = np.random.randn(100, 2) * 50
    >>> env = Environment.from_samples(positions, bin_size=10.0)
    >>> field = np.random.rand(env.n_bins)
    >>> png_bytes = render_field_to_image_bytes(
    ...     env, field, "viridis", 0, 1, image_format="png"
    ... )
    >>> png_bytes[:8]  # PNG signature
    b'\\x89PNG\\r\\n\\x1a\\n'
    >>> jpg_bytes = render_field_to_image_bytes(
    ...     env, field, "viridis", 0, 1, image_format="jpeg"
    ... )
    >>> jpg_bytes[:3]  # JPEG signature
    b'\\xff\\xd8\\xff'
    """
    # Validate format early (defense in depth)
    image_format = image_format.lower()
    if image_format not in ("png", "jpeg"):
        raise ValueError(f"image_format must be 'png' or 'jpeg', got '{image_format}'")

    fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)

    env.plot_field(
        field,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        colorbar=False,
    )

    buf = io.BytesIO()

    # IMPORTANT: Do NOT use bbox_inches="tight" - it creates content-dependent
    # crop that yields inconsistent frame sizes across animation sequences.
    # All frames must have identical pixel dimensions for video encoding.
    if image_format.lower() == "jpeg":
        # For JPEG, render to RGB array then save with PIL
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())  # type: ignore[attr-defined]
        rgb = rgba[:, :, :3]  # Drop alpha channel

        # Use PIL for JPEG compression
        try:
            from PIL import Image

            img = Image.fromarray(rgb)
            img.save(buf, format="JPEG", quality=85, optimize=True)
        except ImportError as e:
            plt.close(fig)
            raise ImportError(
                "JPEG support requires Pillow. Install with:\n"
                "  pip install pillow\n"
                "or\n"
                "  uv add pillow\n"
                "\n"
                "Alternatively, use image_format='png' (no dependencies)"
            ) from e
    else:
        # PNG format (lossless)
        fig.savefig(buf, format="png")

    plt.close(fig)

    buf.seek(0)
    return buf.read()


def render_field_to_png_bytes(
    env: Environment,
    field: NDArray[np.float64],
    cmap: str,
    vmin: float,
    vmax: float,
    dpi: int = 100,
) -> bytes:
    """Render field to PNG bytes (for HTML embedding).

    This is a convenience wrapper around render_field_to_image_bytes
    that always outputs PNG format.

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

    Notes
    -----
    Uses default bbox (no "tight") to ensure all frames have identical
    dimensions. This is critical for animations and video encoding.

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
    return render_field_to_image_bytes(
        env, field, cmap, vmin, vmax, dpi, image_format="png"
    )


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

    Raises
    ------
    ValueError
        If field shape does not match env.n_bins

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
    # Validate field shape early for clear error messages
    if len(field) != env.n_bins:
        raise ValueError(
            f"Field has {len(field)} values but environment has {env.n_bins} bins. "
            f"Expected shape: ({env.n_bins},)"
        )

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

            # Transpose from (n_x, n_y, 3) to (n_y, n_x, 3) for napari (y, x) convention
            return np.transpose(full_rgb, (1, 0, 2))
        else:
            # Regular grid without masking
            # Transpose from (n_x, n_y, 3) to (n_y, n_x, 3) for napari (y, x) convention
            return np.transpose(rgb.reshape((*env.layout.grid_shape, 3)), (1, 0, 2))

    # Non-grid layout: return flat RGB for point cloud rendering
    return rgb
