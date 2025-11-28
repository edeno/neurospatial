"""Shared rendering utilities for all backends."""

from __future__ import annotations

import io
import warnings
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from neurospatial.animation._timing import timing

if TYPE_CHECKING:
    from neurospatial.environment.core import Environment


__all__ = [
    "_validate_frame_labels",
    "compute_global_colormap_range",
    "field_to_rgb_for_napari",
    "render_field_to_image_bytes",
    "render_field_to_png_bytes",
    "render_field_to_rgb",
]


def compute_global_colormap_range(
    fields: list[NDArray[np.float64]] | NDArray[np.float64],
    vmin: float | None = None,
    vmax: float | None = None,
) -> tuple[float, float]:
    """Compute NaN-robust color scale across all fields.

    Vectorized computation for efficiency. Ignores NaN and infinite values
    when computing the range. Automatically stacks list inputs for ~10x
    faster processing on large datasets.

    Parameters
    ----------
    fields : list of ndarray or ndarray
        All fields to animate. Can be:
        - List of 1D arrays, each with shape (n_bins,)
        - 2D array with shape (n_frames, n_bins)
    vmin : float, optional
        Manual minimum limit. If provided, skips computation for minimum.
        Useful for consistent colormaps across multiple animations.
    vmax : float, optional
        Manual maximum limit. If provided, skips computation for maximum.
        Useful for consistent colormaps across multiple animations.

    Returns
    -------
    vmin : float
        Minimum value for color scale.
    vmax : float
        Maximum value for color scale.
        Returns (0.0, 1.0) if all values are NaN/inf or fields is empty.

    Notes
    -----
    For large datasets (e.g., 41k frames), vectorized stacking provides
    ~10x speedup over per-field iteration:
    - List iteration: ~265ms
    - Stacked array: ~22ms

    Examples
    --------
    >>> import numpy as np
    >>> fields = [np.array([0, 1, 2]), np.array([3, 4, 5])]
    >>> vmin, vmax = compute_global_colormap_range(fields)
    >>> vmin, vmax
    (0.0, 5.0)

    Also accepts 2D arrays directly:
    >>> fields_2d = np.array([[0, 1, 2], [3, 4, 5]])
    >>> vmin, vmax = compute_global_colormap_range(fields_2d)
    >>> vmin, vmax
    (0.0, 5.0)
    """
    # Vectorized min/max computation with NaN-robustness
    if vmin is None or vmax is None:
        # Handle empty input
        if len(fields) == 0:
            return (0.0, 1.0)

        # Try to stack list into array for vectorized operations (~10x faster)
        # Falls back to iteration if shapes don't match (rare edge case)
        stacked: NDArray[np.float64] | None = None
        if isinstance(fields, list):
            # Check if all fields have the same shape for stacking
            first_shape = fields[0].shape
            if all(f.shape == first_shape for f in fields):
                # Stack into 2D array (n_frames, n_bins)
                stacked = np.stack(fields, axis=0)
        else:
            # Already an array
            stacked = fields

        # Suppress RuntimeWarning from nanmin/nanmax on all-NaN arrays
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "All-NaN slice encountered")

            if stacked is not None:
                # Fast path: mask out NaN and inf, then take min/max
                # np.isfinite filters both NaN and inf in one vectorized pass
                finite_mask = np.isfinite(stacked)
                if finite_mask.any():
                    finite_values = stacked[finite_mask]
                    all_min = float(finite_values.min())
                    all_max = float(finite_values.max())
                else:
                    # All NaN/inf - will be handled by degenerate case below
                    all_min = float("inf")
                    all_max = float("-inf")
            else:
                # Fallback: iterate when shapes don't match
                all_min = float("inf")
                all_max = float("-inf")
                for field in fields:
                    field_min = float(np.nanmin(field))
                    field_max = float(np.nanmax(field))
                    if np.isfinite(field_min):
                        all_min = min(all_min, field_min)
                    if np.isfinite(field_max):
                        all_max = max(all_max, field_max)

        vmin = vmin if vmin is not None else all_min
        vmax = vmax if vmax is not None else all_max

    # Handle degenerate cases
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        # All NaN/inf or single value: use safe default
        if vmin == vmax and np.isfinite(vmin):
            vmin -= 0.5
            vmax += 0.5
        else:
            vmin, vmax = 0.0, 1.0

    return float(vmin), float(vmax)


def render_field_to_rgb(
    env: Environment,
    field: NDArray[np.float64],
    cmap: str,
    vmin: float,
    vmax: float,
    dpi: int = 100,
    scale_bar: bool | Any = False,  # bool | ScaleBarConfig
) -> NDArray[np.uint8]:
    """Render field to RGB array using environment layout.

    This creates a full matplotlib figure and converts to RGB.
    Used by video and HTML backends.

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
    dpi : int, default=100
        Figure resolution in dots per inch.
    scale_bar : bool or ScaleBarConfig, default=False
        Whether to add a scale bar. If True, uses default config.
        If ScaleBarConfig, uses provided configuration.

    Returns
    -------
    rgb : ndarray of shape (height, width, 3), dtype uint8
        RGB image array with values in range [0, 255].

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
    with timing("render_field_to_rgb"):
        fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)

        # Use environment's plot_field for layout-aware rendering
        env.plot_field(
            field,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            colorbar=False,  # Skip colorbar for animation frames
            scale_bar=scale_bar,
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
    scale_bar: bool | Any = False,  # bool | ScaleBarConfig
) -> bytes:
    """Render field to image bytes (PNG or JPEG) for HTML embedding.

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
    dpi : int, default=100
        Figure resolution in dots per inch.
    image_format : {"png", "jpeg"}, default="png"
        Image format. PNG is lossless, JPEG is smaller but lossy.
    scale_bar : bool or ScaleBarConfig, default=False
        Whether to add a scale bar. If True, uses default config.
        If ScaleBarConfig, uses provided configuration.

    Returns
    -------
    image_bytes : bytes
        Encoded image data (PNG or JPEG format).

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
        scale_bar=scale_bar,
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
    scale_bar: bool | Any = False,  # bool | ScaleBarConfig
) -> bytes:
    """Render field to PNG bytes (for HTML embedding).

    This is a convenience wrapper around render_field_to_image_bytes
    that always outputs PNG format.

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
    dpi : int, default=100
        Figure resolution in dots per inch.
    scale_bar : bool or ScaleBarConfig, default=False
        Whether to add a scale bar. If True, uses default config.
        If ScaleBarConfig, uses provided configuration.

    Returns
    -------
    png_bytes : bytes
        PNG-encoded image data.

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
        env, field, cmap, vmin, vmax, dpi, image_format="png", scale_bar=scale_bar
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
        Environment defining spatial structure (for grid shape if available).
    field : ndarray of shape (n_bins,), dtype float64
        Field values to render. Length must match env.n_bins.
    cmap_lookup : ndarray of shape (256, 3), dtype uint8
        Pre-computed colormap RGB lookup table with values in range [0, 255].
    vmin : float
        Minimum value for color scale normalization.
    vmax : float
        Maximum value for color scale normalization.

    Returns
    -------
    rgb : ndarray of shape (height, width, 3) or (n_bins, 3), dtype uint8
        RGB image for napari. For 2D grid layouts, returns (height, width, 3)
        in napari coordinate convention (y, x). For non-grid layouts, returns
        (n_bins, 3) for point cloud rendering.

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

    # Normalize to [0, 1], guarding against zero-range (constant field or vmin == vmax)
    value_range = vmax - vmin
    if value_range <= 0 or not np.isfinite(value_range):
        # Zero or invalid range: map all values to middle of colormap (index 127)
        normalized = np.full_like(field, 0.5)
    else:
        normalized = (field - vmin) / value_range
        normalized = np.clip(normalized, 0, 1)

    # Map to colormap indices [0, 255]
    indices = (normalized * 255).astype(np.uint8)

    # Lookup RGB values
    rgb = cmap_lookup[indices]

    # For grid-compatible 2D layouts, reshape to 2D image
    # Use is_grid_compatible property for type checking, then verify 2D grid_shape
    is_grid = getattr(env.layout, "is_grid_compatible", False)
    has_2d_grid = (
        hasattr(env.layout, "grid_shape")
        and env.layout.grid_shape is not None
        and len(env.layout.grid_shape) == 2
    )
    if is_grid and has_2d_grid:
        # Determine which bins are active
        grid_shape = env.layout.grid_shape
        assert grid_shape is not None  # for mypy
        rgb_shape = (grid_shape[0], grid_shape[1], 3)
        if hasattr(env.layout, "active_mask") and env.layout.active_mask is not None:
            # Create full grid RGB
            full_rgb = np.zeros(rgb_shape, dtype=np.uint8)

            # Fill active bins
            active_indices = env.layout.active_mask.flatten()
            full_rgb_flat = full_rgb.reshape(-1, 3)
            full_rgb_flat[active_indices] = rgb

            # Transpose from (n_x, n_y, 3) to (n_y, n_x, 3) for napari (y, x) convention
            transposed = np.transpose(full_rgb, (1, 0, 2))
            # Flip vertically: napari displays row 0 at top, but row 0 = min Y (bottom)
            # Environment coordinates have Y increasing upward, napari has Y increasing downward
            return np.flip(transposed, axis=0)
        else:
            # Regular grid without masking
            # Transpose from (n_x, n_y, 3) to (n_y, n_x, 3) for napari (y, x) convention
            transposed = np.transpose(rgb.reshape(rgb_shape), (1, 0, 2))
            # Flip vertically for napari coordinate system
            return np.flip(transposed, axis=0)

    # Non-grid layout: return flat RGB for point cloud rendering
    return rgb


def _validate_frame_labels(
    frame_labels: list[str] | None,
    n_frames: int,
    backend_name: str,
) -> list[str] | None:
    """Validate that frame_labels length matches number of frames.

    Parameters
    ----------
    frame_labels : list of str or None
        Optional labels, one per frame. If None, no labels are enforced here.
    n_frames : int
        Number of frames for the backend.
    backend_name : str
        Name of the backend for error-message context (e.g., "html", "video").

    Returns
    -------
    list of str or None
        The original labels if valid, or None.

    Raises
    ------
    ValueError
        If frame_labels is not None and its length does not match n_frames.

    Examples
    --------
    >>> _validate_frame_labels(["Frame 1", "Frame 2"], 2, "html")
    ['Frame 1', 'Frame 2']

    >>> _validate_frame_labels(None, 10, "video") is None
    True
    """
    if frame_labels is None:
        return None

    if len(frame_labels) != n_frames:
        raise ValueError(
            f"frame_labels length ({len(frame_labels)}) does not match "
            f"number of frames ({n_frames}) for backend '{backend_name}'.\n\n"
            f"WHAT: Mismatch between frame_labels and fields arrays.\n\n"
            f"WHY: Each frame needs exactly one label for the {backend_name} "
            f"backend's controls.\n\n"
            f"HOW: Ensure frame_labels has {n_frames} elements, for example:\n"
            f"  frame_labels = [f'Frame {{i + 1}}' for i in range({n_frames})]\n"
            f"  # Or pass frame_labels=None to use backend defaults (when supported)"
        )

    return frame_labels
