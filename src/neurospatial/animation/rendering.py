"""Shared rendering utilities for all backends."""

from __future__ import annotations

import io
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from neurospatial._stacklevel import _external_stacklevel
from neurospatial.animation._timing import timing

if TYPE_CHECKING:
    from neurospatial.environment import Environment


__all__ = [
    "_validate_frame_labels",
    "field_to_rgb_for_napari",
    "render_field_to_image_bytes",
    "render_field_to_png_bytes",
    "render_field_to_rgb",
]


# Above this many frames an animation's color range is estimated from a
# sample of about 50,000 frames rather than read in full.
_EXACT_RANGE_MAX_FRAMES = 200_000


# Color limits of an animation: these percentiles of the frames' minima and
# maxima.
_ROBUST_PERCENTILES = (1.0, 99.0)


def animation_colormap_range(
    fields: Sequence[NDArray[np.float64]] | NDArray[np.float64],
    vmin: float | None,
    vmax: float | None,
) -> tuple[float, float]:
    """Color limits for an animation, robust to a few extreme frames.

    The top of the colormap is the 99th percentile of the frames' maxima
    and the bottom the 1st percentile of their minima (see
    `robust_limits`). A percentile of all bin values would not do, because
    a sharp posterior fills few bins: its peak would pass the top in every
    frame.

    Parameters
    ----------
    fields : sequence of ndarray, or ndarray of shape (n_frames, n_bins)
        The frames.
    vmin, vmax : float or None
        Given limits; None computes them.

    Returns
    -------
    vmin, vmax : float

    Warns
    -----
    UserWarning
        When the range is estimated from every n-th frame.
    """
    if vmin is not None and vmax is not None:
        return vmin, vmax
    low, high = robust_limits(*frame_extremes(fields))
    return (low if vmin is None else vmin), (high if vmax is None else vmax)


def robust_limits(
    lows: NDArray[np.float64],
    highs: NDArray[np.float64],
    percentiles: tuple[float, float] = _ROBUST_PERCENTILES,
) -> tuple[float, float]:
    """Color limits from each frame's smallest and largest value.

    The bottom is a low percentile of the frames' minima and the top a high
    percentile of their maxima: about 1% of frames (by default) reach past
    each end and are drawn in its color, and a single outlier frame cannot
    wash out the rest.

    Parameters
    ----------
    lows, highs : ndarray of shape (n_frames,)
        Each frame's smallest and largest finite value, NaN for a frame
        with none (as `frame_extremes` returns).
    percentiles : tuple of float, default=(1.0, 99.0)
        The percentile of the minima and of the maxima.

    Returns
    -------
    vmin, vmax : float
        Tidied by `tidy_range`; (0, 1) when no value is finite.
    """
    finite = np.isfinite(highs)
    if not finite.any():
        return 0.0, 1.0
    return tidy_range(
        float(np.percentile(lows[finite], percentiles[0])),
        float(np.percentile(highs[finite], percentiles[1])),
    )


def tidy_range(low: float, high: float) -> tuple[float, float]:
    """A display range: a near-zero floor is zero, and a constant widens.

    A floor between zero and a thousandth of the top (a posterior's 1e-56,
    a rate map's near-zero floor) becomes zero, and equal limits become
    half a unit either side.
    """
    if 0.0 <= low < 1e-3 * high:
        low = 0.0
    if low == high:
        low, high = low - 0.5, high + 0.5
    return low, high


def typical_peak(highs: NDArray[np.float64]) -> float:
    """The median of the frames' maxima: a typical frame's peak.

    Parameters
    ----------
    highs : ndarray of shape (n_frames,)
        Each frame's largest finite value, NaN for none (as
        `frame_extremes` returns).

    Returns
    -------
    float
        1.0 when no value is finite.
    """
    finite = highs[np.isfinite(highs)]
    return float(np.median(finite)) if len(finite) else 1.0


def frame_extremes(
    fields: Sequence[NDArray[np.float64]] | NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Each frame's smallest and largest finite value; NaN for none.

    Very long recordings (over 200,000 frames) use every n-th frame, about
    50,000 of them, with a warning. Frames are read 10,000 at a time, so a
    memory-mapped recording is never loaded whole.

    Parameters
    ----------
    fields : sequence of ndarray, or ndarray of shape (n_frames, ...)
        The frames.

    Returns
    -------
    lows, highs : ndarray of shape (n_used_frames,)

    Warns
    -----
    UserWarning
        When only every n-th frame is used.
    """
    n_frames = len(fields)
    stride = 1
    if n_frames > _EXACT_RANGE_MAX_FRAMES:
        stride = n_frames // 50_000
        warnings.warn(
            f"Estimating colormap range from {n_frames:,} frames using "
            f"sample_stride={stride} (every {stride}th frame). For exact "
            "range, pass explicit vmin/vmax parameters.",
            UserWarning,
            stacklevel=_external_stacklevel(),
        )
    frames = range(0, n_frames, stride)
    lows, highs = [np.empty(0)], [np.empty(0)]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "All-NaN slice encountered")
        for start in range(0, len(frames), 10_000):
            index = frames[start : start + 10_000]
            if isinstance(fields, np.ndarray):
                chunk = np.asarray(
                    fields[index.start : index.stop : stride], dtype=float
                )
            else:
                chunk = np.stack([np.asarray(fields[i], dtype=float) for i in index])
            chunk = np.where(np.isfinite(chunk), chunk, np.nan)
            lows.append(np.nanmin(chunk.reshape(len(chunk), -1), axis=1))
            highs.append(np.nanmax(chunk.reshape(len(chunk), -1), axis=1))
    return np.concatenate(lows), np.concatenate(highs)


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
    from neurospatial.animation._utils import managed_figure

    with timing("render_field_to_rgb"):
        fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)

        # managed_figure guarantees plt.close(fig) even if plot_field or the
        # canvas draw raises, preventing figure leaks on the error path.
        with managed_figure(fig):
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
    overlay_data: Any = None,  # OverlayData
    frame_idx: int = 0,
    show_regions: bool | list[str] = False,
    region_alpha: float = 0.3,
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

    from neurospatial.animation._utils import managed_figure

    buf = io.BytesIO()

    fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)

    # managed_figure guarantees plt.close(fig) even if plot_field or overlay
    # rendering raises, preventing figure leaks on the error path.
    with managed_figure(fig):
        env.plot_field(
            field,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            colorbar=False,
            scale_bar=scale_bar,
        )

        # Render overlays if provided (using matplotlib, not JavaScript)
        if overlay_data is not None:
            from neurospatial.animation._parallel import _render_all_overlays

            _render_all_overlays(
                ax=ax,
                env=env,
                frame_idx=frame_idx,
                overlay_data=overlay_data,
                show_regions=show_regions,
                region_alpha=region_alpha,
            )

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
            except ImportError as e:
                raise ImportError(
                    "JPEG support requires Pillow. Install with:\n"
                    "  pip install pillow\n"
                    "or\n"
                    "  uv add pillow\n"
                    "\n"
                    "Alternatively, use image_format='png' (no dependencies)"
                ) from e

            img = Image.fromarray(rgb)
            img.save(buf, format="JPEG", quality=85, optimize=True)
        else:
            # PNG format (lossless)
            fig.savefig(buf, format="png")

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
        (n_bins, 3) for point cloud rendering. Missing observations and inactive
        grid cells use neutral gray (128, 128, 128), including constant fields.

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
    binning = env.binning
    if binning.grid_frame is not None:
        raise NotImplementedError(
            "Napari raster images do not support framed grids; use Matplotlib, video or HTML rendering."
        )
    # Validate field shape early for clear error messages
    if len(field) != env.n_bins:
        raise ValueError(
            f"Field has {len(field)} values but environment has {env.n_bins} bins. "
            f"Expected shape: ({env.n_bins},)"
        )

    # Preserve missing observations before normalization, including constant fields.
    nan_mask = np.isnan(field)
    missing_rgb = np.array([128, 128, 128], dtype=np.uint8)

    # Normalize to [0, 1], guarding against zero-range (constant field or vmin == vmax)
    value_range = vmax - vmin
    if value_range <= 0 or not np.isfinite(value_range):
        # Zero or invalid range: map all values to middle of colormap (index 127)
        normalized = np.full_like(field, 0.5)
    else:
        normalized = (field - vmin) / value_range
        normalized = np.clip(normalized, 0, 1)

    # Handle NaN values - set to 0 before integer conversion
    # NaN values pass through np.clip unchanged, which causes issues with astype
    if np.any(nan_mask):
        normalized = np.where(nan_mask, 0.0, normalized)

    # Map to colormap indices [0, 255]
    indices = (normalized * 255).astype(np.uint8)

    # Lookup RGB values
    rgb = cmap_lookup[indices]
    rgb[nan_mask] = missing_rgb

    # For grid-compatible 2D layouts, reshape to 2D image
    # Only rectilinear grids rasterize; also require a 2-D grid_shape.
    is_grid = binning.rectilinear is not None
    grid_shape = env.grid_shape
    if is_grid and len(grid_shape) == 2:
        # Create full grid RGB and fill the active bins
        full_rgb = np.full(
            (grid_shape[0], grid_shape[1], 3), missing_rgb, dtype=np.uint8
        )
        full_rgb.reshape(-1, 3)[env.active_mask.flatten()] = rgb

        # Transpose from (n_x, n_y, 3) to (n_y, n_x, 3) for napari (y, x) convention
        transposed = np.transpose(full_rgb, (1, 0, 2))
        # Flip vertically: napari displays row 0 at top, but row 0 = min Y (bottom)
        # Environment coordinates have Y increasing upward, napari has Y increasing downward
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
            f"Fix: Ensure frame_labels has {n_frames} elements, for example:\n"
            f"  frame_labels = [f'Frame {{i + 1}}' for i in range({n_frames})]\n"
            f"  # Or pass frame_labels=None to use backend defaults (when supported)"
        )

    return frame_labels
