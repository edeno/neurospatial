"""Core animation orchestration.

This module provides the main dispatcher for multi-backend animation and utilities
for working with large-scale time series data.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from neurospatial.animation.overlays import OverlayProtocol
    from neurospatial.environment._protocols import EnvironmentProtocol

# Playback speed limits (Task 1.1)
MAX_PLAYBACK_FPS: int = 60  # Display refresh rate limit
MIN_PLAYBACK_FPS: int = 1  # Minimum usable playback
DEFAULT_SPEED: float = 1.0  # Real-time by default


def _compute_playback_fps(
    frame_times: NDArray[np.float64],
    speed: float,
    max_fps: int = MAX_PLAYBACK_FPS,
) -> tuple[int, float]:
    """Compute playback fps from frame timestamps and speed multiplier.

    Parameters
    ----------
    frame_times : NDArray[np.float64]
        Timestamps for each frame in seconds. Shape: (n_frames,).
    speed : float
        Playback speed multiplier (1.0 = real-time).
    max_fps : int, default=MAX_PLAYBACK_FPS
        Maximum allowed playback fps.

    Returns
    -------
    playback_fps : int
        Computed playback fps, clamped to [MIN_PLAYBACK_FPS, max_fps].
    actual_speed : float
        Actual speed after clamping (may differ from requested if capped).

    Notes
    -----
    The function handles edge cases gracefully:
    - Single frame: returns (max_fps, speed)
    - Zero duration: returns (max_fps, speed)
    - Very slow speed: clamps to MIN_PLAYBACK_FPS
    - Very fast speed: clamps to max_fps
    """
    # Edge case: single frame or fewer
    if len(frame_times) < 2:
        return max_fps, speed

    # Compute duration
    duration = float(frame_times[-1] - frame_times[0])

    # Edge case: zero or negative duration
    if duration <= 0:
        return max_fps, speed

    # Compute sample rate from timestamps
    sample_rate_hz = (len(frame_times) - 1) / duration

    # Compute requested fps
    requested_fps = sample_rate_hz * speed

    # Clamp to valid range [MIN_PLAYBACK_FPS, max_fps]
    playback_fps = int(min(max(requested_fps, MIN_PLAYBACK_FPS), max_fps))

    # Compute actual speed after clamping
    actual_speed = playback_fps / sample_rate_hz

    return playback_fps, actual_speed


def _validate_env_pickleable(env: EnvironmentProtocol) -> None:
    """Validate that environment can be pickled for parallel processing.

    Parameters
    ----------
    env : Environment
        Environment to validate

    Raises
    ------
    ValueError
        If environment cannot be pickled, with helpful error message

    Notes
    -----
    This helper provides a single source of truth for pickle validation
    with user-friendly error messages and solutions.
    """
    import pickle

    try:
        pickle.dumps(env, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        from neurospatial.animation._utils import _pickling_guidance

        raise ValueError(
            f"WHAT: Environment is not pickle-able, preventing parallel video "
            f"rendering.\n"
            f"  Pickling failed with: {type(e).__name__}: {e}\n\n"
            f"WHY: Video backend with parallel rendering (n_workers > 1) requires "
            f"pickling the Environment to pass it to worker processes. Unpickleable "
            f"objects include KDTree caches, lambdas, and certain class instances.\n\n"
            f"{_pickling_guidance()}"
        ) from e


def animate_fields(
    env: EnvironmentProtocol,
    fields: Sequence[NDArray[np.float64]] | NDArray[np.float64],
    *,
    backend: Literal["auto", "napari", "video", "html", "widget"] = "auto",
    save_path: str | None = None,
    overlays: list[OverlayProtocol] | None = None,
    frame_times: NDArray[np.float64] | None = None,
    show_regions: bool | list[str] = False,
    region_alpha: float = 0.3,
    scale_bar: bool | Any = False,  # bool | ScaleBarConfig
    **kwargs: Any,
) -> Any:
    """Main animation dispatcher.

    This function validates inputs and routes to the appropriate backend.

    Parameters
    ----------
    env : Environment
        Fitted environment defining spatial structure.
    fields : ndarray of shape (n_frames, n_bins) or list of ndarray of shape (n_bins,)
        Fields to animate, dtype float64. If ndarray, first dimension is time.
        If list of arrays, each array represents one frame.
        Values typically represent probabilities, firing rates, or other spatial
        quantities. Each field length must match env.n_bins.

        **Multi-field mode** (napari backend only): ``fields`` can also be a
        list of field sequences for side-by-side comparison. Each sequence
        contains frames for one field (e.g., one neuron's place field over time)::

            # Single field per frame (standard mode)
            fields = [field_frame0, field_frame1, ...]

            # Multiple field sequences (napari only)
            # Each sequence is a list of frames for that field
            posterior_seq = [posterior_t0, posterior_t1, ...]  # N frames
            likelihood_seq = [likelihood_t0, likelihood_t1, ...]  # N frames
            fields = [posterior_seq, likelihood_seq]  # 2 sequences of N frames
    backend : {"auto", "napari", "video", "html", "widget"}, default="auto"
        Animation backend to use.
    save_path : str, optional
        Output path for video/HTML backends.
    overlays : list of PositionOverlay, BodypartOverlay, HeadDirectionOverlay, or VideoOverlay, optional
        Dynamic overlays to render on top of spatial fields. Supports position
        trajectories, multi-animal pose tracking, head direction visualization,
        and video background/overlay. Multiple overlays can be provided for
        multi-animal tracking. Default is None (no overlays).
    frame_times : ndarray of shape (n_frames,), optional
        Explicit timestamps for each frame, dtype float64, in seconds.
        If provided, overlays with times will be aligned via linear
        interpolation. If None, frames assumed evenly spaced at fps rate.
        Must be monotonically increasing if provided. Default is None.
    show_regions : bool or list of str, default=False
        Whether to render region overlays. If True, all regions defined in env.regions
        are rendered. If list of strings, only those region names are rendered.
        Regions rendered as semi-transparent polygons. Default is False.
    region_alpha : float, default=0.3
        Alpha transparency for region overlays, range [0.0, 1.0] where 0.0 is
        fully transparent and 1.0 is fully opaque. Only used when show_regions
        is True or a list. Default is 0.3.
    **kwargs : dict
        Additional backend-specific parameters.

    Returns
    -------
    result : Any
        Backend-specific return value:
        - Napari: napari.Viewer instance
        - Video/HTML: Path to saved file
        - Widget: None (displays in-place, no return value)

    Raises
    ------
    RuntimeError
        If environment is not fitted
    ValueError
        If fields is empty, field shapes don't match, or backend is invalid
    RuntimeError
        If required dependencies are missing

    Examples
    --------
    Basic animation:

    >>> positions = np.random.randn(100, 2) * 50  # doctest: +SKIP
    >>> env = Environment.from_samples(positions, bin_size=5.0)  # doctest: +SKIP
    >>> fields = [np.random.rand(env.n_bins) for _ in range(20)]  # doctest: +SKIP
    >>> env.animate_fields(fields, backend="napari")  # doctest: +SKIP

    Mixed-rate temporal alignment (e.g., 120 Hz tracking → 10 Hz fields):

    >>> # Position tracked at 120 Hz, fields computed at 10 Hz
    >>> from neurospatial import PositionOverlay  # doctest: +SKIP
    >>> position_overlay = PositionOverlay(
    ...     data=trajectory_120hz,  # (n_samples_120hz, 2)
    ...     times=timestamps_120hz,  # Overlay timestamps
    ...     color="red",
    ...     trail_length=15,
    ... )  # doctest: +SKIP
    >>> env.animate_fields(
    ...     fields_10hz,
    ...     overlays=[position_overlay],
    ...     frame_times=timestamps_10hz,  # Field timestamps
    ...     backend="napari",
    ... )  # doctest: +SKIP

    Notes
    -----
    **Backend Selection Guide:**

    +---------+------------+------------------+----------------------------------+
    | Backend | Best For   | Frame Count      | Features                         |
    +=========+============+==================+==================================+
    | napari  | Exploration| 100K+            | Interactive, O(1) seek, zoom     |
    +---------+------------+------------------+----------------------------------+
    | video   | Export     | Any (parallel)   | MP4 for publications, n_workers  |
    +---------+------------+------------------+----------------------------------+
    | widget  | Notebooks  | <5K              | Inline display, simple scrubbing |
    +---------+------------+------------------+----------------------------------+
    | html    | Sharing    | <500             | Standalone file, no dependencies |
    +---------+------------+------------------+----------------------------------+

    **Auto-selection criteria** (``backend="auto"``):

    - If ``save_path`` ends with ``.mp4``: video
    - If ``save_path`` ends with ``.html``: html
    - If in Jupyter notebook: widget
    - Otherwise: napari

    **Large-dataset recommendations:**

    - **100K+ frames**: Use ``backend="napari"`` for interactive exploration.
      Napari has O(1) frame seek time regardless of dataset size.
    - **Export to video**: Use ``backend="video"`` with ``n_workers=4`` for
      parallel rendering. Call ``env.clear_cache()`` first.
    - **Subsampling**: For high-frequency data (e.g., 250 Hz), subsample to
      30-60 fps with ``subsample_frames(fields, source_fps=250, target_fps=30)``
      from ``neurospatial.animation``.

    **Parallel video rendering:**

    Ensure environment is pickle-able by calling ``env.clear_cache()`` before
    animating with ``n_workers > 1``.
    """
    # Detect if input is a 2D+ numpy array (potential memmap)
    # This must happen before any conversion to preserve array format for napari
    fields_is_array = isinstance(fields, np.ndarray) and fields.ndim >= 2

    # Validate array dimensions if array input
    if isinstance(fields, np.ndarray) and fields.ndim < 2:
        raise ValueError("fields must be at least 2D (n_frames, n_bins)")

    # Compute n_frames from appropriate source
    if fields_is_array:
        # Type assertion: fields_is_array means fields is a 2D+ ndarray
        assert isinstance(fields, np.ndarray)  # nosec: type narrowing for mypy
        n_frames = fields.shape[0]
        if n_frames == 0:
            raise ValueError("fields cannot be empty")
    else:
        # Convert to list for non-array inputs
        fields = list(fields)
        if len(fields) == 0:
            raise ValueError("fields cannot be empty")
        n_frames = len(fields)

    # Validate environment is fitted
    if not hasattr(env, "_is_fitted") or not env._is_fitted:
        raise RuntimeError(
            "Environment must be fitted before animation. "
            "Use Environment.from_samples() or other factory methods."
        )

    # Detect multi-field format (napari-specific feature)
    # Multi-field: list of sequences (e.g., [[field1, field2], [field3, field4]])
    # Note: Arrays are never multi-field (they're always single-field with shape (n_frames, n_bins))
    is_multi_field = (
        not fields_is_array and len(fields) > 0 and isinstance(fields[0], (list, tuple))
    )

    # Validate field shapes (skip for multi-field - backend will validate)
    if not is_multi_field:
        if fields_is_array:
            # Type assertion: fields_is_array means fields is a 2D+ ndarray
            assert isinstance(fields, np.ndarray)  # nosec: type narrowing for mypy
            # For arrays, validate second dimension matches n_bins
            if fields.shape[1] != env.n_bins:
                raise ValueError(
                    f"Fields array has shape {fields.shape} but expected "
                    f"({n_frames}, {env.n_bins}). Second dimension must match env.n_bins."
                )
        else:
            # For lists, validate each element
            for i, field in enumerate(fields):
                if len(field) != env.n_bins:
                    raise ValueError(
                        f"Field {i} has {len(field)} values but environment has {env.n_bins} bins. "
                        f"Expected shape: ({env.n_bins},)"
                    )

    # Update n_frames for multi-field case
    if is_multi_field:
        n_frames = len(fields[0])

    # Build or verify frame_times for overlay alignment
    if overlays is not None or frame_times is not None:
        from neurospatial.animation.overlays import _build_frame_times

        # Get fps from kwargs (default 30)
        fps = kwargs.get("fps", 30)
        frame_times = _build_frame_times(
            n_frames=n_frames, fps=fps, frame_times=frame_times
        )

    # Convert overlays to internal OverlayData if provided
    overlay_data = None
    if overlays is not None:
        from neurospatial.animation.overlays import _convert_overlays_to_data

        # frame_times is guaranteed to be set by _build_frame_times above
        assert frame_times is not None, "frame_times should be built before conversion"
        overlay_data = _convert_overlays_to_data(
            overlays=overlays, frame_times=frame_times, n_frames=n_frames, env=env
        )

        # Validate pickle-ability for parallel rendering (applies to video backend)
        # Check n_workers from kwargs to determine if parallel rendering will be used
        n_workers = kwargs.get("n_workers")
        if n_workers and n_workers > 1:
            from neurospatial.animation.overlays import _validate_pickle_ability

            _validate_pickle_ability(overlay_data, n_workers=n_workers)

    # Auto-select backend
    if backend == "auto":
        backend = _select_backend(n_frames, save_path)

    # Convert arrays to list for non-napari backends
    # Napari backend can handle arrays directly (enables memmap efficiency)
    # Other backends expect list semantics for per-frame iteration
    if fields_is_array and backend != "napari":
        # Type assertion: fields_is_array means fields is a 2D+ ndarray
        assert isinstance(fields, np.ndarray)  # nosec: type narrowing for mypy
        fields = [fields[i] for i in range(fields.shape[0])]

    # Multi-field inputs are only supported by the napari backend
    if is_multi_field and backend != "napari":
        raise ValueError(
            "Multi-field input (list of sequences) is only supported by the "
            "'napari' backend.\n\n"
            "WHAT: Detected list-of-sequences input for 'fields', which encodes "
            "multiple spatial fields per frame (multi-field mode).\n\n"
            "WHY: Video, HTML, and widget backends currently expect a single "
            "field array per frame and cannot render multi-field layouts.\n\n"
            "HOW: Either:\n"
            "  - Use backend='napari' (or backend='auto' without save_path) "
            "to explore multi-field data interactively, or\n"
            "  - Convert your multi-field input into a single field per frame "
            "(e.g., by stacking or selecting one field) before calling "
            "animate_fields()."
        )

    # Route to backend with early validation
    if backend == "napari":
        from neurospatial.animation.backends.napari_backend import render_napari

        return render_napari(
            env,  # type: ignore[arg-type]  # Backend signatures updated in future milestone
            fields,  # Now accepts arrays after Task 2.1
            overlay_data=overlay_data,
            show_regions=show_regions,
            region_alpha=region_alpha,
            scale_bar=scale_bar,
            **kwargs,
        )

    elif backend == "video":
        from neurospatial.animation.backends.video_backend import (
            check_ffmpeg_available,
            render_video,
        )

        # Early validation: fail fast before expensive setup if ffmpeg missing
        if not check_ffmpeg_available():
            raise RuntimeError(
                "WHAT: ffmpeg is not installed or not found in PATH.\n\n"
                "WHY: Video export requires ffmpeg for encoding frames into video.\n\n"
                "HOW: Install ffmpeg:\n"
                "  macOS:   brew install ffmpeg\n"
                "  Ubuntu:  sudo apt install ffmpeg\n"
                "  Windows: Download from https://ffmpeg.org/download.html\n\n"
                "Or use a different backend:\n"
                "  backend='napari' - Interactive viewer (no dependencies)\n"
                "  backend='html'   - Standalone HTML file (no dependencies)\n"
                "  backend='widget' - Jupyter notebook widget"
            )

        if save_path is None:
            raise ValueError("save_path required for video backend")

        # Validate environment pickle-ability for parallel rendering
        n_workers = kwargs.get("n_workers")
        if n_workers and n_workers > 1:
            _validate_env_pickleable(env)

        return render_video(
            env,  # type: ignore[arg-type]  # Backend signatures updated in future milestone
            fields,  # type: ignore[arg-type]  # Converted to list above for non-napari
            save_path,
            overlay_data=overlay_data,
            show_regions=show_regions,
            region_alpha=region_alpha,
            scale_bar=scale_bar,
            **kwargs,
        )

    elif backend == "html":
        from neurospatial.animation.backends.html_backend import render_html

        if save_path is None:
            save_path = "animation.html"
        return render_html(
            env,  # type: ignore[arg-type]  # Backend signatures updated in future milestone
            fields,  # type: ignore[arg-type]  # Converted to list above for non-napari
            save_path,
            overlay_data=overlay_data,
            show_regions=show_regions,
            region_alpha=region_alpha,
            scale_bar=scale_bar,
            **kwargs,
        )

    elif backend == "widget":
        from neurospatial.animation.backends.widget_backend import render_widget

        return render_widget(
            env,  # type: ignore[arg-type]  # Backend signatures updated in future milestone
            fields,  # type: ignore[arg-type]  # Converted to list above for non-napari
            overlay_data=overlay_data,
            show_regions=show_regions,
            region_alpha=region_alpha,
            scale_bar=scale_bar,
            **kwargs,
        )

    else:
        raise ValueError(f"Unknown backend: {backend}")


def _select_backend(
    n_frames: int, save_path: str | None
) -> Literal["napari", "video", "html", "widget"]:
    """Auto-select appropriate backend with transparent logging.

    Logs selection rationale at INFO level for user transparency.

    Parameters
    ----------
    n_frames : int
        Number of frames to animate
    save_path : str, optional
        Output path (if specified)

    Returns
    -------
    backend : {"napari", "video", "html", "widget"}
        Selected backend

    Raises
    ------
    RuntimeError
        If no suitable backend is available
    """
    logger = logging.getLogger(__name__)

    # Check if in Jupyter
    try:
        from IPython import get_ipython  # type: ignore[attr-defined]

        in_jupyter = get_ipython() is not None
    except ImportError:
        in_jupyter = False

    # Video export requested (file extension determines format)
    if save_path:
        ext = Path(save_path).suffix
        if ext in (".mp4", ".webm", ".avi", ".mov"):
            logger.info(f"Auto-selected 'video' backend (save_path extension: {ext})")
            return "video"
        elif ext == ".html":
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
                f"  3. Use HTML (WARNING: {n_frames * 0.1:.0f} MB file):\n"
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


def _subsample_indices(n: int, source_fps: int, target_fps: int) -> NDArray[np.int_]:
    """Compute indices for frame subsampling with proper rounding.

    Avoids duplicate or skipped frames by using rounding instead of truncation,
    and ensures strictly increasing indices.

    Parameters
    ----------
    n : int
        Total number of frames
    source_fps : int
        Original sampling rate
    target_fps : int
        Desired output frame rate

    Returns
    -------
    indices : ndarray
        Frame indices to keep (strictly increasing, no duplicates)
    """
    rate = source_fps / target_fps
    # Use rounding to avoid drift from floating-point truncation
    idx = np.rint(np.arange(0, n, rate)).astype(np.int64)
    # Clamp to valid range
    idx = idx[idx < n]
    # Ensure strictly increasing by removing duplicates (preserves order)
    result: NDArray[np.int_] = np.unique(idx)
    # Ensure first frame is always included
    if result.size > 0 and result[0] != 0:
        result = np.concatenate([[0], result[result != 0]])
    return result


def subsample_frames(
    fields: NDArray | list,
    target_fps: int,
    source_fps: int,
) -> NDArray | list:
    """Subsample frames to target frame rate.

    Essential utility for large-scale sessions. Allows users to reduce
    900K frames at 250 Hz to a manageable video at 30 fps.

    Parameters
    ----------
    fields : ndarray of shape (n_frames, n_bins) or list of ndarray of shape (n_bins,)
        Full field data, dtype float64. If ndarray, first dimension is time.
        If list, each element is a single frame's field values.
    target_fps : int
        Desired output frame rate (e.g., 30 for video). Must be positive and
        less than or equal to source_fps.
    source_fps : int
        Original sampling rate (e.g., 250 Hz for neural recording). Must be positive.

    Returns
    -------
    subsampled : ndarray of shape (n_subsampled, n_bins) or list of ndarray of shape (n_bins,)
        Subsampled fields at target_fps, same type and dtype as input.
        n_subsampled is approximately n_frames * target_fps / source_fps.

    Raises
    ------
    ValueError
        If target_fps exceeds source_fps

    Examples
    --------
    >>> # 250 Hz recording → 30 fps video
    >>> fields_video = subsample_frames(
    ...     fields_full, target_fps=30, source_fps=250
    ... )  # doctest: +SKIP
    >>> env.animate_fields(
    ...     fields_video, save_path="output.mp4", fps=30
    ... )  # doctest: +SKIP

    >>> # 1000 Hz → 60 fps
    >>> fields_video = subsample_frames(
    ...     fields_full, target_fps=60, source_fps=1000
    ... )  # doctest: +SKIP

    Notes
    -----
    Subsampling uses proper rounding to avoid frame drift over long sequences.
    The algorithm ensures:
    - No duplicate frames (strictly increasing indices)
    - No skipped frames due to floating-point truncation
    - First frame (index 0) is always included
    - Works efficiently with memory-mapped arrays

    For example:
    - 250 Hz → 30 fps: takes frames at indices [0, 8, 17, 25, ...]
    - 1000 Hz → 60 fps: takes frames at indices [0, 17, 33, 50, ...]
    """
    if target_fps > source_fps:
        raise ValueError(
            f"target_fps ({target_fps}) cannot exceed source_fps ({source_fps})"
        )

    indices = _subsample_indices(len(fields), source_fps, target_fps)

    if isinstance(fields, np.ndarray):
        result: NDArray | list = fields[indices]
        return result
    else:
        result_list: NDArray | list = [fields[i] for i in indices]
        return result_list


def estimate_colormap_range_from_subset(
    fields: NDArray | list,
    n_samples: int = 10_000,
    percentile: tuple[float, float] = (1.0, 99.0),
    seed: int | None = None,
) -> tuple[float, float]:
    """Estimate colormap range from random subset of frames.

    Essential utility for large-scale sessions where computing exact min/max
    over all frames would be too slow or memory-intensive.

    Parameters
    ----------
    fields : ndarray of shape (n_frames, n_bins) or list of ndarray of shape (n_bins,)
        Full field data, dtype float64. If ndarray, first dimension is time.
        If list, each element is a single frame's field values.
    n_samples : int, default=10_000
        Maximum number of frames to sample. If dataset has fewer frames,
        all frames are used.
    percentile : tuple of float, default=(1.0, 99.0)
        Lower and upper percentiles for range estimation. Using percentiles
        instead of min/max provides robustness against outliers.
    seed : int, optional
        Random seed for reproducibility. If None, results may vary between calls.

    Returns
    -------
    vmin, vmax : tuple of float
        Estimated colormap range (lower and upper bounds).

    Examples
    --------
    >>> rng = np.random.default_rng(42)
    >>> fields = rng.random((100_000, 500))  # 100K frames, 500 bins
    >>> vmin, vmax = estimate_colormap_range_from_subset(fields)
    >>> env.animate_fields(
    ...     fields, vmin=vmin, vmax=vmax, backend="napari"
    ... )  # doctest: +SKIP

    >>> # Tighter range with 5th/95th percentiles
    >>> vmin, vmax = estimate_colormap_range_from_subset(
    ...     fields, percentile=(5.0, 95.0)
    ... )  # doctest: +SKIP

    Notes
    -----
    For uniform random [0, 1] data with default percentiles (1, 99):
    - Expected vmin: ~0.01
    - Expected vmax: ~0.99

    This function is designed to complete quickly (<0.5s) even for
    datasets with 1M+ frames by sampling a subset rather than computing
    over the entire dataset.

    Works efficiently with memory-mapped arrays - only the sampled
    indices are loaded into memory.
    """
    # Get number of frames
    n_frames = fields.shape[0] if isinstance(fields, np.ndarray) else len(fields)

    # Determine indices to sample
    rng = np.random.default_rng(seed)
    if n_frames <= n_samples:
        # Use all frames
        indices = np.arange(n_frames)
    else:
        # Sample without replacement
        indices = rng.choice(n_frames, size=n_samples, replace=False)

    # Collect sampled values
    if isinstance(fields, np.ndarray):
        # For arrays, advanced indexing creates a copy (required for memmaps)
        sampled = fields[indices]  # Shape: (n_samples, n_bins)
        values = sampled.ravel()
    else:
        # For lists, gather sampled frames
        values = np.concatenate([fields[i] for i in indices])

    # Filter NaN/inf values
    valid_values = values[np.isfinite(values)]

    if len(valid_values) == 0:
        # Fallback for all-NaN data
        return (0.0, 1.0)

    # Compute percentiles
    vmin = float(np.percentile(valid_values, percentile[0]))
    vmax = float(np.percentile(valid_values, percentile[1]))

    return (vmin, vmax)


def large_session_napari_config(
    n_frames: int,
    sample_rate_hz: int | None = None,
) -> dict[str, Any]:
    """Get recommended napari animation settings for large datasets.

    Returns a dictionary of keyword arguments suitable for passing to
    `animate_fields()` when working with large-scale neural recordings.

    Parameters
    ----------
    n_frames : int
        Total number of frames in the dataset.
    sample_rate_hz : int, optional
        Original sampling rate in Hz (e.g., 250 for 250 Hz recording).
        If provided, helps determine appropriate playback fps.

    Returns
    -------
    config : dict
        Dictionary containing recommended settings:
        - ``fps``: Playback frame rate
        - ``chunk_size``: Number of frames to pre-render per chunk
        - ``max_chunks``: Maximum chunks to keep in memory

    Examples
    --------
    >>> config = large_session_napari_config(n_frames=500_000, sample_rate_hz=250)
    >>> env.animate_fields(fields, backend="napari", **config)  # doctest: +SKIP

    >>> # Or combine with estimated colormap range
    >>> vmin, vmax = estimate_colormap_range_from_subset(fields)
    >>> config = large_session_napari_config(n_frames=len(fields))
    >>> env.animate_fields(
    ...     fields, vmin=vmin, vmax=vmax, backend="napari", **config
    ... )  # doctest: +SKIP

    Notes
    -----
    Recommendations are based on empirical testing with napari animation:

    - **Small datasets** (<50K frames): Default settings work well
    - **Medium datasets** (50K-200K): Larger chunks improve scrubbing
    - **Large datasets** (200K-1M): Maximum chunk/cache sizes for smooth playback
    - **Very large datasets** (>1M): Same as large, plus explicit vmin/vmax recommended

    The returned config can be unpacked directly into `animate_fields()`:

    >>> config = large_session_napari_config(1_000_000)
    >>> env.animate_fields(fields, **config)  # doctest: +SKIP
    """
    # Determine fps based on sample rate or use sensible default
    # Target 30 fps for smooth playback, but cap at sample rate if provided
    fps = min(30, sample_rate_hz) if sample_rate_hz is not None else 30

    # Scale chunk_size based on dataset size
    # Larger datasets benefit from larger chunks for efficient I/O
    if n_frames < 50_000:
        # Small dataset: default chunk size
        chunk_size = 100
        max_chunks = 10
    elif n_frames < 200_000:
        # Medium dataset: larger chunks
        chunk_size = 500
        max_chunks = 20
    elif n_frames < 1_000_000:
        # Large dataset: maximum chunk sizes
        chunk_size = 1000
        max_chunks = 50
    else:
        # Very large dataset (1M+): same as large
        chunk_size = 1000
        max_chunks = 100

    return {
        "fps": fps,
        "chunk_size": chunk_size,
        "max_chunks": max_chunks,
    }
