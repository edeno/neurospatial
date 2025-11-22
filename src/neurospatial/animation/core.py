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
    from neurospatial.animation.overlays import (
        BodypartOverlay,
        HeadDirectionOverlay,
        PositionOverlay,
    )
    from neurospatial.environment._protocols import EnvironmentProtocol


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
        raise ValueError(
            f"Video backend with parallel rendering requires environment to be pickle-able.\n"
            f"\n"
            f"Error: {e}\n"
            f"\n"
            f"Solutions:\n"
            f"  1. Clear caches: env.clear_cache() before animating\n"
            f"  2. Use n_workers=1 for serial rendering (slower)\n"
            f"  3. Use backend='html' instead (no pickling required)\n"
        ) from e


def animate_fields(
    env: EnvironmentProtocol,
    fields: Sequence[NDArray[np.float64]] | NDArray[np.float64],
    *,
    backend: Literal["auto", "napari", "video", "html", "widget"] = "auto",
    save_path: str | None = None,
    overlays: list[PositionOverlay | BodypartOverlay | HeadDirectionOverlay]
    | None = None,
    frame_times: NDArray[np.float64] | None = None,
    show_regions: bool | list[str] = False,
    region_alpha: float = 0.3,
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
    backend : {"auto", "napari", "video", "html", "widget"}, default="auto"
        Animation backend to use.
    save_path : str, optional
        Output path for video/HTML backends.
    overlays : list of PositionOverlay, BodypartOverlay, or HeadDirectionOverlay, optional
        Dynamic overlays to render on top of spatial fields. Supports position
        trajectories, multi-animal pose tracking, and head direction visualization.
        Multiple overlays can be provided for multi-animal tracking.
        Default is None (no overlays).
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
    >>> positions = np.random.randn(100, 2) * 50  # doctest: +SKIP
    >>> env = Environment.from_samples(positions, bin_size=5.0)  # doctest: +SKIP
    >>> fields = [np.random.rand(env.n_bins) for _ in range(20)]  # doctest: +SKIP
    >>> env.animate_fields(fields, backend="napari")  # doctest: +SKIP

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
    if not hasattr(env, "_is_fitted") or not env._is_fitted:
        raise RuntimeError(
            "Environment must be fitted before animation. "
            "Use Environment.from_samples() or other factory methods."
        )

    # Detect multi-field format (napari-specific feature)
    # Multi-field: list of sequences (e.g., [[field1, field2], [field3, field4]])
    is_multi_field = len(fields) > 0 and isinstance(fields[0], (list, tuple))

    # Validate field shapes (skip for multi-field - backend will validate)
    if not is_multi_field:
        for i, field in enumerate(fields):
            if len(field) != env.n_bins:
                raise ValueError(
                    f"Field {i} has {len(field)} values but environment has {env.n_bins} bins. "
                    f"Expected shape: ({env.n_bins},)"
                )

    # Compute n_frames (for multi-field, use length of first sequence)
    n_frames = len(fields[0]) if is_multi_field else len(fields)

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

    # Route to backend with early validation
    if backend == "napari":
        from neurospatial.animation.backends.napari_backend import render_napari

        return render_napari(
            env,  # type: ignore[arg-type]  # Backend signatures updated in future milestone
            fields,
            overlay_data=overlay_data,
            show_regions=show_regions,
            region_alpha=region_alpha,
            **kwargs,
        )

    elif backend == "video":
        from neurospatial.animation.backends.video_backend import render_video

        # Note: ffmpeg check is done inside render_video for single source of truth

        if save_path is None:
            raise ValueError("save_path required for video backend")

        # Validate environment pickle-ability for parallel rendering
        n_workers = kwargs.get("n_workers")
        if n_workers and n_workers > 1:
            _validate_env_pickleable(env)

        return render_video(
            env,  # type: ignore[arg-type]  # Backend signatures updated in future milestone
            fields,
            save_path,
            overlay_data=overlay_data,
            show_regions=show_regions,
            region_alpha=region_alpha,
            **kwargs,
        )

    elif backend == "html":
        from neurospatial.animation.backends.html_backend import render_html

        if save_path is None:
            save_path = "animation.html"
        return render_html(
            env,  # type: ignore[arg-type]  # Backend signatures updated in future milestone
            fields,
            save_path,
            overlay_data=overlay_data,
            show_regions=show_regions,
            region_alpha=region_alpha,
            **kwargs,
        )

    elif backend == "widget":
        from neurospatial.animation.backends.widget_backend import render_widget

        return render_widget(
            env,  # type: ignore[arg-type]  # Backend signatures updated in future milestone
            fields,
            overlay_data=overlay_data,
            show_regions=show_regions,
            region_alpha=region_alpha,
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
