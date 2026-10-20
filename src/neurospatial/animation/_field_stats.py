"""Renderer-free bounded field statistics for consistent display scales."""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from neurospatial._stacklevel import _external_stacklevel

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
