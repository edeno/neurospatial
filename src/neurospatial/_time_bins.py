"""Half-open time bins tiled inside windows and per-bin spike counts.

Shared by time-binned decoding and peri-event histograms.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def _time_bin_rounding(
    windows: NDArray[np.float64],
    dt: float,
    *,
    context: str = "Time bins",
    width_name: str = "dt",
    clock_fix: str | None = None,
) -> NDArray[np.float64]:
    """Return each window's edge rounding allowance in seconds.

    Parameters
    ----------
    windows : ndarray, shape (n_windows, 2)
        ``[start, stop)`` windows in seconds.
    dt : float
        Bin width in seconds.
    context, width_name, clock_fix : str
        Wording of the error message.

    Returns
    -------
    ndarray, shape (n_windows,)
        A few ulp of each window's largest timestamp magnitude.

    Raises
    ------
    ValueError
        If a window bound is not finite, or if the rounding exceeds 1% of
        ``dt``, so bins of that width cannot be represented there.
    """
    if not np.isfinite(windows).all():
        w = int(np.flatnonzero(~np.isfinite(windows).all(axis=1))[0])
        raise ValueError(
            f"{context}: window {w} is {windows[w].tolist()}; its bounds must "
            "be finite seconds.\nFix: pass finite start and stop times."
        )
    start, stop = windows[:, 0], windows[:, 1]
    rounding: NDArray[np.float64] = 4.0 * np.spacing(
        np.maximum(np.abs(start), np.abs(stop))
    )
    imprecise = rounding > 1e-2 * dt
    if imprecise.any():
        w = int(np.flatnonzero(imprecise)[0])
        fix = clock_fix or (
            "subtract a time origin first (e.g. times - times[0], and the same "
            "offset from spike times and windows), or use a larger dt."
        )
        raise ValueError(
            f"{context} of {width_name}={dt:g} s cannot be represented at timestamps near "
            f"{start[w]:.6g} s: float64 rounding there is {rounding[w]:.3g} s, more "
            f"than 1% of {width_name}.\nFix: {fix}"
        )
    return rounding


def _time_bins_in_windows(
    windows: NDArray[np.float64], dt: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Tile each ``[start, stop)`` window with whole bins of width ``dt``.

    Each window is tiled independently, so no bin spans the space between two
    windows. A remainder shorter than one bin at the end of a window is
    dropped, because every bin must have the same width ``dt`` for counts from
    different bins (decoding likelihoods, peri-event histograms) to be
    comparable. A window that is a whole number of bins up to float rounding
    keeps its last bin, whose right edge is clamped to exactly ``stop``; that
    bin can be short only by the rounding allowance, at most 1% of ``dt``.

    Parameters
    ----------
    windows : ndarray, shape (n_windows, 2)
        Sorted, disjoint windows (seconds).
    dt : float
        Bin width (seconds), already validated.

    Returns
    -------
    left, right : ndarray, shape (n_time_bins,)
        Bin edges, ``left = start + dt * k``. Within a window
        ``right[i] == left[i + 1]`` exactly (same ``k``), and every
        ``right <= stop`` of its window.
    """
    start, stop = windows[:, 0], windows[:, 1]
    # Rounding error of the edges is a few ulp of the timestamps' magnitude.
    # If that is not small against dt, the edges cannot be represented to bin
    # precision, so refuse instead of inventing or merging bins.
    rounding = _time_bin_rounding(windows, dt)
    ratio = (stop - start) / dt
    # A shortfall below the rounding allowance is rounding, not a partial bin.
    # After the precision check the allowance is at most ~0.01, so it can
    # never add a whole bin. A final bin may therefore be up to 1% short; it
    # is kept and clamped to the window stop.
    slack = rounding / dt + 4.0 * np.finfo(np.float64).eps * ratio
    n_per = np.maximum(np.floor(ratio + slack), 0).astype(np.int64)
    window_idx = np.repeat(np.arange(windows.shape[0]), n_per)
    k = np.arange(int(n_per.sum())) - np.repeat(np.cumsum(n_per) - n_per, n_per)
    left = start[window_idx] + dt * k
    right = np.minimum(start[window_idx] + dt * (k + 1), stop[window_idx])
    return left, right


def _count_spikes_in_time_bins(
    spike_trains: Sequence[NDArray[np.float64]],
    left: NDArray[np.float64],
    right: NDArray[np.float64],
) -> NDArray[np.int64]:
    """Count spikes per half-open bin ``[left[i], right[i])``.

    Spikes outside every bin are not counted: those between windows and those
    in the dropped remainder, shorter than one bin, at the end of a window.

    Returns
    -------
    ndarray of int64, shape (n_time_bins, n_neurons)
    """
    counts = np.zeros((left.size, len(spike_trains)), dtype=np.int64)
    if left.size == 0:
        return counts
    for unit, train in enumerate(spike_trains):
        s = np.asarray(train, dtype=np.float64)
        idx = np.searchsorted(left, s, side="right") - 1
        inside = idx >= 0
        inside[inside] = s[inside] < right[idx[inside]]
        counts[:, unit] = np.bincount(idx[inside], minlength=left.size)
    return counts
