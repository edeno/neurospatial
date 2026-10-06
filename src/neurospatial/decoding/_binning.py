"""Temporal binning of spike-time arrays into count matrices.

Provides :func:`bin_spikes_in_time`, the public primitive that turns a
sequence of per-neuron spike-time arrays into a regular time grid of spike
counts. It owns the time-grid construction (left edges plus ``dt / 2`` bin
centers) so the spike -> time-bin -> decode seam has a single, explicit home.

The default ``orient="time_x_neuron"`` -- ``(n_time_bins, n_neurons)`` -- is the
one convention used across the library: both
:func:`neurospatial.decoding.decode_position` and the assembly functions in
:mod:`neurospatial.decoding.assemblies` consume it directly, so the default
output feeds either with no transpose. ``orient="neuron_x_time"`` is offered only
as a convenience for code that wants the transposed layout.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from neurospatial._intervals import as_intervals


def validate_dt(dt: float) -> float:
    """Validate a decoding time-bin width and return it as a plain ``float``.

    Shared guard for every decoding entry point that bins or divides by ``dt``
    (e.g. :func:`bin_spikes_in_time`, :func:`decode_position`, and
    ``decode_session`` / ``decode_session_summary`` via ``_build_encoding_model``).
    It rejects non-numeric inputs (including a numeric *string* like ``"0.1"``,
    which would otherwise leak a raw ``TypeError`` from ``"0.1" <= 0`` or be
    silently coerced) and ``bool`` (``dt=True`` would otherwise pass the numeric
    guards and be used as a chunk size of ``1``), then coerces to ``float`` and
    rejects non-finite / non-positive values.

    Parameters
    ----------
    dt : float
        Candidate bin width. Must be a finite, strictly-positive real number
        (``int``/``float``/NumPy scalar; ``bool`` is rejected).

    Returns
    -------
    float
        The validated ``dt`` as a plain Python ``float``.

    Raises
    ------
    ValueError
        If ``dt`` is non-numeric, a ``bool``, or not finite and strictly
        positive.
    """
    if not isinstance(dt, (int, float, np.integer, np.floating)) or isinstance(
        dt, bool
    ):
        raise ValueError(f"dt must be a finite number > 0, got {dt!r}.")
    dt = float(dt)
    if dt <= 0 or not np.isfinite(dt):
        raise ValueError(f"dt must be a finite number > 0, got {dt!r}.")
    return dt


def _time_bin_rounding(
    windows: NDArray[np.float64],
    dt: float,
    *,
    context: str = "Time bins",
    width_name: str = "dt",
    clock_fix: str | None = None,
) -> NDArray[np.float64]:
    """Return edge rounding allowance, refusing unrepresentable bin widths."""
    start, stop = windows[:, 0], windows[:, 1]
    rounding = 4.0 * np.spacing(np.maximum(np.abs(start), np.abs(stop)))
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


def time_bins_in_windows(
    windows: NDArray[np.float64], dt: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Tile each ``[start, stop)`` window with whole bins of width ``dt``.

    Each window is tiled independently, so no bin spans the space between two
    windows. A remainder shorter than one bin at the end of a window is
    dropped, because the decoder's Poisson likelihood assumes every bin is
    ``dt`` long. A window that is a whole number of bins up to float rounding
    keeps its last bin, whose right edge is clamped to exactly ``stop``.

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


def count_spikes_in_time_bins(
    spike_trains: Sequence[NDArray[np.float64]],
    left: NDArray[np.float64],
    right: NDArray[np.float64],
) -> NDArray[np.int64]:
    """Count spikes per half-open bin ``[left[i], right[i])``.

    Spikes that fall between bins (outside every window) are not counted.

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


def bin_spikes_in_time(
    spike_trains: Sequence[NDArray[np.float64]],
    dt: float,
    t_start: float | None = None,
    t_stop: float | None = None,
    *,
    epochs: Any = None,
    orient: Literal["time_x_neuron", "neuron_x_time"] = "time_x_neuron",
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Bin per-neuron spike times into a count matrix on a regular time grid.

    Builds a regular time grid spanning ``[t_start, t_stop)`` with bin width
    ``dt`` and counts, for each neuron, how many spikes fall in each bin. The
    function owns the time-grid construction so downstream decoding and
    assembly analyses share one consistent definition of bin edges and centers.

    Parameters
    ----------
    spike_trains : Sequence[NDArray[np.float64]]
        One 1-D array of spike times per neuron. Arrays may have different
        lengths (different numbers of spikes); a neuron with no spikes is
        allowed and yields an all-zero row/column. Times are in the same
        units as ``dt``, ``t_start``, and ``t_stop`` (typically seconds).
    dt : float
        Bin width, in the same time units as the spike times. Must be finite
        and strictly positive.
    t_start : float or None, optional
        Left edge of the first bin. If None (default), uses the minimum spike
        time across all neurons (0.0 if every train is empty).
    t_stop : float or None, optional
        Upper bound of the time grid. If None (default), uses the maximum
        spike time across all neurons plus ``dt`` (``t_start + dt`` if every
        train is empty), so the last spike always lands inside the final bin
        and a single-spike train produces a valid result. When passed
        explicitly, must be strictly greater than ``t_start``.
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as the spike times). An interval counts only if it lies entirely
        inside one window. ``None`` (default) means unrestricted.
        Cannot be combined with ``t_start`` or ``t_stop``.
    orient : {"time_x_neuron", "neuron_x_time"}, optional
        Axis order of the returned ``counts`` matrix. ``"time_x_neuron"``
        (default) returns shape ``(n_time_bins, n_neurons)`` — the one convention
        used across the library, consumed directly by both
        :func:`neurospatial.decoding.decode_position` and the assembly functions
        in :mod:`neurospatial.decoding.assemblies`. ``"neuron_x_time"`` returns
        the transposed ``(n_neurons, n_time_bins)`` as a convenience for code
        that wants that layout.

    Returns
    -------
    counts : NDArray[np.int64]
        Spike counts. Shape ``(n_time_bins, n_neurons)`` if
        ``orient="time_x_neuron"``, else ``(n_neurons, n_time_bins)``.
    bin_centers : NDArray[np.float64]
        Shape ``(n_time_bins,)``; bin left edge plus ``dt / 2``.

    Raises
    ------
    ValueError
        If ``dt`` is not finite or not strictly positive, if an explicitly
        passed ``t_stop`` is not strictly greater than ``t_start``, if the
        windows contain no whole bin, if ``epochs`` is combined with explicit
        bounds, or if
        ``orient`` is not one of the allowed values.

    Notes
    -----
    Bins are half-open ``[left, right)`` and are tiled independently inside
    each normalized epoch. A trailing remainder shorter than ``dt`` is
    dropped. A spike exactly at ``t_stop`` or any window stop is not counted.
    Edges that round past a window stop are clamped to that stop.

    Examples
    --------
    Bin two neurons and feed the result straight into ``decode_position``:

    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.decoding import bin_spikes_in_time, decode_position
    >>> spike_trains = [
    ...     np.array([0.01, 0.06, 0.07]),  # neuron 0
    ...     np.array([0.03, 0.09]),  # neuron 1
    ... ]
    >>> counts, bin_centers = bin_spikes_in_time(
    ...     spike_trains, dt=0.025, t_start=0.0, t_stop=0.1
    ... )
    >>> counts
    array([[1, 0],
           [0, 1],
           [2, 0],
           [0, 1]])
    >>> bin_centers
    array([0.0125, 0.0375, 0.0625, 0.0875])
    >>> positions = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
    >>> env = Environment.from_samples(positions, bin_size=5.0)
    >>> encoding_models = np.array(
    ...     [
    ...         np.full(env.n_bins, 5.0),
    ...         np.full(env.n_bins, 3.0),
    ...     ]
    ... )
    >>> result = decode_position(
    ...     env, counts, encoding_models, dt=0.025, times=bin_centers
    ... )
    >>> result.posterior.shape == (len(bin_centers), env.n_bins)
    True
    """
    dt = validate_dt(dt)
    trains = [np.asarray(s, dtype=np.float64) for s in spike_trains]
    if epochs is not None:
        if t_start is not None or t_stop is not None:
            raise ValueError(
                "bin_spikes_in_time got both epochs and t_start/t_stop. "
                "\nWhy: epochs already defines where bins are formed. "
                "\nFix: pass either epochs=[(start, stop), ...] or "
                "t_start=..., t_stop=..., not both."
            )
        windows = as_intervals(epochs, name="epochs")
        assert windows is not None
    else:
        if t_start is None:
            t_start = min((s.min() for s in trains if s.size), default=0.0)
        if t_stop is None:
            t_stop = max((s.max() for s in trains if s.size), default=t_start) + dt
        elif t_stop <= t_start:
            raise ValueError(f"t_stop ({t_stop}) must be > t_start ({t_start}).")
        windows = np.array([[t_start, t_stop]], dtype=np.float64)
    left, right = time_bins_in_windows(windows, dt)
    if left.size == 0:
        longest = float(np.max(np.diff(windows, axis=1), initial=0.0))
        raise ValueError(
            f"Window span ({longest}) is smaller than one bin dt ({dt}); "
            "no whole time bin fits. Why: each window must contain a full bin.\n"
            "Fix: use a smaller dt or widen the time windows."
        )
    counts = count_spikes_in_time_bins(trains, left, right)
    bin_centers = left + dt / 2.0
    if orient == "neuron_x_time":
        counts = counts.T
    elif orient != "time_x_neuron":
        raise ValueError(
            f"orient must be 'time_x_neuron' or 'neuron_x_time', got {orient!r}."
        )
    return counts, bin_centers
