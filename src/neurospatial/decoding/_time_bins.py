"""Temporal binning of spike-time arrays into count matrices.

Provides :func:`bin_spikes_in_time`, the public primitive that turns a
sequence of per-neuron spike-time arrays into a regular time grid of spike
counts. It builds the bin centers (left edges plus ``dt / 2``) so the
spike -> time-bin -> decode seam has a single, explicit home; the tiling of
time windows into whole bins lives in :mod:`neurospatial._time_bins`.

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

from neurospatial._exceptions import _format_error
from neurospatial._intervals import as_intervals
from neurospatial._spikes import as_spike_trains
from neurospatial._time_bins import _count_spikes_in_time_bins, _time_bins_in_windows


def validate_dt(dt: float) -> float:
    """Validate a decoding time-bin width and return it as a plain ``float``.

    Shared guard for every decoding entry point that bins or divides by ``dt``
    (e.g. :func:`bin_spikes_in_time`, :func:`decode_position`, and
    ``decode_session`` / ``decode_session_summary``).
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


def bin_spikes_in_time(
    spike_times: Sequence[NDArray[np.float64]],
    *,
    dt: float,
    t_start: float | None = None,
    t_stop: float | None = None,
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
    spike_times : Sequence[NDArray[np.float64]] or spike group
        One 1-D array of spike times per neuron, or any spike input
        :func:`~neurospatial.decoding.decode_session` accepts. Every spike
        time must be finite. Arrays may have different
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
        same clock as the spike times). Time bins are tiled inside each
        window, so no bin spans two windows. ``None`` (default) means
        unrestricted.
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
        passed ``t_stop`` is not strictly greater than ``t_start``, if
        ``t_start`` is after the last spike and ``t_stop`` is omitted, if the
        windows contain no whole bin, if ``epochs`` is combined with explicit
        bounds, or if ``orient`` is not one of the allowed values.

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
    >>> spike_times = [
    ...     np.array([0.01, 0.06, 0.07]),  # neuron 0
    ...     np.array([0.03, 0.09]),  # neuron 1
    ... ]
    >>> counts, bin_centers = bin_spikes_in_time(
    ...     spike_times, dt=0.025, t_start=0.0, t_stop=0.1
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
    trains = as_spike_trains(spike_times)
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
        start_from_spikes = t_start is None
        if t_start is None:
            t_start = min((s.min() for s in trains if s.size), default=0.0)
        if t_stop is None:
            last = max((s.max() for s in trains if s.size), default=None)
            if last is not None and last + dt <= t_start:
                raise ValueError(
                    _format_error(
                        f"bin_spikes_in_time: t_start ({t_start}) is after the "
                        f"last spike ({last}), so the default t_stop (last "
                        "spike + dt) leaves no time to bin.",
                        why="without t_stop the grid ends one bin after the "
                        "last spike.",
                        fix="pass t_stop for the end of the period to bin, or "
                        "a t_start before the spikes",
                    )
                )
            t_stop = (t_start if last is None else last) + dt
        elif t_stop <= t_start:
            source = " (the earliest spike time)" if start_from_spikes else ""
            raise ValueError(
                _format_error(
                    f"bin_spikes_in_time: t_stop ({t_stop}) must be > t_start "
                    f"({t_start}{source}).",
                    why="the time grid runs from t_start to t_stop.",
                    fix="pass t_start explicitly, or a t_stop after it"
                    if start_from_spikes
                    else "pass t_stop > t_start",
                )
            )
        if not (np.isfinite(t_start) and np.isfinite(t_stop)):
            raise ValueError(
                _format_error(
                    f"bin_spikes_in_time got t_start={t_start}, t_stop={t_stop}.",
                    why="Why: the time grid needs finite bounds in seconds.",
                    fix="pass finite t_start and t_stop, or omit them to span "
                    "the spikes.",
                )
            )
        windows = np.array([[t_start, t_stop]], dtype=np.float64)
    left, right = _time_bins_in_windows(windows, dt)
    if left.size == 0:
        longest = float(np.max(np.diff(windows, axis=1), initial=0.0))
        raise ValueError(
            _format_error(
                f"Window span ({longest}) is smaller than one bin dt ({dt}); no whole time bin fits.",
                fix=f"use dt <= {longest:g} or widen the time windows",
                why="Why: each analyzed window must contain at least one full time bin.",
            )
        )
    counts = _count_spikes_in_time_bins(trains, left, right)
    bin_centers = left + dt / 2.0
    if orient == "neuron_x_time":
        counts = counts.T
    elif orient != "time_x_neuron":
        raise ValueError(
            f"orient must be 'time_x_neuron' or 'neuron_x_time', got {orient!r}."
        )
    return counts, bin_centers
