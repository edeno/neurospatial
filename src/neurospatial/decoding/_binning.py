"""Temporal binning of spike-time arrays into count matrices.

Provides :func:`bin_spikes_in_time`, the public primitive that turns a
sequence of per-neuron spike-time arrays into a regular time grid of spike
counts. It owns the time-grid construction (left edges plus ``dt / 2`` bin
centers) so the spike -> time-bin -> decode seam has a single, explicit home.

The ``orient`` argument makes the count-matrix axis order explicit, defusing
the silent transpose footgun between :func:`neurospatial.decoding.decode_position`
(which expects ``(n_time_bins, n_neurons)``) and the assembly functions in
:mod:`neurospatial.decoding.assemblies` (which expect
``(n_neurons, n_time_bins)``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray


def bin_spikes_in_time(
    spike_trains: Sequence[NDArray[np.float64]],
    dt: float,
    t_start: float | None = None,
    t_stop: float | None = None,
    *,
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
    orient : {"time_x_neuron", "neuron_x_time"}, optional
        Axis order of the returned ``counts`` matrix. ``"time_x_neuron"``
        (default) returns shape ``(n_time_bins, n_neurons)`` — the shape
        :func:`neurospatial.decoding.decode_position` expects.
        ``"neuron_x_time"`` returns shape ``(n_neurons, n_time_bins)`` — the
        shape the assembly functions in
        :mod:`neurospatial.decoding.assemblies` expect.

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
        span ``t_stop - t_start`` is smaller than a single bin ``dt``, or if
        ``orient`` is not one of the allowed values.

    Notes
    -----
    The number of bins is ``n_bins = floor((t_stop - t_start) / dt)`` and the
    edges are constructed deterministically as
    ``edges = t_start + dt * arange(n_bins + 1)``, so the grid spans exactly
    ``[t_start, t_start + n_bins * dt]`` and **no bin ever extends past
    ``t_stop``**. (A small ``1e-9`` slack is added inside the ``floor`` so an
    exact multiple of ``dt`` is not lost to floating-point error.) When the
    span is not an exact multiple of ``dt`` the trailing partial interval
    ``[t_start + n_bins * dt, t_stop)`` is dropped, and any spike at or beyond
    the final edge ``t_start + n_bins * dt`` is excluded — with one boundary
    exception below.

    Bins are half-open on the left, ``[edge, edge + dt)``, following
    :func:`numpy.histogram` semantics, except that the last bin is closed on
    the right: a spike falling exactly on the final edge
    ``t_start + n_bins * dt`` is counted in the last bin. ``bin_centers`` are
    ``edges[:-1] + dt / 2`` and stay aligned with the count rows.

    The grid must contain at least one whole bin; a span smaller than ``dt``
    raises :class:`ValueError`.

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
    if dt <= 0 or not np.isfinite(dt):
        raise ValueError(f"dt must be finite and > 0, got {dt!r}.")
    trains = [np.asarray(s, dtype=np.float64) for s in spike_trains]
    if t_start is None:
        t_start = min((s.min() for s in trains if s.size), default=0.0)
    if t_stop is None:
        # Auto-bound: extend one bin past the last spike so it lands inside
        # the final bin and a single-spike (or single-timestamp) train works.
        t_stop = max((s.max() for s in trains if s.size), default=t_start) + dt
    elif t_stop <= t_start:
        raise ValueError(f"t_stop ({t_stop}) must be > t_start ({t_start}).")
    n_bins = int(np.floor((t_stop - t_start) / dt + 1e-9))
    if n_bins < 1:
        raise ValueError(
            f"Span t_stop - t_start ({t_stop - t_start}) is smaller than one "
            f"bin dt ({dt}); no whole time bin fits."
        )
    edges = t_start + dt * np.arange(n_bins + 1, dtype=np.float64)
    counts = np.stack([np.histogram(s, bins=edges)[0] for s in trains], axis=1).astype(
        np.int64
    )  # (n_time_bins, n_neurons)
    bin_centers = edges[:-1] + dt / 2.0
    if orient == "neuron_x_time":
        counts = counts.T
    elif orient != "time_x_neuron":
        raise ValueError(
            f"orient must be 'time_x_neuron' or 'neuron_x_time', got {orient!r}."
        )
    return counts, bin_centers
