"""Array-native epoch selection (``restrict`` / ``in_epochs``).

"Give me my running periods" / "give me trial N" -- restrict time-series and
spike data to a set of time intervals (epochs). Epochs are plain
``(start, end)`` arrays, so this stays array-first and never imports pynapple;
a pynapple ``IntervalSet`` is nonetheless accepted **transparently** because it
is *duck-typed* (it exposes ``.start`` / ``.end`` arrays), not ``isinstance``-d.

Functions
---------
restrict
    The headline one-liner: ``t, pos = restrict(times, positions,
    epochs=run_epochs)``. Slices ``times`` and any number of arrays **aligned to
    ``times``** (e.g. ``positions``) by the same in-epoch mask, order preserved.
    With no extra arrays it restricts an event-time array by its own timestamps
    (``restrict(spike_train, epochs=...)`` -> the in-epoch spikes).
in_epochs
    The boolean mask primitive: ``True`` where ``t`` falls inside **any**
    interval. Inclusive endpoints by default (``closed="both"``), matching
    :mod:`neurospatial.behavior.segmentation` and pynapple.
restrict_spike_trains
    Ragged per-unit spike times are **not** aligned to a common ``times``; each
    train is restricted by its **own** timestamps. Accepts a plain sequence of
    trains or a :class:`~neurospatial.encoding.SpikeTrains` container.

Aligned vs ragged
-----------------
``restrict`` is for arrays that share one time axis (position samples, a single
spike train). ``restrict_spike_trains`` is for a *ragged* collection where each
unit has its own timestamps and there is no shared ``times`` -- so each train is
masked against itself.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from neurospatial._intervals import as_intervals

__all__ = ["in_epochs", "restrict", "restrict_spike_trains"]

_Closed = Literal["both", "left", "right", "neither"]


def _mask_in_intervals(
    t: NDArray[np.float64],
    intervals: NDArray[np.float64] | None,
    *,
    closed: _Closed,
) -> NDArray[np.bool_]:
    """Boolean mask of ``t`` against PRE-normalized ``(n, 2)`` intervals.

    The vectorized core shared by :func:`in_epochs` and
    :func:`restrict_spike_trains`. ``intervals`` must already be an ``(n, 2)``
    ``float64`` array (as :func:`~neurospatial._intervals.as_intervals` returns), so a caller restricting
    many trains against one epoch set can normalize **once** and reuse this per
    train instead of re-normalizing the same intervals for every unit.

    Parameters
    ----------
    t : ndarray
        Timestamps to test (any shape; the mask has the same shape).
    intervals : ndarray, shape (n_intervals, 2)
        Normalized ``[start, end]`` rows (from :func:`~neurospatial._intervals.as_intervals`).
    closed : {"both", "left", "right", "neither"}
        Which endpoints are inclusive.

    Returns
    -------
    ndarray of bool
        Same shape as ``t``; ``True`` where ``t`` is inside any interval.

    Notes
    -----
    ``closed`` is validated here too, so this stays safe to call directly (e.g.
    per train from :func:`restrict_spike_trains`); :func:`in_epochs` validates it
    a second time, up front, only to report a bad ``closed`` *before* a malformed
    ``epochs`` (this helper runs after ``as_intervals``).
    """
    if closed not in ("both", "left", "right", "neither"):
        raise ValueError(
            f"closed must be one of 'both', 'left', 'right', 'neither', got {closed!r}."
        )

    t_arr = np.asarray(t, dtype=np.float64)

    if intervals is None:
        return np.ones(t_arr.shape, dtype=np.bool_)
    if intervals.shape[0] == 0:
        return np.zeros(t_arr.shape, dtype=np.bool_)

    starts = intervals[:, 0]
    ends = intervals[:, 1]
    # Broadcast samples (..., 1) against intervals (n_intervals,) -> (..., n).
    tt = t_arr[..., np.newaxis]
    if closed == "both":
        inside = (tt >= starts) & (tt <= ends)
    elif closed == "left":
        inside = (tt >= starts) & (tt < ends)
    elif closed == "right":
        inside = (tt > starts) & (tt <= ends)
    else:  # "neither" (validated above)
        inside = (tt > starts) & (tt < ends)
    mask: NDArray[np.bool_] = np.any(inside, axis=-1)
    return mask


def in_epochs(
    t: NDArray[np.float64],
    epochs: Any,
    *,
    closed: _Closed = "both",
) -> NDArray[np.bool_]:
    """Boolean mask: ``True`` where each element of ``t`` falls in any epoch.

    Vectorized over samples (broadcasting, no Python loop) and taking the
    **union** across intervals (overlapping intervals behave as their union).

    Parameters
    ----------
    t : ndarray
        Timestamps to test (any shape; the mask has the same shape).
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Windows in seconds, on the same clock as the input timestamps. Rows
        are sorted and overlapping or touching rows merge. ``None`` leaves
        data unrestricted. Empty windows, non-finite endpoints, and rows with
        ``stop <= start`` raise ``ValueError`` with a fix. Parallel starts/ends
        arrays are not an input form; pass an ``(n, 2)`` array instead.
        ``closed`` controls point membership in the normalized windows.
    closed : {"both", "left", "right", "neither"}, optional
        Which endpoints are inclusive. Default ``"both"`` (``start <= t <=
        end``), matching :mod:`neurospatial.behavior.segmentation` and pynapple.

    Returns
    -------
    ndarray of bool
        Same shape as ``t``; ``True`` where ``t`` is inside any interval.

    Notes
    -----
    NaN timestamps evaluate ``False`` at every ``>=``/``<=``/``>``/``<``
    comparison, so they are (deliberately) excluded from every epoch.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.behavior import in_epochs
    >>> in_epochs(np.array([0.0, 2.5, 5.0, 6.0]), (0.0, 5.0))
    array([ True,  True,  True, False])
    """
    # Validate ``closed`` up front -- before ``as_intervals`` and the
    # empty-epochs early return -- so a bad value always raises (even with no
    # intervals) AND is reported before a malformed ``epochs``.
    if closed not in ("both", "left", "right", "neither"):
        raise ValueError(
            f"closed must be one of 'both', 'left', 'right', 'neither', got {closed!r}."
        )

    intervals = as_intervals(epochs, name="epochs")
    return _mask_in_intervals(t, intervals, closed=closed)


def restrict(
    times: NDArray[np.float64],
    *arrays: NDArray[Any],
    epochs: Any,
    closed: _Closed = "both",
) -> NDArray[Any] | tuple[NDArray[Any], ...]:
    """Restrict ``times`` and time-aligned arrays to a set of epochs.

    The headline one-liner::

        t, pos = restrict(times, positions, epochs=run_epochs)

    ``times`` is the reference 1-D time axis. Each of ``*arrays`` must be
    **aligned to ``times``** (``len(arr) == len(times)``; e.g. ``positions`` of
    shape ``(n, n_dims)``). All are sliced by the single mask
    ``in_epochs(times, epochs, closed=closed)``, so alignment and order are
    preserved.

    With **no** extra arrays, ``restrict`` restricts an event-time array by its
    own timestamps -- ``restrict(spike_train, epochs=...)`` returns the in-epoch
    spikes (here ``times`` *is* the spike train). For ragged per-unit spikes
    (each unit its own timestamps, no shared axis), use
    :func:`restrict_spike_trains` instead.

    Parameters
    ----------
    times : ndarray, shape (n,)
        Reference time array.
    *arrays : ndarray
        Zero or more arrays aligned to ``times`` (first axis length ``n``).
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Windows in seconds, on the same clock as the input timestamps. Rows
        are sorted and overlapping or touching rows merge. ``None`` leaves
        data unrestricted. Empty windows, non-finite endpoints, and rows with
        ``stop <= start`` raise ``ValueError`` with a fix. Parallel starts/ends
        arrays are not an input form; pass an ``(n, 2)`` array instead.
        ``closed`` controls point membership in the normalized windows.
    closed : {"both", "left", "right", "neither"}, optional
        Endpoint inclusivity, forwarded to :func:`in_epochs`. Default
        ``"both"``.

    Returns
    -------
    ndarray or tuple of ndarray
        If no extra arrays were passed, the masked ``times`` (a bare array).
        Otherwise ``(times_kept, *arrays_kept)`` -- each sliced by the same
        mask, order preserved.

    Raises
    ------
    ValueError
        If any array's first-axis length differs from ``len(times)`` (the error
        names the offending positional index).

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.behavior import restrict
    >>> times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    >>> positions = np.column_stack([times, times])
    >>> t, pos = restrict(times, positions, epochs=(1.0, 3.0))
    >>> t
    array([1., 2., 3.])
    """
    times_arr = np.asarray(times)
    n = len(times_arr)

    coerced: list[NDArray[Any]] = []
    for i, arr in enumerate(arrays):
        arr_np = np.asarray(arr)
        if len(arr_np) != n:
            raise ValueError(
                f"array at position {i} has len {len(arr_np)} but must be "
                f"aligned to times (len {n}); every *arrays entry must have the "
                "same first-axis length as times."
            )
        coerced.append(arr_np)

    mask = in_epochs(times_arr, epochs, closed=closed)

    times_kept = times_arr[mask]
    if not coerced:
        return times_kept
    return (times_kept, *(arr[mask] for arr in coerced))


def restrict_spike_trains(
    trains: Sequence[NDArray[np.float64]],
    epochs: Any,
    *,
    closed: _Closed = "both",
) -> list[NDArray[np.float64]]:
    """Restrict each ragged per-unit spike train by its **own** timestamps.

    Ragged per-unit spike times are not aligned to a common ``times`` axis, so
    each train is masked against itself: ``[t[in_epochs(t, epochs)] for t in
    trains]``.

    Parameters
    ----------
    trains : sequence of ndarray, or SpikeTrains
        Per-unit 1-D spike-time arrays. A
        :class:`~neurospatial.encoding.SpikeTrains` container is accepted --
        iterating it yields the per-unit train arrays.
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Windows in seconds, on the same clock as the input timestamps. Rows
        are sorted and overlapping or touching rows merge. ``None`` leaves
        data unrestricted. Empty windows, non-finite endpoints, and rows with
        ``stop <= start`` raise ``ValueError`` with a fix. Parallel starts/ends
        arrays are not an input form; pass an ``(n, 2)`` array instead.
        ``closed`` controls point membership in the normalized windows.
    closed : {"both", "left", "right", "neither"}, optional
        Endpoint inclusivity, forwarded to :func:`in_epochs`. Default
        ``"both"``.

    Returns
    -------
    list of ndarray
        One masked train per input train, in the same order. Always a plain
        ``list`` (type-stable), even when given a
        :class:`~neurospatial.encoding.SpikeTrains` -- ``SpikeTrains`` identity
        is preserved instead by ``Session.restrict``.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.behavior import restrict_spike_trains
    >>> trains = [np.array([0.1, 1.5, 2.9]), np.array([0.5, 3.0, 6.0])]
    >>> restrict_spike_trains(trains, (1.0, 3.5))
    [array([1.5, 2.9]), array([3.])]
    """
    # Normalize the intervals ONCE (not per train): a malformed epochs raises up
    # front, and each train is then masked against the same pre-normalized
    # ``(n, 2)`` array via ``_mask_in_intervals`` -- avoiding the per-unit
    # re-normalization that ``in_epochs`` would repeat on every call.
    intervals = as_intervals(epochs, name="epochs")
    out: list[NDArray[np.float64]] = []
    for train in trains:
        t = np.asarray(train, dtype=np.float64)
        out.append(t[_mask_in_intervals(t, intervals, closed=closed)])
    return out
