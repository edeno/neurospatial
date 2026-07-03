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

__all__ = ["in_epochs", "restrict", "restrict_spike_trains"]

_Closed = Literal["both", "left", "right", "neither"]


def _stack_start_end(start: Any, end: Any) -> NDArray[np.float64]:
    """Column-stack ``(start, end)`` into an ``(n, 2)`` interval array.

    Scalars broadcast to one interval; equal-length 1-D arrays give ``n``
    intervals. Raises when the two are not the same length.
    """
    start_arr = np.atleast_1d(np.asarray(start, dtype=np.float64))
    end_arr = np.atleast_1d(np.asarray(end, dtype=np.float64))
    if start_arr.ndim != 1 or end_arr.ndim != 1:
        raise ValueError(
            "epoch start and end must be scalars or 1-D arrays, got shapes "
            f"{start_arr.shape} and {end_arr.shape}."
        )
    if start_arr.shape[0] != end_arr.shape[0]:
        raise ValueError(
            "epoch start and end must have the same length, got "
            f"{start_arr.shape[0]} and {end_arr.shape[0]}."
        )
    if start_arr.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    return np.column_stack([start_arr, end_arr])


def _sequence_length(x: Any) -> int | None:
    """Return the length of an array-like sequence, or ``None`` if ``x`` is scalar.

    A 1-D+ ``ndarray`` reports its first-axis length; a 0-dim ``ndarray`` (and a
    plain ``float``/``int``) is scalar and reports ``None``. Any other object
    exposing ``__len__`` (list, tuple) reports ``len(x)``.
    """
    if isinstance(x, np.ndarray):
        return int(x.shape[0]) if x.ndim >= 1 else None
    if hasattr(x, "__len__"):
        return len(x)
    return None


def _as_intervals(epochs: Any) -> NDArray[np.float64]:
    """Normalize any accepted ``epochs`` form to an ``(n, 2)`` interval array.

    Accepts, in order of precedence:

    - an **``IntervalSet``-like** object -- duck-typed via
      ``hasattr(epochs, "start") and hasattr(epochs, "end")`` (e.g. a pynapple
      ``IntervalSet``); converted with ``np.column_stack([epochs.start,
      epochs.end])``. This branch never imports pynapple and never
      ``isinstance``-checks a pynapple type.
    - a **2-tuple/list ``(start, end)`` of scalars** -> one interval;
    - **two 1-D arrays** passed as ``(starts, ends)`` -> ``n`` intervals (when
      the two arrays have length != 2 -- length-2 arrays are the ambiguous case
      below);
    - a single **``(n, 2)`` NumPy array** -> ``n`` interval rows (must be an
      ndarray, not a bare nested list).

    The one genuinely ambiguous form **raises**: a length-2 tuple/list whose two
    elements are *themselves* length-2 sequences (e.g. ``[[0, 5], [10, 15]]`` or
    ``(np.array([0, 10]), np.array([5, 15]))``). It could mean two ``(start,
    end)`` interval rows *or* two parallel ``(starts, ends)`` arrays, and there
    is no way to tell which -- so the user must disambiguate by passing an
    ``(n, 2)`` NumPy array (interval rows) or explicit 1-D ``start``/``end``
    arrays. Scalars, ``(n, 2)`` ndarrays, ``IntervalSet``-like objects, and
    parallel arrays whose length is not 2 are all unambiguous and never raise.

    Empty epochs (0 intervals) are allowed and normalize to shape ``(0, 2)``
    (they select nothing).

    Parameters
    ----------
    epochs : IntervalSet-like, tuple, list, or ndarray
        The epochs specification (see above).

    Returns
    -------
    ndarray, shape (n_intervals, 2)
        ``[start, end]`` rows, dtype ``float64``.

    Raises
    ------
    ValueError
        If the input is the ambiguous length-2-pair-of-length-2-sequences form
        (see above), any ``start > end`` (naming the offending interval), any
        endpoint is non-finite, or the array form is not ``(n, 2)``.
    """
    # 1. IntervalSet-like (duck-typed): pynapple IntervalSet and friends. Checked
    #    first because NumPy arrays / tuples never carry .start / .end.
    if hasattr(epochs, "start") and hasattr(epochs, "end"):
        intervals = _stack_start_end(epochs.start, epochs.end)
    # 2. A length-2 tuple/list. Scalars -> one (start, end) interval; two 1-D
    #    arrays -> parallel (starts, ends). But if BOTH elements are themselves
    #    length-2 sequences the input is irreducibly ambiguous (two interval rows
    #    vs two parallel length-2 arrays) -> raise and force disambiguation. (An
    #    (n, 2) array is handled in branch 3: it must be an ndarray, not a nested
    #    list.)
    elif isinstance(epochs, (tuple, list)) and len(epochs) == 2:
        start, end = epochs
        if _sequence_length(start) == 2 and _sequence_length(end) == 2:
            raise ValueError(
                "Ambiguous `epochs`: a length-2 pair of length-2 sequences "
                "could mean two `(start, end)` interval rows or two parallel "
                "`(starts, ends)` arrays. Pass `np.asarray(epochs)` with shape "
                "`(n, 2)` for interval rows (e.g. `np.array([[0, 5], [10, "
                "15]])`), or pass explicit 1-D `start`/`end` arrays."
            )
        intervals = _stack_start_end(start, end)
    # 3. Otherwise treat as an array of intervals: (n, 2), or empty.
    else:
        arr = np.asarray(epochs, dtype=np.float64)
        if arr.size == 0:
            return np.empty((0, 2), dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(
                "epochs array must have shape (n_intervals, 2) of [start, end] "
                f"rows, got shape {arr.shape}. Pass (start, end) scalars/arrays "
                "for the tuple form, or an (n, 2) NumPy array."
            )
        intervals = arr

    if intervals.shape[0] == 0:
        return intervals

    if not np.all(np.isfinite(intervals)):
        raise ValueError(
            "epoch bounds must be finite; found a non-finite (NaN/inf) start or "
            "end. Check the epochs passed to restrict/in_epochs."
        )
    bad = intervals[:, 0] > intervals[:, 1]
    if np.any(bad):
        i = int(np.flatnonzero(bad)[0])
        raise ValueError(
            f"each epoch must have start <= end, but interval {i} is "
            f"[{intervals[i, 0]}, {intervals[i, 1]}]."
        )
    return intervals


def _mask_in_intervals(
    t: NDArray[np.float64],
    intervals: NDArray[np.float64],
    *,
    closed: _Closed,
) -> NDArray[np.bool_]:
    """Boolean mask of ``t`` against PRE-normalized ``(n, 2)`` intervals.

    The vectorized core shared by :func:`in_epochs` and
    :func:`restrict_spike_trains`. ``intervals`` must already be an ``(n, 2)``
    ``float64`` array (as :func:`_as_intervals` returns), so a caller restricting
    many trains against one epoch set can normalize **once** and reuse this per
    train instead of re-normalizing the same intervals for every unit.

    Parameters
    ----------
    t : ndarray
        Timestamps to test (any shape; the mask has the same shape).
    intervals : ndarray, shape (n_intervals, 2)
        Normalized ``[start, end]`` rows (from :func:`_as_intervals`).
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
    ``epochs`` (this helper runs after ``_as_intervals``).
    """
    if closed not in ("both", "left", "right", "neither"):
        raise ValueError(
            f"closed must be one of 'both', 'left', 'right', 'neither', got {closed!r}."
        )

    t_arr = np.asarray(t, dtype=np.float64)

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
    epochs : IntervalSet-like, tuple, list, or ndarray
        Epochs in any form accepted by :func:`_as_intervals`
        (``(start, end)`` scalars/arrays, an ``(n, 2)`` array, or a
        duck-typed ``IntervalSet``). Empty epochs -> all-``False``.
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
    # Validate ``closed`` up front -- before ``_as_intervals`` and the
    # empty-epochs early return -- so a bad value always raises (even with no
    # intervals) AND is reported before a malformed ``epochs``.
    if closed not in ("both", "left", "right", "neither"):
        raise ValueError(
            f"closed must be one of 'both', 'left', 'right', 'neither', got {closed!r}."
        )

    intervals = _as_intervals(epochs)
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
    epochs : IntervalSet-like, tuple, list, or ndarray
        Epochs in any form accepted by :func:`_as_intervals`.
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
    epochs : IntervalSet-like, tuple, list, or ndarray
        Epochs in any form accepted by :func:`_as_intervals`. Empty epochs ->
        every returned train is empty (but present, preserving order/count).
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
    intervals = _as_intervals(epochs)
    out: list[NDArray[np.float64]] = []
    for train in trains:
        t = np.asarray(train, dtype=np.float64)
        out.append(t[_mask_in_intervals(t, intervals, closed=closed)])
    return out
