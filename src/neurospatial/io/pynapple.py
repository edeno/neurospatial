"""pynapple ingress / egress adapters (optional dependency).

These two functions are the only code in neurospatial that imports pynapple,
and they import it lazily: ``import pynapple`` runs *inside* the functions, so
the package and the array path keep working when pynapple is not installed.
Analysis functions also accept a pynapple ``TsGroup`` / ``Ts`` as spike input
and an ``IntervalSet`` as ``epochs=`` / ``spike_window=``, by duck typing and
without importing pynapple; they read only timestamps, unit labels and
interval bounds. :func:`from_pynapple` additionally returns each object's
``time_support`` as epochs, so gaps between recording epochs are not lost.

Functions
---------
from_pynapple
    Convert a pynapple ``Ts`` / ``Tsd`` / ``TsdFrame`` / ``TsGroup`` /
    ``IntervalSet`` to plain arrays plus its time support as ``(n_epochs, 2)``
    epochs.
to_pynapple
    Convert a decoded MAP track (or explicit ``times`` + ``values``) to a
    pynapple ``Tsd`` (1-D) / ``TsdFrame`` (2-D) whose time support is the
    decode's spike window or the given ``epochs``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from neurospatial._pynapple import interval_rows, refuse_permuted_spatial_columns

__all__ = ["from_pynapple", "to_pynapple"]

_IMPORT_ERROR_MSG = (
    "This function requires the optional 'pynapple' extra; install it with "
    "`pip install neurospatial[pynapple]` (or `uv add neurospatial[pynapple]`)."
)


def _require_pynapple() -> Any:
    """Import and return the ``pynapple`` module, or raise a clear ImportError.

    The import is lazy so importing :mod:`neurospatial.io` never requires
    pynapple. Raises a clear, actionable :class:`ImportError` naming the extra
    when pynapple is absent.
    """
    try:
        import pynapple
    except ImportError as exc:  # pragma: no cover - exercised only when absent
        raise ImportError(_IMPORT_ERROR_MSG) from exc
    return pynapple


def from_pynapple(obj: Any) -> tuple[Any, ...] | NDArray[np.float64]:
    """Convert a pynapple object to plain NumPy arrays and its epochs.

    Each time series comes back with its pynapple ``time_support`` as
    ``epochs``, a ``(n_epochs, 2)`` array of half-open ``[start, stop)`` rows
    in seconds. Pass it as ``epochs=`` (or, for spikes, ``spike_window=``) so the
    gaps between recording epochs stay out of occupancy and rates.

    Parameters
    ----------
    obj : pynapple Ts, Tsd, TsdFrame, TsGroup, or IntervalSet
        The pynapple object to convert.

    Returns
    -------
    spike_times, epochs : ndarray, shapes (n_spikes,) and (n_epochs, 2)
        For a ``Ts``: its timestamps and time support.
    times, values, epochs : ndarray, shapes (n_samples,), (n_samples,) or \
(n_samples, n_columns), and (n_epochs, 2)
        For a ``Tsd`` (1-D values) or ``TsdFrame`` (2-D values, columns in
        ``tsdframe.columns`` order).
    trains, unit_ids, epochs : list of ndarray, ndarray of shape (n_units,), \
ndarray of shape (n_epochs, 2)
        For a ``TsGroup``: one spike-time array per unit in group order, the
        group's unit labels, and the group's time support.
    epochs : ndarray, shape (n_epochs, 2)
        For an ``IntervalSet``: one ``[start, stop)`` row per interval.

    All arrays are float64 copies (``unit_ids`` keeps its label type), so
    editing them never changes the pynapple object.

    Raises
    ------
    ImportError
        If pynapple is not installed.
    TypeError
        If ``obj`` is not a pynapple ``Ts``, ``Tsd``, ``TsdFrame``, ``TsGroup``
        or ``IntervalSet`` (a ``TsdTensor`` has no spatial reading).
    ValueError
        If a ``TsdFrame``'s columns are spatial axis names (``x``, ``y``,
        ``z``) in another order, such as ``["y", "x"]``.

    Notes
    -----
    pynapple intervals ``[start, end]`` include their end, while neurospatial
    windows are half-open ``[start, stop)``. Each ``stop`` is the next float
    above ``end`` (``np.nextafter(end, np.inf)``), so the epochs hold exactly
    the timestamps pynapple holds: a sample or spike at ``end``, such as the
    last spike of a support built from first to last spike, stays inside.
    An ``IntervalSet`` passed directly as ``epochs=`` or ``spike_window=`` is
    converted the same way.

    A ``TsGroup`` built from spike times alone gets a default time support
    from its first to its last spike, which is not recording coverage. Use
    its epochs as ``spike_window=`` only when the group's time support was
    set to the acquisition intervals.

    Examples
    --------
    >>> from neurospatial.io import from_pynapple  # doctest: +SKIP
    >>> times, positions, epochs = from_pynapple(tsdframe)  # doctest: +SKIP
    >>> trains, unit_ids, recorded = from_pynapple(tsgroup)  # doctest: +SKIP
    >>> rates = compute_spatial_rates(  # doctest: +SKIP
    ...     env, trains, times, positions, epochs=epochs, spike_window=recorded,
    ...     unit_ids=unit_ids,
    ... )
    """
    nap = _require_pynapple()

    if isinstance(obj, nap.IntervalSet):
        return interval_rows(obj)
    if isinstance(obj, nap.Ts):
        return np.array(obj.t, dtype=np.float64), interval_rows(obj.time_support)
    if isinstance(obj, (nap.Tsd, nap.TsdFrame)):
        if isinstance(obj, nap.TsdFrame):
            refuse_permuted_spatial_columns(obj.columns, fix="from_pynapple({frame})")
        return (
            np.array(obj.t, dtype=np.float64),
            np.array(obj.values, dtype=np.float64),
            interval_rows(obj.time_support),
        )
    if isinstance(obj, nap.TsGroup):
        from neurospatial._spikes import _as_spike_trains_with_ids

        trains, unit_ids = _as_spike_trains_with_ids(obj)
        return (
            [np.array(train) for train in trains],
            unit_ids,
            interval_rows(obj.time_support),
        )

    raise TypeError(
        f"from_pynapple does not recognize {type(obj).__name__!r}.\n"
        "Why: it converts the pynapple types with a spatial-analysis reading.\n"
        "Fix: pass a pynapple Ts, Tsd, TsdFrame, TsGroup or IntervalSet."
    )


def to_pynapple(
    times: Any,
    values: NDArray[np.float64] | None = None,
    *,
    columns: Any = None,
    epochs: Any = None,
) -> Any:
    """Convert a decoded track (or ``times`` + ``values``) to a pynapple object.

    Accepts either explicit ``(times, values)`` arrays, or a single decode
    result exposing ``.times`` and ``.map_position`` (duck-typed, e.g. a
    :class:`~neurospatial.decoding.DecodingResult`). Returns a pynapple ``Tsd``
    for 1-D values or a ``TsdFrame`` for 2-D values. Its ``time_support`` is
    ``epochs`` when given, else the decode result's ``spike_window``, else
    pynapple's default, one interval from the first to the last timestamp.

    Parameters
    ----------
    times : array-like or decode result
        Finite, strictly increasing timestamps (seconds), or a decode result
        exposing ``.times`` and ``.map_position`` (in which case ``values``
        must be ``None``). Times are never sorted here: pynapple would sort
        them without reordering ``values``.
    values : NDArray[np.float64] or None, default=None
        Values sampled at ``times``, shape ``(n,)`` or ``(n, n_dims)``. Required
        when ``times`` is a timestamp array; must be ``None`` when ``times`` is a
        decode result.
    columns : sequence, optional
        Column labels for the resulting ``TsdFrame`` (2-D values only), one per
        value column.
    epochs : (start, stop), array-like of shape (n_epochs, 2), IntervalSet, \
or None, optional
        Time support of the output, in seconds: the epochs over which the
        values were observed, so gaps between them are not read as recorded
        time. Rows are half-open ``[start, stop)``, sorted and touching rows
        merged, as for every ``epochs=`` argument; each pynapple interval ends
        at the float just below ``stop``, so it holds the same timestamps.
        Every timestamp must lie inside a row (pynapple would silently drop
        the others). Default ``None`` uses a decode result's ``spike_window``
        when it has one.

    Returns
    -------
    pynapple.Tsd or pynapple.TsdFrame
        ``Tsd`` for 1-D values, ``TsdFrame`` for 2-D values.

    Raises
    ------
    ImportError
        If pynapple is not installed.
    TypeError
        If ``values`` is ``None`` and ``times`` is not a decode result exposing
        ``.times`` and ``.map_position``.
    ValueError
        If ``times`` is missing from the decode result, not 1-D, finite and
        strictly increasing, ``values`` is not 1-D or 2-D, ``times`` and
        ``values`` differ in length, ``columns`` does not have one label per
        value column, ``epochs`` is not a valid set of windows, or a timestamp
        lies outside ``epochs``. These checks run before pynapple is imported.

    Notes
    -----
    A decode result records its ``spike_window`` but not the ``epochs`` it
    was decoded over. If you restricted decoding with ``epochs=``, pass the
    same windows here so the gaps are kept.

    Examples
    --------
    >>> from neurospatial.io import to_pynapple  # doctest: +SKIP
    >>> tsdframe = to_pynapple(result)  # a DecodingResult MAP track  # doctest: +SKIP
    >>> tsd = to_pynapple(times, linear_positions, epochs=run_epochs)  # doctest: +SKIP
    """
    from neurospatial._intervals import as_intervals, stop_to_pynapple_end

    if values is None:
        # Duck-typed decode result: pull the MAP track off it. Guard the
        # duck-type up front so a non-result `times` yields an actionable error
        # naming the expected inputs, not a bare AttributeError on `.times`.
        result = times
        if not (hasattr(result, "times") and hasattr(result, "map_position")):
            raise TypeError(
                "to_pynapple(times) with values=None expects a decode result "
                "exposing `.times` and `.map_position` (e.g. a "
                "neurospatial.decoding.DecodingResult). To convert plain arrays, "
                "pass to_pynapple(times, values) with an explicit `values` array."
            )
        if result.times is None:
            raise ValueError(
                "The decode result has no `times`.\n"
                "Why: a pynapple Tsd/TsdFrame needs one timestamp per decoded "
                "time bin.\n"
                "Fix: pass to_pynapple(bin_times, result.map_position) with the "
                "time-bin centers in seconds."
            )
        times = np.asarray(result.times, dtype=np.float64)
        values = np.asarray(result.map_position, dtype=np.float64)
        if epochs is None:
            epochs = getattr(result, "spike_window", None)
    else:
        times = np.asarray(times, dtype=np.float64)
        values = np.asarray(values, dtype=np.float64)

    # Prevalidate at this boundary so bad shapes raise an actionable
    # neurospatial ValueError rather than a raw pynapple AssertionError from
    # deep inside nap.Tsd / nap.TsdFrame.
    if times.ndim != 1:
        raise ValueError(
            f"`times` must be 1-D, got shape {times.shape}.\n"
            "  WHY: pynapple indexes a Tsd/TsdFrame by a 1-D time axis.\n"
            "  Fix: pass a 1-D array of timestamps."
        )
    if values.ndim not in (1, 2):
        raise ValueError(
            f"`values` must be 1-D or 2-D, got {values.ndim}-D (shape "
            f"{values.shape}).\n"
            "  WHY: a Tsd holds 1-D values, a TsdFrame holds 2-D "
            "(n_samples, n_columns) values.\n"
            "  Fix: pass values shaped (n,) or (n, n_columns)."
        )
    if len(times) != len(values):
        raise ValueError(
            f"`times` and `values` must have the same length, got "
            f"{len(times)} timestamps and {len(values)} value rows.\n"
            "  WHY: each value (row) is sampled at one timestamp.\n"
            "  Fix: pass times and values with matching first-axis length."
        )
    non_finite = np.flatnonzero(~np.isfinite(times))
    if non_finite.size:
        raise ValueError(
            f"`times` must be finite, but {non_finite.size} timestamp(s) are NaN "
            f"or infinite (first at index {int(non_finite[0])}).\n"
            "Why: a non-finite timestamp has no place on a pynapple time axis.\n"
            "Fix: keep = np.isfinite(times); to_pynapple(times[keep], "
            "values[keep])"
        )
    non_increasing = np.flatnonzero(times[1:] <= times[:-1])
    if non_increasing.size:
        i = int(non_increasing[0])
        raise ValueError(
            f"`times` must be strictly increasing, but timestamps at indices {i} "
            f"and {i + 1} are {times[i]!r} and {times[i + 1]!r}.\n"
            "Why: pynapple sorts timestamps without reordering the values, "
            "which would pair each value with another sample's time; repeated "
            "timestamps have no defined order.\n"
            'Fix: order = np.argsort(times, kind="stable"); '
            "to_pynapple(times[order], values[order]), after resolving any "
            "repeated timestamps."
        )
    if values.ndim == 2 and columns is not None and len(columns) != values.shape[1]:
        raise ValueError(
            f"`columns` has {len(columns)} labels but values has "
            f"{values.shape[1]} value columns.\n"
            "Why: pynapple replaces a mismatched label list with integer "
            "column names.\n"
            f"Fix: pass exactly {values.shape[1]} labels, or omit columns=."
        )

    support = as_intervals(epochs, name="epochs")
    if support is not None:
        # Half-open membership, the same set as the inclusive pynapple
        # intervals built below.
        row = np.searchsorted(support[:, 0], times, side="right") - 1
        inside = (row >= 0) & (times < support[np.maximum(row, 0), 1])
        outside = np.flatnonzero(~inside)
        if outside.size:
            i = int(outside[0])
            raise ValueError(
                f"{outside.size} of {len(times)} timestamp(s) lie outside "
                f"`epochs` (first at index {i}, t={float(times[i])!r}).\n"
                "Why: pynapple silently drops samples outside the time "
                "support.\n"
                "Fix: pass epochs that cover every timestamp, or select the "
                "samples first, e.g. with neurospatial.behavior.restrict(times, "
                "values, epochs=epochs)."
            )

    # Import only after validation, so invalid input never reaches pynapple.
    nap = _require_pynapple()

    time_support = (
        None
        if support is None
        else nap.IntervalSet(
            start=support[:, 0], end=stop_to_pynapple_end(support[:, 1])
        )
    )
    if values.ndim == 1:
        return nap.Tsd(t=times, d=values, time_support=time_support)
    return nap.TsdFrame(t=times, d=values, columns=columns, time_support=time_support)
