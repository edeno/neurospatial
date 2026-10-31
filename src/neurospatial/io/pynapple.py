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
    pynapple ``Tsd`` (1-D) / ``TsdFrame`` (2-D).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["from_pynapple", "to_pynapple"]

_IMPORT_ERROR_MSG = (
    "This function requires the optional 'pynapple' extra; install it with "
    "`pip install neurospatial[pynapple]` (or `uv add neurospatial[pynapple]`)."
)

_SPATIAL_AXES = ("x", "y", "z")


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


def _epochs_of(support: Any) -> NDArray[np.float64]:
    """Return a pynapple ``IntervalSet`` as an owned ``(n, 2)`` float array."""
    return np.column_stack(
        [
            np.array(support.start, dtype=np.float64),
            np.array(support.end, dtype=np.float64),
        ]
    )


def _refuse_permuted_spatial_columns(columns: Any) -> None:
    """Refuse a TsdFrame whose columns are x/y(/z) in another order."""
    labels = [str(c) for c in columns]
    lowered = [label.lower() for label in labels]
    spatial = set(lowered) <= set(_SPATIAL_AXES) and len(set(lowered)) == len(lowered)
    if not spatial or lowered == sorted(lowered):
        return
    canonical = sorted(lowered)
    ordered = sorted(labels, key=str.lower)
    raise ValueError(
        f"TsdFrame columns {labels} are spatial axes not in x, y, z order.\n"
        "Why: neurospatial reads positions by column position, so these values "
        f"would be used as ({', '.join(lowered)}) where every analysis expects "
        f"({', '.join(canonical)}), silently swapping the axes.\n"
        f"Fix: reorder the columns first, from_pynapple(tsdframe.loc[{ordered}])."
    )


def from_pynapple(obj: Any) -> tuple[Any, ...] | NDArray[np.float64]:
    """Convert a pynapple object to plain NumPy arrays and its epochs.

    Each time series comes back with its pynapple ``time_support`` as
    ``epochs``, a ``(n_epochs, 2)`` array of ``[start, stop]`` rows in
    seconds. Pass it as ``epochs=`` (or, for spikes, ``spike_window=``) so the
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
        For an ``IntervalSet``: one ``[start, end]`` row per interval.

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
    pynapple intervals include their end, while neurospatial windows are
    half-open ``[start, stop)``. The bounds are returned unchanged, so a
    sample or spike exactly at an epoch end (for example the last spike of
    a support built from first to last spike) falls outside the epochs in
    neurospatial analyses.

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
        return _epochs_of(obj)
    if isinstance(obj, nap.Ts):
        return np.array(obj.t, dtype=np.float64), _epochs_of(obj.time_support)
    if isinstance(obj, (nap.Tsd, nap.TsdFrame)):
        if isinstance(obj, nap.TsdFrame):
            _refuse_permuted_spatial_columns(obj.columns)
        return (
            np.array(obj.t, dtype=np.float64),
            np.array(obj.values, dtype=np.float64),
            _epochs_of(obj.time_support),
        )
    if isinstance(obj, nap.TsGroup):
        from neurospatial._spikes import _as_spike_trains_with_ids

        trains, unit_ids = _as_spike_trains_with_ids(obj)
        return (
            [np.array(train) for train in trains],
            unit_ids,
            _epochs_of(obj.time_support),
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
) -> Any:
    """Convert a decoded track (or ``times`` + ``values``) to a pynapple object.

    Accepts either explicit ``(times, values)`` arrays, or a single decode
    result exposing ``.times`` and ``.map_position`` (duck-typed, e.g. a
    :class:`~neurospatial.decoding.DecodingResult`). Returns a pynapple ``Tsd``
    for 1-D values or a ``TsdFrame`` for 2-D values.

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
        If ``times`` is not 1-D, finite and strictly increasing, ``values`` is
        not 1-D or 2-D, ``times`` and ``values`` differ in length, or
        ``columns`` does not have one label per value column. These checks run
        before pynapple is imported.

    Examples
    --------
    >>> from neurospatial.io import to_pynapple  # doctest: +SKIP
    >>> tsdframe = to_pynapple(result)  # a DecodingResult MAP track  # doctest: +SKIP
    >>> tsd = to_pynapple(times, linear_positions)  # doctest: +SKIP
    """
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
        times = np.asarray(result.times, dtype=np.float64)
        values = np.asarray(result.map_position, dtype=np.float64)
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

    # Import only after validation, so invalid input never reaches pynapple.
    nap = _require_pynapple()

    if values.ndim == 1:
        return nap.Tsd(t=times, d=values)
    return nap.TsdFrame(t=times, d=values, columns=columns)
