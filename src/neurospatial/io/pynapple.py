"""pynapple ingress / egress shim (optional dependency).

This is the **only** place in neurospatial that touches pynapple, and it does
so lazily: ``import pynapple`` happens *inside* the functions, so the package
imports and the array path keep working when pynapple is not installed. The
scientific modules never import pynapple; they consume the plain NumPy arrays
that :func:`from_pynapple` returns.

Functions
---------
from_pynapple
    Convert a pynapple ``TsGroup`` / ``Tsd`` / ``TsdFrame`` / ``IntervalSet`` to
    plain arrays: ``TsGroup`` -> ``(trains, unit_ids)``; ``Tsd`` / ``TsdFrame``
    -> ``(times, positions)``; ``IntervalSet`` -> ``(start, end)``.
to_pynapple
    Convert a decoded MAP track (or explicit ``times`` + ``values``) to a
    pynapple ``Tsd`` (1-D) / ``TsdFrame`` (2-D).
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

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


def from_pynapple(
    obj: Any,
) -> (
    tuple[list[NDArray[np.float64]], NDArray[Any]]
    | tuple[NDArray[np.float64], NDArray[np.float64]]
):
    """Convert a pynapple object to plain NumPy arrays.

    Dispatches by duck-typed attributes (never ``isinstance`` on a pynapple
    type, so the conversion is decoupled from pynapple's exact class hierarchy):

    - ``Tsd`` / ``TsdFrame`` (has ``.t`` and ``.values`` / ``.d``)
      -> ``(times, positions)``.
    - ``IntervalSet`` (has ``.start`` and ``.end``, no ``.t``) -> ``(start, end)``.
    - ``TsGroup`` (dict-like of unit id -> ``Ts``, has ``.index``)
      -> ``(trains, unit_ids)`` where ``trains`` is a list of per-unit 1-D
      timestamp arrays and ``unit_ids`` are the group's keys.

    Parameters
    ----------
    obj : pynapple Tsd, TsdFrame, IntervalSet, or TsGroup
        The pynapple object to convert.

    Returns
    -------
    tuple of ndarray
        ``(times, positions)`` for a ``Tsd`` / ``TsdFrame``, ``(start, end)``
        for an ``IntervalSet``, or ``(trains, unit_ids)`` for a ``TsGroup``.

    Raises
    ------
    ImportError
        If pynapple is not installed.
    TypeError
        If ``obj`` is not a recognized pynapple type.

    Examples
    --------
    >>> from neurospatial.io import from_pynapple  # doctest: +SKIP
    >>> times, positions = from_pynapple(tsdframe)  # doctest: +SKIP
    >>> trains, unit_ids = from_pynapple(tsgroup)  # doctest: +SKIP
    """
    # Ensure pynapple is installed even though dispatch is duck-typed: a caller
    # cannot hold a genuine pynapple object without it, but this keeps the error
    # actionable and consistent with ``to_pynapple``.
    _require_pynapple()

    # Tsd / TsdFrame: prefer .values, with the pynapple .d alias fallback.
    if hasattr(obj, "t") and (hasattr(obj, "values") or hasattr(obj, "d")):
        values = getattr(obj, "values", None)
        if values is None:
            values = obj.d
        return np.asarray(obj.t, dtype=np.float64), np.asarray(values, dtype=np.float64)

    # IntervalSet: epochs -> (start, end). No public adapter equivalent, so this
    # branch keeps its own coercion.
    if hasattr(obj, "start") and hasattr(obj, "end"):
        start = np.asarray(obj.start, dtype=np.float64)
        end = np.asarray(obj.end, dtype=np.float64)
        return start, end

    # TsGroup: dict-like of unit id -> Ts -> (trains, unit_ids). Delegate to the
    # shared spike boundary adapter, which extracts trains by unit-id index (a
    # TsGroup is a UserDict; iterating it would yield KEYS, not trains) and
    # surfaces the group's ids. For a genuine group the ids are never ``None``,
    # so the cast to the non-optional group return arm is safe.
    if hasattr(obj, "index"):
        from neurospatial.encoding._spikes import as_spike_trains_with_ids

        return cast(
            "tuple[list[NDArray[np.float64]], NDArray[Any]]",
            as_spike_trains_with_ids(obj),
        )

    raise TypeError(
        f"from_pynapple does not recognize {type(obj).__name__!r}. Expected a "
        "pynapple Tsd, TsdFrame, IntervalSet, or TsGroup."
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
