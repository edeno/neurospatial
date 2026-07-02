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

    # Tsd / TsdFrame: a value time series -> (times, positions). Delegate to the
    # shared position boundary adapter -- identical duck-type guard and float64
    # coercion (including the ``.d`` alias fallback), so this stays a single
    # implementation of that coercion.
    if hasattr(obj, "t") and (hasattr(obj, "values") or hasattr(obj, "d")):
        from neurospatial._typing import as_times_positions

        return as_times_positions(obj)

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
        from neurospatial.encoding import as_spike_trains_with_ids

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
        Timestamps (seconds), or a decode result exposing ``.times`` and
        ``.map_position`` (in which case ``values`` must be ``None``).
    values : NDArray[np.float64] or None, default=None
        Values sampled at ``times``, shape ``(n,)`` or ``(n, n_dims)``. Required
        when ``times`` is a timestamp array; must be ``None`` when ``times`` is a
        decode result.
    columns : sequence, optional
        Column labels for the resulting ``TsdFrame`` (2-D values only).

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

    Examples
    --------
    >>> from neurospatial.io import to_pynapple  # doctest: +SKIP
    >>> tsdframe = to_pynapple(result)  # a DecodingResult MAP track  # doctest: +SKIP
    >>> tsd = to_pynapple(times, linear_positions)  # doctest: +SKIP
    """
    nap = _require_pynapple()

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

    if values.ndim == 1:
        return nap.Tsd(t=times, d=values)
    return nap.TsdFrame(t=times, d=values, columns=columns)
