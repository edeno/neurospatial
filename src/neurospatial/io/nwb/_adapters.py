"""
Internal adapter utilities for NWB/ndx containers.

These helpers centralize our assumptions about pynwb and ndx-* types so that
the rest of the neurospatial.io.nwb package can work with simple, protocol-like
interfaces instead of concrete classes. If upstream APIs change, the goal is
that only this module needs to be updated.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def missing_error(what: str, *, fix: str) -> KeyError:
    """A ``KeyError`` for a missing named object, printed as written."""
    from neurospatial._exceptions import _format_error
    from neurospatial.io.nwb._core import _MissingObjectError

    return _MissingObjectError(_format_error(what, fix=fix))


def select_one(
    candidates: list[tuple[str, Any]],
    name: str | None,
    *,
    what: str,
    keyword: str,
) -> Any:
    """Return the one candidate named ``name``, or the only candidate.

    Parameters
    ----------
    candidates : list of (str, Any)
        ``(path, object)`` pairs, e.g.
        ``("processing/behavior/Position/position", series)``.
    name : str or None
        An object's name (the last path component) or its full path.
    what : str
        Description of the objects for messages, e.g. ``"Position series"``.
    keyword : str
        The calling reader's argument that takes ``name``.

    Raises
    ------
    KeyError
        If no candidate matches ``name``.
    ValueError
        If more than one candidate matches; never picks one silently.
    """
    from neurospatial._exceptions import _format_error

    paths = [path for path, _ in candidates]
    if name is not None:
        candidates = [
            (path, obj)
            for path, obj in candidates
            if name in (path, path.rsplit("/", 1)[-1])
        ]
        if not candidates:
            raise missing_error(
                f"{what} '{name}' not found. Available: {paths}",
                fix=f"pass {keyword}= with a name or path from this list.",
            )
    if len(candidates) > 1:
        matching = [path for path, _ in candidates]
        raise ValueError(
            _format_error(
                f"Found {len(matching)} {what} candidates: {matching}.",
                why="they may differ in units, frame or processing, so none is "
                "chosen for you.",
                fix=f"pass {keyword}= with one of these paths, e.g. "
                f"{keyword}={matching[0]!r}.",
            )
        )
    return candidates[0][1]


def supply_coordinate_metadata(
    declared: dict[str, str | None],
    requested: dict[str, str | None],
    *,
    call: str,
    source: str,
    fix: str,
) -> dict[str, str | None]:
    """Fill absent ``units`` / ``frame`` declarations from the caller's values.

    A requested value may restate a declaration (unit aliases are the same
    unit) or supply a missing one; it never erases or relabels a known
    declaration, because metadata does not convert coordinates.

    Parameters
    ----------
    declared : dict
        The source's ``"units"`` and ``"frame"``; None when undeclared.
    requested : dict
        The caller's values for some of those keys. A key that is present is
        checked, even with value None.
    call : str
        The public call, named in the error.
    source : str
        The declaring object, e.g. ``"the Position series"``.
    fix : str
        How to convert the coordinates instead, completing "omit ``key=``
        to use the declaration, or ...".

    Returns
    -------
    dict
        ``declared`` with absent entries filled from ``requested``.

    Raises
    ------
    ValueError
        If a requested value conflicts with a known declaration.
    """
    from neurospatial._exceptions import _format_error
    from neurospatial._units import same_coordinate_declaration

    resolved = dict(declared)
    for key, value in requested.items():
        known = declared[key]
        if known is None:
            resolved[key] = value
        elif not same_coordinate_declaration(
            key, value, known, undeclared_matches=False
        ):
            raise ValueError(
                _format_error(
                    f"{call}({key}={value!r}) conflicts with {source}, which "
                    f"declares {key} {known!r}.",
                    why=f"Why: a {key} label does not convert coordinates, so "
                    "a different one would mislabel them.",
                    fix=f"omit {key}= to use {known!r}, or {fix}",
                )
            )
    return resolved


def timestamps_from_series(series: Any) -> NDArray[np.float64]:
    """
    Extract timestamps from a time series-like object.

    The materialized counterpart to :func:`timestamps_handle_from_series`,
    used for both pynwb.behavior.SpatialSeries and
    ndx_pose.PoseEstimationSeries.

    Parameters
    ----------
    series : Any
        Object with at least ``data`` and either ``timestamps`` or ``rate``
        (and optionally ``starting_time``) attributes.

    Returns
    -------
    NDArray[np.float64]
        1D array of timestamps in seconds.

    Raises
    ------
    ValueError
        If neither explicit timestamps nor a valid sampling ``rate`` are
        available (see :func:`timestamps_handle_from_series`).
    """
    return np.asarray(timestamps_handle_from_series(series)[:], dtype=np.float64)


def scaling_from_series(series: Any) -> tuple[float, float]:
    """
    Return the ``(conversion, offset)`` mapping stored values to ``series.unit``.

    NWB defines the value in ``unit`` as ``data * conversion + offset``.

    Parameters
    ----------
    series : Any
        A ``TimeSeries``-like object, optionally with ``conversion`` and
        ``offset`` attributes.

    Returns
    -------
    tuple of float
        ``(conversion, offset)``; ``(1.0, 0.0)`` when the attributes are absent.
    """
    return float(getattr(series, "conversion", 1.0)), float(
        getattr(series, "offset", 0.0)
    )


def data_from_series(series: Any) -> NDArray[np.float64]:
    """
    Materialize ``series.data`` in its declared unit.

    Parameters
    ----------
    series : Any
        A ``TimeSeries``-like object with ``data``.

    Returns
    -------
    NDArray[np.float64]
        ``data * conversion + offset`` as float64.
    """
    data = np.asarray(series.data[:], dtype=np.float64)
    conversion, offset = scaling_from_series(series)
    return data * conversion + offset if (conversion, offset) != (1.0, 0.0) else data


def require_unscaled_for_lazy(series: Any, *, context: str) -> None:
    """
    Refuse a lazy handle whose stored values are not in ``series.unit``.

    Parameters
    ----------
    series : Any
        A ``TimeSeries``-like object.
    context : str
        Name of the calling reader, used in the error message.

    Raises
    ------
    ValueError
        If the series' ``conversion`` / ``offset`` are not the identity.
    """
    conversion, offset = scaling_from_series(series)
    if (conversion, offset) != (1.0, 0.0):
        raise ValueError(
            f"{context}(lazy=True) would return the stored values of '{series.name}', "
            f"but the series declares conversion={conversion} and offset={offset}, so "
            f"stored values are not in '{series.unit}'.\n"
            f"Fix: call {context}(..., lazy=False); values are converted on read."
        )


def timestamps_handle_from_series(series: Any) -> Any:
    """
    Return a lazy timestamps handle for a time series-like object.

    It follows the standard NWB pattern:

    1. If an explicit ``timestamps`` array is present, that backing store is
       returned **without** copying (an ``h5py.Dataset`` for a file-backed
       series), so it materializes only when sliced or ``np.asarray``-ed.
    2. Otherwise the time axis is computed from ``rate`` (and optionally
       ``starting_time``) and returned as a materialized array; there is no
       on-disk array to reference lazily.

    Parameters
    ----------
    series : Any
        Object with at least ``data`` and either ``timestamps`` or ``rate``
        (and optionally ``starting_time``) attributes.

    Returns
    -------
    Any
        The h5py-backed (or in-memory) ``timestamps`` handle when explicit
        timestamps are present, otherwise a materialized ``NDArray[np.float64]``
        of rate-derived timestamps.

    Raises
    ------
    ValueError
        If neither explicit timestamps nor a sampling rate are available, or
        if ``rate`` is present but not a finite positive value (a non-positive
        or non-finite rate would yield Inf/NaN timestamps).
    """
    timestamps = getattr(series, "timestamps", None)
    if timestamps is not None:
        return timestamps

    rate = getattr(series, "rate", None)
    if rate is None:
        raise ValueError(
            "Series has neither 'timestamps' nor 'rate' attribute; cannot derive "
            "time axis."
        )

    rate = float(rate)
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError(
            f"Series 'rate' must be a finite positive value to derive a time "
            f"axis, got {rate!r}. Provide an explicit 'timestamps' array instead."
        )

    starting_time = float(getattr(series, "starting_time", 0.0) or 0.0)
    return np.arange(len(series.data), dtype=np.float64) / rate + starting_time


def events_table_to_dataframe(events_table: Any, *, table_name: str) -> pd.DataFrame:
    """
    Convert an EventsTable-like object to a DataFrame and validate it.

    Parameters
    ----------
    events_table : Any
        Object expected to behave like ndx_events.EventsTable, i.e. exposing
        a ``to_dataframe()`` method that returns a pandas DataFrame.
    table_name : str
        Logical name for error messages.

    Returns
    -------
    DataFrame
        Events data with a required ``timestamp`` column.

    Raises
    ------
    KeyError
        If the resulting DataFrame does not contain a ``timestamp`` column.
    """
    df = events_table.to_dataframe()

    if "timestamp" not in df.columns:
        raise missing_error(
            f"EventsTable '{table_name}' does not have a 'timestamp' column, "
            "so it may not be a valid EventsTable.",
            fix="read an ndx-events EventsTable, which stores event times in "
            "its 'timestamp' column.",
        )

    return df
