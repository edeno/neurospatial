"""
Internal adapter utilities for NWB/ndx containers.

These helpers centralize our assumptions about pynwb and ndx-* types so that
the rest of the neurospatial.nwb package can work with simple, protocol-like
interfaces instead of concrete classes. If upstream APIs change, the goal is
that only this module needs to be updated.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# =============================================================================
# Constants for NWB adapters
# =============================================================================

# Default starting time for time series when not specified
DEFAULT_STARTING_TIME: float = 0.0


def timestamps_from_series(series: Any) -> NDArray[np.float64]:
    """
    Extract timestamps from a time series-like object.

    This helper is used for both pynwb.behavior.SpatialSeries and
    ndx_pose.PoseEstimationSeries. It follows the standard NWB pattern:

    1. If an explicit ``timestamps`` array is present, use it.
    2. Otherwise, if ``rate`` (and optionally ``starting_time``) are present,
       compute timestamps from these values.

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
        If neither explicit timestamps nor a sampling rate are available, or
        if ``rate`` is present but not a finite positive value (a non-positive
        or non-finite rate would yield Inf/NaN timestamps).
    """
    timestamps_attr = getattr(series, "timestamps", None)
    if timestamps_attr is not None:
        return np.asarray(timestamps_attr[:], dtype=np.float64)

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

    n_samples = len(series.data)
    starting_time = float(
        getattr(series, "starting_time", DEFAULT_STARTING_TIME) or DEFAULT_STARTING_TIME
    )
    timestamps = np.arange(n_samples, dtype=np.float64) / rate + starting_time
    return np.asarray(timestamps, dtype=np.float64)


def timestamps_handle_from_series(series: Any) -> Any:
    """
    Return a lazy timestamps handle for a time series-like object.

    The lazy counterpart to :func:`timestamps_from_series`. When the series
    carries an explicit ``timestamps`` array, that backing store is returned
    **without** copying (an ``h5py.Dataset`` for a file-backed series), so it
    materializes only when sliced or ``np.asarray``-ed. When the series is
    rate-based (no stored ``timestamps``), the time axis must be computed, so a
    fully-materialized array is returned instead -- there is no on-disk array to
    reference lazily.

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
        If neither explicit timestamps nor a valid sampling ``rate`` are
        available (same contract as :func:`timestamps_from_series`).
    """
    timestamps_attr = getattr(series, "timestamps", None)
    if timestamps_attr is not None:
        return timestamps_attr
    return timestamps_from_series(series)


def validate_handle_lengths(name_to_length: dict[str, int]) -> None:
    """Raise ``ValueError`` if named lazy handles do not share a length.

    The lazy counterpart to :func:`neurospatial._validation.validate_lengths`.
    It takes **precomputed integer lengths** (read from a handle's ``.shape[0]``,
    which an ``h5py.Dataset`` exposes WITHOUT materializing its values) instead
    of arrays, so the length check on a lazy read stays lazy. The raised message
    is byte-for-byte identical to the eager ``validate_lengths`` message, so a
    lazy read raises exactly as the eager path does on a mismatch.

    Parameters
    ----------
    name_to_length : dict of str to int
        Mapping from argument name to its length (``handle.shape[0]``).

    Returns
    -------
    None
        Returns nothing when all lengths agree.

    Raises
    ------
    ValueError
        If the lengths do not all agree. The message lists each name and its
        length.
    """
    if len(set(name_to_length.values())) > 1:
        pairs = ", ".join(f"{k}={n}" for k, n in name_to_length.items())
        raise ValueError(f"Length mismatch: {pairs}. These must agree.")


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
        raise KeyError(
            f"EventsTable '{table_name}' does not have a 'timestamp' column. "
            "This may not be a valid EventsTable."
        )

    return df
