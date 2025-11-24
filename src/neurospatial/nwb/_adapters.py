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
        If neither explicit timestamps nor a sampling rate are available.
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

    n_samples = len(series.data)
    starting_time = float(getattr(series, "starting_time", 0.0) or 0.0)
    rate = float(rate)
    timestamps = np.arange(n_samples, dtype=np.float64) / rate + starting_time
    return np.asarray(timestamps, dtype=np.float64)


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
