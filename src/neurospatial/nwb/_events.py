"""
Events reading and writing for NWB ndx-events containers.

This module provides functions for reading and writing event data
using ndx-events EventsTable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import NDArray

if TYPE_CHECKING:
    from pynwb import NWBFile


def read_events(
    nwbfile: NWBFile,
    table_name: str,
    processing_module: str = "behavior",
) -> pd.DataFrame:
    """
    Read events table from NWB file.

    Events are explicitly named (no auto-discovery) because multiple event
    tables are common and have varying semantic meanings.

    Parameters
    ----------
    nwbfile : NWBFile
        The NWB file to read from.
    table_name : str
        Name of the EventsTable to read.
    processing_module : str, default "behavior"
        Name of the processing module containing the EventsTable.

    Returns
    -------
    DataFrame
        Events data with timestamp column and any additional columns.

    Raises
    ------
    KeyError
        If EventsTable not found in specified location.
    TypeError
        If the specified table is not an EventsTable.
    ImportError
        If ndx-events is not installed.

    Examples
    --------
    >>> from pynwb import NWBHDF5IO
    >>> with NWBHDF5IO("session.nwb", "r") as io:
    ...     nwbfile = io.read()
    ...     laps = read_events(nwbfile, "laps")
    ...     print(f"Found {len(laps)} lap events")
    """
    from ndx_events import EventsTable as EventsTableType

    from neurospatial.nwb._core import _require_ndx_events, logger

    # Verify ndx-events is installed (for type validation)
    _require_ndx_events()

    # Check if processing module exists
    if processing_module not in nwbfile.processing:
        raise KeyError(
            f"Processing module '{processing_module}' not found in NWB file. "
            f"Available modules: {list(nwbfile.processing.keys())}"
        )

    module = nwbfile.processing[processing_module]

    # Check if table exists in module
    if table_name not in module.data_interfaces:
        available = list(module.data_interfaces.keys())
        raise KeyError(
            f"EventsTable '{table_name}' not found in processing/{processing_module}. "
            f"Available tables: {available}"
        )

    events_table = module.data_interfaces[table_name]

    # Validate it's an EventsTable
    if not isinstance(events_table, EventsTableType):
        raise TypeError(
            f"'{table_name}' is not an EventsTable (got {type(events_table).__name__}). "
            "Use read_events() only for EventsTable containers."
        )

    logger.debug(
        "Reading EventsTable '%s' from processing/%s", table_name, processing_module
    )

    # Convert to DataFrame
    # EventsTable extends DynamicTable which has to_dataframe() method
    df = events_table.to_dataframe()

    # Ensure timestamp column is present (EventsTable always has it)
    # The to_dataframe() includes the index, but timestamp is a column in EventsTable
    # Reset index to get id as a column if needed, but we primarily need timestamp
    if "timestamp" not in df.columns:
        # In some cases, timestamp might be stored differently
        # EventsTable should always have timestamp as a column
        raise KeyError(
            f"EventsTable '{table_name}' does not have a 'timestamp' column. "
            "This may not be a valid EventsTable."
        )

    return df


def read_intervals(
    nwbfile: NWBFile,
    interval_name: str,
) -> pd.DataFrame:
    """
    Read time intervals from NWB file.

    Reads TimeIntervals tables from the NWB file, including predefined tables
    (trials, epochs, invalid_times) and custom interval tables.

    Parameters
    ----------
    nwbfile : NWBFile
        The NWB file to read from.
    interval_name : str
        Name of the TimeIntervals table to read. Common names include:
        - "trials" : Trial intervals (predefined)
        - "epochs" : Epoch intervals (predefined)
        - "invalid_times" : Invalid time periods (predefined)
        - Custom names for user-defined interval tables

    Returns
    -------
    DataFrame
        Intervals data with start_time, stop_time, and any additional columns.

    Raises
    ------
    KeyError
        If TimeIntervals table not found in NWB file.

    Notes
    -----
    Unlike read_events() which reads point events from ndx-events EventsTable,
    this function reads interval data with start/stop times from the built-in
    NWB TimeIntervals type. No extension dependencies required.

    Examples
    --------
    >>> from pynwb import NWBHDF5IO
    >>> with NWBHDF5IO("session.nwb", "r") as io:
    ...     nwbfile = io.read()
    ...     trials = read_intervals(nwbfile, "trials")
    ...     print(f"Found {len(trials)} trials")
    """
    from neurospatial.nwb._core import _require_pynwb, logger

    # Verify pynwb is installed
    _require_pynwb()

    # Check predefined interval tables first
    if interval_name == "trials" and nwbfile.trials is not None:
        logger.debug("Reading trials table from NWB file")
        return nwbfile.trials.to_dataframe()

    if interval_name == "epochs" and nwbfile.epochs is not None:
        logger.debug("Reading epochs table from NWB file")
        return nwbfile.epochs.to_dataframe()

    if interval_name == "invalid_times" and nwbfile.invalid_times is not None:
        logger.debug("Reading invalid_times table from NWB file")
        return nwbfile.invalid_times.to_dataframe()

    # Check custom intervals (stored in nwbfile.intervals)
    if interval_name in nwbfile.intervals:
        logger.debug("Reading custom intervals '%s' from NWB file", interval_name)
        return nwbfile.intervals[interval_name].to_dataframe()

    # Not found - provide helpful error message
    available = []
    if nwbfile.trials is not None:
        available.append("trials")
    if nwbfile.epochs is not None:
        available.append("epochs")
    if nwbfile.invalid_times is not None:
        available.append("invalid_times")
    available.extend(list(nwbfile.intervals.keys()))

    raise KeyError(
        f"TimeIntervals '{interval_name}' not found in NWB file. "
        f"Available intervals: {available}"
    )


def write_laps(
    nwbfile: NWBFile,
    lap_times: NDArray[np.float64],
    lap_types: NDArray[np.int_] | None = None,
    description: str = "Detected lap events",
) -> None:
    """
    Write lap detection results to NWB EventsTable.

    Parameters
    ----------
    nwbfile : NWBFile
        The NWB file to write to.
    lap_times : NDArray[np.float64], shape (n_laps,)
        Lap start timestamps in seconds.
    lap_types : NDArray[np.int_], optional
        Lap types/directions. If provided, added as 'direction' column.
    description : str, default "Detected lap events"
        Description for the EventsTable.

    Raises
    ------
    ImportError
        If ndx-events is not installed.

    Examples
    --------
    >>> from pynwb import NWBHDF5IO
    >>> with NWBHDF5IO("session.nwb", "r+") as io:
    ...     nwbfile = io.read()
    ...     write_laps(nwbfile, lap_times, lap_types=directions)
    ...     io.write(nwbfile)
    """
    raise NotImplementedError("write_laps not yet implemented")


def write_region_crossings(
    nwbfile: NWBFile,
    crossing_times: NDArray[np.float64],
    region_names: NDArray[np.str_],
    event_types: NDArray[np.str_],
    description: str = "Region crossing events",
) -> None:
    """
    Write region crossing events to NWB EventsTable.

    Parameters
    ----------
    nwbfile : NWBFile
        The NWB file to write to.
    crossing_times : NDArray[np.float64], shape (n_crossings,)
        Crossing timestamps in seconds.
    region_names : NDArray[np.str_], shape (n_crossings,)
        Name of the region for each crossing.
    event_types : NDArray[np.str_], shape (n_crossings,)
        Type of crossing event ('enter' or 'exit').
    description : str, default "Region crossing events"
        Description for the EventsTable.

    Raises
    ------
    ImportError
        If ndx-events is not installed.
    """
    raise NotImplementedError("write_region_crossings not yet implemented")
