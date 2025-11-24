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

from neurospatial.nwb._adapters import events_table_to_dataframe

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
    >>> from pynwb import NWBHDF5IO  # doctest: +SKIP
    >>> with NWBHDF5IO("session.nwb", "r") as io:  # doctest: +SKIP
    ...     nwbfile = io.read()
    ...     laps = read_events(nwbfile, "laps")
    ...     print(f"Found {len(laps)} lap events")
    """
    from neurospatial.nwb._core import _require_ndx_events, logger

    # Verify ndx-events is installed BEFORE importing EventsTable
    # This ensures users see the friendly error message
    ndx_events_module = _require_ndx_events()

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
    if not isinstance(events_table, ndx_events_module.EventsTable):
        raise TypeError(
            f"'{table_name}' is not an EventsTable (got {type(events_table).__name__}). "
            "Use read_events() only for EventsTable containers."
        )

    logger.debug(
        "Reading EventsTable '%s' from processing/%s", table_name, processing_module
    )

    # Convert to validated DataFrame via adapter
    return events_table_to_dataframe(events_table, table_name=table_name)


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
    >>> from pynwb import NWBHDF5IO  # doctest: +SKIP
    >>> with NWBHDF5IO("session.nwb", "r") as io:  # doctest: +SKIP
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
    *,
    name: str = "laps",
    overwrite: bool = False,
) -> None:
    """
    Write lap detection results to NWB EventsTable.

    Creates an EventsTable in the processing/behavior/ module containing
    lap event timestamps and optional direction information.

    Parameters
    ----------
    nwbfile : NWBFile
        The NWB file to write to.
    lap_times : NDArray[np.float64], shape (n_laps,)
        Lap start timestamps in seconds. Must be 1D array.
    lap_types : NDArray[np.int_], optional
        Lap types/directions. If provided, added as 'direction' column.
        Must have same length as lap_times.
    description : str, default "Detected lap events"
        Description for the EventsTable.
    name : str, default "laps"
        Name for the EventsTable in NWB.
    overwrite : bool, default False
        If True, replace existing EventsTable with same name.
        If False, raise ValueError on duplicate name.

    Raises
    ------
    ValueError
        If EventsTable with same name exists and overwrite=False.
        If lap_times is not 1D.
        If lap_types length doesn't match lap_times.
    ImportError
        If ndx-events is not installed.

    Notes
    -----
    The EventsTable is stored in the processing/behavior/ module, following
    NWB conventions for behavioral data. The direction column uses integer
    encoding (e.g., 0=outbound, 1=inbound) to allow flexible interpretation.

    Examples
    --------
    >>> from pynwb import NWBHDF5IO  # doctest: +SKIP
    >>> import numpy as np  # doctest: +SKIP
    >>> lap_times = np.array([1.0, 5.5, 10.2, 15.8])  # doctest: +SKIP
    >>> directions = np.array([0, 1, 0, 1])  # doctest: +SKIP
    >>> with NWBHDF5IO("session.nwb", "r+") as io:  # doctest: +SKIP
    ...     nwbfile = io.read()
    ...     write_laps(nwbfile, lap_times, lap_types=directions)
    ...     io.write(nwbfile)
    """
    from neurospatial.nwb._core import (
        _get_or_create_processing_module,
        _require_ndx_events,
        logger,
    )

    # Verify ndx-events is installed BEFORE importing EventsTable
    # This ensures users see the friendly error message
    ndx_events_module = _require_ndx_events()

    # Validate lap_times is 1D
    lap_times = np.asarray(lap_times, dtype=np.float64)
    if lap_times.ndim != 1:
        raise ValueError(
            f"lap_times must be 1D array, got shape {lap_times.shape}. "
            "Each element should be a single timestamp."
        )

    # Validate timestamps are finite and non-negative
    if len(lap_times) > 0:
        if not np.all(np.isfinite(lap_times)):
            raise ValueError("lap_times contains non-finite values (NaN or Inf)")
        if np.any(lap_times < 0):
            raise ValueError("lap_times contains negative timestamps")

    # Validate lap_types length if provided
    if lap_types is not None:
        lap_types = np.asarray(lap_types)
        if len(lap_types) != len(lap_times):
            raise ValueError(
                f"lap_types length ({len(lap_types)}) must match "
                f"lap_times length ({len(lap_times)})."
            )

    # Get or create behavior processing module
    behavior = _get_or_create_processing_module(
        nwbfile, "behavior", "Behavioral data including laps and events"
    )

    # Check for existing table with same name
    if name in behavior.data_interfaces:
        if not overwrite:
            raise ValueError(
                f"EventsTable '{name}' already exists. Use overwrite=True to replace."
            )
        # Remove existing table for replacement (in-memory only)
        del behavior.data_interfaces[name]
        logger.info("Overwriting existing EventsTable '%s'", name)

    # Create EventsTable
    events_table = ndx_events_module.EventsTable(name=name, description=description)

    # Add direction column if lap_types provided
    if lap_types is not None:
        events_table.add_column(name="direction", description="Lap direction/type")

    # Add rows for each lap
    for i, timestamp in enumerate(lap_times):
        if lap_types is not None:
            events_table.add_row(
                timestamp=float(timestamp), direction=int(lap_types[i])
            )
        else:
            events_table.add_row(timestamp=float(timestamp))

    # Add to behavior module
    behavior.add(events_table)
    logger.debug("Wrote EventsTable '%s' with %d lap events", name, len(lap_times))


def write_region_crossings(
    nwbfile: NWBFile,
    crossing_times: NDArray[np.float64],
    region_names: NDArray[np.str_],
    event_types: NDArray[np.str_],
    description: str = "Region crossing events",
    *,
    name: str = "region_crossings",
    overwrite: bool = False,
) -> None:
    """
    Write region crossing events to NWB EventsTable.

    Creates an EventsTable in the processing/behavior/ module containing
    region crossing event timestamps with region name and event type columns.

    Parameters
    ----------
    nwbfile : NWBFile
        The NWB file to write to.
    crossing_times : NDArray[np.float64], shape (n_crossings,)
        Crossing timestamps in seconds. Must be 1D array.
    region_names : NDArray[np.str_], shape (n_crossings,)
        Name of the region for each crossing (e.g., "start", "goal").
        Must have same length as crossing_times.
    event_types : NDArray[np.str_], shape (n_crossings,)
        Type of crossing event. Common values are 'enter' and 'exit', but
        arbitrary strings are allowed (e.g., 'approach', 'dwell', 'near').
        Must have same length as crossing_times.
    description : str, default "Region crossing events"
        Description for the EventsTable.
    name : str, default "region_crossings"
        Name for the EventsTable in NWB.
    overwrite : bool, default False
        If True, replace existing EventsTable with same name.
        If False, raise ValueError on duplicate name.

    Raises
    ------
    ValueError
        If EventsTable with same name exists and overwrite=False.
        If crossing_times is not 1D.
        If region_names or event_types length doesn't match crossing_times.
        If crossing_times contains non-finite or negative values.
    ImportError
        If ndx-events is not installed.

    Notes
    -----
    The EventsTable is stored in the processing/behavior/ module, following
    NWB conventions for behavioral data. Each row represents a single crossing
    event with timestamp, region name, and event type (enter/exit).

    Examples
    --------
    >>> from pynwb import NWBHDF5IO  # doctest: +SKIP
    >>> import numpy as np  # doctest: +SKIP
    >>> crossing_times = np.array([1.0, 2.5, 5.0, 8.2])  # doctest: +SKIP
    >>> region_names = np.array(["start", "goal", "start", "goal"])  # doctest: +SKIP
    >>> event_types = np.array(["enter", "enter", "exit", "exit"])  # doctest: +SKIP
    >>> with NWBHDF5IO("session.nwb", "r+") as io:  # doctest: +SKIP
    ...     nwbfile = io.read()
    ...     write_region_crossings(nwbfile, crossing_times, region_names, event_types)
    ...     io.write(nwbfile)
    """
    from neurospatial.nwb._core import (
        _get_or_create_processing_module,
        _require_ndx_events,
        logger,
    )

    # Verify ndx-events is installed BEFORE importing EventsTable
    # This ensures users see the friendly error message
    ndx_events_module = _require_ndx_events()

    # Validate crossing_times is 1D
    crossing_times = np.asarray(crossing_times, dtype=np.float64)
    if crossing_times.ndim != 1:
        raise ValueError(
            f"crossing_times must be 1D array, got shape {crossing_times.shape}. "
            "Each element should be a single timestamp."
        )

    # Validate timestamps are finite and non-negative
    if len(crossing_times) > 0:
        if not np.all(np.isfinite(crossing_times)):
            raise ValueError("crossing_times contains non-finite values (NaN or Inf)")
        if np.any(crossing_times < 0):
            raise ValueError("crossing_times contains negative timestamps")

    # Convert region_names and event_types to arrays
    region_names = np.asarray(region_names)
    event_types = np.asarray(event_types)

    # Validate lengths match
    if len(region_names) != len(crossing_times):
        raise ValueError(
            f"region_names length ({len(region_names)}) must match "
            f"crossing_times length ({len(crossing_times)})."
        )
    if len(event_types) != len(crossing_times):
        raise ValueError(
            f"event_types length ({len(event_types)}) must match "
            f"crossing_times length ({len(crossing_times)})."
        )

    # Get or create behavior processing module
    behavior = _get_or_create_processing_module(
        nwbfile, "behavior", "Behavioral data including region crossings and events"
    )

    # Check for existing table with same name
    if name in behavior.data_interfaces:
        if not overwrite:
            raise ValueError(
                f"EventsTable '{name}' already exists. Use overwrite=True to replace."
            )
        # Remove existing table for replacement (in-memory only)
        del behavior.data_interfaces[name]
        logger.info("Overwriting existing EventsTable '%s'", name)

    # Create EventsTable
    events_table = ndx_events_module.EventsTable(name=name, description=description)

    # Add region and event_type columns
    events_table.add_column(name="region", description="Region name")
    events_table.add_column(name="event_type", description="Crossing event type")

    # Add rows for each crossing
    for i, timestamp in enumerate(crossing_times):
        events_table.add_row(
            timestamp=float(timestamp),
            region=str(region_names[i]),
            event_type=str(event_types[i]),
        )

    # Add to behavior module
    behavior.add(events_table)
    logger.debug(
        "Wrote EventsTable '%s' with %d region crossing events",
        name,
        len(crossing_times),
    )
