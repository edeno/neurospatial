"""
Events reading and writing for NWB ndx-events containers.

This module provides functions for reading and writing event data
using ndx-events EventsTable.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from neurospatial.io.nwb._adapters import events_table_to_dataframe

if TYPE_CHECKING:
    import ndx_events
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
    from neurospatial.io.nwb._core import _require_ndx_events, logger

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
    from neurospatial.io.nwb._core import _require_pynwb, logger

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
    start_regions: Sequence[str] | None = None,
    end_regions: Sequence[str] | None = None,
    stop_times: NDArray[np.float64] | None = None,
    name: str = "laps",
    overwrite: bool = False,
) -> None:
    """
    Write lap detection results to NWB EventsTable.

    Creates an EventsTable in the processing/behavior/ module containing
    lap event timestamps and optional direction, region, and stop time information.

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
    start_regions : Sequence[str], optional
        Start region names for each lap. If provided, added as 'start_region'
        column. Must have same length as lap_times.
    end_regions : Sequence[str], optional
        End region names for each lap. If provided, added as 'end_region'
        column. Must have same length as lap_times.
    stop_times : NDArray[np.float64], optional
        Lap end timestamps in seconds. If provided, added as 'stop_time' column.
        Must have same length as lap_times and stop_times >= lap_times.
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
        If lap_types, start_regions, end_regions, or stop_times length
        doesn't match lap_times.
        If stop_times < lap_times for any entry.
        If timestamps contain non-finite or negative values.
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
    >>> start_regions = ["home", "goal", "home", "goal"]  # doctest: +SKIP
    >>> end_regions = ["goal", "home", "goal", "home"]  # doctest: +SKIP
    >>> with NWBHDF5IO("session.nwb", "r+") as io:  # doctest: +SKIP
    ...     nwbfile = io.read()
    ...     write_laps(
    ...         nwbfile,
    ...         lap_times,
    ...         lap_types=directions,
    ...         start_regions=start_regions,
    ...         end_regions=end_regions,
    ...     )
    ...     io.write(nwbfile)
    """
    from neurospatial.io.nwb._core import (
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

    n_laps = len(lap_times)

    # Validate timestamps are finite and non-negative
    if n_laps > 0:
        if not np.all(np.isfinite(lap_times)):
            raise ValueError("lap_times contains non-finite values (NaN or Inf)")
        if np.any(lap_times < 0):
            raise ValueError("lap_times contains negative timestamps")

    # Validate lap_types length if provided
    if lap_types is not None:
        lap_types = np.asarray(lap_types)
        if len(lap_types) != n_laps:
            raise ValueError(
                f"lap_types length ({len(lap_types)}) must match "
                f"lap_times length ({n_laps})."
            )

    # Validate start_regions length if provided
    start_regions_list: list[str] | None = None
    if start_regions is not None:
        start_regions_list = list(start_regions)
        if len(start_regions_list) != n_laps:
            raise ValueError(
                f"start_regions length ({len(start_regions_list)}) must match "
                f"lap_times length ({n_laps})."
            )

    # Validate end_regions length if provided
    end_regions_list: list[str] | None = None
    if end_regions is not None:
        end_regions_list = list(end_regions)
        if len(end_regions_list) != n_laps:
            raise ValueError(
                f"end_regions length ({len(end_regions_list)}) must match "
                f"lap_times length ({n_laps})."
            )

    # Validate stop_times if provided
    stop_times_arr: NDArray[np.float64] | None = None
    if stop_times is not None:
        stop_times_arr = np.asarray(stop_times, dtype=np.float64)
        if len(stop_times_arr) != n_laps:
            raise ValueError(
                f"stop_times length ({len(stop_times_arr)}) must match "
                f"lap_times length ({n_laps})."
            )
        # Validate stop_times are finite and non-negative
        if n_laps > 0:
            if not np.all(np.isfinite(stop_times_arr)):
                raise ValueError("stop_times contains non-finite values (NaN or Inf)")
            if np.any(stop_times_arr < 0):
                raise ValueError("stop_times contains negative timestamps")
            # Check stop_times >= lap_times
            if np.any(stop_times_arr < lap_times):
                bad_idx = np.where(stop_times_arr < lap_times)[0][0]
                raise ValueError(
                    f"stop_time must be >= lap_time (start_time) for all laps. "
                    f"Lap {bad_idx}: lap_time={lap_times[bad_idx]}, "
                    f"stop_time={stop_times_arr[bad_idx]}"
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

    # Add optional columns
    if lap_types is not None:
        events_table.add_column(name="direction", description="Lap direction/type")
    if start_regions_list is not None:
        events_table.add_column(name="start_region", description="Lap start region")
    if end_regions_list is not None:
        events_table.add_column(name="end_region", description="Lap end region")
    if stop_times_arr is not None:
        events_table.add_column(name="stop_time", description="Lap end timestamp")

    # Add rows for each lap
    for i, timestamp in enumerate(lap_times):
        row_kwargs: dict[str, float | int | str] = {"timestamp": float(timestamp)}
        if lap_types is not None:
            row_kwargs["direction"] = int(lap_types[i])
        if start_regions_list is not None:
            row_kwargs["start_region"] = start_regions_list[i]
        if end_regions_list is not None:
            row_kwargs["end_region"] = end_regions_list[i]
        if stop_times_arr is not None:
            row_kwargs["stop_time"] = float(stop_times_arr[i])
        events_table.add_row(**row_kwargs)

    # Add to behavior module
    behavior.add(events_table)
    logger.debug("Wrote EventsTable '%s' with %d lap events", name, n_laps)


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
    from neurospatial.io.nwb._core import (
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


def write_trials(
    nwbfile: NWBFile,
    trials: list | None = None,
    *,
    start_times: NDArray[np.float64] | None = None,
    stop_times: NDArray[np.float64] | None = None,
    start_regions: Sequence[str] | None = None,
    end_regions: Sequence[str] | None = None,
    successes: Sequence[bool] | None = None,
    description: str = "Behavioral trials",
    overwrite: bool = False,
) -> None:
    """
    Write trial data to NWB file.

    Writes trial intervals to the built-in NWB trials table, including
    optional columns for start/end regions and success status.

    Parameters
    ----------
    nwbfile : NWBFile
        The NWB file to write to.
    trials : list[Trial], optional
        List of Trial objects from segment_trials(). If provided, fields
        are extracted automatically. Cannot be used with raw array parameters.
    start_times : NDArray[np.float64], optional
        Trial start timestamps in seconds. Required if trials not provided.
    stop_times : NDArray[np.float64], optional
        Trial stop timestamps in seconds. Required if trials not provided.
    start_regions : Sequence[str], optional
        Start region names for each trial. If provided, must have same length
        as start_times.
    end_regions : Sequence[str], optional
        End region names for each trial (None values stored as "").
        If provided, must have same length as start_times.
    successes : Sequence[bool], optional
        Success status for each trial. If provided, must have same length
        as start_times.
    description : str, default "Behavioral trials"
        Description for the trials table.
    overwrite : bool, default False
        If True, clear existing trials before adding new ones.
        If False, raise ValueError if trials already exist.

    Raises
    ------
    ValueError
        If both trials and raw arrays are provided.
        If neither trials nor required arrays (start_times, stop_times) provided.
        If array lengths don't match.
        If stop_times < start_times for any trial.
        If timestamps contain non-finite or negative values.

    Notes
    -----
    The trials table is stored in NWB's built-in trials location
    (``nwbfile.trials`` / ``/intervals/trials/``).

    Custom columns are added for:
    - ``start_region``: Region where trial started
    - ``end_region``: Region reached (or "" for timeout)
    - ``success``: Whether trial was successful

    Examples
    --------
    >>> from pynwb import NWBHDF5IO  # doctest: +SKIP
    >>> from neurospatial.behavior.segmentation import segment_trials  # doctest: +SKIP
    >>> trials = segment_trials(bins, times, env, ...)  # doctest: +SKIP
    >>> with NWBHDF5IO("session.nwb", "r+") as io:  # doctest: +SKIP
    ...     nwbfile = io.read()
    ...     write_trials(nwbfile, trials)
    ...     io.write(nwbfile)

    See Also
    --------
    read_trials : Read trials from NWB file.
    read_intervals : Read generic TimeIntervals.
    segment_trials : Segment trajectory into trials.
    """
    from neurospatial.io.nwb._core import _require_pynwb, logger

    # Verify pynwb is installed
    _require_pynwb()

    # Validate input combinations
    has_trials = trials is not None
    has_arrays = start_times is not None or stop_times is not None

    if has_trials and has_arrays:
        raise ValueError(
            "Cannot specify both 'trials' and raw arrays (start_times, stop_times). "
            "Use either Trial objects OR raw arrays, not both."
        )

    if not has_trials and not has_arrays:
        raise ValueError(
            "Must provide either 'trials' (list of Trial objects) or "
            "'start_times' and 'stop_times' arrays."
        )

    # Initialize variables used in both branches
    start_regions_list: list[str] | None = None
    end_regions_list: list[str] | None = None
    successes_list: list[bool] | None = None

    # Extract data from Trial objects or validate raw arrays
    if has_trials:
        if not isinstance(trials, (list, tuple)):
            raise TypeError(f"trials must be a list, got {type(trials).__name__}")

        n_trials = len(trials)
        if n_trials > 0:
            start_times_arr = np.array([t.start_time for t in trials], dtype=np.float64)
            stop_times_arr = np.array([t.end_time for t in trials], dtype=np.float64)
            start_regions_list = [t.start_region for t in trials]
            end_regions_list = [
                t.end_region if t.end_region is not None else "" for t in trials
            ]
            successes_list = [t.success for t in trials]
        else:
            start_times_arr = np.array([], dtype=np.float64)
            stop_times_arr = np.array([], dtype=np.float64)
            start_regions_list = []
            end_regions_list = []
            successes_list = []
    else:
        # Validate required arrays
        if start_times is None:
            raise ValueError(
                "start_times is required when not using Trial objects. "
                "Provide start_times array or use 'trials' parameter."
            )
        if stop_times is None:
            raise ValueError(
                "stop_times is required when not using Trial objects. "
                "Provide stop_times array or use 'trials' parameter."
            )

        start_times_arr = np.asarray(start_times, dtype=np.float64)
        stop_times_arr = np.asarray(stop_times, dtype=np.float64)
        n_trials = len(start_times_arr)

        # Validate array lengths
        if len(stop_times_arr) != n_trials:
            raise ValueError(
                f"stop_times length ({len(stop_times_arr)}) must match "
                f"start_times length ({n_trials})."
            )

        # Convert optional arrays if provided
        start_regions_list = list(start_regions) if start_regions is not None else None
        end_regions_list = list(end_regions) if end_regions is not None else None
        successes_list = list(successes) if successes is not None else None

        # Validate optional array lengths
        if start_regions_list is not None and len(start_regions_list) != n_trials:
            raise ValueError(
                f"start_regions length ({len(start_regions_list)}) must match "
                f"start_times length ({n_trials})."
            )
        if end_regions_list is not None and len(end_regions_list) != n_trials:
            raise ValueError(
                f"end_regions length ({len(end_regions_list)}) must match "
                f"start_times length ({n_trials})."
            )
        if successes_list is not None and len(successes_list) != n_trials:
            raise ValueError(
                f"successes length ({len(successes_list)}) must match "
                f"start_times length ({n_trials})."
            )

    # Validate timestamps
    if n_trials > 0:
        # Check for non-finite values
        if not np.all(np.isfinite(start_times_arr)):
            raise ValueError("start_times contains non-finite values (NaN or Inf)")
        if not np.all(np.isfinite(stop_times_arr)):
            raise ValueError("stop_times contains non-finite values (NaN or Inf)")

        # Check for negative values
        if np.any(start_times_arr < 0):
            raise ValueError("start_times contains negative timestamps")
        if np.any(stop_times_arr < 0):
            raise ValueError("stop_times contains negative timestamps")

        # Check stop >= start
        if np.any(stop_times_arr < start_times_arr):
            bad_idx = np.where(stop_times_arr < start_times_arr)[0][0]
            raise ValueError(
                f"stop_time must be >= start_time for all trials. "
                f"Trial {bad_idx}: start_time={start_times_arr[bad_idx]}, "
                f"stop_time={stop_times_arr[bad_idx]}"
            )

    # Handle existing trials table
    if nwbfile.trials is not None:
        if not overwrite:
            raise ValueError(
                "Trials table already exists. Use overwrite=True to replace."
            )
        # NWB doesn't support deleting/resetting trials table directly.
        # For in-memory operations, we replace the internal reference.
        logger.info("Overwriting existing trials table")
        # Replace with a fresh TimeIntervals
        from pynwb.epoch import TimeIntervals

        new_trials = TimeIntervals(
            name="trials",
            description=description,
        )
        # Replace the internal reference
        nwbfile.fields["trials"] = new_trials

    # If no trials to add, skip table creation (NWB requires at least one trial)
    if n_trials == 0:
        logger.debug("No trials to write")
        return

    # Add custom columns (before adding any trials)
    # Check if columns already exist (for overwrite case)
    existing_columns = set()
    if nwbfile.trials is not None:
        existing_columns = set(nwbfile.trials.colnames)

    # For Trial objects, always add all custom columns
    if has_trials and n_trials > 0:
        if "start_region" not in existing_columns:
            nwbfile.add_trial_column(
                name="start_region", description="Region where trial started"
            )
        if "end_region" not in existing_columns:
            nwbfile.add_trial_column(
                name="end_region",
                description="Region reached (empty string if timeout)",
            )
        if "success" not in existing_columns:
            nwbfile.add_trial_column(
                name="success", description="Whether trial was successful"
            )
    elif not has_trials:
        # For raw arrays, only add columns if data provided
        if start_regions_list is not None and "start_region" not in existing_columns:
            nwbfile.add_trial_column(
                name="start_region", description="Region where trial started"
            )
        if end_regions_list is not None and "end_region" not in existing_columns:
            nwbfile.add_trial_column(
                name="end_region",
                description="Region reached (empty string if timeout)",
            )
        if successes_list is not None and "success" not in existing_columns:
            nwbfile.add_trial_column(
                name="success", description="Whether trial was successful"
            )

    # Add trials
    for i in range(n_trials):
        kwargs: dict[str, float | str | bool] = {
            "start_time": float(start_times_arr[i]),
            "stop_time": float(stop_times_arr[i]),
        }

        if has_trials:
            # start_regions_list and end_regions_list are guaranteed non-None here
            assert start_regions_list is not None
            assert end_regions_list is not None
            assert successes_list is not None
            kwargs["start_region"] = start_regions_list[i]
            kwargs["end_region"] = end_regions_list[i]
            kwargs["success"] = successes_list[i]
        else:
            if start_regions_list is not None:
                kwargs["start_region"] = start_regions_list[i]
            if end_regions_list is not None:
                kwargs["end_region"] = end_regions_list[i]
            if successes_list is not None:
                kwargs["success"] = successes_list[i]

        nwbfile.add_trial(**kwargs)

    # Set description for newly created trials table (only works when creating new)
    # NWB sets default description, so only update if custom description provided
    # Note: For overwrite case, description is set during TimeIntervals creation
    # For new trials, NWB creates the table with default description on first add_trial
    # We cannot change it after, so this only works for custom description when
    # we control the table creation (overwrite case)

    logger.debug("Wrote %d trials to NWB file", n_trials)


def read_trials(nwbfile: NWBFile) -> pd.DataFrame:
    """
    Read trials table from NWB file.

    Convenience wrapper around read_intervals("trials") that provides
    a more discoverable API for reading trial data.

    Parameters
    ----------
    nwbfile : NWBFile
        The NWB file to read from.

    Returns
    -------
    pd.DataFrame
        Trials data with columns: start_time, stop_time, and any custom
        columns (start_region, end_region, success, etc.).

    Raises
    ------
    KeyError
        If no trials table exists in NWB file.

    Examples
    --------
    >>> from pynwb import NWBHDF5IO  # doctest: +SKIP
    >>> with NWBHDF5IO("session.nwb", "r") as io:  # doctest: +SKIP
    ...     nwbfile = io.read()
    ...     trials = read_trials(nwbfile)
    ...     print(f"Found {len(trials)} trials")

    See Also
    --------
    write_trials : Write trials to NWB file.
    read_intervals : Read generic TimeIntervals.
    """
    return read_intervals(nwbfile, "trials")


def dataframe_to_events_table(
    df: pd.DataFrame,
    name: str,
    description: str = "Events",
) -> ndx_events.EventsTable:
    """
    Convert a pandas DataFrame to an ndx-events EventsTable.

    This function creates an EventsTable with the same columns as the input
    DataFrame. The DataFrame must have a 'timestamp' column.

    Parameters
    ----------
    df : pd.DataFrame
        Events DataFrame. Must have a 'timestamp' column with numeric values.
        Additional columns become EventsTable columns.
    name : str
        Name for the EventsTable.
    description : str, default "Events"
        Description for the EventsTable.

    Returns
    -------
    ndx_events.EventsTable
        EventsTable containing the events data.

    Raises
    ------
    TypeError
        If df is not a DataFrame.
    ValueError
        If DataFrame is missing timestamp column or contains invalid timestamps.
    ImportError
        If ndx-events is not installed.

    Examples
    --------
    >>> import pandas as pd  # doctest: +SKIP
    >>> df = pd.DataFrame(
    ...     {  # doctest: +SKIP
    ...         "timestamp": [1.0, 2.0, 3.0],
    ...         "label": ["a", "b", "c"],
    ...     }
    ... )
    >>> events_table = dataframe_to_events_table(df, "my_events")  # doctest: +SKIP
    """
    from neurospatial.io.nwb._core import _require_ndx_events

    # Verify ndx-events is installed
    ndx_events_module = _require_ndx_events()

    # Validate input
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"Expected pd.DataFrame, got {type(df).__name__}.\n"
            "  WHY: Events must be a pandas DataFrame.\n"
            "  HOW: Convert using pd.DataFrame({'timestamp': times})"
        )

    if "timestamp" not in df.columns:
        raise ValueError(
            "DataFrame is missing required 'timestamp' column.\n"
            "  WHY: EventsTable requires timestamp for each event.\n"
            f"  HOW: Add timestamp column. Available columns: {list(df.columns)}"
        )

    # Validate timestamps if not empty
    if len(df) > 0:
        timestamps = df["timestamp"].values
        if not np.all(np.isfinite(timestamps)):
            raise ValueError(
                "timestamp column contains non-finite values (NaN or Inf).\n"
                "  WHY: EventsTable requires valid numeric timestamps.\n"
                "  HOW: Remove or replace NaN/Inf values in timestamp column."
            )
        if np.any(timestamps < 0):
            raise ValueError(
                "timestamp column contains negative values.\n"
                "  WHY: NWB timestamps should be non-negative (seconds from session start).\n"
                "  HOW: Adjust timestamps to be relative to session start."
            )

    # Create EventsTable
    events_table = ndx_events_module.EventsTable(name=name, description=description)

    # Add additional columns (excluding timestamp which is built-in)
    extra_columns = [col for col in df.columns if col != "timestamp"]
    for col in extra_columns:
        events_table.add_column(name=col, description=f"Column: {col}")

    # Add rows
    for _, row in df.iterrows():
        row_data = {"timestamp": float(row["timestamp"])}
        for col in extra_columns:
            value = row[col]
            # Convert numpy types to Python types for NWB compatibility
            if isinstance(value, (np.integer, np.floating)):
                value = value.item()
            elif isinstance(value, np.bool_):
                value = bool(value)
            row_data[col] = value
        events_table.add_row(**row_data)

    return events_table


def write_events(
    nwbfile: NWBFile,
    events: pd.DataFrame,
    name: str,
    *,
    description: str = "Event data",
    processing_module: str = "behavior",
    overwrite: bool = False,
) -> None:
    """
    Write generic events DataFrame to NWB EventsTable.

    Creates an EventsTable in a processing module containing event timestamps
    and any additional columns from the DataFrame.

    Parameters
    ----------
    nwbfile : NWBFile
        The NWB file to write to.
    events : pd.DataFrame
        Events DataFrame. Must have a 'timestamp' column with numeric values.
        Additional columns (e.g., 'label', 'value', 'x', 'y') are preserved.
    name : str
        Name for the EventsTable in NWB.
    description : str, default "Event data"
        Description for the EventsTable.
    processing_module : str, default "behavior"
        Processing module to store events in.
    overwrite : bool, default False
        If True, replace existing EventsTable with same name.
        If False, raise ValueError on duplicate name.

    Raises
    ------
    TypeError
        If events is not a DataFrame.
    ValueError
        If EventsTable with same name exists and overwrite=False.
        If DataFrame is missing timestamp column.
        If timestamps contain non-finite or negative values.
    ImportError
        If ndx-events is not installed.

    Notes
    -----
    The EventsTable is stored in the specified processing module. Common
    column conventions:

    - ``timestamp``: Required. Event time in seconds from session start.
    - ``label``: String category/type for event.
    - ``value``: Numeric value associated with event.
    - ``x``, ``y``, ``z``: Spatial position columns.

    All additional columns are preserved as EventsTable columns.

    Examples
    --------
    >>> from pynwb import NWBHDF5IO  # doctest: +SKIP
    >>> import pandas as pd  # doctest: +SKIP
    >>> events = pd.DataFrame(
    ...     {  # doctest: +SKIP
    ...         "timestamp": [1.0, 2.5, 5.0],
    ...         "label": ["reward", "lick", "reward"],
    ...         "x": [10.0, 15.0, 20.0],
    ...         "y": [5.0, 8.0, 12.0],
    ...     }
    ... )
    >>> with NWBHDF5IO("session.nwb", "r+") as io:  # doctest: +SKIP
    ...     nwbfile = io.read()
    ...     write_events(nwbfile, events, name="behavioral_events")
    ...     io.write(nwbfile)

    See Also
    --------
    read_events : Read events from NWB EventsTable.
    write_laps : Write lap events (specialized format).
    write_region_crossings : Write region crossing events (specialized format).
    """
    from neurospatial.io.nwb._core import (
        _get_or_create_processing_module,
        _require_ndx_events,
        logger,
    )

    # Verify ndx-events is installed
    _require_ndx_events()

    # Get or create processing module
    module = _get_or_create_processing_module(
        nwbfile, processing_module, f"{processing_module.capitalize()} data"
    )

    # Check for existing table with same name
    if name in module.data_interfaces:
        if not overwrite:
            raise ValueError(
                f"EventsTable '{name}' already exists in processing/{processing_module}. "
                "Use overwrite=True to replace."
            )
        # Remove existing table for replacement (in-memory only)
        del module.data_interfaces[name]
        logger.info("Overwriting existing EventsTable '%s'", name)

    # Convert DataFrame to EventsTable
    events_table = dataframe_to_events_table(events, name=name, description=description)

    # Add to processing module
    module.add(events_table)
    logger.debug(
        "Wrote EventsTable '%s' with %d events to processing/%s",
        name,
        len(events),
        processing_module,
    )
