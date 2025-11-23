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
    raise NotImplementedError("read_events not yet implemented")


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
