"""
Position and head direction reading from NWB behavior containers.

This module provides functions for reading Position and CompassDirection
data from pynwb.behavior containers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from pynwb import NWBFile


def read_position(
    nwbfile: NWBFile,
    processing_module: str | None = None,
    position_name: str | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Read position data from NWB file.

    Parameters
    ----------
    nwbfile : NWBFile
        The NWB file to read from.
    processing_module : str, optional
        Name of the processing module containing Position data.
        If None, auto-discovers using priority order:
        processing/behavior > processing/* > acquisition.
    position_name : str, optional
        Name of the specific SpatialSeries within Position.
        If None and multiple exist, uses first alphabetically with INFO log.

    Returns
    -------
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Position coordinates.
    timestamps : NDArray[np.float64], shape (n_samples,)
        Timestamps in seconds.

    Raises
    ------
    KeyError
        If no Position container found, or if specified position_name not found.
    ImportError
        If pynwb is not installed.

    Examples
    --------
    >>> from pynwb import NWBHDF5IO
    >>> with NWBHDF5IO("session.nwb", "r") as io:
    ...     nwbfile = io.read()
    ...     positions, timestamps = read_position(nwbfile)
    """
    raise NotImplementedError("read_position not yet implemented")


def read_head_direction(
    nwbfile: NWBFile,
    processing_module: str | None = None,
    compass_name: str | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Read head direction data from NWB file.

    Parameters
    ----------
    nwbfile : NWBFile
        The NWB file to read from.
    processing_module : str, optional
        Name of the processing module containing CompassDirection data.
        If None, auto-discovers using priority order.
    compass_name : str, optional
        Name of the specific SpatialSeries within CompassDirection.
        If None and multiple exist, uses first alphabetically.

    Returns
    -------
    angles : NDArray[np.float64], shape (n_samples,)
        Head direction angles in radians.
    timestamps : NDArray[np.float64], shape (n_samples,)
        Timestamps in seconds.

    Raises
    ------
    KeyError
        If no CompassDirection container found.
    ImportError
        If pynwb is not installed.

    Examples
    --------
    >>> from pynwb import NWBHDF5IO
    >>> with NWBHDF5IO("session.nwb", "r") as io:
    ...     nwbfile = io.read()
    ...     angles, timestamps = read_head_direction(nwbfile)
    """
    raise NotImplementedError("read_head_direction not yet implemented")
