"""
Position and head direction reading from NWB behavior containers.

This module provides functions for reading Position and CompassDirection
data from pynwb.behavior containers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from neurospatial.nwb._core import _find_containers_by_type, _require_pynwb, logger

if TYPE_CHECKING:
    from pynwb import NWBFile
    from pynwb.behavior import Position, SpatialSeries


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
    _require_pynwb()
    from pynwb.behavior import Position as PositionType

    # Find Position container
    position_container = _get_position_container(
        nwbfile, PositionType, processing_module
    )

    # Get SpatialSeries from Position container
    spatial_series = _get_spatial_series(
        position_container, position_name, "SpatialSeries"
    )

    # Extract position data and timestamps
    positions = np.asarray(spatial_series.data[:], dtype=np.float64)
    timestamps = _get_timestamps(spatial_series)

    return positions, timestamps


def _get_position_container(
    nwbfile: NWBFile,
    target_type: type,
    processing_module: str | None,
) -> Position:
    """
    Get Position container from NWB file.

    Parameters
    ----------
    nwbfile : NWBFile
        The NWB file to search.
    target_type : type
        The Position type class.
    processing_module : str, optional
        If provided, look only in this specific module.

    Returns
    -------
    Position
        The Position container.

    Raises
    ------
    KeyError
        If no Position found or if specified module doesn't exist.
    """
    if processing_module is not None:
        # Look in specific module
        if processing_module not in nwbfile.processing:
            raise KeyError(
                f"Processing module '{processing_module}' not found in NWB file. "
                f"Available modules: {list(nwbfile.processing.keys())}"
            )
        module = nwbfile.processing[processing_module]
        # Find Position in this module
        for obj_name in module.data_interfaces:
            obj = module.data_interfaces[obj_name]
            if isinstance(obj, target_type):
                logger.debug(
                    "Found Position at processing/%s/%s", processing_module, obj_name
                )
                return obj
        raise KeyError(f"No Position found in processing module '{processing_module}'")

    # Auto-discover using priority search
    found = _find_containers_by_type(nwbfile, target_type)

    if not found:
        searched_locations = ["processing/*", "acquisition"]
        raise KeyError(
            f"No Position data found in NWB file. Searched: {searched_locations}"
        )

    # Return the first one (highest priority due to sort order)
    path, container = found[0]
    if len(found) > 1:
        all_paths = [p for p, _ in found]
        logger.info(
            "Multiple Position containers found: %s. Using '%s'", all_paths, path
        )
    else:
        logger.debug("Found Position at %s", path)

    return container


def _get_spatial_series(
    position_container: Position,
    series_name: str | None,
    series_type_name: str,
) -> SpatialSeries:
    """
    Get a SpatialSeries from a Position container.

    Parameters
    ----------
    position_container : Position
        The Position container to extract from.
    series_name : str or None
        Name of the specific series. If None, auto-selects.
    series_type_name : str
        Name to use in error messages (e.g., "SpatialSeries").

    Returns
    -------
    SpatialSeries
        The spatial series object.

    Raises
    ------
    KeyError
        If specified series not found or container is empty.
    """
    # Get available series names
    available_series = sorted(position_container.spatial_series.keys())

    if not available_series:
        raise KeyError(f"Position container has no {series_type_name}")

    if series_name is not None:
        # Look for specific series
        if series_name not in position_container.spatial_series:
            raise KeyError(
                f"{series_type_name} '{series_name}' not found. "
                f"Available: {available_series}"
            )
        return position_container.spatial_series[series_name]

    # Auto-select: use first alphabetically
    if len(available_series) > 1:
        logger.info(
            "Multiple %s found: %s. Using '%s'",
            series_type_name,
            available_series,
            available_series[0],
        )

    return position_container.spatial_series[available_series[0]]


def _get_timestamps(spatial_series: SpatialSeries) -> NDArray[np.float64]:
    """
    Get timestamps from a SpatialSeries.

    If explicit timestamps are not available, computes them from rate and starting_time.

    Parameters
    ----------
    spatial_series : SpatialSeries
        The spatial series object.

    Returns
    -------
    NDArray[np.float64]
        Timestamps in seconds.
    """
    if spatial_series.timestamps is not None:
        return np.asarray(spatial_series.timestamps[:], dtype=np.float64)

    # Compute from rate
    n_samples = len(spatial_series.data)
    starting_time = float(spatial_series.starting_time or 0.0)
    rate = float(spatial_series.rate)
    timestamps = np.arange(n_samples, dtype=np.float64) / rate + starting_time
    return np.asarray(timestamps, dtype=np.float64)


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
