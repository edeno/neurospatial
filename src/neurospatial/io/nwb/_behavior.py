"""
Position and head direction reading from NWB behavior containers.

This module provides functions for reading Position and CompassDirection
data from pynwb.behavior containers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from neurospatial.io.nwb._adapters import (
    timestamps_from_series,
    timestamps_handle_from_series,
)
from neurospatial.io.nwb._core import _find_containers_by_type, _require_pynwb, logger

if TYPE_CHECKING:
    from pynwb import NWBFile
    from pynwb.behavior import CompassDirection, Position, SpatialSeries

# =============================================================================
# Constants for behavior data
# =============================================================================

# Fallback units for position data when not specified
DEFAULT_POSITION_UNITS: str = "cm"


def read_position(
    nwbfile: NWBFile,
    processing_module: str | None = None,
    position_name: str | None = None,
    *,
    lazy: bool = False,
) -> tuple[NDArray[np.float64] | Any, NDArray[np.float64] | Any]:
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
    lazy : bool, default False
        If ``False`` (default), positions and timestamps are fully materialized
        into ``NDArray[np.float64]`` -- the historical, byte-for-byte-unchanged
        behavior. If ``True``, the h5py-backed ``SpatialSeries.data`` handle (and
        the timestamps handle) are returned **without** copying, so they
        materialize only when sliced or ``np.asarray``-ed. Use this to keep large
        recordings off-RAM.

    Returns
    -------
    positions : NDArray[np.float64] or h5py.Dataset, shape (n_samples, n_dims)
        Position coordinates. A materialized array when ``lazy=False``; a lazy
        handle when ``lazy=True``.
    timestamps : NDArray[np.float64] or h5py.Dataset, shape (n_samples,)
        Timestamps in seconds. A materialized array when ``lazy=False``; a lazy
        handle when ``lazy=True`` and the series carries explicit timestamps (a
        rate-based series still returns a computed array).

    Raises
    ------
    KeyError
        If no Position container found, or if specified position_name not found.
    ImportError
        If pynwb is not installed.

    Notes
    -----
    Lazy handles are **only valid while the backing ``NWBFile`` / ``NWBHDF5IO``
    is open**. Slice or ``np.asarray`` them inside the ``with NWBHDF5IO(...)``
    block; accessing them after the file is closed raises an error from the HDF5
    layer. ``lazy=False`` returns fully-materialized arrays as before and is safe
    to use after the file closes. Length validation (positions vs. timestamps) is
    performed only on the eager path; the lazy path skips it to avoid forcing a
    full read.

    Examples
    --------
    >>> from pynwb import NWBHDF5IO  # doctest: +SKIP
    >>> with NWBHDF5IO("session.nwb", "r") as io:  # doctest: +SKIP
    ...     nwbfile = io.read()
    ...     positions, timestamps = read_position(nwbfile)

    Lazy read (materialize a slice inside the open-file block):

    >>> with NWBHDF5IO("session.nwb", "r") as io:  # doctest: +SKIP
    ...     nwbfile = io.read()
    ...     positions, timestamps = read_position(nwbfile, lazy=True)
    ...     first_100 = positions[:100]  # only this slice is read from disk
    """
    _require_pynwb()
    from pynwb.behavior import Position as PositionType

    # Find Position container
    position_container = _get_behavior_container(
        nwbfile, PositionType, "Position", processing_module
    )

    # Get SpatialSeries from Position container
    spatial_series = _get_spatial_series(
        position_container, position_name, "SpatialSeries", "Position"
    )

    if lazy:
        # Return the h5py-backed handles without copying; they materialize on
        # slice / np.asarray. Skip validate_lengths (it would force a full read).
        return spatial_series.data, timestamps_handle_from_series(spatial_series)

    # Extract position data and timestamps
    positions = np.asarray(spatial_series.data[:], dtype=np.float64)
    timestamps = _get_timestamps(spatial_series)

    from neurospatial._validation import validate_lengths

    validate_lengths({"positions": positions, "timestamps": timestamps})
    return positions, timestamps


def _get_behavior_container(
    nwbfile: NWBFile,
    target_type: type,
    type_name: str,
    processing_module: str | None,
) -> Position | CompassDirection:
    """
    Get a behavior container (Position or CompassDirection) from NWB file.

    Parameters
    ----------
    nwbfile : NWBFile
        The NWB file to search.
    target_type : type
        The container type class (Position or CompassDirection).
    type_name : str
        Human-readable name for error messages (e.g., "Position", "CompassDirection").
    processing_module : str, optional
        If provided, look only in this specific module.

    Returns
    -------
    Position | CompassDirection
        The behavior container.

    Raises
    ------
    KeyError
        If no container found or if specified module doesn't exist.
    """
    if processing_module is not None:
        # Look in specific module
        if processing_module not in nwbfile.processing:
            raise KeyError(
                f"Processing module '{processing_module}' not found in NWB file. "
                f"Available modules: {list(nwbfile.processing.keys())}"
            )
        module = nwbfile.processing[processing_module]
        # Find container in this module
        for obj_name in module.data_interfaces:
            obj = module.data_interfaces[obj_name]
            if isinstance(obj, target_type):
                logger.debug(
                    "Found %s at processing/%s/%s",
                    type_name,
                    processing_module,
                    obj_name,
                )
                return obj
        raise KeyError(
            f"No {type_name} found in processing module '{processing_module}'"
        )

    # Auto-discover using priority search
    found = _find_containers_by_type(nwbfile, target_type)

    if not found:
        searched_locations = ["processing/*", "acquisition"]
        raise KeyError(
            f"No {type_name} data found in NWB file. Searched: {searched_locations}"
        )

    # Return the first one (highest priority due to sort order)
    path, container = found[0]
    if len(found) > 1:
        all_paths = [p for p, _ in found]
        logger.info(
            "Multiple %s containers found: %s. Using '%s'", type_name, all_paths, path
        )
    else:
        logger.debug("Found %s at %s", type_name, path)

    return container


def _get_spatial_series(
    container: Position | CompassDirection,
    series_name: str | None,
    series_type_name: str,
    container_type_name: str = "Position",
) -> SpatialSeries:
    """
    Get a SpatialSeries from a behavior container.

    Parameters
    ----------
    container : Position | CompassDirection
        The behavior container to extract from.
    series_name : str or None
        Name of the specific series. If None, auto-selects.
    series_type_name : str
        Name to use in error messages (e.g., "SpatialSeries").
    container_type_name : str, default "Position"
        Name of the container type for error messages.

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
    available_series = sorted(container.spatial_series.keys())

    if not available_series:
        raise KeyError(f"{container_type_name} container has no {series_type_name}")

    if series_name is not None:
        # Look for specific series
        if series_name not in container.spatial_series:
            raise KeyError(
                f"{series_type_name} '{series_name}' not found. "
                f"Available: {available_series}"
            )
        return container.spatial_series[series_name]

    # Auto-select: use first alphabetically
    if len(available_series) > 1:
        logger.info(
            "Multiple %s found: %s. Using '%s'",
            series_type_name,
            available_series,
            available_series[0],
        )

    return container.spatial_series[available_series[0]]


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
    return timestamps_from_series(spatial_series)


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
        If None, auto-discovers using priority order:
        processing/behavior > processing/* > acquisition.
    compass_name : str, optional
        Name of the specific SpatialSeries within CompassDirection.
        If None and multiple exist, uses first alphabetically with INFO log.

    Returns
    -------
    angles : NDArray[np.float64], shape (n_samples,)
        Head direction angles in radians, allocentric convention
        (0 = East, increasing counterclockwise; range as stored, typically
        (-pi, pi] from arctan2 or [0, 2*pi) from a wrapped angle series).
        If the source SpatialSeries reports a degree unit, values are
        converted to radians on read.
    timestamps : NDArray[np.float64], shape (n_samples,)
        Timestamps in seconds.

    Raises
    ------
    KeyError
        If no CompassDirection container found, or if specified compass_name
        not found.
    ImportError
        If pynwb is not installed.

    Examples
    --------
    >>> from pynwb import NWBHDF5IO  # doctest: +SKIP
    >>> with NWBHDF5IO("session.nwb", "r") as io:  # doctest: +SKIP
    ...     nwbfile = io.read()
    ...     angles, timestamps = read_head_direction(nwbfile)
    """
    _require_pynwb()
    from pynwb.behavior import CompassDirection as CompassDirectionType

    # Find CompassDirection container
    compass_container = _get_behavior_container(
        nwbfile, CompassDirectionType, "CompassDirection", processing_module
    )

    # Get SpatialSeries from CompassDirection container
    spatial_series = _get_spatial_series(
        compass_container, compass_name, "SpatialSeries", "CompassDirection"
    )

    # Extract angle data and timestamps. CompassDirection SpatialSeries store
    # head direction as either a 1-D angle series or an (n, 2) unit-vector
    # series; handle both. Angles are returned in the allocentric convention
    # (0 = East, increasing counterclockwise).
    raw = np.asarray(spatial_series.data[:], dtype=np.float64)
    timestamps = _get_timestamps(spatial_series)

    if raw.ndim == 2 and raw.shape[1] == 2:
        # (n, 2) heading vectors -> angle via arctan2 (already radians, 0=East).
        angles = np.arctan2(raw[:, 1], raw[:, 0])
    else:
        angles = raw.reshape(-1)
        # Honor the stored unit: convert degrees to radians. NWB SpatialSeries
        # carry a free-text 'unit'; treat anything starting with "deg" as
        # degrees (e.g. "deg", "degree", "degrees").
        unit = getattr(spatial_series, "unit", None)
        if isinstance(unit, str) and unit.strip().lower().startswith("deg"):
            angles = np.deg2rad(angles)

    from neurospatial._validation import validate_lengths

    validate_lengths({"angles": angles, "timestamps": timestamps})

    return angles, timestamps
