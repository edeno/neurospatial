"""
Position and head direction reading from NWB behavior containers.

This module provides functions for reading Position and CompassDirection
data from pynwb.behavior containers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from neurospatial._keyword_hint import keyword_hint
from neurospatial.io.nwb._adapters import (
    data_from_series,
    require_unscaled_for_lazy,
    timestamps_from_series,
    timestamps_handle_from_series,
)
from neurospatial.io.nwb._core import _find_containers_by_type, _require_pynwb, logger
from neurospatial.io.nwb._holders import NWBHeadDirection, NWBPosition

if TYPE_CHECKING:
    from pynwb import NWBFile
    from pynwb.behavior import CompassDirection, Position, SpatialSeries


@keyword_hint("processing_module", "position_name")
def read_position(
    nwbfile: NWBFile,
    *,
    processing_module: str | None = None,
    position_name: str | None = None,
    lazy: bool = False,
) -> NWBPosition:
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
        into ``NDArray[np.float64]``. If ``True``, the h5py-backed
        ``SpatialSeries.data`` handle (and the timestamps handle) are returned
        **without** copying, so they materialize only when sliced or
        ``np.asarray``-ed. Use this to keep large recordings off-RAM. Values are
        in the series' ``unit``: stored × ``conversion`` + ``offset``;
        ``lazy=True`` raises when that map is not the identity.

    Returns
    -------
    NWBPosition
        Frozen, non-iterable holder with ``times`` in seconds, ``positions``
        in the declared physical ``units`` after conversion and offset, and
        ``units=None`` when no unit is declared. Recognized unit aliases are
        normalized to m/cm/mm/um/px without rescaling values. Arrays are eager
        float64 by default; lazy handles remain inside the holder. ``frame``
        preserves a nonempty SpatialSeries ``reference_frame`` verbatim,
        or is None when undeclared. No coordinate transform is inferred.

    Raises
    ------
    KeyError
        If no Position container found, or if specified position_name not found.
    ValueError
        If ``lazy=True`` and the series declares a non-identity ``conversion``
        or ``offset``.
    ImportError
        If pynwb is not installed.

    Notes
    -----
    Lazy handles are **only valid while the backing ``NWBFile`` / ``NWBHDF5IO``
    is open**. Slice or ``np.asarray`` them inside the ``with NWBHDF5IO(...)``
    block; accessing them after the file is closed raises an error from the HDF5
    layer. ``lazy=False`` returns read-only, fully-materialized arrays that
    remain valid after the file closes.

    Positions and timestamps must have the same length on both paths; the lazy
    path compares the handles' ``.shape[0]`` without materializing values.

    ``lazy=True`` returns the **stored** dtype (whatever the SpatialSeries was
    written with, e.g. ``float32``), whereas ``lazy=False`` casts to
    ``float64``. The values are equal; only the dtype may differ.

    Examples
    --------
    >>> from datetime import datetime, timezone
    >>> from pynwb import NWBFile, NWBHDF5IO
    >>> nwbfile = NWBFile(
    ...     "Example recording", "example", datetime(2026, 1, 1, tzinfo=timezone.utc)
    ... )
    >>> import numpy as np
    >>> from pynwb.behavior import Position, SpatialSeries
    >>> from neurospatial.io.nwb import read_position
    >>> position = Position(name="Position")
    >>> _ = position.add_spatial_series(
    ...     SpatialSeries(
    ...         name="xy",
    ...         data=np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
    ...         timestamps=np.arange(3) / 30.0,
    ...         reference_frame="arena origin",
    ...         unit="cm",
    ...     )
    ... )
    >>> _ = nwbfile.create_processing_module("behavior", "Tracking").add(position)
    >>> pos = read_position(nwbfile, processing_module="behavior", position_name="xy")
    >>> pos.positions.shape
    (3, 2)
    >>> with NWBHDF5IO("position.nwb", "w") as io:
    ...     io.write(nwbfile)
    >>> with NWBHDF5IO("position.nwb", "r") as io:
    ...     pos_lazy = read_position(io.read(), lazy=True)
    ...     first_position = np.asarray(pos_lazy.positions[:1])
    >>> first_position.tolist()
    [[0.0, 0.0]]

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
        require_unscaled_for_lazy(spatial_series, context="read_position")
        # The h5py-backed handles materialize on slice / np.asarray; the
        # holder checks their lengths from .shape without reading values.
        timestamps = timestamps_handle_from_series(spatial_series)
        positions = spatial_series.data
    else:
        timestamps = timestamps_from_series(spatial_series)
        positions = data_from_series(spatial_series)
    return NWBPosition(
        timestamps,
        positions,
        spatial_series.unit,
        frame=spatial_series.reference_frame,
    )


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
                f"Processing module '{processing_module}' not found in NWB file. Available modules: {list(nwbfile.processing.keys())}"
                + " Fix: add the requested behavior data to processing/behavior, then pass processing_module='behavior'."
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
            + " Fix: add the requested behavior data to processing/behavior, then pass processing_module='behavior'."
        )

    # Auto-discover using priority search
    found = _find_containers_by_type(nwbfile, target_type)

    if not found:
        searched_locations = ["processing/*", "acquisition"]
        raise KeyError(
            f"No {type_name} data found in NWB file. Searched: {searched_locations}"
            + " Fix: add the requested behavior data to processing/behavior, then pass processing_module='behavior'."
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


@keyword_hint("processing_module", "compass_name")
def read_head_direction(
    nwbfile: NWBFile,
    *,
    processing_module: str | None = None,
    compass_name: str | None = None,
) -> NWBHeadDirection:
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
    NWBHeadDirection
        Frozen, non-iterable holder with ``times`` in seconds and ``headings``
        in radians, zero East and increasing counterclockwise. Stored values
        pass through conversion and offset before degree-to-radian conversion
        or unit-vector arctan2 extraction.

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
    ...     hd = read_head_direction(nwbfile)
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
    raw = data_from_series(spatial_series)
    timestamps = timestamps_from_series(spatial_series)

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

    return NWBHeadDirection(timestamps, angles)
