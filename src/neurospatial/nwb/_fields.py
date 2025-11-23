"""
Spatial fields writing to NWB analysis containers.

This module provides functions for writing spatial analysis results
(place fields, occupancy maps) to NWB analysis/ containers.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from neurospatial.nwb._core import _get_or_create_processing_module, _require_pynwb

if TYPE_CHECKING:
    from pynwb import NWBFile

    from neurospatial import Environment

logger = logging.getLogger("neurospatial.nwb")

# Name for the shared bin_centers dataset in analysis module
BIN_CENTERS_NAME = "bin_centers"


def _validate_field_shape(field: NDArray, n_bins: int) -> None:
    """
    Validate that field shape is compatible with environment.

    Parameters
    ----------
    field : NDArray
        The spatial field to validate.
    n_bins : int
        Expected number of bins from the environment.

    Raises
    ------
    ValueError
        If field shape doesn't match expected n_bins.
    """
    if field.ndim == 1:
        if field.shape[0] != n_bins:
            raise ValueError(
                f"Field shape {field.shape} does not match env.n_bins={n_bins}. "
                f"Expected shape ({n_bins},) for 1D field."
            )
    elif field.ndim == 2:
        if field.shape[1] != n_bins:
            raise ValueError(
                f"Field shape {field.shape} has {field.shape[1]} bins in second "
                f"dimension, but env.n_bins={n_bins}. "
                f"Expected shape (n_time, {n_bins}) for 2D time-varying field."
            )
    else:
        raise ValueError(
            f"Field must be 1D (n_bins,) or 2D (n_time, n_bins), got shape {field.shape}"
        )


def _ensure_bin_centers(nwbfile: NWBFile, env: Environment) -> None:
    """
    Store bin_centers in analysis module if not already present.

    This ensures bin_centers are stored exactly once per environment,
    avoiding duplication when multiple fields are written.

    Parameters
    ----------
    nwbfile : NWBFile
        The NWB file.
    env : Environment
        The environment providing bin centers.
    """
    from pynwb import TimeSeries

    analysis = _get_or_create_processing_module(
        nwbfile, "analysis", "Analysis results including spatial fields"
    )

    # Only add bin_centers if not already present
    if BIN_CENTERS_NAME not in analysis.data_interfaces:
        # TimeSeries requires time on 0th dimension
        # For static data, reshape to (1, n_bins, n_dims)
        bin_centers_data = env.bin_centers[np.newaxis, :, :]  # (1, n_bins, n_dims)
        bin_centers_ts = TimeSeries(
            name=BIN_CENTERS_NAME,
            description="Spatial bin center coordinates for place fields",
            data=bin_centers_data,
            unit=env.units if env.units else "unknown",
            timestamps=[0.0],  # Single timepoint - static data
            comments=f"n_dims={env.bin_centers.shape[1]}, n_bins={env.n_bins}",
        )
        analysis.add(bin_centers_ts)
        logger.debug("Stored bin_centers with shape %s", env.bin_centers.shape)


def write_place_field(
    nwbfile: NWBFile,
    env: Environment,
    field: NDArray[np.float64],
    name: str = "place_field",
    description: str = "",
    *,
    unit: str = "Hz",
    overwrite: bool = False,
) -> None:
    """
    Write spatial field to NWB file.

    Stores field values aligned with environment bin centers in analysis/.

    Parameters
    ----------
    nwbfile : NWBFile
        The NWB file to write to.
    env : Environment
        The Environment providing bin structure.
    field : NDArray[np.float64], shape (n_bins,) or (n_time, n_bins)
        Spatial field values. First dimension must match env.n_bins
        (or second dimension for time-varying fields).
    name : str, default "place_field"
        Name for the place field in NWB.
    description : str, default ""
        Description of the place field.
    unit : str, default "Hz"
        Units for the field values. Default "Hz" is SI-compliant for firing rates.
        Other options: "spikes/s", "probability", "a.u." (arbitrary units).
    overwrite : bool, default False
        If True, replace existing field with same name.
        If False, raise ValueError on duplicate name.

    Raises
    ------
    ValueError
        If field with same name exists and overwrite=False.
        If field shape doesn't match env.n_bins.
    ImportError
        If pynwb is not installed.

    Notes
    -----
    - Static 1D fields of shape ``(n_bins,)`` are stored as ``(1, n_bins)``
      for NWB TimeSeries compatibility (time on 0th dimension).
    - The ``bin_centers`` dataset is stored once and shared across all fields
      in the same analysis module to avoid data duplication.
    - The ``overwrite`` parameter operates on in-memory NWB objects. When
      working with file-backed NWB files, ensure changes are written back.

    Examples
    --------
    >>> from pynwb import NWBHDF5IO
    >>> from neurospatial import compute_place_field
    >>> place_field = compute_place_field(env, spike_times, timestamps, positions)
    >>> with NWBHDF5IO("session.nwb", "r+") as io:
    ...     nwbfile = io.read()
    ...     write_place_field(nwbfile, env, place_field, name="cell_001")
    ...     io.write(nwbfile)
    """
    _require_pynwb()
    from pynwb import TimeSeries

    # Validate field shape
    _validate_field_shape(field, env.n_bins)

    # Get or create analysis processing module
    analysis = _get_or_create_processing_module(
        nwbfile, "analysis", "Analysis results including spatial fields"
    )

    # Check for existing field with same name
    if name in analysis.data_interfaces:
        if not overwrite:
            raise ValueError(
                f"Place field '{name}' already exists. Use overwrite=True to replace."
            )
        # Remove existing field for replacement
        # Note: pynwb doesn't have a direct remove method, we need to recreate
        # For now, we'll delete the reference (in-memory only, works for testing)
        del analysis.data_interfaces[name]
        logger.info("Overwriting existing place field '%s'", name)

    # Ensure bin_centers are stored (deduplicated)
    _ensure_bin_centers(nwbfile, env)

    # Create TimeSeries for the place field
    # Use rate=1.0 for static fields, or actual rate for time-varying
    timestamps_list: list[float] | NDArray[np.float64]
    if field.ndim == 1:
        # Static field - single timepoint
        timestamps_list = [0.0]
        data = field.reshape(1, -1)  # Shape: (1, n_bins)
    else:
        # Time-varying field - use sequential timestamps
        timestamps_list = np.arange(field.shape[0], dtype=np.float64)
        data = field

    place_field_ts = TimeSeries(
        name=name,
        description=description,
        data=data,
        unit=unit,
        timestamps=timestamps_list,
        comments=f"Spatial field with n_bins={env.n_bins}. See '{BIN_CENTERS_NAME}' for coordinates.",
    )

    analysis.add(place_field_ts)
    logger.debug("Wrote place field '%s' with shape %s", name, field.shape)


def write_occupancy(
    nwbfile: NWBFile,
    env: Environment,
    occupancy: NDArray[np.float64],
    name: str = "occupancy",
    description: str = "",
    *,
    units: str = "seconds",
    overwrite: bool = False,
) -> None:
    """
    Write occupancy map to NWB file.

    Parameters
    ----------
    nwbfile : NWBFile
        The NWB file to write to.
    env : Environment
        The Environment providing bin structure.
    occupancy : NDArray[np.float64], shape (n_bins,)
        Occupancy values per bin.
    name : str, default "occupancy"
        Name for the occupancy map in NWB.
    description : str, default ""
        Description of the occupancy map.
    units : str, default "seconds"
        Units for occupancy values (e.g., "seconds", "probability").
    overwrite : bool, default False
        If True, replace existing occupancy with same name.

    Raises
    ------
    ValueError
        If occupancy shape doesn't match env.n_bins.
    ImportError
        If pynwb is not installed.
    """
    raise NotImplementedError("write_occupancy not yet implemented")
