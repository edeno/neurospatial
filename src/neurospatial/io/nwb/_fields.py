"""
Spatial fields writing to NWB analysis containers.

This module provides functions for writing spatial analysis results
(place fields, occupancy maps) to NWB analysis/ containers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from neurospatial.io.nwb._core import (
    _get_or_create_processing_module,
    _require_pynwb,
    logger,
)

if TYPE_CHECKING:
    from pynwb import NWBFile

    from neurospatial import Environment

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


def _validate_1d_field_shape(field: NDArray, n_bins: int) -> None:
    """
    Validate that field is 1D with shape (n_bins,).

    Used for occupancy maps which are always static (not time-varying).

    Parameters
    ----------
    field : NDArray
        The spatial field to validate.
    n_bins : int
        Expected number of bins from the environment.

    Raises
    ------
    ValueError
        If field is not 1D or doesn't match expected n_bins.
    """
    if field.ndim != 1:
        raise ValueError(
            f"Occupancy must be 1D (n_bins,), got shape {field.shape}. "
            f"For time-varying data, use write_place_field() instead."
        )
    if field.shape[0] != n_bins:
        raise ValueError(
            f"Occupancy shape {field.shape} does not match env.n_bins={n_bins}. "
            f"Expected shape ({n_bins},)."
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
    timestamps: NDArray[np.float64] | None = None,
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
    timestamps : NDArray[np.float64], optional
        Physical timestamps for time-varying fields, shape ``(n_time,)``.
        If provided for 2D fields, these timestamps are used instead of
        abstract indices. Ignored for 1D static fields.
    overwrite : bool, default False
        If True, replace existing field with same name.
        If False, raise ValueError on duplicate name.

    Raises
    ------
    ValueError
        If field with same name exists and overwrite=False.
        If field shape doesn't match env.n_bins.
        If timestamps length doesn't match field's time dimension.
    ImportError
        If pynwb is not installed.

    Notes
    -----
    - Static 1D fields of shape ``(n_bins,)`` are stored as ``(1, n_bins)``
      for NWB TimeSeries compatibility (time on 0th dimension).
    - For 2D time-varying fields of shape ``(n_time, n_bins)``:

      - If ``timestamps`` is provided, those physical timestamps are used.
      - Otherwise, timestamps default to ``[0, 1, 2, ..., n_time-1]`` as
        abstract indices.

    - The ``bin_centers`` dataset is stored once and shared across all fields
      in the same analysis module to avoid data duplication.
    - The ``overwrite`` parameter operates on in-memory NWB objects. When
      working with file-backed NWB files, ensure changes are written back.
    - For 2D fields, the time axis is stored as a simple integer index
      (0, 1, ..., n_time-1). If you need physical timestamps for each
      frame, store them separately alongside this TimeSeries.

    Examples
    --------
    >>> from pynwb import NWBHDF5IO  # doctest: +SKIP
    >>> from neurospatial import compute_place_field  # doctest: +SKIP
    >>> place_field = compute_place_field(
    ...     env, spike_times, timestamps, positions
    ... )  # doctest: +SKIP
    >>> with NWBHDF5IO("session.nwb", "r+") as io:  # doctest: +SKIP
    ...     nwbfile = io.read()
    ...     write_place_field(nwbfile, env, place_field, name="cell_001")
    ...     io.write(nwbfile)

    For time-varying fields with physical timestamps:

    >>> time_varying_field = np.random.rand(100, env.n_bins)  # doctest: +SKIP
    >>> timestamps = np.linspace(0, 10, 100)  # doctest: +SKIP
    >>> write_place_field(
    ...     nwbfile, env, time_varying_field, timestamps=timestamps
    ... )  # doctest: +SKIP
    """
    _require_pynwb()
    from pynwb import TimeSeries

    # Validate field shape
    _validate_field_shape(field, env.n_bins)

    # Validate timestamps if provided for 2D fields
    if timestamps is not None and field.ndim == 2:
        timestamps = np.asarray(timestamps, dtype=np.float64)
        if timestamps.ndim != 1:
            raise ValueError(
                f"timestamps must be 1D array, got shape {timestamps.shape}"
            )
        if len(timestamps) != field.shape[0]:
            raise ValueError(
                f"timestamps length ({len(timestamps)}) must match "
                f"field time dimension ({field.shape[0]})"
            )

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
    timestamps_for_ts: list[float] | NDArray[np.float64]
    if field.ndim == 1:
        # Static field - single timepoint
        timestamps_for_ts = [0.0]
        data = field.reshape(1, -1)  # Shape: (1, n_bins)
    else:
        # Time-varying field - use provided timestamps or sequential indices
        if timestamps is not None:
            timestamps_for_ts = timestamps
        else:
            timestamps_for_ts = np.arange(field.shape[0], dtype=np.float64)
        data = field

    place_field_ts = TimeSeries(
        name=name,
        description=description,
        data=data,
        unit=unit,
        timestamps=timestamps_for_ts,
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
    unit: str = "seconds",
    overwrite: bool = False,
) -> None:
    """
    Write occupancy map to NWB file.

    Stores occupancy values aligned with environment bin centers in analysis/.

    Parameters
    ----------
    nwbfile : NWBFile
        The NWB file to write to.
    env : Environment
        The Environment providing bin structure.
    occupancy : NDArray[np.float64], shape (n_bins,)
        Occupancy values per bin. Must be 1D array matching env.n_bins.
    name : str, default "occupancy"
        Name for the occupancy map in NWB.
    description : str, default ""
        Description of the occupancy map.
    unit : str, default "seconds"
        Units for occupancy values. Common options:

        - "seconds" (default): Time spent in each bin
        - "probability": Normalized occupancy (sums to 1)
        - "counts": Raw bin visit counts
    overwrite : bool, default False
        If True, replace existing occupancy with same name.
        If False, raise ValueError on duplicate name.

    Raises
    ------
    ValueError
        If occupancy with same name exists and overwrite=False.
        If occupancy shape doesn't match env.n_bins.
        If occupancy is not 1D.
    ImportError
        If pynwb is not installed.

    Notes
    -----
    - Occupancy maps are always static (1D), unlike place fields which can
      be time-varying (2D). For time-varying occupancy, use separate snapshots.
    - The ``bin_centers`` dataset is stored once and shared across all fields
      in the same analysis module to avoid data duplication.
    - Static 1D occupancy is stored as ``(1, n_bins)`` for NWB TimeSeries
      compatibility (time on 0th dimension).

    Examples
    --------
    >>> from pynwb import NWBHDF5IO  # doctest: +SKIP
    >>> from neurospatial import Environment  # doctest: +SKIP
    >>> env = Environment.from_samples(positions, bin_size=2.0)  # doctest: +SKIP
    >>> occupancy = env.occupancy(times, positions)  # doctest: +SKIP
    >>> with NWBHDF5IO("session.nwb", "r+") as io:  # doctest: +SKIP
    ...     nwbfile = io.read()
    ...     write_occupancy(nwbfile, env, occupancy, name="session_occupancy")
    ...     io.write(nwbfile)

    For probability-normalized occupancy:

    >>> prob_occupancy = occupancy / occupancy.sum()  # doctest: +SKIP
    >>> write_occupancy(
    ...     nwbfile, env, prob_occupancy, unit="probability"
    ... )  # doctest: +SKIP
    """
    _require_pynwb()
    from pynwb import TimeSeries

    # Validate occupancy shape (1D only)
    _validate_1d_field_shape(occupancy, env.n_bins)

    # Get or create analysis processing module
    analysis = _get_or_create_processing_module(
        nwbfile, "analysis", "Analysis results including spatial fields"
    )

    # Check for existing occupancy with same name
    if name in analysis.data_interfaces:
        if not overwrite:
            raise ValueError(
                f"Occupancy '{name}' already exists. Use overwrite=True to replace."
            )
        # Remove existing for replacement (in-memory only)
        del analysis.data_interfaces[name]
        logger.info("Overwriting existing occupancy '%s'", name)

    # Ensure bin_centers are stored (deduplicated)
    _ensure_bin_centers(nwbfile, env)

    # Create TimeSeries for the occupancy map
    # Static 1D occupancy stored as (1, n_bins) for NWB compatibility
    data = occupancy.reshape(1, -1)  # Shape: (1, n_bins)

    occupancy_ts = TimeSeries(
        name=name,
        description=description,
        data=data,
        unit=unit,
        timestamps=[0.0],  # Single timepoint - static data
        comments=f"Occupancy map with n_bins={env.n_bins}. See '{BIN_CENTERS_NAME}' for coordinates.",
    )

    analysis.add(occupancy_ts)
    logger.debug("Wrote occupancy '%s' with shape %s", name, occupancy.shape)
