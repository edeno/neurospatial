"""
Spatial fields writing to NWB analysis containers.

This module provides functions for writing spatial analysis results
(place fields, occupancy maps) to NWB analysis/ containers.
"""

from __future__ import annotations

import json
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
    from neurospatial.encoding.spatial import SpatialRatesResult

# =============================================================================
# Constants for NWB spatial fields
# =============================================================================

# Name for the shared bin_centers dataset in analysis module
BIN_CENTERS_NAME: str = "bin_centers"

# Default names for field containers
DEFAULT_PLACE_FIELD_NAME: str = "place_field"
DEFAULT_OCCUPANCY_NAME: str = "occupancy"

# Default name for the population spatial-rates container (unit axis)
DEFAULT_SPATIAL_RATES_NAME: str = "spatial_rates"

# Schema version for the spatial-rates DynamicTable metadata blob
SPATIAL_RATES_SCHEMA_VERSION: str = "1.0"

# Column names within the spatial-rates DynamicTable
COL_UNIT_ID: str = "unit_id"
COL_FIRING_RATE: str = "firing_rate"

# Default processing module for analysis results
DEFAULT_ANALYSIS_MODULE: str = "analysis"

# Default units for field values
DEFAULT_FIELD_UNIT: str = "Hz"  # SI-compliant for firing rates
DEFAULT_OCCUPANCY_UNIT: str = "seconds"

# Default timestamp for static data
DEFAULT_STATIC_TIMESTAMP: float = 0.0


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
        nwbfile, DEFAULT_ANALYSIS_MODULE, "Analysis results including spatial fields"
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
            timestamps=[DEFAULT_STATIC_TIMESTAMP],  # Single timepoint - static data
            comments=f"n_dims={env.bin_centers.shape[1]}, n_bins={env.n_bins}",
        )
        analysis.add(bin_centers_ts)
        logger.debug("Stored bin_centers with shape %s", env.bin_centers.shape)


def write_place_field(
    nwbfile: NWBFile,
    env: Environment,
    field: NDArray[np.float64],
    name: str = DEFAULT_PLACE_FIELD_NAME,
    description: str = "",
    *,
    unit: str = DEFAULT_FIELD_UNIT,
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
    >>> from neurospatial.encoding import compute_spatial_rate  # doctest: +SKIP
    >>> place_field = compute_spatial_rate(
    ...     env, spike_times, timestamps, positions
    ... ).firing_rate  # doctest: +SKIP
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
        nwbfile, DEFAULT_ANALYSIS_MODULE, "Analysis results including spatial fields"
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
        timestamps_for_ts = [DEFAULT_STATIC_TIMESTAMP]
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
    name: str = DEFAULT_OCCUPANCY_NAME,
    description: str = "",
    *,
    unit: str = DEFAULT_OCCUPANCY_UNIT,
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
        nwbfile, DEFAULT_ANALYSIS_MODULE, "Analysis results including spatial fields"
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
        timestamps=[DEFAULT_STATIC_TIMESTAMP],  # Single timepoint - static data
        comments=f"Occupancy map with n_bins={env.n_bins}. See '{BIN_CENTERS_NAME}' for coordinates.",
    )

    analysis.add(occupancy_ts)
    logger.debug("Wrote occupancy '%s' with shape %s", name, occupancy.shape)


# =============================================================================
# Population spatial-rates round-trip (SpatialRatesResult <-> NWB)
# =============================================================================


def write_spatial_rates(
    nwbfile: NWBFile,
    result: SpatialRatesResult,
    *,
    name: str = DEFAULT_SPATIAL_RATES_NAME,
    overwrite: bool = False,
) -> None:
    """
    Write a population :class:`SpatialRatesResult` to NWB with a unit axis.

    Stores the whole population -- per-unit firing-rate maps, shared occupancy,
    unit identities, and optional per-unit metadata -- so that
    :func:`read_place_field` reconstructs an equal ``SpatialRatesResult``.

    The container is a :class:`~hdmf.common.DynamicTable` (one row per unit) in
    the ``analysis`` processing module, with a ``unit_id`` column, a 2-D
    ``firing_rate`` column of shape ``(n_units, n_bins)`` on the bin axis, and
    one column per ``unit_table`` field. ``smoothing_method``, ``bandwidth``,
    ``n_bins``, ``n_units`` and the ``unit_table`` column names are stored as a
    JSON blob in the table description. Occupancy is stored once (it is shared
    across units) as a companion ``TimeSeries`` named ``f"{name}_occupancy"``,
    and the environment's bin-center coordinates are stored once via the shared
    ``bin_centers`` dataset (deduplicated with any place fields).

    Parameters
    ----------
    nwbfile : NWBFile
        The NWB file to write to.
    result : SpatialRatesResult
        Population spatial-rate result to serialize. Its ``env``,
        ``firing_rates`` ``(n_units, n_bins)``, ``occupancy`` ``(n_bins,)``,
        ``unit_ids``, ``unit_table``, ``smoothing_method`` and ``bandwidth`` are
        all preserved.
    name : str, default "spatial_rates"
        Name for the spatial-rates DynamicTable in ``analysis/``.
    overwrite : bool, default False
        If True, replace an existing container (and its companion occupancy)
        with the same name. If False, raise ``ValueError`` on a duplicate name.

    Raises
    ------
    ValueError
        If a container named ``name`` exists and ``overwrite=False``; if
        ``firing_rates.shape`` is not ``(len(unit_ids), env.n_bins)``; or if
        ``occupancy.shape`` is not ``(env.n_bins,)``.
    ImportError
        If pynwb is not installed.

    Notes
    -----
    This function mutates only ``nwbfile`` (adds the DynamicTable, the companion
    occupancy TimeSeries and, if absent, the shared bin_centers dataset). The
    ``result`` object and its arrays are never modified.

    Examples
    --------
    >>> from pynwb import NWBHDF5IO  # doctest: +SKIP
    >>> from neurospatial.encoding import compute_spatial_rates  # doctest: +SKIP
    >>> rates = compute_spatial_rates(
    ...     env, spike_times, times, positions
    ... )  # doctest: +SKIP
    >>> with NWBHDF5IO("session.nwb", "r+") as io:  # doctest: +SKIP
    ...     nwbfile = io.read()
    ...     write_spatial_rates(nwbfile, rates, name="ca1_place_fields")
    ...     io.write(nwbfile)
    """
    _require_pynwb()
    from hdmf.common import DynamicTable, VectorData
    from pynwb import TimeSeries

    env = result.env
    firing_rates = np.asarray(result.firing_rates)
    occupancy = np.asarray(result.occupancy)
    unit_ids = np.asarray(result.unit_ids)
    n_units = int(unit_ids.shape[0])

    # Validate shapes with param-named errors.
    if firing_rates.shape != (n_units, env.n_bins):
        raise ValueError(
            f"firing_rates shape {firing_rates.shape} does not match "
            f"(len(unit_ids), env.n_bins) = ({n_units}, {env.n_bins})."
        )
    if occupancy.shape != (env.n_bins,):
        raise ValueError(
            f"occupancy shape {occupancy.shape} does not match "
            f"(env.n_bins,) = ({env.n_bins},)."
        )

    analysis = _get_or_create_processing_module(
        nwbfile, DEFAULT_ANALYSIS_MODULE, "Analysis results including spatial fields"
    )
    occupancy_name = f"{name}_occupancy"

    # Overwrite handling (mirror write_place_field): drop the table AND its
    # companion occupancy so a re-write starts clean.
    if name in analysis.data_interfaces:
        if not overwrite:
            raise ValueError(
                f"Spatial rates '{name}' already exists. Use overwrite=True to replace."
            )
        del analysis.data_interfaces[name]
        if occupancy_name in analysis.data_interfaces:
            del analysis.data_interfaces[occupancy_name]
        logger.info("Overwriting existing spatial rates '%s'", name)

    # Shared bin_centers (deduplicated with place fields).
    _ensure_bin_centers(nwbfile, env)

    # Optional per-unit metadata columns.
    unit_table = result.unit_table
    unit_table_columns: list[str] = []
    extra_columns: list[VectorData] = []
    if unit_table is not None:
        unit_table_columns = [str(c) for c in unit_table.columns]
        extra_columns = [
            VectorData(
                name=str(col),
                description=f"unit_table column '{col}'",
                data=unit_table[col].to_numpy(),
            )
            for col in unit_table.columns
        ]

    description = json.dumps(
        {
            "schema_version": SPATIAL_RATES_SCHEMA_VERSION,
            "smoothing_method": str(result.smoothing_method),
            "bandwidth": float(result.bandwidth),
            "n_bins": int(env.n_bins),
            "n_units": n_units,
            "unit_table_columns": unit_table_columns,
            "occupancy_name": occupancy_name,
        }
    )

    table = DynamicTable(
        name=name,
        description=description,
        columns=[
            VectorData(
                name=COL_UNIT_ID,
                description="Per-unit identity labels (unit_ids)",
                data=unit_ids,
            ),
            VectorData(
                name=COL_FIRING_RATE,
                description=(
                    f"Per-unit firing-rate map (n_units, n_bins) in Hz; "
                    f"n_bins={env.n_bins}. See 'bin_centers' for coordinates."
                ),
                data=firing_rates,
            ),
            *extra_columns,
        ],
    )
    analysis.add(table)

    # Occupancy is shared across units: store once as (1, n_bins) TimeSeries.
    occupancy_ts = TimeSeries(
        name=occupancy_name,
        description=f"Occupancy (seconds) shared across units for '{name}'",
        data=occupancy.reshape(1, -1),
        unit=DEFAULT_OCCUPANCY_UNIT,
        timestamps=[DEFAULT_STATIC_TIMESTAMP],
        comments=f"Occupancy for spatial rates '{name}', n_bins={env.n_bins}.",
    )
    analysis.add(occupancy_ts)

    logger.debug(
        "Wrote spatial rates '%s' with %d units and %d bins",
        name,
        n_units,
        env.n_bins,
    )


def read_place_field(
    nwbfile: NWBFile,
    *,
    name: str = DEFAULT_SPATIAL_RATES_NAME,
    env: Environment | None = None,
) -> SpatialRatesResult:
    """
    Read a population :class:`SpatialRatesResult` written by ``write_spatial_rates``.

    The inverse of :func:`write_spatial_rates`. Reconstructs the per-unit
    firing-rate maps, shared occupancy, ``unit_ids``, optional ``unit_table``,
    ``smoothing_method`` and ``bandwidth``.

    Parameters
    ----------
    nwbfile : NWBFile
        The NWB file to read from.
    name : str, default "spatial_rates"
        Name of the spatial-rates DynamicTable in ``analysis/``.
    env : Environment, optional
        Environment to attach to the result. If provided, it is used as-is
        (recommended -- it preserves the original bin geometry exactly). If
        ``None``, an environment is recovered from the file: the stored
        environment (via :func:`read_environment`) when one is present in
        ``scratch/``, otherwise a minimal KDTree-backed environment
        reconstructed from the shared ``bin_centers`` coordinates.

    Returns
    -------
    SpatialRatesResult
        Population result equal to the one originally written.

    Raises
    ------
    KeyError
        If the ``analysis`` module, the named table, or its companion occupancy
        is missing from ``nwbfile``.
    ImportError
        If pynwb is not installed.

    Examples
    --------
    >>> from pynwb import NWBHDF5IO  # doctest: +SKIP
    >>> with NWBHDF5IO("session.nwb", "r") as io:  # doctest: +SKIP
    ...     nwbfile = io.read()
    ...     rates = read_place_field(nwbfile, name="ca1_place_fields")
    ...     rates.firing_rates.shape  # (n_units, n_bins)
    """
    _require_pynwb()
    from neurospatial.encoding.spatial import SpatialRatesResult

    if DEFAULT_ANALYSIS_MODULE not in nwbfile.processing:
        raise KeyError(
            f"No '{DEFAULT_ANALYSIS_MODULE}' processing module in NWB file; "
            f"cannot read spatial rates '{name}'."
        )
    analysis = nwbfile.processing[DEFAULT_ANALYSIS_MODULE]
    if name not in analysis.data_interfaces:
        available = list(analysis.data_interfaces.keys())
        raise KeyError(
            f"Spatial rates '{name}' not found in analysis/. Available: {available}"
        )

    table = analysis[name]
    meta = json.loads(table.description)
    smoothing_method = meta["smoothing_method"]
    bandwidth = meta["bandwidth"]
    unit_table_columns = meta.get("unit_table_columns", [])
    occupancy_name = meta.get("occupancy_name", f"{name}_occupancy")

    firing_rates = np.asarray(table[COL_FIRING_RATE][:])
    unit_ids = np.asarray(table[COL_UNIT_ID][:])

    unit_table = None
    if unit_table_columns:
        import pandas as pd

        unit_table = pd.DataFrame(
            {col: np.asarray(table[col][:]) for col in unit_table_columns}
        )

    if occupancy_name not in analysis.data_interfaces:
        raise KeyError(
            f"Occupancy '{occupancy_name}' for spatial rates '{name}' not found "
            "in analysis/."
        )
    occupancy = np.asarray(analysis[occupancy_name].data[:]).reshape(-1)

    if env is None:
        env = _read_or_reconstruct_env(nwbfile)

    return SpatialRatesResult(
        firing_rates=firing_rates,
        occupancy=occupancy,
        env=env,
        smoothing_method=smoothing_method,
        bandwidth=bandwidth,
        unit_ids=unit_ids,
        unit_table=unit_table,
    )


def _read_bin_centers(nwbfile: NWBFile) -> NDArray[np.float64]:
    """Read the shared bin_centers dataset stored by ``_ensure_bin_centers``.

    The dataset is stored with a leading singleton time axis
    ``(1, n_bins, n_dims)``; the leading axis is dropped here.
    """
    analysis = nwbfile.processing[DEFAULT_ANALYSIS_MODULE]
    data = np.asarray(analysis[BIN_CENTERS_NAME].data[:], dtype=np.float64)
    if data.ndim == 3:
        data = data[0]
    return np.asarray(data, dtype=np.float64)


def _read_or_reconstruct_env(nwbfile: NWBFile) -> Environment:
    """Recover an Environment for a spatial-rates result read without ``env=``.

    Prefers a full environment stored in ``scratch/`` (via
    :func:`read_environment`); otherwise builds a minimal KDTree-backed
    environment from the shared ``bin_centers`` coordinates.
    """
    from neurospatial.io.nwb._environment import (
        DEFAULT_ENVIRONMENT_NAME,
        read_environment,
    )

    if nwbfile.scratch is not None and DEFAULT_ENVIRONMENT_NAME in nwbfile.scratch:
        return read_environment(nwbfile)

    return _minimal_env_from_bin_centers(_read_bin_centers(nwbfile))


def _minimal_env_from_bin_centers(
    bin_centers: NDArray[np.float64],
) -> Environment:
    """Build a minimal KDTree-backed Environment from bin-center coordinates.

    Reuses the reconstruction machinery from ``_environment`` (the same path
    ``read_environment`` uses for non-grid layouts). The result has the correct
    ``n_bins`` and ``bin_centers`` for peak/coordinate accessors; it does not
    reconstruct the original layout engine's exact bin geometry. Pass the
    original ``env=`` to :func:`read_place_field` when exact geometry matters.
    """
    from neurospatial import Environment
    from neurospatial.io.nwb._environment import (
        _reconstruct_graph,
        _ReconstructedLayout,
    )

    bin_centers = np.asarray(bin_centers, dtype=np.float64)
    n_dims = bin_centers.shape[1]
    connectivity = _reconstruct_graph(
        bin_centers,
        np.empty((0, 2), dtype=np.int64),
        np.empty((0,), dtype=np.float64),
    )
    dimension_ranges = [
        (float(bin_centers[:, d].min()), float(bin_centers[:, d].max()))
        for d in range(n_dims)
    ]
    layout = _ReconstructedLayout(
        bin_centers=bin_centers,
        connectivity=connectivity,
        dimension_ranges=dimension_ranges,
        layout_type="unknown",
    )
    env = Environment(layout=layout)  # type: ignore[arg-type]
    env._setup_from_layout()
    return env
