"""
Spatial fields writing to NWB analysis containers.

This module provides functions for writing spatial analysis results
(place fields, occupancy maps) to NWB analysis/ containers.

Schema note
-----------
The estimator is stored under the metadata key ``"method"``
(``SPATIAL_RATES_SCHEMA_VERSION`` ``"2.1"``). Schema ``2.1`` additionally
persists the ``method="glm"`` GAM diagnostics (``coefficients``, ``penalty``,
``penalty_weights``, ``rank``, ``deviance``, ``converged``, ``n_iter``,
``reml_objective``); ratio-method tables carry none of them and read back with
those fields ``None``. This is a clean break with no back-compatibility shim:
tables written by earlier schema versions do not read.
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

# Schema version for the spatial-rates DynamicTable metadata blob.
# 2.0: estimator stored under the key "method"; a clean break -- older tables
#      (which keyed the estimator differently) do not read.
# 2.1: adds the method="glm" GAM diagnostics (scalars + penalty_weights in the
#      metadata blob; deviance + coefficients as per-unit table columns). Ratio
#      tables are byte-identical to 2.0 apart from the version string.
SPATIAL_RATES_SCHEMA_VERSION: str = "2.1"

# Column names within the spatial-rates DynamicTable
COL_UNIT_ID: str = "unit_id"
COL_FIRING_RATE: str = "firing_rate"
# Per-unit GAM (method="glm") columns; absent for ratio methods.
COL_DEVIANCE: str = "deviance"
COL_COEFFICIENTS: str = "coefficients"

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
    one column per ``unit_table`` field. ``method``, ``bandwidth``,
    ``n_bins``, ``n_units`` and the ``unit_table`` column names are stored as a
    JSON blob in the table description. Occupancy is stored once (it is shared
    across units) as a companion ``TimeSeries`` named ``f"{name}_occupancy"``,
    and the environment's bin-center coordinates are stored once via the shared
    ``bin_centers`` dataset (deduplicated with any place fields).

    For ``method="glm"`` results the GAM diagnostics are persisted alongside the
    rates: the scalars ``penalty`` / ``rank`` / ``n_iter`` / ``converged`` /
    ``reml_objective`` and the ``(rank,)`` ``penalty_weights`` vector go into the
    JSON description blob (``penalty`` / ``reml_objective`` as JSON ``null`` when
    ``None``), while ``deviance`` ``(n_units,)`` and ``coefficients`` (stored
    transposed as ``(n_units, rank)`` per-unit rows) become extra table columns.
    Ratio-method results carry none of these; they read back with the GAM fields
    ``None``. ``bandwidth`` is stored nullable (``None`` for glm).

    The full :class:`~neurospatial.Environment` is **persisted** alongside the
    rates (via :func:`~neurospatial.io.nwb.write_environment` under the derived
    name ``f"{name}_environment"``), so it round-trips with its connectivity
    edges and geometry intact. :func:`read_place_field` restores that
    environment when called without ``env=``; the persisted env is a separate
    copy from any default-named environment, so multiple results and a standalone
    ``write_environment`` never collide.

    The write is **atomic**: all name collisions (the table, ``f"{name}_occupancy"``
    and ``f"{name}_environment"``) and all shape validation are resolved *before*
    the first object is added, so a duplicate without ``overwrite`` raises before
    any mutation and ``overwrite=True`` cleans every companion. If a later add
    still fails, the already-added objects are rolled back and a ``ValueError`` is
    raised.

    Parameters
    ----------
    nwbfile : NWBFile
        The NWB file to write to.
    result : SpatialRatesResult
        Population spatial-rate result to serialize. Its ``env``,
        ``firing_rates`` ``(n_units, n_bins)``, ``occupancy`` ``(n_bins,)``,
        ``unit_ids``, ``unit_table``, ``method`` and ``bandwidth`` are
        all preserved.
    name : str, default "spatial_rates"
        Name for the spatial-rates DynamicTable in ``analysis/``.
    overwrite : bool, default False
        If True, replace an existing container and *all* its companions (the
        ``f"{name}_occupancy"`` TimeSeries and the ``f"{name}_environment"``
        environment). If False, raise ``ValueError`` when any of those names is
        already present.

    Raises
    ------
    ValueError
        If a container named ``name`` (or one of its companions) exists and
        ``overwrite=False``; if ``firing_rates.shape`` is not
        ``(len(unit_ids), env.n_bins)``; if ``occupancy.shape`` is not
        ``(env.n_bins,)``; if a ``unit_table`` column is named ``unit_id`` or
        ``firing_rate`` (reserved -- plus ``deviance`` / ``coefficients`` for a
        glm result, whose GAM columns share the table); or if a companion add
        fails after a partial write (the partial objects are rolled back first).
    ImportError
        If pynwb is not installed.

    Notes
    -----
    This function mutates only ``nwbfile`` (adds the DynamicTable, the companion
    occupancy TimeSeries, the persisted environment in ``scratch/`` and, if
    absent, the shared bin_centers dataset). The ``result`` object and its arrays
    are never modified -- ``firing_rates`` and ``occupancy`` are defensively
    copied before being handed to the NWB containers, so mutating the live result
    after the write does not change what was written (and vice versa).

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

    from neurospatial.io.nwb._environment import write_environment

    is_glm = result.method == "glm"

    env = result.env
    # L1: defensively COPY the arrays (np.array, not a no-copy np.asarray) so the
    # NWB containers never alias the live result's memory. Mutating
    # result.firing_rates / result.occupancy after the write must not change what
    # was written, and vice versa. dtype is preserved.
    firing_rates = np.array(result.firing_rates)
    occupancy = np.array(result.occupancy)
    unit_ids = np.asarray(result.unit_ids)
    n_units = int(unit_ids.shape[0])

    occupancy_name = f"{name}_occupancy"
    env_name = f"{name}_environment"

    # --- Preflight (BEFORE any mutation): shape + reserved-name validation ----
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

    # L3: `unit_id` / `firing_rate` are reserved for the table's own columns; a
    # unit_table column with either name would collide inside the DynamicTable.
    # For a glm result the per-unit GAM columns (`deviance` / `coefficients`) are
    # reserved too, since they are written alongside. Raise a clear param-named
    # error instead of a low-level hdmf error.
    reserved_names: tuple[str, ...] = (COL_UNIT_ID, COL_FIRING_RATE)
    if is_glm:
        reserved_names = (*reserved_names, COL_DEVIANCE, COL_COEFFICIENTS)
    unit_table = result.unit_table
    if unit_table is not None:
        reserved_clash = [
            str(c) for c in unit_table.columns if str(c) in reserved_names
        ]
        if reserved_clash:
            raise ValueError(
                f"unit_table has reserved column name(s) {reserved_clash}: "
                f"{list(reserved_names)} are reserved for the spatial-rates "
                f"table's own columns. Rename these unit_table columns."
            )

    analysis = _get_or_create_processing_module(
        nwbfile, DEFAULT_ANALYSIS_MODULE, "Analysis results including spatial fields"
    )

    # FIX 2 (atomic write): resolve ALL name collisions -- the table, its
    # companion occupancy AND the persisted environment -- BEFORE the first add.
    # A duplicate without overwrite raises before any mutation; overwrite=True
    # cleans every companion so a re-write starts clean.
    existing = [n for n in (name, occupancy_name) if n in analysis.data_interfaces]
    if env_name in nwbfile.scratch:
        existing.append(env_name)
    if existing:
        if not overwrite:
            raise ValueError(
                f"Spatial rates '{name}' already exists (found: {existing}). "
                f"Use overwrite=True to replace it and its companions."
            )
        for obj_name in (name, occupancy_name):
            if obj_name in analysis.data_interfaces:
                del analysis.data_interfaces[obj_name]
        if env_name in nwbfile.scratch:
            del nwbfile.scratch[env_name]
        logger.info("Overwriting existing spatial rates '%s' and its companions", name)

    # Shared bin_centers (deduplicated with place fields). Idempotent and shared
    # across fields, so it is intentionally NOT rolled back on a later failure.
    _ensure_bin_centers(nwbfile, env)

    # Optional per-unit metadata columns.
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

    meta_dict: dict[str, object] = {
        "schema_version": SPATIAL_RATES_SCHEMA_VERSION,
        "method": str(result.method),
        # ``bandwidth`` is nullable: ``None`` for method="glm" (which has no
        # bandwidth), a float for the ratio methods.
        "bandwidth": (None if result.bandwidth is None else float(result.bandwidth)),
        "n_bins": int(env.n_bins),
        "n_units": n_units,
        "unit_table_columns": unit_table_columns,
        "occupancy_name": occupancy_name,
    }

    # GAM (method="glm") diagnostics: scalars + the (rank,) penalty_weights go
    # into the JSON blob; the per-unit deviance and coefficients become table
    # columns. penalty / reml_objective serialize as JSON null when None.
    gam_columns: list[VectorData] = []
    if is_glm:
        rank = int(result.rank)  # type: ignore[arg-type]
        penalty = result.penalty
        reml_objective = result.reml_objective
        meta_dict["gam"] = {
            "penalty": (None if penalty is None else float(penalty)),
            "rank": rank,
            "n_iter": int(result.n_iter),  # type: ignore[arg-type]
            "converged": bool(result.converged),
            "reml_objective": (
                None if reml_objective is None else float(reml_objective)
            ),
            "penalty_weights": np.asarray(
                result.penalty_weights, dtype=np.float64
            ).tolist(),
        }
        # coefficients (rank, n_units) -> per-unit rows (n_units, rank), matching
        # the firing_rate per-unit-row layout; deviance (n_units,) as-is.
        coefficients_rows = np.array(result.coefficients, dtype=np.float64).T
        gam_columns = [
            VectorData(
                name=COL_DEVIANCE,
                description="Per-unit unpenalized Poisson deviance (method='glm').",
                data=np.array(result.deviance, dtype=np.float64),
            ),
            VectorData(
                name=COL_COEFFICIENTS,
                description=(
                    f"Per-unit GAM coefficients (n_units, rank={rank}); "
                    f"transpose back to (rank, n_units) on read (method='glm')."
                ),
                data=coefficients_rows,
            ),
        ]

    description = json.dumps(meta_dict)

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
            *gam_columns,
        ],
    )

    # Occupancy is shared across units: store once as (1, n_bins) TimeSeries.
    occupancy_ts = TimeSeries(
        name=occupancy_name,
        description=f"Occupancy (seconds) shared across units for '{name}'",
        data=occupancy.reshape(1, -1),
        unit=DEFAULT_OCCUPANCY_UNIT,
        timestamps=[DEFAULT_STATIC_TIMESTAMP],
        comments=f"Occupancy for spatial rates '{name}', n_bins={env.n_bins}.",
    )

    # FIX 2 (atomic write): add the table, its companion occupancy, then persist
    # the full environment (with connectivity + geometry). If any add after the
    # first fails, best-effort roll back the already-added objects so the file is
    # never left half-written, and surface a neurospatial-level ValueError.
    added: list[tuple[str, str]] = []
    try:
        analysis.add(table)
        added.append(("analysis", name))
        analysis.add(occupancy_ts)
        added.append(("analysis", occupancy_name))
        # Collisions were resolved in preflight, so overwrite=True here just
        # guarantees no late collision raise inside the try-block.
        write_environment(nwbfile, env, name=env_name, overwrite=True)
        added.append(("scratch", env_name))
    except Exception as exc:
        for namespace, obj_name in reversed(added):
            try:
                if namespace == "analysis":
                    del analysis.data_interfaces[obj_name]
                else:
                    del nwbfile.scratch[obj_name]
            except Exception:
                logger.warning(
                    "Rollback of '%s' failed while aborting spatial-rates write '%s'",
                    obj_name,
                    name,
                )
        raise ValueError(
            f"Aborted writing spatial rates '{name}': a companion write failed "
            f"after a partial write ({exc!r}). The already-added objects were "
            f"rolled back, but the NWB file may still need inspection."
        ) from exc

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
    ``method`` and ``bandwidth``.

    Parameters
    ----------
    nwbfile : NWBFile
        The NWB file to read from.
    name : str, default "spatial_rates"
        Name of the spatial-rates DynamicTable in ``analysis/``.
    env : Environment, optional
        Environment to attach to the result. Obtained from EXACTLY two sources,
        in order: (a) this ``env=`` argument when given (used as-is); else (b)
        the environment :func:`write_spatial_rates` persisted under
        ``f"{name}_environment"`` -- which round-trips with its connectivity
        edges and geometry **intact**, so graph operations (``neighbors``,
        ``path_between``, ...) work on the restored env. There is no
        connectivity-less fabrication from bin centers. Whichever env is
        obtained, its ``n_bins`` must equal the stored ``firing_rates`` bin
        count; a mismatched or stale ``env=`` raises ``ValueError``.

    Returns
    -------
    SpatialRatesResult
        Population result equal to the one originally written.

    Raises
    ------
    KeyError
        If the ``analysis`` module, the named table, or its companion occupancy
        is missing from ``nwbfile``.
    ValueError
        If ``name`` points at a table not written by ``write_spatial_rates``; if
        the companion occupancy length does not match the table's recorded
        ``n_bins``; if ``env=`` is omitted and no persisted environment is in the
        file; or if the obtained environment's ``n_bins`` does not match the
        stored ``firing_rates`` (a mismatched or stale ``env=``).
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

    # L2: a ``name`` pointing at a table NOT written by write_spatial_rates has a
    # plain-text (non-JSON) description; surface a clear ValueError rather than a
    # raw JSONDecodeError leaking from json.loads.
    try:
        meta = json.loads(table.description)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(
            f"'{name}' is not a spatial-rates table: its description is not the "
            f"JSON metadata written by write_spatial_rates."
        ) from exc
    if not isinstance(meta, dict) or "method" not in meta or "n_bins" not in meta:
        raise ValueError(
            f"'{name}' is not a spatial-rates table: its description JSON is "
            f"missing the expected spatial-rates metadata ('method', 'n_bins')."
        )

    method = meta["method"]
    bandwidth = meta["bandwidth"]
    unit_table_columns = meta.get("unit_table_columns", [])
    occupancy_name = meta.get("occupancy_name", f"{name}_occupancy")
    meta_n_bins = int(meta["n_bins"])

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

    # FIX 3: occupancy integrity guard at READ time. Validate the companion's
    # length against the table's OWN recorded n_bins (not occupancy's own
    # length), so a wrong-length companion is caught here rather than deferred to
    # a distant spatial_information() call.
    if occupancy.shape != (meta_n_bins,):
        raise ValueError(
            f"Occupancy companion '{occupancy_name}' has length "
            f"{occupancy.shape[0]} but spatial-rates table '{name}' records "
            f"n_bins={meta_n_bins}. The stored occupancy does not match the rates; "
            f"the file may be corrupt or the wrong companion was written."
        )

    # FIX 1: obtain the environment from EXACTLY two sources, in order --
    # (a) the explicit env= arg, else (b) the environment this writer persisted
    # under f"{name}_environment" (full connectivity + geometry). No
    # connectivity-less fabrication from bin centers.
    if env is None:
        env_name = f"{name}_environment"
        if env_name in nwbfile.scratch:
            from neurospatial.io.nwb._environment import read_environment

            env = read_environment(nwbfile, name=env_name)
        else:
            raise ValueError(
                f"Spatial rates '{name}' has no attached environment: env= was "
                f"not provided and no persisted environment '{env_name}' is in "
                f"the file. Pass env= explicitly, or (re)write with "
                f"write_spatial_rates, which now persists the environment "
                f"automatically."
            )

    # FIX 1 geometry guard (kills the silent mismatched/stale-env failure): the
    # obtained env must match the stored rates' bin count.
    n_bins = int(firing_rates.shape[1])
    if env.n_bins != n_bins:
        raise ValueError(
            f"Environment n_bins ({env.n_bins}) does not match the stored "
            f"firing_rates bin count ({n_bins}) for spatial rates '{name}'. This "
            f"usually means a mismatched or stale env= was passed; pass the "
            f"Environment used to compute these rates."
        )

    # GAM (method="glm") diagnostics: reconstruct from the JSON blob + the
    # per-unit columns, preserving shapes. Absent (ratio methods) -> all None.
    # coefficients is stored transposed as (n_units, rank); transpose back to
    # (rank, n_units). The reconstructed fields must satisfy the SpatialRatesResult
    # None-iff-glm invariant (checked at construction below) -- a wrong shape or a
    # missing field raises there, so a corrupt glm record fails loudly.
    gam_fields: dict[str, object | None] = {
        "coefficients": None,
        "penalty": None,
        "penalty_weights": None,
        "rank": None,
        "deviance": None,
        "converged": None,
        "n_iter": None,
        "reml_objective": None,
    }
    if method == "glm":
        gam = meta.get("gam")
        if not isinstance(gam, dict):
            raise ValueError(
                f"Spatial rates '{name}' has method='glm' but no GAM metadata "
                f"blob; the file may be corrupt or written by an older schema."
            )
        penalty = gam.get("penalty")
        reml_objective = gam.get("reml_objective")
        gam_fields = {
            "coefficients": np.asarray(table[COL_COEFFICIENTS][:], dtype=np.float64).T,
            "penalty": None if penalty is None else float(penalty),
            "penalty_weights": np.asarray(gam["penalty_weights"], dtype=np.float64),
            "rank": int(gam["rank"]),
            "deviance": np.asarray(table[COL_DEVIANCE][:], dtype=np.float64),
            "converged": bool(gam["converged"]),
            "n_iter": int(gam["n_iter"]),
            "reml_objective": (
                None if reml_objective is None else float(reml_objective)
            ),
        }

    return SpatialRatesResult(
        firing_rates=firing_rates,
        occupancy=occupancy,
        env=env,
        method=method,
        bandwidth=bandwidth,
        unit_ids=unit_ids,
        unit_table=unit_table,
        **gam_fields,  # type: ignore[arg-type]
    )
