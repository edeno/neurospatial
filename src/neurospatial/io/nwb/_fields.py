"""
Spatial firing-rate maps in NWB files.

``write_spatial_rates`` stores a population ``SpatialRatesResult`` -- rate
maps, occupancy and the environment they were estimated on -- and
``read_spatial_rates`` restores it.

Schema note
-----------
The estimator is stored under the metadata key ``"method"``
(``SPATIAL_RATES_SCHEMA_VERSION`` ``"2.5"``). Schema ``2.1`` persists the
``method="glm"`` GAM diagnostics (``coefficients``, ``penalty``,
``penalty_weights``, ``rank``, ``deviance``, ``converged``, ``n_iter``,
``reml_objective``). Schema ``2.2`` adds the per-neuron-``lambda``
(``pooled=False``) diagnostics: ``pooled`` and a scalar ``reml_at_boundary`` in
the metadata blob, and -- when ``penalty`` / ``reml_objective`` /
``reml_at_boundary`` are per-unit vectors -- those plus ``penalty_selected_by_reml``
as extra per-unit table columns. Ratio-method tables carry none of the GAM
fields and read back with them ``None``. Schema ``2.3`` adds
``unit_ids_generated`` and ``spike_window`` to the metadata, preserving decoder
unit-pairing semantics and spike-recording coverage. Older files default to
explicit unit labels and assumed coverage; their missing provenance cannot be
recovered from the stored labels or rate maps.
Schema ``2.4`` adds resolved ``analysis_settings`` (including epochs, observed
intervals and occupancy thresholds). Older files report these as unknown.
Schema ``2.5`` adds ``min_occupancy``, the occupancy threshold in seconds that
defines ``spatial_support`` (files without it are refused), and each unit's
``n_spikes_outside_intervals`` (``null`` when unknown).

``read_spatial_rates`` enforces the schema **major** version: it reads only tables
whose ``schema_version`` shares the current major (``2.x`` -- e.g. ``2.0``,
``2.1``, ``2.2``, and ``2.3`` are readable, since a minor bump only *adds*
optional fields), and rejects any other (missing, ``1.x``, or an incompatible
future major) with a clear ``ValueError``. A **schema-2.1** glm file carries no
``pooled`` key; the reader defaults it **method-conditionally** -- ``pooled=True``
for ``method="glm"`` (the only glm behavior that schema had) and ``pooled=None``
for a ratio method (where ``pooled`` is meaningless) -- and a missing
``reml_at_boundary`` / vector columns read back as ``None``. This is the
clean break with no back-compatibility shim for the ``1.x`` layout: pre-0.9
tables (which keyed the estimator as ``"smoothing_method"``) do not read.

Forward compatibility is *not* guaranteed (the usual minor-bump direction): a
pre-2.2 reader on a ``pooled=False`` (per-unit) 2.2 file shares the ``2.x`` major
and reads it, but sees the blob's ``penalty`` / ``reml_objective`` JSON ``null``
and ignores the unknown per-unit columns -- silently reconstructing a
``penalty=None`` result and dropping the per-unit ``lambda`` data. Read a
``pooled=False`` file with a >=2.2 reader.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np

from neurospatial._intervals import as_intervals
from neurospatial.io.nwb._core import (
    _get_or_create_processing_module,
    _require_pynwb,
    logger,
)

if TYPE_CHECKING:
    from pynwb import NWBFile

    from neurospatial.encoding.spatial import SpatialRatesResult
    from neurospatial.environment import Environment

# =============================================================================
# Constants for NWB spatial fields
# =============================================================================

# Default name for the population spatial-rates container (unit axis)
DEFAULT_SPATIAL_RATES_NAME: str = "spatial_rates"

# Schema version for the spatial-rates DynamicTable metadata blob.
# 2.0: estimator stored under the key "method"; a clean break -- older tables
#      (which keyed the estimator differently) do not read.
# 2.1: adds the method="glm" GAM diagnostics (scalars + penalty_weights in the
#      metadata blob; deviance + coefficients as per-unit table columns). Ratio
#      tables are byte-identical to 2.0 apart from the version string.
# 2.2: adds the per-neuron-lambda (pooled=False) diagnostics -- ``pooled`` and a
#      scalar ``reml_at_boundary`` in the blob, plus ``penalty`` /
#      ``reml_objective`` / ``reml_at_boundary`` / ``penalty_selected_by_reml``
#      as per-unit columns when they are vectors. A 2.1 file (no ``pooled`` key)
#      reads back method-conditionally (glm -> pooled=True, ratio -> None).
# 2.3: adds generated-unit-label provenance and spike-recording windows.
# 2.4: records resolved analysis settings for reproducible spatial support.
SPATIAL_RATES_SCHEMA_VERSION: str = "2.5"

# The reader accepts any table sharing this MAJOR version (minor bumps only add
# optional fields, so 2.0/2.1/... are mutually readable); other majors (missing,
# 1.x, or an incompatible future major) are rejected on read.
SPATIAL_RATES_SCHEMA_MAJOR: str = SPATIAL_RATES_SCHEMA_VERSION.split(".")[0]

# Column names within the spatial-rates DynamicTable
COL_UNIT_ID: str = "unit_id"
COL_FIRING_RATE: str = "firing_rate"
# Per-unit GAM (method="glm") columns; absent for ratio methods.
COL_DEVIANCE: str = "deviance"
COL_COEFFICIENTS: str = "coefficients"
# Per-unit-lambda (pooled=False) columns; present only when penalty/reml_objective
# are per-unit vectors (schema 2.2).
COL_PENALTY: str = "penalty"
COL_REML_OBJECTIVE: str = "reml_objective"
COL_REML_AT_BOUNDARY: str = "reml_at_boundary"
COL_PENALTY_SELECTED: str = "penalty_selected_by_reml"

# Default processing module for analysis results
DEFAULT_ANALYSIS_MODULE: str = "analysis"

DEFAULT_OCCUPANCY_UNIT: str = "seconds"

# Default timestamp for static data
DEFAULT_STATIC_TIMESTAMP: float = 0.0


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
    :func:`read_spatial_rates` reconstructs an equal ``SpatialRatesResult``.

    The container is a :class:`~hdmf.common.DynamicTable` (one row per unit) in
    the ``analysis`` processing module, with a ``unit_id`` column, a 2-D
    ``firing_rate`` column of shape ``(n_units, n_bins)`` on the bin axis, and
    one column per ``unit_table`` field. ``method``, ``bandwidth``,
    ``n_bins``, ``n_units``, ``unit_ids_generated``, ``spike_window`` and the
    ``unit_table`` column names are stored as a JSON blob in the table
    description. Occupancy is stored once (it is shared
    across units) as a companion ``TimeSeries`` named ``f"{name}_occupancy"``.

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
    edges and geometry intact. :func:`read_spatial_rates` restores that
    environment when called without ``env=``; the persisted env is a separate
    copy from any default-named environment, so multiple results and a standalone
    ``write_environment`` never collide.

    The write is **atomic**: all name collisions (the table, ``f"{name}_occupancy"``
    and ``f"{name}_environment"``) and all validation are resolved *before*
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
        ``unit_ids`` (including whether generated), ``spike_window``,
        ``unit_table``, ``method`` and ``bandwidth`` are all preserved.
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
        ``overwrite=False``; if a ``unit_table`` column is named ``unit_id`` or
        ``firing_rate`` (reserved -- plus ``deviance`` / ``coefficients`` for a
        glm result, whose GAM columns share the table); or if a companion add
        fails after a partial write (the partial objects are rolled back first).
    ImportError
        If pynwb is not installed.

    Notes
    -----
    This function mutates only ``nwbfile`` (adds the DynamicTable, the companion
    occupancy TimeSeries and the persisted environment in ``scratch/``). The
    ``result`` object and its arrays
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
    # The per-unit-lambda columns exist only for a pooled=False per-unit result
    # (keyed off the provenance mask, exactly as the writer decides below), so
    # they are reserved only then -- a pooled=True/fixed-penalty glm write does
    # not emit them and must not reject a unit_table column named e.g. "penalty".
    is_per_unit = is_glm and result.penalty_selected_by_reml is not None

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

    # --- Preflight (BEFORE any mutation): reserved-name validation ----------
    # Shapes need no check here: SpatialRatesResult refuses firing_rates or
    # occupancy that do not match its env's bins when it is constructed.

    # L3: `unit_id` / `firing_rate` are reserved for the table's own columns; a
    # unit_table column with either name would collide inside the DynamicTable.
    # A glm result always writes `deviance` / `coefficients`, so those are always
    # reserved; the per-unit-lambda columns (`penalty` / `reml_objective` /
    # `reml_at_boundary` / `penalty_selected_by_reml`) are written -- and thus
    # reserved -- only for a pooled=False per-unit result. Raise a clear
    # param-named error instead of a low-level hdmf error.
    reserved_names: tuple[str, ...] = (COL_UNIT_ID, COL_FIRING_RATE)
    if is_glm:
        reserved_names = (*reserved_names, COL_DEVIANCE, COL_COEFFICIENTS)
    if is_per_unit:
        reserved_names = (
            *reserved_names,
            COL_PENALTY,
            COL_REML_OBJECTIVE,
            COL_REML_AT_BOUNDARY,
            COL_PENALTY_SELECTED,
        )
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
    # A duplicate without overwrite raises before any mutation.
    #
    # For overwrite=True we CAPTURE the existing objects but do NOT delete them
    # here: deleting the originals before the replacements are built and added
    # would destroy a valid record if a later build/add fails (the rollback then
    # only removes the partial new objects, never restoring the old ones). The
    # deletion is deferred into the try-block below, right before the adds, and
    # the captured originals are restored if that block raises.
    existing = [n for n in (name, occupancy_name) if n in analysis.data_interfaces]
    if env_name in nwbfile.scratch:
        existing.append(env_name)
    old_objects: list[tuple[str, str, object]] = []
    if existing:
        if not overwrite:
            raise ValueError(
                f"Spatial rates '{name}' already exists (found: {existing}). "
                f"Use overwrite=True to replace it and its companions."
            )
        for obj_name in (name, occupancy_name):
            if obj_name in analysis.data_interfaces:
                old_objects.append(
                    ("analysis", obj_name, analysis.data_interfaces[obj_name])
                )
        if env_name in nwbfile.scratch:
            old_objects.append(("scratch", env_name, nwbfile.scratch[env_name]))
        logger.info("Overwriting existing spatial rates '%s' and its companions", name)

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
        "unit_ids_generated": bool(result._unit_ids_generated),
        "min_occupancy": float(result.min_occupancy),
        "n_spikes_outside_intervals": (
            None
            if result.n_spikes_outside_intervals is None
            else result.n_spikes_outside_intervals.tolist()
        ),
        "analysis_settings": result.analysis_settings,
        "spike_window": (
            None if result.spike_window is None else result.spike_window.tolist()
        ),
    }

    # GAM (method="glm") diagnostics: scalars + the (rank,) penalty_weights go
    # into the JSON blob; the per-unit deviance and coefficients become table
    # columns. Under pooled=False (per-unit lambda) the penalty / reml_objective /
    # reml_at_boundary vectors and the penalty_selected_by_reml mask also become
    # per-unit columns; the blob's scalar slots then carry JSON null. ``pooled``
    # (the only reliable source -- scalar cases are value-identical either way)
    # and ``per_unit`` (whether the vector columns are present) always ride along.
    gam_columns: list[VectorData] = []
    if is_glm:
        rank = int(result.rank)  # type: ignore[arg-type]
        # The per-unit vector path is exactly when the provenance mask exists
        # (same discriminant as the reserved-name gate above).
        per_unit = is_per_unit
        penalty = result.penalty
        reml_objective = result.reml_objective
        reml_at_boundary = result.reml_at_boundary
        gam_meta: dict[str, object] = {
            "rank": rank,
            "n_iter": int(result.n_iter),  # type: ignore[arg-type]
            "converged": bool(result.converged),
            "penalty_weights": np.asarray(
                result.penalty_weights, dtype=np.float64
            ).tolist(),
            "pooled": bool(result.pooled),
            "per_unit": per_unit,
        }
        # Scalar slots: real values when scalar, JSON null when the value lives in
        # a per-unit column instead (per_unit=True). A schema-2.1 reader will not
        # find ``reml_at_boundary`` / ``pooled`` / ``per_unit`` and defaults them.
        gam_meta["penalty"] = None if per_unit or penalty is None else float(penalty)
        gam_meta["reml_objective"] = (
            None if per_unit or reml_objective is None else float(reml_objective)
        )
        gam_meta["reml_at_boundary"] = (
            None if per_unit or reml_at_boundary is None else bool(reml_at_boundary)
        )
        meta_dict["gam"] = gam_meta
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
        if per_unit:
            # Per-unit-lambda vectors (pooled=False). reml_objective carries nan
            # for zero-spike fallback units; the provenance mask distinguishes
            # REML-selected (True) from fallback (False) units.
            gam_columns += [
                VectorData(
                    name=COL_PENALTY,
                    description="Per-unit smoothing penalty lambda (pooled=False).",
                    data=np.asarray(result.penalty, dtype=np.float64),
                ),
                VectorData(
                    name=COL_REML_OBJECTIVE,
                    description=(
                        "Per-unit minimized REML objective (nan for zero-spike "
                        "fallback units; pooled=False)."
                    ),
                    data=np.asarray(result.reml_objective, dtype=np.float64),
                ),
                VectorData(
                    name=COL_REML_AT_BOUNDARY,
                    description=(
                        "Per-unit REML boundary flag: lambda on a search bound "
                        "(weakly identified; pooled=False)."
                    ),
                    data=np.asarray(result.reml_at_boundary, dtype=bool),
                ),
                VectorData(
                    name=COL_PENALTY_SELECTED,
                    description=(
                        "Per-unit provenance: True = own REML lambda, False = "
                        "pooled-lambda fallback (zero-spike unit; pooled=False)."
                    ),
                    data=np.asarray(result.penalty_selected_by_reml, dtype=bool),
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
                    f"n_bins={env.n_bins}; bin coordinates are in the environment "
                    f"saved as '{env_name}'."
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

    # FIX 2 (atomic write): remove any originals (overwrite=True), then add the
    # table, its companion occupancy, and persist the full environment (with
    # connectivity + geometry). The originals are deleted HERE (not in preflight)
    # so that if any step fails we can both roll back the newly-added objects AND
    # restore the originals -- an overwrite that aborts leaves the pre-existing
    # record intact rather than destroying it.
    added: list[tuple[str, str]] = []
    removed: list[tuple[str, str, object]] = []
    try:
        for namespace, obj_name, _obj in old_objects:
            if namespace == "analysis":
                del analysis.data_interfaces[obj_name]
            else:
                del nwbfile.scratch[obj_name]
            removed.append((namespace, obj_name, _obj))
        analysis.add(table)
        added.append(("analysis", name))
        analysis.add(occupancy_ts)
        added.append(("analysis", occupancy_name))
        # Originals were removed just above, so overwrite=True here just
        # guarantees no late collision raise inside the try-block.
        write_environment(nwbfile, env, name=env_name, overwrite=True)
        added.append(("scratch", env_name))
    except Exception as exc:
        # Roll back the newly-added (partial) objects...
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
        # ...then restore the originals we removed, so a failed overwrite does not
        # destroy the pre-existing record. Skip any name a partial write left
        # behind (restore only what is actually missing).
        for namespace, obj_name, obj in reversed(removed):
            try:
                if namespace == "analysis":
                    if obj_name not in analysis.data_interfaces:
                        analysis.add(obj)
                elif obj_name not in nwbfile.scratch:
                    nwbfile.add_scratch(obj)
            except Exception:
                logger.warning(
                    "Restore of original '%s' failed while aborting spatial-rates "
                    "overwrite '%s'; the pre-existing record may be lost.",
                    obj_name,
                    name,
                )
        raise ValueError(
            f"Aborted writing spatial rates '{name}': a companion write failed "
            f"after a partial write ({exc!r}). The newly-added objects were rolled "
            f"back and any pre-existing record restored, but the NWB file may still "
            f"need inspection."
        ) from exc

    logger.debug(
        "Wrote spatial rates '%s' with %d units and %d bins",
        name,
        n_units,
        env.n_bins,
    )


def _verify_saved_environment(
    nwbfile: NWBFile, env: Environment, *, name: str, env_name: str
) -> None:
    """Refuse an explicit ``env=`` that differs from the rates' saved environment."""
    from neurospatial._exceptions import IncompatibleEnvironmentError, _format_error
    from neurospatial.environment import _environment_differences
    from neurospatial.io.nwb._environment import read_environment

    if env_name not in nwbfile.scratch:
        raise ValueError(
            _format_error(
                f"read_spatial_rates: cannot verify env= against spatial rates "
                f"'{name}': the file stores no environment for them "
                f"('{env_name}' is missing).",
                why="equal bin counts do not show that env= has the bins, units, "
                "frame and connectivity the rates were estimated on",
                fix="rewrite them with write_spatial_rates, which saves the "
                "environment with the rates",
            )
        )
    saved = read_environment(nwbfile, name=env_name)
    bins, topology = _environment_differences(saved, env)
    mismatch = bins or topology
    if mismatch is not None:
        raise IncompatibleEnvironmentError(
            f"read_spatial_rates: env= differs from the environment saved with "
            f"spatial rates '{name}' in {mismatch}.",
            fix="omit env= to use the environment saved with these rates, or "
            "pass that same environment",
            first=saved,
            second=env,
        )


def read_spatial_rates(
    nwbfile: NWBFile,
    *,
    name: str = DEFAULT_SPATIAL_RATES_NAME,
    env: Environment | None = None,
) -> SpatialRatesResult:
    """
    Read a population :class:`SpatialRatesResult` written by ``write_spatial_rates``.

    The inverse of :func:`write_spatial_rates`. Reconstructs the per-unit
    firing-rate maps, shared occupancy, ``unit_ids``, optional ``unit_table``,
    ``method`` and ``bandwidth``. Schema 2.3 also restores generated-label
    provenance and ``spike_window``; older files retain explicit-label pairing
    and assumed coverage because they did not store this provenance.

    Parameters
    ----------
    nwbfile : NWBFile
        The NWB file to read from.
    name : str, default "spatial_rates"
        Name of the spatial-rates DynamicTable in ``analysis/``.
    env : Environment, optional
        Environment to attach to the result. When omitted, the result gets the
        environment :func:`write_spatial_rates` persisted under
        ``f"{name}_environment"``, which round-trips with its connectivity
        edges and geometry **intact**, so graph operations (``neighbors``,
        ``path_between``, ...) work on the restored env. When given, ``env=``
        is first checked against the stored ``firing_rates`` bin count and then
        against that persisted environment: it must have the same bins
        (coordinates, order, measures and cell shapes, including polygon
        cells and which bin owns each boundary, and, for tracks, the same
        segments, spacing and linear bin edges), coordinate system, units,
        frame, connectivity, smoothing conductances between neighboring bins,
        arena and walls. The result then holds this ``env`` object.
        There is no connectivity-less fabrication from bin centers.

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
        ``n_bins``; if no persisted environment is in the file (whether or not
        ``env=`` is given, since an explicit ``env=`` cannot be verified
        without it); or if the obtained environment's ``n_bins`` does not match
        the stored ``firing_rates`` (a mismatched or stale ``env=``).
    IncompatibleEnvironmentError
        If ``env=`` has the right bin count but differs from the persisted
        environment in bins, units, frame or topology.
    ImportError
        If pynwb is not installed.

    Examples
    --------
    >>> from pynwb import NWBHDF5IO  # doctest: +SKIP
    >>> with NWBHDF5IO("session.nwb", "r") as io:  # doctest: +SKIP
    ...     nwbfile = io.read()
    ...     rates = read_spatial_rates(nwbfile, name="ca1_place_fields")
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

    # Enforce the schema MAJOR version. A minor bump only adds optional fields, so
    # any 2.x table reads (2.0 and 2.1 are mutually readable); a missing version,
    # a 1.x table (pre-0.9 "smoothing_method" layout), or an incompatible future
    # major is rejected -- the clean break, no back-compat shim. Keyed on the
    # major (not the exact version) so a later minor bump still reads this one.
    schema_version = str(meta.get("schema_version", ""))
    if schema_version.split(".")[0] != SPATIAL_RATES_SCHEMA_MAJOR:
        raise ValueError(
            f"Spatial rates '{name}' has schema_version={schema_version!r}, which "
            f"this reader (schema {SPATIAL_RATES_SCHEMA_VERSION}) does not support. "
            f"Only schema major {SPATIAL_RATES_SCHEMA_MAJOR}.x tables read; older "
            f"layouts (e.g. the pre-0.9 'smoothing_method' schema) are a clean "
            f"break with no back-compatibility shim. Recompute and re-write the "
            f"spatial rates with this version."
        )

    method = meta["method"]
    bandwidth = meta["bandwidth"]
    unit_table_columns = meta.get("unit_table_columns", [])
    occupancy_name = meta.get("occupancy_name", f"{name}_occupancy")
    meta_n_bins = int(meta["n_bins"])

    unit_ids_generated = meta.get("unit_ids_generated", False)
    if not isinstance(unit_ids_generated, bool):
        raise ValueError(
            f"Spatial rates '{name}' has invalid unit_ids_generated metadata; "
            "expected a boolean to distinguish generated from explicit labels."
        )
    spike_window = as_intervals(meta.get("spike_window"), name=f"{name}.spike_window")
    if spike_window is not None:
        spike_window.setflags(write=False)

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

    # The rates' environment is the one write_spatial_rates persisted under
    # f"{name}_environment" (full connectivity + geometry). An explicit env= is
    # attached only after it is verified against that saved environment, so a
    # same-size env with other units, frame, bins or topology cannot relabel
    # the maps. No connectivity-less fabrication from bin centers.
    from neurospatial.io.nwb._environment import read_environment

    env_name = f"{name}_environment"
    verify_env = env is not None
    if env is None:
        if env_name not in nwbfile.scratch:
            raise ValueError(
                f"Spatial rates '{name}' has no attached environment: no "
                f"persisted environment '{env_name}' is in the file. Rewrite the "
                f"rates with write_spatial_rates, which saves their environment."
            )
        env = read_environment(nwbfile, name=env_name)

    # The obtained env must match the stored rates' bin count.
    n_bins = int(firing_rates.shape[1])
    if env.n_bins != n_bins:
        raise ValueError(
            f"Environment n_bins ({env.n_bins}) does not match the stored "
            f"firing_rates bin count ({n_bins}) for spatial rates '{name}'. This "
            f"usually means a mismatched or stale env= was passed; pass the "
            f"Environment used to compute these rates."
        )
    if verify_env:
        _verify_saved_environment(nwbfile, env, name=name, env_name=env_name)

    # GAM (method="glm") diagnostics: reconstruct from the JSON blob + the
    # per-unit columns, preserving shapes. Absent (ratio methods) -> all None,
    # including ``pooled`` (meaningless for a ratio result). coefficients is stored
    # transposed as (n_units, rank); transpose back to (rank, n_units). The
    # reconstructed fields must satisfy the SpatialRatesResult None-iff-glm
    # invariant (checked at construction below) -- a wrong shape or a missing field
    # raises there, so a corrupt glm record fails loudly.
    gam_fields: dict[str, object | None] = {
        "coefficients": None,
        "penalty": None,
        "penalty_weights": None,
        "rank": None,
        "deviance": None,
        "converged": None,
        "n_iter": None,
        "reml_objective": None,
        "reml_at_boundary": None,
        "penalty_selected_by_reml": None,
        "pooled": None,
    }
    if method == "glm":
        gam = meta.get("gam")
        if not isinstance(gam, dict):
            raise ValueError(
                f"Spatial rates '{name}' has method='glm' but no GAM metadata "
                f"blob; the file may be corrupt or written by an older schema."
            )
        # Backward-compat (schema 2.1): a missing ``pooled`` key defaults to True
        # for a glm file (the only glm behavior that schema had); ``per_unit`` and
        # ``reml_at_boundary`` default to False / None.
        per_unit = bool(gam.get("per_unit", False))
        gam_fields = {
            "coefficients": np.asarray(table[COL_COEFFICIENTS][:], dtype=np.float64).T,
            "penalty_weights": np.asarray(gam["penalty_weights"], dtype=np.float64),
            "rank": int(gam["rank"]),
            "deviance": np.asarray(table[COL_DEVIANCE][:], dtype=np.float64),
            "converged": bool(gam["converged"]),
            "n_iter": int(gam["n_iter"]),
            "pooled": bool(gam.get("pooled", True)),
        }
        if per_unit:
            # Per-unit-lambda vectors live in table columns (schema 2.2). Verify
            # they are all present first, so a truncated / externally-edited 2.2
            # record fails with a clear schema-aware error (naming the table and
            # the missing column) rather than a raw hdmf KeyError -- matching the
            # missing-GAM-blob handling above.
            missing_cols = [
                c
                for c in (
                    COL_PENALTY,
                    COL_REML_OBJECTIVE,
                    COL_REML_AT_BOUNDARY,
                    COL_PENALTY_SELECTED,
                )
                if c not in table.colnames
            ]
            if missing_cols:
                raise ValueError(
                    f"Spatial rates '{name}' is a per-unit glm record (per_unit=True) "
                    f"but is missing the per-unit column(s) {missing_cols}; the file "
                    "may be corrupt or truncated."
                )
            gam_fields["penalty"] = np.asarray(table[COL_PENALTY][:], dtype=np.float64)
            gam_fields["reml_objective"] = np.asarray(
                table[COL_REML_OBJECTIVE][:], dtype=np.float64
            )
            gam_fields["reml_at_boundary"] = np.asarray(
                table[COL_REML_AT_BOUNDARY][:], dtype=bool
            )
            gam_fields["penalty_selected_by_reml"] = np.asarray(
                table[COL_PENALTY_SELECTED][:], dtype=bool
            )
        else:
            penalty = gam.get("penalty")
            reml_objective = gam.get("reml_objective")
            reml_at_boundary = gam.get("reml_at_boundary")
            gam_fields["penalty"] = None if penalty is None else float(penalty)
            gam_fields["reml_objective"] = (
                None if reml_objective is None else float(reml_objective)
            )
            gam_fields["reml_at_boundary"] = (
                None if reml_at_boundary is None else bool(reml_at_boundary)
            )
            gam_fields["penalty_selected_by_reml"] = None

    return SpatialRatesResult(
        firing_rates=firing_rates,
        occupancy=occupancy,
        env=env,
        method=method,
        bandwidth=bandwidth,
        unit_ids=unit_ids,
        unit_table=unit_table,
        _unit_ids_generated=unit_ids_generated,
        spike_window=spike_window,
        min_occupancy=_recorded_min_occupancy(meta, name),
        n_spikes_outside_intervals=(
            None
            if meta.get("n_spikes_outside_intervals") is None
            else np.asarray(meta["n_spikes_outside_intervals"], dtype=np.int64)
        ),
        analysis_settings=meta.get("analysis_settings"),
        **gam_fields,  # type: ignore[arg-type]
    )


def _recorded_min_occupancy(meta: dict[str, Any], name: str) -> float:
    """The occupancy threshold in seconds that defines the maps' support."""
    if "min_occupancy" not in meta:
        raise ValueError(
            f"Spatial rates '{name}' records no min_occupancy (schema "
            f"{meta.get('schema_version')!r}).\n"
            "Why: spatial_support is derived from the occupancy threshold, and "
            "files before schema 2.5 did not store it.\n"
            "Fix: recompute the rates and write them again with write_spatial_rates."
        )
    return float(meta["min_occupancy"])
