"""
Spatial firing-rate maps in NWB files.

``write_spatial_rates`` stores a population ``SpatialRatesResult`` -- rate
maps, occupancy and the environment they were estimated on -- and
``read_spatial_rates`` restores it.

Schema
------
The metadata is a JSON blob in the table description, versioned by
``SPATIAL_RATES_SCHEMA_VERSION``. ``read_spatial_rates`` reads only that exact
version and refuses any other with instructions to recompute and rewrite the
rates; it never fills in a metadata key the writer always stores.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np

from neurospatial._exceptions import _format_error
from neurospatial._intervals import as_intervals
from neurospatial.io.nwb._core import (
    _get_or_create_processing_module,
    _refuse_in_place_replacement,
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

# Schema version of the spatial-rates metadata blob; the reader accepts only
# this version.
SPATIAL_RATES_SCHEMA_VERSION: str = "2.6"

# Column names within the spatial-rates DynamicTable
COL_UNIT_ID: str = "unit_id"
COL_FIRING_RATE: str = "firing_rate"
# Per-unit GAM (method="glm") columns; absent for ratio methods.
COL_DEVIANCE: str = "deviance"
COL_COEFFICIENTS: str = "coefficients"
# Per-unit-lambda (pooled=False) columns; present only when penalty/reml_objective
# are per-unit vectors.
COL_PENALTY: str = "penalty"
COL_REML_OBJECTIVE: str = "reml_objective"
COL_REML_AT_BOUNDARY: str = "reml_at_boundary"
COL_PENALTY_SELECTED: str = "penalty_selected_by_reml"
# Column of the occupancy companion table (one row per bin).
COL_OCCUPANCY: str = "occupancy"

# Default processing module for analysis results
DEFAULT_ANALYSIS_MODULE: str = "analysis"


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
    the processing module ``nwbfile.processing["analysis"]`` (not NWB's
    top-level ``/analysis`` group), with a ``unit_id`` column, a 2-D
    ``firing_rate`` column of shape ``(n_units, n_bins)`` on the bin axis, and
    one column per ``unit_table`` field. The table description is a JSON blob
    holding ``schema_version``, ``method``, ``bandwidth``, ``n_bins``,
    ``n_units``, the ``unit_table`` column names (``unit_table_columns``),
    ``occupancy_name``, ``unit_ids_generated``, ``min_occupancy`` (seconds),
    ``n_spikes_outside_intervals``, ``analysis_settings``, ``spike_window``
    and, for ``method="glm"``, a ``gam`` block (below). Occupancy is stored once
    (it is shared across units) as a companion ``DynamicTable`` named
    ``f"{name}_occupancy"`` with one row per bin and an ``occupancy`` column in
    seconds. It is not a ``TimeSeries``: its axis is space, not time.

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
    still fails, the already-added objects are removed, any replaced record is
    restored, and a ``ValueError`` says whether that cleanup succeeded.

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
        Name for the spatial-rates DynamicTable in
        ``nwbfile.processing["analysis"]``.
    overwrite : bool, default False
        If True, replace an existing container and *all* its companions (the
        ``f"{name}_occupancy"`` table and the ``f"{name}_environment"``
        environment). If False, raise ``ValueError`` when any of those names is
        already present. Replacing works on an in-memory ``NWBFile`` or one read
        with mode ``"r"`` and then exported to a new path with
        ``NWBHDF5IO(new_path, "w").export(src_io=io, nwbfile=nwbfile)``; pynwb
        cannot delete objects already written to a file opened with ``"r+"`` or
        ``"a"``.

    Raises
    ------
    ValueError
        If a container named ``name`` (or one of its companions) exists and
        ``overwrite=False``; if a ``unit_table`` column is named ``unit_id`` or
        ``firing_rate`` (reserved -- plus ``deviance`` / ``coefficients`` for a
        glm result, whose GAM columns share the table); or if a companion add
        fails after a partial write (the message says what could not be cleaned
        up).
    NotImplementedError
        If ``overwrite=True`` would replace a record in a file opened with
        ``"r+"`` or ``"a"``.
    ImportError
        If pynwb is not installed.

    Notes
    -----
    This function mutates only ``nwbfile`` (adds the DynamicTable and the
    companion occupancy table to the processing module ``analysis``, and the
    persisted environment to ``scratch``). The ``result`` object and its arrays
    are never modified -- ``firing_rates`` and ``occupancy`` are copied before
    being handed to the NWB containers, so the written data never aliases the
    result's memory.

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

    from neurospatial.io.nwb._environment import write_environment

    is_glm = result.method == "glm"
    # The per-unit-lambda columns exist only for a pooled=False per-unit result
    # (keyed off the provenance mask, exactly as the writer decides below), so
    # they are reserved only then -- a pooled=True/fixed-penalty glm write does
    # not emit them and must not reject a unit_table column named e.g. "penalty".
    is_per_unit = is_glm and result.penalty_selected_by_reml is not None

    env = result.env
    # Copy the arrays (np.array, not a no-copy np.asarray) so the
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

    # `unit_id` / `firing_rate` are reserved for the table's own columns; a
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

    # Resolve every name collision -- the table, its companion occupancy and
    # the persisted environment -- before the first add, so a duplicate without
    # overwrite raises before any mutation. For overwrite=True the existing
    # objects are captured here but deleted only in the try-block below, after
    # the replacements are built, so a failed write can restore them.
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
        _refuse_in_place_replacement(nwbfile, f"spatial rates '{name}'")
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
        # a per-unit column instead (per_unit=True).
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

    # Occupancy is shared across units: store it once, one row per bin. A
    # TimeSeries would claim a time axis, and NWB tools would read the bins as
    # samples at t = 0.
    occupancy_table = DynamicTable(
        name=occupancy_name,
        description=(
            f"Occupancy shared across units for spatial rates '{name}'. Rows are "
            f"the {env.n_bins} bins of the environment saved as '{env_name}'."
        ),
        columns=[
            VectorData(
                name=COL_OCCUPANCY,
                description="Time spent in each bin, in seconds",
                data=occupancy,
            )
        ],
    )

    # Remove any originals (overwrite=True), then add the table, its companion
    # occupancy and the full environment (with connectivity and geometry). If
    # any step fails, the newly added objects are rolled back and the originals
    # restored.
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
        analysis.add(occupancy_table)
        added.append(("analysis", occupancy_name))
        write_environment(nwbfile, env, name=env_name)
        added.append(("scratch", env_name))
    except Exception as exc:
        not_removed = []
        for namespace, obj_name in reversed(added):
            try:
                if namespace == "analysis":
                    del analysis.data_interfaces[obj_name]
                else:
                    del nwbfile.scratch[obj_name]
            except Exception:
                not_removed.append(obj_name)
        not_restored = []
        for namespace, obj_name, obj in reversed(removed):
            try:
                if namespace == "analysis":
                    analysis.add(obj)
                else:
                    nwbfile.add_scratch(obj)
            except Exception:
                not_restored.append(obj_name)
        raise ValueError(
            _abort_message(name, exc, removed, not_removed, not_restored)
        ) from exc

    logger.debug(
        "Wrote spatial rates '%s' with %d units and %d bins",
        name,
        n_units,
        env.n_bins,
    )


def _abort_message(
    name: str,
    exc: Exception,
    removed: list[tuple[str, str, object]],
    not_removed: list[str],
    not_restored: list[str],
) -> str:
    """Describe what a failed spatial-rates write left in the NWBFile."""
    what = f"Aborted writing spatial rates '{name}': {exc!r}."
    if not (not_removed or not_restored):
        state = "The partly written objects were removed"
        if removed:
            state += " and the pre-existing record was restored"
        return f"{what} {state}."
    problems = []
    if not_removed:
        problems.append(f"partly written objects could not be removed: {not_removed}")
    if not_restored:
        problems.append(f"pre-existing objects could not be restored: {not_restored}")
    return _format_error(
        f"{what} The NWBFile in memory is inconsistent; " + "; ".join(problems) + ".",
        fix="Do not write this NWBFile. Rebuild it, or reopen it from disk where "
        "the previous record is unchanged, and write the rates again.",
    )


def _verify_saved_environment(
    nwbfile: NWBFile, env: Environment, *, name: str, env_name: str
) -> None:
    """Refuse an explicit ``env=`` that differs from the rates' saved environment."""
    from neurospatial._exceptions import IncompatibleEnvironmentError
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
    firing-rate maps, shared occupancy, ``unit_ids`` (and whether they were
    generated), optional ``unit_table``, ``method``, ``bandwidth``,
    ``min_occupancy`` (and so ``spatial_support``), ``spike_window``, spike
    accounting, ``analysis_settings`` and, for ``method="glm"``, the GAM
    diagnostics.

    Parameters
    ----------
    nwbfile : NWBFile
        The NWB file to read from.
    name : str, default "spatial_rates"
        Name of the spatial-rates DynamicTable in
        ``nwbfile.processing["analysis"]``.
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
        If the processing module ``analysis``, the named table, or its companion
        occupancy is missing from ``nwbfile``.
    ValueError
        If ``name`` points at a table not written by ``write_spatial_rates``; if
        the record was written with a schema version other than
        ``SPATIAL_RATES_SCHEMA_VERSION`` (recompute and rewrite the rates); if
        a metadata key the writer always stores is missing or the recorded unit
        count does not match the table; if the companion occupancy length does not match the table's recorded
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
            f"Spatial rates '{name}' not found in processing module "
            f"'{DEFAULT_ANALYSIS_MODULE}'. Available: {available}"
        )

    table = analysis[name]

    # A table not written by write_spatial_rates has a plain-text description.
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

    _require_current_schema(meta, name)
    method = meta["method"]
    meta_n_bins = int(meta["n_bins"])
    bandwidth = _required_key(meta, "bandwidth", name)
    n_units = int(_required_key(meta, "n_units", name))
    unit_table_columns = _required_key(meta, "unit_table_columns", name)
    occupancy_name = _required_key(meta, "occupancy_name", name)
    min_occupancy = float(_required_key(meta, "min_occupancy", name))
    n_spikes_outside_intervals = _required_key(meta, "n_spikes_outside_intervals", name)
    analysis_settings = _required_key(meta, "analysis_settings", name)

    unit_ids_generated = _required_key(meta, "unit_ids_generated", name)
    if not isinstance(unit_ids_generated, bool):
        raise ValueError(
            f"Spatial rates '{name}' has invalid unit_ids_generated metadata; "
            "expected a boolean to distinguish generated from explicit labels."
        )
    spike_window = as_intervals(
        _required_key(meta, "spike_window", name), name=f"{name}.spike_window"
    )
    if spike_window is not None:
        spike_window.setflags(write=False)

    firing_rates = np.asarray(table[COL_FIRING_RATE][:])
    unit_ids = np.asarray(table[COL_UNIT_ID][:])
    if unit_ids.shape[0] != n_units:
        raise ValueError(
            _format_error(
                f"Corrupt spatial-rates record '{name}': it records "
                f"n_units={n_units} but stores {unit_ids.shape[0]} rows.",
                why="the table was truncated or edited after it was written",
                fix=_RECREATE_FIX,
            )
        )

    unit_table = None
    if unit_table_columns:
        import pandas as pd

        unit_table = pd.DataFrame(
            {col: np.asarray(table[col][:]) for col in unit_table_columns}
        )

    if occupancy_name not in analysis.data_interfaces:
        raise KeyError(
            f"Occupancy '{occupancy_name}' for spatial rates '{name}' not found "
            f"in processing module '{DEFAULT_ANALYSIS_MODULE}'."
        )
    occupancy_record = analysis[occupancy_name]
    if COL_OCCUPANCY not in getattr(occupancy_record, "colnames", ()):
        raise ValueError(
            f"Occupancy companion '{occupancy_name}' for spatial rates '{name}' is "
            f"not a table with an '{COL_OCCUPANCY}' column; the file may be "
            "corrupt or the name was reused for another object."
        )
    occupancy = np.asarray(occupancy_record[COL_OCCUPANCY][:], dtype=np.float64)

    # Check the companion against the table's own recorded n_bins, so a
    # wrong-length occupancy fails here rather than in a later analysis.
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
        gam = _required_key(meta, "gam", name)
        if not isinstance(gam, dict):
            raise ValueError(
                _format_error(
                    f"Corrupt spatial-rates record '{name}': the 'gam' metadata "
                    f"is {type(gam).__name__}, not a mapping.",
                    why="method='glm' records store their GAM diagnostics there",
                    fix=_RECREATE_FIX,
                )
            )

        def gam_key(key: str) -> Any:
            return _required_key(gam, key, name, prefix="gam.")

        per_unit = bool(gam_key("per_unit"))
        # The scalar slots are always written; they hold JSON null when the
        # values live in the per-unit columns instead.
        penalty = gam_key("penalty")
        reml_objective = gam_key("reml_objective")
        reml_at_boundary = gam_key("reml_at_boundary")
        gam_fields = {
            "coefficients": np.asarray(table[COL_COEFFICIENTS][:], dtype=np.float64).T,
            "penalty_weights": np.asarray(gam_key("penalty_weights"), dtype=np.float64),
            "rank": int(gam_key("rank")),
            "deviance": np.asarray(table[COL_DEVIANCE][:], dtype=np.float64),
            "converged": bool(gam_key("converged")),
            "n_iter": int(gam_key("n_iter")),
            "pooled": bool(gam_key("pooled")),
        }
        if per_unit:
            # Name a missing per-unit column instead of a raw hdmf KeyError.
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
        min_occupancy=min_occupancy,
        n_spikes_outside_intervals=(
            None
            if n_spikes_outside_intervals is None
            else np.asarray(n_spikes_outside_intervals, dtype=np.int64)
        ),
        analysis_settings=analysis_settings,
        **gam_fields,  # type: ignore[arg-type]
    )


_RECREATE_FIX = (
    "recompute the rates with this version of neurospatial and write them "
    "again with write_spatial_rates"
)


def _require_current_schema(meta: dict[str, Any], name: str) -> None:
    """Refuse a spatial-rates record written with another schema version."""
    recorded = meta.get("schema_version")
    if recorded != SPATIAL_RATES_SCHEMA_VERSION:
        raise ValueError(
            _format_error(
                f"Spatial rates '{name}' has schema_version={recorded!r}; this "
                f"version of neurospatial reads only schema "
                f"{SPATIAL_RATES_SCHEMA_VERSION}.",
                why="other versions store different data (records before 2.6 "
                "store occupancy as a time series, and records before 2.5 lack "
                "the occupancy threshold that defines spatial_support), and "
                "reading them would mean guessing",
                fix=_RECREATE_FIX,
            )
        )


def _required_key(
    meta: dict[str, Any], key: str, name: str, *, prefix: str = ""
) -> Any:
    """Return ``meta[key]``, refusing a record that lacks a key the writer stores."""
    if key not in meta:
        raise ValueError(
            _format_error(
                f"Corrupt spatial-rates record '{name}': missing metadata key "
                f"'{prefix}{key}'.",
                why="write_spatial_rates always stores this key, so the record "
                "was truncated or edited after it was written",
                fix=_RECREATE_FIX,
            )
        )
    return meta[key]
