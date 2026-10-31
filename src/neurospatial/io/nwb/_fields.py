"""
Spatial firing-rate maps in NWB files.

``write_spatial_rates`` stores a population ``SpatialRatesResult`` -- rate
maps, occupancy and the environment they were estimated on -- in the NWB
file's top-level ``/analysis`` group, and ``read_spatial_rates`` restores it.

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
    _refuse_in_place_replacement,
    _require_pynwb,
    logger,
)

if TYPE_CHECKING:
    from hdmf.common import DynamicTable, DynamicTableRegion
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
SPATIAL_RATES_SCHEMA_VERSION: str = "3.0"

# Column names within the spatial-rates DynamicTable
COL_UNIT_ID: str = "unit_id"
COL_FIRING_RATE: str = "firing_rate"
# Link from each row to its row in nwbfile.units, written when every unit_id is
# a units-table id.
COL_UNITS: str = "units"
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

# Processing module where records before schema 3.0 were written; read only to
# refuse them.
_EARLIER_MODULE: str = "analysis"


# =============================================================================
# Population spatial-rates round-trip (SpatialRatesResult <-> NWB)
# =============================================================================


def write_spatial_rates(
    nwbfile: NWBFile,
    result: SpatialRatesResult,
    *,
    name: str = DEFAULT_SPATIAL_RATES_NAME,
    environment_name: str | None = None,
    overwrite: bool = False,
) -> None:
    """
    Write a population :class:`SpatialRatesResult` to NWB with a unit axis.

    Stores the whole population -- per-unit firing-rate maps, shared occupancy,
    unit identities, and optional per-unit metadata -- so that
    :func:`read_spatial_rates` reconstructs an equal ``SpatialRatesResult``.

    Everything is written to the file's top-level ``/analysis`` group, which
    NWB provides for shareable analysis results. The rates are a
    :class:`~hdmf.common.DynamicTable` (one row per unit) with a ``unit_id``
    column, a 2-D ``firing_rate`` column of shape ``(n_units, n_bins)`` on the
    bin axis, and one column per ``unit_table`` field. The table description
    is a JSON blob holding ``schema_version``, ``method``, ``bandwidth``,
    ``n_bins``, ``n_units``, the ``unit_table`` column names
    (``unit_table_columns``), ``occupancy_name``, ``environment_name``,
    ``unit_ids_generated``, ``min_occupancy`` (seconds),
    ``n_spikes_outside_intervals``, ``analysis_settings``, ``spike_window``
    and, for ``method="glm"``, a ``gam`` block (below). Occupancy is stored once
    (it is shared across units) as a companion ``DynamicTable`` named
    ``f"{name}_occupancy"`` with one row per bin and an ``occupancy`` column in
    seconds. It is not a ``TimeSeries``: its axis is space, not time.

    When the file has a units table (``nwbfile.units``) and every ``unit_id``
    is one of its ids, the rates also get a ``units`` column, a
    ``DynamicTableRegion`` pointing each row at its unit, so NWB tools can
    join a rate map to that unit's spike times, electrodes and quality
    metrics. Otherwise (no units table, generated ids, or any id missing from
    it) only ``unit_id`` is written. :func:`read_spatial_rates` returns the
    same result either way.

    For ``method="glm"`` results the GAM diagnostics are persisted alongside the
    rates: the scalars ``penalty`` / ``rank`` / ``n_iter`` / ``converged`` /
    ``reml_objective`` and the ``(rank,)`` ``penalty_weights`` vector go into the
    JSON description blob (``penalty`` / ``reml_objective`` as JSON ``null`` when
    ``None``), while ``deviance`` ``(n_units,)`` and ``coefficients`` (stored
    transposed as ``(n_units, rank)`` per-unit rows) become extra table columns.
    Ratio-method results carry none of these; they read back with the GAM fields
    ``None``. ``bandwidth`` is stored nullable (``None`` for glm).

    The full :class:`~neurospatial.Environment` the rates were estimated on is
    stored once, as a :func:`~neurospatial.io.nwb.write_environment` record
    named ``environment_name``; the rates record that name and
    :func:`read_spatial_rates` follows it. Rates estimated on the same
    environment can share one record: when a record of that name already
    stores exactly this environment (same bins, arena, regions, landmarks,
    units, frame and name), it is reused rather than written again.

    The write is **atomic**: all name collisions and all validation are
    resolved *before* the first object is added, so a duplicate without
    ``overwrite`` raises before any mutation. If a later add still fails, the
    already-added objects are removed, any replaced record is restored, and a
    ``ValueError`` says whether that cleanup succeeded.

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
        Name for the spatial-rates DynamicTable in ``nwbfile.analysis``.
    environment_name : str, optional
        Name of the environment record in ``nwbfile.analysis``. Default
        ``f"{name}_environment"``. Pass the same name for rates estimated on
        one environment to store it once; pass the name of a record saved
        with :func:`~neurospatial.io.nwb.write_environment` to link to it.
    overwrite : bool, default False
        If True, replace an existing table ``name`` and its
        ``f"{name}_occupancy"`` companion. If False, raise ``ValueError`` when
        either name is already present. An environment record is never
        replaced while other spatial rates use it; with ``overwrite=True`` a
        record used only by the rates being replaced is replaced too.
        Replacing works on an in-memory ``NWBFile`` or one read with mode
        ``"r"`` and then exported to a new path with
        ``NWBHDF5IO(new_path, "w").export(src_io=io, nwbfile=nwbfile)``; pynwb
        cannot delete objects already written to a file opened with ``"r+"``
        or ``"a"``.

    Raises
    ------
    ValueError
        If a table ``name`` (or its occupancy companion) exists and
        ``overwrite=False``; if ``environment_name`` names a record that
        stores a different environment (and is not replaceable as described
        under ``overwrite``) or an ``/analysis`` object that is not an
        environment record; if ``name``, its occupancy companion and
        ``environment_name`` are not three distinct names, or ``name`` or the
        companion is an environment record; if a ``unit_table`` column is
        named ``unit_id``, ``firing_rate`` or ``units`` (reserved -- plus
        ``deviance`` / ``coefficients`` for a glm result, whose GAM columns
        share the table);
        or if a companion add fails after a partial write (the message says
        what could not be cleaned up).
    NotImplementedError
        If ``overwrite=True`` would replace a record in a file opened with
        ``"r+"`` or ``"a"``.
    ImportError
        If pynwb is not installed.

    Notes
    -----
    This function mutates only ``nwbfile``. The ``result`` object and its
    arrays are never modified -- ``firing_rates`` and ``occupancy`` are copied
    before being handed to the NWB containers, so the written data never
    aliases the result's memory.

    Examples
    --------
    >>> from pynwb import NWBHDF5IO  # doctest: +SKIP
    >>> from neurospatial.encoding import compute_spatial_rates  # doctest: +SKIP
    >>> rates = compute_spatial_rates(
    ...     env, spike_times, times, positions
    ... )  # doctest: +SKIP
    >>> with NWBHDF5IO("session.nwb", "r+") as io:  # doctest: +SKIP
    ...     nwbfile = io.read()
    ...     write_spatial_rates(
    ...         nwbfile, rates, name="ca1_place_fields", environment_name="open_field"
    ...     )
    ...     io.write(nwbfile)
    """
    _require_pynwb()
    from hdmf.common import DynamicTable, VectorData

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
    env_name = f"{name}_environment" if environment_name is None else environment_name

    # --- Preflight (BEFORE any mutation): reserved-name validation ----------
    # Shapes need no check here: SpatialRatesResult refuses firing_rates or
    # occupancy that do not match its env's bins when it is constructed.

    # `unit_id` / `firing_rate` / `units` are reserved for the table's own
    # columns; a unit_table column with one of these names would collide inside
    # the DynamicTable.
    # A glm result always writes `deviance` / `coefficients`, so those are always
    # reserved; the per-unit-lambda columns (`penalty` / `reml_objective` /
    # `reml_at_boundary` / `penalty_selected_by_reml`) are written -- and thus
    # reserved -- only for a pooled=False per-unit result. Raise a clear
    # param-named error instead of a low-level hdmf error.
    reserved_names: tuple[str, ...] = (COL_UNIT_ID, COL_FIRING_RATE, COL_UNITS)
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

    # Resolve every name collision -- the table, its companion occupancy and
    # the environment record -- before the first add, so a duplicate without
    # overwrite raises before any mutation. For overwrite=True the existing
    # objects are captured here but deleted only in the try-block below, after
    # the replacements are built, so a failed write can restore them.
    old_objects, env_record = _plan_spatial_rates_write(
        nwbfile, env, name, occupancy_name, env_name, overwrite=overwrite
    )
    if old_objects:
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
        "environment_name": env_name,
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
            *_units_link(nwbfile, unit_ids, generated=result._unit_ids_generated),
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
    # occupancy and, unless an identical record is reused, the environment. If
    # any step fails, the newly added objects are rolled back and the originals
    # restored.
    new_objects = [table, occupancy_table]
    if env_record is not None:
        new_objects.append(env_record)
    added: list[str] = []
    removed: list[tuple[str, object]] = []
    try:
        for obj_name, obj in old_objects:
            del nwbfile.analysis[obj_name]
            removed.append((obj_name, obj))
        for obj in new_objects:
            nwbfile.add_analysis(obj)
            added.append(obj.name)
    except Exception as exc:
        not_removed = []
        for obj_name in reversed(added):
            try:
                del nwbfile.analysis[obj_name]
            except Exception:
                not_removed.append(obj_name)
        not_restored = []
        for obj_name, obj in reversed(removed):
            try:
                nwbfile.add_analysis(obj)
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


def _units_link(
    nwbfile: NWBFile, unit_ids: np.ndarray, *, generated: bool
) -> list[DynamicTableRegion]:
    """A ``units`` column pointing each rate map at its ``nwbfile.units`` row.

    Written only when the file has a units table and every unit_id is one of
    its ids; generated ids (``0..n-1`` positions) are not identities and never
    link. Otherwise the rates keep ``unit_id`` alone.
    """
    from hdmf.common import DynamicTableRegion

    units = nwbfile.units
    if units is None or generated or unit_ids.dtype.kind not in "iu":
        return []
    row_of = {int(unit_id): row for row, unit_id in enumerate(units.id[:])}
    rows = [row_of.get(int(unit_id)) for unit_id in unit_ids]
    if any(row is None for row in rows):
        return []
    return [
        DynamicTableRegion(
            name=COL_UNITS,
            description="Row of each unit in the NWB units table.",
            data=rows,
            table=units,
        )
    ]


def _plan_spatial_rates_write(
    nwbfile: NWBFile,
    env: Environment,
    name: str,
    occupancy_name: str,
    env_name: str,
    *,
    overwrite: bool,
) -> tuple[list[tuple[str, object]], DynamicTable | None]:
    """Check every name a spatial-rates write uses, before any mutation.

    Returns
    -------
    old_objects : list of (str, object)
        ``/analysis`` objects to replace, by name.
    env_record : DynamicTable or None
        The environment record to add; None when an identical record named
        ``env_name`` is reused.
    """
    from neurospatial.io.nwb._environment import (
        _environment_record,
        _is_environment_record,
        _stored_environment_matches,
    )

    analysis = nwbfile.analysis
    if len({name, occupancy_name, env_name}) != 3:
        raise ValueError(
            f"write_spatial_rates: name={name!r}, its occupancy companion "
            f"{occupancy_name!r} and environment_name={env_name!r} must be three "
            "different /analysis names. Choose another environment_name."
        )
    for own in (name, occupancy_name):
        if own in analysis and _is_environment_record(analysis[own]):
            raise ValueError(
                f"write_spatial_rates: /analysis/{own} is an environment record, "
                "which other spatial rates may use. Choose another name= for the "
                "rates."
            )
    existing = [n for n in (name, occupancy_name) if n in analysis]
    if existing and not overwrite:
        raise ValueError(
            f"Spatial rates '{name}' already exists (found: {existing}). "
            f"Use overwrite=True to replace it and its companions."
        )
    old_objects = [(n, analysis[n]) for n in existing]

    if env_name not in analysis:
        env_record = _environment_record(env, env_name)
    elif not _is_environment_record(analysis[env_name]):
        raise ValueError(
            _format_error(
                f"write_spatial_rates: /analysis/{env_name} is a "
                f"{type(analysis[env_name]).__name__}, not an environment record.",
                why="Why: the rates' environment would replace another analysis "
                "result of that name.",
                fix="pass environment_name= another name",
            )
        )
    elif _stored_environment_matches(nwbfile, env_name, env):
        env_record = None
    else:
        users = _spatial_rates_using(nwbfile, env_name)
        if not (overwrite and users == [name]):
            raise ValueError(
                _format_error(
                    f"write_spatial_rates: /analysis/{env_name} stores a different "
                    "environment from the one these rates were estimated on"
                    + (f" (used by spatial rates {users})." if users else "."),
                    why="Why: rates are read with the environment record they "
                    "name, so replacing it would relabel the bins of the rates "
                    "that use it.",
                    fix="pass environment_name= a new name to store this "
                    "environment separately",
                )
            )
        env_record = _environment_record(env, env_name)
        old_objects.append((env_name, analysis[env_name]))
    if old_objects:
        _refuse_in_place_replacement(nwbfile, f"spatial rates '{name}'")
    return old_objects, env_record


def _spatial_rates_using(nwbfile: NWBFile, env_name: str) -> list[str]:
    """Names of the spatial-rates tables in ``/analysis`` that use ``env_name``."""
    users = []
    for table_name, table in nwbfile.analysis.items():
        if COL_FIRING_RATE not in getattr(table, "colnames", ()):
            continue
        try:
            meta = json.loads(table.description)
        except (TypeError, ValueError):
            continue
        if isinstance(meta, dict) and meta.get("environment_name") == env_name:
            users.append(table_name)
    return sorted(users)


def _abort_message(
    name: str,
    exc: Exception,
    removed: list[tuple[str, object]],
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

    if env_name not in nwbfile.analysis:
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
        Name of the spatial-rates DynamicTable in ``nwbfile.analysis``.
    env : Environment, optional
        Environment to attach to the result. When omitted, the result gets the
        environment record the rates name (``environment_name`` in their
        metadata, as :func:`write_spatial_rates` stored it), which round-trips with its connectivity
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
        If the named table or its companion occupancy is missing from
        ``nwbfile.analysis``.
    ValueError
        If ``name`` points at a table not written by ``write_spatial_rates``; if
        the record was written with a schema version other than
        ``SPATIAL_RATES_SCHEMA_VERSION``, including rates an earlier version
        wrote to the processing module ``analysis`` (recompute and rewrite the
        rates); if
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

    analysis = nwbfile.analysis
    if name not in analysis:
        earlier = nwbfile.processing.get(_EARLIER_MODULE)
        if earlier is not None and name in earlier.data_interfaces:
            raise ValueError(
                _format_error(
                    f"Spatial rates '{name}' are in the processing module "
                    f"'{_EARLIER_MODULE}', where neurospatial before spatial-rates "
                    f"schema {SPATIAL_RATES_SCHEMA_VERSION} wrote them; this "
                    "version reads only rates in /analysis.",
                    why="Why: earlier records store a different layout and are "
                    "not converted.",
                    fix=_RECREATE_FIX,
                )
            )
        available = _spatial_rates_names(nwbfile)
        raise KeyError(
            f"Spatial rates '{name}' not found in /analysis. Saved spatial "
            f"rates: {available}"
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

    if occupancy_name not in analysis:
        raise KeyError(
            f"Occupancy '{occupancy_name}' for spatial rates '{name}' not found "
            "in /analysis."
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

    # The rates' environment is the record they name (full connectivity +
    # geometry). An explicit env= is
    # attached only after it is verified against that saved environment, so a
    # same-size env with other units, frame, bins or topology cannot relabel
    # the maps. No connectivity-less fabrication from bin centers.
    from neurospatial.io.nwb._environment import read_environment

    env_name = _required_key(meta, "environment_name", name)
    verify_env = env is not None
    if env is None:
        if env_name not in analysis:
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


def _spatial_rates_names(nwbfile: NWBFile) -> list[str]:
    """Names of the spatial-rates tables in ``nwbfile.analysis``, sorted."""
    return sorted(
        name
        for name, table in nwbfile.analysis.items()
        if COL_FIRING_RATE in getattr(table, "colnames", ())
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
                why="other versions store different data (records before 3.0 "
                "are in a processing module and keep a private copy of their "
                "environment, records before 2.6 store occupancy as a time "
                "series, and records before 2.5 lack the occupancy threshold "
                "that defines spatial_support), and reading them would mean "
                "guessing",
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
