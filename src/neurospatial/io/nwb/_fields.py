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
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from neurospatial._exceptions import _format_error
from neurospatial._intervals import as_intervals
from neurospatial._results import EMPTY_SELECTION
from neurospatial.io.files import _check_schema
from neurospatial.io.nwb._adapters import missing_error
from neurospatial.io.nwb._core import _clear_for_write, _require_pynwb, logger
from neurospatial.io.nwb._environment import (
    SPATIAL_RATES_FORMAT,
    _analysis_records,
    _earlier_location_error,
    _environment_content,
    _environment_record,
    _record_metadata,
    _require_record,
    read_environment,
)

if TYPE_CHECKING:
    from hdmf.common import DynamicTableRegion
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
    is a JSON blob holding ``format`` (``"neurospatial.SpatialRates"``, which
    identifies the record), ``schema_version``, ``method``, ``bandwidth``,
    ``n_bins``, ``n_units``, the ``unit_table`` column names
    (``unit_table_columns``), ``occupancy_name``, ``environment_name``,
    ``environment_sha256``, ``unit_ids_generated``, ``min_occupancy`` (seconds),
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
    units, frame and name), it is reused rather than written again. The
    rates also store ``environment_sha256``, a fingerprint of that record's
    content, and :func:`read_spatial_rates` refuses them if the record has
    changed since, however it was replaced or edited.

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
        record used only by the rates being replaced is replaced too. When
        the replacement names another environment, the replaced rates'
        default record ``f"{name}_environment"`` is removed if nothing else
        uses it; a record saved under any other name is kept.
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
    This function mutates only ``nwbfile``. The NWB containers hold the
    result's own ``firing_rates`` and ``occupancy``, which are read-only, so
    neither the result nor the written data can change the other.

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

    env = result.env
    unit_ids = np.asarray(result.unit_ids)
    n_units = int(unit_ids.shape[0])
    occupancy_name = f"{name}_occupancy"
    env_name = f"{name}_environment" if environment_name is None else environment_name

    # GAM (method="glm") diagnostics: scalars + the (rank,) penalty_weights go
    # into the JSON blob; the per-unit deviance and coefficients become table
    # columns. Under pooled=False (per-unit lambda, marked by the provenance
    # mask) the penalty / reml_objective / reml_at_boundary vectors and the
    # penalty_selected_by_reml mask also become per-unit columns; the blob's
    # scalar slots then carry JSON null. ``pooled`` (the only reliable source
    # -- scalar cases are value-identical either way) and ``per_unit`` (whether
    # the vector columns are present) always ride along.
    gam_meta: dict[str, object] | None = None
    gam_columns: list[VectorData] = []
    if result.method == "glm":
        rank = int(result.rank)  # type: ignore[arg-type]
        per_unit = result.penalty_selected_by_reml is not None
        gam_meta = {
            "rank": rank,
            "n_iter": int(result.n_iter),  # type: ignore[arg-type]
            "converged": bool(result.converged),
            "penalty_weights": np.asarray(
                result.penalty_weights, dtype=np.float64
            ).tolist(),
            "pooled": bool(result.pooled),
            "per_unit": per_unit,
            # Scalar slots: real values when scalar, JSON null when the value
            # lives in a per-unit column instead.
            "penalty": None
            if per_unit or result.penalty is None
            else float(result.penalty),
            "reml_objective": None
            if per_unit or result.reml_objective is None
            else float(result.reml_objective),
            "reml_at_boundary": None
            if per_unit or result.reml_at_boundary is None
            else bool(result.reml_at_boundary),
        }
        # coefficients (rank, n_units) -> per-unit rows (n_units, rank), matching
        # the firing_rate per-unit-row layout; deviance (n_units,) as-is.
        gam_columns = [
            VectorData(
                name=COL_DEVIANCE,
                description="Per-unit unpenalized Poisson deviance (method='glm').",
                data=np.asarray(result.deviance, dtype=np.float64),
            ),
            VectorData(
                name=COL_COEFFICIENTS,
                description=(
                    f"Per-unit GAM coefficients (n_units, rank={rank}); "
                    f"transpose back to (rank, n_units) on read (method='glm')."
                ),
                data=np.asarray(result.coefficients, dtype=np.float64).T,
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

    # A unit_table column may not share a name with a column the table writes
    # itself (``units`` is reserved even when no link is written). Refuse
    # before any mutation, naming the argument, instead of a low-level hdmf
    # error.
    unit_table = result.unit_table
    unit_table_columns = (
        [] if unit_table is None else [str(c) for c in unit_table.columns]
    )
    reserved_names = [
        COL_UNIT_ID,
        COL_FIRING_RATE,
        COL_UNITS,
        *(column.name for column in gam_columns),
    ]
    reserved_clash = [c for c in unit_table_columns if c in reserved_names]
    if reserved_clash:
        msg = (
            f"unit_table has reserved column name(s) {reserved_clash}: "
            f"{reserved_names} are reserved for the spatial-rates "
            f"table's own columns. Rename these unit_table columns."
        )
        raise ValueError(msg)

    # Check every name -- the table, its companion occupancy and the
    # environment record -- before the first mutation. The objects to replace
    # are removed only after the replacements are built, so a failed write
    # can restore them.
    env_metadata, env_arrays, env_fingerprint = _environment_content(env)
    old_names, env_slot = _plan_spatial_rates_write(
        nwbfile, name, occupancy_name, env_name, env_fingerprint, overwrite=overwrite
    )

    meta_dict: dict[str, object] = {
        "format": SPATIAL_RATES_FORMAT,
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
        "environment_sha256": env_fingerprint,
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
    if gam_meta is not None:
        meta_dict["gam"] = gam_meta
    description = json.dumps(meta_dict)

    # The result's arrays are read-only, so the containers can hold them
    # without a copy.
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
                data=result.firing_rates,
            ),
            *_units_link(nwbfile, unit_ids, generated=result._unit_ids_generated),
            *(
                VectorData(
                    name=str(column),
                    description=f"unit_table column '{column}'",
                    data=values.to_numpy(),
                )
                for column, values in (() if unit_table is None else unit_table.items())
            ),
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
                data=result.occupancy,
            )
        ],
    )

    # Remove the originals, then add the table, its companion occupancy and,
    # unless an identical record is reused, the environment. Each removal makes
    # the same overwrite and file-mode checks, so a refusal comes from the
    # first one, before anything is removed. If an add fails, the added objects
    # are removed and the originals restored.
    new_objects = [table, occupancy_table]
    if env_slot != "reuse":
        new_objects.append(_environment_record(env_name, env_metadata, env_arrays))
    what = {
        name: f"spatial rates {name!r}",
        occupancy_name: f"occupancy {occupancy_name!r}",
    }
    removed: list[tuple[str, object]] = []
    try:
        for old_name in old_names:
            old = nwbfile.analysis[old_name]
            _clear_for_write(
                nwbfile,
                nwbfile.analysis,
                old_name,
                overwrite=overwrite,
                what=what.get(old_name, f"environment record {old_name!r}"),
            )
            removed.append((old_name, old))
    except Exception:
        # A refusal comes from the first removal, before anything is removed;
        # put back whatever a later failure would otherwise lose.
        for _old_name, old in reversed(removed):
            nwbfile.add_analysis(old)
        raise
    added: list[str] = []
    try:
        for obj in new_objects:
            nwbfile.add_analysis(obj)
            added.append(obj.name)
    except Exception as exc:
        not_removed = []
        for obj_name in reversed(added):
            try:
                del nwbfile.analysis[obj_name]
            except Exception:  # noqa: PERF203  # best-effort rollback of every record
                not_removed.append(obj_name)
        not_restored = []
        for obj_name, obj in reversed(removed):
            try:
                nwbfile.add_analysis(obj)
            except Exception:  # noqa: PERF203  # best-effort rollback of every record
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
    name: str,
    occupancy_name: str,
    env_name: str,
    env_fingerprint: str,
    *,
    overwrite: bool,
) -> tuple[list[str], Literal["add", "reuse", "replace"]]:
    """Check every name a spatial-rates write uses, before any mutation.

    Returns
    -------
    old_names : list of str
        ``/analysis`` objects to remove: the replaced table and occupancy,
        and an environment record only the replaced rates used when it is
        replaced or, if it was their default ``f"{name}_environment"``, no
        longer used.
    env_slot : {"add", "reuse", "replace"}
        How ``env_name`` stores the environment; see ``_environment_slot``.
    """
    from neurospatial.io.nwb._environment import (
        _environment_slot,
        _is_environment_record,
        _spatial_rates_using,
    )

    analysis = nwbfile.analysis
    if len({name, occupancy_name, env_name}) != 3:
        msg = (
            f"write_spatial_rates: name={name!r}, its occupancy companion "
            f"{occupancy_name!r} and environment_name={env_name!r} must be three "
            "different /analysis names. Choose another environment_name."
        )
        raise ValueError(msg)
    for own in (name, occupancy_name):
        if own in analysis and _is_environment_record(analysis[own]):
            msg = (
                f"write_spatial_rates: /analysis/{own} is an environment record, "
                "which other spatial rates may use. Choose another name= for the "
                "rates."
            )
            raise ValueError(msg)
    old_names = [n for n in (name, occupancy_name) if n in analysis]

    # The replaced rates' own default environment record goes with them when
    # nothing else uses it; a record saved standalone or shared is kept.
    old_meta = _record_metadata(analysis[name]) if name in analysis else None
    old_env = None if old_meta is None else old_meta.get("environment_name")
    if (
        old_env == f"{name}_environment"
        and old_env != env_name
        and old_env in analysis
        and _is_environment_record(analysis[old_env])
        and _spatial_rates_using(nwbfile, old_env) == [name]
    ):
        old_names.append(old_env)

    env_slot = _environment_slot(
        nwbfile,
        env_name,
        env_fingerprint,
        overwrite=overwrite,
        argument="environment_name",
        replacing=name,
    )
    if env_slot == "replace":
        old_names.append(env_name)
    return old_names, env_slot


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


def _require_saved_environment(
    nwbfile: NWBFile,
    env: Environment,
    fingerprint: str,
    n_bins: int,
    *,
    name: str,
    env_name: str,
) -> None:
    """Refuse an explicit ``env=`` that differs from the rates' saved environment.

    ``env`` matches when its record would have the saved ``fingerprint``.
    Otherwise the saved environment is restored to name the difference; one
    that leaves bins, units, frame and topology unchanged, such as the
    environment's name, is accepted.
    """
    from neurospatial._exceptions import IncompatibleEnvironmentError
    from neurospatial.environment import _environment_differences

    if env.n_bins != n_bins:
        msg = (
            f"Environment n_bins ({env.n_bins}) does not match the stored "
            f"firing_rates bin count ({n_bins}) for spatial rates '{name}'. This "
            f"usually means a mismatched or stale env= was passed; pass the "
            f"Environment used to compute these rates."
        )
        raise ValueError(msg)
    if _environment_content(env)[2] == fingerprint:
        return
    saved = read_environment(nwbfile, name=env_name)
    bins, topology = _environment_differences(saved, env)
    mismatch = bins or topology
    if mismatch is not None:
        msg = (
            f"read_spatial_rates: env= differs from the environment saved with "
            f"spatial rates '{name}' in {mismatch}."
        )
        raise IncompatibleEnvironmentError(
            msg,
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
        If ``name`` points at an object whose description does not declare
        the format ``write_spatial_rates`` writes; if the record was written
        with a schema version other than ``SPATIAL_RATES_SCHEMA_VERSION``,
        including rates an earlier version wrote to the processing module
        ``analysis`` (recompute and rewrite the rates); if a metadata key the
        writer always stores is missing or the recorded unit count does not
        match the table; if the companion occupancy length does not match the
        rates' bin count; if the environment record the rates name is not in
        the file (whether or not ``env=`` is given, since an explicit ``env=``
        cannot be verified without it); if that record has changed since the
        rates were written (its content no longer matches the stored
        ``environment_sha256``); or if ``env=`` has another ``n_bins`` than
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
            msg = f"Spatial rates {name!r}"
            raise _earlier_location_error(
                msg,
                f"the processing module {_EARLIER_MODULE!r}",
                fix=_RECREATE_FIX,
            )
        available = list(_analysis_records(nwbfile, SPATIAL_RATES_FORMAT))
        msg = (
            f"Spatial rates '{name}' not found in /analysis. Saved spatial "
            f"rates: {available}"
        )
        raise missing_error(
            msg,
            fix=f"pass name= one of {available}"
            if available
            else "write the rates with write_spatial_rates first",
        )

    table = analysis[name]
    meta = _require_record(
        nwbfile,
        name,
        SPATIAL_RATES_FORMAT,
        what="a spatial-rates",
        writer="write_spatial_rates",
    )
    _check_schema(
        meta,
        f"/analysis/{name}",
        current=SPATIAL_RATES_SCHEMA_VERSION,
        kind="Spatial-rates",
        fix=_RECREATE_FIX,
    )
    method = _required_key(meta, "method", name)
    bandwidth = _required_key(meta, "bandwidth", name)
    n_units = int(_required_key(meta, "n_units", name))
    unit_table_columns = _required_key(meta, "unit_table_columns", name)
    occupancy_name = _required_key(meta, "occupancy_name", name)
    min_occupancy = float(_required_key(meta, "min_occupancy", name))
    n_spikes_outside_intervals = _required_key(meta, "n_spikes_outside_intervals", name)
    analysis_settings = _required_key(meta, "analysis_settings", name)

    unit_ids_generated = _required_key(meta, "unit_ids_generated", name)
    if not isinstance(unit_ids_generated, bool):
        msg = (
            f"Spatial rates '{name}' has invalid unit_ids_generated metadata; "
            "expected a boolean to distinguish generated from explicit labels."
        )
        raise ValueError(msg)
    stored_window = _required_key(meta, "spike_window", name)
    # Results refuse windows without rows unless they are an empty selection,
    # so a stored empty window is one; restore it as that object.
    spike_window = (
        EMPTY_SELECTION
        if stored_window is not None and len(stored_window) == 0
        else as_intervals(stored_window, name=f"{name}.spike_window")
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
        msg = (
            f"Occupancy '{occupancy_name}' for spatial rates '{name}' not found "
            "in /analysis."
        )
        raise missing_error(
            msg,
            fix=_RECREATE_FIX,
        )
    occupancy_record = analysis[occupancy_name]
    if COL_OCCUPANCY not in getattr(occupancy_record, "colnames", ()):
        msg = (
            f"Occupancy companion '{occupancy_name}' for spatial rates '{name}' is "
            f"not a table with an '{COL_OCCUPANCY}' column; the file may be "
            "corrupt or the name was reused for another object."
        )
        raise ValueError(msg)
    occupancy = np.asarray(occupancy_record[COL_OCCUPANCY][:], dtype=np.float64)

    # Check the companion against the rates' bins, so a wrong-length
    # occupancy fails here rather than in a later analysis.
    n_bins = firing_rates.shape[1]
    if occupancy.shape != (n_bins,):
        msg = (
            f"Occupancy companion '{occupancy_name}' has length "
            f"{occupancy.shape[0]} but the rates in '{name}' have {n_bins} bins. "
            "The stored occupancy does not match the rates; the file may be "
            "corrupt or the wrong companion was written."
        )
        raise ValueError(msg)

    # The rates' environment is the record they name, unchanged since they
    # were written. An explicit env= is attached only when it matches that
    # record, so a same-size env with other units, frame, bins or topology
    # cannot relabel the maps.
    env_name = _required_key(meta, "environment_name", name)
    env_fingerprint = _required_key(meta, "environment_sha256", name)
    if env_name not in analysis:
        raise ValueError(
            _format_error(
                f"Spatial rates '{name}' have no saved environment: "
                f"'{env_name}', the environment record they name, is not in "
                "/analysis.",
                why="Why: rates are read with the environment they were "
                "estimated on; equal bin counts do not show equal bins, units "
                "or connectivity.",
                fix="restore the environment record, or recompute the rates and "
                "write them again with write_spatial_rates, which saves it",
            )
        )
    _require_unchanged_environment(
        nwbfile, env_fingerprint, name=name, env_name=env_name
    )
    if env is None:
        env = read_environment(nwbfile, name=env_name)
    else:
        _require_saved_environment(
            nwbfile, env, env_fingerprint, n_bins, name=name, env_name=env_name
        )

    # GAM (method="glm") diagnostics: reconstruct from the JSON blob + the
    # per-unit columns, preserving shapes. Absent (ratio methods) -> all None,
    # including ``pooled`` (meaningless for a ratio result). coefficients is stored
    # transposed as (n_units, rank); transpose back to (rank, n_units). The
    # reconstructed fields must satisfy the SpatialRatesResult None-iff-glm
    # invariant (checked at construction below) -- a wrong shape or a missing field
    # raises there, so a corrupt glm record fails loudly.
    gam_fields: dict[str, object] = {}
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
                msg = (
                    f"Spatial rates '{name}' is a per-unit glm record (per_unit=True) "
                    f"but is missing the per-unit column(s) {missing_cols}; the file "
                    "may be corrupt or truncated."
                )
                raise ValueError(msg)
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


def _require_unchanged_environment(
    nwbfile: NWBFile, fingerprint: str, *, name: str, env_name: str
) -> None:
    """Refuse rates whose environment record is not the one they were written with.

    The record must be a current-schema environment record, read or not:
    ``env=`` skips decoding it, but not this check. Equal bin counts do not
    show equal bins: a record replaced or edited after the rates were written
    could move every bin or relabel its units.
    """
    from neurospatial.io.files import _check_schema
    from neurospatial.io.nwb._environment import (
        ENVIRONMENT_FORMAT,
        _record_fingerprint,
        _require_record,
    )

    metadata = _require_record(
        nwbfile,
        env_name,
        ENVIRONMENT_FORMAT,
        what="an environment",
        writer="write_environment",
    )
    _check_schema(metadata, f"NWB analysis/{env_name}")
    if _record_fingerprint(nwbfile.analysis[env_name]) != fingerprint:
        raise ValueError(
            _format_error(
                f"Spatial rates '{name}': the environment record '{env_name}' "
                "changed since these rates were written.",
                why="Why: its content no longer matches the fingerprint stored "
                "with the rates, so reading would place the maps on bins they "
                "were not estimated on.",
                fix="restore the original environment record, or recompute the "
                "rates on the environment now saved and write them again with "
                "write_spatial_rates",
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
