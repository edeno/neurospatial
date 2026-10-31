"""Lossless environment records in the NWB file's top-level ``/analysis`` group."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from neurospatial._keyword_hint import keyword_hint
from neurospatial.io.nwb._core import _clear_for_write, _require_pynwb

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from hdmf.common import DynamicTable
    from pynwb import NWBFile

    from neurospatial.environment import Environment
    from neurospatial.geometry import Arena
    from neurospatial.layout import LayoutType
    from neurospatial.regions import Region

DEFAULT_ENVIRONMENT_NAME: str = "spatial_environment"
# The "format" each /analysis record this package writes declares in its
# description JSON; records are identified by it, never by column names.
ENVIRONMENT_FORMAT: str = "neurospatial.Environment"
SPATIAL_RATES_FORMAT: str = "neurospatial.SpatialRates"
# Metadata that says what kind of record it is and when and by which release
# it was written, not the environment it describes.
_RECORD_DETAILS = ("format", "schema_version", "library_version", "created_at")


@keyword_hint("name")
def write_environment(
    nwbfile: NWBFile,
    env: Environment,
    *,
    name: str = DEFAULT_ENVIRONMENT_NAME,
    overwrite: bool = False,
) -> None:
    """Store complete bins and Arena in the ``Environment-v4`` format.

    Bins, lookup, cell faces, drawing components, regions, landmarks,
    coordinate metadata and Arena are saved losslessly in the file's
    top-level ``/analysis`` group, in the format :func:`~neurospatial.io.to_file`
    writes. Reading decodes these data without rebuilding a layout, and
    rebuilds the connectivity graph from the binning.
    :func:`read_environment` reads only ``Environment-v4`` and refuses records
    from earlier schemas.

    Parameters
    ----------
    nwbfile : NWBFile
        NWB file receiving the record.
    env : Environment
        Environment to store.
    name : str, default "spatial_environment"
        Record name in ``nwbfile.analysis``.
    overwrite : bool, default False
        Replace an existing environment record with the same name. A record
        that spatial rates use is replaced only by the identical environment,
        so the rates are never read with other bins. pynwb cannot delete a
        record already written to disk, so replace one in a file opened with
        ``mode="r"`` and save a copy with
        ``NWBHDF5IO(new_path, "w").export(src_io=io, nwbfile=nwbfile)``.

    Raises
    ------
    ValueError
        If the name exists and overwrite is false, the name belongs to an
        ``/analysis`` object that is not an environment record, spatial
        rates use the record and ``env`` differs from it (the message names
        them), or a component has no codec.
    NotImplementedError
        If ``overwrite=True`` would replace a record in a file opened with
        ``mode="r+"`` or ``"a"``.
    ImportError
        If pynwb is unavailable.

    Notes
    -----
    The record is a one-row ``DynamicTable``. Each numerical array is one
    ragged column named by its key with ``/`` replaced by ``.`` (for
    example ``binning.bins.centers``), stored flat, gzip-compressed and
    without padding. The table description is one JSON attribute holding
    ``"format": "neurospatial.Environment"``, which identifies the record,
    the remaining metadata and, under ``array_specs``, each array's column,
    shape and dtype. Other tools read bin adjacency from the
    ``binning.connections.all_pairs`` (bin index pairs), ``all_distances``,
    ``is_open`` (False where a wall closes the pair) and ``closed_by``
    columns.

    Examples
    --------
    >>> from datetime import datetime, timezone
    >>> from pynwb import NWBFile
    >>> from neurospatial import Environment
    >>> env = Environment.linear_track(
    ...     endpoints=[(0, 0), (6, 0)], bin_size=2.0, units="cm"
    ... )
    >>> nwb = NWBFile("example", "example", datetime(2025, 1, 1, tzinfo=timezone.utc))
    >>> write_environment(nwb, env)
    >>> "spatial_environment" in nwb.analysis
    True
    >>> read_environment(nwb).bin_sizes.tolist()
    [2.0, 2.0, 2.0]
    """
    _require_pynwb()
    metadata, arrays, fingerprint = _environment_content(env)
    slot = _environment_slot(
        nwbfile, name, fingerprint, overwrite=overwrite, argument="name"
    )
    record = _environment_record(name, metadata, arrays)
    if slot != "add":
        _clear_for_write(
            nwbfile,
            nwbfile.analysis,
            name,
            overwrite=overwrite,
            what=f"environment record {name!r}",
        )
    nwbfile.add_analysis(record)


def _environment_content(
    env: Environment,
) -> tuple[dict[str, Any], dict[str, np.ndarray], str]:
    """The metadata and arrays of ``env``'s record, and their fingerprint."""
    from neurospatial.io.files import _environment_payload

    metadata, arrays = _environment_payload(env)
    return metadata, arrays, _payload_fingerprint(metadata, arrays)


def _environment_record(
    name: str, metadata: dict[str, Any], arrays: Mapping[str, np.ndarray]
) -> DynamicTable:
    """Build the one-row table storing an environment; see ``write_environment``.

    Parameters
    ----------
    name : str
        Record name in ``nwbfile.analysis``.
    metadata, arrays
        The environment's payload from ``_environment_content``. ``metadata``
        gains the record format and each array's column name.
    """
    from hdmf.backends.hdf5 import H5DataIO
    from hdmf.common import DynamicTable, VectorData, VectorIndex

    metadata["format"] = ENVIRONMENT_FORMAT
    columns: list[Any] = []
    for key, array in sorted(arrays.items()):
        # NWB names cannot contain "/"; the canonical keys contain no ".".
        column_name = key.replace("/", ".")
        values = np.asarray(array).reshape(-1).copy()
        column = VectorData(
            name=column_name,
            description=f"Environment array {key!r}, flattened; its shape and "
            "dtype are in the table description under array_specs.",
            data=H5DataIO(values, compression="gzip"),
        )
        columns.extend(
            [
                VectorIndex(
                    name=f"{column_name}_index", data=[values.size], target=column
                ),
                column,
            ]
        )
        metadata["array_specs"][key]["column"] = column_name
    return DynamicTable(
        name=name,
        description=json.dumps(metadata),
        columns=columns,
        id=[0],
    )


def _payload_fingerprint(
    metadata: dict[str, Any], arrays: Mapping[str, np.ndarray]
) -> str:
    """SHA-256 of what a record says about its environment.

    Covers every array (key, dtype, shape and values) and every metadata
    entry except ``_RECORD_DETAILS`` and the NWB column names, so two records
    of the same environment written at different times, or one read back
    from disk, have the same fingerprint.

    Parameters
    ----------
    metadata : dict
        JSON-normalized record metadata, with ``array_specs``.
    arrays : Mapping[str, ndarray]
        The arrays ``array_specs`` describes, in any shape with C-order
        values (flattened NWB columns are accepted).
    """
    described = {k: v for k, v in metadata.items() if k not in _RECORD_DETAILS}
    described["array_specs"] = {
        key: {"shape": list(spec["shape"]), "dtype": spec["dtype"]}
        for key, spec in metadata["array_specs"].items()
    }
    digest = hashlib.sha256(json.dumps(described, sort_keys=True).encode())
    for key in sorted(arrays):
        digest.update(key.encode())
        digest.update(np.ascontiguousarray(arrays[key]).reshape(-1).tobytes())
    return digest.hexdigest()


def _record_arrays(record: Any, metadata: dict[str, Any]) -> dict[str, np.ndarray]:
    """The arrays of a one-row environment record, in their described shapes.

    Dtypes are not checked here; ``files._restore_environment`` validates
    every array against ``array_specs``.

    Raises
    ------
    ValueError, TypeError, KeyError, IndexError
        If the record has other than one row, or its columns do not match
        ``metadata["array_specs"]``.
    """
    if len(record) != 1:
        raise ValueError("a binning record must contain exactly one row")
    return {
        key: np.array(record[spec["column"]][0]).reshape(spec["shape"])
        for key, spec in metadata["array_specs"].items()
    }


def _record_fingerprint(record: Any) -> str | None:
    """Content fingerprint of a stored environment record; None if unreadable."""
    metadata = _record_metadata(record)
    if metadata is None:
        return None
    try:
        return _payload_fingerprint(metadata, _record_arrays(record, metadata))
    except (KeyError, TypeError, ValueError, IndexError, AttributeError):
        return None


def _record_metadata(record: object) -> dict[str, Any] | None:
    """The JSON description of an ``/analysis`` object, or None."""
    try:
        metadata = json.loads(record.description)  # type: ignore[attr-defined]
    except (AttributeError, TypeError, ValueError, RecursionError):
        return None
    return metadata if isinstance(metadata, dict) else None


def _is_environment_record(record: object) -> bool:
    """Whether an ``/analysis`` object is a record ``write_environment`` made."""
    metadata = _record_metadata(record)
    return metadata is not None and metadata.get("format") == ENVIRONMENT_FORMAT


def _analysis_records(nwbfile: NWBFile, record_format: str) -> dict[str, dict]:
    """Metadata of the ``/analysis`` records declaring ``record_format``, by name.

    Names are in sorted order.
    """
    found = {}
    for name in sorted(nwbfile.analysis):
        metadata = _record_metadata(nwbfile.analysis[name])
        if metadata is not None and metadata.get("format") == record_format:
            found[name] = metadata
    return found


def _spatial_rates_using(nwbfile: NWBFile, env_name: str) -> list[str]:
    """Names of the spatial-rates records in ``/analysis`` that use ``env_name``."""
    return [
        name
        for name, metadata in _analysis_records(nwbfile, SPATIAL_RATES_FORMAT).items()
        if metadata.get("environment_name") == env_name
    ]


def _require_record(
    nwbfile: NWBFile, name: str, record_format: str, *, what: str, writer: str
) -> dict[str, Any]:
    """Return the metadata of ``/analysis/{name}``, a record of ``record_format``.

    Parameters
    ----------
    nwbfile : NWBFile
        File holding the record in ``/analysis``.
    name : str
        Name of an existing ``/analysis`` object.
    record_format : str
        The ``"format"`` the record's description declares.
    what : str
        The kind of record with its article, e.g. ``"an environment"``.
    writer : str
        The function that writes such records.

    Raises
    ------
    ValueError
        If the object's description does not declare ``record_format``.
    """
    from neurospatial._exceptions import _format_error

    metadata = _record_metadata(nwbfile.analysis[name])
    if metadata is None or metadata.get("format") != record_format:
        saved = list(_analysis_records(nwbfile, record_format))
        raise ValueError(
            _format_error(
                f"/analysis/{name} is not {what} record (saved: {saved or 'none'}).",
                why=f"Why: its description does not declare format "
                f"{record_format!r}, as every record written by {writer} does.",
                fix=f"pass name= one of {saved}"
                if saved
                else f"write the record with {writer} first",
            )
        )
    return metadata


def _earlier_location_error(what: str, where: str, *, fix: str) -> ValueError:
    """Refusal of a record an earlier version saved outside ``/analysis``."""
    from neurospatial._exceptions import _format_error

    return ValueError(
        _format_error(
            f"{what} is in {where}, where earlier versions of neurospatial saved "
            "it; this version reads only records in /analysis.",
            why="Why: earlier records store a different layout and are not converted.",
            fix=fix,
        )
    )


def _environment_slot(
    nwbfile: NWBFile,
    name: str,
    fingerprint: str,
    *,
    overwrite: bool,
    argument: str,
    replacing: str | None = None,
) -> Literal["add", "reuse", "replace"]:
    """Decide how a write stores the environment ``fingerprint`` as ``name``.

    An environment record that spatial rates use is replaced only by the
    identical environment, so the rates are never read with other bins.

    Parameters
    ----------
    nwbfile : NWBFile
        File receiving the record.
    name : str
        Record name in ``nwbfile.analysis``.
    fingerprint : str
        Fingerprint of the environment to store; see ``_payload_fingerprint``.
    overwrite : bool
        Whether the caller may replace an existing record.
    argument : str
        The caller's argument that names the record, for messages.
    replacing : str, optional
        Spatial rates the caller is rewriting. Only a record they use is
        replaced, never one saved on its own or for other rates.

    Returns
    -------
    {"add", "reuse", "replace"}
        ``"add"`` when the name is free, ``"reuse"`` when the record stores
        this environment, ``"replace"`` when it stores another environment
        that the caller may replace.

    Raises
    ------
    ValueError
        If ``name`` holds an ``/analysis`` object that is not an environment
        record, spatial rates other than ``replacing`` use a record storing
        another environment, or the caller may not replace that record.
    """
    from neurospatial._exceptions import _format_error

    analysis = nwbfile.analysis
    if name not in analysis:
        return "add"
    if not _is_environment_record(analysis[name]):
        raise ValueError(
            _format_error(
                f"/analysis/{name} already holds a "
                f"{type(analysis[name]).__name__} that is not an environment "
                "record.",
                why="Why: the environment would replace another analysis "
                "result of that name.",
                fix=f"pass {argument}= another name",
            )
        )
    if _record_fingerprint(analysis[name]) == fingerprint:
        return "reuse"
    users = _spatial_rates_using(nwbfile, name)
    others = [user for user in users if user != replacing]
    separately = f"pass {argument}= another name to store this environment separately"
    if others:
        raise ValueError(
            _format_error(
                f"Cannot replace environment {name!r}: spatial rates use it "
                f"({others}), and this environment differs from it.",
                why="Why: the rates would be read with bins they were not "
                "estimated on, so every map would be placed at the wrong "
                "positions or in the wrong units.",
                fix=f"{separately}, or recompute the rates {others} on it and "
                "rewrite them with write_spatial_rates(..., overwrite=True)",
            )
        )
    replaceable = replacing is None or replacing in users
    if not (overwrite and replaceable):
        raise ValueError(
            _format_error(
                f"/analysis/{name} stores a different environment.",
                why="Why: a record is replaced only when asked, so whatever "
                "reads it keeps the environment it was saved with.",
                fix=f"{separately}, or overwrite=True to replace it"
                if replaceable
                else separately,
            )
        )
    return "replace"


def _missing_environment_error(nwbfile: NWBFile, name: str) -> KeyError:
    """Say which environments the file stores and how to build one instead."""
    from pynwb.behavior import Position

    from neurospatial.io.nwb._adapters import missing_error
    from neurospatial.io.nwb._core import _find_containers_by_type

    stored = list(_analysis_records(nwbfile, ENVIRONMENT_FORMAT))
    positions = [path for path, _ in _find_containers_by_type(nwbfile, Position)]
    build = "neurospatial.io.nwb.environment_from_position(nwbfile, bin_size=...)"
    if positions:
        build_fix = (
            f"to bin the Position data at {positions[0]} into a new environment, "
            f"call {build}"
        )
    else:
        build_fix = f"to bin a Position container into a new environment, call {build}"
    return missing_error(
        f"No environment named {name!r} is saved in this NWB file "
        f"(saved environments: {stored or 'none'}). read_environment and "
        "Environment.from_nwb load an environment saved with env.to_nwb or "
        "write_environment; they do not bin position data.",
        fix=f"pass name= one of {stored}, or {build_fix}" if stored else build_fix,
    )


@keyword_hint("name")
def read_environment(
    nwbfile: NWBFile,
    *,
    name: str = DEFAULT_ENVIRONMENT_NAME,
) -> Environment[Any]:
    """Decode a saved NWB binning and Arena record without a layout builder.

    Every stored binning capability is retained, including geometric lookup,
    smoothing, boundary bins, plotting and track linearization. Only
    ``Environment-v4`` records are read; records from earlier schemas must be
    recreated with a factory and saved again.

    Parameters
    ----------
    nwbfile : NWBFile
        File holding the environment in its ``/analysis`` group.
    name : str, default "spatial_environment"
        Record name in ``nwbfile.analysis``.

    Returns
    -------
    Environment
        Restored environment, including its original coordinate system.
        Cartesian or egocentric polar, as stored; static type checkers see
        ``Environment[Any]`` because the stored record decides. Check
        ``env.coordinate_kind`` before calling Cartesian-only methods.

    Raises
    ------
    KeyError
        If no environment of that name is saved in the file. The message
        lists the saved environments and, for building one from position
        data, :func:`environment_from_position`.
    ValueError
        If the schema is unsupported (including a record that an earlier
        version saved in NWB scratch space) or the saved data is
        inconsistent.
    ImportError
        If pynwb is unavailable.

    Examples
    --------
    >>> from datetime import datetime, timezone
    >>> from pynwb import NWBFile
    >>> from neurospatial import Environment
    >>> env = Environment.linear_track(
    ...     endpoints=[(0, 0), (6, 0)], bin_size=2.0, units="cm"
    ... )
    >>> nwb = NWBFile("example", "example", datetime(2025, 1, 1, tzinfo=timezone.utc))
    >>> write_environment(nwb, env)
    >>> restored = read_environment(nwb)
    >>> import numpy as np
    >>> restored.to_linear(np.array([[1.0, 0.0]])).tolist()
    [1.0]
    """
    _require_pynwb()
    from neurospatial._exceptions import _format_error
    from neurospatial.io.files import _check_schema, _restore_environment

    if name not in nwbfile.analysis:
        if name in nwbfile.scratch:
            raise _earlier_location_error(
                f"Environment {name!r}",
                "NWB scratch space",
                fix="recreate the environment with an Environment factory and "
                "save it again with this version",
            )
        raise _missing_environment_error(nwbfile, name)
    record = nwbfile.analysis[name]
    source = f"NWB analysis/{name}"
    metadata = _require_record(
        nwbfile,
        name,
        ENVIRONMENT_FORMAT,
        what="an environment",
        writer="write_environment",
    )
    _check_schema(metadata, source)
    try:
        arrays = _record_arrays(record, metadata)
    except (ValueError, TypeError, KeyError, IndexError) as error:
        raise ValueError(
            _format_error(
                f"Invalid environment arrays in {source}: {error}",
                why="Why: the record holds one row whose columns match the "
                "shapes in its description.",
                fix="restore the original environment record, or re-save the original environment",
            )
        ) from error
    return _restore_environment(metadata, source=source, arrays=arrays)


@keyword_hint("bin_size")
def environment_from_position(
    nwbfile: NWBFile,
    *,
    bin_size: float | Sequence[float],
    processing_module: str | None = None,
    position_name: str | None = None,
    columns: Sequence[int] | None = None,
    name: str = "",
    units: str | None = None,
    frame: str | None = None,
    regions: Mapping[str, Region] | None = None,
    arena: Arena | None = None,
    wall_fit: Literal["snap", "split"] = "snap",
    layout: LayoutType | str = "RegularGrid",
    dimension_ranges: Sequence[tuple[float, float]] | None = None,
    infer_active_bins: bool = True,
    bin_count_threshold: int = 0,
    dilate: bool = False,
    fill_holes: bool = False,
    close_gaps: bool = False,
    add_boundary_bins: bool = False,
    connect_diagonal_neighbors: bool = True,
) -> Environment:
    """
    Create an Environment by binning the positions in an NWB Position container.

    Reads the positions with :func:`read_position` and bins them with
    :meth:`Environment.from_samples`. Every binning option means the same as
    in ``from_samples``. To load an environment that was saved in the file,
    use :func:`read_environment` or ``Environment.from_nwb`` instead.

    Parameters
    ----------
    nwbfile : NWBFile
        The NWB file to read from.
    bin_size : float or sequence of floats
        Bin size, in the environment's spatial units (see `units`).
    processing_module : str, optional
        Name of processing module containing Position.
    position_name : str, optional
        Name of specific SpatialSeries within Position.
    columns : sequence of int, optional
        Spatial columns of the series to bin, e.g. ``[0, 1]`` when the
        series also stores speed; see :func:`read_position`. Default bins
        every column. The selected columns take the series unit only when it
        is one length or pixel unit; otherwise pass ``units=``.
    name : str, default ""
        Name of the environment.
    units : str, optional
        Spatial units of the positions. By default, the unit the series
        declares, with NWB long names mapped to ``m`` / ``cm`` / ``mm`` /
        ``px``. Positions are read in that unit (stored × ``conversion`` +
        ``offset``), so ``bin_size`` is in it too. Required when the series
        declares no unit. A unit that differs from the declared one raises:
        a label cannot convert the coordinates.
    frame : str, optional
        Coordinate frame of the positions. By default, the series
        ``reference_frame``. Required to be identical to it when the series
        declares one; supply it when the series does not.
    regions : Mapping[str, Region], optional
        Named regions to snapshot with the environment.
    arena : Arena, optional
        Explicit continuous domain. See :meth:`Environment.from_samples`.
    wall_fit : {'snap', 'split'}, default='snap'
        What to do with a bin that a wall of ``arena`` divides; see
        ``Environment.from_samples``.
    layout : LayoutType or str, default "RegularGrid"
        ``"RegularGrid"`` or ``"Hexagonal"``.
    dimension_ranges : sequence of (float, float), optional
        ``[(min, max), ...]`` extent to tile on each axis.
    infer_active_bins : bool, default True
        Whether to only include bins visited by the animal.
    bin_count_threshold : int, default 0
        Minimum samples per bin. Only with ``infer_active_bins=True``.
    dilate, fill_holes, close_gaps, add_boundary_bins : bool, default False
        Regular grid only; see :meth:`Environment.from_samples`.
    connect_diagonal_neighbors : bool, default True
        Regular grid only; see :meth:`Environment.from_samples`.

        Changing any regular-grid-only option from its default with
        ``layout="Hexagonal"`` raises ``ValueError``, and so does changing
        `bin_count_threshold`, `dilate`, `fill_holes` or `close_gaps` with
        ``infer_active_bins=False``. The defaults are always accepted.

    Returns
    -------
    Environment
        Environment discretized from position data.

    Raises
    ------
    KeyError
        If no Position container found, or if specified position_name not found.
    ValueError
        If the series declares no unit and `units` is not given, if `units`
        or `frame` conflicts with the series declaration, if the series
        needs `columns` or `columns` is invalid (see :func:`read_position`), if
        `layout` is neither "RegularGrid" nor "Hexagonal", or an option
        that the chosen layout or ``infer_active_bins=False`` ignores differs
        from its default. The message names ``environment_from_position``.
    ImportError
        If pynwb is not installed.

    Examples
    --------
    >>> from pynwb import NWBHDF5IO  # doctest: +SKIP
    >>> with NWBHDF5IO("session.nwb", "r") as io:  # doctest: +SKIP
    ...     nwbfile = io.read()
    ...     env = environment_from_position(nwbfile, bin_size=2.0)

    With explicit units and active bin inference:

    >>> env = environment_from_position(  # doctest: +SKIP
    ...     nwbfile,
    ...     bin_size=5.0,
    ...     units="cm",
    ...     infer_active_bins=True,
    ...     bin_count_threshold=5,
    ... )

    Bin only x and y of a series that also stores speed in column 2:

    >>> env = environment_from_position(  # doctest: +SKIP
    ...     nwbfile, bin_size=2.0, columns=[0, 1], units="cm"
    ... )
    """
    _require_pynwb()

    from neurospatial._exceptions import _format_error
    from neurospatial.environment import Environment, _from_samples
    from neurospatial.io.nwb._adapters import supply_coordinate_metadata
    from neurospatial.io.nwb._behavior import read_position

    position_data = read_position(
        nwbfile,
        processing_module=processing_module,
        position_name=position_name,
        columns=columns,
    )
    series = (
        "The Position series"
        if position_name is None
        else f"Position series {position_name!r}"
    )
    if columns is not None and units is None and position_data.units is None:
        raise ValueError(
            _format_error(
                f"{series} states no single unit for columns={list(columns)}.",
                why=(
                    "Why: its unit is missing or is not one length or pixel "
                    "unit, so it may describe the columns left out; bin_size "
                    "and every distance derived from the environment are in "
                    "the position units."
                ),
                fix=(
                    "pass units= with the unit of the selected columns, e.g. "
                    "units='cm' for 'centimeters; centimeters/second'"
                ),
            )
        )
    declared = supply_coordinate_metadata(
        {"units": position_data.units, "frame": position_data.frame},
        {
            key: value
            for key, value in (("units", units), ("frame", frame))
            if value is not None
        },
        call="environment_from_position",
        source=f"{series[0].lower()}{series[1:]}",
        fix="convert the positions and bin them with Environment.from_samples",
    )
    if declared["units"] is None:
        raise ValueError(
            _format_error(
                f"{series} declares no unit, so the physical scale of its "
                "positions is unknown.",
                why=(
                    "Why: bin_size and every distance, speed and area derived "
                    "from the environment are in the position units."
                ),
                fix=(
                    "pass units='cm' (or 'm', 'mm', 'px') to "
                    "environment_from_position, stating the unit the tracking "
                    "system recorded"
                ),
            )
        )

    # Bin the samples exactly as Environment.from_samples does, naming this
    # function in any error about the options.
    return _from_samples(
        Environment,
        position_data.positions,
        bin_size=bin_size,
        name=name,
        units=declared["units"],
        frame=declared["frame"],
        regions=regions,
        arena=arena,
        wall_fit=wall_fit,
        layout=layout,
        dimension_ranges=dimension_ranges,
        infer_active_bins=infer_active_bins,
        bin_count_threshold=bin_count_threshold,
        dilate=dilate,
        fill_holes=fill_holes,
        close_gaps=close_gaps,
        add_boundary_bins=add_boundary_bins,
        connect_diagonal_neighbors=connect_diagonal_neighbors,
        factory="environment_from_position",
    )
