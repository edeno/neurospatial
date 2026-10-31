"""Lossless environment binning records in standard NWB scratch containers."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from neurospatial._keyword_hint import keyword_hint
from neurospatial.io.nwb._core import (
    _refuse_in_place_replacement,
    _require_pynwb,
    logger,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from pynwb import NWBFile

    from neurospatial.environment import Environment
    from neurospatial.geometry import Arena
    from neurospatial.layout import LayoutType
    from neurospatial.regions import Region

ENVIRONMENT_SCHEMA_VERSION: str = "3.0"
DEFAULT_ENVIRONMENT_NAME: str = "spatial_environment"
# Older records keep the version in this column; it is read only to refuse them.
COL_METADATA = "metadata"


@keyword_hint("name")
def write_environment(
    nwbfile: NWBFile,
    env: Environment,
    *,
    name: str = DEFAULT_ENVIRONMENT_NAME,
    overwrite: bool = False,
) -> None:
    """Store complete bins and Arena using NWB environment schema 3.0.

    Bins, lookup, cell faces, drawing components, graph attributes, regions,
    landmarks, coordinate metadata and Arena are saved losslessly. Reading decodes these
    data without rebuilding a layout. :func:`read_environment` reads only
    schema 3.0 and refuses records from earlier schemas.

    Parameters
    ----------
    nwbfile : NWBFile
        NWB file receiving the scratch record.
    env : Environment
        Environment to store.
    name : str, default "spatial_environment"
        Scratch record name.
    overwrite : bool, default False
        Replace an existing record with the same name. pynwb cannot delete a
        record already written to disk, so replace one in a file opened with
        ``mode="r"`` and save a copy with
        ``NWBHDF5IO(new_path, "w").export(src_io=io, nwbfile=nwbfile)``.

    Raises
    ------
    ValueError
        If the name exists and overwrite is false, or a component has no codec.
    NotImplementedError
        If ``overwrite=True`` would replace a record in a file opened with
        ``mode="r+"`` or ``"a"``.
    ImportError
        If pynwb is unavailable.

    Notes
    -----
    Standard one-row ragged columns retain each numerical array without padding.
    The record description is one JSON metadata attribute, including shapes,
    dtypes and the mapping from canonical array keys to NWB column names.

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
    >>> read_environment(nwb).bin_sizes.tolist()
    [2.0, 2.0, 2.0]
    """
    _require_pynwb()
    from hdmf.common import DynamicTable, VectorData, VectorIndex

    from neurospatial.io.files import _environment_payload

    if name in nwbfile.scratch and not overwrite:
        raise ValueError(
            f"Environment '{name}' already exists in scratch. Use overwrite=True to replace."
        )
    metadata, arrays = _environment_payload(env)
    metadata["schema_version"] = ENVIRONMENT_SCHEMA_VERSION
    columns: list[Any] = []
    for index, (key, array) in enumerate(sorted(arrays.items())):
        column_name = f"array_{index}"
        values = np.asarray(array).reshape(-1).copy()
        column = VectorData(name=column_name, description=key, data=values)
        columns.extend(
            [
                VectorIndex(
                    name=f"{column_name}_index", data=[values.size], target=column
                ),
                column,
            ]
        )
        metadata["array_specs"][key]["column"] = column_name
    record = DynamicTable(
        name=name,
        description=json.dumps(metadata),
        columns=columns,
        id=[0],
    )
    # Construct the replacement before removing a valid existing record.
    if name in nwbfile.scratch:
        _refuse_in_place_replacement(nwbfile, f"environment {name!r}")
        del nwbfile.scratch[name]
        logger.info("Overwriting existing environment '%s'", name)
    nwbfile.add_scratch(record)


class _MissingEnvironmentError(KeyError):
    """No saved environment of the requested name; prints its message as written."""

    def __str__(self) -> str:
        return str(self.args[0])


def _missing_environment_message(nwbfile: NWBFile, name: str) -> str:
    """Say which environments the file stores and how to build one instead."""
    from pynwb.behavior import Position

    from neurospatial._exceptions import _format_error
    from neurospatial.io.nwb._core import _find_containers_by_type

    stored = sorted(nwbfile.scratch) if nwbfile.scratch else []
    positions = [path for path, _ in _find_containers_by_type(nwbfile, Position)]
    build = "neurospatial.io.nwb.environment_from_position(nwbfile, bin_size=...)"
    if positions:
        build_fix = (
            f"to bin the Position data at {positions[0]} into a new environment, "
            f"call {build}"
        )
    else:
        build_fix = f"to bin a Position container into a new environment, call {build}"
    fix = f"pass name= one of {stored}, or {build_fix}" if stored else build_fix
    return _format_error(
        f"No environment named {name!r} is saved in this NWB file "
        f"(saved environments: {stored or 'none'}).",
        why=(
            "Why: read_environment and Environment.from_nwb load an environment "
            "saved with env.to_nwb or write_environment; they do not bin "
            "position data."
        ),
        fix=fix,
    )


@keyword_hint("name")
def read_environment(
    nwbfile: NWBFile,
    *,
    name: str = DEFAULT_ENVIRONMENT_NAME,
) -> Environment[Any]:
    """Decode a saved NWB binning and Arena record without a layout builder.

    Every stored binning capability is retained, including geometric lookup,
    smoothing, boundary bins, plotting and track linearization. Only schema 3.0
    records are read; records from earlier schemas must be recreated with a
    factory and saved again.

    Parameters
    ----------
    nwbfile : NWBFile
        File holding the environment in scratch space.
    name : str, default "spatial_environment"
        Scratch record name.

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
        If the schema is unsupported or the saved data is inconsistent.
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
    from neurospatial.io.files import _restore_environment

    if name not in nwbfile.scratch:
        raise _MissingEnvironmentError(_missing_environment_message(nwbfile, name))
    record = nwbfile.scratch[name]
    source = f"NWB scratch/{name}"
    try:
        if COL_METADATA in getattr(record, "colnames", ()):
            metadata = json.loads(record[COL_METADATA][0])
        else:
            metadata = json.loads(record.description)
        if not isinstance(metadata, dict):
            raise ValueError("environment metadata must be a dictionary")
    except (
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        AttributeError,
        RecursionError,
    ) as error:
        raise ValueError(
            _format_error(
                f"Invalid environment metadata in {source}: {error}",
                why="Why: the record must contain complete JSON metadata without excessive nesting.",
                fix="restore the original environment record, or re-save the original environment",
            )
        ) from error
    version = metadata.get("schema_version")
    if version != ENVIRONMENT_SCHEMA_VERSION:
        raise ValueError(
            _format_error(
                f"NWB environment schema {version!r} in {source} is no longer readable; "
                f"this version reads only {ENVIRONMENT_SCHEMA_VERSION!r}.",
                why="Why: older records lack the Arena and complete binning this version restores, and they are not converted.",
                fix="recreate the environment with an Environment factory and save it again with this version",
            )
        )
    try:
        if len(record) != 1:
            raise ValueError("a binning record must contain exactly one row")
        arrays = {}
        for key, spec in metadata["array_specs"].items():
            values = np.asarray(record[spec["column"]][0])
            if values.dtype.str != spec["dtype"]:
                raise ValueError(f"array {key!r} does not match its stored dtype")
            arrays[key] = values.reshape(spec["shape"]).copy()
    except (ValueError, TypeError, KeyError, IndexError) as error:
        raise ValueError(f"Invalid environment arrays in {source}: {error}") from error
    return _restore_environment(metadata, source=source, arrays=arrays)


@keyword_hint("bin_size")
def environment_from_position(
    nwbfile: NWBFile,
    *,
    bin_size: float | Sequence[float],
    processing_module: str | None = None,
    position_name: str | None = None,
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
        or `frame` conflicts with the series declaration, if
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
    """
    _require_pynwb()

    from neurospatial.environment import Environment, _from_samples
    from neurospatial.io.nwb._behavior import read_position

    position_data = read_position(
        nwbfile,
        processing_module=processing_module,
        position_name=position_name,
    )
    series = (
        "The Position series"
        if position_name is None
        else f"Position series {position_name!r}"
    )
    units, frame = _series_coordinates(
        series,
        {"units": position_data.units, "frame": position_data.frame},
        {"units": units, "frame": frame},
    )

    # Bin the samples exactly as Environment.from_samples does, naming this
    # function in any error about the options.
    return _from_samples(
        Environment,
        position_data.positions,
        bin_size=bin_size,
        name=name,
        units=units,
        frame=frame,
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


def _series_coordinates(
    series: str,
    declared: dict[str, str | None],
    requested: dict[str, str | None],
) -> tuple[str, str | None]:
    """Return the environment's units and frame from the series declarations.

    An explicit value may restate a declaration (unit aliases are the same
    unit) or supply a missing one; it never relabels the coordinates.

    Parameters
    ----------
    series : str
        How to name the Position series in messages.
    declared : dict
        The series ``"units"`` and ``"frame"``; None when undeclared.
    requested : dict
        The caller's ``"units"`` and ``"frame"``; None when not given.

    Returns
    -------
    units : str
        The positions' unit.
    frame : str or None
        The positions' coordinate frame, None when neither side gives one.

    Raises
    ------
    ValueError
        If an explicit value conflicts with the declaration, or no unit is
        declared or given.
    """
    from neurospatial._exceptions import _format_error
    from neurospatial._units import same_coordinate_declaration

    resolved = dict(declared)
    for key, value in requested.items():
        known = declared[key]
        if value is None:
            continue
        if known is None:
            resolved[key] = value
        elif not same_coordinate_declaration(
            key, value, known, undeclared_matches=False
        ):
            raise ValueError(
                _format_error(
                    f"environment_from_position({key}={value!r}) conflicts with "
                    f"{series[0].lower()}{series[1:]}, which declares {key} {known!r}.",
                    why=(
                        "Why: the positions are in the series' coordinates; a "
                        f"different {key} label would not convert them, so every "
                        "distance, bin size and speed would be mislabeled."
                    ),
                    fix=(
                        f"omit {key}= to use {known!r}, or convert the positions "
                        "and bin them with Environment.from_samples"
                    ),
                )
            )
    units = resolved["units"]
    if units is None:
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
    return units, resolved["frame"]
