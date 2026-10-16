"""Lossless environment binning records in standard NWB scratch containers."""

from __future__ import annotations

import json
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

from neurospatial._keyword_hint import keyword_hint
from neurospatial.io.nwb._adapters import position_units_from_series
from neurospatial.io.nwb._core import _require_pynwb, logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pynwb import NWBFile

    from neurospatial import Environment
    from neurospatial.geometry._arena import Arena
    from neurospatial.layout import LayoutType
    from neurospatial.regions import Regions

ENVIRONMENT_SCHEMA_VERSION: str = "3.0"
_READABLE_SCHEMA_VERSIONS: frozenset[str] = frozenset({"2.0", "3.0"})
DEFAULT_ENVIRONMENT_NAME: str = "spatial_environment"
# Older records keep the version in this column; it is read only to refuse them.
COL_METADATA = "metadata"


def write_environment(
    nwbfile: NWBFile,
    env: Environment,
    name: str = DEFAULT_ENVIRONMENT_NAME,
    *,
    overwrite: bool = False,
) -> None:
    """Store complete bins and Arena using NWB environment schema 3.0.

    Bins, lookup, cell faces, drawing components, graph attributes, regions and
    coordinate metadata and Arena are saved losslessly. Reading decodes these
    data without rebuilding a layout. Environment records from schemas 1.0 and
    1.1 are refused.

    Parameters
    ----------
    nwbfile : NWBFile
        NWB file receiving the scratch record.
    env : Environment
        Environment to store.
    name : str, default "spatial_environment"
        Scratch record name.
    overwrite : bool, default False
        Replace an existing record with the same name.

    Raises
    ------
    ValueError
        If the name exists and overwrite is false, or a component has no codec.
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


def read_environment(
    nwbfile: NWBFile,
    name: str = DEFAULT_ENVIRONMENT_NAME,
) -> Environment:
    """Decode a saved NWB binning and Arena record without a layout builder.

    Every stored binning capability is retained, including geometric lookup,
    smoothing, boundary bins, plotting and track linearization. Schema 2.0 records
    retain their bins with an explicitly unknown Arena. Older environment
    records (1.0 and 1.1) must be recreated with a factory and saved again.

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
    if not isinstance(version, str) or version not in _READABLE_SCHEMA_VERSIONS:
        raise ValueError(
            _format_error(
                f"Unsupported NWB environment schema {version!r} in {source}; expected '3.0' (or legacy '2.0').",
                why="Why: older records do not store the complete binning needed to restore its capabilities.",
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
    return _restore_environment(
        metadata, source=source, arrays=arrays, legacy_arena=version == "2.0"
    )


def _regions_to_json(regions) -> str:
    """
    Serialize Regions to a JSON string using the canonical Region schema.

    Delegates to ``Region.to_dict`` so the on-disk schema matches the rest of
    the library (``regions.json``, ``Regions.to_json``) and so per-region
    ``metadata`` survives the NWB round-trip. The output is a list of region
    dicts wrapped in a version-tagged payload.

    Parameters
    ----------
    regions : Regions
        The Regions container to serialize.

    Returns
    -------
    str
        JSON string containing the version-tagged region payload. Empty or
        absent collections serialize to ``"{}"`` for backward compatibility.
    """
    if regions is None or len(regions) == 0:
        return "{}"

    from neurospatial.regions.io import _SCHEMA_TAG

    payload = {
        "format": _SCHEMA_TAG,
        "regions": [region.to_dict() for region in regions.values()],
    }
    return json.dumps(payload)


def _json_to_regions(regions_json: str) -> Regions:
    """
    Deserialize regions from a JSON string.

    Reads the canonical Region schema produced by :func:`_regions_to_json`
    (``{"format": ..., "regions": [<Region.to_dict()>, ...]}``) via
    ``Region.from_dict``, so per-region ``metadata`` is restored. For backward
    compatibility, the legacy bespoke schema
    (``{name: {"kind": ..., "point"/"polygon": ...}}``) written by older files
    is also accepted; legacy entries carry no metadata.

    Parameters
    ----------
    regions_json : str
        JSON string containing region definitions.

    Returns
    -------
    Regions
        Reconstructed Regions container.

    Raises
    ------
    json.JSONDecodeError
        If regions_json is not valid JSON.
    KeyError
        If a region is missing the 'kind' key.

    Warns
    -----
    UserWarning
        If a legacy-schema region has None data and is skipped during
        reconstruction.
    """
    from neurospatial.regions import Region, Regions

    blob = json.loads(regions_json)

    if not blob:
        return Regions()

    # Canonical schema: version-tagged payload with a "regions" list.
    if isinstance(blob, dict) and "regions" in blob:
        return Regions(Region.from_dict(d) for d in blob["regions"])

    # Legacy bespoke schema: {name: {"kind": ..., "point"/"polygon": ...}}.
    return _legacy_json_to_regions(blob)


def _legacy_json_to_regions(regions_dict: dict[str, Any]) -> Regions:
    """Reconstruct regions from the pre-canonical NWB schema (no metadata).

    Parameters
    ----------
    regions_dict : dict[str, Any]
        Mapping of region name to ``{"kind": ..., "point"/"polygon": ...}``.

    Returns
    -------
    Regions
        Reconstructed Regions container. Regions with ``None`` geometry are
        skipped with a warning.
    """
    from shapely.geometry import Polygon

    from neurospatial.regions import Region, Regions

    regions_list = []
    for region_name, region_data in regions_dict.items():
        kind = region_data["kind"]

        if kind == "point":
            point_coords = region_data.get("point")
            if point_coords is not None:
                point = np.array(point_coords, dtype=np.float64)
                regions_list.append(Region(name=region_name, kind="point", data=point))
            else:
                warnings.warn(
                    f"Region '{region_name}' has kind='point' but no point data. "
                    "Skipping this region.",
                    UserWarning,
                    stacklevel=2,
                )
        elif kind == "polygon":
            polygon_coords = region_data.get("polygon")
            if polygon_coords is not None:
                polygon = Polygon(polygon_coords)
                regions_list.append(
                    Region(name=region_name, kind="polygon", data=polygon)
                )
            else:
                warnings.warn(
                    f"Region '{region_name}' has kind='polygon' but no polygon data. "
                    "Skipping this region.",
                    UserWarning,
                    stacklevel=2,
                )

    return Regions(regions_list)


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
    regions: Regions | None = None,
    arena: Arena | None = None,
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
        Spatial units for the environment. If None, auto-detected from the
        series ``unit``, with NWB long names mapped to ``m`` / ``cm`` / ``mm``
        / ``px``. Positions are read in that unit (stored × ``conversion`` +
        ``offset``), so ``bin_size`` is in it too. If the series declares no
        unit, a ``UserWarning`` is emitted and ``"cm"`` is assumed; pass
        ``units=`` to state the real unit.
    frame : str, optional
        Coordinate frame identifier for the environment.
    regions : Regions, optional
        Named regions to snapshot with the environment.
    arena : Arena, optional
        Explicit continuous domain. See :meth:`Environment.from_samples`.
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
        If `layout` is neither "RegularGrid" nor "Hexagonal", or an option
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
    from neurospatial.io.nwb._core import _require_pynwb

    _require_pynwb()

    from neurospatial.environment import Environment
    from neurospatial.environment._factories import _from_samples
    from neurospatial.io.nwb._behavior import read_position

    # Read position data from NWB
    position_data = read_position(
        nwbfile,
        processing_module=processing_module,
        position_name=position_name,
    )
    positions = position_data.positions

    # Auto-detect units from SpatialSeries if not provided
    if units is None:
        units = _get_position_units(
            nwbfile,
            processing_module=processing_module,
            position_name=position_name,
        )

    # Bin the samples exactly as Environment.from_samples does, naming this
    # function in any error about the options.
    return _from_samples(
        Environment,
        positions,
        bin_size=bin_size,
        name=name,
        units=units,
        frame=frame,
        regions=regions,
        arena=arena,
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


def _get_position_units(
    nwbfile: NWBFile,
    processing_module: str | None = None,
    position_name: str | None = None,
) -> str:
    """
    Get the units from the Position SpatialSeries.

    Parameters
    ----------
    nwbfile : NWBFile
        The NWB file to read from.
    processing_module : str, optional
        Name of processing module containing Position.
    position_name : str, optional
        Name of specific SpatialSeries within Position.

    Returns
    -------
    str
        The units from the SpatialSeries, with NWB long names mapped to
        ``m`` / ``cm`` / ``mm`` / ``px``, or "cm" as fallback.

    Warns
    -----
    UserWarning
        If the SpatialSeries declares no unit (``unit`` is None or empty).

    Notes
    -----
    **Default fallback behavior:** if the SpatialSeries declares no unit, this
    function warns, then assumes ``"cm"``.
    """
    from pynwb.behavior import Position as PositionType

    from neurospatial.io.nwb._behavior import (
        _get_behavior_container,
        _get_spatial_series,
    )

    # Get Position container
    position_container = _get_behavior_container(
        nwbfile, PositionType, "Position", processing_module
    )

    # Get SpatialSeries
    spatial_series = _get_spatial_series(
        position_container, position_name, "SpatialSeries", "Position"
    )

    if not spatial_series.unit:
        warnings.warn(
            f"Position series '{spatial_series.name}' declares no unit "
            f"(unit={spatial_series.unit!r}); assuming 'cm'. A wrong unit "
            "mislabels every distance, speed and bin size derived from this "
            "environment.\n"
            "Fix: pass units='cm' (or 'm', 'mm', 'px') to "
            "environment_from_position to state the real unit.",
            UserWarning,
            stacklevel=3,
        )
        return "cm"
    return position_units_from_series(spatial_series) or str(spatial_series.unit)
