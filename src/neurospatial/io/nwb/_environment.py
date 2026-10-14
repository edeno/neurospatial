"""Lossless environment binning records in standard NWB scratch containers."""

from __future__ import annotations

import json
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

from neurospatial.environment.decorators import EnvironmentNotFittedError
from neurospatial.io.nwb._adapters import position_units_from_series
from neurospatial.io.nwb._core import _require_pynwb, logger

if TYPE_CHECKING:
    from pynwb import NWBFile

    from neurospatial import Environment
    from neurospatial.regions import Regions

ENVIRONMENT_SCHEMA_VERSION: str = "2.0"
_READABLE_SCHEMA_VERSIONS: frozenset[str] = frozenset({"2.0"})
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
    """Store complete environment binning data in an NWB 2.0 record.

    Bins, lookup, cell faces, drawing components, graph attributes, regions and
    coordinate metadata are saved losslessly. Reading decodes these data without
    rebuilding a layout. Environment records from schemas 1.0 and 1.1 are refused.

    Parameters
    ----------
    nwbfile : NWBFile
        NWB file receiving the scratch record.
    env : Environment
        Fitted environment to store.
    name : str, default "spatial_environment"
        Scratch record name.
    overwrite : bool, default False
        Replace an existing record with the same name.

    Raises
    ------
    ValueError
        If the name exists and overwrite is false, or a component has no codec.
    EnvironmentNotFittedError
        If the environment has not been fitted.
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

    if not env._is_fitted:
        raise EnvironmentNotFittedError("write_environment", is_function=True)
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


def read_environment(
    nwbfile: NWBFile,
    name: str = DEFAULT_ENVIRONMENT_NAME,
) -> Environment:
    """Decode a saved NWB 2.0 binning record without a layout builder.

    Every stored binning capability is retained, including geometric lookup,
    smoothing, boundary bins, plotting and track linearization. Older environment
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
        If the scratch record does not exist.
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
        available = list(nwbfile.scratch) if nwbfile.scratch else []
        raise KeyError(
            f"Environment '{name}' not found in scratch/. Available: {available}"
        )
    record = nwbfile.scratch[name]
    source = f"NWB scratch/{name}"
    try:
        if COL_METADATA in getattr(record, "colnames", ()):
            metadata = json.loads(record[COL_METADATA][0])
        else:
            metadata = json.loads(record.description)
    except (ValueError, TypeError, KeyError, IndexError, AttributeError) as error:
        raise ValueError(
            f"Invalid environment metadata in {source}: {error}"
        ) from error
    version = metadata.get("schema_version")
    if version not in _READABLE_SCHEMA_VERSIONS:
        raise ValueError(
            _format_error(
                f"Unsupported NWB environment schema {version!r} in {source}; expected '2.0'.",
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
    return _restore_environment(metadata, source=source, arrays=arrays)


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


def environment_from_position(
    nwbfile: NWBFile,
    bin_size: float,
    *,
    processing_module: str | None = None,
    position_name: str | None = None,
    units: str | None = None,
    frame: str | None = None,
    **kwargs: Any,
) -> Environment:
    """
    Create Environment from NWB Position data.

    Convenience function that reads position data and creates an Environment
    using Environment.from_samples().

    Parameters
    ----------
    nwbfile : NWBFile
        The NWB file to read from.
    bin_size : float
        Bin size for environment discretization.
    processing_module : str, optional
        Name of processing module containing Position.
    position_name : str, optional
        Name of specific SpatialSeries within Position.
    units : str, optional
        Spatial units for the environment. If None, auto-detected from the
        series ``unit``, with NWB long names mapped to ``m`` / ``cm`` / ``mm``
        / ``px``. Positions are read in that unit (stored × ``conversion`` +
        ``offset``), so ``bin_size`` is in it too. If the series declares no
        unit, a ``UserWarning`` is emitted and ``"cm"`` is assumed; pass
        ``units=`` to state the real unit.
    frame : str, optional
        Coordinate frame identifier for the environment.
    **kwargs
        Additional arguments passed to Environment.from_samples().
        Common kwargs include:
        - infer_active_bins : bool
            Whether to only include bins visited by the animal.
        - bin_count_threshold : int
            Minimum samples per bin when infer_active_bins=True.

    Returns
    -------
    Environment
        Environment discretized from position data.

    Raises
    ------
    KeyError
        If no Position container found, or if specified position_name not found.
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

    from neurospatial import Environment
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

    # Create Environment from position samples
    env = Environment.from_samples(positions, bin_size=bin_size, **kwargs)

    # Set metadata
    env.units = units
    if frame is not None:
        env.frame = frame

    return env


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
