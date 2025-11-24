"""
Environment serialization to/from NWB scratch space.

This module provides functions for writing and reading Environment objects
to NWB scratch/ space using standard NWB types (no custom extension required).
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from neurospatial.nwb._core import _require_pynwb

if TYPE_CHECKING:
    from pynwb import NWBFile

    from neurospatial import Environment

logger = logging.getLogger("neurospatial.nwb")


def write_environment(
    nwbfile: NWBFile,
    env: Environment,
    name: str = "spatial_environment",
    *,
    overwrite: bool = False,
) -> None:
    """
    Write Environment to NWB scratch space using standard types.

    Creates a DynamicTable in scratch space containing environment data
    as columns. Columns are accessible via ``scratch_data["column_name"]``.

    Stored columns:

    - bin_centers : NDArray, shape (n_rows, n_dims)
        Bin center coordinates. First n_bins rows are valid data.
    - edges : NDArray, shape (n_rows, 2)
        Edge list for connectivity graph. First n_edges rows are valid data.
    - edge_weights : NDArray, shape (n_rows,)
        Edge weights (distances). First n_edges values are valid data.
    - dimension_ranges : NDArray, shape (n_rows, 2)
        Min/max extent per dimension. First n_dims rows are valid data.
    - regions : list[str]
        JSON-encoded regions data (repeated for DynamicTable compatibility).
    - metadata : list[str]
        JSON-encoded metadata including n_bins, n_edges, n_dims for
        proper deserialization.

    The description field contains: units, frame, n_dims, layout_type.

    Notes
    -----
    Arrays are padded to uniform row count (required by HDMF DynamicTable).
    Use the n_bins, n_edges, and n_dims values from metadata JSON to
    extract the actual valid data during deserialization.

    Parameters
    ----------
    nwbfile : NWBFile
        The NWB file to write to.
    env : Environment
        The Environment to serialize.
    name : str, default "spatial_environment"
        Name for the environment in scratch/.
    overwrite : bool, default False
        If True, replace existing environment with same name.
        If False, raise ValueError on duplicate name.

    Raises
    ------
    ValueError
        If environment with same name exists and overwrite=False.
    ImportError
        If pynwb is not installed.

    Examples
    --------
    >>> from pynwb import NWBHDF5IO
    >>> with NWBHDF5IO("session.nwb", "r+") as io:
    ...     nwbfile = io.read()
    ...     write_environment(nwbfile, env, name="linear_track")
    ...     io.write(nwbfile)
    """
    _require_pynwb()
    from hdmf.common import DynamicTable, VectorData

    # Validate environment is fitted
    if not env._is_fitted:
        raise RuntimeError(
            "Environment must be fitted before calling write_environment(). "
            "Use factory methods like Environment.from_samples() to create "
            "fitted environments."
        )

    # Check for existing environment
    if name in nwbfile.scratch:
        if not overwrite:
            raise ValueError(
                f"Environment '{name}' already exists in scratch. "
                "Use overwrite=True to replace."
            )
        # Remove existing for replacement (in-memory only)
        del nwbfile.scratch[name]
        logger.info("Overwriting existing environment '%s'", name)

    # Extract edge list and weights from connectivity graph
    edges_list = list(env.connectivity.edges(data=True))
    if edges_list:
        edges = np.array([[e[0], e[1]] for e in edges_list], dtype=np.int64)
        edge_weights = np.array(
            [e[2].get("distance", 1.0) for e in edges_list], dtype=np.float64
        )
    else:
        edges = np.empty((0, 2), dtype=np.int64)
        edge_weights = np.empty((0,), dtype=np.float64)

    # Collect metadata for description
    units = env.units if env.units else "unknown"
    frame = env.frame if env.frame else "unknown"
    n_dims = env.bin_centers.shape[1]
    layout_type = env.layout._layout_type_tag if env.layout else "unknown"

    description = (
        f"Spatial environment: units={units}, frame={frame}, "
        f"n_dims={n_dims}, layout_type={layout_type}"
    )

    # Serialize regions to JSON
    regions_data = _regions_to_json(env.regions) if env.regions else "{}"

    # Serialize extra metadata (include array lengths for deserialization)
    metadata = json.dumps(
        {
            "name": env.name,
            "units": units,
            "frame": frame,
            "n_dims": n_dims,
            "layout_type": layout_type,
            "n_bins": env.n_bins,
            "n_edges": len(edges),  # Needed for proper deserialization
        }
    )

    # Use DynamicTable with row-aligned data (pad shorter arrays to match n_bins)
    # This is required because DynamicTable needs uniform row counts
    n_rows = max(env.n_bins, len(edges), n_dims, 1)

    # Pad arrays to n_rows
    bin_centers_padded = np.zeros((n_rows, n_dims), dtype=np.float64)
    bin_centers_padded[: env.n_bins] = env.bin_centers

    edges_padded = np.zeros((n_rows, 2), dtype=np.int64)
    if len(edges) > 0:
        edges_padded[: len(edges)] = edges

    edge_weights_padded = np.zeros(n_rows, dtype=np.float64)
    if len(edge_weights) > 0:
        edge_weights_padded[: len(edge_weights)] = edge_weights

    dim_ranges_padded = np.zeros((n_rows, 2), dtype=np.float64)
    dim_ranges_padded[:n_dims] = env.dimension_ranges

    # String data - repeat to match n_rows
    regions_list = [regions_data] * n_rows
    metadata_list = [metadata] * n_rows

    # Create DynamicTable with padded columns
    table = DynamicTable(
        name=name,
        description=description,
        columns=[
            VectorData(
                name="bin_centers",
                data=bin_centers_padded,
                description=f"Bin center coordinates, actual shape ({env.n_bins}, {n_dims})",
            ),
            VectorData(
                name="edges",
                data=edges_padded,
                description=f"Edge list for connectivity graph, actual shape ({len(edges)}, 2)",
            ),
            VectorData(
                name="edge_weights",
                data=edge_weights_padded,
                description=f"Edge weights (distances), actual length {len(edge_weights)}",
            ),
            VectorData(
                name="dimension_ranges",
                data=dim_ranges_padded,
                description=f"Min/max extent per dimension, actual shape ({n_dims}, 2)",
            ),
            VectorData(
                name="regions",
                data=regions_list,
                description="JSON-encoded regions (points and polygons)",
            ),
            VectorData(
                name="metadata",
                data=metadata_list,
                description="JSON-encoded metadata",
            ),
        ],
    )

    # Add to scratch space
    nwbfile.add_scratch(table)
    logger.debug(
        "Wrote environment '%s' with %d bins and %d edges",
        name,
        env.n_bins,
        len(edges),
    )


def _regions_to_json(regions) -> str:
    """
    Serialize Regions to JSON string.

    Parameters
    ----------
    regions : Regions
        The Regions container to serialize.

    Returns
    -------
    str
        JSON string containing region definitions.
    """
    if regions is None or len(regions) == 0:
        return "{}"

    regions_dict = {}
    for region_name, region in regions.items():
        region_data: dict[str, Any] = {"kind": region.kind}

        if region.kind == "point":
            # For points, region.data is an ndarray
            if region.data is not None:
                region_data["point"] = list(region.data)
            else:
                region_data["point"] = None
        elif region.kind == "polygon":
            # For polygons, region.data is a Shapely Polygon
            if region.data is not None:
                # Extract polygon coordinates
                coords = list(region.data.exterior.coords)
                region_data["polygon"] = coords
            else:
                region_data["polygon"] = None

        regions_dict[region_name] = region_data

    return json.dumps(regions_dict)


def read_environment(
    nwbfile: NWBFile,
    name: str = "spatial_environment",
) -> Environment:
    """
    Read Environment from NWB scratch space.

    Reconstructs Environment from stored bin_centers and edge list.
    Rebuilds connectivity graph and regions.

    Parameters
    ----------
    nwbfile : NWBFile
        The NWB file to read from.
    name : str, default "spatial_environment"
        Name of the environment group in scratch/.

    Returns
    -------
    Environment
        Reconstructed Environment with all attributes.

    Raises
    ------
    KeyError
        If environment not found in scratch/{name}.
    ImportError
        If pynwb is not installed.

    Examples
    --------
    >>> from pynwb import NWBHDF5IO
    >>> with NWBHDF5IO("session.nwb", "r") as io:
    ...     nwbfile = io.read()
    ...     env = read_environment(nwbfile, name="linear_track")
    """
    raise NotImplementedError("read_environment not yet implemented")


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
        Spatial units for the environment. If None, auto-detected from
        the SpatialSeries unit attribute.
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
    >>> from pynwb import NWBHDF5IO
    >>> with NWBHDF5IO("session.nwb", "r") as io:
    ...     nwbfile = io.read()
    ...     env = environment_from_position(nwbfile, bin_size=2.0)

    With explicit units and active bin inference:

    >>> env = environment_from_position(
    ...     nwbfile,
    ...     bin_size=5.0,
    ...     units="cm",
    ...     infer_active_bins=True,
    ...     bin_count_threshold=5,
    ... )
    """
    from neurospatial.nwb._core import _require_pynwb

    _require_pynwb()

    from neurospatial import Environment
    from neurospatial.nwb._behavior import read_position

    # Read position data from NWB
    positions, _timestamps = read_position(
        nwbfile,
        processing_module=processing_module,
        position_name=position_name,
    )

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
        The units from the SpatialSeries, or "cm" as fallback.
    """
    from pynwb.behavior import Position as PositionType

    from neurospatial.nwb._behavior import (
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

    # Return units (default to "cm" if not set)
    return str(spatial_series.unit) if spatial_series.unit else "cm"
