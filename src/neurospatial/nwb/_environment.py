"""
Environment serialization to/from NWB scratch space.

This module provides functions for writing and reading Environment objects
to NWB scratch/ space using standard NWB types (no custom extension required).
"""

from __future__ import annotations

import json
import logging
import warnings
from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from neurospatial.nwb._core import _require_pynwb

if TYPE_CHECKING:
    from pynwb import NWBFile

    from neurospatial import Environment
    from neurospatial.regions import Regions

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

    Notes
    -----
    The reconstructed Environment has some limitations compared to the original:

    - The ``original_grid_nd_index`` node attribute uses simplified values ``(i,)``
      rather than the original N-D indices. Operations depending on grid structure
      may not work identically.
    - Grid-specific attributes (``grid_shape``, ``active_mask``) are not preserved
      and will be populated by the dummy layout used during reconstruction.
    - The layout engine is a RegularGrid approximation; the original layout type
      is stored in ``_layout_type_used`` for reference but cannot be fully restored.

    Examples
    --------
    >>> from pynwb import NWBHDF5IO
    >>> with NWBHDF5IO("session.nwb", "r") as io:
    ...     nwbfile = io.read()
    ...     env = read_environment(nwbfile, name="linear_track")
    """
    _require_pynwb()

    from neurospatial import Environment

    # Check if environment exists
    if name not in nwbfile.scratch:
        available = list(nwbfile.scratch.keys()) if nwbfile.scratch else []
        raise KeyError(
            f"Environment '{name}' not found in scratch/. Available: {available}"
        )

    scratch_data = nwbfile.scratch[name]

    # Parse metadata JSON (get first row - all rows have same value)
    metadata_json = scratch_data["metadata"][0]
    metadata = json.loads(metadata_json)

    n_bins = metadata["n_bins"]
    n_edges = metadata["n_edges"]
    n_dims = metadata["n_dims"]
    env_name = metadata.get("name", "")
    units = metadata.get("units")
    frame = metadata.get("frame")
    layout_type = metadata.get("layout_type", "RegularGrid")

    # Extract arrays (truncate padding)
    bin_centers = np.array(scratch_data["bin_centers"][:n_bins], dtype=np.float64)
    edges = np.array(scratch_data["edges"][:n_edges], dtype=np.int64)
    edge_weights = np.array(scratch_data["edge_weights"][:n_edges], dtype=np.float64)
    dimension_ranges_raw = np.array(
        scratch_data["dimension_ranges"][:n_dims], dtype=np.float64
    )
    dimension_ranges = [tuple(row) for row in dimension_ranges_raw]

    # Reconstruct connectivity graph from edge list
    connectivity = _reconstruct_graph(bin_centers, edges, edge_weights)

    # Parse regions JSON
    regions_json = scratch_data["regions"][0]
    regions = _json_to_regions(regions_json)

    # Create environment using a minimal layout, then override attributes
    # We compute an approximate bin_size from the data for the dummy layout
    bin_size = _estimate_bin_size(bin_centers, n_dims)

    # Create a RegularGrid layout with dimension_ranges
    env = Environment.from_layout(
        kind="RegularGrid",
        layout_params={
            "bin_size": bin_size,
            "dimension_ranges": dimension_ranges,
            "infer_active_bins": False,
        },
        name=env_name,
        regions=regions,
    )

    # Override computed attributes with stored values
    env.bin_centers = bin_centers
    env.connectivity = connectivity
    env.dimension_ranges = dimension_ranges

    # Set metadata
    if units and units != "unknown":
        env.units = units
    if frame and frame != "unknown":
        env.frame = frame

    # Store layout type info
    env._layout_type_used = layout_type

    logger.debug(
        "Read environment '%s' with %d bins and %d edges",
        name,
        n_bins,
        n_edges,
    )

    return env


def _reconstruct_graph(
    bin_centers: NDArray[np.float64],
    edges: NDArray[np.int64],
    edge_weights: NDArray[np.float64],
) -> nx.Graph:
    """
    Reconstruct connectivity graph from edge list and weights.

    Parameters
    ----------
    bin_centers : NDArray, shape (n_bins, n_dims)
        Bin center coordinates.
    edges : NDArray, shape (n_edges, 2)
        Edge list (node pairs).
    edge_weights : NDArray, shape (n_edges,)
        Edge weights (distances).

    Returns
    -------
    nx.Graph
        Reconstructed connectivity graph with node and edge attributes.

    Raises
    ------
    IndexError
        If edge references a node index that exceeds bin_centers length.
    """
    n_bins = len(bin_centers)
    n_dims = bin_centers.shape[1] if bin_centers.ndim > 1 else 1

    graph = nx.Graph()

    # Add nodes with required attributes
    for i in range(n_bins):
        pos = tuple(bin_centers[i])
        graph.add_node(
            i,
            pos=pos,
            source_grid_flat_index=i,
            original_grid_nd_index=(i,),  # Simplified for reconstructed graph
        )

    # Add edges with attributes
    for idx, (u, v) in enumerate(edges):
        u, v = int(u), int(v)
        distance = float(edge_weights[idx])

        # Compute vector between nodes
        pos_u = bin_centers[u]
        pos_v = bin_centers[v]
        vector = tuple(pos_v - pos_u)

        # Compute angle for 2D layouts
        angle_2d = None
        if n_dims == 2:
            angle_2d = float(np.arctan2(vector[1], vector[0]))

        graph.add_edge(
            u,
            v,
            distance=distance,
            vector=vector,
            edge_id=idx,
            angle_2d=angle_2d,
        )

    return graph


def _json_to_regions(regions_json: str) -> Regions:
    """
    Deserialize regions from JSON string.

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
        If a region has None data and is skipped during reconstruction.
    """
    from shapely.geometry import Polygon

    from neurospatial.regions import Region, Regions

    regions_dict = json.loads(regions_json)

    if not regions_dict:
        return Regions()

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


def _estimate_bin_size(bin_centers: NDArray[np.float64], n_dims: int) -> float:
    """
    Estimate bin size from bin centers.

    Uses the median distance between neighboring bin centers.

    Parameters
    ----------
    bin_centers : NDArray, shape (n_bins, n_dims)
        Bin center coordinates.
    n_dims : int
        Number of dimensions.

    Returns
    -------
    float
        Estimated bin size.
    """
    if len(bin_centers) < 2:
        return 1.0

    # Use KDTree to find nearest neighbors
    from scipy.spatial import KDTree

    tree = KDTree(bin_centers)
    # Query for 2 nearest neighbors (including self)
    distances, _ = tree.query(bin_centers, k=2)
    # Take the second column (distance to nearest non-self neighbor)
    nearest_distances = distances[:, 1]
    # Use median as robust estimate
    return float(np.median(nearest_distances))


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
