"""
Environment serialization to/from NWB scratch space.

This module provides functions for writing and reading Environment objects
to NWB scratch/ space using standard NWB types (no custom extension required).
"""

from __future__ import annotations

import json
import warnings
from typing import TYPE_CHECKING, Any, TypedDict, cast

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from neurospatial.io.nwb._core import _require_pynwb, logger

if TYPE_CHECKING:
    from pynwb import NWBFile

    from neurospatial import Environment
    from neurospatial.regions import Regions


# =============================================================================
# Type definitions for JSON metadata structures
# =============================================================================


class EnvironmentMetadata(TypedDict):
    """
    Type definition for environment metadata JSON structure.

    This TypedDict defines the schema for environment metadata stored in NWB
    scratch space. The schema_version field enables future migrations.
    """

    schema_version: str
    name: str
    units: str
    frame: str
    n_dims: int
    layout_type: str
    n_bins: int
    n_edges: int
    is_1d: bool
    has_grid_data: bool


class GridData(TypedDict, total=False):
    """
    Type definition for grid structure data.

    This TypedDict defines the schema for grid-based layout data used to
    reconstruct point_to_bin_index functionality after NWB round-trip.

    Note: active_mask_flat is stored in a separate DynamicTable column (not JSON)
    to avoid JSON serialization of large boolean arrays. It's added to this dict
    during deserialization for convenience.
    """

    grid_edges: list[list[float]]  # Bin edges per dimension
    grid_shape: list[int]  # Grid dimensions
    total_grid_cells: int  # Total cells for padding calculation
    active_mask_flat: NDArray[np.bool_]  # Added during deserialization (not in JSON)


# =============================================================================
# Column name constants for DynamicTable
# =============================================================================

# These constants reduce magic strings and centralize column naming
COL_BIN_CENTERS = "bin_centers"
COL_EDGES = "edges"
COL_EDGE_WEIGHTS = "edge_weights"
COL_DIMENSION_RANGES = "dimension_ranges"
COL_REGIONS = "regions"
COL_METADATA = "metadata"
COL_GRID_DATA = "grid_data"
COL_ACTIVE_MASK = "active_mask"


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
        If environment is not fitted, or if environment with same name
        exists and overwrite=False.
    ImportError
        If pynwb is not installed.

    Notes
    -----
    **DynamicTable row alignment requirement:**

    HDMF's DynamicTable requires all columns to have the same number of rows.
    Since our data has varying lengths (n_bins bin centers, n_edges edges,
    n_dims dimension ranges), we pad all arrays to a uniform ``n_rows`` and
    store the actual valid counts in the metadata JSON.

    **Why JSON strings are repeated in every row:**

    Scalar metadata (regions JSON, metadata JSON, grid_data JSON) cannot be
    stored as single-row columns because DynamicTable enforces row alignment.
    HDMF does not support broadcasting from length-1 columns. Therefore, we
    repeat the identical JSON string in every row. This is intentional and
    expected - during deserialization, we read only the first row (index 0).

    The metadata JSON includes a ``schema_version`` field (currently "1.0")
    to enable future format migrations without breaking existing files.

    Examples
    --------
    >>> from pynwb import NWBHDF5IO  # doctest: +SKIP
    >>> with NWBHDF5IO("session.nwb", "r+") as io:  # doctest: +SKIP
    ...     nwbfile = io.read()
    ...     write_environment(nwbfile, env, name="linear_track")
    ...     io.write(nwbfile)
    """
    _require_pynwb()
    from hdmf.common import DynamicTable, VectorData

    # Validate environment is fitted
    if not env._is_fitted:
        raise ValueError(
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

    # Extract grid data for grid-based layouts (needed for proper point_to_bin_index)
    grid_data = _extract_grid_data(env)

    # Serialize extra metadata (include array lengths for deserialization)
    # schema_version enables future migrations if format changes
    metadata = json.dumps(
        {
            "schema_version": "1.0",  # For future format migrations
            "name": env.name,
            "units": units,
            "frame": frame,
            "n_dims": n_dims,
            "layout_type": layout_type,
            "n_bins": env.n_bins,
            "n_edges": len(edges),  # Needed for proper deserialization
            "is_1d": env.is_1d,  # Preserve 1D property for Graph layouts
            "has_grid_data": grid_data is not None,
        }
    )

    # Calculate n_rows considering grid data size
    grid_data_size = 0
    if grid_data is not None:
        # Grid data needs: n_dims edges arrays (flattened) + active_mask (flattened)
        grid_data_size = grid_data["total_grid_cells"]

    # Use DynamicTable with row-aligned data (pad shorter arrays to match n_bins)
    # This is required because DynamicTable needs uniform row counts
    n_rows = max(env.n_bins, len(edges), n_dims, grid_data_size, 1)

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

    # Prepare grid data columns (if available)
    # Remove active_mask_flat from JSON (stored in separate column)
    grid_data_for_json = None
    if grid_data is not None:
        grid_data_for_json = {
            k: v for k, v in grid_data.items() if k != "active_mask_flat"
        }
    grid_data_json = (
        json.dumps(grid_data_for_json) if grid_data_for_json is not None else "{}"
    )
    grid_data_list = [grid_data_json] * n_rows

    # Prepare active_mask column (flattened, padded)
    active_mask_padded = np.zeros(n_rows, dtype=np.int8)  # Use int8 for bool storage
    if grid_data is not None and grid_data["active_mask_flat"] is not None:
        mask_len = len(grid_data["active_mask_flat"])
        active_mask_padded[:mask_len] = grid_data["active_mask_flat"].astype(np.int8)

    # Create DynamicTable with padded columns
    table = DynamicTable(
        name=name,
        description=description,
        columns=[
            VectorData(
                name=COL_BIN_CENTERS,
                data=bin_centers_padded,
                description=f"Bin center coordinates, actual shape ({env.n_bins}, {n_dims})",
            ),
            VectorData(
                name=COL_EDGES,
                data=edges_padded,
                description=f"Edge list for connectivity graph, actual shape ({len(edges)}, 2)",
            ),
            VectorData(
                name=COL_EDGE_WEIGHTS,
                data=edge_weights_padded,
                description=f"Edge weights (distances), actual length {len(edge_weights)}",
            ),
            VectorData(
                name=COL_DIMENSION_RANGES,
                data=dim_ranges_padded,
                description=f"Min/max extent per dimension, actual shape ({n_dims}, 2)",
            ),
            VectorData(
                name=COL_REGIONS,
                data=regions_list,
                description="JSON-encoded regions (points and polygons)",
            ),
            VectorData(
                name=COL_METADATA,
                data=metadata_list,
                description="JSON-encoded metadata",
            ),
            VectorData(
                name=COL_GRID_DATA,
                data=grid_data_list,
                description="JSON-encoded grid structure (grid_edges, grid_shape) for layout reconstruction",
            ),
            VectorData(
                name=COL_ACTIVE_MASK,
                data=active_mask_padded,
                description="Flattened active bin mask for grid-based layouts",
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


def _extract_grid_data(env: Environment) -> dict[str, Any] | None:
    """
    Extract grid data from environment layout for serialization.

    For grid-based layouts (RegularGrid, MaskedGrid, ImageMask, ShapelyPolygon),
    extracts the grid_edges, grid_shape, and active_mask needed
    to reconstruct the layout with proper point_to_bin_index support.

    Parameters
    ----------
    env : Environment
        The environment to extract grid data from.

    Returns
    -------
    dict or None
        Dictionary containing grid structure data if the layout is grid-based,
        None otherwise. Dict contains:
        - grid_edges: List of lists (bin edges per dimension)
        - grid_shape: Tuple of ints (grid dimensions)
        - active_mask_flat: Flattened boolean mask (stored separately as int8)
        - total_grid_cells: Total number of cells for padding calculation
    """
    # Defensive check for environments that might not have a layout
    if env.layout is None:
        return None  # type: ignore[unreachable]

    # Don't extract grid data for 1D layouts (Graph) - they have 1D grid structure
    # but N-D bin_centers, which would cause shape mismatches on reconstruction
    is_1d = getattr(env.layout, "is_1d", False)
    if is_1d:
        return None

    # Check if layout has grid_edges and active_mask (grid-based layouts)
    grid_edges = getattr(env.layout, "grid_edges", None)
    grid_shape = getattr(env.layout, "grid_shape", None)
    active_mask = getattr(env.layout, "active_mask", None)

    # Only extract grid data for layouts with non-empty grid_edges
    # (Hexagonal and TriangularMesh have empty grid_edges tuple)
    if grid_edges is None or grid_shape is None or active_mask is None:
        return None
    if len(grid_edges) == 0:
        return None  # Not a proper grid layout

    # Grid dimensionality must match bin_centers dimensionality
    # This catches cases where grid_shape is 1D but bin_centers are N-D
    n_grid_dims = len(grid_shape)
    n_bin_dims = env.bin_centers.shape[1] if env.bin_centers.ndim > 1 else 1
    if n_grid_dims != n_bin_dims:
        return None

    # Convert grid_edges tuple of arrays to list of lists for JSON
    grid_edges_list = [edge.tolist() for edge in grid_edges]

    return {
        "grid_edges": grid_edges_list,
        "grid_shape": list(grid_shape),
        "active_mask_flat": active_mask.ravel(),  # Stored in separate column, NOT in JSON
        "total_grid_cells": int(np.prod(grid_shape)),
    }


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
    Rebuilds connectivity graph and regions. For grid-based layouts,
    fully reconstructs the layout with proper point_to_bin_index support.

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
    For grid-based layouts (RegularGrid, MaskedGrid, ImageMask, ShapelyPolygon,
    Hexagonal), the layout is fully reconstructed from stored grid_edges and
    active_mask, enabling proper point_to_bin_index functionality.

    For non-grid layouts (Graph, TriangularMesh), a KDTree-based layout is used
    which provides nearest-neighbor point mapping.

    Examples
    --------
    >>> from pynwb import NWBHDF5IO  # doctest: +SKIP
    >>> with NWBHDF5IO("session.nwb", "r") as io:  # doctest: +SKIP
    ...     nwbfile = io.read()
    ...     env = read_environment(nwbfile, name="linear_track")
    """
    _require_pynwb()

    # Check if environment exists
    if name not in nwbfile.scratch:
        available = list(nwbfile.scratch.keys()) if nwbfile.scratch else []
        raise KeyError(
            f"Environment '{name}' not found in scratch/. Available: {available}"
        )

    scratch_data = nwbfile.scratch[name]

    # Parse metadata JSON (get first row - all rows have same value)
    metadata_json = scratch_data[COL_METADATA][0]
    metadata = cast("EnvironmentMetadata", json.loads(metadata_json))

    # Check schema version for forward compatibility
    schema_version = metadata.get("schema_version", "1.0")
    if schema_version != "1.0":
        logger.warning(
            "Environment '%s' has schema_version '%s', expected '1.0'. "
            "Attempting to read with current schema.",
            name,
            schema_version,
        )

    n_bins = metadata["n_bins"]
    n_edges = metadata["n_edges"]
    n_dims = metadata["n_dims"]
    env_name = metadata.get("name", "")
    units = metadata.get("units")
    frame = metadata.get("frame")
    layout_type = metadata.get("layout_type", "RegularGrid")
    has_grid_data = metadata.get("has_grid_data", False)

    # Extract arrays (truncate padding)
    bin_centers = np.array(scratch_data[COL_BIN_CENTERS][:n_bins], dtype=np.float64)
    edges = np.array(scratch_data[COL_EDGES][:n_edges], dtype=np.int64)
    edge_weights = np.array(scratch_data[COL_EDGE_WEIGHTS][:n_edges], dtype=np.float64)
    dimension_ranges_raw = np.array(
        scratch_data[COL_DIMENSION_RANGES][:n_dims], dtype=np.float64
    )
    dimension_ranges = [tuple(row) for row in dimension_ranges_raw]

    # Parse regions JSON
    regions_json = scratch_data[COL_REGIONS][0]
    regions = _json_to_regions(regions_json)

    # Parse grid data if available
    grid_data: GridData | None = None
    if has_grid_data and COL_GRID_DATA in scratch_data.colnames:
        grid_data_json = scratch_data[COL_GRID_DATA][0]
        grid_data = json.loads(grid_data_json)

        # Reconstruct active_mask from separate column
        if grid_data and COL_ACTIVE_MASK in scratch_data.colnames:
            total_grid_cells = grid_data.get("total_grid_cells", 0)
            if total_grid_cells > 0:
                active_mask_flat = np.array(
                    scratch_data[COL_ACTIVE_MASK][:total_grid_cells], dtype=np.bool_
                )
                grid_data["active_mask_flat"] = active_mask_flat

    # Create environment with proper layout reconstruction
    env = _reconstruct_environment(
        bin_centers=bin_centers,
        edges=edges,
        edge_weights=edge_weights,
        dimension_ranges=dimension_ranges,
        regions=regions,
        env_name=env_name,
        layout_type=layout_type,
        grid_data=grid_data,
        n_dims=n_dims,
    )

    # Set metadata
    if units and units != "unknown":
        env.units = units
    if frame and frame != "unknown":
        env.frame = frame

    # Store layout type info
    env._layout_type_used = layout_type

    # Restore is_1d property for Graph layouts
    is_1d = metadata.get("is_1d", False)
    if is_1d:
        env._is_1d_env = True

    logger.debug(
        "Read environment '%s' with %d bins and %d edges",
        name,
        n_bins,
        n_edges,
    )

    return env


def _reconstruct_environment(
    bin_centers: NDArray[np.float64],
    edges: NDArray[np.int64],
    edge_weights: NDArray[np.float64],
    dimension_ranges: list[tuple[float, float]],
    regions: Regions,
    env_name: str,
    layout_type: str,
    grid_data: GridData | None,
    n_dims: int,
) -> Environment:
    """
    Reconstruct Environment with appropriate layout type.

    For grid-based layouts, uses MaskedGridLayout with stored grid_edges and
    active_mask to enable proper point_to_bin_index. For non-grid layouts,
    falls back to KDTree-based layout.

    Parameters
    ----------
    bin_centers : NDArray, shape (n_bins, n_dims)
        Bin center coordinates.
    edges : NDArray, shape (n_edges, 2)
        Edge list (node pairs).
    edge_weights : NDArray, shape (n_edges,)
        Edge weights (distances).
    dimension_ranges : list of tuple
        Min/max extent per dimension.
    regions : Regions
        Reconstructed regions.
    env_name : str
        Environment name.
    layout_type : str
        Original layout type tag.
    grid_data : dict or None
        Grid structure data (grid_edges, grid_shape, active_mask_flat).
    n_dims : int
        Number of dimensions.

    Returns
    -------
    Environment
        Reconstructed environment with proper layout.
    """
    from neurospatial import Environment
    from neurospatial.layout.engines.masked_grid import MaskedGridLayout

    # Reconstruct connectivity graph
    connectivity = _reconstruct_graph(bin_centers, edges, edge_weights)

    # Try to reconstruct with grid layout if grid data is available
    if grid_data is not None:
        grid_edges_list = grid_data.get("grid_edges")
        grid_shape = grid_data.get("grid_shape")
        active_mask_flat = grid_data.get("active_mask_flat")

        if (
            grid_edges_list is not None
            and grid_shape is not None
            and active_mask_flat is not None
        ):
            # Convert grid_edges back to tuple of arrays
            grid_edges = tuple(np.array(e, dtype=np.float64) for e in grid_edges_list)

            # Reshape active_mask from flat to N-D
            active_mask = active_mask_flat.reshape(tuple(grid_shape))

            # Create MaskedGridLayout with exact grid structure
            layout = MaskedGridLayout()
            layout.build(
                active_mask=active_mask,
                grid_edges=grid_edges,
                connect_diagonal_neighbors=True,  # Conservative default
            )

            # Create environment from layout
            env = Environment.from_layout(
                kind="MaskedGrid",
                layout_params={
                    "active_mask": active_mask,
                    "grid_edges": grid_edges,
                },
                name=env_name,
                regions=regions,
            )

            # Override connectivity with stored values (preserves edge weights)
            env.connectivity = connectivity
            env.dimension_ranges = dimension_ranges

            return env

    # Fallback: Create environment with KDTree-based layout for non-grid layouts
    # These layouts don't have grid structure, so we use a special reconstructed
    # layout that uses KDTree for point_to_bin_index
    reconstructed_layout = _ReconstructedLayout(
        bin_centers=bin_centers,
        connectivity=connectivity,
        dimension_ranges=dimension_ranges,
        layout_type=layout_type,
    )

    # Create Environment directly with the reconstructed layout
    # _ReconstructedLayout implements LayoutEngine protocol but mypy can't verify
    env = Environment(
        name=env_name,
        layout=reconstructed_layout,  # type: ignore[arg-type]
        regions=regions,
    )

    # Setup the environment from the layout (uses self.layout internally)
    env._setup_from_layout()

    return env


class _ReconstructedLayout:
    """
    A minimal layout for reconstructed environments that uses KDTree for point mapping.

    Used when reading non-grid environments (Graph, Hexagonal, TriangularMesh) from NWB.
    These layouts don't have proper grid structure after round-trip, so we use
    KDTree-based nearest neighbor mapping for point_to_bin_index.

    Notes
    -----
    The ``is_1d`` property always returns ``False`` even for layouts originally created
    from 1D Graph layouts. The original 1D property is preserved separately via
    ``env._is_1d_env`` attribute set during reconstruction. This means ``env.is_1d``
    will return the correct value, but ``env.layout.is_1d`` may differ.
    """

    def __init__(
        self,
        bin_centers: NDArray[np.float64],
        connectivity: nx.Graph,
        dimension_ranges: list[tuple[float, float]],
        layout_type: str,
    ) -> None:
        from scipy.spatial import KDTree

        self.bin_centers = bin_centers
        self.connectivity = connectivity
        self.dimension_ranges = dimension_ranges
        self._layout_type_tag = f"Reconstructed_{layout_type}"
        self._build_params_used = {"original_layout_type": layout_type}

        # Build KDTree for point mapping
        self._kdtree = KDTree(bin_centers) if len(bin_centers) > 0 else None

        # Grid-related attributes (set to None for non-grid layouts)
        self.grid_edges = None
        self.grid_shape = None
        self.active_mask = None

    @property
    def is_1d(self) -> bool:
        """Return False - reconstructed layouts are not 1D linearized."""
        return False

    def point_to_bin_index(self, points: NDArray[np.float64]) -> NDArray[np.intp]:
        """Map points to bin indices using KDTree nearest neighbor search."""
        points = np.atleast_2d(points)
        n_query_dims = points.shape[1]
        n_tree_dims = self.bin_centers.shape[1]

        if self._kdtree is None:
            return np.full(len(points), -1, dtype=np.intp)

        # Handle dimension mismatch
        if n_query_dims != n_tree_dims:
            # If query has fewer dims, pad with zeros
            if n_query_dims < n_tree_dims:
                padded = np.zeros((len(points), n_tree_dims), dtype=np.float64)
                padded[:, :n_query_dims] = points
                points = padded
            # If query has more dims, truncate (take first n_tree_dims)
            else:
                points = points[:, :n_tree_dims]

        _, indices = self._kdtree.query(points)
        return np.asarray(indices, dtype=np.intp)

    def bin_sizes(self) -> NDArray[np.float64]:
        """
        Return estimated bin volumes from nearest neighbor spacing.

        For reconstructed layouts, this computes an approximate "volume" per bin
        as ``spacing ** n_dims``, where spacing is the median nearest-neighbor
        distance. This is a volume estimate (e.g., area in 2D, length in 1D),
        not a linear bin size.

        Returns
        -------
        NDArray[np.float64], shape (n_bins,)
            Estimated bin volume for each bin.
        """
        if len(self.bin_centers) < 2 or self._kdtree is None:
            return np.ones(len(self.bin_centers))

        # Estimate from median nearest neighbor distance
        _, distances = self._kdtree.query(self.bin_centers, k=2)
        median_spacing = float(np.median(distances[:, 1]))
        return np.full(
            len(self.bin_centers), median_spacing ** self.bin_centers.shape[1]
        )

    def plot(self, ax: Any | None = None, **kwargs: Any) -> Any:
        """Basic plot method."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()

        if self.bin_centers.shape[1] >= 2:
            ax.scatter(self.bin_centers[:, 0], self.bin_centers[:, 1], **kwargs)
        elif self.bin_centers.shape[1] == 1:
            ax.scatter(
                self.bin_centers[:, 0], np.zeros(len(self.bin_centers)), **kwargs
            )

        return ax


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

    Notes
    -----
    **Default fallback behavior:**

    If the SpatialSeries does not have a ``unit`` attribute set (or it is None
    or empty string), this function returns ``"cm"`` as a sensible default for
    neuroscience tracking data.

    This is a silent fallback - no warning is emitted. If you need to know
    whether the units were auto-detected or defaulted, compare the returned
    value against your expected units, or access the SpatialSeries directly
    to check if ``unit`` is set.
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

    # Return units (default to "cm" if not set)
    return str(spatial_series.unit) if spatial_series.unit else "cm"
