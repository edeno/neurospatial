"""io.py - Stable serialization for Environment objects
========================================================

This module provides versioned JSON + npz serialization for Environment objects,
enabling reproducible workflows and cross-tool interoperability.

Schema
------
The serialization format uses:
- JSON for metadata, structure, and small arrays
- NumPy .npz for large numerical arrays (bin_centers, etc.)

Files are saved as a directory (or zip) containing:
- metadata.json: Schema version, library version, timestamps, parameters
- arrays.npz: Binary arrays (bin_centers, etc.)
- graph.json: NetworkX graph in node-link format
- regions.json: Regions data (if present)

"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import networkx as nx
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from neurospatial.environment import Environment

# Schema version for serialization format
_SCHEMA_VERSION = "Environment-v1"


def _convert_arrays_to_lists(obj: Any) -> Any:
    """Recursively convert numpy arrays to lists for JSON serialization.

    Parameters
    ----------
    obj : Any
        Object that may contain numpy arrays.

    Returns
    -------
    Any
        Object with numpy arrays converted to lists.

    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _convert_arrays_to_lists(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_arrays_to_lists(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_convert_arrays_to_lists(item) for item in obj)
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    else:
        return obj


def _convert_lists_to_arrays(obj: Any) -> Any:
    """Recursively convert lists to numpy arrays where appropriate.

    Parameters
    ----------
    obj : Any
        Object that may contain lists representing arrays.

    Returns
    -------
    Any
        Object with numeric lists converted to arrays.

    """
    if isinstance(obj, list):
        # Try to convert to array (will work for numeric lists)
        try:
            return np.array(obj)
        except (ValueError, TypeError):
            # Not a numeric list, recursively process elements
            return [_convert_lists_to_arrays(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: _convert_lists_to_arrays(v) for k, v in obj.items()}
    else:
        return obj


def _get_library_version() -> str:
    """Get the neurospatial library version."""
    try:
        from importlib.metadata import version

        return version("neurospatial")
    except Exception:
        return "unknown"


def to_file(env: Environment, path: str | Path) -> None:
    """Save Environment to a versioned JSON + npz file pair.

    Creates two files:
    - `path.json`: Metadata, graph structure, and small arrays
    - `path.npz`: Large numerical arrays (bin_centers, etc.)

    The format is stable across versions and supports forward/backward
    compatibility through schema versioning.

    Parameters
    ----------
    env : Environment
        Environment instance to save.
    path : str or Path
        Base path for output files (without extension).
        Will create `{path}.json` and `{path}.npz`.

    Examples
    --------
    >>> env = Environment.from_samples(data, bin_size=2.0)
    >>> env.to_file(
    ...     "my_environment"
    ... )  # Creates my_environment.json and my_environment.npz

    See Also
    --------
    from_file : Load environment from saved files
    Environment.save : Legacy pickle-based serialization (less safe)

    Notes
    -----
    This format is safer than pickle (no arbitrary code execution) and
    more portable across Python versions and platforms.

    """
    path_obj = Path(path)
    json_path = path_obj.with_suffix(".json")
    npz_path = path_obj.with_suffix(".npz")

    # Ensure parent directory exists
    json_path.parent.mkdir(parents=True, exist_ok=True)

    # Build metadata dictionary
    metadata: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "library_version": _get_library_version(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "name": env.name,
        "n_dims": int(env.n_dims),
        "n_bins": int(env.n_bins),
        "is_1d": bool(env.is_1d),
        "layout_type": env.layout_type,
        "layout_parameters": env.layout_parameters,
    }

    # Add optional attributes
    if env.dimension_ranges is not None:
        metadata["dimension_ranges"] = [
            [float(lo), float(hi)] for lo, hi in env.dimension_ranges
        ]

    if env.grid_shape is not None:
        metadata["grid_shape"] = [int(x) for x in env.grid_shape]

    # Add units and frame if present
    if hasattr(env, "units") and env.units is not None:
        metadata["units"] = env.units
    if hasattr(env, "frame") and env.frame is not None:
        metadata["frame"] = env.frame

    # Serialize graph to node-link format
    graph_data = nx.node_link_data(env.connectivity, edges="links")
    metadata["graph"] = graph_data

    # Serialize regions if present
    if env.regions and len(env.regions) > 0:
        metadata["regions"] = [reg.to_dict() for reg in env.regions.values()]
    else:
        metadata["regions"] = []

    # Convert entire metadata to JSON-safe format (must be done AFTER all modifications)
    metadata = _convert_arrays_to_lists(metadata)

    # Write JSON metadata
    with json_path.open("w") as f:
        json.dump(metadata, f, indent=2)

    # Prepare arrays for npz
    arrays_to_save: dict[str, NDArray] = {
        "bin_centers": env.bin_centers,
    }

    # Add optional arrays
    if env.active_mask is not None:
        arrays_to_save["active_mask"] = env.active_mask

    if env.grid_edges is not None and len(env.grid_edges) > 0:
        # Save grid edges as separate arrays (grid_edges_0, grid_edges_1, ...)
        for i, edges in enumerate(env.grid_edges):
            arrays_to_save[f"grid_edges_{i}"] = edges

    # Write npz arrays
    # numpy.savez_compressed has overly strict type stubs - cast to work around
    np.savez_compressed(str(npz_path), **cast("Any", arrays_to_save))


def from_file(path: str | Path) -> Environment:
    """Load Environment from a versioned JSON + npz file pair.

    Parameters
    ----------
    path : str or Path
        Base path to load from (without extension).
        Will read `{path}.json` and `{path}.npz`.

    Returns
    -------
    Environment
        Reconstructed Environment instance.

    Raises
    ------
    FileNotFoundError
        If required files are not found.
    ValueError
        If schema version is incompatible or data is malformed.

    Examples
    --------
    >>> env = from_file("my_environment")
    >>> print(env.n_bins)

    See Also
    --------
    to_file : Save environment to files
    Environment.load : Legacy pickle-based deserialization

    """
    from neurospatial.environment import Environment
    from neurospatial.regions import Region, Regions

    path_obj = Path(path)
    json_path = path_obj.with_suffix(".json")
    npz_path = path_obj.with_suffix(".npz")

    if not json_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {json_path}")
    if not npz_path.exists():
        raise FileNotFoundError(f"Array file not found: {npz_path}")

    # Load metadata
    with json_path.open("r") as f:
        metadata = json.load(f)

    # Check schema version
    schema_version = metadata.get("schema_version")
    if schema_version != _SCHEMA_VERSION:
        warnings.warn(
            f"Schema version mismatch: file has {schema_version!r}, "
            f"expected {_SCHEMA_VERSION!r}. Attempting to load anyway.",
            stacklevel=2,
        )

    # Load arrays
    arrays = np.load(npz_path)

    # Reconstruct graph
    graph_data = metadata["graph"]
    connectivity = nx.node_link_graph(graph_data, edges="links")

    # Reconstruct dimension_ranges
    dimension_ranges = None
    if "dimension_ranges" in metadata:
        dimension_ranges = [tuple(r) for r in metadata["dimension_ranges"]]

    # Reconstruct grid_edges from separate arrays
    grid_edges = None
    if "grid_shape" in metadata:
        n_dims = metadata["n_dims"]
        grid_edges_list = []
        for i in range(n_dims):
            key = f"grid_edges_{i}"
            if key in arrays:
                grid_edges_list.append(arrays[key])
        if grid_edges_list:
            grid_edges = tuple(grid_edges_list)

    # Reconstruct active_mask
    active_mask = arrays.get("active_mask")

    # Reconstruct grid_shape
    grid_shape = None
    if "grid_shape" in metadata:
        grid_shape = tuple(metadata["grid_shape"])

    # Create layout engine from parameters
    # Note: We use from_layout() pattern to reconstruct
    layout_type = metadata["layout_type"]
    layout_params = metadata["layout_parameters"]

    # Convert lists back to numpy arrays in layout parameters
    layout_params = _convert_lists_to_arrays(layout_params)

    # Create environment from layout
    env = Environment.from_layout(layout_type, layout_params, name=metadata["name"])

    # Override attributes with saved values (handles cases where layout recreation differs)
    env.bin_centers = arrays["bin_centers"]
    env.connectivity = connectivity
    env.dimension_ranges = dimension_ranges
    env.grid_edges = grid_edges
    env.grid_shape = grid_shape
    env.active_mask = active_mask

    # Reconstruct regions
    if metadata.get("regions"):
        regions_list = [Region.from_dict(r) for r in metadata["regions"]]
        env.regions = Regions(regions_list)

    # Restore units and frame if present
    if "units" in metadata:
        env.units = metadata["units"]
    if "frame" in metadata:
        env.frame = metadata["frame"]

    return env


def to_dict(env: Environment) -> dict[str, Any]:
    """Convert Environment to a dictionary for in-memory handoff.

    This is useful for passing environments between processes or for
    temporary storage without writing to disk.

    Parameters
    ----------
    env : Environment
        Environment instance to convert.

    Returns
    -------
    dict[str, Any]
        Dictionary representation of the environment.
        All arrays are converted to lists for JSON compatibility.

    Notes
    -----
    For large environments, prefer `to_file()` which uses efficient
    binary serialization for arrays.

    Examples
    --------
    >>> env = Environment.from_samples(data, bin_size=2.0)
    >>> env_dict = to_dict(env)
    >>> # Pass to another process or serialize to JSON
    >>> import json
    >>> json_str = json.dumps(env_dict)

    See Also
    --------
    from_dict : Reconstruct environment from dictionary
    to_file : Save to disk with efficient binary format

    """
    # Similar to to_file but with arrays as lists
    metadata: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "library_version": _get_library_version(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "name": env.name,
        "n_dims": int(env.n_dims),
        "n_bins": int(env.n_bins),
        "is_1d": bool(env.is_1d),
        "layout_type": env.layout_type,
        "layout_parameters": env.layout_parameters,
        "bin_centers": env.bin_centers.tolist(),
    }

    # Add optional attributes
    if env.dimension_ranges is not None:
        metadata["dimension_ranges"] = [
            [float(lo), float(hi)] for lo, hi in env.dimension_ranges
        ]

    if env.grid_shape is not None:
        metadata["grid_shape"] = [int(x) for x in env.grid_shape]

    if env.active_mask is not None:
        metadata["active_mask"] = env.active_mask.tolist()

    if env.grid_edges is not None and len(env.grid_edges) > 0:
        metadata["grid_edges"] = [edges.tolist() for edges in env.grid_edges]

    # Add units and frame if present
    if hasattr(env, "units") and env.units is not None:
        metadata["units"] = env.units
    if hasattr(env, "frame") and env.frame is not None:
        metadata["frame"] = env.frame

    # Serialize graph
    graph_data = nx.node_link_data(env.connectivity, edges="links")
    metadata["graph"] = graph_data

    # Serialize regions
    if env.regions and len(env.regions) > 0:
        metadata["regions"] = [reg.to_dict() for reg in env.regions.values()]
    else:
        metadata["regions"] = []

    # Convert entire metadata to JSON-safe format
    metadata = _convert_arrays_to_lists(metadata)

    return metadata


def from_dict(data: dict[str, Any]) -> Environment:
    """Reconstruct Environment from dictionary representation.

    Parameters
    ----------
    data : dict[str, Any]
        Dictionary representation from `to_dict()`.

    Returns
    -------
    Environment
        Reconstructed Environment instance.

    Examples
    --------
    >>> env_dict = to_dict(env)
    >>> env_restored = from_dict(env_dict)

    See Also
    --------
    to_dict : Convert environment to dictionary
    from_file : Load from disk files

    """
    from neurospatial.environment import Environment
    from neurospatial.regions import Region, Regions

    # Check schema version
    schema_version = data.get("schema_version")
    if schema_version != _SCHEMA_VERSION:
        warnings.warn(
            f"Schema version mismatch: data has {schema_version!r}, "
            f"expected {_SCHEMA_VERSION!r}. Attempting to load anyway.",
            stacklevel=2,
        )

    # Reconstruct arrays
    bin_centers = np.array(data["bin_centers"], dtype=np.float64)

    # Reconstruct graph
    graph_data = data["graph"]
    connectivity = nx.node_link_graph(graph_data, edges="links")

    # Reconstruct dimension_ranges
    dimension_ranges = None
    if "dimension_ranges" in data:
        dimension_ranges = [tuple(r) for r in data["dimension_ranges"]]

    # Reconstruct grid_edges
    grid_edges = None
    if "grid_edges" in data:
        grid_edges = tuple(np.array(e, dtype=np.float64) for e in data["grid_edges"])

    # Reconstruct active_mask
    active_mask = None
    if "active_mask" in data:
        active_mask = np.array(data["active_mask"], dtype=bool)

    # Reconstruct grid_shape
    grid_shape = None
    if "grid_shape" in data:
        grid_shape = tuple(data["grid_shape"])

    # Create layout and environment
    layout_type = data["layout_type"]
    layout_params = data["layout_parameters"]

    # Convert lists back to numpy arrays in layout parameters
    layout_params = _convert_lists_to_arrays(layout_params)

    # Create environment
    env = Environment.from_layout(layout_type, layout_params, name=data["name"])

    # Override attributes
    env.bin_centers = bin_centers
    env.connectivity = connectivity
    env.dimension_ranges = dimension_ranges
    env.grid_edges = grid_edges
    env.grid_shape = grid_shape
    env.active_mask = active_mask

    # Reconstruct regions
    if data.get("regions"):
        regions_list = [Region.from_dict(r) for r in data["regions"]]
        env.regions = Regions(regions_list)

    # Restore units and frame if present
    if "units" in data:
        env.units = data["units"]
    if "frame" in data:
        env.frame = data["frame"]

    return env
