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

import contextlib
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import networkx as nx
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from neurospatial import Environment

# Type alias for file paths (supports both str and pathlib.Path)
PathLike = str | Path

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
        # Only convert to a numpy array when the result is numeric (int/float/
        # bool/complex). A list of strings (or other objects) would otherwise
        # become a dtype=object array, silently corrupting non-numeric layout
        # parameters (e.g. dimension labels). Such lists are recursed into and
        # left as Python lists instead.
        try:
            arr = np.asarray(obj)
        except (ValueError, TypeError):
            return [_convert_lists_to_arrays(item) for item in obj]
        if arr.dtype.kind in "biufc":  # bool, int, uint, float, complex
            return arr
        # Non-numeric (e.g. strings, ragged/object): recurse element-wise.
        return [_convert_lists_to_arrays(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: _convert_lists_to_arrays(v) for k, v in obj.items()}
    else:
        return obj


def _jsonsafe_layout_parameters(params: dict[str, Any] | None) -> dict[str, Any]:
    """Return a JSON-serializable copy of layout-engine parameters.

    ``env.layout_parameters`` may contain objects that ``json`` cannot encode:
    a ``networkx.Graph`` (Graph layout) or a Shapely geometry (Polygon layout).
    These are converted to portable, self-describing forms so the metadata JSON
    round-trips. The geometry itself is restored on read from the stored
    ``bin_centers``/``active_mask``/``grid_edges`` arrays and the node-link
    ``graph`` (see ``from_file``); these serialized parameters are introspection
    metadata, not the source of truth for reconstruction.

    Parameters
    ----------
    params : dict or None
        The ``env.layout_parameters`` dict (or None).

    Returns
    -------
    dict
        A new dict with every value JSON-serializable.
    """
    if not params:
        return {}

    def _encode(value: Any) -> Any:
        # networkx graph -> node-link dict (same form used for connectivity)
        if isinstance(value, nx.Graph):
            return {"__nx_graph__": nx.node_link_data(value, edges="links")}
        # shapely geometry -> WKT string (round-trippable, human-readable)
        wkt = getattr(value, "wkt", None)
        geom_type = getattr(value, "geom_type", None)
        if wkt is not None and geom_type is not None:
            return {"__shapely_wkt__": wkt}
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, dict):
            return {k: _encode(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_encode(v) for v in value]
        return value

    return {k: _encode(v) for k, v in params.items()}


def _decode_layout_parameters(params: Any) -> Any:
    """Inverse of ``_jsonsafe_layout_parameters`` for the sentinel dicts."""
    if isinstance(params, dict):
        if set(params) == {"__nx_graph__"}:
            return nx.node_link_graph(params["__nx_graph__"], edges="links")
        if set(params) == {"__shapely_wkt__"}:
            try:
                from shapely import wkt as _wkt

                return _wkt.loads(params["__shapely_wkt__"])
            except ImportError:
                # shapely absent at load time: leave the WKT string in place
                # rather than fail; geometry is reconstructed from arrays anyway.
                return params["__shapely_wkt__"]
        decoded = {k: _decode_layout_parameters(v) for k, v in params.items()}
        # The Graph layout's ``edge_order`` is a list of node-id tuples in the
        # original parameters; the generic numeric-list converter turns it into
        # a 2-D ndarray on read, which the Graph engine's ``build`` cannot index
        # (it expects an iterable of edge tuples). Restore the list-of-tuples
        # form so the round-tripped parameters reconstruct an equivalent env.
        edge_order = decoded.get("edge_order")
        if isinstance(edge_order, np.ndarray):
            decoded["edge_order"] = [tuple(e) for e in edge_order.tolist()]
        return decoded
    if isinstance(params, list):
        return [_decode_layout_parameters(v) for v in params]
    return params


def _get_library_version() -> str:
    """Get the neurospatial library version."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("neurospatial")
    except PackageNotFoundError:
        return "unknown"


def _validate_path_safety(path: PathLike) -> Path:
    """Validate that a path is safe from directory traversal attacks.

    Parameters
    ----------
    path : str or Path
        Path to validate.

    Returns
    -------
    Path
        Resolved Path object if safe.

    Raises
    ------
    ValueError
        If path contains parent directory traversal ('..' components).

    Notes
    -----
    This function prevents path traversal attacks by checking for '..'
    components in the path before resolution. It helps ensure that file
    operations stay within intended directories.

    """
    # Convert to Path but don't resolve yet
    path_obj = Path(path)

    # Check if any part of the ORIGINAL path contains parent directory reference
    # Must check before resolve() because resolve() normalizes away the '..'
    if ".." in path_obj.parts:
        raise ValueError(
            f"[E1005] Path traversal detected in path: {path}. "
            f"Use absolute paths or paths without '..' components. "
            f"This restriction helps prevent security vulnerabilities.\n\n"
            f"For more information, see: "
            f"https://edeno.github.io/neurospatial/errors/#e1005-path-traversal-detected"
        )

    # Now resolve to absolute path
    path_obj = path_obj.resolve()

    return path_obj


def to_file(env: Environment, path: PathLike, *, overwrite: bool = False) -> None:
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
        Paths containing '..' components are rejected to prevent
        directory traversal attacks.
    overwrite : bool, default=False
        If ``False`` (the default), raise :class:`FileExistsError` when either
        ``{path}.json`` or ``{path}.npz`` already exists, so an accidental
        re-run cannot silently clobber a saved environment. Pass ``True`` to
        replace existing files.

    Raises
    ------
    ValueError
        If path contains parent directory traversal ('..' components).
    FileExistsError
        If ``overwrite`` is ``False`` and a target file already exists.

    Examples
    --------
    >>> env = Environment.from_samples(data, bin_size=2.0)  # doctest: +SKIP
    >>> env.to_file("my_environment")  # doctest: +SKIP

    See Also
    --------
    from_file : Load environment from saved files

    Notes
    -----
    The JSON + npz format does not execute arbitrary code at load time and
    is portable across Python versions and platforms.

    For security, paths are validated to prevent directory traversal attacks.
    Use absolute paths or relative paths without '..' components.

    """
    # Validate path for security
    path_obj = _validate_path_safety(path)
    json_path = path_obj.with_suffix(".json")
    npz_path = path_obj.with_suffix(".npz")

    # Refuse to clobber an existing environment unless explicitly allowed. The
    # atomic .tmp + replace write below protects against a partial save, not
    # against overwriting the wrong target -- without this guard a single
    # accidental re-run silently destroys a saved env (and any hand-placed
    # regions), with no way to recover it.
    if not overwrite:
        existing = [str(p) for p in (json_path, npz_path) if p.exists()]
        if existing:
            raise FileExistsError(
                f"Refusing to overwrite existing environment file(s): "
                f"{', '.join(existing)}. Pass overwrite=True to replace them, "
                "or choose a different path."
            )

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
        "is_linearized_track": bool(env.is_linearized_track),
        "layout_type": env.layout_type,
        "layout_parameters": _jsonsafe_layout_parameters(env.layout_parameters),
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
    # Persist a "coordinate_kind" marker for egocentric polar environments so
    # the loader can restore the EgocentricPolarEnvironment type. Cartesian
    # envs omit the key (loaders treat a missing key as "cartesian"), keeping
    # the on-disk format stable for non-polar envs.
    if getattr(env, "_POLAR", False):
        metadata["coordinate_kind"] = "polar"

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

    # Write both files atomically: stage to temp files in the same directory,
    # then Path.replace() both into place only after BOTH writes succeed. This
    # prevents a partial save (e.g. a .json with no matching .npz) if the npz
    # write fails. Both temp files are cleaned up on any failure.
    json_tmp = json_path.with_name(json_path.name + ".tmp")
    # numpy.savez_compressed appends ".npz" if the path lacks that suffix, so
    # the staging name must already end in ".npz" to control it exactly.
    npz_tmp = npz_path.with_name(npz_path.stem + ".tmp.npz")
    try:
        with json_tmp.open("w") as f:
            json.dump(metadata, f, indent=2)
        # numpy.savez_compressed has overly strict type stubs - cast to work around
        np.savez_compressed(str(npz_tmp), **cast("Any", arrays_to_save))

        # Both staged successfully; move into final positions.
        json_tmp.replace(json_path)
        npz_tmp.replace(npz_path)
    except BaseException:
        # Clean up any staged temp files so no partial state is left behind.
        for tmp in (json_tmp, npz_tmp):
            with contextlib.suppress(FileNotFoundError):
                tmp.unlink()
        raise


def from_file(path: PathLike) -> Environment:
    """Load Environment from a versioned JSON + npz file pair.

    Parameters
    ----------
    path : str or Path
        Base path to load from (without extension).
        Will read `{path}.json` and `{path}.npz`.
        Paths containing '..' components are rejected to prevent
        directory traversal attacks.

    Returns
    -------
    Environment
        Reconstructed Environment instance.

    Raises
    ------
    FileNotFoundError
        If required files are not found.
    ValueError
        If schema version is incompatible, data is malformed, or path
        contains parent directory traversal ('..' components).

    Examples
    --------
    >>> env = from_file("my_environment")  # doctest: +SKIP
    >>> print(env.n_bins)  # doctest: +SKIP

    See Also
    --------
    to_file : Save environment to files

    Notes
    -----
    For security, paths are validated to prevent directory traversal attacks.
    Use absolute paths or relative paths without '..' components.

    """
    from neurospatial.environment import Environment
    from neurospatial.regions import Region, Regions

    # Validate path for security
    path_obj = _validate_path_safety(path)
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
            category=UserWarning,
        )

    # Load arrays. allow_pickle=False is safe because to_file only stores plain
    # numeric arrays (bin_centers, active_mask, grid_edges_*); it also blocks
    # arbitrary-code execution from a tampered .npz. Use a context manager so
    # the underlying file handle is closed promptly.
    with np.load(npz_path, allow_pickle=False) as arrays:
        # bin_centers: enforce float64 to match the in-memory from_dict path,
        # regardless of the dtype stored on disk.
        bin_centers = np.asarray(arrays["bin_centers"], dtype=np.float64)

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

        # Reconstruct active_mask (read out of the NpzFile before it closes)
        active_mask = arrays.get("active_mask", None)

    # Reconstruct graph
    graph_data = metadata["graph"]
    connectivity = nx.node_link_graph(graph_data, edges="links")

    # Reconstruct dimension_ranges
    dimension_ranges = None
    if "dimension_ranges" in metadata:
        dimension_ranges = [tuple(r) for r in metadata["dimension_ranges"]]

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

    # Decode any JSON-safe sentinels written by _jsonsafe_layout_parameters so
    # round-tripped layout_parameters is equivalent to the original.
    layout_params = _decode_layout_parameters(layout_params)

    # Create environment from layout. Polar environments (coordinate_kind
    # "polar") restore as the distinct EgocentricPolarEnvironment type; the
    # saved connectivity (overridden below) already carries the corrected
    # physical polar edge distances, so no geometry recomputation is needed.
    if metadata.get("coordinate_kind") == "polar":
        from neurospatial.environment.polar import EgocentricPolarEnvironment

        env = EgocentricPolarEnvironment.from_layout(
            layout_type, layout_params, name=metadata["name"]
        )
    else:
        env = Environment.from_layout(layout_type, layout_params, name=metadata["name"])

    # Override attributes with saved values (handles cases where layout recreation differs)
    env.bin_centers = bin_centers
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
    >>> env = Environment.from_samples(data, bin_size=2.0)  # doctest: +SKIP
    >>> env_dict = to_dict(env)  # doctest: +SKIP
    >>> # Pass to another process or serialize to JSON
    >>> import json  # doctest: +SKIP
    >>> json_str = json.dumps(env_dict)  # doctest: +SKIP

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
        "is_linearized_track": bool(env.is_linearized_track),
        "layout_type": env.layout_type,
        "layout_parameters": _jsonsafe_layout_parameters(env.layout_parameters),
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
    # Persist a "coordinate_kind" marker for egocentric polar environments so
    # the loader can restore the EgocentricPolarEnvironment type. Cartesian
    # envs omit the key (loaders treat a missing key as "cartesian"), keeping
    # the on-disk format stable for non-polar envs.
    if getattr(env, "_POLAR", False):
        metadata["coordinate_kind"] = "polar"

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
    >>> env_dict = to_dict(env)  # doctest: +SKIP
    >>> env_restored = from_dict(env_dict)  # doctest: +SKIP

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
            category=UserWarning,
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

    # Decode any JSON-safe sentinels written by _jsonsafe_layout_parameters so
    # round-tripped layout_parameters is equivalent to the original.
    layout_params = _decode_layout_parameters(layout_params)

    # Create environment. Polar environments restore as the distinct
    # EgocentricPolarEnvironment type; the saved connectivity (overridden
    # below) already carries the corrected physical polar edge distances.
    if data.get("coordinate_kind") == "polar":
        from neurospatial.environment.polar import EgocentricPolarEnvironment

        env = EgocentricPolarEnvironment.from_layout(
            layout_type, layout_params, name=data["name"]
        )
    else:
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
