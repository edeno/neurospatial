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
import zlib
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from zipfile import BadZipFile

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error

if TYPE_CHECKING:
    from neurospatial import Environment
    from neurospatial.environment.core import _BaseEnvironment

# Type alias for file paths (supports both str and pathlib.Path)
PathLike = str | Path

# Schema version for serialization format
_SCHEMA_VERSION = "Environment-v3"


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
    elif isinstance(obj, Mapping):
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
            from neurospatial.layout._codecs import graph_to_node_link

            return {"__nx_graph__": graph_to_node_link(value)}
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
            from neurospatial.layout._codecs import graph_from_node_link

            return graph_from_node_link(params["__nx_graph__"])
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


def _environment_file_paths(path: Path) -> tuple[Path, Path]:
    """Append storage extensions while preserving dots in the path prefix."""
    if path.suffix.lower() in {".json", ".npz"}:
        path = path.with_suffix("")
    return path.with_name(path.name + ".json"), path.with_name(path.name + ".npz")


def to_file(env: Environment, path: PathLike, *, overwrite: bool = False) -> None:
    """Save Environment to a versioned JSON + npz file pair.

    Creates two files:
    - `path.json`: Metadata, graph structure, and small arrays
    - `path.npz`: Large numerical arrays (bin_centers, etc.)

    The v3 format stores bins, lookup, cell faces, drawing components and Arena.
    Loading decodes these data without rebuilding a layout. Version 2 remains
    readable with an explicitly unknown Arena; version 1 must be recreated.

    Parameters
    ----------
    env : Environment
        Environment instance to save.
    path : str or Path
        Prefix for output files. Creates ``{path}.json`` and ``{path}.npz``.
        A trailing ``.json`` or ``.npz`` may be supplied; other dots remain
        part of the prefix, so ``rat1.day1`` creates ``rat1.day1.json`` and
        ``rat1.day1.npz``.
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
    json_path, npz_path = _environment_file_paths(path_obj)

    from neurospatial._validation import check_writable

    check_writable(
        [json_path, npz_path],
        overwrite=overwrite,
        what="environment file(s)",
        argument="path",
    )

    # Ensure parent directory exists
    json_path.parent.mkdir(parents=True, exist_ok=True)

    metadata, arrays_to_save = _environment_payload(env)

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
    """Load binning and Arena data without running a layout builder.

    Version 2 files retain their bins with an explicitly unknown Arena. Version 1
    files are refused; recreate and re-save them with a factory.

    Parameters
    ----------
    path : str or Path
        Prefix to load from. Reads ``{path}.json`` and ``{path}.npz``.
        A trailing ``.json`` or ``.npz`` may be supplied; other dots remain
        part of the prefix.
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
    # Validate path for security
    path_obj = _validate_path_safety(path)
    json_path, npz_path = _environment_file_paths(path_obj)

    if not json_path.exists():
        raise FileNotFoundError(
            _format_error(
                f"Metadata file not found: {json_path}",
                fix="save both files with env.to_file(path), then load using the same path prefix",
                why="Why: loading an environment requires both its .json metadata and .npz arrays.",
            )
        )
    if not npz_path.exists():
        raise FileNotFoundError(
            _format_error(
                f"Array file not found: {npz_path}",
                fix="save both files with env.to_file(path), then load using the same path prefix",
                why="Why: loading an environment requires both its .json metadata and .npz arrays.",
            )
        )

    try:
        with json_path.open("r") as f:
            metadata = json.load(f)
    except (json.JSONDecodeError, RecursionError, UnicodeDecodeError) as error:
        raise ValueError(
            _format_error(
                f"Invalid environment metadata in {json_path}: {error}",
                why="Why: the metadata must contain complete JSON without excessive nesting.",
                fix="restore the original JSON and NPZ file pair, or re-save the original environment",
            )
        ) from error

    _check_schema(metadata, str(json_path))
    try:
        archive = np.load(npz_path, allow_pickle=False)
        if not isinstance(archive, np.lib.npyio.NpzFile):
            raise ValueError(
                "expected an NPZ archive containing named numerical arrays"
            )
        with archive:
            arrays = {key: archive[key] for key in archive.files}
    except (BadZipFile, EOFError, ValueError, zlib.error) as error:
        raise ValueError(
            _format_error(
                f"Invalid environment array archive in {npz_path}: {error}",
                why="Why: the array file must be a complete NPZ archive matching its JSON metadata.",
                fix="restore the original JSON and NPZ file pair, or re-save the original environment",
            )
        ) from error
    return _restore_environment(
        metadata,
        source=str(json_path),
        arrays=arrays,
        legacy_arena=metadata["schema_version"] == "Environment-v2",
    )


def _environment_payload(
    env: Environment,
) -> tuple[dict[str, Any], dict[str, NDArray[Any]]]:
    """Build the shared lossless metadata and numerical-array payload."""
    from neurospatial.arena import encode_arena
    from neurospatial.layout._codecs import encode, graph_to_node_link

    binning_meta, binning_arrays = encode(env._binning, "binning")
    arena_meta, arena_arrays = encode_arena(env.arena)
    metadata: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "library_version": _get_library_version(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "name": env.name,
        "n_dims": env.n_dims,
        "n_bins": env.n_bins,
        "is_linearized_track": env.is_linearized_track,
        "layout_type": env.layout_type,
        "layout_category": env.layout.layout_type,
        "layout_parameters": _jsonsafe_layout_parameters(env.layout_parameters),
        "binning": binning_meta,
        "arena": arena_meta,
        "dimension_ranges": env.dimension_ranges,
        "grid_shape": env.grid_shape,
        "graph": graph_to_node_link(env.connectivity),
        "regions": [region.to_dict() for region in env.regions.values()],
        "coordinate_kind": "polar" if getattr(env, "_POLAR", False) else "cartesian",
    }
    for attribute in ("units", "frame"):
        value = getattr(env, attribute)
        if value is not None:
            metadata[attribute] = value
    arrays = {
        "bin_centers": env.bin_centers,
        **binning_arrays,
        **{f"arena/{key}": value for key, value in arena_arrays.items()},
    }
    if env.active_mask is not None:
        arrays["active_mask"] = env.active_mask
    for index, edges in enumerate(env.grid_edges or ()):
        arrays[f"grid_edges_{index}"] = edges
    metadata["array_specs"] = {
        key: {"shape": list(value.shape), "dtype": value.dtype.str}
        for key, value in arrays.items()
    }
    return cast("dict[str, Any]", _convert_arrays_to_lists(metadata)), arrays


def _check_schema(metadata: dict[str, Any], source: str) -> None:
    """Refuse obsolete or unknown file formats with a reproducible recovery."""
    if not isinstance(metadata, dict):
        raise ValueError(
            _format_error(
                f"Invalid environment metadata in {source}: expected a dictionary.",
                why="Why: environment metadata must map field names to saved values.",
                fix="restore a complete environment record saved by this version",
            )
        )
    version = metadata.get("schema_version")
    if version not in ("Environment-v2", _SCHEMA_VERSION):
        raise ValueError(
            _format_error(
                f"Unsupported environment schema {version!r} in {source}; expected {_SCHEMA_VERSION!r}.",
                why="Why: this version stores the binning itself and does not rebuild older layout records.",
                fix="recreate the environment with an Environment factory and save it again with this version",
            )
        )


def _restore_environment(
    metadata: dict[str, Any],
    *,
    source: str,
    arrays: dict[str, NDArray[Any]],
    legacy_arena: bool = False,
) -> Environment:
    """Decode authoritative binning and Arena data without a layout builder."""
    from neurospatial.arena import Arena, UnknownArena, decode_arena
    from neurospatial.environment import Environment
    from neurospatial.layout._binning import Binning, TrackLinearization
    from neurospatial.layout._binning_layout import BinningLayout
    from neurospatial.layout._codecs import decode, graph_from_node_link
    from neurospatial.layout._locators import MeshLocator
    from neurospatial.regions import Region, Regions

    try:
        for key, spec in metadata["array_specs"].items():
            array = arrays[key]
            if list(array.shape) != spec["shape"] or array.dtype.str != spec["dtype"]:
                raise ValueError(
                    f"array {key!r} does not match its stored shape or dtype"
                )
        binning = decode(metadata["binning"], arrays, "binning")
        if not isinstance(binning, Binning):
            raise ValueError("the binning payload is not a Binning")
        if (
            metadata["n_bins"] != binning.bins.n_bins
            or metadata["n_dims"] != binning.bins.n_dims
        ):
            raise ValueError("stored bin counts/dimensions differ from the binning")
        centers = arrays["bin_centers"]
        if not np.all(np.isfinite(centers)):
            raise ValueError("stored bin_centers contain non-finite coordinates")
        if not np.array_equal(centers, binning.bins.centers):
            raise ValueError(
                "stored bin_centers differ from the authoritative binning centers"
            )
        coordinate_kind = metadata.get("coordinate_kind", "cartesian")
        if coordinate_kind not in ("cartesian", "polar"):
            raise ValueError("unknown coordinate_kind")
        arena: Arena
        if legacy_arena:
            # Older records describe bins and drawing outlines, not physical walls.
            arena = UnknownArena(binning.bins.n_dims, coordinate_kind)
        else:
            arena_metadata = metadata["arena"]
            if not isinstance(arena_metadata, dict):
                raise ValueError("the arena record must be a metadata dictionary")
            arena = decode_arena(
                arena_metadata,
                {
                    key.removeprefix("arena/"): arrays[key]
                    for key in metadata["array_specs"]
                    if key.startswith("arena/")
                },
            )
            if arena.n_dims != binning.bins.n_dims:
                raise ValueError("stored arena dimensions differ from the binning")
            if arena.coordinate_kind != coordinate_kind:
                raise ValueError(
                    "stored arena coordinate_kind differs from the environment"
                )
        params = _decode_layout_parameters(
            _convert_lists_to_arrays(metadata["layout_parameters"])
        )
        ranges = metadata.get("dimension_ranges")
        layout = BinningLayout(
            binning,
            connectivity=graph_from_node_link(metadata["graph"]),
            dimension_ranges=None if ranges is None else [tuple(r) for r in ranges],
            layout_type_tag=metadata["layout_type"],
            layout_type=metadata["layout_category"],
            build_params=params,
            is_linearized_track=metadata["is_linearized_track"],
            coordinate_kind=coordinate_kind,
        )
        if (
            binning.rectilinear is not None
            or isinstance(binning.linearization, TrackLinearization)
            or isinstance(binning.locator, MeshLocator)
        ):
            if (
                "grid_shape" in metadata
                and metadata["grid_shape"] is not None
                and tuple(metadata["grid_shape"]) != layout.grid_shape
            ):
                raise ValueError("stored grid_shape differs from the binning")
            if "active_mask" in arrays and not np.array_equal(
                arrays["active_mask"], layout.active_mask
            ):
                raise ValueError("stored active_mask differs from the binning")
            for index, edges in enumerate(layout.grid_edges or ()):
                key = f"grid_edges_{index}"
                if key in arrays and not np.array_equal(arrays[key], edges):
                    raise ValueError(f"stored {key} differs from the binning")
        env_cls: type[_BaseEnvironment] = Environment
        if coordinate_kind == "polar":
            from neurospatial.environment.polar import EgocentricPolarEnvironment

            env_cls = EgocentricPolarEnvironment
        env = env_cls(
            metadata["name"],
            layout,
            metadata["layout_type"],
            params,
            regions=Regions([Region.from_dict(r) for r in metadata.get("regions", [])]),
            restored_from=source,
            arena=arena,
            units=metadata.get("units"),
            frame=metadata.get("frame"),
        )
        return cast("Environment", env)
    except (
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        OverflowError,
        RecursionError,
    ) as error:
        raise ValueError(
            _format_error(
                f"Invalid environment data in {source}: {error}",
                why="Why: saved geometry must be internally consistent and its Arena must use compatible coordinates.",
                fix="re-save the environment from its original factory-built object with this version",
            )
        ) from error


def _arrays_from_lists(metadata: dict[str, Any]) -> dict[str, NDArray[Any]]:
    """Restore JSON numerical arrays, preserving empty shapes and scalar dtypes."""
    arrays = {}
    for key, spec in metadata["array_specs"].items():
        raw = np.asarray(metadata["arrays"][key])
        dtype = np.dtype(spec["dtype"])
        if dtype.kind not in "biuf":
            raise ValueError(f"array {key!r} has unsupported dtype {dtype}")
        if raw.size and raw.dtype.kind not in "biuf":
            raise ValueError(f"array {key!r} must contain numerical values")
        if (
            raw.size
            and dtype.kind in "iu"
            and raw.dtype.kind == "f"
            and np.any(~np.isfinite(raw) | (raw != np.floor(raw)))
        ):
            raise ValueError(f"array {key!r} must contain integer indices")
        if raw.size and dtype.kind in "iu":
            limits = np.iinfo(dtype)
            if np.any(raw < limits.min) or np.any(raw > limits.max):
                raise ValueError(f"array {key!r} contains integers outside {dtype}")
        if raw.size and dtype.kind == "b" and raw.dtype.kind != "b":
            raise ValueError(f"array {key!r} must contain booleans")
        arrays[key] = np.asarray(raw, dtype=dtype).reshape(spec["shape"])
    return arrays


def to_dict(env: Environment) -> dict[str, Any]:
    """Convert v3 binning and Arena data to a dictionary for in-memory handoff.

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
    metadata, arrays = _environment_payload(env)
    metadata["arrays"] = {key: value.tolist() for key, value in arrays.items()}
    return metadata


def from_dict(data: dict[str, Any]) -> Environment:
    """Decode bins and Arena without rebuilding; v2 Arenas remain unknown.

    Version 1 dictionaries must be recreated with a factory and saved again.

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
    # Check schema version
    _check_schema(data, "the environment dictionary")
    try:
        arrays = _arrays_from_lists(data)
    except (ValueError, TypeError, KeyError, OverflowError) as error:
        raise ValueError(
            f"Invalid arrays in the environment dictionary: {error}"
        ) from error
    return _restore_environment(
        data,
        source="the environment dictionary",
        arrays=arrays,
        legacy_arena=data["schema_version"] == "Environment-v2",
    )
