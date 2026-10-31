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
from collections.abc import Iterator, Mapping
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from zipfile import BadZipFile

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error

if TYPE_CHECKING:
    from neurospatial.environment import Environment
    from neurospatial.geometry import Binning

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

    The ``Environment-v3`` format stores bins, lookup, cell faces, drawing
    components and Arena. Loading decodes these data without rebuilding a
    layout.

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
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> x, y = np.meshgrid([0.0, 2.0, 4.0], [0.0, 2.0, 4.0])
    >>> env = Environment.from_samples(np.c_[x.ravel(), y.ravel()], bin_size=2.0)
    >>> to_file(env, "my_environment")  # writes my_environment.json and .npz

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


def from_file(path: PathLike) -> Environment[Any]:
    """Load binning and Arena data without running a layout builder.

    Only ``Environment-v3`` files are read. Files saved by earlier versions are
    refused; recreate the environment with a factory and save it again.

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
        Cartesian or egocentric polar, as stored; static type checkers see
        ``Environment[Any]`` because the stored data decides. Check
        ``env.coordinate_kind`` before calling Cartesian-only methods.

    Raises
    ------
    FileNotFoundError
        If required files are not found.
    ValueError
        If schema version is incompatible, data is malformed, or path
        contains parent directory traversal ('..' components).

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> x, y = np.meshgrid([0.0, 2.0, 4.0], [0.0, 2.0, 4.0])
    >>> env = Environment.from_samples(np.c_[x.ravel(), y.ravel()], bin_size=2.0)
    >>> to_file(env, "my_environment")
    >>> loaded = from_file("my_environment")
    >>> np.array_equal(loaded.bin_centers, env.bin_centers)
    True

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
    return _restore_environment(metadata, source=str(json_path), arrays=arrays)


def _environment_payload(
    env: Environment,
) -> tuple[dict[str, Any], dict[str, NDArray[Any]]]:
    """Build the shared lossless metadata and numerical-array payload."""
    from neurospatial.environment import _build_record
    from neurospatial.geometry import encode, encode_arena, graph_to_node_link

    binning_meta, binning_arrays = encode(env.binning, "binning")
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
        "layout_parameters": env.layout_parameters,
        "derivations": list(env.derivations),
        "length_scale": _build_record(env).length_scale,
        "binning": binning_meta,
        "arena": arena_meta,
        "dimension_ranges": env.dimension_ranges,
        "grid_shape": env.grid_shape,
        "graph": graph_to_node_link(env.connectivity),
        "regions": [region.to_dict() for region in env.regions.values()],
        "landmarks": env.landmarks.to_list(),
        "coordinate_kind": env.coordinate_kind,
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
    if version != _SCHEMA_VERSION:
        raise ValueError(
            _format_error(
                f"Environment schema {version!r} in {source} is no longer readable; "
                f"this version reads only {_SCHEMA_VERSION!r}.",
                why="Why: older records lack the Arena and complete binning this version restores, and they are not converted.",
                fix="recreate the environment with an Environment factory and save it again with this version",
            )
        )


def _reject_cyclic_metadata(metadata: dict[str, Any]) -> None:
    """Reject container cycles before calling recursive component decoders.

    Numerical payloads have already been parsed independently; avoid walking
    their potentially large lists a second time. Shared acyclic containers are
    valid and are visited once. The traversal itself uses no Python recursion.
    """
    active = {id(metadata)}
    complete: set[int] = set()
    stack: list[tuple[int, Iterator[Any]]] = [
        (id(metadata), (value for key, value in metadata.items() if key != "arrays"))
    ]
    while stack:
        identity, children = stack[-1]
        try:
            value = next(children)
        except StopIteration:
            stack.pop()
            active.remove(identity)
            complete.add(identity)
            continue
        if not isinstance(value, (dict, list, tuple)):
            continue
        identity = id(value)
        if identity in active:
            raise ValueError("Circular environment metadata is not supported.")
        if identity in complete:
            continue
        active.add(identity)
        stack.append(
            (identity, iter(value.values() if isinstance(value, dict) else value))
        )


def _restore_environment(
    metadata: dict[str, Any],
    *,
    source: str,
    arrays: dict[str, NDArray[Any]],
) -> Environment:
    """Decode authoritative binning and Arena data without a layout builder."""
    from neurospatial.environment import (
        Environment,
        _declared_metadata,
        _Provenance,
        _publish,
    )
    from neurospatial.geometry import (
        Binning,
        HexLocator,
        MeshLocator,
        TrackLinearization,
        decode,
        decode_arena,
        grid_index,
    )
    from neurospatial.landmarks import Landmarks
    from neurospatial.regions import Region, Regions

    try:
        _reject_cyclic_metadata(metadata)
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
        binning = replace(
            binning, kind=_saved_label(binning, polar=coordinate_kind == "polar")
        )
        params, derivations = _saved_history(metadata, binning.kind)
        length_scale = _saved_length_scale(metadata, derivations)
        ranges = metadata.get("dimension_ranges")
        if ranges is not None:
            binning = replace(binning, dimension_ranges=tuple(tuple(r) for r in ranges))
        if metadata["is_linearized_track"] != (binning.linearization is not None):
            raise ValueError("stored is_linearized_track differs from the binning")
        # The stored graph is written for other tools; the environment
        # rebuilds its graph from the binning.
        context = grid_index(binning)
        if binning.grid_frame is not None and any(
            key.startswith("grid_edges_") for key in arrays
        ):
            raise ValueError(
                "Framed grid source axes cannot be stored as world grid_edges."
            )
        if (
            binning.grid_frame is not None
            or binning.rectilinear is not None
            or isinstance(binning.linearization, TrackLinearization)
            or isinstance(binning.locator, (HexLocator, MeshLocator))
        ):
            if (
                "grid_shape" in metadata
                and metadata["grid_shape"] is not None
                and tuple(metadata["grid_shape"]) != context.shape
            ):
                raise ValueError("stored grid_shape differs from the binning")
            if "active_mask" in arrays and not np.array_equal(
                arrays["active_mask"], context.active_mask
            ):
                raise ValueError("stored active_mask differs from the binning")
            for index, edges in enumerate(context.edges or ()):
                key = f"grid_edges_{index}"
                if key in arrays and not np.array_equal(arrays[key], edges):
                    raise ValueError(f"stored {key} differs from the binning")
        env = _publish(
            Environment,
            binning,
            coordinate_kind=coordinate_kind,
            provenance=_Provenance(params, derivations, length_scale),
            metadata=_declared_metadata(
                name=metadata["name"],
                units=metadata.get("units"),
                frame=metadata.get("frame"),
            ),
            regions=Regions([Region.from_dict(r) for r in metadata.get("regions", [])]),
            landmarks=Landmarks.from_list(metadata["landmarks"]),
            arena=arena,
        )
        return env
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


_TRANSFORMED = "_transformed"


def _saved_label(binning: Binning, *, polar: bool) -> str:
    """Return the canonical layout label of saved bins.

    The label is the binning's ``kind``, spelled as a ``LayoutType`` value.
    Files written before derivations kept their parent's label are read the
    same way: their polar bins (labelled ``"MaskedGrid"``) are ``"Polar"``,
    an ``"_transformed"`` suffix per ``apply_transform`` is dropped, and
    ``"subset"`` and ``"PolygonCells"`` (a subset of non-grid bins and a
    clip) take the label of the bins they were cut from. A subset or clip
    whose bins came from none of the built-in lookups (for example
    ``from_bins`` bins with nearest-center lookup) keeps its saved spelling,
    because the file does not record the original label.

    Parameters
    ----------
    binning : Binning
        The decoded bins.
    polar : bool
        Whether the bins have egocentric polar coordinates, which only
        ``Environment.from_polar_egocentric`` builds.

    Returns
    -------
    str
        The layout label.
    """
    from neurospatial.geometry import (
        ClippedLocator,
        CombinedLocator,
        HexLocator,
        MeshLocator,
        RectilinearLocator,
        SubsetLocator,
        TransformedLocator,
    )
    from neurospatial.layout import LayoutType, find_layout_type

    if polar:
        return LayoutType.POLAR.value
    kind = binning.kind
    while kind.endswith(_TRANSFORMED):
        kind = kind.removesuffix(_TRANSFORMED)
    if kind not in ("subset", "PolygonCells"):
        member = find_layout_type(kind)
        return kind if member is None else member.value
    locator = binning.locator
    while isinstance(locator, (SubsetLocator, TransformedLocator, ClippedLocator)):
        locator = locator.base
    if binning.linearization is not None:
        return LayoutType.GRAPH.value
    for lookup, label in (
        (HexLocator, LayoutType.HEXAGONAL),
        (MeshLocator, LayoutType.TRIANGULAR_MESH),
        (CombinedLocator, LayoutType.COMBINED),
        # A grid subset was always labelled "MaskedGrid".
        (RectilinearLocator, LayoutType.MASKED_GRID),
    ):
        if isinstance(locator, lookup):
            return label.value
    return kind


def _saved_length_scale(
    metadata: dict[str, Any], derivations: tuple[str, ...]
) -> float | None:
    """Return how long a unit of the saved build choices' lengths is now.

    Files written before this was saved give 1.0 when no ``apply_transform``
    followed the last recorded choices (a build, ``rebin`` or ``combine``),
    and None, unknown, when one did.

    Parameters
    ----------
    metadata : dict
        The saved environment metadata.
    derivations : tuple of str
        The saved derivation history, oldest first.

    Returns
    -------
    float or None
        The ``length_scale`` of the environment's build record.
    """
    if "length_scale" in metadata:
        saved: float | None = metadata["length_scale"]  # _Provenance checks it
        return saved
    recorded = max(
        (i for i, step in enumerate(derivations) if step in ("rebin", "combine")),
        default=-1,
    )
    return None if "apply_transform" in derivations[recorded + 1 :] else 1.0


def _saved_history(
    metadata: dict[str, Any], label: str
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Return the saved build choices and derivation history.

    Files without a ``"derivations"`` record kept the input data among the
    layout parameters and marked derivations there: ``"source"`` named a
    subset or clip and ``"transformed": True`` an ``apply_transform``, with
    one ``"_transformed"`` suffix on the layout label for each. Bins that
    read as ``"Combined"`` were made by ``combine``, which comes first in
    their history. Input data is dropped (the bins are stored directly), and
    the markers become the derivation history.

    Parameters
    ----------
    metadata : dict
        The saved environment metadata.
    label : str
        The canonical layout label read from the bins (see ``_saved_label``).

    Returns
    -------
    build_params : dict of str to object
        Scalar build choices.
    derivations : tuple of str
        Derivation names, oldest first.

    Raises
    ------
    ValueError
        If ``layout_parameters`` is not a dictionary or ``derivations`` is not
        a list of names.
    """
    from neurospatial.layout import LayoutType
    from neurospatial.layout.base import build_choices

    params = metadata["layout_parameters"]
    if not isinstance(params, dict):
        raise ValueError("layout_parameters must be a dictionary of build choices")
    choices: dict[str, Any] = {}
    for key, value in params.items():
        with contextlib.suppress(TypeError):
            choices.update(build_choices({key: value}, owner="layout_parameters"))
    if "derivations" in metadata:
        derivations = metadata["derivations"]
        if not isinstance(derivations, list) or not all(
            isinstance(step, str) for step in derivations
        ):
            raise ValueError("derivations must be a list of derivation names")
        return choices, tuple(derivations)
    kind = str(metadata.get("binning", {}).get("kind", ""))
    history = ["combine"] if label == LayoutType.COMBINED.value else []
    if choices.get("source") in ("subset", "clip"):
        history.append(choices.pop("source"))
        choices.pop("original_n_bins", None)
    if choices.pop("transformed", None) is True:
        n_transforms = 0
        while kind.endswith(_TRANSFORMED):
            kind = kind.removesuffix(_TRANSFORMED)
            n_transforms += 1
        history.extend(["apply_transform"] * max(n_transforms, 1))
    return choices, tuple(history)


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
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> x, y = np.meshgrid([0.0, 2.0, 4.0], [0.0, 2.0, 4.0])
    >>> env = Environment.from_samples(np.c_[x.ravel(), y.ravel()], bin_size=2.0)
    >>> import json
    >>> env_dict = to_dict(env)
    >>> # Pass to another process or serialize to JSON
    >>> json_str = json.dumps(env_dict)

    See Also
    --------
    from_dict : Reconstruct environment from dictionary
    to_file : Save to disk with efficient binary format

    """
    # Similar to to_file but with arrays as lists
    metadata, arrays = _environment_payload(env)
    metadata["arrays"] = {key: value.tolist() for key, value in arrays.items()}
    return metadata


def from_dict(data: dict[str, Any]) -> Environment[Any]:
    """Decode bins and Arena from a dictionary made by :func:`to_dict`.

    No layout builder runs. Only ``Environment-v3`` dictionaries are read;
    recreate older ones with an Environment factory.

    Parameters
    ----------
    data : dict[str, Any]
        Dictionary representation from `to_dict()`.

    Returns
    -------
    Environment
        Reconstructed Environment instance.
        Cartesian or egocentric polar, as stored; static type checkers see
        ``Environment[Any]`` because the stored data decides. Check
        ``env.coordinate_kind`` before calling Cartesian-only methods.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> x, y = np.meshgrid([0.0, 2.0, 4.0], [0.0, 2.0, 4.0])
    >>> env = Environment.from_samples(np.c_[x.ravel(), y.ravel()], bin_size=2.0)
    >>> env_restored = from_dict(to_dict(env))
    >>> np.array_equal(env_restored.bin_centers, env.bin_centers)
    True

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
        data, source="the environment dictionary", arrays=arrays
    )
