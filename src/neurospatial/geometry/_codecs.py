"""Pure, versioned codecs for binning geometry, lookup data and arenas.

Unknown codec names fall back to installed snapshot adapters, whose decoded
data is normalized back through these built-in codecs.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from importlib import metadata
from itertools import pairwise
from typing import Any, cast

import networkx as nx
import numpy as np
import shapely
from numpy.typing import NDArray
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

from neurospatial._exceptions import _format_error
from neurospatial.geometry._adapters import (
    _ENTRY_POINT_GROUP,
    SnapshotAdapter,
    _SnapshotRole,
)
from neurospatial.geometry._arena import (
    Arena,
    EgocentricSurround,
    MaskArena,
    OpenArena,
    TrackArena,
    UnknownArena,
    _boolean,
    _CombinedArena,
    _error,
    _TransformedArena,
)
from neurospatial.geometry._binning import (
    BinConnections,
    Binning,
    KnownFaces,
    MissingFaces,
    SpatialBins,
    TrackLinearization,
    graph_attribute_items,
)
from neurospatial.geometry._capabilities import RectilinearGrid
from neurospatial.geometry._components import HexagonShapes, Outline, TriangleShapes
from neurospatial.geometry._composition import CombinedLocator
from neurospatial.geometry._diffusion import FaceGeometry
from neurospatial.geometry._face_normals import FaceNormals
from neurospatial.geometry._grid_frame import FramedGrid
from neurospatial.geometry._locators import (
    HexLocator,
    MeshLocator,
    NearestBinLocator,
    RectilinearLocator,
    SubsetLocator,
    TransformedLocator,
)
from neurospatial.geometry._polygon_cells import (
    ClipMapping,
    ClippedLocator,
    PolygonCells,
)
from neurospatial.geometry._rebin_spec import GridBinningSpec

Payload = tuple[dict[str, Any], dict[str, NDArray[Any]]]

# Version 2 added the required wall names of an OpenArena.
_ARENA_VERSION = 2


@dataclass(frozen=True, eq=False)
class Codec:
    """The Python type and pure functions for a registered geometry kind."""

    kind: type[Any]
    encode: Callable[[Any, str], Payload]
    decode: Callable[[dict[str, Any], Mapping[str, NDArray[Any]], str], object]
    # The one version this codec writes and reads.
    version: int = 1


def encode(obj: object, prefix: str) -> Payload:
    """Encode geometry as JSON-safe metadata and independent array buffers.

    Parameters
    ----------
    obj : object
        A supported geometry record or lookup object.
    prefix : str
        Namespace for arrays, such as ``"binning/locator"``.

    Returns
    -------
    meta : dict
        JSON-safe metadata, including the codec name and integer version.
    arrays : dict of ndarray
        Independent arrays keyed by ``prefix/field``. Lookup memos are omitted.

    Raises
    ------
    ValueError
        If the object's type has no codec or metadata is not JSON-compatible.
    """
    if isinstance(obj, FaceGeometry):
        obj = KnownFaces(obj)
    name = type(obj).__name__
    codec = _CODECS.get(name)
    if codec is None or type(obj) is not codec.kind:
        raise ValueError(
            f"cannot save {name}: no codec is registered for it.\n"
            "Fix: use an environment with supported built-in binning data; "
            "layout-backed lookup and custom layout linearization cannot be saved."
        )
    meta, arrays = codec.encode(obj, prefix)
    return _json_safe({"codec": name, "codec_version": codec.version, **meta}), arrays


def decode(
    meta: dict[str, Any], arrays: Mapping[str, NDArray[Any]], prefix: str
) -> object:
    """Decode stored geometry without running a layout builder.

    Parameters
    ----------
    meta : dict
        Metadata returned by :func:`encode`.
    arrays : mapping of str to ndarray
        Array buffers returned by :func:`encode` or a storage reader.
    prefix : str
        The same array namespace used during encoding.

    Returns
    -------
    object
        Geometry with normalized dtypes and owned, read-only arrays. Locator
        search structures are not stored; lookup rebuilds them.

    Raises
    ------
    ValueError
        If the codec is unknown, its version is unsupported, or data is invalid.
    """
    if not isinstance(meta, dict):
        raise ValueError(f"invalid geometry metadata at {prefix}: expected a dict.")
    name = meta.get("codec")
    codec = _CODECS.get(name) if isinstance(name, str) else None
    if codec is None:
        if isinstance(name, str):
            return _decode_snapshot(meta, arrays, prefix)
        raise ValueError(
            f"cannot load {name!r}: no codec is registered for it.\n"
            "Fix: use a neurospatial version supporting this geometry codec."
        )
    version = meta.get("codec_version")
    supported = codec.version
    if type(version) is not int or version != supported:
        raise ValueError(
            f"cannot load {name} codec_version {version!r}; supported version is {supported}.\n"
            "Fix: recreate and save the environment with this neurospatial version."
        )
    try:
        return codec.decode(meta, arrays, prefix)
    except (KeyError, TypeError, ValueError, IndexError, OverflowError) as exc:
        raise ValueError(f"invalid {name} payload at {prefix}: {exc}") from exc


class _UnsupportedJSONValueError(ValueError):
    """A metadata value JSON cannot store.

    Parameters
    ----------
    message : str
        The full error message.
    reason : str
        What the offending value is, e.g. ``"set value"``.
    """

    def __init__(self, message: str, reason: str) -> None:
        super().__init__(message)
        self.reason = reason


def _json_safe(value: Any) -> Any:
    """Normalize metadata containers and NumPy scalars for JSON storage."""
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        raise _UnsupportedJSONValueError(
            "cannot save non-finite scalar metadata as JSON.\n"
            "Fix: use finite values or None for missing graph metadata.",
            reason="non-finite number",
        )
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise _UnsupportedJSONValueError(
        f"cannot save metadata value of type {type(value).__name__}.\n"
        "Fix: store only JSON-compatible scalar values and sequences in graph attributes.",
        reason=f"{type(value).__name__} value",
    )


def _arrays(prefix: str, **values: NDArray[Any]) -> dict[str, NDArray[Any]]:
    return {
        f"{prefix}/{name}": np.array(value, copy=True) for name, value in values.items()
    }


def _array(
    arrays: Mapping[str, NDArray[Any]], prefix: str, name: str, dtype: Any
) -> NDArray[Any]:
    """Read a typed array without silently truncating indices or masks."""
    key = f"{prefix}/{name}"
    if key not in arrays:
        raise ValueError(f"missing array {key!r}.")
    raw = np.asarray(arrays[key])
    kind = np.dtype(dtype).kind
    if kind in "iu":
        if raw.dtype.kind not in "iuf" or np.any(~np.isfinite(raw)):
            raise ValueError(f"{name} must contain finite integer values.")
        if raw.dtype.kind == "f" and np.any(raw != np.floor(raw)):
            raise ValueError(f"{name} must contain integer values.")
        limits = np.iinfo(dtype)
        above = raw >= limits.max + 1 if raw.dtype.kind == "f" else raw > limits.max
        if np.any(raw < limits.min) or np.any(above):
            raise ValueError(f"{name} contains an out-of-range integer.")
    elif kind == "b" and raw.dtype.kind != "b":
        raise ValueError(f"{name} must contain boolean values.")
    elif kind == "f" and raw.dtype.kind not in "biuf":
        raise ValueError(f"{name} must contain real numeric values.")
    return np.asarray(raw, dtype=dtype)


def _flag(meta: dict[str, Any], name: str) -> bool:
    value = meta[name]
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean.")
    return value


def _shape(meta: dict[str, Any], name: str = "shape") -> tuple[int, ...]:
    raw = meta[name]
    if (
        not isinstance(raw, (list, tuple))
        or not raw
        or any(type(value) is not int or value <= 0 for value in raw)
    ):
        raise ValueError(f"{name} must contain positive integer cell counts.")
    return tuple(raw)


def _finite(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite numeric value, not a boolean.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite numeric value.") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _grid_edges(
    meta: dict[str, Any], arrays: Mapping[str, NDArray[Any]], prefix: str
) -> tuple[tuple[NDArray[np.float64], ...], tuple[int, ...]]:
    shape = _shape(meta)
    edges = tuple(
        _array(arrays, prefix, f"edges_{i}", np.float64) for i in range(len(shape))
    )
    return edges, shape


def _active_map(mapping: NDArray[Any], size: int, name: str) -> None:
    if mapping.shape != (size,) or np.any(mapping < -1):
        raise ValueError(f"{name} must have shape ({size},) and indices >= -1.")
    active = mapping[mapping >= 0]
    if not np.array_equal(np.sort(active), np.arange(len(active))):
        raise ValueError(
            f"{name} active indices must number bins exactly once from zero."
        )


def _encode_bins(obj: SpatialBins, prefix: str) -> Payload:
    return {}, _arrays(
        prefix, centers=obj.centers, sizes=obj.sizes, measure_dims=obj.measure_dims
    )


def _decode_bins(
    meta: dict[str, Any], arrays: Mapping[str, NDArray[Any]], prefix: str
) -> SpatialBins:
    return SpatialBins.from_arrays(
        _array(arrays, prefix, "centers", np.float64),
        _array(arrays, prefix, "sizes", np.float64),
        _array(arrays, prefix, "measure_dims", np.intp),
    )


def _encode_connections(obj: BinConnections, prefix: str) -> Payload:
    return {}, _arrays(
        prefix,
        all_pairs=obj.all_pairs,
        all_distances=obj.all_distances,
        is_open=obj.is_open,
        closed_by=obj.closed_by,
    )


def _decode_connections(
    meta: dict[str, Any], arrays: Mapping[str, NDArray[Any]], prefix: str
) -> BinConnections:
    return BinConnections.from_arrays(
        _array(arrays, prefix, "all_pairs", np.intp),
        _array(arrays, prefix, "all_distances", np.float64),
        _array(arrays, prefix, "is_open", np.bool_),
        _array(arrays, prefix, "closed_by", np.intp),
    )


def _encode_faces(obj: KnownFaces, prefix: str) -> Payload:
    faces = obj.faces
    return {}, _arrays(
        prefix,
        edges=faces.edges,
        A=faces.A,
        distance=faces.distance,
        volumes=faces.volumes,
    )


def _decode_faces(
    meta: dict[str, Any], arrays: Mapping[str, NDArray[Any]], prefix: str
) -> KnownFaces:
    return KnownFaces(
        FaceGeometry(
            _array(arrays, prefix, "edges", np.intp),
            _array(arrays, prefix, "A", np.float64),
            _array(arrays, prefix, "distance", np.float64),
            _array(arrays, prefix, "volumes", np.float64),
        )
    )


def _encode_missing(obj: MissingFaces, prefix: str) -> Payload:
    return {"reason": obj.reason}, {}


def _decode_missing(
    meta: dict[str, Any], arrays: Mapping[str, NDArray[Any]], prefix: str
) -> MissingFaces:
    if not isinstance(meta["reason"], str):
        raise ValueError("reason must be a string explaining unavailable faces.")
    return MissingFaces(meta["reason"])


def _encode_grid(obj: RectilinearGrid, prefix: str) -> Payload:
    values: dict[str, NDArray[Any]] = {
        f"edges_{axis}": edge for axis, edge in enumerate(obj.edges)
    }
    values["active_mask"] = obj.active_mask
    return {"shape": list(obj.shape)}, _arrays(prefix, **values)


def _decode_grid(
    meta: dict[str, Any], arrays: Mapping[str, NDArray[Any]], prefix: str
) -> RectilinearGrid:
    edges, shape = _grid_edges(meta, arrays, prefix)
    return RectilinearGrid(
        edges, shape, _array(arrays, prefix, "active_mask", np.bool_)
    )


def _refuse_json_attributes(graph: nx.Graph) -> None:
    """Name the first track graph attribute that JSON cannot store."""
    for element, key, value in graph_attribute_items(graph):
        try:
            _json_safe(value)
        except _UnsupportedJSONValueError as exc:
            raise ValueError(
                f"cannot save track graph {element} attribute {key!r}: its "
                f"value ({type(value).__name__}) is or contains a {exc.reason}, "
                "which JSON cannot store.\n"
                "Why: to_file, to_dict and NWB store the track graph as JSON, "
                "which holds finite numbers, strings, booleans, None and lists "
                "of them; pickle can still send this environment to worker "
                "processes.\n"
                f"Fix: store {key!r} as one of those values, or remove it, "
                "before building the environment."
            ) from exc


def _encode_track(obj: TrackLinearization, prefix: str) -> Payload:
    _refuse_json_attributes(obj.graph)
    values: dict[str, NDArray[Any]] = {
        f"edges_{axis}": edge for axis, edge in enumerate(obj.grid_edges)
    }
    values["active_mask"] = obj.active_mask
    return {
        "graph": graph_to_node_link(obj.graph),
        "edge_order": obj.edge_order,
        "edge_spacing": obj.edge_spacing,
        "shape": [len(edge) - 1 for edge in obj.grid_edges],
    }, _arrays(prefix, **values)


def _node_id(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    return (
        tuple(_node_id(item) for item in value)
        if isinstance(value, (list, tuple))
        else value
    )


_RESERVED_GRAPH_ATTRIBUTES = "_neurospatial_reserved_attributes"


def graph_to_node_link(graph: nx.Graph) -> dict[str, Any]:
    """Preserve attributes overwritten by NetworkX's node-link identity fields.

    Optional sidecar rows retain node ``id`` and edge ``source``/``target``
    attributes, plus ``key`` for multigraphs. Ordinary records remain unchanged.
    """
    data = nx.node_link_data(graph, edges="links")
    sidecar: dict[str, list[dict[str, Any]]] = {}

    def remember(
        group: str, identity: tuple[str, ...], attributes: list[dict[str, Any]]
    ) -> None:
        rows = []
        for record, original in zip(data[group], attributes, strict=True):
            saved = {key: original[key] for key in identity if key in original}
            rows.append({**{key: record[key] for key in identity}, "attributes": saved})
        if any(row["attributes"] for row in rows):
            sidecar[group] = rows

    remember("nodes", ("id",), [attributes for _, attributes in graph.nodes(data=True)])
    if graph.is_multigraph():
        remember(
            "links",
            ("source", "target", "key"),
            [attributes for _, _, _, attributes in graph.edges(keys=True, data=True)],
        )
    else:
        remember(
            "links",
            ("source", "target"),
            [attributes for _, _, attributes in graph.edges(data=True)],
        )
    if sidecar:
        data[_RESERVED_GRAPH_ATTRIBUTES] = sidecar
    return cast("dict[str, Any]", data)


def graph_from_node_link(data: dict[str, Any]) -> nx.Graph:
    """Restore tuple-valued graph labels without changing supplied metadata.

    Numeric metadata readers may represent sequence labels as ndarrays.
    Nested node and edge endpoint labels are restored consistently before
    NetworkX reconstructs the node-link graph.
    """
    graph_data = copy.deepcopy(data)
    for node in graph_data["nodes"]:
        node["id"] = _node_id(node["id"])
    for link in graph_data["links"]:
        link["source"] = _node_id(link["source"])
        link["target"] = _node_id(link["target"])
        if graph_data.get("multigraph", False) and "key" in link:
            link["key"] = _node_id(link["key"])
    sidecar = graph_data.pop(_RESERVED_GRAPH_ATTRIBUTES, None)
    if sidecar is not None:
        if not isinstance(sidecar, dict) or not set(sidecar) <= {"nodes", "links"}:
            raise ValueError(
                "Reserved graph attributes must contain node/link sidecar rows."
            )
        for group, rows in sidecar.items():
            identity = (
                ("id",)
                if group == "nodes"
                else ("source", "target", "key")
                if graph_data.get("multigraph", False)
                else ("source", "target")
            )
            if not isinstance(rows, list) or len(rows) != len(graph_data[group]):
                raise ValueError(
                    "Reserved graph attribute sidecar length differs from its graph."
                )
            for row, record in zip(rows, graph_data[group], strict=True):
                if not isinstance(row, dict) or set(row) != {*identity, "attributes"}:
                    raise ValueError(
                        "Reserved graph attribute rows need matching identities and attributes."
                    )
                if any(_node_id(row[key]) != record[key] for key in identity):
                    raise ValueError(
                        "Reserved graph attribute sidecar identities differ from its graph."
                    )
                attributes = row["attributes"]
                if not isinstance(attributes, dict) or not set(attributes) <= set(
                    identity
                ):
                    raise ValueError(
                        "Reserved graph attribute sidecars may restore only reserved keys."
                    )
    graph = nx.node_link_graph(graph_data, edges="links")
    if sidecar is not None:
        for group, rows in sidecar.items():
            for row, record in zip(rows, graph_data[group], strict=True):
                if group == "nodes":
                    attributes = graph.nodes[record["id"]]
                elif graph.is_multigraph():
                    attributes = graph.edges[
                        record["source"], record["target"], record["key"]
                    ]
                else:
                    attributes = graph.edges[record["source"], record["target"]]
                attributes.update(row["attributes"])
    return graph


def _decode_track(
    meta: dict[str, Any], arrays: Mapping[str, NDArray[Any]], prefix: str
) -> TrackLinearization:
    edges, _ = _grid_edges(meta, arrays, prefix)
    edge_order = tuple(
        tuple(_node_id(node) for node in edge) for edge in meta["edge_order"]
    )
    return TrackLinearization(
        graph_from_node_link(meta["graph"]),
        edge_order,
        meta["edge_spacing"],
        edges,
        _array(arrays, prefix, "active_mask", np.bool_),
    )


def _encode_hex_shapes(obj: HexagonShapes, prefix: str) -> Payload:
    return {"radius": obj.radius, "orientation": obj.orientation}, _arrays(
        prefix, centers=obj.centers
    )


def _decode_hex_shapes(
    meta: dict[str, Any], arrays: Mapping[str, NDArray[Any]], prefix: str
) -> HexagonShapes:
    return HexagonShapes(
        _array(arrays, prefix, "centers", np.float64),
        _finite(meta["radius"], "radius"),
        _finite(meta["orientation"], "orientation"),
    )


def _encode_triangle_shapes(obj: TriangleShapes, prefix: str) -> Payload:
    return {}, _arrays(prefix, points=obj.points, triangles=obj.triangles)


def _mesh_arrays(
    arrays: Mapping[str, NDArray[Any]], prefix: str, triangle_name: str
) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    points = _array(arrays, prefix, "points", np.float64)
    triangles = _array(arrays, prefix, triangle_name, np.intp)
    if points.ndim != 2 or points.shape[1] != 2 or np.any(~np.isfinite(points)):
        raise ValueError("points must be finite with shape (n_points, 2).")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError(f"{triangle_name} must have shape (n_triangles, 3).")
    if np.any((triangles < 0) | (triangles >= len(points))):
        raise ValueError(f"{triangle_name} contains an out-of-range vertex index.")
    return points, triangles


def _decode_triangle_shapes(
    meta: dict[str, Any], arrays: Mapping[str, NDArray[Any]], prefix: str
) -> TriangleShapes:
    return TriangleShapes(
        _array(arrays, prefix, "points", np.float64),
        _array(arrays, prefix, "triangles", np.intp),
    )


def _encode_outline(obj: Outline, prefix: str) -> Payload:
    return {}, _arrays(
        prefix, wkb=np.frombuffer(shapely.to_wkb(obj.geometry), dtype=np.uint8)
    )


def _decode_outline(
    meta: dict[str, Any], arrays: Mapping[str, NDArray[Any]], prefix: str
) -> Outline:
    encoded = _array(arrays, prefix, "wkb", np.uint8)
    if encoded.ndim != 1:
        raise ValueError("wkb must be a one-dimensional byte array.")
    try:
        geometry = shapely.from_wkb(encoded.tobytes())
    except shapely.errors.GEOSException as exc:
        raise ValueError(f"invalid outline WKB: {exc}") from exc
    if not isinstance(geometry, (Polygon, MultiPolygon)) or not geometry.is_valid:
        raise ValueError("outline WKB must contain a valid Polygon or MultiPolygon.")
    return Outline(geometry)


def _encode_rectilinear_locator(obj: RectilinearLocator, prefix: str) -> Payload:
    values: dict[str, NDArray[Any]] = {
        f"edges_{axis}": edge for axis, edge in enumerate(obj.edges)
    }
    values["active_mask"] = obj.active_mask
    return {"shape": list(obj.shape)}, _arrays(prefix, **values)


def _decode_rectilinear_locator(
    meta: dict[str, Any], arrays: Mapping[str, NDArray[Any]], prefix: str
) -> RectilinearLocator:
    edges, shape = _grid_edges(meta, arrays, prefix)
    return RectilinearLocator(
        edges, shape, _array(arrays, prefix, "active_mask", np.bool_)
    )


def _encode_hex_locator(obj: HexLocator, prefix: str) -> Payload:
    return {
        "offset_x": obj.offset_x,
        "offset_y": obj.offset_y,
        "radius": obj.radius,
        "centers_shape": obj.centers_shape,
    }, _arrays(prefix, flat_to_active=obj.flat_to_active)


def _decode_hex_locator(
    meta: dict[str, Any], arrays: Mapping[str, NDArray[Any]], prefix: str
) -> HexLocator:
    shape = _shape(meta, "centers_shape")
    return HexLocator(
        _finite(meta["offset_x"], "offset_x"),
        _finite(meta["offset_y"], "offset_y"),
        _finite(meta["radius"], "radius"),
        cast("tuple[int, int]", shape),
        _array(arrays, prefix, "flat_to_active", np.intp),
    )


def _encode_mesh_locator(obj: MeshLocator, prefix: str) -> Payload:
    arrays = _arrays(
        prefix,
        points=obj.points,
        simplices=obj.simplices,
        simplex_to_active=obj.simplex_to_active,
    )
    boundary = None
    if obj.boundary is not None:
        boundary, nested = encode(Outline(obj.boundary), f"{prefix}/boundary")
        arrays.update(nested)
    return {"boundary": boundary}, arrays


def _decode_mesh_locator(
    meta: dict[str, Any], arrays: Mapping[str, NDArray[Any]], prefix: str
) -> MeshLocator:
    points, simplices = _mesh_arrays(arrays, prefix, "simplices")
    mapping = _array(arrays, prefix, "simplex_to_active", np.intp)
    _active_map(mapping, len(simplices), "simplex_to_active")
    boundary = None
    if meta["boundary"] is not None:
        component = decode(meta["boundary"], arrays, f"{prefix}/boundary")
        if not isinstance(component, Outline) or not isinstance(
            component.geometry, Polygon
        ):
            raise ValueError("boundary must contain a polygon outline.")
        boundary = component.geometry
    return MeshLocator(points, simplices, mapping, boundary)


def _encode_nearest_locator(obj: NearestBinLocator, prefix: str) -> Payload:
    assert obj.bin_indices is not None
    # JSON metadata holds only finite numbers; an unlimited radius is a flag.
    unbounded = obj.max_distance is not None and np.isinf(obj.max_distance)
    meta = {
        "max_distance": None if unbounded else obj.max_distance,
        "unbounded": unbounded,
    }
    return meta, _arrays(prefix, centers=obj.centers, bin_indices=obj.bin_indices)


def _decode_nearest_locator(
    meta: dict[str, Any], arrays: Mapping[str, NDArray[Any]], prefix: str
) -> NearestBinLocator:
    centers = _array(arrays, prefix, "centers", np.float64)
    indices = _array(arrays, prefix, "bin_indices", np.intp)
    if centers.ndim != 2 or np.any(~np.isfinite(centers)):
        raise ValueError("centers must be a finite two-dimensional array.")
    if indices.shape != (len(centers),) or np.any(indices < 0):
        raise ValueError("bin_indices must give one non-negative index per center.")
    maximum = meta["max_distance"]
    if _flag(meta, "unbounded"):
        maximum = math.inf
    elif maximum is not None:
        maximum = _finite(maximum, "max_distance")
    return NearestBinLocator(centers, maximum, indices)


def _encode_binning(obj: Binning, prefix: str) -> Payload:
    meta: dict[str, Any] = {"kind": obj.kind}
    arrays: dict[str, NDArray[Any]] = {}
    cells: list[PolygonCells] = []
    mappings: list[ClipMapping] = []

    def reference(value: PolygonCells | ClipMapping) -> dict[str, int]:
        snapshots: list[Any] = cells if isinstance(value, PolygonCells) else mappings
        # Owned binnings may keep separate copies of the same immutable mapping.
        index = next(
            (
                i
                for i, snapshot in enumerate(snapshots)
                if snapshot is value
                or (
                    isinstance(value, ClipMapping)
                    and snapshot.n_parent_bins == value.n_parent_bins
                    and np.array_equal(
                        snapshot.parent_bin_indices, value.parent_bin_indices
                    )
                )
            ),
            len(snapshots),
        )
        if index == len(snapshots):
            snapshots.append(value)
        key = (
            "polygon_cells_ref"
            if isinstance(value, PolygonCells)
            else "clip_mapping_ref"
        )
        return {key: index}

    # Current components precede historical locator frames in snapshot order.
    if isinstance(obj.shapes, PolygonCells):
        reference(obj.shapes)

    def locator_payload(locator: object, path: str) -> Payload:
        if not isinstance(locator, (ClippedLocator, SubsetLocator, TransformedLocator)):
            return encode(locator, path)
        base, nested = locator_payload(locator.base, f"{path}/base")
        local: dict[str, Any] = {
            "codec": type(locator).__name__,
            "codec_version": 1,
            "base": base,
        }
        if isinstance(locator, ClippedLocator):
            local["cells_ref"] = reference(locator.cells)["polygon_cells_ref"]
            local["mapping_ref"] = reference(locator.mapping)["clip_mapping_ref"]
        elif isinstance(locator, SubsetLocator):
            nested.update(_arrays(path, parent_to_child=locator.parent_to_child))
        else:
            nested.update(_arrays(path, forward=locator.forward))
        return local, nested

    # Lookup is encoded first so unsupported layout-backed persistence fails
    # with its actionable error before any optional layout work is requested.
    for name in (
        "locator",
        "bins",
        "connections",
        "faces",
        "rectilinear",
        "linearization",
        "shapes",
        "outline",
        "grid_spec",
        "face_normals",
        "grid_frame",
    ):
        part = getattr(obj, name)
        if name == "faces" and not isinstance(part, (KnownFaces, MissingFaces)):
            part = (
                KnownFaces(part.build())
                if part.available
                else MissingFaces(str(part.missing_error("this operation")))
            )
        if part is None:
            meta[name] = None
        elif isinstance(part, (PolygonCells, ClipMapping)):
            meta[name] = reference(part)
        elif name == "locator":
            meta[name], nested = locator_payload(part, f"{prefix}/{name}")
            arrays.update(nested)
        else:
            meta[name], nested = encode(part, f"{prefix}/{name}")
            arrays.update(nested)
    meta["dimension_ranges"] = (
        None
        if obj.dimension_ranges is None
        else [[float(low), float(high)] for low, high in obj.dimension_ranges]
    )
    if cells or mappings:
        meta["snapshots"] = {"polygon_cells": [], "clip_mappings": []}
        for name, snapshots in (("polygon_cells", cells), ("clip_mappings", mappings)):
            for index, snapshot in enumerate(snapshots):
                encoded, nested = encode(snapshot, f"{prefix}/snapshots/{name}/{index}")
                meta["snapshots"][name].append(encoded)
                arrays.update(nested)
    return meta, arrays


def _nonnegative_integer(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")
    return int(value)


def _encode_polygon_cells(obj: PolygonCells, prefix: str) -> Payload:
    pieces = [shapely.to_wkb(polygon) for polygon in obj.polygons]
    offsets = np.array(
        [0, *np.cumsum([len(piece) for piece in pieces], dtype=np.intp)], dtype=np.intp
    )
    data = np.frombuffer(b"".join(pieces), dtype=np.uint8).copy()
    return {"n_cells": len(pieces)}, _arrays(prefix, wkb=data, offsets=offsets)


def _decode_polygon_cells(
    meta: dict[str, Any], arrays: Mapping[str, NDArray[Any]], prefix: str
) -> PolygonCells:
    count = _nonnegative_integer(meta["n_cells"], "n_cells")
    data = np.asarray(arrays[f"{prefix}/wkb"])
    if data.ndim != 1 or data.dtype != np.uint8:
        raise ValueError("wkb must be a one-dimensional uint8 array.")
    offsets = _array(arrays, prefix, "offsets", np.intp)
    if (
        offsets.shape != (count + 1,)
        or offsets[0] != 0
        or offsets[-1] != len(data)
        or np.any(np.diff(offsets) <= 0)
    ):
        raise ValueError("offsets must partition wkb into n_cells nonempty records.")
    polygons = []
    for index, (start, end) in enumerate(pairwise(offsets)):
        try:
            polygon = shapely.from_wkb(data[start:end].tobytes())
        except shapely.errors.GEOSException as exc:
            raise ValueError(f"wkb cell {index} is malformed: {exc}") from exc
        if not isinstance(polygon, Polygon):
            raise ValueError(f"wkb cell {index} must contain one connected Polygon.")
        polygons.append(polygon)
    return PolygonCells(tuple(polygons))


def _encode_clip_mapping(obj: ClipMapping, prefix: str) -> Payload:
    return {"n_parent_bins": obj.n_parent_bins}, _arrays(
        prefix, parent_bin_indices=obj.parent_bin_indices
    )


def _decode_clip_mapping(
    meta: dict[str, Any], arrays: Mapping[str, NDArray[Any]], prefix: str
) -> ClipMapping:
    return ClipMapping(
        _array(arrays, prefix, "parent_bin_indices", np.intp),
        _nonnegative_integer(meta["n_parent_bins"], "n_parent_bins"),
    )


def _encode_clipped_locator(obj: ClippedLocator, prefix: str) -> Payload:
    meta: dict[str, Any] = {}
    arrays: dict[str, NDArray[Any]] = {}
    for name in ("base", "cells", "mapping"):
        meta[name], nested = encode(getattr(obj, name), f"{prefix}/{name}")
        arrays.update(nested)
    return meta, arrays


def _decode_clipped_locator(
    meta: dict[str, Any], arrays: Mapping[str, NDArray[Any]], prefix: str
) -> ClippedLocator:
    base = decode(meta["base"], arrays, f"{prefix}/base")
    cells = decode(meta["cells"], arrays, f"{prefix}/cells")
    mapping = decode(meta["mapping"], arrays, f"{prefix}/mapping")
    if (
        not isinstance(base, _LOCATOR_TYPES)
        or not isinstance(cells, PolygonCells)
        or not isinstance(mapping, ClipMapping)
    ):
        raise ValueError(
            "clipped locator needs a base locator, polygon cells and clip mapping."
        )
    return ClippedLocator(base, cells, mapping)


def _encode_subset_locator(obj: SubsetLocator, prefix: str) -> Payload:
    base, arrays = encode(obj.base, f"{prefix}/base")
    arrays.update(_arrays(prefix, parent_to_child=obj.parent_to_child))
    return {"base": base}, arrays


def _decode_subset_locator(
    meta: dict[str, Any], arrays: Mapping[str, NDArray[Any]], prefix: str
) -> SubsetLocator:
    base = decode(meta["base"], arrays, f"{prefix}/base")
    if not isinstance(base, _LOCATOR_TYPES):
        raise ValueError("base must contain a point-lookup codec.")
    return SubsetLocator(base, _array(arrays, prefix, "parent_to_child", np.intp))


def _encode_transformed_locator(obj: TransformedLocator, prefix: str) -> Payload:
    base, arrays = encode(obj.base, f"{prefix}/base")
    arrays.update(_arrays(prefix, forward=obj.forward))
    return {"base": base}, arrays


def _decode_transformed_locator(
    meta: dict[str, Any], arrays: Mapping[str, NDArray[Any]], prefix: str
) -> TransformedLocator:
    base = decode(meta["base"], arrays, f"{prefix}/base")
    if not isinstance(base, _LOCATOR_TYPES):
        raise ValueError("base must contain a point-lookup codec.")
    return TransformedLocator(base, _array(arrays, prefix, "forward", np.float64))


def _decode_binning(
    meta: dict[str, Any], arrays: Mapping[str, NDArray[Any]], prefix: str
) -> Binning:
    parts: dict[str, Any] = {}
    cells: list[PolygonCells] = []
    mappings: list[ClipMapping] = []
    tables = meta.get("snapshots")
    if tables is not None:
        if not isinstance(tables, dict):
            raise ValueError("snapshots must contain typed snapshot tables.")
        for name, expected_kind in (
            ("polygon_cells", PolygonCells),
            ("clip_mappings", ClipMapping),
        ):
            entries = tables[name]
            if not isinstance(entries, list):
                raise ValueError(f"{name} snapshots must be a list.")
            for index, entry in enumerate(entries):
                snapshot = decode(entry, arrays, f"{prefix}/snapshots/{name}/{index}")
                if not isinstance(snapshot, expected_kind):
                    raise ValueError(f"{name} snapshot has the wrong geometry kind.")
                if isinstance(snapshot, PolygonCells):
                    cells.append(snapshot)
                else:
                    assert isinstance(snapshot, ClipMapping)
                    mappings.append(snapshot)

    def referenced(value: Any, key: str) -> PolygonCells | ClipMapping:
        if not isinstance(value, dict) or set(value) != {key}:
            raise ValueError(f"{key} must be a typed snapshot reference.")
        index = _nonnegative_integer(value[key], f"{key} snapshot reference")
        snapshots: Sequence[PolygonCells | ClipMapping] = (
            cells if key == "polygon_cells_ref" else mappings
        )
        if index >= len(snapshots):
            raise ValueError(f"{key} snapshot reference is out of range.")
        return snapshots[index]

    def locator_value(value: Any, path: str) -> Any:
        if not isinstance(value, dict):
            raise ValueError("locator must contain geometry metadata.")
        kind = value.get("codec")
        if kind not in {"ClippedLocator", "SubsetLocator", "TransformedLocator"}:
            part = decode(value, arrays, path)
            if not isinstance(part, _LOCATOR_TYPES):
                raise ValueError("locator must contain a point-lookup codec.")
            return part
        if type(value.get("codec_version")) is not int or value["codec_version"] != 1:
            raise ValueError("locator codec_version must be 1.")
        base = locator_value(value["base"], f"{path}/base")
        if kind == "ClippedLocator":
            if "cells_ref" not in value or "mapping_ref" not in value:
                raise ValueError(
                    "clipped locators in a binning need typed snapshot references."
                )
            cell_snapshot = referenced(
                {"polygon_cells_ref": value["cells_ref"]}, "polygon_cells_ref"
            )
            map_snapshot = referenced(
                {"clip_mapping_ref": value["mapping_ref"]}, "clip_mapping_ref"
            )
            assert isinstance(cell_snapshot, PolygonCells) and isinstance(
                map_snapshot, ClipMapping
            )
            return ClippedLocator(base, cell_snapshot, map_snapshot)
        if kind == "SubsetLocator":
            return SubsetLocator(base, _array(arrays, path, "parent_to_child", np.intp))
        return TransformedLocator(base, _array(arrays, path, "forward", np.float64))

    expected: dict[str, type | tuple[type, ...]] = {
        "bins": SpatialBins,
        "connections": BinConnections,
        "locator": _LOCATOR_TYPES,
        "faces": (KnownFaces, MissingFaces),
        "rectilinear": RectilinearGrid,
        "linearization": TrackLinearization,
        "shapes": (HexagonShapes, TriangleShapes, PolygonCells),
        "outline": Outline,
        "grid_spec": GridBinningSpec,
        "face_normals": FaceNormals,
        "grid_frame": FramedGrid,
    }
    part: object | None
    for name, kind in expected.items():
        value = meta[name]
        if value is None:
            part = None
        elif (
            name == "shapes"
            and isinstance(value, dict)
            and "polygon_cells_ref" in value
        ):
            part = referenced(value, "polygon_cells_ref")
        elif name == "locator":
            part = locator_value(value, f"{prefix}/{name}")
        else:
            part = decode(value, arrays, f"{prefix}/{name}")
            if isinstance(part, (PolygonCells, ClipMapping)):
                raise ValueError(
                    "polygon components in a binning need typed snapshot references."
                )
        if part is None and name in {"bins", "connections", "locator", "faces"}:
            raise ValueError(f"{name} is required for a binning.")
        if part is not None and not isinstance(part, kind):
            raise ValueError(f"{name} has the wrong geometry codec kind.")
        parts[name] = part
    if not isinstance(meta["kind"], str):
        raise ValueError("kind must be a descriptive string.")
    parts["connections"].check_bins(parts["bins"].n_bins)
    ranges = meta["dimension_ranges"]
    if ranges is not None and (
        not isinstance(ranges, list)
        or any(not isinstance(axis, list) or len(axis) != 2 for axis in ranges)
    ):
        raise ValueError("dimension_ranges must be one [low, high] pair per axis.")
    return Binning(
        kind=meta["kind"],
        **parts,
        dimension_ranges=(
            None
            if ranges is None
            else tuple(
                (_finite(low, "dimension_ranges"), _finite(high, "dimension_ranges"))
                for low, high in ranges
            )
        ),
    )


def _encode_combined(obj: CombinedLocator, prefix: str) -> Payload:
    entries = []
    arrays = _arrays(
        prefix,
        bridge_pairs=obj.bridge_pairs,
        bridge_distances=obj.bridge_distances,
        bridge_conductances=obj.bridge_conductances,
    )
    for index, (child, arena) in enumerate(zip(obj.children, obj.arenas, strict=True)):
        path = f"{prefix}/children/{index}"
        child_meta, child_arrays = encode(child, f"{path}/binning")
        arena_meta, arena_arrays = encode_arena(arena)
        entries.append({"binning": child_meta, "arena": arena_meta})
        arrays.update(child_arrays)
        arrays.update(
            {f"{path}/arena/{key}": value for key, value in arena_arrays.items()}
        )
    return {"keys": list(obj.keys), "children": entries}, arrays


def _decode_combined(
    meta: dict[str, Any], arrays: Mapping[str, NDArray[Any]], prefix: str
) -> CombinedLocator:
    if prefix.count("/children/") > 32:
        raise ValueError(
            "Composition nesting exceeds 32 levels; flatten the components."
        )
    children, arenas = [], []
    for index, entry in enumerate(meta["children"]):
        path = f"{prefix}/children/{index}"
        child = decode(entry["binning"], arrays, f"{path}/binning")
        if not isinstance(child, Binning):
            raise ValueError("Combined child must be a binning snapshot.")
        children.append(child)
        arenas.append(
            decode_arena(
                entry["arena"],
                {
                    key.removeprefix(f"{path}/arena/"): value
                    for key, value in arrays.items()
                    if key.startswith(f"{path}/arena/")
                },
            )
        )
    return CombinedLocator(
        tuple(meta["keys"]),
        tuple(children),
        tuple(arenas),
        _array(arrays, prefix, "bridge_pairs", np.intp),
        _array(arrays, prefix, "bridge_distances", np.float64),
        _array(arrays, prefix, "bridge_conductances", np.float64),
    )


_LOCATOR_TYPES = (
    CombinedLocator,
    RectilinearLocator,
    HexLocator,
    MeshLocator,
    NearestBinLocator,
    SubsetLocator,
    TransformedLocator,
    ClippedLocator,
)


def _encode_grid_spec(obj: GridBinningSpec, prefix: str) -> Payload:
    return {"connect_diagonal": obj.connect_diagonal}, _arrays(
        prefix, origin=obj.origin, forward=obj.forward
    )


def _decode_grid_spec(
    meta: dict[str, Any], arrays: Mapping[str, NDArray[Any]], prefix: str
) -> GridBinningSpec:
    return GridBinningSpec(
        _array(arrays, prefix, "origin", np.float64),
        _array(arrays, prefix, "forward", np.float64),
        _flag(meta, "connect_diagonal"),
    )


def _encode_face_normals(obj: FaceNormals, prefix: str) -> Payload:
    return {}, _arrays(prefix, values=obj.values)


def _decode_face_normals(
    meta: dict[str, Any], arrays: Mapping[str, NDArray[Any]], prefix: str
) -> FaceNormals:
    return FaceNormals(np.asarray(arrays[f"{prefix}/values"]))


def _encode_grid_frame(obj: FramedGrid, prefix: str) -> Payload:
    source, arrays = encode(obj.source, f"{prefix}/source")
    arrays.update(_arrays(prefix, forward=obj.forward))
    return {"source": source}, arrays


def _decode_grid_frame(
    meta: dict[str, Any], arrays: Mapping[str, NDArray[Any]], prefix: str
) -> FramedGrid:
    source = decode(meta["source"], arrays, f"{prefix}/source")
    if type(source) is not RectilinearGrid:
        raise ValueError("FramedGrid.source needs a RectilinearGrid codec.")
    return FramedGrid(source, _array(arrays, prefix, "forward", np.float64))


_CODECS: dict[str, Codec] = {
    "GridBinningSpec": Codec(GridBinningSpec, _encode_grid_spec, _decode_grid_spec),
    "CombinedLocator": Codec(CombinedLocator, _encode_combined, _decode_combined),
    "SpatialBins": Codec(SpatialBins, _encode_bins, _decode_bins, version=2),
    "BinConnections": Codec(BinConnections, _encode_connections, _decode_connections),
    "KnownFaces": Codec(KnownFaces, _encode_faces, _decode_faces),
    "MissingFaces": Codec(MissingFaces, _encode_missing, _decode_missing),
    "RectilinearGrid": Codec(RectilinearGrid, _encode_grid, _decode_grid),
    "TrackLinearization": Codec(TrackLinearization, _encode_track, _decode_track),
    "HexagonShapes": Codec(HexagonShapes, _encode_hex_shapes, _decode_hex_shapes),
    "TriangleShapes": Codec(
        TriangleShapes, _encode_triangle_shapes, _decode_triangle_shapes
    ),
    "Outline": Codec(Outline, _encode_outline, _decode_outline),
    "PolygonCells": Codec(PolygonCells, _encode_polygon_cells, _decode_polygon_cells),
    "ClipMapping": Codec(ClipMapping, _encode_clip_mapping, _decode_clip_mapping),
    "ClippedLocator": Codec(
        ClippedLocator, _encode_clipped_locator, _decode_clipped_locator
    ),
    "RectilinearLocator": Codec(
        RectilinearLocator,
        _encode_rectilinear_locator,
        _decode_rectilinear_locator,
        version=2,
    ),
    "HexLocator": Codec(HexLocator, _encode_hex_locator, _decode_hex_locator),
    "MeshLocator": Codec(MeshLocator, _encode_mesh_locator, _decode_mesh_locator),
    "NearestBinLocator": Codec(
        NearestBinLocator, _encode_nearest_locator, _decode_nearest_locator, version=2
    ),
    "SubsetLocator": Codec(
        SubsetLocator, _encode_subset_locator, _decode_subset_locator
    ),
    "TransformedLocator": Codec(
        TransformedLocator,
        _encode_transformed_locator,
        _decode_transformed_locator,
        version=2,
    ),
    "Binning": Codec(Binning, _encode_binning, _decode_binning, version=5),
    "FaceNormals": Codec(FaceNormals, _encode_face_normals, _decode_face_normals),
    "FramedGrid": Codec(FramedGrid, _encode_grid_frame, _decode_grid_frame),
}


def _wkb(geometry: shapely.Geometry) -> NDArray[np.uint8]:
    result: NDArray[np.uint8] = np.frombuffer(
        shapely.to_wkb(geometry), dtype=np.uint8
    ).copy()
    return result


def encode_arena(arena: Arena) -> tuple[dict[str, Any], dict[str, NDArray[Any]]]:
    """Encode built-in arena data with local array keys, without I/O.

    Parameters
    ----------
    arena : Arena
        A supported immutable arena value.

    Returns
    -------
    metadata : dict
        JSON-compatible tagged data with codec version one.
    arrays : dict of ndarray
        Independent writable numerical arrays using local field names.
    """
    meta: dict[str, Any] = {
        "codec": type(arena).__name__,
        "codec_version": _ARENA_VERSION,
    }
    arrays: dict[str, NDArray[Any]] = {}
    if type(arena) is _CombinedArena:
        entries = []
        for index, child in enumerate(arena.children):
            entry, values = encode_arena(child)
            entries.append(entry)
            arrays.update(
                {f"children/{index}/{key}": value for key, value in values.items()}
            )
        meta["children"] = entries
    elif type(arena) is UnknownArena:
        meta.update(n_dims=arena.n_dims, coordinate_kind=arena.coordinate_kind)
    elif type(arena) is OpenArena:
        meta.update(
            n_walls=len(arena.walls),
            wall_names=list(arena.wall_names),
            physical_footprint_known=arena.physical_footprint_known,
        )
        arrays["polygon_wkb"] = _wkb(arena.polygon)
        arrays.update(
            {f"walls_wkb_{i}": _wkb(wall) for i, wall in enumerate(arena.walls)}
        )
    elif type(arena) is MaskArena:
        meta.update(
            n_dims=arena.n_dims, physical_footprint_known=arena.physical_footprint_known
        )
        arrays.update(
            {f"edges_{axis}": edge.copy() for axis, edge in enumerate(arena.edges)}
        )
        arrays["active_mask"] = arena.active_mask.copy()
    elif type(arena) is TrackArena:
        meta["width"] = arena.width
        arrays["segments"] = arena.segments.copy()
    elif type(arena) is EgocentricSurround:
        meta.update(
            distance_range=list(arena.distance_range),
            angle_range=list(arena.angle_range),
            circular=arena.circular,
        )
    elif type(arena) is _TransformedArena:
        base_meta, base_arrays = encode_arena(arena.base)
        meta["base"] = base_meta
        arrays["forward"] = arena.forward.copy()
        arrays.update({f"base/{key}": value for key, value in base_arrays.items()})
    else:
        raise _error(
            f"Cannot encode unregistered arena type {type(arena).__name__!r}.",
            "use a built-in arena type with a supported codec",
        )
    return meta, arrays


def _count(meta: Mapping[str, Any], name: str, *, minimum: int = 0) -> int:
    value = meta[name]
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise _error(
            f"{name} must be an integer >= {minimum}.", "restore valid arena metadata"
        )
    return int(value)


def _read_wkb(arrays: Mapping[str, NDArray[Any]], name: str) -> shapely.Geometry:
    data = np.asarray(arrays[name])
    if data.ndim != 1 or data.dtype != np.uint8:
        raise _error(
            f"{name} must be a one-dimensional uint8 WKB array.",
            "restore valid arena geometry bytes",
        )
    return cast("shapely.Geometry", shapely.from_wkb(data.tobytes()))


def _check_arena_nesting(meta: Mapping[str, Any]) -> None:
    """Bound composite nesting without limiting valid affine histories."""
    stack: list[tuple[object, int, bool]] = [(meta, 0, False)]
    active: set[int] = set()
    while stack:
        value, depth, leaving = stack.pop()
        if leaving:
            active.remove(id(value))
            continue
        if depth > 32:
            raise ValueError("Arena nesting exceeds 32 levels; flatten the domains.")
        if not isinstance(value, Mapping):
            continue
        if id(value) in active:
            raise ValueError("Cyclic arena metadata cannot describe a valid domain.")
        active.add(id(value))
        stack.append((value, depth, True))
        if value.get("codec") == "_CombinedArena":
            children = value.get("children", ())
            if isinstance(children, (list, tuple)):
                stack.extend((child, depth + 1, False) for child in children)
        elif value.get("codec") == "_TransformedArena":
            stack.append((value.get("base", {}), depth, False))


def decode_arena(meta: Mapping[str, Any], arrays: Mapping[str, NDArray[Any]]) -> Arena:
    """Validate and own an arena payload, without builders or I/O.

    Parameters
    ----------
    meta : mapping
        Tagged arena metadata produced by :func:`encode_arena`.
    arrays : mapping of ndarray
        Local arrays; all published array buffers are copied.

    Returns
    -------
    Arena
        Validated immutable geometry with protected owned arrays.

    Raises
    ------
    ValueError
        If a codec/version is unknown or stored geometry is malformed.
    """
    _check_arena_nesting(meta)
    try:
        kind = meta.get("codec")
        if isinstance(kind, str) and kind not in {
            "UnknownArena",
            "OpenArena",
            "MaskArena",
            "TrackArena",
            "EgocentricSurround",
            "_TransformedArena",
            "_CombinedArena",
        }:
            snapshot = _decode_snapshot(meta, arrays, "")
            if not isinstance(snapshot, Arena):
                raise ValueError("The installed adapter must decode an arena snapshot.")
            return snapshot
        if _count(meta, "codec_version", minimum=1) != _ARENA_VERSION:
            raise _error(
                "Unsupported arena codec version.",
                "read this record with a version that supports its arena codec",
            )
        kind = meta["codec"]
        if kind == "_CombinedArena":
            return _CombinedArena(
                tuple(
                    decode_arena(
                        entry,
                        {
                            key.removeprefix(f"children/{index}/"): value
                            for key, value in arrays.items()
                            if key.startswith(f"children/{index}/")
                        },
                    )
                    for index, entry in enumerate(meta["children"])
                )
            )
        if kind == "UnknownArena":
            return UnknownArena(
                _count(meta, "n_dims", minimum=1),
                coordinate_kind=meta["coordinate_kind"],
            )
        if kind == "OpenArena":
            return OpenArena(
                cast("Polygon | MultiPolygon", _read_wkb(arrays, "polygon_wkb")),
                walls=tuple(
                    cast("LineString", _read_wkb(arrays, f"walls_wkb_{i}"))
                    for i in range(_count(meta, "n_walls"))
                ),
                wall_names=meta["wall_names"],
                physical_footprint_known=_boolean(
                    meta["physical_footprint_known"], "physical_footprint_known"
                ),
            )
        if kind == "MaskArena":
            return MaskArena(
                tuple(
                    arrays[f"edges_{axis}"]
                    for axis in range(_count(meta, "n_dims", minimum=1))
                ),
                arrays["active_mask"],
                physical_footprint_known=_boolean(
                    meta["physical_footprint_known"], "physical_footprint_known"
                ),
            )
        if kind == "TrackArena":
            return TrackArena(arrays["segments"], meta["width"])
        if kind == "EgocentricSurround":
            return EgocentricSurround(
                meta["distance_range"],
                meta["angle_range"],
                _boolean(meta["circular"], "circular"),
            )
        if kind == "_TransformedArena":
            base_arrays = {
                key.removeprefix("base/"): value
                for key, value in arrays.items()
                if key.startswith("base/")
            }
            return _TransformedArena(
                decode_arena(meta["base"], base_arrays), arrays["forward"]
            )
        raise _error(
            f"Unknown arena codec {kind!r}.",
            "install a version supporting this required arena codec",
        )
    except (KeyError, TypeError, shapely.errors.GEOSException) as exc:
        raise _error(
            f"Malformed arena payload: {exc}.", "restore a complete valid arena record"
        ) from exc


def _arena_types() -> tuple[type[Any], ...]:
    return (
        UnknownArena,
        OpenArena,
        MaskArena,
        TrackArena,
        EgocentricSurround,
        _TransformedArena,
        _CombinedArena,
    )


@lru_cache(maxsize=1)
def _snapshot_adapters() -> tuple[SnapshotAdapter, ...]:
    """Discover installed objects once, rejecting ambiguous declarations."""

    reserved = set(_CODECS) | {kind.__name__ for kind in _arena_types()}
    adapters: list[SnapshotAdapter] = []
    names: set[str] = set()
    providers: set[tuple[type[Any], str]] = set()
    entries = metadata.entry_points(group=_ENTRY_POINT_GROUP)
    for entry in sorted(entries, key=lambda item: (item.name, item.value)):
        try:
            adapter = entry.load()
        except ImportError as exc:
            raise ValueError(
                f"Cannot import snapshot adapter entry point {entry.value!r}. "
                "Install its package and required dependencies."
            ) from exc
        if not isinstance(adapter, SnapshotAdapter):
            raise TypeError(
                f"Snapshot entry point {entry.value!r} must expose a "
                "SnapshotAdapter object, rather than a class or factory."
            )
        if adapter.name in reserved:
            raise ValueError(
                f"Snapshot adapter {adapter.name!r} conflicts with a built-in codec. "
                "Choose a globally unique adapter name."
            )
        if adapter.name in names:
            raise ValueError(
                f"duplicate snapshot adapter name {adapter.name!r}. "
                "Rename the adapter or remove the conflicting installed package."
            )
        provider_role = (adapter.provider_type, adapter.role)
        if provider_role in providers:
            raise ValueError(
                "duplicate snapshot adapter for provider "
                f"{adapter.provider_type.__module__}.{adapter.provider_type.__qualname__} "
                f"and role {adapter.role!r}. Keep one installed adapter per provider role."
            )
        names.add(adapter.name)
        providers.add(provider_role)
        adapters.append(adapter)
    return tuple(adapters)


def _normalize_snapshot(value: object, role: _SnapshotRole) -> object:
    """Validate an adapter result and rebuild owned canonical data records."""
    if role == "arena":
        if type(value) not in _arena_types():
            raise ValueError("snapshot adapters must return a built-in arena record.")
        meta, arrays = encode_arena(cast("Arena", value))
        return decode_arena(meta, arrays)

    expected = {
        "locator": _LOCATOR_TYPES,
        "track": (TrackLinearization,),
        "binning": (Binning,),
    }
    if type(value) not in expected[role]:
        raise ValueError(f"snapshot adapters must return a built-in {role} record.")
    meta, arrays = encode(value, "snapshot")
    return decode(meta, arrays, "snapshot")


def snapshot_provider(provider: object, role: _SnapshotRole) -> object:
    """Convert a registered external provider to owned built-in scientific data."""
    for adapter in _snapshot_adapters():
        if type(provider) is adapter.provider_type and role == adapter.role:
            return _normalize_snapshot(adapter.snapshot(provider), role)
    if role == "arena":
        name = type(provider).__name__
        fix = (
            "wrap the polygon in an OpenArena (from neurospatial.arena import "
            "OpenArena) and pass arena=OpenArena(polygon)"
            if isinstance(provider, BaseGeometry)
            else "build the arena with a class from neurospatial.arena, for "
            "example arena=OpenArena(polygon) for a polygonal boundary, or "
            f"install the package that converts {name} to one"
        )
        raise TypeError(
            _format_error(
                f"arena= takes an arena from neurospatial.arena (OpenArena, "
                f"MaskArena, TrackArena or UnknownArena, not a subclass), got "
                f"{name}.",
                why="Why: the environment stores its arena as data it can "
                "save, transform and use to test containment, and no installed "
                f"converter turns a {name} into that data.",
                fix=fix,
            )
        )
    hint = (
        " A layout engine instead supplies its geometry from a binning() "
        "method that returns a Binning."
        if role == "binning"
        else ""
    )
    raise TypeError(
        f"No installed snapshot adapter for {type(provider).__name__} "
        f"with role {role!r}. Install the provider's adapter package or supply "
        f"supported built-in snapshot data.{hint}"
    )


def _decode_snapshot(
    meta: Mapping[str, Any], arrays: Mapping[str, NDArray[Any]], prefix: str
) -> object:
    """Read an installed custom format and return only built-in snapshot data."""
    name = meta.get("codec")
    for adapter in _snapshot_adapters():
        if adapter.name != name:
            continue
        version = meta.get("codec_version")
        if type(version) is not int or version != adapter.version:
            raise ValueError(
                f"Cannot load {name!r} codec version {version!r}; the installed "
                f"adapter supports version {adapter.version}. Install an adapter "
                "package supporting the saved version."
            )
        detached_meta = copy.deepcopy(dict(meta))
        detached_arrays = {
            key: np.array(value, copy=True) for key, value in arrays.items()
        }
        try:
            value = adapter.decode(detached_meta, detached_arrays, prefix)
            return _normalize_snapshot(value, adapter.role)
        except (KeyError, TypeError, ValueError, IndexError, OverflowError) as exc:
            raise ValueError(
                f"invalid {name} snapshot payload at {prefix}: {exc}"
            ) from exc
    raise ValueError(
        f"cannot load {name!r}: no codec or installed snapshot adapter is "
        "registered for it. Fix: install the package providing this snapshot "
        "adapter, then load the original record and save canonical built-in data."
    )
