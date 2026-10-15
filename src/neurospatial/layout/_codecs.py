"""Pure, versioned codecs for binning geometry and lookup data."""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, cast

import networkx as nx
import numpy as np
import shapely
from numpy.typing import NDArray
from shapely.geometry import MultiPolygon, Polygon

from neurospatial.layout._binning import (
    Binning,
    KnownFaces,
    LayoutLocator,
    MissingFaces,
    SpatialBins,
    TrackLinearization,
)
from neurospatial.layout._capabilities import RectilinearGrid
from neurospatial.layout._components import HexagonShapes, Outline, TriangleShapes
from neurospatial.layout._diffusion import FaceGeometry
from neurospatial.layout._locators import (
    HexLocator,
    MeshLocator,
    NearestBinLocator,
    RectilinearLocator,
    SubsetLocator,
    TransformedLocator,
)
from neurospatial.layout._polygon import ClipMapping, ClippedLocator, PolygonCells

Payload = tuple[dict[str, Any], dict[str, NDArray[Any]]]


@dataclass(frozen=True, eq=False)
class Codec:
    """The Python type and pure functions for a registered geometry kind."""

    kind: type[Any]
    encode: Callable[[Any, str], Payload]
    decode: Callable[[dict[str, Any], Mapping[str, NDArray[Any]], str], object]


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
        layout = (
            f" (layout {type(obj.layout).__name__})"
            if isinstance(obj, LayoutLocator)
            else ""
        )
        raise ValueError(
            f"cannot save {name}{layout}: no codec is registered for it.\n"
            "Fix: use an environment with supported built-in binning data; "
            "layout-backed lookup and custom layout linearization cannot be saved."
        )
    meta, arrays = codec.encode(obj, prefix)
    return _json_safe({"codec": name, "codec_version": 1, **meta}), arrays


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
        Geometry with normalized dtypes and owned, read-only arrays. KD-tree
        and Delaunay memos remain lazy and can be rebuilt from stored data.

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
        raise ValueError(
            f"cannot load {name!r}: no codec is registered for it.\n"
            "Fix: use a neurospatial version supporting this geometry codec."
        )
    version = meta.get("codec_version")
    if type(version) is not int or version != 1:
        raise ValueError(
            f"cannot load {name} codec_version {version!r}; supported version is 1.\n"
            "Fix: recreate and save the environment with this neurospatial version."
        )
    try:
        return codec.decode(meta, arrays, prefix)
    except (KeyError, TypeError, ValueError, IndexError, OverflowError) as exc:
        raise ValueError(f"invalid {name} payload at {prefix}: {exc}") from exc


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
        raise ValueError(
            "cannot save non-finite scalar metadata as JSON.\n"
            "Fix: use finite values or None for missing graph metadata."
        )
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(
        f"cannot save metadata value of type {type(value).__name__}.\n"
        "Fix: store only JSON-compatible scalar values and sequences in graph attributes."
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


def _finite(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite numeric value, not a boolean.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite numeric value.") from exc
    if not np.isfinite(result) or (positive and result <= 0):
        condition = "finite and positive" if positive else "finite"
        raise ValueError(f"{name} must be {condition}.")
    return result


def _grid_edges(
    meta: dict[str, Any], arrays: Mapping[str, NDArray[Any]], prefix: str
) -> tuple[tuple[NDArray[np.float64], ...], tuple[int, ...]]:
    shape = _shape(meta)
    edges = tuple(
        _array(arrays, prefix, f"edges_{i}", np.float64) for i in range(len(shape))
    )
    for axis, edge in enumerate(edges):
        if (
            edge.shape != (shape[axis] + 1,)
            or np.any(~np.isfinite(edge))
            or np.any(np.diff(edge) <= 0)
        ):
            raise ValueError(
                f"edges_{axis} must have shape ({shape[axis] + 1},) "
                "and be finite and strictly increasing."
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
    return {"measure_dim": obj.measure_dim}, _arrays(
        prefix,
        centers=obj.centers,
        sizes=obj.sizes,
        pairs=obj.pairs,
        pair_distances=obj.pair_distances,
    )


def _decode_bins(
    meta: dict[str, Any], arrays: Mapping[str, NDArray[Any]], prefix: str
) -> SpatialBins:
    return SpatialBins.from_arrays(
        _array(arrays, prefix, "centers", np.float64),
        _array(arrays, prefix, "sizes", np.float64),
        meta["measure_dim"],
        _array(arrays, prefix, "pairs", np.intp),
        _array(arrays, prefix, "pair_distances", np.float64),
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
    mask = _array(arrays, prefix, "active_mask", np.bool_)
    if mask.shape != shape:
        raise ValueError(f"active_mask must have shape {shape}, got {mask.shape}.")
    return RectilinearGrid(edges, shape, mask)


def _encode_track(obj: TrackLinearization, prefix: str) -> Payload:
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
    edges, shape = _grid_edges(meta, arrays, prefix)
    if len(shape) != 1:
        raise ValueError("track grid edges must describe one linear coordinate.")
    mask = _array(arrays, prefix, "active_mask", np.bool_)
    if mask.shape != shape:
        raise ValueError(f"active_mask must have shape {shape}, got {mask.shape}.")
    graph = graph_from_node_link(meta["graph"])
    if graph.is_directed() or graph.is_multigraph():
        raise ValueError("track graph must be an undirected simple graph.")
    edge_order = tuple(
        tuple(_node_id(node) for node in edge) for edge in meta["edge_order"]
    )
    if any(len(edge) != 2 or not graph.has_edge(*edge) for edge in edge_order):
        raise ValueError("edge_order must contain pairs from the stored track graph.")
    spacing = meta["edge_spacing"]
    if isinstance(spacing, (list, tuple)):
        spacing = tuple(_finite(value, "edge_spacing") for value in spacing)
    else:
        spacing = _finite(spacing, "edge_spacing")
    return TrackLinearization(graph, edge_order, spacing, edges, mask)


def _encode_hex_shapes(obj: HexagonShapes, prefix: str) -> Payload:
    return {"radius": obj.radius, "orientation": obj.orientation}, _arrays(
        prefix, centers=obj.centers
    )


def _decode_hex_shapes(
    meta: dict[str, Any], arrays: Mapping[str, NDArray[Any]], prefix: str
) -> HexagonShapes:
    centers = _array(arrays, prefix, "centers", np.float64)
    if centers.ndim != 2 or centers.shape[1] != 2 or np.any(~np.isfinite(centers)):
        raise ValueError("centers must be finite with shape (n_bins, 2).")
    return HexagonShapes(
        centers,
        _finite(meta["radius"], "radius", positive=True),
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
    return TriangleShapes(*_mesh_arrays(arrays, prefix, "triangles"))


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
    if obj.active_mask is not None:
        values["active_mask"] = obj.active_mask
    if obj.flat_to_active is not None:
        values["flat_to_active"] = obj.flat_to_active
    return {
        "shape": list(obj.shape),
        "has_active_mask": obj.active_mask is not None,
        "has_flat_to_active": obj.flat_to_active is not None,
    }, _arrays(prefix, **values)


def _decode_rectilinear_locator(
    meta: dict[str, Any], arrays: Mapping[str, NDArray[Any]], prefix: str
) -> RectilinearLocator:
    edges, shape = _grid_edges(meta, arrays, prefix)
    mask = (
        _array(arrays, prefix, "active_mask", np.bool_)
        if _flag(meta, "has_active_mask")
        else None
    )
    mapping = (
        _array(arrays, prefix, "flat_to_active", np.intp)
        if _flag(meta, "has_flat_to_active")
        else None
    )
    if mask is not None and mask.shape != shape:
        raise ValueError(f"active_mask must have shape {shape}, got {mask.shape}.")
    if mapping is not None:
        _active_map(mapping, math.prod(shape), "flat_to_active")
        if mask is not None:
            expected = np.full(mask.size, -1, dtype=np.intp)
            expected[mask.ravel()] = np.arange(np.count_nonzero(mask))
            if not np.array_equal(mapping, expected):
                raise ValueError("flat_to_active does not match active_mask ordering.")
    return RectilinearLocator(edges, shape, mask, mapping)


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
    if len(shape) != 2:
        raise ValueError("centers_shape must describe two hexagonal lattice axes.")
    mapping = _array(arrays, prefix, "flat_to_active", np.intp)
    _active_map(mapping, math.prod(shape), "flat_to_active")
    return HexLocator(
        _finite(meta["offset_x"], "offset_x"),
        _finite(meta["offset_y"], "offset_y"),
        _finite(meta["radius"], "radius", positive=True),
        (shape[0], shape[1]),
        mapping,
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
    return {"max_distance": obj.max_distance}, _arrays(
        prefix, centers=obj.centers, bin_indices=obj.bin_indices
    )


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
    if maximum is not None:
        maximum = _finite(maximum, "max_distance")
        if maximum < 0:
            raise ValueError("max_distance must be non-negative.")
    return NearestBinLocator(centers, maximum, indices)


def _encode_binning(obj: Binning, prefix: str) -> Payload:
    meta: dict[str, Any] = {"kind": obj.kind}
    arrays: dict[str, NDArray[Any]] = {}
    cells: list[PolygonCells] = []
    mappings: list[ClipMapping] = []

    def reference(value: PolygonCells | ClipMapping) -> dict[str, int]:
        snapshots: list[Any] = cells if isinstance(value, PolygonCells) else mappings
        index = next(
            (i for i, snapshot in enumerate(snapshots) if snapshot is value),
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
    if obj.clip_mapping is not None:
        reference(obj.clip_mapping)

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
            nested.update(_arrays(path, inverse=locator.inverse))
        return local, nested

    # Lookup is encoded first so unsupported layout-backed persistence fails
    # with its actionable error before any optional layout work is requested.
    for name in (
        "locator",
        "bins",
        "faces",
        "rectilinear",
        "linearization",
        "shapes",
        "outline",
        "clip_mapping",
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
    arrays.update(_arrays(prefix, inverse=obj.inverse))
    return {"base": base}, arrays


def _decode_transformed_locator(
    meta: dict[str, Any], arrays: Mapping[str, NDArray[Any]], prefix: str
) -> TransformedLocator:
    base = decode(meta["base"], arrays, f"{prefix}/base")
    if not isinstance(base, _LOCATOR_TYPES):
        raise ValueError("base must contain a point-lookup codec.")
    return TransformedLocator(base, _array(arrays, prefix, "inverse", np.float64))


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
        return TransformedLocator(base, _array(arrays, path, "inverse", np.float64))

    expected: dict[str, type | tuple[type, ...]] = {
        "bins": SpatialBins,
        "locator": _LOCATOR_TYPES,
        "faces": (KnownFaces, MissingFaces),
        "rectilinear": RectilinearGrid,
        "linearization": TrackLinearization,
        "shapes": (HexagonShapes, TriangleShapes, PolygonCells),
        "outline": Outline,
        "clip_mapping": ClipMapping,
    }
    part: object | None
    for name, kind in expected.items():
        value = meta.get(name) if name == "clip_mapping" else meta[name]
        if value is None:
            part = None
        elif (
            name == "shapes"
            and isinstance(value, dict)
            and "polygon_cells_ref" in value
        ):
            part = referenced(value, "polygon_cells_ref")
        elif (
            name == "clip_mapping"
            and isinstance(value, dict)
            and "clip_mapping_ref" in value
        ):
            part = referenced(value, "clip_mapping_ref")
        elif name == "locator":
            part = locator_value(value, f"{prefix}/{name}")
        else:
            part = decode(value, arrays, f"{prefix}/{name}")
            if isinstance(part, (PolygonCells, ClipMapping)):
                raise ValueError(
                    "polygon components in a binning need typed snapshot references."
                )
        if part is None and name in {"bins", "locator", "faces"}:
            raise ValueError(f"{name} is required for a binning.")
        if part is not None and not isinstance(part, kind):
            raise ValueError(f"{name} has the wrong geometry codec kind.")
        parts[name] = part
    if not isinstance(meta["kind"], str):
        raise ValueError("kind must be a descriptive string.")
    return Binning(kind=meta["kind"], **parts)


_LOCATOR_TYPES = (
    RectilinearLocator,
    HexLocator,
    MeshLocator,
    NearestBinLocator,
    SubsetLocator,
    TransformedLocator,
    ClippedLocator,
)


_CODECS: dict[str, Codec] = {
    "SpatialBins": Codec(SpatialBins, _encode_bins, _decode_bins),
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
        RectilinearLocator, _encode_rectilinear_locator, _decode_rectilinear_locator
    ),
    "HexLocator": Codec(HexLocator, _encode_hex_locator, _decode_hex_locator),
    "MeshLocator": Codec(MeshLocator, _encode_mesh_locator, _decode_mesh_locator),
    "NearestBinLocator": Codec(
        NearestBinLocator, _encode_nearest_locator, _decode_nearest_locator
    ),
    "SubsetLocator": Codec(
        SubsetLocator, _encode_subset_locator, _decode_subset_locator
    ),
    "TransformedLocator": Codec(
        TransformedLocator, _encode_transformed_locator, _decode_transformed_locator
    ),
    "Binning": Codec(Binning, _encode_binning, _decode_binning),
}
