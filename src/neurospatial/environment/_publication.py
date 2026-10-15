"""Owned environment records and protected views of published geometry."""

from __future__ import annotations

import copy as copy_module
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING, Any, Generic, NoReturn, TypeVar, cast, overload

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error
from neurospatial.geometry._binning import Binning
from neurospatial.geometry._coordinates import CartesianSystem, EgocentricPolarSystem

if TYPE_CHECKING:
    from neurospatial.arena import Arena
    from neurospatial.environment.core import Environment
    from neurospatial.geometry._polygon import ClipMapping

_T = TypeVar("_T")


class _Unset(Enum):
    VALUE = "UNSET"

    def __repr__(self) -> str:
        return "UNSET"


UNSET = _Unset.VALUE


@dataclass(frozen=True)
class _Metadata:
    name: str
    units: str | None
    frame: str | None

    def __post_init__(self) -> None:
        for key in ("name", "units", "frame"):
            value = getattr(self, key)
            if not isinstance(value, str) and not (key != "name" and value is None):
                expected = "str" if key == "name" else "str or None"
                raise TypeError(f"Environment.{key} must be a {expected}.")


class _MetadataView(Generic[_T]):
    """A non-shadowable declaration backed by one metadata record."""

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    @overload
    def __get__(self, obj: None, owner: type | None = None) -> _MetadataView[_T]: ...

    @overload
    def __get__(self, obj: object, owner: type | None = None) -> _T: ...

    def __get__(
        self, obj: object | None, owner: type | None = None
    ) -> _MetadataView[_T] | _T:
        if obj is None:
            return self
        return cast("_T", getattr(cast("Any", obj)._metadata, self.name))

    def __set__(self, obj: object, value: object) -> None:
        raise AttributeError(
            f"Environment.{self.name} is read-only; use env.with_metadata({self.name}=...)."
        )

    def __delete__(self, obj: object) -> None:
        self.__set__(obj, None)


def _array_header(array: NDArray[Any]) -> NDArray[Any]:
    """Expose immutable bytes through a header independent of private arrays."""
    buffer: Any = array.base
    while isinstance(buffer, np.ndarray):
        buffer = buffer.base
    assert isinstance(buffer, bytes)
    return np.frombuffer(buffer, dtype=array.dtype).reshape(array.shape)


class _ReadOnlyGraphAttributes(Mapping[str, Any]):
    """Published graph attributes, with pickle and editable-copy support."""

    def __init__(self, attributes: Mapping[str, Any], **kwargs: Any) -> None:
        self._data = copy_module.deepcopy({**attributes, **kwargs})

    def __getitem__(self, key: str) -> Any:
        return copy_module.deepcopy(self._data[key])

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def copy(self) -> dict[str, Any]:
        return copy_module.deepcopy(self._data)

    def _refuse_edit(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise TypeError(
            _format_error(
                "Environment.connectivity attributes are read-only.",
                why="Why: editing geometry would leave cached kernels and operators stale.",
                fix="build a new environment; use env.connectivity.copy() for an editable graph for separate analyses",
            )
        )

    __setitem__ = _refuse_edit
    __delitem__ = _refuse_edit
    clear = _refuse_edit
    pop = _refuse_edit
    popitem = _refuse_edit
    setdefault = _refuse_edit
    update = _refuse_edit
    __ior__ = _refuse_edit

    def __reduce__(self) -> tuple[Any, tuple[dict[str, Any]]]:
        return type(self), (self.copy(),)


class _ReadOnlyGraphMetadata(dict[str, Any]):
    """Detached JSON-compatible metadata that refuses ordinary graph edits."""

    __setitem__ = _ReadOnlyGraphAttributes._refuse_edit
    __delitem__ = _ReadOnlyGraphAttributes._refuse_edit
    clear = _ReadOnlyGraphAttributes._refuse_edit
    pop = _ReadOnlyGraphAttributes._refuse_edit
    popitem = _ReadOnlyGraphAttributes._refuse_edit
    setdefault = _ReadOnlyGraphAttributes._refuse_edit
    update = _ReadOnlyGraphAttributes._refuse_edit
    __ior__ = _ReadOnlyGraphAttributes._refuse_edit

    def __deepcopy__(self, memo: dict[int, Any]) -> dict[str, Any]:
        return copy_module.deepcopy(dict(self), memo)

    def __reduce__(self) -> tuple[Any, tuple[dict[str, Any]]]:
        return dict, (dict(self),)


class _PublishedGraph(nx.Graph):
    """Freeze public graph bindings as well as topology and attribute values.

    NetworkX's cached views remain lazy. Copies construct an ordinary editable
    instance of this subclass; only environment publication seals its bindings.
    """

    @property
    def graph(self) -> dict[str, Any]:
        # node_link_data retains this object directly, unlike node/edge
        # attributes. Return a detached dict so standard JSON encoding works
        # without exposing the authoritative metadata to base-dict mutation.
        attributes = self.__dict__["graph"]
        if not self.__dict__.get("_published", False):
            return cast("dict[str, Any]", attributes)
        return _ReadOnlyGraphMetadata(attributes.copy())

    @graph.setter
    def graph(self, attributes: dict[str, Any]) -> None:
        if self.__dict__.get("_published", False):
            raise AttributeError(
                "Environment.connectivity is read-only; copy the graph to edit it."
            )
        self.__dict__["graph"] = attributes

    def __setattr__(self, name: str, value: Any) -> None:
        if not name.startswith("_") and getattr(self, "_published", False):
            raise AttributeError(
                "Environment.connectivity is read-only; copy the graph to edit it."
            )
        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        if not name.startswith("_") and getattr(self, "_published", False):
            raise AttributeError(
                "Environment.connectivity is read-only; copy the graph to edit it."
            )
        super().__delattr__(name)


def _freeze_graph_attributes(graph: nx.Graph) -> None:
    """Protect node, edge and graph attributes when publishing connectivity."""
    graph._published = False
    # NetworkX's freeze protects topology only. Replace its attribute mappings
    # in place so the layout and environment keep sharing the same graph.
    for node, attrs in graph.nodes(data=True):
        graph._node[node] = _ReadOnlyGraphAttributes(
            attrs,
            pos=tuple(float(value) for value in attrs["pos"]),
            original_grid_nd_index=tuple(
                int(value) for value in attrs["original_grid_nd_index"]
            ),
        )
    for u, v, attrs in graph.edges(data=True):
        frozen = _ReadOnlyGraphAttributes(
            attrs, vector=tuple(float(value) for value in attrs["vector"])
        )
        graph._adj[u][v] = graph._adj[v][u] = frozen
    graph.graph = _ReadOnlyGraphAttributes(graph.graph)
    nx.freeze(graph)
    graph.__class__ = _PublishedGraph
    graph._published = True


@dataclass(frozen=True)
class _PublishedGeometry:
    """The geometry an environment publishes, replaced whole and never edited.

    ``_from_binning`` publishes owned arrays and a frozen graph atomically.
    Each new value owns fresh versioned analysis caches.
    """

    binning: Binning | None
    bin_centers: NDArray[np.float64]
    connectivity: nx.Graph
    dimension_ranges: Sequence[tuple[float, float]] | None
    grid_edges: tuple[NDArray[np.float64], ...] | None
    grid_shape: tuple[int, ...] | None
    active_mask: NDArray[np.bool_] | None
    arena: Arena | None = None
    clip_mapping: ClipMapping | None = None
    coordinates: CartesianSystem | EgocentricPolarSystem | None = None


class _GeometryView(Generic[_T]):
    """Read-only environment attribute backed by its published geometry."""

    def __set_name__(self, owner: type, name: str) -> None:
        self._name = name

    @overload
    def __get__(self, obj: None, objtype: type | None = None) -> _GeometryView[_T]: ...

    @overload
    def __get__(self, obj: Environment[Any], objtype: type | None = None) -> _T: ...

    def __get__(
        self, obj: Environment[Any] | None, objtype: type | None = None
    ) -> _GeometryView[_T] | _T:
        if obj is None:
            return self
        value: _T = getattr(obj._geometry, self._name)
        if isinstance(value, np.ndarray):
            return cast("_T", _array_header(value))
        if isinstance(value, tuple) and self._name == "grid_edges":
            return cast("_T", tuple(_array_header(edge) for edge in value))
        return value

    def __set__(self, obj: object, value: object) -> NoReturn:
        raise AttributeError(
            _format_error(
                f"Environment.{self._name} is read-only.",
                why=(
                    "Why: cached kernels, operators and lookups belong to the "
                    "geometry the environment was built with, so replacing part "
                    "of it would leave them stale."
                ),
                fix=(
                    "build a new environment instead, e.g. with a factory, "
                    "subset, rebin or apply_transform"
                ),
            )
        )

    def __delete__(self, obj: object) -> NoReturn:
        self.__set__(obj, None)


def _freeze_geometry_arrays(geometry: _PublishedGeometry) -> _PublishedGeometry:
    """Own immutable buffers; public descriptors expose separate array headers."""

    def immutable(array: NDArray[Any]) -> NDArray[Any]:
        return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)

    return replace(
        geometry,
        bin_centers=immutable(geometry.bin_centers),
        active_mask=(
            immutable(geometry.active_mask)
            if geometry.active_mask is not None
            else None
        ),
        grid_edges=(
            tuple(immutable(edge) for edge in geometry.grid_edges)
            if geometry.grid_edges is not None
            else None
        ),
    )
