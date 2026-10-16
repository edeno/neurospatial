"""Owned environment records and protected views of published geometry."""

from __future__ import annotations

import copy as copy_module
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    NoReturn,
    TypeVar,
    cast,
    overload,
)

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error
from neurospatial.geometry._binning import Binning
from neurospatial.geometry._coordinates import CartesianSystem, EgocentricPolarSystem
from neurospatial.regions import Regions

if TYPE_CHECKING:
    from neurospatial.arena import Arena
    from neurospatial.environment.core import Environment
    from neurospatial.geometry._context import GridIndex
    from neurospatial.geometry._diffusion import DiffusionGeometry
    from neurospatial.geometry._polygon import ClipMapping
    from neurospatial.layout.base import LayoutEngine

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


def _immutable_array(array: NDArray[Any]) -> NDArray[Any]:
    """Own immutable bytes once so exported headers cannot change cached data."""
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _diffusion_geometry_header(geometry: DiffusionGeometry) -> DiffusionGeometry:
    """Export independent sparse/array headers over immutable cached buffers."""
    from scipy import sparse

    from neurospatial.geometry._diffusion import DiffusionGeometry

    weights = sparse.csr_matrix(geometry.W, copy=False)
    weights.data = _array_header(geometry.W.data)
    weights.indices = _array_header(geometry.W.indices)
    weights.indptr = _array_header(geometry.W.indptr)
    return DiffusionGeometry(
        weights,
        _array_header(geometry.volumes),
        geometry.n_components,
        _array_header(geometry.labels),
    )


def _freeze_diffusion_geometry(geometry: DiffusionGeometry) -> DiffusionGeometry:
    """Own diffusion buffers once before publishing their read-only headers."""
    from scipy import sparse

    from neurospatial.geometry._diffusion import DiffusionGeometry

    weights = sparse.csr_matrix(geometry.W, copy=False)
    weights.data = _immutable_array(geometry.W.data)
    weights.indices = _immutable_array(geometry.W.indices)
    weights.indptr = _immutable_array(geometry.W.indptr)
    return DiffusionGeometry(
        weights,
        _immutable_array(geometry.volumes),
        geometry.n_components,
        _immutable_array(geometry.labels),
    )


def _refuse_graph_edit(*args: Any, **kwargs: Any) -> NoReturn:
    raise TypeError(
        _format_error(
            "Environment.connectivity attributes are read-only.",
            why="Why: editing geometry would leave cached kernels and operators stale.",
            fix="build a new environment; use env.to_networkx() for an editable graph for separate analyses",
        )
    )


_IMMUTABLE_SCALARS = (bool, int, float, complex, str, bytes, type(None), np.generic)


def _frozen_value(value: Any) -> Any:
    """Return an immutable equivalent of a graph attribute value.

    Scalars are kept. Lists and tuples become tuples, numeric arrays become
    read-only arrays over their own buffer, sets become frozensets and
    mappings become read-only attribute mappings, recursively. Any other
    object is copied once; neurospatial never reads it.
    """
    if isinstance(value, _IMMUTABLE_SCALARS):
        return value
    if isinstance(value, (list, tuple)):
        return tuple(_frozen_value(item) for item in value)
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            return _frozen_value(value.tolist())
        return _immutable_array(value)
    if type(value) is _ReadOnlyGraphAttributes:
        return value
    if isinstance(value, Mapping):
        return _ReadOnlyGraphAttributes(value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_frozen_value(item) for item in value)
    return copy_module.deepcopy(value)


def _thawed_value(value: Any) -> Any:
    """Return an editable equivalent of a frozen attribute mapping value."""
    if type(value) is _ReadOnlyGraphAttributes:
        return value.copy()
    return value


class _ReadOnlyGraphAttributes(dict[str, Any]):
    """Published node or edge attributes, with immutable values.

    Values are converted once, when the attributes are created, so reads
    return them without copying. Every edit is refused.
    """

    __slots__ = ()

    def __init__(self, attributes: Mapping[str, Any] | None = None, **kwargs: Any):
        merged = {**(attributes or {}), **kwargs}
        super().__init__((key, _frozen_value(value)) for key, value in merged.items())

    def copy(self) -> dict[str, Any]:
        """Return an editable dict with the same keys and values."""
        return {key: _thawed_value(value) for key, value in self.items()}

    __setitem__ = _refuse_graph_edit
    __delitem__ = _refuse_graph_edit
    clear = _refuse_graph_edit
    pop = _refuse_graph_edit
    popitem = _refuse_graph_edit
    setdefault = _refuse_graph_edit
    update = _refuse_graph_edit
    __ior__ = _refuse_graph_edit

    def __copy__(self) -> _ReadOnlyGraphAttributes:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> _ReadOnlyGraphAttributes:
        return self

    def __reduce__(self) -> tuple[Any, tuple[dict[str, Any]]]:
        return type(self), (dict(self),)


class _ReadOnlyGraphMetadata(dict[str, Any]):
    """Detached JSON-compatible metadata that refuses ordinary graph edits."""

    __setitem__ = _refuse_graph_edit
    __delitem__ = _refuse_graph_edit
    clear = _refuse_graph_edit
    pop = _refuse_graph_edit
    popitem = _refuse_graph_edit
    setdefault = _refuse_graph_edit
    update = _refuse_graph_edit
    __ior__ = _refuse_graph_edit

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
        return _ReadOnlyGraphMetadata(copy_module.deepcopy(attributes))

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


def _published_graph(source: nx.Graph) -> _PublishedGraph:
    """Build the frozen graph an environment publishes, leaving ``source`` as is.

    Node and edge attribute values are converted to immutable values once.
    Node order, adjacency order and edge orientation follow ``source``.
    Graph-level metadata is copied once and exported as a detached copy.
    """
    graph = _PublishedGraph()
    graph.__dict__["graph"] = copy_module.deepcopy(dict(source.graph))
    nodes: dict[Any, _ReadOnlyGraphAttributes] = {}
    for node, attrs in source._node.items():
        nodes[node] = _ReadOnlyGraphAttributes(
            attrs,
            pos=tuple(float(value) for value in attrs["pos"]),
            original_grid_nd_index=tuple(
                int(value) for value in attrs["original_grid_nd_index"]
            ),
        )
    # Undirected NetworkX graphs share one attribute dict between (u, v) and
    # (v, u); freeze each edge once and keep that sharing.
    frozen_edges: dict[int, _ReadOnlyGraphAttributes] = {}
    adjacency: dict[Any, dict[Any, _ReadOnlyGraphAttributes]] = {}
    for u, neighbors in source._adj.items():
        row = adjacency[u] = {}
        for v, attrs in neighbors.items():
            frozen = frozen_edges.get(id(attrs))
            if frozen is None:
                frozen = frozen_edges[id(attrs)] = _ReadOnlyGraphAttributes(
                    attrs, vector=tuple(float(value) for value in attrs["vector"])
                )
            row[v] = frozen
    graph._node = nodes
    graph._adj = adjacency
    nx.freeze(graph)
    graph._published = True
    return graph


def editable_graph(graph: nx.Graph) -> nx.Graph:
    """Return an independent ordinary graph whose attributes can be edited."""
    result = nx.Graph(graph)
    for attrs in result._node.values():
        for key, value in attrs.items():
            attrs[key] = _thawed_value(value)
    for _, _, attrs in result.edges(data=True):
        for key, value in attrs.items():
            attrs[key] = _thawed_value(value)
    return result


@dataclass(frozen=True)
class _PublishedGeometry:
    """The geometry an environment publishes, replaced whole and never edited.

    ``publish`` builds owned arrays and a frozen graph atomically.
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

    return replace(
        geometry,
        bin_centers=_immutable_array(geometry.bin_centers),
        active_mask=(
            _immutable_array(geometry.active_mask)
            if geometry.active_mask is not None
            else None
        ),
        grid_edges=(
            tuple(_immutable_array(edge) for edge in geometry.grid_edges)
            if geometry.grid_edges is not None
            else None
        ),
    )


_SCALAR_CHOICES = (bool, int, float, str, type(None))


def _scalar_choice(value: Any) -> Any:
    """Return ``value`` as a plain scalar, or ``UNSET`` if it is not one."""
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value[()]
    if isinstance(value, np.generic):
        value = value.item()
    return value if isinstance(value, _SCALAR_CHOICES) else UNSET


def build_choices(params: Mapping[str, Any]) -> dict[str, Any]:
    """Keep the scalar build choices of layout parameters and drop their data.

    A choice is a scalar (bool, number, string or None) or a flat sequence
    of scalars, such as a per-axis ``bin_size``: a list, a tuple or a 1-D
    integer or floating-point array. Sequences become tuples and NumPy
    scalars become Python scalars. Everything else is input data, not a
    choice, and is dropped: boolean and multi-dimensional arrays (masks,
    images, positions), graphs, polygons and nested sequences (grid edges,
    edge orders, ranges). The geometry built from that data is owned by the
    environment's binning.

    Parameters
    ----------
    params : mapping of str to object
        Parameters passed to a layout engine or recorded by a derivation.

    Returns
    -------
    dict of str to object
        The choices, in ``params`` order.
    """
    choices: dict[str, Any] = {}
    for key, value in params.items():
        if (
            isinstance(value, np.ndarray)
            and value.ndim == 1
            and value.dtype.kind in "iuf"
        ):
            value = tuple(value.tolist())
        if isinstance(value, (list, tuple)):
            items = tuple(_scalar_choice(item) for item in value)
            if not any(isinstance(item, _Unset) for item in items):
                choices[key] = items
            continue
        choice = _scalar_choice(value)
        if not isinstance(choice, _Unset):
            choices[key] = choice
    return choices


@dataclass(frozen=True, init=False)
class _Provenance:
    """How the bins were built: a layout label, a category and build choices.

    Provenance is descriptive. Geometry and behavior come from the owned
    ``Binning``, graph and arena, never from these labels. Only scalar build
    choices are kept (see :func:`build_choices`), so a saved environment does
    not grow with the data it was built from.
    """

    layout_type: str
    category: str
    choices: tuple[tuple[str, Any], ...]

    def __init__(
        self, layout_type: str, category: str, build_params: Mapping[str, Any]
    ) -> None:
        object.__setattr__(self, "layout_type", layout_type)
        object.__setattr__(self, "category", category)
        object.__setattr__(self, "choices", tuple(build_choices(build_params).items()))

    @property
    def build_params(self) -> dict[str, Any]:
        """Return the build choices as a new dict."""
        return dict(self.choices)


def declared_metadata(
    cls: type[Environment[Any]],
    *,
    name: str = "",
    units: str | None = None,
    frame: str | None = None,
) -> _Metadata:
    """Return a metadata record for labels a caller declares at construction.

    Unrecognized units warn here, once per declaration; derived environments
    reuse their parent's record and do not warn again.
    """
    cls._validate_units(units)
    return _Metadata(name, units, frame)


_EnvironmentT = TypeVar("_EnvironmentT", bound="Environment[Any]")


def publish(
    cls: type[_EnvironmentT],
    binning: Binning,
    *,
    connectivity: nx.Graph,
    dimension_ranges: Sequence[tuple[float, float]] | None,
    provenance: _Provenance,
    metadata: _Metadata,
    is_linearized_track: bool,
    coordinate_kind: Literal["cartesian", "polar"] = "cartesian",
    regions: Regions | None = None,
    arena: Arena | None = None,
    clip_mapping: ClipMapping | None = None,
    restored_from: str | None = None,
    grid_index: GridIndex | None = None,
    check_center_lookup: bool = True,
) -> _EnvironmentT:
    """Validate owned geometry and publish it as one new environment.

    Every factory, derivation and reader creates its environment here, once.
    The connectivity graph and the binning are owned and validated, the
    arena's physical walls are applied, and a frozen copy of the graph is
    published. The caller's ``connectivity`` graph is never modified.

    Parameters
    ----------
    cls : type
        Environment class to allocate.
    binning : Binning
        Bins, lookup and geometry components, in connectivity node order.
    connectivity : networkx.Graph
        Navigation graph over bin IDs ``0..n_bins-1``.
    dimension_ranges : sequence of (float, float) or None
        Bounding range per coordinate axis.
    provenance : _Provenance
        Layout label, category and build choices.
    metadata : _Metadata
        Name, units and frame.
    is_linearized_track : bool
        Whether ``binning`` must carry a track linearization.
    coordinate_kind : {"cartesian", "polar"}, default="cartesian"
        Meaning of the bin coordinates.
    regions : Regions or None, optional
        Named regions to snapshot.
    arena : Arena or None, optional
        Continuous domain. Omission infers one from the binning.
    clip_mapping : ClipMapping or None, optional
        Source-bin mapping of a clip.
    restored_from : str or None, optional
        Source of a read-back record, named in graph errors.
    grid_index : GridIndex or None, optional
        Precomputed grid lookup context; computed from ``binning`` if omitted.
    check_center_lookup : bool, default=True
        Whether to verify that every bin center locates its own bin.

    Returns
    -------
    Environment
        A new instance of ``cls``.
    """
    from neurospatial._logging import log_environment_created
    from neurospatial.arena import EgocentricSurround
    from neurospatial.environment._arena import (
        apply_walls,
        infer_arena,
        validate_arena,
    )
    from neurospatial.geometry._context import grid_index as compute_grid_index
    from neurospatial.geometry._ownership import own_arena, own_binning
    from neurospatial.geometry._validation import (
        _validate_clip_mapping,
        _validate_components,
        _validate_graph,
    )
    from neurospatial.layout.validation import (
        GraphValidationError,
        validate_connectivity_graph,
    )

    if coordinate_kind not in ("cartesian", "polar"):
        raise ValueError("coordinate_kind must be 'cartesian' or 'polar'.")
    binning = own_binning(binning)
    graph = connectivity
    _validate_graph(binning, graph)
    _validate_components(
        binning,
        is_linearized_track,
        coordinate_kind,
        check_center_lookup=check_center_lookup,
    )
    if clip_mapping is not None:
        binning = replace(binning, clip_mapping=copy_module.deepcopy(clip_mapping))
        _validate_clip_mapping(binning)
    polar = coordinate_kind == "polar"
    index = grid_index if grid_index is not None else compute_grid_index(binning)
    coordinates: CartesianSystem | EgocentricPolarSystem
    if polar:
        if not isinstance(arena, EgocentricSurround):
            raise ValueError(
                _format_error(
                    "A polar environment needs an EgocentricSurround arena, which declares whether the angle wraps around.",
                    why="Why: the angular seam decides which bins are neighbors and how distances are measured, and nothing else records it.",
                    fix="build the environment with Environment.from_polar_egocentric, or re-save an older record with this version",
                )
            )
        coordinates = EgocentricPolarSystem(arena.circular)
    else:
        coordinates = CartesianSystem()
        if arena is None:
            arena = infer_arena(binning, polar=False)
    arena = own_arena(arena)
    validate_arena(arena, binning, polar=polar)
    binning, graph = apply_walls(binning, graph, arena)
    try:
        validate_connectivity_graph(
            graph,
            n_dims=binning.bins.n_dims,
            check_node_attrs=True,
            check_edge_attrs=True,
        )
    except GraphValidationError as error:
        if restored_from is not None:
            defect = str(error).split("This is a layout engine bug")[0].strip()
            raise ValueError(
                _format_error(
                    f"The connectivity graph in {restored_from} is invalid: {defect}",
                    why="Why: a read-back environment uses its stored graph, whose nodes and edges must carry the attributes neurospatial writes.",
                    fix="re-save the environment from a factory-built Environment with to_file, to_dict or write_environment",
                )
            ) from error
        raise ValueError(
            f"Invalid connectivity graph from layout engine '{provenance.layout_type}': {error}"
        ) from error
    graph = _published_graph(graph)
    env = cls.__new__(cls)
    env._metadata = metadata
    env._provenance = provenance
    env._is_linearized_track_env = is_linearized_track
    env._geometry = _freeze_geometry_arrays(
        _PublishedGeometry(
            binning=binning,
            bin_centers=binning.bins.centers,
            connectivity=graph,
            dimension_ranges=None
            if dimension_ranges is None
            else tuple((float(low), float(high)) for low, high in dimension_ranges),
            grid_edges=index.edges,
            grid_shape=index.shape,
            active_mask=index.active_mask,
            arena=arena,
            clip_mapping=binning.clip_mapping,
            coordinates=coordinates,
        )
    )
    env._regions = env._validate_regions(Regions() if regions is None else regions)
    env._reset_analysis_caches()
    log_environment_created(
        env_type=provenance.layout_type,
        n_bins=env.n_bins,
        n_dims=env.n_dims,
        env_name=metadata.name,
    )
    return env


def publish_layout(
    cls: type[_EnvironmentT],
    layout: LayoutEngine,
    *,
    metadata: _Metadata,
    layout_type: str | None = None,
    build_params: Mapping[str, Any] | None = None,
    regions: Regions | None = None,
    arena: Arena | None = None,
    coordinates: CartesianSystem | EgocentricPolarSystem | None = None,
    within_arena: bool = False,
) -> _EnvironmentT:
    """Consume a built layout once and publish only its owned data.

    Parameters
    ----------
    cls : type
        Environment class to allocate.
    layout : LayoutEngine
        Built layout engine; it is read, never modified.
    metadata : _Metadata
        Name, units and frame.
    layout_type : str or None, optional
        Layout label; defaults to the binning kind.
    build_params : mapping or None, optional
        Build choices to record; defaults to the layout's own record.
    regions : Regions or None, optional
        Named regions to snapshot.
    arena : Arena or None, optional
        Continuous domain. Omission infers one from the binning.
    coordinates : CartesianSystem or EgocentricPolarSystem or None, optional
        Coordinate meaning; ``None`` means Cartesian.
    within_arena : bool, default=False
        Keep only the bins whose centers lie in a known Cartesian ``arena``
        (strictly inside a polygonal one). The bins are selected on the
        layout's binning, as ``subset`` selects them, before the single
        publication, so the recorded layout and build choices do not depend
        on whether the arena dropped a bin.

    Returns
    -------
    Environment
        A new instance of ``cls``.
    """
    from neurospatial.layout._construction import built_layout_data

    binning, graph, ranges, tag, category, resolved, index = built_layout_data(layout)
    polar = isinstance(coordinates, EgocentricPolarSystem)
    if within_arena and arena is not None and arena.domain_known and not polar:
        import shapely

        from neurospatial.environment._subset import select_bins

        centers = binning.bins.centers
        domain = arena.polygonal_domain()
        keep = (
            shapely.contains_xy(domain, centers[:, 0], centers[:, 1])
            if domain is not None
            else arena.contains(centers)
        )
        if not np.all(keep):
            binning, graph, ranges = select_bins(binning, graph, keep)
            index = None
    if polar and arena is None:
        from neurospatial.environment._arena import infer_arena

        assert isinstance(coordinates, EgocentricPolarSystem)
        arena = infer_arena(binning, polar=True, circular=coordinates.circular)
    return publish(
        cls,
        binning,
        connectivity=graph,
        dimension_ranges=ranges,
        provenance=_Provenance(
            layout_type or tag,
            category,
            dict(resolved) if build_params is None else dict(build_params),
        ),
        metadata=metadata,
        is_linearized_track=binning.linearization is not None,
        coordinate_kind="polar" if polar else "cartesian",
        regions=regions,
        arena=arena,
        grid_index=index,
        check_center_lookup=index is None,
    )


def publish_derived(
    parent: _EnvironmentT,
    binning: Binning,
    *,
    connectivity: nx.Graph,
    dimension_ranges: Sequence[tuple[float, float]] | None,
    provenance: _Provenance,
    name: str | None = None,
    frame: str | None | _Unset = UNSET,
    regions: Regions | None = None,
    arena: Arena | None = None,
    clip_mapping: ClipMapping | None = None,
    grid_index: GridIndex | None = None,
    check_center_lookup: bool = True,
) -> _EnvironmentT:
    """Publish an environment derived from ``parent``, keeping its labels.

    Derivations (subset, clip, rebin, apply_transform and combine) keep the
    parent's class, name, units, frame, regions and arena unless the
    derivation supplies a replacement. Units are never changed by a
    derivation, and an inherited label is not checked again.

    Parameters
    ----------
    parent : Environment
        The environment the new geometry was derived from.
    binning, connectivity, dimension_ranges, provenance
        The derived geometry, as for :func:`publish`.
    name : str or None, optional
        Replacement name; ``None`` keeps the parent's name.
    frame : str or None, optional
        Replacement frame label; omission keeps the parent's frame.
    regions : Regions or None, optional
        Replacement regions; ``None`` keeps the parent's regions.
    arena : Arena or None, optional
        Replacement arena; ``None`` keeps the parent's arena.
    clip_mapping, grid_index, check_center_lookup
        As for :func:`publish`.

    Returns
    -------
    Environment
        A new instance of ``type(parent)``.
    """
    metadata = _Metadata(
        parent.name if name is None else name,
        parent.units,
        parent.frame if isinstance(frame, _Unset) else frame,
    )
    return publish(
        type(parent),
        binning,
        connectivity=connectivity,
        dimension_ranges=dimension_ranges,
        provenance=provenance,
        metadata=metadata,
        is_linearized_track=binning.linearization is not None,
        coordinate_kind=parent.coordinate_kind,
        regions=parent.regions if regions is None else regions,
        arena=parent._geometry.arena if arena is None else arena,
        clip_mapping=clip_mapping,
        grid_index=grid_index,
        check_center_lookup=check_center_lookup,
    )
