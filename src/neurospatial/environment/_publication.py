"""Owned environment records and the one publisher that builds them."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
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
)

import networkx as nx
import numpy as np
import shapely
from scipy import sparse

from neurospatial._exceptions import _format_error
from neurospatial._logging import log_environment_created
from neurospatial._stacklevel import _external_stacklevel
from neurospatial.arena import EgocentricSurround
from neurospatial.environment._cache import AnalysisCache
from neurospatial.environment._regions import _owned_regions
from neurospatial.geometry import (
    Binning,
    CartesianSystem,
    DiffusionGeometry,
    EgocentricPolarSystem,
    GridIndex,
    apply_walls,
    binning_ranges,
    graph_from_binning,
    infer_arena,
    own_arena,
    own_binning,
    select_bins,
    validate_arena,
    validate_components,
)
from neurospatial.geometry import grid_index as compute_grid_index
from neurospatial.layout import _built_layout_data
from neurospatial.layout.base import build_choices
from neurospatial.regions import Regions

if TYPE_CHECKING:
    from neurospatial.arena import Arena
    from neurospatial.environment.core import Environment
    from neurospatial.layout.base import LayoutEngine

_CoordinatesT = TypeVar(
    "_CoordinatesT", bound="CartesianSystem | EgocentricPolarSystem"
)


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


# Unit labels that tools reading ``units`` (scale bars, simulators) recognize.
_UNITS_REGISTRY: frozenset[str] = frozenset({"cm", "m", "mm", "px", "pixels"})


def _validate_units(value: object) -> None:
    """Warn if ``value`` is not in :data:`_UNITS_REGISTRY`.

    Called when constructing or replacing the units declaration. The
    registry is intentionally small and advisory; the goal is to
    catch typos (``"centimeters"``, ``"px "``) without preventing
    legitimate free-form values.
    """
    if value is None:
        return
    if not isinstance(value, str):
        raise TypeError(
            f"Environment.units must be a str or None, got {type(value).__name__}."
        )
    if value not in _UNITS_REGISTRY:
        warnings.warn(
            f"Environment.units={value!r} is not in the standard "
            f"registry {sorted(_UNITS_REGISTRY)}. Free-form strings "
            "are allowed but may not be interpreted by tools that consume "
            "`units` (e.g., scale bars, simulate_trajectory_ou unit checks).",
            category=UserWarning,
            stacklevel=_external_stacklevel(),
        )


def _replace_metadata(
    metadata: _Metadata,
    *,
    name: str | _Unset,
    units: str | None | _Unset,
    frame: str | None | _Unset,
) -> _Metadata:
    """Return ``metadata`` with the labels a caller supplied replaced.

    An omitted label (``UNSET``) is kept; ``None`` clears ``units`` or
    ``frame``. Only newly declared units are checked against the registry.
    """
    replaced = _Metadata(
        metadata.name if isinstance(name, _Unset) else name,
        metadata.units if isinstance(units, _Unset) else units,
        metadata.frame if isinstance(frame, _Unset) else frame,
    )
    if not isinstance(units, _Unset):
        _validate_units(replaced.units)
    return replaced


def _freeze_diffusion_geometry(geometry: DiffusionGeometry) -> DiffusionGeometry:
    """Mark freshly built diffusion arrays read-only before the cache keeps them."""
    for array in (
        geometry.W.data,
        geometry.W.indices,
        geometry.W.indptr,
        geometry.volumes,
        geometry.labels,
    ):
        array.setflags(write=False)
    return geometry


def _diffusion_geometry_view(geometry: DiffusionGeometry) -> DiffusionGeometry:
    """Return the cached diffusion geometry over new read-only array views.

    Rebinding an attribute of the returned matrix or tuple, such as
    ``W.data``, cannot reach the cached value.
    """
    weights = sparse.csr_matrix(geometry.W, copy=False)
    weights.data = geometry.W.data.view()
    weights.indices = geometry.W.indices.view()
    weights.indptr = geometry.W.indptr.view()
    return DiffusionGeometry(
        weights,
        geometry.volumes.view(),
        geometry.n_components,
        geometry.labels.view(),
    )


def _refuse_graph_edit(*args: Any, **kwargs: Any) -> NoReturn:
    raise TypeError(
        _format_error(
            "Environment.connectivity attributes are read-only.",
            why="Why: editing geometry would leave cached kernels and operators stale.",
            fix="build a new environment; use env.to_networkx() for an editable graph for separate analyses",
        )
    )


class _ReadOnlyGraphAttributes(dict[str, Any]):
    """Published node or edge attributes; every edit is refused.

    The values are scalars and tuples of scalars, derived from the binning,
    so reads return them without copying.
    """

    __slots__ = ()

    def copy(self) -> dict[str, Any]:
        """Return an editable dict with the same keys and values."""
        return dict(self)

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


class _PublishedGraph(nx.Graph):
    """Freeze public graph bindings as well as topology and attribute values.

    NetworkX's cached views remain lazy. Copies construct an ordinary editable
    instance of this subclass; only environment publication seals its bindings.
    """

    @property
    def graph(self) -> dict[str, Any]:
        # A published graph stores read-only metadata, frozen once like node
        # and edge attributes; it is a dict, so JSON encoding works.
        return cast("dict[str, Any]", self.__dict__["graph"])

    @graph.setter
    def graph(self, attributes: dict[str, Any]) -> None:
        if self.__dict__.get("_published", False):
            raise AttributeError(
                "Environment.connectivity is read-only; copy the graph to edit it."
            )
        self.__dict__["graph"] = attributes

    def copy(self, as_view: bool = False) -> nx.Graph:
        """Return an editable copy, with editable graph metadata as well."""
        result = super().copy(as_view=as_view)
        if not as_view and self.__dict__.get("_published", False):
            result.graph = dict(self.graph)
        return result

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


def _published_graph(binning: Binning) -> _PublishedGraph:
    """Build the frozen navigation graph an environment publishes for ``binning``.

    Nodes, attributes and edge orientation follow
    :func:`neurospatial.geometry.graph_from_binning`; each undirected
    edge keeps one attribute record shared by both directions.
    """
    graph = graph_from_binning(
        binning, graph_class=_PublishedGraph, attributes=_ReadOnlyGraphAttributes
    )
    graph.__dict__["graph"] = _ReadOnlyGraphAttributes()
    nx.freeze(graph)
    graph._published = True
    return graph


def editable_graph(graph: nx.Graph) -> nx.Graph:
    """Return an independent ordinary graph whose attributes can be edited."""
    result = nx.Graph(graph)
    result.graph = dict(graph.graph)
    return result


@dataclass(frozen=True, eq=False)
class _PublishedGeometry(Generic[_CoordinatesT]):
    """The geometry an environment publishes, replaced whole and never edited.

    :func:`_publish` builds it once. The binning owns the bins, lookup, faces,
    ``dimension_ranges``, clip mapping and track linearization; the graph and
    grid index are derived from it at publication.

    Attributes
    ----------
    binning : Binning
        Owned bins, lookup, faces and optional components.
    graph : _PublishedGraph
        Frozen navigation graph derived from ``binning``.
    grid : GridIndex
        Logical grid edges, shape and active mask derived from ``binning``.
    arena : Arena
        Owned continuous domain.
    coordinates : CartesianSystem or EgocentricPolarSystem
        Meaning of the bin coordinates.
    """

    binning: Binning
    graph: _PublishedGraph
    grid: GridIndex
    arena: Arena
    coordinates: _CoordinatesT


@dataclass(frozen=True, init=False)
class _Provenance:
    """How the bins were built: scalar build choices and the derivations since.

    Provenance is descriptive. Geometry and behavior come from the owned
    ``Binning``, graph and arena, never from these records. The layout label
    is the binning's ``kind``. Only scalar build choices are kept (see
    :func:`neurospatial.layout.base.build_choices`), so a saved environment
    does not grow with the data it was built from.

    Attributes
    ----------
    choices : tuple of (str, object)
        The build choices, in recorded order.
    derivations : tuple of str
        Names of the derivations (``"subset"``, ``"clip"``, ``"rebin"``,
        ``"apply_transform"``, ``"combine"``) applied since the bins were
        built, oldest first.
    length_scale : float or None
        How long one unit of length in the build choices is in the current
        coordinates. It is 1.0 while the bins are in the coordinates the
        choices were recorded in, is multiplied by the scale of each later
        similarity transform, and is None after any other transform, which
        stretches lengths differently along different directions. A
        ``bin_size`` of 2 with a ``length_scale`` of 3 describes cells 6
        wide now.

    Raises
    ------
    TypeError
        If a build choice is not a scalar or a flat sequence of scalars, or a
        derivation name is not a string.
    ValueError
        If ``length_scale`` is not None or a finite positive number.
    """

    choices: tuple[tuple[str, Any], ...]
    derivations: tuple[str, ...]
    length_scale: float | None

    def __init__(
        self,
        build_params: Mapping[str, Any],
        derivations: tuple[str, ...] = (),
        length_scale: float | None = 1.0,
    ) -> None:
        choices = build_choices(build_params, owner="layout_parameters")
        object.__setattr__(self, "choices", tuple(choices.items()))
        derivations = tuple(derivations)
        if not all(isinstance(step, str) for step in derivations):
            raise TypeError("Derivation names must be strings.")
        object.__setattr__(self, "derivations", derivations)
        if length_scale is not None:
            if (
                isinstance(length_scale, bool)
                or not isinstance(length_scale, (int, float, np.number))
                or not np.isfinite(length_scale)
                or length_scale <= 0
            ):
                raise ValueError(
                    "length_scale must be None or a finite positive number, "
                    f"got {length_scale!r}."
                )
            length_scale = float(length_scale)
        object.__setattr__(self, "length_scale", length_scale)

    @property
    def build_params(self) -> dict[str, Any]:
        """Return the build choices as a new dict."""
        return dict(self.choices)


def _declared_metadata(
    *,
    name: str = "",
    units: str | None = None,
    frame: str | None = None,
) -> _Metadata:
    """Return a metadata record for labels a caller declares at construction.

    Unrecognized units warn here, once per declaration; derived environments
    reuse their parent's record and do not warn again.
    """
    _validate_units(units)
    return _Metadata(name, units, frame)


_EnvironmentT = TypeVar("_EnvironmentT", bound="Environment[Any]")


def _polar_coordinates(arena: Arena | None) -> EgocentricPolarSystem:
    """The polar descriptor of ``arena``, which declares whether the angle wraps."""
    if not isinstance(arena, EgocentricSurround):
        raise ValueError(
            _format_error(
                "A polar environment needs an EgocentricSurround arena, which declares whether the angle wraps around.",
                why="Why: the angular seam decides which bins are neighbors and how distances are measured, and nothing else records it.",
                fix="build the environment with Environment.from_polar_egocentric, or re-save an older record with this version",
            )
        )
    return EgocentricPolarSystem(arena.circular)


# The version of the private records an Environment pickles. Bump it whenever
# a pickled record (metadata, provenance, geometry, binning or locators)
# gains, loses or reinterprets a field, so a pickle from another version is
# refused with the advice to use environment files instead of failing later.
STATE_FORMAT = 2


def environment_state(env_dict: dict[str, Any]) -> dict[str, Any]:
    """The state an Environment pickles: its attributes and the format version."""
    return {**env_dict, "_state_format": STATE_FORMAT}


def _restored_state(state: object) -> dict[str, Any]:
    """Validate an unpickled Environment state and re-own its records.

    The geometry records mark their arrays read-only again as they are
    unpickled; the arena and regions are snapshotted again here.

    Raises
    ------
    TypeError
        If ``state`` is not the state of an Environment pickled by this
        version of neurospatial.
    """
    expected = {
        "_state_format": int,
        "_metadata": _Metadata,
        "_regions": Regions,
        "_provenance": _Provenance,
        "_geometry": _PublishedGeometry,
        "_cache": AnalysisCache,
    }
    if (
        not isinstance(state, dict)
        or state.keys() != expected.keys()
        or not all(isinstance(state[key], kind) for key, kind in expected.items())
        or state["_state_format"] != STATE_FORMAT
    ):
        found = sorted(state) if isinstance(state, dict) else type(state).__name__
        raise TypeError(
            _format_error(
                "This pickle does not hold an Environment saved by this "
                f"version of neurospatial (its state is {found}).",
                why=(
                    "Why: pickle stores the class's private layout, which "
                    "changes between versions; it only moves environments "
                    "between processes running the same version."
                ),
                fix=(
                    "save with env.to_file(path) and read with "
                    "Environment.from_file(path), which work across versions"
                ),
            )
        )
    geometry: _PublishedGeometry = state["_geometry"]
    state = {key: value for key, value in state.items() if key != "_state_format"}
    return {
        **state,
        "_geometry": replace(geometry, arena=own_arena(geometry.arena)),
        "_regions": _owned_regions(state["_regions"], geometry.binning.bins.n_dims),
    }


def _publish(
    cls: type[_EnvironmentT],
    binning: Binning,
    *,
    coordinate_kind: Literal["cartesian", "polar"] = "cartesian",
    provenance: _Provenance,
    metadata: _Metadata,
    regions: Regions | None = None,
    arena: Arena | None = None,
) -> _EnvironmentT:
    """Validate owned geometry and publish it as one new environment.

    Every factory, derivation and reader creates its environment here, once.
    The binning is owned and validated, including that every bin center
    locates its own bin; the arena's physical walls are applied; and the
    navigation graph, grid index and ``dimension_ranges`` are derived from
    the resulting binning. The coordinate descriptor is derived here too: a
    polar environment's :class:`EgocentricSurround` arena is the one record
    of whether its angle wraps.

    Parameters
    ----------
    cls : type
        Environment class to allocate.
    binning : Binning
        Bins, lookup and geometry components. Its ``dimension_ranges``, when
        set, are the environment's; otherwise they come from the cells.
    coordinate_kind : {'cartesian', 'polar'}, default='cartesian'
        Meaning of the bin coordinates. A polar environment needs an
        :class:`EgocentricSurround` ``arena``.
    provenance : _Provenance
        Build choices and derivation history.
    metadata : _Metadata
        Name, units and frame.
    regions : Regions or None, optional
        Named regions to snapshot.
    arena : Arena or None, optional
        Continuous domain. Omission infers a Cartesian one from the binning.

    Returns
    -------
    Environment
        A new instance of ``cls``.
    """
    coordinates: CartesianSystem | EgocentricPolarSystem = (
        _polar_coordinates(arena) if coordinate_kind == "polar" else CartesianSystem()
    )
    binning = own_binning(binning)
    # The published binning carries the environment's ranges, so the two
    # never disagree after a derivation or a read-back.
    binning = replace(binning, dimension_ranges=binning_ranges(binning))
    validate_components(binning, coordinates)
    if arena is None:
        arena = infer_arena(binning)
    arena = own_arena(arena)
    validate_arena(arena, binning, coordinates)
    binning = apply_walls(binning, arena)
    # Built here from the validated bins, so its attributes need no check.
    graph = _published_graph(binning)
    env = cls.__new__(cls)
    env._metadata = metadata
    env._provenance = provenance
    env._geometry = _PublishedGeometry(
        binning, graph, compute_grid_index(binning), arena, coordinates
    )
    env._regions = _owned_regions(
        Regions() if regions is None else regions, binning.bins.n_dims
    )
    env._cache = AnalysisCache()
    log_environment_created(
        env_type=binning.kind,
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
    regions: Regions | None = None,
    arena: Arena | None = None,
    coordinate_kind: Literal["cartesian", "polar"] = "cartesian",
) -> _EnvironmentT:
    """Consume a built layout once and publish only its owned data.

    Only the bins whose centers lie in a known Cartesian ``arena`` (strictly
    inside a polygonal one) are kept. They are selected on the layout's
    binning, as ``subset`` selects them, before the single publication, so
    the recorded layout and build choices do not depend on whether the arena
    dropped a bin.

    Parameters
    ----------
    cls : type
        Environment class to allocate.
    layout : LayoutEngine
        Built layout engine; it is read, never modified.
    metadata : _Metadata
        Name, units and frame.
    regions : Regions or None, optional
        Named regions to snapshot.
    arena : Arena or None, optional
        Continuous domain. Omission infers a Cartesian one from the binning;
        a polar layout needs its :class:`EgocentricSurround`.
    coordinate_kind : {'cartesian', 'polar'}, default='cartesian'
        Meaning of the bin coordinates.

    Returns
    -------
    Environment
        A new instance of ``cls``.
    """
    binning, resolved = _built_layout_data(layout)
    if arena is not None and arena.domain_known and coordinate_kind == "cartesian":
        centers = binning.bins.centers
        domain = arena.polygonal_domain()
        keep = (
            shapely.contains_xy(domain, centers[:, 0], centers[:, 1])
            if domain is not None
            else arena.contains(centers)
        )
        if not np.all(keep):
            binning = select_bins(binning, keep)
    return _publish(
        cls,
        binning,
        coordinate_kind=coordinate_kind,
        provenance=_Provenance(resolved),
        metadata=metadata,
        regions=regions,
        arena=arena,
    )


@dataclass(frozen=True)
class _Derivation:
    """Geometry derived from a parent environment, ready to publish.

    Parameters
    ----------
    binning : Binning
        The derived bins, carrying their ``dimension_ranges`` and any clip
        mapping. Its ``kind`` is the layout label of the result.
    step : str
        Name of the derivation, appended to the derivation history.
    build_params : mapping or None, optional
        Build choices of new bins, in their coordinates; ``None`` keeps the
        parent's choices.
    length_scale : float or None, default=1.0
        The factor by which the derivation multiplies lengths when it keeps
        the parent's choices: the scale of a similarity transform, or None
        for a transform that has no single scale.
    continues_history : bool, default=True
        Append ``step`` to the parent's derivation history. ``False`` starts
        a new history at ``step``, for a result with no single parent.
    regions : Regions or None, optional
        Replacement regions; ``None`` keeps the parent's regions.
    arena : Arena or None, optional
        Replacement arena; ``None`` keeps the parent's arena.
    """

    binning: Binning
    step: str
    build_params: Mapping[str, Any] | None = None
    length_scale: float | None = 1.0
    continues_history: bool = True
    regions: Regions | None = None
    arena: Arena | None = None


def publish_derived(
    parent: _EnvironmentT,
    derivation: _Derivation,
    *,
    name: str | None = None,
    frame: str | None | _Unset = UNSET,
) -> _EnvironmentT:
    """Publish an environment derived from ``parent``, keeping its labels.

    Derivations (subset, clip, rebin, apply_transform and combine) keep the
    parent's class, name, units, frame, regions, arena, coordinate meaning
    and build choices unless the derivation supplies a replacement, and add
    their name to the derivation history. Units are never changed by a
    derivation, and an inherited label is not checked again.

    Parameters
    ----------
    parent : Environment
        The environment the new geometry was derived from.
    derivation : _Derivation
        The derived geometry and any replacement regions or arena.
    name : str or None, optional
        Replacement name; ``None`` keeps the parent's name.
    frame : str or None, optional
        Replacement frame label; omission keeps the parent's frame.

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
    inherited = parent._provenance
    history = inherited.derivations if derivation.continues_history else ()
    params: Mapping[str, Any]
    scale: float | None
    if derivation.build_params is None:
        params = inherited.build_params
        scale = (
            None
            if inherited.length_scale is None or derivation.length_scale is None
            else inherited.length_scale * derivation.length_scale
        )
    else:
        params, scale = derivation.build_params, 1.0
    provenance = _Provenance(params, (*history, derivation.step), scale)
    return _publish(
        type(parent),
        derivation.binning,
        coordinate_kind=parent.coordinate_kind,
        provenance=provenance,
        metadata=metadata,
        regions=parent.regions if derivation.regions is None else derivation.regions,
        arena=parent._geometry.arena if derivation.arena is None else derivation.arena,
    )
