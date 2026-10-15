"""Core Environment class with mixin inheritance.

This module defines the shared _BaseEnvironment dataclass and the Cartesian
Environment type, assembling specialized domain mixins:

- EnvironmentFactories: Factory methods (from_samples, from_graph, etc.)
- EnvironmentQueries: Graph queries (neighbors, paths and distances)
- EnvironmentSerialization: Save/load methods (to_file, from_file, etc.)
- EnvironmentRegions: Region operations (bins_in_region, region_mask)
- EnvironmentVisualization: Plotting methods (plot, plot_1d)
- EnvironmentMetrics: Environment metrics (boundary_bins and bin attributes)
- EnvironmentFields: Spatial field operations (compute_kernel, smooth, diffuse)
- EnvironmentTrajectory: Transition matrices
- CartesianCoordinates: Cartesian coordinate and trajectory/transform entry points

The _BaseEnvironment class is the dataclass in the hierarchy - all mixins
are plain classes. This design prevents dataclass field inheritance conflicts
while maintaining clean code organization.
"""

from __future__ import annotations

import copy as copy_module
import logging
import warnings
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
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
from scipy import sparse
from typing_extensions import Self

from neurospatial._exceptions import _format_error
from neurospatial._logging import log_environment_created, log_graph_validation
from neurospatial.environment._coordinates import (
    CartesianSystem,
    EgocentricPolarSystem,
    polar_circularity,
)
from neurospatial.environment._metadata import UNSET, _Metadata, _MetadataView, _Unset
from neurospatial.environment.cartesian import (
    CARTESIAN_GRID_OPERATIONS,
    CARTESIAN_OPERATIONS,
)
from neurospatial.environment.decorators import check_fitted, versioned_cached_property
from neurospatial.environment.factories import EnvironmentFactories
from neurospatial.environment.fields import EnvironmentFields, _interpolate
from neurospatial.environment.metrics import (
    EnvironmentMetrics,
    _cartesian_boundary_bins,
)
from neurospatial.environment.queries import EnvironmentQueries
from neurospatial.environment.regions import EnvironmentRegions
from neurospatial.environment.serialization import EnvironmentSerialization
from neurospatial.environment.trajectory import (
    BinSequenceWithRuns,
    EnvironmentTrajectory,
    _bin_sequence,
    _occupancy,
)
from neurospatial.environment.transforms import _rebin, _subset_mask
from neurospatial.environment.visualization import EnvironmentVisualization
from neurospatial.layout._binning import Binning
from neurospatial.layout.base import LayoutEngine
from neurospatial.layout.validation import (
    GraphValidationError,
    validate_connectivity_graph,
)
from neurospatial.ops.calculus import (
    _compute_divergence_operator,
    compute_differential_operator,
)
from neurospatial.regions import Regions

if TYPE_CHECKING:
    import pandas as pd
    import shapely
    from scipy.spatial import cKDTree

    from neurospatial.arena import Arena
    from neurospatial.environment._coordinates import CoordinateSystem
    from neurospatial.environment._protocols import CartesianEnvironmentProtocol
    from neurospatial.layout._polygon import ClipMapping
    from neurospatial.ops._types import KernelMode
    from neurospatial.ops.transforms import Affine2D, AffineND

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from typing_extensions import TypeVar as _CoordinateTypeVar

    _C = _CoordinateTypeVar(
        "_C", bound=CartesianSystem | EgocentricPolarSystem, default=CartesianSystem
    )
else:
    _C = TypeVar("_C", bound=CartesianSystem | EgocentricPolarSystem)

_T = TypeVar("_T")


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


class _PublishedGraph(nx.Graph):
    """Freeze public graph bindings as well as topology and attribute values.

    NetworkX's cached views remain lazy. Copies construct an ordinary editable
    instance of this subclass; only environment publication seals its bindings.
    """

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

    ``_setup_from_layout`` builds a new instance (with read-only arrays and a
    frozen graph) and advances ``_state_version``, which keys every derived
    cache, in the same step.
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
    component_diagonal: bool = False


class _GeometryView(Generic[_T]):
    """Read-only environment attribute backed by its published geometry."""

    def __set_name__(self, owner: type, name: str) -> None:
        self._name = name

    @overload
    def __get__(self, obj: None, objtype: type | None = None) -> _GeometryView[_T]: ...

    @overload
    def __get__(self, obj: _BaseEnvironment, objtype: type | None = None) -> _T: ...

    def __get__(
        self, obj: _BaseEnvironment | None, objtype: type | None = None
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


@dataclass
class _BaseEnvironment(
    EnvironmentFactories,
    EnvironmentQueries,
    EnvironmentSerialization,
    EnvironmentRegions,
    EnvironmentVisualization,
    EnvironmentMetrics,
    EnvironmentFields,
    EnvironmentTrajectory,
):
    """Shared implementation behind Environment and EgocentricPolarEnvironment.

    This base class composes the mixins and holds the dataclass fields and
    core methods (``__init__``, ``copy``, ``_setup_from_layout``, repr, etc.)
    common to every environment type. Concrete environment types subclass it:

    - :class:`Environment` — Cartesian ``(x, y[, z])`` environments.
    - :class:`~neurospatial.environment.polar.EgocentricPolarEnvironment` —
      egocentric polar ``(distance, angle)`` environments built via
      :meth:`Environment.from_polar_egocentric`.

    ``EgocentricPolarEnvironment`` and ``Environment`` are sibling types.
    Cartesian methods belong to ``CartesianCoordinates``, composed only by
    ``Environment``. Polar environments refuse those names through one table,
    and shared methods guard their Cartesian modes with ``_require_cartesian``.
    Both concrete types satisfy ``EnvironmentProtocol``; the Cartesian type
    also satisfies ``CartesianEnvironmentProtocol``.

    The original ``Environment`` docstring (kept on the concrete subclass):

    This class serves as a comprehensive model of a spatial environment,
    discretized into bins or nodes. It stores the geometric properties of these
    bins (e.g., centers, areas), their connectivity, and provides methods for
    various spatial queries and operations.

    Instances are typically created using one of the provided classmethod
    factories (e.g., `Environment.from_samples(...)`,
    `Environment.from_graph(...)`). These factories handle the underlying
    `LayoutEngine` setup.

    Terminology
    -----------
    **Active Bins**
        In neuroscience experiments, an animal typically explores only a subset
        of the physical environment. "Active bins" are spatial bins that contain
        data (e.g., position samples) or meet specified criteria (e.g., minimum
        sample count). Only active bins are included in the environment's
        `bin_centers` and `connectivity` graph.

        This filtering is scientifically important because:

        - **Meaningful analysis**: Neural activity (e.g., place fields) can only
          be computed in locations the animal actually visited
        - **Computational efficiency**: Excludes empty regions, reducing memory
          and computation costs
        - **Statistical validity**: Prevents analysis of bins with insufficient
          data

        For example, in a plus maze experiment, only the maze arms are active;
        the surrounding room is excluded. In an open field with a circular
        boundary, only bins inside the circle are active.

        The `infer_active_bins` parameter in `Environment.from_samples()` controls
        whether bins are automatically filtered based on data presence. Additional
        parameters (`bin_count_threshold`, `dilate`, `fill_holes`, `close_gaps`)
        provide fine-grained control over which bins are considered active.

    Choosing a Factory Method
    --------------------------
    The `Environment` class provides six factory methods for creating environments.
    Choose based on your data format and use case:

    **Most Common (ordered by frequency of use)**

    1. **from_samples** - Discretize position data into bins
       Use when you have a collection of position samples (e.g., animal tracking
       data) and want to automatically infer the spatial extent and active bins.
       Supports automatic filtering, morphological operations (dilate, fill_holes,
       close_gaps), and flexible bin size specification.
       See `from_samples()`.

    2. **from_polygon** - Create grid masked by a polygon boundary
       Use when your environment has a well-defined geometric boundary (e.g.,
       circular arena, irregular enclosure) specified as a Shapely polygon. The
       grid is automatically clipped to the polygon interior.
       See `from_polygon()`.

    3. **from_graph** - Create 1D linearized track environment
       Use when analyzing data on tracks or mazes where 2D position should be
       projected onto a 1D linearized representation. Supports automatic
       linearization and conversion between 2D and 1D coordinates.
       See `from_graph()`.

    **Specialized Use Cases**

    4. **from_grid_mask** - Create environment from pre-computed mask
       Use when you have already determined which bins should be active (e.g.,
       from external analysis) as an N-D boolean array. Requires explicit
       specification of grid edges.
       See `from_grid_mask()`.

    5. **from_pixel_mask** - Create environment from binary image
       Use when your environment boundary is defined by a binary image (e.g.,
       segmentation mask, overhead camera view). Each white pixel becomes a
       potential bin.
       See `from_pixel_mask()`.

    **Advanced**

    6. **from_layout** - Create environment from custom LayoutEngine
       Use when you need full control over the layout engine (e.g., HexagonalLayout,
       TriangularMeshLayout, custom tessellations) or are implementing advanced
       spatial discretization schemes. The factory method `create_layout()` provides
       access to all available layout engines.
       See `from_layout()` and `neurospatial.layout.factories.create_layout()`.

    Attributes
    ----------
    name : str
        A user-defined name for the environment.
    layout : LayoutEngine
        The layout engine instance that defines the geometry and connectivity
        of the discretized space.
    bin_centers : NDArray[np.float64]
        Coordinates of the center of each *active* bin/node in the environment.
        Shape is (n_active_bins, n_dims). Populated by `_setup_from_layout`.
        Read-only: build a new environment to change geometry.
    connectivity : nx.Graph
        A NetworkX graph where nodes are integers from `0` to `n_active_bins - 1`,
        directly corresponding to the rows of `bin_centers`. Edges represent
        adjacency between bins. Populated by `_setup_from_layout` and frozen
        (no node or edge can be added or removed).
    dimension_ranges : Optional[Sequence[Tuple[float, float]]]
        The effective min/max extent `[(min_d0, max_d0), ..., (min_dN-1, max_dN-1)]`
        covered by the layout's geometry. Populated by `_setup_from_layout`.
    grid_edges : Optional[Tuple[NDArray[np.float64], ...]]
        For grid-based layouts, a tuple where each element is a 1D array of
        bin edge positions for that dimension of the *original, full grid*.
        `None` or `()` for non-grid or point-based layouts. Populated by
        `_setup_from_layout`.
    grid_shape : Optional[Tuple[int, ...]]
        For grid-based layouts, the N-D shape of the *original, full grid*.
        For point-based/cell-based layouts without a full grid concept, this
        may be `(n_active_bins,)`. Populated by `_setup_from_layout`.
    active_mask : Optional[NDArray[np.bool_]]
        - For grid-based layouts: An N-D boolean mask indicating active bins
          on the *original, full grid*.
        - For point-based/cell-based layouts: A 1D array of `True` values,
          shape `(n_active_bins,)`, corresponding to `bin_centers`.
        Populated by `_setup_from_layout`.
    regions : RegionManager
        Manages symbolic spatial regions defined within this environment.
    _is_linearized_track_env : bool
        Internal flag indicating if the environment's layout is primarily 1-dimensional.
        Set based on `layout.is_linearized_track`.
    _is_fitted : bool
        Internal flag indicating if the environment has been fully initialized
        and its layout-dependent attributes are populated.
    _layout_type_used : Optional[str]
        The string identifier of the `LayoutEngine` type used to create this
        environment (e.g., "RegularGrid"). For introspection and serialization.
    _layout_params_used : Dict[str, Any]
        A dictionary of the parameters used to build the `LayoutEngine` instance.
        For introspection and serialization.

    See Also
    --------
    neurospatial.layout.factories.create_layout : Create custom layout engines
    Environment.combine : Combine disjoint owned environments

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> # Create environment from position samples
    >>> data = np.random.rand(500, 2) * 100  # 2D data in cm
    >>> env = Environment.from_samples(data, bin_size=5.0, name="OpenField")
    >>> env.n_bins  # doctest: +SKIP
    400
    >>> env.n_dims  # doctest: +SKIP
    2
    """

    name = _MetadataView[str]()

    @property
    def layout(self) -> LayoutEngine:
        """Read-only compatibility bridge over the published geometry."""
        return self._layout if self._is_fitted else copy_module.deepcopy(self._layout)

    @layout.setter
    def layout(self, value: LayoutEngine) -> NoReturn:
        raise AttributeError(
            "Environment.layout is read-only; build a new environment."
        )

    @layout.deleter
    def layout(self) -> NoReturn:
        raise AttributeError(
            "Environment.layout is read-only; build a new environment."
        )

    # --- Attributes populated from the layout instance ---
    # Read-only views of the published geometry (see _PublishedGeometry). Not
    # annotated, so they are class-level descriptors rather than dataclass fields.
    _geometry: _PublishedGeometry = field(init=False, repr=False)
    bin_centers = _GeometryView[NDArray[np.float64]]()
    connectivity = _GeometryView[nx.Graph]()
    dimension_ranges = _GeometryView["Sequence[tuple[float, float]] | None"]()

    # Grid-specific context (populated if layout is grid-based)
    grid_edges = _GeometryView["tuple[NDArray[np.float64], ...] | None"]()
    grid_shape = _GeometryView["tuple[int, ...] | None"]()
    active_mask = _GeometryView["NDArray[np.bool_] | None"]()

    # Region Management
    @property
    def regions(self) -> Regions:
        """Owned immutable named regions; edit a standalone Regions builder."""
        return self._regions

    @regions.setter
    def regions(self, value: Regions) -> NoReturn:
        raise AttributeError(
            "Environment.regions is read-only; use env.with_regions(regions)."
        )

    @regions.deleter
    def regions(self) -> NoReturn:
        raise AttributeError(
            "Environment.regions is read-only; use env.with_regions(regions)."
        )

    # Units and coordinate frames
    units = _MetadataView[str | None]()
    frame = _MetadataView[str | None]()

    # Persistence type tag: io readers use it to restore the concrete class.
    # Analysis and plotting ask _coordinate_system for coordinate semantics.
    @property
    def _is_polar(self) -> bool:
        """Coordinate kind from the owned per-instance descriptor."""
        return isinstance(self._geometry.coordinates, EgocentricPolarSystem)

    @property
    def _coordinate_system(self) -> CoordinateSystem:
        """How this environment measures distance and names its axes."""
        raise NotImplementedError(
            f"{type(self).__name__} must define _coordinate_system."
        )

    def _require_cartesian(self, operation: str) -> CartesianEnvironmentProtocol:
        """This environment typed as Cartesian, or raise naming ``operation``."""
        raise NotImplementedError(
            f"{type(self).__name__} must define _require_cartesian."
        )

    # Internal state
    _is_linearized_track_env: bool = field(init=False)
    _is_fitted: bool = field(init=False, default=False)

    #: Monotonic counter incremented on every documented mutation
    #: (``_setup_from_layout``, ``subset``, ``apply_transform``, ``rebin``).
    #: Downstream caches can read this to detect "was the environment
    #: mutated since I last looked?" without diffing the underlying
    #: arrays. The :func:`versioned_cached_property` decorator uses this
    #: as the cache-invalidation key.
    _state_version: int = field(init=False, default=0)

    # KD-tree cache for spatial queries (populated lazily by map_points_to_bins)
    _kdtree_cache: cKDTree | None = field(init=False, default=None, repr=False)

    # Kernel cache for smoothing operations (keyed by (bandwidth, mode))
    _kernel_cache: dict[
        tuple[float, KernelMode],
        NDArray[np.float64],
    ] = field(init=False, default_factory=dict, repr=False)

    # For introspection and serialization
    _layout_type_used: str | None = field(init=False, default=None)
    _layout_params_used: dict[str, Any] = field(init=False, default_factory=dict)

    def __init__(
        self,
        name: str = "",
        layout: LayoutEngine | None = None,
        layout_type_used: str | None = None,
        layout_params_used: dict[str, Any] | None = None,
        regions: Regions | None = None,
        *,
        units: str | None = None,
        frame: str | None = None,
        restored_from: str | None = None,
        arena: Arena | None = None,
        clip_mapping: ClipMapping | None = None,
        _coordinates: CartesianSystem | EgocentricPolarSystem | None = None,
    ) -> None:
        """Initialize the Environment.

        Note: This constructor is primarily intended for internal use by factory
        methods. Users should typically create Environment instances using
        classmethods like `Environment.from_samples(...)`. The provided
        `layout` instance is assumed to be already built and configured.

        Parameters
        ----------
        name : str, optional
            Name for the environment, by default "".
        layout : LayoutEngine
            A fully built LayoutEngine instance that defines the environment's
            geometry and connectivity.
        layout_type_used : Optional[str], optional
            The string identifier for the type of layout used. If None, it's
            inferred from `layout._layout_type_tag`. Defaults to None.
        layout_params_used : Optional[Dict[str, Any]], optional
            Parameters used to build the layout. If None, inferred from
            `layout._build_params_used`. Defaults to None.

        regions : Regions or None, optional
            Initial named regions; an empty container is created when omitted.
        units : str or None, optional
            Label for the supplied coordinate units; never scales coordinates.
        frame : str or None, optional
            Coordinate frame label for the supplied coordinates.
        restored_from : str, optional
            Where a read-back environment's saved geometry came from; an invalid
            saved graph is then reported against that data.
        arena : Arena, optional
            Continuous domain. When omitted, infer a domain from the supplied
            geometry without asserting walls from unvisited samples.
        _coordinates : CartesianSystem or EgocentricPolarSystem, optional
            Internal owned coordinate descriptor supplied by factories/readers.
        clip_mapping : ClipMapping, optional
            Source-bin mapping for a clipped environment.

        """
        if layout is None:
            raise ValueError(
                _format_error(
                    "[E1006] Environment cannot be constructed directly — use a factory method.\n\nMost common (from positions you recorded):\n    env = Environment.from_samples(positions, bin_size=2.0)\n\nOther factories, chosen by the data you have:\n    from_polygon     — a Shapely polygon boundary\n    from_graph       — a track/maze graph (linearized 1D)\n    from_grid_mask   — an N-D boolean mask + grid edges\n    from_pixel_mask  — a 2D image / pixel mask\n\nAvoid:\n    env = Environment()  # not supported\n\nSee each factory's docstring for its exact arguments, or:\n    https://edeno.github.io/neurospatial/errors/#e1006-environment-constructed-directly",
                    fix="env = Environment.from_samples(positions, bin_size=2.0)",
                    why="Why: a factory builds the geometry and connectivity required by Environment.",
                )
            )

        if layout_type_used is not None:
            from neurospatial.environment._ownership import (
                _NATIVE_LAYOUTS,
                _OwnedLayout,
                own_layout,
            )
            from neurospatial.layout._binning_layout import BinningLayout

            if type(layout) not in (*_NATIVE_LAYOUTS, BinningLayout, _OwnedLayout):
                layout = own_layout(layout)
                layout_type_used = layout._layout_type_tag
                layout_params_used = layout._build_params_used

        self._metadata = _Metadata(name, units, frame)
        type(self)._validate_units(units)
        self._layout = (
            layout if layout_type_used is not None else copy_module.deepcopy(layout)
        )

        self._layout_type_used = (
            layout_type_used
            if layout_type_used
            else getattr(layout, "_layout_type_tag", None)
        )
        self._layout_params_used = copy_module.deepcopy(
            layout_params_used
            if layout_params_used is not None
            else getattr(layout, "_build_params_used", {})
        )

        self._layout_category = layout.layout_type
        self._is_linearized_track_env = layout.is_linearized_track

        # Initialize attributes that will be populated by _setup_from_layout
        # Placeholder geometry until _setup_from_layout publishes the layout's
        self._geometry = _PublishedGeometry(
            binning=None,
            coordinates=_coordinates or CartesianSystem(),
            bin_centers=np.empty((0, 0)),
            connectivity=nx.Graph(),
            dimension_ranges=None,
            grid_edges=(),
            grid_shape=None,
            active_mask=None,
        )
        self._freeze_geometry_arrays()
        self._is_fitted = False  # Will be set by _setup_from_layout
        if layout_type_used is not None:
            # Populate attributes from the built layout
            self._setup_from_layout(
                restored_from=restored_from, arena=arena, clip_mapping=clip_mapping
            )
        if regions is not None:
            if not isinstance(regions, Regions):
                raise TypeError(
                    f"Expected 'regions' to be a Regions instance, got {type(regions)}.",
                )
            self._regions = self._validate_regions(regions)
        else:
            # Initialize with an empty Regions instance if not provided
            self._regions = Regions()._snapshot()

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore published geometry from pickles predating binning snapshots."""
        self.__dict__.update(state)
        legacy_component_policy = "component_diagonal" not in vars(self._geometry)
        if "_layout" not in state:
            self._layout = state["layout"]
        self.__dict__.pop("layout", None)
        if "_metadata" not in state:
            self._metadata = _Metadata(
                state.get("name", ""), state.get("units"), state.get("frame")
            )
        for key in ("name", "units", "frame"):
            self.__dict__.pop(key, None)
        self._regions = self._validate_regions(
            state.get("_regions", state.get("regions", Regions()))
        )
        self.__dict__.pop("regions", None)
        missing_binning = not hasattr(self._geometry, "binning")
        if missing_binning or getattr(self._geometry, "arena", None) is None:
            from neurospatial.arena import UnknownArena

            n_dims = self._geometry.bin_centers.shape[1]
            arena = (
                UnknownArena(
                    n_dims, coordinate_kind="polar" if self._is_polar else "cartesian"
                )
                if n_dims
                else None
            )
            self._geometry = replace(
                self._geometry,
                binning=getattr(self._geometry, "binning", None),
                arena=arena,
                clip_mapping=getattr(self._geometry, "clip_mapping", None),
            )
            if missing_binning and self._is_fitted:
                self._setup_from_layout(arena=arena)
        if self._is_fitted:
            from neurospatial.environment._ownership import own_arena, own_layout

            self._layout = own_layout(self._layout)
            assert self._geometry.arena is not None
            self._geometry = replace(
                self._geometry, arena=own_arena(self._geometry.arena)
            )
            _freeze_graph_attributes(self._geometry.connectivity)
            self._reset_analysis_caches()
        if legacy_component_policy:
            self._geometry = replace(
                self._geometry,
                component_diagonal=bool(
                    self._layout._build_params_used.get(
                        "connect_diagonal_neighbors", False
                    )
                ),
            )
        if getattr(self._geometry, "coordinates", None) is None:
            self._geometry = replace(
                self._geometry,
                coordinates=(
                    EgocentricPolarSystem(
                        polar_circularity(
                            self._geometry.grid_shape,
                            self._geometry.connectivity,
                            self._layout_params_used,
                        )
                    )
                    if self._is_polar
                    else CartesianSystem()
                ),
            )
        self._freeze_geometry_arrays()

    def _freeze_geometry_arrays(self) -> None:
        """Own immutable buffers; public descriptors expose separate array headers."""

        def immutable(array: NDArray[Any]) -> NDArray[Any]:
            return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(
                array.shape
            )

        geometry = self._geometry
        self._geometry = replace(
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

    @property
    def _binning(self) -> Binning:
        """The binning snapshot published with the current geometry."""
        binning = self._geometry.binning
        assert binning is not None
        return binning

    @property
    def _component_diagonal(self) -> bool:
        """Owned logical segmentation policy, normalized once for legacy files."""
        return self._geometry.component_diagonal

    @property
    @check_fitted
    def arena(self) -> Arena:
        """Continuous domain, distinguished from the active analysis bins."""
        arena = self._geometry.arena
        assert arena is not None
        return copy_module.deepcopy(arena)

    @property
    @check_fitted
    def clip_mapping(self) -> ClipMapping | None:
        """Most recent clip's source-bin mapping, or None before clipping."""
        return copy_module.deepcopy(self._geometry.clip_mapping)

    @property
    @check_fitted
    def edge_pairs(self) -> NDArray[np.intp]:
        """Oriented neighbor pairs in gradient and flux order, shape (n_edges, 2).

        This order preserves explicit bin input, and can differ from NetworkX
        edge iteration. The returned array is independent and read-only.
        """
        pairs: NDArray[np.intp] = self._binning.bins.pairs.copy()
        pairs.setflags(write=False)
        return pairs

    @check_fitted
    def to_networkx(self) -> nx.Graph:
        """Return a detached editable graph with bin IDs and weighted edges.

        Nested graph, node and edge attributes are independently owned. Node
        identities are the bin IDs; ``edge_pairs`` defines numerical edge order.
        Standard NetworkX drawing, editing and JSON conversion can use this graph.

        Returns
        -------
        networkx.Graph
            Editable navigation graph, independent of the environment.
        """
        return nx.Graph(self.connectivity)

    def _locate(self, positions: NDArray[np.float64]) -> NDArray[np.intp]:
        """Apply known arena containment before assigning active bins."""
        arena = self._geometry.arena
        assert arena is not None
        if not arena.domain_known:
            return self._binning.locator.locate(positions).astype(np.intp, copy=False)
        positions = np.atleast_2d(np.asarray(positions, dtype=np.float64))
        if positions.ndim != 2 or positions.shape[1] != self.n_dims:
            raise ValueError(
                f"Dimensionality mismatch: expected positions with shape "
                f"(n_positions, {self.n_dims}), got {positions.shape}."
            )
        inside = arena.contains(positions)
        indices = np.full(len(positions), -1, dtype=np.intp)
        if np.any(inside):
            indices[inside] = self._binning.locator.locate(positions[inside])
        return indices

    # ------------------------------------------------------------------
    # Units validation
    # ------------------------------------------------------------------
    #: Documented unit strings. Setting ``env.units`` to a value outside
    #: this registry warns but does not raise — the field is advisory and
    #: free-form strings remain allowed for unusual cases (arbitrary lab
    #: coordinate frames, calibrated pixels, etc.).
    _UNITS_REGISTRY: ClassVar[frozenset[str]] = frozenset(
        {"cm", "m", "mm", "px", "pixels"}
    )

    @classmethod
    def _validate_units(cls, value: object) -> None:
        """Warn if ``value`` is not in :attr:`_UNITS_REGISTRY`.

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
        if value not in cls._UNITS_REGISTRY:
            warnings.warn(
                f"Environment.units={value!r} is not in the standard "
                f"registry {sorted(cls._UNITS_REGISTRY)}. Free-form strings "
                "are allowed but may not be interpreted by tools that consume "
                "`units` (e.g., scale bars, simulate_trajectory_ou unit checks).",
                category=UserWarning,
                stacklevel=3,
            )

    def _reset_analysis_caches(self) -> None:
        """Discard this instance's memoized analyses without clearing shared values."""
        for legacy_name in ("bin_sizes", "boundary_bins", "linearization_properties"):
            self.__dict__.pop(legacy_name, None)
        if self._is_fitted:
            self.clear_cache(kdtree=False, kernels=False, cached_properties=True)
        self._kernel_cache = {}
        self._kdtree_cache = None

    def _replace_value(self) -> Self:
        """Share published values while giving the new environment fresh caches."""
        result = type(self).__new__(type(self))
        result.__dict__.update(self.__dict__)
        result._reset_analysis_caches()
        return result

    def __copy__(self) -> Self:
        return self._replace_value()

    @check_fitted
    def with_metadata(
        self,
        *,
        name: str | _Unset = UNSET,
        units: str | None | _Unset = UNSET,
        frame: str | None | _Unset = UNSET,
    ) -> Self:
        """Return a new environment with corrected coordinate declarations.

        Parameters
        ----------
        name : str, optional
            New name. Omission preserves the current name.
        units, frame : str or None, optional
            New labels. Omission preserves the label; None clears it.
            Labels describe existing coordinates and never rescale them.

        Returns
        -------
        Environment or EgocentricPolarEnvironment
            Same concrete type and geometry, with fresh analysis caches.
        """
        metadata = _Metadata(
            self.name if isinstance(name, _Unset) else name,
            self.units if isinstance(units, _Unset) else units,
            self.frame if isinstance(frame, _Unset) else frame,
        )
        if not isinstance(units, _Unset):
            type(self)._validate_units(metadata.units)
        result = self._replace_value()
        result._metadata = metadata
        return result

    def _validate_regions(self, regions: Regions) -> Regions:
        if not isinstance(regions, Regions):
            raise TypeError("regions must be a Regions instance.")
        n_dims = self._geometry.bin_centers.shape[1]
        if any(region.n_dims != n_dims for region in regions.values()):
            raise ValueError("Region coordinate dimensions must match the environment.")
        return regions._snapshot()

    @check_fitted
    def with_regions(self, regions: Regions) -> Self:
        """Return a new environment owning a snapshot of a region collection.

        Parameters
        ----------
        regions : Regions
            Complete replacement, in iteration order. An empty builder clears
            all regions. Coordinates must match this environment's dimension;
            regions may extend outside the arena.

        Returns
        -------
        Environment or EgocentricPolarEnvironment
            Same concrete type and geometry, with fresh analysis caches.
        """
        snapshot = self._validate_regions(regions)
        result = self._replace_value()
        result._regions = snapshot
        return result

    def __eq__(self, other: object) -> bool:
        """Check equality with another Environment or string.

        Parameters
        ----------
        other : object
            Object to compare with. If string, compares with environment name.

        Returns
        -------
        bool
            True if names match (when comparing with string), NotImplemented otherwise.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> data = np.random.rand(100, 2)
        >>> env = Environment.from_samples(data, bin_size=2.0, name="test")
        >>> env == "test"
        True
        >>> env == "other"
        False
        """
        if isinstance(other, str):
            return self.name == other
        return NotImplemented

    def __repr__(self) -> str:
        """Generate an informative single-line string representation.

        Returns a concise, single-line representation showing the environment's
        name, dimensionality, number of bins, and layout type. This method
        follows Python repr best practices by being informative rather than
        reconstructive for complex objects.

        Returns
        -------
        str
            Single-line string representation of the Environment.

        See Also
        --------
        _repr_html_ : Rich HTML representation for Jupyter notebooks.

        Notes
        -----
        This representation is designed for interactive use and debugging, not
        for reconstruction. For serialization, use the `to_file()` method.

        Examples
        --------
        >>> import numpy as np
        >>> data = np.random.rand(100, 2) * 10
        >>> env = Environment.from_samples(data, bin_size=2.0, name="MyEnv")
        >>> repr(env)  # doctest: +SKIP
        "Environment(name='MyEnv', 2D, 25 bins, RegularGrid)"

        """
        # Use ``repr(self.name)`` so the empty-string case ``name=""``
        # is visibly distinct from the unset case ``name=None``.
        # The previous ``f"'{name}'" if name else 'None'`` collapsed the
        # two into the same string and hid the difference from users
        # debugging an Environment they thought they had named.
        cls_name = type(self).__name__
        if not self._is_fitted:
            return f"{cls_name}(name={self.name!r}, not fitted)"

        # Fitted environments: show name, dims, bins, layout. Truncate
        # very long names but keep ``None`` distinct from ``""``.
        name = self.name
        if isinstance(name, str) and len(name) > 40:
            name = name[:37] + "..."

        try:
            dims_str = f"{self.n_dims}D"
        except (RuntimeError, AttributeError):
            dims_str = "?D"

        n_bins = self.bin_centers.shape[0] if hasattr(self, "bin_centers") else 0

        layout_type = self._layout_type_used or "Unknown"
        if layout_type.endswith("Layout"):
            layout_type = layout_type[:-6]  # Remove 'Layout' suffix

        return f"{cls_name}(name={name!r}, {dims_str}, {n_bins} bins, {layout_type})"

    def __str__(self) -> str:
        """Human-readable summary, equivalent to :meth:`info`.

        ``str(env)`` returns the same multi-line summary that
        ``env.info()`` produces. ``repr(env)`` keeps its single-line
        ``Environment(name=..., 2D, 100 bins, RegularGrid)`` shape for
        interactive debugging and logs.
        """
        if not self._is_fitted:
            return self.__repr__()
        return self.info()

    @staticmethod
    def _html_table_row(label: str, value: str, highlight: bool = False) -> str:
        """Generate a single HTML table row.

        Parameters
        ----------
        label : str
            Row label (left column).
        value : str
            Row value (right column).
        highlight : bool, default=False
            If True, use highlighted background color.

        Returns
        -------
        str
            HTML string for the table row.

        """
        bg_color = "#fffacd" if highlight else "#fff"
        return (
            f'<tr style="background-color: {bg_color};">'
            f'<td style="padding: 6px 12px; border-top: 1px solid #ddd; '
            f'font-weight: bold; color: #555;">{label}</td>'
            f'<td style="padding: 6px 12px; border-top: 1px solid #ddd; '
            f'color: #000;">{value}</td>'
            "</tr>"
        )

    @staticmethod
    def _html_table_header(title: str) -> str:
        """Generate HTML table header row.

        Parameters
        ----------
        title : str
            Title text for the header (already HTML-escaped).

        Returns
        -------
        str
            HTML string for the header row.

        """
        return (
            '<tr style="background-color: #f0f0f0; border-bottom: 2px solid #999;">'
            '<th colspan="2" style="padding: 8px; text-align: left; '
            'font-weight: bold; font-size: 14px;">'
            f"{title}"
            "</th></tr>"
        )

    def _repr_html_(self) -> str:
        """Generate rich HTML representation for Jupyter notebooks.

        This method is automatically called by Jupyter/IPython to display
        Environment objects in a formatted table. It provides more detailed
        information than `__repr__()`, including spatial extent, bin sizes,
        and region counts.

        Returns
        -------
        str
            HTML string with table representation of the Environment.

        See Also
        --------
        __repr__ : Plain text representation.

        Notes
        -----
        The HTML output includes:

        - Environment name and layout type
        - Dimensionality and number of bins
        - Spatial extent (min/max coordinates per dimension)
        - Number of regions (if any)
        - Linearization status (for 1D environments)

        This method follows IPython rich display conventions. Special characters
        in names are HTML-escaped for safety using the standard library's
        `html.escape()` function.

        Examples
        --------
        In a Jupyter notebook, simply evaluate an Environment object:

        >>> import numpy as np
        >>> data = np.random.rand(100, 2) * 10
        >>> env = Environment.from_samples(data, bin_size=2.0, name="MyEnv")
        >>> env  # In Jupyter, displays rich HTML table automatically  # doctest: +SKIP

        This will display a formatted table with environment details.

        """
        import html

        # Escape HTML special characters in name
        name = html.escape(str(self.name) if self.name else "None")

        # Build table rows
        rows = []
        rows.append(self._html_table_header(f"Environment: {name}"))

        # Check if fitted
        if not self._is_fitted:
            rows.append(self._html_table_row("Status", "Not fitted", highlight=True))
            rows.append(
                self._html_table_row("Layout Type", self._layout_type_used or "Unknown")
            )
            return (
                '<div style="margin: 10px;">'
                '<table style="border-collapse: collapse; border: 1px solid #ddd; '
                'font-family: monospace; font-size: 12px;">'
                f"{''.join(rows)}"
                "</table></div>"
            )

        # Fitted environment - show full details
        rows.append(
            self._html_table_row("Layout Type", self._layout_type_used or "Unknown")
        )

        # Dimensions and bins
        try:
            n_dims = self.n_dims
            rows.append(self._html_table_row("Dimensions", str(n_dims)))
        except (RuntimeError, AttributeError):
            rows.append(self._html_table_row("Dimensions", "Unknown"))
            n_dims = None

        n_bins = self.bin_centers.shape[0] if hasattr(self, "bin_centers") else 0
        rows.append(self._html_table_row("Number of Bins", str(n_bins)))

        # Spatial extent
        if hasattr(self, "dimension_ranges") and self.dimension_ranges:
            extent_parts = []
            for dim_idx, (min_val, max_val) in enumerate(self.dimension_ranges):
                extent_parts.append(f"dim{dim_idx}: [{min_val:.2f}, {max_val:.2f}]")
            extent_str = "<br>".join(extent_parts)
            rows.append(self._html_table_row("Spatial Extent", extent_str))

        # Regions
        n_regions = len(self.regions) if hasattr(self, "regions") else 0
        if n_regions > 0:
            rows.append(self._html_table_row("Regions", f"{n_regions} defined"))
        else:
            rows.append(self._html_table_row("Regions", "None"))

        # 1D-specific info
        if (
            n_dims == 1
            and hasattr(self, "is_linearized_track")
            and self.is_linearized_track
        ):
            rows.append(
                self._html_table_row("Linearization", "Available (1D environment)")
            )

        return (
            '<div style="margin: 10px;">'
            '<table style="border-collapse: collapse; border: 1px solid #ddd; '
            'font-family: monospace; font-size: 12px;">'
            f"{''.join(rows)}"
            "</table></div>"
        )

    @check_fitted
    def info(self) -> str:
        """Return a detailed multi-line diagnostic summary of the environment.

        This method provides comprehensive diagnostic information about the
        environment, including geometric properties, layout configuration, and
        spatial characteristics. The output is formatted for readability with
        clear labels and organized sections.

        Returns
        -------
        str
            Multi-line formatted string containing detailed environment information.

        See Also
        --------
        __repr__ : Single-line concise representation for quick inspection.
        _repr_html_ : Rich HTML representation for Jupyter notebooks.

        Notes
        -----
        This method is particularly useful for:

        - Debugging spatial binning issues
        - Verifying environment configuration
        - Understanding the structure of complex environments
        - Documenting environment parameters for reproducibility

        The output includes all critical diagnostic information:

        - Environment name and layout type
        - Spatial dimensionality and bin count
        - Physical extent in each dimension
        - Bin size statistics (uniform or variable)
        - Region of interest count
        - Linearization status (for 1D environments)

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> data = np.random.rand(500, 2) * 100  # 2D data in cm
        >>> env = Environment.from_samples(data, bin_size=5.0, name="OpenField")
        >>> print(env.info())  # doctest: +SKIP
        Environment Information
        =======================
        Name: OpenField
        Layout Type: RegularGridLayout
        Dimensions: 2
        Number of Bins: 400
        <BLANKLINE>
        Spatial Extent:
          Dimension 0: [-2.50, 102.50] (range: 105.00)
          Dimension 1: [-2.50, 102.50] (range: 105.00)
        <BLANKLINE>
        Bin Sizes:
          Dimension 0: 5.00
          Dimension 1: 5.00
        <BLANKLINE>
        Regions: 0

        """
        # Build output line by line
        lines = []

        # Header
        lines.append("Environment Information")
        lines.append("=" * 23)
        lines.append("")

        # Basic information
        name_display = self.name if self.name else "(unnamed)"
        lines.append(f"Name: {name_display}")
        lines.append(f"Layout Type: {self.layout_type}")
        lines.append(f"Dimensions: {self.n_dims}")
        lines.append(f"Number of Bins: {self.n_bins}")
        lines.append("")

        # Spatial extent
        if self.dimension_ranges is not None:
            lines.append("Spatial Extent:")
            for dim_idx, (dim_min, dim_max) in enumerate(self.dimension_ranges):
                dim_range = dim_max - dim_min
                lines.append(
                    f"  Dimension {dim_idx}: [{dim_min:.2f}, {dim_max:.2f}] "
                    f"(range: {dim_range:.2f})"
                )
            lines.append("")
        else:
            lines.append("Spatial Extent: Not available")
            lines.append("")

        # Bin sizes
        lines.append("Bin Sizes:")
        try:
            bin_sizes_array = self.bin_sizes

            # Check if all bins have the same size (uniform)
            if np.allclose(bin_sizes_array, bin_sizes_array[0]):
                # Uniform bin size - for grids, extract per-dimension from grid_edges
                if self.grid_edges and all(len(e) > 1 for e in self.grid_edges):
                    for dim_idx, edges in enumerate(self.grid_edges):
                        dim_sizes = np.diff(edges)
                        if np.allclose(dim_sizes, dim_sizes[0]):
                            lines.append(f"  Dimension {dim_idx}: {dim_sizes[0]:.2f}")
                        else:
                            lines.append(
                                f"  Dimension {dim_idx}: variable "
                                f"(mean: {np.mean(dim_sizes):.2f}, "
                                f"std: {np.std(dim_sizes):.2f})"
                            )
                else:
                    # Non-grid layout or 1D - show the uniform measure
                    measure_name = (
                        "Size"
                        if self.n_dims == 1
                        else "Area"
                        if self.n_dims == 2
                        else "Volume"
                    )
                    lines.append(f"  {measure_name}: {bin_sizes_array[0]:.2f}")
            else:
                # Variable bin sizes
                lines.append(
                    f"  Variable (mean: {np.mean(bin_sizes_array):.2f}, "
                    f"std: {np.std(bin_sizes_array):.2f}, "
                    f"range: [{np.min(bin_sizes_array):.2f}, {np.max(bin_sizes_array):.2f}])"
                )
        except (AttributeError, RuntimeError, ValueError):
            lines.append("  (not available)")
        lines.append("")

        # Regions
        n_regions = len(self.regions) if self.regions else 0
        if n_regions > 0:
            lines.append(f"Regions: {n_regions} defined")
            # Show region names if not too many
            if n_regions <= 5:
                for region_name in self.regions:
                    lines.append(f"  - {region_name}")
            else:
                lines.append("  (use env.regions to inspect all regions)")
        else:
            lines.append("Regions: None")
        lines.append("")

        # 1D-specific information
        if hasattr(self, "is_linearized_track") and self.is_linearized_track:
            lines.append("Linearization: Available (1D environment)")
            lines.append("")

        return "\n".join(lines)

    def _setup_from_layout(
        self,
        *,
        restored_from: str | None = None,
        arena: Arena | None = None,
        clip_mapping: ClipMapping | None = None,
    ) -> None:
        """Refresh Environment geometry and caches from its built LayoutEngine.

        This internal method is called after the `LayoutEngine` is associated
        with the Environment. It copies essential geometric and connectivity
        information from the layout to the Environment's attributes.
        It also applies fallbacks for certain grid-specific attributes if the
        layout is point-based to ensure consistency. It also advances the state
        version, clears all analysis caches and publishes the geometry
        read-only. Readers decode bins, lookup and graph data into a layout
        facade before the normal constructor, so this runs once per environment
        without rebuilding a layout. Swapping in a different layout or changing
        its bins and calling this again is not supported.

        Parameters
        ----------
        restored_from : str, optional
            Where restored geometry came from (e.g. a file path). An invalid
            graph is then reported as a problem with that saved data rather
            than as a layout engine bug.

        Raises
        ------
        ValueError
            If the connectivity graph from the layout engine is invalid
            (missing required node/edge attributes, wrong dimensions, etc.).
            The environment is left unchanged.

        """
        bin_centers = self._layout.bin_centers
        connectivity = getattr(self._layout, "connectivity", nx.Graph())

        # Validate the graph before touching any environment state, so a failure
        # leaves the previous geometry, version and caches consistent.
        # Note: Calculate n_dims directly here since self.n_dims has @check_fitted
        n_dims = bin_centers.shape[1] if bin_centers is not None else 0
        try:
            n_nodes = len(connectivity.nodes)
            n_edges = len(connectivity.edges)
            log_graph_validation(n_nodes=n_nodes, n_edges=n_edges, n_dims=n_dims)
            validate_connectivity_graph(
                connectivity,
                n_dims=n_dims,
                check_node_attrs=True,
                check_edge_attrs=True,
            )
        except GraphValidationError as e:
            if restored_from is not None:
                # Validator messages assume a layout-built graph and blame the
                # engine; keep only the defect for saved data.
                defect = str(e).split("This is a layout engine bug")[0].strip()
                raise ValueError(
                    _format_error(
                        f"The connectivity graph in {restored_from} is invalid: "
                        f"{defect}",
                        why=(
                            "Why: a read-back environment uses its stored graph, "
                            "whose nodes and edges must carry the attributes "
                            "neurospatial writes."
                        ),
                        fix=(
                            "re-save the environment from a factory-built "
                            "Environment with to_file, to_dict or write_environment"
                        ),
                    )
                ) from e
            raise ValueError(
                f"Invalid connectivity graph from layout engine "
                f"'{self._layout._layout_type_tag}': {e}\n\n"
                f"This is a layout engine bug. Please report it at "
                f"https://github.com/edeno/neurospatial/issues"
            ) from e

        from neurospatial.environment._ownership import own_arena, own_layout

        self._layout = own_layout(self._layout)
        self._is_linearized_track_env = self._layout.is_linearized_track
        binning = self._layout._binning
        connectivity = self._layout._owned_connectivity  # type: ignore[attr-defined]
        bin_centers = binning.bins.centers
        if clip_mapping is None:
            clip_mapping = binning.clip_mapping
        elif clip_mapping is not binning.clip_mapping:
            binning = replace(binning, clip_mapping=clip_mapping)
        from neurospatial.layout._binning_layout import _validate_clip_mapping

        _validate_clip_mapping(binning)
        from neurospatial.environment._arena import (
            apply_walls,
            infer_arena,
            validate_arena,
        )

        if arena is None:
            arena = infer_arena(binning, polar=self._is_polar, layout=self._layout)
        arena = own_arena(arena)
        validate_arena(arena, binning, polar=self._is_polar)
        binning, wall_graph = apply_walls(binning, connectivity, arena)
        if wall_graph is not connectivity:
            from neurospatial.layout._binning_layout import BinningLayout

            self._layout = own_layout(
                BinningLayout(
                    binning,
                    connectivity=wall_graph,
                    dimension_ranges=self._layout.dimension_ranges,
                    layout_type_tag=self._layout._layout_type_tag,
                    layout_type=self._layout.layout_type,
                    build_params=self._layout._build_params_used,
                    is_linearized_track=self._is_linearized_track_env,
                    coordinate_kind="polar" if self._is_polar else "cartesian",
                )
            )
            connectivity = self._layout._owned_connectivity  # type: ignore[attr-defined]
        if clip_mapping is not None and len(clip_mapping.parent_bin_indices) != len(
            bin_centers
        ):
            raise ValueError("clip_mapping must have one source index per output bin.")
        grid_edges = getattr(self._layout, "grid_edges", ())
        grid_shape = getattr(self._layout, "grid_shape", None)
        active_mask = getattr(self._layout, "active_mask", None)

        # If it's not a grid layout, grid_shape might be (n_active_bins,),
        # and active_mask might be 1D all True. This is fine.
        # Ensure they are at least None if not applicable from layout
        if grid_shape is None and bin_centers is not None:
            # Fallback for point-based
            grid_shape = (bin_centers.shape[0],)
        if active_mask is None and bin_centers is not None:
            # Fallback for point-based
            active_mask = np.ones(bin_centers.shape[0], dtype=bool)

        # Published geometry is read-only and replaced whole: derived caches are
        # invalidated only through _state_version (bumped below, in the same
        # step), so an in-place edit would leave them stale. Change geometry by
        # building a new environment (a factory, subset, rebin or
        # apply_transform) instead.
        for array in (bin_centers, active_mask, *(grid_edges or ())):
            if isinstance(array, np.ndarray):
                array.setflags(write=False)
        _freeze_graph_attributes(connectivity)
        self._geometry = _PublishedGeometry(
            binning=binning,
            bin_centers=bin_centers,
            connectivity=connectivity,
            dimension_ranges=self._layout.dimension_ranges,
            grid_edges=grid_edges,
            grid_shape=grid_shape,
            active_mask=active_mask,
            arena=arena,
            clip_mapping=clip_mapping,
            component_diagonal=bool(
                self._layout._build_params_used.get("connect_diagonal_neighbors", False)
            ),
            coordinates=(
                EgocentricPolarSystem(
                    polar_circularity(
                        grid_shape, connectivity, self._layout_params_used
                    )
                )
                if self._is_polar
                else CartesianSystem()
            ),
        )

        self._freeze_geometry_arrays()
        self._is_fitted = True
        # Bump the state version so any downstream cache (including the
        # versioned_cached_property values from this and other mixins)
        # treats this environment as a "freshly built" object.
        self._state_version += 1
        self.clear_cache()

        # Log environment creation
        n_bins = self.bin_centers.shape[0] if self.bin_centers is not None else 0
        log_environment_created(
            env_type=self._layout._layout_type_tag,
            n_bins=n_bins,
            n_dims=n_dims,
            env_name=self.name,
        )

    @cached_property
    @check_fitted
    def _source_flat_to_active_node_id_map(self) -> dict[int, int]:
        """Get or create the mapping from original full grid flat indices
        to active bin IDs (0 to n_active_bins - 1).

        The map is cached on the instance for subsequent calls. This method
        is intended for internal use by other Environment or related manager methods.

        Returns
        -------
        Dict[int, int]
            A dictionary mapping `source_grid_flat_index` from graph nodes
            to the `active_bin_id` (which is the graph node ID).

        Raises
        ------
        RuntimeError
            If the connectivity graph is not available, or if all nodes are
            missing the 'source_grid_flat_index' attribute required for the map.

        Notes
        -----
        **Decorator order**: @check_fitted is placed BELOW @cached_property.
        This ensures the cached_property descriptor is outermost and the
        check_fitted validation happens during the initial computation.

        """
        return {
            data["source_grid_flat_index"]: node_id
            for node_id, data in self.connectivity.nodes(data=True)
            if "source_grid_flat_index" in data
        }

    @property
    def is_linearized_track(self) -> bool:
        """Indicate if the environment's layout is primarily 1-dimensional.

        Returns
        -------
        bool
            True if the underlying `LayoutEngine` (`self.layout`) reports
            itself as 1-dimensional (e.g., `GraphLayout`), False otherwise.
            This is determined by `self.layout.is_linearized_track`.

        """
        return self._is_linearized_track_env

    @property
    @check_fitted
    def n_dims(self) -> int:
        """Return the number of spatial dimensions of the active bin centers.

        Returns
        -------
        int
            The number of dimensions (e.g., 1 for a line, 2 for a plane).
            Derived from the shape of `self.bin_centers`.

        Raises
        ------
        RuntimeError
            If called before the environment is fitted or if `bin_centers`
            is not available.

        """
        return int(self.bin_centers.shape[1])

    @property
    @check_fitted
    def layout_parameters(self) -> dict[str, Any]:
        """Return the parameters used to build the layout engine.

        This includes all parameters that were passed to the `build` method
        of the underlying `LayoutEngine`.

        Returns
        -------
        Dict[str, Any]
            A dictionary of parameters used to create the layout.
            Useful for introspection and serialization.

        Raises
        ------
        RuntimeError
            If called before the environment is fitted.

        """
        return copy_module.deepcopy(self._layout_params_used)

    @property
    @check_fitted
    def layout_type(self) -> str:
        """Return the type of layout used in the environment.

        Returns
        -------
        str
            The layout type (e.g., "RegularGrid", "Hexagonal").

        """
        return (
            self._layout_type_used if self._layout_type_used is not None else "Unknown"
        )

    @property
    @check_fitted
    def n_bins(self) -> int:
        """Return the number of active bins in the environment.

        This is determined by the number of rows in `self.bin_centers`.

        Returns
        -------
        int
            The number of active bins (0 if not fitted).

        Raises
        ------
        RuntimeError
            If called before the environment is fitted.

        """
        return int(self.bin_centers.shape[0])

    @check_fitted
    def get_differential_operator(self) -> sparse.csc_matrix:
        """Compute and cache the inverse-distance edge differential operator.

        The differential operator D is a sparse matrix of shape (n_bins, n_edges)
        that encodes the oriented edge structure of the connectivity graph. It
        provides the foundation for gradient, divergence, and Laplacian operations
        on spatial fields.

        Returns
        -------
        D : scipy.sparse.csc_matrix
            Sparse differential operator matrix of shape (n_bins, n_edges).
            The matrix is cached after first computation for efficiency.

        Raises
        ------
        RuntimeError
            If called before the environment is fitted.
        NotImplementedError
            If the layout has no finite-volume geometry builder.

        Notes
        -----
        Each oriented edge i -> j contributes -1/d at i and +1/d at j,
        using the finite-volume distances (including track junctions).
        ``D.T @ f`` gives directional derivatives in field units per length.
        Divergence uses a separate cached operator ``-M**-1 D diag(A*d)``;
        ``div(grad(f)) = -M**-1 (Deg - W) f`` with ``W = A/d``, the negative
        of the generator used by ``env.smooth``.

        Both matrices use ``@versioned_cached_property`` and are recomputed
        when the environment's state version changes.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> # Create a simple 1D chain
        >>> data = np.array([[0.0], [1.0], [2.0], [3.0]])
        >>> env = Environment.from_samples(data, bin_size=1.0)
        >>> # Access differential operator (computed and cached)
        >>> D = env.get_differential_operator()
        >>> D.shape
        (4, 3)
        >>> # Subsequent access reuses cached matrix
        >>> D2 = env.get_differential_operator()
        >>> np.array_equal(D.toarray(), D2.toarray())
        True

        See Also
        --------
        neurospatial.ops.calculus.compute_differential_operator : Underlying computation

        References
        ----------
        .. [1] PyGSP: Graph Signal Processing in Python
               https://pygsp.readthedocs.io/
        .. [2] Shuman et al. (2013). "The emerging field of signal processing on graphs."
               IEEE Signal Processing Magazine, 30(3), 83-98.
        """
        return self._differential_operator_cached.copy()

    @check_fitted
    def get_divergence_operator(self) -> sparse.csc_matrix:
        """Return the cached finite-volume divergence operator.

        The divergence operator maps a flux on each oriented edge (in
        ``edge_pairs`` order) to the net outward flux per cell volume:
        ``-M**-1 D diag(A * d)``, the negative volume-weighted adjoint of
        :meth:`get_differential_operator`. ``divergence(env, edge_field)``
        applies it.

        Returns
        -------
        scipy.sparse.csc_matrix
            Sparse operator of shape (n_bins, n_edges), cached until the
            environment's geometry changes.

        Raises
        ------
        RuntimeError
            If called before the environment is fitted.
        NotImplementedError
            If the layout has no finite-volume cell geometry.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> env = Environment.from_samples(
        ...     np.array([[0.0], [1.0], [2.0]]), bin_size=1.0
        ... )
        >>> env.get_divergence_operator().shape
        (3, 2)

        See Also
        --------
        get_differential_operator : The edge gradient operator it is adjoint to.
        neurospatial.ops.calculus.divergence : Apply it to an edge field.
        """
        return self._divergence_operator_cached.copy()

    @versioned_cached_property
    def _differential_operator_cached(self) -> sparse.csc_matrix:
        return compute_differential_operator(self)

    @versioned_cached_property
    def _divergence_operator_cached(self) -> sparse.csc_matrix:
        return _compute_divergence_operator(self)

    @check_fitted
    def copy(self, *, deep: bool = True) -> Self:
        """Create a copy of the environment.

        Parameters
        ----------
        deep : bool, default=True
            If True, create a deep copy where modifying the copy will not
            affect the original. Arrays and the connectivity graph are copied.
            If False, create a shallow copy that shares underlying data with
            the original.

        Returns
        -------
        env_copy : Environment
            New environment instance. Transient caches (KDTree, kernels) are
            always cleared regardless of `deep` parameter.

        See Also
        --------
        Environment.subset : Create new environment from bin selection.

        Notes
        -----
        **Deep copy (deep=True, default)**:

        - All numpy arrays are copied (bin_centers, dimension_ranges, etc.)
        - Connectivity graph is deep copied
        - Regions are deep copied
        - Layout object is deep copied

        Adding regions to the copy does not affect the original.

        **Shallow copy (deep=False)**:

        - Arrays, graph and regions are shared with the original

        In both cases the geometry arrays are read-only and the graph is
        frozen, as on every environment.

        **Cache invalidation**:

        Both deep and shallow copies always clear transient caches to ensure
        consistency. Caches are rebuilt on-demand when needed:

        - KDTree cache (used by spatial query methods)
        - Kernel cache (used by smooth() and occupancy())

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> # Create environment
        >>> data = np.array([[i, j] for i in range(10) for j in range(10)])
        >>> env = Environment.from_samples(data, bin_size=1.0)
        >>> env = env.with_metadata(units="cm")
        >>>
        >>> # Deep copy (default): separate, equal geometry
        >>> env_copy = env.copy()
        >>> bool(np.shares_memory(env_copy.bin_centers, env.bin_centers))
        False
        >>>
        >>> # Shallow copy: the same geometry objects
        >>> env_shallow = env.copy(deep=False)
        >>> bool(np.shares_memory(env_shallow.bin_centers, env.bin_centers))
        True
        """
        if deep:
            return copy_module.deepcopy(self)
        return self._replace_value()

    def _clear_explicit_caches(
        self, *, kdtree: bool = True, kernels: bool = True
    ) -> None:
        """Internal method to clear explicit caches (KDTree, kernels).

        This is called by clear_cache().

        Parameters
        ----------
        kdtree : bool, default=True
            Whether to clear the KDTree cache.
        kernels : bool, default=True
            Whether to clear the cached smoothing kernels.
        """
        if kdtree:
            self._kdtree_cache = None
        if kernels:
            self._kernel_cache = {}
            self._gaussian_kernels.clear()

    @check_fitted
    def clear_cache(
        self,
        *,
        kdtree: bool = True,
        kernels: bool = True,
        cached_properties: bool = True,
    ) -> None:
        """Clear caches with optional selective clearing.

        This method clears explicit caches (KDTree, kernels) and/or
        @cached_property values. By default, all caches are cleared.

        Parameters
        ----------
        kdtree : bool, default=True
            Whether to clear the KDTree cache used by map_points_to_bins().
        kernels : bool, default=True
            Whether to clear the cached smoothing kernels: diffusion kernels
            used by compute_kernel(), smooth() and occupancy(), and the
            Gaussian-KDE kernels of rate-map smoothing.
        cached_properties : bool, default=True
            Whether to clear all @cached_property values.

        Caches available for clearing
        ------------------------------
        **Explicit caches** (kdtree, kernels):
            - KDTree cache (_kdtree_cache): Used by map_points_to_bins()
            - Kernel caches (_kernel_cache, _gaussian_kernels): diffusion
              kernels used by smooth() and occupancy(), and Gaussian-KDE
              kernels; each keeps its most recently used kernels

        **Cached properties** (cached_properties):
            - differential_operator: Graph differential operator matrix
            - boundary_bins: Bins on environment boundary
            - bin_sizes: Area/volume of each bin
            - bin_attributes: DataFrame with bin properties (position, degree, etc.)
            - edge_attributes: DataFrame with edge properties (distance, angle, etc.)
            - linearization_properties: 1D linearization metadata
            - _source_flat_to_active_node_id_map: Grid index mapping

        Notes
        -----
        This method is useful when:

        - **Memory management**: Free memory after intensive computations
        - **Testing**: Ensure clean state between test runs
        - **Profiling**: Measure cache rebuild performance
        - **Selective clearing**: Clear only specific cache types
        - **Manual modifications**: Clear stale caches if environment was
          modified (not recommended - create new Environment instead)

        Cached properties are automatically recomputed on next access.

        **Performance impact**: Clearing caches has minimal overhead, but
        recomputing cached properties (especially differential_operator) can
        be expensive for large environments (>100,000 bins).

        See Also
        --------
        copy : Create environment copy (also clears caches)

        Examples
        --------
        Clear all caches (default behavior):

        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> data = np.random.rand(1000, 2) * 100
        >>> env = Environment.from_samples(data, bin_size=5.0)
        >>>
        >>> # Access some cached properties
        >>> _ = env.get_differential_operator()
        >>> _ = env.boundary_bins
        >>>
        >>> # Clear all caches
        >>> env.clear_cache()
        >>>
        >>> # Properties will be recomputed on next access
        >>> _ = env.get_differential_operator()  # Triggers recomputation

        Clear only specific cache types:

        >>> # Clear only KDTree cache (keep kernels and cached properties)
        >>> env.clear_cache(kdtree=True, kernels=False, cached_properties=False)
        >>>
        >>> # Clear only cached properties (keep KDTree and kernels)
        >>> env.clear_cache(kdtree=False, kernels=False, cached_properties=True)
        >>>
        >>> # Clear KDTree and kernels, keep cached properties
        >>> env.clear_cache(kdtree=True, kernels=True, cached_properties=False)
        """
        # Clear explicit caches (KDTree, kernels) selectively
        self._clear_explicit_caches(kdtree=kdtree, kernels=kernels)

        # Clear @cached_property AND @versioned_cached_property values from
        # instance __dict__ if requested. This avoids maintaining a hardcoded
        # list that can become stale when adding new cached properties.
        if cached_properties:
            for cls in type(self).__mro__:
                for name, attr in vars(cls).items():
                    if isinstance(attr, cached_property):
                        self.__dict__.pop(name, None)
                    elif isinstance(attr, versioned_cached_property):
                        # versioned_cached_property uses two keys: the
                        # value at "_versioned_cache__<name>" and the
                        # captured version at
                        # "_versioned_cache__<name>__version".
                        value_key = f"_versioned_cache__{name}"
                        self.__dict__.pop(value_key, None)
                        self.__dict__.pop(f"{value_key}__version", None)


class Environment(_BaseEnvironment, Generic[_C]):
    """A discretized environment with owned coordinate meaning and connectivity.

    ``Environment`` is the central user-facing spatial type. Its
    ``bin_centers`` hold Cartesian ``(x, y[, z])`` coordinates. Geometric
    methods in ``CartesianCoordinates`` (``bin_at``, ``contains``,
    ``distance_between``, ``apply_transform``) interpret inputs as Cartesian
    points. The shared ``distance_to`` method guards its Euclidean mode.

    Instances are created via the factory classmethods inherited from
    :class:`~neurospatial.environment.factories.EnvironmentFactories`
    (``from_samples``, ``from_polygon``, ``from_grid_mask``,
    ``from_pixel_mask``, ``from_graph``, ``from_layout``).

    Cartesian factories store position coordinates in length units. The
    from_polar_egocentric factory returns the same runtime class with radius in
    length units and angle in radians. Cartesian-only methods refuse polar
    instances before interpreting those coordinates.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> positions = np.array([[0.0, 0.0], [0.0, 2.0], [2.0, 0.0], [2.0, 2.0]])
    >>> env = Environment.from_samples(positions, bin_size=1.0, units="cm")
    >>> env.n_dims
    2
    >>> env.bin_at(positions).shape
    (4,)
    """

    @property
    @check_fitted
    def coordinate_kind(self) -> Literal["cartesian", "polar"]:
        """Coordinate meaning: Cartesian positions or egocentric radius/angle.

        Polar radius uses declared length units; angle is always in radians.
        This read-only property replaces public-class checks for dispatch.
        """
        return "polar" if self._is_polar else "cartesian"

    @property
    def _coordinate_system(self) -> _C:
        """Owned coordinate semantics, inferred from the construction factory."""
        return cast("_C", self._geometry.coordinates)

    def _require_cartesian(self, operation: str) -> CartesianEnvironmentProtocol:
        """Refuse Cartesian coordinate math for polar radius/angle bins."""
        if self._is_polar:
            from neurospatial._exceptions import CartesianOnlyError
            from neurospatial.environment.polar import _POLAR_EXPLANATION

            raise CartesianOnlyError(
                type(self).__name__,
                operation,
                reason="Cartesian-coordinate operation",
                explanation=_POLAR_EXPLANATION,
            )
        return cast("CartesianEnvironmentProtocol", self)

    def __getattribute__(self, name: str) -> Any:
        if name in CARTESIAN_OPERATIONS:
            geometry = object.__getattribute__(self, "__dict__").get("_geometry")
            if geometry is not None and isinstance(
                geometry.coordinates, EgocentricPolarSystem
            ):
                from neurospatial._exceptions import CartesianOnlyError
                from neurospatial.environment.polar import _POLAR_EXPLANATION

                raise CartesianOnlyError(
                    type(self).__name__,
                    f"{name}()",
                    reason="Cartesian-grid operation"
                    if name in CARTESIAN_GRID_OPERATIONS
                    else "Cartesian-coordinate operation",
                    explanation=_POLAR_EXPLANATION,
                )
        return super().__getattribute__(name)

    @property
    def _angular_is_circular(self) -> bool:
        """Normalized polar circularity, retained through copies and storage."""
        coordinates = self._geometry.coordinates
        return isinstance(coordinates, EgocentricPolarSystem) and coordinates.circular

    @versioned_cached_property
    @check_fitted
    def _boundary_bins_cached(self) -> NDArray[np.intp]:
        if not self._is_polar:
            return _cartesian_boundary_bins(self)
        from neurospatial.layout.helpers.utils import find_boundary_nodes

        return find_boundary_nodes(
            graph=self.connectivity,
            grid_shape=self.grid_shape,
            active_mask=self.active_mask,
            layout_kind=self._layout_type_used,
            periodic_axes=(1,) if self._angular_is_circular else (),
        ).astype(np.intp, copy=False)

    @classmethod
    def from_polar_egocentric(
        cls,
        distance_range: tuple[float, float],
        angle_range: tuple[float, float],
        distance_bin_size: float,
        angle_bin_size: float,
        *,
        circular_angle: bool = True,
        connect_diagonal_neighbors: bool = True,
        name: str = "",
        units: str | None = None,
        frame: str | None = None,
    ) -> Environment[EgocentricPolarSystem]:
        """Create an egocentric polar coordinate environment.

        Returns the same Environment runtime class as Cartesian factories.
        Radius uses declared length units; angle remains in radians. Cartesian
        operations refuse these coordinates; graph-based operations remain usable.

        Parameters
        ----------
        distance_range : tuple of (float, float)
            The (min, max) range of distances in physical units (e.g., cm).
            Must have min < max.
        angle_range : tuple of (float, float)
            The (min, max) range of angles in radians. For full circle
            coverage use (-π, π) or (0, 2π). Must have min < max.
        distance_bin_size : float
            Size of each distance bin, same units as ``distance_range``.
            Must be positive.
        angle_bin_size : float
            Size of each angle bin in radians. Must be positive.
        circular_angle : bool, default=True
            If True, the angle dimension wraps circularly, connecting the
            first and last angle bins at each distance ring. Appropriate
            when ``angle_range`` spans a full circle.
        connect_diagonal_neighbors : bool, default=True
            Whether to connect diagonally adjacent (distance, angle) bins.
        name : str, default=""
            Optional name for the environment.

        Returns
        -------
        Environment
            A fitted polar environment. Column 0 is distance in physical units;
            column 1 is angle in radians (zero ahead, positive to the left).

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> env = Environment.from_polar_egocentric(
        ...     distance_range=(0.0, 100.0),  # 0-100 cm
        ...     angle_range=(-np.pi, np.pi),  # Full circle
        ...     distance_bin_size=10.0,  # 10 cm distance bins
        ...     angle_bin_size=np.pi / 4,  # 45 degree angle bins
        ... )
        >>> env.n_bins  # 10 distance * 8 angle = 80 bins
        80
        >>> type(env).__name__
        'Environment'
        >>> isinstance(env, Environment)
        True

        See Also
        --------
        Environment.from_samples : Build Cartesian position bins.
        """

        from neurospatial.environment.polar import _create_polar_environment

        return _create_polar_environment(
            distance_range=distance_range,
            angle_range=angle_range,
            distance_bin_size=distance_bin_size,
            angle_bin_size=angle_bin_size,
            circular_angle=circular_angle,
            connect_diagonal_neighbors=connect_diagonal_neighbors,
            name=name,
            units=units,
            frame=frame,
        )

    @property
    @check_fitted
    def component_keys(self: Environment[CartesianSystem]) -> tuple[str, ...]:
        """Stable combined component keys, or an empty tuple for a native environment."""
        self._require_cartesian("component_keys()")
        from neurospatial.layout._composition import composition_view

        result = composition_view(self._binning.locator)
        return () if result is None else result[0].keys

    @property
    @check_fitted
    def bin_sources(
        self: Environment[CartesianSystem],
    ) -> tuple[tuple[str, int], ...] | None:
        """Original component key and local bin ID for each bin, or None when native.

        Subsets and clips retain original source IDs; split cells may repeat an ID.
        """
        self._require_cartesian("bin_sources()")
        from neurospatial.layout._composition import composition_view

        result = composition_view(self._binning.locator)
        if result is None:
            return None
        root, indices, _scale = result
        offsets = root.offsets
        owners = np.searchsorted(offsets[1:], indices, side="right")
        return tuple(
            (root.keys[owner], int(index - offsets[owner]))
            for index, owner in zip(indices, owners, strict=True)
        )

    @check_fitted
    def get_bridge_attributes(self: Environment[CartesianSystem]) -> pd.DataFrame:
        """Return an independent table of retained automatic navigation bridges.

        Rows identify component/local endpoints, current bin IDs, path length
        and geometric conductance. Native environments return an empty table.
        Conductance units are length ** (measure_dim - 2).
        """
        self._require_cartesian("get_bridge_attributes()")
        import pandas as pd

        from neurospatial.layout._composition import composition_view

        columns = [
            "source_component",
            "source_local_bin",
            "target_component",
            "target_local_bin",
            "source",
            "target",
            "distance",
            "conductance",
        ]
        result = composition_view(self._binning.locator)
        if result is None:
            return pd.DataFrame(columns=columns)
        root, indices, _scale = result
        from neurospatial.layout._composition import bridge_values

        by_pair = bridge_values(self._binning)
        offsets = root.offsets
        records = []
        for (source, target), distance in zip(
            self._binning.bins.pairs, self._binning.bins.pair_distances, strict=True
        ):
            pair = frozenset((int(source), int(target)))
            if pair not in by_pair:
                continue
            owners = np.searchsorted(
                offsets[1:], [indices[source], indices[target]], side="right"
            )
            records.append(
                (
                    root.keys[owners[0]],
                    int(indices[source] - offsets[owners[0]]),
                    root.keys[owners[1]],
                    int(indices[target] - offsets[owners[1]]),
                    int(source),
                    int(target),
                    float(distance),
                    float(by_pair[pair]),
                )
            )
        return pd.DataFrame(records, columns=columns)

    @check_fitted
    def bin_at(
        self: Environment[CartesianSystem], points_nd: NDArray[np.float64]
    ) -> NDArray[np.intp]:
        """Map N-dimensional points to bins using geometric containment.

        This method uses the layout-specific geometric logic to determine which
        bin contains each point. This is different from nearest-neighbor mapping
        and respects the actual geometry of the bins (grid cells, hexagons, etc.).

        For nearest-neighbor mapping with caching and tie-breaking, see
        `neurospatial.ops.map_points_to_bins()`.

        Parameters
        ----------
        points_nd : NDArray[np.float64], shape (n_points, n_dims)
            An array of N-dimensional points to map.

        Returns
        -------
        NDArray[np.intp], shape (n_points,)
            An array of active bin indices (0 to `n_active_bins - 1`).
            A value of -1 indicates that the corresponding point did not map
            to any active bin (e.g., it's outside the environment).

        Raises
        ------
        RuntimeError
            If called before the environment is fitted.
        ValueError
            If ``points_nd`` has a different dimensionality than the
            environment (e.g. 3-D points for a 2-D environment).

        See Also
        --------
        neurospatial.ops.map_points_to_bins : Nearest-neighbor mapping with KDTree caching

        Notes
        -----
        **Geometric Containment vs Nearest-Neighbor:**

        - `bin_at()` (this method): Uses layout-specific geometric containment
          to determine which bin actually contains the point. For grids, this
          checks which grid cell the point falls into. For hexagons, which
          hexagon contains it. This is exact but may be slower for large batches.

        - `map_points_to_bins()`: Uses KDTree to find the bin whose center is
          closest to the point. This is fast (O(log N)) with caching, supports
          tie-breaking on boundaries, and allows distance thresholds. Best for
          large batches, trajectory processing, and when approximate assignment
          is acceptable.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> x, y = np.meshgrid([0.0, 2.0, 4.0], [0.0, 2.0, 4.0])
        >>> env = Environment.from_samples(np.c_[x.ravel(), y.ravel()], bin_size=2.0)
        >>> points = np.array([[2.0, 2.0], [15.0, 15.0]])
        >>> indices = env.bin_at(points)
        >>> indices.tolist()  # -1 denotes the point outside the environment
        [4, -1]

        """
        # Normalize to the portable ``np.intp`` index dtype at the Environment
        # boundary, regardless of what an arbitrary ``LayoutEngine`` returns.
        self._require_cartesian("bin_at()")
        return self._locate(points_nd)

    @check_fitted
    def contains(
        self: Environment[CartesianSystem], points_nd: NDArray[np.float64]
    ) -> NDArray[np.bool_]:
        """Check if N-dimensional continuous points fall within any active bin.

        Parameters
        ----------
        points_nd : NDArray[np.float64], shape (n_points, n_dims)
            An array of N-dimensional points to check.

        Returns
        -------
        NDArray[np.bool_], shape (n_points,)
            A boolean array where `True` indicates the corresponding point
            maps to an active bin, and `False` indicates it does not.

        Raises
        ------
        RuntimeError
            If called before the environment is fitted.
        ValueError
            If ``points_nd`` has a different dimensionality than the
            environment (e.g. 3-D points for a 2-D environment).

        Notes
        -----
        This method is optimized to avoid redundant KDTree queries by reusing
        the bin index computation from `bin_at()` and checking for the -1 sentinel.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> data = np.random.rand(100, 2) * 10
        >>> env = Environment.from_samples(data, bin_size=2.0)
        >>> points = np.array([[5.0, 5.0], [15.0, 15.0]])
        >>> mask = env.contains(points)
        >>> print(mask)  # doctest: +SKIP
        [ True False]  # First point in environment, second outside

        """
        # Optimized: compute indices once and check for -1 sentinel
        # This avoids redundant KDTree queries compared to calling bin_at() separately
        self._require_cartesian("contains()")
        indices = self._locate(points_nd)
        return np.asarray(indices != -1, dtype=np.bool_)

    @check_fitted
    def distance_between(
        self: Environment[CartesianSystem],
        point1: NDArray[np.float64],
        point2: NDArray[np.float64],
        edge_weight: str = "distance",
    ) -> float:
        """Calculate the geodesic distance between two points in the environment.

        Points are first mapped to their nearest active bins using `self.bin_at()`.
        The geodesic distance (distance along the shortest path through the space)
        is then the shortest path length in the `connectivity` graph between these
        bins, using the specified `edge_weight`.

        Parameters
        ----------
        point1 : NDArray[np.float64], shape (n_dims,) or (1, n_dims)
            The first N-dimensional point.
        point2 : NDArray[np.float64], shape (n_dims,) or (1, n_dims)
            The second N-dimensional point.
        edge_weight : str, optional
            The edge attribute to use as weight for path calculation,
            by default "distance". If None, the graph is treated as unweighted.

        Returns
        -------
        float
            The geodesic distance. Returns `np.inf` if points do not map to
            valid active bins, if bins are disconnected, or if the connectivity
            graph is not available.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> data = np.random.rand(100, 2) * 10
        >>> env = Environment.from_samples(data, bin_size=2.0)
        >>> p1 = np.array([1.0, 1.0])
        >>> p2 = np.array([9.0, 9.0])
        >>> dist = env.distance_between(p1, p2)
        >>> print(dist > 0)
        True

        """
        self._require_cartesian("distance_between()")
        source_bin = self.bin_at(np.atleast_2d(point1))[0]
        target_bin = self.bin_at(np.atleast_2d(point2))[0]

        if source_bin == -1 or target_bin == -1:
            # One or both points didn't map to a valid active bin. Returning
            # np.inf silently can mask a coordinate-frame or units mistake, so
            # warn when neither point lands in the environment.
            if source_bin == -1 and target_bin == -1:
                warnings.warn(
                    "distance_between() returning np.inf: neither point mapped "
                    "to an active bin (both bin_at() lookups returned -1). "
                    "Check that the points are inside the environment and in "
                    "the same coordinate frame/units as bin_centers.",
                    category=UserWarning,
                    stacklevel=2,
                )
            return np.inf

        try:
            return float(
                nx.shortest_path_length(
                    self.connectivity,
                    source=source_bin,
                    target=target_bin,
                    weight=edge_weight,
                )
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return np.inf

    @check_fitted
    def to_linear(
        self: Environment[CartesianSystem],
        nd_position: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Convert N-D positions to 1D linear coordinates.

        For 1D linearized environments (created with `from_graph()`), this
        converts N-D positions (e.g., 2D x,y coordinates) to their corresponding
        1D linear coordinates along the linearized track.

        Parameters
        ----------
        nd_position : NDArray[np.floating], shape (n_samples, n_dims)
            N-D position array in the original coordinate space.
            For 2D environments, shape is (n_samples, 2).

        Returns
        -------
        NDArray[np.floating], shape (n_samples,)
            1D linear coordinates corresponding to the input positions.

        Raises
        ------
        AttributeError
            If this environment is not 1D (does not support linearization).
            Check `env.is_linearized_track` before calling this method.

        Notes
        -----
        - **Only works for 1D environments** created with `Environment.from_graph()`
        - For N-D environments, use `bin_at()` to map positions to bins instead
        - Linear coordinates start at 0 and increase monotonically along the track

        See Also
        --------
        linear_to_nd : Convert 1D linear coordinates back to N-D positions
        Environment.from_graph : Create 1D linearized environment
        is_linearized_track : Check if environment supports linearization

        Examples
        --------
        >>> import numpy as np
        >>> import networkx as nx
        >>> from neurospatial import Environment
        >>>
        >>> # Create a simple 1D track
        >>> G = nx.Graph()  # doctest: +SKIP
        >>> G.add_node(0, pos=(0.0, 0.0))  # doctest: +SKIP
        >>> G.add_node(1, pos=(10.0, 0.0))  # doctest: +SKIP
        >>> G.add_edge(0, 1, edge_id=0, distance=10.0)  # doctest: +SKIP
        >>> env = Environment.from_graph(
        ...     G, edge_order=[(0, 1)], edge_spacing=0.0, bin_size=2.0
        ... )  # doctest: +SKIP
        >>>
        >>> # Convert 2D position to 1D linear coordinate
        >>> position_2d = np.array(
        ...     [[5.0, 0.0]]
        ... )  # Halfway along track  # doctest: +SKIP
        >>> position_1d = env.to_linear(position_2d)  # doctest: +SKIP
        >>> print(f"Linear position: {position_1d[0]:.1f}")  # doctest: +SKIP
        Linear position: 5.0

        """
        self._require_cartesian("to_linear()")
        if not self._is_linearized_track_env:
            msg = (
                "to_linear() is only available for 1D environments (GraphLayout). "
                f"This environment is {self.n_dims}D. "
                "Use bin_at() to map positions to bins for N-D environments."
            )
            raise AttributeError(msg)

        linearization = self._binning.linearization
        if linearization is None:
            raise AttributeError(
                _format_error(
                    "to_linear() needs the track graph that defines the linearization, and "
                    f"this environment's {self._binning.kind} layout does not keep it.",
                    why="Why: track environments read from NWB restore bins and connectivity, but not the track graph.",
                    fix="rebuild the environment with Environment.from_graph or Environment.linear_track from the same track",
                )
            )
        return linearization.to_linear(nd_position)

    @check_fitted
    def linear_to_nd(
        self: Environment[CartesianSystem],
        linear_position: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Convert 1D linear coordinates to N-D positions.

        For 1D linearized environments (created with `from_graph()`), this
        converts 1D linear coordinates back to the original N-D coordinate
        space (e.g., 2D x,y coordinates).

        Parameters
        ----------
        linear_position : NDArray[np.floating], shape (n_samples,)
            1D linear coordinate array.

        Returns
        -------
        NDArray[np.floating], shape (n_samples, n_dims)
            N-D positions corresponding to the input linear coordinates.
            For 2D environments, shape is (n_samples, 2).

        Raises
        ------
        AttributeError
            If this environment is not 1D (does not support linearization).
            Check `env.is_linearized_track` before calling this method.

        Notes
        -----
        - **Only works for 1D environments** created with `Environment.from_graph()`
        - Inverse operation of `to_linear()`
        - Useful for plotting 1D data in the original 2D space

        See Also
        --------
        to_linear : Convert N-D positions to 1D linear coordinates
        Environment.from_graph : Create 1D linearized environment
        is_linearized_track : Check if environment supports linearization

        Examples
        --------
        >>> import numpy as np
        >>> import networkx as nx
        >>> from neurospatial import Environment
        >>>
        >>> # Create a simple 1D track
        >>> G = nx.Graph()  # doctest: +SKIP
        >>> G.add_node(0, pos=(0.0, 0.0))  # doctest: +SKIP
        >>> G.add_node(1, pos=(10.0, 0.0))  # doctest: +SKIP
        >>> G.add_edge(0, 1, edge_id=0, distance=10.0)  # doctest: +SKIP
        >>> env = Environment.from_graph(
        ...     G, edge_order=[(0, 1)], edge_spacing=0.0, bin_size=2.0
        ... )  # doctest: +SKIP
        >>>
        >>> # Convert 1D linear coordinate to 2D position
        >>> linear_pos = np.array([5.0])  # Halfway along track  # doctest: +SKIP
        >>> position_2d = env.linear_to_nd(linear_pos)  # doctest: +SKIP
        >>> print(f"2D position: {position_2d[0]}")  # doctest: +SKIP
        2D position: [5. 0.]

        """
        self._require_cartesian("linear_to_nd()")
        if not self._is_linearized_track_env:
            msg = (
                "linear_to_nd() is only available for 1D environments (GraphLayout). "
                f"This environment is {self.n_dims}D. "
                "Use bin_center_of() to get bin centers for N-D environments."
            )
            raise AttributeError(msg)

        linearization = self._binning.linearization
        if linearization is None:
            raise AttributeError(
                _format_error(
                    "linear_to_nd() needs the track graph that defines the linearization, and "
                    f"this environment's {self._binning.kind} layout does not keep it.",
                    why="Why: track environments read from NWB restore bins and connectivity, but not the track graph.",
                    fix="rebuild the environment with Environment.from_graph or Environment.linear_track from the same track",
                )
            )
        return linearization.linear_to_nd(linear_position)

    @check_fitted
    def occupancy(
        self: Environment[CartesianSystem],
        times: NDArray[np.float64],
        positions: NDArray[np.float64],
        *,
        speed: NDArray[np.float64] | None = None,
        min_speed: float | None = None,
        max_gap: float | None = 0.5,
        epochs: Any = None,
        bandwidth: float | None = None,
        time_allocation: Literal["start", "linear"] = "start",
        return_seconds: bool = True,
    ) -> NDArray[np.float64]:
        """Compute occupancy (time spent in each bin).

        Accumulates time spent in each bin from continuous trajectory samples.
        Supports optional speed filtering, gap handling, and kernel smoothing.

        Parameters
        ----------
        times : NDArray[np.float64], shape (n_samples,)
            Timestamps in seconds. Must be monotonically increasing.
        positions : NDArray[np.float64], shape (n_samples, n_dims)
            Position coordinates matching environment dimensions.
        speed : NDArray[np.float64], shape (n_samples,), optional
            Instantaneous speed at each sample. If provided with min_speed,
            samples below threshold are excluded from occupancy calculation.
        min_speed : float, optional
            Minimum speed threshold in physical units per second. Requires
            speed parameter. Samples with speed < min_speed are excluded.
        max_gap : float or None, default=0.5
            Longest sampling interval (seconds) treated as continuous recording.
            Longer intervals (dropped frames, pauses between sessions) are excluded
            from occupancy and their spikes are not counted. ``None`` disables the
            gap check.
        epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
            Restrict the analysis to these half-open [start, stop) windows (seconds,
            same clock as ``times``). An interval counts only if it lies entirely
            inside one window. ``None`` (default) means unrestricted.
        bandwidth : float, optional
            If provided, apply diffusion kernel smoothing with this bandwidth
            (in physical units). Uses mode='transition' to preserve total mass.
            Smoothing preserves total occupancy time.
        time_allocation : {'start', 'linear'}, default='start'
            Method for allocating time intervals across bins:

            - 'start': Assign entire Δt to starting bin (fast, works on all layouts).
            - 'linear': Split Δt proportionally across bins traversed by
              straight-line path (grid layouts with rectangular cells).
        return_seconds : bool, default=True
            If True, return time in seconds (time-weighted occupancy).
            If False, return sample counts (unweighted, counts number of
            intervals starting in each bin). Use True for rate calculations
            (e.g., firing rates), False for presence/absence analyses.

        Returns
        -------
        occupancy : NDArray[np.float64], shape (n_bins,)
            If return_seconds=True: Time in seconds spent in each bin.
            If return_seconds=False: Count of intervals starting in each bin.
            The sum equals total valid time (when return_seconds=True) or
            total number of intervals (when return_seconds=False), excluding
            filtered periods and large gaps.

        Raises
        ------
        RuntimeError
            If called before the environment is fitted.
        ValueError
            If times and positions have different lengths, if arrays are
            inconsistent, or if min_speed is provided without speed.
        ValueError
            If positions have wrong number of dimensions.
        ValueError
            If time_allocation is not 'start' or 'linear'.
        ValueError
            If ``times`` contains non-finite values (NaN or Inf). The error
            names ``times`` and the first offending index.
        NotImplementedError
            If linear allocation lacks exact supported cell geometry.

        See Also
        --------
        compute_kernel : Compute diffusion kernel for smoothing.
        bin_at : Map single N-dimensional point to bin index.

        Notes
        -----
        **Time allocation methods**:

        - time_allocation='start' (default): Each time interval Δt is assigned
          entirely to the bin at the starting position. Fast and works on all
          layout types, but may underestimate occupancy in bins the animal
          passed through.

        - time_allocation='linear': Splits Δt proportionally across all bins
          traversed by the straight-line path between consecutive samples.
          Supported for rectangular grids and exact planar polygon cells.
          Polygon allocation retains entry and re-entry intervals, excludes
          holes and outside portions without renormalizing their duration,
          and refuses proper crossings of a known physical wall. Observation
          validity is independent of whether the starting point is in a bin.

        **Mass conservation**: The sum of the returned occupancy array equals
        the total valid time:

        .. math::
            \\sum_i \\text{occupancy}[i] = \\sum_{\\text{valid } k} (t_{k+1} - t_k)

        where valid intervals satisfy:
        - Δt ≤ max_gap (if max_gap is not None)
        - speed[k] ≥ min_speed (if min_speed is not None)
        - positions[k] is inside environment

        Linear allocation works on grids built with ``from_samples``,
        ``from_grid_mask``, ``from_pixel_mask``, ``from_polygon``, and grids
        read from NWB. Inactive cells receive no time.

        For ``time_allocation='linear'`` on a masked grid (one with inactive
        bins), any time the straight-line path spends crossing **inactive**
        bins is dropped rather than reassigned, since the animal cannot occupy
        a bin that does not exist. As a result the linear total may be slightly
        below the ``time_allocation='start'`` total by exactly that
        inactive-bin crossing time.

        **Kernel smoothing**: When bandwidth is provided, smoothing
        is applied after accumulation using mode='transition' normalization
        (kernel columns sum to 1), which preserves the total occupancy mass.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> x, y = np.meshgrid(np.arange(0.0, 22.0, 2.0), np.arange(0.0, 22.0, 2.0))
        >>> env = Environment.from_samples(np.c_[x.ravel(), y.ravel()], bin_size=2.0)
        >>> times = np.array([0.0, 1.0, 2.0, 3.0])
        >>> positions = np.array([[5.0, 5.0], [5.0, 5.0], [10.0, 10.0], [10.0, 10.0]])
        >>> # These 1 Hz demonstration samples are intentionally continuous.
        >>> occ = env.occupancy(times, positions, max_gap=None)
        >>> float(occ.sum())
        3.0
        >>> speeds = np.array([5.0, 5.0, 0.5, 5.0])
        >>> occ_filtered = env.occupancy(
        ...     times,
        ...     positions,
        ...     speed=speeds,
        ...     min_speed=2.0,
        ...     bandwidth=3.0,
        ...     max_gap=None,
        ... )

        """
        self._require_cartesian("occupancy()")
        return _occupancy(
            self,
            times,
            positions,
            speed=speed,
            min_speed=min_speed,
            max_gap=max_gap,
            epochs=epochs,
            bandwidth=bandwidth,
            time_allocation=time_allocation,
            return_seconds=return_seconds,
        )

    def bin_sequence(
        self: Environment[CartesianSystem],
        times: NDArray[np.float64],
        positions: NDArray[np.float64],
        *,
        dedup: bool = True,
        outside_value: int | None = -1,
        max_gap: float | None = 0.5,
        epochs: Any = None,
    ) -> NDArray[np.int32]:
        """Map trajectory to sequence of bin indices.

        Converts a continuous trajectory (times and positions) into a discrete
        sequence of bin indices, with optional deduplication of consecutive
        repeats. To also obtain per-run start indices and lengths, call
        :meth:`bin_sequence_with_runs` instead.

        Parameters
        ----------
        times : NDArray[np.float64], shape (n_samples,)
            Timestamps in seconds. Should be monotonically increasing.
        positions : NDArray[np.float64], shape (n_samples, n_dims)
            Position coordinates matching environment dimensions.
        dedup : bool, default=True
            If True, collapse consecutive repeats: [A,A,A,B] → [A,B].
            If False, return a bin index for every retained sample.
        outside_value : int or None, default=-1
            Bin index for samples outside environment bounds.
            - If -1 (default), outside samples are marked with -1.
            - If None, outside samples are dropped from the sequence entirely.
        max_gap : float or None, default=0.5
            Longest sampling interval (seconds) treated as continuous recording.
            Longer intervals (dropped frames, pauses between sessions) are excluded
            from the analysis. ``None`` disables the gap check.
        epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
            Restrict the analysis to these half-open [start, stop) windows (seconds,
            same clock as ``times``). An interval counts only if it lies entirely
            inside one window. ``None`` (default) means unrestricted.

        Returns
        -------
        bins : NDArray[np.int32], shape (n_sequences,)
            Bin index at each retained time point (or deduplicated sequence).
            Values are in range [0, n_bins-1] for valid bins, or -1 for
            outside samples (when outside_value=-1).

        Raises
        ------
        ValueError
            If times and positions have different lengths, if positions
            have wrong number of dimensions, or if timestamps are not
            monotonically increasing (non-decreasing).

        See Also
        --------
        bin_sequence_with_runs : Same mapping plus run-length boundaries.
        occupancy : Compute time spent in each bin.
        transitions : Build empirical transition matrix from trajectory.

        Notes
        -----
        Each run of samples with gaps no longer than ``max_gap`` (inside ``epochs``)
        is analyzed as a separate recording; no segment spans a pause.

        Samples that touch no valid interval are dropped, including singleton
        inputs. Deduplication never merges samples across an invalid interval.
        With dedup=False, one bin is returned per retained sample.

        Timestamps must be monotonically increasing (non-decreasing).
        Sort your data by time before calling this method if needed.

        Examples
        --------
        >>> bins = env.bin_sequence(times, positions)  # doctest: +SKIP
        >>> bins = env.bin_sequence(times, positions, dedup=False)  # doctest: +SKIP
        >>> bins = env.bin_sequence(
        ...     times, positions, outside_value=None
        ... )  # doctest: +SKIP

        """
        self._require_cartesian("bin_sequence()")
        result = _bin_sequence(
            self,
            times,
            positions,
            dedup=dedup,
            outside_value=outside_value,
            gap_splits_runs=False,
            max_gap=max_gap,
            epochs=epochs,
        )
        return result.bins

    def bin_sequence_with_runs(
        self: Environment[CartesianSystem],
        times: NDArray[np.float64],
        positions: NDArray[np.float64],
        *,
        outside_value: int | None = -1,
        max_gap: float | None = 0.5,
        epochs: Any = None,
    ) -> BinSequenceWithRuns:
        """Map trajectory to bin sequence plus per-run boundaries.

        Returns the run-length encoding of the trajectory so callers
        can recover per-run durations from the original ``times`` array.
        ``bins``, ``run_starts``, and ``run_lengths`` all share shape
        ``(n_runs,)``: the per-run view is intrinsically deduplicated.
        For per-sample bins use :meth:`bin_sequence` with ``dedup=False``.

        Parameters
        ----------
        times : NDArray[np.float64], shape (n_samples,)
            Timestamps in seconds. Should be monotonically increasing.
        positions : NDArray[np.float64], shape (n_samples, n_dims)
            Position coordinates matching environment dimensions.
        outside_value : int or None, default=-1
            Bin index for samples outside environment bounds (see
            :meth:`bin_sequence`). With ``outside_value=None``, outside
            gaps split runs even when consecutive in-env samples land
            in the same bin (so ``run_lengths.sum()`` equals the
            post-filter count of in-env samples).
        max_gap : float or None, default=0.5
            Longest sampling interval (seconds) treated as continuous recording.
            Longer intervals (dropped frames, pauses between sessions) are excluded
            from the analysis. ``None`` disables the gap check.
        epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
            Restrict the analysis to these half-open [start, stop) windows (seconds,
            same clock as ``times``). An interval counts only if it lies entirely
            inside one window. ``None`` (default) means unrestricted.

        Returns
        -------
        BinSequenceWithRuns
            Frozen dataclass with ``bins``, ``run_starts``, and
            ``run_lengths`` fields, all shape ``(n_runs,)``. See
            :class:`BinSequenceWithRuns` for the precise contract.

        See Also
        --------
        bin_sequence : Returns just the bin sequence (supports ``dedup``).

        Notes
        -----
        Each run of samples with gaps no longer than ``max_gap`` (inside ``epochs``)
        is analyzed as a separate recording; no segment spans a pause.

        Samples that touch no valid interval are dropped, including singleton
        inputs. Invalid intervals split runs even if the bins match. Run
        starts and lengths refer to the original input indices, so durations
        computed from them exclude pauses.

        Examples
        --------
        >>> result = env.bin_sequence_with_runs(times, positions)  # doctest: +SKIP
        >>> # Duration of run i in seconds:
        >>> # times[result.run_starts[i] + result.run_lengths[i] - 1]
        >>> #   - times[result.run_starts[i]]
        """
        self._require_cartesian("bin_sequence_with_runs()")
        return _bin_sequence(
            self,
            times,
            positions,
            dedup=True,
            outside_value=outside_value,
            gap_splits_runs=True,
            max_gap=max_gap,
            epochs=epochs,
        )

    def interpolate(
        self: Environment[CartesianSystem],
        field: NDArray[np.float64],
        points: NDArray[np.float64],
        *,
        mode: Literal["nearest", "linear"] = "nearest",
    ) -> NDArray[np.float64]:
        """Interpolate field values at arbitrary points.

        Evaluates bin-valued fields at continuous query points using either
        nearest-neighbor or linear interpolation. Nearest mode works on all
        layout types; linear mode requires native or retained source grid axes.

        Parameters
        ----------
        field : NDArray[np.float64], shape (n_bins,)
            Field values per bin. Must be a 1-D array with length equal to n_bins.
            Must not contain NaN or Inf values.
        points : NDArray[np.float64], shape (n_points, n_dims)
            Query points in environment coordinates. Must be a 2-D array where
            each row is a point with dimensionality matching the environment.
        mode : {'nearest', 'linear'}, default='nearest'
            Interpolation mode:

            - 'nearest': Use value of nearest bin center (all layouts).
              Points outside environment bounds return NaN.
            - 'linear': Bilinear (2D) or trilinear (3D) interpolation for
              grids with rectangular cells: ``from_samples``, ``from_grid_mask``,
              ``from_pixel_mask``, ``from_polygon``, and grids read from NWB.
              Rotated, reflected and sheared 2-D/3-D grids use their source axes;
              queries remain in world coordinates.
              Inactive cells return NaN.
              Points outside grid bounds return NaN.

        Returns
        -------
        values : NDArray[np.float64], shape (n_points,)
            Interpolated field values. Points outside environment → NaN.

        Raises
        ------
        RuntimeError
            If called before the environment is fitted.
        ValueError
            If field has wrong shape, wrong dimensionality, contains NaN/Inf,
            points have wrong dimensionality, mode is invalid, or dimensions
            don't match, or affine projection cannot resolve source cell widths.
        NotImplementedError
            If mode='linear' is requested for non-grid layout.

        See Also
        --------
        smooth : Apply diffusion kernel smoothing to fields.
        occupancy : Compute occupancy with optional smoothing.

        Notes
        -----
        **Nearest-neighbor mode**: Uses KDTree to find closest bin center.
        Deterministic and works on all layout types. Points farther than a
        reasonable threshold from any bin center are marked as outside (NaN).

        **Linear mode**: Uses scipy.interpolate.RegularGridInterpolator for
        smooth interpolation on rectangular grids. For linear functions
        f(x,y) = ax + by + c, interpolation is exact up to numerical precision.
        Retained grids recover source center planes only within a bounded
        projection error below one quarter of the smallest source cell width.
        Positive interpolation weight on a resolved inactive corner returns NaN.

        **Outside handling**: Points outside the environment bounds return NaN
        in both modes. This prevents extrapolation errors.

        Examples
        --------
        >>> # Nearest-neighbor interpolation (all layouts)
        >>> field = np.random.rand(env.n_bins)  # doctest: +SKIP
        >>> query_points = np.array([[5.0, 5.0], [7.5, 3.2]])  # doctest: +SKIP
        >>> values = env.interpolate(
        ...     field, query_points, mode="nearest"
        ... )  # doctest: +SKIP

        >>> # Linear interpolation (grids only)
        >>> # For plane f(x,y) = 2x + 3y, interpolation is exact
        >>> plane_field = (
        ...     2 * env.bin_centers[:, 0] + 3 * env.bin_centers[:, 1]
        ... )  # doctest: +SKIP
        >>> values = env.interpolate(
        ...     plane_field, query_points, mode="linear"
        ... )  # doctest: +SKIP

        >>> # Evaluate rate map at trajectory positions
        >>> rates_at_trajectory = env.interpolate(
        ...     rate_map, positions, mode="linear"
        ... )  # doctest: +SKIP

        """
        self._require_cartesian("interpolate()")
        return _interpolate(self, field, points, mode=mode)

    @check_fitted
    def rebin(
        self: Environment[CartesianSystem],
        factor: int | tuple[int, ...] | None = None,
        *,
        bin_size: float | tuple[float, ...] | None = None,
    ) -> Environment:
        """Re-bin an arena at a new size, or coarsen a grid by integer factor.

        Creates a new environment with coarser spatial resolution by reducing
        the number of bins. This method only modifies the grid geometry and
        connectivity; it does not aggregate any field values.

        Pass exactly one of ``factor`` or ``bin_size``. The new-size mode
        reconstructs over the retained arena using stored rectangular-lattice
        data; analysis crops do not change the arena. It preserves the grid
        frame and lattice anchor, independently of descriptive provenance.

        Parameters
        ----------
        factor : int or tuple of int, optional
            Coarsening factor per dimension. If int, applied uniformly to all
            dimensions. If tuple, must match the number of dimensions.
            Each factor must be a positive integer. Only supported for the
            existing uniform RegularGrid family; retains truncation behavior.
        bin_size : float or tuple of float, optional
            New positive physical bin lengths along the retained grid axes,
            in environment units. May refine or coarsen a rectangular grid.
            Needs a known arena and a supported stored GridBinningSpec.
            Center selection can leave portions of the arena without a bin;
            use ``clip`` separately when exact clipped cells are needed.

        Returns
        -------
        coarse_env : Environment
            New environment at the requested resolution. With ``factor``, the grid shape is
            ``original_shape // factor`` in each dimension. Keep coarse bins
            whose centers lie inside the original arena.

        Raises
        ------
        RuntimeError
            If called before the environment is fitted.
        NotImplementedError
            If environment layout is not RegularGridLayout.
        ValueError
            If factor is not positive or if factor is too large for grid shape.
            Also raised when coarse cells cannot represent an existing wall.

        See Also
        --------
        smooth : Apply diffusion kernel smoothing to fields.
        subset : Extract spatial subset of environment.
        bin_at : Assign retained original bin centers to coarsened bins.

        Notes
        -----
        **Geometry only**: This method only coarsens the grid structure. To
        aggregate field values (occupancy, spike counts, etc.) from the original
        grid to the coarsened grid, map the original bin centers and aggregate:

            >>> coarse = env.rebin(factor=2)  # doctest: +SKIP
            >>> coarse_indices = coarse.bin_at(env.bin_centers)  # doctest: +SKIP
            >>> retained = coarse_indices >= 0  # doctest: +SKIP
            >>> coarse_field = np.bincount(  # doctest: +SKIP
            ...     coarse_indices[retained],
            ...     weights=field[retained],
            ...     minlength=coarse.n_bins,
            ... )

        **Factor mode**: Integer coarsening requires the uniform RegularGrid
        family. New-size mode uses the stored GridBinningSpec instead of a
        descriptive layout name.

        **Non-divisible dimensions**: If the grid shape is not evenly divisible
        by the factor in any dimension, the grid is truncated to the largest
        multiple of the factor. A warning is issued in this case.

        **Connectivity**: The connectivity graph is rebuilt for the coarsened
        grid with the same connectivity pattern as the original (e.g., if
        original had diagonal connections, coarsened grid will too). Physical
        walls still block connections and must align with complete coarse faces.

        **Bin centers**: New bin centers are computed from the coarsened grid
        edges as midpoints between edge positions.

        **Active bins**: Keep centers strictly inside a polygonal arena, or
        contained by another known arena. Unknown domains retain the full
        coarse grid. Rebinning preserves the arena rather than filling its holes.

        **Metadata preservation**: The arena, units and frame are preserved.
        New-size re-binning also retains regions. It supports rectangular
        grids from samples, masks, images and polygon grids, including their
        derived grid frames. Other families and older records without a stored
        specification refuse this mode. It never transfers fields or models.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> # Create 10x10 grid (deterministic)
        >>> rng = np.random.RandomState(42)
        >>> data = rng.rand(1000, 2) * 100
        >>> env = Environment.from_samples(data, bin_size=10.0)
        >>> env.layout.grid_shape  # May vary with random data
        (11, 11)
        >>>
        >>> # Coarsen by factor 2 → ~5x5 grid (11 // 2 = 5)
        >>> coarse = env.rebin(factor=2)
        >>> coarse.layout.grid_shape
        (5, 5)
        >>>
        >>> # Anisotropic coarsening with tuple (11 // 2 = 5, 11 // 5 = 2)
        >>> coarse_aniso = env.rebin(factor=(2, 5))
        >>> coarse_aniso.layout.grid_shape
        (5, 2)
        >>>
        >>> # Aggregate a field to the coarsened grid
        >>> occupancy = np.random.rand(env.n_bins) * 100
        >>> coarse_indices = coarse.bin_at(env.bin_centers)
        >>> retained = coarse_indices >= 0
        >>> coarse_occupancy = np.bincount(
        ...     coarse_indices[retained],
        ...     weights=occupancy[retained],
        ...     minlength=coarse.n_bins,
        ... )
        >>> # The total over retained source-bin centers is preserved
        >>> bool(np.isclose(occupancy[retained].sum(), coarse_occupancy.sum()))
        True

        """
        self._require_cartesian("rebin()")
        if (factor is None) == (bin_size is None):
            raise ValueError("Pass exactly one of factor or bin_size to rebin().")
        if bin_size is not None:
            from neurospatial.environment._rebin import _rebin_resolution

            return _rebin_resolution(self, bin_size)
        assert factor is not None
        return _rebin(self, factor)

    @check_fitted
    def subset(
        self: Environment[CartesianSystem],
        *,
        bins: NDArray[np.bool_] | None = None,
        region_names: Sequence[str] | None = None,
        polygon: shapely.Polygon | None = None,
        invert: bool = False,
    ) -> Environment:
        """Create new environment containing subset of bins.

        Extracts a subgraph from the environment containing only the selected
        bins. Node indices are renumbered to be contiguous [0, n'-1]. This
        operation drops all regions; users can re-add regions to the subset
        environment if needed. Bin sizes, retained faces, cell shapes and track
        coordinates are preserved. Points in discarded bins are outside the
        subset and map to ``-1``.

        Parameters
        ----------
        bins : NDArray[np.bool_], shape (n_bins,), optional
            Boolean mask of bins to keep. True = keep, False = discard.
        region_names : Sequence[str], optional
            Keep bins whose centers lie inside these named regions.
            Regions must exist in self.regions. Only polygon-type regions
            are supported (point-type regions will raise ValueError).
        polygon : shapely.Polygon, optional
            Keep bins whose centers lie inside this polygon. Only works
            for 2D environments.
        invert : bool, default=False
            If True, invert the selection mask (select complement).

        Returns
        -------
        sub_env : Environment
            New environment with selected bins renumbered to [0, n'-1].
            Connectivity is the induced subgraph. All regions are dropped.
            Metadata (units, frame) is preserved.

        Raises
        ------
        ValueError
            If none or multiple selection parameters provided, if mask has
            wrong shape/dtype, if region names don't exist, if selection is empty.

        Notes
        -----
        Exactly one of {bins, region_names, polygon} must be provided.

        The connectivity graph is the induced subgraph: only edges where both
        endpoints are in the selection are kept. This may create disconnected
        components if the selection is not contiguous.

        Node attributes ('pos', 'source_grid_flat_index', 'original_grid_nd_index')
        describe the selected bins, and edge attributes ('distance', 'vector',
        'angle_2d') are preserved from the original graph. Grid subsets number
        'edge_id' from 0; other subsets keep the original ids. A grid subset
        keeps the parent's edges even where they differ from plain cell
        adjacency, such as a 4-connected grid read from NWB.

        Subsetting selects whole bins without clipping their geometry. The child
        keeps smoothing, plotting, interpolation and serialization wherever its
        retained components support them. Track linear coordinates remain
        absolute in the original track; dropped bins leave gaps in its mask.

        See Also
        --------
        rebin : Coarsen grid resolution (grid-only).
        components : Find connected components.

        Examples
        --------
        >>> # Extract bins inside 'goal' region
        >>> goal_env = env.subset(region_names=["goal"])  # doctest: +SKIP
        >>>
        >>> # Crop to polygon
        >>> from shapely.geometry import box  # doctest: +SKIP
        >>> cropped = env.subset(polygon=box(0, 0, 50, 50))  # doctest: +SKIP
        >>>
        >>> # Select bins by boolean mask
        >>> mask = env.bin_centers[:, 0] < 50  # Left half  # doctest: +SKIP
        >>> left_env = env.subset(bins=mask)  # doctest: +SKIP
        >>>
        >>> # Inverted selection (everything except region)
        >>> outside = env.subset(
        ...     region_names=["obstacle"], invert=True
        ... )  # doctest: +SKIP

        """
        self._require_cartesian("subset()")
        mask = _subset_mask(
            self, bins=bins, region_names=region_names, polygon=polygon, invert=invert
        )
        from neurospatial.layout._binning_layout import BinningLayout
        from neurospatial.layout._components import HexagonShapes, TriangleShapes
        from neurospatial.layout._derive import subset_binning

        binning = subset_binning(self._binning, mask)
        selected = np.flatnonzero(mask)
        child_of = {int(old): new for new, old in enumerate(selected)}
        graph = nx.Graph()
        graph.graph.update(self.connectivity.graph)
        # Retain the parent's node iteration order as well as edge order and
        # orientation; edge fields index that order even on loaded graphs.
        for old, data in self.connectivity.nodes(data=True):
            if old in child_of:
                graph.add_node(child_of[old], **data)
        for index, (u, v) in enumerate(binning.bins.pairs):
            old_u, old_v = selected[u], selected[v]
            data = dict(self.connectivity.edges[old_u, old_v])
            if binning.rectilinear is not None:
                data["edge_id"] = index
            graph.add_edge(int(u), int(v), **data)
        params = {"source": "subset", "original_n_bins": self.n_bins}
        tag = "MaskedGrid" if binning.rectilinear is not None else "subset"
        if binning.rectilinear is not None:
            ranges = [
                (float(edge[0]), float(edge[-1])) for edge in binning.rectilinear.edges
            ]
        else:
            shapes = binning.shapes
            vertices = binning.bins.centers
            if binning.grid_frame is not None:
                vertices = binning.grid_frame.corners()
            elif isinstance(shapes, HexagonShapes):
                angles = shapes.orientation + np.pi / 2 + np.arange(6) * np.pi / 3
                offsets = shapes.radius * np.column_stack(
                    (np.cos(angles), np.sin(angles))
                )
                vertices = (shapes.centers[:, None, :] + offsets).reshape(-1, 2)
            elif isinstance(shapes, TriangleShapes):
                vertices = shapes.points[shapes.triangles].reshape(-1, 2)
            else:
                from neurospatial.layout._polygon import PolygonCells

                if isinstance(shapes, PolygonCells):
                    import shapely

                    vertices = np.concatenate(
                        [shapely.get_coordinates(cell) for cell in shapes.polygons]
                    )
            ranges = []
            for axis in vertices.T:
                low, high = float(axis.min()), float(axis.max())
                margin = 0.05 * (high - low) if shapes is not None else 0.0
                ranges.append((low - margin, high + margin))
        layout = BinningLayout(
            binning,
            connectivity=graph,
            dimension_ranges=ranges,
            layout_type_tag=tag,
            layout_type="mask" if binning.rectilinear is not None else "subset",
            build_params=params,
            is_linearized_track=self.is_linearized_track,
        )
        env_cls = self.__class__
        result = env_cls(
            "",
            layout,
            tag,
            params,
            arena=self.arena,
            units=self.units,
            frame=self.frame,
        )
        return result

    @check_fitted
    def clip(
        self: Environment[CartesianSystem],
        polygon: shapely.Polygon | shapely.MultiPolygon,
        *,
        name: str | None = None,
    ) -> Environment:
        """Clip planar cells to a polygon, returning a new environment.

        Parameters
        ----------
        polygon : Polygon or MultiPolygon
            Retained Cartesian analysis area, including any holes.
        name : str, optional
            Output name; defaults to the parent's name.

        Returns
        -------
        Environment
            Exact positive-area cells. Disconnected pieces get separate bins;
            ``clip_mapping.parent_bin_indices`` identifies their input bins.
            Rates, counts and fitted models are not transferred. Numerical
            operations require certified finite-volume geometry. The analysis
            crop preserves the parent arena and does not assert new walls.

        Raises
        ------
        ValueError
            If geometry is invalid, unavailable, unsupported, or has no
            positive-area intersection. Track, polar and 3-D cells are not
            Cartesian planar area cells.
        """
        self._require_cartesian("clip()")
        from neurospatial.environment._clipping import _clip

        return _clip(self, polygon, name=name)

    @check_fitted
    def apply_transform(
        self: Environment[CartesianSystem],
        transform: AffineND | Affine2D,
        *,
        name: str | None = None,
    ) -> Environment:
        """Apply affine transformation to environment, returning a new instance.

        Transforms the environment's bin centers, connectivity graph, and regions
        using an affine transformation. Supports both 2D (Affine2D) and N-D
        (AffineND) transforms. The transformation must match the environment's
        dimensionality.

        Translation and similarity transforms retain finite-volume smoothing
        and differential operators. Cartesian 2-D/3-D grids retain source axes
        for linear interpolation and movement allocation under rotations,
        reflections and shear. Planar grids draw exact cell corners. New numerical
        transforms need stored face normals, certified source flux distances and
        perpendicular transformed center links (normalized residual at most 1e-12). Smoothing
        refuses when those right angles are lost; exact cells remain available.
        Triangle vertices and deformed hexagon polygons retain their exact mapped
        shapes. Tracks accept translation and similarity only.

        Parameters
        ----------
        transform : AffineND or Affine2D
            Affine transformation to apply. Must match environment dimensionality:

            - For 2D environments: use Affine2D or AffineND with n_dims=2
            - For 3D environments: use AffineND with n_dims=3
            - For N-D environments: use AffineND with n_dims=N

            Create transforms using factory functions like `translate()`,
            `scale_2d()`, `from_rotation_matrix()`, or compose using `@` operator.
        name : str, optional
            Name for the transformed environment. If None, appends "_transformed"
            to the original name.

        Returns
        -------
        transformed_env : Environment
            New Environment instance with transformed coordinates. The transformation
            is applied to:

            - `bin_centers`: All bin positions are transformed
            - `connectivity`: Node 'pos' attributes updated, edge distances/vectors recomputed
            - `regions`: Point and polygon regions are transformed
            - `metadata`: Units preserved, frame updated with "_transformed" suffix

            Bin IDs and edge ordering are retained. Cell/face measures and
            available numerical capabilities follow the transformed geometry.
            Bin-aligned values and fitted models are not rescaled or refitted.

        Raises
        ------
        RuntimeError
            If environment is not fitted (use factory methods like
            `Environment.from_samples()`).
        ValueError
            If transform dimensionality doesn't match environment dimensionality,
            the transform is not invertible, or a track receives a non-similarity
            transform. Rebuild such a track from transformed nodes with
            ``Environment.from_graph``.

        See Also
        --------
        estimate_transform : Estimate transformation from point correspondences
        neurospatial.ops.transforms.Affine2D : 2D affine transformation class
        neurospatial.ops.transforms.AffineND : N-D affine transformation class
        neurospatial.ops.transforms.translate : Create translation transform
        neurospatial.ops.transforms.scale_2d : Create scaling transform
        neurospatial.ops.transforms.from_rotation_matrix : Create rotation transform

        Notes
        -----
        **Pure function**: This method does not modify the source environment;
        it returns a new Environment instance.

        **Transformation order**: When composing transforms, use the `@` operator.
        `T1 @ T2` applies T2 first, then T1:

            >>> composed = (
            ...     rotation @ translation
            ... )  # translation first, then rotation  # doctest: +SKIP
            >>> transformed = env.apply_transform(composed)  # doctest: +SKIP

        **Edge attributes**: After transformation:

        - `distance`: Recomputed from transformed positions
        - `vector`: Recomputed as displacement between transformed positions
        - `angle_2d`: Recomputed for 2D environments only

        **Dimension ranges**: Bounding box is recomputed by transforming all
        corner points of the original bounding box.

        Examples
        --------
        Translation (2D):

        >>> from neurospatial import Environment
        >>> from neurospatial.ops.transforms import translate
        >>> import numpy as np
        >>> # Create 2D environment
        >>> data = np.random.rand(200, 2) * 100
        >>> env = Environment.from_samples(data, bin_size=5.0, name="session1")
        >>> # Translate by (10, 20) cm
        >>> transform = translate(10, 20)
        >>> env_shifted = env.apply_transform(transform, name="session1_aligned")
        >>> # Bin centers are translated
        >>> np.allclose(env_shifted.bin_centers, env.bin_centers + [10, 20])
        True

        Rotation (2D):

        >>> from neurospatial.ops.transforms import Affine2D
        >>> # 45-degree rotation
        >>> angle = np.pi / 4
        >>> R = np.array(
        ...     [
        ...         [np.cos(angle), -np.sin(angle), 0],
        ...         [np.sin(angle), np.cos(angle), 0],
        ...         [0, 0, 1],
        ...     ]
        ... )
        >>> transform = Affine2D(R)
        >>> env_rotated = env.apply_transform(transform)
        >>> # Distances preserved under rotation
        >>> assert env_rotated.n_bins == env.n_bins

        Composed transforms (scale → rotate → translate):

        >>> from neurospatial.ops.transforms import (
        ...     scale_2d,
        ...     translate,
        ... )  # doctest: +SKIP
        >>> # Build transformation pipeline
        >>> T = translate(50, 50) @ Affine2D(R) @ scale_2d(1.2)  # doctest: +SKIP
        >>> env_aligned = env.apply_transform(T, name="aligned")  # doctest: +SKIP

        Cross-session alignment using landmarks:

        >>> from neurospatial.ops.transforms import estimate_transform  # doctest: +SKIP
        >>> # Session 1 landmarks (e.g., arena corners)
        >>> landmarks_s1 = np.array(
        ...     [[0, 0], [100, 0], [100, 100], [0, 100]]
        ... )  # doctest: +SKIP
        >>> # Session 2 landmarks (same physical locations, different coordinates)
        >>> landmarks_s2 = np.array(
        ...     [[5, 10], [95, 15], [90, 105], [0, 100]]
        ... )  # doctest: +SKIP
        >>> # Estimate rigid transform (rotation + translation)
        >>> T = estimate_transform(
        ...     landmarks_s1, landmarks_s2, kind="rigid"
        ... )  # doctest: +SKIP
        >>> # Transform session 1 environment to session 2 coordinates
        >>> env_s1_aligned = env_s1.apply_transform(
        ...     T, name="session1_in_s2_coords"
        ... )  # doctest: +SKIP

        3D transformation:

        >>> from scipy.spatial.transform import Rotation  # doctest: +SKIP
        >>> from neurospatial.ops.transforms import (
        ...     from_rotation_matrix,
        ...     translate_3d,
        ... )  # doctest: +SKIP
        >>> # Create 3D environment
        >>> data_3d = np.random.randn(500, 3) * 20  # doctest: +SKIP
        >>> env_3d = Environment.from_samples(data_3d, bin_size=3.0)  # doctest: +SKIP
        >>> # Rotate 45 degrees around z-axis and translate
        >>> R_3d = Rotation.from_euler(
        ...     "z", 45, degrees=True
        ... ).as_matrix()  # doctest: +SKIP
        >>> rotation = from_rotation_matrix(R_3d)  # doctest: +SKIP
        >>> translation = translate_3d(10, 20, 30)  # doctest: +SKIP
        >>> T_3d = translation @ rotation  # doctest: +SKIP
        >>> env_3d_transformed = env_3d.apply_transform(T_3d)  # doctest: +SKIP

        With regions (regions are automatically transformed):

        >>> from neurospatial.regions import Regions
        >>> regions = Regions(env.regions.values())
        >>> regions.add("goal", point=[80, 90])  # doctest: +SKIP
        >>> env = env.with_regions(regions)
        >>> env_transformed = env.apply_transform(translate(10, 10))  # doctest: +SKIP
        >>> # Region is transformed along with environment
        >>> goal_region = env_transformed.regions["goal"]  # doctest: +SKIP
        >>> np.allclose(goal_region.data, [90, 100])  # doctest: +SKIP
        True

        """
        # Import here to avoid circular dependency
        # (transforms.py imports Environment, which imports this mixin)
        # Delegate to free function
        # Cast self to Environment for type checking
        self._require_cartesian("apply_transform()")
        from neurospatial.ops.transforms import (
            apply_transform_to_environment as _apply_transform_impl,
        )

        env = self

        return _apply_transform_impl(env, transform, name=name)
