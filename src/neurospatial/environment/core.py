"""One public Environment with owned geometry and thin domain entry points."""

from __future__ import annotations

import copy
import logging
from collections.abc import Mapping, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    TypeVar,
    overload,
)

import networkx as nx
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse
from typing_extensions import Self

from neurospatial._exceptions import _format_error
from neurospatial._intervals import DEFAULT_MAX_GAP
from neurospatial._keyword_hint import keyword_hint
from neurospatial.environment._affine import _apply_transform
from neurospatial.environment._bin_mapping import BinMapping, _bin_mapping_from
from neurospatial.environment._cache import (
    BIN_ATTRIBUTES,
    BOUNDARY_BINS,
    DIFFERENTIAL_OPERATOR,
    DIVERGENCE_OPERATOR,
    EDGE_ATTRIBUTES,
    AnalysisCache,
)
from neurospatial.environment._cartesian import (
    cartesian_only,
    cartesian_only_members,
)
from neurospatial.environment._clipping import _clip
from neurospatial.environment._combine import _combine
from neurospatial.environment._factories import (
    _OPEN_FIELD_OPTION_DEFAULTS,
    _from_bins,
    _from_graph,
    _from_grid_mask,
    _from_layout,
    _from_nwb,
    _from_pixel_mask,
    _from_polygon,
    _from_samples,
    _linear_track,
    _maze,
)
from neurospatial.environment._fields import (
    _compute_gaussian_kernel,
    _compute_kernel,
    _diffuse,
    _interpolate,
    _mrf_basis,
    _smooth,
    diffusion_eigenbasis,
    diffusion_geometry,
    require_face_geometry,
    smoothing_geometry,
)
from neurospatial.environment._lookup import bin_spacing, center_tree, nearest_bin
from neurospatial.environment._metrics import (
    _bin_attributes,
    _blocked_edges,
    _copy_attributes,
    _edge_attributes,
    _get_bridge_attributes,
)
from neurospatial.environment._polar import _create_polar_environment
from neurospatial.environment._publication import (
    UNSET,
    _diffusion_geometry_view,
    _Metadata,
    _Provenance,
    _PublishedGeometry,
    _replace_metadata,
    _restored_state,
    _Unset,
    editable_graph,
    environment_state,
    owned_landmarks,
    publish_derived,
)
from neurospatial.environment._queries import (
    _bin_center_of,
    _components,
    _distance_to,
    _linear_to_nd,
    _path_between,
    _reachable_from,
    _rings,
    _to_linear,
)
from neurospatial.environment._rebin import _rebin, warn_rebin_losses
from neurospatial.environment._regions import (
    _bins_in_region,
    _owned_regions,
    _region_membership,
    _relabeled_regions,
)
from neurospatial.environment._subset import _subset
from neurospatial.environment._summary import (
    _info,
    _mixed_measure_error,
    _repr,
    _repr_html,
)
from neurospatial.environment._targets import (
    REGION_TARGET_WHY,
    TARGET_WHY,
    former_targets,
    resolve_targets,
)
from neurospatial.environment._track_width import _with_track_width
from neurospatial.environment._trajectory import (
    BinSequenceWithRuns,
    _bin_sequence,
    _occupancy,
    _transitions,
)
from neurospatial.environment._visualization import (
    PlotLabels,
    _plot,
    _plot_1d,
    _plot_field,
    check_field,
)
from neurospatial.geometry import (
    Binning,
    CartesianSystem,
    EgocentricPolarSystem,
    boundary_bins,
    clip_mapping_of,
    composition_bin_sources,
    composition_component_keys,
    locate,
    own_binning,
)
from neurospatial.landmarks import Landmarks
from neurospatial.layout import LayoutType
from neurospatial.ops import (
    _compute_differential_operator,
    _compute_divergence_operator,
    _diffusion_apply_rtol,
    _regions_mask,
)
from neurospatial.ops.distance import geodesic_distance_between_points
from neurospatial.ops.mapping import TieBreakStrategy
from neurospatial.regions import Region, Regions

if TYPE_CHECKING:
    from pathlib import Path

    import fastplotlib
    import matplotlib.axes
    import pandas as pd
    import scipy.sparse
    import shapely
    from matplotlib.colors import Colormap
    from scipy.spatial import cKDTree

    from neurospatial.animation._view import AnimationView
    from neurospatial.animation.overlays import OverlayProtocol
    from neurospatial.geometry import Arena, ClipMapping, DiffusionGeometry
    from neurospatial.layout import ScaleBarConfig
    from neurospatial.layout.base import LayoutEngine
    from neurospatial.ops import KernelMode
    from neurospatial.ops._types import DistanceMetric, ResolvedBackend
    from neurospatial.ops.diffusion import MRFBasis
    from neurospatial.ops.transforms import AffineND

logger = logging.getLogger(__name__)


def _check_backend(backend: str) -> str:
    """Return a drawing backend name, or say which ones exist."""
    if backend not in ("matplotlib", "fastplotlib"):
        raise ValueError(
            _format_error(
                f"Unknown backend {backend!r}.",
                fix="pass backend='matplotlib' (the default) or "
                "backend='fastplotlib' to draw on the GPU",
            )
        )
    return backend


# Smoothing methods require mode: no one mode is right for every field.
_MODE_CHOICE = (
    "a field of values per bin (a rate map) and a field of totals per bin "
    "(counts, occupancy) are smoothed differently, and the modes give "
    "different numbers for the same field.",
    "pass mode='average' for a rate map or other value per bin, "
    "mode='transition' for counts, occupancy or probability mass (keeps the "
    "total), or mode='density' to turn counts into a density",
)

if TYPE_CHECKING:
    from typing_extensions import TypeVar as _CoordinateTypeVar

    _C = _CoordinateTypeVar(
        "_C", bound=CartesianSystem | EgocentricPolarSystem, default=Any
    )
else:
    _C = TypeVar("_C", bound=CartesianSystem | EgocentricPolarSystem)


_METADATA_FIELDS = frozenset({"name", "units", "frame"})


def _read_only(cls: type, name: str) -> AttributeError:
    """The error for assigning or deleting a public Environment attribute."""
    if not hasattr(cls, name):
        return AttributeError(
            _format_error(
                f"Environment has no attribute {name!r} and does not accept new ones.",
                why="Why: an environment never changes after it is built.",
                fix=(
                    "keep your own labels next to it, e.g. in a dict keyed by "
                    "env.name, or rename it with env.with_metadata(name=...)"
                ),
            )
        )
    if name in _METADATA_FIELDS:
        return AttributeError(
            f"Environment.{name} is read-only; use env.with_metadata({name}=...)."
        )
    if name == "regions":
        return AttributeError(
            "Environment.regions is read-only; use env.with_regions(regions)."
        )
    if name == "landmarks":
        return AttributeError(
            "Environment.landmarks is read-only; use env.with_landmarks(landmarks)."
        )
    return AttributeError(
        _format_error(
            f"Environment.{name} is read-only.",
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


class Environment(Generic[_C]):
    """A discretized environment with owned coordinate meaning and connectivity.

    ``Environment`` is the central user-facing spatial type. Each instance
    declares whether ``bin_centers`` hold Cartesian ``(x, y[, z])`` positions
    or egocentric ``(distance, angle)`` coordinates. Methods such as
    ``apply_transform``, ``rebin`` and ``interpolate`` require Cartesian
    positions; ``bin_at``, ``contains``, ``occupancy`` and the bin-sequence
    methods read ``(distance, angle)`` rows on a polar environment. The
    shared ``distance_to`` method guards its Euclidean mode.

    Instances are created via factory classmethods on ``Environment``
    (``from_samples``, ``from_polygon``, ``from_grid_mask``,
    ``from_pixel_mask``, ``from_graph``, ``from_layout``).

    Cartesian factories store position coordinates in length units. The
    from_polar_egocentric factory returns the same runtime class with radius in
    length units and angle in radians. Cartesian-only methods refuse polar
    instances before interpreting those coordinates.

    Notes
    -----
    **Missing data.** NaN marks a missing value, never zero.

    - Per-sample queries (``bin_at``, ``contains``, ``nearest_bin``,
      ``interpolate``): a sample with a NaN coordinate is missing. It gets bin
      ``-1``, ``False`` or a NaN value. ``-1`` also marks samples outside the
      environment.
    - Trajectories (the positions given to ``occupancy``, ``bin_sequence``
      and ``transitions``): a sample with a NaN coordinate is missing and is
      treated exactly like a deleted row, so a dropped frame and a
      low-confidence frame give the same result. The gap between the valid
      samples around it is bridged when it is no longer than ``max_gap``
      (its time held at the sample before it) and excluded otherwise.
    - Explicit targets (``points=`` in ``distance_between``, ``distance_to``,
      ``neighbors`` and the other target queries) refuse NaN with an error,
      because a missing target has no answer. ``inf`` from
      ``distance_between`` only ever means "unreachable".
    - Fields (``smooth``, ``diffuse``, ``interpolate``): a NaN bin has no
      value. Average-mode smoothing averages over the observed bins and fills
      holes from them; a bin stays NaN only when almost none of its smoothing
      weight falls on observed bins. The mass-conserving modes refuse NaN
      (see Missing bins in ``smooth``).

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

    _metadata: _Metadata
    _regions: Regions
    _landmarks: Landmarks
    _provenance: _Provenance
    _geometry: _PublishedGeometry[_C]
    _cache: AnalysisCache

    def __setattr__(self, name: str, value: object) -> None:
        """Refuse public attribute assignment: an environment never changes."""
        if not name.startswith("_"):
            raise _read_only(type(self), name)
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        """Refuse public attribute deletion: an environment never changes."""
        if not name.startswith("_"):
            raise _read_only(type(self), name)
        object.__delattr__(self, name)

    @property
    def name(self) -> str:
        """Environment name; change it with ``env.with_metadata(name=...)``."""
        return self._metadata.name

    @property
    def units(self) -> str | None:
        """Length units of the coordinates, such as ``"cm"``, or None."""
        return self._metadata.units

    @property
    def frame(self) -> str | None:
        """Coordinate frame label, such as ``"room"``, or None."""
        return self._metadata.frame

    @property
    def bin_centers(self) -> NDArray[np.float64]:
        """Bin centers, shape ``(n_bins, n_dims)``, as a read-only array."""
        centers: NDArray[np.float64] = self._binning.bins.centers.view()
        return centers

    @property
    def connectivity(self) -> nx.Graph:
        """Frozen navigation graph over bin IDs; ``to_networkx()`` copies it."""
        return self._geometry.graph

    @property
    def dimension_ranges(self) -> tuple[tuple[float, float], ...] | None:
        """``(low, high)`` per coordinate axis covered by the bins' cells, or None.

        Every layout reports the extent of its cells: a grid its active
        cells (``grid_edges`` keeps the full lattice, inactive cells
        included); hexagons their corners; a mesh its boundary, or the
        triangles a subset keeps; a
        track the stretch of centerline its bins cover (the centerline
        alone, not the track's width). A subset therefore never reports a
        larger range than its parent. None only for bins without cells or
        declared ranges, such as explicit point bins.
        """
        return self._binning.dimension_ranges

    @property
    def bin_spacing(self) -> float:
        """Typical distance between neighboring bins, in the environment's units.

        The one length scale of the bins that default thresholds and widths
        are expressed in (a place field 3 bin spacings wide, a mapped point at
        most one bin spacing from the target bins). On a linearized track it
        is the median bin length along the track, because the nearest bin
        center can lie on another arm. Otherwise it is the median, over every
        bin, of the distance from a bin's center to its nearest other center:
        ``bin_size`` on a regular or hexagonal grid. On an egocentric polar
        environment the distances are measured between places in the animal's
        plane, not between (distance, angle) rows. A single bin off a track
        has no neighbor, so its spacing is ``inf``.

        Computed on first use and kept in the analysis cache.

        Returns
        -------
        float

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> rng = np.random.default_rng(0)
        >>> env = Environment.from_samples(rng.uniform(0, 50, (2000, 2)), bin_size=5.0)
        >>> env.bin_spacing
        5.0
        """
        return bin_spacing(self._cache, self._binning, self._geometry.coordinates)

    @property
    def grid_edges(self) -> tuple[NDArray[np.float64], ...]:
        """Cell edges per axis of the full grid; empty for bins without a grid."""
        return tuple(edge.view() for edge in self._geometry.grid.edges)

    @property
    def grid_shape(self) -> tuple[int, ...]:
        """Shape of the full logical grid, inactive cells included."""
        return self._geometry.grid.shape

    @property
    def active_mask(self) -> NDArray[np.bool_]:
        """Read-only mask of the active cells of the full grid, shape ``grid_shape``."""
        mask: NDArray[np.bool_] = self._geometry.grid.active_mask.view()
        return mask

    @property
    def regions(self) -> Mapping[str, Region]:
        """Named regions, read-only, in their order.

        Their geometry is in this environment's coordinates. To change them,
        edit a builder, ``Regions(env.regions.values())``, and publish it with
        :meth:`with_regions`.
        """
        return self._regions

    @property
    def landmarks(self) -> Landmarks:
        """Named landmarks: cues, monitors, objects and reward ports.

        Their geometry is in this environment's coordinates. Encoding
        functions that take ``object_positions`` accept their names.
        """
        return self._landmarks

    def __init__(self) -> None:
        """Use an Environment factory to build owned geometry and metadata."""
        raise ValueError(
            _format_error(
                "[E1006] Environment cannot be constructed directly — use a factory method.\n\nMost common (from positions you recorded):\n    env = Environment.from_samples(positions, bin_size=2.0)\n\nOther factories, chosen by the data you have:\n    from_polygon     — a Shapely polygon boundary\n    from_graph       — a track/maze graph (linearized 1D)\n    from_grid_mask   — an N-D boolean mask + grid edges\n    from_pixel_mask  — a 2D image / pixel mask\n\nAvoid:\n    env = Environment()  # not supported\n\nSee each factory's docstring for its exact arguments, or:\n    https://edeno.github.io/neurospatial/errors/#e1006-environment-constructed-directly",
                fix="env = Environment.from_samples(positions, bin_size=2.0)",
                why="Why: a factory builds the geometry and connectivity required by Environment.",
            )
        )

    def __getstate__(self) -> dict[str, Any]:
        """Return the owned state with the version of its private format."""
        return environment_state(self.__dict__)

    def __setstate__(self, state: object) -> None:
        """Restore owned state; the analysis cache unpickles empty.

        The geometry records mark their arrays read-only again as they are
        unpickled.

        Raises
        ------
        TypeError
            If ``state`` is not the state of an Environment pickled by this
            version of neurospatial.
        """
        self.__dict__.update(_restored_state(state))

    @property
    def _binning(self) -> Binning:
        """The binning snapshot published with the current geometry."""
        return self._geometry.binning

    @property
    def _arena(self) -> Arena:
        """The arena published with the current geometry."""
        return self._geometry.arena

    @property
    def binning(self) -> Binning:
        """Return a detached snapshot of bins, lookup and geometry components.

        Returns
        -------
        Binning
            Independently owned geometry records in this environment's coordinates.

        Notes
        -----
        The returned records are a copy that shares no array with this
        environment. Their arrays are read-only, so an accidental write
        raises, and reshaping one in place (``array.shape = ...``), which a
        read-only array allows, cannot change this environment.
        Native face recipes remain lazy; reading this property does not build
        numerical geometry. ``Environment.from_bins(env.binning,
        arena=env.arena)`` builds and checks them, so a Cartesian snapshot
        rebuilds an equivalent environment. MissingFaces is already explicit
        and retains its refusal reason.

        Polar snapshots retain radius/angle coordinates and are not inputs to
        the Cartesian from_bins() factory. Use environment storage to restore
        an unchanged polar environment.
        """
        return own_binning(self._binning)

    @property
    def arena(self) -> Arena:
        """Continuous domain, distinguished from the active analysis bins.

        The returned record is a detached copy, as for :attr:`binning`: its
        arrays are read-only, and reshaping one in place cannot change this
        environment.
        """
        return copy.deepcopy(self._geometry.arena)

    @property
    def clip_mapping(self) -> ClipMapping | None:
        """Most recent clip's source-bin mapping, or None before clipping.

        The returned record is a detached copy, as for :attr:`binning`.
        """
        return copy.deepcopy(clip_mapping_of(self._binning.locator))

    @property
    def edge_pairs(self) -> NDArray[np.intp]:
        """Oriented neighbor pairs in gradient and flux order, shape (n_edges, 2).

        This order preserves explicit bin input, and can differ from NetworkX
        edge iteration. The returned array is read-only.
        """
        pairs: NDArray[np.intp] = self._binning.connections.pairs.view()
        return pairs

    def to_networkx(self) -> nx.Graph:
        """Return a detached editable graph with bin IDs and weighted edges.

        ``env.connectivity`` is rebuilt from the bins and holds only the
        standard node and edge attributes, as numbers and tuples, and refuses
        every edit. This export is an ordinary graph with editable attribute
        dicts and graph metadata; tuple values stay tuples. Node identities
        are the bin IDs; ``edge_pairs`` defines numerical edge order. Standard
        NetworkX drawing, editing and JSON conversion can use this graph.

        Returns
        -------
        networkx.Graph
            Editable navigation graph, independent of the environment.
        """
        return editable_graph(self.connectivity)

    def _locate(self, positions: NDArray[np.float64], call: str) -> NDArray[np.intp]:
        """Apply known arena containment before assigning active bins."""
        return locate(self._binning, self._arena, positions, self._cache, call=call)

    def _target_bins(
        self,
        operation: str,
        bins: object,
        points: ArrayLike | None,
        *,
        region: str | None = None,
        takes_region: bool = False,
        count: int | None = None,
    ) -> NDArray[np.intp]:
        """Resolve a query's one explicit target to active-bin indices."""
        return resolve_targets(
            operation,
            bins=bins,
            points=points,
            region=region,
            binning=self._binning,
            arena=self._arena,
            indexes=self._cache,
            coordinates=self._geometry.coordinates,
            regions=self.regions if takes_region else None,
            count=count,
        )

    def _replace_value(self) -> Self:
        """Share published geometry and its analysis cache with a new instance."""
        result = type(self).__new__(type(self))
        result.__dict__.update(self.__dict__)
        return result

    def __copy__(self) -> Self:
        """Return ``self``: an Environment is immutable, so no copy is needed."""
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> Self:
        """Return ``self``: an Environment is immutable, so no copy is needed."""
        return self

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
            Regions share these coordinates, so a new ``units`` label also
            replaces each region's recorded ``metadata["coord_system"]`` when
            it matched the old label. A region recorded in other units than
            the new label is refused, since relabelling converts nothing.

        Returns
        -------
        Environment
            Same concrete type and geometry. It shares this environment's
            analysis cache, so analyses computed on either are reused by both.
        """
        result = self._replace_value()
        result._metadata = _replace_metadata(
            self._metadata, name=name, units=units, frame=frame
        )
        if result.units != self.units:
            result._regions = _relabeled_regions(
                self._regions, self.units, result.units
            )
        return result

    def with_regions(self, regions: Mapping[str, Region]) -> Self:
        """Return a new environment owning a snapshot of a region collection.

        Parameters
        ----------
        regions : mapping of str to Region
            Complete replacement, in iteration order, keyed by region name: a
            :class:`~neurospatial.regions.Regions` builder, another
            environment's ``regions`` or a plain dict. An empty mapping clears
            all regions. Coordinates must match this environment's dimension;
            regions may extend outside the arena.

        Returns
        -------
        Environment
            Same concrete type and geometry. It shares this environment's
            analysis cache, so analyses computed on either are reused by both.

        Raises
        ------
        TypeError
            If ``regions`` is not a mapping or holds a value that is not a
            :class:`~neurospatial.regions.Region`.
        ValueError
            If a key differs from its region's name, a region's dimension
            differs from this environment's, or a region's
            ``metadata["coord_system"]`` (``"pixels"`` or ``"cm"``, set by
            the annotation readers) differs from :attr:`units`.
        """
        snapshot = _owned_regions(regions, self.n_dims, self.units)
        result = self._replace_value()
        result._regions = snapshot
        return result

    def with_landmarks(self, landmarks: Landmarks) -> Self:
        """Return a new environment with these landmarks.

        Parameters
        ----------
        landmarks : Landmarks
            Complete replacement, in order; an empty collection clears them.
            Coordinates must match this environment's dimension, and may lie
            outside the arena, as a cue on the room wall does.

        Returns
        -------
        Environment
            Same concrete type and geometry. It shares this environment's
            analysis cache.

        Raises
        ------
        ValueError
            If a landmark's dimension differs, or this environment is polar.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.landmarks import Landmark, Landmarks
        >>> env = Environment.from_grid_mask(
        ...     np.ones((10, 10), dtype=bool),
        ...     grid_edges=(np.arange(0.0, 101.0, 10.0),) * 2,
        ...     units="cm",
        ... )
        >>> env = env.with_landmarks(
        ...     Landmarks(
        ...         [
        ...             Landmark("cue_card", "point", [50.0, 105.0]),
        ...             Landmark("monitor", "line", [(0.0, 100.0), (40.0, 100.0)]),
        ...         ]
        ...     )
        ... )
        >>> list(env.landmarks)
        ['cue_card', 'monitor']
        """
        snapshot = owned_landmarks(landmarks, self.n_dims, self.coordinate_kind)
        result = self._replace_value()
        result._landmarks = snapshot
        return result

    def with_track_width(
        self: Environment[CartesianSystem], width: float
    ) -> Environment[CartesianSystem]:
        """Return the same track with a known full width.

        A track built without ``track_width``, including every track saved by
        an earlier version of neurospatial, does not know how far from its
        centerline a position is still on the track, so it refuses to look up
        2-D positions. This gives it that width while keeping its bins,
        linearization, name, units, frame and regions. It works on a subset
        or a transformed track too. A track that already has a width gets the
        new one.

        Parameters
        ----------
        width : float
            Full width of the track, in the same units as ``bin_centers``. A
            position is on the track when it is within ``width / 2`` of the
            centerline; round caps extend each end by the same distance.

        Returns
        -------
        Environment
            Same concrete type and bins, with ``"with_track_width"`` added to
            ``derivations``. Positions off the track map to ``-1``.

        Raises
        ------
        TypeError
            If this environment is not a linearized track.
        ValueError
            If ``width`` is not a finite positive number.

        Examples
        --------
        >>> from neurospatial import Environment
        >>> track = Environment.linear_track(
        ...     endpoints=[(0.0, 0.0), (100.0, 0.0)], bin_size=2.0, units="cm"
        ... )
        >>> track = track.with_track_width(8.0)
        >>> track.bin_at(np.array([[51.0, 3.0], [51.0, 30.0]])).tolist()
        [25, -1]
        >>> track.derivations
        ('with_track_width',)
        """
        derivation = _with_track_width(self._binning, self._arena, width)
        return publish_derived(self, derivation)

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
        >>> x, y = np.meshgrid(np.arange(0.0, 10.0), np.arange(0.0, 10.0))
        >>> samples = np.c_[x.ravel(), y.ravel()]
        >>> env = Environment.from_samples(samples, bin_size=2.0, name="MyEnv")
        >>> env
        Environment(name='MyEnv', 2D, 36 bins, RegularGrid)

        """
        return _repr(
            class_name=type(self).__name__,
            name=self.name,
            n_dims=self.n_dims,
            n_bins=self.n_bins,
            layout_type=self.layout_type,
        )

    def __str__(self) -> str:
        """Human-readable summary, equivalent to :meth:`info`.

        ``str(env)`` returns the same multi-line summary that
        ``env.info()`` produces. ``repr(env)`` keeps its single-line
        ``Environment(name=..., 2D, 100 bins, RegularGrid)`` shape for
        interactive debugging and logs.
        """
        return self.info()

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
        In a Jupyter notebook, evaluating an Environment in a cell displays
        this table automatically. Calling the method returns the HTML string:

        >>> import numpy as np
        >>> x, y = np.meshgrid(np.arange(0.0, 10.0), np.arange(0.0, 10.0))
        >>> samples = np.c_[x.ravel(), y.ravel()]
        >>> env = Environment.from_samples(samples, bin_size=2.0, name="MyEnv")
        >>> html = env._repr_html_()
        >>> html.startswith("<div")
        True
        >>> "MyEnv" in html
        True

        """
        return _repr_html(
            name=self.name,
            layout_type=self.layout_type,
            n_dims=self.n_dims,
            n_bins=self.n_bins,
            dimension_ranges=self.dimension_ranges,
            n_regions=len(self.regions),
            is_linearized_track=self.is_linearized_track,
        )

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
        >>> # Positions sampled every 2.5 cm over a 100 x 100 cm arena
        >>> x, y = np.meshgrid(np.linspace(0, 100, 41), np.linspace(0, 100, 41))
        >>> samples = np.c_[x.ravel(), y.ravel()]
        >>> env = Environment.from_samples(samples, bin_size=5.0, name="OpenField")
        >>> print(env.info())
        Environment Information
        =======================
        <BLANKLINE>
        Name: OpenField
        Layout Type: RegularGrid
        Dimensions: 2
        Number of Bins: 441
        <BLANKLINE>
        Spatial Extent:
          Dimension 0: [-2.50, 102.50] (range: 105.00)
          Dimension 1: [-2.50, 102.50] (range: 105.00)
        <BLANKLINE>
        Bin Sizes:
          Dimension 0: 5.00
          Dimension 1: 5.00
        <BLANKLINE>
        Regions: None
        <BLANKLINE>

        """
        return _info(
            name=self.name,
            layout_type=self.layout_type,
            n_dims=self.n_dims,
            bin_sizes=self._binning.bins.sizes,
            measure_dims=self._binning.bins.measure_dims,
            dimension_ranges=self.dimension_ranges,
            grid_edges=self.grid_edges,
            region_names=list(self.regions),
            landmark_names=list(self.landmarks),
            is_linearized_track=self.is_linearized_track,
        )

    @property
    def is_linearized_track(self) -> bool:
        """Whether the bins lie along a linearized track.

        Returns
        -------
        bool
            True when the binning carries a track linearization, as
            ``from_graph`` and ``linear_track`` build, so ``to_linear`` and
            ``linear_to_nd`` apply; False otherwise.

        """
        return self._binning.linearization is not None

    @property
    def n_dims(self) -> int:
        """Return the number of spatial dimensions of the active bin centers.

        Returns
        -------
        int
            The number of dimensions (e.g., 1 for a line, 2 for a plane).
            Derived from the shape of `self.bin_centers`.

        """
        return self._binning.bins.n_dims

    @property
    def layout_parameters(self) -> dict[str, Any]:
        """Return the scalar choices used to build the bins.

        These are the settings passed to the layout engine, such as
        ``bin_size`` or ``connect_diagonal_neighbors``. Input data
        (positions, masks, images, polygons and track graphs) is not kept
        here; the geometry built from it is in ``env.binning``, for example
        ``env.binning.linearization.graph`` for a track. The values describe
        how the environment was built and do not affect any analysis.
        An option the bins did not use is recorded at the value that does
        nothing, so ``open_field(..., infer_active_bins=False)`` records
        ``fill_holes=False``: no bin was filled.

        ``subset``, ``clip`` and ``apply_transform`` keep the bins they were
        given, so they keep these choices, and lengths such as ``bin_size``
        stay in the coordinates the bins were built in: after
        ``apply_transform(scale_2d(2.0))`` the bins are twice as wide as the
        recorded ``bin_size``. ``rebin`` builds new bins and
        records their ``bin_size`` and ``connect_diagonal_neighbors``.
        ``combine`` records none. :attr:`derivations` lists what was applied.

        Returns
        -------
        dict of str to object
            A new dict of scalars and tuples of scalars.
        """
        return self._provenance.build_params

    @property
    def layout_type(self) -> str:
        """Return the label of the layout the bins were built with.

        Built-in layouts report a :class:`~neurospatial.layout.LayoutType`
        value, such as ``"RegularGrid"``, ``"Hexagonal"`` or ``"Polar"``; the
        label is the same as the layout engine's ``layout_type`` and
        ``env.binning.kind``. A derived environment (``subset``, ``clip``,
        ``rebin``, ``apply_transform``) keeps its parent's label, and
        :meth:`combine` reports ``"Combined"``; :attr:`derivations` records
        what was applied. The label is descriptive: analyses read the bins,
        never this label.

        Returns
        -------
        str
            The layout label.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> positions = np.random.default_rng(0).uniform(0, 20, (200, 2))
        >>> env = Environment.from_samples(positions, bin_size=5.0)
        >>> env.layout_type
        'RegularGrid'
        >>> env.subset(mask=np.arange(env.n_bins) < 4).layout_type
        'RegularGrid'
        """
        return self._binning.kind

    @property
    def derivations(self) -> tuple[str, ...]:
        """Return the derivations applied since the bins were built, oldest first.

        Each entry names the method that made a new environment from an
        existing one: ``"subset"``, ``"clip"``, ``"rebin"``,
        ``"apply_transform"``, ``"with_track_width"`` or ``"combine"``. A
        factory-built environment
        has none, and ``combine`` starts a new history because the result
        has no single parent. Relabelling with ``with_metadata`` or
        ``with_regions`` is not a derivation.

        Returns
        -------
        tuple of str
            Derivation names, oldest first.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> grid_edges = (np.arange(5) * 5.0,) * 2  # a 20 x 20 cm grid of 5 cm cells
        >>> env = Environment.from_grid_mask(np.ones((4, 4), dtype=bool), grid_edges)
        >>> env.derivations
        ()
        >>> env.subset(mask=np.arange(env.n_bins) < 8).rebin(2).derivations
        ('subset', 'rebin')
        """
        return self._provenance.derivations

    @property
    def n_bins(self) -> int:
        """Return the number of active bins in the environment.

        This is determined by the number of rows in `self.bin_centers`.

        Returns
        -------
        int
            The number of active bins.

        """
        return self._binning.bins.n_bins

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

        Both matrices are cached privately for this immutable geometry.
        ``clear_cache()`` discards them when needed.

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
        neurospatial.ops.calculus.gradient : Apply it to a scalar field.

        References
        ----------
        .. [1] PyGSP: Graph Signal Processing in Python
               https://pygsp.readthedocs.io/
        .. [2] Shuman et al. (2013). "The emerging field of signal processing on graphs."
               IEEE Signal Processing Magazine, 30(3), 83-98.
        """
        return self._differential_operator_cached.copy()

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
            Sparse operator of shape (n_bins, n_edges). The private cache is
            reused until clear_cache().

        Raises
        ------
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

    @property
    def _differential_operator_cached(self) -> sparse.csc_matrix:
        def build() -> sparse.csc_matrix:
            faces = require_face_geometry(
                self._cache, self._binning, "gradient/divergence"
            )
            return _compute_differential_operator(faces, self.n_bins)

        return self._cache.get(DIFFERENTIAL_OPERATOR, build)

    @property
    def _divergence_operator_cached(self) -> sparse.csc_matrix:
        def build() -> sparse.csc_matrix:
            faces = require_face_geometry(
                self._cache, self._binning, "gradient/divergence"
            )
            return _compute_divergence_operator(
                faces, self._differential_operator_cached
            )

        return self._cache.get(DIVERGENCE_OPERATOR, build)

    def clear_cache(self) -> None:
        """Release every cached analysis of this environment's geometry.

        An environment computes some analyses once and keeps them: the
        KD-tree used by nearest-bin lookup, diffusion and Gaussian smoothing
        kernels, the eigenbasis used by ``diffuse``, ``smooth`` and the MRF
        basis, cell faces and diffusion geometry, the gradient and divergence
        operators, boundary bins and the bin and edge attribute tables. They
        depend only on the geometry, which never changes, so keeping them is
        always correct. Clearing them only frees memory; the next analysis
        computes what it needs again and returns the same results.

        Notes
        -----
        ``with_metadata`` and ``with_regions`` return environments with the
        same geometry, and they share this environment's cache. Clearing it on
        any of them clears it for all of them. Environments derived with
        ``subset``, ``clip``, ``rebin``, ``apply_transform`` or ``combine``,
        and environments loaded from a file, a dict, NWB or a pickle, have
        their own caches.

        Pickling never includes cached analyses, so clearing is not needed
        before sending an environment to worker processes.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> data = np.random.default_rng(0).uniform(0, 100, size=(1000, 2))
        >>> env = Environment.from_samples(data, bin_size=5.0)
        >>> kernel = env.compute_kernel(bandwidth=10.0, mode="average")
        >>> renamed = env.with_metadata(name="session 2")
        >>> env.clear_cache()  # also clears the cache that ``renamed`` uses
        >>> np.allclose(renamed.compute_kernel(bandwidth=10.0, mode="average"), kernel)
        True
        """
        self._cache.clear()

    @property
    def coordinate_kind(self) -> Literal["cartesian", "polar"]:
        """Coordinate meaning: Cartesian positions or egocentric radius/angle.

        Polar radius uses declared length units; angle is always in radians.
        This read-only property replaces public-class checks for dispatch.
        """
        return self._geometry.coordinates.kind

    @property
    def coordinate_system(self) -> _C:
        """Return this environment's immutable coordinate descriptor.

        CartesianSystem describes position columns in length units.
        EgocentricPolarSystem describes radius in length units and angle in
        radians. Both say their ``kind`` and whether an axis wraps
        (``circular``, ``periodic_axes``); a polar environment's arena,
        ``env.arena``, is the record they are derived from.
        """
        return self._geometry.coordinates

    @property
    def _boundary_bins_cached(self) -> NDArray[np.intp]:
        def build() -> NDArray[np.intp]:
            return boundary_bins(
                self._binning,
                periodic_axes=self._geometry.coordinates.periodic_axes,
            )

        return self._cache.get(BOUNDARY_BINS, build)

    @classmethod
    @keyword_hint(
        "distance_range", "angle_range", "distance_bin_size", "angle_bin_size"
    )
    def from_polar_egocentric(
        cls,
        *,
        distance_range: tuple[float, float],
        angle_range: tuple[float, float],
        distance_bin_size: float,
        angle_bin_size: float,
        circular_angle: bool | None = None,
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
        circular_angle : bool or None, default=None
            Whether to join the first and last angle bins at each distance
            ring. None wraps a full-circle range (2π radians) and leaves a
            partial range open. True requires a full circle; False leaves
            the angular axis open even for a full-circle range.
        connect_diagonal_neighbors : bool, default=True
            Whether to connect diagonally adjacent (distance, angle) bins.
        name : str, default=""
            Optional name for the environment.
        units : str or None, default=None
            Length unit of the distance axis (e.g. ``"cm"``); angles are
            always radians.
        frame : str or None, default=None
            Coordinate frame label.

        Returns
        -------
        Environment
            A polar environment. Column 0 is distance in physical units;
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
        return _create_polar_environment(
            cls,
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
    def component_keys(self) -> tuple[str, ...]:
        """Stable combined component keys, or an empty tuple for a native environment."""
        return composition_component_keys(self._binning)

    @property
    def bin_sources(self) -> tuple[tuple[str, int], ...] | None:
        """Original component key and local bin ID for each bin, or None when native.

        Subsets and clips retain original source IDs; split cells may repeat an ID.
        """
        return composition_bin_sources(self._binning)

    def get_bridge_attributes(self) -> pd.DataFrame:
        """Return an independent table of retained automatic navigation bridges.

        Rows identify component/local endpoints, current bin IDs, path length
        and geometric conductance. Native environments return an empty table.
        Conductance units are length ** (measure_dim - 2).
        """
        return _get_bridge_attributes(self._binning)

    def get_blocked_edges(self) -> pd.DataFrame:
        """Return the neighbor pairs that walls close, and the wall closing each.

        A wall along the edge between two bins closes that pair: paths,
        distances, smoothing and gradients do not cross it. The pair stays
        recorded, so this table shows which bins each wall separates.

        Returns
        -------
        pd.DataFrame, shape (n_closed_pairs, 4)
            One row per closed pair, in the binning's pair order:

            - ``'source'``, ``'target'``: bin IDs of the pair.
            - ``'distance'``: navigation distance the pair would have.
            - ``'wall'``: name of the wall covering most of the shared edge
              (``arena.wall_names``), or None for a pair no current wall
              closes, such as one passed closed to :meth:`from_bins`.

            Empty when nothing is closed. A wall that covers only part of an
            edge leaves the pair open with a narrower face, so it is not
            listed.

        See Also
        --------
        get_edge_attributes : The open pairs, ``edge_pairs``.

        Examples
        --------
        >>> import numpy as np
        >>> from shapely.geometry import LineString, box
        >>> from neurospatial import Environment
        >>> from neurospatial.arena import OpenArena
        >>> arena = OpenArena(
        ...     box(0, 0, 4, 2), walls={"divider": LineString([(2, 0), (2, 2)])}
        ... )
        >>> env = Environment.from_grid_mask(
        ...     np.ones((4, 2), dtype=bool),
        ...     grid_edges=(np.arange(5.0), np.arange(3.0)),
        ...     arena=arena,
        ... )
        >>> env.get_blocked_edges()  # the two neighbors across x = 2 and both diagonals
           source  target  distance     wall
        0       2       4  1.000000  divider
        1       2       5  1.414214  divider
        2       3       4  1.414214  divider
        3       3       5  1.000000  divider
        """
        return _blocked_edges(self._binning, self.arena)

    def bin_at(self, positions: NDArray[np.float64]) -> NDArray[np.intp]:
        """Map N-dimensional points to bins using geometric containment.

        This method uses the layout-specific geometric logic to determine which
        bin contains each point. This is different from nearest-neighbor mapping
        and respects the actual geometry of the bins (grid cells, hexagons, etc.).

        For nearest-neighbor mapping with caching and tie-breaking, see
        :meth:`nearest_bin`.

        Parameters
        ----------
        positions : NDArray[np.float64], shape (n_points, n_dims)
            Points to map, such as the animal's positions, in the
            environment's coordinates. On an egocentric polar environment
            each row is ``(distance, angle)``, the angle in radians; an angle
            a whole number of turns outside the angle range names the same
            direction and finds the same bin.

        Returns
        -------
        NDArray[np.intp], shape (n_points,)
            An array of active bin indices (0 to `n_active_bins - 1`).
            A value of -1 indicates that the corresponding point did not map
            to any active bin: it is outside the environment, or a coordinate
            is NaN (a missing sample; see Missing data in the class Notes).

        Raises
        ------
        ValueError
            If ``positions`` has a different dimensionality than the
            environment (e.g. 3-D points for a 2-D environment).
        NotImplementedError
            On a track built without ``track_width``: its centerline does not
            say how far from the track a position is still on it.

        See Also
        --------
        nearest_bin : Nearest-neighbor mapping with KDTree caching

        Notes
        -----
        **Geometric Containment vs Nearest-Neighbor:**

        - `bin_at()` (this method): Uses layout-specific geometric containment
          to determine which bin actually contains the point. For grids, this
          checks which grid cell the point falls into. For hexagons, which
          hexagon contains it. This is exact but may be slower for large batches.

        - `nearest_bin()`: Uses KDTree to find the bin whose center is
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
        return self._locate(positions, "Environment.bin_at")

    def nearest_bin(
        self,
        positions: NDArray[np.float64],
        *,
        tie_break: TieBreakStrategy = "lowest_index",
        return_dist: bool = False,
        max_distance: float | None = None,
        max_distance_factor: float | None = None,
    ) -> NDArray[np.intp] | tuple[NDArray[np.intp], NDArray[np.float64]]:
        """Find the nearest bin center for every point, by physical distance.

        Parameters
        ----------
        positions : ndarray, shape (n_points, n_dims)
            Coordinates to compare with bin centers, such as the animal's
            positions, in the environment's stored coordinates:
            ``(distance, angle)`` rows on an egocentric polar environment.
        tie_break : {"lowest_index", "closest_center"}, default="lowest_index"
            For a point equidistant from several centers: "lowest_index"
            returns the tied bin with the smallest ID, so boundary points are
            reproducible; "closest_center" returns whichever tied center the
            KD-tree finds first, not necessarily the lowest ID.
        return_dist : bool, default=False
            Also return the distances to the selected centers, in the
            environment's length units.
        max_distance : float or None, optional
            Maximum distance, in length units; rejected points return bin
            ID -1.
        max_distance_factor : float or None, optional
            Maximum distance as a multiple of ``bin_spacing``. Cannot be
            combined with max_distance. Without either limit every finite
            point gets its nearest bin, however far away; use bin_at() or
            contains() to ask whether a point is inside the environment.

        Returns
        -------
        ndarray or tuple of ndarray
            Bin IDs, shape (n_points,). When return_dist=True, also return
            distances with infinity for rejected points. A point with a NaN
            coordinate is a missing sample: bin -1 and distance NaN. A point
            with an infinite coordinate is rejected: bin -1 and distance
            infinity. See Missing data in the class Notes.

        Raises
        ------
        ValueError
            If distance limits or tie-breaking settings are invalid, or query
            points have the wrong coordinate width.

        Notes
        -----
        This compares Euclidean distance and reuses an environment-owned
        KD-tree. It does not test containment; use bin_at() for occupancy and
        trajectory assignment. On an egocentric polar environment the bin
        centers and the points are first placed in the animal's Cartesian
        plane, ``(distance * cos(angle), distance * sin(angle))`` (x ahead,
        y to the left), so the distance is the physical distance between the
        two places, angles a whole turn apart coincide, and a point near the
        animal can be nearest to any bin of the innermost ring.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> surround = Environment.from_polar_egocentric(
        ...     distance_range=(0.0, 40.0),
        ...     angle_range=(-np.pi, np.pi),
        ...     distance_bin_size=10.0,
        ...     angle_bin_size=np.pi / 4,
        ... )
        >>> bins, dist = surround.nearest_bin(
        ...     np.array([[15.0, 0.4], [15.0, 0.4 + 2 * np.pi]]), return_dist=True
        ... )
        >>> bool(bins[0] == bins[1])
        True
        >>> center_distance, center_angle = surround.bin_centers[bins[0]]
        >>> expected = np.hypot(
        ...     15 * np.cos(0.4) - center_distance * np.cos(center_angle),
        ...     15 * np.sin(0.4) - center_distance * np.sin(center_angle),
        ... )
        >>> bool(np.isclose(dist[0], expected))
        True
        """
        return nearest_bin(
            self._cache,
            self._binning,
            positions,
            coordinates=self._geometry.coordinates,
            tie_break=tie_break,
            return_dist=return_dist,
            max_distance=max_distance,
            max_distance_factor=max_distance_factor,
        )

    @cartesian_only(
        why="A KD-tree over (distance, angle) rows would add centimeters to "
        "radians and treat angles a turn apart as far apart, so its neighbors "
        "would disagree with nearest_bin, which measures physical distance.",
        fix="use env.nearest_bin, which finds the nearest bin by physical "
        "distance on a polar environment",
        example="env.nearest_bin(positions)",
    )
    def bin_center_tree(self: Environment[CartesianSystem]) -> cKDTree:
        """Return the KD-tree over ``bin_centers``, the one ``nearest_bin`` searches.

        The tree is built on first use and kept in the analysis cache, shared
        with relabelled copies and released by ``clear_cache()``. Each call
        returns an independent copy of it (about a millisecond for 200,000
        bins), so changing the copy's arrays cannot change this environment
        or its lookups. Use it for queries that ``nearest_bin`` does not
        answer, such as the ``k`` nearest bin centers with their distances.
        Tree indices are bin indices.

        Returns
        -------
        scipy.spatial.cKDTree
            Tree over ``bin_centers``, shape ``(n_bins, n_dims)``.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> env = Environment.from_samples(
        ...     np.array([[0.0, 0.0], [3.0, 0.0]]), bin_size=1.0
        ... )
        >>> distances, bins = env.bin_center_tree().query([[0.4, 0.5]], k=2)
        >>> bins.tolist()
        [[0, 1]]
        >>> env.bin_center_tree() is env.bin_center_tree()
        False
        """
        return copy.deepcopy(center_tree(self._cache, self._binning))

    def contains(self, positions: NDArray[np.float64]) -> NDArray[np.bool_]:
        """Check if N-dimensional continuous points fall within any active bin.

        Parameters
        ----------
        positions : NDArray[np.float64], shape (n_points, n_dims)
            Points to check, such as the animal's positions, in the
            environment's coordinates: ``(distance, angle)`` rows on an
            egocentric polar environment, as for :meth:`bin_at`.

        Returns
        -------
        NDArray[np.bool_], shape (n_points,)
            A boolean array where `True` indicates the corresponding point
            maps to an active bin, and `False` indicates it does not. A point
            with a NaN coordinate is a missing sample and reads `False` (see
            Missing data in the class Notes).

        Raises
        ------
        ValueError
            If ``positions`` has a different dimensionality than the
            environment (e.g. 3-D points for a 2-D environment).
        NotImplementedError
            On a track built without ``track_width``.

        Notes
        -----
        Tracks require a declared width (``track_width=``): a centerline and
        topology alone do not establish which off-track points are inside.
        Without one, ``contains``, ``bin_at`` and every other position lookup
        raise NotImplementedError with the code to rebuild the track. Every
        layout reuses bin lookup and checks for the -1 sentinel.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> x, y = np.meshgrid(np.arange(0.5, 10.0), np.arange(0.5, 10.0))
        >>> env = Environment.from_samples(np.c_[x.ravel(), y.ravel()], bin_size=2.0)
        >>> points = np.array([[5.0, 5.0], [15.0, 15.0]])
        >>> env.contains(points)  # the second point is outside the arena
        array([ True, False])

        """
        indices = self._locate(positions, "Environment.contains")
        return np.asarray(indices != -1, dtype=np.bool_)

    @keyword_hint(
        former=("point1", "point2", "edge_weight"),
        translate=former_targets(2),
        why=TARGET_WHY,
    )
    def distance_between(
        self,
        *,
        bins: Sequence[int] | NDArray[np.integer] | None = None,
        points: ArrayLike | None = None,
        edge_weight: str | None = "distance",
    ) -> float:
        """Geodesic distance between two bins or two points.

        The distance is the length of the shortest path through the
        connectivity graph, so it goes around walls and between maze arms the
        way the animal can. Give the two ends either as bin indices or as
        coordinates; points are first mapped to the bins that contain them.

        Parameters
        ----------
        bins : sequence of int, shape (2,), optional
            Start and end bin indices, ``[start, end]``.
        points : array-like, shape (2, n_dims), optional
            Start and end coordinates, one row each, in the same coordinates
            and units as ``bin_centers``: (distance, angle) on an egocentric
            polar environment.
        edge_weight : str or None, default="distance"
            Edge attribute summed along the path. ``"distance"`` gives the
            path length in the environment's units; None counts edges (hops).

        Returns
        -------
        float
            The path length, or ``np.inf`` when the two bins are in
            disconnected parts of the graph.

        Raises
        ------
        TypeError
            If neither or both of ``bins`` and ``points`` are given, or
            ``bins`` holds non-integers.
        ValueError
            If there are not exactly two targets, or a point is outside the
            environment or has NaN coordinates.
        BinIndexOutOfRangeError
            If a bin index is outside ``[0, n_bins)``.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> x, y = np.meshgrid(np.arange(0.0, 10.0), np.arange(0.0, 10.0))
        >>> env = Environment.from_samples(np.c_[x.ravel(), y.ravel()], bin_size=2.0)
        >>> path_cm = env.distance_between(points=[[1.0, 1.0], [9.0, 1.0]])
        >>> bool(path_cm > 0)
        True
        >>> start, end = env.bin_at(np.array([[1.0, 1.0], [9.0, 1.0]]))
        >>> env.distance_between(bins=[start, end]) == path_cm
        True

        """
        start, end = self._target_bins("distance_between()", bins, points, count=2)
        return geodesic_distance_between_points(
            self.connectivity, int(start), int(end), weight=edge_weight
        )

    def to_linear(
        self: Environment[CartesianSystem],
        positions: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Convert N-D positions to 1D linear coordinates.

        For 1D linearized environments (created with `from_graph()`), this
        converts N-D positions (e.g., 2D x,y coordinates) to their corresponding
        1D linear coordinates along the linearized track.

        Parameters
        ----------
        positions : NDArray[np.floating], shape (n_samples, n_dims)
            N-D position array in the original coordinate space.
            For 2D environments, shape is (n_samples, 2).

        Returns
        -------
        NDArray[np.floating], shape (n_samples,)
            1D linear coordinates corresponding to the input positions.

        Raises
        ------
        TypeError
            If this environment is not a linearized track
            (``env.is_linearized_track`` is False).

        Notes
        -----
        - **Only works for track environments** built with `Environment.from_graph()`,
          `Environment.linear_track()` or `Environment.maze()`
        - For N-D environments, use `bin_at()` to map positions to bins instead
        - Linear coordinates start at 0 and increase monotonically along the track
        - Every position is projected onto the nearest point of the centerline,
          however far off the track it is, with or without ``track_width``.
          To exclude reflections and tracking jumps, build the track with
          ``track_width`` and keep the samples with ``bin_at(positions) >= 0``.

        See Also
        --------
        linear_to_nd : Convert 1D linear coordinates back to N-D positions
        Environment.from_graph : Create 1D linearized environment
        is_linearized_track : Check if environment supports linearization

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> # A 100 cm linear track along the x axis
        >>> env = Environment.linear_track(
        ...     endpoints=[[0.0, 0.0], [100.0, 0.0]], bin_size=5.0, units="cm"
        ... )
        >>> # Halfway along the track, 3 cm off the centerline
        >>> env.to_linear(np.array([[50.0, 3.0]]))
        array([50.])

        """
        return _to_linear(self._binning, positions)

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
            For 2D environments, shape is (n_samples, 2). A linear position
            on no track segment has a row of NaN: one below 0, past the end
            of the track, inside a gap between segments (``edge_spacing``),
            or not finite. Both ends of each segment are on the track.

        Raises
        ------
        TypeError
            If this environment is not a linearized track
            (``env.is_linearized_track`` is False).

        Notes
        -----
        - **Only works for track environments** built with `Environment.from_graph()`,
          `Environment.linear_track()` or `Environment.maze()`
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
        >>> from neurospatial import Environment
        >>> # A 100 cm linear track along the x axis
        >>> env = Environment.linear_track(
        ...     endpoints=[[0.0, 0.0], [100.0, 0.0]], bin_size=5.0, units="cm"
        ... )
        >>> # 50 cm along the track is the middle of the centerline
        >>> env.linear_to_nd(np.array([50.0]))
        array([[50.,  0.]])
        >>> # 150 cm is past the end of the track, so it has no position
        >>> env.linear_to_nd(np.array([100.0, 150.0]))
        array([[100.,   0.],
               [ nan,  nan]])

        """
        return _linear_to_nd(self._binning, linear_position)

    def occupancy(
        self,
        times: NDArray[np.float64],
        positions: NDArray[np.float64],
        *,
        speed: NDArray[np.float64] | None = None,
        min_speed: float | None = None,
        max_gap: float | None = DEFAULT_MAX_GAP,
        epochs: Any = None,
        break_before: ArrayLike | None = None,
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
            Position coordinates matching environment dimensions, located as
            :meth:`bin_at` does: on an egocentric polar environment, each
            sample's ``(distance, angle)`` of the object or place the bins
            describe, relative to the animal. A sample with a NaN
            coordinate is missing and treated as a deleted row (see Missing
            data in the class Notes). With the default
            ``time_allocation='start'``, an interval adds occupancy only when
            both of its valid samples are inside the environment: one that
            starts or ends outside adds none. This is the rule
            ``compute_spatial_rate`` applies to its spike counts and occupancy.
            With ``time_allocation='linear'`` an interval with an end outside
            the environment adds only the time its straight path spends inside.
        speed : NDArray[np.float64], shape (n_samples,), optional
            Instantaneous speed at each sample. If provided with min_speed,
            samples below threshold are excluded from occupancy calculation.
        min_speed : float, optional
            Minimum speed threshold in physical units per second. Requires
            speed parameter. Samples with speed < min_speed are excluded.
        max_gap : float or None, default=1.0
            Longest stretch of missing tracking, in seconds, to bridge: the time
            between consecutive valid samples (a dropped frame and a NaN row are
            the same missing sample). Longer gaps are excluded as tracking losses;
            ``None`` bridges every gap. The default is calibrated for typical video
            and DeepLabCut tracking; choose a value longer than routine dropped or
            low-confidence frames and shorter than real tracking losses or pickups.
        epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
            Restrict the analysis to these half-open [start, stop) windows (seconds,
            same clock as ``times``). The interval between two consecutive samples
            counts only if it lies entirely inside one window. ``None`` (default)
            means unrestricted.
        break_before : ArrayLike of bool, shape (n_samples,), optional
            ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
            ``k`` (for example where a new recording starts), whatever the elapsed
            time. Entry 0 is ignored.
        bandwidth : float, optional
            If provided, apply diffusion kernel smoothing with this bandwidth
            (in physical units). Uses mode='transition' to preserve total mass.
            Smoothing preserves total occupancy time.
        time_allocation : {'start', 'linear'}, default='start'
            Method for allocating time intervals across bins:

            - 'start': Assign entire Δt to starting bin (fast, works on all layouts).
            - 'linear': Split Δt proportionally across bins traversed by
              straight-line path (grid layouts with rectangular cells; not
              polar environments, where a straight path is not straight in
              ``(distance, angle)``).
        return_seconds : bool, default=True
            If True, return time in seconds (time-weighted occupancy).
            If False, return sample counts (unweighted, counts number of
            intervals starting in each bin; with ``time_allocation='linear'``
            each interval is one count split like its duration). Use True for
            rate calculations (e.g., firing rates), False for presence/absence
            analyses.

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
            If linear allocation lacks exact supported cell geometry, or the
            environment is a track built without ``track_width``.
        CartesianOnlyError
            If ``time_allocation='linear'`` on a polar environment.
        ValueError
            If ``break_before`` is not a Boolean array with one entry per sample.

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
          Supported for rectangular grids in any dimension and exact planar
          polygon cells. It retains entry, exit and re-entry intervals and
          drops the share of the path in holes, inactive cells or outside the
          environment without renormalizing: an interval half inside adds
          half its duration in seconds, or half a count with
          ``return_seconds=False``, whichever way it travels. Planar polygon
          allocation refuses proper crossings of a known physical wall.
          Observation validity is independent of whether the starting point
          is in a bin.

        **Mass conservation**: The sum of the returned occupancy array equals
        the total valid time:

        .. math::
            \\sum_i \\text{occupancy}[i] = \\sum_{\\text{valid } k} (t_{k+1} - t_k)

        where valid intervals satisfy:
        - Δt ≤ max_gap (if max_gap is not None)
        - speed[k] ≥ min_speed (if min_speed is not None)
        - no break is declared before sample k + 1 (``break_before[k + 1]``)
        - positions[k] and positions[k + 1] are inside the environment
          (``time_allocation='start'``)

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

        If a second recording starts at sample 2 on the same clock, declare
        the break; the interval across the join then adds no time:

        >>> new_recording = np.array([False, False, True, False])
        >>> occ = env.occupancy(
        ...     times, positions, max_gap=None, break_before=new_recording
        ... )
        >>> float(occ.sum())
        2.0

        """
        return _occupancy(
            times,
            positions,
            binning=self._binning,
            arena=self._arena,
            coordinates=self._geometry.coordinates,
            cache=self._cache,
            speed=speed,
            min_speed=min_speed,
            max_gap=max_gap,
            epochs=epochs,
            break_before=break_before,
            bandwidth=bandwidth,
            time_allocation=time_allocation,
            return_seconds=return_seconds,
        )

    def bin_sequence(
        self,
        times: NDArray[np.float64],
        positions: NDArray[np.float64],
        *,
        dedup: bool = True,
        outside_value: int | None = -1,
        max_gap: float | None = DEFAULT_MAX_GAP,
        epochs: Any = None,
        break_before: ArrayLike | None = None,
    ) -> NDArray[np.intp]:
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
            Position coordinates matching environment dimensions, located as
            :meth:`bin_at` does: ``(distance, angle)`` rows on an egocentric
            polar environment.
        dedup : bool, default=True
            If True, collapse consecutive repeats: [A,A,A,B] → [A,B].
            If False, return a bin index for every retained sample.
        outside_value : int or None, default=-1
            Bin index for samples outside environment bounds or with a NaN
            coordinate (missing; see Missing data in the class Notes).
            - If -1 (default), outside samples are marked with -1.
            - If None, outside samples are dropped from the sequence entirely.
        max_gap : float or None, default=1.0
            Longest stretch of missing tracking, in seconds, to bridge: the time
            between consecutive valid samples (a dropped frame and a NaN row are
            the same missing sample). Longer gaps are excluded as tracking losses;
            ``None`` bridges every gap. The default is calibrated for typical video
            and DeepLabCut tracking; choose a value longer than routine dropped or
            low-confidence frames and shorter than real tracking losses or pickups.
        epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
            Restrict the analysis to these half-open [start, stop) windows (seconds,
            same clock as ``times``). The interval between two consecutive samples
            counts only if it lies entirely inside one window. ``None`` (default)
            means unrestricted.
        break_before : ArrayLike of bool, shape (n_samples,), optional
            ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
            ``k`` (for example where a new recording starts), whatever the elapsed
            time. Entry 0 is ignored.

        Returns
        -------
        bins : NDArray[np.intp], shape (n_sequences,)
            Bin index at each retained time point (or deduplicated sequence).
            Values are in range [0, n_bins-1] for valid bins, or -1 for
            outside samples (when outside_value=-1).

        Raises
        ------
        ValueError
            If times and positions have different lengths, if positions
            have wrong number of dimensions, or if timestamps are not
            monotonically increasing (non-decreasing).
        ValueError
            If ``break_before`` is not a Boolean array with one entry per sample.

        See Also
        --------
        bin_sequence_with_runs : Same mapping plus run-length boundaries.
        occupancy : Compute time spent in each bin.
        transitions : Build empirical transition matrix from trajectory.

        Notes
        -----
        Each run of samples with gaps no longer than ``max_gap`` (inside ``epochs``,
        not crossing a ``break_before``) is analyzed as a separate recording;
        no segment spans a pause.

        Samples that touch no valid interval are dropped, including singleton
        inputs. Deduplication never merges samples across an invalid interval.
        With dedup=False, one bin is returned per retained sample.

        Timestamps must be monotonically increasing (non-decreasing).
        Sort your data by time before calling this method if needed.

        Examples
        --------
        A 10 x 10 cm arena with 2 cm bins. The animal runs along y = 1 and
        one tracking sample jumps outside the arena:

        >>> import numpy as np
        >>> x, y = np.meshgrid(np.arange(0.0, 10.0), np.arange(0.0, 10.0))
        >>> env = Environment.from_samples(np.c_[x.ravel(), y.ravel()], bin_size=2.0)
        >>> times = np.arange(6) * 0.1  # seconds
        >>> x_cm = np.array([1.0, 1.5, 3.0, 3.5, 20.0, 5.0])  # 20 cm is outside
        >>> positions = np.c_[x_cm, np.ones(6)]
        >>> env.bin_sequence(times, positions)  # repeated bins collapse
        array([ 7, 13, -1, 19])
        >>> env.bin_sequence(times, positions, dedup=False)  # one bin per sample
        array([ 7,  7, 13, 13, -1, 19])
        >>> env.bin_sequence(times, positions, outside_value=None)  # drop outside
        array([ 7, 13, 19])

        """
        result = _bin_sequence(
            times,
            positions,
            binning=self._binning,
            arena=self._arena,
            cache=self._cache,
            call="Environment.bin_sequence",
            dedup=dedup,
            outside_value=outside_value,
            gap_splits_runs=False,
            max_gap=max_gap,
            epochs=epochs,
            break_before=break_before,
        )
        return result.bins

    def bin_sequence_with_runs(
        self,
        times: NDArray[np.float64],
        positions: NDArray[np.float64],
        *,
        outside_value: int | None = -1,
        max_gap: float | None = DEFAULT_MAX_GAP,
        epochs: Any = None,
        break_before: ArrayLike | None = None,
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
            in the same bin. A run's length is its extent in the original
            rows, including missing (NaN) rows bridged inside it, so
            ``run_lengths.sum()`` can exceed the number of in-env samples.
        max_gap : float or None, default=1.0
            Longest stretch of missing tracking, in seconds, to bridge: the time
            between consecutive valid samples (a dropped frame and a NaN row are
            the same missing sample). Longer gaps are excluded as tracking losses;
            ``None`` bridges every gap. The default is calibrated for typical video
            and DeepLabCut tracking; choose a value longer than routine dropped or
            low-confidence frames and shorter than real tracking losses or pickups.
        epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
            Restrict the analysis to these half-open [start, stop) windows (seconds,
            same clock as ``times``). The interval between two consecutive samples
            counts only if it lies entirely inside one window. ``None`` (default)
            means unrestricted.
        break_before : ArrayLike of bool, shape (n_samples,), optional
            ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
            ``k`` (for example where a new recording starts), whatever the elapsed
            time. Entry 0 is ignored.

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
        Each run of samples with gaps no longer than ``max_gap`` (inside ``epochs``,
        not crossing a ``break_before``) is analyzed as a separate recording;
        no segment spans a pause.

        Samples that touch no valid interval are dropped, including singleton
        inputs. Invalid intervals split runs even if the bins match. Run
        starts and lengths refer to the original input indices, so durations
        computed from them exclude pauses.

        Examples
        --------
        >>> import numpy as np
        >>> x, y = np.meshgrid(np.arange(0.0, 10.0), np.arange(0.0, 10.0))
        >>> env = Environment.from_samples(np.c_[x.ravel(), y.ravel()], bin_size=2.0)
        >>> times = np.arange(6) * 0.1  # seconds
        >>> x_cm = np.array([1.0, 1.5, 3.0, 3.5, 20.0, 5.0])  # 20 cm is outside
        >>> positions = np.c_[x_cm, np.ones(6)]
        >>> result = env.bin_sequence_with_runs(times, positions)
        >>> result.bins
        array([ 7, 13, -1, 19])
        >>> result.run_lengths  # samples in each run
        array([2, 2, 1, 1])
        >>> # Time from the first to the last sample of each run, in seconds
        >>> last = result.run_starts + result.run_lengths - 1
        >>> np.round(times[last] - times[result.run_starts], 2)
        array([0.1, 0.1, 0. , 0. ])
        """
        return _bin_sequence(
            times,
            positions,
            binning=self._binning,
            arena=self._arena,
            cache=self._cache,
            call="Environment.bin_sequence_with_runs",
            dedup=True,
            outside_value=outside_value,
            gap_splits_runs=True,
            max_gap=max_gap,
            epochs=epochs,
            break_before=break_before,
        )

    @cartesian_only(
        why="Its nearest mode reads the nearest bin center and its linear mode "
        "interpolates along Cartesian grid axes; neither is the cell that "
        "contains a (distance, angle) row.",
        fix="read the field in the bin that contains each row with env.bin_at; "
        "a row with bin -1 is outside and gets NaN, not field[-1]",
        example="bins = env.bin_at(positions); "
        "np.where(bins >= 0, field[bins], np.nan)",
    )
    def interpolate(
        self: Environment[CartesianSystem],
        field: NDArray[np.float64],
        positions: NDArray[np.float64],
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
            NaN marks a bin with no value (for example an unvisited bin of a
            rate map). Must not contain Inf values.
        positions : NDArray[np.float64], shape (n_points, n_dims)
            Query points in environment coordinates. Must be a 2-D array where
            each row is a point with dimensionality matching the environment.
            A point with a NaN coordinate is a missing sample and reads NaN.
        mode : {'nearest', 'linear'}, default='nearest'
            Interpolation mode:

            - 'nearest': Use value of nearest bin center (all layouts).
              Points outside environment bounds return NaN.
            - 'linear': Bilinear (2D) or trilinear (3D) interpolation for
              grids with rectangular cells: ``from_samples``, ``from_grid_mask``,
              ``from_pixel_mask``, ``from_polygon``, and grids read from NWB.
              Rotated, reflected and sheared 2-D/3-D grids use their source axes;
              queries remain in world coordinates.
              Inactive cells and NaN field values return NaN. Between the
              grid's edge and its outermost bin centers the edge value is
              held along that axis. Points outside the environment return
              NaN.

        Returns
        -------
        values : NDArray[np.float64], shape (n_points,)
            Interpolated field values. Points outside the environment, points
            with NaN coordinates and points that need a NaN field value read
            NaN (see Missing data in the class Notes).

        Raises
        ------
        ValueError
            If field has wrong shape, wrong dimensionality, contains Inf,
            positions have wrong dimensionality, mode is invalid, or dimensions
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
        Positive interpolation weight on a resolved inactive corner, or on a
        bin whose field value is NaN, returns NaN.

        **Outside handling**: Points outside the environment bounds return NaN
        in both modes. This prevents extrapolation errors.

        Examples
        --------
        A field that rises linearly across the arena, f(x, y) = 2x + 3y,
        read at two points inside the arena and one outside it:

        >>> import numpy as np
        >>> x, y = np.meshgrid(np.arange(0.0, 10.0), np.arange(0.0, 10.0))
        >>> env = Environment.from_samples(np.c_[x.ravel(), y.ravel()], bin_size=2.0)
        >>> plane_field = 2 * env.bin_centers[:, 0] + 3 * env.bin_centers[:, 1]
        >>> query_points = np.array([[5.0, 5.0], [7.5, 3.2], [50.0, 5.0]])

        Linear interpolation (grids only) recovers the plane exactly:

        >>> env.interpolate(plane_field, query_points, mode="linear")
        array([25. , 24.6,  nan])

        Nearest-neighbor interpolation (all layouts) returns the value of the
        closest bin center:

        >>> env.interpolate(plane_field, query_points, mode="nearest").round(2)
        array([27.5, 25.5,  nan])

        To read a rate map at every tracked position, pass the positions
        array, ``env.interpolate(rate_map, positions, mode="linear")``.

        """
        return _interpolate(
            field,
            positions,
            binning=self._binning,
            arena=self._arena,
            cache=self._cache,
            mode=mode,
        )

    @cartesian_only(
        why="It merges blocks of cells on a Cartesian grid.",
        fix="build a coarser polar environment over the same ranges with larger "
        "distance_bin_size or angle_bin_size",
        example="Environment.from_polar_egocentric(distance_range=env.dimension_ranges[0], "
        "angle_range=env.dimension_ranges[1], distance_bin_size=20.0, "
        "angle_bin_size=np.pi / 2)",
    )
    def rebin(
        self: Environment[CartesianSystem],
        factor: int | tuple[int, ...] | None = None,
        *,
        bin_size: float | tuple[float, ...] | None = None,
        name: str | None = None,
    ) -> Environment[CartesianSystem]:
        """Re-bin an arena at a new size, or coarsen a grid by integer factor.

        Creates a new environment with coarser spatial resolution by reducing
        the number of bins. This method only modifies the grid geometry and
        connectivity; it does not aggregate any field values.

        Pass exactly one of ``factor`` or ``bin_size``. Either mode covers the
        bins this environment has, within its arena: rebinning a ``subset`` or
        ``clip`` keeps that selection rather than restoring the space it left
        out. The new-size mode uses stored rectangular-lattice data and
        preserves the grid frame and lattice anchor, independently of
        descriptive provenance.

        Parameters
        ----------
        factor : int or tuple of int, optional
            Coarsening factor per dimension. If int, applied uniformly to all
            dimensions. If tuple, must match the number of dimensions.
            Each factor must be a positive integer. Only supported for the
            existing axis-aligned uniform rectangular cells; retains truncation behavior.
        bin_size : float or tuple of float, optional
            New positive physical bin lengths along the retained grid axes,
            in environment units. May refine or coarsen a rectangular grid.
            Needs a known arena and a supported stored GridBinningSpec.
            Center selection can leave portions of the arena without a bin;
            use ``clip`` separately when exact clipped cells are needed.
        name : str, optional
            Name of the new environment; defaults to this environment's name.

        Returns
        -------
        coarse_env : Environment
            New environment at the requested resolution. With ``factor``, the grid shape is
            ``original_shape // factor`` in each dimension. A new bin is kept
            when it overlaps one of this environment's bins and its center
            lies inside the arena.

        Raises
        ------
        NotImplementedError
            If cells are not a uniform rectangular grid, or factor rebinning
            would replace custom navigation distances or connections.
        ValueError
            If factor is not positive or if factor is too large for grid shape.
            Also raised when coarse cells cannot represent an existing wall.

        See Also
        --------
        bin_mapping_from : Carry counts, occupancy and labels to the new bins
            exactly.
        smooth : Apply diffusion kernel smoothing to fields.
        subset : Extract spatial subset of environment.

        Notes
        -----
        **Geometry only**: This method only coarsens the grid structure. To
        aggregate field values (occupancy, spike counts, etc.) from the original
        grid to the coarsened grid, use ``coarse.bin_mapping_from(env)``: it
        pairs each original bin with the new bin that wholly contains it, and
        its ``sum`` adds counts and occupancy, as the last example below
        shows. Pool rates from counts and occupancy with its ``rates``, never
        by averaging, and re-estimate smoothed maps on the new bins.

        **Factor mode**: Integer coarsening requires uniform rectangular cells
        with native grid connections and straight center-to-center navigation
        distances, allowing connections removed by physical walls. Custom
        paths cannot be reconstructed from cell geometry; use ``subset`` to
        retain them or construct the target bins and paths explicitly.

        **Non-divisible dimensions**: If the grid shape is not evenly divisible
        by the factor in any dimension, the grid is truncated to the largest
        multiple of the factor. A warning is issued in this case.

        **Connectivity**: The connectivity graph is rebuilt for the coarsened
        grid using the stored diagonal-connection choice, or actual edges when
        an older grid has no stored choice. Physical walls still block
        connections and must align with complete coarse faces. The arena
        outline does not: neighbors that meet only outside a polygonal arena,
        such as bins on both sides of a barrier narrower than a bin, stay
        connected, and a ``UserWarning`` names them. Declare the barrier's
        edges as internal walls to remove those connections.

        **Bin centers**: New bin centers are computed from the coarsened grid
        edges as midpoints between edge positions.

        **Active bins**: A new cell must overlap this environment's bins: with
        ``factor``, its block of fine cells holds at least one bin; with
        ``bin_size``, it shares a positive area or volume with a bin (a bin
        that only touches its boundary does not count, whatever rounding a
        rotation adds).
        It must also lie in the arena: with ``factor``, its center lies in the
        known arena, including on its boundary, as ``contains`` does; with
        ``bin_size``, strictly inside a polygonal arena, or contained by
        another known arena. A factor of 2 therefore merges every fine bin of
        a grid mask, and rebinning keeps both a subset's selection and the
        arena's holes. When original bins (inside the
        truncated extent) are left without a new bin, a ``UserWarning`` gives
        their count and share of the area, since positions in them map to -1.

        **Metadata preservation**: The name, arena, units, frame and named
        regions are preserved. Region coordinates and metadata remain intact
        even when truncation removes the bins near a region. New-size
        re-binning supports rectangular
        grids from samples, masks, images and polygon grids, including their
        derived grid frames. Other families and older records without a stored
        specification refuse this mode. It never transfers fields or models.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> # A 100 x 100 cm arena of 10 cm cells
        >>> grid_edges = (np.arange(11) * 10.0,) * 2
        >>> env = Environment.from_grid_mask(np.ones((10, 10), dtype=bool), grid_edges)
        >>> env.grid_shape
        (10, 10)
        >>>
        >>> # Coarsen by factor 2: 20 cm cells
        >>> coarse = env.rebin(factor=2)
        >>> coarse.grid_shape
        (5, 5)
        >>>
        >>> # Anisotropic coarsening with tuple (10 // 2 = 5, 10 // 5 = 2)
        >>> coarse_aniso = env.rebin(factor=(2, 5))
        >>> coarse_aniso.grid_shape
        (5, 2)

        A factor that does not divide the grid shape truncates the grid and warns:

        >>> import warnings
        >>> with warnings.catch_warnings(record=True) as caught:
        ...     warnings.simplefilter("always")
        ...     coarse_3 = env.rebin(factor=3)
        >>> coarse_3.grid_shape
        (3, 3)
        >>> print(caught[0].message)
        Grid shape (10, 10) is not evenly divisible by factor (3, 3). Grid will be truncated to (9, 9) before coarsening.

        Aggregate a field to the coarsened grid:

        >>> occupancy = np.random.default_rng(0).random(env.n_bins) * 100
        >>> mapping = coarse.bin_mapping_from(env)  # each fine bin's coarse bin
        >>> coarse_occupancy = mapping.sum(occupancy)
        >>> # Every fine bin lies in a coarse bin, so the total is preserved
        >>> bool(np.isclose(occupancy.sum(), coarse_occupancy.sum()))
        True

        """
        derivation, check = _rebin(
            self._binning,
            self._arena,
            self.connectivity,
            self._provenance,
            factor=factor,
            bin_size=bin_size,
        )
        rebinned = publish_derived(self, derivation, name=name)
        warn_rebin_losses(self._binning, self._arena, rebinned, check)
        return rebinned

    @cartesian_only(
        why="It rebuilds the kept cells, their regions and their boundary in "
        "Cartesian coordinates.",
        fix="keep the full polar environment and select bins with a boolean mask",
        example="near = env.bin_centers[:, 0] < 20.0; field[near]",
    )
    @keyword_hint(
        former=("bins",),
        translate=lambda values: (
            "pass the boolean array of bins to keep as mask=",
            {"mask": values["bins"]},
        ),
        why="Why: the boolean array of bins to keep is passed by keyword as mask=.",
    )
    def subset(
        self: Environment[CartesianSystem],
        *,
        mask: NDArray[np.bool_] | None = None,
        region_names: Sequence[str] | None = None,
        polygon: shapely.Polygon | None = None,
        invert: bool = False,
        name: str | None = None,
    ) -> Environment[CartesianSystem]:
        """Create new environment containing subset of bins.

        Extracts a subgraph from the environment containing only the selected
        bins. Node indices are renumbered to be contiguous [0, n'-1]. Bin
        sizes, retained faces, cell shapes and track coordinates are
        preserved, and so are the name, units, frame, arena and named regions.
        Regions may extend beyond the retained bins. Points in discarded bins
        are outside the subset and map to ``-1``.

        Parameters
        ----------
        mask : NDArray[np.bool_], shape (n_bins,), optional
            Boolean mask of bins to keep. True = keep, False = discard.
        region_names : Sequence[str], optional
            Keep the bins these named regions cover, by the rule of
            :meth:`bins_in_region`: the bin containing a point region, and
            the bins whose centers lie in a polygon region. Regions must
            exist in self.regions.
        polygon : shapely.Polygon, optional
            Keep bins whose centers lie in this polygon, by the rule of
            :meth:`bins_in_region` (a center exactly on the edge is kept when
            nudging it north-east moves it inside). Only works for 2D
            environments.
        invert : bool, default=False
            If True, invert the selection mask (select complement).
        name : str, optional
            Name of the new environment; defaults to this environment's name.

        Returns
        -------
        sub_env : Environment
            New environment with selected bins renumbered to [0, n'-1].
            Connectivity is the induced subgraph. Name, units, frame, arena
            and regions are preserved.

        Raises
        ------
        ValueError
            If none or multiple selection parameters provided, if mask has
            wrong shape/dtype, if region names don't exist, if selection is empty.

        Notes
        -----
        Exactly one of {mask, region_names, polygon} must be provided.

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
        bin_mapping_from : Pair the kept bins with this environment's bins.
        rebin : Coarsen grid resolution (grid-only).
        components : Find connected components.

        Examples
        --------
        A 100 x 100 cm arena with 10 cm bins and a goal zone in one corner:

        >>> import numpy as np
        >>> from shapely.geometry import box
        >>> from neurospatial import Environment
        >>> from neurospatial.regions import Regions
        >>> x, y = np.meshgrid(np.linspace(0, 100, 41), np.linspace(0, 100, 41))
        >>> env = Environment.from_samples(
        ...     np.c_[x.ravel(), y.ravel()], bin_size=10.0, units="cm"
        ... )
        >>> regions = Regions()
        >>> _ = regions.add("goal", polygon=box(70, 70, 100, 100))
        >>> env = env.with_regions(regions)
        >>> env.n_bins
        121

        Keep the bins inside the goal zone, or everything except it. Bin
        centers lie on the zone's edges: those on its south and west edges are
        kept and those on its north and east edges are not, so the 30 x 30 cm
        zone keeps 3 x 3 bins of 10 cm:

        >>> env.subset(region_names=["goal"]).n_bins
        9
        >>> env.subset(region_names=["goal"], invert=True).n_bins
        112

        Crop to a polygon, or select bins with a boolean mask:

        >>> env.subset(polygon=box(0, 0, 50, 50)).n_bins
        25
        >>> left_half = env.bin_centers[:, 0] < 50
        >>> env.subset(mask=left_half).n_bins
        55

        """
        derivation = _subset(
            self._binning,
            self._arena,
            self.regions,
            self._cache,
            mask=mask,
            region_names=region_names,
            polygon=polygon,
            invert=invert,
        )
        return publish_derived(self, derivation, name=name)

    @cartesian_only(
        why="It intersects the cells with a polygon in Cartesian coordinates.",
        fix="keep the full polar environment and select bins with a boolean mask",
        example="ahead = np.abs(env.bin_centers[:, 1]) < np.pi / 2; field[ahead]",
    )
    def clip(
        self: Environment[CartesianSystem],
        polygon: shapely.Polygon | shapely.MultiPolygon,
        *,
        name: str | None = None,
    ) -> Environment[CartesianSystem]:
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
            Exact positive-area cells, with the parent's units, frame, arena
            and named regions. Disconnected pieces get separate bins;
            ``clip_mapping.parent_bin_indices`` identifies their input bins.
            A grid drops the cells whose centers lie outside a slanted or
            curved outline and gives a position in their arena part to the
            nearest neighboring bin; that bin's clipped cell takes that
            arena, so the clip keeps every position inside ``polygon`` that
            the parent gives a bin, in a piece of that bin.
            Rates, counts and fitted models are not transferred. Numerical
            operations require certified finite-volume geometry. The analysis
            crop preserves the parent arena and does not assert new walls.

        Raises
        ------
        ValueError
            If geometry is invalid, unavailable, unsupported, or has no
            positive-area intersection. Track, polar and 3-D cells are not
            Cartesian planar area cells.

        See Also
        --------
        bin_mapping_from : Pair the cells the clip kept whole with this
            environment's bins, and list the cells it cut.
        subset : Keep whole bins without cutting them.
        """
        return publish_derived(
            self, _clip(self._binning, self._arena, polygon), name=name
        )

    def bin_mapping_from(self, parent: Environment) -> BinMapping:
        """Pair this environment's bins with the bins of ``parent`` they contain.

        Use it to carry values computed on ``parent`` to this environment
        exactly: compare a map on a subset with the full map, or coarsen
        counts and occupancy after ``rebin``. The pairs come from the bins'
        geometry, so the mapping works on environments read from files, and
        only pairs that can be proven are reported:

        - a bin that is the same cell as a parent bin (same center, measure,
          cell shape and boundary ownership): a ``subset``, the cells a
          ``clip`` did not cut, an identical rebuild;
        - parent cells that lie wholly inside one cell of a coarser grid drawn
          on the same lattice: ``rebin(factor)``, and ``rebin(bin_size=...)``
          when the new cell edges fall on the old ones. Edges count as equal
          within a billionth of a cell width, the tolerance rebinning uses.

        A parent cell divided between bins (cut by a clip) is reported in
        ``split_parent_bins`` and never paired: its count cannot be divided
        without knowing where in the cell each observation fell.

        Parameters
        ----------
        parent : Environment
            The environment the values were computed on, usually the one this
            environment was made from.

        Returns
        -------
        BinMapping
            Rows of ``(bin in parent, bin in this environment)`` in ``pairs``,
            the parent bins left out or split, the bins of this environment
            covered by no parent bin, and methods that carry values: ``sum``
            for counts and occupancy, ``rates`` for rates pooled from counts
            and occupancy, ``category`` for labels.

        Raises
        ------
        IncompatibleEnvironmentError
            If the two environments differ in coordinate system, units or
            frame.
        ValueError
            If no bin can be paired exactly, such as for a finer grid, a
            rotated or offset lattice, or an independently built
            environment.
        TypeError
            If ``parent`` is not an Environment.

        See Also
        --------
        subset, clip, rebin : Make the environments this maps between.
        neurospatial.ops.map_probabilities : Approximate transfer by nearest
            bin center, for environments with no exact correspondence.

        Notes
        -----
        ``apply_transform`` needs no mapping: transforms keep bin IDs, so bin
        ``i`` of the transformed environment is bin ``i`` of the original.

        Two bins are the same cell when they have identical centers, sizes
        and measured dimensions; the same cell shape where both environments
        record cells (grid boxes, triangles, hexagons or polygon cells); for
        polygon cells, the same bin owning each boundary point, whatever the
        bins' numbers, so a renumbered copy of a clipped arena pairs with the
        original, but the pieces of a cell cut in several must keep their
        relative order, because a point on the line between two pieces
        belongs to the earlier one; and, for tracks, the same track segments
        and spacing and the same linear edges of each bin. Explicit point
        bins, or a track paired with bins of another kind, are compared by
        center and measure only.

        Same-cell pairs meet the check ``compare_spatial_rates(bin_mapping=...)``
        applies, so ``pairs`` of a subset, a clip or an identical rebuild can be
        passed to it. Polygon cells kept whole by a clip have the parent's
        polygon exactly and own the same boundary points; their centers and
        areas are recomputed from the polygon and may differ from the
        parent's by rounding only.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> grid_edges = (np.arange(0.0, 41.0, 10.0),) * 2  # 40 x 40 cm, 10 cm cells
        >>> env = Environment.from_grid_mask(np.ones((4, 4), dtype=bool), grid_edges)
        >>> left = env.subset(mask=env.bin_centers[:, 0] < 20)
        >>> mapping = left.bin_mapping_from(env)
        >>> mapping.kind, len(mapping.pairs)
        ('one_to_one', 8)
        >>> bool(np.array_equal(mapping.pairs[:, 1], np.arange(left.n_bins)))
        True

        Merge 2 x 2 blocks of bins and add up the time spent in them:

        >>> coarse = env.rebin(2)
        >>> occupancy = np.arange(16.0)
        >>> coarse.bin_mapping_from(env).sum(occupancy).tolist()
        [10.0, 18.0, 42.0, 50.0]
        """
        return _bin_mapping_from(self, parent)

    @cartesian_only(
        why="An affine map moves Cartesian positions; it has no meaning for "
        "(distance, angle) pairs.",
        fix="transform the allocentric positions, then compute egocentric "
        "distance and angle from them",
        example="shifted = positions + np.array([10.0, 0.0]); np.stack("
        "[neurospatial.ops.compute_egocentric_distance(shifted, headings, "
        "object_positions), neurospatial.ops.compute_egocentric_bearing(shifted, "
        "headings, object_positions)], axis=-1)",
    )
    def apply_transform(
        self: Environment[CartesianSystem],
        transform: AffineND,
        *,
        name: str | None = None,
        frame: str | None = None,
        units: str | None | _Unset = UNSET,
    ) -> Environment[CartesianSystem]:
        """Apply affine transformation to environment, returning a new instance.

        Transforms the environment's bin centers, connectivity graph, and regions
        using an affine transformation. The transformation must match the
        environment's dimensionality.

        Translation and similarity transforms retain finite-volume smoothing
        and differential operators. Cartesian 2-D/3-D grids retain source axes
        for linear interpolation and movement allocation under rotations,
        reflections and shear. Planar grids draw exact cell corners. New numerical
        transforms need stored face normals, certified source flux distances and
        perpendicular transformed center links (normalized residual at most 1e-12). Smoothing
        refuses when those right angles are lost; exact cells remain available.
        A general affine fitted to noisy landmarks absorbs the noise as a small
        stretch or shear, so smoothing refuses; fit landmarks with
        ``estimate_transform(..., kind="similarity")`` to keep it.
        Triangle vertices and deformed hexagon polygons retain their exact mapped
        shapes. Tracks accept translation and similarity only.

        Parameters
        ----------
        transform : AffineND
            Affine transformation to apply. Its ``n_dims`` must match the
            environment's dimensionality.

            Create transforms using factory functions like `translate()`,
            `scale_2d()`, `from_rotation_matrix()`, or compose using `@` operator.
        name : str, optional
            Name for the transformed environment; defaults to this
            environment's name.
        frame : str, optional
            Coordinate frame label of the transformed positions, for example
            ``"session2"``. The default ``None`` leaves the frame unlabeled,
            because the transformed coordinates are no longer in this
            environment's frame.
        units : str or None, optional
            Length unit of the transformed coordinates, such as ``"cm"`` after
            a pixel-to-centimeter calibration. Omission keeps this
            environment's units, which is allowed only when the transform
            keeps lengths (a translation, rotation or reflection) or the
            environment has no units. A transform that scales or shears
            lengths may be a unit conversion or a stretch in the same units,
            so it needs ``units=``: the new unit, or ``env.units`` to keep it.

        Returns
        -------
        transformed_env : Environment
            New Environment instance with transformed coordinates. The transformation
            is applied to:

            - `bin_centers`: All bin positions are transformed
            - `connectivity`: Node 'pos' attributes updated, edge distances/vectors recomputed
            - `regions`: Point and polygon regions are transformed
            - `metadata`: Name preserved; units set from ``units`` (kept when
              omitted); frame set from ``frame``

            Bin IDs and edge ordering are retained. Cell/face measures and
            available numerical capabilities follow the transformed geometry.
            Bin-aligned values and fitted models are not rescaled or refitted.

        Raises
        ------
        ValueError
            If transform dimensionality doesn't match environment dimensionality
            or a track receives a non-similarity transform. Rebuild such a track from transformed nodes with
            ``Environment.from_graph``. Also if the transform changes lengths,
            the environment has units, and ``units`` is omitted.

        See Also
        --------
        estimate_transform : Estimate transformation from point correspondences
        neurospatial.ops.transforms.AffineND : N-D affine transformation class
        neurospatial.ops.transforms.translate : Create translation transform
        neurospatial.ops.transforms.scale_2d : Create scaling transform
        neurospatial.ops.transforms.from_rotation_matrix : Create rotation transform

        Notes
        -----
        **Pure function**: This method does not modify the source environment;
        it returns a new Environment instance.

        **Transformation order**: When composing transforms, use the `@` operator.
        ``T1 @ T2`` applies ``T2`` first, then ``T1``, so
        ``rotation @ translation`` translates first and then rotates (see the
        composed-transform example below).

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
        >>> rng = np.random.default_rng(0)
        >>> data = rng.random((200, 2)) * 100
        >>> env = Environment.from_samples(data, bin_size=5.0, name="session1")
        >>> # Translate by (10, 20) cm
        >>> transform = translate(10, 20)
        >>> env_shifted = env.apply_transform(transform, name="session1_aligned")
        >>> # Bin centers are translated
        >>> np.allclose(env_shifted.bin_centers, env.bin_centers + [10, 20])
        True

        Rotation (2D):

        >>> from neurospatial.ops.transforms import AffineND
        >>> # 45-degree rotation
        >>> angle = np.pi / 4
        >>> R = np.array(
        ...     [
        ...         [np.cos(angle), -np.sin(angle), 0],
        ...         [np.sin(angle), np.cos(angle), 0],
        ...         [0, 0, 1],
        ...     ]
        ... )
        >>> transform = AffineND(R)
        >>> env_rotated = env.apply_transform(transform)
        >>> # Distances between bin centers are preserved under rotation
        >>> def steps(centers):
        ...     return np.linalg.norm(np.diff(centers, axis=0), axis=1)
        >>> np.allclose(steps(env_rotated.bin_centers), steps(env.bin_centers))
        True

        Composed transforms (scale, then rotate, then translate):

        >>> from neurospatial.ops.transforms import scale_2d
        >>> T = translate(50, 50) @ AffineND(R) @ scale_2d(1.2)
        >>> env_aligned = env.apply_transform(T, name="aligned")
        >>> np.allclose(env_aligned.bin_centers, T(env.bin_centers))
        True
        >>> # Areas scale by the square of the 1.2 scale factor
        >>> np.allclose(env_aligned.bin_sizes, env.bin_sizes * 1.2**2)
        True

        Cross-session alignment using landmarks:

        >>> from neurospatial.ops.transforms import estimate_transform
        >>> # Session 1 landmarks (e.g., arena corners)
        >>> landmarks_s1 = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], float)
        >>> # Session 2 landmarks (same physical locations, different coordinates)
        >>> landmarks_s2 = np.array([[5, 10], [95, 15], [90, 105], [0, 100]], float)
        >>> # Estimate rigid transform (rotation + translation)
        >>> T = estimate_transform(landmarks_s1, landmarks_s2, kind="rigid")
        >>> # A rigid fit cannot absorb the stretch between sessions; check the
        >>> # landmark residuals (cm) before trusting the alignment
        >>> np.linalg.norm(T(landmarks_s1) - landmarks_s2, axis=1).round(1)
        array([7., 7., 7., 7.])
        >>> # Transform the session 1 environment to session 2 coordinates
        >>> env_s1_aligned = env.apply_transform(
        ...     T, name="session1_in_s2_coords", frame="session2"
        ... )
        >>> env_s1_aligned.frame
        'session2'

        3D transformation:

        >>> from scipy.spatial.transform import Rotation
        >>> from neurospatial.ops.transforms import (
        ...     from_rotation_matrix,
        ...     translate_3d,
        ... )
        >>> # Create 3D environment
        >>> data_3d = rng.random((500, 3)) * 20
        >>> env_3d = Environment.from_samples(data_3d, bin_size=5.0)
        >>> # Rotate 45 degrees around z-axis and translate
        >>> R_3d = Rotation.from_euler("z", 45, degrees=True).as_matrix()
        >>> rotation = from_rotation_matrix(R_3d)
        >>> translation = translate_3d(10, 20, 30)
        >>> T_3d = translation @ rotation
        >>> env_3d_transformed = env_3d.apply_transform(T_3d)
        >>> # A rotation and translation preserve distances between bins
        >>> def steps_3d(centers):
        ...     return np.linalg.norm(np.diff(centers, axis=0), axis=1)
        >>> np.allclose(
        ...     steps_3d(env_3d_transformed.bin_centers), steps_3d(env_3d.bin_centers)
        ... )
        True

        With regions (regions are automatically transformed):

        >>> from neurospatial.regions import Regions
        >>> regions = Regions(env.regions.values())
        >>> _ = regions.add("goal", point=[80, 90])
        >>> env = env.with_regions(regions)
        >>> env_transformed = env.apply_transform(translate(10, 10))
        >>> # Region is transformed along with environment
        >>> goal_region = env_transformed.regions["goal"]
        >>> np.allclose(goal_region.data, [90, 100])
        True

        """
        derivation = _apply_transform(
            transform,
            binning=self._binning,
            regions=self.regions,
            landmarks=self.landmarks,
            arena=self._arena,
            cache=self._cache,
            units=self.units if isinstance(units, _Unset) else units,
            old_units=self.units,
        )
        scale = derivation.length_scale
        if (
            isinstance(units, _Unset)
            and self.units is not None
            and (scale is None or abs(scale - 1.0) > 1e-9)
        ):
            change = (
                f"scales lengths by {scale:.6g}"
                if scale is not None
                else "stretches or shears lengths"
            )
            raise ValueError(
                _format_error(
                    f"apply_transform() {change}, so lengths may no longer be "
                    f"in {self.units!r}.",
                    why="Why: a transform that changes lengths is often a unit "
                    "conversion, such as pixels to centimeters, but can also "
                    "stretch an arena in the same units; only the caller knows "
                    "which.",
                    fix="pass the unit of the transformed coordinates, as in "
                    "env.apply_transform(transform, units='cm'), or "
                    f"units={self.units!r} to keep it",
                )
            )
        return publish_derived(self, derivation, name=name, frame=frame, units=units)

    @classmethod
    @keyword_hint()
    def combine(
        cls,
        components: Mapping[str, Environment],
        *,
        auto_bridge: bool = True,
        max_mnn_distance: float | None = None,
        bridge_conductances: Mapping[tuple[tuple[str, int], tuple[str, int]], float]
        | None = None,
        name: str = "",
    ) -> Environment[CartesianSystem]:
        """Combine disjoint components into an ordinary owned environment.

        Parameters
        ----------
        components : mapping of str to Environment
            Ordered, uniquely keyed Cartesian components in matching units and frame.
        auto_bridge : bool, default=True
            Infer mutual-nearest-neighbor navigation bridges. These do not carry diffusion.
        max_mnn_distance : float or None, optional
            Inclusive maximum inferred bridge length, in coordinate units.
        bridge_conductances : mapping or None, optional
            Explicit geometric conductance for inferred bridges, keyed by pairs
            of ``(component_key, local_bin_id)`` endpoints. Units are
            ``length ** (measure_dim - 2)``, matching finite-volume diffusion.
        name : str, default=""
            Name of the resulting environment.

        Returns
        -------
        Environment
            Owned bins in component input order. Regions are qualified by key;
            ``bin_sources`` and ``get_bridge_attributes()`` expose correspondence.

        Raises
        ------
        ValueError
            If coordinates disagree, components overlap, or support cannot be certified.
        """
        return _combine(
            Environment,
            components,
            auto_bridge=auto_bridge,
            max_mnn_distance=max_mnn_distance,
            bridge_conductances=bridge_conductances,
            name=name,
        )

    @classmethod
    @keyword_hint()
    def from_bins(
        cls,
        binning: Binning,
        *,
        name: str = "",
        units: str | None = None,
        frame: str | None = None,
        regions: Mapping[str, Region] | None = None,
        arena: Arena | None = None,
    ) -> Environment[CartesianSystem]:
        """Construct an environment from explicit Cartesian binning data.

        Parameters
        ----------
        binning : neurospatial.binning.Binning
            Bin centers, measures and ordered neighbor pairs, with an explicit
            supported locator. KnownFaces enables finite-volume operations;
            MissingFaces explicitly refuses them. Shapes are required for exact
            clipping. Centers and sizes alone cannot establish these capabilities.
            An object from another package with an installed snapshot adapter
            is converted to a Binning first.
        name : str, default=""
            Environment name.
        units, frame : str or None, optional
            Coordinate declarations; they do not convert coordinates.
        regions : mapping of str to Region, or None, optional
            Named regions to snapshot at construction.
        arena : Arena or None, optional
            Continuous domain. Omission infers only what the supplied geometry
            establishes, without inventing physical walls. Bins located by
            ``NearestBinLocator`` without ``max_distance`` and with no known
            domain refuse position lookups; give the locator a radius in
            length units (``math.inf`` for unlimited nearest lookup) or pass
            the arena that bounds the bins.

        Returns
        -------
        Environment
            Owned environment preserving bin IDs and supplied pair order.
            ``edge_pairs`` defines gradient and flux order; NetworkX iteration
            may use a different order.
        """
        return _from_bins(
            cls,
            binning,
            name=name,
            units=units,
            frame=frame,
            regions=regions,
            arena=arena,
        )

    @classmethod
    @keyword_hint("bin_size")
    def from_samples(
        cls,
        positions: NDArray[np.float64],
        *,
        bin_size: float | Sequence[float],
        name: str = "",
        units: str | None = None,
        frame: str | None = None,
        regions: Mapping[str, Region] | None = None,
        arena: Arena | None = None,
        wall_fit: Literal["snap", "split"] = "snap",
        layout: LayoutType | str = LayoutType.REGULAR_GRID,
        dimension_ranges: Sequence[tuple[float, float]] | None = None,
        infer_active_bins: bool = True,
        bin_count_threshold: int = 0,
        dilate: bool = False,
        fill_holes: bool = False,
        close_gaps: bool = False,
        add_boundary_bins: bool = False,
        connect_diagonal_neighbors: bool = True,
    ) -> Environment[CartesianSystem]:
        """Create an Environment by binning (discretizing) `positions` into a layout grid.

        The common path (`positions` + `bin_size`) needs none of the advanced /
        cleanup knobs listed under *Other Parameters*; reach for those (e.g.
        ``dilate``, ``fill_holes``, ``close_gaps``) only when sparse or noisy
        sampling leaves holes, gaps, or a ragged boundary in the active-bin mask.

        For common apparatus, start from a preset instead:

        - :meth:`open_field` -- an open arena from tracked positions; the same
          binning with interior holes filled, so a briefly unvisited patch in
          the middle of the arena stays in the analysis.
        - :meth:`linear_track` -- a straight or piecewise-linear track from its
          endpoints, linearized for 1-D place fields.
        - :meth:`maze` -- a plus, T or W maze from the positions of its
          junctions and arm ends.

        Parameters
        ----------
        positions : array, shape (n_samples, n_dims)
            Coordinates of sample points used to infer which bins are "active."
            A row with a NaN or infinite coordinate is a missing sample and is
            left out.
        bin_size : float or sequence of floats
            Size of each bin in the same units as `positions` coordinates.
            For RegularGrid: length of each square bin side (or per-dimension if sequence).
            For Hexagonal: hexagon width (flat-to-flat distance across hexagon).
            If your data is in centimeters, bin_size=5.0 creates 5cm bins.
            Grid cells are exactly this size; when the extent is not a whole
            number of cells, the grid reaches equally past both ends of it.
        name : str, default ""
            Optional name for the resulting Environment.
        units : str or None, default None
            Spatial units for the environment coordinates (e.g. ``"cm"``,
            ``"m"``). If provided, sets ``env.units`` on the returned
            Environment.
        frame : str or None, default None
            Coordinate frame identifier for the environment (e.g. a session
            label). If provided, sets ``env.frame`` on the returned Environment.
        regions : mapping of str to Region, optional
            Named regions to snapshot with the constructed environment.
        arena : Arena, optional
            Explicit continuous domain. Without one, infer an estimated domain
            from the active cells; an unvisited patch is not a declared wall.
            With a known outline, the grid tiles the arena's bounds unless
            `dimension_ranges` is given.
        wall_fit : {'snap', 'split'}, default='snap'
            What to do with a bin that a wall of `arena` divides. ``'snap'``
            keeps every bin and moves the wall onto the bin edges between the
            bins it divides, warning how far it moved (at most about half a
            bin); every analysis then works as on any grid. ``'split'`` cuts
            each such bin along the wall into one exact polygon bin per side,
            keeping the measured wall. Smoothing and other finite-volume
            operations refuse a split bin next to a wall's end or a diagonal
            wall, so use it with ``method='binned'`` rates or wall-distance
            measures. Bins no wall divides are the same either way, except
            that, along a slanted or curved outline, ``'split'`` makes the
            bins exact polygons that cover the arena: a bin sticking out past
            the outline is clipped to it, and a bin next to a grid cell dropped
            there takes the part of that cell inside the arena, whose
            positions it already held.
        layout : LayoutType | str, default LayoutType.REGULAR_GRID
            ``"RegularGrid"`` or ``"Hexagonal"``, as a LayoutType member or a
            case-insensitive string. For "Hexagonal", `bin_size` is the
            hexagon width. Other layouts are built with
            :func:`neurospatial.layout.create_layout` and
            :meth:`from_layout`.
        dimension_ranges : sequence of (float, float), optional
            ``[(min, max), ...]`` extent to tile on each axis, in the units of
            `positions`. By default the bounds of `arena` when it has a known
            outline, otherwise the extent of the samples plus half a bin on
            each side. Cells of exactly `bin_size` cover it; when it is a whole
            number of cells, cell edges start at its ends, so walls on that
            lattice fall on cell faces (a 100 cm box with 5 cm bins), and
            otherwise the grid reaches equally past both ends.
        infer_active_bins : bool, default True
            If True, only bins containing ≥ `bin_count_threshold` samples are
            "active." If False, every bin in the extent is active, and the
            mask options (`bin_count_threshold`, `dilate`, `fill_holes`,
            `close_gaps`) must keep their defaults.
        connect_diagonal_neighbors : bool, default True
            Regular grid only. If True, connect grid bins diagonally when
            building connectivity. A hexagonal layout accepts only the
            default.

        Other Parameters
        ----------------
        bin_count_threshold : int, default 0
            Minimum number of data points required for a bin to be considered
            "active." Only with ``infer_active_bins=True``.
        dilate : bool, default False
            Regular grid only, with ``infer_active_bins=True``. If True, apply
            morphological dilation to the active-bin mask.
        fill_holes : bool, default False
            Regular grid only, with ``infer_active_bins=True``. If True, fill
            holes in the active-bin mask.
        close_gaps : bool, default False
            Regular grid only, with ``infer_active_bins=True``. If True, close
            small gaps between active bins. `close_gaps`, `fill_holes` and
            `dilate` act only on grids with two or more dimensions, in that
            order (closing, then hole filling, then dilation); with 1-D
            positions they have no effect.
        add_boundary_bins : bool, default False
            Regular grid only. If True, add peripheral bins around the
            bounding region of samples.

        Returns
        -------
        env : Environment
            A new Environment binned from the samples.

        Raises
        ------
        ValueError
            If `positions` is not 2D or contains invalid coordinates; if
            `layout` is neither "RegularGrid" nor "Hexagonal"; if a
            regular-grid-only option (`dilate`, `fill_holes`, `close_gaps`,
            `add_boundary_bins`, `connect_diagonal_neighbors`) differs from
            its default with ``layout="Hexagonal"``; or if a mask option
            differs from its default with ``infer_active_bins=False``. The
            default values themselves are accepted, so a parameter set can
            switch layouts without editing these options.

        See Also
        --------
        open_field : Open-arena preset of this factory (fills interior holes).
        linear_track : Linear or piecewise-linear track preset.
        maze : Plus, T or W maze preset.
        from_polygon : Create environment with polygon-defined boundary.
        from_grid_mask : Create environment from pre-defined boolean mask.
        from_pixel_mask : Create environment from binary image mask.
        from_graph : Create 1D linearized track environment.
        from_layout : Create environment from a built layout engine.

        Examples
        --------
        Create a simple 2D environment from position data:

        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> # Simulate animal position data in a 100x100 cm arena
        >>> rng = np.random.default_rng(42)  # For reproducible examples
        >>> positions = rng.random((1000, 2)) * 100  # cm
        >>> # Create environment with 5cm x 5cm bins
        >>> env = Environment.from_samples(
        ...     positions=positions,
        ...     bin_size=5.0,
        ...     name="arena",  # bin_size in cm
        ... )
        >>> env.n_dims
        2
        >>> env.n_bins > 0
        True

        Create environment with morphological operations to clean up the active region:

        >>> env = Environment.from_samples(
        ...     positions=positions,
        ...     bin_size=5.0,  # 5cm bins
        ...     bin_count_threshold=5,  # Require 5 samples per bin
        ...     dilate=True,  # Expand active region
        ...     fill_holes=True,  # Fill interior holes
        ... )

        Create a hexagonal grid environment:

        >>> from neurospatial.layout import LayoutType
        >>> env = Environment.from_samples(
        ...     positions=positions,
        ...     layout=LayoutType.HEXAGONAL,  # or layout="Hexagonal"
        ...     bin_size=5.0,  # 5cm hexagon width
        ... )

        Common Pitfalls
        ---------------
        1. **bin_size too large**: If bin_size is too large relative to your data
           range, you may end up with very few bins or no active bins at all.
           For example, if your data spans 0-100 cm and you use bin_size=200.0,
           you'll only get 1 bin. Try reducing bin_size to create more spatial
           resolution (e.g., bin_size=5.0 for 5cm bins).

        2. **bin_count_threshold too high**: Setting bin_count_threshold higher
           than the number of samples per bin will result in no active bins.
           If you have sparse data with only a few samples per location, try
           reducing bin_count_threshold to 0 or 1, or use morphological operations
           to expand the active region.

        3. **Mismatched units**: Ensure bin_size and positions use the same
           units. If your data is in centimeters, bin_size should also be in
           centimeters. Mixing units (e.g., data in meters, bin_size in centimeters)
           will result in incorrect spatial binning. For example, if your data spans
           0-1 meters (100 cm) and you set bin_size=5.0 thinking it's centimeters,
           you'll get only 1 bin instead of 20 bins.

        4. **Missing morphological operations with sparse data**: If your data is
           sparse (animal didn't visit all locations uniformly), the active region
           may have holes or gaps. Enable dilate=True, fill_holes=True, or
           close_gaps=True to create a more continuous active region. These
           operations are particularly useful for connecting isolated bins or
           filling small unvisited areas within explored regions.

        """
        return _from_samples(
            cls,
            positions,
            bin_size=bin_size,
            name=name,
            units=units,
            frame=frame,
            regions=regions,
            arena=arena,
            wall_fit=wall_fit,
            layout=layout,
            dimension_ranges=dimension_ranges,
            infer_active_bins=infer_active_bins,
            bin_count_threshold=bin_count_threshold,
            dilate=dilate,
            fill_holes=fill_holes,
            close_gaps=close_gaps,
            add_boundary_bins=add_boundary_bins,
            connect_diagonal_neighbors=connect_diagonal_neighbors,
        )

    @classmethod
    @keyword_hint("bin_size")
    def open_field(
        cls,
        positions: NDArray[np.float64],
        *,
        bin_size: float | Sequence[float],
        name: str = "",
        units: str | None = None,
        frame: str | None = None,
        regions: Mapping[str, Region] | None = None,
        arena: Arena | None = None,
        wall_fit: Literal["snap", "split"] = "snap",
        layout: LayoutType | str = LayoutType.REGULAR_GRID,
        dimension_ranges: Sequence[tuple[float, float]] | None = None,
        infer_active_bins: bool = True,
        bin_count_threshold: int = 0,
        dilate: bool = False,
        fill_holes: bool = True,
        close_gaps: bool = False,
        add_boundary_bins: bool = False,
        connect_diagonal_neighbors: bool = True,
    ) -> Environment[CartesianSystem]:
        """Create an open-arena Environment from sampled positions.

        This is the only positions-based preset. It is :meth:`from_samples`
        with ``fill_holes=True`` by default on a regular grid: interior gaps
        left by uneven sampling are filled, so a freely explored 2D arena
        reads as one filled region rather than a speckled mask. Every other
        option means the same as in :meth:`from_samples`.

        Parameters
        ----------
        positions : array, shape (n_samples, n_dims)
            Coordinates of sample points used to infer which bins are "active."
            A row with a NaN or infinite coordinate is a missing sample and is
            left out.
        bin_size : float or sequence of floats
            Size of each bin in the same units as `positions` coordinates.
        name : str, default ""
            Optional name for the resulting Environment.
        units : str or None, default None
            Spatial units for the environment coordinates (e.g. ``"cm"``).
        frame : str or None, default None
            Coordinate frame identifier.
        regions : mapping of str to Region, optional
            Named regions to snapshot with the constructed environment.
        arena : Arena, optional
            Explicit continuous domain. Without one, infer an estimated domain
            from the active cells.
        wall_fit : {'snap', 'split'}, default='snap'
            What to do with a bin that a wall of `arena` divides. ``'snap'``
            keeps every bin and moves the wall onto the bin edges between the
            bins it divides, warning how far it moved (at most about half a
            bin); every analysis then works as on any grid. ``'split'`` cuts
            each such bin along the wall into one exact polygon bin per side,
            keeping the measured wall. Smoothing and other finite-volume
            operations refuse a split bin next to a wall's end or a diagonal
            wall, so use it with ``method='binned'`` rates or wall-distance
            measures. Bins no wall divides are the same either way, except
            that, along a slanted or curved outline, ``'split'`` makes the
            bins exact polygons that cover the arena: a bin sticking out past
            the outline is clipped to it, and a bin next to a grid cell dropped
            there takes the part of that cell inside the arena, whose
            positions it already held.
        layout : LayoutType | str, default LayoutType.REGULAR_GRID
            ``"RegularGrid"`` or ``"Hexagonal"``. See :meth:`from_samples`.
        dimension_ranges : sequence of (float, float), optional
            ``[(min, max), ...]`` extent to tile on each axis. See
            :meth:`from_samples`.
        infer_active_bins : bool, default True
            If True, only bins containing ≥ `bin_count_threshold` samples are
            "active." If False, every bin in the extent is active, and the
            mask options must keep their defaults. See :meth:`from_samples`.
        bin_count_threshold : int, default 0
            Minimum number of data points required for a bin to be "active."
            Only with ``infer_active_bins=True``.
        dilate : bool, default False
            Regular grid only, with ``infer_active_bins=True``. If True, apply
            morphological dilation to the active-bin mask.
        fill_holes : bool, default True
            Regular grid only, with ``infer_active_bins=True``. If True, fill
            holes in the active-bin mask. Pass False to keep the unvisited
            interior bins out.
        close_gaps : bool, default False
            Regular grid only, with ``infer_active_bins=True``. If True, close
            small gaps between active bins.
        add_boundary_bins : bool, default False
            Regular grid only. If True, add peripheral bins around the
            bounding region of samples.
        connect_diagonal_neighbors : bool, default True
            Regular grid only. If True, connect grid bins diagonally when
            building connectivity.

        Returns
        -------
        env : Environment
            A new open-arena Environment binned from the samples.

        Raises
        ------
        ValueError
            If `layout` is neither "RegularGrid" nor "Hexagonal", if a
            regular-grid-only option differs from its default with
            ``layout="Hexagonal"``, or if a mask option differs from its
            default with ``infer_active_bins=False``. Those layouts never
            fill holes, so ``fill_holes=False`` is accepted with them too.

        See Also
        --------
        from_samples : The general factory this preset delegates to.
        linear_track : Linear / piecewise-linear track preset (explicit topology).
        maze : W / plus / T maze preset (explicit topology).

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> rng = np.random.default_rng(0)
        >>> positions = rng.random((1000, 2)) * 100  # cm
        >>> env = Environment.open_field(positions, bin_size=5.0)
        >>> env.is_linearized_track
        False
        """
        return _from_samples(
            cls,
            positions,
            bin_size=bin_size,
            name=name,
            units=units,
            frame=frame,
            regions=regions,
            arena=arena,
            wall_fit=wall_fit,
            layout=layout,
            dimension_ranges=dimension_ranges,
            infer_active_bins=infer_active_bins,
            bin_count_threshold=bin_count_threshold,
            dilate=dilate,
            fill_holes=fill_holes,
            close_gaps=close_gaps,
            add_boundary_bins=add_boundary_bins,
            connect_diagonal_neighbors=connect_diagonal_neighbors,
            factory="open_field",
            defaults=_OPEN_FIELD_OPTION_DEFAULTS,
        )

    @classmethod
    @keyword_hint()
    def linear_track(
        cls,
        *,
        endpoints: Sequence[Sequence[float]] | None = None,
        node_positions: Sequence[Sequence[float]] | None = None,
        bin_size: float,
        edge_spacing: float | Sequence[float] = 0.0,
        track_width: float | None = None,
        name: str = "",
        units: str | None = None,
        frame: str | None = None,
        regions: Mapping[str, Region] | None = None,
    ) -> Environment[CartesianSystem]:
        """Create a 1D linearized track Environment from an explicit topology.

        Unlike :meth:`open_field`, a linear track is defined by its *topology*,
        not by raw positions -- a cloud of (x, y) samples cannot tell apart a
        straight track, an L-shaped track, or an open arena. You must therefore
        supply either ``endpoints`` (a straight two-point track) or
        ``node_positions`` (a piecewise-linear sequence of waypoints). A
        :mod:`networkx` track graph is assembled (nodes carry ``pos``
        attributes, edges join consecutive nodes with a ``distance`` weight)
        and passed to :meth:`from_graph`.

        Parameters
        ----------
        endpoints : sequence of two (x, y) points, optional
            The two ends of a straight track, e.g. ``[(0, 0), (100, 0)]``.
            Mutually exclusive with `node_positions`.
        node_positions : sequence of (x, y) points, optional
            Ordered waypoints of a piecewise-linear track (>= 2 points). Edges
            connect each consecutive pair. Mutually exclusive with `endpoints`.
        bin_size : float
            Length of each bin along the linearized track, in the units of the
            node coordinates.
        edge_spacing : float or sequence of floats, default 0.0
            Spacing inserted between consecutive edges during linearization.
            For a single contiguous track this is typically 0.0.
        track_width : float or None, optional
            Full physical track width. Required for contains(); bin_at()
            remains nearest-track projection when width is unknown.
        name : str, default ""
            Optional name for the resulting Environment.
        units : str or None, default None
            Spatial units for the environment coordinates (e.g. ``"cm"``). If
            provided, sets ``env.units`` on the returned Environment.
        frame : str or None, default None
            Coordinate frame identifier. If provided, sets ``env.frame`` on the
            returned Environment.
        regions : mapping of str to Region, optional
            Named regions to snapshot with the constructed environment.

        Returns
        -------
        Environment
            A 1D linearized-track Environment (``is_linearized_track`` is True).

        Raises
        ------
        ValueError
            If neither `endpoints` nor `node_positions` is given (a linear track
            needs an explicit topology, not raw positions), if both are given,
            or if too few points are supplied.

        See Also
        --------
        from_graph : The general factory this preset delegates to.
        open_field : The only positions-based preset.
        maze : W / plus / T maze preset (explicit topology).

        Examples
        --------
        Straight track from its two endpoints:

        >>> from neurospatial import Environment
        >>> env = Environment.linear_track(endpoints=[(0, 0), (100, 0)], bin_size=5.0)
        >>> env.is_linearized_track
        True
        >>> env.n_bins
        20

        L-shaped (piecewise-linear) track from waypoints:

        >>> env = Environment.linear_track(
        ...     node_positions=[(0, 0), (50, 0), (50, 50)], bin_size=5.0
        ... )
        >>> env.is_linearized_track
        True
        """
        return _linear_track(
            cls,
            endpoints=endpoints,
            node_positions=node_positions,
            bin_size=bin_size,
            edge_spacing=edge_spacing,
            track_width=track_width,
            name=name,
            units=units,
            frame=frame,
            regions=regions,
        )

    @classmethod
    @keyword_hint("kind")
    def maze(
        cls,
        *,
        kind: str,
        track_graph: nx.Graph | None = None,
        node_positions: dict[Any, Sequence[float]] | None = None,
        bin_size: float,
        edge_spacing: float | Sequence[float] = 0.0,
        track_width: float | None = None,
        name: str = "",
        units: str | None = None,
        frame: str | None = None,
        regions: Mapping[str, Region] | None = None,
    ) -> Environment[CartesianSystem]:
        """Create a 1D linearized W / plus / T maze Environment.

        Like :meth:`linear_track`, a maze is defined by its *topology*, not by
        raw positions. You must supply either a ready ``track_graph`` (a
        :mod:`networkx` graph whose nodes carry ``pos`` attributes) or
        ``node_positions`` from which the standard edge topology for `kind` is
        assembled. The graph is then passed to :meth:`from_graph`.

        Standard topologies assembled from `node_positions`. **The topology is
        derived from the ORDER of the `node_positions` entries, not from their
        x/y coordinates** -- this is a coordinate-system-independent contract,
        so it works identically whether y increases up or down and whichever
        direction the arms extend. The required order per `kind` is:

        - ``"plus"`` (5 nodes): ``[center, arm1, arm2, arm3, arm4]``. The first
          node is the center; the remaining four are arm tips. Edges join the
          center to each arm tip (a star). For example::

              {
                  "center": (0, 0),
                  "north": (0, 50),
                  "south": (0, -50),
                  "east": (50, 0),
                  "west": (-50, 0),
              }

        - ``"t"`` (4 nodes): ``[stem_end, junction, arm_left, arm_right]``. The
          stem joins stem_end--junction; the crossbar joins junction--arm_left
          and junction--arm_right. For example::

              {
                  "start": (0, 0),
                  "junction": (0, 50),
                  "left": (-50, 50),
                  "right": (50, 50),
              }

        - ``"w"`` (6 nodes): ``[base_left, base_mid, base_right, arm_left,
          arm_mid, arm_right]`` -- three base nodes followed by three arm tips.
          The base nodes are connected by a horizontal connector
          (base_left--base_mid--base_right); each arm tip connects to the base
          node at the matching position in this order (base_left--arm_left,
          base_mid--arm_mid, base_right--arm_right). For example::

              {
                  "base_left": (0, 0),
                  "base_mid": (50, 0),
                  "base_right": (100, 0),
                  "arm_left": (0, 50),
                  "arm_mid": (50, 50),
                  "arm_right": (100, 50),
              }

        Python dicts preserve insertion order, so the order you write the
        entries in is the order used.

        Parameters
        ----------
        kind : {"w", "plus", "t"}
            Which standard maze topology to build.
        track_graph : networkx.Graph, optional
            A ready track graph with ``pos`` node attributes. Edges join the
            nodes that define the maze; ``distance`` weights are filled in from
            node positions if absent. Mutually exclusive with `node_positions`.
        node_positions : dict, optional
            Mapping of node label -> (x, y) coordinate from which the standard
            `kind` topology is assembled. Mutually exclusive with `track_graph`.
        bin_size : float
            Length of each bin along the linearized track.
        edge_spacing : float or sequence of floats, default 0.0
            Spacing inserted between consecutive edges during linearization.
        track_width : float or None, optional
            Full physical track width. Required for contains(); bin_at()
            remains nearest-track projection when width is unknown.
        name : str, default ""
            Optional name for the resulting Environment.
        units : str or None, default None
            Spatial units for the environment coordinates (e.g. ``"cm"``). If
            provided, sets ``env.units`` on the returned Environment.
        frame : str or None, default None
            Coordinate frame identifier. If provided, sets ``env.frame`` on the
            returned Environment.
        regions : mapping of str to Region, optional
            Named regions to snapshot with the constructed environment.

        Returns
        -------
        Environment
            A 1D linearized-track Environment (``is_linearized_track`` is True).

        Raises
        ------
        ValueError
            If `kind` is not one of ``{"w", "plus", "t"}``; if neither
            `track_graph` nor `node_positions` is given (positions alone cannot
            infer maze topology); if both are given; or if `node_positions` does
            not contain the expected number of nodes for `kind`.

        See Also
        --------
        from_graph : The general factory this preset delegates to.
        linear_track : Linear / piecewise-linear track preset.
        open_field : The only positions-based preset.

        Examples
        --------
        Plus maze from labelled node positions:

        >>> from neurospatial import Environment
        >>> nodes = {
        ...     "center": (0, 0),
        ...     "north": (0, 50),
        ...     "south": (0, -50),
        ...     "east": (50, 0),
        ...     "west": (-50, 0),
        ... }
        >>> env = Environment.maze(kind="plus", node_positions=nodes, bin_size=5.0)
        >>> env.is_linearized_track
        True
        """
        return _maze(
            cls,
            kind=kind,
            track_graph=track_graph,
            node_positions=node_positions,
            bin_size=bin_size,
            edge_spacing=edge_spacing,
            track_width=track_width,
            name=name,
            units=units,
            frame=frame,
            regions=regions,
        )

    @classmethod
    @keyword_hint("edge_spacing", "bin_size")
    def from_graph(
        cls,
        graph: nx.Graph,
        edge_order: list[tuple[Any, Any]],
        *,
        bin_size: float,
        edge_spacing: float | Sequence[float] = 0.0,
        name: str = "",
        units: str | None = None,
        frame: str | None = None,
        regions: Mapping[str, Region] | None = None,
        track_width: float | None = None,
    ) -> Environment[CartesianSystem]:
        """Create an Environment from a user-defined graph structure.

        This method is used for 1D environments where the spatial layout is
        defined by a graph, an ordered list of its edges, and spacing between
        these edges. The track is then linearized and binned.

        Parameters
        ----------
        graph : nx.Graph
            The NetworkX graph defining the track segments. Nodes are expected
            to have a 'pos' attribute for their N-D coordinates.
            ``GraphLayout.build`` numbers ``edge_id`` by ``graph.edges()``
            order; any ``edge_id`` on the input graph is ignored. The
            environment keeps the graph's attributes, so each must be a
            value pickle can store (an attribute holding a function or a lock
            raises ``TypeError``); ``to_file``, ``to_dict`` and NWB export
            also need JSON values: finite numbers, strings, booleans, None
            and lists or dicts of them.
        edge_order : List[Tuple[Any, Any]]
            An ordered list of edge tuples (node1, node2) from `graph` that
            defines the 1D bin ordering.
        bin_size : float
            The length of each bin along the linearized track, in the same units
            as the graph node coordinates. For example, if node positions are in
            centimeters, bin_size=2.0 creates 2cm bins along the track.
        edge_spacing : float or sequence of floats, default 0.0
            The spacing to insert between consecutive edges in `edge_order`
            during linearization, in the same units as the graph node coordinates.
            If a float, applies to all gaps. If a sequence, specifies spacing for
            each gap. The default 0.0 places consecutive edges end to end, as in
            :meth:`linear_track` and :meth:`maze`.
        name : str, optional
            A name for the created environment. Defaults to "".
        units : str, optional
            Unit label for the supplied coordinates; does not scale them.
        frame : str, optional
            Coordinate frame label for the supplied coordinates.
        regions : mapping of str to Region, optional
            Named regions to snapshot with the constructed environment.
        track_width : float, optional
            Physical full width of the track in coordinate units. A positive
            width limits containment to the union of round-ended segment
            corridors. When omitted, the physical footprint stays unknown
            and the existing nearest-track lookup remains available.

        Returns
        -------
        Environment
            A new Environment instance with a `GraphLayout`.

        See Also
        --------
        from_samples : Create environment by binning position data.
        from_layout : Create environment from a built layout engine.

        Examples
        --------
        Linear track. Three nodes laid out left-to-right; one edge per
        segment, all in centimeters:

        >>> import networkx as nx
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>>
        >>> g = nx.Graph()
        >>> g.add_node(0, pos=(0.0,))
        >>> g.add_node(1, pos=(50.0,))
        >>> g.add_node(2, pos=(100.0,))
        >>> g.add_edge(0, 1, distance=50.0)
        >>> g.add_edge(1, 2, distance=50.0)
        >>>
        >>> env = Environment.from_graph(
        ...     graph=g,
        ...     edge_order=[(0, 1), (1, 2)],
        ...     edge_spacing=0.0,
        ...     bin_size=2.0,
        ... )
        >>> env.is_linearized_track
        True
        >>> env.n_bins
        50

        T-maze. A central stem with two arms branching at the
        decision point. ``edge_order`` controls the linearization
        sequence (here: stem first, then left arm, then right arm).
        ``edge_spacing`` inserts a gap between non-contiguous arm
        endpoints so the linearized coordinate doesn't fold both
        arms onto the same range:

        >>> g = nx.Graph()
        >>> g.add_nodes_from(
        ...     [
        ...         (0, {"pos": (0.0, 0.0)}),  # stem start
        ...         (1, {"pos": (0.0, 40.0)}),  # decision point
        ...         (2, {"pos": (-30.0, 40.0)}),  # left arm tip
        ...         (3, {"pos": (30.0, 40.0)}),  # right arm tip
        ...     ]
        ... )
        >>> g.add_edge(0, 1, distance=40.0)
        >>> g.add_edge(1, 2, distance=30.0)
        >>> g.add_edge(1, 3, distance=30.0)
        >>>
        >>> env = Environment.from_graph(
        ...     graph=g,
        ...     edge_order=[(0, 1), (1, 2), (1, 3)],
        ...     edge_spacing=10.0,  # 10 cm gap between left and right arms
        ...     bin_size=5.0,
        ... )

        Plus-maze (cross). Four arms meeting at the centre node.
        ``edge_order`` lists each arm in turn so the linearized
        coordinate runs N -> E -> S -> W:

        >>> g = nx.Graph()
        >>> g.add_node("center", pos=(0.0, 0.0))
        >>> g.add_nodes_from(
        ...     [
        ...         ("N", {"pos": (0.0, 50.0)}),
        ...         ("E", {"pos": (50.0, 0.0)}),
        ...         ("S", {"pos": (0.0, -50.0)}),
        ...         ("W", {"pos": (-50.0, 0.0)}),
        ...     ]
        ... )
        >>> for arm in ("N", "E", "S", "W"):
        ...     g.add_edge("center", arm, distance=50.0)
        >>>
        >>> env = Environment.from_graph(
        ...     graph=g,
        ...     edge_order=[
        ...         ("center", "N"),
        ...         ("center", "E"),
        ...         ("center", "S"),
        ...         ("center", "W"),
        ...     ],
        ...     edge_spacing=5.0,
        ...     bin_size=2.0,
        ... )
        """
        return _from_graph(
            cls,
            graph,
            edge_order,
            bin_size=bin_size,
            edge_spacing=edge_spacing,
            name=name,
            units=units,
            frame=frame,
            regions=regions,
            track_width=track_width,
        )

    @classmethod
    @keyword_hint("bin_size")
    def from_polygon(
        cls,
        polygon: shapely.Polygon,
        *,
        bin_size: float | Sequence[float],
        name: str = "",
        units: str | None = None,
        frame: str | None = None,
        regions: Mapping[str, Region] | None = None,
        arena: Arena | None = None,
        wall_fit: Literal["snap", "split"] = "snap",
        connect_diagonal_neighbors: bool = True,
    ) -> Environment[CartesianSystem]:
        """Create a 2D grid Environment masked by a Shapely Polygon.

        A regular grid of cells exactly `bin_size` wide covers the polygon's
        bounds, reaching equally past both ends of an axis whose length is not
        a whole number of cells. Only grid cells whose centers are contained
        within the polygon are considered active.

        Parameters
        ----------
        polygon : shapely.geometry.Polygon
            The Shapely Polygon object that defines the boundary of the active area.
        bin_size : float or sequence of floats
            The side length(s) of the grid cells, in the same units as the polygon
            coordinates. If a float, creates square bins. If a sequence, specifies
            bin size per dimension.
        name : str, optional
            A name for the created environment. Defaults to "".
        units : str, optional
            Unit label for the supplied coordinates; does not scale them.
        frame : str, optional
            Coordinate frame label for the supplied coordinates.
        regions : mapping of str to Region, optional
            Named regions to snapshot with the constructed environment.
        connect_diagonal_neighbors : bool, optional
            Whether to connect diagonally adjacent active grid cells.
            Defaults to True.
        arena : Arena, optional
            Continuous domain and asserted physical walls. Polygonal domains
            retain whole bins by strict center inclusion; use ``env.clip``
            to change cell areas. When omitted, the supplied polygon's
            outline is an opaque wall; for an elevated track or platform pass
            ``arena=OpenArena(polygon, outline="drop")``.
        wall_fit : {'snap', 'split'}, default='snap'
            What to do with a bin that a wall of `arena` divides. ``'snap'``
            keeps every bin and moves the wall onto the bin edges between the
            bins it divides, warning how far it moved (at most about half a
            bin); every analysis then works as on any grid. ``'split'`` cuts
            each such bin along the wall into one exact polygon bin per side,
            keeping the measured wall. Smoothing and other finite-volume
            operations refuse a split bin next to a wall's end or a diagonal
            wall, so use it with ``method='binned'`` rates or wall-distance
            measures. Bins no wall divides are the same either way, except
            that, along a slanted or curved outline, ``'split'`` makes the
            bins exact polygons that cover the arena: a bin sticking out past
            the outline is clipped to it, and a bin next to a grid cell dropped
            there takes the part of that cell inside the arena, whose
            positions it already held.

        Returns
        -------
        Environment
            A new Environment instance with a `ShapelyPolygonLayout`.

        See Also
        --------
        from_samples : Create environment by binning position data.
        from_grid_mask : Create environment from pre-defined boolean mask.
        from_pixel_mask : Create environment from binary image mask.

        Examples
        --------
        Create an environment from a rectangular polygon:

        >>> from shapely.geometry import Polygon
        >>> from neurospatial import Environment
        >>> # Create a simple rectangular arena (100cm x 50cm)
        >>> polygon = Polygon([(0, 0), (100, 0), (100, 50), (0, 50)])  # cm
        >>> env = Environment.from_polygon(
        ...     polygon=polygon,
        ...     bin_size=5.0,
        ...     name="rectangular_arena",  # 5cm bins
        ... )
        >>> env.n_dims
        2

        Create an environment from a circular arena:

        >>> from shapely.geometry import Point
        >>> center = Point(50, 50)  # cm
        >>> circular_polygon = center.buffer(25)  # Circle with radius 25cm
        >>> env = Environment.from_polygon(
        ...     polygon=circular_polygon,
        ...     bin_size=2.0,  # 2cm bins
        ... )

        """
        return _from_polygon(
            cls,
            polygon,
            bin_size=bin_size,
            name=name,
            units=units,
            frame=frame,
            regions=regions,
            arena=arena,
            wall_fit=wall_fit,
            connect_diagonal_neighbors=connect_diagonal_neighbors,
        )

    @classmethod
    @keyword_hint()
    def from_grid_mask(
        cls,
        active_mask: NDArray[np.bool_],
        grid_edges: tuple[NDArray[np.float64], ...],
        *,
        name: str = "",
        units: str | None = None,
        frame: str | None = None,
        regions: Mapping[str, Region] | None = None,
        arena: Arena | None = None,
        wall_fit: Literal["snap", "split"] = "snap",
        connect_diagonal_neighbors: bool = True,
    ) -> Environment[CartesianSystem]:
        """Create an Environment from a pre-defined N-D boolean mask and grid edges.

        This factory method allows for precise specification of active bins in
        an N-dimensional grid. Use it when you already have an N-D boolean mask
        and the explicit grid-edge arrays describing the bin boundaries.

        Parameters
        ----------
        active_mask : NDArray[np.bool_]
            An N-dimensional boolean array where `True` indicates an active bin.
            The shape of this mask must correspond to the number of bins implied
            by `grid_edges` (i.e., `tuple(len(e)-1 for e in grid_edges)`).
        grid_edges : Tuple[NDArray[np.float64], ...]
            A tuple where each element is a 1D NumPy array of bin edge positions
            for that dimension, in physical units (e.g., cm, meters). The edges
            define the boundaries of bins along each dimension. For example, edges
            [0, 10, 20, 30] define three bins: [0-10], [10-20], [20-30]. Edges
            must be finite, strictly increasing, and uniformly spaced along each
            axis; axes may differ.
        name : str, optional
            A name for the created environment. Defaults to "".
        units : str, optional
            Unit label for the supplied coordinates; does not scale them.
        frame : str, optional
            Coordinate frame label for the supplied coordinates.
        regions : mapping of str to Region, optional
            Named regions to snapshot with the constructed environment.
        connect_diagonal_neighbors : bool, optional
            Whether to connect diagonally adjacent active grid cells.
            Defaults to True.
        arena : Arena, optional
            Continuous domain and asserted physical walls. Polygonal domains
            retain whole bins by strict center inclusion. When omitted, the
            mask defines the computational domain without declaring its
            outline a physical boundary; pass ``MaskArena(..., outline=...)``
            to declare one.
        wall_fit : {'snap', 'split'}, default='snap'
            What to do with a bin that a wall of `arena` divides. ``'snap'``
            keeps every bin and moves the wall onto the bin edges between the
            bins it divides, warning how far it moved (at most about half a
            bin); every analysis then works as on any grid. ``'split'`` cuts
            each such bin along the wall into one exact polygon bin per side,
            keeping the measured wall. Smoothing and other finite-volume
            operations refuse a split bin next to a wall's end or a diagonal
            wall, so use it with ``method='binned'`` rates or wall-distance
            measures. Bins no wall divides are the same either way, except
            that, along a slanted or curved outline, ``'split'`` makes the
            bins exact polygons that cover the arena: a bin sticking out past
            the outline is clipped to it, and a bin next to a grid cell dropped
            there takes the part of that cell inside the arena, whose
            positions it already held.

        Returns
        -------
        Environment
            A new Environment instance with a `MaskedGridLayout`.

        See Also
        --------
        from_samples : Create environment by binning position data.
        from_polygon : Create environment with polygon-defined boundary.
        from_pixel_mask : Create environment from binary image / pixel mask.

        Examples
        --------
        Create an environment from a custom mask:

        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> # Create a simple 2D mask (10x10 grid with center region active)
        >>> mask = np.zeros((10, 10), dtype=bool)
        >>> mask[3:7, 3:7] = True  # Center 4x4 region is active
        >>> # Define grid edges (creates 10cm x 10cm bins)
        >>> grid_edges = (
        ...     np.linspace(0, 100, 11),  # x edges in cm
        ...     np.linspace(0, 100, 11),  # y edges in cm
        ... )
        >>> env = Environment.from_grid_mask(
        ...     active_mask=mask, grid_edges=grid_edges, name="center_region"
        ... )
        >>> env.n_bins
        16

        """
        return _from_grid_mask(
            cls,
            active_mask,
            grid_edges,
            name=name,
            units=units,
            frame=frame,
            regions=regions,
            arena=arena,
            wall_fit=wall_fit,
            connect_diagonal_neighbors=connect_diagonal_neighbors,
        )

    @classmethod
    @keyword_hint("pixel_size")
    def from_pixel_mask(
        cls,
        image_mask: NDArray[np.bool_],
        *,
        pixel_size: float | tuple[float, float],
        name: str = "",
        units: str | None = None,
        frame: str | None = None,
        regions: Mapping[str, Region] | None = None,
        arena: Arena | None = None,
        wall_fit: Literal["snap", "split"] = "snap",
        connect_diagonal_neighbors: bool = True,
    ) -> Environment[CartesianSystem]:
        """Create a 2D Environment from a binary image / pixel mask.

        Each ``True`` pixel in the ``image_mask`` becomes an active bin in the
        environment. ``pixel_size`` determines the spatial scale of those
        pixels in physical units.

        Parameters
        ----------
        image_mask : NDArray[np.bool_], shape (n_rows, n_cols)
            A 2D boolean array where ``True`` pixels define active bins.
            Columns map to x and rows to y, with row 0 at the lowest y: the
            environment, drawn with y up, is flipped vertically relative to
            ``plt.imshow(image_mask)``. Pass ``image_mask[::-1]`` if the top
            of your image should be high y.
        pixel_size : float or tuple of (float, float)
            The spatial size of each pixel in physical units (e.g., cm,
            meters). If a float, pixels are square. If a tuple
            ``(width, height)``, specifies pixel dimensions. For example,
            if your camera captures images where each pixel represents
            0.5 cm, pass ``pixel_size=0.5``.
        name : str, optional
            A name for the created environment. Defaults to "".
        units : str, optional
            Unit label for the supplied coordinates; does not scale them.
        frame : str, optional
            Coordinate frame label for the supplied coordinates.
        regions : mapping of str to Region, optional
            Named regions to snapshot with the constructed environment.
        connect_diagonal_neighbors : bool, optional
            Whether to connect diagonally adjacent active pixel-bins.
            Defaults to True.
        arena : Arena, optional
            Continuous domain and asserted physical walls. Polygonal domains
            retain whole pixels by strict center inclusion. When omitted, the
            image mask defines a computational domain without declaring its
            outline a physical boundary.
        wall_fit : {'snap', 'split'}, default='snap'
            What to do with a bin that a wall of `arena` divides. ``'snap'``
            keeps every bin and moves the wall onto the bin edges between the
            bins it divides, warning how far it moved (at most about half a
            bin); every analysis then works as on any grid. ``'split'`` cuts
            each such bin along the wall into one exact polygon bin per side,
            keeping the measured wall. Smoothing and other finite-volume
            operations refuse a split bin next to a wall's end or a diagonal
            wall, so use it with ``method='binned'`` rates or wall-distance
            measures. Bins no wall divides are the same either way, except
            that, along a slanted or curved outline, ``'split'`` makes the
            bins exact polygons that cover the arena: a bin sticking out past
            the outline is clipped to it, and a bin next to a grid cell dropped
            there takes the part of that cell inside the arena, whose
            positions it already held.

        Returns
        -------
        Environment
            A new Environment instance with an ``ImageMaskLayout``.

        See Also
        --------
        from_grid_mask : Create environment from pre-defined boolean mask
            with explicit grid edges.
        from_polygon : Create environment with polygon-defined boundary.
        from_samples : Create environment by binning position data.

        Examples
        --------
        Create an environment from a binary image mask:

        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> # Create a simple binary image (e.g., from thresholding camera frame)
        >>> image_height, image_width = 48, 64
        >>> mask = np.zeros((image_height, image_width), dtype=bool)
        >>> # Mark a rectangular region as active
        >>> mask[10:40, 15:50] = True
        >>> env = Environment.from_pixel_mask(
        ...     image_mask=mask,
        ...     pixel_size=5.0,  # Each pixel = 5 cm
        ...     name="arena_from_image",
        ... )
        >>> env.n_bins  # one bin per active pixel
        1050

        """
        return _from_pixel_mask(
            cls,
            image_mask,
            pixel_size=pixel_size,
            name=name,
            units=units,
            frame=frame,
            regions=regions,
            arena=arena,
            wall_fit=wall_fit,
            connect_diagonal_neighbors=connect_diagonal_neighbors,
        )

    @classmethod
    @keyword_hint(
        fix=(
            "build the engine first and pass it alone, e.g. "
            "Environment.from_layout(create_layout('TriangularMesh', "
            "boundary_polygon=polygon, point_spacing=2.0)), with create_layout "
            "from neurospatial.layout; pass name=, units=, frame=, regions= "
            "and arena= by keyword"
        )
    )
    def from_layout(
        cls,
        layout: LayoutEngine,
        *,
        name: str = "",
        units: str | None = None,
        frame: str | None = None,
        regions: Mapping[str, Region] | None = None,
        arena: Arena | None = None,
        wall_fit: Literal["snap", "split"] = "snap",
    ) -> Environment[CartesianSystem]:
        """Create an Environment from a built layout engine.

        Use this for a layout that has no dedicated factory: a triangular
        mesh, a grid tiled over explicit ranges without samples, or your own
        layout engine. Build the engine first, with
        :func:`neurospatial.layout.create_layout` or the engine's own
        ``build(...)``, then pass it here. The environment copies what the
        engine's ``binning()`` returns and keeps no reference to the engine.

        Parameters
        ----------
        layout : LayoutEngine
            A built layout engine: a built-in engine or any object that
            implements the ``LayoutEngine`` protocol, including ``binning()``.
        name : str, optional
            A name for the created environment. Defaults to "".
        units : str, optional
            Unit label for the supplied coordinates; does not scale them.
        frame : str, optional
            Coordinate frame label for the supplied coordinates.
        regions : mapping of str to Region, optional
            Named regions to snapshot with the constructed environment.
        arena : Arena, optional
            Continuous domain. Bins whose centers fall outside a known arena
            are left out. Otherwise the domain is inferred from the layout.
        wall_fit : {'snap', 'split'}, default='snap'
            What to do with a bin that a wall of `arena` divides. ``'snap'``
            keeps every bin and moves the wall onto the bin edges between the
            bins it divides, warning how far it moved (at most about half a
            bin); every analysis then works as on any grid. ``'split'`` cuts
            each such bin along the wall into one exact polygon bin per side,
            keeping the measured wall. Smoothing and other finite-volume
            operations refuse a split bin next to a wall's end or a diagonal
            wall, so use it with ``method='binned'`` rates or wall-distance
            measures. Bins no wall divides are the same either way, except
            that, along a slanted or curved outline, ``'split'`` makes the
            bins exact polygons that cover the arena: a bin sticking out past
            the outline is clipped to it, and a bin next to a grid cell dropped
            there takes the part of that cell inside the arena, whose
            positions it already held.

        Returns
        -------
        Environment
            A new Environment instance. ``layout_type`` is the engine's
            ``layout_type`` label, or its binning's ``kind`` when it has
            none, and ``layout_parameters`` are its recorded build choices.

        Raises
        ------
        TypeError
            If `layout` is a layout name or class instead of a built engine.
        LayoutNotBuiltError
            If the engine has not been built.

        See Also
        --------
        neurospatial.layout.create_layout : Build a built-in layout engine.
        from_samples : Create environment by binning position data.
        from_polygon : Create environment with polygon-defined boundary.
        from_grid_mask : Create environment from pre-defined boolean mask.
        from_pixel_mask : Create environment from binary image mask.
        from_graph : Create 1D linearized track environment.
        from_bins : Create environment from explicit binning records.

        Examples
        --------
        A triangular mesh over a square arena, in cm:

        >>> from shapely.geometry import box
        >>> from neurospatial import Environment
        >>> from neurospatial.layout import create_layout
        >>> mesh = create_layout(
        ...     "TriangularMesh", boundary_polygon=box(0, 0, 30, 30), point_spacing=4.0
        ... )
        >>> env = Environment.from_layout(mesh, name="arena", units="cm")
        >>> env.layout_type
        'TriangularMesh'

        A hexagonal tiling of explicit ranges, without position samples:

        >>> hexagons = create_layout(
        ...     "Hexagonal", hexagon_width=4.0, dimension_ranges=[(0, 30), (0, 30)]
        ... )
        >>> Environment.from_layout(hexagons).n_bins > 0
        True
        """
        return _from_layout(
            cls,
            layout,
            name=name,
            units=units,
            frame=frame,
            regions=regions,
            arena=arena,
            wall_fit=wall_fit,
        )

    @classmethod
    @keyword_hint(redirect="neurospatial.io.nwb.environment_from_position")
    def from_nwb(
        cls,
        nwbfile: Any,
        *,
        name: str = "spatial_environment",
    ) -> Environment[Any]:
        """Load an Environment saved in an NWB file's scratch space.

        Reads a record written by :meth:`to_nwb` (or
        :func:`neurospatial.io.nwb.write_environment`). Schema 3.0 restores the
        complete binning, arena and regions without a layout builder. To
        build a new environment from the file's Position data, use
        :func:`neurospatial.io.nwb.environment_from_position`.

        Parameters
        ----------
        nwbfile : NWBFile
            The NWB file to read from. Must be a pynwb.NWBFile instance.
        name : str, default "spatial_environment"
            Name of the stored record in scratch/, as passed to
            :meth:`to_nwb`. The restored environment keeps its own saved
            ``env.name``.

        Returns
        -------
        Environment
            The stored Environment, in its original coordinate system.
            Static type checkers see ``Environment[Any]``: whether the
            stored environment is Cartesian or egocentric polar is read from
            the NWB record, so it is not known from the call. Check
            ``env.coordinate_kind`` before calling Cartesian-only methods.

        Raises
        ------
        KeyError
            If no record called `name` is in the NWB scratch space. The
            message lists the saved environments and points to
            :func:`neurospatial.io.nwb.environment_from_position`.
        TypeError
            If a building option such as ``bin_size`` is passed; the message
            shows the ``environment_from_position`` call that takes it.
        ValueError
            If the stored record uses an unsupported schema or is inconsistent.
        ImportError
            If pynwb is not installed.

        See Also
        --------
        to_nwb : Write Environment to NWB file.
        neurospatial.io.nwb.read_environment : The NWB reader this method calls.
        neurospatial.io.nwb.environment_from_position : Bin NWB Position data.

        Examples
        --------
        >>> from datetime import datetime, timezone
        >>> from pynwb import NWBFile
        >>> from neurospatial import Environment
        >>> env = Environment.linear_track(
        ...     endpoints=[(0, 0), (6, 0)], bin_size=2.0, units="cm"
        ... )
        >>> nwbfile = NWBFile(
        ...     "example", "example", datetime(2025, 1, 1, tzinfo=timezone.utc)
        ... )
        >>> env.to_nwb(nwbfile, name="linear_track")
        >>> Environment.from_nwb(nwbfile, name="linear_track").n_bins
        3
        """
        return _from_nwb(cls, nwbfile, name=name)

    def bin_center_of(
        self,
        bin_indices: int | Sequence[int] | NDArray[np.integer],
    ) -> NDArray[np.float64]:
        """Given one or more active-bin indices, return their N-D center coordinates.

        Parameters
        ----------
        bin_indices : int or sequence of int
            Index (or list/array of indices) of active bins (0 <= idx < self.n_bins).

        Returns
        -------
        centers : NDArray[np.float64]
            The center coordinate(s) of the requested bin(s).
            Shape is (len(bin_indices), n_dims) if multiple indices,
            or (n_dims,) if single index.

        Raises
        ------
        TypeError
            If the indices are floats or booleans; use :meth:`bin_at` to find
            the bin that contains a coordinate.
        BinIndexOutOfRangeError
            If any bin index is outside ``[0, n_bins)``, including negative
            indices (a ``ValueError`` subclass).

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> data = np.random.rand(100, 2) * 10
        >>> env = Environment.from_samples(data, bin_size=2.0)
        >>> center = env.bin_center_of(0)
        >>> print(center.shape)
        (2,)
        >>> centers = env.bin_center_of([0, 1, 2])
        >>> print(centers.shape)
        (3, 2)

        """
        return _bin_center_of(self.bin_centers, bin_indices, self._geometry.coordinates)

    @keyword_hint(former=("bin_index",), translate=former_targets(1), why=TARGET_WHY)
    def neighbors(
        self,
        *,
        bins: int | Sequence[int] | NDArray[np.integer] | None = None,
        points: ArrayLike | None = None,
    ) -> list[int]:
        """Return the bins connected to one bin in the connectivity graph.

        Parameters
        ----------
        bins : int, optional
            The bin index, ``0 <= bins < n_bins``.
        points : array-like, shape (1, n_dims), optional
            A coordinate, mapped to the bin that contains it, in the same
            coordinates and units as ``bin_centers``.

        Returns
        -------
        list[int]
            Indices of the neighboring bins. Can be used directly for array
            indexing.

        Raises
        ------
        TypeError
            If neither or both of ``bins`` and ``points`` are given, or
            ``bins`` holds non-integers.
        ValueError
            If more than one bin or point is given, or the point is outside
            the environment or has NaN coordinates.
        BinIndexOutOfRangeError
            If the bin index is outside ``[0, n_bins)``.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> x, y = np.meshgrid([0.0, 2.0, 4.0], [0.0, 2.0, 4.0])
        >>> env = Environment.from_samples(np.c_[x.ravel(), y.ravel()], bin_size=2.0)
        >>> len(env.neighbors(bins=0))  # corner bin, with diagonal connectivity
        3
        >>> sorted(env.neighbors(points=[[2.0, 2.0]])) == sorted(
        ...     env.neighbors(bins=int(env.bin_at(np.array([[2.0, 2.0]]))[0]))
        ... )
        True

        """
        (start,) = self._target_bins("neighbors()", bins, points, count=1)
        return list(self.connectivity.neighbors(int(start)))

    @property
    def bin_sizes(self) -> NDArray[np.float64]:
        """Calculate the area (for 2D) or volume (for 3D+) of each active bin.

        This represent the actual size of each bin in the environment, as
        opposed to the requested `bin_size` which is the nominal size used
        during layout creation.

        For 1D environments, this typically returns the length of each bin.
        Measures come from the environment's owned binning data.

        Returns
        -------
        NDArray[np.float64], shape (n_active_bins,)
            An array containing the area/volume/length of each active bin.

        Raises
        ------
        ValueError
            If the bins mix measures, as a track combined with an open field
            does: a sum, ratio or density over lengths and areas together has
            no unit. Analyze each component separately, or read each bin's
            measure with its dimension from ``env.binning.bins.sizes`` and
            ``env.binning.bins.measure_dims``.

        Notes
        -----
        Cell measures are cached internally. Each access returns an independent
        read-only array, so changing an export's shape cannot alter the bins.

        For environments with >100,000 bins, this may consume significant memory.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> data = np.random.rand(100, 2) * 10
        >>> env = Environment.from_samples(data, bin_size=2.0)
        >>> sizes = env.bin_sizes
        >>> print(sizes.shape[0] == env.n_bins)
        True

        """
        bins = self._binning.bins
        if bins.measure_dim is None and bins.n_bins:
            raise ValueError(_mixed_measure_error(bins.measure_dims, self.units))
        sizes: NDArray[np.float64] = bins.sizes.copy()
        sizes.setflags(write=False)
        return sizes

    @keyword_hint(
        former=("source_active_bin_idx", "target_active_bin_idx"),
        translate=former_targets(2),
        why=TARGET_WHY,
    )
    def path_between(
        self,
        *,
        bins: Sequence[int] | NDArray[np.integer] | None = None,
        points: ArrayLike | None = None,
    ) -> list[int]:
        """Find the shortest path between two bins or two points.

        The path is the sequence of bin indices along the shortest route
        through the connectivity graph, weighted by the ``"distance"`` of each
        edge.

        Parameters
        ----------
        bins : sequence of int, shape (2,), optional
            Start and end bin indices, ``[start, end]``.
        points : array-like, shape (2, n_dims), optional
            Start and end coordinates, one row each, mapped to the bins that
            contain them.

        Returns
        -------
        list[int]
            Bin indices from start to end, both included. A single-element
            list when both ends are the same bin, and an empty list (with a
            warning) when no path exists.

        Raises
        ------
        TypeError
            If neither or both of ``bins`` and ``points`` are given, or
            ``bins`` holds non-integers.
        ValueError
            If there are not exactly two targets, or a point is outside the
            environment or has NaN coordinates.
        BinIndexOutOfRangeError
            If a bin index is outside ``[0, n_bins)``.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> x, y = np.meshgrid(np.arange(0.0, 10.0), np.arange(0.0, 10.0))
        >>> env = Environment.from_samples(np.c_[x.ravel(), y.ravel()], bin_size=2.0)
        >>> path = env.path_between(bins=[0, env.n_bins - 1])
        >>> path[0], path[-1] == env.n_bins - 1
        (0, True)
        >>> center = env.bin_centers[3]
        >>> env.path_between(points=[center, center + 0.1])  # both in bin 3
        [3]

        """
        start, end = self._target_bins("path_between()", bins, points, count=2)
        return _path_between(self.connectivity, int(start), int(end))

    @keyword_hint(
        former=("targets",),
        translate=former_targets(1, region=True),
        why=REGION_TARGET_WHY,
    )
    def distance_to(
        self,
        *,
        bins: int | Sequence[int] | NDArray[np.integer] | None = None,
        points: ArrayLike | None = None,
        region: str | None = None,
        metric: DistanceMetric = "geodesic",
    ) -> NDArray[np.float64]:
        """Distance from every bin to the nearest target.

        Give the target as bin indices, as coordinates or as a region name,
        for example the distance from every bin to a reward port for a goal
        or approach analysis. With several targets, each bin gets the distance
        to the nearest one.

        Parameters
        ----------
        bins : int or sequence of int, optional
            Target bin indices.
        points : array-like, shape (n_points, n_dims), optional
            Target coordinates, one row each, mapped to the bins that contain
            them. Use the same coordinates and units as ``bin_centers``.
        region : str, optional
            Name of a region in ``self.regions``. Its targets are the bins it
            covers, by the rule of :meth:`bins_in_region`: the bin that
            contains a point region, or the bins whose centers lie inside a
            polygon region or on its edge.
        metric : {'euclidean', 'geodesic'}, default='geodesic'
            ``'geodesic'`` follows the connectivity graph, so it goes around
            walls; ``'euclidean'`` is the straight-line distance between bin
            centers. Both are in the units of ``bin_centers``.

        Returns
        -------
        distances : NDArray[np.float64], shape (n_bins,)
            Distance from each bin to the nearest target, ``0`` at the
            targets. Bins in a part of the graph with no target are
            ``np.inf`` for the geodesic metric.

        Raises
        ------
        TypeError
            If not exactly one of ``bins``, ``points`` and ``region`` is
            given, or ``bins`` holds non-integers.
        ValueError
            If a target is empty, a point is outside the environment or has
            NaN coordinates, the region covers no bin, or ``metric`` is
            invalid.
        BinIndexOutOfRangeError
            If a bin index is outside ``[0, n_bins)``.
        RegionNotFoundError
            If ``region`` is not in ``self.regions`` (a ``KeyError``).
        CartesianOnlyError
            For ``metric="euclidean"`` on a polar environment.

        See Also
        --------
        rings : Bins grouped by hop distance from one bin.
        reachable_from : Bins reachable from one bin within a radius.
        neurospatial.ops.distance.distance_field : Graph distances from bins.

        Notes
        -----
        The geodesic metric runs Dijkstra's algorithm on the connectivity
        graph, with each edge's ``"distance"`` (the physical distance between
        bin centers) as its length.

        A point target is the bin that contains it, so the distance to a point
        equals the distance to its bin.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment, Regions
        >>> data = np.array([[i, j] for i in range(10) for j in range(10)])
        >>> env = Environment.from_samples(data, bin_size=1.0)
        >>> regions = Regions()
        >>> _ = regions.add("reward", point=(8.2, 8.7))
        >>> env = env.with_regions(regions)
        >>> to_reward = env.distance_to(region="reward")
        >>> to_reward.shape
        (100,)
        >>> bool(np.array_equal(to_reward, env.distance_to(points=[[8.2, 8.7]])))
        True
        >>> corners = env.distance_to(bins=[0, env.n_bins - 1], metric="euclidean")
        >>> float(corners[0]), float(corners[-1])
        (0.0, 0.0)

        """
        targets = self._target_bins(
            "distance_to()", bins, points, region=region, takes_region=True
        )
        return _distance_to(
            targets,
            metric=metric,
            bin_centers=self.bin_centers,
            graph=self.connectivity,
            coordinates=self._geometry.coordinates,
        )

    @keyword_hint(former=("source_bin",), translate=former_targets(1), why=TARGET_WHY)
    def reachable_from(
        self,
        *,
        bins: int | Sequence[int] | NDArray[np.integer] | None = None,
        points: ArrayLike | None = None,
        radius: int | float | None = None,
        metric: Literal["hops", "geodesic"] = "hops",
    ) -> NDArray[np.bool_]:
        """Find all bins reachable from source within optional radius.

        This method performs graph traversal to find which bins can be reached
        from a starting bin, optionally constrained by a maximum distance.
        Useful for:
        - Computing neighborhoods and local regions
        - Identifying reachable areas from a starting position
        - Building distance-limited queries

        Parameters
        ----------
        bins : int, optional
            The start bin index, ``0 <= bins < n_bins``.
        points : array-like, shape (1, n_dims), optional
            The start coordinate, mapped to the bin that contains it, in the
            same coordinates and units as ``bin_centers``.
        radius : int, float, or None, optional
            Maximum distance/hops. If None, find all reachable bins in the
            same connected component.
            - For metric='hops': radius is maximum number of edges.
            - For metric='geodesic': radius is maximum graph distance in
              physical units.
        metric : {'hops', 'geodesic'}, default='hops'
            Distance metric to use:
            - 'hops': Count graph edges (breadth-first search).
            - 'geodesic': Sum edge distances in physical units (Dijkstra).

        Returns
        -------
        reachable : NDArray[np.bool_], shape (n_bins,)
            Boolean mask where True indicates reachable bins.
            The resolved source bin is always reachable.

        Raises
        ------
        TypeError
            If neither or both of ``bins`` and ``points`` are given, or
            ``bins`` holds non-integers.
        BinIndexOutOfRangeError
            If the bin index is outside ``[0, n_bins)``.
        ValueError
            If more than one bin or point is given, the point is outside the
            environment or has NaN coordinates, ``radius`` is negative, or
            ``metric`` is not ``'hops'`` or ``'geodesic'``.

        See Also
        --------
        components : Find connected components.
        distance_between : Compute distance between two bins.

        Notes
        -----
        **Algorithm details**:
        - metric='hops': Uses breadth-first search (BFS) to specified depth.
        - metric='geodesic': Uses Dijkstra's algorithm with distance cutoff.

        **Performance**:
        - With radius=None: O(V + E) where V=bins, E=edges
        - With radius: Depends on local graph density, typically much faster

        The geodesic metric uses the 'distance' attribute on graph edges,
        which represents the Euclidean distance between bin centers.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> data = np.array([[i, j] for i in range(10) for j in range(10)])
        >>> env = Environment.from_samples(data, bin_size=1.0)
        >>> within_one_hop = env.reachable_from(bins=55, radius=1, metric="hops")
        >>> int(within_one_hop.sum())  # the bin and its 8 neighbors
        9
        >>> within_2cm = env.reachable_from(
        ...     points=[[5.0, 5.0]], radius=2.0, metric="geodesic"
        ... )
        >>> bool(within_2cm[55])
        True

        """
        (start,) = self._target_bins("reachable_from()", bins, points, count=1)
        return _reachable_from(
            self.connectivity, int(start), radius=radius, metric=metric
        )

    def components(
        self,
        *,
        largest_only: bool = False,
    ) -> list[NDArray[np.intp]]:
        """Find connected components of the environment graph.

        A connected component is a maximal subset of bins where every pair
        of bins is connected by a path through the graph. This is useful for:

        - Identifying disconnected regions in masked environments
        - Finding traversable subregions
        - Detecting isolated islands in the environment

        Parameters
        ----------
        largest_only : bool, default=False
            If True, return only the largest component.
            If False, return all components sorted by size (largest first).

        Returns
        -------
        components : list[NDArray[np.intp]]
            List of bin index arrays, one per component.
            Components are sorted by size (largest first).
            Each array contains the bin indices in that component.

        See Also
        --------
        reachable_from : Find bins reachable from a source within a radius.

        Notes
        -----
        This method uses NetworkX's connected_components algorithm to identify
        connected subgraphs in the environment's connectivity graph.

        For environments with a single connected region (e.g., most regular grids),
        this will return a single component containing all bins.

        Examples
        --------
        Two chambers of a 100 x 40 cm grid separated by a wall one bin wide:

        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> mask = np.ones((10, 4), dtype=bool)
        >>> mask[4, :] = False  # the wall
        >>> grid_edges = (np.arange(11) * 10.0, np.arange(5) * 10.0)
        >>> env = Environment.from_grid_mask(mask, grid_edges)
        >>> comps = env.components()
        >>> print(f"Found {len(comps)} components")
        Found 2 components
        >>> [len(component) for component in comps]
        [20, 16]

        >>> largest = env.components(largest_only=True)[0]
        >>> print(f"Largest component: {len(largest)} of {env.n_bins} bins")
        Largest component: 20 of 36 bins

        """
        return _components(self.connectivity, largest_only=largest_only)

    @keyword_hint(former=("center_bin",), translate=former_targets(1), why=TARGET_WHY)
    def rings(
        self,
        *,
        bins: int | Sequence[int] | NDArray[np.integer] | None = None,
        points: ArrayLike | None = None,
        hops: int,
    ) -> list[NDArray[np.intp]]:
        """Compute k-hop neighborhoods (BFS layers).

        This method performs breadth-first search (BFS) from the center bin,
        organizing bins into "rings" by their hop distance. Ring k contains
        all bins exactly k graph edges away from the center. Useful for:
        - Local neighborhood analysis
        - Distance-based feature extraction
        - Spatial smoothing with varying radii
        - Analyzing connectivity patterns

        Parameters
        ----------
        bins : int, optional
            The center bin index, ``0 <= bins < n_bins``.
        points : array-like, shape (1, n_dims), optional
            The center coordinate, mapped to the bin that contains it, in the
            same coordinates and units as ``bin_centers``.
        hops : int
            Number of hop layers to compute (non-negative).

        Returns
        -------
        rings : list[NDArray[np.intp]], length hops+1
            List of bin index arrays, one per hop distance.
            rings[k] contains all bins exactly k hops from center.
            rings[0] is the center bin itself.
            If fewer than hops layers exist (small or disconnected graph),
            later rings will be empty arrays.

        Raises
        ------
        TypeError
            If neither or both of ``bins`` and ``points`` are given, ``bins``
            holds non-integers, or ``hops`` is not an integer.
        ValueError
            If more than one bin or point is given, the point is outside the
            environment or has NaN coordinates, or ``hops`` is negative.
        BinIndexOutOfRangeError
            If the bin index is outside ``[0, n_bins)``.

        See Also
        --------
        distance_to : Compute distance from each bin to target set.
        reachable_from : Find bins reachable from source within a radius.
        components : Find connected components of the graph.

        Notes
        -----
        **Hop Distance vs Physical Distance**:
        Rings are based on graph edges (hops), not physical distance. In a
        regular grid, 1 hop = 1 grid edge. In irregular graphs, hop distance
        may not correlate with Euclidean distance.

        **Disconnected Graphs**:
        If the center bin is in a disconnected component, rings will only
        cover bins in the same component. Bins in other components will never
        appear in any ring.

        **Ring Coverage**:
        The union of all rings equals the set of bins reachable from center
        within `hops` edges. Rings are mutually disjoint.

        **Performance**:
        Uses BFS with NetworkX. Complexity is O(E + V) where E = edges,
        V = vertices (bins). Very fast even for large graphs.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> data = np.array([[i, j] for i in range(10) for j in range(10)])
        >>> env = Environment.from_samples(data, bin_size=1.0)
        >>> rings_result = env.rings(bins=55, hops=2)
        >>> [len(ring) for ring in rings_result]  # 1 center, 8 around it, then 16
        [1, 8, 16]
        >>> same = env.rings(points=[[5.0, 5.0]], hops=2)
        >>> all(np.array_equal(a, b) for a, b in zip(same, rings_result))
        True

        """
        (center,) = self._target_bins("rings()", bins, points, count=1)
        return _rings(self.connectivity, int(center), hops=hops)

    def to_file(self, path: str | Path, *, overwrite: bool = False) -> None:
        """Save Environment-v3 binning and Arena data to JSON and NPZ files.

        Loading restores Arena, bins, lookup, faces, drawing components and clip
        mapping without rebuilding a layout. Creates two files:
        `{path}.json` (metadata) and `{path}.npz` (arrays).

        Parameters
        ----------
        path : str or Path
            Base path for output files (without extension).
            Will create `{path}.json` and `{path}.npz`.
        overwrite : bool, default=False
            If ``False`` (the default), raise :class:`FileExistsError` rather
            than overwrite an existing ``{path}.json`` / ``{path}.npz``, so an
            accidental re-run cannot silently destroy a saved environment. Pass
            ``True`` to replace existing files.

        Raises
        ------
        FileExistsError
            If ``overwrite`` is ``False`` and a target file already exists.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> x, y = np.meshgrid([0.0, 2.0, 4.0], [0.0, 2.0, 4.0])
        >>> env = Environment.from_samples(np.c_[x.ravel(), y.ravel()], bin_size=2.0)
        >>> env.to_file("my_environment")  # Creates .json and .npz companions
        >>> env.to_file("my_environment", overwrite=True)  # Explicit replacement

        See Also
        --------
        from_file : Load environment from saved files
        to_dict : Convert to dictionary for in-memory use

        Notes
        -----
        The JSON file contains metadata, regions, and the serialization
        version, while the npz file holds numpy arrays in efficient binary
        form. No arbitrary code is executed at load time.

        """
        from neurospatial.io import to_file

        to_file(self, path, overwrite=overwrite)

    @classmethod
    def from_file(cls, path: str | Path) -> Environment[Any]:
        """Load Environment from versioned JSON + npz files.

        Parameters
        ----------
        path : str or Path
            Base path to load from (without extension).
            Will read `{path}.json` and `{path}.npz`.

        Returns
        -------
        Environment
            Reconstructed Environment instance, in its saved coordinate system.
            Static type checkers see ``Environment[Any]``: whether the
            stored environment is Cartesian or egocentric polar is read from
            the file, so it is not known from the call. Check
            ``env.coordinate_kind`` before calling Cartesian-only methods.

        Raises
        ------
        FileNotFoundError
            If JSON or npz files are missing.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> x, y = np.meshgrid([0.0, 2.0, 4.0], [0.0, 2.0, 4.0])
        >>> env = Environment.from_samples(np.c_[x.ravel(), y.ravel()], bin_size=2.0)
        >>> env.to_file("my_environment")
        >>> loaded = Environment.from_file("my_environment")
        >>> np.array_equal(loaded.bin_centers, env.bin_centers)
        True

        See Also
        --------
        to_file : Save environment to files
        from_dict : Reconstruct from dictionary

        Notes
        -----
        Environment-v3 restores Arena, bins, lookup, faces, drawing components
        and clip mapping without rebuilding a layout. Environment-v2 remains
        readable with an explicitly unknown Arena; old outlines and masks do
        not establish a physical footprint. Recreate version 1 environments
        with a factory and save them again.

        """
        from neurospatial.io import from_file

        return from_file(path)

    def to_dict(self) -> dict[str, Any]:
        """Convert Environment to dictionary for in-memory handoff.

        Stores Environment-v3 binning and Arena data with numerical arrays as
        lists and explicit shapes/dtypes. Loading decodes this data without
        rebuilding a layout. Version 2 remains readable with an unknown Arena;
        version 1 must be recreated with a factory.

        Returns
        -------
        dict[str, Any]
            Dictionary representation with all arrays as lists.

        Examples
        --------
        >>> import json
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> x, y = np.meshgrid([0.0, 2.0, 4.0], [0.0, 2.0, 4.0])
        >>> env = Environment.from_samples(np.c_[x.ravel(), y.ravel()], bin_size=2.0)
        >>> env_dict = env.to_dict()
        >>> # The dictionary holds only JSON types, so it can be written as JSON
        >>> json_str = json.dumps(env_dict)

        See Also
        --------
        from_dict : Reconstruct from dictionary
        to_file : Save to disk with efficient binary format

        Notes
        -----
        This method creates an in-memory representation. For disk storage,
        use `to_file()` instead, which stores arrays in efficient binary
        format (npz) rather than converting to lists.

        """
        from neurospatial.io import to_dict

        return to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Environment[Any]:
        """Decode Environment-v3 binning and Arena without rebuilding a layout.

        Version 2 dictionaries retain bins with an explicitly unknown Arena.
        Version 1 must be recreated and re-saved with an Environment factory.

        Parameters
        ----------
        data : dict[str, Any]
            Dictionary from `to_dict()` or equivalent structure.

        Returns
        -------
        Environment
            Reconstructed instance, in its saved coordinate system.
            Static type checkers see ``Environment[Any]``: whether the
            stored environment is Cartesian or egocentric polar is read from
            the dictionary, so it is not known from the call. Check
            ``env.coordinate_kind`` before calling Cartesian-only methods.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> x, y = np.meshgrid([0.0, 2.0, 4.0], [0.0, 2.0, 4.0])
        >>> env = Environment.from_samples(np.c_[x.ravel(), y.ravel()], bin_size=2.0)
        >>> reconstructed = Environment.from_dict(env.to_dict())

        The reconstruction is a new object, so ``env == reconstructed`` is
        False (environments compare by identity). Compare their contents:

        >>> np.array_equal(env.bin_centers, reconstructed.bin_centers)
        True
        >>> np.array_equal(env.edge_pairs, reconstructed.edge_pairs)
        True

        See Also
        --------
        to_dict : Convert to dictionary
        from_file : Load from disk files

        """
        from neurospatial.io import from_dict

        return from_dict(data)

    @keyword_hint("name")
    def to_nwb(
        self,
        nwbfile: Any,
        *,
        name: str = "spatial_environment",
        overwrite: bool = False,
    ) -> None:
        """Write Environment to NWB file scratch space.

        Serializes the Environment to the NWB file's scratch space, allowing
        it to be stored alongside neural data and retrieved later using
        ``Environment.from_nwb()``.

        Environment schema 3.0 stores Arena, explicit walls, bins, exact lookup,
        cell faces, drawing components, clip mapping, track linearization, graph
        attributes, regions and coordinate metadata. Loading does not rebuild a
        layout. Schema 2.0 remains readable with an unknown Arena; schemas 1.0
        and 1.1 must be recreated with a factory and saved again.

        Parameters
        ----------
        nwbfile : NWBFile
            The NWB file to write to. Must be a pynwb.NWBFile instance.
        name : str, default "spatial_environment"
            Name for the environment in scratch/. Use this name when loading
            with ``Environment.from_nwb(nwbfile, name=name)``.
        overwrite : bool, default False
            If True, replace existing environment with the same name.
            If False, raise ValueError on duplicate name.

        Raises
        ------
        ValueError
            If environment with same name exists and overwrite=False.
        ImportError
            If pynwb is not installed.

        See Also
        --------
        from_nwb : Load Environment from NWB file.
        neurospatial.io.nwb.write_environment : Low-level NWB writing function.

        Examples
        --------
        Save environment to NWB file:

        >>> import tempfile
        >>> from datetime import datetime, timezone
        >>> from pathlib import Path
        >>> import numpy as np
        >>> from pynwb import NWBHDF5IO, NWBFile
        >>> from neurospatial import Environment
        >>>
        >>> # Create environment
        >>> rng = np.random.default_rng(0)
        >>> positions = rng.random((1000, 2)) * 100
        >>> env = Environment.from_samples(positions, bin_size=5.0, units="cm")
        >>>
        >>> # Save to NWB
        >>> nwbfile = NWBFile(
        ...     session_description="Test session",
        ...     identifier="test_001",
        ...     session_start_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
        ... )
        >>> env.to_nwb(nwbfile, name="open_field")
        >>>
        >>> # Write to disk (use a temporary directory, not the working directory)
        >>> path = Path(tempfile.mkdtemp()) / "session.nwb"
        >>> with NWBHDF5IO(path, "w") as io:
        ...     io.write(nwbfile)

        Load environment back:

        >>> with NWBHDF5IO(path, "r") as io:
        ...     loaded_env = Environment.from_nwb(io.read(), name="open_field")
        >>> np.array_equal(loaded_env.bin_centers, env.bin_centers)
        True
        >>> loaded_env.units
        'cm'

        """
        from neurospatial.io.nwb import write_environment

        write_environment(nwbfile, self, name=name, overwrite=overwrite)

    def bins_in_region(self, region_name: str) -> NDArray[np.intp]:
        """Return the indices of the bins a named region covers.

        A point region covers the one bin that contains the point. A polygon
        region covers the bins whose centers lie inside it. A center exactly
        on the polygon's edge is covered when nudging it a hair north-east
        would move it inside: polygons that share an edge, such as the
        quadrants of an open field, split the bins on it, so no bin is in
        two of them, and a rectangle holds the centers on its south and west
        edges but not those on its north and east edges, as a bin ``[a, b)``
        holds ``a``. This is the same
        rule as :meth:`region_mask`, :meth:`region_membership`,
        ``distance_to(region=...)`` and ``subset(region_names=...)``.

        Parameters
        ----------
        region_name : str
            The name of a region in ``self.regions``.

        Returns
        -------
        NDArray[np.intp]
            Sorted active-bin indices in the region. Empty when the region
            covers no bin, for example a point outside the environment.

        Raises
        ------
        KeyError
            If ``region_name`` is not found in ``self.regions``.
        ValueError
            If a point region has a different number of coordinates than the
            bins, or a polygon region is used in an environment that is not
            2-D.

        Notes
        -----
        On an egocentric polar environment, a polygon region is read in the
        environment's own ``(distance, angle)`` coordinates, as ``bin_at``
        reads positions: x is distance and y is angle in radians, so a box is
        a wedge of a distance band. Angles are compared modulo a full turn,
        as for positions: a bin is covered when its center angle plus any
        whole number of turns lies inside the polygon, so a wedge across the
        ±π seam, such as angles 3π/4 to 5π/4 behind the animal, is one
        polygon.

        Examples
        --------
        >>> import numpy as np
        >>> from shapely.geometry import box
        >>> from neurospatial import Environment
        >>> from neurospatial.regions import Regions
        >>> edges = np.arange(0.0, 11.0, 2.0)  # bin centers 1, 3, 5, 7, 9
        >>> env = Environment.from_grid_mask(
        ...     np.ones((5, 5), dtype=bool), (edges, edges)
        ... )
        >>> regions = Regions()
        >>> _ = regions.add("reward", point=(5.2, 4.9))
        >>> _ = regions.add("roi", polygon=box(3, 3, 7, 7))
        >>> env = env.with_regions(regions)
        >>> env.bin_centers[env.bins_in_region("reward")]
        array([[5., 5.]])

        Centers lie on every edge of the box: those on its south and west
        edges (3) are in it, those on its north and east edges (7) are not.

        >>> env.bin_centers[env.bins_in_region("roi")]
        array([[3., 3.],
               [3., 5.],
               [5., 3.],
               [5., 5.]])

        See Also
        --------
        region_mask : Boolean mask over bins for one or more regions.
        region_membership : Bins-by-regions membership table.

        """
        return _bins_in_region(
            region_name,
            self.regions,
            binning=self._binning,
            arena=self._arena,
            cache=self._cache,
        )

    def region_membership(
        self,
        regions: Mapping[str, Region] | None = None,
    ) -> NDArray[np.bool_]:
        """Check which bins belong to which regions.

        This method performs vectorized containment checks to determine which
        bins are inside each region. Useful for:
        - Filtering bins by region
        - Computing region-specific statistics
        - Identifying spatial distributions across regions
        - Selecting bins for subset operations

        Parameters
        ----------
        regions : mapping of str to Region, optional
            Regions to test against, such as a Regions builder. If None
            (default), uses self.regions.
            Allows testing against external region sets without modifying
            the environment.

        Returns
        -------
        membership : NDArray[np.bool_], shape (n_bins, n_regions)
            Boolean array where membership[i, j] = True if bin i is in region j.
            Columns are ordered according to region iteration order.
            If regions is empty, returns array with shape (n_bins, 0).

        Raises
        ------
        TypeError
            If regions is not a mapping of names to Region values or None.

        See Also
        --------
        subset : Create new environment from bin selection.
        bins_in_region : Get bin indices for a specific region.

        Notes
        -----
        **Region Types**:
        - Polygon regions: bins whose centers lie inside. A center exactly on
          the edge is inside when nudging it a hair north-east would move it
          inside, so regions that share an edge split its bins.
        - Point regions: the one bin that contains the point.

        **Polar Environments**:
        On an egocentric polar environment, a polygon region is read in the
        environment's own ``(distance, angle)`` coordinates, as ``bin_at``
        reads positions: x is distance and y is angle in radians, so a box is
        a wedge of a distance band. Angles are compared modulo a full turn,
        as for positions: a bin is covered when its center angle plus any
        whole number of turns lies inside the polygon, so a wedge across the
        ±π seam, such as angles 3π/4 to 5π/4 behind the animal, is one
        polygon.

        This is the same rule as :meth:`bins_in_region`, :meth:`region_mask`
        and ``distance_to(region=...)``.

        **Performance**:
        This method uses vectorized Shapely operations for efficiency.
        For N bins and R regions, complexity is O(N * R), but vectorized
        operations make it fast even for thousands of bins.

        **Region Order**:
        The column order in the output array matches the iteration order of
        the regions mapping. For self.regions, this is insertion order.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from shapely.geometry import box
        >>> data = np.array([[i, j] for i in range(11) for j in range(11)])
        >>> env = Environment.from_samples(data, bin_size=2.0)
        >>> from neurospatial.regions import Regions
        >>> regions = Regions(env.regions.values())
        >>> _ = regions.add("west", polygon=box(-1, -1, 6, 11))
        >>> _ = regions.add("east", polygon=box(6, -1, 11, 11))
        >>> env = env.with_regions(regions)
        >>> membership = env.region_membership()
        >>> membership.shape[1]
        2
        >>> membership.dtype
        dtype('bool')

        Each column counts the bins of one region. The column of bin centers
        at x = 6 lies on the shared edge and goes to the east half, so every
        bin is in exactly one half:

        >>> membership.sum(axis=0)
        array([18, 18])
        >>> bool(np.all(membership.sum(axis=1) == 1))
        True

        Regions that are not attached to the environment can be tested too:

        >>> external = Regions()
        >>> _ = external.add("test", polygon=box(2, 2, 8, 8))
        >>> env.region_membership(regions=external).shape
        (36, 1)

        """
        return _region_membership(
            self.regions if regions is None else regions,
            binning=self._binning,
            arena=self._arena,
            cache=self._cache,
        )

    def region_mask(
        self,
        regions: str | list[str | Region] | Region | Mapping[str, Region],
    ) -> NDArray[np.bool_]:
        """Boolean mask over bins for one or more regions.

        A point region covers the one bin that contains the point. A polygon
        region covers the bins whose centers lie inside it. A center exactly
        on the polygon's edge is covered when nudging it a hair north-east
        would move it inside: polygons that share an edge, such as the
        quadrants of an open field, split the bins on it, so no bin is in
        two of them, and a rectangle holds the centers on its south and west
        edges but not those on its north and east edges, as a bin ``[a, b)``
        holds ``a``. This is the same
        rule as :meth:`bins_in_region`, :meth:`region_membership`,
        ``distance_to(region=...)`` and
        :func:`neurospatial.ops.mapping.regions_to_mask`.

        Parameters
        ----------
        regions : str, list of str or Region, Region, or mapping of str to Region
            Region(s) to select. A name refers to an entry in ``self.regions``;
            Region objects, and the values of a mapping such as a Regions
            builder, are used directly.

        Returns
        -------
        mask : NDArray[np.bool_], shape (n_bins,)
            True for each bin covered by any of the regions (their union).

        Raises
        ------
        RegionNotFoundError
            If a region name is not in ``self.regions`` (a ``KeyError``).
        TypeError
            If ``regions`` has an invalid type.

        Notes
        -----
        A list of regions gives their union. For an intersection or a
        difference, combine separate masks with ``&`` and ``~`` (see the last
        example).

        On an egocentric polar environment, a polygon region is read in the
        environment's own ``(distance, angle)`` coordinates, as ``bin_at``
        reads positions: x is distance and y is angle in radians, so a box is
        a wedge of a distance band. Angles are compared modulo a full turn,
        as for positions: a bin is covered when its center angle plus any
        whole number of turns lies inside the polygon, so a wedge across the
        ±π seam, such as angles 3π/4 to 5π/4 behind the animal, is one
        polygon.

        Examples
        --------
        >>> import numpy as np
        >>> from shapely.geometry import box
        >>> from neurospatial import Environment
        >>> from neurospatial.regions import Regions
        >>> data = np.array([[i, j] for i in range(11) for j in range(11)])
        >>> env = Environment.from_samples(data, bin_size=2.0)
        >>> regions = Regions()
        >>> _ = regions.add("center", polygon=box(3, 3, 7, 7))
        >>> _ = regions.add("reward", point=(9.0, 9.0))
        >>> env = env.with_regions(regions)
        >>> int(env.region_mask("center").sum())
        4
        >>> int(env.region_mask("reward").sum())  # the bin containing the point
        1
        >>> int(env.region_mask(["center", "reward"]).sum())
        5
        >>> center, reward = env.region_mask("center"), env.region_mask("reward")
        >>> int((center & reward).sum()), int((center & ~reward).sum())
        (0, 4)

        See Also
        --------
        region_membership : Bins-by-regions membership table.
        bins_in_region : Bin indices for one named region.
        neurospatial.ops.mapping.regions_to_mask : Free-function form.
        """
        return _regions_mask(
            regions,
            named_regions=self.regions,
            binning=self._binning,
            arena=self._arena,
        )

    def _plot_labels(self) -> PlotLabels:
        """Names the plots show, which the binning does not carry."""
        return PlotLabels(
            name=self.name,
            layout_type=self.layout_type,
            units=self.units,
            coordinates=self._geometry.coordinates,
        )

    @overload
    def plot(
        self,
        ax: matplotlib.axes.Axes | None = None,
        *,
        show_regions: bool = True,
        show_walls: bool = True,
        show_landmarks: bool = True,
        layout_plot_kwargs: dict[str, Any] | None = None,
        regions_plot_kwargs: dict[str, Any] | None = None,
        scale_bar: bool | ScaleBarConfig = False,
        backend: Literal["matplotlib"] = "matplotlib",
        **kwargs: Any,
    ) -> matplotlib.axes.Axes: ...

    @overload
    def plot(
        self,
        ax: None = None,
        *,
        show_regions: bool = True,
        show_walls: bool = True,
        show_landmarks: bool = True,
        backend: Literal["fastplotlib"],
    ) -> fastplotlib.Figure: ...

    @keyword_hint(
        "show_regions",
        "show_walls",
        "show_landmarks",
        "layout_plot_kwargs",
        "regions_plot_kwargs",
        "scale_bar",
        "backend",
    )
    def plot(
        self,
        ax: matplotlib.axes.Axes | None = None,
        *,
        show_regions: bool = True,
        show_walls: bool = True,
        show_landmarks: bool = True,
        layout_plot_kwargs: dict[str, Any] | None = None,
        regions_plot_kwargs: dict[str, Any] | None = None,
        scale_bar: bool | ScaleBarConfig = False,
        backend: Literal["matplotlib", "fastplotlib"] = "matplotlib",
        **kwargs: Any,
    ) -> matplotlib.axes.Axes | fastplotlib.Figure:
        """Plot the bins with everything specified in the environment.

        Draws the bins and the connections between them, then the declared
        walls, the named landmarks and the regions, so one figure shows
        whether the environment matches the apparatus. The axes widen to
        show landmarks outside the arena, such as a distal cue.

        With ``backend="fastplotlib"`` the same picture is drawn on the GPU in
        a figure you can pan, zoom and, for a 3-D environment, rotate.

        Parameters
        ----------
        ax : Optional[matplotlib.axes.Axes], optional
            The Matplotlib axes to plot on. If None, a new figure and axes
            are created. Defaults to None. Matplotlib only.
        show_regions : bool, default=True
            Draw the regions in ``env.regions``, with a legend.
        show_walls : bool, default=True
            Draw the declared walls in ``env.arena.walls``, labeled with their
            names.
        show_landmarks : bool, default=True
            Draw the landmarks in ``env.landmarks``, labeled with their names.
        layout_plot_kwargs : Optional[Dict[str, Any]], optional
            Keyword arguments to pass to the `layout.plot()` method.
            Defaults to None.
        regions_plot_kwargs : Optional[Dict[str, Any]], optional
            Keyword arguments to pass to the `regions.plot_regions()` method.
            Defaults to None.
        scale_bar : bool or ScaleBarConfig, default=False
            Whether to add a scale bar showing physical scale. If True, uses
            auto-calculated length based on environment extent. If ScaleBarConfig,
            uses the specified configuration. Default: False (no scale bar).
        backend : {"matplotlib", "fastplotlib"}, default="matplotlib"
            ``"fastplotlib"`` draws on the GPU (``pip install
            "neurospatial[fastplotlib]"``, Python 3.11 or newer). Inside
            Jupyter it renders on the machine running the kernel and streams
            to the browser, so it works on a remote GPU server; elsewhere it
            opens a window. Call ``.show()`` on the returned figure.
            ``layout_plot_kwargs``, ``regions_plot_kwargs``, ``scale_bar``
            and ``**kwargs`` are Matplotlib only.
        **kwargs : Any
            Additional keyword arguments that are passed to `layout.plot()`.
            These can be overridden by `layout_plot_kwargs`.

        Returns
        -------
        matplotlib.axes.Axes or fastplotlib.Figure
            The axes on which the environment was plotted, or with
            ``backend="fastplotlib"`` the figure.

        Raises
        ------
        TypeError
            If a Matplotlib-only option is passed with ``backend="fastplotlib"``.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> x, y = np.meshgrid(np.arange(0.0, 10.0), np.arange(0.0, 10.0))
        >>> samples = np.c_[x.ravel(), y.ravel()]
        >>> env = Environment.from_samples(samples, bin_size=2.0, units="cm")
        >>> ax = env.plot()
        >>> ax.get_xlabel()
        'X Position (cm)'

        Plot with regions:

        >>> from neurospatial.regions import Regions
        >>> regions = Regions(env.regions.values())
        >>> _ = regions.add("goal", point=[5.0, 5.0])
        >>> env = env.with_regions(regions)
        >>> ax = env.plot()
        >>> ax.get_legend_handles_labels()[1]
        ['goal']
        >>> plt.close("all")

        On the GPU, for example from Jupyter on a remote server:

        >>> env.plot(backend="fastplotlib").show()  # doctest: +SKIP

        """
        if _check_backend(backend) == "fastplotlib":
            from neurospatial.environment import _fastplotlib

            _fastplotlib.refuse_matplotlib_options(
                "plot",
                kwargs,
                ax=ax is not None,
                layout_plot_kwargs=layout_plot_kwargs is not None,
                regions_plot_kwargs=regions_plot_kwargs is not None,
                scale_bar=scale_bar is not False,
            )
            return _fastplotlib.plot_environment(
                self,
                show_regions=show_regions,
                show_walls=show_walls,
                show_landmarks=show_landmarks,
            )
        return _plot(
            self._binning,
            self.connectivity,
            self._regions,
            self._plot_labels(),
            ax,
            show_regions,
            self.arena if show_walls else None,
            self.landmarks if show_landmarks else None,
            layout_plot_kwargs,
            regions_plot_kwargs,
            scale_bar,
            **kwargs,
        )

    @keyword_hint("layout_plot_kwargs")
    def plot_1d(
        self,
        ax: matplotlib.axes.Axes | None = None,
        *,
        layout_plot_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> matplotlib.axes.Axes | None:
        """Plot a 1D representation of the environment, if applicable.

        For track environments (``is_linearized_track`` is True, as built by
        ``from_graph``, ``linear_track`` and ``maze``), this draws the bins
        along the linearized track coordinate.

        Parameters
        ----------
        ax : Optional[matplotlib.axes.Axes], optional
            The Matplotlib axes to plot on. If None, a new figure and axes
            are created. Defaults to None.
        layout_plot_kwargs : Optional[Dict[str, Any]], optional
            Keyword arguments to pass to the layout's 1D plotting method.
        **kwargs : Any
            Additional keyword arguments passed to the layout's 1D plotting method.

        Returns
        -------
        matplotlib.axes.Axes | None
            The axes on which the 1D layout was plotted, or the original `ax`
            (which may be None) if plotting was not applicable.

        Warns
        -----
        UserWarning
            If the environment is not a track (``is_linearized_track`` is
            False); nothing is drawn and ``ax`` is returned unchanged.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> from neurospatial import Environment
        >>> env = Environment.linear_track(
        ...     endpoints=[[0.0, 0.0], [100.0, 0.0]], bin_size=5.0, units="cm"
        ... )
        >>> ax = env.plot_1d()
        >>> ax.get_xlabel()
        'Linearized Position'
        >>> plt.close("all")

        """
        return _plot_1d(
            self._binning, self.layout_type, ax, layout_plot_kwargs, **kwargs
        )

    @overload
    def plot_field(
        self,
        field: NDArray[np.float64],
        ax: matplotlib.axes.Axes | None = None,
        *,
        cmap: str | Colormap = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
        colorbar: bool = True,
        colorbar_label: str = "",
        nan_color: str | None = "lightgray",
        rasterized: bool = True,
        scale_bar: bool | ScaleBarConfig = False,
        backend: Literal["matplotlib"] = "matplotlib",
        **kwargs: Any,
    ) -> matplotlib.axes.Axes: ...

    @overload
    def plot_field(
        self,
        field: NDArray[np.float64],
        ax: None = None,
        *,
        cmap: str | Colormap = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
        colorbar: bool = True,
        colorbar_label: str = "",
        nan_color: str | None = "lightgray",
        backend: Literal["fastplotlib"],
    ) -> fastplotlib.Figure: ...

    @keyword_hint(
        "cmap",
        "vmin",
        "vmax",
        "colorbar",
        "colorbar_label",
        "nan_color",
        "rasterized",
        "scale_bar",
        "backend",
    )
    def plot_field(
        self,
        field: NDArray[np.float64],
        ax: matplotlib.axes.Axes | None = None,
        *,
        cmap: str | Colormap = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
        colorbar: bool = True,
        colorbar_label: str = "",
        nan_color: str | None = "lightgray",
        rasterized: bool = True,
        scale_bar: bool | ScaleBarConfig = False,
        backend: Literal["matplotlib", "fastplotlib"] = "matplotlib",
        **kwargs: Any,
    ) -> matplotlib.axes.Axes | fastplotlib.Figure:
        """Plot spatial field data over the environment.

        Automatically selects the appropriate visualization method based on the
        layout type to render bins with their actual geometric shapes:

        - **1D native grids**: Line plot over physical bin coordinates
        - **2D grid layouts**: Uses ``pcolormesh`` for crisp rectangular bins
        - **Hexagonal**: Colored hexagons in one ``PolyCollection``
        - **Triangular mesh**: Colored triangle faces via ``tripcolor``
        - **1D tracks**: Line plot over physical linearized coordinates, with
          breaks at missing values and artificial gaps between track segments
        - **Other**: Scatter plot with markers sized in display points; override
          their area with ``s=`` (in points squared, as in Matplotlib).

        Parameters
        ----------
        field : NDArray[np.float64], shape (n_bins,)
            Spatial field values for each bin (e.g., firing rate, probability,
            decoded position density).
        ax : Optional[matplotlib.axes.Axes], optional
            Axes to plot on. If None, creates new figure and axes. Default: None.
        cmap : str or matplotlib.colors.Colormap, default="viridis"
            Matplotlib colormap name (e.g., "hot", "Blues", "viridis").
        vmin : Optional[float], optional
            Minimum value for colormap normalization. If None, uses
            ``np.nanmin(field)``. Default: None.
        vmax : Optional[float], optional
            Maximum value for colormap normalization. If None, uses
            ``np.nanmax(field)``. Default: None.
        colorbar : bool, default=True
            Whether to add a colorbar. Not applicable for 1D plots.
        colorbar_label : str, default=""
            Label text for the colorbar.
        nan_color : Optional[str], default="lightgray"
            Color for bins with NaN values. If None, NaN bins are not rendered
            (skipped). Use "lightgray", "white", etc.
        rasterized : bool, default=True
            Rasterize output for better performance and smaller file sizes with
            large grids. Recommended for environments with >1000 bins.
        scale_bar : bool or ScaleBarConfig, default=False
            Whether to add a scale bar showing physical scale. If True, uses
            auto-calculated length based on environment extent. If ScaleBarConfig,
            uses the specified configuration. Default: False (no scale bar).
        backend : {"matplotlib", "fastplotlib"}, default="matplotlib"
            ``"fastplotlib"`` draws on the GPU (``pip install
            "neurospatial[fastplotlib]"``, Python 3.11 or newer): each bin's
            own cell, or for tracks, points and 3-D environments a point the
            size of a bin, with walls and landmarks on top. Inside Jupyter it
            renders on the machine running the kernel and streams to the
            browser, so it works on a remote GPU server. Call ``.show()`` on
            the returned figure. ``ax``, ``rasterized``, ``scale_bar`` and
            ``**kwargs`` are Matplotlib only.
        **kwargs : Any
            Additional keyword arguments passed to underlying plot function
            (e.g., ``pcolormesh``, ``PatchCollection.set``).

        Returns
        -------
        matplotlib.axes.Axes or fastplotlib.Figure
            The axes on which the field was plotted, or with
            ``backend="fastplotlib"`` the figure.

        Raises
        ------
        ValueError
            If ``field`` shape does not match ``(n_bins,)``.
        TypeError
            If a Matplotlib-only option is passed with ``backend="fastplotlib"``.
        NotImplementedError
            With Matplotlib, if environment dimensionality is >2D; use
            ``backend="fastplotlib"`` to draw a 3-D field.
        NotImplementedError
            If attempting to use ``pcolormesh`` on non-2D grid.

        See Also
        --------
        compute_spatial_rate : Compute firing rate maps from spike trains.
        plot : Plot environment structure (bins and connectivity).
        plot_1d : Plot 1D linearized environment.

        Notes
        -----
        **NaN Handling:**

        - ``nan_color=None``: NaN bins are skipped (not rendered)
        - ``nan_color="lightgray"``: NaN bins rendered with specified color
        - If all field values are NaN and ``nan_color=None``, an empty plot is
          produced (no colorbar will be added).

        **Performance:**

        For large environments (>10,000 bins), set ``rasterized=True`` (default)
        to improve rendering performance and reduce file size.

        **Colormaps:**

        Common choices for neuroscience:

        - ``"hot"``: Firing rate maps (black → red → yellow)
        - ``"viridis"``: General purpose, perceptually uniform
        - ``"Blues"``: Probability distributions
        - ``"RdBu_r"``: Diverging data (e.g., correlation)

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> x, y = np.meshgrid([0.0, 2.0, 4.0], [0.0, 2.0, 4.0])
        >>> env = Environment.from_samples(np.c_[x.ravel(), y.ravel()], bin_size=2.0)
        >>> import matplotlib.pyplot as plt
        >>> field = np.arange(env.n_bins, dtype=float)
        >>> fig, ax = plt.subplots()
        >>> _ = env.plot_field(field, ax=ax)
        >>> fig.savefig("field.png")

        On the GPU, for example from Jupyter on a remote server:

        >>> env.plot_field(field, backend="fastplotlib").show()  # doctest: +SKIP

        """
        check_field(field, self.n_bins)
        if _check_backend(backend) == "fastplotlib":
            from neurospatial.environment import _fastplotlib

            _fastplotlib.refuse_matplotlib_options(
                "plot_field",
                kwargs,
                ax=ax is not None,
                rasterized=rasterized is not True,
                scale_bar=scale_bar is not False,
            )
            return _fastplotlib.plot_field(
                self,
                np.asarray(field, dtype=np.float64),
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                colorbar=colorbar,
                colorbar_label=colorbar_label,
                nan_color=nan_color,
            )
        return _plot_field(
            self._binning,
            self._plot_labels(),
            field,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            colorbar=colorbar,
            colorbar_label=colorbar_label,
            nan_color=nan_color,
            rasterized=rasterized,
            scale_bar=scale_bar,
            **kwargs,
        )

    def animate_fields(
        self,
        fields: Sequence[NDArray[np.float64]] | NDArray[np.float64] | Mapping[str, Any],
        *,
        frame_times: NDArray[np.float64],
        backend: Literal[
            "auto", "napari", "video", "html", "widget", "fastplotlib"
        ] = "auto",
        save_path: str | Path | None = None,
        overwrite: bool = False,
        overlays: list[OverlayProtocol] | None = None,
        speed: float = 1.0,
        max_playback_fps: int = 60,
        max_gap: float | None = DEFAULT_MAX_GAP,
        cmap: str = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
        title: str | None = None,
        show_regions: bool | list[str] = False,
        region_alpha: float = 0.3,
        scale_bar: bool | ScaleBarConfig | None = None,
        **kwargs: Any,
    ) -> AnimationView | Path:
        """Animate spatial fields on their recording clock.

        Shows how a field such as a decoded position posterior changes over
        time, with the animal's position, pose, events and signals drawn on
        it. Each row of ``fields`` is shown from its time in ``frame_times``
        until the next row; across a gap longer than ``max_gap`` the field
        shows as missing. The call builds an
        :class:`~neurospatial.animation.Animation` and opens a view of it or
        writes a file. Build an ``Animation`` yourself to inspect identified
        events, open several views, or export selected time windows.

        Parameters
        ----------
        fields : ndarray, shape (n_frames, n_bins), or sequence or mapping
            Field values, one row per entry of ``frame_times``. A list of
            ``(n_bins,)`` arrays is one field. Several fields shown side by
            side are a mapping from name to field, or a list of fields named
            by ``layer_names``. Rows are read lazily, so a memory-mapped array
            stays on disk.
        frame_times : ndarray, shape (n_frames,)
            Strictly increasing time of each row, in seconds.
        backend : {"auto", "napari", "video", "html", "widget", "fastplotlib"}
            ``"video"`` and ``"html"`` write ``save_path``. ``"napari"``
            opens a desktop viewer, ``"widget"`` a notebook player and
            ``"fastplotlib"`` a GPU view in Jupyter or a window
            (``pip install "neurospatial[fastplotlib]"``). ``"auto"`` exports
            when ``save_path`` is a video or ``.html`` file, uses the notebook
            widget in Jupyter for up to 10,000 frames, and napari otherwise.
        save_path : str or Path, optional
            Output file: ``.mp4``, ``.webm``, ``.mov``, ``.avi``, ``.gif`` or
            ``.html``.
        overwrite : bool, default=False
            Allow replacing an existing output file.
        overlays : list of overlays, optional
            Positions, pose, head direction, video, events, signals, spike
            rasters, epochs and object vectors, each on its own clock.
        speed : float, default=1.0
            Playback speed relative to real time: 0.1 plays replay at a tenth
            of its recorded speed.
        max_playback_fps : int, default=60
            Highest frame rate shown or written. The frame rate is the
            sampling rate of ``frame_times`` times ``speed``; a file capped
            here keeps ``speed`` and skips rows between output frames, with a
            warning.
        max_gap : float or None, default=1.0
            Longest interval, in seconds, over which a field row is held until
            the next one; across longer gaps the field shows as missing. Pass
            None to hold every row until the next, for example for one field per
            trial. The default is shared with the trajectory analyses, calibrated
            for typical video and DeepLabCut tracking; choose a value longer than
            the spacing of your rows and shorter than real pauses in the recording.
        cmap : str, default="viridis"
            Matplotlib colormap name.
        vmin, vmax : float, optional
            Fixed color limits. By default, robust limits from up to 100 rows
            of each field.
        title : str, optional
            Title shown above the fields.
        show_regions : bool or list of str, default=False
            Draw all environment regions, or the named ones.
        region_alpha : float, default=0.3
            Region opacity, between 0 and 1.
        scale_bar : bool or ScaleBarConfig, optional
            Draw a scale bar in the environment's units.
        **kwargs
            Further options. For every backend: ``colorbar_label``,
            ``layer_names``, ``window_seconds`` (seconds of signal shown
            around the current time) and ``track`` (a linearized track of the
            same maze, to draw a 2-D field along it). For files: ``dpi``,
            ``figsize``, ``n_workers`` (worker processes rendering frames;
            None uses every CPU) and, for video, ``codec``, ``crf``,
            ``preset``, ``bitrate``; for HTML, ``image_format``,
            ``max_html_frames``, ``embed``, ``frames_dir``. For views:
            ``dpi`` and ``figsize`` (widget); ``viewer``, ``show``,
            ``ndisplay`` and ``layout`` (napari); ``layout``,
            ``frame_labels``, ``field_rendering``, ``volume_threshold`` and
            ``slice_plane`` (fastplotlib). An option no backend uses raises
            ``TypeError``; one that only another backend uses warns.

        Returns
        -------
        AnimationView or Path
            The open view, whose ``native`` attribute is the napari viewer,
            notebook widget or fastplotlib animation, or the written file.
            Closing the view releases the video readers it opened.

        Raises
        ------
        FileExistsError
            If the output exists and ``overwrite`` is False.
        TypeError
            If an option is not an option of any backend.
        ValueError
            If the fields do not match the environment's bins or
            ``frame_times``, or the backend cannot draw an overlay.

        See Also
        --------
        neurospatial.animation.Animation : Reusable scenes for views, event
            inspection and export.
        plot_field : Plot one field.
        neurospatial.animation.subsample_frames : Thin long recordings.

        Examples
        --------
        A field that spreads from the arena center, sampled at 30 Hz:

        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> x, y = np.meshgrid(np.arange(0.0, 24.0, 4.0), np.arange(0.0, 24.0, 4.0))
        >>> env = Environment.from_samples(np.c_[x.ravel(), y.ravel()], bin_size=4.0)
        >>> distances = env.distance_to(bins=env.n_bins // 2)
        >>> fields = np.array(
        ...     [np.exp(-distances / (4.0 + frame)) for frame in range(8)]
        ... )
        >>> frame_times = np.arange(len(fields)) / 30.0

        Explore it in napari, or write a video at half speed. Napari needs a
        display, so these are not run here:

        >>> view = env.animate_fields(
        ...     fields, frame_times=frame_times, backend="napari"
        ... )  # doctest: +SKIP
        >>> env.animate_fields(
        ...     fields,
        ...     frame_times=frame_times,
        ...     backend="video",
        ...     save_path="learning.mp4",
        ...     speed=0.5,
        ... )  # doctest: +SKIP

        Share an HTML player, with the animal's position and heading:

        >>> from neurospatial.animation import HeadDirectionOverlay, PositionOverlay
        >>> trajectory = np.c_[np.linspace(2.0, 18.0, 8), np.full(8, 10.0)]
        >>> position = PositionOverlay(trajectory, times=frame_times, trail_length=3)
        >>> heading = HeadDirectionOverlay(np.zeros(8), times=frame_times)
        >>> path = env.animate_fields(
        ...     fields,
        ...     frame_times=frame_times,
        ...     overlays=[position, heading],
        ...     save_path="overlays.html",
        ... )
        >>> path.name
        'overlays.html'

        Compare two fields side by side in a notebook:

        >>> view = env.animate_fields(
        ...     {"early": fields, "late": fields[::-1]},
        ...     frame_times=frame_times,
        ...     backend="widget",
        ... )
        >>> view.seek(0.1)
        >>> view.close()
        """
        from neurospatial.animation.core import animate_fields

        return animate_fields(
            self,
            fields,
            frame_times=frame_times,
            backend=backend,
            save_path=save_path,
            overwrite=overwrite,
            overlays=overlays,
            speed=speed,
            max_playback_fps=max_playback_fps,
            max_gap=max_gap,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            title=title,
            show_regions=show_regions,
            region_alpha=region_alpha,
            scale_bar=scale_bar,
            **kwargs,
        )

    @property
    def boundary_bins(self) -> NDArray[np.intp]:
        """
        Identify boundary bins (bins on the edge of the active region).

        Boundary bins are active bins that have at least one inactive neighbor
        or are on the edge of the grid. Useful for identifying walls, barriers,
        or the perimeter of an arena.

        Returns
        -------
        NDArray[np.intp], shape (n_boundary_bins,)
            Array of bin IDs that are on the boundary.

        Raises
        ------
        NotImplementedError
            If the bins are measured by area or volume but carry no cell
            geometry (for example points passed to ``from_bins`` without
            shapes), so the outline is unknown.

        Notes
        -----
        - Detection is cached; each access returns an independent array.
        - For large environments (>100,000 bins), initial computation may take a few seconds
        - Grid and hexagonal bins use their cell lattice: a bin is on the
          boundary when an orthogonal grid neighbor, or any of a hexagon's six
          neighbors, is inactive or outside the lattice. Triangular meshes and
          exact polygon cells use the outline of their union: a bin is on the
          boundary when its cell shares an edge with it. Track bins are on the
          boundary at the track's dead ends (bins with fewer than two
          neighbors). A combined environment uses each component's rule.

        See Also
        --------
        Environment.neighbors : Get neighbors of a bin

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment

        A 50 x 50 cm arena of 10 cm bins: the outer ring of 16 bins is the
        boundary and the 9 bins inside it are not.

        >>> mask = np.ones((5, 5), dtype=bool)
        >>> grid_edges = (np.arange(6) * 10.0, np.arange(6) * 10.0)
        >>> env = Environment.from_grid_mask(mask, grid_edges)
        >>> boundary = env.boundary_bins
        >>> print(f"Found {len(boundary)} boundary bins of {env.n_bins}")
        Found 16 boundary bins of 25
        >>> center_bin = env.bin_at(np.array([[25.0, 25.0]]))[0]
        >>> bool(np.isin(center_bin, boundary))
        False

        """
        boundary: NDArray[np.intp] = self._boundary_bins_cached.copy()
        return boundary

    @property
    def linearization_properties(
        self,
    ) -> dict[str, Any]:
        """
        Get linearization metadata for track environments.

        Returns
        -------
        dict[str, Any]
            ``{"is_linearized_track": bool}``: True for a track built with
            ``from_graph``, ``linear_track`` or ``maze``. Each access returns
            an independent dictionary. The track geometry itself (graph, edge
            order, linear bin edges) is in ``env.binning.linearization``.

        See Also
        --------
        to_linear : Convert N-D positions to 1D linear coordinates
        linear_to_nd : Convert 1D linear coordinates to N-D positions
        Environment.from_graph : Create 1D linearized environment

        Examples
        --------
        >>> from neurospatial import Environment
        >>> env = Environment.linear_track(
        ...     endpoints=[[0.0, 0.0], [100.0, 0.0]], bin_size=5.0, units="cm"
        ... )
        >>> env.linearization_properties
        {'is_linearized_track': True}

        """
        return {"is_linearized_track": self._binning.linearization is not None}

    def get_bin_attributes(self) -> pd.DataFrame:
        """
        Extract bin (node) attributes as a pandas DataFrame.

        Returns a DataFrame where each row corresponds to a bin, with columns
        for bin attributes like positions, grid indices, and any custom metadata.

        Returns
        -------
        pd.DataFrame, shape (n_bins, n_attributes)
            DataFrame with bin attributes. Always includes:

            - `'source_grid_flat_index'`: Flat index in original grid
            - `'original_grid_nd_index'`: N-D grid index tuple
            - `'pos_dim0'`, `'pos_dim1'`, ...: Position coordinates

        Notes
        -----
        - Method (not property) since v0.4: the construction cost
          (one DataFrame allocation per bin) is now visible at the
          call site. Attribute extraction is cached privately; each call returns
          an independent DataFrame, including nested object values.
        - For large environments (>100,000 bins), the first call may
          take a few seconds.
        - Memory usage: ~8 bytes per attribute per bin (e.g., 10
          attributes × 10k bins = 800 KB).

        See Also
        --------
        get_edge_attributes : Extract edge attributes as a DataFrame.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> x, y = np.meshgrid(np.arange(0.0, 10.0), np.arange(0.0, 10.0))
        >>> env = Environment.from_samples(np.c_[x.ravel(), y.ravel()], bin_size=2.0)
        >>> df = env.get_bin_attributes()
        >>> print(df.columns.tolist())
        ['source_grid_flat_index', 'original_grid_nd_index', 'pos_dim0', 'pos_dim1']
        >>> print(df.shape)  # one row per bin
        (36, 4)

        """
        return _copy_attributes(self._bin_attributes_cached)

    @property
    def _bin_attributes_cached(self) -> pd.DataFrame:
        return self._cache.get(
            BIN_ATTRIBUTES, lambda: _bin_attributes(self.connectivity)
        )

    def get_edge_attributes(self) -> pd.DataFrame:
        """
        Extract edge attributes as a pandas DataFrame.

        Returns a DataFrame where each row corresponds to an edge (connection
        between bins), with columns for edge attributes like distances and
        vectors.

        Returns
        -------
        pd.DataFrame, shape (n_edges, n_attributes)
            DataFrame with edge attributes. Always includes:

            - `'source'`: Source bin ID
            - `'target'`: Target bin ID
            - `'distance'`: Stored navigation distance between bins
            - `'vector'`: Displacement vector tuple from source to target
              center, in length units. On an egocentric polar environment it
              is the displacement in the animal's plane (x ahead, y left),
              not a difference of ``(distance, angle)`` coordinates.
            - `'vector_dim0'`, `'vector_dim1'`, ...: Displacement components
            - `'edge_id'`: Unique edge ID
            - `'angle_2d'`: Direction of ``vector`` in radians, for 2D
              layouts (optional); egocentric on a polar environment
              (0 = ahead, +π/2 = left)

        Notes
        -----
        - Method (not property) since v0.4: edge-DataFrame allocation is
          O(n_edges) and visible at the call site. Attribute extraction is
          cached privately; each call returns an independent DataFrame,
          including nested object values.
        - For large environments (>100,000 edges), the first call may
          take a few seconds.
        - Memory usage: ~8 bytes per attribute per edge (e.g., 6
          attributes × 10k edges = 480 KB).

        See Also
        --------
        get_bin_attributes : Extract bin attributes as a DataFrame.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> x, y = np.meshgrid(np.arange(0.0, 10.0), np.arange(0.0, 10.0))
        >>> env = Environment.from_samples(np.c_[x.ravel(), y.ravel()], bin_size=2.0)
        >>> df = env.get_edge_attributes()
        >>> print(df.columns.tolist())
        ['distance', 'vector', 'edge_id', 'angle_2d', 'source', 'target', 'vector_dim0', 'vector_dim1']
        >>> print(df.shape)  # one row per edge
        (110, 8)

        """
        return _copy_attributes(self._edge_attributes_cached)

    @property
    def _edge_attributes_cached(self) -> pd.DataFrame:
        return self._cache.get(
            EDGE_ATTRIBUTES,
            lambda: _edge_attributes(
                self.connectivity, self._binning, self.coordinate_system
            ),
        )

    @keyword_hint("bandwidth", missing={"mode": _MODE_CHOICE})
    def compute_kernel(
        self,
        *,
        bandwidth: float,
        mode: KernelMode,
        cache: bool = True,
    ) -> NDArray[np.float64]:
        """Compute the finite-volume diffusion kernel for smoothing operations.

        Builds the dense heat operator ``exp(-t L)`` from this environment's
        cached layout geometry and cell volumes, so ``bandwidth`` is the true
        physical standard deviation (σ) of the smoothing on any supported
        layout, independent of bin size. :meth:`smooth` and :meth:`diffuse`
        apply the same operator without building this matrix.

        Parameters
        ----------
        bandwidth : float
            Smoothing bandwidth in physical units (the standard deviation σ of
            the diffusion), must be > 0. This is the true physical σ: a
            point source smoothed by this kernel has physical standard
            deviation ``bandwidth`` regardless of bin size.
        mode : {'transition', 'density', 'average'}
            Kernel orientation, chosen by what the field holds; there is no
            default, because the modes give different numbers for one field:

            - 'transition': ``Hᵀ`` — column-stochastic (``sum_i K[i, j] = 1``).
              ``K @ field`` conserves ``sum(field)``; use for **extensive**
              quantities (occupancy, spike counts, discrete probability *mass*).
            - 'density': ``H·M⁻¹`` — M-weighted columns integrate to 1
              (``sum_i M_i K[i, j] = 1``). Takes an **extensive** input (counts)
              and returns a **density** (KDE). Do NOT apply to an already
              intensive field (a rate map): on non-uniform bin volumes that
              divides by cell volume twice.
            - 'average': ``H`` — row-stochastic (``sum_j K[i, j] = 1``).
              ``K @ field`` averages an **intensive** field (a rate map or
              probability *density*), volume-unbiased on non-uniform ``M``. Not
              for discrete probability *mass* (use 'transition').
        cache : bool, default=True
            If True, cache the computed kernel for reuse. Subsequent calls
            with the same (bandwidth, mode) reuse the computation and return
            independent copies of the cached result.
            The environment keeps the most recently used kernels (eight by
            default); ``clear_cache()`` releases them.

        Returns
        -------
        kernel : NDArray[np.float64], shape (n_bins, n_bins)
            Independent, editable diffusion kernel matrix.

        Raises
        ------
        ValueError
            If bandwidth is not positive, or ``mode`` is not one of
            ``{'transition', 'density', 'average'}``.

        See Also
        --------
        neurospatial.ops.diffusion.diffusion_kernel :
            Lower-level function operating directly on an environment.
        neurospatial.ops.smoothing.compute_diffusion_kernels :
            Graph-level primitive with explicit face measures.

        Notes
        -----
        The kernel is the finite-volume heat operator ``H = exp(-t L)`` with
        ``t = σ² / 2`` and ``L = M⁻¹ (D − W)``, ``W[i, j] = A[i, j] / d[i, j]``
        (``A`` the shared-face measure, ``d`` the center distance, ``M`` the
        per-bin cell volumes). On any K-orthogonal layout ``L`` has the
        continuum limit ``−∇²``, so the smoothing width equals ``σ``.

        **Memory cost is O(n²).** The diffusion heat kernel ``exp(-t L)`` of a
        connected graph is *dense by construction* (every entry > 0), so the
        returned matrix occupies ``n_bins**2 * 8`` bytes of float64 memory —
        for example ``20000**2 * 8 / 1e9 ≈ 3.2 GB`` at 20,000 bins. This peak
        cannot be avoided while using the dense diffusion kernel. There is no
        hard limit: a ``UserWarning`` estimating the GB is issued above 3,000
        bins and the kernel is then built regardless of size. The matrix
        exponential is also O(n³) in time, so large environments are slow as
        well as memory-hungry.

        With ``cache=True``, another matrix of the same size remains in the
        private cache. The public matrix is an independent, editable export;
        use ``cache=False`` to avoid retaining this additional matrix.

        **To avoid this dense matrix entirely, use** :meth:`diffuse` (or
        :meth:`smooth`), which apply the same operator **matrix-free** via a
        cached truncated eigenbasis in ``O(n_bins * rank)`` memory — and the
        higher-level ``"diffusion_kde"`` / ``"binned"`` encoding methods route
        through it. ``compute_kernel`` is for callers that genuinely need the
        explicit ``(n, n)`` matrix. For a smaller matrix, increase ``bin_size``.

        The physical-σ guarantee assumes uniform bin spacing per axis (the
        standard grid, hex, polar-sector, graph, and mesh layouts).
        ``MaskedGridLayout.build`` rejects nonuniform ``grid_edges``.

        Examples
        --------
        >>> import numpy as np
        >>> x, y = np.meshgrid(np.linspace(0, 50, 26), np.linspace(0, 50, 26))
        >>> env = Environment.from_samples(
        ...     np.c_[x.ravel(), y.ravel()], bin_size=5.0, units="cm"
        ... )
        >>> kernel = env.compute_kernel(bandwidth=5.0, mode="density")
        >>> kernel.shape  # (n_bins, n_bins)
        (121, 121)
        >>> # Spread one spike in the center bin into a density (spikes / cm²)
        >>> spikes = np.zeros(env.n_bins)
        >>> spikes[env.n_bins // 2] = 1.0
        >>> density = kernel @ spikes
        >>> # Integrating the density over the bin areas recovers the spike
        >>> round(float((density * env.bin_sizes).sum()), 6)
        1.0

        """
        return _compute_kernel(
            self._cache, self._binning, bandwidth, mode=mode, use_cache=cache
        )

    @keyword_hint("bandwidth")
    def compute_gaussian_kernel(self, *, bandwidth: float) -> NDArray[np.float64]:
        """Return Gaussian weights between bin centers, ignoring walls.

        Parameters
        ----------
        bandwidth : float
            Finite positive standard deviation in environment length units.

        Returns
        -------
        ndarray, shape (n_bins, n_bins)
            Read-only weights ``exp(-distance**2 / (2 * bandwidth**2))``.
            These weights are not normalized by row or by cell measure.

        Notes
        -----
        Cartesian coordinates use Euclidean distances. Polar coordinates use
        radial differences and mean-radius angular arc lengths, with angular
        wrapping only for a circular coordinate system. Weights can cross
        physical walls; use diffuse() for boundary-aware smoothing.

        The dense matrix costs O(n_bins**2) memory. Repeated calls reuse an
        environment-owned bounded cache and return independent array headers
        without copying the matrix. Use .copy() for an editable matrix.
        clear_cache() releases the cached weights.
        """
        return _compute_gaussian_kernel(
            self._cache, self.bin_centers, self._geometry.coordinates, bandwidth
        )

    def get_diffusion_geometry(self) -> DiffusionGeometry:
        """Return read-only conductances, cell measures and diffusion components.

        Returns
        -------
        DiffusionGeometry
            Sparse symmetric conductances W, positive cell volumes, component
            count and labels in active-bin order. Components follow positive
            conductance, so navigation-only links do not transport a field.

        Notes
        -----
        The environment builds this geometry once and owns its private cache.
        Every export is a new sparse matrix and tuple over read-only views of
        the cached arrays, so writing to them raises, and replacing an
        exported matrix's arrays or changing an array's shape cannot change
        the environment. Reading the export does not copy O(n_edges) data.
        After clear_cache() it is rebuilt on demand.

        Raises
        ------
        NotImplementedError
            If the stored binning cannot supply numerical diffusion geometry.

        Warns
        -----
        UserWarning
            If the bins are a triangular mesh with more than 5% of interior
            edges skewed by over 30 degrees, where a smoothing bandwidth is
            only approximate. It is raised once per environment, by this
            method or by the first smoothing operation (``smooth``,
            ``diffuse``, ``compute_kernel``, ``get_mrf_basis``), and again
            after ``clear_cache()``.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> x, y = np.meshgrid(np.arange(0.5, 10.0), np.arange(0.5, 4.0))
        >>> positions = np.column_stack([x.ravel(), y.ravel()])
        >>> env = Environment.from_samples(positions, bin_size=2.0)
        >>> geometry = env.get_diffusion_geometry()
        >>> geometry.W.shape == (env.n_bins, env.n_bins)
        True
        >>> int(geometry.n_components)  # one connected open field
        1
        >>> geometry.volumes.shape == (env.n_bins,)
        True
        """
        return _diffusion_geometry_view(smoothing_geometry(self._cache, self._binning))

    @keyword_hint("bandwidth")
    def diffusion_accuracy(self, *, bandwidth: float) -> float:
        """Return the relative numerical noise floor of diffuse() at a bandwidth.

        Parameters
        ----------
        bandwidth : float
            The physical bandwidth passed to diffuse().

        Returns
        -------
        float
            Relative truncation tolerance, or float64 roundoff when a full
            basis was used. A bandwidth not yet resolved by diffuse() reports
            the more conservative truncation tolerance. This query does not
            build an eigenbasis.
        """
        return _diffusion_apply_rtol(
            diffusion_eigenbasis(self._cache), self.n_bins, bandwidth
        )

    @keyword_hint("bandwidth", missing={"mode": _MODE_CHOICE})
    def smooth(
        self,
        field: NDArray[np.float64],
        *,
        bandwidth: float,
        mode: KernelMode,
    ) -> NDArray[np.float64]:
        """Apply diffusion kernel smoothing to a field.

        This method smooths bin-valued fields using diffusion kernels computed
        via the graph Laplacian. It works uniformly across all layout types
        (grids, graphs, meshes) and respects the connectivity structure.

        Parameters
        ----------
        field : NDArray[np.float64], shape (n_bins,) or (n_bins, n_fields)
            Field values per bin to smooth, bins first as in :meth:`diffuse`;
            a 2-D batch smooths each column as one field. Population rate
            arrays are stored ``(n_units, n_bins)``, so pass
            ``rates.firing_rates.T`` and transpose the result back; an array
            with bins on the second axis is refused with that fix, and so is
            a rate result itself. NaN marks a missing bin, such as an
            unvisited bin of a rate map, and is accepted with
            ``mode='average'``; see Missing bins below. Must not contain Inf.
        bandwidth : float
            Smoothing bandwidth in physical units (σ). Controls the scale
            of spatial smoothing. Must be positive.
        mode : {'transition', 'density', 'average'}
            Smoothing mode, chosen by what the field holds. There is no
            default: the modes give different numbers for the same field, and
            only the caller knows whether it holds totals or values per bin.
            Use ``'average'`` for a firing-rate map.

            - 'transition': Mass-conserving smoothing of an **extensive**
              quantity (a total per bin). Total sum is preserved:
              ``smoothed.sum() == field.sum()``. Use for occupancy, spike
              counts, and discrete probability *mass* (a posterior summing to 1
              stays summing to 1).
            - 'density': Count → density (KDE). Takes an **extensive** input
              (counts / occupancy) and returns a density whose integral under
              bin volumes equals the input total: ``sum(smoothed * bin_sizes)
              == sum(field)`` (each kernel column integrates to 1 under bin
              volumes). Do **not** apply to an already **intensive** field (a
              rate map or probability *density*): on non-uniform bin volumes
              that divides by cell volume twice.
            - 'average': Volume-unbiased averaging of an **intensive** field (a
              rate map or probability *density*). Row-stochastic ``H``, so a
              constant field is preserved and there is no cell-volume bias on
              non-uniform ``M``. Not for discrete probability *mass* (use
              'transition').

        Returns
        -------
        smoothed : NDArray[np.float64], shape matching ``field``
            Smoothed field values. NaN where too little of a bin's smoothing
            weight falls on observed bins (see Missing bins below).

        Raises
        ------
        TypeError
            If field is a rate result rather than its array of rates.
        ValueError
            If field has wrong shape, wrong dimensionality, contains Inf,
            contains NaN with ``mode='transition'`` or ``'density'``,
            bandwidth is not positive, or mode is invalid.

        See Also
        --------
        compute_kernel : Compute the smoothing kernel explicitly.
        occupancy : Compute occupancy with optional smoothing.

        Notes
        -----
        The smoothing operation is:

        .. math::
            \\text{smoothed} = K \\cdot \\text{field}

        where :math:`K` is the finite-volume heat operator
        :math:`\\exp(-tL)`, :math:`t = \\sigma^2/2`, :math:`L = M^{-1}(D - W)`.
        It is applied **matrix-free** (see Implementation below), not formed
        explicitly.

        For mode='transition', mass is conserved:

        .. math::
            \\sum_i \\text{smoothed}_i = \\sum_i \\text{field}_i

        For mode='density', the kernel maps an extensive input (counts) to a
        density whose integral under bin volumes equals the input total
        (``sum(smoothed * bin_sizes) == sum(field)``); it must not be applied to
        an already-intensive field (see the ``mode`` parameter).

        For mode='average', the kernel is row-stochastic, so a constant field is
        preserved and an intensive field (rate map) is averaged without the
        cell-volume bias that ``density`` would introduce on non-uniform ``M``.

        ``bandwidth`` is the true physical standard deviation (σ) of the
        smoothing on any supported layout, independent of bin size.

        **Implementation.** Smoothing is applied matrix-free via
        :meth:`diffuse` (a cached truncated eigenbasis), so it never materializes
        the dense ``(n_bins, n_bins)`` kernel and scales to large/fine grids. The
        eigenbasis is cached on the environment and reused across bandwidths,
        modes, and calls.

        **Approximation contract.** ``smooth`` is a pure linear operator (signed
        fields are preserved) that reproduces the dense kernel to within a
        near-lossless truncation tolerance: the deviation is ≤ ``tol`` (default
        ``1e-6``) in the **M-weighted norm** relative to the dense output (the raw
        per-bin error carries a volume-conditioning factor ``κ(M)``, worst on
        polar ``r→0`` / skewed mesh). On a **non-negative** input this can leave
        tolerance-level negatives instead of the dense kernel's exact 0-floor —
        clip the result yourself if you need a strict floor. Mass conservation
        (``mode='transition'``) and constant-field preservation
        (``mode='average'``) hold **exactly** under truncation (the per-component
        null mode is always retained).

        Edge preservation: Smoothing respects graph connectivity. Mass does
        not leak between disconnected components (even under truncation, because
        the eigenbasis modes are component-local).

        **Missing bins.** NaN bins are missing observations, not zeros (see
        Missing data in the class Notes). In ``mode='average'``, with ``A`` the
        row-stochastic operator and ``m`` the observed mask, the result is
        ``A(field * m) / s``, with NaN read as 0 in the product, where
        ``s = A(m)`` is the share of each bin's smoothing weight that falls on
        observed bins. Holes are filled from their observed neighbors, and a
        constant field with holes stays constant. A bin, observed or not,
        stays NaN when its ``s`` is at most 0.1% of the largest ``s`` in its
        connected part of the environment, which includes every bin of a part
        with no observed bin. The floor is 1000 times the operator's relative
        truncation error (``1e-6``). That error scales with the field's largest
        magnitude in the connected part, not with each value: a kept value is
        accurate to about 0.1% of that magnitude or better. For a field
        spanning 1000:1 (1000 Hz next to 1 Hz) a kept bin near 1 Hz can be off
        by about 1% of its own value, roughly 1e-5 of the 1000 Hz peak.
        ``mode='transition'`` and ``mode='density'`` refuse NaN:
        missing bins have no defensible mass under a conserving operator, so
        fill them explicitly or use ``mode='average'`` for an intensive field.
        A field without NaN gets the plain ``K · field``.

        Examples
        --------
        A 50 x 50 cm arena with 5 cm bins:

        >>> import numpy as np
        >>> x, y = np.meshgrid(np.linspace(0, 50, 26), np.linspace(0, 50, 26))
        >>> env = Environment.from_samples(
        ...     np.c_[x.ravel(), y.ravel()], bin_size=5.0, units="cm"
        ... )

        Smoothing a rate map with ``mode="average"`` keeps its units and
        level, so a flat 4 Hz map stays 4 Hz:

        >>> firing_rate = np.full(env.n_bins, 4.0)
        >>> smoothed = env.smooth(firing_rate, bandwidth=5.0, mode="average")
        >>> bool(np.allclose(smoothed, 4.0))
        True

        Smoothing spike counts with ``mode="transition"`` conserves the total:

        >>> spike_counts = np.zeros(env.n_bins)
        >>> spike_counts[[10, 60, 61]] = [3, 5, 2]
        >>> smoothed_counts = env.smooth(spike_counts, bandwidth=5.0, mode="transition")
        >>> bool(np.isclose(smoothed_counts.sum(), spike_counts.sum()))
        True

        ``mode="density"`` turns counts into spikes per cm²; integrating over
        the bin areas gives the total back:

        >>> spike_density = env.smooth(spike_counts, bandwidth=3.0, mode="density")
        >>> round(float((spike_density * env.bin_sizes).sum()), 6)
        10.0

        A discrete probability distribution is mass, so it also uses
        ``mode="transition"`` and still sums to 1:

        >>> posterior = np.full(env.n_bins, 1.0 / env.n_bins)
        >>> smoothed_prob = env.smooth(posterior, bandwidth=2.0, mode="transition")
        >>> round(float(smoothed_prob.sum()), 6)
        1.0

        """
        return _smooth(self._cache, self._binning, field, bandwidth, mode=mode)

    @keyword_hint("bandwidth", missing={"mode": _MODE_CHOICE})
    def diffuse(
        self,
        fields: Any,
        *,
        bandwidth: float,
        mode: KernelMode,
        backend: ResolvedBackend = "numpy",
    ) -> NDArray[np.float64]:
        """Matrix-free finite-volume diffusion smoothing (no ``(n, n)`` kernel).

        Applies the finite-volume heat operator ``H = exp(-t L)``, ``t = σ²/2``,
        ``L = M⁻¹(D − W)``, to one or more fields **without ever materializing
        the dense ``(n_bins, n_bins)`` kernel**. It uses a cached, per-component,
        bandwidth-aware **truncated symmetric eigenbasis** of
        ``S = M^{-1/2}(D − W) M^{-1/2}``, so both time and memory scale with
        ``n_bins × rank`` (``rank ~ measure(domain)/σ^d``), not ``n_bins²``. The
        eigenbasis depends only on geometry, so it is built once and reused across
        every bandwidth, mode, and neuron. Environments never change, and a
        derived environment starts with its own cache.

        On fields without NaN, ``diffuse`` is a **pure linear operator**: it
        applies **no output clip and no renormalization**. The always-retained per-component null mode
        makes mass conservation exact under truncation (``mode="transition"``
        preserves ``sum``; a constant field is preserved by ``mode="average"``),
        so smoothing a **signed** field is preserved. Where the result must be
        non-negative (a density, a rate), the **caller** clips it (as the KDE
        encoders do); ``diffuse`` itself never does.

        Parameters
        ----------
        fields : NDArray[np.float64], shape (n_bins,) or (n_bins, n_fields)
            Field(s) to smooth, indexed by bin first, as in :meth:`smooth`. A
            1-D field is smoothed and returned 1-D; a 2-D batch smooths every
            column in one pass. An array with bins on the second axis, such
            as population rates ``(n_units, n_bins)``, is refused with the
            fix ``fields.T``, and a rate result is refused with the array to
            pass. NaN
            marks a missing bin: in ``mode='average'`` each column is averaged
            over its own observed bins, as described under Missing bins in
            :meth:`smooth`, and columns without NaN are unchanged; the other
            modes refuse NaN. ``backend='jax'`` refuses a concrete input with
            NaN (use ``backend='numpy'``); a traced input cannot be inspected,
            so NaN in it reaches every bin. Inf is not validated;
            :meth:`smooth` refuses it.
        bandwidth : float
            Physical smoothing standard deviation σ (> 0), independent of bin
            size on any supported layout.
        mode : {'transition', 'density', 'average'}
            Kernel orientation, required as for :meth:`smooth`, matching
            :meth:`compute_kernel` / :meth:`smooth`
            (``'transition'`` = ``Hᵀ`` mass-conserving extensive smoothing;
            ``'density'`` = ``H·M⁻¹`` count→density; ``'average'`` = ``H``
            row-stochastic intensive averaging).
        backend : {'numpy', 'jax'}, default='numpy'
            Where the apply runs. ``'jax'`` casts the cached NumPy eigenbasis to
            ``jax.numpy`` and runs the apply on device, so ``jit`` / ``grad`` /
            GPU work through the smoothing; the returned array is a ``jax.Array``.
            The eigenbasis **build** always uses NumPy/SciPy.

        Returns
        -------
        smoothed : NDArray[np.float64], shape matching ``fields``
            The diffused field(s). ``numpy.ndarray`` for ``backend='numpy'``,
            ``jax.Array`` for ``backend='jax'``.

        Raises
        ------
        TypeError
            If ``fields`` is a rate result rather than its array of rates.
        ValueError
            If ``bandwidth`` is not positive, ``mode`` is invalid, ``backend`` is
            invalid, or ``fields`` is not 1-D/2-D with leading dim ``n_bins``;
            or if ``fields`` holds NaN with ``mode='transition'`` or
            ``'density'``, or as a concrete input with ``backend='jax'``.

        See Also
        --------
        smooth : Smoothing with input validation, default ``mode='average'``
            (delegates here).
        compute_kernel : Build the DENSE kernel matrix (kept for callers that
            genuinely need the ``(n, n)`` matrix; uses the dense ``expm`` path).

        Notes
        -----
        **Approximation contract.** ``diffuse`` reproduces the dense operator to
        within a near-lossless truncation tolerance: dropped modes contribute at
        most ``tol`` (default ``1e-6``) in the **M-weighted norm** (per-bin error
        carries a volume-conditioning factor ``κ(M) = sqrt(max vol / min vol)``,
        worst on polar ``r→0`` / skewed mesh). On a **non-negative** input the
        linear apply can therefore leave tolerance-level negatives (bounded
        relative to the dense output in the M-norm) instead of the dense kernel's
        exact 0-floor — clip the result yourself if you need a strict floor.

        A light bandwidth on a large grid needs most modes; there the eigenbasis
        cannot beat the dense kernel, so ``diffuse`` solves the full dense basis
        once, with the same large-matrix ``UserWarning`` as
        :meth:`compute_kernel`, and keeps it for later calls at any bandwidth.
        :meth:`clear_cache` releases it.
        """
        return _diffuse(
            self._cache, self._binning, fields, bandwidth, mode=mode, backend=backend
        )

    @property
    def _diffusion_geometry(self) -> DiffusionGeometry:
        """Cached diffusion geometry ``(W, volumes, n_components, labels)``.

        Shared by dense kernels, :meth:`diffuse`, smoothness bases, and the
        W-component support gates.
        """
        return diffusion_geometry(self._cache, self._binning)

    def get_mrf_basis(
        self,
        occupancy: NDArray[np.float64],
        *,
        rank: int | None = None,
    ) -> MRFBasis:
        """Build a reduced-rank Markov random field smoothness basis.

        Parameters
        ----------
        occupancy : ndarray, shape (n_bins,)
            Nonnegative exposure per active bin. Positive exposure selects
            the connected components retained in the basis.
        rank : int or None, optional
            Requested number of modes. None uses the default cap of 250;
            the result is also capped by the occupied support. Each retained
            component needs its constant mode, even when rank is smaller.

        Returns
        -------
        MRFBasis
            Independent editable arrays B, shape (n_live_bins, n_modes),
            penalty weights d, shape (n_modes,), and live_bins, which maps rows
            back to active bin IDs; also the number of retained components.

        Notes
        -----
        The environment owns and reuses the expensive geometry and global
        truncated eigenbasis. Requests reuse a cached basis when it is
        sufficient; near-full bases are rebuilt per call rather than retained.
        Returned arrays cannot change the cache. Unoccupied components
        contribute no basis rows; every occupied component has one unpenalized
        constant mode, followed by modes with positive smoothness penalties.
        """
        return _mrf_basis(self._cache, self._binning, occupancy, rank=rank)

    def transitions(
        self,
        bins: NDArray[np.intp] | None = None,
        *,
        times: NDArray[np.float64] | None = None,
        positions: NDArray[np.float64] | None = None,
        # Empirical parameters
        lag: int = 1,
        allow_teleports: bool = False,
        # Model-based parameters
        method: Literal["diffusion", "random_walk"] | None = None,
        bandwidth: float | None = None,
        # Common parameters
        normalize: bool = True,
        max_gap: float | None = DEFAULT_MAX_GAP,
        epochs: Any = None,
        break_before: ArrayLike | None = None,
    ) -> scipy.sparse.csr_matrix:
        """Compute transition matrix (empirical or model-based).

        Two modes of operation:

        1. **Empirical**: Count observed transitions from trajectory data.
           Requires bins OR (times + positions). Analyzes actual behavior.

        2. **Model-based**: Generate theoretical transitions from graph structure.
           Requires method parameter. Models expected behavior.

        Parameters
        ----------
        bins : NDArray[np.intp], shape (n_samples,), optional
            [Empirical mode] Precomputed bin sequence. If None, computed from
            times/positions. Must contain valid bin indices in range [0, n_bins).
            Outside values (-1) are not allowed.
        times : NDArray[np.float64], shape (n_samples,), optional
            [Empirical mode] Timestamps in seconds. Must be provided together
            with positions.
        positions : NDArray[np.float64], shape (n_samples, n_dims), optional
            [Empirical mode] Position coordinates matching environment dimensions.
            Must be provided together with times. A sample outside the
            environment or with a NaN coordinate (missing; see Missing data in
            the class Notes) adds no transition to or from it.
        lag : int, default=1
            [Empirical mode] Temporal lag for transitions: count bins[t] → bins[t+lag].
            Must be positive. lag=1 counts consecutive transitions, lag=2 skips one bin.
        allow_teleports : bool, default=False
            [Empirical mode] If False, only count transitions between graph-adjacent
            bins. Non-adjacent transitions (e.g., from tracking errors) are excluded.
            Self-transitions (staying in same bin) are always counted.
            If True, count all transitions including non-local jumps.
        method : {'diffusion', 'random_walk'}, optional
            [Model mode] Type of model-based transitions:
            - 'random_walk': Uniform transitions to graph neighbors
            - 'diffusion': Distance-weighted transitions via heat kernel
            If provided, empirical parameters (bins/times/positions/lag/allow_teleports)
            are ignored.
        bandwidth : float, optional
            [Model: diffusion] Diffusion bandwidth in physical units (σ).
            Required when method='diffusion'. Larger values produce more uniform
            transitions; smaller values emphasize local transitions.
        normalize : bool, default=True
            If True, return row-stochastic matrix where each row sums to 1
            (representing transition probabilities).
            If False, return raw counts (empirical) or unnormalized weights (model).
        max_gap : float or None, default=1.0
            Longest stretch of missing tracking, in seconds, to bridge: the time
            between consecutive valid samples (a dropped frame and a NaN row are
            the same missing sample). Longer gaps are excluded as tracking losses;
            ``None`` bridges every gap. The default is calibrated for typical video
            and DeepLabCut tracking; choose a value longer than routine dropped or
            low-confidence frames and shorter than real tracking losses or pickups.
        epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
            Restrict the analysis to these half-open [start, stop) windows (seconds,
            same clock as ``times``). The interval between two consecutive samples
            counts only if it lies entirely inside one window. ``None`` (default)
            means unrestricted.
        break_before : ArrayLike of bool, shape (n_samples,), optional
            ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
            ``k`` (for example where a new recording starts), whatever the elapsed
            time. Entry 0 is ignored.

        Returns
        -------
        T : scipy.sparse.csr_matrix, shape (n_bins, n_bins)
            Transition matrix where T[i,j] represents:
            - If normalize=True: P(next_bin=j | current_bin=i)
            - If normalize=False: count/weight of i→j transitions

            For normalized matrices, each row sums to 1.0 (rows with no
            transitions sum to 0.0).

        Raises
        ------
        ValueError
            If method is None and neither bins nor times/positions are provided.
            If method is provided together with empirical inputs (bins/times/positions).
            If method is provided together with empirical parameters (lag != 1 or allow_teleports != False).
            If method='random_walk' but bandwidth is provided.
            If method='diffusion' but bandwidth is not provided.
            If method='diffusion' but normalize=False (not supported).
            If bins contains invalid indices outside [0, n_bins).
            If lag is not positive (empirical mode).
        ValueError
            If ``break_before`` is not a Boolean array with one entry per sample
            (or per bin of ``bins``), or is given with ``method``.

        See Also
        --------
        bin_sequence : Convert trajectory to bin indices.
        occupancy : Compute time spent in each bin.
        compute_kernel : Low-level diffusion kernel computation.

        Notes
        -----
        Each run of samples with gaps no longer than ``max_gap`` (inside ``epochs``,
        not crossing a ``break_before``) is analyzed as a separate recording;
        no segment spans a pause.

        When times are supplied, a pair (k, k + lag) counts only if every
        interval from k through k + lag - 1 is valid and samples k and
        k + lag are both inside the environment. Raw bin-only inputs have no
        time gate, but a ``break_before`` with one entry per bin still drops
        every pair it separates. Model-based transitions take neither.

        **Empirical mode**: Counts observed transitions from trajectory data.
        When allow_teleports=False, filters out non-adjacent transitions using
        the connectivity graph. Useful for removing tracking errors.

        **Model mode**: Generates theoretical transition probabilities:

        - 'random_walk': Each bin transitions uniformly to all graph neighbors.
          Equivalent to normalized adjacency matrix.
        - 'diffusion': Transitions weighted by spatial proximity using heat kernel.
          Models continuous-time random walk with Gaussian steps.

        The sparse CSR format is memory-efficient for large environments
        where most bin pairs have no transitions.

        Examples
        --------
        A 50 x 50 cm arena with 5 cm bins and a 10 s run along its middle:

        >>> import numpy as np
        >>> x, y = np.meshgrid(np.linspace(0, 50, 26), np.linspace(0, 50, 26))
        >>> env = Environment.from_samples(
        ...     np.c_[x.ravel(), y.ravel()], bin_size=5.0, units="cm"
        ... )
        >>> times = np.arange(0.0, 10.0, 0.1)  # seconds
        >>> positions = np.c_[np.linspace(1.0, 49.0, 100), np.full(100, 25.0)]

        Empirical transitions counted from the trajectory, one row per bin:

        >>> T_empirical = env.transitions(times=times, positions=positions)
        >>> T_empirical.shape
        (121, 121)

        Transitions two samples apart, from an existing bin sequence:

        >>> bin_sequence = env.bin_sequence(times, positions)
        >>> T_lag2 = env.transitions(
        ...     bins=bin_sequence, lag=2, allow_teleports=True, normalize=False
        ... )
        >>> # One count per pair of visits two steps apart
        >>> int(T_lag2.sum()) == len(bin_sequence) - 2
        True

        Model transitions need no trajectory. Each row is a probability
        distribution over the next bin:

        >>> T_random = env.transitions(method="random_walk")
        >>> T_diffusion = env.transitions(method="diffusion", bandwidth=5.0)
        >>> bool(np.allclose(T_random.sum(axis=1), 1.0))
        True
        >>> bool(np.allclose(T_diffusion.sum(axis=1), 1.0))
        True
        >>> # Compare the observed movement with the diffusion model
        >>> difference = (T_empirical - T_diffusion).toarray()
        >>> difference.shape  # (from bin, to bin)
        (121, 121)

        """
        return _transitions(
            bins,
            graph=self.connectivity,
            binning=self._binning,
            arena=self._arena,
            cache=self._cache,
            times=times,
            positions=positions,
            lag=lag,
            allow_teleports=allow_teleports,
            method=method,
            bandwidth=bandwidth,
            normalize=normalize,
            max_gap=max_gap,
            epochs=epochs,
            break_before=break_before,
        )


# Each Cartesian-only member, collected from the @cartesian_only decorators.
CARTESIAN_OPERATIONS: frozenset[str] = cartesian_only_members(Environment)
