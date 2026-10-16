"""One public Environment with owned geometry and thin domain entry points."""

from __future__ import annotations

import copy as copy_module
import logging
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import replace
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
)

import networkx as nx
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse
from typing_extensions import Self

from neurospatial._exceptions import _format_error
from neurospatial._stacklevel import _external_stacklevel
from neurospatial.environment._publication import (
    UNSET,
    _freeze_geometry_arrays,
    _GeometryView,
    _Metadata,
    _MetadataView,
    _Provenance,
    _PublishedGeometry,
    _Unset,
)
from neurospatial.environment._rebin import _rebin_factor, _rebin_resolution
from neurospatial.environment.cartesian import (
    CARTESIAN_GRID_OPERATIONS,
    CARTESIAN_OPERATIONS,
)
from neurospatial.environment.fields import _interpolate
from neurospatial.environment.metrics import (
    _cartesian_boundary_bins,
)
from neurospatial.environment.trajectory import (
    BinSequenceWithRuns,
    _bin_sequence,
    _occupancy,
)
from neurospatial.geometry._binning import Binning
from neurospatial.geometry._coordinates import (
    CartesianSystem,
    EgocentricPolarSystem,
)
from neurospatial.layout.base import LayoutEngine
from neurospatial.layout.factories import LayoutType
from neurospatial.ops.binning import TieBreakStrategy
from neurospatial.ops.calculus import (
    _compute_differential_operator,
    _compute_divergence_operator,
)
from neurospatial.regions import Region, Regions

if TYPE_CHECKING:
    from pathlib import Path

    import matplotlib.axes
    import pandas as pd
    import scipy.sparse
    import shapely
    from matplotlib.colors import Colormap
    from scipy.spatial import cKDTree

    from neurospatial.animation.config import ScaleBarConfig
    from neurospatial.animation.overlays import OverlayProtocol
    from neurospatial.environment._protocols import CartesianLookupConsumer
    from neurospatial.environment.factories import PolygonType
    from neurospatial.geometry._arena import Arena
    from neurospatial.geometry._diffusion import DiffusionGeometry, FaceGeometry
    from neurospatial.geometry._polygon import ClipMapping
    from neurospatial.ops._types import (
        DiffusionEigenbasisCache,
        KernelMode,
        MRFEigenbasisCache,
    )
    from neurospatial.ops.diffusion import MRFBasis
    from neurospatial.ops.transforms import Affine2D, AffineND

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from typing_extensions import TypeVar as _CoordinateTypeVar

    _C = _CoordinateTypeVar(
        "_C", bound=CartesianSystem | EgocentricPolarSystem, default=Any
    )
else:
    _C = TypeVar("_C", bound=CartesianSystem | EgocentricPolarSystem)


class Environment(Generic[_C]):
    """A discretized environment with owned coordinate meaning and connectivity.

    ``Environment`` is the central user-facing spatial type. Each instance
    declares whether ``bin_centers`` hold Cartesian ``(x, y[, z])`` positions
    or egocentric ``(distance, angle)`` coordinates. Methods such as ``bin_at``,
    ``contains``, ``distance_between`` and ``apply_transform`` require Cartesian
    positions. The shared ``distance_to`` method guards its Euclidean mode.

    Instances are created via factory classmethods on ``Environment``
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

    _metadata: _Metadata
    _regions: Regions
    _provenance: _Provenance

    name = _MetadataView[str]()

    _geometry: _PublishedGeometry

    bin_centers = _GeometryView[NDArray[np.float64]]()

    connectivity = _GeometryView[nx.Graph]()

    dimension_ranges = _GeometryView["Sequence[tuple[float, float]] | None"]()

    grid_edges = _GeometryView["tuple[NDArray[np.float64], ...] | None"]()

    grid_shape = _GeometryView["tuple[int, ...] | None"]()

    active_mask = _GeometryView["NDArray[np.bool_] | None"]()

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

    units = _MetadataView[str | None]()

    frame = _MetadataView[str | None]()

    @property
    def _is_polar(self) -> bool:
        """Coordinate kind from the owned per-instance descriptor."""
        return isinstance(self._geometry.coordinates, EgocentricPolarSystem)

    _is_linearized_track_env: bool

    _kdtree_cache: cKDTree | None

    _kernel_cache: dict[
        tuple[float, KernelMode],
        NDArray[np.float64],
    ]

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
        """Export owned state without copying or serializing transient analyses."""
        state = self.__dict__.copy()
        for name in ("_kernel_cache", "_kdtree_cache"):
            state.pop(name, None)
        for cls in type(self).__mro__:
            for name, descriptor in vars(cls).items():
                if isinstance(descriptor, cached_property):
                    state.pop(name, None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore owned state and reset transient caches without a builder."""
        self.__dict__.update(state)
        from neurospatial.geometry._ownership import own_arena

        assert self._geometry.arena is not None
        self._geometry = replace(
            self._geometry,
            arena=own_arena(self._geometry.arena),
        )
        self._regions = self._validate_regions(self._regions)
        self._reset_analysis_caches()
        self._freeze_geometry_arrays()

    def _freeze_geometry_arrays(self) -> None:
        """Own immutable buffers; public descriptors expose separate array headers."""
        self._geometry = _freeze_geometry_arrays(self._geometry)

    @property
    def _binning(self) -> Binning:
        """The binning snapshot published with the current geometry."""
        binning = self._geometry.binning
        assert binning is not None
        return binning

    @property
    def binning(self) -> Binning:
        """Return a detached snapshot of bins, lookup and geometry components.

        Returns
        -------
        Binning
            Independently owned geometry records in this environment's coordinates.

        Notes
        -----
        The returned records and NumPy array headers are independently owned.
        Editing their shapes or writable flags cannot change this environment.
        Native face recipes remain lazy; reading this property does not build
        numerical geometry. To pass a native Cartesian snapshot to from_bins(),
        explicitly replace its faces with KnownFaces(snapshot.faces.build()).
        MissingFaces is already explicit and retains its refusal reason.

        Polar snapshots retain radius/angle coordinates and are not inputs to
        the Cartesian from_bins() factory. Use environment storage to restore
        an unchanged polar environment.
        """
        from neurospatial.geometry._ownership import own_binning

        return own_binning(self._binning)

    @property
    def arena(self) -> Arena:
        """Continuous domain, distinguished from the active analysis bins."""
        arena = self._geometry.arena
        assert arena is not None
        return copy_module.deepcopy(arena)

    @property
    def clip_mapping(self) -> ClipMapping | None:
        """Most recent clip's source-bin mapping, or None before clipping."""
        return copy_module.deepcopy(self._geometry.clip_mapping)

    @property
    def edge_pairs(self) -> NDArray[np.intp]:
        """Oriented neighbor pairs in gradient and flux order, shape (n_edges, 2).

        This order preserves explicit bin input, and can differ from NetworkX
        edge iteration. The returned array is independent and read-only.
        """
        pairs: NDArray[np.intp] = self._binning.bins.pairs.copy()
        pairs.setflags(write=False)
        return pairs

    def to_networkx(self) -> nx.Graph:
        """Return a detached editable graph with bin IDs and weighted edges.

        ``env.connectivity`` stores its graph, node and edge attributes as
        immutable values: lists read back as tuples, dicts as read-only
        mappings and arrays as read-only arrays. This export restores them as
        independent, editable copies: lists, dicts, sets and writable arrays,
        at every nesting level. Values stored as tuples stay tuples. Node
        identities are the bin IDs; ``edge_pairs`` defines numerical edge order.
        Standard NetworkX drawing, editing and JSON conversion can use this graph.

        Returns
        -------
        networkx.Graph
            Editable navigation graph, independent of the environment.
        """
        from neurospatial.environment._publication import editable_graph

        return editable_graph(self.connectivity)

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
                stacklevel=_external_stacklevel(),
            )

    def _reset_analysis_caches(self) -> None:
        """Discard this instance's memoized analyses without clearing shared values."""
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

        Returns
        -------
        Environment
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
        Environment
            Same concrete type and geometry, with fresh analysis caches.
        """
        snapshot = self._validate_regions(regions)
        result = self._replace_value()
        result._regions = snapshot
        return result

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
        from neurospatial.environment._summary import _repr

        return _repr(self)

    def __str__(self) -> str:
        """Human-readable summary, equivalent to :meth:`info`.

        ``str(env)`` returns the same multi-line summary that
        ``env.info()`` produces. ``repr(env)`` keeps its single-line
        ``Environment(name=..., 2D, 100 bins, RegularGrid)`` shape for
        interactive debugging and logs.
        """
        from neurospatial.environment._summary import _str

        return _str(self)

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
        from neurospatial.environment._summary import _repr_html

        return _repr_html(self)

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
        from neurospatial.environment._summary import _info

        return _info(self)

    @cached_property
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
            If `bin_centers` is not available.

        """
        return int(self.bin_centers.shape[1])

    @property
    def layout_parameters(self) -> dict[str, Any]:
        """Return the scalar choices used to build the bins.

        These are the settings passed to the layout engine, such as
        ``bin_size`` or ``connect_diagonal_neighbors``, or recorded by a
        derivation. Input data (positions, masks, images, polygons and track
        graphs) is not kept here; the geometry built from it is in
        ``env.binning``, for example ``env.binning.linearization.graph`` for a
        track. The values describe how the environment was built and do not
        affect any analysis.

        Returns
        -------
        dict of str to object
            A new dict of scalars and tuples of scalars.
        """
        return self._provenance.build_params

    @property
    def layout_type(self) -> str:
        """Return the type of layout used in the environment.

        Returns
        -------
        str
            The layout type (e.g., "RegularGrid", "Hexagonal").

        """
        return self._provenance.layout_type

    @property
    def n_bins(self) -> int:
        """Return the number of active bins in the environment.

        This is determined by the number of rows in `self.bin_centers`.

        Returns
        -------
        int
            The number of active bins.

        """
        return int(self._geometry.bin_centers.shape[0])

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
        ``clear_cache(cached_properties=True)`` discards them when needed.

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
            reused until clear_cache(cached_properties=True).

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

    @cached_property
    def _differential_operator_cached(self) -> sparse.csc_matrix:
        faces = self._require_face_geometry("gradient/divergence")
        return _compute_differential_operator(faces, self.n_bins)

    @cached_property
    def _divergence_operator_cached(self) -> sparse.csc_matrix:
        faces = self._require_face_geometry("gradient/divergence")
        return _compute_divergence_operator(faces, self._differential_operator_cached)

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

        # Geometry is immutable; cached analyses last until explicitly cleared.
        if cached_properties:
            for cls in type(self).__mro__:
                for name, attr in vars(cls).items():
                    if isinstance(attr, cached_property):
                        self.__dict__.pop(name, None)

    @property
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

    @property
    def coordinate_system(self) -> _C:
        """Return this environment's immutable coordinate descriptor.

        CartesianSystem describes position columns in length units.
        EgocentricPolarSystem describes radius in length units and angle in
        radians, including whether the angular axis wraps. The descriptor is
        an independent frozen value; changing it cannot change the environment.
        """
        return copy_module.copy(self._coordinate_system)

    def _require_cartesian(self, operation: str) -> CartesianLookupConsumer:
        """Refuse Cartesian coordinate math for polar radius/angle bins."""
        if self._is_polar:
            from neurospatial._exceptions import CartesianOnlyError
            from neurospatial.environment.polar import _POLAR_EXPLANATION

            raise CartesianOnlyError(
                type(self).__name__,
                operation,
                reason="Cartesian-grid operation"
                if operation.partition("(")[0] in CARTESIAN_GRID_OPERATIONS
                else "Cartesian-coordinate operation",
                explanation=_POLAR_EXPLANATION,
            )
        return cast("CartesianLookupConsumer", self)

    # Runtime capability filtering must not make unknown attributes appear as
    # Any to type checkers, which would also admit missing protocol members.
    if not TYPE_CHECKING:

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

    @cached_property
    def _boundary_bins_cached(self) -> NDArray[np.intp]:
        if not self._is_polar:
            return _cartesian_boundary_bins(self)
        from neurospatial.layout.helpers.utils import find_boundary_nodes

        return find_boundary_nodes(
            graph=self.connectivity,
            grid_shape=self.grid_shape,
            active_mask=self.active_mask,
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
    def component_keys(self: Environment[CartesianSystem]) -> tuple[str, ...]:
        """Stable combined component keys, or an empty tuple for a native environment."""
        self._require_cartesian("component_keys()")
        from neurospatial.geometry._composition import _component_keys

        return _component_keys(self._binning)

    @property
    def bin_sources(
        self: Environment[CartesianSystem],
    ) -> tuple[tuple[str, int], ...] | None:
        """Original component key and local bin ID for each bin, or None when native.

        Subsets and clips retain original source IDs; split cells may repeat an ID.
        """
        self._require_cartesian("bin_sources()")
        from neurospatial.geometry._composition import _bin_sources

        return _bin_sources(self._binning)

    def get_bridge_attributes(self: Environment[CartesianSystem]) -> pd.DataFrame:
        """Return an independent table of retained automatic navigation bridges.

        Rows identify component/local endpoints, current bin IDs, path length
        and geometric conductance. Native environments return an empty table.
        Conductance units are length ** (measure_dim - 2).
        """
        self._require_cartesian("get_bridge_attributes()")
        from neurospatial.environment.metrics import _get_bridge_attributes

        return _get_bridge_attributes(self._binning)

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

    def nearest_bin(
        self,
        points: NDArray[np.float64],
        *,
        tie_break: TieBreakStrategy
        | Literal["lowest_index", "closest_center"] = TieBreakStrategy.LOWEST_INDEX,
        return_dist: bool = False,
        max_distance: float | None = None,
        max_distance_factor: float | None = None,
    ) -> NDArray[np.int64] | tuple[NDArray[np.int64], NDArray[np.float64]]:
        """Find the nearest bin center for every point in stored coordinates.

        Parameters
        ----------
        points : ndarray, shape (n_points, n_dims)
            Coordinates to compare with bin centers.
        tie_break : TieBreakStrategy or str, default=TieBreakStrategy.LOWEST_INDEX
            Select the lowest bin ID among ties, or use "closest_center" for
            the tree's nearest candidate.
        return_dist : bool, default=False
            Also return Euclidean distances to the selected centers.
        max_distance : float or None, optional
            Maximum Euclidean distance; rejected points return bin ID -1.
        max_distance_factor : float or None, optional
            Maximum distance as a multiple of typical center spacing. Cannot
            be combined with max_distance. Without either limit, the existing
            ten-times-typical-spacing rule rejects distant points.

        Returns
        -------
        ndarray or tuple of ndarray
            Bin IDs, shape (n_points,). When return_dist=True, also return
            distances with infinity for rejected points.

        Raises
        ------
        ValueError
            If distance limits or tie-breaking settings are invalid, or query
            points have the wrong coordinate width or nonfinite values.

        Notes
        -----
        This compares Euclidean distance in stored coordinate columns and
        reuses an environment-owned KD-tree. It does not test containment;
        use bin_at() for Cartesian occupancy and trajectory assignment.
        On polar environments, radius and angle are compared as raw columns
        with mixed coordinate units. That preserves map_points_to_bins()
        behavior and is neither physical polar distance nor containment.
        """
        from scipy.spatial import cKDTree

        from neurospatial.ops.binning import (
            _nearest_bin_query,
            _validate_nearest_options,
        )

        resolved = _validate_nearest_options(
            tie_break, max_distance, max_distance_factor
        )
        centers = self._binning.bins.centers
        if self._kdtree_cache is None:
            self._kdtree_cache = cKDTree(centers)
        return _nearest_bin_query(
            centers,
            self._kdtree_cache,
            points,
            tie_break=resolved,
            return_dist=return_dist,
            max_distance=max_distance,
            max_distance_factor=max_distance_factor,
        )

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
        ValueError
            If ``points_nd`` has a different dimensionality than the
            environment (e.g. 3-D points for a 2-D environment).

        Notes
        -----
        Tracks require a declared width: a centerline and topology alone do
        not establish which off-track points are inside. Unknown width raises
        NotImplementedError with guidance; bin_at() still projects to the track.
        Other layouts reuse bin lookup and check for the -1 sentinel.

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
        self._require_cartesian("contains()")
        if self.is_linearized_track and not self.arena.domain_known:
            raise NotImplementedError(
                _format_error(
                    "contains() requires a known track footprint.",
                    why="A centerline and bin size do not determine the physical track width.",
                    fix="supply track_width when constructing the track, or use bin_at() for nearest-track projection",
                )
            )
        indices = self._locate(points_nd)
        return np.asarray(indices != -1, dtype=np.bool_)

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
                    stacklevel=_external_stacklevel(),
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
        TypeError
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
            raise TypeError(msg)

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
        TypeError
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
            raise TypeError(msg)

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
            ``original_shape // factor`` in each dimension. Keep coarse bins
            whose centers lie inside the original arena.

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

        **Active bins**: With ``factor``, keep coarse cells whose centers lie
        in the known arena, including on its boundary, as ``contains`` does;
        a factor of 2 therefore merges every fine bin of a grid mask. With
        ``bin_size``, keep new cells whose centers lie strictly inside a
        polygonal arena, or are contained by another known arena. Unknown
        domains retain the full coarse grid. Rebinning preserves the arena
        rather than filling its holes. When original bins (inside the
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
        >>> # Create 10x10 grid (deterministic)
        >>> rng = np.random.RandomState(42)
        >>> data = rng.rand(1000, 2) * 100
        >>> env = Environment.from_samples(data, bin_size=10.0)
        >>> env.grid_shape  # May vary with random data
        (11, 11)
        >>>
        >>> # Coarsen by factor 2 → ~5x5 grid (11 // 2 = 5)
        >>> coarse = env.rebin(factor=2)
        >>> coarse.grid_shape
        (5, 5)
        >>>
        >>> # Anisotropic coarsening with tuple (11 // 2 = 5, 11 // 5 = 2)
        >>> coarse_aniso = env.rebin(factor=(2, 5))
        >>> coarse_aniso.grid_shape
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
            return _rebin_resolution(self, bin_size, name=name)
        assert factor is not None
        return _rebin_factor(self, factor, name=name)

    def subset(
        self: Environment[CartesianSystem],
        *,
        bins: NDArray[np.bool_] | None = None,
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
        from neurospatial.environment._subset import _subset

        return _subset(
            self,
            bins=bins,
            region_names=region_names,
            polygon=polygon,
            invert=invert,
            name=name,
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

    def apply_transform(
        self: Environment[CartesianSystem],
        transform: AffineND | Affine2D,
        *,
        name: str | None = None,
        frame: str | None = None,
    ) -> Environment[CartesianSystem]:
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
        A general affine fitted to noisy landmarks absorbs the noise as a small
        stretch or shear, so smoothing refuses; fit landmarks with
        ``estimate_transform(..., kind="similarity")`` to keep it.
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
            Name for the transformed environment; defaults to this
            environment's name.
        frame : str, optional
            Coordinate frame label of the transformed positions, for example
            ``"session2"``. The default ``None`` leaves the frame unlabeled,
            because the transformed coordinates are no longer in this
            environment's frame.

        Returns
        -------
        transformed_env : Environment
            New Environment instance with transformed coordinates. The transformation
            is applied to:

            - `bin_centers`: All bin positions are transformed
            - `connectivity`: Node 'pos' attributes updated, edge distances/vectors recomputed
            - `regions`: Point and polygon regions are transformed
            - `metadata`: Name and units preserved; frame set from ``frame``

            Bin IDs and edge ordering are retained. Cell/face measures and
            available numerical capabilities follow the transformed geometry.
            Bin-aligned values and fitted models are not rescaled or refitted.

        Raises
        ------
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
        self._require_cartesian("apply_transform()")
        from neurospatial.environment._affine import _apply_transform_to_environment

        return _apply_transform_to_environment(self, transform, name=name, frame=frame)

    @classmethod
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
        from neurospatial.environment.factories import _combine

        return _combine(
            cls,
            components,
            auto_bridge=auto_bridge,
            max_mnn_distance=max_mnn_distance,
            bridge_conductances=bridge_conductances,
            name=name,
        )

    @classmethod
    def from_bins(
        cls,
        binning: object,
        *,
        name: str = "",
        units: str | None = None,
        frame: str | None = None,
        regions: Regions | None = None,
        arena: Arena | None = None,
    ) -> Environment[CartesianSystem]:
        """Construct an environment from explicit Cartesian binning data.

        Parameters
        ----------
        binning : neurospatial.binning.Binning or registered provider
            Bin centers, measures and ordered neighbor pairs, with an explicit
            supported locator. KnownFaces enables finite-volume operations;
            MissingFaces explicitly refuses them. Shapes are required for exact
            clipping. Centers and sizes alone cannot establish these capabilities.
        name : str, default=""
            Environment name.
        units, frame : str or None, optional
            Coordinate declarations; they do not convert coordinates.
        regions : Regions or None, optional
            Named regions to snapshot at construction.
        arena : Arena or None, optional
            Continuous domain. Omission infers only what the supplied geometry
            establishes, without inventing physical walls.

        Returns
        -------
        Environment
            Owned environment preserving bin IDs and supplied pair order.
            ``edge_pairs`` defines gradient and flux order; NetworkX iteration
            may use a different order.
        """
        from neurospatial.environment.factories import _from_bins

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
    def from_samples(
        cls,
        positions: NDArray[np.float64],
        bin_size: float | Sequence[float],
        *,
        name: str = "",
        units: str | None = None,
        frame: str | None = None,
        regions: Regions | None = None,
        layout: LayoutType | str = LayoutType.REGULAR_GRID,
        infer_active_bins: bool = True,
        bin_count_threshold: int = 0,
        dilate: bool = False,
        fill_holes: bool = False,
        close_gaps: bool = False,
        add_boundary_bins: bool = False,
        connect_diagonal_neighbors: bool = True,
        arena: Arena | None = None,
        **layout_specific_kwargs: Any,
    ) -> Environment[CartesianSystem]:
        """Create an Environment by binning (discretizing) `positions` into a layout grid.

        The common path (`positions` + `bin_size`) needs none of the advanced /
        cleanup knobs listed under *Other Parameters*; reach for those (e.g.
        ``dilate``, ``fill_holes``, ``close_gaps``) only when sparse or noisy
        sampling leaves holes, gaps, or a ragged boundary in the active-bin mask.

        Parameters
        ----------
        positions : array, shape (n_samples, n_dims)
            Coordinates of sample points used to infer which bins are "active."
        bin_size : float or sequence of floats
            Size of each bin in the same units as `positions` coordinates.
            For RegularGrid: length of each square bin side (or per-dimension if sequence).
            For Hexagonal: hexagon width (flat-to-flat distance across hexagon).
            If your data is in centimeters, bin_size=5.0 creates 5cm bins.
        name : str, default ""
            Optional name for the resulting Environment.
        units : str or None, default None
            Spatial units for the environment coordinates (e.g. ``"cm"``,
            ``"m"``). If provided, sets ``env.units`` on the returned
            Environment.
        frame : str or None, default None
            Coordinate frame identifier for the environment (e.g. a session
            label). If provided, sets ``env.frame`` on the returned Environment.
        regions : Regions, optional
            Named regions to snapshot with the constructed environment.
        layout : LayoutType | str, default LayoutType.REGULAR_GRID
            Layout engine type to use. Can be a LayoutType enum member (recommended
            for IDE autocomplete) or a case-insensitive string. For RegularGrid and
            Hexagonal layouts, `bin_size` is supported. For "Hexagonal", `bin_size`
            is interpreted as `hexagon_width`. See `list_available_layouts()` for
            all options and `get_layout_parameters()` for layout-specific parameters.
        infer_active_bins : bool, default True
            If True, only bins containing ≥ `bin_count_threshold` samples are "active."
        connect_diagonal_neighbors : bool, default True
            If True, connect grid bins diagonally when building connectivity.
        arena : Arena, optional
            Explicit continuous domain. Without one, infer an estimated domain
            from the active cells; an unvisited patch is not a declared wall.

        Other Parameters
        ----------------
        bin_count_threshold : int, default 0
            Minimum number of data points required for a bin to be considered "active."
        dilate : bool, default False
            If True, apply morphological dilation to the active-bin mask.
        fill_holes : bool, default False
            If True, fill holes in the active-bin mask.
        close_gaps : bool, default False
            If True, close small gaps between active bins.
        add_boundary_bins : bool, default False
            If True, add peripheral bins around the bounding region of samples.

        **layout_specific_kwargs : Any
            Additional options passed to the selected layout engine; see
            ``get_layout_parameters(layout)`` for supported names.

        Returns
        -------
        env : Environment
            A newly created Environment, fitted to the discretized samples.

        Raises
        ------
        ValueError
            If `positions` is not 2D or contains invalid coordinates.
        NotImplementedError
            If `layout` is neither "RegularGrid" nor "Hexagonal".

        See Also
        --------
        from_polygon : Create environment with polygon-defined boundary.
        from_grid_mask : Create environment from pre-defined boolean mask.
        from_pixel_mask : Create environment from binary image mask.
        from_graph : Create 1D linearized track environment.
        from_layout : Create environment with custom LayoutEngine.
        neurospatial.layout.factories.list_available_layouts : Get all available layout types.
        neurospatial.layout.factories.get_layout_parameters : Get parameters for a layout type.

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
        ...     bin_count_threshold=5,  # Require 5 samples per bin (lowered from 10)
        ...     dilate=True,  # Expand active region
        ...     fill_holes=True,  # Fill interior holes
        ... )

        Create a hexagonal grid environment:

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
        # Convert and validate positions array with helpful error messages
        from neurospatial.environment.factories import _from_samples

        return _from_samples(
            cls,
            positions,
            bin_size,
            name=name,
            units=units,
            frame=frame,
            regions=regions,
            layout=layout,
            infer_active_bins=infer_active_bins,
            bin_count_threshold=bin_count_threshold,
            dilate=dilate,
            fill_holes=fill_holes,
            close_gaps=close_gaps,
            add_boundary_bins=add_boundary_bins,
            connect_diagonal_neighbors=connect_diagonal_neighbors,
            arena=arena,
            **layout_specific_kwargs,
        )

    @classmethod
    def open_field(
        cls,
        positions: NDArray[np.float64],
        bin_size: float | Sequence[float],
        *,
        name: str = "",
        units: str | None = None,
        frame: str | None = None,
        layout: LayoutType | str = LayoutType.REGULAR_GRID,
        infer_active_bins: bool = True,
        bin_count_threshold: int = 0,
        dilate: bool = False,
        close_gaps: bool = False,
        add_boundary_bins: bool = False,
        connect_diagonal_neighbors: bool = True,
        **layout_specific_kwargs: Any,
    ) -> Environment[CartesianSystem]:
        """Create an open-arena Environment from sampled positions.

        This is the only positions-based preset. It delegates to
        :meth:`from_samples` and flips ``fill_holes=True`` -- a sensible
        open-arena default that fills interior gaps left by uneven sampling,
        so a freely explored 2D arena reads as one filled region rather than
        a speckled mask. All other :meth:`from_samples` options pass through
        unchanged.

        Parameters
        ----------
        positions : array, shape (n_samples, n_dims)
            Coordinates of sample points used to infer which bins are "active."
        bin_size : float or sequence of floats
            Size of each bin in the same units as `positions` coordinates.
        name : str, default ""
            Optional name for the resulting Environment.
        units : str or None, default None
            Spatial units for the environment coordinates (e.g. ``"cm"``).
            Forwarded to :meth:`from_samples`, which sets ``env.units``.
        frame : str or None, default None
            Coordinate frame identifier. Forwarded to :meth:`from_samples`,
            which sets ``env.frame``.
        layout : LayoutType | str, default LayoutType.REGULAR_GRID
            Layout engine type to use. See :meth:`from_samples`.
        infer_active_bins : bool, default True
            If True, only bins containing ≥ `bin_count_threshold` samples are "active."
        bin_count_threshold : int, default 0
            Minimum number of data points required for a bin to be "active."
        dilate : bool, default False
            If True, apply morphological dilation to the active-bin mask.
        close_gaps : bool, default False
            If True, close small gaps between active bins.
        add_boundary_bins : bool, default False
            If True, add peripheral bins around the bounding region of samples.
        connect_diagonal_neighbors : bool, default True
            If True, connect grid bins diagonally when building connectivity.
        **layout_specific_kwargs
            Additional keyword arguments forwarded to :meth:`from_samples`.

        Returns
        -------
        env : Environment
            A newly created open-arena Environment, fitted to the samples.

        See Also
        --------
        from_samples : The general factory this preset delegates to.
        linear_track : Linear / piecewise-linear track preset (explicit topology).
        maze : W / plus / T maze preset (explicit topology).

        Notes
        -----
        ``fill_holes`` is forced to ``True`` and is therefore not exposed as a
        parameter here. If you need the un-filled active mask, call
        :meth:`from_samples` directly with ``fill_holes=False``.

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
        from neurospatial.environment.factories import _open_field

        return _open_field(
            cls,
            positions,
            bin_size,
            name=name,
            units=units,
            frame=frame,
            layout=layout,
            infer_active_bins=infer_active_bins,
            bin_count_threshold=bin_count_threshold,
            dilate=dilate,
            close_gaps=close_gaps,
            add_boundary_bins=add_boundary_bins,
            connect_diagonal_neighbors=connect_diagonal_neighbors,
            **layout_specific_kwargs,
        )

    @classmethod
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
        from neurospatial.environment.factories import _linear_track

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
        )

    @classmethod
    def maze(
        cls,
        kind: str,
        *,
        track_graph: nx.Graph | None = None,
        node_positions: dict[Any, Sequence[float]] | None = None,
        bin_size: float,
        edge_spacing: float | Sequence[float] = 0.0,
        track_width: float | None = None,
        name: str = "",
        units: str | None = None,
        frame: str | None = None,
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
        >>> env = Environment.maze("plus", node_positions=nodes, bin_size=5.0)
        >>> env.is_linearized_track
        True
        """
        from neurospatial.environment.factories import _maze

        return _maze(
            cls,
            kind,
            track_graph=track_graph,
            node_positions=node_positions,
            bin_size=bin_size,
            edge_spacing=edge_spacing,
            track_width=track_width,
            name=name,
            units=units,
            frame=frame,
        )

    @classmethod
    def from_graph(
        cls,
        graph: nx.Graph,
        edge_order: list[tuple[Any, Any]],
        edge_spacing: float | Sequence[float],
        bin_size: float,
        *,
        name: str = "",
        units: str | None = None,
        frame: str | None = None,
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
            order; any ``edge_id`` on the input graph is ignored.
        edge_order : List[Tuple[Any, Any]]
            An ordered list of edge tuples (node1, node2) from `graph` that
            defines the 1D bin ordering.
        edge_spacing : Union[float, Sequence[float]]
            The spacing to insert between consecutive edges in `edge_order`
            during linearization, in the same units as the graph node coordinates.
            If a float, applies to all gaps. If a sequence, specifies spacing for
            each gap.
        bin_size : float
            The length of each bin along the linearized track, in the same units
            as the graph node coordinates. For example, if node positions are in
            centimeters, bin_size=2.0 creates 2cm bins along the track.
        name : str, optional
            A name for the created environment. Defaults to "".
        units : str, optional
            Unit label for the supplied coordinates; does not scale them.
        frame : str, optional
            Coordinate frame label for the supplied coordinates.
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
        from_layout : Create environment with custom LayoutEngine.

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
        from neurospatial.environment.factories import _from_graph

        return _from_graph(
            cls,
            graph,
            edge_order,
            edge_spacing,
            bin_size,
            name=name,
            units=units,
            frame=frame,
            track_width=track_width,
        )

    @classmethod
    def from_polygon(
        cls,
        polygon: PolygonType,
        bin_size: float | Sequence[float],
        *,
        name: str = "",
        units: str | None = None,
        frame: str | None = None,
        connect_diagonal_neighbors: bool = True,
        arena: Arena | None = None,
    ) -> Environment[CartesianSystem]:
        """Create a 2D grid Environment masked by a Shapely Polygon.

        A regular grid is formed based on the polygon's bounds and `bin_size`.
        Only grid cells whose centers are contained within the polygon are
        considered active.

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
        connect_diagonal_neighbors : bool, optional
            Whether to connect diagonally adjacent active grid cells.
            Defaults to True.
        arena : Arena, optional
            Continuous domain and asserted physical walls. Polygonal domains
            retain whole bins by strict center inclusion; use ``env.clip``
            to change cell areas. When omitted, the supplied polygon defines
            the physical footprint.

        Returns
        -------
        Environment
            A new Environment instance with a `ShapelyPolygonLayout`.

        Raises
        ------
        RuntimeError
            If the 'shapely' package is not installed.

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
        from neurospatial.environment.factories import _from_polygon

        return _from_polygon(
            cls,
            polygon,
            bin_size,
            name=name,
            units=units,
            frame=frame,
            connect_diagonal_neighbors=connect_diagonal_neighbors,
            arena=arena,
        )

    @classmethod
    def from_grid_mask(
        cls,
        active_mask: NDArray[np.bool_],
        grid_edges: tuple[NDArray[np.float64], ...],
        *,
        name: str = "",
        units: str | None = None,
        frame: str | None = None,
        connect_diagonal_neighbors: bool = True,
        arena: Arena | None = None,
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
        connect_diagonal_neighbors : bool, optional
            Whether to connect diagonally adjacent active grid cells.
            Defaults to True.
        arena : Arena, optional
            Continuous domain and asserted physical walls. Polygonal domains
            retain whole bins by strict center inclusion. When omitted, the
            mask defines the computational domain without asserting physical
            walls or a measured footprint.

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
        from neurospatial.environment.factories import _from_grid_mask

        return _from_grid_mask(
            cls,
            active_mask,
            grid_edges,
            name=name,
            units=units,
            frame=frame,
            connect_diagonal_neighbors=connect_diagonal_neighbors,
            arena=arena,
        )

    @classmethod
    def from_pixel_mask(
        cls,
        image_mask: NDArray[np.bool_],
        pixel_size: float | tuple[float, float],
        *,
        connect_diagonal_neighbors: bool = True,
        name: str = "",
        units: str | None = None,
        frame: str | None = None,
        arena: Arena | None = None,
    ) -> Environment[CartesianSystem]:
        """Create a 2D Environment from a binary image / pixel mask.

        Each ``True`` pixel in the ``image_mask`` becomes an active bin in the
        environment. ``pixel_size`` determines the spatial scale of those
        pixels in physical units.

        Parameters
        ----------
        image_mask : NDArray[np.bool_], shape (n_rows, n_cols)
            A 2D boolean array where ``True`` pixels define active bins.
        pixel_size : float or tuple of (float, float)
            The spatial size of each pixel in physical units (e.g., cm,
            meters). If a float, pixels are square. If a tuple
            ``(width, height)``, specifies pixel dimensions. For example,
            if your camera captures images where each pixel represents
            0.5 cm, pass ``pixel_size=0.5``.
        connect_diagonal_neighbors : bool, optional
            Whether to connect diagonally adjacent active pixel-bins.
            Defaults to True.
        name : str, optional
            A name for the created environment. Defaults to "".
        units : str, optional
            Unit label for the supplied coordinates; does not scale them.
        frame : str, optional
            Coordinate frame label for the supplied coordinates.
        arena : Arena, optional
            Continuous domain and asserted physical walls. Polygonal domains
            retain whole pixels by strict center inclusion. When omitted, the
            image mask defines a computational domain without asserting
            physical walls or a measured footprint.

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
        >>> image_height, image_width = 480, 640
        >>> mask = np.zeros((image_height, image_width), dtype=bool)
        >>> # Mark a rectangular region as active
        >>> mask[100:400, 150:500] = True
        >>> env = Environment.from_pixel_mask(
        ...     image_mask=mask,
        ...     pixel_size=0.5,  # Each pixel = 0.5cm
        ...     name="arena_from_image",
        ... )
        >>> env.n_dims
        2

        """
        from neurospatial.environment.factories import _from_pixel_mask

        return _from_pixel_mask(
            cls,
            image_mask,
            pixel_size,
            connect_diagonal_neighbors=connect_diagonal_neighbors,
            name=name,
            units=units,
            frame=frame,
            arena=arena,
        )

    @classmethod
    def from_layout(
        cls,
        kind: LayoutType | str,
        layout_params: dict[str, Any],
        *,
        name: str = "",
        units: str | None = None,
        frame: str | None = None,
        regions: Regions | None = None,
        arena: Arena | None = None,
    ) -> Environment[CartesianSystem]:
        """Create an Environment with a specified layout type and its build parameters.

        Parameters
        ----------
        kind : LayoutType | str
            The layout engine type to use. Can be a LayoutType enum member (recommended
            for IDE autocomplete) or a case-insensitive string name
            (e.g., "RegularGrid", "Hexagonal").
        layout_params : Dict[str, Any]
            A dictionary of parameters that will be passed to the `build`
            method of the chosen `LayoutEngine`. Only the scalar choices among
            them are kept in ``layout_parameters``; input data such as
            positions, masks and polygons is not stored.
        name : str, optional
            A name for the created environment. Defaults to "".
        units : str, optional
            Unit label for the supplied coordinates; does not scale them.
        frame : str, optional
            Coordinate frame label for the supplied coordinates.
        regions : Optional[Regions], optional
            A Regions instance to manage symbolic spatial regions within the environment.
        arena : Arena, optional
            Continuous domain. Otherwise inferred from the supplied layout.

        Returns
        -------
        Environment
            A new Environment instance.

        See Also
        --------
        from_samples : Create environment by binning position data.
        from_polygon : Create environment with polygon-defined boundary.
        from_grid_mask : Create environment from pre-defined boolean mask.
        from_pixel_mask : Create environment from binary image mask.
        from_graph : Create 1D linearized track environment.

        """
        # The layout is built from copies of the inputs, so editing the caller's
        # arrays (or the base of a view) afterwards cannot change it.
        from neurospatial.environment.factories import _from_layout

        return _from_layout(
            cls,
            kind,
            layout_params,
            name=name,
            units=units,
            frame=frame,
            regions=regions,
            arena=arena,
        )

    @classmethod
    def _create_layout(
        cls, kind: LayoutType | str, layout_params: dict[str, Any]
    ) -> LayoutEngine:
        """Build the layout engine ``from_layout`` wraps; subclasses may substitute."""
        from neurospatial.environment.factories import _create_layout

        return _create_layout(cls, kind, layout_params)

    @classmethod
    def from_nwb(
        cls,
        nwbfile: Any,
        *,
        scratch_name: str | None = None,
        bin_size: float | None = None,
        **kwargs: Any,
    ) -> Environment[Any]:
        """Create Environment from NWB file.

        This classmethod provides two modes of operation:

        1. **Load from scratch**: If `scratch_name` is provided, loads a previously
           stored Environment from the NWB file's scratch space. Schema 2.0
           restores the complete binning without a layout builder. Older records
           must be recreated with a factory and saved again.

        2. **Create from position**: If `bin_size` is provided (and `scratch_name`
           is not), creates a new Environment by discretizing position data found
           in the NWB file's behavior processing module.

        Parameters
        ----------
        nwbfile : NWBFile
            The NWB file to read from. Must be a pynwb.NWBFile instance.
        scratch_name : str, optional
            Name of the stored environment in scratch/. If provided, loads
            the Environment from NWB scratch space (takes precedence over bin_size).
        bin_size : float, optional
            Bin size for environment discretization when creating from Position data.
            Required if scratch_name is not provided.
        **kwargs
            Additional keyword arguments forwarded to environment_from_position()
            when creating from position data. Common kwargs include:
            - units : str - Spatial units for the environment
            - frame : str - Coordinate frame identifier
            - infer_active_bins : bool - Whether to only include visited bins
            - bin_count_threshold : int - Minimum samples per bin

        Returns
        -------
        Environment
            The loaded or newly created Environment.

        Raises
        ------
        ValueError
            If neither scratch_name nor bin_size is provided.
        KeyError
            If scratch_name is provided but not found in NWB scratch space,
            or if bin_size is provided but no Position data is found.
        ImportError
            If pynwb is not installed.

        See Also
        --------
        to_nwb : Write Environment to NWB file.
        neurospatial.io.nwb.read_environment : Low-level NWB reading function.
        neurospatial.io.nwb.environment_from_position : Create from Position data.

        Examples
        --------
        Load a previously stored environment from scratch:

        >>> from pynwb import NWBHDF5IO  # doctest: +SKIP
        >>> from neurospatial import Environment  # doctest: +SKIP
        >>> with NWBHDF5IO("session.nwb", "r") as io:  # doctest: +SKIP
        ...     nwbfile = io.read()
        ...     env = Environment.from_nwb(nwbfile, scratch_name="linear_track")

        Create environment from Position data:

        >>> with NWBHDF5IO("session.nwb", "r") as io:  # doctest: +SKIP
        ...     nwbfile = io.read()
        ...     env = Environment.from_nwb(nwbfile, bin_size=5.0, units="cm")

        """
        # Validate arguments
        from neurospatial.environment.factories import _from_nwb

        return _from_nwb(
            cls, nwbfile, scratch_name=scratch_name, bin_size=bin_size, **kwargs
        )

    def bin_center_of(
        self,
        bin_indices: int | Sequence[int] | NDArray[np.int_],
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
        >>> print(centers.shape)  # doctest: +SKIP
        (3, 2)

        """
        from neurospatial.environment.queries import _bin_center_of

        return _bin_center_of(self, bin_indices)

    def neighbors(self, bin_index: int | ArrayLike) -> list[int]:
        """Find indices of neighboring active bins for a bin or point.

        This method delegates to the `neighbors` method of the
        underlying `LayoutEngine`, which typically uses the `connectivity`.

        Parameters
        ----------
        bin_index : int or array-like
            An active bin index (``0`` to ``n_active_bins - 1``), OR a
            coordinate point (array-like of length ``n_dims``), which is mapped
            to its active bin via :meth:`bin_at`. Integers are treated as
            indices; coordinate arrays/floats are treated as points.

        Returns
        -------
        list[int]
            A list of active bin indices that are neighbors to `bin_index`.
            Can be used directly for array indexing.

        Raises
        ------
        BinIndexOutOfRangeError
            If bin_index is an integer index outside [0, n_bins).
        ValueError
            If bin_index is a coordinate point outside every active bin.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> x, y = np.meshgrid([0.0, 2.0, 4.0], [0.0, 2.0, 4.0])
        >>> env = Environment.from_samples(np.c_[x.ravel(), y.ravel()], bin_size=2.0)
        >>> neighbors = env.neighbors(0)
        >>> len(neighbors)  # Corner bin, including diagonal connectivity
        3

        """
        from neurospatial.environment.queries import _neighbors

        return _neighbors(self, bin_index)

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
        from neurospatial.environment.queries import _bin_sizes

        return _bin_sizes(self)

    @cached_property
    def _bin_sizes_cached(self) -> NDArray[np.float64]:
        from neurospatial.environment.queries import _bin_sizes_cached

        return _bin_sizes_cached(self)

    def path_between(
        self,
        source_active_bin_idx: int | ArrayLike,
        target_active_bin_idx: int | ArrayLike,
    ) -> list[int]:
        """Find the shortest path between two active bins (or points).

        The path is a sequence of active bin indices (0 to n_active_bins - 1)
        connecting the source to the target. Path calculation uses the
        'distance' attribute on the edges of the `connectivity`
        as weights.

        Parameters
        ----------
        source_active_bin_idx : int or array-like
            The active bin index (0 to n_active_bins - 1) for the start of the
            path, OR a coordinate point (mapped to its bin via :meth:`bin_at`).
        target_active_bin_idx : int or array-like
            The active bin index (0 to n_active_bins - 1) for the end of the
            path, OR a coordinate point (mapped to its bin via :meth:`bin_at`).

        Returns
        -------
        list[int]
            A list of active bin indices representing the shortest path from
            source to target. The list includes both the source and target
            indices. Returns a single-element list ``[source]`` when source and
            target resolve to the same bin, and an empty list when no path
            exists (disconnected components).

        Raises
        ------
        BinIndexOutOfRangeError
            If source/target is an integer index outside [0, n_bins).
        ValueError
            If source/target is a coordinate point outside every active bin.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> data = np.random.rand(100, 2) * 10
        >>> env = Environment.from_samples(data, bin_size=2.0)
        >>> path = env.path_between(0, 5)
        >>> print(len(path) > 0)  # doctest: +SKIP
        True

        """
        from neurospatial.environment.queries import _path_between

        return _path_between(self, source_active_bin_idx, target_active_bin_idx)

    def distance_to(
        self,
        targets: Sequence[int] | str,
        *,
        metric: Literal["euclidean", "geodesic"] = "geodesic",
    ) -> NDArray[np.float64]:
        """Compute distance from each bin to target set.

        This method computes the distance from every bin in the environment to
        the nearest target bin. Useful for:
        - Navigation and path planning
        - Computing distance-based features
        - Analyzing spatial distributions relative to landmarks
        - Creating distance fields for visualization

        Parameters
        ----------
        targets : Sequence[int] or str
            Target bin indices, or a region name. If a region name is provided,
            all bins inside that region are used as targets (multi-source).
        metric : {'euclidean', 'geodesic'}, default='geodesic'
            Distance metric to use:
            - 'euclidean': Straight-line distance in physical coordinates (same units as bin_centers).
            - 'geodesic': Graph distance respecting connectivity (shortest path, in physical units).

        Returns
        -------
        distances : NDArray[np.float64], shape (n_bins,)
            Distance from each bin to the nearest target, in the same units as
            bin_centers (e.g., cm, meters, pixels). For bins unreachable
            from all targets (disconnected graph components), returns np.inf.

        Raises
        ------
        ValueError
            If targets is empty, or if target bin indices are out of range,
            or if metric is invalid.
        CartesianOnlyError
            A NotImplementedError for metric="euclidean" on environments without
            Cartesian bin centers, such as polar environments.
        KeyError
            If targets is a string (region name) that doesn't exist in self.regions.
        TypeError
            If targets is neither a sequence of integers nor a string.

        See Also
        --------
        rings : Compute k-hop neighborhoods (BFS layers).
        reachable_from : Find bins reachable from a source within a radius.
        distance_field : Low-level function for computing geodesic distances.

        Notes
        -----
        **Geodesic Distance**:
        Uses Dijkstra's algorithm to compute shortest paths on the connectivity
        graph. Edge weights are the 'distance' attribute (physical distance between
        bin centers). For disconnected graphs, unreachable bins have distance np.inf.

        **Euclidean Distance**:
        Computes straight-line distance in the coordinate space, ignoring graph
        connectivity. This is faster but doesn't respect physical barriers.

        **Multi-Source Distances**:
        When multiple targets are provided (or a region containing multiple bins),
        each bin's distance is the minimum distance to any target.

        **Region-Based Targets**:
        If targets is a string, it must be a valid region name in self.regions.
        All bins inside that region (as determined by region_membership) become
        target bins.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from shapely.geometry import box
        >>> data = np.array([[i, j] for i in range(10) for j in range(10)])
        >>> env = Environment.from_samples(data, bin_size=1.0)
        >>> from neurospatial import Regions
        >>> regions = Regions()
        >>> _ = regions.add("goal", polygon=box(8.0, 8.0, 10.0, 10.0))
        >>> env = env.with_regions(regions)
        >>> dist = env.distance_to("goal", metric="geodesic")
        >>> dist.shape
        (100,)
        >>> bool(np.all(dist >= 0.0))
        True

        >>> targets = [0, env.n_bins - 1]  # doctest: +SKIP
        >>> dist = env.distance_to(targets, metric="euclidean")  # doctest: +SKIP
        >>> float(dist[targets[0]])  # doctest: +SKIP
        0.0  # doctest: +SKIP
        >>> float(dist[targets[1]])  # doctest: +SKIP
        0.0  # doctest: +SKIP

        """
        # Validate metric
        from neurospatial.environment.queries import _distance_to

        return _distance_to(self, targets, metric=metric)

    def reachable_from(
        self,
        source_bin: int | ArrayLike,
        *,
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
        source_bin : int or array-like
            An active bin index (0 to n_bins - 1), OR a coordinate point
            (mapped to its bin via :meth:`bin_at`). Integers are treated as
            indices; coordinate arrays/floats are treated as points.
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
        BinIndexOutOfRangeError
            If source_bin is an integer index outside [0, n_bins).
        ValueError
            If source_bin is a coordinate point outside every active bin.
            If radius is negative.
            If metric is not 'hops' or 'geodesic'.

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
        >>> mask = env.reachable_from(10, radius=3, metric="hops")  # doctest: +SKIP
        >>> neighbor_bins = np.where(mask)[0]  # doctest: +SKIP
        >>> print(
        ...     f"Found {len(neighbor_bins)} neighbors within 3 hops"
        ... )  # doctest: +SKIP
        Found 37 neighbors within 3 hops  # doctest: +SKIP

        >>> goal_bin = env.bins_in_region("goal")[0]  # doctest: +SKIP
        >>> mask = env.reachable_from(
        ...     goal_bin, radius=50.0, metric="geodesic"
        ... )  # doctest: +SKIP
        >>> print(f"Bins within 50 units: {mask.sum()}")  # doctest: +SKIP
        Bins within 50 units: 125  # doctest: +SKIP

        >>> mask = env.reachable_from(source_bin=0, radius=None)  # doctest: +SKIP
        >>> print(f"Component size: {mask.sum()} bins")  # doctest: +SKIP
        Component size: 1000 bins  # doctest: +SKIP

        """
        # Resolve a bin index or a coordinate point to an active-bin index.
        from neurospatial.environment.queries import _reachable_from

        return _reachable_from(self, source_bin, radius=radius, metric=metric)

    def components(
        self,
        *,
        largest_only: bool = False,
    ) -> list[NDArray[np.int32]]:
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
        components : list[NDArray[np.int32]]
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
        >>> comps = env.components()  # doctest: +SKIP
        >>> print(f"Found {len(comps)} components")  # doctest: +SKIP
        Found 2 components  # doctest: +SKIP
        >>> print(f"Largest component has {len(comps[0])} bins")  # doctest: +SKIP
        Largest component has 150 bins  # doctest: +SKIP

        >>> largest = env.components(largest_only=True)[0]  # doctest: +SKIP
        >>> print(
        ...     f"Largest component: {len(largest)} of {env.n_bins} bins"
        ... )  # doctest: +SKIP
        Largest component: 150 of 200 bins  # doctest: +SKIP

        """
        # Find connected components using NetworkX
        from neurospatial.environment.queries import _components

        return _components(self, largest_only=largest_only)

    def rings(
        self,
        center_bin: int,
        *,
        hops: int,
    ) -> list[NDArray[np.int32]]:
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
        center_bin : int
            Starting bin index.
        hops : int
            Number of hop layers to compute (non-negative).

        Returns
        -------
        rings : list[NDArray[np.int32]], length hops+1
            List of bin index arrays, one per hop distance.
            rings[k] contains all bins exactly k hops from center.
            rings[0] = [center_bin] (the center itself).
            If fewer than hops layers exist (small or disconnected graph),
            later rings will be empty arrays.

        Raises
        ------
        ValueError
            If center_bin is out of range [0, n_bins), or if hops is negative.
        TypeError
            If center_bin is not an integer type, or if hops is not an integer.

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
        >>> rings_result = env.rings(center_bin=50, hops=2)  # doctest: +SKIP
        >>> len(rings_result)  # doctest: +SKIP
        3  # doctest: +SKIP
        >>> len(rings_result[0])  # doctest: +SKIP
        1  # doctest: +SKIP
        >>> len(rings_result[1]) > 0  # doctest: +SKIP
        True  # doctest: +SKIP

        >>> all_bins = np.concatenate(rings_result)  # doctest: +SKIP
        >>> len(all_bins) == len(np.unique(all_bins))  # doctest: +SKIP
        True  # doctest: +SKIP

        """
        # Type validation for center_bin
        from neurospatial.environment.queries import _rings

        return _rings(self, center_bin, hops=hops)

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
        from neurospatial.environment.serialization import _to_file

        return _to_file(self, path, overwrite=overwrite)

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
            Reconstructed Environment instance.

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
        >>> loaded.n_bins == env.n_bins
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
        from neurospatial.environment.serialization import _from_file

        return _from_file(cls, path)

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
        >>> env = Environment.from_samples(data, bin_size=2.0)  # doctest: +SKIP
        >>> env_dict = env.to_dict()  # doctest: +SKIP
        >>> # Dictionary can be converted to JSON
        >>> import json  # doctest: +SKIP
        >>> json_str = json.dumps(env_dict)  # doctest: +SKIP

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
        from neurospatial.environment.serialization import _to_dict

        return _to_dict(self)

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
            Reconstructed instance.

        Examples
        --------
        >>> env = Environment.from_samples(data, bin_size=2.0)  # doctest: +SKIP
        >>> env_dict = env.to_dict()  # doctest: +SKIP
        >>> reconstructed = Environment.from_dict(env_dict)  # doctest: +SKIP
        >>> assert env == reconstructed  # doctest: +SKIP

        See Also
        --------
        to_dict : Convert to dictionary
        from_file : Load from disk files

        """
        from neurospatial.environment.serialization import _from_dict

        return _from_dict(cls, data)

    def to_nwb(
        self,
        nwbfile: Any,
        name: str = "spatial_environment",
        *,
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
            with ``Environment.from_nwb(nwbfile, scratch_name=name)``.
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

        >>> from pynwb import NWBHDF5IO, NWBFile  # doctest: +SKIP
        >>> from datetime import datetime  # doctest: +SKIP
        >>> from neurospatial import Environment  # doctest: +SKIP
        >>> import numpy as np  # doctest: +SKIP
        >>> import tempfile  # doctest: +SKIP
        >>> from pathlib import Path  # doctest: +SKIP
        >>>
        >>> # Create environment
        >>> positions = np.random.rand(1000, 2) * 100  # doctest: +SKIP
        >>> env = Environment.from_samples(positions, bin_size=5.0)  # doctest: +SKIP
        >>> env = env.with_metadata(units="cm")  # doctest: +SKIP
        >>>
        >>> # Save to NWB
        >>> nwbfile = NWBFile(  # doctest: +SKIP
        ...     session_description="Test session",
        ...     identifier="test_001",
        ...     session_start_time=datetime.now().astimezone(),
        ... )
        >>> env.to_nwb(nwbfile, name="linear_track")  # doctest: +SKIP
        >>>
        >>> # Write to disk (use a temp directory, not the CWD)
        >>> path = Path(tempfile.mkdtemp()) / "session.nwb"  # doctest: +SKIP
        >>> with NWBHDF5IO(path, "w") as io:  # doctest: +SKIP
        ...     io.write(nwbfile)

        Load environment back:

        >>> with NWBHDF5IO(path, "r") as io:  # doctest: +SKIP
        ...     nwbfile = io.read()
        ...     loaded_env = Environment.from_nwb(nwbfile, scratch_name="linear_track")

        """
        # Lazy import to keep pynwb optional
        from neurospatial.environment.serialization import _to_nwb

        return _to_nwb(self, nwbfile, name, overwrite=overwrite)

    def bins_in_region(self, region_name: str) -> NDArray[np.intp]:
        """Get active bin indices that fall within a specified named region.

        This method identifies all active bins whose centers fall within the
        geometric bounds of a named region. For point regions, returns the
        bin containing that point. For polygon regions (2D only), returns all
        bins whose centers are contained by the polygon.

        Parameters
        ----------
        region_name : str
            The name of a defined region in `self.regions`.

        Returns
        -------
        NDArray[np.intp]
            Array of active bin indices (0 to n_active_bins - 1)
            that are part of the region. Returns empty array if no bins
            fall within the region.

        Raises
        ------
        KeyError
            If `region_name` is not found in `self.regions`.
        ValueError
            If region point dimension does not match environment dimension,
            if polygon regions are used in non-2D environments, or if
            region kind is unsupported.
        RuntimeError
            If polygon region is requested but shapely is not installed.

        Notes
        -----
        - Point regions return at most one bin (the bin containing the point,
          or empty array if point is outside environment)
        - Polygon regions require shapely library and 2D environments
        - Bin membership is determined by whether the bin center falls within
          the region geometry

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> data = np.array([[0.0, 0.0], [10.0, 10.0]])
        >>> env = Environment.from_samples(data, bin_size=2.0, infer_active_bins=False)
        >>> from neurospatial.regions import Regions
        >>> regions = Regions(env.regions.values())
        >>> _ = regions.add("center", point=(5.0, 5.0))
        >>> env = env.with_regions(regions)
        >>> bins = env.bins_in_region("center")
        >>> bins.shape
        (1,)

        For polygon regions (requires shapely):

        >>> from shapely.geometry import box  # doctest: +SKIP
        >>> from neurospatial.regions import Regions
        >>> regions = Regions(env.regions.values())
        >>> regions.add("roi", polygon=box(2, 2, 8, 8))  # doctest: +SKIP
        >>> env = env.with_regions(regions)
        >>> bins = env.bins_in_region("roi")  # doctest: +SKIP
        >>> len(bins)  # Number of bins in the 6x6 square  # doctest: +SKIP
        9  # doctest: +SKIP

        See Also
        --------
        region_mask : Get boolean mask for one or more regions (canonical
            method form for "is this bin in this region?").
        neurospatial.regions.Region : Region data structure

        """
        from neurospatial.environment.regions import _bins_in_region

        return _bins_in_region(self, region_name)

    def region_membership(
        self,
        regions: Regions | None = None,
        *,
        include_boundary: bool = True,
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
        regions : Regions, optional
            Regions to test against. If None (default), uses self.regions.
            Allows testing against external region sets without modifying
            the environment.
        include_boundary : bool, default=True
            How to handle bins on region boundaries:
            - True: Bins on boundary count as inside (shapely.covers).
            - False: Only bins strictly inside count (shapely.contains).

        Returns
        -------
        membership : NDArray[np.bool_], shape (n_bins, n_regions)
            Boolean array where membership[i, j] = True if bin i is in region j.
            Columns are ordered according to region iteration order.
            If regions is empty, returns array with shape (n_bins, 0).

        Raises
        ------
        TypeError
            If regions parameter is not a Regions instance or None.
            If include_boundary is not a boolean.

        See Also
        --------
        subset : Create new environment from bin selection.
        bins_in_region : Get bin indices for a specific region.

        Notes
        -----
        **Region Types**:
        - Polygon regions: Uses Shapely containment (covers/contains).
        - Point regions: Always return False (points have no area).

        **Performance**:
        This method uses vectorized Shapely operations for efficiency.
        For N bins and R regions, complexity is O(N * R), but vectorized
        operations make it fast even for thousands of bins.

        **Boundary Semantics**:
        The include_boundary parameter controls the Shapely predicate used:
        - include_boundary=True: Uses shapely.covers(region, point)
          Returns True if point is inside region OR on its boundary.
        - include_boundary=False: Uses shapely.contains(region, point)
          Returns True only if point is strictly inside region.

        For most applications, include_boundary=True is appropriate, as it
        avoids ambiguity for bins whose centers lie exactly on region edges.

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
        >>> _ = regions.add("left", polygon=box(0, 0, 5, 10))
        >>> _ = regions.add("right", polygon=box(5, 0, 10, 10))
        >>> env = env.with_regions(regions)
        >>> membership = env.region_membership()
        >>> membership.shape[1]
        2
        >>> membership.dtype
        dtype('bool')

        >>> left_bins = np.where(membership[:, 0])[0]  # doctest: +SKIP
        >>> len(left_bins) > 0  # doctest: +SKIP
        True  # doctest: +SKIP

        >>> both = np.all(membership, axis=1)  # doctest: +SKIP
        >>> overlapping_bins = np.where(both)[0]  # doctest: +SKIP
        >>> len(overlapping_bins) >= 0  # doctest: +SKIP
        True  # doctest: +SKIP

        >>> from neurospatial.regions import Regions  # doctest: +SKIP
        >>> external = Regions()  # doctest: +SKIP
        >>> _ = external.add("test", polygon=box(2, 2, 8, 8))  # doctest: +SKIP
        >>> test_membership = env.region_membership(regions=external)  # doctest: +SKIP
        >>> test_membership.shape[1]  # doctest: +SKIP
        1  # doctest: +SKIP

        >>> interior = env.region_membership(include_boundary=False)  # doctest: +SKIP
        >>> bool(interior.sum() <= membership.sum())  # doctest: +SKIP
        True  # doctest: +SKIP

        """
        # Import here to avoid circular dependency
        from neurospatial.environment.regions import _region_membership

        return _region_membership(self, regions, include_boundary=include_boundary)

    def region_mask(
        self,
        regions: str | list[str] | Region | Regions,
        *,
        include_boundary: bool = True,
    ) -> NDArray[np.bool_]:
        """Get boolean mask over bins for one or more regions.

        This method provides a convenient interface to rasterize continuous regions
        onto discrete environment bins. It accepts multiple input types and returns
        a boolean mask indicating which bins fall within the region(s).

        This is the method-based interface to the `regions_to_mask()` free function,
        providing better discoverability and ergonomics when working with a single
        environment.

        Parameters
        ----------
        regions : str, list[str], Region, or Regions
            Region(s) to rasterize. Can be:
            - A single region name (str) from self.regions
            - A list of region names (list[str]) from self.regions
            - A single Region object
            - A Regions collection
        include_boundary : bool, default=True
            Whether to include bins whose centers lie on region boundaries:
            - True: Include boundary bins (uses shapely.covers)
            - False: Exclude boundary bins (uses shapely.contains)

        Returns
        -------
        mask : NDArray[np.bool_], shape (n_bins,)
            Boolean mask where True indicates bin center is inside region(s).
            For multiple regions, returns the union (logical OR).
            Point regions always return all False (points have no area).

        Raises
        ------
        KeyError
            If a region name is not found in self.regions.
        TypeError
            If regions parameter has invalid type or include_boundary is not bool.

        Notes
        -----
        **Region Types**:
        - Polygon regions: Rasterized using Shapely containment checks
        - Point regions: Always return all False (points have no area)

        **Multiple Regions**:

        When multiple regions are provided, the result is their **union** (logical OR).
        Bins are included if they fall inside ANY of the regions. Overlapping
        regions are counted only once.

        For intersection or difference, use boolean operations on separate masks:

        >>> left_mask = env.region_mask("left")  # doctest: +SKIP
        >>> right_mask = env.region_mask("right")  # doctest: +SKIP
        >>> intersection = left_mask & right_mask  # doctest: +SKIP
        >>> difference = left_mask & ~right_mask  # doctest: +SKIP

        **Boundary Semantics**:

        - include_boundary=True: Uses shapely.covers (includes boundary)
        - include_boundary=False: Uses shapely.contains (excludes boundary)

        Examples
        --------
        >>> import numpy as np
        >>> from shapely.geometry import box
        >>> from neurospatial import Environment

        Create environment with regions:

        >>> data = np.array([[i, j] for i in range(11) for j in range(11)])
        >>> env = Environment.from_samples(data, bin_size=2.0)
        >>> from neurospatial.regions import Regions
        >>> regions = Regions(env.regions.values())
        >>> _ = regions.add("center", polygon=box(3, 3, 7, 7))
        >>> _ = regions.add("left", polygon=box(0, 0, 4, 10))
        >>> _ = regions.add("right", polygon=box(6, 0, 10, 10))
        >>> env = env.with_regions(regions)

        Single region by name:

        >>> mask = env.region_mask("center")
        >>> mask.shape  # doctest: +ELLIPSIS
        (...,)
        >>> mask.dtype
        dtype('bool')

        Multiple regions (union):

        >>> mask = env.region_mask(["left", "right"])  # doctest: +SKIP
        >>> np.any(mask)  # doctest: +SKIP
        True  # doctest: +SKIP

        All regions in environment:

        >>> mask = env.region_mask(env.regions)  # doctest: +SKIP
        >>> mask.shape  # doctest: +SKIP
        (36,)  # doctest: +SKIP

        Exclude boundary bins:

        >>> mask_no_boundary = env.region_mask(
        ...     "center", include_boundary=False
        ... )  # doctest: +SKIP
        >>> mask_with_boundary = env.region_mask(
        ...     "center", include_boundary=True
        ... )  # doctest: +SKIP
        >>> bool(mask_no_boundary.sum() <= mask_with_boundary.sum())  # doctest: +SKIP
        True  # doctest: +SKIP

        Use with external Region object:

        >>> from neurospatial.regions import Region  # doctest: +SKIP
        >>> external_region = Region(
        ...     name="ext", kind="polygon", data=box(4, 4, 6, 6)
        ... )  # doctest: +SKIP
        >>> mask = env.region_mask(external_region)  # doctest: +SKIP
        >>> mask.shape  # doctest: +SKIP
        (36,)  # doctest: +SKIP

        See Also
        --------
        region_membership : 2D membership array for all regions
        bins_in_region : Get bin indices for a specific region
        neurospatial.spatial.regions_to_mask : Free function version

        Notes
        -----
        **Point regions**: ``region_mask`` reports the single bin
        whose center is closest to the point (via :meth:`bins_in_region`,
        which uses :meth:`bin_at`). The free-function
        :func:`regions_to_mask` returns an empty mask for points
        because points have no area; the method-form here special-cases
        them so ``env.region_mask("goal")`` matches what users mean
        when they write that.
        """
        # Import here to avoid circular dependency
        from neurospatial.environment.regions import _region_mask

        return _region_mask(self, regions, include_boundary=include_boundary)

    def plot(
        self,
        ax: matplotlib.axes.Axes | None = None,
        show_regions: bool = False,
        layout_plot_kwargs: dict[str, Any] | None = None,
        regions_plot_kwargs: dict[str, Any] | None = None,
        scale_bar: bool | ScaleBarConfig = False,
        **kwargs: Any,
    ) -> matplotlib.axes.Axes:
        """Plot the environment's layout and optionally defined regions.

        This method delegates plotting of the base layout to the `plot` method
        of the underlying `LayoutEngine`. If `show_regions` is True, it then
        overlays any defined spatial regions managed by `self.regions`.

        Parameters
        ----------
        ax : Optional[matplotlib.axes.Axes], optional
            The Matplotlib axes to plot on. If None, a new figure and axes
            are created. Defaults to None.
        show_regions : bool, optional
            If True, plot defined spatial regions on top of the layout.
            Defaults to False.
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
        **kwargs : Any
            Additional keyword arguments that are passed to `layout.plot()`.
            These can be overridden by `layout_plot_kwargs`.

        Returns
        -------
        matplotlib.axes.Axes
            The axes on which the environment was plotted.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> data = np.random.rand(100, 2) * 10
        >>> env = Environment.from_samples(data, bin_size=2.0)
        >>> ax = env.plot()  # doctest: +SKIP

        Plot with regions:

        >>> from neurospatial.regions import Regions
        >>> regions = Regions(env.regions.values())
        >>> regions.add("goal", point=[5.0, 5.0])  # doctest: +SKIP
        >>> env = env.with_regions(regions)
        >>> ax = env.plot(show_regions=True)  # doctest: +SKIP

        """
        from neurospatial.environment.visualization import _plot

        return _plot(
            self,
            ax,
            show_regions,
            layout_plot_kwargs,
            regions_plot_kwargs,
            scale_bar,
            **kwargs,
        )

    def plot_1d(
        self,
        ax: matplotlib.axes.Axes | None = None,
        layout_plot_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> matplotlib.axes.Axes | None:
        """Plot a 1D representation of the environment, if applicable.

        This method is primarily for environments where `is_linearized_track` is True
        (e.g., using `GraphLayout`). It calls the `plot_linear_layout`
        method of the underlying layout if it exists and the layout is 1D.

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

        Raises
        ------
        AttributeError
            If `self.layout.is_linearized_track` is True but the layout does not have a
            `plot_linear_layout` method.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from track_linearization import make_track_graph  # doctest: +SKIP
        >>> position = np.random.rand(100, 2) * 10  # doctest: +SKIP
        >>> track_graph = make_track_graph(position, use_HMM=False)  # doctest: +SKIP
        >>> env = Environment.from_graph(
        ...     track_graph, track_graph_name="test_track"
        ... )  # doctest: +SKIP
        >>> if env.is_linearized_track:  # doctest: +SKIP
        ...     ax = env.plot_1d()  # doctest: +SKIP

        """
        from neurospatial.environment.visualization import _plot_1d

        return _plot_1d(self, ax, layout_plot_kwargs, **kwargs)

    def plot_field(
        self,
        field: NDArray[np.float64],
        ax: matplotlib.axes.Axes | None = None,
        cmap: str | Colormap = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
        colorbar: bool = True,
        colorbar_label: str = "",
        nan_color: str | None = "lightgray",
        rasterized: bool = True,
        scale_bar: bool | ScaleBarConfig = False,
        **kwargs: Any,
    ) -> matplotlib.axes.Axes:
        """Plot spatial field data over the environment.

        Automatically selects the appropriate visualization method based on the
        layout type to render bins with their actual geometric shapes:

        - **1D native grids**: Line plot over physical bin coordinates
        - **2D grid layouts**: Uses ``pcolormesh`` for crisp rectangular bins
        - **Hexagonal**: Colored hexagon patches via ``PatchCollection``
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
        **kwargs : Any
            Additional keyword arguments passed to underlying plot function
            (e.g., ``pcolormesh``, ``PatchCollection.set``).

        Returns
        -------
        matplotlib.axes.Axes
            The axes on which the field was plotted.

        Raises
        ------
        ValueError
            If ``field`` shape does not match ``(n_bins,)``.
        NotImplementedError
            If environment dimensionality is >2D (not supported for spatial plots).
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

        """
        # Validate field shape
        from neurospatial.environment.visualization import _plot_field

        return _plot_field(
            self,
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
        fields: Sequence[NDArray[np.float64]] | NDArray[np.float64],
        *,
        frame_times: NDArray[np.float64],
        backend: Literal["auto", "napari", "video", "html", "widget"] = "auto",
        save_path: str | None = None,
        overwrite: bool = False,
        speed: float = 1.0,
        cmap: str = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
        frame_labels: Sequence[str] | None = None,
        overlay_trajectory: NDArray[np.float64] | None = None,
        title: str = "Spatial Field Animation",
        dpi: int = 100,
        codec: str = "h264",
        bitrate: int = 5000,
        n_workers: int | None = None,
        dry_run: bool = False,
        image_format: Literal["png", "jpeg"] = "png",
        max_html_frames: int = 500,
        contrast_limits: tuple[float, float] | None = None,
        show_colorbar: bool = False,
        colorbar_label: str = "",
        overlays: list[OverlayProtocol] | None = None,
        show_regions: bool | list[str] = False,
        region_alpha: float = 0.3,
        scale_bar: bool | ScaleBarConfig = False,
        **kwargs: Any,
    ) -> Any:
        """Animate spatial fields over time with multiple backend options.

        Creates animations of spatial field data across different time points,
        with support for four different backends optimized for different use cases:
        Napari (GPU-accelerated interactive viewing), Video (MP4/WebM export),
        HTML (standalone shareable player), and Jupyter Widget (notebook integration).

        Parameters
        ----------
        fields : Sequence[NDArray[np.float64]] or NDArray[np.float64]
            Spatial field data to animate. Can be:
            - List of arrays: Each array shape (n_bins,) represents one frame
            - Single ndarray: Shape (n_frames, n_bins) with first dimension as time
            Each field must match the number of bins in this environment.
        backend : {"auto", "napari", "video", "html", "widget"}, default="auto"
            Animation backend to use:
            - "auto": Automatically selects based on context and data size
            - "napari": GPU-accelerated interactive viewer (best for large datasets)
            - "video": Export MP4/WebM with parallel rendering (best for publications)
            - "html": Standalone HTML player with instant scrubbing (best for sharing)
            - "widget": Jupyter widget with slider control (best for notebooks)
        save_path : str, optional
            Output file path. If provided, file extension determines format:
            - .mp4, .webm, .avi, .mov: video export (requires ffmpeg)
            - .html: standalone HTML player (no dependencies)
            - None: display interactively (napari or widget depending on context)
        overwrite : bool, default=False
            Allow replacing existing video/HTML files and nonempty HTML frames
            directories. By default, an existing target raises before rendering.
        speed : float, default=1.0
            Playback speed relative to real-time:

            - 1.0: Real-time playback (1 second of data = 1 second viewing)
            - 0.1: 10% speed (slow motion, good for replay analysis)
            - 2.0: 2x speed (fast forward)

            The actual playback fps is computed as:
            ``playback_fps = min(sample_rate_hz * speed, MAX_PLAYBACK_FPS)``

            where sample_rate_hz is inferred from frame_times.
            Playback is capped at MAX_PLAYBACK_FPS (60) for display compatibility.
        cmap : str, default="viridis"
            Matplotlib colormap name (e.g., "hot", "Blues", "viridis", "RdBu_r")
        vmin : float, optional
            Minimum value for colormap normalization. If None, computed from all frames.
        vmax : float, optional
            Maximum value for colormap normalization. If None, computed from all frames.
        frame_labels : Sequence[str], optional
            Labels for each frame (e.g., ["Trial 1", "Trial 2", ...]). If None,
            auto-generated as "Frame 1", "Frame 2", etc.
        overlay_trajectory : NDArray[np.float64], optional
            Trajectory to overlay on animation. Shape: (n_timepoints, n_dims).
            For 2D environments: rendered as track line. For higher dimensions:
            rendered as point cloud. Only supported by napari backend.
        title : str, default="Spatial Field Animation"
            Animation title shown in player controls or window title
        dpi : int, default=100
            Resolution for rendering frames (video and HTML backends). Higher values
            produce sharper images but larger file sizes.
        codec : str, default="h264"
            Video codec for MP4 export. Options: "h264", "h265", "vp9", "mpeg4".
            Only used with video backend.
        bitrate : int, default=5000
            Video bitrate in kbps. Higher values produce better quality but larger
            files. Only used with video backend.
        n_workers : int, optional
            Number of parallel workers for video rendering. If None, uses
            CPU count / 2. Set to 1 for serial rendering. Requires environment
            to be pickle-able (call `env.clear_cache()` if pickle errors occur).
            Only used with video backend.
        dry_run : bool, default=False
            If True, estimate rendering time and file size without actually rendering.
            Only used with video backend.
        image_format : {"png", "jpeg"}, default="png"
            Image format for frame embedding in HTML. JPEG produces smaller files
            but with lossy compression. Only used with HTML backend.
        max_html_frames : int, default=500
            Maximum number of frames allowed for HTML export. Prevents creating
            huge files that crash browsers. Only used with HTML backend.
        contrast_limits : tuple[float, float], optional
            Napari contrast limits (min, max). If provided, overrides vmin/vmax.
            Only used with napari backend.
        show_colorbar : bool, default=False
            Whether to include colorbar in rendered frames (not yet implemented)
        colorbar_label : str, default=""
            Label for colorbar axis (not yet implemented)
        overlays : list of overlay objects, optional
            List of overlay objects to render on top of the spatial field animation.
            Supported types: PositionOverlay, BodypartOverlay, HeadDirectionOverlay, VideoOverlay.
            Multiple overlays of the same type can be provided for multi-animal tracking.
            See neurospatial.animation.overlays for details. Default: None (no overlays).
        frame_times : NDArray[np.float64], shape (n_frames,)
            Timestamps for each frame in seconds. **Required** - provides the
            temporal structure of your data. Use timestamps from your data source
            (e.g., position timestamps, decoding time bins).

            Overlays with their own timestamps will be aligned via interpolation
            to these frame times.
        show_regions : bool or list of str, default=False
            Whether to render region overlays. If True, all regions defined in
            env.regions are rendered. If a list of region names, only those regions
            are rendered. Regions are rendered as semi-transparent polygons.
            Default: False (no region overlays).
        region_alpha : float, default=0.3
            Alpha transparency for region overlays. Range: [0.0, 1.0] where
            0.0 is fully transparent and 1.0 is fully opaque. Only used when
            show_regions is True or a list. Default: 0.3.
        scale_bar : bool or ScaleBarConfig, default=False
            Whether to add a scale bar to rendered frames. If True, uses default
            ScaleBarConfig with auto-calculated length. If ScaleBarConfig instance,
            uses provided configuration.

            - **napari backend**: Configures napari's native scale bar
            - **video/html/widget backends**: Adds scale bar to each rendered frame

            Default: False (no scale bar).
        **kwargs : dict
            Backend-specific parameters. Common backend-specific options:

            **Napari backend:**
            - ``layout`` : {"horizontal", "vertical", "grid"}, optional
              Layout for multi-field viewing (required when fields is a list of lists)
            - ``layer_names`` : list of str, optional
              Custom names for each layer in multi-field mode
            - ``cache_size`` : int, default=1000
              Maximum frames to cache (per-frame caching)
            - ``chunk_size`` : int, default=10
              Frames per chunk (chunked caching for >10K frames)
            - ``max_chunks`` : int, default=100
              Maximum chunks to cache

            **Video backend:**
            - ``crf`` : int, default=18
              Constant rate factor for quality (0=lossless, 51=worst)
            - ``preset`` : str, default="medium"
              Encoding speed preset (ultrafast, fast, medium, slow, veryslow)

            **Widget backend:**
            - ``initial_cache_size`` : int, optional
              Number of frames to pre-render
            - ``cache_limit`` : int, default=1000
              Maximum cache size

        Returns
        -------
        viewer or path or widget
            Return value depends on backend:
            - napari: napari.Viewer instance (blocking - shows window)
            - video: Path to saved video file
            - html: Path to saved HTML file
            - widget: ipywidgets.interact instance (displays automatically in notebook)

        Raises
        ------
        FileExistsError
            If an output already exists and overwrite is False.
        ValueError
            If field shapes don't match environment n_bins
        ImportError
            If required backend dependencies are not installed
        RuntimeError
            If ffmpeg is not available for video backend

        See Also
        --------
        plot_field : Plot static spatial field
        plot : Plot environment structure
        neurospatial.animation.subsample_frames : Subsample frames for large datasets

        Notes
        -----
        **Backend Selection (auto mode):**
        When backend="auto", the selection logic is:
        - If save_path has video extension (.mp4, .webm, etc.) → video backend
        - If save_path is .html → HTML backend
        - If >10,000 frames → napari backend (GPU acceleration needed)
        - If in Jupyter notebook → widget backend
        - Otherwise → napari backend (if available) or error

        **Layout Support:**
        All layout types are supported:
        - Grid layouts (RegularGrid, MaskedGrid, etc.): Direct rendering
        - Hexagonal/Triangular: Layout-aware rendering with proper shapes
        - Graph layouts (1D): Rendered as 1D line plots
        The animation automatically uses the appropriate renderer based on layout type.

        **Performance Tips:**
        - For large datasets (>10K frames), use memory-mapped arrays to avoid loading
          all data into RAM
        - Use napari backend for interactive exploration of large datasets
        - For video export, increase n_workers for faster parallel rendering
        - For HTML, use image_format="jpeg" to reduce file size (with quality loss)
        - Use subsample_frames() utility to downsample high-frequency recordings

        **Memory Considerations:**
        - Napari backend: Frames loaded on-demand (LRU cache), handles 900K+ frames
        - Video backend: Renders in chunks, temporary files cleaned up automatically
        - HTML backend: All frames embedded in single file (limit: 500 frames default)
        - Widget backend: Pre-renders first 500 frames, on-demand for rest

        Examples
        --------
        Create a short evolving field on a complete grid:

        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> x, y = np.meshgrid(np.arange(0.0, 24.0, 4.0), np.arange(0.0, 24.0, 4.0))
        >>> env = Environment.from_samples(np.c_[x.ravel(), y.ravel()], bin_size=4.0)
        >>> distances = env.distance_to([env.n_bins // 2])
        >>> fields = [np.exp(-distances / (4.0 + frame)) for frame in range(8)]
        >>> frame_times = np.arange(len(fields)) / 30.0

        Interactive exploration requires a display; video export requires ffmpeg:

        >>> viewer = env.animate_fields(
        ...     fields, frame_times=frame_times, backend="napari"
        ... )  # doctest: +SKIP
        >>> env.clear_cache()  # Clear caches before using parallel video workers
        >>> env.animate_fields(  # doctest: +SKIP
        ...     fields,
        ...     frame_times=frame_times,
        ...     backend="video",
        ...     save_path="learning.mp4",
        ...     speed=0.5,
        ...     n_workers=4,
        ...     frame_labels=[f"Trial {i + 1}" for i in range(len(fields))],
        ... )

        Share an HTML player, or use an interactive notebook widget:

        >>> path = env.animate_fields(
        ...     fields,
        ...     frame_times=frame_times,
        ...     backend="html",
        ...     save_path="exploration.html",
        ... )
        >>> widget_viewer = env.animate_fields(
        ...     fields, frame_times=frame_times, backend="widget"
        ... )
        VBox(...)

        Position, pose and heading overlays use coordinates in the environment frame:

        >>> from neurospatial.animation import (
        ...     PositionOverlay,
        ...     BodypartOverlay,
        ...     Skeleton,
        ...     HeadDirectionOverlay,
        ... )
        >>> trajectory = np.c_[
        ...     np.linspace(2.0, 18.0, len(fields)), np.full(len(fields), 10.0)
        ... ]
        >>> position_overlay = PositionOverlay(
        ...     positions=trajectory, color="red", trail_length=3
        ... )
        >>> skeleton = Skeleton.from_edge_list([("nose", "tail")], name="rodent")
        >>> bodypart_overlay = BodypartOverlay(
        ...     data={"nose": trajectory + [0.5, 0.0], "tail": trajectory - [0.5, 0.0]},
        ...     skeleton=skeleton,
        ...     colors={"nose": "red", "tail": "blue"},
        ... )
        >>> heading_overlay = HeadDirectionOverlay(
        ...     headings=np.linspace(0.0, np.pi, len(fields)), color="yellow"
        ... )
        >>> from neurospatial.regions import Regions
        >>> regions = Regions(env.regions.values())
        >>> _ = regions.add("goal", point=(18.0, 10.0))
        >>> env = env.with_regions(regions)
        >>> path = env.animate_fields(
        ...     fields,
        ...     frame_times=frame_times,
        ...     backend="html",
        ...     save_path="overlays.html",
        ...     overlays=[position_overlay, bodypart_overlay, heading_overlay],
        ...     show_regions=True,
        ... )

        Multiple animals can have separate position overlays:

        >>> second_animal = PositionOverlay(
        ...     positions=trajectory + [1.0, 1.0], color="blue"
        ... )
        >>> path = env.animate_fields(
        ...     fields,
        ...     frame_times=frame_times,
        ...     backend="html",
        ...     save_path="animals.html",
        ...     overlays=[position_overlay, second_animal],
        ... )

        Subsample a memory-mapped field array before rendering:

        >>> from neurospatial.animation import subsample_frames
        >>> fields_mmap = np.lib.format.open_memmap(
        ...     "fields.npy",
        ...     mode="w+",
        ...     dtype="float64",
        ...     shape=(len(fields), env.n_bins),
        ... )
        >>> fields_mmap[:] = np.asarray(fields)
        >>> fields_sub = subsample_frames(fields_mmap, target_fps=15, source_fps=30)
        >>> frame_times_sub = np.arange(len(fields_sub)) / 15.0
        >>> path = env.animate_fields(
        ...     fields_sub,
        ...     frame_times=frame_times_sub,
        ...     backend="html",
        ...     save_path="subsampled.html",
        ... )

        Timestamps align overlays recorded at a different sampling rate:

        >>> overlay_times = np.arange(60) / 250.0
        >>> trajectory_250hz = np.c_[np.linspace(2.0, 18.0, 60), np.full(60, 10.0)]
        >>> aligned_overlay = PositionOverlay(
        ...     positions=trajectory_250hz, times=overlay_times
        ... )
        >>> path = env.animate_fields(
        ...     fields,
        ...     frame_times=frame_times,
        ...     overlays=[aligned_overlay],
        ...     backend="html",
        ...     save_path="aligned.html",
        ... )

        """
        from neurospatial.environment.visualization import _animate_fields

        return _animate_fields(
            self,
            fields,
            frame_times=frame_times,
            backend=backend,
            save_path=save_path,
            overwrite=overwrite,
            speed=speed,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            frame_labels=frame_labels,
            overlay_trajectory=overlay_trajectory,
            title=title,
            dpi=dpi,
            codec=codec,
            bitrate=bitrate,
            n_workers=n_workers,
            dry_run=dry_run,
            image_format=image_format,
            max_html_frames=max_html_frames,
            contrast_limits=contrast_limits,
            show_colorbar=show_colorbar,
            colorbar_label=colorbar_label,
            overlays=overlays,
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

        Notes
        -----
        - Detection is cached; each access returns an independent array.
        - For large environments (>100,000 bins), initial computation may take a few seconds
        - Uses graph-based boundary detection (checks node degrees and connectivity)

        See Also
        --------
        Environment.neighbors : Get neighbors of a bin

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> data = np.array([[1, 1], [1, 2], [2, 1], [2, 2], [1, 3], [2, 3]])
        >>> env = Environment.from_samples(data, bin_size=1.0)
        >>> boundary = env.boundary_bins
        >>> print(f"Found {len(boundary)} boundary bins")  # doctest: +SKIP
        Found 6 boundary bins

        """
        from neurospatial.environment.metrics import _boundary_bins

        return _boundary_bins(self)

    @property
    def linearization_properties(
        self,
    ) -> dict[str, Any]:
        """
        Get linearization metadata for 1D environments.

        Returns a dictionary with linearization information, including:
        - `is_linearized_track`: Whether this is a 1D linearized environment
        - `linear_bin_edges`: Bin edges in linear coordinates (1D only)
        - `linear_bin_centers_1d`: Bin centers in linear coordinates (1D only)
        - `track_graph`: The underlying track graph (1D only)

        Returns
        -------
        dict[str, Any]
            Dictionary with linearization metadata. For non-1D environments,
            only contains `{'is_linearized_track': False}`.

        Notes
        -----
        - Metadata is cached; each access returns an independent dictionary.
        - Only 1D environments (created with `from_graph()`) have full linearization info
        - For N-D environments, returns `{'is_linearized_track': False}`

        See Also
        --------
        to_linear : Convert N-D positions to 1D linear coordinates
        linear_to_nd : Convert 1D linear coordinates to N-D positions
        Environment.from_graph : Create 1D linearized environment

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
        >>> props = env.linearization_properties  # doctest: +SKIP
        >>> print(props["is_linearized_track"])  # doctest: +SKIP
        True

        """
        from neurospatial.environment.metrics import _linearization_properties

        return _linearization_properties(self)

    @cached_property
    def _linearization_properties_cached(self) -> dict[str, Any]:
        from neurospatial.environment.metrics import _linearization_properties_cached

        return _linearization_properties_cached(self)

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
        >>> data = np.random.rand(50, 2) * 10
        >>> env = Environment.from_samples(data, bin_size=2.0)
        >>>
        >>> df = env.get_bin_attributes()
        >>> print(df.columns.tolist())  # doctest: +SKIP
        ['source_grid_flat_index', 'original_grid_nd_index', 'pos_dim0', 'pos_dim1']
        >>> print(df.shape)  # doctest: +SKIP
        (30, 4)

        """
        from neurospatial.environment.metrics import _get_bin_attributes

        return _get_bin_attributes(self)

    @cached_property
    def _bin_attributes_cached(self) -> pd.DataFrame:
        # Extract node attributes from connectivity graph
        from neurospatial.environment.metrics import _bin_attributes_cached

        return _bin_attributes_cached(self)

    def get_edge_attributes(self) -> pd.DataFrame:
        """
        Extract edge attributes as a pandas DataFrame.

        Returns a DataFrame where each row corresponds to an edge (connection
        between bins), with columns for edge attributes like distances, vectors,
        and any custom metadata.

        Returns
        -------
        pd.DataFrame, shape (n_edges, n_attributes)
            DataFrame with edge attributes. Always includes:
            - `'source'`: Source bin ID
            - `'target'`: Target bin ID
            - `'distance'`: Stored navigation distance between bins
            - `'vector'`: Displacement vector tuple
            - `'edge_id'`: Unique edge ID
            - `'angle_2d'`: Angle for 2D layouts (optional)

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
        >>> data = np.random.rand(50, 2) * 10
        >>> env = Environment.from_samples(data, bin_size=2.0)
        >>>
        >>> df = env.get_edge_attributes()
        >>> print(df.columns.tolist())  # doctest: +SKIP
        ['source', 'target', 'distance', 'vector', 'edge_id', 'angle_2d']
        >>> print(df.shape)  # doctest: +SKIP
        (100, 6)

        """
        from neurospatial.environment.metrics import _get_edge_attributes

        return _get_edge_attributes(self)

    @cached_property
    def _edge_attributes_cached(self) -> pd.DataFrame:
        # Extract edge attributes from connectivity graph
        from neurospatial.environment.metrics import _edge_attributes_cached

        return _edge_attributes_cached(self)

    def compute_kernel(
        self,
        bandwidth: float,
        *,
        mode: KernelMode = "density",
        cache: bool = True,
    ) -> NDArray[np.float64]:
        """Compute the finite-volume diffusion kernel for smoothing operations.

        Wrapper around :func:`neurospatial.ops.diffusion.diffusion_kernel` that
        resolves this environment's layout geometry and cell volumes, so
        ``bandwidth`` is the true physical standard deviation (σ) of the
        smoothing on any supported layout, independent of bin size.

        Parameters
        ----------
        bandwidth : float
            Smoothing bandwidth in physical units (the standard deviation σ of
            the diffusion), must be > 0. This is the true physical σ: a
            point source smoothed by this kernel has physical standard
            deviation ``bandwidth`` regardless of bin size.
        mode : {'transition', 'density', 'average'}, default='density'
            Kernel orientation, distinguished by the input field type:

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
            default); ``clear_cache(kernels=True)`` releases them.

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
        >>> env = Environment.from_samples(data, bin_size=2.0)  # doctest: +SKIP
        >>> # Compute kernel for smoothing
        >>> kernel = env.compute_kernel(bandwidth=5.0, mode="density")  # doctest: +SKIP
        >>> # Apply to field
        >>> smoothed_field = kernel @ field  # doctest: +SKIP

        """
        from neurospatial.environment.fields import _compute_kernel

        return _compute_kernel(self, bandwidth, mode=mode, cache=cache)

    def compute_gaussian_kernel(self, bandwidth: float) -> NDArray[np.float64]:
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
        clear_cache(kernels=True) releases the cached weights.
        """
        from neurospatial.environment.fields import _compute_gaussian_kernel

        return _compute_gaussian_kernel(self, bandwidth)

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
        Every export has independent sparse and NumPy headers over immutable
        buffers. Replacing an exported matrix's buffers or changing an array's
        shape cannot change the environment. Reading the export does not copy
        O(n_edges) data. clear_cache(cached_properties=True) rebuilds on demand.

        Raises
        ------
        NotImplementedError
            If the stored binning cannot supply numerical diffusion geometry.
        """
        from neurospatial.environment._publication import _diffusion_geometry_header

        return _diffusion_geometry_header(self._diffusion_geometry)

    def diffusion_accuracy(self, bandwidth: float) -> float:
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
        from neurospatial.ops.diffusion import _diffusion_apply_rtol

        return _diffusion_apply_rtol(self._diffusion_eigenbasis, self.n_bins, bandwidth)

    def smooth(
        self,
        field: NDArray[np.float64],
        bandwidth: float,
        *,
        mode: KernelMode = "average",
    ) -> NDArray[np.float64]:
        """Apply diffusion kernel smoothing to a field.

        This method smooths bin-valued fields using diffusion kernels computed
        via the graph Laplacian. It works uniformly across all layout types
        (grids, graphs, meshes) and respects the connectivity structure.

        Parameters
        ----------
        field : NDArray[np.float64], shape (n_bins,)
            Field values per bin to smooth. Must be a 1-D array with length
            equal to n_bins.
        bandwidth : float
            Smoothing bandwidth in physical units (σ). Controls the scale
            of spatial smoothing. Must be positive.
        mode : {'transition', 'density', 'average'}, default='average'
            Smoothing mode, distinguished by the **input type**. The default
            ``'average'`` preserves the units and level of an intensive field
            such as a firing-rate map. For the previous count-to-density
            behavior, pass ``mode='density'`` explicitly.

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
        smoothed : NDArray[np.float64], shape (n_bins,)
            Smoothed field values.

        Raises
        ------
        ValueError
            If field has wrong shape, wrong dimensionality, bandwidth is not
            positive, or mode is invalid.

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

        Examples
        --------
        >>> # Smooth an intensive rate map without changing its units or level
        >>> smoothed_rate = env.smooth(firing_rate, bandwidth=5.0)  # doctest: +SKIP

        >>> # Smooth spike counts (mass-conserving)
        >>> smoothed_counts = env.smooth(
        ...     spike_counts, bandwidth=5.0, mode="transition"
        ... )  # doctest: +SKIP
        >>> # Total spikes preserved
        >>> assert np.isclose(
        ...     smoothed_counts.sum(), spike_counts.sum()
        ... )  # doctest: +SKIP

        >>> # Turn spike counts into a smoothed density (KDE)
        >>> spike_density = env.smooth(
        ...     spike_counts, bandwidth=3.0, mode="density"
        ... )  # doctest: +SKIP

        >>> # Smooth a discrete probability distribution (mass, sums to 1)
        >>> smoothed_prob = env.smooth(
        ...     posterior, bandwidth=2.0, mode="transition"
        ... )  # doctest: +SKIP

        """
        # Input validation
        from neurospatial.environment.fields import _smooth

        return _smooth(self, field, bandwidth, mode=mode)

    def diffuse(
        self,
        fields: Any,
        bandwidth: float,
        *,
        mode: KernelMode = "density",
        backend: Literal["numpy", "jax"] = "numpy",
    ) -> NDArray[np.float64]:
        """Matrix-free finite-volume diffusion smoothing (no ``(n, n)`` kernel).

        Applies the finite-volume heat operator ``H = exp(-t L)``, ``t = σ²/2``,
        ``L = M⁻¹(D − W)``, to one or more fields **without ever materializing
        the dense ``(n_bins, n_bins)`` kernel**. It uses a cached, per-component,
        bandwidth-aware **truncated symmetric eigenbasis** of
        ``S = M^{-1/2}(D − W) M^{-1/2}``, so both time and memory scale with
        ``n_bins × rank`` (``rank ~ measure(domain)/σ^d``), not ``n_bins²``. The
        eigenbasis depends only on geometry, so it is built once and reused across
        every bandwidth, mode, and neuron (auto-invalidated when the environment
        is mutated).

        ``diffuse`` is a **pure linear operator**: it applies **no output clip
        and no renormalization**. The always-retained per-component null mode
        makes mass conservation exact under truncation (``mode="transition"``
        preserves ``sum``; a constant field is preserved by ``mode="average"``),
        so smoothing a **signed** field is preserved. Where the result must be
        non-negative (a density, a rate), the **caller** clips it (as the KDE
        encoders do); ``diffuse`` itself never does.

        Parameters
        ----------
        fields : NDArray[np.float64], shape (n_bins,) or (n_bins, n_fields)
            Field(s) to smooth, indexed by bin. A 1-D field is smoothed and
            returned 1-D; a 2-D batch smooths every column in one pass. NaN/Inf
            are not validated (``diffuse`` is a raw linear operator: NaN in →
            NaN out); callers needing validation use :meth:`smooth`.
        bandwidth : float
            Physical smoothing standard deviation σ (> 0), independent of bin
            size on any supported layout.
        mode : {'transition', 'density', 'average'}, default='density'
            Kernel orientation, matching :meth:`compute_kernel` / :meth:`smooth`
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
        ValueError
            If ``bandwidth`` is not positive, ``mode`` is invalid, ``backend`` is
            invalid, or ``fields`` is not 1-D/2-D with leading dim ``n_bins``.

        See Also
        --------
        smooth : 1-D smoothing with input validation (delegates here).
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
        cannot beat the dense kernel, so ``diffuse`` builds a **transient** dense
        basis (applied then dropped, never cached) with the same large-matrix
        ``UserWarning`` as :meth:`compute_kernel`.
        """
        from neurospatial.environment.fields import _diffuse

        return _diffuse(self, fields, bandwidth, mode=mode, backend=backend)

    @cached_property
    def _face_geometry(self) -> FaceGeometry:
        """Validated cell faces, cached for this immutable geometry.

        Shared by the diffusion snapshot and the gradient and divergence
        operators. Read it through :meth:`_require_face_geometry`, which names
        the analysis when the layout has no cell faces.
        """
        from neurospatial.environment.fields import _face_geometry

        return _face_geometry(self)

    def _require_face_geometry(self, operation: str) -> FaceGeometry:
        """Cached cell faces, or an actionable error naming ``operation``.

        Parameters
        ----------
        operation : str
            Analysis named in the error when the layout has no cell faces.

        Returns
        -------
        FaceGeometry
            Validated faces and cell measures in authoritative bin-pair order.
        """
        from neurospatial.environment.fields import _require_face_geometry

        return _require_face_geometry(self, operation)

    @cached_property
    def _diffusion_geometry(
        self,
    ) -> DiffusionGeometry:
        """Cached diffusion geometry ``(W, volumes, n_components, labels)``.

        Geometry-only, so it is built once and discarded by
        ``clear_cache(cached_properties=True)``. Shared by dense kernels,
        :meth:`diffuse`, smoothness bases, and the W-component support gates.
        """
        from neurospatial.environment.fields import _diffusion_geometry

        return _diffusion_geometry(self)

    @cached_property
    def _diffusion_eigenbasis(self) -> DiffusionEigenbasisCache:
        """Growable single-entry truncated-eigenbasis cache for :meth:`diffuse`.

        Returns a small mutable holder (the max-rank truncated basis, grown by
        replace, plus a per-``(sigma, tol)`` resolved-rank memo) that
        :func:`neurospatial.ops.diffusion.diffusion_apply` mutates in place. The
        private cached property owns it, so clearing cached properties drops it
        wholesale. It holds a single basis strictly below ``dense_fraction·n``
        — never a persistent ``(n, n)``.
        """
        from neurospatial.environment.fields import _diffusion_eigenbasis

        return _diffusion_eigenbasis(self)

    @cached_property
    def _gaussian_kernels(self) -> dict[float, NDArray[np.float64]]:
        """Dense Gaussian-KDE weight matrices for this geometry, by bandwidth.

        The smoothing adapter fills this holder with the most recently used
        kernels (``_constants.MAX_CACHED_KERNELS``). It is emptied by
        ``clear_cache(kernels=True)`` and freed with the environment.
        """
        from neurospatial.environment.fields import _gaussian_kernels

        return _gaussian_kernels(self)

    @cached_property
    def _mrf_eigenbasis(self) -> MRFEigenbasisCache:
        """Growable global eigenbasis holder for the MRF penalty-basis resolver.

        A mutable ``{"Q": (n_bins, G), "Lam": (G,), "mode_comp": (G,), "G": int}``
        holding the ``G`` globally-smallest modes of the symmetric conjugate
        ``S`` and each mode's source W-component, grown by replace by
        :func:`neurospatial.ops.diffusion._ensure_global_modes`. Geometry-only
        (no ``(sigma, tol)`` key — the MRF penalty basis is bandwidth-independent)
        and dropped wholesale by ``clear_cache(cached_properties=True)``.
        Starts empty (``G == 0``) and never stores a dense ``(n, n)`` basis
        (dense solves stay call-local). Distinct from
        :meth:`_diffusion_eigenbasis`, which is bandwidth-keyed for ``diffuse``.
        """
        from neurospatial.environment.fields import _mrf_eigenbasis

        return _mrf_eigenbasis(self)

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
        from neurospatial.environment.fields import _mrf_basis

        return _mrf_basis(self, occupancy, rank=rank)

    def transitions(
        self,
        bins: NDArray[np.int32] | None = None,
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
        max_gap: float | None = 0.5,
        epochs: Any = None,
    ) -> scipy.sparse.csr_matrix:
        """Compute transition matrix (empirical or model-based).

        Two modes of operation:

        1. **Empirical**: Count observed transitions from trajectory data.
           Requires bins OR (times + positions). Analyzes actual behavior.

        2. **Model-based**: Generate theoretical transitions from graph structure.
           Requires method parameter. Models expected behavior.

        Parameters
        ----------
        bins : NDArray[np.int32], shape (n_samples,), optional
            [Empirical mode] Precomputed bin sequence. If None, computed from
            times/positions. Must contain valid bin indices in range [0, n_bins).
            Outside values (-1) are not allowed.
        times : NDArray[np.float64], shape (n_samples,), optional
            [Empirical mode] Timestamps in seconds. Must be provided together
            with positions.
        positions : NDArray[np.float64], shape (n_samples, n_dims), optional
            [Empirical mode] Position coordinates matching environment dimensions.
            Must be provided together with times.
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

        See Also
        --------
        bin_sequence : Convert trajectory to bin indices.
        occupancy : Compute time spent in each bin.
        compute_kernel : Low-level diffusion kernel computation.

        Notes
        -----
        Each run of samples with gaps no longer than ``max_gap`` (inside ``epochs``)
        is analyzed as a separate recording; no segment spans a pause.

        When times are supplied, a pair (k, k + lag) counts only if every
        interval from k through k + lag - 1 is valid. Raw bin-only inputs
        and model-based transitions have no time gate.

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
        >>> T_empirical = env.transitions(
        ...     times=times, positions=positions
        ... )  # doctest: +SKIP
        >>> T_lag2 = env.transitions(
        ...     bins=bin_sequence, lag=2, allow_teleports=True
        ... )  # doctest: +SKIP
        >>> T_random = env.transitions(method="random_walk")  # doctest: +SKIP
        >>> T_diffusion = env.transitions(
        ...     method="diffusion", bandwidth=5.0
        ... )  # doctest: +SKIP
        >>> diff = (T_empirical - T_diffusion).toarray()  # doctest: +SKIP

        """
        from neurospatial.environment.trajectory import _transitions

        return _transitions(
            self,
            bins,
            times=times,
            positions=positions,
            lag=lag,
            allow_teleports=allow_teleports,
            method=method,
            bandwidth=bandwidth,
            normalize=normalize,
            max_gap=max_gap,
            epochs=epochs,
        )
