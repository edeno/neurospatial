"""Factory methods for creating Environment instances.

This module contains classmethods that provide different ways to construct
Environment objects from various input sources: sample data, graphs, polygons,
masks, images, and custom layouts.

Classes
-------
EnvironmentFactories
    Mixin class providing factory classmethods for Environment creation.

Notes
-----
This is a mixin class (plain class, NOT a dataclass) that provides factory
classmethods to the Environment dataclass via inheritance.

Type Checking
-------------
Uses TYPE_CHECKING guard to prevent circular imports when referencing Environment
in type hints. At runtime, TYPE_CHECKING is False, so no import occurs.

"""

from __future__ import annotations

import itertools
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from neurospatial.layout.factories import (
    LayoutType,
    create_layout,
    list_available_layouts,
)
from neurospatial.layout.factories import _normalize_name as _normalize_layout_name
from neurospatial.regions import Regions

if TYPE_CHECKING:
    pass
    from neurospatial.environment.core import Environment

try:
    import shapely.geometry as _shp

    _HAS_SHAPELY = True
except ModuleNotFoundError:
    _HAS_SHAPELY = False

    class _Shp:
        class Polygon:
            pass

    _shp = _Shp


PolygonType = type[_shp.Polygon]


def _add_edge_with_distance(graph: nx.Graph, u: Any, v: Any, edge_id: int) -> None:
    """Add an edge to `graph` with `distance` (from node pos) and `edge_id`.

    Only ``distance`` is consumed by neurospatial's linearization path
    (``_get_graph_bins`` builds its own edge_id map from edge enumeration and
    never reads the input ``edge_id``). The ``edge_id`` attribute is set for
    interoperability/parity with ``track_linearization.make_track_graph`` but is
    not required for ``to_linear()`` to work.
    """
    p1 = np.asarray(graph.nodes[u]["pos"], dtype=float)
    p2 = np.asarray(graph.nodes[v]["pos"], dtype=float)
    graph.add_edge(u, v, distance=float(np.linalg.norm(p2 - p1)), edge_id=edge_id)


def _assemble_maze_graph(
    kind: str,
    node_positions: dict[Any, Sequence[float]],
) -> tuple[nx.Graph, list[tuple[Any, Any]]]:
    """Assemble a standard W / plus / T maze graph from ordered node positions.

    Topology is derived from the DOCUMENTED ORDER of ``node_positions`` (a
    coordinate-system-independent contract), never from x/y heuristics. The
    required ordering per `kind` is:

    - ``"plus"`` (5 nodes): ``[center, arm1, arm2, arm3, arm4]`` -> edges
      center--each arm.
    - ``"t"`` (4 nodes): ``[stem_end, junction, arm_left, arm_right]`` -> edges
      stem_end--junction, junction--arm_left, junction--arm_right.
    - ``"w"`` (6 nodes): ``[base_left, base_mid, base_right, arm_left, arm_mid,
      arm_right]`` -> connector edges base_left--base_mid--base_right plus arm
      edges base_left--arm_left, base_mid--arm_mid, base_right--arm_right.

    Because the topology comes from order alone, the SAME graph is produced for
    any consistent coordinate system (e.g. y increasing downward, or arms in
    -y); coordinates are stored only as ``pos`` attributes / edge distances.

    Parameters
    ----------
    kind : {"w", "plus", "t"}
        Normalized (lower-case) maze kind.
    node_positions : dict
        Ordered mapping of node label -> (x, y) coordinate. The insertion order
        defines the topology (see above). Python dicts preserve insertion order.

    Returns
    -------
    graph : networkx.Graph
        Track graph with `pos` node attributes and `distance` edge weights.
    edge_order : list of (node, node)
        Ordered edge list for linearization.

    Raises
    ------
    ValueError
        If `node_positions` does not contain the expected number of nodes for
        `kind`, or if `kind` is unknown.
    """
    labels = list(node_positions.keys())
    graph = nx.Graph()
    for label, pos in node_positions.items():
        graph.add_node(label, pos=tuple(float(c) for c in pos))

    edge_order: list[tuple[Any, Any]] = []
    edge_id = 0

    if kind == "plus":
        # Order contract: [center, arm1, arm2, arm3, arm4].
        # Star topology: center connects to each of the four arm tips.
        if len(labels) != 5:
            raise ValueError(
                "A plus maze needs exactly 5 nodes in `node_positions`, ordered "
                "[center, arm1, arm2, arm3, arm4]: one center followed by four "
                f"arm tips. Got {len(labels)}."
            )
        center = labels[0]
        for arm in labels[1:]:
            _add_edge_with_distance(graph, center, arm, edge_id)
            edge_order.append((center, arm))
            edge_id += 1

    elif kind == "t":
        # Order contract: [stem_end, junction, arm_left, arm_right].
        # Stem (stem_end->junction) + crossbar (junction->each arm tip).
        if len(labels) != 4:
            raise ValueError(
                "A T maze needs exactly 4 nodes in `node_positions`, ordered "
                "[stem_end, junction, arm_left, arm_right]. "
                f"Got {len(labels)}."
            )
        stem_end, junction, arm_left, arm_right = labels
        _add_edge_with_distance(graph, stem_end, junction, edge_id)
        edge_order.append((stem_end, junction))
        edge_id += 1
        for arm in (arm_left, arm_right):
            _add_edge_with_distance(graph, junction, arm, edge_id)
            edge_order.append((junction, arm))
            edge_id += 1

    elif kind == "w":
        # Order contract:
        #   [base_left, base_mid, base_right, arm_left, arm_mid, arm_right].
        # Horizontal connector along the first three base nodes, then one
        # vertical arm edge joining each base node to the arm tip at the
        # matching position in the documented order.
        if len(labels) != 6:
            raise ValueError(
                "A W maze needs exactly 6 nodes in `node_positions`, ordered "
                "[base_left, base_mid, base_right, arm_left, arm_mid, "
                "arm_right]: three base/junction nodes followed by three arm "
                f"tips (5 base+arm edges). Got {len(labels)}."
            )
        base = labels[:3]
        arms = labels[3:]
        # Horizontal connector along the base (base_left -> base_mid -> base_right).
        for left, right in itertools.pairwise(base):
            _add_edge_with_distance(graph, left, right, edge_id)
            edge_order.append((left, right))
            edge_id += 1
        # Vertical arms: each base node up to the arm at the matching position.
        for base_node, arm_node in zip(base, arms, strict=True):
            _add_edge_with_distance(graph, base_node, arm_node, edge_id)
            edge_order.append((base_node, arm_node))
            edge_id += 1

    else:
        raise ValueError(f"Unknown maze kind {kind!r}")

    return graph, edge_order


# A spatial environment has very few dimensions (1-D linear tracks, 2-D open
# fields, occasionally 3-D). A column count above this, *combined with* more
# columns than rows, signals a transposed `positions` array rather than a
# genuine high-D environment (see `from_samples`).
_MAX_PLAUSIBLE_SPATIAL_DIMS = 8


class EnvironmentFactories:
    """Factory methods mixin for creating Environment instances.

    This mixin provides classmethods for constructing Environment objects
    from different input sources and representations. All methods are classmethods
    that return fully initialized Environment instances.

    Methods
    -------
    from_samples(positions, bin_size, ...)
        Create environment by discretizing sample data into bins.
    from_graph(graph, edge_order, edge_spacing, bin_size, ...)
        Create 1D linearized track environment from graph structure.
    from_polygon(polygon, bin_size, ...)
        Create 2D grid environment masked by Shapely polygon.
    from_grid_mask(active_mask, grid_edges, ...)
        Create environment from pre-defined boolean mask and grid edges.
    from_pixel_mask(image_mask, pixel_size, ...)
        Create 2D environment from binary image mask.
    from_layout(kind, layout_params, ...)
        Create environment with specified layout type and parameters.

    Notes
    -----
    This is a mixin class (plain class, NOT @dataclass). Only the Environment
    class in core.py should use the @dataclass decorator. Mixins must be plain
    classes to avoid dataclass field inheritance conflicts with multiple inheritance.

    All classmethods automatically work on Environment when inherited due to Python's
    method resolution order (MRO). When called as `Environment.from_samples(...)`,
    the `cls` parameter is Environment, not EnvironmentFactories.

    Examples
    --------
    Create environment from sample data:

    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> positions = np.random.rand(1000, 2) * 100  # cm
    >>> env = Environment.from_samples(positions, bin_size=5.0)

    Create environment from graph (1D track):

    >>> import networkx as nx  # doctest: +SKIP
    >>> graph = nx.path_graph(10)  # doctest: +SKIP
    >>> for i, node in enumerate(graph.nodes()):  # doctest: +SKIP
    ...     graph.nodes[node]["pos"] = (i * 10.0, 0.0)
    >>> env = Environment.from_graph(  # doctest: +SKIP
    ...     graph=graph, edge_order=list(graph.edges()), edge_spacing=0.0, bin_size=2.0
    ... )

    """

    @classmethod
    def from_samples(
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
        fill_holes: bool = False,
        close_gaps: bool = False,
        add_boundary_bins: bool = False,
        connect_diagonal_neighbors: bool = True,
        **layout_specific_kwargs: Any,
    ) -> Environment:
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
        try:
            positions = np.asarray(positions, dtype=float)
        except (TypeError, ValueError) as e:
            actual_type = type(positions).__name__
            raise TypeError(
                f"positions must be a numeric array-like object (e.g., numpy array, "
                f"list of lists, pandas DataFrame). Got {actual_type}: {positions!r}"
            ) from e

        if positions.ndim != 2:
            raise ValueError(
                f"positions must be a 2D array of shape (n_points, n_dims), "
                f"got shape {positions.shape}.",
            )

        # Warn on a likely-transposed positions array before building the grid.
        # The classic footgun is np.array([x, y]) -- shape (n_dims, n_samples) --
        # instead of column-stacking, so `positions` is transposed and silently
        # builds a nonsensical high-D env. Two signals together flag it: (1) more
        # columns (apparent dims) than rows (apparent samples) -- real position
        # data has many more samples than dims; and (2) an implausibly high
        # dimension count for a spatial environment. This is only a *heuristic*
        # (a genuine low-sample high-D env, e.g. shape (1, 9), trips it too), so
        # it warns rather than rejects -- from_samples supports N-dimensional
        # environments. A truly catastrophic transpose (huge n_dims) is still
        # caught downstream by the int64 bin-count guard in
        # layout.helpers.utils, so nothing silently OOMs.
        n_samples, n_dims = positions.shape
        if n_dims > n_samples and n_dims > _MAX_PLAUSIBLE_SPATIAL_DIMS:
            warnings.warn(
                f"positions has {n_dims} columns (spatial dimensions) but only "
                f"{n_samples} rows (samples) -- fewer samples than dimensions "
                f"(shape {positions.shape}). If this is a genuine "
                f"{n_dims}-D environment, ignore this; otherwise positions is "
                f"likely transposed (it must be (n_samples, n_dims), one row per "
                f"sample) -- pass positions.T.",
                UserWarning,
                stacklevel=2,
            )

        # Validate bin_size early to provide helpful error messages
        if not isinstance(bin_size, (int, float, list, tuple, np.ndarray)):
            actual_type = type(bin_size).__name__
            raise TypeError(
                f"bin_size must be a numeric value or sequence of numeric values. "
                f"Got {actual_type}: {bin_size!r}"
            )

        # Convert LayoutType enum to string if needed, then normalize
        layout_str = layout.value if isinstance(layout, LayoutType) else layout
        layout_normalized = _normalize_layout_name(layout_str)

        # Check if this is a RegularGrid or Hexagonal layout (the only ones supporting from_samples)
        if layout_normalized not in ("regulargrid", "hexagonal"):
            available = list_available_layouts()
            raise NotImplementedError(
                f"Layout '{layout_str}' (normalized: '{layout_normalized}') is not supported "
                f"by from_samples(). Only 'RegularGrid' and 'Hexagonal' layouts are supported. "
                f"For other layouts, use from_layout() or from_grid_mask(). "
                f"Available layouts: {', '.join(available)}"
            )

        # Build the dict of layout parameters
        # Common parameters for all layouts
        common_params = {
            "positions": positions,
            "infer_active_bins": infer_active_bins,
            "bin_count_threshold": bin_count_threshold,
        }

        # Layout-specific parameters
        specific_params = {
            "regulargrid": {
                "bin_size": bin_size,
                "add_boundary_bins": add_boundary_bins,
                "dilate": dilate,
                "fill_holes": fill_holes,
                "close_gaps": close_gaps,
                "connect_diagonal_neighbors": connect_diagonal_neighbors,
            },
            "hexagonal": {
                "hexagon_width": bin_size,
            },
        }

        # Build final params dict (validation already done above)
        layout_params = {
            **common_params,
            **specific_params[layout_normalized],
            **layout_specific_kwargs,
        }

        env = cls.from_layout(kind=layout_str, layout_params=layout_params, name=name)
        if units is not None:
            env.units = units
        if frame is not None:
            env.frame = frame
        return env

    # ------------------------------------------------------------------
    # Experiment-shaped presets
    #
    # These classmethods speak experiment vocabulary (open field, linear
    # track, W/plus/T maze) and delegate to the general-purpose ``from_*``
    # factories. Track/maze presets require an EXPLICIT topology spec --
    # raw positions alone cannot infer a linear/W/plus/T graph. Only
    # ``open_field`` is positions-based.
    # ------------------------------------------------------------------

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
    ) -> Environment:
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
        return cls.from_samples(
            positions,
            bin_size,
            name=name,
            units=units,
            frame=frame,
            layout=layout,
            infer_active_bins=infer_active_bins,
            bin_count_threshold=bin_count_threshold,
            dilate=dilate,
            fill_holes=True,
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
        name: str = "",
        units: str | None = None,
        frame: str | None = None,
    ) -> Environment:
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
        if endpoints is not None and node_positions is not None:
            raise ValueError(
                "Provide exactly one of `endpoints` or `node_positions`, not both."
            )

        if endpoints is not None:
            points = [tuple(float(c) for c in p) for p in endpoints]
            if len(points) != 2:
                raise ValueError(
                    "`endpoints` must contain exactly two (x, y) points defining "
                    f"the ends of a straight track. Got {len(points)} point(s)."
                )
        elif node_positions is not None:
            points = [tuple(float(c) for c in p) for p in node_positions]
            if len(points) < 2:
                raise ValueError(
                    "`node_positions` must contain at least two (x, y) waypoints "
                    f"for a piecewise-linear track. Got {len(points)} point(s)."
                )
        else:
            raise ValueError(
                "linear_track requires an explicit topology: pass `endpoints` "
                "(two (x, y) points for a straight track) or `node_positions` "
                "(a sequence of (x, y) waypoints for a piecewise-linear track). "
                "Raw positions cannot be used to infer a 1D track -- a point "
                "cloud is indistinguishable from an open arena. For an open "
                "arena use Environment.open_field(positions, bin_size)."
            )

        graph = nx.Graph()
        for i, pos in enumerate(points):
            graph.add_node(i, pos=pos)
        edge_order: list[tuple[Any, Any]] = []
        for i in range(len(points) - 1):
            _add_edge_with_distance(graph, i, i + 1, edge_id=i)
            edge_order.append((i, i + 1))

        total_length = sum(float(graph.edges[u, v]["distance"]) for u, v in edge_order)
        if total_length <= 0.0:
            source = "endpoints" if endpoints is not None else "waypoints"
            raise ValueError(
                f"linear_track {source} are coincident (total track length is 0); "
                f"a 1-D track needs at least two distinct points. Got points: "
                f"{points}."
            )

        env = cls.from_graph(
            graph=graph,
            edge_order=edge_order,
            edge_spacing=edge_spacing,
            bin_size=bin_size,
            name=name,
        )
        if units is not None:
            env.units = units
        if frame is not None:
            env.frame = frame
        return env

    @classmethod
    def maze(
        cls,
        kind: str,
        *,
        track_graph: nx.Graph | None = None,
        node_positions: dict[Any, Sequence[float]] | None = None,
        bin_size: float,
        edge_spacing: float | Sequence[float] = 0.0,
        name: str = "",
        units: str | None = None,
        frame: str | None = None,
    ) -> Environment:
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
        allowed = ("w", "plus", "t")
        kind_normalized = kind.lower() if isinstance(kind, str) else kind
        if kind_normalized not in allowed:
            raise ValueError(
                f"Unknown maze kind {kind!r}. `kind` must be one of "
                f"{allowed} (W maze, plus/cross maze, or T maze)."
            )

        if track_graph is not None and node_positions is not None:
            raise ValueError(
                "Provide exactly one of `track_graph` or `node_positions`, not both."
            )

        if track_graph is not None:
            # `kind` is validated above but cannot be reconciled with an
            # arbitrary supplied graph; the graph topology is authoritative.
            warnings.warn(
                f"maze(kind={kind!r}, track_graph=...): `kind` is not consulted "
                "when an explicit `track_graph` is supplied; the supplied "
                "`track_graph` is authoritative for the maze topology.",
                UserWarning,
                stacklevel=2,
            )
            # Operate on a copy so the caller's graph is never mutated.
            graph = track_graph.copy()
            edge_order = list(graph.edges())
            # Fill in `distance` from node positions where missing; this is the
            # only edge attribute the neurospatial linearization path consumes
            # (`_get_graph_bins` builds its own edge_id map from edge
            # enumeration and never reads an input `edge_id`). We also set
            # `edge_id` for interoperability/parity with
            # ``track_linearization.make_track_graph``, but neurospatial's
            # ``to_linear()`` does not consume it.
            for assigned_id, (u, v) in enumerate(graph.edges()):
                if "distance" not in graph.edges[u, v]:
                    p1 = np.asarray(graph.nodes[u]["pos"], dtype=float)
                    p2 = np.asarray(graph.nodes[v]["pos"], dtype=float)
                    graph.edges[u, v]["distance"] = float(np.linalg.norm(p2 - p1))
                if "edge_id" not in graph.edges[u, v]:
                    graph.edges[u, v]["edge_id"] = assigned_id
        elif node_positions is not None:
            graph, edge_order = _assemble_maze_graph(kind_normalized, node_positions)
        else:
            raise ValueError(
                f"maze({kind!r}) requires an explicit topology: pass a ready "
                "`track_graph` (a networkx.Graph with `pos` node attributes) or "
                "`node_positions` (a mapping of node label -> (x, y) coordinate) "
                "from which the standard topology is assembled. Raw positions "
                "cannot be used to infer maze topology."
            )

        total_length = sum(float(graph.edges[u, v]["distance"]) for u, v in edge_order)
        if total_length <= 0.0:
            node_pos = {
                n: tuple(float(c) for c in graph.nodes[n]["pos"]) for n in graph.nodes
            }
            raise ValueError(
                f"maze({kind!r}) node positions are coincident (total track "
                "length is 0); a 1-D track needs at least two distinct points. "
                f"Got node positions: {node_pos}."
            )

        env = cls.from_graph(
            graph=graph,
            edge_order=edge_order,
            edge_spacing=edge_spacing,
            bin_size=bin_size,
            name=name,
        )
        if units is not None:
            env.units = units
        if frame is not None:
            env.frame = frame
        return env

    @classmethod
    def from_graph(
        cls,
        graph: nx.Graph,
        edge_order: list[tuple[Any, Any]],
        edge_spacing: float | Sequence[float],
        bin_size: float,
        *,
        name: str = "",
    ) -> Environment:
        """Create an Environment from a user-defined graph structure.

        This method is used for 1D environments where the spatial layout is
        defined by a graph, an ordered list of its edges, and spacing between
        these edges. The track is then linearized and binned.

        Parameters
        ----------
        graph : nx.Graph
            The NetworkX graph defining the track segments. Nodes are expected
            to have a 'pos' attribute for their N-D coordinates.
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
        layout_params = {
            "graph_definition": graph,
            "edge_order": edge_order,
            "edge_spacing": edge_spacing,
            "bin_size": bin_size,
        }
        return cls.from_layout(kind="Graph", layout_params=layout_params, name=name)

    @classmethod
    def from_polygon(
        cls,
        polygon: PolygonType,
        bin_size: float | Sequence[float],
        *,
        name: str = "",
        connect_diagonal_neighbors: bool = True,
    ) -> Environment:
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
        connect_diagonal_neighbors : bool, optional
            Whether to connect diagonally adjacent active grid cells.
            Defaults to True.

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
        layout_params = {
            "polygon": polygon,
            "bin_size": bin_size,
            "connect_diagonal_neighbors": connect_diagonal_neighbors,
        }
        return cls.from_layout(
            kind="ShapelyPolygon",
            layout_params=layout_params,
            name=name,
        )

    @classmethod
    def from_grid_mask(
        cls,
        active_mask: NDArray[np.bool_],
        grid_edges: tuple[NDArray[np.float64], ...],
        *,
        name: str = "",
        connect_diagonal_neighbors: bool = True,
    ) -> Environment:
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
            [0, 10, 20, 30] define three bins: [0-10], [10-20], [20-30].
        name : str, optional
            A name for the created environment. Defaults to "".
        connect_diagonal_neighbors : bool, optional
            Whether to connect diagonally adjacent active grid cells.
            Defaults to True.

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
        layout_params = {
            "active_mask": active_mask,
            "grid_edges": grid_edges,
            "connect_diagonal_neighbors": connect_diagonal_neighbors,
        }

        return cls.from_layout(
            kind="MaskedGrid",
            layout_params=layout_params,
            name=name,
        )

    @classmethod
    def from_pixel_mask(
        cls,
        image_mask: NDArray[np.bool_],
        pixel_size: float | tuple[float, float],
        *,
        connect_diagonal_neighbors: bool = True,
        name: str = "",
    ) -> Environment:
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
        layout_params = {
            "image_mask": image_mask,
            "pixel_size": pixel_size,
            "connect_diagonal_neighbors": connect_diagonal_neighbors,
        }

        return cls.from_layout(kind="ImageMask", layout_params=layout_params, name=name)

    @classmethod
    def from_layout(
        cls,
        kind: LayoutType | str,
        layout_params: dict[str, Any],
        *,
        name: str = "",
        regions: Regions | None = None,
    ) -> Environment:
        """Create an Environment with a specified layout type and its build parameters.

        Parameters
        ----------
        kind : LayoutType | str
            The layout engine type to use. Can be a LayoutType enum member (recommended
            for IDE autocomplete) or a case-insensitive string name
            (e.g., "RegularGrid", "Hexagonal").
        layout_params : Dict[str, Any]
            A dictionary of parameters that will be passed to the `build`
            method of the chosen `LayoutEngine`.
        name : str, optional
            A name for the created environment. Defaults to "".
        regions : Optional[Regions], optional
            A Regions instance to manage symbolic spatial regions within the environment.

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
        layout_instance = create_layout(kind=kind, **layout_params)
        # Note: This call will work when cls is Environment (via inheritance).
        # Mypy can't infer this for mixins, so we use cast() to help the type checker.
        env_cls = cast("type[Environment]", cls)
        return env_cls(name, layout_instance, kind, layout_params, regions=regions)

    @classmethod
    def from_nwb(
        cls,
        nwbfile: Any,
        *,
        scratch_name: str | None = None,
        bin_size: float | None = None,
        **kwargs: Any,
    ) -> Environment:
        """Create Environment from NWB file.

        This classmethod provides two modes of operation:

        1. **Load from scratch**: If `scratch_name` is provided, loads a previously
           stored Environment from the NWB file's scratch space. Use this when
           the Environment was saved using `env.to_nwb()`.

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
        if scratch_name is None and bin_size is None:
            raise ValueError(
                "Either scratch_name or bin_size must be provided. "
                "Use scratch_name to load a stored environment, or bin_size "
                "to create a new environment from Position data."
            )

        # Lazy import to keep pynwb optional
        try:
            from neurospatial.io.nwb import environment_from_position, read_environment
        except ImportError as e:
            raise ImportError(
                "pynwb is required for NWB integration. Install with: pip install pynwb"
            ) from e

        # Load from scratch if scratch_name is provided (takes precedence)
        if scratch_name is not None:
            return cast("Environment", read_environment(nwbfile, name=scratch_name))

        # Create from position data
        return cast(
            "Environment",
            environment_from_position(nwbfile, bin_size=bin_size, **kwargs),
        )
