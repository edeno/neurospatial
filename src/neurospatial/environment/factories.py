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

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from neurospatial.layout.factories import create_layout
from neurospatial.regions import Regions

if TYPE_CHECKING:
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


class EnvironmentFactories:
    """Factory methods mixin for creating Environment instances.

    This mixin provides classmethods for constructing Environment objects
    from different input sources and representations. All methods are classmethods
    that return fully initialized Environment instances.

    Methods
    -------
    from_samples(data_samples, bin_size, ...)
        Create environment by discretizing sample data into bins.
    from_graph(graph, edge_order, edge_spacing, bin_size, ...)
        Create 1D linearized track environment from graph structure.
    from_polygon(polygon, bin_size, ...)
        Create 2D grid environment masked by Shapely polygon.
    from_mask(active_mask, grid_edges, ...)
        Create environment from pre-defined boolean mask and grid edges.
    from_image(image_mask, bin_size, ...)
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

    >>> import networkx as nx
    >>> graph = nx.path_graph(10)
    >>> for i, node in enumerate(graph.nodes()):
    ...     graph.nodes[node]["pos"] = (i * 10.0, 0.0)
    >>> env = Environment.from_graph(
    ...     graph=graph, edge_order=list(graph.edges()), edge_spacing=0.0, bin_size=2.0
    ... )

    """

    @classmethod
    def from_samples(
        cls,
        data_samples: NDArray[np.float64],
        bin_size: float | Sequence[float],
        name: str = "",
        layout_kind: str = "RegularGrid",
        infer_active_bins: bool = True,
        bin_count_threshold: int = 0,
        dilate: bool = False,
        fill_holes: bool = False,
        close_gaps: bool = False,
        add_boundary_bins: bool = False,
        connect_diagonal_neighbors: bool = True,
        **layout_specific_kwargs: Any,
    ) -> Environment:
        """Create an Environment by binning (discretizing) `data_samples` into a layout grid.

        Parameters
        ----------
        data_samples : array, shape (n_samples, n_dims)
            Coordinates of sample points used to infer which bins are "active."
        bin_size : float or sequence of floats
            Size of each bin in the same units as `data_samples` coordinates.
            For RegularGrid: length of each square bin side (or per-dimension if sequence).
            For Hexagonal: hexagon width (flat-to-flat distance across hexagon).
            If your data is in centimeters, bin_size=5.0 creates 5cm bins.
        name : str, default ""
            Optional name for the resulting Environment.
        layout_kind : str, default "RegularGrid"
            Either "RegularGrid" or "Hexagonal" (case-insensitive). Determines
            bin shape. For "Hexagonal", `bin_size` is interpreted as `hexagon_width`.
        infer_active_bins : bool, default True
            If True, only bins containing ≥ `bin_count_threshold` samples are "active."
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
        connect_diagonal_neighbors : bool, default True
            If True, connect grid bins diagonally when building connectivity.

        Returns
        -------
        env : Environment
            A newly created Environment, fitted to the discretized samples.

        Raises
        ------
        ValueError
            If `data_samples` is not 2D or contains invalid coordinates.
        NotImplementedError
            If `layout_kind` is neither "RegularGrid" nor "Hexagonal".

        See Also
        --------
        from_polygon : Create environment with polygon-defined boundary.
        from_mask : Create environment from pre-defined boolean mask.
        from_image : Create environment from binary image mask.
        from_graph : Create 1D linearized track environment.
        from_layout : Create environment with custom LayoutEngine.

        Examples
        --------
        Create a simple 2D environment from position data:

        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> # Simulate animal position data in a 100x100 cm arena
        >>> np.random.seed(42)  # For reproducible examples
        >>> positions = np.random.rand(1000, 2) * 100  # cm
        >>> # Create environment with 5cm x 5cm bins
        >>> env = Environment.from_samples(
        ...     data_samples=positions,
        ...     bin_size=5.0,
        ...     name="arena",  # bin_size in cm
        ... )
        >>> env.n_dims
        2
        >>> env.n_bins > 0
        True

        Create environment with morphological operations to clean up the active region:

        >>> env = Environment.from_samples(
        ...     data_samples=positions,
        ...     bin_size=5.0,  # 5cm bins
        ...     bin_count_threshold=5,  # Require 5 samples per bin (lowered from 10)
        ...     dilate=True,  # Expand active region
        ...     fill_holes=True,  # Fill interior holes
        ... )

        Create a hexagonal grid environment:

        >>> env = Environment.from_samples(
        ...     data_samples=positions,
        ...     layout_kind="Hexagonal",
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

        3. **Mismatched units**: Ensure bin_size and data_samples use the same
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
        # Convert and validate data_samples array with helpful error messages
        try:
            data_samples = np.asarray(data_samples, dtype=float)
        except (TypeError, ValueError) as e:
            actual_type = type(data_samples).__name__
            raise TypeError(
                f"data_samples must be a numeric array-like object (e.g., numpy array, "
                f"list of lists, pandas DataFrame). Got {actual_type}: {data_samples!r}"
            ) from e

        if data_samples.ndim != 2:
            raise ValueError(
                f"data_samples must be a 2D array of shape (n_points, n_dims), "
                f"got shape {data_samples.shape}.",
            )

        # Validate bin_size early to provide helpful error messages
        if not isinstance(bin_size, (int, float, list, tuple, np.ndarray)):
            actual_type = type(bin_size).__name__
            raise TypeError(
                f"bin_size must be a numeric value or sequence of numeric values. "
                f"Got {actual_type}: {bin_size!r}"
            )

        # Standardize layout_kind to lowercase for comparison
        kind_lower = layout_kind.lower()
        if kind_lower not in ("regulargrid", "hexagonal"):
            raise NotImplementedError(
                f"Layout kind '{layout_kind}' is not supported. "
                "Use 'RegularGrid' or 'Hexagonal'.",
            )

        # Build the dict of layout parameters
        # Common parameters for all layouts
        common_params = {
            "data_samples": data_samples,
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

        if kind_lower not in specific_params:
            raise NotImplementedError(
                f"Layout kind '{layout_kind}' is not supported. "
                "Use 'RegularGrid' or 'Hexagonal'.",
            )

        # Build final params dict
        layout_params = {
            **common_params,
            **specific_params[kind_lower],
            **layout_specific_kwargs,
        }

        return cls.from_layout(kind=layout_kind, layout_params=layout_params, name=name)

    @classmethod
    def from_graph(
        cls,
        graph: nx.Graph,
        edge_order: list[tuple[Any, Any]],
        edge_spacing: float | Sequence[float],
        bin_size: float,
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
        from_mask : Create environment from pre-defined boolean mask.
        from_image : Create environment from binary image mask.

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
    def from_mask(
        cls,
        active_mask: NDArray[np.bool_],
        grid_edges: tuple[NDArray[np.float64], ...],
        name: str = "",
        connect_diagonal_neighbors: bool = True,
    ) -> Environment:
        """Create an Environment from a pre-defined N-D boolean mask and grid edges.

        This factory method allows for precise specification of active bins in
        an N-dimensional grid.

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
        from_image : Create environment from binary image mask.

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
        >>> env = Environment.from_mask(
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
    def from_image(
        cls,
        image_mask: NDArray[np.bool_],
        bin_size: float | tuple[float, float],
        connect_diagonal_neighbors: bool = True,
        name: str = "",
    ) -> Environment:
        """Create a 2D Environment from a binary image mask.

        Each `True` pixel in the `image_mask` becomes an active bin in the
        environment. The `bin_size` determines the spatial scale of these pixels.

        Parameters
        ----------
        image_mask : NDArray[np.bool_], shape (n_rows, n_cols)
            A 2D boolean array where `True` pixels define active bins.
        bin_size : float or tuple of (float, float)
            The spatial size of each pixel in physical units (e.g., cm, meters).
            If a float, pixels are square. If a tuple `(width, height)`, specifies
            pixel dimensions. For example, if your camera captures images where
            each pixel represents 0.5cm, use bin_size=0.5.
        connect_diagonal_neighbors : bool, optional
            Whether to connect diagonally adjacent active pixel-bins.
            Defaults to True.
        name : str, optional
            A name for the created environment. Defaults to "".

        Returns
        -------
        Environment
            A new Environment instance with an `ImageMaskLayout`.

        See Also
        --------
        from_mask : Create environment from pre-defined boolean mask.
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
        >>> env = Environment.from_image(
        ...     image_mask=mask,
        ...     bin_size=0.5,  # Each pixel = 0.5cm
        ...     name="arena_from_image",
        ... )
        >>> env.n_dims
        2

        """
        layout_params = {
            "image_mask": image_mask,
            "bin_size": bin_size,
            "connect_diagonal_neighbors": connect_diagonal_neighbors,
        }

        return cls.from_layout(kind="ImageMask", layout_params=layout_params, name=name)

    @classmethod
    def from_layout(
        cls,
        kind: str,
        layout_params: dict[str, Any],
        name: str = "",
        regions: Regions | None = None,
    ) -> Environment:
        """Create an Environment with a specified layout type and its build parameters.

        Parameters
        ----------
        kind : str
            The string identifier of the `LayoutEngine` to use
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
        from_mask : Create environment from pre-defined boolean mask.
        from_image : Create environment from binary image mask.
        from_graph : Create 1D linearized track environment.

        """
        layout_instance = create_layout(kind=kind, **layout_params)
        # Note: This call will work when cls is Environment (via inheritance).
        # Mypy can't infer this for mixins, so we use cast() to help the type checker.
        env_cls = cast("type[Environment]", cls)
        return env_cls(name, layout_instance, kind, layout_params, regions=regions)
