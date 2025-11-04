from __future__ import annotations

import logging
import pickle
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property, wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import scipy.sparse
    import shapely

import matplotlib.axes
import networkx as nx
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from neurospatial._logging import log_environment_created, log_graph_validation
from neurospatial.layout.base import LayoutEngine
from neurospatial.layout.factories import create_layout
from neurospatial.layout.helpers.utils import find_boundary_nodes
from neurospatial.layout.validation import (
    GraphValidationError,
    validate_connectivity_graph,
)
from neurospatial.regions import Regions

logger = logging.getLogger(__name__)

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


# --- Decorator ---
def check_fitted(method):
    """Decorator to ensure that an Environment method is called only after fitting.

    Parameters
    ----------
    method : callable
        Method to decorate.

    Returns
    -------
    callable
        Wrapped method that checks fitted status before execution.

    Raises
    ------
    RuntimeError
        If the method is called on an Environment instance that has not been
        fully initialized (i.e., `_is_fitted` is False).

    """

    @wraps(method)
    def _inner(self: Environment, *args, **kwargs):
        if not getattr(self, "_is_fitted", False):
            raise RuntimeError(
                f"{self.__class__.__name__}.{method.__name__}() "
                "requires the environment to be fully initialized. "
                "Ensure it was created with a factory method.\n\n"
                "Example (correct usage):\n"
                "    env = Environment.from_samples(data, bin_size=2.0)\n"
                "    result = env.bin_at(points)\n\n"
                "Avoid:\n"
                "    env = Environment()  # This will not work!",
            )
        return method(self, *args, **kwargs)

    return _inner


# --- Main Environment Class ---
@dataclass
class Environment:
    """Represents a discretized N-dimensional space with connectivity.

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

    4. **from_mask** - Create environment from pre-computed mask
       Use when you have already determined which bins should be active (e.g.,
       from external analysis) as an N-D boolean array. Requires explicit
       specification of grid edges.
       See `from_mask()`.

    5. **from_image** - Create environment from binary image
       Use when your environment boundary is defined by a binary image (e.g.,
       segmentation mask, overhead camera view). Each white pixel becomes a
       potential bin.
       See `from_image()`.

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
    connectivity : nx.Graph
        A NetworkX graph where nodes are integers from `0` to `n_active_bins - 1`,
        directly corresponding to the rows of `bin_centers`. Edges represent
        adjacency between bins. Populated by `_setup_from_layout`.
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
    _is_1d_env : bool
        Internal flag indicating if the environment's layout is primarily 1-dimensional.
        Set based on `layout.is_1d`.
    _is_fitted : bool
        Internal flag indicating if the environment has been fully initialized
        and its layout-dependent attributes are populated.
    _layout_type_used : Optional[str]
        The string identifier of the `LayoutEngine` type used to create this
        environment (e.g., "RegularGrid"). For introspection and serialization.
    _layout_params_used : Dict[str, Any]
        A dictionary of the parameters used to build the `LayoutEngine` instance.
        For introspection and serialization.

    """

    name: str
    layout: LayoutEngine

    # --- Attributes populated from the layout instance ---
    bin_centers: NDArray[np.float64] = field(init=False)
    connectivity: nx.Graph = field(init=False)
    dimension_ranges: Sequence[tuple[float, float]] | None = field(init=False)

    # Grid-specific context (populated if layout is grid-based)
    grid_edges: tuple[NDArray[np.float64], ...] | None = field(init=False)
    grid_shape: tuple[int, ...] | None = field(init=False)
    active_mask: NDArray[np.bool_] | None = field(init=False)

    # Region Management
    regions: Regions = field(init=False, repr=False)

    # Units and coordinate frames
    units: str | None = field(init=False, default=None)
    frame: str | None = field(init=False, default=None)

    # Internal state
    _is_1d_env: bool = field(init=False)
    _is_fitted: bool = field(init=False, default=False)

    # KD-tree cache for spatial queries (populated lazily by map_points_to_bins)
    _kdtree_cache: Any = field(init=False, default=None, repr=False)

    # Kernel cache for smoothing operations (keyed by (bandwidth, mode))
    _kernel_cache: dict[tuple[float, str], NDArray] = field(
        init=False, default_factory=dict, repr=False
    )

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
    ):
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

        """
        if layout is None:
            raise ValueError("layout parameter is required")

        self.name = name
        self.layout = layout

        self._layout_type_used = (
            layout_type_used
            if layout_type_used
            else getattr(layout, "_layout_type_tag", None)
        )
        self._layout_params_used = (
            layout_params_used
            if layout_params_used is not None
            else getattr(layout, "_build_params_used", {})
        )

        self._is_1d_env = self.layout.is_1d

        # Initialize attributes that will be populated by _setup_from_layout
        self.bin_centers = np.empty((0, 0))  # Placeholder
        self.connectivity = nx.Graph()
        self.dimension_ranges = None
        self.grid_edges = ()
        self.grid_shape = None
        self.active_mask = None
        self._is_fitted = False  # Will be set by _setup_from_layout
        if layout_type_used is not None:
            self._setup_from_layout()  # Populate attributes from the built layout
        if regions is not None:
            if not isinstance(regions, Regions):
                raise TypeError(
                    f"Expected 'regions' to be a Regions instance, got {type(regions)}.",
                )
            self.regions = regions
        else:
            # Initialize with an empty Regions instance if not provided
            self.regions = Regions()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.name == other
        return NotImplemented

    def __repr__(self: Environment) -> str:
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
        for reconstruction. For serialization, use the `save()` method instead.

        Examples
        --------
        >>> import numpy as np
        >>> data = np.random.rand(100, 2) * 10
        >>> env = Environment.from_samples(data, bin_size=2.0, name="MyEnv")
        >>> repr(env)  # doctest: +SKIP
        "Environment(name='MyEnv', 2D, 25 bins, RegularGrid)"

        """
        # Handle unfitted environments
        if not self._is_fitted:
            name_str = f"'{self.name}'" if self.name else "None"
            return f"Environment(name={name_str}, not fitted)"

        # Fitted environments: show name, dims, bins, layout
        # Truncate very long names
        name = self.name if self.name else ""
        if len(name) > 40:
            name = name[:37] + "..."
        name_str = f"'{name}'" if name else "None"

        # Get dimensionality
        try:
            dims_str = f"{self.n_dims}D"
        except (RuntimeError, AttributeError):
            dims_str = "?D"

        # Get bin count
        n_bins = self.bin_centers.shape[0] if hasattr(self, "bin_centers") else 0

        # Get layout type (remove 'Layout' suffix for brevity if present)
        layout_type = self._layout_type_used or "Unknown"
        if layout_type.endswith("Layout"):
            layout_type = layout_type[:-6]  # Remove 'Layout' suffix

        return f"Environment(name={name_str}, {dims_str}, {n_bins} bins, {layout_type})"

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

        # Build HTML table
        html_parts = []
        html_parts.append('<div style="margin: 10px;">')
        html_parts.append(
            '<table style="border-collapse: collapse; border: 1px solid #ddd; '
            'font-family: monospace; font-size: 12px;">'
        )

        # Header row
        html_parts.append(
            '<tr style="background-color: #f0f0f0; border-bottom: 2px solid #999;">'
        )
        html_parts.append(
            '<th colspan="2" style="padding: 8px; text-align: left; '
            'font-weight: bold; font-size: 14px;">'
        )
        html_parts.append(f"Environment: {name}")
        html_parts.append("</th></tr>")

        # Helper function to add rows
        def add_row(label: str, value: str, highlight: bool = False) -> None:
            bg_color = "#fffacd" if highlight else "#fff"
            html_parts.append(f'<tr style="background-color: {bg_color};">')
            html_parts.append(
                f'<td style="padding: 6px 12px; border-top: 1px solid #ddd; '
                f'font-weight: bold; color: #555;">{label}</td>'
            )
            html_parts.append(
                f'<td style="padding: 6px 12px; border-top: 1px solid #ddd; '
                f'color: #000;">{value}</td>'
            )
            html_parts.append("</tr>")

        # Check if fitted
        if not self._is_fitted:
            add_row("Status", "Not fitted", highlight=True)
            add_row("Layout Type", self._layout_type_used or "Unknown")
            html_parts.append("</table></div>")
            return "".join(html_parts)

        # Fitted environment - show full details
        add_row("Layout Type", self._layout_type_used or "Unknown")

        # Dimensions and bins
        try:
            n_dims = self.n_dims
            add_row("Dimensions", str(n_dims))
        except (RuntimeError, AttributeError):
            add_row("Dimensions", "Unknown")
            n_dims = None

        n_bins = self.bin_centers.shape[0] if hasattr(self, "bin_centers") else 0
        add_row("Number of Bins", str(n_bins))

        # Spatial extent
        if hasattr(self, "dimension_ranges") and self.dimension_ranges:
            extent_parts = []
            for dim_idx, (min_val, max_val) in enumerate(self.dimension_ranges):
                extent_parts.append(f"dim{dim_idx}: [{min_val:.2f}, {max_val:.2f}]")
            extent_str = "<br>".join(extent_parts)
            add_row("Spatial Extent", extent_str)

        # Regions
        n_regions = len(self.regions) if hasattr(self, "regions") else 0
        if n_regions > 0:
            add_row("Regions", f"{n_regions} defined")
        else:
            add_row("Regions", "None")

        # 1D-specific info
        if n_dims == 1 and hasattr(self, "is_1d") and self.is_1d:
            add_row("Linearization", "Available (1D environment)")

        html_parts.append("</table></div>")
        return "".join(html_parts)

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
        if hasattr(self, "is_1d") and self.is_1d:
            lines.append("Linearization: Available (1D environment)")
            lines.append("")

        return "\n".join(lines)

    def _setup_from_layout(self) -> None:
        """Populate Environment attributes from its (built) LayoutEngine.

        This internal method is called after the `LayoutEngine` is associated
        with the Environment. It copies essential geometric and connectivity
        information from the layout to the Environment's attributes.
        It also applies fallbacks for certain grid-specific attributes if the
        layout is point-based to ensure consistency.

        Raises
        ------
        ValueError
            If the connectivity graph from the layout engine is invalid
            (missing required node/edge attributes, wrong dimensions, etc.)

        """
        self.bin_centers = self.layout.bin_centers
        self.connectivity = getattr(self.layout, "connectivity", nx.Graph())
        self.dimension_ranges = self.layout.dimension_ranges

        # Validate connectivity graph has required metadata
        # This catches layout engine bugs early with clear error messages
        # Note: Calculate n_dims directly here since self.n_dims has @check_fitted
        n_dims = self.bin_centers.shape[1] if self.bin_centers is not None else 0
        try:
            n_nodes = len(self.connectivity.nodes)
            n_edges = len(self.connectivity.edges)
            log_graph_validation(n_nodes=n_nodes, n_edges=n_edges, n_dims=n_dims)
            validate_connectivity_graph(
                self.connectivity,
                n_dims=n_dims,
                check_node_attrs=True,
                check_edge_attrs=True,
            )
        except GraphValidationError as e:
            raise ValueError(
                f"Invalid connectivity graph from layout engine "
                f"'{self.layout._layout_type_tag}': {e}\n\n"
                f"This is a bug in the layout engine. Please report this issue.\n"
                f"See CLAUDE.md section 'Graph Metadata Requirements' for details."
            ) from e

        # Grid-specific attributes
        self.grid_edges = getattr(self.layout, "grid_edges", ())
        self.grid_shape = getattr(self.layout, "grid_shape", None)
        self.active_mask = getattr(self.layout, "active_mask", None)

        # If it's not a grid layout, grid_shape might be (n_active_bins,),
        # and active_mask might be 1D all True. This is fine.
        # Ensure they are at least None if not applicable from layout
        if self.grid_shape is None and self.bin_centers is not None:
            # Fallback for point-based
            self.grid_shape = (self.bin_centers.shape[0],)
        if self.active_mask is None and self.bin_centers is not None:
            # Fallback for point-based
            self.active_mask = np.ones(self.bin_centers.shape[0], dtype=bool)

        self._is_fitted = True

        # Log environment creation
        n_bins = self.bin_centers.shape[0] if self.bin_centers is not None else 0
        log_environment_created(
            env_type=self.layout._layout_type_tag,
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

        """
        return {
            data["source_grid_flat_index"]: node_id
            for node_id, data in self.connectivity.nodes(data=True)
            if "source_grid_flat_index" in data
        }

    # --- Factory Methods ---
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
            If True, only bins containing ≥ `bin_count_threshold` samples are “active.”
        bin_count_threshold : int, default 0
            Minimum number of data points required for a bin to be considered “active.”
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
        layout_params: dict[str, Any] = {
            "data_samples": data_samples,
            "infer_active_bins": infer_active_bins,
            "bin_count_threshold": bin_count_threshold,
            **layout_specific_kwargs,
        }

        if kind_lower == "regulargrid":
            layout_params.update(
                {
                    "bin_size": bin_size,
                    "add_boundary_bins": add_boundary_bins,
                    "dilate": dilate,
                    "fill_holes": fill_holes,
                    "close_gaps": close_gaps,
                    "connect_diagonal_neighbors": connect_diagonal_neighbors,
                },
            )
        elif kind_lower == "hexagonal":
            layout_params.update(
                {
                    "hexagon_width": bin_size,
                },
            )
        else:
            raise NotImplementedError(
                f"Layout kind '{layout_kind}' is not supported. "
                "Use 'RegularGrid' or 'Hexagonal'.",
            )

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
        return cls(name, layout_instance, kind, layout_params, regions=regions)

    @property
    def is_1d(self) -> bool:
        """Indicate if the environment's layout is primarily 1-dimensional.

        Returns
        -------
        bool
            True if the underlying `LayoutEngine` (`self.layout`) reports
            itself as 1-dimensional (e.g., `GraphLayout`), False otherwise.
            This is determined by `self.layout.is_1d`.

        """
        return self._is_1d_env

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
        return self._layout_params_used

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
    def bin_at(self, points_nd: NDArray[np.float64]) -> NDArray[np.int_]:
        """Map N-dimensional continuous points to discrete active bin indices.

        This method delegates to the `point_to_bin_index` method of the
        underlying `LayoutEngine`.

        Parameters
        ----------
        points_nd : NDArray[np.float64], shape (n_points, n_dims)
            An array of N-dimensional points to map.

        Returns
        -------
        NDArray[np.int_], shape (n_points,)
            An array of active bin indices (0 to `n_active_bins - 1`).
            A value of -1 indicates that the corresponding point did not map
            to any active bin (e.g., it's outside the environment).

        Raises
        ------
        RuntimeError
            If called before the environment is fitted.

        """
        return self.layout.point_to_bin_index(points_nd)

    @check_fitted
    def contains(self, points_nd: NDArray[np.float64]) -> NDArray[np.bool_]:
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

        Notes
        -----
        This method is optimized to avoid redundant KDTree queries by reusing
        the bin index computation from `bin_at()` and checking for the -1 sentinel.

        """
        # Optimized: compute indices once and check for -1 sentinel
        # This avoids redundant KDTree queries compared to calling bin_at() separately
        indices = self.layout.point_to_bin_index(points_nd)
        return np.asarray(indices != -1, dtype=np.bool_)

    @check_fitted
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
        centers : array, shape (len(bin_indices), n_dims) if multiple indices,
                        (n_dims,) if single index
            The center coordinate(s) of the requested bin(s).

        Raises
        ------
        RuntimeError
            If the environment is not fitted.
        IndexError
            If any bin index is out of range.

        """
        return np.asarray(
            self.bin_centers[np.asarray(bin_indices, dtype=int)], dtype=np.float64
        )

    @check_fitted
    def neighbors(self, bin_index: int) -> list[int]:
        """Find indices of neighboring active bins for a given active bin index.

        This method delegates to the `neighbors` method of the
        underlying `LayoutEngine`, which typically uses the `connectivity`.

        Parameters
        ----------
        bin_index : int
            The index (0 to `n_active_bins - 1`) of the active bin for which
            to find neighbors.

        Returns
        -------
        List[int]
            A list of active bin indices that are neighbors to `bin_index`.

        Raises
        ------
        RuntimeError
            If called before the environment is fitted.

        """
        return list(self.connectivity.neighbors(bin_index))

    @cached_property
    @check_fitted
    def bin_sizes(self) -> NDArray[np.float64]:
        """Calculate the area (for 2D) or volume (for 3D+) of each active bin.

        This represent the actual size of each bin in the environment, as
        opposed to the requested `bin_size` which is the nominal size used
        during layout creation.

        For 1D environments, this typically returns the length of each bin.
        This method delegates to the `bin_sizes` method of the
        underlying `LayoutEngine`.

        Returns
        -------
        NDArray[np.float64], shape (n_active_bins,)
            An array containing the area/volume/length of each active bin.

        Raises
        ------
        RuntimeError
            If called before the environment is fitted.

        """
        return self.layout.bin_sizes()

    @check_fitted
    def compute_kernel(
        self,
        bandwidth: float,
        *,
        mode: Literal["transition", "density"] = "density",
        cache: bool = True,
    ) -> NDArray[np.float64]:
        """Compute diffusion kernel for smoothing operations.

        Convenience wrapper for kernels.compute_diffusion_kernels() that
        automatically uses this environment's connectivity graph and bin sizes.

        Parameters
        ----------
        bandwidth : float
            Smoothing bandwidth in physical units (σ in the Gaussian kernel).
            Controls the scale of diffusion.
        mode : {'transition', 'density'}, default='density'
            Normalization mode:

            - 'transition': Each column sums to 1 (discrete probability).
            - 'density': Each column integrates to 1 over bin volumes
              (continuous density).
        cache : bool, default=True
            If True, cache the computed kernel for reuse. Subsequent calls
            with the same (bandwidth, mode) will return the cached result.

        Returns
        -------
        kernel : NDArray[np.float64], shape (n_bins, n_bins)
            Diffusion kernel matrix where kernel[:, j] represents the smoothed
            distribution resulting from a unit mass at bin j.

        Raises
        ------
        RuntimeError
            If called before the environment is fitted.
        ValueError
            If bandwidth is not positive.

        See Also
        --------
        neurospatial.kernels.compute_diffusion_kernels :
            Lower-level function with more control.

        Notes
        -----
        The kernel is computed via matrix exponential of the graph Laplacian:

        .. math::
            K = \\exp(-t L)

        where :math:`t = \\sigma^2 / 2` and :math:`L` is the graph Laplacian.

        For mode='density', the Laplacian is volume-corrected to properly
        handle bins of varying sizes.

        Performance warning: Kernel computation has O(n³) complexity where
        n is the number of bins. For large environments (>1000 bins),
        computation may be slow. Consider caching or using smaller bandwidths.

        Examples
        --------
        >>> env = Environment.from_samples(data, bin_size=2.0)
        >>> # Compute kernel for smoothing
        >>> kernel = env.compute_kernel(bandwidth=5.0, mode="density")
        >>> # Apply to field
        >>> smoothed_field = kernel @ field

        """
        from neurospatial.kernels import compute_diffusion_kernels

        # Initialize cache if it doesn't exist
        # (for backward compatibility with environments deserialized from older versions)
        if not hasattr(self, "_kernel_cache"):
            self._kernel_cache = {}

        # Check cache first if enabled
        cache_key = (bandwidth, mode)
        if cache and cache_key in self._kernel_cache:
            return self._kernel_cache[cache_key]

        # Compute kernel
        kernel = compute_diffusion_kernels(
            graph=self.connectivity,
            bandwidth_sigma=bandwidth,
            bin_sizes=self.bin_sizes if mode == "density" else None,
            mode=mode,
        )

        # Store in cache if enabled
        if cache:
            self._kernel_cache[cache_key] = kernel

        return kernel

    @check_fitted
    def smooth(
        self,
        field: NDArray[np.float64],
        bandwidth: float,
        *,
        mode: Literal["transition", "density"] = "density",
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
        mode : {'transition', 'density'}, default='density'
            Smoothing mode that controls normalization:

            - 'transition': Mass-conserving smoothing. Total sum is preserved:
              smoothed.sum() = field.sum(). Use for count data (occupancy,
              spike counts).
            - 'density': Volume-corrected smoothing. Accounts for varying bin
              sizes. Use for continuous density fields (rate maps,
              probability distributions).

        Returns
        -------
        smoothed : NDArray[np.float64], shape (n_bins,)
            Smoothed field values.

        Raises
        ------
        RuntimeError
            If called before the environment is fitted.
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

        where :math:`K` is the diffusion kernel computed via matrix exponential
        of the graph Laplacian.

        For mode='transition', mass is conserved:

        .. math::
            \\sum_i \\text{smoothed}_i = \\sum_i \\text{field}_i

        For mode='density', the kernel accounts for bin volumes, making it
        appropriate for continuous density fields.

        The kernel is cached automatically, so repeated smoothing operations
        with the same bandwidth and mode are efficient.

        Edge preservation: Smoothing respects graph connectivity. Mass does
        not leak between disconnected components.

        Examples
        --------
        >>> # Smooth spike counts (mass-conserving)
        >>> smoothed_counts = env.smooth(spike_counts, bandwidth=5.0, mode="transition")
        >>> # Total spikes preserved
        >>> assert np.isclose(smoothed_counts.sum(), spike_counts.sum())

        >>> # Smooth a rate map (volume-corrected)
        >>> smoothed_rates = env.smooth(rate_map, bandwidth=3.0, mode="density")

        >>> # Smooth a probability distribution
        >>> smoothed_prob = env.smooth(posterior, bandwidth=2.0, mode="transition")
        """
        # Input validation
        field = np.asarray(field, dtype=np.float64)

        # Check field dimensionality
        if field.ndim != 1:
            raise ValueError(
                f"Field must be 1-D array (got {field.ndim}-D array). "
                f"Expected shape (n_bins,) = ({self.n_bins},), got shape {field.shape}."
            )

        # Check field shape matches n_bins
        if field.shape[0] != self.n_bins:
            raise ValueError(
                f"Field shape {field.shape} must match n_bins={self.n_bins}. "
                f"Expected shape (n_bins,) = ({self.n_bins},), got ({field.shape[0]},)."
            )

        # Check for NaN/Inf values
        if np.any(np.isnan(field)):
            raise ValueError(
                "Field contains NaN values. "
                f"Found {np.sum(np.isnan(field))} NaN values out of {len(field)} bins. "
                "NaN values are not supported in smoothing operations."
            )

        if np.any(np.isinf(field)):
            raise ValueError(
                "Field contains infinite values. "
                f"Found {np.sum(np.isinf(field))} infinite values out of {len(field)} bins. "
                "Infinite values are not supported in smoothing operations."
            )

        # Validate bandwidth
        if bandwidth <= 0:
            raise ValueError(
                f"bandwidth must be positive (got {bandwidth}). "
                "Bandwidth controls the spatial scale of smoothing."
            )

        # Validate mode
        valid_modes = {"transition", "density"}
        if mode not in valid_modes:
            raise ValueError(
                f"mode must be one of {valid_modes} (got '{mode}'). "
                "Use 'transition' for mass-conserving smoothing or 'density' "
                "for volume-corrected smoothing."
            )

        # Compute kernel (uses cache automatically)
        kernel = self.compute_kernel(bandwidth, mode=mode, cache=True)

        # Apply smoothing
        smoothed: NDArray[np.float64] = kernel @ field

        return smoothed

    @check_fitted
    def rebin(
        self,
        factor: int | tuple[int, ...],
    ) -> Environment:
        """Coarsen regular grid by integer factor (geometry-only operation).

        Creates a new environment with coarser spatial resolution by reducing
        the number of bins. This method only modifies the grid geometry and
        connectivity; it does not aggregate any field values.

        Only supported for RegularGridLayout environments.

        Parameters
        ----------
        factor : int or tuple of int
            Coarsening factor per dimension. If int, applied uniformly to all
            dimensions. If tuple, must match the number of dimensions.
            Each factor must be a positive integer.

        Returns
        -------
        coarse_env : Environment
            New environment with reduced resolution. The new grid shape is
            ``original_shape // factor`` in each dimension. All bins in the
            coarsened grid are marked as active.

        Raises
        ------
        RuntimeError
            If called before the environment is fitted.
        NotImplementedError
            If environment layout is not RegularGridLayout.
        ValueError
            If factor is not positive or if factor is too large for grid shape.

        See Also
        --------
        smooth : Apply diffusion kernel smoothing to fields.
        subset : Extract spatial subset of environment.
        map_points_to_bins : Map original bin centers to coarsened bins.

        Notes
        -----
        **Geometry only**: This method only coarsens the grid structure. To
        aggregate field values (occupancy, spike counts, etc.) from the original
        grid to the coarsened grid, map the original bin centers and aggregate:

            >>> from neurospatial import map_points_to_bins
            >>> coarse = env.rebin(factor=2)
            >>> coarse_indices = map_points_to_bins(env.bin_centers, coarse)
            >>> coarse_field = np.bincount(
            ...     coarse_indices, weights=field, minlength=coarse.n_bins
            ... )

        **Grid-only operation**: This method only works for environments with
        RegularGridLayout. Other layout types will raise NotImplementedError.

        **Non-divisible dimensions**: If the grid shape is not evenly divisible
        by the factor in any dimension, the grid is truncated to the largest
        multiple of the factor. A warning is issued in this case.

        **Connectivity**: The connectivity graph is rebuilt for the coarsened
        grid with the same connectivity pattern as the original (e.g., if
        original had diagonal connections, coarsened grid will too).

        **Bin centers**: New bin centers are computed from the coarsened grid
        edges as midpoints between edge positions.

        **Active bins**: All bins in the coarsened environment are marked as
        active, even if the original grid had inactive regions.

        **Metadata preservation**: The units and frame attributes are copied
        from the original environment to the coarsened environment.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> # Create 10x10 grid
        >>> data = np.random.rand(1000, 2) * 100
        >>> env = Environment.from_samples(data, bin_size=10.0)
        >>> env.layout.grid_shape
        (10, 10)
        >>>
        >>> # Coarsen by factor 2 → 5x5 grid
        >>> coarse = env.rebin(factor=2)
        >>> coarse.layout.grid_shape
        (5, 5)
        >>>
        >>> # Anisotropic coarsening with tuple
        >>> coarse_aniso = env.rebin(factor=(2, 5))
        >>> coarse_aniso.layout.grid_shape
        (5, 2)
        >>>
        >>> # Aggregate a field to the coarsened grid
        >>> from neurospatial import map_points_to_bins
        >>> occupancy = np.random.rand(env.n_bins) * 100
        >>> coarse_indices = map_points_to_bins(env.bin_centers, coarse)
        >>> coarse_occupancy = np.bincount(
        ...     coarse_indices, weights=occupancy, minlength=coarse.n_bins
        ... )
        >>> # Total time is preserved
        >>> np.isclose(occupancy.sum(), coarse_occupancy.sum())
        True
        """
        from neurospatial.layout.engines.regular_grid import RegularGridLayout

        # --- Input validation ---

        # Check layout type
        if not isinstance(self.layout, RegularGridLayout):  # type: ignore[unreachable]
            raise NotImplementedError(
                "rebin() is only supported for RegularGridLayout. "
                f"Current layout type: {self.layout._layout_type_tag}. "
                "For other layout types, consider using smooth() for field smoothing "
                "or subset() for spatial cropping."
            )

        # Parse factor
        grid_shape = self.layout.grid_shape  # type: ignore[unreachable]
        n_dims = len(grid_shape)

        factor_tuple = (factor,) * n_dims if isinstance(factor, int) else tuple(factor)

        # Validate factor dimensions
        if len(factor_tuple) != n_dims:
            raise ValueError(
                f"factor has {len(factor_tuple)} elements but environment has "
                f"{n_dims} dimensions. factor must be int or tuple matching "
                "environment dimensionality."
            )

        # Validate factor values
        for i, f in enumerate(factor_tuple):
            if not isinstance(f, (int, np.integer)):
                raise ValueError(
                    f"factor[{i}] = {f} must be an integer, got {type(f).__name__}"
                )
            if f <= 0:
                raise ValueError(
                    f"factor[{i}] = {f} must be positive. "
                    "Coarsening factor must be at least 1."
                )
            if f > grid_shape[i]:
                raise ValueError(
                    f"factor[{i}] = {f} is too large for grid shape {grid_shape}. "
                    f"Dimension {i} has only {grid_shape[i]} bins."
                )

        # Check for non-divisible dimensions
        truncated_shape = tuple(
            s // f * f for s, f in zip(grid_shape, factor_tuple, strict=True)
        )
        if truncated_shape != grid_shape:
            warnings.warn(
                f"Grid shape {grid_shape} is not evenly divisible by factor "
                f"{factor_tuple}. Grid will be truncated to {truncated_shape} "
                f"before coarsening.",
                UserWarning,
                stacklevel=2,
            )

        # --- Compute new grid parameters ---

        # Truncate grid edges if needed
        grid_edges = self.layout.grid_edges
        truncated_edges = []
        for edges, trunc_size in zip(grid_edges, truncated_shape, strict=True):
            # Keep edges up to truncated_size + 1 (edges define bins)
            truncated_edges.append(edges[: trunc_size + 1])

        # Compute new coarsened edges
        coarse_edges = tuple(
            edges[::f] for edges, f in zip(truncated_edges, factor_tuple, strict=True)
        )

        # New grid shape
        coarse_shape = tuple(len(edges) - 1 for edges in coarse_edges)

        # --- Compute new bin centers from coarsened edges ---

        # For each coarse bin, compute center from the coarse grid edges
        # This avoids issues with active/inactive bins from the original grid

        # Create bin centers from coarse edges using meshgrid
        coarse_grid_centers = []
        for edges in coarse_edges:
            # Bin centers are midpoints between edges
            centers = (edges[:-1] + edges[1:]) / 2
            coarse_grid_centers.append(centers)

        # Create meshgrid of bin centers
        if n_dims == 1:
            center_coords = [coarse_grid_centers[0]]
        else:
            center_grids = np.meshgrid(*coarse_grid_centers, indexing="ij")
            center_coords = [grid.ravel() for grid in center_grids]

        # Stack into (n_bins, n_dims)
        coarse_bin_centers = np.column_stack(center_coords)

        # --- Build new connectivity graph ---

        # Check if original had diagonal connections
        # Sample: check degree of a center node (not on boundary)
        center_node = grid_shape[0] // 2
        if n_dims == 2:
            center_flat_idx = center_node * grid_shape[1] + grid_shape[1] // 2
        else:
            # For higher dims, just check if any node has more than 2*n_dims neighbors
            center_flat_idx = 0

        # Get degree
        if center_flat_idx in self.connectivity:
            degree = self.connectivity.degree(center_flat_idx)
            # 2D: 4-conn has degree 4, 8-conn has degree 8
            # 3D: 6-conn has degree 6, 26-conn has degree 26
            # Heuristic: if degree > 2*n_dims, assume diagonal connections
            connect_diagonal = degree > 2 * n_dims
        else:
            # Default to True (common case)
            connect_diagonal = True

        # Create new layout
        from neurospatial.layout.helpers.regular_grid import (
            _create_regular_grid_connectivity_graph,
        )

        # Build connectivity for the coarsened grid (all active)
        active_mask_coarse = np.ones(coarse_shape, dtype=bool)

        coarse_connectivity = _create_regular_grid_connectivity_graph(
            full_grid_bin_centers=coarse_bin_centers,
            active_mask_nd=active_mask_coarse,
            grid_shape=coarse_shape,
            connect_diagonal=connect_diagonal,
        )

        # --- Create new Environment ---

        # Create new layout instance
        new_layout = RegularGridLayout()
        new_layout.bin_centers = coarse_bin_centers
        new_layout.connectivity = coarse_connectivity
        new_layout.dimension_ranges = tuple(
            (edges[0], edges[-1]) for edges in coarse_edges
        )
        new_layout.grid_edges = coarse_edges
        new_layout.grid_shape = coarse_shape
        new_layout.active_mask = active_mask_coarse
        new_layout._layout_type_tag = "RegularGrid"
        new_layout._build_params_used = {
            "bin_size": tuple(
                (edges[-1] - edges[0]) / (len(edges) - 1) for edges in coarse_edges
            ),
            "dimension_ranges": new_layout.dimension_ranges,
            "rebinned_from": f"factor={factor_tuple}",
        }

        # Create new environment
        coarse_env = Environment(
            layout=new_layout,
            name=f"{self.name}_rebinned" if self.name else "",
            regions=Regions(),  # Start with empty regions
        )
        coarse_env._layout_type_used = "RegularGrid"
        coarse_env._layout_params_used = new_layout._build_params_used
        coarse_env._setup_from_layout()

        # Preserve metadata
        if hasattr(self, "units") and self.units is not None:
            coarse_env.units = self.units
        if hasattr(self, "frame") and self.frame is not None:
            coarse_env.frame = self.frame

        return coarse_env

    @check_fitted
    def subset(
        self,
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
        environment if needed.

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
        and edge attributes ('distance', 'vector', 'edge_id', 'angle_2d') are
        preserved from the original graph.

        See Also
        --------
        rebin : Coarsen grid resolution (grid-only).
        components : Find connected components.

        Examples
        --------
        >>> # Extract bins inside 'goal' region
        >>> goal_env = env.subset(region_names=["goal"])
        >>>
        >>> # Crop to polygon
        >>> from shapely.geometry import box
        >>> cropped = env.subset(polygon=box(0, 0, 50, 50))
        >>>
        >>> # Select bins by boolean mask
        >>> mask = env.bin_centers[:, 0] < 50  # Left half
        >>> left_env = env.subset(bins=mask)
        >>>
        >>> # Inverted selection (everything except region)
        >>> outside = env.subset(region_names=["obstacle"], invert=True)
        """
        # --- Input Validation ---

        # Exactly one selection parameter must be provided
        n_params = sum(
            [bins is not None, region_names is not None, polygon is not None]
        )
        if n_params == 0:
            raise ValueError(
                "Exactly one of {bins, region_names, polygon} must be provided."
            )
        if n_params > 1:
            raise ValueError(
                "Exactly one of {bins, region_names, polygon} must be provided. "
                f"Got {n_params} parameters."
            )

        # --- Build Selection Mask ---

        if bins is not None:
            # Validate bins parameter
            bins = np.asarray(bins)

            # Check dtype
            if bins.dtype != bool:
                raise ValueError(
                    f"bins must be boolean array (dtype=bool), got dtype={bins.dtype}"
                )

            # Check shape
            if bins.shape != (self.n_bins,):
                raise ValueError(
                    f"bins must have shape (n_bins,) = ({self.n_bins},), "
                    f"got shape {bins.shape}"
                )

            mask = bins

        elif region_names is not None:
            # Validate region_names parameter
            if not isinstance(region_names, (list, tuple)):
                raise ValueError(
                    f"region_names must be a list or tuple, got {type(region_names)}"
                )

            if len(region_names) == 0:
                raise ValueError("region_names cannot be empty")

            # Check all regions exist
            for name in region_names:
                if name not in self.regions:
                    available = list(self.regions.keys())
                    raise ValueError(
                        f"Region '{name}' not found in environment. "
                        f"Available regions: {available}"
                    )

            # Build mask from regions
            mask = np.zeros(self.n_bins, dtype=bool)
            for name in region_names:
                region = self.regions[name]
                if region.kind == "point":
                    raise ValueError(
                        f"Region '{name}' is a point-type region. "
                        "subset() only supports polygon-type regions. "
                        "Use a boolean mask (bins parameter) to select bins containing specific points."
                    )
                elif region.kind == "polygon":
                    # Use vectorized shapely operation for performance
                    from shapely import contains_xy

                    if self.bin_centers.shape[1] != 2:
                        raise ValueError(
                            f"Polygon regions only work for 2D environments. "
                            f"This environment has {self.bin_centers.shape[1]} dimensions. "
                            "Use bins parameter for N-dimensional selection."
                        )

                    # Vectorized containment check (much faster than loop)
                    in_region = contains_xy(
                        region.data, self.bin_centers[:, 0], self.bin_centers[:, 1]
                    )
                    mask |= in_region

        elif polygon is not None:
            # Validate polygon parameter
            try:
                import shapely.geometry.base
                from shapely import contains_xy

                # Type check
                if not isinstance(polygon, shapely.geometry.base.BaseGeometry):
                    raise TypeError(
                        f"polygon must be a Shapely geometry object, got {type(polygon)}"
                    )

                # Dimension check
                if self.bin_centers.shape[1] != 2:
                    raise ValueError(
                        f"Polygon selection only works for 2D environments. "
                        f"This environment has {self.bin_centers.shape[1]} dimensions."
                    )

                # Vectorized containment check (150x faster than Python loop)
                mask = contains_xy(
                    polygon, self.bin_centers[:, 0], self.bin_centers[:, 1]
                )

            except (AttributeError, TypeError, ValueError) as e:
                raise ValueError(f"Invalid polygon: {e}") from e

        else:
            # Should never reach here due to earlier validation
            raise RuntimeError("No selection method specified (should be unreachable)")

        # Apply invert if requested
        if invert:
            mask = ~mask

        # Check that selection is not empty
        if not np.any(mask):
            raise ValueError(
                f"No bins selected. Selection resulted in empty mask. (invert={invert})"
            )

        # --- Extract Subgraph ---

        # Get selected node indices
        selected_nodes = np.where(mask)[0].tolist()

        # Extract induced subgraph
        subgraph = self.connectivity.subgraph(selected_nodes).copy()

        # --- Renumber Nodes ---

        # Create mapping: old_node_id -> new_node_id
        old_to_new = {old_id: new_id for new_id, old_id in enumerate(selected_nodes)}

        # Create new graph with renumbered nodes
        import networkx as nx

        new_graph = nx.Graph()

        # Add nodes with renumbered IDs and preserved attributes
        for old_id in selected_nodes:
            new_id = old_to_new[old_id]
            node_attrs = self.connectivity.nodes[old_id].copy()
            new_graph.add_node(new_id, **node_attrs)

        # Add edges with renumbered node IDs and preserved attributes
        for u, v, edge_data in subgraph.edges(data=True):
            new_u = old_to_new[u]
            new_v = old_to_new[v]
            new_graph.add_edge(new_u, new_v, **edge_data)

        # --- Extract Bin Centers ---

        # Extract bin centers for selected bins (in new order)
        new_bin_centers = self.bin_centers[selected_nodes]

        # --- Create New Environment ---

        # Use from_layout factory method with custom layout
        # We need to create a minimal layout object that provides the required interface

        # Create a custom layout that wraps our subset data
        # We'll use a simple object that implements the LayoutEngine protocol
        class SubsetLayout:
            """Minimal layout for subset environment."""

            def __init__(
                self, bin_centers, connectivity, dimension_ranges, build_params
            ):
                self.bin_centers = bin_centers
                self.connectivity = connectivity
                self.dimension_ranges = dimension_ranges
                self._layout_type_tag = "subset"
                self._build_params_used = build_params
                self.is_1d = False

            def build(self):
                pass  # Already built

            def point_to_bin_index(self, point):
                # Use KDTree for nearest neighbor
                from scipy.spatial import cKDTree

                tree = cKDTree(self.bin_centers)
                _, idx = tree.query(point)
                return int(idx)

            def bin_sizes(self):
                # Estimate from connectivity graph
                # Use edge distances to estimate bin sizes
                sizes = np.ones(len(self.bin_centers))
                for node in self.connectivity.nodes():
                    neighbors = list(self.connectivity.neighbors(node))
                    if neighbors:
                        distances = [
                            self.connectivity[node][n]["distance"] for n in neighbors
                        ]
                        sizes[node] = np.mean(distances)
                return sizes

            def plot(self, ax=None, **kwargs):
                import matplotlib.pyplot as plt

                if ax is None:
                    _, ax = plt.subplots()

                # Plot bin centers
                if self.bin_centers.shape[1] == 2:
                    ax.scatter(
                        self.bin_centers[:, 0],
                        self.bin_centers[:, 1],
                        **kwargs,
                    )
                return ax

        # Compute dimension ranges from bin centers
        n_dims = new_bin_centers.shape[1]
        dimension_ranges = tuple(
            (float(new_bin_centers[:, i].min()), float(new_bin_centers[:, i].max()))
            for i in range(n_dims)
        )

        # Create layout
        layout = SubsetLayout(
            bin_centers=new_bin_centers,
            connectivity=new_graph,
            dimension_ranges=dimension_ranges,
            build_params={"source": "subset", "original_n_bins": self.n_bins},
        )

        # Create new environment - directly instantiate
        # (from_layout is for factory pattern with string kind)
        sub_env = Environment(
            name="",
            layout=layout,  # type: ignore[arg-type]  # SubsetLayout implements LayoutEngine protocol
            layout_type_used="subset",
            layout_params_used={"source": "subset", "original_n_bins": self.n_bins},
            regions=Regions(),  # Empty regions as documented
        )

        # --- Preserve Metadata ---

        # Copy units and frame if present
        if hasattr(self, "units") and self.units is not None:
            sub_env.units = self.units
        if hasattr(self, "frame") and self.frame is not None:
            sub_env.frame = self.frame

        # Note: Regions are intentionally dropped (as documented)

        return sub_env

    @check_fitted
    def occupancy(
        self,
        times: NDArray[np.float64],
        positions: NDArray[np.float64],
        *,
        speed: NDArray[np.float64] | None = None,
        min_speed: float | None = None,
        max_gap: float | None = 0.5,
        kernel_bandwidth: float | None = None,
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
        max_gap : float, optional
            Maximum time gap in seconds. Intervals with Δt > max_gap are
            not counted toward occupancy. Default: 0.5 seconds. Set to None
            to count all intervals regardless of gap size.
        kernel_bandwidth : float, optional
            If provided, apply diffusion kernel smoothing with this bandwidth
            (in physical units). Uses mode='transition' to preserve total mass.
            Smoothing preserves total occupancy time.

        Returns
        -------
        occupancy : NDArray[np.float64], shape (n_bins,)
            Time in seconds spent in each bin. The sum of occupancy equals
            the total valid time (within numerical precision), excluding
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

        See Also
        --------
        compute_kernel : Compute diffusion kernel for smoothing.
        bin_at : Map single N-dimensional point to bin index.

        Notes
        -----
        **Time allocation**: Each time interval Δt = times[i+1] - times[i]
        is assigned entirely to the bin at positions[i] (the starting position).
        This is a simple and fast approximation. For more accurate boundary
        handling on regular grids, see the 'linear' allocation method (P2.11).

        **Mass conservation**: The sum of the returned occupancy array equals
        the total valid time:

        .. math::
            \\sum_i \\text{occupancy}[i] = \\sum_{\\text{valid } k} (t_{k+1} - t_k)

        where valid intervals satisfy:
        - Δt ≤ max_gap (if max_gap is not None)
        - speed[k] ≥ min_speed (if min_speed is not None)
        - positions[k] is inside environment

        **Kernel smoothing**: When kernel_bandwidth is provided, smoothing
        is applied after accumulation using mode='transition' normalization
        (kernel columns sum to 1), which preserves the total occupancy mass.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> # Create environment
        >>> data = np.array([[0, 0], [20, 20]])
        >>> env = Environment.from_samples(data, bin_size=2.0)
        >>>
        >>> # Basic occupancy
        >>> times = np.array([0.0, 1.0, 2.0, 3.0])
        >>> positions = np.array([[5, 5], [5, 5], [10, 10], [10, 10]])
        >>> occ = env.occupancy(times, positions)
        >>> occ.sum()  # Total time = 3.0 seconds
        3.0
        >>>
        >>> # Filter slow periods and smooth
        >>> speeds = np.array([5.0, 5.0, 0.5, 5.0])
        >>> occ_filtered = env.occupancy(
        ...     times, positions, speed=speeds, min_speed=2.0, kernel_bandwidth=3.0
        ... )

        """
        from neurospatial.spatial import map_points_to_bins

        # Input validation
        times = np.asarray(times, dtype=np.float64)
        positions = np.asarray(positions, dtype=np.float64)

        # Validate monotonicity of timestamps
        if len(times) > 1 and not np.all(np.diff(times) >= 0):
            decreasing_indices = np.where(np.diff(times) < 0)[0]
            raise ValueError(
                "times must be monotonically increasing (non-decreasing). "
                f"Found {len(decreasing_indices)} decreasing interval(s) at "
                f"indices: {decreasing_indices.tolist()[:5]}"  # Show first 5
                + (" ..." if len(decreasing_indices) > 5 else "")
            )

        # Check array shapes
        if times.ndim != 1:
            raise ValueError(
                f"times must be 1-dimensional array, got shape {times.shape}"
            )

        if positions.ndim != 2:
            raise ValueError(
                f"positions must be 2-dimensional array (n_samples, n_dims), "
                f"got shape {positions.shape}"
            )

        if len(times) != len(positions):
            raise ValueError(
                f"times and positions must have same length. "
                f"Got times: {len(times)}, positions: {len(positions)}"
            )

        # Validate positions dimensionality
        if self.dimension_ranges is not None:
            expected_dims = len(self.dimension_ranges)
            if positions.shape[1] != expected_dims:
                raise ValueError(
                    f"positions must have {expected_dims} dimensions to match environment. "
                    f"Got {positions.shape[1]} dimensions."
                )

        # Validate speed parameters
        if min_speed is not None and speed is None:
            raise ValueError(
                "min_speed parameter requires speed array to be provided. "
                "Pass speed=<array> along with min_speed=<threshold>."
            )

        if speed is not None:
            speed = np.asarray(speed, dtype=np.float64)
            if len(speed) != len(times):
                raise ValueError(
                    f"speed and times must have same length. "
                    f"Got speed: {len(speed)}, times: {len(times)}"
                )

        # Handle empty arrays
        if len(times) == 0:
            return np.zeros(self.n_bins, dtype=np.float64)

        # Handle single sample (no intervals to accumulate)
        if len(times) == 1:
            return np.zeros(self.n_bins, dtype=np.float64)

        # Map positions to bin indices
        bin_indices: NDArray[np.int64] = map_points_to_bins(  # type: ignore[assignment]
            positions, self, tie_break="lowest_index"
        )

        # Compute time intervals
        dt = np.diff(times)

        # Build mask for valid intervals
        valid_mask = np.ones(len(dt), dtype=bool)

        # Filter by max_gap
        if max_gap is not None:
            valid_mask &= dt <= max_gap

        # Filter by min_speed (applied to starting position of each interval)
        if min_speed is not None and speed is not None:
            valid_mask &= speed[:-1] >= min_speed

        # Filter out intervals starting outside environment bounds
        # (map_points_to_bins returns -1 for points that don't map to any bin)
        valid_mask &= bin_indices[:-1] >= 0

        # Initialize occupancy array
        occupancy = np.zeros(self.n_bins, dtype=np.float64)

        # Accumulate valid time intervals
        valid_bins = bin_indices[:-1][valid_mask]
        valid_dt = dt[valid_mask]

        # Use np.bincount for efficient accumulation
        # (handles repeated indices automatically)
        if len(valid_bins) > 0:
            counts = np.bincount(valid_bins, weights=valid_dt, minlength=self.n_bins)
            occupancy[:] = counts[: self.n_bins]

        # Apply kernel smoothing if requested
        if kernel_bandwidth is not None:
            # Use mode='transition' for occupancy (counts), not 'density'
            # This ensures mass conservation: kernel columns sum to 1
            kernel = self.compute_kernel(
                bandwidth=kernel_bandwidth, mode="transition", cache=True
            )
            occupancy = kernel @ occupancy

        return occupancy

    @check_fitted
    def bin_sequence(
        self,
        times: NDArray[np.float64],
        positions: NDArray[np.float64],
        *,
        dedup: bool = True,
        return_runs: bool = False,
        outside_value: int | None = -1,
    ) -> (
        NDArray[np.int32]
        | tuple[NDArray[np.int32], NDArray[np.int64], NDArray[np.int64]]
    ):
        """Map trajectory to sequence of bin indices.

        Converts a continuous trajectory (times and positions) into a discrete
        sequence of bin indices, with optional deduplication of consecutive
        repeats and run-length encoding.

        Parameters
        ----------
        times : NDArray[np.float64], shape (n_samples,)
            Timestamps in seconds. Should be monotonically increasing.
        positions : NDArray[np.float64], shape (n_samples, n_dims)
            Position coordinates matching environment dimensions.
        dedup : bool, default=True
            If True, collapse consecutive repeats: [A,A,A,B] → [A,B].
            If False, return bin index for every sample.
        return_runs : bool, default=False
            If True, also return run boundaries (indices into times array).
            A "run" is a maximal contiguous subsequence in the same bin.
        outside_value : int or None, default=-1
            Bin index for samples outside environment bounds.
            - If -1 (default), outside samples are marked with -1.
            - If None, outside samples are dropped from the sequence entirely.

        Returns
        -------
        bins : NDArray[np.int32], shape (n_sequences,)
            Bin index at each time point (or deduplicated sequence).
            Values are in range [0, n_bins-1] for valid bins, or -1 for
            outside samples (when outside_value=-1).
        run_start_idx : NDArray[np.int64], shape (n_runs,), optional
            Start index (into original times array) of each contiguous run.
            Only returned if return_runs=True.
        run_end_idx : NDArray[np.int64], shape (n_runs,), optional
            End index (inclusive, into original times array) of each run.
            Only returned if return_runs=True.

        Raises
        ------
        ValueError
            If times and positions have different lengths, if positions
            have wrong number of dimensions, or if timestamps are not
            monotonically increasing (non-decreasing).

        See Also
        --------
        occupancy : Compute time spent in each bin.
        transitions : Build empirical transition matrix from trajectory.

        Notes
        -----
        A "run" is a maximal contiguous subsequence where all samples map to
        the same bin. When outside_value=-1, runs are split at boundary
        crossings (transitions to/from outside).

        When outside_value=None and samples fall outside the environment,
        they are completely removed from the sequence. This affects run
        boundaries if return_runs=True.

        Timestamps must be monotonically increasing (non-decreasing).
        Sort your data by time before calling this method if needed.

        Examples
        --------
        >>> # Basic usage: deduplicated bin sequence
        >>> bins = env.bin_sequence(times, positions)
        >>>
        >>> # Get run boundaries for duration calculations
        >>> bins, starts, ends = env.bin_sequence(times, positions, return_runs=True)
        >>> # Duration of first run:
        >>> duration = times[ends[0]] - times[starts[0]]
        >>>
        >>> # Keep all samples (no deduplication)
        >>> bins = env.bin_sequence(times, positions, dedup=False)
        >>>
        >>> # Drop outside samples entirely
        >>> bins = env.bin_sequence(times, positions, outside_value=None)

        """
        # Input validation
        times = np.asarray(times, dtype=np.float64)
        positions = np.asarray(positions, dtype=np.float64)

        # Validate positions is 2D (consistent with occupancy())
        if positions.ndim != 2:
            raise ValueError(
                f"positions must be a 2-dimensional array (n_samples, n_dims), "
                f"got shape {positions.shape}"
            )

        # Validate lengths match
        if len(times) != len(positions):
            raise ValueError(
                f"times and positions must have the same length. "
                f"Got times: {len(times)}, positions: {len(positions)}"
            )

        # Validate dimensions match environment
        n_dims = self.n_dims
        if positions.shape[1] != n_dims:
            raise ValueError(
                f"positions must have {n_dims} dimensions to match environment. "
                f"Got positions.shape[1] = {positions.shape[1]}"
            )

        # Check for monotonic timestamps (raise error for consistency with occupancy())
        if len(times) > 1 and not np.all(np.diff(times) >= 0):
            decreasing_indices = np.where(np.diff(times) < 0)[0]
            raise ValueError(
                "times must be monotonically increasing (non-decreasing). "
                f"Found {len(decreasing_indices)} decreasing interval(s) at "
                f"indices: {decreasing_indices.tolist()[:5]}"
                + (" ..." if len(decreasing_indices) > 5 else "")
            )

        # Handle empty input
        if len(times) == 0:
            empty_bins = np.array([], dtype=np.int32)
            if return_runs:
                empty_runs = np.array([], dtype=np.int64)
                return empty_bins, empty_runs, empty_runs
            return empty_bins

        # Map positions to bin indices
        # Use bin_at which returns -1 for points outside environment
        bin_indices = self.bin_at(positions).astype(np.int32)  # Ensure int32 dtype

        # Handle outside_value=None (drop outside samples)
        if outside_value is None:
            # Filter out samples that are outside (bin_indices == -1)
            valid_mask = bin_indices != -1
            bin_indices = bin_indices[valid_mask]

            # Track original indices for run boundaries
            original_indices = np.arange(len(times))[valid_mask]

            if len(bin_indices) == 0:
                # All samples were outside
                empty_bins = np.array([], dtype=np.int32)
                if return_runs:
                    empty_runs = np.array([], dtype=np.int64)
                    return empty_bins, empty_runs, empty_runs
                return empty_bins
        else:
            # Keep original indices (no filtering)
            original_indices = np.arange(len(times))

        # Apply deduplication if requested
        deduplicated_bins: NDArray[np.int32]
        deduplicated_indices: NDArray[np.int_]

        if dedup:
            if len(bin_indices) == 0:
                # Already empty, nothing to deduplicate
                deduplicated_bins = bin_indices
                deduplicated_indices = original_indices
            else:
                # Find change points (where bin index changes)
                # Prepend True to include first element
                change_points = np.concatenate(
                    [[True], bin_indices[1:] != bin_indices[:-1]]
                )
                deduplicated_bins = bin_indices[change_points]
                deduplicated_indices = original_indices[change_points]
        else:
            deduplicated_bins = bin_indices
            deduplicated_indices = original_indices

        # Return just bins if runs not requested
        if not return_runs:
            return deduplicated_bins

        # Compute run boundaries
        if len(deduplicated_bins) == 0:
            # No runs
            empty_runs = np.array([], dtype=np.int64)
            return deduplicated_bins, empty_runs, empty_runs

        # For each run, find start and end indices in the *original* times array
        if dedup:
            # deduplicated_indices already contains the start of each run
            run_starts = deduplicated_indices

            # End of each run is just before the start of the next run
            # (or the last valid index for the final run)
            if outside_value is None:
                # Use the last valid index from original_indices
                run_ends = np.concatenate(
                    [deduplicated_indices[1:] - 1, [original_indices[-1]]]
                )
            else:
                # Use len(times) - 1 for the last run end
                run_ends = np.concatenate(
                    [deduplicated_indices[1:] - 1, [len(times) - 1]]
                )
        else:
            # No dedup: find runs in the un-deduplicated bin_indices
            # Find change points to identify run boundaries
            if len(bin_indices) == 1:
                # Single sample = single run
                run_starts = np.array([original_indices[0]], dtype=np.int64)
                run_ends = np.array([original_indices[0]], dtype=np.int64)
            else:
                # Find where bin index changes
                # A change occurs when bin_indices[i] != bin_indices[i-1]
                is_change = np.concatenate(
                    [[True], bin_indices[1:] != bin_indices[:-1]]
                )
                change_positions = np.where(is_change)[0]

                # Start of each run is at a change position
                run_starts = original_indices[change_positions]

                # End of each run is just before the next change (or last index)
                run_ends = np.concatenate(
                    [original_indices[change_positions[1:] - 1], [original_indices[-1]]]
                )

        return deduplicated_bins, run_starts, run_ends

    def _empirical_transitions(
        self,
        bins: NDArray[np.int32] | None = None,
        *,
        times: NDArray[np.float64] | None = None,
        positions: NDArray[np.float64] | None = None,
        lag: int = 1,
        normalize: bool = True,
        allow_teleports: bool = False,
    ) -> scipy.sparse.csr_matrix:
        """Compute empirical transition matrix from observed trajectory data.

        Internal helper for transitions() method. Counts observed transitions
        between bins in a trajectory.

        Parameters
        ----------
        bins : NDArray[np.int32], shape (n_samples,), optional
            Precomputed bin sequence. If None, computed from times/positions.
            Cannot be provided together with times/positions.
            Must contain valid bin indices in range [0, n_bins). Outside values
            (-1) are not allowed; use times/positions input to handle outside samples.
        times : NDArray[np.float64], shape (n_samples,), optional
            Timestamps in seconds. Required if bins is None.
            Must be provided together with positions.
        positions : NDArray[np.float64], shape (n_samples, n_dims), optional
            Position coordinates matching environment dimensions.
            Required if bins is None. Must be provided together with times.
        lag : int, default=1
            Temporal lag for transitions: count bins[t] → bins[t+lag].
            Must be positive. lag=1 counts consecutive transitions,
            lag=2 skips one bin, etc.
        normalize : bool, default=True
            If True, return row-stochastic matrix where each row sums to 1
            (representing transition probabilities).
            If False, return raw transition counts.
        allow_teleports : bool, default=False
            If False, only count transitions between graph-adjacent bins.
            Non-adjacent transitions (e.g., from tracking errors) are excluded.
            Self-transitions (staying in same bin) are always counted.
            If True, count all transitions including non-local jumps.

        Returns
        -------
        T : scipy.sparse.csr_matrix, shape (n_bins, n_bins)
            Transition matrix where T[i,j] represents:
            - If normalize=True: P(next_bin=j | current_bin=i)
            - If normalize=False: count of i→j transitions

            For normalized matrices, each row sums to 1.0 (rows with no
            transitions sum to 0.0).

        Raises
        ------
        ValueError
            If neither bins nor times/positions are provided.
            If both bins and times/positions are provided.
            If only one of times or positions is provided.
            If bins contains invalid indices outside [0, n_bins).
            If lag is not positive.

        See Also
        --------
        bin_sequence : Convert trajectory to bin indices.
        occupancy : Compute time spent in each bin.

        Notes
        -----
        When allow_teleports=False, the method filters out non-adjacent
        transitions by checking the environment's connectivity graph. This
        helps remove artifacts from tracking errors or data gaps.

        Self-transitions (staying in the same bin) are always counted.

        The sparse CSR format is memory-efficient for large environments
        where most bin pairs have no observed transitions.

        Examples
        --------
        >>> # Compute transition probabilities from trajectory
        >>> T = env.transitions(times=times, positions=positions)
        >>> # Probability of moving from bin 10 to its neighbors
        >>> T[10, :].toarray()

        >>> # Get raw transition counts with teleport filtering
        >>> T_counts = env.transitions(
        ...     bins=bin_sequence, normalize=False, allow_teleports=False
        ... )

        >>> # Multi-step transitions (lag=2)
        >>> T_2step = env.transitions(bins=bin_sequence, lag=2)

        """
        import scipy.sparse

        # Validation: Ensure exactly one input method is used
        bins_provided = bins is not None
        trajectory_provided = times is not None or positions is not None

        if not bins_provided and not trajectory_provided:
            raise ValueError(
                "Must provide either 'bins' or both 'times' and 'positions'."
            )

        if bins_provided and trajectory_provided:
            raise ValueError(
                "Cannot provide both 'bins' and 'times'/'positions'. "
                "Use one input method only."
            )

        # If times/positions provided, validate both are present
        if trajectory_provided:
            if times is None or positions is None:
                raise ValueError(
                    "Both times and positions must be provided together "
                    "when computing transitions from trajectory."
                )

            # Compute bin sequence from trajectory
            bins = self.bin_sequence(times, positions, dedup=False, outside_value=-1)

        # Convert to numpy array and validate dtype
        bins = np.asarray(bins)
        if not np.issubdtype(bins.dtype, np.integer):
            raise ValueError(
                f"bins must be an integer array, got dtype {bins.dtype}. "
                f"Ensure bin indices are integers before calling transitions()."
            )
        bins = bins.astype(np.int32)

        # Validate lag
        if lag < 1:
            raise ValueError(f"lag must be positive (got {lag}).")

        # Handle empty or single-element sequences
        if len(bins) == 0 or len(bins) <= lag:
            # Return empty sparse matrix
            return scipy.sparse.csr_matrix((self.n_bins, self.n_bins), dtype=float)

        # Validate bin indices (must be in [0, n_bins))
        # Note: -1 is used for outside values, which is invalid for transitions
        if np.any(bins < 0) or np.any(bins >= self.n_bins):
            invalid_mask = (bins < 0) | (bins >= self.n_bins)
            invalid_indices = np.where(invalid_mask)[0]
            invalid_values = bins[invalid_mask]
            raise ValueError(
                f"Invalid bin indices found outside range [0, {self.n_bins}). "
                f"Found {len(invalid_indices)} invalid values at indices "
                f"{invalid_indices[:5].tolist()}{'...' if len(invalid_indices) > 5 else ''}: "
                f"{invalid_values[:5].tolist()}{'...' if len(invalid_values) > 5 else ''}. "
                f"Note: -1 (outside) values are not allowed in transitions."
            )

        # Extract transition pairs with lag
        source_bins = bins[:-lag]
        target_bins = bins[lag:]

        # Filter non-adjacent transitions if requested
        if not allow_teleports:
            # Build adjacency set from connectivity graph
            adjacency_set = set()
            for u, v in self.connectivity.edges():
                adjacency_set.add((u, v))
                adjacency_set.add((v, u))  # Undirected graph

            # Also include self-transitions (always adjacent)
            for node in self.connectivity.nodes():
                adjacency_set.add((node, node))

            # Filter transitions to only adjacent pairs
            is_adjacent = np.array(
                [
                    (src, tgt) in adjacency_set
                    for src, tgt in zip(source_bins, target_bins, strict=True)
                ]
            )

            source_bins = source_bins[is_adjacent]
            target_bins = target_bins[is_adjacent]

        # Count transitions using sparse COO format
        # Use np.ones to count occurrences
        transition_counts = np.ones(len(source_bins), dtype=float)

        # Build sparse matrix in COO format
        transition_matrix = scipy.sparse.coo_matrix(
            (transition_counts, (source_bins, target_bins)),
            shape=(self.n_bins, self.n_bins),
            dtype=float,
        )

        # Convert to CSR for efficient row operations
        transition_matrix = transition_matrix.tocsr()

        # Sum duplicate entries (multiple transitions between same bins)
        transition_matrix.sum_duplicates()

        # Normalize rows if requested
        if normalize:
            # Get row sums
            row_sums = np.array(transition_matrix.sum(axis=1)).flatten()

            # Avoid division by zero: only normalize rows with transitions
            nonzero_rows = row_sums > 0

            # Create diagonal matrix for normalization
            # Use reciprocal of row sums for nonzero rows, 0 otherwise
            inv_row_sums = np.zeros(self.n_bins)
            inv_row_sums[nonzero_rows] = 1.0 / row_sums[nonzero_rows]

            # Normalize: T_normalized = diag(1/row_sums) @ T
            normalizer = scipy.sparse.diags(inv_row_sums, format="csr")
            transition_matrix = normalizer @ transition_matrix

        return transition_matrix

    def _random_walk_transitions(
        self,
        *,
        normalize: bool = True,
    ) -> scipy.sparse.csr_matrix:
        """Compute uniform random walk transition matrix from graph structure.

        Internal helper for transitions(method='random_walk'). Creates a
        transition matrix where each bin transitions uniformly to its neighbors.
        """
        import scipy.sparse

        # Get adjacency matrix from connectivity graph
        adjacency = nx.adjacency_matrix(self.connectivity, nodelist=range(self.n_bins))

        # Convert to float and ensure CSR format
        transition_matrix = adjacency.astype(float).tocsr()

        if normalize:
            # Normalize rows: T[i,j] = 1/degree(i) if j is neighbor of i
            row_sums = np.array(transition_matrix.sum(axis=1)).flatten()

            # Avoid division by zero for isolated nodes
            nonzero_rows = row_sums > 0
            inv_row_sums = np.zeros(self.n_bins)
            inv_row_sums[nonzero_rows] = 1.0 / row_sums[nonzero_rows]

            # Normalize
            normalizer = scipy.sparse.diags(inv_row_sums, format="csr")
            transition_matrix = normalizer @ transition_matrix

        return transition_matrix

    def _diffusion_transitions(
        self,
        bandwidth: float,
        *,
        normalize: bool = True,
    ) -> scipy.sparse.csr_matrix:
        """Compute diffusion-based transition matrix using heat kernel.

        Internal helper for transitions(method='diffusion'). Uses the heat
        kernel to model continuous-time diffusion on the graph.
        """
        import scipy.sparse

        # Use existing compute_kernel infrastructure
        kernel = self.compute_kernel(bandwidth=bandwidth, mode="transition")

        # kernel is already row-stochastic from compute_kernel
        # Convert to sparse if needed
        if not scipy.sparse.issparse(kernel):
            kernel = scipy.sparse.csr_matrix(kernel)

        if not normalize:
            raise ValueError(
                "method='diffusion' does not support normalize=False. "
                "Heat kernel transitions are inherently normalized (row-stochastic). "
                "Set normalize=True or use method='random_walk'."
            )

        return kernel

    @check_fitted
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
        >>> # Empirical transitions from trajectory
        >>> T_empirical = env.transitions(times=times, positions=positions)

        >>> # Empirical from precomputed bins with lag
        >>> T_lag2 = env.transitions(bins=bin_sequence, lag=2, allow_teleports=True)

        >>> # Model: uniform random walk
        >>> T_random = env.transitions(method="random_walk")

        >>> # Model: diffusion with spatial bias
        >>> T_diffusion = env.transitions(method="diffusion", bandwidth=5.0)

        >>> # Compare empirical vs model
        >>> diff = (T_empirical - T_diffusion).toarray()
        >>> # Large differences indicate non-random exploration

        """
        # Dispatch based on mode
        if method is not None:
            # MODEL-BASED MODE
            # Validate that empirical inputs aren't provided
            if bins is not None or times is not None or positions is not None:
                raise ValueError(
                    "Cannot provide both 'method' (model-based) and empirical "
                    "inputs (bins/times/positions). Choose one mode."
                )

            # Validate that empirical parameters aren't silently ignored
            if lag != 1:
                raise ValueError(
                    f"Parameter 'lag' is only valid in empirical mode. "
                    f"Got lag={lag} with method='{method}'. "
                    f"Remove 'lag' parameter or set method=None for empirical mode."
                )
            if allow_teleports is not False:
                raise ValueError(
                    f"Parameter 'allow_teleports' is only valid in empirical mode. "
                    f"Got allow_teleports={allow_teleports} with method='{method}'. "
                    f"Remove 'allow_teleports' parameter or set method=None for empirical mode."
                )

            # Validate bandwidth parameter usage
            if method == "random_walk" and bandwidth is not None:
                raise ValueError(
                    f"Parameter 'bandwidth' is only valid with method='diffusion'. "
                    f"Got bandwidth={bandwidth} with method='random_walk'. "
                    f"Remove 'bandwidth' parameter."
                )

            # Dispatch to model-based method
            if method == "random_walk":
                return self._random_walk_transitions(normalize=normalize)
            elif method == "diffusion":
                if bandwidth is None:
                    raise ValueError(
                        "method='diffusion' requires 'bandwidth' parameter. "
                        "Provide bandwidth in physical units (sigma)."
                    )
                return self._diffusion_transitions(
                    bandwidth=bandwidth, normalize=normalize
                )
            else:
                raise ValueError(
                    f"Unknown method '{method}'. "
                    f"Valid options: 'random_walk', 'diffusion'."
                )
        else:
            # EMPIRICAL MODE
            return self._empirical_transitions(
                bins=bins,
                times=times,
                positions=positions,
                lag=lag,
                normalize=normalize,
                allow_teleports=allow_teleports,
            )

    @cached_property
    @check_fitted
    def boundary_bins(self) -> NDArray[np.int_]:
        """Get the boundary bin indices.

        Returns
        -------
        NDArray[np.int_], shape (n_boundary_bins,)
            An array of indices of the boundary bins in the environment.
            These are the bins that are at the edges of the active area.

        """
        return find_boundary_nodes(
            graph=self.connectivity,
            grid_shape=self.grid_shape,
            active_mask=self.active_mask,
            layout_kind=self._layout_type_used,
        )

    @cached_property
    @check_fitted
    def linearization_properties(
        self: Environment,
    ) -> dict[str, Any] | None:
        """If the environment uses a GraphLayout, returns properties needed
        for linearization (converting a 2D/3D track to a 1D line) using the
        `track_linearization` library.

        These properties are typically passed to `track_linearization.get_linearized_position`.

        Returns
        -------
        Optional[Dict[str, Any]]
            A dictionary with keys 'track_graph', 'edge_order', 'edge_spacing'
            if the layout is `GraphLayout` and parameters are available.
            Returns `None` otherwise.

        """
        # Use hasattr instead of isinstance to avoid Protocol/concrete class conflict
        if hasattr(self.layout, "to_linear") and hasattr(self.layout, "linear_to_nd"):
            return {
                "track_graph": self._layout_params_used.get("graph_definition"),
                "edge_order": self._layout_params_used.get("edge_order"),
                "edge_spacing": self._layout_params_used.get("edge_spacing"),
            }
        return None

    @cached_property
    @check_fitted
    def bin_attributes(self) -> pd.DataFrame:
        """Build a DataFrame of attributes for each active bin (node) in the environment's graph.

        Returns
        -------
        df : pandas.DataFrame
            Rows are indexed by `active_bin_id` (int), matching 0..(n_bins-1).
            Columns correspond to node attributes. If a 'pos' attribute exists
            for any node and is non-null, it will be expanded into columns
            'pos_dim0', 'pos_dim1', ..., with numeric coordinates.

        Raises
        ------
        ValueError
            If there are no active bins (graph has zero nodes).

        """
        graph = self.connectivity
        if graph.number_of_nodes() == 0:
            raise ValueError("No active bins in the environment.")

        df = pd.DataFrame.from_dict(dict(graph.nodes(data=True)), orient="index")
        df.index.name = "active_bin_id"  # Index is 0..N-1

        if "pos" in df.columns and not df["pos"].dropna().empty:
            pos_df = pd.DataFrame(df["pos"].tolist(), index=df.index)
            pos_df.columns = [f"pos_dim{i}" for i in range(pos_df.shape[1])]
            df = pd.concat([df.drop(columns="pos"), pos_df], axis=1)

        return df

    @cached_property
    @check_fitted
    def edge_attributes(self) -> pd.DataFrame:
        """Return a Pandas DataFrame where each row corresponds to one directed edge
        (u → v) in the connectivity graph, and columns include all stored edge
        attributes (e.g. 'distance', 'vector', 'weight', 'angle_2d', etc.).

        The DataFrame will have a MultiIndex of (source_bin, target_bin). If you
        prefer flat columns, you can reset the index.

        Returns
        -------
        pd.DataFrame
            A DataFrame whose index is a MultiIndex (source_bin, target_bin),
            and whose columns are the union of all attribute-keys stored on each edge.

        Raises
        ------
        ValueError
            If there are no edges in the connectivity graph.
        RuntimeError
            If called before the environment is fitted.

        """
        G = self.connectivity
        if G.number_of_edges() == 0:
            raise ValueError("No edges in the connectivity graph.")

        # Build a dict of edge_attr_dicts keyed by (u, v)
        # networkx's G.edges(data=True) yields (u, v, attr_dict)
        edge_dict: dict[tuple[int, int], dict] = {
            (u, v): data.copy() for u, v, data in G.edges(data=True)
        }

        # Convert that to a DataFrame, using the (u, v) tuples as a MultiIndex
        df = pd.DataFrame.from_dict(edge_dict, orient="index")
        # The index is now a MultiIndex of (u, v)
        df.index = pd.MultiIndex.from_tuples(
            df.index,
            names=["source_bin", "target_bin"],
        )

        return df

    def distance_between(
        self,
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
        point1 : PtArr, shape (n_dims,) or (1, n_dims)
            The first N-dimensional point.
        point2 : PtArr, shape (n_dims,) or (1, n_dims)
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

        """
        source_bin = self.bin_at(np.atleast_2d(point1))[0]
        target_bin = self.bin_at(np.atleast_2d(point2))[0]

        if source_bin == -1 or target_bin == -1:
            # One or both points didn't map to a valid active bin
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
    def shortest_path(
        self,
        source_active_bin_idx: int,
        target_active_bin_idx: int,
    ) -> list[int]:
        """Find the shortest path between two active bins.

        The path is a sequence of active bin indices (0 to n_active_bins - 1)
        connecting the source to the target. Path calculation uses the
        'distance' attribute on the edges of the `connectivity`
        as weights.

        Parameters
        ----------
        source_active_bin_idx : int
            The active bin index (0 to n_active_bins - 1) for the start of the path.
        target_active_bin_idx : int
            The active bin index (0 to n_active_bins - 1) for the end of the path.

        Returns
        -------
        List[int]
            A list of active bin indices representing the shortest path from
            source to target. The list includes both the source and target indices.
            Returns an empty list if the source and target are the same, or if
            no path exists, or if nodes are not found.

        Raises
        ------
        RuntimeError
            If called before the environment is fitted.
        nx.NodeNotFound
            If `source_active_bin_idx` or `target_active_bin_idx` is not
            a node in the `connectivity`.

        """
        graph = self.connectivity

        if source_active_bin_idx == target_active_bin_idx:
            return [source_active_bin_idx]

        try:
            path = nx.shortest_path(
                graph,
                source=source_active_bin_idx,
                target=target_active_bin_idx,
                weight="distance",
            )
            return list(path)
        except nx.NetworkXNoPath:
            warnings.warn(
                f"No path found between active bin {source_active_bin_idx} "
                f"and {target_active_bin_idx}.",
                UserWarning,
            )
            return []
        except nx.NodeNotFound as e:
            # Re-raise if the user provides an invalid node index for active bins
            raise nx.NodeNotFound(
                f"Node not found in connectivity graph: {e}. "
                "Ensure source/target indices are valid active bin indices.",
            ) from e

    @check_fitted
    def to_linear(self, points_nd: NDArray[np.float64]) -> NDArray[np.float64]:
        """Convert N-dimensional points to 1D linearized coordinates.

        This method is only applicable if the environment uses a `GraphLayout`
        and `is_1d` is True. It delegates to the layout's
        `to_linear` method.

        Parameters
        ----------
        points_nd : NDArray[np.float64], shape (n_points, n_dims)
            N-dimensional points to linearize.

        Returns
        -------
        NDArray[np.float64], shape (n_points,)
            1D linearized coordinates corresponding to the input points.

        Raises
        ------
        TypeError
            If the environment is not 1D or not based on a `GraphLayout`.
        RuntimeError
            If called before the environment is fitted.

        """
        # Use hasattr instead of isinstance to avoid Protocol/concrete class conflict
        if not self.is_1d or not hasattr(self.layout, "to_linear"):
            raise TypeError("Linearized coordinate only for GraphLayout environments.")
        result = self.layout.to_linear(points_nd)
        return np.asarray(result, dtype=np.float64)

    @check_fitted
    def linear_to_nd(
        self,
        linear_coordinates: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Convert 1D linearized coordinates back to N-dimensional coordinates.

        This method is only applicable if the environment uses a `GraphLayout`
        and `is_1d` is True. It delegates to the layout's
        `linear_to_nd` method.

        Parameters
        ----------
        linear_coordinates : NDArray[np.float64], shape (n_points,)
            1D linearized coordinates to map to N-D space.

        Returns
        -------
        NDArray[np.float64], shape (n_points, n_dims)
            N-dimensional coordinates corresponding to the input linear coordinates.

        Raises
        ------
        TypeError
            If the environment is not 1D or not based on a `GraphLayout`.
        RuntimeError
            If called before the environment is fitted.

        """
        # Use hasattr instead of isinstance to avoid Protocol/concrete class conflict
        if not self.is_1d or not hasattr(self.layout, "linear_to_nd"):
            raise TypeError("Mapping linear to N-D only for GraphLayout environments.")
        result = self.layout.linear_to_nd(linear_coordinates)
        return np.asarray(result, dtype=np.float64)

    @check_fitted
    def plot(
        self,
        ax: matplotlib.axes.Axes | None = None,
        show_regions: bool = False,
        layout_plot_kwargs: dict[str, Any] | None = None,
        regions_plot_kwargs: dict[str, Any] | None = None,
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
        **kwargs : Any
            Additional keyword arguments that are passed to `layout.plot()`.
            These can be overridden by `layout_plot_kwargs`.

        Returns
        -------
        matplotlib.axes.Axes
            The axes on which the environment was plotted.

        Raises
        ------
        RuntimeError
            If called before the environment is fitted.

        """
        l_kwargs = layout_plot_kwargs if layout_plot_kwargs is not None else {}
        l_kwargs.update(kwargs)  # Allow direct kwargs to override for layout.plot

        ax = self.layout.plot(ax=ax, **l_kwargs)

        if show_regions and hasattr(self, "regions") and self.regions is not None:
            from neurospatial.regions.plot import plot_regions

            r_kwargs = regions_plot_kwargs if regions_plot_kwargs is not None else {}
            plot_regions(self.regions, ax=ax, **r_kwargs)

        plot_title = self.name
        if (
            self.layout
            and hasattr(self.layout, "_layout_type_tag")
            and self.layout._layout_type_tag
        ):
            plot_title += f" ({self.layout._layout_type_tag})"

        # Only set title if layout.plot didn't set one or user didn't pass one via kwargs to layout.plot
        if ax.get_title() == "":
            ax.set_title(plot_title)

        return ax

    def plot_1d(
        self,
        ax: matplotlib.axes.Axes | None = None,
        layout_plot_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        """Plot a 1D representation of the environment, if applicable.

        This method is primarily for environments where `is_1d` is True
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
        matplotlib.axes.Axes
            The axes on which the 1D layout was plotted, or the original `ax`
            if plotting was not applicable.

        Raises
        ------
        RuntimeError
            If called before the environment is fitted.
        AttributeError
            If `self.layout.is_1d` is True but the layout does not have a
            `plot_linear_layout` method.

        """
        l_kwargs = layout_plot_kwargs if layout_plot_kwargs is not None else {}
        l_kwargs.update(kwargs)  # Allow direct kwargs to override for layout.plot
        if self.layout.is_1d:
            if hasattr(self.layout, "plot_linear_layout"):
                ax = self.layout.plot_linear_layout(ax=ax, **l_kwargs)
            else:
                warnings.warn(
                    f"Layout '{self._layout_type_used}' is 1D but does not "
                    "have a 'plot_linear_layout' method. Skipping 1D plot.",
                    UserWarning,
                )
        else:
            warnings.warn(
                "Environment is not 1D. Skipping 1D plot. Use regular plot() method.",
                UserWarning,
            )

        return ax

    @check_fitted
    def save(self, filename: str = "environment.pkl") -> None:
        """Save the Environment object to a file using pickle.

        Parameters
        ----------
        filename : str, optional
            The name of the file to save the environment to.
            Defaults to "environment.pkl".

        Warnings
        --------
        This method uses pickle for serialization. Pickle files can execute
        arbitrary code during deserialization. Only share pickle files with
        trusted users and only load files from trusted sources.

        See Also
        --------
        load : Load an Environment from a pickle file.

        """
        with Path(filename).open("wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Environment saved to %s", filename)

    @classmethod
    def load(cls, filename: str) -> Environment:
        """Load an Environment object from a pickled file.

        Parameters
        ----------
        filename : str
            The name of the file to load the environment from.

        Returns
        -------
        Environment
            The loaded Environment object.

        Raises
        ------
        TypeError
            If the loaded object is not an instance of the Environment class.

        Warnings
        --------
        This method uses pickle for deserialization. **Only load files from
        trusted sources**, as pickle can execute arbitrary code during
        deserialization. Do not load pickle files from untrusted or
        unknown sources.

        See Also
        --------
        save : Save an Environment to a pickle file.

        """
        with Path(filename).open("rb") as fh:
            environment = pickle.load(fh)
        if not isinstance(environment, cls):
            raise TypeError(f"Loaded object is not type {cls.__name__}")
        return environment

    def to_file(self, path: str | Path) -> None:
        """Save Environment to versioned JSON + npz files.

        This method provides stable, reproducible serialization that is safer
        than pickle and compatible across Python versions. Creates two files:
        `{path}.json` (metadata) and `{path}.npz` (arrays).

        Parameters
        ----------
        path : str or Path
            Base path for output files (without extension).
            Will create `{path}.json` and `{path}.npz`.

        Examples
        --------
        >>> env = Environment.from_samples(data, bin_size=2.0)
        >>> env.to_file("my_environment")

        See Also
        --------
        from_file : Load environment from saved files
        save : Legacy pickle-based serialization

        Notes
        -----
        This format is safer than pickle (no arbitrary code execution) and
        more portable across Python versions and platforms.

        """
        from neurospatial.io import to_file as _to_file

        _to_file(self, path)

    @classmethod
    def from_file(cls, path: str | Path) -> Environment:
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

        Examples
        --------
        >>> env = Environment.from_file("my_environment")

        See Also
        --------
        to_file : Save environment to files
        load : Legacy pickle-based deserialization

        """
        from neurospatial.io import from_file as _from_file

        return _from_file(path)

    def to_dict(self) -> dict[str, Any]:
        """Convert Environment to dictionary for in-memory handoff.

        Returns
        -------
        dict[str, Any]
            Dictionary representation with all arrays as lists.

        See Also
        --------
        from_dict : Reconstruct from dictionary
        to_file : Save to disk with efficient binary format

        """
        from neurospatial.io import to_dict as _to_dict

        return _to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Environment:
        """Reconstruct Environment from dictionary representation.

        Parameters
        ----------
        data : dict[str, Any]
            Dictionary from `to_dict()`.

        Returns
        -------
        Environment
            Reconstructed instance.

        See Also
        --------
        to_dict : Convert to dictionary

        """
        from neurospatial.io import from_dict as _from_dict

        return _from_dict(data)

    @check_fitted
    def bins_in_region(self, region_name: str) -> NDArray[np.int_]:
        """Get active bin indices that fall within a specified named region.

        Parameters
        ----------
        region_name : str
            The name of a defined region in `self.regions`.

        Returns
        -------
        NDArray[np.int_]
            Array of active bin indices (0 to n_active_bins - 1)
            that are part of the region.

        Raises
        ------
        KeyError
            If `region_name` is not found in `self.regions`.
        ValueError
            If region kind is unsupported or mask dimensions mismatch.

        """
        region = self.regions[region_name]

        if region.kind == "point":
            point_nd = np.asarray(region.data).reshape(1, -1)
            if point_nd.shape[1] != self.n_dims:
                raise ValueError(
                    f"Region point dimension {point_nd.shape[1]} "
                    f"does not match environment dimension {self.n_dims}.",
                )
            bin_idx = self.bin_at(point_nd)
            return np.asarray(bin_idx[bin_idx != -1], dtype=int)

        if region.kind == "polygon":
            if not _HAS_SHAPELY:  # pragma: no cover
                raise RuntimeError("Polygon region queries require 'shapely'.")
            if self.n_dims != 2:  # pragma: no cover
                raise ValueError(
                    "Polygon regions are only supported for 2D environments.",
                )

            import shapely

            polygon = region.data
            contained_mask = shapely.contains_xy(
                polygon,
                self.bin_centers[:, 0],
                self.bin_centers[:, 1],
            )

            return np.flatnonzero(contained_mask)

        # pragma: no cover
        raise ValueError(f"Unsupported region kind: {region.kind}")

    @check_fitted
    def mask_for_region(self, region_name: str) -> NDArray[np.bool_]:
        """Get a boolean mask over active bins indicating membership in a region.

        Parameters
        ----------
        region_name : str
            Name of region to query.

        Returns
        -------
        NDArray[np.bool_]
            Boolean array of shape (n_active_bins,). True if an active bin
            is part of the region.

        """
        active_bins_for_mask = self.bins_in_region(region_name)
        mask = np.zeros(self.bin_centers.shape[0], dtype=bool)
        if active_bins_for_mask.size > 0:
            mask[active_bins_for_mask] = True
        return mask

    @check_fitted
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
        >>> # Find all components in environment
        >>> comps = env.components()
        >>> print(f"Found {len(comps)} components")
        Found 2 components
        >>> print(f"Largest component has {len(comps[0])} bins")
        Largest component has 150 bins

        >>> # Get only the largest component
        >>> largest = env.components(largest_only=True)[0]
        >>> print(f"Largest component: {len(largest)} of {env.n_bins} bins")
        Largest component: 150 of 200 bins

        """
        # Find connected components using NetworkX
        component_sets = nx.connected_components(self.connectivity)

        # Convert sets to arrays and sort by size (largest first)
        components = [
            np.asarray(sorted(comp), dtype=np.int32) for comp in component_sets
        ]
        components.sort(key=len, reverse=True)

        # Return only largest if requested
        if largest_only:
            return components[:1]

        return components

    @check_fitted
    def reachable_from(
        self,
        source_bin: int,
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
        source_bin : int
            Starting bin index. Must be in range [0, n_bins).
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
            The source bin is always reachable (reachable[source_bin] = True).

        Raises
        ------
        ValueError
            If source_bin is not in valid range [0, n_bins).
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
        >>> # All bins within 3 edges of bin 10
        >>> mask = env.reachable_from(10, radius=3, metric="hops")
        >>> neighbor_bins = np.where(mask)[0]
        >>> print(f"Found {len(neighbor_bins)} neighbors within 3 hops")
        Found 37 neighbors within 3 hops

        >>> # All bins within 50.0 units geodesic distance from goal region
        >>> goal_bin = env.bins_in_region("goal")[0]
        >>> mask = env.reachable_from(goal_bin, radius=50.0, metric="geodesic")
        >>> print(f"Bins within 50 units: {mask.sum()}")
        Bins within 50 units: 125

        >>> # All bins in same component (no radius limit)
        >>> mask = env.reachable_from(source_bin=0, radius=None)
        >>> print(f"Component size: {mask.sum()} bins")
        Component size: 1000 bins

        """
        # Input validation
        if not isinstance(source_bin, (int, np.integer)):
            raise TypeError(
                f"source_bin must be an integer, got {type(source_bin).__name__}"
            )

        if not 0 <= source_bin < self.n_bins:
            raise ValueError(
                f"source_bin must be in range [0, n_bins) where n_bins={self.n_bins}. "
                f"Got source_bin={source_bin}"
            )

        if radius is not None and radius < 0:
            raise ValueError(
                f"radius must be non-negative or None. Got radius={radius}"
            )

        if metric not in ("hops", "geodesic"):
            raise ValueError(
                f"metric must be 'hops' or 'geodesic'. Got metric='{metric}'"
            )

        # Initialize result mask
        reachable = np.zeros(self.n_bins, dtype=bool)

        # Case 1: No radius limit - find entire connected component
        if radius is None:
            # Use NetworkX to find all nodes in same component
            for component_nodes in nx.connected_components(self.connectivity):
                if source_bin in component_nodes:
                    for node in component_nodes:
                        reachable[node] = True
                    break
            return reachable

        # Case 2: Radius-limited search
        if metric == "hops":
            # Breadth-first search to specified depth
            # Use NetworkX's single_source_shortest_path_length with cutoff
            distances = nx.single_source_shortest_path_length(
                self.connectivity, source_bin, cutoff=int(radius)
            )
            # Mark all nodes within radius as reachable
            for node in distances:
                reachable[node] = True

        else:  # metric == 'geodesic'
            # Dijkstra's algorithm with distance cutoff
            # Use NetworkX's single_source_dijkstra_path_length
            try:
                distances = nx.single_source_dijkstra_path_length(
                    self.connectivity, source_bin, cutoff=radius, weight="distance"
                )
                # Mark all nodes within radius as reachable
                for node in distances:
                    reachable[node] = True
            except nx.NetworkXError:
                # If source_bin has no edges, only mark itself as reachable
                reachable[source_bin] = True

        return reachable
