"""Core Environment class with mixin inheritance.

This module defines the main Environment dataclass that assembles all
functionality from specialized mixin classes:

- EnvironmentFactories: Factory methods (from_samples, from_graph, etc.)
- EnvironmentQueries: Spatial queries (bin_at, contains, neighbors, etc.)
- EnvironmentSerialization: Save/load methods (to_file, from_file, etc.)
- EnvironmentRegions: Region operations (bins_in_region, mask_for_region)
- EnvironmentVisualization: Plotting methods (plot, plot_1d)
- EnvironmentMetrics: Environment metrics (boundary_bins, bin_attributes, to_linear, etc.)
- EnvironmentFields: Spatial field operations (compute_kernel, smooth, interpolate)
- EnvironmentTrajectory: Trajectory analysis (occupancy, bin_sequence, transitions)
- EnvironmentTransforms: Transform operations (rebin, subset)

The Environment class is the ONLY dataclass in the hierarchy - all mixins
are plain classes. This design prevents dataclass field inheritance conflicts
while maintaining clean code organization.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Literal

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from neurospatial._logging import log_environment_created, log_graph_validation
from neurospatial.differential import compute_differential_operator
from neurospatial.environment.decorators import check_fitted
from neurospatial.environment.factories import EnvironmentFactories
from neurospatial.environment.fields import EnvironmentFields
from neurospatial.environment.metrics import EnvironmentMetrics
from neurospatial.environment.queries import EnvironmentQueries
from neurospatial.environment.regions import EnvironmentRegions
from neurospatial.environment.serialization import EnvironmentSerialization
from neurospatial.environment.trajectory import EnvironmentTrajectory
from neurospatial.environment.transforms import EnvironmentTransforms
from neurospatial.environment.visualization import EnvironmentVisualization
from neurospatial.layout.base import LayoutEngine
from neurospatial.layout.validation import (
    GraphValidationError,
    validate_connectivity_graph,
)
from neurospatial.regions import Regions

logger = logging.getLogger(__name__)


@dataclass
class Environment(
    EnvironmentFactories,
    EnvironmentQueries,
    EnvironmentSerialization,
    EnvironmentRegions,
    EnvironmentVisualization,
    EnvironmentMetrics,
    EnvironmentFields,
    EnvironmentTrajectory,
    EnvironmentTransforms,
):
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

    See Also
    --------
    neurospatial.layout.factories.create_layout : Create custom layout engines
    neurospatial.CompositeEnvironment : Merge multiple environments

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
    _kernel_cache: dict[
        tuple[float, Literal["transition", "density"]], NDArray[np.float64]
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
        if n_dims == 1 and hasattr(self, "is_1d") and self.is_1d:
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

    @cached_property
    @check_fitted
    def differential_operator(self) -> sparse.csc_matrix:
        """Compute and cache the differential operator matrix for graph signal processing.

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

        Notes
        -----
        The differential operator satisfies the fundamental relationship:
        L = D @ D.T, where L is the graph Laplacian matrix.

        This property is cached using ``@cached_property``, meaning the matrix
        is computed only once and reused on subsequent accesses. The cache is
        cleared when the environment is copied or modified.

        The differential operator enables efficient graph signal processing:

        - Gradient: grad(f) = D.T @ f  (scalar field → edge field)
        - Divergence: div(g) = D @ g   (edge field → scalar field)
        - Laplacian: lap(f) = D @ D.T @ f = div(grad(f))

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> # Create a simple 1D chain
        >>> data = np.array([[0.0], [1.0], [2.0], [3.0]])
        >>> env = Environment.from_samples(data, bin_size=1.0)
        >>> # Access differential operator (computed and cached)
        >>> D = env.differential_operator
        >>> D.shape
        (4, 3)
        >>> # Subsequent access reuses cached matrix
        >>> D2 = env.differential_operator
        >>> D is D2
        True

        See Also
        --------
        neurospatial.differential.compute_differential_operator : Underlying computation

        References
        ----------
        .. [1] PyGSP: Graph Signal Processing in Python
               https://pygsp.readthedocs.io/
        .. [2] Shuman et al. (2013). "The emerging field of signal processing on graphs."
               IEEE Signal Processing Magazine, 30(3), 83-98.
        """
        return compute_differential_operator(self)

    def copy(self, *, deep: bool = True) -> Environment:
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

        Modifying the copy will not affect the original environment.

        **Shallow copy (deep=False)**:

        - Arrays and graph are shared with the original
        - Modifying the copy will affect the original

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
        >>> env.units = "cm"
        >>>
        >>> # Deep copy (default)
        >>> env_copy = env.copy()
        >>> env_copy.bin_centers[0, 0] = 999.0
        >>> bool(env.bin_centers[0, 0] != 999.0)  # Original unchanged
        True
        >>>
        >>> # Shallow copy
        >>> env_shallow = env.copy(deep=False)
        >>> original_value = env.bin_centers[0, 0]
        >>> env_shallow.bin_centers[0, 0] = 888.0
        >>> bool(env.bin_centers[0, 0] == 888.0)  # Original changed
        True
        >>> # Restore for other tests
        >>> env.bin_centers[0, 0] = original_value
        """
        import copy as copy_module

        if deep:
            # Deep copy: arrays, graph, regions, layout
            env_copy = Environment(
                name=self.name,
                layout=copy_module.deepcopy(self.layout),
                layout_type_used=self._layout_type_used,
                layout_params_used=copy_module.deepcopy(self._layout_params_used),
                regions=copy_module.deepcopy(self.regions),
            )

            # Copy metadata
            env_copy.units = self.units
            env_copy.frame = self.frame
        else:
            # Shallow copy: share references
            env_copy = Environment(
                name=self.name,
                layout=self.layout,  # Shared reference
                layout_type_used=self._layout_type_used,
                layout_params_used=self._layout_params_used,  # Shared reference
                regions=self.regions,  # Shared reference
            )

            # Copy metadata
            env_copy.units = self.units
            env_copy.frame = self.frame

        # Always clear caches (regardless of deep/shallow)
        # This ensures caches are rebuilt for the new environment object
        env_copy._kdtree_cache = None
        env_copy._kernel_cache = {}

        return env_copy
