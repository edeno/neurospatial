"""Analysis methods for Environment class.

This module provides methods for analyzing and extracting information from
spatial environments, including boundary detection, attribute extraction,
and coordinate transformations.

Key Features
------------
- Boundary bin detection for identifying edge bins
- Attribute extraction to DataFrames for analysis
- Linearization support for 1D environments (GraphLayout)
- Cached properties for efficient repeated access

Notes
-----
This is a mixin class designed to be used with Environment. It should NOT
be decorated with @dataclass. Only the main Environment class in core.py
should be a dataclass.

TYPE_CHECKING Pattern
---------------------
To avoid circular imports, we import Environment only for type checking:

    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
    import scipy.sparse
        from neurospatial.environment.core import Environment

Then use string annotations in method signatures: `self: "Environment"`

Examples
--------
This class is not used directly. Instead, it's mixed into Environment:

    >>> from neurospatial import Environment
    >>> import numpy as np
    >>> data = np.random.rand(100, 2) * 10
    >>> env = Environment.from_samples(data, bin_size=2.0)
    >>>
    >>> # Get boundary bins (from EnvironmentAnalysis)
    >>> boundary = env.boundary_bins
    >>> print(f"Found {len(boundary)} boundary bins")
    Found ... boundary bins
    >>>
    >>> # Get bin attributes as DataFrame
    >>> df = env.bin_attributes
    >>> print(df.columns.tolist())
    ['source_grid_flat_index', 'original_grid_nd_index', 'pos_dim0', 'pos_dim1']

"""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal

import networkx as nx
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from neurospatial.environment.decorators import check_fitted
from neurospatial.layout.helpers.utils import find_boundary_nodes

if TYPE_CHECKING:
    import scipy.sparse

    from neurospatial.environment.core import Environment


class EnvironmentAnalysis:
    """Analysis methods mixin for Environment.

    This mixin provides methods for analyzing spatial environments:
    - Boundary detection
    - Attribute extraction to DataFrames
    - Linearization (for 1D environments)
    - Linearization properties

    All methods assume the Environment is fitted (has been initialized
    via a factory method).

    See Also
    --------
    Environment : Main class that uses this mixin
    EnvironmentQueries : Spatial query methods
    EnvironmentVisualization : Plotting methods

    """

    @cached_property
    @check_fitted  # Check only on first access, then value is cached
    def boundary_bins(self: Environment) -> NDArray[np.int_]:
        """Get the boundary bin indices.

        Returns
        -------
        NDArray[np.int_], shape (n_boundary_bins,)
            An array of indices of the boundary bins in the environment.
            These are the bins that are at the edges of the active area.

        Notes
        -----
        This property is cached after first access. The fitted check only
        runs on the first computation, not on subsequent cached accesses.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> data = np.random.rand(100, 2) * 10
        >>> env = Environment.from_samples(data, bin_size=2.0)
        >>> boundary = env.boundary_bins
        >>> print(f"Number of boundary bins: {len(boundary)}")
        Number of boundary bins: ...

        """
        return find_boundary_nodes(
            graph=self.connectivity,
            grid_shape=self.grid_shape,
            active_mask=self.active_mask,
            layout_kind=self._layout_type_used,
        )

    @cached_property
    @check_fitted  # Check only on first access, then value is cached
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
            Individual values may be None if not available in layout parameters.
            Returns `None` for non-1D environments.

        Notes
        -----
        This property is cached after first access. The fitted check only
        runs on the first computation, not on subsequent cached accesses.

        Examples
        --------
        >>> from neurospatial import Environment
        >>> # Create a 1D environment (requires GraphLayout)
        >>> # env = Environment.from_graph(...)  # Requires track-linearization
        >>> # props = env.linearization_properties
        >>> # if props is not None:
        >>> #     print(props.keys())
        >>> #     dict_keys(['track_graph', 'edge_order', 'edge_spacing'])

        Notes
        -----
        This property returns None for non-1D environments or environments
        not created from graph-based layouts.

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
    @check_fitted  # Check only on first access, then value is cached
    def bin_attributes(self: Environment) -> pd.DataFrame:
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

        Notes
        -----
        This property is cached after first access. The fitted check only
        runs on the first computation, not on subsequent cached accesses.

        For very large environments (>100,000 bins), this method converts
        the entire connectivity graph to a DataFrame which may consume
        significant memory. Consider querying specific attributes from
        self.connectivity directly if you only need a subset.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> data = np.random.rand(100, 2) * 10
        >>> env = Environment.from_samples(data, bin_size=2.0)
        >>> df = env.bin_attributes
        >>> print(df.columns.tolist())
        ['source_grid_flat_index', 'original_grid_nd_index', 'pos_dim0', 'pos_dim1']
        >>> print(f"Shape: {df.shape}")
        Shape: (..., 4)

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
    @check_fitted  # Check only on first access, then value is cached
    def edge_attributes(self: Environment) -> pd.DataFrame:
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

        Notes
        -----
        This property is cached after first access. The fitted check only
        runs on the first computation, not on subsequent cached accesses.

        For very large environments (>100,000 edges), this method converts
        the entire connectivity graph to a DataFrame which may consume
        significant memory. Consider querying specific attributes from
        self.connectivity directly if you only need a subset.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> data = np.random.rand(100, 2) * 10
        >>> env = Environment.from_samples(data, bin_size=2.0)
        >>> df = env.edge_attributes
        >>> print(df.columns.tolist())
        ['distance', 'vector', 'edge_id', 'angle_2d']
        >>> print(f"Number of edges: {len(df)}")
        Number of edges: ...

        """
        graph = self.connectivity
        if graph.number_of_edges() == 0:
            raise ValueError("No edges in the connectivity graph.")

        # Build a dict of edge_attr_dicts keyed by (u, v)
        # networkx's graph.edges(data=True) yields (u, v, attr_dict)
        edge_dict: dict[tuple[int, int], dict] = {
            (u, v): data.copy() for u, v, data in graph.edges(data=True)
        }

        # Convert that to a DataFrame, using the (u, v) tuples as a MultiIndex
        df = pd.DataFrame.from_dict(edge_dict, orient="index")
        # The index is now a MultiIndex of (u, v)
        df.index = pd.MultiIndex.from_tuples(
            df.index,
            names=["source_bin", "target_bin"],
        )

        return df

    @check_fitted
    def to_linear(
        self: Environment, points_nd: NDArray[np.float64]
    ) -> NDArray[np.float64]:
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

        Examples
        --------
        >>> from neurospatial import Environment
        >>> # Create a 1D environment (requires GraphLayout)
        >>> # env = Environment.from_graph(...)  # Requires track-linearization
        >>> # import numpy as np
        >>> # points = np.array([[1.0, 2.0], [3.0, 4.0]])
        >>> # linear_pos = env.to_linear(points)
        >>> # print(linear_pos)
        >>> # [0.5 1.2]  # Example output

        Notes
        -----
        This method requires that the environment was created using a GraphLayout
        (typically via `Environment.from_graph()`). For N-D grid environments,
        this method will raise a TypeError.

        See Also
        --------
        linear_to_nd : Convert linearized coordinates back to N-D
        is_1d : Property indicating if environment is 1D
        linearization_properties : Properties needed for linearization

        """
        # Use hasattr instead of isinstance to avoid Protocol/concrete class conflict
        if not self.is_1d or not hasattr(self.layout, "to_linear"):
            raise TypeError(
                "Linearization is only available for 1D environments (GraphLayout). "
                f"This environment has is_1d={self.is_1d}. "
                "Use Environment.from_graph() to create a 1D environment."
            )
        result = self.layout.to_linear(points_nd)
        return np.asarray(result, dtype=np.float64)

    @check_fitted
    def linear_to_nd(
        self: Environment,
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

        Examples
        --------
        >>> from neurospatial import Environment
        >>> # Create a 1D environment (requires GraphLayout)
        >>> # env = Environment.from_graph(...)  # Requires track-linearization
        >>> # import numpy as np
        >>> # linear_pos = np.array([0.5, 1.2])
        >>> # points = env.linear_to_nd(linear_pos)
        >>> # print(points)
        >>> # [[1.0 2.0]
        >>> #  [3.0 4.0]]  # Example output

        Notes
        -----
        This method requires that the environment was created using a GraphLayout
        (typically via `Environment.from_graph()`). For N-D grid environments,
        this method will raise a TypeError.

        See Also
        --------
        to_linear : Convert N-D points to linearized coordinates
        is_1d : Property indicating if environment is 1D
        linearization_properties : Properties needed for linearization

        """
        # Use hasattr instead of isinstance to avoid Protocol/concrete class conflict
        if not self.is_1d or not hasattr(self.layout, "linear_to_nd"):
            raise TypeError(
                "Linearization is only available for 1D environments (GraphLayout). "
                f"This environment has is_1d={self.is_1d}. "
                "Use Environment.from_graph() to create a 1D environment."
            )
        result = self.layout.linear_to_nd(linear_coordinates)
        return np.asarray(result, dtype=np.float64)

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

    def interpolate(
        self,
        field: NDArray[np.float64],
        points: NDArray[np.float64],
        *,
        mode: Literal["nearest", "linear"] = "nearest",
    ) -> NDArray[np.float64]:
        """Interpolate field values at arbitrary points.

        Evaluates bin-valued fields at continuous query points using either
        nearest-neighbor or linear interpolation. Nearest mode works on all
        layout types; linear mode requires regular grid layouts.

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
              regular grids. Only supported for RegularGridLayout.
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
            don't match.
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

        **Outside handling**: Points outside the environment bounds return NaN
        in both modes. This prevents extrapolation errors.

        Examples
        --------
        >>> # Nearest-neighbor interpolation (all layouts)
        >>> field = np.random.rand(env.n_bins)
        >>> query_points = np.array([[5.0, 5.0], [7.5, 3.2]])
        >>> values = env.interpolate(field, query_points, mode="nearest")

        >>> # Linear interpolation (grids only)
        >>> # For plane f(x,y) = 2x + 3y, interpolation is exact
        >>> plane_field = 2 * env.bin_centers[:, 0] + 3 * env.bin_centers[:, 1]
        >>> values = env.interpolate(plane_field, query_points, mode="linear")

        >>> # Evaluate rate map at trajectory positions
        >>> rates_at_trajectory = env.interpolate(rate_map, positions, mode="linear")

        """
        # Input validation - field
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

        # Check for NaN/Inf values in field
        if np.any(np.isnan(field)):
            raise ValueError(
                "Field contains NaN values. "
                f"Found {np.sum(np.isnan(field))} NaN values out of {len(field)} bins. "
                "NaN values are not supported in interpolation operations."
            )

        if np.any(np.isinf(field)):
            raise ValueError(
                "Field contains infinite values. "
                f"Found {np.sum(np.isinf(field))} infinite values out of {len(field)} bins. "
                "Infinite values are not supported in interpolation operations."
            )

        # Input validation - points
        points = np.asarray(points, dtype=np.float64)

        # Check points dimensionality
        if points.ndim != 2:
            raise ValueError(
                f"Points must be 2-D array (got {points.ndim}-D array). "
                f"Expected shape (n_points, n_dims), got shape {points.shape}."
            )

        # Check points dimension matches environment
        n_dims = self.bin_centers.shape[1]
        if points.shape[1] != n_dims:
            raise ValueError(
                f"Points dimension {points.shape[1]} must match environment "
                f"dimension {n_dims}. Expected shape (n_points, {n_dims}), "
                f"got shape {points.shape}."
            )

        # Check for NaN/Inf values in points
        if np.any(~np.isfinite(points)):
            n_invalid = np.sum(~np.isfinite(points))
            raise ValueError(
                f"Points array contains {n_invalid} non-finite value(s) (NaN or Inf). "
                f"All point coordinates must be finite. Check your input data for "
                f"missing values or infinities."
            )

        # Validate mode
        valid_modes = {"nearest", "linear"}
        if mode not in valid_modes:
            raise ValueError(
                f"mode must be one of {valid_modes} (got '{mode}'). "
                "Use 'nearest' for nearest-neighbor interpolation or 'linear' "
                "for bilinear/trilinear interpolation (grids only)."
            )

        # Handle empty points array
        if points.shape[0] == 0:
            return np.array([], dtype=np.float64)

        # Dispatch based on mode
        if mode == "nearest":
            return self._interpolate_nearest(field, points)
        else:  # mode == "linear"
            return self._interpolate_linear(field, points)

    def occupancy(
        self,
        times: NDArray[np.float64],
        positions: NDArray[np.float64],
        *,
        speed: NDArray[np.float64] | None = None,
        min_speed: float | None = None,
        max_gap: float | None = 0.5,
        kernel_bandwidth: float | None = None,
        time_allocation: Literal["start", "linear"] = "start",
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
        time_allocation : {'start', 'linear'}, default='start'
            Method for allocating time intervals across bins:

            - 'start': Assign entire Δt to starting bin (fast, works on all layouts).
            - 'linear': Split Δt proportionally across bins traversed by
              straight-line path (more accurate, RegularGridLayout only).

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
        ValueError
            If time_allocation is not 'start' or 'linear'.
        NotImplementedError
            If time_allocation='linear' is used on non-RegularGridLayout.

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
          More accurate for trajectories that cross multiple bins, but only
          supported on RegularGridLayout. Requires ray-grid intersection
          calculations.

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

        # Validate time_allocation parameter
        if time_allocation not in ("start", "linear"):
            raise ValueError(
                f"time_allocation must be 'start' or 'linear' (got '{time_allocation}'). "
                "Use 'start' for simple allocation (all layouts) or 'linear' for "
                "ray-grid intersection (RegularGridLayout only)."
            )

        # Check layout compatibility for linear allocation
        if (
            time_allocation == "linear"
            and type(self.layout).__name__ != "RegularGridLayout"
        ):
            raise NotImplementedError(
                "time_allocation='linear' is only supported for RegularGridLayout. "
                f"Current layout type: {type(self.layout).__name__}. "
                "Use time_allocation='start' for other layout types."
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

        # Dispatch to appropriate time allocation method
        if time_allocation == "start":
            # Simple allocation: entire interval goes to starting bin
            valid_bins = bin_indices[:-1][valid_mask]
            valid_dt = dt[valid_mask]

            # Use np.bincount for efficient accumulation
            if len(valid_bins) > 0:
                counts = np.bincount(
                    valid_bins, weights=valid_dt, minlength=self.n_bins
                )
                occupancy[:] = counts[: self.n_bins]

        elif time_allocation == "linear":
            # Linear allocation: split time across bins traversed by ray
            occupancy = self._allocate_time_linear(
                positions, dt, valid_mask, bin_indices
            )

        # Apply kernel smoothing if requested
        if kernel_bandwidth is not None:
            # Use mode='transition' for occupancy (counts), not 'density'
            # This ensures mass conservation: kernel columns sum to 1
            kernel = self.compute_kernel(
                bandwidth=kernel_bandwidth, mode="transition", cache=True
            )
            occupancy = kernel @ occupancy

        return occupancy

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

    def _interpolate_nearest(
        self,
        field: NDArray[np.float64],
        points: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Nearest-neighbor interpolation using KDTree.

        Parameters
        ----------
        field : NDArray[np.float64], shape (n_bins,)
            Field values.
        points : NDArray[np.float64], shape (n_points, n_dims)
            Query points.

        Returns
        -------
        values : NDArray[np.float64], shape (n_points,)
            Interpolated values (NaN for points outside).

        """
        from typing import cast

        from neurospatial.spatial import map_points_to_bins

        # Map points to bins (-1 for outside points)
        # With return_dist=False, we get just the indices (not a tuple)
        bin_indices = cast(
            "NDArray[np.int64]",
            map_points_to_bins(
                points, self, tie_break="lowest_index", return_dist=False
            ),
        )

        # Initialize result with NaN
        result = np.full(points.shape[0], np.nan, dtype=np.float64)

        # Fill in values for points inside environment
        inside_mask = bin_indices >= 0
        result[inside_mask] = field[bin_indices[inside_mask]]

        return result

    def _interpolate_linear(
        self,
        field: NDArray[np.float64],
        points: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Linear interpolation using scipy RegularGridInterpolator.

        Parameters
        ----------
        field : NDArray[np.float64], shape (n_bins,)
            Field values.
        points : NDArray[np.float64], shape (n_points, n_dims)
            Query points.

        Returns
        -------
        values : NDArray[np.float64], shape (n_points,)
            Interpolated values (NaN for points outside).

        Raises
        ------
        NotImplementedError
            If layout is not RegularGridLayout.

        """
        # Check layout type - must be RegularGridLayout, not masked/polygon layouts
        # Use _layout_type_tag to avoid mypy Protocol isinstance issues
        if self.layout._layout_type_tag != "RegularGrid":
            raise NotImplementedError(
                f"Linear interpolation (mode='linear') is only supported for "
                f"RegularGridLayout. Current layout type: {type(self.layout).__name__}. "
                f"Use mode='nearest' for non-grid layouts, or create a regular grid "
                f"environment with Environment.from_samples()."
            )

        # Import scipy
        try:
            from scipy.interpolate import RegularGridInterpolator
        except ImportError as e:
            raise ImportError(
                "Linear interpolation requires scipy. Install with: pip install scipy"
            ) from e

        # Get grid properties (we know layout has these from the check above)
        # Cast to Any to work around mypy Protocol limitation
        from typing import cast

        layout_any = cast("Any", self.layout)
        grid_shape: tuple[int, ...] = layout_any.grid_shape
        grid_edges: tuple[NDArray[np.float64], ...] = layout_any.grid_edges
        n_dims = len(grid_shape)

        # Reshape field to grid
        # Note: RegularGridLayout stores bin_centers in row-major order
        field_grid = field.reshape(grid_shape)

        # Create grid points for each dimension (bin centers)
        grid_points: list[NDArray[np.float64]] = []
        for dim in range(n_dims):
            edges = grid_edges[dim]
            # Bin centers are midpoints between edges
            centers = (edges[:-1] + edges[1:]) / 2
            grid_points.append(centers)

        # Create interpolator
        # bounds_error=False, fill_value=np.nan → outside points return NaN
        interpolator = RegularGridInterpolator(
            grid_points,
            field_grid,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )

        # Evaluate at query points
        result: NDArray[np.float64] = interpolator(points)

        return result

    def _allocate_time_linear(
        self,
        positions: NDArray[np.float64],
        dt: NDArray[np.float64],
        valid_mask: NDArray[np.bool_],
        bin_indices: NDArray[np.int64],
    ) -> NDArray[np.float64]:
        """Allocate time intervals linearly across traversed bins (helper for occupancy).

        This method implements ray-grid intersection to split each time interval
        proportionally across all bins crossed by the straight-line path between
        consecutive position samples.

        Parameters
        ----------
        positions : NDArray[np.float64], shape (n_samples, n_dims)
            Position samples.
        dt : NDArray[np.float64], shape (n_samples-1,)
            Time intervals between consecutive samples.
        valid_mask : NDArray[np.bool_], shape (n_samples-1,)
            Boolean mask indicating which intervals are valid (pass filtering).
        bin_indices : NDArray[np.int64], shape (n_samples,)
            Bin indices for each position (-1 if outside environment).

        Returns
        -------
        occupancy : NDArray[np.float64], shape (n_bins,)
            Time allocated to each bin via linear interpolation.

        """
        from neurospatial.layout.engines.regular_grid import RegularGridLayout

        # Ensure we have RegularGridLayout (already validated in occupancy())
        layout: RegularGridLayout = self.layout  # type: ignore[assignment]

        # Get grid structure
        grid_edges = layout.grid_edges
        grid_shape = layout.grid_shape

        # Assert non-None for mypy (RegularGridLayout always has these)
        assert grid_edges is not None, "RegularGridLayout must have grid_edges"
        assert grid_shape is not None, "RegularGridLayout must have grid_shape"

        # Initialize occupancy array
        occupancy = np.zeros(self.n_bins, dtype=np.float64)

        # Process each valid interval
        for i in np.where(valid_mask)[0]:
            start_pos = positions[i]
            end_pos = positions[i + 1]
            interval_time = dt[i]

            # Get starting and ending bin indices
            start_bin = bin_indices[i]
            end_bin = bin_indices[i + 1]

            # If both points are in same bin, simple allocation
            if start_bin == end_bin and start_bin >= 0:
                occupancy[start_bin] += interval_time
                continue

            # Compute ray-grid intersections
            bin_times = self._compute_ray_grid_intersections(
                start_pos, end_pos, list(grid_edges), grid_shape, interval_time
            )

            # Accumulate time to each bin
            for bin_idx, time_in_bin in bin_times:
                if 0 <= bin_idx < self.n_bins:
                    occupancy[bin_idx] += time_in_bin

        return occupancy

    def _compute_ray_grid_intersections(
        self,
        start_pos: NDArray[np.float64],
        end_pos: NDArray[np.float64],
        grid_edges: list[NDArray[np.float64]],
        grid_shape: tuple[int, ...],
        total_time: float,
    ) -> list[tuple[int, float]]:
        """Compute time spent in each bin along a ray (helper for linear allocation).

        Uses DDA-like algorithm to traverse grid and compute intersection distances.

        Parameters
        ----------
        start_pos : NDArray[np.float64], shape (n_dims,)
            Starting position.
        end_pos : NDArray[np.float64], shape (n_dims,)
            Ending position.
        grid_edges : list[NDArray[np.float64]]
            Grid edges per dimension.
        grid_shape : tuple[int, ...]
            Grid shape.
        total_time : float
            Total time interval to split across bins.

        Returns
        -------
        bin_times : list[tuple[int, float]]
            List of (bin_index, time_in_bin) pairs.

        """
        n_dims = len(grid_shape)

        # Compute ray direction and total distance
        ray_dir = end_pos - start_pos
        total_distance = np.linalg.norm(ray_dir)

        # Handle zero-distance case (no movement)
        if total_distance < 1e-12:
            # No movement - allocate all time to starting bin
            start_bin_idx = self._position_to_flat_index(
                start_pos, list(grid_edges), grid_shape
            )
            if start_bin_idx >= 0:
                return [(start_bin_idx, total_time)]
            return []

        # Normalize ray direction
        ray_dir = ray_dir / total_distance

        # Find all grid crossings along each dimension
        crossings: list[tuple[float, int, int]] = []  # (t, dim, grid_index)

        for dim in range(n_dims):
            if abs(ray_dir[dim]) < 1e-12:
                # Ray parallel to this dimension - no crossings
                continue

            edges = grid_edges[dim]
            # Find which edges the ray crosses
            for edge_idx, edge_pos in enumerate(edges):
                # Parametric intersection: start + t * ray_dir = edge_pos
                t = (edge_pos - start_pos[dim]) / ray_dir[dim]
                if 0 < t < total_distance:  # Exclude endpoints
                    crossings.append((t, dim, edge_idx))

        # Sort crossings by distance along ray
        crossings.sort(key=lambda x: x[0])

        # Add start and end points
        segments = [0.0] + [t for t, _, _ in crossings] + [total_distance]

        # Compute bin index and time for each segment
        bin_times: list[tuple[int, float]] = []
        for seg_idx in range(len(segments) - 1):
            # Midpoint of segment (to determine which bin we're in)
            t_mid = (segments[seg_idx] + segments[seg_idx + 1]) / 2
            mid_pos = start_pos + t_mid * ray_dir

            # Get bin index at midpoint
            bin_idx = self._position_to_flat_index(
                mid_pos, list(grid_edges), grid_shape
            )

            if bin_idx >= 0:
                # Compute time in this segment
                seg_distance = segments[seg_idx + 1] - segments[seg_idx]
                seg_time = total_time * (seg_distance / total_distance)
                bin_times.append((bin_idx, seg_time))

        return bin_times

    def _position_to_flat_index(
        self,
        pos: NDArray[np.float64],
        grid_edges: list[NDArray[np.float64]],
        grid_shape: tuple[int, ...],
    ) -> int:
        """Convert N-D position to flat bin index (helper for ray intersection).

        Parameters
        ----------
        pos : NDArray[np.float64], shape (n_dims,)
            Position coordinates.
        grid_edges : list[NDArray[np.float64]]
            Grid edges per dimension.
        grid_shape : tuple[int, ...]
            Grid shape.

        Returns
        -------
        flat_index : int
            Flat bin index, or -1 if position is outside grid bounds.

        """
        n_dims = len(grid_shape)
        nd_index = []

        for dim in range(n_dims):
            edges = grid_edges[dim]
            coord = pos[dim]

            # Find which bin this coordinate falls into
            # bins are [edges[i], edges[i+1])
            bin_idx = np.searchsorted(edges, coord, side="right") - 1

            # Check bounds
            if bin_idx < 0 or bin_idx >= grid_shape[dim]:
                return -1  # Outside grid

            nd_index.append(bin_idx)

        # Convert N-D index to flat index (row-major order)
        flat_idx = 0
        stride = 1
        for dim in reversed(range(n_dims)):
            flat_idx += nd_index[dim] * stride
            stride *= grid_shape[dim]

        return flat_idx
