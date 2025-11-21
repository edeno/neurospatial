"""Spatial field operations for Environment class.

This module provides methods for processing spatial fields, including
kernel computation, smoothing, and interpolation.

Key Features
------------
- Diffusion kernel computation with caching
- Graph-based field smoothing
- Field interpolation at arbitrary points (nearest/linear modes)

Notes
-----
This is a mixin class designed to be used with Environment. It should NOT
be decorated with @dataclass. Only the main Environment class in core.py
should be a dataclass.

TYPE_CHECKING Pattern
---------------------
To avoid circular imports, we import Environment only for type checking.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from neurospatial.environment._protocols import SelfEnv

if TYPE_CHECKING:
    pass
    from neurospatial import Environment


class EnvironmentFields:
    """Spatial field operations mixin.

    Provides methods for processing fields over spatial environments.
    """

    def compute_kernel(
        self: SelfEnv,
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
            Smoothing bandwidth in physical units (σ in the Gaussian kernel),
            must be > 0. Controls the scale of diffusion.
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
        self: SelfEnv,
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
        self: SelfEnv,
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

    def _interpolate_nearest(
        self: SelfEnv,
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
                points,
                cast("Environment", self),
                tie_break="lowest_index",
                return_dist=False,
            ),
        )

        # Initialize result with NaN
        result = np.full(points.shape[0], np.nan, dtype=np.float64)

        # Fill in values for points inside environment
        inside_mask = bin_indices >= 0
        result[inside_mask] = field[bin_indices[inside_mask]]

        return result

    def _interpolate_linear(
        self: SelfEnv,
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
                f"RegularGridLayout. Current layout type: {self.layout._layout_type_tag}. "
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
