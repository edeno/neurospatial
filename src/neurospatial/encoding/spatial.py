"""Spatial rate computation for place, grid, and border cells.

This module provides result classes and compute functions for spatial firing
rate analysis. The result classes wrap firing rate maps with their metadata
and provide convenient methods for analysis and visualization.

Result Classes
--------------
SpatialRateResult
    Single-neuron spatial rate map with convenience methods
SpatialRatesResult
    Multi-neuron spatial rate maps with batch methods and iteration

Compute Functions (to be implemented in Task 2.8-2.9)
-----------------
compute_spatial_rate
    Compute spatial firing rate for one neuron
compute_spatial_rates
    Compute spatial firing rates for multiple neurons

Examples
--------
>>> from neurospatial import Environment
>>> from neurospatial.encoding.spatial import SpatialRateResult
>>> import numpy as np

>>> # Create environment
>>> positions = np.random.rand(100, 2) * 100
>>> env = Environment.from_samples(positions, bin_size=5.0)

>>> # Create result (typically from compute_spatial_rate)
>>> firing_rate = np.random.rand(env.n_bins) * 10
>>> occupancy = np.ones(env.n_bins)
>>> result = SpatialRateResult(
...     firing_rate=firing_rate,
...     occupancy=occupancy,
...     env=env,
...     smoothing_method="diffusion_kde",
...     bandwidth=5.0,
... )

>>> # Use inherited mixin methods
>>> peak = result.peak_locations()  # (n_dims,) coordinates of peak
>>> peak_rate = result.peak_firing_rates()  # scalar max firing rate
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurospatial.encoding._base import SpatialResultMixin, _to_numpy

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes

    from neurospatial import Environment
    from neurospatial.encoding.grid import GridProperties
    from neurospatial.environment._protocols import EnvironmentProtocol

__all__ = [
    "SpatialRateResult",
    "SpatialRatesResult",
    "compute_spatial_rate",
    "compute_spatial_rates",
]


@dataclass(frozen=True)
class SpatialRateResult(SpatialResultMixin):
    """Result of spatial rate computation for a single neuron.

    This class wraps a spatial firing rate map with its associated metadata
    (occupancy, environment, smoothing parameters). It inherits from
    `SpatialResultMixin` for common methods like `peak_locations()` and
    `peak_firing_rates()`.

    Parameters
    ----------
    firing_rate : ArrayLike
        Firing rate map in Hz. Shape is (n_bins,) where n_bins is the
        number of active bins in the environment. Can contain NaN for
        bins with insufficient occupancy.
    occupancy : ArrayLike
        Time spent in each bin in seconds. Shape is (n_bins,).
    env : Environment
        The spatial environment used for the computation. Provides bin
        centers, connectivity, and plotting methods.
    smoothing_method : str
        Smoothing method used: "diffusion_kde", "gaussian_kde", or "binned".
    bandwidth : float
        Smoothing bandwidth in the same units as the environment's bin_size.

    Attributes
    ----------
    firing_rate : ArrayLike
        Firing rate map in Hz. Shape is (n_bins,).
    occupancy : ArrayLike
        Time spent in each bin in seconds. Shape is (n_bins,).
    env : Environment
        The spatial environment.
    smoothing_method : str
        Smoothing method used.
    bandwidth : float
        Smoothing bandwidth.

    Notes
    -----
    This is a frozen dataclass (immutable). All fields are set at construction
    and cannot be modified afterward.

    Inherits from `SpatialResultMixin`, which provides:

    - `peak_locations()`: Returns (n_dims,) coordinates of peak firing
    - `peak_firing_rates()`: Returns scalar max firing rate

    Additional convenience methods (plot, metrics) are implemented in
    Tasks 2.2-2.3.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding.spatial import SpatialRateResult

    >>> # Create a simple environment
    >>> positions = np.random.rand(100, 2) * 50
    >>> env = Environment.from_samples(positions, bin_size=5.0)

    >>> # Create result
    >>> firing_rate = np.random.rand(env.n_bins) * 10
    >>> occupancy = np.ones(env.n_bins)
    >>> result = SpatialRateResult(
    ...     firing_rate=firing_rate,
    ...     occupancy=occupancy,
    ...     env=env,
    ...     smoothing_method="diffusion_kde",
    ...     bandwidth=5.0,
    ... )

    >>> # Access fields
    >>> result.firing_rate.shape
    (n_bins,)
    >>> result.smoothing_method
    'diffusion_kde'

    >>> # Use mixin methods
    >>> peak_coords = result.peak_locations()  # (n_dims,)
    >>> max_rate = result.peak_firing_rates()  # float

    See Also
    --------
    SpatialRatesResult : Batch version for multiple neurons
    compute_spatial_rate : Function to compute this result
    SpatialResultMixin : Provides peak_locations() and peak_firing_rates()
    """

    firing_rate: ArrayLike
    occupancy: ArrayLike
    env: Environment
    smoothing_method: str
    bandwidth: float

    def plot(self, ax: Axes | None = None, **kwargs: Any) -> Axes:
        """Plot the spatial rate map.

        Delegates to the environment's plot_field method for consistent
        visualization across the codebase.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates a new figure and axes.
        **kwargs
            Additional keyword arguments passed to env.plot_field().
            Common options include:
            - cmap : str or Colormap, default="viridis"
            - vmin, vmax : float, colorbar limits
            - add_colorbar : bool, default=True

        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the plot.

        Examples
        --------
        >>> result = SpatialRateResult(...)
        >>> ax = result.plot()
        >>> plt.show()

        >>> fig, ax = plt.subplots()
        >>> result.plot(ax=ax, cmap="hot", vmax=20.0)
        """
        return self.env.plot_field(_to_numpy(self.firing_rate), ax=ax, **kwargs)

    def peak_location(self) -> NDArray[np.float64]:
        """Coordinates of peak firing rate.

        Convenience alias for `peak_locations()` for single-neuron results.

        Returns
        -------
        ndarray, shape (n_dims,)
            Spatial coordinates of the bin with maximum firing rate.

        See Also
        --------
        peak_locations : Inherited method that handles both single and batch
        peak_firing_rates : Get the maximum firing rate value

        Examples
        --------
        >>> result = SpatialRateResult(...)
        >>> peak = result.peak_location()
        >>> print(f"Peak at ({peak[0]:.1f}, {peak[1]:.1f}) cm")
        """
        return self.peak_locations()

    def spatial_information(self) -> float:
        """Skaggs spatial information (bits per spike).

        Quantifies how much information each spike conveys about the
        animal's spatial location. Higher values indicate more spatially
        selective firing.

        Returns
        -------
        float
            Spatial information in bits/spike. Always non-negative.
            Returns 0.0 for uniform firing.

        Notes
        -----
        Uses the Skaggs et al. (1993) formula:

        .. math::

            I = \\sum_i p_i \\frac{r_i}{\\bar{r}} \\log_2 \\left( \\frac{r_i}{\\bar{r}} \\right)

        **Interpretation**:

        - Place cells typically have 1-3 bits/spike
        - Higher values indicate more spatially selective firing
        - Zero means uniform firing (no spatial selectivity)

        References
        ----------
        .. [1] Skaggs, W. E., McNaughton, B. L., & Gothard, K. M. (1993).
               An information-theoretic approach to deciphering the hippocampal code.

        Examples
        --------
        >>> result = SpatialRateResult(...)
        >>> info = result.spatial_information()
        >>> print(f"Spatial information: {info:.2f} bits/spike")
        """
        from neurospatial.encoding._metrics import spatial_information

        return spatial_information(
            _to_numpy(self.firing_rate), _to_numpy(self.occupancy)
        )

    def sparsity(self) -> float:
        """Sparsity of spatial firing.

        Measures what fraction of the environment elicits significant
        firing. Lower values indicate sparser, more selective place fields.

        Returns
        -------
        float
            Sparsity value in range [0, 1].
            - Low (0.1-0.3): Sparse, selective place field
            - High (~1.0): Uniform firing throughout environment

        Notes
        -----
        Uses the Skaggs et al. (1996) formula:

        .. math::

            S = \\frac{\\left( \\sum_i p_i r_i \\right)^2}{\\sum_i p_i r_i^2}

        References
        ----------
        .. [1] Skaggs, W. E., McNaughton, B. L., et al. (1996). Theta phase
               precession in hippocampal neuronal populations.

        Examples
        --------
        >>> result = SpatialRateResult(...)
        >>> spars = result.sparsity()
        >>> print(f"Sparsity: {spars:.2f}")
        """
        from neurospatial.encoding._metrics import sparsity

        return sparsity(_to_numpy(self.firing_rate), _to_numpy(self.occupancy))

    def grid_score(self) -> float:
        """Grid score (hexagonal periodicity).

        Quantifies the hexagonal periodicity of the firing rate map, which
        is characteristic of grid cells. Higher values indicate stronger
        hexagonal grid patterns.

        Returns
        -------
        float
            Grid score in range [-2, 2].
            - score > 0.4: Strong hexagonal grid (typical threshold)
            - score ≈ 0: No hexagonal structure
            - score < 0: Anti-hexagonal structure (rare)
            Returns NaN if grid score cannot be computed (e.g., non-2D grid).

        Notes
        -----
        Computes the spatial autocorrelation of the firing rate map and
        extracts the grid score based on rotational symmetry.

        Uses the Sargolini et al. (2006) algorithm:

        1. Compute 2D spatial autocorrelation via FFT
        2. Rotate by 30°, 60°, 90°, 120°, 150°
        3. Grid score = min(r60, r120) - max(r30, r90, r150)

        This method delegates to ``neurospatial.encoding.grid.grid_score()``.

        References
        ----------
        .. [1] Sargolini, F., Fyhn, M., et al. (2006). Conjunctive
               representation of position, direction, and velocity in
               entorhinal cortex. Science, 312(5774), 758-762.

        See Also
        --------
        grid_properties : Full grid cell metrics (score, scale, orientation)

        Examples
        --------
        >>> result = SpatialRateResult(...)
        >>> score = result.grid_score()
        >>> print(f"Grid score: {score:.3f}")
        """
        from neurospatial.encoding.grid import grid_score as gs_func
        from neurospatial.encoding.grid import spatial_autocorrelation

        firing_rate = _to_numpy(self.firing_rate)

        try:
            # Use "auto" method like batch_grid_scores for consistency
            autocorr = spatial_autocorrelation(self.env, firing_rate, method="auto")

            # spatial_autocorrelation returns 2D array for FFT, tuple for graph
            if isinstance(autocorr, tuple):
                # Graph-based method not compatible with grid_score
                return np.nan

            return gs_func(autocorr)
        except (ValueError, RuntimeError):
            # Handle errors gracefully (e.g., non-2D grid, constant firing, all NaN)
            return np.nan

    def grid_properties(self) -> GridProperties:
        """Full grid cell metrics (score, scale, orientation).

        Returns a comprehensive set of grid cell metrics computed from the
        spatial autocorrelation of the firing rate map.

        Returns
        -------
        GridProperties
            Dataclass containing:

            - score : float
                Grid score in range [-2, 2]
            - scale : float
                Grid spacing in physical units (same as bin_size)
            - orientation : float
                Grid orientation in degrees [0, 60)
            - orientation_std : float
                Standard deviation of orientation estimates
            - peak_coords : NDArray
                Detected peak coordinates (n_peaks, 2)
            - n_peaks : int
                Number of peaks detected

        Notes
        -----
        This method is more efficient than calling ``grid_score()`` separately
        when you need multiple grid metrics, as it performs peak detection
        only once.

        Delegates to ``neurospatial.encoding.grid.grid_properties()``.

        References
        ----------
        .. [1] Sargolini, F., Fyhn, M., et al. (2006). Conjunctive
               representation of position, direction, and velocity in
               entorhinal cortex. Science, 312(5774), 758-762.
        .. [2] Hafting, T., Fyhn, M., et al. (2005). Microstructure of a
               spatial map in the entorhinal cortex. Nature, 436(7052), 801-806.

        See Also
        --------
        grid_score : Just the grid score
        neurospatial.encoding.grid.GridProperties : Return type details

        Examples
        --------
        >>> result = SpatialRateResult(...)
        >>> props = result.grid_properties()
        >>> print(f"Score: {props.score:.2f}")
        >>> print(f"Scale: {props.scale:.1f} cm")
        >>> print(f"Orientation: {props.orientation:.1f}°")
        """
        from neurospatial.encoding.grid import grid_properties as gp_func
        from neurospatial.encoding.grid import spatial_autocorrelation

        firing_rate = _to_numpy(self.firing_rate)
        autocorr = spatial_autocorrelation(self.env, firing_rate, method="fft")
        # FFT method always returns 2D array, not tuple
        if isinstance(autocorr, tuple):
            raise RuntimeError("FFT autocorrelation should return array, not tuple")
        # Use minimum bin size for grid properties (typically same for isotropic grids)
        bin_size = float(np.min(self.env.bin_sizes))
        return gp_func(autocorr, bin_size=bin_size)

    def border_score(
        self,
        threshold: float = 0.3,
        min_area: float = 0.0,
        distance_metric: Literal["geodesic", "euclidean"] = "geodesic",
    ) -> float:
        """Border score (boundary proximity tuning).

        Quantifies how much the cell's firing field is aligned with
        environmental boundaries (walls). Higher values indicate stronger
        border cell properties.

        Parameters
        ----------
        threshold : float, default 0.3
            Fraction of peak firing rate used to segment the field.
            Bins with firing rate >= threshold * peak are included in field.
        min_area : float, default 0.0
            Minimum field area in physical units (e.g., cm²). Fields smaller
            than this return NaN. Default 0.0 (no filtering). For rat
            hippocampal data, Solstad et al. (2008) used 200 cm².
        distance_metric : {'geodesic', 'euclidean'}, default 'geodesic'
            Distance metric for computing distance from field to boundaries.
            - 'geodesic': Graph shortest path distance (respects obstacles)
            - 'euclidean': Straight-line distance in physical space

        Returns
        -------
        float
            Border score in range [-1, 1].
            - +1: Perfect border cell (field on boundary)
            - 0: No boundary preference
            - -1: Anti-border (field in center)
            Returns NaN if border score cannot be computed.

        Notes
        -----
        Uses the Solstad et al. (2008) algorithm:

        1. Segment field at threshold * peak
        2. Compute boundary coverage (fraction of boundary in field)
        3. Compute normalized distance from field to boundary
        4. Border score = (coverage - distance) / (coverage + distance)

        Delegates to ``neurospatial.encoding.border.border_score()``.

        References
        ----------
        .. [1] Solstad, T., Boccara, C. N., et al. (2008). Representation of
               geometric borders in the entorhinal cortex. Science, 322(5909),
               1865-1868.

        See Also
        --------
        region_coverage : Coverage of specific spatial regions

        Examples
        --------
        >>> result = SpatialRateResult(...)
        >>> score = result.border_score()
        >>> print(f"Border score: {score:.3f}")
        """
        from neurospatial.encoding.border import border_score as bs_func

        firing_rate = _to_numpy(self.firing_rate)
        # Cast to EnvironmentProtocol for type checker (Environment implements it)
        env = cast("EnvironmentProtocol", self.env)
        return bs_func(
            env,
            firing_rate,
            threshold=threshold,
            min_area=min_area,
            distance_metric=distance_metric,
        )

    def region_coverage(
        self,
        threshold: float = 0.3,
        regions: list[str] | None = None,
    ) -> dict[str, float]:
        """Coverage of each spatial region by the firing field.

        Computes what fraction of each region's bins are covered by the
        firing field (bins where firing_rate >= threshold * peak).

        Parameters
        ----------
        threshold : float, default 0.3
            Fraction of peak firing rate used to define the field.
            Bins with firing rate >= threshold * peak are included.
        regions : list of str, optional
            Region names to analyze. If None, analyzes all regions
            defined in env.regions.

        Returns
        -------
        dict[str, float]
            Mapping from region name to coverage fraction [0, 1].
            Coverage = (region bins in field) / (total region bins).

        Notes
        -----
        This method is useful for:

        - **Border cell analysis**: Determine which wall a border cell prefers
        - **Task-relevant regions**: Check if fields overlap with reward zones
        - **Multi-zone analysis**: Quantify field distribution across zones

        Delegates to ``neurospatial.encoding.border.compute_region_coverage()``.

        See Also
        --------
        border_score : Overall border preference score
        neurospatial.encoding.border.compute_region_coverage : Direct call

        Examples
        --------
        >>> result = SpatialRateResult(...)
        >>> coverage = result.region_coverage()
        >>> for region, cov in sorted(coverage.items()):
        ...     print(f"{region}: {cov:.1%}")
        """
        from neurospatial.encoding.border import compute_region_coverage

        firing_rate = _to_numpy(self.firing_rate)

        # Threshold field at fraction of peak
        peak_rate = np.nanmax(firing_rate)
        if peak_rate == 0 or np.isnan(peak_rate):
            # No firing, return zero coverage for all regions
            if regions is None:
                regions = list(self.env.regions.keys())
            return dict.fromkeys(regions, 0.0)

        field_mask = firing_rate >= threshold * peak_rate
        field_bins = np.where(field_mask)[0]

        # Cast to EnvironmentProtocol for type checker (Environment implements it)
        env = cast("EnvironmentProtocol", self.env)
        return compute_region_coverage(field_bins, env, regions=regions)


@dataclass(frozen=True)
class SpatialRatesResult(SpatialResultMixin):
    """Result of spatial rate computation for multiple neurons.

    This class wraps spatial firing rate maps for a population of neurons
    with shared occupancy and environment. It inherits from `SpatialResultMixin`
    for common methods and provides iteration over individual neuron results.

    Parameters
    ----------
    firing_rates : ArrayLike
        Firing rate maps in Hz. Shape is (n_neurons, n_bins) where n_bins
        is the number of active bins in the environment. Each row is a
        single neuron's rate map.
    occupancy : ArrayLike
        Time spent in each bin in seconds. Shape is (n_bins,). Shared
        across all neurons.
    env : Environment
        The spatial environment used for the computation.
    smoothing_method : str
        Smoothing method used: "diffusion_kde", "gaussian_kde", or "binned".
    bandwidth : float
        Smoothing bandwidth in the same units as the environment's bin_size.

    Attributes
    ----------
    firing_rates : ArrayLike
        Firing rate maps in Hz. Shape is (n_neurons, n_bins).
    occupancy : ArrayLike
        Time spent in each bin in seconds. Shape is (n_bins,).
    env : Environment
        The spatial environment.
    smoothing_method : str
        Smoothing method used.
    bandwidth : float
        Smoothing bandwidth.

    Notes
    -----
    This is a frozen dataclass (immutable). All fields are set at construction
    and cannot be modified afterward.

    **Iteration Support**:

    This class supports len(), indexing, and iteration:

    - `len(result)`: Number of neurons
    - `result[i]`: Returns `SpatialRateResult` for neuron i
    - `for r in result`: Iterates over single-neuron results

    **Inherited Methods from SpatialResultMixin**:

    - `peak_locations()`: Returns (n_neurons, n_dims) coordinates of peaks
    - `peak_firing_rates()`: Returns (n_neurons,) max firing rates

    Additional batch methods (grid_scores, border_scores, classify, etc.)
    are implemented in Tasks 2.4-2.6.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding.spatial import SpatialRatesResult

    >>> # Create environment
    >>> positions = np.random.rand(100, 2) * 50
    >>> env = Environment.from_samples(positions, bin_size=5.0)

    >>> # Create batch result (5 neurons)
    >>> n_neurons = 5
    >>> firing_rates = np.random.rand(n_neurons, env.n_bins) * 10
    >>> occupancy = np.ones(env.n_bins)
    >>> result = SpatialRatesResult(
    ...     firing_rates=firing_rates,
    ...     occupancy=occupancy,
    ...     env=env,
    ...     smoothing_method="diffusion_kde",
    ...     bandwidth=5.0,
    ... )

    >>> # Access fields
    >>> result.firing_rates.shape
    (5, n_bins)
    >>> len(result)
    5

    >>> # Index to get single-neuron result
    >>> single = result[0]
    >>> type(single).__name__
    'SpatialRateResult'

    >>> # Iterate over neurons
    >>> for i, r in enumerate(result):
    ...     print(f"Neuron {i}: peak rate = {r.peak_firing_rates():.2f} Hz")

    >>> # Use mixin methods (batch)
    >>> peak_coords = result.peak_locations()  # (n_neurons, n_dims)
    >>> max_rates = result.peak_firing_rates()  # (n_neurons,)

    See Also
    --------
    SpatialRateResult : Single-neuron version
    compute_spatial_rates : Function to compute this result
    SpatialResultMixin : Provides peak_locations() and peak_firing_rates()
    """

    firing_rates: ArrayLike
    occupancy: ArrayLike
    env: Environment
    smoothing_method: str
    bandwidth: float

    def __len__(self) -> int:
        """Return number of neurons.

        Returns
        -------
        int
            Number of neurons (first dimension of firing_rates).
        """
        rates: NDArray[np.float64] = np.asarray(self.firing_rates)
        return int(rates.shape[0])

    def __getitem__(self, idx: int) -> SpatialRateResult:
        """Get single-neuron result by index.

        Parameters
        ----------
        idx : int
            Neuron index.

        Returns
        -------
        SpatialRateResult
            Result for the specified neuron with shared occupancy,
            environment, and smoothing parameters.

        Examples
        --------
        >>> single = result[0]
        >>> single.firing_rate.shape
        (n_bins,)
        """
        rates: NDArray[np.float64] = np.asarray(self.firing_rates)
        return SpatialRateResult(
            firing_rate=rates[idx],
            occupancy=self.occupancy,
            env=self.env,
            smoothing_method=self.smoothing_method,
            bandwidth=self.bandwidth,
        )

    def __iter__(self) -> Iterator[SpatialRateResult]:
        """Iterate over single-neuron results.

        Yields
        ------
        SpatialRateResult
            Result for each neuron in order.

        Examples
        --------
        >>> for result in results:
        ...     print(result.peak_firing_rates())
        """
        for i in range(len(self)):
            yield self[i]

    def plot(self, idx: int, ax: Axes | None = None, **kwargs: Any) -> Axes:
        """Plot the spatial rate map for a specific neuron.

        Delegates to the environment's plot_field method for consistent
        visualization across the codebase.

        Parameters
        ----------
        idx : int
            Index of the neuron to plot (0-indexed).
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates a new figure and axes.
        **kwargs
            Additional keyword arguments passed to env.plot_field().
            Common options include:
            - cmap : str or Colormap, default="viridis"
            - vmin, vmax : float, colorbar limits
            - add_colorbar : bool, default=True

        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the plot.

        Examples
        --------
        >>> result = SpatialRatesResult(...)
        >>> ax = result.plot(idx=0)
        >>> plt.show()

        >>> fig, ax = plt.subplots()
        >>> result.plot(idx=2, ax=ax, cmap="hot", vmax=20.0)
        """
        rates: NDArray[np.float64] = np.asarray(self.firing_rates)
        return self.env.plot_field(_to_numpy(rates[idx]), ax=ax, **kwargs)

    def spatial_information(self) -> NDArray[np.float64]:
        """Skaggs spatial information (bits per spike) for all neurons.

        Quantifies how much information each spike conveys about the
        animal's spatial location. Higher values indicate more spatially
        selective firing.

        Returns
        -------
        ndarray, shape (n_neurons,)
            Spatial information in bits/spike for each neuron.
            Always non-negative. Returns 0.0 for uniform firing.

        Notes
        -----
        Uses the Skaggs et al. (1993) formula:

        .. math::

            I = \\sum_i p_i \\frac{r_i}{\\bar{r}} \\log_2 \\left( \\frac{r_i}{\\bar{r}} \\right)

        **Interpretation**:

        - Place cells typically have 1-3 bits/spike
        - Higher values indicate more spatially selective firing
        - Zero means uniform firing (no spatial selectivity)

        References
        ----------
        .. [1] Skaggs, W. E., McNaughton, B. L., & Gothard, K. M. (1993).
               An information-theoretic approach to deciphering the hippocampal code.

        Examples
        --------
        >>> result = SpatialRatesResult(...)
        >>> info = result.spatial_information()
        >>> print(f"Mean spatial information: {info.mean():.2f} bits/spike")
        """
        from neurospatial.encoding._metrics import batch_spatial_information

        return batch_spatial_information(
            _to_numpy(self.firing_rates), _to_numpy(self.occupancy)
        )

    def sparsity(self) -> NDArray[np.float64]:
        """Sparsity of spatial firing for all neurons.

        Measures what fraction of the environment elicits significant
        firing. Lower values indicate sparser, more selective place fields.

        Returns
        -------
        ndarray, shape (n_neurons,)
            Sparsity values in range [0, 1] for each neuron.
            - Low (0.1-0.3): Sparse, selective place field
            - High (~1.0): Uniform firing throughout environment

        Notes
        -----
        Uses the Skaggs et al. (1996) formula:

        .. math::

            S = \\frac{\\left( \\sum_i p_i r_i \\right)^2}{\\sum_i p_i r_i^2}

        References
        ----------
        .. [1] Skaggs, W. E., McNaughton, B. L., et al. (1996). Theta phase
               precession in hippocampal neuronal populations.

        Examples
        --------
        >>> result = SpatialRatesResult(...)
        >>> spars = result.sparsity()
        >>> print(f"Mean sparsity: {spars.mean():.2f}")
        """
        from neurospatial.encoding._metrics import batch_sparsity

        return batch_sparsity(_to_numpy(self.firing_rates), _to_numpy(self.occupancy))

    def grid_scores(self) -> NDArray[np.float64]:
        """Grid scores (hexagonal periodicity) for all neurons.

        Quantifies the hexagonal periodicity of each neuron's firing rate map,
        which is characteristic of grid cells. Higher values indicate stronger
        hexagonal grid patterns.

        Returns
        -------
        ndarray, shape (n_neurons,)
            Grid scores in range [-2, 2] for each neuron.
            - score > 0.4: Strong hexagonal grid (typical threshold)
            - score ≈ 0: No hexagonal structure
            - score < 0: Anti-hexagonal structure (rare)
            Returns NaN for neurons where grid score cannot be computed.

        Notes
        -----
        For each neuron, computes the spatial autocorrelation and extracts
        the grid score based on rotational symmetry.

        Uses the Sargolini et al. (2006) algorithm:

        1. Compute 2D spatial autocorrelation via FFT
        2. Rotate by 30°, 60°, 90°, 120°, 150°
        3. Grid score = min(r60, r120) - max(r30, r90, r150)

        Delegates to ``neurospatial.encoding._metrics.batch_grid_scores()``.

        References
        ----------
        .. [1] Sargolini, F., Fyhn, M., et al. (2006). Conjunctive
               representation of position, direction, and velocity in
               entorhinal cortex. Science, 312(5774), 758-762.

        See Also
        --------
        SpatialRateResult.grid_score : Single-neuron grid score
        SpatialRateResult.grid_properties : Full grid cell metrics

        Examples
        --------
        >>> result = SpatialRatesResult(...)
        >>> scores = result.grid_scores()
        >>> print(f"Mean grid score: {np.nanmean(scores):.3f}")
        >>> n_grid_cells = np.sum(scores > 0.4)
        """
        from neurospatial.encoding._metrics import batch_grid_scores

        return batch_grid_scores(self.env, _to_numpy(self.firing_rates))

    def border_scores(
        self,
        threshold: float = 0.3,
        min_area: float = 0.0,
        distance_metric: Literal["geodesic", "euclidean"] = "geodesic",
    ) -> NDArray[np.float64]:
        """Border scores (boundary proximity tuning) for all neurons.

        Quantifies how much each neuron's firing field is aligned with
        environmental boundaries (walls). Higher values indicate stronger
        border cell properties.

        Parameters
        ----------
        threshold : float, default 0.3
            Fraction of peak firing rate used to segment the field.
            Bins with firing rate >= threshold * peak are included in field.
        min_area : float, default 0.0
            Minimum field area in physical units (e.g., cm²). Fields smaller
            than this return NaN. Default 0.0 (no filtering).
        distance_metric : {'geodesic', 'euclidean'}, default 'geodesic'
            Distance metric for computing distance from field to boundaries.
            - 'geodesic': Graph shortest path distance (respects obstacles)
            - 'euclidean': Straight-line distance in physical space

        Returns
        -------
        ndarray, shape (n_neurons,)
            Border scores in range [-1, 1] for each neuron.
            - +1: Perfect border cell (field on boundary)
            - 0: No boundary preference
            - -1: Anti-border (field in center)
            Returns NaN for neurons where border score cannot be computed.

        Notes
        -----
        Uses the Solstad et al. (2008) algorithm for each neuron.

        Delegates to ``neurospatial.encoding._metrics.batch_border_scores()``.

        References
        ----------
        .. [1] Solstad, T., Boccara, C. N., et al. (2008). Representation of
               geometric borders in the entorhinal cortex. Science, 322(5909),
               1865-1868.

        See Also
        --------
        SpatialRateResult.border_score : Single-neuron border score

        Examples
        --------
        >>> result = SpatialRatesResult(...)
        >>> scores = result.border_scores()
        >>> print(f"Mean border score: {np.nanmean(scores):.3f}")
        >>> n_border_cells = np.sum(scores > 0.5)
        """
        from neurospatial.encoding._metrics import batch_border_scores

        return batch_border_scores(
            self.env,
            _to_numpy(self.firing_rates),
            threshold=threshold,
            min_area=min_area,
            distance_metric=distance_metric,
        )

    def classify(
        self,
        min_spatial_info: float = 0.5,
        min_grid_score: float = 0.4,
        min_border_score: float = 0.5,
    ) -> NDArray[np.str_]:
        """Classify neurons into spatial cell types.

        Applies threshold-based classification to label neurons as place cells,
        grid cells, border cells, or unclassified based on their spatial
        information, grid score, and border score.

        Parameters
        ----------
        min_spatial_info : float, default 0.5
            Minimum spatial information (bits/spike) to be classified as a
            spatially tuned cell. Neurons below this are labeled "unclassified".
        min_grid_score : float, default 0.4
            Minimum grid score to be classified as a grid cell. Standard
            threshold from Sargolini et al. (2006).
        min_border_score : float, default 0.5
            Minimum border score to be classified as a border cell. Standard
            threshold from Solstad et al. (2008).

        Returns
        -------
        ndarray, shape (n_neurons,)
            String labels for each neuron. One of:
            - "grid": Grid cell (high grid score, passes spatial info threshold)
            - "border": Border cell (high border score, passes spatial info threshold)
            - "place": Place cell (high spatial info, not grid or border)
            - "unclassified": Does not meet criteria for any spatial cell type

        Notes
        -----
        **Classification priority** (higher takes precedence):

        1. **Grid cell**: grid_score >= min_grid_score
        2. **Border cell**: border_score >= min_border_score
        3. **Place cell**: spatial_info >= min_spatial_info (and not grid/border)
        4. **Unclassified**: Does not meet any criteria

        **Typical thresholds** (from literature):

        - Spatial information: 0.5-1.0 bits/spike (varies by study)
        - Grid score: 0.4-0.5 (Sargolini et al., 2006)
        - Border score: 0.5-0.6 (Solstad et al., 2008)

        References
        ----------
        .. [1] Sargolini, F., et al. (2006). Science, 312(5774), 758-762.
        .. [2] Solstad, T., et al. (2008). Science, 322(5909), 1865-1868.
        .. [3] Skaggs, W. E., et al. (1993). NIPS, 5, 1030-1037.

        See Also
        --------
        spatial_information : Compute spatial information
        grid_scores : Compute grid scores
        border_scores : Compute border scores

        Examples
        --------
        >>> result = SpatialRatesResult(...)
        >>> labels = result.classify()
        >>> print(f"Grid cells: {np.sum(labels == 'grid')}")
        >>> print(f"Border cells: {np.sum(labels == 'border')}")
        >>> print(f"Place cells: {np.sum(labels == 'place')}")
        >>> print(f"Unclassified: {np.sum(labels == 'unclassified')}")
        """
        n_neurons = len(self)
        labels = np.empty(n_neurons, dtype="<U14")  # Max length: "unclassified"

        # Compute all metrics
        spatial_info = self.spatial_information()
        grid_scores = self.grid_scores()
        border_scores = self.border_scores()

        # Apply classification with priority: grid > border > place > unclassified
        for i in range(n_neurons):
            if not np.isnan(grid_scores[i]) and grid_scores[i] >= min_grid_score:
                labels[i] = "grid"
            elif (
                not np.isnan(border_scores[i]) and border_scores[i] >= min_border_score
            ):
                labels[i] = "border"
            elif spatial_info[i] >= min_spatial_info:
                labels[i] = "place"
            else:
                labels[i] = "unclassified"

        return labels

    def to_dataframe(
        self,
        neuron_ids: Sequence[str] | None = None,
        include_classification: bool = True,
    ) -> pd.DataFrame:
        """Export metrics to DataFrame for exploratory analysis.

        Computes all spatial metrics and exports them to a pandas DataFrame
        for easy filtering, sorting, and analysis. This is a host-only method;
        all metrics are computed as NumPy arrays (not JAX).

        Parameters
        ----------
        neuron_ids : sequence of str, optional
            Identifiers for each neuron. If None, uses integer indices
            (0, 1, 2, ..., n_neurons-1).
        include_classification : bool, default True
            Whether to include the cell_type column with classification
            labels from ``classify()``.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:

            - neuron_id: identifier for each neuron
            - peak_x: x-coordinate of peak firing location
            - peak_y: y-coordinate of peak firing location (NaN for 1D)
            - peak_rate: maximum firing rate (Hz)
            - spatial_info: spatial information (bits/spike)
            - sparsity: sparsity measure (0-1)
            - grid_score: grid score (hexagonal periodicity)
            - border_score: border score (boundary proximity tuning)
            - cell_type: classification label (if include_classification=True)

        Notes
        -----
        This method computes all metrics at once, which may be slow for
        large populations. For selective metric computation, use the
        individual methods (``spatial_information()``, ``grid_scores()``, etc.).

        **Common pandas workflows**:

        - Filter: ``df[df["cell_type"] == "place"]``
        - Sort: ``df.sort_values("spatial_info", ascending=False)``
        - Top-N: ``df.nlargest(10, "peak_rate")``

        Examples
        --------
        >>> result = SpatialRatesResult(...)
        >>> df = result.to_dataframe()
        >>> print(df.head())
           neuron_id  peak_x  peak_y  peak_rate  spatial_info  ...

        >>> # Filter for place cells
        >>> place_cells = df[df["cell_type"] == "place"]
        >>> print(f"Found {len(place_cells)} place cells")

        >>> # Sort by spatial information
        >>> top_cells = df.sort_values("spatial_info", ascending=False).head(10)

        >>> # Custom neuron identifiers
        >>> df = result.to_dataframe(neuron_ids=["unit_0", "unit_1", "unit_2"])

        See Also
        --------
        classify : Cell type classification
        spatial_information : Batch spatial information computation
        grid_scores : Batch grid score computation
        border_scores : Batch border score computation
        """
        import pandas as pd

        n_neurons = len(self)

        # Use integer indices if no neuron_ids provided
        if neuron_ids is None:
            neuron_ids_list: list[str | int] = list(range(n_neurons))
        else:
            neuron_ids_list = list(neuron_ids)
            if len(neuron_ids_list) != n_neurons:
                raise ValueError(
                    f"neuron_ids has {len(neuron_ids_list)} elements but "
                    f"result contains {n_neurons} neurons"
                )

        # Compute peak locations
        peaks = self.peak_locations()
        n_dims = peaks.shape[1] if peaks.ndim > 1 else 1

        # Build data dictionary
        data: dict[str, Any] = {
            "neuron_id": neuron_ids_list,
            "peak_x": peaks[:, 0],
            "peak_y": peaks[:, 1] if n_dims > 1 else np.full(n_neurons, np.nan),
            "peak_rate": self.peak_firing_rates(),
            "spatial_info": self.spatial_information(),
            "sparsity": self.sparsity(),
            "grid_score": self.grid_scores(),
            "border_score": self.border_scores(),
        }

        if include_classification:
            data["cell_type"] = self.classify()

        return pd.DataFrame(data)


# ==============================================================================
# Compute Functions
# ==============================================================================


def compute_spatial_rate(
    env: Environment,
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    smoothing_method: Literal[
        "diffusion_kde", "gaussian_kde", "binned"
    ] = "diffusion_kde",
    bandwidth: float = 5.0,
    min_occupancy: float = 0.0,
    backend: Literal["numpy", "jax", "auto"] = "numpy",
) -> SpatialRateResult:
    """Compute spatial firing rate map for one neuron.

    This function computes a smoothed firing rate map from spike times
    and trajectory data. The result is a SpatialRateResult object containing
    the firing rate map, occupancy, and metadata.

    Parameters
    ----------
    env : Environment
        The spatial environment defining the bin structure. Must be fitted
        (e.g., created via ``Environment.from_samples()``).
    spike_times : ndarray, shape (n_spikes,)
        Times of spike events in seconds. Can be empty.
    times : ndarray, shape (n_samples,)
        Timestamps of trajectory samples in seconds.
    positions : ndarray, shape (n_samples, n_dims)
        Position coordinates at each time sample.
    smoothing_method : {"diffusion_kde", "gaussian_kde", "binned"}, default="diffusion_kde"
        Smoothing method to use:

        - **diffusion_kde** (recommended): Graph-based boundary-aware KDE.
          Respects environment boundaries (walls, obstacles). Uses diffusion
          kernel computed from environment graph.
        - **gaussian_kde**: Standard Euclidean KDE. Uses Gaussian kernel based
          on Euclidean distance between bin centers. Ignores boundaries (mass
          can "bleed through" walls).
        - **binned**: Legacy method. Computes raw rate first, then smooths.
          Can introduce discretization artifacts.

    bandwidth : float, default=5.0
        Smoothing bandwidth in the same units as bin_size. Larger values
        produce more smoothing.
    min_occupancy : float, default=0.0
        Minimum occupancy (seconds) for a bin to be included. Bins with
        occupancy below this threshold are set to NaN.
    backend : {"numpy"}, default="numpy"
        Computation backend. Currently only "numpy" is implemented.
        Reserved for future JAX/GPU support.

    Returns
    -------
    SpatialRateResult
        Result object containing:

        - ``firing_rate``: Firing rate map in Hz, shape (n_bins,)
        - ``occupancy``: Time in each bin in seconds, shape (n_bins,)
        - ``env``: The environment used
        - ``smoothing_method``: Method used for smoothing
        - ``bandwidth``: Bandwidth used for smoothing

    See Also
    --------
    compute_spatial_rates : Batch version for multiple neurons
    SpatialRateResult : Result class with convenience methods

    Notes
    -----
    The function uses the binning layer (``_binning.py``) to convert spike
    times to spike counts, then the smoothing layer (``_smoothing.py``) to
    compute the smoothed firing rate.

    **Algorithm**:

    1. Map trajectory positions to spatial bins
    2. Interpolate spike positions from trajectory using spike times
    3. Count spikes in each spatial bin
    4. Compute occupancy (time spent in each bin)
    5. Apply smoothing (method-dependent, see ``_smoothing.py``)

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding.spatial import compute_spatial_rate

    >>> # Create environment from positions
    >>> positions = np.random.rand(1000, 2) * 100
    >>> env = Environment.from_samples(positions, bin_size=5.0)

    >>> # Create trajectory and spike times
    >>> times = np.linspace(0, 10, 1000)
    >>> trajectory = np.random.rand(1000, 2) * 100
    >>> spike_times = np.array([1.0, 2.5, 4.0, 7.5, 8.2])

    >>> # Compute spatial rate
    >>> result = compute_spatial_rate(
    ...     env,
    ...     spike_times,
    ...     times,
    ...     trajectory,
    ...     smoothing_method="diffusion_kde",
    ...     bandwidth=10.0,
    ... )

    >>> # Access results
    >>> print(f"Peak rate: {result.peak_firing_rates():.2f} Hz")
    >>> print(f"Peak location: {result.peak_location()}")
    >>> print(f"Spatial information: {result.spatial_information():.2f} bits/spike")

    >>> # Plot the rate map
    >>> ax = result.plot()
    """
    from neurospatial.encoding._backend import SUPPORTED_BACKENDS, is_jax_available
    from neurospatial.encoding._binning import bin_spike_train, compute_occupancy
    from neurospatial.encoding._smoothing import smooth_rate_map

    # Validate backend
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unknown backend: {backend!r}. "
            f"Supported backends are: {', '.join(repr(b) for b in SUPPORTED_BACKENDS)}"
        )

    # For now, only numpy is implemented; jax raises NotImplementedError
    if backend == "jax":
        if not is_jax_available():
            raise ImportError(
                "JAX backend requested but JAX is not available. "
                "Install JAX or use backend='numpy'."
            )
        # JAX implementation not yet available
        raise NotImplementedError(
            "JAX backend for compute_spatial_rate is not yet implemented. "
            "Use backend='numpy' for now."
        )
    # For 'auto' and 'numpy', use numpy implementation

    # Convert inputs to arrays
    spike_times = np.asarray(spike_times, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)

    # Bin spike train into spatial bins
    spike_counts = bin_spike_train(env, spike_times, times, positions)

    # Compute occupancy
    occupancy = compute_occupancy(env, times, positions)

    # Apply smoothing to compute firing rate
    firing_rate = smooth_rate_map(
        env,
        spike_counts,
        occupancy,
        method=smoothing_method,
        bandwidth=bandwidth,
        min_occupancy=min_occupancy,
    )

    # Return result
    return SpatialRateResult(
        firing_rate=firing_rate,
        occupancy=occupancy,
        env=env,
        smoothing_method=smoothing_method,
        bandwidth=bandwidth,
    )


def compute_spatial_rates(
    env: Environment,
    spike_times: Sequence[NDArray[np.float64]] | NDArray[np.float64],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    smoothing_method: Literal[
        "diffusion_kde", "gaussian_kde", "binned"
    ] = "diffusion_kde",
    bandwidth: float = 5.0,
    min_occupancy: float = 0.0,
    n_jobs: int = 1,
    backend: Literal["numpy", "jax", "auto"] = "numpy",
) -> SpatialRatesResult:
    """Compute spatial firing rate maps for multiple neurons.

    This is the batch version of ``compute_spatial_rate()`` that efficiently
    processes multiple neurons with shared trajectory data. It precomputes
    shared quantities (occupancy, position bins, diffusion kernel) once and
    optionally parallelizes spike counting with joblib.

    Parameters
    ----------
    env : Environment
        The spatial environment defining the bin structure. Must be fitted
        (e.g., created via ``Environment.from_samples()``).
    spike_times : sequence of arrays or 2D array
        Spike times for each neuron. Accepted formats:

        - List/tuple of 1D arrays: ``[spikes_0, spikes_1, ...]`` (canonical)
        - 2D array with NaN padding: shape ``(n_neurons, max_spikes)``
        - 1D array (single neuron): wrapped in list automatically

        All formats are normalized via ``normalize_spike_times()``.
    times : ndarray, shape (n_samples,)
        Timestamps of trajectory samples in seconds.
    positions : ndarray, shape (n_samples, n_dims)
        Position coordinates at each time sample.
    smoothing_method : {"diffusion_kde", "gaussian_kde", "binned"}, default="diffusion_kde"
        Smoothing method to use. See ``compute_spatial_rate()`` for details.
    bandwidth : float, default=5.0
        Smoothing bandwidth in the same units as bin_size.
    min_occupancy : float, default=0.0
        Minimum occupancy (seconds) for a bin to be included.
    n_jobs : int, default=1
        Number of parallel jobs for spike counting. Use -1 for all CPUs.
        1 means sequential processing (no parallelization overhead).
    backend : {"numpy"}, default="numpy"
        Computation backend. Currently only "numpy" is implemented.
        Reserved for future JAX/GPU support.

    Returns
    -------
    SpatialRatesResult
        Result object containing:

        - ``firing_rates``: Firing rate maps, shape ``(n_neurons, n_bins)``
        - ``occupancy``: Time in each bin in seconds, shape ``(n_bins,)``
        - ``env``: The environment used
        - ``smoothing_method``: Method used for smoothing
        - ``bandwidth``: Bandwidth used for smoothing

        The result supports iteration: ``for single in result: ...``
        and indexing: ``single = result[0]``.

    See Also
    --------
    compute_spatial_rate : Single-neuron version
    SpatialRatesResult : Result class with batch methods

    Notes
    -----
    **Efficiency advantages over calling ``compute_spatial_rate()`` in a loop**:

    1. Occupancy is computed once and shared across all neurons
    2. Diffusion kernel (for ``diffusion_kde`` method) is computed once
    3. Position-to-bin mapping is done once
    4. Spike binning can be parallelized with joblib

    **When to use batch vs single**:

    - **Batch** (this function): Processing 3+ neurons, or any case where
      efficiency matters. The overhead of precomputing shared quantities
      is amortized over multiple neurons.
    - **Single** (``compute_spatial_rate``): Processing 1-2 neurons, or when
      you need fine-grained control over individual neurons.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding.spatial import compute_spatial_rates

    >>> # Create environment from positions
    >>> positions = np.random.rand(1000, 2) * 100
    >>> env = Environment.from_samples(positions, bin_size=5.0)

    >>> # Create trajectory
    >>> times = np.linspace(0, 10, 1000)
    >>> trajectory = np.random.rand(1000, 2) * 100

    >>> # Spike times for 3 neurons
    >>> spike_times = [
    ...     np.array([1.0, 2.5, 4.0]),  # Neuron 0
    ...     np.array([0.5, 1.5, 2.5, 3.5]),  # Neuron 1
    ...     np.array([5.0, 8.0]),  # Neuron 2
    ... ]

    >>> # Compute spatial rates for all neurons
    >>> result = compute_spatial_rates(
    ...     env,
    ...     spike_times,
    ...     times,
    ...     trajectory,
    ...     smoothing_method="diffusion_kde",
    ...     bandwidth=10.0,
    ...     n_jobs=2,  # Parallel spike binning
    ... )

    >>> # Access results
    >>> print(f"Number of neurons: {len(result)}")
    >>> print(f"Firing rates shape: {result.firing_rates.shape}")
    >>> print(f"Peak rates: {result.peak_firing_rates()}")

    >>> # Iterate over neurons
    >>> for i, single in enumerate(result):
    ...     print(f"Neuron {i}: peak = {single.peak_firing_rates():.2f} Hz")

    >>> # Get metrics for all neurons
    >>> df = result.to_dataframe()
    >>> print(df)

    >>> # Use 2D array with NaN padding
    >>> spike_times_2d = np.array(
    ...     [
    ...         [0.1, 0.5, 1.0, np.nan],
    ...         [0.2, 0.3, 0.8, 1.2],
    ...     ]
    ... )
    >>> result2 = compute_spatial_rates(env, spike_times_2d, times, trajectory)
    """
    from neurospatial.encoding._backend import SUPPORTED_BACKENDS, is_jax_available
    from neurospatial.encoding._binning import bin_spike_trains
    from neurospatial.encoding._smoothing import smooth_rate_maps_batch
    from neurospatial.encoding._spikes import normalize_spike_times

    # Validate backend
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unknown backend: {backend!r}. "
            f"Supported backends are: {', '.join(repr(b) for b in SUPPORTED_BACKENDS)}"
        )

    # For now, only numpy is implemented; jax raises NotImplementedError
    if backend == "jax":
        if not is_jax_available():
            raise ImportError(
                "JAX backend requested but JAX is not available. "
                "Install JAX or use backend='numpy'."
            )
        # JAX implementation not yet available
        raise NotImplementedError(
            "JAX backend for compute_spatial_rates is not yet implemented. "
            "Use backend='numpy' for now."
        )
    # For 'auto' and 'numpy', use numpy implementation

    # Normalize spike times to canonical list-of-arrays format
    spike_times_list = normalize_spike_times(spike_times)
    n_neurons = len(spike_times_list)

    # Convert inputs to arrays
    times = np.asarray(times, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)

    # Handle edge case: no neurons
    # Still compute occupancy from trajectory (occupancy is independent of neural data)
    if n_neurons == 0:
        from neurospatial.encoding._binning import compute_occupancy

        # Use compute_occupancy which handles 1D position reshaping
        occupancy = compute_occupancy(env, times, positions)
        return SpatialRatesResult(
            firing_rates=np.empty((0, env.n_bins), dtype=np.float64),
            occupancy=occupancy,
            env=env,
            smoothing_method=smoothing_method,
            bandwidth=bandwidth,
        )

    # Bin spike trains and compute occupancy
    # bin_spike_trains returns (spike_counts, occupancy)
    spike_counts, occupancy = bin_spike_trains(
        env,
        spike_times_list,
        times,
        positions,
        n_jobs=n_jobs,
    )

    # Apply batch smoothing to compute firing rates
    firing_rates = smooth_rate_maps_batch(
        env,
        spike_counts,
        occupancy,
        method=smoothing_method,
        bandwidth=bandwidth,
        min_occupancy=min_occupancy,
    )

    # Return result
    return SpatialRatesResult(
        firing_rates=firing_rates,
        occupancy=occupancy,
        env=env,
        smoothing_method=smoothing_method,
        bandwidth=bandwidth,
    )
