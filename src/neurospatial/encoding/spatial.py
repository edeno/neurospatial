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

Compute Functions
-----------------
compute_spatial_rate
    Compute spatial firing rate for one neuron
compute_spatial_rates
    Compute spatial firing rates for multiple neurons

Examples
--------
>>> import numpy as np
>>> from neurospatial import Environment
>>> from neurospatial.encoding.spatial import compute_spatial_rate

>>> # Create environment from a seeded trajectory
>>> rng = np.random.default_rng(0)
>>> positions = rng.uniform(0, 50, (500, 2))
>>> env = Environment.from_samples(positions, bin_size=5.0)

>>> # Compute a single-neuron spatial rate map (returns SpatialRateResult)
>>> times = np.linspace(0, 50, 500)
>>> spike_times = np.sort(rng.uniform(0, 50, 30))
>>> result = compute_spatial_rate(env, spike_times, times, positions, bandwidth=10.0)

>>> # Use inherited mixin methods
>>> peak = result.peak_location()  # (n_dims,) coordinates of peak
>>> peak_rate = result.peak_firing_rate()  # scalar max firing rate
>>> peak.shape
(2,)
"""

from __future__ import annotations

import warnings
from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurospatial._results import ResultMixin
from neurospatial.encoding._base import SpatialResultMixin, _to_numpy
from neurospatial.encoding._metrics import BatchScoresResult

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes

    from neurospatial import Environment
    from neurospatial.encoding.grid import GridProperties
    from neurospatial.environment._protocols import EnvironmentProtocol

# ruff: noqa: RUF022 - intentionally grouped by category
__all__ = [
    # Result classes
    "SpatialRateResult",
    "SpatialRatesResult",
    "PlaceFieldsResult",
    # Compute functions
    "compute_spatial_rate",
    "compute_spatial_rates",
    # Directional place fields
    "DirectionalPlaceFields",
    "compute_directional_place_fields",
    # Field detection
    "detect_place_fields",
]


@dataclass(frozen=True)
class PlaceFieldsResult(ResultMixin):
    """Result of ``detect_place_fields()``: detected fields plus exclusion metadata.

    Returned by :func:`detect_place_fields`. Distinguishes "this neuron
    has no detectable place fields" (``fields=[]``, ``excluded_reason=None``)
    from "this neuron was excluded by the interneuron-rate filter"
    (``fields=[]``, ``excluded_reason="mean_rate_above_threshold"``) so a
    population pipeline can branch without listening for warnings.

    Attributes
    ----------
    fields : list of NDArray[np.int64], length n_fields
        Each element is a 1-D array of bin indices belonging to one
        place field. Empty list if no fields were detected, or if the
        neuron was excluded by the ``max_mean_rate`` filter.
    excluded_reason : str | None
        ``None`` when the neuron passed all filters and ``fields``
        reflects the actual detection result. A non-None string when a
        filter caused detection to short-circuit. The only value used
        currently is ``"mean_rate_above_threshold"`` (putative interneuron);
        future filters (e.g. ``"all_nan_rate_map"``) may add more.
    n_excluded : int
        ``1`` if ``excluded_reason`` is set, else ``0``. Provided so
        downstream population aggregation can sum exclusions across
        neurons without parsing the reason string.

    Notes
    -----
    The result is iterable and indexable like a ``list[NDArray[np.int64]]``
    so callers that previously wrote ``for f in detect_place_fields(...)``
    or ``len(detect_place_fields(...))`` keep working.
    """

    fields: list[NDArray[np.int64]]
    excluded_reason: str | None = None
    n_excluded: int = 0

    def __len__(self) -> int:
        return len(self.fields)

    def __getitem__(self, idx: int) -> NDArray[np.int64]:
        return self.fields[idx]

    def __iter__(self) -> Iterator[NDArray[np.int64]]:
        return iter(self.fields)

    def __bool__(self) -> bool:
        # Truthy iff at least one field was detected. Matches the
        # ergonomic of `if detect_place_fields(...): ...` against the
        # old list[NDArray] return.
        return len(self.fields) > 0

    def summary(self) -> dict[str, Any]:
        """Scalar headline metrics for the detected place fields.

        Returns
        -------
        dict
            Mapping with keys ``n_fields`` (int, number of detected fields),
            ``total_bins`` (int, bins across all fields), ``n_excluded``
            (int), and ``excluded_reason`` (str or ``None``).

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.encoding.spatial import PlaceFieldsResult
        >>> result = PlaceFieldsResult(fields=[np.array([0, 1, 2])])
        >>> result.summary()["n_fields"]
        1
        >>> result.summary()["total_bins"]
        3
        """
        total_bins = int(sum(len(f) for f in self.fields))
        return {
            "n_fields": len(self.fields),
            "total_bins": total_bins,
            "n_excluded": int(self.n_excluded),
            "excluded_reason": self.excluded_reason,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Tidy/long-form table of field membership: one row per (field, bin).

        Each detected place field contributes one row per member bin, with a
        ``field`` index column. Neurons excluded by a filter (``fields=[]``)
        yield an empty table; ``excluded_reason`` is exposed via
        :meth:`summary`.

        Returns
        -------
        pandas.DataFrame
            Long-form table with columns ``field`` (int, field index) and
            ``bin`` (int, member bin index). Empty (with those columns) when
            no fields were detected.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.encoding.spatial import PlaceFieldsResult
        >>> result = PlaceFieldsResult(fields=[np.array([0, 1]), np.array([5])])
        >>> df = result.to_dataframe()
        >>> df["field"].tolist()
        [0, 0, 1]
        >>> df["bin"].tolist()
        [0, 1, 5]
        """
        import pandas as pd

        field_col: list[int] = []
        bin_col: list[int] = []
        for field_idx, field_bins in enumerate(self.fields):
            for b in np.asarray(field_bins).ravel():
                field_col.append(field_idx)
                bin_col.append(int(b))
        return pd.DataFrame(
            {
                "field": np.asarray(field_col, dtype=np.int64),
                "bin": np.asarray(bin_col, dtype=np.int64),
            }
        )


@dataclass(frozen=True)
class SpatialRateResult(SpatialResultMixin):
    """Result of spatial rate computation for a single neuron.

    This class wraps a spatial firing rate map with its associated metadata
    (occupancy, environment, smoothing parameters). It inherits from
    `SpatialResultMixin` for common methods like `peak_location()` and
    `peak_firing_rate()`.

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

    - `peak_location()`: Returns (n_dims,) coordinates of peak firing
    - `peak_firing_rate()`: Returns scalar max firing rate

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding.spatial import compute_spatial_rate

    >>> # Create a simple environment from a seeded trajectory
    >>> rng = np.random.default_rng(0)
    >>> positions = rng.uniform(0, 50, (500, 2))
    >>> env = Environment.from_samples(positions, bin_size=5.0)

    >>> # Compute result (returns SpatialRateResult)
    >>> times = np.linspace(0, 50, 500)
    >>> spike_times = np.sort(rng.uniform(0, 50, 30))
    >>> result = compute_spatial_rate(
    ...     env, spike_times, times, positions, bandwidth=10.0
    ... )

    >>> # Access fields
    >>> result.firing_rate.shape == (env.n_bins,)
    True
    >>> result.smoothing_method
    'diffusion_kde'

    >>> # Use mixin methods
    >>> peak_coords = result.peak_location()  # (n_dims,)
    >>> max_rate = result.peak_firing_rate()  # float

    See Also
    --------
    SpatialRatesResult : Batch version for multiple neurons
    compute_spatial_rate : Function to compute this result
    SpatialResultMixin : Provides peak_location() and peak_firing_rate()
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
        >>> import matplotlib
        >>> matplotlib.use("Agg")  # non-interactive backend for doctest
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.encoding.spatial import compute_spatial_rate
        >>> rng = np.random.default_rng(0)
        >>> positions = rng.uniform(0, 50, (500, 2))
        >>> env = Environment.from_samples(positions, bin_size=5.0)
        >>> times = np.linspace(0, 50, 500)
        >>> spike_times = np.sort(rng.uniform(0, 50, 30))
        >>> result = compute_spatial_rate(
        ...     env, spike_times, times, positions, bandwidth=10.0
        ... )
        >>> ax = result.plot()
        >>> type(ax).__name__
        'Axes'
        """
        return self.env.plot_field(_to_numpy(self.firing_rate), ax=ax, **kwargs)

    def spatial_information(self) -> float | Any:
        """Skaggs spatial information (bits per spike).

        Quantifies how much information each spike conveys about the
        animal's spatial location. Higher values indicate more spatially
        selective firing.

        Returns
        -------
        float | jax.Array
            Spatial information in bits/spike. Always non-negative.
            Returns 0.0 for uniform firing.

            **Backend-aware**: Returns float for NumPy input,
            JAX scalar for JAX input.

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
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.encoding.spatial import compute_spatial_rate
        >>> rng = np.random.default_rng(0)
        >>> positions = rng.uniform(0, 50, (500, 2))
        >>> env = Environment.from_samples(positions, bin_size=5.0)
        >>> times = np.linspace(0, 50, 500)
        >>> spike_times = np.sort(rng.uniform(0, 50, 30))
        >>> result = compute_spatial_rate(
        ...     env, spike_times, times, positions, bandwidth=10.0
        ... )
        >>> info = result.spatial_information()
        >>> bool(info >= 0.0)
        True

        See Also
        --------
        neurospatial.encoding._metrics.spatial_information : Underlying computation
        """
        from neurospatial.encoding._metrics import spatial_information

        # Pass arrays directly - _metrics.py handles JAX dispatch
        return spatial_information(self.firing_rate, self.occupancy)

    def sparsity(self) -> float | Any:
        """Sparsity of spatial firing.

        Measures what fraction of the environment elicits significant
        firing. Lower values indicate sparser, more selective place fields.

        Returns
        -------
        float | jax.Array
            Sparsity value in range [0, 1].
            - Low (0.1-0.3): Sparse, selective place field
            - High (~1.0): Uniform firing throughout environment

            **Backend-aware**: Returns float for NumPy input,
            JAX scalar for JAX input.

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
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.encoding.spatial import compute_spatial_rate
        >>> rng = np.random.default_rng(0)
        >>> positions = rng.uniform(0, 50, (500, 2))
        >>> env = Environment.from_samples(positions, bin_size=5.0)
        >>> times = np.linspace(0, 50, 500)
        >>> spike_times = np.sort(rng.uniform(0, 50, 30))
        >>> result = compute_spatial_rate(
        ...     env, spike_times, times, positions, bandwidth=10.0
        ... )
        >>> spars = result.sparsity()
        >>> bool(0.0 <= spars <= 1.0)
        True

        See Also
        --------
        neurospatial.encoding._metrics.sparsity : Underlying computation
        """
        from neurospatial.encoding._metrics import sparsity

        # Pass arrays directly - _metrics.py handles JAX dispatch
        return sparsity(self.firing_rate, self.occupancy)

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
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.encoding.spatial import compute_spatial_rate
        >>> rng = np.random.default_rng(0)
        >>> positions = rng.uniform(0, 50, (500, 2))
        >>> env = Environment.from_samples(positions, bin_size=5.0)
        >>> times = np.linspace(0, 50, 500)
        >>> spike_times = np.sort(rng.uniform(0, 50, 30))
        >>> result = compute_spatial_rate(
        ...     env, spike_times, times, positions, bandwidth=10.0
        ... )
        >>> score = result.grid_score()
        >>> bool(-2.0 <= score <= 2.0)
        True
        """
        from neurospatial.encoding.grid import grid_score as gs_func
        from neurospatial.encoding.grid import spatial_autocorrelation

        firing_rate = _to_numpy(self.firing_rate)

        try:
            autocorr = spatial_autocorrelation(self.env, firing_rate)
            return gs_func(autocorr)
        except (ValueError, RuntimeError):
            # Irregular env, constant firing, or all-NaN: grid_score is
            # undefined. Return NaN; callers using batch_grid_scores can
            # see the same NaN with the failures mask separating
            # legitimate-NaN from caught failures.
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
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.encoding.spatial import compute_spatial_rate
        >>> rng = np.random.default_rng(0)
        >>> positions = rng.uniform(0, 50, (500, 2))
        >>> env = Environment.from_samples(positions, bin_size=5.0)
        >>> times = np.linspace(0, 50, 500)
        >>> spike_times = np.sort(rng.uniform(0, 50, 30))
        >>> result = compute_spatial_rate(
        ...     env, spike_times, times, positions, bandwidth=10.0
        ... )
        >>> props = result.grid_properties()
        >>> type(props).__name__
        'GridProperties'
        """
        from neurospatial.encoding.grid import grid_properties as gp_func
        from neurospatial.encoding.grid import spatial_autocorrelation

        firing_rate = _to_numpy(self.firing_rate)
        autocorr = spatial_autocorrelation(self.env, firing_rate)
        # Use minimum bin size for grid properties (typically same for isotropic grids)
        bin_size = float(np.min(self.env.bin_sizes))
        return gp_func(autocorr, bin_size=bin_size)

    def border_score(
        self,
        threshold: float = 0.3,
        min_area: float = 0.0,
        metric: Literal["geodesic", "euclidean"] = "geodesic",
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
        metric : {'geodesic', 'euclidean'}, default 'geodesic'
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
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.encoding.spatial import compute_spatial_rate
        >>> rng = np.random.default_rng(0)
        >>> positions = rng.uniform(0, 50, (500, 2))
        >>> env = Environment.from_samples(positions, bin_size=5.0)
        >>> times = np.linspace(0, 50, 500)
        >>> spike_times = np.sort(rng.uniform(0, 50, 30))
        >>> result = compute_spatial_rate(
        ...     env, spike_times, times, positions, bandwidth=10.0
        ... )
        >>> score = result.border_score()
        >>> bool(-1.0 <= score <= 1.0)
        True
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
            metric=metric,
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
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.encoding.spatial import compute_spatial_rate
        >>> rng = np.random.default_rng(0)
        >>> positions = rng.uniform(0, 50, (500, 2))
        >>> env = Environment.from_samples(positions, bin_size=5.0)
        >>> times = np.linspace(0, 50, 500)
        >>> spike_times = np.sort(rng.uniform(0, 50, 30))
        >>> result = compute_spatial_rate(
        ...     env, spike_times, times, positions, bandwidth=10.0
        ... )
        >>> coverage = result.region_coverage()  # no regions defined -> {}
        >>> isinstance(coverage, dict)
        True
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

    - `peak_location()`: Returns (n_neurons, n_dims) coordinates of peaks
    - `peak_firing_rate()`: Returns (n_neurons,) max firing rates

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding.spatial import compute_spatial_rates

    >>> # Create environment from a seeded trajectory
    >>> rng = np.random.default_rng(0)
    >>> positions = rng.uniform(0, 50, (500, 2))
    >>> env = Environment.from_samples(positions, bin_size=5.0)

    >>> # Compute batch result for 3 neurons (returns SpatialRatesResult)
    >>> times = np.linspace(0, 50, 500)
    >>> spike_times = [np.sort(rng.uniform(0, 50, n)) for n in (30, 40, 20)]
    >>> result = compute_spatial_rates(
    ...     env, spike_times, times, positions, bandwidth=10.0
    ... )

    >>> # Access fields
    >>> result.firing_rates.shape == (3, env.n_bins)
    True
    >>> len(result)
    3

    >>> # Index to get single-neuron result
    >>> single = result[0]
    >>> type(single).__name__
    'SpatialRateResult'

    >>> # Iterate over neurons
    >>> peaks = [round(float(r.peak_firing_rate()), 2) for r in result]
    >>> len(peaks)
    3

    >>> # Use mixin methods (batch)
    >>> peak_coords = result.peak_location()  # (n_neurons, n_dims)
    >>> max_rates = result.peak_firing_rate()  # (n_neurons,)
    >>> peak_coords.shape
    (3, 2)

    See Also
    --------
    SpatialRateResult : Single-neuron version
    compute_spatial_rates : Function to compute this result
    SpatialResultMixin : Provides peak_location() and peak_firing_rate()
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
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.encoding.spatial import compute_spatial_rates
        >>> rng = np.random.default_rng(0)
        >>> positions = rng.uniform(0, 50, (500, 2))
        >>> env = Environment.from_samples(positions, bin_size=5.0)
        >>> times = np.linspace(0, 50, 500)
        >>> spike_times = [np.sort(rng.uniform(0, 50, n)) for n in (30, 40, 20)]
        >>> result = compute_spatial_rates(
        ...     env, spike_times, times, positions, bandwidth=10.0
        ... )
        >>> single = result[0]
        >>> single.firing_rate.shape == (env.n_bins,)
        True
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
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.encoding.spatial import compute_spatial_rates
        >>> rng = np.random.default_rng(0)
        >>> positions = rng.uniform(0, 50, (500, 2))
        >>> env = Environment.from_samples(positions, bin_size=5.0)
        >>> times = np.linspace(0, 50, 500)
        >>> spike_times = [np.sort(rng.uniform(0, 50, n)) for n in (30, 40, 20)]
        >>> results = compute_spatial_rates(
        ...     env, spike_times, times, positions, bandwidth=10.0
        ... )
        >>> peaks = [float(r.peak_firing_rate()) for r in results]
        >>> len(peaks)
        3
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
        >>> import matplotlib
        >>> matplotlib.use("Agg")  # non-interactive backend for doctest
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.encoding.spatial import compute_spatial_rates
        >>> rng = np.random.default_rng(0)
        >>> positions = rng.uniform(0, 50, (500, 2))
        >>> env = Environment.from_samples(positions, bin_size=5.0)
        >>> times = np.linspace(0, 50, 500)
        >>> spike_times = [np.sort(rng.uniform(0, 50, n)) for n in (30, 40, 20)]
        >>> result = compute_spatial_rates(
        ...     env, spike_times, times, positions, bandwidth=10.0
        ... )
        >>> ax = result.plot(idx=0)
        >>> type(ax).__name__
        'Axes'
        """
        rates: NDArray[np.float64] = np.asarray(self.firing_rates)
        return self.env.plot_field(_to_numpy(rates[idx]), ax=ax, **kwargs)

    def to_xarray(self) -> Any:
        """Convert the firing-rate maps to an :class:`xarray.DataArray`.

        Wraps the ``(n_neurons, n_bins)`` firing-rate matrix in a labeled
        :class:`xarray.DataArray` with dims ``("neuron", "bin")``. The
        ``neuron`` coordinate is the integer neuron index and the ``bin``
        coordinate is the integer bin index.

        Returns
        -------
        xarray.DataArray
            Firing rates (Hz) with dims ``("neuron", "bin")``. The array
            ``.values`` equal :attr:`firing_rates`. ``coords["neuron"]`` is
            ``np.arange(n_neurons)`` and ``coords["bin"]`` is
            ``np.arange(n_bins)``.

        Raises
        ------
        ImportError
            If ``xarray`` is not installed. xarray is an optional dependency;
            install it with ``pip install neurospatial[xarray]`` or
            ``pip install xarray``.

        Notes
        -----
        ``xarray`` is imported lazily inside this method, so it never becomes
        an import-time dependency of ``neurospatial``.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.encoding.spatial import compute_spatial_rates
        >>> rng = np.random.default_rng(0)
        >>> positions = rng.uniform(0, 50, (500, 2))
        >>> env = Environment.from_samples(positions, bin_size=5.0)
        >>> times = np.linspace(0, 50, 500)
        >>> spike_times = [np.sort(rng.uniform(0, 50, n)) for n in (30, 40, 20)]
        >>> result = compute_spatial_rates(
        ...     env, spike_times, times, positions, bandwidth=10.0
        ... )
        >>> da = result.to_xarray()  # doctest: +SKIP
        >>> da.dims  # doctest: +SKIP
        ('neuron', 'bin')

        See Also
        --------
        spatial_information : Per-neuron Skaggs spatial information.
        """
        try:
            import xarray as xr
        except ImportError as exc:
            raise ImportError(
                "to_xarray() requires the optional 'xarray' dependency, which "
                "is not installed. Install it with "
                "'pip install neurospatial[xarray]' or 'pip install xarray'."
            ) from exc

        rates: NDArray[np.float64] = np.asarray(self.firing_rates)
        n_neurons, n_bins = rates.shape
        return xr.DataArray(
            rates,
            dims=("neuron", "bin"),
            coords={
                "neuron": np.arange(n_neurons),
                "bin": np.arange(n_bins),
            },
            name="firing_rate",
        )

    def spatial_information(self) -> NDArray[np.float64] | Any:
        """Skaggs spatial information (bits per spike) for all neurons.

        Quantifies how much information each spike conveys about the
        animal's spatial location. Higher values indicate more spatially
        selective firing.

        Returns
        -------
        ndarray | jax.Array, shape (n_neurons,)
            Spatial information in bits/spike for each neuron.
            Always non-negative. Returns 0.0 for uniform firing.

            **Backend-aware**: Returns NumPy array for NumPy input,
            JAX array for JAX input.

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
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.encoding.spatial import compute_spatial_rates
        >>> rng = np.random.default_rng(0)
        >>> positions = rng.uniform(0, 50, (500, 2))
        >>> env = Environment.from_samples(positions, bin_size=5.0)
        >>> times = np.linspace(0, 50, 500)
        >>> spike_times = [np.sort(rng.uniform(0, 50, n)) for n in (30, 40, 20)]
        >>> result = compute_spatial_rates(
        ...     env, spike_times, times, positions, bandwidth=10.0
        ... )
        >>> info = result.spatial_information()
        >>> info.shape
        (3,)

        See Also
        --------
        neurospatial.encoding._metrics.batch_spatial_information : Underlying computation
        """
        from neurospatial.encoding._metrics import batch_spatial_information

        # Pass arrays directly - _metrics.py handles JAX dispatch
        return batch_spatial_information(self.firing_rates, self.occupancy)

    def sparsity(self) -> NDArray[np.float64] | Any:
        """Sparsity of spatial firing for all neurons.

        Measures what fraction of the environment elicits significant
        firing. Lower values indicate sparser, more selective place fields.

        Returns
        -------
        ndarray | jax.Array, shape (n_neurons,)
            Sparsity values in range [0, 1] for each neuron.
            - Low (0.1-0.3): Sparse, selective place field
            - High (~1.0): Uniform firing throughout environment

            **Backend-aware**: Returns NumPy array for NumPy input,
            JAX array for JAX input.

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
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.encoding.spatial import compute_spatial_rates
        >>> rng = np.random.default_rng(0)
        >>> positions = rng.uniform(0, 50, (500, 2))
        >>> env = Environment.from_samples(positions, bin_size=5.0)
        >>> times = np.linspace(0, 50, 500)
        >>> spike_times = [np.sort(rng.uniform(0, 50, n)) for n in (30, 40, 20)]
        >>> result = compute_spatial_rates(
        ...     env, spike_times, times, positions, bandwidth=10.0
        ... )
        >>> spars = result.sparsity()
        >>> spars.shape
        (3,)

        See Also
        --------
        neurospatial.encoding._metrics.batch_sparsity : Underlying computation
        """
        from neurospatial.encoding._metrics import batch_sparsity

        # Pass arrays directly - _metrics.py handles JAX dispatch
        return batch_sparsity(self.firing_rates, self.occupancy)

    def grid_scores(self) -> BatchScoresResult:
        """Grid scores (hexagonal periodicity) for all neurons.

        Quantifies the hexagonal periodicity of each neuron's firing rate map,
        which is characteristic of grid cells. Higher values indicate stronger
        hexagonal grid patterns.

        Returns
        -------
        BatchScoresResult
            Container with ``scores`` (shape ``(n_neurons,)``, range [-2, 2])
            and ``failures`` (boolean mask, ``True`` for neurons whose grid
            score computation raised an exception that was caught and
            converted to NaN). Use ``result.scores`` for the raw array if
            your downstream code expects a plain ndarray.

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
        >>> result = SpatialRatesResult(...)  # doctest: +SKIP
        >>> scores = result.grid_scores()  # doctest: +SKIP
        >>> # The result wraps an ndarray; reach in via .scores for math
        >>> # operations that need a real array.
        >>> print(f"Mean grid score: {np.nanmean(scores.scores):.3f}")  # doctest: +SKIP
        >>> n_grid_cells = int(np.sum(scores.scores > 0.4))  # doctest: +SKIP
        """
        from neurospatial.encoding._metrics import batch_grid_scores

        return batch_grid_scores(self.env, _to_numpy(self.firing_rates))

    def border_scores(
        self,
        threshold: float = 0.3,
        min_area: float = 0.0,
        metric: Literal["geodesic", "euclidean"] = "geodesic",
    ) -> BatchScoresResult:
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
        metric : {'geodesic', 'euclidean'}, default 'geodesic'
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
        >>> result = SpatialRatesResult(...)  # doctest: +SKIP
        >>> scores = result.border_scores()  # doctest: +SKIP
        >>> # The result wraps an ndarray; reach in via .scores for math
        >>> # operations that need a real array.
        >>> print(
        ...     f"Mean border score: {np.nanmean(scores.scores):.3f}"
        ... )  # doctest: +SKIP
        >>> n_border_cells = int(np.sum(scores.scores > 0.5))  # doctest: +SKIP
        """
        from neurospatial.encoding._metrics import batch_border_scores

        return batch_border_scores(
            self.env,
            _to_numpy(self.firing_rates),
            threshold=threshold,
            min_area=min_area,
            metric=metric,
        )

    def detect_cell_types(
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
        EgocentricRatesResult.detect_ovcs : Sibling batch classifier
        ViewRatesResult.detect_view_cells : Sibling batch classifier

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.encoding.spatial import compute_spatial_rates
        >>> rng = np.random.default_rng(0)
        >>> positions = rng.uniform(0, 50, (500, 2))
        >>> env = Environment.from_samples(positions, bin_size=5.0)
        >>> times = np.linspace(0, 50, 500)
        >>> spike_times = [np.sort(rng.uniform(0, 50, n)) for n in (30, 40, 20)]
        >>> result = compute_spatial_rates(
        ...     env, spike_times, times, positions, bandwidth=10.0
        ... )
        >>> labels = result.detect_cell_types()
        >>> labels.shape
        (3,)
        >>> valid = {"grid", "border", "place", "unclassified"}
        >>> set(labels.tolist()).issubset(valid)
        True
        """
        n_neurons = len(self)

        spatial_info = self.spatial_information()
        # grid_scores() / border_scores() return BatchScoresResult; pull
        # the float array out via .scores for the boolean masks below.
        grid_scores_arr = self.grid_scores().scores
        border_scores_arr = self.border_scores().scores

        labels = np.full(n_neurons, "unclassified", dtype="<U14")
        is_place = spatial_info >= min_spatial_info
        is_border = (~np.isnan(border_scores_arr)) & (
            border_scores_arr >= min_border_score
        )
        is_grid = (~np.isnan(grid_scores_arr)) & (grid_scores_arr >= min_grid_score)

        # Priority: grid > border > place > unclassified (assign in reverse so
        # higher-priority labels overwrite lower ones).
        labels[is_place] = "place"
        labels[is_border] = "border"
        labels[is_grid] = "grid"

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
            labels from ``detect_cell_types()``.

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
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.encoding.spatial import compute_spatial_rates
        >>> rng = np.random.default_rng(0)
        >>> positions = rng.uniform(0, 50, (500, 2))
        >>> env = Environment.from_samples(positions, bin_size=5.0)
        >>> times = np.linspace(0, 50, 500)
        >>> spike_times = [np.sort(rng.uniform(0, 50, n)) for n in (30, 40, 20)]
        >>> result = compute_spatial_rates(
        ...     env, spike_times, times, positions, bandwidth=10.0
        ... )
        >>> df = result.to_dataframe()
        >>> "cell_type" in df.columns
        True
        >>> len(df)
        3

        >>> # Filter for place cells
        >>> place_cells = df[df["cell_type"] == "place"]

        >>> # Sort by spatial information
        >>> top_cells = df.sort_values("spatial_info", ascending=False)

        >>> # Custom neuron identifiers
        >>> df = result.to_dataframe(neuron_ids=["unit_0", "unit_1", "unit_2"])
        >>> df["neuron_id"].tolist()
        ['unit_0', 'unit_1', 'unit_2']

        See Also
        --------
        detect_cell_types : Cell type classification
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
        peaks = self.peak_location()
        n_dims = peaks.shape[1] if peaks.ndim > 1 else 1

        # Build data dictionary
        data: dict[str, Any] = {
            "neuron_id": neuron_ids_list,
            "peak_x": peaks[:, 0],
            "peak_y": peaks[:, 1] if n_dims > 1 else np.full(n_neurons, np.nan),
            "peak_rate": self.peak_firing_rate(),
            "spatial_info": self.spatial_information(),
            "sparsity": self.sparsity(),
            "grid_score": self.grid_scores().scores,
            "border_score": self.border_scores().scores,
        }

        if include_classification:
            data["cell_type"] = self.detect_cell_types()

        return pd.DataFrame(data)


# ==============================================================================
# Compute Functions
# ==============================================================================


def _fill_nan(rates: ArrayLike, fill_value: float) -> ArrayLike:
    """Replace NaN entries of a rate map with ``fill_value``.

    Works for both NumPy and JAX arrays. The masked/low-occupancy bins set
    to NaN by ``min_occupancy`` are the only NaN entries in a rate map, so
    this targets exactly those bins.

    Parameters
    ----------
    rates : ArrayLike
        Firing rate map (NumPy or JAX array), any shape.
    fill_value : float
        Value substituted wherever ``rates`` is NaN.

    Returns
    -------
    ArrayLike
        A new array of the same type/shape with NaN replaced by ``fill_value``.
    """
    # JAX arrays expose .at/.dtype but are not numpy ndarrays; dispatch on
    # whether the object is a numpy array. Both backends support np.isnan via
    # the array's own namespace, but jnp.where keeps the result on-device.
    if isinstance(rates, np.ndarray):
        return np.where(np.isnan(rates), fill_value, rates)
    import jax.numpy as jnp

    rates_jax = cast("Any", rates)
    return cast("ArrayLike", jnp.where(jnp.isnan(rates_jax), fill_value, rates_jax))


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
    fill_value: float | None = None,
    backend: Literal["numpy", "jax", "auto"] = "numpy",
    warn_on_drop: bool = True,
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
        Position coordinates at each time sample. NaN values are treated as
        missing data and excluded from occupancy and firing-rate computation;
        callers do not need to pre-filter tracking dropouts.
    smoothing_method : {"diffusion_kde", "gaussian_kde", "binned"}, default="diffusion_kde"
        Smoothing method to use:

        - **diffusion_kde** (recommended): Graph-based boundary-aware KDE.
          Respects environment boundaries (walls, obstacles). Uses diffusion
          kernel computed from environment graph.
        - **gaussian_kde**: Standard Euclidean KDE. Uses Gaussian kernel based
          on Euclidean distance between bin centers. Ignores boundaries (mass
          can "bleed through" walls).
        - **binned**: Bin-then-smooth method. Computes raw rate first, then
          smooths. Can introduce discretization artifacts.

    bandwidth : float, default=5.0
        Smoothing bandwidth in the same units as bin_size. Larger values
        produce more smoothing.
    min_occupancy : float, default=0.0
        Minimum occupancy (seconds) for a bin to be included. Bins with
        occupancy below this threshold are set to NaN.
    fill_value : float | None, default=None
        Value used to replace NaN bins (masked/low-occupancy bins produced
        by ``min_occupancy``). When ``None`` (the default), NaN is preserved
        so existing callers see no behavior change. Pass ``fill_value=0.0``
        for the recommended decoding golden path: a zero-rate map composes
        directly with :func:`~neurospatial.decoding.posterior.decode_position`
        without manual NaN scrubbing. ``occupancy`` is unaffected, so callers
        can still recover which bins were masked via
        ``result.occupancy < min_occupancy``.
    backend : {"numpy", "jax", "auto"}, default="numpy"
        Computation backend for rate map smoothing:

        - ``"numpy"``: Use NumPy for all computations. Works everywhere.
        - ``"jax"``: Use JAX for rate computation. Requires JAX installation.
          Enables GPU acceleration and JAX transformations (jit, grad).
        - ``"auto"``: Use JAX if available, otherwise NumPy.

        Note: Binning operations (spike counting, occupancy) always use NumPy.
        Only the smoothing/rate computation uses the selected backend.
    warn_on_drop : bool, default=True
        If ``True`` (the default), emit a ``UserWarning`` when a large
        fraction of spikes are silently dropped — either because they
        fall outside the position time window or because they map to
        inactive/out-of-environment bins.  A warning is always emitted
        when **all** spikes are dropped (regardless of threshold).  This
        guards against common unit mismatches (e.g. spike_times in
        milliseconds while times is in seconds).  Set to ``False`` to
        suppress all drop-related warnings.

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

    >>> # Create environment from a seeded trajectory
    >>> rng = np.random.default_rng(0)
    >>> positions = rng.uniform(0, 100, (1000, 2))
    >>> env = Environment.from_samples(positions, bin_size=5.0)

    >>> # Create trajectory timestamps and (sorted) spike times
    >>> times = np.linspace(0, 10, 1000)
    >>> spike_times = np.array([1.0, 2.5, 4.0, 7.5, 8.2])

    >>> # Compute spatial rate
    >>> result = compute_spatial_rate(
    ...     env,
    ...     spike_times,
    ...     times,
    ...     positions,
    ...     smoothing_method="diffusion_kde",
    ...     bandwidth=10.0,
    ... )

    >>> # Access results
    >>> result.firing_rate.shape == (env.n_bins,)
    True
    >>> bool(result.peak_firing_rate() >= 0.0)
    True
    >>> result.peak_location().shape
    (2,)
    """
    from neurospatial.encoding._backend import (
        SUPPORTED_BACKENDS,
        get_backend_name,
        is_jax_available,
    )
    from neurospatial.encoding._binning import bin_spike_train, compute_occupancy
    from neurospatial.encoding._smoothing import (
        _validate_smoothing_parameters,
        smooth_rate_map,
    )
    from neurospatial.encoding._validation import (
        validate_env_fitted,
        validate_spike_times,
        validate_trajectory,
    )

    validate_env_fitted(env, context="compute_spatial_rate")

    # Validate backend
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unknown backend: {backend!r}. "
            f"Supported backends are: {', '.join(repr(b) for b in SUPPORTED_BACKENDS)}"
        )

    # Resolve backend (handles "auto" → "numpy" or "jax")
    # This raises ImportError if backend="jax" and JAX is unavailable
    resolved_backend = get_backend_name(backend)

    _validate_smoothing_parameters(smoothing_method, bandwidth)

    # Convert inputs to arrays
    spike_times = np.asarray(spike_times, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)

    validate_trajectory(times, positions=positions, context="compute_spatial_rate")
    validate_spike_times(spike_times, context="compute_spatial_rate")

    # Bin spike train into spatial bins (always NumPy - CPU/joblib)
    spike_counts = bin_spike_train(
        env,
        spike_times,
        times,
        positions,
        context="compute_spatial_rate",
        warn_on_drop=warn_on_drop,
    )

    # Compute occupancy (always NumPy)
    occupancy = compute_occupancy(env, times, positions, context="compute_spatial_rate")

    # Apply smoothing to compute firing rate
    # When backend="jax", uses JAX for the core rate computation
    firing_rate = smooth_rate_map(
        env,
        spike_counts,
        occupancy,
        method=smoothing_method,
        bandwidth=bandwidth,
        min_occupancy=min_occupancy,
        backend=resolved_backend,
    )

    # Replace masked/low-occupancy NaN bins with fill_value when requested.
    # Default (None) preserves NaN so existing callers see no behavior change.
    if fill_value is not None:
        firing_rate = _fill_nan(firing_rate, fill_value)

    # Convert occupancy to JAX if JAX backend is selected
    # (firing_rate is already JAX from smooth_rate_map)
    if resolved_backend == "jax" and is_jax_available():
        import jax.numpy as jnp

        occupancy = jnp.asarray(occupancy, dtype=jnp.float64)  # type: ignore[assignment]

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
    fill_value: float | None = None,
    n_jobs: int = 1,
    backend: Literal["numpy", "jax", "auto"] = "numpy",
    warn_on_drop: bool = True,
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
        Position coordinates at each time sample. NaN values are treated as
        missing data and excluded from occupancy and firing-rate computation;
        callers do not need to pre-filter tracking dropouts.
    smoothing_method : {"diffusion_kde", "gaussian_kde", "binned"}, default="diffusion_kde"
        Smoothing method to use. See ``compute_spatial_rate()`` for details.
    bandwidth : float, default=5.0
        Smoothing bandwidth in the same units as bin_size.
    min_occupancy : float, default=0.0
        Minimum occupancy (seconds) for a bin to be included.
    fill_value : float | None, default=None
        Value used to replace NaN bins (masked/low-occupancy bins produced
        by ``min_occupancy``). When ``None`` (the default), NaN is preserved
        so existing callers see no behavior change. Pass ``fill_value=0.0``
        for the recommended decoding golden path: zero-rate maps compose
        directly with :func:`~neurospatial.decoding.posterior.decode_position`
        without manual NaN scrubbing. ``occupancy`` is unaffected, so callers
        can still recover which bins were masked via
        ``result.occupancy < min_occupancy``.
    n_jobs : int, default=1
        Number of parallel jobs for spike counting. Use -1 for all CPUs.
        1 means sequential processing (no parallelization overhead).
    backend : {"numpy", "jax", "auto"}, default="numpy"
        Computation backend for rate map smoothing:

        - ``"numpy"``: Use NumPy for all computations. Works everywhere.
        - ``"jax"``: Use JAX for rate computation. Requires JAX installation.
          Enables GPU acceleration and JAX transformations (jit, grad).
        - ``"auto"``: Use JAX if available, otherwise NumPy.

        Note: Binning operations (spike counting, occupancy) always use NumPy.
        Only the smoothing/rate computation uses the selected backend.
    warn_on_drop : bool, default=True
        If ``True`` (the default), emit a single ``UserWarning`` (per drop
        cause) when a large fraction of spikes are silently dropped across
        all neurons.  The warning is computed in the main process from
        aggregate statistics, so it fires exactly once even when
        ``n_jobs != 1`` (joblib worker warnings are commonly swallowed).
        Set to ``False`` to suppress all drop-related warnings.

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

    >>> # Create environment from a seeded trajectory
    >>> rng = np.random.default_rng(0)
    >>> positions = rng.uniform(0, 100, (1000, 2))
    >>> env = Environment.from_samples(positions, bin_size=5.0)

    >>> # Create trajectory timestamps
    >>> times = np.linspace(0, 10, 1000)

    >>> # Spike times for 3 neurons (each sorted ascending)
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
    ...     positions,
    ...     smoothing_method="diffusion_kde",
    ...     bandwidth=10.0,
    ... )

    >>> # Access results
    >>> len(result)
    3
    >>> result.firing_rates.shape == (3, env.n_bins)
    True

    >>> # Iterate over neurons
    >>> peaks = [round(float(single.peak_firing_rate()), 2) for single in result]
    >>> len(peaks)
    3

    >>> # Get metrics for all neurons
    >>> df = result.to_dataframe()
    >>> len(df)
    3

    >>> # Use 2D array with NaN padding
    >>> spike_times_2d = np.array(
    ...     [
    ...         [0.1, 0.5, 1.0, np.nan],
    ...         [0.2, 0.3, 0.8, 1.2],
    ...     ]
    ... )
    >>> result2 = compute_spatial_rates(env, spike_times_2d, times, positions)
    >>> len(result2)
    2
    """
    from neurospatial.encoding._backend import (
        SUPPORTED_BACKENDS,
        get_backend_name,
        is_jax_available,
    )
    from neurospatial.encoding._binning import bin_spike_trains
    from neurospatial.encoding._smoothing import (
        _validate_smoothing_parameters,
        smooth_rate_maps_batch,
    )
    from neurospatial.encoding._spikes import normalize_spike_times
    from neurospatial.encoding._validation import (
        validate_env_fitted,
        validate_spike_times,
        validate_trajectory,
    )

    validate_env_fitted(env, context="compute_spatial_rates")

    # Validate backend
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unknown backend: {backend!r}. "
            f"Supported backends are: {', '.join(repr(b) for b in SUPPORTED_BACKENDS)}"
        )

    # Resolve backend (handles "auto" → "numpy" or "jax")
    # This raises ImportError if backend="jax" and JAX is unavailable
    resolved_backend = get_backend_name(backend)

    _validate_smoothing_parameters(smoothing_method, bandwidth)

    # Normalize spike times to canonical list-of-arrays format
    spike_times_list = normalize_spike_times(spike_times)
    n_neurons = len(spike_times_list)

    # Convert inputs to arrays
    times = np.asarray(times, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)

    validate_trajectory(times, positions=positions, context="compute_spatial_rates")
    for i, st in enumerate(spike_times_list):
        validate_spike_times(st, context=f"compute_spatial_rates (neuron {i})")

    # Handle edge case: no neurons
    # Still compute occupancy from trajectory (occupancy is independent of neural data)
    if n_neurons == 0:
        from neurospatial.encoding._binning import compute_occupancy

        # Use compute_occupancy which handles 1D position reshaping
        occupancy = compute_occupancy(
            env, times, positions, context="compute_spatial_rates"
        )

        # Convert to JAX if needed
        firing_rates_result: ArrayLike = np.empty((0, env.n_bins), dtype=np.float64)
        if resolved_backend == "jax" and is_jax_available():
            import jax.numpy as jnp

            firing_rates_result = jnp.asarray(firing_rates_result, dtype=jnp.float64)
            occupancy = jnp.asarray(occupancy, dtype=jnp.float64)  # type: ignore[assignment]

        return SpatialRatesResult(
            firing_rates=firing_rates_result,
            occupancy=occupancy,
            env=env,
            smoothing_method=smoothing_method,
            bandwidth=bandwidth,
        )

    # Bin spike trains and compute occupancy (always NumPy - CPU/joblib)
    # bin_spike_trains returns (spike_counts, occupancy)
    spike_counts, occupancy = bin_spike_trains(
        env,
        spike_times_list,
        times,
        positions,
        n_jobs=n_jobs,
        warn_on_drop=warn_on_drop,
    )

    # Apply batch smoothing to compute firing rates
    # When backend="jax", uses JAX for the core rate computation
    firing_rates = smooth_rate_maps_batch(
        env,
        spike_counts,
        occupancy,
        method=smoothing_method,
        bandwidth=bandwidth,
        min_occupancy=min_occupancy,
        backend=resolved_backend,
    )

    # Replace masked/low-occupancy NaN bins with fill_value when requested.
    # Default (None) preserves NaN so existing callers see no behavior change.
    if fill_value is not None:
        firing_rates = _fill_nan(firing_rates, fill_value)

    # Convert occupancy to JAX if JAX backend is selected
    # (firing_rates is already JAX from smooth_rate_maps_batch)
    if resolved_backend == "jax" and is_jax_available():
        import jax.numpy as jnp

        occupancy = jnp.asarray(occupancy, dtype=jnp.float64)  # type: ignore[assignment]

    # Return result
    return SpatialRatesResult(
        firing_rates=firing_rates,
        occupancy=occupancy,
        env=env,
        smoothing_method=smoothing_method,
        bandwidth=bandwidth,
    )


# ==============================================================================
# Directional Place Fields
# ==============================================================================


@dataclass(frozen=True)
class DirectionalPlaceFields(ResultMixin):
    """Container for direction-conditioned place field results.

    Stores firing rate maps computed separately for different movement
    directions or trial types. This enables analysis of directional
    tuning in place cells.

    Attributes
    ----------
    firing_rates : Mapping[str, NDArray[np.float64]]
        Dictionary mapping direction labels (e.g., "A→B", "forward") to
        firing rate arrays. Each array has shape (n_bins,) matching the
        environment's bin structure.
    occupancy : Mapping[str, NDArray[np.float64]]
        Per-direction occupancy (time spent in each bin) in seconds.
        Shape (n_bins,). Same keys as ``firing_rates``.
    env : Environment
        Spatial environment used to compute the per-direction fields.
        Shared across all labels (the per-direction split is over time,
        not over space).
    labels : tuple[str, ...]
        Tuple of direction labels in iteration order. Preserves the order
        in which directions were processed, enabling reproducible iteration.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> env = Environment.from_samples(
    ...     np.linspace(0, 10, 11)[:, None], bin_size=1.0
    ... )  # doctest: +SKIP
    >>> firing_rates = {
    ...     "home→goal": np.array([1.0, 2.0, 3.0]),
    ...     "goal→home": np.array([3.0, 2.0, 1.0]),
    ... }
    >>> occupancy = {
    ...     "home→goal": np.array([1.0, 1.0, 1.0]),
    ...     "goal→home": np.array([1.0, 1.0, 1.0]),
    ... }
    >>> result = DirectionalPlaceFields(  # doctest: +SKIP
    ...     firing_rates=firing_rates,
    ...     occupancy=occupancy,
    ...     env=env,
    ...     labels=("home→goal", "goal→home"),
    ... )

    See Also
    --------
    compute_directional_place_fields : Compute directional place fields from spike data.
    """

    firing_rates: Mapping[str, NDArray[np.float64]]
    occupancy: Mapping[str, NDArray[np.float64]]
    env: Environment
    labels: tuple[str, ...]

    def correlation(self, label_a: str, label_b: str) -> float:
        """Pearson correlation between two directions' rate maps.

        Quantifies how similar the place-field map is between two movement
        directions (or trial types). A correlation near ``1.0`` means the cell
        fires in the same locations regardless of direction; values near
        ``0`` (or negative) indicate direction-specific tuning.

        Bins where either map is NaN are excluded pairwise before the
        correlation is computed.

        Parameters
        ----------
        label_a, label_b : str
            Direction labels to compare. Must be present in ``labels``.

        Returns
        -------
        float
            Pearson correlation coefficient in ``[-1, 1]``. Returns ``nan``
            if fewer than two finite overlapping bins exist or if either map
            has zero variance over the overlap.

        Raises
        ------
        KeyError
            If ``label_a`` or ``label_b`` is not a known direction label.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.encoding.spatial import DirectionalPlaceFields
        >>> env = Environment.from_samples(
        ...     np.linspace(0, 9, 100)[:, None], bin_size=1.0
        ... )
        >>> n = env.n_bins
        >>> rate = np.linspace(1.0, 5.0, n)
        >>> result = DirectionalPlaceFields(
        ...     firing_rates={"fwd": rate, "rev": rate.copy()},
        ...     occupancy={"fwd": np.ones(n), "rev": np.ones(n)},
        ...     env=env,
        ...     labels=("fwd", "rev"),
        ... )
        >>> bool(np.isclose(result.correlation("fwd", "rev"), 1.0))
        True
        """
        for label in (label_a, label_b):
            if label not in self.firing_rates:
                raise KeyError(
                    f"Unknown direction label {label!r}. "
                    f"Known labels: {tuple(self.firing_rates)}."
                )
        a = _to_numpy(self.firing_rates[label_a]).ravel()
        b = _to_numpy(self.firing_rates[label_b]).ravel()
        finite = np.isfinite(a) & np.isfinite(b)
        if int(finite.sum()) < 2:
            return float("nan")
        a, b = a[finite], b[finite]
        if a.std() == 0.0 or b.std() == 0.0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    def directionality_index(self, label_a: str, label_b: str) -> float:
        """Per-bin directionality index between two directions.

        Returns the mean absolute normalized rate difference between the two
        directions across bins:

        .. math::

            \\mathrm{DI} = \\mathrm{mean}_i
                \\frac{|r^a_i - r^b_i|}{r^a_i + r^b_i}

        Values near ``0`` indicate direction-independent firing; values near
        ``1`` indicate strongly direction-selective firing. Bins where either
        map is NaN, or where both rates are zero, are excluded.

        Parameters
        ----------
        label_a, label_b : str
            Direction labels to compare. Must be present in ``labels``.

        Returns
        -------
        float
            Mean directionality index in ``[0, 1]``. Returns ``nan`` if no
            bin has a positive summed rate.

        Raises
        ------
        KeyError
            If ``label_a`` or ``label_b`` is not a known direction label.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.encoding.spatial import DirectionalPlaceFields
        >>> env = Environment.from_samples(
        ...     np.linspace(0, 9, 100)[:, None], bin_size=1.0
        ... )
        >>> n = env.n_bins
        >>> rate = np.linspace(1.0, 5.0, n)
        >>> result = DirectionalPlaceFields(
        ...     firing_rates={"fwd": rate, "rev": rate.copy()},
        ...     occupancy={"fwd": np.ones(n), "rev": np.ones(n)},
        ...     env=env,
        ...     labels=("fwd", "rev"),
        ... )
        >>> bool(np.isclose(result.directionality_index("fwd", "rev"), 0.0))
        True
        """
        for label in (label_a, label_b):
            if label not in self.firing_rates:
                raise KeyError(
                    f"Unknown direction label {label!r}. "
                    f"Known labels: {tuple(self.firing_rates)}."
                )
        a = _to_numpy(self.firing_rates[label_a]).ravel()
        b = _to_numpy(self.firing_rates[label_b]).ravel()
        total = a + b
        valid = np.isfinite(a) & np.isfinite(b) & (total > 0.0)
        if not valid.any():
            return float("nan")
        di = np.abs(a[valid] - b[valid]) / total[valid]
        return float(np.mean(di))

    def summary(self) -> dict[str, Any]:
        """Scalar headline metrics across directions.

        Returns
        -------
        dict
            Mapping with ``n_directions`` (int), ``n_bins`` (int), and one
            ``peak_<label>`` entry per direction giving that direction's peak
            firing rate (Hz). When exactly two directions are present, a
            ``correlation`` entry (Pearson r between the two maps) is also
            included.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.encoding.spatial import DirectionalPlaceFields
        >>> env = Environment.from_samples(
        ...     np.linspace(0, 9, 100)[:, None], bin_size=1.0
        ... )
        >>> n = env.n_bins
        >>> result = DirectionalPlaceFields(
        ...     firing_rates={
        ...         "fwd": np.linspace(1.0, 5.0, n),
        ...         "rev": np.linspace(5.0, 1.0, n),
        ...     },
        ...     occupancy={"fwd": np.ones(n), "rev": np.ones(n)},
        ...     env=env,
        ...     labels=("fwd", "rev"),
        ... )
        >>> result.summary()["n_directions"]
        2
        """
        out: dict[str, Any] = {
            "n_directions": len(self.labels),
            "n_bins": int(self.env.n_bins),
        }
        for label in self.labels:
            rate = _to_numpy(self.firing_rates[label])
            out[f"peak_{label}"] = float(np.nanmax(rate))
        if len(self.labels) == 2:
            out["correlation"] = self.correlation(self.labels[0], self.labels[1])
        return out

    def to_dataframe(self) -> pd.DataFrame:
        """Tidy/long-form table: one row per (direction, bin).

        Stacks the per-direction rate maps into long form with a ``direction``
        identifier column, so directional results ``pandas.concat`` cleanly
        with other tidy result tables. Bin-center coordinates are emitted as
        ``coord_0``, ``coord_1``, ... columns.

        Returns
        -------
        pandas.DataFrame
            Long-form table with columns ``direction`` (str), ``bin`` (int),
            ``coord_0`` ... (float), ``firing_rate`` (float, Hz), and
            ``occupancy`` (float, seconds).

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.encoding.spatial import DirectionalPlaceFields
        >>> env = Environment.from_samples(
        ...     np.linspace(0, 9, 100)[:, None], bin_size=1.0
        ... )
        >>> n = env.n_bins
        >>> result = DirectionalPlaceFields(
        ...     firing_rates={
        ...         "fwd": np.linspace(1.0, 5.0, n),
        ...         "rev": np.linspace(5.0, 1.0, n),
        ...     },
        ...     occupancy={"fwd": np.ones(n), "rev": np.ones(n)},
        ...     env=env,
        ...     labels=("fwd", "rev"),
        ... )
        >>> df = result.to_dataframe()
        >>> sorted(df["direction"].unique())
        ['fwd', 'rev']
        >>> len(df) == 2 * n
        True
        """
        import pandas as pd

        bin_centers = np.asarray(self.env.bin_centers)
        if bin_centers.ndim == 1:
            bin_centers = bin_centers[:, None]
        n_bins, n_dims = bin_centers.shape

        frames: list[pd.DataFrame] = []
        for label in self.labels:
            rate = _to_numpy(self.firing_rates[label]).ravel()
            occ = _to_numpy(self.occupancy[label]).ravel()
            data: dict[str, Any] = {
                "direction": np.repeat(label, n_bins),
                "bin": np.arange(n_bins, dtype=np.int64),
            }
            for d in range(n_dims):
                data[f"coord_{d}"] = bin_centers[:, d]
            data["firing_rate"] = rate
            data["occupancy"] = occ
            frames.append(pd.DataFrame(data))
        if not frames:
            # No labelled directions (e.g. compute_directional_place_fields
            # excluded every "other" sample, leaving labels == ()). Return an
            # empty frame with the documented column schema instead of letting
            # pd.concat([]) raise "No objects to concatenate".
            empty: dict[str, NDArray[Any]] = {
                "direction": np.array([], dtype=object),
                "bin": np.array([], dtype=np.int64),
            }
            for d in range(n_dims):
                empty[f"coord_{d}"] = np.array([], dtype=np.float64)
            empty["firing_rate"] = np.array([], dtype=np.float64)
            empty["occupancy"] = np.array([], dtype=np.float64)
            return pd.DataFrame(empty)
        return pd.concat(frames, ignore_index=True)

    def plot(self, ax: Axes | None = None, **kwargs: Any) -> Axes:
        """Overlay the per-direction firing rate maps on a single axis.

        Plots each direction's firing rate against bin index as one line,
        producing one artist per direction. This is most informative for
        1-D / linearized environments where bin index maps to track position.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. If None, a new figure and axes are created.
        **kwargs
            Additional keyword arguments forwarded to ``ax.plot`` for each
            direction's line.

        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the overlay (one line per direction).

        Examples
        --------
        >>> import matplotlib
        >>> matplotlib.use("Agg")  # non-interactive backend for doctest
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.encoding.spatial import DirectionalPlaceFields
        >>> env = Environment.from_samples(
        ...     np.linspace(0, 9, 100)[:, None], bin_size=1.0
        ... )
        >>> n = env.n_bins
        >>> result = DirectionalPlaceFields(
        ...     firing_rates={
        ...         "fwd": np.linspace(1.0, 5.0, n),
        ...         "rev": np.linspace(5.0, 1.0, n),
        ...     },
        ...     occupancy={"fwd": np.ones(n), "rev": np.ones(n)},
        ...     env=env,
        ...     labels=("fwd", "rev"),
        ... )
        >>> ax = result.plot()
        >>> len(ax.get_lines())
        2
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()

        for label in self.labels:
            rate = _to_numpy(self.firing_rates[label]).ravel()
            ax.plot(np.arange(rate.shape[0]), rate, label=str(label), **kwargs)

        ax.set_xlabel("Bin index")
        ax.set_ylabel("Firing rate (Hz)")
        ax.set_title("Directional place fields")
        ax.legend()
        return ax


def _subset_spikes_by_time_mask(
    times: NDArray[np.float64],
    spike_times: NDArray[np.float64],
    mask: NDArray[np.bool_],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Subset spike times by a boolean mask over trajectory times.

    Extracts spikes that fall within the time ranges defined by contiguous
    True segments in the mask. Uses binary search (searchsorted) for
    efficient O(log n) spike slicing per segment.

    Parameters
    ----------
    times : NDArray[np.float64], shape (n_timepoints,)
        Timestamps of trajectory samples (seconds). Must be sorted.
    spike_times : NDArray[np.float64], shape (n_spikes,)
        Timestamps of spike occurrences (seconds). Must be sorted.
    mask : NDArray[np.bool_], shape (n_timepoints,)
        Boolean mask indicating which timepoints to include.
        Contiguous True segments define time ranges for spike inclusion.

    Returns
    -------
    times_sub : NDArray[np.float64]
        Subset of times where mask is True. Same as ``times[mask]``.
    spike_times_sub : NDArray[np.float64]
        Spikes that fall within the time ranges of contiguous True segments.
        Boundaries are inclusive: spikes at segment start/end are included.

    Notes
    -----
    For each contiguous segment of True values in mask:
    - ``t_start = times[segment_first_index]``
    - ``t_end = times[segment_last_index]``
    - Spikes in ``[t_start, t_end]`` (inclusive) are selected

    This function is designed for conditioning place field analysis on
    subsets of the trajectory (e.g., by movement direction, trial type).

    Examples
    --------
    >>> import numpy as np
    >>> times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    >>> spike_times = np.array([0.5, 1.5, 2.5, 3.5])
    >>> mask = np.array([False, True, True, False, False])
    >>> times_sub, spikes_sub = _subset_spikes_by_time_mask(times, spike_times, mask)
    >>> times_sub
    array([1., 2.])
    >>> spikes_sub
    array([1.5])
    """
    # Fast path: empty mask
    if not np.any(mask):
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    # Get indices where mask is True
    true_indices = np.where(mask)[0]

    # Find contiguous segments by looking for gaps > 1
    # diff > 1 indicates a break in contiguity
    if len(true_indices) == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    # Find segment boundaries: where consecutive indices are not adjacent
    breaks = np.where(np.diff(true_indices) > 1)[0] + 1
    segment_starts = np.concatenate([[0], breaks])
    segment_ends = np.concatenate([breaks, [len(true_indices)]])

    # Fast path: empty spike train
    if len(spike_times) == 0:
        return times[mask], np.array([], dtype=np.float64)

    # Collect spikes from each segment
    spike_slices = []

    for seg_start_idx, seg_end_idx in zip(segment_starts, segment_ends, strict=True):
        # Get the actual time indices for this segment
        first_time_idx = true_indices[seg_start_idx]
        last_time_idx = true_indices[seg_end_idx - 1]

        # Get time boundaries
        t_start = times[first_time_idx]
        t_end = times[last_time_idx]

        # Use searchsorted for O(log n) spike slicing
        # side="left" for t_start: include spikes at exactly t_start
        # side="right" for t_end: include spikes at exactly t_end
        spike_start = np.searchsorted(spike_times, t_start, side="left")
        spike_end = np.searchsorted(spike_times, t_end, side="right")

        if spike_start < spike_end:
            spike_slices.append(spike_times[spike_start:spike_end])

    # Concatenate all spike slices
    if spike_slices:
        spike_times_sub = np.concatenate(spike_slices)
    else:
        spike_times_sub = np.array([], dtype=np.float64)

    return times[mask], spike_times_sub


def compute_directional_place_fields(
    env: Environment,
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    direction_labels: NDArray[np.object_],
    *,
    smoothing_method: Literal[
        "diffusion_kde", "gaussian_kde", "binned"
    ] = "diffusion_kde",
    bandwidth: float = 5.0,
    min_occupancy: float = 0.0,
) -> DirectionalPlaceFields:
    """Compute place fields conditioned on movement direction or trial type.

    Separates trajectory data by direction labels and computes independent
    place fields for each direction. This enables analysis of directional
    tuning in place cells, where firing rates differ based on which way
    the animal is moving through a location.

    Parameters
    ----------
    env : Environment
        Spatial environment defining the discretization.
    spike_times : NDArray[np.float64], shape (n_spikes,)
        Timestamps of spike occurrences (seconds).
    times : NDArray[np.float64], shape (n_timepoints,)
        Timestamps of trajectory samples (seconds). Must be sorted.
    positions : NDArray[np.float64], shape (n_timepoints, n_dims) or (n_timepoints,)
        Position trajectory. For 1D, can be shape (n_timepoints,) or (n_timepoints, 1).
    direction_labels : NDArray[object], shape (n_timepoints,)
        Direction label for each timepoint. Each label is a hashable string
        (e.g., "A→B", "forward", "CW"). The special label "other" is excluded
        from results, allowing unlabeled periods to be ignored.
    smoothing_method : {"diffusion_kde", "gaussian_kde", "binned"}, default="diffusion_kde"
        Estimation method passed through to the place-field helper.
    bandwidth : float, default=5.0
        Smoothing bandwidth in environment units (e.g., cm).
    min_occupancy : float, default=0.0
        Minimum occupancy threshold in seconds. Bins below this threshold are
        set to NaN.

    Returns
    -------
    DirectionalPlaceFields
        Container with:
        - ``firing_rates``: Mapping from direction label to firing rate array (n_bins,)
        - ``occupancy``: Mapping from direction label to per-bin occupancy (n_bins,)
        - ``env``: Spatial environment shared across labels
        - ``labels``: Tuple of direction labels in iteration order

    Raises
    ------
    ValueError
        If ``direction_labels`` length doesn't match ``times`` length.
    ValueError
        If ``bandwidth`` is not positive.

    See Also
    --------
    compute_spatial_rate : Compute single (non-directional) spatial rate.

    Notes
    -----
    The "other" label is reserved for timepoints that should be excluded from
    analysis (e.g., inter-trial intervals, stationary periods). Any timepoints
    with label "other" are ignored when computing fields.

    For each unique non-"other" label, this function:
    1. Creates a boolean mask for timepoints with that label
    2. Extracts the trajectory and spikes within those masked periods
    3. Calls ``compute_spatial_rate`` on the subset
    4. Stores the resulting field in the output mapping

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding.spatial import compute_directional_place_fields
    >>>
    >>> # Create environment and trajectory (seeded for reproducibility)
    >>> rng = np.random.default_rng(0)
    >>> positions = rng.uniform(0, 100, (1000, 2))
    >>> times = np.linspace(0, 100, 1000)
    >>> env = Environment.from_samples(positions, bin_size=10.0)
    >>>
    >>> # Create directional labels (first half forward, second half backward)
    >>> labels = np.array(["forward"] * 500 + ["backward"] * 500, dtype=object)
    >>> spike_times = np.sort(rng.uniform(0, 100, 50))  # spikes must be sorted
    >>>
    >>> # Compute directional place fields
    >>> result = compute_directional_place_fields(
    ...     env, spike_times, times, positions, labels, bandwidth=10.0
    ... )
    >>> "forward" in result.firing_rates
    True
    >>> "backward" in result.firing_rates
    True
    """
    # Validate direction_labels length matches times
    if len(direction_labels) != len(times):
        raise ValueError(
            f"direction_labels must have same length as times, "
            f"got {len(direction_labels)} and {len(times)}"
        )

    # Convert labels to array
    labels_arr = np.asarray(direction_labels, dtype=object)

    # Get unique labels, excluding "other"
    unique_labels = [label for label in np.unique(labels_arr) if label != "other"]

    # Sort labels for reproducibility
    unique_labels = sorted(unique_labels, key=str)

    # Compute place field for each direction
    firing_rates_dict: dict[str, NDArray[np.float64]] = {}
    occupancy_dict: dict[str, NDArray[np.float64]] = {}

    for label in unique_labels:
        # Build mask for this direction
        mask = labels_arr == label

        # Get subsets using our helper
        times_sub, spike_times_sub = _subset_spikes_by_time_mask(
            times, spike_times, mask
        )
        positions_sub = positions[mask]

        single = compute_spatial_rate(
            env,
            spike_times_sub,
            times_sub,
            positions_sub,
            smoothing_method=smoothing_method,
            bandwidth=bandwidth,
            min_occupancy=min_occupancy,
        )
        firing_rates_dict[str(label)] = np.asarray(single.firing_rate, dtype=np.float64)
        occupancy_dict[str(label)] = np.asarray(single.occupancy, dtype=np.float64)

    return DirectionalPlaceFields(
        firing_rates=firing_rates_dict,
        occupancy=occupancy_dict,
        env=env,
        labels=tuple(str(label) for label in unique_labels),
    )


# ==============================================================================
# Place Field Detection
# ==============================================================================


def detect_place_fields(
    env: Environment,
    firing_rate: NDArray[np.float64],
    *,
    threshold: float = 0.2,
    min_size: int | None = None,
    max_mean_rate: float = 10.0,
    detect_subfields: bool = True,
) -> PlaceFieldsResult:
    """Detect place fields using iterative peak-based approach (neurocode method).

    This implements the field-standard algorithm used by neurocode (AyA Lab)
    with support for subfield discrimination and interneuron exclusion.

    Parameters
    ----------
    env : Environment
        Spatial environment for binning.
    firing_rate : array, shape (n_bins,)
        Firing rate map (Hz) from neuron.
    threshold : float, default=0.2
        Fraction of peak rate for field boundary detection (0-1).
        Standard value is 0.2 (20% of peak).
    min_size : int, optional
        Minimum number of bins for a valid field. If None, defaults to 9 bins.
    max_mean_rate : float, default=10.0
        Maximum mean firing rate (Hz). Neurons exceeding this are excluded
        as putative interneurons (vandermeerlab convention).
    detect_subfields : bool, default=True
        If True, recursively detect subfields within large fields using
        higher thresholds. This discriminates coalescent place fields.

    Returns
    -------
    PlaceFieldsResult
        Container with ``fields`` (list of bin-index arrays, one per
        detected field) plus ``excluded_reason`` and ``n_excluded``
        attributes that distinguish "no detectable fields" from
        "neuron excluded by interneuron-rate filter". The result is
        iterable and indexable like the underlying list, so existing
        ``for f in result`` / ``len(result)`` / ``result[i]`` patterns
        keep working.

    Notes
    -----
    **Algorithm (neurocode approach)**:

    1. **Interneuron exclusion**: If mean rate > max_mean_rate, return no fields
    2. **Peak detection**: Find global maximum in firing rate map
    3. **Field segmentation**: Threshold at fraction of peak to define boundary
    4. **Connected component**: Extract bins above threshold connected to peak
    5. **Size filtering**: Discard fields smaller than min_size
    6. **Subfield recursion**: If detect_subfields=True, recursively apply
       higher thresholds (0.5, 0.7) to discriminate coalescent fields
    7. **Iteration**: Remove detected field bins and repeat until no peaks remain

    **Interneuron exclusion**: Following vandermeerlab convention, neurons with
    mean firing rate > 10 Hz are excluded as putative interneurons. Pyramidal
    cells (place cells) typically fire at 0.5-5 Hz.

    **Subfield detection**: When two place fields are close together, they may
    appear as a single broad field at low thresholds. Recursive thresholding
    at 0.5× and 0.7× peak discriminates true subfields.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding.spatial import detect_place_fields
    >>> # Create synthetic place cell
    >>> positions = np.random.randn(5000, 2) * 20
    >>> env = Environment.from_samples(positions, bin_size=2.0)
    >>> firing_rate = np.zeros(env.n_bins)
    >>> # Add Gaussian place field at center
    >>> for i in range(env.n_bins):
    ...     dist = np.linalg.norm(env.bin_centers[i])
    ...     firing_rate[i] = 8.0 * np.exp(-(dist**2) / (2 * 3.0**2))
    >>> fields = detect_place_fields(env, firing_rate)
    >>> len(fields)  # doctest: +SKIP
    1

    See Also
    --------
    SpatialRateResult : Result class with rate map and metrics

    References
    ----------
    .. [1] neurocode repository (AyA Lab, Cornell): FindPlaceFields.m
    .. [2] Wilson & McNaughton (1993). Dynamics of hippocampal ensemble code
           for space. Science 261(5124).
    """
    # Validate inputs
    if firing_rate.shape[0] != env.n_bins:
        raise ValueError(
            f"firing_rate shape {firing_rate.shape} does not match "
            f"env.n_bins ({env.n_bins})"
        )

    if not 0 < threshold < 1:
        raise ValueError(f"threshold must be in (0, 1), got {threshold}")

    # Set default min_size
    if min_size is None:
        min_size = 9  # Standard minimum (3×3 bins for 2D)

    # Interneuron exclusion. Emit a UserWarning AND surface the reason
    # in the returned PlaceFieldsResult so a caller running
    # detect_place_fields over a population can tell "this neuron has
    # no detectable place fields" (empty fields, excluded_reason=None)
    # from "this neuron was excluded as a putative interneuron"
    # (empty fields, excluded_reason="mean_rate_above_threshold").
    # The structured signal removes the need to listen for warnings.
    mean_rate = np.nanmean(firing_rate)
    if mean_rate > max_mean_rate:
        warnings.warn(
            f"detect_place_fields: neuron excluded as putative interneuron "
            f"(mean rate {float(mean_rate):.2f} Hz > max_mean_rate "
            f"{max_mean_rate} Hz). Returning empty field list. Pass a larger "
            "max_mean_rate to include fast-firing cells.",
            UserWarning,
            stacklevel=2,
        )
        return PlaceFieldsResult(
            fields=[],
            excluded_reason="mean_rate_above_threshold",
            n_excluded=1,
        )

    # Make a copy to modify during iteration
    rate_map = firing_rate.copy()
    fields = []

    # Iteratively find fields
    while True:
        # Handle all-NaN case
        if not np.any(np.isfinite(rate_map)):
            break  # No valid values remaining

        # Find peak
        peak_idx = int(np.nanargmax(rate_map))
        peak_rate = rate_map[peak_idx]

        # Check if peak is meaningful
        if peak_rate <= 0 or not np.isfinite(peak_rate):
            break

        # Threshold at fraction of peak
        threshold_rate = peak_rate * threshold

        # Find bins above threshold
        above_threshold = rate_map >= threshold_rate

        # Extract connected component containing peak
        field_bins = _extract_connected_component(peak_idx, above_threshold, env)

        # Check minimum size
        if len(field_bins) < min_size:
            # Remove this small field and continue
            rate_map[field_bins] = 0
            continue

        # Check for subfields (recursive thresholding)
        if detect_subfields and len(field_bins) > min_size * 2:
            # Try higher thresholds to discriminate subfields
            subfields = _detect_subfields(
                firing_rate[field_bins], field_bins, peak_rate, env, min_size
            )
            if len(subfields) > 1:
                # Found subfields - add them separately
                fields.extend(subfields)
            else:
                # No subfields - add as single field
                fields.append(field_bins)
        else:
            # Add field
            fields.append(field_bins)

        # Remove field bins from rate map
        rate_map[field_bins] = 0

        # Check if any meaningful peaks remain
        if np.nanmax(rate_map) < threshold_rate:
            break

    return PlaceFieldsResult(fields=fields, excluded_reason=None, n_excluded=0)


def _extract_connected_component_scipy(
    seed_idx: int,
    mask: NDArray[np.bool_],
    env: Environment,
) -> NDArray[np.int64]:
    """Extract connected component using scipy.ndimage.label (fast path for grids).

    This is the optimized path for grid-based environments, providing ~6× speedup
    over graph-based flood-fill by leveraging scipy's optimized N-D labeling.

    Parameters
    ----------
    seed_idx : int
        Starting bin index in active bin indexing.
    mask : array, shape (n_bins,)
        Boolean mask of candidate bins (active bin indexing).
    env : Environment
        Spatial environment (must be grid-based with grid_shape and active_mask).

    Returns
    -------
    component : array
        Bin indices in connected component (active bin indexing, sorted).

    Raises
    ------
    ValueError
        If environment does not have grid_shape or active_mask attributes.

    Notes
    -----
    This function only works for grid-based environments (RegularGridLayout,
    MaskedGridLayout, etc.). For non-grid environments (1D tracks, irregular
    graphs), use _extract_connected_component_graph() instead.

    The algorithm:
    1. Reshape flat mask to N-D grid using grid_shape
    2. Apply scipy.ndimage.label to find connected components
    3. Identify which component contains the seed
    4. Convert back to flat active bin indices
    """
    from scipy import ndimage

    # Validate environment has required attributes
    if env.grid_shape is None or env.active_mask is None:
        raise ValueError("scipy path requires grid_shape and active_mask")

    # Reshape flat mask (active bin indexing) to N-D grid (original grid indexing)
    grid_mask = np.zeros(env.grid_shape, dtype=bool)
    grid_mask[env.active_mask] = mask

    # Determine connectivity structure to match graph connectivity
    # Check if environment uses diagonal neighbors
    n_dims = len(env.grid_shape)
    if hasattr(env.layout, "_build_params_used"):
        params = env.layout._build_params_used
        connect_diagonal = params.get("connect_diagonal_neighbors", False)
    else:
        # Default: no diagonal connections (4-connected in 2D, 6-connected in 3D)
        connect_diagonal = False

    # Create connectivity structure for scipy
    if connect_diagonal:
        # Full connectivity (includes diagonals): connectivity = n_dims
        structure = ndimage.generate_binary_structure(n_dims, n_dims)
    else:
        # Axial connectivity only (no diagonals): connectivity = 1
        structure = ndimage.generate_binary_structure(n_dims, 1)

    # Label connected components in N-D grid
    labeled, _n_components = ndimage.label(grid_mask, structure=structure)

    # Convert seed from active bin index to grid coordinates
    # active_mask.ravel() gives flat indices of active bins in original grid
    active_flat_indices = np.where(env.active_mask.ravel())[0]
    seed_grid_flat_idx = active_flat_indices[seed_idx]
    seed_grid_coords = np.unravel_index(seed_grid_flat_idx, env.grid_shape)

    # Get label of component containing seed
    seed_label = labeled[seed_grid_coords]

    if seed_label == 0:
        # Seed not in any component (shouldn't happen if mask[seed_idx] is True)
        return np.array([seed_idx], dtype=np.int64)

    # Extract all grid positions in this component
    component_grid_mask = labeled == seed_label

    # Convert back to flat active bin indices
    # Find which active bins correspond to this component
    component_in_active_bins = component_grid_mask.ravel() & env.active_mask.ravel()
    component_grid_flat_indices = np.where(component_in_active_bins)[0]

    # Map from original grid flat indices to active bin indices
    component_bins = np.searchsorted(active_flat_indices, component_grid_flat_indices)

    return np.array(sorted(component_bins), dtype=np.int64)


def _extract_connected_component_graph(
    seed_idx: int,
    mask: NDArray[np.bool_],
    env: Environment,
) -> NDArray[np.int64]:
    """Extract connected component using graph-based flood-fill (fallback path).

    This is the fallback path for non-grid environments (1D tracks, irregular
    graphs) and works for any graph structure. It uses breadth-first search
    with direct graph.neighbors() queries.

    Parameters
    ----------
    seed_idx : int
        Starting bin index.
    mask : array, shape (n_bins,)
        Boolean mask of candidate bins.
    env : Environment
        Spatial environment for connectivity.

    Returns
    -------
    component : array
        Bin indices in connected component (sorted).

    Notes
    -----
    This is the original implementation, proven to be already optimal for
    sparse connected components on arbitrary graphs. Benchmarking showed
    this is faster than NetworkX's connected_components() due to avoiding
    subgraph creation overhead.
    """
    # Flood fill using graph connectivity (BFS)
    component_set = {seed_idx}
    frontier = deque([seed_idx])

    while frontier:
        current = frontier.popleft()
        # Get neighbors from graph
        neighbors = list(env.connectivity.neighbors(current))
        for neighbor in neighbors:
            if mask[neighbor] and neighbor not in component_set:
                component_set.add(neighbor)
                frontier.append(neighbor)

    return np.array(sorted(component_set), dtype=np.int64)


def _extract_connected_component(
    seed_idx: int,
    mask: NDArray[np.bool_],
    env: Environment,
) -> NDArray[np.int64]:
    """Extract connected component of bins from seed (routes to optimal method).

    Automatically selects the optimal algorithm based on environment type:
    - Grid environments (2D/3D): Uses scipy.ndimage.label (~6× faster)
    - Non-grid environments: Uses graph-based flood-fill

    Parameters
    ----------
    seed_idx : int
        Starting bin index.
    mask : array, shape (n_bins,)
        Boolean mask of candidate bins.
    env : Environment
        Spatial environment for connectivity.

    Returns
    -------
    component : array
        Bin indices in connected component (sorted).

    Notes
    -----
    The routing logic checks for grid-based environments using:
    - env.grid_shape is not None
    - len(env.grid_shape) >= 2 (2D or 3D grids)
    - env.active_mask is not None

    For grid environments, uses scipy.ndimage.label for ~6× speedup.
    For non-grid environments, uses graph-based flood-fill (already optimal).
    """
    # Check if scipy fast path is applicable
    if (
        env.grid_shape is not None
        and len(env.grid_shape) >= 2
        and env.active_mask is not None
    ):
        # Fast path: scipy.ndimage.label for grid environments
        return _extract_connected_component_scipy(seed_idx, mask, env)
    else:
        # Fallback path: graph-based flood-fill for non-grid environments
        return _extract_connected_component_graph(seed_idx, mask, env)


def _detect_subfields(
    field_rates: NDArray[np.float64],
    field_bins: NDArray[np.int64],
    peak_rate: float,
    env: Environment,
    min_size: int,
) -> list[NDArray[np.int64]]:
    """Recursively detect subfields using higher thresholds.

    Parameters
    ----------
    field_rates : array
        Firing rates within field bins.
    field_bins : array
        Bin indices of field.
    peak_rate : float
        Peak firing rate in field.
    env : Environment
        Spatial environment.
    min_size : int
        Minimum field size.

    Returns
    -------
    subfields : list of arrays
        List of subfield bin indices. If only one subfield found,
        returns list with original field.
    """
    # Try thresholds: 0.5 and 0.7 of peak
    subfield_thresholds = [0.5, 0.7]

    for thresh in subfield_thresholds:
        threshold_rate = peak_rate * thresh
        above_threshold = field_rates >= threshold_rate

        # Find connected components
        subfields = []
        remaining_mask = above_threshold.copy()

        while remaining_mask.any():
            # Find a seed
            seed_local_idx = np.where(remaining_mask)[0][0]
            seed_global_idx = field_bins[seed_local_idx]

            # Build mask in global coordinates
            global_mask = np.zeros(env.n_bins, dtype=bool)
            global_mask[field_bins[above_threshold]] = True

            # Extract component
            component_global = _extract_connected_component(
                seed_global_idx, global_mask, env
            )

            if len(component_global) >= min_size:
                subfields.append(component_global)

            # Remove from remaining mask
            for bin_idx in component_global:
                # Find local index
                local_indices = np.where(field_bins == bin_idx)[0]
                if len(local_indices) > 0:
                    remaining_mask[local_indices[0]] = False

        # If found multiple subfields, return them
        if len(subfields) > 1:
            return subfields

    # No subfields found
    return [field_bins]
