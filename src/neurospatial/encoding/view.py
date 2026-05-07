"""View rate computation for spatial view cells.

This module provides result classes and compute functions for view field
analysis. Spatial view cells fire when an animal *views* a specific location,
regardless of where the animal is positioned. This is distinct from place cells,
which fire when an animal is *at* a specific location.

Unlike spatial encoding which uses standard occupancy (time spent at each
location), view encoding uses view_occupancy (time spent *viewing* each
location). The view occupancy depends on the gaze model used.

Result Classes
--------------
ViewRateResult
    Single-neuron view field with convenience methods
ViewRatesResult
    Multi-neuron view fields with batch methods and iteration

Compute Functions
-----------------
compute_view_rate
    Compute view field for one neuron
compute_view_rates
    Compute view fields for multiple neurons (batch)

Key Difference: Place Cells vs Spatial View Cells
--------------------------------------------------
- **Place cell**: Fires when animal is *at* a specific location
- **Spatial view cell**: Fires when animal is *looking at* a specific location

For place cells, both place field and view field are similar (because viewing
location correlates with position). For spatial view cells, the view field
has higher spatial information than the place field.

Examples
--------
>>> from neurospatial import Environment
>>> from neurospatial.encoding.view import ViewRateResult
>>> import numpy as np

>>> # Create environment
>>> positions = np.random.rand(100, 2) * 100
>>> env = Environment.from_samples(positions, bin_size=5.0)

>>> # Create result (typically from compute_view_rate)
>>> firing_rate = np.random.rand(env.n_bins) * 10
>>> view_occupancy = np.ones(env.n_bins)
>>> result = ViewRateResult(
...     firing_rate=firing_rate,
...     view_occupancy=view_occupancy,
...     env=env,
...     gaze_model="fixed_distance",
...     view_distance=10.0,
...     smoothing_method="diffusion_kde",
...     bandwidth=5.0,
... )

References
----------
Rolls, E. T., et al. (1997). Spatial view cells in the primate hippocampus.
    European Journal of Neuroscience, 9(8), 1789-1794.
Georges-François, P., Rolls, E. T., & Robertson, R. G. (1999). Spatial view
    cells in the primate hippocampus: allocentric view not head direction or
    eye position or place. Cerebral Cortex, 9(3), 197-212.

See Also
--------
neurospatial.encoding.spatial : Spatial rate computation for place cells
neurospatial.encoding.spatial_view : Field-level spatial view utilities
neurospatial.ops.visibility : Visibility and gaze computation
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurospatial.encoding._base import SpatialResultMixin, _to_numpy

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes

    from neurospatial import Environment

# Re-export visibility functions for convenience in view cell workflow
from neurospatial.ops.visibility import (
    FieldOfView,
    compute_viewed_location,
    compute_viewshed,
    visibility_occupancy,
)

# ruff: noqa: RUF022 - intentionally grouped by category
__all__ = [
    # Result classes
    "ViewRateResult",
    "ViewRatesResult",
    # Compute functions
    "compute_view_rate",
    "compute_view_rates",
    # Convenience functions
    "is_spatial_view_cell",
    # Re-exports from ops.visibility
    "FieldOfView",
    "compute_viewed_location",
    "compute_viewshed",
    "visibility_occupancy",
]


@dataclass(frozen=True)
class ViewRateResult(SpatialResultMixin):
    """Result of view rate computation for a single neuron.

    This class wraps a view field (firing rate by viewed location) with its
    associated metadata (view occupancy, environment, gaze model parameters).
    View cells fire when the animal *views* a location, not when it is *at*
    that location.

    Parameters
    ----------
    firing_rate : ArrayLike
        Firing rate by viewed location in Hz. Shape is (n_bins,) where n_bins
        is the number of active bins in the environment. Can contain NaN for
        bins with insufficient view occupancy.
    view_occupancy : ArrayLike
        Time spent *viewing* each spatial bin in seconds. Shape is (n_bins,).
        This differs from standard occupancy (time at bin) - it represents
        the time the animal was looking at each location.
    env : Environment
        The spatial environment used for the computation. Provides bin
        centers, connectivity, and plotting methods.
    gaze_model : str
        Gaze model used for computing viewed location:
        - "fixed_distance": Point at fixed distance in gaze direction
        - "ray_cast": Intersection with environment boundary
        - "boundary": Nearest boundary point in gaze direction
    view_distance : float
        Distance parameter for the gaze model (relevant for "fixed_distance").
    smoothing_method : str
        Smoothing method used: "diffusion_kde", "gaussian_kde", or "binned".
    bandwidth : float
        Smoothing bandwidth in the same units as the environment's bin_size.

    Attributes
    ----------
    firing_rate : ArrayLike
        Firing rate by viewed location in Hz. Shape is (n_bins,).
    view_occupancy : ArrayLike
        Time spent viewing each bin in seconds. Shape is (n_bins,).
    env : Environment
        The spatial environment.
    gaze_model : str
        Gaze model used.
    view_distance : float
        Distance parameter for gaze model.
    smoothing_method : str
        Smoothing method used.
    bandwidth : float
        Smoothing bandwidth.

    Notes
    -----
    This is a frozen dataclass (immutable). All fields are set at construction
    and cannot be modified afterward.

    **View occupancy vs standard occupancy**: View occupancy tracks time spent
    *viewing* each location, which depends on the gaze model. For spatial view
    cells, firing rate should be computed using view occupancy, not standard
    occupancy.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding.view import ViewRateResult

    >>> # Create a simple environment
    >>> positions = np.random.rand(100, 2) * 50
    >>> env = Environment.from_samples(positions, bin_size=5.0)

    >>> # Create result
    >>> firing_rate = np.random.rand(env.n_bins) * 10
    >>> view_occupancy = np.ones(env.n_bins)
    >>> result = ViewRateResult(
    ...     firing_rate=firing_rate,
    ...     view_occupancy=view_occupancy,
    ...     env=env,
    ...     gaze_model="fixed_distance",
    ...     view_distance=10.0,
    ...     smoothing_method="diffusion_kde",
    ...     bandwidth=5.0,
    ... )

    >>> # Access fields
    >>> result.firing_rate.shape
    (n_bins,)
    >>> result.gaze_model
    'fixed_distance'

    See Also
    --------
    ViewRatesResult : Batch version for multiple neurons
    compute_view_rate : Function to compute this result
    """

    firing_rate: ArrayLike
    view_occupancy: ArrayLike
    env: Environment
    gaze_model: str
    view_distance: float
    smoothing_method: str
    bandwidth: float

    def plot(self, ax: Axes | None = None, **kwargs: Any) -> Axes:
        """Plot the view field (firing rate by viewed location).

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

        Notes
        -----
        The view field shows firing rate indexed by *viewed* location (where
        the animal looked), not by animal position. This is the key difference
        from place fields.

        Examples
        --------
        >>> result = ViewRateResult(...)
        >>> ax = result.plot()
        >>> plt.show()

        >>> fig, ax = plt.subplots()
        >>> result.plot(ax=ax, cmap="hot", vmax=20.0)

        See Also
        --------
        peak_view_location : Get location of peak view response
        """
        return self.env.plot_field(_to_numpy(self.firing_rate), ax=ax, **kwargs)

    def peak_view_location(self) -> NDArray[np.float64]:
        """Location of peak view response.

        Returns the spatial coordinates where the neuron shows maximum firing
        rate when the animal *views* that location.

        Returns
        -------
        ndarray, shape (n_dims,)
            Spatial coordinates of the bin with maximum firing rate.
            Uses nanargmax to handle NaN values in the firing rate map.

        Notes
        -----
        For spatial view cells, the peak view location represents where the
        neuron fires most when the animal looks there, regardless of where
        the animal is positioned.

        Examples
        --------
        >>> result = ViewRateResult(...)
        >>> peak = result.peak_view_location()
        >>> print(f"Peak view response at ({peak[0]:.1f}, {peak[1]:.1f}) cm")

        See Also
        --------
        view_spatial_information : Quantify spatial selectivity of view response
        plot : Visualize the view field
        """
        firing_rate = _to_numpy(self.firing_rate)
        peak_bin = np.nanargmax(firing_rate)
        result: NDArray[np.float64] = self.env.bin_centers[peak_bin]
        return result

    def view_spatial_information(self) -> float:
        """Skaggs spatial information based on view occupancy (bits per spike).

        Quantifies how much information each spike conveys about the *viewed*
        location (where the animal is looking), not the animal's position.
        Higher values indicate more spatially selective view responses.

        Returns
        -------
        float
            Spatial information in bits/spike. Always non-negative.
            Returns 0.0 for uniform firing.

        Notes
        -----
        Uses the Skaggs et al. (1993) formula with **view occupancy**:

        .. math::

            I = \\sum_i p_i \\frac{r_i}{\\bar{r}} \\log_2 \\left( \\frac{r_i}{\\bar{r}} \\right)

        where :math:`p_i` is the fraction of time spent *viewing* bin i
        (not time spent *at* bin i).

        **Key difference from spatial_information()**: This metric uses
        ``view_occupancy`` (time viewing each location) rather than standard
        ``occupancy`` (time at each location). For true spatial view cells,
        this metric should be higher than standard spatial information.

        **Interpretation**:

        - Spatial view cells typically have 0.5-2 bits/spike
        - Higher values indicate more spatially selective view responses
        - Zero means uniform view-response (no spatial view selectivity)

        References
        ----------
        .. [1] Skaggs, W. E., McNaughton, B. L., & Gothard, K. M. (1993).
               An information-theoretic approach to deciphering the hippocampal code.
        .. [2] Rolls, E. T., et al. (1997). Spatial view cells in the primate
               hippocampus. European Journal of Neuroscience, 9(8), 1789-1794.

        Examples
        --------
        >>> result = ViewRateResult(...)
        >>> info = result.view_spatial_information()
        >>> print(f"View spatial information: {info:.2f} bits/spike")

        See Also
        --------
        peak_view_location : Get location of peak view response
        neurospatial.encoding._metrics.spatial_information : Underlying computation
        """
        from neurospatial.encoding._metrics import spatial_information

        return spatial_information(
            _to_numpy(self.firing_rate), _to_numpy(self.view_occupancy)
        )

    def is_view_cell(self, min_info: float = 0.5) -> bool:
        """Classify as spatial view cell based on view spatial information.

        A neuron is classified as a spatial view cell if its view spatial
        information exceeds the minimum threshold. Spatial view cells fire
        when the animal *looks at* a specific location, regardless of where
        the animal is positioned.

        Parameters
        ----------
        min_info : float, default=0.5
            Minimum view spatial information threshold in bits/spike.

            **How was 0.5 chosen?**

            This threshold is based on typical values reported in primate
            hippocampus studies (Rolls et al., 1997; Georges-François et al.,
            1999). Empirically:

            - Strong view cells: 1.0-2.0+ bits/spike
            - Moderate view cells: 0.5-1.0 bits/spike
            - Weak/non-view cells: < 0.5 bits/spike

            **When to adjust:**

            - Different brain regions: May need 0.3-0.7
            - Different species: Validate threshold first
            - Noisy recordings: Consider 0.3 (more permissive)
            - Publication quality: Use 0.7 or higher (more conservative)

        Returns
        -------
        bool
            True if view_spatial_information() > min_info, False otherwise.

        Notes
        -----
        Unlike head direction cells which use both mean vector length and
        Rayleigh test, view cell classification is typically based solely
        on spatial information because view fields don't have the circular
        structure of directional tuning.

        **Place cells vs spatial view cells**: Both may show high spatial
        information, but spatial view cells have higher *view* spatial
        information (using view occupancy) than *position* spatial information
        (using standard occupancy).

        References
        ----------
        .. [1] Rolls, E. T., et al. (1997). Spatial view cells in the primate
               hippocampus. European Journal of Neuroscience, 9(8), 1789-1794.
        .. [2] Georges-François, P., Rolls, E. T., & Robertson, R. G. (1999).
               Spatial view cells in the primate hippocampus: allocentric view
               not head direction or eye position or place.

        Examples
        --------
        >>> result = ViewRateResult(...)
        >>> if result.is_view_cell():
        ...     print("This is a spatial view cell!")
        >>> if result.is_view_cell(min_info=0.3):
        ...     print("This cell passes a more permissive threshold")

        See Also
        --------
        view_spatial_information : Compute the spatial information metric
        peak_view_location : Get location of peak view response
        """
        return self.view_spatial_information() > min_info


@dataclass(frozen=True)
class ViewRatesResult(SpatialResultMixin):
    """Result of view rate computation for multiple neurons.

    This class wraps view fields for a population of neurons with shared
    metadata (view occupancy, environment, gaze model parameters). It supports
    iteration and indexing to access individual neuron results.

    Parameters
    ----------
    firing_rates : ArrayLike
        Firing rates by viewed location for all neurons in Hz.
        Shape is (n_neurons, n_bins) where n_bins is the number of active
        bins in the environment.
    view_occupancy : ArrayLike
        Time spent *viewing* each spatial bin in seconds. Shape is (n_bins,).
        This is shared across all neurons since gaze direction depends on
        behavior, not neural activity.
    env : Environment
        The spatial environment used for the computation.
    gaze_model : str
        Gaze model used for computing viewed location.
    view_distance : float
        Distance parameter for the gaze model.
    smoothing_method : str
        Smoothing method used: "diffusion_kde", "gaussian_kde", or "binned".
    bandwidth : float
        Smoothing bandwidth in the same units as the environment's bin_size.

    Attributes
    ----------
    firing_rates : ArrayLike
        Firing rates for all neurons. Shape is (n_neurons, n_bins).
    view_occupancy : ArrayLike
        Time spent viewing each bin in seconds. Shape is (n_bins,). Shared.
    env : Environment
        The spatial environment.
    gaze_model : str
        Gaze model used.
    view_distance : float
        Distance parameter for gaze model.
    smoothing_method : str
        Smoothing method used.
    bandwidth : float
        Smoothing bandwidth.

    Notes
    -----
    This is a frozen dataclass (immutable). All fields are set at construction
    and cannot be modified afterward.

    **Iteration interface**: Supports `len()`, indexing with `[]`, and
    iteration with `for`. Each element is a `ViewRateResult` for one neuron.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding.view import ViewRatesResult

    >>> # Create a simple environment
    >>> positions = np.random.rand(100, 2) * 50
    >>> env = Environment.from_samples(positions, bin_size=5.0)

    >>> # Create batch result for 3 neurons
    >>> firing_rates = np.random.rand(3, env.n_bins) * 10
    >>> view_occupancy = np.ones(env.n_bins)
    >>> result = ViewRatesResult(
    ...     firing_rates=firing_rates,
    ...     view_occupancy=view_occupancy,
    ...     env=env,
    ...     gaze_model="fixed_distance",
    ...     view_distance=10.0,
    ...     smoothing_method="diffusion_kde",
    ...     bandwidth=5.0,
    ... )

    >>> # Access fields
    >>> len(result)
    3
    >>> result[0]  # Get first neuron as ViewRateResult
    ViewRateResult(...)

    >>> # Iterate over neurons
    >>> for single in result:
    ...     print(single.firing_rate.max())

    See Also
    --------
    ViewRateResult : Single-neuron version
    compute_view_rates : Function to compute this result
    """

    firing_rates: ArrayLike
    view_occupancy: ArrayLike
    env: Environment
    gaze_model: str
    view_distance: float
    smoothing_method: str
    bandwidth: float

    def __len__(self) -> int:
        """Return the number of neurons.

        Returns
        -------
        int
            Number of neurons in the batch.

        Examples
        --------
        >>> len(result)
        5
        """
        return len(self.firing_rates)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> ViewRateResult:
        """Get single-neuron result by index.

        Parameters
        ----------
        idx : int
            Index of the neuron (0-based).

        Returns
        -------
        ViewRateResult
            View rate result for the specified neuron.

        Examples
        --------
        >>> single = result[0]
        >>> isinstance(single, ViewRateResult)
        True
        """
        return ViewRateResult(
            firing_rate=self.firing_rates[idx],  # type: ignore[index]
            view_occupancy=self.view_occupancy,
            env=self.env,
            gaze_model=self.gaze_model,
            view_distance=self.view_distance,
            smoothing_method=self.smoothing_method,
            bandwidth=self.bandwidth,
        )

    def __iter__(self) -> Iterator[ViewRateResult]:
        """Iterate over single-neuron results.

        Yields
        ------
        ViewRateResult
            View rate result for each neuron in order.

        Examples
        --------
        >>> for single in result:
        ...     print(single.firing_rate.max())
        """
        for i in range(len(self)):
            yield self[i]

    def plot(self, idx: int, ax: Axes | None = None, **kwargs: Any) -> Axes:
        """Plot the view field for a specific neuron.

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

        Notes
        -----
        The view field shows firing rate indexed by *viewed* location (where
        the animal looked), not by animal position. This is the key difference
        from place fields.

        Examples
        --------
        >>> # Plot the first neuron's view field
        >>> ax = result.plot(idx=0)
        >>> plt.show()

        >>> # Plot neuron 5 with custom colormap
        >>> fig, ax = plt.subplots()
        >>> result.plot(idx=5, ax=ax, cmap="hot", vmax=20.0)

        See Also
        --------
        peak_view_locations : Get locations of peak view responses
        ViewRateResult.plot : Plot for single-neuron result
        """
        return self.env.plot_field(
            _to_numpy(self.firing_rates[idx]),  # type: ignore[index]
            ax=ax,
            **kwargs,
        )

    def peak_view_locations(self) -> NDArray[np.float64]:
        """Locations of peak view responses for all neurons.

        Returns the spatial coordinates where each neuron shows maximum
        firing rate when the animal *views* that location.

        Returns
        -------
        ndarray, shape (n_neurons, n_dims)
            Spatial coordinates of the bins with maximum firing rate for
            each neuron. Uses nanargmax to handle NaN values. Returns NaN
            coordinates for neurons with all-NaN firing rates.

        Notes
        -----
        For spatial view cells, peak view locations represent where each
        neuron fires most when the animal looks there, regardless of where
        the animal is positioned.

        If all firing rates for a neuron are NaN, the peak location will
        be NaN coordinates.

        Examples
        --------
        >>> peaks = result.peak_view_locations()
        >>> print(f"Neuron 0 peak at ({peaks[0, 0]:.1f}, {peaks[0, 1]:.1f}) cm")

        See Also
        --------
        ViewRateResult.peak_view_location : Single-neuron version
        view_spatial_information : Quantify spatial selectivity
        """
        firing_rates = _to_numpy(self.firing_rates)
        n_neurons = firing_rates.shape[0]
        n_dims = self.env.bin_centers.shape[1]

        peak_locs = np.empty((n_neurons, n_dims), dtype=np.float64)
        for i in range(n_neurons):
            valid_mask = ~np.isnan(firing_rates[i])
            if not np.any(valid_mask):
                # All NaN - return NaN coordinates
                peak_locs[i] = np.nan
            else:
                peak_idx = int(np.nanargmax(firing_rates[i]))
                peak_locs[i] = self.env.bin_centers[peak_idx]
        return peak_locs

    def view_spatial_information(self) -> NDArray[np.float64]:
        """View spatial information for all neurons (bits per spike).

        Quantifies how much information each spike conveys about the *viewed*
        location for each neuron.

        Returns
        -------
        ndarray, shape (n_neurons,)
            Spatial information in bits/spike for each neuron.
            Always non-negative. Returns 0.0 for uniform firing.

        Notes
        -----
        Uses the Skaggs et al. (1993) formula with **view occupancy**:

        .. math::

            I = \\sum_i p_i \\frac{r_i}{\\bar{r}} \\log_2 \\left( \\frac{r_i}{\\bar{r}} \\right)

        where :math:`p_i` is the fraction of time spent *viewing* bin i.

        This is computed by delegating to the batch spatial information
        function in ``_metrics.py``.

        Examples
        --------
        >>> info = result.view_spatial_information()
        >>> print(f"Neuron with highest info: {np.argmax(info)}")
        >>> print(f"Info values: {info[:5]}")

        See Also
        --------
        ViewRateResult.view_spatial_information : Single-neuron version
        detect_view_cells : Classify as view cells based on this metric
        """
        from neurospatial.encoding._metrics import batch_spatial_information

        return batch_spatial_information(
            _to_numpy(self.firing_rates), _to_numpy(self.view_occupancy)
        )

    def detect_view_cells(self, min_info: float = 0.5) -> NDArray[np.bool_]:
        """Classify neurons as spatial view cells.

        A neuron is classified as a spatial view cell if its view spatial
        information exceeds the minimum threshold.

        Parameters
        ----------
        min_info : float, default=0.5
            Minimum view spatial information threshold in bits/spike.
            See ViewRateResult.is_view_cell() for threshold rationale.

        Returns
        -------
        ndarray, shape (n_neurons,)
            Boolean array where True indicates the neuron is classified
            as a spatial view cell.

        Notes
        -----
        Uses vectorized computation of view_spatial_information() for
        efficiency with large populations.

        Examples
        --------
        >>> is_view_cell = result.detect_view_cells()
        >>> print(f"Found {is_view_cell.sum()} view cells")

        >>> # Use stricter threshold
        >>> is_view_cell = result.detect_view_cells(min_info=1.0)

        See Also
        --------
        ViewRateResult.is_view_cell : Single-neuron classification
        view_spatial_information : The metric used for classification
        """
        info = self.view_spatial_information()
        return info > min_info

    def to_dataframe(
        self,
        neuron_ids: Sequence[str | int] | None = None,
    ) -> pd.DataFrame:
        """Export metrics to DataFrame for exploratory analysis.

        Computes all view field metrics and exports them to a pandas DataFrame
        for easy filtering, sorting, and analysis.

        Parameters
        ----------
        neuron_ids : sequence of str or int, optional
            Identifiers for each neuron. If None, uses integer indices
            (0, 1, 2, ..., n_neurons-1).

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:

            - neuron_id: identifier for each neuron
            - peak_view_x: x-coordinate of peak view location
            - peak_view_y: y-coordinate of peak view location
            - peak_rate: maximum firing rate (Hz)
            - view_spatial_info: view spatial information (bits/spike)
            - is_view_cell: whether classified as view cell (using default threshold)

        Raises
        ------
        ValueError
            If neuron_ids has a different length than the number of neurons.

        Notes
        -----
        This method computes all metrics at once, which may be slow for
        large populations. For selective metric computation, use the
        individual methods (``view_spatial_information()``, ``detect_view_cells()``, etc.).

        **Common pandas workflows**:

        - Filter: ``df[df["is_view_cell"] == True]``
        - Sort: ``df.sort_values("view_spatial_info", ascending=False)``
        - Top-N: ``df.nlargest(10, "peak_rate")``

        Examples
        --------
        >>> result = ViewRatesResult(...)
        >>> df = result.to_dataframe()
        >>> print(df.head())

        >>> # With custom neuron IDs
        >>> df = result.to_dataframe(neuron_ids=["unit_1", "unit_2", ...])

        >>> # Filter to view cells only
        >>> view_cells_df = df[df["is_view_cell"]]

        See Also
        --------
        peak_view_locations : Get peak view locations for all neurons
        view_spatial_information : Get spatial information for all neurons
        detect_view_cells : Classify neurons as view cells
        """
        import pandas as pd

        n_neurons = len(self)

        # Validate and convert neuron_ids
        if neuron_ids is None:
            neuron_ids_list: list[str | int] = list(range(n_neurons))
        else:
            neuron_ids_list = list(neuron_ids)
            if len(neuron_ids_list) != n_neurons:
                raise ValueError(
                    f"neuron_ids has {len(neuron_ids_list)} elements but "
                    f"result contains {n_neurons} neurons"
                )

        # Compute all metrics
        peak_locs = self.peak_view_locations()
        firing_rates = _to_numpy(self.firing_rates)
        peak_rates = np.nanmax(firing_rates, axis=1) if n_neurons > 0 else np.array([])
        view_info = self.view_spatial_information()
        is_view_cell = self.detect_view_cells()

        # Determine dimensionality for peak location columns
        # View encoding is typically 2D, but handle 1D for robustness
        n_dims = peak_locs.shape[1] if peak_locs.ndim > 1 else 1

        # Build DataFrame
        data: dict[str, Any] = {
            "neuron_id": neuron_ids_list,
            "peak_view_x": peak_locs[:, 0],
            "peak_view_y": peak_locs[:, 1]
            if n_dims > 1
            else np.full(n_neurons, np.nan),
            "peak_rate": peak_rates,
            "view_spatial_info": view_info,
            "is_view_cell": is_view_cell,
        }

        return pd.DataFrame(data)


# =============================================================================
# Compute Functions
# =============================================================================


def compute_view_rate(
    env: Environment,
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
    *,
    gaze_model: Literal["fixed_distance", "ray_cast", "boundary"] = "fixed_distance",
    view_distance: float = 10.0,
    gaze_offsets: NDArray[np.float64] | None = None,
    smoothing_method: Literal[
        "diffusion_kde", "gaussian_kde", "binned"
    ] = "diffusion_kde",
    bandwidth: float = 5.0,
    min_occupancy: float = 0.0,
    backend: Literal["numpy", "jax", "auto"] = "numpy",
) -> ViewRateResult:
    """Compute view field for one neuron.

    This function computes a smoothed firing rate map indexed by *viewed*
    location (where the animal was looking), not the animal's position.
    This is the key metric for identifying spatial view cells.

    Parameters
    ----------
    env : Environment
        The spatial environment defining the bin structure. Must be fitted
        (e.g., created via ``Environment.from_samples()``).
    spike_times : ndarray, shape (n_spikes,)
        Times of spike events in seconds. Can be empty.
    times : ndarray, shape (n_samples,)
        Timestamps of trajectory samples in seconds.
    positions : ndarray, shape (n_samples, 2)
        Position coordinates at each time sample.
    headings : ndarray, shape (n_samples,)
        Head direction at each time sample (radians, 0=East).
    gaze_model : {"fixed_distance", "ray_cast", "boundary"}, default="fixed_distance"
        Method for computing viewed location:

        - **fixed_distance**: Point at fixed distance in gaze direction.
          Fast and simple, good default for most analyses.
        - **ray_cast**: Intersection with environment boundary. More
          realistic for environments with walls.
        - **boundary**: Nearest boundary point in gaze direction.

    view_distance : float, default=10.0
        Distance for fixed_distance gaze model (environment units).
        Ignored for ray_cast and boundary models.
    gaze_offsets : ndarray, shape (n_samples,), optional
        Offset from head direction to actual gaze direction (radians).
        Positive values indicate gaze to the left of heading.
        If None (default), gaze is aligned with head direction.
        Use this for eye-tracking data in primate spatial view cell studies
        where gaze direction differs from head direction.
    smoothing_method : {"diffusion_kde", "gaussian_kde", "binned"}, default="diffusion_kde"
        Smoothing method to use:

        - **diffusion_kde** (recommended): Graph-based boundary-aware KDE.
          Respects environment boundaries (walls, obstacles).
        - **gaussian_kde**: Standard Euclidean KDE. Ignores boundaries.
        - **binned**: Bin-then-smooth method. Computes raw rate first, then smooths.

    bandwidth : float, default=5.0
        Smoothing bandwidth in the same units as bin_size. Larger values
        produce more smoothing.
    min_occupancy : float, default=0.0
        Minimum view occupancy (seconds) for a bin to be included. Bins with
        view occupancy below this threshold are set to NaN.
    backend : {'numpy', 'jax', 'auto'}, default='numpy'
        Computation backend.

        - 'numpy': Use NumPy (always available)
        - 'jax': Use JAX for rate computation (requires JAX installation)
        - 'auto': Use JAX if available, otherwise NumPy

    Returns
    -------
    ViewRateResult
        Result object containing:

        - ``firing_rate``: Firing rate by viewed location in Hz, shape (n_bins,)
        - ``view_occupancy``: Time viewing each bin in seconds, shape (n_bins,)
        - ``env``: The environment used
        - ``gaze_model``: Gaze model used
        - ``view_distance``: View distance parameter
        - ``smoothing_method``: Method used for smoothing
        - ``bandwidth``: Bandwidth used for smoothing

    Raises
    ------
    ValueError
        If gaze_model is not one of the valid options.
        If inputs have mismatched lengths.

    See Also
    --------
    compute_view_rates : Batch version for multiple neurons
    ViewRateResult : Result class with convenience methods
    compute_spatial_rate : Standard spatial rate (by animal position)

    Notes
    -----
    The function uses the view binning layer (``_view_binning.py``) to convert
    spike times to spike counts based on *viewed* location, then the smoothing
    layer (``_smoothing.py``) to compute the smoothed firing rate.

    **Algorithm**:

    1. Compute viewed locations from positions and headings using gaze model
    2. Bin spikes by viewed location (not animal position)
    3. Compute view occupancy (time spent viewing each bin)
    4. Apply smoothing (method-dependent, see ``_smoothing.py``)

    **Place cells vs spatial view cells**: For place cells, the view field
    and place field are correlated (viewing location correlates with position).
    For true spatial view cells, the view field shows stronger spatial
    selectivity than the place field.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding.view import compute_view_rate

    >>> # Create environment from sample positions
    >>> sample_positions = np.random.rand(1000, 2) * 100
    >>> env = Environment.from_samples(sample_positions, bin_size=5.0)

    >>> # Create trajectory with positions and headings
    >>> times = np.linspace(0, 10, 1000)
    >>> positions = np.random.rand(1000, 2) * 100
    >>> headings = np.random.uniform(0, 2 * np.pi, 1000)
    >>> spike_times = np.array([1.0, 2.5, 4.0, 7.5, 8.2])

    >>> # Compute view rate
    >>> result = compute_view_rate(
    ...     env,
    ...     spike_times,
    ...     times,
    ...     positions,
    ...     headings,
    ...     gaze_model="fixed_distance",
    ...     view_distance=10.0,
    ... )

    >>> # Access results
    >>> peak = result.peak_view_location()
    >>> info = result.view_spatial_information()
    >>> is_view_cell = result.is_view_cell()

    >>> # Plot the view field
    >>> ax = result.plot()

    References
    ----------
    .. [1] Rolls, E. T., et al. (1997). Spatial view cells in the primate
           hippocampus. European Journal of Neuroscience, 9(8), 1789-1794.
    """
    from neurospatial.encoding._backend import (
        SUPPORTED_BACKENDS,
        get_backend_name,
        is_jax_available,
    )
    from neurospatial.encoding._smoothing import (
        _validate_smoothing_parameters,
        smooth_rate_map,
    )
    from neurospatial.encoding._validation import validate_trajectory
    from neurospatial.encoding._view_binning import bin_view_spike_trains

    # Validate backend
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unknown backend: {backend!r}. "
            f"Supported backends are: {', '.join(repr(b) for b in SUPPORTED_BACKENDS)}"
        )

    # Resolve backend (handles "auto" → "numpy" or "jax")
    # This raises ImportError if backend="jax" and JAX is unavailable
    resolved_backend = get_backend_name(backend)

    # Validate gaze_model
    valid_gaze_models = {"fixed_distance", "ray_cast", "boundary"}
    if gaze_model not in valid_gaze_models:
        raise ValueError(
            f"Invalid gaze_model: '{gaze_model}'. "
            f"Must be one of {sorted(valid_gaze_models)}"
        )

    _validate_smoothing_parameters(smoothing_method, bandwidth)

    # Convert inputs to arrays
    spike_times = np.asarray(spike_times, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    headings = np.asarray(headings, dtype=np.float64)

    validate_trajectory(times, positions=positions, headings=headings)
    n_samples = len(times)

    # Validate gaze_offsets if provided
    if gaze_offsets is not None:
        gaze_offsets = np.asarray(gaze_offsets, dtype=np.float64)
        if len(gaze_offsets) != n_samples:
            raise ValueError(
                f"gaze_offsets length ({len(gaze_offsets)}) must match "
                f"times length ({n_samples})"
            )

    # Reuse the batch binning path for the single-neuron API so viewed
    # coordinates are computed once and shared by spike counts and occupancy.
    spike_counts_batch, view_occupancy = bin_view_spike_trains(
        env,
        [spike_times],
        times,
        positions,
        headings,
        gaze_model=gaze_model,
        view_distance=view_distance,
        gaze_offsets=gaze_offsets,
        n_jobs=1,
    )
    spike_counts = spike_counts_batch[0]

    # Apply smoothing to compute firing rate
    # smooth_rate_map dispatches to JAX or NumPy based on backend
    firing_rate = smooth_rate_map(
        env,
        spike_counts,
        view_occupancy,
        method=smoothing_method,
        bandwidth=bandwidth,
        min_occupancy=min_occupancy,
        backend=resolved_backend,
    )

    # Convert occupancy to JAX if JAX backend is selected
    # (firing_rate is already JAX from smooth_rate_map)
    if resolved_backend == "jax" and is_jax_available():
        import jax.numpy as jnp

        view_occupancy = jnp.asarray(view_occupancy, dtype=jnp.float64)  # type: ignore[assignment]

    # Return result
    return ViewRateResult(
        firing_rate=firing_rate,
        view_occupancy=view_occupancy,
        env=env,
        gaze_model=gaze_model,
        view_distance=view_distance,
        smoothing_method=smoothing_method,
        bandwidth=bandwidth,
    )


def compute_view_rates(
    env: Environment,
    spike_times: Sequence[NDArray[np.float64]] | NDArray[np.float64],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
    *,
    gaze_model: Literal["fixed_distance", "ray_cast", "boundary"] = "fixed_distance",
    view_distance: float = 10.0,
    gaze_offsets: NDArray[np.float64] | None = None,
    smoothing_method: Literal[
        "diffusion_kde", "gaussian_kde", "binned"
    ] = "diffusion_kde",
    bandwidth: float = 5.0,
    min_occupancy: float = 0.0,
    n_jobs: int = 1,
    backend: Literal["numpy", "jax", "auto"] = "numpy",
) -> ViewRatesResult:
    """Compute view fields for multiple neurons.

    This is the batch version of ``compute_view_rate()`` that efficiently
    processes multiple neurons with shared trajectory data. It precomputes
    shared quantities (view occupancy, viewed bins) once and optionally
    parallelizes spike counting with joblib.

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
    positions : ndarray, shape (n_samples, 2)
        Position coordinates at each time sample.
    headings : ndarray, shape (n_samples,)
        Head direction at each time sample (radians, 0=East).
    gaze_model : {"fixed_distance", "ray_cast", "boundary"}, default="fixed_distance"
        Method for computing viewed location:

        - **fixed_distance**: Point at fixed distance in gaze direction.
          Fast and simple, good default for most analyses.
        - **ray_cast**: Intersection with environment boundary. More
          realistic for environments with walls.
        - **boundary**: Nearest boundary point in gaze direction.

    view_distance : float, default=10.0
        Distance for fixed_distance gaze model (environment units).
        Ignored for ray_cast and boundary models.
    gaze_offsets : ndarray, shape (n_samples,), optional
        Offset from head direction to actual gaze direction (radians).
        Positive values indicate gaze to the left of heading.
        If None (default), gaze is aligned with head direction.
        Use this for eye-tracking data in primate spatial view cell studies
        where gaze direction differs from head direction.
    smoothing_method : {"diffusion_kde", "gaussian_kde", "binned"}, default="diffusion_kde"
        Smoothing method to use. See ``compute_view_rate()`` for details.
    bandwidth : float, default=5.0
        Smoothing bandwidth in the same units as bin_size.
    min_occupancy : float, default=0.0
        Minimum view occupancy (seconds) for a bin to be included.
    n_jobs : int, default=1
        Number of parallel jobs for spike counting. Use -1 for all CPUs.
        1 means sequential processing (no parallelization overhead).
    backend : {'numpy', 'jax', 'auto'}, default='numpy'
        Computation backend.

        - 'numpy': Use NumPy (always available)
        - 'jax': Use JAX for rate computation (requires JAX installation)
        - 'auto': Use JAX if available, otherwise NumPy

    Returns
    -------
    ViewRatesResult
        Result object containing:

        - ``firing_rates``: Firing rate maps, shape ``(n_neurons, n_bins)``
        - ``view_occupancy``: Time viewing each bin in seconds, shape ``(n_bins,)``
        - ``env``: The environment used
        - ``gaze_model``: Gaze model used
        - ``view_distance``: View distance parameter
        - ``smoothing_method``: Method used for smoothing
        - ``bandwidth``: Bandwidth used for smoothing

        The result supports iteration: ``for single in result: ...``
        and indexing: ``single = result[0]``.

    Raises
    ------
    ValueError
        If gaze_model is not one of the valid options.
        If inputs have mismatched lengths.

    See Also
    --------
    compute_view_rate : Single-neuron version
    ViewRatesResult : Result class with batch methods
    compute_spatial_rates : Standard spatial rates (by animal position)

    Notes
    -----
    **Efficiency advantages over calling ``compute_view_rate()`` in a loop**:

    1. View occupancy is computed once and shared across all neurons
    2. Diffusion kernel (for ``diffusion_kde`` method) is computed once
    3. Viewed-bin mapping is done once
    4. Spike binning can be parallelized with joblib

    **When to use batch vs single**:

    - **Batch** (this function): Processing 3+ neurons, or any case where
      efficiency matters. The overhead of precomputing shared quantities
      is amortized over multiple neurons.
    - **Single** (``compute_view_rate``): Processing 1-2 neurons, or when
      you need fine-grained control over individual neurons.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding.view import compute_view_rates

    >>> # Create environment from sample positions
    >>> sample_positions = np.random.rand(1000, 2) * 100
    >>> env = Environment.from_samples(sample_positions, bin_size=5.0)

    >>> # Create trajectory with positions and headings
    >>> times = np.linspace(0, 10, 1000)
    >>> positions = np.random.rand(1000, 2) * 100
    >>> headings = np.random.uniform(0, 2 * np.pi, 1000)

    >>> # Spike times for 3 neurons
    >>> spike_times = [
    ...     np.array([1.0, 2.5, 4.0]),  # Neuron 0
    ...     np.array([0.5, 1.5, 2.5, 3.5]),  # Neuron 1
    ...     np.array([5.0, 8.0]),  # Neuron 2
    ... ]

    >>> # Compute view rates for all neurons
    >>> result = compute_view_rates(
    ...     env,
    ...     spike_times,
    ...     times,
    ...     positions,
    ...     headings,
    ...     gaze_model="fixed_distance",
    ...     view_distance=10.0,
    ...     n_jobs=2,  # Parallel spike binning
    ... )

    >>> # Access results
    >>> print(f"Number of neurons: {len(result)}")
    >>> print(f"Firing rates shape: {result.firing_rates.shape}")

    >>> # Iterate over neurons
    >>> for i, single in enumerate(result):
    ...     peak = single.peak_view_location()
    ...     print(f"Neuron {i}: peak view at ({peak[0]:.1f}, {peak[1]:.1f})")

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
    >>> result2 = compute_view_rates(env, spike_times_2d, times, positions, headings)

    References
    ----------
    .. [1] Rolls, E. T., et al. (1997). Spatial view cells in the primate
           hippocampus. European Journal of Neuroscience, 9(8), 1789-1794.
    """
    from neurospatial.encoding._backend import (
        SUPPORTED_BACKENDS,
        get_backend_name,
        is_jax_available,
    )
    from neurospatial.encoding._smoothing import (
        _validate_smoothing_parameters,
        smooth_rate_maps_batch,
    )
    from neurospatial.encoding._spikes import normalize_spike_times
    from neurospatial.encoding._validation import validate_trajectory
    from neurospatial.encoding._view_binning import bin_view_spike_trains

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

    # Validate gaze_model
    valid_gaze_models = {"fixed_distance", "ray_cast", "boundary"}
    if gaze_model not in valid_gaze_models:
        raise ValueError(
            f"Invalid gaze_model: '{gaze_model}'. "
            f"Must be one of {sorted(valid_gaze_models)}"
        )

    # Normalize spike times to canonical list-of-arrays format
    spike_times_list = normalize_spike_times(spike_times)
    n_neurons = len(spike_times_list)

    # Convert inputs to arrays
    times = np.asarray(times, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    headings = np.asarray(headings, dtype=np.float64)

    validate_trajectory(times, positions=positions, headings=headings)
    n_samples = len(times)

    # Validate gaze_offsets if provided
    if gaze_offsets is not None:
        gaze_offsets = np.asarray(gaze_offsets, dtype=np.float64)
        if len(gaze_offsets) != n_samples:
            raise ValueError(
                f"gaze_offsets length ({len(gaze_offsets)}) must match "
                f"times length ({n_samples})"
            )

    # Handle edge case: no neurons
    if n_neurons == 0:
        # Still need to compute view_occupancy for consistency
        from neurospatial.encoding._view_binning import compute_view_occupancy

        view_occupancy = compute_view_occupancy(
            env,
            times,
            positions,
            headings,
            gaze_model=gaze_model,
            view_distance=view_distance,
            gaze_offsets=gaze_offsets,
        )
        firing_rates_result: ArrayLike = np.empty((0, env.n_bins), dtype=np.float64)
        if resolved_backend == "jax" and is_jax_available():
            import jax.numpy as jnp

            firing_rates_result = jnp.asarray(firing_rates_result)
            view_occupancy = jnp.asarray(view_occupancy, dtype=jnp.float64)  # type: ignore[assignment]
        return ViewRatesResult(
            firing_rates=firing_rates_result,
            view_occupancy=view_occupancy,
            env=env,
            gaze_model=gaze_model,
            view_distance=view_distance,
            smoothing_method=smoothing_method,
            bandwidth=bandwidth,
        )

    # Bin spike trains by viewed location and compute view occupancy
    # bin_view_spike_trains returns (spike_counts, view_occupancy)
    spike_counts, view_occupancy = bin_view_spike_trains(
        env,
        spike_times_list,
        times,
        positions,
        headings,
        gaze_model=gaze_model,
        view_distance=view_distance,
        gaze_offsets=gaze_offsets,
        n_jobs=n_jobs,
    )

    # Apply batch smoothing to compute firing rates
    # smooth_rate_maps_batch dispatches to JAX or NumPy based on backend
    firing_rates = smooth_rate_maps_batch(
        env,
        spike_counts,
        view_occupancy,
        method=smoothing_method,
        bandwidth=bandwidth,
        min_occupancy=min_occupancy,
        backend=resolved_backend,
    )

    # Convert occupancy to JAX if JAX backend is selected
    # (firing_rates is already JAX from smooth_rate_maps_batch)
    if resolved_backend == "jax" and is_jax_available():
        import jax.numpy as jnp

        view_occupancy = jnp.asarray(view_occupancy, dtype=jnp.float64)  # type: ignore[assignment]

    # Return result
    return ViewRatesResult(
        firing_rates=firing_rates,
        view_occupancy=view_occupancy,
        env=env,
        gaze_model=gaze_model,
        view_distance=view_distance,
        smoothing_method=smoothing_method,
        bandwidth=bandwidth,
    )


# ==============================================================================
# Convenience Functions
# ==============================================================================


def is_spatial_view_cell(
    env: Environment,
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
    *,
    view_distance: float = 10.0,
    gaze_model: Literal["fixed_distance", "ray_cast", "boundary"] = "fixed_distance",
    smoothing_method: Literal[
        "diffusion_kde", "gaussian_kde", "binned"
    ] = "diffusion_kde",
    bandwidth: float = 5.0,
    min_info: float = 0.5,
) -> bool:
    """Quick check: Is this a spatial view cell?

    Convenience function for fast screening of neurons. Computes view field
    and checks if the neuron meets spatial view cell criteria based on
    view spatial information.

    For detailed metrics, use ``compute_view_rate()`` and inspect the result's
    methods (``is_view_cell()``, ``view_spatial_information()``, etc.).

    Parameters
    ----------
    env : Environment
        Spatial environment defining the discretization.
    spike_times : NDArray[np.float64], shape (n_spikes,)
        Times of spikes.
    times : NDArray[np.float64], shape (n_time,)
        Timestamps for each behavioral sample.
    positions : NDArray[np.float64], shape (n_time, 2)
        Animal positions in allocentric coordinates.
    headings : NDArray[np.float64], shape (n_time,)
        Animal heading at each time (radians).
    view_distance : float, default=10.0
        Distance for fixed_distance gaze model.
    gaze_model : {"fixed_distance", "ray_cast", "boundary"}, default="fixed_distance"
        Method for computing viewed location.
    smoothing_method : {"diffusion_kde", "gaussian_kde", "binned"}, default="diffusion_kde"
        Rate map smoothing method.
    bandwidth : float, default=5.0
        Smoothing bandwidth in environment units.
    min_info : float, default=0.5
        Minimum view spatial information threshold in bits/spike.

    Returns
    -------
    bool
        True if neuron passes spatial view cell criteria.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding.view import is_spatial_view_cell
    >>> positions = np.random.rand(1000, 2) * 100
    >>> env = Environment.from_samples(positions, bin_size=5.0)
    >>> times = np.linspace(0, 100, 1000)
    >>> headings = np.random.uniform(0, 2 * np.pi, 1000)
    >>> spike_times = np.random.uniform(0, 100, 50)
    >>> result = is_spatial_view_cell(env, spike_times, times, positions, headings)
    >>> type(result)
    <class 'bool'>

    See Also
    --------
    compute_view_rate : Full view rate computation
    ViewRateResult.is_view_cell : View cell classification on result object
    """
    try:
        result = compute_view_rate(
            env,
            spike_times,
            times,
            positions,
            headings,
            view_distance=view_distance,
            gaze_model=gaze_model,
            smoothing_method=smoothing_method,
            bandwidth=bandwidth,
        )
        return result.is_view_cell(min_info=min_info)
    except (ValueError, RuntimeError):
        return False
