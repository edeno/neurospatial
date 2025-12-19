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

Compute Functions (to be implemented in Tasks 4.7-4.8)
------------------------------------------------------
compute_view_rate
    Compute view field for one neuron
compute_view_rates
    Compute view fields for multiple neurons

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
neurospatial.encoding.spatial_view : Legacy spatial view module
neurospatial.ops.visibility : Visibility and gaze computation
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from neurospatial import Environment

__all__ = [
    "ViewRateResult",
    "ViewRatesResult",
    "compute_view_rate",
    "compute_view_rates",
]


@dataclass(frozen=True)
class ViewRateResult:
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

    Convenience methods (plot, metrics, classification) are implemented in
    Tasks 4.2-4.3.

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


@dataclass(frozen=True)
class ViewRatesResult:
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

    Batch methods (plot, metrics, classification) are implemented in
    Tasks 4.4-4.5.

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


# =============================================================================
# Compute Functions (stubs for Tasks 4.7-4.8)
# =============================================================================


def compute_view_rate() -> ViewRateResult:
    """Compute view field for one neuron.

    To be implemented in Task 4.7.

    Raises
    ------
    NotImplementedError
        This function is not yet implemented.
    """
    raise NotImplementedError("compute_view_rate will be implemented in Task 4.7")


def compute_view_rates() -> ViewRatesResult:
    """Compute view fields for multiple neurons.

    To be implemented in Task 4.8.

    Raises
    ------
    NotImplementedError
        This function is not yet implemented.
    """
    raise NotImplementedError("compute_view_rates will be implemented in Task 4.8")
