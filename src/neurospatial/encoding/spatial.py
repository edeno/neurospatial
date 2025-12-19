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

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurospatial.encoding._base import SpatialResultMixin

if TYPE_CHECKING:
    from neurospatial import Environment

__all__ = [
    "SpatialRateResult",
    "SpatialRatesResult",
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
