"""Directional rate computation for head direction cells.

This module provides result classes and compute functions for directional firing
rate analysis, specifically for head direction (HD) cells. HD cells fire
preferentially when an animal faces a particular direction.

Unlike spatial encoding, directional encoding does not require an Environment.
Directional data is represented by bin centers (angles in radians) rather than
spatial coordinates.

Result Classes
--------------
DirectionalRateResult
    Single-neuron directional tuning curve with convenience methods
DirectionalRatesResult
    Multi-neuron directional tuning curves with batch methods and iteration

Compute Functions (to be implemented in Tasks 3.8-3.9)
------------------------------------------------------
compute_directional_rate
    Compute directional firing rate for one neuron
compute_directional_rates
    Compute directional firing rates for multiple neurons

Examples
--------
>>> import numpy as np
>>> from neurospatial.encoding.directional import DirectionalRateResult

>>> # Create result (typically from compute_directional_rate)
>>> n_bins = 60  # 6 degree resolution
>>> bin_centers = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)
>>> firing_rate = np.random.rand(n_bins) * 10  # Hz
>>> occupancy = np.ones(n_bins) * 0.5  # seconds per bin
>>> result = DirectionalRateResult(
...     firing_rate=firing_rate,
...     occupancy=occupancy,
...     bin_centers=bin_centers,
...     bin_size=np.pi / 30,  # 6 degrees in radians
...     smoothing_sigma=None,
... )

See Also
--------
neurospatial.encoding.head_direction : Legacy head direction analysis module
neurospatial.stats.circular : Circular statistics utilities
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "DirectionalRateResult",
    "DirectionalRatesResult",
]


@dataclass(frozen=True)
class DirectionalRateResult:
    """Result of directional rate computation for a single neuron.

    This class wraps a directional tuning curve with its associated metadata
    (occupancy, bin centers, bin size, smoothing parameters). Unlike spatial
    results, directional results do not require an Environment since head
    direction is independent of spatial position.

    Parameters
    ----------
    firing_rate : ArrayLike
        Firing rate by direction in Hz. Shape is (n_bins,) where n_bins is
        the number of angular bins (e.g., 60 bins for 6° resolution).
    occupancy : ArrayLike
        Time spent at each direction in seconds. Shape is (n_bins,).
    bin_centers : ArrayLike
        Center of each angular bin in radians [0, 2π). Shape is (n_bins,).
    bin_size : float
        Width of each angular bin in radians.
    smoothing_sigma : float or None
        Gaussian smoothing bandwidth in radians, or None if unsmoothed.

    Attributes
    ----------
    firing_rate : ArrayLike
        Firing rate by direction in Hz. Shape is (n_bins,).
    occupancy : ArrayLike
        Time at each direction in seconds. Shape is (n_bins,).
    bin_centers : ArrayLike
        Angular bin centers in radians. Shape is (n_bins,).
    bin_size : float
        Angular bin width in radians.
    smoothing_sigma : float or None
        Smoothing bandwidth in radians, or None.

    Notes
    -----
    This is a frozen dataclass (immutable). All fields are set at construction
    and cannot be modified afterward.

    **No Environment dependency**: Unlike SpatialRateResult, directional results
    do not store an Environment. Head direction is a 1D circular variable that
    does not depend on spatial position. The bin_centers field provides the
    angular coordinates needed for analysis and plotting.

    Convenience methods (plot, preferred_direction, mean_vector_length, etc.)
    are implemented in Tasks 3.2-3.4.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding.directional import DirectionalRateResult

    >>> # Create a simple tuning curve
    >>> n_bins = 60
    >>> bin_centers = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)
    >>> # Von Mises-like tuning with peak at π/2 (90°)
    >>> firing_rate = 10.0 * np.exp(2.0 * (np.cos(bin_centers - np.pi / 2) - 1))
    >>> occupancy = np.ones(n_bins) * 0.5

    >>> result = DirectionalRateResult(
    ...     firing_rate=firing_rate,
    ...     occupancy=occupancy,
    ...     bin_centers=bin_centers,
    ...     bin_size=np.pi / 30,
    ...     smoothing_sigma=None,
    ... )

    >>> # Access fields
    >>> result.firing_rate.shape
    (60,)
    >>> result.bin_size  # 6 degrees in radians
    0.10471975511965977

    See Also
    --------
    DirectionalRatesResult : Batch version for multiple neurons
    compute_directional_rate : Function to compute this result
    """

    firing_rate: ArrayLike
    occupancy: ArrayLike
    bin_centers: ArrayLike
    bin_size: float
    smoothing_sigma: float | None


@dataclass(frozen=True)
class DirectionalRatesResult:
    """Result of directional rate computation for multiple neurons.

    This class wraps directional tuning curves for a population of neurons
    with shared occupancy and bin structure. It provides iteration over
    individual neuron results.

    Parameters
    ----------
    firing_rates : ArrayLike
        Firing rates by direction in Hz. Shape is (n_neurons, n_bins) where
        n_bins is the number of angular bins. Each row is a single neuron's
        tuning curve.
    occupancy : ArrayLike
        Time spent at each direction in seconds. Shape is (n_bins,). Shared
        across all neurons.
    bin_centers : ArrayLike
        Center of each angular bin in radians [0, 2π). Shape is (n_bins,).
    bin_size : float
        Width of each angular bin in radians.
    smoothing_sigma : float or None
        Gaussian smoothing bandwidth in radians, or None if unsmoothed.

    Attributes
    ----------
    firing_rates : ArrayLike
        Firing rates by direction in Hz. Shape is (n_neurons, n_bins).
    occupancy : ArrayLike
        Time at each direction in seconds. Shape is (n_bins,).
    bin_centers : ArrayLike
        Angular bin centers in radians. Shape is (n_bins,).
    bin_size : float
        Angular bin width in radians.
    smoothing_sigma : float or None
        Smoothing bandwidth in radians, or None.

    Notes
    -----
    This is a frozen dataclass (immutable). All fields are set at construction
    and cannot be modified afterward.

    **Iteration Support**:

    This class supports len(), indexing, and iteration:

    - `len(result)`: Number of neurons
    - `result[i]`: Returns `DirectionalRateResult` for neuron i
    - `for r in result`: Iterates over single-neuron results

    Batch methods (preferred_directions, mean_vector_lengths, etc.) are
    implemented in Tasks 3.5-3.6.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding.directional import DirectionalRatesResult

    >>> # Create batch result (5 neurons)
    >>> n_neurons = 5
    >>> n_bins = 60
    >>> bin_centers = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)
    >>> firing_rates = np.random.rand(n_neurons, n_bins) * 10
    >>> occupancy = np.ones(n_bins) * 0.5

    >>> result = DirectionalRatesResult(
    ...     firing_rates=firing_rates,
    ...     occupancy=occupancy,
    ...     bin_centers=bin_centers,
    ...     bin_size=np.pi / 30,
    ...     smoothing_sigma=None,
    ... )

    >>> # Access fields
    >>> result.firing_rates.shape
    (5, 60)
    >>> len(result)
    5

    >>> # Index to get single-neuron result
    >>> single = result[0]
    >>> type(single).__name__
    'DirectionalRateResult'

    >>> # Iterate over neurons
    >>> for i, r in enumerate(result):
    ...     peak_rate = np.max(r.firing_rate)
    ...     print(f"Neuron {i}: peak = {peak_rate:.2f} Hz")

    See Also
    --------
    DirectionalRateResult : Single-neuron version
    compute_directional_rates : Function to compute this result
    """

    firing_rates: ArrayLike
    occupancy: ArrayLike
    bin_centers: ArrayLike
    bin_size: float
    smoothing_sigma: float | None

    def __len__(self) -> int:
        """Return number of neurons.

        Returns
        -------
        int
            Number of neurons (first dimension of firing_rates).
        """
        rates: NDArray[np.float64] = np.asarray(self.firing_rates)
        return int(rates.shape[0])

    def __getitem__(self, idx: int) -> DirectionalRateResult:
        """Get single-neuron result by index.

        Parameters
        ----------
        idx : int
            Neuron index (0-indexed).

        Returns
        -------
        DirectionalRateResult
            Result for the specified neuron with shared occupancy,
            bin_centers, bin_size, and smoothing_sigma.

        Examples
        --------
        >>> single = result[0]
        >>> single.firing_rate.shape
        (n_bins,)
        """
        rates: NDArray[np.float64] = np.asarray(self.firing_rates)
        return DirectionalRateResult(
            firing_rate=rates[idx],
            occupancy=self.occupancy,
            bin_centers=self.bin_centers,
            bin_size=self.bin_size,
            smoothing_sigma=self.smoothing_sigma,
        )

    def __iter__(self) -> Iterator[DirectionalRateResult]:
        """Iterate over single-neuron results.

        Yields
        ------
        DirectionalRateResult
            Result for each neuron in order.

        Examples
        --------
        >>> for result in results:
        ...     peak = np.max(result.firing_rate)
        ...     print(f"Peak rate: {peak:.2f} Hz")
        """
        for i in range(len(self)):
            yield self[i]
