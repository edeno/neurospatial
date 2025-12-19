"""NumPy core array operations for encoding computations.

This module provides the NumPy implementation of core array operations used
by encoding functions. These are pure array operations that work on shapes
like ``(n_neurons, n_bins)`` and ``(n_bins,)``.

Functions
---------
compute_firing_rate_single
    Convert spike counts and occupancy to firing rate for single neuron.
compute_firing_rates_batch
    Convert spike counts and occupancy to firing rates for multiple neurons.
smooth_rate_map_single
    Apply spatial smoothing to a single firing rate map.
smooth_rate_maps_batch
    Apply spatial smoothing to multiple firing rate maps.

Notes
-----
This module contains stubs for Phase 0 of the encoding refactor. Actual
implementations will be added in Phase 1 (Milestone 1).

The JAX equivalent of this module is ``_core_jax.py``, which provides
the same interface but uses JAX operations for GPU acceleration.

See Also
--------
neurospatial.encoding._backend : Backend selection infrastructure.
neurospatial.encoding._metrics : Shared metric implementations.
neurospatial.encoding._smoothing : Detailed smoothing implementations.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "compute_firing_rate_single",
    "compute_firing_rates_batch",
    "smooth_rate_map_single",
    "smooth_rate_maps_batch",
]


def compute_firing_rate_single(
    spike_counts: NDArray[np.float64],
    occupancy: NDArray[np.float64],
    *,
    min_occupancy: float = 0.0,
) -> NDArray[np.float64]:
    """Convert spike counts and occupancy to firing rate for single neuron.

    Computes firing rate as spike_counts / occupancy, with handling for
    low-occupancy bins.

    Parameters
    ----------
    spike_counts : NDArray[np.float64], shape (n_bins,)
        Number of spikes in each spatial bin.
    occupancy : NDArray[np.float64], shape (n_bins,)
        Time spent in each spatial bin (seconds).
    min_occupancy : float, default=0.0
        Minimum occupancy threshold. Bins with occupancy below this value
        will have NaN firing rate. Use 0.0 to mask only zero-occupancy bins.

    Returns
    -------
    NDArray[np.float64], shape (n_bins,)
        Firing rate in Hz (spikes per second). Bins with insufficient
        occupancy are set to NaN.

    Notes
    -----
    This function performs the core rate computation:

    .. math::

        r_i = \\frac{n_i}{t_i}

    where :math:`n_i` is the spike count and :math:`t_i` is the occupancy
    time for bin :math:`i`.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding._core_numpy import compute_firing_rate_single
    >>> spike_counts = np.array([0, 5, 10, 5, 0], dtype=np.float64)
    >>> occupancy = np.array([1.0, 1.0, 2.0, 1.0, 0.0], dtype=np.float64)
    >>> rate = compute_firing_rate_single(spike_counts, occupancy)  # doctest: +SKIP
    >>> rate  # doctest: +SKIP
    array([ 0.,  5.,  5.,  5., nan])
    """
    raise NotImplementedError(
        "compute_firing_rate_single is not yet implemented. "
        "This is a stub for Phase 0 of the encoding refactor."
    )


def compute_firing_rates_batch(
    spike_counts: NDArray[np.float64],
    occupancy: NDArray[np.float64],
    *,
    min_occupancy: float = 0.0,
) -> NDArray[np.float64]:
    """Convert spike counts and occupancy to firing rates for multiple neurons.

    Batch version of :func:`compute_firing_rate_single` that operates on
    multiple neurons efficiently.

    Parameters
    ----------
    spike_counts : NDArray[np.float64], shape (n_neurons, n_bins)
        Number of spikes in each spatial bin for each neuron.
    occupancy : NDArray[np.float64], shape (n_bins,)
        Time spent in each spatial bin (seconds). Shared across all neurons.
    min_occupancy : float, default=0.0
        Minimum occupancy threshold. Bins with occupancy below this value
        will have NaN firing rate across all neurons.

    Returns
    -------
    NDArray[np.float64], shape (n_neurons, n_bins)
        Firing rates in Hz (spikes per second) for each neuron.
        Bins with insufficient occupancy are set to NaN.

    Notes
    -----
    This is the batch equivalent of :func:`compute_firing_rate_single`.
    It broadcasts the occupancy array across all neurons for efficient
    computation.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding._core_numpy import compute_firing_rates_batch
    >>> spike_counts = np.array(
    ...     [
    ...         [0, 5, 10, 5, 0],
    ...         [5, 0, 5, 0, 5],
    ...     ],
    ...     dtype=np.float64,
    ... )
    >>> occupancy = np.array([1.0, 1.0, 2.0, 1.0, 0.0], dtype=np.float64)
    >>> rates = compute_firing_rates_batch(spike_counts, occupancy)  # doctest: +SKIP
    >>> rates.shape  # doctest: +SKIP
    (2, 5)
    """
    raise NotImplementedError(
        "compute_firing_rates_batch is not yet implemented. "
        "This is a stub for Phase 0 of the encoding refactor."
    )


def smooth_rate_map_single(
    firing_rate: NDArray[np.float64],
    adjacency: NDArray[np.float64],
    *,
    bandwidth: float = 5.0,
    method: Literal["diffusion_kde", "gaussian_kde", "binned"] = "diffusion_kde",
) -> NDArray[np.float64]:
    """Apply spatial smoothing to a single firing rate map.

    Parameters
    ----------
    firing_rate : NDArray[np.float64], shape (n_bins,)
        Unsmoothed firing rate map (Hz).
    adjacency : NDArray[np.float64], shape (n_bins, n_bins)
        Adjacency matrix encoding spatial connectivity between bins.
        Used for graph-based smoothing methods.
    bandwidth : float, default=5.0
        Smoothing bandwidth in physical units (e.g., cm). The interpretation
        depends on the smoothing method.
    method : {"diffusion_kde", "gaussian_kde", "binned"}, default="diffusion_kde"
        Smoothing method to apply:

        - ``"diffusion_kde"``: Graph-based boundary-aware kernel density
          estimation using heat diffusion on the connectivity graph.
        - ``"gaussian_kde"``: Standard Gaussian kernel smoothing, treating
          the rate map as a 2D image.
        - ``"binned"``: No smoothing (returns input unchanged).

    Returns
    -------
    NDArray[np.float64], shape (n_bins,)
        Smoothed firing rate map.

    Notes
    -----
    The ``"diffusion_kde"`` method is recommended for environments with
    irregular boundaries or obstacles, as it respects the connectivity
    structure and avoids bleeding across barriers.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding._core_numpy import smooth_rate_map_single
    >>> firing_rate = np.array([0.0, 1.0, 5.0, 1.0, 0.0], dtype=np.float64)
    >>> adjacency = np.eye(5, dtype=np.float64)  # Identity for simplicity
    >>> smoothed = smooth_rate_map_single(
    ...     firing_rate, adjacency, bandwidth=2.0
    ... )  # doctest: +SKIP
    """
    raise NotImplementedError(
        "smooth_rate_map_single is not yet implemented. "
        "This is a stub for Phase 0 of the encoding refactor."
    )


def smooth_rate_maps_batch(
    firing_rates: NDArray[np.float64],
    adjacency: NDArray[np.float64],
    *,
    bandwidth: float = 5.0,
    method: Literal["diffusion_kde", "gaussian_kde", "binned"] = "diffusion_kde",
) -> NDArray[np.float64]:
    """Apply spatial smoothing to multiple firing rate maps.

    Batch version of :func:`smooth_rate_map_single` that operates on
    multiple neurons efficiently.

    Parameters
    ----------
    firing_rates : NDArray[np.float64], shape (n_neurons, n_bins)
        Unsmoothed firing rate maps for each neuron (Hz).
    adjacency : NDArray[np.float64], shape (n_bins, n_bins)
        Adjacency matrix encoding spatial connectivity between bins.
        Shared across all neurons.
    bandwidth : float, default=5.0
        Smoothing bandwidth in physical units (e.g., cm).
    method : {"diffusion_kde", "gaussian_kde", "binned"}, default="diffusion_kde"
        Smoothing method to apply. See :func:`smooth_rate_map_single` for
        detailed descriptions of each method.

    Returns
    -------
    NDArray[np.float64], shape (n_neurons, n_bins)
        Smoothed firing rate maps for each neuron.

    Notes
    -----
    This function applies the same smoothing kernel to all neurons,
    which is more efficient than calling :func:`smooth_rate_map_single`
    in a loop when the adjacency matrix and bandwidth are shared.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding._core_numpy import smooth_rate_maps_batch
    >>> firing_rates = np.array(
    ...     [
    ...         [0.0, 1.0, 5.0, 1.0, 0.0],
    ...         [1.0, 3.0, 1.0, 0.0, 0.0],
    ...     ],
    ...     dtype=np.float64,
    ... )
    >>> adjacency = np.eye(5, dtype=np.float64)
    >>> smoothed = smooth_rate_maps_batch(
    ...     firing_rates, adjacency, bandwidth=2.0
    ... )  # doctest: +SKIP
    >>> smoothed.shape  # doctest: +SKIP
    (2, 5)
    """
    raise NotImplementedError(
        "smooth_rate_maps_batch is not yet implemented. "
        "This is a stub for Phase 0 of the encoding refactor."
    )
