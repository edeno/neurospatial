"""NumPy core array operations for encoding computations.

This module provides the NumPy implementation of core array operations used
by encoding functions. These are pure array operations that work on shapes
like ``(n_neurons, n_bins)`` and ``(n_bins,)``.

Functions
---------
compute_firing_rates_batch
    Convert spike counts and occupancy to firing rates for multiple neurons.
spatial_information_single, spatial_information_batch
    Skaggs spatial information (bits per spike).
sparsity_single, sparsity_batch
    Skaggs sparsity.

Notes
-----
This module provides the NumPy backend for the encoding core's array
operations (firing rates, spatial information, sparsity).

The JAX equivalent of this module is ``_core_jax.py``, which provides
the same interface but uses JAX operations for GPU acceleration.

See Also
--------
neurospatial.encoding._backend : Backend selection infrastructure.
neurospatial.encoding._metrics : Shared metric implementations.
neurospatial.encoding._smoothing : Rate-map smoothing.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "compute_firing_rates_batch",
    "sparsity_batch",
    "sparsity_single",
    "spatial_information_batch",
    "spatial_information_single",
]


def compute_firing_rates_batch(
    spike_counts: NDArray[np.float64],
    occupancy: NDArray[np.float64],
    *,
    min_occupancy: float = 0.0,
) -> NDArray[np.float64]:
    """Convert spike counts and occupancy to firing rates for multiple neurons.

    Computes each rate as spike_counts / occupancy, NaN where the occupancy
    is zero or below ``min_occupancy``.

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
    This function performs the core rate computation:

    .. math::

        r_{ui} = \\frac{n_{ui}}{t_i}

    where :math:`n_{ui}` is neuron :math:`u`'s spike count and :math:`t_i`
    the occupancy time for bin :math:`i`. The occupancy array broadcasts
    across all neurons.

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
    >>> rates = compute_firing_rates_batch(spike_counts, occupancy)
    >>> rates.shape
    (2, 5)
    """
    spike_counts = np.asarray(spike_counts, dtype=np.float64)
    occupancy = np.asarray(occupancy, dtype=np.float64)

    # Compute firing rate with safe division (broadcasting over neurons)
    with np.errstate(divide="ignore", invalid="ignore"):
        firing_rates = np.where(
            occupancy > 0,
            spike_counts / occupancy,
            np.nan,
        )

    # Apply min_occupancy threshold
    if min_occupancy > 0:
        firing_rates = np.where(occupancy >= min_occupancy, firing_rates, np.nan)

    return firing_rates.astype(np.float64)


def spatial_information_single(
    firing_rate: NDArray[np.float64],
    occupancy: NDArray[np.float64],
    *,
    base: float = 2.0,
) -> float:
    """Compute Skaggs spatial information (bits per spike) for a single neuron.

    NumPy mirror of :func:`neurospatial.encoding._core_jax.spatial_information_single`.
    Both implementations follow the same algorithm: exclude non-finite
    rate/occupancy pairs, normalize occupancy, return NaN for zero total
    occupancy or a non-positive mean rate (undefined bits per spike), clamp to
    non-negative.

    Parameters
    ----------
    firing_rate : NDArray[np.float64], shape (n_bins,)
        Firing rate map in Hz. Bins with non-finite rates are excluded.
    occupancy : NDArray[np.float64], shape (n_bins,)
        Time spent in each bin. Bins with non-finite occupancy are excluded.
    base : float, default=2.0
        Logarithm base. Use 2.0 for bits, ``np.e`` for nats.

    Returns
    -------
    float
        Spatial information in bits per spike (if ``base=2.0``).
        NaN when total occupancy is zero or the mean rate is non-positive.
    """
    # One neuron is a population of one; ravel so any map shape is one row.
    firing_rate = np.asarray(firing_rate, dtype=np.float64).reshape(1, -1)
    occupancy = np.asarray(occupancy, dtype=np.float64).ravel()
    return float(spatial_information_batch(firing_rate, occupancy, base=base)[0])


def spatial_information_batch(
    firing_rates: NDArray[np.float64],
    occupancy: NDArray[np.float64],
    *,
    base: float = 2.0,
) -> NDArray[np.float64]:
    """Compute Skaggs spatial information for a population.

    NumPy mirror of :func:`neurospatial.encoding._core_jax.spatial_information_batch`.
    Vectorized one-pass implementation over ``(n_neurons, n_bins)``.
    """
    firing_rates = np.asarray(firing_rates, dtype=np.float64)
    occupancy = np.asarray(occupancy, dtype=np.float64)

    n_neurons = firing_rates.shape[0]
    if n_neurons == 0:
        return np.empty(0, dtype=np.float64)

    # Each neuron may have a different set of observed spatial bins.
    valid = np.isfinite(firing_rates) & np.isfinite(occupancy)
    firing_rates_clean = np.where(valid, firing_rates, 0.0)
    occupancy_clean = np.where(valid, occupancy, 0.0)

    occ_sum = occupancy_clean.sum(axis=1, keepdims=True)
    occ_prob = occupancy_clean / np.where(occ_sum > 0, occ_sum, 1.0)

    mean_rates = (occ_prob * firing_rates_clean).sum(axis=1)
    valid_neuron = mean_rates > 0
    safe_mean = np.where(valid_neuron, mean_rates, 1.0)[:, np.newaxis]
    valid_bin = (occ_prob > 0) & (firing_rates_clean > 0)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(
            valid_bin & valid_neuron[:, np.newaxis],
            firing_rates_clean / safe_mean,
            1.0,
        )
        # Subnormal firing_rate / mean_rate can underflow to 0; log(0) = -inf
        # and 0 * -inf = NaN. Clamp to 1 (log=0) for those bins.
        ratio = np.where(ratio > 0, ratio, 1.0)
        log_ratio = np.log(ratio) / np.log(base)
        contributions = np.where(
            valid_bin & valid_neuron[:, np.newaxis],
            occ_prob * ratio * log_ratio,
            0.0,
        )

    info = np.maximum(contributions.sum(axis=1), 0.0)
    return np.where(valid_neuron, info, np.nan).astype(np.float64, copy=False)


def sparsity_single(
    firing_rate: NDArray[np.float64],
    occupancy: NDArray[np.float64],
) -> float:
    """Compute sparsity for a single neuron.

    NumPy mirror of :func:`neurospatial.encoding._core_jax.sparsity_single`.
    Returns NaN for a silent map or zero total occupancy; clamps to ``[0, 1]``.
    """
    # One neuron is a population of one; ravel so any map shape is one row.
    firing_rate = np.asarray(firing_rate, dtype=np.float64).reshape(1, -1)
    occupancy = np.asarray(occupancy, dtype=np.float64).ravel()
    return float(sparsity_batch(firing_rate, occupancy)[0])


def sparsity_batch(
    firing_rates: NDArray[np.float64],
    occupancy: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute sparsity for a population.

    NumPy mirror of :func:`neurospatial.encoding._core_jax.sparsity_batch`.
    """
    firing_rates = np.asarray(firing_rates, dtype=np.float64)
    occupancy = np.asarray(occupancy, dtype=np.float64)

    n_neurons = firing_rates.shape[0]
    if n_neurons == 0:
        return np.empty(0, dtype=np.float64)

    valid = np.isfinite(firing_rates) & np.isfinite(occupancy)
    firing_rates_clean = np.where(valid, firing_rates, 0.0)
    occupancy_clean = np.where(valid, occupancy, 0.0)

    occ_sum = occupancy_clean.sum(axis=1, keepdims=True)
    occ_prob = occupancy_clean / np.where(occ_sum > 0, occ_sum, 1.0)

    # Sparsity is unchanged by rescaling rates; dividing by each neuron's
    # peak keeps the squared moments from underflowing or overflowing.
    peak = np.abs(firing_rates_clean).max(axis=1, keepdims=True, initial=0.0)
    firing_rates_clean = firing_rates_clean / np.where(peak > 0, peak, 1.0)
    numerator = (occ_prob * firing_rates_clean).sum(axis=1) ** 2
    denominator = (occ_prob * firing_rates_clean**2).sum(axis=1)
    sparsity_values = np.where(
        denominator > 0,
        numerator / np.where(denominator > 0, denominator, 1.0),
        np.nan,
    )
    return np.clip(sparsity_values, 0.0, 1.0).astype(np.float64, copy=False)
