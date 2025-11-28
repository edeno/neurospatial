"""Likelihood functions for Bayesian decoding.

This module provides numerically stable likelihood computation for
Bayesian position decoding from neural activity.

Functions
---------
log_poisson_likelihood : Compute log-likelihood under Poisson model (primary)
    Numerically stable log-likelihood computation. Use this for all
    decoding pipelines.

poisson_likelihood : Compute likelihood in probability space (thin wrapper)
    Convenience function that exponentiates log-likelihood with underflow
    protection. Prefer log_poisson_likelihood + normalize_to_posterior.

Notes
-----
The Poisson likelihood model assumes spikes are generated as a Poisson
process with rate lambda(x) at position x:

    P(n spikes | position) = (lambda * dt)^n * exp(-lambda * dt) / n!

In log-space (dropping the constant n! term):

    log P(n | x) proportional to n * log(lambda(x) * dt) - lambda(x) * dt

The n! term is omitted because it is constant across positions and
cancels during posterior normalization.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray


def log_poisson_likelihood(
    spike_counts: NDArray[np.int64],
    encoding_models: NDArray[np.float64],
    dt: float,
    *,
    min_rate: float = 1e-10,
) -> NDArray[np.float64]:
    """Compute log Poisson likelihood (numerically stable).

    This is the **primary likelihood function** and should be used for all
    decoding pipelines.

    Computes the log-likelihood up to an additive constant:

        log P(spikes | position) proportional to
            sum_i [n_i * log(lambda_i * dt) - lambda_i * dt]

    where i indexes neurons, n_i is the spike count for neuron i, and
    lambda_i is the firing rate of neuron i at each spatial bin.

    Parameters
    ----------
    spike_counts : NDArray[np.int64], shape (n_time_bins, n_neurons)
        Spike counts per neuron per time bin.
    encoding_models : NDArray[np.float64], shape (n_neurons, n_bins)
        Firing rate maps (place fields) in Hz. Typical values: 0-50 Hz.
        Values are clipped to [min_rate, inf) internally.
    dt : float
        Time bin width in seconds. Typical values: 0.001-0.1s.
        Note: lambda*dt should typically be in [0, 5] for numerical stability.
    min_rate : float, default=1e-10
        Minimum firing rate floor to avoid log(0).

    Returns
    -------
    log_likelihood : NDArray[np.float64], shape (n_time_bins, n_bins)
        Log-likelihood up to an additive constant per time bin.

    Notes
    -----
    The -log(n_i!) term is **omitted** because it is constant across positions
    for each time bin. Since we normalize to posterior via softmax, this constant
    cancels out and omitting it saves O(n_neurons) log-gamma evaluations per
    time bin. This optimization is especially beneficial for large populations.

    The returned values are log-likelihoods up to an additive constant per time
    bin. They are suitable for `normalize_to_posterior()` but NOT for model
    comparison across different spike patterns.

    For extremely high firing rates (lambda*dt >> 10), consider increasing dt
    or normalizing encoding_models to avoid large exponents.

    Examples
    --------
    >>> spike_counts = np.array([[0, 1], [2, 0]], dtype=np.int64)
    >>> encoding_models = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    >>> ll = log_poisson_likelihood(spike_counts, encoding_models, dt=0.025)
    >>> ll.shape
    (2, 3)
    >>> bool(np.isfinite(ll).all())
    True

    See Also
    --------
    poisson_likelihood : Thin wrapper returning probability-space likelihoods
    normalize_to_posterior : Convert log-likelihood to posterior
    """
    # Ensure inputs are proper arrays
    spike_counts = np.asarray(spike_counts)
    encoding_models = np.asarray(encoding_models)

    # Clip rates to avoid log(0)
    # Shape: (n_neurons, n_bins)
    clipped_rates = np.maximum(encoding_models, min_rate)

    # Compute expected spike count (lambda * dt) for each neuron at each bin
    # Shape: (n_neurons, n_bins)
    expected_counts = clipped_rates * dt

    # Log of expected counts for the spike count term
    # Shape: (n_neurons, n_bins)
    log_expected = np.log(expected_counts)

    # Compute log-likelihood:
    # sum_i [n_i * log(lambda_i * dt) - lambda_i * dt]
    #
    # For spike_counts shape (n_time_bins, n_neurons) and
    # log_expected shape (n_neurons, n_bins), we need:
    # - spike_counts @ log_expected -> (n_time_bins, n_bins) [spike count term]
    # - sum over neurons of expected_counts -> (n_bins,) [rate penalty term]

    # Spike count term: n_i * log(lambda_i * dt)
    # Matrix multiplication: (n_time_bins, n_neurons) @ (n_neurons, n_bins)
    # Result: (n_time_bins, n_bins)
    spike_term = spike_counts @ log_expected

    # Rate penalty term: -sum_i lambda_i * dt
    # Sum over neurons for each bin, broadcast across time bins
    # Shape: (n_bins,)
    rate_penalty = -np.sum(expected_counts, axis=0)

    # Total log-likelihood: spike_term + rate_penalty (broadcast)
    # Shape: (n_time_bins, n_bins)
    log_likelihood = spike_term + rate_penalty

    return cast("NDArray[np.float64]", log_likelihood)


def poisson_likelihood(
    spike_counts: NDArray[np.int64],
    encoding_models: NDArray[np.float64],
    dt: float,
    *,
    min_rate: float = 1e-10,
) -> NDArray[np.float64]:
    """Compute Poisson likelihood in probability space (thin wrapper).

    **Warning**: This function can underflow/overflow for realistic spike
    trains with large populations. Prefer `log_poisson_likelihood` +
    `normalize_to_posterior` for decoding.

    This is implemented as::

        log_ll = log_poisson_likelihood(spike_counts, encoding_models, dt, min_rate)
        return np.exp(log_ll - log_ll.max(axis=1, keepdims=True))

    The likelihoods are normalized per row to prevent underflow, but this
    means they are NOT true probabilities and should only be used for
    visualization or when probability-space is explicitly required.

    Parameters
    ----------
    spike_counts : NDArray[np.int64], shape (n_time_bins, n_neurons)
        Spike counts per neuron per time bin.
    encoding_models : NDArray[np.float64], shape (n_neurons, n_bins)
        Firing rate maps (place fields) in Hz.
    dt : float
        Time bin width in seconds.
    min_rate : float, default=1e-10
        Minimum firing rate floor to avoid log(0).

    Returns
    -------
    likelihood : NDArray[np.float64], shape (n_time_bins, n_bins)
        Likelihood ratios (normalized per time bin to prevent underflow).
        Maximum value per row is 1.0.

    Notes
    -----
    The normalization per row means:

    - Maximum likelihood per row is exactly 1.0
    - Other values are relative likelihoods (ratios)
    - NOT true probabilities (don't sum to 1)
    - Ranking of bins is preserved (same argmax as log version)

    Examples
    --------
    >>> spike_counts = np.array([[0, 1], [2, 0]], dtype=np.int64)
    >>> encoding_models = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    >>> likelihood = poisson_likelihood(spike_counts, encoding_models, dt=0.025)
    >>> likelihood.max(axis=1)  # Maximum per row is 1.0
    array([1., 1.])
    >>> bool((likelihood >= 0).all())
    True

    See Also
    --------
    log_poisson_likelihood : Primary function for log-likelihood computation
    normalize_to_posterior : Proper normalization to posterior probabilities
    """
    # Get log-likelihoods
    log_ll = log_poisson_likelihood(
        spike_counts, encoding_models, dt, min_rate=min_rate
    )

    # Shift by max per row for numerical stability, then exponentiate
    log_ll_shifted = log_ll - log_ll.max(axis=1, keepdims=True)
    likelihood = np.exp(log_ll_shifted)

    return cast("NDArray[np.float64]", likelihood)
