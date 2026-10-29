"""Likelihood functions for Bayesian decoding.

This module provides numerically stable likelihood computation for
Bayesian position decoding from neural activity.

Functions
---------
log_poisson_likelihood : Compute log-likelihood under Poisson model (primary)
    Numerically stable log-likelihood computation. Use this for all
    decoding pipelines.


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
from scipy import sparse

from neurospatial._exceptions import _format_error
from neurospatial.decoding._time_bins import validate_dt


def log_poisson_likelihood(
    spike_counts: NDArray[np.int64],
    encoding_models: NDArray[np.float64],
    *,
    dt: float,
    min_rate: float = 1e-10,
) -> NDArray[np.float64]:
    """Compute the Poisson log-likelihood of spike counts at every bin.

    Computes the log-likelihood up to an additive constant per time bin:

        log P(spikes | position) = sum_i [n_i * log(lambda_i * dt) - lambda_i * dt]
                                   - sum_i log(n_i!)

    where i indexes neurons, n_i is the spike count for neuron i, and
    lambda_i is the firing rate of neuron i at each spatial bin. The
    ``log(n_i!)`` term does not depend on position and is omitted.

    Parameters
    ----------
    spike_counts : NDArray[np.int64], shape (n_time_bins, n_neurons)
        Spike counts per neuron per time bin. Must be 2-D. A 1-D array of
        shape ``(n_neurons,)`` is rejected (it would silently collapse the
        time axis); for a single time bin pass ``spike_counts[np.newaxis, :]``.
    encoding_models : NDArray[np.float64], shape (n_neurons, n_bins)
        Firing rate maps in Hz. Every rate must be finite and non-negative;
        rates below ``min_rate`` are raised to ``min_rate``.
    dt : float
        Time bin width in seconds.
    min_rate : float, default=1e-10
        Rate floor in Hz, so a zero rate gives a large finite penalty instead
        of ``log(0)``.

    Returns
    -------
    log_likelihood : NDArray[np.float64], shape (n_time_bins, n_bins)
        Log-likelihood up to an additive constant per time bin.

    Raises
    ------
    ValueError
        If ``dt`` is not a finite number > 0 (non-numeric, ``bool``, ``NaN``,
        ``Inf``, or ``<= 0`` all raise), or if ``min_rate`` is not positive.
        If ``spike_counts`` is not 2-D ``(n_time_bins, n_neurons)`` or
        ``encoding_models`` is not 2-D ``(n_neurons, n_bins)``.
        If the neuron axis of ``spike_counts`` (axis 1) does not match the
        neuron axis of ``encoding_models`` (axis 0).
        If ``encoding_models`` contains a NaN, infinite or negative rate.

    Notes
    -----
    The returned values differ from the full Poisson log-probability by a
    constant per time bin, which cancels in :func:`normalize_to_posterior`.
    They cannot be compared across different spike patterns.

    Rate maps with unsupported bins stored as NaN cannot be passed here. Pass
    the rate result itself to
    :func:`~neurospatial.decoding.decode_position`, which excludes those bins,
    or zero-fill the array and pass
    ``prior=rates.spatial_support * env.bin_sizes``.

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
    normalize_to_posterior : Convert log-likelihood to posterior
    """
    dt = validate_dt(dt)
    if min_rate <= 0:
        raise ValueError(
            f"min_rate must be positive, got {min_rate}. "
            f"This floor prevents log(0) in likelihood computation."
        )
    spike_counts = np.asarray(spike_counts)
    encoding_models = np.asarray(encoding_models)

    # A 1-D (n_neurons,) array would matmul to (n_bins,), silently dropping
    # the time axis.
    if spike_counts.ndim != 2:
        raise ValueError(
            f"spike_counts must be 2-D with shape (n_time_bins, n_neurons), "
            f"got ndim={spike_counts.ndim} with shape {spike_counts.shape}. "
            f"For a single time bin pass shape (1, n_neurons), e.g. "
            f"`spike_counts[np.newaxis, :]`."
        )
    if encoding_models.ndim != 2:
        raise ValueError(
            f"encoding_models must be 2-D with shape (n_neurons, n_bins), "
            f"got ndim={encoding_models.ndim} with shape {encoding_models.shape}."
        )
    if spike_counts.shape[1] != encoding_models.shape[0]:
        raise ValueError(
            _format_error(
                f"Neuron-count mismatch: spike_counts has {spike_counts.shape[1]} neurons (axis 1) but encoding_models has {encoding_models.shape[0]} neurons (axis 0). These must agree for the Poisson likelihood.",
                fix="build spike_counts and encoding_models from the same unit list, in the same order",
                why="Why: the neuron axis must align counts with the corresponding firing-rate model.",
            )
        )
    invalid = ~np.isfinite(encoding_models) | (encoding_models < 0)
    if invalid.any():
        n_nan = int(np.isnan(encoding_models).sum())
        raise ValueError(
            _format_error(
                f"encoding_models has {int(invalid.sum())} rate(s) that are not "
                f"finite and non-negative ({n_nan} NaN).",
                why="the Poisson likelihood needs a rate estimate for every "
                "neuron in every bin; NaN usually marks a bin without training "
                "data, which must be excluded from the posterior, not scored.",
                fix="decode with decode_position(env, counts, rates, dt=dt), "
                "passing the rate result itself so unsupported bins are "
                "excluded, or zero-fill the array and pass "
                "prior=rates.spatial_support * env.bin_sizes.",
            )
        )

    expected_counts = np.maximum(encoding_models, min_rate) * dt
    log_expected = np.log(expected_counts)

    # A sparse CSR product skips the zero counts, which dominate binned spike
    # trains; its construction overhead pays off only for mostly-zero
    # matrices that are not tiny. Both products give the same values up to
    # floating-point summation order.
    n_entries = spike_counts.size
    if n_entries > 1000 and np.count_nonzero(spike_counts) < 0.2 * n_entries:
        spike_term = np.asarray(
            sparse.csr_matrix(spike_counts.astype(np.float64)) @ log_expected
        )
    else:
        spike_term = spike_counts.astype(np.float64) @ log_expected

    log_likelihood = spike_term - expected_counts.sum(axis=0)
    return cast("NDArray[np.float64]", log_likelihood)
