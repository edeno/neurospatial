"""Estimate functions for Bayesian decoding results.

This module provides standalone functions for computing derived quantities
from posterior distributions. These functions mirror the corresponding
properties on DecodingResult but operate on raw arrays, enabling use
without creating a DecodingResult container.

Functions
---------
map_estimate : Maximum a posteriori bin index
map_position : MAP position in environment coordinates
mean_position : Posterior mean position (expected value)
entropy : Posterior entropy in bits (uncertainty measure)
credible_region : Highest posterior density region

Notes
-----
Function names mirror DecodingResult property names for consistency.
Users can use either the result properties or these standalone functions
interchangeably.

Examples
--------
>>> from neurospatial.decoding.estimates import map_estimate, entropy
>>> import numpy as np
>>>
>>> # Uniform posterior (maximum uncertainty)
>>> posterior = np.ones((10, 100)) / 100
>>> bins = map_estimate(posterior)
>>> ent = entropy(posterior)
>>> print(f"Entropy: {ent[0]:.2f} bits")  # log2(100) = 6.64 bits
Entropy: 6.64 bits
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from neurospatial.environment import Environment


def map_estimate(
    posterior: NDArray[np.float64],
) -> NDArray[np.int64]:
    """Maximum a posteriori bin index for each time bin.

    Returns the index of the bin with highest posterior probability
    at each time step.

    Parameters
    ----------
    posterior : NDArray[np.float64], shape (n_time_bins, n_bins)
        Posterior probability distribution over positions.
        Each row should sum to 1.0.

    Returns
    -------
    NDArray[np.int64], shape (n_time_bins,)
        Bin indices of maximum posterior probability.

    Notes
    -----
    Uses ``np.argmax(axis=1)`` which returns the first maximum
    in case of ties.

    This function mirrors ``DecodingResult.map_estimate``.

    See Also
    --------
    map_position : MAP position in environment coordinates
    DecodingResult.map_estimate : Property version on result container

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.decoding.estimates import map_estimate
    >>>
    >>> # Delta posterior at bin 5
    >>> posterior = np.zeros((1, 10))
    >>> posterior[0, 5] = 1.0
    >>> bins = map_estimate(posterior)
    >>> print(bool(bins[0] == 5))
    True
    """
    result: NDArray[np.int64] = np.argmax(posterior, axis=1).astype(np.int64)
    return result


def map_position(
    env: Environment,
    posterior: NDArray[np.float64],
) -> NDArray[np.float64]:
    """MAP position in environment coordinates.

    Returns the coordinates of the bin with highest posterior probability
    at each time step.

    Parameters
    ----------
    env : Environment
        Spatial environment providing bin_centers for coordinate lookup.
    posterior : NDArray[np.float64], shape (n_time_bins, n_bins)
        Posterior probability distribution over positions.
        Each row should sum to 1.0.

    Returns
    -------
    NDArray[np.float64], shape (n_time_bins, n_dims)
        MAP positions in environment coordinates.

    Notes
    -----
    Computed as: ``env.bin_centers[map_estimate(posterior)]``

    This function mirrors ``DecodingResult.map_position``.

    See Also
    --------
    map_estimate : MAP bin index
    mean_position : Posterior mean position
    DecodingResult.map_position : Property version on result container

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.decoding.estimates import map_position
    >>>
    >>> positions = np.column_stack([np.linspace(0, 10, 50), np.linspace(0, 10, 50)])
    >>> env = Environment.from_samples(positions, bin_size=2.0)
    >>>
    >>> # Delta posterior at bin 0
    >>> posterior = np.zeros((1, env.n_bins))
    >>> posterior[0, 0] = 1.0
    >>> pos = map_position(env, posterior)
    >>> pos.shape
    (1, 2)
    """
    indices = map_estimate(posterior)
    return env.bin_centers[indices]


def mean_position(
    env: Environment,
    posterior: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Posterior mean position (expected value).

    Computes the probability-weighted average of bin center coordinates.

    Parameters
    ----------
    env : Environment
        Spatial environment providing bin_centers.
    posterior : NDArray[np.float64], shape (n_time_bins, n_bins)
        Posterior probability distribution over positions.
        Each row should sum to 1.0.

    Returns
    -------
    NDArray[np.float64], shape (n_time_bins, n_dims)
        Mean positions in environment coordinates.

    Notes
    -----
    Computed as: ``posterior @ env.bin_centers``

    For unimodal posteriors, this is similar to the MAP position.
    For multimodal posteriors, this may fall between modes.

    This function mirrors ``DecodingResult.mean_position``.

    See Also
    --------
    map_position : MAP position (mode of posterior)
    DecodingResult.mean_position : Property version on result container

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.decoding.estimates import mean_position
    >>>
    >>> positions = np.column_stack([np.linspace(0, 10, 50), np.linspace(0, 10, 50)])
    >>> env = Environment.from_samples(positions, bin_size=2.0)
    >>>
    >>> # Uniform posterior - mean is centroid of all bins
    >>> posterior = np.ones((1, env.n_bins)) / env.n_bins
    >>> mean_pos = mean_position(env, posterior)
    >>> mean_pos.shape
    (1, 2)
    """
    return posterior @ env.bin_centers


def entropy(
    posterior: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Posterior entropy in bits (uncertainty measure).

    Measures the uncertainty in the position estimate. Higher values
    indicate more spread-out (uncertain) posteriors.

    Parameters
    ----------
    posterior : NDArray[np.float64], shape (n_time_bins, n_bins)
        Posterior probability distribution over positions.
        Each row should sum to 1.0.

    Returns
    -------
    NDArray[np.float64], shape (n_time_bins,)
        Entropy values in bits. Range: [0, log2(n_bins)].

    Notes
    -----
    Uses mask-based computation to avoid bias from exact zeros:

    .. math::

        H = -\\sum_{i: p_i > 0} p_i \\log_2(p_i)

    This is more accurate than global clipping to ``[1e-10, 1]`` which
    can slightly bias entropy upward when many exact zeros occur.

    Maximum entropy (uniform distribution) is ``log2(n_bins)``.
    Minimum entropy (delta distribution) is 0.

    This function mirrors ``DecodingResult.uncertainty`` (named ``entropy``
    here to be precise about the statistic being computed).

    See Also
    --------
    DecodingResult.uncertainty : Property version on result container

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.decoding.estimates import entropy
    >>>
    >>> # Uniform posterior (maximum entropy)
    >>> posterior = np.ones((1, 8)) / 8
    >>> ent = entropy(posterior)
    >>> print(f"{ent[0]:.2f}")  # log2(8) = 3.0
    3.00
    >>>
    >>> # Delta posterior (minimum entropy)
    >>> posterior = np.zeros((1, 8))
    >>> posterior[0, 0] = 1.0
    >>> ent = entropy(posterior)
    >>> bool(abs(ent[0]) < 0.01)  # Should be 0 bits (within numerical precision)
    True
    """
    p = np.clip(posterior, 0.0, 1.0)
    # Vectorized mask-based entropy: avoid log(0) by using np.where
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.where(p > 0, np.log2(p), 0.0)
    result: NDArray[np.float64] = cast(
        "NDArray[np.float64]", -np.sum(p * log_p, axis=1)
    )
    return result


def credible_region(
    env: Environment,
    posterior: NDArray[np.float64],
    level: float = 0.95,
) -> list[NDArray[np.int64]]:
    """Highest posterior density region containing specified probability mass.

    Returns the smallest set of bins that contains at least the specified
    probability mass, selected in order of decreasing probability density.

    Parameters
    ----------
    env : Environment
        Spatial environment (used for validation, provides n_bins).
    posterior : NDArray[np.float64], shape (n_time_bins, n_bins)
        Posterior probability distribution over positions.
        Each row should sum to 1.0.
    level : float, default=0.95
        Probability mass to include. Must be in (0, 1).
        Common values: 0.5 (median region), 0.9, 0.95, 0.99.

    Returns
    -------
    list[NDArray[np.int64]]
        List of arrays, one per time bin. Each array contains the bin
        indices in the HPD region, sorted by probability (highest first).

    Raises
    ------
    ValueError
        If level is not in (0, 1).

    Notes
    -----
    The Highest Posterior Density (HPD) region is the smallest region
    containing the specified probability mass. It is constructed by
    including bins in order of decreasing probability until the
    cumulative mass reaches the target level.

    For a uniform posterior, the HPD region contains ceil(level * n_bins)
    bins (any subset works, but we return the first in sorted order).

    For a unimodal posterior, the HPD region is a contiguous set of bins
    around the mode.

    See Also
    --------
    entropy : Scalar uncertainty measure
    map_estimate : Point estimate (mode)

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.decoding.estimates import credible_region
    >>>
    >>> positions = np.column_stack([np.linspace(0, 10, 50), np.linspace(0, 10, 50)])
    >>> env = Environment.from_samples(positions, bin_size=2.0)
    >>>
    >>> # Delta posterior - HPD is single bin
    >>> posterior = np.zeros((1, env.n_bins))
    >>> posterior[0, 5] = 1.0
    >>> regions = credible_region(env, posterior, level=0.95)
    >>> len(regions[0])
    1
    """
    if not 0 < level < 1:
        raise ValueError(f"level must be between 0 and 1 (exclusive), got {level}")

    n_time_bins = posterior.shape[0]
    result: list[NDArray[np.int64]] = []

    for t in range(n_time_bins):
        row = posterior[t]

        # Sort bins by probability (descending)
        sorted_indices = np.argsort(row)[::-1]
        sorted_probs = row[sorted_indices]

        # Find cumulative sum
        cumsum = np.cumsum(sorted_probs)

        # Find how many bins needed to reach level
        # Use >= to ensure we include enough mass
        n_bins_needed = int(np.searchsorted(cumsum, level, side="right") + 1)

        # Get the HPD bin indices
        hpd_indices = sorted_indices[:n_bins_needed].astype(np.int64)

        result.append(hpd_indices)

    return result
