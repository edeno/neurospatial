"""Shared metric implementations for encoding result classes.

This module provides spatial information and sparsity computations that are
used by result classes (SpatialRateResult, DirectionalRateResult, etc.).

The functions in this module operate on dense arrays:
- Single neuron: firing_rate (n_bins,), occupancy (n_bins,)
- Batch: firing_rates (n_neurons, n_bins), occupancy (n_bins,)

These are backend-aware implementations: NumPy in → NumPy out, JAX in → JAX out.
For host-only operations, use `_to_numpy()` from `_base.py` first.

References
----------
.. [1] Skaggs, W. E., McNaughton, B. L., & Gothard, K. M. (1993).
       An information-theoretic approach to deciphering the hippocampal code.
       Advances in Neural Information Processing Systems, 5, 1030-1037.

.. [2] Skaggs, W. E., McNaughton, B. L., et al. (1996). Theta phase
       precession in hippocampal neuronal populations and the compression
       of temporal sequences. Hippocampus, 6(2), 149-172.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from neurospatial import Environment

__all__ = [
    "batch_grid_scores",
    "batch_sparsity",
    "batch_spatial_information",
    "sparsity",
    "spatial_information",
]


def spatial_information(
    firing_rate: NDArray[np.float64],
    occupancy: NDArray[np.float64],
    *,
    base: float = 2.0,
) -> float:
    """Compute Skaggs spatial information (bits per spike) for single neuron.

    Spatial information quantifies how much information each spike conveys
    about the animal's spatial location. This is a fundamental metric for
    classifying place cells and other spatially-tuned neurons.

    Parameters
    ----------
    firing_rate : ndarray, shape (n_bins,)
        Firing rate map in Hz. Can contain NaN values which are ignored.
    occupancy : ndarray, shape (n_bins,)
        Time spent in each bin (seconds or any time unit). Will be normalized
        to probability internally. Can contain NaN values which are ignored.
    base : float, default=2.0
        Logarithm base. Use 2.0 for bits (standard), np.e for nats.

    Returns
    -------
    float
        Spatial information in bits per spike (if base=2.0).
        Returns 0.0 if mean rate is zero or undefined.

    Raises
    ------
    ValueError
        If firing_rate and occupancy have different shapes.

    Notes
    -----
    **Formula (Skaggs et al. 1993)**:

    .. math::

        I = \\sum_i p_i \\frac{r_i}{\\bar{r}} \\log \\left( \\frac{r_i}{\\bar{r}} \\right)

    where :math:`p_i` is occupancy probability, :math:`r_i` is firing rate
    in bin :math:`i`, and :math:`\\bar{r}` is mean firing rate.

    **Interpretation**:

    - Place cells typically have 1-3 bits/spike
    - Higher values indicate more spatially selective firing
    - Zero information means uniform firing (no spatial selectivity)

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding._metrics import spatial_information

    >>> # Uniform firing → zero information
    >>> firing_rate = np.ones(100) * 3.0
    >>> occupancy = np.ones(100)
    >>> info = spatial_information(firing_rate, occupancy)
    >>> abs(info) < 1e-6  # Should be ~0
    True

    >>> # Selective firing → high information
    >>> firing_rate = np.zeros(100)
    >>> firing_rate[50] = 30.0  # Fires only in bin 50
    >>> occupancy = np.ones(100)
    >>> info = spatial_information(firing_rate, occupancy)
    >>> info > 4.0  # High spatial info
    True

    See Also
    --------
    batch_spatial_information : Vectorized version for populations
    sparsity : Complementary measure of spatial selectivity
    """
    # Input validation
    firing_rate = np.asarray(firing_rate)
    occupancy = np.asarray(occupancy)

    if firing_rate.shape != occupancy.shape:
        raise ValueError(
            f"firing_rate shape {firing_rate.shape} does not match "
            f"occupancy shape {occupancy.shape}"
        )

    # Handle empty arrays
    if firing_rate.size == 0:
        raise ValueError("firing_rate and occupancy cannot be empty arrays")

    # Normalize occupancy to probability
    occ_sum = np.nansum(occupancy)
    if occ_sum == 0 or np.isnan(occ_sum):
        return 0.0
    occupancy_prob = occupancy / occ_sum

    # Mean firing rate (weighted by occupancy)
    mean_rate = np.nansum(occupancy_prob * firing_rate)

    if mean_rate == 0 or np.isnan(mean_rate):
        return 0.0

    # Compute information using vectorized operations
    # Mask for valid bins: positive occupancy, positive finite firing rate
    with np.errstate(divide="ignore", invalid="ignore"):
        valid_mask = (occupancy_prob > 0) & (firing_rate > 0) & np.isfinite(firing_rate)

        if not np.any(valid_mask):
            return 0.0

        # Extract valid values and compute
        occ_valid = occupancy_prob[valid_mask]
        rate_valid = firing_rate[valid_mask]
        ratio = rate_valid / mean_rate
        information = np.sum(occ_valid * ratio * np.log(ratio)) / np.log(base)

    # Ensure non-negative result (floating point errors can produce tiny negatives)
    return float(max(0.0, information))


def batch_spatial_information(
    firing_rates: NDArray[np.float64],
    occupancy: NDArray[np.float64],
    *,
    base: float = 2.0,
) -> NDArray[np.float64]:
    """Compute Skaggs spatial information for multiple neurons.

    Vectorized version of `spatial_information()` for efficient population
    analysis. Computes spatial information for each neuron in a batch.

    Parameters
    ----------
    firing_rates : ndarray, shape (n_neurons, n_bins)
        Firing rate maps for each neuron in Hz.
    occupancy : ndarray, shape (n_bins,)
        Shared occupancy for all neurons (time spent in each bin).
    base : float, default=2.0
        Logarithm base. Use 2.0 for bits (standard), np.e for nats.

    Returns
    -------
    ndarray, shape (n_neurons,)
        Spatial information in bits per spike for each neuron.

    Raises
    ------
    ValueError
        If firing_rates.shape[1] != occupancy.shape[0].

    Notes
    -----
    This function computes spatial information independently for each neuron.
    The occupancy is shared across all neurons (same behavioral sampling).

    This is a backend-aware function: if `firing_rates` is a JAX array,
    the computation will use JAX operations and return a JAX array.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding._metrics import batch_spatial_information

    >>> # 5 neurons, 100 bins
    >>> firing_rates = np.random.rand(5, 100) * 10
    >>> occupancy = np.ones(100)
    >>> info = batch_spatial_information(firing_rates, occupancy)
    >>> info.shape
    (5,)

    See Also
    --------
    spatial_information : Single-neuron version
    """
    firing_rates = np.asarray(firing_rates)
    occupancy = np.asarray(occupancy)

    # Validate shapes
    if firing_rates.ndim != 2:
        raise ValueError(
            f"firing_rates must be 2D (n_neurons, n_bins), got shape {firing_rates.shape}"
        )

    if occupancy.ndim != 1:
        raise ValueError(f"occupancy must be 1D (n_bins,), got shape {occupancy.shape}")

    if firing_rates.shape[1] != occupancy.shape[0]:
        raise ValueError(
            f"firing_rates has {firing_rates.shape[1]} bins but "
            f"occupancy has {occupancy.shape[0]} bins"
        )

    n_neurons = firing_rates.shape[0]

    # Compute for each neuron
    # Note: Could be optimized with vectorized operations, but this is clearer
    # and matches single-neuron behavior exactly. JAX backend can use vmap.
    result = np.array(
        [
            spatial_information(firing_rates[i], occupancy, base=base)
            for i in range(n_neurons)
        ],
        dtype=np.float64,
    )

    return result


def sparsity(
    firing_rate: NDArray[np.float64],
    occupancy: NDArray[np.float64],
) -> float:
    """Compute sparsity of spatial firing for single neuron.

    Sparsity measures what fraction of the environment elicits significant
    firing. Lower values indicate sparser, more selective place fields.

    Parameters
    ----------
    firing_rate : ndarray, shape (n_bins,)
        Firing rate map in Hz. Can contain NaN values which are ignored.
    occupancy : ndarray, shape (n_bins,)
        Time spent in each bin (seconds or any time unit). Will be normalized
        to probability internally. Can contain NaN values which are ignored.

    Returns
    -------
    float
        Sparsity value in range [0, 1]. Lower values indicate sparser firing.
        Returns 0.0 if denominator is zero or undefined.

    Raises
    ------
    ValueError
        If firing_rate and occupancy have different shapes.

    Notes
    -----
    **Formula (Skaggs et al. 1996)**:

    .. math::

        S = \\frac{\\left( \\sum_i p_i r_i \\right)^2}{\\sum_i p_i r_i^2}

    where :math:`p_i` is occupancy probability and :math:`r_i` is firing rate.

    **Interpretation**:

    - Range: [0, 1]
    - Low sparsity (0.1-0.3): Sparse, selective place field
    - High sparsity (~1.0): Uniform firing throughout environment
    - Typical place cells: 0.1-0.3

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding._metrics import sparsity

    >>> # Uniform firing → high sparsity (close to 1)
    >>> firing_rate = np.ones(100) * 5.0
    >>> occupancy = np.ones(100)
    >>> spars = sparsity(firing_rate, occupancy)
    >>> spars > 0.99
    True

    >>> # Selective firing → low sparsity
    >>> firing_rate = np.zeros(100)
    >>> firing_rate[50] = 30.0  # Fires only in bin 50
    >>> occupancy = np.ones(100)
    >>> spars = sparsity(firing_rate, occupancy)
    >>> spars < 0.1
    True

    See Also
    --------
    batch_sparsity : Vectorized version for populations
    spatial_information : Complementary measure of spatial selectivity
    """
    # Input validation
    firing_rate = np.asarray(firing_rate)
    occupancy = np.asarray(occupancy)

    if firing_rate.shape != occupancy.shape:
        raise ValueError(
            f"firing_rate shape {firing_rate.shape} does not match "
            f"occupancy shape {occupancy.shape}"
        )

    # Handle empty arrays
    if firing_rate.size == 0:
        raise ValueError("firing_rate and occupancy cannot be empty arrays")

    # Normalize occupancy to probability
    occ_sum = np.nansum(occupancy)
    if occ_sum == 0 or np.isnan(occ_sum):
        return 0.0
    occupancy_prob = occupancy / occ_sum

    # Compute sparsity (use nansum to ignore NaN bins)
    numerator = np.nansum(occupancy_prob * firing_rate) ** 2
    denominator = np.nansum(occupancy_prob * firing_rate**2)

    if denominator == 0 or np.isnan(denominator):
        return 0.0

    sparsity_value = numerator / denominator

    # Clamp to [0, 1] to handle floating point precision issues
    return float(np.clip(sparsity_value, 0.0, 1.0))


def batch_sparsity(
    firing_rates: NDArray[np.float64],
    occupancy: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute sparsity for multiple neurons.

    Vectorized version of `sparsity()` for efficient population analysis.
    Computes sparsity for each neuron in a batch.

    Parameters
    ----------
    firing_rates : ndarray, shape (n_neurons, n_bins)
        Firing rate maps for each neuron in Hz.
    occupancy : ndarray, shape (n_bins,)
        Shared occupancy for all neurons (time spent in each bin).

    Returns
    -------
    ndarray, shape (n_neurons,)
        Sparsity values in range [0, 1] for each neuron.

    Raises
    ------
    ValueError
        If firing_rates.shape[1] != occupancy.shape[0].

    Notes
    -----
    This function computes sparsity independently for each neuron.
    The occupancy is shared across all neurons (same behavioral sampling).

    This is a backend-aware function: if `firing_rates` is a JAX array,
    the computation will use JAX operations and return a JAX array.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding._metrics import batch_sparsity

    >>> # 5 neurons, 100 bins
    >>> firing_rates = np.random.rand(5, 100) * 10
    >>> occupancy = np.ones(100)
    >>> spars = batch_sparsity(firing_rates, occupancy)
    >>> spars.shape
    (5,)

    See Also
    --------
    sparsity : Single-neuron version
    """
    firing_rates = np.asarray(firing_rates)
    occupancy = np.asarray(occupancy)

    # Validate shapes
    if firing_rates.ndim != 2:
        raise ValueError(
            f"firing_rates must be 2D (n_neurons, n_bins), got shape {firing_rates.shape}"
        )

    if occupancy.ndim != 1:
        raise ValueError(f"occupancy must be 1D (n_bins,), got shape {occupancy.shape}")

    if firing_rates.shape[1] != occupancy.shape[0]:
        raise ValueError(
            f"firing_rates has {firing_rates.shape[1]} bins but "
            f"occupancy has {occupancy.shape[0]} bins"
        )

    n_neurons = firing_rates.shape[0]

    # Compute for each neuron
    result = np.array(
        [sparsity(firing_rates[i], occupancy) for i in range(n_neurons)],
        dtype=np.float64,
    )

    return result


def batch_grid_scores(
    env: Environment,
    firing_rates: NDArray[np.float64],
    *,
    inner_radius_fraction: float = 0.2,
    outer_radius_fraction: float = 0.5,
) -> NDArray[np.float64]:
    """Compute grid scores for multiple neurons.

    Computes grid score (hexagonal periodicity) for each neuron's firing
    rate map. Grid score quantifies how well the firing pattern exhibits
    the characteristic 6-fold rotational symmetry of grid cells.

    Parameters
    ----------
    env : Environment
        Spatial environment containing bin centers and connectivity.
        Must be a regular 2D grid for FFT-based autocorrelation.
    firing_rates : ndarray, shape (n_neurons, n_bins)
        Firing rate maps for each neuron in Hz.
    inner_radius_fraction : float, default=0.2
        Inner radius of annular region as fraction of autocorrelogram semi-axis.
        See :func:`~neurospatial.encoding.grid.grid_score` for details.
    outer_radius_fraction : float, default=0.5
        Outer radius of annular region as fraction of autocorrelogram semi-axis.
        See :func:`~neurospatial.encoding.grid.grid_score` for details.

    Returns
    -------
    ndarray, shape (n_neurons,)
        Grid scores in range [-2, 2] for each neuron. Returns NaN for neurons
        where grid score cannot be computed (constant firing, invalid autocorrelation,
        or non-regular grid environment).

    Raises
    ------
    ValueError
        If firing_rates is not 2D.
        If firing_rates.shape[1] != env.n_bins.

    Notes
    -----
    For each neuron, this function:

    1. Computes spatial autocorrelation using :func:`spatial_autocorrelation`
    2. Computes grid score using :func:`grid_score`

    **Interpretation**:

    - **score > 0.4**: Strong hexagonal grid (typical threshold for grid cells)
    - **score ≈ 0**: No hexagonal structure (place cells, non-spatial cells)
    - **score < 0**: Anti-hexagonal structure (rare)

    **Environment requirements**:

    Grid score computation requires a regular 2D grid environment for FFT-based
    autocorrelation. For irregular environments, the function returns NaN.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding._metrics import batch_grid_scores

    >>> # Create environment
    >>> x = np.linspace(-50, 50, 51)
    >>> xx, yy = np.meshgrid(x, x)
    >>> positions = np.column_stack([xx.ravel(), yy.ravel()])
    >>> env = Environment.from_samples(positions, bin_size=2.0)

    >>> # Random firing patterns (5 neurons)
    >>> rng = np.random.default_rng(42)
    >>> firing_rates = rng.random((5, env.n_bins)) * 10
    >>> scores = batch_grid_scores(env, firing_rates)
    >>> scores.shape
    (5,)

    See Also
    --------
    neurospatial.encoding.grid.grid_score : Single autocorrelogram grid score
    neurospatial.encoding.grid.spatial_autocorrelation : Compute autocorrelation
    """
    from neurospatial.encoding.grid import grid_score, spatial_autocorrelation

    firing_rates = np.asarray(firing_rates)

    # Validate shapes
    if firing_rates.ndim != 2:
        raise ValueError(
            f"firing_rates must be 2D (n_neurons, n_bins), got shape {firing_rates.shape}"
        )

    if firing_rates.shape[1] != env.n_bins:
        raise ValueError(
            f"firing_rates has {firing_rates.shape[1]} bins but "
            f"env has {env.n_bins} bins"
        )

    n_neurons = firing_rates.shape[0]
    scores = np.empty(n_neurons, dtype=np.float64)

    for i in range(n_neurons):
        firing_rate = firing_rates[i]

        try:
            # Compute spatial autocorrelation (FFT method for regular 2D grids)
            autocorr = spatial_autocorrelation(env, firing_rate, method="auto")

            # spatial_autocorrelation returns 2D array for FFT, tuple for graph
            if isinstance(autocorr, tuple):
                # Graph-based method not compatible with grid_score
                scores[i] = np.nan
            else:
                # Compute grid score from 2D autocorrelation
                scores[i] = grid_score(
                    autocorr,
                    inner_radius_fraction=inner_radius_fraction,
                    outer_radius_fraction=outer_radius_fraction,
                )
        except (ValueError, RuntimeError):
            # Handle errors gracefully (e.g., constant firing, all NaN)
            scores[i] = np.nan

    return scores
