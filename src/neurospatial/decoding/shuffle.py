"""Shuffle-based significance testing for Bayesian decoding.

This module provides statistical shuffling procedures to establish null
distributions and test the significance of decoded sequences. These methods
are essential for replay analysis to rule out non-specific factors like
firing rate biases.

Design Principles
-----------------
1. **Generators, not accumulators**: Functions yield shuffled data one at a
   time for memory efficiency.
2. **Reproducibility**: All functions accept `rng` parameter for reproducible
   results.
3. **Clear naming**: Function names indicate what is being shuffled.
4. **Composable**: Shuffles can be combined in analysis pipelines.

Shuffle Categories
------------------
| Category | Null Hypothesis Tested |
|----------|----------------------|
| **Temporal** | Sequential structure is not significant |
| **Cell Identity** | Spatial code coherence is not significant |
| **Posterior** | Trajectory detection is not biased |
| **Surrogate** | Structure exceeds rate-based expectations |

Examples
--------
>>> import numpy as np
>>> from neurospatial.decoding import shuffle_time_bins, decode_position

>>> # Typical usage pattern
>>> spike_counts = np.array([[0, 1], [2, 0], [1, 1]], dtype=np.int64)
>>> null_scores = []
>>> for shuffled in shuffle_time_bins(spike_counts, n_shuffles=100, rng=42):
...     # result = decode_position(env, shuffled, models, dt)
...     # null_scores.append(compute_sequence_score(result))
...     pass

See Also
--------
neurospatial.decoding.trajectory : Trajectory fitting functions
neurospatial.decoding.metrics : Decoding quality metrics
"""

from __future__ import annotations

from collections.abc import Generator

import numpy as np
from numpy.typing import NDArray


def _ensure_rng(
    rng: np.random.Generator | int | None,
) -> np.random.Generator:
    """Convert rng parameter to a Generator instance.

    Parameters
    ----------
    rng : np.random.Generator | int | None
        Random number generator, seed, or None.

    Returns
    -------
    np.random.Generator
        A random number generator instance.
    """
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


# =============================================================================
# I. Temporal Order Shuffles - Test sequential structure within events
# =============================================================================


def shuffle_time_bins(
    spike_counts: NDArray[np.int64],
    *,
    n_shuffles: int = 1000,
    rng: np.random.Generator | int | None = None,
) -> Generator[NDArray[np.int64], None, None]:
    """Shuffle temporal order of time bins within an event.

    **Primary test for sequential structure.** Disrupts temporal order while
    preserving instantaneous firing characteristics.

    This shuffle randomly permutes the rows (time bins) of the spike count
    matrix. Each row is kept intact, so the spike counts per neuron per
    time bin are preserved, but the temporal sequence is destroyed.

    Parameters
    ----------
    spike_counts : NDArray[np.int64], shape (n_time_bins, n_neurons)
        Spike counts for a single candidate event (PBE/SWR).
    n_shuffles : int, default=1000
        Number of shuffled versions to generate.
    rng : np.random.Generator | int | None, default=None
        Random number generator for reproducibility.

        - If Generator: Use directly
        - If int: Seed for ``np.random.default_rng()``
        - If None: Use default RNG (not reproducible)

    Yields
    ------
    shuffled_counts : NDArray[np.int64], shape (n_time_bins, n_neurons)
        Spike counts with time bins in random order.

    Notes
    -----
    - Preserves spike counts per neuron per time bin
    - Preserves total spikes per neuron across event
    - Destroys temporal sequence information
    - Underpowered with very few time bins (<5)

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.decoding.shuffle import shuffle_time_bins

    >>> spike_counts = np.array([[0, 1], [2, 0], [1, 1]], dtype=np.int64)
    >>> for i, shuffled in enumerate(
    ...     shuffle_time_bins(spike_counts, n_shuffles=3, rng=42)
    ... ):
    ...     print(f"Shuffle {i}: shape={shuffled.shape}, sum={shuffled.sum()}")
    Shuffle 0: shape=(3, 2), sum=5
    Shuffle 1: shape=(3, 2), sum=5
    Shuffle 2: shape=(3, 2), sum=5

    See Also
    --------
    shuffle_time_bins_coherent : Coherent shuffle (same permutation for all neurons)
    """
    generator = _ensure_rng(rng)
    n_time_bins = spike_counts.shape[0]

    for _ in range(n_shuffles):
        # Generate a random permutation of row indices
        perm = generator.permutation(n_time_bins)
        # Apply permutation to rows
        yield spike_counts[perm].copy()


def shuffle_time_bins_coherent(
    spike_counts: NDArray[np.int64],
    *,
    n_shuffles: int = 1000,
    rng: np.random.Generator | int | None = None,
) -> Generator[NDArray[np.int64], None, None]:
    """Shuffle time bins coherently across all neurons (time-swap shuffle).

    Preserves population co-activation structure while disrupting temporal
    order. This is the same as ``shuffle_time_bins`` in implementation, but
    is named differently to emphasize that the permutation is coherent
    across all neurons (preserving instantaneous population vectors).

    This shuffle is **less conservative** than independent per-neuron shuffles
    because pairwise correlations between neurons are maintained within each
    time bin.

    Parameters
    ----------
    spike_counts : NDArray[np.int64], shape (n_time_bins, n_neurons)
        Spike counts for a single candidate event.
    n_shuffles : int, default=1000
        Number of shuffled versions to generate.
    rng : np.random.Generator | int | None, default=None
        Random number generator for reproducibility.

        - If Generator: Use directly
        - If int: Seed for ``np.random.default_rng()``
        - If None: Use default RNG (not reproducible)

    Yields
    ------
    shuffled_counts : NDArray[np.int64], shape (n_time_bins, n_neurons)
        Spike counts with rows (time bins) permuted coherently.

    Notes
    -----
    - All neurons see the same temporal permutation
    - Preserves instantaneous population vectors
    - Destroys temporal progression but keeps co-firing structure
    - Mathematically equivalent to ``shuffle_time_bins``, but the function
      name clarifies the intent (testing temporal structure while preserving
      population co-activation)

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.decoding.shuffle import shuffle_time_bins_coherent

    >>> spike_counts = np.array([[0, 1, 2], [2, 0, 1], [1, 2, 0]], dtype=np.int64)
    >>> for shuffled in shuffle_time_bins_coherent(spike_counts, n_shuffles=2, rng=42):
    ...     # Each row is preserved exactly (coherent permutation)
    ...     original_rows = {tuple(row) for row in spike_counts}
    ...     shuffled_rows = {tuple(row) for row in shuffled}
    ...     print(f"Row sets match: {original_rows == shuffled_rows}")
    Row sets match: True
    Row sets match: True

    See Also
    --------
    shuffle_time_bins : Primary temporal shuffle (same implementation)
    shuffle_cell_identity : Shuffle neuron-to-place-field mapping
    """
    generator = _ensure_rng(rng)
    n_time_bins = spike_counts.shape[0]

    for _ in range(n_shuffles):
        # Generate a random permutation of row indices
        # The same permutation is applied to all columns (coherent)
        perm = generator.permutation(n_time_bins)
        # Apply permutation to rows
        yield spike_counts[perm].copy()
