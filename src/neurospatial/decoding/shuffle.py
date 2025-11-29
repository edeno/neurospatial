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
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from neurospatial.environment.core import Environment


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


# =============================================================================
# II. Cell Identity Shuffles - Test spatial code coherence
# =============================================================================


def shuffle_cell_identity(
    spike_counts: NDArray[np.int64],
    encoding_models: NDArray[np.float64],
    *,
    n_shuffles: int = 1000,
    rng: np.random.Generator | int | None = None,
) -> Generator[tuple[NDArray[np.int64], NDArray[np.float64]], None, None]:
    """Shuffle mapping between spike trains and place fields.

    **Primary test for spatial code coherence.** Disrupts the learned
    relationship between a neuron's activity and its encoded spatial location
    by randomly permuting which spike train is associated with which place
    field.

    This shuffle randomly permutes columns (neuron axis) of spike_counts,
    effectively reassigning which place field goes with which spike train.
    The encoding models are returned unchanged.

    Parameters
    ----------
    spike_counts : NDArray[np.int64], shape (n_time_bins, n_neurons)
        Spike counts per neuron per time bin.
    encoding_models : NDArray[np.float64], shape (n_neurons, n_bins)
        Firing rate maps (place fields) for each neuron.
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
        Spike counts with neuron identities permuted (columns shuffled).
    encoding_models : NDArray[np.float64], shape (n_neurons, n_bins)
        Original encoding models (unchanged, same object).

    Notes
    -----
    - Randomly permutes columns of spike_counts (neuron axis)
    - Encoding models remain fixed (same object returned each iteration)
    - Equivalent to randomly reassigning which place field goes with which
      spike train
    - Preserves spike counts per neuron per time bin
    - Preserves total spikes per time bin
    - Caution: Can introduce noise if firing rates differ greatly between
      cells

    Alternative implementation (equivalent):
        Instead of shuffling spike_counts columns, shuffle encoding_models
        rows. This yields the same decoded posteriors.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.decoding.shuffle import shuffle_cell_identity

    >>> spike_counts = np.array([[0, 1, 2], [2, 0, 1]], dtype=np.int64)
    >>> encoding_models = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    >>> for i, (shuffled, models) in enumerate(
    ...     shuffle_cell_identity(spike_counts, encoding_models, n_shuffles=3, rng=42)
    ... ):
    ...     print(
    ...         f"Shuffle {i}: sum={shuffled.sum()}, models unchanged={models is encoding_models}"
    ...     )
    Shuffle 0: sum=6, models unchanged=True
    Shuffle 1: sum=6, models unchanged=True
    Shuffle 2: sum=6, models unchanged=True

    See Also
    --------
    shuffle_place_fields_circular : Circular shift of place fields
    shuffle_time_bins : Temporal order shuffle
    """
    generator = _ensure_rng(rng)
    n_neurons = spike_counts.shape[1]

    for _ in range(n_shuffles):
        # Generate a random permutation of column indices
        perm = generator.permutation(n_neurons)
        # Apply permutation to columns (neuron axis)
        yield spike_counts[:, perm].copy(), encoding_models


def shuffle_place_fields_circular(
    encoding_models: NDArray[np.float64],
    *,
    n_shuffles: int = 1000,
    rng: np.random.Generator | int | None = None,
) -> Generator[NDArray[np.float64], None, None]:
    """Circularly shift each place field by a random amount.

    **Conservative null hypothesis.** Preserves individual cell spiking
    properties and local place field structure while disrupting spatial
    relationships between neurons.

    Each neuron's place field is independently shifted by a random amount
    along the position axis. This preserves the shape of each place field
    but destroys the spatial relationships between neurons.

    Parameters
    ----------
    encoding_models : NDArray[np.float64], shape (n_neurons, n_bins)
        Firing rate maps (place fields) for each neuron.
    n_shuffles : int, default=1000
        Number of shuffled versions to generate.
    rng : np.random.Generator | int | None, default=None
        Random number generator for reproducibility.

        - If Generator: Use directly
        - If int: Seed for ``np.random.default_rng()``
        - If None: Use default RNG (not reproducible)

    Yields
    ------
    shuffled_models : NDArray[np.float64], shape (n_neurons, n_bins)
        Place fields with each row circularly shifted by a random amount.

    Notes
    -----
    - Each neuron's place field is shifted independently
    - Preserves the shape of each place field
    - Destroys spatial relationships between neurons
    - More conservative than cell identity shuffle
    - For 2D environments: consider ``shuffle_place_fields_circular_2d``

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.decoding.shuffle import shuffle_place_fields_circular

    >>> encoding_models = np.array(
    ...     [
    ...         [1.0, 2.0, 3.0, 4.0],
    ...         [4.0, 3.0, 2.0, 1.0],
    ...     ]
    ... )
    >>> for i, shuffled in enumerate(
    ...     shuffle_place_fields_circular(encoding_models, n_shuffles=3, rng=42)
    ... ):
    ...     print(f"Shuffle {i}: shape={shuffled.shape}")
    Shuffle 0: shape=(2, 4)
    Shuffle 1: shape=(2, 4)
    Shuffle 2: shape=(2, 4)

    See Also
    --------
    shuffle_place_fields_circular_2d : 2D circular shift for 2D environments
    shuffle_cell_identity : Shuffle neuron-to-place-field mapping
    """
    generator = _ensure_rng(rng)
    n_neurons, n_bins = encoding_models.shape

    for _ in range(n_shuffles):
        # Generate random shift amounts for each neuron
        shifts = generator.integers(0, n_bins, size=n_neurons)
        # Apply circular shifts to each row independently
        shuffled = np.empty_like(encoding_models)
        for i in range(n_neurons):
            shift_amount: int = int(shifts[i])  # type: ignore[index]
            shuffled[i, :] = np.roll(encoding_models[i, :], shift_amount)
        yield shuffled


def shuffle_place_fields_circular_2d(
    encoding_models: NDArray[np.float64],
    env: Environment,
    *,
    n_shuffles: int = 1000,
    rng: np.random.Generator | int | None = None,
) -> Generator[NDArray[np.float64], None, None]:
    """Circularly shift 2D place fields in both dimensions.

    For 2D environments, shifts place fields in both x and y dimensions.
    This is the 2D analog of ``shuffle_place_fields_circular``.

    Each neuron's place field is reshaped to a 2D grid, shifted by random
    amounts in both dimensions, and flattened back to 1D.

    Parameters
    ----------
    encoding_models : NDArray[np.float64], shape (n_neurons, n_bins)
        Firing rate maps (place fields) for each neuron.
    env : Environment
        2D environment with grid layout (provides ``grid_shape``).
    n_shuffles : int, default=1000
        Number of shuffled versions to generate.
    rng : np.random.Generator | int | None, default=None
        Random number generator for reproducibility.

        - If Generator: Use directly
        - If int: Seed for ``np.random.default_rng()``
        - If None: Use default RNG (not reproducible)

    Yields
    ------
    shuffled_models : NDArray[np.float64], shape (n_neurons, n_bins)
        Place fields with 2D circular shifts applied.

    Raises
    ------
    ValueError
        If environment is not 2D or doesn't have grid layout.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.decoding.shuffle import shuffle_place_fields_circular_2d

    >>> positions = np.random.default_rng(42).uniform(0, 10, (100, 2))
    >>> env = Environment.from_samples(positions, bin_size=2.0)
    >>> encoding_models = np.random.default_rng(42).random((3, env.n_bins))
    >>> for i, shuffled in enumerate(
    ...     shuffle_place_fields_circular_2d(encoding_models, env, n_shuffles=3, rng=42)
    ... ):
    ...     print(f"Shuffle {i}: shape={shuffled.shape}")
    Shuffle 0: shape=(3, 36)
    Shuffle 1: shape=(3, 36)
    Shuffle 2: shape=(3, 36)

    See Also
    --------
    shuffle_place_fields_circular : 1D circular shift
    shuffle_cell_identity : Shuffle neuron-to-place-field mapping
    """
    # Validate 2D environment
    if env.n_dims != 2:
        raise ValueError(
            f"shuffle_place_fields_circular_2d requires a 2D environment, "
            f"got {env.n_dims}D"
        )

    # Get grid shape from environment
    if not hasattr(env.layout, "grid_shape") or env.layout.grid_shape is None:
        raise ValueError(
            "shuffle_place_fields_circular_2d requires environment with grid layout "
            "(grid_shape attribute)"
        )

    grid_shape = env.layout.grid_shape

    # Verify encoding models match expected size
    expected_bins = int(np.prod(grid_shape))
    actual_bins = encoding_models.shape[1]
    if actual_bins != expected_bins:
        raise ValueError(
            f"shuffle_place_fields_circular_2d requires encoding models with "
            f"{expected_bins} bins (matching grid_shape {grid_shape}), but got "
            f"{actual_bins} bins. This may occur if the environment has inactive "
            f"bins (masked grid). Consider using shuffle_place_fields_circular "
            f"for masked environments."
        )
    generator = _ensure_rng(rng)
    n_neurons = encoding_models.shape[0]

    for _ in range(n_shuffles):
        # Generate random shift amounts for each neuron in each dimension
        shifts_x = generator.integers(0, grid_shape[0], size=n_neurons)
        shifts_y = generator.integers(0, grid_shape[1], size=n_neurons)

        # Apply 2D circular shifts to each neuron
        shuffled = np.empty_like(encoding_models)
        for i in range(n_neurons):
            shift_x: int = int(shifts_x[i])  # type: ignore[index]
            shift_y: int = int(shifts_y[i])  # type: ignore[index]
            # Reshape to 2D grid
            field_2d = encoding_models[i, :].reshape(grid_shape)
            # Apply circular shift in both dimensions
            shifted_2d = np.roll(
                np.roll(field_2d, shift_x, axis=0),
                shift_y,
                axis=1,
            )
            # Flatten back to 1D
            shuffled[i, :] = shifted_2d.ravel()

        yield shuffled


# =============================================================================
# III. Posterior/Position Shuffles - Test trajectory detection
# =============================================================================


def shuffle_posterior_circular(
    posterior: NDArray[np.float64],
    *,
    n_shuffles: int = 1000,
    rng: np.random.Generator | int | None = None,
) -> Generator[NDArray[np.float64], None, None]:
    """Circularly shift posterior at each time bin independently.

    Controls for chance linear alignment of position estimates by disrupting
    trajectory progression while preserving local smoothness. Each time bin's
    posterior is independently shifted by a random amount along the position
    axis.

    Parameters
    ----------
    posterior : NDArray[np.float64], shape (n_time_bins, n_bins)
        Posterior probability distribution from decoding. Each row should
        sum to 1.0.
    n_shuffles : int, default=1000
        Number of shuffled versions to generate.
    rng : np.random.Generator | int | None, default=None
        Random number generator for reproducibility.

        - If Generator: Use directly
        - If int: Seed for ``np.random.default_rng()``
        - If None: Use default RNG (not reproducible)

    Yields
    ------
    shuffled_posterior : NDArray[np.float64], shape (n_time_bins, n_bins)
        Posterior with each row circularly shifted by a random amount.

    Notes
    -----
    - Each time bin is shifted independently
    - Preserves the shape of each instantaneous posterior
    - Destroys temporal continuity of decoded positions
    - Caution: Can generate position representations that don't exist in
      original data (edge effects near track boundaries)
    - Normalization is preserved (each row still sums to 1.0)

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.decoding.shuffle import shuffle_posterior_circular

    >>> # Create a normalized posterior (3 time bins, 5 spatial bins)
    >>> raw = np.array(
    ...     [
    ...         [0.1, 0.2, 0.4, 0.2, 0.1],
    ...         [0.5, 0.3, 0.1, 0.05, 0.05],
    ...         [0.05, 0.1, 0.2, 0.4, 0.25],
    ...     ]
    ... )
    >>> for i, shuffled in enumerate(
    ...     shuffle_posterior_circular(raw, n_shuffles=3, rng=42)
    ... ):
    ...     print(
    ...         f"Shuffle {i}: shape={shuffled.shape}, sums to 1={np.allclose(shuffled.sum(axis=1), 1.0)}"
    ...     )
    Shuffle 0: shape=(3, 5), sums to 1=True
    Shuffle 1: shape=(3, 5), sums to 1=True
    Shuffle 2: shape=(3, 5), sums to 1=True

    See Also
    --------
    shuffle_posterior_weighted_circular : Weighted circular shift with edge
        effect mitigation
    shuffle_place_fields_circular : Circular shift of place fields
    """
    generator = _ensure_rng(rng)
    n_time_bins, n_bins = posterior.shape

    for _ in range(n_shuffles):
        # Generate random shift amounts for each time bin
        shifts = generator.integers(0, n_bins, size=n_time_bins)
        # Apply circular shifts to each row independently
        shuffled = np.empty_like(posterior)
        for i in range(n_time_bins):
            shift_amount: int = int(shifts[i])  # type: ignore[index]
            shuffled[i, :] = np.roll(posterior[i, :], shift_amount)
        yield shuffled


def shuffle_posterior_weighted_circular(
    posterior: NDArray[np.float64],
    *,
    edge_buffer: int = 5,
    n_shuffles: int = 1000,
    rng: np.random.Generator | int | None = None,
) -> Generator[NDArray[np.float64], None, None]:
    """Weighted circular shift with edge effect mitigation.

    Refined version of posterior shuffle that maintains non-uniformity and
    reduces edge effects by restricting shifts when the MAP position is near
    track boundaries. This is more conservative than standard circular shuffle.

    Parameters
    ----------
    posterior : NDArray[np.float64], shape (n_time_bins, n_bins)
        Posterior probability distribution from decoding. Each row should
        sum to 1.0.
    edge_buffer : int, default=5
        Number of bins from track ends where shifts are restricted.
        When the MAP position is within ``edge_buffer`` bins of either edge,
        the shift is restricted to keep the MAP position within bounds
        (not wrapping to the other end).
    n_shuffles : int, default=1000
        Number of shuffled versions to generate.
    rng : np.random.Generator | int | None, default=None
        Random number generator for reproducibility.

        - If Generator: Use directly
        - If int: Seed for ``np.random.default_rng()``
        - If None: Use default RNG (not reproducible)

    Yields
    ------
    shuffled_posterior : NDArray[np.float64], shape (n_time_bins, n_bins)
        Posterior with weighted circular shifts applied.

    Notes
    -----
    - More conservative than standard circular shuffle
    - Shifts are restricted near track ends to mitigate edge effects
    - When MAP position is within ``edge_buffer`` of an edge, shift is
      constrained to avoid wrapping probability mass to the other end
    - Preserves row normalization (each row still sums to 1.0)

    The edge restriction works as follows:
    - For each time bin, compute MAP position (argmax)
    - If MAP is within ``edge_buffer`` of left edge (bin < edge_buffer):
      restrict shift to [-MAP, n_bins - 2*edge_buffer] to keep MAP in center
    - If MAP is within ``edge_buffer`` of right edge (bin >= n_bins - edge_buffer):
      restrict shift to [-(MAP - center), n_bins - MAP] to keep MAP in center
    - Otherwise: allow full circular shift [0, n_bins)

    Example with n_bins=20 and edge_buffer=5:
    - MAP at bin 2 (near left): shift limited to [-2, 10) so MAP stays in [0, 11]
    - MAP at bin 17 (near right): shift limited to [-7, 3) so MAP stays in [10, 19]
    - MAP at bin 10 (center): full circular shift [0, 20)

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.decoding.shuffle import shuffle_posterior_weighted_circular

    >>> # Create a normalized posterior (3 time bins, 10 spatial bins)
    >>> raw = np.random.default_rng(42).random((3, 10))
    >>> posterior = raw / raw.sum(axis=1, keepdims=True)
    >>> for i, shuffled in enumerate(
    ...     shuffle_posterior_weighted_circular(
    ...         posterior, edge_buffer=2, n_shuffles=3, rng=42
    ...     )
    ... ):
    ...     print(f"Shuffle {i}: sums to 1={np.allclose(shuffled.sum(axis=1), 1.0)}")
    Shuffle 0: sums to 1=True
    Shuffle 1: sums to 1=True
    Shuffle 2: sums to 1=True

    See Also
    --------
    shuffle_posterior_circular : Standard circular shift without edge restriction
    shuffle_place_fields_circular : Circular shift of place fields
    """
    generator = _ensure_rng(rng)
    n_time_bins, n_bins = posterior.shape

    # Handle empty posterior
    if n_time_bins == 0:
        for _ in range(n_shuffles):
            yield posterior.copy()
        return

    # Compute MAP positions for each time bin (used to determine shift restrictions)
    map_positions = np.argmax(posterior, axis=1)

    for _ in range(n_shuffles):
        shuffled = np.empty_like(posterior)
        for i in range(n_time_bins):
            map_pos = map_positions[i]

            # Determine allowed shift range based on MAP position
            # The key principle: restrict shifts to prevent probability mass from
            # wrapping to the opposite end of the track when near edges.
            if edge_buffer == 0 or n_bins <= 2 * edge_buffer:
                # No edge restriction or buffer spans entire track
                # Allow any shift in [0, n_bins)
                shift_amount = generator.integers(0, n_bins)
            elif map_pos < edge_buffer:
                # Near left edge: restrict shift range to avoid wrapping to far right
                # Allow negative shifts (move left) but limit positive shifts
                min_shift = -map_pos  # Don't shift MAP below 0
                # Limit positive shift: keep shifted MAP within center region
                max_shift = min(n_bins - map_pos, n_bins - 2 * edge_buffer)
                if max_shift <= min_shift:
                    max_shift = min_shift + 1  # Ensure valid range
                shift_amount = generator.integers(min_shift, max_shift)
            elif map_pos >= n_bins - edge_buffer:
                # Near right edge: restrict shift range to avoid wrapping to far left
                # Allow positive shifts (move right) but limit negative shifts
                max_shift = n_bins - map_pos  # Don't shift MAP beyond last bin
                # Limit negative shift: keep shifted MAP within center region
                min_shift = max(-(map_pos - (n_bins - 2 * edge_buffer)), -(map_pos))
                if max_shift <= min_shift:
                    min_shift = max_shift - 1  # Ensure valid range
                shift_amount = generator.integers(min_shift, max_shift)
            else:
                # Not near edge: allow full circular shift
                shift_amount = generator.integers(0, n_bins)

            shuffled[i, :] = np.roll(posterior[i, :], shift_amount)
        yield shuffled
