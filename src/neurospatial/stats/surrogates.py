"""Surrogate data generation for statistical significance testing.

This module provides methods for generating surrogate spike trains with
controlled statistical properties. Surrogates are essential for testing
whether observed neural patterns exceed rate-based or temporal expectations.

Design Principles
-----------------
1. **Generators, not accumulators**: Functions yield surrogate data one at a
   time for memory efficiency.
2. **Reproducibility**: All functions accept `rng` parameter for reproducible
   results.
3. **Clear naming**: Function names indicate the type of surrogate generated.
4. **Composable**: Surrogates can be combined in analysis pipelines.

Surrogate Types
---------------
| Type | Null Hypothesis Tested |
|------|----------------------|
| **Poisson** | Rate-matched random spikes |
| **Inhomogeneous Poisson** | Time-varying rate expectations |
| **Jittered** | Precise spike timing is not significant |

Imports
-------
>>> from neurospatial.stats.surrogates import generate_poisson_surrogates
>>> from neurospatial.stats import generate_jittered_spikes

Examples
--------
>>> import numpy as np
>>> from neurospatial.stats.surrogates import generate_poisson_surrogates

>>> # Generate rate-matched surrogates
>>> spike_counts = np.array([[0, 1], [2, 0], [1, 1]], dtype=np.int64)
>>> for surrogate in generate_poisson_surrogates(
...     spike_counts, dt=0.025, n_surrogates=3, rng=42
... ):
...     pass  # Analyze surrogate

See Also
--------
neurospatial.stats.shuffle : Shuffle-based significance testing
neurospatial.stats.circular : Circular statistics
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


def generate_poisson_surrogates(
    spike_counts: NDArray[np.int64],
    dt: float,
    *,
    n_surrogates: int = 1000,
    rng: np.random.Generator | int | None = None,
) -> Generator[NDArray[np.int64], None, None]:
    """Generate homogeneous Poisson surrogate spike trains.

    Creates surrogate spike counts by sampling from Poisson distributions
    with rates equal to the mean firing rate of each neuron across all time
    bins. This destroys all temporal structure while preserving average rates.

    **Tests that observed structure exceeds rate-based expectations.** If
    sequential decoding patterns are significant, they should score higher
    than patterns from rate-matched surrogates.

    Parameters
    ----------
    spike_counts : NDArray[np.int64], shape (n_time_bins, n_neurons)
        Original spike counts per neuron per time bin.
    dt : float
        Time bin width in seconds. **Note:** This parameter is accepted for
        API consistency with other decoding functions but is NOT used in the
        computation. Surrogates are generated using mean spike counts directly
        as Poisson rate parameters.
    n_surrogates : int, default=1000
        Number of surrogate spike trains to generate.
    rng : np.random.Generator | int | None, default=None
        Random number generator for reproducibility.

        - If Generator: Use directly
        - If int: Seed for ``np.random.default_rng()``
        - If None: Use default RNG (not reproducible)

    Yields
    ------
    surrogate : NDArray[np.int64], shape (n_time_bins, n_neurons)
        Surrogate spike counts sampled from Poisson distributions.

    Notes
    -----
    - Computes mean spike count per neuron across all time bins
    - Each time bin independently samples from Poisson(mean_count_per_neuron)
    - Destroys all temporal correlations and patterns
    - Preserves mean firing rates per neuron (statistically)
    - More conservative than shuffle methods (generates truly independent counts)

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.stats.surrogates import generate_poisson_surrogates

    >>> spike_counts = np.array([[0, 1], [2, 0], [1, 1]], dtype=np.int64)
    >>> for i, surrogate in enumerate(
    ...     generate_poisson_surrogates(spike_counts, dt=0.025, n_surrogates=3, rng=42)
    ... ):
    ...     print(f"Surrogate {i}: shape={surrogate.shape}, total={surrogate.sum()}")
    Surrogate 0: shape=(3, 2), total=3
    Surrogate 1: shape=(3, 2), total=5
    Surrogate 2: shape=(3, 2), total=6

    See Also
    --------
    generate_inhomogeneous_poisson_surrogates : Time-varying rate surrogates
    generate_jittered_spikes : Temporal jitter surrogates
    """
    generator = _ensure_rng(rng)
    n_time_bins, n_neurons = spike_counts.shape

    # Handle empty spike counts
    if n_time_bins == 0:
        for _ in range(n_surrogates):
            yield np.zeros((0, n_neurons), dtype=np.int64)
        return

    # Compute mean spike count per neuron (averaged across time bins)
    # This is the lambda parameter for our Poisson distribution
    mean_counts = spike_counts.mean(axis=0)  # Shape: (n_neurons,)

    for _ in range(n_surrogates):
        # Generate surrogate by sampling from Poisson with mean rate
        # Each (time_bin, neuron) pair is sampled independently
        surrogate = generator.poisson(lam=mean_counts, size=(n_time_bins, n_neurons))
        yield surrogate.astype(np.int64)


def generate_inhomogeneous_poisson_surrogates(
    spike_counts: NDArray[np.int64],
    dt: float,
    *,
    smoothing_window: int = 3,
    n_surrogates: int = 1000,
    rng: np.random.Generator | int | None = None,
) -> Generator[NDArray[np.int64], None, None]:
    """Generate inhomogeneous Poisson surrogate spike trains with smoothed rates.

    Creates surrogate spike counts by sampling from Poisson distributions
    with time-varying rates estimated from smoothed spike counts. This
    preserves slow rate fluctuations while destroying fine temporal structure.

    **Tests that structure exceeds time-varying rate expectations.** More
    conservative than homogeneous surrogates - controls for rate modulation
    while testing for sequence-specific coding.

    Parameters
    ----------
    spike_counts : NDArray[np.int64], shape (n_time_bins, n_neurons)
        Original spike counts per neuron per time bin.
    dt : float
        Time bin width in seconds. **Note:** This parameter is accepted for
        API consistency with other decoding functions but is NOT used in the
        computation. Surrogates are generated using smoothed spike counts
        directly as time-varying Poisson rate parameters.
    smoothing_window : int, default=3
        Size of the uniform smoothing window (in time bins) applied to
        estimate time-varying rates. Larger values preserve less temporal
        detail (closer to homogeneous). Must be at least 1.
    n_surrogates : int, default=1000
        Number of surrogate spike trains to generate.
    rng : np.random.Generator | int | None, default=None
        Random number generator for reproducibility.

        - If Generator: Use directly
        - If int: Seed for ``np.random.default_rng()``
        - If None: Use default RNG (not reproducible)

    Yields
    ------
    surrogate : NDArray[np.int64], shape (n_time_bins, n_neurons)
        Surrogate spike counts sampled from time-varying Poisson distributions.

    Notes
    -----
    - Uses uniform 1D filter to smooth spike counts in time
    - Smoothed counts are used as time-varying Poisson rate parameters
    - Preserves slow rate fluctuations (e.g., event onsets)
    - Destroys fine temporal correlations and millisecond-scale patterns
    - More appropriate than homogeneous surrogates when rate modulation
      is a confound (e.g., ramping activity during replay events)

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.stats.surrogates import (
    ...     generate_inhomogeneous_poisson_surrogates,
    ... )

    >>> spike_counts = np.array(
    ...     [[0, 1], [2, 2], [3, 3], [2, 2], [1, 1]], dtype=np.int64
    ... )
    >>> for i, surrogate in enumerate(
    ...     generate_inhomogeneous_poisson_surrogates(
    ...         spike_counts, dt=0.025, smoothing_window=3, n_surrogates=3, rng=42
    ...     )
    ... ):
    ...     print(f"Surrogate {i}: shape={surrogate.shape}")
    Surrogate 0: shape=(5, 2)
    Surrogate 1: shape=(5, 2)
    Surrogate 2: shape=(5, 2)

    See Also
    --------
    generate_poisson_surrogates : Homogeneous Poisson surrogates
    generate_jittered_spikes : Temporal jitter surrogates
    """
    from scipy.ndimage import uniform_filter1d

    generator = _ensure_rng(rng)
    n_time_bins, n_neurons = spike_counts.shape

    # Handle empty spike counts
    if n_time_bins == 0:
        for _ in range(n_surrogates):
            yield np.zeros((0, n_neurons), dtype=np.int64)
        return

    # Compute smoothed rates (time-varying lambda for Poisson)
    # Use uniform filter along time axis with 'nearest' mode for edge handling
    smoothed_counts = uniform_filter1d(
        spike_counts.astype(np.float64),
        size=smoothing_window,
        axis=0,
        mode="nearest",
    )

    # Ensure non-negative rates (should already be, but be safe)
    smoothed_counts = np.maximum(smoothed_counts, 0.0)

    for _ in range(n_surrogates):
        # Generate surrogate by sampling from Poisson with time-varying rates
        # Each (time_bin, neuron) pair uses its corresponding smoothed rate
        surrogate = generator.poisson(lam=smoothed_counts)
        yield surrogate.astype(np.int64)


def generate_jittered_spikes(
    spike_times: NDArray[np.float64],
    jitter_std: float,
    *,
    n_surrogates: int = 1000,
    rng: np.random.Generator | int | None = None,
    window: tuple[float, float] | None = None,
) -> Generator[NDArray[np.float64], None, None]:
    """Generate temporally jittered surrogate spike trains.

    Creates surrogate spike trains by adding Gaussian noise to each spike
    time. This tests whether precise spike timing is significant while
    preserving the approximate temporal distribution of spikes.

    **Tests that precise spike timing matters.** If observed effects depend
    on exact spike times, jittered surrogates should produce weaker effects.

    Parameters
    ----------
    spike_times : NDArray[np.float64], shape (n_spikes,)
        Sorted array of spike times for a single neuron.
    jitter_std : float
        Standard deviation of the Gaussian jitter added to each spike time.
        Units should match spike_times (typically seconds).
    n_surrogates : int, default=1000
        Number of surrogate spike trains to generate.
    rng : np.random.Generator | int | None, default=None
        Random number generator for reproducibility.

        - If Generator: Use directly
        - If int: Seed for ``np.random.default_rng()``
        - If None: Use default RNG (not reproducible)
    window : tuple[float, float] | None, default=None
        If provided, clip spike times to stay within (min_time, max_time).
        Useful to prevent spikes from jittering outside the recording period.

    Yields
    ------
    jittered_times : NDArray[np.float64], shape (n_spikes,)
        Spike times with Gaussian jitter applied (sorted).

    Notes
    -----
    - Each spike is independently jittered by adding Gaussian noise
    - Output spike times are always sorted (monotonically increasing)
    - Preserves the number of spikes exactly
    - Preserves approximate temporal distribution at scales > jitter_std
    - Destroys precise timing relationships at scales < jitter_std
    - Useful for testing phase locking, synchrony, or sequence timing

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.stats.surrogates import generate_jittered_spikes

    >>> spike_times = np.array([0.1, 0.15, 0.25, 0.4, 0.45, 0.7])
    >>> for i, jittered in enumerate(
    ...     generate_jittered_spikes(
    ...         spike_times, jitter_std=0.01, n_surrogates=3, rng=42
    ...     )
    ... ):
    ...     print(f"Surrogate {i}: n_spikes={len(jittered)}")
    Surrogate 0: n_spikes=6
    Surrogate 1: n_spikes=6
    Surrogate 2: n_spikes=6

    See Also
    --------
    generate_poisson_surrogates : Rate-matched surrogates (destroys all timing)
    neurospatial.stats.shuffle.shuffle_spikes_isi : ISI-preserving shuffle
    """
    generator = _ensure_rng(rng)
    n_spikes = len(spike_times)

    # Handle empty spike array
    if n_spikes == 0:
        for _ in range(n_surrogates):
            yield np.array([], dtype=np.float64)
        return

    for _ in range(n_surrogates):
        # Add Gaussian jitter to each spike time
        jitter = generator.normal(0.0, jitter_std, size=n_spikes)
        jittered_times = spike_times + jitter

        # Clip to window if specified
        if window is not None:
            jittered_times = np.clip(jittered_times, window[0], window[1])

        # Sort to maintain monotonicity
        jittered_times = np.sort(jittered_times)

        yield jittered_times


__all__ = [
    "generate_inhomogeneous_poisson_surrogates",
    "generate_jittered_spikes",
    "generate_poisson_surrogates",
]
