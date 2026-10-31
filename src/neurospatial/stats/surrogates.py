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
>>> for surrogate in generate_poisson_surrogates(spike_counts, n_surrogates=3, rng=42):
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

from neurospatial._exceptions import _format_error
from neurospatial.stats.shuffle import _validate_draw_count


def _validated_counts(spike_counts: NDArray[np.int64]) -> NDArray[np.float64]:
    """Spike counts as a float array, refusing what is not a count matrix."""
    counts = np.asarray(spike_counts, dtype=np.float64)
    if (
        counts.ndim != 2
        or not np.all(np.isfinite(counts))
        or np.any(counts < 0)
        or np.any(counts != np.round(counts))
    ):
        raise ValueError(
            _format_error(
                f"spike_counts must be a 2-D array of non-negative whole-number "
                f"counts, got shape {counts.shape}.",
                why="Why: Poisson surrogates resample spike counts per time bin; "
                "rates in Hz or NaN would change the surrogate's mean",
                fix="pass integer spike counts with shape (n_time_bins, n_neurons)",
            )
        )
    return counts


def generate_poisson_surrogates(
    spike_counts: NDArray[np.int64],
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
    - The least conservative null here: it removes all structure except each
      neuron's mean count, so beating it only rules out a constant-rate
      explanation.

    Raises
    ------
    ValueError
        If ``spike_counts`` is not a 2-D array of non-negative whole-number
        counts, or ``n_surrogates`` is not a positive integer.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.stats.surrogates import generate_poisson_surrogates

    >>> spike_counts = np.array([[0, 1], [2, 0], [1, 1]], dtype=np.int64)
    >>> for i, surrogate in enumerate(
    ...     generate_poisson_surrogates(spike_counts, n_surrogates=3, rng=42)
    ... ):
    ...     print(f"Surrogate {i}: shape={surrogate.shape}")
    Surrogate 0: shape=(3, 2)
    Surrogate 1: shape=(3, 2)
    Surrogate 2: shape=(3, 2)

    See Also
    --------
    generate_inhomogeneous_poisson_surrogates : Time-varying rate surrogates
    generate_jittered_spikes : Temporal jitter surrogates
    """
    counts = _validated_counts(spike_counts)
    _validate_draw_count(n_surrogates, "n_surrogates")
    # Each (time_bin, neuron) pair samples Poisson(mean count of its neuron).
    lam = np.broadcast_to(counts.mean(axis=0) if len(counts) else 0.0, counts.shape)
    return _poisson_draws(lam, n_surrogates, np.random.default_rng(rng))


def _poisson_draws(
    lam: NDArray[np.float64], n_surrogates: int, generator: np.random.Generator
) -> Generator[NDArray[np.int64], None, None]:
    """Yield Poisson count matrices with means ``lam``."""
    for _ in range(n_surrogates):
        yield generator.poisson(lam=lam).astype(np.int64, copy=False)


def generate_inhomogeneous_poisson_surrogates(
    spike_counts: NDArray[np.int64],
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
    conservative than homogeneous surrogates: it keeps slow rate modulation,
    so it tests for structure beyond it.

    Parameters
    ----------
    spike_counts : NDArray[np.int64], shape (n_time_bins, n_neurons)
        Original spike counts per neuron per time bin.
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

    Raises
    ------
    ValueError
        If ``spike_counts`` is not a 2-D array of non-negative whole-number
        counts, ``smoothing_window`` is not a positive integer, or
        ``n_surrogates`` is not a positive integer.

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
    ...         spike_counts, smoothing_window=3, n_surrogates=3, rng=42
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

    counts = _validated_counts(spike_counts)
    _validate_draw_count(smoothing_window, "smoothing_window")
    _validate_draw_count(n_surrogates, "n_surrogates")
    # Time-varying Poisson means: counts smoothed along time, edges extended.
    lam = (
        uniform_filter1d(counts, size=smoothing_window, axis=0, mode="nearest")
        if len(counts)
        else counts
    )
    return _poisson_draws(lam, n_surrogates, np.random.default_rng(rng))


def generate_jittered_spikes(
    spike_times: NDArray[np.float64],
    *,
    jitter_std: float,
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
        Finite spike times for a single neuron.
    jitter_std : float
        Standard deviation of the Gaussian jitter added to each spike time.
        Must be positive; units should match spike_times (typically seconds).
    n_surrogates : int, default=1000
        Number of surrogate spike trains to generate.
    rng : np.random.Generator | int | None, default=None
        Random number generator for reproducibility.

        - If Generator: Use directly
        - If int: Seed for ``np.random.default_rng()``
        - If None: Use default RNG (not reproducible)
    window : tuple[float, float] | None, default=None
        Half-open recording window ``[start, stop)`` the jittered spikes must
        stay in. Each spike's jitter is then drawn from the Gaussian
        truncated to the window, as if out-of-window draws were redrawn, so
        no spikes pile up at the edges. Every spike must lie inside it.

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

    Raises
    ------
    ValueError
        If ``spike_times`` is not a finite 1-D array, ``jitter_std`` is not
        positive and finite, ``n_surrogates`` is not a positive integer, or
        ``window`` is not a finite ``(start, stop)`` with start < stop that
        contains every spike.

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
    spike_times = np.asarray(spike_times, dtype=np.float64)
    if spike_times.ndim != 1 or not np.all(np.isfinite(spike_times)):
        raise ValueError(
            _format_error(
                f"spike_times must be a finite 1-D array, got shape "
                f"{spike_times.shape}.",
                why="Why: each spike time is jittered independently",
                fix="pass spike_times[np.isfinite(spike_times)] for one neuron",
            )
        )
    if not (np.isfinite(jitter_std) and jitter_std > 0):
        raise ValueError(
            _format_error(
                f"jitter_std={jitter_std!r} must be positive and finite.",
                why="Why: it is the standard deviation of the timing noise",
                fix="pass jitter_std in the units of spike_times, e.g. 0.01 s",
            )
        )
    _validate_draw_count(n_surrogates, "n_surrogates")
    if window is not None:
        start, stop = (float(v) for v in window)
        outside = (spike_times < start) | (spike_times >= stop)
        if not (np.isfinite(start) and np.isfinite(stop) and start < stop) or (
            outside.any()
        ):
            raise ValueError(
                _format_error(
                    f"window={window!r} must be a finite (start, stop) with "
                    f"start < stop that contains every spike; "
                    f"{int(outside.sum())} spike(s) fall outside it.",
                    why="Why: jittered spikes are kept inside the window, so "
                    "spikes must start there",
                    fix="drop spikes outside the recording window first, or "
                    "pass the window the spikes were recorded in",
                )
            )
        window = (start, stop)
    return _jitter_draws(
        spike_times, jitter_std, window, n_surrogates, np.random.default_rng(rng)
    )


def _jitter_draws(
    spike_times: NDArray[np.float64],
    jitter_std: float,
    window: tuple[float, float] | None,
    n_surrogates: int,
    generator: np.random.Generator,
) -> Generator[NDArray[np.float64], None, None]:
    """Yield sorted jittered trains, the jitter truncated to ``window``."""
    # Jitter leaves a sorted train nearly sorted, where a stable sort is close
    # to linear. Each draw is made in input order and moved with its spike.
    order = np.argsort(spike_times, kind="stable")
    spike_times = spike_times[order]
    if window is None:
        for _ in range(n_surrogates):
            noise = generator.normal(0.0, jitter_std, spike_times.size)[order]
            yield np.sort(spike_times + noise, kind="stable")
        return
    from scipy.special import ndtr, ndtri

    # Inverse-CDF sampling of N(0, jitter_std) truncated to the window. Each
    # spike lies inside it, so [low, high] contains 0.5 and stays accurate.
    low = ndtr((window[0] - spike_times) / jitter_std)
    width = ndtr((window[1] - spike_times) / jitter_std) - low
    last = np.nextafter(window[1], window[0])
    for _ in range(n_surrogates):
        u = low + generator.uniform(size=spike_times.size)[order] * width
        jittered = spike_times + jitter_std * ndtri(u)
        yield np.sort(np.clip(jittered, window[0], last), kind="stable")


__all__ = [
    "generate_inhomogeneous_poisson_surrogates",
    "generate_jittered_spikes",
    "generate_poisson_surrogates",
]
