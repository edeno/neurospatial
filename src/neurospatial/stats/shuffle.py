"""Shuffle-based significance testing for statistical analysis.

This module provides statistical shuffling procedures to establish null
distributions and test the significance of neural and behavioral data.
These methods are essential for replay analysis, decoding validation, and
behavioral analysis to rule out non-specific factors like rate biases.

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
Shuffles of rate maps and posteriors for replay tests (cell identity,
place-field and posterior shifts) are in :mod:`neurospatial.replay`.

| Category | Null Hypothesis Tested |
|----------|----------------------|
| **Temporal** | Sequential structure is not significant |
| **Trial** | Trial identity is not significant |
| **ISI** | Inter-spike interval ordering is not significant |
| **Circular Spike Time** | Spike alignment to behavior exceeds shifted alignment |


Imports
-------
>>> from neurospatial.stats.shuffle import shuffle_time_bins
>>> from neurospatial.stats import shuffle_time_bins, compute_shuffle_pvalue

Examples
--------
>>> import numpy as np
>>> from neurospatial.stats.shuffle import shuffle_time_bins

>>> # Typical usage pattern
>>> spike_counts = np.array([[0, 1], [2, 0], [1, 1]], dtype=np.int64)
>>> null_scores = []
>>> for shuffled in shuffle_time_bins(spike_counts, n_shuffles=100, rng=42):
...     # result = decode_position(env, shuffled, models, dt=dt)
...     # null_scores.append(compute_sequence_score(result))
...     pass

See Also
--------
neurospatial.stats.circular : Circular statistics
neurospatial.stats.surrogates : Surrogate data generation
"""

from __future__ import annotations

import warnings
from collections.abc import Generator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error
from neurospatial._intervals import as_intervals
from neurospatial._results import ReadOnlyArraysMixin

# Import internal utilities from canonical location
from neurospatial.stats._utils import _ensure_rng

if TYPE_CHECKING:
    from matplotlib.axes import Axes


# =============================================================================
# I. Temporal Order Shuffles - Test sequential structure within events
# =============================================================================


def _validate_circular_shift_settings(n_shuffles: int, min_shift: float) -> None:
    """Reject invalid counts and offsets before allocating shuffle output."""
    problems = []
    if (
        isinstance(n_shuffles, (bool, np.bool_))
        or not isinstance(n_shuffles, (int, np.integer))
        or n_shuffles < 1
    ):
        problems.append(f"n_shuffles={n_shuffles!r} must be a positive integer")
    if (
        not isinstance(min_shift, (int, float, np.integer, np.floating))
        or not np.isfinite(min_shift)
        or min_shift < 0
    ):
        problems.append(f"min_shift={min_shift!r} must be finite and non-negative")
    if problems:
        raise ValueError(
            _format_error(
                "; ".join(problems) + ".",
                why="Why: a null distribution needs a positive draw count and a valid time offset",
                fix="pass n_shuffles=1000 and a finite min_shift>=0 in seconds",
            )
        )


def shuffle_spike_times_circular(
    spike_times: NDArray[np.float64],
    windows: NDArray[np.float64],
    *,
    n_shuffles: int = 1000,
    min_shift: float = 20.0,
    rng: np.random.Generator | int | None = None,
) -> Generator[NDArray[np.float64], None, None]:
    """Circularly shift spikes on the joined analyzed-time axis.

    Join valid recording windows into one circular axis. Each draw adds one
    uniform offset in [min_shift, T-min_shift] to all retained spikes. Counts
    and circular spacings on that compressed clock are preserved; spikes never
    enter a recording gap. Physical spacings across removed gaps can differ.

    Parameters
    ----------
    spike_times : ndarray, shape (n_spikes,)
        Spike timestamps in seconds. Spikes outside windows are dropped.
    windows : ndarray, shape (n_windows, 2)
        Half-open analyzed recording windows in seconds, on the spike clock.
        The shared interval normalizer sorts and merges touching/overlapping rows.
    n_shuffles : int, default=1000
        Positive number of shifted trains to yield.
    min_shift : float, default=20.0
        Minimum offset in either direction, in seconds of analyzed time.
    rng : numpy.random.Generator, int or None, default=None
        Random generator or reproducible integer seed.

    Yields
    ------
    ndarray, shape (n_spikes_in_windows,)
        Sorted shifted spike times on the original recording clock.

    Raises
    ------
    ValueError
        If windows/settings are invalid or analyzed time is not longer than
        twice min_shift.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.stats import shuffle_spike_times_circular
    >>> draws = list(
    ...     shuffle_spike_times_circular(
    ...         np.array([1.0, 10.0, 205.0]),
    ...         np.array([[0.0, 100.0], [200.0, 300.0]]),
    ...         n_shuffles=3,
    ...         rng=7,
    ...     )
    ... )
    >>> [len(train) for train in draws]
    [3, 3, 3]
    """
    _validate_circular_shift_settings(n_shuffles, min_shift)
    try:
        normalized = as_intervals(windows, name="windows")
    except ValueError as exc:
        problems = str(exc).split("\nWhy:", 1)[0]
        raise ValueError(
            _format_error(
                f"shuffle_spike_times_circular: {problems}",
                why="Why: circular shifting requires finite half-open analyzed-time windows with start < stop",
                fix="pass windows=np.array([[0.0, 100.0], [200.0, 300.0]]) in seconds",
            )
        ) from None
    if normalized is None:
        raise ValueError(
            _format_error(
                "windows must contain analyzed recording intervals.",
                why="Why: spikes need explicit valid time bounds for circular shifting",
                fix="pass windows=np.array([[start, stop]]) in seconds",
            )
        )
    spike_times = np.asarray(spike_times, dtype=np.float64)
    if spike_times.ndim != 1 or not np.all(np.isfinite(spike_times)):
        raise ValueError(
            _format_error(
                f"spike_times must be a finite 1-D array, got shape {spike_times.shape}.",
                why="Why: circular shifting operates on one spike train in seconds",
                fix="pass a finite 1-D spike_times array",
            )
        )
    offsets = np.concatenate([[0.0], np.cumsum(normalized[:, 1] - normalized[:, 0])])
    total = float(offsets[-1])
    if total <= 2.0 * min_shift:
        raise ValueError(
            _format_error(
                f"Analyzed time is {total:.1f} s, not longer than 2 * min_shift = {2.0 * min_shift:.1f} s.",
                why="Why: no circular offset can move spikes by at least min_shift in both directions",
                fix=f"pass min_shift={total / 4:.1f}, or analyze a longer recording",
            )
        )
    idx = np.searchsorted(normalized[:, 0], spike_times, side="right") - 1
    inside = (idx >= 0) & (spike_times < normalized[np.maximum(idx, 0), 1])
    idx = idx[inside]
    compressed = offsets[idx] + (spike_times[inside] - normalized[idx, 0])
    # Validation above runs at call time; only the draws are lazy.
    return _circular_shift_draws(
        compressed,
        normalized,
        offsets,
        total,
        n_shuffles=n_shuffles,
        min_shift=min_shift,
        generator=_ensure_rng(rng),
    )


def _circular_shift_draws(
    compressed: NDArray[np.float64],
    windows: NDArray[np.float64],
    offsets: NDArray[np.float64],
    total: float,
    *,
    n_shuffles: int,
    min_shift: float,
    generator: np.random.Generator,
) -> Generator[NDArray[np.float64], None, None]:
    """Yield shifted spike trains from validated, compressed-clock inputs."""
    for _ in range(n_shuffles):
        wrapped = np.mod(
            compressed + generator.uniform(min_shift, total - min_shift), total
        )
        j = np.minimum(
            np.searchsorted(offsets, wrapped, side="right") - 1, len(windows) - 1
        )
        shifted = windows[j, 0] + (wrapped - offsets[j])
        # A sum on a large absolute clock may round up to an excluded stop.
        shifted = np.minimum(shifted, np.nextafter(windows[j, 1], windows[j, 0]))
        yield np.sort(shifted)


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
    >>> from neurospatial.stats.shuffle import shuffle_time_bins

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
    >>> from neurospatial.stats.shuffle import shuffle_time_bins_coherent

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
    neurospatial.replay.shuffle_cell_identity : Shuffle neuron-to-place-field mapping
    """
    # This is an exact alias of shuffle_time_bins: permuting whole rows
    # already applies one coherent permutation across all neurons. We
    # delegate rather than duplicate the body so the two cannot silently
    # diverge; the separate public name documents the intent (testing
    # temporal structure while preserving population co-activation).
    yield from shuffle_time_bins(spike_counts, n_shuffles=n_shuffles, rng=rng)


# =============================================================================
# II. Significance Testing Functions
# =============================================================================


@dataclass(frozen=True, eq=False)
class ShuffleTestResult(ReadOnlyArraysMixin):
    """Result container for shuffle-based significance testing.

    A frozen dataclass containing the observed score, null distribution,
    and computed statistics (p-value, z-score) from a shuffle test.

    Parameters
    ----------
    observed_score : float
        The score computed from the original (non-shuffled) data.
    null_scores : NDArray[np.float64]
        Array of scores computed from shuffled data, forming the null
        distribution. Stored as a read-only float64 copy.
    p_value : float
        Monte Carlo p-value with correction: (k + 1) / (n + 1) where k is
        the count of finite null scores at least as extreme as observed and
        n is the number of finite null scores (``n_shuffles`` unless some
        shuffles produced NaN).
    z_score : float
        Standard score: (observed - mean(null)) / std(null), over the finite
        null scores. NaN if they have zero variance.
    shuffle_type : str
        Name of the shuffle method used (e.g., "time_bins", "cell_identity").
    n_shuffles : int
        Number of shuffles performed.

    Attributes
    ----------
    is_significant : bool
        True if p_value < 0.05 (convenience property).

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.stats.shuffle import ShuffleTestResult

    >>> result = ShuffleTestResult(
    ...     observed_score=5.0,
    ...     null_scores=np.array([1.0, 2.0, 3.0, 4.0]),
    ...     p_value=0.2,
    ...     z_score=1.5,
    ...     shuffle_type="time_bins",
    ...     n_shuffles=4,
    ... )
    >>> result.is_significant
    False
    >>> result.p_value
    0.2

    See Also
    --------
    compute_shuffle_pvalue : Compute Monte Carlo p-value from null distribution.
    compute_shuffle_zscore : Compute z-score from null distribution.
    """

    observed_score: float
    null_scores: NDArray[np.float64]
    p_value: float
    z_score: float
    shuffle_type: str
    n_shuffles: int

    def __post_init__(self) -> None:
        """Own the null distribution as a read-only float64 copy."""
        object.__setattr__(
            self, "null_scores", np.asarray(self.null_scores, dtype=np.float64)
        )
        self._own_arrays()

    @property
    def is_significant(self) -> bool:
        """Return True if p_value < 0.05.

        Returns
        -------
        bool
            True if the result is statistically significant at alpha=0.05.
        """
        return self.p_value < 0.05

    def plot(self, ax: Axes | None = None, **kwargs) -> Axes:
        """Plot the null distribution with observed score.

        Creates a histogram of the null distribution with a vertical line
        showing the observed score and annotations for p-value and z-score.

        Parameters
        ----------
        ax : matplotlib.axes.Axes | None, default=None
            Axes to plot on. If None, creates a new figure.
        **kwargs
            Additional keyword arguments passed to matplotlib's hist().

        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the plot.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.stats.shuffle import ShuffleTestResult

        >>> result = ShuffleTestResult(
        ...     observed_score=5.0,
        ...     null_scores=np.random.default_rng(42).normal(2.0, 1.0, 100),
        ...     p_value=0.01,
        ...     z_score=3.0,
        ...     shuffle_type="time_bins",
        ...     n_shuffles=100,
        ... )
        >>> ax = result.plot()  # doctest: +SKIP
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()

        # Default histogram settings
        hist_kwargs = {
            "bins": 30,
            "alpha": 0.7,
            "color": "steelblue",
            "edgecolor": "white",
        }
        hist_kwargs.update(kwargs)

        # Plot null distribution
        ax.hist(self.null_scores, **hist_kwargs)  # type: ignore[arg-type]

        # Add observed score line
        ax.axvline(
            self.observed_score,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Observed: {self.observed_score:.3f}",
        )

        # Add annotations
        sig_marker = "*" if self.is_significant else ""
        ax.set_title(
            f"Shuffle Test ({self.shuffle_type})\n"
            f"p = {self.p_value:.4f}{sig_marker}, z = {self.z_score:.2f}"
        )
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")
        ax.legend()

        return ax


def compute_shuffle_pvalue(
    observed: float,
    null_scores: NDArray[np.float64],
    *,
    tail: Literal["greater", "less", "two-sided"] = "greater",
) -> float:
    """Compute Monte Carlo p-value with Phipson-Smyth correction.

    Computes the probability of observing a score at least as extreme as
    the observed value under the null distribution using the formula
    (k + 1) / (n + 1), which provides an unbiased estimate and avoids
    p-values of exactly zero.

    Parameters
    ----------
    observed : float
        The observed score from the original (non-shuffled) data.
    null_scores : NDArray[np.float64]
        Array of scores from shuffled data, forming the null distribution.
    tail : {"greater", "less", "two-sided"}, default="greater"
        Direction of the test:

        - "greater": Tests if observed is significantly greater than null
          (counts null >= observed).
        - "less": Tests if observed is significantly less than null
          (counts null <= observed).
        - "two-sided": Tests if observed is significantly different from
          null in either direction. Computed as 2 * min(p_greater, p_less),
          capped at 1.0.

    Returns
    -------
    float
        The Monte Carlo p-value in the range (0, 1].

    Notes
    -----
    The Phipson-Smyth correction (k + 1) / (n + 1) ensures that:

    1. The p-value is never exactly zero, which is important for downstream
       corrections (e.g., Bonferroni, FDR).
    2. The estimator is unbiased for uniformly distributed test statistics.

    For replay analysis, "greater" is typically used because we expect
    significant sequences to have higher scores than shuffled controls.

    Non-finite null scores (NaN or Inf) are dropped with a ``UserWarning``
    and excluded from ``n`` before the p-value is computed; otherwise they
    would silently bias the p-value toward significance. If no finite null
    scores remain, a ``ValueError`` is raised.

    References
    ----------
    .. [1] Phipson, B., & Smyth, G. K. (2010). Permutation P-values should
           never be zero: calculating exact P-values when permutations are
           randomly drawn. Statistical Applications in Genetics and Molecular
           Biology, 9(1).

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.stats.shuffle import compute_shuffle_pvalue

    >>> # Observed score higher than all null values
    >>> observed = 10.0
    >>> null = np.array([1.0, 2.0, 3.0, 4.0])
    >>> compute_shuffle_pvalue(observed, null, tail="greater")
    0.2

    >>> # Two-sided test
    >>> compute_shuffle_pvalue(observed, null, tail="two-sided")
    0.4

    See Also
    --------
    compute_shuffle_zscore : Compute z-score from null distribution.
    ShuffleTestResult : Container for shuffle test results.
    """
    if not np.isfinite(observed):
        raise ValueError(
            f"observed must be a finite value, got {observed!r}. A non-finite "
            "observed score makes all comparisons against the null False, which "
            "would silently return the floor p-value 1/(n+1)."
        )

    null_scores = np.asarray(null_scores, dtype=np.float64).ravel()
    finite = np.isfinite(null_scores)
    n_total = null_scores.size
    if not finite.all():
        n_dropped = int(n_total - finite.sum())
        warnings.warn(
            f"Dropped {n_dropped} non-finite null score(s) out of {n_total} "
            f"before computing the p-value. Using {int(finite.sum())} finite "
            f"null scores; non-finite nulls would otherwise bias the p-value "
            f"toward significance.",
            category=UserWarning,
            stacklevel=2,
        )
        null_scores = null_scores[finite]

    n = len(null_scores)
    if n == 0:
        raise ValueError(
            "null_scores contains no finite values; cannot compute a "
            "shuffle p-value. Check that the shuffle produced valid scores."
        )

    if tail == "greater":
        # Count how many null values are >= observed
        k = int(np.sum(null_scores >= observed))
        return float((k + 1) / (n + 1))

    elif tail == "less":
        # Count how many null values are <= observed
        k = int(np.sum(null_scores <= observed))
        return float((k + 1) / (n + 1))

    elif tail == "two-sided":
        # Compute both tails and take 2 * min, capped at 1.0
        k_greater = int(np.sum(null_scores >= observed))
        k_less = int(np.sum(null_scores <= observed))
        p_greater = (k_greater + 1) / (n + 1)
        p_less = (k_less + 1) / (n + 1)
        return float(min(2 * min(p_greater, p_less), 1.0))

    else:
        raise ValueError(
            f"tail must be 'greater', 'less', or 'two-sided', got '{tail}'"
        )


def compute_shuffle_zscore(
    observed: float,
    null_scores: NDArray[np.float64],
) -> float:
    """Compute z-score of observed value relative to null distribution.

    Calculates the standard score: (observed - mean(null)) / std(null).
    Returns NaN if the null distribution has zero variance.

    Parameters
    ----------
    observed : float
        The observed score from the original (non-shuffled) data.
    null_scores : NDArray[np.float64]
        Array of scores from shuffled data, forming the null distribution.

    Returns
    -------
    float
        The z-score (standard score). NaN if the finite null scores have
        zero variance, or fewer than two of them are finite.

    Notes
    -----
    Non-finite null scores (NaN or Inf), such as shuffles whose score is
    undefined, are left out of the mean and standard deviation, as
    :func:`compute_shuffle_pvalue` leaves them out of the p-value.

    The z-score indicates how many standard deviations the observed value
    is from the mean of the null distribution. Positive values indicate
    the observed is above the null mean; negative values indicate below.

    Common interpretation thresholds:

    - |z| > 1.96: Approximately p < 0.05 (two-tailed) if null is normal
    - |z| > 2.58: Approximately p < 0.01 (two-tailed) if null is normal
    - |z| > 3.0: Strong evidence against null hypothesis

    Unlike p-values, z-scores are useful for comparing effect sizes across
    different analyses.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.stats.shuffle import compute_shuffle_zscore

    >>> null = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> observed = 5.0
    >>> z = compute_shuffle_zscore(observed, null)
    >>> bool(np.isclose(z, (5.0 - 3.0) / np.std(null)))
    True

    >>> # Zero variance returns NaN
    >>> compute_shuffle_zscore(5.0, np.array([3.0, 3.0, 3.0]))
    nan

    See Also
    --------
    compute_shuffle_pvalue : Compute Monte Carlo p-value from null distribution.
    ShuffleTestResult : Container for shuffle test results.
    """
    null_scores = np.asarray(null_scores, dtype=np.float64).ravel()
    null_scores = null_scores[np.isfinite(null_scores)]
    if null_scores.size < 2:
        return float("nan")
    mean_null = float(np.mean(null_scores))
    std_null = float(np.std(null_scores))

    # Handle zero variance (all values equal)
    if std_null == 0.0:
        return float("nan")

    return float((observed - mean_null) / std_null)


# =============================================================================
# III. Trial and Behavioral Shuffles
# =============================================================================


def shuffle_trials(
    trial_labels: NDArray[np.int64],
    *,
    n_shuffles: int = 1000,
    rng: np.random.Generator | int | None = None,
) -> Generator[NDArray[np.int64], None, None]:
    """Shuffle trial labels to test trial identity significance.

    Randomly permutes trial labels to create a null distribution for testing
    whether observed effects depend on trial identity. Useful for testing
    learning effects, trial-type differences, or temporal dependencies.

    Parameters
    ----------
    trial_labels : NDArray[np.int64], shape (n_samples,)
        Array of trial labels/indices for each sample (e.g., timepoint).
        Labels can be any integers (e.g., 0, 1, 2 or trial IDs).
    n_shuffles : int, default=1000
        Number of shuffled versions to generate.
    rng : np.random.Generator | int | None, default=None
        Random number generator for reproducibility.

        - If Generator: Use directly
        - If int: Seed for ``np.random.default_rng()``
        - If None: Use default RNG (not reproducible)

    Yields
    ------
    shuffled_labels : NDArray[np.int64], shape (n_samples,)
        Trial labels with permuted order.

    Notes
    -----
    - Permutes the entire label array (all samples)
    - Preserves the distribution of labels (same counts per label)
    - Destroys the temporal association between samples and trials
    - Useful for testing trial-type effects or learning-related changes

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.stats.shuffle import shuffle_trials

    >>> trial_labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    >>> for i, shuffled in enumerate(
    ...     shuffle_trials(trial_labels, n_shuffles=3, rng=42)
    ... ):
    ...     print(
    ...         f"Shuffle {i}: unique values preserved = {set(shuffled) == set(trial_labels)}"
    ...     )
    Shuffle 0: unique values preserved = True
    Shuffle 1: unique values preserved = True
    Shuffle 2: unique values preserved = True

    See Also
    --------
    shuffle_time_bins : Shuffle temporal order of spike counts.
    neurospatial.replay.shuffle_cell_identity : Shuffle neuron-to-place-field mapping.
    """
    generator = _ensure_rng(rng)
    n_samples = len(trial_labels)

    for _ in range(n_shuffles):
        # Generate a random permutation of all sample indices
        perm = generator.permutation(n_samples)
        yield trial_labels[perm].copy()


def shuffle_spikes_isi(
    spike_times: NDArray[np.float64],
    *,
    n_shuffles: int = 1000,
    rng: np.random.Generator | int | None = None,
) -> Generator[NDArray[np.float64], None, None]:
    """Shuffle inter-spike intervals to test ISI ordering significance.

    Creates surrogate spike trains by randomly permuting the inter-spike
    intervals (ISIs). The first spike time is preserved, and ISIs are
    shuffled and cumulatively summed to create new spike times. This
    preserves the ISI distribution while destroying temporal correlations.

    Parameters
    ----------
    spike_times : NDArray[np.float64], shape (n_spikes,)
        Sorted array of spike times for a single neuron.
    n_shuffles : int, default=1000
        Number of shuffled versions to generate.
    rng : np.random.Generator | int | None, default=None
        Random number generator for reproducibility.

        - If Generator: Use directly
        - If int: Seed for ``np.random.default_rng()``
        - If None: Use default RNG (not reproducible)

    Yields
    ------
    shuffled_times : NDArray[np.float64], shape (n_spikes,)
        Spike times with shuffled ISIs (same first spike time).

    Notes
    -----
    - Preserves the first spike time exactly
    - Preserves the ISI distribution (same ISI values, different order)
    - Destroys temporal correlations between consecutive ISIs
    - Output spike times are always sorted (monotonically increasing)
    - Useful for testing ISI-dependent effects (e.g., burst detection,
      adaptation)

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.stats.shuffle import shuffle_spikes_isi

    >>> spike_times = np.array([0.1, 0.15, 0.25, 0.4, 0.45, 0.7])
    >>> for i, shuffled in enumerate(
    ...     shuffle_spikes_isi(spike_times, n_shuffles=3, rng=42)
    ... ):
    ...     print(
    ...         f"Shuffle {i}: first spike preserved = {shuffled[0] == spike_times[0]}"
    ...     )
    Shuffle 0: first spike preserved = True
    Shuffle 1: first spike preserved = True
    Shuffle 2: first spike preserved = True

    See Also
    --------
    shuffle_time_bins : Shuffle temporal order of binned spike counts.
    generate_poisson_surrogates : Generate Poisson surrogate spike trains.
    """
    generator = _ensure_rng(rng)

    # Handle edge cases
    if len(spike_times) <= 1:
        for _ in range(n_shuffles):
            yield spike_times.copy()
        return

    # Compute inter-spike intervals
    isis = np.diff(spike_times)
    first_spike = spike_times[0]

    for _ in range(n_shuffles):
        # Shuffle the ISIs
        shuffled_isis = generator.permutation(isis)
        # Reconstruct spike times from shuffled ISIs
        shuffled_times = np.concatenate(
            [[first_spike], first_spike + np.cumsum(shuffled_isis)]
        )
        yield shuffled_times
