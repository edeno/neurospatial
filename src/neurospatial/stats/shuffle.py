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
| **Temporal** | The order of time bins within an event carries no information |
| **Trial** | The effect does not depend on which trial is which |
| **ISI** | The order of inter-spike intervals carries no information |
| **Circular Spike Time** | Spike timing is unrelated to behavior beyond the train's own spike-time spacing |


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
...     # null_scores.append(score_event(shuffled))  # your event score
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
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error
from neurospatial._intervals import as_intervals
from neurospatial._results import ReadOnlyArraysMixin, ResultMixin
from neurospatial._stacklevel import _external_stacklevel
from neurospatial.stats._utils import _ensure_rng

if TYPE_CHECKING:
    from matplotlib.axes import Axes


# =============================================================================
# I. Temporal Order Shuffles - Test sequential structure within events
# =============================================================================


def _is_positive_count(value: object) -> bool:
    """Whether ``value`` is an integer of at least 1 (bools are not counts)."""
    return bool(
        isinstance(value, (int, np.integer))
        and not isinstance(value, (bool, np.bool_))
        and value >= 1
    )


def _validate_draw_count(value: int, name: str) -> None:
    """Reject a shuffle or surrogate count that would yield nothing."""
    if not _is_positive_count(value):
        raise ValueError(
            _format_error(
                f"{name}={value!r} must be a positive integer.",
                why="Why: a null distribution needs at least one draw",
                fix=f"pass {name}=1000",
            )
        )


def _validate_circular_shift_settings(n_shuffles: int, min_shift: float) -> None:
    """Reject invalid counts and offsets before allocating shuffle output."""
    problems = []
    if not _is_positive_count(n_shuffles):
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
        Spike timestamps in seconds. Spikes outside windows are dropped; a
        warning says so when every spike falls outside them.
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
    if spike_times.size and not inside.any():
        warnings.warn(
            f"shuffle_spike_times_circular: none of the {spike_times.size} spikes "
            f"falls inside the windows ([{normalized[0, 0]:g}, "
            f"{normalized[-1, 1]:g}) s), so every shuffle is empty. Check that "
            "spike_times and windows share a clock and units (seconds).",
            UserWarning,
            stacklevel=_external_stacklevel(),
        )
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

    Raises
    ------
    ValueError
        If ``n_shuffles`` is not a positive integer.

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
    shuffle_time_bins_coherent : The same shuffle under a name that stresses
        that all neurons share one permutation.
    """
    _validate_draw_count(n_shuffles, "n_shuffles")
    return _row_permutations(np.asarray(spike_counts), n_shuffles, _ensure_rng(rng))


def _row_permutations(
    values: NDArray[Any], n_shuffles: int, generator: np.random.Generator
) -> Generator[NDArray[Any], None, None]:
    """Yield ``values`` with its rows (first axis) randomly permuted."""
    for _ in range(n_shuffles):
        yield values[generator.permutation(len(values))]


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

    Raises
    ------
    ValueError
        If ``n_shuffles`` is not a positive integer.

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
    # Permuting whole rows already applies one permutation to all neurons;
    # the separate public name documents that intent.
    return shuffle_time_bins(spike_counts, n_shuffles=n_shuffles, rng=rng)


# =============================================================================
# II. Significance Testing Functions
# =============================================================================


_TAILS = ("greater", "less", "two-sided")


@dataclass(frozen=True, repr=False, eq=False)
class ShuffleTestResult(ResultMixin, ReadOnlyArraysMixin):
    """Result container for shuffle-based significance testing.

    A frozen dataclass containing the observed score, null distribution,
    and computed statistics (p-value, z-score) from a shuffle test. Build
    one from scores with :meth:`from_null`.

    Parameters
    ----------
    observed_score : float
        The score computed from the original (non-shuffled) data.
    null_scores : NDArray[np.float64], shape (n_shuffles,)
        Score of each shuffle, forming the null distribution. NaN marks a
        shuffle whose score is undefined. Stored as a read-only float64 copy.
    p_value : float
        Monte Carlo p-value (k + 1) / (n + 1) in (0, 1], where k counts the
        defined null scores at least as extreme as observed in the direction
        of ``tail`` and n is ``n_shuffles_used``. NaN when the test is
        undefined: a non-finite observed score or no defined null score.
    z_score : float
        Standard score: (observed - mean(null)) / std(null), over the finite
        null scores. NaN if they have zero variance, fewer than two are
        finite, or a null score is infinite.
    shuffle_type : str
        Name of the shuffle method used (e.g., "time_bins", "cell_identity").
    n_shuffles : int
        Number of shuffles performed, ``len(null_scores)``, including any
        with an undefined score.
    tail : {"greater", "less", "two-sided"}, default="greater"
        Direction of the test the p-value belongs to.

    Attributes
    ----------
    n_shuffles_used : int
        Shuffles with a defined (non-NaN) score, the n of the p-value.

    Raises
    ------
    ValueError
        If ``null_scores`` is not 1-D, ``n_shuffles`` is not its length,
        ``p_value`` is neither in (0, 1] nor NaN, or ``tail`` is unknown.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.stats.shuffle import ShuffleTestResult

    >>> result = ShuffleTestResult.from_null(
    ...     5.0, np.array([1.0, 2.0, 3.0, 4.0]), shuffle_type="time_bins"
    ... )
    >>> result.p_value
    0.2
    >>> result.is_significant()
    False
    >>> result.is_significant(alpha=0.25)
    True

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
    tail: Literal["greater", "less", "two-sided"] = "greater"

    def __post_init__(self) -> None:
        """Check the fields agree, then own the null as a read-only copy."""
        null = np.asarray(self.null_scores, dtype=np.float64)
        problems = []
        if null.ndim != 1:
            problems.append(f"null_scores must be 1-D, got shape {null.shape}")
        elif self.n_shuffles != null.size:
            problems.append(
                f"n_shuffles={self.n_shuffles!r} must equal len(null_scores)="
                f"{null.size}"
            )
        if not (np.isnan(self.p_value) or 0.0 < self.p_value <= 1.0):
            problems.append(f"p_value={self.p_value!r} must be in (0, 1] or NaN")
        if self.tail not in _TAILS:
            problems.append(f"tail={self.tail!r} must be one of {_TAILS}")
        if problems:
            raise ValueError(
                _format_error(
                    "ShuffleTestResult: " + "; ".join(problems) + ".",
                    why="a result's p-value, z-score and counts must describe its "
                    "null scores",
                    fix="build it with ShuffleTestResult.from_null(observed, "
                    "null_scores, shuffle_type=...)",
                )
            )
        object.__setattr__(self, "null_scores", null)
        self._own_arrays()

    @classmethod
    def from_null(
        cls,
        observed_score: float,
        null_scores: NDArray[np.float64],
        *,
        shuffle_type: str,
        tail: Literal["greater", "less", "two-sided"] = "greater",
        decimals: int | None = None,
    ) -> ShuffleTestResult:
        """Test an observed score against the scores of its shuffles.

        Parameters
        ----------
        observed_score : float
            The score of the original (non-shuffled) data.
        null_scores : NDArray[np.float64], shape (n_shuffles,)
            The score of each shuffle. NaN scores (shuffles whose score is
            undefined) are left out of the p-value and z-score, with a
            warning; ``n_shuffles_used`` counts the rest.
        shuffle_type : str
            Name of the shuffle method (e.g., "time_bins", "cell_identity").
        tail : {"greater", "less", "two-sided"}, default="greater"
            Direction of the test, as for :func:`compute_shuffle_pvalue`.
        decimals : int or None, default=None
            Compare the observed and shuffled scores rounded to this many
            decimals for the p-value, so that scores equal up to
            floating-point error tie. None compares them as they are.

        Returns
        -------
        ShuffleTestResult
            The p-value of :func:`compute_shuffle_pvalue` and the z-score of
            :func:`compute_shuffle_zscore`, over the defined null scores.

        Raises
        ------
        ValueError
            If ``observed_score`` is not finite, no null score is defined,
            ``tail`` is unknown, or ``null_scores`` is not 1-D.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.stats.shuffle import ShuffleTestResult
        >>> result = ShuffleTestResult.from_null(
        ...     10.0, np.array([1.0, 2.0, 3.0, 4.0]), shuffle_type="time_bins"
        ... )
        >>> result.p_value, result.n_shuffles
        (0.2, 4)
        """
        null = np.asarray(null_scores, dtype=np.float64)
        undefined = np.isnan(null)
        defined = null[~undefined]
        if undefined.any() and defined.size:
            warnings.warn(
                f"{shuffle_type}: {int(undefined.sum())} of {null.size} shuffles "
                f"have an undefined (NaN) score and are left out; the p-value "
                f"counts the other {defined.size} (n_shuffles_used).",
                UserWarning,
                stacklevel=_external_stacklevel(),
            )
        if decimals is None:
            p_value = compute_shuffle_pvalue(observed_score, defined, tail=tail)
        else:
            p_value = compute_shuffle_pvalue(
                round(observed_score, decimals), np.round(defined, decimals), tail=tail
            )
        return cls(
            observed_score=observed_score,
            null_scores=null,
            p_value=p_value,
            z_score=compute_shuffle_zscore(observed_score, defined),
            shuffle_type=shuffle_type,
            n_shuffles=null.size,
            tail=tail,
        )

    @property
    def n_shuffles_used(self) -> int:
        """Shuffles with a defined (non-NaN) score: the n of the p-value."""
        return int(np.sum(~np.isnan(self.null_scores)))

    def is_significant(self, alpha: float = 0.05) -> bool:
        """Return True if ``p_value < alpha``.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level.

        Returns
        -------
        bool
            True if the result is significant at ``alpha``; False when the
            p-value is NaN.
        """
        return bool(self.p_value < alpha)

    def summary(self) -> dict[str, Any]:
        """Flat dict of scalar headline metrics for this result."""
        return {
            "observed_score": float(self.observed_score),
            "p_value": float(self.p_value),
            "z_score": float(self.z_score),
            "tail": self.tail,
            "shuffle_type": self.shuffle_type,
            "n_shuffles": self.n_shuffles,
            "n_shuffles_used": self.n_shuffles_used,
        }

    def plot(self, ax: Axes | None = None, **kwargs) -> Axes:
        """Plot the null distribution with observed score.

        Creates a histogram of the finite null scores with a vertical line
        showing the observed score and annotations for the p-value, its tail
        and the z-score. Undefined (NaN) and infinite null scores cannot be
        drawn; the title gives how many are left out.

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

        >>> result = ShuffleTestResult.from_null(
        ...     5.0,
        ...     np.random.default_rng(42).normal(2.0, 1.0, 100),
        ...     shuffle_type="time_bins",
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

        finite = self.null_scores[np.isfinite(self.null_scores)]
        ax.hist(finite, **hist_kwargs)  # type: ignore[arg-type]

        # Add observed score line
        ax.axvline(
            self.observed_score,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Observed: {self.observed_score:.3f}",
        )

        # Add annotations
        sig_marker = "*" if self.is_significant() else ""
        n_hidden = self.n_shuffles - finite.size
        hidden = f", {n_hidden} non-finite not drawn" if n_hidden else ""
        ax.set_title(
            f"Shuffle Test ({self.shuffle_type}, {self.tail})\n"
            f"p = {self.p_value:.4f}{sig_marker}, z = {self.z_score:.2f}{hidden}"
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
    (k + 1) / (n + 1). Counting the observed data as one draw from the null
    makes the p-value valid (never anti-conservative) and never zero.

    Parameters
    ----------
    observed : float
        The observed score from the original (non-shuffled) data. Must be
        finite.
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

    Raises
    ------
    ValueError
        If ``observed`` is not finite, ``tail`` is unknown, or no null score
        is defined (all NaN or empty).

    Notes
    -----
    The Phipson-Smyth correction (k + 1) / (n + 1) ensures that:

    1. The p-value is never exactly zero, which is important for downstream
       corrections (e.g., Bonferroni, FDR).
    2. P(p <= alpha) <= alpha under the null: the test is valid, and
       slightly conservative as an estimate of the exact permutation p-value.

    For replay analysis, "greater" is typically used because we expect
    significant sequences to have higher scores than shuffled controls.

    NaN null scores (shuffles whose score is undefined) are dropped with a
    ``UserWarning`` and excluded from ``n``. Infinite null scores are
    defined, extreme scores and are counted like any other.

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
    if tail not in _TAILS:
        raise ValueError(
            f"tail must be 'greater', 'less', or 'two-sided', got '{tail}'"
        )

    null_scores = np.asarray(null_scores, dtype=np.float64).ravel()
    undefined = np.isnan(null_scores)
    if undefined.all():
        raise ValueError(
            "null_scores contains no defined (non-NaN) values; cannot compute a "
            "shuffle p-value. Check that the shuffle produced valid scores."
        )
    if undefined.any():
        warnings.warn(
            f"Dropped {int(undefined.sum())} undefined (NaN) null score(s) out "
            f"of {null_scores.size} before computing the p-value; it counts the "
            f"other {int((~undefined).sum())}.",
            category=UserWarning,
            stacklevel=_external_stacklevel(),
        )
    return float(_shuffle_p_values(observed, null_scores, tail))


def _shuffle_p_values(observed: Any, null: Any, tail: str) -> Any:
    """Monte Carlo p-values over the defined null scores, elementwise.

    Parameters
    ----------
    observed : float or NDArray[np.float64], shape (...)
        Observed scores.
    null : NDArray[np.float64], shape (n_shuffles, ...)
        Null scores along the first axis. NaN scores are left out of ``n``;
        infinite scores are counted like any other.
    tail : {"greater", "less", "two-sided"}
        Direction of the test.

    Returns
    -------
    float or NDArray[np.float64], shape (...)
        ``(k + 1) / (n + 1)``; NaN where the observed score is not finite or
        no null score is defined.
    """
    observed = np.asarray(observed, dtype=np.float64)
    null = np.asarray(null, dtype=np.float64)
    n = np.sum(~np.isnan(null), axis=0)
    p = _tail_p_values(
        np.sum(null >= observed, axis=0), np.sum(null <= observed, axis=0), n, tail
    )
    return np.where(np.isfinite(observed) & (n > 0), p, np.nan)


def _tail_p_values(k_greater: Any, k_less: Any, n: Any, alternative: str) -> Any:
    """Monte Carlo p-values ``(k + 1) / (n + 1)`` for one side or both.

    Parameters
    ----------
    k_greater, k_less : int or NDArray[np.integer]
        Null scores at least as large as, and at most as large as, the
        observed score; arrays broadcast elementwise.
    n : int or NDArray[np.integer]
        Null scores counted.
    alternative : {"greater", "less", "two-sided"}
        The side; ``"two-sided"`` doubles the smaller side, capped at 1.

    Returns
    -------
    float or NDArray[np.float64]
        The p-values, shaped like the broadcast counts.
    """
    p_greater = (1.0 + k_greater) / (1.0 + n)
    p_less = (1.0 + k_less) / (1.0 + n)
    if alternative == "greater":
        return p_greater
    if alternative == "less":
        return p_less
    return np.minimum(2.0 * np.minimum(p_greater, p_less), 1.0)


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
        The z-score (standard score). NaN if the defined null scores have
        zero variance, fewer than two are defined, or one is infinite (the
        mean and standard deviation are then undefined).

    Notes
    -----
    Undefined (NaN) null scores are left out of the mean and standard
    deviation, as :func:`compute_shuffle_pvalue` leaves them out of the
    p-value.

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
    null_scores = null_scores[~np.isnan(null_scores)]
    if null_scores.size < 2 or np.isinf(null_scores).any():
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
    """Shuffle labels across trials to test trial identity significance.

    A trial is a run of consecutive samples with the same label. Each
    shuffle permutes the labels among the trials and gives every sample its
    trial's new label, so each trial's samples stay together. Use it to test
    learning effects or trial-type differences: compute your statistic with
    the shuffled labels to build the null.

    Parameters
    ----------
    trial_labels : NDArray[np.int64], shape (n_samples,)
        Label of the trial each sample (e.g., time point) belongs to, in time
        order. Labels can be trial IDs (0, 1, 2, ...) or trial types (e.g.,
        0 for left and 1 for right trials).
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
        Trial labels permuted across trials.

    Raises
    ------
    ValueError
        If ``trial_labels`` is not 1-D or ``n_shuffles`` is not a positive
        integer.

    Notes
    -----
    - Keeps each trial's samples together, so within-trial autocorrelation
      is preserved; permuting single samples would scatter every trial
      across time and make the null anti-conservative.
    - Preserves how many trials carry each label.
    - Trial lengths are not exchanged: a trial keeps its samples and takes
      another trial's label.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.stats.shuffle import shuffle_trials

    >>> trial_labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    >>> for shuffled in shuffle_trials(trial_labels, n_shuffles=2, rng=42):
    ...     print(shuffled)
    [2 2 2 1 1 1 0 0 0]
    [0 0 0 2 2 2 1 1 1]

    See Also
    --------
    shuffle_time_bins : Shuffle temporal order of spike counts.
    neurospatial.replay.shuffle_cell_identity : Shuffle neuron-to-place-field mapping.
    """
    trial_labels = np.asarray(trial_labels)
    if trial_labels.ndim != 1:
        raise ValueError(
            _format_error(
                f"trial_labels must be 1-D, got shape {trial_labels.shape}.",
                why="Why: trials are runs of consecutive samples with one label",
                fix="pass one label per sample, in time order",
            )
        )
    _validate_draw_count(n_shuffles, "n_shuffles")
    # Trial index of each sample: a new trial starts where the label changes.
    trial = np.concatenate([[0], np.cumsum(trial_labels[1:] != trial_labels[:-1])])
    run_labels = trial_labels[np.flatnonzero(np.diff(trial, prepend=-1))]
    return _relabeled_trials(run_labels, trial, n_shuffles, _ensure_rng(rng))


def _relabeled_trials(
    run_labels: NDArray[Any],
    trial: NDArray[np.int64],
    n_shuffles: int,
    generator: np.random.Generator,
) -> Generator[NDArray[Any], None, None]:
    """Yield per-sample labels with the trials' labels permuted."""
    for shuffled in _row_permutations(run_labels, n_shuffles, generator):
        yield shuffled[trial]


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
        Sorted, finite spike times for a single neuron.
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

    Raises
    ------
    ValueError
        If ``spike_times`` is not a sorted, finite 1-D array, or
        ``n_shuffles`` is not a positive integer.

    Warns
    -----
    UserWarning
        With fewer than three spikes there is at most one ISI, so every
        shuffle equals the input.

    Notes
    -----
    - Preserves the first spike time exactly
    - Preserves the ISI distribution (same ISI values, different order)
    - Preserves the last spike time, up to floating-point rounding
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
    spike_times = np.asarray(spike_times, dtype=np.float64)
    if (
        spike_times.ndim != 1
        or not np.all(np.isfinite(spike_times))
        or np.any(np.diff(spike_times) < 0)
    ):
        raise ValueError(
            _format_error(
                f"spike_times must be a sorted, finite 1-D array, got shape "
                f"{spike_times.shape}.",
                why="Why: inter-spike intervals are differences of consecutive "
                "spike times",
                fix="pass np.sort(spike_times[np.isfinite(spike_times)])",
            )
        )
    _validate_draw_count(n_shuffles, "n_shuffles")
    if spike_times.size < 3:
        warnings.warn(
            f"shuffle_spikes_isi: {spike_times.size} spike(s) give at most one "
            "inter-spike interval, so every shuffle equals the input.",
            UserWarning,
            stacklevel=_external_stacklevel(),
        )
    return _isi_permutations(spike_times, n_shuffles, _ensure_rng(rng))


def _isi_permutations(
    spike_times: NDArray[np.float64],
    n_shuffles: int,
    generator: np.random.Generator,
) -> Generator[NDArray[np.float64], None, None]:
    """Yield spike trains rebuilt from permuted inter-spike intervals."""
    if spike_times.size == 0:
        for _ in range(n_shuffles):
            yield spike_times.copy()
        return
    isis = np.diff(spike_times)
    for shuffled in _row_permutations(isis, n_shuffles, generator):
        yield np.concatenate([spike_times[:1], spike_times[0] + np.cumsum(shuffled)])
