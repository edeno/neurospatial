"""Shared tolerance rule for tests of a null's false-positive rate.

A test that rejects each null case with probability ``rate`` rejects
``Binomial(n, rate)`` of ``n`` cases. Calibration tests assert that the count
falls inside the central binomial interval, from both ends, so a test that
rejects too often and one that never rejects both fail. They also check that
null p-values are uniform where the test claims exact p-values.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

COVERAGE = 0.99
"""Probability that an exact test's count falls inside the asserted interval."""


def null_rejection_interval(
    n_events: int, rate: float, coverage: float = COVERAGE
) -> tuple[int, int]:
    """Central binomial interval for how many of ``n_events`` null cases reject.

    The count falls outside the returned ``(low, high)`` with probability at
    most ``1 - coverage`` when each case rejects with probability ``rate``.
    """
    from scipy import stats

    tail = (1.0 - coverage) / 2.0
    return (
        int(stats.binom.ppf(tail, n_events, rate)),
        int(stats.binom.ppf(1.0 - tail, n_events, rate)),
    )


def monte_carlo_rejection_rate(
    n_shuffles: int, alpha: float, strict: bool = True
) -> float:
    """Exact rejection rate of ``p = (k + 1) / (n_shuffles + 1)`` under the null.

    With exchangeable shuffles ``k`` is uniform on ``0..n_shuffles``, so the
    rate is the share of those values whose p-value is below (``strict``) or
    at most ``alpha``: 0.045, not 0.05, for ``p < 0.05`` with 199 shuffles.
    """
    p_values = np.arange(1, n_shuffles + 2) / (n_shuffles + 1)
    rejected = p_values < alpha if strict else p_values <= alpha + 1e-12
    return float(np.mean(rejected))


def assert_rejections_at_rate(
    rejected: ArrayLike,
    rate: float,
    *,
    coverage: float = COVERAGE,
    conservative: bool = False,
) -> None:
    """Assert that the share of null cases rejected matches ``rate``.

    ``rejected`` holds one bool per null case. A ``conservative`` test, one
    documented to reject at most ``rate``, is held to the high end only.
    """
    rejected = np.asarray(rejected, dtype=bool)
    n = rejected.size
    count = int(rejected.sum())
    low, high = null_rejection_interval(n, rate, coverage)
    message = f"{count} of {n} null cases rejected"
    assert count <= high, f"{message}; at most {high} expected at rate {rate:g}"
    if not conservative:
        assert count >= low, f"{message}; at least {low} expected at rate {rate:g}"


def assert_pvalues_uniform(p_values: ArrayLike, *, level: float = 0.001) -> None:
    """Assert that exact null p-values are uniform, on [0, 1] or on a grid.

    Valid for p-values with ``P(p <= v) = v`` at every value ``v`` they can
    take: continuous p-values, and Monte Carlo p-values
    ``(k + 1) / (n_shuffles + 1)`` of an exchangeable null. For both, a false
    failure has probability at most ``level``.

    The statistic is the largest ``|share of p-values <= v - v|`` over the
    observed values ``v``, referred to the Kolmogorov-Smirnov null
    distribution. The Kolmogorov-Smirnov distance also compares just below
    each observed value, where the distribution function of p-values on a
    grid is a whole step under the diagonal: ``scipy.stats.kstest`` fails
    exact p-values from 39 shuffles in 4.6% of samples of 1000 at ``level``
    0.001. Rounding a continuous uniform sample up to the grid leaves the
    share at or below each grid value unchanged, so without that comparison
    the grid cannot add false failures.

    The comparison just below a value exceeds the one at the value by the
    share of the sample at that value, so a departure of that size can go
    unseen. p-values that pass but are too coarse for the sample are
    therefore refused: the most common value must hold a smaller share of
    the sample than the distance rejected at ``level``. A p-value that is
    always 1 is refused, and a sample at twice the Kolmogorov-Smirnov
    rejection distance always fails.

    Raises
    ------
    AssertionError
        If the p-values are not finite or not uniform.
    ValueError
        If the p-values are too coarse for the check.
    """
    from scipy import stats

    p_values = np.asarray(p_values, dtype=float)
    assert np.all(np.isfinite(p_values)), "null p-values must be finite"
    n = p_values.size
    values, counts = np.unique(p_values, return_counts=True)
    distance = float(np.max(np.abs(np.cumsum(counts) / n - values)))
    p_uniform = float(stats.kstwo.sf(distance, n))
    assert p_uniform >= level, (
        f"{n} null p-values are not uniform: largest distance of their "
        f"distribution function from the diagonal = {distance:.3f}, "
        f"p = {p_uniform:.2g}"
    )
    largest_share = float(counts.max() / n)
    rejected_distance = float(stats.kstwo.isf(level, n))
    if largest_share >= rejected_distance:
        msg = (
            f"{n} null p-values are too coarse to check for uniformity: "
            f"{counts.max()} of them ({largest_share:.3g} of the sample) equal "
            f"{values[counts.argmax()]:.3g}, and the check rejects at a "
            f"distance of {rejected_distance:.3g}.\n"
            "Why: a departure from uniform as large as the share of the "
            "sample at one value can go unseen, so that share must be smaller "
            "than the distance the check rejects at.\n"
            f"Fix: use at least {int(np.ceil(2 / rejected_distance))} shuffles "
            "per p-value, which puts half that distance at each value, or "
            "check the rejection count at chosen levels with "
            "assert_rejections_at_rate."
        )
        raise ValueError(msg)
