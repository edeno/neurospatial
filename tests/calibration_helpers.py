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
    """Assert that null p-values are uniform on [0, 1] (Kolmogorov-Smirnov).

    Monte Carlo p-values ``(k + 1) / (n + 1)`` are discrete; with hundreds
    of shuffles the step is too small for the KS test to see. ``level`` is
    the KS p-value below which uniformity is rejected.
    """
    from scipy import stats

    p_values = np.asarray(p_values, dtype=float)
    assert np.all(np.isfinite(p_values)), "null p-values must be finite"
    result = stats.kstest(p_values, "uniform")
    assert result.pvalue >= level, (
        f"{p_values.size} null p-values are not uniform: KS D = "
        f"{result.statistic:.3f}, p = {result.pvalue:.2g}"
    )
