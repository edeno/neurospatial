"""Private rate helper shared by the replay and reactivation packages."""

from __future__ import annotations

import numpy as np


def _binomial_rate(
    n_called: int, n_total: int, confidence: float
) -> tuple[float, float, float]:
    """Rate ``n_called / n_total`` with its exact (Clopper-Pearson) interval.

    Parameters
    ----------
    n_called, n_total : int
        Successes and trials.
    confidence : float
        Confidence level of the interval, in (0, 1).

    Returns
    -------
    rate, low, high : float
        All NaN when ``n_total`` is 0.
    """
    if not n_total:
        return np.nan, np.nan, np.nan
    from scipy.stats import binomtest

    interval = binomtest(n_called, n_total).proportion_ci(
        confidence_level=confidence, method="exact"
    )
    return n_called / n_total, float(interval.low), float(interval.high)
