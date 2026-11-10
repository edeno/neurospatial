"""Hold a published test that misses its nominal rate to its measured rate.

Such a test has a strict ``xfail(raises=AssertionError)`` on its
nominal-rate assertion. Everything before that assertion fails with another
exception: a count that leaves its measured rate
(``require_measured_rejections``), a broken result, a crash. The mark accepts
none of them, so they fail the test.
"""

from __future__ import annotations

import numpy as np
import pytest
from calibration_helpers import COVERAGE, null_rejection_interval
from numpy.typing import ArrayLike


def require_measured_rejections(
    rejected: ArrayLike, measured: int, *, coverage: float = COVERAGE
) -> None:
    """Fail the test unless the count of rejections is near ``measured``.

    ``measured`` is the count these null cases gave when the rate was last
    written down. The count must fall inside the central binomial interval
    at the rate ``measured / n``: the spread over other draws of the cases
    if that rate is the true one. A rate that moves either way fails.

    The failure is ``pytest.fail``, not an ``AssertionError``, so a test
    marked ``xfail(raises=AssertionError)`` for its nominal rate fails here
    instead of counting this as its expected failure.
    """
    rejected = np.asarray(rejected, dtype=bool)
    n = rejected.size
    count = int(rejected.sum())
    low, high = null_rejection_interval(n, measured / n, coverage)
    if not low <= count <= high:
        pytest.fail(
            f"{count} of {n} null cases rejected; {low} to {high} expected "
            f"around the measured {measured}. If the change is intended, "
            "update the measured count here and in the recipe's notes."
        )
