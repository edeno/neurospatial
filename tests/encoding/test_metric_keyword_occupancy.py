"""Rate-map metrics take ``occupancy`` by keyword.

A rate map and its occupancy have the same shape and are both non-negative,
so a swapped positional pair would return a plausible wrong number.
"""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial.encoding import (
    information_per_second,
    selectivity,
    sparsity,
    spatial_information,
)


@pytest.fixture(scope="module")
def rate_map():
    """A place field and an uneven occupancy that sums to 1."""
    rng = np.random.default_rng(3)
    x = np.linspace(0.0, 1.0, 50)
    rate = 0.5 + 12.0 * np.exp(-((x - 0.3) ** 2) / 0.005)
    occupancy = rng.uniform(0.5, 1.5, x.size)
    return rate, occupancy / occupancy.sum()


def _skaggs(rate, p):
    mean = np.sum(p * rate)
    return float(np.sum(p * rate / mean * np.log2(rate / mean)))


EXPECTED = {
    spatial_information: _skaggs,
    sparsity: lambda r, p: float(np.sum(p * r) ** 2 / np.sum(p * r**2)),
    selectivity: lambda r, p: float(r.max() / np.sum(p * r)),
    information_per_second: lambda r, p: _skaggs(r, p) * float(np.sum(p * r)),
}


@pytest.mark.parametrize("metric", list(EXPECTED), ids=lambda f: f.__name__)
def test_metric_occupancy_is_keyword_only(metric, rate_map):
    rate, occupancy = rate_map
    with pytest.raises(TypeError, match="positional argument"):
        metric(rate, occupancy)
    np.testing.assert_allclose(
        metric(rate, occupancy=occupancy), EXPECTED[metric](rate, occupancy), rtol=1e-12
    )
