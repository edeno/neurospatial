"""Encoding entry points describe positions with the wrong column count as Environment does."""

import numpy as np
import pytest

from neurospatial import Environment, compute_spatial_rate
from neurospatial.encoding._spatial_counts import compute_occupancy

CLAUSE = (
    "positions with shape (100, 3), but this environment's bins are 2-D, "
    "so positions needs shape (n_positions, 2)"
)


@pytest.fixture(scope="module")
def env():
    rng = np.random.default_rng(0)
    return Environment.from_samples(rng.random((500, 2)) * 50, bin_size=5.0)


@pytest.fixture(scope="module")
def trajectory():
    times = np.linspace(0.0, 10.0, 100)
    positions = np.random.default_rng(1).random((100, 3)) * 50
    return times, positions


def test_spatial_rate_reports_the_shared_clause(env, trajectory):
    times, positions = trajectory
    with pytest.raises(ValueError, match=r"positions with shape \(100, 3\)") as error:
        compute_spatial_rate(env, np.array([1.0, 2.0]), times, positions)
    assert CLAUSE in str(error.value)


def test_occupancy_reports_the_shared_clause(env, trajectory):
    times, positions = trajectory
    with pytest.raises(ValueError) as error:
        compute_occupancy(env, times, positions)
    assert CLAUSE in str(error.value)
    with pytest.raises(ValueError, match="positions with shape"):
        env.occupancy(times, positions)
