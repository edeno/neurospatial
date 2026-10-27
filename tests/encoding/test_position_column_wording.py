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


@pytest.mark.parametrize(
    "significance",
    [
        "object_vector_cell_significance",
        "egocentric_object_vector_cell_significance",
        "spatial_view_cell_significance",
    ],
)
def test_significance_reports_the_shared_clause(env, trajectory, significance):
    import neurospatial.encoding as encoding

    times, positions = trajectory
    headings = np.zeros(len(times))
    objects = np.array([[25.0, 25.0]])
    args = {
        "object_vector_cell_significance": (positions, objects),
        "egocentric_object_vector_cell_significance": (positions, headings, objects),
        "spatial_view_cell_significance": (positions, headings),
    }[significance]
    with pytest.raises(ValueError) as error:
        getattr(encoding, significance)(
            env, [np.array([1.0, 2.0])], times, *args, n_shuffles=2, rng=0
        )
    assert CLAUSE in str(error.value)


@pytest.mark.parametrize(
    "function",
    [
        "compute_object_vector_rates",
        "object_vector_cell_significance",
    ],
)
def test_object_vectors_without_env_need_planar_positions(trajectory, function):
    import neurospatial.encoding as encoding

    times, positions = trajectory
    kwargs = {"n_shuffles": 2, "rng": 0} if "significance" in function else {}
    with pytest.raises(ValueError, match=r"positions with shape \(100, 3\)") as error:
        getattr(encoding, function)(
            None,
            [np.array([1.0, 2.0])],
            times,
            positions,
            np.array([[25.0, 25.0]]),
            bandwidth=5.0,
            distance_range=(0.0, 50.0),
            distance_bin_size=5.0,
            **kwargs,
        )
    message = str(error.value)
    assert "(n_positions, 2)" in message and "Fix:" in message
    assert "environment's bins" not in message
