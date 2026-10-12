"""Analytic regression cases for information and sparsity metrics."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from neurospatial.encoding import (
    information_per_second,
    mutual_information,
    sparsity,
    spatial_information,
)
from neurospatial.encoding._metrics import batch_sparsity, batch_spatial_information


@pytest.fixture(params=["numpy", "jax"])
def xp(request):
    if request.param == "numpy":
        return np
    from neurospatial.encoding._backend import is_jax_available

    if not is_jax_available():
        pytest.skip("JAX is unavailable")
    import jax.numpy as jnp

    return jnp


@pytest.mark.parametrize("missing", [np.nan, np.inf])
def test_missing_rates_do_not_create_selectivity(xp, missing):
    rates = xp.asarray([2.0, 2.0, missing])
    occupancy = xp.asarray([1.0, 1.0, 10.0])
    assert_allclose(spatial_information(rates, occupancy), 0.0, atol=1e-7)
    assert_allclose(sparsity(rates, occupancy), 1.0, atol=1e-7)


def test_population_metrics_normalize_each_neurons_observed_support(xp):
    rates = xp.asarray([[2.0, 2.0, np.nan, 2.0], [np.nan, 0.0, 4.0, 8.0], [np.nan] * 4])
    occupancy = xp.asarray([1.0, 2.0, 3.0, np.nan])
    # The second neuron fires only in a bin carrying 3/5 of its valid occupancy.
    expected_info = [0.0, np.log2(5 / 3), 0.0]
    expected_sparsity = [1.0, 3 / 5, 0.0]
    assert_allclose(
        batch_spatial_information(rates, occupancy), expected_info, atol=1e-6
    )
    assert_allclose(batch_sparsity(rates, occupancy), expected_sparsity, atol=1e-6)
    for row, info, sparse in zip(rates, expected_info, expected_sparsity, strict=True):
        assert_allclose(spatial_information(row, occupancy), info, atol=1e-6)
        assert_allclose(sparsity(row, occupancy), sparse, atol=1e-6)


def test_zero_activity_remains_observed(xp):
    rates = xp.asarray([0.0, 4.0, np.nan])
    occupancy = xp.asarray([1.0, 1.0, 100.0])
    assert_allclose(spatial_information(rates, occupancy), 1.0, atol=1e-7)
    assert_allclose(sparsity(rates, occupancy), 0.5, atol=1e-7)


def test_information_rate_uses_same_observed_support_as_spatial_information():
    rates = np.array([1.0, np.nan, 3.0])
    occupancy = np.array([1.0, 2.0, 3.0])
    expected = 0.25 * np.log2(1 / 2.5) + 0.75 * 3 * np.log2(3 / 2.5)
    assert_allclose(information_per_second(rates, occupancy), expected)


@pytest.mark.parametrize("base", [2.0, np.e, 10.0])
def test_mutual_information_perfect_response_equals_position_entropy(base):
    counts = np.diag([1.0, 3.0, 0.0])
    expected = -(0.25 * np.log(0.25) + 0.75 * np.log(0.75)) / np.log(base)
    assert_allclose(mutual_information(counts, base=base), expected, atol=1e-12)


def test_mutual_information_independent_response_is_zero():
    counts = np.outer([1, 2, 0, 5], [3, 0, 7])
    assert_allclose(mutual_information(counts), 0.0, atol=1e-12)


def test_mutual_information_noisy_binary_channel():
    error = 0.1
    joint = np.array([[1 - error, error], [error, 1 - error]]) / 2
    binary_entropy = -error * np.log2(error) - (1 - error) * np.log2(1 - error)
    assert_allclose(mutual_information(joint), 1 - binary_entropy, atol=1e-12)


def test_mutual_information_symmetric_and_bounded_by_both_entropies():
    joint = np.array([[3.0, 0.0, 1.0], [1.0, 2.0, 4.0]])
    probability = joint / joint.sum()
    entropies = [
        -np.sum(p * np.log2(p))
        for p in (probability.sum(axis=0), probability.sum(axis=1))
    ]
    actual = mutual_information(joint)
    assert 0 <= actual <= min(entropies)
    assert_allclose(actual, mutual_information(joint.T))
    assert_allclose(actual, mutual_information(joint * 100))


@pytest.mark.parametrize("scale", [1e-300, 1e300])
def test_mutual_information_normalizes_extreme_finite_counts(scale):
    assert_allclose(mutual_information(np.eye(4) * scale), 2.0)


@pytest.mark.parametrize("base", [0, 0.5, 1, -1, np.inf, np.nan])
def test_mutual_information_invalid_base(base):
    with pytest.raises(ValueError, match="base"):
        mutual_information(np.eye(2), base=base)


@pytest.mark.parametrize(
    "counts",
    [np.zeros((2, 2)), [[-1, 2]], [[np.nan, 1]], [[np.inf, 1]], np.empty((2, 0))],
)
def test_mutual_information_invalid_histogram(counts):
    with pytest.raises(ValueError):
        mutual_information(counts)


def test_mutual_information_rejects_rate_map_with_migration_guidance():
    with pytest.raises(ValueError, match="information_per_second"):
        mutual_information([0.0, 100.0], [0.5, 0.5])
    with pytest.raises(ValueError, match="joint position-response histogram"):
        mutual_information([0.0, 100.0])
