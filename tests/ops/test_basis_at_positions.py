"""A spatial basis evaluated per sample keeps rows aligned with time."""

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.ops import basis_at_positions, geodesic_rbf_basis


@pytest.fixture(scope="module")
def env_and_basis():
    rng = np.random.default_rng(0)
    env = Environment.from_samples(rng.uniform(0, 50, size=(1000, 2)), bin_size=5.0)
    return env, geodesic_rbf_basis(env, n_centers=8, bandwidth=10.0, rng=0)


def test_rows_are_the_basis_of_each_sample_s_bin(env_and_basis):
    env, basis = env_and_basis
    # One sample at each of three bin centers, then one far outside and one
    # with a missing coordinate.
    positions = np.vstack([env.bin_centers[[0, 5, 9]], [500.0, 500.0], [np.nan, 1.0]])
    X = basis_at_positions(env, basis, positions)
    assert X.shape == (5, basis.shape[0])
    np.testing.assert_array_equal(X[:3], basis[:, [0, 5, 9]].T)
    assert np.isnan(X[3:]).all()


def test_a_basis_from_another_environment_is_refused(env_and_basis):
    env, basis = env_and_basis
    with pytest.raises(ValueError, match="one column per bin"):
        basis_at_positions(env, basis[:, :-1], env.bin_centers[:2])
    with pytest.raises(ValueError, match="one column per bin"):
        basis_at_positions(env, basis.T, env.bin_centers[:2])
