"""Coordinate systems preserve physical distances and environment-owned caches."""

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.encoding._smoothing import _get_gaussian_kernel
from neurospatial.environment import CartesianSystem, EgocentricPolarSystem


@pytest.fixture
def cartesian_env() -> Environment:
    """Small Cartesian cells with unit edge lengths."""
    return Environment.from_samples(
        np.array([[0.0, 0.0], [0.0, 2.0], [2.0, 0.0], [2.0, 2.0]]),
        bin_size=1.0,
    )


@pytest.fixture(params=[True, False], ids=["circular", "open"])
def polar_env(request):
    """Identical radial and angular cells with two seam topologies."""
    return Environment.from_polar_egocentric(
        distance_range=(0.0, 30.0),
        angle_range=(-np.pi, np.pi),
        distance_bin_size=5.0,
        angle_bin_size=np.pi / 6,
        circular_angle=request.param,
    )


@pytest.mark.parametrize("n_dims", [1, 2, 3])
def test_cartesian_distances_match_direct_computation(n_dims):
    centers = np.random.default_rng(0).uniform(-10, 10, (7, n_dims))
    expected = ((centers[:, None] - centers[None]) ** 2).sum(axis=-1)
    np.testing.assert_allclose(
        CartesianSystem().squared_distances(centers), expected, rtol=0, atol=1e-9
    )


@pytest.mark.parametrize("circular", [True, False])
def test_polar_distances_wrap_only_circular_seams(circular):
    centers = np.array([[20.0, -np.pi + 0.1], [20.0, np.pi - 0.1]])
    distances = EgocentricPolarSystem(circular=circular).squared_distances(centers)
    delta_angle = 0.2 if circular else 2 * np.pi - 0.2
    assert distances[0, 1] == pytest.approx((20.0 * delta_angle) ** 2)
    np.testing.assert_array_equal(np.diag(distances), np.zeros(2))
    np.testing.assert_allclose(distances, distances.T, rtol=0, atol=1e-9)


@pytest.mark.parametrize("units", [None, "cm"])
def test_axis_names_and_length_ranges(units):
    ranges = ((0.0, 30.0), (-np.pi, np.pi), (-5.0, 5.0))
    cartesian = CartesianSystem()
    polar = EgocentricPolarSystem(circular=True)
    suffix = " (cm)" if units else ""
    assert cartesian.axis_labels(units) == (
        f"X Position{suffix}",
        f"Y Position{suffix}",
    )
    assert polar.axis_labels(units) == (f"Distance{suffix}", "Angle (rad)")
    assert cartesian.length_ranges(ranges) == ranges
    assert polar.length_ranges(ranges) == ranges[:1]
    assert cartesian.equal_aspect
    assert not polar.equal_aspect


def test_cartesian_declaration_and_fallback_lengths(cartesian_env):
    assert isinstance(cartesian_env.coordinate_system, CartesianSystem)
    np.testing.assert_array_equal(
        cartesian_env.coordinate_system.fallback_lengths(cartesian_env),
        np.ones(cartesian_env.n_bins),
    )


def test_polar_declaration_and_fallback_lengths(polar_env):
    system = polar_env.coordinate_system
    assert isinstance(system, EgocentricPolarSystem)
    assert system.circular == polar_env._angular_is_circular
    assert system.circular == polar_env.layout_parameters["circular_angle"]
    np.testing.assert_array_equal(system.fallback_lengths(polar_env), np.full(6, 5.0))


def test_gaussian_cache_reuses_only_its_own_kernel(cartesian_env):
    kernel = _get_gaussian_kernel(cartesian_env, 1.0)
    assert np.shares_memory(_get_gaussian_kernel(cartesian_env, 1.0), kernel)
    copied = cartesian_env.with_metadata()
    other = _get_gaussian_kernel(copied, 1.0)
    assert not np.shares_memory(other, kernel)
    np.testing.assert_array_equal(other, kernel)
    assert not np.shares_memory(_get_gaussian_kernel(cartesian_env, 2.0), kernel)


def test_clear_cache_rebuilds_an_identical_gaussian_kernel(cartesian_env):
    original = _get_gaussian_kernel(cartesian_env, 1.0)
    cartesian_env.clear_cache()
    assert cartesian_env._gaussian_kernels == {}
    rebuilt = _get_gaussian_kernel(cartesian_env, 1.0)
    assert rebuilt is not original
    np.testing.assert_array_equal(rebuilt, original)
