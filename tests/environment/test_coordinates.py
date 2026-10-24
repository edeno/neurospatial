"""Coordinate systems preserve physical distances and environment-owned caches."""

import pickle

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.encoding._smoothing import _get_gaussian_kernel
from neurospatial.environment import CartesianSystem, EgocentricPolarSystem
from neurospatial.environment._cache import GAUSSIAN_KERNELS


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


def _embedded_squared_distances(system, centers):
    """Squared Euclidean distances between the places in the animal's plane."""
    places = system.cartesian_embedding(centers)
    return ((places[:, None] - places[None]) ** 2).sum(axis=-1)


@pytest.mark.parametrize("circular", [True, False])
def test_polar_distances_are_physical_between_places(circular):
    """Bins are places in the animal's plane: distance is the straight line."""
    system = EgocentricPolarSystem(circular=circular)
    rng = np.random.default_rng(1)
    # Angles within pi of each other, so no straight line crosses the seam.
    centers = np.c_[rng.uniform(0, 40, 9), rng.uniform(-np.pi / 2, np.pi / 2, 9)]
    np.testing.assert_allclose(
        system.squared_distances(centers),
        _embedded_squared_distances(system, centers),
        rtol=1e-12,
        atol=1e-9,
    )
    # Two places 2.5 cm away, nearly opposite: the chord, not the arc.
    pair = np.array([[2.5, 0.0], [2.5, np.pi - 0.2]])
    assert np.sqrt(system.squared_distances(pair)[0, 1]) == pytest.approx(
        5.0 * np.sin((np.pi - 0.2) / 2)
    )


@pytest.mark.parametrize("circular", [True, False])
def test_polar_distances_cross_only_circular_seams(circular):
    """A circular seam is open space; an open seam is a cut to go around."""
    system = EgocentricPolarSystem(circular=circular)
    centers = np.array([[20.0, -np.pi + 0.1], [10.0, np.pi - 0.1]])
    distances = system.squared_distances(centers)
    if circular:
        np.testing.assert_allclose(
            distances, _embedded_squared_distances(system, centers), atol=1e-9
        )
    else:
        # Around the cut's end at the animal: 20 cm in, then 10 cm out.
        assert distances[0, 1] == pytest.approx(30.0**2)
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
    assert system.circular == polar_env.arena.circular
    assert system.circular == polar_env.layout_parameters["circular_angle"]
    np.testing.assert_array_equal(system.fallback_lengths(polar_env), np.full(6, 5.0))


def test_gaussian_cache_is_shared_only_with_metadata_copies(cartesian_env):
    kernel = _get_gaussian_kernel(cartesian_env, 1.0)
    assert np.shares_memory(_get_gaussian_kernel(cartesian_env, 1.0), kernel)
    renamed = cartesian_env.with_metadata(name="renamed")
    assert np.shares_memory(_get_gaussian_kernel(renamed, 1.0), kernel)
    restored = pickle.loads(pickle.dumps(cartesian_env))
    other = _get_gaussian_kernel(restored, 1.0)
    assert not np.shares_memory(other, kernel)
    np.testing.assert_array_equal(other, kernel)
    assert not np.shares_memory(_get_gaussian_kernel(cartesian_env, 2.0), kernel)


def test_clear_cache_rebuilds_an_identical_gaussian_kernel(cartesian_env):
    original = _get_gaussian_kernel(cartesian_env, 1.0)
    cartesian_env.clear_cache()
    assert GAUSSIAN_KERNELS not in cartesian_env._cache
    rebuilt = _get_gaussian_kernel(cartesian_env, 1.0)
    assert rebuilt is not original
    np.testing.assert_array_equal(rebuilt, original)


def test_descriptors_say_their_kind_and_which_axes_wrap(cartesian_env, polar_env):
    cartesian = cartesian_env.coordinate_system
    assert (cartesian.kind, cartesian.circular, cartesian.periodic_axes) == (
        "cartesian",
        False,
        (),
    )
    polar = polar_env.coordinate_system
    assert polar.kind == polar_env.coordinate_kind == "polar"
    assert polar.circular == polar_env.layout_parameters["circular_angle"]
    assert polar.periodic_axes == ((1,) if polar.circular else ())
