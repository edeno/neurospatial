"""Nearest-center queries use the environment-owned KD-tree."""

from unittest.mock import patch

import numpy as np
from scipy.spatial import cKDTree

from neurospatial import Environment


def test_nearest_queries_reuse_one_tree_until_the_cache_is_cleared():
    env = Environment.from_grid_mask(
        np.ones((3, 3), dtype=bool), grid_edges=(np.arange(4.0),) * 2
    )
    points = np.array([[0.5, 0.5], [1.0, 1.0], [100.0, 100.0]])
    with patch("neurospatial.environment._lookup.cKDTree", wraps=cKDTree) as build:
        expected = env.nearest_bin(points, return_dist=True)
        actual = env.nearest_bin(points, return_dist=True)
        for left, right in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(left, right)
        assert build.call_count == 1
        env.clear_cache()
        env.nearest_bin(points)
        assert build.call_count == 2


def _surround() -> Environment:
    return Environment.from_polar_egocentric(
        distance_range=(0, 4),
        angle_range=(-np.pi, np.pi),
        distance_bin_size=1,
        angle_bin_size=np.pi / 2,
    )


def test_polar_nearest_query_measures_physical_distance():
    """Distance is measured between places, not between (cm, rad) columns."""
    env = _surround()
    rng = np.random.default_rng(1)
    points = np.column_stack((rng.uniform(0.0, 4.0, 200), rng.uniform(-9, 9, 200)))
    # Each (distance, angle) row as a complex position in the animal's plane.
    centers = env.bin_centers[:, 0] * np.exp(1j * env.bin_centers[:, 1])
    places = points[:, 0] * np.exp(1j * points[:, 1])
    distances = np.abs(places[:, None] - centers[None])
    expected = np.argmin(distances, axis=1)
    actual, actual_distances = env.nearest_bin(
        points, return_dist=True, max_distance=np.inf
    )
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(
        actual_distances, distances[np.arange(len(points)), expected], atol=1e-12
    )
    # The raw-column metric would pick other bins for some of these points.
    raw = np.argmin(
        np.linalg.norm(env.bin_centers[None] - points[:, None], axis=2), axis=1
    )
    assert np.any(raw != expected)


def test_polar_nearest_query_reads_whole_turns_as_one_direction():
    env = _surround()
    points = np.array([[1.2, 0.2], [2.7, -2.0], [3.5, 3.0]])
    turned = points + np.array([0.0, 4 * np.pi])
    np.testing.assert_array_equal(env.nearest_bin(points), env.nearest_bin(turned))


def test_polar_nearest_query_breaks_ties_at_the_animal_by_lowest_index():
    env = _surround()
    # The animal's own position is equally far from every innermost bin.
    assert env.nearest_bin(np.array([[0.0, 1.0]])).tolist() == [0]


def test_polar_nearest_queries_reuse_one_embedded_tree():
    env = _surround()
    points = np.array([[1.2, 0.2], [2.7, -2.0]])
    with patch("neurospatial.environment._lookup.cKDTree", wraps=cKDTree) as build:
        env.nearest_bin(points)
        env.nearest_bin(points)
        assert build.call_count == 1
