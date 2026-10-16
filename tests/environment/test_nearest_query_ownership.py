"""Nearest-center queries use the environment-owned KD-tree."""

from unittest.mock import patch

import numpy as np
from scipy.spatial import cKDTree

from neurospatial import Environment
from neurospatial.ops.binning import map_points_to_bins


def test_nearest_queries_reuse_one_tree_until_the_cache_is_cleared():
    env = Environment.from_grid_mask(
        np.ones((3, 3), dtype=bool), grid_edges=(np.arange(4.0),) * 2
    )
    points = np.array([[0.5, 0.5], [1.0, 1.0], [100.0, 100.0]])
    with patch("neurospatial.environment._lookup.cKDTree", wraps=cKDTree) as build:
        expected = env.nearest_bin(points, return_dist=True)
        actual = map_points_to_bins(env, points, return_dist=True)
        for left, right in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(left, right)
        assert build.call_count == 1
        env.clear_cache()
        env.nearest_bin(points)
        assert build.call_count == 2


def test_public_nearest_query_keeps_raw_polar_coordinate_semantics():
    env = Environment.from_polar_egocentric(
        distance_range=(0, 4),
        angle_range=(-np.pi, np.pi),
        distance_bin_size=1,
        angle_bin_size=np.pi / 2,
    )
    points = np.array([[1.2, 0.2], [2.7, -2.0]])
    distances = np.linalg.norm(env.bin_centers[None] - points[:, None], axis=2)
    expected = np.argmin(distances, axis=1)
    actual, actual_distances = env.nearest_bin(
        points, return_dist=True, max_distance=np.inf
    )
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(
        actual_distances, distances[np.arange(len(points)), expected]
    )


def test_compatibility_wrappers_need_no_private_cache_attributes():
    class Queries:
        def nearest_bin(self, points, **options):
            assert options["max_distance"] == 2.0
            return np.zeros(len(points), dtype=np.int64)

    queries = Queries()
    np.testing.assert_array_equal(
        map_points_to_bins(queries, np.ones((2, 2)), max_distance=2.0), [0, 0]
    )
