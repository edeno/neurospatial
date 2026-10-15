"""Nearest-center queries use environment-owned caches and public controls."""

from unittest.mock import patch

import numpy as np
from scipy.spatial import cKDTree

from neurospatial import Environment
from neurospatial.ops.binning import clear_kdtree_cache, map_points_to_bins


def test_nearest_queries_reuse_one_tree_and_clear_only_that_cache():
    env = Environment.from_grid_mask(
        np.ones((3, 3), dtype=bool), grid_edges=(np.arange(4.0),) * 2
    )
    points = np.array([[0.5, 0.5], [1.0, 1.0], [100.0, 100.0]])
    env.compute_kernel(0.5)
    with patch("scipy.spatial.cKDTree", wraps=cKDTree) as build:
        expected = env.nearest_bin(points, return_dist=True)
        actual = map_points_to_bins(env, points, return_dist=True)
        for left, right in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(left, right)
        assert build.call_count == 1
        clear_kdtree_cache(env)
        env.nearest_bin(points)
        assert build.call_count == 2
    from neurospatial.ops import diffusion

    with patch.object(
        diffusion,
        "diffusion_kernel",
        side_effect=AssertionError("Kernel cache was cleared"),
    ):
        env.compute_kernel(0.5)


def test_public_nearest_query_keeps_raw_polar_coordinate_semantics():
    env = Environment.from_polar_egocentric((0, 4), (-np.pi, np.pi), 1, np.pi / 2)
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

        def clear_cache(self, **options):
            assert options == {
                "kdtree": True,
                "kernels": False,
                "cached_properties": False,
            }

    queries = Queries()
    np.testing.assert_array_equal(
        map_points_to_bins(queries, np.ones((2, 2)), max_distance=2.0), [0, 0]
    )
    clear_kdtree_cache(queries)
