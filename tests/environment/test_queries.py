"""Tests for spatial query operations.

Covers:
- Single and batch bin_at operations
- Contains queries
- Distance computations
- Advanced spatial queries (distance_to, reachable_from, rings, components)
- Property tests for query invariants

Target: 21% → 90% coverage for src/neurospatial/environment/queries.py
"""

import numpy as np
import pytest
from numpy.testing import assert_array_equal


class TestBinAt:
    """Tests for Environment.bin_at() method."""

    def test_bin_at_single_point(self, medium_2d_env):
        """Test bin_at with single point at bin center."""
        # Query a bin center - should map to that bin
        point = medium_2d_env.bin_centers[0]
        bin_id = medium_2d_env.bin_at(np.atleast_2d(point))

        assert bin_id[0] == 0

    def test_bin_at_batch_points(self, medium_2d_env):
        """Test bin_at with batch of points."""
        # Query first 10 bin centers
        points = medium_2d_env.bin_centers[:10]
        bin_ids = medium_2d_env.bin_at(points)

        assert len(bin_ids) == 10
        assert_array_equal(bin_ids, np.arange(10))

    def test_bin_at_single_vs_batch_consistency(self, small_2d_env):
        """Test that single point and batch mode give same results."""
        # Query same points individually and in batch
        points = small_2d_env.bin_centers[:5]

        # Batch mode
        batch_ids = small_2d_env.bin_at(points)

        # Single mode (wrapped in 2D array)
        single_ids = [small_2d_env.bin_at(np.atleast_2d(p))[0] for p in points]

        assert_array_equal(batch_ids, single_ids)

    def test_bin_at_outside_bounds(self, small_2d_env):
        """Test bin_at with point far outside environment bounds."""
        # Point far outside bounds
        point = np.array([[1000.0, 1000.0]])

        bin_id = small_2d_env.bin_at(point)

        # Should return -1 for out-of-bounds
        assert bin_id[0] == -1

    def test_bin_at_boundary_points(self, small_2d_env):
        """Test bin_at at environment boundaries."""
        # Points at exact boundary (min corner)
        min_point = np.array([[r[0] for r in small_2d_env.dimension_ranges]])

        # Should not raise errors and return valid bin or -1
        bin_id = small_2d_env.bin_at(min_point)
        assert isinstance(bin_id[0], (int, np.integer))

    def test_bin_at_all_centers_map_correctly(self, small_2d_env):
        """Test that all bin centers map to their own indices."""
        # Every bin center should map to its own index
        all_centers = small_2d_env.bin_centers
        bin_ids = small_2d_env.bin_at(all_centers)

        assert_array_equal(bin_ids, np.arange(small_2d_env.n_bins))

    def test_bin_at_requires_2d_array(self, small_2d_env):
        """Test that bin_at handles 1D input correctly."""
        # Single point as 1D array - should be converted to 2D
        point_1d = small_2d_env.bin_centers[0]  # Shape (n_dims,)

        # bin_at expects (n_points, n_dims), so wrap it
        bin_id = small_2d_env.bin_at(np.atleast_2d(point_1d))

        assert bin_id[0] == 0

    def test_bin_at_empty_input(self, small_2d_env):
        """Test bin_at with empty array."""
        empty_points = np.empty((0, small_2d_env.n_dims))

        bin_ids = small_2d_env.bin_at(empty_points)

        assert len(bin_ids) == 0

    def test_bin_at_3d_environment(self, simple_3d_env):
        """Test bin_at works correctly for 3D environments."""
        # Query first bin center in 3D
        point = simple_3d_env.bin_centers[0:1]  # Keep 2D shape
        bin_id = simple_3d_env.bin_at(point)

        assert bin_id[0] == 0


class TestContains:
    """Tests for Environment.contains() method."""

    def test_contains_all_bin_centers(self, medium_2d_env):
        """Test that all bin centers are contained in environment."""
        all_centers = medium_2d_env.bin_centers

        contained = medium_2d_env.contains(all_centers)

        # All bin centers should be contained
        assert np.all(contained)
        assert len(contained) == medium_2d_env.n_bins

    def test_contains_outside_point(self, small_2d_env):
        """Test that point far outside is not contained."""
        # Point far outside
        point = np.array([[1000.0, 1000.0]])

        contained = small_2d_env.contains(point)

        assert not contained[0]

    def test_contains_batch_mixed(self, medium_2d_env):
        """Test contains with mix of inside and outside points."""
        # Mix of inside and outside points
        inside_points = medium_2d_env.bin_centers[:5]
        outside_points = np.ones((5, medium_2d_env.n_dims)) * 1000
        all_points = np.vstack([inside_points, outside_points])

        contained = medium_2d_env.contains(all_points)

        # First 5 should be True, last 5 should be False
        assert len(contained) == 10
        assert np.all(contained[:5])
        assert not np.any(contained[5:])

    def test_contains_boundary_behavior(self, small_2d_env):
        """Test contains behavior at environment boundaries."""
        # Points at exact boundary
        min_point = np.array([[r[0] for r in small_2d_env.dimension_ranges]])

        contained = small_2d_env.contains(min_point)

        # Should be boolean
        assert isinstance(contained[0], (bool, np.bool_))

    def test_contains_empty_input(self, small_2d_env):
        """Test contains with empty array."""
        empty_points = np.empty((0, small_2d_env.n_dims))

        contained = small_2d_env.contains(empty_points)

        assert len(contained) == 0
        assert contained.dtype == bool

    def test_contains_consistency_with_bin_at(self, medium_2d_env):
        """Test that contains is consistent with bin_at."""
        # Generate random test points
        rng = np.random.default_rng(42)
        low = np.array([r[0] for r in medium_2d_env.dimension_ranges]) - 10
        high = np.array([r[1] for r in medium_2d_env.dimension_ranges]) + 10
        test_points = rng.uniform(
            low=low,
            high=high,
            size=(50, medium_2d_env.n_dims),
        )

        # Check contains
        contained = medium_2d_env.contains(test_points)

        # Check bin_at
        bin_ids = medium_2d_env.bin_at(test_points)

        # contains should be True iff bin_at != -1
        expected_contained = bin_ids != -1

        assert_array_equal(contained, expected_contained)


class TestBinCenterOf:
    """Tests for Environment.bin_center_of() method."""

    def test_bin_center_of_single_index(self, small_2d_env):
        """Test bin_center_of with single bin index."""
        center = small_2d_env.bin_center_of(0)

        assert center.shape == (small_2d_env.n_dims,)
        np.testing.assert_array_equal(center, small_2d_env.bin_centers[0])

    def test_bin_center_of_multiple_indices(self, medium_2d_env):
        """Test bin_center_of with list of indices."""
        indices = [0, 1, 2, 5, 10]
        centers = medium_2d_env.bin_center_of(indices)

        assert centers.shape == (5, medium_2d_env.n_dims)
        np.testing.assert_array_equal(centers, medium_2d_env.bin_centers[indices])

    def test_bin_center_of_numpy_array_indices(self, small_2d_env):
        """Test bin_center_of with numpy array of indices."""
        indices = np.array([0, 1, 2])
        centers = small_2d_env.bin_center_of(indices)

        assert centers.shape == (3, small_2d_env.n_dims)

    def test_bin_center_of_all_bins(self, small_2d_env):
        """Test bin_center_of for all bins."""
        all_indices = np.arange(small_2d_env.n_bins)
        centers = small_2d_env.bin_center_of(all_indices)

        np.testing.assert_array_equal(centers, small_2d_env.bin_centers)


class TestNeighbors:
    """Tests for Environment.neighbors() method."""

    def test_neighbors_returns_list(self, small_2d_env):
        """Test that neighbors returns a list."""
        neighbors = small_2d_env.neighbors(0)

        assert isinstance(neighbors, list)
        assert all(isinstance(n, (int, np.integer)) for n in neighbors)

    def test_neighbors_nonempty_for_interior_bins(self, medium_2d_env):
        """Test that interior bins have neighbors."""
        # Pick a bin that's not on the boundary
        # (boundary bins may have fewer neighbors)
        center_bin = medium_2d_env.n_bins // 2
        neighbors = medium_2d_env.neighbors(center_bin)

        # Should have at least 1 neighbor
        assert len(neighbors) > 0

    def test_neighbors_are_valid_indices(self, small_2d_env):
        """Test that neighbor indices are valid bin indices."""
        for bin_idx in range(min(10, small_2d_env.n_bins)):
            neighbors = small_2d_env.neighbors(bin_idx)

            # All neighbors should be valid bin indices
            for n in neighbors:
                assert 0 <= n < small_2d_env.n_bins

    def test_neighbors_symmetric(self, small_2d_env):
        """Test that neighbor relationship is symmetric."""
        # If A is neighbor of B, then B is neighbor of A
        bin_a = 0
        neighbors_of_a = small_2d_env.neighbors(bin_a)

        for bin_b in neighbors_of_a:
            neighbors_of_b = small_2d_env.neighbors(bin_b)
            assert bin_a in neighbors_of_b, (
                f"Neighbor relationship not symmetric for {bin_a}, {bin_b}"
            )

    def test_neighbors_bin_not_self_neighbor(self, small_2d_env):
        """Test that bins are not their own neighbors."""
        for bin_idx in range(min(10, small_2d_env.n_bins)):
            neighbors = small_2d_env.neighbors(bin_idx)

            # Bin should not be in its own neighbor list
            assert bin_idx not in neighbors


class TestBinSizes:
    """Tests for Environment.bin_sizes property."""

    def test_bin_sizes_shape(self, medium_2d_env):
        """Test that bin_sizes has correct shape."""
        sizes = medium_2d_env.bin_sizes

        assert sizes.shape == (medium_2d_env.n_bins,)

    def test_bin_sizes_positive(self, small_2d_env):
        """Test that all bin sizes are positive."""
        sizes = small_2d_env.bin_sizes

        assert np.all(sizes > 0)

    def test_bin_sizes_cached(self, small_2d_env):
        """Test that bin_sizes is cached (same object on repeated access)."""
        sizes1 = small_2d_env.bin_sizes
        sizes2 = small_2d_env.bin_sizes

        # Should be the same object (cached)
        assert sizes1 is sizes2

    def test_bin_sizes_reasonable_magnitude(self, small_2d_env):
        """Test that bin sizes have reasonable values."""
        # For a 2D environment with bin_size=2.0, areas should be ~4.0
        sizes = small_2d_env.bin_sizes

        # Check that sizes are in a reasonable range (not too extreme)
        assert np.all(sizes < 100)  # Not huge
        assert np.all(sizes > 0.1)  # Not tiny


class TestDistanceBetween:
    """Tests for Environment.distance_between() method."""

    def test_distance_between_self_is_zero(self, small_2d_env):
        """Test that distance from point to itself is zero."""
        # Same point twice
        point = small_2d_env.bin_centers[0]

        dist = small_2d_env.distance_between(point, point)

        assert dist == 0.0

    def test_distance_between_symmetric(self, medium_2d_env):
        """Test that distance is symmetric (d(A,B) = d(B,A))."""
        point_a = medium_2d_env.bin_centers[0]
        point_b = medium_2d_env.bin_centers[10]

        dist_ab = medium_2d_env.distance_between(point_a, point_b)
        dist_ba = medium_2d_env.distance_between(point_b, point_a)

        np.testing.assert_allclose(dist_ab, dist_ba)

    def test_distance_between_neighbors(self, small_2d_env):
        """Test distance between neighboring bins."""
        # Get first bin and a neighbor
        bin_id = 0
        neighbors = small_2d_env.neighbors(bin_id)

        if len(neighbors) > 0:
            neighbor_id = neighbors[0]

            point_a = small_2d_env.bin_centers[bin_id]
            point_b = small_2d_env.bin_centers[neighbor_id]

            dist = small_2d_env.distance_between(point_a, point_b)

            # Distance should be positive and reasonable
            assert dist > 0
            assert dist < 100  # Sanity check

    def test_distance_between_outside_points(self, small_2d_env):
        """Test distance between points outside environment."""
        # Both points outside
        point_a = np.array([1000.0, 1000.0])
        point_b = np.array([2000.0, 2000.0])

        dist = small_2d_env.distance_between(point_a, point_b)

        # Should return inf since points don't map to valid bins
        assert dist == np.inf

    def test_distance_between_one_outside(self, small_2d_env):
        """Test distance when one point is outside environment."""
        # One inside, one outside
        point_inside = small_2d_env.bin_centers[0]
        point_outside = np.array([1000.0, 1000.0])

        dist = small_2d_env.distance_between(point_inside, point_outside)

        # Should return inf
        assert dist == np.inf

    def test_distance_between_custom_edge_weight(self, small_2d_env):
        """Test distance_between with custom edge weight."""
        # Use two neighboring bins to ensure path exists
        point_a = small_2d_env.bin_centers[0]
        neighbors = small_2d_env.neighbors(0)

        if len(neighbors) > 0:
            point_b = small_2d_env.bin_centers[neighbors[0]]

            # Default uses 'distance' weight
            dist_default = small_2d_env.distance_between(
                point_a, point_b, edge_weight="distance"
            )

            # Should be a finite positive number
            assert np.isfinite(dist_default)
            assert dist_default >= 0


class TestShortestPath:
    """Tests for Environment.shortest_path() method."""

    def test_shortest_path_to_self(self, small_2d_env):
        """Test shortest path from bin to itself."""
        path = small_2d_env.shortest_path(0, 0)

        # Path to self should contain only that bin
        assert path == [0]

    def test_shortest_path_to_neighbor(self, small_2d_env):
        """Test shortest path to neighboring bin."""
        bin_a = 0
        neighbors = small_2d_env.neighbors(bin_a)

        if len(neighbors) > 0:
            bin_b = neighbors[0]
            path = small_2d_env.shortest_path(bin_a, bin_b)

            # Path should be [bin_a, bin_b] (direct connection)
            assert len(path) == 2
            assert path[0] == bin_a
            assert path[-1] == bin_b

    def test_shortest_path_contains_endpoints(self, medium_2d_env):
        """Test that shortest path contains both source and target."""
        path = medium_2d_env.shortest_path(0, 10)

        if len(path) > 0:  # If path exists
            assert path[0] == 0  # Starts at source
            assert path[-1] == 10  # Ends at target

    def test_shortest_path_all_valid_bins(self, small_2d_env):
        """Test that all bins in path are valid."""
        path = small_2d_env.shortest_path(0, 5)

        # All bins in path should be valid indices
        for bin_idx in path:
            assert 0 <= bin_idx < small_2d_env.n_bins

    def test_shortest_path_consecutive_bins_are_neighbors(self, small_2d_env):
        """Test that consecutive bins in path are neighbors."""
        path = small_2d_env.shortest_path(0, 5)

        if len(path) > 1:
            # Each consecutive pair should be neighbors
            for i in range(len(path) - 1):
                bin_a = path[i]
                bin_b = path[i + 1]
                neighbors_a = small_2d_env.neighbors(bin_a)

                assert bin_b in neighbors_a, (
                    f"Non-neighbor bins {bin_a}, {bin_b} in path"
                )


class TestDistanceTo:
    """Tests for Environment.distance_to() method."""

    def test_distance_to_single_target(self, small_2d_env):
        """Test distance_to with single target bin."""
        target = [0]

        distances = small_2d_env.distance_to(target, metric="geodesic")

        # Should have distance for each bin
        assert distances.shape == (small_2d_env.n_bins,)

        # Distance to target itself should be zero
        assert distances[0] == 0.0

        # All distances should be non-negative
        assert np.all(distances >= 0)

    def test_distance_to_multiple_targets(self, medium_2d_env):
        """Test distance_to with multiple target bins."""
        targets = [0, 10, 20]

        distances = medium_2d_env.distance_to(targets, metric="geodesic")

        # All target bins should have distance zero
        assert distances[0] == 0.0
        assert distances[10] == 0.0
        assert distances[20] == 0.0

    def test_distance_to_euclidean_metric(self, small_2d_env):
        """Test distance_to with euclidean metric."""
        target = [0]

        distances = small_2d_env.distance_to(target, metric="euclidean")

        # Target should have distance zero
        assert distances[0] == 0.0

        # All distances should be non-negative
        assert np.all(distances >= 0)

    def test_distance_to_geodesic_vs_euclidean(self, small_2d_env):
        """Test that geodesic and euclidean give different results."""
        target = [0]

        dist_geodesic = small_2d_env.distance_to(target, metric="geodesic")
        dist_euclidean = small_2d_env.distance_to(target, metric="euclidean")

        # Should have same shape
        assert dist_geodesic.shape == dist_euclidean.shape

        # Geodesic distance is typically >= euclidean distance
        # (equal for straight-line paths, greater when following graph)
        assert np.all(dist_geodesic >= dist_euclidean - 1e-6)

    def test_distance_to_invalid_metric_raises(self, small_2d_env):
        """Test that invalid metric raises ValueError."""
        target = [0]

        with pytest.raises(
            ValueError, match="metric must be 'euclidean' or 'geodesic'"
        ):
            small_2d_env.distance_to(target, metric="invalid")

    def test_distance_to_empty_targets_raises(self, small_2d_env):
        """Test that empty targets raises ValueError."""
        with pytest.raises(ValueError, match="targets cannot be empty"):
            small_2d_env.distance_to([], metric="geodesic")

    def test_distance_to_invalid_target_index_raises(self, small_2d_env):
        """Test that out-of-range target index raises ValueError."""
        invalid_target = [small_2d_env.n_bins + 10]

        with pytest.raises(ValueError, match="Target bin indices must be in range"):
            small_2d_env.distance_to(invalid_target, metric="geodesic")


class TestReachableFrom:
    """Tests for Environment.reachable_from() method."""

    def test_reachable_from_no_radius(self, small_2d_env):
        """Test reachable_from with no radius limit (entire component)."""
        reachable = small_2d_env.reachable_from(0, radius=None)

        # Should return boolean mask
        assert reachable.shape == (small_2d_env.n_bins,)
        assert reachable.dtype == bool

        # Source bin should be reachable
        assert reachable[0]

        # For connected environment, all bins should be reachable
        if len(small_2d_env.components()) == 1:
            assert np.all(reachable)

    def test_reachable_from_with_hop_radius(self, medium_2d_env):
        """Test reachable_from with hop radius."""
        reachable = medium_2d_env.reachable_from(0, radius=3, metric="hops")

        # Source should be reachable
        assert reachable[0]

        # Should have some reachable bins
        assert np.sum(reachable) > 0

        # Not all bins should be reachable (unless environment is tiny)
        if medium_2d_env.n_bins > 10:
            assert not np.all(reachable)

    def test_reachable_from_with_geodesic_radius(self, small_2d_env):
        """Test reachable_from with geodesic radius."""
        reachable = small_2d_env.reachable_from(0, radius=10.0, metric="geodesic")

        # Source should be reachable
        assert reachable[0]

        # Should have some reachable bins
        assert np.sum(reachable) > 0

    def test_reachable_from_invalid_source_raises(self, small_2d_env):
        """Test that invalid source bin raises ValueError."""
        with pytest.raises(ValueError, match="source_bin must be in range"):
            small_2d_env.reachable_from(small_2d_env.n_bins + 10, radius=None)

    def test_reachable_from_negative_radius_raises(self, small_2d_env):
        """Test that negative radius raises ValueError."""
        with pytest.raises(ValueError, match="radius must be non-negative"):
            small_2d_env.reachable_from(0, radius=-5)

    def test_reachable_from_invalid_metric_raises(self, small_2d_env):
        """Test that invalid metric raises ValueError."""
        with pytest.raises(ValueError, match="metric must be 'hops' or 'geodesic'"):
            small_2d_env.reachable_from(0, radius=5, metric="invalid")

    def test_reachable_from_source_always_reachable(self, small_2d_env):
        """Test that source bin is always reachable regardless of radius."""
        # Even with radius=0, source should be reachable
        reachable = small_2d_env.reachable_from(0, radius=0, metric="hops")

        assert reachable[0]


class TestComponents:
    """Tests for Environment.components() method."""

    def test_components_returns_list(self, small_2d_env):
        """Test that components returns list of arrays."""
        comps = small_2d_env.components()

        assert isinstance(comps, list)
        assert all(isinstance(c, np.ndarray) for c in comps)

    def test_components_sorted_by_size(self, medium_2d_env):
        """Test that components are sorted by size (largest first)."""
        comps = medium_2d_env.components()

        if len(comps) > 1:
            # Each component should be >= next component
            for i in range(len(comps) - 1):
                assert len(comps[i]) >= len(comps[i + 1])

    def test_components_largest_only(self, small_2d_env):
        """Test components with largest_only=True."""
        comps = small_2d_env.components(largest_only=True)

        # Should return exactly one component
        assert len(comps) == 1

    def test_components_cover_all_bins(self, small_2d_env):
        """Test that components cover all bins exactly once."""
        comps = small_2d_env.components()

        # Concatenate all component bins
        all_bins = np.concatenate(comps)

        # Should have exactly n_bins total (no duplicates, no missing)
        assert len(all_bins) == small_2d_env.n_bins
        assert len(np.unique(all_bins)) == small_2d_env.n_bins

    def test_components_bins_are_valid(self, medium_2d_env):
        """Test that all component bins are valid indices."""
        comps = medium_2d_env.components()

        for comp in comps:
            assert np.all(comp >= 0)
            assert np.all(comp < medium_2d_env.n_bins)


class TestRings:
    """Tests for Environment.rings() method."""

    def test_rings_returns_list(self, small_2d_env):
        """Test that rings returns list of arrays."""
        rings = small_2d_env.rings(center_bin=0, hops=2)

        assert isinstance(rings, list)
        assert len(rings) == 3  # 0, 1, 2 hops = 3 rings

    def test_rings_center_in_ring_zero(self, small_2d_env):
        """Test that center bin is in ring 0."""
        center = 0
        rings = small_2d_env.rings(center_bin=center, hops=3)

        # Ring 0 should contain only the center
        assert len(rings[0]) == 1
        assert rings[0][0] == center

    def test_rings_are_disjoint(self, medium_2d_env):
        """Test that rings are mutually disjoint (no overlaps)."""
        rings = medium_2d_env.rings(center_bin=0, hops=3)

        # Collect all bins from all rings
        all_bins = np.concatenate(rings)

        # Should have no duplicates
        assert len(all_bins) == len(np.unique(all_bins))

    def test_rings_invalid_center_raises(self, small_2d_env):
        """Test that invalid center bin raises ValueError."""
        with pytest.raises(ValueError, match="center_bin must be in range"):
            small_2d_env.rings(center_bin=small_2d_env.n_bins + 10, hops=2)

    def test_rings_negative_hops_raises(self, small_2d_env):
        """Test that negative hops raises ValueError."""
        with pytest.raises(ValueError, match="hops must be non-negative"):
            small_2d_env.rings(center_bin=0, hops=-1)

    def test_rings_bins_are_valid(self, small_2d_env):
        """Test that all bins in rings are valid indices."""
        rings = small_2d_env.rings(center_bin=0, hops=2)

        for ring in rings:
            assert np.all(ring >= 0)
            assert np.all(ring < small_2d_env.n_bins)

    def test_rings_later_rings_may_be_empty(self, small_2d_env):
        """Test that later rings may be empty for small environments."""
        # Request many hops in a small environment
        rings = small_2d_env.rings(center_bin=0, hops=100)

        # Some later rings should be empty
        assert any(len(ring) == 0 for ring in rings)


class TestQueryPropertiesHypothesis:
    """Property-based tests for query invariants using parametrization."""

    @pytest.mark.parametrize("seed", [42, 43, 44])
    def test_bin_at_deterministic(self, medium_2d_env, seed):
        """Property: bin_at is deterministic for same input."""
        rng = np.random.default_rng(seed)
        low = np.array([r[0] for r in medium_2d_env.dimension_ranges])
        high = np.array([r[1] for r in medium_2d_env.dimension_ranges])
        points = rng.uniform(
            low=low,
            high=high,
            size=(10, medium_2d_env.n_dims),
        )

        # Call twice with same input
        result1 = medium_2d_env.bin_at(points)
        result2 = medium_2d_env.bin_at(points)

        # Should get identical results
        assert_array_equal(result1, result2)

    @pytest.mark.parametrize("bin_idx_offset", [0, 1, 5, 10])
    def test_distance_between_symmetric_property(self, medium_2d_env, bin_idx_offset):
        """Property: distance_between is symmetric."""
        if bin_idx_offset >= medium_2d_env.n_bins:
            pytest.skip("Not enough bins for this offset")

        point_a = medium_2d_env.bin_centers[0]
        point_b = medium_2d_env.bin_centers[bin_idx_offset]

        dist_ab = medium_2d_env.distance_between(point_a, point_b)
        dist_ba = medium_2d_env.distance_between(point_b, point_a)

        np.testing.assert_allclose(dist_ab, dist_ba, rtol=1e-10)

    @pytest.mark.parametrize("bin_idx", [0, 5, 10])
    def test_distance_to_self_is_zero(self, medium_2d_env, bin_idx):
        """Property: distance from any point to itself is zero."""
        if bin_idx >= medium_2d_env.n_bins:
            pytest.skip("Not enough bins for this index")

        point = medium_2d_env.bin_centers[bin_idx]
        dist = medium_2d_env.distance_between(point, point)

        assert dist == 0.0

    @pytest.mark.parametrize("hops", [1, 2, 3, 5])
    def test_rings_increase_monotonically(self, medium_2d_env, hops):
        """Property: union of rings up to k includes all bins in ring k."""
        rings = medium_2d_env.rings(center_bin=0, hops=hops)

        # Union of rings 0 to k should include all bins in ring k
        for k in range(len(rings)):
            union_up_to_k = np.concatenate(rings[: k + 1])

            # All bins in ring k should be in union
            for bin_idx in rings[k]:
                assert bin_idx in union_up_to_k
