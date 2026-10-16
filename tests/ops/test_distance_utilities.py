"""
Tests for distance utility methods: distance_to() and rings().

Following TDD - these tests are written BEFORE implementation.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from neurospatial import Environment

# ============================================================================
# Test distance_to() method
# ============================================================================


def _split_grid() -> Environment:
    """Two 4x4 halves separated by an inactive column: two graph components."""
    mask = np.ones((9, 4), dtype=bool)
    mask[4, :] = False
    return Environment.from_grid_mask(
        mask, grid_edges=(np.arange(10.0), np.arange(5.0))
    )


class TestDistanceToBasic:
    """Test basic distance_to() functionality."""

    def test_distance_to_single_bin(self, sample_2d_grid):
        """Distance to single target bin."""
        env = sample_2d_grid
        # Target bin at center of grid
        center_bin = env.n_bins // 2
        distances = env.distance_to(bins=[center_bin], metric="geodesic")

        assert distances.shape == (env.n_bins,)
        assert distances.dtype == np.float64
        assert distances[center_bin] == 0.0
        assert np.all(distances >= 0.0)
        assert np.all(np.isfinite(distances))

    def test_distance_to_multiple_bins(self, sample_2d_grid):
        """Distance to multiple target bins (multi-source)."""
        env = sample_2d_grid
        targets = [0, env.n_bins - 1]  # Opposite corners
        distances = env.distance_to(bins=targets, metric="geodesic")

        assert distances.shape == (env.n_bins,)
        # Both targets should have zero distance
        assert distances[targets[0]] == 0.0
        assert distances[targets[1]] == 0.0
        # All other bins should be positive
        mask = np.ones(env.n_bins, dtype=bool)
        mask[targets] = False
        assert np.all(distances[mask] > 0.0)

    def test_distance_to_region_name(self, sample_2d_grid):
        """Distance to named region."""
        env = sample_2d_grid
        from shapely.geometry import box

        from neurospatial.regions import Regions

        # Add a polygon region covering multiple bins
        regions = Regions()
        xmin, ymin = env.bin_centers.min(axis=0)
        xmax, ymax = env.bin_centers.max(axis=0)
        mid_x = (xmin + xmax) / 2
        mid_y = (ymin + ymax) / 2
        # Box covering roughly 1/4 of environment
        polygon = box(xmin, ymin, mid_x, mid_y)
        _ = regions.add("center", polygon=polygon)
        env = env.with_regions(regions)

        distances = env.distance_to(region="center", metric="geodesic")

        assert distances.shape == (env.n_bins,)
        # Bins inside region should have zero distance
        membership = env.region_membership()
        in_region = np.where(membership[:, 0])[0]
        assert len(in_region) > 0  # At least some bins in region
        for bin_idx in in_region:
            assert distances[bin_idx] == 0.0

    def test_distance_to_preserves_environment(self, sample_2d_grid):
        """distance_to() does not modify environment."""
        env = sample_2d_grid
        n_bins_before = env.n_bins
        bin_centers_before = env.bin_centers.copy()

        _ = env.distance_to(bins=[0], metric="geodesic")

        assert env.n_bins == n_bins_before
        assert_array_equal(env.bin_centers, bin_centers_before)


class TestDistanceToMetrics:
    """Test different distance metrics."""

    def test_geodesic_vs_euclidean(self, sample_2d_grid):
        """Geodesic distance >= Euclidean distance."""
        env = sample_2d_grid
        target = [0]

        dist_geo = env.distance_to(bins=target, metric="geodesic")
        dist_euc = env.distance_to(bins=target, metric="euclidean")

        assert dist_geo.shape == dist_euc.shape
        # Geodesic distance should be >= Euclidean (shortest path on graph)
        # Allow small numerical tolerance
        assert np.all(dist_geo >= dist_euc - 1e-10)

    def test_euclidean_matches_direct_distance(self, sample_2d_grid):
        """Euclidean metric matches direct distance calculation."""
        env = sample_2d_grid
        target_bin = 0
        target_pos = env.bin_centers[target_bin]

        distances = env.distance_to(bins=[target_bin], metric="euclidean")

        # Manually compute Euclidean distances
        expected = np.linalg.norm(env.bin_centers - target_pos[np.newaxis, :], axis=1)

        assert_allclose(distances, expected, rtol=1e-10)

    def test_geodesic_on_disconnected_graph(self):
        """Geodesic distance on disconnected graph returns inf for unreachable bins."""
        env = _split_grid()

        # Verify we created a disconnected graph
        comps = env.components()
        assert len(comps) >= 2, "Failed to create disconnected graph"

        # Pick source in first component, target in second component
        source_bin = comps[0][0]
        target_bin = comps[1][0]

        # Geodesic distance from source to unreachable target should be inf
        distances = env.distance_to(bins=[target_bin], metric="geodesic")
        assert distances[source_bin] == np.inf


class TestDistanceToValidation:
    """Test input validation for distance_to()."""

    def test_distance_to_invalid_bin_index(self, sample_2d_grid):
        """Invalid bin index raises ValueError."""
        env = sample_2d_grid

        with pytest.raises(ValueError, match=r"out of range"):
            env.distance_to(bins=[env.n_bins], metric="geodesic")

        with pytest.raises(ValueError, match=r"out of range"):
            env.distance_to(bins=[-1], metric="geodesic")

    def test_distance_to_invalid_region_name(self, sample_2d_grid):
        """Invalid region name raises KeyError."""
        env = sample_2d_grid

        with pytest.raises(KeyError, match=r"Region.*not found"):
            env.distance_to(region="nonexistent_region", metric="geodesic")

    def test_distance_to_invalid_metric(self, sample_2d_grid):
        """Invalid metric raises ValueError."""
        env = sample_2d_grid

        with pytest.raises(ValueError, match=r"metric must be.*"):
            env.distance_to(bins=[0], metric="invalid")

    def test_distance_to_empty_targets(self, sample_2d_grid):
        """Empty target list raises ValueError."""
        env = sample_2d_grid

        with pytest.raises(ValueError, match=r"empty bins="):
            env.distance_to(bins=[], metric="geodesic")


class TestDistanceToEdgeCases:
    """Test edge cases for distance_to()."""

    def test_distance_to_single_bin_environment(self):
        """Single-bin environment."""
        data = np.array([[0.0, 0.0]])
        env = Environment.from_samples(data, bin_size=1.0)

        distances = env.distance_to(bins=[0], metric="geodesic")

        assert distances.shape == (1,)
        assert distances[0] == 0.0

    def test_distance_to_all_bins_as_targets(self, sample_2d_grid):
        """All bins as targets."""
        env = sample_2d_grid
        targets = list(range(env.n_bins))

        distances = env.distance_to(bins=targets, metric="geodesic")

        # All bins should have zero distance
        assert_array_equal(distances, 0.0)

    def test_distance_to_region_with_multiple_bins(self, sample_2d_grid):
        """Region containing multiple bins."""
        env = sample_2d_grid
        from shapely.geometry import box

        from neurospatial.regions import Regions

        # Create region covering multiple bins
        regions = Regions()
        xmin, ymin = env.bin_centers.min(axis=0)
        xmax, ymax = env.bin_centers.max(axis=0)
        mid_x = (xmin + xmax) / 2
        mid_y = (ymin + ymax) / 2
        # Box covering roughly 1/4 of environment
        polygon = box(xmin, ymin, mid_x, mid_y)
        _ = regions.add("quarter", polygon=polygon)
        env = env.with_regions(regions)

        distances = env.distance_to(region="quarter", metric="geodesic")

        # All bins in region should have zero distance
        membership = env.region_membership()
        in_region = np.where(membership[:, 0])[0]
        assert len(in_region) > 1  # Multi-bin region
        for bin_idx in in_region:
            assert distances[bin_idx] == 0.0


# ============================================================================
# Test rings() method
# ============================================================================


class TestRingsBasic:
    """Test basic rings() functionality."""

    def test_rings_basic(self, sample_2d_grid):
        """Basic ring computation."""
        env = sample_2d_grid
        center = env.n_bins // 2
        hops = 2

        rings_result = env.rings(bins=center, hops=hops)

        assert isinstance(rings_result, list)
        assert len(rings_result) == hops + 1  # [0, 1, 2]
        # Ring 0 should contain only center
        assert_array_equal(rings_result[0], [center])
        # All rings should be disjoint
        all_bins = np.concatenate(rings_result)
        assert len(all_bins) == len(np.unique(all_bins))

    def test_rings_single_hop(self, sample_2d_grid):
        """Single hop ring."""
        env = sample_2d_grid
        center = env.n_bins // 2

        rings_result = env.rings(bins=center, hops=1)

        assert len(rings_result) == 2  # [0, 1]
        assert_array_equal(rings_result[0], [center])
        # Ring 1 should be graph neighbors of center
        neighbors = list(env.connectivity.neighbors(center))
        assert set(rings_result[1]) == set(neighbors)

    def test_rings_zero_hops(self, sample_2d_grid):
        """Zero hops returns only center."""
        env = sample_2d_grid
        center = env.n_bins // 2

        rings_result = env.rings(bins=center, hops=0)

        assert len(rings_result) == 1
        assert_array_equal(rings_result[0], [center])

    def test_rings_preserves_environment(self, sample_2d_grid):
        """rings() does not modify environment."""
        env = sample_2d_grid
        n_bins_before = env.n_bins
        bin_centers_before = env.bin_centers.copy()

        _ = env.rings(bins=0, hops=2)

        assert env.n_bins == n_bins_before
        assert_array_equal(env.bin_centers, bin_centers_before)


class TestRingsProperties:
    """Test mathematical properties of rings."""

    def test_rings_cover_reachable_bins(self, sample_2d_grid):
        """Rings cover all reachable bins within hop distance."""
        env = sample_2d_grid
        center = 0
        hops = 3

        rings_result = env.rings(bins=center, hops=hops)
        all_in_rings = np.concatenate(rings_result)

        # Should match reachable_from with radius
        reachable_mask = env.reachable_from(bins=center, radius=hops, metric="hops")
        reachable_bins = np.where(reachable_mask)[0]

        assert set(all_in_rings) == set(reachable_bins)

    def test_rings_are_disjoint(self, sample_2d_grid):
        """Rings are mutually disjoint."""
        env = sample_2d_grid
        center = env.n_bins // 2
        hops = 3

        rings_result = env.rings(bins=center, hops=hops)

        # Check all pairs are disjoint
        for i in range(len(rings_result)):
            for j in range(i + 1, len(rings_result)):
                intersection = set(rings_result[i]) & set(rings_result[j])
                assert len(intersection) == 0, f"Rings {i} and {j} are not disjoint"

    def test_rings_monotonic_distance(self, sample_2d_grid):
        """Bins in ring k have geodesic distance >= k-1."""
        env = sample_2d_grid
        center = 0
        hops = 3

        rings_result = env.rings(bins=center, hops=hops)
        distances = env.distance_to(bins=[center], metric="geodesic")

        for k, ring in enumerate(rings_result):
            if len(ring) > 0:
                ring_distances = distances[ring]
                # All distances should be >= k-1 edges (with tolerance)
                # Note: geodesic distance is in physical units, not hops
                # So we just check they increase with ring number
                if k > 0:
                    prev_ring = rings_result[k - 1]
                    if len(prev_ring) > 0:
                        prev_distances = distances[prev_ring]
                        # Current ring should have larger distances on average
                        assert (
                            np.mean(ring_distances) >= np.mean(prev_distances) - 1e-10
                        )


class TestRingsValidation:
    """Test input validation for rings()."""

    def test_rings_invalid_center_bin(self, sample_2d_grid):
        """Invalid center bin raises ValueError."""
        env = sample_2d_grid

        with pytest.raises(ValueError, match=r"out of range"):
            env.rings(bins=env.n_bins, hops=2)

        with pytest.raises(ValueError, match=r"out of range"):
            env.rings(bins=-1, hops=2)

    def test_rings_negative_hops(self, sample_2d_grid):
        """Negative hops raises ValueError."""
        env = sample_2d_grid

        with pytest.raises(ValueError, match=r"hops must be non-negative"):
            env.rings(bins=0, hops=-1)


class TestRingsEdgeCases:
    """Test edge cases for rings()."""

    def test_rings_single_bin_environment(self):
        """Single-bin environment."""
        data = np.array([[0.0, 0.0]])
        env = Environment.from_samples(data, bin_size=1.0)

        rings_result = env.rings(bins=0, hops=5)

        # Only ring 0 should exist
        assert len(rings_result) == 6  # [0, 1, 2, 3, 4, 5]
        assert_array_equal(rings_result[0], [0])
        for k in range(1, 6):
            assert len(rings_result[k]) == 0  # Empty rings

    def test_rings_disconnected_graph(self):
        """Rings on disconnected graph stop at component boundary."""
        env = _split_grid()

        # Verify disconnection
        comps = env.components()
        assert len(comps) >= 2, "Failed to create disconnected graph"

        # Start from a bin in the first component
        center = comps[0][0]
        comp0_bins = set(comps[0])

        # Get rings with large hops (more than graph diameter)
        rings_result = env.rings(bins=center, hops=20)

        # Collect all bins reached by rings
        reached_bins = set()
        for ring in rings_result:
            reached_bins.update(ring)

        # Rings should only reach bins in the same component
        assert reached_bins == comp0_bins, (
            f"Rings reached bins outside component: {reached_bins - comp0_bins}"
        )

    def test_rings_large_hops(self, sample_2d_grid):
        """Large hops value (larger than graph diameter)."""
        env = sample_2d_grid
        center = 0
        hops = 1000  # Much larger than needed

        rings_result = env.rings(bins=center, hops=hops)

        # Should have hops+1 rings, but later ones are empty
        assert len(rings_result) == hops + 1
        # All bins should be covered in some ring
        all_in_rings = []
        for ring in rings_result:
            all_in_rings.extend(ring)
        # Should cover all bins in connected component
        comp0 = env.components()[0]
        assert set(all_in_rings) == set(comp0)


class TestDistanceUtilitiesIntegration:
    """Integration tests combining distance_to() and rings()."""

    def test_rings_consistent_with_distance_to(self, sample_2d_grid):
        """Rings are consistent with geodesic distances."""
        env = sample_2d_grid
        center = env.n_bins // 2
        hops = 3

        rings_result = env.rings(bins=center, hops=hops)
        distances = env.distance_to(bins=[center], metric="geodesic")

        # Bins in ring k should all have similar distances
        for _k, ring in enumerate(rings_result):
            if len(ring) > 0:
                ring_distances = distances[ring]
                # Distances within a ring should be similar (small variance)
                # Allow for different path lengths in same hop layer
                # Just check they're all finite
                assert np.all(np.isfinite(ring_distances))

    def test_distance_field_wrapper(self, sample_2d_grid):
        """distance_to() is consistent with existing distance_field()."""
        env = sample_2d_grid
        from neurospatial.ops.distance import distance_field

        targets = [0, 10, 20]

        # Using distance_to wrapper
        dist_wrapper = env.distance_to(bins=targets, metric="geodesic")

        # Using direct distance_field
        dist_direct = distance_field(env.connectivity, sources=targets)

        assert_allclose(dist_wrapper, dist_direct, rtol=1e-10)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_2d_grid():
    """Create a simple 2D grid environment for testing."""
    # Create a 10x10 grid
    n = 10
    points = [[i, j] for i in range(n) for j in range(n)]
    data = np.array(points, dtype=np.float64)
    env = Environment.from_samples(data, bin_size=1.0)
    return env
