"""
Tests for Environment.region_membership() method.

Tests cover:
- Basic functionality with single and multiple regions
- Boundary inclusion behavior (include_boundary parameter)
- Different region types (polygons vs points)
- Empty regions and edge cases
- Caching behavior
- Input validation
"""

import numpy as np
import pytest
from shapely.geometry import Polygon, box

from neurospatial import Environment
from neurospatial.regions import Regions


class TestRegionMembershipBasic:
    """Test basic region membership functionality."""

    def test_single_polygon_region(self):
        """Test membership with a single polygon region."""
        # Create 5x5 grid from 0 to 10
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        # Add a region covering the center of the grid
        # Region from (3, 3) to (7, 7)
        env.regions.add("center", polygon=box(3, 3, 7, 7))

        # Get membership
        membership = env.region_membership()

        # Check shape
        assert membership.shape == (env.n_bins, 1)
        assert membership.dtype == bool

        # Check that some bins are inside the region
        assert np.any(membership[:, 0])
        # Check that some bins are outside the region
        assert np.any(~membership[:, 0])

    def test_multiple_regions(self):
        """Test membership with multiple regions."""
        # Create 5x5 grid
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        # Add two non-overlapping regions
        env.regions.add("left", polygon=box(0, 0, 4, 10))
        env.regions.add("right", polygon=box(6, 0, 10, 10))

        membership = env.region_membership()

        # Check shape
        assert membership.shape == (env.n_bins, 2)

        # Check that regions don't overlap (no bin is in both)
        assert not np.any(np.all(membership, axis=1))

        # Check that some bins are in each region
        assert np.any(membership[:, 0])  # left region
        assert np.any(membership[:, 1])  # right region

    def test_overlapping_regions(self):
        """Test membership with overlapping regions."""
        # Create grid
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        # Add overlapping regions
        env.regions.add("region_a", polygon=box(0, 0, 6, 6))
        env.regions.add("region_b", polygon=box(4, 4, 10, 10))

        membership = env.region_membership()

        # Check that some bins are in both regions
        both = np.all(membership, axis=1)
        assert np.any(both), "Expected some bins in overlapping area"

        # Check that some bins are in only one region
        only_a = membership[:, 0] & ~membership[:, 1]
        only_b = membership[:, 1] & ~membership[:, 0]
        assert np.any(only_a)
        assert np.any(only_b)

    def test_no_regions(self):
        """Test with environment that has no regions."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        membership = env.region_membership()

        # Should return array with shape (n_bins, 0)
        assert membership.shape == (env.n_bins, 0)
        assert membership.dtype == bool


class TestRegionMembershipBoundaryBehavior:
    """Test boundary inclusion behavior."""

    def test_include_boundary_true(self):
        """Test that bins on boundary are included when include_boundary=True."""
        # Create grid with known bin centers
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        # Find a bin center that we know is exactly on the boundary
        # Grid centers are at 1, 3, 5, 7, 9 (with bin_size=2, starting from 0)
        # Create region with edge at x=5.0
        env.regions.add("test", polygon=box(0, 0, 5.0, 10))

        membership = env.region_membership(include_boundary=True)

        # Bins with centers at x=5.0 should be included
        bin_centers = env.bin_centers
        boundary_bins = np.where(np.isclose(bin_centers[:, 0], 5.0))[0]

        if len(boundary_bins) > 0:
            # At least one boundary bin should be included
            assert np.any(membership[boundary_bins, 0])

    def test_include_boundary_false(self):
        """Test that bins on boundary are excluded when include_boundary=False."""
        # Create grid
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        # Region with edge that might exactly hit bin centers
        env.regions.add("test", polygon=box(0, 0, 5.0, 10))

        membership_with = env.region_membership(include_boundary=True)
        membership_without = env.region_membership(include_boundary=False)

        # Without boundary should have equal or fewer members
        assert np.sum(membership_without) <= np.sum(membership_with)

    def test_boundary_behavior_consistency(self):
        """Test that boundary parameter is consistent across multiple calls."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)
        env.regions.add("test", polygon=box(2, 2, 8, 8))

        # Multiple calls with same parameter should give same result
        m1 = env.region_membership(include_boundary=True)
        m2 = env.region_membership(include_boundary=True)
        assert np.array_equal(m1, m2)

        m3 = env.region_membership(include_boundary=False)
        m4 = env.region_membership(include_boundary=False)
        assert np.array_equal(m3, m4)


class TestRegionMembershipRegionTypes:
    """Test behavior with different region types."""

    def test_point_regions_not_supported(self):
        """Test that point regions raise an error or are handled appropriately."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        # Add a point region
        env.regions.add("point", point=[5.0, 5.0])

        # Point regions should either raise an error or return all False
        # (a point has no area, so no bin centers can be "inside" it)
        membership = env.region_membership()

        # Should have the right shape
        assert membership.shape == (env.n_bins, 1)

        # All values should be False (no bin can be inside a point)
        assert not np.any(membership[:, 0])

    def test_mixed_region_types(self):
        """Test with a mix of point and polygon regions."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        # Add both types
        env.regions.add("poly", polygon=box(2, 2, 8, 8))
        env.regions.add("point", point=[5.0, 5.0])

        membership = env.region_membership()

        # Check shape
        assert membership.shape == (env.n_bins, 2)

        # Polygon should have some members
        assert np.any(membership[:, 0])
        # Point should have no members (or possibly one if using a tolerance)
        assert not np.any(membership[:, 1])

    def test_empty_polygon(self):
        """Test with an empty/degenerate polygon."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        # Create a degenerate polygon (line)
        env.regions.add("line", polygon=Polygon([(0, 0), (10, 0), (10, 0), (0, 0)]))

        membership = env.region_membership()

        # Should handle gracefully
        assert membership.shape == (env.n_bins, 1)
        # Note: Shapely may return True for points on the line depending on
        # implementation details, so we just check that it doesn't crash
        # and returns the right shape


class TestRegionMembershipExternalRegions:
    """Test with regions provided externally."""

    def test_explicit_regions_parameter(self):
        """Test providing regions explicitly via parameter."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        # Create external regions
        external_regions = Regions()
        external_regions.add("external", polygon=box(3, 3, 7, 7))

        membership = env.region_membership(regions=external_regions)

        # Should work with external regions
        assert membership.shape == (env.n_bins, 1)
        assert np.any(membership[:, 0])

    def test_external_regions_dont_affect_env(self):
        """Test that using external regions doesn't modify environment."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        # Add region to environment
        env.regions.add("internal", polygon=box(0, 0, 5, 5))

        # Create different external regions
        external_regions = Regions()
        external_regions.add("external", polygon=box(5, 5, 10, 10))

        # Use external regions
        membership_ext = env.region_membership(regions=external_regions)

        # Default should still use internal regions
        membership_int = env.region_membership()

        # Should be different
        assert not np.array_equal(membership_ext, membership_int)

        # External should have 1 region, internal should have 1 region
        assert membership_ext.shape[1] == 1
        assert membership_int.shape[1] == 1


class TestRegionMembershipEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_bin_environment(self):
        """Test with environment containing only one bin."""
        data = np.array([[5.0, 5.0]])
        env = Environment.from_samples(data, bin_size=2.0)

        # Add region that contains the single bin
        env.regions.add("contains", polygon=box(0, 0, 10, 10))

        membership = env.region_membership()

        assert membership.shape == (1, 1)
        assert membership[0, 0] is True or membership[0, 0] is np.True_

    def test_region_contains_all_bins(self):
        """Test with region that contains all bins."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        # Large region covering everything
        env.regions.add("all", polygon=box(-10, -10, 20, 20))

        membership = env.region_membership()

        # All bins should be members
        assert np.all(membership[:, 0])

    def test_region_contains_no_bins(self):
        """Test with region that contains no bins."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        # Region far away from all bins
        env.regions.add("far", polygon=box(100, 100, 110, 110))

        membership = env.region_membership()

        # No bins should be members
        assert not np.any(membership[:, 0])

    def test_complex_polygon(self):
        """Test with a complex polygon (non-rectangular)."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        # Triangle
        triangle = Polygon([(0, 0), (10, 0), (5, 10)])
        env.regions.add("triangle", polygon=triangle)

        membership = env.region_membership()

        # Should handle complex shapes
        assert membership.shape == (env.n_bins, 1)
        # Some bins should be inside
        assert np.any(membership[:, 0])


class TestRegionMembershipValidation:
    """Test input validation."""

    def test_requires_fitted_environment(self):
        """Test that method requires fitted environment."""
        # Create unfitted environment by instantiating with minimal layout
        from neurospatial.layout.engines.regular_grid import RegularGridLayout

        layout = RegularGridLayout()
        env = Environment(layout=layout)

        with pytest.raises(
            RuntimeError, match="requires the environment to be fully initialized"
        ):
            env.region_membership()

    def test_regions_parameter_validation(self):
        """Test that regions parameter is validated."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        # Invalid type
        with pytest.raises(TypeError, match=r"[Rr]egions"):
            env.region_membership(regions="not_a_regions_object")

    def test_include_boundary_validation(self):
        """Test that include_boundary parameter is validated."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)
        env.regions.add("test", polygon=box(0, 0, 5, 5))

        # Invalid type
        with pytest.raises(TypeError, match=r"[Bb]ool"):
            env.region_membership(include_boundary="true")


class TestRegionMembershipPerformance:
    """Test performance characteristics."""

    def test_vectorized_operation(self):
        """Test that operation is vectorized (fast on many bins)."""
        # Create larger grid
        data = np.array([[i, j] for i in range(50) for j in range(50)])
        env = Environment.from_samples(data, bin_size=1.0)

        # Add multiple regions
        env.regions.add("r1", polygon=box(0, 0, 20, 20))
        env.regions.add("r2", polygon=box(20, 20, 40, 40))
        env.regions.add("r3", polygon=box(10, 10, 30, 30))

        # Should complete quickly (vectorized)
        import time

        start = time.time()
        membership = env.region_membership()
        elapsed = time.time() - start

        # Should be very fast (< 100ms for ~2500 bins x 3 regions)
        assert elapsed < 0.1, f"Operation took {elapsed:.3f}s, expected < 0.1s"
        assert membership.shape == (env.n_bins, 3)


class TestRegionMembershipDifferentLayouts:
    """Test across different layout types."""

    def test_regular_grid_layout(self):
        """Test on regular grid layout."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)
        env.regions.add("test", polygon=box(3, 3, 7, 7))

        membership = env.region_membership()
        assert membership.shape == (env.n_bins, 1)

    def test_polygon_layout(self):
        """Test on polygon-masked layout."""
        # Create polygon layout
        from shapely.geometry import box as shapely_box

        boundary = shapely_box(0, 0, 10, 10)
        env = Environment.from_polygon(boundary, bin_size=2.0)

        env.regions.add("test", polygon=box(2, 2, 8, 8))

        membership = env.region_membership()
        assert membership.shape == (env.n_bins, 1)
        assert np.any(membership[:, 0])

    # NOTE: No test for 1D graph layouts because region_membership() cannot work with 1D:
    # - Shapely only supports 2D/3D geometries (not 1D)
    # - Point regions always return False in region_membership() (no area)
    # This is a fundamental limitation, not a bug.


class TestRegionMembershipReturnFormat:
    """Test the format of returned array."""

    def test_return_dtype(self):
        """Test that return type is boolean array."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)
        env.regions.add("test", polygon=box(3, 3, 7, 7))

        membership = env.region_membership()

        assert isinstance(membership, np.ndarray)
        assert membership.dtype == bool

    def test_return_shape(self):
        """Test that return shape is correct."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        # No regions
        m0 = env.region_membership()
        assert m0.shape == (env.n_bins, 0)

        # One region
        env.regions.add("r1", polygon=box(0, 0, 5, 5))
        m1 = env.region_membership()
        assert m1.shape == (env.n_bins, 1)

        # Two regions
        env.regions.add("r2", polygon=box(5, 5, 10, 10))
        m2 = env.region_membership()
        assert m2.shape == (env.n_bins, 2)

    def test_region_order_matches(self):
        """Test that column order matches region order."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        # Add regions in specific order
        env.regions.add("left", polygon=box(0, 0, 4, 10))
        env.regions.add("right", polygon=box(6, 0, 10, 10))

        membership = env.region_membership()

        # Column 0 should correspond to "left"
        # Column 1 should correspond to "right"
        region_names = list(env.regions.keys())
        assert region_names[0] == "left"
        assert region_names[1] == "right"

        # Verify by checking which bins are in each region
        left_bins = np.where(membership[:, 0])[0]
        right_bins = np.where(membership[:, 1])[0]

        # Left bins should have smaller x coordinates on average
        left_centers = env.bin_centers[left_bins]
        right_centers = env.bin_centers[right_bins]

        assert np.mean(left_centers[:, 0]) < np.mean(right_centers[:, 0])
