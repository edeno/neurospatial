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
        region_builder = Regions(env.regions.values())
        region_builder.add("center", polygon=box(3, 3, 7, 7))
        env = env.with_regions(region_builder)

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
        region_builder = Regions(env.regions.values())
        region_builder.add("left", polygon=box(0, 0, 4, 10))
        region_builder.add("right", polygon=box(6, 0, 10, 10))
        env = env.with_regions(region_builder)

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
        region_builder = Regions(env.regions.values())
        region_builder.add("region_a", polygon=box(0, 0, 6, 6))
        region_builder.add("region_b", polygon=box(4, 4, 10, 10))
        env = env.with_regions(region_builder)

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
        region_builder = Regions(env.regions.values())
        region_builder.add("test", polygon=box(0, 0, 5.0, 10))
        env = env.with_regions(region_builder)

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
        region_builder = Regions(env.regions.values())
        region_builder.add("test", polygon=box(0, 0, 5.0, 10))
        env = env.with_regions(region_builder)

        membership_with = env.region_membership(include_boundary=True)
        membership_without = env.region_membership(include_boundary=False)

        # Without boundary should have equal or fewer members
        assert np.sum(membership_without) <= np.sum(membership_with)

    def test_boundary_behavior_consistency(self):
        """Test that boundary parameter is consistent across multiple calls."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)
        region_builder = Regions(env.regions.values())
        region_builder.add("test", polygon=box(2, 2, 8, 8))
        env = env.with_regions(region_builder)

        # Multiple calls with same parameter should give same result
        m1 = env.region_membership(include_boundary=True)
        m2 = env.region_membership(include_boundary=True)
        assert np.array_equal(m1, m2)

        m3 = env.region_membership(include_boundary=False)
        m4 = env.region_membership(include_boundary=False)
        assert np.array_equal(m3, m4)


class TestRegionMembershipRegionTypes:
    """Test behavior with different region types."""

    def test_point_region_covers_the_bin_containing_it(self):
        """A point region covers exactly the bin that contains the point."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)
        region_builder = Regions(env.regions.values())
        region_builder.add("point", point=[5.0, 5.0])
        env = env.with_regions(region_builder)

        membership = env.region_membership()

        assert membership.shape == (env.n_bins, 1)
        np.testing.assert_array_equal(
            np.flatnonzero(membership[:, 0]), env.bin_at(np.array([[5.0, 5.0]]))
        )

    def test_mixed_region_types(self):
        """Point and polygon columns each follow the one region rule."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)
        region_builder = Regions(env.regions.values())
        region_builder.add("poly", polygon=box(2, 2, 8, 8))
        region_builder.add("point", point=[5.0, 5.0])
        env = env.with_regions(region_builder)

        membership = env.region_membership()

        assert membership.shape == (env.n_bins, 2)
        np.testing.assert_array_equal(
            np.flatnonzero(membership[:, 0]), env.bins_in_region("poly")
        )
        np.testing.assert_array_equal(
            np.flatnonzero(membership[:, 1]), env.bins_in_region("point")
        )
        assert membership[:, 1].sum() == 1

    def test_empty_polygon(self):
        """Test with an empty/degenerate polygon."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        # Create a degenerate polygon (line)
        region_builder = Regions(env.regions.values())
        region_builder.add("line", polygon=Polygon([(0, 0), (10, 0), (10, 0), (0, 0)]))
        env = env.with_regions(region_builder)

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
        region_builder = Regions(env.regions.values())
        region_builder.add("internal", polygon=box(0, 0, 5, 5))
        env = env.with_regions(region_builder)

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
        region_builder = Regions(env.regions.values())
        region_builder.add("contains", polygon=box(0, 0, 10, 10))
        env = env.with_regions(region_builder)

        membership = env.region_membership()

        assert membership.shape == (1, 1)
        assert membership[0, 0] is True or membership[0, 0] is np.True_

    def test_region_contains_all_bins(self):
        """Test with region that contains all bins."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        # Large region covering everything
        region_builder = Regions(env.regions.values())
        region_builder.add("all", polygon=box(-10, -10, 20, 20))
        env = env.with_regions(region_builder)

        membership = env.region_membership()

        # All bins should be members
        assert np.all(membership[:, 0])

    def test_region_contains_no_bins(self):
        """Test with region that contains no bins."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        # Region far away from all bins
        region_builder = Regions(env.regions.values())
        region_builder.add("far", polygon=box(100, 100, 110, 110))
        env = env.with_regions(region_builder)

        membership = env.region_membership()

        # No bins should be members
        assert not np.any(membership[:, 0])

    def test_complex_polygon(self):
        """Test with a complex polygon (non-rectangular)."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        # Triangle
        triangle = Polygon([(0, 0), (10, 0), (5, 10)])
        region_builder = Regions(env.regions.values())
        region_builder.add("triangle", polygon=triangle)
        env = env.with_regions(region_builder)

        membership = env.region_membership()

        # Should handle complex shapes
        assert membership.shape == (env.n_bins, 1)
        # Some bins should be inside
        assert np.any(membership[:, 0])


class TestRegionMembershipValidation:
    """Test input validation."""

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
        region_builder = Regions(env.regions.values())
        region_builder.add("test", polygon=box(0, 0, 5, 5))
        env = env.with_regions(region_builder)

        # Invalid type
        with pytest.raises(TypeError, match=r"[Bb]ool"):
            env.region_membership(include_boundary="true")


class TestRegionMembershipPerformance:
    """Test performance characteristics."""

    def test_scales_to_many_bins_and_regions(self):
        """Membership is computed correctly over many bins and regions.

        Exercises the vectorized path on a large (~2500-bin) grid with
        several overlapping regions. Asserts correctness rather than
        wall-clock time, which is unreliable under parallel test execution.
        """
        # Create larger grid
        data = np.array([[i, j] for i in range(50) for j in range(50)])
        env = Environment.from_samples(data, bin_size=1.0)

        # Add multiple regions (r3 overlaps both r1 and r2)
        region_builder = Regions(env.regions.values())
        region_builder.add("r1", polygon=box(0, 0, 20, 20))
        region_builder.add("r2", polygon=box(20, 20, 40, 40))
        region_builder.add("r3", polygon=box(10, 10, 30, 30))
        env = env.with_regions(region_builder)

        membership = env.region_membership()

        # Shape/dtype contract holds at scale.
        assert membership.shape == (env.n_bins, 3)
        assert membership.dtype == bool

        # Each region covers part (but not all) of the grid, so every column
        # has some members and some non-members — i.e. real per-bin results,
        # not an all-True/all-False degenerate output.
        r1, r2, r3 = membership[:, 0], membership[:, 1], membership[:, 2]
        for col in (r1, r2, r3):
            n_members = int(col.sum())
            assert 0 < n_members < env.n_bins

        # Geometric consistency across regions (the vectorized result is
        # correct per bin, not just well-shaped):
        # r3 = box(10,10,30,30) overlaps both r1 = box(0,0,20,20) and
        # r2 = box(20,20,40,40), so those intersections are non-empty.
        assert (r1 & r3).any()
        assert (r2 & r3).any()


class TestRegionMembershipDifferentLayouts:
    """Test across different layout types."""

    def test_regular_grid_layout(self):
        """Test on regular grid layout."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)
        region_builder = Regions(env.regions.values())
        region_builder.add("test", polygon=box(3, 3, 7, 7))
        env = env.with_regions(region_builder)

        membership = env.region_membership()
        assert membership.shape == (env.n_bins, 1)

    def test_polygon_layout(self):
        """Test on polygon-masked layout."""
        # Create polygon layout
        from shapely.geometry import box as shapely_box

        boundary = shapely_box(0, 0, 10, 10)
        env = Environment.from_polygon(boundary, bin_size=2.0)

        region_builder = Regions(env.regions.values())
        region_builder.add("test", polygon=box(2, 2, 8, 8))
        env = env.with_regions(region_builder)

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
        region_builder = Regions(env.regions.values())
        region_builder.add("test", polygon=box(3, 3, 7, 7))
        env = env.with_regions(region_builder)

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
        region_builder = Regions(env.regions.values())
        region_builder.add("r1", polygon=box(0, 0, 5, 5))
        env = env.with_regions(region_builder)
        m1 = env.region_membership()
        assert m1.shape == (env.n_bins, 1)

        # Two regions
        region_builder = Regions(env.regions.values())
        region_builder.add("r2", polygon=box(5, 5, 10, 10))
        env = env.with_regions(region_builder)
        m2 = env.region_membership()
        assert m2.shape == (env.n_bins, 2)

    def test_region_order_matches(self):
        """Test that column order matches region order."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        # Add regions in specific order
        region_builder = Regions(env.regions.values())
        region_builder.add("left", polygon=box(0, 0, 4, 10))
        region_builder.add("right", polygon=box(6, 0, 10, 10))
        env = env.with_regions(region_builder)

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
