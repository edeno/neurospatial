"""Test for regions_to_mask() function.

This module tests the rasterization of continuous polygon regions onto
discrete environment bins (the dual of mask_to_polygon).

Tests cover:
- Basic polygon rasterization
- Multiple regions
- Different include_boundary behaviors
- Input validation and error handling
"""

import numpy as np
import pytest
from shapely.geometry import box

from neurospatial import Environment
from neurospatial.ops.binning import regions_to_mask
from neurospatial.regions import Region, Regions


class TestRegionsToMaskBasic:
    """Test basic regions_to_mask functionality."""

    def test_single_polygon_region(self):
        """Test rasterizing a single polygon region onto bins."""
        # Create a 10x10 grid from 0 to 10
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        # Create a region covering the center (3,3) to (7,7)
        regions = Regions()
        regions.add("center", polygon=box(3, 3, 7, 7))

        # Rasterize onto bins
        mask = regions_to_mask(env, regions)

        # Should return bool array with length n_bins
        assert mask.shape == (env.n_bins,)
        assert mask.dtype == bool

        # Some bins should be inside, some outside
        assert np.any(mask)
        assert np.any(~mask)

        # Verify bins that should be inside are inside
        # Bin centers at positions 4,4 should be inside box(3,3,7,7)
        bin_at_center = env.bin_at(np.array([[5.0, 5.0]]))
        assert mask[bin_at_center[0]]

    def test_single_region_by_name(self):
        """Test passing a single region name from env.regions."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        # Add region to environment
        env.regions.add("center", polygon=box(3, 3, 7, 7))

        # Rasterize using region name
        mask = regions_to_mask(env, "center")

        assert mask.shape == (env.n_bins,)
        assert mask.dtype == bool
        assert np.any(mask)

    def test_multiple_region_names(self):
        """Test passing list of region names."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        env.regions.add("left", polygon=box(0, 0, 4, 10))
        env.regions.add("right", polygon=box(6, 0, 10, 10))

        # Rasterize union of both regions
        mask = regions_to_mask(env, ["left", "right"])

        assert mask.shape == (env.n_bins,)
        assert np.any(mask)

        # Bins at (2,5) and (8,5) should both be in the mask
        left_bin = env.bin_at(np.array([[2.0, 5.0]]))
        right_bin = env.bin_at(np.array([[8.0, 5.0]]))
        assert mask[left_bin[0]]
        assert mask[right_bin[0]]

    def test_single_region_object(self):
        """Test passing a single Region object."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        # Create region directly
        region = Region(name="test", kind="polygon", data=box(3, 3, 7, 7))

        # Rasterize
        mask = regions_to_mask(env, region)

        assert mask.shape == (env.n_bins,)
        assert np.any(mask)

    def test_overlapping_regions_union(self):
        """Test that overlapping regions produce union (logical OR)."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        regions = Regions()
        regions.add("region_a", polygon=box(0, 0, 6, 6))
        regions.add("region_b", polygon=box(4, 4, 10, 10))

        mask = regions_to_mask(env, regions)

        # Bins in overlap should be True (only counted once)
        overlap_bin = env.bin_at(np.array([[5.0, 5.0]]))
        assert mask[overlap_bin[0]]

        # Count bins - should be union not sum
        # (overlapping bins counted once)
        assert np.sum(mask) < env.n_bins  # Not all bins
        assert np.sum(mask) > 0  # But some bins


class TestRegionsToMaskBoundary:
    """Test boundary inclusion behavior."""

    def test_include_boundary_true(self):
        """Test that bins on boundary are included when include_boundary=True."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        # Create polygon with boundary exactly on bin center
        # Bin centers are at 1, 3, 5, 7, 9 (with bin_size=2)
        regions = Regions()
        regions.add("test", polygon=box(3, 3, 7, 7))

        mask_include = regions_to_mask(env, regions, include_boundary=True)

        # Bins with centers at (3,3), (3,5), (3,7) etc should be included
        boundary_bin = env.bin_at(np.array([[3.0, 5.0]]))
        assert mask_include[boundary_bin[0]]

    def test_include_boundary_false(self):
        """Test that bins on boundary are excluded when include_boundary=False."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        regions = Regions()
        regions.add("test", polygon=box(3, 3, 7, 7))

        mask_exclude = regions_to_mask(env, regions, include_boundary=False)
        mask_include = regions_to_mask(env, regions, include_boundary=True)

        # Excluding boundary should give <= bins than including boundary
        assert np.sum(mask_exclude) <= np.sum(mask_include)

    def test_boundary_default_is_true(self):
        """Test that default behavior includes boundary."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        regions = Regions()
        regions.add("test", polygon=box(3, 3, 7, 7))

        mask_default = regions_to_mask(env, regions)
        mask_explicit = regions_to_mask(env, regions, include_boundary=True)

        np.testing.assert_array_equal(mask_default, mask_explicit)


class TestRegionsToMaskEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_regions(self):
        """Test with empty Regions collection."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        empty_regions = Regions()
        mask = regions_to_mask(env, empty_regions)

        # All False when no regions
        assert mask.shape == (env.n_bins,)
        assert not np.any(mask)

    def test_region_outside_environment(self):
        """Test region completely outside environment bounds."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        regions = Regions()
        # Region far outside environment (which spans ~0 to 10)
        regions.add("outside", polygon=box(100, 100, 110, 110))

        mask = regions_to_mask(env, regions)

        # No bins should be inside
        assert not np.any(mask)

    def test_point_region_returns_false(self):
        """Test that point regions return all False (points have no area)."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        regions = Regions()
        regions.add("point", point=(5.0, 5.0))

        mask = regions_to_mask(env, regions)

        # Points have no area, so no bins can be "inside"
        assert not np.any(mask)

    def test_mixed_point_and_polygon(self):
        """Test with both point and polygon regions."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        regions = Regions()
        regions.add("point", point=(5.0, 5.0))
        regions.add("poly", polygon=box(3, 3, 7, 7))

        mask = regions_to_mask(env, regions)

        # Should only capture polygon bins (point has no area)
        assert np.any(mask)


class TestRegionsToMaskInputValidation:
    """Test input validation and error messages."""

    def test_invalid_region_name(self):
        """Test that invalid region name raises KeyError."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        with pytest.raises(KeyError, match="nonexistent"):
            regions_to_mask(env, "nonexistent")

    def test_invalid_regions_type(self):
        """Test that invalid regions parameter type raises TypeError."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        with pytest.raises(TypeError):
            regions_to_mask(env, 123)  # Invalid type

    def test_invalid_boundary_type(self):
        """Test that invalid include_boundary type raises TypeError."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        regions = Regions()
        regions.add("test", polygon=box(3, 3, 7, 7))

        with pytest.raises(TypeError):
            regions_to_mask(env, regions, include_boundary="yes")  # Should be bool


class TestRegionsToMaskDuality:
    """Test that regions_to_mask is the dual of mask_to_polygon."""

    def test_consistent_with_region_membership(self):
        """Test that results match Environment.region_membership()."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        env.regions.add("center", polygon=box(3, 3, 7, 7))
        env.regions.add("corner", polygon=box(0, 0, 3, 3))

        # Using regions_to_mask with all env.regions
        mask = regions_to_mask(env, env.regions)

        # Using region_membership (returns 2D array)
        membership = env.region_membership()

        # Union of all regions in region_membership
        membership_union = np.any(membership, axis=1)

        # Should match
        np.testing.assert_array_equal(mask, membership_union)
