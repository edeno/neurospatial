"""Test for regions_to_mask() function.

This module tests the rasterization of continuous polygon regions onto
discrete environment bins (the dual of mask_to_polygon).

Tests cover:
- Basic polygon rasterization
- Multiple regions
- Bin centers on a polygon's edge
- Input validation and error handling
"""

import numpy as np
import pytest
from shapely.geometry import box

from neurospatial import Environment
from neurospatial.ops.mapping import regions_to_mask
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
        region_builder = Regions(env.regions.values())
        region_builder.add("center", polygon=box(3, 3, 7, 7))
        env = env.with_regions(region_builder)

        # Rasterize using region name
        mask = regions_to_mask(env, "center")

        assert mask.shape == (env.n_bins,)
        assert mask.dtype == bool
        assert np.any(mask)

    def test_multiple_region_names(self):
        """Test passing list of region names."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        region_builder = Regions(env.regions.values())
        region_builder.add("left", polygon=box(0, 0, 4, 10))
        region_builder.add("right", polygon=box(6, 0, 10, 10))
        env = env.with_regions(region_builder)

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
    """Bin centers on a polygon's edge follow the half-open rule."""

    def test_centers_on_the_south_and_west_edges_are_in(self):
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)  # centers 0, 2, ..., 10

        regions = Regions()
        regions.add("test", polygon=box(2, 2, 6, 6))

        np.testing.assert_array_equal(
            env.bin_centers[regions_to_mask(env, regions)],
            [[2.0, 2.0], [2.0, 4.0], [4.0, 2.0], [4.0, 4.0]],
        )


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

    def test_point_region_selects_the_bin_containing_it(self):
        """A point region selects the one bin that contains the point."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        regions = Regions()
        regions.add("point", point=(5.0, 5.0))

        mask = regions_to_mask(env, regions)

        np.testing.assert_array_equal(
            np.flatnonzero(mask), env.bin_at(np.array([[5.0, 5.0]]))
        )

    def test_mixed_point_and_polygon(self):
        """Test with both point and polygon regions."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        regions = Regions()
        regions.add("point", point=(5.0, 5.0))
        regions.add("poly", polygon=box(3, 3, 7, 7))

        mask = regions_to_mask(env, regions)

        # The union of the polygon bins and the bin containing the point
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


class TestRegionsToMaskDuality:
    """Test that regions_to_mask is the dual of mask_to_polygon."""

    def test_consistent_with_region_membership(self):
        """Test that results match Environment.region_membership()."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        region_builder = Regions(env.regions.values())
        region_builder.add("center", polygon=box(3, 3, 7, 7))
        region_builder.add("corner", polygon=box(0, 0, 3, 3))
        env = env.with_regions(region_builder)

        # Using regions_to_mask with all env.regions
        mask = regions_to_mask(env, env.regions)

        # Using region_membership (returns 2D array)
        membership = env.region_membership()

        # Union of all regions in region_membership
        membership_union = np.any(membership, axis=1)

        # Should match
        np.testing.assert_array_equal(mask, membership_union)
