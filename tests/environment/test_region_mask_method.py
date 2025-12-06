"""Test for Environment.region_mask() method.

This module tests the region_mask() method on Environment, which provides
a convenient method-based interface to regions_to_mask() function.

Tests cover:
- Single region name
- Multiple region names (union)
- Single Region object
- Regions container
- Different include_boundary behaviors
- Input validation and error handling
- Feature parity with regions_to_mask() function
"""

import numpy as np
import pytest
from shapely.geometry import box

from neurospatial import Environment
from neurospatial.ops.binning import regions_to_mask
from neurospatial.regions import Region, Regions


class TestRegionMaskBasic:
    """Test basic region_mask functionality."""

    def test_single_region_by_name(self):
        """Test region_mask with single region name."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        # Add region to environment
        env.regions.add("center", polygon=box(3, 3, 7, 7))

        # Get mask using method
        mask = env.region_mask("center")

        assert mask.shape == (env.n_bins,)
        assert mask.dtype == bool
        assert np.any(mask)

        # Verify bins that should be inside are inside
        bin_at_center = env.bin_at(np.array([[5.0, 5.0]]))
        assert mask[bin_at_center[0]]

    def test_multiple_region_names(self):
        """Test region_mask with list of region names."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        env.regions.add("left", polygon=box(0, 0, 4, 10))
        env.regions.add("right", polygon=box(6, 0, 10, 10))

        # Get union of both regions
        mask = env.region_mask(["left", "right"])

        assert mask.shape == (env.n_bins,)
        assert np.any(mask)

        # Bins at (2,5) and (8,5) should both be in the mask
        left_bin = env.bin_at(np.array([[2.0, 5.0]]))
        right_bin = env.bin_at(np.array([[8.0, 5.0]]))
        assert mask[left_bin[0]]
        assert mask[right_bin[0]]

    def test_single_region_object(self):
        """Test region_mask with single Region object."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        # Create region directly (not added to env.regions)
        region = Region(name="test", kind="polygon", data=box(3, 3, 7, 7))

        # Get mask
        mask = env.region_mask(region)

        assert mask.shape == (env.n_bins,)
        assert np.any(mask)

        # Verify correct bins are included
        bin_at_center = env.bin_at(np.array([[5.0, 5.0]]))
        assert mask[bin_at_center[0]]

    def test_regions_container(self):
        """Test region_mask with Regions container."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        # Create regions container
        regions = Regions()
        regions.add("region_a", polygon=box(0, 0, 6, 6))
        regions.add("region_b", polygon=box(4, 4, 10, 10))

        # Get mask for all regions
        mask = env.region_mask(regions)

        assert mask.shape == (env.n_bins,)
        assert np.any(mask)

        # Verify union behavior - bins in overlap counted once
        overlap_bin = env.bin_at(np.array([[5.0, 5.0]]))
        assert mask[overlap_bin[0]]

    def test_all_env_regions(self):
        """Test region_mask with all regions from env.regions."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        env.regions.add("center", polygon=box(3, 3, 7, 7))
        env.regions.add("corner", polygon=box(0, 0, 3, 3))

        # Get mask for all env.regions
        mask = env.region_mask(env.regions)

        assert mask.shape == (env.n_bins,)
        assert np.any(mask)

        # Should be union of both regions
        center_bin = env.bin_at(np.array([[5.0, 5.0]]))
        corner_bin = env.bin_at(np.array([[1.0, 1.0]]))
        assert mask[center_bin[0]]
        assert mask[corner_bin[0]]


class TestRegionMaskBoundary:
    """Test boundary inclusion behavior."""

    def test_include_boundary_true(self):
        """Test that bins on boundary are included when include_boundary=True."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        env.regions.add("test", polygon=box(3, 3, 7, 7))

        mask_include = env.region_mask("test", include_boundary=True)

        # Bins with centers at (3,3), (3,5), (3,7) etc should be included
        boundary_bin = env.bin_at(np.array([[3.0, 5.0]]))
        assert mask_include[boundary_bin[0]]

    def test_include_boundary_false(self):
        """Test that bins on boundary are excluded when include_boundary=False."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        env.regions.add("test", polygon=box(3, 3, 7, 7))

        mask_exclude = env.region_mask("test", include_boundary=False)
        mask_include = env.region_mask("test", include_boundary=True)

        # Excluding boundary should give <= bins than including boundary
        assert np.sum(mask_exclude) <= np.sum(mask_include)

    def test_boundary_default_is_true(self):
        """Test that default behavior includes boundary."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        env.regions.add("test", polygon=box(3, 3, 7, 7))

        mask_default = env.region_mask("test")
        mask_explicit = env.region_mask("test", include_boundary=True)

        np.testing.assert_array_equal(mask_default, mask_explicit)


class TestRegionMaskEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_regions_container(self):
        """Test with empty Regions collection."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        empty_regions = Regions()
        mask = env.region_mask(empty_regions)

        # All False when no regions
        assert mask.shape == (env.n_bins,)
        assert not np.any(mask)

    def test_region_outside_environment(self):
        """Test region completely outside environment bounds."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        # Create region far outside environment (which spans ~0 to 10)
        outside_region = Region(
            name="outside", kind="polygon", data=box(100, 100, 110, 110)
        )

        mask = env.region_mask(outside_region)

        # No bins should be inside
        assert not np.any(mask)

    def test_point_region_returns_false(self):
        """Test that point regions return all False (points have no area)."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        env.regions.add("point", point=(5.0, 5.0))

        mask = env.region_mask("point")

        # Points have no area, so no bins can be "inside"
        assert not np.any(mask)

    def test_mixed_point_and_polygon(self):
        """Test with both point and polygon regions in list."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        env.regions.add("point", point=(5.0, 5.0))
        env.regions.add("poly", polygon=box(3, 3, 7, 7))

        mask = env.region_mask(["point", "poly"])

        # Should only capture polygon bins (point has no area)
        assert np.any(mask)


class TestRegionMaskInputValidation:
    """Test input validation and error messages."""

    def test_invalid_region_name(self):
        """Test that invalid region name raises KeyError."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        with pytest.raises(KeyError, match="nonexistent"):
            env.region_mask("nonexistent")

    def test_invalid_region_name_in_list(self):
        """Test that invalid region name in list raises KeyError."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        env.regions.add("valid", polygon=box(3, 3, 7, 7))

        with pytest.raises(KeyError, match="invalid"):
            env.region_mask(["valid", "invalid"])

    def test_invalid_regions_type(self):
        """Test that invalid regions parameter type raises TypeError."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        with pytest.raises(TypeError):
            env.region_mask(123)  # Invalid type

    def test_invalid_boundary_type(self):
        """Test that invalid include_boundary type raises TypeError."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        env.regions.add("test", polygon=box(3, 3, 7, 7))

        with pytest.raises(TypeError):
            env.region_mask("test", include_boundary="yes")  # Should be bool


class TestRegionMaskFeatureParity:
    """Test feature parity with regions_to_mask() free function."""

    def test_matches_regions_to_mask_single_name(self):
        """Test that region_mask matches regions_to_mask for single region name."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        env.regions.add("center", polygon=box(3, 3, 7, 7))

        # Method version
        mask_method = env.region_mask("center")

        # Function version
        mask_function = regions_to_mask(env, "center")

        # Should be identical
        np.testing.assert_array_equal(mask_method, mask_function)

    def test_matches_regions_to_mask_multiple_names(self):
        """Test that region_mask matches regions_to_mask for multiple names."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        env.regions.add("left", polygon=box(0, 0, 4, 10))
        env.regions.add("right", polygon=box(6, 0, 10, 10))

        # Method version
        mask_method = env.region_mask(["left", "right"])

        # Function version
        mask_function = regions_to_mask(env, ["left", "right"])

        # Should be identical
        np.testing.assert_array_equal(mask_method, mask_function)

    def test_matches_regions_to_mask_region_object(self):
        """Test that region_mask matches regions_to_mask for Region object."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        region = Region(name="test", kind="polygon", data=box(3, 3, 7, 7))

        # Method version
        mask_method = env.region_mask(region)

        # Function version
        mask_function = regions_to_mask(env, region)

        # Should be identical
        np.testing.assert_array_equal(mask_method, mask_function)

    def test_matches_regions_to_mask_regions_container(self):
        """Test that region_mask matches regions_to_mask for Regions container."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        regions = Regions()
        regions.add("region_a", polygon=box(0, 0, 6, 6))
        regions.add("region_b", polygon=box(4, 4, 10, 10))

        # Method version
        mask_method = env.region_mask(regions)

        # Function version
        mask_function = regions_to_mask(env, regions)

        # Should be identical
        np.testing.assert_array_equal(mask_method, mask_function)

    def test_matches_regions_to_mask_boundary_false(self):
        """Test that region_mask matches regions_to_mask with include_boundary=False."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        env.regions.add("test", polygon=box(3, 3, 7, 7))

        # Method version
        mask_method = env.region_mask("test", include_boundary=False)

        # Function version
        mask_function = regions_to_mask(env, "test", include_boundary=False)

        # Should be identical
        np.testing.assert_array_equal(mask_method, mask_function)
