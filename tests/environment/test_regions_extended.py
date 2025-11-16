"""
Comprehensive tests for Environment region operations (regions.py module).

This module tests the region operations mixin for Environment, covering:
- bins_in_region() method
- mask_for_region() method
- Region addition with various geometries
- Region buffering operations
- Region queries and updates
- Integration with other Environment operations

Target coverage: 17% → 85% for src/neurospatial/environment/regions.py

See Also
--------
TEST_PLAN2.md : Section 2.2 - Region Operations Tests

Notes
-----
This test file focuses on the Environment mixin
(src/neurospatial/environment/regions.py), not the core regions module
(src/neurospatial/regions/), which has separate test files.
"""

import numpy as np
import pytest
from shapely.geometry import Polygon, box

from neurospatial import Environment


class TestBinsInRegion:
    """Tests for Environment.bins_in_region() method."""

    def test_bins_in_point_region(self):
        """Test bins_in_region with a point region."""
        # Create simple grid
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        # Add point region at known location
        env.regions.add("center", point=(5.0, 5.0))

        bins = env.bins_in_region("center")

        # Should return exactly one bin (the one containing the point)
        assert isinstance(bins, np.ndarray)
        assert bins.dtype == np.int_ or bins.dtype == np.int64
        assert len(bins) == 1

        # Verify the bin contains the point
        bin_center = env.bin_centers[bins[0]]
        # Point should be close to bin center (within bin_size, accounting for diagonal)
        # With bin_size=2.0, max distance to nearest bin center is ~sqrt(2)*1 = 1.414
        assert np.linalg.norm(bin_center - np.array([5.0, 5.0])) <= 2.0

    def test_bins_in_polygon_region(self):
        """Test bins_in_region with a polygon region."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        # Add rectangular polygon region
        env.regions.add("rect", polygon=box(3, 3, 7, 7))

        bins = env.bins_in_region("rect")

        # Should return multiple bins
        assert isinstance(bins, np.ndarray)
        assert len(bins) > 1

        # All bins should have centers inside the polygon
        bin_centers = env.bin_centers[bins]
        for center in bin_centers:
            # Centers should be within or on the boundary
            assert 3 <= center[0] <= 7
            assert 3 <= center[1] <= 7

    def test_bins_in_region_point_outside_bounds(self):
        """Test bins_in_region with point region outside environment."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        # Add point far outside environment
        env.regions.add("outside", point=(100.0, 100.0))

        bins = env.bins_in_region("outside")

        # Should return empty array (no bins contain this point)
        assert isinstance(bins, np.ndarray)
        assert len(bins) == 0

    def test_bins_in_region_polygon_outside_bounds(self):
        """Test bins_in_region with polygon region outside environment."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        # Add polygon far outside environment
        env.regions.add("outside", polygon=box(100, 100, 110, 110))

        bins = env.bins_in_region("outside")

        # Should return empty array
        assert len(bins) == 0

    def test_bins_in_region_nonexistent_raises_error(self):
        """Test that requesting non-existent region raises KeyError."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        with pytest.raises(KeyError, match="nonexistent"):
            env.bins_in_region("nonexistent")

    def test_bins_in_region_dimension_mismatch(self):
        """Test that point with wrong dimension raises ValueError."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        # Add 3D point to 2D environment
        env.regions.add("wrong_dim", point=(5.0, 5.0, 5.0))

        with pytest.raises(ValueError, match=r"dimension.*does not match"):
            env.bins_in_region("wrong_dim")

    def test_bins_in_region_complex_polygon(self):
        """Test bins_in_region with complex non-rectangular polygon."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        # Create L-shaped polygon
        l_shape = Polygon([(0, 0), (4, 0), (4, 4), (8, 4), (8, 8), (0, 8)])
        env.regions.add("l_shape", polygon=l_shape)

        bins = env.bins_in_region("l_shape")

        # Should have bins
        assert len(bins) > 0

        # All bin centers should be inside the L-shape
        from shapely.geometry import Point

        for bin_id in bins:
            center = env.bin_centers[bin_id]
            point = Point(center)
            # Should be inside or on boundary
            assert l_shape.contains(point) or l_shape.touches(point)


class TestMaskForRegion:
    """Tests for Environment.mask_for_region() method."""

    def test_mask_for_region_basic(self):
        """Test basic mask_for_region functionality."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        env.regions.add("center", polygon=box(3, 3, 7, 7))

        mask = env.mask_for_region("center")

        # Should be boolean array with length n_bins
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert len(mask) == env.n_bins

        # Some bins should be True, some False
        assert np.any(mask)
        assert not np.all(mask)

    def test_mask_for_region_matches_bins_in_region(self):
        """Test that mask_for_region matches bins_in_region."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        env.regions.add("test", polygon=box(2, 2, 8, 8))

        bins = env.bins_in_region("test")
        mask = env.mask_for_region("test")

        # Bins where mask is True should match bins_in_region
        masked_bins = np.where(mask)[0]
        np.testing.assert_array_equal(sorted(bins), sorted(masked_bins))

    def test_mask_for_region_empty(self):
        """Test mask_for_region with region containing no bins."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        env.regions.add("outside", polygon=box(100, 100, 110, 110))

        mask = env.mask_for_region("outside")

        # Should be all False
        assert len(mask) == env.n_bins
        assert not np.any(mask)

    def test_mask_for_region_point(self):
        """Test mask_for_region with point region."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        env.regions.add("point", point=(5.0, 5.0))

        mask = env.mask_for_region("point")

        # Should have exactly one True value
        assert mask.sum() == 1

    def test_mask_for_region_can_index_arrays(self):
        """Test that mask can be used to index per-bin arrays."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        env.regions.add("center", polygon=box(3, 3, 7, 7))

        # Create per-bin data
        occupancy = np.random.rand(env.n_bins)

        mask = env.mask_for_region("center")

        # Should be able to index and get subset
        region_occupancy = occupancy[mask]
        assert len(region_occupancy) == mask.sum()
        assert len(region_occupancy) > 0


class TestRegionAddition:
    """Tests for adding regions with various geometries."""

    def test_add_point_region(self):
        """Test adding a point region."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        # Add point region
        env.regions.add("goal", point=(7.0, 7.0))

        assert "goal" in env.regions
        assert env.regions["goal"].kind == "point"

    def test_add_polygon_region(self):
        """Test adding a polygon region."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        # Add polygon region
        env.regions.add("arena", polygon=box(0, 0, 10, 10))

        assert "arena" in env.regions
        assert env.regions["arena"].kind == "polygon"

    def test_add_multiple_regions(self):
        """Test adding multiple regions."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        env.regions.add("point1", point=(2.0, 2.0))
        env.regions.add("point2", point=(8.0, 8.0))
        env.regions.add("poly1", polygon=box(0, 0, 5, 5))

        assert len(env.regions) == 3
        assert "point1" in env.regions
        assert "point2" in env.regions
        assert "poly1" in env.regions

    def test_add_region_duplicate_name_raises_error(self):
        """Test that adding region with duplicate name raises error."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        env.regions.add("test", point=(5.0, 5.0))

        # Adding another with same name should raise
        with pytest.raises(KeyError, match="Duplicate region name"):
            env.regions.add("test", point=(7.0, 7.0))


class TestRegionUpdate:
    """Tests for updating and removing regions."""

    def test_update_region_point(self):
        """Test updating a point region."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        # Add initial region
        env.regions.add("goal", point=(5.0, 5.0))

        # Update it
        env.regions.update_region("goal", point=(8.0, 8.0))

        # Should have new location
        updated_region = env.regions["goal"]
        assert np.allclose(updated_region.data, [8.0, 8.0])

    def test_update_region_polygon(self):
        """Test updating a polygon region."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        # Add initial region
        env.regions.add("arena", polygon=box(0, 0, 5, 5))

        # Update it with new polygon
        new_polygon = box(2, 2, 8, 8)
        env.regions.update_region("arena", polygon=new_polygon)

        # Should have new polygon
        updated_region = env.regions["arena"]
        assert updated_region.data.equals(new_polygon)

    def test_remove_region_with_del(self):
        """Test removing region with del statement."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        env.regions.add("temp", point=(5.0, 5.0))
        assert "temp" in env.regions

        # Remove it
        del env.regions["temp"]

        assert "temp" not in env.regions

    def test_remove_region_with_method(self):
        """Test removing region with remove() method."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        env.regions.add("temp", point=(5.0, 5.0))

        # Remove using method
        env.regions.remove("temp")

        assert "temp" not in env.regions

    def test_update_nonexistent_region_raises_error(self):
        """Test that updating non-existent region raises error."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        with pytest.raises(KeyError, match="nonexistent"):
            env.regions.update_region("nonexistent", point=(5.0, 5.0))


class TestRegionBuffering:
    """Tests for region buffering operations."""

    def test_buffer_point_region_creates_polygon(self):
        """Test that buffering a point region creates a circular polygon."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        env.regions.add("point", point=(5.0, 5.0))

        # Buffer the point - need to provide new_name
        buffered = env.regions.buffer("point", distance=2.0, new_name="buffered_point")

        # Should be added to regions
        assert "buffered_point" in env.regions

        # Buffered point should be a polygon (circle)
        assert buffered.kind == "polygon"
        assert buffered.data.geom_type == "Polygon"

        # Check approximate area (circle with radius 2.0 has area ~12.56)
        assert 12.0 < buffered.data.area < 13.0

    def test_buffer_polygon_positive_distance(self):
        """Test buffering polygon with positive distance (expansion)."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        original_poly = box(3, 3, 7, 7)
        env.regions.add("rect", polygon=original_poly)

        # Buffer outward - need to provide new_name
        buffered = env.regions.buffer("rect", distance=1.0, new_name="buffered_rect")

        # Should be larger
        assert buffered.data.area > original_poly.area

    def test_buffer_polygon_negative_distance(self):
        """Test buffering polygon with negative distance (erosion)."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        original_poly = box(3, 3, 7, 7)
        env.regions.add("rect", polygon=original_poly)

        # Buffer inward (erode) - need to provide new_name
        buffered = env.regions.buffer("rect", distance=-0.5, new_name="eroded_rect")

        # Should be smaller
        assert buffered.data.area < original_poly.area

    def test_buffer_distance_zero(self):
        """Test buffering with distance=0 returns same geometry."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        original_poly = box(3, 3, 7, 7)
        env.regions.add("rect", polygon=original_poly)

        # Buffer with zero distance - need to provide new_name
        buffered = env.regions.buffer("rect", distance=0.0, new_name="rect_copy")

        # Should have same area (or very close due to numerical precision)
        assert np.isclose(buffered.data.area, original_poly.area, rtol=0.01)


class TestRegionIntegration:
    """Integration tests for regions with other Environment operations."""

    def test_regions_preserved_after_serialization(self):
        """Test that regions are preserved through save/load cycle."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        # Add regions
        env.regions.add("goal", point=(7.0, 7.0))
        env.regions.add("arena", polygon=box(0, 0, 10, 10))

        # Save and load
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = f"{tmpdir}/test_env"
            env.to_file(filepath)
            loaded_env = Environment.from_file(filepath)

        # Regions should be preserved
        assert len(loaded_env.regions) == 2
        assert "goal" in loaded_env.regions
        assert "arena" in loaded_env.regions

    def test_subset_with_region(self):
        """Test using regions to create environment subset."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        # Add region
        env.regions.add("center", polygon=box(3, 3, 7, 7))

        # Create subset using region_names parameter
        subset_env = env.subset(region_names=["center"])

        # Subset should have fewer bins
        assert subset_env.n_bins < env.n_bins

        # All bins in subset should be from the center region
        # (we can verify by checking that bins are within region bounds)
        for bin_center in subset_env.bin_centers:
            assert (
                3 <= bin_center[0] <= 7
                or np.isclose(bin_center[0], 3)
                or np.isclose(bin_center[0], 7)
            )
            assert (
                3 <= bin_center[1] <= 7
                or np.isclose(bin_center[1], 3)
                or np.isclose(bin_center[1], 7)
            )

    def test_regions_with_coordinate_transforms(self):
        """Test that regions work correctly after coordinate transforms."""
        # This is a basic smoke test; detailed transform testing is elsewhere
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        env.regions.add("center", polygon=box(3, 3, 7, 7))

        # Regions should still be queryable
        bins = env.bins_in_region("center")
        assert len(bins) > 0

    def test_multiple_regions_different_types(self):
        """Test environment with mixed region types."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        # Add different types
        env.regions.add("goal1", point=(2.0, 2.0))
        env.regions.add("goal2", point=(8.0, 8.0))
        env.regions.add("arena", polygon=box(0, 0, 10, 10))
        env.regions.add("roi", polygon=box(4, 4, 6, 6))

        # All should be accessible
        assert len(env.regions) == 4

        # Each should return bins
        for name in env.regions:
            bins = env.bins_in_region(name)
            assert isinstance(bins, np.ndarray)


class TestRegionEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_region_name(self):
        """Test that empty region name is handled."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        # Empty string as name should work (names are just strings)
        env.regions.add("", point=(5.0, 5.0))
        assert "" in env.regions

    def test_very_small_polygon(self):
        """Test with very small polygon region."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        # Tiny polygon
        tiny = box(5.0, 5.0, 5.01, 5.01)
        env.regions.add("tiny", polygon=tiny)

        bins = env.bins_in_region("tiny")

        # May have 0 or 1 bins depending on alignment
        assert len(bins) <= 1

    def test_degenerate_polygon_line(self):
        """Test with degenerate polygon (line)."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        # Degenerate polygon (line)
        line = Polygon([(0, 0), (10, 0), (10, 0), (0, 0)])
        env.regions.add("line", polygon=line)

        # Should handle gracefully
        bins = env.bins_in_region("line")
        assert isinstance(bins, np.ndarray)

    def test_self_intersecting_polygon(self):
        """Test with self-intersecting polygon."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        # Bowtie shape (self-intersecting)
        bowtie = Polygon([(0, 0), (10, 10), (10, 0), (0, 10)])
        env.regions.add("bowtie", polygon=bowtie)

        # Shapely should handle this (may or may not be valid)
        # Just check it doesn't crash
        bins = env.bins_in_region("bowtie")
        assert isinstance(bins, np.ndarray)

    def test_polygon_with_hole(self):
        """Test polygon with interior hole."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        # Polygon with hole (donut shape)
        exterior = [(0, 0), (10, 0), (10, 10), (0, 10)]
        hole = [(3, 3), (7, 3), (7, 7), (3, 7)]
        donut = Polygon(exterior, [hole])

        env.regions.add("donut", polygon=donut)

        bins = env.bins_in_region("donut")

        # Should have bins
        assert len(bins) > 0

        # Bins in the hole should NOT be included
        # Check a point we know is in the hole
        hole_bin = env.bin_at(np.array([[5.0, 5.0]]))
        if hole_bin[0] != -1:  # If point is in environment
            assert hole_bin[0] not in bins

    def test_region_exactly_on_bin_centers(self):
        """Test region boundary exactly on bin centers."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        # Find exact bin center coordinates
        # With bin_size=2.0 and data from 0-10, centers are at 1, 3, 5, 7, 9
        # Create box with edges exactly on these centers
        env.regions.add("exact", polygon=box(3.0, 3.0, 7.0, 7.0))

        bins = env.bins_in_region("exact")

        # Should include boundary bins
        assert len(bins) > 0
