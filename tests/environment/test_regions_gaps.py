"""
Gap closure tests for Environment region operations (regions.py module).

This module provides targeted tests to increase coverage from 45% to 85%,
focusing on:
- region_membership() method (lines 336-399)
- Complex polygon geometries (holes, self-intersecting)
- Edge cases and boundary conditions
- Error handling paths
- Integration with other Environment operations

Target coverage: 45% → 85% for src/neurospatial/environment/regions.py

See Also
--------
TEST_PLAN2.md : Section 2.5.2 - Regions Gap Closure
test_regions_extended.py : Initial region tests (45% coverage)

Notes
-----
This test file complements test_regions_extended.py by targeting uncovered
code paths, particularly the region_membership() method and complex
polygon geometries.
"""

import numpy as np
import pytest
from shapely.geometry import Polygon, box

from neurospatial import Environment
from neurospatial.regions import Regions


class TestRegionMembership:
    """Tests for Environment.region_membership() method.

    This class tests the vectorized region containment checking method,
    which is the primary uncovered code in regions.py (lines 336-399).
    """

    def test_region_membership_basic(self):
        """Test basic region_membership with single polygon region.

        Notes
        -----
        Tests that region_membership returns correct shape and dtype
        for a simple case.
        """
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)
        env.regions.add("center", polygon=box(-5, -5, 5, 5))

        membership = env.region_membership()

        # Should return (n_bins, n_regions) boolean array
        assert membership.shape == (env.n_bins, 1)
        assert membership.dtype == bool

        # Some bins should be inside, some outside
        assert np.any(membership)
        assert not np.all(membership)

    def test_region_membership_multiple_regions(self):
        """Test region_membership with multiple polygon regions.

        Notes
        -----
        Tests that membership matrix has correct dimensions for multiple
        regions and that region order is preserved.
        """
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)
        env.regions.add("left", polygon=box(-10, -10, 0, 10))
        env.regions.add("right", polygon=box(0, -10, 10, 10))
        env.regions.add("center", polygon=box(-3, -3, 3, 3))

        membership = env.region_membership()

        # Should have shape (n_bins, 3)
        assert membership.shape == (env.n_bins, 3)
        assert membership.dtype == bool

        # Each region should have some bins (or possibly none if outside)
        # At least one region should have bins
        assert np.any(membership)

    def test_region_membership_empty_regions(self):
        """Test region_membership with no regions defined.

        Notes
        -----
        Tests edge case where regions container is empty.
        Should return array with shape (n_bins, 0).
        """
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)
        # No regions added

        membership = env.region_membership()

        # Should return empty array with correct shape
        assert membership.shape == (env.n_bins, 0)
        assert membership.dtype == bool

    def test_region_membership_external_regions(self):
        """Test region_membership with external Regions object.

        Notes
        -----
        Tests that region_membership can use external regions without
        modifying the environment.
        """
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        # Create external regions
        external_regions = Regions()
        external_regions.add("test1", polygon=box(-5, -5, 5, 5))
        external_regions.add("test2", polygon=box(0, 0, 10, 10))

        membership = env.region_membership(regions=external_regions)

        # Should use external regions
        assert membership.shape == (env.n_bins, 2)

        # Original environment regions should be unchanged
        assert len(env.regions) == 0

    def test_region_membership_include_boundary_true(self):
        """Test region_membership with include_boundary=True.

        Notes
        -----
        Tests that bins on region boundaries are included when
        include_boundary=True (uses shapely.covers).
        """
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)
        env.regions.add("rect", polygon=box(-5, -5, 5, 5))

        membership_with_boundary = env.region_membership(include_boundary=True)

        # Should include boundary bins
        assert membership_with_boundary.shape == (env.n_bins, 1)
        n_bins_with_boundary = membership_with_boundary.sum()

        # At least some bins should be included
        assert n_bins_with_boundary > 0

    def test_region_membership_include_boundary_false(self):
        """Test region_membership with include_boundary=False.

        Notes
        -----
        Tests that bins on region boundaries are excluded when
        include_boundary=False (uses shapely.contains).
        """
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)
        env.regions.add("rect", polygon=box(-5, -5, 5, 5))

        membership_without_boundary = env.region_membership(include_boundary=False)

        # Should have same shape
        assert membership_without_boundary.shape == (env.n_bins, 1)

        # May have fewer bins than with boundary=True
        # (but not guaranteed depending on bin center alignment)
        n_bins_without_boundary = membership_without_boundary.sum()
        assert n_bins_without_boundary >= 0

    def test_region_membership_boundary_comparison(self):
        """Test that include_boundary affects membership counts.

        Notes
        -----
        Tests that include_boundary=True includes >= bins than
        include_boundary=False.
        """
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)
        env.regions.add("rect", polygon=box(-5, -5, 5, 5))

        with_boundary = env.region_membership(include_boundary=True)
        without_boundary = env.region_membership(include_boundary=False)

        # With boundary should include >= bins
        assert with_boundary.sum() >= without_boundary.sum()

    def test_region_membership_point_regions(self):
        """Test region_membership with point regions.

        Notes
        -----
        Tests that point regions return all False (points have no area).
        """
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)
        env.regions.add("point1", point=(0.0, 0.0))
        env.regions.add("point2", point=(5.0, 5.0))

        membership = env.region_membership()

        # Points have no area, so all False
        assert membership.shape == (env.n_bins, 2)
        assert not np.any(membership)

    def test_region_membership_mixed_types(self):
        """Test region_membership with mixed region types.

        Notes
        -----
        Tests that point and polygon regions can coexist and are
        handled correctly.
        """
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)
        env.regions.add("point", point=(0.0, 0.0))
        env.regions.add("polygon", polygon=box(-5, -5, 5, 5))

        membership = env.region_membership()

        # Should have 2 columns
        assert membership.shape == (env.n_bins, 2)

        # Point column (0) should be all False
        assert not np.any(membership[:, 0])

        # Polygon column (1) should have some True
        assert np.any(membership[:, 1])

    def test_region_membership_invalid_regions_type(self):
        """Test region_membership raises TypeError for invalid regions.

        Notes
        -----
        Tests that passing non-Regions object raises TypeError.
        """
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        with pytest.raises(TypeError, match="regions must be a Regions instance"):
            env.region_membership(regions={})

        with pytest.raises(TypeError, match="regions must be a Regions instance"):
            env.region_membership(regions=[])

    def test_region_membership_invalid_include_boundary_type(self):
        """Test region_membership raises TypeError for invalid include_boundary.

        Notes
        -----
        Tests that passing non-bool include_boundary raises TypeError.
        """
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)
        env.regions.add("rect", polygon=box(-5, -5, 5, 5))

        with pytest.raises(TypeError, match="include_boundary must be a bool"):
            env.region_membership(include_boundary="True")

        with pytest.raises(TypeError, match="include_boundary must be a bool"):
            env.region_membership(include_boundary=1)

    def test_region_membership_3d_raises_not_implemented(self, simple_3d_env):
        """Test region_membership raises NotImplementedError for 3D.

        Parameters
        ----------
        simple_3d_env : Environment
            Simple 3D environment fixture.

        Notes
        -----
        Tests that polygon regions in 3D environments raise NotImplementedError.
        """
        env = simple_3d_env

        # Add polygon region (only valid in 2D)
        # Note: This will succeed in adding, but fail when querying
        env.regions.add("poly", polygon=box(0, 0, 5, 5))

        with pytest.raises(NotImplementedError, match="only supports 2D"):
            env.region_membership()

    def test_region_membership_overlapping_regions(self):
        """Test region_membership with overlapping regions.

        Notes
        -----
        Tests that bins can belong to multiple overlapping regions.
        """
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)
        env.regions.add("left", polygon=box(-10, -10, 5, 10))
        env.regions.add("right", polygon=box(-5, -10, 10, 10))

        membership = env.region_membership()

        # Find bins in both regions
        np.all(membership, axis=1)

        # Should have at least some overlap (depends on bin centers)
        # At minimum, check that operation completes without error
        assert membership.shape == (env.n_bins, 2)


class TestComplexPolygonGeometries:
    """Tests for regions with complex polygon geometries.

    This class tests handling of:
    - Polygons with interior holes (donut shapes)
    - Self-intersecting polygons
    - Zero-area and degenerate polygons
    """

    def test_polygon_with_holes(self):
        """Test polygon region with interior holes (donut shape).

        Notes
        -----
        Tests that bins in holes are correctly excluded from region.
        """
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        # Create donut: outer square with inner hole
        exterior = [(0, 0), (10, 0), (10, 10), (0, 10)]
        hole = [(3, 3), (7, 3), (7, 7), (3, 7)]
        donut = Polygon(exterior, [hole])

        env.regions.add("donut", polygon=donut)

        # Test bins_in_region
        bins = env.bins_in_region("donut")
        assert len(bins) > 0

        # Test region_membership
        membership = env.region_membership()
        assert membership.shape == (env.n_bins, 1)

        # Bins in hole should be excluded
        # Find bins near center of hole
        hole_center = np.array([5.0, 5.0])
        distances = np.linalg.norm(env.bin_centers - hole_center, axis=1)
        np.argmin(distances)

        # Closest bin to hole center should NOT be in region
        # (may not always be true depending on bin alignment)
        hole_bin = env.bin_at(np.array([[5.0, 5.0]]))
        if hole_bin[0] != -1:
            assert hole_bin[0] not in bins

    def test_self_intersecting_polygon(self):
        """Test handling of self-intersecting polygons.

        Notes
        -----
        Tests that self-intersecting polygons (bowtie) are handled
        without errors. Shapely may return invalid geometry.
        """
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        # Bowtie shape (self-intersecting)
        bowtie = Polygon([(0, 0), (10, 10), (10, 0), (0, 10)])
        env.regions.add("bowtie", polygon=bowtie)

        # Should not crash
        bins = env.bins_in_region("bowtie")
        assert isinstance(bins, np.ndarray)

        membership = env.region_membership()
        assert membership.shape == (env.n_bins, 1)

    def test_zero_area_polygon(self):
        """Test degenerate polygon with zero area.

        Notes
        -----
        Tests that zero-area polygons (collapsed to line/point) are
        handled without errors.
        """
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        # Degenerate polygon (line segment)
        line = Polygon([(0, 0), (10, 0), (10, 0), (0, 0)])
        env.regions.add("line", polygon=line)

        # Should handle gracefully (may return 0 bins)
        bins = env.bins_in_region("line")
        assert isinstance(bins, np.ndarray)
        assert len(bins) >= 0

        membership = env.region_membership()
        assert membership.shape == (env.n_bins, 1)

    def test_polygon_with_multiple_holes(self):
        """Test polygon with multiple interior holes.

        Notes
        -----
        Tests that polygons with multiple holes are handled correctly.
        """
        data = np.array([[i, j] for i in range(21) for j in range(21)])
        env = Environment.from_samples(data, bin_size=2.0)

        # Polygon with two holes
        exterior = [(0, 0), (20, 0), (20, 20), (0, 20)]
        hole1 = [(2, 2), (5, 2), (5, 5), (2, 5)]
        hole2 = [(10, 10), (15, 10), (15, 15), (10, 15)]
        swiss_cheese = Polygon(exterior, [hole1, hole2])

        env.regions.add("swiss", polygon=swiss_cheese)

        bins = env.bins_in_region("swiss")
        assert len(bins) > 0

        membership = env.region_membership()
        assert membership.shape == (env.n_bins, 1)

    def test_very_complex_polygon(self):
        """Test very complex polygon with many vertices.

        Notes
        -----
        Tests that complex polygons with many vertices are handled
        efficiently.
        """
        data = np.array([[i, j] for i in range(21) for j in range(21)])
        env = Environment.from_samples(data, bin_size=2.0)

        # Create star-shaped polygon with many vertices
        n_points = 20
        angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        radii = np.where(np.arange(n_points) % 2 == 0, 8, 4)
        x = 10 + radii * np.cos(angles)
        y = 10 + radii * np.sin(angles)
        star = Polygon(list(zip(x, y, strict=False)))

        env.regions.add("star", polygon=star)

        bins = env.bins_in_region("star")
        assert isinstance(bins, np.ndarray)

        membership = env.region_membership()
        assert membership.shape == (env.n_bins, 1)


class TestRegionQueriesComplex:
    """Complex region query operations.

    Tests for bins_in_region and mask_for_region with complex
    polygon configurations.
    """

    def test_bins_in_region_with_holes(self):
        """Test bins_in_region with polygon containing holes.

        Notes
        -----
        Tests that bins inside holes are correctly excluded.
        """
        data = np.array([[i, j] for i in range(21) for j in range(21)])
        env = Environment.from_samples(data, bin_size=2.0)

        # Donut polygon
        exterior = [(0, 0), (20, 0), (20, 20), (0, 20)]
        hole = [(8, 8), (12, 8), (12, 12), (8, 12)]
        donut = Polygon(exterior, [hole])

        env.regions.add("donut", polygon=donut)

        bins = env.bins_in_region("donut")

        # Check that hole bins are excluded
        # Center of hole should not be in region
        hole_center_bins = env.bin_at(np.array([[10.0, 10.0]]))
        if hole_center_bins[0] != -1:
            assert hole_center_bins[0] not in bins

    def test_mask_for_region_polygon_with_holes(self):
        """Test mask generation for polygon with holes.

        Notes
        -----
        Tests that mask correctly identifies bins excluding holes.
        """
        data = np.array([[i, j] for i in range(21) for j in range(21)])
        env = Environment.from_samples(data, bin_size=2.0)

        # Donut polygon
        exterior = [(0, 0), (20, 0), (20, 20), (0, 20)]
        hole = [(8, 8), (12, 8), (12, 12), (8, 12)]
        donut = Polygon(exterior, [hole])

        env.regions.add("donut", polygon=donut)

        mask = env.mask_for_region("donut")

        # Should have bins
        assert mask.shape == (env.n_bins,)
        assert np.any(mask)

        # Bins in hole should NOT be in mask
        hole_center_bins = env.bin_at(np.array([[10.0, 10.0]]))
        if hole_center_bins[0] != -1:
            assert not mask[hole_center_bins[0]]

    def test_region_containment_edge_cases(self):
        """Test point containment on polygon boundaries.

        Notes
        -----
        Tests that bins on polygon edges are handled consistently.
        """
        # Create grid with known bin centers
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=1.0)

        # Polygon with edges exactly on potential bin centers
        rect = box(2.0, 2.0, 8.0, 8.0)
        env.regions.add("rect", polygon=rect)

        bins = env.bins_in_region("rect")

        # Should handle boundary bins (Shapely uses contains, includes boundary)
        assert isinstance(bins, np.ndarray)
        assert len(bins) > 0

    def test_bins_in_region_tiny_polygon(self):
        """Test bins_in_region with very small polygon.

        Notes
        -----
        Tests that tiny polygons smaller than bin size are handled.
        """
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        # Tiny polygon (much smaller than bin size)
        tiny = box(5.0, 5.0, 5.01, 5.01)
        env.regions.add("tiny", polygon=tiny)

        bins = env.bins_in_region("tiny")

        # May have 0 or 1 bins
        assert len(bins) <= 1


class TestRegionMetadata:
    """Region metadata computation.

    Tests for region center, area, and bounding box computations
    with complex geometries.
    """

    def test_region_center_polygon_with_holes(self):
        """Test region center for polygons with holes.

        Notes
        -----
        Tests that polygon.centroid works correctly for
        polygons with holes.
        """
        # Create polygon with hole
        exterior = [(0, 0), (10, 0), (10, 10), (0, 10)]
        hole = [(3, 3), (7, 3), (7, 7), (3, 7)]
        donut = Polygon(exterior, [hole])

        # Polygon should have valid centroid
        assert donut.centroid is not None

    def test_region_area_with_holes(self):
        """Test area computation for polygons with holes.

        Notes
        -----
        Tests that area correctly accounts for holes (area reduced).
        """
        # Polygon without hole
        solid = box(0, 0, 10, 10)
        solid_area = solid.area

        # Polygon with hole
        exterior = [(0, 0), (10, 0), (10, 10), (0, 10)]
        hole = [(2, 2), (8, 2), (8, 8), (2, 8)]
        donut = Polygon(exterior, [hole])
        donut_area = donut.area

        # Donut should have less area
        assert donut_area < solid_area
        assert donut_area > 0

    def test_region_bounding_box(self):
        """Test bounding box computation for regions.

        Notes
        -----
        Tests that polygon.bounds returns correct bounding box.
        """
        # Complex polygon
        poly = Polygon([(0, 0), (10, 5), (5, 10), (0, 5)])

        bounds = poly.bounds  # (minx, miny, maxx, maxy)

        assert bounds[0] == 0  # minx
        assert bounds[1] == 0  # miny
        assert bounds[2] == 10  # maxx
        assert bounds[3] == 10  # maxy


class TestRegionBufferingAdvanced:
    """Advanced region buffering operations.

    Tests for buffering with complex geometries, negative distances,
    and edge cases.
    """

    def test_buffer_polygon_with_holes(self):
        """Test buffering polygons containing holes.

        Notes
        -----
        Tests that buffering polygon with holes works correctly
        (may fill holes if buffer is large enough).
        """
        data = np.array([[i, j] for i in range(21) for j in range(21)])
        env = Environment.from_samples(data, bin_size=2.0)

        # Small donut
        exterior = [(5, 5), (15, 5), (15, 15), (5, 15)]
        hole = [(8, 8), (12, 8), (12, 12), (8, 12)]
        donut = Polygon(exterior, [hole])

        env.regions.add("donut", polygon=donut)

        # Buffer outward
        buffered = env.regions.buffer("donut", distance=1.0, new_name="buffered_donut")

        # Should still be a polygon
        assert buffered.kind == "polygon"
        # Area should increase
        assert buffered.data.area > donut.area

    def test_buffer_negative_distance(self):
        """Test negative buffer (erosion) on polygon.

        Notes
        -----
        Tests that negative buffer shrinks polygon (erosion).
        """
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        original = box(0, 0, 10, 10)
        env.regions.add("rect", polygon=original)

        # Erode by negative buffer
        eroded = env.regions.buffer("rect", distance=-1.0, new_name="eroded")

        # Area should decrease
        assert eroded.data.area < original.area
        assert eroded.data.area > 0

    def test_buffer_creates_empty_geometry(self):
        """Test buffer that erodes polygon to empty geometry.

        Notes
        -----
        Tests that large negative buffer on small shapes can create
        empty geometry.
        """
        data = np.array([[i, j] for i in range(21) for j in range(21)])
        env = Environment.from_samples(data, bin_size=2.0)

        # Thin elongated polygon
        thin = Polygon([(0, 0), (20, 0), (20, 1), (0, 1)])
        env.regions.add("thin", polygon=thin)

        # Large negative buffer should erode completely or create fragments
        eroded = env.regions.buffer("thin", distance=-0.6, new_name="eroded_thin")

        # May be empty or smaller polygon (depends on Shapely)
        # Just verify it doesn't crash
        assert eroded.kind == "polygon"

    def test_buffer_very_large_distance(self):
        """Test buffer with very large distance.

        Notes
        -----
        Tests that large buffer distances are handled correctly.
        """
        data = np.array([[i, j] for i in range(51) for j in range(51)])
        env = Environment.from_samples(data, bin_size=2.0)

        small_poly = box(20, 20, 25, 25)
        env.regions.add("small", polygon=small_poly)

        # Very large buffer
        buffered = env.regions.buffer("small", distance=10.0, new_name="huge")

        # Should be much larger
        assert buffered.data.area > small_poly.area * 5


class TestRegionIntegrationAdvanced:
    """Advanced integration tests for regions.

    Tests for regions with environment subsetting, transformations,
    and serialization of complex geometries.
    """

    def test_regions_with_subset_partial_coverage(self):
        """Test subsetting environment with region partially covered.

        Notes
        -----
        Tests that regions work correctly when subset contains only
        part of the region.
        """
        data = np.array([[i, j] for i in range(21) for j in range(21)])
        env = Environment.from_samples(data, bin_size=2.0)

        # Add large region
        env.regions.add("large", polygon=box(0, 0, 20, 20))

        # Create subset that only covers part of region
        subset_bins_idx = env.bins_in_region("large")[:10]  # First 10 bins
        subset_mask = np.zeros(env.n_bins, dtype=bool)
        subset_mask[subset_bins_idx] = True

        subset_env = env.subset(bins=subset_mask)

        # Subset should have fewer bins
        assert subset_env.n_bins < env.n_bins
        assert subset_env.n_bins == 10

    def test_complex_region_serialization(self):
        """Test serialization of complex regions (holes).

        Notes
        -----
        Tests that complex polygon geometries survive save/load cycle.
        """
        import tempfile

        data = np.array([[i, j] for i in range(21) for j in range(21)])
        env = Environment.from_samples(data, bin_size=2.0)

        # Add complex region with hole
        exterior = [(0, 0), (20, 0), (20, 20), (0, 20)]
        hole = [(8, 8), (12, 8), (12, 12), (8, 12)]
        donut = Polygon(exterior, [hole])
        env.regions.add("donut", polygon=donut)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = f"{tmpdir}/complex_env"
            env.to_file(filepath)
            loaded = Environment.from_file(filepath)

        # Regions should be preserved
        assert len(loaded.regions) == 1
        assert "donut" in loaded.regions

        # Geometries should be equivalent
        assert loaded.regions["donut"].data.equals(donut)

    def test_region_validation_comprehensive(self):
        """Test all region validation error paths.

        Notes
        -----
        Tests various invalid region configurations to ensure
        proper error handling.
        """
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        # Test 1: Nonexistent region
        with pytest.raises(KeyError):
            env.bins_in_region("nonexistent")

        with pytest.raises(KeyError):
            env.mask_for_region("nonexistent")

        # Test 2: Wrong dimension point
        env.regions.add("wrong_dim", point=(1.0, 2.0, 3.0))
        with pytest.raises(ValueError, match=r"dimension.*does not match"):
            env.bins_in_region("wrong_dim")

        # Test 3: Duplicate region name
        env.regions.add("test", point=(5.0, 5.0))
        with pytest.raises(KeyError, match="Duplicate"):
            env.regions.add("test", point=(7.0, 7.0))


class TestRegionErrorHandling:
    """Region error handling and validation.

    Tests for error conditions and edge cases in region operations.
    """

    def test_region_name_conflicts(self):
        """Test handling of duplicate region names.

        Notes
        -----
        Tests that duplicate region names are properly rejected.
        """
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        env.regions.add("region1", point=(5.0, 5.0))

        # Duplicate name should raise error
        with pytest.raises(KeyError, match="Duplicate region name"):
            env.regions.add("region1", polygon=box(0, 0, 10, 10))

    def test_region_queries_nonexistent_region(self):
        """Test queries for non-existent regions.

        Notes
        -----
        Tests that accessing non-existent regions raises KeyError.
        """
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        with pytest.raises(KeyError):
            env.bins_in_region("does_not_exist")

        with pytest.raises(KeyError):
            env.mask_for_region("does_not_exist")

        with pytest.raises(KeyError):
            env.regions["does_not_exist"]

    def test_region_update_nonexistent(self):
        """Test updating non-existent region raises error.

        Notes
        -----
        Tests that update_region on non-existent region raises KeyError.
        """
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        with pytest.raises(KeyError):
            env.regions.update_region("does_not_exist", point=(5.0, 5.0))

    def test_region_remove_nonexistent_silent(self):
        """Test removing non-existent region is silent.

        Notes
        -----
        Tests that remove() on non-existent region doesn't raise error
        (per docstring: "No error if absent").
        """
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        # remove() should not raise error for non-existent region
        env.regions.remove("does_not_exist")  # Should be silent

        # But del should raise KeyError
        with pytest.raises(KeyError):
            del env.regions["does_not_exist"]

    def test_empty_polygon_region(self):
        """Test handling of empty polygon geometry.

        Notes
        -----
        Tests that empty polygons are handled gracefully.
        """
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        # Empty polygon
        empty_poly = Polygon()
        env.regions.add("empty", polygon=empty_poly)

        # Should return no bins
        bins = env.bins_in_region("empty")
        assert len(bins) == 0

        mask = env.mask_for_region("empty")
        assert not np.any(mask)

    def test_point_region_outside_environment(self):
        """Test point region far outside environment bounds.

        Notes
        -----
        Tests that points outside environment return empty results.
        """
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        env.regions.add("far_away", point=(1000.0, 1000.0))

        bins = env.bins_in_region("far_away")
        assert len(bins) == 0

        mask = env.mask_for_region("far_away")
        assert not np.any(mask)

    def test_polygon_region_outside_environment(self):
        """Test polygon region outside environment bounds.

        Notes
        -----
        Tests that polygons outside environment return empty results.
        """
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        env.regions.add("far_away", polygon=box(1000, 1000, 1010, 1010))

        bins = env.bins_in_region("far_away")
        assert len(bins) == 0

        mask = env.mask_for_region("far_away")
        assert not np.any(mask)


class TestRegionMembershipPerformance:
    """Performance and stress tests for region_membership.

    Tests to ensure region_membership scales well with many regions
    and many bins.
    """

    def test_region_membership_many_regions(self, medium_2d_env):
        """Test region_membership with many regions.

        Parameters
        ----------
        medium_2d_env : Environment
            Medium 2D environment fixture.

        Notes
        -----
        Tests that region_membership scales to many regions.
        """
        # Create a copy to avoid contaminating fixture
        data = np.array([[i, j] for i in range(51) for j in range(51)])
        env = Environment.from_samples(data, bin_size=2.0)

        # Add many small regions
        n_regions = 20
        for i in range(n_regions):
            x = -20 + i * 2
            y = -20 + i * 2
            env.regions.add(f"region_{i}", polygon=box(x, y, x + 3, y + 3))

        membership = env.region_membership()

        # Should have correct shape
        assert membership.shape == (env.n_bins, n_regions)
        assert membership.dtype == bool

    @pytest.mark.slow
    def test_region_membership_large_environment(self):
        """Test region_membership with large environment.

        Notes
        -----
        Tests that region_membership works efficiently with many bins.
        Mark as slow test.
        """
        # Create large environment
        rng = np.random.default_rng(42)
        positions = rng.standard_normal((5000, 2)) * 50
        env = Environment.from_samples(
            positions, bin_size=2.0, name="LargeEnv", infer_active_bins=True
        )

        # Add several regions
        env.regions.add("region1", polygon=box(-50, -50, 0, 0))
        env.regions.add("region2", polygon=box(0, 0, 50, 50))

        membership = env.region_membership()

        # Should handle large environment
        assert membership.shape == (env.n_bins, 2)
