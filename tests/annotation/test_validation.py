"""Tests for annotation validation module."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from shapely import Polygon

from neurospatial.annotation.validation import (
    DEFAULT_MIN_AREA_THRESHOLD,
    DEFAULT_OVERLAP_THRESHOLD,
    validate_annotations,
    validate_polygon_geometry,
    validate_region_overlap,
    validate_region_within_boundary,
)
from neurospatial.regions import Region, Regions

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def valid_square_polygon():
    """Create a valid square polygon."""
    return Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])


@pytest.fixture
def self_intersecting_polygon():
    """Create a self-intersecting bowtie polygon."""
    # This creates a figure-8 shape that crosses itself
    return Polygon([(0, 0), (10, 10), (10, 0), (0, 10)])


@pytest.fixture
def tiny_polygon():
    """Create a very small polygon."""
    return Polygon([(0, 0), (1e-8, 0), (1e-8, 1e-8), (0, 1e-8)])


@pytest.fixture
def boundary_region():
    """Create a boundary region covering [0, 100] x [0, 100]."""
    return Region(
        name="arena",
        kind="polygon",
        data=Polygon([(0, 0), (100, 0), (100, 100), (0, 100)]),
    )


@pytest.fixture
def region_inside_boundary():
    """Create a region fully inside the boundary."""
    return Region(
        name="goal",
        kind="polygon",
        data=Polygon([(10, 10), (30, 10), (30, 30), (10, 30)]),
    )


@pytest.fixture
def region_outside_boundary():
    """Create a region mostly outside the boundary."""
    return Region(
        name="outside",
        kind="polygon",
        data=Polygon([(90, 90), (150, 90), (150, 150), (90, 150)]),
    )


@pytest.fixture
def overlapping_regions():
    """Create two heavily overlapping regions."""
    r1 = Region(
        name="region1",
        kind="polygon",
        data=Polygon([(0, 0), (50, 0), (50, 50), (0, 50)]),
    )
    r2 = Region(
        name="region2",
        kind="polygon",
        data=Polygon([(10, 10), (60, 10), (60, 60), (10, 60)]),
    )
    return r1, r2


# =============================================================================
# Tests for validate_polygon_geometry
# =============================================================================


class TestValidatePolygonGeometry:
    """Tests for polygon geometry validation."""

    def test_valid_polygon_no_issues(self, valid_square_polygon):
        """Valid polygon should produce no issues."""
        issues = validate_polygon_geometry(valid_square_polygon, "test")
        assert len(issues) == 0

    def test_self_intersecting_polygon_warns(self, self_intersecting_polygon):
        """Self-intersecting polygon should warn."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            issues = validate_polygon_geometry(self_intersecting_polygon, "bowtie")

            assert len(issues) > 0
            assert "self-intersect" in issues[0].lower()
            assert len(w) >= 1
            assert "self-intersecting" in str(w[0].message).lower()

    def test_tiny_polygon_warns(self, tiny_polygon):
        """Tiny polygon should warn about small area."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            issues = validate_polygon_geometry(
                tiny_polygon, "tiny", min_area=DEFAULT_MIN_AREA_THRESHOLD
            )

            assert len(issues) > 0
            assert "small area" in issues[0].lower()
            assert len(w) >= 1

    def test_empty_polygon_warns(self):
        """Empty polygon should warn."""
        empty_poly = Polygon()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            issues = validate_polygon_geometry(empty_poly, "empty")

            assert len(issues) > 0
            assert any("empty" in issue.lower() for issue in issues)
            assert len(w) >= 1

    def test_validation_can_be_disabled(self, self_intersecting_polygon):
        """Validation warnings can be disabled."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Call validation but don't need to check issues
            validate_polygon_geometry(
                self_intersecting_polygon,
                "bowtie",
                warn_self_intersecting=False,
                warn_small_area=False,
            )

            # Issues still detected but no warnings
            # (issues list captures for return, warn_* controls warnings)
            assert len(w) == 0


# =============================================================================
# Tests for validate_region_within_boundary
# =============================================================================


class TestValidateRegionWithinBoundary:
    """Tests for region containment validation."""

    def test_region_inside_boundary_no_issues(
        self, region_inside_boundary, boundary_region
    ):
        """Region inside boundary should have no issues."""
        issues = validate_region_within_boundary(
            region_inside_boundary, boundary_region
        )
        assert len(issues) == 0

    def test_region_outside_boundary_warns(
        self, region_outside_boundary, boundary_region
    ):
        """Region outside boundary should warn."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            issues = validate_region_within_boundary(
                region_outside_boundary, boundary_region
            )

            assert len(issues) > 0
            assert "outside" in issues[0].lower()
            assert len(w) >= 1

    def test_tolerance_allows_small_overlap(self, boundary_region):
        """Small overlap within tolerance should not warn."""
        # Region that extends 1% outside boundary
        slightly_outside = Region(
            name="slightly_outside",
            kind="polygon",
            data=Polygon([(95, 95), (101, 95), (101, 101), (95, 101)]),
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Call validation - don't need to check returned issues
            validate_region_within_boundary(
                slightly_outside, boundary_region, tolerance=0.5
            )

            # With 50% tolerance, this should not trigger (region is ~66% inside)
            assert len(w) == 0

    def test_point_regions_skipped(self, boundary_region):
        """Point regions should be skipped (only polygon checked)."""
        point_region = Region(name="point", kind="point", data=np.array([50, 50]))
        issues = validate_region_within_boundary(point_region, boundary_region)
        assert len(issues) == 0


# =============================================================================
# Tests for validate_region_overlap
# =============================================================================


class TestValidateRegionOverlap:
    """Tests for region overlap validation."""

    def test_non_overlapping_regions_no_issues(self):
        """Non-overlapping regions should have no issues."""
        r1 = Region(
            name="r1",
            kind="polygon",
            data=Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
        )
        r2 = Region(
            name="r2",
            kind="polygon",
            data=Polygon([(20, 20), (30, 20), (30, 30), (20, 30)]),
        )
        regions = Regions([r1, r2])

        issues = validate_region_overlap(regions)
        assert len(issues) == 0

    def test_heavily_overlapping_regions_warns(self, overlapping_regions):
        """Heavily overlapping regions should warn."""
        r1, r2 = overlapping_regions
        regions = Regions([r1, r2])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            issues = validate_region_overlap(
                regions, overlap_threshold=DEFAULT_OVERLAP_THRESHOLD
            )

            assert len(issues) > 0
            assert "overlap" in issues[0].lower()
            assert len(w) >= 1

    def test_threshold_controls_sensitivity(self, overlapping_regions):
        """Higher threshold should not warn for moderate overlap."""
        r1, r2 = overlapping_regions
        regions = Regions([r1, r2])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # With threshold at 95%, moderate overlap should not warn
            validate_region_overlap(regions, overlap_threshold=0.95)

            assert len(w) == 0

    def test_single_region_no_issues(self, region_inside_boundary):
        """Single region should have no overlap issues."""
        regions = Regions([region_inside_boundary])
        issues = validate_region_overlap(regions)
        assert len(issues) == 0


# =============================================================================
# Tests for validate_annotations (comprehensive validation)
# =============================================================================


class TestValidateAnnotations:
    """Tests for comprehensive annotation validation."""

    def test_valid_annotations_no_issues(self, boundary_region, region_inside_boundary):
        """Valid annotations should have no issues."""
        regions = Regions([region_inside_boundary])
        issues = validate_annotations(regions, boundary_region)
        assert len(issues) == 0

    def test_multiple_issues_detected(
        self, boundary_region, region_outside_boundary, overlapping_regions
    ):
        """Multiple issues should all be detected."""
        r1, r2 = overlapping_regions
        regions = Regions([r1, r2, region_outside_boundary])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            issues = validate_annotations(regions, boundary_region)

            # Should detect overlap and outside boundary issues
            assert len(issues) >= 2
            assert len(w) >= 2

    def test_no_boundary_only_checks_overlap(self, overlapping_regions):
        """Without boundary, only overlap is checked."""
        r1, r2 = overlapping_regions
        regions = Regions([r1, r2])

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            issues = validate_annotations(regions, boundary=None)

            # Should only have overlap issue
            assert len(issues) >= 1
            assert "overlap" in issues[0].lower()


# =============================================================================
# Property-Based Tests for Validation
# =============================================================================


@st.composite
def valid_polygon_strategy(draw, min_size: float = 1.0, max_size: float = 100.0):
    """Generate valid (non-self-intersecting) polygons."""
    # Generate 4-8 vertices for a simple polygon
    n_vertices = draw(st.integers(min_value=4, max_value=8))

    # Generate angles for vertices (sorted to avoid self-intersection)
    angles = sorted(
        draw(
            st.lists(
                st.floats(min_value=0, max_value=2 * np.pi - 0.1),
                min_size=n_vertices,
                max_size=n_vertices,
                unique=True,
            )
        )
    )

    # Generate radius for each vertex
    radii = draw(
        st.lists(
            st.floats(min_value=min_size, max_value=max_size),
            min_size=n_vertices,
            max_size=n_vertices,
        )
    )

    # Convert polar to Cartesian
    vertices = [
        (r * np.cos(a), r * np.sin(a)) for r, a in zip(radii, angles, strict=True)
    ]

    return Polygon(vertices)


class TestValidationProperties:
    """Property-based tests for validation invariants."""

    @given(st.floats(min_value=1.0, max_value=1000.0))
    @settings(max_examples=50, deadline=5000)
    def test_valid_square_always_passes(self, size: float):
        """Property: Regular squares should always pass validation."""
        poly = Polygon([(0, 0), (size, 0), (size, size), (0, size)])

        issues = validate_polygon_geometry(
            poly, "square", warn_self_intersecting=False, warn_small_area=False
        )

        # Valid squares should have no self-intersection issues
        assert not any("self-intersect" in i.lower() for i in issues)

    @given(
        st.floats(min_value=10.0, max_value=100.0),
        st.floats(min_value=0.1, max_value=0.5),  # Fraction to extend outside
    )
    @settings(max_examples=50, deadline=5000)
    def test_containment_fraction_correct(self, boundary_size: float, fraction: float):
        """Property: Regions extending outside boundary are detected."""
        # Create boundary
        boundary = Region(
            name="boundary",
            kind="polygon",
            data=Polygon(
                [
                    (0, 0),
                    (boundary_size, 0),
                    (boundary_size, boundary_size),
                    (0, boundary_size),
                ]
            ),
        )

        # Create region that clearly extends past boundary
        # Region extends from boundary_size * 0.8 to boundary_size * (1 + fraction)
        # This ensures part is inside and part is outside
        region_start = boundary_size * 0.8
        region_end = boundary_size * (1 + fraction)

        region = Region(
            name="region",
            kind="polygon",
            data=Polygon(
                [
                    (region_start, 0),
                    (region_end, 0),
                    (region_end, boundary_size * 0.5),
                    (region_start, boundary_size * 0.5),
                ]
            ),
        )

        # Validate should catch region extending outside
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            issues = validate_region_within_boundary(
                region, boundary, tolerance=0.0, warn_outside=False
            )

            # Region definitely extends outside, so issues should be present
            assert len(issues) > 0

    @given(st.integers(min_value=2, max_value=10))
    @settings(max_examples=30, deadline=5000)
    def test_no_false_positives_for_non_overlapping_grid(self, n_regions: int):
        """Property: Non-overlapping grid regions should never warn about overlap."""
        # Create non-overlapping grid of regions
        regions_list = []
        for i in range(n_regions):
            region = Region(
                name=f"region_{i}",
                kind="polygon",
                data=Polygon(
                    [(i * 20, 0), ((i + 1) * 20, 0), ((i + 1) * 20, 10), (i * 20, 10)]
                ),
            )
            regions_list.append(region)

        regions = Regions(regions_list)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            issues = validate_region_overlap(regions)

            # Non-overlapping regions should have no issues
            assert len(issues) == 0
            assert len(w) == 0


# =============================================================================
# Edge Case Tests for Complex Geometries
# =============================================================================


class TestPolygonWithHoles:
    """Tests for polygons with interior holes."""

    def test_polygon_with_hole_validates_successfully(self):
        """Polygon with interior hole should validate without issues."""
        # Create a donut shape - outer ring with interior hole
        outer = [(0, 0), (100, 0), (100, 100), (0, 100)]
        hole = [(40, 40), (60, 40), (60, 60), (40, 60)]
        poly_with_hole = Polygon(outer, [hole])

        # Should be valid and have positive area
        assert poly_with_hole.is_valid
        assert poly_with_hole.area > 0

        issues = validate_polygon_geometry(poly_with_hole, "donut")
        assert len(issues) == 0

    def test_polygon_with_multiple_holes(self):
        """Polygon with multiple interior holes should validate."""
        outer = [(0, 0), (200, 0), (200, 100), (0, 100)]
        hole1 = [(20, 20), (40, 20), (40, 80), (20, 80)]
        hole2 = [(60, 20), (80, 20), (80, 80), (60, 80)]
        hole3 = [(100, 20), (120, 20), (120, 80), (100, 80)]
        poly = Polygon(outer, [hole1, hole2, hole3])

        assert poly.is_valid
        issues = validate_polygon_geometry(poly, "swiss_cheese")
        assert len(issues) == 0

    def test_boundary_with_hole_validates(self):
        """Environment boundary with hole should validate."""
        outer = [(0, 0), (100, 0), (100, 100), (0, 100)]
        obstacle = [(40, 40), (60, 40), (60, 60), (40, 60)]
        boundary_poly = Polygon(outer, [obstacle])

        boundary = Region(
            name="arena_with_obstacle",
            kind="polygon",
            data=boundary_poly,
        )

        from neurospatial.annotation.validation import validate_environment_boundary

        issues = validate_environment_boundary(boundary)
        assert len(issues) == 0

    def test_region_containment_with_boundary_hole(self):
        """Region inside boundary hole should be detected as outside."""
        # Boundary with a hole in the middle
        outer = [(0, 0), (100, 0), (100, 100), (0, 100)]
        hole = [(40, 40), (60, 40), (60, 60), (40, 60)]
        boundary_poly = Polygon(outer, [hole])

        boundary = Region(
            name="arena",
            kind="polygon",
            data=boundary_poly,
        )

        # Region inside the hole (not actually inside boundary)
        region_in_hole = Region(
            name="inside_hole",
            kind="polygon",
            data=Polygon([(45, 45), (55, 45), (55, 55), (45, 55)]),
        )

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            issues = validate_region_within_boundary(
                region_in_hole, boundary, tolerance=0.0
            )

        # Region is inside the hole, so it's "outside" the valid boundary
        assert len(issues) > 0
        assert "outside" in issues[0].lower()


class TestMultiPolygonHandling:
    """Tests for MultiPolygon scenarios (e.g., from buffer(0) fix)."""

    def test_self_intersecting_becomes_multipolygon(self):
        """Self-intersecting polygon may become MultiPolygon after buffer(0)."""
        # Figure-8 bowtie that becomes two separate polygons
        bowtie = Polygon([(0, 0), (10, 10), (10, 0), (0, 10)])

        # Verify it's self-intersecting
        assert not bowtie.is_valid

        # buffer(0) is the standard fix - may return MultiPolygon
        fixed = bowtie.buffer(0)

        # The fix should produce a valid geometry
        assert fixed.is_valid

        # Note: Depending on Shapely version, this might be MultiPolygon or Polygon
        # The important thing is it's valid and has area
        assert fixed.area > 0

    def test_validation_warns_about_self_intersecting(self, self_intersecting_polygon):
        """Validation should warn about self-intersecting polygons."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            issues = validate_polygon_geometry(self_intersecting_polygon, "bowtie")

            assert len(issues) > 0
            assert "self-intersect" in issues[0].lower()
            assert len(w) >= 1

    def test_region_overlap_handles_complex_geometries(self):
        """Overlap detection should handle polygons with holes."""
        # Region with hole
        outer1 = [(0, 0), (50, 0), (50, 50), (0, 50)]
        hole1 = [(20, 20), (30, 20), (30, 30), (20, 30)]
        r1 = Region(
            name="donut1",
            kind="polygon",
            data=Polygon(outer1, [hole1]),
        )

        # Simple region that overlaps with the outer ring but not the hole
        r2 = Region(
            name="simple",
            kind="polygon",
            data=Polygon([(10, 10), (40, 10), (40, 15), (10, 15)]),
        )

        regions = Regions([r1, r2])

        # Should complete without error
        issues = validate_region_overlap(regions, warn_overlap=False)
        # The overlap is partial, may or may not trigger depending on threshold
        assert isinstance(issues, list)


class TestEdgeCaseGeometries:
    """Tests for unusual but valid geometries."""

    def test_very_thin_polygon(self):
        """Very thin (but valid) polygon should pass geometry validation."""
        thin = Polygon([(0, 0), (100, 0), (100, 0.1), (0, 0.1)])
        assert thin.is_valid
        assert thin.area > 0

        # Should pass with sufficiently low min_area
        issues = validate_polygon_geometry(thin, "thin", min_area=1e-10)
        assert len(issues) == 0

    def test_concave_polygon(self):
        """Concave (non-convex) polygon should validate."""
        # L-shaped polygon
        concave = Polygon([(0, 0), (50, 0), (50, 25), (25, 25), (25, 50), (0, 50)])
        assert concave.is_valid

        issues = validate_polygon_geometry(concave, "L_shape")
        assert len(issues) == 0

    def test_polygon_with_many_vertices(self):
        """Polygon with many vertices should validate efficiently."""
        # Create a star-like polygon with many points
        n_points = 100
        angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        radii = [50 if i % 2 == 0 else 30 for i in range(n_points)]
        vertices = [
            (r * np.cos(a), r * np.sin(a)) for r, a in zip(radii, angles, strict=True)
        ]
        star = Polygon(vertices)

        assert star.is_valid
        issues = validate_polygon_geometry(star, "star")
        assert len(issues) == 0

    def test_overlapping_regions_different_sizes(self):
        """Overlap detection correctly handles regions of different sizes."""
        # Large region
        large = Region(
            name="large",
            kind="polygon",
            data=Polygon([(0, 0), (100, 0), (100, 100), (0, 100)]),
        )

        # Small region completely inside large
        small = Region(
            name="small",
            kind="polygon",
            data=Polygon([(40, 40), (60, 40), (60, 60), (40, 60)]),
        )

        regions = Regions([large, small])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Default threshold is 0.5, small is 100% inside large
            issues = validate_region_overlap(regions, overlap_threshold=0.5)

            # Small is 100% overlapping with large, should warn
            assert len(issues) > 0
            assert len(w) >= 1
