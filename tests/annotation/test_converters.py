"""Tests for annotation converters."""

import numpy as np
import pytest
import shapely.geometry as shp
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from neurospatial.annotation.converters import (
    env_from_boundary_region,
    shapes_to_regions,
)
from neurospatial.regions import Region
from neurospatial.transforms import Affine2D, VideoCalibration


class TestShapesToRegions:
    """Tests for shapes_to_regions function."""

    def test_basic_conversion(self):
        """Convert napari shapes to regions without calibration."""
        # Napari format: (row, col) order
        shapes_data = [
            np.array([[0, 0], [0, 100], [100, 100], [100, 0]], dtype=float),
        ]
        names = ["test_region"]
        roles = ["region"]

        regions, env_boundary, holes = shapes_to_regions(shapes_data, names, roles)

        assert len(regions) == 1
        assert "test_region" in regions
        assert env_boundary is None
        assert len(holes) == 0
        # Verify coordinates swapped: (row, col) -> (x, y)
        assert regions["test_region"].kind == "polygon"

    def test_environment_boundary_extraction(self):
        """Extract environment boundary from shapes."""
        shapes_data = [
            np.array([[0, 0], [0, 100], [100, 100], [100, 0]], dtype=float),
            np.array([[10, 10], [10, 20], [20, 20], [20, 10]], dtype=float),
        ]
        names = ["arena", "reward_zone"]
        roles = ["environment", "region"]

        regions, env_boundary, holes = shapes_to_regions(shapes_data, names, roles)

        assert len(regions) == 1
        assert "reward_zone" in regions
        assert env_boundary is not None
        assert env_boundary.name == "arena"
        assert len(holes) == 0

    def test_with_calibration(self):
        """Apply calibration transform to coordinates."""
        # Simple 2x scale transform
        scale_matrix = np.array(
            [
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        transform = Affine2D(scale_matrix)
        calibration = VideoCalibration(transform, frame_size_px=(640, 480))

        # Square in napari (row, col): corners at (0,0), (0,10), (10,10), (10,0)
        # After swap to (x, y): (0,0), (10,0), (10,10), (0,10)
        # After 2x scale: (0,0), (20,0), (20,20), (0,20)
        shapes_data = [
            np.array([[0, 0], [0, 10], [10, 10], [10, 0]], dtype=float),
        ]
        names = ["scaled_region"]
        roles = ["region"]

        regions, _, _ = shapes_to_regions(shapes_data, names, roles, calibration)

        assert len(regions) == 1
        poly = regions["scaled_region"].data
        bounds = poly.bounds  # (minx, miny, maxx, maxy)
        assert bounds[2] == pytest.approx(20.0)  # maxx
        assert bounds[3] == pytest.approx(20.0)  # maxy

    def test_skip_invalid_polygons(self):
        """Skip shapes with fewer than 3 vertices."""
        shapes_data = [
            np.array([[0, 0], [10, 10]], dtype=float),  # Line, not polygon
            np.array([[0, 0], [0, 100], [100, 100]], dtype=float),  # Valid
        ]
        names = ["line", "triangle"]
        roles = ["region", "region"]

        regions, _, _ = shapes_to_regions(shapes_data, names, roles)

        assert len(regions) == 1
        assert "triangle" in regions

    def test_metadata_populated(self):
        """Check metadata is properly set."""
        shapes_data = [
            np.array([[0, 0], [0, 100], [100, 100], [100, 0]], dtype=float),
        ]
        names = ["test"]
        roles = ["region"]

        regions, _, _ = shapes_to_regions(shapes_data, names, roles)

        metadata = regions["test"].metadata
        assert metadata["source"] == "napari_annotation"
        assert metadata["coord_system"] == "pixels"
        assert metadata["role"] == "region"

    def test_simplify_tolerance(self):
        """Simplify polygon with tolerance parameter."""
        # Create a polygon with many redundant vertices on a line
        # Square with extra points along edges
        vertices = np.array(
            [
                [0, 0],
                [0, 25],
                [0, 50],
                [0, 75],
                [0, 100],  # Left edge
                [25, 100],
                [50, 100],
                [75, 100],
                [100, 100],  # Top edge
                [100, 75],
                [100, 50],
                [100, 25],
                [100, 0],  # Right edge
                [75, 0],
                [50, 0],
                [25, 0],  # Bottom edge
            ],
            dtype=float,
        )
        shapes_data = [vertices]
        names = ["detailed"]
        roles = ["region"]

        # Without simplification
        regions_full, _, _ = shapes_to_regions(shapes_data, names, roles)
        poly_full = regions_full["detailed"].data
        n_coords_full = len(poly_full.exterior.coords)

        # With simplification (tolerance=5.0 should remove colinear points)
        regions_simple, _, _ = shapes_to_regions(
            shapes_data, names, roles, simplify_tolerance=5.0
        )
        poly_simple = regions_simple["detailed"].data
        n_coords_simple = len(poly_simple.exterior.coords)

        # Simplified should have fewer vertices (just 4 corners + closing point)
        assert n_coords_simple < n_coords_full
        assert n_coords_simple == 5  # 4 corners + closing point

    def test_warns_multiple_environment_boundaries(self):
        """Warn when multiple shapes have role='environment'."""
        # Two shapes both marked as environment
        poly1 = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=float)
        poly2 = np.array([[50, 50], [150, 50], [150, 150], [50, 150]], dtype=float)
        shapes_data = [poly1, poly2]
        names = ["boundary1", "boundary2"]
        roles = ["environment", "environment"]

        with pytest.warns(UserWarning, match="Multiple environment boundaries"):
            _, env_boundary, _ = shapes_to_regions(shapes_data, names, roles)

        # Only the last one should be returned
        assert env_boundary is not None
        assert env_boundary.name == "boundary2"

    def test_hole_extraction(self):
        """Extract holes from shapes."""
        shapes_data = [
            np.array([[0, 0], [0, 100], [100, 100], [100, 0]], dtype=float),  # boundary
            np.array([[20, 20], [20, 40], [40, 40], [40, 20]], dtype=float),  # hole
            np.array([[60, 60], [60, 80], [80, 80], [80, 60]], dtype=float),  # region
        ]
        names = ["arena", "obstacle", "reward_zone"]
        roles = ["environment", "hole", "region"]

        regions, env_boundary, holes = shapes_to_regions(shapes_data, names, roles)

        assert len(regions) == 1
        assert "reward_zone" in regions
        assert env_boundary is not None
        assert env_boundary.name == "arena"
        assert len(holes) == 1
        assert holes[0].name == "obstacle"
        assert holes[0].metadata["role"] == "hole"


class TestEnvFromBoundaryRegion:
    """Tests for env_from_boundary_region function."""

    def test_basic_environment_creation(self):
        """Create environment from polygon boundary."""
        poly = shp.Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
        boundary = Region(name="arena", kind="polygon", data=poly)

        env = env_from_boundary_region(boundary, bin_size=10.0)

        assert env._is_fitted
        assert env.n_bins > 0

    def test_rejects_non_polygon(self):
        """Raise error for non-polygon regions."""
        point = Region(name="point", kind="point", data=np.array([50.0, 50.0]))

        with pytest.raises(ValueError, match="must be polygon"):
            env_from_boundary_region(point, bin_size=10.0)

    def test_passes_kwargs(self):
        """Forward kwargs to Environment.from_polygon."""
        poly = shp.Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
        boundary = Region(name="arena", kind="polygon", data=poly)

        env = env_from_boundary_region(
            boundary,
            bin_size=10.0,
            connect_diagonal_neighbors=False,
        )

        assert env._is_fitted

    def test_environment_with_holes(self):
        """Create environment with holes subtracted from boundary."""

        # Create boundary polygon
        poly = shp.Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
        boundary = Region(name="arena", kind="polygon", data=poly)

        # Create hole polygon (smaller square inside)
        hole_poly = shp.Polygon([(40, 40), (60, 40), (60, 60), (40, 60)])
        hole = Region(name="obstacle", kind="polygon", data=hole_poly)

        # Create environment with hole
        env = env_from_boundary_region(boundary, bin_size=10.0, holes=[hole])

        assert env._is_fitted
        # Bin at center of hole should not exist (if bin resolution allows)
        # The environment should have fewer bins than without hole
        env_no_hole = env_from_boundary_region(boundary, bin_size=10.0)
        assert env.n_bins <= env_no_hole.n_bins


class TestSubtractHolesFromBoundary:
    """Tests for subtract_holes_from_boundary function."""

    def test_single_hole_subtraction(self):
        """Subtract a single hole from boundary."""
        from neurospatial.annotation.converters import subtract_holes_from_boundary

        # Create boundary polygon (100x100 square)
        boundary_poly = shp.Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
        boundary = Region(name="arena", kind="polygon", data=boundary_poly)

        # Create hole polygon (20x20 square in center)
        hole_poly = shp.Polygon([(40, 40), (60, 40), (60, 60), (40, 60)])
        hole = Region(name="obstacle", kind="polygon", data=hole_poly)

        result = subtract_holes_from_boundary(boundary, [hole])

        # Result should have reduced area
        assert result.data.area == pytest.approx(10000 - 400)  # 100*100 - 20*20
        assert result.metadata["holes_subtracted"] == 1
        # Result polygon should have an interior ring (the hole)
        assert len(result.data.interiors) == 1

    def test_multiple_holes_subtraction(self):
        """Subtract multiple holes from boundary."""
        from neurospatial.annotation.converters import subtract_holes_from_boundary

        # Create boundary polygon
        boundary_poly = shp.Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
        boundary = Region(name="arena", kind="polygon", data=boundary_poly)

        # Create two holes
        hole1_poly = shp.Polygon([(10, 10), (30, 10), (30, 30), (10, 30)])  # 20x20
        hole2_poly = shp.Polygon([(70, 70), (90, 70), (90, 90), (70, 90)])  # 20x20
        hole1 = Region(name="hole1", kind="polygon", data=hole1_poly)
        hole2 = Region(name="hole2", kind="polygon", data=hole2_poly)

        result = subtract_holes_from_boundary(boundary, [hole1, hole2])

        # Result should have area reduced by both holes
        assert result.data.area == pytest.approx(10000 - 400 - 400)
        assert result.metadata["holes_subtracted"] == 2

    def test_empty_holes_list(self):
        """Return boundary unchanged when no holes provided."""
        from neurospatial.annotation.converters import subtract_holes_from_boundary

        boundary_poly = shp.Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
        boundary = Region(name="arena", kind="polygon", data=boundary_poly)

        result = subtract_holes_from_boundary(boundary, [])

        # Should return original boundary
        assert result is boundary

    def test_non_intersecting_hole(self):
        """Hole that doesn't intersect boundary has no effect."""
        from neurospatial.annotation.converters import subtract_holes_from_boundary

        # Boundary in one area
        boundary_poly = shp.Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
        boundary = Region(name="arena", kind="polygon", data=boundary_poly)

        # Hole outside boundary
        hole_poly = shp.Polygon([(200, 200), (220, 200), (220, 220), (200, 220)])
        hole = Region(name="outside", kind="polygon", data=hole_poly)

        result = subtract_holes_from_boundary(boundary, [hole])

        # Area should be unchanged
        assert result.data.area == pytest.approx(10000)


class TestCalibrationRoundTrip:
    """Tests for calibration coordinate transforms."""

    def test_px_to_cm_round_trip(self):
        """Convert pixels to cm and back."""
        # Create 2x scale transform (pixels to cm)
        scale = 0.1  # 10 pixels = 1 cm
        scale_matrix = np.array(
            [
                [scale, 0.0, 0.0],
                [0.0, scale, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        transform = Affine2D(scale_matrix)
        calibration = VideoCalibration(transform, frame_size_px=(640, 480))

        # Original points in pixels
        pts_px = np.array([[100, 200], [300, 400]], dtype=float)

        # Convert to cm
        pts_cm = calibration.transform_px_to_cm(pts_px)
        assert pts_cm[0, 0] == pytest.approx(10.0)  # 100 * 0.1
        assert pts_cm[0, 1] == pytest.approx(20.0)  # 200 * 0.1

        # Convert back to pixels
        pts_px_back = calibration.transform_cm_to_px(pts_cm)
        np.testing.assert_allclose(pts_px_back, pts_px)

    def test_calibration_with_offset(self):
        """Test calibration with translation component."""
        # Scale 0.1 + translate origin
        matrix = np.array(
            [
                [0.1, 0.0, -10.0],  # x: 0.1 * x - 10
                [0.0, 0.1, -5.0],  # y: 0.1 * y - 5
                [0.0, 0.0, 1.0],
            ]
        )
        transform = Affine2D(matrix)
        calibration = VideoCalibration(transform, frame_size_px=(640, 480))

        # Point at (100, 50) pixels should become (0, 0) cm
        pts_px = np.array([[100.0, 50.0]])
        pts_cm = calibration.transform_px_to_cm(pts_px)
        assert pts_cm[0, 0] == pytest.approx(0.0)
        assert pts_cm[0, 1] == pytest.approx(0.0)

        # Round trip
        pts_px_back = calibration.transform_cm_to_px(pts_cm)
        np.testing.assert_allclose(pts_px_back, pts_px)

    def test_shapes_to_regions_calibration_consistency(self):
        """Ensure shapes_to_regions applies calibration correctly."""
        # Create a simple 2x scale calibration
        scale_matrix = np.array(
            [
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        transform = Affine2D(scale_matrix)
        calibration = VideoCalibration(transform, frame_size_px=(640, 480))

        # Simple unit square in napari (row, col) format
        shapes_data = [np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=float)]
        names = ["unit_square"]
        roles = ["region"]

        # Convert with calibration
        regions, _, _ = shapes_to_regions(shapes_data, names, roles, calibration)

        # After coordinate swap (row,col -> x,y) and 2x scale:
        # Original (row,col): (0,0), (0,1), (1,1), (1,0)
        # As (x,y): (0,0), (1,0), (1,1), (0,1)
        # After 2x: (0,0), (2,0), (2,2), (0,2)
        poly = regions["unit_square"].data
        bounds = poly.bounds  # (minx, miny, maxx, maxy)
        assert bounds[2] == pytest.approx(2.0)  # maxx
        assert bounds[3] == pytest.approx(2.0)  # maxy
        assert poly.area == pytest.approx(4.0)  # 2x2 = 4


class TestMultipleBoundaryStrategies:
    """Tests for multiple_boundaries parameter handling."""

    def test_multiple_boundaries_first(self):
        """Use first boundary when multiple_boundaries='first'."""
        poly1 = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=float)
        poly2 = np.array([[50, 50], [150, 50], [150, 150], [50, 150]], dtype=float)
        shapes_data = [poly1, poly2]
        names = ["boundary1", "boundary2"]
        roles = ["environment", "environment"]

        with pytest.warns(UserWarning, match="first"):
            _, env_boundary, _ = shapes_to_regions(
                shapes_data, names, roles, multiple_boundaries="first"
            )

        assert env_boundary is not None
        assert env_boundary.name == "boundary1"

    def test_multiple_boundaries_error(self):
        """Raise ValueError when multiple_boundaries='error'."""
        poly1 = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=float)
        poly2 = np.array([[50, 50], [150, 50], [150, 150], [50, 150]], dtype=float)
        shapes_data = [poly1, poly2]
        names = ["boundary1", "boundary2"]
        roles = ["environment", "environment"]

        with pytest.raises(ValueError, match="Multiple environment boundaries"):
            shapes_to_regions(shapes_data, names, roles, multiple_boundaries="error")

    def test_single_boundary_no_error(self):
        """Single boundary works with any strategy."""
        poly = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=float)
        shapes_data = [poly]
        names = ["arena"]
        roles = ["environment"]

        for strategy in ["first", "last", "error"]:
            _, env_boundary, _ = shapes_to_regions(
                shapes_data, names, roles, multiple_boundaries=strategy
            )
            assert env_boundary is not None
            assert env_boundary.name == "arena"


# --- Hypothesis-based property tests ---


# Custom strategies for generating test data
@st.composite
def polygon_vertices(draw, min_vertices=3, max_vertices=20, coord_range=(0, 1000)):
    """Generate valid polygon vertices in napari (row, col) format."""
    n_vertices = draw(st.integers(min_value=min_vertices, max_value=max_vertices))
    # Generate points and ensure they form a valid polygon
    coords = draw(
        st.lists(
            st.tuples(
                st.floats(
                    min_value=coord_range[0],
                    max_value=coord_range[1],
                    allow_nan=False,
                    allow_infinity=False,
                ),
                st.floats(
                    min_value=coord_range[0],
                    max_value=coord_range[1],
                    allow_nan=False,
                    allow_infinity=False,
                ),
            ),
            min_size=n_vertices,
            max_size=n_vertices,
        )
    )
    return np.array(coords, dtype=float)


@st.composite
def tiny_polygon_vertices(draw, size_range=(0.001, 1.0)):
    """Generate tiny polygon (very small area)."""
    # Generate a small square
    base_x = draw(st.floats(min_value=0, max_value=100, allow_nan=False))
    base_y = draw(st.floats(min_value=0, max_value=100, allow_nan=False))
    size = draw(
        st.floats(min_value=size_range[0], max_value=size_range[1], allow_nan=False)
    )
    return np.array(
        [
            [base_y, base_x],
            [base_y, base_x + size],
            [base_y + size, base_x + size],
            [base_y + size, base_x],
        ],
        dtype=float,
    )


@st.composite
def near_collinear_polygon(draw, n_extra_points=5):
    """Generate polygon with near-collinear points on edges."""
    # Base square
    base_coords = [(0, 0), (0, 100), (100, 100), (100, 0)]

    # Add near-collinear points on each edge
    all_coords = []
    for i in range(4):
        all_coords.append(base_coords[i])
        # Add points along the edge
        start = base_coords[i]
        end = base_coords[(i + 1) % 4]
        for _j in range(n_extra_points):
            t = draw(st.floats(min_value=0.1, max_value=0.9, allow_nan=False))
            # Small perturbation perpendicular to edge
            perturb = draw(st.floats(min_value=-0.01, max_value=0.01, allow_nan=False))
            x = start[0] + t * (end[0] - start[0]) + perturb
            y = start[1] + t * (end[1] - start[1]) + perturb
            all_coords.append((x, y))

    # Convert to napari (row, col) format
    return np.array([(y, x) for x, y in all_coords], dtype=float)


@st.composite
def calibration_transform(draw, scale_range=(0.01, 10.0)):
    """Generate a valid calibration transform."""
    scale = draw(
        st.floats(min_value=scale_range[0], max_value=scale_range[1], allow_nan=False)
    )
    offset_x = draw(st.floats(min_value=-100, max_value=100, allow_nan=False))
    offset_y = draw(st.floats(min_value=-100, max_value=100, allow_nan=False))
    matrix = np.array(
        [
            [scale, 0.0, offset_x],
            [0.0, scale, offset_y],
            [0.0, 0.0, 1.0],
        ]
    )
    return VideoCalibration(Affine2D(matrix), frame_size_px=(640, 480))


class TestHypothesisShapesToRegions:
    """Property-based tests for shapes_to_regions using Hypothesis."""

    @given(vertices=polygon_vertices())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.filter_too_much])
    def test_valid_polygon_creates_region(self, vertices):
        """Any valid polygon should create a region without error."""
        # Create shapely polygon to check validity
        poly = shp.Polygon(vertices[:, ::-1])  # Convert to (x, y)
        assume(poly.is_valid and poly.area > 0)

        shapes_data = [vertices]
        names = ["test"]
        roles = ["region"]

        regions, env_boundary, holes = shapes_to_regions(shapes_data, names, roles)

        assert "test" in regions
        assert env_boundary is None
        assert len(holes) == 0

    @given(vertices=tiny_polygon_vertices())
    @settings(max_examples=50)
    def test_tiny_polygons_preserved(self, vertices):
        """Very small polygons should still create valid regions."""
        shapes_data = [vertices]
        names = ["tiny"]
        roles = ["region"]

        regions, _, _ = shapes_to_regions(shapes_data, names, roles)

        assert "tiny" in regions
        assert regions["tiny"].data.area > 0

    @given(vertices=near_collinear_polygon())
    @settings(
        max_examples=30,
        suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow],
    )
    def test_near_collinear_points_handled(self, vertices):
        """Polygons with near-collinear points should be processed."""
        # Create shapely polygon to check validity
        poly = shp.Polygon(vertices[:, ::-1])
        assume(poly.is_valid and poly.area > 0)

        shapes_data = [vertices]
        names = ["collinear"]
        roles = ["region"]

        regions, _, _ = shapes_to_regions(shapes_data, names, roles)

        assert "collinear" in regions

    @given(vertices=near_collinear_polygon())
    @settings(
        max_examples=30,
        suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow],
    )
    def test_simplification_reduces_collinear_points(self, vertices):
        """Simplification should reduce near-collinear points."""
        poly = shp.Polygon(vertices[:, ::-1])
        assume(poly.is_valid and poly.area > 0)

        shapes_data = [vertices]
        names = ["test"]
        roles = ["region"]

        # Without simplification
        regions_full, _, _ = shapes_to_regions(shapes_data, names, roles)
        n_coords_full = len(regions_full["test"].data.exterior.coords)

        # With simplification
        regions_simple, _, _ = shapes_to_regions(
            shapes_data, names, roles, simplify_tolerance=1.0
        )
        n_coords_simple = len(regions_simple["test"].data.exterior.coords)

        # Simplified should have fewer or equal vertices
        assert n_coords_simple <= n_coords_full

    @given(calibration=calibration_transform())
    @settings(max_examples=30)
    def test_calibration_preserves_shape(self, calibration):
        """Calibration should scale area proportionally."""
        # Simple unit square
        vertices = np.array([[0, 0], [0, 10], [10, 10], [10, 0]], dtype=float)
        shapes_data = [vertices]
        names = ["square"]
        roles = ["region"]

        # Without calibration
        regions_px, _, _ = shapes_to_regions(shapes_data, names, roles)
        area_px = regions_px["square"].data.area

        # With calibration
        regions_cal, _, _ = shapes_to_regions(
            shapes_data, names, roles, calibration=calibration
        )
        area_cal = regions_cal["square"].data.area

        # Get scale factor from calibration's internal transform (Affine2D uses A attribute)
        scale = calibration.transform_px_to_cm.A[0, 0]  # Uniform scale assumed
        expected_area = area_px * (scale**2)

        assert area_cal == pytest.approx(expected_area, rel=1e-6)


class TestHypothesisHoles:
    """Property-based tests for hole handling."""

    @given(
        hole_scale=st.floats(min_value=0.1, max_value=0.5, allow_nan=False),
        hole_offset_x=st.floats(min_value=0.2, max_value=0.5, allow_nan=False),
        hole_offset_y=st.floats(min_value=0.2, max_value=0.5, allow_nan=False),
    )
    @settings(max_examples=30)
    def test_hole_inside_boundary(self, hole_scale, hole_offset_x, hole_offset_y):
        """Holes inside boundary reduce area correctly."""
        # Boundary: 100x100 square at origin
        boundary = np.array(
            [[0, 0], [0, 100], [100, 100], [100, 0]],
            dtype=float,
        )

        # Hole: smaller square inside boundary
        hole_size = 100 * hole_scale
        hx = 100 * hole_offset_x
        hy = 100 * hole_offset_y
        # Ensure hole fits inside boundary
        assume(hx + hole_size < 100 and hy + hole_size < 100)

        hole = np.array(
            [
                [hy, hx],
                [hy, hx + hole_size],
                [hy + hole_size, hx + hole_size],
                [hy + hole_size, hx],
            ],
            dtype=float,
        )

        shapes_data = [boundary, hole]
        names = ["arena", "obstacle"]
        roles = ["environment", "hole"]

        _, env_boundary, holes = shapes_to_regions(shapes_data, names, roles)

        assert env_boundary is not None
        assert len(holes) == 1

        # Subtract holes and verify area
        from neurospatial.annotation.converters import subtract_holes_from_boundary

        result = subtract_holes_from_boundary(env_boundary, holes)
        expected_area = 100 * 100 - hole_size * hole_size
        assert result.data.area == pytest.approx(expected_area, rel=1e-6)

    @given(n_holes=st.integers(min_value=1, max_value=5))
    @settings(max_examples=20)
    def test_multiple_non_overlapping_holes(self, n_holes):
        """Multiple non-overlapping holes are all extracted."""
        # Boundary
        boundary = np.array(
            [[0, 0], [0, 1000], [1000, 1000], [1000, 0]],
            dtype=float,
        )

        shapes_data = [boundary]
        names = ["arena"]
        roles = ["environment"]

        # Add non-overlapping holes in a grid pattern
        hole_size = 50
        for i in range(n_holes):
            # Place holes in a row, well-separated
            x_offset = 100 + i * 150
            y_offset = 100
            hole = np.array(
                [
                    [y_offset, x_offset],
                    [y_offset, x_offset + hole_size],
                    [y_offset + hole_size, x_offset + hole_size],
                    [y_offset + hole_size, x_offset],
                ],
                dtype=float,
            )
            shapes_data.append(hole)
            names.append(f"hole_{i}")
            roles.append("hole")

        _, _env_boundary, holes = shapes_to_regions(shapes_data, names, roles)

        assert len(holes) == n_holes


class TestHypothesisCalibrationRoundTrip:
    """Property-based tests for calibration coordinate transforms."""

    @given(
        scale=st.floats(min_value=0.01, max_value=10.0, allow_nan=False),
        offset_x=st.floats(min_value=-100, max_value=100, allow_nan=False),
        offset_y=st.floats(min_value=-100, max_value=100, allow_nan=False),
        pts=st.lists(
            st.tuples(
                # Use reasonable coordinate values (not subnormal floats)
                st.floats(min_value=1.0, max_value=500, allow_nan=False),
                st.floats(min_value=1.0, max_value=500, allow_nan=False),
            ),
            min_size=1,
            max_size=10,
        ),
    )
    @settings(max_examples=50)
    def test_round_trip_preserves_coordinates(self, scale, offset_x, offset_y, pts):
        """Converting px→cm→px should preserve coordinates."""
        matrix = np.array(
            [
                [scale, 0.0, offset_x],
                [0.0, scale, offset_y],
                [0.0, 0.0, 1.0],
            ]
        )
        calibration = VideoCalibration(Affine2D(matrix), frame_size_px=(640, 480))

        pts_px = np.array(pts, dtype=float)

        # Round trip
        pts_cm = calibration.transform_px_to_cm(pts_px)
        pts_back = calibration.transform_cm_to_px(pts_cm)

        # Use both relative and absolute tolerance for numerical stability
        np.testing.assert_allclose(pts_back, pts_px, rtol=1e-10, atol=1e-10)
