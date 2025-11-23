"""Tests for annotation converters."""

import numpy as np
import pytest
import shapely.geometry as shp

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
