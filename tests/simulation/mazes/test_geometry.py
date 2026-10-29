"""Tests for maze geometry helper functions."""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import MultiPolygon, Point, Polygon

from neurospatial.simulation.mazes._base import MazeDims
from neurospatial.simulation.mazes._geometry import (
    make_buffered_line,
    make_circular_arena,
    make_corridor_polygon,
    single_floor,
    union_polygons,
)


class TestMakeCorridorPolygon:
    """Tests for make_corridor_polygon function."""

    def test_horizontal_corridor(self):
        """Create a horizontal corridor."""
        poly = make_corridor_polygon(start=(0, 0), end=(100, 0), width=10)
        assert isinstance(poly, Polygon)
        assert poly.is_valid

        # Should be roughly 100 x 10 = 1000 area
        assert 900 < poly.area < 1100  # Allow some tolerance

    def test_vertical_corridor(self):
        """Create a vertical corridor."""
        poly = make_corridor_polygon(start=(0, 0), end=(0, 100), width=10)
        assert isinstance(poly, Polygon)
        assert poly.is_valid
        assert 900 < poly.area < 1100

    def test_diagonal_corridor(self):
        """Create a diagonal corridor."""
        poly = make_corridor_polygon(start=(0, 0), end=(100, 100), width=10)
        assert isinstance(poly, Polygon)
        assert poly.is_valid

        # Length is sqrt(100^2 + 100^2) = ~141.4
        # Area should be ~141.4 * 10 = ~1414
        assert 1300 < poly.area < 1600

    def test_corridor_contains_endpoints(self):
        """Corridor should contain its endpoints (on boundary)."""
        poly = make_corridor_polygon(start=(0, 0), end=(100, 0), width=10)
        # Use covers() instead of contains() - covers() is True for boundary points
        assert poly.covers(Point(0, 0))
        assert poly.covers(Point(100, 0))

    def test_width_is_total_width(self):
        """Width parameter is total width, not half-width."""
        poly = make_corridor_polygon(start=(0, 0), end=(100, 0), width=10)
        # Check vertical extent
        _minx, miny, _maxx, maxy = poly.bounds
        assert (maxy - miny) == pytest.approx(10, abs=0.5)


class TestMakeBufferedLine:
    """Tests for make_buffered_line function."""

    def test_creates_polygon(self):
        """Creates a valid polygon from a line segment."""
        poly = make_buffered_line(start=(0, 0), end=(100, 0), width=10)
        assert isinstance(poly, Polygon)
        assert poly.is_valid

    def test_buffered_line_area(self):
        """Buffered line has appropriate area."""
        poly = make_buffered_line(start=(0, 0), end=(100, 0), width=10)
        # Buffer creates rounded ends, so area > simple rectangle
        # Rectangle would be 100 * 10 = 1000
        # With rounded ends, should be larger
        assert poly.area > 1000

    def test_contains_line(self):
        """Buffered polygon contains the original line."""
        poly = make_buffered_line(start=(0, 0), end=(100, 0), width=10)
        assert poly.contains(Point(0, 0))
        assert poly.contains(Point(50, 0))
        assert poly.contains(Point(100, 0))


class TestMakeCircularArena:
    """Tests for make_circular_arena function."""

    def test_creates_polygon(self):
        """Creates a valid circular polygon."""
        poly = make_circular_arena(center=(0, 0), radius=50)
        assert isinstance(poly, Polygon)
        assert poly.is_valid

    def test_area_is_pi_r_squared(self):
        """Circle area should be approximately pi * r^2."""
        radius = 50
        poly = make_circular_arena(center=(0, 0), radius=radius)
        expected_area = np.pi * radius**2
        assert poly.area == pytest.approx(expected_area, rel=0.01)  # 1% tolerance

    def test_centered_at_origin(self):
        """Circle centered at origin has correct bounds."""
        radius = 50
        poly = make_circular_arena(center=(0, 0), radius=radius)
        minx, miny, maxx, maxy = poly.bounds
        assert minx == pytest.approx(-radius, abs=1)
        assert maxx == pytest.approx(radius, abs=1)
        assert miny == pytest.approx(-radius, abs=1)
        assert maxy == pytest.approx(radius, abs=1)

    def test_centered_at_offset(self):
        """Circle centered at offset has correct bounds."""
        center = (100, 50)
        radius = 25
        poly = make_circular_arena(center=center, radius=radius)
        minx, _miny, maxx, _maxy = poly.bounds
        assert minx == pytest.approx(center[0] - radius, abs=1)
        assert maxx == pytest.approx(center[0] + radius, abs=1)

    def test_contains_center(self):
        """Circle contains its center point."""
        center = (100, 50)
        poly = make_circular_arena(center=center, radius=25)
        assert poly.contains(Point(center))


class TestUnionPolygons:
    """Tests for union_polygons function."""

    def test_union_two_polygons(self):
        """Union of two overlapping polygons."""
        poly1 = make_corridor_polygon((0, 0), (100, 0), width=10)
        poly2 = make_corridor_polygon((50, 0), (50, 100), width=10)
        result = union_polygons([poly1, poly2])
        assert isinstance(result, Polygon)
        assert result.is_valid

    def test_union_non_overlapping_is_a_multipolygon(self):
        """Pieces that do not touch stay separate."""
        poly1 = make_corridor_polygon((0, 0), (50, 0), width=10)
        poly2 = make_corridor_polygon((100, 0), (150, 0), width=10)
        result = union_polygons([poly1, poly2])
        assert isinstance(result, MultiPolygon)
        assert result.is_valid

    def test_union_single_polygon(self):
        """Union of a single polygon returns valid polygon."""
        poly = make_corridor_polygon((0, 0), (100, 0), width=10)
        result = union_polygons([poly])
        assert result.is_valid
        # Area should be approximately same
        assert result.area == pytest.approx(poly.area, rel=0.01)

    def test_union_empty_list_raises(self):
        """Union of empty list raises ValueError."""
        with pytest.raises(ValueError, match="at least one polygon"):
            union_polygons([])


class TestSingleFloor:
    """A maze floor must be one connected piece."""

    def test_returns_a_connected_floor(self):
        floor = union_polygons(
            [
                make_corridor_polygon((0, 0), (100, 0), width=10),
                make_corridor_polygon((50, 0), (50, 100), width=10),
            ]
        )
        assert single_floor(floor, MazeDims()) is floor

    def test_refuses_a_floor_in_pieces_naming_the_dimensions(self):
        pieces = union_polygons(
            [
                make_corridor_polygon((0, 0), (50, 0), width=10),
                make_corridor_polygon((100, 0), (150, 0), width=10),
            ]
        )
        with pytest.raises(ValueError, match=r"MazeDims\(\) builds a maze floor in 2"):
            single_floor(pieces, MazeDims())
