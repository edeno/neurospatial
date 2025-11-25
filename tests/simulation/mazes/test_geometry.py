"""Tests for maze geometry helper functions."""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Point, Polygon

from neurospatial.simulation.mazes._geometry import (
    make_buffered_line,
    make_circular_arena,
    make_corridor_polygon,
    make_star_graph,
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

    def test_union_non_overlapping(self):
        """Union of non-overlapping polygons creates valid geometry."""
        poly1 = make_corridor_polygon((0, 0), (50, 0), width=10)
        poly2 = make_corridor_polygon((100, 0), (150, 0), width=10)
        result = union_polygons([poly1, poly2])
        # May return MultiPolygon for non-overlapping, but should be valid
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


class TestMakeStarGraph:
    """Tests for make_star_graph function."""

    def test_creates_graph(self):
        """Creates a networkx graph."""
        import networkx as nx

        graph = make_star_graph(
            center=(0, 0),
            arm_endpoints=[(50, 0), (-50, 0), (0, 50), (0, -50)],
        )
        assert isinstance(graph, nx.Graph)

    def test_star_with_4_arms(self):
        """Star graph with 4 arms has correct structure."""
        graph = make_star_graph(
            center=(0, 0),
            arm_endpoints=[(50, 0), (-50, 0), (0, 50), (0, -50)],
        )
        # Should have center + 4 endpoints = 5 nodes
        assert graph.number_of_nodes() == 5
        # Should have 4 edges (center to each endpoint)
        assert graph.number_of_edges() == 4

    def test_star_with_8_arms(self):
        """Star graph with 8 arms (like radial arm maze)."""
        # Generate 8 arm endpoints at equal angles
        n_arms = 8
        arm_length = 50
        angles = np.linspace(0, 2 * np.pi, n_arms, endpoint=False)
        arm_endpoints = [
            (arm_length * np.cos(a), arm_length * np.sin(a)) for a in angles
        ]

        graph = make_star_graph(center=(0, 0), arm_endpoints=arm_endpoints)
        assert graph.number_of_nodes() == n_arms + 1  # center + endpoints
        assert graph.number_of_edges() == n_arms

    def test_nodes_have_positions(self):
        """Graph nodes have 'pos' attribute."""
        graph = make_star_graph(
            center=(0, 0),
            arm_endpoints=[(50, 0), (0, 50)],
        )
        for node in graph.nodes():
            assert "pos" in graph.nodes[node]
            pos = graph.nodes[node]["pos"]
            assert len(pos) == 2  # 2D position

    def test_edges_have_distance(self):
        """Graph edges have 'distance' attribute."""
        graph = make_star_graph(
            center=(0, 0),
            arm_endpoints=[(50, 0), (0, 50)],
        )
        for u, v in graph.edges():
            assert "distance" in graph[u][v]
            assert graph[u][v]["distance"] > 0

    def test_center_node_is_labeled(self):
        """Graph has a node labeled 'center'."""
        graph = make_star_graph(
            center=(0, 0),
            arm_endpoints=[(50, 0), (0, 50)],
        )
        assert "center" in graph.nodes()

    def test_spacing_adds_intermediate_nodes(self):
        """Providing spacing adds intermediate nodes along arms."""
        # Without spacing
        graph_simple = make_star_graph(
            center=(0, 0),
            arm_endpoints=[(100, 0)],
            spacing=None,
        )
        n_simple = graph_simple.number_of_nodes()

        # With spacing (should add intermediate nodes)
        graph_spaced = make_star_graph(
            center=(0, 0),
            arm_endpoints=[(100, 0)],
            spacing=25.0,  # Add nodes every 25 units
        )
        n_spaced = graph_spaced.number_of_nodes()

        # Spaced version should have more nodes
        assert n_spaced > n_simple
