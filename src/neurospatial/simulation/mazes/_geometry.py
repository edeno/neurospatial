"""Geometry helper functions for maze construction.

This module provides utility functions for creating maze geometries:
- Corridor polygons from line segments
- Circular arenas
- Star-topology track graphs

All dimensions are in centimeters (cm) by convention.

Examples
--------
>>> from neurospatial.simulation.mazes._geometry import make_corridor_polygon
>>> poly = make_corridor_polygon(start=(0, 0), end=(100, 0), width=10)
>>> poly.is_valid
True
"""

from __future__ import annotations

from collections.abc import Sequence

import networkx as nx
import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union


def make_corridor_polygon(
    start: tuple[float, float],
    end: tuple[float, float],
    width: float,
) -> Polygon:
    """Create a rectangular corridor polygon between two points.

    Creates a rectangle with the given width, centered on the line
    segment from start to end. The corridor is perpendicular to the
    line at both endpoints (square ends, not rounded).

    Parameters
    ----------
    start : tuple[float, float]
        Starting point (x, y) of the corridor center.
    end : tuple[float, float]
        Ending point (x, y) of the corridor center.
    width : float
        Total width of the corridor (perpendicular to the direction).

    Returns
    -------
    Polygon
        A Shapely Polygon representing the rectangular corridor.

    Examples
    --------
    >>> poly = make_corridor_polygon(start=(0, 0), end=(100, 0), width=10)
    >>> poly.is_valid
    True
    >>> 900 < poly.area < 1100  # ~100 x 10 = 1000
    True
    """
    # Convert to numpy arrays for vector math
    p1 = np.array(start)
    p2 = np.array(end)

    # Direction vector
    direction = p2 - p1
    length = np.linalg.norm(direction)

    if length == 0:
        raise ValueError("Start and end points must be different")

    # Normalize and get perpendicular
    unit_dir = direction / length
    perpendicular = np.array([-unit_dir[1], unit_dir[0]])

    # Half-width offset
    half_width = width / 2

    # Four corners of the rectangle
    corners = [
        p1 - perpendicular * half_width,
        p1 + perpendicular * half_width,
        p2 + perpendicular * half_width,
        p2 - perpendicular * half_width,
    ]

    return Polygon(corners)


def make_buffered_line(
    start: tuple[float, float],
    end: tuple[float, float],
    width: float,
) -> Polygon:
    """Create a polygon by buffering a line segment.

    Creates a polygon by applying a buffer (distance) around the line
    segment. This results in rounded ends (cap_style='round' by default).

    Parameters
    ----------
    start : tuple[float, float]
        Starting point (x, y) of the line.
    end : tuple[float, float]
        Ending point (x, y) of the line.
    width : float
        Total width of the resulting polygon (buffer is half this).

    Returns
    -------
    Polygon
        A Shapely Polygon with rounded ends.

    Examples
    --------
    >>> poly = make_buffered_line(start=(0, 0), end=(100, 0), width=10)
    >>> poly.is_valid
    True
    >>> poly.contains(Point(0, 0))
    True
    """
    line = LineString([start, end])
    # Buffer by half-width to get total width
    return line.buffer(width / 2, cap_style="round")


def make_circular_arena(
    center: tuple[float, float],
    radius: float,
    quad_segs: int = 16,
) -> Polygon:
    """Create a circular polygon (arena, pool, platform).

    Parameters
    ----------
    center : tuple[float, float]
        Center point (x, y) of the circle.
    radius : float
        Radius of the circle.
    quad_segs : int, optional
        Number of segments per quadrant to approximate the circle.
        Default is 16, giving 64 segments total.

    Returns
    -------
    Polygon
        A Shapely Polygon approximating a circle.

    Examples
    --------
    >>> import numpy as np
    >>> poly = make_circular_arena(center=(0, 0), radius=50)
    >>> poly.is_valid
    True
    >>> abs(poly.area - np.pi * 50**2) < 100  # Close to pi*r^2
    True
    """
    point = Point(center)
    return point.buffer(radius, quad_segs=quad_segs)


def union_polygons(
    polygons: Sequence[Polygon],
    simplify_tolerance: float | None = None,
) -> Polygon:
    """Combine multiple polygons into a single geometry.

    Performs a union of all polygons, optionally simplifying the result
    to remove small artifacts from floating-point geometry.

    Parameters
    ----------
    polygons : Sequence[Polygon]
        List of Shapely Polygons to combine.
    simplify_tolerance : float, optional
        If provided, simplify the result with this tolerance.
        Useful for cleaning up small geometric artifacts.

    Returns
    -------
    Polygon
        Combined polygon (may be MultiPolygon if inputs don't overlap).

    Raises
    ------
    ValueError
        If polygons sequence is empty.

    Examples
    --------
    >>> poly1 = make_corridor_polygon((0, 0), (100, 0), width=10)
    >>> poly2 = make_corridor_polygon((50, 0), (50, 100), width=10)
    >>> combined = union_polygons([poly1, poly2])
    >>> combined.is_valid
    True
    """
    if not polygons:
        raise ValueError("union_polygons requires at least one polygon")

    result = unary_union(polygons)

    if simplify_tolerance is not None:
        result = result.simplify(simplify_tolerance)

    return result


def make_star_graph(
    center: tuple[float, float],
    arm_endpoints: Sequence[tuple[float, float]],
    spacing: float | None = None,
) -> nx.Graph:
    """Create a star-topology track graph.

    Creates a graph with a central node connected to each arm endpoint.
    Optionally adds intermediate nodes along each arm at regular spacing.

    Parameters
    ----------
    center : tuple[float, float]
        Position (x, y) of the central node.
    arm_endpoints : Sequence[tuple[float, float]]
        List of (x, y) positions for arm endpoints.
    spacing : float, optional
        If provided, add intermediate nodes along arms at this spacing.
        Nodes are placed from center outward.

    Returns
    -------
    nx.Graph
        NetworkX graph with:
        - Node attributes: 'pos' (2D position tuple)
        - Edge attributes: 'distance' (Euclidean distance)

    Examples
    --------
    >>> graph = make_star_graph(
    ...     center=(0, 0),
    ...     arm_endpoints=[(50, 0), (-50, 0), (0, 50), (0, -50)],
    ... )
    >>> graph.number_of_nodes()
    5
    >>> graph.number_of_edges()
    4
    >>> "center" in graph.nodes()
    True
    """
    graph = nx.Graph()

    # Add center node
    graph.add_node("center", pos=center)

    center_arr = np.array(center)

    for i, endpoint in enumerate(arm_endpoints):
        endpoint_arr = np.array(endpoint)
        arm_direction = endpoint_arr - center_arr
        arm_length = np.linalg.norm(arm_direction)

        if arm_length == 0:
            raise ValueError(f"Arm endpoint {i} is at the same position as center")

        endpoint_name = f"arm_{i}_end"

        if spacing is None or spacing >= arm_length:
            # Simple case: just center and endpoint
            graph.add_node(endpoint_name, pos=tuple(endpoint_arr))
            graph.add_edge("center", endpoint_name, distance=arm_length)
        else:
            # Add intermediate nodes
            unit_dir = arm_direction / arm_length

            # How many intermediate nodes?
            n_segments = int(np.ceil(arm_length / spacing))
            actual_spacing = arm_length / n_segments

            prev_node = "center"
            for seg_idx in range(1, n_segments):
                # Position along arm
                pos = center_arr + unit_dir * (seg_idx * actual_spacing)
                node_name = f"arm_{i}_node_{seg_idx}"
                graph.add_node(node_name, pos=tuple(pos))
                graph.add_edge(prev_node, node_name, distance=actual_spacing)
                prev_node = node_name

            # Add endpoint
            graph.add_node(endpoint_name, pos=tuple(endpoint_arr))
            # Distance from last intermediate to endpoint
            last_dist = arm_length - (n_segments - 1) * actual_spacing
            graph.add_edge(prev_node, endpoint_name, distance=last_dist)

    return graph
