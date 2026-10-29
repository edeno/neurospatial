"""Geometry helper functions for maze construction.

This module provides utility functions for creating maze geometries:
- Corridor polygons from line segments
- Circular arenas
- The union of corridor pieces into one floor

All dimensions are in centimeters (cm) by convention.

Examples
--------
>>> from neurospatial.simulation.mazes._geometry import make_corridor_polygon
>>> poly = make_corridor_polygon(start=(0, 0), end=(100, 0), width=10)
>>> poly.is_valid
True
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box
from shapely.ops import unary_union

if TYPE_CHECKING:
    from neurospatial.simulation.mazes._base import MazeDims


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


def corridor_end_point(
    start: tuple[float, float],
    end: tuple[float, float],
    width: float,
) -> tuple[float, float]:
    """Return the center of a corridor's end square, where a well sits.

    The point lies on the corridor's axis, half a corridor width back from
    ``end``, so a region there covers the bins at the end of the arm. The end
    point itself lies on the corridor's outline.

    Parameters
    ----------
    start, end : tuple of float
        Corridor axis endpoints, as for :func:`make_corridor_polygon`.
    width : float
        Corridor width.

    Returns
    -------
    tuple of float
        ``(x, y)`` of the end square's center.
    """
    start_xy, end_xy = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    axis = end_xy - start_xy
    point = end_xy - axis / np.linalg.norm(axis) * (width / 2.0)
    return (float(point[0]), float(point[1]))


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
    polygons: Sequence[Polygon | MultiPolygon],
    simplify_tolerance: float | None = None,
) -> Polygon | MultiPolygon:
    """Combine multiple polygons into a single geometry.

    Performs a union of all polygons, optionally simplifying the result
    to remove small artifacts from floating-point geometry.

    Parameters
    ----------
    polygons : Sequence[Polygon or MultiPolygon]
        Shapely polygons to combine.
    simplify_tolerance : float, optional
        If provided, simplify the result with this tolerance.
        Useful for cleaning up small geometric artifacts.

    Returns
    -------
    Polygon or MultiPolygon
        The combined geometry: a Polygon when the pieces join, a
        MultiPolygon when some do not touch the others. Use
        :func:`single_floor` to require one connected floor.

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


def single_floor(geometry: Polygon | MultiPolygon, dims: MazeDims) -> Polygon:
    """Return a maze's floor as one polygon, or refuse dimensions that split it.

    Parameters
    ----------
    geometry : Polygon or MultiPolygon
        The union of the maze's corridors, rooms and boxes.
    dims : MazeDims
        The dimensions the floor was built from, named in the error.

    Returns
    -------
    Polygon
        ``geometry`` itself when it is one connected piece.

    Raises
    ------
    ValueError
        If the pieces do not join into one floor.

    Examples
    --------
    >>> from neurospatial.simulation.mazes._base import MazeDims
    >>> floor = union_polygons([make_corridor_polygon((0, 0), (100, 0), width=10)])
    >>> single_floor(floor, MazeDims()).area
    1000.0
    """
    if isinstance(geometry, Polygon) and not geometry.is_empty:
        return geometry
    n_pieces = len(getattr(geometry, "geoms", ()))
    raise ValueError(
        f"{dims!r} builds a maze floor in {n_pieces} separate pieces, so an "
        "animal could not walk between them. Choose dimensions whose "
        "corridors, rooms and boxes meet."
    )


def compass_quadrants(floor: Polygon) -> dict[str, Polygon]:
    """Split an origin-centred floor into its NE, NW, SE and SW quarters.

    Each quadrant is the part of ``floor`` on one side of both the
    north-south and the east-west line through the origin, with north +y
    and east +x (NE is x >= 0, y >= 0). Together they cover the floor
    without overlapping.

    Parameters
    ----------
    floor : Polygon
        An open-field floor centred on the origin, in cm.

    Returns
    -------
    dict of str to Polygon
        ``{"NE": ..., "NW": ..., "SE": ..., "SW": ...}``.

    Examples
    --------
    >>> from shapely.geometry import box
    >>> quadrants = compass_quadrants(box(-15.0, -15.0, 15.0, 15.0))
    >>> quadrants["NE"].bounds
    (0.0, 0.0, 15.0, 15.0)
    """
    reach = float(np.max(np.abs(floor.bounds)))
    quadrants = {}
    for name, (sign_x, sign_y) in {
        "NE": (1, 1),
        "NW": (-1, 1),
        "SE": (1, -1),
        "SW": (-1, -1),
    }.items():
        quarter = box(
            min(0.0, sign_x * reach),
            min(0.0, sign_y * reach),
            max(0.0, sign_x * reach),
            max(0.0, sign_y * reach),
        )
        quadrants[name] = floor.intersection(quarter)
    return quadrants


def corridor_width(corridor: Polygon) -> float:
    """Width of a straight corridor: the short side of its tightest rectangle.

    Parameters
    ----------
    corridor : Polygon
        An elongated corridor or arm, in cm.

    Returns
    -------
    float
        The corridor's width in cm.

    Examples
    --------
    >>> from shapely.geometry import box
    >>> corridor_width(box(0.0, 0.0, 50.0, 8.0))
    8.0
    """
    corners = np.asarray(corridor.minimum_rotated_rectangle.exterior.coords)[:4]
    sides = np.hypot(*(np.roll(corners, -1, axis=0) - corners).T)
    return float(sides.min())


def widest_corridor(
    walkable: Polygon,
    segments: Sequence[tuple[Sequence[float], Sequence[float]]],
) -> float:
    """Widest corridor along a track: twice the largest mid-segment clearance.

    The clearance of a segment is the distance from its midpoint to the edge
    of the walkable area, half the corridor's width there.

    Parameters
    ----------
    walkable : Polygon
        The area the animal can walk on, in cm.
    segments : sequence of (start, end)
        Track segments, each a pair of ``(x, y)`` points.

    Returns
    -------
    float
        The widest corridor width in cm.

    Examples
    --------
    >>> from shapely.geometry import box
    >>> widest_corridor(box(0.0, 0.0, 50.0, 8.0), [((0.0, 4.0), (50.0, 4.0))])
    8.0
    """
    widths = []
    for start, end in segments:
        middle = Point((np.asarray(start) + np.asarray(end)) / 2)
        if walkable.contains(middle):
            widths.append(2.0 * walkable.boundary.distance(middle))
    return float(max(widths))


def warn_if_bins_bridge_walls(
    bin_size: float, wall_thickness: float, maze: str
) -> None:
    """Warn when bins are too coarse to keep a maze's thinnest wall.

    A 2-D maze keeps the bins whose centers lie on the floor and joins
    neighbouring bins. A wall wider than one bin always drops a row of bins;
    a wall no wider than a bin can drop none, joining the corridors on either
    side.

    Parameters
    ----------
    bin_size : float
        Spatial bin size in cm.
    wall_thickness : float
        Thickness of the maze's thinnest wall between corridors, in cm.
    maze : str
        Name of the maze, for the message.

    Examples
    --------
    >>> import warnings
    >>> with warnings.catch_warnings(record=True) as caught:
    ...     warnings.simplefilter("always")
    ...     warn_if_bins_bridge_walls(4.0, 3.1, "Hampton Court maze")
    >>> len(caught)
    1
    """
    if bin_size >= wall_thickness:
        warnings.warn(
            f"bin_size={bin_size} cm is not smaller than the {wall_thickness:.1f} cm "
            f"walls of the {maze}, so bins on either side of a wall can join and "
            "let paths pass through it. Use bin_size below "
            f"{wall_thickness:.1f} cm to keep every wall.",
            UserWarning,
            stacklevel=3,
        )
