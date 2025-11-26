"""Honeycomb Maze environment for spatial navigation research.

The Honeycomb Maze consists of 37 hexagonal platforms arranged in concentric rings
around a central platform. Platforms can be raised or lowered individually, creating
flexible spatial navigation tasks. This design is particularly useful for studying
goal-directed navigation and path integration in a discrete, graph-like environment.

Reference: Wood et al. 2018 (eLife) - "The honeycomb maze provides a novel test to
study hippocampal-dependent spatial navigation"

The maze has:
- 1 central platform (platform_0)
- 6 platforms in ring 1
- 12 platforms in ring 2
- 18 platforms in ring 3
Total: 37 platforms with hexagonal 6-connectivity

Examples
--------
>>> from neurospatial.simulation.mazes.honeycomb import (
...     make_honeycomb_maze,
...     HoneycombDims,
... )
>>> maze = make_honeycomb_maze()
>>> maze.env_2d.units
'cm'
>>> "platform_0" in maze.env_2d.regions
True
>>> len([r for r in maze.env_2d.regions if r.startswith("platform_")])
37
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
from shapely.geometry import Polygon

from neurospatial import Environment
from neurospatial.simulation.mazes._base import MazeDims, MazeEnvironments
from neurospatial.simulation.mazes._geometry import union_polygons


@dataclass(frozen=True)
class HoneycombDims(MazeDims):
    """Dimension specifications for Honeycomb Maze.

    The Honeycomb Maze consists of hexagonal platforms arranged in concentric
    rings. Each platform is a regular hexagon, and adjacent platforms are
    separated by a fixed spacing.

    Attributes
    ----------
    spacing : float
        Distance between adjacent platform centers in cm. Default is 25.0.
    n_rings : int
        Number of rings around the central platform. Default is 3, which
        creates 1 + 6 + 12 + 18 = 37 platforms total.

    Examples
    --------
    >>> dims = HoneycombDims()
    >>> dims.spacing
    25.0
    >>> dims.n_rings
    3

    >>> custom = HoneycombDims(spacing=30.0, n_rings=4)
    >>> custom.spacing
    30.0
    >>> custom.n_rings
    4
    """

    spacing: float = 25.0
    n_rings: int = 3


def make_honeycomb_maze(
    dims: HoneycombDims | None = None,
    bin_size: float = 2.0,
    include_track: bool = True,
) -> MazeEnvironments:
    """Create a Honeycomb Maze environment.

    Creates a maze with hexagonal platforms arranged in concentric rings.
    The platforms can be used to study discrete spatial navigation, with
    each platform connected to up to 6 neighbors in a hexagonal lattice.

    Parameters
    ----------
    dims : HoneycombDims, optional
        Maze dimensions. If None, uses default dimensions
        (25 cm spacing, 3 rings = 37 platforms).
    bin_size : float, optional
        Spatial bin size in cm (default: 2.0).
    include_track : bool, optional
        Whether to create linearized track graph (default: True).
        The track graph provides a 1D representation for trajectory analysis.

    Returns
    -------
    MazeEnvironments
        Contains:
        - env_2d: 2D polygon-based environment
        - env_track: 1D linearized track environment (if include_track=True)

    Notes
    -----
    The maze is centered at the origin with platform_0 at (0, 0).

    Platform numbering:
    - platform_0: Center (ring 0)
    - platform_1 to platform_6: Ring 1 (6 platforms)
    - platform_7 to platform_18: Ring 2 (12 platforms)
    - platform_19 to platform_36: Ring 3 (18 platforms)

    Regions:
    - platform_0 through platform_36: Point regions at platform centers

    Track Graph Topology:
    - Hexagonal 6-connectivity: Each platform connects to up to 6 neighbors
    - Ring 0 (center) connects to all 6 platforms in ring 1
    - Ring 1 platforms connect to center and ring 2 neighbors
    - Ring 2 platforms connect to ring 1 and ring 3 neighbors
    - Ring 3 (outermost) platforms connect to ring 2 neighbors only

    Examples
    --------
    Create a default honeycomb maze:

    >>> maze = make_honeycomb_maze()
    >>> maze.env_2d.units
    'cm'
    >>> maze.env_2d.n_bins > 0
    True

    Create with custom dimensions:

    >>> dims = HoneycombDims(spacing=30.0, n_rings=2)
    >>> maze = make_honeycomb_maze(dims=dims, bin_size=4.0)
    >>> "platform_0" in maze.env_2d.regions
    True

    Skip track graph creation:

    >>> maze = make_honeycomb_maze(include_track=False)
    >>> maze.env_track is None
    True

    See Also
    --------
    HoneycombDims : Dimension specifications for Honeycomb Maze.
    """
    if dims is None:
        dims = HoneycombDims()

    # Generate all platform positions
    positions = _generate_hexagonal_grid(dims.spacing, dims.n_rings)

    # Create hexagonal polygon for each platform
    # Platform radius is approximately spacing / 2 to allow slight gap between platforms
    platform_radius = dims.spacing / 2.5
    platform_polygons = [
        _make_hexagon_polygon(center, platform_radius) for center in positions
    ]

    # Union all platforms into single polygon
    honeycomb_polygon = union_polygons(platform_polygons)

    # Convert to Polygon if result is MultiPolygon (extract convex hull or buffer slightly)
    # For honeycomb, we want one contiguous polygon, so use convex_hull or buffer
    from shapely.geometry import MultiPolygon

    if isinstance(honeycomb_polygon, MultiPolygon):
        # Buffer by a small amount to merge nearby polygons
        honeycomb_polygon = honeycomb_polygon.buffer(0.1).buffer(-0.1)
        # If still MultiPolygon, take convex hull
        if isinstance(honeycomb_polygon, MultiPolygon):
            honeycomb_polygon = honeycomb_polygon.convex_hull

    # Create 2D environment
    env_2d = Environment.from_polygon(
        polygon=honeycomb_polygon,
        bin_size=bin_size,
        name="honeycomb_maze",
        connect_diagonal_neighbors=True,
    )
    env_2d.units = "cm"

    # Add regions for each platform (numbered 0 through n_platforms-1)
    for i, pos in enumerate(positions):
        env_2d.regions.add(f"platform_{i}", point=pos)

    # Create track graph if requested
    env_track = None
    if include_track:
        env_track = _create_honeycomb_track_graph(dims, bin_size)

    return MazeEnvironments(env_2d=env_2d, env_track=env_track)


def _generate_hexagonal_grid(
    spacing: float,
    n_rings: int,
) -> list[tuple[float, float]]:
    """Generate positions for hexagonal platform grid.

    Creates a hexagonal grid with a central platform and concentric rings.
    Each ring has 6 * ring_number platforms.

    Parameters
    ----------
    spacing : float
        Distance between adjacent platform centers.
    n_rings : int
        Number of rings around the center (0 = center only).

    Returns
    -------
    list[tuple[float, float]]
        List of (x, y) positions for each platform, starting with center.
    """
    positions = []

    # Ring 0: center platform
    positions.append((0.0, 0.0))

    # Generate each ring
    for ring in range(1, n_rings + 1):
        ring_positions = _generate_hex_ring(ring, spacing)
        positions.extend(ring_positions)

    return positions


def _generate_hex_ring(ring_number: int, spacing: float) -> list[tuple[float, float]]:
    """Generate positions for a single hexagonal ring.

    A hexagonal ring has 6 * ring_number platforms, arranged in a hexagon
    with ring_number platforms per edge.

    Parameters
    ----------
    ring_number : int
        Ring number (1, 2, 3, ...). Ring 1 has 6 platforms, ring 2 has 12, etc.
    spacing : float
        Distance between adjacent platform centers.

    Returns
    -------
    list[tuple[float, float]]
        List of (x, y) positions for platforms in this ring.
    """
    if ring_number == 0:
        return [(0.0, 0.0)]

    positions = []

    # Generate platforms along each of the 6 edges of the hexagon
    # Start at corner 0 (directly to the right, angle = 0)
    for corner in range(6):
        # Corner positions are at 60° intervals
        corner_angle = corner * np.pi / 3

        # Starting position for this edge
        start_x = ring_number * spacing * np.cos(corner_angle)
        start_y = ring_number * spacing * np.sin(corner_angle)

        # Next corner position
        next_corner_angle = (corner + 1) * np.pi / 3
        next_x = ring_number * spacing * np.cos(next_corner_angle)
        next_y = ring_number * spacing * np.sin(next_corner_angle)

        # Add platforms along this edge (including starting corner, excluding end)
        for i in range(ring_number):
            t = i / ring_number
            x = start_x + t * (next_x - start_x)
            y = start_y + t * (next_y - start_y)
            positions.append((x, y))

    return positions


def _make_hexagon_polygon(
    center: tuple[float, float],
    radius: float,
) -> Polygon:
    """Create a regular hexagon polygon.

    Parameters
    ----------
    center : tuple[float, float]
        Center point (x, y) of the hexagon.
    radius : float
        Distance from center to vertices (circumradius).

    Returns
    -------
    Polygon
        Shapely polygon representing the hexagon.
    """
    # Generate 6 vertices at 60° intervals
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]  # 6 vertices, exclude duplicate endpoint
    vertices = [
        (center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle))
        for angle in angles
    ]
    return Polygon(vertices)


def _create_honeycomb_track_graph(
    dims: HoneycombDims,
    bin_size: float,
) -> Environment:
    """Create the 1D linearized track graph for Honeycomb Maze.

    The track graph represents the hexagonal connectivity between platforms.
    Each platform can connect to up to 6 neighbors (except edge platforms).

    Parameters
    ----------
    dims : HoneycombDims
        Maze dimensions.
    bin_size : float
        Spatial bin size in cm.

    Returns
    -------
    Environment
        1D linearized environment representing the honeycomb track.
    """
    # Generate all platform positions
    positions = _generate_hexagonal_grid(dims.spacing, dims.n_rings)

    # Create graph with all platforms as nodes
    graph = nx.Graph()

    # Add nodes with positions
    for i, pos in enumerate(positions):
        graph.add_node(f"platform_{i}", pos=pos)

    # Add edges between adjacent platforms (hexagonal 6-connectivity)
    # Two platforms are adjacent if their distance is approximately equal to spacing
    distance_threshold = dims.spacing * 1.1  # Allow 10% tolerance

    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            pos_i = np.array(positions[i])
            pos_j = np.array(positions[j])
            distance = np.linalg.norm(pos_i - pos_j)

            if distance <= distance_threshold:
                graph.add_edge(
                    f"platform_{i}",
                    f"platform_{j}",
                    distance=distance,
                )

    # Create edge order for linearization
    # We'll use a breadth-first traversal starting from the center
    edge_order = []
    visited_edges = set()

    def add_edges_from_node(node: str) -> None:
        """Add all unvisited edges from a node to edge_order."""
        for neighbor in graph.neighbors(node):
            edge = tuple(sorted([node, neighbor]))
            if edge not in visited_edges:
                visited_edges.add(edge)
                edge_order.append(edge)
                add_edges_from_node(neighbor)

    # Start from center platform
    add_edges_from_node("platform_0")

    # Create the 1D environment
    env_track = Environment.from_graph(
        graph=graph,
        edge_order=edge_order,
        edge_spacing=0.0,  # No spacing between edges
        bin_size=bin_size,
        name="honeycomb_1d",
    )
    env_track.units = "cm"

    return env_track
