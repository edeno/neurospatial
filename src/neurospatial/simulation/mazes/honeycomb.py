"""Honeycomb Maze environment for spatial navigation research.

The Honeycomb Maze consists of 37 hexagonal platforms in an overall hexagonal
configuration. Each platform can be raised or lowered on its own, so on each
step the animal chooses between two raised neighbours of its current
platform, a discrete navigation task with controlled choices.

The geometry follows Wood et al. (2018), Methods ("Maze Description"): "The
Honeycomb Maze consists of 37 tessellated hexagonal platforms (11.5 cm along
each side) in an overall hexagonal configuration (total maze diameter
145.5 cm, five platforms across with 1 cm gaps between each platform)".
Neighbouring platforms face each other across a flat side, so their centers
are one flat-to-flat width (11.5 cm x sqrt(3)) plus the 1 cm gap apart. Seven
platforms then span 145.4 cm flat to flat, matching the published diameter;
a 37-platform hexagon is seven platforms across its widest row, not five as
the quoted sentence says.

The maze has:
- 1 central platform (platform_0)
- 6 platforms in ring 1
- 12 platforms in ring 2
- 18 platforms in ring 3
Total: 37 platforms with hexagonal 6-connectivity

The maze has no landmarks, because Wood et al. (2018) fix nothing on it
beyond the platforms themselves. The goal and start platforms are chosen by
the testing schedule: "The animal's objective is to reach a specific goal
platform from eight or nine different starting locations", and "the starting
platform changed between trials and these eight or nine platforms were
distributed approximately equally around the maze"; which platforms they
were is shown only in Extended Data Fig. 2. There is no reward port: "Upon
reaching the goal animals were confined there and given one Cheerio® after a
delay of approximately five seconds". The cues are the room's: "The maze sits
within a standard laboratory giving the animal access to abundant extra-maze
cues", with no placement given. For a session's goal, use its
``platform_*`` region for visits, and add a landmark at that platform's
centre if an encoding analysis needs the goal as an object position.

References
----------
Wood, R. A., Bauza, M., Krupic, J., Burton, S., Delekate, A., Chan, D., &
O'Keefe, J. (2018). The honeycomb maze provides a novel test to study
hippocampal-dependent spatial navigation. Nature, 554(7690), 102-105.
https://doi.org/10.1038/nature25433

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
import shapely
from shapely.geometry import Polygon

from neurospatial.environment import Environment
from neurospatial.regions import Regions
from neurospatial.simulation.mazes._base import MazeDims, MazeEnvironments


@dataclass(frozen=True)
class HoneycombDims(MazeDims):
    """Dimension specifications for Honeycomb Maze.

    Each platform is a regular hexagon; neighbouring platforms face each
    other across a flat side, separated by a fixed gap.

    Attributes
    ----------
    platform_side : float
        Side length of each hexagonal platform in cm, which is also its
        center-to-corner radius. Default is 11.5 (Wood et al. 2018).
    gap : float
        Gap between the facing sides of neighbouring platforms in cm.
        Default is 1.0 (Wood et al. 2018).
    n_rings : int
        Number of rings around the central platform. Default is 3, which
        creates 1 + 6 + 12 + 18 = 37 platforms total.

    Examples
    --------
    >>> dims = HoneycombDims()
    >>> dims.platform_side
    11.5
    >>> dims.gap
    1.0
    >>> dims.n_rings
    3
    >>> round(dims.spacing, 2)  # neighbouring centers, cm
    20.92

    >>> custom = HoneycombDims(platform_side=10.0, n_rings=4)
    >>> custom.n_rings
    4
    """

    platform_side: float = 11.5
    gap: float = 1.0
    n_rings: int = 3

    @property
    def spacing(self) -> float:
        """Distance between neighbouring platform centers in cm.

        One platform's flat-to-flat width, ``platform_side * sqrt(3)``, plus
        the gap.
        """
        return float(self.platform_side * np.sqrt(3.0) + self.gap)


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
        Maze dimensions. If None, uses the published dimensions
        (11.5 cm platform sides, 1 cm gaps, 3 rings = 37 platforms).
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
        - env_track: 1D linearized track environment (if include_track=True);
          its ``track_width`` is one platform's corner-to-corner width, so a
          sample anywhere on a platform is on the track, as are a few in the
          gaps between platforms

    Notes
    -----
    The maze is centered at the origin with platform_0 at (0, 0).
    Neighbouring platforms lie along 0, 60, 120, ... degrees from each other
    and face each other across a flat side.

    The 2-D environment is the union of the hexagonal cells that tile the
    plane around each platform center (each cell is its platform grown by
    half a gap), so the floor is one connected piece. It includes the 1 cm
    gaps between platforms, which are narrower than a bin, and a half-gap
    rim around the outside; the ``platform_*`` regions are the platforms
    themselves.

    Platform numbering:
    - platform_0: Center (ring 0)
    - platform_1 to platform_6: Ring 1 (6 platforms)
    - platform_7 to platform_18: Ring 2 (12 platforms)
    - platform_19 to platform_36: Ring 3 (18 platforms)

    Regions:
    - platform_0 through platform_36: Hexagonal polygon regions, one per
      platform

    Track Graph Topology:
    - One node per platform center, one edge per pair of neighbouring
      platforms (90 edges for 37 platforms)
    - Platforms in rings 0-2 have 6 neighbours; on the outer ring, the six
      corner platforms have 3 and the others 4

    Examples
    --------
    Create a default honeycomb maze:

    >>> maze = make_honeycomb_maze()
    >>> maze.env_2d.units
    'cm'
    >>> maze.env_2d.n_bins > 0
    True

    Create with custom dimensions:

    >>> dims = HoneycombDims(platform_side=15.0, n_rings=2)
    >>> maze = make_honeycomb_maze(dims=dims, bin_size=4.0)
    >>> "platform_18" in maze.env_2d.regions
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

    positions = _generate_hexagonal_grid(dims.spacing, dims.n_rings)

    # The cell around each platform has an apothem of half the center
    # spacing, so neighbouring cells share a side and tile without gaps.
    # Snapping to a 1e-6 cm grid makes the shared corners, computed from
    # different centers, coincide exactly, so no sliver holes remain.
    cell_radius = dims.spacing / np.sqrt(3.0)
    floor = shapely.union_all(
        [_make_hexagon_polygon(center, cell_radius) for center in positions],
        grid_size=1e-6,
    )

    env_2d = Environment.from_polygon(
        polygon=floor,
        bin_size=bin_size,
        name="honeycomb_maze",
        connect_diagonal_neighbors=True,
    )
    env_2d = env_2d.with_metadata(units="cm")

    region_builder = Regions()
    for i, pos in enumerate(positions):
        region_builder.add(
            f"platform_{i}", polygon=_make_hexagon_polygon(pos, dims.platform_side)
        )
    env_2d = env_2d.with_regions(region_builder)

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
    """Create a regular hexagon whose flat sides face its lattice neighbours.

    Vertices sit at 30, 90, 150, ... degrees, so the sides are perpendicular
    to the 0, 60, 120, ... degree directions in which neighbouring platforms
    lie.

    Parameters
    ----------
    center : tuple[float, float]
        Center point (x, y) of the hexagon.
    radius : float
        Distance from center to vertices (circumradius), equal to the side.

    Returns
    -------
    Polygon
        Shapely polygon representing the hexagon.
    """
    angles = np.pi / 6 + np.arange(6) * np.pi / 3
    return Polygon(
        np.column_stack(
            [center[0] + radius * np.cos(angles), center[1] + radius * np.sin(angles)]
        )
    )


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

    # The platforms are the walkable surface, so the track is as wide as a
    # platform corner to corner: every point of a platform lies within one
    # side length of its center, which also admits a few samples in the gaps.
    env_track = Environment.from_graph(
        graph=graph,
        edge_order=edge_order,
        edge_spacing=0.0,  # No spacing between edges
        bin_size=bin_size,
        track_width=2.0 * dims.platform_side,
        name="honeycomb_1d",
    )
    env_track = env_track.with_metadata(units="cm")

    return env_track
