"""W-Maze environment for spatial navigation research.

The W-Maze (also called E-maze or M-maze) consists of 3 parallel vertical
corridors connected at the bottom by a horizontal corridor, forming a "W"
or "E" shape. This design allows for multiple goal locations and complex
spatial memory tasks involving discrimination between parallel wells.

Common uses include testing spatial working memory, goal-directed navigation,
and discrimination learning with multiple reward locations.

Examples
--------
>>> from neurospatial.simulation.mazes.w_maze import make_w_maze, WMazeDims
>>> maze = make_w_maze()
>>> maze.env_2d.units
'cm'
>>> len([k for k in maze.env_2d.regions.keys() if k.startswith("well_")]) == 3
True
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from neurospatial import Environment
from neurospatial.simulation.mazes._base import MazeDims, MazeEnvironments
from neurospatial.simulation.mazes._geometry import (
    make_corridor_polygon,
    union_polygons,
)


@dataclass(frozen=True)
class WMazeDims(MazeDims):
    """Dimension specifications for W-Maze.

    The W-Maze consists of 3 parallel vertical wells connected at the bottom
    by a horizontal base corridor, forming a "W" or "E" shape when viewed
    from above.

    Attributes
    ----------
    width : float
        Total horizontal extent of the maze in cm. Default is 120.0.
    height : float
        Length of vertical corridors (wells) in cm. Default is 80.0.
    corridor_width : float
        Width of all corridors in cm. Default is 10.0.
    n_wells : int
        Number of vertical wells. Default is 3.

    Examples
    --------
    >>> dims = WMazeDims()
    >>> dims.width
    120.0
    >>> dims.height
    80.0
    >>> dims.corridor_width
    10.0
    >>> dims.n_wells
    3

    >>> custom = WMazeDims(width=150.0, height=100.0, corridor_width=15.0)
    >>> custom.width
    150.0
    >>> custom.height
    100.0
    """

    width: float = 120.0
    height: float = 80.0
    corridor_width: float = 10.0
    n_wells: int = 3


def make_w_maze(
    dims: WMazeDims | None = None,
    bin_size: float = 2.0,
    include_track: bool = True,
) -> MazeEnvironments:
    """Create a W-Maze environment.

    Creates a W-shaped maze with 3 parallel vertical wells connected by a
    horizontal base corridor. The animal can navigate along the base and
    choose which well to ascend, making it ideal for spatial discrimination
    and working memory tasks.

    Parameters
    ----------
    dims : WMazeDims, optional
        Maze dimensions. If None, uses default dimensions
        (120 cm width, 80 cm height, 10 cm corridor width, 3 wells).
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
    The maze is positioned with the base at y=0:
    - Horizontal base runs from x = -(width/2 + corridor_width/2) to
      x = +(width/2 + corridor_width/2) at y = 0, extending to the outer
      edges of the leftmost and rightmost arms
    - Vertical wells run from y = 0 to y = height
    - For n_wells=3, wells are at x = -width/2, 0, +width/2 (edges and center)

    Regions:
    - well_1: Point at top of first well (-width/2, height)
    - well_2: Point at top of second well (0, height)
    - well_3: Point at top of third well (+width/2, height)

    Track Graph Topology:
    - Horizontal base with connection points at each well bottom
    - Vertical edges from base to each well top
    - Forms a tree structure with base as central path

    Examples
    --------
    Create a default W-maze:

    >>> maze = make_w_maze()
    >>> maze.env_2d.units
    'cm'
    >>> maze.env_2d.n_bins > 0
    True

    Create with custom dimensions:

    >>> dims = WMazeDims(width=150.0, height=100.0, corridor_width=15.0)
    >>> maze = make_w_maze(dims=dims, bin_size=4.0)
    >>> "well_1" in maze.env_2d.regions
    True

    Skip track graph creation:

    >>> maze = make_w_maze(include_track=False)
    >>> maze.env_track is None
    True

    See Also
    --------
    WMazeDims : Dimension specifications for W-Maze.
    """
    if dims is None:
        dims = WMazeDims()

    # Key positions (base at y=0, wells extend upward to y=height)
    half_width = dims.width / 2.0
    half_corridor = dims.corridor_width / 2.0

    # Horizontal base corridor from left to right, extending to outer edges of arms
    # The outer arms are centered at ±half_width, so outer edges are at ±(half_width + half_corridor)
    base_start = (-half_width - half_corridor, 0.0)
    base_end = (half_width + half_corridor, 0.0)

    # Create horizontal base corridor
    base_polygon = make_corridor_polygon(
        start=base_start,
        end=base_end,
        width=dims.corridor_width,
    )

    # Create vertical wells at evenly spaced x positions across full width
    # For 3 wells: x = -width/2, 0, +width/2 (at left edge, center, right edge)
    # Wells span from one edge to the other
    well_x_positions = np.linspace(-half_width, half_width, dims.n_wells)
    well_polygons = []

    for x_pos in well_x_positions:
        well_base = (x_pos, 0.0)
        well_top = (x_pos, dims.height)
        well_polygon = make_corridor_polygon(
            start=well_base,
            end=well_top,
            width=dims.corridor_width,
        )
        well_polygons.append(well_polygon)

    # Union all corridors into single polygon
    w_polygon = union_polygons([base_polygon, *well_polygons])

    # Create 2D environment
    env_2d = Environment.from_polygon(
        polygon=w_polygon,
        bin_size=bin_size,
        name="w_maze",
        connect_diagonal_neighbors=True,
    )
    env_2d.units = "cm"

    # Add regions for key locations - wells at top of each vertical corridor
    for i, x_pos in enumerate(well_x_positions, start=1):
        well_top_pos = (x_pos, dims.height)
        env_2d.regions.add(f"well_{i}", point=well_top_pos)

    # Create track graph if requested
    env_track = None
    if include_track:
        env_track = _create_w_maze_track_graph(dims, bin_size)

    return MazeEnvironments(env_2d=env_2d, env_track=env_track)


def _create_w_maze_track_graph(
    dims: WMazeDims,
    bin_size: float,
) -> Environment:
    """Create the 1D linearized track graph for W-Maze.

    The track graph represents the W-maze topology with a horizontal base
    and vertical wells extending upward.

    Parameters
    ----------
    dims : WMazeDims
        Maze dimensions.
    bin_size : float
        Spatial bin size in cm.

    Returns
    -------
    Environment
        1D linearized environment representing the W-maze track.
    """
    half_width = dims.width / 2.0
    half_corridor = dims.corridor_width / 2.0

    # Well x-positions (for 3 wells: -width/2, 0, +width/2)
    well_x_positions = np.linspace(-half_width, half_width, dims.n_wells)

    # Create track graph
    graph = nx.Graph()

    # Add nodes along the base corridor
    # Base extends to outer edges of the outer arms
    base_node_positions = [
        ("base_left", (-half_width - half_corridor, 0.0)),
    ]

    # Add junction nodes at each well base
    for i, x_pos in enumerate(well_x_positions, start=1):
        base_node_positions.append((f"junction_{i}", (x_pos, 0.0)))

    # Add end of base
    base_node_positions.append(("base_right", (half_width + half_corridor, 0.0)))

    # Add all base nodes
    for node_name, pos in base_node_positions:
        graph.add_node(node_name, pos=pos)

    # Add well top nodes
    for i, x_pos in enumerate(well_x_positions, start=1):
        well_top_pos = (x_pos, dims.height)
        graph.add_node(f"well_{i}_top", pos=well_top_pos)

    # Add edges along the base corridor
    for i in range(len(base_node_positions) - 1):
        node1_name = base_node_positions[i][0]
        node2_name = base_node_positions[i + 1][0]
        pos1 = np.array(base_node_positions[i][1])
        pos2 = np.array(base_node_positions[i + 1][1])
        distance = np.linalg.norm(pos2 - pos1)
        graph.add_edge(node1_name, node2_name, distance=distance)

    # Add edges from junctions to well tops
    # Each well is vertical with height = dims.height
    well_distance = dims.height
    for i in range(1, dims.n_wells + 1):
        junction_name = f"junction_{i}"
        well_top_name = f"well_{i}_top"
        graph.add_edge(junction_name, well_top_name, distance=well_distance)

    # Edge order for linearization (base path + wells)
    edge_order = []

    # Base corridor edges
    for i in range(len(base_node_positions) - 1):
        node1 = base_node_positions[i][0]
        node2 = base_node_positions[i + 1][0]
        edge_order.append((node1, node2))

    # Well edges
    for i in range(1, dims.n_wells + 1):
        junction = f"junction_{i}"
        well_top = f"well_{i}_top"
        edge_order.append((junction, well_top))

    # Create the 1D environment
    env_track = Environment.from_graph(
        graph=graph,
        edge_order=edge_order,
        edge_spacing=0.0,  # No spacing between edges
        bin_size=bin_size,
        name="w_maze_1d",
    )
    env_track.units = "cm"

    return env_track
