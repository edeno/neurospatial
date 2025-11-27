"""Plus Maze (Cruciate Maze) environment for spatial navigation research.

The Plus Maze (also called Cruciate Maze) is a four-arm maze forming a "+"
shape with perpendicular arms extending in all cardinal directions from a
central junction. It is commonly used for rule-switching tasks where animals
must learn and flexibly switch between different response rules (e.g., go
east vs go west, or go to the light vs go to the dark arm).

Reference: Wijnen et al. 2024 (Brain Structure & Function)

The plus maze provides more decision alternatives than a T-maze while
maintaining a simple structure that supports both egocentric (body-turn)
and allocentric (world-centered) navigation strategies.

Examples
--------
>>> from neurospatial.simulation.mazes import make_plus_maze, PlusMazeDims
>>> maze = make_plus_maze()
>>> maze.env_2d.units
'cm'
>>> "center" in maze.env_2d.regions
True
>>> "north_end" in maze.env_2d.regions
True
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from neurospatial import Environment
from neurospatial.simulation.mazes._base import MazeDims, MazeEnvironments
from neurospatial.simulation.mazes._geometry import (
    make_corridor_polygon,
    union_polygons,
)


@dataclass(frozen=True)
class PlusMazeDims(MazeDims):
    """Dimension specifications for Plus Maze (Cruciate Maze).

    The Plus Maze consists of four arms extending from a central junction
    in the cardinal directions (north, south, east, west). All arms are
    symmetric by default.

    Attributes
    ----------
    arm_length : float
        Length of each arm from center to end in cm. Default is 45.0,
        giving a total span of 100 cm (45 + 10/2 + 45 + 10/2 ≈ 100 cm).
    width : float
        Width of all corridors in cm. Default is 10.0.

    Notes
    -----
    Default dimensions create a 100 cm x 100 cm maze footprint:
    - Total width/height = 2 * arm_length + width = 2 * 45 + 10 = 100 cm

    Examples
    --------
    >>> dims = PlusMazeDims()
    >>> dims.arm_length
    45.0
    >>> dims.width
    10.0

    >>> custom = PlusMazeDims(arm_length=60.0, width=15.0)
    >>> custom.arm_length
    60.0
    """

    arm_length: float = 45.0
    width: float = 10.0


def make_plus_maze(
    dims: PlusMazeDims | None = None,
    bin_size: float = 2.0,
    include_track: bool = True,
) -> MazeEnvironments:
    """Create a Plus Maze (Cruciate Maze) environment.

    Creates a plus-shaped maze with four arms extending in cardinal directions
    from a central junction. Animals can navigate to any of the four endpoints,
    making it ideal for studying rule-switching, spatial memory, and decision-
    making with multiple alternatives.

    Parameters
    ----------
    dims : PlusMazeDims, optional
        Maze dimensions. If None, uses default dimensions creating a
        100 cm x 100 cm maze (45 cm arms, 10 cm width).
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
    The maze is centered at the origin:
    - North arm: y = 0 to y = +arm_length
    - South arm: y = 0 to y = -arm_length
    - East arm: x = 0 to x = +arm_length
    - West arm: x = 0 to x = -arm_length

    Regions:
    - center: Central junction at (0, 0)
    - north_end: North arm endpoint (0, +arm_length)
    - south_end: South arm endpoint (0, -arm_length)
    - east_end: East arm endpoint (+arm_length, 0)
    - west_end: West arm endpoint (-arm_length, 0)

    Track Graph Topology (star):
    - center -> north_end
    - center -> south_end
    - center -> east_end
    - center -> west_end

    Examples
    --------
    Create a default plus maze:

    >>> maze = make_plus_maze()
    >>> maze.env_2d.units
    'cm'
    >>> maze.env_2d.n_bins > 0
    True

    Create with custom dimensions:

    >>> dims = PlusMazeDims(arm_length=60.0, width=15.0)
    >>> maze = make_plus_maze(dims=dims, bin_size=4.0)
    >>> "center" in maze.env_2d.regions
    True

    Skip track graph creation:

    >>> maze = make_plus_maze(include_track=False)
    >>> maze.env_track is None
    True

    See Also
    --------
    PlusMazeDims : Dimension specifications for Plus Maze.
    make_t_maze : Simpler T-shaped maze with only 3 arms.
    """
    if dims is None:
        dims = PlusMazeDims()

    # Key positions (maze centered at origin)
    center_pos = (0.0, 0.0)
    north_end_pos = (0.0, dims.arm_length)
    south_end_pos = (0.0, -dims.arm_length)
    east_end_pos = (dims.arm_length, 0.0)
    west_end_pos = (-dims.arm_length, 0.0)

    # Create corridor polygons for each arm
    # North arm: center to north
    north_polygon = make_corridor_polygon(
        start=center_pos,
        end=north_end_pos,
        width=dims.width,
    )

    # South arm: center to south
    south_polygon = make_corridor_polygon(
        start=center_pos,
        end=south_end_pos,
        width=dims.width,
    )

    # East arm: center to east
    east_polygon = make_corridor_polygon(
        start=center_pos,
        end=east_end_pos,
        width=dims.width,
    )

    # West arm: center to west
    west_polygon = make_corridor_polygon(
        start=center_pos,
        end=west_end_pos,
        width=dims.width,
    )

    # Union all corridors into single plus-shaped polygon
    plus_polygon = union_polygons(
        [north_polygon, south_polygon, east_polygon, west_polygon]
    )

    # Create 2D environment
    env_2d = Environment.from_polygon(
        polygon=plus_polygon,
        bin_size=bin_size,
        name="plus_maze",
        connect_diagonal_neighbors=True,
    )
    env_2d.units = "cm"

    # Add regions for key locations
    env_2d.regions.add("center", point=center_pos)
    env_2d.regions.add("north_end", point=north_end_pos)
    env_2d.regions.add("south_end", point=south_end_pos)
    env_2d.regions.add("east_end", point=east_end_pos)
    env_2d.regions.add("west_end", point=west_end_pos)

    # Create track graph if requested
    env_track = None
    if include_track:
        env_track = _create_plus_maze_track_graph(dims, bin_size)

    return MazeEnvironments(env_2d=env_2d, env_track=env_track)


def _create_plus_maze_track_graph(
    dims: PlusMazeDims,
    bin_size: float,
) -> Environment:
    """Create the 1D linearized track graph for Plus Maze.

    The track graph represents the plus maze topology with four edges
    radiating from a central node (star topology):
    - center -> north_end
    - center -> south_end
    - center -> east_end
    - center -> west_end

    Parameters
    ----------
    dims : PlusMazeDims
        Maze dimensions.
    bin_size : float
        Spatial bin size in cm.

    Returns
    -------
    Environment
        1D linearized environment representing the plus maze track.
    """
    # Key positions
    center_pos = (0.0, 0.0)
    north_end_pos = (0.0, dims.arm_length)
    south_end_pos = (0.0, -dims.arm_length)
    east_end_pos = (dims.arm_length, 0.0)
    west_end_pos = (-dims.arm_length, 0.0)

    # Create track graph (star topology)
    graph = nx.Graph()

    # Add nodes with positions
    graph.add_node("center", pos=center_pos)
    graph.add_node("north_end", pos=north_end_pos)
    graph.add_node("south_end", pos=south_end_pos)
    graph.add_node("east_end", pos=east_end_pos)
    graph.add_node("west_end", pos=west_end_pos)

    # Add edges with distances (all arms have same length)
    arm_distance = dims.arm_length
    graph.add_edge("center", "north_end", distance=arm_distance)
    graph.add_edge("center", "south_end", distance=arm_distance)
    graph.add_edge("center", "east_end", distance=arm_distance)
    graph.add_edge("center", "west_end", distance=arm_distance)

    # Edge order for linearization (traversal from center outward)
    edge_order = [
        ("center", "north_end"),
        ("center", "east_end"),
        ("center", "south_end"),
        ("center", "west_end"),
    ]

    # Create the 1D environment
    env_track = Environment.from_graph(
        graph=graph,
        edge_order=edge_order,
        edge_spacing=0.0,  # No spacing between edges
        bin_size=bin_size,
        name="plus_maze_1d",
    )
    env_track.units = "cm"

    return env_track
