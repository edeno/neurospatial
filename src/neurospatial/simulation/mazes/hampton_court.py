"""Hampton Court Maze environment for spatial navigation research.

The Hampton Court Maze is a complex labyrinth with multiple dead ends and winding
passages. It was inspired by the famous hedge maze at Hampton Court Palace in England
and is one of the first mazes used in rodent research (Small, 1901).

This implementation creates a simplified procedural version of the maze with:
- Grid-based labyrinth layout
- Single solution path from start to goal
- Multiple dead ends to increase complexity
- Start at edge, goal at center

Reference: Small (1901) - Early maze research with rodents

Examples
--------
>>> from neurospatial.simulation.mazes.hampton_court import (
...     make_hampton_court_maze,
...     HamptonCourtDims,
... )
>>> maze = make_hampton_court_maze()
>>> maze.env_2d.units
'cm'
>>> "goal" in maze.env_2d.regions
True
>>> "start" in maze.env_2d.regions
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
class HamptonCourtDims(MazeDims):
    """Dimension specifications for Hampton Court Maze.

    The Hampton Court Maze is a complex labyrinth with winding corridors
    and multiple dead ends. This simplified version is based on a grid
    layout with procedurally defined corridors.

    Attributes
    ----------
    size : float
        Overall size of the maze (width and height) in cm. Default is 300.0.
    corridor_width : float
        Width of corridors in cm. Default is 11.0.

    Examples
    --------
    >>> dims = HamptonCourtDims()
    >>> dims.size
    300.0
    >>> dims.corridor_width
    11.0

    >>> custom = HamptonCourtDims(size=400.0, corridor_width=15.0)
    >>> custom.size
    400.0
    """

    size: float = 300.0
    corridor_width: float = 11.0


def make_hampton_court_maze(
    dims: HamptonCourtDims | None = None,
    bin_size: float = 2.0,
    include_track: bool = True,
) -> MazeEnvironments:
    """Create a Hampton Court Maze environment.

    Creates a complex labyrinth maze with winding passages and multiple dead ends.
    The maze is inspired by the Hampton Court Palace hedge maze and represents
    one of the first maze designs used in rodent research.

    Parameters
    ----------
    dims : HamptonCourtDims, optional
        Maze dimensions. If None, uses default dimensions
        (300 cm size, 11 cm corridor width).
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
    The maze is centered at the origin with a simplified grid-based layout:
    - Overall size: ~300 × 300 cm (default)
    - Corridors arranged in a winding pattern from edge to center
    - Multiple dead ends to increase complexity
    - Start region at edge (bottom-left)
    - Goal region at center

    Regions:
    - start: Point at maze entrance (edge)
    - goal: Point at maze center

    Track Graph Topology:
    - Main path from start to goal with branches
    - Dead-end branches off the main path
    - Fully connected graph with single component

    Examples
    --------
    Create a default Hampton Court maze:

    >>> maze = make_hampton_court_maze()
    >>> maze.env_2d.units
    'cm'
    >>> maze.env_2d.n_bins > 0
    True

    Create with custom dimensions:

    >>> dims = HamptonCourtDims(size=400.0, corridor_width=15.0)
    >>> maze = make_hampton_court_maze(dims=dims, bin_size=4.0)
    >>> "goal" in maze.env_2d.regions
    True

    Skip track graph creation:

    >>> maze = make_hampton_court_maze(include_track=False)
    >>> maze.env_track is None
    True

    See Also
    --------
    HamptonCourtDims : Dimension specifications for Hampton Court Maze.
    """
    if dims is None:
        dims = HamptonCourtDims()

    # Create a simplified labyrinth based on a 5x5 grid
    # This captures the essence of Hampton Court (winding paths, dead ends, central goal)
    cell_size = dims.size / 5.0

    # Maze centered at origin
    half_size = dims.size / 2.0

    # Define corridors as list of (start, end) tuples
    # Creates a winding path from bottom-left to center with dead ends
    corridors = []

    # Main path: bottom-left -> center (winding route)
    # Start at bottom-left corner
    start_pos = (-half_size + cell_size * 0.5, -half_size + cell_size * 0.5)

    # Segment 1: Move right from start
    corridors.append(
        (start_pos, (-half_size + cell_size * 1.5, -half_size + cell_size * 0.5))
    )

    # Segment 2: Move up
    corridors.append(
        (
            (-half_size + cell_size * 1.5, -half_size + cell_size * 0.5),
            (-half_size + cell_size * 1.5, -half_size + cell_size * 2.5),
        )
    )

    # Segment 3: Move right
    corridors.append(
        (
            (-half_size + cell_size * 1.5, -half_size + cell_size * 2.5),
            (-half_size + cell_size * 3.5, -half_size + cell_size * 2.5),
        )
    )

    # Segment 4: Move up
    corridors.append(
        (
            (-half_size + cell_size * 3.5, -half_size + cell_size * 2.5),
            (-half_size + cell_size * 3.5, -half_size + cell_size * 4.5),
        )
    )

    # Segment 5: Move left to center
    corridors.append(
        (
            (-half_size + cell_size * 3.5, -half_size + cell_size * 4.5),
            (-half_size + cell_size * 2.5, -half_size + cell_size * 4.5),
        )
    )

    # Segment 6: Move down to center
    corridors.append(
        (
            (-half_size + cell_size * 2.5, -half_size + cell_size * 4.5),
            (-half_size + cell_size * 2.5, -half_size + cell_size * 2.5),
        )
    )

    # Dead end 1: Right branch from first vertical segment
    corridors.append(
        (
            (-half_size + cell_size * 1.5, -half_size + cell_size * 1.5),
            (-half_size + cell_size * 0.5, -half_size + cell_size * 1.5),
        )
    )

    # Dead end 2: Up branch from second horizontal segment
    corridors.append(
        (
            (-half_size + cell_size * 2.5, -half_size + cell_size * 2.5),
            (-half_size + cell_size * 2.5, -half_size + cell_size * 1.5),
        )
    )

    # Dead end 3: Left branch from second vertical segment
    corridors.append(
        (
            (-half_size + cell_size * 3.5, -half_size + cell_size * 3.5),
            (-half_size + cell_size * 4.5, -half_size + cell_size * 3.5),
        )
    )

    # Dead end 4: Right extension from top
    corridors.append(
        (
            (-half_size + cell_size * 3.5, -half_size + cell_size * 4.5),
            (-half_size + cell_size * 4.5, -half_size + cell_size * 4.5),
        )
    )

    # Create polygon for each corridor and union them
    corridor_polygons = [
        make_corridor_polygon(seg_start, seg_end, dims.corridor_width)
        for seg_start, seg_end in corridors
    ]
    labyrinth_polygon = union_polygons(corridor_polygons)

    # Create 2D environment
    env_2d = Environment.from_polygon(
        polygon=labyrinth_polygon,
        bin_size=bin_size,
        name="hampton_court_maze",
        connect_diagonal_neighbors=True,
    )
    env_2d.units = "cm"

    # Add regions
    # Start at the entrance (bottom-left)
    env_2d.regions.add("start", point=start_pos)

    # Goal at the center
    goal_pos = (0.0, 0.0)
    env_2d.regions.add("goal", point=goal_pos)

    # Create track graph if requested
    env_track = None
    if include_track:
        env_track = _create_hampton_court_track_graph(dims, bin_size)

    return MazeEnvironments(env_2d=env_2d, env_track=env_track)


def _create_hampton_court_track_graph(
    dims: HamptonCourtDims,
    bin_size: float,
) -> Environment:
    """Create the 1D linearized track graph for Hampton Court Maze.

    The track graph represents the labyrinth topology with nodes at
    junctions and edges along corridors. The graph includes the main
    path from start to goal plus dead-end branches.

    Parameters
    ----------
    dims : HamptonCourtDims
        Maze dimensions.
    bin_size : float
        Spatial bin size in cm.

    Returns
    -------
    Environment
        1D linearized environment representing the Hampton Court track.
    """
    cell_size = dims.size / 5.0
    half_size = dims.size / 2.0

    # Create track graph with nodes at key junctions
    graph = nx.Graph()

    # Define key positions (matching the corridor layout)
    positions = {
        "start": (-half_size + cell_size * 0.5, -half_size + cell_size * 0.5),
        "j1": (-half_size + cell_size * 1.5, -half_size + cell_size * 0.5),
        "j2": (-half_size + cell_size * 1.5, -half_size + cell_size * 1.5),
        "j3": (-half_size + cell_size * 1.5, -half_size + cell_size * 2.5),
        "j4": (-half_size + cell_size * 2.5, -half_size + cell_size * 2.5),
        "j5": (-half_size + cell_size * 3.5, -half_size + cell_size * 2.5),
        "j6": (-half_size + cell_size * 3.5, -half_size + cell_size * 3.5),
        "j7": (-half_size + cell_size * 3.5, -half_size + cell_size * 4.5),
        "j8": (-half_size + cell_size * 2.5, -half_size + cell_size * 4.5),
        "goal": (0.0, 0.0),
        # Dead ends
        "dead_end_1": (-half_size + cell_size * 0.5, -half_size + cell_size * 1.5),
        "dead_end_2": (-half_size + cell_size * 2.5, -half_size + cell_size * 1.5),
        "dead_end_3": (-half_size + cell_size * 4.5, -half_size + cell_size * 3.5),
        "dead_end_4": (-half_size + cell_size * 4.5, -half_size + cell_size * 4.5),
    }

    # Add nodes with positions
    for node_name, pos in positions.items():
        graph.add_node(node_name, pos=pos)

    # Add edges along corridors with distances
    edges = [
        # Main path
        ("start", "j1"),
        ("j1", "j2"),
        ("j2", "j3"),
        ("j3", "j4"),
        ("j4", "j5"),
        ("j5", "j6"),
        ("j6", "j7"),
        ("j7", "j8"),
        ("j8", "j4"),
        ("j4", "goal"),
        # Dead ends
        ("j2", "dead_end_1"),
        ("j4", "dead_end_2"),
        ("j6", "dead_end_3"),
        ("j7", "dead_end_4"),
    ]

    for node1, node2 in edges:
        pos1 = np.array(positions[node1])
        pos2 = np.array(positions[node2])
        distance = float(np.linalg.norm(pos2 - pos1))
        graph.add_edge(node1, node2, distance=distance)

    # Edge order for linearization (main path first, then branches)
    edge_order = [
        ("start", "j1"),
        ("j1", "j2"),
        ("j2", "j3"),
        ("j3", "j4"),
        ("j4", "j5"),
        ("j5", "j6"),
        ("j6", "j7"),
        ("j7", "j8"),
        ("j8", "j4"),
        ("j4", "goal"),
        # Dead-end branches
        ("j2", "dead_end_1"),
        ("j4", "dead_end_2"),
        ("j6", "dead_end_3"),
        ("j7", "dead_end_4"),
    ]

    # Create the 1D environment
    env_track = Environment.from_graph(
        graph=graph,
        edge_order=edge_order,
        edge_spacing=0.0,
        bin_size=bin_size,
        name="hampton_court_1d",
    )
    env_track.units = "cm"

    return env_track
