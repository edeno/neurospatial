"""Hampton Court Maze environment for spatial navigation research.

The Hampton Court Maze is a complex labyrinth inspired by the famous hedge maze at
Hampton Court Palace in England. It is one of the first maze designs used in rodent
research (Small, 1901).

This implementation creates a labyrinth structure with:
- A central rectangular goal area
- Multiple concentric corridor rings wrapping around the center
- Dead-end branches that increase navigational complexity
- Entry from the outside, goal in the center

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

    The Hampton Court Maze is a complex labyrinth with a central goal area
    surrounded by concentric corridor rings with multiple dead ends.

    Attributes
    ----------
    size : float
        Overall size of the maze (width and height) in cm. Default is 300.0.
    corridor_width : float
        Width of corridors in cm. Default is 11.0.
    n_rings : int
        Number of concentric corridor rings around the center. Default is 3.
    center_size : float
        Size of the central goal area in cm. Default is 60.0.

    Examples
    --------
    >>> dims = HamptonCourtDims()
    >>> dims.size
    300.0
    >>> dims.corridor_width
    11.0
    >>> dims.n_rings
    3
    >>> dims.center_size
    60.0

    >>> custom = HamptonCourtDims(size=400.0, corridor_width=15.0)
    >>> custom.size
    400.0
    """

    size: float = 300.0
    corridor_width: float = 11.0
    n_rings: int = 3
    center_size: float = 60.0


def make_hampton_court_maze(
    dims: HamptonCourtDims | None = None,
    bin_size: float = 2.0,
    include_track: bool = True,
) -> MazeEnvironments:
    """Create a Hampton Court Maze environment.

    Creates a complex labyrinth maze with concentric corridor rings around a
    central goal area. The maze is inspired by the Hampton Court Palace hedge
    maze and represents one of the first maze designs used in rodent research.

    Parameters
    ----------
    dims : HamptonCourtDims, optional
        Maze dimensions. If None, uses default dimensions
        (300 cm size, 11 cm corridor width, 3 rings, 60 cm center).
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
    The maze is centered at the origin with a labyrinth structure:
    - Central rectangular goal area at the center
    - Multiple concentric corridor rings wrapping around the center
    - Connections between rings create a winding path
    - Dead-end branches increase navigational complexity
    - Entry from the bottom edge

    Regions:
    - start: Point at maze entrance (bottom edge)
    - goal: Point at maze center

    Track Graph Topology:
    - Main path spiraling from start to goal
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

    half_size = dims.size / 2.0
    half_center = dims.center_size / 2.0

    # Calculate ring spacing
    ring_spacing = (half_size - half_center) / (dims.n_rings + 1)

    corridors = []

    # Define ring radii (from center outward)
    ring_radii = [half_center + (i + 1) * ring_spacing for i in range(dims.n_rings)]
    inner_r = ring_radii[0]
    mid_r = ring_radii[1] if len(ring_radii) > 1 else inner_r
    outer_r = ring_radii[-1]

    # Create a proper labyrinth structure with connected rings
    # The design creates winding paths from entry to center with dead ends

    # Entry corridor from outside leading into outer ring (this ensures connectivity)
    corridors.append(((0.0, -outer_r * 1.2), (0.0, -outer_r)))

    # === OUTERMOST RING (ring 2) ===
    # Complete rectangular ring (all sides)
    corridors.append(((-outer_r, -outer_r), (outer_r, -outer_r)))  # Bottom
    corridors.append(((outer_r, -outer_r), (outer_r, outer_r)))  # Right
    corridors.append(((-outer_r, outer_r), (outer_r, outer_r)))  # Top
    corridors.append(((-outer_r, -outer_r), (-outer_r, outer_r)))  # Left

    # Connector from outer ring corner to middle ring (left side, toward bottom)
    corridors.append(((-outer_r, -outer_r * 0.5), (-mid_r, -outer_r * 0.5)))

    # === MIDDLE RING (ring 1) ===
    # Complete rectangular ring (all sides)
    corridors.append(((-mid_r, -mid_r), (mid_r, -mid_r)))  # Bottom
    corridors.append(((mid_r, -mid_r), (mid_r, mid_r)))  # Right
    corridors.append(((-mid_r, mid_r), (mid_r, mid_r)))  # Top
    corridors.append(((-mid_r, -mid_r), (-mid_r, mid_r)))  # Left

    # Connector from middle ring to inner ring (right side, toward top)
    corridors.append(((mid_r, mid_r * 0.5), (inner_r, mid_r * 0.5)))

    # === INNERMOST RING (ring 0) ===
    # Complete rectangular ring (all sides)
    corridors.append(((-inner_r, -inner_r), (inner_r, -inner_r)))  # Bottom
    corridors.append(((inner_r, -inner_r), (inner_r, inner_r)))  # Right
    corridors.append(((-inner_r, inner_r), (inner_r, inner_r)))  # Top
    corridors.append(((-inner_r, -inner_r), (-inner_r, inner_r)))  # Left

    # Connector from inner ring to center goal (top side)
    corridors.append(((0.0, inner_r), (0.0, half_center * 0.3)))

    # === DEAD ENDS ===
    # Dead end 1: from outer ring right side going further right
    corridors.append(((outer_r, outer_r * 0.3), (outer_r * 1.2, outer_r * 0.3)))
    # Dead end 2: from middle ring bottom going down
    corridors.append(((mid_r * 0.4, -mid_r), (mid_r * 0.4, -mid_r * 1.3)))
    # Dead end 3: from inner ring left side going left
    corridors.append(((-inner_r, -inner_r * 0.3), (-inner_r * 1.4, -inner_r * 0.3)))

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
    # Start at the entry (outside the maze)
    start_pos = (0.0, -outer_r * 1.2)
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
    path spiraling from start to goal plus dead-end branches.

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
    half_size = dims.size / 2.0
    half_center = dims.center_size / 2.0
    ring_spacing = (half_size - half_center) / (dims.n_rings + 1)
    ring_radii = [half_center + (i + 1) * ring_spacing for i in range(dims.n_rings)]

    graph = nx.Graph()

    outer_r = ring_radii[-1]
    inner_r = ring_radii[0]
    mid_r = ring_radii[-2] if len(ring_radii) > 1 else inner_r

    # Define key positions matching the corridor layout
    positions = {
        "start": (0.0, -outer_r),
        # Outer ring junctions
        "outer_bl": (-outer_r, -outer_r),
        "outer_br": (outer_r, -outer_r),
        "outer_tr": (outer_r, outer_r),
        "outer_tl": (-outer_r, outer_r),
        # Middle ring junctions
        "mid_bl": (-mid_r, -mid_r),
        "mid_br": (mid_r, -mid_r),
        "mid_tr": (mid_r, mid_r),
        "mid_tl": (-mid_r, mid_r),
        # Inner ring junctions
        "inner_bl": (-inner_r, -inner_r),
        "inner_br": (inner_r, -inner_r),
        "inner_tr": (inner_r, inner_r),
        "inner_tl": (-inner_r, inner_r),
        # Goal at center
        "goal": (0.0, 0.0),
        # Dead ends
        "dead_1": (outer_r * 1.2, outer_r * 0.5),
        "dead_2": (-mid_r * 1.3, -mid_r * 0.3),
        "dead_3": (inner_r * 0.7, -inner_r * 1.3),
    }

    # Add nodes with positions
    for node_name, pos in positions.items():
        graph.add_node(node_name, pos=pos)

    # Add edges along corridors with distances
    edges = [
        # Start to outer ring
        ("start", "outer_bl"),
        ("start", "outer_br"),
        # Outer ring
        ("outer_bl", "outer_tl"),
        ("outer_tl", "outer_tr"),
        ("outer_tr", "outer_br"),
        # Outer to middle connector
        ("outer_bl", "mid_bl"),
        # Middle ring
        ("mid_bl", "mid_br"),
        ("mid_br", "mid_tr"),
        ("mid_tr", "mid_tl"),
        # Middle to inner connector
        ("mid_tr", "inner_tr"),
        # Inner ring
        ("inner_bl", "inner_br"),
        ("inner_br", "inner_tr"),
        ("inner_tr", "inner_tl"),
        # Inner to goal
        ("inner_tl", "goal"),
        # Dead ends
        ("outer_tr", "dead_1"),
        ("mid_bl", "dead_2"),
        ("inner_br", "dead_3"),
    ]

    for node1, node2 in edges:
        pos1 = np.array(positions[node1])
        pos2 = np.array(positions[node2])
        distance = float(np.linalg.norm(pos2 - pos1))
        graph.add_edge(node1, node2, distance=distance)

    # Edge order for linearization (main path first, then branches)
    edge_order = [
        ("start", "outer_bl"),
        ("outer_bl", "outer_tl"),
        ("outer_tl", "outer_tr"),
        ("outer_tr", "outer_br"),
        ("outer_br", "start"),
        ("outer_bl", "mid_bl"),
        ("mid_bl", "mid_br"),
        ("mid_br", "mid_tr"),
        ("mid_tr", "mid_tl"),
        ("mid_tr", "inner_tr"),
        ("inner_tr", "inner_br"),
        ("inner_br", "inner_bl"),
        ("inner_tr", "inner_tl"),
        ("inner_tl", "goal"),
        # Dead-end branches
        ("outer_tr", "dead_1"),
        ("mid_bl", "dead_2"),
        ("inner_br", "dead_3"),
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
