"""Repeated T-Maze environment for spatial navigation research.

The Repeated T-Maze consists of a horizontal spine with perpendicular T-arms
branching upward at evenly spaced junctions. This creates a comb/rake-like
structure ideal for studying sequential decision-making and repeated choice
behaviors. Animals traverse the spine and can enter/exit perpendicular arms
at multiple locations.

The maze structure facilitates:
- Multi-location navigation
- Sequential decision paradigms
- Repeated sampling of reward locations
- Spatial working memory tasks

Examples
--------
>>> from neurospatial.simulation.mazes.repeated_t import (
...     make_repeated_t_maze,
...     RepeatedTDims,
... )
>>> maze = make_repeated_t_maze()
>>> maze.env_2d.units
'cm'
>>> "junction_0" in maze.env_2d.regions
True
>>> "start" in maze.env_2d.regions
True
>>> len([r for r in maze.env_2d.regions if r.startswith("junction_")])
3
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
class RepeatedTDims(MazeDims):
    """Dimension specifications for Repeated T-Maze.

    The Repeated T-Maze consists of a horizontal spine with perpendicular
    arms branching upward at evenly spaced junctions, forming a comb/rake shape.

    Attributes
    ----------
    spine_length : float
        Length of the horizontal spine in cm. Default is 150.0.
    arm_length : float
        Length of each perpendicular arm in cm. Default is 40.0.
    n_junctions : int
        Number of T-junctions with upward arms. Default is 3.
    width : float
        Width of all corridors in cm. Default is 10.0.

    Examples
    --------
    >>> dims = RepeatedTDims()
    >>> dims.spine_length
    150.0
    >>> dims.arm_length
    40.0
    >>> dims.n_junctions
    3
    >>> dims.width
    10.0

    >>> custom = RepeatedTDims(spine_length=200.0, arm_length=50.0, n_junctions=4)
    >>> custom.spine_length
    200.0
    >>> custom.n_junctions
    4
    """

    spine_length: float = 150.0
    arm_length: float = 40.0
    n_junctions: int = 3
    width: float = 10.0


def make_repeated_t_maze(
    dims: RepeatedTDims | None = None,
    bin_size: float = 2.0,
    include_track: bool = True,
) -> MazeEnvironments:
    """Create a Repeated T-Maze environment.

    Creates a comb/rake-shaped maze with a horizontal spine and multiple
    perpendicular arms branching upward at evenly spaced junctions. Animals
    can traverse the spine and enter/exit arms at multiple locations.

    Parameters
    ----------
    dims : RepeatedTDims, optional
        Maze dimensions. If None, uses default dimensions
        (150 cm spine, 40 cm arms, 3 junctions, 10 cm width).
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
    - Spine runs horizontally from x = -spine_length/2 to x = +spine_length/2
    - Junctions are evenly spaced along spine at x = -L/2 + k*(L/(n+1))
      where L = spine_length, k = 1, 2, ..., n_junctions
    - Arms extend perpendicular (90°) upward from y = 0 to y = arm_length

    Regions:
    - start: Point at left end of spine (-spine_length/2, 0)
    - junction_i: Point at each T-junction (i = 0, 1, ..., n_junctions-1)
    - arm_i_end: Point at top of each arm (i = 0, 1, ..., n_junctions-1)

    Track Graph Topology:
    - Spine nodes connected sequentially along horizontal spine
    - Branch nodes at each junction leading to arm_end
    - All arms branch upward (perpendicular) from spine

    Examples
    --------
    Create a default Repeated T-maze:

    >>> maze = make_repeated_t_maze()
    >>> maze.env_2d.units
    'cm'
    >>> maze.env_2d.n_bins > 0
    True

    Create with custom dimensions:

    >>> dims = RepeatedTDims(spine_length=200.0, arm_length=50.0, n_junctions=4)
    >>> maze = make_repeated_t_maze(dims=dims, bin_size=4.0)
    >>> "junction_3" in maze.env_2d.regions
    True

    Skip track graph creation:

    >>> maze = make_repeated_t_maze(include_track=False)
    >>> maze.env_track is None
    True

    See Also
    --------
    RepeatedTDims : Dimension specifications for Repeated T-Maze.
    """
    if dims is None:
        dims = RepeatedTDims()

    # Calculate junction positions along spine
    # Spine runs from -spine_length/2 to +spine_length/2
    # Junctions are evenly spaced: divide spine into (n_junctions + 1) segments
    junction_spacing = dims.spine_length / (dims.n_junctions + 1)

    # Start position at left end of spine
    start_pos = (-dims.spine_length / 2, 0.0)

    # Create spine polygon (horizontal corridor)
    spine_start = (-dims.spine_length / 2, 0.0)
    spine_end = (dims.spine_length / 2, 0.0)
    spine_polygon = make_corridor_polygon(
        start=spine_start,
        end=spine_end,
        width=dims.width,
    )

    # Create arm polygons at each junction
    arm_polygons = []
    junction_positions = []
    arm_end_positions = []

    for i in range(dims.n_junctions):
        # Junction x-position: -spine_length/2 + (i+1) * spacing
        junction_x = -dims.spine_length / 2 + (i + 1) * junction_spacing
        junction_pos = (junction_x, 0.0)
        junction_positions.append(junction_pos)

        # Arm extends perpendicular (upward) from junction
        arm_end_pos = (junction_x, dims.arm_length)
        arm_end_positions.append(arm_end_pos)

        # Create arm corridor polygon
        arm_polygon = make_corridor_polygon(
            start=junction_pos,
            end=arm_end_pos,
            width=dims.width,
        )
        arm_polygons.append(arm_polygon)

    # Union all corridors into single polygon
    comb_polygon = union_polygons([spine_polygon, *arm_polygons])

    # Create 2D environment
    env_2d = Environment.from_polygon(
        polygon=comb_polygon,
        bin_size=bin_size,
        name="repeated_t_maze",
        connect_diagonal_neighbors=True,
    )
    env_2d.units = "cm"

    # Add regions for key locations
    env_2d.regions.add("start", point=start_pos)

    # Add junction regions
    for i, junction_pos in enumerate(junction_positions):
        env_2d.regions.add(f"junction_{i}", point=junction_pos)

    # Add arm end regions
    for i, arm_end_pos in enumerate(arm_end_positions):
        env_2d.regions.add(f"arm_{i}_end", point=arm_end_pos)

    # Create track graph if requested
    env_track = None
    if include_track:
        env_track = _create_repeated_t_maze_track_graph(dims, bin_size)

    return MazeEnvironments(env_2d=env_2d, env_track=env_track)


def _create_repeated_t_maze_track_graph(
    dims: RepeatedTDims,
    bin_size: float,
) -> Environment:
    """Create the 1D linearized track graph for Repeated T-Maze.

    The track graph represents the comb/rake topology with:
    - Spine nodes connected sequentially along horizontal spine
    - Branch edges from each junction to corresponding arm_end

    Parameters
    ----------
    dims : RepeatedTDims
        Maze dimensions.
    bin_size : float
        Spatial bin size in cm.

    Returns
    -------
    Environment
        1D linearized environment representing the Repeated T-maze track.
    """
    # Calculate junction positions
    junction_spacing = dims.spine_length / (dims.n_junctions + 1)

    # Create track graph
    graph = nx.Graph()

    # Start node at left end of spine
    start_pos = (-dims.spine_length / 2, 0.0)
    graph.add_node("start", pos=start_pos)

    # Add junction and arm_end nodes
    junction_nodes = []
    for i in range(dims.n_junctions):
        junction_x = -dims.spine_length / 2 + (i + 1) * junction_spacing

        # Junction node on spine
        junction_node = f"junction_{i}"
        junction_pos = (junction_x, 0.0)
        graph.add_node(junction_node, pos=junction_pos)
        junction_nodes.append(junction_node)

        # Arm end node
        arm_end_node = f"arm_{i}_end"
        arm_end_pos = (junction_x, dims.arm_length)
        graph.add_node(arm_end_node, pos=arm_end_pos)

        # Edge from junction to arm end (perpendicular branch)
        graph.add_edge(junction_node, arm_end_node, distance=dims.arm_length)

    # Add end node at right end of spine
    end_pos = (dims.spine_length / 2, 0.0)
    graph.add_node("end", pos=end_pos)

    # Connect spine nodes sequentially
    # start -> junction_0 -> junction_1 -> ... -> junction_{n-1} -> end
    prev_node = "start"
    for junction_node in junction_nodes:
        # Distance from previous node to current junction
        distance = junction_spacing
        graph.add_edge(prev_node, junction_node, distance=distance)
        prev_node = junction_node

    # Final edge from last junction to end
    graph.add_edge(prev_node, "end", distance=junction_spacing)

    # Edge order for linearization
    # Main path: start -> junction_0 -> junction_1 -> ... -> end (spine)
    # Branches: junction_i -> arm_i_end for each i
    edge_order = []

    # Spine edges
    prev_node = "start"
    for junction_node in junction_nodes:
        edge_order.append((prev_node, junction_node))
        prev_node = junction_node
    # Final spine edge to end
    edge_order.append((prev_node, "end"))

    # Arm branches
    for i in range(dims.n_junctions):
        edge_order.append((f"junction_{i}", f"arm_{i}_end"))

    # Create the 1D environment
    env_track = Environment.from_graph(
        graph=graph,
        edge_order=edge_order,
        edge_spacing=0.0,  # No spacing between edges
        bin_size=bin_size,
        name="repeated_t_maze_1d",
    )
    env_track.units = "cm"

    return env_track
