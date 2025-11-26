"""Hamlet Maze environment for spatial navigation research.

The Hamlet Maze features a pentagonal ring center with 5 radiating arms,
each splitting into 2 terminal goal boxes (10 total goals). This complex
maze design is used to study goal-directed navigation and flexible routing
in multiple-target environments.

Reference: Crouzier et al. 2018 (bioRxiv), Figure 1

"The Hamlet maze consists of a pentagonal center with 5 radiating arms,
each ending in a bifurcation leading to two terminal reward boxes"

Examples
--------
>>> from neurospatial.simulation.mazes.hamlet import make_hamlet_maze, HamletDims
>>> maze = make_hamlet_maze()
>>> maze.env_2d.units
'cm'
>>> "ring_0" in maze.env_2d.regions
True
>>> "goal_0" in maze.env_2d.regions
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
class HamletDims(MazeDims):
    """Dimension specifications for Hamlet Maze.

    The Hamlet Maze consists of a pentagonal ring center with 5 radiating arms,
    each splitting into 2 terminal goal boxes. This creates 10 distinct goal
    locations arranged around a central pentagon.

    Attributes
    ----------
    central_radius : float
        Radius of the pentagonal ring in cm. Default is 30.0.
    arm_length : float
        Length of each radiating arm from ring to fork point in cm. Default is 40.0.
    corridor_width : float
        Width of all corridors in cm. Default is 10.0.
    n_peripheral_arms : int
        Number of radiating arms (default: 5 for pentagon).

    Examples
    --------
    >>> dims = HamletDims()
    >>> dims.central_radius
    30.0
    >>> dims.arm_length
    40.0
    >>> dims.corridor_width
    10.0
    >>> dims.n_peripheral_arms
    5

    >>> custom = HamletDims(central_radius=40.0, arm_length=60.0, corridor_width=15.0)
    >>> custom.central_radius
    40.0
    """

    central_radius: float = 30.0
    arm_length: float = 40.0
    corridor_width: float = 10.0
    n_peripheral_arms: int = 5


def make_hamlet_maze(
    dims: HamletDims | None = None,
    bin_size: float = 2.0,
    include_track: bool = True,
) -> MazeEnvironments:
    """Create a Hamlet Maze environment.

    Creates a pentagonal ring with 5 radiating arms, each splitting into 2
    terminal goal boxes. This results in 10 distinct goal locations arranged
    around a central pentagon. The maze is ideal for studying goal-directed
    navigation with multiple targets.

    Parameters
    ----------
    dims : HamletDims, optional
        Maze dimensions. If None, uses default dimensions
        (30 cm central radius, 40 cm arms, 10 cm width, 5 arms).
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
    - Pentagon ring vertices at distance `central_radius` from origin
    - Arms radiate outward from ring vertices
    - Each arm splits into 2 goal boxes at fork point
    - Total of 10 goal regions (2 per arm)

    Regions:
    - ring_0 to ring_4: Pentagon vertices (5 regions)
    - goal_0 to goal_9: Terminal goal boxes (10 regions)
      - Goals 0-1 from arm 0 (left/right fork)
      - Goals 2-3 from arm 1
      - Goals 4-5 from arm 2
      - Goals 6-7 from arm 3
      - Goals 8-9 from arm 4

    Track Graph Topology:
    - Pentagon ring: ring_0 -> ring_1 -> ring_2 -> ring_3 -> ring_4 -> ring_0
    - Radiating arms: ring_i -> arm_i_end
    - Goal forks: arm_i_end -> goal_{2*i}, arm_i_end -> goal_{2*i+1}

    Examples
    --------
    Create a default Hamlet maze:

    >>> maze = make_hamlet_maze()
    >>> maze.env_2d.units
    'cm'
    >>> maze.env_2d.n_bins > 0
    True

    Create with custom dimensions:

    >>> dims = HamletDims(central_radius=40.0, arm_length=60.0)
    >>> maze = make_hamlet_maze(dims=dims, bin_size=4.0)
    >>> "ring_0" in maze.env_2d.regions
    True
    >>> "goal_9" in maze.env_2d.regions
    True

    Skip track graph creation:

    >>> maze = make_hamlet_maze(include_track=False)
    >>> maze.env_track is None
    True

    See Also
    --------
    HamletDims : Dimension specifications for Hamlet Maze.
    """
    if dims is None:
        dims = HamletDims()

    # Pentagon vertices (72° apart = 360°/5)
    n_arms = dims.n_peripheral_arms
    ring_positions = []
    for i in range(n_arms):
        angle = 2 * np.pi * i / n_arms + np.pi / 2  # Start from top
        x = dims.central_radius * np.cos(angle)
        y = dims.central_radius * np.sin(angle)
        ring_positions.append((x, y))

    # Create pentagon ring corridors (connect adjacent vertices)
    ring_corridors = []
    for i in range(n_arms):
        start = ring_positions[i]
        end = ring_positions[(i + 1) % n_arms]
        ring_corridors.append(make_corridor_polygon(start, end, dims.corridor_width))

    # Create radiating arms from each ring vertex
    arm_corridors = []
    arm_ends = []  # Store arm end positions for forking
    for i, ring_pos in enumerate(ring_positions):
        # Arm direction: outward from center through ring vertex
        angle = 2 * np.pi * i / n_arms + np.pi / 2
        arm_end_x = ring_pos[0] + dims.arm_length * np.cos(angle)
        arm_end_y = ring_pos[1] + dims.arm_length * np.sin(angle)
        arm_end = (arm_end_x, arm_end_y)
        arm_ends.append(arm_end)
        arm_corridors.append(
            make_corridor_polygon(ring_pos, arm_end, dims.corridor_width)
        )

    # Create forked goal boxes at end of each arm
    # Fork angle: ±30° from arm direction
    fork_length = dims.corridor_width * 2  # Short fork
    goal_corridors = []
    goal_positions = []
    for i, arm_end in enumerate(arm_ends):
        arm_angle = 2 * np.pi * i / n_arms + np.pi / 2
        for fork_dir in [-1, 1]:  # Left and right fork
            fork_angle = arm_angle + fork_dir * np.pi / 6  # ±30°
            goal_x = arm_end[0] + fork_length * np.cos(fork_angle)
            goal_y = arm_end[1] + fork_length * np.sin(fork_angle)
            goal_pos = (goal_x, goal_y)
            goal_positions.append(goal_pos)
            goal_corridors.append(
                make_corridor_polygon(arm_end, goal_pos, dims.corridor_width)
            )

    # Union all corridors into single polygon
    hamlet_polygon = union_polygons(ring_corridors + arm_corridors + goal_corridors)

    # Create 2D environment
    env_2d = Environment.from_polygon(
        polygon=hamlet_polygon,
        bin_size=bin_size,
        name="hamlet_maze",
        connect_diagonal_neighbors=True,
    )
    env_2d.units = "cm"

    # Add regions for ring vertices
    for i, ring_pos in enumerate(ring_positions):
        env_2d.regions.add(f"ring_{i}", point=ring_pos)

    # Add regions for goal positions
    for i, goal_pos in enumerate(goal_positions):
        env_2d.regions.add(f"goal_{i}", point=goal_pos)

    # Create track graph if requested
    env_track = None
    if include_track:
        env_track = _create_hamlet_track_graph(dims, bin_size)

    return MazeEnvironments(env_2d=env_2d, env_track=env_track)


def _create_hamlet_track_graph(
    dims: HamletDims,
    bin_size: float,
) -> Environment:
    """Create the 1D linearized track graph for Hamlet Maze.

    The track graph represents the Hamlet maze topology:
    - Pentagon ring with edges connecting adjacent vertices
    - Radiating arms from each ring vertex
    - Fork edges from arm ends to goal positions

    Parameters
    ----------
    dims : HamletDims
        Maze dimensions.
    bin_size : float
        Spatial bin size in cm.

    Returns
    -------
    Environment
        1D linearized environment representing the Hamlet maze track.
    """
    n_arms = dims.n_peripheral_arms

    # Pentagon vertices
    ring_positions = []
    for i in range(n_arms):
        angle = 2 * np.pi * i / n_arms + np.pi / 2
        x = dims.central_radius * np.cos(angle)
        y = dims.central_radius * np.sin(angle)
        ring_positions.append((x, y))

    # Arm end positions
    arm_ends = []
    for i, ring_pos in enumerate(ring_positions):
        angle = 2 * np.pi * i / n_arms + np.pi / 2
        arm_end_x = ring_pos[0] + dims.arm_length * np.cos(angle)
        arm_end_y = ring_pos[1] + dims.arm_length * np.sin(angle)
        arm_ends.append((arm_end_x, arm_end_y))

    # Goal positions (2 per arm)
    fork_length = dims.corridor_width * 2
    goal_positions = []
    for i, arm_end in enumerate(arm_ends):
        arm_angle = 2 * np.pi * i / n_arms + np.pi / 2
        for fork_dir in [-1, 1]:  # Left and right fork
            fork_angle = arm_angle + fork_dir * np.pi / 6
            goal_x = arm_end[0] + fork_length * np.cos(fork_angle)
            goal_y = arm_end[1] + fork_length * np.sin(fork_angle)
            goal_positions.append((goal_x, goal_y))

    # Create track graph
    graph = nx.Graph()

    # Add ring nodes
    for i, ring_pos in enumerate(ring_positions):
        graph.add_node(f"ring_{i}", pos=ring_pos)

    # Add arm end nodes
    for i, arm_end in enumerate(arm_ends):
        graph.add_node(f"arm_{i}_end", pos=arm_end)

    # Add goal nodes
    for i, goal_pos in enumerate(goal_positions):
        graph.add_node(f"goal_{i}", pos=goal_pos)

    # Add pentagon ring edges (connect adjacent ring vertices)
    for i in range(n_arms):
        pos1 = np.array(ring_positions[i])
        pos2 = np.array(ring_positions[(i + 1) % n_arms])
        distance = np.linalg.norm(pos2 - pos1)
        graph.add_edge(f"ring_{i}", f"ring_{(i + 1) % n_arms}", distance=distance)

    # Add arm edges (ring vertex to arm end)
    for i in range(n_arms):
        pos1 = np.array(ring_positions[i])
        pos2 = np.array(arm_ends[i])
        distance = np.linalg.norm(pos2 - pos1)
        graph.add_edge(f"ring_{i}", f"arm_{i}_end", distance=distance)

    # Add fork edges (arm end to goals)
    for i in range(n_arms):
        arm_end_pos = np.array(arm_ends[i])
        # Left fork
        goal_left_idx = 2 * i
        goal_left_pos = np.array(goal_positions[goal_left_idx])
        distance_left = np.linalg.norm(goal_left_pos - arm_end_pos)
        graph.add_edge(f"arm_{i}_end", f"goal_{goal_left_idx}", distance=distance_left)

        # Right fork
        goal_right_idx = 2 * i + 1
        goal_right_pos = np.array(goal_positions[goal_right_idx])
        distance_right = np.linalg.norm(goal_right_pos - arm_end_pos)
        graph.add_edge(
            f"arm_{i}_end", f"goal_{goal_right_idx}", distance=distance_right
        )

    # Edge order for linearization
    # Start with pentagon ring, then each arm with its forks
    edge_order = []

    # Pentagon ring edges
    for i in range(n_arms):
        edge_order.append((f"ring_{i}", f"ring_{(i + 1) % n_arms}"))

    # Arm and fork edges
    for i in range(n_arms):
        edge_order.append((f"ring_{i}", f"arm_{i}_end"))
        edge_order.append((f"arm_{i}_end", f"goal_{2 * i}"))
        edge_order.append((f"arm_{i}_end", f"goal_{2 * i + 1}"))

    # Create the 1D environment
    env_track = Environment.from_graph(
        graph=graph,
        edge_order=edge_order,
        edge_spacing=0.0,  # No spacing between edges
        bin_size=bin_size,
        name="hamlet_maze_1d",
    )
    env_track.units = "cm"

    return env_track
