"""Repeated Y-Maze environment for spatial navigation research.

The Repeated Y-Maze consists of a chain of Y-junctions connected in series.
Each Y-junction has exactly three arms meeting at 120° angles. The junctions
alternate in orientation: odd junctions point upward (arms at 90°, 210°, 330°),
even junctions point downward (arms at 270°, 30°, 150°). This creates a
characteristic zigzag backbone where consecutive junctions are connected
diagonally.

This maze is used to study sequential decision-making, working memory across
multiple choice points, and the ability to learn complex spatial strategies.

Reference: Based on classic alternation maze designs

Examples
--------
>>> from neurospatial.simulation.mazes.repeated_y import (
...     make_repeated_y_maze,
...     RepeatedYDims,
... )
>>> maze = make_repeated_y_maze()
>>> maze.env_2d.units
'cm'
>>> "junction_1" in maze.env_2d.regions
True
>>> "goal" in maze.env_2d.regions
True
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from neurospatial import Environment
from neurospatial.simulation.mazes._base import MazeDims, MazeEnvironments
from neurospatial.simulation.mazes._geometry import make_buffered_line, union_polygons


@dataclass(frozen=True)
class RepeatedYDims(MazeDims):
    """Dimension specifications for Repeated Y-Maze.

    The Repeated Y-Maze consists of a chain of Y-junctions where each junction
    has exactly three arms at 120° angles. Junctions alternate orientation
    (up/down) creating a zigzag backbone. Each terminal arm ends in a small
    Y-fork with two branches.

    Attributes
    ----------
    n_junctions : int
        Number of Y-junctions in series. Default is 4.
    arm_length : float
        Length of each arm in the Y-junction in cm. Default is 25.0.
    fork_length : float
        Length of the small fork arms at each terminal in cm. Default is 12.0.
    width : float
        Width of all corridors in cm. Default is 8.0.

    Examples
    --------
    >>> dims = RepeatedYDims()
    >>> dims.n_junctions
    4
    >>> dims.arm_length
    25.0
    >>> dims.fork_length
    12.0
    >>> dims.width
    8.0

    >>> custom = RepeatedYDims(n_junctions=6, arm_length=30.0, width=10.0)
    >>> custom.n_junctions
    6
    """

    n_junctions: int = 4
    arm_length: float = 25.0
    fork_length: float = 12.0
    width: float = 8.0


def make_repeated_y_maze(
    dims: RepeatedYDims | None = None,
    bin_size: float = 2.0,
    include_track: bool = True,
) -> MazeEnvironments:
    """Create a Repeated Y-Maze environment.

    Creates a maze with a chain of Y-junctions connected in series. Each junction
    has exactly three arms meeting at 120° angles. The junctions alternate in
    orientation (up/down), creating a zigzag backbone.

    Parameters
    ----------
    dims : RepeatedYDims, optional
        Maze dimensions. If None, uses default dimensions
        (4 junctions, 25 cm arms, 8 cm width).
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
    Maze Structure:
    - Chain of Y-junctions with 120° angles between all three arms
    - Odd junctions (1, 3, ...): Y points up (arms at 90°, 210°, 330°)
    - Even junctions (2, 4, ...): Y points down (arms at 270°, 30°, 150°)
    - Junctions connected diagonally creating zigzag backbone
    - Each junction has a terminal arm ending in a small Y-fork

    Regions:
    - start: Entry point at first junction's stem
    - junction_1, junction_2, ...: Points at each Y-junction vertex
    - arm_i_fork_left, arm_i_fork_right: Endpoints of fork at each terminal
    - goal: Exit point at last junction's stem

    Track Graph Topology:
    - start -> junction_1 -> junction_2 -> ... -> junction_N -> goal
    - Each junction has a terminal branch with forked endpoints

    Examples
    --------
    Create a default repeated Y-maze:

    >>> maze = make_repeated_y_maze()
    >>> maze.env_2d.units
    'cm'
    >>> maze.env_2d.n_bins > 0
    True

    Create with custom number of junctions:

    >>> dims = RepeatedYDims(n_junctions=6, arm_length=30.0)
    >>> maze = make_repeated_y_maze(dims=dims, bin_size=4.0)
    >>> "junction_5" in maze.env_2d.regions
    True

    Skip track graph creation:

    >>> maze = make_repeated_y_maze(include_track=False)
    >>> maze.env_track is None
    True

    See Also
    --------
    RepeatedYDims : Dimension specifications for Repeated Y-Maze.
    """
    if dims is None:
        dims = RepeatedYDims()

    # Build the maze geometry
    polygons = []
    regions = {}

    # Y-junction angles (120° apart)
    # Up Y: arms at 90° (up), 210° (down-left), 330° (down-right)
    # Down Y: arms at 270° (down), 30° (up-right), 150° (up-left)
    up_angles = [np.deg2rad(90), np.deg2rad(210), np.deg2rad(330)]
    down_angles = [np.deg2rad(270), np.deg2rad(30), np.deg2rad(150)]

    # Calculate junction positions such that arm tips of adjacent Y's meet
    # Up Y's 330° arm tip must meet Down Y's 150° arm tip
    # Up Y at (x1, y1): 330° tip at (x1 + L*cos(330°), y1 + L*sin(330°))
    # Down Y at (x2, y2): 150° tip at (x2 + L*cos(150°), y2 + L*sin(150°))
    # For tips to meet: x2 - x1 = 2*L*cos(30°), y1 - y2 = L (since sin(330°)=-0.5, sin(150°)=+0.5)
    h_spacing = 2 * dims.arm_length * np.cos(np.deg2rad(30))  # ≈ 1.732 * L

    junction_positions = []
    for i in range(dims.n_junctions):
        x = i * h_spacing
        # Up-Y (even indices) at y = arm_length, Down-Y (odd indices) at y = 0
        # This ensures 330° tip of Up-Y meets 150° tip of Down-Y
        y = dims.arm_length if i % 2 == 0 else 0.0
        junction_positions.append(np.array([x, y]))

    # Create corridors for each junction - draw all 3 arms
    for junction_idx, junction_pos in enumerate(junction_positions):
        is_up = junction_idx % 2 == 0
        angles = up_angles if is_up else down_angles

        # Calculate all 3 arm endpoints
        arm_ends = []
        for angle in angles:
            end = junction_pos + dims.arm_length * np.array(
                [np.cos(angle), np.sin(angle)]
            )
            arm_ends.append(end)

        # Draw all 3 arms from junction center to arm tips
        for _arm_idx, arm_end in enumerate(arm_ends):
            polygons.append(
                make_buffered_line(
                    start=tuple(junction_pos),
                    end=tuple(arm_end),
                    width=dims.width,
                )
            )

        # Arm 0 is always the terminal (up for up-Y, down for down-Y)
        # Add a Y-fork at the terminal arm tip
        terminal_tip = arm_ends[0]
        terminal_angle = angles[0]

        # Fork arms are at ±60° from the terminal arm direction
        fork_angle_left = terminal_angle + np.deg2rad(60)
        fork_angle_right = terminal_angle - np.deg2rad(60)

        fork_left_end = terminal_tip + dims.fork_length * np.array(
            [np.cos(fork_angle_left), np.sin(fork_angle_left)]
        )
        fork_right_end = terminal_tip + dims.fork_length * np.array(
            [np.cos(fork_angle_right), np.sin(fork_angle_right)]
        )

        # Draw the fork arms
        polygons.append(
            make_buffered_line(
                start=tuple(terminal_tip),
                end=tuple(fork_left_end),
                width=dims.width,
            )
        )
        polygons.append(
            make_buffered_line(
                start=tuple(terminal_tip),
                end=tuple(fork_right_end),
                width=dims.width,
            )
        )

        # Store fork endpoints as regions
        regions[f"arm_{junction_idx + 1}_fork_left"] = fork_left_end.copy()
        regions[f"arm_{junction_idx + 1}_fork_right"] = fork_right_end.copy()

        # For first junction, arm 1 (210° for up-Y) is the start
        if junction_idx == 0:
            regions["start"] = arm_ends[1].copy()

        # For last junction, the forward arm is the goal
        if junction_idx == dims.n_junctions - 1:
            if is_up:
                regions["goal"] = arm_ends[2].copy()  # 330° arm
            else:
                regions["goal"] = arm_ends[1].copy()  # 30° arm

        # Store junction center position
        regions[f"junction_{junction_idx + 1}"] = junction_pos.copy()

    # Union all corridors into single polygon
    maze_polygon = union_polygons(polygons)

    # Create 2D environment
    env_2d = Environment.from_polygon(
        polygon=maze_polygon,
        bin_size=bin_size,
        name="repeated_y_maze",
        connect_diagonal_neighbors=True,
    )
    env_2d.units = "cm"

    # Add regions
    for region_name, region_pos in regions.items():
        env_2d.regions.add(region_name, point=tuple(region_pos))

    # Create track graph if requested
    env_track = None
    if include_track:
        env_track = _create_repeated_y_track_graph(dims, bin_size)

    return MazeEnvironments(env_2d=env_2d, env_track=env_track)


def _create_repeated_y_track_graph(
    dims: RepeatedYDims,
    bin_size: float,
) -> Environment:
    """Create the 1D linearized track graph for Repeated Y-Maze.

    The track graph represents the maze topology as a chain of Y-junctions
    with terminal branches at each junction.

    Parameters
    ----------
    dims : RepeatedYDims
        Maze dimensions.
    bin_size : float
        Spatial bin size in cm.

    Returns
    -------
    Environment
        1D linearized environment representing the maze track.
    """
    graph = nx.Graph()

    # Y-junction angles (120° apart)
    up_angles = [np.deg2rad(90), np.deg2rad(210), np.deg2rad(330)]
    down_angles = [np.deg2rad(270), np.deg2rad(30), np.deg2rad(150)]

    # Calculate junction positions (same as 2D geometry)
    h_spacing = 2 * dims.arm_length * np.cos(np.deg2rad(30))

    junction_positions = []
    for i in range(dims.n_junctions):
        x = i * h_spacing
        y = dims.arm_length if i % 2 == 0 else 0.0
        junction_positions.append((x, y))

    # Add junction nodes
    for junction_idx, junction_pos in enumerate(junction_positions):
        junction_name = f"junction_{junction_idx + 1}"
        graph.add_node(junction_name, pos=junction_pos)

        # Add terminal arm with fork at the tip
        is_up = junction_idx % 2 == 0
        angles = up_angles if is_up else down_angles
        terminal_angle = angles[0]
        terminal_tip = (
            junction_pos[0] + dims.arm_length * np.cos(terminal_angle),
            junction_pos[1] + dims.arm_length * np.sin(terminal_angle),
        )

        # Terminal tip node (where fork branches)
        terminal_tip_name = f"arm_{junction_idx + 1}_tip"
        graph.add_node(terminal_tip_name, pos=terminal_tip)
        graph.add_edge(junction_name, terminal_tip_name, distance=dims.arm_length)

        # Fork arms at ±60° from terminal direction
        fork_angle_left = terminal_angle + np.deg2rad(60)
        fork_angle_right = terminal_angle - np.deg2rad(60)

        fork_left_end = (
            terminal_tip[0] + dims.fork_length * np.cos(fork_angle_left),
            terminal_tip[1] + dims.fork_length * np.sin(fork_angle_left),
        )
        fork_right_end = (
            terminal_tip[0] + dims.fork_length * np.cos(fork_angle_right),
            terminal_tip[1] + dims.fork_length * np.sin(fork_angle_right),
        )

        fork_left_name = f"arm_{junction_idx + 1}_fork_left"
        fork_right_name = f"arm_{junction_idx + 1}_fork_right"

        graph.add_node(fork_left_name, pos=fork_left_end)
        graph.add_node(fork_right_name, pos=fork_right_end)
        graph.add_edge(terminal_tip_name, fork_left_name, distance=dims.fork_length)
        graph.add_edge(terminal_tip_name, fork_right_name, distance=dims.fork_length)

    # Add start and goal nodes
    first_pos = junction_positions[0]
    start_angle = up_angles[1]  # 210° for first (up) junction
    start_pos = (
        first_pos[0] + dims.arm_length * np.cos(start_angle),
        first_pos[1] + dims.arm_length * np.sin(start_angle),
    )
    graph.add_node("start", pos=start_pos)
    graph.add_edge("start", "junction_1", distance=dims.arm_length)

    last_pos = junction_positions[-1]
    is_last_up = (dims.n_junctions - 1) % 2 == 0
    goal_angle = up_angles[2] if is_last_up else down_angles[1]  # 330° or 30°
    goal_pos = (
        last_pos[0] + dims.arm_length * np.cos(goal_angle),
        last_pos[1] + dims.arm_length * np.sin(goal_angle),
    )
    graph.add_node("goal", pos=goal_pos)
    graph.add_edge(f"junction_{dims.n_junctions}", "goal", distance=dims.arm_length)

    # Connect consecutive junctions
    for i in range(dims.n_junctions - 1):
        pos1 = np.array(junction_positions[i])
        pos2 = np.array(junction_positions[i + 1])
        distance = float(np.linalg.norm(pos2 - pos1))
        graph.add_edge(f"junction_{i + 1}", f"junction_{i + 2}", distance=distance)

    # Define edge order for linearization
    edge_order = []

    # Main path: start -> junctions -> goal
    edge_order.append(("start", "junction_1"))
    for i in range(1, dims.n_junctions):
        edge_order.append((f"junction_{i}", f"junction_{i + 1}"))
    edge_order.append((f"junction_{dims.n_junctions}", "goal"))

    # Terminal arms with forks
    for i in range(1, dims.n_junctions + 1):
        edge_order.append((f"junction_{i}", f"arm_{i}_tip"))
        edge_order.append((f"arm_{i}_tip", f"arm_{i}_fork_left"))
        edge_order.append((f"arm_{i}_tip", f"arm_{i}_fork_right"))

    # Create the 1D environment
    env_track = Environment.from_graph(
        graph=graph,
        edge_order=edge_order,
        edge_spacing=0.0,
        bin_size=bin_size,
        name="repeated_y_maze_1d",
    )
    env_track.units = "cm"

    return env_track
