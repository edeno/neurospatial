"""Repeated Y-Maze environment for spatial navigation research.

The Repeated Y-Maze (Warner-Warden maze) is a series of Y-junctions where animals
must learn to consistently choose the correct arm at each decision point. A key
feature is the Warner-Warden trick: dead ends are split into two small forked
corridors so the animal cannot see that the path is blocked until committed.

This maze is used to study sequential decision-making, working memory across
multiple choice points, and the ability to learn complex spatial strategies.

Reference: Warden (1929a, 1929b)

Warner-Warden trick: "Dead ends split into two small corridors so the animal
cannot see the dead end from the junction point." This prevents animals from
using simple visual cues to solve the maze.

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

    The Repeated Y-Maze consists of a chain of Y-junctions where the animal
    must make sequential left/right decisions. Dead ends use the Warner-Warden
    trick (split into two small corridors) to prevent visual detection.

    Attributes
    ----------
    n_junctions : int
        Number of Y-junctions in series. Default is 3.
    segment_length : float
        Length of each corridor segment in cm. Default is 50.0.
    width : float
        Width of all corridors in cm. Default is 10.0.

    Examples
    --------
    >>> dims = RepeatedYDims()
    >>> dims.n_junctions
    3
    >>> dims.segment_length
    50.0
    >>> dims.width
    10.0

    >>> custom = RepeatedYDims(n_junctions=5, segment_length=40.0, width=12.0)
    >>> custom.n_junctions
    5
    """

    n_junctions: int = 3
    segment_length: float = 50.0
    width: float = 10.0


def make_repeated_y_maze(
    dims: RepeatedYDims | None = None,
    bin_size: float = 2.0,
    include_track: bool = True,
) -> MazeEnvironments:
    """Create a Repeated Y-Maze environment.

    Creates a maze with a series of Y-junctions where animals must make
    sequential decisions. Dead ends are split into forked corridors using
    the Warner-Warden trick to prevent visual detection of the dead end.

    Parameters
    ----------
    dims : RepeatedYDims, optional
        Maze dimensions. If None, uses default dimensions
        (3 junctions, 50 cm segments, 10 cm width).
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
    - Each junction branches at 120° angles (Y-shape)
    - Correct path continues forward, dead end branches to the side
    - Dead ends are forked into two small corridors (Warner-Warden trick)
    - Final junction leads to goal (no dead end)

    Regions:
    - start: Point at maze entrance
    - junction_1, junction_2, ..., junction_N: Points at each Y-junction
    - dead_1_left, dead_1_right, ...: Endpoints of forked dead ends
    - goal: Point at final endpoint

    Track Graph Topology:
    - Chain of junction nodes with branch nodes for dead ends
    - Each junction has two exits: one to next junction, one to dead end fork

    Examples
    --------
    Create a default repeated Y-maze:

    >>> maze = make_repeated_y_maze()
    >>> maze.env_2d.units
    'cm'
    >>> maze.env_2d.n_bins > 0
    True

    Create with custom number of junctions:

    >>> dims = RepeatedYDims(n_junctions=5, segment_length=40.0)
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

    # Junction angle: 120 degrees (2π/3 radians)
    # Arms split at ±60° from forward direction
    branch_angle = np.deg2rad(60)  # Deviation from straight
    fork_angle = np.deg2rad(30)  # Dead end fork splits at ±30°
    fork_length = dims.segment_length * 0.3  # Short forked corridors

    # Start position
    current_pos = np.array([0.0, 0.0])
    current_direction = np.deg2rad(90)  # Start pointing up (+y direction)

    regions["start"] = current_pos.copy()

    # Build junctions in sequence
    for junction_idx in range(1, dims.n_junctions + 1):
        # Move forward to junction position
        junction_pos = current_pos + dims.segment_length * np.array(
            [np.cos(current_direction), np.sin(current_direction)]
        )

        # Add corridor from previous position to junction
        polygons.append(
            make_buffered_line(
                start=tuple(current_pos),
                end=tuple(junction_pos),
                width=dims.width,
            )
        )

        regions[f"junction_{junction_idx}"] = junction_pos.copy()

        if junction_idx < dims.n_junctions:
            # Not the last junction - create dead end fork
            # Dead end branches to the left
            dead_direction = current_direction + branch_angle
            dead_branch_pos = junction_pos + dims.segment_length * np.array(
                [np.cos(dead_direction), np.sin(dead_direction)]
            )

            # Add corridor to dead end branch point
            polygons.append(
                make_buffered_line(
                    start=tuple(junction_pos),
                    end=tuple(dead_branch_pos),
                    width=dims.width,
                )
            )

            # Warner-Warden fork: split dead end into two corridors
            # Left fork
            left_fork_dir = dead_direction - fork_angle
            left_fork_end = dead_branch_pos + fork_length * np.array(
                [np.cos(left_fork_dir), np.sin(left_fork_dir)]
            )
            polygons.append(
                make_buffered_line(
                    start=tuple(dead_branch_pos),
                    end=tuple(left_fork_end),
                    width=dims.width,
                )
            )
            regions[f"dead_{junction_idx}_left"] = left_fork_end.copy()

            # Right fork
            right_fork_dir = dead_direction + fork_angle
            right_fork_end = dead_branch_pos + fork_length * np.array(
                [np.cos(right_fork_dir), np.sin(right_fork_dir)]
            )
            polygons.append(
                make_buffered_line(
                    start=tuple(dead_branch_pos),
                    end=tuple(right_fork_end),
                    width=dims.width,
                )
            )
            regions[f"dead_{junction_idx}_right"] = right_fork_end.copy()

            # Correct path continues forward (slight right turn)
            current_direction = current_direction - branch_angle

        else:
            # Last junction - add goal corridor
            goal_pos = junction_pos + dims.segment_length * np.array(
                [np.cos(current_direction), np.sin(current_direction)]
            )
            polygons.append(
                make_buffered_line(
                    start=tuple(junction_pos),
                    end=tuple(goal_pos),
                    width=dims.width,
                )
            )
            regions["goal"] = goal_pos.copy()

        # Update current position for next junction
        current_pos = junction_pos

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

    The track graph represents the maze topology as a chain of junctions
    with branches to dead end forks.

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

    # Build positions matching the 2D geometry
    branch_angle = np.deg2rad(60)
    fork_angle = np.deg2rad(30)
    fork_length = dims.segment_length * 0.3

    current_pos = np.array([0.0, 0.0])
    current_direction = np.deg2rad(90)  # Up

    # Add start node
    graph.add_node("start", pos=tuple(current_pos))

    prev_node = "start"

    for junction_idx in range(1, dims.n_junctions + 1):
        # Junction position
        junction_pos = current_pos + dims.segment_length * np.array(
            [np.cos(current_direction), np.sin(current_direction)]
        )
        junction_name = f"junction_{junction_idx}"
        graph.add_node(junction_name, pos=tuple(junction_pos))

        # Edge from previous node to junction
        dist = np.linalg.norm(junction_pos - current_pos)
        graph.add_edge(prev_node, junction_name, distance=dist)

        if junction_idx < dims.n_junctions:
            # Add dead end branch
            dead_direction = current_direction + branch_angle
            dead_branch_pos = junction_pos + dims.segment_length * np.array(
                [np.cos(dead_direction), np.sin(dead_direction)]
            )

            # Branch node (before fork)
            dead_branch_name = f"dead_{junction_idx}_branch"
            graph.add_node(dead_branch_name, pos=tuple(dead_branch_pos))
            graph.add_edge(
                junction_name,
                dead_branch_name,
                distance=dims.segment_length,
            )

            # Left fork endpoint
            left_fork_dir = dead_direction - fork_angle
            left_fork_end = dead_branch_pos + fork_length * np.array(
                [np.cos(left_fork_dir), np.sin(left_fork_dir)]
            )
            left_name = f"dead_{junction_idx}_left"
            graph.add_node(left_name, pos=tuple(left_fork_end))
            graph.add_edge(dead_branch_name, left_name, distance=fork_length)

            # Right fork endpoint
            right_fork_dir = dead_direction + fork_angle
            right_fork_end = dead_branch_pos + fork_length * np.array(
                [np.cos(right_fork_dir), np.sin(right_fork_dir)]
            )
            right_name = f"dead_{junction_idx}_right"
            graph.add_node(right_name, pos=tuple(right_fork_end))
            graph.add_edge(dead_branch_name, right_name, distance=fork_length)

            # Update direction for correct path
            current_direction = current_direction - branch_angle
            prev_node = junction_name

        else:
            # Last junction - add goal
            goal_pos = junction_pos + dims.segment_length * np.array(
                [np.cos(current_direction), np.sin(current_direction)]
            )
            graph.add_node("goal", pos=tuple(goal_pos))
            graph.add_edge(junction_name, "goal", distance=dims.segment_length)

        current_pos = junction_pos

    # Define edge order for linearization
    # Main path: start -> junction_1 -> ... -> junction_N -> goal
    edge_order = []
    edge_order.append(("start", "junction_1"))

    for junction_idx in range(1, dims.n_junctions):
        # Dead end branches
        dead_branch_name = f"dead_{junction_idx}_branch"
        edge_order.append((f"junction_{junction_idx}", dead_branch_name))
        edge_order.append((dead_branch_name, f"dead_{junction_idx}_left"))
        edge_order.append((dead_branch_name, f"dead_{junction_idx}_right"))

        # Next junction
        edge_order.append((f"junction_{junction_idx}", f"junction_{junction_idx + 1}"))

    # Final edge to goal
    edge_order.append((f"junction_{dims.n_junctions}", "goal"))

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
