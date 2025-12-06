"""Hamlet Maze environment for spatial navigation research.

The Hamlet Maze features a central hub connected to 5 inner boxes arranged in
a pentagon, which are then connected to 5 outer boxes at the vertices of an
outer pentagon. The inner boxes are also connected to each other forming an
inner ring. This creates a star-like structure with 11 distinct locations.

Reference: Crouzier et al. 2018 (bioRxiv), Figure 1

The maze consists of:
- 1 central hub
- 5 inner boxes (pentagon arrangement, connected to center and each other)
- 5 outer boxes (at outer pentagon vertices, connected to inner boxes)

Examples
--------
>>> from neurospatial.simulation.mazes.hamlet import make_hamlet_maze, HamletDims
>>> maze = make_hamlet_maze()
>>> maze.env_2d.units
'cm'
>>> len(maze.env_2d.regions) > 0  # Has regions defined
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

    The Hamlet Maze consists of a central hub connected to 5 inner boxes in a
    pentagon, which connect to 5 outer boxes. Inner boxes are also connected
    to each other forming an inner ring.

    Attributes
    ----------
    inner_radius : float
        Distance from center to inner boxes in cm. Default is 40.0.
    outer_radius : float
        Distance from center to outer boxes in cm. Default is 80.0.
    corridor_width : float
        Width of all corridors in cm. Default is 10.0.
    n_arms : int
        Number of arms/boxes in each ring (default: 5 for pentagon).

    Examples
    --------
    >>> dims = HamletDims()
    >>> dims.inner_radius
    40.0
    >>> dims.outer_radius
    80.0
    >>> dims.corridor_width
    10.0
    >>> dims.n_arms
    5

    >>> custom = HamletDims(inner_radius=50.0, outer_radius=100.0, corridor_width=15.0)
    >>> custom.inner_radius
    50.0
    """

    inner_radius: float = 40.0
    outer_radius: float = 80.0
    corridor_width: float = 10.0
    n_arms: int = 5


def make_hamlet_maze(
    dims: HamletDims | None = None,
    bin_size: float = 2.0,
    include_track: bool = True,
) -> MazeEnvironments:
    """Create a Hamlet Maze environment.

    Creates a maze with a central hub, 5 inner boxes arranged in a pentagon,
    and 5 outer boxes at the outer pentagon vertices. Inner boxes are connected
    to each other forming an inner ring. This creates 11 distinct locations.

    Parameters
    ----------
    dims : HamletDims, optional
        Maze dimensions. If None, uses default dimensions
        (40 cm inner radius, 80 cm outer radius, 10 cm width, 5 arms).
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
    - Central hub at origin
    - 5 inner boxes at `inner_radius` distance (pentagon)
    - 5 outer boxes at `outer_radius` distance (outer pentagon)
    - Inner boxes connected to center and to each other (inner ring)
    - Inner boxes connected to corresponding outer boxes

    Regions:
    - center: Central hub at origin
    - inner_0 to inner_4: Inner pentagon boxes (5 regions)
    - outer_0 to outer_4: Outer pentagon boxes (5 regions)

    Track Graph Topology:
    - Center connected to all 5 inner boxes
    - Inner boxes connected to adjacent inner boxes (inner ring)
    - Each inner box connected to corresponding outer box

    Examples
    --------
    Create a default Hamlet maze:

    >>> maze = make_hamlet_maze()
    >>> maze.env_2d.units
    'cm'
    >>> maze.env_2d.n_bins > 0
    True

    Create with custom dimensions:

    >>> dims = HamletDims(inner_radius=50.0, outer_radius=100.0)
    >>> maze = make_hamlet_maze(dims=dims, bin_size=4.0)
    >>> "center" in maze.env_2d.regions
    True
    >>> "outer_4" in maze.env_2d.regions
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

    n_arms = dims.n_arms
    corridors = []

    # Calculate inner and outer box positions (pentagon arrangement)
    # Starting from top (90°)
    inner_positions = []
    outer_positions = []
    for i in range(n_arms):
        angle = 2 * np.pi * i / n_arms + np.pi / 2  # Start from top
        inner_x = dims.inner_radius * np.cos(angle)
        inner_y = dims.inner_radius * np.sin(angle)
        inner_positions.append((inner_x, inner_y))

        outer_x = dims.outer_radius * np.cos(angle)
        outer_y = dims.outer_radius * np.sin(angle)
        outer_positions.append((outer_x, outer_y))

    # Center at origin
    center_pos = (0.0, 0.0)

    # Create corridors from center to each inner box
    for inner_pos in inner_positions:
        corridors.append(
            make_corridor_polygon(center_pos, inner_pos, dims.corridor_width)
        )

    # Create inner ring corridors (connect adjacent inner boxes)
    for i in range(n_arms):
        start = inner_positions[i]
        end = inner_positions[(i + 1) % n_arms]
        corridors.append(make_corridor_polygon(start, end, dims.corridor_width))

    # Create corridors from inner boxes to outer boxes
    for i in range(n_arms):
        inner_pos = inner_positions[i]
        outer_pos = outer_positions[i]
        corridors.append(
            make_corridor_polygon(inner_pos, outer_pos, dims.corridor_width)
        )

    # Union all corridors into single polygon
    hamlet_polygon = union_polygons(corridors)

    # Create 2D environment
    env_2d = Environment.from_polygon(
        polygon=hamlet_polygon,
        bin_size=bin_size,
        name="hamlet_maze",
        connect_diagonal_neighbors=True,
    )
    env_2d.units = "cm"

    # Add region for center
    env_2d.regions.add("center", point=center_pos)

    # Add regions for inner boxes
    for i, inner_pos in enumerate(inner_positions):
        env_2d.regions.add(f"inner_{i}", point=inner_pos)

    # Add regions for outer boxes
    for i, outer_pos in enumerate(outer_positions):
        env_2d.regions.add(f"outer_{i}", point=outer_pos)

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
    - Center connected to all 5 inner boxes
    - Inner boxes connected to adjacent inner boxes (inner ring)
    - Each inner box connected to corresponding outer box

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
    n_arms = dims.n_arms

    # Calculate positions
    inner_positions = []
    outer_positions = []
    for i in range(n_arms):
        angle = 2 * np.pi * i / n_arms + np.pi / 2
        inner_x = dims.inner_radius * np.cos(angle)
        inner_y = dims.inner_radius * np.sin(angle)
        inner_positions.append((inner_x, inner_y))

        outer_x = dims.outer_radius * np.cos(angle)
        outer_y = dims.outer_radius * np.sin(angle)
        outer_positions.append((outer_x, outer_y))

    center_pos = (0.0, 0.0)

    # Create track graph
    graph = nx.Graph()

    # Add center node
    graph.add_node("center", pos=center_pos)

    # Add inner nodes
    for i, inner_pos in enumerate(inner_positions):
        graph.add_node(f"inner_{i}", pos=inner_pos)

    # Add outer nodes
    for i, outer_pos in enumerate(outer_positions):
        graph.add_node(f"outer_{i}", pos=outer_pos)

    # Add edges from center to inner boxes
    for i in range(n_arms):
        graph.add_edge("center", f"inner_{i}", distance=dims.inner_radius)

    # Add inner ring edges (connect adjacent inner boxes)
    for i in range(n_arms):
        pos1 = np.array(inner_positions[i])
        pos2 = np.array(inner_positions[(i + 1) % n_arms])
        distance = float(np.linalg.norm(pos2 - pos1))
        graph.add_edge(f"inner_{i}", f"inner_{(i + 1) % n_arms}", distance=distance)

    # Add edges from inner to outer boxes
    outer_distance = dims.outer_radius - dims.inner_radius
    for i in range(n_arms):
        graph.add_edge(f"inner_{i}", f"outer_{i}", distance=outer_distance)

    # Edge order for linearization
    edge_order = []

    # Center to inner edges
    for i in range(n_arms):
        edge_order.append(("center", f"inner_{i}"))

    # Inner ring edges
    for i in range(n_arms):
        edge_order.append((f"inner_{i}", f"inner_{(i + 1) % n_arms}"))

    # Inner to outer edges
    for i in range(n_arms):
        edge_order.append((f"inner_{i}", f"outer_{i}"))

    # Create the 1D environment
    env_track = Environment.from_graph(
        graph=graph,
        edge_order=edge_order,
        edge_spacing=0.0,
        bin_size=bin_size,
        name="hamlet_maze_1d",
    )
    env_track.units = "cm"

    return env_track
