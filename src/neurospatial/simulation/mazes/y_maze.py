"""Y-Maze environment for spatial navigation research.

The Y-Maze is a three-armed maze where arms are separated by 120° angles,
forming a Y shape. It is commonly used to study spatial working memory and
spontaneous alternation behavior. Animals typically explore all three arms,
and researchers measure whether they prefer to alternate between arms rather
than revisiting the same arm consecutively.

The standard Y-maze has three identical arms radiating from a central junction
at 60°, 180°, and 300° (or equivalently: up-right, down, up-left). In this
implementation, we use 90°, 210°, 330° (up, down-left, down-right) for cleaner
visualization.

Examples
--------
>>> from neurospatial.simulation.mazes import make_y_maze, YMazeDims
>>> maze = make_y_maze()
>>> maze.env_2d.units
'cm'
>>> "center" in maze.env_2d.regions
True
>>> len([r for r in maze.env_2d.regions if r.startswith("arm")]) == 3
True
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neurospatial import Environment
from neurospatial.simulation.mazes._base import MazeDims, MazeEnvironments
from neurospatial.simulation.mazes._geometry import (
    make_buffered_line,
    make_star_graph,
    union_polygons,
)


@dataclass(frozen=True)
class YMazeDims(MazeDims):
    """Dimension specifications for Y-Maze.

    The Y-Maze consists of three identical arms radiating from a central
    junction at 120° intervals. The arms are oriented at 90° (up), 210°
    (down-left), and 330° (down-right) from the origin.

    Attributes
    ----------
    arm_length : float
        Length of each arm in cm. Default is 50.0.
    width : float
        Width of all corridors in cm. Default is 10.0.

    Examples
    --------
    >>> dims = YMazeDims()
    >>> dims.arm_length
    50.0
    >>> dims.width
    10.0

    >>> custom = YMazeDims(arm_length=75.0, width=15.0)
    >>> custom.arm_length
    75.0
    """

    arm_length: float = 50.0
    width: float = 10.0


def make_y_maze(
    dims: YMazeDims | None = None,
    bin_size: float = 2.0,
    include_track: bool = True,
) -> MazeEnvironments:
    """Create a Y-Maze environment.

    Creates a Y-shaped maze with three identical arms radiating from a central
    junction at 120° intervals. The maze is commonly used to study spatial
    working memory and spontaneous alternation behavior.

    Parameters
    ----------
    dims : YMazeDims, optional
        Maze dimensions. If None, uses default dimensions
        (50 cm arms, 10 cm width).
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
    The maze is centered at the origin with three arms:
    - Arm 1: Points up (90°) from center to (0, arm_length)
    - Arm 2: Points down-left (210°) from center to (-0.866*arm_length, -0.5*arm_length)
    - Arm 3: Points down-right (330°) from center to (+0.866*arm_length, -0.5*arm_length)

    Regions:
    - center: Point at origin (0, 0)
    - arm1_end: Point at arm 1 endpoint (90°)
    - arm2_end: Point at arm 2 endpoint (210°)
    - arm3_end: Point at arm 3 endpoint (330°)

    Track Graph Topology:
    - center -> arm1_end: Edge along arm 1
    - center -> arm2_end: Edge along arm 2
    - center -> arm3_end: Edge along arm 3

    The corridors have rounded ends (cap_style='round') to avoid sharp corners.

    Examples
    --------
    Create a default Y-maze:

    >>> maze = make_y_maze()
    >>> maze.env_2d.units
    'cm'
    >>> maze.env_2d.n_bins > 0
    True

    Create with custom dimensions:

    >>> dims = YMazeDims(arm_length=80.0, width=15.0)
    >>> maze = make_y_maze(dims=dims, bin_size=4.0)
    >>> "center" in maze.env_2d.regions
    True

    Skip track graph creation:

    >>> maze = make_y_maze(include_track=False)
    >>> maze.env_track is None
    True

    See Also
    --------
    YMazeDims : Dimension specifications for Y-Maze.
    """
    if dims is None:
        dims = YMazeDims()

    # Arm angles (in degrees): 90° (up), 210° (down-left), 330° (down-right)
    arm_angles_deg = [90.0, 210.0, 330.0]
    arm_angles_rad = np.radians(arm_angles_deg)

    # Center position at origin
    center_pos = (0.0, 0.0)

    # Calculate arm endpoints using trigonometry
    arm_endpoints = [
        (
            dims.arm_length * np.cos(angle),
            dims.arm_length * np.sin(angle),
        )
        for angle in arm_angles_rad
    ]

    # Create corridor polygons with rounded ends
    arm_polygons = [
        make_buffered_line(
            start=center_pos,
            end=endpoint,
            width=dims.width,
        )
        for endpoint in arm_endpoints
    ]

    # Union all corridors into single polygon
    y_polygon = union_polygons(arm_polygons)

    # Create 2D environment
    env_2d = Environment.from_polygon(
        polygon=y_polygon,
        bin_size=bin_size,
        name="y_maze",
        connect_diagonal_neighbors=True,
    )
    env_2d.units = "cm"

    # Add regions for key locations
    env_2d.regions.add("center", point=center_pos)
    env_2d.regions.add("arm1_end", point=arm_endpoints[0])
    env_2d.regions.add("arm2_end", point=arm_endpoints[1])
    env_2d.regions.add("arm3_end", point=arm_endpoints[2])

    # Create track graph if requested
    env_track = None
    if include_track:
        env_track = _create_y_maze_track_graph(dims, bin_size)

    return MazeEnvironments(env_2d=env_2d, env_track=env_track)


def _create_y_maze_track_graph(
    dims: YMazeDims,
    bin_size: float,
) -> Environment:
    """Create the 1D linearized track graph for Y-Maze.

    The track graph represents the Y-maze topology with a star shape:
    three edges radiating from a central node to each arm endpoint.

    Parameters
    ----------
    dims : YMazeDims
        Maze dimensions.
    bin_size : float
        Spatial bin size in cm.

    Returns
    -------
    Environment
        1D linearized environment representing the Y-maze track.
    """
    # Arm angles (in degrees)
    arm_angles_deg = [90.0, 210.0, 330.0]
    arm_angles_rad = np.radians(arm_angles_deg)

    # Center position at origin
    center_pos = (0.0, 0.0)

    # Calculate arm endpoints
    arm_endpoints = [
        (
            dims.arm_length * np.cos(angle),
            dims.arm_length * np.sin(angle),
        )
        for angle in arm_angles_rad
    ]

    # Create star graph with center and three arms
    graph = make_star_graph(
        center=center_pos,
        arm_endpoints=arm_endpoints,
        spacing=None,  # No intermediate nodes, just direct edges
    )

    # Edge order for linearization
    # We'll linearize as: center -> arm1_end, center -> arm2_end, center -> arm3_end
    edge_order = [
        ("center", "arm_0_end"),  # Arm 1 (90°)
        ("center", "arm_1_end"),  # Arm 2 (210°)
        ("center", "arm_2_end"),  # Arm 3 (330°)
    ]

    # Create the 1D environment
    env_track = Environment.from_graph(
        graph=graph,
        edge_order=edge_order,
        edge_spacing=0.0,  # No spacing between edges
        bin_size=bin_size,
        name="y_maze_1d",
    )
    env_track.units = "cm"

    return env_track
