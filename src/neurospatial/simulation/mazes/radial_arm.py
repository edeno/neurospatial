"""Radial Arm Maze environment for spatial navigation research.

The Radial Arm Maze consists of a central platform with arms radiating outward,
typically used for spatial memory and foraging tasks. Animals can visit each arm
to retrieve rewards, making it ideal for studying spatial working memory and
reference memory.

Reference: Olton and Samuelson (1976) - "Remembrance of places passed:
Spatial memory in rats"

The standard configuration has 8 arms (for rats), though 6 arms are common
for mice. Each arm can be independently baited, allowing researchers to test
different memory strategies and foraging behaviors.

Examples
--------
>>> from neurospatial.simulation.mazes.radial_arm import (
...     make_radial_arm_maze,
...     RadialArmDims,
... )
>>> maze = make_radial_arm_maze()
>>> maze.env_2d.units
'cm'
>>> "center" in maze.env_2d.regions
True
>>> "arm_0" in maze.env_2d.regions
True
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neurospatial import Environment
from neurospatial.simulation.mazes._base import MazeDims, MazeEnvironments
from neurospatial.simulation.mazes._geometry import (
    make_circular_arena,
    make_corridor_polygon,
    make_star_graph,
    union_polygons,
)


@dataclass(frozen=True)
class RadialArmDims(MazeDims):
    """Dimension specifications for Radial Arm Maze.

    The Radial Arm Maze consists of a central circular platform with arms
    radiating outward at equal angular spacing. Standard configurations use
    8 arms for rats and 6 arms for mice.

    Attributes
    ----------
    center_radius : float
        Radius of the central platform in cm. Default is 15.0.
    arm_length : float
        Length of each arm from the edge of the center to the end in cm.
        Default is 50.0.
    arm_width : float
        Width of each arm corridor in cm. Default is 10.0.
    n_arms : int
        Number of arms radiating from the center. Default is 8 (standard
        for rats; use 6 for mice).

    Examples
    --------
    >>> dims = RadialArmDims()
    >>> dims.center_radius
    15.0
    >>> dims.arm_length
    50.0
    >>> dims.arm_width
    10.0
    >>> dims.n_arms
    8

    >>> custom = RadialArmDims(center_radius=20.0, arm_length=60.0, n_arms=6)
    >>> custom.center_radius
    20.0
    >>> custom.n_arms
    6
    """

    center_radius: float = 15.0
    arm_length: float = 50.0
    arm_width: float = 10.0
    n_arms: int = 8


def make_radial_arm_maze(
    dims: RadialArmDims | None = None,
    bin_size: float = 2.0,
    include_track: bool = True,
) -> MazeEnvironments:
    """Create a Radial Arm Maze environment.

    Creates a maze with a central circular platform and arms radiating outward
    at equal angular spacing. Animals start at the center and can visit each
    arm to retrieve rewards. This design is ideal for spatial working memory
    tasks.

    Parameters
    ----------
    dims : RadialArmDims, optional
        Maze dimensions. If None, uses default dimensions
        (15 cm center radius, 50 cm arms, 10 cm width, 8 arms).
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
    - Central platform is a circle of radius `center_radius`
    - Arms radiate from the edge of the center at angles: 2π * i / n_arms
    - Each arm extends `arm_length` from the center edge
    - Arm i is at angle: 2π * i / n_arms (starting from positive x-axis)

    Regions:
    - center: Point at origin (0, 0)
    - arm_0, arm_1, ..., arm_{n-1}: Points at each arm endpoint

    Track Graph Topology:
    - Star graph with center node connected to each arm endpoint
    - center -> arm_0_end: Edge along arm 0
    - center -> arm_1_end: Edge along arm 1
    - ... (and so on for all arms)

    Examples
    --------
    Create a default 8-arm radial maze:

    >>> maze = make_radial_arm_maze()
    >>> maze.env_2d.units
    'cm'
    >>> maze.env_2d.n_bins > 0
    True

    Create a 6-arm maze (common for mice):

    >>> dims = RadialArmDims(n_arms=6)
    >>> maze = make_radial_arm_maze(dims=dims, bin_size=3.0)
    >>> "arm_5" in maze.env_2d.regions
    True
    >>> "arm_6" in maze.env_2d.regions
    False

    Create with custom dimensions:

    >>> dims = RadialArmDims(center_radius=20.0, arm_length=80.0)
    >>> maze = make_radial_arm_maze(dims=dims)
    >>> "center" in maze.env_2d.regions
    True

    Skip track graph creation:

    >>> maze = make_radial_arm_maze(include_track=False)
    >>> maze.env_track is None
    True

    See Also
    --------
    RadialArmDims : Dimension specifications for Radial Arm Maze.
    """
    if dims is None:
        dims = RadialArmDims()

    # Create central platform
    center_polygon = make_circular_arena(center=(0.0, 0.0), radius=dims.center_radius)

    # Create arms radiating outward
    arm_polygons = []
    arm_endpoints = []

    for i in range(dims.n_arms):
        # Angle for this arm (in radians)
        angle = 2 * np.pi * i / dims.n_arms

        # Arm starts from center (to ensure overlap with center platform)
        start_pos = (0.0, 0.0)

        # Arm ends at center_radius + arm_length from origin
        end_x = (dims.center_radius + dims.arm_length) * np.cos(angle)
        end_y = (dims.center_radius + dims.arm_length) * np.sin(angle)
        end_pos = (end_x, end_y)

        # Create corridor polygon for this arm
        arm_polygon = make_corridor_polygon(
            start=start_pos,
            end=end_pos,
            width=dims.arm_width,
        )
        arm_polygons.append(arm_polygon)
        arm_endpoints.append(end_pos)

    # Union center and all arms into single polygon
    radial_polygon = union_polygons([center_polygon, *arm_polygons])

    # Create 2D environment
    env_2d = Environment.from_polygon(
        polygon=radial_polygon,
        bin_size=bin_size,
        name="radial_arm_maze",
        connect_diagonal_neighbors=True,
    )
    env_2d.units = "cm"

    # Add regions for center and arm endpoints
    env_2d.regions.add("center", point=(0.0, 0.0))

    for i, endpoint in enumerate(arm_endpoints):
        env_2d.regions.add(f"arm_{i}", point=endpoint)

    # Create track graph if requested
    env_track = None
    if include_track:
        env_track = _create_radial_arm_track_graph(dims, bin_size)

    return MazeEnvironments(env_2d=env_2d, env_track=env_track)


def _create_radial_arm_track_graph(
    dims: RadialArmDims,
    bin_size: float,
) -> Environment:
    """Create the 1D linearized track graph for Radial Arm Maze.

    The track graph represents the radial arm maze topology as a star graph:
    - center node at the origin
    - n_arms edge extending from center to each arm endpoint

    Parameters
    ----------
    dims : RadialArmDims
        Maze dimensions.
    bin_size : float
        Spatial bin size in cm.

    Returns
    -------
    Environment
        1D linearized environment representing the radial arm track.
    """
    # Calculate arm endpoints
    arm_endpoints = []
    for i in range(dims.n_arms):
        angle = 2 * np.pi * i / dims.n_arms
        end_x = (dims.center_radius + dims.arm_length) * np.cos(angle)
        end_y = (dims.center_radius + dims.arm_length) * np.sin(angle)
        arm_endpoints.append((end_x, end_y))

    # Create star graph using geometry helper
    graph = make_star_graph(
        center=(0.0, 0.0),
        arm_endpoints=arm_endpoints,
    )

    # Build edge order for linearization
    # Start from center, visit each arm in sequence
    edge_order = []
    for i in range(dims.n_arms):
        edge_order.append(("center", f"arm_{i}_end"))

    # Create the 1D environment
    env_track = Environment.from_graph(
        graph=graph,
        edge_order=edge_order,
        edge_spacing=0.0,  # No spacing between edges
        bin_size=bin_size,
        name="radial_arm_maze_1d",
    )
    env_track.units = "cm"

    return env_track
