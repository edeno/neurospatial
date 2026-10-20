"""Radial Arm Maze environment for spatial navigation research.

The Radial Arm Maze consists of a central platform with arms radiating outward,
typically used for spatial memory and foraging tasks. Animals can visit each arm
to retrieve rewards, making it ideal for studying spatial working memory and
reference memory.

The task was introduced by Olton & Samuelson (1976), "Remembrance of places
passed: Spatial memory in rats", J Exp Psychol Anim Behav Process 2:97-116,
doi:10.1037/0097-7403.2.2.97. Their apparatus dimensions could not be
verified, so the defaults follow the eight-arm rat maze of Zoladek & Roberts
(1978), "The sensory basis of spatial memory in the rat", Anim Learn Behav
6:77-81, doi:10.3758/BF03212006: "an elevated, eight-arm radial maze
consisting of a circular central platform, 35.5 cm in diameter, and eight
identical arms, each 76 cm long x 9 cm wide", with a food cup "fixed to its
extreme peripheral end".

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

import networkx as nx
import numpy as np

from neurospatial.environment import Environment
from neurospatial.regions import Regions
from neurospatial.simulation.mazes._base import MazeDims, MazeEnvironments
from neurospatial.simulation.mazes._geometry import (
    corridor_end_point,
    make_circular_arena,
    make_corridor_polygon,
    union_polygons,
)


@dataclass(frozen=True)
class RadialArmDims(MazeDims):
    """Dimension specifications for Radial Arm Maze.

    The Radial Arm Maze consists of a central circular platform with arms
    radiating outward at equal angular spacing. Standard configurations use
    8 arms for rats and 6 arms for mice. The defaults are the rat maze of
    Zoladek & Roberts (1978), doi:10.3758/BF03212006: a 35.5 cm central
    platform and eight arms 76 cm long and 9 cm wide.

    Attributes
    ----------
    center_radius : float
        Radius of the central platform in cm. Default is 17.75 (35.5 cm
        diameter).
    arm_length : float
        Length of each arm from the edge of the central platform to the arm's
        far end in cm. Default is 76.0.
    arm_width : float
        Width of each arm corridor in cm. Default is 9.0.
    n_arms : int
        Number of arms radiating from the center. Default is 8 (standard
        for rats; use 6 for mice).

    Examples
    --------
    >>> dims = RadialArmDims()
    >>> dims.center_radius * 2  # central platform diameter, cm
    35.5
    >>> dims.arm_length
    76.0
    >>> dims.arm_width
    9.0
    >>> dims.n_arms
    8

    >>> custom = RadialArmDims(center_radius=20.0, arm_length=60.0, n_arms=6)
    >>> custom.center_radius
    20.0
    >>> custom.n_arms
    6
    """

    center_radius: float = 17.75
    arm_length: float = 76.0
    arm_width: float = 9.0
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
        Maze dimensions. If None, uses the Zoladek & Roberts (1978) rat maze
        (35.5 cm central platform, 8 arms 76 cm long and 9 cm wide).
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
    - arm_0, arm_1, ..., arm_{n-1}: Points centred in the end square of each
      arm, half an arm width back from its far end, where the food cup is.

    Track Graph Topology:

    - Star graph with a ``center`` node joined to one end node per arm, each
      named like the arm's region (``arm_0``, ``arm_1``, ...) and placed at
      the arm's far end.

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
        arm_endpoints.append(corridor_end_point(start_pos, end_pos, dims.arm_width))

    # Union center and all arms into single polygon
    radial_polygon = union_polygons([center_polygon, *arm_polygons])

    # Create 2D environment
    env_2d = Environment.from_polygon(
        polygon=radial_polygon,
        bin_size=bin_size,
        name="radial_arm_maze",
        connect_diagonal_neighbors=True,
    )
    env_2d = env_2d.with_metadata(units="cm")

    # Add regions for center and arm endpoints
    region_builder = Regions()
    region_builder.add("center", point=(0.0, 0.0))

    for i, endpoint in enumerate(arm_endpoints):
        region_builder.add(f"arm_{i}", point=endpoint)
    env_2d = env_2d.with_regions(region_builder)

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
    - one edge from the center to each arm's far end, whose node is named
      like the arm's region (``arm_i``)

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
    # One edge from the centre to each arm's far end; end nodes share the
    # arm regions' names
    reach = dims.center_radius + dims.arm_length
    graph = nx.Graph()
    graph.add_node("center", pos=(0.0, 0.0))
    edge_order = []
    for i in range(dims.n_arms):
        angle = 2 * np.pi * i / dims.n_arms
        end_pos = (float(reach * np.cos(angle)), float(reach * np.sin(angle)))
        graph.add_node(f"arm_{i}", pos=end_pos)
        graph.add_edge("center", f"arm_{i}", distance=reach)
        edge_order.append(("center", f"arm_{i}"))

    # Create the 1D environment
    env_track = Environment.from_graph(
        graph=graph,
        edge_order=edge_order,
        edge_spacing=0.0,  # No spacing between edges
        bin_size=bin_size,
        track_width=dims.arm_width,
        name="radial_arm_maze_1d",
    )
    env_track = env_track.with_metadata(units="cm")

    return env_track
