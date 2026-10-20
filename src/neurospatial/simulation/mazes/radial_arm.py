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
identical arms, each 76 cm long x 9 cm wide". "Each arm had a rectangular
metal food cup, 7.5 cm long x 5 cm wide x 2.5 cm deep, fixed to its extreme
peripheral end", and on each trial "an animal was placed on the central
platform". The maze therefore has a food cup at the end of every arm and no
start box. Zoladek & Roberts list the room's furniture but not where it
stood, so the maze has no cue landmarks.

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
>>> maze.env_2d.landmarks["food_cup_0"].metadata["role"]
'reward_port'
>>> round(maze.env_2d.regions["food_cup_0"].data.area, 1)  # 7.5 x 5 cm
37.5
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from neurospatial.environment import Environment
from neurospatial.landmarks import Landmark, Landmarks
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
    platform, eight arms 76 cm long and 9 cm wide, and a 7.5 cm x 5 cm food
    cup at the end of each arm.

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
    food_cup_length : float
        Length in cm of each arm's food cup along the arm, from the arm's far
        end inward. Default is 7.5.
    food_cup_width : float
        Width in cm of each arm's food cup across the arm. Default is 5.0.

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
    food_cup_length: float = 7.5
    food_cup_width: float = 5.0


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
        (35.5 cm central platform, 8 arms 76 cm long and 9 cm wide, a
        7.5 x 5 cm food cup at each arm's end).
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

    - center: Point at origin (0, 0), on the central platform where the rat
      is placed at the start of a trial.
    - food_cup_0, food_cup_1, ..., food_cup_{n-1}: The footprint of each
      arm's food cup, ``food_cup_length`` along the arm from its far end and
      ``food_cup_width`` across it, centred on the arm's axis.

    Each food cup is also a landmark in ``env_2d.landmarks`` with role
    ``"reward_port"``, at the centre of its footprint. Zoladek & Roberts
    give the cup as "7.5 cm long x 5 cm wide" without saying which side runs
    along the arm; the cup is laid with its length along the arm, as they
    give the arms' "76 cm long x 9 cm wide", so the 5 cm side fits across
    the 9 cm arm. With bins much larger than the cup, a cup's region may
    cover no bin centre.

    Track Graph Topology:

    - Star graph with a ``center`` node joined to one end node per arm, each
      named like the arm's food cup (``food_cup_0``, ``food_cup_1``, ...)
      and placed at the arm's far end.

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
    >>> "food_cup_5" in maze.env_2d.regions
    True
    >>> "food_cup_6" in maze.env_2d.regions
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

    # Create arms radiating outward, each with a food cup at its far end
    arm_polygons = []
    food_cups = []

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

        # The cup is fixed to the arm's far end, its length along the arm
        cup_inner_end = corridor_end_point(
            start_pos, end_pos, 2.0 * dims.food_cup_length
        )
        food_cups.append(
            (
                corridor_end_point(start_pos, end_pos, dims.food_cup_length),
                make_corridor_polygon(
                    start=cup_inner_end, end=end_pos, width=dims.food_cup_width
                ),
            )
        )

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

    # Regions for the start platform and each food cup's footprint; each cup
    # is also a reward-port landmark at its centre
    region_builder = Regions()
    region_builder.add("center", point=(0.0, 0.0))
    landmarks = []
    for i, (cup_center, cup_polygon) in enumerate(food_cups):
        region_builder.add(f"food_cup_{i}", polygon=cup_polygon)
        landmarks.append(
            Landmark(
                f"food_cup_{i}",
                "point",
                cup_center,
                metadata={"role": "reward_port", "port_type": "food_cup"},
            )
        )
    env_2d = env_2d.with_regions(region_builder).with_landmarks(Landmarks(landmarks))

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
      like the arm's food cup (``food_cup_i``)

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
    # names of the food cups there
    reach = dims.center_radius + dims.arm_length
    graph = nx.Graph()
    graph.add_node("center", pos=(0.0, 0.0))
    edge_order = []
    for i in range(dims.n_arms):
        angle = 2 * np.pi * i / dims.n_arms
        end_pos = (float(reach * np.cos(angle)), float(reach * np.sin(angle)))
        graph.add_node(f"food_cup_{i}", pos=end_pos)
        graph.add_edge("center", f"food_cup_{i}", distance=reach)
        edge_order.append(("center", f"food_cup_{i}"))

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
