"""Y-Maze environment for spatial navigation research.

The Y-Maze is a three-armed maze where arms are separated by 120° angles,
forming a Y shape. It is commonly used to study spatial working memory and
spontaneous alternation behavior. Animals typically explore all three arms,
and researchers measure whether they prefer to alternate between arms rather
than revisiting the same arm consecutively.

The arms point at 270° (down), 30° (up-right) and 150° (up-left), so the
single arm points down as in the Y-maze drawn in Wijnen et al. (2024),
Brain Struct Funct 229:823-841, doi:10.1007/s00429-024-02771-x, Fig. 1a.
Y-maze sizes vary between labs; the 50 cm arms and 10 cm width are
figure-consistent defaults, not a published apparatus.

The maze has no landmarks. Wijnen et al. (2024) write that "the Y-maze is
most often used for experiments on continuous SAB [spontaneous alternation
behaviour]", "meaning that the animal is not removed from the maze after it
has visited one arm but instead is able to continue visiting the other arms
without interference of an experimenter". They trace spontaneous
alternation to "the animals' innate explorative nature", so the task needs
no reward: the arms have no reward ports, there is no start box, and the arm
ends are plain regions. Add a reward-port landmark at an arm end for a
rewarded Y-maze task.

The arms are walled (``arena.outline == "wall"``): Wijnen et al. (2024) note
that in a Y-maze "an animal could continue its trajectory into one of the
arms if it is running alongside a wall". They do not give the wall material;
the walls are modeled as opaque.

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

import networkx as nx
import numpy as np

from neurospatial.environment import Environment
from neurospatial.regions import Regions
from neurospatial.simulation.mazes._base import (
    MazeDims,
    MazeEnvironments,
    check_lengths,
    check_longer_than_wide,
)
from neurospatial.simulation.mazes._geometry import (
    corridor_end_point,
    make_corridor_polygon,
    single_floor,
    union_polygons,
)

# Arm directions in degrees, counterclockwise from +x: down, up-right, up-left.
_ARM_ANGLES_DEG = (270.0, 30.0, 150.0)


@dataclass(frozen=True)
class YMazeDims(MazeDims):
    """Dimension specifications for Y-Maze.

    The Y-Maze consists of three identical arms radiating from a central
    junction at 120° intervals. The arms point at 270° (down), 30°
    (up-right) and 150° (up-left).

    Attributes
    ----------
    arm_length : float
        Length of each arm's centreline in cm, from the junction centre,
        where the arms' centrelines meet, to the arm's far end wall, as in
        the T, plus and W mazes. Default is 50.0.
    corridor_width : float
        Width of all corridors in cm. Default is 10.0.

    Examples
    --------
    >>> dims = YMazeDims()
    >>> dims.arm_length
    50.0
    >>> dims.corridor_width
    10.0

    >>> custom = YMazeDims(arm_length=75.0, corridor_width=15.0)
    >>> custom.arm_length
    75.0
    """

    arm_length: float = 50.0
    corridor_width: float = 10.0

    def __post_init__(self) -> None:
        check_lengths(self, "arm_length", "corridor_width")
        check_longer_than_wide(self, "arm_length", "arm")


def _arm_endpoints(dims: YMazeDims) -> list[tuple[float, float]]:
    """Far end of each arm's centreline, in arm order."""
    angles = np.radians(_ARM_ANGLES_DEG)
    return [
        (float(dims.arm_length * np.cos(a)), float(dims.arm_length * np.sin(a)))
        for a in angles
    ]


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
    The maze is centered at the origin with three square-ended arms, each
    reaching ``arm_length`` from the centre:

    - Arm 1: Points down (270°) to (0, -arm_length)
    - Arm 2: Points up-right (30°) to (+0.866*arm_length, +0.5*arm_length)
    - Arm 3: Points up-left (150°) to (-0.866*arm_length, +0.5*arm_length)

    Regions:

    - center: Point at origin (0, 0)
    - arm1_end, arm2_end, arm3_end: Points centred in the end square of each
      arm, half a corridor width back from its far end.

    Track Graph Topology:

    - center -> arm1_end, center -> arm2_end, center -> arm3_end: one edge
      along each arm, ending at the arm's far end.

    Examples
    --------
    Create a default Y-maze:

    >>> maze = make_y_maze()
    >>> maze.env_2d.units
    'cm'
    >>> maze.env_2d.n_bins > 0
    True

    Create with custom dimensions:

    >>> dims = YMazeDims(arm_length=80.0, corridor_width=15.0)
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

    center_pos = (0.0, 0.0)
    arm_endpoints = _arm_endpoints(dims)

    # Square-ended corridors, so each arm ends exactly arm_length from centre
    arm_polygons = [
        make_corridor_polygon(start=center_pos, end=endpoint, width=dims.corridor_width)
        for endpoint in arm_endpoints
    ]

    # Union all corridors into single polygon
    y_polygon = single_floor(union_polygons(arm_polygons), dims)

    # Create 2D environment
    env_2d = Environment.from_polygon(
        polygon=y_polygon,
        bin_size=bin_size,
        name="y_maze",
        connect_diagonal_neighbors=True,
    )
    env_2d = env_2d.with_metadata(units="cm")

    # Add regions for key locations
    region_builder = Regions()
    region_builder.add("center", point=center_pos)
    for i, endpoint in enumerate(arm_endpoints, start=1):
        region_builder.add(
            f"arm{i}_end",
            point=corridor_end_point(center_pos, endpoint, dims.corridor_width),
        )
    env_2d = env_2d.with_regions(region_builder)

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

    The track graph is a star: one edge from the ``center`` node to each arm's
    far end, with end nodes named like the arm-end regions
    (``arm1_end``, ``arm2_end``, ``arm3_end``).

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
    graph = nx.Graph()
    graph.add_node("center", pos=(0.0, 0.0))
    edge_order = []
    for i, endpoint in enumerate(_arm_endpoints(dims), start=1):
        graph.add_node(f"arm{i}_end", pos=endpoint)
        graph.add_edge("center", f"arm{i}_end", distance=dims.arm_length)
        edge_order.append(("center", f"arm{i}_end"))

    # Create the 1D environment
    env_track = Environment.from_graph(
        graph=graph,
        edge_order=edge_order,
        edge_spacing=0.0,  # No spacing between edges
        bin_size=bin_size,
        track_width=dims.corridor_width,
        name="y_maze_1d",
    )
    env_track = env_track.with_metadata(units="cm")

    return env_track
