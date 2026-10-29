"""W-Maze environment for spatial navigation research.

The W-Maze (W-track) consists of 3 parallel arms joined at their bottom ends by
a horizontal base corridor, with a reward well at the end of each arm. Rats on
the W-track alternate between the centre arm and the two outer arms, which
makes it a standard task for spatial working memory and for hippocampal
replay of the paths between arms.

The default dimensions follow the Frank-lab W-track of Karlsson & Frank (2009),
Nat Neurosci 12:913-918, doi:10.1038/nn.2344: "The W-tracks were 76 by 76 cm
with 7 cm wide track sections." and "Each environment had one reward site at
the endpoint of each arm." Kim & Frank (2009), PLoS One 4:e5494,
doi:10.1371/journal.pone.0005494, describe the same reward sites as wells:
"Sweetened milk was automatically dispensed in food wells located at the
three ends of the track."

Karlsson & Frank (2009) do not say whether the W-tracks were elevated. Kim &
Frank (2009), from the same laboratory, ran rats on "an elevated linear
track", and the W-track is modeled the same way: its edges are drops
(``arena.outline == "drop"``).

Examples
--------
>>> from neurospatial.simulation.mazes.w_maze import make_w_maze, WMazeDims
>>> maze = make_w_maze()
>>> maze.env_2d.units
'cm'
>>> list(maze.env_2d.landmarks)
['well_1', 'well_2', 'well_3']
>>> maze.env_2d.landmarks["well_2"].metadata["role"]
'reward_port'
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from neurospatial.arena import OpenArena
from neurospatial.environment import Environment
from neurospatial.landmarks import Landmark, Landmarks
from neurospatial.regions import Regions
from neurospatial.simulation.mazes._base import MazeDims, MazeEnvironments
from neurospatial.simulation.mazes._geometry import (
    corridor_end_point,
    make_corridor_polygon,
    single_floor,
    union_polygons,
)


@dataclass(frozen=True)
class WMazeDims(MazeDims):
    """Dimension specifications for W-Maze.

    The W-Maze consists of 3 parallel vertical arms joined at the bottom by a
    horizontal base corridor, forming a "W" when viewed from above.

    The defaults reproduce the 76 x 76 cm W-track with 7 cm wide track sections
    of Karlsson & Frank (2009), doi:10.1038/nn.2344, reading 76 cm as the outer
    footprint: outer-arm centrelines 76 - 7 = 69 cm apart, and arms whose
    centrelines run 76 - 7 / 2 = 72.5 cm from the base centreline to the arm's
    far end.

    Attributes
    ----------
    width : float
        Distance in cm between the centrelines of the two outer arms. The
        outer footprint is ``width + corridor_width`` wide. Default is 69.0.
    height : float
        Length in cm of each arm's centreline, from the base centreline to the
        arm's far end. The outer footprint is ``height + corridor_width / 2``
        tall. Default is 72.5.
    corridor_width : float
        Width of all track sections in cm. Default is 7.0.
    n_wells : int
        Number of arms, each with one reward well at its end. Default is 3.

    Examples
    --------
    >>> dims = WMazeDims()
    >>> dims.width + dims.corridor_width  # outer footprint, cm
    76.0
    >>> dims.height + dims.corridor_width / 2
    76.0
    >>> dims.n_wells
    3

    >>> custom = WMazeDims(width=150.0, height=100.0, corridor_width=15.0)
    >>> custom.width
    150.0
    >>> custom.height
    100.0
    """

    width: float = 69.0
    height: float = 72.5
    corridor_width: float = 7.0
    n_wells: int = 3


def make_w_maze(
    dims: WMazeDims | None = None,
    bin_size: float = 2.0,
    include_track: bool = True,
) -> MazeEnvironments:
    """Create a W-Maze environment.

    Creates a W-shaped maze with 3 parallel vertical arms joined by a
    horizontal base corridor, with a reward well at the end of each arm. The
    animal runs along the base and chooses which arm to enter, which makes the
    maze suited to spatial alternation and working memory tasks.

    Parameters
    ----------
    dims : WMazeDims, optional
        Maze dimensions. If None, uses the 76 x 76 cm W-track of Karlsson &
        Frank (2009) (69 cm outer-arm spacing, 72.5 cm arms, 7 cm corridors,
        3 arms).
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
    The maze is positioned with the base centreline at y=0:

    - Arm centrelines are at x = -width/2, 0, +width/2 and run from y = 0 to
      y = height.
    - The base corridor spans the outer edges of the outer arms, from
      x = -(width + corridor_width)/2 to +(width + corridor_width)/2.

    Each food well is a landmark in ``env_2d.landmarks`` with role
    ``"reward_port"``, and a point region of the same name:

    - well_1, well_2, well_3: the wells at the ends of the arms, left to
      right, each centred in its arm's end square, half a corridor width
      back from the arm's far end. With three arms, ``well_2`` is the centre
      well.

    The regions are points because neither Karlsson & Frank (2009) nor Kim
    & Frank (2009) gives the wells' size.

    Track Graph Topology:

    - The base runs between the junctions at the bottom of the outer arms.
    - Each arm runs from its junction to its far end, a node named like the
      arm's well.
    - The graph is a tree with one leaf per arm.

    Examples
    --------
    Create a default W-maze:

    >>> maze = make_w_maze()
    >>> maze.env_2d.units
    'cm'
    >>> maze.env_2d.n_bins > 0
    True

    Create with custom dimensions:

    >>> dims = WMazeDims(width=150.0, height=100.0, corridor_width=15.0)
    >>> maze = make_w_maze(dims=dims, bin_size=4.0)
    >>> "well_1" in maze.env_2d.regions
    True

    Skip track graph creation:

    >>> maze = make_w_maze(include_track=False)
    >>> maze.env_track is None
    True

    See Also
    --------
    WMazeDims : Dimension specifications for W-Maze.
    """
    if dims is None:
        dims = WMazeDims()

    # Key positions (base centreline at y=0, arms extend upward to y=height)
    half_width = dims.width / 2.0
    half_corridor = dims.corridor_width / 2.0

    # The base corridor reaches the outer edges of the outer arms, so the
    # outer corners of the W are square.
    base_polygon = make_corridor_polygon(
        start=(-half_width - half_corridor, 0.0),
        end=(half_width + half_corridor, 0.0),
        width=dims.corridor_width,
    )

    # Vertical arms at evenly spaced x positions across the width
    arm_x_positions = np.linspace(-half_width, half_width, dims.n_wells)
    arm_polygons = [
        make_corridor_polygon(
            start=(x_pos, 0.0),
            end=(x_pos, dims.height),
            width=dims.corridor_width,
        )
        for x_pos in arm_x_positions
    ]

    # Union all corridors into single polygon
    w_polygon = single_floor(union_polygons([base_polygon, *arm_polygons]), dims)

    # Create 2D environment
    env_2d = Environment.from_polygon(
        polygon=w_polygon,
        bin_size=bin_size,
        arena=OpenArena(w_polygon, outline="drop"),
        name="w_maze",
        connect_diagonal_neighbors=True,
    )
    env_2d = env_2d.with_metadata(units="cm")

    # One food well in the end square of each arm; its size is unpublished
    region_builder = Regions()
    landmarks = []
    for i, x_pos in enumerate(arm_x_positions, start=1):
        well_pos = corridor_end_point(
            (float(x_pos), 0.0), (float(x_pos), dims.height), dims.corridor_width
        )
        region_builder.add(f"well_{i}", point=well_pos)
        landmarks.append(
            Landmark(
                f"well_{i}",
                "point",
                well_pos,
                metadata={"role": "reward_port", "port_type": "food_well"},
            )
        )
    env_2d = env_2d.with_regions(region_builder).with_landmarks(Landmarks(landmarks))

    # Create track graph if requested
    env_track = None
    if include_track:
        env_track = _create_w_maze_track_graph(dims, bin_size)

    return MazeEnvironments(env_2d=env_2d, env_track=env_track)


def _create_w_maze_track_graph(
    dims: WMazeDims,
    bin_size: float,
) -> Environment:
    """Create the 1D linearized track graph for W-Maze.

    The base runs between the junctions at the bottom of the outer arms, and
    each arm runs from its junction to its far end, so the graph is a tree
    whose only leaves are the arm ends.

    Parameters
    ----------
    dims : WMazeDims
        Maze dimensions.
    bin_size : float
        Spatial bin size in cm.

    Returns
    -------
    Environment
        1D linearized environment representing the W-maze track.
    """
    half_width = dims.width / 2.0
    arm_x_positions = np.linspace(-half_width, half_width, dims.n_wells)

    graph = nx.Graph()
    for i, x_pos in enumerate(arm_x_positions, start=1):
        graph.add_node(f"junction_{i}", pos=(float(x_pos), 0.0))
        graph.add_node(f"well_{i}", pos=(float(x_pos), dims.height))

    # Base edges between neighbouring junctions, then one edge up each arm
    edge_order = [
        (f"junction_{i}", f"junction_{i + 1}") for i in range(1, dims.n_wells)
    ] + [(f"junction_{i}", f"well_{i}") for i in range(1, dims.n_wells + 1)]
    for node1, node2 in edge_order:
        distance = np.linalg.norm(
            np.subtract(graph.nodes[node2]["pos"], graph.nodes[node1]["pos"])
        )
        graph.add_edge(node1, node2, distance=float(distance))

    # Create the 1D environment
    env_track = Environment.from_graph(
        graph=graph,
        edge_order=edge_order,
        edge_spacing=0.0,  # No spacing between edges
        bin_size=bin_size,
        track_width=dims.corridor_width,
        name="w_maze_1d",
    )
    env_track = env_track.with_metadata(units="cm")

    return env_track
