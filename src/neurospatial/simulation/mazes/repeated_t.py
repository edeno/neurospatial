"""Repeated T-Maze: Blodgett's six-unit T maze (Tolman & Nyswander, 1927).

Tolman and Nyswander (1927) describe the maze of their Experiment VI(a):

    "Six-unit T maze with doors. This maze was devised by Mr. H. C. Blodgett,
    and a diagram of it is shown in figure 5. It will be seen that the maze
    divides up into a succession of equivalent T units."

A rat leaves the start box and meets six T-shaped choice points in turn; at
each, one arm leads on and the other is a blind alley, and the last choice
point opens onto the goal. The dimensions are those written on the figure:
alleys 4 in (10.16 cm) wide, and each alley segment 16 in (40.64 cm) long
along its walls, except the two segments that join choice points 4, 5 and 6,
which are 18 in (45.72 cm). At the drawing's scale (the 4-in width marker),
these labelled lengths run between the walls of the crossing alleys, so
neighbouring choice points are 20 in (or 22 in) apart centre to centre. The
start box, its stem and the partition where blind alley 4 ends against the
first alley are not dimensioned; their sizes are scaled from the drawing and
are estimates. The doors, closed behind the rat to stop retracing, are not
modelled.

The layout is turned a quarter turn clockwise from the figure, as drawn in
Wijnen et al. (2024, Fig. 1b): start at the left, goal at the right::

                        b3
                        |
       b2 ----- 2 ----- 3      b6 ----- 6 ----- F
                |       |               |
                |       |               |
       S ====== 1    b4 4 ------------- 5
                |                       |
                b1                      b5

Choice points 1-6 in order; b1-b6 are the ends of the blind alleys. Blind
alley 4 runs back toward alley 1 and stops a partition short of it.

References
----------
Tolman, E. C., & Nyswander, D. B. (1927). The reliability and validity of
maze-measures for rats. Journal of Comparative Psychology, 7(6), 425-460.
https://doi.org/10.1037/h0072851

Wijnen, K., Genzel, L., & van der Meij, J. (2024). Rodent maze studies: from
following simple rules to complex map learning. Brain Structure and Function,
229, 823.
https://doi.org/10.1007/s00429-024-02771-x

Examples
--------
>>> from neurospatial.simulation.mazes.repeated_t import (
...     make_repeated_t_maze,
...     RepeatedTDims,
... )
>>> maze = make_repeated_t_maze()
>>> maze.env_2d.units
'cm'
>>> [name for name in maze.env_2d.regions if name.startswith("choice_")]
['choice_1', 'choice_2', 'choice_3', 'choice_4', 'choice_5', 'choice_6']
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
from shapely.affinity import translate
from shapely.geometry import box

from neurospatial.environment import Environment
from neurospatial.regions import Regions
from neurospatial.simulation.mazes._base import MazeDims, MazeEnvironments
from neurospatial.simulation.mazes._geometry import (
    corridor_end_point,
    make_corridor_polygon,
    union_polygons,
    warn_if_bins_bridge_walls,
)

_N_CHOICES = 6


@dataclass(frozen=True)
class RepeatedTDims(MazeDims):
    """Dimensions of Blodgett's six-unit T maze (Tolman & Nyswander, 1927).

    Lengths are measured along an alley's walls, from the wall of the alley
    it leaves to its end or to the wall of the next alley.

    Attributes
    ----------
    width : float
        Width of every alley in cm. Default is 10.16 (4 in, from the figure).
    arm_length : float
        Length of each alley segment in cm. Default is 40.64 (16 in, from the
        figure).
    long_arm_length : float
        Length of the two segments that join choice points 4, 5 and 6, in
        cm. Default is 45.72 (18 in, from the figure). It must exceed
        ``arm_length``: the difference is the wall between blind alley 6 and
        the alley to choice point 3.
    wall_thickness : float
        Partition where blind alley 4 ends against the first alley, in cm.
        Default is 2.1 (about 0.8 in, scaled from the figure; an estimate).
    start_stem_length : float
        Alley from the start box to choice point 1, in cm. Default is 10.16
        (about 4 in, scaled from the figure; an estimate).
    start_box_width : float
        Width of the start box across the stem, in cm. Default is 19.05
        (about 7.5 in, scaled from the figure; an estimate).
    start_box_depth : float
        Depth of the start box along the stem, in cm. Default is 15.24
        (about 6 in, scaled from the figure; an estimate).

    Examples
    --------
    >>> dims = RepeatedTDims()
    >>> dims.width, dims.arm_length, dims.long_arm_length
    (10.16, 40.64, 45.72)

    >>> RepeatedTDims(arm_length=50.0).arm_length
    50.0
    """

    width: float = 10.16
    arm_length: float = 40.64
    long_arm_length: float = 45.72
    wall_thickness: float = 2.1
    start_stem_length: float = 10.16
    start_box_width: float = 19.05
    start_box_depth: float = 15.24


def make_repeated_t_maze(
    dims: RepeatedTDims | None = None,
    bin_size: float = 2.0,
    include_track: bool = True,
) -> MazeEnvironments:
    """Create Blodgett's six-unit T maze (Tolman & Nyswander, 1927, Fig. 5).

    Parameters
    ----------
    dims : RepeatedTDims, optional
        Maze dimensions. If None, uses the dimensions written on the figure.
    bin_size : float, optional
        Spatial bin size in cm (default: 2.0). Keep it below the thinnest
        wall (2.1 cm at the default dimensions); larger bins can bridge it
        and trigger a warning.
    include_track : bool, optional
        Whether to create the linearized track graph (default: True).

    Returns
    -------
    MazeEnvironments
        Contains:
        - env_2d: the alleys and start box
        - env_track: alley centrelines as a track graph (if include_track)

    Raises
    ------
    ValueError
        If ``long_arm_length`` does not exceed ``arm_length`` or
        ``wall_thickness`` is not positive, so that alleys would touch.

    Notes
    -----
    The origin is the centre of the maze's bounding box. The start box is
    at the left and the goal at the right end of the top row.

    Regions:

    - start: centre of the start box
    - choice_1 ... choice_6: the T junctions, in the order the rat meets them
    - blind_1 ... blind_6: the end of the blind alley at each choice point
    - goal: the end of the last alley, Fig. 5's F

    Track graph: start, the six choice points, the six blind-alley ends and
    the goal; the true path runs start, choice_1, ..., choice_6, goal.

    Examples
    --------
    >>> maze = make_repeated_t_maze()
    >>> maze.env_2d.units
    'cm'
    >>> maze.env_2d.n_bins > 0
    True

    Skip track graph creation:

    >>> maze = make_repeated_t_maze(include_track=False)
    >>> maze.env_track is None
    True

    See Also
    --------
    RepeatedTDims : Dimension specifications for the repeated T-maze.
    """
    if dims is None:
        dims = RepeatedTDims()
    thinnest_wall = min(dims.wall_thickness, dims.long_arm_length - dims.arm_length)
    if thinnest_wall <= 0:
        raise ValueError(
            f"RepeatedTDims(long_arm_length={dims.long_arm_length}, "
            f"arm_length={dims.arm_length}, wall_thickness={dims.wall_thickness}) "
            "makes alleys touch. Use long_arm_length greater than arm_length "
            "and a positive wall_thickness."
        )
    warn_if_bins_bridge_walls(bin_size, thinnest_wall, "repeated T-maze")

    positions, edges = _layout(dims)
    alleys = [
        make_corridor_polygon(positions[u], positions[v], dims.width) for u, v in edges
    ]
    stem_end = positions["choice_1"][0] - dims.width / 2.0 - dims.start_stem_length
    start_box = box(
        stem_end - dims.start_box_depth,
        -dims.start_box_width / 2.0,
        stem_end,
        dims.start_box_width / 2.0,
    )
    maze_polygon = union_polygons([*alleys, start_box])

    # Put the origin at the centre of the maze's bounding box.
    min_x, min_y, max_x, max_y = maze_polygon.bounds
    offset = np.array([(min_x + max_x) / 2.0, (min_y + max_y) / 2.0])
    maze_polygon = translate(maze_polygon, -offset[0], -offset[1])
    positions = {
        name: (float(x - offset[0]), float(y - offset[1]))
        for name, (x, y) in positions.items()
    }

    env_2d = Environment.from_polygon(
        polygon=maze_polygon,
        bin_size=bin_size,
        name="repeated_t_maze",
        connect_diagonal_neighbors=True,
    )
    env_2d = env_2d.with_metadata(units="cm")

    regions = Regions()
    regions.add("start", point=positions["start"])
    for k in range(1, _N_CHOICES + 1):
        regions.add(f"choice_{k}", point=positions[f"choice_{k}"])
    for k in range(1, _N_CHOICES + 1):
        end = corridor_end_point(
            positions[f"choice_{k}"], positions[f"blind_{k}"], dims.width
        )
        regions.add(f"blind_{k}", point=end)
    regions.add(
        "goal",
        point=corridor_end_point(
            positions[f"choice_{_N_CHOICES}"], positions["goal"], dims.width
        ),
    )
    env_2d = env_2d.with_regions(regions)

    env_track = None
    if include_track:
        env_track = _create_repeated_t_maze_track_graph(
            dims, bin_size, positions, edges
        )

    return MazeEnvironments(env_2d=env_2d, env_track=env_track)


def _layout(
    dims: RepeatedTDims,
) -> tuple[dict[str, tuple[float, float]], list[tuple[str, str]]]:
    """Track nodes (choice point 1 at the origin) and edges in running order."""
    w, half = dims.width, dims.width / 2.0
    arm, long_arm = dims.arm_length, dims.long_arm_length
    near = arm + w  # centre-to-centre distance across one 16-in segment
    far = long_arm + w  # ... across one 18-in segment
    choices = [
        (0.0, 0.0),
        (0.0, near),
        (near, near),
        (near, 0.0),
        (near + far, 0.0),
        (near + far, far),
    ]
    blind_ends = [
        (0.0, -(arm + half)),
        (-(arm + half), near),
        (near, near + arm + half),
        (half + dims.wall_thickness, 0.0),  # stops a partition short of alley 1
        (near + far, -(arm + half)),
        (near + far - arm - half, far),
    ]
    positions = {
        "start": (-half - dims.start_stem_length - dims.start_box_depth / 2.0, 0.0),
        "goal": (near + far + arm + half, far),
    }
    edges = [("start", "choice_1")]
    for k, (choice, blind_end) in enumerate(
        zip(choices, blind_ends, strict=True), start=1
    ):
        positions[f"choice_{k}"] = choice
        positions[f"blind_{k}"] = blind_end
        edges.append((f"choice_{k}", f"blind_{k}"))
        edges.append((f"choice_{k}", f"choice_{k + 1}" if k < _N_CHOICES else "goal"))
    return positions, edges


def _create_repeated_t_maze_track_graph(
    dims: RepeatedTDims,
    bin_size: float,
    positions: dict[str, tuple[float, float]],
    edges: list[tuple[str, str]],
) -> Environment:
    """Create the 1D linearized track graph for the repeated T-maze.

    Parameters
    ----------
    dims : RepeatedTDims
        Maze dimensions.
    bin_size : float
        Spatial bin size in cm.
    positions : dict
        Node positions in cm, centred on the maze.
    edges : list of (str, str)
        Track edges in running order: along the true path, each blind alley
        as its choice point is reached.

    Returns
    -------
    Environment
        1D linearized environment representing the repeated T-maze track.
    """
    graph = nx.Graph()
    for name, position in positions.items():
        graph.add_node(name, pos=position)
    for u, v in edges:
        distance = float(np.linalg.norm(np.subtract(positions[v], positions[u])))
        graph.add_edge(u, v, distance=distance)

    env_track = Environment.from_graph(
        graph=graph,
        edge_order=edges,
        edge_spacing=0.0,
        bin_size=bin_size,
        track_width=dims.width,
        name="repeated_t_maze_1d",
    )
    return env_track.with_metadata(units="cm")
