"""Plus Maze (Cruciate Maze) environment for spatial navigation research.

The Plus Maze (also called Cruciate Maze) is a four-arm maze forming a "+"
shape with perpendicular arms extending in all cardinal directions from a
central junction. It is commonly used for rule-switching tasks where animals
must learn and flexibly switch between different response rules (e.g., go
east vs go west, or go to the light vs go to the dark arm).

The plus maze provides more decision alternatives than a T-maze while
maintaining a simple structure that supports both egocentric (body-turn)
and allocentric (world-centered) navigation strategies. Wijnen et al. (2024),
Brain Struct Funct 229:823-841, doi:10.1007/s00429-024-02771-x, discuss plus
mazes in their review text; the plus maze is not among the mazes drawn in
their Fig. 1. Plus-maze sizes vary between labs, and the defaults here are
not taken from a published apparatus.

The maze has no landmarks, because its start and goal depend on the
protocol. Wijnen et al. (2024) introduce it as a maze "in which both these
locations [start and end points] are frequently variable". In the place
versus response task of Packard & McGaugh (1996), Neurobiol Learn Mem
65:65-72, doi:10.1006/nlme.1996.0007, "Rats were trained to approach a
consistently baited arm in a cross-maze from the same start box", and on
probe trials "rats were placed in the start box opposite that used in
training". Which arm holds the food and which arms hold start boxes is set
by the experimenter, and the abstract gives neither the food cup's nor the
start boxes' size, so the four arm ends are plain regions. Add reward-port
and start-box landmarks at the arm ends your protocol uses.

No published apparatus fixes the build. The maze is modeled as an elevated
open plus whose edges are drops (``arena.outline == "drop"``); rebuild the
floor with ``Environment.from_polygon`` for an enclosed version.

Examples
--------
>>> from neurospatial.simulation.mazes import make_plus_maze, PlusMazeDims
>>> maze = make_plus_maze()
>>> maze.env_2d.units
'cm'
>>> "center" in maze.env_2d.regions
True
>>> "north_end" in maze.env_2d.regions
True
"""

from __future__ import annotations

from dataclasses import dataclass

from neurospatial.arena import OpenArena
from neurospatial.environment import Environment
from neurospatial.regions import Regions
from neurospatial.simulation.mazes._base import (
    MazeDims,
    MazeEnvironments,
    check_lengths,
    check_longer_than_wide,
    warn_if_bins_too_coarse,
)
from neurospatial.simulation.mazes._geometry import (
    corridor_end_point,
    make_corridor_polygon,
    single_floor,
    union_polygons,
)


@dataclass(frozen=True)
class PlusMazeDims(MazeDims):
    """Dimension specifications for Plus Maze (Cruciate Maze).

    The Plus Maze consists of four arms extending from a central junction
    in the cardinal directions (north, south, east, west). All arms are
    symmetric by default.

    Attributes
    ----------
    arm_length : float
        Length of each arm's centreline in cm, from the junction centre,
        where the arms' centrelines cross, to the arm's far end wall, as in
        the T, Y and W mazes. Default is 45.0.
    corridor_width : float
        Width of all corridors in cm. Default is 10.0.

    Notes
    -----
    Default dimensions create a 90 cm x 90 cm maze footprint:
    total width/height = 2 * arm_length = 2 * 45 = 90 cm.

    Examples
    --------
    >>> dims = PlusMazeDims()
    >>> dims.arm_length
    45.0
    >>> dims.corridor_width
    10.0

    >>> custom = PlusMazeDims(arm_length=60.0, corridor_width=15.0)
    >>> custom.arm_length
    60.0
    """

    arm_length: float = 45.0
    corridor_width: float = 10.0

    def __post_init__(self) -> None:
        check_lengths(self, "arm_length", "corridor_width")
        check_longer_than_wide(self, "arm_length", "arm")


def make_plus_maze(
    dims: PlusMazeDims | None = None,
    bin_size: float = 2.0,
    include_track: bool = True,
) -> MazeEnvironments:
    """Create a Plus Maze (Cruciate Maze) environment.

    Creates a plus-shaped maze with four arms extending in cardinal directions
    from a central junction. Animals can navigate to any of the four endpoints,
    making it ideal for studying rule-switching, spatial memory, and decision-
    making with multiple alternatives.

    Parameters
    ----------
    dims : PlusMazeDims, optional
        Maze dimensions. If None, uses default dimensions creating a
        90 cm x 90 cm maze (45 cm arms, 10 cm width).
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
    - North arm: y = 0 to y = +arm_length
    - South arm: y = 0 to y = -arm_length
    - East arm: x = 0 to x = +arm_length
    - West arm: x = 0 to x = -arm_length

    Regions (arm ends are centred in the end square of each arm, half a
    corridor width back from its far end, with ``d = arm_length - corridor_width/2``):

    - center: Central junction at (0, 0)
    - north_end: North arm end (0, +d)
    - south_end: South arm end (0, -d)
    - east_end: East arm end (+d, 0)
    - west_end: West arm end (-d, 0)

    Track Graph Topology (star):
    - center -> north_end
    - center -> south_end
    - center -> east_end
    - center -> west_end

    Examples
    --------
    Create a default plus maze:

    >>> maze = make_plus_maze()
    >>> maze.env_2d.units
    'cm'
    >>> maze.env_2d.n_bins > 0
    True

    Create with custom dimensions:

    >>> dims = PlusMazeDims(arm_length=60.0, corridor_width=15.0)
    >>> maze = make_plus_maze(dims=dims, bin_size=4.0)
    >>> "center" in maze.env_2d.regions
    True

    Skip track graph creation:

    >>> maze = make_plus_maze(include_track=False)
    >>> maze.env_track is None
    True

    See Also
    --------
    PlusMazeDims : Dimension specifications for Plus Maze.
    make_t_maze : Simpler T-shaped maze with only 3 arms.
    """
    if dims is None:
        dims = PlusMazeDims()

    # Key positions (maze centered at origin)
    center_pos = (0.0, 0.0)
    north_end_pos = (0.0, dims.arm_length)
    south_end_pos = (0.0, -dims.arm_length)
    east_end_pos = (dims.arm_length, 0.0)
    west_end_pos = (-dims.arm_length, 0.0)

    # Create corridor polygons for each arm
    # North arm: center to north
    north_polygon = make_corridor_polygon(
        start=center_pos,
        end=north_end_pos,
        width=dims.corridor_width,
    )

    # South arm: center to south
    south_polygon = make_corridor_polygon(
        start=center_pos,
        end=south_end_pos,
        width=dims.corridor_width,
    )

    # East arm: center to east
    east_polygon = make_corridor_polygon(
        start=center_pos,
        end=east_end_pos,
        width=dims.corridor_width,
    )

    # West arm: center to west
    west_polygon = make_corridor_polygon(
        start=center_pos,
        end=west_end_pos,
        width=dims.corridor_width,
    )

    # Union all corridors into single plus-shaped polygon
    plus_polygon = single_floor(
        union_polygons([north_polygon, south_polygon, east_polygon, west_polygon]),
        dims,
    )

    # Create 2D environment
    env_2d = Environment.from_polygon(
        polygon=plus_polygon,
        bin_size=bin_size,
        arena=OpenArena(plus_polygon, outline="drop"),
        name="plus_maze",
        connect_diagonal_neighbors=True,
    )
    env_2d = env_2d.with_metadata(units="cm")

    # Add regions for key locations
    region_builder = Regions()
    region_builder.add("center", point=center_pos)
    for name, end_pos in (
        ("north_end", north_end_pos),
        ("south_end", south_end_pos),
        ("east_end", east_end_pos),
        ("west_end", west_end_pos),
    ):
        region_builder.add(
            name, point=corridor_end_point(center_pos, end_pos, dims.corridor_width)
        )
    env_2d = env_2d.with_regions(region_builder)

    # Create track graph if requested
    env_track = None
    if include_track:
        env_track = _create_plus_maze_track_graph(dims, bin_size)

    warn_if_bins_too_coarse(
        env_2d,
        maze="plus maze",
        bin_size=bin_size,
        narrowest_corridor=dims.corridor_width,
    )
    return MazeEnvironments(env_2d=env_2d, env_track=env_track)


def _create_plus_maze_track_graph(
    dims: PlusMazeDims,
    bin_size: float,
) -> Environment:
    """Create the 1D linearized track graph for Plus Maze.

    The track graph represents the plus maze topology with four edges
    radiating from a central node (star topology):
    - center -> north_end
    - center -> south_end
    - center -> east_end
    - center -> west_end

    Parameters
    ----------
    dims : PlusMazeDims
        Maze dimensions.
    bin_size : float
        Spatial bin size in cm.

    Returns
    -------
    Environment
        1D linearized environment representing the plus maze track.
    """
    # Node order [center, arm1, ..., arm4] gives one edge per arm,
    # linearized clockwise from north.
    return Environment.maze(
        kind="plus",
        node_positions={
            "center": (0.0, 0.0),
            "north_end": (0.0, dims.arm_length),
            "east_end": (dims.arm_length, 0.0),
            "south_end": (0.0, -dims.arm_length),
            "west_end": (-dims.arm_length, 0.0),
        },
        bin_size=bin_size,
        track_width=dims.corridor_width,
        edge_spacing=0.0,
        name="plus_maze_1d",
        units="cm",
    )
