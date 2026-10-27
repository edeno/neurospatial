"""T-Maze environment for spatial navigation research.

The T-Maze is one of the simplest maze designs for binary decision-making tasks.
It consists of a central stem with a perpendicular crossbar forming a "T" shape.
Animals must choose between turning left or right at the junction, making it ideal
for studying egocentric (body-turn) learning and alternation behaviors.

Wijnen et al. (2024), Brain Struct Funct 229:823-841,
doi:10.1007/s00429-024-02771-x, review the design: "Thus, in a T-maze the
animal must make an active decision between left and right without being able
to see what is ahead (Stewart et al.)."

Rat T-maze sizes vary from lab to lab and there is no canonical apparatus. The
defaults here are consistent with the T-maze drawn in Wijnen et al. (2024)
Fig. 1a, whose crossbar spans about the 1 m scale bar.

Deacon & Rawlins (2006), Nat Protoc 1:7-12, doi:10.1038/nprot.2006.2, run
both spontaneous and rewarded alternation on it: "Animals are started from
the base of the T and allowed to choose one of the goal arms", and the
alternation "can be reinforced by making the animal hungry and rewarding it
with a preferred food if it alternates". Each goal arm ends in a food well,
``well_left`` and ``well_right``: reward-port landmarks with point regions of
the same name, centred in the end square of their arm. For spontaneous
alternation, ignore the wells.

The arms are walled (``arena.outline == "wall"``): Wijnen et al. (2024) note
that in a T-maze "animals would collide with a wall" and "cannot look inside
one of the arms until arrival at the intersection". They do not give the
wall material; the walls are modeled as opaque.

Examples
--------
>>> from neurospatial.simulation.mazes import make_t_maze, TMazeDims
>>> maze = make_t_maze()
>>> maze.env_2d.units
'cm'
>>> "junction" in maze.env_2d.regions
True
>>> "start" in maze.env_2d.regions
True
"""

from __future__ import annotations

from dataclasses import dataclass

from neurospatial.environment import Environment
from neurospatial.landmarks import Landmark, Landmarks
from neurospatial.regions import Regions
from neurospatial.simulation.mazes._base import MazeDims, MazeEnvironments
from neurospatial.simulation.mazes._geometry import (
    corridor_end_point,
    make_corridor_polygon,
    union_polygons,
)


@dataclass(frozen=True)
class TMazeDims(MazeDims):
    """Dimension specifications for T-Maze.

    The T-Maze consists of a vertical stem with perpendicular arms at the top.
    Rat T-maze sizes vary between labs; the defaults are consistent with the
    T-maze in Wijnen et al. (2024) Fig. 1a (doi:10.1007/s00429-024-02771-x),
    whose crossbar spans about its 1 m scale bar. They are an estimate read
    from the figure, not a published apparatus.

    Attributes
    ----------
    stem_length : float
        Length of the stem's centreline in cm, from its far end to the
        crossbar's centreline. Default is 100.0.
    arm_length : float
        Length of each arm's centreline in cm, from the stem's centreline to
        the arm's far end. Default is 50.0.
    width : float
        Width of all corridors in cm. Default is 10.0.

    Examples
    --------
    >>> dims = TMazeDims()
    >>> dims.stem_length
    100.0
    >>> dims.arm_length
    50.0
    >>> dims.width
    10.0

    >>> custom = TMazeDims(stem_length=150.0, arm_length=75.0, width=15.0)
    >>> custom.stem_length
    150.0
    """

    stem_length: float = 100.0
    arm_length: float = 50.0
    width: float = 10.0


def make_t_maze(
    dims: TMazeDims | None = None,
    bin_size: float = 2.0,
    include_track: bool = True,
) -> MazeEnvironments:
    """Create a T-Maze environment.

    Creates a T-shaped maze with a vertical stem and perpendicular arms.
    The animal starts at the base of the stem and must choose left or right
    at the junction. This is the simplest binary decision maze design.

    Parameters
    ----------
    dims : TMazeDims, optional
        Maze dimensions. If None, uses default dimensions
        (100 cm stem, 50 cm arms, 10 cm width).
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
    - Stem runs from y = -stem_length/2 to y = +stem_length/2
    - Arms run from x = -arm_length to x = +arm_length at y = +stem_length/2
    - Junction is at (0, +stem_length/2)

    Regions (the start and arm-end points are centred in the end square of
    their corridor, half a corridor width back from its far end):

    - start: Point at stem base (0, -stem_length/2 + width/2)
    - junction: Point at T-junction (0, +stem_length/2)
    - well_left: Food well at the left arm end (-arm_length + width/2,
      +stem_length/2), also a reward-port landmark
    - well_right: Food well at the right arm end (+arm_length - width/2,
      +stem_length/2), also a reward-port landmark

    Track Graph Topology:
    - start -> junction: Vertical edge along stem
    - junction -> well_left: Horizontal edge along left arm
    - junction -> well_right: Horizontal edge along right arm

    Examples
    --------
    Create a default T-maze:

    >>> maze = make_t_maze()
    >>> maze.env_2d.units
    'cm'
    >>> maze.env_2d.n_bins > 0
    True

    Create with custom dimensions:

    >>> dims = TMazeDims(stem_length=150.0, arm_length=80.0)
    >>> maze = make_t_maze(dims=dims, bin_size=4.0)
    >>> "junction" in maze.env_2d.regions
    True

    Skip track graph creation:

    >>> maze = make_t_maze(include_track=False)
    >>> maze.env_track is None
    True

    See Also
    --------
    TMazeDims : Dimension specifications for T-Maze.
    """
    if dims is None:
        dims = TMazeDims()

    # Key positions (maze centered at origin)
    half_stem = dims.stem_length / 2.0

    # Start at bottom of stem
    start_pos = (0.0, -half_stem)
    # Junction at top of stem
    junction_pos = (0.0, half_stem)
    # Arm endpoints
    left_end_pos = (-dims.arm_length, half_stem)
    right_end_pos = (dims.arm_length, half_stem)

    # Create corridor polygons
    # Stem: vertical corridor from start to junction
    stem_polygon = make_corridor_polygon(
        start=start_pos,
        end=junction_pos,
        width=dims.width,
    )

    # Left arm: horizontal corridor from junction to left end
    left_arm_polygon = make_corridor_polygon(
        start=junction_pos,
        end=left_end_pos,
        width=dims.width,
    )

    # Right arm: horizontal corridor from junction to right end
    right_arm_polygon = make_corridor_polygon(
        start=junction_pos,
        end=right_end_pos,
        width=dims.width,
    )

    # Union all corridors into single polygon
    t_polygon = union_polygons([stem_polygon, left_arm_polygon, right_arm_polygon])

    # Create 2D environment
    env_2d = Environment.from_polygon(
        polygon=t_polygon,
        bin_size=bin_size,
        name="t_maze",
        connect_diagonal_neighbors=True,
    )
    env_2d = env_2d.with_metadata(units="cm")

    # Add regions for key locations
    region_builder = Regions()
    region_builder.add(
        "start", point=corridor_end_point(junction_pos, start_pos, dims.width)
    )
    region_builder.add("junction", point=junction_pos)
    wells = {
        "well_left": corridor_end_point(junction_pos, left_end_pos, dims.width),
        "well_right": corridor_end_point(junction_pos, right_end_pos, dims.width),
    }
    for name, point in wells.items():
        region_builder.add(name, point=point)
    env_2d = env_2d.with_regions(region_builder)
    env_2d = env_2d.with_landmarks(
        Landmarks(
            [
                Landmark(
                    name,
                    "point",
                    point,
                    {"role": "reward_port", "port_type": "food_well"},
                )
                for name, point in wells.items()
            ]
        )
    )

    # Create track graph if requested
    env_track = None
    if include_track:
        env_track = _create_t_maze_track_graph(dims, bin_size)

    return MazeEnvironments(env_2d=env_2d, env_track=env_track)


def _create_t_maze_track_graph(
    dims: TMazeDims,
    bin_size: float,
) -> Environment:
    """Create the 1D linearized track graph for T-Maze.

    The track graph represents the T-maze topology with three edges:
    - start -> junction (stem)
    - junction -> well_left (left arm)
    - junction -> well_right (right arm)

    Parameters
    ----------
    dims : TMazeDims
        Maze dimensions.
    bin_size : float
        Spatial bin size in cm.

    Returns
    -------
    Environment
        1D linearized environment representing the T-maze track.
    """
    half_stem = dims.stem_length / 2.0
    # Node order [stem_end, junction, arm_left, arm_right] gives the stem
    # and both arms, linearized in that order.
    return Environment.maze(
        kind="t",
        node_positions={
            "start": (0.0, -half_stem),
            "junction": (0.0, half_stem),
            "well_left": (-dims.arm_length, half_stem),
            "well_right": (dims.arm_length, half_stem),
        },
        bin_size=bin_size,
        track_width=dims.width,
        edge_spacing=0.0,
        name="t_maze_1d",
        units="cm",
    )
