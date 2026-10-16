"""T-Maze environment for spatial navigation research.

The T-Maze is one of the simplest maze designs for binary decision-making tasks.
It consists of a central stem with a perpendicular crossbar forming a "T" shape.
Animals must choose between turning left or right at the junction, making it ideal
for studying egocentric (body-turn) learning and alternation behaviors.

Reference: Wijnen et al. 2024 (Brain Structure & Function), Figure 1

"in a T-maze the animal must make an active decision between left and right
without being able to see what is ahead" (p. 827)

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
from neurospatial.regions import Regions
from neurospatial.simulation.mazes._base import MazeDims, MazeEnvironments
from neurospatial.simulation.mazes._geometry import (
    make_corridor_polygon,
    union_polygons,
)


@dataclass(frozen=True)
class TMazeDims(MazeDims):
    """Dimension specifications for T-Maze.

    The T-Maze consists of a vertical stem with perpendicular arms at the top.
    Dimensions are based on the Wijnen et al. 2024 figure scale bar.

    Attributes
    ----------
    stem_length : float
        Length of the vertical stem in cm. Default is 100.0.
    arm_length : float
        Length of each horizontal arm in cm. Default is 50.0.
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

    Regions:
    - start: Point at stem base (0, -stem_length/2)
    - junction: Point at T-junction (0, +stem_length/2)
    - left_end: Point at left arm end (-arm_length, +stem_length/2)
    - right_end: Point at right arm end (+arm_length, +stem_length/2)

    Track Graph Topology:
    - start -> junction: Vertical edge along stem
    - junction -> left_end: Horizontal edge along left arm
    - junction -> right_end: Horizontal edge along right arm

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
    region_builder.add("start", point=start_pos)
    region_builder.add("junction", point=junction_pos)
    region_builder.add("left_end", point=left_end_pos)
    region_builder.add("right_end", point=right_end_pos)
    env_2d = env_2d.with_regions(region_builder)

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
    - junction -> left_end (left arm)
    - junction -> right_end (right arm)

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
            "left_end": (-dims.arm_length, half_stem),
            "right_end": (dims.arm_length, half_stem),
        },
        bin_size=bin_size,
        edge_spacing=0.0,
        name="t_maze_1d",
        units="cm",
    )
