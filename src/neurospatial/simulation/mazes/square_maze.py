"""Square Maze (Open Field Box) environment for spatial navigation research.

The Square Maze is a simple rectangular open-field environment commonly used
for sleep studies, rest periods, and baseline behavioral recordings. Unlike
corridor-based mazes, animals can move freely in any direction within the
square arena.

This environment is often used as a "sleep box" or "rest box" between trials
in multi-maze experiments, providing a neutral environment for consolidation
studies. A rest box holds no reward ports, start or goal boxes, or cues, so
the maze has no landmarks.

The box is walled (``arena.outline == "wall"``), with opaque walls.

Examples
--------
>>> from neurospatial.simulation.mazes import make_square_maze, SquareMazeDims
>>> maze = make_square_maze()
>>> maze.env_2d.units
'cm'
>>> "center" in maze.env_2d.regions
True
>>> maze.env_track is None  # Open field has no track topology
True
"""

from __future__ import annotations

from dataclasses import dataclass

from shapely.geometry import box

from neurospatial.environment import Environment
from neurospatial.regions import Regions
from neurospatial.simulation.mazes._base import (
    MazeDims,
    MazeEnvironments,
    check_lengths,
    warn_if_bins_too_coarse,
)
from neurospatial.simulation.mazes._geometry import compass_quadrants


@dataclass(frozen=True)
class SquareMazeDims(MazeDims):
    """Dimension specifications for Square Maze (Open Field Box).

    The Square Maze is a simple rectangular arena. Animals can move freely
    in any direction within the boundaries.

    Attributes
    ----------
    side_length : float
        Length of each side of the square in cm. Default is 30.0.

    Notes
    -----
    Default dimensions (30 x 30 cm) are typical for rat sleep boxes.
    For mouse experiments, smaller dimensions (20 x 20 cm) may be appropriate.
    For larger open field tests, consider 50-100 cm.

    Examples
    --------
    >>> dims = SquareMazeDims()
    >>> dims.side_length
    30.0

    >>> custom = SquareMazeDims(side_length=50.0)
    >>> custom.side_length
    50.0
    """

    side_length: float = 30.0

    def __post_init__(self) -> None:
        check_lengths(self, "side_length")


def make_square_maze(
    dims: SquareMazeDims | None = None,
    bin_size: float = 2.0,
) -> MazeEnvironments:
    """Create a Square Maze (Open Field Box) environment.

    Creates a simple square arena for open-field navigation, sleep studies,
    and rest periods. As an open-field environment, animals can move freely
    in any direction, and no track graph is provided.

    Parameters
    ----------
    dims : SquareMazeDims, optional
        Maze dimensions. If None, uses default dimensions (30 x 30 cm).
    bin_size : float, optional
        Spatial bin size in cm (default: 2.0).

    Returns
    -------
    MazeEnvironments
        Contains:
        - env_2d: 2D square environment
        - env_track: None (open field has no track topology)

    Notes
    -----
    The maze is centered at the origin:
    - X range: -side_length/2 to +side_length/2
    - Y range: -side_length/2 to +side_length/2

    North is +y and east is +x. Regions:

    - center: a point at the centre of the box, (0, 0), covering the one
      bin there.
    - NE, NW, SE, SW: the box's quarters, bounded by the north-south and
      east-west lines through its centre, with the same meaning as the
      watermaze's quadrants, for quadrant occupancy and spatial preference.
      Together they cover the box without overlapping, and every bin is in
      exactly one quadrant. With an odd number of bins across, a row and a
      column of bin centres lie on the dividing lines; a centre on a line
      counts as east of the north-south line and north of the east-west
      line, so NE holds the bins with x >= 0 and y >= 0 and SW those with
      x < 0 and y < 0 (see :meth:`~neurospatial.Environment.bins_in_region`).
      The default 15 x 15 grid has 64 bins in NE, 56 in NW and SE, and 49 in
      SW.

    For thigmotaxis (wall-following), measure how far the animal is from the
    walls with :func:`neurospatial.behavior.distance_to_boundary` or
    :func:`neurospatial.ops.nearest_boundary` rather than a region.

    Examples
    --------
    Create a default square maze (30 x 30 cm sleep box):

    >>> maze = make_square_maze()
    >>> maze.env_2d.units
    'cm'
    >>> maze.env_2d.n_bins > 0
    True
    >>> maze.env_track is None
    True

    Create with custom dimensions:

    >>> dims = SquareMazeDims(side_length=50.0)
    >>> maze = make_square_maze(dims=dims, bin_size=4.0)
    >>> "center" in maze.env_2d.regions
    True

    Create a larger open field:

    >>> dims = SquareMazeDims(side_length=100.0)
    >>> maze = make_square_maze(dims=dims)
    >>> maze.env_2d.regions["NE"].data.bounds  # the north-east quarter, cm
    (0.0, 0.0, 50.0, 50.0)

    See Also
    --------
    SquareMazeDims : Dimension specifications for Square Maze.
    make_watermaze : Circular open-field environment.
    """
    if dims is None:
        dims = SquareMazeDims()

    # Half side for centering at origin
    half_side = dims.side_length / 2.0

    # Create square polygon centered at origin using shapely.box
    # box(minx, miny, maxx, maxy)
    square_polygon = box(-half_side, -half_side, half_side, half_side)

    # Create 2D environment
    env_2d = Environment.from_polygon(
        polygon=square_polygon,
        bin_size=bin_size,
        name="square_maze",
        connect_diagonal_neighbors=True,
    )
    env_2d = env_2d.with_metadata(units="cm")

    # Add center region
    region_builder = Regions()
    region_builder.add("center", point=(0.0, 0.0))

    # Each quadrant is the box's quarter on one side of both axes, as in
    # the watermaze
    for name, quadrant in compass_quadrants(square_polygon).items():
        region_builder.add(name, polygon=quadrant)
    env_2d = env_2d.with_regions(region_builder)

    # No track graph for open field
    env_track = None

    warn_if_bins_too_coarse(
        env_2d, maze="square maze", bin_size=bin_size, narrowest_corridor=None
    )
    return MazeEnvironments(env_2d=env_2d, env_track=env_track)
