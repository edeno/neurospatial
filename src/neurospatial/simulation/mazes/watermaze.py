"""Morris Water Maze environment for spatial navigation research.

The Morris Water Maze (MWM) is a widely used behavioral test for spatial learning
and memory. Animals navigate in a circular pool to find a hidden platform, requiring
allocentric (world-centered) spatial memory and the use of distal cues.

Unlike corridor-based mazes (T-maze, Y-maze), the MWM is an open-field environment
with no physical constraints on movement paths. Animals can swim in any direction,
making it ideal for studying flexible spatial strategies.

The pool's side wall is modeled as opaque (``arena.outline == "wall"``).

Reference: Morris (1984). "Developments of a water-maze procedure for studying
spatial learning in the rat." J Neurosci Methods 11:47-60,
doi:10.1016/0165-0270(84)90007-4.

The default sizes follow Morris (2008), "Morris water maze", Scholarpedia
3:6315, doi:10.4249/scholarpedia.6315: pools are "generally 1.5 to 2 metres
in diameter", and the platform is "usually about 10-15 cm in diameter and thus
between 1/50th and 1/100th of the surface area of the pool". The default
150 cm pool and 15 cm platform lie in those ranges.

The platform sits by default in the centre of the NE quadrant, which by the
usual convention is halfway between the pool centre and the wall along the
quadrant's diagonal; that placement is a convention, not taken from Morris
(1984) (UNVERIFIED). North is +y and east is +x.

The quadrants are quarters of the pool for probe trials, in which "the
escape platform is removed from the pool" and "a well-trained rat will swim
to the target quadrant of the pool and repeatedly across the former location
of the platform" (Morris 2008). Animals are "placed into the water at and
facing the sidewalls of the pool, at different start positions across
trials" (Morris 2008), with no fixed start positions given, so the maze has no
start landmarks. The pool sits among "distinctive 2- and 3-D distal cues"
whose placement is the laboratory's, so it has no cue landmarks either.

Examples
--------
>>> from neurospatial.simulation.mazes.watermaze import make_watermaze, WatermazeDims
>>> maze = make_watermaze()
>>> maze.env_2d.units
'cm'
>>> maze.env_2d.landmarks["platform"].metadata["role"]
'platform'
>>> maze.env_2d.regions["NE"].kind
'polygon'
>>> maze.env_track is None
True
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neurospatial.environment import Environment
from neurospatial.landmarks import Landmark, Landmarks
from neurospatial.regions import Regions
from neurospatial.simulation.mazes._base import (
    MazeDims,
    MazeEnvironments,
    check_lengths,
    refuse,
)
from neurospatial.simulation.mazes._geometry import (
    compass_quadrants,
    make_circular_arena,
)


@dataclass(frozen=True)
class WatermazeDims(MazeDims):
    """Dimension specifications for Morris Water Maze.

    The Morris Water Maze is a circular pool containing a hidden platform.
    Animals must use spatial cues to locate the platform beneath the water surface.

    Attributes
    ----------
    pool_diameter : float
        Diameter of the circular pool in cm. Default is 150.0, at the small
        end of the 1.5-2 m range given by Morris (2008),
        doi:10.4249/scholarpedia.6315.
    platform_radius : float
        Radius of the circular platform in cm. Default is 7.5 (a 15 cm
        platform, at the large end of the 10-15 cm range given by Morris
        (2008)).

    Examples
    --------
    >>> dims = WatermazeDims()
    >>> dims.pool_diameter
    150.0
    >>> dims.platform_radius * 2  # platform diameter, cm
    15.0

    >>> custom = WatermazeDims(pool_diameter=200.0, platform_radius=10.0)
    >>> custom.pool_diameter
    200.0
    """

    pool_diameter: float = 150.0
    platform_radius: float = 7.5

    def __post_init__(self) -> None:
        check_lengths(self, "pool_diameter", "platform_radius")
        if 2.0 * self.platform_radius >= self.pool_diameter:
            refuse(
                self,
                ("pool_diameter", "platform_radius"),
                "makes the platform no smaller than the pool",
                "Use a platform_radius below half the pool_diameter.",
            )


def make_watermaze(
    dims: WatermazeDims | None = None,
    bin_size: float = 2.0,
    platform_position: tuple[float, float] | None = None,
) -> MazeEnvironments:
    """Create a Morris Water Maze environment.

    Creates a circular pool environment with a hidden platform and quadrant regions.
    This is an open-field maze with no track graph (animals can swim freely in any
    direction).

    Parameters
    ----------
    dims : WatermazeDims, optional
        Maze dimensions. If None, uses default dimensions
        (150 cm pool diameter, 15 cm platform diameter).
    bin_size : float, optional
        Spatial bin size in cm (default: 2.0).
    platform_position : tuple[float, float], optional
        Position (x, y) of the platform center in cm. If None, defaults to
        the centre of the NE quadrant, halfway between the pool centre and the
        wall along the NE diagonal: ``(R / (2 * sqrt(2)), R / (2 * sqrt(2)))``
        with ``R = pool_diameter / 2``. The whole platform must lie in the
        pool: its centre at most ``R - platform_radius`` from the pool centre.

    Returns
    -------
    MazeEnvironments
        Contains:
        - env_2d: 2D circular environment
        - env_track: None (open field has no track topology)

    Raises
    ------
    ValueError
        If the platform does not fit in the pool at ``platform_position``.

    Notes
    -----
    The maze is centered at the origin:
    - Pool boundary is a circle with radius R = pool_diameter / 2
    - North is +y and east is +x
    - Platform defaults to the centre of the NE quadrant

    Landmarks and regions:

    - platform: the hidden platform, a landmark with role ``"platform"``
      and a circular region of the platform's diameter, for platform
      crossings and time in the platform zone.
    - NE, NW, SE, SW: the pool's quarters, bounded by the north-south and
      east-west lines through its centre (NE is x > 0, y > 0). They are
      regions only, for quadrant occupancy in probe trials; together they
      cover the pool without overlapping. A bin whose centre lies exactly on
      a dividing line belongs to both quadrants it touches.

    Examples
    --------
    Create a default water maze:

    >>> maze = make_watermaze()
    >>> maze.env_2d.units
    'cm'
    >>> maze.env_2d.n_bins > 0
    True
    >>> maze.env_track is None
    True

    Create with custom dimensions:

    >>> dims = WatermazeDims(pool_diameter=200.0, platform_radius=10.0)
    >>> maze = make_watermaze(dims=dims, bin_size=4.0)
    >>> round(maze.env_2d.regions["platform"].data.area)  # pi * 10**2, cm^2
    314

    Custom platform position:

    >>> maze = make_watermaze(platform_position=(20.0, 30.0))
    >>> import numpy as np
    >>> platform = maze.env_2d.landmarks["platform"]
    >>> np.allclose(platform.data, [20.0, 30.0])
    True

    See Also
    --------
    WatermazeDims : Dimension specifications for Morris Water Maze.
    """
    if dims is None:
        dims = WatermazeDims()

    # Pool radius
    radius = dims.pool_diameter / 2.0

    # The NE quadrant's centre: R/2 from the pool centre along its diagonal
    c = radius / 2.0 / np.sqrt(2.0)

    # Default platform position: centre of the NE quadrant
    if platform_position is None:
        platform_position = (c, c)
    platform_x, platform_y = (float(v) for v in platform_position)
    distance = np.hypot(platform_x, platform_y)
    if not np.isfinite(distance) or distance + dims.platform_radius > radius:
        raise ValueError(
            f"platform_position={platform_position!r} puts the "
            f"{dims.platform_radius} cm-radius platform where it does not fit in "
            f"the pool, {radius} cm in radius around (0, 0): its centre must be at "
            f"most {radius - dims.platform_radius:.4g} cm from the pool centre. "
            "Move the platform toward the centre or use a smaller platform_radius."
        )

    # Create circular pool polygon
    pool_polygon = make_circular_arena(
        center=(0.0, 0.0),
        radius=radius,
    )

    # Create 2D environment
    env_2d = Environment.from_polygon(
        polygon=pool_polygon,
        bin_size=bin_size,
        name="watermaze",
        connect_diagonal_neighbors=True,
    )
    env_2d = env_2d.with_metadata(units="cm")

    platform_center = (platform_x, platform_y)
    region_builder = Regions()
    region_builder.add(
        "platform",
        polygon=make_circular_arena(
            center=platform_center, radius=dims.platform_radius
        ),
    )

    # Each quadrant is the pool's quarter on one side of both axes
    for name, quadrant in compass_quadrants(pool_polygon).items():
        region_builder.add(name, polygon=quadrant)

    platform = Landmark(
        "platform", "point", platform_center, metadata={"role": "platform"}
    )
    env_2d = env_2d.with_regions(region_builder).with_landmarks(Landmarks([platform]))

    # No track graph for open field
    env_track = None

    return MazeEnvironments(env_2d=env_2d, env_track=env_track)
