"""Morris Water Maze environment for spatial navigation research.

The Morris Water Maze (MWM) is a widely used behavioral test for spatial learning
and memory. Animals navigate in a circular pool to find a hidden platform, requiring
allocentric (world-centered) spatial memory and the use of distal cues.

Unlike corridor-based mazes (T-maze, Y-maze), the MWM is an open-field environment
with no physical constraints on movement paths. Animals can swim in any direction,
making it ideal for studying flexible spatial strategies.

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
quadrant's diagonal. North is +y and east is +x.

Examples
--------
>>> from neurospatial.simulation.mazes.watermaze import make_watermaze, WatermazeDims
>>> maze = make_watermaze()
>>> maze.env_2d.units
'cm'
>>> "platform" in maze.env_2d.regions
True
>>> "NE" in maze.env_2d.regions
True
>>> maze.env_track is None
True
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neurospatial.environment import Environment
from neurospatial.regions import Regions
from neurospatial.simulation.mazes._base import MazeDims, MazeEnvironments
from neurospatial.simulation.mazes._geometry import make_circular_arena


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
        with ``R = pool_diameter / 2``.

    Returns
    -------
    MazeEnvironments
        Contains:
        - env_2d: 2D circular environment
        - env_track: None (open field has no track topology)

    Notes
    -----
    The maze is centered at the origin:
    - Pool boundary is a circle with radius R = pool_diameter / 2
    - North is +y and east is +x
    - Platform defaults to the centre of the NE quadrant

    Regions (quadrant centres lie halfway between the pool centre and the
    wall, R/2 from the centre along each quadrant's diagonal, with
    ``c = R / (2 * sqrt(2))``):

    - platform: Point region at platform location
    - NE: Northeast quadrant centre (+c, +c)
    - NW: Northwest quadrant centre (-c, +c)
    - SE: Southeast quadrant centre (+c, -c)
    - SW: Southwest quadrant centre (-c, -c)

    The quadrant regions are useful for:
    - Analyzing spatial bias (time spent in each quadrant)
    - Measuring probe trial performance (target quadrant occupancy)
    - Studying search strategies

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
    >>> "platform" in maze.env_2d.regions
    True

    Custom platform position:

    >>> maze = make_watermaze(platform_position=(20.0, 30.0))
    >>> import numpy as np
    >>> platform = maze.env_2d.regions["platform"]
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

    # Quadrant centres: R/2 from the pool centre along each diagonal
    c = radius / 2.0 / np.sqrt(2.0)

    # Default platform position: centre of the NE quadrant
    if platform_position is None:
        platform_position = (c, c)

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

    # Add platform region
    region_builder = Regions()
    region_builder.add("platform", point=platform_position)

    # Add quadrant regions at quadrant centres
    region_builder.add("NE", point=(c, c))
    region_builder.add("NW", point=(-c, c))
    region_builder.add("SE", point=(c, -c))
    region_builder.add("SW", point=(-c, -c))
    env_2d = env_2d.with_regions(region_builder)

    # No track graph for open field
    env_track = None

    return MazeEnvironments(env_2d=env_2d, env_track=env_track)
