"""Morris Water Maze environment for spatial navigation research.

The Morris Water Maze (MWM) is a widely used behavioral test for spatial learning
and memory. Animals navigate in a circular pool to find a hidden platform, requiring
allocentric (world-centered) spatial memory and the use of distal cues.

Unlike corridor-based mazes (T-maze, Y-maze), the MWM is an open-field environment
with no physical constraints on movement paths. Animals can swim in any direction,
making it ideal for studying flexible spatial strategies.

Reference: Morris (1984). "Developments of a water-maze procedure for studying
spatial learning in the rat." Journal of Neuroscience Methods, 11(1), 47-60.

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

from neurospatial import Environment
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
        Diameter of the circular pool in cm. Default is 150.0.
    platform_radius : float
        Radius of the circular platform in cm. Default is 5.0.

    Examples
    --------
    >>> dims = WatermazeDims()
    >>> dims.pool_diameter
    150.0
    >>> dims.platform_radius
    5.0

    >>> custom = WatermazeDims(pool_diameter=200.0, platform_radius=10.0)
    >>> custom.pool_diameter
    200.0
    """

    pool_diameter: float = 150.0
    platform_radius: float = 5.0


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
        (150 cm pool diameter, 5 cm platform radius).
    bin_size : float, optional
        Spatial bin size in cm (default: 2.0).
    platform_position : tuple[float, float], optional
        Position (x, y) of the platform center in cm. If None, defaults to
        the center of the NE quadrant: (pool_diameter/4, pool_diameter/4).

    Returns
    -------
    MazeEnvironments
        Contains:
        - env_2d: 2D circular environment
        - env_track: None (open field has no track topology)

    Notes
    -----
    The maze is centered at the origin:
    - Pool boundary is a circle with radius = pool_diameter / 2
    - Platform defaults to center of NE quadrant

    Regions:
    - platform: Point region at platform location
    - NE: Northeast quadrant center (pool_diameter/4, pool_diameter/4)
    - NW: Northwest quadrant center (-pool_diameter/4, pool_diameter/4)
    - SE: Southeast quadrant center (pool_diameter/4, -pool_diameter/4)
    - SW: Southwest quadrant center (-pool_diameter/4, -pool_diameter/4)

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

    # Default platform position: center of NE quadrant
    if platform_position is None:
        platform_position = (dims.pool_diameter / 4, dims.pool_diameter / 4)

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
    env_2d.units = "cm"

    # Add platform region
    env_2d.regions.add("platform", point=platform_position)

    # Add quadrant regions at quadrant centers
    # Quadrant centers are at radius/2 from origin
    half_radius = radius / 2.0
    env_2d.regions.add("NE", point=(half_radius, half_radius))
    env_2d.regions.add("NW", point=(-half_radius, half_radius))
    env_2d.regions.add("SE", point=(half_radius, -half_radius))
    env_2d.regions.add("SW", point=(-half_radius, -half_radius))

    # No track graph for open field
    env_track = None

    return MazeEnvironments(env_2d=env_2d, env_track=env_track)
