"""Barnes Maze environment for spatial memory research.

The Barnes Maze (Barnes 1979) is a circular open platform with holes along the
perimeter. Animals explore the platform and must locate the single escape hole,
which leads to a dark refuge underneath. This design minimizes the aversive
water exposure of the Morris Water Maze while testing spatial memory.

Unlike the radial arm maze, the Barnes Maze is an open field - animals can
explore freely without being confined to corridors.

The defaults follow the original apparatus of Barnes (1979), J Comp Physiol
Psychol 93:74-104, doi:10.1037/h0077579, as described by Gawel et al. (2018),
Naunyn-Schmiedeberg's Arch Pharmacol 392:1-18, doi:10.1007/s00210-018-1589-y:
Barnes, "in her original study, utilized a circular platform (122 cm in
diameter) with 18 circular holes (9.5 cm in diameter) evenly spaced around the
periphery." That account does not give the distance from the rim to the
holes; the 12 cm default is an estimate read from the Barnes maze drawn in Wijnen et al.
(2024), Brain Struct Funct 229:823-841, doi:10.1007/s00429-024-02771-x,
Fig. 1c, where the hole centres lie at about 0.8 of the platform radius.

Examples
--------
>>> from neurospatial.simulation.mazes.barnes import make_barnes_maze, BarnesDims
>>> maze = make_barnes_maze()
>>> maze.env_2d.units
'cm'
>>> "escape_hole" in maze.env_2d.regions
True
>>> len([r for r in maze.env_2d.regions.keys() if r.startswith("hole_")])
18
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neurospatial.environment import Environment
from neurospatial.regions import Regions
from neurospatial.simulation.mazes._base import MazeDims, MazeEnvironments
from neurospatial.simulation.mazes._geometry import make_circular_arena


@dataclass(frozen=True)
class BarnesDims(MazeDims):
    """Dimension specifications for Barnes Maze.

    The Barnes Maze consists of a circular platform with holes evenly
    distributed along the perimeter. The defaults are the original apparatus
    of Barnes (1979), doi:10.1037/h0077579, as reported by Gawel et al.
    (2018), doi:10.1007/s00210-018-1589-y: a 122 cm platform with 18 holes of
    9.5 cm diameter.

    Attributes
    ----------
    diameter : float
        Diameter of the circular platform in cm. Default is 122.0.
    n_holes : int
        Number of holes around the perimeter. Default is 18.
    hole_radius : float
        Radius of each hole in cm. Default is 4.75 (9.5 cm diameter).
    hole_inset : float
        Distance in cm from the platform's rim to each hole's centre, measured
        along the radius. Default is 12.0, an estimate read from Wijnen et al.
        (2024) Fig. 1c (hole centres at about 0.8 of the platform radius); the
        account of Gawel et al. (2018) does not give it.

    Examples
    --------
    >>> dims = BarnesDims()
    >>> dims.diameter
    122.0
    >>> dims.n_holes
    18
    >>> dims.hole_radius * 2  # hole diameter, cm
    9.5
    >>> dims.diameter / 2 - dims.hole_inset  # radius of the ring of holes, cm
    49.0

    >>> custom = BarnesDims(diameter=150.0, n_holes=20, hole_radius=3.0)
    >>> custom.diameter
    150.0
    """

    diameter: float = 122.0
    n_holes: int = 18
    hole_radius: float = 4.75
    hole_inset: float = 12.0


def make_barnes_maze(
    dims: BarnesDims | None = None,
    escape_hole_index: int = 0,
    bin_size: float = 2.0,
) -> MazeEnvironments:
    """Create a Barnes Maze environment.

    Creates a circular platform with holes evenly distributed around the
    perimeter. One hole is designated as the escape hole (goal). This is
    an open field environment - animals can explore freely without being
    confined to tracks or corridors.

    Parameters
    ----------
    dims : BarnesDims, optional
        Maze dimensions. If None, uses the original Barnes (1979) apparatus
        (122 cm diameter, 18 holes of 9.5 cm diameter), with hole centres an
        estimated 12 cm in from the rim.
    escape_hole_index : int, optional
        Index of the hole that is the escape hole (0 to n_holes-1).
        Default is 0.
    bin_size : float, optional
        Spatial bin size in cm (default: 2.0).

    Returns
    -------
    MazeEnvironments
        Contains:
        - env_2d: 2D circular platform environment
        - env_track: None (open field, no track graph)

    Notes
    -----
    The maze is centered at the origin (0, 0).

    Holes are evenly spaced on a ring around the periphery:

    - Radius: diameter/2 - hole_inset
    - Angles: 2π * i / n_holes for i = 0, 1, ..., n_holes-1

    Regions:
    - hole_0, hole_1, ..., hole_{n-1}: Point regions at each hole position
    - escape_hole: Point region at the escape hole (same position as one hole)

    Unlike T-Maze or Linear Track, the Barnes Maze has no track graph (env_track=None)
    because it is an open field environment.

    Examples
    --------
    Create a default Barnes maze:

    >>> maze = make_barnes_maze()
    >>> maze.env_2d.units
    'cm'
    >>> maze.env_2d.n_bins > 0
    True
    >>> maze.env_track is None
    True

    Create with custom dimensions:

    >>> dims = BarnesDims(diameter=150.0, n_holes=20)
    >>> maze = make_barnes_maze(dims=dims, bin_size=3.0)
    >>> "hole_19" in maze.env_2d.regions
    True

    Set a different escape hole:

    >>> maze = make_barnes_maze(escape_hole_index=9)
    >>> "escape_hole" in maze.env_2d.regions
    True

    See Also
    --------
    BarnesDims : Dimension specifications for Barnes Maze.
    """
    if dims is None:
        dims = BarnesDims()

    # Validate escape_hole_index
    if not (0 <= escape_hole_index < dims.n_holes):
        raise ValueError(
            f"escape_hole_index must be between 0 and {dims.n_holes - 1}, "
            f"got {escape_hole_index}"
        )

    # Create circular platform centered at origin
    radius = dims.diameter / 2.0
    platform_polygon = make_circular_arena(center=(0.0, 0.0), radius=radius)

    # Create 2D environment
    env_2d = Environment.from_polygon(
        polygon=platform_polygon,
        bin_size=bin_size,
        name="barnes_maze",
        connect_diagonal_neighbors=True,
    )
    env_2d = env_2d.with_metadata(units="cm")

    # Hole centres lie on a ring hole_inset in from the rim
    hole_placement_radius = radius - dims.hole_inset

    region_builder = Regions()
    for i in range(dims.n_holes):
        angle = 2 * np.pi * i / dims.n_holes
        x = hole_placement_radius * np.cos(angle)
        y = hole_placement_radius * np.sin(angle)
        region_builder.add(f"hole_{i}", point=(x, y))

    # Mark the escape hole
    escape_hole_pos = region_builder[f"hole_{escape_hole_index}"].data
    region_builder.add("escape_hole", point=tuple(escape_hole_pos))
    env_2d = env_2d.with_regions(region_builder)

    # No track graph for open field
    env_track = None

    return MazeEnvironments(env_2d=env_2d, env_track=env_track)
