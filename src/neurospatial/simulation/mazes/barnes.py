"""Barnes Maze environment for spatial memory research.

The Barnes Maze (Barnes 1979) is a circular open platform with holes along the
perimeter. Animals explore the platform and must locate the single escape hole,
which leads to a dark refuge underneath. This design minimizes the aversive
water exposure of the Morris Water Maze while testing spatial memory.

The original Barnes (1979) design used 18 holes evenly spaced around a 120 cm
diameter circular platform. Unlike the radial arm maze, the Barnes Maze is an
open field - animals can explore freely without being confined to corridors.

Reference: Barnes, C.A. (1979). Memory deficits associated with senescence:
a neurophysiological and behavioral study in the rat. Journal of Comparative
and Physiological Psychology, 93(1), 74-104.

"the circular platform (120 cm diameter) had 18 holes (5 cm diameter) equally
spaced around the perimeter" (p. 75)

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

from neurospatial import Environment
from neurospatial.simulation.mazes._base import MazeDims, MazeEnvironments
from neurospatial.simulation.mazes._geometry import make_circular_arena


@dataclass(frozen=True)
class BarnesDims(MazeDims):
    """Dimension specifications for Barnes Maze.

    The Barnes Maze consists of a circular platform with holes evenly
    distributed along the perimeter. Dimensions are based on the original
    Barnes (1979) design.

    Attributes
    ----------
    diameter : float
        Diameter of the circular platform in cm. Default is 120.0 (Barnes 1979).
    n_holes : int
        Number of holes around the perimeter. Default is 18 (Barnes 1979).
    hole_radius : float
        Radius of each hole in cm. Default is 2.5 (5 cm diameter in original).

    Examples
    --------
    >>> dims = BarnesDims()
    >>> dims.diameter
    120.0
    >>> dims.n_holes
    18
    >>> dims.hole_radius
    2.5

    >>> custom = BarnesDims(diameter=150.0, n_holes=20, hole_radius=3.0)
    >>> custom.diameter
    150.0
    """

    diameter: float = 120.0
    n_holes: int = 18
    hole_radius: float = 2.5


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
        Maze dimensions. If None, uses default dimensions
        (120 cm diameter, 18 holes, 2.5 cm hole radius).
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

    Holes are positioned on the perimeter at:
    - Radius: (diameter/2) - hole_radius (slightly inside edge)
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
    env_2d.units = "cm"

    # Calculate hole positions on perimeter
    # Holes sit slightly inside the edge
    hole_placement_radius = radius - dims.hole_radius

    for i in range(dims.n_holes):
        angle = 2 * np.pi * i / dims.n_holes
        x = hole_placement_radius * np.cos(angle)
        y = hole_placement_radius * np.sin(angle)
        env_2d.regions.add(f"hole_{i}", point=(x, y))

    # Mark the escape hole
    escape_hole_pos = env_2d.regions[f"hole_{escape_hole_index}"].data
    env_2d.regions.add("escape_hole", point=tuple(escape_hole_pos))

    # No track graph for open field
    env_track = None

    return MazeEnvironments(env_2d=env_2d, env_track=env_track)
