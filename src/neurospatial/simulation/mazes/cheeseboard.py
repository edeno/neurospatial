"""Cheeseboard Maze environment for spatial navigation research.

The Cheeseboard Maze is a circular platform with a regular grid of reward locations
distributed across the entire surface. Reward locations contain food rewards (not
escape routes like in Barnes Maze). This design allows for studying spatial navigation,
working memory, and reference memory in an open-field setting with discrete targets.

Key contrast: Cheeseboard has reward locations distributed across entire surface;
Barnes Maze has holes only on the perimeter.

References:
- Gilbert et al. (1998) - Spatial navigation impairments in mice
- Dupret et al. (2010) - Hippocampal replay in awake rats

Examples
--------
>>> from neurospatial.simulation.mazes.cheeseboard import (
...     make_cheeseboard_maze,
...     CheeseboardDims,
... )
>>> maze = make_cheeseboard_maze()
>>> maze.env_2d.units
'cm'
>>> len([k for k in maze.env_2d.regions.keys() if k.startswith("reward_")]) > 0
True
>>> maze.env_track is None
True
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neurospatial import Environment
from neurospatial.simulation.mazes._base import MazeDims, MazeEnvironments
from neurospatial.simulation.mazes._geometry import make_circular_arena


@dataclass(frozen=True)
class CheeseboardDims(MazeDims):
    """Dimension specifications for Cheeseboard Maze.

    The Cheeseboard Maze is a circular platform with reward locations
    distributed in a regular grid pattern across the entire surface.
    Reward locations contain food rewards (not escape routes).

    Attributes
    ----------
    diameter : float
        Diameter of the circular platform in cm. Default is 110.0.
    grid_spacing : float
        Spacing between reward locations in the grid in cm. Default is 9.0.
    reward_radius : float
        Radius of each reward location in cm. Default is 1.5.
        Used to determine boundary margin for location placement.

    Examples
    --------
    >>> dims = CheeseboardDims()
    >>> dims.diameter
    110.0
    >>> dims.grid_spacing
    9.0
    >>> dims.reward_radius
    1.5

    >>> custom = CheeseboardDims(diameter=120.0, grid_spacing=10.0, reward_radius=2.0)
    >>> custom.diameter
    120.0
    """

    diameter: float = 110.0
    grid_spacing: float = 9.0
    reward_radius: float = 1.5


def make_cheeseboard_maze(
    dims: CheeseboardDims | None = None,
    bin_size: float = 2.0,
) -> MazeEnvironments:
    """Create a Cheeseboard Maze environment.

    Creates a circular platform with reward locations distributed in a regular
    grid pattern across the entire surface. Unlike Barnes Maze (holes only on
    perimeter), reward locations are distributed throughout the platform.

    Parameters
    ----------
    dims : CheeseboardDims, optional
        Maze dimensions. If None, uses default dimensions
        (110 cm diameter, 9 cm grid spacing, 1.5 cm reward radius).
    bin_size : float, optional
        Spatial bin size in cm (default: 2.0).

    Returns
    -------
    MazeEnvironments
        Contains:
        - env_2d: 2D polygon-based environment (circular platform)
        - env_track: None (open field, no linearized track)

    Notes
    -----
    The maze is centered at the origin (0, 0).

    Reward locations are distributed in a regular grid pattern across the
    entire circular platform, not just on the perimeter. Each location is
    added as a point region with format 'reward_i_j' where i, j are grid
    indices.

    Only locations that fall within the circular boundary (radius - reward_radius)
    are included.

    Regions:
    - reward_i_j: Point region for each reward location within circular
      boundary, where i and j are grid indices starting from 0.

    Examples
    --------
    Create a default cheeseboard maze:

    >>> maze = make_cheeseboard_maze()
    >>> maze.env_2d.units
    'cm'
    >>> maze.env_2d.n_bins > 0
    True
    >>> maze.env_track is None
    True

    Create with custom dimensions:

    >>> dims = CheeseboardDims(diameter=120.0, grid_spacing=10.0)
    >>> maze = make_cheeseboard_maze(dims=dims, bin_size=4.0)
    >>> len([k for k in maze.env_2d.regions.keys() if k.startswith("reward_")]) > 0
    True

    Check that rewards are distributed across surface, not just perimeter:

    >>> maze = make_cheeseboard_maze()
    >>> reward_names = [
    ...     k for k in maze.env_2d.regions.keys() if k.startswith("reward_")
    ... ]
    >>> positions = np.array([maze.env_2d.regions[n].data for n in reward_names])
    >>> distances = np.sqrt(positions[:, 0] ** 2 + positions[:, 1] ** 2)
    >>> bool(np.min(distances) < 20.0)  # Has rewards near center
    True

    See Also
    --------
    CheeseboardDims : Dimension specifications for Cheeseboard Maze.
    """
    if dims is None:
        dims = CheeseboardDims()

    radius = dims.diameter / 2.0
    center = (0.0, 0.0)

    # Create circular platform polygon
    platform_polygon = make_circular_arena(center=center, radius=radius)

    # Create 2D environment
    env_2d = Environment.from_polygon(
        polygon=platform_polygon,
        bin_size=bin_size,
        name="cheeseboard_maze",
        connect_diagonal_neighbors=True,
    )
    env_2d.units = "cm"

    # Create regular grid of reward locations across entire surface
    # Grid extends from -radius to +radius in both dimensions
    # Calculate number of locations per side
    n_locations_per_side = int(np.ceil(dims.diameter / dims.grid_spacing)) + 1
    half_extent = (n_locations_per_side - 1) * dims.grid_spacing / 2

    # Generate grid positions centered at origin
    xs = np.linspace(-half_extent, half_extent, n_locations_per_side)
    ys = np.linspace(-half_extent, half_extent, n_locations_per_side)

    # Add reward locations that fall within circular boundary
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            # Check if within circular boundary (with reward_radius buffer)
            distance_from_center = np.sqrt(x**2 + y**2)
            if distance_from_center <= radius - dims.reward_radius:
                env_2d.regions.add(f"reward_{i}_{j}", point=(x, y))

    # No track graph for open field
    env_track = None

    return MazeEnvironments(env_2d=env_2d, env_track=env_track)
