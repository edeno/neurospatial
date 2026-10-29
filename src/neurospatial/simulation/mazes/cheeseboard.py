"""Cheeseboard Maze environment for spatial navigation research.

The Cheeseboard Maze is a circular platform with a regular grid of reward locations
distributed across the entire surface. Reward locations contain food rewards (not
escape routes like in Barnes Maze). This design allows for studying spatial navigation,
working memory, and reference memory in an open-field setting with discrete targets.

Key contrast: Cheeseboard has reward locations distributed across entire surface;
Barnes Maze has holes only on the perimeter.

The defaults follow the rat cheeseboard described by Mátyás et al. (2025),
Brain Sci 15:881, doi:10.3390/brainsci15080881: "a 120 cm in diameter
circular board with 177 food wells (2.5 cm id)". Roux et al. (2017), Nat
Neurosci 20:845-853, doi:10.1038/nn.4543, give the mouse version: "a circular
platform 80 cm in diameter with 177 wells (1.5 mm deep; 4 mm in diameter; 5 cm
spacing between the wells)".

Mátyás et al. do not give the well spacing. The 7.5 cm default is derived,
not published: it scales the mouse board's 5 cm spacing by the ratio of board
diameters (120 / 80). On an origin-centred square lattice with 7.5 cm spacing,
keeping the wells whose centres lie within 54.6-57.1 cm of the centre gives
exactly 177 wells; the default 4 cm rim margin keeps wells within 56 cm. The
same construction with 5 cm spacing on the 80 cm mouse board also gives 177
wells.

Related work: Dupret et al. (2010), "The reorganization and reactivation of
hippocampal maps predict spatial memory performance", Nat Neurosci
13:995-1002, doi:10.1038/nn.2599, recorded rat CA1 and CA3 place cells while
rats learned three hidden reward locations on a cheeseboard maze. Gilbert et
al. (1998), "Memory for spatial location: role of the hippocampus in mediating
spatial pattern separation", J Neurosci 18:804-810,
doi:10.1523/JNEUROSCI.18-02-00804.1998, tested rats on a dry-land version of
the water maze, choosing between food wells along a row.

Any well can be baited: Dupret et al. (2010) placed "a new set of 3 hidden
rewards ... in randomly selected food-wells every day", so every well is a
reward-port landmark and none is fixed as the goal. Both studies run trials
from a start box beside the board (Mátyás et al.: "The start box with a
sliding door was used as a starting point for the learning trials"; "The
cheeseboard was cleaned and rotated 90 degrees relative to the start box
between learning trials"), but neither gives the box's size or where it
meets the board, so the maze has no start box. Dupret et al. placed cue
objects "near the baited food-wells" only in a cued condition, so the maze
has no cue landmarks.

Neither Mátyás et al. (2025) nor Roux et al. (2017) say whether the board was
raised or walled. It is modeled as an open board whose edge is a drop
(``arena.outline == "drop"``), the build of the Barnes maze it resembles.

Examples
--------
>>> from neurospatial.simulation.mazes.cheeseboard import (
...     make_cheeseboard_maze,
...     CheeseboardDims,
... )
>>> maze = make_cheeseboard_maze()
>>> maze.env_2d.units
'cm'
>>> len([k for k in maze.env_2d.regions.keys() if k.startswith("well_")])
177
>>> maze.env_2d.landmarks["well_7_7"].metadata["role"]
'reward_port'
>>> maze.env_track is None
True
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neurospatial.arena import OpenArena
from neurospatial.environment import Environment
from neurospatial.landmarks import Landmark, Landmarks
from neurospatial.regions import Regions
from neurospatial.simulation.mazes._base import (
    MazeDims,
    MazeEnvironments,
    check_lengths,
    refuse,
    warn_if_bins_too_coarse,
)
from neurospatial.simulation.mazes._geometry import make_circular_arena


@dataclass(frozen=True)
class CheeseboardDims(MazeDims):
    """Dimension specifications for Cheeseboard Maze.

    The Cheeseboard Maze is a circular platform with food wells on a square
    lattice across the entire surface, with one well at the centre. The
    defaults give the 120 cm rat board with 177 wells of 2.5 cm diameter
    (doi:10.3390/brainsci15080881).

    Attributes
    ----------
    diameter : float
        Diameter of the circular platform in cm. Default is 120.0.
    grid_spacing : float
        Distance in cm between neighbouring wells along the lattice rows and
        columns. Default is 7.5, derived by scaling the 5 cm spacing of the
        80 cm mouse board (doi:10.1038/nn.4543) to the 120 cm rat board.
    reward_radius : float
        Radius of each food well in cm. Default is 1.25 (2.5 cm diameter).
    rim_margin : float
        Smallest distance in cm from the board's rim to a well centre; lattice
        points closer to the rim have no well. Default is 4.0, keeping wells
        within 56 cm of the centre, which with 7.5 cm spacing on the 120 cm
        board gives exactly 177 wells (any margin from about 2.9 to 5.4 cm
        does).

    Examples
    --------
    >>> dims = CheeseboardDims()
    >>> dims.diameter
    120.0
    >>> dims.grid_spacing
    7.5
    >>> dims.reward_radius * 2  # well diameter, cm
    2.5

    >>> custom = CheeseboardDims(diameter=120.0, grid_spacing=10.0, reward_radius=2.0)
    >>> custom.diameter
    120.0
    """

    diameter: float = 120.0
    grid_spacing: float = 7.5
    reward_radius: float = 1.25
    rim_margin: float = 4.0

    def __post_init__(self) -> None:
        check_lengths(self, "diameter", "grid_spacing", "reward_radius")
        check_lengths(self, "rim_margin", allow_zero=True)
        if self.rim_margin < self.reward_radius:
            refuse(
                self,
                ("rim_margin", "reward_radius"),
                "lets wells closer to the rim than their radius, so they run "
                "off the board",
                "Use a rim_margin of at least reward_radius.",
            )
        if self.rim_margin >= self.diameter / 2.0:
            refuse(
                self,
                ("rim_margin", "diameter"),
                "leaves no room for wells on the board",
                "Use a rim_margin below half the diameter.",
            )
        if self.grid_spacing <= 2.0 * self.reward_radius:
            refuse(
                self,
                ("grid_spacing", "reward_radius"),
                "puts neighbouring wells so close that they overlap",
                "Use a grid_spacing greater than twice reward_radius.",
            )


def make_cheeseboard_maze(
    dims: CheeseboardDims | None = None,
    bin_size: float = 2.0,
) -> MazeEnvironments:
    """Create a Cheeseboard Maze environment.

    Creates a circular platform with food wells on a square lattice across the
    entire surface. Unlike Barnes Maze (holes only on perimeter), reward
    locations are distributed throughout the platform.

    Parameters
    ----------
    dims : CheeseboardDims, optional
        Maze dimensions. If None, uses the 120 cm rat board with 177 wells of
        2.5 cm diameter on a 7.5 cm lattice.
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

    Wells sit on a square lattice centred on the origin, so one well is at
    the centre of the board. Only lattice points at least ``rim_margin`` in
    from the rim get a well.

    Every food well is a landmark in ``env_2d.landmarks`` with role
    ``"reward_port"`` and ``port_type`` ``"food_well"``, and a point region
    of the same name at its centre:

    - well_i_j: the well in lattice column i and row j, counted from 0 at the
      lowest x and y. With the defaults ``well_7_7`` is the centre well.

    A well's region is a point, not a circle of its 2.5 cm diameter, because
    the well is smaller than a bin: a circle that small contains no bin
    centre for some wells at the default 2 cm bins (and for most at 3 cm),
    so it would cover no bins. A point region covers the one bin that
    contains the well centre at any bin size. The landmark records the
    well's diameter in ``metadata["diameter"]`` (cm). For a goal area, such
    as the 10 cm circle around a baited well that Dupret et al. (2010)
    used, add a region of that size.

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
    >>> len([k for k in maze.env_2d.regions.keys() if k.startswith("well_")]) > 0
    True

    Check that rewards are distributed across surface, not just perimeter:

    >>> maze = make_cheeseboard_maze()
    >>> len(maze.env_2d.regions)  # 177 wells
    177
    >>> well_names = [k for k in maze.env_2d.regions.keys() if k.startswith("well_")]
    >>> positions = np.array([maze.env_2d.regions[n].data for n in well_names])
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
        arena=OpenArena(platform_polygon, outline="drop"),
        name="cheeseboard_maze",
        connect_diagonal_neighbors=True,
    )
    env_2d = env_2d.with_metadata(units="cm")

    # Square lattice of wells centred on the origin, kept rim_margin in from
    # the rim
    max_well_radius = radius - dims.rim_margin
    n_half = int(np.floor(max_well_radius / dims.grid_spacing))
    coords = np.arange(-n_half, n_half + 1) * dims.grid_spacing

    landmarks = []
    region_builder = Regions()
    for i, x in enumerate(coords):
        for j, y in enumerate(coords):
            if np.hypot(x, y) <= max_well_radius:
                name, center = f"well_{i}_{j}", (float(x), float(y))
                landmarks.append(
                    Landmark(
                        name,
                        "point",
                        center,
                        metadata={
                            "role": "reward_port",
                            "port_type": "food_well",
                            "diameter": 2.0 * dims.reward_radius,
                        },
                    )
                )
                region_builder.add(name, point=center)
    env_2d = env_2d.with_regions(region_builder).with_landmarks(Landmarks(landmarks))

    # No track graph for open field
    env_track = None

    warn_if_bins_too_coarse(
        env_2d, maze="cheeseboard maze", bin_size=bin_size, narrowest_corridor=None
    )
    return MazeEnvironments(env_2d=env_2d, env_track=env_track)
