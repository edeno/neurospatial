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
(2024), Brain Struct Funct 229(4), 823–841, doi:10.1007/s00429-024-02771-x,
Fig. 1c, where the hole centres lie at about 0.8 of the platform radius.

Gawel et al. (2018) also describe where the apparatus puts the animal and its
refuge. The escape box sits under one hole: "the escape box located under one
of the holes is often black", and the escape hole "is selected randomly for
each animal and in the classical BM protocol remains constant". In Barnes's
(1979) protocol "the rat was gently placed in the middle of the maze and the
start chamber was placed over it". Neither the escape box's nor the start
chamber's size is given. The light and noise that drive the animal off the
platform have no published placement, so the maze has no cue landmarks.

The platform's edge is a drop (``arena.outline == "drop"``): Gawel et al.
(2018) write that Barnes's apparatus "was located 91 cm above the floor to
prevent the animals (rats) from jumping down".

Examples
--------
>>> from neurospatial.simulation.mazes.barnes import make_barnes_maze, BarnesDims
>>> maze = make_barnes_maze()
>>> maze.env_2d.units
'cm'
>>> maze.env_2d.landmarks["escape_hole"].metadata["role"]
'escape_hole'
>>> len([r for r in maze.env_2d.regions.keys() if r.startswith("hole_")])
17
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
    check_count,
    check_lengths,
    refuse,
    warn_if_bins_too_coarse,
)
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

    def __post_init__(self) -> None:
        check_lengths(self, "diameter", "hole_radius", "hole_inset")
        check_count(
            self,
            "n_holes",
            2,
            "a Barnes maze needs the escape hole and at least one other hole",
        )
        if self.hole_inset < self.hole_radius:
            refuse(
                self,
                ("hole_inset", "hole_radius"),
                "puts each hole's centre closer to the rim than the hole's "
                "radius, so the holes run off the platform",
                "Use a hole_inset of at least hole_radius.",
            )
        # Neighbouring holes on the ring are 2 r sin(pi / n_holes) apart.
        ring_radius = self.diameter / 2.0 - self.hole_inset
        if ring_radius * np.sin(np.pi / self.n_holes) <= self.hole_radius:
            refuse(
                self,
                ("diameter", "n_holes", "hole_radius", "hole_inset"),
                "puts the holes so close together on their ring, "
                f"{ring_radius:.4g} cm from the centre, that neighbouring holes "
                "overlap",
                "Use fewer or smaller holes, or a smaller hole_inset.",
            )


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

    Every hole is a landmark in ``env_2d.landmarks`` with a region of the
    same name: a circle of the hole's diameter (9.5 cm by default). With
    bins coarser than about 6.7 cm (the hole's radius times sqrt(2)), a
    hole's circle can hold no bin centre; the maze then warns, naming a bin
    size that covers every hole.

    - escape_hole: the hole above the escape box, the hole at
      ``escape_hole_index``. Landmark role ``"escape_hole"``.
    - hole_i: every other hole, numbered i around the ring. Landmark role
      ``"hole"``; visits to these are the errors of a Barnes-maze trial.
    - start_box: the start chamber in the middle of the platform (Barnes
      1979). Landmark role ``"start_box"``; its size is not published, so
      its region is a point at the platform centre.

    The escape hole keeps one fixed name because its ring position changes
    between animals, so an analysis such as the time spent at the escape
    hole works for every animal unchanged. It is one hole, not a second
    region on top of ``hole_i``; ``hole_i`` for ``i = escape_hole_index``
    does not exist. Each hole's landmark records its ring position in
    ``metadata["index"]``, so the distance in holes from the escape hole is
    the difference of the two indices modulo ``n_holes``.

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
    >>> maze.env_2d.landmarks["escape_hole"].metadata["index"]
    9
    >>> "hole_9" in maze.env_2d.regions
    False

    Each hole's ring position, for hole-deviation analyses:

    >>> holes = {
    ...     name: lm.metadata["index"]
    ...     for name, lm in maze.env_2d.landmarks.items()
    ...     if lm.metadata["role"] in ("hole", "escape_hole")
    ... }
    >>> holes["escape_hole"], holes["hole_10"]
    (9, 10)

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
        arena=OpenArena(platform_polygon, outline="drop"),
        name="barnes_maze",
        connect_diagonal_neighbors=True,
        units="cm",
    )

    # Hole centres lie on a ring hole_inset in from the rim
    hole_placement_radius = radius - dims.hole_inset

    landmarks = []
    region_builder = Regions()
    for i in range(dims.n_holes):
        angle = 2 * np.pi * i / dims.n_holes
        center = (
            float(hole_placement_radius * np.cos(angle)),
            float(hole_placement_radius * np.sin(angle)),
        )
        is_escape = i == escape_hole_index
        name = "escape_hole" if is_escape else f"hole_{i}"
        role = "escape_hole" if is_escape else "hole"
        landmarks.append(
            Landmark(name, "point", center, metadata={"role": role, "index": i})
        )
        region_builder.add(
            name, polygon=make_circular_arena(center=center, radius=dims.hole_radius)
        )

    # Barnes (1979): the start chamber is placed over the rat in the middle
    # of the maze. Its size is not published, so its region is a point.
    landmarks.append(
        Landmark("start_box", "point", (0.0, 0.0), metadata={"role": "start_box"})
    )
    region_builder.add("start_box", point=(0.0, 0.0))

    env_2d = env_2d.with_regions(region_builder).with_landmarks(Landmarks(landmarks))

    # No track graph for open field
    env_track = None

    warn_if_bins_too_coarse(
        env_2d, maze="Barnes maze", bin_size=bin_size, narrowest_corridor=None
    )
    return MazeEnvironments(env_2d=env_2d, env_track=env_track)
