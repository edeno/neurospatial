"""Base classes for maze environments.

This module provides the foundation for all maze implementations:
- MazeDims: Base class for maze dimension specifications
- MazeEnvironments: Container for 2D and track graph environments

All mazes return MazeEnvironments containing:
- env_2d: A 2D polygon/grid-based Environment for spatial analysis
- env_track: Optional 1D linearized track graph for path analysis

Examples
--------
>>> from neurospatial.simulation.mazes._base import MazeDims, MazeEnvironments
>>> dims = MazeDims()
>>> dims  # doctest: +ELLIPSIS
MazeDims()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping

    from neurospatial.environment import Environment
    from neurospatial.regions import Region, Regions


MAZE_LANDMARK_ROLES = frozenset(
    {
        "reward_port",  # a well, port, food cup or feeder that delivers reward
        "hole",  # a Barnes-maze hole without an escape box
        "escape_hole",  # the Barnes-maze hole above the escape box
        "platform",  # a watermaze's hidden platform
        "start_box",
        "goal_box",
        "cue",  # a cue whose placement the source publishes
    }
)
"""Values of ``metadata["role"]`` on a built-in maze's landmarks."""


@dataclass(frozen=True)
class MazeDims:
    """Base class for maze dimension specifications.

    This is an empty base class that all maze dimension dataclasses
    inherit from. Subclasses define specific dimension parameters
    (e.g., arm_length, corridor_width) for each maze type.

    Attributes
    ----------
    None (base class has no attributes)

    Notes
    -----
    All MazeDims subclasses are frozen dataclasses, making dimension
    specifications immutable after creation. Lengths are in cm.

    ``arm_length`` is measured from the junction centre, where the
    corridors' centrelines meet, to the arm's far end wall in the T, Y, plus
    and W mazes. The radial-arm and repeated T mazes keep their source's
    measure (from the platform's edge, and along the walls between crossing
    alleys), so their published lengths stay as published; the repeated Y
    maze measures to the centre of an arm's rounded end. Each class's
    attributes say exactly what is measured.

    Examples
    --------
    >>> dims = MazeDims()
    >>> dims  # doctest: +ELLIPSIS
    MazeDims()
    """

    pass


@dataclass
class MazeEnvironments:
    """Container for 2D and track-graph maze environments.

    This dataclass holds the environments created by maze factory functions.
    Every maze provides a 2D environment; some also provide a 1D linearized
    track graph for analyzing movement along defined paths.

    Attributes
    ----------
    env_2d : Environment
        2D polygon/grid-based environment for spatial analysis.
        Always present and contains the full maze geometry.
    env_track : Environment | None
        1D linearized track graph environment, or None if not applicable.
        Present for corridor-based mazes (T-maze, Y-maze, etc.) but not
        for open-field mazes (watermaze, Barnes maze).

    Notes
    -----
    The apparatus is described twice, for two kinds of analysis. Each
    physical thing a maze's source places, a reward port, a hole, a hidden
    platform, a start or goal box, or a published cue, is a landmark in
    ``env_2d.landmarks`` whose ``metadata["role"]`` is one of
    ``MAZE_LANDMARK_ROLES``. Encoding functions take its name as an object
    position. Every landmark except a cue also has a region of the same name
    and its real size, for visits and occupancy. The track carries the same
    landmarks as the 2-D maze.

    Each maze declares what its edges are, as its source describes the
    apparatus: ``env_2d.arena.outline`` is ``"wall"`` for a walled maze and
    ``"drop"`` for an elevated track or platform, and named walls in
    ``env_2d.arena.walls`` carry their own kinds (the Sungod maze's clear
    walls between arms are ``"transparent_wall"``). Border scores and
    boundary-vector analyses use every edge; a gaze stops only at opaque
    walls, so it passes a drop edge or a clear wall.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.simulation.mazes._base import MazeEnvironments
    >>> positions = np.random.default_rng(42).uniform(0, 100, (500, 2))
    >>> env = Environment.from_samples(positions, bin_size=5.0)
    >>> env = env.with_metadata(units="cm")
    >>> maze = MazeEnvironments(env_2d=env)
    >>> maze.env_2d.units
    'cm'
    >>> maze.env_track is None
    True
    """

    env_2d: Environment
    env_track: Environment | None = None

    def __post_init__(self) -> None:
        # The track names the same wells, arm ends and goals as the 2-D maze,
        # each point placed at the nearest location on the track.
        if self.env_track is not None and not self.env_track.regions:
            self.env_track = self.env_track.with_regions(
                _regions_on_track(self.env_2d.regions, self.env_track)
            )
        # Landmarks are not bins, so they keep their 2-D positions.
        if self.env_track is not None and not self.env_track.landmarks:
            self.env_track = self.env_track.with_landmarks(self.env_2d.landmarks)


def _regions_on_track(regions: Mapping[str, Region], track: Environment) -> Regions:
    """Copy ``regions``, moving each point onto the track's nearest location.

    A simplified track graph need not pass through every well of the 2-D maze;
    projecting the point keeps the region on the track, where the animal's
    linearized position can reach it. Polygon regions are kept unchanged.
    """
    from neurospatial.regions import Regions

    on_track = Regions()
    for name, region in regions.items():
        if region.kind == "point":
            point = np.asarray(region.data, dtype=np.float64)[np.newaxis]
            nearest = track.linear_to_nd(track.to_linear(point))[0]
            on_track.add(name, point=nearest, metadata=region.metadata)
        else:
            on_track.add(name, polygon=region.data, metadata=region.metadata)
    return on_track
