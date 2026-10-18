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
    from neurospatial.environment import Environment
    from neurospatial.regions import Regions


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
    specifications immutable after creation.

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


def _regions_on_track(regions: Regions, track: Environment) -> Regions:
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
