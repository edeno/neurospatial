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

if TYPE_CHECKING:
    from neurospatial.environment.core import Environment


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
    >>> env.units = "cm"
    >>> maze = MazeEnvironments(env_2d=env)
    >>> maze.env_2d.units
    'cm'
    >>> maze.env_track is None
    True
    """

    env_2d: Environment
    env_track: Environment | None = None
