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

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, fields
from numbers import Integral, Real
from typing import TYPE_CHECKING, NoReturn, cast

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping

    from shapely.geometry import MultiPolygon, Polygon

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
"""Values of ``metadata["role"]`` on a built-in maze's landmarks.

Every ``"reward_port"`` landmark also records the hardware in
``metadata["port_type"]``: ``"food_well"``, ``"food_cup"``,
``"plastic_cap"`` (a cap the reward is placed on) or ``"liquid_port"`` (a
port that dispenses liquid reward). Where a port sits in the maze is a
separate key, such as the Sungod maze's ``metadata["location"]``.
"""


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

    Every subclass checks its dimensions when it is built and refuses a
    length that is not positive and finite, a count too small to make the
    apparatus, and dimensions that would merge or misplace its parts:

    >>> from neurospatial.simulation.mazes import TMazeDims
    >>> TMazeDims(arm_length=-50.0)  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    ValueError: TMazeDims(arm_length=-50.0) is not a length: ...
    """

    pass


def check_lengths(dims: MazeDims, *names: str, allow_zero: bool = False) -> None:
    """Refuse any of ``names`` that is not a finite, positive length in cm.

    Parameters
    ----------
    dims : MazeDims
        The dimensions to check.
    *names : str
        Attributes of ``dims`` that are lengths.
    allow_zero : bool, default False
        Accept a length of zero, for a gap that may close.

    Raises
    ------
    ValueError
        If a length is not a real number, not finite, or negative (or zero,
        unless ``allow_zero``).
    """
    for name in names:
        value = getattr(dims, name)
        is_number = isinstance(value, Real) and not isinstance(value, bool)
        if (
            not is_number
            or not math.isfinite(value)
            or value < 0
            or (value == 0 and not allow_zero)
        ):
            kind = "zero or more" if allow_zero else "positive"
            raise ValueError(
                f"{type(dims).__name__}({name}={value!r}) is not a length: {name} "
                f"must be a finite number of cm, {kind}. Pass a value such as the "
                f"default, {name}={_default(dims, name)!r}."
            )


def check_count(dims: MazeDims, name: str, minimum: int, why: str) -> None:
    """Refuse a count that is not a whole number of at least ``minimum``.

    Parameters
    ----------
    dims : MazeDims
        The dimensions to check.
    name : str
        Attribute of ``dims`` that counts parts of the apparatus.
    minimum : int
        The smallest count that makes the apparatus.
    why : str
        Why the apparatus needs ``minimum``, for the message.

    Raises
    ------
    ValueError
        If the count is not an integer or is below ``minimum``.
    """
    value = getattr(dims, name)
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(
            f"{type(dims).__name__}({name}={value!r}) cannot be built: {why}. "
            f"Use a whole number {name} of at least {minimum}."
        )


def check_longer_than_wide(dims: MazeDims, name: str, part: str) -> None:
    """Refuse a corridor whose length ``name`` does not exceed its width.

    A corridor no longer than ``dims.corridor_width`` does not reach past the
    corridor it leaves, so a well or arm end half a width from its end lands
    on the junction, or behind it.

    Parameters
    ----------
    dims : MazeDims
        The dimensions to check; they have a ``corridor_width``.
    name : str
        The attribute holding the corridor's length.
    part : str
        What the corridor is, for the message ("arm", "stem", ...).

    Raises
    ------
    ValueError
        If ``getattr(dims, name) <= dims.corridor_width``.
    """
    if getattr(dims, name) <= dims.corridor_width:  # type: ignore[attr-defined]
        refuse(
            dims,
            (name, "corridor_width"),
            f"makes each {part} no longer than it is wide, so its end overlaps "
            "the junction it leaves",
            f"Use {name} greater than corridor_width.",
        )


def refuse(dims: MazeDims, names: Sequence[str], problem: str, fix: str) -> NoReturn:
    """Raise a ``ValueError`` naming the dimensions that conflict.

    Parameters
    ----------
    dims : MazeDims
        The dimensions being checked.
    names : sequence of str
        The attributes whose values conflict, shown in the message.
    problem : str
        What these values would do to the apparatus.
    fix : str
        How to choose values that build it.
    """
    values = ", ".join(f"{name}={getattr(dims, name)!r}" for name in names)
    raise ValueError(f"{type(dims).__name__}({values}) {problem}. {fix}")


def _default(dims: MazeDims, name: str) -> object:
    """The default of field ``name`` of ``dims``'s class."""
    return next(f.default for f in fields(dims) if f.name == name)


def warn_if_bins_too_coarse(
    env: Environment,
    *,
    maze: str,
    bin_size: float,
    narrowest_corridor: float | None,
) -> None:
    """Warn when bins are too coarse for a maze's corridors or regions.

    A 2-D maze keeps the bins whose centres lie on its floor. Bins wider
    than about half a corridor can leave a stretch of corridor with no bin
    centre, splitting the maze into pieces an animal could never walk
    between; and a region smaller than a bin can contain no bin centre, so
    visit and occupancy analyses refuse it.

    Parameters
    ----------
    env : Environment
        The maze's 2-D environment, with its regions.
    maze : str
        The maze's name, for the message.
    bin_size : float
        The bin size the maze was built with, in cm.
    narrowest_corridor : float or None
        Width of the maze's narrowest corridor or passage in cm, or None
        for an open field.

    Examples
    --------
    >>> import warnings
    >>> from neurospatial.simulation.mazes import make_crossword_maze
    >>> with warnings.catch_warnings(record=True) as caught:
    ...     warnings.simplefilter("always")
    ...     maze = make_crossword_maze(bin_size=5.0, include_track=False)
    >>> print(caught[0].message)  # doctest: +ELLIPSIS
    bin_size=5.0 cm splits the crossword maze into 2 unconnected pieces...
    """
    import shapely

    n_pieces = len(env.components())
    if n_pieces > 1:
        corridor = (
            ""
            if narrowest_corridor is None
            else f": its narrowest corridor is {narrowest_corridor:.2f} cm wide"
        )
        fix = (
            "a smaller bin_size"
            if narrowest_corridor is None
            else f"bin_size at most {_round_down(narrowest_corridor / 2)} cm, half "
            "the narrowest corridor"
        )
        warnings.warn(
            f"bin_size={bin_size} cm splits the {maze} into {n_pieces} "
            f"unconnected pieces{corridor}, and bins this coarse leave stretches "
            "of it without a bin, so paths and distances cannot cross them. "
            f"Use {fix}.",
            UserWarning,
            stacklevel=3,
        )

    # Every maze is built from its floor polygon.
    floor = cast("Polygon | MultiPolygon", env.arena.polygonal_domain())
    empty = [name for name in env.regions if len(env.bins_in_region(name)) == 0]
    if empty:
        # A disc of radius r always holds a bin centre when bin_size <= r
        # sqrt(2), and a point lies in an active bin when the bin's centre,
        # within bin_size / sqrt(2) of it, is on the floor.
        clearance = []
        for name in empty:
            region = env.regions[name]
            if region.kind == "point":
                clearance.append(floor.boundary.distance(shapely.Point(region.data)))
            else:
                clearance.append(shapely.maximum_inscribed_circle(region.data).length)
        largest = _round_down(min(clearance) * math.sqrt(2.0))
        warnings.warn(
            f"At bin_size={bin_size} cm, the {maze}'s region"
            f"{'s' if len(empty) > 1 else ''} {', '.join(empty)} cover"
            f"{'' if len(empty) > 1 else 's'} no bin, so visit and occupancy "
            "analyses refuse "
            f"{'them' if len(empty) > 1 else 'it'}. Use bin_size at most "
            f"{largest} cm, or add a larger region around "
            f"{'them' if len(empty) > 1 else 'it'} for visits.",
            UserWarning,
            stacklevel=3,
        )


def _round_down(value: float) -> float:
    """``value`` rounded down to 0.1."""
    return math.floor(value * 10.0) / 10.0


@dataclass(frozen=True)
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
    position. Every landmark except a cue also has a region of the same
    name, for visits and occupancy. The region is the thing's own extent
    where its source gives the size (Barnes holes, radial food cups, the
    watermaze platform, the crossword, Hamlet and repeated T boxes, Hampton
    Court's central enclosure), the hex the rat reaches a small hex port
    from, and a point where the size is not published (linear, T, W and
    cheeseboard wells, crossword caps, Sungod ports, the Barnes start box and
    the repeated Y boxes). The track carries the same landmarks and regions
    as the 2-D maze, each point region moved onto the track.

    The container is frozen, and a track must be a linearized track in the
    2-D maze's units.

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
        track = self.env_track
        if track is None:
            return
        if not track.is_linearized_track:
            raise ValueError(
                f"env_track must be a linearized track, got a {track.n_dims}-D "
                f"environment {track.name!r}. Build the track with "
                "Environment.from_graph, or pass env_track=None for an open field."
            )
        if track.units != self.env_2d.units:
            raise ValueError(
                f"env_track is in {track.units!r} but env_2d is in "
                f"{self.env_2d.units!r}, so positions would not match between "
                "them. Build both in the same units."
            )
        # The track names the same wells, arm ends and goals as the 2-D maze,
        # each point placed at the nearest location on the track.
        if not track.regions:
            track = track.with_regions(_regions_on_track(self.env_2d.regions, track))
        # Landmarks are not bins, so they keep their 2-D positions.
        if not track.landmarks:
            track = track.with_landmarks(self.env_2d.landmarks)
        object.__setattr__(self, "env_track", track)


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
