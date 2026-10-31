"""The physical apparatus around an environment's bins.

An environment's bins cover where the animal went or could go. Its arena
records the continuous space the bins sit in: the box, maze, track or
platform the animal is in, and what its edges are. ``env.arena`` returns it.
Analyses that need physical boundaries, such as border scores,
boundary-vector cells and gaze, read its walls and outline; an arena
inferred from the visited cells asserts no wall.

Arena kinds
-----------
OpenArena
    A known floor polygon, such as a walled box or a maze, with the kind of
    its outline and any named interior walls. ``Environment.from_polygon``
    gives one; pass ``arena=OpenArena(polygon)`` to other factories to
    declare the walls you measured.
MaskArena
    The union of grid cells, such as the cells the animal visited.
    ``Environment.from_samples``, ``open_field`` and ``from_grid_mask`` give
    one. Its outline is not a physical wall unless you say so.
TrackArena
    A track of known (or unknown) width around its centerline segments.
    ``linear_track``, ``maze`` and ``from_graph`` give one.
EgocentricSurround
    The distance and angle window around the animal of an egocentric polar
    environment (``from_polar_egocentric``). It has no physical walls.
UnknownArena
    No known geometry; only the number and kind of coordinates.

Boundary kinds
--------------
``BoundaryKind`` names what a physical boundary is, which decides what it
blocks. Every kind keeps the animal in:

- ``"wall"``: an opaque wall. Blocks movement and gaze.
- ``"transparent_wall"``: a clear wall, such as a plexiglass divider. Blocks
  movement; the animal sees through it.
- ``"drop"``: the edge of an elevated track or platform. Blocks movement;
  the animal sees past it.

Analyses of movement (border scores, boundary vectors) use every physical
boundary; analyses of gaze use only opaque walls. An animal climbing a wall
or leaning over a drop is not modeled.

Examples
--------
>>> import numpy as np
>>> from shapely.geometry import box
>>> from neurospatial import Environment
>>> from neurospatial.arena import OpenArena
>>> positions = np.random.default_rng(0).uniform(0.0, 100.0, (500, 2))
>>> type(Environment.from_samples(positions, bin_size=10.0).arena).__name__
'MaskArena'

Declare a 1 m elevated platform, whose edge is a drop rather than a wall:

>>> env = Environment.from_samples(
...     positions,
...     bin_size=10.0,
...     arena=OpenArena(box(0.0, 0.0, 100.0, 100.0), outline="drop"),
... )
>>> env.arena.outline
'drop'
"""

from neurospatial.geometry import (
    Arena,
    BoundaryKind,
    EgocentricSurround,
    MaskArena,
    OpenArena,
    TrackArena,
    UnknownArena,
)

__all__ = [
    "Arena",
    "BoundaryKind",
    "EgocentricSurround",
    "MaskArena",
    "OpenArena",
    "TrackArena",
    "UnknownArena",
]
