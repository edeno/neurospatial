"""Public arena classes for Environment geometry.

The implementations live in ``neurospatial.geometry``; this module is the
public import path.
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
