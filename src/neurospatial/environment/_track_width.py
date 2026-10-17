"""Give a track's bins a known width without rebuilding them."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error
from neurospatial.environment._publication import _Derivation
from neurospatial.geometry import Arena, Binning, TrackArena, _TransformedArena


def _centerline(arena: Arena) -> NDArray[np.float64] | None:
    """Return a track arena's centerline segments in its current coordinates."""
    if isinstance(arena, TrackArena):
        return arena.segments
    if isinstance(arena, _TransformedArena):
        segments = _centerline(arena.base)
        if segments is None:
            return None
        forward = arena.forward
        return np.asarray(
            segments @ forward[:-1, :-1].T + forward[:-1, -1], dtype=np.float64
        )
    return None


def _with_track_width(binning: Binning, arena: Arena, width: float) -> _Derivation:
    """Keep the bins and centerline of a track and set its full width.

    Parameters
    ----------
    binning : Binning
        The track's bins; they carry its linearization.
    arena : Arena
        The track's centerline footprint, with or without a width, in this
        frame or mapped into it by ``apply_transform``.
    width : float
        Full width of the track, in the units of the bin centers.

    Returns
    -------
    _Derivation
        The same bins with a ``TrackArena`` of that width around the
        centerline in the bins' coordinates.

    Raises
    ------
    TypeError
        If the environment is not a track built along a centerline.
    ValueError
        If ``width`` is not a finite positive number.
    """
    segments = _centerline(arena)
    if binning.linearization is None or segments is None:
        what = (
            "it is not a linearized track"
            if binning.linearization is None
            else f"its arena is a {type(arena).__name__}, not a track centerline"
        )
        raise TypeError(
            _format_error(
                f"with_track_width() needs a track environment, but {what}.",
                why="Why: a track width says how far from a track's centerline "
                "a position is still on the track; other environments take "
                "their extent from their bins or from arena=.",
                fix="build tracks with Environment.linear_track, Environment.maze "
                "or Environment.from_graph, which take track_width=; leave other "
                "environments as they are",
            )
        )
    if (
        isinstance(width, (bool, np.bool_))
        or not isinstance(width, (int, float, np.integer, np.floating))
        or not np.isfinite(width)
        or width <= 0
    ):
        raise ValueError(
            _format_error(
                f"with_track_width() takes a finite positive width, got {width!r}.",
                why="Why: the width is the full width of the track, so a position "
                "is on the track when it is within half of it from the centerline.",
                fix="pass the measured full width in the units of the bin centers, "
                "for example env.with_track_width(10.0) for a 10 cm wide track "
                "recorded in cm",
            )
        )
    return _Derivation(
        binning, "with_track_width", arena=TrackArena(segments, width=width)
    )
