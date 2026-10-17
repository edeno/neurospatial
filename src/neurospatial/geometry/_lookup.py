"""Assign positions to bins inside a known continuous domain."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error
from neurospatial.geometry._arena import EgocentricSurround

if TYPE_CHECKING:
    from neurospatial.geometry._arena import Arena
    from neurospatial.geometry._binning import Binning
    from neurospatial.geometry._locators import LookupIndexes


def locate(
    binning: Binning,
    arena: Arena,
    positions: NDArray[np.float64],
    indexes: LookupIndexes | None = None,
) -> NDArray[np.intp]:
    """Assign positions to active bins, applying known arena containment first.

    Parameters
    ----------
    binning : Binning
        Published bins and their point locator.
    arena : Arena
        Continuous domain; positions outside a known domain map to ``-1``.
    positions : ndarray of float64, shape (n_positions, n_dims)
        Query positions in the environment's coordinates. Egocentric polar
        angles are first wrapped into the angle range, so ``angle + 2π``
        finds the same bin as ``angle``.
    indexes : LookupIndexes, optional
        Keeps the locator's search structures, usually the environment's
        analysis cache. None builds them for this call only.

    Returns
    -------
    ndarray of intp, shape (n_positions,)
        Active-bin index of each position, or ``-1`` outside.

    Raises
    ------
    NotImplementedError
        If the bins lie along a track in two or more dimensions whose width
        is unknown (see :func:`require_track_width`).
    """
    if not arena.domain_known and binning.linearization is None:
        return binning.locator.locate(positions, indexes).astype(np.intp, copy=False)
    n_dims = binning.bins.n_dims
    positions = np.atleast_2d(np.asarray(positions, dtype=np.float64))
    if positions.ndim != 2 or positions.shape[1] != n_dims:
        raise ValueError(
            f"Dimensionality mismatch: expected positions with shape "
            f"(n_positions, {n_dims}), got {positions.shape}."
        )
    if not arena.domain_known:
        require_track_width(binning, arena)
        return binning.locator.locate(positions, indexes).astype(np.intp, copy=False)
    if isinstance(arena, EgocentricSurround):
        # An angle a whole turn outside the range names a direction inside it.
        positions = arena.wrap_angles(positions)
    inside = arena.contains(positions)
    indices = np.full(len(positions), -1, dtype=np.intp)
    if np.any(inside):
        indices[inside] = binning.locator.locate(positions[inside], indexes)
    return indices


TRACK_WIDTH_FIX = (
    "build the track with its measured full width, in the units of the "
    "positions, by passing track_width= to Environment.linear_track, "
    "Environment.maze or Environment.from_graph; for this track, with width "
    "its full width: "
)
TRACK_WIDTH_EXAMPLE = (
    "Environment.from_graph(env.binning.linearization.graph, "
    "list(env.binning.linearization.edge_order), "
    "bin_size=env.layout_parameters['bin_size'], "
    "edge_spacing=env.binning.linearization.edge_spacing, track_width=width, "
    "name=env.name, units=env.units, frame=env.frame, regions=env.regions)"
)


def require_track_width(binning: Binning, arena: Arena) -> None:
    """Refuse to map positions to a track's bins when its width is unknown.

    A centerline and a bin size do not say how far from the centerline a
    position is still on the track, so without a width every position, however
    far off the track, would be put in its nearest bin. A track whose
    positions have one coordinate has no off-track direction and is not
    refused.

    Parameters
    ----------
    binning : Binning
        The bins; a track carries a linearization.
    arena : Arena
        The continuous domain; a track built with a width knows it.

    Raises
    ------
    NotImplementedError
        If the bins lie along a track in two or more dimensions and the arena
        does not know the track's footprint.
    """
    if (
        binning.linearization is not None
        and binning.bins.n_dims >= 2
        and not arena.domain_known
    ):
        raise NotImplementedError(
            _format_error(
                "Positions cannot be assigned to the bins of a track built "
                "without track_width.",
                why="Why: a centerline and bin size do not say how far from the "
                "centerline a position is still on the track, so a position "
                "far off the track (a reflection or a tracking jump) would be "
                "put in its nearest bin.",
                fix=TRACK_WIDTH_FIX + TRACK_WIDTH_EXAMPLE,
            )
        )
