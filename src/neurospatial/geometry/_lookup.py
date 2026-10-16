"""Assign positions to bins inside a known continuous domain."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from neurospatial.geometry._arena import Arena
    from neurospatial.geometry._binning import Binning


def locate(
    binning: Binning, arena: Arena, positions: NDArray[np.float64]
) -> NDArray[np.intp]:
    """Assign positions to active bins, applying known arena containment first.

    Parameters
    ----------
    binning : Binning
        Published bins and their point locator.
    arena : Arena
        Continuous domain; positions outside a known domain map to ``-1``.
    positions : ndarray of float64, shape (n_positions, n_dims)
        Query positions in the environment's coordinates.

    Returns
    -------
    ndarray of intp, shape (n_positions,)
        Active-bin index of each position, or ``-1`` outside.
    """
    if not arena.domain_known:
        return binning.locator.locate(positions).astype(np.intp, copy=False)
    n_dims = binning.bins.n_dims
    positions = np.atleast_2d(np.asarray(positions, dtype=np.float64))
    if positions.ndim != 2 or positions.shape[1] != n_dims:
        raise ValueError(
            f"Dimensionality mismatch: expected positions with shape "
            f"(n_positions, {n_dims}), got {positions.shape}."
        )
    inside = arena.contains(positions)
    indices = np.full(len(positions), -1, dtype=np.intp)
    if np.any(inside):
        indices[inside] = binning.locator.locate(positions[inside])
    return indices
