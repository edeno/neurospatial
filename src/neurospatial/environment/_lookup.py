"""Point-to-bin lookups shared by Environment methods and their helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from neurospatial.environment._cache import KDTREE, AnalysisCache
from neurospatial.geometry._binning import Binning
from neurospatial.ops.binning import (
    TieBreakStrategy,
    _nearest_bin_query,
    _validate_nearest_options,
)

if TYPE_CHECKING:
    from neurospatial.geometry._arena import Arena


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


def nearest_bin(
    cache: AnalysisCache,
    binning: Binning,
    points: NDArray[np.float64],
    *,
    tie_break: TieBreakStrategy | Literal["lowest_index", "closest_center"],
    return_dist: bool,
    max_distance: float | None,
    max_distance_factor: float | None,
) -> NDArray[np.int64] | tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Find nearest bin centers with the environment's cached KD-tree."""
    resolved = _validate_nearest_options(tie_break, max_distance, max_distance_factor)
    centers = binning.bins.centers
    return _nearest_bin_query(
        centers,
        cache.get(KDTREE, lambda: cKDTree(centers)),
        points,
        tie_break=resolved,
        return_dist=return_dist,
        max_distance=max_distance,
        max_distance_factor=max_distance_factor,
    )
