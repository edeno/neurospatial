"""Nearest-bin queries with the environment's cached KD-tree."""

from __future__ import annotations

from typing import Literal

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
