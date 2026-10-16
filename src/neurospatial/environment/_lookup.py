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
    """Find nearest bin centers with the environment's cached KD-tree.

    A point with a NaN coordinate is a missing sample: bin ``-1`` and distance
    NaN. A point with an infinite coordinate is outside: bin ``-1`` and
    distance ``inf``. Only finite points are queried.
    """
    resolved = _validate_nearest_options(tie_break, max_distance, max_distance_factor)
    centers = binning.bins.centers
    tree = cache.get(KDTREE, lambda: cKDTree(centers))

    def query(
        rows: NDArray[np.float64],
    ) -> NDArray[np.int64] | tuple[NDArray[np.int64], NDArray[np.float64]]:
        return _nearest_bin_query(
            centers,
            tree,
            rows,
            tie_break=resolved,
            return_dist=return_dist,
            max_distance=max_distance,
            max_distance_factor=max_distance_factor,
        )

    array = np.asarray(points)
    if array.ndim != 2 or array.dtype.kind != "f":
        return query(points)
    finite = np.isfinite(array).all(axis=1)
    if finite.all():
        return query(points)

    bins = np.full(array.shape[0], -1, dtype=np.int64)
    distances = np.where(np.isnan(array).any(axis=1), np.nan, np.inf)
    if finite.any():
        found = query(array[finite])
        if isinstance(found, tuple):
            bins[finite], distances[finite] = found
        else:
            bins[finite] = found
    return (bins, distances) if return_dist else bins
