"""Nearest-bin queries with the environment's cached KD-tree."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from neurospatial.environment._cache import KDTREE, AnalysisCache
from neurospatial.geometry._binning import Binning
from neurospatial.geometry._locators import NearestBinLocator
from neurospatial.ops.mapping import (
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
    tree = cache.get(KDTREE, lambda: _centers_tree(cache, binning))

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


def _centers_tree(cache: AnalysisCache, binning: Binning) -> cKDTree:
    """Build the KD-tree over the bin centers, reusing the locator's when it has one.

    A nearest-center locator over exactly these centers, in bin order, already
    searches them with a KD-tree. Reusing it keeps one tree per environment.
    """
    centers = binning.bins.centers
    locator = binning.locator
    if (
        isinstance(locator, NearestBinLocator)
        and locator.bin_indices is not None
        and np.array_equal(locator.bin_indices, np.arange(len(centers)))
        and np.array_equal(locator.centers, centers)
    ):
        return locator.kdtree(cache)
    return cKDTree(centers)
