"""Nearest-bin queries with the environment's cached KD-tree."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from neurospatial._constants import KDTREE_LEAF_SIZE
from neurospatial.environment._cache import EMBEDDED_KDTREE, KDTREE, AnalysisCache
from neurospatial.geometry import (
    Binning,
    CartesianSystem,
    EgocentricPolarSystem,
    NearestBinLocator,
    require_position_columns,
)
from neurospatial.ops import _nearest_bin_query, _validate_nearest_options
from neurospatial.ops.mapping import TieBreakStrategy


def nearest_bin(
    cache: AnalysisCache,
    binning: Binning,
    points: NDArray[np.float64],
    *,
    coordinates: CartesianSystem | EgocentricPolarSystem,
    tie_break: TieBreakStrategy,
    return_dist: bool,
    max_distance: float | None,
    max_distance_factor: float | None,
) -> NDArray[np.intp] | tuple[NDArray[np.intp], NDArray[np.float64]]:
    """Find nearest bin centers with the environment's cached KD-tree.

    Distances are physical: on an egocentric polar environment both the bin
    centers and the points are first placed in the animal's Cartesian plane,
    ``(distance · cos(angle), distance · sin(angle))``.

    A point with a NaN coordinate is a missing sample: bin ``-1`` and distance
    NaN. A point with an infinite coordinate is outside: bin ``-1`` and
    distance ``inf``. Only finite points are queried. The number of coordinates
    is checked first, so an all-missing array of the wrong shape is refused.
    """
    points = np.asarray(points, dtype=np.float64)
    require_position_columns(
        points, binning.bins.n_dims, call="Environment.nearest_bin"
    )
    resolved = _validate_nearest_options(tie_break, max_distance, max_distance_factor)
    if coordinates.kind == "polar":
        centers, tree = cache.get(
            EMBEDDED_KDTREE, lambda: _embedded_tree(binning, coordinates)
        )
    else:
        centers, tree = binning.bins.centers, center_tree(cache, binning)

    def query(
        rows: NDArray[np.float64],
    ) -> NDArray[np.intp] | tuple[NDArray[np.intp], NDArray[np.float64]]:
        rows = np.asarray(rows)
        if rows.ndim == 2 and rows.shape[1] == binning.bins.n_dims:
            rows = coordinates.cartesian_embedding(rows.astype(np.float64))
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
    if array.ndim != 2 or array.dtype.kind != "f" or np.isfinite(array).all():
        found = query(points)
        if isinstance(found, tuple):
            return found[0].astype(np.intp, copy=False), found[1]
        return found.astype(np.intp, copy=False)
    finite = np.isfinite(array).all(axis=1)

    bins = np.full(array.shape[0], -1, dtype=np.intp)
    distances = np.where(np.isnan(array).any(axis=1), np.nan, np.inf)
    if finite.any():
        found = query(array[finite])
        if isinstance(found, tuple):
            bins[finite], distances[finite] = found
        else:
            bins[finite] = found
    return (bins, distances) if return_dist else bins


def _embedded_tree(
    binning: Binning, coordinates: EgocentricPolarSystem
) -> tuple[NDArray[np.float64], cKDTree]:
    """The bin centers in the egocentric Cartesian plane, and their KD-tree."""
    embedded = coordinates.cartesian_embedding(binning.bins.centers)
    embedded.setflags(write=False)
    return embedded, cKDTree(embedded, leafsize=KDTREE_LEAF_SIZE)


def center_tree(cache: AnalysisCache, binning: Binning) -> cKDTree:
    """Return the environment's one cached KD-tree over its bin centers."""
    return cache.get(KDTREE, lambda: _centers_tree(cache, binning))


def _centers_tree(cache: AnalysisCache, binning: Binning) -> cKDTree:
    """Build the KD-tree over the bin centers, reusing the locator's when it has one.

    A nearest-center locator over exactly these centers, in bin order, already
    searches them with a KD-tree. Reusing it keeps one tree per environment.
    Both use ``KDTREE_LEAF_SIZE``: queries are exact, but the leaf size decides
    which of several equidistant centers a single-neighbor query returns.
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
    return cKDTree(centers, leafsize=KDTREE_LEAF_SIZE)
