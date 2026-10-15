"""Logical indexing from owned components, distinct from tensor-grid geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from neurospatial.geometry._binning import Binning, TrackLinearization
from neurospatial.geometry._locators import HexLocator, MeshLocator


@dataclass(frozen=True, eq=False)
class GridIndex:
    """Logical grid indexing; hexagon indices do not assert rectangular cells."""

    edges: tuple[NDArray[np.float64], ...]
    shape: tuple[int, ...]
    active_mask: NDArray[np.bool_]


def grid_index(binning: Binning) -> GridIndex:
    """Return native logical indexing or a flat index for general cells."""
    grid = binning.rectilinear
    if grid is not None:
        return GridIndex(grid.edges, grid.shape, grid.active_mask)
    track = binning.linearization
    if isinstance(track, TrackLinearization):
        return GridIndex(track.grid_edges, track.active_mask.shape, track.active_mask)
    locator = binning.locator
    if isinstance(locator, HexLocator):
        mask = (locator.flat_to_active >= 0).reshape(locator.centers_shape)
        mask.setflags(write=False)
        return GridIndex((), locator.centers_shape, mask)
    if isinstance(locator, MeshLocator):
        mask = locator.simplex_to_active >= 0
        mask.setflags(write=False)
        return GridIndex((), (len(locator.simplices),), mask)
    flat_mask = np.ones(binning.bins.n_bins, dtype=bool)
    flat_mask.setflags(write=False)
    return GridIndex((), (binning.bins.n_bins,), flat_mask)
