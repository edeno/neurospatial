"""Logical indexing from owned components, distinct from tensor-grid geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from neurospatial.geometry._arrays import ReadOnlyState
from neurospatial.geometry._binning import Binning, TrackLinearization
from neurospatial.geometry._locators import (
    HexLocator,
    MeshLocator,
    SubsetLocator,
    TransformedLocator,
)


@dataclass(frozen=True, eq=False)
class GridIndex(ReadOnlyState):
    """Logical grid indexing; hexagon indices do not assert rectangular cells."""

    edges: tuple[NDArray[np.float64], ...]
    shape: tuple[int, ...]
    active_mask: NDArray[np.bool_]


def grid_index(binning: Binning) -> GridIndex:
    """Return native logical indexing or a flat index for general cells.

    Hexagons and triangles keep their lattice through affine maps and
    subsets, with the kept cells active, as an axis-aligned grid keeps its
    shape. A rotated, reflected or sheared rectangular grid is indexed flat:
    its source axes are not world axes, so they are not offered as a
    world-aligned ``grid_shape``.
    """
    grid = binning.rectilinear
    if grid is not None:
        return GridIndex(grid.edges, grid.shape, grid.active_mask)
    track = binning.linearization
    if isinstance(track, TrackLinearization):
        return GridIndex(track.grid_edges, track.active_mask.shape, track.active_mask)
    lattice = _lattice(binning.locator)
    if lattice is not None:
        shape, mapping = lattice
        mask = (mapping >= 0).reshape(shape)
        mask.setflags(write=False)
        return GridIndex((), shape, mask)
    flat_mask = np.ones(binning.bins.n_bins, dtype=bool)
    flat_mask.setflags(write=False)
    return GridIndex((), (binning.bins.n_bins,), flat_mask)


def _lattice(locator: object) -> tuple[tuple[int, ...], NDArray[np.intp]] | None:
    """Hexagon or triangle lattice of a locator, through maps and subsets.

    Returns
    -------
    tuple or None
        The lattice shape and, for each lattice cell in row-major order, its
        current bin index or -1; None when the bins have no such lattice.
    """
    if isinstance(locator, HexLocator):
        return tuple(locator.centers_shape), locator.flat_to_active
    if isinstance(locator, MeshLocator):
        return (len(locator.simplices),), locator.simplex_to_active
    if isinstance(locator, TransformedLocator):
        return _lattice(locator.base)
    if isinstance(locator, SubsetLocator):
        parent = _lattice(locator.base)
        if parent is None:
            return None
        shape, mapping = parent
        child = np.full(mapping.shape, -1, dtype=np.intp)
        inside = mapping >= 0
        child[inside] = locator.parent_to_child[mapping[inside]]
        return shape, child
    return None
