"""Logical indexing from owned components, distinct from tensor-grid geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NamedTuple

import numpy as np
from numpy.typing import NDArray

from neurospatial.geometry._arrays import ReadOnlyState, owned_grid
from neurospatial.geometry._binning import Binning, TrackLinearization
from neurospatial.geometry._locators import (
    HexLocator,
    MeshLocator,
    RectilinearLocator,
    SubsetLocator,
    TransformedLocator,
)


@dataclass(frozen=True, eq=False)
class GridIndex(ReadOnlyState):
    """Logical grid indexing; hexagon indices do not assert rectangular cells.

    Attributes
    ----------
    edges : tuple of ndarray
        Cell edges along each axis for rectangular and linear cells, or empty
        when the cells are hexagons, triangles or unstructured bins.
    shape : tuple of int
        Number of cells along each logical axis.
    active_mask : ndarray of bool, shape ``shape``
        Which logical cells are bins, read-only.
    """

    edges: tuple[NDArray[np.float64], ...]
    shape: tuple[int, ...]
    active_mask: NDArray[np.bool_]

    def __post_init__(self) -> None:
        if self.edges:
            edges, shape, mask = owned_grid(
                self.edges, self.shape, self.active_mask, name="GridIndex"
            )
        else:
            edges, shape = (), tuple(int(count) for count in self.shape)
            mask = np.array(self.active_mask, dtype=np.bool_, copy=True)
            if mask.shape != shape:
                raise ValueError(
                    f"GridIndex active_mask has shape {mask.shape}, not {shape}."
                )
            mask.setflags(write=False)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "active_mask", mask)


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
    lattice = cell_lattice(binning.locator)
    if lattice is not None and lattice.kind != "grid":
        mask = (lattice.flat_to_active >= 0).reshape(lattice.shape)
        mask.setflags(write=False)
        return GridIndex((), lattice.shape, mask)
    flat_mask = np.ones(binning.bins.n_bins, dtype=bool)
    flat_mask.setflags(write=False)
    return GridIndex((), (binning.bins.n_bins,), flat_mask)


class CellLattice(NamedTuple):
    """The logical cell lattice a locator's bins come from.

    Attributes
    ----------
    kind : {"grid", "hex", "triangle"}
        Rectangular cells with orthogonal neighbors, hexagons with six
        neighbors, or the triangles of a mesh.
    shape : tuple of int
        Logical lattice shape; ``(n_triangles,)`` for a mesh.
    flat_to_active : ndarray of intp, shape (prod(shape),)
        Current bin index of each lattice cell in row-major order, or -1.
    """

    kind: Literal["grid", "hex", "triangle"]
    shape: tuple[int, ...]
    flat_to_active: NDArray[np.intp]


def cell_lattice(locator: object) -> CellLattice | None:
    """The cell lattice of a locator's bins, through affine maps and subsets.

    An affine map keeps each cell's place in the lattice and a subset keeps
    some cells, so both keep the lattice. Clipping makes new cells, so a
    clipped locator has none.

    Returns
    -------
    CellLattice or None
        A writable lattice of the current bins; None when they have none.
    """
    if isinstance(locator, RectilinearLocator):
        return CellLattice("grid", locator.shape, locator.flat_to_active.copy())
    if isinstance(locator, HexLocator):
        return CellLattice(
            "hex", tuple(locator.centers_shape), locator.flat_to_active.copy()
        )
    if isinstance(locator, MeshLocator):
        return CellLattice(
            "triangle", (len(locator.simplices),), locator.simplex_to_active.copy()
        )
    if isinstance(locator, TransformedLocator):
        return cell_lattice(locator.base)
    if isinstance(locator, SubsetLocator):
        lattice = cell_lattice(locator.base)
        if lattice is not None:
            # Compose the subset map without indexing the -1 sentinel.
            mapping = lattice.flat_to_active
            inside = mapping >= 0
            mapping[inside] = locator.parent_to_child[mapping[inside]]
        return lattice
    return None
