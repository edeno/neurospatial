"""Owned rectangular cells and capability checks on stored binning data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from neurospatial.geometry._arrays import ReadOnlyState


@dataclass(frozen=True, eq=False)
class RectilinearGrid(ReadOnlyState):
    """Axis-aligned rectangular cells, in the layout's own coordinates.

    Attributes
    ----------
    edges : tuple of ndarray
        Cell edges along each axis; ``edges[d]`` has shape ``(shape[d] + 1,)``.
        Spacing may vary along an axis.
    shape : tuple of int
        Number of cells along each axis, active or not.
    active_mask : ndarray of bool, shape ``shape``
        Which cells are environment bins; active bins are numbered in
        row-major order of this mask.
    """

    edges: tuple[NDArray[np.float64], ...]
    shape: tuple[int, ...]
    active_mask: NDArray[np.bool_]

    def __post_init__(self) -> None:
        edges = tuple(
            np.array(edge, dtype=np.float64, copy=True) for edge in self.edges
        )
        active_mask = np.array(self.active_mask, dtype=np.bool_, copy=True)
        for array in (*edges, active_mask):
            array.setflags(write=False)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "active_mask", active_mask)
