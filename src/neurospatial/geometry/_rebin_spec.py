"""Owned construction choices for future re-binning, separate from provenance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from neurospatial.geometry._affine import _check_homogeneous
from neurospatial.geometry._arrays import ReadOnlyState


@dataclass(frozen=True, eq=False)
class GridBinningSpec(ReadOnlyState):
    """Rectangular lattice origin, frame and navigation choices.

    Parameters
    ----------
    origin : ndarray, shape (n_dims,)
        Lattice anchor in the original grid coordinates. New edges are integer
        multiples of the requested local spacing from this anchor.
    forward : ndarray, shape (n_dims + 1, n_dims + 1)
        Homogeneous affine map from grid coordinates to environment coordinates.
        New bin sizes are physical lengths along its column directions.
    connect_diagonal : bool, default=False
        Whether the new grid includes diagonal navigation connections.

    Notes
    -----
    This record describes future construction, not the current active bins.
    Cropping does not change it or the retained arena. It does not establish
    a physical footprint. Unsupported families have no grid specification.
    """

    origin: NDArray[np.float64]
    forward: NDArray[np.float64]
    connect_diagonal: bool = False

    def __post_init__(self) -> None:
        if any(
            np.asarray(value).dtype.kind not in "iuf"
            for value in (self.origin, self.forward)
        ):
            raise ValueError(
                "Grid specification origin and forward must be real numeric arrays."
            )
        origin = np.array(self.origin, dtype=np.float64, copy=True)
        matrix = np.array(self.forward, dtype=np.float64, copy=True)
        if origin.ndim != 1 or not len(origin) or not np.all(np.isfinite(origin)):
            raise ValueError(
                "Grid specification origin must be a finite coordinate vector."
            )
        _check_homogeneous(matrix)
        if matrix.shape != (len(origin) + 1, len(origin) + 1):
            raise ValueError(
                "Grid specification frame dimensions must match its origin."
            )
        if type(self.connect_diagonal) is not bool:
            raise ValueError("Grid specification connect_diagonal must be boolean.")
        origin.setflags(write=False)
        matrix.setflags(write=False)
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "forward", matrix)
