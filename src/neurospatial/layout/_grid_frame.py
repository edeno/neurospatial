"""Current rectangular cells in a retained source coordinate frame."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from neurospatial.layout._capabilities import RectilinearGrid, require_rectilinear
from neurospatial.layout._locators import (
    RectilinearLocator,
    SubsetLocator,
    TransformedLocator,
    _check_homogeneous,
)

if TYPE_CHECKING:
    from neurospatial.layout._binning import Binning, Locator


@dataclass(frozen=True, eq=False)
class FramedGrid:
    """Owned current grid cells with an affine map into world coordinates.

    Parameters
    ----------
    source : RectilinearGrid
        Increasing source axes and active mask. Bin IDs retain the mask's
        row-major order, including after reflection.
    forward : ndarray, shape (n_dims + 1, n_dims + 1)
        Finite invertible homogeneous map from source to world coordinates.
        This describes current cells, independently of future rebin settings.
    """

    source: RectilinearGrid
    forward: NDArray[np.float64]
    inverse: NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.source) is not RectilinearGrid:
            raise TypeError("FramedGrid.source must be a RectilinearGrid.")
        grid = self.source
        if (
            len(grid.shape) not in (2, 3)
            or len(grid.edges) != len(grid.shape)
            or grid.active_mask.shape != grid.shape
            or any(
                not isinstance(n, (int, np.integer))
                or isinstance(n, (bool, np.bool_))
                or n <= 0
                for n in grid.shape
            )
            or any(
                edge.shape != (n + 1,)
                or not np.all(np.isfinite(edge))
                or np.any(np.diff(edge) <= 0)
                for edge, n in zip(grid.edges, grid.shape, strict=True)
            )
        ):
            raise ValueError(
                "FramedGrid needs finite increasing 2-D/3-D grid axes and a matching mask."
            )
        if np.asarray(self.forward).dtype.kind not in "iuf":
            raise ValueError("FramedGrid.forward must be real numeric data.")
        matrix = np.array(self.forward, dtype=np.float64, copy=True)
        _check_homogeneous(matrix)
        if matrix.shape != (len(grid.shape) + 1,) * 2:
            raise ValueError("FramedGrid dimensions must match its source grid.")
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            inverse = np.asarray(np.linalg.inv(matrix), dtype=np.float64)
        if not np.all(np.isfinite(inverse)):
            raise ValueError(
                "FramedGrid inverse is not representable; rescale coordinates."
            )
        matrix.setflags(write=False)
        inverse.setflags(write=False)
        object.__setattr__(
            self,
            "source",
            RectilinearGrid(
                grid.edges, tuple(int(n) for n in grid.shape), grid.active_mask
            ),
        )
        object.__setattr__(self, "forward", matrix)
        object.__setattr__(self, "inverse", inverse)

    def __setstate__(self, state: dict[str, Any]) -> None:
        for name, value in state.items():
            object.__setattr__(self, name, value)
        self.forward.setflags(write=False)
        self.inverse.setflags(write=False)

    def _map(
        self, points: NDArray[np.float64], matrix: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        values = np.asarray(points, dtype=np.float64)
        n_dims = len(self.source.shape)
        if values.ndim != 2 or values.shape[1] != n_dims:
            raise ValueError(
                f"Expected points of shape (n_points, {n_dims}), got {values.shape}."
            )
        result = np.full(values.shape, np.nan)
        finite = np.all(np.isfinite(values), axis=1)
        result[finite] = values[finite] @ matrix[:-1, :-1].T + matrix[:-1, -1]
        return result

    def to_source(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Map ``(n_points, n_dims)`` world points into source coordinates."""
        return self._map(points, self.inverse)

    def to_world(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Map ``(n_points, n_dims)`` source points into world coordinates."""
        return self._map(points, self.forward)

    def corners(self) -> NDArray[np.float64]:
        """Return world corners of the full source grid's bounding box."""
        return self.to_world(
            np.array(
                list(product(*[(edge[0], edge[-1]) for edge in self.source.edges]))
            )
        )


def require_grid_context(
    binning: Binning, operation: str, *, fix: str
) -> tuple[RectilinearGrid, FramedGrid | None]:
    """Read source axes and their optional world frame without copying grids."""
    if binning.grid_frame is not None:
        return binning.grid_frame.source, binning.grid_frame
    return require_rectilinear(binning, operation, fix=fix), None


def frame_from_locator(locator: Locator) -> FramedGrid:
    """Recover stored source axes, ordered mask and composed frame; no builder."""
    if isinstance(locator, TransformedLocator):
        base = frame_from_locator(locator.base)
        return FramedGrid(base.source, np.linalg.inv(locator.inverse) @ base.forward)
    if isinstance(locator, SubsetLocator):
        base = frame_from_locator(locator.base)
        indices = np.flatnonzero(base.source.active_mask)
        selected = np.flatnonzero(locator.parent_to_child >= 0)
        selected = selected[np.argsort(locator.parent_to_child[selected])]
        indices = indices[selected]
        if np.any(np.diff(indices) <= 0):
            raise ValueError("FramedGrid requires row-major bin ordering.")
        mask = np.zeros(base.source.shape, dtype=bool)
        mask.flat[indices] = True
        return FramedGrid(
            RectilinearGrid(base.source.edges, base.source.shape, mask), base.forward
        )
    if not isinstance(locator, RectilinearLocator):
        raise ValueError("FramedGrid requires a matching rectangular locator.")
    mask = (
        np.ones(locator.shape, dtype=bool)
        if locator.active_mask is None
        else locator.active_mask
    )
    if locator.flat_to_active is not None and not np.array_equal(
        locator.flat_to_active[mask.ravel()], np.arange(np.count_nonzero(mask))
    ):
        raise ValueError("FramedGrid requires row-major locator bin ordering.")
    return FramedGrid(
        RectilinearGrid(locator.edges, locator.shape, mask),
        np.eye(len(locator.shape) + 1),
    )
