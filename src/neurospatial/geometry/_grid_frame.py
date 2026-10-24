"""Current rectangular cells in a retained source coordinate frame."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import product
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from neurospatial.geometry._arrays import ReadOnlyState
from neurospatial.geometry._capabilities import RectilinearGrid
from neurospatial.geometry._locators import (
    RectilinearLocator,
    SubsetLocator,
    TransformedLocator,
    _check_homogeneous,
)

if TYPE_CHECKING:
    from neurospatial.geometry._locators import Locator


@dataclass(frozen=True, eq=False)
class FramedGrid(ReadOnlyState):
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
    # Fixed by the map and the grid, so built once: the half-open source cells
    # and the bound on how far inverse-map roundoff moves a source coordinate
    # (None when the map is too ill-conditioned to resolve the cells).
    _cells: RectilinearLocator = field(init=False, repr=False)
    _sensitivity: NDArray[np.float64] | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.source) is not RectilinearGrid:
            raise TypeError("FramedGrid.source must be a RectilinearGrid.")
        grid = self.source
        if len(grid.shape) not in (2, 3):
            raise ValueError("FramedGrid needs a 2-D or 3-D source grid.")
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
        object.__setattr__(self, "forward", matrix)
        object.__setattr__(self, "inverse", inverse)
        source = self.source
        object.__setattr__(
            self,
            "_cells",
            RectilinearLocator(source.edges, source.shape, source.active_mask),
        )
        linear, inverse_linear = matrix[:-1, :-1], inverse[:-1, :-1]
        dims = len(linear)
        residual = np.abs(np.eye(dims) - inverse_linear @ linear)
        residual += 2 * _gamma(dims) * (np.abs(inverse_linear) @ np.abs(linear))
        sensitivity = (
            None
            if np.max(residual.sum(axis=1)) >= 0.5
            else 2 * np.linalg.solve(np.eye(dims) - residual, np.abs(inverse_linear))
        )
        if sensitivity is not None:
            sensitivity.setflags(write=False)
        object.__setattr__(self, "_sensitivity", sensitivity)

    def _map(
        self,
        points: NDArray[np.float64],
        matrix: NDArray[np.float64],
        *,
        inverse: bool = False,
    ) -> NDArray[np.float64]:
        values = np.asarray(points, dtype=np.float64)
        n_dims = len(self.source.shape)
        if values.ndim != 2 or values.shape[1] != n_dims:
            raise ValueError(
                f"Expected points of shape (n_points, {n_dims}), got {values.shape}."
            )
        result = np.full(values.shape, np.nan)
        finite = np.all(np.isfinite(values), axis=1)
        if inverse:
            result[finite] = (values[finite] - self.forward[:-1, -1]) @ matrix[
                :-1, :-1
            ].T
        else:
            result[finite] = values[finite] @ matrix[:-1, :-1].T + matrix[:-1, -1]
        return result

    def to_source(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Map ``(n_points, n_dims)`` world points into source coordinates."""
        return self._map(points, self.inverse, inverse=True)

    def _project(
        self,
        points: NDArray[np.float64],
        incoming: NDArray[np.float64] | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Project world queries with a forward-residual roundoff bound.

        ``incoming`` (shape (n_points, n_dims)) is a bound the world
        coordinates already carry, such as an enclosing map's roundoff.
        """
        if self._sensitivity is None:
            raise ValueError(
                "Grid projection has unresolved precision; recenter or rescale coordinates."
            )
        original = self.to_source(points)
        linear = self.forward[:-1, :-1]
        gamma = _gamma(len(linear))
        remapped = self.to_world(original)
        difference = np.abs(points - remapped)
        error = difference + 0.5 * (
            np.spacing(difference)
            + np.spacing(np.abs(remapped))
            + np.spacing(np.abs(points))
        )
        error += 2 * gamma * (np.abs(original) @ np.abs(linear).T)
        if incoming is not None:
            error += incoming
        return original, error @ self._sensitivity.T

    def _project_to_planes(
        self,
        points: NDArray[np.float64],
        planes: Sequence[NDArray[np.float64]],
        *,
        operation: str,
        incoming: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Recover center/edge planes within bounded inverse-map roundoff.

        Cap the bound at a quarter of the smallest source cell width on each
        axis. A query too imprecise for that resolution refuses; meaningful
        positive weights on inactive cells are never filled.
        """
        original, errors = self._project(points, incoming)
        for axis, coordinates in enumerate(planes):
            edges = self.source.edges[axis]
            cap = float(np.min(np.diff(edges))) / 4
            relevant = (original[:, axis] >= edges[0]) & (
                original[:, axis] <= edges[-1]
            )
            if np.any(relevant & (errors[:, axis] >= cap)):
                raise ValueError(
                    f"{operation} cannot resolve source coordinate planes; recenter or rescale coordinates."
                )
            insertion = np.searchsorted(coordinates, original[:, axis])
            lower = coordinates[np.clip(insertion - 1, 0, len(coordinates) - 1)]
            upper = coordinates[np.clip(insertion, 0, len(coordinates) - 1)]
            nearest = np.where(
                np.abs(original[:, axis] - lower) <= np.abs(original[:, axis] - upper),
                lower,
                upper,
            )
            snap = np.abs(original[:, axis] - nearest) <= np.minimum(
                errors[:, axis], cap
            )
            original[snap, axis] = nearest[snap]
        return original

    def _source_locator(self) -> RectilinearLocator:
        """Half-open source cells, numbering active cells in row-major order."""
        return self._cells

    def _locate(
        self,
        points: NDArray[np.float64],
        *,
        operation: str,
        half_open: bool = False,
        incoming: NDArray[np.float64] | None = None,
    ) -> NDArray[np.intp]:
        """Return the active bin of each world point, or -1 outside active cells.

        A point is mapped back to source coordinates and snapped onto a cell
        face it lies on within the inverse map's roundoff (see
        :meth:`_project_to_planes`). The half-open source cells then own it,
        so a point on a face shared by two cells belongs to the cell above
        it, as on an axis-aligned grid, however the map rounds.

        Parameters
        ----------
        points : ndarray of float64, shape (n_points, n_dims)
            World coordinates; non-finite rows return -1.
        operation : str
            Names the calling operation in a precision error.
        half_open : bool, default=False
            Leave points on the outline of the active cells unowned instead
            of closing it (see :meth:`RectilinearLocator.locate_half_open`).
        incoming : ndarray of float64, shape (n_points, n_dims), optional
            Bound the world coordinates already carry, added to the map's
            own roundoff.

        Returns
        -------
        ndarray of intp, shape (n_points,)
            Active-bin index of each point, or -1.
        """
        source = self._project_to_planes(
            points, self.source.edges, operation=operation, incoming=incoming
        )
        cells = self._source_locator()
        return cells.locate_half_open(source) if half_open else cells.locate(source)

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


def _gamma(n_dims: int) -> float:
    """Relative rounding bound of the ``n_dims``-term dot products in a projection."""
    eps = float(np.finfo(np.float64).eps)
    return (n_dims + 4) * eps / (1 - (n_dims + 4) * eps)


def _underflow_error(
    matrix: NDArray[np.floating[Any]], other: NDArray[np.floating[Any]] | None = None
) -> NDArray[np.float64]:
    """Absolute roundoff of products whose results fall below the normal range.

    Relative roundoff (``eps * |value|``) vanishes for subnormal and zero
    entries, such as the sine of a tiny rotation angle, but a multiply or
    divide there still rounds by up to ``smallest_subnormal`` in absolute
    terms. Inverting the stored inverse ``B`` moves those absolute errors
    through ``dA = A dB A``, so each entry is bounded by the operation count
    times ``smallest_subnormal`` times ``(|A| J |A|)``, with ``J`` all ones.
    With ``other``, it bounds the product ``matrix @ other`` instead.
    """
    floor = 16 * len(matrix) * np.finfo(np.float64).smallest_subnormal
    left = np.abs(matrix).sum(axis=1)
    right = (np.abs(matrix) if other is None else np.abs(other)).sum(axis=0)
    error: NDArray[np.float64] = floor * (1.0 + np.outer(left, right))
    return error


def frame_from_locator(locator: Locator) -> tuple[FramedGrid, NDArray[np.float64]]:
    """Recover stored axes/frame and propagated matrix roundoff; no builder."""
    if isinstance(locator, TransformedLocator):
        base, parent_error = frame_from_locator(locator.base)
        matrix = np.linalg.inv(locator.inverse)
        rounding = 16 * len(matrix) * np.finfo(float).eps
        inverse_error = rounding * (
            np.abs(matrix) @ np.abs(locator.inverse) @ np.abs(matrix)
        )
        inverse_error += 4 * np.spacing(np.abs(matrix))
        inverse_error += _underflow_error(matrix)
        error = (np.abs(matrix) + inverse_error) @ parent_error
        error += inverse_error @ np.abs(base.forward)
        error += rounding * (np.abs(matrix) @ np.abs(base.forward))
        error += _underflow_error(matrix, base.forward)
        return FramedGrid(base.source, matrix @ base.forward), error
    if isinstance(locator, SubsetLocator):
        base, error = frame_from_locator(locator.base)
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
        ), error
    if not isinstance(locator, RectilinearLocator):
        raise ValueError("FramedGrid requires a matching rectangular locator.")
    return FramedGrid(
        RectilinearGrid(locator.edges, locator.shape, locator.active_mask),
        np.eye(len(locator.shape) + 1),
    ), np.zeros((len(locator.shape) + 1,) * 2)


def _same_frame(left: FramedGrid | None, right: FramedGrid | None) -> bool:
    return left is right or (
        left is not None
        and right is not None
        and np.array_equal(left.forward, right.forward)
        and np.array_equal(left.source.active_mask, right.source.active_mask)
        and all(
            np.array_equal(a, b)
            for a, b in zip(left.source.edges, right.source.edges, strict=True)
        )
    )


def framed_locator(locator: Locator, frame: FramedGrid | None) -> Locator:
    """Give the affine step of a binning's locator the binning's frame.

    A binning's ``grid_frame`` is the one record of its frame: the locator's
    copy is always set from it here, and removed when it is None. The
    outermost :class:`TransformedLocator` then assigns points by the frame's
    half-open source cells. Below a :class:`SubsetLocator`, it gets the same
    map over the parent's active cells. Other locators are returned
    unchanged; publication validates that they match the frame.

    Parameters
    ----------
    locator : Locator
        The binning's locator.
    frame : FramedGrid or None
        The binning's ``grid_frame``.

    Returns
    -------
    Locator
        ``locator`` itself when it already carries this frame (or, for None,
        no frame), or a copy that does.
    """
    if isinstance(locator, TransformedLocator):
        if _same_frame(locator.frame, frame):
            return locator
        return locator._framed(frame)
    if isinstance(locator, SubsetLocator):
        parent = None
        if frame is not None:
            parent_mask = frame_from_locator(locator.base)[0].source.active_mask
            parent = FramedGrid(
                RectilinearGrid(frame.source.edges, frame.source.shape, parent_mask),
                frame.forward,
            )
        base = framed_locator(locator.base, parent)
        if base is locator.base:
            return locator
        return SubsetLocator(base, locator.parent_to_child)
    return locator
