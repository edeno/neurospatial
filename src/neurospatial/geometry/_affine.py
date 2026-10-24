"""Homogeneous affine maps with a bound on the roundoff of mapping points back."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error
from neurospatial.geometry._arrays import ReadOnlyState


def _check_homogeneous(matrix: NDArray[np.float64]) -> None:
    if (
        matrix.ndim != 2
        or matrix.shape[0] < 2
        or matrix.shape[0] != matrix.shape[1]
        or not np.all(np.isfinite(matrix))
        or not np.array_equal(matrix[-1], np.r_[np.zeros(len(matrix) - 1), 1.0])
    ):
        raise ValueError(
            "inverse/affine matrix must be finite, square and homogeneous."
        )
    # Only the sign is read; a near-identity map can underflow harmlessly in
    # the factorization, which some LAPACK builds report.
    with np.errstate(under="ignore"):
        sign = np.linalg.slogdet(matrix[:-1, :-1])[0]
    if sign == 0:
        raise ValueError("inverse/affine matrix must be invertible.")


def _gamma(n_dims: int) -> float:
    """Relative rounding bound of the ``n_dims``-term dot products in a projection."""
    eps = float(np.finfo(np.float64).eps)
    return (n_dims + 4) * eps / (1 - (n_dims + 4) * eps)


def _inverse_residual(
    linear: NDArray[np.float64], inverse_linear: NDArray[np.float64], gamma: float
) -> NDArray[np.float64]:
    """Componentwise bound on ``I - inverse_linear @ linear`` and its rounding.

    Both matrices have shape ``(n_dims, n_dims)``. When every row sums below
    one half, ``2 * solve(I - residual, |inverse_linear|)`` bounds how a
    forward residual carries back through the numerical inverse.
    """
    residual = np.abs(np.eye(len(linear)) - inverse_linear @ linear)
    residual += 2 * gamma * (np.abs(inverse_linear) @ np.abs(linear))
    return residual


def _forward_residual(
    query: NDArray[np.float64],
    original: NDArray[np.float64],
    remapped: NDArray[np.float64],
    linear: NDArray[np.float64],
    incoming: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Bound how far ``query`` is from the exact forward image of ``original``.

    ``query``, ``original`` (its inverse projection) and ``remapped`` (the
    rounded forward image of ``original``) have shape ``(n_points, n_dims)``;
    ``linear`` is the forward map's ``(n_dims, n_dims)`` linear part and
    ``incoming`` a bound ``query`` already carries. Translation is one
    rounded addition, not a dot product of origin-sized terms, so half-ULPs
    of the operands bound it without growing an origin buffer.
    """
    difference = np.abs(query - remapped)
    residual: NDArray[np.float64] = (
        difference
        + 0.5
        * (
            np.spacing(difference)
            + np.spacing(np.abs(remapped))
            + np.spacing(np.abs(query))
        )
        + 2 * _gamma(len(linear)) * (np.abs(original) @ np.abs(linear).T)
    )
    if incoming is not None:
        residual += incoming
    return residual


# Largest Gram residual of a normalized linear map that still counts as a
# similarity (rotation, reflection and uniform scale).
SIMILARITY_TOLERANCE = 1e-12


class LinearScale(NamedTuple):
    """Determinant, uniform scale and similarity residual of a linear map.

    Attributes
    ----------
    sign : float
        Sign of the determinant: 1, -1 for a reflection, 0 if singular.
    logdet : float
        Natural log of the determinant's absolute value.
    scale : float
        Geometric-mean length scale, ``exp(logdet / n_dims)``.
    residual : float
        ``max |N N^T - I|`` for ``N = linear / scale``; zero for an exact
        similarity, NaN when the product overflows.
    """

    sign: float
    logdet: float
    scale: float
    residual: float

    def is_similarity(self, slack: float = 0.0) -> bool:
        """Whether the map scales every direction alike, within ``slack``."""
        return bool(self.residual <= SIMILARITY_TOLERANCE + slack)


def linear_scale(linear: NDArray[np.float64]) -> LinearScale:
    """Measure how a ``(n_dims, n_dims)`` linear map scales lengths."""
    n_dims = len(linear)
    # Strong anisotropy can overflow the Gram product without being singular.
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        sign, logdet = np.linalg.slogdet(linear)
        scale = math.exp(float(logdet) / n_dims)
        normalized = linear / scale
        residual = float(np.max(np.abs(normalized @ normalized.T - np.eye(n_dims))))
    return LinearScale(float(sign), float(logdet), scale, residual)


@dataclass(frozen=True, eq=False)
class AffineMap(ReadOnlyState):
    """An invertible homogeneous affine map and the roundoff of inverting it.

    Bins, arenas and rotated grids that were moved by a map each hold one,
    so mapping a point back into their stored coordinates rounds the same
    way and carries the same error bound wherever it happens.

    Parameters
    ----------
    forward : ndarray of float64, shape (n_dims + 1, n_dims + 1)
        Finite invertible homogeneous map from stored to current coordinates.

    Attributes
    ----------
    inverse : ndarray of float64, shape (n_dims + 1, n_dims + 1)
        The numerical inverse of ``forward``.
    sensitivity : ndarray of float64, shape (n_dims, n_dims), or None
        Componentwise bound on how a forward residual in current coordinates
        carries back through ``inverse`` into stored coordinates. None when
        the map is too ill-conditioned to bound.

    Raises
    ------
    ValueError
        If ``forward`` is not a finite invertible homogeneous matrix, or its
        inverse exceeds floating-point range.
    """

    forward: NDArray[np.float64]
    inverse: NDArray[np.float64] = field(init=False, repr=False)
    sensitivity: NDArray[np.float64] | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        forward = np.array(self.forward, dtype=np.float64, copy=True)
        _check_homogeneous(forward)
        dims = len(forward) - 1
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            inverse = np.asarray(np.linalg.inv(forward), dtype=np.float64)
        if not np.all(np.isfinite(inverse)):
            raise ValueError(
                _format_error(
                    "The affine map's inverse exceeds floating-point range.",
                    fix="rescale the coordinates before transforming",
                )
            )
        linear, inverse_linear = forward[:-1, :-1], inverse[:-1, :-1]
        # A componentwise Neumann bound accounts for the numerical inverse.
        # The factor two leaves a margin for rounding in the small bound solve.
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            residual = _inverse_residual(linear, inverse_linear, _gamma(dims))
            sensitivity = (
                2 * np.linalg.solve(np.eye(dims) - residual, np.abs(inverse_linear))
                if np.all(np.isfinite(residual)) and np.max(residual.sum(axis=1)) < 0.5
                else None
            )
        if sensitivity is not None and not np.all(np.isfinite(sensitivity)):
            sensitivity = None
        for array in (forward, inverse, sensitivity):
            if array is not None:
                array.setflags(write=False)
        object.__setattr__(self, "forward", forward)
        object.__setattr__(self, "inverse", inverse)
        object.__setattr__(self, "sensitivity", sensitivity)

    @property
    def n_dims(self) -> int:
        """Number of coordinate axes."""
        return len(self.forward) - 1

    def same_as(self, other: AffineMap) -> bool:
        """Whether ``other`` is the same map, so both share stored coordinates."""
        return bool(np.array_equal(self.forward, other.forward))

    def apply(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Map ``(n_points, n_dims)`` stored coordinates to current ones."""
        return np.asarray(
            points @ self.forward[:-1, :-1].T + self.forward[:-1, -1],
            dtype=np.float64,
        )

    def project(
        self,
        query: NDArray[np.float64],
        incoming: NDArray[np.float64] | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Map points back to stored coordinates with a bound on their roundoff.

        Parameters
        ----------
        query : ndarray of float64, shape (n_points, n_dims)
            Current coordinates.
        incoming : ndarray of float64, shape (n_points, n_dims), optional
            Bound ``query`` already carries, such as an enclosing map's
            roundoff.

        Returns
        -------
        original : ndarray of float64, shape (n_points, n_dims)
            Stored coordinates; NaN for non-finite rows.
        error : ndarray of float64, shape (n_points, n_dims)
            Bound on how far each stored coordinate is from the point the
            forward map sends exactly to ``query``; zero for non-finite rows.

        Raises
        ------
        ValueError
            If the map is too ill-conditioned to bound its roundoff.
        """
        if self.sensitivity is None:
            raise ValueError(
                _format_error(
                    "Points cannot be mapped back through this affine map with a "
                    "bounded roundoff: the map is too ill-conditioned.",
                    fix="recenter or rescale the coordinates before transforming",
                )
            )
        original = np.full(query.shape, np.nan, dtype=np.float64)
        error = np.zeros(query.shape, dtype=np.float64)
        finite = np.all(np.isfinite(query), axis=1)
        if not np.any(finite):
            return original, error
        values = query[finite]
        linear, shift = self.forward[:-1, :-1], self.forward[:-1, -1]
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            source = (values - shift) @ self.inverse[:-1, :-1].T
            remapped = source @ linear.T + shift
            residual = _forward_residual(
                values,
                source,
                remapped,
                linear,
                None if incoming is None else incoming[finite],
            )
            error[finite] = residual @ self.sensitivity.T
        original[finite] = source
        return original, error
