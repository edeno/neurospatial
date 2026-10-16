"""How an environment's coordinates measure distance and name their axes.

Each environment declares ``coordinate_system`` so shared operations
ask it for coordinate math, length scales and plot labels.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, Protocol

import numpy as np
from numpy.typing import NDArray


class CellLengthConsumer(Protocol):
    """Cell measures, dimension count and radial edges for length-scale defaults."""

    @property
    def bin_sizes(self) -> NDArray[np.float64]: ...

    @property
    def n_dims(self) -> int: ...

    @property
    def grid_edges(self) -> tuple[NDArray[np.float64], ...] | None: ...


class CoordinateSystem(Protocol):
    """Physical measurement and axis names for an environment's bin centers."""

    equal_aspect: ClassVar[bool]

    def squared_distances(self, centers: NDArray[np.float64]) -> NDArray[np.float64]:
        """Squared physical distances between every pair of points, ``(n, n)``."""
        ...

    def length_ranges(
        self, dimension_ranges: Sequence[tuple[float, float]]
    ) -> Sequence[tuple[float, float]]:
        """The ranges of the axes measured in length units."""
        ...

    def fallback_lengths(self, env: CellLengthConsumer) -> NDArray[np.float64]:
        """Length scales when no connectivity edge has a positive distance."""
        ...

    def axis_labels(self, units: str | None) -> tuple[str, str]:
        """Labels for the first two axes of a field plot."""
        ...


@dataclass(frozen=True)
class CartesianSystem:
    """Bin centers are ``(x, y[, z])`` positions in one length unit."""

    equal_aspect: ClassVar[bool] = True

    def squared_distances(self, centers: NDArray[np.float64]) -> NDArray[np.float64]:
        """Squared Euclidean distances; may be slightly negative from rounding."""
        bin_sq_norm = np.sum(centers**2, axis=1, keepdims=True)
        dist_sq: NDArray[np.float64] = (
            bin_sq_norm + bin_sq_norm.T - 2 * (centers @ centers.T)
        )
        return dist_sq

    def length_ranges(
        self, dimension_ranges: Sequence[tuple[float, float]]
    ) -> Sequence[tuple[float, float]]:
        """Every axis is a length."""
        return dimension_ranges

    def fallback_lengths(self, env: CellLengthConsumer) -> NDArray[np.float64]:
        """Edge length of a cube with each bin's volume."""
        return np.asarray(env.bin_sizes) ** (1.0 / env.n_dims)

    def axis_labels(self, units: str | None) -> tuple[str, str]:
        """Cartesian axis labels, including declared length units."""
        suffix = f" ({units})" if units else ""
        return f"X Position{suffix}", f"Y Position{suffix}"


@dataclass(frozen=True)
class EgocentricPolarSystem:
    """Bin centers are ``(distance, angle)``: a length and radians.

    Attributes
    ----------
    circular : bool
        Whether the angular axis wraps across the seam.
    """

    circular: bool
    equal_aspect: ClassVar[bool] = False

    def squared_distances(self, centers: NDArray[np.float64]) -> NDArray[np.float64]:
        """Squared physical distance, with angular wrapping only if circular."""
        # Egocentric polar env: centers[:, 0] is distance (length units),
        # centers[:, 1] is angle (radians). A naive Euclidean norm on
        # these columns collapses cm and radians into one scalar. Instead use
        # the physical polar distance between bin centers:
        #     d² = Δr² + (r̄ · Δθ)²
        # with r̄ the mean radius of the two bins. ``bandwidth`` is then a
        # single physical length (e.g. cm), consistent with the corrected
        # connectivity edge geometry.
        r = centers[:, 0]
        theta = centers[:, 1]
        d_r = r[:, None] - r[None, :]
        r_mean = 0.5 * (r[:, None] + r[None, :])
        d_theta_raw = theta[:, None] - theta[None, :]
        # Wrap the angular difference into [-pi, pi] ONLY when the angular axis
        # is circular, so bins straddling the -pi/+pi seam are treated as
        # adjacent (Delta theta ~ 0) rather than ~2*pi apart. Without this wrap
        # the seam gets a vanishing Gaussian weight, a hard artifact for a
        # full-circle egocentric-polar gaussian_kde. When the angular axis is
        # OPEN (circular_angle=False -- no seam edges in the graph), bins at -pi
        # and +pi are genuinely far apart, so wrapping would leak smoothing
        # across a boundary the caller deliberately left open; use the raw
        # angular difference instead. Circularity comes from the environment's
        # EgocentricSurround arena, which the layout's seam edges agree with
        # and which survives serialization. The diffusion_kde path is
        # unaffected: it smooths over the environment graph directly.
        if self.circular:
            d_theta = (d_theta_raw + np.pi) % (2.0 * np.pi) - np.pi
        else:
            d_theta = d_theta_raw
        arc = r_mean * d_theta
        dist_sq = d_r**2 + arc**2
        return dist_sq

    def length_ranges(
        self, dimension_ranges: Sequence[tuple[float, float]]
    ) -> Sequence[tuple[float, float]]:
        """Only the distance axis is a length."""
        return dimension_ranges[:1]

    def fallback_lengths(self, env: CellLengthConsumer) -> NDArray[np.float64]:
        """Radial bin widths from the polar grid's distance axis."""
        edges = env.grid_edges
        assert edges is not None
        return np.diff(edges[0])

    def axis_labels(self, units: str | None) -> tuple[str, str]:
        """Distance and angle axis labels, with length units and radians."""
        suffix = f" ({units})" if units else ""
        return f"Distance{suffix}", "Angle (rad)"


_POLAR_COORDINATES = (
    "this is a polar environment whose bin_centers[:, 0] is distance and "
    "bin_centers[:, 1] is angle in radians, so operations that assume Cartesian "
    "(x, y[, z]) coordinates or a Cartesian grid would silently return "
    "geometric nonsense."
)


def _require_cartesian_coordinates(
    env: object, operation: str, *, instead: str
) -> None:
    """Refuse a domain function that reads polar bin centers as Cartesian.

    Parameters
    ----------
    env : Environment
        Environment whose bin centers the operation reads. Duck-typed
        objects without owned coordinates are treated as Cartesian.
    operation : str
        The refused call as the caller wrote it, e.g. ``"mean_position()"``
        or ``"rate_map_centroid(method='euclidean')"``.
    instead : str
        One sentence naming the polar-safe alternative.

    Raises
    ------
    CartesianOnlyError
        If ``env`` stores polar ``(distance, angle)`` bins.
    """
    # Duck-typed environments without owned coordinates are Cartesian arrays.
    if isinstance(getattr(env, "_coordinate_system", None), EgocentricPolarSystem):
        from neurospatial._exceptions import CartesianOnlyError

        raise CartesianOnlyError(
            type(env).__name__,
            operation,
            reason="Cartesian-coordinate operation",
            explanation=f"{_POLAR_COORDINATES} {instead}",
        )
