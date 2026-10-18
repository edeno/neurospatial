"""How an environment's coordinates measure distance and name their axes.

Each environment declares ``coordinate_system`` so shared operations
ask it for coordinate math, length scales and plot labels.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import CartesianOnlyError

if TYPE_CHECKING:
    from neurospatial.geometry._binning import Binning


class CellLengthConsumer(Protocol):
    """Cell measures, dimension count and radial edges for length-scale defaults."""

    @property
    def binning(self) -> Binning: ...

    @property
    def n_dims(self) -> int: ...

    @property
    def grid_edges(self) -> tuple[NDArray[np.float64], ...] | None: ...


class CoordinateSystem(Protocol):
    """Physical measurement and axis names for an environment's bin centers."""

    equal_aspect: ClassVar[bool]

    @property
    def kind(self) -> Literal["cartesian", "polar"]:
        """Cartesian positions or egocentric ``(distance, angle)``."""
        ...

    @property
    def circular(self) -> bool:
        """Whether an angular axis wraps across its seam; False for Cartesian."""
        ...

    @property
    def periodic_axes(self) -> tuple[int, ...]:
        """Axes whose two ends are neighbors."""
        ...

    def squared_distances(self, centers: NDArray[np.float64]) -> NDArray[np.float64]:
        """Squared physical distances between every pair of points, ``(n, n)``."""
        ...

    def length_ranges(
        self, dimension_ranges: Sequence[tuple[float, float]]
    ) -> Sequence[tuple[float, float]]:
        """The ranges of the axes measured in length units."""
        ...

    def cartesian_embedding(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Points in a Cartesian frame where Euclidean distance is physical distance."""
        ...

    def fallback_lengths(self, env: CellLengthConsumer) -> NDArray[np.float64]:
        """Length scales when no connectivity edge has a positive distance."""
        ...

    def axis_labels(self, units: str | None) -> tuple[str, str]:
        """Labels for the first two axes of a field plot."""
        ...


@dataclass(frozen=True)
class CartesianSystem:
    """Bin centers are ``(x, y[, z])`` positions in one length unit.

    Attributes
    ----------
    kind : {'cartesian'}
        The coordinate kind.
    circular : bool
        Always False: no axis wraps.
    periodic_axes : tuple of int
        Always empty.
    """

    kind: ClassVar[Literal["cartesian"]] = "cartesian"
    circular: ClassVar[bool] = False
    periodic_axes: ClassVar[tuple[int, ...]] = ()
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

    def cartesian_embedding(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """The points themselves: Cartesian positions are already Cartesian."""
        return points

    def fallback_lengths(self, env: CellLengthConsumer) -> NDArray[np.float64]:
        """Each bin's own length: a track bin's length, a square's side, a cube's."""
        bins = env.binning.bins
        lengths = np.empty(bins.n_bins)
        for dim in np.unique(bins.measure_dims):
            measured = bins.measure_dims == dim
            lengths[measured] = bins.sizes[measured] ** (1.0 / int(dim))
        return lengths

    def axis_labels(self, units: str | None) -> tuple[str, str]:
        """Cartesian axis labels, including declared length units."""
        suffix = f" ({units})" if units else ""
        return f"X Position{suffix}", f"Y Position{suffix}"


@dataclass(frozen=True)
class EgocentricPolarSystem:
    """Bin centers are ``(distance, angle)``: a length and radians.

    Attributes
    ----------
    kind : {'polar'}
        The coordinate kind.
    circular : bool
        Whether the angular axis wraps across the seam.
    """

    circular: bool
    kind: ClassVar[Literal["polar"]] = "polar"
    equal_aspect: ClassVar[bool] = False

    @property
    def periodic_axes(self) -> tuple[int, ...]:
        """The angle axis, ``(1,)``, when it wraps; otherwise empty."""
        return (1,) if self.circular else ()

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

    def cartesian_embedding(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Place ``(distance, angle)`` rows in the animal's egocentric plane.

        Parameters
        ----------
        points : ndarray, shape (n_points, 2)
            ``(distance, angle)`` rows, angles in radians (0 = ahead,
            +π/2 = left).

        Returns
        -------
        ndarray, shape (n_points, 2)
            ``(distance · cos(angle), distance · sin(angle))``: x ahead of the
            animal and y to its left, in the distance units. Euclidean
            distance between two rows is the physical distance between the
            places they name, and angles a whole turn apart coincide.
        """
        distance, angle = points[:, 0], points[:, 1]
        return np.column_stack((distance * np.cos(angle), distance * np.sin(angle)))

    def fallback_lengths(self, env: CellLengthConsumer) -> NDArray[np.float64]:
        """Radial bin widths from the polar grid's distance axis."""
        edges = env.grid_edges
        assert edges is not None
        return np.diff(edges[0])

    def axis_labels(self, units: str | None) -> tuple[str, str]:
        """Distance and angle axis labels, with length units and radians."""
        suffix = f" ({units})" if units else ""
        return f"Distance{suffix}", "Angle (rad)"


def require_cartesian(
    coordinates: CartesianSystem | EgocentricPolarSystem,
    operation: str,
    *,
    why: str,
    fix: str,
    example: str | None = None,
) -> None:
    """Refuse an operation that reads polar ``(distance, angle)`` bins as Cartesian.

    Cartesian-only refusals that hold the environment's coordinate descriptor
    go through this check: the ``@cartesian_only`` Environment members, the
    Cartesian-only modes of shared methods, and analysis functions in other
    packages. Code that holds only an arena, such as ``boundary_geometry``,
    raises :class:`CartesianOnlyError` itself from ``arena.coordinate_kind``,
    with the same message layout.

    Parameters
    ----------
    coordinates : CartesianSystem or EgocentricPolarSystem
        The environment's coordinate descriptor, ``env.coordinate_system``.
    operation : str
        The refused call as the caller wrote it, e.g. ``"Environment.bin_at()"``
        or ``"rate_map_centroid(method='euclidean')"``.
    why : str
        One sentence on why the operation needs Cartesian coordinates.
    fix : str
        What works on a polar environment instead.
    example : str or None, optional
        Code for that alternative (see :class:`CartesianOnlyError`).

    Raises
    ------
    CartesianOnlyError
        If ``coordinates`` is an :class:`EgocentricPolarSystem`.
    """
    if coordinates.kind != "cartesian":
        raise CartesianOnlyError(operation, why=why, fix=fix, example=example)
