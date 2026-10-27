"""Border and boundary-vector cell models: firing driven by the walls.

``BorderCellModel`` fires as a function of the distance to the nearest wall,
the simple wall-hugging cell that border scores look for.
``BoundaryVectorCellModel`` is the boundary vector cell of Hartley et al.
(2000) and Barry et al. (2006): every piece of wall the animal can see adds
firing according to its distance and allocentric direction.

Both measure the walls themselves (the arena's declared outline and internal
walls), not the edges of the bins, so a rate depends on the animal's exact
position and not on the bin size.

References
----------
Hartley, T., Burgess, N., Lever, C., Cacucci, F., & O'Keefe, J. (2000).
    Modeling place fields in terms of the cortical inputs to the
    hippocampus. Hippocampus, 10, 369-379.
Barry, C., Lever, C., Hayman, R., Hartley, T., Burton, S., O'Keefe, J.,
    Jeffery, K., & Burgess, N. (2006). The boundary vector cell model of
    place cell firing and spatial memory. Reviews in the Neurosciences, 17,
    71-97.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error
from neurospatial.geometry import (
    boundary_geometry,
    require_cartesian,
    require_position_columns,
    valid_observers,
)
from neurospatial.ops.boundary_vectors import boundary_directions, nearest_boundary
from neurospatial.simulation._walls import Walls

if TYPE_CHECKING:
    from neurospatial.environment import Environment
    from neurospatial.geometry._boundaries import BoundarySource

__all__ = ["BorderCellModel", "BoundaryVectorCellModel"]

# Rays per turn summed by BoundaryVectorCellModel: one every 2 degrees, the
# angular resolution (``dtheta``) RatInABox's BoundaryVectorCells integrate
# over (George et al. 2023).
_N_RAYS = 180


def _check_rates(max_rate: float, baseline_rate: float) -> None:
    if not (np.isfinite(max_rate) and max_rate > 0):
        raise ValueError(f"max_rate must be positive (got {max_rate})")
    if not (np.isfinite(baseline_rate) and baseline_rate >= 0):
        raise ValueError(f"baseline_rate must be non-negative (got {baseline_rate})")
    if baseline_rate >= max_rate:
        raise ValueError(
            f"baseline_rate ({baseline_rate}) must be less than max_rate ({max_rate})"
        )


def _check_positive(name: str, value: float, *, allow_zero: bool = False) -> None:
    ok = np.isfinite(value) and (value >= 0 if allow_zero else value > 0)
    if not ok:
        bound = "non-negative" if allow_zero else "positive"
        raise ValueError(
            _format_error(
                f"{name} must be finite and {bound}, got {value!r}.",
                fix=f"pass a finite {bound} {name} in the environment's units",
            )
        )


def _require_walls(
    env: Environment, boundary_source: BoundarySource, call: str
) -> None:
    """Refuse an environment whose walls are unknown, before any rate is asked."""
    require_cartesian(
        env.coordinate_system,
        call,
        why="It models firing from the distance and direction to walls, from "
        "the animal's allocentric Cartesian position.",
        fix="simulate on the allocentric environment the animal moves in",
    )
    boundary_geometry(env.binning, env.arena, boundary_source)


class BorderCellModel:
    """Firing as a function of the distance to the nearest wall.

    The rate is a Gaussian of the straight-line distance ``d`` from the
    animal to the closest wall point:

    ``rate = baseline_rate + (max_rate - baseline_rate) * exp(-(d - preferred_distance)² / (2 distance_tolerance²))``

    The closest wall point is always in view: a wall in the way would itself
    be closer.

    Parameters
    ----------
    env : Environment
        2-D environment whose arena declares the walls (for example
        ``Environment.from_polygon``).
    distance_tolerance : float
        Standard deviation of the distance tuning, in environment units.
    preferred_distance : float, default=0.0
        Distance from the wall at which the cell fires most, in environment
        units. 0 is a border cell, which fires along the walls.
    max_rate : float, default=15.0
        Peak firing rate in Hz. An illustrative value; it is not taken from a
        published measurement.
    baseline_rate : float, default=0.01
        Firing rate far from the preferred distance, in Hz. An illustrative
        value.
    boundary_source : {'physical', 'analysis'}, default='physical'
        The arena's declared walls, or the edges of the environment's bins.

    Raises
    ------
    ValueError
        If a distance or rate is not finite, or ``baseline_rate >= max_rate``.
    NotImplementedError
        If the arena has no known walls and ``boundary_source='physical'``
        (an elevated track, or bins inferred from positions).

    See Also
    --------
    BoundaryVectorCellModel : Firing from walls at a distance and direction.
    neurospatial.ops.nearest_boundary : Distance to the closest wall.

    Examples
    --------
    >>> import numpy as np
    >>> from shapely.geometry import box
    >>> from neurospatial import Environment
    >>> from neurospatial.simulation import BorderCellModel
    >>> env = Environment.from_polygon(box(0, 0, 100, 100), bin_size=2.0, units="cm")
    >>> cell = BorderCellModel(env, distance_tolerance=4.0, baseline_rate=0.0)
    >>> cell.firing_rate(np.array([[0.5, 50.0], [4.0, 50.0], [50.0, 50.0]])).round(2)
    array([14.88,  9.1 ,  0.  ])
    """

    def __init__(
        self,
        env: Environment,
        *,
        distance_tolerance: float,
        preferred_distance: float = 0.0,
        max_rate: float = 15.0,
        baseline_rate: float = 0.01,
        boundary_source: BoundarySource = "physical",
    ) -> None:
        _check_positive("distance_tolerance", distance_tolerance)
        _check_positive("preferred_distance", preferred_distance, allow_zero=True)
        _check_rates(max_rate, baseline_rate)
        _require_walls(env, boundary_source, "BorderCellModel()")
        self.env = env
        self.preferred_distance = float(preferred_distance)
        self.distance_tolerance = float(distance_tolerance)
        self.max_rate = float(max_rate)
        self.baseline_rate = float(baseline_rate)
        self.boundary_source = boundary_source

    def firing_rate(
        self,
        positions: NDArray[np.float64],
        times: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Firing rate at each position.

        Parameters
        ----------
        positions : NDArray[np.float64], shape (n_time, 2)
            Positions in the environment's coordinates.
        times : NDArray[np.float64], shape (n_time,), optional
            Unused; accepted for the ``NeuralModel`` protocol.

        Returns
        -------
        NDArray[np.float64], shape (n_time,)
            Rate in Hz; NaN for a position outside the walls, where the cell
            has no defined input.
        """
        distances, _directions = nearest_boundary(
            self.env, positions, boundary_source=self.boundary_source
        )
        tuning = np.exp(
            -0.5
            * ((distances - self.preferred_distance) / self.distance_tolerance) ** 2
        )
        rates: NDArray[np.float64] = (
            self.baseline_rate + (self.max_rate - self.baseline_rate) * tuning
        )
        return rates

    @property
    def ground_truth(self) -> dict[str, Any]:
        """The model's parameters.

        Returns
        -------
        dict
            ``preferred_distance``, ``distance_tolerance``, ``max_rate`` and
            ``baseline_rate``.
        """
        return {
            "preferred_distance": self.preferred_distance,
            "distance_tolerance": self.distance_tolerance,
            "max_rate": self.max_rate,
            "baseline_rate": self.baseline_rate,
        }


class BoundaryVectorCellModel:
    """Boundary vector cell: walls at a preferred distance and direction.

    Following Barry et al. (2006, eqs. 1-2), each piece of wall seen at
    distance ``r`` in allocentric direction ``θ``, subtending angle ``δθ``,
    adds ``g(r, θ) δθ`` to the cell's drive, with

    ``g(r, θ) = exp(-(r - preferred_distance)² / (2 distance_tolerance²)) · exp(κ (cos(θ - preferred_direction) - 1))``,

    and the drive sums over every direction. As in RatInABox's
    ``BoundaryVectorCells``, the sum casts a ray every 2° and takes the first
    wall along it, and the angular tuning is a von Mises function. The drive
    is scaled so that a straight wall at ``preferred_distance``, square to
    ``preferred_direction``, gives ``max_rate``. Because every visible wall
    contributes, two walls in the preferred direction (a corner) drive the
    cell above ``max_rate``, and a barrier inside the arena gives the cell a
    second field beside the barrier.

    Parameters
    ----------
    env : Environment
        2-D environment whose arena declares the walls.
    preferred_distance : float
        Distance to the wall that drives the cell most, in environment units.
    preferred_direction : float
        Allocentric direction of that wall in radians (0 = East, +pi/2 =
        North).
    distance_tolerance : float
        Standard deviation of the distance tuning, in environment units.
        Barry et al. (2006) let it grow linearly with the preferred
        distance, so a far-tuned cell needs a wider tolerance.
    direction_tolerance : float
        Half-width at half-maximum of the angular tuning, in radians. The
        von Mises concentration is ``κ = ln 2 / (1 - cos(direction_tolerance))``.
    max_rate : float, default=15.0
        Rate in Hz next to a straight wall at the preferred distance and
        direction. An illustrative value; it is not taken from a published
        measurement.
    baseline_rate : float, default=0.01
        Rate in Hz with no wall near the preferred distance and direction. An
        illustrative value.
    boundary_source : {'physical', 'analysis'}, default='physical'
        The arena's declared walls, or the edges of the environment's bins.

    Raises
    ------
    ValueError
        If a distance, angle or rate is not finite, or
        ``baseline_rate >= max_rate``.
    NotImplementedError
        If the arena has no known walls and ``boundary_source='physical'``.

    See Also
    --------
    BorderCellModel : Firing from the distance to the nearest wall.
    neurospatial.encoding.compute_boundary_vector_rate : The matching rate map.

    Examples
    --------
    A cell for a wall 10 cm to the west:

    >>> import numpy as np
    >>> from shapely.geometry import box
    >>> from neurospatial import Environment
    >>> from neurospatial.simulation import BoundaryVectorCellModel
    >>> env = Environment.from_polygon(box(0, 0, 100, 100), bin_size=2.0, units="cm")
    >>> cell = BoundaryVectorCellModel(
    ...     env,
    ...     preferred_distance=10.0,
    ...     preferred_direction=np.pi,
    ...     distance_tolerance=4.0,
    ...     direction_tolerance=np.pi / 6,
    ...     baseline_rate=0.0,
    ... )
    >>> cell.firing_rate(np.array([[10.0, 50.0], [90.0, 50.0]])).round(1)
    array([15.,  0.])
    """

    def __init__(
        self,
        env: Environment,
        *,
        preferred_distance: float,
        preferred_direction: float,
        distance_tolerance: float,
        direction_tolerance: float,
        max_rate: float = 15.0,
        baseline_rate: float = 0.01,
        boundary_source: BoundarySource = "physical",
    ) -> None:
        _check_positive("preferred_distance", preferred_distance, allow_zero=True)
        _check_positive("distance_tolerance", distance_tolerance)
        if not np.isfinite(preferred_direction):
            raise ValueError(
                f"preferred_direction must be a finite angle in radians, got "
                f"{preferred_direction!r}."
            )
        if not (np.isfinite(direction_tolerance) and 0 < direction_tolerance <= np.pi):
            raise ValueError(
                _format_error(
                    "direction_tolerance must be a half-width in (0, pi] radians, "
                    f"got {direction_tolerance!r}.",
                    fix="pass the angle at which the tuning falls to half, e.g. "
                    "np.pi / 6 for 30 degrees",
                )
            )
        _check_rates(max_rate, baseline_rate)
        _require_walls(env, boundary_source, "BoundaryVectorCellModel()")
        self.env = env
        self.preferred_distance = float(preferred_distance)
        self.preferred_direction = float(np.angle(np.exp(1j * preferred_direction)))
        self.distance_tolerance = float(distance_tolerance)
        self.direction_tolerance = float(direction_tolerance)
        self.max_rate = float(max_rate)
        self.baseline_rate = float(baseline_rate)
        self.boundary_source = boundary_source
        self._walls = Walls.from_geometry(
            boundary_geometry(env.binning, env.arena, boundary_source)
        )
        self._directions = boundary_directions(n_directions=_N_RAYS)
        # von Mises concentration with half-maximum at direction_tolerance.
        kappa = np.log(2.0) / (1.0 - np.cos(self.direction_tolerance))
        self._angular = np.exp(
            kappa * (np.cos(self._directions - self.preferred_direction) - 1.0)
        )
        # A straight wall at the preferred distance, square to the preferred
        # direction, lies at r = d / cos(θ - φ) along every ray toward it.
        facing = np.cos(self._directions - self.preferred_direction)
        with np.errstate(divide="ignore"):
            reference = np.where(facing > 0, self.preferred_distance / facing, np.inf)
        self._reference_drive = float(np.sum(self._drive_terms(reference)))

    def _drive_terms(self, distances: NDArray[np.float64]) -> NDArray[np.float64]:
        """Each ray's term ``g(r, θ)``, shape like ``distances`` (..., n_rays)."""
        radial = np.exp(
            -0.5
            * ((distances - self.preferred_distance) / self.distance_tolerance) ** 2
        )
        terms: NDArray[np.float64] = radial * self._angular
        return terms

    def firing_rate(
        self,
        positions: NDArray[np.float64],
        times: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Firing rate at each position.

        Parameters
        ----------
        positions : NDArray[np.float64], shape (n_time, 2)
            Positions in the environment's coordinates.
        times : NDArray[np.float64], shape (n_time,), optional
            Unused; accepted for the ``NeuralModel`` protocol.

        Returns
        -------
        NDArray[np.float64], shape (n_time,)
            Rate in Hz; NaN for a position outside the walls.
        """
        positions = np.asarray(positions, dtype=np.float64)
        require_position_columns(
            positions, 2, call="BoundaryVectorCellModel.firing_rate"
        )
        observed = valid_observers(
            self.env.binning, self.env.arena, positions, self.boundary_source
        )
        drive = np.full(len(positions), np.nan)
        if observed.any():
            distances = self._walls.ray_distances(positions[observed], self._directions)
            drive[observed] = self._drive_terms(distances).sum(axis=1)
        rates: NDArray[np.float64] = self.baseline_rate + (
            self.max_rate - self.baseline_rate
        ) * (drive / self._reference_drive)
        return rates

    @property
    def ground_truth(self) -> dict[str, Any]:
        """The model's parameters.

        Returns
        -------
        dict
            ``preferred_distance``, ``preferred_direction`` (radians),
            ``distance_tolerance``, ``direction_tolerance`` (radians),
            ``max_rate`` and ``baseline_rate``.
        """
        return {
            "preferred_distance": self.preferred_distance,
            "preferred_direction": self.preferred_direction,
            "distance_tolerance": self.distance_tolerance,
            "direction_tolerance": self.direction_tolerance,
            "max_rate": self.max_rate,
            "baseline_rate": self.baseline_rate,
        }
