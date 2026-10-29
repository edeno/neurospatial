"""Border and boundary-vector cell models: firing driven by the walls.

``BorderCellModel`` fires as a function of the distance to the nearest wall,
the simple wall-hugging cell that border scores look for.
``BoundaryVectorCellModel`` is the boundary vector cell of Hartley et al.
(2000) and Barry et al. (2006): every piece of wall the animal can see adds
firing according to its distance and allocentric direction.

Both measure the walls themselves (the arena's declared outline and internal
walls), not the edges of the bins, so a rate depends on the animal's exact
position and not on the bin size. Every physical boundary counts, whatever
its kind: opaque and transparent walls, and the drop edges of an elevated
track or platform. Boundary-vector cells respond to drops as to walls (Lever
et al., 2009), and most border cells keep their fields when the walls are
removed and only a drop edge confines the animal (van Wijngaarden et al.,
2020).

References
----------
Hartley, T., Burgess, N., Lever, C., Cacucci, F., & O'Keefe, J. (2000).
    Modeling place fields in terms of the cortical inputs to the
    hippocampus. Hippocampus, 10, 369-379.
Barry, C., Lever, C., Hayman, R., Hartley, T., Burton, S., O'Keefe, J.,
    Jeffery, K., & Burgess, N. (2006). The boundary vector cell model of
    place cell firing and spatial memory. Reviews in the Neurosciences, 17,
    71-97.
Lever, C., Burton, S., Jeewajee, A., O'Keefe, J., & Burgess, N. (2009).
    Boundary vector cells in the subiculum of the hippocampal formation.
    Journal of Neuroscience, 29(31), 9771-9777.
van Wijngaarden, J. B. G., Babl, S. S., & Ito, H. T. (2020).
    Entorhinal-retrosplenial circuits for allocentric-egocentric
    transformation of boundary coding. eLife, 9, e59816.
Høydal, Ø. A., Skytøen, E. R., Andersson, S. O., Moser, M.-B., & Moser,
    E. I. (2019). Object-vector coding in the medial entorhinal cortex.
    Nature, 568, 400-404.
George, T. M., Rastogi, M., de Cothi, W., Clopath, C., Stachenfeld, K., &
    Barry, C. (2024). RatInABox, a toolkit for modelling locomotion and
    neuronal activity in continuous environments. eLife, 13, e85274.
"""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error
from neurospatial._units import default_length
from neurospatial.geometry import (
    boundary_geometry,
    require_cartesian,
    valid_observers,
)
from neurospatial.ops.boundary_vectors import boundary_directions, nearest_boundary
from neurospatial.simulation._walls import Walls
from neurospatial.simulation.models.base import (
    _check_angle,
    _check_positive,
    _check_rates,
    _model_inputs,
)

if TYPE_CHECKING:
    from neurospatial.environment import Environment
    from neurospatial.geometry._boundaries import BoundarySource

__all__ = ["BorderCellModel", "BoundaryVectorCellModel"]

#: Default border-cell distance tolerance in centimeters, derived from the
#: border-field centers of Høydal et al. (2019); see ``BorderCellModel`` Notes.
BORDER_DISTANCE_TOLERANCE_CM = 9.8

#: Default boundary-vector preferred distance in centimeters: illustrative,
#: the middle of RatInABox's 5-30 cm tuning-distance range.
BVC_PREFERRED_DISTANCE_CM = 17.5

# Rays per turn summed by BoundaryVectorCellModel: one every 2 degrees, the
# angular resolution (``dtheta``) RatInABox's BoundaryVectorCells integrate
# over (George et al. 2024).
_N_RAYS = 180


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
    boundary_geometry(env.binning, env.arena, boundary_source, blocking="movement")


@dataclass(frozen=True, eq=False)
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
    distance_tolerance : float, optional
        Standard deviation of the distance tuning, in environment units. If
        None, 9.8 cm expressed in ``env.units`` (see Notes). An environment
        without metric length units needs an explicit value.
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

    Attributes
    ----------
    ground_truth : dict[str, Any]
        The values the model fires with, including the default tolerance.
        Every parameter is also an attribute holding the argument as passed,
        so ``distance_tolerance`` reads None when left to its default.
        ``dataclasses.replace`` builds a new model as the constructor would.

    Raises
    ------
    ValueError
        If a distance or rate is not finite, a rate is negative,
        ``baseline_rate > max_rate``, or ``distance_tolerance`` is None and
        ``env.units`` is not a metric length unit.
    CartesianOnlyError
        If the environment holds egocentric polar coordinates.
    NotImplementedError
        If the arena declares no physical boundary and
        ``boundary_source='physical'`` (bins inferred from positions).

    Notes
    -----
    The default ``distance_tolerance`` is derived from Høydal et al. (2019),
    whose mouse border-cell fields, defined as the area above 20% of the
    peak rate, had centers 6-14 cm from the wall (mean 8.8 cm). With
    ``preferred_distance=0`` the field above 20% of peak spans the walls out
    to ``σ √(2 ln 5)``, so its center lies at ``σ √(2 ln 5) / 2`` and the
    mean gives ``σ = 2 · 8.8 / √(2 ln 5) = 9.8 cm``.

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
    >>> positions = np.array([[0.5, 50.0], [4.0, 50.0], [50.0, 50.0]])
    >>> times = np.array([0.0, 0.1, 0.2])
    >>> cell.firing_rate(times, positions).round(2)
    array([14.88,  9.1 ,  0.  ])
    """

    env: Environment = field(repr=False)
    _: KW_ONLY
    distance_tolerance: float | None = None
    preferred_distance: float = 0.0
    max_rate: float = 15.0
    baseline_rate: float = 0.01
    boundary_source: BoundarySource = "physical"
    _distance_tolerance: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        tolerance = self.distance_tolerance
        if tolerance is None:
            tolerance = default_length(
                self.env,
                BORDER_DISTANCE_TOLERANCE_CM,
                name="distance_tolerance",
                call="BorderCellModel()",
            )
        _check_positive("distance_tolerance", tolerance)
        _check_positive("preferred_distance", self.preferred_distance, allow_zero=True)
        _check_rates(self.max_rate, self.baseline_rate)
        _require_walls(self.env, self.boundary_source, "BorderCellModel()")
        if self.distance_tolerance is not None:
            object.__setattr__(self, "distance_tolerance", float(tolerance))
        object.__setattr__(self, "_distance_tolerance", float(tolerance))
        object.__setattr__(self, "preferred_distance", float(self.preferred_distance))
        object.__setattr__(self, "max_rate", float(self.max_rate))
        object.__setattr__(self, "baseline_rate", float(self.baseline_rate))

    @property
    def requires_headings(self) -> bool:
        """False: the rate depends on position only."""
        return False

    def firing_rate(
        self,
        times: NDArray[np.float64],
        positions: NDArray[np.float64],
        *,
        headings: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Firing rate at each position.

        Parameters
        ----------
        times : NDArray[np.float64], shape (n_time,)
            Sample times in seconds; the rate does not depend on them.
        positions : NDArray[np.float64], shape (n_time, 2)
            Positions in the environment's coordinates.
        headings : NDArray[np.float64], shape (n_time,), optional
            Unused; accepted for the ``NeuralModel`` interface.

        Returns
        -------
        NDArray[np.float64], shape (n_time,)
            Rate in Hz; NaN for a missing position or one outside the walls,
            where the cell has no defined input.

        Raises
        ------
        ValueError
            If ``positions`` is not one row of 2 coordinates per time.
        """
        _times, positions = _model_inputs(
            times, positions, 2, call="BorderCellModel.firing_rate()"
        )
        distances, _directions = nearest_boundary(
            self.env, positions, boundary_source=self.boundary_source
        )
        tuning = np.exp(
            -0.5
            * ((distances - self.preferred_distance) / self._distance_tolerance) ** 2
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
            ``cell_type`` (``'border'``), ``preferred_distance``,
            ``distance_tolerance``, ``max_rate``, ``baseline_rate`` and
            ``boundary_source``.
        """
        return {
            "cell_type": "border",
            "preferred_distance": self.preferred_distance,
            "distance_tolerance": self._distance_tolerance,
            "max_rate": self.max_rate,
            "baseline_rate": self.baseline_rate,
            "boundary_source": self.boundary_source,
        }


@dataclass(frozen=True, eq=False)
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
    preferred_distance : float, optional
        Distance to the wall that drives the cell most, in environment units.
        If None, 17.5 cm expressed in ``env.units``: an illustrative value,
        the middle of the 5-30 cm range RatInABox's ``BoundaryVectorCells``
        draw tuning distances from.
    preferred_direction : float, optional
        Allocentric direction of that wall in radians (0 = East, +pi/2 =
        North). If None, drawn uniformly from [-pi, pi) with ``seed``, as
        every direction is equally likely.
    distance_tolerance : float, optional
        Standard deviation of the distance tuning, in environment units.
        Hartley et al. (2000) let it grow linearly with the preferred
        distance, so a far-tuned cell needs a wider tolerance. If None,
        ``preferred_distance / 12 + 8 cm``, the linear relationship
        (inverse slope 12, offset 8 cm) of RatInABox's default
        ``sigma_distance=(0.08, 12)`` for this model.
    direction_tolerance : float, default=pi/6
        Half-width at half-maximum of the angular tuning, in radians. The
        von Mises concentration is ``κ = ln 2 / (1 - cos(direction_tolerance))``.
        The 30° default is an illustrative value.
    max_rate : float, default=15.0
        Rate in Hz next to a straight wall at the preferred distance and
        direction. An illustrative value; it is not taken from a published
        measurement.
    baseline_rate : float, default=0.01
        Rate in Hz with no wall near the preferred distance and direction. An
        illustrative value.
    boundary_source : {'physical', 'analysis'}, default='physical'
        The arena's declared walls, or the edges of the environment's bins.
    seed : int or None, optional
        Seed for the random ``preferred_direction``.

    Attributes
    ----------
    ground_truth : dict[str, Any]
        The values the model fires with: the drawn direction, wrapped to
        [-pi, pi], and the default distance and tolerance. Every parameter is
        also an attribute holding the argument as passed, so
        ``preferred_distance``, ``preferred_direction`` and
        ``distance_tolerance`` read None when left to their defaults.
        ``dataclasses.replace(model, preferred_distance=40.0)`` builds a new
        model as the constructor would, so a default tolerance follows the
        new distance.

    Raises
    ------
    ValueError
        If a distance, angle or rate is not finite, a rate is negative,
        ``baseline_rate > max_rate``, or a length default is needed and
        the environment has no metric length units.
    CartesianOnlyError
        If the environment holds egocentric polar coordinates.
    NotImplementedError
        If the arena declares no physical boundary and
        ``boundary_source='physical'``.

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
    >>> positions = np.array([[10.0, 50.0], [90.0, 50.0]])
    >>> cell.firing_rate(np.array([0.0, 0.1]), positions).round(1)
    array([15.,  0.])
    """

    env: Environment = field(repr=False)
    _: KW_ONLY
    preferred_distance: float | None = None
    preferred_direction: float | None = None
    distance_tolerance: float | None = None
    direction_tolerance: float = np.pi / 6
    max_rate: float = 15.0
    baseline_rate: float = 0.01
    boundary_source: BoundarySource = "physical"
    seed: int | None = None
    _preferred_distance: float = field(init=False, repr=False)
    _preferred_direction: float = field(init=False, repr=False)
    _distance_tolerance: float = field(init=False, repr=False)
    _walls: Walls = field(init=False, repr=False)
    _directions: NDArray[np.float64] = field(init=False, repr=False)
    _angular: NDArray[np.float64] = field(init=False, repr=False)
    _reference_drive: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        env = self.env
        call = "BoundaryVectorCellModel()"
        distance = self.preferred_distance
        if distance is None:
            distance = default_length(
                env, BVC_PREFERRED_DISTANCE_CM, name="preferred_distance", call=call
            )
        tolerance = self.distance_tolerance
        if tolerance is None:
            tolerance = distance / 12.0 + default_length(
                env, 8.0, name="distance_tolerance", call=call
            )
        direction = self.preferred_direction
        if direction is None:
            direction = float(np.random.default_rng(self.seed).uniform(-np.pi, np.pi))
        _check_positive("preferred_distance", distance, allow_zero=True)
        _check_positive("distance_tolerance", tolerance)
        _check_angle("preferred_direction", direction)
        half_width = self.direction_tolerance
        if not (np.isfinite(half_width) and 0 < half_width <= np.pi):
            raise ValueError(
                _format_error(
                    "direction_tolerance must be a half-width in (0, pi] radians, "
                    f"got {half_width!r}.",
                    fix="pass the angle at which the tuning falls to half, e.g. "
                    "np.pi / 6 for 30 degrees",
                )
            )
        _check_rates(self.max_rate, self.baseline_rate)
        _require_walls(env, self.boundary_source, call)
        wrapped = float(np.angle(np.exp(1j * direction)))
        walls = Walls.from_geometry(
            boundary_geometry(
                env.binning, env.arena, self.boundary_source, blocking="movement"
            )
        )
        directions = boundary_directions(n_directions=_N_RAYS)
        # von Mises concentration with half-maximum at direction_tolerance.
        kappa = np.log(2.0) / (1.0 - np.cos(half_width))
        angular = np.exp(kappa * (np.cos(directions - wrapped) - 1.0))
        directions.setflags(write=False)
        angular.setflags(write=False)
        for name, argument in (
            ("preferred_distance", self.preferred_distance),
            ("preferred_direction", self.preferred_direction),
            ("distance_tolerance", self.distance_tolerance),
        ):
            if argument is not None:
                object.__setattr__(self, name, float(argument))
        for name, value in (
            ("_preferred_distance", float(distance)),
            ("_preferred_direction", wrapped),
            ("_distance_tolerance", float(tolerance)),
            ("direction_tolerance", float(half_width)),
            ("max_rate", float(self.max_rate)),
            ("baseline_rate", float(self.baseline_rate)),
            ("_walls", walls),
            ("_directions", directions),
            ("_angular", angular),
        ):
            object.__setattr__(self, name, value)
        # A straight wall at the preferred distance, square to the preferred
        # direction, lies at r = d / cos(θ - φ) along every ray toward it.
        facing = np.cos(directions - wrapped)
        with np.errstate(divide="ignore"):
            reference = np.where(facing > 0, float(distance) / facing, np.inf)
        object.__setattr__(
            self, "_reference_drive", float(np.sum(self._drive_terms(reference)))
        )

    def _drive_terms(self, distances: NDArray[np.float64]) -> NDArray[np.float64]:
        """Each ray's term ``g(r, θ)``, shape like ``distances`` (..., n_rays)."""
        radial = np.exp(
            -0.5
            * ((distances - self._preferred_distance) / self._distance_tolerance) ** 2
        )
        terms: NDArray[np.float64] = radial * self._angular
        return terms

    @property
    def requires_headings(self) -> bool:
        """False: the drive is summed over allocentric directions."""
        return False

    def firing_rate(
        self,
        times: NDArray[np.float64],
        positions: NDArray[np.float64],
        *,
        headings: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Firing rate at each position.

        Parameters
        ----------
        times : NDArray[np.float64], shape (n_time,)
            Sample times in seconds; the rate does not depend on them.
        positions : NDArray[np.float64], shape (n_time, 2)
            Positions in the environment's coordinates.
        headings : NDArray[np.float64], shape (n_time,), optional
            Unused; accepted for the ``NeuralModel`` interface.

        Returns
        -------
        NDArray[np.float64], shape (n_time,)
            Rate in Hz; NaN for a missing position or one outside the walls.

        Raises
        ------
        ValueError
            If ``positions`` is not one row of 2 coordinates per time.
        """
        _times, positions = _model_inputs(
            times, positions, 2, call="BoundaryVectorCellModel.firing_rate()"
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
            ``cell_type`` (``'boundary_vector'``), ``preferred_distance``,
            ``preferred_direction`` (radians), ``distance_tolerance``,
            ``direction_tolerance`` (radians), ``max_rate``,
            ``baseline_rate`` and ``boundary_source``.
        """
        return {
            "cell_type": "boundary_vector",
            "preferred_distance": self._preferred_distance,
            "preferred_direction": self._preferred_direction,
            "distance_tolerance": self._distance_tolerance,
            "direction_tolerance": self.direction_tolerance,
            "max_rate": self.max_rate,
            "baseline_rate": self.baseline_rate,
            "boundary_source": self.boundary_source,
        }
