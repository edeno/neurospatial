"""Object-vector cell models for simulating distance and direction tuned firing.

Object-vector cells (OVCs) fire when an animal is at a specific distance and
direction from an object in the environment. They were discovered in the medial
entorhinal cortex (MEC) and are thought to support object-location memory.

Key Features
------------
- **Distance tuning**: Gaussian tuning around preferred distance from objects
- **Direction tuning**: Optional von Mises (circular Gaussian) tuning around
  preferred direction in the configured frame
- **Object selectivity**: Respond to any object, nearest object, or specific object
- **Distance metrics**: Euclidean (straight-line) or geodesic (path through graph)

Coordinate Conventions
----------------------
**Allocentric direction (default)**: 0 = East and +pi/2 = North.
Set ``direction_frame="egocentric"`` for heading-relative tuning.

**Egocentric direction (when explicitly selected)**:
- 0 radians = object is directly ahead of animal
- π/2 radians = object is to the left
- -π/2 radians = object is to the right
- ±π radians = object is behind

This matches the coordinate convention in ``neurospatial.ops.egocentric``.

Examples
--------
Create a simple object-vector cell:

>>> import numpy as np
>>> from neurospatial import Environment
>>> from neurospatial.simulation.models.object_vector_cells import ObjectVectorCellModel
>>>
>>> # Create environment
>>> from shapely.geometry import box
>>> env = Environment.from_polygon(box(0, 0, 100, 100), bin_size=2.0)
>>>
>>> # Define object positions
>>> objects = np.array([[25.0, 50.0], [75.0, 50.0]])
>>>
>>> # Create OVC that fires 10 units from every object
>>> ovc = ObjectVectorCellModel(
...     env=env,
...     object_positions=objects,
...     preferred_distance=10.0,
...     distance_tolerance=5.0,
... )

Create a directionally-tuned allocentric OVC (fires when the object is East):

>>> ovc_directional = ObjectVectorCellModel(
...     env=env,
...     object_positions=objects,
...     preferred_distance=10.0,
...     distance_tolerance=5.0,
...     preferred_direction=0.0,  # Object East of the animal
...     direction_concentration=4.0,  # ~34° half-width
... )

References
----------
.. [1] Hoydal, O. A., et al. (2019). Object-vector coding in the medial
       entorhinal cortex. Nature, 568(7752), 400-404.
       https://doi.org/10.1038/s41586-019-1077-7

.. [2] Wang, C., et al. (2018). Egocentric coding of external items in the
       lateral entorhinal cortex. Science, 362, 945-949.
       https://doi.org/10.1126/science.aau4940

.. [3] Deshmukh, S. S., & Knierim, J. J. (2011). Representation of non-spatial
       and spatial information in the lateral entorhinal cortex. Frontiers in
       Behavioral Neuroscience, 5, 69.

See Also
--------
neurospatial.ops.egocentric : Coordinate transformations
PlaceCellModel : Gaussian place fields
BoundaryVectorCellModel : Wall-distance and direction tuned cells
"""

from __future__ import annotations

import warnings
from dataclasses import KW_ONLY, dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error
from neurospatial._units import default_length
from neurospatial.geometry import require_cartesian
from neurospatial.ops.distance import distance_field
from neurospatial.ops.egocentric import compute_egocentric_bearing
from neurospatial.simulation.models.base import (
    _check_angle,
    _check_metric,
    _check_positive,
    _check_rates,
    _model_headings,
    _model_inputs,
    _read_only,
    geodesic_lookup_bins,
)

if TYPE_CHECKING:
    from neurospatial.environment import Environment
    from neurospatial.ops._types import DistanceMetric

__all__ = ["ObjectVectorCellModel"]

#: Default preferred distance in centimeters: the mean object-to-field-center
#: distance of the object-vector cells of Høydal et al. (2019).
OVC_PREFERRED_DISTANCE_CM = 21.2

#: Default distance tuning width in centimeters, derived from the same
#: paper's field extents; see ``ObjectVectorCellModel`` Notes.
OVC_DISTANCE_TOLERANCE_CM = 6.9


@dataclass(frozen=True, eq=False)
class ObjectVectorCellModel:
    """Distance and direction tuned firing model for objects.

    Models object-vector cells (OVCs) that fire at specific distances and
    optionally specific directions from objects in the environment.

    Parameters
    ----------
    env : Environment
        The spatial environment.
    object_positions : NDArray[np.float64], shape (n_objects, 2)
        Positions of objects in allocentric coordinates.
    preferred_distance : float, optional
        Preferred distance from object in environment units. If None,
        21.2 cm expressed in ``env.units``: the mean distance between object
        and field centers of the object-vector cells of Høydal et al. (2019).
    distance_tolerance : float, optional
        Distance tuning width (standard deviation of Gaussian) in
        environment units. If None, 6.9 cm expressed in ``env.units`` (see
        Notes). An environment without metric length units needs explicit
        lengths.
    preferred_direction : float | None, optional
        Preferred animal-to-object direction in radians (default: None).
        If None, responds at the preferred distance in any direction.
        Otherwise applies von Mises tuning in ``direction_frame``.
    direction_frame : {"allocentric", "egocentric"}, default="allocentric"
        Allocentric: 0 = East, +pi/2 = North; no headings needed.
        Egocentric: 0 = ahead, +pi/2 = left; directional tuning requires
        headings in ``firing_rate()``. The reverse object-to-animal vector
        has preferred direction plus pi, wrapped to [-pi, pi].
    direction_concentration : float, optional
        Direction tuning concentration parameter (default: 4.0, a 34°
        half-width at half-maximum, ``arccos(1 - ln(2)/κ)``). Higher values =
        sharper tuning. An illustrative value; it is not taken from a
        published measurement.
    max_rate : float, optional
        Peak firing rate in Hz (default: 20.0). An illustrative value; it is
        not taken from a published measurement.
    baseline_rate : float, optional
        Firing rate outside the tuned region, in Hz (default: 0.01). An
        illustrative value for a cell nearly silent away from its fields.
    object_selectivity : {'any', 'nearest', 'specific'}, optional
        How to aggregate responses across objects (default: 'any', a field
        at the preferred vector from every object). Høydal et al. (2019)
        tested object-vector cells with 2-6 objects at once and found similar
        vector fields for each object.

        - 'any': Maximum response across all objects
        - 'nearest': Response to nearest object only
        - 'specific': Response to specific object (requires ``specific_object_index``)
    specific_object_index : int | None, optional
        Index of object to respond to when ``object_selectivity='specific'``.
    metric : {'euclidean', 'geodesic'}, optional
        Distance calculation method (default: 'euclidean').

        - 'euclidean': Straight-line distance
        - 'geodesic': Path distance through environment connectivity

    Attributes
    ----------
    env : Environment
        The spatial environment.
    object_positions : NDArray[np.float64]
        Object positions (n_objects, 2).
    preferred_distance : float
        Preferred distance from object.
    distance_tolerance : float
        Distance tuning width.
    preferred_direction : float | None
        Preferred direction in the configured frame (radians) or None.
    direction_concentration : float
        Direction tuning concentration.
    max_rate : float
        Peak firing rate in Hz.
    baseline_rate : float
        Baseline firing rate in Hz.
    object_selectivity : str
        Object selectivity mode.
    specific_object_index : int | None
        The object a 'specific' cell responds to.
    metric : str
        Distance metric used.

    Raises
    ------
    ValueError
        If ``object_positions`` is not one finite ``(x, y)`` row per object;
        a distance, the direction, the concentration or a rate is not finite
        or is negative, or ``baseline_rate`` exceeds ``max_rate``;
        ``direction_frame``, ``object_selectivity`` or ``metric`` is unknown;
        ``object_selectivity='specific'`` lacks a valid
        ``specific_object_index``; an object lies outside the environment
        with ``metric='geodesic'``; or a length default is needed and
        ``env.units`` is not a metric length unit.
    CartesianOnlyError
        If the environment holds egocentric polar coordinates.

    Warns
    -----
    UserWarning
        If an object lies outside the environment with
        ``metric='euclidean'``.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.simulation.models.object_vector_cells import (
    ...     ObjectVectorCellModel,
    ... )
    >>>
    >>> # Create environment
    >>> from shapely.geometry import box
    >>> env = Environment.from_polygon(box(0, 0, 100, 100), bin_size=2.0)
    >>>
    >>> # Create OVC
    >>> ovc = ObjectVectorCellModel(
    ...     env=env,
    ...     object_positions=np.array([[50.0, 50.0]]),
    ...     preferred_distance=10.0,
    ...     distance_tolerance=5.0,
    ... )
    >>>
    >>> # Compute firing rates along a trajectory
    >>> positions = np.random.default_rng(42).uniform(0, 100, (100, 2))
    >>> times = np.arange(100) / 30.0  # s
    >>> rates = ovc.firing_rate(times, positions)
    >>> rates.shape
    (100,)

    Notes
    -----
    **Default distances**: Høydal et al. (2019) report, for mouse
    object-vector cells, a mean distance of 21.2 cm from the object to the
    field center and of 8.9 cm from the object to the field's contour, so a
    field extends about 12.3 cm from its center toward the object. Reading
    that contour as 20% of the peak rate (their border-field threshold; their
    object-vector contour was set by the rate map's variability instead) and
    the field as a Gaussian, ``σ = 12.3 / √(2 ln 5) = 6.9 cm``.

    **Distance Tuning**: Firing rate follows Gaussian around preferred_distance:

    .. math::

        r_d(d) = \\exp\\left(-\\frac{(d - d_{pref})^2}{2 \\sigma_d^2}\\right)

    **Direction Tuning**: When ``preferred_direction`` is specified, applies
    von Mises (circular Gaussian) tuning:

    .. math::

        r_\\theta(\\theta) = \\frac{\\exp(\\kappa \\cos(\\theta - \\theta_{pref}))}{\\exp(\\kappa)}

    The normalization ensures peak response = 1 when θ = θ_pref.

    **Combined Response**:

    .. math::

        r(x) = r_{base} + (r_{max} - r_{base}) \\cdot r_d(d) \\cdot r_\\theta(\\theta)

    See Also
    --------
    PlaceCellModel : Gaussian place fields
    BoundaryVectorCellModel : Wall-distance and direction tuned cells
    compute_egocentric_bearing : Compute bearing to targets
    """

    env: Environment = field(repr=False)
    _: KW_ONLY
    object_positions: NDArray[np.float64]
    preferred_distance: float | None = None
    distance_tolerance: float | None = None
    preferred_direction: float | None = None
    direction_frame: Literal["allocentric", "egocentric"] = "allocentric"
    direction_concentration: float = 4.0
    max_rate: float = 20.0
    baseline_rate: float = 0.01
    object_selectivity: Literal["any", "nearest", "specific"] = "any"
    specific_object_index: int | None = None
    metric: DistanceMetric = "euclidean"
    _distance_fields: NDArray[np.float64] | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        env = self.env
        call = "ObjectVectorCellModel()"
        require_cartesian(
            env.coordinate_system,
            call,
            why=(
                "It models firing from the distance and direction to objects, "
                "from the animal's allocentric Cartesian position."
            ),
            fix="simulate on the allocentric environment the animal moves in",
        )
        if self.direction_frame not in ("allocentric", "egocentric"):
            raise ValueError(
                _format_error(
                    f"Unknown direction_frame={self.direction_frame!r}.",
                    why="Why: direction tuning needs an allocentric or egocentric reference frame",
                    fix='pass direction_frame="allocentric" or direction_frame="egocentric"',
                )
            )
        objects = _read_only(self.object_positions)
        if objects.ndim != 2 or objects.shape[1] != 2 or len(objects) == 0:
            raise ValueError(
                _format_error(
                    "object_positions must have shape (n_objects, 2), got shape "
                    f"{objects.shape}.",
                    fix="pass one (x, y) row per object, e.g. "
                    "np.array([[25.0, 50.0]]) for one object",
                )
            )
        if not np.all(np.isfinite(objects)):
            missing = np.flatnonzero(~np.isfinite(objects).all(axis=1))
            raise ValueError(
                _format_error(
                    f"object_positions has non-finite coordinates in row(s) "
                    f"{missing.tolist()}.",
                    why="an object-vector cell fires at a distance from each "
                    "object, so every object needs a position.",
                    fix="pass finite (x, y) coordinates for every object, or "
                    "leave out the objects whose position is unknown",
                )
            )

        distance = self.preferred_distance
        if distance is None:
            distance = default_length(
                env, OVC_PREFERRED_DISTANCE_CM, name="preferred_distance", call=call
            )
        tolerance = self.distance_tolerance
        if tolerance is None:
            tolerance = default_length(
                env, OVC_DISTANCE_TOLERANCE_CM, name="distance_tolerance", call=call
            )
        _check_positive("preferred_distance", distance, allow_zero=True)
        _check_positive("distance_tolerance", tolerance)
        if self.preferred_direction is not None:
            _check_angle("preferred_direction", self.preferred_direction)
        _check_positive(
            "direction_concentration", self.direction_concentration, units=None
        )
        _check_rates(self.max_rate, self.baseline_rate)
        _check_metric(self.metric, call)

        valid_selectivities = ("any", "nearest", "specific")
        if self.object_selectivity not in valid_selectivities:
            raise ValueError(
                _format_error(
                    f"object_selectivity must be one of {valid_selectivities}, "
                    f"got {self.object_selectivity!r}.",
                    fix="pass 'any' for a field at every object, 'nearest' for "
                    "the closest object, or 'specific' with specific_object_index",
                )
            )
        index = self.specific_object_index
        if self.object_selectivity == "specific" and (
            index is None or not (0 <= index < len(objects))
        ):
            raise ValueError(
                _format_error(
                    f"object_selectivity='specific' needs specific_object_index "
                    f"in [0, {len(objects)}), got {index!r}.",
                    fix="pass the row of object_positions the cell responds to",
                )
            )

        outside = ~env.contains(objects)
        if outside.any() and self.metric == "geodesic":
            raise ValueError(
                _format_error(
                    f"{call} got {int(outside.sum())} object(s) outside the "
                    f"environment with metric='geodesic': "
                    f"{objects[outside].tolist()}.",
                    why="a geodesic distance is a path through the "
                    "environment's bins, and an object outside them has no "
                    "path to the animal.",
                    fix="check the objects' coordinates and units, or use "
                    "metric='euclidean' for an object beyond the arena",
                )
            )
        if outside.any():
            warnings.warn(
                f"{int(outside.sum())} object(s) lie outside the environment: "
                f"{objects[outside].tolist()}. Euclidean distances are measured "
                "to the objects themselves, so their fields may fall partly or "
                "wholly outside the arena. If the objects should be inside, "
                "check their coordinates and units.",
                UserWarning,
                stacklevel=3,
            )

        fields: NDArray[np.float64] | None = None
        if self.metric == "geodesic":
            object_bins = geodesic_lookup_bins(env, objects)
            fields = np.stack(
                [
                    distance_field(env.connectivity, sources=[int(b)])
                    for b in object_bins
                ],
                axis=1,
            )
            fields.setflags(write=False)
        for name, value in (
            ("object_positions", objects),
            ("preferred_distance", float(distance)),
            ("distance_tolerance", float(tolerance)),
            (
                "preferred_direction",
                None
                if self.preferred_direction is None
                else float(self.preferred_direction),
            ),
            ("direction_concentration", float(self.direction_concentration)),
            ("max_rate", float(self.max_rate)),
            ("baseline_rate", float(self.baseline_rate)),
            ("_distance_fields", fields),
        ):
            object.__setattr__(self, name, value)

    @property
    def requires_headings(self) -> bool:
        """True for egocentric direction tuning, which is relative to the heading."""
        return (
            self.direction_frame == "egocentric"
            and self.preferred_direction is not None
        )

    def firing_rate(
        self,
        times: NDArray[np.float64],
        positions: NDArray[np.float64],
        *,
        headings: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Compute object-vector cell firing rate.

        Parameters
        ----------
        times : NDArray[np.float64], shape (n_time,)
            Sample times in seconds; the rate does not depend on them.
        positions : NDArray[np.float64], shape (n_time, 2)
            Animal positions in allocentric coordinates.
        headings : NDArray[np.float64], shape (n_time,), optional
            Animal heading in radians (allocentric convention: 0=East).
            Required only when ``requires_headings`` is True
            (``direction_frame="egocentric"`` with ``preferred_direction``
            set). Other models ignore headings.

        Returns
        -------
        firing_rate : NDArray[np.float64], shape (n_time,)
            Firing rate in Hz at each position; NaN for a missing position
            or heading.

        Raises
        ------
        ValueError
            If ``positions`` is not one row of 2 coordinates per time, or
            egocentric directional tuning needs headings and none (or not one
            per time) are supplied.

        Notes
        -----
        Implementation:

        1. Compute distances from positions to all objects
        2. Apply Gaussian distance tuning
        3. If directional: compute bearing to objects and apply von Mises tuning
        4. Aggregate by object selectivity mode
        5. Scale by max_rate and add baseline
        """
        call = "ObjectVectorCellModel.firing_rate()"
        times, positions = _model_inputs(times, positions, 2, call=call)
        if self.requires_headings:
            headings = _model_headings(
                headings,
                len(times),
                call=call,
                alternative='; or configure direction_frame="allocentric" for '
                "tuning to compass directions",
            )
        assert self.preferred_distance is not None
        assert self.distance_tolerance is not None

        # Distances from every position to every object, (n_time, n_objects).
        if self._distance_fields is None:
            distances = np.linalg.norm(
                positions[:, np.newaxis, :] - self.object_positions[np.newaxis, :, :],
                axis=2,
            )
        else:
            bin_indices = geodesic_lookup_bins(self.env, positions)
            # A missing position has no distance, so its rate stays NaN.
            distances = np.full((len(times), len(self.object_positions)), np.nan)
            observed = bin_indices >= 0
            distances[observed] = self._distance_fields[bin_indices[observed]]

        distance_response = np.exp(
            -0.5
            * ((distances - self.preferred_distance) / self.distance_tolerance) ** 2
        )

        if self.preferred_direction is not None:
            # Animal-to-object direction of each object, (n_time, n_objects).
            if self.direction_frame == "allocentric":
                delta = self.object_positions[None, :, :] - positions[:, None, :]
                bearings = np.arctan2(delta[:, :, 1], delta[:, :, 0])
            else:
                assert headings is not None
                bearings = compute_egocentric_bearing(
                    positions, headings, self.object_positions
                )
            # von Mises tuning with peak 1 at θ_pref, exp(κ (cos(θ - θ_pref) - 1)),
            # which equals exp(κ cos(θ - θ_pref)) / exp(κ) without overflowing
            # exp(κ) for a sharp cell.
            direction_response = np.exp(
                self.direction_concentration
                * (np.cos(bearings - self.preferred_direction) - 1.0)
            )
            combined_response = distance_response * direction_response
        else:
            combined_response = distance_response

        if self.object_selectivity == "any":
            response = np.max(combined_response, axis=1)
        elif self.object_selectivity == "nearest":
            nearest_obj = np.argmin(distances, axis=1)
            response = combined_response[np.arange(len(times)), nearest_obj]
        else:
            assert self.specific_object_index is not None
            response = combined_response[:, self.specific_object_index]

        rates: NDArray[np.float64] = (
            self.baseline_rate + (self.max_rate - self.baseline_rate) * response
        )
        return rates

    @property
    def ground_truth(self) -> dict[str, Any]:
        """Return ground truth parameters for validation.

        Returns
        -------
        parameters : dict[str, Any]
            Dictionary with keys:

            - 'cell_type': 'object_vector'
            - 'object_positions': NDArray - positions of objects (read-only)
            - 'preferred_distance': float - preferred distance from object
            - 'distance_tolerance': float - distance tuning width
            - 'preferred_direction': float | None - preferred direction (radians)
            - 'direction_frame': str - allocentric or egocentric direction
            - 'direction_concentration': float - direction tuning concentration
            - 'max_rate': float - peak firing rate in Hz
            - 'baseline_rate': float - baseline firing rate in Hz
            - 'object_selectivity': str - object selectivity mode
            - 'specific_object_index': int | None - the object of a
              'specific' cell
            - 'metric': str - distance metric

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.simulation.models.object_vector_cells import (
        ...     ObjectVectorCellModel,
        ... )
        >>> from shapely.geometry import box
        >>> env = Environment.from_polygon(box(0, 0, 100, 100), bin_size=2.0)
        >>> ovc = ObjectVectorCellModel(
        ...     env=env,
        ...     object_positions=np.array([[50.0, 50.0]]),
        ...     preferred_distance=10.0,
        ...     distance_tolerance=5.0,
        ... )
        >>> ovc.ground_truth["preferred_distance"]
        10.0
        """
        return {
            "cell_type": "object_vector",
            "object_positions": self.object_positions,
            "preferred_distance": self.preferred_distance,
            "distance_tolerance": self.distance_tolerance,
            "preferred_direction": self.preferred_direction,
            "direction_frame": self.direction_frame,
            "direction_concentration": self.direction_concentration,
            "max_rate": self.max_rate,
            "baseline_rate": self.baseline_rate,
            "object_selectivity": self.object_selectivity,
            "specific_object_index": self.specific_object_index,
            "metric": self.metric,
        }
