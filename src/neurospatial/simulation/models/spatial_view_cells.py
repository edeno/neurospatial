"""Spatial view cell models for simulating gaze-based firing.

Spatial view cells (SVCs) fire when an animal is looking at a specific location
in allocentric space, regardless of the animal's own position. They were
discovered in primate hippocampus and are thought to support spatial navigation
and memory by encoding "what is being viewed".

Key Features
------------
- **View-centered tuning**: Gaussian tuning around preferred viewed location
- **Gaze models**: the first wall along the gaze (ray casting), or a point a
  fixed distance ahead that stops at a nearer wall; the same gaze the view
  analysis (``compute_view_rate``) uses
- **Field of view**: Optional FOV constraints matching species-specific vision
- **Visibility**: Optional line-of-sight requirements

Coordinate Conventions
----------------------
**Heading** (allocentric, for gaze direction):
- 0 radians = facing East
- π/2 radians = facing North
- ±π radians = facing West
- -π/2 radians = facing South

This matches the convention in ``neurospatial.ops.egocentric``.

Examples
--------
Create a simple spatial view cell:

>>> import numpy as np
>>> from neurospatial import Environment
>>> from neurospatial.simulation.models.spatial_view_cells import SpatialViewCellModel
>>>
>>> from shapely.geometry import box
>>>
>>> # A walled 100 x 100 cm box
>>> env = Environment.from_polygon(box(0, 0, 100, 100), bin_size=2.0, units="cm")
>>>
>>> # Create SVC that fires when the gaze reaches the east wall at (100, 50)
>>> svc = SpatialViewCellModel(
...     env=env,
...     preferred_view_location=np.array([100.0, 50.0]),
...     view_field_width=10.0,  # Gaussian tuning width
... )
>>> svc.firing_rate(np.array([[50.0, 50.0]]), headings=np.array([0.0])).round(1)
array([20.])

Create a spatial view cell with visibility requirements:

>>> from neurospatial.ops.visibility import FieldOfView
>>> svc_visible = SpatialViewCellModel(
...     env=env,
...     preferred_view_location=np.array([100.0, 50.0]),
...     view_field_width=10.0,
...     require_visibility=True,
...     fov=FieldOfView.primate(),  # ~180 degree forward FOV
... )

References
----------
.. [1] Rolls, E. T., Robertson, R. G., & Georges-François, P. (1997).
       Spatial view cells in the primate hippocampus. European Journal of
       Neuroscience, 9(8), 1789-1794.

.. [2] Georges-François, P., Rolls, E. T., & Robertson, R. G. (1999).
       Spatial view cells in the primate hippocampus: allocentric view not
       head direction or eye position or place. Cerebral Cortex, 9(3), 197-212.

.. [3] Robertson, R. G., Rolls, E. T., & Georges-François, P. (1998). Spatial
       view cells in the primate hippocampus: effects of removal of view
       details. Journal of Neurophysiology, 79(3), 1145-1156.

See Also
--------
neurospatial.ops.visibility : Visibility and gaze computation
neurospatial.ops.egocentric : Coordinate transformations
PlaceCellModel : Position-based place fields
ObjectVectorCellModel : Object-distance tuned cells
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from neurospatial.geometry import (
    CallIndexes,
    line_of_sight_clear,
    require_cartesian,
)
from neurospatial.ops.visibility import FieldOfView, gaze_points
from neurospatial.simulation.models.base import default_length

if TYPE_CHECKING:
    from neurospatial.environment import Environment
    from neurospatial.geometry._boundaries import BoundarySource

__all__ = ["SpatialViewCellModel"]


class SpatialViewCellModel:
    """Gaze-based firing model for spatial view cells.

    Models spatial view cells (SVCs) that fire when the animal is looking at
    a specific location in allocentric space, regardless of its own position.

    Parameters
    ----------
    env : Environment
        The spatial environment.
    preferred_view_location : NDArray[np.float64], shape (2,)
        Location in allocentric coordinates that triggers peak firing when viewed.
    view_field_width : float, optional
        Standard deviation of the Gaussian tuning around the preferred
        location, in environment units. Larger values produce broader spatial
        tuning. If None, 10 cm expressed in ``env.units``: an illustrative
        value, not taken from a published measurement. An environment without
        metric length units needs an explicit ``view_field_width``.
    view_distance : float or None, optional
        Gaze distance for ``gaze_model='fixed_distance'``, in environment
        units. None (the default) uses 10 cm, as ``compute_view_rate`` does,
        and needs ``env.units`` to be a length unit. Must be None with
        ``ray_cast``.
    gaze_model : {'ray_cast', 'fixed_distance'}, optional
        How the viewed location is found (default: 'ray_cast', as in
        ``compute_view_rate``):

        - 'ray_cast': the first wall along the gaze.
        - 'fixed_distance': the point ``view_distance`` ahead, or the first
          wall when one is nearer.

        Both stop at the environment's walls, so the gaze never passes
        through one; :func:`neurospatial.ops.visibility.compute_viewed_location`
        computes the same points.
    boundary_source : {'physical', 'analysis'}, default='physical'
        Declared physical walls or explicitly selected analysis boundaries for
        the gaze and line of sight. Unknown physical walls refuse.
    max_rate : float, optional
        Peak firing rate in Hz (default: 20.0). An illustrative value; it is
        not taken from a published measurement.
    baseline_rate : float, optional
        Firing rate when the preferred location is not viewed, in Hz
        (default: 0.01). An illustrative value.
    require_visibility : bool, optional
        If True, returns baseline rate when view is blocked (default: False).
    fov : FieldOfView | None, optional
        Field of view constraint. If None, uses full 360 degrees (default: None).

    Attributes
    ----------
    env : Environment
        The spatial environment.
    preferred_view_location : NDArray[np.float64]
        Preferred viewed location (2,).
    view_field_width : float
        Gaussian tuning width.
    view_distance : float or None
        Gaze distance of the fixed_distance model in environment units; None
        for ray casting.
    gaze_model : str
        Gaze computation model.
    max_rate : float
        Peak firing rate in Hz.
    baseline_rate : float
        Baseline firing rate in Hz.
    require_visibility : bool
        Whether visibility check is required.
    fov : FieldOfView | None
        Field of view constraint.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.simulation.models.spatial_view_cells import (
    ...     SpatialViewCellModel,
    ... )
    >>>
    >>> from shapely.geometry import box
    >>>
    >>> env = Environment.from_polygon(box(0, 0, 100, 100), bin_size=2.0, units="cm")
    >>>
    >>> # A cell for the middle of the north wall
    >>> svc = SpatialViewCellModel(
    ...     env=env,
    ...     preferred_view_location=np.array([50.0, 100.0]),
    ... )
    >>>
    >>> # Compute firing rates along a trajectory
    >>> positions = np.random.default_rng(42).uniform(0, 100, (100, 2))
    >>> headings = np.random.default_rng(42).uniform(-np.pi, np.pi, 100)
    >>> rates = svc.firing_rate(positions, headings=headings)
    >>> rates.shape
    (100,)

    Notes
    -----
    **Spatial View Tuning**: Firing rate follows Gaussian around preferred
    viewed location:

    .. math::

        r(v) = r_{base} + (r_{max} - r_{base}) \\cdot
               \\exp\\left(-\\frac{||v - v_{pref}||^2}{2 \\sigma^2}\\right)

    where :math:`v` is the viewed location computed from position and heading,
    :math:`v_{pref}` is the preferred view location, and :math:`\\sigma` is the
    view field width.

    **Gaze Models**: the viewed location comes from
    :func:`neurospatial.ops.visibility.compute_viewed_location` with this
    model's environment, so it is the location the view analysis assigns the
    sample to:

    - ``ray_cast``: the first wall the gaze reaches.
    - ``fixed_distance``: the point ``view_distance`` ahead, or the first
      wall when one is nearer.

    Where the observer is outside the walls the viewed location is undefined
    and the cell fires at its baseline rate.

    See Also
    --------
    PlaceCellModel : Position-based place fields
    ObjectVectorCellModel : Object-distance tuned cells
    compute_viewed_location : Gaze computation function
    """

    def __init__(
        self,
        env: Environment,
        *,
        preferred_view_location: NDArray[np.float64],
        view_field_width: float | None = None,
        view_distance: float | None = None,
        gaze_model: Literal["ray_cast", "fixed_distance"] = "ray_cast",
        max_rate: float = 20.0,
        baseline_rate: float = 0.01,
        require_visibility: bool = False,
        fov: FieldOfView | None = None,
        boundary_source: BoundarySource = "physical",
    ) -> None:
        require_cartesian(
            env.coordinate_system,
            "SpatialViewCellModel()",
            why=(
                "It models firing from the allocentric Cartesian location the "
                "animal looks at."
            ),
            fix="simulate on the allocentric environment the animal moves in",
        )
        # Convert and validate preferred_view_location
        preferred_view_location = np.asarray(preferred_view_location, dtype=np.float64)
        if preferred_view_location.shape != (2,):
            msg = (
                f"preferred_view_location must have shape (2,), "
                f"got shape {preferred_view_location.shape}"
            )
            raise ValueError(msg)

        if view_field_width is None:
            view_field_width = default_length(
                env, 10.0, name="view_field_width", call="SpatialViewCellModel()"
            )
        # Validate view_field_width
        if view_field_width <= 0:
            msg = f"view_field_width must be positive, got {view_field_width}"
            raise ValueError(msg)

        # Validate max_rate
        if max_rate <= 0:
            msg = f"max_rate must be positive, got {max_rate}"
            raise ValueError(msg)

        # Validate baseline_rate
        if baseline_rate < 0:
            msg = f"baseline_rate must be non-negative, got {baseline_rate}"
            raise ValueError(msg)

        if baseline_rate >= max_rate:
            msg = (
                f"baseline_rate ({baseline_rate}) must be less than "
                f"max_rate ({max_rate})"
            )
            raise ValueError(msg)

        # The view analysis's own gaze models, gaze-distance rule and default.
        from neurospatial.encoding.view import (
            _resolve_view_distance,
            _validate_gaze_model,
        )

        _validate_gaze_model(gaze_model)
        view_distance = _resolve_view_distance(
            env, gaze_model, view_distance, context="SpatialViewCellModel()"
        )

        # Check if preferred_view_location is outside environment bounds
        if not env.contains(preferred_view_location[None, :])[0]:
            warnings.warn(
                f"preferred_view_location {preferred_view_location} is outside "
                f"the environment bounds. Model may not behave as expected.",
                UserWarning,
                stacklevel=2,
            )

        # Store parameters
        self.env = env
        self.preferred_view_location = preferred_view_location
        self.view_field_width = view_field_width
        self.view_distance = view_distance
        self.gaze_model = gaze_model
        if boundary_source not in ("physical", "analysis"):
            raise ValueError("boundary_source must be 'physical' or 'analysis'.")
        self.boundary_source = boundary_source
        self.max_rate = max_rate
        self.baseline_rate = baseline_rate
        self.require_visibility = require_visibility
        self.fov = fov

    def firing_rate(
        self,
        positions: NDArray[np.float64],
        times: NDArray[np.float64] | None = None,
        headings: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Compute spatial view cell firing rate.

        Parameters
        ----------
        positions : NDArray[np.float64], shape (n_time, 2)
            Animal positions in allocentric coordinates.
        times : NDArray[np.float64], shape (n_time,), optional
            Time points in seconds (not used, for API compatibility).
        headings : NDArray[np.float64], shape (n_time,)
            Animal heading in radians (allocentric convention: 0=East).
            **Required for all SpatialViewCellModel computations.**

        Returns
        -------
        firing_rate : NDArray[np.float64], shape (n_time,)
            Firing rate in Hz at each position/heading.

        Raises
        ------
        ValueError
            If ``headings`` is not provided.

        Notes
        -----
        Implementation:

        1. Compute viewed location from position and heading
        2. Compute distance from viewed location to preferred
        3. Apply Gaussian tuning
        4. Optionally check visibility
        5. Scale by max_rate and add baseline
        """
        positions = np.asarray(positions, dtype=np.float64)
        n_time = len(positions)

        # Validate headings
        if headings is None:
            msg = "headings is required for SpatialViewCellModel"
            raise ValueError(msg)

        headings = np.asarray(headings, dtype=np.float64)

        # The gaze the view analysis uses: it stops at the first wall.
        viewed_locations, _at_wall = gaze_points(
            positions,
            headings,
            method=self.gaze_model,
            view_distance=self.view_distance,
            gaze_offsets=None,
            env=self.env,
            max_distance=None,
            boundary_source=self.boundary_source,
        )
        if self.require_visibility:
            # One binning copy and one store of its lookup structures for
            # every timestep's line of sight.
            binning, arena = self.env.binning, self.env.arena
            indexes = CallIndexes()

        # Compute distance from viewed location to preferred
        # Shape: (n_time,)
        distances = np.linalg.norm(
            viewed_locations - self.preferred_view_location, axis=1
        )

        # Apply Gaussian tuning
        response = np.exp(-0.5 * (distances / self.view_field_width) ** 2)

        # Handle NaN viewed locations (viewing outside environment)
        nan_mask = np.isnan(distances)
        response[nan_mask] = 0.0

        if self.require_visibility:
            # Visibility path: a timestep fires only if the *preferred* location
            # is within the FOV (when an FOV is set) and there is a clear line of
            # sight to it. Compute the FOV mask for all timesteps at once, then
            # run the inherently scalar line-of-sight check only on the surviving
            # candidates.
            from neurospatial.ops.egocentric import compute_egocentric_bearing

            if self.fov is not None:
                # Bearing to the fixed preferred location at every timestep.
                bearings = compute_egocentric_bearing(
                    positions,
                    headings,
                    self.preferred_view_location[None, :],
                )[:, 0]
                in_fov = np.asarray(self.fov.contains_angle(bearings))
                response[~in_fov] = 0.0
            else:
                in_fov = np.ones(n_time, dtype=bool)

            # Line-of-sight check (scalar per call) for the remaining candidates:
            # not NaN, in FOV, and still responding above zero.
            candidates = np.flatnonzero(~nan_mask & in_fov & (response > 0.0))
            for i in candidates:
                if not line_of_sight_clear(
                    binning,
                    arena,
                    positions[i],
                    self.preferred_view_location,
                    boundary_source=self.boundary_source,
                    indexes=indexes,
                ):
                    response[i] = 0.0

        elif self.fov is not None:
            # FOV-only path (no visibility requirement): a timestep fires only if
            # the *viewed* location is within the FOV. The viewed location varies
            # per timestep, so pass per-time targets shaped (n_time, 1, 2).
            from neurospatial.ops.egocentric import compute_egocentric_bearing

            bearings = compute_egocentric_bearing(
                positions,
                headings,
                viewed_locations[:, None, :],
            )[:, 0]
            in_fov = np.asarray(self.fov.contains_angle(bearings))
            # NaN viewed locations already have response 0; restrict only the
            # valid ones that fall outside the FOV.
            response[~nan_mask & ~in_fov] = 0.0

        # Scale by max_rate and add baseline
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

            - 'preferred_view_location': NDArray - preferred viewed location
            - 'view_field_width': float - Gaussian tuning width
            - 'view_distance': float or None - gaze distance of the
              fixed_distance model, None for ray casting
            - 'gaze_model': str - gaze computation model
            - 'max_rate': float - peak firing rate in Hz
            - 'baseline_rate': float - baseline firing rate in Hz
            - 'require_visibility': bool - visibility requirement
            - 'fov': FieldOfView | None - field of view constraint

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.simulation.models.spatial_view_cells import (
        ...     SpatialViewCellModel,
        ... )
        >>> from shapely.geometry import box
        >>> env = Environment.from_polygon(box(0, 0, 100, 100), bin_size=2.0)
        >>> svc = SpatialViewCellModel(
        ...     env=env,
        ...     preferred_view_location=np.array([50.0, 100.0]),
        ...     view_field_width=15.0,
        ... )
        >>> svc.ground_truth["view_field_width"]
        15.0
        """
        return {
            "preferred_view_location": self.preferred_view_location.copy(),
            "view_field_width": self.view_field_width,
            "view_distance": self.view_distance,
            "gaze_model": self.gaze_model,
            "boundary_source": self.boundary_source,
            "max_rate": self.max_rate,
            "baseline_rate": self.baseline_rate,
            "require_visibility": self.require_visibility,
            "fov": self.fov,
        }
