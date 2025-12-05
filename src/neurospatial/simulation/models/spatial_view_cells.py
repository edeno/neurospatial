"""Spatial view cell models for simulating gaze-based firing.

Spatial view cells (SVCs) fire when an animal is looking at a specific location
in allocentric space, regardless of the animal's own position. They were
discovered in primate hippocampus and are thought to support spatial navigation
and memory by encoding "what is being viewed".

Key Features
------------
- **View-centered tuning**: Gaussian tuning around preferred viewed location
- **Gaze models**: Fixed distance, ray casting, or boundary-based viewing
- **Field of view**: Optional FOV constraints matching species-specific vision
- **Visibility**: Optional line-of-sight requirements

Coordinate Conventions
----------------------
**Heading** (allocentric, for gaze direction):
- 0 radians = facing East
- π/2 radians = facing North
- ±π radians = facing West
- -π/2 radians = facing South

This matches the convention in ``neurospatial.reference_frames``.

Examples
--------
Create a simple spatial view cell:

>>> import numpy as np
>>> from neurospatial import Environment
>>> from neurospatial.simulation.models.spatial_view_cells import SpatialViewCellModel
>>>
>>> # Create environment
>>> samples = np.random.default_rng(42).uniform(0, 100, (500, 2))
>>> env = Environment.from_samples(samples, bin_size=2.0)
>>>
>>> # Create SVC that fires when looking at (50, 50)
>>> svc = SpatialViewCellModel(
...     env=env,
...     preferred_view_location=np.array([50.0, 50.0]),
...     view_field_width=10.0,  # Gaussian tuning width
...     view_distance=20.0,  # Fixed viewing distance
... )

Create a spatial view cell with visibility requirements:

>>> from neurospatial.visibility import FieldOfView
>>> svc_visible = SpatialViewCellModel(
...     env=env,
...     preferred_view_location=np.array([50.0, 50.0]),
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
neurospatial.visibility : Visibility and gaze computation
neurospatial.reference_frames : Coordinate transformations
PlaceCellModel : Position-based place fields
ObjectVectorCellModel : Object-distance tuned cells
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from neurospatial.visibility import (
    FieldOfView,
    _line_of_sight_clear,
    compute_viewed_location,
)

if TYPE_CHECKING:
    from neurospatial import Environment

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
        Width of Gaussian tuning around preferred location (default: 10.0).
        Larger values produce broader spatial tuning.
    view_distance : float, optional
        Distance at which gaze is computed for fixed_distance model (default: 20.0).
        Only used when ``gaze_model='fixed_distance'``.
    gaze_model : {'fixed_distance', 'ray_cast', 'boundary'}, optional
        Model for computing viewed location (default: 'fixed_distance').

        - 'fixed_distance': Point at fixed distance in gaze direction
        - 'ray_cast': Intersection with environment boundary
        - 'boundary': Nearest boundary point in gaze direction
    max_rate : float, optional
        Peak firing rate in Hz (default: 20.0).
    baseline_rate : float, optional
        Baseline firing rate when not viewing preferred location (default: 0.001 Hz).
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
    view_distance : float
        View distance for fixed_distance model.
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
    >>> # Create environment
    >>> samples = np.random.default_rng(42).uniform(0, 100, (500, 2))
    >>> env = Environment.from_samples(samples, bin_size=2.0)
    >>>
    >>> # Create SVC
    >>> svc = SpatialViewCellModel(
    ...     env=env,
    ...     preferred_view_location=np.array([50.0, 50.0]),
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

    **Gaze Models**:

    - ``fixed_distance``: Viewed location is at a fixed distance from the
      animal in the gaze direction. Simplest model.
    - ``ray_cast``: Ray is cast from animal position in gaze direction until
      it exits the environment.
    - ``boundary``: Similar to ray_cast, returns nearest boundary point.

    See Also
    --------
    PlaceCellModel : Position-based place fields
    ObjectVectorCellModel : Object-distance tuned cells
    compute_viewed_location : Gaze computation function
    """

    def __init__(
        self,
        env: Environment,
        preferred_view_location: NDArray[np.float64],
        view_field_width: float = 10.0,
        view_distance: float = 20.0,
        gaze_model: Literal[
            "fixed_distance", "ray_cast", "boundary"
        ] = "fixed_distance",
        max_rate: float = 20.0,
        baseline_rate: float = 0.001,
        require_visibility: bool = False,
        fov: FieldOfView | None = None,
    ) -> None:
        # Convert and validate preferred_view_location
        preferred_view_location = np.asarray(preferred_view_location, dtype=np.float64)
        if preferred_view_location.shape != (2,):
            msg = (
                f"preferred_view_location must have shape (2,), "
                f"got shape {preferred_view_location.shape}"
            )
            raise ValueError(msg)

        # Validate view_field_width
        if view_field_width <= 0:
            msg = f"view_field_width must be positive, got {view_field_width}"
            raise ValueError(msg)

        # Validate view_distance
        if view_distance <= 0:
            msg = f"view_distance must be positive, got {view_distance}"
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

        # Validate gaze_model
        valid_gaze_models = ("fixed_distance", "ray_cast", "boundary")
        if gaze_model not in valid_gaze_models:
            msg = f"gaze_model must be one of {valid_gaze_models}, got '{gaze_model}'"
            raise ValueError(msg)

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
            Animal heading in radians (0=East). Required.

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

        # Compute viewed location
        viewed_locations = compute_viewed_location(
            positions,
            headings,
            method=self.gaze_model,
            view_distance=self.view_distance,
            env=self.env if self.gaze_model != "fixed_distance" else None,
        )

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

        # Optional visibility check
        if self.require_visibility:
            for i in range(n_time):
                if nan_mask[i]:
                    continue

                # Check if preferred location is in FOV
                if self.fov is not None:
                    from neurospatial.reference_frames import compute_egocentric_bearing

                    bearing = compute_egocentric_bearing(
                        self.preferred_view_location[None, :],
                        positions[i : i + 1],
                        headings[i : i + 1],
                    )[0, 0]

                    if not self.fov.contains_angle(bearing):
                        response[i] = 0.0
                        continue

                # Check line of sight to preferred location
                if not _line_of_sight_clear(
                    self.env, positions[i], self.preferred_view_location
                ):
                    response[i] = 0.0

        # Apply FOV restriction (even without visibility requirement)
        if self.fov is not None and not self.require_visibility:
            from neurospatial.reference_frames import compute_egocentric_bearing

            # Check if viewed location is in FOV
            for i in range(n_time):
                if nan_mask[i]:
                    continue

                # Compute bearing to viewed location
                bearing = compute_egocentric_bearing(
                    viewed_locations[i : i + 1],
                    positions[i : i + 1],
                    headings[i : i + 1],
                )[0, 0]

                if not self.fov.contains_angle(bearing):
                    response[i] = 0.0

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
            - 'view_distance': float - view distance for fixed_distance model
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
        >>> samples = np.random.default_rng(42).uniform(0, 100, (500, 2))
        >>> env = Environment.from_samples(samples, bin_size=2.0)
        >>> svc = SpatialViewCellModel(
        ...     env=env,
        ...     preferred_view_location=np.array([50.0, 50.0]),
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
            "max_rate": self.max_rate,
            "baseline_rate": self.baseline_rate,
            "require_visibility": self.require_visibility,
            "fov": self.fov,
        }
