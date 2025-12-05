"""Object-vector cell models for simulating distance and direction tuned firing.

Object-vector cells (OVCs) fire when an animal is at a specific distance and
direction from an object in the environment. They were discovered in the medial
entorhinal cortex (MEC) and are thought to support object-location memory.

Key Features
------------
- **Distance tuning**: Gaussian tuning around preferred distance from objects
- **Direction tuning**: Optional von Mises (circular Gaussian) tuning around
  preferred egocentric direction
- **Object selectivity**: Respond to any object, nearest object, or specific object
- **Distance metrics**: Euclidean (straight-line) or geodesic (path through graph)

Coordinate Conventions
----------------------
**Egocentric direction (when ``preferred_direction`` is specified)**:
- 0 radians = object is directly ahead of animal
- π/2 radians = object is to the left
- -π/2 radians = object is to the right
- ±π radians = object is behind

This matches the coordinate convention in ``neurospatial.reference_frames``.

Examples
--------
Create a simple object-vector cell:

>>> import numpy as np
>>> from neurospatial import Environment
>>> from neurospatial.simulation.models.object_vector_cells import ObjectVectorCellModel
>>>
>>> # Create environment
>>> samples = np.random.default_rng(42).uniform(0, 100, (500, 2))
>>> env = Environment.from_samples(samples, bin_size=2.0)
>>>
>>> # Define object positions
>>> objects = np.array([[25.0, 50.0], [75.0, 50.0]])
>>>
>>> # Create OVC that fires 10 units from nearest object
>>> ovc = ObjectVectorCellModel(
...     env=env,
...     object_positions=objects,
...     preferred_distance=10.0,
...     distance_width=5.0,
... )

Create a directionally-tuned OVC (fires when object is ahead):

>>> ovc_directional = ObjectVectorCellModel(
...     env=env,
...     object_positions=objects,
...     preferred_distance=10.0,
...     distance_width=5.0,
...     preferred_direction=0.0,  # Object ahead
...     direction_kappa=4.0,  # ~30° half-width
... )

References
----------
.. [1] Hoydal, O. A., et al. (2019). Object-vector coding in the medial
       entorhinal cortex. Nature, 568(7752), 400-404.
       https://doi.org/10.1038/s41586-019-1077-7

.. [2] Deshmukh, S. S., & Knierim, J. J. (2011). Representation of non-spatial
       and spatial information in the lateral entorhinal cortex. Frontiers in
       Behavioral Neuroscience, 5, 69.

See Also
--------
neurospatial.reference_frames : Coordinate transformations
PlaceCellModel : Gaussian place fields
BoundaryCellModel : Boundary-distance tuned cells
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from neurospatial.distance import distance_field
from neurospatial.reference_frames import compute_egocentric_bearing

if TYPE_CHECKING:
    from neurospatial import Environment

__all__ = ["ObjectVectorCellModel"]


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
    preferred_distance : float
        Preferred distance from object in environment units.
    distance_width : float
        Distance tuning width (standard deviation of Gaussian).
    preferred_direction : float | None, optional
        Preferred egocentric direction in radians (default: None).
        If None, responds to objects at preferred distance in any direction.
        If specified, uses von Mises directional tuning.
        Convention: 0=ahead, π/2=left, -π/2=right, ±π=behind.
    direction_kappa : float, optional
        Direction tuning concentration parameter (default: 4.0).
        Higher values = sharper tuning. κ=4 ≈ 30° half-width at half-max.
    max_rate : float, optional
        Peak firing rate in Hz (default: 20.0).
    baseline_rate : float, optional
        Baseline firing rate outside tuned region (default: 0.001 Hz).
    object_selectivity : {'any', 'nearest', 'specific'}, optional
        How to aggregate responses across objects (default: 'nearest').

        - 'any': Maximum response across all objects
        - 'nearest': Response to nearest object only
        - 'specific': Response to specific object (requires ``specific_object_index``)
    specific_object_index : int | None, optional
        Index of object to respond to when ``object_selectivity='specific'``.
    distance_metric : {'euclidean', 'geodesic'}, optional
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
    distance_width : float
        Distance tuning width.
    preferred_direction : float | None
        Preferred egocentric direction (radians) or None.
    direction_kappa : float
        Direction tuning concentration.
    max_rate : float
        Peak firing rate in Hz.
    baseline_rate : float
        Baseline firing rate in Hz.
    object_selectivity : str
        Object selectivity mode.
    distance_metric : str
        Distance metric used.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.simulation.models.object_vector_cells import (
    ...     ObjectVectorCellModel,
    ... )
    >>>
    >>> # Create environment
    >>> samples = np.random.default_rng(42).uniform(0, 100, (500, 2))
    >>> env = Environment.from_samples(samples, bin_size=2.0)
    >>>
    >>> # Create OVC
    >>> ovc = ObjectVectorCellModel(
    ...     env=env,
    ...     object_positions=np.array([[50.0, 50.0]]),
    ...     preferred_distance=10.0,
    ...     distance_width=5.0,
    ... )
    >>>
    >>> # Compute firing rates along a trajectory
    >>> positions = np.random.default_rng(42).uniform(0, 100, (100, 2))
    >>> rates = ovc.firing_rate(positions)
    >>> rates.shape
    (100,)

    Notes
    -----
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
    BoundaryCellModel : Boundary-distance tuned cells
    compute_egocentric_bearing : Compute bearing to targets
    """

    def __init__(
        self,
        env: Environment,
        object_positions: NDArray[np.float64],
        preferred_distance: float,
        distance_width: float,
        preferred_direction: float | None = None,
        direction_kappa: float = 4.0,
        max_rate: float = 20.0,
        baseline_rate: float = 0.001,
        object_selectivity: Literal["any", "nearest", "specific"] = "nearest",
        specific_object_index: int | None = None,
        distance_metric: Literal["euclidean", "geodesic"] = "euclidean",
    ) -> None:
        # Convert and validate object_positions
        object_positions = np.asarray(object_positions, dtype=np.float64)
        if object_positions.ndim != 2:
            msg = (
                f"object_positions must be 2D with shape (n_objects, 2), "
                f"got shape {object_positions.shape}"
            )
            raise ValueError(msg)

        if object_positions.shape[1] != 2:
            msg = (
                f"object_positions must have 2 columns (x, y coordinates), "
                f"got {object_positions.shape[1]} columns"
            )
            raise ValueError(msg)

        # Validate preferred_distance
        if preferred_distance < 0:
            msg = f"preferred_distance must be non-negative, got {preferred_distance}"
            raise ValueError(msg)

        # Validate distance_width
        if distance_width <= 0:
            msg = f"distance_width must be positive, got {distance_width}"
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

        # Validate direction_kappa (only if direction tuning is enabled)
        if preferred_direction is not None and direction_kappa <= 0:
            msg = f"direction_kappa must be positive, got {direction_kappa}"
            raise ValueError(msg)

        # Validate object_selectivity
        valid_selectivities = ("any", "nearest", "specific")
        if object_selectivity not in valid_selectivities:
            msg = (
                f"object_selectivity must be one of {valid_selectivities}, "
                f"got '{object_selectivity}'"
            )
            raise ValueError(msg)

        # Validate specific_object_index
        if object_selectivity == "specific":
            if specific_object_index is None:
                msg = "specific_object_index is required when object_selectivity='specific'"
                raise ValueError(msg)
            if not (0 <= specific_object_index < len(object_positions)):
                msg = (
                    f"specific_object_index ({specific_object_index}) must be in range "
                    f"[0, {len(object_positions)})"
                )
                raise ValueError(msg)

        # Validate distance_metric
        valid_metrics = ("euclidean", "geodesic")
        if distance_metric not in valid_metrics:
            msg = (
                f"distance_metric must be one of {valid_metrics}, "
                f"got '{distance_metric}'"
            )
            raise ValueError(msg)

        # Check if objects are outside environment bounds
        contains_mask = env.contains(object_positions)
        if not np.all(contains_mask):
            n_outside = np.sum(~contains_mask)
            warnings.warn(
                f"{n_outside} object(s) are outside the environment bounds. "
                f"Their distances may be computed from the nearest bin.",
                UserWarning,
                stacklevel=2,
            )

        # Store parameters
        self.env = env
        self.object_positions = object_positions
        self.preferred_distance = preferred_distance
        self.distance_width = distance_width
        self.preferred_direction = preferred_direction
        self.direction_kappa = direction_kappa
        self.max_rate = max_rate
        self.baseline_rate = baseline_rate
        self.object_selectivity = object_selectivity
        self.specific_object_index = specific_object_index
        self.distance_metric = distance_metric

        # Precompute distance fields for geodesic metric
        self._distance_fields: list[NDArray[np.float64]] | None
        if distance_metric == "geodesic":
            self._distance_fields = []
            for obj_pos in object_positions:
                # Find bin containing the object
                if env.contains(obj_pos[None, :])[0]:
                    obj_bin = int(env.bin_at(obj_pos[None, :])[0])
                else:
                    # Use nearest bin if outside environment
                    distances = np.linalg.norm(env.bin_centers - obj_pos, axis=1)
                    obj_bin = int(np.argmin(distances))

                # Compute distance field from this object's bin
                dist_field = distance_field(env.connectivity, sources=[obj_bin])
                self._distance_fields.append(dist_field)
        else:
            self._distance_fields = None

    def firing_rate(
        self,
        positions: NDArray[np.float64],
        times: NDArray[np.float64] | None = None,
        headings: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Compute object-vector cell firing rate.

        Parameters
        ----------
        positions : NDArray[np.float64], shape (n_time, 2)
            Animal positions in allocentric coordinates.
        times : NDArray[np.float64], shape (n_time,), optional
            Time points in seconds (not used, for API compatibility).
        headings : NDArray[np.float64], shape (n_time,), optional
            Animal heading in radians. Required for directional tuning.
            If None and ``preferred_direction`` is set, will raise ValueError.

        Returns
        -------
        firing_rate : NDArray[np.float64], shape (n_time,)
            Firing rate in Hz at each position.

        Raises
        ------
        ValueError
            If ``preferred_direction`` is set but ``headings`` is None.

        Notes
        -----
        Implementation:

        1. Compute distances from positions to all objects
        2. Apply Gaussian distance tuning
        3. If directional: compute bearing to objects and apply von Mises tuning
        4. Aggregate by object selectivity mode
        5. Scale by max_rate and add baseline
        """
        positions = np.asarray(positions, dtype=np.float64)
        n_time = len(positions)
        n_objects = len(self.object_positions)

        # Validate headings if directional tuning is enabled
        if self.preferred_direction is not None and headings is None:
            msg = "headings is required when preferred_direction is set"
            raise ValueError(msg)

        # Compute distances from positions to all objects
        # Shape: (n_time, n_objects)
        if self.distance_metric == "euclidean":
            # Euclidean distance
            # positions: (n_time, 2)
            # object_positions: (n_objects, 2)
            distances = np.linalg.norm(
                positions[:, np.newaxis, :] - self.object_positions[np.newaxis, :, :],
                axis=2,
            )
        else:
            # Geodesic distance
            assert self._distance_fields is not None

            # Map positions to bins
            inside_mask = self.env.contains(positions)
            bin_indices = np.empty(n_time, dtype=np.int_)

            if inside_mask.any():
                bin_indices[inside_mask] = self.env.bin_at(positions[inside_mask])

            if not inside_mask.all():
                outside_positions = positions[~inside_mask]
                dists_to_bins = np.linalg.norm(
                    self.env.bin_centers[None, :, :] - outside_positions[:, None, :],
                    axis=2,
                )
                bin_indices[~inside_mask] = np.argmin(dists_to_bins, axis=1)

            # Lookup distances from precomputed fields
            distances = np.empty((n_time, n_objects), dtype=np.float64)
            for i, dist_field in enumerate(self._distance_fields):
                distances[:, i] = dist_field[bin_indices]

        # Apply Gaussian distance tuning
        # distance_response: (n_time, n_objects)
        distance_diff = distances - self.preferred_distance
        distance_response = np.exp(-0.5 * (distance_diff / self.distance_width) ** 2)

        # Apply directional tuning if specified
        if self.preferred_direction is not None:
            assert headings is not None

            # Compute bearing to each object
            # Shape: (n_time, n_objects)
            bearings = compute_egocentric_bearing(
                self.object_positions, positions, headings
            )

            # Apply von Mises directional tuning
            # von Mises: exp(κ * cos(θ - θ_pref)) / exp(κ)
            angle_diff = bearings - self.preferred_direction
            direction_response = np.exp(
                self.direction_kappa * np.cos(angle_diff)
            ) / np.exp(self.direction_kappa)

            # Combine distance and direction tuning
            combined_response = distance_response * direction_response
        else:
            combined_response = distance_response

        # Aggregate by object selectivity
        if self.object_selectivity == "any":
            # Maximum response across all objects
            response = np.max(combined_response, axis=1)
        elif self.object_selectivity == "nearest":
            # Response to nearest object only
            nearest_obj = np.argmin(distances, axis=1)
            response = combined_response[np.arange(n_time), nearest_obj]
        else:  # specific
            assert self.specific_object_index is not None
            response = combined_response[:, self.specific_object_index]

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

            - 'object_positions': NDArray - positions of objects
            - 'preferred_distance': float - preferred distance from object
            - 'distance_width': float - distance tuning width
            - 'preferred_direction': float | None - preferred direction (radians)
            - 'direction_kappa': float - direction tuning concentration
            - 'max_rate': float - peak firing rate in Hz
            - 'baseline_rate': float - baseline firing rate in Hz
            - 'object_selectivity': str - object selectivity mode

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.simulation.models.object_vector_cells import (
        ...     ObjectVectorCellModel,
        ... )
        >>> samples = np.random.default_rng(42).uniform(0, 100, (500, 2))
        >>> env = Environment.from_samples(samples, bin_size=2.0)
        >>> ovc = ObjectVectorCellModel(
        ...     env=env,
        ...     object_positions=np.array([[50.0, 50.0]]),
        ...     preferred_distance=10.0,
        ...     distance_width=5.0,
        ... )
        >>> ovc.ground_truth["preferred_distance"]
        10.0
        """
        return {
            "object_positions": self.object_positions.copy(),
            "preferred_distance": self.preferred_distance,
            "distance_width": self.distance_width,
            "preferred_direction": self.preferred_direction,
            "direction_kappa": self.direction_kappa,
            "max_rate": self.max_rate,
            "baseline_rate": self.baseline_rate,
            "object_selectivity": self.object_selectivity,
        }
