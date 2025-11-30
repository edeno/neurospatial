"""Boundary cell models for simulating boundary-distance tuned firing patterns."""

from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.stats import vonmises

from neurospatial import Environment
from neurospatial.distance import distance_field


class BoundaryCellModel:
    """Distance-to-boundary tuned firing model.

    Models boundary vector cells (BVCs) or border cells that fire at specific
    distances from environment boundaries, optionally with directional tuning.

    Parameters
    ----------
    env : Environment
        The spatial environment.
    preferred_distance : float, optional
        Preferred distance from boundary in environment units (default: 5.0).
    distance_tolerance : float, optional
        Tuning width (standard deviation) around preferred distance (default: 3.0).
    preferred_direction : float | None, optional
        Preferred allocentric direction in radians (2D only).
        If None, responds to all directions (classic border cell).
        If specified, responds to boundaries in that direction (BVC).
        Convention: 0 = East, π/2 = North, π = West, -π/2 = South
    direction_tolerance : float, optional
        Direction tuning width (concentration parameter for von Mises) in
        radians (default: π/4). Smaller values = sharper tuning.
    max_rate : float, optional
        Peak firing rate in Hz (default: 15.0).
    baseline_rate : float, optional
        Baseline firing rate away from preferred distance (default: 0.001 Hz).
    distance_metric : {'geodesic', 'euclidean'}, optional
        Distance calculation method (default: 'geodesic').

        - 'geodesic': Path distance through environment connectivity graph.
          Biologically accurate for complex environments.
        - 'euclidean': Straight-line distance. Faster but ignores barriers.

    Attributes
    ----------
    env : Environment
        The spatial environment.
    preferred_distance : float
        Preferred distance from boundary.
    distance_tolerance : float
        Distance tuning width.
    preferred_direction : float | None
        Preferred direction (radians) or None for omnidirectional.
    direction_tolerance : float
        Direction tuning width (radians).
    max_rate : float
        Peak firing rate in Hz.
    baseline_rate : float
        Baseline firing rate in Hz.
    distance_metric : str
        Distance metric used ('geodesic' or 'euclidean').

    Examples
    --------
    Classic border cell (fires at 5 cm from any wall):

    >>> from neurospatial import Environment
    >>> from neurospatial.simulation import BoundaryCellModel, simulate_trajectory_ou
    >>> import numpy as np
    >>>
    >>> # Create environment
    >>> samples = np.random.uniform(0, 100, (1000, 2))
    >>> env = Environment.from_samples(samples, bin_size=2.0)
    >>> env.units = "cm"
    >>>
    >>> # Create omnidirectional boundary cell
    >>> bc = BoundaryCellModel(env, preferred_distance=5.0, distance_tolerance=3.0)
    >>>
    >>> # Generate trajectory and compute firing rates
    >>> positions, times = simulate_trajectory_ou(env, duration=10.0, seed=42)
    >>> rates = bc.firing_rate(positions)
    >>> rates.shape[0] == len(times)
    True
    >>> bool(rates.max() > 0.0)  # Should have some firing
    True

    Boundary vector cell (prefers south wall):

    >>> # BVC tuned to south wall (negative y direction)
    >>> bvc_south = BoundaryCellModel(
    ...     env,
    ...     preferred_distance=10.0,
    ...     preferred_direction=-np.pi / 2,  # South
    ...     direction_tolerance=np.pi / 6,  # ±30 degrees
    ... )
    >>> rates_directional = bvc_south.firing_rate(positions)

    See Also
    --------
    PlaceCellModel : Gaussian place fields
    GridCellModel : Hexagonal grid patterns
    simulate_trajectory_ou : Generate realistic trajectories
    generate_poisson_spikes : Convert rates to spike times

    Notes
    -----
    **Boundary Detection**: Uses ``env.boundary_bins()`` to identify boundary
    locations, then precomputes distance field from all boundary bins.

    **Gaussian Distance Tuning**: Firing rate follows Gaussian around
    preferred_distance:

    .. math::

        r_{dist}(d) = \\exp\\left(-\\frac{(d - d_{pref})^2}{2 \\sigma_d^2}\\right)

    where :math:`d` is distance to nearest boundary, :math:`d_{pref}` is
    preferred_distance, and :math:`\\sigma_d` is distance_tolerance.

    **Directional Tuning**: For BVCs (when preferred_direction is specified),
    applies von Mises (circular Gaussian) tuning around preferred direction:

    .. math::

        r_{dir}(\\theta) = \\exp(\\kappa \\cos(\\theta - \\theta_{pref}))

    where :math:`\\theta` is direction to boundary, :math:`\\theta_{pref}` is
    preferred_direction, and :math:`\\kappa` is direction concentration
    (inversely related to direction_tolerance).

    **Final Rate**: Combines distance and directional tuning:

    .. math::

        r(x) = r_{base} + (r_{max} - r_{base}) \\cdot r_{dist}(d) \\cdot r_{dir}(\\theta)
    """

    def __init__(
        self,
        env: Environment,
        preferred_distance: float = 5.0,
        distance_tolerance: float = 3.0,
        preferred_direction: float | None = None,
        direction_tolerance: float = np.pi / 4,
        max_rate: float = 15.0,
        baseline_rate: float = 0.001,
        distance_metric: Literal["geodesic", "euclidean"] = "geodesic",
    ) -> None:
        # Validate parameters
        if preferred_distance < 0:
            msg = f"preferred_distance must be non-negative (got {preferred_distance})"
            raise ValueError(msg)

        if distance_tolerance <= 0:
            msg = f"distance_tolerance must be positive (got {distance_tolerance})"
            raise ValueError(msg)

        if max_rate <= 0:
            msg = f"max_rate must be positive (got {max_rate})"
            raise ValueError(msg)

        if baseline_rate < 0:
            msg = f"baseline_rate must be non-negative (got {baseline_rate})"
            raise ValueError(msg)

        if baseline_rate >= max_rate:
            msg = f"baseline_rate ({baseline_rate}) must be less than max_rate ({max_rate})"
            raise ValueError(msg)

        if direction_tolerance <= 0:
            msg = f"direction_tolerance must be positive (got {direction_tolerance})"
            raise ValueError(msg)

        self.env = env
        self.preferred_distance = preferred_distance
        self.distance_tolerance = distance_tolerance
        self.preferred_direction = preferred_direction
        self.direction_tolerance = direction_tolerance
        self.max_rate = max_rate
        self.baseline_rate = baseline_rate
        self.distance_metric = distance_metric

        # Precompute boundary bins
        self._boundary_bins = env.boundary_bins

        if len(self._boundary_bins) == 0:
            msg = "No boundary bins found in environment"
            raise ValueError(msg)

        # Precompute distance field from all boundary bins (type annotation for mypy)
        self._distance_field: NDArray[np.float64] | None
        if distance_metric == "geodesic":
            # Geodesic distance through graph
            self._distance_field = distance_field(
                env.connectivity,
                sources=self._boundary_bins.tolist(),
            )
        else:
            # Will compute Euclidean distances on-the-fly in firing_rate()
            self._distance_field = None

    def firing_rate(
        self,
        positions: NDArray[np.float64],
        times: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Compute boundary-tuned firing rate.

        Parameters
        ----------
        positions : NDArray[np.float64], shape (n_time, n_dims)
            Position coordinates in the same units as the environment.
        times : NDArray[np.float64], shape (n_time,), optional
            Time points in seconds (not used, for API compatibility).

        Returns
        -------
        firing_rate : NDArray[np.float64], shape (n_time,)
            Firing rate in Hz at each position/time point.

        Notes
        -----
        Implementation:

        1. **Compute distance to nearest boundary** for each position:

           - For geodesic: lookup from precomputed distance field
           - For euclidean: compute minimum distance to all boundary centers

        2. **Apply Gaussian distance tuning** around preferred_distance:

           .. math::

               g_d = \\exp\\left(-\\frac{(d - d_{pref})^2}{2 \\sigma_d^2}\\right)

        3. **If directional**, compute direction to nearest boundary and apply
           von Mises directional tuning around preferred_direction.

        4. **Scale and add baseline**:

           .. math::

               r = r_{base} + (r_{max} - r_{base}) \\cdot g_d \\cdot g_{dir}
        """
        # Precompute boundary centers (used in multiple paths)
        boundary_centers = self.env.bin_centers[self._boundary_bins]

        # Compute distances to boundary
        if self.distance_metric == "geodesic":
            # Map positions to bins and lookup distances - vectorized for efficiency
            # contains() returns ndarray, so we can use it directly for batch check
            inside_mask = self.env.contains(positions)
            bin_indices = np.empty(len(positions), dtype=np.int_)

            # For points inside environment, use bin_at
            if inside_mask.any():
                bin_indices[inside_mask] = self.env.bin_at(positions[inside_mask])

            # For points outside, use vectorized nearest neighbor
            if not inside_mask.all():
                outside_positions = positions[~inside_mask]
                # Vectorized distance computation: (n_outside, n_bins)
                distances_to_bins = np.linalg.norm(
                    self.env.bin_centers[None, :, :] - outside_positions[:, None, :],
                    axis=2,
                )
                bin_indices[~inside_mask] = np.argmin(distances_to_bins, axis=1)

            # Lookup distances from precomputed field
            assert self._distance_field is not None
            distances = self._distance_field[bin_indices]

        else:  # euclidean
            # Compute minimum Euclidean distance to any boundary bin center
            # Shape: (n_positions, n_boundary_bins)
            dists_to_boundaries = np.linalg.norm(
                positions[:, np.newaxis, :] - boundary_centers[np.newaxis, :, :],
                axis=2,
            )
            # Minimum distance for each position
            distances = np.min(dists_to_boundaries, axis=1)

        # Apply Gaussian distance tuning around preferred_distance
        # g_d = exp(-(d - d_pref)^2 / (2 * sigma^2))
        distance_diff = distances - self.preferred_distance
        distance_tuning = np.exp(-0.5 * (distance_diff / self.distance_tolerance) ** 2)

        # Apply directional tuning if preferred_direction is specified
        if self.preferred_direction is not None:
            # Compute direction to nearest boundary for each position
            # For geodesic metric, use Euclidean distance for direction
            # (direction is inherently geometric, not path-based)
            if self.distance_metric == "geodesic":
                # Recompute Euclidean distances for directional tuning
                dists_to_boundaries = np.linalg.norm(
                    positions[:, np.newaxis, :] - boundary_centers[np.newaxis, :, :],
                    axis=2,
                )

            # Now both paths have dists_to_boundaries defined
            nearest_boundary_idx = np.argmin(dists_to_boundaries, axis=1)
            nearest_boundary_centers = boundary_centers[nearest_boundary_idx]

            # Compute direction vectors from positions to nearest boundaries
            direction_vectors = nearest_boundary_centers - positions

            # Compute angles (only works for 2D environments)
            if positions.shape[1] == 2:
                angles = np.arctan2(direction_vectors[:, 1], direction_vectors[:, 0])

                # Apply von Mises directional tuning
                # von Mises PDF: exp(kappa * cos(theta - theta_pref))
                # Convert direction_tolerance to kappa (concentration parameter)
                # Smaller tolerance -> larger kappa -> sharper tuning
                kappa = 1.0 / self.direction_tolerance

                # Compute angle difference (wrapped to [-π, π])
                angle_diff = np.angle(np.exp(1j * (angles - self.preferred_direction)))
                directional_tuning = vonmises.pdf(angle_diff, kappa, loc=0.0)

                # Normalize von Mises to have max = 1.0
                directional_tuning = directional_tuning / vonmises.pdf(
                    0.0, kappa, loc=0.0
                )
            else:
                # For non-2D environments, ignore directional tuning
                directional_tuning = 1.0
        else:
            # Omnidirectional (no directional preference)
            directional_tuning = 1.0

        # Combine distance and directional tuning
        combined_tuning = distance_tuning * directional_tuning

        # Scale by max_rate and add baseline
        rates: NDArray[np.float64] = (
            self.baseline_rate + (self.max_rate - self.baseline_rate) * combined_tuning
        )

        return rates

    @property
    def ground_truth(self) -> dict[str, Any]:
        """Return ground truth parameters for validation.

        Returns
        -------
        parameters : dict[str, Any]
            Dictionary with keys:

            - 'preferred_distance': float - preferred distance from boundary
            - 'distance_tolerance': float - distance tuning width
            - 'preferred_direction': float | None - preferred direction (radians)
            - 'direction_tolerance': float - direction tuning width (radians)
            - 'max_rate': float - peak firing rate in Hz
            - 'baseline_rate': float - baseline firing rate in Hz

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.simulation import BoundaryCellModel
        >>> samples = np.random.uniform(0, 100, (100, 2))
        >>> env = Environment.from_samples(samples, bin_size=2.0)
        >>> env.units = "cm"
        >>> bc = BoundaryCellModel(env, preferred_distance=5.0, distance_tolerance=3.0)
        >>> bc.ground_truth["preferred_distance"]
        5.0
        >>> bc.ground_truth["distance_tolerance"]
        3.0
        >>> bc.ground_truth["preferred_direction"] is None
        True
        """
        return {
            "preferred_distance": self.preferred_distance,
            "distance_tolerance": self.distance_tolerance,
            "preferred_direction": self.preferred_direction,
            "direction_tolerance": self.direction_tolerance,
            "max_rate": self.max_rate,
            "baseline_rate": self.baseline_rate,
        }
