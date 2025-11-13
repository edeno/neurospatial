"""Place cell models for simulating spatial firing patterns."""

from collections.abc import Callable
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from neurospatial import Environment
from neurospatial.distance import distance_field


class PlaceCellModel:
    """Gaussian place field model.

    Generalizes existing place field simulation to work with neurospatial's
    N-D environments. Supports both Euclidean and geodesic distance metrics,
    as well as conditional firing (direction-selective, speed-gated, etc.).

    Parameters
    ----------
    env : Environment
        The spatial environment.
    center : NDArray[np.float64], shape (n_dims,), optional
        Place field center in continuous coordinates.
        If None, randomly chosen from env.bin_centers.
    width : float | NDArray[np.float64], optional
        Place field width (standard deviation of Gaussian).
        Can be scalar (isotropic) or array (anisotropic per dimension).
        If None, defaults to 3 * env.bin_size.
    max_rate : float, optional
        Peak firing rate in Hz (default: 20.0).
    baseline_rate : float, optional
        Baseline firing rate outside field (default: 0.001 Hz).
    distance_metric : {'euclidean', 'geodesic'}, optional
        Method for computing distance from positions to field center.

        - 'euclidean': Straight-line distance (default). Fast but ignores
          barriers. Use for open field environments or when performance matters.
        - 'geodesic': Path distance through environment connectivity graph.
          Biologically accurate for complex environments with barriers/walls.
          ~100x slower than euclidean.
    condition : Callable[[NDArray, NDArray], NDArray[bool]] | None, optional
        Optional condition function: (positions, times) → bool mask.
        Allows direction-selective, speed-gated, or context-dependent firing.

        Common patterns:

        - Direction-selective: ``lambda pos, t: np.gradient(pos[:, 0]) > 0``
        - Speed-gated: ``lambda pos, t: compute_speed(pos, t) > threshold``
        - Region-specific: ``lambda pos, t: (pos[:, 0] > x_min) & (pos[:, 0] < x_max)``
        - Trial-type: ``lambda pos, t: trial_labels[t] == "correct"``
    seed : int | None, optional
        Random seed for reproducible random center selection.

    Attributes
    ----------
    env : Environment
        The spatial environment.
    center : NDArray[np.float64], shape (n_dims,)
        Place field center coordinates.
    width : float | NDArray[np.float64]
        Place field width(s).
    max_rate : float
        Peak firing rate in Hz.
    baseline_rate : float
        Baseline firing rate in Hz.
    distance_metric : str
        Distance metric used ('euclidean' or 'geodesic').
    condition : Callable | None
        Condition function for gated firing.

    Examples
    --------
    Simple place cell in 2D arena:

    >>> from neurospatial import Environment
    >>> from neurospatial.simulation import PlaceCellModel, simulate_trajectory_ou
    >>> import numpy as np
    >>>
    >>> # Create environment
    >>> samples = np.random.uniform(0, 100, (1000, 2))
    >>> env = Environment.from_samples(samples, bin_size=2.0)
    >>> env.units = "cm"
    >>>
    >>> # Create place cell at center of environment
    >>> center = env.bin_centers[len(env.bin_centers) // 2]
    >>> pc = PlaceCellModel(env, center=center, width=10.0, max_rate=25.0)
    >>>
    >>> # Generate trajectory and compute firing rates
    >>> positions, times = simulate_trajectory_ou(env, duration=10.0, seed=42)
    >>> rates = pc.firing_rate(positions)
    >>> rates.shape[0] == len(times)
    True
    >>> bool(rates.max() > 0.0)  # Should have some firing
    True

    Direction-selective place cell (fires only when moving right):

    >>> def rightward_only(positions, times):
    ...     if len(positions) < 2:
    ...         return np.ones(len(positions), dtype=bool)
    ...     velocity = np.gradient(positions[:, 0])
    ...     return velocity > 0
    >>>
    >>> pc_directional = PlaceCellModel(
    ...     env,
    ...     center=[50.0, 50.0],
    ...     width=10.0,
    ...     condition=rightward_only,
    ... )
    >>> rates_directional = pc_directional.firing_rate(positions, times)

    Multiple place cells tiling the environment:

    >>> # Create place cells at every 10th bin center
    >>> place_cells = [
    ...     PlaceCellModel(env, center=c, width=8.0) for c in env.bin_centers[::10]
    ... ]
    >>>
    >>> # Compute all firing rates
    >>> all_rates = np.column_stack([pc.firing_rate(positions) for pc in place_cells])
    >>> all_rates.shape[0] == len(times)  # (n_time, n_cells)
    True
    >>> all_rates.shape[1] == len(place_cells)
    True

    Geodesic distance for environment with barriers:

    >>> pc_geodesic = PlaceCellModel(
    ...     env,
    ...     center=[50.0, 50.0],
    ...     width=10.0,
    ...     distance_metric="geodesic",
    ... )
    >>> rates_geodesic = pc_geodesic.firing_rate(positions)

    See Also
    --------
    BoundaryCellModel : Boundary-distance tuned cells
    GridCellModel : Hexagonal grid patterns
    simulate_trajectory_ou : Generate realistic trajectories
    generate_poisson_spikes : Convert rates to spike times

    Notes
    -----
    **Numerical Stability**: Distances > 5*width are clipped to avoid numerical
    underflow in the Gaussian computation (contribution < 1e-6).

    **Geodesic Distance**: For geodesic metric, the distance field is precomputed
    in ``__init__`` using graph shortest paths. This makes subsequent firing rate
    computations fast (O(1) lookup per position) but adds initialization cost.

    **Condition Function**: The condition function receives the full trajectory
    and should return a boolean mask. It's called once per firing_rate() invocation,
    so it can use temporal information (e.g., velocity, acceleration).
    """

    def __init__(
        self,
        env: Environment,
        center: NDArray[np.float64] | None = None,
        width: float | NDArray[np.float64] | None = None,
        max_rate: float = 20.0,
        baseline_rate: float = 0.001,
        distance_metric: Literal["euclidean", "geodesic"] = "euclidean",
        condition: Callable[
            [NDArray[np.float64], NDArray[np.float64] | None], NDArray[np.bool_]
        ]
        | None = None,
        seed: int | None = None,
    ):
        self.env = env
        self.max_rate = max_rate
        self.baseline_rate = baseline_rate
        self.distance_metric = distance_metric
        self.condition = condition

        # Initialize random number generator for center selection
        rng = np.random.default_rng(seed)

        # Set center
        if center is None:
            # Randomly choose from bin centers
            center_idx = rng.integers(0, env.n_bins)
            self.center = env.bin_centers[center_idx].copy()
        else:
            self.center = np.asarray(center, dtype=np.float64)

        # Set width
        if width is None:
            # Default to 3 * bin_size
            # Use the mean bin_size if non-uniform bins
            bin_sizes = env.bin_sizes  # Property, not method
            self.width = 3.0 * np.mean(bin_sizes)
        else:
            self.width = width

        # Validate width compatibility with distance metric
        if distance_metric == "geodesic":
            width_arr = np.asarray(self.width)
            if width_arr.ndim > 0 and len(width_arr) > 1:
                msg = (
                    "Anisotropic width (array-like) is not supported with "
                    "distance_metric='geodesic'. Geodesic distance is a scalar "
                    "(path length through graph) without directional components. "
                    "Use scalar width or distance_metric='euclidean'."
                )
                raise ValueError(msg)

        # Precompute distance field for geodesic metric
        self._distance_field: NDArray[np.float64] | None
        if distance_metric == "geodesic":
            # Find bin containing the center
            # Always use contains check first to avoid errors
            if env.contains(self.center):
                center_bin = int(env.bin_at(self.center)[0])
            else:
                # If center is outside environment, use nearest bin
                distances = np.linalg.norm(env.bin_centers - self.center, axis=1)
                center_bin = int(np.argmin(distances))

            # Compute distance field from center bin
            # Ensure center_bin is valid
            if 0 <= center_bin < env.n_bins:
                self._distance_field = distance_field(
                    env.connectivity,
                    sources=[center_bin],
                )
            else:
                msg = f"Invalid center bin index: {center_bin}. Must be in range [0, {env.n_bins})"
                raise ValueError(msg)
        else:
            self._distance_field = None

    def firing_rate(
        self,
        positions: NDArray[np.float64],
        times: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Compute Gaussian place field firing rate.

        Parameters
        ----------
        positions : NDArray[np.float64], shape (n_time, n_dims)
            Position coordinates in the same units as the environment.
        times : NDArray[np.float64], shape (n_time,), optional
            Time points in seconds. Used for condition functions.

        Returns
        -------
        firing_rate : NDArray[np.float64], shape (n_time,)
            Firing rate in Hz at each position/time point.

        Notes
        -----
        Implementation:

        1. Compute distance from positions to center (euclidean or geodesic)
        2. Apply Gaussian with numerical stability:

           - Clip distances > 5*width (contribution < 1e-6)
           - rate = baseline + (max - baseline) * exp(-0.5 * (d/width)^2)

        3. Apply condition mask if provided (multiply rates by mask)

        For geodesic distance:

        - Map positions to bin indices using env.bin_at()
        - Precompute distance field from center bin (done in __init__)
        - Lookup distances (O(1) per position)
        """
        # Compute distances
        if self.distance_metric == "euclidean":
            # Straight-line distance
            distances = np.linalg.norm(positions - self.center, axis=1)

        elif self.distance_metric == "geodesic":
            # Graph-based distance (path through connectivity)
            # Map positions to bins
            bin_indices = np.array(
                [
                    self.env.bin_at(pos)
                    if self.env.contains(pos)
                    else np.argmin(np.linalg.norm(self.env.bin_centers - pos, axis=1))
                    for pos in positions
                ],
                dtype=np.int_,
            )

            # Lookup distances from precomputed field
            assert (
                self._distance_field is not None
            )  # Set in __init__ for geodesic metric
            distances = self._distance_field[bin_indices]

        # Note: No else needed - Literal type ensures all cases covered

        # Convert width to array for anisotropic case
        width = np.asarray(self.width)

        # For anisotropic width, compute effective distance using Mahalanobis-like metric
        if width.ndim > 0 and len(width) > 1:
            # Anisotropic: each dimension has its own width
            # d_eff = sqrt(sum((x_i - c_i)^2 / width_i^2))
            diff = positions - self.center
            normalized_diff = diff / width
            distances = np.linalg.norm(normalized_diff, axis=1)
            # Use mean width for clipping threshold
            width_for_clip = np.mean(width)
        else:
            # Isotropic: single width value
            width_for_clip = float(width)

        # Clip distances for numerical stability (Gaussian < 1e-6 beyond 5σ)
        distances = np.minimum(distances, 5 * width_for_clip)

        # Compute Gaussian firing rate
        # rate = baseline + (max - baseline) * exp(-0.5 * (d/width)^2)
        gaussian = np.exp(-0.5 * (distances / width_for_clip) ** 2)
        rates = self.baseline_rate + (self.max_rate - self.baseline_rate) * gaussian

        # Apply condition mask if provided
        if self.condition is not None:
            mask = self.condition(positions, times)
            rates = rates * mask.astype(np.float64)

        return rates  # type: ignore[no-any-return]

    @property
    def ground_truth(self) -> dict[str, Any]:
        """Return ground truth parameters for validation.

        Returns
        -------
        parameters : dict[str, Any]
            Dictionary with keys:

            - 'center': NDArray[np.float64] - field center coordinates
            - 'width': float | NDArray[np.float64] - field width
            - 'max_rate': float - peak firing rate in Hz
            - 'baseline_rate': float - baseline firing rate in Hz

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.simulation import PlaceCellModel
        >>> samples = np.random.uniform(0, 100, (100, 2))
        >>> env = Environment.from_samples(samples, bin_size=2.0)
        >>> env.units = "cm"
        >>> pc = PlaceCellModel(env, center=[50.0, 50.0], width=10.0, max_rate=20.0)
        >>> np.allclose(pc.ground_truth["center"], [50.0, 50.0])
        True
        >>> pc.ground_truth["width"]
        10.0
        >>> pc.ground_truth["max_rate"]
        20.0
        """
        return {
            "center": self.center,
            "width": self.width,
            "max_rate": self.max_rate,
            "baseline_rate": self.baseline_rate,
        }
