"""Trajectory characterization metrics.

This module provides functions for analyzing spatial trajectories, including
turn angles, step lengths, home range estimation, and mean square displacement
analysis. These metrics are commonly used in behavioral ecology and neuroscience
to characterize movement patterns and spatial exploration.

References
----------
.. [1] Traja: https://traja.readthedocs.io/ - Trajectory analysis toolkit
.. [2] Fiaschi, L., Nair, R., Koethe, U., & Hamprecht, F. A. (2012).
       "Learning to count with regression forest and structured labels."
       ICPR 2012.
.. [3] Kays, R., Crofoot, M. C., Jetz, W., & Wikelski, M. (2015).
       "Terrestrial animal tracking as an eye on life and planet."
       Science, 348(6240), aaa2478.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from neurospatial.environment.core import Environment


def compute_turn_angles(
    positions: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compute turn angles between consecutive movement vectors from continuous positions.

    Turn angles quantify changes in movement direction at each position along
    the trajectory. A turn angle of 0 indicates straight movement, positive
    angles indicate left turns, and negative angles indicate right turns.

    This function operates on continuous position data, preserving sub-bin spatial
    precision for accurate directional analysis. This is the ecology-standard
    approach and is more accurate than bin-based computation.

    Parameters
    ----------
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Trajectory positions in continuous space. Each row is a position vector.
        For 2D trajectories, shape is (n_samples, 2). For 1D, shape is (n_samples, 1).

    Returns
    -------
    NDArray[np.float64], shape (n_angles,)
        Turn angles in radians, in the range [-π, π]. Length is the number
        of movement transitions minus 1 (n_samples - 2). Consecutive duplicate
        positions (stationary periods) are filtered out before computing angles.

    Notes
    -----
    Turn angles are computed from the angle between consecutive movement vectors:

    .. math::

        \\theta_i = \\text{atan2}(v_{i+1} \\times v_i, v_{i+1} \\cdot v_i)

    where :math:`v_i` is the movement vector from position i-1 to position i.

    For 2D trajectories, this uses the 2D cross product (scalar). For higher
    dimensions, the function uses the first two dimensions only.

    Stationary periods (consecutive duplicate positions) are removed before
    computing turn angles to avoid undefined angles from zero-length vectors.

    **Why continuous positions?**

    Using continuous positions instead of discretized bins preserves directional
    precision. For example, a gradual turn from [0,0]→[1,0.1]→[2,0.3] would all
    map to the same bins on a coarse grid, giving zero turn angle (incorrect),
    whereas continuous computation correctly captures the subtle turning.

    See Also
    --------
    compute_step_lengths : Compute distances between consecutive positions
    mean_square_displacement : Analyze diffusive properties of trajectory

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.behavior.trajectory import compute_turn_angles
    >>>
    >>> # Straight line trajectory
    >>> positions = np.column_stack([np.linspace(0, 100, 20), np.zeros(20)])
    >>> angles = compute_turn_angles(positions)
    >>> np.allclose(angles, 0.0, atol=0.01)  # Straight = no turning
    True
    >>>
    >>> # Right-angle turn: [0,0] → [10,0] → [10,10]
    >>> positions = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
    >>> angles = compute_turn_angles(positions)
    >>> angles[0]  # doctest: +SKIP
    1.5707963267948966  # ≈ π/2 (90° left turn)

    References
    ----------
    .. [1] Batschelet, E. (1981). Circular Statistics in Biology.
           Academic Press.
    .. [2] Muller, M., & Wehner, R. (1988). "Path integration in desert ants."
           PNAS, 85(14), 5287-5290.
    .. [3] Traja documentation: https://traja.readthedocs.io/
    """
    # Input validation
    if positions.ndim != 2:
        raise ValueError(
            f"positions must be 2D array (n_samples, n_dims), got {positions.ndim}D"
        )

    if len(positions) < 3:
        # Need at least 3 positions to compute turn angles
        return np.array([], dtype=np.float64)

    # Remove consecutive duplicates (stationary periods)
    # Compare all dimensions for exact equality
    is_different = np.any(positions[1:] != positions[:-1], axis=1)
    # Keep first position and all positions that differ from previous
    mask = np.concatenate([[True], is_different])
    unique_positions = positions[mask]

    # Need at least 3 unique positions to compute turn angles
    if len(unique_positions) < 3:
        return np.array([], dtype=np.float64)

    # Compute movement vectors (differences between consecutive positions)
    vectors = np.diff(unique_positions, axis=0)  # shape (n_unique-1, n_dims)

    # Compute turn angles from consecutive vectors
    # For each pair of consecutive vectors, compute the angle between them
    n_angles = len(vectors) - 1
    angles = np.zeros(n_angles, dtype=np.float64)

    for i in range(n_angles):
        v1 = vectors[i]
        v2 = vectors[i + 1]

        # Check for zero-length vectors (should not occur after duplicate removal, but be safe)
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            angles[i] = 0.0
            continue

        # Use atan2 for proper quadrant handling
        # For 2D: angle = atan2(cross product, dot product)
        # For higher dims: use first 2 dimensions
        if len(v1) >= 2:
            # 2D cross product (scalar)
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            dot = np.dot(v1, v2)
            angles[i] = np.arctan2(cross, dot)
        else:
            # 1D: no turn angles (always 0 or π)
            # Check if vectors point in same direction
            if v1[0] * v2[0] > 0:
                angles[i] = 0.0
            else:
                angles[i] = np.pi

    return angles


def compute_step_lengths(
    positions: NDArray[np.float64],
    *,
    distance_type: Literal["euclidean", "geodesic"] = "euclidean",
    env: Environment | None = None,
) -> NDArray[np.float64]:
    """
    Compute step lengths (distances) between consecutive positions.

    Step lengths quantify the spatial displacement at each step of the trajectory.
    This function supports both Euclidean (straight-line) and geodesic (graph
    shortest-path) distance metrics for different use cases.

    Parameters
    ----------
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Trajectory positions in continuous space. Each row is a position vector.
    distance_type : {"euclidean", "geodesic"}, default="euclidean"
        Distance metric to use:
        - "euclidean": Straight-line distance in physical space (ecology standard).
          Fast and appropriate for open environments without obstacles.
        - "geodesic": Graph shortest-path distance respecting environment topology.
          Requires `env` parameter. Appropriate for constrained navigation
          (walls, obstacles, tracks).
    env : Environment, optional
        Environment instance for computing geodesic distances. Required if
        distance_type="geodesic", ignored otherwise.

    Returns
    -------
    NDArray[np.float64], shape (n_samples - 1,)
        Step lengths in the same units as the position coordinates.

    Raises
    ------
    ValueError
        If distance_type="geodesic" but env is None.
        If positions is not a 2D array.
        If distance_type is not "euclidean" or "geodesic".

    Notes
    -----
    **Euclidean distance** (default):

    .. math::

        d_i = ||r_{i+1} - r_i||_2 = \\sqrt{\\sum_j (r_{i+1,j} - r_{i,j})^2}

    This is the straight-line distance in physical space, matching ecology
    literature standards (Kays et al. 2015, Traja).

    **Geodesic distance** (graph-based):

    Uses NetworkX shortest_path_length on env.connectivity graph with edge
    weights. This respects environmental constraints (walls, obstacles) and is
    appropriate for neuroscience applications with constrained environments.

    **When to use each**:

    - **Euclidean**: Open-field recordings, ecology studies, comparisons to
      movement ecology literature, maximum precision
    - **Geodesic**: Linear tracks, T-mazes, environments with barriers,
      neuroscience-specific analyses

    See Also
    --------
    compute_turn_angles : Compute turn angles between movements
    mean_square_displacement : Analyze diffusive properties

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.behavior.trajectory import compute_step_lengths
    >>>
    >>> # Straight line trajectory with Euclidean distance
    >>> positions = np.column_stack([np.linspace(0, 100, 20), np.zeros(20)])
    >>> step_lengths = compute_step_lengths(positions, distance_type="euclidean")
    >>> len(step_lengths)
    19
    >>> np.allclose(step_lengths, step_lengths[0], rtol=0.01)  # Uniform steps
    True
    >>>
    >>> # Geodesic distance on graph (requires env)
    >>> from neurospatial import Environment
    >>> env = Environment.from_samples(positions, bin_size=5.0)
    >>> # Map positions to bins for geodesic
    >>> bins = env.bin_at(positions)
    >>> # Use bin centers as proxy positions for geodesic
    >>> bin_positions = env.bin_centers[bins]
    >>> step_lengths_geo = compute_step_lengths(
    ...     bin_positions, distance_type="geodesic", env=env
    ... )  # doctest: +SKIP

    References
    ----------
    .. [1] Kays, R., et al. (2015). "Terrestrial animal tracking as an eye
           on life and planet." Science, 348(6240), aaa2478.
    .. [2] Traja documentation: https://traja.readthedocs.io/
    """
    # Input validation
    if positions.ndim != 2:
        raise ValueError(
            f"positions must be 2D array (n_samples, n_dims), got {positions.ndim}D"
        )

    if distance_type not in ("euclidean", "geodesic"):
        raise ValueError(
            f"distance_type must be 'euclidean' or 'geodesic', got '{distance_type}'"
        )

    if distance_type == "geodesic" and env is None:
        raise ValueError(
            "distance_type='geodesic' requires env parameter. "
            "Use distance_type='euclidean' if env is not available."
        )

    n_steps = len(positions) - 1
    step_lengths = np.zeros(n_steps, dtype=np.float64)

    if distance_type == "euclidean":
        # Vectorized Euclidean distance computation
        displacements = np.diff(positions, axis=0)
        step_lengths = np.linalg.norm(displacements, axis=1)

    else:  # distance_type == "geodesic"
        # Map positions to bins
        assert env is not None  # Already checked above, but satisfy type checker
        trajectory_bins = env.bin_at(positions)

        # Optimization: Precompute geodesic distance matrix
        # This is O(V*E*log(V)) once vs O(n_steps * V*log(V)) per-step Dijkstra
        from neurospatial.ops.distance import geodesic_distance_matrix

        # Get full distance matrix (uses scipy.sparse.csgraph.shortest_path)
        dist_matrix = geodesic_distance_matrix(
            env.connectivity, env.n_bins, weight="distance"
        )

        # Look up distances from precomputed matrix
        for i in range(n_steps):
            bin_i = trajectory_bins[i]
            bin_j = trajectory_bins[i + 1]
            step_lengths[i] = dist_matrix[bin_i, bin_j]

    return step_lengths


def compute_home_range(
    trajectory_bins: NDArray[np.int_],
    *,
    percentile: float = 95.0,
) -> NDArray[np.int_]:
    """
    Compute home range as bins containing a percentile of time spent.

    The home range is defined as the set of bins that collectively contain
    a specified percentile of the total time spent in the environment. This
    is analogous to the minimum convex polygon (MCP) method in ecology, but
    adapted for discretized spatial environments.

    Parameters
    ----------
    trajectory_bins : NDArray[np.int_], shape (n_samples,)
        Sequence of bin indices representing the trajectory.
    percentile : float, default=95.0
        Percentile of time to include in home range (0 to 100). Common values:
        - 50%: core area (most frequently used)
        - 95%: standard home range
        - 100%: all visited bins

    Returns
    -------
    NDArray[np.int_], shape (n_bins_in_range,)
        Bin indices in the home range, sorted by occupancy (most visited first).

    Notes
    -----
    The home range is computed by:
    1. Computing occupancy (visit counts) for each bin
    2. Sorting bins by occupancy (descending)
    3. Selecting bins until cumulative occupancy reaches the percentile threshold

    This is related to the utilization distribution (UD) in movement ecology,
    where the UD is estimated from occupancy and the home range is defined
    as the isopleth containing a specified probability mass.

    See Also
    --------
    neurospatial.Environment.occupancy : Compute time-weighted occupancy

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.behavior.trajectory import compute_home_range
    >>> # Trajectory with known distribution
    >>> trajectory_bins = np.concatenate(
    ...     [
    ...         np.repeat(0, 50),  # Bin 0: 50 visits
    ...         np.repeat(1, 30),  # Bin 1: 30 visits
    ...         np.repeat(2, 15),  # Bin 2: 15 visits
    ...         np.repeat(3, 5),  # Bin 3: 5 visits
    ...     ]
    ... )
    >>> home_range = compute_home_range(trajectory_bins, percentile=95.0)
    >>> set(home_range) == {0, 1, 2}  # 95% includes bins 0, 1, 2
    True

    References
    ----------
    .. [1] Worton, B. J. (1989). "Kernel methods for estimating the utilization
           distribution in home-range studies." Ecology, 70(1), 164-168.
    .. [2] Börger, L., et al. (2006). "Effects of sampling regime on the mean
           and variance of home range size estimates." Journal of Animal
           Ecology, 75(6), 1393-1405.
    """
    # Compute occupancy (visit counts)
    unique_bins, counts = np.unique(trajectory_bins, return_counts=True)

    # Sort bins by occupancy (descending)
    sort_idx = np.argsort(counts)[::-1]
    sorted_bins = unique_bins[sort_idx]
    sorted_counts = counts[sort_idx]

    # Compute cumulative percentage
    total_counts = np.sum(sorted_counts)
    cumulative_pct = np.cumsum(sorted_counts) / total_counts * 100.0

    # Find bins to include (cumulative percentage >= threshold)
    # searchsorted with side='left' gives first index where cumulative_pct >= percentile
    # We want bins 0 through that index (inclusive), so add 1 for Python slicing
    n_bins_include = np.searchsorted(cumulative_pct, percentile, side="left") + 1
    n_bins_include = min(n_bins_include, len(sorted_bins))

    # Return home range bins (sorted by occupancy)
    home_range = sorted_bins[:n_bins_include]

    return home_range.astype(np.intp)


def mean_square_displacement(
    positions: NDArray[np.float64],
    times: NDArray[np.float64],
    *,
    distance_type: Literal["euclidean", "geodesic"] = "euclidean",
    env: Environment | None = None,
    max_tau: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute mean square displacement (MSD) from continuous trajectory positions.

    Mean square displacement quantifies how spatial displacement grows with time
    lag, which is critical for characterizing diffusive motion and classifying
    movement patterns (ballistic, diffusive, subdiffusive, confined). This function
    operates on continuous position data for maximum accuracy.

    Parameters
    ----------
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Trajectory positions in continuous space.
    times : NDArray[np.float64], shape (n_samples,)
        Timestamps corresponding to each sample in the trajectory.
    distance_type : {"euclidean", "geodesic"}, default="euclidean"
        Distance metric for computing displacements:
        - "euclidean": Straight-line distance (ecology standard, most accurate).
          Use for diffusion analysis, comparing to ecology literature, and
          classification of movement types (ballistic vs. diffusive).
        - "geodesic": Graph shortest-path distance (neuroscience-specific).
          Requires `env` parameter. Use for constrained navigation analysis.
    env : Environment, optional
        Environment instance for computing geodesic distances. Required if
        distance_type="geodesic", ignored otherwise.
    max_tau : float, optional
        Maximum lag time to compute. If None, uses half the total duration.
        Recommended: use max_tau ≤ T/4 where T is total duration, to ensure
        sufficient samples for reliable averaging.

    Returns
    -------
    tau_values : NDArray[np.float64], shape (n_lags,)
        Lag times at which MSD was computed.
    msd_values : NDArray[np.float64], shape (n_lags,)
        Mean square displacement values at each lag time.

    Raises
    ------
    ValueError
        If distance_type="geodesic" but env is None.
        If positions and times have different lengths.
        If positions is not a 2D array.

    Notes
    -----
    The mean square displacement is defined as:

    .. math::

        MSD(\\tau) = \\langle |r(t + \\tau) - r(t)|^2 \\rangle_t

    where :math:`r(t)` is the position at time t, and the angle brackets denote
    averaging over all time points t.

    **Movement classification from scaling exponent**:

    MSD scales with time lag τ as :math:`MSD \\sim \\tau^\\alpha`:

    - **α ≈ 2**: Ballistic motion (directed, constant velocity)
    - **α ≈ 1**: Diffusive motion (random walk, Brownian motion)
    - **α < 1**: Subdiffusive motion (obstructed, confined)
    - **α > 2**: Superdiffusive motion (Lévy flight)
    - **α ≈ 0**: Confined motion (bounded, plateau)

    The exponent α can be estimated from log-log regression: log(MSD) ~ α·log(τ).

    **Why continuous positions?**

    Using continuous positions (not bins) is critical for MSD accuracy because:

    1. **Precision**: Bin discretization introduces a "floor" on displacements
       equal to the bin size, artificially inflating MSD for small displacements.
    2. **Scaling exponent**: Discretization can bias the diffusion exponent α,
       making diffusive motion appear more subdiffusive.
    3. **Ecology standard**: All movement ecology literature uses continuous MSD
       (Saxton 1997, Ferrari 2001, Kepten 2015).

    Example: Movement [0,0]→[0.1,0.1]→[0.2,0.2] maps to the same bin, giving
    MSD=0 (incorrect), whereas continuous gives MSD ∝ t (correct diffusive).

    See Also
    --------
    compute_turn_angles : Analyze directional changes
    compute_step_lengths : Compute step distances

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.behavior.trajectory import mean_square_displacement
    >>>
    >>> # Random walk trajectory (diffusive motion, α ≈ 1)
    >>> np.random.seed(42)
    >>> n_steps = 100
    >>> steps = np.random.randn(n_steps, 2) * 5
    >>> positions = np.cumsum(steps, axis=0)
    >>> times = np.arange(n_steps) * 0.1
    >>>
    >>> # Compute MSD with Euclidean distance (default)
    >>> tau_values, msd_values = mean_square_displacement(
    ...     positions, times, distance_type="euclidean", max_tau=5.0
    ... )
    >>> len(tau_values) > 0
    True
    >>> bool(msd_values[-1] > msd_values[0])  # MSD increases with lag
    True
    >>>
    >>> # Estimate diffusion exponent from log-log fit
    >>> log_tau = np.log(tau_values)
    >>> log_msd = np.log(msd_values + 1e-10)  # Add small constant to avoid log(0)
    >>> alpha = np.polyfit(log_tau, log_msd, 1)[0]  # doctest: +SKIP
    >>> print(f"Diffusion exponent: {alpha:.2f}")  # doctest: +SKIP
    Diffusion exponent: 1.02  # ≈ 1.0 (diffusive motion)

    References
    ----------
    .. [1] Saxton, M. J. (1997). "Single-particle tracking: the distribution
           of diffusion coefficients." Biophysical Journal, 72(4), 1744-1753.
    .. [2] Ferrari, R., et al. (2001). "Tracking single particles in biological
           systems." Physical Review E, 63(4), 041904.
    .. [3] Kepten, E., et al. (2015). "Improved estimation of anomalous
           diffusion exponents in single-particle tracking experiments."
           Physical Review E, 87(5), 052713.
    .. [4] Traja documentation: https://traja.readthedocs.io/
    """
    # Input validation
    if positions.ndim != 2:
        raise ValueError(
            f"positions must be 2D array (n_samples, n_dims), got {positions.ndim}D"
        )

    if len(positions) != len(times):
        raise ValueError(
            f"positions and times must have same length, "
            f"got {len(positions)} and {len(times)}"
        )

    if distance_type not in ("euclidean", "geodesic"):
        raise ValueError(
            f"distance_type must be 'euclidean' or 'geodesic', got '{distance_type}'"
        )

    if distance_type == "geodesic" and env is None:
        raise ValueError(
            "distance_type='geodesic' requires env parameter. "
            "Use distance_type='euclidean' if env is not available."
        )

    n_samples = len(positions)

    # Determine lag times to compute
    if max_tau is None:
        max_tau = (times[-1] - times[0]) / 2.0

    # Find all unique time differences up to max_tau
    time_diffs = np.diff(times)
    # Create lag times by accumulating time differences
    # We'll use a simple approach: sample lag times at regular intervals
    # from the time step up to max_tau
    dt = np.median(time_diffs)  # Typical time step
    n_lags = int(max_tau / dt)
    n_lags = max(1, min(n_lags, n_samples // 2))  # At least 1, at most n_samples/2

    tau_values = np.linspace(dt, max_tau, n_lags)
    msd_values = np.zeros(n_lags, dtype=np.float64)

    if distance_type == "euclidean":
        # Vectorized Euclidean MSD computation (fast)
        for i, tau in enumerate(tau_values):
            # Find all pairs of time points separated by approximately tau
            # For each starting time t_idx, find future indices at t + tau
            squared_displacements = []

            for t_idx in range(n_samples):
                t = times[t_idx]
                # Find indices where times are approximately t + tau
                future_idx = np.where(np.abs(times - (t + tau)) < dt / 2)[0]

                if len(future_idx) > 0:
                    # Compute squared Euclidean distance to all matching future positions
                    pos_current = positions[t_idx]
                    pos_future = positions[future_idx]
                    # Vectorized distance computation
                    displacements = pos_future - pos_current
                    sq_dists = np.sum(displacements**2, axis=1)
                    squared_displacements.extend(sq_dists)

            # Average squared displacements
            if len(squared_displacements) > 0:
                msd_values[i] = np.mean(squared_displacements)
            else:
                msd_values[i] = 0.0

    else:  # distance_type == "geodesic"
        # Map positions to bins for geodesic distance
        assert env is not None  # Already checked above
        trajectory_bins = env.bin_at(positions)

        # Optimization: Precompute geodesic distance matrix ONCE
        # This is O(V*E*log(V)) once vs O(n_lags * n_samples * V*log(V)) per lookup
        from neurospatial.ops.distance import geodesic_distance_matrix

        dist_matrix = geodesic_distance_matrix(
            env.connectivity, env.n_bins, weight="distance"
        )

        # Compute MSD for each lag time using precomputed distances
        for i, tau in enumerate(tau_values):
            # Find all pairs of time points separated by approximately tau
            squared_displacements = []

            for t_idx in range(n_samples):
                t = times[t_idx]
                # Find indices where times are approximately t + tau
                future_idx = np.where(np.abs(times - (t + tau)) < dt / 2)[0]

                for f_idx in future_idx:
                    if f_idx < n_samples:
                        # Look up distance from precomputed matrix
                        bin_i = trajectory_bins[t_idx]
                        bin_j = trajectory_bins[f_idx]
                        distance = dist_matrix[bin_i, bin_j]

                        # Skip disconnected bins (inf distance)
                        if np.isinf(distance):
                            continue

                        squared_displacements.append(distance**2)

            # Average squared displacements
            if len(squared_displacements) > 0:
                msd_values[i] = np.mean(squared_displacements)
            else:
                msd_values[i] = 0.0

    return tau_values, msd_values


def compute_trajectory_curvature(
    trajectory_positions: NDArray[np.float64],
    times: NDArray[np.float64] | None = None,
    *,
    smooth_window: float | None = 0.2,
) -> NDArray[np.float64]:
    """Compute trajectory curvature from position data.

    Works for any dimensionality (1D, 2D, 3D, N-D). Computes signed
    angle between consecutive movement direction vectors.

    Parameters
    ----------
    trajectory_positions : NDArray[np.float64], shape (n_samples, n_dims)
        Position coordinates over time in any dimensional space.
    times : NDArray[np.float64], shape (n_samples,), optional
        Timestamps for temporal smoothing. If None, assumes uniform sampling.
    smooth_window : float, optional
        Temporal smoothing window in seconds. Default: 0.2s (typical for 30-60 Hz
        tracking data).

        **Important**: For high-speed tracking (120+ Hz) or fast-moving animals,
        use shorter windows (0.05-0.1s) or disable smoothing (smooth_window=None)
        to preserve rapid turns.

        Set to None for no smoothing.

    Returns
    -------
    NDArray[np.float64], shape (n_samples,)
        Heading change at each timepoint in radians:
        - Positive values: counterclockwise turn (left in 2D top-down view)
        - Negative values: clockwise turn (right in 2D top-down view)
        - Zero: straight movement

    Notes
    -----
    This function wraps `compute_turn_angles()` and adds:
    - Padding to match input length (n_samples)
    - Optional temporal smoothing

    `compute_turn_angles()` uses atan2(cross, dot) for proper signed angles in [-π, π],
    filters stationary periods automatically, and returns length (n_samples - 2).

    **Padding Strategy**: The returned curvature array matches input length by symmetric
    padding with zeros. Since `compute_turn_angles()` filters stationary periods, the
    actual number of angles may be less than (n_samples - 2). The remaining positions
    are padded equally at start and end to maintain temporal centering.

    For N-D trajectories where N > 2, only the first 2 dimensions are used
    (consistent with `compute_turn_angles()` implementation).

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.behavior.trajectory import compute_trajectory_curvature
    >>>
    >>> # 2D trajectory on any environment (grid, graph, continuous)
    >>> trajectory_positions = np.column_stack([np.linspace(0, 100, 20), np.zeros(20)])
    >>> curvature = compute_trajectory_curvature(trajectory_positions)
    >>>
    >>> # Detect sharp turns (> 45 degrees)
    >>> sharp_left = np.where(curvature > np.pi / 4)[0]
    >>> sharp_right = np.where(curvature < -np.pi / 4)[0]
    >>>
    >>> # 3D trajectory (e.g., climbing, flying)
    >>> trajectory_3d = np.column_stack(
    ...     [np.linspace(0, 100, 20), np.zeros(20), np.linspace(0, 10, 20)]
    ... )
    >>> curvature_3d = compute_trajectory_curvature(trajectory_3d)
    >>>
    >>> # Smooth for noisy tracking data
    >>> times = np.linspace(0, 10, 20)
    >>> curvature_smooth = compute_trajectory_curvature(
    ...     trajectory_positions, times, smooth_window=0.5
    ... )

    See Also
    --------
    compute_turn_angles : Raw turn angles without padding
    """
    # 1. Compute turn angles using existing function
    # Returns length (n_angles,) where n_angles <= n_samples - 2
    # Filters stationary periods automatically
    angles = compute_turn_angles(trajectory_positions)

    # 2. Pad to match input length (n_samples)
    # compute_turn_angles returns variable length due to duplicate filtering
    # Pad with 0 at start and end to reach n_samples
    n_samples = len(trajectory_positions)
    n_angles = len(angles)

    # Calculate padding needed
    if n_angles == 0:
        # Edge case: < 3 unique positions
        curvature = np.zeros(n_samples, dtype=np.float64)
    else:
        # Pad symmetrically: add (n_samples - n_angles) / 2 to each side
        # If uneven, add extra to the end
        pad_total = n_samples - n_angles
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        curvature = np.pad(
            angles, (pad_left, pad_right), mode="constant", constant_values=0.0
        )

    # 3. Optional temporal smoothing
    if smooth_window is not None and times is not None:
        from scipy.ndimage import gaussian_filter1d

        # Compute sigma from time resolution
        dt_median = np.median(np.diff(times))
        sigma = smooth_window / dt_median

        # Apply Gaussian smoothing
        curvature = gaussian_filter1d(curvature, sigma=sigma)

    return curvature
