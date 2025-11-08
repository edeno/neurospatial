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

from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from neurospatial.environment.core import Environment


def compute_turn_angles(
    trajectory_bins: NDArray[np.int_],
    env: Environment,
) -> NDArray[np.float64]:
    """
    Compute turn angles between consecutive movement vectors.

    Turn angles quantify changes in movement direction at each position along
    the trajectory. A turn angle of 0 indicates straight movement, positive
    angles indicate left turns, and negative angles indicate right turns.

    Parameters
    ----------
    trajectory_bins : NDArray[np.int_], shape (n_samples,)
        Sequence of bin indices representing the trajectory.
    env : Environment
        Environment instance containing bin positions.

    Returns
    -------
    NDArray[np.float64], shape (n_angles,)
        Turn angles in radians, in the range [-π, π]. Length is the number
        of non-stationary transitions minus 1. Consecutive duplicate bins
        (stationary periods) are filtered out before computing angles.

    Notes
    -----
    Turn angles are computed from the angle between consecutive movement vectors:

    .. math::

        \\theta_i = \\text{atan2}(v_{i+1} \\times v_i, v_{i+1} \\cdot v_i)

    where :math:`v_i` is the movement vector from bin i-1 to bin i.

    For 2D trajectories, this uses the 2D cross product (scalar). For higher
    dimensions, the function uses the first two dimensions only.

    Stationary periods (consecutive duplicate bins) are removed before computing
    turn angles to avoid undefined angles from zero-length vectors.

    See Also
    --------
    compute_step_lengths : Compute distances between consecutive positions
    mean_square_displacement : Analyze diffusive properties of trajectory

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.metrics.trajectory import compute_turn_angles
    >>> # Create straight line trajectory
    >>> positions = np.linspace(0, 100, 20)[:, None]
    >>> env = Environment.from_samples(positions, bin_size=5.0)
    >>> trajectory_bins = np.arange(10)
    >>> angles = compute_turn_angles(trajectory_bins, env)
    >>> np.allclose(angles, 0.0, atol=0.1)  # Straight = no turning
    True

    References
    ----------
    .. [1] Batschelet, E. (1981). Circular Statistics in Biology.
           Academic Press.
    .. [2] Muller, M., & Wehner, R. (1988). "Path integration in desert ants."
           PNAS, 85(14), 5287-5290.
    """
    # Remove consecutive duplicates (stationary periods) using vectorized operations
    mask = np.concatenate([[True], trajectory_bins[1:] != trajectory_bins[:-1]])
    unique_bins = trajectory_bins[mask]

    # Need at least 3 unique positions to compute turn angles
    if len(unique_bins) < 3:
        return np.array([], dtype=np.float64)

    # Get bin centers
    positions = env.bin_centers[unique_bins]

    # Compute movement vectors (differences between consecutive positions)
    vectors = np.diff(positions, axis=0)  # shape (n_unique-1, n_dims)

    # Compute turn angles from consecutive vectors
    # For each pair of consecutive vectors, compute the angle between them
    n_angles = len(vectors) - 1
    angles = np.zeros(n_angles, dtype=np.float64)

    for i in range(n_angles):
        v1 = vectors[i]
        v2 = vectors[i + 1]

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
    trajectory_bins: NDArray[np.int_],
    env: Environment,
) -> NDArray[np.float64]:
    """
    Compute step lengths (distances) between consecutive positions.

    Step lengths quantify the spatial displacement at each step of the trajectory.
    Distances are computed using the graph geodesic distances, accounting for
    connectivity constraints.

    Parameters
    ----------
    trajectory_bins : NDArray[np.int_], shape (n_samples,)
        Sequence of bin indices representing the trajectory.
    env : Environment
        Environment instance for computing distances between bins.

    Returns
    -------
    NDArray[np.float64], shape (n_samples - 1,)
        Step lengths in the same units as the environment. Consecutive duplicate
        bins have step length 0.

    Notes
    -----
    Step lengths are computed using graph geodesic distances via
    `env.distance_between()`, which accounts for the connectivity structure
    of the environment. For regular grids with 8-connectivity, diagonal steps
    are longer than cardinal steps by a factor of √2.

    See Also
    --------
    compute_turn_angles : Compute turn angles between movements
    mean_square_displacement : Analyze diffusive properties

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.metrics.trajectory import compute_step_lengths
    >>> # Create 1D trajectory
    >>> positions = np.linspace(0, 100, 21)[:, None]
    >>> env = Environment.from_samples(positions, bin_size=5.0)
    >>> trajectory_bins = np.arange(10)
    >>> step_lengths = compute_step_lengths(trajectory_bins, env)
    >>> len(step_lengths)
    9
    >>> np.allclose(step_lengths, step_lengths[0], rtol=0.1)  # Uniform steps
    True

    References
    ----------
    .. [1] Kays, R., et al. (2015). "Terrestrial animal tracking as an eye
           on life and planet." Science, 348(6240), aaa2478.
    """
    n_steps = len(trajectory_bins) - 1
    step_lengths = np.zeros(n_steps, dtype=np.float64)

    for i in range(n_steps):
        bin_i = trajectory_bins[i]
        bin_j = trajectory_bins[i + 1]

        # Handle same bin (stationary) case
        if bin_i == bin_j:
            step_lengths[i] = 0.0
        else:
            # Compute graph geodesic distance between bins
            try:
                step_lengths[i] = float(
                    nx.shortest_path_length(
                        env.connectivity,
                        source=int(bin_i),
                        target=int(bin_j),
                        weight="distance",
                    )
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                # Bins are disconnected - distance is infinite
                step_lengths[i] = np.inf

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
    >>> from neurospatial.metrics.trajectory import compute_home_range
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
    trajectory_bins: NDArray[np.int_],
    times: NDArray[np.float64],
    env: Environment,
    *,
    max_tau: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute mean square displacement (MSD) as a function of lag time.

    Mean square displacement quantifies how the spatial displacement grows
    with time lag, which is useful for characterizing diffusive motion and
    classifying movement patterns (e.g., random walk, directed motion, confined).

    Parameters
    ----------
    trajectory_bins : NDArray[np.int_], shape (n_samples,)
        Sequence of bin indices representing the trajectory.
    times : NDArray[np.float64], shape (n_samples,)
        Timestamps corresponding to each sample in the trajectory.
    env : Environment
        Environment instance for computing distances between bins.
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

    Notes
    -----
    The mean square displacement is defined as:

    .. math::

        MSD(\\tau) = \\langle |r(t + \\tau) - r(t)|^2 \\rangle_t

    where :math:`r(t)` is the position at time t, and the angle brackets denote
    averaging over all time points t.

    MSD scales with time lag τ according to the movement pattern:
    - **Ballistic motion**: :math:`MSD \\sim \\tau^2` (directed, constant velocity)
    - **Diffusive motion**: :math:`MSD \\sim \\tau^1` (random walk)
    - **Confined motion**: :math:`MSD \\sim \\tau^0` (bounded, plateau)

    The exponent α in :math:`MSD \\sim \\tau^\\alpha` can be estimated from
    log-log regression to classify movement types.

    See Also
    --------
    compute_turn_angles : Analyze directional changes
    compute_step_lengths : Compute step distances

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.metrics.trajectory import mean_square_displacement
    >>> # Create random walk trajectory (diffusive motion)
    >>> np.random.seed(42)
    >>> n_steps = 100
    >>> steps = np.random.randn(n_steps, 2) * 5
    >>> trajectory = np.cumsum(steps, axis=0) + 50
    >>> env = Environment.from_samples(trajectory, bin_size=3.0)
    >>> trajectory_bins = env.bin_at(trajectory)
    >>> times = np.arange(n_steps) * 0.1
    >>> tau_values, msd_values = mean_square_displacement(
    ...     trajectory_bins, times, env, max_tau=5.0
    ... )
    >>> len(tau_values) > 0
    True
    >>> msd_values[-1] > msd_values[0]  # MSD increases with lag
    True

    References
    ----------
    .. [1] Saxton, M. J. (1997). "Single-particle tracking: the distribution
           of diffusion coefficients." Biophysical Journal, 72(4), 1744-1753.
    .. [2] Ferrari, R., et al. (2001). "Tracking single particles in biological
           systems." Physical Review E, 63(4), 041904.
    .. [3] Kepten, E., et al. (2015). "Improved estimation of anomalous
           diffusion exponents in single-particle tracking experiments."
           Physical Review E, 87(5), 052713.
    """
    n_samples = len(trajectory_bins)

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

    # Compute MSD for each lag time
    for i, tau in enumerate(tau_values):
        # Find all pairs of time points separated by approximately tau
        displacements = []

        for t_idx in range(n_samples):
            t = times[t_idx]
            # Find indices where times are approximately t + tau
            future_idx = np.where(np.abs(times - (t + tau)) < dt / 2)[0]

            for f_idx in future_idx:
                if f_idx < n_samples:
                    # Compute squared displacement
                    bin_i = trajectory_bins[t_idx]
                    bin_j = trajectory_bins[f_idx]

                    # Handle same bin case (stationary)
                    if bin_i == bin_j:
                        distance = 0.0
                    else:
                        # Compute graph geodesic distance between bins
                        try:
                            distance = float(
                                nx.shortest_path_length(
                                    env.connectivity,
                                    source=int(bin_i),
                                    target=int(bin_j),
                                    weight="distance",
                                )
                            )
                        except (nx.NetworkXNoPath, nx.NodeNotFound):
                            # Bins are disconnected - skip this pair
                            continue

                    displacements.append(distance**2)

        # Average squared displacements
        if len(displacements) > 0:
            msd_values[i] = np.mean(displacements)
        else:
            msd_values[i] = 0.0

    return tau_values, msd_values
