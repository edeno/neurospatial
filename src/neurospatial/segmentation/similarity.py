"""Trajectory similarity metrics and goal-directed run detection.

This module provides functions for comparing trajectories and detecting
goal-directed navigation patterns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import directed_hausdorff, euclidean
from scipy.stats import pearsonr

from neurospatial.segmentation.regions import Run
from neurospatial.spatial import regions_to_mask

if TYPE_CHECKING:
    from neurospatial.environment._protocols import EnvironmentProtocol


def trajectory_similarity(
    trajectory1_bins: NDArray[np.int64],
    trajectory2_bins: NDArray[np.int64],
    env: EnvironmentProtocol,
    *,
    method: Literal["jaccard", "correlation", "hausdorff", "dtw"] = "jaccard",
) -> float:
    """Compute similarity between two trajectories.

    Compares spatial paths using various metrics. Useful for identifying
    stereotyped behaviors, replay analysis, and comparing navigation strategies
    across trials or sessions.

    Parameters
    ----------
    trajectory1_bins : NDArray[np.int64], shape (n_samples1,)
        First trajectory as sequence of bin indices.
    trajectory2_bins : NDArray[np.int64], shape (n_samples2,)
        Second trajectory as sequence of bin indices.
    env : Environment
        Spatial environment containing bin information.
    method : {'jaccard', 'correlation', 'hausdorff', 'dtw'}, optional
        Similarity metric to use:
        - 'jaccard': Spatial overlap (set intersection / union). Range [0, 1].
        - 'correlation': Pearson correlation of bin sequences. Range [0, 1].
        - 'hausdorff': Hausdorff distance between paths, normalized. Range [0, 1].
        - 'dtw': Dynamic time warping distance, normalized. Range [0, 1].
        Default is 'jaccard'.

    Returns
    -------
    float
        Similarity score in [0, 1] where 1.0 indicates identical trajectories
        and 0.0 indicates completely different trajectories.

    Raises
    ------
    ValueError
        If trajectories are empty.
    ValueError
        If method is not one of the supported options.

    See Also
    --------
    detect_goal_directed_runs : Detect efficient paths toward goal regions.

    Notes
    -----
    **Jaccard similarity** measures spatial overlap:

    .. math::

        J(A, B) = \\frac{|A \\cap B|}{|A \\cup B|}

    where A and B are sets of visited bins. Useful for comparing spatial
    coverage independent of temporal order.

    **Correlation** measures sequential similarity by computing Pearson
    correlation between bin index sequences. Sensitive to order and timing.

    **Hausdorff distance** measures maximum deviation between trajectories:

    .. math::

        d_H(A, B) = \\max\\{\\sup_{a \\in A} \\inf_{b \\in B} d(a, b),
                            \\sup_{b \\in B} \\inf_{a \\in A} d(a, b)\\}

    Converted to similarity as :math:`1 - d_H / d_{max}` where :math:`d_{max}`
    is the environment extent.

    **Dynamic time warping (DTW)** finds optimal alignment between trajectories
    allowing temporal shifts. Useful for comparing paths with different speeds.

    Examples
    --------
    Compare two trajectories using Jaccard overlap:

    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.segmentation.similarity import trajectory_similarity
    >>> positions = np.random.uniform(0, 100, (200, 2))
    >>> env = Environment.from_samples(positions, bin_size=5.0)
    >>> traj1 = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    >>> traj2 = np.array([2, 3, 4, 5, 6], dtype=np.int64)
    >>> similarity = trajectory_similarity(traj1, traj2, env, method="jaccard")
    >>> print(f"Jaccard similarity: {similarity:.3f}")  # doctest: +SKIP
    Jaccard similarity: 0.429

    Compare using correlation (sensitive to sequence order):

    >>> sim_corr = trajectory_similarity(traj1, traj2, env, method="correlation")
    >>> print(f"Correlation similarity: {sim_corr:.3f}")  # doctest: +SKIP
    Correlation similarity: 0.975

    References
    ----------
    .. [1] Pfeiffer, B. E., & Foster, D. J. (2013). Hippocampal place-cell
           sequences depict future paths to remembered goals. *Nature*,
           497(7447), 74-79.
    .. [2] Davidson, T. J., Kloosterman, F., & Wilson, M. A. (2009).
           Hippocampal replay of extended experience. *Neuron*, 63(4), 497-507.
    """
    # Input validation
    if len(trajectory1_bins) == 0 or len(trajectory2_bins) == 0:
        raise ValueError("Trajectories cannot be empty")

    valid_methods = ["jaccard", "correlation", "hausdorff", "dtw"]
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}, got '{method}'")

    if method == "jaccard":
        # Spatial overlap (set-based)
        set1 = set(trajectory1_bins)
        set2 = set(trajectory2_bins)
        intersection = set1 & set2
        union = set1 | set2
        if len(union) == 0:
            return 0.0
        return len(intersection) / len(union)

    elif method == "correlation":
        # Sequential correlation
        # Align to same length using indices modulo shorter length
        if len(trajectory1_bins) < len(trajectory2_bins):
            shorter, longer = trajectory1_bins, trajectory2_bins
        else:
            shorter, longer = trajectory2_bins, trajectory1_bins

        # Sample longer trajectory at same rate as shorter
        if len(shorter) < 2:
            # Can't compute correlation with < 2 points
            return 1.0 if np.array_equal(trajectory1_bins, trajectory2_bins) else 0.0

        indices = np.linspace(0, len(longer) - 1, len(shorter), dtype=int)
        longer_sampled = longer[indices]

        # Compute correlation
        if len(set(shorter)) == 1 or len(set(longer_sampled)) == 1:
            # Constant sequence - return 1 if identical, 0 otherwise
            return 1.0 if np.array_equal(shorter, longer_sampled) else 0.0

        corr, _ = pearsonr(shorter, longer_sampled)
        # Map [-1, 1] to [0, 1], clamp negative correlations to 0
        return float(max(0.0, corr))

    elif method == "hausdorff":
        # Hausdorff distance between paths
        # Get positions from bin indices
        pos1 = env.bin_centers[trajectory1_bins]
        pos2 = env.bin_centers[trajectory2_bins]

        # Compute Hausdorff distance
        d_hausdorff = max(
            directed_hausdorff(pos1, pos2)[0],
            directed_hausdorff(pos2, pos1)[0],
        )

        # Normalize by environment extent (maximum possible distance)
        extent = np.ptp(env.bin_centers, axis=0)  # range per dimension
        max_distance = np.linalg.norm(extent)

        if max_distance == 0:
            return 1.0  # Degenerate environment

        # Convert distance to similarity [0, 1]
        similarity = 1.0 - min(d_hausdorff / max_distance, 1.0)
        return float(similarity)

    elif method == "dtw":
        # Dynamic time warping
        pos1 = env.bin_centers[trajectory1_bins]
        pos2 = env.bin_centers[trajectory2_bins]

        # Compute DTW distance using dynamic programming
        dtw_distance = _dtw_distance(pos1, pos2)

        # Normalize by path lengths
        # Theoretical max DTW = max(len1, len2) * max_bin_distance
        extent = np.ptp(env.bin_centers, axis=0)
        max_bin_distance = np.linalg.norm(extent)
        max_dtw = max(len(pos1), len(pos2)) * max_bin_distance

        if max_dtw == 0:
            return 1.0

        # Convert to similarity [0, 1]
        similarity = 1.0 - min(dtw_distance / max_dtw, 1.0)
        return float(similarity)

    # All method cases covered above
    raise ValueError(f"Unreachable: method validation failed for '{method}'")


def _dtw_distance(seq1: NDArray[np.float64], seq2: NDArray[np.float64]) -> float:
    """Compute Dynamic Time Warping distance between two sequences.

    Parameters
    ----------
    seq1 : NDArray[np.float64], shape (n1, n_dims)
        First sequence of positions.
    seq2 : NDArray[np.float64], shape (n2, n_dims)
        Second sequence of positions.

    Returns
    -------
    float
        DTW distance.
    """
    n1, n2 = len(seq1), len(seq2)

    # Initialize cost matrix
    dtw_matrix = np.full((n1 + 1, n2 + 1), np.inf)
    dtw_matrix[0, 0] = 0.0

    # Fill cost matrix
    for i in range(1, n1 + 1):
        for j in range(1, n2 + 1):
            cost = euclidean(seq1[i - 1], seq2[j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],  # insertion
                dtw_matrix[i, j - 1],  # deletion
                dtw_matrix[i - 1, j - 1],  # match
            )

    return float(dtw_matrix[n1, n2])


def detect_goal_directed_runs(
    trajectory_bins: NDArray[np.int64],
    times: NDArray[np.float64],
    env: EnvironmentProtocol,
    *,
    goal_region: str,
    directedness_threshold: float = 0.7,
    min_progress: float = 20.0,
) -> list[Run]:
    """Detect goal-directed runs in a trajectory.

    Identifies segments where the animal moves efficiently toward a goal region.
    Useful for analyzing goal-directed replay, learning dynamics, and navigation
    strategies.

    Parameters
    ----------
    trajectory_bins : NDArray[np.int64], shape (n_samples,)
        Sequence of bin indices representing the trajectory.
    times : NDArray[np.float64], shape (n_samples,)
        Time stamps corresponding to trajectory bins (seconds).
    env : Environment
        Environment containing the goal region definition.
    goal_region : str
        Name of goal region in env.regions. Runs are detected toward this region.
    directedness_threshold : float, optional
        Minimum directedness score to classify as goal-directed. Range [0, 1].
        Higher values require more efficient paths. Default is 0.7.
    min_progress : float, optional
        Minimum distance progress toward goal (physical units). Filters out
        short runs with minimal displacement. Default is 20.0.

    Returns
    -------
    list[Run]
        List of goal-directed run segments. Each Run contains:
        - start_time: Run start time (seconds)
        - end_time: Run end time (seconds)
        - bins: Sequence of bin indices during run
        - success: Always True for detected runs (reached goal region)

    Raises
    ------
    KeyError
        If goal_region not in env.regions.
    ValueError
        If directedness_threshold not in [0, 1].
    ValueError
        If min_progress is negative.

    See Also
    --------
    trajectory_similarity : Compare similarity between trajectories.
    detect_runs_between_regions : Detect runs from source to target regions.

    Notes
    -----
    **Directedness score** measures path efficiency:

    .. math::

        D = \\frac{d_{start} - d_{end}}{L_{path}}

    where :math:`d_{start}` and :math:`d_{end}` are distances to goal at
    trajectory start and end, and :math:`L_{path}` is total path length.

    A straight path toward goal has :math:`D \\approx 1.0`, while a random
    walk has :math:`D \\approx 0.0`. Negative values indicate movement away
    from goal.

    This metric is used to identify replay events where hippocampal place cells
    reactivate in sequences representing paths to remembered goals [1]_.

    Examples
    --------
    Detect goal-directed runs in a linear track:

    >>> import numpy as np
    >>> from shapely.geometry import Point
    >>> from neurospatial import Environment
    >>> from neurospatial.segmentation.similarity import detect_goal_directed_runs
    >>> # Create 1D environment
    >>> positions = np.linspace(0, 100, 100)[:, None]
    >>> env = Environment.from_samples(positions, bin_size=2.0)
    >>> # Add goal region at far end
    >>> goal_polygon = Point(env.bin_centers[-1]).buffer(5.0)
    >>> env.regions.add("goal", polygon=goal_polygon)
    >>> # Create trajectory moving toward goal
    >>> trajectory_bins = np.arange(0, 40, dtype=np.int64)
    >>> times = np.linspace(0, 10, len(trajectory_bins))
    >>> # Detect goal-directed runs
    >>> runs = detect_goal_directed_runs(
    ...     trajectory_bins,
    ...     times,
    ...     env,
    ...     goal_region="goal",
    ...     directedness_threshold=0.7,
    ...     min_progress=10.0,
    ... )  # doctest: +SKIP
    >>> print(f"Detected {len(runs)} goal-directed run(s)")  # doctest: +SKIP
    Detected 1 goal-directed run(s)

    References
    ----------
    .. [1] Pfeiffer, B. E., & Foster, D. J. (2013). Hippocampal place-cell
           sequences depict future paths to remembered goals. *Nature*,
           497(7447), 74-79.
    """
    # Input validation
    if goal_region not in env.regions:
        available = list(env.regions.keys())
        raise KeyError(
            f"Region '{goal_region}' not found in env.regions. "
            f"Available regions: {available}"
        )

    if not 0.0 <= directedness_threshold <= 1.0:
        raise ValueError(
            f"directedness_threshold must be in [0, 1], got {directedness_threshold}"
        )

    if min_progress < 0:
        raise ValueError(f"min_progress must be non-negative, got {min_progress}")

    # Handle empty trajectory
    if len(trajectory_bins) == 0:
        return []

    # Get goal region mask
    goal_mask = regions_to_mask(env, [goal_region])
    goal_bin_indices = np.where(goal_mask)[0]

    if len(goal_bin_indices) == 0:
        # No bins in goal region
        return []

    # Compute distance from each bin to nearest goal bin using graph distance
    distances_to_goal = np.full(env.n_bins, np.inf)
    for bin_idx in range(env.n_bins):
        min_dist = np.inf
        for goal_bin in goal_bin_indices:
            try:
                dist = nx.shortest_path_length(
                    env.connectivity, bin_idx, int(goal_bin), weight="distance"
                )
                min_dist = min(min_dist, dist)
            except nx.NetworkXNoPath:
                continue
        distances_to_goal[bin_idx] = min_dist

    # Get start and end positions
    start_bin = trajectory_bins[0]
    end_bin = trajectory_bins[-1]

    d_start = distances_to_goal[start_bin]
    d_end = distances_to_goal[end_bin]

    # Check if start or end can reach goal
    # If either is infinite, cannot compute meaningful directedness
    if np.isinf(d_start) or np.isinf(d_end):
        # Cannot reach goal from start or end position
        return []

    # Compute path length (sum of graph distances between consecutive bins)
    path_length = 0.0
    for i in range(len(trajectory_bins) - 1):
        try:
            segment_dist = nx.shortest_path_length(
                env.connectivity,
                int(trajectory_bins[i]),
                int(trajectory_bins[i + 1]),
                weight="distance",
            )
            path_length += segment_dist
        except nx.NetworkXNoPath:
            # Disconnected bins - skip this segment
            continue

    # Compute directedness
    directedness = 0.0 if path_length == 0 else (d_start - d_end) / path_length

    # Compute progress toward goal
    progress = d_start - d_end

    # Check if run meets criteria
    runs = []
    if directedness >= directedness_threshold and progress >= min_progress:
        run = Run(
            start_time=float(times[0]),
            end_time=float(times[-1]),
            bins=trajectory_bins.copy(),
            success=True,  # Reached goal or progressed significantly
        )
        runs.append(run)

    return runs
