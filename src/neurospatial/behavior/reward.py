"""Reward field generation for reinforcement learning applications.

This module provides functions for creating reward fields from spatial regions
and goal locations. These are essential primitives for reinforcement learning
(RL) and reward shaping in spatial navigation tasks.

Imports
-------
>>> from neurospatial.behavior.reward import goal_reward_field, region_reward_field

Or via behavior package:

>>> from neurospatial.behavior import goal_reward_field, region_reward_field
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from neurospatial.environment import Environment

from neurospatial.ops.distance import distance_field
from neurospatial.ops.mapping import _require_region_mask

__all__ = [
    "goal_reward_field",
    "region_reward_field",
]


def region_reward_field(
    env: Environment,
    *,
    region_name: str,
    reward_value: float = 1.0,
    decay: Literal["constant", "linear", "gaussian"] = "constant",
    bandwidth: float | None = None,
) -> NDArray[np.float64]:
    """Generate reward field from a named region.

    Creates a spatial reward field based on a region defined in the environment.
    Supports different decay profiles for reward shaping in reinforcement learning.

    Parameters
    ----------
    env : Environment
        Environment with regions defined.
    region_name : str
        Name of region in env.regions to use as reward source.
    reward_value : float, default=1.0
        Maximum reward value in the region.
    decay : {'constant', 'linear', 'gaussian'}, default='constant'
        Reward decay profile:
        - 'constant': Binary reward (reward_value inside, 0 outside)
        - 'linear': Linear decay from region boundary
        - 'gaussian': Smooth Gaussian falloff from region
    bandwidth : float, optional
        Gaussian smoothing bandwidth (required for decay='gaussian').
        Should be in same units as environment coordinates.

    Returns
    -------
    reward : NDArray[np.float64], shape (n_bins,)
        Reward field with specified decay profile.

    Raises
    ------
    KeyError
        If region_name is not found in env.regions.
    ValueError
        If bandwidth is not provided for decay='gaussian'.

    Notes
    -----
    **Decay Types**:

    - **Constant**: Binary reward useful for sparse reward RL tasks.
      Creates clear goal regions with no reward shaping.

    - **Linear**: Linear decay from region boundary. Provides gradient
      information while maintaining clear boundaries. Distance computed
      using graph connectivity (respects environment structure) and
      normalized by the farthest bin that can reach the region. Bins that
      cannot reach the region (another connected component) get 0.

    - **Gaussian**: Smooth falloff using Gaussian kernel smoothing.
      After smoothing, rescales so maximum *within the region* equals
      reward_value. This preserves the intended reward magnitude while
      providing smooth gradients for policy learning.

    **Reward Shaping**: These primitives implement potential-based reward
    shaping (Ng et al., 1999). Gaussian decay provides the smoothest
    gradients but may inadvertently guide agents away from optimal paths.
    Use with caution and validate against sparse reward baseline.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.behavior.reward import region_reward_field
    >>> from shapely.geometry import Point
    >>> # Create environment
    >>> positions = np.random.default_rng(0).normal(0, 50, (1000, 2))
    >>> env = Environment.from_samples(positions, bin_size=5.0)
    >>> # Add goal region (use polygon, not point)
    >>> goal_polygon = Point([0.0, 0.0]).buffer(10.0)
    >>> from neurospatial.regions import Regions
    >>> regions = Regions(env.regions.values())
    >>> _ = regions.add("goal", polygon=goal_polygon)
    >>> env = env.with_regions(regions)
    >>> # Binary reward (sparse RL)
    >>> reward_binary = region_reward_field(env, region_name="goal", decay="constant")
    >>> # Smooth Gaussian reward shaping
    >>> reward_smooth = region_reward_field(
    ...     env, region_name="goal", decay="gaussian", bandwidth=10.0
    ... )

    See Also
    --------
    goal_reward_field : Distance-based rewards from goal bins
    neurospatial.distance_field : Compute distance maps
    neurospatial.spatial.regions_to_mask : Convert regions to binary masks

    References
    ----------
    .. [1] Ng, A. Y., Harada, D., & Russell, S. (1999). Policy invariance
           under reward transformations: Theory and application to reward
           shaping. ICML.
    """
    # Validate region exists
    region_mask = _require_region_mask(
        env, region_name, operation="region_reward_field()", argument="region_name"
    )

    if decay == "constant":
        # Binary reward: reward_value inside, 0 outside
        reward = np.where(region_mask, reward_value, 0.0)

    elif decay == "linear":
        # Geodesic distance to the nearest region bin: 0 inside the region,
        # inf for bins that cannot reach it. A shortest path into the region
        # enters through a boundary bin, so this is the distance from the
        # region boundary.
        distances = distance_field(
            env.connectivity, sources=np.flatnonzero(region_mask)
        )
        reachable = np.isfinite(distances) & ~region_mask
        reward = np.where(region_mask, reward_value, 0.0)
        if reachable.any():
            decay_length = distances[reachable].max()
            reward[reachable] = reward_value * np.maximum(
                0.0, 1.0 - distances[reachable] / decay_length
            )

    elif decay == "gaussian":
        # Gaussian falloff using smoothing
        if bandwidth is None:
            raise ValueError(
                "bandwidth parameter is required for decay='gaussian'. "
                "Provide a smoothing bandwidth in the same units as your environment."
            )

        # Create indicator field (1 inside region, 0 outside)
        indicator = np.where(region_mask, 1.0, 0.0)

        # Smooth the indicator field. env.smooth is a linear operator, so under
        # truncation it can leave tolerance-level negatives at far bins; a reward
        # field is non-negative, so floor it (the dense clipped kernel did this
        # implicitly). Positivity is the consumer's job for the linear smoother.
        smoothed = np.maximum(
            np.asarray(env.smooth(indicator, bandwidth=bandwidth, mode="average")),
            0.0,
        )

        # CRITICAL FIX: Scale by max IN REGION (not global max)
        # This ensures the peak reward within the actual region equals reward_value
        max_in_region = smoothed[region_mask].max()

        if max_in_region > 0:
            reward = (smoothed / max_in_region) * reward_value
        else:
            # Edge case: region has no bins or smoothing failed
            reward = np.zeros(env.n_bins, dtype=np.float64)

    else:
        raise ValueError(
            f"Invalid decay type: {decay}. "
            f"Must be one of: 'constant', 'linear', 'gaussian'"
        )

    return reward


def goal_reward_field(
    env: Environment,
    goal_bins: int | list[int] | NDArray[np.int_],
    *,
    decay: Literal["linear", "exponential", "inverse"] = "exponential",
    scale: float = 1.0,
    max_distance: float | None = None,
) -> NDArray[np.float64]:
    """Generate distance-based reward field from goal bins.

    Creates a reward field that decays with distance from goal locations.
    Common for goal-directed navigation tasks and reward shaping in RL.

    Parameters
    ----------
    env : Environment
        Spatial environment.
    goal_bins : int or list[int] or NDArray[np.int_]
        Goal bin index or indices. Can be scalar or array.
    decay : {'linear', 'exponential', 'inverse'}, default='exponential'
        Distance-based decay function:
        - 'linear': reward = scale * max(0, 1 - d/max_distance)
        - 'exponential': reward = scale * exp(-d/scale)
        - 'inverse': reward = scale / (1 + d)
    scale : float, default=1.0
        Reward scale parameter. Interpretation depends on decay type:
        - 'linear': maximum reward at goal
        - 'exponential': reward at goal (also decay rate)
        - 'inverse': reward scale factor
    max_distance : float, optional
        Maximum distance for linear decay. Required for decay='linear'.
        Reward is zero beyond this distance.

    Returns
    -------
    reward : NDArray[np.float64], shape (n_bins,)
        Distance-based reward field.

    Raises
    ------
    ValueError
        If scale <= 0 for decay='exponential'.
        If max_distance not provided for decay='linear'.
        If goal_bins contains invalid indices.

    Notes
    -----
    **Decay Functions**:

    - **Exponential**: Most common in RL literature. Provides smooth gradients
      that decay exponentially with distance. Scale parameter controls both
      peak reward and decay rate. Larger scale = slower decay.

    - **Linear**: Reaches exactly zero at max_distance. Useful when reward
      should be strictly local. Provides constant gradient within range.

    - **Inverse**: Inverse distance function (1/(1+d)). Never reaches zero,
      providing global gradients. Can lead to suboptimal policies if not
      carefully tuned.

    **Multiple Goals**: When goal_bins contains multiple indices, distance
    is computed to the *nearest* goal. This creates a Voronoi-like partition
    where each bin is influenced by its closest goal.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.behavior.reward import goal_reward_field
    >>> # Create environment with fixed seed for reproducibility
    >>> rng = np.random.default_rng(42)
    >>> positions = rng.standard_normal((1000, 2)) * 50
    >>> env = Environment.from_samples(positions, bin_size=5.0)
    >>> # Select goal bin at center of environment
    >>> goal_bin = len(env.bin_centers) // 2
    >>> # Exponential decay (most common)
    >>> reward = goal_reward_field(
    ...     env, goal_bins=goal_bin, decay="exponential", scale=10.0
    ... )
    >>> # Linear decay with cutoff
    >>> reward_linear = goal_reward_field(
    ...     env, goal_bins=goal_bin, decay="linear", scale=1.0, max_distance=50.0
    ... )

    See Also
    --------
    region_reward_field : Reward fields from spatial regions
    neurospatial.distance_field : Compute distance maps
    """
    from neurospatial.environment import _validate_target_bins

    goal_bins_list = _validate_target_bins(
        goal_bins,
        n_bins=env.n_bins,
        coordinates=env.coordinate_system,
        operation="goal_reward_field()",
        argument="goal_bins",
    ).tolist()

    # Compute distance field from goals
    distances = distance_field(env.connectivity, sources=goal_bins_list)

    if decay == "exponential":
        # Exponential decay: reward = scale * exp(-d / scale)
        if scale <= 0:
            raise ValueError(
                f"scale must be positive for exponential decay (got {scale})"
            )
        reward = scale * np.exp(-distances / scale)

    elif decay == "linear":
        # Linear decay: reward = scale * max(0, 1 - d/max_distance)
        if max_distance is None:
            raise ValueError(
                "max_distance parameter is required for decay='linear'. "
                "Specify the distance at which reward reaches zero."
            )
        reward = scale * np.maximum(0, 1 - distances / max_distance)

    elif decay == "inverse":
        # Inverse distance: reward = scale / (1 + d)
        reward = scale / (1 + distances)

    else:
        raise ValueError(
            f"Invalid decay type: {decay}. "
            f"Must be one of: 'linear', 'exponential', 'inverse'"
        )

    return reward
