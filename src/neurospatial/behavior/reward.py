"""Reward field generation for reinforcement learning applications.

This module provides functions for creating reward fields from spatial regions
and goal locations, for reinforcement learning (RL) models of spatial
navigation. A reward field can serve as the potential of potential-based
reward shaping (Ng et al., 1999).

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
    from neurospatial.geometry import CartesianSystem

from neurospatial._exceptions import _format_error
from neurospatial._validation import validate_finite_scalar
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
    max_distance: float | None = None,
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
        Gaussian smoothing bandwidth, in the environment's distance units.
        Required for, and only accepted with, ``decay='gaussian'``.
    max_distance : float, optional
        Geodesic distance from the region, in the environment's distance
        units, at which the linear reward reaches 0. Only accepted with
        ``decay='linear'``. Default: the distance of the farthest bin that
        can reach the region, so every reachable bin outside the region gets
        a nonzero gradient. That default depends on the arena size: the same
        distance gets a different reward in a larger arena. Pass
        ``max_distance`` to compare reward fields across arenas.

    Returns
    -------
    reward : NDArray[np.float64], shape (n_bins,)
        Reward field with specified decay profile.

    Raises
    ------
    RegionNotFoundError
        If region_name is not found in env.regions (a ``KeyError``).
    ValueError
        If ``decay`` is unknown; if ``bandwidth`` is missing for
        ``decay='gaussian'``; if ``bandwidth`` or ``max_distance`` is given
        for a decay that does not use it; or if either is not finite and
        positive.

    Notes
    -----
    **Decay Types**:

    - **Constant**: Binary reward useful for sparse reward RL tasks.
      Creates clear goal regions with no reward shaping.

    - **Linear**: ``reward_value * max(0, 1 - d / max_distance)``, where
      ``d`` is the geodesic distance from the bin centre to the nearest region
      bin centre: the length of the shortest path between two positions that
      stays inside the environment (it goes around walls and never leaves the
      environment, and on a track it runs along the track's centerline). Bins
      that cannot reach the region (another connected component) get 0.

    - **Gaussian**: Smooth falloff using Gaussian kernel smoothing.
      After smoothing, rescales so maximum *within the region* equals
      reward_value. This preserves the intended reward magnitude while
      providing smooth gradients for policy learning.

    **Reward Shaping**: A reward field can serve as the potential
    ``Phi(s)`` of potential-based reward shaping [1]_, which adds
    ``gamma * Phi(s') - Phi(s)`` to the task reward and leaves the optimal
    policy unchanged. Adding the field itself to the task reward is not
    potential-based and can change the optimal policy. Gaussian decay
    provides the smoothest gradients; validate any shaped agent against a
    sparse-reward baseline.

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
    >>> # Linear decay that reaches 0 at 30 units from the region
    >>> reward_linear = region_reward_field(
    ...     env, region_name="goal", decay="linear", max_distance=30.0
    ... )
    >>> float(reward_linear.max()), float(reward_linear.min())
    (1.0, 0.0)

    See Also
    --------
    goal_reward_field : Distance-based rewards from goal bins
    neurospatial.ops.distance_field : Bin-graph distance maps
    neurospatial.ops.regions_to_mask : Convert regions to binary masks

    References
    ----------
    .. [1] Ng, A. Y., Harada, D., & Russell, S. (1999). Policy invariance
           under reward transformations: Theory and application to reward
           shaping. ICML.
    """
    if decay not in ("constant", "linear", "gaussian"):
        raise ValueError(
            f"Invalid decay type: {decay!r}. "
            "Must be one of: 'constant', 'linear', 'gaussian'."
        )
    _reject_unused_parameter("bandwidth", bandwidth, decay, uses="gaussian")
    _reject_unused_parameter("max_distance", max_distance, decay, uses="linear")
    validate_finite_scalar(
        bandwidth, name="bandwidth", positive=True, none_allowed=True
    )
    validate_finite_scalar(
        max_distance, name="max_distance", positive=True, none_allowed=True
    )
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
        from neurospatial.behavior._environment_paths import environment_paths

        distances = environment_paths(env, call="region_reward_field").from_bins(
            np.flatnonzero(region_mask)
        )
        reachable = np.isfinite(distances) & ~region_mask
        reward = np.where(region_mask, reward_value, 0.0)
        if reachable.any():
            decay_length = (
                distances[reachable].max() if max_distance is None else max_distance
            )
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

        # Scale by the maximum inside the region (not the global maximum) so
        # the peak reward in the region equals reward_value. The region has at
        # least one bin and each bin keeps positive weight on itself, so this
        # maximum is positive.
        reward = (smoothed / smoothed[region_mask].max()) * reward_value

    return reward


def _reject_unused_parameter(
    name: str, value: float | None, decay: str, *, uses: str
) -> None:
    """Raise when a decay-specific parameter is passed with another decay."""
    if value is not None and decay != uses:
        raise ValueError(
            _format_error(
                f"{name}={value!r} has no effect with decay={decay!r}.",
                why=f"{name} only sets the shape of decay={uses!r}.",
                fix=f"remove {name}, or pass decay={uses!r}.",
            )
        )


def goal_reward_field(
    env: Environment[CartesianSystem],
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
        Distance-based decay function of the geodesic distance ``d`` (in the
        environment's distance units) from the bin centre to the nearest goal
        bin centre, the length of the shortest path between two positions that
        stays inside the environment (it goes around walls and never leaves the
        environment, and on a track it runs along the track's centerline):

        - 'linear': reward = scale * max(0, 1 - d/max_distance)
        - 'exponential': reward = scale * exp(-d/scale)
        - 'inverse': reward = scale / (1 + d)
    scale : float, default=1.0
        Reward at the goal for every decay. For 'exponential' it is also the
        decay length, in distance units (not a rate): the reward falls to
        ``scale / e`` at ``d = scale``, so a larger scale decays more slowly.
        Must be finite and positive for 'exponential'.
    max_distance : float, optional
        Distance, in the environment's distance units, at which the linear
        reward reaches 0. Required for, and only accepted with,
        ``decay='linear'``; must be finite and positive.

    Returns
    -------
    reward : NDArray[np.float64], shape (n_bins,)
        Distance-based reward field. Bins that cannot reach any goal (another
        connected component) get 0 for every decay.

    Raises
    ------
    ValueError
        If ``decay`` is unknown; if ``scale`` is not finite and positive for
        ``decay='exponential'``; if ``max_distance`` is missing, not finite
        and positive for ``decay='linear'``, or given for another decay; or
        if ``goal_bins`` contains invalid indices.

    Notes
    -----
    **Decay Functions**:

    - **Exponential**: Most common in RL literature. Provides smooth gradients
      that decay exponentially with distance. ``scale`` sets both the peak
      reward and the decay length, so they cannot be chosen separately; for
      a unit peak with decay length ``L``, divide the field from
      ``scale=L`` by ``L``.

    - **Linear**: Reaches exactly zero at max_distance. Useful when reward
      should be strictly local. Provides constant gradient within range.

    - **Inverse**: Inverse distance function (1/(1+d)). Never reaches zero,
      providing global gradients. Can lead to suboptimal policies if not
      carefully tuned. The ``1`` in ``1 + d`` is one distance unit, so the
      shape depends on the units of the environment: the reward halves 1 cm
      from the goal in a centimetre environment but 1 m from it in a metre
      environment.

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
    neurospatial.ops.distance_field : Bin-graph distance maps
    """
    from neurospatial.environment import _validate_target_bins

    if decay not in ("linear", "exponential", "inverse"):
        raise ValueError(
            f"Invalid decay type: {decay!r}. "
            "Must be one of: 'linear', 'exponential', 'inverse'."
        )
    _reject_unused_parameter("max_distance", max_distance, decay, uses="linear")
    validate_finite_scalar(
        max_distance, name="max_distance", positive=True, none_allowed=True
    )
    goal_bins_list = _validate_target_bins(
        goal_bins,
        n_bins=env.n_bins,
        coordinates=env.coordinate_system,
        operation="goal_reward_field()",
        argument="goal_bins",
    ).tolist()

    from neurospatial.behavior._environment_paths import environment_paths

    distances = environment_paths(env, call="goal_reward_field").from_bins(
        np.unique(goal_bins_list)
    )

    # Unreachable bins have d = inf, which every decay below maps to 0.
    if decay == "exponential":
        validate_finite_scalar(scale, name="scale", positive=True)
        reward = scale * np.exp(-distances / scale)
    elif decay == "linear":
        if max_distance is None:
            raise ValueError(
                "max_distance parameter is required for decay='linear'. "
                "Specify the distance at which reward reaches zero."
            )
        reward = scale * np.maximum(0, 1 - distances / max_distance)
    else:
        reward = scale / (1 + distances)

    return reward
