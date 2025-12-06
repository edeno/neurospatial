"""
GLM regressor generation for neurospatial events.

This module provides functions to generate regressors for GLM design matrices:

Temporal regressors:
- time_to_nearest_event: Signed time to nearest event (peri-event time)
- event_count_in_window: Count events in time window
- event_indicator: Binary indicator of event presence
- exponential_kernel: Convolve events with exponential kernel

Spatial regressors:
- distance_to_reward: Distance to reward location (event-triggered)
- distance_to_boundary: Distance to environment boundary/obstacles
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from neurospatial import Environment


def time_to_nearest_event(
    sample_times: NDArray[np.float64],
    event_times: NDArray[np.float64],
    *,
    signed: bool = True,
    max_time: float | None = None,
) -> NDArray[np.float64]:
    """
    Compute time to nearest event for each sample.

    Returns signed time relative to nearest event: negative values indicate
    time before the event, positive values indicate time after. This matches
    the convention used in peri-event time histograms (PSTH).

    Parameters
    ----------
    sample_times : NDArray[np.float64], shape (n_samples,)
        Times at which to compute regressor.
    event_times : NDArray[np.float64], shape (n_events,)
        Event timestamps (seconds).
    signed : bool, default=True
        If True, return signed time (negative before event, positive after).
        If False, return absolute distance to nearest event.
    max_time : float, optional
        Maximum absolute time to return. Values beyond this are clipped.
        For signed=True, clips to [-max_time, +max_time].
        For signed=False, clips to [0, max_time].

    Returns
    -------
    NDArray[np.float64], shape (n_samples,)
        Time to nearest event (seconds).
        - signed=True: Negative before event, positive after, zero at event.
        - signed=False: Absolute distance to nearest event.
        - NaN when no events are provided.

    Raises
    ------
    ValueError
        If sample_times or event_times contain NaN or Inf values.
        If max_time is negative.

    Notes
    -----
    This function finds the nearest event for each sample and computes the
    signed time difference. For peri-event analysis:

    - Use `signed=True` (default) for PSTH-like time axis
    - Use `signed=False` for distance-based filtering
    - Use `max_time` to define a peri-event window

    Common use cases:
    - PSTH time axis: samples within window of events
    - GLM regressor: continuous time-to-event predictor
    - Event-triggered filtering: select samples near events

    At exact midpoints between two events, the earlier event is used
    (tie-breaking is consistent but arbitrary).

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.events import time_to_nearest_event

    Basic usage - PSTH-like time axis:

    >>> sample_times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    >>> event_times = np.array([2.0])
    >>> time_to_nearest_event(sample_times, event_times)
    array([-2., -1.,  0.,  1.,  2.])

    Filter to peri-event window:

    >>> times = time_to_nearest_event(sample_times, event_times)
    >>> peri_event_mask = np.abs(times) <= 1.5  # +/- 1.5s window
    >>> sample_times[peri_event_mask]
    array([1., 2., 3.])

    Clip to maximum time (for GLM design matrix):

    >>> time_to_nearest_event(sample_times, event_times, max_time=1.0)
    array([-1., -1.,  0.,  1.,  1.])
    """
    # Convert to numpy arrays and validate
    sample_times = np.asarray(sample_times, dtype=np.float64)
    event_times = np.asarray(event_times, dtype=np.float64)

    # Input validation
    if np.any(np.isnan(sample_times)):
        raise ValueError(
            "sample_times contains NaN values.\n"
            "  WHY: Times must be valid numeric values.\n"
            "  HOW: Remove or interpolate NaN values before calling."
        )

    if np.any(np.isinf(sample_times)):
        raise ValueError(
            "sample_times contains inf values.\n"
            "  WHY: Times must be finite.\n"
            "  HOW: Remove or clip infinite values before calling."
        )

    if len(event_times) > 0:
        if np.any(np.isnan(event_times)):
            raise ValueError(
                "event_times contains NaN values.\n"
                "  WHY: Event times must be valid numeric values.\n"
                "  HOW: Remove NaN values from event times."
            )

        if np.any(np.isinf(event_times)):
            raise ValueError(
                "event_times contains inf values.\n"
                "  WHY: Event times must be finite.\n"
                "  HOW: Remove infinite values from event times."
            )

    if max_time is not None and max_time < 0:
        raise ValueError(
            f"max_time must be non-negative, got {max_time}.\n"
            "  WHY: max_time defines a symmetric window around events.\n"
            "  HOW: Use max_time >= 0 or None for no clipping."
        )

    # Handle empty sample_times
    if len(sample_times) == 0:
        return np.array([], dtype=np.float64)

    # Handle empty events
    if len(event_times) == 0:
        return np.full(len(sample_times), np.nan, dtype=np.float64)

    # Sort events (handles unsorted input)
    sorted_events = np.sort(event_times)

    # Find indices of surrounding events using searchsorted
    # idx_right: index of first event >= sample_time (or len if none)
    idx_right = np.searchsorted(sorted_events, sample_times, side="left")

    # idx_left: index of last event < sample_time (or -1 if none)
    idx_left = idx_right - 1

    # Compute distances to left and right events
    n_events = len(sorted_events)
    n_samples = len(sample_times)

    # Initialize with large values
    dist_left = np.full(n_samples, np.inf, dtype=np.float64)
    dist_right = np.full(n_samples, np.inf, dtype=np.float64)

    # Distance to left event (if exists)
    has_left = idx_left >= 0
    dist_left[has_left] = sample_times[has_left] - sorted_events[idx_left[has_left]]

    # Distance to right event (if exists)
    has_right = idx_right < n_events
    dist_right[has_right] = (
        sorted_events[idx_right[has_right]] - sample_times[has_right]
    )

    # Select nearest event
    # For ties (dist_left == dist_right), prefer left (earlier) event
    use_left = dist_left <= dist_right

    # Compute signed time to nearest event
    result = np.empty(n_samples, dtype=np.float64)
    result[use_left] = dist_left[use_left]  # Positive (after left event)
    result[~use_left] = -dist_right[~use_left]  # Negative (before right event)

    # Convert to unsigned if requested
    if not signed:
        result = np.abs(result)

    # Convert -0.0 to 0.0 for cleaner output
    result = result + 0.0  # Adding 0.0 converts -0.0 to 0.0

    # Apply max_time clipping
    if max_time is not None:
        result = np.clip(result, -max_time if signed else 0.0, max_time)

    return result


def event_count_in_window(
    sample_times: NDArray[np.float64],
    event_times: NDArray[np.float64],
    window: tuple[float, float],
) -> NDArray[np.int64]:
    """
    Count events within time window around each sample.

    For each sample time, counts the number of events that fall within
    the specified window. Useful for creating GLM regressors that capture
    recent event history (e.g., "rewards in last 5 seconds").

    Parameters
    ----------
    sample_times : NDArray[np.float64], shape (n_samples,)
        Times at which to compute count.
    event_times : NDArray[np.float64], shape (n_events,)
        Event timestamps (seconds).
    window : tuple[float, float]
        Time window (start, end) relative to sample_time.
        E.g., (-1.0, 0.0) counts events in previous 1 second.
        E.g., (0.0, 1.0) counts events in next 1 second.
        E.g., (-0.5, 0.5) counts events within +/- 0.5 seconds.
        Window boundaries are inclusive.

    Returns
    -------
    NDArray[np.int64], shape (n_samples,)
        Number of events within window of each sample.
        Always non-negative integers.

    Raises
    ------
    ValueError
        If sample_times or event_times contain NaN or Inf values.
        If window start > window end.

    Notes
    -----
    Common use cases:

    - Count rewards in last N seconds: ``window=(-N, 0.0)``
    - Count licks before reward: ``window=(-1.0, 0.0)``
    - Count spikes around event: ``window=(-0.5, 0.5)``

    Window boundaries are inclusive on both ends, so an event exactly
    at ``sample_time + window[0]`` or ``sample_time + window[1]`` is counted.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.events import event_count_in_window

    Count rewards in previous 5 seconds:

    >>> sample_times = np.array([0.0, 3.0, 6.0, 10.0])
    >>> reward_times = np.array([1.0, 2.0, 5.0])
    >>> event_count_in_window(sample_times, reward_times, window=(-5.0, 0.0))
    array([0, 2, 3, 1])

    Count events in symmetric window:

    >>> event_count_in_window(sample_times, reward_times, window=(-1.5, 1.5))
    array([1, 1, 1, 0])
    """
    # Convert to numpy arrays and validate
    sample_times = np.asarray(sample_times, dtype=np.float64)
    event_times = np.asarray(event_times, dtype=np.float64)

    # Input validation
    if np.any(np.isnan(sample_times)):
        raise ValueError(
            "sample_times contains NaN values.\n"
            "  WHY: Times must be valid numeric values.\n"
            "  HOW: Remove or interpolate NaN values before calling."
        )

    if np.any(np.isinf(sample_times)):
        raise ValueError(
            "sample_times contains inf values.\n"
            "  WHY: Times must be finite.\n"
            "  HOW: Remove or clip infinite values before calling."
        )

    if len(event_times) > 0:
        if np.any(np.isnan(event_times)):
            raise ValueError(
                "event_times contains NaN values.\n"
                "  WHY: Event times must be valid numeric values.\n"
                "  HOW: Remove NaN values from event times."
            )

        if np.any(np.isinf(event_times)):
            raise ValueError(
                "event_times contains inf values.\n"
                "  WHY: Event times must be finite.\n"
                "  HOW: Remove infinite values from event times."
            )

    # Validate window
    window_start, window_end = window
    if window_start > window_end:
        raise ValueError(
            f"window start ({window_start}) must be <= window end ({window_end}).\n"
            "  WHY: Window defines a time range [start, end] relative to sample.\n"
            "  HOW: Use window=(start, end) where start <= end."
        )

    # Handle empty sample_times
    if len(sample_times) == 0:
        return np.array([], dtype=np.int64)

    # Handle empty events
    if len(event_times) == 0:
        return np.zeros(len(sample_times), dtype=np.int64)

    # Sort events for efficient searchsorted
    sorted_events = np.sort(event_times)

    # For each sample, compute window bounds in absolute time
    # and count events within those bounds
    window_starts = sample_times + window_start
    window_ends = sample_times + window_end

    # Use searchsorted to find event indices at window boundaries
    # left_indices: first event >= window_start
    # right_indices: first event > window_end (so count = right - left)
    left_indices = np.searchsorted(sorted_events, window_starts, side="left")
    right_indices = np.searchsorted(sorted_events, window_ends, side="right")

    # Count = number of events in [window_start, window_end]
    counts = right_indices - left_indices

    return counts.astype(np.int64)


def event_indicator(
    sample_times: NDArray[np.float64],
    event_times: NDArray[np.float64],
    *,
    window: float = 0.0,
) -> NDArray[np.bool_]:
    """
    Binary indicator of whether an event occurs near each sample time.

    For each sample, returns True if any event falls within the specified
    window around that sample. Useful for creating indicator variables in
    GLM design matrices.

    Parameters
    ----------
    sample_times : NDArray[np.float64], shape (n_samples,)
        Sample timestamps.
    event_times : NDArray[np.float64], shape (n_events,)
        Event timestamps.
    window : float, default=0.0
        Symmetric half-width of temporal window (seconds).
        Creates a window [sample_time - window, sample_time + window].
        If 0, only exact matches count (sample_time == event_time).
        If > 0, events within the symmetric window are considered a match.

    Returns
    -------
    NDArray[np.bool_], shape (n_samples,)
        True if event occurs within window of sample, False otherwise.

    Raises
    ------
    ValueError
        If sample_times or event_times contain NaN or Inf values.
        If window is negative.

    Notes
    -----
    Common use cases:

    - Binary event regressor: ``window=0.0`` for impulse at event time
    - Smoothed indicator: ``window=bin_size/2`` for time-binned data
    - Event presence detection: ``window=0.1`` for events within 100ms

    Window boundaries are inclusive, so events exactly at ``sample_time ± window``
    are included (returns True).

    This function is semantically clearer than ``event_count_in_window() > 0``
    for creating binary indicators and returns bool dtype directly.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.events import event_indicator

    Check for exact event occurrence:

    >>> sample_times = np.array([0.0, 1.0, 2.0, 3.0])
    >>> event_times = np.array([1.0, 3.0])
    >>> event_indicator(sample_times, event_times)
    array([False,  True, False,  True])

    Check for events within window:

    >>> event_indicator(sample_times, event_times, window=0.5)
    array([False,  True, False,  True])

    Create GLM design matrix column:

    >>> indicator = event_indicator(sample_times, event_times, window=0.1)
    >>> X = indicator.astype(float)  # Convert to float for design matrix
    """
    # Convert to numpy arrays and validate
    sample_times = np.asarray(sample_times, dtype=np.float64)
    event_times = np.asarray(event_times, dtype=np.float64)

    # Input validation
    if np.any(np.isnan(sample_times)):
        raise ValueError(
            "sample_times contains NaN values.\n"
            "  WHY: Times must be valid numeric values.\n"
            "  HOW: Remove or interpolate NaN values before calling."
        )

    if np.any(np.isinf(sample_times)):
        raise ValueError(
            "sample_times contains inf values.\n"
            "  WHY: Times must be finite.\n"
            "  HOW: Remove or clip infinite values before calling."
        )

    if len(event_times) > 0:
        if np.any(np.isnan(event_times)):
            raise ValueError(
                "event_times contains NaN values.\n"
                "  WHY: Event times must be valid numeric values.\n"
                "  HOW: Remove NaN values from event times."
            )

        if np.any(np.isinf(event_times)):
            raise ValueError(
                "event_times contains inf values.\n"
                "  WHY: Event times must be finite.\n"
                "  HOW: Remove infinite values from event times."
            )

    if window < 0:
        raise ValueError(
            f"window must be non-negative, got {window}.\n"
            "  WHY: window defines a symmetric half-width around each sample.\n"
            "  HOW: Use window >= 0."
        )

    # Handle empty sample_times
    if len(sample_times) == 0:
        return np.array([], dtype=np.bool_)

    # Handle empty events
    if len(event_times) == 0:
        return np.zeros(len(sample_times), dtype=np.bool_)

    # Sort events for efficient searchsorted
    sorted_events = np.sort(event_times)

    # For each sample, check if any event falls within [sample - window, sample + window]
    # Use searchsorted to find events at window boundaries
    window_starts = sample_times - window
    window_ends = sample_times + window

    # left_indices: first event >= window_start
    # right_indices: first event > window_end
    left_indices = np.searchsorted(sorted_events, window_starts, side="left")
    right_indices = np.searchsorted(sorted_events, window_ends, side="right")

    # If right > left, there's at least one event in the window
    result = right_indices > left_indices

    return result


def distance_to_reward(
    env: Environment,
    positions: NDArray[np.float64],
    times: NDArray[np.float64],
    reward_times: NDArray[np.float64],
    reward_positions: NDArray[np.float64] | None = None,
    *,
    metric: Literal["geodesic", "euclidean"] = "geodesic",
    mode: Literal["nearest", "last", "next"] = "nearest",
) -> NDArray[np.float64]:
    """
    Compute distance to reward location at each sample.

    For each sample, computes the distance from current position to the
    reward location. The reward location is determined by the ``mode``:

    - "nearest": Uses the closest reward in time (before or after)
    - "last": Uses the most recent reward before current time
    - "next": Uses the next upcoming reward after current time

    Parameters
    ----------
    env : Environment
        Fitted spatial environment.
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Position coordinates at each sample time. Must match environment
        dimensionality (e.g., 2D for grid environments).
    times : NDArray[np.float64], shape (n_samples,)
        Sample timestamps (seconds).
    reward_times : NDArray[np.float64], shape (n_events,)
        Reward event timestamps (seconds).
    reward_positions : NDArray[np.float64], shape (n_events, n_dims), optional
        Reward locations in environment coordinates. If None, reward
        locations are inferred by interpolating ``positions`` at
        ``reward_times``.
    metric : {'geodesic', 'euclidean'}, default='geodesic'
        Distance metric:

        - 'geodesic': Shortest-path distance respecting environment geometry
          (walls, obstacles). Uses graph-based distances.
        - 'euclidean': Straight-line L2 distance. Ignores obstacles.
    mode : {'nearest', 'last', 'next'}, default='nearest'
        Which reward to use for distance computation:

        - 'nearest': Distance to temporally closest reward (before or after).
        - 'last': Distance to most recent reward. NaN before first reward.
        - 'next': Distance to next upcoming reward. NaN after last reward.

    Returns
    -------
    NDArray[np.float64], shape (n_samples,)
        Distance from current position to the reward location at each
        sample. Units match environment coordinates (e.g., cm).
        Returns NaN when:

        - No relevant reward exists (e.g., mode="last" before first reward)
        - Position is outside environment (invalid bin)
        - Reward position is outside environment

    Raises
    ------
    ValueError
        If positions and times have different lengths.
        If reward_positions provided but doesn't match reward_times length.
        If positions dimensionality doesn't match environment.

    Notes
    -----
    **Performance**:

    This function computes distance fields for each unique reward location,
    which is O(V log V + E) per location using Dijkstra's algorithm. For
    many unique reward locations, consider caching or using Euclidean
    distances.

    **Interpolation**:

    When ``reward_positions`` is not provided, positions are interpolated
    using linear interpolation. This assumes smooth movement between
    samples. For teleporting animals or discontinuous tracking, provide
    explicit reward positions.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.events import distance_to_reward
    >>> # Distance to nearest reward (default)
    >>> dist = distance_to_reward(  # doctest: +SKIP
    ...     env,
    ...     positions,
    ...     times,
    ...     reward_times,
    ...     metric="geodesic",
    ...     mode="nearest",
    ... )
    >>>
    >>> # Distance to most recent reward
    >>> dist_last = distance_to_reward(  # doctest: +SKIP
    ...     env,
    ...     positions,
    ...     times,
    ...     reward_times,
    ...     mode="last",
    ... )
    >>>
    >>> # Distance to next upcoming reward
    >>> dist_next = distance_to_reward(  # doctest: +SKIP
    ...     env,
    ...     positions,
    ...     times,
    ...     reward_times,
    ...     mode="next",
    ... )
    >>>
    >>> # GLM design matrix
    >>> X = np.column_stack(  # doctest: +SKIP
    ...     [
    ...         time_to_nearest_event(times, reward_times, max_time=5.0),
    ...         distance_to_reward(env, positions, times, reward_times),
    ...     ]
    ... )

    See Also
    --------
    time_to_nearest_event : Temporal distance to events
    distance_to_boundary : Distance to environment boundaries
    """
    from neurospatial.ops.distance import distance_field

    # Input validation
    positions = np.asarray(positions, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    reward_times = np.asarray(reward_times, dtype=np.float64)

    n_samples = len(times)

    if len(positions) != n_samples:
        raise ValueError(
            f"positions and times must have same length. "
            f"Got positions: {len(positions)}, times: {n_samples}."
        )

    # Handle empty inputs
    if n_samples == 0:
        return np.array([], dtype=np.float64)

    if len(reward_times) == 0:
        return np.full(n_samples, np.nan, dtype=np.float64)

    # Get or interpolate reward positions
    if reward_positions is not None:
        reward_positions = np.asarray(reward_positions, dtype=np.float64)
        if len(reward_positions) != len(reward_times):
            raise ValueError(
                f"reward_positions and reward_times must have same length. "
                f"Got reward_positions: {len(reward_positions)}, "
                f"reward_times: {len(reward_times)}."
            )
    else:
        # Interpolate positions at reward times
        # Handle edge cases: clip to valid time range
        clipped_reward_times = np.clip(reward_times, times.min(), times.max())
        reward_positions = np.column_stack(
            [
                np.interp(clipped_reward_times, times, positions[:, dim])
                for dim in range(positions.shape[1])
            ]
        )

    # Map positions to bins
    trajectory_bins = env.bin_at(positions)

    # Map reward positions to bins
    reward_bins = env.bin_at(reward_positions)

    # Sort rewards by time for efficient lookup
    sorted_idx = np.argsort(reward_times)
    sorted_reward_times = reward_times[sorted_idx]
    sorted_reward_bins = reward_bins[sorted_idx]

    # Find relevant reward for each sample based on mode
    if mode == "nearest":
        # Find nearest reward in time (before or after)
        # Use searchsorted to find insertion points
        idx_right = np.searchsorted(sorted_reward_times, times, side="left")
        idx_left = idx_right - 1

        # Compute time distances to left and right rewards
        n_rewards = len(sorted_reward_times)
        dist_left = np.full(n_samples, np.inf)
        dist_right = np.full(n_samples, np.inf)

        has_left = idx_left >= 0
        dist_left[has_left] = times[has_left] - sorted_reward_times[idx_left[has_left]]

        has_right = idx_right < n_rewards
        dist_right[has_right] = (
            sorted_reward_times[idx_right[has_right]] - times[has_right]
        )

        # Select nearest
        use_left = dist_left <= dist_right
        relevant_reward_idx = np.where(use_left, idx_left, idx_right)

        # Handle edge case: no rewards exist
        no_reward = (idx_left < 0) & (idx_right >= n_rewards)
        relevant_reward_idx = np.clip(relevant_reward_idx, 0, n_rewards - 1)

    elif mode == "last":
        # Find most recent reward before each sample
        # searchsorted gives index of first reward > time, so idx-1 is last reward <= time
        idx = np.searchsorted(sorted_reward_times, times, side="right") - 1
        relevant_reward_idx = idx
        no_reward = idx < 0

    elif mode == "next":
        # Find next upcoming reward after each sample
        # searchsorted gives index of first reward >= time
        idx = np.searchsorted(sorted_reward_times, times, side="left")
        relevant_reward_idx = idx
        no_reward = idx >= len(sorted_reward_times)

    else:
        raise ValueError(f"mode must be 'nearest', 'last', or 'next', got '{mode}'.")

    # Clip indices to valid range for array access
    relevant_reward_idx = np.clip(relevant_reward_idx, 0, len(sorted_reward_bins) - 1)

    # Get reward bins for each sample
    target_bins = sorted_reward_bins[relevant_reward_idx]

    # Compute distances
    # Find unique target bins to minimize distance field computations
    unique_targets = np.unique(target_bins[~no_reward])
    unique_targets = unique_targets[unique_targets >= 0]  # Filter invalid bins

    # Initialize distances
    distances = np.full(n_samples, np.nan, dtype=np.float64)

    # Compute distance field for each unique target
    for target_bin in unique_targets:
        # Find samples with this target
        mask = (target_bins == target_bin) & ~no_reward

        if not np.any(mask):
            continue

        # Compute distance field from this reward location
        dist_field = distance_field(
            env.connectivity,
            [int(target_bin)],
            metric=metric,
            bin_centers=env.bin_centers if metric == "euclidean" else None,
        )

        # Look up distances for trajectory bins
        sample_bins = trajectory_bins[mask]
        valid_bins = sample_bins >= 0
        distances[mask] = np.where(
            valid_bins,
            dist_field[np.clip(sample_bins, 0, len(dist_field) - 1)],
            np.nan,
        )

    # Set NaN for samples with no relevant reward
    distances[no_reward] = np.nan

    # Set NaN for invalid trajectory bins
    distances[trajectory_bins < 0] = np.nan

    # Set NaN for invalid reward bins
    invalid_reward_mask = target_bins < 0
    distances[invalid_reward_mask & ~no_reward] = np.nan

    return distances


def distance_to_boundary(
    env: Environment,
    positions: NDArray[np.float64],
    *,
    boundary_type: Literal["edge", "obstacle", "region"] = "edge",
    region_name: str | None = None,
    metric: Literal["geodesic", "euclidean"] = "geodesic",
) -> NDArray[np.float64]:
    """
    Compute distance to environment boundary at each position.

    Measures the instantaneous distance from trajectory positions to various
    types of boundaries: environment edges, obstacles, or named regions.

    Parameters
    ----------
    env : Environment
        Fitted spatial environment.
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Position coordinates at each sample. Must match environment
        dimensionality.
    boundary_type : {'edge', 'obstacle', 'region'}, default='edge'
        Type of boundary to compute distance to:

        - 'edge': Distance to environment boundary (unoccupied bins).
          Includes arena walls and any holes in the environment.
        - 'obstacle': Distance to obstacle regions. Requires environment
          to have regions with ``obstacle=True`` attribute.
        - 'region': Distance to boundary of a specific named region.
          Requires ``region_name`` parameter.
    region_name : str, optional
        Name of region for boundary_type='region'. Required when
        boundary_type='region', ignored otherwise.
    metric : {'geodesic', 'euclidean'}, default='geodesic'
        Distance metric:

        - 'geodesic': Shortest-path distance respecting environment geometry.
        - 'euclidean': Straight-line L2 distance.

    Returns
    -------
    NDArray[np.float64], shape (n_samples,)
        Distance from each position to the nearest boundary bin.
        Units match environment coordinates (e.g., cm).
        Returns NaN for positions outside the environment.

    Raises
    ------
    ValueError
        If boundary_type='region' but region_name is not provided.
        If region_name is provided but region doesn't exist.
        If no boundary bins found for the specified boundary_type.

    Notes
    -----
    **Boundary detection**:

    - **edge**: Finds bins adjacent to unoccupied space. A bin is an edge
      bin if any of its graph neighbors are not in the environment.
    - **obstacle**: Finds bins adjacent to obstacle regions. Obstacles
      are regions marked with ``obstacle=True`` during annotation.
    - **region**: Finds bins at the boundary of the named region. Boundary
      bins are region bins adjacent to non-region bins.

    **Performance**:

    The distance field is computed once and cached for the duration of the
    call. For repeated queries with different positions but the same
    boundary, consider caching the distance field manually using
    ``distance_field(env.connectivity, boundary_bins)``.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.events import distance_to_boundary
    >>> # Distance to environment walls
    >>> dist_wall = distance_to_boundary(
    ...     env, positions, boundary_type="edge"
    ... )  # doctest: +SKIP
    >>>
    >>> # Distance to obstacle
    >>> dist_obstacle = distance_to_boundary(
    ...     env, positions, boundary_type="obstacle"
    ... )  # doctest: +SKIP
    >>>
    >>> # Distance to goal region boundary
    >>> dist_goal = distance_to_boundary(  # doctest: +SKIP
    ...     env, positions, boundary_type="region", region_name="goal"
    ... )
    >>>
    >>> # GLM design matrix with multiple boundary features
    >>> X = np.column_stack(  # doctest: +SKIP
    ...     [
    ...         distance_to_boundary(env, positions, boundary_type="edge"),
    ...         distance_to_boundary(
    ...             env, positions, boundary_type="region", region_name="goal"
    ...         ),
    ...     ]
    ... )

    See Also
    --------
    distance_to_reward : Distance to reward locations
    time_to_nearest_event : Temporal distance to events
    """
    from neurospatial.ops.distance import distance_field

    # Input validation
    positions = np.asarray(positions, dtype=np.float64)
    n_samples = len(positions)

    if n_samples == 0:
        return np.array([], dtype=np.float64)

    if boundary_type == "region" and region_name is None:
        raise ValueError(
            "region_name is required when boundary_type='region'. "
            "Provide the name of the region to compute distance to."
        )

    # Find boundary bins based on boundary_type
    if boundary_type == "edge":
        # Edge bins: bins adjacent to unoccupied space
        # A bin is an edge bin if it has fewer neighbors than expected
        # for a fully-surrounded bin
        boundary_bins = _find_edge_bins(env)

    elif boundary_type == "obstacle":
        # Obstacle bins: bins adjacent to obstacle regions
        boundary_bins = _find_obstacle_boundary_bins(env)

    elif boundary_type == "region":
        # Region boundary bins: boundary of the named region
        # region_name is already validated as not None by the earlier check
        assert region_name is not None  # for type checker
        if region_name not in env.regions:
            raise ValueError(
                f"Region '{region_name}' not found in environment. "
                f"Available regions: {list(env.regions.keys())}."
            )
        boundary_bins = _find_region_boundary_bins(env, region_name)

    else:
        raise ValueError(
            f"boundary_type must be 'edge', 'obstacle', or 'region', "
            f"got '{boundary_type}'."
        )

    if len(boundary_bins) == 0:
        raise ValueError(
            f"No boundary bins found for boundary_type='{boundary_type}'"
            + (f", region_name='{region_name}'" if region_name else "")
            + ". Check that the environment has valid boundaries."
        )

    # Compute distance field from boundary bins
    dist_field = distance_field(
        env.connectivity,
        list(boundary_bins),
        metric=metric,
        bin_centers=env.bin_centers if metric == "euclidean" else None,
    )

    # Map positions to bins
    trajectory_bins = env.bin_at(positions)

    # Look up distances
    distances = np.full(n_samples, np.nan, dtype=np.float64)
    valid_bins = trajectory_bins >= 0
    distances[valid_bins] = dist_field[trajectory_bins[valid_bins]]

    return distances


def _find_edge_bins(env: Environment) -> list[int]:
    """Find bins at the edge of the environment.

    A bin is an edge bin if it has fewer than the maximum number of
    neighbors (indicating it's adjacent to unoccupied space).

    Parameters
    ----------
    env : Environment
        Fitted environment.

    Returns
    -------
    list[int]
        Bin indices at the environment boundary.
    """
    graph = env.connectivity
    all_bins = set(graph.nodes())
    max_degree = _get_max_expected_degree(env)

    edge_bins = []
    for bin_idx in all_bins:
        # A bin is an edge bin if it has fewer neighbors than expected
        # for a fully-surrounded bin in this layout (i.e., it's adjacent
        # to unoccupied space)
        if graph.degree(bin_idx) < max_degree:
            edge_bins.append(bin_idx)

    return edge_bins


def _get_max_expected_degree(env: Environment) -> int:
    """Get the expected maximum degree for a fully-surrounded bin.

    This varies by layout type:
    - Regular grid: 4 (orthogonal) or 8 (with diagonals)
    - Hexagonal: 6
    - Graph-based: varies

    Parameters
    ----------
    env : Environment
        Fitted environment.

    Returns
    -------
    int
        Maximum expected degree for interior bins.
    """
    graph = env.connectivity
    if graph.number_of_nodes() == 0:
        return 0

    # Use the maximum degree in the graph as the expected interior degree
    # This assumes at least some bins are fully surrounded
    degrees = [graph.degree(n) for n in graph.nodes()]
    return max(degrees) if degrees else 0


def _find_obstacle_boundary_bins(env: Environment) -> list[int]:
    """Find bins adjacent to obstacle regions.

    Parameters
    ----------
    env : Environment
        Fitted environment with obstacle regions.

    Returns
    -------
    list[int]
        Bin indices at obstacle boundaries.
    """
    # Find all obstacle regions
    obstacle_regions = [
        name
        for name, region in env.regions.items()
        if hasattr(region, "obstacle") and region.obstacle
    ]

    if not obstacle_regions:
        # No obstacle regions defined - return empty
        return []

    # For environments with obstacle regions, bins adjacent to obstacles
    # are typically edge bins (they have fewer neighbors because the
    # obstacle removes connectivity). This is currently a simplification.
    # A more complete implementation would require the environment to
    # explicitly track which bins were removed due to obstacles.
    return _find_edge_bins(env)


def _find_region_boundary_bins(env: Environment, region_name: str) -> list[int]:
    """Find bins at the boundary of a named region.

    A bin is a region boundary bin if it's in the region AND has at least
    one neighbor outside the region.

    Parameters
    ----------
    env : Environment
        Fitted environment.
    region_name : str
        Name of the region.

    Returns
    -------
    list[int]
        Bin indices at the region boundary.
    """
    graph = env.connectivity
    region_bins = set(env.bins_in_region(region_name))

    if not region_bins:
        return []

    boundary_bins = []
    for bin_idx in region_bins:
        neighbors = set(graph.neighbors(bin_idx))
        # Check if any neighbor is outside the region
        if not neighbors.issubset(region_bins):
            boundary_bins.append(bin_idx)

    return boundary_bins
