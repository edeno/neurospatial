"""
GLM regressor generation for neurospatial events.

This module provides functions to generate regressors for GLM design matrices:

Temporal regressors:
- time_since_event: Time since most recent event
- time_to_event: Time until next event
- event_count_in_window: Count events in time window
- event_indicator: Binary indicator of event presence
- exponential_kernel: Convolve events with exponential kernel

Spatial regressors:
- distance_to_event_at_time: Distance to last/next event location
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray


def time_since_event(
    sample_times: NDArray[np.float64],
    event_times: NDArray[np.float64],
    *,
    max_time: float | None = None,
    fill_before_first: float | None = None,
    nan_policy: Literal["raise", "fill", "propagate"] = "propagate",
) -> NDArray[np.float64]:
    """
    Compute time since most recent event for each sample.

    Parameters
    ----------
    sample_times : NDArray[np.float64], shape (n_samples,)
        Times at which to compute regressor.
    event_times : NDArray[np.float64], shape (n_events,)
        Event timestamps (seconds).
    max_time : float, optional
        Maximum time to return (clips at this value).
        Useful for capping distant events.
    fill_before_first : float, optional
        Value to use before first event. Default: NaN.
    nan_policy : {"raise", "fill", "propagate"}, default="propagate"
        How to handle NaN values in output:
        - "raise": Raise ValueError if any output would be NaN
        - "fill": Fill NaN with `fill_before_first` (required if policy="fill")
        - "propagate": Keep NaN values (default, suitable for GLMs with NaN handling)

    Returns
    -------
    NDArray[np.float64], shape (n_samples,)
        Time since most recent event (seconds).
        NaN for samples before first event (unless fill_before_first set).

    Raises
    ------
    ValueError
        If nan_policy="raise" and output would contain NaN values.
        If nan_policy="fill" but fill_before_first is not provided.
        If sample_times or event_times contain NaN or Inf values.
        If max_time is negative.

    Notes
    -----
    Common use cases:
    - Time since reward: Captures reward expectation decay
    - Time since cue: Captures cue-triggered anticipation
    - Time since zone entry: Captures spatial context

    Examples
    --------
    >>> from neurospatial.events import time_since_event
    >>> sample_times = np.linspace(0, 10, 100)
    >>> reward_times = np.array([2.0, 5.0, 8.0])
    >>> time_since_reward = time_since_event(sample_times, reward_times)
    >>> # Use in GLM design matrix
    >>> X = sm.add_constant(time_since_reward[:, None])
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
            "  WHY: max_time clips the time since event to a positive value.\n"
            "  HOW: Use max_time >= 0 or None for no clipping."
        )

    if nan_policy == "fill" and fill_before_first is None:
        raise ValueError(
            "nan_policy='fill' requires fill_before_first to be specified.\n"
            "  WHY: Cannot fill NaN values without a fill value.\n"
            "  HOW: Provide fill_before_first=value or use nan_policy='propagate'."
        )

    # Handle empty sample_times
    if len(sample_times) == 0:
        return np.array([], dtype=np.float64)

    # Handle empty events
    if len(event_times) == 0:
        result = np.full(len(sample_times), np.nan, dtype=np.float64)
        if fill_before_first is not None:
            result[:] = fill_before_first
        if nan_policy == "raise" and np.any(np.isnan(result)):
            raise ValueError(
                "Output contains NaN values (no events provided).\n"
                "  WHY: nan_policy='raise' requires no NaN in output.\n"
                "  HOW: Provide events, set fill_before_first, or use nan_policy='propagate'."
            )
        return result

    # Sort events (handles unsorted input)
    sorted_events = np.sort(event_times)

    # Use searchsorted to find the index of the most recent event
    # searchsorted returns the index where each sample would be inserted
    # to maintain sorted order, so we subtract 1 to get the most recent event
    indices = np.searchsorted(sorted_events, sample_times, side="right") - 1

    # Compute result
    result = np.empty(len(sample_times), dtype=np.float64)

    # Samples before first event
    before_first = indices < 0

    # Samples after at least one event
    after_event = ~before_first

    # Time since most recent event for samples after events
    result[after_event] = (
        sample_times[after_event] - sorted_events[indices[after_event]]
    )

    # Handle samples before first event
    if fill_before_first is not None:
        result[before_first] = fill_before_first
    else:
        result[before_first] = np.nan

    # Apply max_time clipping (only to non-NaN values)
    if max_time is not None:
        valid_mask = ~np.isnan(result)
        result[valid_mask] = np.minimum(result[valid_mask], max_time)

    # Check nan_policy
    if nan_policy == "raise" and np.any(np.isnan(result)):
        raise ValueError(
            "Output contains NaN values (samples before first event).\n"
            "  WHY: nan_policy='raise' requires no NaN in output.\n"
            "  HOW: Set fill_before_first to handle samples before first event, "
            "or use nan_policy='propagate'."
        )

    return result
