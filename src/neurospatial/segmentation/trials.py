"""Trial segmentation functions for behavioral task analysis.

This module provides functions for segmenting trajectories into discrete
behavioral trials based on start and end regions, commonly used in spatial
navigation tasks like T-mazes, Y-mazes, and radial arm mazes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from neurospatial import Environment


@dataclass(frozen=True)
class Trial:
    """Behavioral trial event.

    Attributes
    ----------
    start_time : float
        Time when trial started (start region entry, seconds).
    end_time : float
        Time when trial ended (end region entry or timeout, seconds).
    start_region : str
        Name of the region where trial started.
    end_region : str or None
        Name of end region reached, or None if trial timed out.
    success : bool
        True if end region reached within max_duration, False if timeout.
    """

    start_time: float
    end_time: float
    start_region: str
    end_region: str | None
    success: bool


def segment_trials(
    trajectory_bins: NDArray[np.int64],
    times: NDArray[np.float64],
    env: Environment,
    *,
    start_region: str | None = None,
    end_regions: list[str] | None = None,
    min_duration: float = 1.0,
    max_duration: float = 15.0,
) -> list[Trial]:
    """Segment trajectory into behavioral trials.

    Identifies discrete behavioral trials defined by entries to a start region
    followed by navigation to one of several end regions. Commonly used in
    spatial navigation tasks like T-mazes, Y-mazes, and radial arm mazes to
    analyze choice behavior, learning curves, and spatial strategies.

    A trial is defined as:
    1. Entry to start region (trial start)
    2. Trajectory through environment
    3. Either:
       - Entry to one of the end regions (successful trial)
       - Timeout after max_duration without reaching end region (failed trial)

    Parameters
    ----------
    trajectory_bins : NDArray[np.int64], shape (n_samples,)
        Sequence of bin indices representing the trajectory.
    times : NDArray[np.float64], shape (n_samples,)
        Time stamps corresponding to trajectory bins (seconds).
    env : Environment
        Environment containing region definitions.
    start_region : str, optional
        Name of start region. Must exist in env.regions.
        Required parameter (must be provided).
    end_regions : list[str], optional
        List of end region names (e.g., ['left', 'right'] for T-maze).
        Must all exist in env.regions. Required parameter (must be provided).
    min_duration : float, optional
        Minimum trial duration in seconds. Default: 1.0.
        Trials shorter than this are excluded.
    max_duration : float, optional
        Maximum trial duration in seconds. Default: 15.0.
        Trials exceeding this are marked as failed (timeout).

    Returns
    -------
    list[Trial]
        List of detected trials. Each trial has:
        - start_time: time of start region entry
        - end_time: time of end region entry or timeout
        - start_region: name of start region
        - end_region: name of end region reached (or None if timeout)
        - success: True if end region reached, False if timeout

    Raises
    ------
    ValueError
        If start_region is None or not in env.regions.
    ValueError
        If end_regions is None, empty, or any region not in env.regions.
    ValueError
        If trajectory_bins and times have different lengths.
    ValueError
        If min_duration is not positive.
    ValueError
        If max_duration < min_duration.

    See Also
    --------
    detect_region_crossings : Detect individual region crossings
    detect_runs_between_regions : Detect runs between two specific regions

    Notes
    -----
    Trial segmentation is fundamental for analyzing spatial navigation tasks:

    **T-maze**: start_region='start', end_regions=['left', 'right']
    - Analyzes spatial alternation behavior
    - Learning curves: fraction correct over trials
    - Strategy changes: random → alternation → place-based

    **Y-maze**: start_region='center', end_regions=['arm1', 'arm2', 'arm3']
    - Spontaneous alternation (working memory)
    - Arm entry sequences and patterns
    - Exploration vs exploitation

    **Radial arm maze**: start_region='center', end_regions=['arm1', ..., 'arm8']
    - Working and reference memory tasks
    - Optimal foraging strategies
    - Error analysis (revisits, skipped arms)

    Trials are filtered by duration to exclude:
    - Very brief entries/exits (min_duration filter)
    - Stuck or disengaged periods (max_duration timeout)

    If an animal re-enters the start region before reaching an end region,
    this is treated as a new trial (previous trial is discarded or timed out
    depending on duration).

    Examples
    --------
    >>> from neurospatial import Environment
    >>> from shapely.geometry import Point
    >>> import numpy as np
    >>> # Create T-maze environment
    >>> x = np.linspace(0, 100, 100)
    >>> y = np.linspace(0, 100, 100)
    >>> xx, yy = np.meshgrid(x, y)
    >>> positions = np.column_stack([xx.ravel(), yy.ravel()])
    >>> env = Environment.from_samples(positions, bin_size=3.0)
    >>> # Define regions: start at bottom, left/right at top
    >>> _ = env.regions.add("start", polygon=Point(50.0, 10.0).buffer(8.0))
    >>> _ = env.regions.add("left", polygon=Point(20.0, 80.0).buffer(8.0))
    >>> _ = env.regions.add("right", polygon=Point(80.0, 80.0).buffer(8.0))
    >>> # Create trajectory with 2 trials: start→left, start→right
    >>> x_traj = np.concatenate(
    ...     [
    ...         np.linspace(50, 50, 10),  # In start
    ...         np.linspace(50, 20, 15),  # Go to left
    ...         np.linspace(20, 50, 15),  # Return to start
    ...         np.linspace(50, 80, 15),  # Go to right
    ...     ]
    ... )
    >>> y_traj = np.concatenate(
    ...     [
    ...         np.linspace(10, 10, 10),
    ...         np.linspace(10, 80, 15),
    ...         np.linspace(80, 10, 15),
    ...         np.linspace(10, 80, 15),
    ...     ]
    ... )
    >>> trajectory = np.column_stack([x_traj, y_traj])
    >>> trajectory_bins = env.bin_at(trajectory)
    >>> times = np.arange(len(trajectory), dtype=float)
    >>> # Segment into trials
    >>> trials = segment_trials(
    ...     trajectory_bins,
    ...     times,
    ...     env,
    ...     start_region="start",
    ...     end_regions=["left", "right"],
    ...     min_duration=5.0,
    ...     max_duration=50.0,
    ... )
    >>> len(trials) >= 1  # Should detect at least one trial
    True
    >>> all(t.success for t in trials)  # Both should be successful
    True
    >>> end_regions = [t.end_region for t in trials]
    >>> "left" in end_regions and "right" in end_regions  # Both choices present
    True

    References
    ----------
    .. [1] Olton, D. S., & Samuelson, R. J. (1976). Remembrance of places
           passed: Spatial memory in rats. Journal of Experimental Psychology:
           Animal Behavior Processes, 2(2), 97-116.
    .. [2] Wood, E. R., Dudchenko, P. A., Robitsek, R. J., & Eichenbaum, H.
           (2000). Hippocampal neurons encode information about different
           types of memory episodes occurring in the same location. Neuron,
           27(3), 623-633.
    """
    # Validate inputs
    if start_region is None:
        raise ValueError("start_region must be provided")

    if start_region not in env.regions:
        available = list(env.regions.keys())
        raise ValueError(
            f"start_region '{start_region}' not found in environment. "
            f"Available regions: {available}"
        )

    if end_regions is None:
        raise ValueError("end_regions must be provided")

    if len(end_regions) == 0:
        raise ValueError("end_regions cannot be empty")

    for region in end_regions:
        if region not in env.regions:
            available = list(env.regions.keys())
            raise ValueError(
                f"end_regions contains '{region}' which is not found in environment. "
                f"Available regions: {available}"
            )

    # Prevent start_region from being in end_regions (typical neuroscience practice)
    if start_region in end_regions:
        raise ValueError(
            f"start_region '{start_region}' cannot be in end_regions. "
            f"Start and end regions must be spatially distinct for trial segmentation."
        )

    if len(trajectory_bins) != len(times):
        raise ValueError(
            f"trajectory_bins and times must have same length. "
            f"Got {len(trajectory_bins)} and {len(times)}"
        )

    if min_duration <= 0:
        raise ValueError(f"min_duration must be positive, got {min_duration}")

    if max_duration <= min_duration:
        raise ValueError(
            f"max_duration must be greater than min_duration. "
            f"Got max_duration={max_duration}, min_duration={min_duration}"
        )

    if len(trajectory_bins) == 0:
        return []

    # Get region masks using existing functionality
    from neurospatial.ops.binning import regions_to_mask

    start_mask = regions_to_mask(env, [start_region])
    end_masks = {region: regions_to_mask(env, [region]) for region in end_regions}

    # Check which trajectory samples are in which regions
    in_start = start_mask[trajectory_bins]
    in_end_regions = {
        region: mask[trajectory_bins] for region, mask in end_masks.items()
    }

    # Segment into trials
    trials: list[Trial] = []
    trial_start_idx: int | None = None
    trial_start_time: float | None = None

    for i in range(len(trajectory_bins)):
        # Check if entering start region (trial initiation)
        if in_start[i] and (i == 0 or not in_start[i - 1]):
            # If we had a previous trial in progress, it timed out
            if trial_start_idx is not None and trial_start_time is not None:
                duration = times[i - 1] - trial_start_time
                if duration >= max_duration and duration >= min_duration:
                    # Previous trial timed out
                    trials.append(
                        Trial(
                            start_time=trial_start_time,
                            end_time=times[i - 1],
                            start_region=start_region,
                            end_region=None,
                            success=False,
                        )
                    )

            # Start new trial
            trial_start_idx = i
            trial_start_time = times[i]

        # If we're in a trial, check for end region entry
        if trial_start_idx is not None and trial_start_time is not None:
            # Check if we entered any end region
            for region, in_region in in_end_regions.items():
                if in_region[i] and (i == 0 or not in_region[i - 1]):
                    # Reached end region
                    duration = times[i] - trial_start_time

                    # Check duration bounds
                    if duration >= min_duration and duration <= max_duration:
                        trials.append(
                            Trial(
                                start_time=trial_start_time,
                                end_time=times[i],
                                start_region=start_region,
                                end_region=region,
                                success=True,
                            )
                        )

                    # Reset trial tracking
                    trial_start_idx = None
                    trial_start_time = None
                    break

            # Check for timeout (still in trial at this point)
            if trial_start_idx is not None and trial_start_time is not None:
                duration = times[i] - trial_start_time
                if duration >= max_duration:
                    # Trial timed out
                    if duration >= min_duration:
                        trials.append(
                            Trial(
                                start_time=trial_start_time,
                                end_time=times[i],
                                start_region=start_region,
                                end_region=None,
                                success=False,
                            )
                        )
                    # Reset trial tracking
                    trial_start_idx = None
                    trial_start_time = None

    # Handle final trial if still in progress
    if trial_start_idx is not None and trial_start_time is not None:
        duration = times[-1] - trial_start_time
        if duration >= max_duration and duration >= min_duration:
            # Final trial timed out
            trials.append(
                Trial(
                    start_time=trial_start_time,
                    end_time=times[-1],
                    start_region=start_region,
                    end_region=None,
                    success=False,
                )
            )

    return trials
