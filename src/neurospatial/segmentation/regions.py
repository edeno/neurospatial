"""Region-based trajectory segmentation functions.

This module provides functions for segmenting trajectories based on spatial
regions and velocity thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from neurospatial import Environment


@dataclass(frozen=True)
class Crossing:
    """Region crossing event.

    Attributes
    ----------
    time : float
        Time of crossing event (seconds).
    direction : str
        Direction of crossing: 'entry' or 'exit'.
    bin_index : int
        Bin index at crossing.
    """

    time: float
    direction: Literal["entry", "exit"]
    bin_index: int


@dataclass(frozen=True)
class Run:
    """Run between regions.

    Attributes
    ----------
    start_time : float
        Start time of run (seconds).
    end_time : float
        End time of run (seconds).
    bins : NDArray[np.int64]
        Sequence of bin indices during run.
    success : bool
        True if run reached target region, False if timeout.
    """

    start_time: float
    end_time: float
    bins: NDArray[np.int64]
    success: bool


def detect_region_crossings(
    trajectory_bins: NDArray[np.int64],
    times: NDArray[np.float64],
    region_name: str,
    env: Environment,
    *,
    direction: Literal["both", "entry", "exit"] = "both",
) -> list[Crossing]:
    """Detect region entry and exit events in a trajectory.

    Identifies time points where the trajectory enters or exits a named region.
    Useful for analyzing exploration patterns, region preference, and behavioral
    transitions.

    Parameters
    ----------
    trajectory_bins : NDArray[np.int64], shape (n_samples,)
        Sequence of bin indices representing the trajectory.
    times : NDArray[np.float64], shape (n_samples,)
        Time stamps corresponding to trajectory bins (seconds).
    region_name : str
        Name of region to detect crossings for. Must exist in env.regions.
    env : Environment
        Environment containing the region definition.
    direction : {'both', 'entry', 'exit'}, optional
        Which crossings to detect:
        - 'both': detect entries and exits (default)
        - 'entry': only detect entries
        - 'exit': only detect exits

    Returns
    -------
    list[Crossing]
        List of crossing events, each with time, direction, and bin_index.
        Sorted by time in ascending order.

    Raises
    ------
    ValueError
        If region_name not in env.regions.
    ValueError
        If trajectory_bins and times have different lengths.

    See Also
    --------
    detect_runs_between_regions : Detect runs from source to target regions
    segment_by_velocity : Segment trajectory by velocity threshold

    Notes
    -----
    A crossing is detected when the trajectory transitions from outside to inside
    the region (entry) or inside to outside (exit). The crossing time is assigned
    to the first sample inside (for entry) or outside (for exit) the region.

    Examples
    --------
    >>> from neurospatial import Environment
    >>> from shapely.geometry import Point
    >>> import numpy as np
    >>> # Create environment and add region
    >>> positions = np.linspace(0, 100, 100)[:, None]
    >>> env = Environment.from_samples(positions, bin_size=2.0)
    >>> env.regions.add("goal", polygon=Point(50.0, 0.0).buffer(10.0))
    >>> # Create trajectory that crosses region
    >>> trajectory = np.array([10.0, 30.0, 50.0, 70.0, 50.0, 30.0])[:, None]
    >>> trajectory_bins = env.bin_at(trajectory)
    >>> times = np.arange(len(trajectory), dtype=float)
    >>> # Detect crossings
    >>> crossings = detect_region_crossings(
    ...     trajectory_bins, times, "goal", env, direction="both"
    ... )
    >>> len(crossings) > 0  # Should detect entries and exits
    True
    """
    # Validate inputs
    if region_name not in env.regions:
        available = list(env.regions.keys())
        raise ValueError(
            f"Region '{region_name}' not found in environment. "
            f"Available regions: {available}"
        )

    if len(trajectory_bins) != len(times):
        raise ValueError(
            f"trajectory_bins and times must have same length. "
            f"Got {len(trajectory_bins)} and {len(times)}"
        )

    if len(trajectory_bins) == 0:
        return []

    # Get bins in region using existing regions_to_mask functionality
    from neurospatial.ops.binning import regions_to_mask

    region_mask = regions_to_mask(env, [region_name])

    # Check which trajectory samples are in region
    in_region = region_mask[trajectory_bins]

    # Detect transitions
    crossings = []

    for i in range(1, len(in_region)):
        was_in = in_region[i - 1]
        is_in = in_region[i]

        # Entry: transition from outside to inside
        if not was_in and is_in and direction in ["both", "entry"]:
            crossings.append(
                Crossing(time=times[i], direction="entry", bin_index=trajectory_bins[i])
            )

        # Exit: transition from inside to outside
        elif was_in and not is_in and direction in ["both", "exit"]:
            crossings.append(
                Crossing(time=times[i], direction="exit", bin_index=trajectory_bins[i])
            )

    return crossings


def detect_runs_between_regions(
    trajectory_positions: NDArray[np.float64],
    times: NDArray[np.float64],
    env: Environment,
    *,
    source: str,
    target: str,
    min_duration: float = 0.5,
    max_duration: float = 10.0,
    velocity_threshold: float | None = None,
) -> list[Run]:
    """Detect runs from source region to target region.

    Identifies behavioral epochs where the animal exits the source region,
    travels through the environment, and either reaches the target region
    (successful run) or times out (failed run). Useful for analyzing
    goal-directed navigation, learning trajectories, and spatial strategies.

    Parameters
    ----------
    trajectory_positions : NDArray[np.float64], shape (n_samples, n_dims)
        Continuous position samples (e.g., in cm).
    times : NDArray[np.float64], shape (n_samples,)
        Time stamps corresponding to positions (seconds).
    env : Environment
        Environment containing source and target region definitions.
    source : str
        Name of source region (start point).
    target : str
        Name of target region (goal).
    min_duration : float, optional
        Minimum run duration in seconds. Default: 0.5.
        Runs shorter than this are excluded.
    max_duration : float, optional
        Maximum run duration in seconds. Default: 10.0.
        Runs exceeding this are marked as failed (timeout).
    velocity_threshold : float or None, optional
        Minimum velocity (units/second) to count as movement.
        If None, no velocity filtering is applied. Default: None.

    Returns
    -------
    list[Run]
        List of detected runs. Each run has:
        - start_time: time of source region exit
        - end_time: time of target entry or timeout
        - bins: sequence of bin indices during run
        - success: True if reached target, False if timeout

    Raises
    ------
    ValueError
        If source or target regions not in env.regions.
    ValueError
        If trajectory_positions and times have different lengths.

    See Also
    --------
    detect_region_crossings : Detect individual region crossings
    segment_by_velocity : Segment by velocity threshold

    Notes
    -----
    A run is defined as:
    1. Exit from source region
    2. Trajectory through environment
    3. Either entry to target (success=True) or timeout (success=False)

    Runs are filtered by:
    - Duration must be >= min_duration and <= max_duration
    - Optional velocity threshold (if specified)

    This function is commonly used in:
    - Spatial alternation tasks (T-maze, Y-maze)
    - Goal-directed navigation experiments
    - Trajectory analysis for learning curves
    - Replay detection (theta sequences, sharp-wave ripples)

    Examples
    --------
    >>> from neurospatial import Environment
    >>> from shapely.geometry import Point
    >>> import numpy as np
    >>> # Create environment with source and target
    >>> positions = np.linspace(0, 100, 200)[:, None]
    >>> env = Environment.from_samples(positions, bin_size=2.0)
    >>> env.regions.add("start", polygon=Point(10.0, 0.0).buffer(5.0))
    >>> env.regions.add("goal", polygon=Point(90.0, 0.0).buffer(5.0))
    >>> # Create run trajectory
    >>> trajectory = np.linspace(10.0, 90.0, 100)[:, None]
    >>> times = np.linspace(0, 5.0, 100)
    >>> # Detect runs
    >>> runs = detect_runs_between_regions(
    ...     trajectory,
    ...     times,
    ...     env,
    ...     source="start",
    ...     target="goal",
    ...     min_duration=0.5,
    ...     max_duration=10.0,
    ... )
    >>> len(runs) > 0  # Should detect successful run
    True
    """
    # Validate inputs
    if source not in env.regions:
        available = list(env.regions.keys())
        raise ValueError(
            f"Source region '{source}' not found. Available regions: {available}"
        )

    if target not in env.regions:
        available = list(env.regions.keys())
        raise ValueError(
            f"Target region '{target}' not found. Available regions: {available}"
        )

    if len(trajectory_positions) != len(times):
        raise ValueError(
            f"trajectory_positions and times must have same length. "
            f"Got {len(trajectory_positions)} and {len(times)}"
        )

    if len(trajectory_positions) == 0:
        return []

    # Map positions to bins
    trajectory_bins = env.bin_at(trajectory_positions)

    # Get region masks
    from neurospatial.ops.binning import regions_to_mask

    source_mask = regions_to_mask(env, [source])
    target_mask = regions_to_mask(env, [target])

    # Check which samples are in each region
    in_source = source_mask[trajectory_bins]
    in_target = target_mask[trajectory_bins]

    # Detect source exits
    source_exits = []
    for i in range(1, len(in_source)):
        if in_source[i - 1] and not in_source[i]:  # Exit source
            source_exits.append(i)

    # For each exit, track trajectory until target entry or timeout
    runs = []

    for exit_idx in source_exits:
        start_time = times[exit_idx]
        run_bins = [trajectory_bins[exit_idx]]

        # Track trajectory
        reached_target = False
        end_idx = exit_idx

        for j in range(exit_idx + 1, len(trajectory_bins)):
            elapsed = times[j] - start_time

            # Check timeout
            if elapsed > max_duration:
                break

            run_bins.append(trajectory_bins[j])

            # Check if reached target
            if in_target[j]:
                reached_target = True
                end_idx = j
                break

            end_idx = j

        end_time = times[end_idx]
        duration = end_time - start_time

        # Filter by duration
        if duration < min_duration:
            continue

        # Optional velocity filter
        if velocity_threshold is not None:
            # Compute velocity during run
            run_positions = trajectory_positions[exit_idx : end_idx + 1]
            run_times = times[exit_idx : end_idx + 1]

            if len(run_times) > 1:
                displacements = np.diff(run_positions, axis=0)
                distances = np.linalg.norm(displacements, axis=1)
                dt = np.diff(run_times)
                velocities = distances / dt
                mean_velocity = np.mean(velocities)

                if mean_velocity < velocity_threshold:
                    continue

        # Create run
        runs.append(
            Run(
                start_time=start_time,
                end_time=end_time,
                bins=np.array(run_bins, dtype=np.int64),
                success=reached_target,
            )
        )

    return runs


def segment_by_velocity(
    trajectory_positions: NDArray[np.float64],
    times: NDArray[np.float64],
    threshold: float,
    *,
    min_duration: float = 0.5,
    hysteresis: float = 2.0,
    smooth_window: float = 0.2,
) -> list[tuple[float, float]]:
    """Segment trajectory into movement and rest periods based on velocity.

    Uses hysteresis thresholding to classify trajectory epochs as movement
    (velocity above threshold) or rest (velocity below threshold). Filters
    out brief segments shorter than min_duration. Useful for identifying
    behavioral states, analyzing exploration patterns, and preprocessing
    for place field analysis.

    Parameters
    ----------
    trajectory_positions : NDArray[np.float64], shape (n_samples, n_dims)
        Continuous position samples (e.g., in cm).
    times : NDArray[np.float64], shape (n_samples,)
        Time stamps corresponding to positions (seconds).
    threshold : float
        Velocity threshold for movement classification (units/second).
        Samples with velocity > threshold are considered movement.
    min_duration : float, optional
        Minimum segment duration in seconds. Default: 0.5.
        Segments shorter than this are excluded.
    hysteresis : float, optional
        Hysteresis factor for threshold. Default: 2.0.
        - Movement starts when velocity > threshold
        - Movement ends when velocity < threshold / hysteresis
        Hysteresis prevents rapid switching near threshold.
    smooth_window : float, optional
        Temporal window for velocity smoothing in seconds. Default: 0.2.
        Velocities are smoothed with a moving average to reduce noise.

    Returns
    -------
    list[tuple[float, float]]
        List of movement segments as (start_time, end_time) tuples.
        Times are in seconds. Segments are sorted chronologically.

    Raises
    ------
    ValueError
        If trajectory_positions and times have different lengths.
    ValueError
        If threshold <= 0 or hysteresis <= 1.

    See Also
    --------
    detect_runs_between_regions : Detect runs between spatial regions
    detect_region_crossings : Detect region entry/exit events

    Notes
    -----
    Velocity is computed as Euclidean distance between consecutive samples
    divided by time difference. Velocities are smoothed using a moving average
    to reduce noise from measurement errors.

    Hysteresis thresholding prevents "flickering" near the threshold:
    - Movement starts when velocity exceeds threshold
    - Movement continues until velocity drops below threshold/hysteresis
    - This creates a "buffer zone" for stable state transitions

    Common use cases:
    - Preprocessing for place field analysis (exclude stationary periods)
    - Identifying exploration vs. exploitation epochs
    - Detecting theta sequences (movement) vs. sharp-wave ripples (rest)
    - Computing speed-filtered occupancy maps

    Examples
    --------
    >>> import numpy as np
    >>> # Create trajectory with clear movement and rest
    >>> rest1 = np.zeros((50, 1))
    >>> movement = np.linspace(0, 100, 100)[:, None]
    >>> rest2 = np.ones((50, 1)) * 100
    >>> trajectory = np.vstack([rest1, movement, rest2])
    >>> times = np.linspace(0, 20, len(trajectory))
    >>> # Segment by velocity
    >>> segments = segment_by_velocity(
    ...     trajectory, times, threshold=2.0, min_duration=0.5
    ... )
    >>> len(segments) > 0  # Should detect movement period
    True
    >>> # Each segment is (start_time, end_time)
    >>> for start, end in segments:
    ...     duration = end - start
    ...     assert duration >= 0.5  # min_duration enforced
    """
    # Validate inputs
    if len(trajectory_positions) != len(times):
        raise ValueError(
            f"trajectory_positions and times must have same length. "
            f"Got {len(trajectory_positions)} and {len(times)}"
        )

    if threshold <= 0:
        raise ValueError(f"threshold must be positive. Got {threshold}")

    if hysteresis <= 1.0:
        raise ValueError(f"hysteresis must be > 1.0 for stability. Got {hysteresis}")

    if len(trajectory_positions) < 2:
        return []

    # Compute velocities
    displacements = np.diff(trajectory_positions, axis=0)
    distances = np.linalg.norm(displacements, axis=1)
    dt = np.diff(times)
    velocities = distances / dt

    # Smooth velocities
    if smooth_window > 0:
        # Convert window to samples
        median_dt = np.median(dt)
        window_samples = max(1, int(smooth_window / median_dt))

        # Apply moving average
        if window_samples > 1:
            kernel = np.ones(window_samples) / window_samples
            velocities = np.convolve(velocities, kernel, mode="same")

    # Hysteresis thresholding
    lower_threshold = threshold / hysteresis
    is_moving = np.zeros(len(velocities), dtype=bool)
    currently_moving = False

    for i in range(len(velocities)):
        if currently_moving:
            # Continue moving until velocity drops below lower threshold
            if velocities[i] < lower_threshold:
                currently_moving = False
            else:
                is_moving[i] = True
        else:
            # Start moving when velocity exceeds threshold
            if velocities[i] > threshold:
                currently_moving = True
                is_moving[i] = True

    # Detect segments (note: velocities correspond to transitions, so index times[1:])
    segments = []
    in_segment = False
    segment_start = None

    for i in range(len(is_moving)):
        time_idx = i + 1  # Velocity at i corresponds to transition from i to i+1

        if is_moving[i] and not in_segment:
            # Start new segment
            in_segment = True
            segment_start = times[time_idx]

        elif not is_moving[i] and in_segment:
            # End segment
            segment_end = times[time_idx]
            assert segment_start is not None  # Type narrowing for mypy
            duration = segment_end - segment_start

            if duration >= min_duration:
                segments.append((segment_start, segment_end))

            in_segment = False

    # Handle case where trajectory ends while moving
    if in_segment:
        segment_end = times[-1]
        assert segment_start is not None  # Type narrowing for mypy
        duration = segment_end - segment_start

        if duration >= min_duration:
            segments.append((segment_start, segment_end))

    return segments
