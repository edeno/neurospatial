"""Behavioral segmentation and trajectory analysis.

This module provides functions for segmenting trajectories based on regions,
velocity, and behavioral epochs.

Imports
-------
All segmentation functions are accessible from this module:

>>> from neurospatial.behavior.segmentation import (
...     Crossing,
...     Lap,
...     Run,
...     Trial,
...     detect_laps,
...     detect_region_crossings,
...     detect_runs_between_regions,
...     detect_goal_directed_runs,
...     segment_by_velocity,
...     segment_trials,
...     trajectory_similarity,
... )

Or import from the behavior package:

>>> from neurospatial.behavior import (
...     Crossing,
...     Lap,
...     Run,
...     Trial,
...     detect_laps,
...     detect_region_crossings,
...     segment_trials,
... )

Functions
---------
detect_region_crossings
    Detect entry and exit events for a spatial region
detect_runs_between_regions
    Detect runs from source region to target region
segment_by_velocity
    Segment trajectory into movement and rest periods
detect_laps
    Detect laps on circular tracks
segment_trials
    Segment trajectory into behavioral trials (T-maze, Y-maze, etc.)
trajectory_similarity
    Compare similarity between two trajectories
detect_goal_directed_runs
    Detect goal-directed navigation segments

Classes
-------
Trial
    Dataclass representing a behavioral trial with fields:
    - start_time: Trial onset time
    - end_time: Trial offset time
    - start_region: Region where trial started
    - end_region: Region reached (None if timeout)
    - success: True if trial reached end_region
Crossing
    Dataclass for region entry/exit events
Lap
    Dataclass for lap detection results
Run
    Dataclass for runs between regions
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import directed_hausdorff, euclidean
from scipy.stats import pearsonr

if TYPE_CHECKING:
    from neurospatial import Environment


# =============================================================================
# Dataclasses from regions.py
# =============================================================================


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


# =============================================================================
# Dataclass from laps.py
# =============================================================================


@dataclass(frozen=True)
class Lap:
    """Detected lap on circular track.

    Attributes
    ----------
    start_time : float
        Start time of lap (seconds).
    end_time : float
        End time of lap (seconds).
    direction : str
        Direction of lap: 'clockwise', 'counter-clockwise', or 'unknown'.
    overlap_score : float
        Jaccard overlap with template lap [0, 1].
    """

    start_time: float
    end_time: float
    direction: Literal["clockwise", "counter-clockwise", "unknown"]
    overlap_score: float


# =============================================================================
# Dataclass from trials.py
# =============================================================================


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


# =============================================================================
# Functions from regions.py
# =============================================================================


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
    >>> # Create 2D environment and add region
    >>> x = np.linspace(0, 100, 100)
    >>> y = np.linspace(0, 100, 100)
    >>> positions = np.column_stack([x, y])
    >>> env = Environment.from_samples(positions, bin_size=5.0)
    >>> _ = env.regions.add("goal", polygon=Point(50.0, 50.0).buffer(10.0))
    >>> # Create trajectory that crosses region
    >>> traj_x = np.array([10.0, 30.0, 50.0, 70.0, 50.0, 30.0])
    >>> traj_y = np.array([50.0, 50.0, 50.0, 50.0, 50.0, 50.0])
    >>> trajectory = np.column_stack([traj_x, traj_y])
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

    # Detect transitions using vectorized diff operation
    # Convert bool to int: in_region[i] - in_region[i-1]
    # +1 = entry (False -> True), -1 = exit (True -> False)
    transitions = np.diff(in_region.astype(np.int8))

    # Find entry and exit indices
    crossings = []

    if direction in ["both", "entry"]:
        entry_indices = np.where(transitions == 1)[0] + 1  # +1 because diff shifts by 1
        for idx in entry_indices:
            crossings.append(
                Crossing(
                    time=times[idx], direction="entry", bin_index=trajectory_bins[idx]
                )
            )

    if direction in ["both", "exit"]:
        exit_indices = np.where(transitions == -1)[0] + 1
        for idx in exit_indices:
            crossings.append(
                Crossing(
                    time=times[idx], direction="exit", bin_index=trajectory_bins[idx]
                )
            )

    # Sort by time if we collected both entry and exit
    if direction == "both" and len(crossings) > 1:
        crossings.sort(key=lambda c: c.time)

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
    >>> from neurospatial import Environment  # doctest: +SKIP
    >>> from shapely.geometry import Point  # doctest: +SKIP
    >>> import numpy as np  # doctest: +SKIP
    >>> x = np.linspace(0, 100, 200)  # doctest: +SKIP
    >>> y = np.linspace(0, 100, 200)  # doctest: +SKIP
    >>> positions = np.column_stack([x, y])  # doctest: +SKIP
    >>> env = Environment.from_samples(positions, bin_size=5.0)  # doctest: +SKIP
    >>> _ = env.regions.add(
    ...     "start", polygon=Point(10.0, 50.0).buffer(5.0)
    ... )  # doctest: +SKIP
    >>> _ = env.regions.add(
    ...     "goal", polygon=Point(90.0, 50.0).buffer(5.0)
    ... )  # doctest: +SKIP
    >>> traj_x = np.linspace(10.0, 90.0, 100)  # doctest: +SKIP
    >>> traj_y = np.ones(100) * 50.0  # doctest: +SKIP
    >>> trajectory = np.column_stack([traj_x, traj_y])  # doctest: +SKIP
    >>> times = np.linspace(0, 5.0, 100)  # doctest: +SKIP
    >>> runs = detect_runs_between_regions(  # doctest: +SKIP
    ...     trajectory,  # doctest: +SKIP
    ...     times,  # doctest: +SKIP
    ...     env,  # doctest: +SKIP
    ...     source="start",  # doctest: +SKIP
    ...     target="goal",  # doctest: +SKIP
    ...     min_duration=0.5,  # doctest: +SKIP
    ...     max_duration=10.0,  # doctest: +SKIP
    ... )  # doctest: +SKIP
    >>> len(runs) > 0  # doctest: +SKIP
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


# =============================================================================
# Functions from laps.py
# =============================================================================


def _compute_overlap_jaccard(
    bins1: NDArray[np.int64], bins2: NDArray[np.int64]
) -> float:
    """Compute Jaccard overlap between two bin sequences.

    Parameters
    ----------
    bins1 : NDArray[np.int64]
        First sequence of bin indices.
    bins2 : NDArray[np.int64]
        Second sequence of bin indices.

    Returns
    -------
    float
        Jaccard coefficient: |intersection| / |union|, range [0, 1].
    """
    if len(bins1) == 0 or len(bins2) == 0:
        return 0.0

    set1 = set(bins1)
    set2 = set(bins2)

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    if union == 0:
        return 0.0

    return intersection / union


def _detect_lap_direction(
    bin_centers: NDArray[np.float64], bins: NDArray[np.int64]
) -> Literal["clockwise", "counter-clockwise", "unknown"]:
    """Detect lap direction using signed area.

    Parameters
    ----------
    bin_centers : NDArray[np.float64], shape (n_bins, n_dims)
        Bin center coordinates.
    bins : NDArray[np.int64], shape (n_samples,)
        Sequence of bin indices forming the lap.

    Returns
    -------
    str
        'clockwise', 'counter-clockwise', or 'unknown'.
    """
    if len(bins) < 3:
        return "unknown"

    # Get trajectory positions
    positions = bin_centers[bins]

    # For 2D, compute signed area using shoelace formula
    if positions.shape[1] == 2:
        x = positions[:, 0]
        y = positions[:, 1]
        # Signed area = 0.5 * sum(x[i] * y[i+1] - x[i+1] * y[i])
        area = 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])

        if area > 0:
            return "counter-clockwise"
        elif area < 0:
            return "clockwise"
        else:
            return "unknown"
    else:
        # For non-2D, cannot determine direction
        return "unknown"


def detect_laps(
    trajectory_bins: NDArray[np.int64],
    times: NDArray[np.float64],
    env: Environment,
    *,
    method: Literal["auto", "reference", "region"] = "auto",
    min_overlap: float = 0.8,
    direction: Literal["both", "clockwise", "counter-clockwise"] = "both",
    reference_lap: NDArray[np.int64] | None = None,
    start_region: str | None = None,
) -> list[Lap]:
    """Detect laps in a circular track trajectory.

    Identifies repeating traversals of a circular track using spatial overlap
    with a template lap. Useful for analyzing learning, spatial coding, and
    behavioral consistency across laps.

    Parameters
    ----------
    trajectory_bins : NDArray[np.int64], shape (n_samples,)
        Sequence of bin indices representing the trajectory.
    times : NDArray[np.float64], shape (n_samples,)
        Time stamps corresponding to trajectory bins (seconds).
    env : Environment
        Environment containing spatial structure.
    method : {'auto', 'reference', 'region'}, optional
        Method for detecting laps:
        - 'auto': Extract template from first 10% of trajectory (default)
        - 'reference': Use user-provided reference_lap
        - 'region': Detect laps as segments between start_region crossings
    min_overlap : float, optional
        Minimum Jaccard overlap with template to count as lap [0, 1].
        Default is 0.8. Not used for 'region' method.
    direction : {'both', 'clockwise', 'counter-clockwise'}, optional
        Which lap directions to include:
        - 'both': all laps (default)
        - 'clockwise': only clockwise laps
        - 'counter-clockwise': only counter-clockwise laps
    reference_lap : NDArray[np.int64] | None, optional
        Reference lap bin sequence for 'reference' method.
        Required when method='reference'.
    start_region : str | None, optional
        Name of start region for 'region' method.
        Required when method='region'.

    Returns
    -------
    list[Lap]
        List of detected laps, each with start_time, end_time, direction,
        and overlap_score. Sorted by start_time in ascending order.

    Raises
    ------
    ValueError
        If method is not one of 'auto', 'reference', 'region'.
    ValueError
        If method='reference' but reference_lap not provided.
    ValueError
        If method='region' but start_region not provided.
    ValueError
        If start_region not in env.regions.

    See Also
    --------
    detect_region_crossings : Detect region entry/exit events
    detect_runs_between_regions : Detect runs between regions

    Notes
    -----
    **Lap Detection Methods:**

    - **Auto**: Automatically extracts template from first 10% of trajectory,
      then searches for repeating patterns using sliding window with Jaccard
      overlap.
    - **Reference**: Uses user-provided reference lap as template, useful when
      first lap is atypical or for comparing to a canonical lap pattern.
    - **Region**: Defines laps as segments between consecutive crossings of
      a start region, useful for well-controlled tasks with clear start zones.

    **Direction Detection:**

    For 2D trajectories, direction is determined using the signed area of the
    polygon formed by the lap trajectory (shoelace formula). Positive area
    indicates counter-clockwise, negative indicates clockwise.

    **Applications:**

    - Lap-by-lap learning curves (Barnes et al., 1997)
    - Trajectory stereotypy quantification
    - Performance variability across trials
    - Spatial strategy consistency

    Examples
    --------
    Detect laps on circular track with auto template:

    >>> import numpy as np  # doctest: +SKIP
    >>> from neurospatial import Environment  # doctest: +SKIP
    >>> from neurospatial.behavior.segmentation import detect_laps  # doctest: +SKIP
    >>> theta = np.linspace(0, 4 * np.pi, 200)  # doctest: +SKIP
    >>> x = 50 + 30 * np.cos(theta)  # doctest: +SKIP
    >>> y = 50 + 30 * np.sin(theta)  # doctest: +SKIP
    >>> positions = np.column_stack([x, y])  # doctest: +SKIP
    >>> env = Environment.from_samples(positions, bin_size=3.0)  # doctest: +SKIP
    >>> trajectory_bins = env.bin_at(positions)  # doctest: +SKIP
    >>> times = np.linspace(0, 40, 200)  # doctest: +SKIP
    >>> laps = detect_laps(trajectory_bins, times, env, method="auto")  # doctest: +SKIP
    >>> len(laps) >= 1  # doctest: +SKIP
    True

    Detect laps with user-provided reference:

    >>> reference = trajectory_bins[:50]  # doctest: +SKIP
    >>> laps = detect_laps(  # doctest: +SKIP
    ...     trajectory_bins,
    ...     times,
    ...     env,
    ...     method="reference",
    ...     reference_lap=reference,  # doctest: +SKIP
    ... )  # doctest: +SKIP
    >>> all(lap.overlap_score >= 0.8 for lap in laps)  # doctest: +SKIP
    True

    Filter laps by direction:

    >>> laps_cw = detect_laps(
    ...     trajectory_bins, times, env, direction="clockwise"
    ... )  # doctest: +SKIP
    >>> laps_ccw = detect_laps(  # doctest: +SKIP
    ...     trajectory_bins,
    ...     times,
    ...     env,
    ...     direction="counter-clockwise",  # doctest: +SKIP
    ... )  # doctest: +SKIP
    >>> len(laps_cw) + len(laps_ccw) >= 0  # doctest: +SKIP
    True

    References
    ----------
    .. [1] Barnes, C. A., Suster, M. S., Shen, J., & McNaughton, B. L. (1997).
           Multistability of cognitive maps in the hippocampus of old rats.
           Nature, 388(6639), 272-275.
    .. [2] Dupret, D., O'Neill, J., Pleydell-Bouverie, B., & Csicsvari, J. (2010).
           The reorganization and reactivation of hippocampal maps predict
           spatial memory performance. Nature Neuroscience, 13(8), 995-1002.
    """
    # Validate method
    if method not in ("auto", "reference", "region"):
        raise ValueError(
            f"method must be one of 'auto', 'reference', 'region', got '{method}'"
        )

    # Validate method-specific requirements
    if method == "reference" and reference_lap is None:
        raise ValueError("reference_lap is required when method='reference'")

    if method == "region":
        if start_region is None:
            raise ValueError("start_region is required when method='region'")
        if start_region not in env.regions:
            raise ValueError(
                f"start_region '{start_region}' not in env.regions. "
                f"Available regions: {list(env.regions.keys())}"
            )

    # Handle empty trajectory
    if len(trajectory_bins) == 0:
        return []

    # Initialize laps list (type annotation here for all branches)
    laps: list[Lap] = []

    # Method-specific lap detection
    if method == "region":
        # Use region crossings to define laps
        # Type narrowing for mypy - start_region already validated above
        assert start_region is not None

        crossings = detect_region_crossings(
            trajectory_bins, times, start_region, env, direction="entry"
        )

        if len(crossings) < 2:
            return []

        # Each pair of consecutive entries defines a lap
        for i in range(len(crossings) - 1):
            start_idx = int(np.searchsorted(times, crossings[i].time))
            end_idx = int(np.searchsorted(times, crossings[i + 1].time))

            if end_idx > start_idx:
                lap_bins = trajectory_bins[start_idx:end_idx]
                lap_direction = _detect_lap_direction(env.bin_centers, lap_bins)

                # Filter by direction
                if direction != "both" and lap_direction != direction:
                    continue

                laps.append(
                    Lap(
                        start_time=crossings[i].time,
                        end_time=crossings[i + 1].time,
                        direction=lap_direction,
                        overlap_score=1.0,  # Region method doesn't use overlap
                    )
                )

        return laps

    # For 'auto' and 'reference' methods, use sliding window with overlap
    if method == "auto":
        # Extract template from first 10% of trajectory
        template_size = max(1, len(trajectory_bins) // 10)
        template = trajectory_bins[:template_size]
        search_start = template_size
    else:  # method == 'reference'
        template = reference_lap  # type: ignore[assignment]
        search_start = 0

    template_length = len(template)

    # Sliding window to find laps
    i = int(search_start)

    while i < len(trajectory_bins):
        # Try different window sizes around template length
        best_overlap = 0.0
        best_end = i

        for window_size in range(
            max(1, int(template_length * 0.7)),
            min(len(trajectory_bins) - i, int(template_length * 1.3)) + 1,
        ):
            end_idx = int(min(i + window_size, len(trajectory_bins)))
            window = trajectory_bins[i:end_idx]

            overlap = _compute_overlap_jaccard(template, window)

            if overlap > best_overlap:
                best_overlap = overlap
                best_end = int(end_idx)

        # If overlap exceeds threshold, we found a lap
        if best_overlap >= min_overlap:
            lap_bins = trajectory_bins[i:best_end]
            lap_direction = _detect_lap_direction(env.bin_centers, lap_bins)

            # Filter by direction
            if direction == "both" or lap_direction == direction:
                laps.append(
                    Lap(
                        start_time=times[i],
                        end_time=times[best_end - 1] if best_end > i else times[i],
                        direction=lap_direction,
                        overlap_score=best_overlap,
                    )
                )

            # Skip past this lap
            i = best_end
        else:
            # No lap found, move forward
            i += 1

    return laps


# =============================================================================
# Functions from trials.py
# =============================================================================


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


# =============================================================================
# Functions from similarity.py
# =============================================================================


def trajectory_similarity(
    trajectory1_bins: NDArray[np.int64],
    trajectory2_bins: NDArray[np.int64],
    env: Environment,
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
    >>> from neurospatial.behavior.segmentation import trajectory_similarity
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
    env: Environment,
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

    >>> import numpy as np  # doctest: +SKIP
    >>> from shapely.geometry import Point  # doctest: +SKIP
    >>> from neurospatial import Environment  # doctest: +SKIP
    >>> from neurospatial.behavior.segmentation import (
    ...     detect_goal_directed_runs,
    ... )  # doctest: +SKIP
    >>> positions = np.linspace(0, 100, 100)[:, None]  # doctest: +SKIP
    >>> env = Environment.from_samples(positions, bin_size=2.0)  # doctest: +SKIP
    >>> goal_center = env.bin_centers[-1]  # doctest: +SKIP
    >>> goal_polygon = Point(float(goal_center[0]), 0.0).buffer(5.0)  # doctest: +SKIP
    >>> env.regions.add("goal", polygon=goal_polygon)  # doctest: +SKIP
    >>> trajectory_bins = np.arange(0, 40, dtype=np.int64)  # doctest: +SKIP
    >>> times = np.linspace(0, 10, len(trajectory_bins))  # doctest: +SKIP
    >>> runs = detect_goal_directed_runs(  # doctest: +SKIP
    ...     trajectory_bins,  # doctest: +SKIP
    ...     times,  # doctest: +SKIP
    ...     env,  # doctest: +SKIP
    ...     goal_region="goal",  # doctest: +SKIP
    ...     directedness_threshold=0.7,  # doctest: +SKIP
    ...     min_progress=10.0,  # doctest: +SKIP
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
    from neurospatial.ops.binning import regions_to_mask

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


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Dataclasses
    "Crossing",
    "Lap",
    "Run",
    "Trial",
    # Functions
    "detect_goal_directed_runs",
    "detect_laps",
    "detect_region_crossings",
    "detect_runs_between_regions",
    "segment_by_velocity",
    "segment_trials",
    "trajectory_similarity",
]
