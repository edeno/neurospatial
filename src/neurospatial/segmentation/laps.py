"""Lap detection functions for circular track analysis.

This module provides functions for detecting and analyzing laps on circular
tracks, which is common in neuroscience experiments studying spatial navigation
and learning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from neurospatial.environment._protocols import EnvironmentProtocol


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
    env: EnvironmentProtocol,
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

    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.segmentation import detect_laps
    >>> # Create circular trajectory (2 laps)
    >>> theta = np.linspace(0, 4 * np.pi, 200)
    >>> x = 50 + 30 * np.cos(theta)
    >>> y = 50 + 30 * np.sin(theta)
    >>> positions = np.column_stack([x, y])
    >>> env = Environment.from_samples(positions, bin_size=3.0)
    >>> trajectory_bins = env.bin_at(positions)
    >>> times = np.linspace(0, 40, 200)
    >>> laps = detect_laps(trajectory_bins, times, env, method="auto")
    >>> len(laps) >= 1  # Should detect at least 1 lap
    True

    Detect laps with user-provided reference:

    >>> reference = trajectory_bins[:50]  # First quarter as template
    >>> laps = detect_laps(
    ...     trajectory_bins, times, env, method="reference", reference_lap=reference
    ... )
    >>> all(lap.overlap_score >= 0.8 for lap in laps)
    True

    Filter laps by direction:

    >>> laps_cw = detect_laps(trajectory_bins, times, env, direction="clockwise")
    >>> laps_ccw = detect_laps(
    ...     trajectory_bins, times, env, direction="counter-clockwise"
    ... )
    >>> len(laps_cw) + len(laps_ccw) >= 0
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
        from neurospatial.segmentation.regions import detect_region_crossings

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
