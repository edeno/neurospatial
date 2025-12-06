"""Object-vector cell metrics for analyzing distance and direction tuning.

Object-vector cells (OVCs) fire when an animal is at a specific distance and
direction from an object in the environment. This module provides metrics for:

- Computing 2D tuning curves in egocentric (distance, direction) space
- Quantifying object-vector selectivity scores
- Classifying neurons as object-vector cells

Which Function Should I Use?
----------------------------
**Computing tuning curve from raw data?**
    Use ``compute_object_vector_tuning()`` to compute firing rate as a function
    of egocentric distance and direction to nearest object.

**Quantifying selectivity?**
    Use ``object_vector_score()`` to compute combined distance/direction
    selectivity metric.

**Screening many neurons?**
    Use ``is_object_vector_cell()`` for fast boolean filtering.

**Visualizing tuning?**
    Use ``plot_object_vector_tuning()`` for polar heatmap visualization.

Typical Workflow
----------------
1. Compute tuning curve from spike times and behavioral data::

    >>> metrics = compute_object_vector_tuning(  # doctest: +SKIP
    ...     spike_times, times, positions, headings, object_positions, env
    ... )

2. Inspect metrics and classify::

    >>> print(metrics)  # Human-readable interpretation  # doctest: +SKIP
    >>> if is_object_vector_cell(metrics.tuning_curve, metrics.peak_rate):
    ...     print(f"OVC! Preferred distance: {metrics.preferred_distance:.1f}")

3. Visualize::

    >>> plot_object_vector_tuning(  # doctest: +SKIP
    ...     metrics.tuning_curve, metrics.distance_bins, metrics.direction_bins
    ... )

Coordinate Conventions
----------------------
**Egocentric direction**:
- 0 radians = object is directly ahead of animal
- pi/2 radians = object is to the left
- -pi/2 radians = object is to the right
- +/-pi radians = object is behind

This matches the coordinate convention in ``neurospatial.reference_frames``.

References
----------
Hoydal, O. A., et al. (2019). Object-vector coding in the medial entorhinal
    cortex. Nature, 568(7752), 400-404.
Deshmukh, S. S., & Knierim, J. J. (2011). Representation of non-spatial and
    spatial information in the lateral entorhinal cortex. Frontiers in
    Behavioral Neuroscience, 5, 69.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from neurospatial.ops.egocentric import compute_egocentric_bearing
from neurospatial.stats.circular import _mean_resultant_length

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.projections.polar import PolarAxes

    from neurospatial import Environment
    from neurospatial.environment._protocols import EnvironmentProtocol

__all__ = [
    "ObjectVectorFieldResult",
    "ObjectVectorMetrics",
    "compute_object_vector_field",
    "compute_object_vector_tuning",
    "is_object_vector_cell",
    "object_vector_score",
    "plot_object_vector_tuning",
]


@dataclass(frozen=True)
class ObjectVectorMetrics:
    """Metrics for object-vector cell analysis.

    Attributes
    ----------
    preferred_distance : float
        Distance with peak firing rate (environment units).
    preferred_direction : float
        Direction with peak firing rate (radians, egocentric).
        Convention: 0=ahead, pi/2=left, -pi/2=right.
    distance_selectivity : float
        Distance selectivity: peak rate / mean rate.
        Higher values indicate sharper distance tuning.
    direction_selectivity : float
        Direction selectivity: mean resultant length of direction tuning.
        Range [0, 1]. Higher values indicate sharper direction tuning.
    object_vector_score : float
        Combined object-vector score in [0, 1].
        Combines distance and direction selectivity.
    peak_rate : float
        Maximum firing rate (Hz) in tuning curve.
    mean_rate : float
        Mean firing rate (Hz) across all bins.
    tuning_curve : NDArray[np.float64]
        2D tuning curve, shape (n_distance_bins, n_direction_bins).
        Values are firing rates in Hz.
    distance_bins : NDArray[np.float64]
        Distance bin edges, shape (n_distance_bins + 1,).
    direction_bins : NDArray[np.float64]
        Direction bin edges, shape (n_direction_bins + 1,).

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.metrics.object_vector_cells import ObjectVectorMetrics
    >>> metrics = ObjectVectorMetrics(
    ...     preferred_distance=10.0,
    ...     preferred_direction=0.0,
    ...     distance_selectivity=3.0,
    ...     direction_selectivity=0.6,
    ...     object_vector_score=0.8,
    ...     peak_rate=20.0,
    ...     mean_rate=5.0,
    ...     tuning_curve=np.ones((10, 12)),
    ...     distance_bins=np.linspace(0, 50, 11),
    ...     direction_bins=np.linspace(-np.pi, np.pi, 13),
    ... )
    >>> metrics.preferred_distance
    10.0
    """

    preferred_distance: float
    preferred_direction: float
    distance_selectivity: float
    direction_selectivity: float
    object_vector_score: float
    peak_rate: float
    mean_rate: float
    tuning_curve: NDArray[np.float64]
    distance_bins: NDArray[np.float64]
    direction_bins: NDArray[np.float64]

    def interpretation(self) -> str:
        """Human-readable interpretation of object-vector metrics.

        Returns
        -------
        str
            Multi-line interpretation.
        """
        lines = []
        pref_dir_deg = np.degrees(self.preferred_direction)

        lines.append("Object-Vector Cell Metrics")
        lines.append("-" * 30)
        lines.append(f"Preferred distance: {self.preferred_distance:.1f}")
        lines.append(f"Preferred direction: {pref_dir_deg:.1f} deg")
        lines.append("  (0=ahead, 90=left, -90=right)")
        lines.append(f"Distance selectivity: {self.distance_selectivity:.2f}")
        lines.append(f"Direction selectivity: {self.direction_selectivity:.3f}")
        lines.append(f"Object-vector score: {self.object_vector_score:.3f}")
        lines.append(f"Peak rate: {self.peak_rate:.1f} Hz")
        lines.append(f"Mean rate: {self.mean_rate:.2f} Hz")

        return "\n".join(lines)

    def __str__(self) -> str:
        """String representation with interpretation."""
        return self.interpretation()


def compute_object_vector_tuning(
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
    object_positions: NDArray[np.float64],
    env: Environment,
    *,
    max_distance: float = 50.0,
    n_distance_bins: int = 10,
    n_direction_bins: int = 12,
    min_occupancy_seconds: float = 0.1,
) -> ObjectVectorMetrics:
    """Compute object-vector tuning curve from spike data.

    Bins spikes by egocentric distance and direction to nearest object,
    then normalizes by occupancy to get firing rate.

    Parameters
    ----------
    spike_times : NDArray[np.float64], shape (n_spikes,)
        Times of spikes in the same units as ``times``.
    times : NDArray[np.float64], shape (n_time,)
        Timestamps for each behavioral sample.
    positions : NDArray[np.float64], shape (n_time, 2)
        Animal positions in allocentric coordinates.
    headings : NDArray[np.float64], shape (n_time,)
        Animal heading at each time (radians).
    object_positions : NDArray[np.float64], shape (n_objects, 2)
        Positions of objects in allocentric coordinates.
    env : Environment
        The spatial environment (used for distance calculations).
    max_distance : float, default=50.0
        Maximum distance to include in tuning curve.
    n_distance_bins : int, default=10
        Number of distance bins.
    n_direction_bins : int, default=12
        Number of direction bins (covers full circle -pi to pi).
    min_occupancy_seconds : float, default=0.1
        Minimum occupancy required in a bin. Bins with less occupancy
        are set to NaN.

    Returns
    -------
    ObjectVectorMetrics
        Dataclass with tuning curve and computed metrics.

    Raises
    ------
    ValueError
        If spike_times is empty or arrays have mismatched lengths.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.metrics.object_vector_cells import (
    ...     compute_object_vector_tuning,
    ... )
    >>> rng = np.random.default_rng(42)
    >>> samples = rng.uniform(0, 100, (500, 2))
    >>> env = Environment.from_samples(samples, bin_size=5.0)
    >>> n_time = 1000
    >>> times = np.linspace(0, 100, n_time)
    >>> positions = rng.uniform(0, 100, (n_time, 2))
    >>> headings = rng.uniform(-np.pi, np.pi, n_time)
    >>> object_positions = np.array([[50.0, 50.0]])
    >>> spike_times = rng.choice(times, size=100, replace=False)
    >>> metrics = compute_object_vector_tuning(
    ...     spike_times, times, positions, headings, object_positions, env
    ... )
    >>> metrics.tuning_curve.shape
    (10, 12)
    """
    # Convert inputs
    spike_times = np.asarray(spike_times, dtype=np.float64).ravel()
    times = np.asarray(times, dtype=np.float64).ravel()
    positions = np.asarray(positions, dtype=np.float64)
    headings = np.asarray(headings, dtype=np.float64).ravel()
    object_positions = np.asarray(object_positions, dtype=np.float64)

    # Validate inputs
    if len(spike_times) == 0:
        raise ValueError(
            "spike_times cannot be empty. Need at least one spike "
            "to compute tuning curve."
        )

    if len(times) != len(positions):
        raise ValueError(
            f"times and positions must have the same length. "
            f"Got times: {len(times)}, positions: {len(positions)}."
        )

    if len(times) != len(headings):
        raise ValueError(
            f"times and headings must have the same length. "
            f"Got times: {len(times)}, headings: {len(headings)}."
        )

    n_time = len(times)

    # Compute time step
    dt = np.median(np.diff(times))

    # Create bin edges
    distance_bins = np.linspace(0, max_distance, n_distance_bins + 1)
    direction_bins = np.linspace(-np.pi, np.pi, n_direction_bins + 1)

    # Compute distance and bearing to all objects at each timepoint
    # distances: (n_time, n_objects)
    distances = np.linalg.norm(
        positions[:, np.newaxis, :] - object_positions[np.newaxis, :, :],
        axis=2,
    )

    # bearings: (n_time, n_objects)
    bearings = compute_egocentric_bearing(object_positions, positions, headings)

    # Find nearest object at each timepoint
    nearest_obj_idx = np.argmin(distances, axis=1)
    nearest_distances = distances[np.arange(n_time), nearest_obj_idx]
    nearest_bearings = bearings[np.arange(n_time), nearest_obj_idx]

    # Compute occupancy in each (distance, direction) bin
    occupancy = np.zeros((n_distance_bins, n_direction_bins), dtype=np.float64)

    # Assign each frame to a bin
    dist_bin_idx = np.digitize(nearest_distances, distance_bins) - 1
    dir_bin_idx = np.digitize(nearest_bearings, direction_bins) - 1

    # Clip to valid range
    dist_bin_idx = np.clip(dist_bin_idx, 0, n_distance_bins - 1)
    dir_bin_idx = np.clip(dir_bin_idx, 0, n_direction_bins - 1)

    # Accumulate occupancy
    for t in range(n_time):
        occupancy[dist_bin_idx[t], dir_bin_idx[t]] += dt

    # Count spikes in each bin
    spike_counts = np.zeros((n_distance_bins, n_direction_bins), dtype=np.float64)

    # Filter spikes to valid time range
    valid_mask = (spike_times >= times[0]) & (spike_times <= times[-1])
    valid_spike_times = spike_times[valid_mask]

    if len(valid_spike_times) > 0:
        # Find nearest behavioral frame for each spike
        spike_frame_idx = np.searchsorted(times, valid_spike_times, side="right") - 1
        spike_frame_idx = np.clip(spike_frame_idx, 0, n_time - 1)

        # Get distance/direction at each spike
        spike_distances = nearest_distances[spike_frame_idx]
        spike_bearings = nearest_bearings[spike_frame_idx]

        # Assign spikes to bins
        spike_dist_bins = np.digitize(spike_distances, distance_bins) - 1
        spike_dir_bins = np.digitize(spike_bearings, direction_bins) - 1

        spike_dist_bins = np.clip(spike_dist_bins, 0, n_distance_bins - 1)
        spike_dir_bins = np.clip(spike_dir_bins, 0, n_direction_bins - 1)

        # Count spikes per bin
        for i in range(len(valid_spike_times)):
            spike_counts[spike_dist_bins[i], spike_dir_bins[i]] += 1

    # Compute firing rate (Hz) = spike count / occupancy
    tuning_curve = np.zeros((n_distance_bins, n_direction_bins), dtype=np.float64)
    sufficient_occupancy = occupancy >= min_occupancy_seconds

    tuning_curve[sufficient_occupancy] = (
        spike_counts[sufficient_occupancy] / occupancy[sufficient_occupancy]
    )
    tuning_curve[~sufficient_occupancy] = np.nan

    # Compute metrics
    valid_rates = tuning_curve[np.isfinite(tuning_curve)]

    if len(valid_rates) == 0:
        # No valid bins
        peak_rate = 0.0
        mean_rate = 0.0
        preferred_distance = float(distance_bins[n_distance_bins // 2])
        preferred_direction = 0.0
        distance_selectivity = 1.0
        direction_selectivity = 0.0
        ov_score = 0.0
    else:
        peak_rate = float(np.nanmax(tuning_curve))
        mean_rate = float(np.nanmean(tuning_curve))

        # Find peak location
        peak_idx = np.unravel_index(np.nanargmax(tuning_curve), tuning_curve.shape)
        dist_bin_centers = (distance_bins[:-1] + distance_bins[1:]) / 2
        dir_bin_centers = (direction_bins[:-1] + direction_bins[1:]) / 2

        preferred_distance = float(dist_bin_centers[peak_idx[0]])
        preferred_direction = float(dir_bin_centers[peak_idx[1]])

        # Compute selectivity metrics
        distance_selectivity = peak_rate / mean_rate if mean_rate > 0 else 1.0
        direction_selectivity = _compute_direction_selectivity(
            tuning_curve, direction_bins
        )

        ov_score = object_vector_score(tuning_curve)

    return ObjectVectorMetrics(
        preferred_distance=preferred_distance,
        preferred_direction=preferred_direction,
        distance_selectivity=distance_selectivity,
        direction_selectivity=direction_selectivity,
        object_vector_score=ov_score,
        peak_rate=peak_rate,
        mean_rate=mean_rate,
        tuning_curve=tuning_curve,
        distance_bins=distance_bins,
        direction_bins=direction_bins,
    )


def _compute_direction_selectivity(
    tuning_curve: NDArray[np.float64],
    direction_bins: NDArray[np.float64],
) -> float:
    """Compute direction selectivity using mean resultant length.

    Parameters
    ----------
    tuning_curve : NDArray, shape (n_dist, n_dir)
        Firing rate tuning curve.
    direction_bins : NDArray, shape (n_dir + 1,)
        Direction bin edges.

    Returns
    -------
    float
        Mean resultant length in [0, 1].
    """
    dir_bin_centers = (direction_bins[:-1] + direction_bins[1:]) / 2

    # Marginalize over distance (sum across distance bins)
    direction_tuning = np.nansum(tuning_curve, axis=0)

    # Normalize to valid weights
    total = np.sum(direction_tuning)
    if total == 0:
        return 0.0

    # Compute mean resultant length
    return _mean_resultant_length(dir_bin_centers, weights=direction_tuning)


def object_vector_score(
    tuning_curve: NDArray[np.float64],
    max_distance_selectivity: float = 10.0,
) -> float:
    """Compute combined object-vector selectivity score.

    The score combines distance selectivity and direction selectivity
    following the formula:

        s_OV = ((s_d - 1) / (s_d* - 1)) * s_theta

    where:
    - s_d = peak / mean (distance selectivity)
    - s_d* = max_distance_selectivity (normalization constant)
    - s_theta = mean resultant length (direction selectivity)

    Parameters
    ----------
    tuning_curve : NDArray[np.float64], shape (n_dist, n_dir)
        2D firing rate tuning curve.
    max_distance_selectivity : float, default=10.0
        Maximum expected distance selectivity for normalization.
        Must be > 1.

    Returns
    -------
    float
        Object-vector score in [0, 1].

    Raises
    ------
    ValueError
        If max_distance_selectivity <= 1.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.metrics.object_vector_cells import object_vector_score
    >>> # Sharp tuning
    >>> tc = np.zeros((10, 12)) + 0.1
    >>> tc[5, 6] = 20.0
    >>> score = object_vector_score(tc)
    >>> score > 0.5
    True
    """
    if max_distance_selectivity <= 1.0:
        raise ValueError(
            f"max_distance_selectivity must be > 1, got {max_distance_selectivity}"
        )

    tuning_curve = np.asarray(tuning_curve, dtype=np.float64)

    # Handle NaN values
    valid_mask = np.isfinite(tuning_curve)
    if not np.any(valid_mask):
        return np.nan

    valid_rates = tuning_curve[valid_mask]

    # Compute distance selectivity
    peak_rate = np.max(valid_rates)
    mean_rate = np.mean(valid_rates)

    if mean_rate == 0:
        return 0.0

    distance_selectivity = peak_rate / mean_rate

    # Normalize distance selectivity to [0, 1]
    normalized_dist_sel = (distance_selectivity - 1.0) / (
        max_distance_selectivity - 1.0
    )
    normalized_dist_sel = float(np.clip(normalized_dist_sel, 0.0, 1.0))

    # Compute direction selectivity (mean resultant length)
    n_dir = tuning_curve.shape[1]
    direction_bins = np.linspace(-np.pi, np.pi, n_dir + 1)
    dir_bin_centers = (direction_bins[:-1] + direction_bins[1:]) / 2

    # Marginalize over distance
    direction_tuning = np.nansum(tuning_curve, axis=0)
    total = np.sum(direction_tuning)

    if total == 0:
        direction_selectivity = 0.0
    else:
        direction_selectivity = _mean_resultant_length(
            dir_bin_centers, weights=direction_tuning
        )

    # Combined score
    score = normalized_dist_sel * direction_selectivity

    return float(np.clip(score, 0.0, 1.0))


def is_object_vector_cell(
    tuning_curve: NDArray[np.float64],
    peak_rate: float,
    score_threshold: float = 0.3,
    min_peak_rate: float = 5.0,
) -> bool:
    """Classify neuron as object-vector cell.

    A neuron is classified as an OVC if it has:
    1. Object-vector score above threshold
    2. Peak firing rate above minimum

    Parameters
    ----------
    tuning_curve : NDArray[np.float64], shape (n_dist, n_dir)
        2D firing rate tuning curve.
    peak_rate : float
        Peak firing rate in Hz.
    score_threshold : float, default=0.3
        Minimum object-vector score to classify as OVC.
    min_peak_rate : float, default=5.0
        Minimum peak firing rate in Hz.

    Returns
    -------
    bool
        True if classified as object-vector cell.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.metrics.object_vector_cells import is_object_vector_cell
    >>> # Sharp tuning with high rate
    >>> tc = np.zeros((10, 12)) + 0.1
    >>> tc[5, 6] = 25.0
    >>> is_object_vector_cell(tc, peak_rate=25.0, score_threshold=0.3)
    True
    """
    if peak_rate < min_peak_rate:
        return False

    score = object_vector_score(tuning_curve)

    if np.isnan(score):
        return False

    return score >= score_threshold


def plot_object_vector_tuning(
    tuning_curve: NDArray[np.float64],
    distance_bins: NDArray[np.float64],
    direction_bins: NDArray[np.float64],
    ax: Axes | PolarAxes | None = None,
    *,
    show_peak: bool = True,
    add_colorbar: bool = False,
    cmap: str = "viridis",
    **kwargs: Any,
) -> Axes | PolarAxes:
    """Plot object-vector tuning curve as polar heatmap.

    Creates a polar plot where:
    - Radial axis = distance from object
    - Angular axis = egocentric direction to object

    Parameters
    ----------
    tuning_curve : NDArray[np.float64], shape (n_dist, n_dir)
        2D firing rate tuning curve.
    distance_bins : NDArray[np.float64], shape (n_dist + 1,)
        Distance bin edges.
    direction_bins : NDArray[np.float64], shape (n_dir + 1,)
        Direction bin edges.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure with polar projection.
    show_peak : bool, default=True
        If True, mark the peak location with a marker.
    add_colorbar : bool, default=False
        If True, add a colorbar.
    cmap : str, default='viridis'
        Colormap name.
    **kwargs : dict
        Additional keyword arguments passed to pcolormesh.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object with the plot.

    Raises
    ------
    ValueError
        If tuning_curve shape doesn't match bin sizes.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.metrics.object_vector_cells import plot_object_vector_tuning
    >>> tc = np.random.default_rng(42).random((10, 12))
    >>> dist_bins = np.linspace(0, 50, 11)
    >>> dir_bins = np.linspace(-np.pi, np.pi, 13)
    >>> ax = plot_object_vector_tuning(tc, dist_bins, dir_bins)  # doctest: +SKIP
    """
    import matplotlib.pyplot as plt
    from matplotlib.projections.polar import PolarAxes

    tuning_curve = np.asarray(tuning_curve, dtype=np.float64)
    distance_bins = np.asarray(distance_bins, dtype=np.float64)
    direction_bins = np.asarray(direction_bins, dtype=np.float64)

    # Validate shapes
    n_dist, n_dir = tuning_curve.shape
    if len(distance_bins) != n_dist + 1:
        raise ValueError(
            f"distance_bins has wrong shape. Expected {n_dist + 1} edges "
            f"for {n_dist} distance bins, got {len(distance_bins)}."
        )
    if len(direction_bins) != n_dir + 1:
        raise ValueError(
            f"direction_bins has wrong shape. Expected {n_dir + 1} edges "
            f"for {n_dir} direction bins, got {len(direction_bins)}."
        )

    # Create figure if needed
    if ax is None:
        _, ax = plt.subplots(subplot_kw={"projection": "polar"})

    # Create mesh grid for polar plot
    theta, r = np.meshgrid(direction_bins, distance_bins)

    # Plot heatmap
    mesh = ax.pcolormesh(theta, r, tuning_curve, cmap=cmap, shading="flat", **kwargs)

    # Configure polar plot
    if isinstance(ax, PolarAxes):
        ax.set_theta_zero_location("N")  # 0 degrees at top (ahead)
        ax.set_theta_direction(-1)  # Clockwise

    # Mark peak if requested
    if show_peak:
        valid_mask = np.isfinite(tuning_curve)
        if np.any(valid_mask):
            # Find peak
            peak_idx = np.unravel_index(np.nanargmax(tuning_curve), tuning_curve.shape)
            dist_centers = (distance_bins[:-1] + distance_bins[1:]) / 2
            dir_centers = (direction_bins[:-1] + direction_bins[1:]) / 2

            peak_r = dist_centers[peak_idx[0]]
            peak_theta = dir_centers[peak_idx[1]]

            ax.scatter(
                [peak_theta],
                [peak_r],
                color="red",
                s=100,
                marker="*",
                zorder=5,
                label="Peak",
            )

    # Add colorbar if requested
    if add_colorbar:
        plt.colorbar(mesh, ax=ax, label="Firing rate (Hz)")

    return ax


# =============================================================================
# Object-vector field computation
# =============================================================================


@dataclass(frozen=True)
class ObjectVectorFieldResult:
    """Result of object-vector field computation.

    Attributes
    ----------
    field : NDArray[np.float64], shape (n_bins,)
        Firing rate (Hz) in each egocentric polar bin.
        NaN values indicate insufficient occupancy.
    ego_env : Environment
        Egocentric polar coordinate environment.
        - ``ego_env.bin_centers[:, 0]`` = distances
        - ``ego_env.bin_centers[:, 1]`` = directions (radians)
    occupancy : NDArray[np.float64], shape (n_bins,)
        Time spent (seconds) in each egocentric bin.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding.object_vector import ObjectVectorFieldResult
    >>> ego_env = Environment.from_polar_egocentric(
    ...     distance_range=(0.0, 50.0),
    ...     angle_range=(-np.pi, np.pi),
    ...     distance_bin_size=10.0,
    ...     angle_bin_size=np.pi / 4,
    ... )
    >>> result = ObjectVectorFieldResult(
    ...     field=np.zeros(ego_env.n_bins),
    ...     ego_env=ego_env,
    ...     occupancy=np.ones(ego_env.n_bins),
    ... )
    >>> len(result.field) == result.ego_env.n_bins
    True
    """

    field: NDArray[np.float64]
    ego_env: Environment
    occupancy: NDArray[np.float64]


def compute_object_vector_field(
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
    object_positions: NDArray[np.float64],
    *,
    max_distance: float = 50.0,
    n_distance_bins: int = 10,
    n_direction_bins: int = 12,
    min_occupancy_seconds: float = 0.1,
    smoothing_method: Literal["binned", "diffusion_kde"] = "binned",
    bandwidth: float = 5.0,
    allocentric_env: Environment | None = None,
    distance_metric: Literal["euclidean", "geodesic"] = "euclidean",
) -> ObjectVectorFieldResult:
    """Compute object-vector field in egocentric polar coordinates.

    Computes firing rate as a function of egocentric distance and direction
    to the nearest object, creating a field over an egocentric polar
    coordinate system.

    Parameters
    ----------
    spike_times : NDArray[np.float64], shape (n_spikes,)
        Times of spikes in the same units as ``times``.
    times : NDArray[np.float64], shape (n_time,)
        Timestamps for each behavioral sample.
    positions : NDArray[np.float64], shape (n_time, 2)
        Animal positions in allocentric coordinates.
    headings : NDArray[np.float64], shape (n_time,)
        Animal heading at each time (radians).
    object_positions : NDArray[np.float64], shape (n_objects, 2)
        Positions of objects in allocentric coordinates.
    max_distance : float, default=50.0
        Maximum distance to include in field. Units should match the
        position coordinates (e.g., centimeters).
    n_distance_bins : int, default=10
        Number of distance bins in the egocentric polar environment.
    n_direction_bins : int, default=12
        Number of direction bins (covers full circle -pi to pi).
    min_occupancy_seconds : float, default=0.1
        Minimum occupancy required in a bin (seconds). Bins with less
        occupancy are set to NaN.
    smoothing_method : {"binned", "diffusion_kde"}, default="binned"
        Smoothing method:
        - "binned": Simple histogram-based field
        - "diffusion_kde": Graph-smoothed field using diffusion KDE
    bandwidth : float, default=5.0
        Smoothing bandwidth for diffusion_kde smoothing_method. Units should match
        the position coordinates (e.g., centimeters).
    allocentric_env : Environment, optional
        Allocentric environment for geodesic distance calculation.
        Required when ``distance_metric="geodesic"``.
    distance_metric : {"euclidean", "geodesic"}, default="euclidean"
        Distance metric:
        - "euclidean": Straight-line distance
        - "geodesic": Path distance respecting environment boundaries

    Returns
    -------
    ObjectVectorFieldResult
        Dataclass with field, ego_env, and occupancy.

    Raises
    ------
    ValueError
        If spike_times is empty, arrays have mismatched lengths,
        smoothing_method is invalid, or geodesic requires allocentric_env.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding.object_vector import compute_object_vector_field
    >>> rng = np.random.default_rng(42)
    >>> n_time = 1000
    >>> times = np.linspace(0, 100, n_time)
    >>> positions = rng.uniform(0, 100, (n_time, 2))
    >>> headings = rng.uniform(-np.pi, np.pi, n_time)
    >>> object_positions = np.array([[50.0, 50.0]])
    >>> spike_times = rng.choice(times, size=100, replace=False)
    >>> result = compute_object_vector_field(
    ...     spike_times, times, positions, headings, object_positions
    ... )
    >>> len(result.field) == result.ego_env.n_bins
    True
    """
    from neurospatial import Environment
    from neurospatial.ops.egocentric import compute_egocentric_bearing

    # Convert inputs
    spike_times = np.asarray(spike_times, dtype=np.float64).ravel()
    times = np.asarray(times, dtype=np.float64).ravel()
    positions = np.asarray(positions, dtype=np.float64)
    headings = np.asarray(headings, dtype=np.float64).ravel()
    object_positions = np.asarray(object_positions, dtype=np.float64)

    # Validate inputs
    if len(spike_times) == 0:
        raise ValueError(
            "Cannot compute object-vector field: no spikes.\n\n"
            "WHAT: spike_times array is empty\n"
            "WHY: Need at least one spike to compute firing rate\n\n"
            "HOW to fix:\n"
            "1. Check spike extraction - verify this neuron fired during session\n"
            "2. Verify time window overlaps with behavioral data\n"
            "3. Check that spike_times and times are in same units (seconds)\n"
            "4. Filter neurons by minimum spike count before analysis"
        )

    if len(times) != len(positions):
        raise ValueError(
            f"Times/positions length mismatch.\n\n"
            f"WHAT: times has {len(times)} samples, positions has {len(positions)}\n"
            f"WHY: Need synchronized behavioral data at each timepoint\n\n"
            f"HOW to fix:\n"
            f"1. Ensure times and positions are aligned to same sampling\n"
            f"2. Check for dropped frames in position tracking\n"
            f"3. Interpolate to common timebase if needed"
        )

    if len(times) != len(headings):
        raise ValueError(
            f"Times/headings length mismatch.\n\n"
            f"WHAT: times has {len(times)} samples, headings has {len(headings)}\n"
            f"WHY: Need heading at each behavioral timepoint\n\n"
            f"HOW to fix:\n"
            f"1. Compute headings from same positions array:\n"
            f"   headings = heading_from_velocity(positions, dt=times[1]-times[0])\n"
            f"2. Check for dropped frames or different sampling rates"
        )

    if smoothing_method not in ("binned", "diffusion_kde"):
        raise ValueError(
            f"Invalid smoothing_method: '{smoothing_method}'.\n\n"
            f"WHAT: smoothing_method must be 'binned' or 'diffusion_kde'\n"
            f"WHY: These are the supported spatial smoothing algorithms\n\n"
            f"HOW to choose:\n"
            f"1. 'binned' - Simple histogram (faster, noisier)\n"
            f"2. 'diffusion_kde' - Graph-based smoothing (default, respects boundaries)"
        )

    # Validate distance_metric values first
    if distance_metric not in ("euclidean", "geodesic"):
        raise ValueError(
            f"Invalid distance metric: '{distance_metric}'.\n\n"
            f"WHAT: distance_metric must be 'euclidean' or 'geodesic'\n"
            f"WHY: These are the supported distance algorithms\n\n"
            f"HOW to choose:\n"
            f"1. 'euclidean' - Straight-line distances (default, faster)\n"
            f"2. 'geodesic' - Boundary-respecting distances (requires allocentric_env)"
        )

    # Then validate parameter dependencies
    if distance_metric == "geodesic" and allocentric_env is None:
        raise ValueError(
            "Cannot compute geodesic distances: missing environment.\n\n"
            "WHAT: distance_metric='geodesic' requires allocentric_env parameter\n"
            "WHY: Geodesic distances follow paths that respect environment boundaries\n\n"
            "HOW to fix:\n"
            "1. Pass the environment:\n"
            "   result = compute_object_vector_field(\n"
            "       ..., distance_metric='geodesic', allocentric_env=env\n"
            "   )\n"
            "2. Or use Euclidean distances (straight-line):\n"
            "   result = compute_object_vector_field(..., distance_metric='euclidean')"
        )

    n_time = len(times)

    # Compute time step
    dt = np.median(np.diff(times))

    # Create egocentric polar environment
    ego_env = Environment.from_polar_egocentric(
        distance_range=(0.0, max_distance),
        angle_range=(-np.pi, np.pi),
        distance_bin_size=max_distance / n_distance_bins,
        angle_bin_size=2 * np.pi / n_direction_bins,
        circular_angle=True,
    )

    # Compute distance and bearing to all objects at each timepoint
    if distance_metric == "euclidean":
        # distances: (n_time, n_objects)
        distances = np.linalg.norm(
            positions[:, np.newaxis, :] - object_positions[np.newaxis, :, :],
            axis=2,
        )
    else:
        # Geodesic distance using allocentric environment
        from neurospatial.ops.distance import distance_field as compute_distance_field

        assert allocentric_env is not None
        distances = np.zeros((n_time, len(object_positions)), dtype=np.float64)

        for i, obj_pos in enumerate(object_positions):
            # Find bin containing object
            obj_bins = allocentric_env.bin_at(obj_pos.reshape(1, -1))
            obj_bin = int(obj_bins[0])

            if obj_bin < 0:
                distances[:, i] = np.nan
                continue

            # Get distance field from this object
            dist_field = compute_distance_field(
                allocentric_env.connectivity, sources=[obj_bin]
            )

            # Get distance at each animal position
            for t in range(n_time):
                pos_bins = allocentric_env.bin_at(positions[t].reshape(1, -1))
                bin_idx = int(pos_bins[0])
                if 0 <= bin_idx < len(dist_field):
                    distances[t, i] = float(dist_field[bin_idx])
                else:
                    distances[t, i] = np.nan

    # bearings: (n_time, n_objects)
    bearings = compute_egocentric_bearing(object_positions, positions, headings)

    # Find nearest object at each timepoint
    nearest_obj_idx = np.argmin(distances, axis=1)
    nearest_distances = distances[np.arange(n_time), nearest_obj_idx]
    nearest_bearings = bearings[np.arange(n_time), nearest_obj_idx]

    # Compute occupancy in each egocentric bin
    occupancy = np.zeros(ego_env.n_bins, dtype=np.float64)
    spike_counts = np.zeros(ego_env.n_bins, dtype=np.float64)

    # Map each behavioral frame to an egocentric bin
    distance_bin_size = max_distance / n_distance_bins
    angle_bin_size = 2 * np.pi / n_direction_bins

    # Create lookup for bin assignment
    # Bin indices in distance and angle dimensions
    # Handle NaN values to avoid casting warnings
    dist_bin_idx = np.zeros(n_time, dtype=int)
    valid_dist = np.isfinite(nearest_distances)
    dist_bin_idx[valid_dist] = np.floor(
        nearest_distances[valid_dist] / distance_bin_size
    ).astype(int)
    dist_bin_idx = np.clip(dist_bin_idx, 0, n_distance_bins - 1)

    # Angle bins: shift from [-pi, pi] to [0, 2*pi], then divide
    angle_shifted = nearest_bearings + np.pi  # Now [0, 2*pi]
    angle_bin_idx = np.zeros(n_time, dtype=int)
    valid_angle = np.isfinite(angle_shifted)
    angle_bin_idx[valid_angle] = np.floor(
        angle_shifted[valid_angle] / angle_bin_size
    ).astype(int)
    angle_bin_idx = np.clip(angle_bin_idx, 0, n_direction_bins - 1)

    # Convert 2D bin indices to 1D flat index
    # The polar environment bins are ordered: distance varies slow, angle varies fast
    # First need to understand the ego_env bin ordering
    flat_bin_idx = dist_bin_idx * n_direction_bins + angle_bin_idx

    # Make sure indices are valid
    valid_behavior = (
        np.isfinite(nearest_distances)
        & (nearest_distances >= 0)
        & (nearest_distances < max_distance)
    )

    # Accumulate occupancy (vectorized)
    valid_indices = flat_bin_idx[valid_behavior]
    np.add.at(occupancy, valid_indices, dt)

    # Count spikes in each bin
    # Filter spikes to valid time range
    valid_mask = (spike_times >= times[0]) & (spike_times <= times[-1])
    valid_spike_times = spike_times[valid_mask]

    if len(valid_spike_times) > 0:
        # Find nearest behavioral frame for each spike
        spike_frame_idx = np.searchsorted(times, valid_spike_times, side="right") - 1
        spike_frame_idx = np.clip(spike_frame_idx, 0, n_time - 1)

        # Get distance/direction at each spike
        spike_distances = nearest_distances[spike_frame_idx]
        spike_bearings = nearest_bearings[spike_frame_idx]

        # Assign spikes to bins (handle NaN values)
        n_valid_spikes = len(valid_spike_times)
        spike_dist_bins = np.zeros(n_valid_spikes, dtype=int)
        valid_spike_dist = np.isfinite(spike_distances)
        spike_dist_bins[valid_spike_dist] = np.floor(
            spike_distances[valid_spike_dist] / distance_bin_size
        ).astype(int)
        spike_dist_bins = np.clip(spike_dist_bins, 0, n_distance_bins - 1)

        spike_angle_shifted = spike_bearings + np.pi
        spike_angle_bins = np.zeros(n_valid_spikes, dtype=int)
        valid_spike_angle = np.isfinite(spike_angle_shifted)
        spike_angle_bins[valid_spike_angle] = np.floor(
            spike_angle_shifted[valid_spike_angle] / angle_bin_size
        ).astype(int)
        spike_angle_bins = np.clip(spike_angle_bins, 0, n_direction_bins - 1)

        spike_flat_bins = spike_dist_bins * n_direction_bins + spike_angle_bins

        # Filter for valid spikes
        valid_spikes = (
            np.isfinite(spike_distances)
            & (spike_distances >= 0)
            & (spike_distances < max_distance)
        )

        # Count spikes per bin (vectorized)
        valid_spike_indices = spike_flat_bins[valid_spikes]
        np.add.at(spike_counts, valid_spike_indices, 1.0)

    # Compute firing rate
    field = np.zeros(ego_env.n_bins, dtype=np.float64)
    sufficient_occupancy = occupancy >= min_occupancy_seconds

    if smoothing_method == "binned":
        field[sufficient_occupancy] = (
            spike_counts[sufficient_occupancy] / occupancy[sufficient_occupancy]
        )
        field[~sufficient_occupancy] = np.nan

    else:  # diffusion_kde
        # Apply diffusion smoothing before normalizing (spread→normalize)
        # Get diffusion kernel (respects circular boundary via graph)
        kernel = cast("EnvironmentProtocol", ego_env).compute_kernel(
            bandwidth, mode="density", cache=False
        )

        # Spread spike counts and occupancy using kernel
        smoothed_spike_counts = kernel @ spike_counts
        smoothed_occupancy = kernel @ occupancy

        # Normalize
        safe_occupancy = np.where(smoothed_occupancy > 0, smoothed_occupancy, np.nan)
        field = smoothed_spike_counts / safe_occupancy

        # Still mask out bins with insufficient raw occupancy
        field[~sufficient_occupancy] = np.nan

    return ObjectVectorFieldResult(
        field=field,
        ego_env=ego_env,
        occupancy=occupancy,
    )
