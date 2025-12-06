"""Spatial view cell analysis.

Spatial view cells fire when an animal *views* a specific location, regardless
of where the animal is positioned. This module provides tools for computing
spatial view fields and metrics for classifying spatial view cells.

Import Paths
------------
After package reorganization (v0.4.0+), use::

    from neurospatial.encoding.spatial_view import (
        # Field computation
        SpatialViewFieldResult,
        compute_spatial_view_field,
        # Metrics and classification
        SpatialViewMetrics,
        spatial_view_cell_metrics,
        is_spatial_view_cell,
        # Visibility utilities (re-exports)
        compute_viewed_location,
        compute_viewshed,
        visibility_occupancy,
        FieldOfView,
    )

    # Or import from encoding package
    from neurospatial.encoding import (
        compute_spatial_view_field,
        spatial_view_cell_metrics,
    )

Typical Workflow
----------------
1. Compute spatial view field from spike times and behavioral data::

    >>> result = compute_spatial_view_field(  # doctest: +SKIP
    ...     env, spike_times, times, positions, headings
    ... )

2. Compute metrics and classify as spatial view cell::

    >>> metrics = spatial_view_cell_metrics(  # doctest: +SKIP
    ...     env, spike_times, times, positions, headings
    ... )
    >>> if metrics.is_spatial_view_cell:  # doctest: +SKIP
    ...     print("Spatial view cell detected!")

3. Or use quick classifier for screening::

    >>> if is_spatial_view_cell(env, spike_times, times, positions, headings):
    ...     print("Spatial view cell!")  # doctest: +SKIP

Key Difference: Place Cells vs Spatial View Cells
-------------------------------------------------
- **Place cell**: Fires when animal is *at* a specific location
- **Spatial view cell**: Fires when animal is *looking at* a specific location

For place cells, both place field and view field are similar (because viewing
location correlates with position). For spatial view cells, the view field
has higher spatial information than the place field.

See Also
--------
neurospatial.encoding.place : Place cell analysis
neurospatial.ops.visibility : Visibility and gaze computation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from numpy.typing import NDArray

from neurospatial.metrics.spatial_view_cells import (
    SpatialViewMetrics,
    is_spatial_view_cell,
    spatial_view_cell_metrics,
)
from neurospatial.ops.visibility import (
    FieldOfView,
    compute_viewed_location,
    compute_viewshed,
    visibility_occupancy,
)

if TYPE_CHECKING:
    from neurospatial import Environment
    from neurospatial.environment._protocols import EnvironmentProtocol


@dataclass(frozen=True)
class SpatialViewFieldResult:
    """Result of spatial view field computation.

    Attributes
    ----------
    field : NDArray[np.float64], shape (n_bins,)
        Firing rate (Hz) in each spatial bin, binned by *viewed location*
        (not animal position). NaN values indicate insufficient view occupancy.
    env : Environment
        The spatial environment used for the computation.
    view_occupancy : NDArray[np.float64], shape (n_bins,)
        Time spent (seconds) viewing each spatial bin. This differs from
        standard occupancy (time in bin) - it's time the animal was *looking at*
        each bin.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding.spatial_view import SpatialViewFieldResult
    >>> rng = np.random.default_rng(42)
    >>> positions = rng.uniform(0, 100, (500, 2))
    >>> env = Environment.from_samples(positions, bin_size=5.0)
    >>> result = SpatialViewFieldResult(
    ...     field=np.zeros(env.n_bins),
    ...     env=env,
    ...     view_occupancy=np.ones(env.n_bins),
    ... )
    >>> len(result.field) == env.n_bins
    True
    """

    field: NDArray[np.float64]
    env: Environment
    view_occupancy: NDArray[np.float64]


def compute_spatial_view_field(
    env: Environment,
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
    *,
    view_distance: float = 10.0,
    gaze_model: Literal["fixed_distance", "ray_cast", "boundary"] = "fixed_distance",
    gaze_offsets: NDArray[np.float64] | None = None,
    smoothing_method: Literal[
        "diffusion_kde", "gaussian_kde", "binned"
    ] = "diffusion_kde",
    bandwidth: float = 5.0,
    min_occupancy_seconds: float = 0.1,
) -> SpatialViewFieldResult:
    """Compute spatial view field (firing rate by viewed location).

    Computes firing rate as a function of *where the animal is looking*,
    not where the animal is located. This is the defining feature of
    spatial view cells.

    Parameters
    ----------
    env : Environment
        Spatial environment defining the discretization.
    spike_times : NDArray[np.float64], shape (n_spikes,)
        Times of spikes in the same units as ``times``.
    times : NDArray[np.float64], shape (n_time,)
        Timestamps for each behavioral sample.
    positions : NDArray[np.float64], shape (n_time, 2)
        Animal positions in allocentric coordinates.
    headings : NDArray[np.float64], shape (n_time,)
        Animal heading at each time (radians, 0=East).
    view_distance : float, default=10.0
        Distance for fixed_distance gaze model.
    gaze_model : {"fixed_distance", "ray_cast", "boundary"}, default="fixed_distance"
        Method for computing viewed location:
        - "fixed_distance": Point at fixed distance in gaze direction
        - "ray_cast": Intersection with environment boundary
        - "boundary": Nearest boundary point in gaze direction
    gaze_offsets : NDArray[np.float64], shape (n_time,), optional
        Offset from heading to gaze direction (e.g., from eye tracking).
        If None, gaze is aligned with heading.
    smoothing_method : {"diffusion_kde", "gaussian_kde", "binned"}, default="diffusion_kde"
        Smoothing method:
        - "diffusion_kde": Graph-smoothed field using diffusion kernel
        - "gaussian_kde": Standard Gaussian KDE (Euclidean distance)
        - "binned": Simple histogram-based field
    bandwidth : float, default=5.0
        Smoothing bandwidth for KDE methods (environment units).
    min_occupancy_seconds : float, default=0.1
        Minimum view occupancy required in a bin (seconds). Bins with less
        view occupancy are set to NaN.

    Returns
    -------
    SpatialViewFieldResult
        Dataclass with field, env, and view_occupancy.

    Raises
    ------
    ValueError
        If spike_times is empty, arrays have mismatched lengths,
        smoothing_method is invalid, or gaze_model is invalid.

    See Also
    --------
    neurospatial.encoding.place.compute_place_field : Standard place field computation.
    neurospatial.ops.visibility.compute_viewed_location : Gaze computation.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding.spatial_view import compute_spatial_view_field
    >>> rng = np.random.default_rng(42)
    >>> positions = rng.uniform(0, 100, (500, 2))
    >>> env = Environment.from_samples(positions, bin_size=5.0)
    >>> n_time = 1000
    >>> times = np.linspace(0, 100, n_time)
    >>> trajectory = rng.uniform(20, 80, (n_time, 2))  # Stay inside
    >>> headings = rng.uniform(-np.pi, np.pi, n_time)
    >>> spike_times = rng.choice(times, size=100, replace=False)
    >>> result = compute_spatial_view_field(
    ...     env, spike_times, times, trajectory, headings, view_distance=10.0
    ... )
    >>> len(result.field) == env.n_bins
    True
    """
    # Convert inputs
    spike_times = np.asarray(spike_times, dtype=np.float64).ravel()
    times = np.asarray(times, dtype=np.float64).ravel()
    positions = np.asarray(positions, dtype=np.float64)
    headings = np.asarray(headings, dtype=np.float64).ravel()

    # Validate inputs
    if len(spike_times) == 0:
        raise ValueError(
            "Cannot compute spatial view field: no spikes.\n\n"
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
            f"WHY: Need heading at each behavioral timepoint for gaze computation\n\n"
            f"HOW to fix:\n"
            f"1. Compute headings from same positions array:\n"
            f"   headings = heading_from_velocity(positions, dt=times[1]-times[0])\n"
            f"2. Check for dropped frames or different sampling rates"
        )

    valid_methods = {"diffusion_kde", "gaussian_kde", "binned"}
    if smoothing_method not in valid_methods:
        raise ValueError(
            f"Invalid smoothing_method: '{smoothing_method}'.\n\n"
            f"WHAT: smoothing_method must be one of {sorted(valid_methods)}\n"
            f"WHY: Each smoothing_method uses different spatial smoothing algorithms\n\n"
            f"HOW to choose:\n"
            f"1. 'diffusion_kde' (default) - Graph-based smoothing respecting boundaries\n"
            f"2. 'gaussian_kde' - Standard Gaussian kernel (Euclidean distance)\n"
            f"3. 'binned' - No smoothing (histogram-based, noisier)"
        )

    valid_gaze_models = {"fixed_distance", "ray_cast", "boundary"}
    if gaze_model not in valid_gaze_models:
        raise ValueError(
            f"Invalid gaze model: '{gaze_model}'.\n\n"
            f"WHAT: gaze_model must be one of {sorted(valid_gaze_models)}\n"
            f"WHY: Gaze model determines how viewed location is computed\n\n"
            f"HOW to choose:\n"
            f"1. 'fixed_distance' - View point at fixed distance ahead (default)\n"
            f"2. 'ray_cast' - Nearest boundary intersection along gaze direction\n"
            f"3. 'boundary' - Nearest boundary point in gaze direction"
        )

    if gaze_offsets is not None:
        gaze_offsets = np.asarray(gaze_offsets, dtype=np.float64).ravel()
        if len(gaze_offsets) != len(times):
            raise ValueError(
                f"Gaze offsets length mismatch.\n\n"
                f"WHAT: gaze_offsets has {len(gaze_offsets)} samples, "
                f"times has {len(times)}\n"
                f"WHY: Need one gaze offset per timepoint for eye tracking\n\n"
                f"HOW to fix:\n"
                f"1. Interpolate gaze_offsets to match behavioral sampling\n"
                f"2. Or omit gaze_offsets to use heading direction"
            )

    n_time = len(times)

    # Compute time step
    dt = np.median(np.diff(times))

    # Step 1: Compute viewed locations for all timepoints
    viewed_locations = compute_viewed_location(
        positions,
        headings,
        method=gaze_model,
        view_distance=view_distance,
        gaze_offsets=gaze_offsets,
        env=env if gaze_model in ("ray_cast", "boundary") else None,
    )

    # Step 2: Identify valid viewed locations (inside environment, not NaN)
    valid_view_mask = np.all(np.isfinite(viewed_locations), axis=1)

    # Also check if viewed location is inside the environment
    view_bins = np.full(n_time, -1, dtype=np.intp)
    if np.any(valid_view_mask):
        valid_viewed = viewed_locations[valid_view_mask]
        valid_bins = env.bin_at(valid_viewed)
        view_bins[valid_view_mask] = valid_bins
        # Update valid mask to only include views inside environment
        valid_view_mask = view_bins >= 0

    # Step 3: Compute view occupancy (time spent viewing each bin)
    view_occupancy = np.zeros(env.n_bins, dtype=np.float64)
    valid_indices = view_bins[valid_view_mask]
    np.add.at(view_occupancy, valid_indices, dt)

    # Step 4: Count spikes in each viewed bin
    spike_counts = np.zeros(env.n_bins, dtype=np.float64)

    # Filter spikes to valid time range
    valid_spike_mask = (spike_times >= times[0]) & (spike_times <= times[-1])
    valid_spike_times = spike_times[valid_spike_mask]

    if len(valid_spike_times) > 0:
        # Find nearest behavioral frame for each spike
        spike_frame_idx = np.searchsorted(times, valid_spike_times, side="right") - 1
        spike_frame_idx = np.clip(spike_frame_idx, 0, n_time - 1)

        # Get viewed bin at each spike time
        spike_view_bins = view_bins[spike_frame_idx]

        # Only count spikes where view was valid
        valid_spike_views = spike_view_bins >= 0
        valid_spike_view_bins = spike_view_bins[valid_spike_views]

        # Count spikes per viewed bin (vectorized)
        np.add.at(spike_counts, valid_spike_view_bins, 1.0)

    # Step 5: Compute firing rate based on smoothing_method
    field = np.zeros(env.n_bins, dtype=np.float64)
    sufficient_occupancy = view_occupancy >= min_occupancy_seconds

    if smoothing_method == "binned":
        # Simple occupancy-normalized field
        field[sufficient_occupancy] = (
            spike_counts[sufficient_occupancy] / view_occupancy[sufficient_occupancy]
        )
        field[~sufficient_occupancy] = np.nan

    elif smoothing_method == "diffusion_kde":
        # Apply diffusion smoothing before normalizing (spread→normalize)
        kernel = cast("EnvironmentProtocol", env).compute_kernel(
            bandwidth, mode="density", cache=True
        )

        # Spread spike counts and view occupancy using kernel
        smoothed_spike_counts = kernel @ spike_counts
        smoothed_view_occupancy = kernel @ view_occupancy

        # Normalize
        safe_occupancy = np.where(
            smoothed_view_occupancy > 0, smoothed_view_occupancy, np.nan
        )
        field = smoothed_spike_counts / safe_occupancy

        # Still mask out bins with insufficient raw occupancy
        field[~sufficient_occupancy] = np.nan

    else:  # gaussian_kde
        # Standard Gaussian KDE with Euclidean distance
        two_sigma_sq = 2 * bandwidth**2

        for i, bin_center in enumerate(env.bin_centers):
            # Spike density: weight by distance to viewed locations at spike times
            if len(valid_spike_times) > 0:
                spike_frame_idx = (
                    np.searchsorted(times, valid_spike_times, side="right") - 1
                )
                spike_frame_idx = np.clip(spike_frame_idx, 0, n_time - 1)
                spike_viewed = viewed_locations[spike_frame_idx]

                # Only include spikes with valid views
                valid_spike_mask_local = np.all(np.isfinite(spike_viewed), axis=1)
                if np.any(valid_spike_mask_local):
                    spike_viewed_valid = spike_viewed[valid_spike_mask_local]
                    spike_distances_sq = np.sum(
                        (spike_viewed_valid - bin_center) ** 2, axis=1
                    )
                    spike_weights = np.exp(-spike_distances_sq / two_sigma_sq)
                    spike_density = np.sum(spike_weights)
                else:
                    spike_density = 0.0
            else:
                spike_density = 0.0

            # View occupancy density: weight trajectory by distance from view to bin
            valid_viewed_local = viewed_locations[valid_view_mask]
            if len(valid_viewed_local) > 0:
                view_distances_sq = np.sum(
                    (valid_viewed_local - bin_center) ** 2, axis=1
                )
                view_weights = np.exp(-view_distances_sq / two_sigma_sq)
                view_dens = np.sum(view_weights) * dt
            else:
                view_dens = 0.0

            # Normalize
            if view_dens > 0:
                field[i] = spike_density / view_dens
            else:
                field[i] = np.nan

    return SpatialViewFieldResult(
        field=field,
        env=env,
        view_occupancy=view_occupancy,
    )


__all__ = [  # noqa: RUF022 - organized by category
    # Field computation
    "SpatialViewFieldResult",
    "compute_spatial_view_field",
    # Metrics and classification
    "SpatialViewMetrics",
    "spatial_view_cell_metrics",
    "is_spatial_view_cell",
    # Re-exports from ops.visibility
    "compute_viewed_location",
    "compute_viewshed",
    "visibility_occupancy",
    "FieldOfView",
]
