"""Compute object-vector fields in egocentric polar coordinates.

Object-vector fields represent neural activity as a function of egocentric
distance and direction to objects in the environment. Unlike allocentric
place fields (which show activity by world location), object-vector fields
show activity relative to object positions from the animal's perspective.

Which Function Should I Use?
----------------------------
**Computing object-vector tuning from spikes?**
    Use ``compute_object_vector_field()`` to compute firing rate as a function
    of egocentric distance and direction to nearest object.

**Need binned vs smoothed fields?**
    Use ``smoothing_method="binned"`` for simple histogram-based fields, or
    ``smoothing_method="diffusion_kde"`` for graph-smoothed fields.

**Using geodesic distances?**
    Pass ``allocentric_env`` and ``distance_metric="geodesic"`` to compute
    distances respecting environment boundaries.

Typical Workflow
----------------
1. Compute object-vector field from spike times and behavioral data::

    >>> result = compute_object_vector_field(  # doctest: +SKIP
    ...     spike_times, times, positions, headings, object_positions
    ... )

2. Access the field and egocentric environment::

    >>> field = result.field  # Firing rate per egocentric bin  # doctest: +SKIP
    >>> ego_env = result.ego_env  # Egocentric polar environment

3. Visualize using the egocentric environment::

    >>> ego_env.plot_field(field)  # doctest: +SKIP

Coordinate Conventions
----------------------
**Egocentric direction** (in the output ego_env):
- 0 radians = object is directly ahead of animal
- pi/2 radians = object is to the left
- -pi/2 radians = object is to the right
- +/-pi radians = object is behind

This matches the coordinate convention in ``neurospatial.reference_frames``.

**Egocentric polar bins** (ego_env.bin_centers):
- bin_centers[:, 0] = distance to nearest object
- bin_centers[:, 1] = egocentric direction to nearest object

References
----------
Hoydal, O. A., et al. (2019). Object-vector coding in the medial entorhinal
    cortex. Nature, 568(7752), 400-404.

See Also
--------
neurospatial.metrics.object_vector_cells : Metrics for OVC analysis
neurospatial.reference_frames : Egocentric coordinate transforms
neurospatial.Environment.from_polar_egocentric : Egocentric polar environment
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from numpy.typing import NDArray

from neurospatial.reference_frames import compute_egocentric_bearing

if TYPE_CHECKING:
    from neurospatial import Environment
    from neurospatial.environment._protocols import EnvironmentProtocol

__all__ = [
    "ObjectVectorFieldResult",
    "compute_object_vector_field",
]


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
    >>> from neurospatial.object_vector_field import ObjectVectorFieldResult
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
    >>> from neurospatial.object_vector_field import compute_object_vector_field
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
        from neurospatial.distance import distance_field as compute_distance_field

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
        # Cast to EnvironmentProtocol to satisfy mypy's strict type checking
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
