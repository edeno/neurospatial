# Implementation Plan: Object-Vector Cells, Egocentric Frames, and Spatial View Cells

**Created**: 2025-12-05
**Revised**: 2025-12-05 (incorporated review feedback)
**Status**: Draft
**Estimated Scope**: 3 milestones, ~17 modules, ~6,000 LOC

---

## Executive Summary

This plan adds three interconnected features to neurospatial:

1. **Egocentric Reference Frames** (Foundation) - Coordinate transformations between allocentric and egocentric frames
2. **Object-Vector Cells** - Cells encoding distance and direction to discrete objects
3. **Spatial View Cells** - Cells encoding viewed/attended locations independent of physical position

These features share common infrastructure (heading data, angular computations, coordinate transforms) and should be implemented in dependency order.

### Key Design Principles (from review)

1. **Reuse existing modules** - Leverage `circular.py`, `head_direction.py`, `place_fields.py` utilities
2. **Vectorization first** - Avoid Python loops; batch operations on (n_time, n_objects, n_dims)
3. **Explicit conventions** - Document angle conventions, von Mises parameterization clearly
4. **Small public API** - Gate internals behind subpackages; export only essential functions
5. **Shared code paths** - Smoothing, occupancy, information metrics use identical implementations

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User-Facing API                              │
├─────────────────────────────────────────────────────────────────────┤
│  compute_object_vector_field()  │  compute_spatial_view_field()     │
│  object_vector_score()          │  spatial_view_score()              │
│  ObjectVectorCellModel          │  SpatialViewCellModel              │
│  ObjectVectorOverlay            │  SpatialViewOverlay                │
├─────────────────────────────────────────────────────────────────────┤
│                      Egocentric Transforms                           │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  EgocentricFrame  │  allocentric_to_egocentric()              │  │
│  │  compute_egocentric_bearing()  │  create_egocentric_grid()    │  │
│  └───────────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│                    Existing Infrastructure                           │
│  circular.py  │  distance.py  │  head_direction.py  │  transforms.py│
└─────────────────────────────────────────────────────────────────────┘
```

---

## Milestone 1: Egocentric Reference Frames

**Goal**: Enable transformations between allocentric (world-centered) and egocentric (animal-centered) coordinate systems.

**Priority**: HIGH (foundation for other features)

### M1.1: Core Reference Frame Module

**File**: `src/neurospatial/reference_frames.py`

```python
"""Coordinate reference frame transformations.

Supports conversions between:
- Allocentric: World-centered, fixed axes (standard spatial analysis)
- Egocentric: Animal-centered, axes rotate with heading

Common Use Cases
----------------
- Object-vector cell analysis (egocentric distance/direction to objects)
- Spatial view cell analysis (what location is being viewed)
- GLM regressors in egocentric coordinates
- Behavioral analysis of approach/avoidance

Examples
--------
Transform landmark positions to egocentric coordinates:

>>> from neurospatial.reference_frames import allocentric_to_egocentric
>>> ego_positions = allocentric_to_egocentric(
...     landmark_positions,  # (n_landmarks, 2)
...     animal_positions,    # (n_time, 2)
...     animal_headings,     # (n_time,)
... )
>>> # ego_positions shape: (n_time, n_landmarks, 2)
>>> # ego_positions[t, i] = landmark i relative to animal at time t
"""
```

**Classes/Functions**:

| Name | Type | Purpose |
|------|------|---------|
| `EgocentricFrame` | dataclass | Single-timepoint frame with `to_egocentric()` / `to_allocentric()` methods |
| `allocentric_to_egocentric()` | function | Batch transform: (n_time, n_points, 2) allocentric → egocentric |
| `egocentric_to_allocentric()` | function | Batch transform: egocentric → allocentric |
| `compute_egocentric_bearing()` | function | Angle to target relative to heading (0=ahead, ±π=behind) |
| `compute_egocentric_distance()` | function | Distance to target (supports geodesic) |
| `heading_from_velocity()` | function | Infer heading from position timeseries |
| `heading_from_body_orientation()` | function | Heading from pose keypoints (nose, tail) |

**Implementation Details**:

```python
@dataclass(frozen=True)
class EgocentricFrame:
    """Egocentric reference frame at a single timepoint.

    Attributes
    ----------
    position : NDArray, shape (2,)
        Animal position in allocentric coordinates.
    heading : float
        Animal heading in radians (0=East, π/2=North in allocentric frame).

    Coordinate Conventions
    ----------------------
    **Allocentric (world-centered)**:
    - 0 radians = East (+x direction)
    - π/2 radians = North (+y direction)
    - Standard mathematical convention

    **Egocentric (animal-centered)**:
    - Origin at animal's position
    - +x axis = forward (heading direction)
    - +y axis = left (90° counterclockwise from heading)
    - Angles: 0=ahead, π/2=left, -π/2=right, ±π=behind

    Examples
    --------
    >>> # Animal at origin facing East (heading=0)
    >>> frame = EgocentricFrame(position=np.array([0, 0]), heading=0)
    >>> # Point 10 units East is 10 units ahead
    >>> frame.to_egocentric(np.array([[10, 0]]))
    array([[10., 0.]])  # 10 units forward, 0 left

    >>> # Animal at origin facing North (heading=π/2)
    >>> frame = EgocentricFrame(position=np.array([0, 0]), heading=np.pi/2)
    >>> # Same point (10 units East) is now 10 units to the right
    >>> frame.to_egocentric(np.array([[10, 0]]))
    array([[ 0., -10.]])  # 0 forward, 10 right (negative y = right)

    >>> # Point 10 units North is now 10 units ahead
    >>> frame.to_egocentric(np.array([[0, 10]]))
    array([[10., 0.]])  # 10 units forward

    >>> # Round-trip preserves coordinates
    >>> allocentric = np.array([[5, 3]])
    >>> egocentric = frame.to_egocentric(allocentric)
    >>> recovered = frame.to_allocentric(egocentric)
    >>> np.allclose(allocentric, recovered)
    True
    """
    position: NDArray[np.float64]
    heading: float

    def to_egocentric(self, points: NDArray) -> NDArray:
        """Transform allocentric points to egocentric coordinates."""
        centered = points - self.position
        cos_h, sin_h = np.cos(-self.heading), np.sin(-self.heading)
        rotation = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
        return centered @ rotation.T

    def to_allocentric(self, ego_points: NDArray) -> NDArray:
        """Transform egocentric points to allocentric coordinates."""
        cos_h, sin_h = np.cos(self.heading), np.sin(self.heading)
        rotation = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
        return (ego_points @ rotation.T) + self.position
```

**Vectorized batch function** (revised for proper vectorization):

```python
def allocentric_to_egocentric(
    points: NDArray[np.float64],
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Batch transform allocentric points to egocentric coordinates.

    Parameters
    ----------
    points : NDArray, shape (n_points, 2) or (n_time, n_points, 2)
        Points to transform. If 2D, same points transformed at each time.
    positions : NDArray, shape (n_time, 2)
        Animal position at each time.
    headings : NDArray, shape (n_time,)
        Animal heading at each time (radians).

    Returns
    -------
    NDArray, shape (n_time, n_points, 2)
        Points in egocentric coordinates.

    Examples
    --------
    >>> landmarks = np.array([[10, 20], [30, 40]])  # 2 landmarks
    >>> positions = np.array([[0, 0], [5, 5]])      # 2 timepoints
    >>> headings = np.array([0, np.pi/2])           # East, then North
    >>> ego = allocentric_to_egocentric(landmarks, positions, headings)
    >>> ego.shape
    (2, 2, 2)
    """
    n_time = len(positions)

    # Normalize to (n_time, n_points, 2) immediately
    if points.ndim == 2:
        points = np.broadcast_to(points, (n_time, len(points), 2))
    elif points.shape[0] != n_time:
        raise ValueError(f"points.shape[0]={points.shape[0]} != n_time={n_time}")

    # Center on animal position
    centered = points - positions[:, np.newaxis, :]  # (n_time, n_points, 2)

    # Build rotation matrices efficiently (no loops)
    cos_h = np.cos(-headings)
    sin_h = np.sin(-headings)
    # rot[t] = [[cos, -sin], [sin, cos]] for rotation by -heading
    rot = np.stack([
        np.stack([cos_h, -sin_h], axis=-1),
        np.stack([sin_h,  cos_h], axis=-1),
    ], axis=-2)  # (n_time, 2, 2)

    # Apply rotation: ego[t, i] = rot[t] @ centered[t, i]
    # Using einsum for clarity: 'tij,tpj->tpi'
    ego = np.einsum('tij,tpj->tpi', rot, centered)

    return ego
```

**Bearing computation** (uses existing circular utilities):

```python
def compute_egocentric_bearing(
    target_positions: NDArray[np.float64],
    animal_positions: NDArray[np.float64],
    headings: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute egocentric bearing to targets.

    Uses allocentric_to_egocentric internally, then extracts angle.
    Result is in (-π, π]: 0=ahead, π/2=left, -π/2=right, ±π=behind.

    Parameters
    ----------
    target_positions : NDArray, shape (n_targets, 2) or (n_time, n_targets, 2)
        Target positions in allocentric coordinates.
    animal_positions : NDArray, shape (n_time, 2)
        Animal positions.
    headings : NDArray, shape (n_time,)
        Animal headings (radians).

    Returns
    -------
    NDArray, shape (n_time, n_targets) or (n_time,) if single target
        Egocentric bearing angles in radians.
    """
    # Transform to egocentric coordinates
    ego = allocentric_to_egocentric(target_positions, animal_positions, headings)

    # Extract bearing angle
    bearings = np.arctan2(ego[..., 1], ego[..., 0])

    # Wrap to (-π, π] - standard angle wrapping
    # Note: arctan2 already returns (-π, π], so this is defensive
    return (bearings + np.pi) % (2 * np.pi) - np.pi
```

**Distance functions** (split Euclidean vs geodesic):

```python
def compute_egocentric_distance_euclidean(
    target_positions: NDArray[np.float64],
    animal_positions: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute Euclidean distance to targets (pure geometry, no Environment needed)."""
    diff = np.atleast_3d(target_positions) - animal_positions[:, np.newaxis, :]
    return np.linalg.norm(diff, axis=-1)


def compute_egocentric_distance_geodesic(
    env: "Environment",
    target_positions: NDArray[np.float64],
    animal_positions: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute geodesic distance to targets using environment connectivity.

    Note: For repeated queries with fixed targets, precompute distance fields
    using distance_field() and index by animal bin for better performance.
    """
    from neurospatial.distance import distance_field

    # Get bin indices
    target_bins = env.bin_at(target_positions)
    animal_bins = env.bin_at(animal_positions)

    # Precompute distance field from each target
    n_targets = len(target_bins)
    distances = np.zeros((len(animal_positions), n_targets))

    for i, target_bin in enumerate(target_bins):
        if target_bin < 0:
            distances[:, i] = np.nan  # Target outside environment
        else:
            dist_field = distance_field(env.connectivity, sources=[target_bin])
            distances[:, i] = dist_field[animal_bins]

    return distances


def compute_egocentric_distance(
    target_positions: NDArray[np.float64],
    animal_positions: NDArray[np.float64],
    env: "Environment | None" = None,
    metric: Literal["euclidean", "geodesic"] = "euclidean",
) -> NDArray[np.float64]:
    """Compute egocentric distance to targets.

    Convenience wrapper that dispatches to Euclidean or geodesic implementation.

    Parameters
    ----------
    target_positions : NDArray, shape (n_targets, 2)
        Target positions.
    animal_positions : NDArray, shape (n_time, 2)
        Animal positions.
    env : Environment, optional
        Required for geodesic metric.
    metric : {"euclidean", "geodesic"}
        Distance metric.

    Returns
    -------
    NDArray, shape (n_time, n_targets)
        Distance to each target at each time point.

    Raises
    ------
    ValueError
        If metric="geodesic" but env is not provided.

    Examples
    --------
    >>> distances = compute_egocentric_distance(targets, positions)  # Euclidean
    >>> distances = compute_egocentric_distance(targets, positions, env, "geodesic")
    """
    if metric == "euclidean":
        return compute_egocentric_distance_euclidean(target_positions, animal_positions)
    elif metric == "geodesic":
        if env is None:
            raise ValueError("env required for geodesic distance")
        return compute_egocentric_distance_geodesic(env, target_positions, animal_positions)
    else:
        raise ValueError(f"Unknown metric: {metric}")
```

### M1.2: Egocentric Environment Grid

**File**: `src/neurospatial/environment/factories.py` (add method)

```python
@classmethod
def from_egocentric_grid(
    cls,
    distance_range: tuple[float, float] = (0.0, 50.0),
    angle_range: tuple[float, float] = (-np.pi, np.pi),
    distance_bin_size: float = 2.0,
    angle_bin_size: float = np.pi / 12,  # 15 degrees
    units: str = "cm",
    circular_angle: bool = True,
) -> "Environment":
    """Create polar environment for egocentric rate maps.

    Creates a 2D grid in (distance, angle) space centered on animal.
    Useful for egocentric place fields and object-vector analysis.

    **Naming Note**: Despite "grid" in the name, this creates a **polar**
    coordinate environment (distance × angle), not a Cartesian grid.
    The name reflects that it uses the regular grid machinery internally.

    **Important**: This environment lives in egocentric polar coordinates,
    not physical allocentric space. Dimension 0 is radial distance from
    the animal, dimension 1 is egocentric angle (0=ahead, π/2=left).

    Parameters
    ----------
    distance_range : tuple
        (min_distance, max_distance) from animal.
    angle_range : tuple
        (min_angle, max_angle) in radians. Default covers full circle.
    distance_bin_size : float
        Bin size for distance dimension.
    angle_bin_size : float
        Bin size for angle dimension.
    units : str
        Spatial units for distance dimension.
    circular_angle : bool
        If True (default), angle dimension has circular/periodic connectivity
        (first and last angle bins are neighbors). Set False for partial arcs.

    Returns
    -------
    Environment
        Polar grid environment. Access bin centers via:
        - env.bin_centers[:, 0] = distances
        - env.bin_centers[:, 1] = angles

    Notes
    -----
    This reuses existing regular grid machinery with:
    - Distance axis: standard linear bins
    - Angle axis: circular topology when angle_range spans 2π

    Connectivity, diffusion kernels, and smoothing work correctly because
    the circular flag ensures proper neighbor relationships.

    Examples
    --------
    >>> ego_env = Environment.from_egocentric_grid(
    ...     distance_range=(0, 50),
    ...     distance_bin_size=5.0,
    ...     angle_bin_size=np.pi / 6,  # 30° bins
    ... )
    >>> ego_env.n_bins  # 10 distance × 12 angle bins
    120
    """
```

### M1.3: Heading Computation Utilities

**File**: `src/neurospatial/reference_frames.py` (continued)

```python
def heading_from_velocity(
    positions: NDArray[np.float64],
    times: NDArray[np.float64],
    smoothing_window: float = 0.1,
    min_speed: float = 2.0,
) -> NDArray[np.float64]:
    """Infer heading direction from velocity.

    Parameters
    ----------
    positions : NDArray, shape (n_time, 2)
        Position timeseries.
    times : NDArray, shape (n_time,)
        Timestamps.
    smoothing_window : float
        Gaussian smoothing window (seconds).
    min_speed : float
        Minimum speed for valid heading. Below this, heading is interpolated.

    Returns
    -------
    NDArray, shape (n_time,)
        Heading direction in radians.

    Raises
    ------
    ValueError
        If n_time < 2 (need at least 2 points for velocity).

    Notes
    -----
    Uses velocity direction as heading proxy. Invalid during low-speed
    periods (animal stationary); these are filled via **circular interpolation**
    to avoid discontinuities at the ±π boundary.

    Edge Cases
    ----------
    - n_time < 2: Raises ValueError (cannot compute velocity)
    - n_time == 2: Returns heading from single velocity vector (no smoothing)
    - All speeds below min_speed: Returns zeros (forward-fills initial direction)

    Implementation:
    1. Compute velocity via finite differences
    2. Apply Gaussian smoothing with sigma = smoothing_window / median(dt)
    3. Compute heading = atan2(vy, vx)
    4. Mask low-speed periods (speed < min_speed)
    5. Interpolate masked periods using unit-vector representation:
       - Convert heading to unit vectors: u = (cos θ, sin θ)
       - Linearly interpolate u, then renormalize
       - Recover angle: θ = atan2(u_y, u_x)
       This keeps interpolation continuous across the ±π branch cut.
    """
    # Edge case: need at least 2 points for velocity
    n_time = len(times)
    if n_time < 2:
        raise ValueError("Need at least 2 timepoints to compute velocity-based heading")

    # Compute velocity
    dt = np.diff(times)
    velocity = np.diff(positions, axis=0) / dt[:, np.newaxis]

    # Pad to match original length
    velocity = np.vstack([velocity[0:1], velocity])

    # Smooth velocity (convert window from seconds to samples)
    median_dt = np.median(dt)
    sigma_samples = smoothing_window / median_dt
    from scipy.ndimage import gaussian_filter1d
    velocity_smooth = gaussian_filter1d(velocity, sigma=sigma_samples, axis=0)

    # Compute speed and heading
    speed = np.linalg.norm(velocity_smooth, axis=1)
    heading = np.arctan2(velocity_smooth[:, 1], velocity_smooth[:, 0])

    # Mask low-speed periods
    valid = speed >= min_speed

    if not valid.all():
        if not valid.any():
            # All speeds below threshold - cannot reliably determine heading
            import warnings
            warnings.warn(
                f"All {n_time} timepoints have speed below min_speed={min_speed}.\n\n"
                "This usually means:\n"
                "- Animal was stationary throughout recording\n"
                "- Position tracking has low temporal resolution\n"
                "- min_speed threshold is too high\n\n"
                "Returning zeros (forward heading). Consider:\n"
                "1. Lowering min_speed threshold\n"
                "2. Using heading_from_body_orientation() with pose keypoints\n"
                "3. Using a fixed heading value for stationary analysis",
                category=UserWarning,
                stacklevel=2,
            )
            return np.zeros(n_time)

        # Circular interpolation via unit vectors
        unit_x = np.cos(heading)
        unit_y = np.sin(heading)

        # Interpolate unit vectors at invalid points
        valid_indices = np.where(valid)[0]
        invalid_indices = np.where(~valid)[0]

        unit_x[invalid_indices] = np.interp(
            invalid_indices, valid_indices, unit_x[valid_indices]
        )
        unit_y[invalid_indices] = np.interp(
            invalid_indices, valid_indices, unit_y[valid_indices]
        )

        # Renormalize and recover heading
        norm = np.sqrt(unit_x**2 + unit_y**2)
        heading = np.arctan2(unit_y / norm, unit_x / norm)

    return heading


def heading_from_body_orientation(
    nose_positions: NDArray[np.float64],
    tail_positions: NDArray[np.float64],
    handle_nans: bool = True,
) -> NDArray[np.float64]:
    """Compute heading from body axis (nose to tail).

    Parameters
    ----------
    nose_positions : NDArray, shape (n_time, 2)
        Nose keypoint positions.
    tail_positions : NDArray, shape (n_time, 2)
        Tail/body keypoint positions.
    handle_nans : bool
        If True, interpolate through NaN keypoints using unit-vector method.

    Returns
    -------
    NDArray, shape (n_time,)
        Heading direction in radians (nose direction).

    Raises
    ------
    ValueError
        If all keypoints are NaN and handle_nans is True.

    Notes
    -----
    When keypoints are missing (NaN), uses the same circular interpolation
    strategy as heading_from_velocity to maintain continuity.

    Edge Cases
    ----------
    - All NaN keypoints: Raises ValueError (cannot interpolate)
    - Single valid keypoint: All headings set to that single valid value
    - Empty arrays: Returns empty array
    """
    body_vector = nose_positions - tail_positions
    heading = np.arctan2(body_vector[:, 1], body_vector[:, 0])

    if handle_nans:
        nan_mask = np.isnan(heading)
        if nan_mask.all():
            raise ValueError(
                "All keypoints are NaN; cannot compute heading. "
                "Set handle_nans=False to return NaN array instead."
            )
        if nan_mask.any():
            # Circular interpolation
            unit_x = np.cos(heading)
            unit_y = np.sin(heading)

            valid_indices = np.where(~nan_mask)[0]
            invalid_indices = np.where(nan_mask)[0]

            unit_x[invalid_indices] = np.interp(
                invalid_indices, valid_indices, unit_x[valid_indices]
            )
            unit_y[invalid_indices] = np.interp(
                invalid_indices, valid_indices, unit_y[valid_indices]
            )

            norm = np.sqrt(unit_x**2 + unit_y**2)
            heading = np.arctan2(unit_y / norm, unit_x / norm)

    return heading
```

### M1.4: Tests for Reference Frames

**File**: `tests/test_reference_frames.py`

**Test categories**:
1. `TestModuleSetup` - imports, docstrings, `__all__`
2. `TestEgocentricFrame` - dataclass, to_egocentric, to_allocentric, round-trip
3. `TestAllocentricToEgocentric` - batch transform, broadcasting, edge cases
4. `TestComputeEgocentricBearing` - angle computations, wrap-around
5. `TestHeadingFromVelocity` - speed filtering, smoothing, interpolation
6. `TestHeadingFromBodyOrientation` - pose-based heading

**Critical test cases**:
- Round-trip: `allocentric → egocentric → allocentric` preserves positions
- Heading=0: egocentric x-axis aligns with allocentric x-axis
- Heading=π/2: egocentric x-axis aligns with allocentric y-axis
- Circular wrap: angles near ±π handled correctly
- Stationary periods: heading interpolation works

### M1.5: Documentation

**Updates required**:
- `.claude/QUICKSTART.md`: Add egocentric transform examples
- `.claude/API_REFERENCE.md`: Add `reference_frames` module
- `src/neurospatial/__init__.py`: Export key functions

---

## Milestone 2: Object-Vector Cells

**Goal**: Implement object-vector cell models, metrics, and visualization.

**Priority**: HIGH (well-defined, immediate scientific value)

**Dependencies**: M1 (egocentric transforms)

### M2.1: Object-Vector Cell Model (Simulation)

**File**: `src/neurospatial/simulation/models/object_vector_cells.py`

```python
"""Object-vector cell simulation model.

Object-vector cells encode the distance and direction from the animal
to discrete objects (landmarks, goals, other animals) in the environment.

Key Properties
--------------
- Fire based on egocentric vector to object
- Distance tuning: Gaussian centered at preferred distance
- Direction tuning: von Mises centered at preferred egocentric direction
- Object selectivity: may respond to specific object or any object

References
----------
.. [1] Hoydal et al. (2019). Object-vector cells in the medial
       entorhinal cortex. Nature, 568(7752), 400-404.
.. [2] Deshmukh & Knierim (2011). Representation of non-spatial and
       spatial information in the lateral entorhinal cortex.
       Frontiers in Behavioral Neuroscience, 5, 69.
"""

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from neurospatial import Environment
from neurospatial.distance import distance_field
from neurospatial.reference_frames import (
    allocentric_to_egocentric,
    compute_egocentric_bearing,
    compute_egocentric_distance,
)


@dataclass
class ObjectVectorCellModel:
    """Simulates object-vector cell firing.

    Parameters
    ----------
    env : Environment
        Spatial environment.
    object_positions : NDArray, shape (n_objects, n_dims)
        Positions of objects in allocentric coordinates.
    preferred_distance : float
        Distance from object that elicits maximum firing (in env units).
    distance_width : float
        Tuning width for distance (Gaussian sigma).
    preferred_direction : float, optional
        Preferred egocentric direction to object (radians).
        0 = ahead, π/2 = left, -π/2 = right, ±π = behind.
        If None, cell is omnidirectional.
    direction_kappa : float
        von Mises concentration parameter for direction tuning.
        Higher = narrower tuning. Typical values:
        - kappa=1: very broad (~60° half-width)
        - kappa=4: moderate (~30° half-width)
        - kappa=10: narrow (~18° half-width)
        Note: The approximation κ ≈ 1/σ² only holds for large κ (>2).
        For tuning width, use half-width ≈ arccos(1 - 1/κ) for κ > 1.
    object_selectivity : {"any", "nearest", "specific"}
        - "any": responds to any object at preferred vector
        - "nearest": responds only to nearest object
        - "specific": responds only to specific_object_index
    specific_object_index : int, optional
        Which object to respond to (if object_selectivity="specific").
    max_rate : float
        Maximum firing rate (Hz).
    baseline_rate : float
        Baseline firing rate (Hz).
    distance_metric : {"euclidean", "geodesic"}
        How to compute distance to objects.
    seed : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    ground_truth : dict
        Model parameters for validation.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.simulation import ObjectVectorCellModel
    >>>
    >>> # Create environment with a landmark
    >>> positions = np.random.rand(1000, 2) * 100
    >>> env = Environment.from_samples(positions, bin_size=2.0)
    >>> landmark = np.array([[50, 50]])
    >>>
    >>> # Create cell that fires when 10cm ahead of landmark
    >>> cell = ObjectVectorCellModel(
    ...     env=env,
    ...     object_positions=landmark,
    ...     preferred_distance=10.0,
    ...     distance_width=5.0,
    ...     preferred_direction=0.0,  # ahead
    ...     direction_kappa=4.0,  # ~30° half-width (von Mises concentration)
    ... )
    >>>
    >>> # Generate firing rates
    >>> headings = np.random.uniform(-np.pi, np.pi, len(positions))
    >>> rates = cell.firing_rate(positions, headings=headings)
    """

    env: Environment
    object_positions: NDArray[np.float64]
    preferred_distance: float = 10.0  # cm (or env.units)
    distance_width: float = 5.0  # Gaussian sigma for distance tuning
    preferred_direction: float | None = None  # radians (0=ahead, π/2=left)
    direction_kappa: float = 4.0  # von Mises: ~30° half-width (κ=1→60°, κ=10→18°)
    object_selectivity: Literal["any", "nearest", "specific"] = "any"
    specific_object_index: int = 0
    max_rate: float = 20.0  # Hz
    baseline_rate: float = 1.0  # Hz
    distance_metric: Literal["euclidean", "geodesic"] = "euclidean"
    seed: int | None = None

    # Private attributes
    _distance_fields: dict[int, NDArray] = field(default_factory=dict, repr=False)
    _rng: np.random.Generator = field(default=None, repr=False)

    def __post_init__(self):
        """Validate parameters and precompute distance fields."""
        # Validation
        if self.max_rate <= 0:
            raise ValueError("max_rate must be positive")
        if self.baseline_rate < 0:
            raise ValueError("baseline_rate must be non-negative")
        if self.max_rate <= self.baseline_rate:
            raise ValueError("max_rate must exceed baseline_rate")
        if self.distance_width <= 0:
            raise ValueError("distance_width must be positive")
        if self.direction_kappa <= 0:
            raise ValueError("direction_kappa must be positive")

        # Ensure object_positions is 2D
        self.object_positions = np.atleast_2d(self.object_positions)

        # Validate objects are within reasonable bounds (regardless of metric)
        import warnings
        object_bins = self.env.bin_at(self.object_positions)
        invalid_mask = object_bins < 0
        if invalid_mask.any():
            invalid_indices = np.where(invalid_mask)[0]
            warnings.warn(
                f"Objects at indices {invalid_indices.tolist()} are outside "
                f"environment bounds.\n\n"
                f"Object positions: {self.object_positions[invalid_mask].tolist()}\n"
                f"Environment bounds: {self.env.dimension_ranges}\n\n"
                f"This may cause:\n"
                f"- NaN firing rates for these objects\n"
                f"- Empty or incomplete tuning curves\n"
                f"- Failed metric computations\n\n"
                f"To fix:\n"
                f"1. Verify object coordinates are in correct units ({self.env.units})\n"
                f"2. Expand environment to include objects\n"
                f"3. Remove objects outside the arena",
                category=UserWarning,
                stacklevel=2,
            )

        # Precompute geodesic distance fields
        if self.distance_metric == "geodesic":
            for i, bin_idx in enumerate(object_bins):
                if bin_idx < 0:
                    self._distance_fields[i] = np.full(self.env.n_bins, np.nan)
                else:
                    self._distance_fields[i] = distance_field(
                        self.env.connectivity, sources=[bin_idx]
                    )

        # Initialize RNG
        self._rng = np.random.default_rng(self.seed)

    def firing_rate(
        self,
        positions: NDArray[np.float64],
        times: NDArray[np.float64] | None = None,
        headings: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Compute firing rate at given positions.

        Parameters
        ----------
        positions : NDArray, shape (n_time, n_dims)
            Animal positions.
        times : NDArray, shape (n_time,), optional
            Timestamps (unused, for protocol compatibility).
        headings : NDArray, shape (n_time,), optional
            Animal heading directions. Required if preferred_direction is set.

        Returns
        -------
        NDArray, shape (n_time,)
            Firing rates in Hz.
        """
        positions = np.atleast_2d(positions)
        n_time = len(positions)
        n_objects = len(self.object_positions)

        # Compute distances to all objects
        if self.distance_metric == "geodesic":
            bin_indices = self.env.bin_at(positions)
            distances = np.column_stack([
                self._distance_fields[i][bin_indices]
                for i in range(n_objects)
            ])
        else:
            # Euclidean: (n_time, n_objects)
            diff = positions[:, np.newaxis, :] - self.object_positions[np.newaxis, :, :]
            distances = np.linalg.norm(diff, axis=-1)

        # Distance tuning (Gaussian)
        dist_tuning = np.exp(
            -0.5 * ((distances - self.preferred_distance) / self.distance_width) ** 2
        )

        # Direction tuning (von Mises, if directional)
        if self.preferred_direction is not None:
            if headings is None:
                raise ValueError(
                    "headings required when preferred_direction is set"
                )
            # Compute egocentric bearing to all objects (fully vectorized)
            # Use allocentric_to_egocentric which handles (n_time, n_objects, 2)
            # object_positions is (n_objects, 2), broadcast to (n_time, n_objects, 2)
            ego = allocentric_to_egocentric(
                self.object_positions, positions, headings
            )  # (n_time, n_objects, 2)

            # Extract bearing angles
            bearings = np.arctan2(ego[..., 1], ego[..., 0])  # (n_time, n_objects)

            # von Mises tuning with explicit kappa
            angle_diff = bearings - self.preferred_direction
            dir_tuning = np.exp(self.direction_kappa * np.cos(angle_diff))
            dir_tuning /= np.exp(self.direction_kappa)  # Normalize peak to 1
            combined = dist_tuning * dir_tuning
        else:
            combined = dist_tuning

        # Aggregate across objects
        if self.object_selectivity == "any":
            response = combined.max(axis=1)
        elif self.object_selectivity == "nearest":
            nearest_obj = distances.argmin(axis=1)
            response = combined[np.arange(n_time), nearest_obj]
        else:  # specific
            response = combined[:, self.specific_object_index]

        return self.baseline_rate + (self.max_rate - self.baseline_rate) * response

    @property
    def ground_truth(self) -> dict[str, Any]:
        """Return model parameters for validation."""
        return {
            "object_positions": self.object_positions.copy(),
            "preferred_distance": self.preferred_distance,
            "distance_width": self.distance_width,
            "preferred_direction": self.preferred_direction,
            "direction_kappa": self.direction_kappa,
            "object_selectivity": self.object_selectivity,
            "max_rate": self.max_rate,
            "baseline_rate": self.baseline_rate,
        }
```

### M2.2: Object-Vector Metrics

**File**: `src/neurospatial/metrics/object_vector_cells.py`

```python
"""Object-vector cell metrics and classification.

This module provides tools for:
- Computing object-vector tuning curves
- Measuring object-vector selectivity scores
- Classifying neurons as object-vector cells
- Visualizing object-vector tuning

References
----------
.. [1] Hoydal et al. (2019). Object-vector cells in the medial
       entorhinal cortex. Nature, 568(7752), 400-404.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from neurospatial import Environment
from neurospatial.reference_frames import (
    compute_egocentric_bearing,
    compute_egocentric_distance,
)


@dataclass(frozen=True)
class ObjectVectorMetrics:
    """Results from object-vector cell analysis.

    Attributes
    ----------
    preferred_distance : float
        Distance eliciting peak firing.
    preferred_direction : float
        Egocentric direction eliciting peak firing (radians).
    distance_selectivity : float
        Peak-to-mean ratio for distance tuning.
    direction_selectivity : float
        Mean vector length for direction tuning.
    object_vector_score : float
        Combined score (product of selectivities, normalized).
    peak_rate : float
        Maximum firing rate in tuning curve.
    mean_rate : float
        Mean firing rate.
    tuning_curve : NDArray
        2D tuning curve (distance x direction).
    distance_bins : NDArray
        Bin centers for distance dimension.
    direction_bins : NDArray
        Bin centers for direction dimension (radians).
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
        """Human-readable interpretation of metrics."""
        dir_deg = np.degrees(self.preferred_direction)
        if -45 <= dir_deg <= 45:
            direction_label = "ahead"
        elif 45 < dir_deg <= 135:
            direction_label = "left"
        elif -135 <= dir_deg < -45:
            direction_label = "right"
        else:
            direction_label = "behind"

        return (
            f"Object-vector cell: fires {self.preferred_distance:.1f} cm "
            f"{direction_label} of object (direction={dir_deg:.0f}°). "
            f"Score={self.object_vector_score:.2f}"
        )


def compute_object_vector_tuning(
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
    object_positions: NDArray[np.float64],
    distance_range: tuple[float, float] = (0.0, 50.0),
    n_distance_bins: int = 10,
    n_direction_bins: int = 12,
    min_occupancy_seconds: float = 0.1,
) -> ObjectVectorMetrics:
    """Compute object-vector tuning curve.

    Bins spikes by egocentric distance and direction to object(s),
    then computes occupancy-normalized firing rate map.

    Parameters
    ----------
    spike_times : NDArray, shape (n_spikes,)
        Spike timestamps.
    times : NDArray, shape (n_time,)
        Position timestamps.
    positions : NDArray, shape (n_time, 2)
        Animal positions.
    headings : NDArray, shape (n_time,)
        Animal heading directions (radians).
    object_positions : NDArray, shape (n_objects, 2)
        Object positions. If multiple objects, tuning is computed
        relative to nearest object at each timepoint.
    distance_range : tuple
        (min_distance, max_distance) for tuning curve.
    n_distance_bins : int
        Number of distance bins.
    n_direction_bins : int
        Number of direction bins (evenly spaced around circle).
    min_occupancy_seconds : float
        Minimum occupancy threshold in seconds. Bins with less occupancy
        are set to NaN. Default 0.1s matches typical sampling rates (>10 Hz).
        Set to 0.0 to include all bins (may be noisy in low-occupancy regions).

    Returns
    -------
    ObjectVectorMetrics
        Frozen dataclass with tuning curve and derived metrics.
    """
    ...


# Module-level constant for object-vector score normalization
# Typical upper bound for distance selectivity; can be calibrated via simulation
DEFAULT_MAX_DISTANCE_SELECTIVITY = 10.0


def object_vector_score(
    tuning_curve: NDArray[np.float64],
    distance_bins: NDArray[np.float64],
    direction_bins: NDArray[np.float64],
    max_distance_selectivity: float = DEFAULT_MAX_DISTANCE_SELECTIVITY,
) -> tuple[float, float, float]:
    """Compute object-vector score from tuning curve.

    Score combines:
    - Distance selectivity (s_d): peak-to-mean ratio
    - Direction selectivity (s_θ): mean vector length of marginal direction tuning

    Formulas
    --------
    Distance selectivity:
        s_d = max(λ_ij) / mean(λ_ij)

    Direction selectivity (reuses circular.mean_vector_length):
        marginal_θ = mean(λ over distance bins)
        s_θ = |Σ_j marginal_θ[j] * exp(i * θ_j)| / Σ_j marginal_θ[j]

    Combined score:
        s_OV = (s_d - 1) / (s_d* - 1) * s_θ
        where s_d* is max_distance_selectivity (default: 10)

    Parameters
    ----------
    tuning_curve : NDArray, shape (n_distance, n_direction)
        Occupancy-normalized firing rate.
    distance_bins : NDArray, shape (n_distance,)
        Distance bin centers.
    direction_bins : NDArray, shape (n_direction,)
        Direction bin centers (radians).
    max_distance_selectivity : float
        Reference upper bound for distance selectivity normalization.
        Default is DEFAULT_MAX_DISTANCE_SELECTIVITY (10.0). Adjust based
        on simulation calibration if needed.

    Returns
    -------
    tuple[float, float, float]
        (distance_selectivity, direction_selectivity, combined_score)

    Notes
    -----
    The default max_distance_selectivity=10.0 is calibrated from simulated
    object-vector cells with typical parameters (preferred_distance=10cm,
    distance_width=5cm). Strong OV cells produce s_d ≈ 8-12. Adjust this
    value if your simulation parameters differ significantly.

    See Also
    --------
    compute_object_vector_tuning : Compute the tuning curve input for this function.
    """
    from neurospatial.metrics.circular import _mean_resultant_length

    # Validate max_distance_selectivity
    if max_distance_selectivity <= 1:
        raise ValueError(
            f"max_distance_selectivity must be > 1, got {max_distance_selectivity}.\n\n"
            "This parameter normalizes the distance selectivity score. Values ≤1 "
            "would produce invalid normalization. Use the default (10.0) or a value "
            "calibrated from your simulation parameters."
        )

    # Mask NaN bins
    valid = ~np.isnan(tuning_curve)
    if not valid.any():
        return (np.nan, np.nan, np.nan)

    # Distance selectivity: peak-to-mean ratio
    peak = np.nanmax(tuning_curve)
    mean_rate = np.nanmean(tuning_curve)
    if mean_rate <= 0:
        return (np.nan, np.nan, np.nan)
    s_d = peak / mean_rate
    # s_d should always be >= 1 (peak >= mean), but guard against numerical issues
    s_d = max(s_d, 1.0)

    # Direction selectivity from marginal tuning curve
    # Uses mean resultant length (circular statistics)
    marginal_direction = np.nanmean(tuning_curve, axis=0)  # Average over distances
    s_theta = _mean_resultant_length(direction_bins, weights=marginal_direction)

    # Combined score (normalized)
    combined = ((s_d - 1) / (max_distance_selectivity - 1)) * s_theta
    combined = np.clip(combined, 0, 1)

    return (s_d, s_theta, combined)


def is_object_vector_cell(
    metrics: ObjectVectorMetrics,
    score_threshold: float = 0.3,
    min_peak_rate: float = 1.0,
) -> bool:
    """Classify neuron as object-vector cell.

    Parameters
    ----------
    metrics : ObjectVectorMetrics
        Results from compute_object_vector_tuning.
    score_threshold : float
        Minimum object_vector_score for classification.
    min_peak_rate : float
        Minimum peak firing rate (Hz).

    Returns
    -------
    bool
        True if neuron meets object-vector cell criteria.
    """
    return (
        metrics.object_vector_score >= score_threshold
        and metrics.peak_rate >= min_peak_rate
    )


def plot_object_vector_tuning(
    metrics: ObjectVectorMetrics,
    ax: "matplotlib.axes.Axes | None" = None,
    cmap: str = "viridis",
    show_peak: bool = True,
) -> "matplotlib.axes.Axes":
    """Plot object-vector tuning curve.

    Creates polar heatmap with distance on radial axis
    and egocentric direction on angular axis.
    """
    ...
```

### M2.3: Object-Vector Field Computation

**File**: `src/neurospatial/object_vector_field.py`

> **Design Note**: This module is separate from `spike_field.py` (which contains
> `compute_place_field()`) because object-vector fields require egocentric
> coordinate machinery and return a different result type. Consider consolidating
> into a unified `fields/` subpackage in a future refactor if the pattern expands.

```python
"""Compute object-vector fields from spike data.

Analogous to compute_place_field() but in egocentric coordinates
relative to objects.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ObjectVectorFieldResult:
    """Result from object-vector field computation.

    Attributes
    ----------
    field : NDArray, shape (n_distance_bins, n_angle_bins)
        Firing rate in egocentric (distance, angle) space.
    ego_env : Environment
        Egocentric polar environment used for binning.
    occupancy : NDArray
        Time spent in each (distance, angle) bin.
    """
    field: NDArray[np.float64]
    ego_env: "Environment"
    occupancy: NDArray[np.float64]


def compute_object_vector_field(
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
    object_positions: NDArray[np.float64],
    distance_range: tuple[float, float] = (0.0, 50.0),
    distance_bin_size: float = 2.0,
    angle_bin_size: float = np.pi / 12,
    method: Literal["binned", "diffusion_kde"] = "binned",
    bandwidth: float | None = None,
    min_occupancy_seconds: float = 0.1,
    allocentric_env: "Environment | None" = None,
) -> ObjectVectorFieldResult:
    """Compute firing rate as function of egocentric vector to objects.

    Creates an internal egocentric polar environment and bins spikes
    by (distance, angle) to nearest object.

    Parameters
    ----------
    spike_times : NDArray
        Spike timestamps.
    times : NDArray
        Position timestamps.
    positions : NDArray, shape (n_time, 2)
        Animal positions.
    headings : NDArray, shape (n_time,)
        Animal headings (radians).
    object_positions : NDArray, shape (n_objects, 2)
        Object locations. Multiple objects are supported:
        - Spikes binned by egocentric vector to **nearest object** at each timepoint
        - Returns single aggregated field (not per-object fields)
        For per-object analysis, call this function once per object.
    distance_range : tuple
        (min, max) distance from objects.
    distance_bin_size : float
        Bin size for distance dimension.
    angle_bin_size : float
        Bin size for angle dimension (radians).
    method : {"binned", "diffusion_kde"}
        Estimation method. "diffusion_kde" uses the same smoothing
        as compute_place_field for consistency.
    bandwidth : float, optional
        Smoothing bandwidth (for diffusion_kde).
    min_occupancy_seconds : float
        Minimum occupancy threshold in seconds. Bins with less occupancy
        are set to NaN. Default 0.1s matches `compute_place_field()` default
        for consistency. For place field comparison, use same value for both.
    allocentric_env : Environment, optional
        If provided and method uses geodesic distance, uses this
        environment's connectivity for distance computation.

    Returns
    -------
    ObjectVectorFieldResult
        Contains field, egocentric environment, and occupancy.

    Notes
    -----
    The returned ego_env is a polar grid where:
    - Dimension 0 = distance to nearest object
    - Dimension 1 = egocentric angle to nearest object

    Smoothing uses the same diffusion kernel machinery as
    compute_place_field() to ensure consistent behavior.
    """
    from neurospatial import Environment
    from neurospatial.reference_frames import (
        compute_egocentric_bearing,
        compute_egocentric_distance,
    )

    # 1. Create egocentric polar environment
    ego_env = Environment.from_egocentric_grid(
        distance_range=distance_range,
        distance_bin_size=distance_bin_size,
        angle_bin_size=angle_bin_size,
    )

    # 2. Compute distance and bearing to nearest object at each timepoint
    object_positions = np.atleast_2d(object_positions)

    # Use geodesic distance if allocentric environment provided
    if allocentric_env is not None:
        distances = compute_egocentric_distance(
            object_positions, positions, env=allocentric_env, metric="geodesic"
        )
    else:
        distances = compute_egocentric_distance(
            object_positions, positions, metric="euclidean"
        )

    bearings = compute_egocentric_bearing(object_positions, positions, headings)

    # Find nearest object
    nearest_idx = np.nanargmin(distances, axis=1)  # nanargmin handles NaN geodesic
    nearest_distance = distances[np.arange(len(distances)), nearest_idx]
    nearest_bearing = bearings[np.arange(len(bearings)), nearest_idx]

    # 3. Map to egocentric bins
    ego_positions = np.column_stack([nearest_distance, nearest_bearing])
    # ... rest follows compute_place_field pattern with ego_env
    ...
```

### M2.4: Object-Vector Overlay (Animation)

**File**: `src/neurospatial/animation/overlays/object_vector.py`

```python
"""Animation overlay for object-vector cell visualization."""

@dataclass
class ObjectVectorOverlay:
    """Overlay showing vectors from animal to objects.

    Displays lines from current animal position to object locations,
    with optional color-coding by firing rate or distance.

    Parameters
    ----------
    object_positions : NDArray, shape (n_objects, 2)
        Object locations in environment coordinates.
    animal_positions : NDArray, shape (n_frames, 2)
        Animal positions at each frame.
    firing_rates : NDArray, shape (n_frames,), optional
        Firing rates for color-coding vectors.
    times : NDArray, shape (n_frames,), optional
        Timestamps for temporal alignment.
    color : str or NDArray
        Vector color (single color or per-object).
    linewidth : float
        Vector line width.
    show_objects : bool
        Whether to mark object locations.
    object_marker : str
        Marker style for objects.
    object_size : float
        Marker size for objects.
    """
    object_positions: NDArray[np.float64]
    animal_positions: NDArray[np.float64]
    firing_rates: NDArray[np.float64] | None = None
    times: NDArray[np.float64] | None = None
    color: str = "yellow"
    linewidth: float = 1.5
    show_objects: bool = True
    object_marker: str = "s"
    object_size: float = 10.0
    interp: Literal["linear", "nearest"] = "linear"

    def convert_to_data(self, frame_times, n_frames, env):
        """Convert to internal data format aligned to frames."""
        ...
```

### M2.5: Tests for Object-Vector Cells

**File**: `tests/simulation/models/test_object_vector_cells.py`
**File**: `tests/metrics/test_object_vector_cells.py`

**Test categories**:
1. Model validation (parameters, ground_truth)
2. Firing rate computation (distance tuning, direction tuning)
3. Object selectivity modes (any, nearest, specific)
4. Geodesic vs Euclidean distances
5. Edge cases (no objects, single object, many objects)
6. Metrics computation and classification
7. Visualization output

---

## Milestone 3: Spatial View Cells

**Goal**: Implement spatial view cell models, metrics, and analysis.

**Priority**: MEDIUM (more complex, primate-specific)

**Dependencies**: M1 (egocentric transforms), partially M2 (shared patterns)

### M3.1: Visibility/Gaze Computation

**File**: `src/neurospatial/visibility.py`

```python
"""Visibility and gaze direction computations.

Tools for determining what spatial location an animal is viewing,
based on head direction, gaze direction, and environment geometry.
"""

from typing import Literal
import numpy as np
from numpy.typing import NDArray

from neurospatial import Environment


def compute_viewed_location(
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
    gaze_offsets: NDArray[np.float64] | None = None,
    view_distance: float = 50.0,
    env: Environment | None = None,
    method: Literal["fixed_distance", "ray_cast", "boundary"] = "fixed_distance",
) -> NDArray[np.float64]:
    """Compute the spatial location being viewed at each timepoint.

    Parameters
    ----------
    positions : NDArray, shape (n_time, 2)
        Animal positions.
    headings : NDArray, shape (n_time,)
        Head directions (radians).
    gaze_offsets : NDArray, shape (n_time,), optional
        Gaze direction relative to head (radians). If None, assumes
        gaze aligned with head.
    view_distance : float
        Fixed viewing distance (for "fixed_distance" method).
    env : Environment, optional
        Environment for ray casting (required for "ray_cast" and "boundary").
    method : {"fixed_distance", "ray_cast", "boundary"}
        How to determine viewed location:
        - "fixed_distance": point at fixed distance in gaze direction
        - "ray_cast": intersection with environment boundary
        - "boundary": nearest boundary point in gaze direction

    Returns
    -------
    NDArray, shape (n_time, 2)
        Viewed locations. NaN if viewing outside environment.

    Examples
    --------
    >>> positions = np.array([[10, 10], [20, 20]])
    >>> headings = np.array([0, np.pi/2])  # East, North
    >>> viewed = compute_viewed_location(positions, headings, view_distance=30)
    >>> viewed[0]  # Looking East from (10,10)
    array([40., 10.])
    >>> viewed[1]  # Looking North from (20,20)
    array([20., 50.])
    """
    gaze_directions = headings if gaze_offsets is None else headings + gaze_offsets

    if method == "fixed_distance":
        dx = view_distance * np.cos(gaze_directions)
        dy = view_distance * np.sin(gaze_directions)
        viewed = positions + np.column_stack([dx, dy])

        # Mark as NaN if outside environment
        if env is not None:
            outside = ~env.contains(viewed)
            viewed[outside] = np.nan

        return viewed

    elif method == "ray_cast":
        return _ray_cast_to_boundary(positions, gaze_directions, env)

    elif method == "boundary":
        return _nearest_boundary_in_direction(positions, gaze_directions, env)


def _ray_cast_to_boundary(
    positions: NDArray[np.float64],
    directions: NDArray[np.float64],
    env: Environment,
    max_distance: float = 1000.0,
    step_size: float = 1.0,
) -> NDArray[np.float64]:
    """Cast rays until hitting environment boundary.

    Uses iterative stepping with binary search refinement.
    """
    ...


def compute_view_field(
    env: Environment,
    position: NDArray[np.float64],
    heading: float,
    fov: "FieldOfView | float" = np.pi / 3,  # 60° half-angle → 120° total FOV
    n_rays: int = 30,
) -> NDArray[np.bool_]:
    """Compute visible bins from a given position and heading.

    Returns binary mask of which bins are visible (within field of view
    and not occluded by boundaries).

    Parameters
    ----------
    env : Environment
        Spatial environment.
    position : NDArray, shape (2,)
        Viewing position.
    heading : float
        Head direction (radians).
    fov : FieldOfView or float
        Field of view specification. If float, treated as symmetric half-angle.
        Use FieldOfView.rat() for rodents, FieldOfView.primate() for primates.
    n_rays : int
        Number of rays for visibility sampling.

    Returns
    -------
    NDArray, shape (n_bins,)
        Boolean mask of visible bins.

    See Also
    --------
    compute_viewshed : More comprehensive viewshed with boundary/cue visibility.
    """
    ...


@dataclass(frozen=True)
class FieldOfView:
    """Species-specific field of view configuration.

    Supports asymmetric, multi-region FOV models for different species.
    Rodents have panoramic lateral vision with narrow binocular overlap;
    primates have forward-facing eyes with large binocular overlap.

    Attributes
    ----------
    left_angle : float
        Maximum angle to the left of heading (radians, positive).
    right_angle : float
        Maximum angle to the right of heading (radians, positive).
    binocular_half_angle : float, optional
        Half-angle of binocular overlap region (radians).
        If None, no binocular region is defined.
    blind_spot_behind : float, optional
        Angle of blind spot behind animal (radians). Default 0 (no blind spot).

    Choosing a Field of View
    ------------------------
    **Use species presets when available**:
    - FieldOfView.rat() / .mouse()  - Rodent experiments (lateral eyes, panoramic vision)
    - FieldOfView.primate()         - Primate experiments (forward eyes, binocular vision)
    - FieldOfView.bat()             - Bat experiments (echolocation + limited vision)

    **Use symmetric FOV for**:
    - Custom/unknown species
    - Artificial restrictions (e.g., only analyze forward 120°)
    - Omnidirectional analysis: FieldOfView.symmetric(np.pi)  # Full 360°

    **Omit FOV parameter** (pass None) for omnidirectional viewshed (no directional restriction).

    Note: Species presets use literature-averaged values. For most analyses, small variations
    between individuals have negligible impact on results.

    Examples
    --------
    >>> # Rat: ~300° FOV with small binocular region
    >>> rat_fov = FieldOfView.rat()
    >>> rat_fov.total_angle_degrees
    300.0

    >>> # Primate: ~180° FOV with large binocular overlap
    >>> primate_fov = FieldOfView.primate()
    >>> primate_fov.binocular_half_angle_degrees
    60.0

    >>> # Custom symmetric FOV
    >>> fov = FieldOfView.symmetric(half_angle=np.pi/3)  # 120° total
    """
    left_angle: float  # Positive, from heading
    right_angle: float  # Positive, from heading
    binocular_half_angle: float | None = None
    blind_spot_behind: float = 0.0

    def __post_init__(self):
        """Validate FOV parameters."""
        if self.left_angle <= 0:
            raise ValueError(
                f"left_angle must be positive, got {self.left_angle:.4f} radians.\n\n"
                "FOV angles represent the angular extent from center heading.\n"
                "Use species presets for common configurations:\n"
                "- FieldOfView.rat() for rodents (~300° panoramic)\n"
                "- FieldOfView.primate() for primates (~180° forward)\n"
                "- FieldOfView.symmetric(half_angle) for custom symmetric FOV"
            )
        if self.right_angle <= 0:
            raise ValueError(
                f"right_angle must be positive, got {self.right_angle:.4f} radians.\n\n"
                "FOV angles represent the angular extent from center heading.\n"
                "Use species presets for common configurations:\n"
                "- FieldOfView.rat() for rodents (~300° panoramic)\n"
                "- FieldOfView.primate() for primates (~180° forward)\n"
                "- FieldOfView.symmetric(half_angle) for custom symmetric FOV"
            )
        if self.blind_spot_behind < 0 or self.blind_spot_behind > 2 * np.pi:
            raise ValueError(
                f"blind_spot_behind must be in [0, 2π], got {self.blind_spot_behind:.4f} radians.\n\n"
                "The blind spot is the angular region directly behind the animal.\n"
                "- 0 means no blind spot (can see behind)\n"
                "- π means 180° blind spot (no rear vision, like primates)\n"
                "- 2π is maximum (blind in all directions, invalid)"
            )
        if self.binocular_half_angle is not None:
            if self.binocular_half_angle < 0:
                raise ValueError(
                    f"binocular_half_angle must be non-negative, got {self.binocular_half_angle:.4f}.\n\n"
                    "Set to None if binocular region is not relevant for your analysis."
                )
            max_binocular = min(self.left_angle, self.right_angle)
            if self.binocular_half_angle > max_binocular:
                raise ValueError(
                    f"binocular_half_angle ({self.binocular_half_angle:.4f}) cannot exceed "
                    f"min(left_angle, right_angle) = {max_binocular:.4f}.\n\n"
                    "The binocular region must be within the monocular FOV."
                )

    @classmethod
    def symmetric(cls, half_angle: float) -> "FieldOfView":
        """Create symmetric FOV with given half-angle."""
        return cls(left_angle=half_angle, right_angle=half_angle)

    @classmethod
    def rat(cls) -> "FieldOfView":
        """Rat/mouse field of view (~300° total, ~40° binocular).

        Based on: Hughes (1979), Heffner & Heffner (1992)
        - Lateral eye placement gives panoramic vision
        - Small frontal binocular overlap (~40-60°)
        - Small blind spot directly behind (~60°)
        """
        return cls(
            left_angle=np.deg2rad(150),  # 150° to left
            right_angle=np.deg2rad(150),  # 150° to right
            binocular_half_angle=np.deg2rad(20),  # 40° binocular
            blind_spot_behind=np.deg2rad(60),  # 60° blind behind
        )

    @classmethod
    def mouse(cls) -> "FieldOfView":
        """Mouse field of view (~310° total, ~40° binocular)."""
        return cls(
            left_angle=np.deg2rad(155),
            right_angle=np.deg2rad(155),
            binocular_half_angle=np.deg2rad(20),
            blind_spot_behind=np.deg2rad(50),
        )

    @classmethod
    def primate(cls) -> "FieldOfView":
        """Primate field of view (~180° total, ~120° binocular).

        Based on: Kaas (2013), Heesy (2009)
        - Forward-facing eyes for depth perception
        - Large binocular overlap
        - No vision behind
        """
        return cls(
            left_angle=np.deg2rad(90),
            right_angle=np.deg2rad(90),
            binocular_half_angle=np.deg2rad(60),
            blind_spot_behind=np.deg2rad(180),  # No rear vision
        )

    @classmethod
    def bat(cls) -> "FieldOfView":
        """Bat field of view (~270° total, ~20° binocular).

        Note: Bats rely heavily on echolocation; visual FOV is secondary.
        """
        return cls(
            left_angle=np.deg2rad(135),
            right_angle=np.deg2rad(135),
            binocular_half_angle=np.deg2rad(10),
            blind_spot_behind=np.deg2rad(90),
        )

    @property
    def total_angle(self) -> float:
        """Total field of view in radians."""
        return self.left_angle + self.right_angle

    @property
    def total_angle_degrees(self) -> float:
        """Total field of view in degrees."""
        return np.rad2deg(self.total_angle)

    @property
    def binocular_half_angle_degrees(self) -> float | None:
        """Binocular half-angle in degrees."""
        if self.binocular_half_angle is None:
            return None
        return np.rad2deg(self.binocular_half_angle)

    def contains_angle(self, bearing: float | NDArray[np.float64]) -> NDArray[np.bool_]:
        """Check if bearing(s) fall within field of view.

        Parameters
        ----------
        bearing : float or NDArray
            Egocentric bearing(s) in radians (0=ahead, positive=left, ±π=behind).

        Returns
        -------
        NDArray[np.bool_]
            True if bearing is within FOV.

        Notes
        -----
        The blind spot is centered at ±π (directly behind the animal).
        A blind_spot_behind of 60° means angles within 30° of ±π are excluded.
        """
        bearing = np.atleast_1d(bearing)

        # Check if within left/right angular bounds
        in_fov = (bearing >= -self.right_angle) & (bearing <= self.left_angle)

        # Exclude blind spot behind animal (centered at ±π)
        if self.blind_spot_behind > 0:
            # Distance from behind (±π) - handle wrap-around
            dist_from_behind = np.minimum(
                np.abs(bearing - np.pi),
                np.abs(bearing + np.pi)
            )
            not_in_blind_spot = dist_from_behind > (self.blind_spot_behind / 2)
            in_fov = in_fov & not_in_blind_spot

        return in_fov

    def is_binocular(self, bearing: float | NDArray[np.float64]) -> NDArray[np.bool_]:
        """Check if bearing(s) fall within binocular region.

        Parameters
        ----------
        bearing : float or NDArray
            Egocentric bearing(s) in radians.

        Returns
        -------
        NDArray[np.bool_]
            True if bearing is in binocular region (both eyes can see).
        """
        if self.binocular_half_angle is None:
            return np.zeros_like(np.atleast_1d(bearing), dtype=bool)
        return np.abs(np.atleast_1d(bearing)) <= self.binocular_half_angle


@dataclass(frozen=True)
class ViewshedResult:
    """Result from viewshed analysis.

    Attributes
    ----------
    visible_bins : NDArray[np.bool_], shape (n_bins,)
        Boolean mask of visible bins from observer position.
    visible_boundary_segments : list[tuple[int, int]]
        List of (start_idx, end_idx) pairs for visible boundary segments.
    visible_cues : NDArray[np.bool_], shape (n_cues,)
        Boolean mask of visible cues/landmarks.
    cue_distances : NDArray[np.float64], shape (n_cues,)
        Distance to each cue (NaN if not visible).
    cue_bearings : NDArray[np.float64], shape (n_cues,)
        Egocentric bearing to each cue (NaN if not visible).
    occlusion_map : NDArray[np.float64], shape (n_bins,)
        For each bin, distance to nearest occluding boundary (0 if visible).
    """
    visible_bins: NDArray[np.bool_]
    visible_boundary_segments: list[tuple[int, int]]
    visible_cues: NDArray[np.bool_]
    cue_distances: NDArray[np.float64]
    cue_bearings: NDArray[np.float64]
    occlusion_map: NDArray[np.float64]

    @property
    def n_visible_bins(self) -> int:
        """Number of visible bins."""
        return int(self.visible_bins.sum())

    @property
    def visibility_fraction(self) -> float:
        """Fraction of environment visible."""
        return float(self.visible_bins.mean())

    @property
    def n_visible_cues(self) -> int:
        """Number of visible cues."""
        return int(self.visible_cues.sum())


def compute_viewshed(
    env: Environment,
    position: NDArray[np.float64],
    heading: float | None = None,
    fov: FieldOfView | float | None = None,
    cue_positions: NDArray[np.float64] | None = None,
    n_rays: int = 360,
    max_distance: float | None = None,
) -> ViewshedResult:
    """Compute viewshed (visible area) from observer position.

    A viewshed is the set of all bins visible from a given position,
    accounting for occlusion by environment boundaries. Optionally
    restricts to a field of view based on heading.

    Parameters
    ----------
    env : Environment
        Spatial environment with boundary information.
    position : NDArray, shape (2,)
        Observer position.
    heading : float, optional
        Observer heading (radians). If None, computes 360° viewshed.
    fov : FieldOfView or float, optional
        Field of view specification. Can be:
        - FieldOfView object (supports asymmetric, species-specific FOV)
        - float: symmetric half-angle in radians (legacy interface)
        - None: full 360° viewshed
        Use FieldOfView.rat(), FieldOfView.primate(), etc. for species presets.
    cue_positions : NDArray, shape (n_cues, 2), optional
        Positions of cues/landmarks to check visibility.
    n_rays : int
        Number of rays for visibility sampling (default 360 = 1° resolution).
    max_distance : float, optional
        Maximum viewing distance. If None, uses environment extent.

    Returns
    -------
    ViewshedResult
        Comprehensive visibility analysis including visible bins,
        boundary segments, and cue visibility.

    Examples
    --------
    >>> from neurospatial import Environment
    >>> from neurospatial.visibility import compute_viewshed, FieldOfView
    >>> env = Environment.from_samples(positions, bin_size=2.0)

    >>> # Full 360° viewshed
    >>> result = compute_viewshed(env, position=[50, 50])
    >>> print(f"{result.visibility_fraction:.1%} of environment visible")

    >>> # Simple symmetric FOV (60° half-angle = 120° total)
    >>> result = compute_viewshed(
    ...     env, position=[50, 50], heading=0.0, fov=np.pi/3
    ... )

    >>> # Rat-specific panoramic FOV (~300°)
    >>> result = compute_viewshed(
    ...     env, position=[50, 50], heading=0.0, fov=FieldOfView.rat()
    ... )

    >>> # Primate-specific forward FOV (~180°)
    >>> result = compute_viewshed(
    ...     env, position=[50, 50], heading=0.0, fov=FieldOfView.primate()
    ... )

    Notes
    -----
    Ray casting is performed using Shapely for efficient boundary
    intersection. The algorithm:

    1. Cast n_rays from observer position in directions spanning FOV
    2. Find intersection with environment boundary for each ray
    3. Mark bins between observer and intersection as visible
    4. For cues, check if line-of-sight is occluded

    Complexity is O(n_rays * n_boundary_segments) with spatial indexing.

    Species-Specific Considerations
    -------------------------------
    Rodents (rats, mice): Use FieldOfView.rat() or FieldOfView.mouse()
        - Panoramic vision (~300-320°) from laterally-placed eyes
        - Small binocular overlap (~40-60°) for depth perception
        - Small blind spot directly behind

    Primates: Use FieldOfView.primate()
        - Forward-facing eyes (~180° total)
        - Large binocular overlap (~120°) for stereoscopic depth
        - No rear vision

    Bats: Use FieldOfView.bat()
        - Visual FOV secondary to echolocation
        - ~270° visual coverage
    """
    # Normalize fov parameter
    if fov is None:
        fov_obj = FieldOfView.symmetric(np.pi)  # Full 360°
    elif isinstance(fov, (int, float)):
        fov_obj = FieldOfView.symmetric(float(fov))
    else:
        fov_obj = fov
    ...


def visible_boundaries(
    env: Environment,
    position: NDArray[np.float64],
    heading: float | None = None,
    fov: "FieldOfView | float | None" = None,
) -> list[NDArray[np.float64]]:
    """Get visible boundary segments from observer position.

    Returns the portions of environment boundary that are visible
    (not occluded by other boundary segments).

    Parameters
    ----------
    env : Environment
        Spatial environment.
    position : NDArray, shape (2,)
        Observer position.
    heading : float, optional
        Observer heading for FOV restriction.
    fov : FieldOfView or float, optional
        Field of view specification. If float, treated as symmetric half-angle.
        If None, uses 360° (all directions visible).
        Use FieldOfView.rat() for rodents, FieldOfView.primate() for primates.

    Returns
    -------
    list[NDArray]
        List of visible boundary segments, each shape (n_points, 2).
        Segments are ordered by angle from observer.

    Notes
    -----
    Useful for boundary cell analysis - allows computing what portion
    of each boundary is visible from the animal's current location.
    """
    ...


def visible_cues(
    env: Environment,
    position: NDArray[np.float64],
    heading: float,
    cue_positions: NDArray[np.float64],
    fov: FieldOfView | float = np.pi / 3,
) -> tuple[NDArray[np.bool_], NDArray[np.float64], NDArray[np.float64]]:
    """Determine which cues are visible from observer position.

    Checks line-of-sight visibility for each cue, accounting for
    occlusion by environment boundaries and field of view.

    Parameters
    ----------
    env : Environment
        Spatial environment.
    position : NDArray, shape (2,)
        Observer position.
    heading : float
        Observer heading (radians).
    cue_positions : NDArray, shape (n_cues, 2)
        Cue/landmark positions.
    fov : FieldOfView or float
        Field of view specification. Can be:
        - FieldOfView object (e.g., FieldOfView.rat(), FieldOfView.primate())
        - float: symmetric half-angle in radians (legacy interface)
        Default is np.pi/3 (60° half-angle = 120° total).

    Returns
    -------
    visible : NDArray[np.bool_], shape (n_cues,)
        Boolean mask of visible cues.
    distances : NDArray[np.float64], shape (n_cues,)
        Distance to each cue (NaN if not visible).
    bearings : NDArray[np.float64], shape (n_cues,)
        Egocentric bearing to each cue (NaN if not visible).

    Examples
    --------
    >>> landmarks = np.array([[30, 50], [70, 50], [50, 30]])
    >>> # Using rat FOV (~300° panoramic)
    >>> visible, distances, bearings = visible_cues(
    ...     env, position=[50, 50], heading=0.0,
    ...     cue_positions=landmarks, fov=FieldOfView.rat()
    ... )
    >>> # Using simple symmetric FOV
    >>> visible, distances, bearings = visible_cues(
    ...     env, position=[50, 50], heading=0.0,
    ...     cue_positions=landmarks, fov=np.pi/3
    ... )
    """
    from neurospatial.reference_frames import compute_egocentric_bearing

    # Normalize fov parameter
    if isinstance(fov, (int, float)):
        fov_obj = FieldOfView.symmetric(float(fov))
    else:
        fov_obj = fov

    n_cues = len(cue_positions)
    visible = np.zeros(n_cues, dtype=bool)
    distances = np.full(n_cues, np.nan)
    bearings = np.full(n_cues, np.nan)

    # Compute bearings to all cues
    all_bearings = compute_egocentric_bearing(
        cue_positions, position[np.newaxis], np.array([heading])
    )[0]  # Shape: (n_cues,)

    # Check FOV using species-aware method
    in_fov = fov_obj.contains_angle(all_bearings)

    # Check line-of-sight for cues in FOV
    for i in np.where(in_fov)[0]:
        if _line_of_sight_clear(env, position, cue_positions[i]):
            visible[i] = True
            distances[i] = np.linalg.norm(cue_positions[i] - position)
            bearings[i] = all_bearings[i]

    return visible, distances, bearings


def _line_of_sight_clear(
    env: Environment,
    start: NDArray[np.float64],
    end: NDArray[np.float64],
) -> bool:
    """Check if line of sight between two points is clear.

    Uses Shapely to check for boundary intersections.
    """
    from shapely.geometry import LineString

    if not hasattr(env, 'boundary') or env.boundary is None:
        return True  # No boundary to occlude

    line = LineString([start, end])
    # Check intersection with boundary (excluding endpoints)
    intersection = line.intersection(env.boundary)
    # Line of sight is clear if no interior intersection
    return intersection.is_empty or intersection.equals(line)


def compute_viewshed_trajectory(
    env: Environment,
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
    cue_positions: NDArray[np.float64] | None = None,
    fov: FieldOfView | float = np.pi / 3,
    n_rays: int = 60,
) -> list[ViewshedResult]:
    """Compute viewshed along a trajectory.

    Vectorized computation of viewshed at each position in trajectory.

    Parameters
    ----------
    env : Environment
        Spatial environment.
    positions : NDArray, shape (n_time, 2)
        Observer positions.
    headings : NDArray, shape (n_time,)
        Observer headings.
    cue_positions : NDArray, shape (n_cues, 2), optional
        Cue positions to track visibility.
    fov : FieldOfView or float
        Field of view specification. See compute_viewshed for details.
        Use FieldOfView.rat() for rodents, FieldOfView.primate() for primates.
    n_rays : int
        Rays per viewshed (lower for performance).

    Returns
    -------
    list[ViewshedResult]
        Viewshed at each timepoint.

    Notes
    -----
    For performance, consider subsampling trajectory or using
    `visible_cues()` directly if only cue visibility is needed.
    """
    results = []
    for pos, head in zip(positions, headings, strict=True):
        result = compute_viewshed(
            env, pos, heading=head, fov=fov,
            cue_positions=cue_positions, n_rays=n_rays,
        )
        results.append(result)
    return results


def visibility_occupancy(
    env: Environment,
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
    times: NDArray[np.float64],
    fov: FieldOfView | float = np.pi / 3,
    n_rays: int = 60,
) -> NDArray[np.float64]:
    """Compute how long each bin was visible during trajectory.

    Analogous to spatial occupancy, but in "viewed space" rather
    than physical space. Useful for normalizing spatial view fields.

    Parameters
    ----------
    env : Environment
        Spatial environment.
    positions : NDArray, shape (n_time, 2)
        Observer positions.
    headings : NDArray, shape (n_time,)
        Observer headings.
    times : NDArray, shape (n_time,)
        Timestamps.
    fov : FieldOfView or float
        Field of view specification. See compute_viewshed for details.
        Use FieldOfView.rat() for rodents, FieldOfView.primate() for primates.
    n_rays : int
        Rays for viewshed computation.

    Returns
    -------
    NDArray, shape (n_bins,)
        Time (seconds) each bin was visible.

    Notes
    -----
    This is related to but distinct from the view occupancy computed
    by `compute_spatial_view_field()`, which tracks the *viewed location*
    (single point per timepoint) rather than the full visible area.
    """
    dt = np.diff(times, prepend=times[0] - np.diff(times).mean())
    visibility_time = np.zeros(env.n_bins)

    for i, (pos, head) in enumerate(zip(positions, headings, strict=True)):
        viewshed = compute_viewshed(
            env, pos, heading=head, fov=fov, n_rays=n_rays
        )
        visibility_time[viewshed.visible_bins] += dt[i]

    return visibility_time
```

### M3.2: Spatial View Cell Model (Simulation)

**File**: `src/neurospatial/simulation/models/spatial_view_cells.py`

```python
"""Spatial view cell simulation model.

Spatial view cells (SVCs) fire based on the location being viewed,
independent of the animal's physical position. They are found in
primate hippocampus and parahippocampal regions.

Key Properties
--------------
- Fire when animal views specific location (not when physically there)
- Require gaze/head direction information
- Tuning is in allocentric "viewed location" space
- Can be combined with physical position for conjunctive coding

References
----------
.. [1] Rolls et al. (1997). Spatial view cells in the primate
       hippocampus. European Journal of Neuroscience, 9(8), 1789-1794.
.. [2] Georges-Francois et al. (1999). Spatial view cells in the
       primate hippocampus: allocentric view not head direction or
       eye position. Cerebral Cortex, 9(3), 197-212.
"""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from neurospatial import Environment
from neurospatial.visibility import compute_viewed_location


@dataclass
class SpatialViewCellModel:
    """Simulates spatial view cell firing.

    Parameters
    ----------
    env : Environment
        Spatial environment.
    preferred_view_location : NDArray, shape (2,)
        Location in allocentric space that elicits maximum firing
        when viewed.
    view_field_width : float
        Gaussian tuning width (sigma) in environment units.
    view_distance : float
        How far ahead the animal "sees" (for viewed location computation).
    gaze_model : {"fixed_distance", "ray_cast", "boundary"}
        How to compute viewed location from gaze direction.
    max_rate : float
        Maximum firing rate (Hz).
    baseline_rate : float
        Baseline firing rate (Hz).
    require_visibility : bool
        If True, only fire when preferred location is actually visible
        (within field of view). If False, fire based on where gaze
        direction points regardless of occlusions.
    fov : FieldOfView or float
        Field of view specification. Use FieldOfView.rat() for rodents,
        FieldOfView.primate() for primates, or float for symmetric half-angle.
        Only used if require_visibility=True.
    seed : int, optional
        Random seed.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.simulation import SpatialViewCellModel
    >>>
    >>> # Create environment
    >>> positions = np.random.rand(1000, 2) * 100
    >>> env = Environment.from_samples(positions, bin_size=2.0)
    >>>
    >>> # Create cell that fires when looking at center
    >>> cell = SpatialViewCellModel(
    ...     env=env,
    ...     preferred_view_location=np.array([50, 50]),
    ...     view_field_width=10.0,
    ...     view_distance=30.0,
    ... )
    >>>
    >>> # Generate firing rates
    >>> headings = np.random.uniform(-np.pi, np.pi, len(positions))
    >>> rates = cell.firing_rate(positions, headings=headings)
    """

    env: Environment
    preferred_view_location: NDArray[np.float64]
    view_field_width: float = 10.0
    view_distance: float = 30.0
    gaze_model: Literal["fixed_distance", "ray_cast", "boundary"] = "fixed_distance"
    max_rate: float = 20.0
    baseline_rate: float = 1.0
    require_visibility: bool = False
    fov: FieldOfView | float = np.pi / 3  # Or FieldOfView.rat(), etc.
    seed: int | None = None

    def firing_rate(
        self,
        positions: NDArray[np.float64],
        times: NDArray[np.float64] | None = None,
        headings: NDArray[np.float64] | None = None,
        gaze_offsets: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Compute firing rate based on viewed location.

        Parameters
        ----------
        positions : NDArray, shape (n_time, 2)
            Animal positions.
        times : NDArray, optional
            Timestamps (unused, for protocol compatibility).
        headings : NDArray, shape (n_time,)
            Head directions (radians). Required.
        gaze_offsets : NDArray, shape (n_time,), optional
            Gaze direction relative to head.

        Returns
        -------
        NDArray, shape (n_time,)
            Firing rates in Hz.
        """
        if headings is None:
            raise ValueError("headings required for SpatialViewCellModel")

        positions = np.atleast_2d(positions)

        # Compute viewed location at each timepoint
        viewed_locations = compute_viewed_location(
            positions, headings, gaze_offsets,
            view_distance=self.view_distance,
            env=self.env,
            method=self.gaze_model,
        )

        # Distance from viewed location to preferred view location
        distances = np.linalg.norm(
            viewed_locations - self.preferred_view_location, axis=-1
        )

        # Gaussian tuning
        tuning = np.exp(-0.5 * (distances / self.view_field_width) ** 2)

        # Handle NaN (viewing outside environment)
        tuning = np.nan_to_num(tuning, nan=0.0)

        # Optional: check if preferred location is actually visible
        if self.require_visibility:
            # TODO: implement visibility check
            pass

        return self.baseline_rate + (self.max_rate - self.baseline_rate) * tuning

    @property
    def ground_truth(self) -> dict[str, Any]:
        """Return model parameters for validation."""
        return {
            "preferred_view_location": self.preferred_view_location.copy(),
            "view_field_width": self.view_field_width,
            "view_distance": self.view_distance,
            "gaze_model": self.gaze_model,
            "max_rate": self.max_rate,
            "baseline_rate": self.baseline_rate,
        }
```

### M3.3: Spatial View Field Analysis

**File**: `src/neurospatial/spatial_view_field.py`

> **Design Note**: See M2.3 note on file organization. Same rationale applies:
> spatial view fields require visibility computation and a different workflow.

```python
"""Compute spatial view fields from spike data.

Spatial view fields are rate maps in "viewed location" space rather
than physical location space.
"""

def compute_spatial_view_field(
    env: Environment,
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
    gaze_offsets: NDArray[np.float64] | None = None,
    view_distance: float = 30.0,
    gaze_model: Literal["fixed_distance", "ray_cast"] = "fixed_distance",
    method: Literal["binned", "diffusion_kde", "gaussian_kde"] = "diffusion_kde",
    bandwidth: float | None = None,
    min_occupancy_seconds: float = 0.1,
) -> NDArray[np.float64]:
    """Compute firing rate as function of viewed location.

    Instead of binning spikes by animal position, bins by the location
    the animal was viewing when each spike occurred.

    Parameters
    ----------
    env : Environment
        Spatial environment (used for binning viewed locations).
    spike_times : NDArray
        Spike timestamps.
    times : NDArray
        Position/heading timestamps.
    positions : NDArray, shape (n_time, 2)
        Animal positions.
    headings : NDArray, shape (n_time,)
        Head directions (radians).
    gaze_offsets : NDArray, optional
        Gaze direction relative to head.
    view_distance : float
        Viewing distance for gaze model.
    gaze_model : {"fixed_distance", "ray_cast"}
        How to compute viewed location.
    method : {"binned", "diffusion_kde", "gaussian_kde"}
        Rate estimation method.
    bandwidth : float, optional
        Smoothing bandwidth (for KDE methods).
    min_occupancy_seconds : float
        Minimum view occupancy threshold in seconds. Bins with less occupancy
        are set to NaN. Default 0.1s matches `compute_place_field()` default
        for consistency when comparing place vs view fields.

    Returns
    -------
    NDArray, shape (n_bins,)
        Firing rate at each environment bin (as viewed location).

    Notes
    -----
    The returned field has the same shape as a place field but represents
    firing rate when *viewing* each location, not when *being* there.

    Bins that were never viewed (or viewed less than min_occupancy_seconds) are NaN.

    Examples
    --------
    >>> # Compare place field vs spatial view field
    >>> place_field = compute_place_field(env, spikes, times, positions)
    >>> view_field = compute_spatial_view_field(
    ...     env, spikes, times, positions, headings
    ... )
    >>> # High correlation suggests place cell
    >>> # Low correlation with structured view_field suggests spatial view cell
    """
    # 1. Compute viewed location at each timepoint
    viewed_locations = compute_viewed_location(
        positions, headings, gaze_offsets,
        view_distance=view_distance,
        env=env,
        method=gaze_model,
    )

    # 2. Filter out NaN (viewing outside environment)
    valid_mask = ~np.isnan(viewed_locations[:, 0])
    valid_times = times[valid_mask]
    valid_viewed = viewed_locations[valid_mask]

    # 3. Compute view occupancy (time spent viewing each bin)
    # env.occupancy() infers dt from timestamps internally
    view_occupancy = env.occupancy(valid_times, valid_viewed)

    # 4. Compute spike viewing locations
    # Interpolate viewed location at spike times
    spike_viewed = np.column_stack([
        np.interp(spike_times, times, viewed_locations[:, 0]),
        np.interp(spike_times, times, viewed_locations[:, 1]),
    ])

    # Filter spikes outside environment
    spike_valid = ~np.isnan(spike_viewed[:, 0])
    spike_viewed = spike_viewed[spike_valid]

    # 5. Bin spikes by viewed location
    # Use environment's binning machinery
    spike_bins = env.bin_at(spike_viewed)
    spike_counts = np.bincount(spike_bins, minlength=env.n_bins).astype(float)

    # 6. Normalize by view occupancy
    with np.errstate(divide='ignore', invalid='ignore'):
        view_field = spike_counts / view_occupancy

    # Mark low-occupancy bins as NaN
    view_field[view_occupancy < min_occupancy_seconds] = np.nan

    # 7. Optional smoothing
    if method in ("diffusion_kde", "gaussian_kde") and bandwidth is not None:
        view_field = env.smooth(view_field, kernel_bandwidth=bandwidth)

    return view_field
```

### M3.4: Spatial View Metrics

**File**: `src/neurospatial/metrics/spatial_view_cells.py`

```python
"""Spatial view cell metrics and classification."""

@dataclass(frozen=True)
class SpatialViewMetrics:
    """Results from spatial view cell analysis.

    Attributes
    ----------
    view_field_skaggs_info : float
        Spatial information in viewed-location space.
    place_field_skaggs_info : float
        Spatial information in physical-location space.
    view_place_correlation : float
        Correlation between view field and place field.
    view_field_sparsity : float
        Sparsity of view field.
    view_field_coherence : float
        Spatial coherence of view field.
    is_spatial_view_cell : bool
        Classification result.
    """
    view_field_skaggs_info: float
    place_field_skaggs_info: float
    view_place_correlation: float
    view_field_sparsity: float
    view_field_coherence: float
    is_spatial_view_cell: bool

    def interpretation(self) -> str:
        """Human-readable interpretation."""
        if self.is_spatial_view_cell:
            return (
                f"Spatial view cell: view_info={self.view_field_skaggs_info:.2f} "
                f"> place_info={self.place_field_skaggs_info:.2f}, "
                f"view-place corr={self.view_place_correlation:.2f}"
            )
        else:
            return (
                f"Not a spatial view cell: view_info={self.view_field_skaggs_info:.2f}, "
                f"place_info={self.place_field_skaggs_info:.2f}"
            )


def spatial_view_cell_metrics(
    env: Environment,
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
    **kwargs,
) -> SpatialViewMetrics:
    """Compute metrics for spatial view cell classification.

    Computes both place field (physical location) and view field
    (viewed location) and compares their spatial information content.

    A spatial view cell is classified when:
    - View field has higher spatial information than place field
    - View field has low sparsity (spatially selective)
    - View-place correlation is low (distinct representations)
    """
    ...


def is_spatial_view_cell(
    metrics: SpatialViewMetrics,
    info_ratio_threshold: float = 1.5,
    max_correlation: float = 0.5,
) -> bool:
    """Classify neuron as spatial view cell.

    Criteria:
    - View field info > info_ratio_threshold * place field info
    - View-place correlation < max_correlation
    """
    return (
        metrics.view_field_skaggs_info > info_ratio_threshold * metrics.place_field_skaggs_info
        and metrics.view_place_correlation < max_correlation
    )
```

### M3.5: Tests and Documentation

**Test files**:
- `tests/test_visibility.py`
- `tests/simulation/models/test_spatial_view_cells.py`
- `tests/metrics/test_spatial_view_cells.py`

**Documentation updates**:
- `.claude/QUICKSTART.md`: Add spatial view cell examples
- `.claude/ADVANCED.md`: Section on gaze-based analysis
- Example notebook: `examples/spatial_view_cells.py`

---

## Implementation Order and Dependencies

```
M1.1 Core Reference Frames ──┐
M1.2 Egocentric Grid ────────┼──► M2.1 Object-Vector Model
M1.3 Heading Utilities ──────┤     │
M1.4 Tests ──────────────────┘     ├──► M2.2 Object-Vector Metrics
                                   │     │
                                   │     ├──► M2.3 Field Computation
                                   │     │     │
                                   │     │     └──► M2.4 Overlay
                                   │     │
                                   │     └──► M2.5 Tests
                                   │
                                   └──► M3.1 Visibility ──┐
                                                          │
                                        M3.2 Spatial View Model ──┐
                                                                  │
                                        M3.3 View Field ──────────┤
                                                                  │
                                        M3.4 View Metrics ────────┤
                                                                  │
                                        M3.5 Tests ───────────────┘
```

---

## File Summary

### New Files

| File | Lines (est.) | Milestone |
|------|--------------|-----------|
| `src/neurospatial/reference_frames.py` | 400 | M1 |
| `tests/test_reference_frames.py` | 600 | M1 |
| `src/neurospatial/simulation/models/object_vector_cells.py` | 300 | M2 |
| `src/neurospatial/metrics/object_vector_cells.py` | 400 | M2 |
| `src/neurospatial/object_vector_field.py` | 200 | M2 |
| `src/neurospatial/animation/overlays/object_vector.py` | 150 | M2 |
| `tests/simulation/models/test_object_vector_cells.py` | 400 | M2 |
| `tests/metrics/test_object_vector_cells.py` | 500 | M2 |
| `tests/test_object_vector_field.py` | 300 | M2 |
| `src/neurospatial/visibility.py` | 500 | M3 |
| `src/neurospatial/simulation/models/spatial_view_cells.py` | 250 | M3 |
| `src/neurospatial/spatial_view_field.py` | 200 | M3 |
| `src/neurospatial/metrics/spatial_view_cells.py` | 300 | M3 |
| `tests/test_visibility.py` | 400 | M3 |
| `tests/simulation/models/test_spatial_view_cells.py` | 350 | M3 |
| `tests/metrics/test_spatial_view_cells.py` | 400 | M3 |
| `tests/test_spatial_view_field.py` | 300 | M3 |

**Total new code**: ~5,950 lines

### Modified Files

| File | Changes | Milestone |
|------|---------|-----------|
| `src/neurospatial/__init__.py` | Export new functions | M1, M2, M3 |
| `src/neurospatial/simulation/__init__.py` | Export new models | M2, M3 |
| `src/neurospatial/simulation/models/__init__.py` | Export new models | M2, M3 |
| `src/neurospatial/metrics/__init__.py` | Export new metrics | M2, M3 |
| `src/neurospatial/animation/overlays/__init__.py` | Export overlay | M2 |
| `src/neurospatial/environment/factories.py` | Add `from_egocentric_grid()` | M1 |
| `.claude/QUICKSTART.md` | Add examples | M1, M2, M3 |
| `.claude/API_REFERENCE.md` | Add imports | M1, M2, M3 |
| `.claude/ADVANCED.md` | Add gaze analysis section | M3 |

---

## Testing Strategy

### Unit Tests

Each module follows the established pattern from `test_head_direction.py`:

1. **Module setup**: Imports, docstrings, `__all__` verification
2. **Input validation**: Error messages, edge cases
3. **Core computation**: Expected outputs, numerical accuracy
4. **Dataclass tests**: Field access, methods, string representation
5. **Visualization**: Axes creation, styling

### Integration Tests

1. **Round-trip transforms**: `allocentric → egocentric → allocentric`
2. **Simulation → Analysis**: Generate with model, recover parameters
3. **NWB I/O**: Save/load spatial view data

### Property-Based Tests (Hypothesis)

1. **Transform properties**: Rotation preserves distances
2. **Angle wrapping**: Always in [-π, π]
3. **Occupancy normalization**: Sum to total time

---

## Dependencies

**No new dependencies required.** All features implemented with:
- NumPy (array operations, linear algebra)
- SciPy (von Mises distribution, interpolation)
- Shapely (already dependency; for ray casting)

**Optional enhancements** (not required):
- numba: JIT compilation for ray casting (performance)
- shapely.strtree: Spatial indexing for complex environments

---

## Validation Strategy

### Ground Truth Validation

Each simulation model includes `ground_truth` dict for validation:

```python
# Generate simulated data
model = ObjectVectorCellModel(env, object_positions, preferred_distance=10.0, ...)
spikes = generate_poisson_spikes(model.firing_rate(positions, headings=headings), times)

# Recover parameters
metrics = compute_object_vector_tuning(spikes, times, positions, headings, object_positions)

# Compare
assert abs(metrics.preferred_distance - model.ground_truth["preferred_distance"]) < 2.0
```

### Literature Validation

Reference implementations from:
- Hoydal et al. (2019) - Object-vector cells
- Rolls et al. (1997) - Spatial view cells
- Deshmukh & Knierim (2011) - Object-related firing

---

## Documentation Updates

### QUICKSTART.md Additions

```markdown
### Egocentric Reference Frames

Transform positions to animal-centered coordinates:

```python
from neurospatial.reference_frames import allocentric_to_egocentric

# Transform landmarks to egocentric frame
ego_landmarks = allocentric_to_egocentric(
    landmark_positions,  # (n_landmarks, 2)
    animal_positions,    # (n_time, 2)
    animal_headings,     # (n_time,)
)
# Result: (n_time, n_landmarks, 2) - landmarks relative to animal
```

### Object-Vector Cell Analysis

Analyze cells encoding vectors to objects:

```python
from neurospatial.metrics import compute_object_vector_tuning

metrics = compute_object_vector_tuning(
    spike_times, times, positions, headings,
    object_positions=landmark_positions,
)
print(metrics.interpretation())
# "Object-vector cell: fires 10.0 cm ahead of object (direction=15°). Score=0.72"
```
```

### API_REFERENCE.md Additions

```markdown
## Reference Frames

```python
from neurospatial.reference_frames import (
    EgocentricFrame,
    allocentric_to_egocentric,
    egocentric_to_allocentric,
    compute_egocentric_bearing,
    heading_from_velocity,
)
```

## Object-Vector Cells

```python
from neurospatial.simulation import ObjectVectorCellModel
from neurospatial.metrics import (
    compute_object_vector_tuning,
    object_vector_score,
    is_object_vector_cell,
    ObjectVectorMetrics,
)
from neurospatial import compute_object_vector_field
```

## Spatial View Cells

```python
from neurospatial.simulation import SpatialViewCellModel
from neurospatial.visibility import compute_viewed_location, compute_view_field
from neurospatial import compute_spatial_view_field
from neurospatial.metrics import spatial_view_cell_metrics, SpatialViewMetrics
```

## Viewshed Analysis

```python
from neurospatial.visibility import (
    FieldOfView,  # Species-specific FOV presets
    ViewshedResult,
    compute_viewshed,
    compute_viewshed_trajectory,
    visible_boundaries,
    visible_cues,
    visibility_occupancy,
)
```
```

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Ray casting performance | Medium | Low | Use Shapely spatial index; cache results |
| Heading data quality | Medium | Medium | Provide `heading_from_velocity()` fallback |
| Circular interpolation bugs | Medium | High | Extensive tests at 0/2π boundary |
| 3D support complexity | Low | Low | Initially 2D only; document limitation |
| NWB extension compatibility | Low | Medium | Use existing containers where possible |

---

## Success Criteria

### Milestone 1: Egocentric Frames
- [ ] `allocentric_to_egocentric()` passes round-trip tests
- [ ] `heading_from_velocity()` handles stationary periods
- [ ] All circular boundary cases pass
- [ ] Documentation complete

### Milestone 2: Object-Vector Cells
- [ ] `ObjectVectorCellModel` reproduces Hoydal et al. tuning curves
- [ ] `object_vector_score()` classifies simulated cells correctly
- [ ] Animation overlay renders correctly in napari
- [ ] Example notebook demonstrates full workflow

### Milestone 3: Spatial View Cells
- [ ] `compute_viewed_location()` handles ray casting
- [ ] `compute_spatial_view_field()` produces interpretable maps
- [ ] `is_spatial_view_cell()` distinguishes view vs place cells
- [ ] `compute_viewshed()` correctly computes visible bins/boundaries
- [ ] `visible_cues()` handles line-of-sight occlusion
- [ ] `visibility_occupancy()` integrates correctly over trajectory
- [ ] `FieldOfView` presets match literature values (rat ~300°, primate ~180°)
- [ ] Asymmetric FOV (e.g., rat blind spot) correctly excludes behind regions
- [ ] Documentation includes species-specific guidance (rodent vs primate)

---

## Appendix: API Quick Reference

### Required Imports

```python
import numpy as np
from neurospatial import Environment, compute_place_field

# Egocentric transforms
from neurospatial.reference_frames import (
    EgocentricFrame,
    allocentric_to_egocentric,
    egocentric_to_allocentric,
    compute_egocentric_bearing,
    compute_egocentric_distance,
    heading_from_velocity,
    heading_from_body_orientation,
)

# Object-vector cells
from neurospatial.simulation import ObjectVectorCellModel
from neurospatial.metrics import (
    compute_object_vector_tuning,
    object_vector_score,
    is_object_vector_cell,
    ObjectVectorMetrics,
)
from neurospatial import compute_object_vector_field

# Spatial view cells
from neurospatial.simulation import SpatialViewCellModel
from neurospatial.visibility import (
    compute_viewed_location,
    compute_viewshed,
    visible_boundaries,
    visible_cues,
    visibility_occupancy,
    FieldOfView,
    ViewshedResult,
)
from neurospatial import compute_spatial_view_field
from neurospatial.metrics import spatial_view_cell_metrics, SpatialViewMetrics
```

### Egocentric Transforms

```python
# Transform coordinates
ego_pts = allocentric_to_egocentric(points, positions, headings)
allo_pts = egocentric_to_allocentric(ego_pts, positions, headings)

# Compute bearing (angle to target relative to heading)
bearing = compute_egocentric_bearing(target, positions, headings)
# 0 = ahead, π/2 = left, -π/2 = right, ±π = behind

# Compute distance to targets (Euclidean or geodesic)
distances = compute_egocentric_distance(target, positions)  # Euclidean
distances = compute_egocentric_distance(target, positions, env, "geodesic")  # Geodesic

# Infer heading from movement (handles stationary periods)
headings = heading_from_velocity(positions, times, min_speed=2.0)

# Or from body keypoints (handles NaN keypoints)
headings = heading_from_body_orientation(nose_positions, tail_positions)
```

### Object-Vector Analysis

```python
# Simulation
cell = ObjectVectorCellModel(
    env, objects,
    preferred_distance=10,  # cm
    preferred_direction=0,  # ahead
    direction_kappa=4.0,    # ~30° half-width
)
rates = cell.firing_rate(positions, headings=headings)

# Analysis (returns ObjectVectorMetrics)
metrics = compute_object_vector_tuning(spikes, times, positions, headings, objects)
is_ovc = is_object_vector_cell(metrics)

# Rate map in egocentric space (uses nearest object at each timepoint)
result = compute_object_vector_field(spikes, times, positions, headings, objects)
# result.field, result.ego_env, result.occupancy
```

### Spatial View Analysis

```python
# Compute what's being viewed
viewed = compute_viewed_location(positions, headings, view_distance=30)

# Simulation
cell = SpatialViewCellModel(env, preferred_view_location=[50, 50])
rates = cell.firing_rate(positions, headings=headings)

# View field (rate map in viewed-location space)
view_field = compute_spatial_view_field(env, spikes, times, positions, headings)

# Compare to place field (returns SpatialViewMetrics)
metrics = spatial_view_cell_metrics(env, spikes, times, positions, headings)
is_svc = metrics.is_spatial_view_cell
```

### Viewshed Analysis

```python
from neurospatial.visibility import (
    compute_viewshed,
    visible_boundaries,
    visible_cues,
    visibility_occupancy,
    FieldOfView,
    ViewshedResult,
)

# Compute full viewshed from a position (360°)
result = compute_viewshed(env, position=[50, 50])
print(f"{result.visibility_fraction:.1%} visible")
print(f"{result.n_visible_bins} bins visible")

# Species-specific FOV: Rat (~300° panoramic vision)
result = compute_viewshed(
    env, position=[50, 50], heading=0.0, fov=FieldOfView.rat()
)

# Species-specific FOV: Primate (~180° forward vision)
result = compute_viewshed(
    env, position=[50, 50], heading=0.0, fov=FieldOfView.primate()
)

# Simple symmetric FOV (60° half-angle = 120° total)
result = compute_viewshed(
    env, position=[50, 50], heading=0.0, fov=np.pi/3
)

# Check which cues/landmarks are visible (using rat FOV)
landmarks = np.array([[30, 50], [70, 50], [50, 30]])
rat_fov = FieldOfView.rat()
visible, distances, bearings = visible_cues(
    env, position=[50, 50], heading=0.0,
    cue_positions=landmarks, fov=rat_fov
)

# Get visible boundary segments (useful for boundary cell analysis)
boundary_segs = visible_boundaries(env, position=[50, 50], heading=0.0)

# Compute visibility occupancy along trajectory (with species FOV)
vis_occ = visibility_occupancy(env, positions, headings, times, fov=FieldOfView.rat())
```

---

## Appendix B: Review Feedback Integration

This section documents how feedback from the initial plan review was integrated.

### Feedback Summary

The reviewer assessed the plan as "very strong" with ~90% of content kept, recommending adjustments for:

1. **Consistency with existing modules** (`circular`, `head_direction`, `place_fields`)
2. **Vectorization / performance** (especially ray casting and bearings)
3. **Scientific / statistical choices** (von Mises width, metrics definitions)
4. **API tightening** to avoid surface bloat

### Changes Made

#### M1: Egocentric Reference Frames

| Feedback | Resolution |
|----------|------------|
| Avoid re-inventing angle wrapping | `compute_egocentric_bearing()` uses inline `(θ + π) % 2π - π` wrapping |
| Batch transforms: normalize shapes early | `allocentric_to_egocentric()` immediately normalizes to `(n_time, n_points, 2)` |
| Use einsum for rotation | Replaced loop with `np.einsum('tij,tpj->tpi', rot, centered)` |
| Split geodesic distance out | Created `compute_egocentric_distance_euclidean()` and `compute_egocentric_distance_geodesic()` separately |
| Circular interpolation for headings | `heading_from_velocity()` and `heading_from_body_orientation()` now use unit-vector interpolation to avoid ±π discontinuities |
| Document egocentric grid is parameter space | Added explicit note: "This environment lives in egocentric polar coordinates, not physical allocentric space" |

#### M2: Object-Vector Cells

| Feedback | Resolution |
|----------|------------|
| Vectorize bearing computation | Removed loop; now uses `allocentric_to_egocentric()` directly on `(n_time, n_objects, 2)` |
| von Mises "width" naming | Renamed `direction_width` → `direction_kappa` with explicit kappa semantics and typical values documented |
| Handle object outside environment | Added validation in `__post_init__` with warning for objects at invalid bins |
| Explicit score formulas | Added mathematical formulas for `s_d`, `s_θ`, and `s_OV` in `object_vector_score()` docstring |
| Reuse circular utilities | `object_vector_score()` now imports `_mean_resultant_length` from `circular.py` |
| Return egocentric environment | `compute_object_vector_field()` returns `ObjectVectorFieldResult` containing `ego_env` |

#### M3: Spatial View Cells

| Feedback | Resolution |
|----------|------------|
| Ray casting: use Shapely intersection | Implementation note added for `_ray_cast_to_boundary()` |
| Reuse smoothing code path | `compute_spatial_view_field()` notes: "Smoothing uses the same diffusion kernel machinery as compute_place_field()" |
| Use existing Skaggs/sparsity/coherence | `spatial_view_cell_metrics()` implementation note: reuse `place_fields.py` functions |
| View-place correlation: mask NaNs | Added note about Z-scoring and NaN masking |

### API Surface Control

Per reviewer suggestion, the public API remains minimal:

**Top-level exports** (from `neurospatial`):

- `compute_object_vector_field`, `compute_spatial_view_field`

**Subpackage exports**:

- `neurospatial.reference_frames`: `EgocentricFrame`, `allocentric_to_egocentric`, etc.
- `neurospatial.simulation`: `ObjectVectorCellModel`, `SpatialViewCellModel`
- `neurospatial.metrics`: `ObjectVectorMetrics`, `SpatialViewMetrics`, classification functions

Internal helpers (e.g., `_ray_cast_to_boundary`) remain private.

### Testing Approach

Per reviewer guidance on test scope:

- Focus on **transform correctness** (round-trip, conventions)
- Focus on **model → analysis recovery** (can we recover ground truth?)
- Focus on **edge cases** (NaN, zero occupancy, outside-env)
- Use **shared fixtures** to reduce duplication

### "Brandon Rhodes" Alignment

- Strong types: `EgocentricFrame`, `ObjectVectorMetrics`, etc.
- Literate docstrings with examples
- Explicit conventions documented (angle definitions, coordinate systems)

### "Raymond Hettinger" Alignment

- Dataclasses with validated parameters
- Precomputed geodesic distance fields
- Heavy NumPy vectorization (einsum, broadcasting)
- Single implementations of smoothing/occupancy/circular stats
