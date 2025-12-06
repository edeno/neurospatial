"""Visibility and gaze computation for spatial view analysis.

Provides tools for computing what an animal can see from any position in
an environment, including field-of-view constraints, line-of-sight checks,
and viewshed analysis.

Common Use Cases
----------------
- Spatial view cell analysis (what location is being viewed)
- Visibility-based neural encoding models
- Gaze-contingent analysis
- Landmark visibility during navigation

Which Function Should I Use?
----------------------------
**Computing viewed location from gaze direction?**
    Use ``compute_viewed_location()`` with method="fixed_distance" for a
    point at fixed distance, or method="ray_cast" for boundary intersection.

**Need full visibility analysis from a position?**
    Use ``compute_viewshed()`` to get all visible bins, boundaries, and cues.

**Just need a binary visibility mask?**
    Use ``compute_view_field()`` for a simple bool array of visible bins.

**Check if specific cues are visible?**
    Use ``visible_cues()`` for line-of-sight checks to cue positions.

**Analyze visibility along a trajectory?**
    Use ``compute_viewshed_trajectory()`` or ``visibility_occupancy()``.

Species-Specific Field of View
------------------------------
Use ``FieldOfView`` presets for biologically realistic constraints:

- ``FieldOfView.rat()`` - ~300 degree FOV with blind spot behind
- ``FieldOfView.mouse()`` - Similar to rat
- ``FieldOfView.primate()`` - ~180 degree forward-facing FOV

Coordinate Conventions
----------------------
**Egocentric direction** (for field of view):
- 0 radians = directly ahead
- pi/2 radians = to the left
- -pi/2 radians = to the right
- +/-pi radians = behind

This matches the convention in ``neurospatial.reference_frames``.

References
----------
.. [1] Rolls, E. T., et al. (1997). Spatial view cells in the primate
       hippocampus. European Journal of Neuroscience, 9(8), 1789-1794.
.. [2] Wallace, D. J., et al. (2013). Rats maintain an overhead binocular
       field at the expense of constant visual coverage of the environment.
       Vision Research, 78, 1-10.

See Also
--------
neurospatial.reference_frames : Egocentric coordinate transforms
neurospatial.spatial_view_field : Spatial view field computation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from neurospatial import Environment

__all__ = [
    "FieldOfView",
    "ViewshedResult",
    "compute_view_field",
    "compute_viewed_location",
    "compute_viewshed",
    "compute_viewshed_trajectory",
    "visibility_occupancy",
    "visible_cues",
]


@dataclass(frozen=True)
class FieldOfView:
    """Field of view specification for visibility analysis.

    Defines the angular extent of what an animal can see, with support for
    asymmetric fields, binocular regions, and blind spots.

    Attributes
    ----------
    left_angle : float
        Maximum angle to the left in radians. Positive value (e.g., pi/2 = 90
        degrees left).
    right_angle : float
        Maximum angle to the right in radians. Negative value (e.g., -pi/2 =
        90 degrees right).
    binocular_half_angle : float, optional
        Half-angle of binocular overlap region centered on 0. Default is 0.0
        (no binocular region).
    blind_spot_behind : float, optional
        Half-angle of blind spot centered on +-pi (behind). Default is 0.0
        (no blind spot).

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.visibility import FieldOfView

    Create a 180-degree symmetric field of view:

    >>> fov = FieldOfView.symmetric(half_angle=np.pi / 2)
    >>> fov.total_angle_degrees
    180.0

    Use species-specific presets:

    >>> fov_rat = FieldOfView.rat()
    >>> 290 < fov_rat.total_angle_degrees < 340
    True

    >>> fov_primate = FieldOfView.primate()
    >>> 150 < fov_primate.total_angle_degrees < 200
    True

    Check if an angle is within the field of view:

    >>> fov = FieldOfView.symmetric(half_angle=np.pi / 2)
    >>> fov.contains_angle(0.0)  # Ahead
    True
    >>> fov.contains_angle(np.pi)  # Behind
    False
    """

    left_angle: float
    right_angle: float
    binocular_half_angle: float = 0.0
    blind_spot_behind: float = 0.0

    def __post_init__(self) -> None:
        """Validate field of view parameters."""
        if self.left_angle < self.right_angle:
            raise ValueError(
                f"left_angle ({self.left_angle:.3f}) must be >= right_angle "
                f"({self.right_angle:.3f}). "
                "Note: left_angle should be positive, right_angle negative."
            )
        if self.binocular_half_angle < 0:
            raise ValueError(
                f"binocular_half_angle must be non-negative, "
                f"got {self.binocular_half_angle}"
            )
        if self.blind_spot_behind < 0:
            raise ValueError(
                f"blind_spot_behind must be non-negative, got {self.blind_spot_behind}"
            )

    @classmethod
    def symmetric(cls, half_angle: float) -> FieldOfView:
        """Create a symmetric field of view.

        Parameters
        ----------
        half_angle : float
            Half-angle of the field of view in radians.
            The total FOV will be 2 * half_angle.

        Returns
        -------
        FieldOfView
            Symmetric field of view.

        Examples
        --------
        >>> from neurospatial.visibility import FieldOfView
        >>> import numpy as np
        >>> fov = FieldOfView.symmetric(half_angle=np.pi / 2)
        >>> fov.total_angle_degrees
        180.0
        """
        return cls(left_angle=half_angle, right_angle=-half_angle)

    @classmethod
    def rat(cls) -> FieldOfView:
        """Field of view for a rat.

        Rats have a ~300-320 degree field of view with a ~40-60 degree
        blind spot directly behind, and a ~40-60 degree binocular region
        in front.

        Returns
        -------
        FieldOfView
            Rat-like field of view.

        References
        ----------
        Wallace, D. J., et al. (2013). Vision Research, 78, 1-10.
        """
        # ~320 degree total FOV (160 each side from center)
        # ~20 degree blind spot each side behind
        # ~25 degree binocular half-angle
        return cls(
            left_angle=np.pi * 160 / 180,  # 160 degrees
            right_angle=-np.pi * 160 / 180,
            binocular_half_angle=np.pi * 25 / 180,  # 25 degrees
            blind_spot_behind=np.pi * 20 / 180,  # 20 degrees each side
        )

    @classmethod
    def mouse(cls) -> FieldOfView:
        """Field of view for a mouse.

        Mice have a similar field of view to rats, with a large lateral
        field and small blind spot behind.

        Returns
        -------
        FieldOfView
            Mouse-like field of view.
        """
        # Similar to rat
        return cls(
            left_angle=np.pi * 155 / 180,  # 155 degrees
            right_angle=-np.pi * 155 / 180,
            binocular_half_angle=np.pi * 20 / 180,  # 20 degrees
            blind_spot_behind=np.pi * 25 / 180,  # 25 degrees each side
        )

    @classmethod
    def primate(cls) -> FieldOfView:
        """Field of view for a primate.

        Primates have forward-facing eyes with ~180 degree total field of
        view and large binocular overlap.

        Returns
        -------
        FieldOfView
            Primate-like field of view.
        """
        return cls(
            left_angle=np.pi * 90 / 180,  # 90 degrees
            right_angle=-np.pi * 90 / 180,
            binocular_half_angle=np.pi * 60 / 180,  # 60 degrees
            blind_spot_behind=0.0,  # No additional blind spot
        )

    @property
    def total_angle(self) -> float:
        """Total field of view angle in radians.

        Returns
        -------
        float
            Total angle from left_angle to right_angle.
        """
        return self.left_angle - self.right_angle

    @property
    def total_angle_degrees(self) -> float:
        """Total field of view angle in degrees.

        Returns
        -------
        float
            Total angle in degrees.
        """
        return float(np.degrees(self.total_angle))

    def contains_angle(
        self, angle: float | NDArray[np.float64]
    ) -> bool | NDArray[np.bool_]:
        """Check if an angle is within the field of view.

        Parameters
        ----------
        angle : float or NDArray
            Egocentric angle(s) in radians. 0=ahead, pi/2=left, -pi/2=right.

        Returns
        -------
        bool or NDArray[bool]
            True if angle is within field of view.

        Examples
        --------
        >>> from neurospatial.visibility import FieldOfView
        >>> import numpy as np
        >>> fov = FieldOfView.symmetric(half_angle=np.pi / 2)
        >>> fov.contains_angle(0.0)
        True
        >>> fov.contains_angle(np.pi)
        False
        """
        angle = np.asarray(angle)

        # Wrap angle to (-pi, pi]
        wrapped = (angle + np.pi) % (2 * np.pi) - np.pi

        # Check if within left/right bounds
        in_bounds = (wrapped <= self.left_angle) & (wrapped >= self.right_angle)

        # Check if in blind spot behind
        if self.blind_spot_behind > 0:
            # Blind spot is centered on +-pi
            # An angle is in blind spot if |angle| > pi - blind_spot_behind
            in_blind_spot = np.abs(wrapped) > (np.pi - self.blind_spot_behind)
            in_bounds = in_bounds & ~in_blind_spot

        # Return scalar if input was scalar
        if angle.ndim == 0:
            return bool(in_bounds)
        return in_bounds

    def is_binocular(
        self, angle: float | NDArray[np.float64]
    ) -> bool | NDArray[np.bool_]:
        """Check if an angle is in the binocular region.

        Parameters
        ----------
        angle : float or NDArray
            Egocentric angle(s) in radians.

        Returns
        -------
        bool or NDArray[bool]
            True if angle is in the binocular region.

        Examples
        --------
        >>> from neurospatial.visibility import FieldOfView
        >>> import numpy as np
        >>> fov = FieldOfView(
        ...     left_angle=np.pi / 2,
        ...     right_angle=-np.pi / 2,
        ...     binocular_half_angle=np.pi / 6,
        ... )
        >>> fov.is_binocular(0.0)  # Center is binocular
        True
        >>> fov.is_binocular(np.pi / 3)  # 60 degrees is not
        False
        """
        if self.binocular_half_angle == 0:
            return np.zeros_like(angle, dtype=bool) if np.ndim(angle) > 0 else False

        angle = np.asarray(angle)
        wrapped = (angle + np.pi) % (2 * np.pi) - np.pi

        in_binocular = np.abs(wrapped) <= self.binocular_half_angle

        if angle.ndim == 0:
            return bool(in_binocular)
        return in_binocular


@dataclass(frozen=True)
class ViewshedResult:
    """Result of viewshed analysis from a single observer position.

    Attributes
    ----------
    visible_bins : NDArray[np.intp]
        Indices of environment bins visible from the observer position.
    visible_cues : NDArray[np.intp]
        Indices of visible cues (if cue_positions were provided).
    cue_distances : NDArray[np.float64]
        Distances to visible cues.
    cue_bearings : NDArray[np.float64]
        Egocentric bearings to visible cues.
    occlusion_map : NDArray[np.float64]
        Per-bin occlusion score [0, 1] where 1 means fully visible.
    _total_bins : int, optional
        Total number of bins in the environment (for computing fractions).

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.visibility import ViewshedResult
    >>> result = ViewshedResult(
    ...     visible_bins=np.array([0, 1, 2]),
    ...     visible_cues=np.array([]),
    ...     cue_distances=np.array([]),
    ...     cue_bearings=np.array([]),
    ...     occlusion_map=np.zeros(10),
    ... )
    >>> result.n_visible_bins
    3
    """

    visible_bins: NDArray[np.intp]
    visible_cues: NDArray[np.intp]
    cue_distances: NDArray[np.float64]
    cue_bearings: NDArray[np.float64]
    occlusion_map: NDArray[np.float64]
    _total_bins: int = field(default=0)

    @property
    def n_visible_bins(self) -> int:
        """Number of visible bins."""
        return len(self.visible_bins)

    @property
    def visibility_fraction(self) -> float:
        """Fraction of total bins that are visible.

        Returns
        -------
        float
            Fraction in [0, 1]. Returns 0 if _total_bins not set.
        """
        if self._total_bins == 0:
            return 0.0
        return self.n_visible_bins / self._total_bins

    @property
    def n_visible_cues(self) -> int:
        """Number of visible cues."""
        return len(self.visible_cues)

    def filter_cues(
        self, cue_ids: list[int] | NDArray[np.intp]
    ) -> tuple[NDArray[np.intp], NDArray[np.float64], NDArray[np.float64]]:
        """Filter to specific cue IDs.

        Parameters
        ----------
        cue_ids : list or NDArray
            Cue IDs to filter to.

        Returns
        -------
        visible_ids : NDArray[np.intp]
            Cue IDs that are both in cue_ids and visible.
        distances : NDArray[np.float64]
            Distances to the filtered visible cues.
        bearings : NDArray[np.float64]
            Bearings to the filtered visible cues.
        """
        cue_ids = np.asarray(cue_ids)
        mask = np.isin(self.visible_cues, cue_ids)
        return (
            self.visible_cues[mask],
            self.cue_distances[mask],
            self.cue_bearings[mask],
        )

    def visible_bin_centers(self, env: Environment) -> NDArray[np.float64]:
        """Get allocentric positions of visible bin centers.

        Parameters
        ----------
        env : Environment
            The environment used for the viewshed computation.

        Returns
        -------
        NDArray[np.float64], shape (n_visible_bins, n_dims)
            Allocentric positions of visible bin centers.
        """
        return env.bin_centers[self.visible_bins]


def compute_viewed_location(
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
    *,
    method: Literal["fixed_distance", "ray_cast", "boundary"] = "fixed_distance",
    view_distance: float = 10.0,
    gaze_offsets: NDArray[np.float64] | None = None,
    env: Environment | None = None,
    max_distance: float = 100.0,
) -> NDArray[np.float64]:
    """Compute the location being viewed by the animal.

    Parameters
    ----------
    positions : NDArray[np.float64], shape (n_time, 2)
        Animal positions in allocentric coordinates.
    headings : NDArray[np.float64], shape (n_time,)
        Animal heading at each time (radians, 0=East).
    method : {"fixed_distance", "ray_cast", "boundary"}, default "fixed_distance"
        Method for computing viewed location:
        - "fixed_distance": Point at fixed distance in gaze direction
        - "ray_cast": Intersection with environment boundary
        - "boundary": Nearest boundary point in gaze direction
    view_distance : float, default 10.0
        Distance for fixed_distance method.
    gaze_offsets : NDArray[np.float64], shape (n_time,), optional
        Offset from heading to gaze direction (e.g., eye tracking).
        If None, gaze is aligned with heading.
    env : Environment, optional
        Required for ray_cast and boundary methods.
    max_distance : float, default 100.0
        Maximum ray distance for ray_cast method.

    Returns
    -------
    NDArray[np.float64], shape (n_time, 2)
        Viewed location in allocentric coordinates. NaN if ray doesn't
        intersect environment.

    Raises
    ------
    ValueError
        If method is invalid or ray_cast/boundary without environment.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.visibility import compute_viewed_location

    Fixed distance method:

    >>> positions = np.array([[0.0, 0.0]])
    >>> headings = np.array([0.0])  # Facing East
    >>> viewed = compute_viewed_location(
    ...     positions, headings, method="fixed_distance", view_distance=10.0
    ... )
    >>> np.allclose(viewed[0], [10.0, 0.0])
    True
    """
    positions = np.asarray(positions, dtype=np.float64)
    headings = np.asarray(headings, dtype=np.float64)

    if method not in ("fixed_distance", "ray_cast", "boundary"):
        raise ValueError(
            f"method must be 'fixed_distance', 'ray_cast', or 'boundary', "
            f"got '{method}'"
        )

    if method in ("ray_cast", "boundary") and env is None:
        raise ValueError(f"env is required when method='{method}'")

    n_time = len(positions)

    # Compute gaze direction
    gaze_direction = headings + gaze_offsets if gaze_offsets is not None else headings

    viewed = np.full((n_time, 2), np.nan, dtype=np.float64)

    if method == "fixed_distance":
        # Simple: point at fixed distance in gaze direction
        viewed[:, 0] = positions[:, 0] + view_distance * np.cos(gaze_direction)
        viewed[:, 1] = positions[:, 1] + view_distance * np.sin(gaze_direction)

    else:  # ray_cast or boundary (both use ray casting)
        assert env is not None
        # Cast ray until it exits environment or hits max distance
        for i in range(n_time):
            intersection = _ray_cast_to_boundary(
                env,
                positions[i],
                gaze_direction[i],
                max_distance=max_distance,
            )
            if intersection is not None:
                viewed[i] = intersection

    return viewed


def _ray_cast_to_boundary(
    env: Environment,
    origin: NDArray[np.float64],
    direction: float,
    max_distance: float = 100.0,
    n_steps: int = 100,
) -> NDArray[np.float64] | None:
    """Cast a ray from origin in direction until it exits the environment.

    Uses iterative stepping with binary search refinement.

    Parameters
    ----------
    env : Environment
        The environment to ray cast within.
    origin : NDArray[np.float64], shape (2,)
        Starting position.
    direction : float
        Direction in radians (0=East).
    max_distance : float
        Maximum distance to search.
    n_steps : int
        Number of steps for initial search.

    Returns
    -------
    NDArray[np.float64] or None
        Intersection point, or None if no intersection found.
    """
    # Create ray points
    distances = np.linspace(0, max_distance, n_steps)
    dx = np.cos(direction)
    dy = np.sin(direction)

    ray_x = origin[0] + distances * dx
    ray_y = origin[1] + distances * dy
    ray_points = np.column_stack([ray_x, ray_y])

    # Find which points are inside environment
    bin_indices = env.bin_at(ray_points)
    inside = bin_indices >= 0

    # Find first transition from inside to outside
    if not np.any(inside):
        # Start point outside environment
        return None

    # Find transition points
    transitions = np.diff(inside.astype(int))
    exit_indices = np.where(transitions == -1)[0]

    if len(exit_indices) == 0:
        # Ray never exits (all points inside or outside)
        if inside[-1]:
            # Ray ends inside - return max distance point
            result: NDArray[np.float64] = ray_points[-1]
            return result
        return None

    # Refine first exit point with binary search
    exit_idx = exit_indices[0]
    low_dist = distances[exit_idx]
    high_dist = distances[exit_idx + 1]

    # Binary search for boundary
    for _ in range(10):  # 10 iterations gives ~0.1% precision
        mid_dist = (low_dist + high_dist) / 2
        mid_point = np.array([origin[0] + mid_dist * dx, origin[1] + mid_dist * dy])
        mid_bin = env.bin_at(mid_point.reshape(1, -1))[0]

        if mid_bin >= 0:
            low_dist = mid_dist
        else:
            high_dist = mid_dist

    # Return last point inside
    boundary_point = np.array([origin[0] + low_dist * dx, origin[1] + low_dist * dy])
    return boundary_point


def _line_of_sight_clear(
    env: Environment,
    observer: NDArray[np.float64],
    target: NDArray[np.float64],
    n_samples: int = 20,
) -> bool:
    """Check if line of sight between observer and target is clear.

    Parameters
    ----------
    env : Environment
        The environment.
    observer : NDArray[np.float64], shape (2,)
        Observer position.
    target : NDArray[np.float64], shape (2,)
        Target position.
    n_samples : int
        Number of points to sample along line.

    Returns
    -------
    bool
        True if line of sight is clear (all points inside environment).
    """
    # Sample points along line
    t = np.linspace(0, 1, n_samples)
    line_x = observer[0] + t * (target[0] - observer[0])
    line_y = observer[1] + t * (target[1] - observer[1])
    line_points = np.column_stack([line_x, line_y])

    # Check if all points are inside environment
    bin_indices = env.bin_at(line_points)
    return bool(np.all(bin_indices >= 0))


def compute_viewshed(
    env: Environment,
    position: NDArray[np.float64],
    heading: float,
    *,
    fov: FieldOfView | float | None = None,
    n_rays: int = 360,
    cue_positions: NDArray[np.float64] | None = None,
) -> ViewshedResult:
    """Compute visible bins from an observer position.

    Casts rays from the observer position to determine which bins are
    visible, optionally restricted by field of view.

    Parameters
    ----------
    env : Environment
        The environment to analyze.
    position : NDArray[np.float64], shape (2,)
        Observer position in allocentric coordinates.
    heading : float
        Observer heading in radians (0=East).
    fov : FieldOfView, float, or None, default None
        Field of view constraint. If FieldOfView, uses its bounds.
        If float, interpreted as full angle in radians.
        If None, uses full 360 degrees.
    n_rays : int, default 360
        Number of rays to cast (higher = more accurate).
    cue_positions : NDArray[np.float64], shape (n_cues, 2), optional
        Positions of cues to check visibility for.

    Returns
    -------
    ViewshedResult
        Viewshed analysis result with visible bins, boundaries, and cues.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.visibility import compute_viewshed, FieldOfView
    >>> rng = np.random.default_rng(42)
    >>> positions = rng.uniform(0, 100, (1000, 2))
    >>> env = Environment.from_samples(positions, bin_size=5.0)
    >>> result = compute_viewshed(env, np.array([50.0, 50.0]), heading=0.0)
    >>> result.n_visible_bins > 0
    True
    """
    from neurospatial.ops.egocentric import compute_egocentric_bearing

    position = np.asarray(position, dtype=np.float64)

    # Determine angular range
    if fov is None:
        min_angle = -np.pi
        max_angle = np.pi
        fov_obj = None
    elif isinstance(fov, (int, float)):
        half_angle = float(fov) / 2
        min_angle = -half_angle
        max_angle = half_angle
        fov_obj = FieldOfView.symmetric(half_angle)
    else:
        min_angle = fov.right_angle
        max_angle = fov.left_angle
        fov_obj = fov

    # Generate ray angles (in egocentric coordinates)
    ray_angles_ego = np.linspace(min_angle, max_angle, n_rays)

    # Convert to allocentric
    ray_angles_allo = ray_angles_ego + heading

    # Cast rays and track visible bins
    visible_bin_set = set()
    occlusion_scores = np.zeros(env.n_bins, dtype=np.float64)
    ray_counts = np.zeros(env.n_bins, dtype=np.float64)

    for angle in ray_angles_allo:
        # Cast ray
        boundary_point = _ray_cast_to_boundary(
            env, position, angle, max_distance=200.0, n_steps=100
        )

        if boundary_point is None:
            continue

        # Mark bins along ray as visible
        distance = np.linalg.norm(boundary_point - position)
        # Use minimum bin size for sampling resolution
        min_bin_size = float(np.min(env.bin_sizes))
        n_samples = max(int(distance / min_bin_size) * 2, 10)

        t = np.linspace(0, 1, n_samples)
        ray_x = position[0] + t * (boundary_point[0] - position[0])
        ray_y = position[1] + t * (boundary_point[1] - position[1])
        ray_points = np.column_stack([ray_x, ray_y])

        bin_indices = env.bin_at(ray_points)
        valid_bins = bin_indices[bin_indices >= 0]

        for bin_idx in valid_bins:
            visible_bin_set.add(int(bin_idx))
            occlusion_scores[bin_idx] += 1.0
            ray_counts[bin_idx] += 1.0

    visible_bins = np.array(sorted(visible_bin_set), dtype=np.intp)

    # Normalize occlusion scores
    max_count = np.max(ray_counts) if np.max(ray_counts) > 0 else 1.0
    occlusion_map = np.where(ray_counts > 0, occlusion_scores / max_count, 0.0)

    # Check cue visibility
    if cue_positions is not None:
        cue_positions = np.asarray(cue_positions, dtype=np.float64)
        n_cues = len(cue_positions)

        visible_cue_list = []
        cue_distances_list = []
        cue_bearings_list = []

        # Compute bearings
        bearings = compute_egocentric_bearing(
            cue_positions, position.reshape(1, -1), np.array([heading])
        )[0]

        for i in range(n_cues):
            cue_pos = cue_positions[i]

            # Check if in FOV
            if fov_obj is not None and not fov_obj.contains_angle(bearings[i]):
                continue

            # Check line of sight
            if _line_of_sight_clear(env, position, cue_pos):
                visible_cue_list.append(i)
                cue_distances_list.append(np.linalg.norm(cue_pos - position))
                cue_bearings_list.append(bearings[i])

        visible_cues = np.array(visible_cue_list, dtype=np.intp)
        cue_distances = np.array(cue_distances_list, dtype=np.float64)
        cue_bearings = np.array(cue_bearings_list, dtype=np.float64)
    else:
        visible_cues = np.array([], dtype=np.intp)
        cue_distances = np.array([], dtype=np.float64)
        cue_bearings = np.array([], dtype=np.float64)

    return ViewshedResult(
        visible_bins=visible_bins,
        visible_cues=visible_cues,
        cue_distances=cue_distances,
        cue_bearings=cue_bearings,
        occlusion_map=occlusion_map,
        _total_bins=env.n_bins,
    )


def compute_view_field(
    env: Environment,
    position: NDArray[np.float64],
    heading: float,
    *,
    fov: FieldOfView | float | None = None,
    n_rays: int = 360,
) -> NDArray[np.bool_]:
    """Compute binary visibility mask from observer position.

    Parameters
    ----------
    env : Environment
        The environment to analyze.
    position : NDArray[np.float64], shape (2,)
        Observer position.
    heading : float
        Observer heading in radians.
    fov : FieldOfView, float, or None, default None
        Field of view constraint.
    n_rays : int, default 360
        Number of rays to cast.

    Returns
    -------
    NDArray[np.bool_], shape (n_bins,)
        True for visible bins, False otherwise.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.visibility import compute_view_field
    >>> rng = np.random.default_rng(42)
    >>> positions = rng.uniform(0, 100, (1000, 2))
    >>> env = Environment.from_samples(positions, bin_size=5.0)
    >>> mask = compute_view_field(env, np.array([50.0, 50.0]), heading=0.0)
    >>> mask.dtype == bool
    True
    """
    result = compute_viewshed(env, position, heading, fov=fov, n_rays=n_rays)
    mask = np.zeros(env.n_bins, dtype=bool)
    mask[result.visible_bins] = True
    return mask


def visible_cues(
    env: Environment,
    position: NDArray[np.float64],
    heading: float,
    cue_positions: NDArray[np.float64],
    *,
    fov: FieldOfView | float | None = None,
) -> tuple[NDArray[np.bool_], NDArray[np.float64], NDArray[np.float64]]:
    """Check visibility of cue positions.

    Parameters
    ----------
    env : Environment
        The environment.
    position : NDArray[np.float64], shape (2,)
        Observer position.
    heading : float
        Observer heading in radians.
    cue_positions : NDArray[np.float64], shape (n_cues, 2)
        Positions of cues to check.
    fov : FieldOfView, float, or None, default None
        Field of view constraint.

    Returns
    -------
    visible : NDArray[np.bool_], shape (n_cues,)
        True for visible cues.
    distances : NDArray[np.float64], shape (n_cues,)
        Distances to all cues (NaN for not visible).
    bearings : NDArray[np.float64], shape (n_cues,)
        Egocentric bearings to all cues (NaN for not visible).

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.visibility import visible_cues

    Create environment and check cue visibility:

    >>> positions = np.random.rand(100, 2) * 100
    >>> env = Environment.from_samples(positions, bin_size=5.0)
    >>> cues = np.array([[50, 50], [80, 20]])  # 2 cues
    >>> visible, distances, bearings = visible_cues(
    ...     env, position=np.array([25, 25]), heading=0.0, cue_positions=cues
    ... )  # doctest: +SKIP
    """
    from neurospatial.ops.egocentric import compute_egocentric_bearing

    position = np.asarray(position, dtype=np.float64)
    cue_positions = np.asarray(cue_positions, dtype=np.float64)
    n_cues = len(cue_positions)

    # Parse FOV
    if fov is None:
        fov_obj = None
    elif isinstance(fov, (int, float)):
        fov_obj = FieldOfView.symmetric(float(fov) / 2)
    else:
        fov_obj = fov

    # Compute distances and bearings
    distances = np.linalg.norm(cue_positions - position, axis=1)
    bearings = compute_egocentric_bearing(
        cue_positions, position.reshape(1, -1), np.array([heading])
    )[0]

    visible = np.zeros(n_cues, dtype=bool)

    for i in range(n_cues):
        # Check FOV
        if fov_obj is not None and not fov_obj.contains_angle(bearings[i]):
            continue

        # Check line of sight
        if _line_of_sight_clear(env, position, cue_positions[i]):
            visible[i] = True

    # Set NaN for non-visible
    result_distances = np.where(visible, distances, np.nan)
    result_bearings = np.where(visible, bearings, np.nan)

    return visible, result_distances, result_bearings


def compute_viewshed_trajectory(
    env: Environment,
    trajectory: NDArray[np.float64],
    headings: NDArray[np.float64],
    *,
    fov: FieldOfView | float | None = None,
    n_rays: int = 360,
    cue_positions: NDArray[np.float64] | None = None,
) -> list[ViewshedResult]:
    """Compute viewshed at each point along a trajectory.

    Parameters
    ----------
    env : Environment
        The environment.
    trajectory : NDArray[np.float64], shape (n_time, 2)
        Positions along trajectory.
    headings : NDArray[np.float64], shape (n_time,)
        Headings at each position.
    fov : FieldOfView, float, or None, default None
        Field of view constraint.
    n_rays : int, default 360
        Number of rays per viewshed.
    cue_positions : NDArray[np.float64], shape (n_cues, 2), optional
        Cue positions to check.

    Returns
    -------
    list of ViewshedResult
        Viewshed result at each trajectory point.
    """
    trajectory = np.asarray(trajectory, dtype=np.float64)
    headings = np.asarray(headings, dtype=np.float64)

    results = []
    for i in range(len(trajectory)):
        result = compute_viewshed(
            env,
            trajectory[i],
            headings[i],
            fov=fov,
            n_rays=n_rays,
            cue_positions=cue_positions,
        )
        results.append(result)

    return results


def visibility_occupancy(
    env: Environment,
    trajectory: NDArray[np.float64],
    headings: NDArray[np.float64],
    times: NDArray[np.float64],
    *,
    fov: FieldOfView | float | None = None,
    n_rays: int = 360,
) -> NDArray[np.float64]:
    """Compute time each bin was visible during trajectory.

    Parameters
    ----------
    env : Environment
        The environment.
    trajectory : NDArray[np.float64], shape (n_time, 2)
        Positions along trajectory.
    headings : NDArray[np.float64], shape (n_time,)
        Headings at each position.
    times : NDArray[np.float64], shape (n_time,)
        Timestamps for each position.
    fov : FieldOfView, float, or None, default None
        Field of view constraint.
    n_rays : int, default 360
        Number of rays per viewshed.

    Returns
    -------
    NDArray[np.float64], shape (n_bins,)
        Total time (seconds) each bin was visible.
    """
    trajectory = np.asarray(trajectory, dtype=np.float64)
    headings = np.asarray(headings, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)

    n_time = len(times)
    occupancy = np.zeros(env.n_bins, dtype=np.float64)

    # Compute dt for each frame
    dt = np.diff(times)
    dt = np.append(dt, dt[-1] if len(dt) > 0 else 0.0)  # Repeat last dt

    for i in range(n_time):
        result = compute_viewshed(
            env,
            trajectory[i],
            headings[i],
            fov=fov,
            n_rays=n_rays,
        )
        occupancy[result.visible_bins] += dt[i]

    return occupancy
