"""Visibility and gaze computation for spatial view analysis.

Provides tools for computing what an animal can see from any position in
an environment, including field-of-view constraints, line-of-sight checks,
and viewshed analysis.

Import Paths
------------
After package reorganization (v0.4.0+), use::

    from neurospatial.ops.visibility import (
        FieldOfView,
        ViewshedResult,
        compute_viewed_location,
        compute_viewshed,
        compute_view_field,
        compute_viewshed_trajectory,
        visibility_occupancy,
        visible_cues,
    )

    # Or import from the ops package
    from neurospatial.ops import FieldOfView, compute_viewshed

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

This matches the convention in ``neurospatial.ops.egocentric``.

References
----------
.. [1] Rolls, E. T., et al. (1997). Spatial view cells in the primate
       hippocampus. European Journal of Neuroscience, 9(8), 1789-1794.
.. [2] Wallace, D. J., et al. (2013). Rats maintain an overhead binocular
       field at the expense of constant visual coverage of the environment.
       Vision Research, 78, 1-10.

See Also
--------
neurospatial.ops.egocentric : Egocentric coordinate transforms
neurospatial.spatial_view_field : Spatial view field computation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurospatial.geometry import (
    CallIndexes,
    batch_line_of_sight_clear,
    boundary_view,
    locate,
)

if TYPE_CHECKING:
    from neurospatial.geometry import Arena, Binning, LookupIndexes
    from neurospatial.ops._protocols import BinCentersConsumer, BinningConsumer

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
    >>> from neurospatial.ops.visibility import FieldOfView

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
        >>> from neurospatial.ops.visibility import FieldOfView
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
        >>> from neurospatial.ops.visibility import FieldOfView
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
        >>> from neurospatial.ops.visibility import FieldOfView
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
    >>> from neurospatial.ops.visibility import ViewshedResult
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

    def visible_bin_centers(self, env: BinCentersConsumer) -> NDArray[np.float64]:
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
    env: BinningConsumer | None = None,
    max_distance: float = 100.0,
    boundary_source: Literal["physical", "analysis"] = "physical",
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

    boundary_source : {'physical', 'analysis'}, default='physical'
        Declared physical walls, or explicitly requested analysis-cell boundaries.
        Physical mode requires exact supported wall geometry; missing analysis
        bins do not occlude a physical ray. Fixed-distance gaze ignores this.

    Returns
    -------
    NDArray[np.float64], shape (n_time, 2)
        Viewed location in allocentric coordinates. NaN if ray doesn't
        intersect environment.

    Raises
    ------
    ValueError
        If method is invalid, ray_cast/boundary has no environment, or the
        requested search distance is not finite and positive.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.ops.visibility import compute_viewed_location

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

    else:
        assert env is not None
        viewed = boundary_view(
            env.binning,
            env.arena,
            positions,
            gaze_direction,
            max_distance,
            boundary_source,
            CallIndexes(),
        )

    return viewed


def _ray_cast_to_boundary(
    env: BinningConsumer,
    origin: NDArray[np.float64],
    direction: float,
    max_distance: float = 100.0,
    n_steps: int = 100,
    *,
    boundary_source: Literal["physical", "analysis"] = "physical",
) -> NDArray[np.float64] | None:
    """Cast a ray from origin in direction until it exits the environment.

    Uses exact geometric intersection; n_steps is retained for compatibility.

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
        Retained for compatibility; geometric intersections do not use steps.

    boundary_source : {'physical', 'analysis'}, default='physical'
        Declared physical walls, or explicitly requested analysis-cell boundaries.
        Physical mode requires exact supported wall geometry; missing analysis
        bins do not occlude a physical ray. Fixed-distance gaze ignores this.

    Returns
    -------
    NDArray[np.float64] or None
        Intersection point, or None if no intersection found.
    """
    viewed = compute_viewed_location(
        np.asarray(origin).reshape(1, 2),
        np.array([direction]),
        method="ray_cast",
        env=env,
        max_distance=max_distance,
        boundary_source=boundary_source,
    )
    return None if not np.all(np.isfinite(viewed[0])) else viewed[0]


def compute_viewshed(
    env: BinningConsumer,
    position: NDArray[np.float64],
    heading: float,
    *,
    fov: FieldOfView | float | None = None,
    n_rays: int = 360,
    cue_positions: NDArray[np.float64] | None = None,
    boundary_source: Literal["physical", "analysis"] = "physical",
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

    boundary_source : {'physical', 'analysis'}, default='physical'
        Declared physical walls, or explicitly requested analysis-cell boundaries.
        Physical mode requires exact supported wall geometry; missing analysis
        bins do not occlude a physical ray. Fixed-distance gaze ignores this.

    Returns
    -------
    ViewshedResult
        Viewshed analysis result with visible bins, boundaries, and cues.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.ops.visibility import compute_viewshed, FieldOfView
    >>> rng = np.random.default_rng(42)
    >>> positions = rng.uniform(0, 100, (1000, 2))
    >>> env = Environment.from_samples(positions, bin_size=5.0)
    >>> result = compute_viewshed(
    ...     env, np.array([50.0, 50.0]), heading=0.0, boundary_source="analysis"
    ... )
    >>> result.n_visible_bins > 0
    True
    """
    return _viewshed(
        env.binning,
        env.arena,
        position,
        heading,
        fov=fov,
        n_rays=n_rays,
        cue_positions=cue_positions,
        boundary_source=boundary_source,
    )


def _angular_range(
    fov: FieldOfView | float | None,
) -> tuple[float, float, FieldOfView | None, bool]:
    """Egocentric ray range ``(min, max)``, the FieldOfView, and full circle.

    A number is the total field-of-view angle in radians.
    """
    if fov is None:
        return -np.pi, np.pi, None, True
    if isinstance(fov, (int, float)):
        half_angle = float(fov) / 2
        return -half_angle, half_angle, FieldOfView.symmetric(half_angle), False
    return fov.right_angle, fov.left_angle, fov, False


def _check_n_rays(n_rays: int) -> None:
    """Raise ValueError unless ``n_rays`` is a positive integer."""
    if (
        isinstance(n_rays, (bool, np.bool_))
        or not isinstance(n_rays, (int, np.integer))
        or n_rays < 1
    ):
        raise ValueError("n_rays must be a positive integer.")


def _check_viewshed_options(
    binning: Binning,
    arena: Arena,
    *,
    fov: FieldOfView | float | None,
    n_rays: int,
    boundary_source: Literal["physical", "analysis"],
) -> None:
    """Raise the errors `_viewshed` raises for these options, in its order."""
    from neurospatial.geometry import boundary_geometry

    _angular_range(fov)
    _check_n_rays(n_rays)
    boundary_geometry(binning, arena, boundary_source)


def _viewshed(
    binning: Binning,
    arena: Arena,
    position: NDArray[np.float64],
    heading: float,
    *,
    fov: FieldOfView | float | None,
    n_rays: int,
    cue_positions: NDArray[np.float64] | None,
    boundary_source: Literal["physical", "analysis"],
    indexes: LookupIndexes | None = None,
) -> ViewshedResult:
    """Compute a viewshed from the bins and domain; see :func:`compute_viewshed`.

    ``indexes`` keeps the locator's search structures across the frames of
    one trajectory; None keeps them for this viewshed only.
    """
    from neurospatial.ops.egocentric import compute_egocentric_bearing

    position = np.asarray(position, dtype=np.float64)
    n_bins = binning.bins.n_bins
    indexes = CallIndexes() if indexes is None else indexes

    min_angle, max_angle, fov_obj, full_circle = _angular_range(fov)
    # Before np.linspace, which raises a bare TypeError for a non-integer.
    _check_n_rays(n_rays)

    # Generate ray angles (in egocentric coordinates).
    # For the full circle, -pi and +pi are the same direction (behind), so
    # exclude the closing endpoint to avoid double-counting that ray. For a
    # bounded FOV both endpoints are distinct and should be kept.
    ray_angles_ego = np.linspace(min_angle, max_angle, n_rays, endpoint=not full_circle)

    # Convert to allocentric
    ray_angles_allo = ray_angles_ego + heading

    from neurospatial.geometry import (
        analysis_interior,
        boundary_geometry,
        physical_ray_hits,
        ray_hits,
        valid_observers,
    )

    walls = boundary_geometry(binning, arena, boundary_source)
    if position.shape != (2,):
        raise ValueError("position must have shape (2,).")
    if (
        not np.isfinite(heading)
        or not valid_observers(
            binning, arena, position.reshape(1, 2), boundary_source, indexes
        )[0]
    ):
        return ViewshedResult(
            np.empty(0, dtype=np.intp),
            np.empty(0, dtype=np.intp),
            np.empty(0),
            np.empty(0),
            np.zeros(n_bins),
            _total_bins=n_bins,
        )
    # Enclose both known walls and retained bin centers; no fixed 200-unit cap.
    coordinates = np.vstack(
        (np.array(walls.bounds).reshape(2, 2), binning.bins.centers)
    )
    max_distance = max(
        float(np.hypot.reduce(coordinates - position, axis=1).max()) * 2, 1.0
    )
    n_steps = 100
    origins = np.broadcast_to(position, (n_rays, 2))
    hits = (
        physical_ray_hits(arena, origins, ray_angles_allo, max_distance)
        if boundary_source == "physical"
        else ray_hits(
            walls,
            origins,
            ray_angles_allo,
            max_distance,
            analysis_interior(binning, arena, indexes),
        )
    )
    hit_distance = np.hypot.reduce(hits - position, axis=1)
    ray_lengths = np.where(np.isfinite(hit_distance), hit_distance, max_distance)

    # Pre-compute direction vectors for all rays: shape (n_rays, 2)
    dx = np.cos(ray_angles_allo)
    dy = np.sin(ray_angles_allo)
    directions = np.column_stack([dx, dy])  # (n_rays, 2)

    # Create ray sample points: distances along ray
    distances = np.linspace(0, 1, n_steps, endpoint=False)  # strictly before contact

    # Generate all ray points at once: (n_rays, n_steps, 2)
    # position: (2,), directions: (n_rays, 2), distances: (n_steps,)
    ray_points_all = (
        position[np.newaxis, np.newaxis, :]
        + directions[:, np.newaxis, :]
        * (ray_lengths[:, np.newaxis] * distances)[..., np.newaxis]
    )

    # A zero-length ray ends on the observer's wall; it exposes no cell interior.
    ray_points_all[ray_lengths == 0] = np.nan

    # Reshape for batch bin_at query: (n_rays * n_steps, 2)
    flat_ray_points = ray_points_all.reshape(-1, 2)
    flat_bin_indices = locate(
        binning, arena, flat_ray_points, indexes, call="compute_viewshed"
    )

    # Reshape back: (n_rays, n_steps)
    bin_indices_2d = flat_bin_indices.reshape(n_rays, n_steps)

    # Sample analysis bins before the first chosen wall contact. Missing bins
    # do not occlude physical rays.
    occlusion_scores = np.zeros(n_bins, dtype=np.float64)
    ray_counts = np.zeros(n_bins, dtype=np.float64)
    visible_bin_mask = np.zeros(n_bins, dtype=bool)

    for ray_idx in range(n_rays):
        valid_bins = bin_indices_2d[ray_idx]
        valid_bins = valid_bins[valid_bins >= 0]

        # Use numpy's unique and bincount for fast accumulation
        unique_bins, counts = np.unique(valid_bins, return_counts=True)
        visible_bin_mask[unique_bins] = True
        occlusion_scores[unique_bins] += counts.astype(np.float64)
        ray_counts[unique_bins] += counts.astype(np.float64)

    visible_bins = np.where(visible_bin_mask)[0].astype(np.intp)

    # Normalize occlusion scores
    max_count = np.max(ray_counts) if np.max(ray_counts) > 0 else 1.0
    occlusion_map = np.where(ray_counts > 0, occlusion_scores / max_count, 0.0)

    # Check cue visibility (vectorized)
    if cue_positions is not None:
        cue_positions = np.asarray(cue_positions, dtype=np.float64)
        n_cues = len(cue_positions)

        if n_cues > 0:
            # Compute bearings (already vectorized)
            bearings = compute_egocentric_bearing(
                position.reshape(1, -1), np.array([heading]), cue_positions
            )[0]

            # Compute distances (vectorized)
            cue_dists = np.linalg.norm(cue_positions - position, axis=1)

            # Check FOV (vectorized) - FieldOfView.contains_angle handles arrays
            if fov_obj is not None:
                in_fov = fov_obj.contains_angle(bearings)
            else:
                in_fov = np.ones(n_cues, dtype=bool)

            # Only check line of sight for cues in FOV (vectorized batch check)
            fov_indices = np.where(in_fov)[0]
            if len(fov_indices) > 0:
                los_clear = batch_line_of_sight_clear(
                    binning,
                    arena,
                    position,
                    cue_positions[fov_indices],
                    boundary_source=boundary_source,
                    indexes=indexes,
                )
                # Build visibility mask
                visible_mask = np.zeros(n_cues, dtype=bool)
                visible_mask[fov_indices] = los_clear

                visible_cue_indices = np.where(visible_mask)[0]
                visible_cues = visible_cue_indices.astype(np.intp)
                cue_distances = cue_dists[visible_cue_indices]
                cue_bearings = bearings[visible_cue_indices]
            else:
                visible_cues = np.array([], dtype=np.intp)
                cue_distances = np.array([], dtype=np.float64)
                cue_bearings = np.array([], dtype=np.float64)
        else:
            visible_cues = np.array([], dtype=np.intp)
            cue_distances = np.array([], dtype=np.float64)
            cue_bearings = np.array([], dtype=np.float64)
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
        _total_bins=n_bins,
    )


def compute_view_field(
    env: BinningConsumer,
    position: NDArray[np.float64],
    heading: float,
    *,
    fov: FieldOfView | float | None = None,
    n_rays: int = 360,
    boundary_source: Literal["physical", "analysis"] = "physical",
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

    boundary_source : {'physical', 'analysis'}, default='physical'
        Declared physical walls, or explicitly requested analysis-cell boundaries.
        Physical mode requires exact supported wall geometry; missing analysis
        bins do not occlude a physical ray. Fixed-distance gaze ignores this.

    Returns
    -------
    NDArray[np.bool_], shape (n_bins,)
        True for visible bins, False otherwise.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.ops.visibility import compute_view_field
    >>> rng = np.random.default_rng(42)
    >>> positions = rng.uniform(0, 100, (1000, 2))
    >>> env = Environment.from_samples(positions, bin_size=5.0)
    >>> mask = compute_view_field(
    ...     env, np.array([50.0, 50.0]), heading=0.0, boundary_source="analysis"
    ... )
    >>> mask.dtype == bool
    True
    """
    result = compute_viewshed(
        env, position, heading, fov=fov, n_rays=n_rays, boundary_source=boundary_source
    )
    mask = np.zeros(result.occlusion_map.shape[0], dtype=bool)
    mask[result.visible_bins] = True
    return mask


def visible_cues(
    env: BinningConsumer,
    position: NDArray[np.float64],
    heading: float,
    cue_positions: NDArray[np.float64],
    *,
    fov: FieldOfView | float | None = None,
    boundary_source: Literal["physical", "analysis"] = "physical",
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

    boundary_source : {'physical', 'analysis'}, default='physical'
        Declared physical walls, or explicitly requested analysis-cell boundaries.
        Physical mode requires exact supported wall geometry; missing analysis
        bins do not occlude a physical ray. Fixed-distance gaze ignores this.

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
    >>> from neurospatial.ops.visibility import visible_cues

    Create environment and check cue visibility:

    >>> positions = np.random.rand(100, 2) * 100
    >>> env = Environment.from_samples(positions, bin_size=5.0)
    >>> cues = np.array([[50, 50], [80, 20]])  # 2 cues
    >>> visible, distances, bearings = visible_cues(
    ...     env,
    ...     position=np.array([25, 25]),
    ...     heading=0.0,
    ...     cue_positions=cues,
    ...     boundary_source="analysis",
    ... )  # doctest: +SKIP
    """
    from neurospatial.ops.egocentric import compute_egocentric_bearing

    position = np.asarray(position, dtype=np.float64)
    cue_positions = np.asarray(cue_positions, dtype=np.float64)
    n_cues = len(cue_positions)

    if n_cues == 0:
        return (
            np.array([], dtype=bool),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
        )

    # Parse FOV
    if fov is None:
        fov_obj = None
    elif isinstance(fov, (int, float)):
        fov_obj = FieldOfView.symmetric(float(fov) / 2)
    else:
        fov_obj = fov

    # Compute distances and bearings (vectorized)
    distances = np.linalg.norm(cue_positions - position, axis=1)
    bearings = compute_egocentric_bearing(
        position.reshape(1, -1), np.array([heading]), cue_positions
    )[0]

    # Check FOV (vectorized) - FieldOfView.contains_angle handles arrays
    if fov_obj is not None:
        in_fov = fov_obj.contains_angle(bearings)
    else:
        in_fov = np.ones(n_cues, dtype=bool)

    # Only check line of sight for cues in FOV (vectorized batch check)
    visible = np.zeros(n_cues, dtype=bool)
    fov_indices = np.where(in_fov)[0]

    if len(fov_indices) > 0:
        los_clear = batch_line_of_sight_clear(
            env.binning,
            env.arena,
            position,
            cue_positions[fov_indices],
            boundary_source=boundary_source,
        )
        visible[fov_indices] = los_clear

    # Set NaN for non-visible
    result_distances = np.where(visible, distances, np.nan)
    result_bearings = np.where(visible, bearings, np.nan)

    return visible, result_distances, result_bearings


def compute_viewshed_trajectory(
    env: BinningConsumer,
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
    *,
    fov: FieldOfView | float | None = None,
    n_rays: int = 360,
    cue_positions: NDArray[np.float64] | None = None,
    boundary_source: Literal["physical", "analysis"] = "physical",
) -> list[ViewshedResult]:
    """Compute viewshed at each point along a trajectory.

    Parameters
    ----------
    env : Environment
        The environment.
    positions : NDArray[np.float64], shape (n_time, 2)
        Positions along trajectory.
    headings : NDArray[np.float64], shape (n_time,)
        Headings at each position.
    fov : FieldOfView, float, or None, default None
        Field of view constraint.
    n_rays : int, default 360
        Number of rays per viewshed.
    cue_positions : NDArray[np.float64], shape (n_cues, 2), optional
        Cue positions to check.

    boundary_source : {'physical', 'analysis'}, default='physical'
        Declared physical walls, or explicitly requested analysis-cell boundaries.
        Physical mode requires exact supported wall geometry; missing analysis
        bins do not occlude a physical ray. Fixed-distance gaze ignores this.

    Returns
    -------
    list of ViewshedResult
        Viewshed result at each trajectory point.
    """
    positions = np.asarray(positions, dtype=np.float64)
    headings = np.asarray(headings, dtype=np.float64)

    binning, arena = env.binning, env.arena
    indexes = CallIndexes()
    return [
        _viewshed(
            binning,
            arena,
            positions[i],
            headings[i],
            fov=fov,
            n_rays=n_rays,
            cue_positions=cue_positions,
            boundary_source=boundary_source,
            indexes=indexes,
        )
        for i in range(len(positions))
    ]


def visibility_occupancy(
    env: BinningConsumer,
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
    *,
    fov: FieldOfView | float | None = None,
    n_rays: int = 360,
    boundary_source: Literal["physical", "analysis"] = "physical",
    max_gap: float | None = 0.5,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Compute time each bin was visible during trajectory.

    Each sampling interval ``[times[k], times[k + 1])`` adds its duration to
    every bin visible from sample ``k`` (its position and heading), the same
    start-sample rule :meth:`Environment.occupancy` uses. The last sample has
    no interval and adds no time.

    Parameters
    ----------
    env : Environment
        The environment.
    times : NDArray[np.float64], shape (n_time,)
        Timestamps in seconds for each position; finite and non-decreasing.
    positions : NDArray[np.float64], shape (n_time, 2)
        Positions along trajectory. A row with a NaN coordinate is a missing
        sample (a lost tracking frame); intervals that start or end at it add
        no time.
    headings : NDArray[np.float64], shape (n_time,)
        Allocentric headings in radians at each position. A NaN heading is a
        missing sample, like a NaN position.
    fov : FieldOfView, float, or None, default None
        Field of view constraint.
    n_rays : int, default 360
        Number of rays per viewshed.

    boundary_source : {'physical', 'analysis'}, default='physical'
        Declared physical walls, or explicitly requested analysis-cell boundaries.
        Physical mode requires exact supported wall geometry; missing analysis
        bins do not occlude a physical ray. Fixed-distance gaze ignores this.
    max_gap : float or None, default=0.5
        Longest sampling interval (seconds) treated as continuous recording.
        Longer intervals (dropped frames, pauses between sessions) add no
        viewing time. ``None`` disables the gap check.
        A ``UserWarning`` names the cause when every interval is excluded,
        or when the median sampling interval exceeds ``max_gap`` (usually
        ``times`` that are not in seconds).
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``times``). An interval counts only if it lies entirely
        inside one window. ``None`` (default) means unrestricted.
    break_before : ArrayLike of bool, shape (n_time,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored.

    Returns
    -------
    NDArray[np.float64], shape (n_bins,)
        Total time (seconds) each bin was visible. A bin visible from every
        sample receives the total usable time, which equals
        ``env.occupancy(times, positions, max_gap=..., epochs=...,
        break_before=...).sum()`` when every sample is inside the environment
        and no heading is missing.

    Raises
    ------
    ValueError
        If ``times`` and ``positions`` are misaligned, ``times`` is not finite
        and non-decreasing, ``positions`` is not ``(n_time, 2)``, ``headings``
        is not ``(n_time,)``, ``max_gap`` is not positive, or ``epochs`` or
        ``break_before`` is malformed.

    Warns
    -----
    UserWarning
        When every interval is excluded (by ``max_gap``, ``epochs``,
        ``break_before`` or missing samples), or when the median sampling
        interval exceeds ``max_gap``.

    Notes
    -----
    Interval ``k`` adds time only when it passes ``max_gap``, lies inside
    ``epochs``, is not crossed by a declared break, and both of its samples
    have a finite position and heading. A tracking pause or the join between
    two recordings is therefore not counted as viewing time. Viewsheds are
    computed only for samples that start such an interval.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.ops import visibility_occupancy
    >>> x, y = np.meshgrid(np.arange(0.0, 20.0), np.arange(0.0, 20.0))
    >>> env = Environment.from_samples(np.c_[x.ravel(), y.ravel()], bin_size=5.0)
    >>> times = np.array([0.0, 0.1, 0.2, 5.0, 5.1])  # a 4.8 s tracking pause
    >>> positions = np.full((5, 2), 10.0)
    >>> headings = np.zeros(5)
    >>> seen = visibility_occupancy(
    ...     env, times, positions, headings, n_rays=36, boundary_source="analysis"
    ... )
    >>> round(float(seen.max()), 6)  # 0.2 s before the pause + 0.1 s after
    0.3
    """
    from neurospatial._exceptions import _format_error
    from neurospatial._intervals import (
        as_intervals,
        interval_valid_mask,
        warn_excluded_intervals,
    )
    from neurospatial._observation import validate_break_before, validate_max_gap
    from neurospatial._validation import validate_times_positions

    times, positions = validate_times_positions(
        times, positions, call="visibility_occupancy"
    )
    headings = np.asarray(headings, dtype=np.float64)
    n_time = len(times)
    if positions.shape != (n_time, 2):
        raise ValueError(
            _format_error(
                "visibility_occupancy() needs positions of shape (n_time, 2); "
                f"got {positions.shape}.",
                why="Why: each viewshed is cast from a 2-D (x, y) position.",
                fix="pass one (x, y) row per timestamp",
            )
        )
    if headings.shape != (n_time,):
        raise ValueError(
            _format_error(
                "visibility_occupancy() needs one heading per timestamp, shape "
                f"({n_time},); got {headings.shape}.",
                why="Why: each sample's field of view is centered on its heading.",
                fix="pass headings with the same length as times (NaN marks a "
                "missing heading)",
            )
        )
    validate_max_gap(max_gap)
    breaks = validate_break_before(break_before, n_time, call="visibility_occupancy")
    resolved_epochs = as_intervals(epochs, name="epochs")

    binning, arena = env.binning, env.arena
    # Checked here because no viewshed is computed when no interval is usable.
    _check_viewshed_options(
        binning, arena, fov=fov, n_rays=n_rays, boundary_source=boundary_source
    )
    occupancy = np.zeros(binning.bins.n_bins, dtype=np.float64)
    if n_time < 2:
        return occupancy

    observed = np.isfinite(positions).all(axis=1) & np.isfinite(headings)
    usable = interval_valid_mask(
        times, max_gap=max_gap, epochs=resolved_epochs, break_before=breaks
    )
    usable &= observed[:-1] & observed[1:]
    warn_excluded_intervals(
        times,
        usable,
        max_gap=max_gap,
        epochs=resolved_epochs,
        start_bin_gate=not bool(observed.all()),
        break_before=breaks,
        result="no bin received viewing time",
    )

    dt = np.diff(times)
    indexes = CallIndexes()
    # Viewsheds are per-frame; compute them only for frames that add time.
    for i in np.flatnonzero(usable & (dt > 0)):
        result = _viewshed(
            binning,
            arena,
            positions[i],
            headings[i],
            fov=fov,
            n_rays=n_rays,
            cue_positions=None,
            boundary_source=boundary_source,
            indexes=indexes,
        )
        np.add.at(occupancy, result.visible_bins, dt[i])

    return occupancy
