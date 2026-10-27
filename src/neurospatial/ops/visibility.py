"""Visibility and gaze computation for spatial view analysis.

Provides tools for computing what an animal can see from any position in
an environment, including field-of-view constraints, line-of-sight checks,
and viewshed analysis. Import from ``neurospatial.ops.visibility`` or
``neurospatial.ops``.

Common Use Cases
----------------
- Spatial view cell analysis (what location is being viewed)
- Visibility-based neural encoding models
- Gaze-contingent analysis
- Landmark visibility during navigation

Which Function Should I Use?
----------------------------
**Computing viewed location from gaze direction?**
    Use ``compute_viewed_location()`` with method="ray_cast" for the first
    wall along the gaze, or method="fixed_distance" for a point at a fixed
    distance that stops at a nearer wall.

**Need full visibility analysis from a position?**
    Use ``compute_viewshed()`` to get the visible bins and cues. A bin is
    visible when its center is inside the field of view with a clear line of
    sight.

**Just need a binary visibility mask?**
    Use ``compute_view_field()`` for a simple bool array of visible bins.

**Check if specific cues are visible?**
    Use ``visible_cues()`` for line-of-sight checks to cue positions.

**Analyze visibility along a trajectory?**
    Use ``compute_viewshed_trajectory()`` or ``visibility_occupancy()``.

Species-Specific Field of View
------------------------------
Use ``FieldOfView`` presets for biologically realistic constraints:

- ``FieldOfView.rat()`` - 320 degree FOV with a 40 degree blind spot behind
- ``FieldOfView.mouse()`` - 310 degree FOV, similar to rat
- ``FieldOfView.primate()`` - 180 degree forward-facing FOV

Coordinate Conventions
----------------------
**Egocentric direction** (for field of view):
- 0 radians = directly ahead
- pi/2 radians = to the left
- -pi/2 radians = to the right
- +/-pi radians = behind (reported as +pi)

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
neurospatial.encoding.compute_view_rate : Firing rate by viewed location
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurospatial._angles import wrap_angle
from neurospatial._exceptions import _format_error
from neurospatial._results import ReadOnlyArraysMixin
from neurospatial.geometry import (
    CallIndexes,
    batch_line_of_sight_clear,
    boundary_view,
    valid_observers,
)
from neurospatial.ops.egocentric import _check_pose

if TYPE_CHECKING:
    from collections.abc import Callable

    from neurospatial.geometry import Arena, Binning, LookupIndexes
    from neurospatial.geometry._boundaries import BoundarySource
    from neurospatial.ops._protocols import BinningConsumer, BinsConsumer

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

    Defines the egocentric angular extent of what an animal can see, with an
    optional binocular region straight ahead. Angles are egocentric radians:
    0 = ahead, +pi/2 = left, -pi/2 = right, +/-pi = behind. The field of view
    is the closed range ``[right_angle, left_angle]``; the directions behind
    the animal outside it form the blind spot.

    Attributes
    ----------
    left_angle : float
        Leftmost visible direction in radians, in ``[right_angle, pi]``
        (e.g., pi/2 = 90 degrees left).
    right_angle : float
        Rightmost visible direction in radians, in ``[-pi, left_angle]``
        (e.g., -pi/2 = 90 degrees right).
    binocular_half_angle : float, optional
        Half-angle in radians of the binocular overlap region centered on 0
        (straight ahead). Default is 0.0 (no binocular region). A positive
        value must fit inside the field of view:
        ``binocular_half_angle <= min(left_angle, -right_angle)``.

    Raises
    ------
    ValueError
        If an angle is not finite, the angles are not ordered
        ``-pi <= right_angle <= left_angle <= pi``, or the binocular region
        is negative or extends past the field of view.

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
    >>> round(fov_rat.total_angle_degrees)  # a 40-degree blind spot behind
    320

    >>> fov_primate = FieldOfView.primate()
    >>> round(fov_primate.total_angle_degrees)
    180

    Check if an angle is within the field of view:

    >>> fov = FieldOfView.symmetric(half_angle=np.pi / 2)
    >>> fov.contains_angle(0.0)  # Ahead
    True
    >>> fov.contains_angle(np.pi)  # Behind
    False

    A field of view covering only the left side:

    >>> left_only = FieldOfView(left_angle=np.pi / 2, right_angle=0.1)
    >>> left_only.contains_angle(0.5), left_only.contains_angle(-0.5)
    (True, False)
    """

    left_angle: float
    right_angle: float
    binocular_half_angle: float = 0.0

    def __post_init__(self) -> None:
        """Validate field of view parameters."""
        angles = {
            "left_angle": self.left_angle,
            "right_angle": self.right_angle,
            "binocular_half_angle": self.binocular_half_angle,
        }
        bad = [name for name, value in angles.items() if not np.isfinite(value)]
        if bad:
            raise ValueError(
                _format_error(
                    "FieldOfView needs finite angles; got "
                    + ", ".join(f"{name}={angles[name]}" for name in bad)
                    + ".",
                    fix="give every angle as a finite number of radians",
                )
            )
        if not -np.pi <= self.right_angle <= self.left_angle <= np.pi:
            raise ValueError(
                _format_error(
                    "FieldOfView needs -pi <= right_angle <= left_angle <= pi; "
                    f"got left_angle={self.left_angle:.3f}, "
                    f"right_angle={self.right_angle:.3f}.",
                    why="Why: angles are egocentric radians (0 = ahead, "
                    "+pi/2 = left, -pi/2 = right), and the field of view is "
                    "the range from right_angle to left_angle.",
                    fix="pass the left edge as left_angle and the right edge "
                    "as right_angle, both in [-pi, pi]; a full circle is "
                    "left_angle=pi, right_angle=-pi",
                )
            )
        half = self.binocular_half_angle
        if half < 0 or (
            half > 0 and not self.right_angle <= -half <= half <= self.left_angle
        ):
            raise ValueError(
                _format_error(
                    f"FieldOfView binocular_half_angle={half:.3f} does not fit "
                    f"in the field of view [{self.right_angle:.3f}, "
                    f"{self.left_angle:.3f}].",
                    why="Why: the binocular region is the range "
                    "[-binocular_half_angle, binocular_half_angle] straight "
                    "ahead that both eyes see, so it lies inside the field of "
                    "view.",
                    fix="pass 0 for no binocular region, or a half-angle no "
                    "larger than min(left_angle, -right_angle)",
                )
            )

    @classmethod
    def symmetric(cls, half_angle: float) -> FieldOfView:
        """Create a symmetric field of view.

        Parameters
        ----------
        half_angle : float
            Half-angle of the field of view in radians, in ``[0, pi]``.
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

        Rats see about 320 degrees, with a 40-degree blind spot directly
        behind (20 degrees each side) and a binocular region of about 50
        degrees in front.

        Returns
        -------
        FieldOfView
            Rat-like field of view.

        References
        ----------
        Wallace, D. J., et al. (2013). Vision Research, 78, 1-10.
        """
        return cls(
            left_angle=np.pi * 160 / 180,
            right_angle=-np.pi * 160 / 180,
            binocular_half_angle=np.pi * 25 / 180,
        )

    @classmethod
    def mouse(cls) -> FieldOfView:
        """Field of view for a mouse.

        Mice have a field of view similar to rats: about 310 degrees, with a
        50-degree blind spot directly behind and a binocular region of about
        40 degrees in front.

        Returns
        -------
        FieldOfView
            Mouse-like field of view.
        """
        return cls(
            left_angle=np.pi * 155 / 180,
            right_angle=-np.pi * 155 / 180,
            binocular_half_angle=np.pi * 20 / 180,
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
            left_angle=np.pi * 90 / 180,
            right_angle=-np.pi * 90 / 180,
            binocular_half_angle=np.pi * 60 / 180,
        )

    @property
    def total_angle(self) -> float:
        """Total field of view angle in radians.

        Returns
        -------
        float
            Angle from right_angle to left_angle, in ``[0, 2 * pi]``.
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
            Egocentric angle(s) in radians: 0 = ahead, pi/2 = left,
            -pi/2 = right. Wrapped to (-pi, pi] first, so directly behind is
            +pi. NaN is never inside.

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
        wrapped = wrap_angle(angle)
        in_bounds = (wrapped <= self.left_angle) & (wrapped >= self.right_angle)
        # Wrapping sends -pi to +pi; a field whose right edge is -pi still
        # reaches that direction, straight behind.
        if self.right_angle == -np.pi:
            in_bounds |= wrapped == np.pi
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
            Egocentric angle(s) in radians, wrapped to (-pi, pi] first.

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
        in_binocular = np.abs(wrap_angle(angle)) <= self.binocular_half_angle
        if angle.ndim == 0:
            return bool(in_binocular)
        return in_binocular


@dataclass(frozen=True, eq=False)
class ViewshedResult(ReadOnlyArraysMixin):
    """Result of viewshed analysis from a single observer position.

    Attributes
    ----------
    visibility_map : NDArray[np.float64], shape (n_bins,)
        1.0 where the bin is visible, 0.0 where it is outside the field of
        view or hidden behind a wall. NaN in every bin for a missing
        observation (a NaN or infinite position or heading). Averaging the
        maps of observed frames gives the fraction of frames each bin was in
        view.
    visible_cues : NDArray[np.intp], shape (n_visible_cues,)
        Indices into ``cue_positions`` of the visible cues, increasing.
    cue_distances : NDArray[np.float64], shape (n_visible_cues,)
        Distance from the observer to each visible cue, in environment units.
    cue_bearings : NDArray[np.float64], shape (n_visible_cues,)
        Egocentric bearing of each visible cue in radians, (-pi, pi]:
        0 = ahead, +pi/2 = left, -pi/2 = right.

    Raises
    ------
    ValueError
        If ``visibility_map`` is not a non-empty 1-D array of 1, 0 and NaN
        (all NaN or none), the cue arrays differ in length, the cue indices
        are not increasing non-negative integers, or a missing observation
        has visible cues.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.ops.visibility import ViewshedResult
    >>> result = ViewshedResult(
    ...     visibility_map=np.array([1.0, 1.0, 0.0, 1.0]),
    ...     visible_cues=np.array([], dtype=np.intp),
    ...     cue_distances=np.array([]),
    ...     cue_bearings=np.array([]),
    ... )
    >>> result.visible_bins
    array([0, 1, 3])
    >>> result.visibility_fraction
    0.75
    """

    visibility_map: NDArray[np.float64]
    visible_cues: NDArray[np.intp]
    cue_distances: NDArray[np.float64]
    cue_bearings: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Check that the fields describe one viewshed, then own the arrays."""
        problems = self._inconsistencies()
        if problems:
            raise ValueError(
                "ViewshedResult fields are inconsistent: " + "; ".join(problems) + "."
            )
        self._own_arrays()

    def _inconsistencies(self) -> list[str]:
        """List every way the fields disagree with each other."""
        problems: list[str] = []
        visibility = np.asarray(self.visibility_map, dtype=np.float64)
        if visibility.ndim != 1 or visibility.size == 0:
            problems.append("visibility_map needs shape (n_bins,) with n_bins >= 1")
        else:
            missing = np.isnan(visibility)
            if not np.all((visibility == 0) | (visibility == 1) | missing):
                problems.append("visibility_map holds only 1.0, 0.0 or NaN")
            elif missing.any() and not missing.all():
                problems.append(
                    "visibility_map is NaN in every bin (a missing observation) "
                    "or in none"
                )
            elif missing.all() and np.size(self.visible_cues):
                problems.append("a missing observation has no visible cues")
        cues = np.asarray(self.visible_cues)
        n_cues = cues.shape
        if cues.ndim != 1:
            problems.append("visible_cues needs shape (n_visible_cues,)")
        elif cues.size and not (
            np.issubdtype(cues.dtype, np.integer)
            and cues[0] >= 0
            and np.all(np.diff(cues) > 0)
        ):
            problems.append("visible_cues are increasing non-negative cue indices")
        for name in ("cue_distances", "cue_bearings"):
            if np.shape(getattr(self, name)) != n_cues:
                problems.append(f"{name} needs shape {n_cues}, like visible_cues")
        return problems

    @property
    def visible_bins(self) -> NDArray[np.intp]:
        """Indices of the visible bins, increasing; empty when missing."""
        return np.flatnonzero(self.visibility_map == 1.0)

    @property
    def n_bins(self) -> int:
        """Number of bins in the environment."""
        return int(self.visibility_map.size)

    @property
    def n_visible_bins(self) -> int:
        """Number of visible bins."""
        return int(np.count_nonzero(self.visibility_map == 1.0))

    @property
    def visibility_fraction(self) -> float:
        """Fraction of the environment's bins that are visible.

        Returns
        -------
        float
            Fraction in [0, 1], or NaN for a missing observation
            (``visibility_map`` all NaN).
        """
        return float(np.mean(self.visibility_map))

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

    def visible_bin_centers(self, env: BinsConsumer) -> NDArray[np.float64]:
        """Get allocentric positions of visible bin centers.

        Parameters
        ----------
        env : Environment
            The environment used for the viewshed computation.

        Returns
        -------
        NDArray[np.float64], shape (n_visible_bins, n_dims)
            Allocentric positions of visible bin centers.

        Raises
        ------
        ValueError
            If ``env`` has a different number of bins than this viewshed.
        """
        if env.n_bins != self.n_bins:
            raise ValueError(
                _format_error(
                    f"This viewshed covers {self.n_bins} bins, but env has "
                    f"{env.n_bins}.",
                    why="Why: visible bin indices only name bin centers in the "
                    "environment the viewshed was computed in.",
                    fix="pass the environment given to compute_viewshed",
                )
            )
        return env.bin_centers[self.visible_bins]


def compute_viewed_location(
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
    *,
    method: Literal["fixed_distance", "ray_cast"] = "ray_cast",
    view_distance: float | None = None,
    gaze_offsets: NDArray[np.float64] | None = None,
    env: BinningConsumer | None = None,
    max_distance: float | None = None,
    boundary_source: BoundarySource = "physical",
) -> NDArray[np.float64]:
    """Compute the location being viewed by the animal.

    Parameters
    ----------
    positions : NDArray[np.float64], shape (n_time, 2)
        Animal positions in allocentric coordinates (environment units).
    headings : NDArray[np.float64], shape (n_time,)
        Animal heading at each time (allocentric radians: 0 = East,
        +pi/2 = North).
    method : {"ray_cast", "fixed_distance"}, default "ray_cast"
        Method for computing viewed location:

        - "ray_cast": the first wall the gaze ray reaches. Needs ``env``.
        - "fixed_distance": the point ``view_distance`` ahead along the gaze,
          or the first wall when one is nearer. Without ``env`` the walls are
          unknown and the point is not stopped.
    view_distance : float or None, default None
        Distance along the gaze to the viewed point, in environment units
        (e.g. cm). Required with ``method="fixed_distance"`` and refused
        with ``"ray_cast"``: how far ahead an animal looks depends on the
        species, task and arena, so there is no default.
    gaze_offsets : NDArray[np.float64], shape (n_time,), optional
        Gaze direction relative to the heading, in egocentric radians
        (0 = ahead, + = left, - = right), e.g. from eye tracking. If None,
        gaze is aligned with heading.
    env : Environment, optional
        Supplies the walls. Required for the ray_cast method; with
        fixed_distance it stops the gaze at the first wall.
    max_distance : float or None, default None
        Longest gaze ray for the ray_cast method, in environment units.
        ``None`` searches far enough to reach every wall: twice the diagonal
        of the box enclosing the walls and the positions.
    boundary_source : {'physical', 'analysis'}, default='physical'
        Declared physical walls, or explicitly requested analysis-cell boundaries.
        Physical mode requires exact supported wall geometry; missing analysis
        bins do not occlude a physical ray. Unused without ``env``.

    Returns
    -------
    NDArray[np.float64], shape (n_time, 2)
        Viewed location in allocentric coordinates. NaN for a missing
        observation and, with ``env``, for an observer outside the walls;
        with ray_cast also where no wall lies within ``max_distance``.

    Raises
    ------
    ValueError
        If method is invalid, ray_cast has no environment, ``view_distance``
        is missing with fixed_distance, given with ray_cast or not a
        positive finite distance, the shapes of ``positions``, ``headings``
        and ``gaze_offsets`` disagree, or ``max_distance`` is not finite and
        positive.

    Examples
    --------
    >>> import numpy as np
    >>> from shapely.geometry import box
    >>> from neurospatial import Environment
    >>> from neurospatial.ops.visibility import compute_viewed_location

    Fixed distance method:

    >>> positions = np.array([[0.0, 0.0]])
    >>> headings = np.array([0.0])  # Facing East
    >>> viewed = compute_viewed_location(
    ...     positions, headings, method="fixed_distance", view_distance=10.0
    ... )
    >>> np.allclose(viewed[0], [10.0, 0.0])
    True

    The first wall along the gaze, however far away:

    >>> env = Environment.from_polygon(box(0, 0, 300, 300), bin_size=10.0)
    >>> compute_viewed_location(
    ...     np.array([[10.0, 150.0]]), np.array([0.0]), method="ray_cast", env=env
    ... )
    array([[300., 150.]])

    A fixed-distance gaze stops at a nearer wall instead of passing through it:

    >>> compute_viewed_location(
    ...     np.array([[295.0, 150.0]]),
    ...     np.array([0.0]),
    ...     method="fixed_distance",
    ...     view_distance=10.0,
    ...     env=env,
    ... )
    array([[300., 150.]])
    """
    viewed, _at_wall = gaze_points(
        positions,
        headings,
        method=method,
        view_distance=view_distance,
        gaze_offsets=gaze_offsets,
        env=env,
        max_distance=max_distance,
        boundary_source=boundary_source,
    )
    return viewed


def _resolve_gaze_distance(
    method: str,
    view_distance: float | None,
    *,
    units: str | None,
    context: str,
    argument: str,
) -> float | None:
    """The fixed gaze distance in environment units, or None for ray casting.

    Parameters
    ----------
    method : {"ray_cast", "fixed_distance"}
        The gaze model.
    view_distance : float or None
        The caller's distance along the gaze.
    units : str or None
        Length units of the positions, for the message.
    context : str
        Public call for error messages.
    argument : str
        Name of the caller's gaze-model argument (``"method"`` or
        ``"gaze_model"``).

    Raises
    ------
    ValueError
        If ``view_distance`` is given with ``ray_cast``, is missing with
        ``fixed_distance``, or is not a positive finite number.
    """
    if method == "ray_cast":
        if view_distance is not None:
            raise ValueError(
                _format_error(
                    f"{context}: view_distance={view_distance!r} has no meaning "
                    f"with {argument}='ray_cast'.",
                    why="Why: ray casting takes the first wall along the gaze, at "
                    "whatever distance it lies",
                    fix=f"drop view_distance, or pass {argument}='fixed_distance'",
                )
            )
        return None
    if view_distance is None:
        raise ValueError(
            _format_error(
                f"{context}: {argument}='fixed_distance' needs view_distance.",
                why="Why: how far ahead an animal looks depends on the species, "
                "task and arena, and no published value applies by default",
                fix="pass view_distance, the distance along the gaze to the "
                f"viewed point in the units of positions (env.units={units!r}), "
                f"or use {argument}='ray_cast' to take the first wall",
            )
        )
    if not (np.isfinite(view_distance) and view_distance > 0):
        raise ValueError(
            f"{context}: view_distance must be a positive finite distance, "
            f"got {view_distance!r}."
        )
    return float(view_distance)


def gaze_points(
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
    *,
    method: Literal["fixed_distance", "ray_cast"],
    view_distance: float | None,
    gaze_offsets: NDArray[np.float64] | None,
    env: BinningConsumer | None,
    max_distance: float | None,
    boundary_source: BoundarySource,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Viewed points, and which of them lie on a wall.

    The arguments are those of :func:`compute_viewed_location`.

    Returns
    -------
    viewed : NDArray[np.float64], shape (n_time, 2)
        As returned by :func:`compute_viewed_location`.
    at_wall : NDArray[np.bool_], shape (n_time,)
        True where the gaze stopped at a wall: every finite ray-cast point,
        and each fixed-distance point a nearer wall cut short.
    """
    positions = np.asarray(positions, dtype=np.float64)
    headings = np.asarray(headings, dtype=np.float64)

    if method not in ("fixed_distance", "ray_cast"):
        raise ValueError(
            f"method must be 'fixed_distance' or 'ray_cast', got {method!r}."
        )
    if method == "ray_cast" and env is None:
        raise ValueError(
            _format_error(
                "compute_viewed_location(): method='ray_cast' needs env.",
                why="Why: ray casting stops the gaze at the first wall, and the "
                "walls come from the environment",
                fix="pass the env the positions were recorded in, or "
                "method='fixed_distance' with a view_distance",
            )
        )
    _check_pose(
        "compute_viewed_location()",
        positions,
        headings,
        why="each sample's gaze starts at its position and points along its heading.",
    )
    gaze_direction = headings
    if gaze_offsets is not None:
        gaze_offsets = np.asarray(gaze_offsets, dtype=np.float64)
        if gaze_offsets.shape != headings.shape:
            raise ValueError(
                _format_error(
                    "compute_viewed_location() needs one gaze offset per "
                    f"heading, shape {headings.shape}; got {gaze_offsets.shape}.",
                    fix="pass gaze_offsets with the same length as headings",
                )
            )
        gaze_direction = headings + gaze_offsets
    view_distance = _resolve_gaze_distance(
        method,
        view_distance,
        units=getattr(env, "units", None),
        context="compute_viewed_location()",
        argument="method",
    )

    if method == "fixed_distance":
        assert view_distance is not None  # resolved above
        fixed = positions + view_distance * np.column_stack(
            (np.cos(gaze_direction), np.sin(gaze_direction))
        )
        if env is None:
            return fixed, np.zeros(len(positions), dtype=np.bool_)
        # The first wall within view_distance; NaN where none is nearer or
        # the observer is outside the walls.
        indexes = CallIndexes()
        hits = boundary_view(
            env.binning,
            env.arena,
            positions,
            gaze_direction,
            float(view_distance),
            boundary_source,
            indexes,
        )
        at_wall = np.all(np.isfinite(hits), axis=1)
        viewed = np.where(at_wall[:, None], hits, fixed)
        observed = valid_observers(
            env.binning, env.arena, positions, boundary_source, indexes
        )
        viewed[~observed] = np.nan
        return viewed, at_wall

    from neurospatial.geometry import boundary_geometry, search_distance

    assert env is not None
    binning, arena = env.binning, env.arena
    if max_distance is None:
        max_distance = search_distance(
            boundary_geometry(binning, arena, boundary_source), positions
        )
    viewed = boundary_view(
        binning,
        arena,
        positions,
        gaze_direction,
        max_distance,
        boundary_source,
        CallIndexes(),
    )
    return viewed, np.all(np.isfinite(viewed), axis=1)


def compute_viewshed(
    env: BinningConsumer,
    position: NDArray[np.float64],
    heading: float,
    *,
    fov: FieldOfView | float | None = None,
    cue_positions: NDArray[np.float64] | None = None,
    boundary_source: BoundarySource = "physical",
) -> ViewshedResult:
    """Compute visible bins from an observer position.

    A bin is visible when the direction from the observer to the bin center
    lies inside the field of view and the straight line to the center
    crosses no wall. Each bin is tested exactly, so the result does not
    depend on bin size or on the distance to the bin. Cues use the same
    test.

    Parameters
    ----------
    env : Environment
        The environment to analyze.
    position : NDArray[np.float64], shape (2,)
        Observer position in allocentric coordinates (environment units).
    heading : float
        Observer heading in radians (allocentric: 0 = East, +pi/2 = North).
    fov : FieldOfView, float, or None, default None
        Field of view constraint. If FieldOfView, uses its bounds.
        If float, the total angle in radians, in (0, 2*pi], centered on the
        heading. If None, uses full 360 degrees.
    cue_positions : NDArray[np.float64], shape (n_cues, 2), optional
        Positions of cues to check visibility for; must be finite.
    boundary_source : {'physical', 'analysis'}, default='physical'
        Declared physical walls, or explicitly requested analysis-cell boundaries.
        Physical mode requires exact supported wall geometry; missing analysis
        bins do not occlude a physical ray.

    Returns
    -------
    ViewshedResult
        Viewshed analysis result with visible bins and cues.
        For a NaN or infinite ``position`` or ``heading`` (a missing
        observation) its ``visibility_map`` is NaN in every bin. An observer
        outside the region the boundary source covers sees nothing: no
        visible bins or cues and a zero ``visibility_map``.

    Raises
    ------
    ValueError
        If ``position`` is not shape ``(2,)``, ``fov`` is not a valid field
        of view, or ``cue_positions`` is not finite with shape
        ``(n_cues, 2)``.

    Examples
    --------
    >>> import numpy as np
    >>> from shapely.geometry import box
    >>> from neurospatial import Environment
    >>> from neurospatial.ops.visibility import compute_viewshed
    >>> env = Environment.from_polygon(box(0, 0, 100, 100), bin_size=5.0)
    >>> result = compute_viewshed(env, np.array([50.0, 50.0]), heading=0.0)
    >>> result.visibility_fraction  # an open box: every bin is in view
    1.0
    >>> ahead = compute_viewshed(env, np.array([50.0, 50.0]), 0.0, fov=np.pi)
    >>> bool(np.all(env.bin_centers[ahead.visible_bins, 0] > 50.0))  # East
    True
    """
    binning, arena = env.binning, env.arena
    indexes = CallIndexes()
    return _viewshed(
        binning,
        arena,
        position,
        heading,
        fov=_field_of_view(fov),
        cue_positions=_cue_array(cue_positions, "compute_viewshed"),
        boundary_source=boundary_source,
        clear=_sight_lines(binning, arena, boundary_source, indexes),
        indexes=indexes,
    )


def _field_of_view(fov: FieldOfView | float | None) -> FieldOfView | None:
    """Resolve the ``fov`` argument; None means the full circle.

    A number is the total field-of-view angle in radians, in (0, 2*pi].
    """
    if fov is None or isinstance(fov, FieldOfView):
        return fov
    angle = float(fov)
    if not (np.isfinite(angle) and 0 < angle <= 2 * np.pi):
        raise ValueError(
            _format_error(
                f"fov={fov!r} is not a field-of-view angle.",
                why="Why: a number is the total angle in radians that the "
                "animal sees, centered on its heading.",
                fix="pass a total angle in (0, 2*pi] radians (np.pi for "
                "180 degrees), a FieldOfView, or None for the full circle",
            )
        )
    return FieldOfView.symmetric(angle / 2)


def _observer(position: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return one observer position as a float array of shape (2,)."""
    position = np.asarray(position, dtype=np.float64)
    if position.shape != (2,):
        raise ValueError(
            _format_error(
                f"The observer position needs shape (2,); got {position.shape}.",
                fix="pass one (x, y) position; for a trajectory use "
                "compute_viewshed_trajectory",
            )
        )
    return position


def _cue_array(
    cue_positions: NDArray[np.float64] | None, call: str
) -> NDArray[np.float64]:
    """Validate cue positions as finite ``(n_cues, 2)``; None means none.

    ``call`` is the public function, named in the error message.
    """
    if cue_positions is None:
        return np.empty((0, 2))
    cues = np.asarray(cue_positions, dtype=np.float64)
    if cues.size == 0:
        return np.empty((0, 2))
    if cues.ndim != 2 or cues.shape[1] != 2:
        raise ValueError(
            _format_error(
                f"{call}() needs cue_positions of shape (n_cues, 2); got {cues.shape}.",
                fix="pass one (x, y) row per cue",
            )
        )
    if not np.all(np.isfinite(cues)):
        raise ValueError(
            _format_error(
                f"{call}() needs finite cue_positions; row "
                f"{int(np.flatnonzero(~np.isfinite(cues).all(axis=1))[0])} "
                "is NaN or infinite.",
                fix="pass the measured position of every cue, or drop the cues "
                "whose position is unknown",
            )
        )
    return cues


def _sight_lines(
    binning: Binning,
    arena: Arena,
    boundary_source: BoundarySource,
    indexes: LookupIndexes,
) -> Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.bool_]]:
    """Line-of-sight test from one observer to many targets, walls built once.

    Raises the boundary source's errors now. The returned test takes an
    observer that ``valid_observers`` accepts, shape (2,), and finite
    targets, shape (n_targets, 2), and returns a bool per target, as
    ``batch_line_of_sight_clear`` does.
    """
    from neurospatial.geometry import (
        analysis_interior,
        boundary_geometry,
        clear_segments,
        physical_clear_segments,
    )

    walls = boundary_geometry(binning, arena, boundary_source)
    if boundary_source == "physical":
        return lambda observer, targets: physical_clear_segments(
            arena, observer, targets
        )
    interior = analysis_interior(binning, arena, indexes)
    return lambda observer, targets: clear_segments(walls, observer, targets, interior)


def _in_view(
    fov: FieldOfView | None,
    position: NDArray[np.float64],
    heading: float,
    targets: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Egocentric bearing of each target and whether it is inside ``fov``.

    A target at the observer's own position is inside every field of view.

    Returns
    -------
    bearings : ndarray, shape (n_targets,)
        Egocentric bearings in radians, (-pi, pi]: 0 = ahead, +pi/2 = left.
    in_fov : ndarray of bool, shape (n_targets,)
    """
    offsets = targets - position
    bearings = wrap_angle(np.arctan2(offsets[:, 1], offsets[:, 0]) - heading)
    if fov is None:
        return bearings, np.ones(len(targets), dtype=bool)
    in_fov = np.asarray(fov.contains_angle(bearings), dtype=bool)
    return bearings, in_fov | np.all(offsets == 0, axis=1)


def _viewshed(
    binning: Binning,
    arena: Arena,
    position: NDArray[np.float64],
    heading: float,
    *,
    fov: FieldOfView | None,
    cue_positions: NDArray[np.float64],
    boundary_source: BoundarySource,
    clear: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.bool_]],
    indexes: LookupIndexes,
) -> ViewshedResult:
    """Compute one viewshed; see :func:`compute_viewshed`.

    ``fov`` and ``cue_positions`` are already resolved, ``clear`` comes from
    :func:`_sight_lines`, and ``indexes`` keeps the locator's search
    structures across the frames of one trajectory.
    """
    from neurospatial.geometry import valid_observers

    position = _observer(position)
    centers = binning.bins.centers
    n_bins = len(centers)
    observed = bool(np.all(np.isfinite(position)) and np.isfinite(heading))
    if (
        not observed
        or not valid_observers(
            binning, arena, position.reshape(1, 2), boundary_source, indexes
        )[0]
    ):
        # A missing observation is missing, not a view of nothing.
        return ViewshedResult(
            visibility_map=np.zeros(n_bins) if observed else np.full(n_bins, np.nan),
            visible_cues=np.empty(0, dtype=np.intp),
            cue_distances=np.empty(0),
            cue_bearings=np.empty(0),
        )

    # Bin centers and cues get one test: inside the field of view, then a
    # clear straight line from the observer.
    targets = np.vstack((centers, cue_positions))
    bearings, in_fov = _in_view(fov, position, heading, targets)
    candidates = np.flatnonzero(in_fov)
    visible = np.zeros(len(targets), dtype=bool)
    if len(candidates):
        visible[candidates] = clear(position, targets[candidates])
    visible_cues = np.flatnonzero(visible[n_bins:])
    cue_rows = n_bins + visible_cues
    return ViewshedResult(
        visibility_map=visible[:n_bins].astype(np.float64),
        visible_cues=visible_cues,
        cue_distances=np.hypot.reduce(targets[cue_rows] - position, axis=1),
        cue_bearings=bearings[cue_rows],
    )


def _require_observation(
    call: str, position: NDArray[np.float64], heading: float
) -> None:
    """Refuse a missing (non-finite) observer position or heading.

    Parameters
    ----------
    call : str
        The public function, named in the message.
    position : ndarray of float64, shape (2,)
        Observer position.
    heading : float
        Observer heading in radians.

    Raises
    ------
    ValueError
        If ``position`` or ``heading`` is NaN or infinite.
    """
    if np.all(np.isfinite(position)) and np.isfinite(heading):
        return
    raise ValueError(
        _format_error(
            f"{call}() needs an observed position and heading; got position "
            f"{np.asarray(position).tolist()} and heading {heading}.",
            why="Why: a NaN or infinite value is a missing observation (a lost "
            "tracking frame), and reporting nothing visible from it would read "
            "missing data as occlusion.",
            fix="call it only for observed frames, or use "
            "compute_viewshed_trajectory, which keeps missing frames missing",
        )
    )


def compute_view_field(
    env: BinningConsumer,
    position: NDArray[np.float64],
    heading: float,
    *,
    fov: FieldOfView | float | None = None,
    boundary_source: BoundarySource = "physical",
) -> NDArray[np.bool_]:
    """Compute binary visibility mask from observer position.

    A bin is visible when the direction to its center lies inside the field
    of view and the straight line to its center crosses no wall, as in
    :func:`compute_viewshed`.

    Parameters
    ----------
    env : Environment
        The environment to analyze.
    position : NDArray[np.float64], shape (2,)
        Observer position (environment units).
    heading : float
        Observer heading in radians (allocentric: 0 = East, +pi/2 = North).
    fov : FieldOfView, float, or None, default None
        Field of view constraint; a float is the total angle in radians.
        None is the full circle.
    boundary_source : {'physical', 'analysis'}, default='physical'
        Declared physical walls, or explicitly requested analysis-cell boundaries.
        Physical mode requires exact supported wall geometry; missing analysis
        bins do not occlude a physical ray.

    Returns
    -------
    NDArray[np.bool_], shape (n_bins,)
        True for visible bins, False otherwise.

    Raises
    ------
    ValueError
        If ``position`` or ``heading`` is not finite: a missing observation
        has no view field, and an all-False mask would read it as nothing
        visible. ``compute_viewshed_trajectory`` keeps missing frames missing.

    Examples
    --------
    >>> import numpy as np
    >>> from shapely.geometry import box
    >>> from neurospatial import Environment
    >>> from neurospatial.ops.visibility import compute_view_field
    >>> env = Environment.from_polygon(box(0, 0, 100, 100), bin_size=10.0)
    >>> facing_north = compute_view_field(
    ...     env, np.array([50.0, 50.0]), heading=np.pi / 2, fov=np.pi / 2
    ... )
    >>> bool(np.all(env.bin_centers[facing_north, 1] > 50.0))
    True
    """
    _require_observation("compute_view_field", position, heading)
    result = compute_viewshed(
        env, position, heading, fov=fov, boundary_source=boundary_source
    )
    return np.asarray(result.visibility_map == 1.0)


def visible_cues(
    env: BinningConsumer,
    position: NDArray[np.float64],
    heading: float,
    cue_positions: NDArray[np.float64],
    *,
    fov: FieldOfView | float | None = None,
    boundary_source: BoundarySource = "physical",
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
        bins do not occlude a physical ray.

    Returns
    -------
    visible : NDArray[np.bool_], shape (n_cues,)
        True for cues inside the field of view with a clear line of sight.
        All False when the observer is outside the region the boundary
        source covers.
    distances : NDArray[np.float64], shape (n_cues,)
        Distances to all cues (NaN for not visible).
    bearings : NDArray[np.float64], shape (n_cues,)
        Egocentric bearings to all cues (NaN for not visible).

    Raises
    ------
    ValueError
        If ``position``, ``heading`` or a cue position is not finite. A
        missing observation is not evidence that the cues were hidden; check
        only observed frames and keep the others missing.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.ops.visibility import visible_cues

    A box with a wall at x = 50 from y = 20 to y = 80; the animal is west of
    the wall, facing East:

    >>> from shapely.geometry import LineString, box
    >>> from neurospatial.arena import OpenArena
    >>> env = Environment.from_grid_mask(
    ...     np.ones((10, 10), dtype=bool),
    ...     grid_edges=(np.arange(0.0, 101.0, 10.0),) * 2,
    ...     arena=OpenArena(
    ...         box(0, 0, 100, 100), walls=(LineString([(50, 20), (50, 80)]),)
    ...     ),
    ... )
    >>> cues = np.array([[40.0, 50.0], [80.0, 50.0], [10.0, 50.0]])
    >>> visible, distances, bearings = visible_cues(
    ...     env, np.array([25.0, 50.0]), 0.0, cues, fov=np.pi
    ... )
    >>> visible.tolist()  # ahead; behind the wall; behind the animal
    [True, False, False]
    >>> distances.round(1).tolist()
    [15.0, nan, nan]
    """
    position = _observer(position)
    _require_observation("visible_cues", position, heading)
    fov_obj = _field_of_view(fov)
    cue_positions = _cue_array(cue_positions, "visible_cues")
    n_cues = len(cue_positions)

    distances = np.hypot.reduce(cue_positions - position, axis=1)
    bearings, in_fov = _in_view(fov_obj, position, heading, cue_positions)
    visible = np.zeros(n_cues, dtype=bool)
    candidates = np.flatnonzero(in_fov)
    if len(candidates):
        visible[candidates] = batch_line_of_sight_clear(
            env.binning,
            env.arena,
            position,
            cue_positions[candidates],
            boundary_source=boundary_source,
        )

    return (
        visible,
        np.where(visible, distances, np.nan),
        np.where(visible, bearings, np.nan),
    )


def compute_viewshed_trajectory(
    env: BinningConsumer,
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
    *,
    fov: FieldOfView | float | None = None,
    cue_positions: NDArray[np.float64] | None = None,
    boundary_source: BoundarySource = "physical",
) -> list[ViewshedResult]:
    """Compute viewshed at each point along a trajectory.

    Each frame is :func:`compute_viewshed` from that frame's position and
    heading; the walls are built once for the whole trajectory.

    Parameters
    ----------
    env : Environment
        The environment.
    positions : NDArray[np.float64], shape (n_time, 2)
        Positions along trajectory (environment units).
    headings : NDArray[np.float64], shape (n_time,)
        Allocentric headings in radians at each position (0 = East,
        +pi/2 = North).
    fov : FieldOfView, float, or None, default None
        Field of view constraint; a float is the total angle in radians.
        None is the full circle.
    cue_positions : NDArray[np.float64], shape (n_cues, 2), optional
        Cue positions to check; must be finite.
    boundary_source : {'physical', 'analysis'}, default='physical'
        Declared physical walls, or explicitly requested analysis-cell boundaries.
        Physical mode requires exact supported wall geometry; missing analysis
        bins do not occlude a physical ray.

    Returns
    -------
    list of ViewshedResult
        Viewshed result at each trajectory point. A frame with a NaN or
        infinite position or heading is a missing observation: its
        ``visibility_map`` is NaN and its ``visibility_fraction`` is NaN, so
        it is not mistaken for a frame from which nothing was visible.

    Raises
    ------
    ValueError
        If ``positions`` is not ``(n_time, 2)``, ``headings`` is not
        ``(n_time,)``, ``fov`` is not a valid field of view, or
        ``cue_positions`` is not finite with shape ``(n_cues, 2)``.
    """
    positions = np.asarray(positions, dtype=np.float64)
    headings = np.asarray(headings, dtype=np.float64)
    _check_pose(
        "compute_viewshed_trajectory()",
        positions,
        headings,
        why="each sample's gaze starts at its position and points along its heading.",
    )

    binning, arena = env.binning, env.arena
    indexes = CallIndexes()
    fov_obj = _field_of_view(fov)
    cues = _cue_array(cue_positions, "compute_viewshed_trajectory")
    clear = _sight_lines(binning, arena, boundary_source, indexes)
    return [
        _viewshed(
            binning,
            arena,
            positions[i],
            headings[i],
            fov=fov_obj,
            cue_positions=cues,
            boundary_source=boundary_source,
            clear=clear,
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
    boundary_source: BoundarySource = "physical",
    max_gap: float | None = 0.5,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Compute time each bin was visible during trajectory.

    Each sampling interval ``[times[k], times[k + 1])`` adds its duration to
    every bin visible from sample ``k`` (its position and heading), the same
    start-sample rule :meth:`Environment.occupancy` uses. The last sample has
    no interval and adds no time. Visibility is decided per bin as in
    :func:`compute_viewshed`: the bin center is inside the field of view
    with a clear line of sight.

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
        Field of view constraint; a float is the total angle in radians.
        None is the full circle.
    boundary_source : {'physical', 'analysis'}, default='physical'
        Declared physical walls, or explicitly requested analysis-cell boundaries.
        Physical mode requires exact supported wall geometry; missing analysis
        bins do not occlude a physical ray.
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
    ...     env, times, positions, headings, boundary_source="analysis"
    ... )
    >>> round(float(seen.max()), 6)  # 0.2 s before the pause + 0.1 s after
    0.3
    """
    from neurospatial._intervals import (
        as_intervals,
        interval_valid_mask,
        validate_max_gap,
        warn_excluded_intervals,
    )
    from neurospatial._observation import validate_break_before
    from neurospatial._validation import validate_times_positions

    times, positions = validate_times_positions(
        times, positions, call="visibility_occupancy"
    )
    headings = np.asarray(headings, dtype=np.float64)
    n_time = len(times)
    _check_pose(
        "visibility_occupancy()",
        positions,
        headings,
        why="each sample's field of view is centered on its heading.",
    )
    validate_max_gap(max_gap)
    breaks = validate_break_before(break_before, n_time, call="visibility_occupancy")
    resolved_epochs = as_intervals(epochs, name="epochs")

    binning, arena = env.binning, env.arena
    # Resolved before any interval check, so bad options raise even when no
    # interval is usable.
    fov_obj = _field_of_view(fov)
    indexes = CallIndexes()
    clear = _sight_lines(binning, arena, boundary_source, indexes)
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
    no_cues = np.empty((0, 2))
    # Viewsheds are per-frame; compute them only for frames that add time.
    for i in np.flatnonzero(usable & (dt > 0)):
        result = _viewshed(
            binning,
            arena,
            positions[i],
            headings[i],
            fov=fov_obj,
            cue_positions=no_cues,
            boundary_source=boundary_source,
            clear=clear,
            indexes=indexes,
        )
        occupancy[result.visible_bins] += dt[i]

    return occupancy
