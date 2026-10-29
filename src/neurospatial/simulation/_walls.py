"""Straight wall segments of a 2-D environment, for simulated movement and rays.

The simulators need two things from walls many times over: whether one step
of a walk crosses a wall, and how far a ray travels before it meets one. Both
are a few vectorized line-segment intersections against every wall segment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import shapely
from numpy.typing import NDArray
from shapely.ops import unary_union

from neurospatial.geometry._locators import _coordinate_roundoff

if TYPE_CHECKING:
    from neurospatial.environment import Environment

# Ray-segment intersections computed at once, bounding the temporary arrays
# of one chunk of ray distances to about 4 million elements each.
_INTERSECTIONS_PER_CHUNK = 4_000_000

# Mirrors of one step before a blocked walker waits: 32 turn a step out of a
# corner as sharp as 180 / 32, about 6 degrees.
_MAX_BOUNCES = 32


class Walls:
    """Straight wall segments.

    Parameters
    ----------
    starts, ends : ndarray of float64, shape (n_segments, 2)
        Endpoints of every wall segment; zero-length segments are dropped.
    one_sided : ndarray of bool, shape (n_segments,), optional
        Segments of the arena's outline, open only on their left side (the
        inside of a counter-clockwise outline or a clockwise hole). Other
        segments, such as internal walls, are open on both sides. Default:
        none are one-sided.

    Notes
    -----
    A walker is on a wall when it lies within :attr:`tolerance` of it: the
    rounding a point placed on the wall by arithmetic can carry at the
    arena's coordinates. From there it may leave toward an open side; once
    off, it never comes that close to a wall again.
    """

    def __init__(
        self,
        starts: NDArray[np.float64],
        ends: NDArray[np.float64],
        one_sided: NDArray[np.bool_] | None = None,
    ) -> None:
        keep = np.any(starts != ends, axis=1)
        if one_sided is None:
            one_sided = np.zeros(len(starts), dtype=bool)
        self.starts = starts[keep]
        self.vectors = ends[keep] - starts[keep]
        self.one_sided = one_sided[keep]
        self.lengths = np.linalg.norm(self.vectors, axis=1)
        # Left normals: into the arena for an outline segment.
        self.normals = np.column_stack([-self.vectors[:, 1], self.vectors[:, 0]])
        self.normals /= self.lengths[:, None]
        self.squared_lengths = self.lengths**2
        # A walker inside the walls has coordinates no larger than theirs.
        reference = float(np.max(np.abs(np.concatenate([starts, ends])), initial=0.0))
        self.tolerance = float(
            _coordinate_roundoff(np.array([[reference]]), reference)[0]
        )
        # Distance a walker can travel before it could come within
        # ``tolerance`` of a wall.
        self.clearance = 0.0

    @staticmethod
    def _segments(
        lines: NDArray[np.object_],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Starts and ends of the straight pieces of ``lines``, each (n, 2)."""
        # A zero tolerance drops only vertices on a straight run of wall.
        coords = [
            shapely.get_coordinates(line) for line in shapely.simplify(lines, 0.0)
        ]
        if not coords:
            return np.empty((0, 2)), np.empty((0, 2))
        starts = np.concatenate([c[:-1] for c in coords])
        ends = np.concatenate([c[1:] for c in coords])
        return starts, ends

    @classmethod
    def from_geometry(cls, geometry: shapely.Geometry) -> Walls:
        """The straight segments of lines or polygon boundaries, open on both sides."""
        lines = shapely.get_parts(shapely.line_merge(unary_union(geometry)))
        return cls(*cls._segments(lines))

    @classmethod
    def enclosing(cls, env: Environment) -> Walls | None:
        """What keeps an animal inside ``env``: its outline and internal walls.

        The outline is the arena's polygon when it is known, else the outline
        of the bins; None when neither is planar geometry. Outline segments
        are one-sided, open toward the inside.
        """
        from neurospatial.geometry import materialize_cells

        arena = env.arena
        domain = arena.polygonal_domain() if arena.domain_known else None
        if domain is None:
            try:
                domain = unary_union(list(materialize_cells(env.binning)))
            except ValueError:
                return None
        # Counter-clockwise outlines and clockwise holes: the inside is on
        # the left of every outline segment.
        rings = shapely.get_rings(shapely.get_parts(shapely.orient_polygons(domain)))
        outline = cls._segments(rings)
        inner = cls._segments(
            shapely.get_parts(shapely.line_merge(unary_union(list(arena.walls))))
        )
        one_sided = np.repeat([True, False], [len(outline[0]), len(inner[0])])
        return cls(
            np.concatenate([outline[0], inner[0]]),
            np.concatenate([outline[1], inner[1]]),
            one_sided,
        )

    def distance(self, point: NDArray[np.float64]) -> float:
        """Distance from ``point``, shape (2,), to the nearest wall."""
        offset = point - self.starts
        along = np.clip(
            np.einsum("ij,ij->i", offset, self.vectors) / self.squared_lengths, 0, 1
        )
        nearest = self.starts + along[:, None] * self.vectors
        return float(np.min(np.linalg.norm(point - nearest, axis=1)))

    def _hits(
        self, origins: NDArray[np.float64], rays: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Fraction of each ray travelled to each segment, shape (n_rays, n_segments).

        ``origins`` and ``rays`` have shape (n_rays, 2). ``inf`` where the ray
        (``0 < t``) does not meet the segment.
        """
        vx, vy = self.vectors[:, 0], self.vectors[:, 1]
        denominator = rays[:, :1] * vy - rays[:, 1:] * vx
        ox = self.starts[:, 0] - origins[:, :1]
        oy = self.starts[:, 1] - origins[:, 1:]
        with np.errstate(divide="ignore", invalid="ignore"):
            along_ray = (ox * vy - oy * vx) / denominator
            along_wall = (ox * rays[:, 1:] - oy * rays[:, :1]) / denominator
        hit = (
            (denominator != 0) & (along_ray > 0) & (along_wall >= 0) & (along_wall <= 1)
        )
        return np.where(hit, along_ray, np.inf)

    def first_hit(
        self, position: NDArray[np.float64], step: NDArray[np.float64]
    ) -> int | None:
        """The first wall segment a step meets, or None.

        A step from off a wall meets one when it crosses the wall or ends
        within :attr:`tolerance` of it. A step from on a wall meets it when
        the wall is one-sided and the step heads out of the open side; a
        step along a wall or away from it does not.
        """
        moved = position + step
        before = np.einsum("ij,ij->i", position - self.starts, self.normals)
        after = np.einsum("ij,ij->i", moved - self.starts, self.normals)
        on = np.abs(before) <= self.tolerance
        # Fraction of the step travelled to the wall's line.
        with np.errstate(divide="ignore", invalid="ignore"):
            reach = np.where(on, 0.0, before / (before - after))
        contact = position + np.clip(reach, 0.0, 1.0)[:, None] * step
        along = (
            np.einsum("ij,ij->i", contact - self.starts, self.vectors)
            / self.squared_lengths
        )
        slack = self.tolerance / self.lengths
        within = (along >= -slack) & (along <= 1 + slack)
        meets = np.where(
            on,
            self.one_sided & (after < before),
            np.sign(before) * after <= self.tolerance,
        )
        reach = np.where(within & meets, reach, np.inf)
        wall = int(np.argmin(reach))
        return wall if np.isfinite(reach[wall]) else None

    def ray_distances(
        self, origins: NDArray[np.float64], angles: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Distance from each origin to the first wall along each direction.

        Parameters
        ----------
        origins : ndarray of float64, shape (n_points, 2)
        angles : ndarray of float64, shape (n_angles,)
            Ray directions in radians (0 = East, +pi/2 = North).

        Returns
        -------
        ndarray of float64, shape (n_points, n_angles)
            ``inf`` where no wall lies along the ray.
        """
        units = np.column_stack([np.cos(angles), np.sin(angles)])
        n_angles = len(angles)
        out = np.empty((len(origins), n_angles))
        per_chunk = max(1, _INTERSECTIONS_PER_CHUNK // (n_angles * len(self.starts)))
        for start in range(0, len(origins), per_chunk):
            chunk = origins[start : start + per_chunk]
            starts = np.repeat(chunk, n_angles, axis=0)
            rays = np.tile(units, (len(chunk), 1))
            out[start : start + len(chunk)] = (
                self._hits(starts, rays).min(axis=1).reshape(len(chunk), n_angles)
            )
        return out

    def move(
        self,
        position: NDArray[np.float64],
        step: NDArray[np.float64],
        unit: NDArray[np.float64],
        boundary_mode: str,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Take one step without crossing a wall; return position and heading.

        A step that would cross a wall is mirrored about it, and again about
        each wall the mirrored step meets, up to :data:`_MAX_BOUNCES` times:
        in a corner of angle ``a`` the mirrors alternate between its two
        sides and turn the step by ``2 a`` per pair, so a step heading into
        a corner as sharp as about 6 degrees turns out of it. A walker still
        blocked, or one in ``"stop"`` mode, stays put. A walker starting on a
        wall leaves it toward an open side.
        """
        length = float(np.linalg.norm(step))
        if length < self.clearance:
            self.clearance -= length
            return position + step, unit
        for bounce in range(_MAX_BOUNCES + 1):
            wall = self.first_hit(position, step)
            if wall is None:
                moved = position + step
                self._set_clearance(moved)
                return moved, unit
            if boundary_mode == "stop" or bounce == _MAX_BOUNCES:
                break
            # Mirror about the wall: keep the component along it, reverse the
            # component into it.
            normal = self.normals[wall]
            step = step - 2 * (step @ normal) * normal
            unit = unit - 2 * (unit @ normal) * normal
        self._set_clearance(position)
        return position, unit

    def _set_clearance(self, position: NDArray[np.float64]) -> None:
        # Keep unchecked steps out of the band within ``tolerance`` of a
        # wall, allowing as much again for rounding in the distance.
        self.clearance = self.distance(position) - 2 * self.tolerance
