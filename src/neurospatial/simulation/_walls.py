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

if TYPE_CHECKING:
    from neurospatial.environment import Environment

# Ray-segment intersections computed at once, bounding the temporary arrays
# of one chunk of ray distances to about 4 million elements each.
_INTERSECTIONS_PER_CHUNK = 4_000_000


class Walls:
    """Straight wall segments.

    Parameters
    ----------
    starts, ends : ndarray of float64, shape (n_segments, 2)
        Endpoints of every wall segment; zero-length segments are dropped.
    """

    def __init__(self, starts: NDArray[np.float64], ends: NDArray[np.float64]) -> None:
        keep = np.any(starts != ends, axis=1)
        self.starts = starts[keep]
        self.vectors = ends[keep] - starts[keep]
        lengths = np.linalg.norm(self.vectors, axis=1)
        self.normals = np.column_stack([-self.vectors[:, 1], self.vectors[:, 0]])
        self.normals /= lengths[:, None]
        self.squared_lengths = lengths**2
        # Distance a walker can travel before it could reach a wall.
        self.clearance = 0.0

    @classmethod
    def from_geometry(cls, geometry: shapely.Geometry) -> Walls:
        """The straight segments of lines or polygon boundaries."""
        lines = shapely.get_parts(shapely.line_merge(unary_union(geometry)))
        # A zero tolerance drops only vertices on a straight run of wall.
        lines = shapely.simplify(lines, 0.0)
        coords = [shapely.get_coordinates(line) for line in lines]
        starts = np.concatenate([c[:-1] for c in coords])
        ends = np.concatenate([c[1:] for c in coords])
        return cls(starts, ends)

    @classmethod
    def enclosing(cls, env: Environment) -> Walls | None:
        """What keeps an animal inside ``env``: its outline and internal walls.

        The outline is the arena's polygon when it is known, else the outline
        of the bins; None when neither is planar geometry.
        """
        from neurospatial.geometry import materialize_cells

        arena = env.arena
        domain = arena.polygonal_domain() if arena.domain_known else None
        if domain is None:
            try:
                domain = unary_union(list(materialize_cells(env.binning)))
            except ValueError:
                return None
        return cls.from_geometry(unary_union([domain.boundary, *arena.walls]))

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
        """The first wall segment a step reaches, or None."""
        along = self._hits(position[None, :], step[None, :])[0]
        wall = int(np.argmin(along))
        return wall if along[wall] <= 1 else None

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

        A step that would cross a wall is mirrored about it, up to twice (a
        corner); a walker still blocked, or one in ``"stop"`` mode, stays put.
        """
        length = float(np.linalg.norm(step))
        if length < self.clearance:
            self.clearance -= length
            return position + step, unit
        for _bounce in range(3):
            wall = self.first_hit(position, step)
            if wall is None:
                moved = position + step
                self.clearance = self.distance(moved)
                return moved, unit
            if boundary_mode == "stop":
                break
            # Mirror about the wall: keep the component along it, reverse the
            # component into it.
            normal = self.normals[wall]
            step = step - 2 * (step @ normal) * normal
            unit = unit - 2 * (unit @ normal) * normal
        self.clearance = self.distance(position)
        return position, unit
