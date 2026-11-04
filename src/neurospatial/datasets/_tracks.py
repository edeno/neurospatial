"""Where tracked samples lie against a declared track's segments."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

__all__ = ["NearestSegment", "nearest_segment"]


class NearestSegment(NamedTuple):
    """Each point's nearest segment and where the point lies against it.

    Attributes
    ----------
    distance : ndarray, shape (n_points,)
        Distance to the nearest segment, in the points' unit.
    segment : ndarray of int, shape (n_points,)
        Index of that segment; the first of equally near ones.
    along : ndarray, shape (n_points,)
        Where the point projects along that segment, as a share of its length
        from its start, clipped to [0, 1].
    lateral : ndarray, shape (n_points,)
        Signed distance across that segment's line, positive to the left of
        the direction from its start to its end.
    """

    distance: NDArray[np.float64]
    segment: NDArray[np.intp]
    along: NDArray[np.float64]
    lateral: NDArray[np.float64]


def nearest_segment(
    points: NDArray[np.float64],
    starts: NDArray[np.float64],
    ends: NDArray[np.float64],
) -> NearestSegment:
    """Find each point's nearest segment, and its offsets from that segment.

    Parameters
    ----------
    points : ndarray, shape (n_points, 2)
        Finite points.
    starts, ends : ndarray, shape (n_segments, 2)
        Each segment's two ends, of nonzero length.

    Returns
    -------
    NearestSegment
    """
    distance = np.full(len(points), np.inf)
    segment = np.zeros(len(points), dtype=np.intp)
    along = np.zeros(len(points))
    # A loop over segments, vectorized over points: tracks have few segments.
    for k, (start, end) in enumerate(zip(starts, ends, strict=True)):
        step = end - start
        relative = points - start
        t = np.clip(relative @ step / (step @ step), 0.0, 1.0)
        here = np.linalg.norm(relative - t[:, None] * step, axis=1)
        closer = here < distance
        distance[closer] = here[closer]
        segment[closer] = k
        along[closer] = t[closer]
    steps = ends - starts
    lengths = np.array([np.linalg.norm(step) for step in steps])
    relative = points - starts[segment]
    cross = steps[segment, 0] * relative[:, 1] - steps[segment, 1] * relative[:, 0]
    return NearestSegment(distance, segment, along, cross / lengths[segment])
