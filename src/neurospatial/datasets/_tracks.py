"""Where tracked samples lie against a declared track's segments."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

__all__ = ["NearestSegment", "nearest_segment", "placement_tolerance"]


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


def placement_tolerance(
    distance: NDArray[np.float64],
    quantile: float,
    *,
    floor: float,
    measured: str,
    reason: str,
) -> tuple[float, float | None, str]:
    """A track's ``track_width`` from its tracked samples' distance to it, and why.

    Twice ``quantile`` of ``distance``, and at least ``floor``, so that about
    ``1 - quantile`` of the samples fall in no bin by construction.

    Parameters
    ----------
    distance : ndarray, shape (n_tracked,)
        Each tracked sample's distance from the track, cm.
    quantile : float
        The quantile of ``distance`` that half the width is.
    floor : float
        The narrowest width, cm: one bin, or a stated physical width.
    measured : str
        What ``distance`` is, the evidence's opening words, for example
        ``"the tracked point's distance from the track"``.
    reason : str
        Why that quantile, one or more sentences.

    Returns
    -------
    width : float
        ``track_width``, cm.
    width_quantile : float or None
        ``quantile`` when it set the width; None when ``floor`` did.
    evidence : str
        The inferred ``Conversion``'s evidence.
    """
    reach = float(np.quantile(distance, quantile))
    width = max(2 * reach, floor)
    evidence = (
        f"{measured}: median {np.median(distance):.2f} cm, "
        f"{quantile:.0%} quantile {reach:.3g} cm; "
        f"{np.mean(distance > width / 2):.1%} of tracked samples lie beyond the "
        f"tolerance, by construction, and fall in no bin. {reason} track_width "
        "decides only which samples are placed, not the bins, their lengths or "
        "connectivity"
    )
    return width, quantile if width == 2 * reach else None, evidence
