"""The one rule that decides which bins a region covers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import shapely
import shapely.geometry.base
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error
from neurospatial.geometry._lookup import locate

if TYPE_CHECKING:
    from neurospatial.geometry._arena import Arena
    from neurospatial.geometry._binning import Binning
    from neurospatial.geometry._locators import LookupIndexes
    from neurospatial.regions.core import Region


# A point exactly on a region's edge belongs to the region it would enter if
# nudged a hair in this direction: north-east, 0.65 rad (about 37°)
# counterclockwise from east, so not along an axis-aligned, diagonal or
# hexagonal (30°, 60°) edge.
TIE_DIRECTION = np.array([np.cos(0.65), np.sin(0.65)])
# The nudge, as a fraction of the polygon's largest coordinate magnitude:
# far above float64 rounding there (2**-52), far below any drawn feature.
_TIE_STEP = 2.0**-30


def polygon_bin_mask(
    polygon: shapely.geometry.base.BaseGeometry,
    centers: NDArray[np.float64],
) -> NDArray[np.bool_]:
    """Select the bins whose centers lie in a polygon, edges half-open.

    A center inside the polygon is in it. A center exactly on its edge is in
    it when nudging the center a hair north-east (along
    :data:`TIE_DIRECTION`) moves it inside. Polygons that share edges
    therefore split the centers on those edges, each to one polygon, and a
    rectangle holds its south and west edges but not its north and east
    edges, as a bin ``[a, b)`` holds ``a``.

    Parameters
    ----------
    polygon : shapely geometry
        Area in the same 2-D coordinates as ``centers``. This function tests
        the centers as given; :func:`region_bin_masks` also tests polar
        centers moved by whole turns.
    centers : ndarray of float64, shape (n_bins, 2)
        Bin centers, or any 2-D points.

    Returns
    -------
    ndarray of bool, shape (n_bins,)
        True for each bin whose center is in the polygon.

    Raises
    ------
    ValueError
        If ``centers`` are not 2-D coordinates.
    """
    if centers.shape[1] != 2:
        raise ValueError(
            _format_error(
                f"A polygon region needs 2-D bins; this environment has "
                f"{centers.shape[1]}-D bin centers.",
                why="Why: a polygon describes an area in a plane.",
                fix="use a point region, or select bins with a boolean mask",
            )
        )
    # The _xy predicates are exact, take raw coordinates and prepare the
    # polygon, so no Point objects are built. For a point, intersects is
    # covers: inside or on the edge.
    x, y = centers[:, 0], centers[:, 1]
    inside = np.asarray(shapely.intersects_xy(polygon, x, y), dtype=np.bool_)
    covered = np.flatnonzero(inside)
    on_edge = covered[~shapely.contains_xy(polygon, x[covered], y[covered])]
    if on_edge.size:
        step = _TIE_STEP * float(np.max(np.abs(polygon.bounds)))
        nudged = centers[on_edge] + step * TIE_DIRECTION
        inside[on_edge] = shapely.contains_xy(polygon, nudged[:, 0], nudged[:, 1])
    return inside


def disc_mask(
    center: NDArray[np.float64], radius: float, points: NDArray[np.float64]
) -> NDArray[np.bool_]:
    """Select the points in a disc, by the half-open rule of polygons.

    A point closer to ``center`` than ``radius`` is in the disc. A point
    exactly on the rim is in it when nudging it a hair along
    :data:`TIE_DIRECTION` moves it inside, as for a polygon's edge: the rim
    facing away from north-east is held, the rest is not.

    Parameters
    ----------
    center : ndarray of float64, shape (2,)
        Center of the disc.
    radius : float
        Radius of the disc, positive, in the units of the points.
    points : ndarray of float64, shape (n_points, 2)
        Points to test.

    Returns
    -------
    ndarray of bool, shape (n_points,)
        True for each point in the disc.
    """
    offsets = points - center
    distance = np.hypot(offsets[:, 0], offsets[:, 1])
    # Moving along TIE_DIRECTION from the rim enters the disc exactly when it
    # points toward the center.
    entering = offsets @ TIE_DIRECTION < 0
    return np.asarray(
        (distance < radius) | ((distance == radius) & entering), dtype=np.bool_
    )


def region_bin_mask(
    region: Region,
    *,
    binning: Binning,
    arena: Arena,
    indexes: LookupIndexes | None = None,
) -> NDArray[np.bool_]:
    """Select the bins a region covers.

    A point region covers the one bin that contains the point (none when the
    point is outside the environment). A polygon region covers the bins whose
    centers lie in it, with a center exactly on the edge decided by the
    half-open rule of :func:`polygon_bin_mask`, so polygons that share edges
    split the bins between them. Every region query of an environment uses
    this rule.

    Both are read in the bins' own coordinates. For polar
    ``(distance, angle)`` bins, angles are compared modulo a full turn: a
    point's angle is wrapped into the angle range as for any position, and a
    polygon covers a bin when the center's angle plus any whole number of
    turns lies inside it, so a wedge across the ±π seam is one polygon.

    Parameters
    ----------
    region : Region
        Point or polygon region in the environment's coordinates.
    binning : Binning
        Bins and their point locator.
    arena : Arena
        Continuous domain; a point outside a known domain covers no bin.
    indexes : LookupIndexes, optional
        Keeps the locator's search structures, usually the environment's
        analysis cache. None builds them for this call only.

    Returns
    -------
    ndarray of bool, shape (n_bins,)
        True for each bin the region covers.

    Raises
    ------
    ValueError
        If a point region has a different number of coordinates than the bins,
        or a polygon region is used with bins that are not 2-D.
    """
    return region_bin_masks(
        [region],
        binning=binning,
        arena=arena,
        indexes=indexes,
    )[:, 0]


def region_bin_masks(
    regions: Sequence[Region],
    *,
    binning: Binning,
    arena: Arena,
    indexes: LookupIndexes | None = None,
) -> NDArray[np.bool_]:
    """Select the bins each of several regions covers, by one rule.

    Each column is :func:`region_bin_mask` of one region.

    Parameters
    ----------
    regions : sequence of Region
        Point or polygon regions in the environment's coordinates.
    binning : Binning
        Bins and their point locator.
    arena : Arena
        Continuous domain; a point outside a known domain covers no bin.
    indexes : LookupIndexes, optional
        Keeps the locator's search structures, usually the environment's
        analysis cache. None builds them for this call only.

    Returns
    -------
    ndarray of bool, shape (n_bins, n_regions)
        True where the bin (row) is covered by the region (column).

    Raises
    ------
    ValueError
        If a point region has a different number of coordinates than the bins,
        or a polygon region is used with bins that are not 2-D.
    """
    n_bins, n_dims = binning.bins.n_bins, binning.bins.n_dims
    centers = binning.bins.centers
    masks = np.zeros((n_bins, len(regions)), dtype=np.bool_)
    for column, region in enumerate(regions):
        if region.kind == "polygon":
            masks[:, column] = polygon_bin_mask(region.data, centers)
            if arena.coordinate_kind == "polar":
                for turns in _polygon_turns(region.data, centers):
                    masks[:, column] |= polygon_bin_mask(
                        region.data,
                        centers + np.array([0.0, turns * 2 * np.pi]),
                    )
            continue
        point = np.asarray(region.data, dtype=np.float64).reshape(1, -1)
        if point.shape[1] != n_dims:
            raise ValueError(
                _format_error(
                    f"Point region {region.name!r} has {point.shape[1]} coordinates, "
                    f"but the bins have {n_dims}.",
                    why="Why: a point region is located in the same coordinates as the bin centers.",
                    fix=f"give the point {n_dims} coordinates",
                )
            )
        call = f"Point region {region.name!r}"
        index = int(locate(binning, arena, point, indexes, call=call)[0])
        if index >= 0:
            masks[index, column] = True
    return masks


def _polygon_turns(
    polygon: shapely.geometry.base.BaseGeometry, centers: NDArray[np.float64]
) -> list[int]:
    """Return the nonzero whole turns that can move a polar center into a polygon.

    Parameters
    ----------
    polygon : shapely geometry
        Area in ``(distance, angle)`` coordinates.
    centers : ndarray of float64, shape (n_bins, 2)
        Polar bin centers.

    Returns
    -------
    list of int
        Each whole number of turns ``k``, other than zero, for which some
        center angle plus ``2πk`` can fall inside the polygon's angle bounds.
    """
    if len(centers) == 0:
        return []
    _, low, _, high = polygon.bounds
    turn = 2 * np.pi
    first = int(np.floor((low - np.max(centers[:, 1])) / turn))
    last = int(np.ceil((high - np.min(centers[:, 1])) / turn))
    return [turns for turns in range(first, last + 1) if turns != 0]
