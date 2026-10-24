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


def polygon_bin_mask(
    polygon: shapely.geometry.base.BaseGeometry,
    centers: NDArray[np.float64],
    *,
    include_boundary: bool = True,
    points: NDArray[np.object_] | None = None,
) -> NDArray[np.bool_]:
    """Select the bins whose centers lie inside a polygon.

    Parameters
    ----------
    polygon : shapely geometry
        Area in the same 2-D coordinates as ``centers``. This function tests
        the centers as given; :func:`region_bin_masks` also tests polar
        centers moved by whole turns.
    centers : ndarray of float64, shape (n_bins, 2)
        Bin centers.
    include_boundary : bool, default=True
        Count a center exactly on the polygon's edge as inside.
    points : ndarray of shapely Point, shape (n_bins,), optional
        ``shapely.points(centers)``, built once by a caller that tests several
        polygons. None builds them for this call.

    Returns
    -------
    ndarray of bool, shape (n_bins,)
        True for each bin whose center is inside.

    Raises
    ------
    TypeError
        If ``include_boundary`` is not a bool.
    ValueError
        If ``centers`` are not 2-D coordinates.
    """
    if not isinstance(include_boundary, bool):
        raise TypeError(
            f"include_boundary must be a bool, got {type(include_boundary).__name__}."
        )
    if centers.shape[1] != 2:
        raise ValueError(
            _format_error(
                f"A polygon region needs 2-D bins; this environment has "
                f"{centers.shape[1]}-D bin centers.",
                why="Why: a polygon describes an area in a plane.",
                fix="use a point region, or select bins with a boolean mask",
            )
        )
    if points is None:
        points = shapely.points(centers)
    covered = (
        shapely.covers(polygon, points)
        if include_boundary
        else shapely.contains(polygon, points)
    )
    return np.asarray(covered, dtype=np.bool_)


def region_bin_mask(
    region: Region,
    *,
    binning: Binning,
    arena: Arena,
    include_boundary: bool = True,
    indexes: LookupIndexes | None = None,
) -> NDArray[np.bool_]:
    """Select the bins a region covers.

    A point region covers the one bin that contains the point (none when the
    point is outside the environment). A polygon region covers the bins whose
    centers lie inside it; ``include_boundary`` decides a center exactly on
    the edge. Every region query of an environment uses this rule.

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
    include_boundary : bool, default=True
        For polygon regions, count a center on the edge as inside. Point
        regions ignore it.
    indexes : LookupIndexes, optional
        Keeps the locator's search structures, usually the environment's
        analysis cache. None builds them for this call only.

    Returns
    -------
    ndarray of bool, shape (n_bins,)
        True for each bin the region covers.

    Raises
    ------
    TypeError
        If ``include_boundary`` is not a bool.
    ValueError
        If a point region has a different number of coordinates than the bins,
        or a polygon region is used with bins that are not 2-D.
    """
    return region_bin_masks(
        [region],
        binning=binning,
        arena=arena,
        include_boundary=include_boundary,
        indexes=indexes,
    )[:, 0]


def region_bin_masks(
    regions: Sequence[Region],
    *,
    binning: Binning,
    arena: Arena,
    include_boundary: bool = True,
    indexes: LookupIndexes | None = None,
) -> NDArray[np.bool_]:
    """Select the bins each of several regions covers, by one rule.

    Each column is :func:`region_bin_mask` of one region. The bin centers'
    shapely points are built once for all polygon regions.

    Parameters
    ----------
    regions : sequence of Region
        Point or polygon regions in the environment's coordinates.
    binning : Binning
        Bins and their point locator.
    arena : Arena
        Continuous domain; a point outside a known domain covers no bin.
    include_boundary : bool, default=True
        For polygon regions, count a center on the edge as inside. Point
        regions ignore it.
    indexes : LookupIndexes, optional
        Keeps the locator's search structures, usually the environment's
        analysis cache. None builds them for this call only.

    Returns
    -------
    ndarray of bool, shape (n_bins, n_regions)
        True where the bin (row) is covered by the region (column).

    Raises
    ------
    TypeError
        If ``include_boundary`` is not a bool.
    ValueError
        If a point region has a different number of coordinates than the bins,
        or a polygon region is used with bins that are not 2-D.
    """
    if not isinstance(include_boundary, bool):
        raise TypeError(
            f"include_boundary must be a bool, got {type(include_boundary).__name__}."
        )
    n_bins, n_dims = binning.bins.n_bins, binning.bins.n_dims
    centers = binning.bins.centers
    masks = np.zeros((n_bins, len(regions)), dtype=np.bool_)
    points: NDArray[np.object_] | None = None
    for column, region in enumerate(regions):
        if region.kind == "polygon":
            if points is None and n_dims == 2:
                points = shapely.points(centers)
            masks[:, column] = polygon_bin_mask(
                region.data, centers, include_boundary=include_boundary, points=points
            )
            if arena.coordinate_kind == "polar":
                for turns in _polygon_turns(region.data, centers):
                    masks[:, column] |= polygon_bin_mask(
                        region.data,
                        centers + np.array([0.0, turns * 2 * np.pi]),
                        include_boundary=include_boundary,
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
