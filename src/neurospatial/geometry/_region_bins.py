"""The one rule that decides which bins a region covers."""

from __future__ import annotations

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
    from neurospatial.regions.core import Region


def polygon_bin_mask(
    polygon: shapely.geometry.base.BaseGeometry,
    centers: NDArray[np.float64],
    *,
    include_boundary: bool = True,
) -> NDArray[np.bool_]:
    """Select the bins whose centers lie inside a polygon.

    Parameters
    ----------
    polygon : shapely geometry
        Area in the same 2-D coordinates as ``centers``.
    centers : ndarray of float64, shape (n_bins, 2)
        Bin centers.
    include_boundary : bool, default=True
        Count a center exactly on the polygon's edge as inside.

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
) -> NDArray[np.bool_]:
    """Select the bins a region covers.

    A point region covers the one bin that contains the point (none when the
    point is outside the environment). A polygon region covers the bins whose
    centers lie inside it; ``include_boundary`` decides a center exactly on
    the edge. Every region query of an environment uses this rule.

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
    if not isinstance(include_boundary, bool):
        raise TypeError(
            f"include_boundary must be a bool, got {type(include_boundary).__name__}."
        )
    n_bins, n_dims = binning.bins.n_bins, binning.bins.n_dims
    if region.kind == "polygon":
        return polygon_bin_mask(
            region.data, binning.bins.centers, include_boundary=include_boundary
        )
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
    mask = np.zeros(n_bins, dtype=np.bool_)
    index = int(locate(binning, arena, point)[0])
    if index >= 0:
        mask[index] = True
    return mask
