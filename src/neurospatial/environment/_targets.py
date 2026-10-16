"""Resolve the explicit ``bins=``, ``points=`` and ``region=`` query targets."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurospatial._exceptions import (
    BinIndexOutOfRangeError,
    RegionNotFoundError,
    _format_error,
)
from neurospatial.geometry._coordinates import CartesianSystem, EgocentricPolarSystem
from neurospatial.geometry._lookup import locate
from neurospatial.geometry._region_bins import region_bin_mask

if TYPE_CHECKING:
    from neurospatial.geometry._arena import Arena
    from neurospatial.geometry._binning import Binning
    from neurospatial.regions import Regions

_COUNT_WORDS = {1: "one bin", 2: "two bins, a start and an end"}


def coordinate_hint(
    coordinates: CartesianSystem | EgocentricPolarSystem, *, takes_points: bool
) -> str:
    """Say how to give a coordinate instead of a bin index, for this geometry.

    Parameters
    ----------
    coordinates : CartesianSystem or EgocentricPolarSystem
        The environment's coordinate descriptor.
    takes_points : bool
        Whether the refused call itself takes ``points=``.

    Returns
    -------
    str
        The ``Fix:`` text for a float or boolean bin target.
    """
    polar = isinstance(coordinates, EgocentricPolarSystem)
    rows = "(distance, angle) rows" if polar else "coordinate rows"
    if takes_points:
        return f"pass {rows} with points= instead, one row per point"
    lookup = "env.nearest_bin(points)" if polar else "env.bin_at(points)"
    return (
        f"pass integer bin indices; to find the bins of {rows}, use {lookup}, "
        "or give the rows with points= to a query such as env.distance_to"
    )


def validate_bin_indices(
    bins: object, *, n_bins: int, operation: str, fix: str
) -> NDArray[np.intp]:
    """Check that ``bins`` holds integer active-bin indices.

    Parameters
    ----------
    bins : object
        An int, or an integer array of shape ``(n,)``.
    n_bins : int
        Number of active bins; valid indices are ``[0, n_bins)``.
    operation : str
        The call named in errors, e.g. ``"distance_to()"``.
    fix : str
        What to do instead when ``bins`` holds floats or booleans.

    Returns
    -------
    ndarray of intp, shape (n,)
        The indices, flattened.

    Raises
    ------
    TypeError
        If ``bins`` is a bool, holds booleans, floats or non-numbers.
    ValueError
        If ``bins`` is empty or has more than one dimension.
    BinIndexOutOfRangeError
        If an index is outside ``[0, n_bins)``.
    """
    array = np.asarray(bins)
    if array.dtype.kind == "b":
        raise TypeError(
            _format_error(
                f"{operation} got booleans in bins=; bins= takes integer bin indices.",
                why="Why: True and False would be read as bins 1 and 0.",
                fix=f"turn a boolean mask into indices with np.flatnonzero(mask), or {fix}",
            )
        )
    if array.size == 0:
        raise ValueError(
            _format_error(
                f"{operation} got an empty bins=.",
                why="Why: a query needs at least one bin to start from.",
                fix="pass at least one integer bin index",
            )
        )
    if array.dtype.kind not in "iu":
        raise TypeError(
            _format_error(
                f"{operation} got {array.dtype} values in bins=; bins= takes "
                "integer bin indices.",
                why="Why: a coordinate cast to an integer names an unrelated bin.",
                fix=fix,
            )
        )
    if array.ndim > 1:
        raise ValueError(
            _format_error(
                f"{operation} got bins= with shape {array.shape}.",
                why="Why: bins= is a list of bin indices.",
                fix="pass an int or a 1-D integer array of bin indices",
            )
        )
    indices = array.reshape(-1)
    out_of_range = (indices < 0) | (indices >= n_bins)
    if np.any(out_of_range):
        raise BinIndexOutOfRangeError(int(indices[out_of_range][0]), n_bins=n_bins)
    return indices.astype(np.intp, copy=False)


def locate_points(
    points: ArrayLike, *, binning: Binning, arena: Arena, operation: str
) -> NDArray[np.intp]:
    """Map explicit query points to the bins that contain them.

    Parameters
    ----------
    points : array-like, shape (n_points, n_dims)
        Finite coordinates in the same coordinates as ``bin_centers``.
    binning : Binning
        Bins and their point locator.
    arena : Arena
        Continuous domain; points outside a known domain have no bin.
    operation : str
        The call named in errors.

    Returns
    -------
    ndarray of intp, shape (n_points,)
        The bin containing each point.

    Raises
    ------
    TypeError
        If ``points`` is not numeric.
    ValueError
        If ``points`` has the wrong shape, is empty, holds NaN or inf, or a
        point lies outside every bin.
    """
    n_dims = binning.bins.n_dims
    try:
        array = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise TypeError(
            _format_error(
                f"{operation} got points= that are not numbers.",
                fix=f"pass coordinates as an array of shape (n_points, {n_dims})",
            )
        ) from error
    if array.ndim != 2 or array.shape[1] != n_dims:
        example = {
            1: "points=np.reshape(points, (-1, 1))",
            2: "points=[[x, y]]",
        }.get(n_dims, "points=np.atleast_2d(point)")
        raise ValueError(
            _format_error(
                f"{operation} got points= with shape {array.shape}; expected "
                f"(n_points, {n_dims}) for this {n_dims}-D environment.",
                why="Why: each row is one point, in the same coordinates as env.bin_centers.",
                fix=f"pass one row per point, for example {example}",
            )
        )
    if array.shape[0] == 0:
        raise ValueError(
            _format_error(
                f"{operation} got an empty points=.",
                why="Why: a query needs at least one point to start from.",
                fix=f"pass at least one point of shape (1, {n_dims})",
            )
        )
    missing = ~np.isfinite(array).all(axis=1)
    if np.any(missing):
        raise ValueError(
            _format_error(
                f"{operation} got {int(missing.sum())} point(s) with NaN or inf "
                f"coordinates, first at row {int(np.flatnonzero(missing)[0])}.",
                why="Why: a missing coordinate is not a location, so it has no bin.",
                fix="drop the rows with missing coordinates before the query",
            )
        )
    indices = locate(binning, arena, array)
    outside = indices < 0
    if np.any(outside):
        first = array[outside][0].tolist()
        raise ValueError(
            _format_error(
                f"{operation} got {int(outside.sum())} point(s) outside the "
                f"environment, first {first} at row {int(np.flatnonzero(outside)[0])}.",
                why="Why: a point outside every bin has no bin to measure from.",
                fix="pass points inside the environment, in the same coordinates "
                "and units as env.bin_centers",
            )
        )
    return indices


def resolve_targets(
    operation: str,
    *,
    bins: object,
    points: ArrayLike | None,
    region: str | None = None,
    binning: Binning,
    arena: Arena,
    coordinates: CartesianSystem | EgocentricPolarSystem,
    regions: Regions | None = None,
    count: int | None = None,
) -> NDArray[np.intp]:
    """Resolve exactly one of ``bins=``, ``points=`` or ``region=`` to bins.

    Parameters
    ----------
    operation : str
        The call named in errors, e.g. ``"distance_to()"``.
    bins : object
        Integer bin indices, or None.
    points : array-like or None
        Coordinates of shape ``(n_points, n_dims)``, or None.
    region : str or None, optional
        Name of a region in ``regions``, or None.
    binning : Binning
        Bins and their point locator.
    arena : Arena
        Continuous domain of the bins.
    coordinates : CartesianSystem or EgocentricPolarSystem
        Coordinate meaning of the bins, for geometry-specific hints.
    regions : Regions or None, optional
        Named regions; None when the query takes no ``region=``.
    count : int or None, optional
        Exact number of targets the query needs; None accepts any number.

    Returns
    -------
    ndarray of intp, shape (n_targets,)
        Active-bin indices, in the order given.

    Raises
    ------
    TypeError
        If none or more than one target keyword is given, or a target has the
        wrong type.
    ValueError
        If a target is empty, malformed or outside the environment, a region
        covers no bin, or the number of targets is not ``count``.
    RegionNotFoundError
        If ``region`` is not a name in ``regions``.
    """
    keywords = (
        ("bins=", "points=") if regions is None else ("bins=", "points=", "region=")
    )
    given = [
        name
        for name, value in zip(keywords, (bins, points, region), strict=False)
        if value is not None
    ]
    if len(given) != 1:
        listed = (
            " or ".join(keywords) if regions is None else "bins=, points= or region="
        )
        got = "none" if not given else " and ".join(given)
        raise TypeError(
            _format_error(
                f"{operation} needs exactly one target, {listed}; got {got}.",
                why="Why: an integer bin index and a coordinate cannot be told apart "
                "from their values, so each kind has its own keyword.",
                fix="pass integer bin indices with bins=, coordinates with points= "
                "(one row per point)"
                + ("" if regions is None else ", or a region name with region="),
            )
        )
    if bins is not None:
        resolved = validate_bin_indices(
            bins,
            n_bins=binning.bins.n_bins,
            operation=operation,
            fix=coordinate_hint(coordinates, takes_points=True),
        )
    elif points is not None:
        resolved = locate_points(
            points, binning=binning, arena=arena, operation=operation
        )
    else:
        resolved = _region_targets(
            region, regions, binning=binning, arena=arena, operation=operation
        )
    if count is not None and resolved.size != count:
        raise ValueError(
            _format_error(
                f"{operation} takes {_COUNT_WORDS[count]}; got {resolved.size}.",
                why="Why: this query starts from a fixed number of bins.",
                fix=f"pass {count} bin index(es) with bins=, or {count} point(s), "
                "one row each, with points=",
            )
        )
    return resolved


def _region_targets(
    region: object,
    regions: Regions | None,
    *,
    binning: Binning,
    arena: Arena,
    operation: str,
) -> NDArray[np.intp]:
    """The bins a named region covers, refusing a region that covers none."""
    if regions is None or not isinstance(region, str):
        raise TypeError(
            f"{operation} takes a region name in region=, got {type(region).__name__}."
        )
    if region not in regions:
        raise RegionNotFoundError(
            region, available=list(regions.keys()), argument="region"
        )
    resolved = np.flatnonzero(
        region_bin_mask(regions[region], binning=binning, arena=arena)
    ).astype(np.intp, copy=False)
    if resolved.size == 0:
        raise ValueError(
            _format_error(
                f"Region {region!r} covers no bins, so {operation} has no target.",
                why="Why: a point region outside the environment, or a polygon "
                "that contains no bin center, selects no bins.",
                fix="move the region inside the environment, or enlarge the "
                "polygon until it contains a bin center",
            )
        )
    return resolved
