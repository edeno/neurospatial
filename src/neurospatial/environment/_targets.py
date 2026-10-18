"""Resolve the explicit ``bins=``, ``points=`` and ``region=`` query targets."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurospatial._exceptions import (
    BinIndexOutOfRangeError,
    RegionNotFoundError,
    _format_error,
)
from neurospatial.geometry import (
    CartesianSystem,
    EgocentricPolarSystem,
    locate,
    region_bin_mask,
    require_position_columns,
)

if TYPE_CHECKING:
    from neurospatial._keyword_hint import Translate
    from neurospatial.geometry import Arena, Binning, LookupIndexes
    from neurospatial.regions import Regions

_COUNT_WORDS = {1: "one bin", 2: "two bins, a start and an end"}
_COUNT_FIXES = {
    1: "pass one bin, for example bins=3, or one point, for example points=[[x, y]]",
    2: "pass two bins, a start and an end, as bins=[start, end], or two points, "
    "one row each, as points=[[x0, y0], [x1, y1]]",
}


TARGET_WHY = (
    "Why: a query target is keyword-only and named by its kind, bins= for bin "
    "indices or points= for coordinates, so a value passed by position or under "
    "an earlier name cannot be matched to one."
)
REGION_TARGET_WHY = TARGET_WHY.replace(
    "bins= for bin indices or points= for coordinates",
    "bins= for bin indices, points= for coordinates or region= for a region name",
)


def former_targets(count: int, *, region: bool = False) -> Translate:
    """Map the bins or points an earlier version took by position to keywords.

    Earlier versions took a query's target by position and decided from its
    type whether it was a bin index or a coordinate. The returned function
    makes the same decision, for the suggested call only: integers become
    ``bins=``, real numbers ``points=`` rows and, when ``region`` is True, a
    string becomes ``region=``.

    Parameters
    ----------
    count : int
        Number of separate target arguments the earlier version took: 1 for
        one bin, 2 for a start and an end.
    region : bool, default=False
        Whether the query also takes ``region=``.

    Returns
    -------
    Translate
        The ``translate`` argument of :func:`~neurospatial._keyword_hint.keyword_hint`.
    """
    kinds = (
        "pass bin indices as bins=, coordinates as points= (one row per point) "
        "or a region name as region="
        if region
        else "pass bin indices as bins= and coordinates as points= (one row per point)"
    )

    def translate(values: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
        given = list(values.values())
        if region and len(given) == 1 and isinstance(given[0], str):
            return kinds, {"region": given[0]}
        arrays = [np.asarray(value) for value in given]
        if all(array.dtype.kind in "iu" for array in arrays):
            bins = arrays[0].tolist() if count == 1 else [a.tolist() for a in arrays]
            return kinds, {"bins": bins}
        if all(array.dtype.kind == "f" for array in arrays):
            rows = [row for a in arrays for row in np.atleast_2d(a).tolist()]
            return kinds, {"points": rows}
        return f"{kinds}; give every target the same way", None

    return translate


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
    return (
        f"pass integer bin indices; to find the bins of {rows}, use env.bin_at(points), "
        "or give the rows with points= to a query such as env.distance_to"
    )


def validate_bin_indices(
    bins: object,
    *,
    n_bins: int,
    coordinates: CartesianSystem | EgocentricPolarSystem,
    operation: str,
    takes_points: bool,
    argument: str = "bins",
    allow_missing: bool = False,
) -> NDArray[np.intp]:
    """Check that ``bins`` holds integer active-bin indices.

    Parameters
    ----------
    bins : object
        An int, or an integer array of shape ``(n,)``.
    n_bins : int
        Number of active bins; valid indices are ``[0, n_bins)``.
    coordinates : CartesianSystem or EgocentricPolarSystem
        The environment's coordinate descriptor; the fixes name lookups that
        work on this geometry.
    operation : str
        The call named in errors, e.g. ``"distance_to()"``.
    takes_points : bool
        Whether the refused call itself takes ``points=``.
    argument : str, default="bins"
        The parameter name shown in errors, e.g. ``"goal_bins"``.
    allow_missing : bool, default=False
        Accept per-sample targets: ``-1`` marks a sample with no target (as
        ``bin_at`` marks a position outside), and an empty array is an empty
        recording.

    Returns
    -------
    ndarray of intp, shape (n,)
        The indices, flattened.

    Raises
    ------
    TypeError
        If ``bins`` is a bool, holds booleans, floats or non-numbers.
    ValueError
        If ``bins`` is empty (unless ``allow_missing``) or has more than one
        dimension.
    BinIndexOutOfRangeError
        If an index is outside ``[0, n_bins)`` (``-1`` is allowed with
        ``allow_missing``).
    """
    fix = coordinate_hint(coordinates, takes_points=takes_points)
    array = np.asarray(bins)
    if array.dtype.kind == "b":
        raise TypeError(
            _format_error(
                f"{operation} got booleans in {argument}=; {argument}= takes "
                "integer bin indices.",
                why="Why: True and False would be read as bins 1 and 0.",
                fix=f"turn a boolean mask into indices with np.flatnonzero(mask), or {fix}",
            )
        )
    if array.size == 0:
        if allow_missing:
            return np.empty(0, dtype=np.intp)
        raise ValueError(
            _format_error(
                f"{operation} got an empty {argument}=.",
                why="Why: a query needs at least one bin to start from.",
                fix="pass at least one integer bin index",
            )
        )
    if array.dtype.kind not in "iu":
        raise TypeError(
            _format_error(
                f"{operation} got {array.dtype} values in {argument}=; "
                f"{argument}= takes integer bin indices.",
                why="Why: a coordinate cast to an integer names an unrelated bin.",
                fix=fix,
            )
        )
    if array.ndim > 1:
        raise ValueError(
            _format_error(
                f"{operation} got {argument}= with shape {array.shape}.",
                why=f"Why: {argument}= is a list of bin indices.",
                fix="pass an int or a 1-D integer array of bin indices",
            )
        )
    indices = array.reshape(-1)
    out_of_range = (indices < 0) | (indices >= n_bins)
    if allow_missing:
        out_of_range &= indices != -1
    if np.any(out_of_range):
        raise BinIndexOutOfRangeError(
            int(indices[out_of_range][0]),
            n_bins=n_bins,
            coordinate_kind=(
                "polar"
                if isinstance(coordinates, EgocentricPolarSystem)
                else "cartesian"
            ),
        )
    return indices.astype(np.intp, copy=False)


def _validate_target_bins(
    bins: object,
    *,
    n_bins: int,
    coordinates: CartesianSystem | EgocentricPolarSystem,
    operation: str,
    argument: str,
    allow_missing: bool = False,
) -> NDArray[np.intp]:
    """Validate an analysis function's bin-target argument.

    The same check as the environment queries' ``bins=``, with a fix that
    names the lookup for this geometry, since the function takes no
    ``points=`` of its own.

    Parameters
    ----------
    bins : object
        An int, or an integer array of shape ``(n,)``.
    n_bins : int
        Number of active bins.
    coordinates : CartesianSystem or EgocentricPolarSystem
        The environment's coordinate descriptor, for the fix.
    operation : str
        The call named in errors, e.g. ``"path_progress()"``.
    argument : str
        The parameter name, e.g. ``"goal_bins"``.
    allow_missing : bool, default=False
        Accept ``-1`` (no target for that sample) and an empty array.

    Returns
    -------
    ndarray of intp, shape (n,)
        The indices, flattened.
    """
    return validate_bin_indices(
        bins,
        n_bins=n_bins,
        coordinates=coordinates,
        operation=operation,
        takes_points=False,
        argument=argument,
        allow_missing=allow_missing,
    )


def _validate_sample_targets(
    bins: object,
    *,
    n_samples: int,
    n_bins: int,
    coordinates: CartesianSystem | EgocentricPolarSystem,
    operation: str,
    argument: str,
) -> tuple[NDArray[np.intp], bool]:
    """Validate a target that is either per sample or one bin for every sample.

    Only an array of shape ``(n_samples,)`` is per sample, and only there does
    ``-1`` mark a sample with no target. Any other input, including the
    shape-``(1,)`` result of ``env.bin_at([[x, y]])``, is one target and must be
    a bin: ``-1`` (the outside marker) is refused.

    Parameters
    ----------
    bins : object
        An int, a one-element array, or an integer array of shape
        ``(n_samples,)``.
    n_samples : int
        Number of samples the targets align with.
    n_bins : int
        Number of active bins.
    coordinates : CartesianSystem or EgocentricPolarSystem
        The environment's coordinate descriptor, for the fix.
    operation : str
        The call named in errors, e.g. ``"distance_to_region()"``.
    argument : str
        The parameter name, e.g. ``"target_bins"``.

    Returns
    -------
    targets : ndarray of intp
        Shape ``(n_samples,)`` when per sample, else shape ``(1,)``.
    per_sample : bool
        Whether the targets are per sample.

    Raises
    ------
    ValueError
        If an array has neither one element nor ``n_samples`` elements.
    """
    shape = np.asarray(bins).shape
    per_sample = shape == (n_samples,)
    if not per_sample and shape not in ((), (1,)):
        raise ValueError(
            _format_error(
                f"{operation} got {argument}= with shape {shape}, but there are "
                f"{n_samples} samples.",
                why=f"Why: {argument}= is one bin for every sample, or one bin "
                "per sample.",
                fix=f"pass one bin index, or an array of shape ({n_samples},) "
                "with -1 for samples that have no target",
            )
        )
    targets = _validate_target_bins(
        bins,
        n_bins=n_bins,
        coordinates=coordinates,
        operation=operation,
        argument=argument,
        allow_missing=per_sample,
    )
    return targets, per_sample


def locate_points(
    points: ArrayLike,
    *,
    binning: Binning,
    arena: Arena,
    operation: str,
    indexes: LookupIndexes | None = None,
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
    indexes : LookupIndexes, optional
        Keeps the locator's search structures, usually the environment's
        analysis cache. None builds them for this call only.

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
    # Before the empty and missing-coordinate checks, which read rows.
    require_position_columns(array, n_dims, call=operation, name="points")
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
    indices = locate(binning, arena, array, indexes, call=operation)
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
    indexes: LookupIndexes | None = None,
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
    indexes : LookupIndexes, optional
        Keeps the locator's search structures, usually the environment's
        analysis cache. None builds them for this call only.

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
            coordinates=coordinates,
            operation=operation,
            takes_points=True,
        )
    elif points is not None:
        resolved = locate_points(
            points,
            binning=binning,
            arena=arena,
            operation=operation,
            indexes=indexes,
        )
    else:
        resolved = _region_targets(
            region,
            regions,
            binning=binning,
            arena=arena,
            operation=operation,
            indexes=indexes,
        )
    if count is not None and resolved.size != count:
        raise ValueError(
            _format_error(
                f"{operation} takes {_COUNT_WORDS[count]}; got {resolved.size}.",
                why="Why: this query starts from a fixed number of bins.",
                fix=_COUNT_FIXES[count],
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
    indexes: LookupIndexes | None,
) -> NDArray[np.intp]:
    """The bins a named region covers, refusing a region that covers none."""
    if regions is None or not isinstance(region, str):
        raise TypeError(
            _format_error(
                f"{operation} takes a region name in region=, got "
                f"{type(region).__name__}.",
                why="Why: region= names an entry in env.regions.",
                fix="pass the region's name, for example region='reward'; for a "
                "Region object, add it with env.with_regions(...) first",
            )
        )
    if region not in regions:
        raise RegionNotFoundError(
            region, available=list(regions.keys()), argument="region"
        )
    resolved = np.flatnonzero(
        region_bin_mask(regions[region], binning=binning, arena=arena, indexes=indexes)
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
