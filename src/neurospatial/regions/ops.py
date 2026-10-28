"""Which regions contain each 2-D position.

These queries test positions directly against region shapes, without binning
them first:

- :func:`points_in_any_region` returns an ``(n_positions,)`` boolean mask that is
  True where a position lies in at least one region.
- :func:`regions_containing_points` returns, for each position, which regions
  contain it: a boolean DataFrame with one column per region (the default), or
  a list of region names per position.

A position is in a polygon region when it lies inside the polygon (on its edge
too, with the default ``include_boundary=True``). A point region, such as a
reward well, has no area, so a position counts as in it only within a radius
you choose with ``point_radius``, in the regions' units.

Both functions accept a ``transform`` that maps the positions into the regions'
coordinates first: any callable that maps an ``(n, 2)`` array to an ``(n, 2)``
array, such as an ``AffineND``.

For the bin-based rule used by analyses on an environment, see
``env.bins_in_region`` and ``env.region_mask``.

Examples
--------
>>> import numpy as np
>>> from shapely.geometry import box
>>> from neurospatial import Regions
>>> from neurospatial.regions import points_in_any_region, regions_containing_points
>>> regions = Regions()
>>> _ = regions.add("start_box", polygon=box(0.0, 0.0, 20.0, 20.0))
>>> _ = regions.add("reward_well", point=(80.0, 80.0))
>>> positions = np.array([[10.0, 10.0], [79.0, 81.0], [50.0, 50.0]])  # cm
>>> points_in_any_region(positions, regions, point_radius=5.0)
array([ True,  True, False])
>>> regions_containing_points(positions, regions, point_radius=5.0)
   start_box  reward_well
0       True        False
1      False         True
2      False        False
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal, overload

import numpy as np
import shapely
from numpy.typing import ArrayLike, NDArray

from neurospatial._exceptions import RegionNotFoundError, _format_error
from neurospatial._validation import validate_finite_scalar

from .core import Region

if TYPE_CHECKING:
    import pandas as pd

    from ..ops.transforms import SpatialTransform


def _refuse_non_finite_rows(array: NDArray[np.float64], *, name: str, fix: str) -> None:
    """Raise if any row of an ``(n, 2)`` array has a NaN or infinite coordinate."""
    bad = ~np.isfinite(array).all(axis=1)
    if bad.any():
        first = int(np.argmax(bad))
        raise ValueError(
            _format_error(
                f"{name} has {int(bad.sum())} row(s) with a NaN or infinite "
                f"coordinate (first at row {first}: {array[first].tolist()}).",
                why="a missing position is not outside every region; counting "
                "it as outside would turn tracking dropouts into time spent "
                "elsewhere.",
                fix=fix,
            )
        )


def _prepare_positions(
    positions: ArrayLike, transform: SpatialTransform | None
) -> NDArray[np.float64]:
    """Return finite ``(n_positions, 2)`` positions in the regions' coordinates.

    Parameters
    ----------
    positions : array-like, shape (n_positions, 2)
        Positions in the input space of ``transform``, or in the regions'
        coordinates when ``transform`` is None.
    transform : SpatialTransform or None
        Maps ``(n, 2)`` positions to ``(n, 2)`` positions in the regions'
        coordinates.

    Returns
    -------
    NDArray[np.float64], shape (n_positions, 2)

    Raises
    ------
    ValueError
        If ``positions`` is not numeric, is not ``(n_positions, 2)``, has a
        non-finite row, or ``transform`` returns a different shape or a
        non-finite row.
    """
    try:
        array = np.asarray(positions, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(
            _format_error(
                "positions must be a numeric (n_positions, 2) array.",
                fix="pass x, y positions as a float array, for example "
                "np.column_stack([x, y])",
            )
        ) from error
    if array.shape == (2,):
        raise ValueError(
            _format_error(
                "positions must have shape (n_positions, 2), got (2,).",
                why="a 1-D array of two numbers could be one (x, y) position or "
                "two samples of a 1-D position.",
                fix="for one (x, y) position pass positions[None] or "
                "np.atleast_2d(positions)",
            )
        )
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(
            _format_error(
                f"positions must have shape (n_positions, 2), got {array.shape}.",
                why="region membership compares (x, y) positions with 2-D "
                "polygons and points.",
                fix="pass the two coordinates the regions use, for example "
                "positions[:, :2], or for other dimensions bin the positions "
                "and use env.bins_in_region or env.region_mask",
            )
        )
    _refuse_non_finite_rows(
        array,
        name="positions",
        fix="drop or interpolate missing samples first, or query only the "
        "finite rows: keep = np.isfinite(positions).all(axis=1), then pass "
        "positions[keep] and treat the other rows as missing",
    )
    if transform is None or len(array) == 0:
        return array

    mapped = np.asarray(transform(array), dtype=np.float64)
    if mapped.shape != array.shape:
        raise ValueError(
            _format_error(
                f"transform returned shape {mapped.shape} for positions of "
                f"shape {array.shape}.",
                why="each position must map to one (x, y) position in the "
                "regions' coordinates.",
                fix="pass a transform that maps (n, 2) arrays to (n, 2) arrays, "
                "such as a 2-D AffineND",
            )
        )
    _refuse_non_finite_rows(
        mapped,
        name="transform(positions)",
        fix="check the transform: it maps finite positions to NaN or infinite "
        "coordinates",
    )
    return mapped


def _check_regions(selected: Sequence[Region], point_radius: float | None) -> None:
    """Refuse non-2-D point regions, and point regions without ``point_radius``."""
    validate_finite_scalar(
        point_radius, name="point_radius", positive=False, none_allowed=True
    )
    for region in selected:
        if region.n_dims != 2:
            raise ValueError(
                _format_error(
                    f"Region {region.name!r} is a {region.n_dims}-D point; "
                    "region membership queries take 2-D positions only.",
                    fix="project the positions and the region onto two "
                    "coordinates, or bin the positions in an environment of "
                    "matching dimension and use "
                    f"env.bins_in_region({region.name!r}) or env.region_mask",
                )
            )
    point_names = [region.name for region in selected if region.kind == "point"]
    if point_names and point_radius is None:
        listed = ", ".join(repr(name) for name in point_names)
        raise ValueError(
            _format_error(
                f"Region(s) {listed} are point regions, so membership needs "
                "point_radius.",
                why="a point has no area; whether a position is at it depends "
                "on a distance that only you can choose.",
                fix="pass point_radius= in the regions' units, for example "
                "point_radius=5.0 for a 5 cm reward-well radius, or bin the "
                f"positions and use env.bins_in_region({point_names[0]!r}) for "
                "the bin rule",
            )
        )


def _region_mask(
    positions: NDArray[np.float64],
    region: Region,
    *,
    point_radius: float | None,
    include_boundary: bool,
) -> NDArray[np.bool_]:
    """Return which of ``(n_positions, 2)`` positions lie in ``region``.

    A point region is a disc of radius ``point_radius`` (required by
    :func:`_check_regions`); ``include_boundary`` keeps its rim.
    """
    if region.kind == "point":
        assert point_radius is not None  # enforced by _check_regions
        offsets = positions - region.data
        distance = np.hypot(offsets[:, 0], offsets[:, 1])
        if include_boundary:
            return np.asarray(distance <= point_radius, dtype=bool)
        return np.asarray(distance < point_radius, dtype=bool)
    query = shapely.points(positions)
    predicate = shapely.covers if include_boundary else shapely.contains
    return np.asarray(predicate(region.data, query), dtype=bool)


def points_in_any_region(
    positions: ArrayLike,
    regions: Mapping[str, Region],
    *,
    transform: SpatialTransform | None = None,
    point_radius: float | None = None,
    include_boundary: bool = True,
) -> NDArray[np.bool_]:
    """Return whether each position lies in at least one region.

    Parameters
    ----------
    positions : array-like, shape (n_positions, 2)
        Positions in the regions' coordinates, or in the input space of
        ``transform`` when it is given. Every row must be finite.
    regions : Mapping[str, Region]
        Regions to test against; all of them are used.
    transform : SpatialTransform, optional
        Maps ``(n, 2)`` positions into the regions' coordinates, for example
        pixels to cm.
    point_radius : float, optional
        Radius around each point region, in the regions' units, within which a
        position counts as in it (Euclidean distance), for example the radius
        of a reward well. Required when ``regions`` holds a point region,
        because a point has no area; not used for polygons.
    include_boundary : bool, default=True
        Count positions on a polygon's edge, or exactly ``point_radius`` from a
        point region, as inside. If False, only strictly interior positions
        count.

    Returns
    -------
    mask : NDArray[np.bool_], shape (n_positions,)
        True where the position lies in at least one region.

    Raises
    ------
    ValueError
        If ``positions`` is not ``(n_positions, 2)`` or has a NaN or infinite
        row; if ``transform`` changes the shape or returns non-finite rows; if
        a point region is not 2-D; if ``regions`` holds a point region and
        ``point_radius`` is None; or if ``point_radius`` is negative or not
        finite.

    See Also
    --------
    regions_containing_points : Which regions contain each position.

    Examples
    --------
    >>> import numpy as np
    >>> from shapely.geometry import box
    >>> from neurospatial import Regions
    >>> from neurospatial.regions import points_in_any_region
    >>> regions = Regions()
    >>> _ = regions.add("start_box", polygon=box(0.0, 0.0, 20.0, 20.0))
    >>> positions = np.array([[10.0, 10.0], [20.0, 5.0], [50.0, 50.0]])  # cm
    >>> points_in_any_region(positions, regions)
    array([ True,  True, False])
    >>> points_in_any_region(positions, regions, include_boundary=False)
    array([ True, False, False])

    A point region needs a radius:

    >>> _ = regions.add("reward_well", point=(80.0, 80.0))
    >>> points_in_any_region([[83.0, 84.0]], regions, point_radius=5.0)
    array([ True])
    """
    transformed = _prepare_positions(positions, transform)
    selected = list(regions.values())
    _check_regions(selected, point_radius)

    mask = np.zeros(len(transformed), dtype=bool)
    for region in selected:
        if mask.all():
            break
        mask |= _region_mask(
            transformed,
            region,
            point_radius=point_radius,
            include_boundary=include_boundary,
        )
    return mask


@overload
def regions_containing_points(
    positions: ArrayLike,
    regions: Mapping[str, Region],
    *,
    transform: SpatialTransform | None = ...,
    region_names: Sequence[str] | None = ...,
    return_dataframe: Literal[True] = ...,
    point_radius: float | None = ...,
    include_boundary: bool = ...,
) -> pd.DataFrame: ...


@overload
def regions_containing_points(
    positions: ArrayLike,
    regions: Mapping[str, Region],
    *,
    transform: SpatialTransform | None = ...,
    region_names: Sequence[str] | None = ...,
    return_dataframe: Literal[False],
    point_radius: float | None = ...,
    include_boundary: bool = ...,
) -> list[list[str]]: ...


@overload
def regions_containing_points(
    positions: ArrayLike,
    regions: Mapping[str, Region],
    *,
    transform: SpatialTransform | None = ...,
    region_names: Sequence[str] | None = ...,
    return_dataframe: bool,
    point_radius: float | None = ...,
    include_boundary: bool = ...,
) -> pd.DataFrame | list[list[str]]: ...


def regions_containing_points(
    positions: ArrayLike,
    regions: Mapping[str, Region],
    *,
    transform: SpatialTransform | None = None,
    region_names: Sequence[str] | None = None,
    return_dataframe: bool = True,
    point_radius: float | None = None,
    include_boundary: bool = True,
) -> pd.DataFrame | list[list[str]]:
    """Return which regions contain each position.

    Parameters
    ----------
    positions : array-like, shape (n_positions, 2)
        Positions in the regions' coordinates, or in the input space of
        ``transform`` when it is given. Every row must be finite.
    regions : Mapping[str, Region]
        Regions to test against.
    transform : SpatialTransform, optional
        Maps ``(n, 2)`` positions into the regions' coordinates, for example
        pixels to cm.
    region_names : str or sequence of str, optional
        Regions to test, in column order. A single string is one name. If
        None, every region in ``regions`` is used, in collection order.
    return_dataframe : bool, default=True
        If True, return a boolean DataFrame with one column per region. If
        False, return a list with the names of the regions containing each
        position.
    point_radius : float, optional
        Radius around each point region, in the regions' units, within which a
        position counts as in it (Euclidean distance), for example the radius
        of a reward well. Required when a selected region is a point region,
        because a point has no area; not used for polygons.
    include_boundary : bool, default=True
        Count positions on a polygon's edge, or exactly ``point_radius`` from a
        point region, as inside. If False, only strictly interior positions
        count.

    Returns
    -------
    pandas.DataFrame or list of list of str
        With ``return_dataframe=True``, a DataFrame of shape
        ``(n_positions, n_selected_regions)`` with a ``RangeIndex`` and one
        bool column per region; entry ``(i, name)`` is True when
        ``positions[i]`` lies in region ``name``. With
        ``return_dataframe=False``, a list of length ``n_positions`` whose
        ``i``-th entry lists the names of the regions containing
        ``positions[i]``, in column order.

    Raises
    ------
    RegionNotFoundError
        If a name in ``region_names`` is not in ``regions``; the message names
        the closest region.
    ValueError
        If ``region_names`` repeats a name; if ``positions`` is not
        ``(n_positions, 2)`` or has a NaN or infinite row; if ``transform``
        changes the shape or returns non-finite rows; if a selected point
        region is not 2-D; if a selected region is a point region and
        ``point_radius`` is None; or if ``point_radius`` is negative or not
        finite.

    See Also
    --------
    points_in_any_region : Whether each position lies in any region.

    Examples
    --------
    >>> import numpy as np
    >>> from shapely.geometry import box
    >>> from neurospatial import Regions
    >>> from neurospatial.regions import regions_containing_points
    >>> regions = Regions()
    >>> _ = regions.add("left_arm", polygon=box(0.0, 0.0, 20.0, 100.0))
    >>> _ = regions.add("center", polygon=box(10.0, 40.0, 60.0, 60.0))
    >>> positions = np.array([[5.0, 80.0], [15.0, 50.0], [90.0, 50.0]])  # cm
    >>> regions_containing_points(positions, regions)
       left_arm  center
    0      True   False
    1      True    True
    2     False   False
    >>> regions_containing_points(positions, regions, return_dataframe=False)
    [['left_arm'], ['left_arm', 'center'], []]

    The fraction of samples in each region:

    >>> regions_containing_points(positions, regions).mean().round(3)
    left_arm    0.667
    center      0.333
    dtype: float64
    """
    import pandas as pd

    if isinstance(region_names, str):
        region_names = [region_names]  # one name, not its characters
    if region_names is None:
        selected = list(regions.values())
    else:
        for name in region_names:
            if name not in regions:
                raise RegionNotFoundError(
                    name, available=list(regions), argument="region_names", in_list=True
                )
        seen: set[str] = set()
        for name in region_names:
            if name in seen:
                raise ValueError(
                    _format_error(
                        f"region_names lists {name!r} more than once.",
                        why="each region is one column of the membership table.",
                        fix="pass each name once, for example "
                        "list(dict.fromkeys(region_names))",
                    )
                )
            seen.add(name)
        selected = [regions[name] for name in region_names]

    transformed = _prepare_positions(positions, transform)
    _check_regions(selected, point_radius)

    masks = {
        region.name: _region_mask(
            transformed,
            region,
            point_radius=point_radius,
            include_boundary=include_boundary,
        )
        for region in selected
    }
    if return_dataframe:
        return pd.DataFrame(masks, index=pd.RangeIndex(len(transformed)))
    if not masks:
        return [[] for _ in range(len(transformed))]
    names = list(masks)
    membership = np.column_stack(list(masks.values()))
    return [[names[j] for j in np.flatnonzero(row)] for row in membership]
