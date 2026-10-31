"""Which regions contain each 2-D position.

These queries test positions directly against region shapes, without binning
them first:

- :func:`points_in_any_region` returns an ``(n_positions,)`` boolean mask that is
  True where a position lies in at least one region.
- :func:`regions_containing_points` returns, for each position, which regions
  contain it: a boolean DataFrame with one column per region (the default), or
  a list of region names per position.

A position is in a polygon region when it lies inside the polygon. A point
region, such as a reward well, has no area, so a position counts as in it only
within a radius you choose with ``point_radius``, in the regions' units.

A position exactly on a polygon region's edge is in the region when nudging
it a hair north-east would move it inside. Regions that share an edge, such
as the four quadrants of an open field, therefore split the positions on
that edge, each to one region. These functions know no environment, so the
union of the polygon regions passed is the space a position can occupy:
where the nudge north-east would leave that union, as on an arena's north
or east wall, the position is nudged the other way, south-west, instead (at
a corner where that leaves too, the first of six further directions that
stays inside).
Every position the polygons cover is in at least one of them, and in
exactly one when they tile the space, so tracking clipped to the arena's
walls keeps its wall samples in the quadrants. A position on an edge with no
other region beyond it, such as the north edge of a lone box, is in that
region; membership on an edge therefore depends on the other polygon
regions in ``regions``, all of them, also those not named in
``region_names``. A position exactly ``point_radius`` from a point region is
in it when nudging it north-east moves it toward the point.

``env.bins_in_region`` applies the same rule to bin centers with the
environment as the space: a bin center's nudge north-east stays in its own
bin, so a bin is in a region when that nudge enters it. Where regions tile
the environment, as the quadrants of an arena built from its outline do, a
bin and a position at its center are in the same region.

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
from neurospatial._pynapple import reads_pynapple_samples
from neurospatial._validation import validate_finite_scalar
from neurospatial.geometry._region_bins import disc_mask, polygon_point_mask

from .core import Region

if TYPE_CHECKING:
    import pandas as pd
    from shapely.geometry import Polygon

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


def _transform_points(
    points: NDArray[np.float64],
    transform: SpatialTransform | None,
    *,
    name: str,
    where: str,
) -> NDArray[np.float64]:
    """Apply ``transform`` to ``(n, 2)`` points; refuse a reshaped or non-finite result.

    Returns ``points`` unchanged when ``transform`` is None. ``name`` is the
    argument that holds the transform and ``where`` names the points, for
    the error.
    """
    if transform is None:
        return points
    mapped = np.asarray(transform(points), dtype=np.float64)
    if mapped.shape != points.shape:
        raise ValueError(
            _format_error(
                f"{name} returned shape {mapped.shape} for {where}; expected "
                f"{points.shape}, one (x, y) row per point.",
                why="regions are 2-D, so each (x, y) must map to one (x, y)",
                fix="pass a transform that maps (n, 2) arrays to (n, 2) arrays, "
                "such as VideoCalibration.transform_px_to_cm or a 2-D AffineND",
            )
        )
    bad = ~np.isfinite(mapped).all(axis=1)
    if bad.any():
        first = int(np.argmax(bad))
        raise ValueError(
            _format_error(
                f"{name} returned non-finite coordinates for {int(bad.sum())} of "
                f"{len(mapped)} points of {where} (first: {points[first].tolist()}).",
                why="a region needs finite coordinates",
                fix="check that the transform is defined over all of these "
                "points, for example that a homography does not send them to "
                "infinity",
            )
        )
    return mapped


def _transform_polygon(
    polygon: Polygon,
    transform: SpatialTransform | None,
    *,
    name: str,
    where: str,
) -> Polygon:
    """Apply :func:`_transform_points` to every vertex, holes included."""
    if transform is None:
        return polygon
    return shapely.transform(
        polygon, lambda xy: _transform_points(xy, transform, name=name, where=where)
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
    if len(array) == 0:
        return array
    return _transform_points(array, transform, name="transform", where="positions")


def _named_regions(
    regions: Mapping[str, Region], region_names: str | Sequence[str] | None
) -> list[str]:
    """Return the names to use: all of ``regions``, or ``region_names`` checked.

    A single string is one name, not its characters. A name not in ``regions``
    raises :class:`~neurospatial.RegionNotFoundError` with the closest name.
    """
    if region_names is None:
        return list(regions)
    if isinstance(region_names, str):
        region_names = [region_names]
    for name in region_names:
        if name not in regions:
            raise RegionNotFoundError(
                name, available=list(regions), argument="region_names", in_list=True
            )
    return list(region_names)


def _check_regions(selected: Sequence[Region], point_radius: float | None) -> None:
    """Refuse non-2-D point regions, and point regions without ``point_radius``."""
    validate_finite_scalar(
        point_radius, name="point_radius", positive=False, none_allowed=True
    )
    if point_radius is not None and point_radius <= 0:
        raise ValueError(
            _format_error(
                f"point_radius must be positive, got {point_radius}.",
                why="a point region is a disc of radius point_radius; a disc "
                "of radius 0 has no area, so no position would be in it.",
                fix="pass the radius of the place, for example "
                "point_radius=5.0 for a 5 cm reward well",
            )
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


def _polygon_domain(
    regions: Mapping[str, Region],
) -> shapely.geometry.base.BaseGeometry | None:
    """Return the union of the polygon regions, prepared; None when there are none.

    Positions on a polygon's edge are split between the polygons on either
    side within this union, and an edge on its outline is held.
    """
    polygons = [region.data for region in regions.values() if region.kind == "polygon"]
    if not polygons:
        return None
    domain = shapely.union_all(polygons)
    shapely.prepare(domain)
    return domain


def _region_mask(
    positions: NDArray[np.float64],
    region: Region,
    *,
    point_radius: float | None,
    domain: shapely.geometry.base.BaseGeometry | None,
) -> NDArray[np.bool_]:
    """Return which of ``(n_positions, 2)`` positions lie in ``region``.

    A point region is a disc of radius ``point_radius`` (required by
    :func:`_check_regions`) whose rim follows the half-open rule of
    :func:`~neurospatial.geometry.polygon_bin_mask`. A polygon's edges
    follow that rule within ``domain``, from :func:`_polygon_domain`.
    """
    if region.kind == "point":
        assert point_radius is not None  # enforced by _check_regions
        return disc_mask(region.data, point_radius, positions)
    return polygon_point_mask(region.data, positions, domain=domain)


@reads_pynapple_samples
def points_in_any_region(
    positions: ArrayLike,
    regions: Mapping[str, Region],
    *,
    transform: SpatialTransform | None = None,
    point_radius: float | None = None,
) -> NDArray[np.bool_]:
    """Return whether each position lies in at least one region.

    Parameters
    ----------
    positions : array-like, shape (n_positions, 2)
        Positions in the regions' coordinates, or in the input space of
        ``transform`` when it is given. Every row must be finite.
    regions : Mapping[str, Region]
        Regions to test against; all of them are used. A position on a
        polygon's edge goes to the polygon a nudge north-east enters, or,
        where that nudge leaves every polygon, the one a nudge south-west
        enters, so positions on the outline of their union are in a region.
    transform : SpatialTransform, optional
        Maps ``(n, 2)`` positions into the regions' coordinates, for example
        pixels to cm.
    point_radius : float, optional
        Radius around each point region, in the regions' units, within which a
        position counts as in it (Euclidean distance), for example the radius
        of a reward well. Required when ``regions`` holds a point region,
        because a point has no area; not used for polygons.

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
        ``point_radius`` is None; or if ``point_radius`` is not positive and
        finite.

    See Also
    --------
    regions_containing_points : Which regions contain each position.

    Examples
    --------
    >>> import numpy as np
    >>> from shapely.geometry import box
    >>> from neurospatial import Regions
    >>> from neurospatial.regions import points_in_any_region, regions_containing_points
    >>> regions = Regions()
    >>> _ = regions.add("start_box", polygon=box(0.0, 0.0, 20.0, 20.0))
    >>> positions = np.array([[10.0, 10.0], [20.0, 5.0], [50.0, 50.0]])  # cm
    >>> points_in_any_region(positions, regions)
    array([ True,  True, False])

    (20, 5) lies on the box's east edge. With no other region beyond that
    edge, the box holds it; a region beside the box would take it, so boxes
    side by side split their shared edges:

    >>> _ = regions.add("next_box", polygon=box(20.0, 0.0, 40.0, 20.0))
    >>> regions_containing_points([[20.0, 5.0]], regions, return_dataframe=False)
    [['next_box']]

    A point region needs a radius:

    >>> _ = regions.add("reward_well", point=(80.0, 80.0))
    >>> points_in_any_region([[82.0, 83.0]], regions, point_radius=5.0)
    array([ True])
    """
    transformed = _prepare_positions(positions, transform)
    selected = list(regions.values())
    _check_regions(selected, point_radius)

    domain = _polygon_domain(regions)
    mask = np.zeros(len(transformed), dtype=bool)
    for region in selected:
        # Test only the positions no earlier region contains.
        outside = np.flatnonzero(~mask)
        if not outside.size:
            break
        mask[outside] = _region_mask(
            transformed[outside],
            region,
            point_radius=point_radius,
            domain=domain,
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
) -> pd.DataFrame | list[list[str]]: ...


@reads_pynapple_samples
def regions_containing_points(
    positions: ArrayLike,
    regions: Mapping[str, Region],
    *,
    transform: SpatialTransform | None = None,
    region_names: Sequence[str] | None = None,
    return_dataframe: bool = True,
    point_radius: float | None = None,
) -> pd.DataFrame | list[list[str]]:
    """Return which regions contain each position.

    Parameters
    ----------
    positions : array-like, shape (n_positions, 2)
        Positions in the regions' coordinates, or in the input space of
        ``transform`` when it is given. Every row must be finite.
    regions : Mapping[str, Region]
        Regions to test against. A position on a polygon's edge goes to the
        polygon a nudge north-east enters, or, where that nudge leaves every
        polygon in ``regions`` (selected in ``region_names`` or not), the
        one a nudge south-west enters. Regions that tile an arena therefore
        give every position in it, walls included, exactly one region.
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
        ``point_radius`` is None; or if ``point_radius`` is not positive and
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

    names = _named_regions(regions, region_names)
    seen: set[str] = set()
    for name in names:
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
    selected = [regions[name] for name in names]

    transformed = _prepare_positions(positions, transform)
    _check_regions(selected, point_radius)

    domain = _polygon_domain(regions)
    masks = {
        region.name: _region_mask(
            transformed,
            region,
            point_radius=point_radius,
            domain=domain,
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
