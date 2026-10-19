"""Read bin-valued features of the environment at the animal's positions.

A feature map gives one value per bin of a fixed environment: floor height, a
texture code, light level. ``sample_spatial_features`` reads each feature at
each tracked position from the bin that contains the position, so the sampled
features can be aligned to events or used as regressors next to the neural
data.

Lookup is by containment only. A position outside the environment, or one
with a non-finite coordinate, has no bin and its value is missing (NaN), never the value of the
nearest bin. A NaN in a feature map is an unknown value and stays missing; a
finite zero is a valid value. Values are the bins' own values: nothing is
smoothed or interpolated, and ``support`` beside every value says whether it
was observed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurospatial._exceptions import _format_error
from neurospatial._results import ReadOnlyArraysMixin, ResultMixin, as_label_array
from neurospatial.geometry import (
    CallIndexes,
    locate,
    require_cartesian,
    require_position_columns,
)
from neurospatial.ops.egocentric import egocentric_to_allocentric

if TYPE_CHECKING:
    import pandas as pd

    from neurospatial.geometry import Arena, Binning
    from neurospatial.ops._protocols import FeatureLookupConsumer

__all__ = [
    "SpatialFeatureResult",
    "sample_egocentric_features",
    "sample_spatial_features",
]


@dataclass(frozen=True, repr=False)
class SpatialFeatureResult(ReadOnlyArraysMixin, ResultMixin):
    """Feature-map values at query points, from ``sample_spatial_features``.

    Each query point is a tracked position, or for egocentric sampling a point
    at a fixed offset from the animal. ``values`` keeps the sample axis first,
    so row ``k`` belongs to input sample ``k``.

    Attributes
    ----------
    values : NDArray[np.float64], shape (n_time, n_features) or (n_time, n_points, n_features)
        Feature value at each query point: the value of the bin containing it.
        NaN where ``support`` is False.
    support : NDArray[np.bool_], same shape as ``values``
        True where the query point is finite, lies in a bin, and that bin's
        feature value is finite.
    bins : NDArray[np.intp], shape (n_time,) or (n_time, n_points)
        Bin containing each query point, or -1 for a point outside the
        environment or with a non-finite coordinate.
    feature_ids : NDArray, shape (n_features,)
        Label of each feature column.
    units : str or None
        Length units of the environment's coordinates.
    frame : str or None
        Coordinate frame label of the environment.
    ego_points : NDArray[np.float64] or None, shape (n_points, 2)
        Egocentric offsets that were sampled (x ahead, y left), in ``units``;
        None for sampling at the positions themselves.
    query_positions : NDArray[np.float64] or None, shape (n_time, n_points, 2)
        Allocentric position of each egocentric query point; None for sampling
        at the positions themselves.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.ops import sample_spatial_features
    >>> x, y = np.meshgrid(np.arange(0.5, 10), np.arange(0.5, 10))
    >>> env = Environment.from_samples(np.c_[x.ravel(), y.ravel()], bin_size=2.0)
    >>> height = np.zeros(env.n_bins)
    >>> result = sample_spatial_features(
    ...     env, height, np.array([[5.0, 5.0], [50.0, 50.0]]), feature_ids=["height"]
    ... )
    >>> result.summary()
    {'n_samples': 2, 'n_features': 1, 'n_points': 0, 'fraction_supported': 0.5, 'n_outside': 1}
    """

    values: NDArray[np.float64]
    support: NDArray[np.bool_]
    bins: NDArray[np.intp]
    feature_ids: NDArray[Any]
    units: str | None
    frame: str | None
    ego_points: NDArray[np.float64] | None = None
    query_positions: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        """Check that the fields describe one sampling, then own the arrays."""
        problems = self._inconsistencies()
        if problems:
            raise ValueError(
                "SpatialFeatureResult fields are inconsistent: "
                + "; ".join(problems)
                + "."
            )
        self._own_arrays()

    def _inconsistencies(self) -> list[str]:
        """List every way the fields disagree with each other."""
        shape = np.shape(self.values)
        if len(shape) not in (2, 3):
            return [
                "values needs shape (n_time, n_features) or "
                "(n_time, n_points, n_features)"
            ]
        problems: list[str] = []
        if np.shape(self.support) != shape:
            problems.append(f"support needs shape {shape}")
        if np.shape(self.bins) != shape[:-1]:
            problems.append(f"bins needs shape {shape[:-1]}")
        if np.shape(self.feature_ids) != (shape[-1],):
            problems.append(f"feature_ids needs shape ({shape[-1]},)")
        if len(shape) == 2:
            if self.ego_points is not None or self.query_positions is not None:
                problems.append(
                    "ego_points and query_positions are only for egocentric "
                    "values of shape (n_time, n_points, n_features)"
                )
        else:
            n_time, n_points = shape[:2]
            if self.ego_points is None or np.shape(self.ego_points) != (n_points, 2):
                problems.append(f"ego_points needs shape ({n_points}, 2)")
            query = self.query_positions
            if query is None or np.shape(query) != (n_time, n_points, 2):
                problems.append(
                    f"query_positions needs shape ({n_time}, {n_points}, 2)"
                )
        return problems

    def samples_table(self) -> pd.DataFrame:
        """One row per sample, query point (egocentric only) and feature.

        Returns
        -------
        pandas.DataFrame
            Columns ``sample_index``, ``point_index`` (egocentric sampling
            only), ``feature_id``, ``bin`` (-1 where the point has no bin),
            ``value`` (NaN where unsupported) and ``supported``, ordered by
            sample, then point, then feature. ``df.attrs`` holds ``units`` and
            ``frame``.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.ops import sample_spatial_features
        >>> x, y = np.meshgrid(np.arange(0.5, 10), np.arange(0.5, 10))
        >>> env = Environment.from_samples(
        ...     np.c_[x.ravel(), y.ravel()], bin_size=2.0, units="cm"
        ... )
        >>> height = np.where(env.bin_centers[:, 0] > 5.0, 2.0, 0.0)  # raised half
        >>> result = sample_spatial_features(
        ...     env,
        ...     height,
        ...     np.array([[8.0, 5.0], [50.0, 50.0]]),
        ...     feature_ids=["height"],
        ... )
        >>> table = result.samples_table()
        >>> table[["sample_index", "feature_id", "value", "supported"]]
           sample_index feature_id  value  supported
        0             0     height    2.0       True
        1             1     height    NaN      False
        >>> table.attrs
        {'units': 'cm', 'frame': None}
        """
        import pandas as pd

        shape = self.values.shape
        n_features = shape[-1]
        index = np.indices(shape[:-1]).reshape(len(shape) - 1, -1)
        data: dict[str, Any] = {"sample_index": np.repeat(index[0], n_features)}
        if len(shape) == 3:
            data["point_index"] = np.repeat(index[1], n_features)
        data["feature_id"] = np.tile(self.feature_ids, index.shape[1])
        data["bin"] = np.repeat(self.bins.reshape(-1), n_features)
        data["value"] = self.values.reshape(-1)
        data["supported"] = self.support.reshape(-1)
        df = pd.DataFrame(data)
        df.attrs.update(units=self.units, frame=self.frame)
        return df

    def summary(self) -> dict[str, Any]:
        """Scalar description of how much of the sampling was supported.

        Returns
        -------
        dict
            ``n_samples``, ``n_features``, ``n_points`` (0 when sampling at
            the positions themselves), ``fraction_supported`` (supported
            values over all values; NaN with no values) and ``n_outside``
            (query points with no bin: outside the environment or with a
            non-finite coordinate).

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.ops import sample_spatial_features
        >>> x, y = np.meshgrid(np.arange(0.5, 10), np.arange(0.5, 10))
        >>> env = Environment.from_samples(np.c_[x.ravel(), y.ravel()], bin_size=2.0)
        >>> inside, outside = [5.0, 5.0], [50.0, 50.0]
        >>> result = sample_spatial_features(
        ...     env, np.ones(env.n_bins), np.array([inside, outside])
        ... )
        >>> summary = result.summary()
        >>> summary["fraction_supported"], summary["n_outside"]
        (0.5, 1)
        """
        n_values = self.support.size
        return {
            "n_samples": int(self.values.shape[0]),
            "n_features": int(self.values.shape[-1]),
            "n_points": int(self.values.shape[1]) if self.values.ndim == 3 else 0,
            "fraction_supported": (
                float(self.support.sum() / n_values) if n_values else float("nan")
            ),
            "n_outside": int(np.count_nonzero(self.bins < 0)),
        }


def sample_spatial_features(
    env: FeatureLookupConsumer,
    fields: ArrayLike,
    positions: ArrayLike,
    *,
    feature_ids: ArrayLike | None = None,
) -> SpatialFeatureResult:
    """Read bin-valued feature maps at the animal's positions.

    Each position takes the value of the bin that contains it
    (``env.bin_at``). A position outside the environment or with a
    non-finite coordinate has no bin, and a NaN in a feature map marks an unknown value:
    both give NaN with ``support`` False. A finite zero is a valid value.
    Nothing is smoothed or interpolated, so a step in a map (a texture
    boundary, a raised platform) stays a step.

    Parameters
    ----------
    env : Environment
        Allocentric environment the feature maps are defined on. Its layout
        must know which positions lie inside its bins: a grid, a polygon, or a
        track built with ``track_width``.
    fields : array-like, shape (n_features, n_bins) or (n_bins,)
        One value per bin for each feature, in that feature's own units. NaN
        marks a bin whose value is unknown. A 1-D array is one feature.
    positions : array-like, shape (n_time, n_dims)
        Allocentric positions in the environment's coordinates and units.
    feature_ids : array-like, shape (n_features,), optional
        Unique label of each feature. Default ``np.arange(n_features)``.

    Returns
    -------
    SpatialFeatureResult
        ``values`` and ``support`` of shape ``(n_time, n_features)``, the bin
        of each position in ``bins`` (-1 for none), and ``units`` and
        ``frame`` copied from ``env``.

    Raises
    ------
    CartesianOnlyError
        If ``env`` is an egocentric polar environment.
    NotImplementedError
        If the environment's layout has no known domain, such as a track built
        without ``track_width`` or a linearized 1-D track: it cannot tell
        whether a position is inside a bin.
    ValueError
        If ``fields`` does not have one value per bin or contains inf, if
        ``positions`` is not ``(n_time, n_dims)``, or if ``feature_ids`` are
        not one unique label per feature.

    See Also
    --------
    neurospatial.events.align_continuous_to_events : Align the sampled
        features to events.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.ops import sample_spatial_features
    >>> x, y = np.meshgrid(np.arange(0.5, 10), np.arange(0.5, 10))
    >>> env = Environment.from_samples(
    ...     np.c_[x.ravel(), y.ravel()], bin_size=2.0, units="cm"
    ... )
    >>> height = np.where(env.bin_centers[:, 0] > 5.0, 2.0, 0.0)  # raised half
    >>> light = np.full(env.n_bins, 0.0)
    >>> positions = np.array([[2.0, 5.0], [8.0, 5.0], [50.0, 5.0]])
    >>> result = sample_spatial_features(
    ...     env, np.stack([height, light]), positions, feature_ids=["height", "light"]
    ... )
    >>> result.values
    array([[ 0.,  0.],
           [ 2.,  0.],
           [nan, nan]])
    >>> result.support
    array([[ True,  True],
           [ True,  True],
           [False, False]])
    >>> int(result.bins[2])  # outside the environment: no bin
    -1
    """
    call = "sample_spatial_features()"
    binning, arena, field_rows, ids = _lookup_inputs(
        env,
        fields,
        feature_ids,
        call,
        why="positions are the animal's allocentric positions, located in the "
        "environment's bins.",
        fix="sample feature maps on the allocentric environment the positions "
        "were tracked in",
    )
    points = np.asarray(positions, dtype=np.float64)
    require_position_columns(points, binning.bins.n_dims, call=call)
    bins = locate(binning, arena, points, CallIndexes(), call=call)
    values, support = _read_bins(field_rows, bins)
    return SpatialFeatureResult(
        values=values,
        support=support,
        bins=bins,
        feature_ids=ids,
        units=env.units,
        frame=env.frame,
    )


def sample_egocentric_features(
    env: FeatureLookupConsumer,
    fields: ArrayLike,
    positions: ArrayLike,
    headings: ArrayLike,
    ego_points: ArrayLike,
    *,
    feature_ids: ArrayLike | None = None,
) -> SpatialFeatureResult:
    """Read bin-valued feature maps at points fixed in the animal's frame.

    Each ``ego_points`` row is an offset in the animal's frame, such as 10 cm
    ahead or 10 cm to the left. At every sample it is placed relative to the
    animal's position and heading, and takes the value of the bin that
    contains it, exactly as in `sample_spatial_features`. A point outside the
    environment, a point whose position or heading is NaN, and a NaN in a
    feature map are all missing: NaN with ``support`` False. A finite zero is a
    valid value. Nothing is smoothed or interpolated.

    Parameters
    ----------
    env : Environment
        Two-dimensional allocentric environment the feature maps are defined
        on. Its layout must know which positions lie inside its bins: a grid,
        a polygon, or a track built with ``track_width``.
    fields : array-like, shape (n_features, n_bins) or (n_bins,)
        One value per bin for each feature, in that feature's own units. NaN
        marks a bin whose value is unknown. A 1-D array is one feature.
    positions : array-like, shape (n_time, 2)
        Allocentric positions in the environment's coordinates and units.
    headings : array-like, shape (n_time,)
        Allocentric heading in radians (0 = East, +pi/2 = North).
    ego_points : array-like, shape (n_points, 2)
        Offsets in the animal's frame, in the environment's length units:
        column 0 is ahead and column 1 is to the left (angle 0 = ahead,
        +pi/2 = left).
    feature_ids : array-like, shape (n_features,), optional
        Unique label of each feature. Default ``np.arange(n_features)``.

    Returns
    -------
    SpatialFeatureResult
        ``values`` and ``support`` of shape ``(n_time, n_points, n_features)``,
        ``bins`` of shape ``(n_time, n_points)`` (-1 for none),
        ``query_positions`` of shape ``(n_time, n_points, 2)`` holding the
        allocentric position of each point, ``ego_points``, and ``units`` and
        ``frame`` copied from ``env``.

    Raises
    ------
    CartesianOnlyError
        If ``env`` is an egocentric polar environment.
    NotImplementedError
        If the environment's layout has no known domain, such as a track built
        without ``track_width`` or a linearized 1-D track.
    ValueError
        If ``env`` is not 2-D (headings rotate offsets in the plane), if
        ``positions`` is not ``(n_time, 2)``, ``headings`` is not
        ``(n_time,)`` or ``ego_points`` is not ``(n_points, 2)``, or if
        ``fields`` or ``feature_ids`` are invalid as in
        `sample_spatial_features`.

    See Also
    --------
    sample_spatial_features : Sample at the positions themselves.
    egocentric_to_allocentric : The frame transform used here.

    Examples
    --------
    A point 20 cm ahead of an animal facing East is East of it; 20 cm to its
    left is North.

    >>> import numpy as np
    >>> from shapely.geometry import box
    >>> from neurospatial import Environment
    >>> from neurospatial.ops import sample_egocentric_features
    >>> env = Environment.from_polygon(box(0, 0, 100, 100), bin_size=10.0, units="cm")
    >>> x_of_bin = env.bin_centers[:, 0]
    >>> result = sample_egocentric_features(
    ...     env,
    ...     x_of_bin,
    ...     np.array([[50.0, 50.0]]),
    ...     np.array([0.0]),
    ...     np.array([[20.0, 0.0], [0.0, 20.0]]),
    ...     feature_ids=["x"],
    ... )
    >>> result.query_positions[0].round(1)
    array([[70., 50.],
           [50., 70.]])
    >>> result.values[0, :, 0]
    array([75., 55.])
    """
    call = "sample_egocentric_features()"
    binning, arena, field_rows, ids = _lookup_inputs(
        env,
        fields,
        feature_ids,
        call,
        why="positions and headings are the animal's allocentric pose, and the "
        "offsets are placed relative to it.",
        fix="sample feature maps on the allocentric environment the positions "
        "were tracked in",
    )
    if binning.bins.n_dims != 2:
        raise ValueError(
            _format_error(
                f"{call} needs a 2-D environment; this one has "
                f"{binning.bins.n_dims} dimension(s).",
                why="A heading rotates an offset within the plane, so ahead and "
                "left are only defined in 2-D.",
                fix="Use sample_spatial_features for 1-D or 3-D environments.",
            )
        )
    points = np.asarray(positions, dtype=np.float64)
    require_position_columns(points, 2, call=call)
    offsets = np.asarray(ego_points, dtype=np.float64)
    require_position_columns(offsets, 2, call=call, name="ego_points")
    angles = np.asarray(headings, dtype=np.float64)
    if angles.shape != (points.shape[0],):
        raise ValueError(
            _format_error(
                f"{call} got headings of shape {angles.shape} for "
                f"{points.shape[0]} positions.",
                why="Each position needs the heading the animal had there.",
                fix="Pass one heading in radians per row of positions.",
            )
        )
    n_time, n_points = points.shape[0], offsets.shape[0]
    query = egocentric_to_allocentric(
        points, angles, np.broadcast_to(offsets, (n_time, n_points, 2))
    )
    bins = locate(
        binning, arena, query.reshape(-1, 2), CallIndexes(), call=call
    ).reshape(n_time, n_points)
    values, support = _read_bins(field_rows, bins)
    return SpatialFeatureResult(
        values=values,
        support=support,
        bins=bins,
        feature_ids=ids,
        units=env.units,
        frame=env.frame,
        ego_points=offsets,
        query_positions=query,
    )


def _lookup_inputs(
    env: FeatureLookupConsumer,
    fields: ArrayLike,
    feature_ids: ArrayLike | None,
    call: str,
    *,
    why: str,
    fix: str,
) -> tuple[Binning, Arena, NDArray[np.float64], NDArray[Any]]:
    """Check the environment and feature maps shared by both sampling routes."""
    require_cartesian(env.coordinate_system, call, why=why, fix=fix)
    binning, arena = env.binning, env.arena
    _require_known_domain(arena, call)
    field_rows = _validated_fields(fields, binning.bins.n_bins, call)
    ids = _validated_feature_ids(feature_ids, field_rows.shape[0], call)
    return binning, arena, field_rows, ids


def _require_known_domain(arena: Arena, call: str) -> None:
    """Refuse a layout that cannot say whether a position is inside its bins."""
    if not arena.domain_known:
        raise NotImplementedError(
            _format_error(
                f"{call} cannot tell which positions are inside this "
                "environment's bins.",
                why="Its layout has no known domain, so a lookup would put "
                "far-away positions in their nearest bin.",
                fix="Build the environment with from_polygon, from_samples/grid, "
                "or from_graph(..., track_width=...).",
            )
        )


def _validated_fields(fields: ArrayLike, n_bins: int, call: str) -> NDArray[np.float64]:
    """Return ``fields`` as float64 ``(n_features, n_bins)``, refusing inf."""
    rows = np.asarray(fields, dtype=np.float64)
    if rows.ndim == 1:
        rows = rows[np.newaxis]
    if rows.ndim != 2 or rows.shape[0] == 0 or rows.shape[1] != n_bins:
        raise ValueError(
            _format_error(
                f"{call} got fields of shape {np.shape(fields)} for an "
                f"environment with {n_bins} bins.",
                why="Each feature map needs one value per bin, with the bin axis last.",
                fix=f"Pass fields of shape (n_features, {n_bins}), or ({n_bins},) "
                "for one feature; use NaN for bins whose value is unknown.",
            )
        )
    if np.isinf(rows).any():
        raise ValueError(
            _format_error(
                f"{call} got inf in fields.",
                why="An infinite feature value is neither a measurement nor a "
                "marked unknown.",
                fix="Replace inf with the bin's finite value, or with NaN where "
                "the value is unknown.",
            )
        )
    return rows


def _validated_feature_ids(
    feature_ids: ArrayLike | None, n_features: int, call: str
) -> NDArray[Any]:
    """Return one unique label per feature, defaulting to ``arange``."""
    if feature_ids is None:
        return np.arange(n_features)
    if isinstance(feature_ids, str | bytes) or np.ndim(feature_ids) == 0:
        raise ValueError(
            _format_error(
                f"{call} got feature_ids={feature_ids!r}; it must be a sequence "
                "of labels, one per feature.",
                why="A single string or number would be split into characters or "
                "cannot label each feature.",
                fix="Pass a list or array, e.g. feature_ids=['height'] for one "
                "feature.",
            )
        )
    ids = as_label_array(feature_ids)
    if ids.shape != (n_features,):
        raise ValueError(
            _format_error(
                f"{call} got feature_ids of shape {ids.shape} for {n_features} "
                "features.",
                fix="Pass one label per row of fields.",
            )
        )
    labels = ids.tolist()
    if len(set(labels)) != len(labels):
        raise ValueError(
            _format_error(
                f"{call} got feature_ids that are not unique: {labels}.",
                why="Each feature column is reported under its id.",
                fix="Pass a distinct label for every row of fields.",
            )
        )
    return ids


def _read_bins(
    fields: NDArray[np.float64], bins: NDArray[np.intp]
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Read ``fields`` ``(n_features, n_bins)`` at ``bins`` of any shape.

    ``-1`` never indexes a field: points without a bin stay NaN. Returns
    ``values`` and ``support`` of shape ``bins.shape + (n_features,)``.
    """
    inside = bins >= 0
    values = np.full((*bins.shape, fields.shape[0]), np.nan)
    values[inside] = fields[:, bins[inside]].T
    support = inside[..., np.newaxis] & np.isfinite(values)
    return values, support
