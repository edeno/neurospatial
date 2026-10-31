"""
Position and head direction reading from NWB behavior containers.

This module provides functions for reading Position and CompassDirection
data from pynwb.behavior containers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error
from neurospatial._keyword_hint import keyword_hint
from neurospatial._units import _UNIT_SPELLINGS
from neurospatial.io.nwb._adapters import (
    data_from_series,
    missing_error,
    require_unscaled_for_lazy,
    select_one,
    timestamps_from_series,
    timestamps_handle_from_series,
)
from neurospatial.io.nwb._core import _find_containers_by_type, _require_pynwb, logger
from neurospatial.io.nwb._holders import (
    NWBHeadDirection,
    NWBPosition,
    _declared_units,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pynwb import NWBFile
    from pynwb.behavior import SpatialSeries


@keyword_hint("processing_module", "position_name")
def read_position(
    nwbfile: NWBFile,
    *,
    processing_module: str | None = None,
    position_name: str | None = None,
    columns: Sequence[int] | None = None,
    lazy: bool = False,
) -> NWBPosition:
    """
    Read position data from NWB file.

    Parameters
    ----------
    nwbfile : NWBFile
        The NWB file to read from.
    processing_module : str, optional
        Name of the processing module containing Position data. If None,
        every processing module and acquisition is searched.
    position_name : str, optional
        Name of the SpatialSeries within a Position container, or its full
        path such as ``"processing/behavior/Position/position"``. Required
        when the search finds more than one series.
    columns : sequence of int, optional
        Indices of the spatial columns of a 2-D series, in the order to
        return them, for a series that also stores non-spatial values, for
        example ``columns=[0, 1]`` for x and y when column 2 is speed.
        Default returns every column. Required when the series has more than
        two columns and its unit is not one length or pixel unit, or when the
        unit lists several units (``"centimeters; centimeters/second"``).
    lazy : bool, default False
        If ``False`` (default), positions and timestamps are fully materialized
        into ``NDArray[np.float64]``. If ``True``, the h5py-backed
        ``SpatialSeries.data`` handle (and the timestamps handle) are returned
        **without** copying, so they materialize only when sliced or
        ``np.asarray``-ed. Use this to keep large recordings off-RAM. Values are
        in the series' ``unit``: stored × ``conversion`` + ``offset``;
        ``lazy=True`` raises when that map is not the identity.

    Returns
    -------
    NWBPosition
        Frozen, non-iterable holder with ``times`` in seconds, ``positions``
        in the declared physical ``units`` after conversion and offset, and
        ``units=None`` when no unit is declared. Recognized unit aliases are
        normalized to m/cm/mm/um/px without rescaling values. Arrays are eager,
        read-only float64 by default; lazy handles remain inside the holder. ``frame``
        preserves a nonempty SpatialSeries ``reference_frame`` verbatim,
        or is None when undeclared. No coordinate transform is inferred.
        With ``columns=``, ``positions`` has one column per selected index,
        and ``units`` is the series unit only when that is one recognized
        length or pixel unit; otherwise it is None, because the declaration
        may describe the other columns.

    Raises
    ------
    KeyError
        If no Position container found, or if specified position_name not found.
    ValueError
        If more than one Position series matches (pass ``position_name=``),
        ``lazy=True`` and the series declares a non-identity ``conversion``
        or ``offset``, the series needs ``columns=`` (see above), or
        ``columns`` does not name distinct column indices of a 2-D series.
    ImportError
        If pynwb is not installed.

    Notes
    -----
    Lazy handles are **only valid while the backing ``NWBFile`` / ``NWBHDF5IO``
    is open**. Slice or ``np.asarray`` them inside the ``with NWBHDF5IO(...)``
    block; accessing them after the file is closed raises an error from the HDF5
    layer. ``lazy=False`` returns read-only, fully-materialized arrays that
    remain valid after the file closes.

    Positions and timestamps must have the same length on both paths; the lazy
    path compares the handles' ``.shape[0]`` without materializing values.

    ``lazy=True`` returns the **stored** dtype (whatever the SpatialSeries was
    written with, e.g. ``float32``), whereas ``lazy=False`` casts to
    ``float64``. The values are equal; only the dtype may differ. With
    ``columns=``, the lazy handle reads the requested rows and keeps the
    selected columns each time it is sliced.

    NWB gives a SpatialSeries one ``unit`` string for all its columns, so a
    file that stores x, y and speed together cannot say which column is
    which in a machine-readable way; its ``description`` and ``comments``
    often do, and the error for an ambiguous series quotes them.

    Examples
    --------
    >>> from datetime import datetime, timezone
    >>> from pynwb import NWBFile, NWBHDF5IO
    >>> nwbfile = NWBFile(
    ...     "Example recording", "example", datetime(2026, 1, 1, tzinfo=timezone.utc)
    ... )
    >>> import numpy as np
    >>> from pynwb.behavior import Position, SpatialSeries
    >>> from neurospatial.io.nwb import read_position
    >>> position = Position(name="Position")
    >>> _ = position.add_spatial_series(
    ...     SpatialSeries(
    ...         name="xy",
    ...         data=np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
    ...         timestamps=np.arange(3) / 30.0,
    ...         reference_frame="arena origin",
    ...         unit="centimeters",
    ...     )
    ... )
    >>> _ = nwbfile.create_processing_module("behavior", "Tracking").add(position)
    >>> pos = read_position(nwbfile, processing_module="behavior", position_name="xy")
    >>> pos.positions.shape
    (3, 2)
    >>> with NWBHDF5IO("position.nwb", "w") as io:
    ...     io.write(nwbfile)
    >>> with NWBHDF5IO("position.nwb", "r") as io:
    ...     pos_lazy = read_position(io.read(), lazy=True)
    ...     first_position = np.asarray(pos_lazy.positions[:1])
    >>> first_position.tolist()
    [[0.0, 0.0]]

    """
    _require_pynwb()
    from pynwb.behavior import Position as PositionType

    spatial_series = _get_spatial_series(
        nwbfile,
        PositionType,
        "Position",
        processing_module,
        position_name,
        "position_name",
    )

    shape = np.shape(spatial_series.data)
    unit = spatial_series.unit
    if columns is None:
        _require_unambiguous_columns(spatial_series, shape)
    else:
        columns = _checked_columns(columns, spatial_series.name, shape)
        if _declared_units(unit) not in _SPATIAL_UNITS:
            # The declaration may describe the columns left out.
            unit = None

    if lazy:
        require_unscaled_for_lazy(spatial_series, context="read_position")
        # The h5py-backed handles materialize on slice / np.asarray; the
        # holder checks their lengths from .shape without reading values.
        timestamps = timestamps_handle_from_series(spatial_series)
        positions = spatial_series.data
        if columns is not None:
            positions = _ColumnView(positions, columns)
    else:
        timestamps = timestamps_from_series(spatial_series)
        positions = data_from_series(spatial_series)
        if columns is not None:
            positions = positions[:, columns]
    return NWBPosition(
        timestamps,
        positions,
        unit,
        frame=spatial_series.reference_frame,
    )


_SPATIAL_UNITS = frozenset(_UNIT_SPELLINGS.values())

# pynwb's placeholders for an empty description and comments.
_NWB_TEXT_DEFAULTS = frozenset({"", "no description", "no comments"})


def _require_unambiguous_columns(series: SpatialSeries, shape: tuple[int, ...]) -> None:
    """Refuse a series whose columns may not all be spatial coordinates.

    A unit that lists several units (``"centimeters; centimeters/second"``)
    or, with three or more columns, any unit that is not one length or pixel
    unit leaves open that a column is speed or another non-spatial value.
    """
    n_columns = shape[1] if len(shape) == 2 else 1
    unit = str(series.unit or "")
    compound = any(separator in unit for separator in ";,")
    if n_columns < 2 or not (
        compound or (n_columns > 2 and _declared_units(unit) not in _SPATIAL_UNITS)
    ):
        return
    notes = [
        str(text).strip()
        for text in (series.description, series.comments)
        if text is not None and str(text).strip().lower() not in _NWB_TEXT_DEFAULTS
    ]
    described = (
        " The series describes its data as: " + " / ".join(repr(n) for n in notes) + "."
        if notes
        else ""
    )
    raise ValueError(
        _format_error(
            f"Position series '{series.name}' has {n_columns} columns and "
            f"declares unit {series.unit!r}, which is not one length or pixel "
            "unit.",
            why="a column may hold speed or another non-spatial value; using "
            f"every column would bin it as a spatial axis of a {n_columns}-D "
            "environment.",
            fix="pass columns= with the spatial columns, e.g. columns=[0, 1] "
            f"for x and y, or columns={list(range(n_columns))} if all are "
            f"spatial.{described}",
        )
    )


def _checked_columns(
    columns: Sequence[int], name: str, shape: tuple[int, ...]
) -> list[int]:
    """Validate ``columns=`` against a series of the given ``shape``."""
    if len(shape) != 2:
        raise ValueError(
            _format_error(
                f"read_position(columns=...) selects columns, but Position "
                f"series '{name}' stores one value per sample (shape {shape}).",
                fix="omit columns= for this series.",
            )
        )
    n_columns = shape[1]
    fix = (
        f"pass distinct integer column indices from 0 to {n_columns - 1}, "
        "e.g. columns=[0, 1]."
    )
    if np.ndim(columns) != 1:
        raise ValueError(
            _format_error(
                f"columns= takes a sequence of integer column indices, got "
                f"{columns!r}.",
                fix=fix,
            )
        )
    selected = list(columns)
    if not selected:
        raise ValueError(
            _format_error("columns= must name at least one column.", fix=fix)
        )
    for column in selected:
        if isinstance(column, bool) or not isinstance(column, (int, np.integer)):
            raise ValueError(
                _format_error(
                    f"columns= takes integer column indices, got {column!r}.",
                    fix=fix,
                )
            )
        if not 0 <= column < n_columns:
            raise ValueError(
                _format_error(
                    f"columns=: column {column} is out of range for Position "
                    f"series '{name}', which has {n_columns} columns.",
                    fix=fix,
                )
            )
    repeated = sorted({int(c) for c in selected if selected.count(c) > 1})
    if repeated:
        raise ValueError(
            _format_error(f"columns= repeats column {repeated[0]}.", fix=fix)
        )
    return [int(c) for c in selected]


class _ColumnView:
    """Lazy view of some columns of a 2-D file-backed dataset.

    Slicing reads the requested rows from the dataset and keeps the selected
    columns; ``np.asarray`` reads every row. Valid only while the backing
    file is open.
    """

    __slots__ = ("_columns", "_data")

    def __init__(self, data: Any, columns: list[int]) -> None:
        self._data = data
        self._columns = columns

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self._data.shape[0]), len(self._columns))

    @property
    def ndim(self) -> int:
        return 2

    @property
    def dtype(self) -> np.dtype[Any]:
        dtype: np.dtype[Any] = np.dtype(self._data.dtype)
        return dtype

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, key: Any) -> Any:
        rows, *rest = key if isinstance(key, tuple) else (key,)
        if len(rest) > 1:
            raise IndexError(
                f"too many indices for a 2-D position array: {len(rest) + 1}"
            )
        selected = np.asarray(self._data[rows])[..., self._columns]
        return selected[..., rest[0]] if rest else selected

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> NDArray[Any]:
        values: NDArray[Any] = np.asarray(self[:])
        return values if dtype is None else values.astype(dtype)


def _get_spatial_series(
    nwbfile: NWBFile,
    container_type: type,
    type_name: str,
    processing_module: str | None,
    series_name: str | None,
    keyword: str,
) -> SpatialSeries:
    """
    Get the one SpatialSeries of a behavior container type.

    Parameters
    ----------
    nwbfile : NWBFile
        The NWB file to search.
    container_type : type
        The container type class (Position or CompassDirection).
    type_name : str
        Human-readable name for error messages, e.g. ``"Position"``.
    processing_module : str or None
        If provided, look only in this processing module.
    series_name : str or None
        Series name, or its full path such as
        ``"processing/behavior/Position/position"``.
    keyword : str
        The reader argument that takes ``series_name``.

    Returns
    -------
    SpatialSeries
        The selected series.

    Raises
    ------
    KeyError
        If the module, a container or the named series is missing.
    ValueError
        If more than one series matches.
    """
    fix = (
        f"store the {type_name} container in processing/behavior, or pass "
        "processing_module= with the module that holds it."
    )
    found = _find_containers_by_type(nwbfile, container_type)
    if processing_module is not None:
        if processing_module not in nwbfile.processing:
            raise missing_error(
                f"Processing module '{processing_module}' not found in NWB "
                f"file. Available modules: {list(nwbfile.processing.keys())}.",
                fix=fix,
            )
        found = [
            (path, container)
            for path, container in found
            if path.startswith(f"processing/{processing_module}/")
        ]
        searched = f"processing/{processing_module}"
    else:
        searched = "processing/* and acquisition"
    if not found:
        raise missing_error(
            f"No {type_name} data found in NWB file. Searched: {searched}.",
            fix=fix,
        )
    candidates = [
        (f"{path}/{name}", container.spatial_series[name])
        for path, container in found
        for name in sorted(container.spatial_series)
    ]
    if not candidates:
        raise missing_error(
            f"{type_name} container(s) {[path for path, _ in found]} "
            "has no SpatialSeries.",
            fix=f"add a SpatialSeries to the {type_name} container.",
        )
    series = select_one(
        candidates, series_name, what=f"{type_name} series", keyword=keyword
    )
    logger.debug("Reading %s series '%s'", type_name, series.name)
    return series


@keyword_hint("processing_module", "compass_name")
def read_head_direction(
    nwbfile: NWBFile,
    *,
    processing_module: str | None = None,
    compass_name: str | None = None,
) -> NWBHeadDirection:
    """
    Read head direction data from NWB file.

    Parameters
    ----------
    nwbfile : NWBFile
        The NWB file to read from.
    processing_module : str, optional
        Name of the processing module containing CompassDirection data. If
        None, every processing module and acquisition is searched.
    compass_name : str, optional
        Name of the SpatialSeries within a CompassDirection container, or its
        full path such as
        ``"processing/behavior/CompassDirection/head_direction"``. Required
        when the search finds more than one series.

    Returns
    -------
    NWBHeadDirection
        Frozen, non-iterable holder with ``times`` in seconds, ``headings``
        in radians and ``frame``, the series' ``reference_frame`` verbatim
        (None when blank). Angles stored as ``(n_samples,)`` values pass
        through conversion and offset and are converted from the declared
        unit, which must be radians or degrees. Direction vectors stored as
        ``(n_samples, 2)`` become ``arctan2(y, x)``; a zero vector is NaN.

    Raises
    ------
    KeyError
        If no CompassDirection container found, or if specified compass_name
        not found.
    ValueError
        If more than one CompassDirection series matches (pass
        ``compass_name=``), the series ``unit`` is not a radian or degree
        spelling (pynwb's
        default ``"meters"`` included) for angle data, or the data is
        neither ``(n_samples,)`` nor ``(n_samples, 2)``.
    ImportError
        If pynwb is not installed.

    Notes
    -----
    ``headings`` are assumed to follow the package convention: zero East and
    increasing counterclockwise. The reader does not interpret
    ``reference_frame`` text. Check ``frame``; if it describes another
    convention, for example zero North and clockwise, convert with
    ``np.pi / 2 - headings``.

    Examples
    --------
    >>> from pynwb import NWBHDF5IO  # doctest: +SKIP
    >>> with NWBHDF5IO("session.nwb", "r") as io:  # doctest: +SKIP
    ...     nwbfile = io.read()
    ...     hd = read_head_direction(nwbfile)
    """
    _require_pynwb()
    from pynwb.behavior import CompassDirection as CompassDirectionType

    spatial_series = _get_spatial_series(
        nwbfile,
        CompassDirectionType,
        "CompassDirection",
        processing_module,
        compass_name,
        "compass_name",
    )

    raw = data_from_series(spatial_series)
    if raw.ndim == 2 and raw.shape[1] == 2:
        # (n, 2) heading vectors along the frame's x and y axes; the angle
        # does not depend on their length or unit. A zero vector has no
        # direction.
        angles = np.arctan2(raw[:, 1], raw[:, 0])
        angles[(raw[:, 0] == 0) & (raw[:, 1] == 0)] = np.nan
    elif raw.ndim == 1 or (raw.ndim == 2 and raw.shape[1] == 1):
        angles = raw.reshape(-1) * _radians_per_unit(spatial_series)
    else:
        raise ValueError(
            _format_error(
                f"CompassDirection series '{spatial_series.name}' has data of "
                f"shape {raw.shape}.",
                why="head direction is stored as one angle per sample, shape "
                "(n_samples,), or as (n_samples, 2) direction vectors.",
                fix="store the head direction in one of those shapes.",
            )
        )
    return NWBHeadDirection(
        timestamps_from_series(spatial_series),
        angles,
        frame=spatial_series.reference_frame,
    )


_RADIANS_PER_ANGLE_UNIT = {
    **dict.fromkeys(["radians", "radian", "rad", "rads"], 1.0),
    **dict.fromkeys(["degrees", "degree", "deg", "degs"], np.pi / 180),
}


def _radians_per_unit(series: SpatialSeries) -> float:
    """Radians in one stored angle unit; refuse any other unit declaration."""
    unit = getattr(series, "unit", None)
    key = str(unit).strip().lower() if unit is not None else ""
    if key not in _RADIANS_PER_ANGLE_UNIT:
        raise ValueError(
            _format_error(
                f"CompassDirection series '{series.name}' declares unit "
                f"{unit!r}, which is not an angle unit.",
                why="the angles could be degrees or radians; pynwb gives a "
                "SpatialSeries saved without unit= the unit 'meters'.",
                fix="rewrite the series with unit='radians' or unit='degrees'.",
            )
        )
    return _RADIANS_PER_ANGLE_UNIT[key]
