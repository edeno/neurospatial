"""Named array holders returned by the NWB component readers.

Each holder checks its own shapes, normalizes its coordinate declarations
and keeps read-only copies of the NumPy arrays it is given. Lazy file
handles (``h5py.Dataset``) are kept as given; their shapes are checked
without reading values.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error
from neurospatial._results import (
    ReadOnlyArraysMixin,
    _copy_unit_table,
    align_unit_table,
    read_only_copy,
    reduce_through_constructor,
)

if TYPE_CHECKING:
    import pandas as pd

    from neurospatial.animation.skeleton import Skeleton


def _declared_units(units: str | None) -> str | None:
    """Return a unit declaration in canonical spelling, or None when blank.

    Recognized spellings become ``"m"``, ``"cm"``, ``"mm"``, ``"um"`` or
    ``"px"``; any other declaration is kept with surrounding whitespace
    removed. Values are never rescaled.
    """
    if units is None or not str(units).strip():
        return None
    from neurospatial._units import _UNIT_SPELLINGS

    declaration = str(units).strip()
    return _UNIT_SPELLINGS.get(declaration.lower(), declaration)


def _declared_frame(frame: str | None) -> str | None:
    """Return a reference-frame description verbatim, or None when blank."""
    return str(frame) if frame is not None and str(frame).strip() else None


def _require_rows(
    holder: str, times: Any, **arrays: tuple[Any, tuple[int, ...]]
) -> None:
    """Check that ``times`` is 1-D and each array has one row per time.

    ``arrays`` maps a field name to ``(value, allowed_ndims)``. Shapes are
    read with ``np.shape`` so lazy handles are not materialized.
    """
    times_shape = np.shape(times)
    if len(times_shape) != 1:
        raise ValueError(
            _format_error(
                f"{holder} times must be 1-D, got shape {times_shape}.",
                fix="pass one timestamp in seconds per sample.",
            )
        )
    n_times = times_shape[0]
    for name, (value, ndims) in arrays.items():
        shape = np.shape(value)
        if len(shape) not in ndims:
            raise ValueError(
                _format_error(
                    f"{holder} {name} must have {' or '.join(map(str, ndims))} "
                    f"dimension(s), got shape {shape}.",
                    fix=f"pass one row of {name} per sample.",
                )
            )
        if shape[0] != n_times:
            raise ValueError(
                _format_error(
                    f"Length mismatch: {name}={shape[0]}, times={n_times}. "
                    "These must agree.",
                    fix=f"pass one row of {name} per timestamp.",
                )
            )


@dataclass(frozen=True)
class NWBPosition(ReadOnlyArraysMixin):
    """Position samples and coordinate declarations from an NWB file.

    Attributes
    ----------
    times : NDArray[np.float64] or lazy handle
        Sample timestamps in seconds, shape (n_samples,).
    positions : NDArray[np.float64] or lazy handle
        Coordinates after the stored conversion and offset, shape
        (n_samples, n_dims) or (n_samples,). Lazy reads retain the stored
        dtype and require an identity conversion and offset.
    units : str | None
        Declared physical unit, normalized to m/cm/mm/um/px when recognized.
        Other nonempty declarations are kept without surrounding whitespace;
        None means undeclared.
    frame : str | None, default None
        Nonempty SpatialSeries reference-frame description, preserved verbatim.
        None means undeclared; no coordinate transform is inferred.

    Raises
    ------
    ValueError
        If ``times`` is not 1-D or ``positions`` does not have one row per
        timestamp.

    Notes
    -----
    NumPy arrays are stored as read-only copies; copy with ``np.array(...)``
    to modify values. Lazy handles are stored as given.

    Examples
    --------
    >>> from neurospatial.io.nwb import NWBPosition
    >>> pos = NWBPosition(np.array([0.0, 0.1]), np.zeros((2, 2)), "centimeters")
    >>> pos.positions.shape, pos.units
    ((2, 2), 'cm')
    """

    times: NDArray[np.float64] | Any
    positions: NDArray[np.float64] | Any
    units: str | None
    frame: str | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        _require_rows("NWBPosition", self.times, positions=(self.positions, (1, 2)))
        object.__setattr__(self, "units", _declared_units(self.units))
        object.__setattr__(self, "frame", _declared_frame(self.frame))
        self._own_arrays()


@dataclass(frozen=True)
class NWBHeadDirection(ReadOnlyArraysMixin):
    """Head direction samples from an NWB file.

    Attributes
    ----------
    times : NDArray[np.float64]
        Sample timestamps in seconds, shape (n_samples,).
    headings : NDArray[np.float64]
        Angles in radians, shape (n_samples,), in the package convention:
        zero East and increasing counterclockwise.
    frame : str | None, default None
        Nonempty SpatialSeries reference-frame description, preserved
        verbatim. None means undeclared. Nothing is inferred from it: check
        that it describes zero East and counterclockwise angles, and convert
        ``headings`` yourself when it does not.

    Raises
    ------
    ValueError
        If ``times`` or ``headings`` is not 1-D, or their lengths differ.

    Notes
    -----
    NumPy arrays are stored as read-only copies; copy with ``np.array(...)``
    to modify values.

    Examples
    --------
    >>> from neurospatial.io.nwb import NWBHeadDirection
    >>> hd = NWBHeadDirection(np.array([0.0, 0.1]), np.array([0.0, np.pi]))
    >>> hd.headings.shape
    (2,)
    """

    times: NDArray[np.float64]
    headings: NDArray[np.float64]
    frame: str | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        _require_rows("NWBHeadDirection", self.times, headings=(self.headings, (1,)))
        object.__setattr__(self, "frame", _declared_frame(self.frame))
        self._own_arrays()


@_copy_unit_table
@dataclass(frozen=True)
class NWBUnits(ReadOnlyArraysMixin):
    """Unit spikes, identities and acquisition coverage from an NWB file.

    Attributes
    ----------
    spike_times : tuple of NDArray[np.float64] or lazy handles
        One sorted spike-time array in seconds per selected unit.
    unit_ids : NDArray[np.int64]
        Unique table IDs aligned with spike_times and obs_intervals.
    obs_intervals : tuple of NDArray[np.float64] or None
        One (n_intervals, 2) acquisition-window array per selected unit,
        or None when the NWB column is absent. A unit with no observation
        rows has shape (0, 2).
    spike_window : NDArray[np.float64] or None
        Intersection of all selected units' observation intervals, shape
        (n_intervals, 2), or None when ``obs_intervals`` is None. An empty
        intersection has shape (0, 2). Derived from ``obs_intervals``, not
        passed to the constructor.
    unit_table : pandas.DataFrame or None, default None
        Per-unit metadata, one row per unit in ``unit_ids`` order, for
        example brain region and putative cell type. A ``unit_id`` column or
        a named or non-default index aligns rows by identity; a plain
        ``RangeIndex`` uses spike-train order. Reading returns a copy.
    resolution : float or None, default None
        Smallest possible difference between two spike times, in seconds
        (usually one over the acquisition sampling rate), or None when
        unknown.

    Raises
    ------
    ValueError
        If the per-unit fields do not match one-to-one, ``unit_ids`` repeats
        an id, an ``obs_intervals`` entry is not (n_intervals, 2) with
        finite rows whose stop follows their start, ``unit_table`` rows do
        not match ``unit_ids``, or ``resolution`` is not a finite positive
        number.

    Notes
    -----
    NumPy arrays are stored as read-only copies; copy with ``np.array(...)``
    to modify values. Lazy spike-train handles are stored as given.

    Examples
    --------
    >>> from neurospatial.io.nwb import NWBUnits
    >>> units = NWBUnits(
    ...     [np.array([0.1]), np.array([0.6])],
    ...     np.array([7, 11]),
    ...     [np.array([[0.0, 1.0]]), np.array([[0.5, 2.0]])],
    ... )
    >>> units.unit_ids.tolist(), units.spike_window.tolist()
    ([7, 11], [[0.5, 1.0]])
    """

    spike_times: Sequence[NDArray[np.float64] | Any]
    unit_ids: NDArray[np.int64]
    obs_intervals: Sequence[NDArray[np.float64]] | None
    spike_window: NDArray[np.float64] | None = field(init=False)
    unit_table: pd.DataFrame | None = field(default=None, kw_only=True, compare=False)
    resolution: float | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        unit_ids = np.asarray(self.unit_ids)
        n = len(self.spike_times)
        n_obs = n if self.obs_intervals is None else len(self.obs_intervals)
        if not (len(unit_ids) == n_obs == n):
            raise ValueError(
                f"spike_times has {n} units, unit_ids {len(unit_ids)}, "
                f"obs_intervals {n_obs}; these must match one-to-one.\n"
                "Fix: pass one unit_id and one obs_intervals entry per spike train."
            )
        values, counts = np.unique(unit_ids, return_counts=True)
        if np.any(counts > 1):
            raise ValueError(
                _format_error(
                    f"unit_ids repeats {values[counts > 1].tolist()}.",
                    why="each id must name exactly one unit.",
                    fix="list each unit id once.",
                )
            )
        object.__setattr__(self, "unit_ids", unit_ids)
        object.__setattr__(self, "spike_times", tuple(map(_own, self.spike_times)))
        window = None
        if self.obs_intervals is not None:
            obs = tuple(
                _unit_obs_intervals(rows, unit_id)
                for rows, unit_id in zip(self.obs_intervals, unit_ids, strict=True)
            )
            object.__setattr__(self, "obs_intervals", obs)
            window = _shared_window(obs, unit_ids)
        object.__setattr__(self, "spike_window", window)
        object.__setattr__(
            self,
            "unit_table",
            align_unit_table(self.unit_table, unit_ids, context="NWBUnits"),
        )
        if self.resolution is not None:
            resolution = float(self.resolution)
            if not (np.isfinite(resolution) and resolution > 0):
                raise ValueError(
                    _format_error(
                        f"NWBUnits resolution must be a finite positive number "
                        f"of seconds, got {self.resolution!r}.",
                        fix="pass the spike-time precision, usually one over the "
                        "sampling rate (e.g. 1 / 30000), or None when unknown.",
                    )
                )
            object.__setattr__(self, "resolution", resolution)
        self._own_arrays()


def _own(value: Any) -> Any:
    """Read-only copy of a NumPy array; a lazy handle is kept as given."""
    return read_only_copy(value) if isinstance(value, np.ndarray) else value


def _unit_obs_intervals(rows: Any, unit_id: Any) -> NDArray[np.float64]:
    """One unit's observation rows as a read-only (n_intervals, 2) array."""
    rows = np.asarray(rows, dtype=np.float64)
    if rows.size == 0:
        # NWB allows a unit with no observation rows (never observed); pynwb
        # returns those as shape (0,).
        rows = np.empty((0, 2))
    if rows.ndim != 2 or rows.shape[1] != 2:
        raise ValueError(
            _format_error(
                f"obs_intervals for unit {unit_id} must have shape "
                f"(n_intervals, 2), got {rows.shape}.",
                fix="give each unit's observation windows as "
                "[[start, stop], ...] rows in seconds.",
            )
        )
    return read_only_copy(rows)


def _shared_window(
    obs_intervals: tuple[NDArray[np.float64], ...], unit_ids: NDArray[Any]
) -> NDArray[np.float64]:
    """Intersection of every unit's observation rows, shape (n_intervals, 2)."""
    from neurospatial._intervals import as_intervals, intersect_intervals

    window = None
    for rows, unit_id in zip(obs_intervals, unit_ids, strict=True):
        if rows.shape[0] == 0:
            return np.empty((0, 2), dtype=np.float64)
        normalized = as_intervals(rows, name=f"obs_intervals[{unit_id}]")
        assert normalized is not None
        window = (
            normalized if window is None else intersect_intervals(window, normalized)
        )
    return np.empty((0, 2), dtype=np.float64) if window is None else window


@dataclass(frozen=True)
class NWBPose(ReadOnlyArraysMixin):
    """Pose-estimation keypoints and their coordinate declarations from NWB.

    Attributes
    ----------
    bodyparts : Mapping[str, NDArray[np.float64] or lazy handle]
        Read-only mapping from ``PoseEstimationSeries`` name to keypoint
        coordinates, each shape (n_samples, n_dims), after the stored
        conversion and offset. Untracked samples stay NaN.
    times : NDArray[np.float64] or lazy handle
        Timestamps in seconds shared by every bodypart, shape (n_samples,).
    skeleton : Skeleton or None
        Skeleton linked from the ``PoseEstimation`` container, or None when
        the container links none.
    units : str or None, default None
        Unit shared by every bodypart, normalized to m/cm/mm/um/px when
        recognized; None means undeclared.
    frame : str or None, default None
        Shared reference-frame description, preserved verbatim; None means
        undeclared.
    confidence : Mapping[str, NDArray[np.float64] or lazy handle] or None, default None
        Read-only mapping from bodypart name to the pose estimator's
        per-sample confidence, shape (n_samples,), for each bodypart whose
        series stores one. None when no series stores confidence.

    Raises
    ------
    ValueError
        If there are no bodyparts, ``times`` is not 1-D, a bodypart or
        confidence array does not have one row per timestamp, or
        ``confidence`` names a bodypart that is not in ``bodyparts``.

    Notes
    -----
    Low-confidence keypoints are not masked: no confidence threshold suits
    every tracker and recording. Choose one for your data and mask before
    analysis, for example
    ``np.where((pose.confidence["nose"] < 0.9)[:, None], np.nan, xy)``.
    NumPy arrays are stored as read-only copies; lazy handles as given.

    Examples
    --------
    >>> from neurospatial.io.nwb import NWBPose
    >>> pose = NWBPose(
    ...     {"nose": np.array([[1.0, 2.0], [np.nan, np.nan]])},
    ...     np.array([0.0, 0.1]),
    ...     None,
    ...     units="centimeters",
    ...     confidence={"nose": np.array([0.98, 0.01])},
    ... )
    >>> pose.units, pose.bodyparts["nose"].shape
    ('cm', (2, 2))
    """

    bodyparts: Mapping[str, NDArray[np.float64] | Any]
    times: NDArray[np.float64] | Any
    skeleton: Skeleton | None
    units: str | None = field(default=None, kw_only=True)
    frame: str | None = field(default=None, kw_only=True)
    confidence: Mapping[str, NDArray[np.float64] | Any] | None = field(
        default=None, kw_only=True
    )

    def __post_init__(self) -> None:
        if not self.bodyparts:
            raise ValueError(
                _format_error(
                    "NWBPose has no bodyparts.",
                    fix="pass at least one bodypart trajectory.",
                )
            )
        arrays = {
            f"bodyparts[{name!r}]": (value, (2,))
            for name, value in self.bodyparts.items()
        }
        if self.confidence is not None:
            unknown = sorted(set(self.confidence) - set(self.bodyparts))
            if unknown:
                raise ValueError(
                    _format_error(
                        f"NWBPose confidence names unknown bodyparts {unknown}.",
                        fix=f"use bodypart names from {sorted(self.bodyparts)}.",
                    )
                )
            arrays |= {
                f"confidence[{name!r}]": (value, (1,))
                for name, value in self.confidence.items()
            }
            object.__setattr__(self, "confidence", _owned_mapping(self.confidence))
        _require_rows("NWBPose", self.times, **arrays)
        object.__setattr__(self, "bodyparts", _owned_mapping(self.bodyparts))
        object.__setattr__(self, "units", _declared_units(self.units))
        object.__setattr__(self, "frame", _declared_frame(self.frame))
        self._own_arrays()

    def __reduce__(self) -> Any:
        """Pickle through the constructor; read-only mappings cannot pickle."""
        return reduce_through_constructor(self)


def _owned_mapping(mapping: Mapping[str, Any]) -> Mapping[str, Any]:
    """Read-only mapping of read-only array copies (lazy handles kept)."""
    return MappingProxyType({name: _own(value) for name, value in mapping.items()})
