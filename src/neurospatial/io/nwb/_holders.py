"""Named array holders returned by the NWB component readers.

Each holder checks its own shapes, normalizes its coordinate declarations
and keeps read-only copies of the NumPy arrays it is given. Lazy file
handles (``h5py.Dataset``) are kept as given; their shapes are checked
without reading values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error
from neurospatial._results import ReadOnlyArraysMixin


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


@dataclass(frozen=True)
class NWBUnits:
    """Unit spikes, identities and acquisition coverage from an NWB file.

    Attributes
    ----------
    spike_times : list of NDArray[np.float64] or lazy handles
        One sorted spike-time array in seconds per selected unit.
    unit_ids : NDArray[np.int64]
        Table IDs aligned with spike_times and obs_intervals.
    obs_intervals : list of NDArray[np.float64] | None
        One (n_intervals, 2) acquisition-window array per selected unit,
        or None when the NWB column is absent.
    spike_window : NDArray[np.float64] | None
        Intersection of all selected units' observation intervals, shape
        (n_intervals, 2), or None when the column is absent. An empty
        intersection has shape (0, 2).

    Examples
    --------
    >>> from neurospatial.io.nwb import NWBUnits
    >>> units = NWBUnits([np.array([0.1])], np.array([7]), None, None)
    >>> units.unit_ids.tolist()
    [7]
    """

    spike_times: list[NDArray[np.float64] | Any]
    unit_ids: NDArray[np.int64]
    obs_intervals: list[NDArray[np.float64]] | None
    spike_window: NDArray[np.float64] | None

    def __post_init__(self) -> None:
        n = len(self.spike_times)
        n_obs = n if self.obs_intervals is None else len(self.obs_intervals)
        if not (len(self.unit_ids) == n_obs == n):
            raise ValueError(
                f"spike_times has {n} units, unit_ids {len(self.unit_ids)}, "
                f"obs_intervals {n_obs}; these must match one-to-one.\n"
                "Fix: pass one unit_id and one obs_intervals entry per spike train."
            )
