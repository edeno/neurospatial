"""Named array holders returned by the NWB component readers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class NWBPosition:
    """Position samples and declared physical units from an NWB file.

    Attributes
    ----------
    times : NDArray[np.float64] or lazy handle
        Sample timestamps in seconds, shape (n_samples,).
    positions : NDArray[np.float64] or lazy handle
        Coordinates after the stored conversion and offset, shape
        (n_samples, n_dims). Lazy reads retain the stored dtype and require
        an identity conversion and offset.
    units : str | None
        Declared physical unit, normalized to m/cm/mm/px when recognized.
        Other nonempty declarations are preserved; None means undeclared.

    Examples
    --------
    >>> from neurospatial.io.nwb import NWBPosition
    >>> pos = NWBPosition(np.array([0.0, 0.1]), np.zeros((2, 2)), "cm")
    >>> pos.positions.shape, pos.units
    ((2, 2), 'cm')
    """

    times: NDArray[np.float64] | Any
    positions: NDArray[np.float64] | Any
    units: str | None


@dataclass(frozen=True)
class NWBHeadDirection:
    """Allocentric head direction samples from an NWB file.

    Attributes
    ----------
    times : NDArray[np.float64]
        Sample timestamps in seconds, shape (n_samples,).
    headings : NDArray[np.float64]
        Angles in radians, zero East and increasing counterclockwise,
        shape (n_samples,).

    Examples
    --------
    >>> from neurospatial.io.nwb import NWBHeadDirection
    >>> hd = NWBHeadDirection(np.array([0.0, 0.1]), np.array([0.0, np.pi]))
    >>> hd.headings.shape
    (2,)
    """

    times: NDArray[np.float64]
    headings: NDArray[np.float64]


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
