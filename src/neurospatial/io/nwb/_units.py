"""
Spike-time reading from an NWB ``units`` table.

This module provides :func:`read_units`, which extracts per-neuron spike-time
arrays from the ragged ``units`` :class:`~hdmf.common.table.DynamicTable` of an
NWB file. Spikes are the one neural data type lacking a reader, so without it
every multi-cell workflow has to start in bare ``pynwb``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from neurospatial.io.nwb._core import _require_pynwb

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pynwb import NWBFile


def read_units(
    nwbfile: NWBFile,
    *,
    unit_ids: Sequence[int] | None = None,
) -> tuple[list[NDArray[np.float64]], NDArray]:
    """
    Read spike-time arrays from an NWB ``units`` table.

    Parameters
    ----------
    nwbfile : pynwb.NWBFile
        The NWB file to read from. Must contain a ``units`` table.
    unit_ids : sequence of int, optional
        Subset of units to read, given as the table's ``id`` values (the
        identifiers in ``units.id``), not row indices. Each requested id is
        matched against ``units.id``; any id not present raises ``ValueError``
        naming the missing id(s). Default reads all units in table order.

    Returns
    -------
    spike_trains : list of NDArray[np.float64]
        One sorted 1-D array of spike times (seconds) per unit, aligned with
        ``unit_ids``.
    unit_ids : NDArray
        The unit identifiers, aligned with ``spike_trains``.

    Raises
    ------
    ValueError
        If the file has no ``units`` table, or if any requested ``unit_ids``
        value is not present in ``units.id``.
    ImportError
        If pynwb is not installed.

    Notes
    -----
    The returned ``spike_trains`` / ``unit_ids`` tuple is the standard input
    for downstream population analyses: temporally bin the trains into a
    spike-count matrix and pass it, with a place-field model, to
    :func:`neurospatial.decoding.decode_position` to reconstruct the animal's
    trajectory.

    Examples
    --------
    >>> import numpy as np
    >>> from pynwb import NWBFile
    >>> from datetime import datetime, timezone
    >>> nwbfile = NWBFile(
    ...     session_description="demo",
    ...     identifier="demo",
    ...     session_start_time=datetime(2020, 1, 1, tzinfo=timezone.utc),
    ... )
    >>> nwbfile.add_unit(spike_times=[0.1, 0.5, 1.2], id=7)
    >>> nwbfile.add_unit(spike_times=[0.3, 0.9], id=11)
    >>> spike_trains, unit_ids = read_units(nwbfile)
    >>> unit_ids
    array([ 7, 11])
    >>> spike_trains[0]
    array([0.1, 0.5, 1.2])
    """
    _require_pynwb()

    units = nwbfile.units
    if units is None:
        raise ValueError(
            "NWBFile has no `units` table. Add sorted spikes with "
            "`nwbfile.add_unit(spike_times=...)` before reading."
        )

    ids = np.asarray(units.id.data[:])
    if unit_ids is None:
        rows: list[int] = list(range(len(ids)))
        out_ids = ids
    else:
        # `unit_ids` are matched against the table's id values ONLY. Resolve
        # each to its row; an id with no match is a hard error (do not fall
        # back to interpreting it as a row index, which would silently return
        # the wrong unit on a typo).
        rows = []
        missing: list[int] = []
        for u in unit_ids:
            match = np.flatnonzero(ids == u)
            if match.size == 0:
                missing.append(u)
            else:
                rows.append(int(match[0]))
        if missing:
            raise ValueError(
                f"unit_ids not found in the units table: {missing}. "
                f"Available ids: {ids.tolist()}."
            )
        out_ids = ids[rows]

    # pynwb exposes ragged columns via row-then-column indexing
    # (`units[row, "spike_times"]`), returning one neuron's spike-time list.
    spike_trains = [
        np.sort(np.asarray(units[i, "spike_times"], dtype=np.float64)) for i in rows
    ]
    return spike_trains, out_ids
