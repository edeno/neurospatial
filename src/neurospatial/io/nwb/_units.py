"""
Spike-time reading from an NWB ``units`` table.

This module provides :func:`read_units`, which extracts per-neuron spike-time
arrays, unit ids, observation intervals and per-unit metadata from the ragged
``units`` :class:`~hdmf.common.table.DynamicTable` of an NWB file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error
from neurospatial.io.nwb._core import _require_pynwb
from neurospatial.io.nwb._holders import NWBUnits

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pynwb import NWBFile


class _LazyUnitSpikeTrain:
    """Lazy per-unit spike-time handle backed by the open NWB ``units`` table.

    Wraps a ``(units, row)`` pair and defers reading that unit's ragged
    spike-time slice until it is materialized via ``np.asarray`` or indexing.
    On materialization it returns the same sorted ``float64`` array the eager
    path produces, reading only the requested unit's slice from the h5py-backed
    ragged column. Valid only while the backing ``NWBFile`` is open.

    The materialized array is **memoized** on first access: the ragged slice is
    read and sorted once, then cached read-only and reused on every subsequent
    ``__getitem__`` / ``__len__`` / ``np.asarray``, so repeated access does not
    re-read and re-sort the whole unit off disk. ``np.array(train)`` or
    ``np.asarray(train, copy=True)`` returns a writable copy.
    """

    __slots__ = ("_cache", "_row", "_units")

    def __init__(self, units: Any, row: int) -> None:
        self._units = units
        self._row = row
        # ``None`` is the "not yet materialized" sentinel; a materialized train
        # is always an ndarray (empty for a unit with no spikes), never None.
        self._cache: NDArray[np.float64] | None = None

    def _materialize(self) -> NDArray[np.float64]:
        if self._cache is None:
            cache = np.sort(
                np.asarray(self._units[self._row, "spike_times"], dtype=np.float64)
            )
            cache.flags.writeable = False
            self._cache = cache
        return self._cache

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> NDArray[Any]:
        arr = self._materialize()
        if dtype is not None and np.dtype(dtype) != arr.dtype:
            if copy is False:
                raise ValueError(
                    f"Converting spike times to {np.dtype(dtype)} requires a copy."
                )
            return arr.astype(dtype)
        return arr.copy() if copy else arr

    def __getitem__(self, key: Any) -> Any:
        # Indexing may yield an array or a scalar depending on ``key``.
        return self._materialize()[key]

    def __len__(self) -> int:
        return len(self._materialize())


def read_units(
    nwbfile: NWBFile,
    *,
    unit_ids: Sequence[int] | None = None,
    lazy: bool = False,
) -> NWBUnits:
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
    lazy : bool, default False
        If ``False`` (default), each unit's spike train is fully materialized
        into a sorted, read-only ``NDArray[np.float64]`` up front. If ``True``,
        each ``spike_times`` entry is a lazy per-unit handle that reads and sorts only that unit's spike-time slice
        when it is ``np.asarray``-ed or indexed, keeping large recordings
        off-RAM until a unit is actually accessed.

    Returns
    -------
    NWBUnits
        Frozen, non-iterable holder with ``spike_times`` (sorted arrays or lazy
        handles), aligned table ``unit_ids``, per-unit ``obs_intervals`` and
        their shared intersection ``spike_window``, ``unit_table`` and the
        table's spike-time ``resolution`` in seconds (None when the file does
        not state it or stores a value that is not finite and positive, such
        as NaN, 0 or -1). Coverage fields are None when the observation-interval
        column is absent.

        ``unit_table`` is a DataFrame indexed by ``unit_id`` in ``unit_ids``
        order. It holds every Units column with one scalar per unit, such
        as ``location``, ``cell_type`` or ``quality``, verbatim. When units
        link electrodes it adds ``electrode_location``, the brain region from
        the electrodes table ``location`` column, and ``electrode_group``, the
        electrode group name; a unit on electrodes in several regions lists
        each once, joined by ``", "``. Ragged and multi-dimensional columns
        (for example ``waveform_mean``) are not included.

    Raises
    ------
    ValueError
        If the file has no ``units`` table, the table repeats an id, any
        requested ``unit_ids`` value is not present in ``units.id`` or is
        requested twice, or a unit's ``obs_intervals`` are not valid
        (n_intervals, 2) rows.
    ImportError
        If pynwb is not installed.

    Notes
    -----
    Pass ``units.spike_times`` and ``units.unit_ids`` explicitly to population
    analyses. The holder has no tuple-unpacking compatibility interface.

    Pass ``spike_window=units.spike_window`` to analyses to preserve acquisition
    coverage. This window is the time every selected unit was observed; it is
    separate from selected analysis ``epochs``. Read units with different
    coverage in separate calls (``unit_ids=``) if their intersection is too
    short. An empty intersection is stored as shape ``(0, 2)``; passing it to
    an analysis raises its "no rows" window error. Observation intervals are
    read eagerly even when spike arrays are lazy.

    To analyze units from one brain region, build
    ``SpikeTrains(units.spike_times, unit_ids=units.unit_ids,
    unit_table=units.unit_table)`` and select with
    ``.filter("electrode_location == 'CA1'")`` (or the file's own region
    column); encoding results carry the table. The metadata table is read
    eagerly; with ``lazy=True`` only the spike times stay on disk.

    Lazy handles are **only valid while the backing ``NWBFile`` / ``NWBHDF5IO``
    is open**. Materialize them (``np.asarray`` or indexing) inside the
    ``with NWBHDF5IO(...)`` block; ``lazy=False`` returns arrays that remain
    valid after the file closes.

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
    >>> nwbfile.add_unit_column("location", "Brain region of the unit")
    >>> nwbfile.add_unit(spike_times=[0.1, 0.5, 1.2], id=7, location="CA1")
    >>> nwbfile.add_unit(spike_times=[0.3, 0.9], id=11, location="CA3")
    >>> units = read_units(nwbfile)
    >>> units.unit_ids
    array([ 7, 11])
    >>> units.spike_times[0]
    array([0.1, 0.5, 1.2])
    >>> units.unit_table["location"].tolist()
    ['CA1', 'CA3']
    """
    _require_pynwb()

    units = nwbfile.units
    if units is None:
        raise ValueError(
            "NWBFile has no `units` table. Add sorted spikes with "
            "`nwbfile.add_unit(spike_times=...)` before reading."
        )

    ids = np.asarray(units.id.data[:])
    values, counts = np.unique(ids, return_counts=True)
    if np.any(counts > 1):
        raise ValueError(
            _format_error(
                f"The units table repeats id(s) {values[counts > 1].tolist()}.",
                why="Why: unit_ids names table identities, so each id must name "
                "one row.",
                fix="rewrite the units table with one unique id per unit.",
            )
        )
    if unit_ids is None:
        rows: list[int] = list(range(len(ids)))
    else:
        # `unit_ids` are matched against the table's id values ONLY; an id
        # with no match is an error, never reinterpreted as a row index.
        row_of = {int(unit_id): row for row, unit_id in enumerate(ids)}
        missing = [u for u in unit_ids if u not in row_of]
        if missing:
            raise ValueError(
                _format_error(
                    f"unit_ids not found in the units table: {missing}. Available ids: {ids.tolist()}.",
                    fix="pass unit_ids= with existing table ids; inspect np.asarray(nwbfile.units.id[:]) for available ids",
                    why="Why: unit_ids names table identities, not row positions.",
                )
            )
        rows = [row_of[u] for u in unit_ids]

    # pynwb exposes ragged columns via row-then-column indexing
    # (`units[row, "spike_times"]`), returning one neuron's spike-time list.
    spike_trains: list[NDArray[np.float64] | Any]
    if lazy:
        spike_trains = [_LazyUnitSpikeTrain(units, i) for i in rows]
    else:
        spike_trains = [
            np.sort(np.asarray(units[i, "spike_times"], dtype=np.float64)) for i in rows
        ]
    obs_intervals = (
        [units[row, "obs_intervals"] for row in rows]
        if "obs_intervals" in units.colnames
        else None
    )
    return NWBUnits(
        spike_trains,
        ids[rows],
        obs_intervals,
        unit_table=_unit_table(units, rows, ids[rows]),
        resolution=_known_resolution(getattr(units, "resolution", None)),
    )


def _known_resolution(resolution: Any) -> float | None:
    """The stored spike-time resolution, or None when it states no precision.

    The schema has no "unknown" value; files write NaN, 0 or -1 instead, so
    any value that is not finite and positive reads as unknown.
    """
    if resolution is None:
        return None
    value = float(resolution)
    return value if np.isfinite(value) and value > 0 else None


# Columns read into other NWBUnits fields or summarized from the electrodes.
_NOT_METADATA = frozenset(
    {"spike_times", "obs_intervals", "electrodes", "electrode_group"}
)


def _unit_table(units: Any, rows: list[int], unit_ids: NDArray[Any]) -> pd.DataFrame:
    """Per-unit metadata of the selected ``rows``, indexed by ``unit_id``.

    Keeps every Units column holding one scalar per unit (ragged columns,
    references to other tables and multi-dimensional columns such as
    ``waveform_mean`` are skipped) and adds
    ``electrode_location`` and ``electrode_group`` from the unit's electrodes.
    Only these small columns are read; spike times are not touched.

    Parameters
    ----------
    units : pynwb.misc.Units
        The open units table.
    rows : list of int
        Selected table rows, in output order.
    unit_ids : NDArray, shape (n_selected,)
        Table ids of ``rows``.

    Returns
    -------
    pandas.DataFrame
        One row per selected unit, index named ``unit_id``.
    """
    from hdmf.common.table import DynamicTableRegion, VectorIndex

    columns: dict[str, Any] = {}
    for name in units.colnames:
        column = units[name]
        if name in _NOT_METADATA or isinstance(
            column, (VectorIndex, DynamicTableRegion)
        ):
            continue
        values = column.data[:]
        if not all(np.ndim(value) == 0 for value in values):
            continue
        columns[name] = [values[row] for row in rows]
    columns.update(_electrode_columns(units, rows))
    return pd.DataFrame(columns, index=pd.Index(unit_ids, name="unit_id"))


def _electrode_columns(units: Any, rows: list[int]) -> dict[str, list[str | None]]:
    """Brain region and electrode group of each selected unit.

    Regions come from the electrodes table ``location`` column, groups from
    its ``group`` column, for the electrodes the unit's ``electrodes`` row
    lists. Without an ``electrodes`` column, the Units ``electrode_group``
    column gives both (the group name and its ``location``). A unit on
    electrodes in several regions or groups lists each once, in electrode
    order, joined by ``", "``; a unit that lists no electrodes gets None.

    Returns
    -------
    dict
        ``{"electrode_location": [...], "electrode_group": [...]}`` with one
        entry per selected unit, or ``{}`` when the table links no
        electrodes.
    """
    from hdmf.common.table import VectorIndex

    if "electrodes" in units.colnames:
        column = units["electrodes"]
        if isinstance(column, VectorIndex):
            # Ragged: each unit lists several electrodes.
            region = column.target
            ends = np.asarray(column.data[:])
            electrode_rows = np.asarray(region.data[:])
            per_unit = [
                electrode_rows[(ends[row - 1] if row else 0) : ends[row]]
                for row in rows
            ]
        else:
            # Older files store one electrode per unit without an index.
            region = column
            electrode_rows = np.asarray(region.data[:])
            per_unit = [electrode_rows[row : row + 1] for row in rows]
        electrodes = region.table
        groups = electrodes["group"].data[:]
        locations = (
            electrodes["location"].data[:]
            if "location" in electrodes.colnames
            else [group.location for group in groups]
        )
        return {
            "electrode_location": [
                _join_unique(locations[i] for i in channels) for channels in per_unit
            ],
            "electrode_group": [
                _join_unique(groups[i].name for i in channels) for channels in per_unit
            ],
        }
    if "electrode_group" in units.colnames:
        groups = units["electrode_group"].data[:]
        return {
            "electrode_location": [str(groups[row].location) for row in rows],
            "electrode_group": [str(groups[row].name) for row in rows],
        }
    return {}


def _join_unique(values: Any) -> str | None:
    """Distinct values in first-seen order joined by ``", "``; None if empty."""
    return ", ".join(dict.fromkeys(str(value) for value in values)) or None
