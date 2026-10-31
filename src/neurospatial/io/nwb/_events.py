"""
Events and intervals reading and writing for NWB files.

Point events (rewards, region crossings) are stored in ndx-events
EventsTables; intervals (trials, laps) in NWB TimeIntervals tables.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, get_args, get_type_hints

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from neurospatial._keyword_hint import keyword_hint
from neurospatial.io.nwb._adapters import events_table_to_dataframe

if TYPE_CHECKING:
    import ndx_events
    from pynwb import NWBFile

    from neurospatial.behavior.segmentation import Crossing, Lap, Outcome, Trial

# =============================================================================
# Constants for NWB events
# =============================================================================

# Default processing module for behavioral data
DEFAULT_PROCESSING_MODULE: str = "behavior"

# Default names for event tables
DEFAULT_LAPS_NAME: str = "laps"
DEFAULT_REGION_CROSSINGS_NAME: str = "region_crossings"
DEFAULT_TRIALS_NAME: str = "trials"

# Predefined NWB interval table names
PREDEFINED_TRIALS: str = "trials"
PREDEFINED_EPOCHS: str = "epochs"
PREDEFINED_INVALID_TIMES: str = "invalid_times"
_PREDEFINED_INTERVALS: tuple[str, ...] = (
    PREDEFINED_TRIALS,
    PREDEFINED_EPOCHS,
    PREDEFINED_INVALID_TIMES,
)

_LAP_DIRECTION_DESCRIPTION = (
    "Direction of the lap: clockwise, counter-clockwise or unknown"
)
_REGION_DESCRIPTION = "Name of the region crossed"
_CROSSING_DIRECTION_DESCRIPTION = "Direction of the crossing: entry or exit"
_BIN_INDEX_DESCRIPTION = "Environment bin index of the sample at the crossing time"
_OVERLAP_SCORE_DESCRIPTION = (
    "Jaccard overlap of the lap's bins with the template lap's bins, in [0, 1]; "
    "NaN when laps were detected without a template (method='region')"
)


def _build_events_table(
    name: str,
    description: str,
    timestamps: NDArray[np.float64],
    columns: dict[str, tuple[str, NDArray[np.generic]]],
) -> ndx_events.EventsTable:
    """Build an EventsTable column by column.

    Each column is stored from a typed array, so a table with no events keeps
    its columns and dtypes when written to disk.

    Parameters
    ----------
    name, description : str
        Name and description of the table.
    timestamps : NDArray[np.float64], shape (n_events,)
        Event times in seconds from session start.
    columns : dict[str, tuple[str, NDArray]]
        Extra columns, each ``name -> (description, values)`` with values of
        shape ``(n_events,)``.
    """
    from hdmf.common import VectorData

    from neurospatial.io.nwb._core import _require_ndx_events

    ndx_events_module = _require_ndx_events()
    timestamp = ndx_events_module.TimestampVectorData(
        name="timestamp",
        description="Time of each event, in seconds from the session start.",
        data=np.array(timestamps, dtype=np.float64),
    )
    return ndx_events_module.EventsTable(
        name=name,
        description=description,
        columns=[
            timestamp,
            *(
                VectorData(name=column, description=text, data=values)
                for column, (text, values) in columns.items()
            ),
        ],
    )


def _add_events_table(
    nwbfile: NWBFile,
    events_table: ndx_events.EventsTable,
    *,
    processing_module: str,
    module_description: str,
    overwrite: bool,
) -> None:
    """Add a built events table to a processing module, replacing on request.

    The table is built and validated by the caller first, so a failed write
    never removes the table it was meant to replace.
    """
    from neurospatial.io.nwb._core import (
        _get_or_create_processing_module,
        _refuse_in_place_replacement,
        logger,
    )

    name = events_table.name
    module = _get_or_create_processing_module(
        nwbfile, processing_module, module_description
    )
    if name in module.data_interfaces:
        if not overwrite:
            raise ValueError(
                f"EventsTable '{name}' already exists in processing/"
                f"{processing_module}. Use overwrite=True to replace it."
            )
        _refuse_in_place_replacement(nwbfile, f"events table '{name}'")
        del module.data_interfaces[name]
        logger.info("Overwriting existing EventsTable '%s'", name)
    module.add(events_table)


@keyword_hint("table_name", "processing_module")
def read_events(
    nwbfile: NWBFile,
    *,
    table_name: str,
    processing_module: str = DEFAULT_PROCESSING_MODULE,
) -> pd.DataFrame:
    """
    Read events table from NWB file.

    Events are explicitly named (no auto-discovery) because multiple event
    tables are common and have varying semantic meanings.

    Parameters
    ----------
    nwbfile : NWBFile
        The NWB file to read from.
    table_name : str
        Name of the EventsTable to read.
    processing_module : str, default "behavior"
        Name of the processing module containing the EventsTable.

    Returns
    -------
    DataFrame
        Events data with timestamp column and any additional columns.

    Raises
    ------
    KeyError
        If EventsTable not found in specified location.
    TypeError
        If the specified table is not an EventsTable.
    ImportError
        If ndx-events is not installed.

    Examples
    --------
    >>> from pynwb import NWBHDF5IO  # doctest: +SKIP
    >>> with NWBHDF5IO("session.nwb", "r") as io:  # doctest: +SKIP
    ...     nwbfile = io.read()
    ...     rewards = read_events(nwbfile, table_name="rewards")
    ...     print(f"Found {len(rewards)} reward events")
    """
    from neurospatial.io.nwb._core import _require_ndx_events, logger

    # Verify ndx-events is installed BEFORE importing EventsTable
    # This ensures users see the friendly error message
    ndx_events_module = _require_ndx_events()

    # Check if processing module exists
    if processing_module not in nwbfile.processing:
        raise KeyError(
            f"Processing module '{processing_module}' not found in NWB file. "
            f"Available modules: {list(nwbfile.processing.keys())}"
        )

    module = nwbfile.processing[processing_module]

    # Check if table exists in module
    if table_name not in module.data_interfaces:
        available = list(module.data_interfaces.keys())
        raise KeyError(
            f"EventsTable '{table_name}' not found in processing/{processing_module}. "
            f"Available tables: {available}"
        )

    events_table = module.data_interfaces[table_name]

    # Validate it's an EventsTable
    if not isinstance(events_table, ndx_events_module.EventsTable):
        raise TypeError(
            f"'{table_name}' is not an EventsTable (got {type(events_table).__name__}). "
            "Use read_events() only for EventsTable containers."
        )

    logger.debug(
        "Reading EventsTable '%s' from processing/%s", table_name, processing_module
    )

    # Convert to validated DataFrame via adapter
    return events_table_to_dataframe(events_table, table_name=table_name)


@keyword_hint("interval_name")
def read_intervals(
    nwbfile: NWBFile,
    *,
    interval_name: str,
) -> pd.DataFrame:
    """
    Read time intervals from NWB file.

    Reads TimeIntervals tables from the NWB file, including predefined tables
    (trials, epochs, invalid_times) and custom interval tables.

    Parameters
    ----------
    nwbfile : NWBFile
        The NWB file to read from.
    interval_name : str
        Name of the TimeIntervals table to read. Common names include:
        - "trials" : Trial intervals (predefined)
        - "epochs" : Epoch intervals (predefined)
        - "invalid_times" : Invalid time periods (predefined)
        - Custom names for user-defined interval tables

    Returns
    -------
    DataFrame
        Intervals data with start_time, stop_time, and any additional columns.

    Raises
    ------
    KeyError
        If TimeIntervals table not found in NWB file.

    Notes
    -----
    Unlike read_events() which reads point events from ndx-events EventsTable,
    this function reads interval data with start/stop times from the built-in
    NWB TimeIntervals type. No extension dependencies required.

    Examples
    --------
    >>> from pynwb import NWBHDF5IO  # doctest: +SKIP
    >>> with NWBHDF5IO("session.nwb", "r") as io:  # doctest: +SKIP
    ...     nwbfile = io.read()
    ...     trials = read_intervals(nwbfile, interval_name="trials")
    ...     print(f"Found {len(trials)} trials")
    """
    from neurospatial.io.nwb._core import _require_pynwb, logger

    # Verify pynwb is installed
    _require_pynwb()

    # The predefined tables have their own attributes; a file read from disk
    # also lists them in nwbfile.intervals, so each name is kept once.
    tables = {
        name: table
        for name, table in (
            (PREDEFINED_TRIALS, nwbfile.trials),
            (PREDEFINED_EPOCHS, nwbfile.epochs),
            (PREDEFINED_INVALID_TIMES, nwbfile.invalid_times),
        )
        if table is not None
    }
    for name, table in nwbfile.intervals.items():
        tables.setdefault(name, table)

    if interval_name not in tables:
        raise KeyError(
            f"TimeIntervals '{interval_name}' not found in NWB file. "
            f"Available intervals: {list(tables)}"
        )
    logger.debug("Reading TimeIntervals '%s' from NWB file", interval_name)
    return tables[interval_name].to_dataframe()


def _records(
    records: object,
    record_class: type,
    argument: str,
    *,
    arrays: str = "start_times/stop_times",
) -> list[Any]:
    """Return ``records`` as a list after checking every element's type."""
    if not isinstance(records, (list, tuple)):
        raise TypeError(
            f"{argument} must be a list of {record_class.__name__} objects; got "
            f"{type(records).__name__}."
        )
    for i, record in enumerate(records):
        if not isinstance(record, record_class):
            raise TypeError(
                f"{argument}[{i}] is a {type(record).__name__}, not a "
                f"{record_class.__name__}. Pass the list returned by the detector, "
                f"or the {arrays} arrays instead."
            )
    return list(records)


def _refuse_records_with_arrays(
    argument: str, arrays: dict[str, object | None]
) -> None:
    """Refuse array arguments passed together with record objects."""
    passed = [name for name, value in arrays.items() if value is not None]
    if passed:
        raise ValueError(
            f"Cannot combine '{argument}' with {', '.join(passed)}.\n"
            f"Why: the {argument} objects already carry every column, so the "
            "arrays would be ignored.\n"
            f"Fix: pass either '{argument}' or the arrays, not both."
        )


def _checked_intervals(
    start_times: object, stop_times: object, *, record: str
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return start and stop times as validated 1-D float arrays (copies)."""
    start = np.array(start_times, dtype=np.float64)
    stop = np.array(stop_times, dtype=np.float64)
    for argument, values in (("start_times", start), ("stop_times", stop)):
        if values.ndim != 1:
            raise ValueError(
                f"{argument} must be a 1-D array of times in seconds; got shape "
                f"{values.shape}."
            )
    if len(stop) != len(start):
        raise ValueError(
            f"stop_times length ({len(stop)}) must match start_times length "
            f"({len(start)})."
        )
    for argument, values in (("start_times", start), ("stop_times", stop)):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{argument} contains non-finite values (NaN or Inf)")
        if np.any(values < 0):
            raise ValueError(f"{argument} contains negative timestamps")
    backwards = np.flatnonzero(stop < start)
    if backwards.size:
        i = backwards[0]
        raise ValueError(
            f"stop_time must be >= start_time for every {record}. "
            f"{record.capitalize()} {i}: start_time={start[i]}, stop_time={stop[i]}"
        )
    return start, stop


def _per_record(
    values: Sequence[Any] | None, argument: str, n: int, reference: str = "start_times"
) -> list[Any]:
    """Return one value per record, checking the length against ``reference``."""
    if values is None:
        return []
    if isinstance(values, str):
        raise TypeError(
            f"{argument} must be a sequence with one value per record, not the "
            f"single string {values!r}; a string would be split into its "
            f"characters. Pass [{values!r}] * {n} to use it for every record."
        )
    values = list(values)
    if len(values) != n:
        raise ValueError(
            f"{argument} length ({len(values)}) must match {reference} length ({n})."
        )
    return values


def _refuse_unknown(
    values: list[Any],
    allowed: tuple[str, ...],
    *,
    argument: str,
    what: str,
    why: str,
) -> None:
    """Refuse a value outside the vocabulary of a record field."""
    bad = [value for value in values if value not in allowed]
    if bad:
        raise ValueError(
            f"{argument} contains {bad[0]!r}, which is not a {what}.\n"
            f"Why: {why}\n"
            f"Fix: use one of {allowed}."
        )


def _build_time_intervals(
    name: str,
    description: str,
    start_times: NDArray[np.float64],
    stop_times: NDArray[np.float64],
    columns: dict[str, tuple[str, NDArray[np.generic]]],
) -> Any:
    """Build a TimeIntervals table column by column.

    Each column is stored from a typed array, so a table with no intervals
    keeps its columns and dtypes when written to disk.

    Parameters
    ----------
    name, description : str
        Name and description of the table.
    start_times, stop_times : NDArray[np.float64], shape (n_intervals,)
        Interval bounds in seconds from session start.
    columns : dict[str, tuple[str, NDArray]]
        Extra columns, each ``name -> (description, values)`` with values of
        shape ``(n_intervals,)``.

    Returns
    -------
    pynwb.epoch.TimeIntervals
    """
    from hdmf.common import VectorData
    from pynwb.epoch import TimeIntervals

    return TimeIntervals(
        name=name,
        description=description,
        columns=[
            VectorData(
                name="start_time",
                description="Start time of the interval, in seconds",
                data=start_times,
            ),
            VectorData(
                name="stop_time",
                description="Stop time of the interval, in seconds",
                data=stop_times,
            ),
            *(
                VectorData(name=column, description=text, data=values)
                for column, (text, values) in columns.items()
            ),
        ],
    )


@keyword_hint("description")
def write_laps(
    nwbfile: NWBFile,
    laps: Sequence[Lap] | None = None,
    *,
    start_times: NDArray[np.float64] | None = None,
    stop_times: NDArray[np.float64] | None = None,
    directions: Sequence[str] | None = None,
    name: str = DEFAULT_LAPS_NAME,
    description: str = "Detected laps",
    overwrite: bool = False,
) -> None:
    """
    Write laps to an NWB TimeIntervals table.

    Each lap is an interval with a start and stop time, so laps are stored
    as a ``TimeIntervals`` table under ``/intervals/<name>``, next to the
    trials table. Read them back with ``read_intervals(nwbfile,
    interval_name="laps")``; pynapple loads the table as an ``IntervalSet``.

    Parameters
    ----------
    nwbfile : NWBFile
        The NWB file to write to.
    laps : Sequence[Lap], optional
        Laps from :func:`~neurospatial.behavior.segmentation.detect_laps`.
        Writes the ``direction`` and ``overlap_score`` of every lap. Cannot be
        combined with the array arguments.
    start_times : NDArray[np.float64], shape (n_laps,), optional
        Lap start times in seconds. Required if ``laps`` is not given.
    stop_times : NDArray[np.float64], shape (n_laps,), optional
        Lap end times in seconds, each at or after its start time. Required if
        ``laps`` is not given.
    directions : Sequence[str], shape (n_laps,), optional
        Direction of each lap: ``"clockwise"``, ``"counter-clockwise"`` or
        ``"unknown"``, as in :class:`~neurospatial.behavior.segmentation.Lap`.
        Stored as the ``direction`` column.
    name : str, default "laps"
        Name of the table. ``"trials"``, ``"epochs"`` and ``"invalid_times"``
        are reserved by NWB.
    description : str, default "Detected laps"
        Description of the table.
    overwrite : bool, default False
        If True, replace an existing table with the same name. Replacing a
        table in a file opened with ``"r+"`` or ``"a"`` is refused; open it
        with ``"r"`` and export a copy instead.

    Raises
    ------
    ValueError
        If ``laps`` is combined with array arguments, or neither is given.
        If the arrays are not 1-D, their lengths differ, or a direction is
        not a lap direction.
        If times are non-finite or negative, or a lap stops before it starts.
        If ``name`` is reserved by NWB, or a table with that name exists and
        ``overwrite=False``.
    TypeError
        If ``laps`` is not a list of ``Lap`` objects, or ``directions`` is a
        single string.
    NotImplementedError
        If ``overwrite=True`` would replace a table in a file opened with
        ``"r+"`` or ``"a"``.
    ImportError
        If pynwb is not installed.

    See Also
    --------
    detect_laps : Detect laps on a circular track.
    read_intervals : Read the laps back as a DataFrame.
    write_trials : Write trials to the predefined NWB trials table.

    Examples
    --------
    >>> from pynwb import NWBHDF5IO  # doctest: +SKIP
    >>> from neurospatial.behavior import detect_laps  # doctest: +SKIP
    >>> laps = detect_laps(position_bins, times, env)  # doctest: +SKIP
    >>> with NWBHDF5IO("session.nwb", "r+") as io:  # doctest: +SKIP
    ...     nwbfile = io.read()
    ...     write_laps(nwbfile, laps)
    ...     io.write(nwbfile)
    >>> with NWBHDF5IO("session.nwb", "r") as io:  # doctest: +SKIP
    ...     laps_table = read_intervals(io.read(), interval_name="laps")
    """
    from neurospatial.behavior.segmentation import Lap
    from neurospatial.io.nwb._core import (
        _refuse_in_place_replacement,
        _require_pynwb,
        logger,
    )

    _require_pynwb()
    if name in _PREDEFINED_INTERVALS:
        raise ValueError(
            f"write_laps cannot use name={name!r}: NWB reserves it for the "
            "predefined table of that name. Use another name, such as 'laps'."
        )

    columns: dict[str, tuple[str, NDArray[np.generic]]] = {}
    if laps is not None:
        _refuse_records_with_arrays(
            "laps",
            {
                "start_times": start_times,
                "stop_times": stop_times,
                "directions": directions,
            },
        )
        records = _records(laps, Lap, "laps")
        start, stop = _checked_intervals(
            [lap.start_time for lap in records],
            [lap.end_time for lap in records],
            record="lap",
        )
        columns["direction"] = (
            _LAP_DIRECTION_DESCRIPTION,
            np.array([lap.direction for lap in records], dtype=str),
        )
        columns["overlap_score"] = (
            _OVERLAP_SCORE_DESCRIPTION,
            np.array([lap.overlap_score for lap in records], dtype=np.float64),
        )
    else:
        if start_times is None or stop_times is None:
            raise ValueError(
                "write_laps needs either 'laps' (from detect_laps) or both "
                "'start_times' and 'stop_times'."
            )
        start, stop = _checked_intervals(start_times, stop_times, record="lap")
        if directions is not None:
            directions_list = _per_record(directions, "directions", len(start))
            _refuse_unknown(
                directions_list,
                get_args(get_type_hints(Lap)["direction"]),
                argument="directions",
                what="lap direction",
                why="the direction column uses the vocabulary of Lap.direction.",
            )
            columns["direction"] = (
                _LAP_DIRECTION_DESCRIPTION,
                np.array(directions_list, dtype=str),
            )

    table = _build_time_intervals(name, description, start, stop, columns)

    if name in nwbfile.intervals:
        if not overwrite:
            raise ValueError(
                f"TimeIntervals '{name}' already exists. Use overwrite=True to "
                "replace it."
            )
        _refuse_in_place_replacement(nwbfile, f"laps table '{name}'")
        del nwbfile.intervals[name]
        logger.info("Overwriting existing TimeIntervals '%s'", name)
    nwbfile.add_time_intervals(table)
    logger.debug("Wrote TimeIntervals '%s' with %d laps", name, len(start))


@keyword_hint("description")
def write_region_crossings(
    nwbfile: NWBFile,
    crossings: Sequence[Crossing] | None = None,
    *,
    times: NDArray[np.float64] | None = None,
    regions: Sequence[str] | None = None,
    directions: Sequence[str] | None = None,
    name: str = DEFAULT_REGION_CROSSINGS_NAME,
    description: str = "Region crossing events",
    overwrite: bool = False,
) -> None:
    """
    Write region crossings to an NWB EventsTable.

    Each crossing is a point event, so crossings are stored as an ndx-events
    ``EventsTable`` in the ``processing/behavior`` module, with ``region`` and
    ``direction`` columns. Read them back with ``read_events(nwbfile,
    table_name="region_crossings")``.

    Parameters
    ----------
    nwbfile : NWBFile
        The NWB file to write to.
    crossings : Sequence[Crossing], optional
        Crossings from
        :func:`~neurospatial.behavior.segmentation.detect_region_crossings`.
        Writes the ``region``, ``direction`` and ``bin_index`` of every
        crossing. Cannot be combined with the array arguments.
    times : NDArray[np.float64], shape (n_crossings,), optional
        Crossing times in seconds. Required if ``crossings`` is not given.
    regions : Sequence[str], shape (n_crossings,), optional
        Name of the region crossed. Required if ``crossings`` is not given.
    directions : Sequence[str], shape (n_crossings,), optional
        ``"entry"`` or ``"exit"`` for each crossing, as in
        :class:`~neurospatial.behavior.segmentation.Crossing`. Required if
        ``crossings`` is not given.
    name : str, default "region_crossings"
        Name of the EventsTable.
    description : str, default "Region crossing events"
        Description of the EventsTable.
    overwrite : bool, default False
        If True, replace an existing table with the same name. Replacing a
        table in a file opened with ``"r+"`` or ``"a"`` is refused; open it
        with ``"r"`` and export a copy instead.

    Raises
    ------
    ValueError
        If ``crossings`` is combined with array arguments, or neither is given.
        If ``times`` is not 1-D, the arrays' lengths differ, or a direction is
        not ``"entry"`` or ``"exit"``.
        If times are non-finite or negative.
        If a table with the same name exists and ``overwrite=False``.
    TypeError
        If ``crossings`` is not a list of ``Crossing`` objects, ``regions`` or
        ``directions`` is a single string, or a region is not a string.
    NotImplementedError
        If ``overwrite=True`` would replace a table in a file opened with
        ``"r+"`` or ``"a"``.
    ImportError
        If ndx-events is not installed.

    See Also
    --------
    detect_region_crossings : Detect entries into and exits from a region.
    read_events : Read the crossings back as a DataFrame.

    Examples
    --------
    >>> from pynwb import NWBHDF5IO  # doctest: +SKIP
    >>> from neurospatial.behavior import detect_region_crossings  # doctest: +SKIP
    >>> crossings = detect_region_crossings(  # doctest: +SKIP
    ...     position_bins, times, env, region_name="goal"
    ... )
    >>> with NWBHDF5IO("session.nwb", "r+") as io:  # doctest: +SKIP
    ...     nwbfile = io.read()
    ...     write_region_crossings(nwbfile, crossings)
    ...     io.write(nwbfile)
    """
    from neurospatial.behavior.segmentation import Crossing
    from neurospatial.io.nwb._core import _require_ndx_events, logger

    # Check ndx-events first so users see the install hint.
    _require_ndx_events()

    columns: dict[str, tuple[str, NDArray[np.generic]]] = {}
    if crossings is not None:
        _refuse_records_with_arrays(
            "crossings",
            {"times": times, "regions": regions, "directions": directions},
        )
        records = _records(crossings, Crossing, "crossings", arrays="times")
        crossing_times = np.array([c.time for c in records], dtype=np.float64)
        regions_list = [c.region for c in records]
        directions_list: list[Any] = [c.direction for c in records]
        columns["bin_index"] = (
            _BIN_INDEX_DESCRIPTION,
            np.array([c.bin_index for c in records], dtype=np.int64),
        )
    else:
        if times is None or regions is None or directions is None:
            raise ValueError(
                "write_region_crossings needs either 'crossings' (from "
                "detect_region_crossings) or all of 'times', 'regions' and "
                "'directions'."
            )
        crossing_times = np.array(times, dtype=np.float64)
        if crossing_times.ndim != 1:
            raise ValueError(
                "times must be a 1-D array of crossing times in seconds; got "
                f"shape {crossing_times.shape}."
            )
        regions_list = _per_record(regions, "regions", len(crossing_times), "times")
        directions_list = _per_record(
            directions, "directions", len(crossing_times), "times"
        )
        _refuse_unknown(
            directions_list,
            get_args(get_type_hints(Crossing)["direction"]),
            argument="directions",
            what="crossing direction",
            why="the direction column uses the vocabulary of Crossing.direction.",
        )
    missing = [
        i for i, region in enumerate(regions_list) if not isinstance(region, str)
    ]
    if missing:
        raise TypeError(
            f"Crossing {missing[0]} has region {regions_list[missing[0]]!r}; every "
            "crossing needs the region name as a string."
        )
    if not np.all(np.isfinite(crossing_times)):
        raise ValueError("times contains non-finite values (NaN or Inf)")
    if np.any(crossing_times < 0):
        raise ValueError("times contains negative timestamps")

    events_table = _build_events_table(
        name,
        description,
        crossing_times,
        {
            "region": (_REGION_DESCRIPTION, np.array(regions_list, dtype=str)),
            "direction": (
                _CROSSING_DIRECTION_DESCRIPTION,
                np.array(directions_list, dtype=str),
            ),
            **columns,
        },
    )
    _add_events_table(
        nwbfile,
        events_table,
        processing_module=DEFAULT_PROCESSING_MODULE,
        module_description="Behavioral data including region crossings and events",
        overwrite=overwrite,
    )
    logger.debug(
        "Wrote EventsTable '%s' with %d region crossing events",
        name,
        len(crossing_times),
    )


_OUTCOME_DESCRIPTION = (
    "How the trial ended: reached (end region entered), timeout, returned "
    "(back to the start region) or censored (cut off by the recording)"
)


def write_trials(
    nwbfile: NWBFile,
    trials: Sequence[Trial] | None = None,
    *,
    start_times: NDArray[np.float64] | None = None,
    stop_times: NDArray[np.float64] | None = None,
    start_regions: Sequence[str] | None = None,
    end_regions: Sequence[str | None] | None = None,
    outcomes: Sequence[Outcome] | None = None,
    description: str = "Behavioral trials",
    overwrite: bool = False,
) -> None:
    """
    Write trial data to NWB file.

    Writes trial intervals to the built-in NWB trials table, including
    optional columns for start/end regions and how each trial ended. Zero
    trials write an empty table, so a session without trials reads back as
    an empty DataFrame rather than a missing table.

    Parameters
    ----------
    nwbfile : NWBFile
        The NWB file to write to.
    trials : Sequence[Trial], optional
        Trials from :func:`~neurospatial.behavior.segmentation.segment_trials`.
        Writes the ``start_region``, ``end_region`` and ``outcome`` of every
        trial. Cannot be combined with the array arguments.
    start_times : NDArray[np.float64], shape (n_trials,), optional
        Trial start times in seconds. Required if ``trials`` is not given.
    stop_times : NDArray[np.float64], shape (n_trials,), optional
        Trial stop times in seconds. Required if ``trials`` is not given.
    start_regions : Sequence[str], shape (n_trials,), optional
        Start region name of each trial.
    end_regions : Sequence[str or None], shape (n_trials,), optional
        End region reached by each trial; None (no region reached) is stored
        as ``""``.
    outcomes : Sequence[Outcome], shape (n_trials,), optional
        How each trial ended: ``"reached"``, ``"timeout"``, ``"returned"`` or
        ``"censored"`` (see :class:`~neurospatial.behavior.segmentation.Trial`).
    description : str, default "Behavioral trials"
        Description of the trials table.
    overwrite : bool, default False
        If True, replace an existing trials table in an ``NWBFile`` built in
        memory. A file read from disk keeps its written trials table: pynwb
        cannot remove it, and an export would silently keep the old trials,
        so replacing it raises ``NotImplementedError``.

    Raises
    ------
    ValueError
        If ``trials`` is combined with array arguments, or neither
        ``trials`` nor both ``start_times`` and ``stop_times`` are given.
        If array lengths don't match, or an outcome is not one of the four.
        If stop_times < start_times for any trial.
        If timestamps contain non-finite or negative values.
        If a trials table already exists and ``overwrite=False``, or another
        table refers to the rows of the trials table being replaced.
    TypeError
        If ``trials`` is not a list of ``Trial`` objects, or a per-trial
        argument is a single string.
    NotImplementedError
        If ``overwrite=True`` would replace the trials table of a file read
        from disk.
    ImportError
        If pynwb is not installed.

    Notes
    -----
    The trials table is stored in NWB's built-in trials location
    (``nwbfile.trials`` / ``/intervals/trials/``).

    Custom columns are added for:
    - ``start_region``: Region where trial started
    - ``end_region``: Region reached (or "" when the trial did not reach one)
    - ``outcome``: How the trial ended (``"reached"``, ``"timeout"``,
      ``"returned"`` or ``"censored"``). A successful trial is
      ``outcome == "reached"``; a censored trial was cut off by the recording,
      not failed.

    Examples
    --------
    >>> from pynwb import NWBHDF5IO  # doctest: +SKIP
    >>> from neurospatial.behavior.segmentation import segment_trials  # doctest: +SKIP
    >>> trials = segment_trials(bins, times, env, ...)  # doctest: +SKIP
    >>> with NWBHDF5IO("session.nwb", "r+") as io:  # doctest: +SKIP
    ...     nwbfile = io.read()
    ...     write_trials(nwbfile, trials)
    ...     io.write(nwbfile)

    See Also
    --------
    read_trials : Read trials from NWB file.
    read_intervals : Read generic TimeIntervals.
    segment_trials : Segment trajectory into trials.
    """
    from neurospatial.behavior.segmentation import Outcome, Trial
    from neurospatial.io.nwb._core import _require_pynwb, logger

    _require_pynwb()

    columns: dict[str, tuple[str, NDArray[np.generic]]] = {}
    if trials is not None:
        _refuse_records_with_arrays(
            "trials",
            {
                "start_times": start_times,
                "stop_times": stop_times,
                "start_regions": start_regions,
                "end_regions": end_regions,
                "outcomes": outcomes,
            },
        )
        records = _records(trials, Trial, "trials")
        start, stop = _checked_intervals(
            [trial.start_time for trial in records],
            [trial.end_time for trial in records],
            record="trial",
        )
        start_regions_list = [trial.start_region for trial in records]
        end_regions_list = [trial.end_region for trial in records]
        outcomes_list = [trial.outcome for trial in records]
    else:
        if start_times is None or stop_times is None:
            raise ValueError(
                "write_trials needs either 'trials' (from segment_trials) or "
                "both 'start_times' and 'stop_times'."
            )
        start, stop = _checked_intervals(start_times, stop_times, record="trial")
        start_regions_list = _per_record(start_regions, "start_regions", len(start))
        end_regions_list = _per_record(end_regions, "end_regions", len(start))
        outcomes_list = _per_record(outcomes, "outcomes", len(start))
        _refuse_unknown(
            outcomes_list,
            get_args(Outcome),
            argument="outcomes",
            what="trial outcome",
            why="the outcome column records how each trial ended.",
        )
    if trials is not None or start_regions is not None:
        columns["start_region"] = (
            "Region where trial started",
            np.array(start_regions_list, dtype=str),
        )
    if trials is not None or end_regions is not None:
        columns["end_region"] = (
            "Region reached (empty string if none was reached)",
            np.array(
                ["" if region is None else region for region in end_regions_list],
                dtype=str,
            ),
        )
    if trials is not None or outcomes is not None:
        columns["outcome"] = (
            _OUTCOME_DESCRIPTION,
            np.array(outcomes_list, dtype=str),
        )

    table = _build_time_intervals(PREDEFINED_TRIALS, description, start, stop, columns)
    if nwbfile.trials is not None:
        if not overwrite:
            raise ValueError(
                "A trials table already exists in this NWB file. Pass "
                "overwrite=True to replace it."
            )
        if getattr(nwbfile, "read_io", None) is not None:
            raise NotImplementedError(
                "Cannot replace the trials table of a file read from disk: "
                "pynwb cannot remove a written trials table, and exporting the "
                "file would silently keep the old trials. Build a new NWBFile "
                "and write the trials there, or store these intervals under "
                "another name with nwbfile.add_time_intervals(...) (read back "
                "with read_intervals)."
            )
        old = nwbfile.trials
        referrers = sorted(
            f"column {obj.name!r} of {getattr(obj.parent, 'name', None)!r}"
            for obj in nwbfile.all_children()
            if getattr(obj, "table", None) is old
        )
        if referrers:
            raise ValueError(
                f"Cannot replace the trials table: {', '.join(referrers)} "
                "refers to its rows and would point at a table that is no "
                "longer in the file. Build a new NWBFile, or remove those "
                "references first."
            )
        # The trials attribute can be set only once; clearing the field and
        # detaching the old table is the only way to replace it in memory.
        nwbfile.fields.pop(PREDEFINED_TRIALS)
        old.reset_parent()
        logger.info("Overwriting existing trials table")
    nwbfile.trials = table
    if overwrite:
        # all_children() rebuilt pynwb's object cache with the old table in it.
        nwbfile.all_children()
    logger.debug("Wrote %d trials to NWB file", len(start))


def read_trials(nwbfile: NWBFile) -> pd.DataFrame:
    """
    Read trials table from NWB file.

    Convenience wrapper around read_intervals("trials") that provides
    a more discoverable API for reading trial data.

    Parameters
    ----------
    nwbfile : NWBFile
        The NWB file to read from.

    Returns
    -------
    pd.DataFrame
        Trials data with columns: start_time, stop_time, and any custom
        columns (start_region, end_region, outcome, etc.).

    Raises
    ------
    KeyError
        If no trials table exists in NWB file.

    Examples
    --------
    >>> from pynwb import NWBHDF5IO  # doctest: +SKIP
    >>> with NWBHDF5IO("session.nwb", "r") as io:  # doctest: +SKIP
    ...     nwbfile = io.read()
    ...     trials = read_trials(nwbfile)
    ...     print(f"Found {len(trials)} trials")

    See Also
    --------
    write_trials : Write trials to NWB file.
    read_intervals : Read generic TimeIntervals.
    """
    return read_intervals(nwbfile, interval_name="trials")


@keyword_hint("name", "description")
def dataframe_to_events_table(
    df: pd.DataFrame,
    *,
    name: str,
    description: str = "Events",
) -> ndx_events.EventsTable:
    """
    Convert a pandas DataFrame to an ndx-events EventsTable.

    This function creates an EventsTable with the same columns as the input
    DataFrame. The DataFrame must have a 'timestamp' column.

    Parameters
    ----------
    df : pd.DataFrame
        Events DataFrame. Must have a 'timestamp' column with numeric values.
        Additional columns become EventsTable columns.
    name : str
        Name for the EventsTable.
    description : str, default "Events"
        Description for the EventsTable.

    Returns
    -------
    ndx_events.EventsTable
        EventsTable containing the events data.

    Raises
    ------
    TypeError
        If df is not a DataFrame.
    ValueError
        If DataFrame is missing timestamp column or contains invalid timestamps.
    ImportError
        If ndx-events is not installed.

    Examples
    --------
    >>> import pandas as pd  # doctest: +SKIP
    >>> df = pd.DataFrame(
    ...     {  # doctest: +SKIP
    ...         "timestamp": [1.0, 2.0, 3.0],
    ...         "label": ["a", "b", "c"],
    ...     }
    ... )
    >>> events_table = dataframe_to_events_table(df, name="my_events")  # doctest: +SKIP
    """
    from neurospatial.io.nwb._core import _require_ndx_events

    # Verify ndx-events is installed
    _require_ndx_events()

    # Validate input
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"Expected pd.DataFrame, got {type(df).__name__}.\n"
            "  WHY: Events must be a pandas DataFrame.\n"
            "  Fix: Convert using pd.DataFrame({'timestamp': times})"
        )

    if "timestamp" not in df.columns:
        raise ValueError(
            "DataFrame is missing required 'timestamp' column.\n"
            "  WHY: EventsTable requires timestamp for each event.\n"
            f"  Fix: Add timestamp column. Available columns: {list(df.columns)}"
        )

    timestamps = np.array(df["timestamp"], dtype=np.float64)
    if not np.all(np.isfinite(timestamps)):
        raise ValueError(
            "timestamp column contains non-finite values (NaN or Inf).\n"
            "  WHY: EventsTable requires valid numeric timestamps.\n"
            "  Fix: Remove or replace NaN/Inf values in timestamp column."
        )
    if np.any(timestamps < 0):
        raise ValueError(
            "timestamp column contains negative values.\n"
            "  WHY: NWB timestamps should be non-negative (seconds from session start).\n"
            "  Fix: Adjust timestamps to be relative to session start."
        )

    return _build_events_table(
        name,
        description,
        timestamps,
        {
            str(col): (f"Column: {col}", df[col].to_numpy(copy=True))
            for col in df.columns
            if col != "timestamp"
        },
    )


@keyword_hint("name")
def write_events(
    nwbfile: NWBFile,
    events: pd.DataFrame,
    *,
    name: str,
    description: str = "Event data",
    processing_module: str = DEFAULT_PROCESSING_MODULE,
    overwrite: bool = False,
) -> None:
    """
    Write generic events DataFrame to NWB EventsTable.

    Creates an EventsTable in a processing module containing event timestamps
    and any additional columns from the DataFrame.

    Parameters
    ----------
    nwbfile : NWBFile
        The NWB file to write to.
    events : pd.DataFrame
        Events DataFrame. Must have a 'timestamp' column with numeric values.
        Additional columns (e.g., 'label', 'value', 'x', 'y') are preserved.
    name : str
        Name for the EventsTable in NWB.
    description : str, default "Event data"
        Description for the EventsTable.
    processing_module : str, default "behavior"
        Processing module to store events in.
    overwrite : bool, default False
        If True, replace existing EventsTable with same name.
        If False, raise ValueError on duplicate name.

    Raises
    ------
    TypeError
        If events is not a DataFrame.
    ValueError
        If EventsTable with same name exists and overwrite=False.
        If DataFrame is missing timestamp column.
        If timestamps contain non-finite or negative values.
    ImportError
        If ndx-events is not installed.

    Notes
    -----
    The EventsTable is stored in the specified processing module. Common
    column conventions:

    - ``timestamp``: Required. Event time in seconds from session start.
    - ``label``: String category/type for event.
    - ``value``: Numeric value associated with event.
    - ``x``, ``y``, ``z``: Spatial position columns.

    All additional columns are preserved as EventsTable columns.

    Examples
    --------
    >>> from pynwb import NWBHDF5IO  # doctest: +SKIP
    >>> import pandas as pd  # doctest: +SKIP
    >>> events = pd.DataFrame(
    ...     {  # doctest: +SKIP
    ...         "timestamp": [1.0, 2.5, 5.0],
    ...         "label": ["reward", "lick", "reward"],
    ...         "x": [10.0, 15.0, 20.0],
    ...         "y": [5.0, 8.0, 12.0],
    ...     }
    ... )
    >>> with NWBHDF5IO("session.nwb", "r+") as io:  # doctest: +SKIP
    ...     nwbfile = io.read()
    ...     write_events(nwbfile, events, name="behavioral_events")
    ...     io.write(nwbfile)

    See Also
    --------
    read_events : Read events from NWB EventsTable.
    write_laps : Write laps as an interval table.
    write_region_crossings : Write region crossing events (specialized format).
    """
    from neurospatial.io.nwb._core import logger

    events_table = dataframe_to_events_table(events, name=name, description=description)
    _add_events_table(
        nwbfile,
        events_table,
        processing_module=processing_module,
        module_description=f"{processing_module.capitalize()} data",
        overwrite=overwrite,
    )
    logger.debug(
        "Wrote EventsTable '%s' with %d events to processing/%s",
        name,
        len(events),
        processing_module,
    )
