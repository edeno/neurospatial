"""
Events and intervals reading and writing for NWB files.

Point events (rewards, region crossings) are stored in ndx-events
EventsTables; intervals (trials, laps) in NWB TimeIntervals tables.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import repeat
from typing import TYPE_CHECKING, Any, Literal, get_args, get_origin, get_type_hints

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error
from neurospatial._keyword_hint import keyword_hint
from neurospatial._pynapple import reads_pynapple_samples
from neurospatial._validation import validate_finite
from neurospatial.io.nwb._adapters import events_table_to_dataframe, missing_error
from neurospatial.io.nwb._core import (
    _clear_for_write,
    _get_or_create_processing_module,
    _require_ndx_events,
    _require_pynwb,
    logger,
)

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

# Predefined NWB interval table names
PREDEFINED_TRIALS: str = "trials"
PREDEFINED_EPOCHS: str = "epochs"
PREDEFINED_INVALID_TIMES: str = "invalid_times"
_PREDEFINED_INTERVALS: tuple[str, ...] = (
    PREDEFINED_TRIALS,
    PREDEFINED_EPOCHS,
    PREDEFINED_INVALID_TIMES,
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
    never removes the table it was meant to replace. A table with no events
    is not added, and no processing module is created for it; with
    ``overwrite=True`` it still removes the table it replaces.
    """

    name = events_table.name
    empty = len(events_table) == 0
    module = (
        nwbfile.processing.get(processing_module)
        if empty
        else _get_or_create_processing_module(
            nwbfile, processing_module, module_description
        )
    )
    if module is not None:
        _clear_for_write(
            nwbfile,
            module.data_interfaces,
            name,
            overwrite=overwrite,
            what=f"events table '{name}' in processing/{processing_module}",
        )
    if empty:
        return
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

    # Verify ndx-events is installed BEFORE importing EventsTable
    # This ensures users see the friendly error message
    ndx_events_module = _require_ndx_events()

    # Check if processing module exists
    if processing_module not in nwbfile.processing:
        msg = (
            f"Processing module '{processing_module}' not found in NWB file. "
            f"Available modules: {list(nwbfile.processing.keys())}."
        )
        raise missing_error(
            msg,
            fix="pass processing_module= with a module from this list.",
        )

    module = nwbfile.processing[processing_module]

    # Check if table exists in module
    if table_name not in module.data_interfaces:
        msg = (
            f"EventsTable '{table_name}' not found in processing/"
            f"{processing_module}. Available tables: "
            f"{list(module.data_interfaces.keys())}."
        )
        raise missing_error(
            msg,
            fix="pass table_name= with a table from this list.",
        )

    events_table = module.data_interfaces[table_name]

    # Validate it's an EventsTable
    if not isinstance(events_table, ndx_events_module.EventsTable):
        msg = (
            f"'{table_name}' is not an EventsTable (got {type(events_table).__name__}). "
            "Use read_events() only for EventsTable containers."
        )
        raise TypeError(msg)

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

    Pass the rows as ``epochs=`` (``df[["start_time", "stop_time"]].to_numpy()``)
    to read each as the half-open window ``[start_time, stop_time)``, the
    meaning :func:`write_trials` and :func:`write_laps` give them. pynapple's
    ``load_file`` reads the same table with inclusive ends; see
    :func:`write_trials` for how to get identical selections on both routes.

    Examples
    --------
    >>> from pynwb import NWBHDF5IO  # doctest: +SKIP
    >>> with NWBHDF5IO("session.nwb", "r") as io:  # doctest: +SKIP
    ...     nwbfile = io.read()
    ...     trials = read_intervals(nwbfile, interval_name="trials")
    ...     print(f"Found {len(trials)} trials")
    """

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
        msg = (
            f"TimeIntervals '{interval_name}' not found in NWB file. "
            f"Available intervals: {list(tables)}."
        )
        raise missing_error(
            msg,
            fix="pass interval_name= with a name from this list.",
        )
    logger.debug("Reading TimeIntervals '%s' from NWB file", interval_name)
    return tables[interval_name].to_dataframe()


def _records(
    records: object,
    record_class: type,
    argument: str,
    *,
    arrays: str,
) -> list[Any]:
    """Return ``records`` as a list after checking every element's type."""
    if not isinstance(records, (list, tuple)):
        msg = (
            f"{argument} must be a list of {record_class.__name__} objects; got "
            f"{type(records).__name__}."
        )
        raise TypeError(msg)
    for i, record in enumerate(records):
        if not isinstance(record, record_class):
            msg = (
                f"{argument}[{i}] is a {type(record).__name__}, not a "
                f"{record_class.__name__}. Pass the list returned by the detector, "
                f"or the {arrays} arrays instead."
            )
            raise TypeError(msg)
    return list(records)


def _refuse_records_with_arrays(
    argument: str, arrays: dict[str, object | None]
) -> None:
    """Refuse array arguments passed together with record objects."""
    passed = [name for name, value in arrays.items() if value is not None]
    if passed:
        raise ValueError(
            _format_error(
                f"Cannot combine '{argument}' with {', '.join(passed)}.",
                why=f"the {argument} objects already carry every column, so the "
                "arrays would be ignored.",
                fix=f"pass either '{argument}' or the arrays, not both.",
            )
        )


def _checked_event_times(values: object, argument: str) -> NDArray[np.float64]:
    """Return ``values`` as a float64 copy after checking they are NWB times.

    NWB times are finite seconds from the session start, so none is negative.
    """
    times = np.array(values, dtype=np.float64)
    if times.ndim != 1:
        msg = (
            f"{argument} must be a 1-D array of times in seconds; got shape "
            f"{times.shape}."
        )
        raise ValueError(msg)
    validate_finite(times, name=argument)
    negative = np.flatnonzero(times < 0)
    if negative.size:
        raise ValueError(
            _format_error(
                f"{argument} contains {negative.size} negative timestamp(s) "
                f"(first at index {negative[0]}: {times[negative[0]]}).",
                why="NWB times are seconds from the session start.",
                fix="subtract the session start time from every time.",
            )
        )
    return times


def _refuse_reversed_intervals(
    start: NDArray[np.float64], stop: NDArray[np.float64], *, record: str
) -> None:
    """Refuse an interval that stops before it starts."""
    backwards = np.flatnonzero(stop < start)
    if backwards.size:
        i = backwards[0]
        msg = (
            f"stop_time must be >= start_time for every {record}. "
            f"{record.capitalize()} {i}: start_time={start[i]}, stop_time={stop[i]}"
        )
        raise ValueError(msg)


@dataclass(frozen=True)
class _Column:
    """A per-record column, read from record objects or from an array argument.

    ``name`` is both the column name and the record attribute that holds it.
    The attribute's annotation on the record class sets what the column
    accepts: a ``Literal`` lists the allowed strings, ``str`` takes any
    string, ``str | None`` also takes None (stored as ``""``), and ``float``
    and ``int`` are stored as float64 and int64.
    """

    name: str
    description: str
    argument: str | None = None
    """Array keyword of the writer; None when only the records carry it."""
    required: bool = False
    """Whether the writer needs the array when it is given no records."""


_INTERVAL_TIMES = {"start_times": "start_time", "stop_times": "end_time"}
_LAP_COLUMNS = (
    _Column(
        "direction",
        "Direction of the lap: clockwise, counter-clockwise or unknown",
        "directions",
    ),
    _Column(
        "overlap_score",
        "Jaccard overlap of the lap's bins with the template lap's bins, in "
        "[0, 1]; NaN for a lap without a template score; the column is left out "
        "when no lap has one (laps detected with method='region')",
    ),
)
_CROSSING_COLUMNS = (
    _Column("region", "Name of the region crossed", "regions", required=True),
    _Column(
        "direction",
        "Direction of the crossing: entry or exit",
        "directions",
        required=True,
    ),
    _Column("bin_index", "Environment bin index of the sample at the crossing time"),
)
_TRIAL_COLUMNS = (
    _Column("start_region", "Region where trial started", "start_regions"),
    _Column(
        "end_region",
        "Region reached (empty string if none was reached)",
        "end_regions",
    ),
    _Column(
        "outcome",
        "How the trial ended: reached (end region entered), timeout, returned "
        "(back to the start region) or censored (cut off by the recording)",
        "outcomes",
    ),
)


def _record_columns(
    record_class: type,
    records: object,
    arrays: dict[str, Any],
    *,
    times: dict[str, str],
    columns: tuple[_Column, ...],
    needs: str,
) -> tuple[list[NDArray[np.float64]], dict[str, tuple[str, NDArray[np.generic]]]]:
    """Return the times and columns of a table, from records or from arrays.

    Parameters
    ----------
    record_class : type
        The record type: ``Lap``, ``Crossing`` or ``Trial``.
    records : sequence of record_class or None
        The writer's records argument; None to read ``arrays`` instead.
    arrays : dict of str to array-like or None
        Every array argument of the writer, by keyword.
    times : dict of str to str
        Array keyword of each time to the record attribute that holds it,
        e.g. ``{"start_times": "start_time", "stop_times": "end_time"}``.
    columns : tuple of _Column
        The other columns, in table order.
    needs : str
        Error message for arrays without every required one.

    Returns
    -------
    times : list of NDArray[np.float64], each shape (n_records,)
        The checked times, in the order of ``times``.
    columns : dict of str to (str, NDArray), values shape (n_records,)
        ``name -> (description, values)`` for each column given.
    """
    record = record_class.__name__.lower()
    attributes = {**times, **{c.argument or c.name: c.name for c in columns}}
    if records is not None:
        _refuse_records_with_arrays(f"{record}s", arrays)
        objects = _records(records, record_class, f"{record}s", arrays="/".join(times))
        given = {
            key: [getattr(o, attribute) for o in objects]
            for key, attribute in attributes.items()
        }
    else:
        required = [*times, *(c.argument or c.name for c in columns if c.required)]
        if any(arrays[key] is None for key in required):
            raise ValueError(needs)
        given = {key: value for key, value in arrays.items() if value is not None}

    reference = next(iter(times))
    checked = [_checked_event_times(given[key], key) for key in times]
    n = len(checked[0])
    for key, values in zip(times, checked, strict=True):
        _refuse_length(len(values), key, n, reference)
    hints = get_type_hints(record_class)
    built = {
        c.name: (
            c.description,
            _column_values(
                given[key], hints[c.name], key, n, reference, record_class, c.name
            ),
        )
        for c in columns
        if (key := c.argument or c.name) in given
    }
    return checked, built


def _refuse_length(length: int, argument: str, n: int, reference: str) -> None:
    """Refuse an argument whose length differs from the reference's."""
    if length != n:
        msg = f"{argument} length ({length}) must match {reference} length ({n})."
        raise ValueError(msg)


def _column_values(
    values: Any,
    hint: Any,
    argument: str,
    n: int,
    reference: str,
    record_class: type,
    name: str,
) -> NDArray[np.generic]:
    """Return one value per record as the array type ``hint`` names.

    A list of strings becomes an object array first, so numpy never turns a
    number or None into text before it is checked.
    """
    if isinstance(values, str):
        msg = (
            f"{argument} must be a sequence with one value per record, not the "
            f"single string {values!r}; a string would be split into its "
            f"characters. Pass [{values!r}] * {n} to use it for every record."
        )
        raise TypeError(msg)
    if hint is float or hint is int:
        numbers = np.array(values, dtype=np.float64 if hint is float else np.int64)
        _refuse_length(len(numbers), argument, n, reference)
        return numbers
    array: NDArray[Any] = (
        values if isinstance(values, np.ndarray) else np.fromiter(values, object)
    )
    _refuse_length(len(array), argument, n, reference)
    if array.dtype.kind != "U":
        array = array.astype(object)
        if type(None) in get_args(hint):
            array = np.array(["" if v is None else v for v in array], object)
        if all(map(isinstance, array, repeat(str))):
            array = array.astype(str)
    record = record_class.__name__
    if get_origin(hint) is Literal:
        allowed = get_args(hint)
        bad = np.flatnonzero(~np.isin(array, allowed))
        if bad.size:
            raise ValueError(
                _format_error(
                    f"{argument} contains {array.item(bad[0])!r}, which is not a "
                    f"{record.lower()} {name}.",
                    why=f"the {name} column uses the vocabulary of {record}.{name}.",
                    fix=f"use one of {allowed}.",
                )
            )
    if array.dtype.kind != "U":
        i = next(i for i, value in enumerate(array) if not isinstance(value, str))
        raise TypeError(
            _format_error(
                f"{record} {i} has {name} {array.item(i)!r}, not a string.",
                fix=f"give every {record.lower()} its {name} as a string.",
            )
        )
    return array.astype(str)


def _build_time_intervals(
    name: str,
    description: str,
    start_times: NDArray[np.float64],
    stop_times: NDArray[np.float64],
    columns: dict[str, tuple[str, NDArray[np.generic]]],
) -> Any:
    """Build a TimeIntervals table column by column.

    Each column is stored from a typed array, so it keeps its dtype when
    written to disk.

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
@reads_pynapple_samples
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
    Zero laps write nothing, without a warning: NWB files should not hold empty tables
    (nwbinspector ``check_empty_table``). Reading the table back then raises
    ``KeyError``; treat a missing table as zero laps. With ``overwrite=True``,
    zero laps still remove the table they replace.

    Parameters
    ----------
    nwbfile : NWBFile
        The NWB file to write to.
    laps : Sequence[Lap], optional
        Laps from :func:`~neurospatial.behavior.segmentation.detect_laps`.
        Writes the ``direction`` and ``overlap_score`` of every lap. Laps
        detected without a template (``method="region"``) have no overlap
        score, so when every score is NaN the ``overlap_score`` column is left
        out. Cannot be combined with the array arguments.
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

    Notes
    -----
    ``stop_time`` is exclusive: neurospatial reads each lap as the half-open
    window ``[start_time, stop_time)``, so a sample or spike exactly at
    ``stop_time`` is outside it. NWB does not say whether ``stop_time`` is
    inclusive, and pynapple's ``load_file`` reads the table as an
    ``IntervalSet`` whose ``end`` is inclusive, so pynapple keeps one more
    sample per lap whenever a sample falls exactly on ``stop_time``; the
    ``IntervalSet`` passed to a neurospatial ``epochs=`` keeps it too. For the
    same selection on both routes, pick one meaning: the half-open one with
    ``epochs=np.column_stack([iset.start, iset.end])`` (a plain array is
    half-open), or the inclusive one with ``in_epochs(t, rows,
    closed="both")`` on the rows from ``read_intervals(nwbfile, interval_name="laps")``.

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

    _require_pynwb()
    if name in _PREDEFINED_INTERVALS:
        msg = (
            f"write_laps cannot use name={name!r}: NWB reserves it for the "
            "predefined table of that name. Use another name, such as 'laps'."
        )
        raise ValueError(msg)

    (start, stop), columns = _record_columns(
        Lap,
        laps,
        {
            "start_times": start_times,
            "stop_times": stop_times,
            "directions": directions,
        },
        times=_INTERVAL_TIMES,
        columns=_LAP_COLUMNS,
        needs="write_laps needs either 'laps' (from detect_laps) or both "
        "'start_times' and 'stop_times'.",
    )
    _refuse_reversed_intervals(start, stop, record="lap")
    # Laps detected without a template (method="region") have no score;
    # an all-NaN column breaks NWB best practice, so it is left out.
    if "overlap_score" in columns and np.all(np.isnan(columns["overlap_score"][1])):
        del columns["overlap_score"]

    table = _build_time_intervals(name, description, start, stop, columns)

    _clear_for_write(
        nwbfile,
        nwbfile.intervals,
        name,
        overwrite=overwrite,
        what=f"laps table '{name}'",
    )
    if start.size == 0:
        return
    nwbfile.add_time_intervals(table)
    logger.debug("Wrote TimeIntervals '%s' with %d laps", name, len(start))


@keyword_hint("description")
@reads_pynapple_samples
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
    Zero crossings write nothing, without a warning: NWB files should not hold empty tables
    (nwbinspector ``check_empty_table``). Reading the table back then raises
    ``KeyError``; treat a missing table as zero crossings. With ``overwrite=True``,
    zero crossings still remove the table they replace.

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

    Notes
    -----
    With pynwb 4 or later (NWB schema 2.10+), ``EventsTable`` is part of the
    NWB core schema and ``ndx_events.EventsTable`` is ``pynwb.event.EventsTable``.
    A file written that way cannot be read by pynwb 3, which does not know the
    newer core types and reports the table as missing; read it with pynwb 4
    or later. Files written with pynwb 3 read with either version.

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

    # Check ndx-events first so users see the install hint.
    _require_ndx_events()

    (crossing_times,), columns = _record_columns(
        Crossing,
        crossings,
        {"times": times, "regions": regions, "directions": directions},
        times={"times": "time"},
        columns=_CROSSING_COLUMNS,
        needs="write_region_crossings needs either 'crossings' (from "
        "detect_region_crossings) or all of 'times', 'regions' and 'directions'.",
    )
    events_table = _build_events_table(name, description, crossing_times, columns)
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


@reads_pynapple_samples
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
    optional columns for start/end regions and how each trial ended.
    Zero trials write nothing, without a warning: NWB files should not hold empty tables
    (nwbinspector ``check_empty_table``). Reading the table back then raises
    ``KeyError``; treat a missing table as zero trials. With ``overwrite=True``,
    zero trials still remove the table they replace.

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
        argument is a single string, or a region is not a string (None is
        allowed in ``end_regions``).
    NotImplementedError
        If ``overwrite=True`` would replace the trials table of a file read
        from disk.
    ImportError
        If pynwb is not installed.

    Notes
    -----
    The trials table is stored in NWB's built-in trials location
    (``nwbfile.trials`` / ``/intervals/trials/``).

    ``stop_time`` is exclusive: neurospatial reads each trial as the half-open
    window ``[start_time, stop_time)``, so a sample or spike exactly at
    ``stop_time`` is outside it. NWB does not say whether ``stop_time`` is
    inclusive, and pynapple's ``load_file`` reads the table as an
    ``IntervalSet`` whose ``end`` is inclusive, so pynapple keeps one more
    sample per trial whenever a sample falls exactly on ``stop_time``; the
    ``IntervalSet`` passed to a neurospatial ``epochs=`` keeps it too. For the
    same selection on both routes, pick one meaning: the half-open one with
    ``epochs=np.column_stack([iset.start, iset.end])`` (a plain array is
    half-open), or the inclusive one with ``in_epochs(t, rows,
    closed="both")`` on the rows from read_trials.

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
    from neurospatial.behavior.segmentation import Trial

    _require_pynwb()

    (start, stop), columns = _record_columns(
        Trial,
        trials,
        {
            "start_times": start_times,
            "stop_times": stop_times,
            "start_regions": start_regions,
            "end_regions": end_regions,
            "outcomes": outcomes,
        },
        times=_INTERVAL_TIMES,
        columns=_TRIAL_COLUMNS,
        needs="write_trials needs either 'trials' (from segment_trials) or "
        "both 'start_times' and 'stop_times'.",
    )
    _refuse_reversed_intervals(start, stop, record="trial")

    table = _build_time_intervals(PREDEFINED_TRIALS, description, start, stop, columns)
    removed = nwbfile.trials is not None
    if removed:
        if not overwrite:
            msg = (
                "A trials table already exists in this NWB file. Pass "
                "overwrite=True to replace it."
            )
            raise ValueError(msg)
        if getattr(nwbfile, "read_io", None) is not None:
            msg = (
                "Cannot replace the trials table of a file read from disk: "
                "pynwb cannot remove a written trials table, and exporting the "
                "file would silently keep the old trials. Build a new NWBFile "
                "and write the trials there, or store these intervals under "
                "another name with nwbfile.add_time_intervals(...) (read back "
                "with read_intervals)."
            )
            raise NotImplementedError(msg)
        old = nwbfile.trials
        referrers = sorted(
            f"column {obj.name!r} of {getattr(obj.parent, 'name', None)!r}"
            for obj in nwbfile.all_children()
            if getattr(obj, "table", None) is old
        )
        if referrers:
            msg = (
                f"Cannot replace the trials table: {', '.join(referrers)} "
                "refers to its rows and would point at a table that is no "
                "longer in the file. Build a new NWBFile, or remove those "
                "references first."
            )
            raise ValueError(msg)
        # The trials attribute can be set only once; clearing the field and
        # detaching the old table is the only way to replace it in memory.
        nwbfile.fields.pop(PREDEFINED_TRIALS)
        old.reset_parent()
        logger.info("Overwriting existing trials table")
    if start.size:
        nwbfile.trials = table
    if removed:
        # The referrer search cached pynwb's objects with the old table in
        # them; walk the file again to drop it.
        nwbfile.all_children()
    if start.size == 0:
        return
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

    Notes
    -----
    Each row is the half-open window ``[start_time, stop_time)``: a sample
    exactly at ``stop_time`` is outside the trial. pynapple's ``load_file``
    reads the same table with inclusive ends; see :func:`write_trials` for
    how to get identical selections on both routes.

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

    # Verify ndx-events is installed
    _require_ndx_events()

    # Validate input
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            _format_error(
                f"Expected pd.DataFrame, got {type(df).__name__}.",
                why="events must be a pandas DataFrame.",
                fix="convert using pd.DataFrame({'timestamp': times}).",
            )
        )

    if "timestamp" not in df.columns:
        raise ValueError(
            _format_error(
                "DataFrame is missing required 'timestamp' column. Available "
                f"columns: {list(df.columns)}.",
                why="an EventsTable needs the time of each event.",
                fix="add a 'timestamp' column of event times in seconds.",
            )
        )

    return _build_events_table(
        name,
        description,
        _checked_event_times(df["timestamp"], "timestamp column"),
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
    and any additional columns from the DataFrame. An empty DataFrame writes
    nothing, without a warning: NWB files should not hold empty tables (nwbinspector
    ``check_empty_table``). Reading the table back then raises ``KeyError``;
    treat a missing table as zero events. With ``overwrite=True``, an empty
    DataFrame still removes the table it replaces.

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

    With pynwb 4 or later (NWB schema 2.10+), ``EventsTable`` is part of the
    NWB core schema and ``ndx_events.EventsTable`` is ``pynwb.event.EventsTable``.
    A file written that way cannot be read by pynwb 3, which does not know the
    newer core types and reports the table as missing; read it with pynwb 4
    or later. Files written with pynwb 3 read with either version.

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
