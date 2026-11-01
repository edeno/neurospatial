"""Tables of half-open time intervals with stable ids and named fields.

:class:`Intervals` holds behavioral records (trials, task-log rows, NWB
trials) as one container: rows sorted by time, a stable ``interval_id`` per
row, and a fixed field schema. It passes as ``epochs=`` to any analysis and
assigns samples to rows. :class:`Interval` is the generic record for rows
without a detector class.
"""

from __future__ import annotations

import dataclasses
import numbers
import warnings
from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence
from copy import deepcopy
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Generic, TypeVar, get_type_hints, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurospatial._exceptions import _format_error
from neurospatial._observation import _validated_signal_times
from neurospatial._pynapple import reads_pynapple_samples
from neurospatial._stacklevel import _external_stacklevel

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["Interval", "Intervals"]

_TIME_FIELDS = ("start_time", "stop_time")


def _plain(value: Any) -> Any:
    """A NumPy scalar as the matching Python scalar; anything else unchanged."""
    return value.item() if isinstance(value, np.generic) else value


def _checked_times(owner: str, start: object, stop: object) -> tuple[float, float]:
    """Finite ``start <= stop`` as Python floats."""
    try:
        start_f, stop_f = float(start), float(stop)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        start_f = stop_f = np.nan
    if not (np.isfinite(start_f) and np.isfinite(stop_f) and start_f <= stop_f):
        raise ValueError(
            _format_error(
                f"{owner} needs finite start_time <= stop_time; got "
                f"start_time={start!r}, stop_time={stop!r}.",
                why="an interval is the half-open window [start_time, stop_time) "
                "in seconds.",
                fix="pass finite times in seconds with start_time <= stop_time.",
            )
        )
    return start_f, stop_f


def _checked_value(owner: str, name: str, value: Any) -> Any:
    """A field value: hashable, ``None`` for missing, never ``""``."""
    value = _plain(value)
    if isinstance(value, str) and value == "":
        raise ValueError(
            _format_error(
                f"{owner} field {name!r} is the empty string.",
                why="a missing value is None. NWB string columns cannot hold "
                'None, so writers store it as "" and readers turn "" back into '
                "None; an empty string would come back as missing.",
                fix=f"pass None for a missing {name}.",
            )
        )
    if not isinstance(value, Hashable):
        raise TypeError(
            _format_error(
                f"{owner} field {name!r} holds a {type(value).__name__}, which is "
                "not hashable.",
                why="records are immutable and hashable, like the detector "
                "records, so they can be compared, pickled and used as keys.",
                fix=f"store {name} as a hashable value, for example a tuple "
                "instead of a list or array.",
            )
        )
    return value


@dataclasses.dataclass(frozen=True)
class Interval:
    """A half-open time window ``[start_time, stop_time)`` with named fields.

    The generic record for intervals without a detector class: rows of a
    task log, an NWB trials table or a pynapple ``IntervalSet``. Fields are
    read as attributes (``interval.outcome``) or as a mapping
    (``interval.fields``).

    Parameters
    ----------
    start_time, stop_time : float
        Window bounds in seconds, finite, ``start_time <= stop_time``. A
        sample at ``stop_time`` is outside the window.
    items : mapping or sequence of (str, value) pairs, optional
        The fields. Values must be hashable; a missing value is ``None``. The
        empty string is refused, because NWB stores a missing string as
        ``""`` and reads it back as ``None``.

    Attributes
    ----------
    items : tuple of (str, value)
        The fields as ``(name, value)`` pairs, in order. Pairs keep the
        record hashable, picklable and deep-copyable.

    Examples
    --------
    >>> from neurospatial import Interval
    >>> trial = Interval(2.0, 5.5, {"instructed_goal": "left", "rewarded": True})
    >>> trial.instructed_goal
    'left'
    >>> dict(trial.fields)
    {'instructed_goal': 'left', 'rewarded': True}
    """

    start_time: float
    stop_time: float
    items: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        """Check the times and fields and store plain Python values."""
        start, stop = _checked_times("Interval", self.start_time, self.stop_time)
        object.__setattr__(self, "start_time", start)
        object.__setattr__(self, "stop_time", stop)
        raw: Any = self.items
        pairs: list[Any] = list(raw.items()) if isinstance(raw, Mapping) else list(raw)
        items: list[tuple[str, Any]] = []
        for pair in pairs:
            if not (isinstance(pair, tuple) and len(pair) == 2):
                msg = (
                    "Interval items must be (name, value) pairs or a mapping; got "
                    f"{pair!r}."
                )
                raise TypeError(msg)
            name, value = pair
            _check_field_name(name, owner="Interval", reserved=_INTERVAL_MEMBERS)
            items.append((name, _checked_value("Interval", name, value)))
        names = [name for name, _ in items]
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            msg = f"Interval field names must be unique; repeated: {repeated}."
            raise ValueError(msg)
        object.__setattr__(self, "items", tuple(items))

    @property
    def fields(self) -> Mapping[str, Any]:
        """The fields as a read-only mapping, built on access."""
        return MappingProxyType(dict(self.items))

    def __getattr__(self, name: str) -> Any:
        """Read a field by name."""
        if name.startswith("_"):
            raise AttributeError(name)
        for key, value in object.__getattribute__(self, "items"):
            if key == name:
                return value
        names = [key for key, _ in object.__getattribute__(self, "items")]
        msg = f"Interval has no field {name!r}; its fields are {names}."
        raise AttributeError(msg)


_INTERVAL_MEMBERS = frozenset({*_TIME_FIELDS, "items", "fields"})


def _check_field_name(name: object, *, owner: str, reserved: Iterable[str]) -> None:
    """Refuse a field name that is not a public identifier or is taken."""
    if not isinstance(name, str) or not name.isidentifier():
        msg = f"{owner} field names must be identifiers; got {name!r}."
        raise ValueError(msg)
    if name.startswith("_"):
        msg = (
            f"{owner} field names cannot start with '_'; got {name!r}. Names "
            "starting with '_' are kept for the container itself."
        )
        raise ValueError(msg)
    if name in reserved:
        raise ValueError(
            _format_error(
                f"{owner} cannot have a field named {name!r}.",
                why=f"{name!r} is already an attribute of {owner}, so the field "
                "could not be read by name.",
                fix="rename the field or column.",
            )
        )


R = TypeVar("R")


def _record_schema(record_type: type) -> tuple[tuple[str, ...], dict[str, np.dtype]]:
    """Field names and default dtypes of a detector record dataclass."""
    if not dataclasses.is_dataclass(record_type):
        msg = (
            f"Intervals holds dataclass records with start_time and stop_time "
            f"fields; got {record_type.__name__}."
        )
        raise TypeError(msg)
    names = [f.name for f in dataclasses.fields(record_type)]
    missing = [name for name in _TIME_FIELDS if name not in names]
    if missing:
        raise TypeError(
            _format_error(
                f"{record_type.__name__} records have no {' or '.join(missing)} "
                "field, so they are not intervals.",
                why="every interval is the half-open window [start_time, stop_time).",
                fix="pass records with start_time and stop_time fields, or build "
                "Intervals.from_arrays(start, stop, ...).",
            )
        )
    fields = tuple(name for name in names if name not in _TIME_FIELDS)
    hints = get_type_hints(record_type)
    dtypes = {name: _dtype_for_hint(hints.get(name)) for name in fields}
    return fields, dtypes


def _dtype_for_hint(hint: object) -> np.dtype:
    """The column dtype of a plainly typed record field; object otherwise."""
    if hint is bool:
        return np.dtype(bool)
    if hint is int:
        return np.dtype(np.int64)
    if hint is float:
        return np.dtype(np.float64)
    return np.dtype(object)


def _object_array(values: Sequence[Any]) -> NDArray[np.object_]:
    """A 1-D object array of ``values``; tuples and arrays stay whole."""
    return np.fromiter(values, dtype=object, count=len(values))


def _column(values: Sequence[Any], dtype: np.dtype | None) -> NDArray[Any]:
    """One field as an array of ``dtype``, inferred from the values when None.

    Numbers and Booleans give a numeric array; anything else (strings,
    ``None``, tuples, arrays) an object array.
    """
    if dtype is not None and dtype.kind != "O":
        return np.asarray(values, dtype=dtype)
    if (
        dtype is None
        and values
        and all(
            isinstance(v, (numbers.Number, np.bool_)) and not isinstance(v, complex)
            for v in values
        )
    ):
        return np.asarray(values)
    return _object_array(values)


def _read_only(array: NDArray[Any]) -> NDArray[Any]:
    array.setflags(write=False)
    return array


def _checked_ids(ids: ArrayLike, n: int, *, owner: str) -> NDArray[np.int64]:
    """``ids`` as unique, non-negative int64 of length ``n``."""
    raw = np.asarray(ids)
    if raw.size == 0:
        raw = raw.astype(np.int64)
    if raw.dtype.kind not in "iu":
        raise TypeError(
            _format_error(
                f"{owner} interval_ids must be integers; got dtype {raw.dtype}.",
                fix="pass integer ids, for example the trial numbers of a task log.",
            )
        )
    if raw.shape != (n,):
        msg = (
            f"{owner} needs one interval_id per interval: got shape {raw.shape} "
            f"for {n} interval(s)."
        )
        raise ValueError(msg)
    negative = np.unique(raw[raw < 0])
    values, counts = np.unique(raw, return_counts=True)
    repeated = values[counts > 1]
    if negative.size or repeated.size:
        problems = []
        if negative.size:
            problems.append(f"negative ids {negative[:5].tolist()}")
        if repeated.size:
            problems.append(f"repeated ids {repeated[:5].tolist()}")
        raise ValueError(
            _format_error(
                f"{owner} interval_ids must be unique and >= 0; got "
                f"{' and '.join(problems)}.",
                why="an interval_id identifies one interval through every "
                "selection and round trip.",
                fix="pass distinct non-negative integers, or omit interval_ids to "
                "number the intervals 0 to n-1 in input order.",
            )
        )
    return raw.astype(np.int64)


class Intervals(Generic[R]):
    """Half-open time intervals with stable ids and named fields.

    One row per interval ``[start, stop)`` in seconds: a trial, a task-log
    row, a row of an NWB trials table. Rows are sorted by
    ``(start, stop, interval_id)``. Each row keeps its ``interval_id``
    through every selection and round trip, so a trial selected from a
    subset is the same trial in the full table.

    ``[]`` selects by position, ``sel(interval_id=...)`` by id, as for unit
    containers. Fields are read as arrays by name (``trials.outcome``) and
    select conditions with ``trials[trials.outcome == "reached"]``. The
    container passes as ``epochs=`` (its :attr:`windows`) to any analysis.

    Parameters
    ----------
    records : iterable of records
        Records of one type: the detector records (``Trial``) or
        :class:`Interval`. Each has ``start_time`` and ``stop_time``.
    record_type : type, optional
        The record class. Inferred from the records; required when there are
        none.
    fields : sequence of str, optional
        The field names of :class:`Interval` records. Inferred from the first
        record; required for no :class:`Interval` records. Detector records
        take their fields from the class.
    interval_ids : array-like of int, shape (n,), optional
        Stable id of each record, in input order: unique and ``>= 0``.
        Default ``0`` to ``n - 1`` in input order.
    analysis_settings : mapping, optional
        How the intervals were made (for example a detector's parameters).

    Attributes
    ----------
    interval_ids : ndarray of int64, shape (n,)
        Stable id of each row.
    fields : tuple of str
        Field names, kept by empty selections.
    record_type : type
        The record class.
    start, stop, durations : ndarray of float64, shape (n,)
        Row bounds and ``stop - start``, in seconds.
    windows : ndarray of float64, shape (n_windows, 2)
        Rows with ``stop > start``: the time the intervals cover.

    Notes
    -----
    **Observations and windows.** A row with ``stop == start`` (for example a
    visit seen in one sample) stays in the container: it is counted,
    selected and exported. It holds no time, so it is not in :attr:`windows`,
    contains no sample and never overlaps another row.

    **Overlap.** Rows may overlap. Methods that give each sample one interval
    (:meth:`sample_interval_ids`, :meth:`sample_values`) refuse overlapping
    rows; select a non-overlapping subset first. Touching rows
    (``stop[i] == start[j]``) do not overlap. :meth:`sample_slices` is per row
    and allows overlap.

    **Missing values.** A missing field value is ``None``; the empty string
    is refused (NWB writes ``None`` as ``""``). Samples outside every interval
    get ``NaN`` (float values) or ``None`` (others), never ``0`` or ``""``.

    **Per-interval arrays** are in row order, the order of the container
    (sorted by time). An array built in the order of input records that were
    not sorted is misaligned; a ``pandas.Series`` indexed by ``interval_id``
    is aligned by id.

    Examples
    --------
    A task log with one row per trial:

    >>> import numpy as np
    >>> import pandas as pd
    >>> from neurospatial import Intervals
    >>> log = pd.DataFrame(
    ...     {
    ...         "trial": [11, 12, 13],
    ...         "start_time": [0.0, 10.0, 20.0],
    ...         "stop_time": [8.0, 17.0, 29.0],
    ...         "instructed_goal": ["left", "right", "left"],
    ...     }
    ... )
    >>> trials = Intervals.from_dataframe(log, id_column="trial")
    >>> trials
    Intervals[Interval]: 3 intervals in [0, 29) s; fields: instructed_goal
    >>> left = trials[trials.instructed_goal == "left"]
    >>> left.interval_ids.tolist()
    [11, 13]
    >>> left.sel(interval_id=13).stop_time
    29.0

    Each sample's interval and field value, missing outside:

    >>> times = np.array([1.0, 9.0, 12.0, 25.0])
    >>> trials.sample_interval_ids(times).tolist()
    [11, -1, 12, 13]
    >>> trials.sample_values(times, "instructed_goal").tolist()
    ['left', None, 'right', 'left']

    Restrict an analysis to the left trials with ``epochs=left``:

    >>> left.windows.tolist()
    [[0.0, 8.0], [20.0, 29.0]]
    """

    def __init__(
        self,
        records: Iterable[R] = (),
        *,
        record_type: type[R] | None = None,
        fields: Sequence[str] | None = None,
        interval_ids: ArrayLike | None = None,
        analysis_settings: Mapping[str, Any] | None = None,
    ) -> None:
        if isinstance(records, Intervals):
            msg = (
                "Intervals(...) got an Intervals; it is already one. Select rows "
                "with [] or sel(interval_id=...)."
            )
            raise TypeError(msg)
        rows = tuple(records)
        kind = record_type if record_type is not None else _infer_type(rows)
        mixed = sorted({type(r).__name__ for r in rows if type(r) is not kind})
        if mixed:
            raise TypeError(
                _format_error(
                    f"Intervals of {kind.__name__} records got "
                    f"{', '.join(mixed)} records too.",
                    why="one container holds one record type, with one field schema.",
                    fix="build one Intervals per record type.",
                )
            )
        names, dtypes = _schema(kind, rows, fields)
        for name in names:
            _check_field_name(name, owner="Intervals", reserved=_MEMBERS)
        values = {name: [_plain(getattr(r, name)) for r in rows] for name in names}
        for name, column in values.items():
            for value in column:
                _checked_value(kind.__name__, name, value)
        columns = {
            name: _column(values[name], None if rows else dtypes[name])
            for name in names
        }
        ids = _checked_ids(
            np.arange(len(rows)) if interval_ids is None else interval_ids,
            len(rows),
            owner="Intervals",
        )
        start = np.array([r.start_time for r in rows], dtype=np.float64)  # type: ignore[attr-defined]
        stop = np.array([r.stop_time for r in rows], dtype=np.float64)  # type: ignore[attr-defined]
        order = np.lexsort((ids, stop, start))
        self._init_parts(
            records=tuple(rows[i] for i in order),
            record_type=kind,
            fields=names,
            dtypes={name: column.dtype for name, column in columns.items()},
            ids=ids[order],
            start=start[order],
            stop=stop[order],
            settings=dict(analysis_settings or {}),
        )

    def _init_parts(
        self,
        *,
        records: tuple[R, ...],
        record_type: type[R],
        fields: tuple[str, ...],
        dtypes: dict[str, np.dtype],
        ids: NDArray[np.int64],
        start: NDArray[np.float64],
        stop: NDArray[np.float64],
        settings: dict[str, Any],
    ) -> None:
        self._records = records
        self._record_type = record_type
        self._fields = fields
        self._dtypes = dtypes
        self._ids = _read_only(np.asarray(ids, dtype=np.int64))
        self._start = _read_only(np.asarray(start, dtype=np.float64))
        self._stop = _read_only(np.asarray(stop, dtype=np.float64))
        self._settings = deepcopy(settings)
        self._columns: dict[str, NDArray[Any]] = {}

    def _subset(self, positions: NDArray[np.intp]) -> Intervals[R]:
        """Rows at sorted, distinct ``positions``; keeps ids, schema and settings."""
        new = object.__new__(type(self))
        new._init_parts(
            records=tuple(self._records[i] for i in positions),
            record_type=self._record_type,
            fields=self._fields,
            dtypes=self._dtypes,
            ids=self._ids[positions],
            start=self._start[positions],
            stop=self._stop[positions],
            settings=self._settings,
        )
        return new

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_columns"] = {}
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        for name in ("_ids", "_start", "_stop"):
            _read_only(self.__dict__[name])

    # ----------------------------------------------------------------- identity

    @property
    def interval_ids(self) -> NDArray[np.int64]:
        """Stable id of each row, shape ``(n,)``; read-only."""
        return self._ids

    @property
    def fields(self) -> tuple[str, ...]:
        """Field names, in schema order."""
        return self._fields

    @property
    def record_type(self) -> type[R]:
        """The record class."""
        return self._record_type

    @property
    def analysis_settings(self) -> dict[str, Any]:
        """How the intervals were made; an independent copy on every read."""
        return deepcopy(self._settings)

    # ------------------------------------------------------------------ columns

    @property
    def start(self) -> NDArray[np.float64]:
        """Row starts in seconds, shape ``(n,)``; read-only."""
        return self._start

    @property
    def stop(self) -> NDArray[np.float64]:
        """Row stops in seconds (exclusive), shape ``(n,)``; read-only."""
        return self._stop

    @property
    def durations(self) -> NDArray[np.float64]:
        """``stop - start`` in seconds, shape ``(n,)``."""
        return _read_only(self._stop - self._start)

    @property
    def windows(self) -> NDArray[np.float64]:
        """Rows with ``stop > start``, shape ``(n_windows, 2)``, sorted.

        The time the intervals cover; ``epochs=intervals`` reads it. Rows
        without duration hold no time and are left out. Empty when no row has
        duration: an empty selection, not an error.
        """
        keep = self._stop > self._start
        return _read_only(np.column_stack([self._start[keep], self._stop[keep]]))

    def __getattr__(self, name: str) -> NDArray[Any]:
        """One field as a read-only array, shape ``(n,)``, in row order."""
        if name.startswith("_"):
            raise AttributeError(name)
        if name not in self._fields:
            msg = (
                f"Intervals has no field {name!r}; its fields are {list(self._fields)}."
            )
            raise AttributeError(msg)
        column = self._columns.get(name)
        if column is None:
            values = [getattr(r, name) for r in self._records]
            column = _read_only(_column(values, self._dtypes[name]))
            self._columns[name] = column
        return column

    # ---------------------------------------------------------------- selection

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[R]:
        return iter(self._records)

    @overload
    def __getitem__(self, key: int) -> R: ...

    @overload
    def __getitem__(
        self, key: slice | Sequence[int] | Sequence[bool] | NDArray[Any]
    ) -> Intervals[R]: ...

    def __getitem__(self, key: Any) -> R | Intervals[R]:
        """Select rows by position: an int gives a record, anything else a subset.

        A slice, an integer array or a Boolean mask of length ``n`` returns
        an :class:`Intervals` with the selected rows in time order and their
        ``interval_ids``.
        """
        n = len(self)
        if isinstance(key, (bool, np.bool_)):
            msg = "Select Intervals with a Boolean mask of length n, not one bool."
            raise TypeError(msg)
        if isinstance(key, (numbers.Integral, np.integer)):
            return self._records[int(key)]
        if isinstance(key, slice):
            return self._subset(np.arange(n)[key])
        raw = np.asarray(key)
        if raw.size == 0:
            return self._subset(np.empty(0, dtype=np.intp))
        if raw.dtype == bool:
            if raw.shape != (n,):
                msg = (
                    f"A Boolean selection of Intervals needs one entry per "
                    f"interval ({n}); got shape {raw.shape}."
                )
                raise IndexError(msg)
            return self._subset(np.flatnonzero(raw))
        if raw.dtype.kind not in "iu" or raw.ndim != 1:
            msg = (
                "Intervals are selected by an int, a slice, a 1-D integer array "
                f"of positions or a Boolean mask; got {type(key).__name__}"
                f" of dtype {raw.dtype}."
            )
            raise TypeError(msg)
        positions = np.arange(n)[raw]
        unique = np.unique(positions)
        if unique.size != positions.size:
            msg = (
                "An Intervals selection cannot repeat a row: each interval_id "
                "appears once."
            )
            raise ValueError(msg)
        return self._subset(unique)

    @overload
    def sel(self, *, interval_id: int) -> R: ...

    @overload
    def sel(self, *, interval_id: ArrayLike) -> R | Intervals[R]: ...

    def sel(self, *, interval_id: ArrayLike) -> R | Intervals[R]:
        """Select by stable id: one id gives a record, several a subset.

        Parameters
        ----------
        interval_id : int or array-like of int
            Id(s) from :attr:`interval_ids`.

        Returns
        -------
        record or Intervals
            The record with this id, or the rows with these ids in time
            order.

        Raises
        ------
        KeyError
            If an id is not in :attr:`interval_ids`; every missing id is
            listed.
        """
        wanted = np.asarray(interval_id)
        if wanted.size and wanted.dtype.kind not in "iu":
            msg = (
                f"interval_id must be an integer or integer array; got {interval_id!r}."
            )
            raise TypeError(msg)
        where = {int(i): p for p, i in enumerate(self._ids.tolist())}
        flat = [int(i) for i in wanted.reshape(-1).tolist()]
        missing = [i for i in flat if i not in where]
        if missing:
            shown = ", ".join(map(str, missing[:10]))
            more = f" (+{len(missing) - 10} more)" if len(missing) > 10 else ""
            msg = (
                f"interval_id {shown}{more} not in these intervals; ids range "
                f"over {_id_span(self._ids)}."
            )
            raise KeyError(msg)
        if wanted.ndim == 0:
            return self._records[where[flat[0]]]
        return self[np.array([where[i] for i in flat], dtype=np.intp)]

    # ------------------------------------------------------------------ samples

    def _refuse_overlap(self, call: str) -> None:
        """Refuse positive-duration rows that overlap (zero-length rows never do)."""
        pos = np.flatnonzero(self._stop > self._start)
        s, e = self._start[pos], self._stop[pos]
        # Rows are sorted by start, so a row overlaps an earlier one exactly
        # when it starts before the latest stop so far.
        latest = np.maximum.accumulate(e)
        bad = np.flatnonzero(s[1:] < latest[:-1])
        if bad.size:
            j = int(pos[bad[0] + 1])
            i = int(pos[np.flatnonzero(e[: bad[0] + 1] > s[bad[0] + 1])[0]])
            raise ValueError(
                _format_error(
                    f"{call}: intervals {self._ids[i]} and {self._ids[j]} overlap "
                    f"([{self._start[i]:g}, {self._stop[i]:g}) and "
                    f"[{self._start[j]:g}, {self._stop[j]:g}) s).",
                    why="a sample in both would belong to two intervals.",
                    fix="select a non-overlapping subset first, for example rows "
                    "with one value of a field.",
                )
            )

    def _warn_if_no_sample_inside(
        self, times: NDArray[np.float64], inside: NDArray[np.bool_], call: str
    ) -> None:
        """Warn when rows have duration but contain no sample (another clock)."""
        timed = self._stop > self._start
        if not timed.any() or inside.any():
            return
        warnings.warn(
            f"{call}: none of the {int(timed.sum())} interval(s) (spanning "
            f"[{self._start[timed].min():g}, {self._stop[timed].max():g}) s) "
            f"contains a sample of times (spanning [{times[0]:g}, {times[-1]:g}] "
            "s), so every sample is outside the intervals. Fix: check that the "
            "intervals and times are on the same clock, in seconds (for example "
            "intervals detected on the same recording).",
            UserWarning,
            stacklevel=_external_stacklevel(),
        )

    def _sample_positions(self, times: ArrayLike, call: str) -> NDArray[np.intp]:
        """Row position of each sample, ``-1`` outside. Refuses overlap."""
        t = _validated_signal_times(times, call=call)
        self._refuse_overlap(call)
        pos = np.flatnonzero(self._stop > self._start)
        s, e = self._start[pos], self._stop[pos]
        k = np.searchsorted(s, t, side="right") - 1
        inside = k >= 0
        inside[inside] = t[inside] < e[k[inside]]
        out = np.full(t.shape, -1, dtype=np.intp)
        out[inside] = pos[k[inside]]
        self._warn_if_no_sample_inside(t, inside, call)
        return out

    @reads_pynapple_samples
    def sample_interval_ids(self, times: ArrayLike) -> NDArray[np.int64]:
        """The ``interval_id`` containing each sample, ``-1`` outside every row.

        Sample ``j`` is in row ``i`` when ``start[i] <= times[j] < stop[i]``;
        a sample at a row's stop belongs to the next touching row, or none.

        Parameters
        ----------
        times : array-like, shape (n_samples,)
            Sample times in seconds, finite and strictly increasing.

        Returns
        -------
        ndarray of int64, shape (n_samples,)

        Raises
        ------
        ValueError
            If ``times`` is invalid, or two rows with duration overlap.

        Warns
        -----
        UserWarning
            If rows have duration but no sample falls in any (usually times on
            another clock).
        """
        p = self._sample_positions(times, "Intervals.sample_interval_ids")
        return np.where(p >= 0, self._ids[np.maximum(p, 0)], -1).astype(np.int64)

    @reads_pynapple_samples
    def sample_values(
        self, times: ArrayLike, values: str | tuple[str, ...] | ArrayLike
    ) -> NDArray[Any]:
        """Each sample's value of its interval, missing outside every row.

        Parameters
        ----------
        times : array-like, shape (n_samples,)
            Sample times in seconds, finite and strictly increasing.
        values : str, tuple of str, or array-like of shape (n,)
            A field name; a tuple of field names, giving each sample a tuple
            (for example ``("start_region", "end_region")`` as a condition
            label); or one value per row in row order (a ``pandas.Series``
            indexed by ``interval_id`` is aligned by id).

        Returns
        -------
        ndarray, shape (n_samples,)
            Float values with ``NaN`` outside every row; any other values as
            an object array with ``None`` outside. Never ``0`` or ``""``.

        Raises
        ------
        ValueError
            If ``times`` is invalid, two rows with duration overlap, or
            ``values`` does not have one entry per row.
        AttributeError
            If a name is not a field.

        Warns
        -----
        UserWarning
            If rows have duration but no sample falls in any.
        """
        per_row = self._per_row(values, name="values")
        p = self._sample_positions(times, "Intervals.sample_values")
        out: NDArray[Any]
        if per_row.dtype.kind == "f":
            out = np.full(p.shape, np.nan, dtype=per_row.dtype)
        else:
            out = np.full(p.shape, None, dtype=object)
        inside = p >= 0
        out[inside] = per_row[p[inside]]
        return out

    def _per_row(self, values: Any, *, name: str) -> NDArray[Any]:
        """``values`` as one entry per row, in row order."""
        if isinstance(values, str):
            column: NDArray[Any] = getattr(self, values)
            return column
        if isinstance(values, tuple) and all(isinstance(v, str) for v in values):
            columns = [getattr(self, v).tolist() for v in values]
            return _object_array(list(zip(*columns, strict=True)))
        index = getattr(values, "index", None)
        if index is not None and hasattr(values, "reindex"):
            aligned = values.reindex(self._ids)
            unknown = np.setdiff1d(np.asarray(index), self._ids)
            if unknown.size or aligned.index.size != len(self):
                msg = (
                    f"{name} is a Series indexed by interval_id, but ids "
                    f"{unknown[:5].tolist()} are not in these intervals."
                )
                raise ValueError(msg)
            values = aligned.to_numpy()
        array = np.asarray(values)
        if array.dtype.kind in "OUS":
            array = _object_array(list(values))
        if array.shape[:1] != (len(self),) or array.ndim != 1:
            raise ValueError(
                _format_error(
                    f"{name} must have one entry per interval ({len(self)}); got "
                    f"shape {array.shape}.",
                    fix="pass a field name, or an array in row order (the order "
                    "of the Intervals, sorted by time).",
                )
            )
        return array

    @reads_pynapple_samples
    def sample_slices(self, times: ArrayLike) -> list[slice]:
        """The samples of each row, as one slice per row in row order.

        ``times[s]`` for the ``i``-th slice ``s`` are the samples with
        ``start[i] <= t < stop[i]``. Rows may overlap: each slice is a view
        of its own row, so do not use the slices to write one shared
        per-sample array.

        Parameters
        ----------
        times : array-like, shape (n_samples,)
            Sample times in seconds, finite and strictly increasing.

        Returns
        -------
        list of slice, length n

        Warns
        -----
        UserWarning
            If rows have duration but no sample falls in any.
        """
        t = _validated_signal_times(times, call="Intervals.sample_slices")
        lo = np.searchsorted(t, self._start, side="left")
        hi = np.searchsorted(t, self._stop, side="left")
        self._warn_if_no_sample_inside(t, hi > lo, "Intervals.sample_slices")
        return [slice(int(a), int(b)) for a, b in zip(lo, hi, strict=True)]

    # ----------------------------------------------------- tables and building

    def intervals_table(self) -> pd.DataFrame:
        """One row per interval, indexed by ``interval_id``.

        Returns
        -------
        pandas.DataFrame
            Columns ``start_time``, ``stop_time``, then :attr:`fields`, in row
            order; ``df.attrs["analysis_settings"]`` holds the settings. An
            empty selection gives an empty frame with the same columns.
        """
        import pandas as pd

        data: dict[str, Any] = {
            "start_time": np.array(self._start),
            "stop_time": np.array(self._stop),
        }
        for name in self._fields:
            data[name] = np.array(getattr(self, name))
        table = pd.DataFrame(
            data, index=pd.Index(np.array(self._ids), name="interval_id")
        )
        table.attrs["analysis_settings"] = self.analysis_settings
        return table

    @classmethod
    def from_arrays(
        cls,
        start: ArrayLike,
        stop: ArrayLike,
        *,
        interval_ids: ArrayLike | None = None,
        **fields: ArrayLike,
    ) -> Intervals[Interval]:
        """Build :class:`Interval` rows from parallel arrays.

        Parameters
        ----------
        start, stop : array-like, shape (n,)
            Row bounds in seconds.
        interval_ids : array-like of int, shape (n,), optional
            Stable ids in input order. Default ``0`` to ``n - 1``.
        **fields : array-like, shape (n,)
            One array per field, in input order.

        Returns
        -------
        Intervals of Interval

        Examples
        --------
        >>> from neurospatial import Intervals
        >>> runs = Intervals.from_arrays([5.0, 0.0], [9.0, 4.0], speed=[30.0, 12.0])
        >>> runs.start.tolist(), runs.speed.tolist(), runs.interval_ids.tolist()
        ([0.0, 5.0], [12.0, 30.0], [1, 0])
        """
        starts = np.asarray(start, dtype=np.float64)
        stops = np.asarray(stop, dtype=np.float64)
        if starts.ndim != 1 or stops.shape != starts.shape:
            msg = (
                f"start and stop must be 1-D arrays of equal length; got shapes "
                f"{starts.shape} and {stops.shape}."
            )
            raise ValueError(msg)
        columns: dict[str, NDArray[Any]] = {}
        for name, values in fields.items():
            array = np.asarray(values)
            if array.dtype.kind in "OUS":
                array = _object_array(list(values))  # type: ignore[arg-type]
            if array.shape != starts.shape:
                msg = (
                    f"field {name!r} needs one value per interval ({starts.size}); "
                    f"got shape {array.shape}."
                )
                raise ValueError(msg)
            columns[name] = array
        return _from_columns(starts, stops, columns, interval_ids=interval_ids)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        *,
        start_column: str = "start_time",
        stop_column: str = "stop_time",
        id_column: str | None = None,
    ) -> Intervals[Interval]:
        """Build :class:`Interval` rows from a table, one row per interval.

        Every other column becomes a field, in column order. Missing values in
        a text column (``NaN``, ``None``) become ``None``.

        Parameters
        ----------
        df : pandas.DataFrame
            For example a task log or ``nwbfile.trials.to_dataframe()``.
        start_column, stop_column : str
            Columns with the bounds in seconds; a row is
            ``[start, stop)``.
        id_column : str, optional
            Column with stable integer ids (for example trial numbers). Default:
            ``0`` to ``n - 1`` in table order.

        Returns
        -------
        Intervals of Interval

        Raises
        ------
        ValueError
            If a named column is missing.
        """
        import pandas as pd

        if not isinstance(df, pd.DataFrame):
            msg = f"from_dataframe needs a pandas DataFrame; got {type(df).__name__}."
            raise TypeError(msg)
        named = {
            "start_column": start_column,
            "stop_column": stop_column,
            "id_column": id_column,
        }
        absent = [
            f"{arg}={col!r}"
            for arg, col in named.items()
            if col is not None and col not in df.columns
        ]
        if absent:
            raise ValueError(
                _format_error(
                    f"from_dataframe: no column {', '.join(absent)} in the table.",
                    why=f"its columns are {list(map(str, df.columns))}.",
                    fix="name the start, stop and id columns with start_column=, "
                    "stop_column= and id_column=.",
                )
            )
        used = {start_column, stop_column, id_column}
        columns: dict[str, NDArray[Any]] = {}
        for name in df.columns:
            if name in used:
                continue
            series = df[name]
            if series.dtype.kind in "biuf":
                columns[str(name)] = series.to_numpy()
            else:
                columns[str(name)] = _object_array(
                    [_missing_as_none(v) for v in series.tolist()]
                )
        return _from_columns(
            df[start_column].to_numpy(dtype=np.float64),
            df[stop_column].to_numpy(dtype=np.float64),
            columns,
            interval_ids=None if id_column is None else df[id_column].to_numpy(),
        )

    def __repr__(self) -> str:
        name = f"Intervals[{self._record_type.__name__}]"
        n = len(self)
        span = f" in [{self._start.min():g}, {self._stop.max():g}) s" if n else ""
        fields = ", ".join(self._fields) if self._fields else "none"
        return f"{name}: {n} interval{'s' if n != 1 else ''}{span}; fields: {fields}"


_MEMBERS = frozenset(name for name in dir(Intervals) if not name.startswith("_")) | {
    *_TIME_FIELDS,
}


def _missing_as_none(value: Any) -> Any:
    """``None`` for a missing table value (``NaN``, ``None``, ``pd.NA``)."""
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if type(value).__name__ in {"NAType", "NaTType"}:
        return None
    return value


def _id_span(ids: NDArray[np.int64]) -> str:
    return f"[{ids.min()}, {ids.max()}]" if ids.size else "nothing (no intervals)"


def _infer_type(rows: tuple[Any, ...]) -> type:
    if not rows:
        raise TypeError(
            _format_error(
                "Intervals() got no records and no record_type.",
                why="the record type sets the field schema, which an empty "
                "selection keeps.",
                fix="pass record_type= (and fields= for Interval records), or "
                "select zero rows from an existing Intervals.",
            )
        )
    return type(rows[0])


def _schema(
    kind: type, rows: tuple[Any, ...], fields: Sequence[str] | None
) -> tuple[tuple[str, ...], dict[str, np.dtype]]:
    """Field names and default dtypes for ``kind`` records."""
    if kind is not Interval:
        if fields is not None:
            msg = (
                f"fields= applies to Interval records; {kind.__name__} records "
                "take their fields from the class."
            )
            raise TypeError(msg)
        return _record_schema(kind)
    if fields is None:
        if not rows:
            raise TypeError(
                _format_error(
                    "Intervals of no Interval records needs fields=.",
                    why="the field schema is kept by empty selections.",
                    fix="pass fields=() or the field names, e.g. "
                    "fields=('instructed_goal',).",
                )
            )
        names = tuple(name for name, _ in rows[0].items)
    else:
        names = tuple(fields)
    expected = set(names)
    for i, row in enumerate(rows):
        got = [name for name, _ in row.items]
        if set(got) != expected or len(got) != len(names):
            raise ValueError(
                _format_error(
                    f"Interval {i} has fields {got}, not {list(names)}.",
                    why="every row of an Intervals has the same fields.",
                    fix="give every Interval the same field names; use None for a "
                    "missing value.",
                )
            )
    return names, dict.fromkeys(names, np.dtype(object))


def _from_columns(
    start: NDArray[np.float64],
    stop: NDArray[np.float64],
    columns: dict[str, NDArray[Any]],
    *,
    interval_ids: ArrayLike | None,
) -> Intervals[Interval]:
    """Build Interval rows from parallel columns, keeping column dtypes."""
    for name in columns:
        _check_field_name(name, owner="Intervals", reserved=_MEMBERS)
    names = tuple(columns)
    rows = tuple(
        Interval(
            float(start[i]),
            float(stop[i]),
            tuple((name, columns[name][i]) for name in names),
        )
        for i in range(start.size)
    )
    ids = _checked_ids(
        np.arange(start.size) if interval_ids is None else interval_ids,
        start.size,
        owner="Intervals",
    )
    order = np.lexsort((ids, stop, start))
    intervals: Intervals[Interval] = object.__new__(Intervals)
    intervals._init_parts(
        records=tuple(rows[i] for i in order),
        record_type=Interval,
        fields=names,
        # Each column keeps its dtype, for integer fields and empty selections.
        dtypes={
            name: array.dtype if array.dtype.kind in "biuf" else np.dtype(object)
            for name, array in columns.items()
        },
        ids=ids[order],
        start=start[order],
        stop=stop[order],
        settings={},
    )
    return intervals
