"""Coerce spike input into the canonical spike-trains structure.

This module provides helpers to convert the various ways a user might pass
spike data into one canonical structure: a list of 1D NumPy arrays, one array
per neuron. It changes the *container shape* only -- the spike-time values are
never shifted, rescaled, or aligned.

The main function is `as_spike_trains()`, which accepts:

- 1D array (single neuron) → wrapped in list
- 2D array (n_neurons, max_spikes) with trailing NaN padding → split, padding
  removed
- list/tuple of scalars (single neuron) → converted to 1D array, wrapped in list
- list/tuple of 1D arrays (canonical format) → each element converted to array
- one unit's train (pynapple ``Ts``, pandas ``Series``) → wrapped in list
- a spike group (pynapple ``TsGroup``, ``SpikeTrains``, dict of trains) →
  one train per unit

Every spike time must be finite. This coercion happens at the entry point of encoding, decoding and peri-event
functions, ensuring consistent internal handling regardless of how the user
provides spike data.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error
from neurospatial._intervals import _is_pynapple_object
from neurospatial._pynapple import is_pynapple_type, tsgroup_unit_table
from neurospatial._results import as_label_array, plain_value, resolve_unit_ids

if TYPE_CHECKING:
    import pandas as pd


def as_spike_trains(
    spike_times: Any,
) -> list[NDArray[np.float64]]:
    """Coerce spike input into the canonical list-of-spike-trains structure.

    Returns a list of 1D arrays, one per neuron — the canonical structure the
    encoding/decoding functions consume. This standardizes the *container
    shape* only; the spike-time values are not shifted, rescaled, or aligned.

    Parameters
    ----------
    spike_times : array, sequence of arrays, single train or spike group
        Spike times in seconds, in one of these formats:

        - 1D array (single neuron) → wrapped in list
        - 2D array (n_neurons, max_spikes) → split along axis 0, trailing NaN
          padding removed from each row
        - List/tuple of scalars (single neuron) → converted to 1D array,
          wrapped in list (e.g., ``[0.1, 0.5, 1.2]``)
        - List/tuple of 1D arrays (canonical format) → each element converted
          to float64 array
        - One unit's train (pynapple ``Ts``/``Tsd``, pandas ``Series``) →
          its timestamps (``.t``) or values, wrapped in list
        - A spike group (pynapple ``TsGroup``, ``SpikeTrains``, or a dict
          mapping unit id to train) → one train per unit, in the group's order

    Returns
    -------
    list[NDArray[np.float64]]
        List of 1D spike time arrays, one per neuron. Each array contains
        the spike times for that neuron in float64 dtype.

    Raises
    ------
    ValueError
        If a spike time is NaN or infinite (other than trailing padding of a
        2D array), if the input is a ragged object array, a DataFrame, a
        scalar or more than 2D, if a train is not 1D, or if a 2D array is a
        ``(n, 1)`` column or has its units in columns.

    Examples
    --------
    Single neuron (1D array):

    >>> import numpy as np
    >>> from neurospatial._spikes import as_spike_trains
    >>> spikes = np.array([0.1, 0.5, 1.2])
    >>> normalized = as_spike_trains(spikes)
    >>> len(normalized)
    1
    >>> normalized[0]
    array([0.1, 0.5, 1.2])

    Single neuron (list of scalars - common user input):

    >>> spikes = [0.1, 0.5, 1.2]  # Plain list of floats
    >>> normalized = as_spike_trains(spikes)
    >>> len(normalized)
    1
    >>> normalized[0]
    array([0.1, 0.5, 1.2])

    Multiple neurons (list of arrays):

    >>> spikes = [np.array([0.1, 0.5]), np.array([0.2, 0.3, 0.8])]
    >>> normalized = as_spike_trains(spikes)
    >>> len(normalized)
    2

    NaN-padded 2D array:

    >>> spikes = np.array([[0.1, 0.5, np.nan], [0.2, 0.3, 0.8]])
    >>> normalized = as_spike_trains(spikes)
    >>> normalized[0]  # NaN removed
    array([0.1, 0.5])
    >>> normalized[1]  # No NaN to remove
    array([0.2, 0.3, 0.8])

    Empty list returns empty list:

    >>> as_spike_trains([])
    []
    """
    return _as_spike_trains_with_ids(spike_times)[0]


_SPIKE_FORMS = (
    "a 1-D array for one unit, a list of 1-D arrays (one per unit), a 2-D "
    "(n_units, max_spikes) array padded with trailing NaN, or a spike group "
    "(pynapple TsGroup, SpikeTrains, dict of unit id to train)"
)


def _one_d_train(train: Any, unit: int | None) -> NDArray[np.float64]:
    """Return one train as a 1-D float64 array."""
    arr = np.asarray(train, dtype=np.float64)
    if arr.ndim != 1:
        which = "spike_times" if unit is None else f"spike_times for unit {unit}"
        raise ValueError(
            _format_error(
                f"{which} must be 1-D, got shape {arr.shape}.",
                why=f"Why: spike input is {_SPIKE_FORMS}.",
                fix="pass one 1-D array of spike times per unit.",
            )
        )
    return arr


def require_finite_spike_times(
    trains: Sequence[NDArray[np.float64]], unit_ids: NDArray[Any] | None = None
) -> None:
    """Refuse NaN or infinite spike times, naming the unit by its label.

    Parameters
    ----------
    trains : sequence of ndarray, each shape (n_spikes,)
        Coerced spike trains, in seconds.
    unit_ids : ndarray, shape (n_units,), or None
        Labels used in the message; positions when None.
    """
    for i, train in enumerate(trains):
        bad = np.flatnonzero(~np.isfinite(train))
        if bad.size:
            label = i if unit_ids is None else plain_value(unit_ids[i])
            raise ValueError(
                _format_error(
                    f"spike_times for unit {label!r} has {bad.size} NaN or "
                    f"infinite value(s), first at index {int(bad[0])}.",
                    why="Why: a spike time is an observed event; a missing one "
                    "cannot be counted in any time bin, so dropping it "
                    "silently would bias counts low.",
                    fix="remove missing spikes before the call, e.g. "
                    "train[np.isfinite(train)].",
                )
            )


def _array_spike_trains(spike_times: Any) -> list[NDArray[np.float64]]:
    """Coerce array or sequence spike input (no unit labels)."""
    # A list of scalars is one unit, and a ragged list of trains cannot become
    # a regular array, so read sequences before calling np.asarray.
    if isinstance(spike_times, (list, tuple)):
        if len(spike_times) == 0:
            return []
        if isinstance(spike_times[0], (int, float, np.floating, np.integer)):
            return [_one_d_train(spike_times, None)]
        return [_one_d_train(row, i) for i, row in enumerate(spike_times)]

    arr = np.asarray(spike_times)
    if arr.dtype == object:
        raise ValueError(
            _format_error(
                "Received ragged array (dtype=object).",
                why=f"Why: spike input is {_SPIKE_FORMS}.",
                fix="pass spike times as a list of 1D arrays, e.g. "
                "[np.array([0.1, 0.2]), np.array([0.3])].",
            )
        )
    if arr.ndim == 1:
        return [_one_d_train(arr, None)]
    if arr.ndim != 2:
        raise ValueError(
            _format_error(
                f"spike_times must be 1-D or 2-D, got shape {arr.shape}.",
                why=f"Why: spike input is {_SPIKE_FORMS}.",
                fix="pass a 1-D array for one unit or a list of 1-D arrays.",
            )
        )
    return _padded_spike_trains(arr.astype(np.float64))


def _padded_spike_trains(arr: NDArray[np.float64]) -> list[NDArray[np.float64]]:
    """Split a 2-D ``(n_units, max_spikes)`` array, removing trailing NaN."""
    n_rows, n_cols = arr.shape
    if n_cols == 1 and n_rows > 1:
        raise ValueError(
            _format_error(
                f"spike_times has shape {arr.shape}, a single column of "
                f"{n_rows} values.",
                why="Why: a 2-D array is read as one row per unit "
                f"(n_units, max_spikes), so this would be {n_rows} units with "
                "one spike each. MATLAB and HDF5 exports often store one "
                "unit's spike times this way.",
                fix="for one unit's spike times pass spike_times[:, 0]; for "
                f"{n_rows} units with at most one spike each pass a list of "
                "1-D arrays.",
            )
        )
    rows_decrease = bool(np.any(np.diff(arr, axis=1) < 0))
    if rows_decrease and n_rows > 1 and not np.any(np.diff(arr, axis=0) < 0):
        raise ValueError(
            _format_error(
                f"spike_times of shape {arr.shape} looks transposed: its "
                "columns increase but its rows do not.",
                why="Why: a 2-D array is read as one row per unit "
                "(n_units, max_spikes).",
                fix="if each column is one unit, pass spike_times.T.",
            )
        )
    present = ~np.isnan(arr)
    interior = np.flatnonzero((~present[:, :-1] & present[:, 1:]).any(axis=1))
    if interior.size:
        raise ValueError(
            _format_error(
                f"spike_times row(s) {interior[:5].tolist()} have NaN before a "
                "spike time.",
                why="Why: NaN in a 2-D array only pads the end of shorter "
                "rows; NaN before a spike is missing data, not padding.",
                fix="remove missing spikes and pass a list of 1-D arrays, or "
                "move the padding to the end of each row.",
            )
        )
    return [row[keep] for row, keep in zip(arr, present, strict=True)]


def _single_train(obj: Any) -> Any | None:
    """Return the spike times of a one-unit train object, else None.

    A pynapple ``Ts``/``Tsd`` exposes its timestamps as ``.t``; a pandas
    ``Series`` holds them as values. Both have a non-callable ``.index``
    (timestamps or row labels), so they must be recognized before a spike
    group, whose ``.index`` holds unit ids.
    """
    if isinstance(obj, Mapping):
        return None
    if _is_pynapple_object(obj):
        t = getattr(obj, "t", None)
        if t is not None and np.ndim(t) == 1:
            return t
    if getattr(obj, "ndim", None) == 1 and hasattr(obj, "to_numpy"):
        return obj.to_numpy()
    return None


def _looks_like_spike_group(obj: Any) -> bool:
    """Return whether ``obj`` is a spike group carrying unit ids.

    A group is a Mapping from unit id to train (a pynapple ``TsGroup`` or a
    dict), or another non-array object with a non-callable ``.index`` of unit
    ids (``SpikeTrains``). A ``list``/``tuple`` has an ``.index`` *method*, and
    a NumPy array has no ``.index``, so both stay on the array path. One
    unit's train (``Ts``, ``Series``) is detected first by
    :func:`_single_train`.
    """
    if isinstance(obj, (list, tuple, np.ndarray)):
        return False
    if isinstance(obj, Mapping):
        return True
    index = getattr(obj, "index", None)
    return index is not None and not callable(index)


def _coerce_spike_input(
    spike_times: Any,
) -> tuple[list[NDArray[np.float64]], NDArray[Any] | None]:
    """Split spike input into 1-D trains and any unit ids it carries."""
    single = _single_train(spike_times)
    if single is not None:
        return [_one_d_train(single, None)], None
    if getattr(spike_times, "ndim", None) == 2 and hasattr(spike_times, "columns"):
        raise ValueError(
            _format_error(
                "spike_times is a DataFrame.",
                why="Why: a table does not say whether its columns are units "
                "or fields such as unit and time.",
                fix="pass one array per unit, e.g. "
                "{c: df[c].dropna().to_numpy() for c in df} for one column per "
                "unit, or {u: g['time'].to_numpy() for u, g in "
                "df.groupby('unit')} for a long table.",
            )
        )
    if not _looks_like_spike_group(spike_times):
        return _array_spike_trains(spike_times), None

    index = getattr(spike_times, "index", None)
    if isinstance(spike_times, Mapping):
        keys = index if index is not None and not callable(index) else spike_times
        unit_ids = as_label_array(list(keys))
        trains = [
            _one_d_train(getattr(spike_times[uid], "t", spike_times[uid]), i)
            for i, uid in enumerate(unit_ids)
        ]
    else:
        unit_ids = as_label_array(list(spike_times.index))
        trains = [_one_d_train(t, i) for i, t in enumerate(spike_times)]
    if getattr(spike_times, "_unit_ids_generated", False):
        # Generated ``arange`` labels are not caller-supplied identity.
        return trains, None
    return trains, unit_ids


def spike_unit_table(spike_times: Any) -> pd.DataFrame | None:
    """Return the per-unit metadata a spike input carries, or None.

    A :class:`~neurospatial.encoding.SpikeTrains` carries its ``unit_table``;
    a pynapple ``TsGroup`` carries its metadata without pynapple's computed
    ``rate`` column (see :func:`neurospatial._pynapple.tsgroup_unit_table`).
    Other inputs carry none.

    Parameters
    ----------
    spike_times : any spike input

    Returns
    -------
    pandas.DataFrame or None
        One row per unit, or None.
    """
    if is_pynapple_type(spike_times, "TsGroup"):
        return tsgroup_unit_table(spike_times)
    table: pd.DataFrame | None = getattr(spike_times, "unit_table", None)
    return table


def _as_spike_trains_with_ids(
    spike_times: Any,
) -> tuple[list[NDArray[np.float64]], NDArray[Any] | None]:
    """Coerce spike input to canonical trains AND surface unit ids when present.

    Like :func:`as_spike_trains`, but also returns the unit ids a spike group
    carries, so identity is not dropped. Mapping groups (pynapple ``TsGroup``,
    dict) are read by ``group[uid]``, because iterating a Mapping yields its
    keys; their ids are ``.index`` when present, else the keys. Other groups
    (``SpikeTrains``) are iterated. Detection is duck-typed and never imports
    pynapple.

    Parameters
    ----------
    spike_times : array, sequence of arrays, single train or spike group
        Any input :func:`as_spike_trains` accepts.

    Returns
    -------
    trains : list[NDArray[np.float64]]
        Per-unit spike-time arrays, exactly as :func:`as_spike_trains` produces.
    unit_ids : NDArray or None
        Unit ids carried by a spike group (one per train), or ``None`` for
        input without ids and for a group whose labels were generated rather
        than supplied (an object with ``_unit_ids_generated`` set, such as
        ``SpikeTrains(trains)``).

    Examples
    --------
    Plain sequence input carries no ids:

    >>> import numpy as np
    >>> from neurospatial._spikes import _as_spike_trains_with_ids
    >>> trains, ids = _as_spike_trains_with_ids([np.array([0.1]), np.array([0.2])])
    >>> len(trains), ids
    (2, None)
    >>> trains, ids = _as_spike_trains_with_ids({"a": [0.1], "b": [0.2, 0.3]})
    >>> ids.tolist()
    ['a', 'b']
    """
    trains, unit_ids = _coerce_spike_input(spike_times)
    require_finite_spike_times(trains, unit_ids)
    return trains, unit_ids


def as_labelled_spike_trains(
    spike_times: Any,
    unit_ids: NDArray[Any] | Sequence[Any] | None,
    *,
    context: str,
) -> tuple[list[NDArray[np.float64]], NDArray[Any], bool]:
    """Coerce population spike input and resolve one label per train.

    Parameters
    ----------
    spike_times : array, sequence of arrays, or spike group
        Any input :func:`_as_spike_trains_with_ids` accepts.
    unit_ids : array-like or None
        Caller-supplied labels, one per train. A spike group that carries its
        own labels must agree with them.
    context : str
        Name of the calling function, used in error messages.

    Returns
    -------
    trains : list of ndarray, each shape (n_spikes,)
    unit_ids : ndarray, shape (n_units,)
        ``unit_ids``, else the labels a spike group carries, else ``arange``.
    labels_supplied : bool
        Whether the labels came from the caller (``unit_ids`` or a spike
        group's own labels) rather than the ``arange`` default.

    Raises
    ------
    ValueError
        If the labels do not match the number of trains, are duplicated, or
        disagree with the labels a spike group carries, or if a spike time is
        NaN or infinite (the message names the unit by its label).

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial._spikes import as_labelled_spike_trains
    >>> trains = [np.array([0.1, 0.5]), np.array([0.2])]
    >>> _, ids, supplied = as_labelled_spike_trains(trains, None, context="demo")
    >>> ids.tolist(), supplied
    ([0, 1], False)
    >>> _, ids, supplied = as_labelled_spike_trains(trains, ["a", "b"], context="demo")
    >>> ids.tolist(), supplied
    (['a', 'b'], True)
    """
    trains, input_ids = _coerce_spike_input(spike_times)
    resolved = resolve_unit_ids(
        unit_ids, len(trains), context=context, input_ids=input_ids
    )
    require_finite_spike_times(trains, resolved)
    return trains, resolved, unit_ids is not None or input_ids is not None
