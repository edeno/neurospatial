"""Array-native epoch selection (``restrict`` / ``in_epochs``).

"Give me my running periods" / "give me trial N" -- restrict time-series and
spike data to a set of time intervals (epochs). Epochs are plain
``(start, end)`` arrays, so this stays array-first and never imports pynapple;
a pynapple ``IntervalSet`` is nonetheless accepted **transparently** because it
is *duck-typed* (it exposes ``.start`` / ``.end`` arrays), not ``isinstance``-d.
Its inclusive ends become half-open stops, so the default ``closed="left"``
keeps the same samples as pynapple's ``restrict``.

Functions
---------
restrict
    The headline one-liner: ``t, pos = restrict(times, positions,
    epochs=run_epochs)``. Slices ``times`` and any number of arrays **aligned to
    ``times``** (e.g. ``positions``) by the same in-epoch mask, order preserved.
    With no extra arrays it restricts an event-time array by its own timestamps
    (``restrict(spike_train, epochs=...)`` -> the in-epoch spikes).
in_epochs
    The boolean mask primitive: ``True`` where ``t`` falls inside **any**
    interval. Half-open ``[start, stop)`` by default (``closed="left"``), the
    same windows every analysis ``epochs=`` argument and
    :func:`labels_to_epochs` use, so adjacent windows never share a point.
restrict_spike_trains
    Ragged per-unit spike times are **not** aligned to a common ``times``; each
    train is restricted by its **own** timestamps. Accepts any spike input and
    returns a :class:`~neurospatial.encoding.SpikeTrains` that keeps the unit
    labels and metadata.
labels_to_epochs
    Turn one label per sample (task phase, running direction, trial type) into
    an ``Intervals`` of half-open ``[start, stop)`` epochs with a ``label``
    field; select one condition (``conditions[conditions.label == "out"]``)
    and pass it as ``epochs=`` to an analysis. A missing sample is bridged like a deleted row when the
    gap around it is no longer than ``max_gap``; a longer gap, a declared
    break or an ``epochs`` edge ends an epoch.

Aligned vs ragged
-----------------
``restrict`` is for arrays that share one time axis (position samples, a single
spike train). ``restrict_spike_trains`` is for a *ragged* collection where each
unit has its own timestamps and there is no shared ``times`` -- so each train is
masked against itself.
"""

from __future__ import annotations

import warnings
from collections.abc import Hashable
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error
from neurospatial._interval_table import Interval, Intervals
from neurospatial._intervals import (
    DEFAULT_MAX_GAP,
    as_intervals,
    hold_last_valid,
    intervals_contain,
    times_in_intervals,
    validate_max_gap,
)
from neurospatial._observation import (
    _validated_sample_flags,
    _validated_signal_times,
    label_codes,
    label_runs,
    resolve_sample_support,
    validate_break_before,
)
from neurospatial._pynapple import reads_pynapple_samples
from neurospatial._results import as_label_array
from neurospatial._stacklevel import _external_stacklevel
from neurospatial._validation import validate_choice

if TYPE_CHECKING:
    from neurospatial.encoding.spike_trains import SpikeTrains

__all__ = [
    "in_epochs",
    "labels_to_epochs",
    "restrict",
    "restrict_spike_trains",
]

_Closed = Literal["both", "left", "right", "neither"]
_CLOSED_CHOICES = ("both", "left", "right", "neither")


def _mask_in_intervals(
    t: NDArray[np.float64],
    intervals: NDArray[np.float64] | None,
    *,
    closed: _Closed,
) -> NDArray[np.bool_]:
    """Boolean mask of ``t`` against PRE-normalized ``(n, 2)`` intervals.

    The vectorized core shared by :func:`in_epochs` and
    :func:`restrict_spike_trains`. ``intervals`` must already be an ``(n, 2)``
    ``float64`` array (as :func:`~neurospatial._intervals.as_intervals` returns), so a caller restricting
    many trains against one epoch set can normalize **once** and reuse this per
    train instead of re-normalizing the same intervals for every unit.

    Parameters
    ----------
    t : ndarray
        Timestamps to test (any shape; the mask has the same shape).
    intervals : ndarray, shape (n_intervals, 2)
        Normalized ``[start, end]`` rows (from :func:`~neurospatial._intervals.as_intervals`).
    closed : {"both", "left", "right", "neither"}
        Which endpoints are inclusive.

    Returns
    -------
    ndarray of bool
        Same shape as ``t``; ``True`` where ``t`` is inside any interval.

    Notes
    -----
    Callers validate ``closed`` before normalizing ``epochs``, so a bad value
    is reported first.
    """
    t_arr = np.asarray(t, dtype=np.float64)

    if intervals is None:
        return np.ones(t_arr.shape, dtype=np.bool_)
    if closed == "left":
        # The half-open windows of every analysis ``epochs=``.
        return times_in_intervals(intervals, t_arr)
    if intervals.shape[0] == 0:
        return np.zeros(t_arr.shape, dtype=np.bool_)

    # Rows are sorted and separated, so the only row that can hold a time is
    # the last one starting at or before it. A NaN time sorts last and fails
    # both comparisons.
    starts = intervals[:, 0]
    row = np.searchsorted(starts, t_arr, side="right") - 1
    start = starts[np.maximum(row, 0)]
    end = intervals[np.maximum(row, 0), 1]
    after_start = t_arr >= start if closed in ("both", "left") else t_arr > start
    before_end = t_arr <= end if closed in ("both", "right") else t_arr < end
    mask: NDArray[np.bool_] = (row >= 0) & after_start & before_end
    return mask


def in_epochs(
    t: NDArray[np.float64],
    epochs: Any,
    *,
    closed: _Closed = "left",
) -> NDArray[np.bool_]:
    """Boolean mask: ``True`` where each element of ``t`` falls in any epoch.

    Vectorized over samples (broadcasting, no Python loop) and taking the
    **union** across intervals (overlapping intervals behave as their union).

    Parameters
    ----------
    t : ndarray
        Timestamps to test (any shape; the mask has the same shape).
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Windows in seconds, on the same clock as the input timestamps. Rows
        are sorted and overlapping or touching rows merge. ``None`` leaves
        data unrestricted. Empty windows, non-finite endpoints, and rows with
        ``stop <= start`` raise ``ValueError`` with a fix. Parallel starts/ends
        arrays are not an input form; pass an ``(n, 2)`` array instead.
        ``closed`` controls point membership in the normalized windows.
    closed : {"both", "left", "right", "neither"}, optional
        Which endpoints are inclusive. Default ``"left"``: ``start <= t <
        stop``, the half-open windows of every analysis ``epochs=`` and
        ``spike_window=`` argument, so a spike exactly at the stop of one
        window and the start of the next lands in exactly one of them.
        A pynapple ``IntervalSet``'s inclusive ends become half-open stops,
        so the default already matches pynapple's ``restrict``; use ``"both"``
        for plain arrays of pynapple bounds. Touching rows
        merge first, so a boundary shared by two rows is inside the merged
        window for every ``closed`` value.

    Returns
    -------
    ndarray of bool
        Same shape as ``t``; ``True`` where ``t`` is inside any interval.

    See Also
    --------
    restrict : Slice times and aligned arrays to the epochs.
    neurospatial.events.filter_by_intervals : Keep the rows of an events
        DataFrame that fall in Intervals.

    Notes
    -----
    NaN timestamps evaluate ``False`` at every ``>=``/``<=``/``>``/``<``
    comparison, so they are (deliberately) excluded from every epoch.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.behavior import in_epochs
    >>> in_epochs(np.array([0.0, 2.5, 5.0, 6.0]), (0.0, 5.0))
    array([ True,  True, False, False])
    >>> in_epochs(np.array([0.0, 2.5, 5.0, 6.0]), (0.0, 5.0), closed="both")
    array([ True,  True,  True, False])
    """
    # Validate ``closed`` up front -- before ``as_intervals`` and the
    # empty-epochs early return -- so a bad value always raises (even with no
    # intervals) AND is reported before a malformed ``epochs``.
    validate_choice(closed, _CLOSED_CHOICES, name="closed", owner="in_epochs()")

    intervals = as_intervals(epochs, name="epochs")
    return _mask_in_intervals(t, intervals, closed=closed)


@reads_pynapple_samples
def restrict(
    times: NDArray[np.float64],
    *arrays: NDArray[Any],
    epochs: Any,
    closed: _Closed = "left",
) -> NDArray[Any] | tuple[NDArray[Any], ...]:
    """Restrict ``times`` and time-aligned arrays to a set of epochs.

    The headline one-liner::

        t, pos = restrict(times, positions, epochs=run_epochs)

    ``times`` is the reference 1-D time axis. Each of ``*arrays`` must be
    **aligned to ``times``** (``len(arr) == len(times)``; e.g. ``positions`` of
    shape ``(n, n_dims)``). All are sliced by the single mask
    ``in_epochs(times, epochs, closed=closed)``, so alignment and order are
    preserved.

    With **no** extra arrays, ``restrict`` restricts an event-time array by its
    own timestamps -- ``restrict(spike_train, epochs=...)`` returns the in-epoch
    spikes (here ``times`` *is* the spike train). For ragged per-unit spikes
    (each unit its own timestamps, no shared axis), use
    :func:`restrict_spike_trains` instead.

    For occupancy, firing rates, velocity or decoding, pass the original arrays
    with ``epochs=`` to the analysis. Sliced arrays cannot encode the excluded
    time between windows, so an analysis may reconnect nearby windows, and the
    sample on a window's stop is left out, so the window's last sampling
    interval is lost. This function warns when aligned arrays would lose
    either.
    Windows are half-open by default, so disjoint train/test windows split
    point events without sharing any.

    Parameters
    ----------
    times : ndarray, shape (n,)
        Reference time array.
    *arrays : ndarray
        Zero or more arrays aligned to ``times`` (first axis length ``n``).
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Windows in seconds, on the same clock as the input timestamps. Rows
        are sorted and overlapping or touching rows merge. ``None`` leaves
        data unrestricted. Empty windows, non-finite endpoints, and rows with
        ``stop <= start`` raise ``ValueError`` with a fix. Parallel starts/ends
        arrays are not an input form; pass an ``(n, 2)`` array instead.
        ``closed`` controls point membership in the normalized windows.
    closed : {"both", "left", "right", "neither"}, optional
        Endpoint inclusivity, forwarded to :func:`in_epochs`. Default
        ``"left"``: ``start <= t < stop``, the half-open windows of every
        analysis ``epochs=`` argument. A pynapple ``IntervalSet`` already
        matches pynapple's ``restrict`` under the default; use ``"both"`` for
        plain arrays of pynapple bounds.

    Returns
    -------
    ndarray or tuple of ndarray
        If no extra arrays were passed, the masked ``times`` (a bare array).
        Otherwise ``(times_kept, *arrays_kept)`` -- each sliced by the same
        mask, order preserved.

    Raises
    ------
    ValueError
        If any array's first-axis length differs from ``len(times)`` (the error
        names the offending positional index).

    See Also
    --------
    in_epochs : The boolean mask this function slices by.
    restrict_spike_trains : Restrict ragged per-unit spike trains.
    neurospatial.events.filter_by_intervals : Keep the rows of an events
        DataFrame that fall in Intervals.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.behavior import restrict
    >>> times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    >>> positions = np.column_stack([times, times])
    >>> t, pos = restrict(times, positions, epochs=(1.0, 2.5))
    >>> t
    array([1., 2.])
    """
    times_arr = np.asarray(times)
    n = len(times_arr)

    coerced: list[NDArray[Any]] = []
    for i, arr in enumerate(arrays):
        arr_np = np.asarray(arr)
        if len(arr_np) != n:
            msg = (
                f"array at position {i} has len {len(arr_np)} but must be "
                f"aligned to times (len {n}); every *arrays entry must have the "
                "same first-axis length as times."
            )
            raise ValueError(msg)
        coerced.append(arr_np)

    mask = in_epochs(times_arr, epochs, closed=closed)

    times_kept = times_arr[mask]
    if not coerced:
        return times_kept
    intervals = as_intervals(epochs, name="epochs")
    if intervals is not None:
        # The first sample dropped after a kept one, when it sits on a window
        # stop, closes that window's last interval, which slicing loses.
        closing = np.flatnonzero(mask[:-1] & ~mask[1:]) + 1
        loses_last = bool(np.isin(times_arr[closing], intervals[:, 1]).any())
        bridges = len(times_kept) > 1 and not np.all(
            intervals_contain(intervals, times_kept[:-1], times_kept[1:])
        )
        if loses_last or bridges:
            warnings.warn(
                "restrict() returns samples without their observation windows: "
                "it keeps samples in [start, stop), so the interval that ends "
                "at a window's stop is lost, and joining samples can bridge "
                "excluded time. For occupancy, rates, movement or decoding, pass "
                "the original times and arrays with epochs= to the analysis. If "
                "you use these sliced arrays, pass epochs= again; max_gap alone "
                "is insufficient.",
                UserWarning,
                stacklevel=_external_stacklevel(),
            )
    return (times_kept, *(arr[mask] for arr in coerced))


@reads_pynapple_samples
def restrict_spike_trains(
    spike_times: Any,
    epochs: Any,
    *,
    closed: _Closed = "left",
) -> SpikeTrains:
    """Restrict each ragged per-unit spike train by its **own** timestamps.

    Ragged per-unit spike times are not aligned to a common ``times`` axis, so
    each train is masked against itself: ``t[in_epochs(t, epochs)]`` for every
    train ``t``. Unit labels and per-unit metadata are kept.

    Parameters
    ----------
    spike_times : array, sequence of arrays, or spike group
        Per-unit spike times in seconds, in any form an analysis accepts: a
        list of 1-D arrays, a :class:`~neurospatial.encoding.SpikeTrains`, a
        pynapple ``TsGroup`` or a ``dict`` from unit id to train.
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Windows in seconds, on the same clock as the input timestamps. Rows
        are sorted and overlapping or touching rows merge. ``None`` leaves
        data unrestricted. Empty windows, non-finite endpoints, and rows with
        ``stop <= start`` raise ``ValueError`` with a fix. Parallel starts/ends
        arrays are not an input form; pass an ``(n, 2)`` array instead.
        ``closed`` controls point membership in the normalized windows.
    closed : {"both", "left", "right", "neither"}, optional
        Endpoint inclusivity, forwarded to :func:`in_epochs`. Default
        ``"left"``: ``start <= t < stop``, the half-open windows of every
        analysis ``epochs=`` argument. A pynapple ``IntervalSet`` already
        matches pynapple's ``restrict`` under the default; use ``"both"`` for
        plain arrays of pynapple bounds.

    Returns
    -------
    SpikeTrains
        One restricted train per input unit, in the same order. A labelled
        input keeps its unit labels (a ``TsGroup``'s index, a ``dict``'s keys,
        a ``SpikeTrains``' ``unit_ids``) and its per-unit metadata (a
        ``SpikeTrains``' ``unit_table``, a ``TsGroup``'s metadata without
        pynapple's ``rate`` column); a plain list is numbered from 0. Pass it
        straight to an analysis, so each result row keeps its unit's label.

    Raises
    ------
    ValueError
        If ``epochs`` is invalid, or a spike time is NaN or infinite.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.behavior import restrict_spike_trains
    >>> spike_times = {5: np.array([0.1, 1.5, 2.9]), 7: np.array([0.5, 3.0, 6.0])}
    >>> kept = restrict_spike_trains(spike_times, (1.0, 3.5))
    >>> kept.unit_ids
    array([5, 7])
    >>> kept.sel(unit_id=7)
    array([3.])
    """
    from neurospatial._spikes import as_labelled_spike_trains, spike_unit_table
    from neurospatial.encoding.spike_trains import SpikeTrains

    # Normalize the intervals ONCE (not per train): a malformed epochs raises up
    # front, and each train is then masked against the same pre-normalized
    # ``(n, 2)`` array via ``_mask_in_intervals`` -- avoiding the per-unit
    # re-normalization that ``in_epochs`` would repeat on every call.
    validate_choice(
        closed, _CLOSED_CHOICES, name="closed", owner="restrict_spike_trains()"
    )
    intervals = as_intervals(epochs, name="epochs")
    trains, unit_ids = as_labelled_spike_trains(
        spike_times, None, context="restrict_spike_trains"
    )
    kept = [t[_mask_in_intervals(t, intervals, closed=closed)] for t in trains]
    return SpikeTrains(
        kept,
        unit_ids=unit_ids,
        unit_table=spike_unit_table(spike_times),
    )


def _label_array(labels: Any) -> NDArray[Any]:
    """Return labels as a 1-D array; tuples stay single labels."""
    if isinstance(labels, str | bytes) or not np.iterable(labels):
        return np.asarray(labels)
    if isinstance(labels, np.ndarray):
        return labels
    values = list(labels)
    if any(isinstance(v, tuple) for v in values):
        # Element-wise fill: np.asarray would broadcast tuples into a 2-D array.
        array = np.empty(len(values), dtype=object)
        for i, v in enumerate(values):
            array[i] = v
        return array
    return as_label_array(values)


@reads_pynapple_samples
def labels_to_epochs(
    times: Any,
    labels: Any,
    *,
    valid: Any = None,
    break_before: Any = None,
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    missing_label: Hashable | None = None,
) -> Intervals[Interval]:
    """Convert one label per sample into half-open epochs, one row per epoch.

    Use this to split a session into behavioral conditions (outbound and
    inbound runs, task phases, cue or trial types) and then estimate a rate
    map for each condition: select one condition with
    ``conditions[conditions.label == "outbound"]`` and pass it as ``epochs=``
    together with the original ``times``, ``positions`` and ``spike_times``.

    Interval ``k`` is ``[times[k], times[k + 1])`` and carries the label of
    its start sample, ``labels[k]``. An unobserved sample (``valid`` False)
    is skipped as if its row were deleted: an interval inside the gap
    between two observed samples carries the label of the observed sample
    that starts the gap. An interval is kept when its gap passes
    ``max_gap``, lies inside one ``epochs`` row and contains no declared
    break, and its label is not missing. Consecutive kept intervals with the
    same label form one epoch.

    Parameters
    ----------
    times : array-like, shape (n_samples,)
        Sample times in seconds; finite and strictly increasing.
    labels : array-like, shape (n_samples,)
        One hashable label per sample. ``None``, ``NaN`` and ``missing_label``
        mark a sample without a label. ``0`` and ``-1`` are ordinary labels.
        A tuple such as ``("arm1", "out")`` is one label, so a combined
        condition needs no string joining. The empty string is refused: pass
        ``missing_label=""`` if it means "no label". Labels are stored as
        plain Python values.
    valid : array-like of bool, shape (n_samples,), optional
        False marks a sample that was not observed (for example lost
        tracking). Default: every sample is valid.
    break_before : array-like of bool, shape (n_samples,), optional
        True at sample ``k`` declares a recording break between samples
        ``k - 1`` and ``k``; entry 0 is ignored.
    max_gap : float or None, default=1.0
        Longest stretch of missing tracking, in seconds, to bridge: the time
        between consecutive valid samples (a dropped frame and a NaN row are
        the same missing sample). Longer gaps are excluded as tracking losses;
        ``None`` bridges every gap. The default is calibrated for typical video
        and DeepLabCut tracking; choose a value longer than routine dropped or
        low-confidence frames and shorter than real tracking losses or pickups.
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Half-open windows in seconds; only intervals entirely inside one
        window are kept.
    missing_label : hashable, optional
        A label value that means "no label" (for example ``"none"`` or
        ``-1`` from a labeling function), excluded like ``None``.

    Returns
    -------
    Intervals of Interval
        One row per epoch, ``[start, stop)`` in seconds, in time order, ids
        ``0`` to ``n - 1``, with a ``label`` field. ``analysis_settings``
        records ``labels`` (the labels with epochs, in order of their first
        epoch), ``unsupported_labels`` (labels that start an interval but
        have no epoch, in first-seen order), ``max_gap``, ``epochs`` (the
        windows applied, or None), ``missing_label`` and
        ``n_declared_breaks`` (breaks declared by ``break_before``; entry 0
        is not a break).

    Raises
    ------
    ValueError
        If ``times`` is empty, not finite or not strictly increasing; if
        ``labels`` is not 1-D with one entry per sample or holds the empty
        string; or if ``valid``, ``break_before`` or ``max_gap`` is
        malformed.

    See Also
    --------
    neurospatial.Intervals.sample_values : Per-sample labels from laps or trials.

    Notes
    -----
    Time is allocated to the label at the start of each interval, so the
    final sample adds no time and a label held by a single sample before a
    change contributes one interval. The same label after a gap longer than
    ``max_gap`` (counting unobserved samples as missing rows), a declared
    break, an excluded window or a different label starts a new epoch; no
    duration is inferred across any of them. A label that starts intervals
    but has none kept has no row; it is listed in
    ``analysis_settings["unsupported_labels"]``.

    Labels here are disjoint by construction: each interval has one label,
    so the rows never overlap. Conditions passed to an analysis as separate
    ``epochs=`` sets may overlap elsewhere. Labels assigned after the fact,
    for example from the path the animal later took, describe the
    trajectory; they are not causal predictors of the animal's choice.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.behavior import labels_to_epochs
    >>> times = np.arange(10) / 10
    >>> labels = np.array(["out"] * 5 + ["in"] * 5)
    >>> valid = np.ones(10, dtype=bool)
    >>> valid[7] = False  # tracking lost at 0.7 s for one frame: bridged
    >>> conditions = labels_to_epochs(times, labels, valid=valid)
    >>> conditions.label.tolist()
    ['out', 'in']
    >>> conditions[conditions.label == "in"].windows
    array([[0.5, 0.9]])

    A shorter ``max_gap`` treats the lost frame as a tracking loss, which
    splits the inbound epoch:

    >>> split = labels_to_epochs(times, labels, valid=valid, max_gap=0.15)
    >>> split[split.label == "in"].windows
    array([[0.5, 0.6],
           [0.8, 0.9]])

    Tuples are single labels, for combined conditions:

    >>> labels = [("arm1", "out")] * 3 + [("arm1", "in")] * 2
    >>> labels_to_epochs(np.arange(5.0) / 10, labels).label.tolist()
    [('arm1', 'out'), ('arm1', 'in')]
    """
    call = "labels_to_epochs"
    t = _validated_signal_times(times, call=call)
    n_samples = t.size
    # as_label_array keeps mixed labels as objects: np.asarray would turn
    # [-1, "x"] into strings and ["a", nan] into the label "nan".
    label_array = _label_array(labels)
    if label_array.shape != (n_samples,):
        raise ValueError(
            _format_error(
                f"{call}() needs labels of shape ({n_samples},), one per sample "
                f"in times; got shape {label_array.shape}.",
                why="each label names the condition of the matching sample.",
                fix="pass a 1-D array of hashable labels aligned with times; use "
                "None or NaN for samples without a label.",
            )
        )
    sample_valid = _validated_sample_flags(valid, n_samples, call=call)
    validate_max_gap(max_gap)
    breaks = validate_break_before(break_before, n_samples, call=call)
    windows = as_intervals(epochs, name="epochs")

    sample_codes, values = label_codes(label_array[:-1], missing_label)
    if any(isinstance(value, str) and value == "" for value in values):
        raise ValueError(
            _format_error(
                f"{call}() got the empty string as a label.",
                why="each epoch stores its label as a field, and a missing field "
                'value is None: NWB writes None as "" and reads "" back as None, '
                "so an empty-string label would come back as missing.",
                fix='pass missing_label="" if the empty string means "no label", '
                "or give that condition a name.",
            )
        )
    # An interval inside a bridged gap carries the label of the gap's start.
    codes = hold_last_valid(sample_codes, sample_valid[:-1], fill=-1)

    support = resolve_sample_support(
        t,
        sample_valid,
        max_gap=max_gap,
        epochs=windows,
        break_before=breaks,
        result="no label has an epoch",
    )
    first, last, epoch_codes = label_runs(codes, support.interval_mask)

    # Labels in the order of their first epoch; every label in ``values``
    # starts an interval, so those without an epoch are unsupported.
    _, first_epoch = np.unique(epoch_codes, return_index=True)
    ordered = epoch_codes[np.sort(first_epoch)].tolist()
    unsupported = sorted(set(range(len(values))) - set(ordered))
    settings = {
        "labels": [values[c] for c in ordered],
        "unsupported_labels": [values[c] for c in unsupported],
        "max_gap": None if max_gap is None else float(max_gap),
        "epochs": None if windows is None else windows.tolist(),
        "missing_label": missing_label,
        "n_declared_breaks": 0 if breaks is None else int(breaks[1:].sum()),
    }
    return Intervals(
        [
            Interval(float(t[i]), float(t[j + 1]), (("label", values[code]),))
            for i, j, code in zip(
                first.tolist(), last.tolist(), epoch_codes.tolist(), strict=True
            )
        ],
        record_type=Interval,
        fields=("label",),
        analysis_settings=settings,
    )
