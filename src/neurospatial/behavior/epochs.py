"""Array-native epoch selection (``restrict`` / ``in_epochs``).

"Give me my running periods" / "give me trial N" -- restrict time-series and
spike data to a set of time intervals (epochs). Epochs are plain
``(start, end)`` arrays, so this stays array-first and never imports pynapple;
a pynapple ``IntervalSet`` is nonetheless accepted **transparently** because it
is *duck-typed* (it exposes ``.start`` / ``.end`` arrays), not ``isinstance``-d.

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
    train is restricted by its **own** timestamps. Accepts a plain sequence of
    trains or a :class:`~neurospatial.encoding.SpikeTrains` container.
labels_to_epochs
    Turn one label per sample (task phase, running direction, trial type) into
    half-open ``[start, stop)`` epochs per label, ready to pass as ``epochs=``
    to an analysis. A missing sample is bridged like a deleted row when the
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
from collections.abc import Hashable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error
from neurospatial._intervals import (
    DEFAULT_MAX_GAP,
    as_intervals,
    hold_last_valid,
    intervals_contain,
    validate_max_gap,
)
from neurospatial._observation import (
    _validated_signal_times,
    label_codes,
    label_runs,
    resolve_sample_support,
    validate_break_before,
)
from neurospatial._results import (
    ReadOnlyArraysMixin,
    ResultMixin,
    as_label_array,
    read_only_copy,
)

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "LabelEpochsResult",
    "in_epochs",
    "labels_to_epochs",
    "restrict",
    "restrict_spike_trains",
]

_Closed = Literal["both", "left", "right", "neither"]


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
    ``closed`` is validated here too, so this stays safe to call directly (e.g.
    per train from :func:`restrict_spike_trains`); :func:`in_epochs` validates it
    a second time, up front, only to report a bad ``closed`` *before* a malformed
    ``epochs`` (this helper runs after ``as_intervals``).
    """
    if closed not in ("both", "left", "right", "neither"):
        raise ValueError(
            f"closed must be one of 'both', 'left', 'right', 'neither', got {closed!r}."
        )

    t_arr = np.asarray(t, dtype=np.float64)

    if intervals is None:
        return np.ones(t_arr.shape, dtype=np.bool_)
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
        ``"both"`` matches pynapple's inclusive ``restrict``. Touching rows
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
        DataFrame that fall in an intervals DataFrame.

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
    if closed not in ("both", "left", "right", "neither"):
        raise ValueError(
            f"closed must be one of 'both', 'left', 'right', 'neither', got {closed!r}."
        )

    intervals = as_intervals(epochs, name="epochs")
    return _mask_in_intervals(t, intervals, closed=closed)


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
        analysis ``epochs=`` argument. ``"both"`` matches pynapple's
        inclusive ``restrict``.

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
        DataFrame that fall in an intervals DataFrame.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.behavior import restrict
    >>> times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    >>> positions = np.column_stack([times, times])
    >>> t, pos = restrict(times, positions, epochs=(1.0, 3.0))
    >>> t
    array([1., 2.])
    """
    times_arr = np.asarray(times)
    n = len(times_arr)

    coerced: list[NDArray[Any]] = []
    for i, arr in enumerate(arrays):
        arr_np = np.asarray(arr)
        if len(arr_np) != n:
            raise ValueError(
                f"array at position {i} has len {len(arr_np)} but must be "
                f"aligned to times (len {n}); every *arrays entry must have the "
                "same first-axis length as times."
            )
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
                stacklevel=2,
            )
    return (times_kept, *(arr[mask] for arr in coerced))


def restrict_spike_trains(
    spike_times: Sequence[NDArray[np.float64]],
    epochs: Any,
    *,
    closed: _Closed = "left",
) -> list[NDArray[np.float64]]:
    """Restrict each ragged per-unit spike train by its **own** timestamps.

    Ragged per-unit spike times are not aligned to a common ``times`` axis, so
    each train is masked against itself: ``[t[in_epochs(t, epochs)] for t in
    spike_times]``.

    Parameters
    ----------
    spike_times : sequence of ndarray, or SpikeTrains
        Per-unit 1-D spike-time arrays. A
        :class:`~neurospatial.encoding.SpikeTrains` container is accepted --
        iterating it yields the per-unit train arrays.
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
        analysis ``epochs=`` argument. ``"both"`` matches pynapple's
        inclusive ``restrict``.

    Returns
    -------
    list of ndarray
        One masked train per input train, in the same order. Always a plain
        ``list`` (type-stable), even when given a
        :class:`~neurospatial.encoding.SpikeTrains` -- ``SpikeTrains`` identity
        can be preserved with explicit array selection and retained unit labels.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.behavior import restrict_spike_trains
    >>> spike_times = [np.array([0.1, 1.5, 2.9]), np.array([0.5, 3.0, 6.0])]
    >>> restrict_spike_trains(spike_times, (1.0, 3.5))
    [array([1.5, 2.9]), array([3.])]
    """
    # Normalize the intervals ONCE (not per train): a malformed epochs raises up
    # front, and each train is then masked against the same pre-normalized
    # ``(n, 2)`` array via ``_mask_in_intervals`` -- avoiding the per-unit
    # re-normalization that ``in_epochs`` would repeat on every call.
    intervals = as_intervals(epochs, name="epochs")
    out: list[NDArray[np.float64]] = []
    for train in spike_times:
        t = np.asarray(train, dtype=np.float64)
        out.append(t[_mask_in_intervals(t, intervals, closed=closed)])
    return out


@dataclass(frozen=True, eq=False, repr=False)
class LabelEpochsResult(
    ReadOnlyArraysMixin, ResultMixin, Mapping[Hashable, NDArray[np.float64]]
):
    """Half-open epochs per label, returned by :func:`labels_to_epochs`.

    A read-only mapping from each label to its ``(n_epochs, 2)`` array of
    ``[start, stop)`` windows in seconds. Pass ``result[label]`` as ``epochs=``
    to an analysis to restrict it to the time spent in that condition.

    Attributes
    ----------
    labels : tuple of hashable
        The mapping's keys, in the order of their first retained interval.
    epoch_bounds : NDArray[np.float64], shape (n_epochs, 2)
        ``[start, stop)`` of every epoch in seconds, in time order.
    epoch_labels : NDArray[np.intp], shape (n_epochs,)
        Index into ``labels`` of each epoch.
    epoch_intervals : NDArray[np.intp], shape (n_epochs, 2)
        Inclusive first and last interval ``k`` of each epoch; interval ``k``
        is ``[times[k], times[k + 1])``.
    interval_labels : NDArray[np.intp], shape (max(n_samples - 1, 0),)
        Index into ``labels`` of each interval, or -1 when the interval was
        not retained (unsupported or missing label).
    interval_supported : NDArray[np.bool_], shape (max(n_samples - 1, 0),)
        True where the gap between the valid samples around the interval
        passes ``max_gap``, the interval lies inside one ``epochs`` row, and
        no break is declared. Intervals before the first or after the last
        valid sample are False.
    unsupported_labels : tuple of hashable
        Labels that start an interval but have no supported interval, in
        first-seen order. They are not keys.
    n_samples : int
        Number of input samples.
    max_gap : float or None
        The gap limit that was applied, in seconds.
    input_epochs : NDArray[np.float64] or None, shape (n_windows, 2)
        The normalized ``epochs=`` windows that were applied, or None.
    missing_label : hashable or None
        The label value treated as missing, besides ``None`` and ``NaN``.
    n_declared_breaks : int
        Number of breaks declared by ``break_before`` (entry 0 is not a break).

    Notes
    -----
    Each label is one condition; the epochs of different labels never overlap.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.behavior import labels_to_epochs
    >>> times = np.arange(5) * 0.25
    >>> conditions = labels_to_epochs(times, ["a", "a", "b", "b", "b"])
    >>> list(conditions)
    ['a', 'b']
    >>> conditions["b"]
    array([[0.5, 1. ]])
    """

    labels: tuple[Hashable, ...]
    epoch_bounds: NDArray[np.float64]
    epoch_labels: NDArray[np.intp]
    epoch_intervals: NDArray[np.intp]
    interval_labels: NDArray[np.intp]
    interval_supported: NDArray[np.bool_]
    unsupported_labels: tuple[Hashable, ...]
    n_samples: int
    max_gap: float | None
    input_epochs: NDArray[np.float64] | None
    missing_label: Hashable | None
    n_declared_breaks: int

    def __post_init__(self) -> None:
        """Own read-only copies of the arrays."""
        self._own_arrays()

    def __getitem__(self, label: Hashable) -> NDArray[np.float64]:
        """Return the ``(n_epochs, 2)`` ``[start, stop)`` windows of ``label``.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.behavior import labels_to_epochs
        >>> times = np.arange(4) * 0.25
        >>> labels_to_epochs(times, ["a", "a", "b", "b"])["a"]
        array([[0. , 0.5]])
        """
        if label not in self.labels:
            if label in self.unsupported_labels:
                why = (
                    f"label {label!r} occurs but none of its intervals is "
                    "supported: each lies in a gap between observed samples "
                    "longer than max_gap, crosses a declared break, lies "
                    "outside epochs, or comes before the first or after the "
                    "last observed sample."
                )
            else:
                why = f"label {label!r} starts no interval."
            raise KeyError(
                _format_error(
                    f"No epochs for label {label!r}.",
                    why=why,
                    fix=f"use one of the labels with epochs: {list(self.labels)}.",
                )
            )
        rows = self.epoch_bounds[self.epoch_labels == self.labels.index(label)]
        return read_only_copy(rows)

    def __iter__(self) -> Iterator[Hashable]:
        """Iterate over labels in the order of their first retained interval.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.behavior import labels_to_epochs
        >>> times = np.arange(5) * 0.25
        >>> conditions = labels_to_epochs(times, ["b", "b", "a", "a", "b"])
        >>> for label in conditions:
        ...     print(label, conditions[label].tolist())
        b [[0.0, 0.5]]
        a [[0.5, 1.0]]
        """
        return iter(self.labels)

    def __len__(self) -> int:
        """Number of labels with at least one epoch.

        A label whose intervals were all left out is not counted; it is in
        ``unsupported_labels``.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.behavior import labels_to_epochs
        >>> times = np.arange(5) * 0.25
        >>> valid = np.array([True, True, True, False, True])
        >>> labels = ["a", "a", "b", "c", "c"]
        >>> conditions = labels_to_epochs(times, labels, valid=valid, max_gap=0.3)
        >>> len(conditions)
        1
        >>> conditions.unsupported_labels
        ('b', 'c')
        """
        return len(self.labels)

    def __eq__(self, other: object) -> bool:
        """Results are equal only when they are the same object."""
        return self is other

    @property
    def durations(self) -> dict[Hashable, float]:
        """Total epoch duration of each label, in seconds.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.behavior import labels_to_epochs
        >>> times = np.arange(4) * 0.25
        >>> labels_to_epochs(times, ["a", "a", "b", "b"]).durations
        {'a': 0.5, 'b': 0.25}
        """
        totals = np.bincount(
            self.epoch_labels,
            weights=self.epoch_bounds[:, 1] - self.epoch_bounds[:, 0],
            minlength=len(self.labels),
        )
        return dict(zip(self.labels, totals.tolist(), strict=True))

    def epochs_table(self) -> pd.DataFrame:
        """One row per epoch, in time order.

        Returns
        -------
        pandas.DataFrame
            Columns ``label``, ``start``, ``stop``, ``duration`` (seconds),
            ``first_interval``, ``last_interval`` and ``n_intervals``.
            ``df.attrs`` holds ``max_gap``, ``input_epochs`` (list of rows or None),
            ``missing_label``, ``n_declared_breaks`` and
            ``allocation="interval_start"``.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.behavior import labels_to_epochs
        >>> times = np.arange(4) * 0.25
        >>> table = labels_to_epochs(times, ["a", "a", "b", "b"]).epochs_table()
        >>> table[["label", "start", "stop", "n_intervals"]]
          label  start  stop  n_intervals
        0     a    0.0  0.50            2
        1     b    0.5  0.75            1

        The ``epochs=`` windows that were applied are recorded as
        ``input_epochs``:

        >>> result = labels_to_epochs(times, ["a", "a", "b", "b"], epochs=[(0.0, 0.5)])
        >>> result.input_epochs
        array([[0. , 0.5]])
        >>> result.epochs_table().attrs["input_epochs"]
        [[0.0, 0.5]]
        """
        import pandas as pd

        start = self.epoch_bounds[:, 0].copy()
        stop = self.epoch_bounds[:, 1].copy()
        first = self.epoch_intervals[:, 0].copy()
        last = self.epoch_intervals[:, 1].copy()
        df = pd.DataFrame(
            {
                "label": [self.labels[i] for i in self.epoch_labels.tolist()],
                "start": start,
                "stop": stop,
                "duration": stop - start,
                "first_interval": first,
                "last_interval": last,
                "n_intervals": last - first + 1,
            }
        )
        df.attrs.update(
            max_gap=self.max_gap,
            input_epochs=(
                None if self.input_epochs is None else self.input_epochs.tolist()
            ),
            missing_label=self.missing_label,
            n_declared_breaks=self.n_declared_breaks,
            allocation="interval_start",
        )
        return df

    def summary(self) -> dict[str, Any]:
        """Flat dict of scalar headline metrics.

        ``n_unsupported_intervals`` counts intervals the support rules
        excluded; ``n_missing_label_intervals`` counts supported intervals
        whose label is missing. ``labeled_duration`` is in seconds.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.behavior import labels_to_epochs
        >>> times = np.arange(4) * 0.25
        >>> summary = labels_to_epochs(times, ["a", None, "b", "b"]).summary()
        >>> summary["n_missing_label_intervals"], summary["labeled_duration"]
        (1, 0.5)
        """
        bounds = self.epoch_bounds
        return {
            "n_labels": len(self.labels),
            "n_epochs": int(bounds.shape[0]),
            "labeled_duration": float((bounds[:, 1] - bounds[:, 0]).sum()),
            "n_unsupported_intervals": int((~self.interval_supported).sum()),
            "n_missing_label_intervals": int(
                (self.interval_supported & (self.interval_labels == -1)).sum()
            ),
            "n_unsupported_labels": len(self.unsupported_labels),
            "max_gap": self.max_gap,
        }


def _validated_sample_flags(valid: Any, n_samples: int) -> NDArray[np.bool_]:
    """Return ``valid`` as a Boolean ``(n_samples,)`` array; None means all valid."""
    if valid is None:
        return np.ones(n_samples, dtype=bool)
    flags = np.asarray(valid)
    if flags.dtype != np.bool_ or flags.shape != (n_samples,):
        raise ValueError(
            _format_error(
                "labels_to_epochs() needs valid to be a Boolean array of shape "
                f"({n_samples},); got dtype {flags.dtype} and shape {flags.shape}.",
                why="valid marks the samples that were observed. An unobserved "
                "sample is skipped as if its row were deleted: the gap around it "
                "is bridged when it is no longer than max_gap.",
                fix="pass a Boolean array with one entry per sample, e.g. "
                "valid = np.isfinite(positions).all(axis=1)",
            )
        )
    return flags


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


def labels_to_epochs(
    times: Any,
    labels: Any,
    *,
    valid: Any = None,
    break_before: Any = None,
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    missing_label: Hashable | None = None,
) -> LabelEpochsResult:
    """Convert one label per sample into half-open epochs per label.

    Use this to split a session into behavioral conditions (outbound and
    inbound runs, task phases, cue or trial types) and then estimate a rate
    map for each condition by passing ``conditions[label]`` as ``epochs=``
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
        mark a sample without a label. ``0``, ``-1`` and ``""`` are ordinary
        labels. A tuple such as ``("arm1", "out")`` is one label, so a combined
        condition needs no string joining. Result keys are plain Python values.
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
    LabelEpochsResult
        A mapping from each label to its ``(n_epochs, 2)`` ``[start, stop)``
        windows in seconds, with per-interval provenance.

    Raises
    ------
    ValueError
        If ``times`` is empty, not finite or not strictly increasing; if
        ``labels`` is not 1-D with one entry per sample; or if ``valid``,
        ``break_before`` or ``max_gap`` is malformed.

    See Also
    --------
    neurospatial.behavior.laps_to_direction_labels : Per-sample labels from laps.
    neurospatial.behavior.running_direction_labels : Per-sample running direction.

    Notes
    -----
    Time is allocated to the label at the start of each interval, so the
    final sample adds no time and a label held by a single sample before a
    change contributes one interval. The same label after a gap longer than
    ``max_gap`` (counting unobserved samples as missing rows), a declared
    break, an excluded window or a different label starts a new epoch; no
    duration is inferred across any of them. A label that starts intervals but has none kept is listed in
    ``unsupported_labels`` rather than returned with an empty window set.

    Labels here are disjoint by construction: each interval has one label.
    Conditions passed to an analysis as separate ``epochs=`` sets may overlap
    elsewhere. Labels assigned after the fact, for example from the path the
    animal later took, describe the trajectory; they are not causal
    predictors of the animal's choice.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.behavior import labels_to_epochs
    >>> times = np.arange(10) / 10
    >>> labels = np.array(["out"] * 5 + ["in"] * 5)
    >>> valid = np.ones(10, dtype=bool)
    >>> valid[7] = False  # tracking lost at 0.7 s for one frame: bridged
    >>> conditions = labels_to_epochs(times, labels, valid=valid)
    >>> list(conditions)
    ['out', 'in']
    >>> conditions["in"]
    array([[0.5, 0.9]])
    >>> labels_to_epochs(times, labels, valid=valid, max_gap=0.15)["in"]
    array([[0.5, 0.6],
           [0.8, 0.9]])

    Tuples are single labels, for combined conditions:

    >>> labels = [("arm1", "out")] * 3 + [("arm1", "in")] * 2
    >>> list(labels_to_epochs(np.arange(5.0) / 10, labels))
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
    sample_valid = _validated_sample_flags(valid, n_samples)
    validate_max_gap(max_gap)
    breaks = validate_break_before(break_before, n_samples, call=call)
    windows = as_intervals(epochs, name="epochs")

    sample_codes, values = label_codes(label_array[:-1], missing_label)
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

    # Re-index codes so labels follow their first retained interval.
    _, first_epoch = np.unique(epoch_codes, return_index=True)
    ordered = epoch_codes[np.sort(first_epoch)]
    # One extra slot so that remap[-1] (a missing code) stays -1.
    remap = np.full(len(values) + 1, -1, dtype=np.intp)
    remap[ordered] = np.arange(ordered.size, dtype=np.intp)
    retained = support.interval_mask & (codes >= 0)
    interval_labels = np.where(retained, remap[codes], -1).astype(np.intp)
    occurs = np.zeros(len(values), dtype=bool)
    occurs[sample_codes[sample_codes >= 0]] = True
    unsupported = np.flatnonzero(occurs & (remap[:-1] < 0))

    return LabelEpochsResult(
        labels=tuple(values[c] for c in ordered.tolist()),
        epoch_bounds=np.column_stack([t[first], t[last + 1]]),
        epoch_labels=remap[epoch_codes],
        epoch_intervals=np.column_stack([first, last]).astype(np.intp),
        interval_labels=interval_labels,
        interval_supported=support.interval_mask,
        unsupported_labels=tuple(values[c] for c in unsupported.tolist()),
        n_samples=n_samples,
        max_gap=max_gap,
        input_epochs=windows,
        missing_label=missing_label,
        n_declared_breaks=0 if breaks is None else int(breaks[1:].sum()),
    )
