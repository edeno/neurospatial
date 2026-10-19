"""Align continuous behavioral signals to events.

Running speed, head direction or the occupied zone around each reward is the
behavioral counterpart of a peri-event time histogram. These signals are
sampled at fixed times relative to each event, only where they were observed:
nothing is interpolated across a tracking gap, a declared break or an epoch
edge, and nothing is extrapolated past the recording. Each feature is
summarized by its kind: linear features by their mean, angles by their
circular mean and resultant length, and categorical codes by counts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurospatial._exceptions import _format_error
from neurospatial._intervals import as_intervals
from neurospatial._observation import validated_signals
from neurospatial._results import ReadOnlyArraysMixin, ResultMixin, as_label_array
from neurospatial.events._continuous import (
    _EVENT_STATUSES,
    _VALUE_KINDS,
    _check_category_codes,
    _circular_means,
    _event_chunks,
    _overlapping,
    _relative_times,
    _resolve_features,
    _sample_events,
    _Summaries,
    _window_status,
)
from neurospatial.events.alignment import _validate_window

__all__ = ["ContinuousEventResult", "align_continuous_to_events"]

_WINDOW_COVERAGES = ("complete", "partial")
_MEAN_STATUSES = ("defined", "no_support", "undefined_direction", "not_numeric")


@dataclass(frozen=True, repr=False)
class ContinuousEventResult(ReadOnlyArraysMixin, ResultMixin):
    """Continuous signals sampled around events, from ``align_continuous_to_events``.

    Every input event has a status: ``"retained"`` or the reason it was
    excluded. Retained events are summarized per relative time and feature by
    sufficient statistics (``counts``, ``sums``, ``sin_sums``, ``cos_sums``,
    ``category_counts``), from which ``means``, ``resultant_lengths`` and
    ``mean_status`` are derived. The sampled values themselves are kept in
    ``values`` unless the alignment ran with ``retain_aligned=False``.

    Attributes
    ----------
    relative_times : NDArray[np.float64], shape (n_times,)
        Query offsets from each event, in seconds.
    feature_ids : NDArray, shape (n_features,)
        Feature labels.
    value_kinds : tuple of str, length n_features
        ``"linear"``, ``"circular"`` or ``"categorical"`` per feature.
    interpolation : tuple of str, length n_features
        Resolved interpolation per feature (``"linear"``, ``"circular"`` or
        ``"previous"``).
    feature_units : tuple of str or None, length n_features
        Unit per feature; ``"rad"`` for circular features.
    event_status : NDArray[np.str_], shape (n_input_events,)
        ``"retained"``, ``"outside_recording"``, ``"outside_epochs"``,
        ``"incomplete_support"`` or ``"no_support"`` per input event.
    overlapping : NDArray[np.bool_], shape (n_input_events,)
        True when a retained event's window shares time with another retained
        event's window (overlapping windows reuse samples). Duplicate event
        times overlap each other; excluded events are False. Bounds are
        compared in floating point, so windows meant only to touch can be
        flagged.
    counts : NDArray[np.int64], shape (n_times, n_features)
        Retained events with a supported value in each cell.
    sums : NDArray[np.float64], shape (n_times, n_features)
        Sum of supported values; NaN columns for non-linear features.
    sin_sums, cos_sums : NDArray[np.float64], shape (n_times, n_features)
        Sum of the sine and cosine of supported angles; NaN columns for
        non-circular features.
    category_features : NDArray[np.intp], shape (n_categories,)
        Feature column of each category row.
    category_codes : NDArray[np.int64], shape (n_categories,)
        Integer code of each category row, sorted within each feature.
    category_labels : NDArray[np.str_], shape (n_categories,)
        Label of each category row.
    category_counts : NDArray[np.int64], shape (n_times, n_categories)
        Retained events holding each category row's code, per relative time.
    window : tuple of float
        ``(start, stop)`` in seconds relative to each event, half-open.
    dt : float
        Query spacing in seconds.
    window_coverage : {"complete", "partial"}
        The coverage rule that decided which events were kept.
    resultant_threshold : float
        Circular means are undefined at resultant lengths at or below it.
    max_gap : float or None
        The gap limit that was applied, in seconds.
    epochs : NDArray[np.float64] or None, shape (n_epochs, 2)
        Normalized half-open ``[start, stop)`` windows that were applied.
    values : NDArray[np.float64] or None, shape (n_events, n_times, n_features)
        Sampled values of the retained events, in input order; NaN where
        unsupported. Categorical codes are stored as floats. None when the
        alignment ran with ``retain_aligned=False``.
    support : NDArray[np.bool_] or None, shape (n_events, n_times, n_features)
        True where ``values`` was observed. None with ``values``.
    event_ids : NDArray or None, shape (n_input_events,)
        Caller labels of the input events, when given.
    """

    relative_times: NDArray[np.float64]
    feature_ids: NDArray[Any]
    value_kinds: tuple[str, ...]
    interpolation: tuple[str, ...]
    feature_units: tuple[str | None, ...]
    event_status: NDArray[np.str_]
    overlapping: NDArray[np.bool_]
    counts: NDArray[np.int64]
    sums: NDArray[np.float64]
    sin_sums: NDArray[np.float64]
    cos_sums: NDArray[np.float64]
    category_features: NDArray[np.intp]
    category_codes: NDArray[np.int64]
    category_labels: NDArray[np.str_]
    category_counts: NDArray[np.int64]
    window: tuple[float, float]
    dt: float
    window_coverage: str
    resultant_threshold: float
    max_gap: float | None
    epochs: NDArray[np.float64] | None
    values: NDArray[np.float64] | None
    support: NDArray[np.bool_] | None
    event_ids: NDArray[Any] | None

    def __post_init__(self) -> None:
        """Check that the fields describe one alignment, then own the arrays."""
        problems = self._inconsistencies()
        if problems:
            raise ValueError(
                "ContinuousEventResult fields are inconsistent: "
                + "; ".join(problems)
                + "."
            )
        self._own_arrays()

    def _inconsistencies(self) -> list[str]:
        """List every way the fields disagree with each other."""
        problems: list[str] = []
        relative_times = np.asarray(self.relative_times)
        feature_ids = np.asarray(self.feature_ids)
        status = np.asarray(self.event_status)
        if relative_times.ndim != 1 or feature_ids.ndim != 1 or status.ndim != 1:
            return ["relative_times, feature_ids and event_status must be 1-D"]
        n_times, n_features, n_input = (
            relative_times.size,
            feature_ids.size,
            status.size,
        )

        for name in ("value_kinds", "interpolation", "feature_units"):
            if len(getattr(self, name)) != n_features:
                problems.append(f"{name} needs {n_features} entries")
        if not set(self.value_kinds) <= set(_VALUE_KINDS):
            problems.append(f"value_kinds must be in {list(_VALUE_KINDS)}")
        if not np.isin(status, _EVENT_STATUSES).all():
            problems.append(f"event_status must be in {list(_EVENT_STATUSES)}")
        if np.shape(self.overlapping) != (n_input,):
            problems.append(f"overlapping needs shape ({n_input},)")
        if self.event_ids is not None and np.shape(self.event_ids) != (n_input,):
            problems.append(f"event_ids needs shape ({n_input},)")

        cell = (n_times, n_features)
        for name in ("counts", "sums", "sin_sums", "cos_sums"):
            if np.shape(getattr(self, name)) != cell:
                problems.append(f"{name} needs shape {cell}")

        category_features = np.asarray(self.category_features)
        n_categories = category_features.size
        if category_features.ndim != 1 or any(
            np.shape(getattr(self, name)) != (n_categories,)
            for name in ("category_codes", "category_labels")
        ):
            problems.append(
                "category_features, category_codes and category_labels need one "
                "entry per category row"
            )
        elif np.shape(self.category_counts) != (n_times, n_categories):
            problems.append(f"category_counts needs shape ({n_times}, {n_categories})")
        elif n_categories and len(self.value_kinds) == n_features:
            in_range = (category_features >= 0) & (category_features < n_features)
            kinds = np.asarray(self.value_kinds)
            if not (
                in_range.all() and (kinds[category_features] == "categorical").all()
            ):
                problems.append("category_features must name categorical features")

        aligned = (int(np.count_nonzero(status == "retained")), *cell)
        if self.values is None or self.support is None:
            if (self.values is None) != (self.support is None):
                problems.append("values and support are both given or both None")
        elif np.shape(self.values) != aligned or np.shape(self.support) != aligned:
            problems.append(
                f"values and support need shape {aligned}, one row per retained event"
            )

        if self.window_coverage not in _WINDOW_COVERAGES:
            problems.append(f"window_coverage must be in {list(_WINDOW_COVERAGES)}")
        threshold = self.resultant_threshold
        if not (np.isfinite(threshold) and 0.0 <= threshold <= 1.0):
            problems.append("resultant_threshold must be finite and in [0, 1]")
        return problems

    @property
    def event_indices(self) -> NDArray[np.intp]:
        """Input positions of the retained events, shape ``(n_events,)``.

        Row ``i`` of ``values`` belongs to input event ``event_indices[i]``;
        use ``event_times[result.event_indices]`` for the matching cohort.
        """
        return np.flatnonzero(self.event_status == "retained")

    @property
    def n_events(self) -> int:
        """Number of retained events."""
        return int(np.count_nonzero(self.event_status == "retained"))

    @property
    def n_events_excluded(self) -> int:
        """Number of input events that were excluded (see ``event_status``)."""
        return int(self.event_status.size - self.n_events)

    @property
    def means(self) -> NDArray[np.float64]:
        """Mean value per relative time and feature, shape ``(n_times, n_features)``.

        Linear features: ``sums / counts``. Circular features: the circular
        mean ``atan2(sin_sums, cos_sums)`` in ``(-pi, pi]``, NaN where the
        resultant length is at or below ``resultant_threshold`` (the angles
        cancel and have no mean direction). NaN for categorical features and
        for cells without support; ``mean_status`` says which.
        """
        means, _, _ = _circular_means(
            self.sin_sums, self.cos_sums, self.counts, self.resultant_threshold
        )
        linear = np.asarray(self.value_kinds) == "linear"
        linear_means = np.full(self.counts.shape, np.nan)
        np.divide(
            self.sums,
            self.counts,
            out=linear_means,
            where=(self.counts > 0) & linear,
        )
        means[:, linear] = linear_means[:, linear]
        return means

    @property
    def resultant_lengths(self) -> NDArray[np.float64]:
        """Mean resultant length, shape ``(n_times, n_features)``.

        ``hypot(sin_sums, cos_sums) / counts`` in ``[0, 1]`` for circular
        features: 1 when every event has the same angle, near 0 when the
        angles cancel. NaN for non-circular features and cells without
        support.
        """
        _, lengths, _ = _circular_means(
            self.sin_sums, self.cos_sums, self.counts, self.resultant_threshold
        )
        return lengths

    @property
    def mean_status(self) -> NDArray[np.str_]:
        """Why each entry of ``means`` is or is not defined, shape ``(n_times, n_features)``.

        ``"defined"``; ``"no_support"`` (no retained event was observed in
        the cell); ``"undefined_direction"`` (circular, resultant length at or
        below ``resultant_threshold``); ``"not_numeric"`` (categorical: see
        ``category_counts``).
        """
        _, _, defined = _circular_means(
            self.sin_sums, self.cos_sums, self.counts, self.resultant_threshold
        )
        kinds = np.asarray(self.value_kinds)
        width = max(len(name) for name in _MEAN_STATUSES)
        status = np.full(self.counts.shape, "defined", dtype=f"<U{width}")
        undefined = (self.counts > 0) & ~defined & (kinds == "circular")
        status[undefined] = "undefined_direction"
        status[self.counts == 0] = "no_support"
        status[:, kinds == "categorical"] = "not_numeric"
        return status


def align_continuous_to_events(
    times: ArrayLike,
    signals: ArrayLike,
    event_times: ArrayLike,
    window: tuple[float, float],
    *,
    dt: float,
    feature_ids: Sequence[Any] | NDArray[Any] | None = None,
    feature_units: Mapping[Any, str | None] | None = None,
    event_ids: Sequence[Any] | NDArray[Any] | None = None,
    value_kinds: Mapping[Any, str] | None = None,
    category_labels: Mapping[Any, Mapping[int, str]] | None = None,
    interpolation: Mapping[Any, str] | None = None,
    resultant_threshold: float = 1e-12,
    max_gap: float | None = 0.5,
    epochs: Any = None,
    valid: ArrayLike | None = None,
    break_before: ArrayLike | None = None,
    window_coverage: str = "complete",
    retain_aligned: bool = True,
) -> ContinuousEventResult:
    """
    Sample continuous signals at fixed times around each event.

    For each event, every signal column (feature) is sampled at
    ``event + relative_times``, with ``relative_times = start + dt * j`` for
    every ``j >= 0`` before ``stop``. A value is taken only where the feature
    was observed: at a valid sample's exact time, or inside a usable interval
    between two valid samples. Nothing is interpolated across a gap longer
    than ``max_gap``, an invalid sample, a declared break or an ``epochs``
    edge, and nothing is extrapolated before the first or after the last
    sample. This samples a signal; it does not bin or average it.

    With the default ``window_coverage="complete"``, an event is kept only
    when its whole window ``[event + start, event + stop)`` lies inside one
    continuous run of observation for every feature, so every event
    contributes to every cell. ``window_coverage="partial"`` keeps every event
    with at least one supported cell, and each cell is summarized over the
    events observed there (``counts``).

    Parameters
    ----------
    times : ArrayLike, shape (n_samples,)
        Sample times in seconds; finite and strictly increasing.
    signals : ArrayLike, shape (n_samples, n_features) or (n_samples,)
        One column per feature, on the clock of ``times``. NaN or infinity
        marks a missing sample of that feature; a finite zero is an
        observation.
    event_times : ArrayLike, shape (n_input_events,)
        Event times in seconds, on the same clock. Input order is kept and
        duplicate times are separate events.
    window : tuple of float
        ``(start, stop)`` in seconds relative to each event, half-open,
        ``start < stop``. For example ``(-1.0, 2.0)``.
    dt : float
        Spacing of the query times in seconds. No default: choose the time
        resolution of your analysis, usually no finer than the sampling
        interval.
    feature_ids : sequence, shape (n_features,), optional
        Unique feature labels; default ``0 .. n_features - 1``. The mappings
        below are keyed by these ids.
    feature_units : mapping of feature id to str or None, optional
        Unit of each feature, recorded on the result. Circular features are
        radians: their unit must be ``"rad"`` (the default) or omitted.
    event_ids : sequence, shape (n_input_events,), optional
        Your labels for the events (trial numbers, reward ids), kept for every
        input event.
    value_kinds : mapping of feature id to str, optional
        ``"linear"`` (default), ``"circular"`` (angles in radians, such as
        head direction) or ``"categorical"`` (integer codes, such as the
        occupied zone). Units never imply a kind.
    category_labels : mapping of feature id to mapping of int to str, optional
        Required for every categorical feature: ``{code: label}``, for example
        ``{"zone": {0: "center", 1: "arm"}}``. Code 0 is valid. Every finite
        code in the signal must be listed.
    interpolation : mapping of feature id to str, optional
        How to sample between two samples. Linear features: ``"linear"``
        (default) or ``"previous"``. Circular: ``"circular"`` (default,
        shorter arc) or ``"previous"``. Categorical: ``"previous"`` only (the
        last observed code).
    resultant_threshold : float, default=1e-12
        Circular means are undefined (NaN, ``"undefined_direction"``) where
        the mean resultant length is at or below this value, in ``[0, 1]``.
    max_gap : float or None, default=0.5
        Longest interval, in seconds, that counts as continuous observation.
        ``None`` keeps every interval.
    epochs : None, (start, stop) or array-like of shape (n_epochs, 2), optional
        Half-open ``[start, stop)`` windows in seconds, on the clock of
        ``times``. An interval must lie inside one window to be used.
    valid : ArrayLike of bool, shape (n_samples,) or (n_samples, n_features), optional
        Extra sample validity, for all features (1-D) or per feature (2-D),
        for example False where tracking confidence is low.
    break_before : ArrayLike of bool, shape (n_samples,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1``
        and ``k`` (for example where a new recording starts). Entry 0 is
        ignored.
    window_coverage : {"complete", "partial"}, default="complete"
        Which events to keep; see above.
    retain_aligned : bool, default=True
        Keep the sampled ``values`` and ``support`` of every retained event.
        ``False`` keeps only the summaries, computed in bounded-memory chunks,
        for many events or long windows; the summaries are the same.

    Returns
    -------
    ContinuousEventResult
        Per-event status and overlap flags, per-cell counts and summaries
        (``means``, ``resultant_lengths``, ``category_counts``) and, unless
        ``retain_aligned=False``, the sampled values.

    Raises
    ------
    ValueError
        If ``window`` is not ``start < stop`` in seconds or holds no query
        time, ``dt`` is not a positive finite number, ``window_coverage``,
        ``resultant_threshold`` or ``retain_aligned`` is invalid, the signals,
        feature metadata, ``event_times`` or ``event_ids`` are malformed, or
        every event is excluded.

    Notes
    -----
    Exclusion reasons, one per excluded event in ``event_status``:

    - ``"outside_recording"``: complete coverage, the window is not inside
      ``[times[0], times[-1]]``; partial coverage, the window does not
      overlap it at all.
    - ``"outside_epochs"``: complete coverage, ``epochs`` given and the
      window is not inside one epoch.
    - ``"incomplete_support"``: complete coverage, some feature has no single
      run of observation covering the window.
    - ``"no_support"``: partial coverage, no feature was observed at any
      query time of the window.

    Overlapping windows reuse the same samples; ``overlapping`` flags those
    events so you can decide whether to keep them.

    See Also
    --------
    peri_event_histogram : Spike counts around events.

    Examples
    --------
    Running speed that grows as ``2 * t`` cm/s, sampled at 10 Hz, around two
    rewards. Each mean is the average of the two events' speeds at that
    relative time.

    >>> import numpy as np
    >>> from neurospatial.events import align_continuous_to_events
    >>> times = np.arange(100) / 10
    >>> speed = 2 * times
    >>> result = align_continuous_to_events(
    ...     times, speed, [2.0, 5.0], (-1.0, 1.0), dt=0.5, feature_ids=["speed"]
    ... )
    >>> result.relative_times
    array([-1. , -0.5,  0. ,  0.5])
    >>> result.means[:, 0]
    array([5., 6., 7., 8.])
    >>> result.event_status.tolist()
    ['retained', 'retained']

    Head direction near +pi around one reward and near -pi around the other
    points the same way. The circular mean is pi with a resultant length near
    1, where an arithmetic mean would give 0.

    >>> heading = np.where(times < 3.5, 3.1, -3.1)
    >>> result = align_continuous_to_events(
    ...     times,
    ...     np.column_stack([speed, heading]),
    ...     [2.0, 5.0],
    ...     (-1.0, 1.0),
    ...     dt=0.5,
    ...     feature_ids=["speed", "heading"],
    ...     value_kinds={"heading": "circular"},
    ... )
    >>> np.round(result.means[:, 1], 4)
    array([3.1416, 3.1416, 3.1416, 3.1416])
    >>> np.round(result.resultant_lengths[:, 1], 3)
    array([0.999, 0.999, 0.999, 0.999])
    >>> result.feature_units
    (None, 'rad')
    """
    call = "align_continuous_to_events"
    _validate_window(window, context=call)
    start, stop = float(window[0]), float(window[1])
    if not start < stop:
        raise ValueError(
            _format_error(
                f"{call}: window start ({start}) must be less than stop ({stop}).",
                why="The window is the half-open interval [start, stop) of times "
                "relative to each event, so it needs a positive length.",
                fix="pass window=(-1.0, 2.0), in seconds",
            )
        )
    dt = _positive_number(dt, "dt", call=call)
    relative_times = _relative_times((start, stop), dt)
    if relative_times.size == 0:
        raise ValueError(
            _format_error(
                f"{call}: window={window!r} with dt={dt} holds no query time.",
                why=f"The window is {stop - start:g} s long, far shorter than dt.",
                fix="widen window or use a smaller dt",
            )
        )
    if window_coverage not in _WINDOW_COVERAGES:
        raise ValueError(
            _format_error(
                f"{call}: window_coverage={window_coverage!r} is not supported.",
                why="'complete' keeps events whose whole window was observed; "
                "'partial' keeps every event observed at some query time.",
                fix="pass window_coverage='complete' or window_coverage='partial'",
            )
        )
    try:
        threshold_ok = bool(
            np.isfinite(resultant_threshold) and 0.0 <= resultant_threshold <= 1.0
        )
    except TypeError:
        threshold_ok = False
    if not threshold_ok:
        raise ValueError(
            _format_error(
                f"{call}: resultant_threshold={resultant_threshold!r} is not in "
                "[0, 1].",
                why="It is compared with mean resultant lengths, which lie in [0, 1].",
                fix="keep the default 1e-12, or pass a value such as 0.1",
            )
        )
    if not isinstance(retain_aligned, bool | np.bool_):
        raise ValueError(
            _format_error(
                f"{call}: retain_aligned={retain_aligned!r} is not a bool.",
                fix="pass retain_aligned=True to keep the sampled values, or "
                "False to keep only the summaries",
            )
        )

    normalized_epochs = as_intervals(epochs, name="epochs")
    t, sig, supports = validated_signals(
        times,
        signals,
        valid=valid,
        call=call,
        max_gap=max_gap,
        epochs=normalized_epochs,
        break_before=break_before,
        result="no event window can be sampled",
    )
    n_features = sig.shape[1]
    spec = _resolve_features(
        n_features,
        feature_ids=feature_ids,
        feature_units=feature_units,
        value_kinds=value_kinds,
        category_labels=category_labels,
        interpolation=interpolation,
        call=call,
    )
    sample_valid = np.column_stack([support.sample_valid for support in supports])
    _check_category_codes(sig, sample_valid, spec, call=call)
    events = _validated_event_times(event_times, call=call)
    ids = _validated_event_ids(event_ids, events.size, call=call)

    status = _window_status(
        events, (start, stop), t, supports, normalized_epochs, window_coverage
    )
    provisional = np.flatnonzero(status == "retained")
    summaries = _Summaries.empty(relative_times.size, spec)
    kept_values: list[NDArray[np.float64]] = []
    kept_support: list[NDArray[np.bool_]] = []
    for chunk in _event_chunks(provisional.size, relative_times.size * n_features):
        rows = provisional[chunk]
        values, support = _sample_events(
            events[rows], relative_times, t, sig, supports, spec, normalized_epochs
        )
        observed = support.any(axis=(1, 2))
        status[rows[~observed]] = "no_support"
        summaries.add(values, support)
        if retain_aligned:
            kept_values.append(values[observed])
            kept_support.append(support[observed])

    retained = status == "retained"
    if not retained.any():
        reasons = ", ".join(
            f"{np.count_nonzero(status == reason)} {reason}"
            for reason in _EVENT_STATUSES[1:]
            if np.any(status == reason)
        )
        raise ValueError(
            _format_error(
                f"{call}: All {events.size} events were excluded: {reasons}.",
                why="No event window was observed well enough to be sampled.",
                fix="check that event_times and times share a clock and units, "
                "that epochs and max_gap keep the event windows, use "
                "window_coverage='partial' to keep partly observed events, or "
                "narrow window",
            )
        )

    return ContinuousEventResult(
        relative_times=relative_times,
        feature_ids=spec.feature_ids,
        value_kinds=spec.value_kinds,
        interpolation=spec.interpolation,
        feature_units=spec.units,
        event_status=status,
        overlapping=_overlapping(events, (start, stop), retained),
        counts=summaries.counts,
        sums=summaries.sums,
        sin_sums=summaries.sin_sums,
        cos_sums=summaries.cos_sums,
        category_features=spec.category_features,
        category_codes=spec.category_codes,
        category_labels=spec.category_labels,
        category_counts=summaries.category_counts,
        window=(start, stop),
        dt=dt,
        window_coverage=window_coverage,
        resultant_threshold=float(resultant_threshold),
        max_gap=None if max_gap is None else float(max_gap),
        epochs=normalized_epochs,
        values=np.concatenate(kept_values) if retain_aligned else None,
        support=np.concatenate(kept_support) if retain_aligned else None,
        event_ids=ids,
    )


def _positive_number(value: Any, name: str, *, call: str) -> float:
    """Return ``value`` as a float, or raise unless it is positive and finite."""
    try:
        ok = bool(np.isfinite(value) and value > 0)
    except TypeError:  # None, strings and other non-numbers
        ok = False
    if not ok:
        raise ValueError(
            _format_error(
                f"{call}: {name} must be a positive finite number of seconds; "
                f"got {value!r}.",
                why=f"{name} is the spacing of the query times around each event.",
                fix=f"pass {name}=0.02 for 20 ms steps",
            )
        )
    return float(value)


def _validated_event_times(event_times: ArrayLike, *, call: str) -> NDArray[np.float64]:
    """Return event times as a 1-D finite float array, or raise."""
    fix = "pass a 1-D array of event times in seconds, e.g. reward_times"
    try:
        events = np.asarray(event_times, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError(
            _format_error(f"{call}: event_times must be numbers in seconds.", fix=fix)
        ) from None
    if events.ndim != 1 or events.size == 0:
        raise ValueError(
            _format_error(
                f"{call}: event_times must be a non-empty 1-D array; got shape "
                f"{events.shape}.",
                fix=fix,
            )
        )
    bad = np.flatnonzero(~np.isfinite(events))
    if bad.size:
        raise ValueError(
            _format_error(
                f"{call}: event_times has {bad.size} non-finite value(s), first at "
                f"index {bad[0]} ({events[bad[0]]}).",
                why="An event without a time has no window to sample.",
                fix="drop those events, e.g. event_times[np.isfinite(event_times)]",
            )
        )
    return events


def _validated_event_ids(
    event_ids: Sequence[Any] | NDArray[Any] | None, n_events: int, *, call: str
) -> NDArray[Any] | None:
    """Return the event labels as an array of length ``n_events``, or None."""
    if event_ids is None:
        return None
    ids = as_label_array(event_ids)
    if ids.shape != (n_events,):
        raise ValueError(
            _format_error(
                f"{call}: got event_ids of shape {ids.shape} for {n_events} "
                "event_times.",
                why="event_ids label every input event, retained or not.",
                fix="pass one label per entry of event_times",
            )
        )
    return ids
