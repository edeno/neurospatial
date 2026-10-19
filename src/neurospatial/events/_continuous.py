"""Private helpers for continuous event alignment.

Per-feature metadata (value kind, interpolation, units, category labels) is
validated once into a frozen :class:`_FeatureSpec`, and category codes in the
data are checked against it. :func:`_sample_feature` samples one feature at
arbitrary times without crossing an unusable interval or leaving a run.
:func:`_window_status` decides which events are kept and why,
:func:`_sample_events` samples event chunks for every feature, and
:class:`_Summaries` accumulates per-kind sufficient statistics chunk by chunk.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurospatial._exceptions import _format_error
from neurospatial._intervals import intervals_contain
from neurospatial._results import as_label_array, read_only_copy

if TYPE_CHECKING:
    from neurospatial._observation import ObservationSupport

_VALUE_KINDS = ("linear", "circular", "categorical")
_ALLOWED_INTERPOLATION = {
    "linear": ("linear", "previous"),
    "circular": ("circular", "previous"),
    "categorical": ("previous",),
}
_RADIAN_UNITS = frozenset({"rad", "radian", "radians"})
_EVENT_STATUSES = (
    "retained",
    "outside_recording",
    "outside_epochs",
    "incomplete_support",
    "no_support",
)
_MAX_CHUNK_CELLS = 2**22  # events * n_times * n_features per chunk


@dataclass(frozen=True)
class _FeatureSpec:
    """Validated per-feature metadata.

    Attributes
    ----------
    feature_ids : ndarray, shape (n_features,)
        Unique feature labels.
    value_kinds : tuple of str
        ``"linear"``, ``"circular"`` or ``"categorical"`` per feature.
    interpolation : tuple of str
        Resolved interpolation per feature.
    units : tuple of str or None
        Unit per feature; circular features default to ``"rad"``.
    category_features : ndarray of intp, shape (n_categories,)
        Feature column of each category row.
    category_codes : ndarray of int64, shape (n_categories,)
        Integer codes, sorted within each feature.
    category_labels : ndarray of str, shape (n_categories,)
        Label of each category row.
    """

    feature_ids: NDArray[Any]
    value_kinds: tuple[str, ...]
    interpolation: tuple[str, ...]
    units: tuple[str | None, ...]
    category_features: NDArray[np.intp]
    category_codes: NDArray[np.int64]
    category_labels: NDArray[np.str_]


def _plain(value: Any) -> Any:
    """Return a NumPy scalar as a plain Python value."""
    return value.item() if isinstance(value, np.generic) else value


def _key_index(
    mapping: Mapping[Any, Any] | None,
    name: str,
    ids: list[Any],
    call: str,
) -> dict[int, Any]:
    """Map ``{feature_id: value}`` to ``{column: value}``, rejecting unknown keys."""
    if mapping is None:
        return {}
    position = {fid: i for i, fid in enumerate(ids)}
    out: dict[int, Any] = {}
    for key, value in mapping.items():
        try:
            column = position[_plain(key)]
        except (KeyError, TypeError):
            raise ValueError(
                _format_error(
                    f"{call}: {name} has the key {key!r}, which is not a feature id.",
                    why=f"Valid feature ids are {ids}.",
                    fix=f"Key {name} by the ids in feature_ids.",
                )
            ) from None
        out[column] = value
    return out


def _resolve_features(
    n_features: int,
    *,
    feature_ids: Sequence[Any] | NDArray[Any] | None,
    feature_units: Mapping[Any, str | None] | None,
    value_kinds: Mapping[Any, str] | None,
    category_labels: Mapping[Any, Mapping[int, str]] | None,
    interpolation: Mapping[Any, str] | None,
    call: str,
) -> _FeatureSpec:
    """Validate per-feature metadata into a :class:`_FeatureSpec`.

    Parameters
    ----------
    n_features : int
        Number of signal columns.
    feature_ids : sequence or None
        Unique labels, shape ``(n_features,)``; default ``arange(n_features)``.
    feature_units, value_kinds, category_labels, interpolation : mapping or None
        Keyed by feature id; omitted features take defaults.
    call : str
        Calling function's name for error messages.

    Returns
    -------
    _FeatureSpec
        Frozen, validated metadata.

    Raises
    ------
    ValueError
        On any inconsistent or unsupported metadata.
    """
    if feature_ids is None:
        ids_array = np.arange(n_features)
    else:
        ids_array = as_label_array(feature_ids)
    if ids_array.ndim != 1 or len(ids_array) != n_features:
        raise ValueError(
            _format_error(
                f"{call}: got {len(ids_array)} feature_ids for {n_features} "
                "signal columns.",
                fix="Pass one unique id per column of signals.",
            )
        )
    ids = [_plain(f) for f in ids_array]
    seen = set()
    for fid in ids:
        if fid in seen:
            raise ValueError(
                _format_error(
                    f"{call}: duplicate feature id {fid!r}.",
                    why="Mappings are keyed by feature id, so ids must be unique.",
                    fix="Pass distinct feature_ids.",
                )
            )
        seen.add(fid)

    kinds_in = _key_index(value_kinds, "value_kinds", ids, call)
    interp_in = _key_index(interpolation, "interpolation", ids, call)
    units_in = _key_index(feature_units, "feature_units", ids, call)
    labels_in = _key_index(category_labels, "category_labels", ids, call)

    kinds: list[str] = []
    interps: list[str] = []
    units: list[str | None] = []
    cat_features: list[int] = []
    cat_codes: list[int] = []
    cat_labels: list[str] = []

    for col, fid in enumerate(ids):
        kind = kinds_in.get(col, "linear")
        if kind not in _VALUE_KINDS:
            raise ValueError(
                _format_error(
                    f"{call}: value_kinds[{fid!r}] is {kind!r}.",
                    why=f"Supported kinds are {list(_VALUE_KINDS)}.",
                    fix=f"Use one of {list(_VALUE_KINDS)}.",
                )
            )
        allowed = _ALLOWED_INTERPOLATION[kind]
        method = interp_in.get(col, allowed[0])
        if method not in allowed:
            why = (
                f"Feature {fid!r} is {kind}; allowed interpolation is {list(allowed)}."
            )
            if kind == "categorical":
                why = (
                    f"Feature {fid!r} is categorical: codes are labels, a value "
                    "between two codes is not a category."
                )
            raise ValueError(
                _format_error(
                    f"{call}: interpolation[{fid!r}] is {method!r}.",
                    why=why,
                    fix=f"Pass interpolation={{{fid!r}: {allowed[-1]!r}}}.",
                )
            )
        unit = units_in.get(col)
        if kind == "circular":
            if unit is None:
                unit = "rad"
            elif unit not in _RADIAN_UNITS:
                raise ValueError(
                    _format_error(
                        f"{call}: circular feature {fid!r} has unit {unit!r}.",
                        why="Circular values are radians.",
                        fix="Convert with np.deg2rad and pass feature_units="
                        f"{{{fid!r}: 'rad'}}.",
                    )
                )
        kinds.append(kind)
        interps.append(method)
        units.append(unit)

        mapping = labels_in.get(col)
        if kind != "categorical":
            if mapping is not None:
                raise ValueError(
                    _format_error(
                        f"{call}: category_labels given for feature {fid!r}, "
                        f"which is {kind}.",
                        fix=f"Set value_kinds={{{fid!r}: 'categorical'}} or drop "
                        "its category_labels.",
                    )
                )
            continue
        if mapping is None:
            raise ValueError(
                _format_error(
                    f"{call}: categorical feature {fid!r} has no category_labels.",
                    why="Codes are only meaningful with their labels.",
                    fix=f"Pass category_labels={{{fid!r}: {{0: 'label', ...}}}}.",
                )
            )
        if len(mapping) == 0:
            raise ValueError(
                _format_error(
                    f"{call}: category_labels for feature {fid!r} is empty.",
                    fix="List every code the signal can take with its label.",
                )
            )
        for code, label in mapping.items():
            if isinstance(code, bool | np.bool_) or not isinstance(
                code, int | np.integer
            ):
                raise ValueError(
                    _format_error(
                        f"{call}: category code {code!r} for feature {fid!r} is "
                        "not an integer.",
                        why="Codes must be integers (0 is allowed); bool is not.",
                        fix="Use integer keys such as {0: 'center', 1: 'arm'}.",
                    )
                )
            if not isinstance(label, str):
                raise ValueError(
                    _format_error(
                        f"{call}: label {label!r} for code {code} of feature "
                        f"{fid!r} is not a str.",
                        fix="Use string labels.",
                    )
                )
        for code, label in sorted(mapping.items(), key=lambda kv: int(kv[0])):
            cat_features.append(col)
            cat_codes.append(int(code))
            cat_labels.append(label)

    return _FeatureSpec(
        feature_ids=read_only_copy(ids_array),
        value_kinds=tuple(kinds),
        interpolation=tuple(interps),
        units=tuple(units),
        category_features=read_only_copy(cat_features, dtype=np.intp),
        category_codes=read_only_copy(cat_codes, dtype=np.int64),
        category_labels=read_only_copy(cat_labels, dtype=np.str_),
    )


def _check_category_codes(
    signals: NDArray[np.float64],
    sample_valid: NDArray[np.bool_],
    spec: _FeatureSpec,
    *,
    call: str,
) -> None:
    """Raise when a valid categorical sample is not a listed integer code.

    Parameters
    ----------
    signals : ndarray, shape (n_samples, n_features)
        Signal values.
    sample_valid : ndarray of bool, shape (n_samples, n_features)
        Per-feature sample validity; invalid samples are not checked.
    spec : _FeatureSpec
        Validated metadata.
    call : str
        Calling function's name for error messages.

    Raises
    ------
    ValueError
        Naming the first offending sample index and value.
    """
    for col, kind in enumerate(spec.value_kinds):
        if kind != "categorical":
            continue
        fid = _plain(spec.feature_ids[col])
        listed = spec.category_codes[spec.category_features == col]
        column = signals[:, col]
        checked = sample_valid[:, col] & np.isfinite(column)
        integer = column == np.round(column)
        listed_mask = np.isin(column, listed)
        bad = np.flatnonzero(checked & ~(integer & listed_mask))
        if len(bad) == 0:
            continue
        index = int(bad[0])
        value = float(column[index])
        reason = (
            f"is not in category_labels (codes {listed.tolist()})"
            if integer[index]
            else "is not an integer"
        )
        raise ValueError(
            _format_error(
                f"{call}: categorical feature {fid!r} has value {value!r} at "
                f"sample {index}, which {reason}.",
                why="Category codes must be listed integers; NaN marks missing.",
                fix="Add the code to category_labels, or set unknown samples to "
                "NaN (or False in valid).",
            )
        )


def _relative_times(window: tuple[float, float], dt: float) -> NDArray[np.float64]:
    """Return the query offsets ``start + dt * j`` that fall before ``stop``.

    Parameters
    ----------
    window : tuple of float
        ``(start, stop)`` in seconds relative to each event, ``start < stop``.
    dt : float
        Query spacing in seconds, ``> 0``.

    Returns
    -------
    ndarray of float64, shape (n_times,)
        ``start + dt * j`` for every integer ``j >= 0`` with
        ``start + dt * j < stop - 1e-9 * dt``; the tolerance keeps a rounding
        error from adding a query at ``stop``.
    """
    start, stop = window
    n_candidates = int(np.ceil((stop - start) / dt)) + 1
    offsets = start + dt * np.arange(n_candidates, dtype=np.float64)
    kept: NDArray[np.float64] = offsets[offsets < stop - 1e-9 * dt]
    return kept


def _wrap(angles: ArrayLike) -> NDArray[np.float64]:
    """Wrap angles in radians to ``[-pi, pi)``; a half-turn maps to ``-pi``."""
    wrapped = np.mod(np.asarray(angles, dtype=np.float64) + np.pi, 2 * np.pi) - np.pi
    # np.mod rounds a tiny negative remainder up to 2*pi, which would give +pi.
    out: NDArray[np.float64] = np.where(wrapped >= np.pi, -np.pi, wrapped)
    return out


def _sample_feature(
    times: NDArray[np.float64],
    values: NDArray[np.float64],
    support: ObservationSupport,
    queries: NDArray[np.float64],
    *,
    interpolation: str,
    kind: str,
    epochs: NDArray[np.float64] | None,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Sample one feature at arbitrary times, only where it was observed.

    A query exactly at a sample time is a point observation: it is supported
    when that sample is valid and, if ``epochs`` is given, the instant lies in
    one half-open epoch row. Any other query is supported only inside a usable
    interval ``[t_k, t_{k+1})`` of ``support``, so a value is never
    interpolated across a gap, invalid sample, declared break or epoch edge,
    and never extrapolated before the first or after the last sample.

    Parameters
    ----------
    times : ndarray of float64, shape (n_samples,)
        Sample times in seconds, strictly increasing.
    values : ndarray of float64, shape (n_samples,)
        One feature's values on ``times``.
    support : ObservationSupport
        This feature's resolved support on ``times``.
    queries : ndarray of float64, any shape
        Absolute query times in seconds.
    interpolation : {"linear", "circular", "previous"}
        ``"linear"``: straight line between the bracketing samples.
        ``"circular"``: shorter arc in radians; an exact half-turn goes the
        negative way. ``"previous"``: the value of the preceding sample.
    kind : {"linear", "circular", "categorical"}
        The feature's value kind. Every sampled value of a circular feature is
        wrapped to ``[-pi, pi)``, whatever the interpolation.
    epochs : ndarray of float64, shape (n_epochs, 2), or None
        Sorted, merged ``[start, stop)`` rows (from ``as_intervals``), the same
        epochs used to resolve ``support``. They gate exact-time queries,
        which use no interval.

    Returns
    -------
    sampled : ndarray of float64, same shape as ``queries``
        Sampled values; NaN where unsupported.
    supported : ndarray of bool, same shape as ``queries``
        True where the value is an observation or lies inside a usable
        interval.
    """
    q = np.asarray(queries, dtype=np.float64)
    n_samples = times.size
    k = np.searchsorted(times, q, side="right") - 1
    k_safe = np.clip(k, 0, n_samples - 1)
    exact = (k >= 0) & (times[k_safe] == q)

    at_sample = exact & support.sample_valid[k_safe]
    if epochs is not None:
        row = np.searchsorted(epochs[:, 0], q, side="right") - 1
        at_sample &= (row >= 0) & (q < epochs[np.maximum(row, 0), 1])

    between = ~exact & (k >= 0) & (k < n_samples - 1)
    if n_samples > 1:
        between &= support.interval_mask[np.clip(k, 0, n_samples - 2)]

    sampled = np.full(q.shape, np.nan)
    sampled[at_sample] = values[k[at_sample]]

    left = k[between]
    x0 = values[left]
    if interpolation == "previous":
        sampled[between] = x0
    else:
        weight = (q[between] - times[left]) / (times[left + 1] - times[left])
        step = values[left + 1] - x0
        if interpolation == "circular":
            sampled[between] = x0 + weight * _wrap(step)
        elif interpolation == "linear":
            sampled[between] = x0 + weight * step
        else:
            raise ValueError(f"unknown interpolation {interpolation!r}")

    supported: NDArray[np.bool_] = at_sample | between
    if kind == "circular":
        sampled[supported] = _wrap(sampled[supported])
    return sampled, supported


def _window_status(
    event_times: NDArray[np.float64],
    window: tuple[float, float],
    times: NDArray[np.float64],
    supports: Sequence[ObservationSupport],
    epochs: NDArray[np.float64] | None,
    window_coverage: str,
) -> NDArray[np.str_]:
    """Decide from each event's window interval whether the event is kept.

    The decision tests the interval ``[event + start, event + stop)`` itself,
    independently of the query grid, so an unusable interval between two
    queries or an unobserved tail after the last query is not missed.

    Parameters
    ----------
    event_times : ndarray of float64, shape (n_events,)
        Event times in seconds, input order.
    window : tuple of float
        ``(start, stop)`` in seconds relative to each event.
    times : ndarray of float64, shape (n_samples,)
        Sample times in seconds, strictly increasing.
    supports : sequence of ObservationSupport
        One resolved support per feature.
    epochs : ndarray of float64, shape (n_epochs, 2), or None
        Normalized ``[start, stop)`` rows (from ``as_intervals``).
    window_coverage : {"complete", "partial"}
        ``"complete"``: kept only when the whole window lies inside the
        recording, inside one epoch row (when given) and, for every feature,
        inside one run's ``[t_first, t_last]``. ``"partial"``: kept unless the
        window misses ``[times[0], times[-1]]`` entirely; whether any cell is
        supported is decided after sampling.

    Returns
    -------
    ndarray of str, shape (n_events,)
        ``"retained"`` or the first failing reason, in the order
        ``"outside_recording"``, ``"outside_epochs"``, ``"incomplete_support"``.
    """
    starts = event_times + window[0]
    stops = event_times + window[1]
    width = max(len(name) for name in _EVENT_STATUSES)
    status = np.full(event_times.shape, "retained", dtype=f"<U{width}")
    if window_coverage == "partial":
        status[(stops <= times[0]) | (starts > times[-1])] = "outside_recording"
        return status

    covered = np.ones(event_times.shape, dtype=bool)
    for support in supports:
        covered &= intervals_contain(support.run_time_bounds, starts, stops)
    status[~covered] = "incomplete_support"
    if epochs is not None:
        status[~intervals_contain(epochs, starts, stops)] = "outside_epochs"
    status[(starts < times[0]) | (stops > times[-1])] = "outside_recording"
    return status


def _overlapping(
    event_times: NDArray[np.float64],
    window: tuple[float, float],
    retained: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    """Flag retained events whose window intersects another retained window.

    Parameters
    ----------
    event_times : ndarray of float64, shape (n_events,)
        Event times in seconds, input order.
    window : tuple of float
        ``(start, stop)`` in seconds relative to each event; half-open.
    retained : ndarray of bool, shape (n_events,)
        Events kept by :func:`_window_status`.

    Returns
    -------
    ndarray of bool, shape (n_events,)
        True when the event is retained and its window shares time with
        another retained event's window. Duplicate times overlap each other;
        windows that only touch do not. Excluded events are False.

    Notes
    -----
    Window bounds are ``event + start`` and ``event + stop`` in floating
    point, so two windows meant to touch can round to a tiny overlap and be
    flagged: with ``window=(-0.1, 0.1)``, events at 0.1 and 0.3 s give a stop
    of ``0.1 + 0.1 == 0.2`` and a start of ``0.3 - 0.1 == 0.19999999999999998``.
    """
    kept = np.flatnonzero(retained)
    starts = event_times[kept] + window[0]
    order = np.argsort(starts, kind="stable")
    starts = starts[order]
    stops = event_times[kept][order] + window[1]
    # Sorted by start: a window overlaps an earlier one iff it starts before
    # the latest earlier stop, and a later one iff the next window starts
    # before it stops.
    latest_stop = np.maximum.accumulate(stops)
    with_earlier = np.r_[False, starts[1:] < latest_stop[:-1]]
    with_later = np.r_[starts[1:] < stops[:-1], False]
    out = np.zeros(event_times.shape, dtype=bool)
    out[kept[order]] = with_earlier | with_later
    return out


def _sample_events(
    event_times: NDArray[np.float64],
    relative_times: NDArray[np.float64],
    times: NDArray[np.float64],
    signals: NDArray[np.float64],
    supports: Sequence[ObservationSupport],
    spec: _FeatureSpec,
    epochs: NDArray[np.float64] | None,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Sample every feature around each event, in input order.

    Parameters
    ----------
    event_times : ndarray of float64, shape (n_chunk,)
        Event times in seconds.
    relative_times : ndarray of float64, shape (n_times,)
        Query offsets in seconds (:func:`_relative_times`).
    times : ndarray of float64, shape (n_samples,)
        Sample times in seconds.
    signals : ndarray of float64, shape (n_samples, n_features)
        Signal values.
    supports : sequence of ObservationSupport
        One resolved support per feature, built with ``epochs``.
    spec : _FeatureSpec
        Per-feature value kind and interpolation.
    epochs : ndarray of float64, shape (n_epochs, 2), or None
        The normalized epochs used to resolve ``supports``.

    Returns
    -------
    values : ndarray of float64, shape (n_chunk, n_times, n_features)
        Sampled values; NaN where unsupported. Categorical codes as floats.
    support : ndarray of bool, shape (n_chunk, n_times, n_features)
        True where the value was observed.
    """
    queries = event_times[:, None] + relative_times[None, :]
    shape = (*queries.shape, signals.shape[1])
    values = np.empty(shape, dtype=np.float64)
    support = np.empty(shape, dtype=bool)
    for f, feature_support in enumerate(supports):
        values[..., f], support[..., f] = _sample_feature(
            times,
            signals[:, f],
            feature_support,
            queries,
            interpolation=spec.interpolation[f],
            kind=spec.value_kinds[f],
            epochs=epochs,
        )
    return values, support


def _event_chunks(
    n_events: int, cells_per_event: int, max_cells: int = _MAX_CHUNK_CELLS
) -> Iterator[slice]:
    """Split ``range(n_events)`` into consecutive slices of bounded size.

    Each slice holds at most ``max_cells // cells_per_event`` events, and at
    least one event even when a single event exceeds ``max_cells``.
    """
    per_chunk = max(1, max_cells // max(1, cells_per_event))
    for first in range(0, n_events, per_chunk):
        yield slice(first, min(first + per_chunk, n_events))


@dataclass
class _Summaries:
    """Per-cell sufficient statistics, accumulated one event chunk at a time.

    Attributes
    ----------
    counts : ndarray of int64, shape (n_times, n_features)
        Supported events per cell.
    sums : ndarray of float64, shape (n_times, n_features)
        Sum of supported values; NaN columns for non-linear features.
    sin_sums, cos_sums : ndarray of float64, shape (n_times, n_features)
        Sum of sine and cosine of supported angles; NaN columns for
        non-circular features.
    category_counts : ndarray of int64, shape (n_times, n_categories)
        Supported events holding each category row's code.
    spec : _FeatureSpec
        Value kinds and category rows of the features.
    """

    counts: NDArray[np.int64]
    sums: NDArray[np.float64]
    sin_sums: NDArray[np.float64]
    cos_sums: NDArray[np.float64]
    category_counts: NDArray[np.int64]
    spec: _FeatureSpec = field(repr=False)

    @classmethod
    def empty(cls, n_times: int, spec: _FeatureSpec) -> _Summaries:
        """Return zeroed statistics, NaN in columns that do not apply."""
        kinds = np.asarray(spec.value_kinds)
        n_features = kinds.size
        sums = np.zeros((n_times, n_features))
        sums[:, kinds != "linear"] = np.nan
        sin_sums = np.zeros((n_times, n_features))
        sin_sums[:, kinds != "circular"] = np.nan
        return cls(
            counts=np.zeros((n_times, n_features), dtype=np.int64),
            sums=sums,
            sin_sums=sin_sums,
            cos_sums=sin_sums.copy(),
            category_counts=np.zeros(
                (n_times, spec.category_codes.size), dtype=np.int64
            ),
            spec=spec,
        )

    def add(self, values: NDArray[np.float64], support: NDArray[np.bool_]) -> None:
        """Add one chunk of sampled events.

        Parameters
        ----------
        values : ndarray of float64, shape (n_chunk, n_times, n_features)
            Sampled values (:func:`_sample_events`).
        support : ndarray of bool, shape (n_chunk, n_times, n_features)
            Where ``values`` were observed; other cells are ignored.
        """
        kinds = np.asarray(self.spec.value_kinds)
        self.counts += support.sum(axis=0, dtype=np.int64)
        linear = kinds == "linear"
        self.sums[:, linear] += np.where(
            support[..., linear], values[..., linear], 0.0
        ).sum(axis=0)
        circular = kinds == "circular"
        angles = values[..., circular]
        observed = support[..., circular]
        self.sin_sums[:, circular] += np.where(observed, np.sin(angles), 0.0).sum(
            axis=0
        )
        self.cos_sums[:, circular] += np.where(observed, np.cos(angles), 0.0).sum(
            axis=0
        )
        columns = self.spec.category_features
        matches = support[..., columns] & (
            values[..., columns] == self.spec.category_codes
        )
        self.category_counts += matches.sum(axis=0, dtype=np.int64)


def _circular_means(
    sin_sums: NDArray[np.float64],
    cos_sums: NDArray[np.float64],
    counts: NDArray[np.int64],
    threshold: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Mean direction and resultant length from summed sines and cosines.

    Parameters
    ----------
    sin_sums, cos_sums : ndarray of float64, shape (n_times, n_features)
        Summed sine and cosine; NaN columns for non-circular features.
    counts : ndarray of int64, shape (n_times, n_features)
        Number of summed angles per cell.
    threshold : float
        Resultant lengths at or below it have no defined direction.

    Returns
    -------
    means : ndarray of float64, shape (n_times, n_features)
        ``atan2(sin_sums, cos_sums)`` in ``(-pi, pi]``; NaN unless defined.
    resultant_lengths : ndarray of float64, shape (n_times, n_features)
        ``hypot(sin_sums, cos_sums) / counts``; NaN where ``counts == 0``.
    defined : ndarray of bool, shape (n_times, n_features)
        ``counts > 0`` and resultant length ``> threshold``.
    """
    observed = counts > 0
    lengths = np.full(counts.shape, np.nan)
    np.divide(np.hypot(sin_sums, cos_sums), counts, out=lengths, where=observed)
    defined: NDArray[np.bool_] = observed & (lengths > threshold)
    means = np.where(defined, np.arctan2(sin_sums, cos_sums), np.nan)
    # Sampled angles lie in [-pi, pi), so a half-turn's sine sum is a tiny
    # negative number and atan2 returns -pi; report the half-turn as +pi.
    means[means == -np.pi] = np.pi
    return means, lengths, defined
