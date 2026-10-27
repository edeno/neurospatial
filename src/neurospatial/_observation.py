"""Resolve which samples and intervals of a trajectory were actually observed.

Shared validation of times, positions and signals, and resolution of observed
samples, intervals and runs, used by the behavior, event, animation and
encoding analyses. A sample is valid when
every coordinate is finite. A missing sample is treated exactly like a deleted
row: interval ``k`` (samples ``k`` and ``k + 1``) belongs to the gap between
the last valid sample at or before ``k`` and the first valid sample after it,
and is usable when that gap passes the ``max_gap`` and ``epochs`` gates and no
recording break is declared inside it. Runs are maximal sequences of
consecutive usable intervals; they start and end at valid samples. Nothing
here depends on an ``Environment``.
"""

from __future__ import annotations

import warnings
from collections.abc import Hashable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurospatial._exceptions import _format_error
from neurospatial._intervals import (
    as_intervals,
    gap_endpoints,
    interval_valid_mask,
    observed_interval_mask,
    ragged_arange,
    run_sample_bounds,
    validate_max_gap,
)
from neurospatial._results import ReadOnlyArraysMixin, plain_value, read_only_copy
from neurospatial._stacklevel import _external_stacklevel
from neurospatial._validation import (
    times_positions_problems,
    validate_strictly_increasing,
    validate_times_positions,
)


@dataclass(frozen=True, repr=False, eq=False)
class ObservationSupport(ReadOnlyArraysMixin):
    """Valid samples, usable intervals and runs of one trajectory.

    Attributes
    ----------
    sample_valid : ndarray of bool, shape (n_samples,)
        True where every coordinate is finite.
    interval_gates : ndarray of bool, shape (max(n_samples - 1, 0),)
        True for intervals whose gap passes ``max_gap``, ``epochs`` and
        ``break_before``. The gap runs between the valid samples on either
        side; an interval with no valid sample on one side is gated on its
        own samples. False marks a boundary: a tracking loss longer than
        ``max_gap``, an epoch edge or a declared break.
    interval_mask : ndarray of bool, shape (max(n_samples - 1, 0),)
        True for usable intervals: gated and with a valid sample on both
        sides, so intervals inside a bridged stretch of missing samples are
        usable.
    run_ids : ndarray of intp, shape (n_samples,)
        Run index ``0..n_runs-1`` in time order of each valid sample; -1 for
        missing samples and valid samples that touch no usable interval.
    run_sample_bounds : ndarray of intp, shape (n_runs, 2)
        Inclusive ``(first, last)`` original sample indices of each run; both
        are valid samples, and missing samples between them are bridged.
    run_time_bounds : ndarray of float64, shape (n_runs, 2)
        ``times`` at those indices, in seconds.
    """

    sample_valid: NDArray[np.bool_]
    interval_gates: NDArray[np.bool_]
    interval_mask: NDArray[np.bool_]
    run_ids: NDArray[np.intp]
    run_sample_bounds: NDArray[np.intp]
    run_time_bounds: NDArray[np.float64]

    def __post_init__(self) -> None:
        self._own_arrays()

    @property
    def isolated(self) -> NDArray[np.bool_]:
        """Valid samples that touch no usable interval, shape (n_samples,)."""
        isolated: NDArray[np.bool_] = self.sample_valid & (self.run_ids == -1)
        return isolated

    def run_samples(self) -> list[NDArray[np.intp]]:
        """Indices of the valid samples of each run, in time order."""
        return [
            first + np.flatnonzero(self.sample_valid[first : last + 1])
            for first, last in self.run_sample_bounds
        ]


def validate_break_before(
    break_before: Any, n_samples: int, *, call: str
) -> NDArray[np.bool_] | None:
    """Return a read-only copy of ``break_before`` as a Boolean array, or None.

    The copy means a result or a shuffle null holding it cannot change when
    the caller edits their array.
    """
    if break_before is None:
        return None
    flags = np.asarray(break_before)
    if flags.dtype != np.bool_ or flags.shape != (n_samples,):
        raise ValueError(
            _format_error(
                f"{call}() needs break_before to be a Boolean array of shape "
                f"({n_samples},); got dtype {flags.dtype} and shape {flags.shape}.",
                why="True at sample k declares a recording break between k-1 and k.",
                fix=(
                    "pass a Boolean array with one entry per sample, e.g. "
                    "np.diff(recording_ids, prepend=recording_ids[0]) != 0"
                ),
            )
        )
    return read_only_copy(flags)


def _support_from_gates(
    times: NDArray[np.float64],
    sample_valid: NDArray[np.bool_],
    raw_gates: NDArray[np.bool_],
    *,
    max_gap: float | None,
    epochs: NDArray[np.float64] | None,
    break_before: NDArray[np.bool_] | None,
) -> ObservationSupport:
    """Resolve intervals and runs for one sample-validity vector.

    Missing samples are bridged like deleted rows: the gates are evaluated on
    the gap between consecutive valid samples (``interval_valid_mask`` with
    ``sample_valid``). ``raw_gates`` are the gates of each interval's own
    samples, used for intervals with no valid sample on one side.
    """
    n_samples = times.size
    interval_mask = interval_valid_mask(
        times,
        max_gap=max_gap,
        epochs=epochs,
        break_before=break_before,
        sample_valid=sample_valid,
    )
    _, _, linked = gap_endpoints(sample_valid)
    interval_gates = np.where(linked, interval_mask, raw_gates)
    bounds = run_sample_bounds(interval_mask)
    run_ids = np.full(n_samples, -1, dtype=np.intp)
    lengths = bounds[:, 1] - bounds[:, 0] + 1
    samples = np.repeat(bounds[:, 0], lengths) + ragged_arange(lengths)
    run_ids[samples] = np.repeat(np.arange(lengths.size, dtype=np.intp), lengths)
    run_ids[~sample_valid] = -1
    return ObservationSupport(
        sample_valid=sample_valid,
        interval_gates=interval_gates,
        interval_mask=interval_mask,
        run_ids=run_ids,
        run_sample_bounds=bounds,
        run_time_bounds=times[bounds],
    )


def resolve_observation_support(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    max_gap: float | None,
    epochs: Any,
    break_before: NDArray[np.bool_] | None,
    result: str,
) -> ObservationSupport:
    """Resolve usable intervals and runs from validated inputs.

    Parameters
    ----------
    times : ndarray, shape (n_samples,)
        Seconds; already validated and strictly increasing.
    positions : ndarray, shape (n_samples, n_dims)
        Float coordinates; NaN or infinite entries mark a missing sample.
    max_gap, epochs
        The shared interval gates (``observed_interval_mask``).
    break_before : ndarray of bool, shape (n_samples,) or None
        Declared recording breaks; entry 0 is ignored.
    result : str
        What an all-excluded selection leaves, for the shared warning.
    """
    sample_valid: NDArray[np.bool_] = np.asarray(np.isfinite(positions).all(axis=1))
    return resolve_sample_support(
        times,
        sample_valid,
        max_gap=max_gap,
        epochs=epochs,
        break_before=break_before,
        result=result,
    )


def resolve_sample_support(
    times: NDArray[np.float64],
    sample_valid: NDArray[np.bool_],
    *,
    max_gap: float | None,
    epochs: Any,
    break_before: NDArray[np.bool_] | None,
    result: str,
) -> ObservationSupport:
    """Resolve usable intervals and runs from a per-sample validity vector.

    Parameters
    ----------
    times : ndarray, shape (n_samples,)
        Seconds; already validated and strictly increasing.
    sample_valid : ndarray of bool, shape (n_samples,)
        True for samples that were observed.
    max_gap, epochs
        The shared interval gates (``observed_interval_mask``).
    break_before : ndarray of bool, shape (n_samples,) or None
        Declared recording breaks; entry 0 is ignored.
    result : str
        What an all-excluded selection leaves, for the shared warning.

    Examples
    --------
    The missing sample sits in a 0.2 s gap, bridged at ``max_gap=0.5``
    and a tracking loss at ``max_gap=0.15``:

    >>> import numpy as np
    >>> valid = np.array([True, True, False, True, True])
    >>> for max_gap in (0.5, 0.15):
    ...     support = resolve_sample_support(
    ...         np.arange(5) * 0.1,
    ...         valid,
    ...         max_gap=max_gap,
    ...         epochs=None,
    ...         break_before=None,
    ...         result="the result is empty",
    ...     )
    ...     print(support.interval_mask)
    [ True  True  True  True]
    [ True False False  True]
    """
    raw_gates = observed_interval_mask(
        times, max_gap=max_gap, epochs=epochs, break_before=break_before, result=result
    )
    return _support_from_gates(
        times,
        sample_valid,
        raw_gates,
        max_gap=max_gap,
        epochs=as_intervals(epochs, name="epochs"),
        break_before=break_before,
    )


def label_codes(
    labels: NDArray[Any], missing_label: Hashable | None
) -> tuple[NDArray[np.intp], list[Hashable]]:
    """Integer code of each label in first-seen order; -1 for a missing label.

    None, NaN and ``missing_label`` (when not None) are missing. Values are
    plain Python scalars.

    Parameters
    ----------
    labels : ndarray, shape (n_labels,)
        Hashable labels, usually an object array.
    missing_label : hashable or None
        An extra label value that means "no label"; None declares no extra
        value.

    Returns
    -------
    codes : ndarray of intp, shape (n_labels,)
        Index of each label into ``values``; -1 marks a missing label.
    values : list of hashable
        Distinct non-missing labels in first-seen order.

    Examples
    --------
    >>> import numpy as np
    >>> label_codes(np.array(["a", None, "b", "a"], dtype=object), None)
    (array([ 0, -1,  1,  0]), ['a', 'b'])
    >>> label_codes(np.array(["rest", "a", "rest"], dtype=object), "rest")
    (array([-1,  0, -1]), ['a'])
    """
    import pandas as pd

    factorized, uniques = pd.factorize(labels, use_na_sentinel=True)
    codes = factorized.astype(np.intp)
    # tolist() already yields Python scalars; object arrays can still hold NumPy ones.
    values: list[Hashable] = [plain_value(v) for v in np.asarray(uniques).tolist()]
    if missing_label is not None:
        keep = np.array([not bool(v == missing_label) for v in values], dtype=bool)
        if keep.all():
            _warn_near_missing_label(missing_label, values)
        # One extra slot so that code -1 (already missing) stays -1.
        renumber = np.full(len(values) + 1, -1, dtype=np.intp)
        renumber[:-1][keep] = np.arange(np.count_nonzero(keep), dtype=np.intp)
        codes = renumber[codes]
        values = [v for v, k in zip(values, keep, strict=True) if k]
    return codes, values


def _warn_near_missing_label(missing_label: Hashable, values: list[Hashable]) -> None:
    """Warn when ``missing_label`` is absent but a label differs only in case."""
    if not isinstance(missing_label, str):
        return
    key = missing_label.strip().casefold()
    near = [v for v in values if isinstance(v, str) and v.strip().casefold() == key]
    if near:
        warnings.warn(
            f"missing_label={missing_label!r} matches no label, but {near[0]!r} "
            "differs from it only in case or spaces, so it is kept as a real "
            f"condition. Fix: pass missing_label={near[0]!r}.",
            UserWarning,
            stacklevel=_external_stacklevel(),
        )


def label_runs(
    codes: NDArray[np.intp], interval_mask: NDArray[np.bool_]
) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.intp]]:
    """Runs of consecutive retained intervals sharing a label code.

    A run ends at an interval the support resolver rejected, at a missing
    label, or where the label code changes; nothing joins across those.

    Parameters
    ----------
    codes : ndarray of intp, shape (n_intervals,)
        Label code of each interval (the label at its start sample); -1 marks a
        missing label.
    interval_mask : ndarray of bool, shape (n_intervals,)
        Intervals the support resolver accepted.

    Returns
    -------
    first_interval, last_interval, code : ndarray of intp, shape (n_runs,)
        Inclusive first and last interval of each run and its label code,
        in time order.

    Examples
    --------
    >>> import numpy as np
    >>> codes = np.array([0, 0, 1, 0, 0], dtype=np.intp)
    >>> mask = np.array([True, True, True, False, True])
    >>> label_runs(codes, mask)
    (array([0, 2, 4]), array([1, 2, 4]), array([0, 1, 0]))
    """
    kept = np.flatnonzero(interval_mask & (codes >= 0))
    new = np.ones(kept.size, dtype=bool)
    new[1:] = (kept[1:] != kept[:-1] + 1) | (codes[kept[1:]] != codes[kept[:-1]])
    first = kept[new]
    last = kept[np.r_[new[1:], True]] if kept.size else kept
    return first, last, codes[first]


def validated_observations(
    times: Any,
    positions: Any,
    *,
    call: str,
    max_gap: float | None,
    epochs: Any,
    break_before: Any,
    result: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64], ObservationSupport]:
    """Validate a ``(times, positions)`` pair and resolve its observation support.

    Parameters
    ----------
    times : array-like, shape (n_samples,)
        Seconds; finite and strictly increasing.
    positions : array-like, shape (n_samples, n_dims) or (n_samples,)
        Coordinates; NaN rows are missing samples.
    call : str
        Public function name, used in messages.
    max_gap, epochs, break_before
        The shared interval gates (``observed_interval_mask``).
    result : str
        What an all-excluded selection leaves, for the shared warning.

    Returns
    -------
    times : ndarray, shape (n_samples,)
    positions : ndarray, shape (n_samples, n_dims)
        1-D input is reshaped to ``(n_samples, 1)``.
    support : ObservationSupport

    Raises
    ------
    ValueError
        If the inputs are malformed or empty (zero samples), if ``times``
        are not strictly increasing, or if ``max_gap`` or ``break_before``
        is invalid. One sample is valid input: it has a position but no
        observed interval.
    """
    t, p = validate_times_positions(times, positions, call=call)
    if t.size == 0:
        raise empty_samples_error(call, ("times", "positions"))
    validate_strictly_increasing(t, call=call)
    if p.ndim == 1:
        p = p[:, None]
    validate_max_gap(max_gap)
    breaks = validate_break_before(break_before, t.size, call=call)
    support = resolve_observation_support(
        t, p, max_gap=max_gap, epochs=epochs, break_before=breaks, result=result
    )
    return t, p, support


def empty_samples_error(call: str, names: tuple[str, ...]) -> ValueError:
    """The error for zero samples, naming the empty arguments of ``call``.

    Examples
    --------
    >>> print(empty_samples_error("f", ("times", "nose", "tail")))
    f() got no samples: times, nose and tail are empty.
    Why: there is no observation to measure.
    Fix: pass at least one sample; if the arrays came from an epoch or trial selection, check that the selection kept any samples.
    """
    listed = names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"
    verb = "is" if len(names) == 1 else "are"
    return ValueError(
        _format_error(
            f"{call}() got no samples: {listed} {verb} empty.",
            why="Why: there is no observation to measure.",
            fix=(
                "pass at least one sample; if the arrays came from an epoch or "
                "trial selection, check that the selection kept any samples."
            ),
        )
    )


def _validated_signal_times(times: ArrayLike, *, call: str) -> NDArray[np.float64]:
    try:
        t = np.asarray(times, dtype=np.float64)
    except (TypeError, ValueError):
        t = None
    problems = (
        ["times must contain numeric values convertible to float64."]
        if t is None
        else times_positions_problems(t, None)[0]
    )
    if problems:
        raise ValueError(
            _format_error(
                f"Invalid times passed to {call}():\n- " + "\n- ".join(problems),
                why="each interval [times[k], times[k+1]) is judged by its duration, "
                "so mis-shaped or unsorted timestamps give wrong numbers.",
                fix="pass times as a sorted 1-D array of finite timestamps in seconds.",
            )
        )
    assert t is not None
    if t.size == 0:
        raise empty_samples_error(call, ("times",))
    validate_strictly_increasing(t, call=call)
    return t


_MAX_EXACT_INTEGER = 2**53


def _require_exact_integers(raw: NDArray[Any], *, call: str) -> None:
    """Refuse integer arrays whose values float64 cannot hold exactly."""
    if raw.dtype.kind not in "iu" or raw.size == 0:
        return
    # Compare in Python ints so uint64 and int64 extremes cannot wrap.
    low, high = int(raw.min()), int(raw.max())
    if low >= -_MAX_EXACT_INTEGER and high <= _MAX_EXACT_INTEGER:
        return
    values = raw.reshape(raw.shape[0], -1) if raw.ndim else raw.reshape(1, 1)
    bad = np.flatnonzero(
        ((values < -_MAX_EXACT_INTEGER) | (values > _MAX_EXACT_INTEGER)).any(axis=0)
    )
    where = f"column(s) {bad.tolist()}" if raw.ndim == 2 else "the signal"
    raise ValueError(
        _format_error(
            f"{call}() got integer values beyond +/-2**53 in {where} "
            f"(range {low} to {high}).",
            why="float64 cannot represent such integers exactly, so distinct "
            "codes or counts would silently merge.",
            fix="re-code categories to small integers, e.g. "
            "_, codes = np.unique(x, return_inverse=True).",
        )
    )


def _validated_signal_array(
    signals: ArrayLike, n_samples: int, *, call: str
) -> NDArray[np.float64]:
    fix = (
        "pass a real-valued array of shape (n_samples, n_features) or "
        "(n_samples,) with one row per timestamp; use NaN for missing samples."
    )
    raw = np.asarray(signals)
    if raw.dtype.kind not in "biuf":
        raise ValueError(
            _format_error(
                f"{call}() needs real numeric signals; got dtype {raw.dtype}.",
                why="complex and non-numeric values have no ordering or average "
                "that the analysis could use.",
                fix=fix,
            )
        )
    _require_exact_integers(raw, call=call)
    sig = raw.astype(np.float64)
    if sig.ndim not in (1, 2) or sig.shape[0] != n_samples:
        raise ValueError(
            _format_error(
                f"{call}() needs signals of shape (n_samples, n_features) or "
                f"(n_samples,) with n_samples={n_samples}; got shape {sig.shape}.",
                why="each signal row is the observation at the matching timestamp.",
                fix=fix,
            )
        )
    if sig.ndim == 2 and sig.shape[1] == 0:
        raise ValueError(
            _format_error(
                f"{call}() needs at least one feature; got signals of shape "
                f"{sig.shape}.",
                why="each column of signals is one feature to align, and there is "
                "nothing to align without one.",
                fix=fix,
            )
        )
    return sig[:, None] if sig.ndim == 1 else sig


def _validated_signal_flags(
    valid: ArrayLike | None, n_samples: int, n_features: int, *, call: str
) -> NDArray[np.bool_] | None:
    if valid is None:
        return None
    flags = np.asarray(valid)
    if flags.dtype != np.bool_ or flags.shape not in (
        (n_samples,),
        (n_samples, n_features),
    ):
        raise ValueError(
            _format_error(
                f"{call}() needs valid to be a Boolean array of shape "
                f"({n_samples},) or ({n_samples}, {n_features}); got dtype "
                f"{flags.dtype} and shape {flags.shape}.",
                why="valid marks which samples may be used, for all features "
                "(1-D) or per feature (2-D).",
                fix="pass a Boolean array, e.g. valid = speed > 5.0",
            )
        )
    return flags if flags.ndim == 2 else flags[:, None]


def validated_signals(
    times: ArrayLike,
    signals: ArrayLike,
    *,
    valid: ArrayLike | None,
    call: str,
    max_gap: float | None,
    epochs: Any,
    break_before: Any,
    result: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64], tuple[ObservationSupport, ...]]:
    """Validate a signal table and resolve observation support for each feature.

    Features share one clock, so the ``max_gap``, ``epochs`` and ``break_before``
    gates are resolved once (one excluded-interval warning). Validity differs
    by feature: sample ``i`` of feature ``f`` is valid when
    ``isfinite(signals[i, f])`` and ``valid[i, f]``. A finite zero is valid.

    Parameters
    ----------
    times : array-like, shape (n_samples,)
        Seconds; finite, strictly increasing, non-empty.
    signals : array-like, shape (n_samples, n_features) or (n_samples,)
        Real values; NaN or infinity marks a missing sample of that feature.
    valid : array-like of bool, shape (n_samples,) or (n_samples, n_features), or None
        Extra validity for every feature (1-D) or per feature (2-D).
    call : str
        Public function name, used in messages.
    max_gap, epochs, break_before
        The shared interval gates, as for ``validated_observations``.
    result : str
        What an all-excluded selection leaves, for the shared warning.

    Returns
    -------
    times : ndarray, shape (n_samples,)
    signals : ndarray, shape (n_samples, n_features)
        float64; 1-D input is reshaped to ``(n_samples, 1)``.
    supports : tuple of ObservationSupport
        One per feature.

    Raises
    ------
    ValueError
        If the inputs are malformed, empty or repeat a timestamp.
    """
    t = _validated_signal_times(times, call=call)
    sig = _validated_signal_array(signals, t.size, call=call)
    flags = _validated_signal_flags(valid, t.size, sig.shape[1], call=call)
    validate_max_gap(max_gap)
    breaks = validate_break_before(break_before, t.size, call=call)
    sample_valid = np.isfinite(sig)
    if flags is not None:
        sample_valid = sample_valid & flags
    raw_gates = observed_interval_mask(
        t, max_gap=max_gap, epochs=epochs, break_before=breaks, result=result
    )
    resolved_epochs = as_intervals(epochs, name="epochs")
    supports = tuple(
        _support_from_gates(
            t,
            np.ascontiguousarray(sample_valid[:, f]),
            raw_gates,
            max_gap=max_gap,
            epochs=resolved_epochs,
            break_before=breaks,
        )
        for f in range(sig.shape[1])
    )
    return t, sig, supports
