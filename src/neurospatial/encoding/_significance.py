"""Shared circular-shift engine and cell-predicate criterion validation."""

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Callable, Hashable, Sequence
from types import MappingProxyType
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurospatial._exceptions import _format_error
from neurospatial._intervals import resolve_time_windows
from neurospatial._random import _entropy, _stream
from neurospatial._results import plain_value, resolve_unit_ids
from neurospatial._spikes import _as_spike_trains_with_ids
from neurospatial._stacklevel import _external_stacklevel
from neurospatial.stats import _validate_circular_shift_settings
from neurospatial.stats.shuffle import (
    ShuffleTestResult,
    _shuffle_p_values,
    compute_shuffle_zscore,
    shuffle_spike_times_circular,
)

# Shuffle settings every cell-type shuffle test shares. The ``*_significance``
# signatures spell ``n_shuffles`` and ``min_shift`` out so the API reference
# shows the values; a test pins them to this mapping.
_SHUFFLE_DEFAULTS = MappingProxyType(
    {"n_shuffles": 1000, "min_shift": 20.0, "alpha": 0.05}
)


def _stream_key(label: Hashable) -> int:
    """Stable key shared by Python and NumPy scalar representations."""
    value = np.asarray(label).item()
    return int.from_bytes(hashlib.sha256(repr(value).encode()).digest()[:8], "little")


def resolve_shuffle_settings(
    n_shuffles: int | None, min_shift: float | None
) -> dict[str, Any]:
    """Fill unset shuffle-predicate settings from ``_SHUFFLE_DEFAULTS``."""
    return {
        "n_shuffles": int(_SHUFFLE_DEFAULTS["n_shuffles"])
        if n_shuffles is None
        else n_shuffles,
        "min_shift": _SHUFFLE_DEFAULTS["min_shift"] if min_shift is None else min_shift,
    }


class ShuffleInputs(NamedTuple):
    """A shuffle test's own copies of its inputs, with labels and windows resolved."""

    trains: list[NDArray[np.float64]]
    """One spike-time array per unit, seconds."""
    times: NDArray[np.float64]
    """Sample times, seconds, shape (n_time,)."""
    copies: tuple[NDArray[np.float64], ...]
    """The per-sample arrays passed as ``copies`` (positions, headings)."""
    unit_ids: NDArray[Any]
    """One label per unit."""
    epochs: NDArray[np.float64] | None
    """Resolved ``epochs`` intervals."""
    spike_window: NDArray[np.float64] | None
    """Resolved ``spike_window`` interval."""


def shuffle_inputs(
    spike_times: Any,
    times: ArrayLike,
    copies: Sequence[ArrayLike] = (),
    *,
    unit_ids: NDArray[Any] | Sequence[Any] | None,
    epochs: Any,
    spike_window: Any,
    call: str,
) -> ShuffleInputs:
    """Copy a shuffle test's inputs and resolve its unit labels and windows.

    The test reads its inputs once per shuffle, so it works on float64
    copies: caller mutation during the shuffles cannot change the null.

    Parameters
    ----------
    spike_times : spike trains
        Any spike input the ``*_significance`` functions accept.
    times : array_like, shape (n_time,)
        Sample times, seconds.
    copies : sequence of array_like, default=()
        Further per-sample arrays to copy, such as positions and headings.
    unit_ids : sequence or None
        Caller labels; ``None`` uses the labels the spike input carries, or
        ``0..n_units-1``.
    epochs, spike_window : interval-like or None
        As the ``*_significance`` functions accept them.
    call : str
        Public function name for error messages.

    Returns
    -------
    ShuffleInputs
    """
    trains, input_ids = _as_spike_trains_with_ids(spike_times)
    trains = [np.array(train, dtype=np.float64, copy=True) for train in trains]
    times = np.array(times, dtype=np.float64, copy=True)
    copied = tuple(np.array(x, dtype=np.float64, copy=True) for x in copies)
    ids = np.array(
        resolve_unit_ids(unit_ids, len(trains), input_ids=input_ids, context=call),
        copy=True,
    )
    resolved_epochs, resolved_spike_window = resolve_time_windows(epochs, spike_window)
    return ShuffleInputs(
        trains, times, copied, ids, resolved_epochs, resolved_spike_window
    )


def snapshot_options(options: dict[str, Any], **fixed: Any) -> dict[str, Any]:
    """Copy array-like options so caller mutation cannot leak into the null."""
    snapshot = {
        key: value.copy()
        if isinstance(value, np.ndarray)
        else np.array(value, copy=True)
        if isinstance(value, (list, tuple))
        else value
        for key, value in options.items()
    }
    snapshot.update(fixed)
    return snapshot


def run_shuffle_test(
    statistic: Callable[[list[NDArray[np.float64]]], ArrayLike],
    spike_times: list[NDArray[np.float64]],
    windows: NDArray[np.float64],
    unit_ids: NDArray[Any],
    *,
    n_shuffles: int,
    min_shift: float,
    rng: np.random.Generator | int | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Recompute observed/null statistics using a random stream per unit label."""
    _validate_circular_shift_settings(n_shuffles, min_shift)
    entropy = _entropy(rng)
    streams = [
        shuffle_spike_times_circular(
            train,
            windows,
            n_shuffles=n_shuffles,
            min_shift=min_shift,
            rng=_stream(entropy, _stream_key(uid)),
        )
        for train, uid in zip(spike_times, unit_ids, strict=True)
    ]

    def evaluate(trains: list[NDArray[np.float64]]) -> NDArray[np.float64]:
        values = np.asarray(statistic(trains), dtype=np.float64)
        return values.reshape(len(trains), -1) if trains else values.reshape(0, 1)

    observed = evaluate(list(spike_times))
    null = np.empty((n_shuffles, *observed.shape), dtype=np.float64)
    # Observed warnings reach the caller once. Shifted data-quality heuristics
    # describe the null, not the supplied recording, and should not repeat.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k in range(n_shuffles):
            null[k] = evaluate([next(stream) for stream in streams])
    return observed, null


def shuffle_information_verdict(
    significance: Callable[[], ShuffleTestResult],
    rate_map: Callable[[], Any],
    *,
    alpha: float | None,
    min_info: float | None,
    default_min_info: float,
    kind: str,
) -> bool:
    """Shuffle verdict of an information-tuned cell type for one unit.

    The unit is a ``kind`` cell when its circular-shift information test is
    significant (``p < alpha``) and the bias-corrected information of its rate
    map is at or above ``min_info``.

    Parameters
    ----------
    significance : callable
        Runs the unit's circular-shift test and returns its result.
    rate_map : callable
        Computes the unit's rate map. It is called only for a significant
        unit, with warnings suppressed: the test already showed the warnings
        of this same map.
    alpha : float or None
        Significance level; ``None`` uses ``_SHUFFLE_DEFAULTS["alpha"]``.
    min_info : float or None
        Cutoff on bias-corrected information (bits/spike); ``None`` uses
        ``default_min_info``.
    default_min_info : float
        The cell type's default cutoff.
    kind : str
        Cell type named in the unknown-bias warning, such as ``"place"``.

    Returns
    -------
    bool
        Whether the unit is a ``kind`` cell.
    """
    from neurospatial.encoding._metrics import _beyond_bias

    level = _SHUFFLE_DEFAULTS["alpha"] if alpha is None else alpha
    if not significance().p_value < level:
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = rate_map()
    cutoff = default_min_info if min_info is None else min_info
    return _beyond_bias(result, cutoff, kind=kind)


def shuffle_pvalues(
    observed: NDArray[np.float64], null: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Upper-tail Monte Carlo p-values over the defined (non-NaN) null scores.

    The p-value is NaN when the observed score is not finite or no null score
    is defined: there is nothing to rank against.
    """
    p_values: NDArray[np.float64] = _shuffle_p_values(observed, null, "greater")
    return p_values


def to_shuffle_results(
    observed: NDArray[np.float64],
    null: NDArray[np.float64],
    unit_ids: NDArray[Any],
) -> dict[Hashable, ShuffleTestResult]:
    """Return independent per-unit score arrays in caller label order.

    Warns, naming the units, when undefined (NaN) null scores were left out
    of a unit's p-value; units whose observed score is undefined are not named.
    """
    p = shuffle_pvalues(observed, null)
    n_dropped = np.sum(np.isnan(null), axis=0)[:, 0]
    affected = np.flatnonzero((n_dropped > 0) & np.isfinite(observed[:, 0]))
    if affected.size:
        details = ", ".join(
            f"unit {plain_value(unit_ids[i])!r}: {int(n_dropped[i])}" for i in affected
        )
        warnings.warn(
            f"Dropped {int(n_dropped[affected].sum())} undefined (NaN) null "
            f"score(s) out of {null.shape[0]} per unit before computing p-values "
            f"({details}). Each p-value uses only its unit's defined null scores; "
            "a unit with none gets p_value=NaN. Shifted spike trains that leave no spikes in "
            "the analyzed bins give undefined scores; more spikes or a longer "
            "recording reduce them.",
            UserWarning,
            stacklevel=_external_stacklevel(),
        )
    result = {
        plain_value(label): shuffle_test_result(
            float(observed[i, 0]), null[:, i, 0], float(p[i, 0])
        )
        for i, label in enumerate(unit_ids)
    }
    assert len(result) == len(unit_ids)
    return result


def shuffle_test_result(
    observed: float, null_scores: NDArray[np.float64], p_value: float
) -> ShuffleTestResult:
    """One unit's upper-tail circular-shift test from its precomputed p-value."""
    return ShuffleTestResult(
        observed_score=observed,
        null_scores=null_scores,
        p_value=p_value,
        z_score=compute_shuffle_zscore(observed, null_scores),
        shuffle_type="circular_time_shift",
        n_shuffles=len(null_scores),
    )


def check_criterion(criterion: str, allowed: tuple[str, ...], *, call: str) -> None:
    if criterion not in allowed:
        choices = ", ".join(repr(value) for value in allowed)
        suffix = (
            " To detect a field rather than classify a cell, use has_place_field()."
            if call.endswith("is_place_cell")
            else ""
        )
        raise ValueError(
            _format_error(
                f"{call} got criterion={criterion!r}; expected one of {choices}.",
                why="Why: the criterion determines which statistic or null test supplies the verdict",
                fix=f"pass criterion={allowed[0]!r} or criterion={allowed[-1]!r}.{suffix}",
            )
        )


def check_mode_keywords(
    criterion: str,
    *,
    threshold: dict[str, Any],
    shuffle: dict[str, Any],
    call: str,
) -> None:
    wrong = threshold if criterion == "shuffle" else shuffle
    offending = {name: value for name, value in wrong.items() if value is not None}
    if offending:
        values = ", ".join(f"{name}={value!r}" for name, value in offending.items())
        alternate = "shuffle" if criterion != "shuffle" else "threshold"
        raise ValueError(
            _format_error(
                f"{call}(criterion={criterion!r}) got {values}, which apply only with criterion={alternate!r}.",
                why="Why: those keywords would otherwise be silently ignored by the selected criterion",
                fix=f"drop those keywords, or pass criterion={alternate!r}",
            )
        )
