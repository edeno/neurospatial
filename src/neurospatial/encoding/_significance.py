"""Shared circular-shift engine and cell-predicate criterion validation."""

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Callable, Hashable
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurospatial._exceptions import _format_error
from neurospatial.stats.shuffle import (
    ShuffleTestResult,
    _validate_circular_shift_settings,
    shuffle_spike_times_circular,
)

_SHUFFLE_DEFAULTS = MappingProxyType(
    {"n_shuffles": 1000, "min_shift": 20.0, "alpha": 0.05}
)


def _stream_key(label: Hashable) -> int:
    """Stable key shared by Python and NumPy scalar representations."""
    value = np.asarray(label).item()
    return int.from_bytes(hashlib.sha256(repr(value).encode()).digest()[:8], "little")


def _entropy(rng: np.random.Generator | int | None) -> int:
    if isinstance(rng, np.random.Generator):
        return int(rng.integers(2**63))
    if rng is None:
        return int(np.random.SeedSequence().entropy)
    return int(rng)


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
            rng=np.random.default_rng(
                np.random.SeedSequence(entropy, spawn_key=(_stream_key(uid),))
            ),
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


def shuffle_pvalues(
    observed: NDArray[np.float64], null: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Correct upper-tail Monte Carlo ranks over finite null scores."""
    finite = np.isfinite(null)
    exceed = np.sum(finite & (null >= observed[None]), axis=0)
    p = (1.0 + exceed) / (1.0 + np.sum(finite, axis=0))
    return np.where(np.isfinite(observed), p, np.nan)


def to_shuffle_results(
    observed: NDArray[np.float64],
    null: NDArray[np.float64],
    p: NDArray[np.float64],
    unit_ids: NDArray[Any],
    column: int = 0,
) -> dict[Hashable, ShuffleTestResult]:
    """Return independent per-unit score arrays in caller label order."""
    result = {}
    for i, label in enumerate(unit_ids):
        scores = null[:, i, column].copy()
        finite = scores[np.isfinite(scores)]
        std = float(np.std(finite)) if finite.size else float("nan")
        obs = float(observed[i, column])
        z = (
            (obs - float(np.mean(finite))) / std
            if finite.size and std > 0
            else float("nan")
        )
        result[label.item() if isinstance(label, np.generic) else label] = (
            ShuffleTestResult(
                observed_score=obs,
                null_scores=scores,
                p_value=float(p[i, column]),
                z_score=z,
                shuffle_type="circular_time_shift",
                n_shuffles=len(scores),
            )
        )
    assert len(result) == len(unit_ids)
    return result


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
        alternate = (
            "shuffle"
            if criterion != "shuffle"
            else ("spatial_info" if call.endswith("is_place_cell") else "threshold")
        )
        raise ValueError(
            _format_error(
                f"{call}(criterion={criterion!r}) got {values}, which apply only with criterion={alternate!r}.",
                why="Why: those keywords would otherwise be silently ignored by the selected criterion",
                fix=f"drop those keywords, or pass criterion={alternate!r}",
            )
        )
