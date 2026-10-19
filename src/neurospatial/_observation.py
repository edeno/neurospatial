"""Resolve which samples and intervals of a trajectory were actually observed.

Shared by the distance, resampling and turning analyses. A sample is valid when
every coordinate is finite. Interval ``k`` (samples ``k`` and ``k + 1``) is
usable when both samples are valid, the elapsed time passes the ``max_gap`` and
``epochs`` gates, and no recording break is declared before sample ``k + 1``.
Runs are maximal sequences of consecutive usable intervals. Nothing here
depends on an ``Environment``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error
from neurospatial._intervals import observed_interval_mask, run_sample_bounds
from neurospatial._results import ReadOnlyArraysMixin
from neurospatial._validation import (
    validate_strictly_increasing,
    validate_times_positions,
)


@dataclass(frozen=True, repr=False)
class ObservationSupport(ReadOnlyArraysMixin):
    """Valid samples, usable intervals and runs of one trajectory.

    Attributes
    ----------
    sample_valid : ndarray of bool, shape (n_samples,)
        True where every coordinate is finite.
    interval_mask : ndarray of bool, shape (max(n_samples - 1, 0),)
        True for usable intervals.
    run_ids : ndarray of intp, shape (n_samples,)
        Run index ``0..n_runs-1`` in time order; -1 for samples that touch no
        usable interval.
    run_sample_bounds : ndarray of intp, shape (n_runs, 2)
        Inclusive ``(first, last)`` original sample indices of each run.
    run_time_bounds : ndarray of float64, shape (n_runs, 2)
        ``times`` at those indices, in seconds.
    """

    sample_valid: NDArray[np.bool_]
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


def validate_break_before(
    break_before: Any, n_samples: int, *, call: str
) -> NDArray[np.bool_] | None:
    """Return ``break_before`` as a Boolean array, or None when not given."""
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
    return flags


def validate_max_gap(max_gap: float | None) -> float | None:
    """Return ``max_gap`` unchanged if None or a positive finite number."""
    if max_gap is not None and (not np.isfinite(max_gap) or max_gap <= 0):
        raise ValueError(
            "max_gap must be a positive finite number of seconds or None "
            f"(None keeps every interval, however long); got {max_gap!r}."
        )
    return max_gap


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
    n_samples = times.size
    sample_valid: NDArray[np.bool_] = np.asarray(np.isfinite(positions).all(axis=1))
    interval_mask = observed_interval_mask(
        times, max_gap=max_gap, epochs=epochs, result=result
    )
    interval_mask = interval_mask & sample_valid[:-1] & sample_valid[1:]
    if break_before is not None:
        interval_mask = interval_mask & ~break_before[1:]
    bounds = run_sample_bounds(interval_mask)
    run_ids = np.full(n_samples, -1, dtype=np.intp)
    lengths = bounds[:, 1] - bounds[:, 0] + 1
    if lengths.size:
        starts_in_output = np.cumsum(lengths) - lengths
        samples = np.arange(lengths.sum()) + np.repeat(
            bounds[:, 0] - starts_in_output, lengths
        )
        run_ids[samples] = np.repeat(np.arange(lengths.size, dtype=np.intp), lengths)
    return ObservationSupport(
        sample_valid=sample_valid,
        interval_mask=interval_mask,
        run_ids=run_ids,
        run_sample_bounds=bounds,
        run_time_bounds=times[bounds],
    )


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

    Returns
    -------
    times : ndarray, shape (n_samples,)
    positions : ndarray, shape (n_samples, n_dims)
        1-D input is reshaped to ``(n_samples, 1)``.
    support : ObservationSupport
    """
    t, p = validate_times_positions(times, positions, call=call)
    validate_strictly_increasing(t, call=call)
    if p.ndim == 1:
        p = p[:, None]
    validate_max_gap(max_gap)
    breaks = validate_break_before(break_before, t.size, call=call)
    support = resolve_observation_support(
        t, p, max_gap=max_gap, epochs=epochs, break_before=breaks, result=result
    )
    return t, p, support
