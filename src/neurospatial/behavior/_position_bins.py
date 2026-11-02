"""One check of the per-sample ``position_bins`` behavior functions take."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurospatial._exceptions import _format_error
from neurospatial._intervals import (
    EMPTY_RESULT,
    start_allocation,
    warn_excluded_intervals,
)
from neurospatial._stacklevel import _external_stacklevel

if TYPE_CHECKING:
    from neurospatial.environment import Environment


def validated_position_bins(
    position_bins: ArrayLike,
    env: Environment | None = None,
    *,
    call: str,
    times: ArrayLike | None = None,
    n_bins: int | None = None,
) -> NDArray[np.int64]:
    """Check per-sample bin indices: integers in ``[0, n_bins)`` or ``-1``.

    ``-1`` is ``env.bin_at``'s marker for a position outside the environment
    (a missing sample). Any other negative index would silently count from
    the end of the bin arrays, an index at or past ``n_bins`` names no bin,
    and a float bin would be truncated to an unrelated bin, so all are
    refused.

    Parameters
    ----------
    position_bins : array-like of int, shape (n_samples,)
        Bin index at each sample.
    env : Environment, optional
        Environment the bins belong to. Give ``env`` or ``n_bins``.
    call : str
        Public function name, used in messages.
    times : array-like, shape (n_samples,), optional
        Timestamps of the same samples; when given, both must be 1-D with
        one timestamp per bin. Only their shape is checked here.
    n_bins : int, optional
        Number of bins, for a function without an environment (such as one
        taking a per-bin label array).

    Returns
    -------
    ndarray of int64, shape (n_samples,)

    Raises
    ------
    ValueError
        If ``position_bins`` is not 1-D, or does not match ``times``.
    TypeError
        If ``position_bins`` holds booleans, floats or non-numbers.
    BinIndexOutOfRangeError
        If a bin index is outside ``[0, n_bins)`` and is not ``-1``.
    """
    from neurospatial.environment import _validate_target_bins
    from neurospatial.geometry import CartesianSystem

    shape = np.shape(position_bins)
    if times is not None:
        times_shape = np.shape(times)
        if len(shape) != 1 or len(times_shape) != 1 or shape != times_shape:
            raise ValueError(
                _format_error(
                    f"{call}() needs position_bins and times as 1-D arrays of "
                    f"the same length; got shapes {shape} and {times_shape}.",
                    why="each sample's bin is matched to its timestamp.",
                    fix="pass env.bin_at(positions) for the same samples as times.",
                )
            )
    elif len(shape) != 1:
        raise ValueError(
            _format_error(
                f"{call}() got position_bins with shape {shape}.",
                why="position_bins holds one bin index per sample.",
                fix="pass a 1-D integer array, for example env.bin_at(positions).",
            )
        )
    if env is not None:
        n_bins, coordinates = env.n_bins, env.coordinate_system
    else:
        assert n_bins is not None, "give env or n_bins"
        coordinates = CartesianSystem()
    bins = _validate_target_bins(
        position_bins,
        n_bins=n_bins,
        coordinates=coordinates,
        operation=f"{call}()",
        argument="position_bins",
        allow_missing=True,
    )
    return bins.astype(np.int64, copy=False)


def sample_bins(
    env: Environment, times: ArrayLike, positions: ArrayLike, *, call: str
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Validated ``times`` and each position's bin, ``-1`` outside or missing.

    The behavior detectors take ``(env, times, positions)`` and bin the
    positions here, so users never handle bin indices. A NaN position (lost
    tracking) or one outside the environment is bin ``-1``, an unobserved
    sample.

    Parameters
    ----------
    env : Environment
        Allocentric environment the positions were recorded in.
    times : array-like, shape (n_samples,)
        Finite, strictly increasing timestamps in seconds.
    positions : array-like, shape (n_samples, n_dims)
        Positions in the environment's coordinates; NaN where missing.
    call : str
        Public function name, used in messages.

    Returns
    -------
    times : ndarray of float64, shape (n_samples,)
    bins : ndarray of int64, shape (n_samples,)

    Raises
    ------
    ValueError
        If times are invalid, the arrays do not match, or the positions do
        not have the environment's number of coordinates.

    Warns
    -----
    UserWarning
        If more than half of the positions with finite coordinates are
        outside the environment (usually a unit or coordinate-frame
        mismatch).
    """
    from neurospatial._validation import validate_times_positions

    t, p = validate_times_positions(times, positions, call=call)
    if p.ndim == 1:  # a 1-D track: one coordinate per sample
        p = p[:, np.newaxis]
    if p.shape[1] != env.n_dims:
        raise ValueError(
            _format_error(
                f"{call}(): positions have {p.shape[1]} coordinate(s) per sample; "
                f"the environment has {env.n_dims}.",
                why="positions are located in the environment's bins.",
                fix="pass positions in the environment's coordinates, shape "
                "(n_samples, n_dims).",
            )
        )
    bins = np.asarray(env.bin_at(p), dtype=np.int64)
    finite = np.asarray(np.isfinite(p).all(axis=1), dtype=bool)
    _warn_if_mostly_outside(bins, finite, call=call)
    return t, bins


def _warn_if_mostly_outside(
    bins: NDArray[np.int64], finite: NDArray[np.bool_], *, call: str
) -> None:
    """Warn once when more than half of the finite positions are outside.

    NaN positions are missing, not outside, so they are left out of the
    count. A majority outside usually means positions in other units (mm
    against an environment in cm) or another coordinate frame, which leaves
    a detector running on the few samples that happen to land inside.
    """
    n_finite = int(np.count_nonzero(finite))
    n_outside = int(np.count_nonzero(finite & (bins < 0)))
    if n_finite == 0 or 2 * n_outside <= n_finite:
        return
    warnings.warn(
        f"{call}(): {n_outside} of {n_finite} positions with finite coordinates "
        f"({n_outside / n_finite:.0%}) are outside the environment, so the "
        "result rests on the few samples inside and is probably misleading. "
        "Fix: check that positions and the environment use the same units "
        "(for example mm against cm) and coordinate frame.",
        UserWarning,
        stacklevel=_external_stacklevel(),
    )


def bridge_missing_positions(
    times: NDArray[np.float64],
    bins: NDArray[np.int64],
    positions: ArrayLike,
    *,
    max_gap: float | None,
    epochs: NDArray[np.float64] | None,
    break_before: NDArray[np.bool_] | None,
    result: str = EMPTY_RESULT,
) -> tuple[
    NDArray[np.float64], NDArray[np.int64], NDArray[np.bool_] | None, NDArray[np.intp]
]:
    """Drop NaN position rows as deleted rows, warning when nothing is observed.

    A NaN position is a missing sample (lost tracking), the same as a
    deleted row: the detector runs on the rows left, so the samples around
    it are joined when no more than ``max_gap`` apart. A position outside
    the environment (bin -1 at a finite position) stays a row, observed but
    not located, so it still ends observation.

    The warning judges the whole recording with the missing samples bridged
    (as ``env.occupancy`` does), so epochs on another clock, all-NaN
    positions or gaps longer than ``max_gap`` warn even when no row is left.

    Parameters
    ----------
    times : ndarray of float64, shape (n_samples,)
        Validated timestamps in seconds.
    bins : ndarray of int64, shape (n_samples,)
        Each sample's bin (:func:`sample_bins`), -1 outside or missing.
    positions : array-like, shape (n_samples, n_dims) or (n_samples,)
        The positions the bins came from; NaN where missing.
    max_gap, epochs, break_before
        The detector's validated gates; ``epochs`` as resolved windows.
    result : str
        What an all-excluded recording leaves, for the warning.

    Returns
    -------
    times : ndarray of float64, shape (n_kept,)
    bins : ndarray of int64, shape (n_kept,)
    break_before : ndarray of bool, shape (n_kept,), or None
        A break declared at a dropped row moves to the next kept row.
    kept : ndarray of intp, shape (n_kept,)
        Indices of the kept rows in the original arrays.
    """
    samples = np.asarray(positions, dtype=np.float64)
    tracked: NDArray[np.bool_] = (
        np.asarray(np.isfinite(samples.reshape(times.size, -1)).all(axis=1))
        if times.size
        else np.zeros(0, dtype=bool)
    )
    gates = start_allocation(
        times, bins, tracked, max_gap=max_gap, epochs=epochs, break_before=break_before
    ).gates
    warn_excluded_intervals(gates, result=result)
    kept = np.flatnonzero(tracked)
    if break_before is not None:
        declared = np.cumsum(np.asarray(break_before, dtype=np.intp))
        moved = np.zeros(kept.size, dtype=bool)
        moved[1:] = declared[kept[1:]] > declared[kept[:-1]]
        break_before = moved
    return times[kept], bins[kept], break_before, kept
