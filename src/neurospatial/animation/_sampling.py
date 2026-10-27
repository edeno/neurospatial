"""Sampling of observed overlays without crossing missing tracking or breaks.

An overlay's support is resolved once, when it is converted: a sample is valid
when every coordinate is finite and nothing rejected it, and an interval is
valid when both of its samples are valid and no gap or break lies between
them (:func:`resolve_support`). An overlay's missing rows are bridged as in
trajectory analyses: one inside a gap that passes ``max_gap`` is filled from
the rows that bound the gap (:func:`observed_rows`). Sampling and moving rows onto another clock
then trust those masks (:func:`sample_at_times`, :func:`resample`).
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from neurospatial._intervals import validate_max_gap
from neurospatial._observation import (
    resolve_sample_support,
    validate_break_before,
)


def observed_rows(
    times: NDArray[np.float64],
    values: NDArray[np.float64],
    *,
    max_gap: float | None,
    break_before: NDArray[np.bool_] | None = None,
    interp: Literal["linear", "nearest"],
    circular: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.bool_]]:
    """Resolve an overlay's observed rows the way trajectory analyses do.

    A row with a non-finite value is missing. As in every analysis, a missing
    row is treated like a deleted one: when the gap between the valid rows
    around it passes ``max_gap`` and ``break_before``, it is filled from
    those rows the way the views sample between two rows (linearly in time,
    along the shorter arc for ``circular`` angles, or from the nearer row for
    ``interp="nearest"``), so drawing and sampling bridge the gap exactly.

    Parameters
    ----------
    times : ndarray, shape (n_samples,)
        Times of the rows, in seconds.
    values : ndarray, shape (n_samples,) or (n_samples, n_dims)
        The rows; NaN marks a missing value.
    max_gap : float or None
        Longest gap between valid rows that is bridged.
    break_before : ndarray of bool, shape (n_samples,), or None
        Declared recording breaks.
    interp : {"linear", "nearest"}
        How the views sample between rows.
    circular : bool, default=False
        The rows are angles in radians.

    Returns
    -------
    values : ndarray of float64, same shape as ``values``
        A copy with bridged rows filled.
    sample_valid : ndarray of bool, shape (n_samples,)
        Valid and filled rows.
    interval_valid : ndarray of bool, shape (n_samples - 1,)
        Intervals that may be crossed.
    """
    from neurospatial._intervals import gap_endpoints

    validate_max_gap(max_gap)
    breaks = validate_break_before(break_before, len(times), call="overlay")
    owned = np.array(values, dtype=np.float64, copy=True)
    valid = np.asarray(np.isfinite(owned.reshape(len(times), -1)).all(axis=1))
    support = resolve_sample_support(
        times,
        valid,
        max_gap=max_gap,
        epochs=None,
        break_before=breaks,
        result="overlay interpolation is unavailable between samples",
    )
    bridged = np.array(support.interval_mask, copy=True)
    valid = valid.copy()
    if len(owned) > 2:
        start, end, _ = gap_endpoints(valid)
        inner = np.flatnonzero(~valid[1:-1]) + 1
        filled = inner[bridged[inner - 1] & bridged[inner]]
        a, b = start[filled], end[filled]
        fraction = (times[filled] - times[a]) / (times[b] - times[a])
        if interp == "nearest":
            owned[filled] = owned[np.where(fraction > 0.5, b, a)]
        elif circular:
            cosine = (1 - fraction) * np.cos(owned[a]) + fraction * np.cos(owned[b])
            sine = (1 - fraction) * np.sin(owned[a]) + fraction * np.sin(owned[b])
            owned[filled] = np.arctan2(sine, cosine)
        else:
            weight = fraction.reshape((-1,) + (1,) * (owned.ndim - 1))
            owned[filled] = (1 - weight) * owned[a] + weight * owned[b]
        valid[filled] = True
    return owned, valid, bridged


def resolve_support(
    values: Any,
    sample_valid: NDArray[np.bool_] | None = None,
    interval_valid: NDArray[np.bool_] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.bool_]]:
    """Own observed rows and make their effective support explicit.

    Parameters
    ----------
    values : array-like, shape (n_samples,) or (n_samples, n_dims)
        Observed rows; NaN marks a missing coordinate.
    sample_valid : ndarray of bool, shape (n_samples,), optional
        Samples that may be used. None accepts every finite row.
    interval_valid : ndarray of bool, shape (n_samples - 1,), optional
        Intervals that may be crossed. None accepts every interval between
        valid samples.

    Returns
    -------
    values : ndarray of float64
        A copy of ``values`` with every rejected row NaN, so renderers that
        draw raw rows agree with sampling.
    sample_valid : ndarray of bool, shape (n_samples,)
        Finite rows that ``sample_valid`` allows.
    interval_valid : ndarray of bool, shape (n_samples - 1,)
        Intervals between valid samples that ``interval_valid`` allows. A
        rejected interval leaves its finite endpoints valid.
    """
    owned = np.array(values, dtype=np.float64, copy=True)
    valid = np.asarray(
        np.isfinite(owned.reshape(len(owned), -1)).all(axis=1), dtype=np.bool_
    )
    if sample_valid is not None:
        valid &= sample_valid
    supported = valid[:-1] & valid[1:]
    if interval_valid is not None:
        supported &= interval_valid
    owned[~valid] = np.nan
    return owned, valid, supported


def sample_at_times(
    times: NDArray[np.float64],
    values: NDArray[np.float64],
    query_times: NDArray[np.float64],
    *,
    interp: Literal["linear", "nearest"],
    sample_valid: NDArray[np.bool_],
    interval_valid: NDArray[np.bool_],
    circular: bool = False,
) -> NDArray[np.float64]:
    """Sample observed values; exact samples survive even beside a break.

    Between samples both interpolation methods require a valid interval and
    two valid samples. Nearest ties select the earlier observation. Circular
    interpolation takes the short path on the unit circle. Queries outside
    the observed clock return NaN. The masks describe the rows, as
    :func:`resolve_support` returns them; values are not checked again.
    """
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    query_times = np.atleast_1d(np.asarray(query_times, dtype=np.float64))
    if interp not in ("linear", "nearest"):
        raise ValueError("interp must be 'linear' or 'nearest'.")
    result = np.full((len(query_times), *values.shape[1:]), np.nan)
    if not len(times):
        return result
    right = np.searchsorted(times, query_times, side="left")
    last = np.minimum(right, len(times) - 1)
    exact = (right < len(times)) & (times[last] == query_times) & sample_valid[last]
    exact_rows = np.flatnonzero(exact)
    result[exact_rows] = values[right[exact_rows]]
    inside = (~exact) & (right > 0) & (right < len(times))
    rows = np.flatnonzero(inside)
    r = right[rows]
    keep = interval_valid[r - 1] & sample_valid[r - 1] & sample_valid[r]
    rows, r = rows[keep], r[keep]
    left = r - 1
    fraction = (query_times[rows] - times[left]) / (times[r] - times[left])
    if interp == "nearest":
        result[rows] = values[np.where(fraction > 0.5, r, left)]
    elif circular:
        cosine = (1 - fraction) * np.cos(values[left]) + fraction * np.cos(values[r])
        sine = (1 - fraction) * np.sin(values[left]) + fraction * np.sin(values[r])
        result[rows] = np.arctan2(sine, cosine)
    else:
        weight = fraction.reshape((-1,) + (1,) * (values.ndim - 1))
        result[rows] = (1 - weight) * values[left] + weight * values[r]
    return result


def sample_parts(
    times: NDArray[np.float64],
    parts: list[NDArray[np.float64]],
    sample_valid: NDArray[np.bool_],
    interval_valid: NDArray[np.bool_],
    time: float,
    *,
    interp: Literal["linear", "nearest"],
) -> NDArray[np.float64]:
    """Sample several streams on one clock at one time, each with its support.

    Equal to :func:`sample_at_times` on each part, with one search of the
    clock for all of them.

    Parameters
    ----------
    times : ndarray of float64, shape (n_samples,)
        The parts' shared clock, in seconds.
    parts : list of ndarray of float64, each shape (n_samples, n_dims)
        Each part's rows, as :func:`resolve_support` returns them.
    sample_valid : ndarray of bool, shape (n_samples, n_parts)
        Each part's valid samples.
    interval_valid : ndarray of bool, shape (n_samples - 1, n_parts)
        Each part's valid intervals.
    time : float
        The time to sample, in seconds.
    interp : {"linear", "nearest"}
        How rows are sampled between observations.

    Returns
    -------
    ndarray of float64, shape (n_parts, n_dims)
        NaN for a part without support at ``time``.
    """
    if not parts:
        return np.empty((0, 0))
    n_samples = len(times)
    result = np.full((len(parts), *parts[0].shape[1:]), np.nan)
    if not n_samples:
        return result
    right = int(np.searchsorted(times, time, side="left"))
    exact = np.zeros(len(parts), dtype=bool)
    if right < n_samples and times[right] == time:
        exact = sample_valid[right]
        result[exact] = np.stack([part[right] for part in parts])[exact]
    if 0 < right < n_samples:
        left = right - 1
        keep = ~exact & interval_valid[left] & sample_valid[left] & sample_valid[right]
        if keep.any():
            fraction = (time - times[left]) / (times[right] - times[left])
            if interp == "nearest":
                row = right if fraction > 0.5 else left
                result[keep] = np.stack([part[row] for part in parts])[keep]
            else:
                before = np.stack([part[left] for part in parts])[keep]
                after = np.stack([part[right] for part in parts])[keep]
                result[keep] = (1 - fraction) * before + fraction * after
    return result


def resample(
    times: NDArray[np.float64],
    values: NDArray[np.float64],
    sample_valid: NDArray[np.bool_],
    interval_valid: NDArray[np.bool_],
    clock: NDArray[np.float64],
    *,
    interp: Literal["linear", "nearest"],
    circular: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.bool_]]:
    """Move observed rows and their support onto another clock.

    Parameters
    ----------
    times : ndarray, shape (n_samples,)
        The rows' clock, in seconds.
    values : ndarray, shape (n_samples,) or (n_samples, n_dims)
        Observed rows, with the support ``sample_valid`` and
        ``interval_valid`` from :func:`resolve_support`.
    sample_valid, interval_valid : ndarray of bool
        Shapes (n_samples,) and (n_samples - 1,).
    clock : ndarray, shape (n_clock,)
        Strictly increasing times to move the rows to.
    interp : {"linear", "nearest"}
        How rows are sampled between observations.
    circular : bool, default=False
        Interpolate angles in radians on the unit circle.

    Returns
    -------
    values : ndarray, shape (n_clock, ...)
        The rows sampled at ``clock``; NaN where unsupported.
    sample_valid : ndarray of bool, shape (n_clock,)
        Finite sampled rows.
    interval_valid : ndarray of bool, shape (n_clock - 1,)
        Clock intervals whose samples are valid and that span only supported
        source intervals, so a coarse clock that skips over a gap or break
        still splits there.
    """
    samples = sample_at_times(
        times,
        values,
        clock,
        interp=interp,
        sample_valid=sample_valid,
        interval_valid=interval_valid,
        circular=circular,
    )
    valid = np.asarray(np.isfinite(samples.reshape(len(clock), -1)).all(axis=1))
    left = np.clip(
        np.searchsorted(times, clock[:-1], side="right") - 1, 0, len(times) - 1
    )
    right = np.clip(np.searchsorted(times, clock[1:], side="left"), 0, len(times) - 1)
    excluded = np.r_[0, np.cumsum(~interval_valid)]
    intervals = valid[:-1] & valid[1:] & (excluded[left] == excluded[right])
    return samples, valid, intervals


def sample_observation(
    item: Any,
    values: NDArray[np.float64],
    time: float,
    *,
    name: str | None = None,
    circular: bool = False,
) -> NDArray[np.float64]:
    """Sample one of a prepared overlay's streams at ``time``.

    ``name`` selects the stream's support, as ``item.support(name)`` gives it:
    a bodypart, ``"firing_rates"``, or None for the overlay's positions.
    """
    sample_valid, interval_valid = item.support(name)
    sampled = sample_at_times(
        item.times,
        values,
        np.array([time]),
        interp=item.interp,
        sample_valid=sample_valid,
        interval_valid=interval_valid,
        circular=circular,
    )
    return np.asarray(sampled[0], dtype=np.float64)
