"""Sampling of observed overlays without crossing missing tracking or breaks."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from neurospatial._observation import (
    ObservationSupport,
    resolve_sample_support,
    validate_break_before,
    validate_max_gap,
)


def observation_support(
    times: NDArray[np.float64],
    values: NDArray[np.float64],
    *,
    max_gap: float | None = 0.5,
    break_before: NDArray[np.bool_] | None = None,
) -> ObservationSupport:
    """Resolve the same observed intervals used by trajectory analyses."""
    validate_max_gap(max_gap)
    breaks = validate_break_before(break_before, len(times), call="overlay")
    valid = np.asarray(np.isfinite(values.reshape(len(times), -1)).all(axis=1))
    return resolve_sample_support(
        times,
        valid,
        max_gap=max_gap,
        epochs=None,
        break_before=breaks,
        result="overlay interpolation is unavailable between samples",
    )


def sample_at_times(
    times: NDArray[np.float64],
    values: NDArray[np.float64],
    query_times: NDArray[np.float64],
    *,
    interp: Literal["linear", "nearest"] = "linear",
    sample_valid: NDArray[np.bool_] | None = None,
    interval_valid: NDArray[np.bool_] | None = None,
    circular: bool = False,
) -> NDArray[np.float64]:
    """Sample observed values; exact samples survive even beside a break.

    Between samples both interpolation methods require a valid interval. Nearest
    ties select the earlier observation. Circular interpolation takes the short
    path on the unit circle. Queries outside the observed clock return NaN.
    Inputs are validated by the overlay conversion pipeline.
    """
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    query_times = np.atleast_1d(np.asarray(query_times, dtype=np.float64))
    if interp not in ("linear", "nearest"):
        raise ValueError("interp must be 'linear' or 'nearest'.")
    result = np.full((len(query_times), *values.shape[1:]), np.nan)
    if not len(times):
        return result
    finite = np.asarray(np.isfinite(values.reshape(len(times), -1)).all(axis=1))
    sample_valid = finite if sample_valid is None else sample_valid & finite
    if interval_valid is None:
        interval_valid = sample_valid[:-1] & sample_valid[1:]
    right = np.searchsorted(times, query_times, side="left")
    exact = (right < len(times)) & (
        times[np.minimum(right, len(times) - 1)] == query_times
    )
    exact &= sample_valid[np.minimum(right, len(times) - 1)]
    result[exact] = values[right[exact]]
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
