"""Binning layer for directional encoding.

This module converts spike trains and head direction data into discrete spike
counts and occupancy arrays for head direction cell analysis.

``prepare_directional_binning`` handles:
1. Circular binning of head directions into angular bins over [-π, π)
2. The interval support shared by occupancy and spike counts

Its ``FrameBinning.count`` then counts each neuron's spikes from the most
recent frame and accumulates occupancy from the same intervals.

Output shapes:
- Spike counts: (n_neurons, n_bins)
- Occupancy: (n_bins,) - always shared across neurons
- Bin centers: (n_bins,) - angles in radians [-π, π)

The binning layer is separated from smoothing so occupancy and bin centers
are computed once and shared across neurons.

Notes
-----
Unlike spatial binning, directional binning does not require an Environment.
Head direction is a 1D circular variable independent of spatial position.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from neurospatial._angles import to_radians
from neurospatial._intervals import DEFAULT_MAX_GAP
from neurospatial.encoding._spatial_counts import FrameBinning, prepare_frame_binning

__all__ = [
    "directional_frame_bins",
    "prepare_directional_binning",
]


def _validate_directional_samples(
    times: NDArray[np.float64], headings: NDArray[np.float64]
) -> None:
    """Validate aligned directional samples and positive sampling intervals."""
    # Validate inputs
    if len(headings) != len(times):
        raise ValueError(
            f"headings and times must have the same length. "
            f"Got headings: {len(headings)}, times: {len(times)}.\n"
            f"Fix: Ensure both arrays represent the same time series."
        )

    if len(times) < 3:
        raise ValueError(
            f"Need at least 3 samples to compute occupancy. "
            f"Got {len(times)} samples.\n"
            f"Fix: Provide more data points."
        )

    # Check strict monotonicity (no duplicates, no decreasing)
    time_diffs = np.diff(times)
    if np.any(time_diffs <= 0):
        n_problems = np.sum(time_diffs <= 0)
        raise ValueError(
            f"times must be strictly monotonically increasing (no duplicates). "
            f"Found {n_problems} non-increasing time steps.\n"
            f"Fix: Remove duplicate timestamps or check for timestamp errors."
        )


DEFAULT_HEADING_BIN_SIZE_DEG = 6.0


def directional_frame_bins(
    headings: NDArray[np.float64],
    bin_size: float | None,
    *,
    angle_unit: Literal["rad", "deg"] = "rad",
) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
    """Map finite headings to circular bins; non-finite headings map to -1.

    Any finite angle is accepted and wrapped to [-π, π); bins tile [-π, π)
    and their centers are returned in radians. A 0 to 2π heading and the
    same heading minus 2π fall in the same bin. ``bin_size=None`` means 6°
    expressed in ``angle_unit``.
    """
    if bin_size is None:
        bin_size = (
            DEFAULT_HEADING_BIN_SIZE_DEG
            if angle_unit == "deg"
            else float(np.radians(DEFAULT_HEADING_BIN_SIZE_DEG))
        )

    if bin_size <= 0:
        raise ValueError(
            f"bin_size must be positive, got {bin_size}.\n"
            f"Fix: Use a positive bin size (e.g., np.pi/30 radians or 6 degrees)."
        )

    headings_rad = to_radians(headings, angle_unit)
    bin_size_rad = float(to_radians(bin_size, angle_unit))

    n_bins = int(np.round(2 * np.pi / bin_size_rad))
    if n_bins < 1:
        raise ValueError(
            f"bin_size is too large: {bin_size} ({angle_unit}). "
            f"Results in {n_bins} bins (need at least 1).\n"
            f"Fix: Use a smaller bin_size (max ~2π radians or 360 degrees)."
        )

    finite = np.isfinite(headings_rad)
    wrapped = (headings_rad[finite] + np.pi) % (2 * np.pi) - np.pi
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    frame_bins = np.full(headings.shape, -1, dtype=np.intp)
    bins = np.digitize(wrapped, bin_edges) - 1
    bins[bins >= n_bins] = 0
    frame_bins[finite] = bins
    return frame_bins, bin_centers


def prepare_directional_binning(
    times: NDArray[np.float64],
    headings: NDArray[np.float64],
    bin_size: float | None,
    *,
    angle_unit: Literal["rad", "deg"] = "rad",
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: NDArray[np.float64] | None = None,
    spike_window: NDArray[np.float64] | None = None,
    break_before: NDArray[np.bool_] | None = None,
) -> tuple[FrameBinning, NDArray[np.float64]]:
    """Prepare geometry and exact temporal support before counting spikes.

    ``break_before`` is already validated; the interval ending at each
    declared recording break is dropped from counts and occupancy.
    """
    times = np.asarray(times, dtype=np.float64).ravel()
    headings = np.asarray(headings, dtype=np.float64).ravel()
    _validate_directional_samples(times, headings)
    frame_bins, bin_centers = directional_frame_bins(
        headings, bin_size, angle_unit=angle_unit
    )
    n_bins = len(bin_centers)
    prepared = prepare_frame_binning(
        times,
        frame_bins,
        n_bins,
        max_gap=max_gap,
        epochs=epochs,
        spike_window=spike_window,
        break_before=break_before,
        samples="headings",
        sample_valid=np.isfinite(headings),
    )
    return prepared, bin_centers
