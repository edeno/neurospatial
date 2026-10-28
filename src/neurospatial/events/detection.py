"""
Spatial annotation of event tables.

- add_positions: Add x, y (and z) columns to events by interpolation
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurospatial._intervals import (
    DEFAULT_MAX_GAP,
    as_intervals,
    observed_interval_mask,
    recording_index,
    run_time_bounds,
)
from neurospatial._observation import validate_break_before
from neurospatial.events._core import _validate_events_dataframe

if TYPE_CHECKING:
    import pandas as pd


def add_positions(
    events: pd.DataFrame,
    *,
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    timestamp_column: str = "timestamp",
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
) -> pd.DataFrame:
    """
    Add spatial coordinates to events by interpolating from trajectory.

    Interpolates positions at each event's timestamp and adds x, y (and z
    for 3D) columns to the events DataFrame. The original DataFrame is not
    modified; a new DataFrame is returned.

    Parameters
    ----------
    events : pd.DataFrame
        Events DataFrame with a timestamp column.
    times : NDArray[np.float64], shape (n_samples,), keyword-only
        Timestamps corresponding to each position sample, in seconds.
    positions : NDArray[np.float64], shape (n_samples, n_dims), keyword-only
        Position trajectory. Shape is (n_samples, 1) for 1D, (n_samples, 2)
        for 2D, or (n_samples, 3) for 3D trajectories. ``times`` and
        ``positions`` are keyword-only because a bare 1-D trajectory makes both
        1-D and a positional swap would be shape-indistinguishable and silently
        mis-interpolate.
    timestamp_column : str, default="timestamp"
        Name of the column in events containing timestamps.
    max_gap : float or None, default=1.0
        Longest stretch of missing tracking, in seconds, to bridge: the time
        between consecutive valid samples (a dropped frame and a NaN row are
        the same missing sample). Longer gaps are excluded as tracking losses;
        ``None`` bridges every gap. The default is calibrated for typical video
        and DeepLabCut tracking; choose a value longer than routine dropped or
        low-confidence frames and shorter than real tracking losses or pickups.
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``times``). The interval between two consecutive samples
        counts only if it lies entirely inside one window. ``None`` (default)
        means unrestricted.
    break_before : ArrayLike of bool, shape (n_samples,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored.

    Returns
    -------
    pd.DataFrame
        Copy of events with added columns:
        - "x": x-coordinate (always added)
        - "y": y-coordinate (for 2D and 3D)
        - "z": z-coordinate (for 3D only)

    Raises
    ------
    TypeError
        If events is not a pandas DataFrame.
    ValueError
        If timestamp_column is not in events or is not numeric.
        If times is not 1-D, or positions has more than 2 dimensions or more
        than 3 columns.
        If times and positions have different lengths.
        If the trajectory has fewer than 2 samples.
        If trajectory ``times`` repeat a value or contain non-finite values.
        If ``break_before`` is not a Boolean array with one entry per sample.

    Notes
    -----
    Each run of samples with gaps no longer than ``max_gap`` (inside ``epochs``,
    not crossing a ``break_before``) is analyzed as a separate recording;
    no position is interpolated across a pause.

    Missing positions (NaN rows) are skipped as if their rows were deleted,
    so an event in a bridged gap is interpolated between the valid samples
    on either side. Positions are available only within each run's closed
    sample span ``[times[first], times[last]]``. Events exactly at either
    endpoint get that observed sample's position. Events strictly inside a
    pause, at isolated samples or outside all observed spans have NaN
    coordinates.
    Positions are never extrapolated outside the tracked span.

    This function only adds coordinate columns (x, y, z). It does not add
    derived columns like ``bin_index`` or ``region``; map the added coordinates
    to bins yourself with ``env.bin_at(events[["x", "y"]].to_numpy())`` for
    spatial analysis.

    Interpolation uses linear interpolation between trajectory samples.
    Events before or after the trajectory have NaN coordinates.

    Events with NaN timestamps will have NaN positions.

    If trajectory times are unsorted, they will be sorted internally. A
    ``break_before`` refers to the input order: after sorting, samples that
    came from different recordings are never interpolated between.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from neurospatial.events import add_positions

    Add positions to reward events:

    >>> rewards = pd.DataFrame({"timestamp": [1.5, 3.5], "size": [1, 2]})
    >>> times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    >>> positions = np.array([[0, 0], [2, 2], [4, 4], [6, 6], [8, 8]])
    >>> result = add_positions(rewards, times=times, positions=positions, max_gap=None)
    >>> result[["x", "y"]].values
    array([[3., 3.],
           [7., 7.]])
    """
    _validate_events_dataframe(
        events, timestamp_column=timestamp_column, context="add_positions"
    )

    # Convert to numpy arrays
    positions = np.asarray(positions, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)

    # times and positions are keyword-only (see the signature): a bare 1-D
    # trajectory makes both arrays 1-D, so a positional swap would be shape-
    # indistinguishable and silently mis-interpolate. Keyword-only removes that
    # ambiguity. This remaining guard catches a mislabeled 2-D array handed to
    # `times=` (still a swap, just via keywords) before the length check, which a
    # swap passes since both arrays share n_samples.
    if times.ndim != 1:
        raise ValueError(
            f"times must be a 1-D array of timestamps, got shape {times.shape}. "
            "Did you swap the `times=` and `positions=` arguments?"
        )
    if positions.ndim > 2:
        raise ValueError(
            f"positions must be 1-D (n_samples,) or 2-D (n_samples, n_dims), got "
            f"shape {positions.shape}."
        )

    # Validate matching lengths
    if len(times) != len(positions):
        raise ValueError(
            f"times and positions must have same length, got {len(times)} and {len(positions)}.\n"
            "  WHY: Each position sample must have a corresponding timestamp.\n"
            "  Fix: Ensure positions.shape[0] == times.shape[0]."
        )
    breaks = validate_break_before(break_before, len(times), call="add_positions")

    # Ensure positions is 2D
    if positions.ndim == 1:
        positions = positions.reshape(-1, 1)

    n_dims = positions.shape[1]
    if not 1 <= n_dims <= 3:
        raise ValueError(
            f"positions has {n_dims} columns; add_positions writes x, y and z, "
            "so it takes 1 to 3 position dimensions.\n"
            "  Fix: pass positions of shape (n_samples, 1), (n_samples, 2) or "
            "(n_samples, 3); interpolate other per-sample signals with "
            "align_continuous_to_events."
        )

    # Handle empty events FIRST: with no events to position there is nothing to
    # interpolate, so we return early without validating the trajectory. This
    # lets callers pass a 1-sample (or empty) trajectory alongside an empty
    # event table without tripping the >= 2 sample / finite / non-zero-span
    # guards below, which only matter when interpolation actually happens.
    if len(events) == 0:
        as_intervals(epochs, name="epochs")
        result = events.copy()
        result["x"] = np.array([], dtype=np.float64)
        if n_dims >= 2:
            result["y"] = np.array([], dtype=np.float64)
        if n_dims >= 3:
            result["z"] = np.array([], dtype=np.float64)
        return result

    from neurospatial._validation import validate_finite

    # Trajectory timestamps must be finite to define the interpolant; a NaN/Inf
    # time silently maps every event to NaN/Inf. (A NaN event timestamp lies
    # in no tracked run below, so its position is NaN.)
    times = validate_finite(times, name="times")

    # A linear interpolant needs at least two samples spanning a non-zero time
    # range. A single sample, or all-identical times, leaves interp1d undefined
    # and would return all-NaN/Inf positions silently.
    if len(times) < 2:
        raise ValueError(
            f"add_positions needs at least 2 trajectory samples to "
            f"interpolate, got {len(times)}.\n"
            "  WHY: linear interpolation is undefined for a single sample and "
            "would return NaN for every event position.\n"
            "  Fix: pass a trajectory with >= 2 samples spanning the event times."
        )
    # Sort trajectory by time if needed
    sort_idx = np.argsort(times, kind="stable")
    sorted_times = times[sort_idx]
    repeated = np.flatnonzero(np.diff(sorted_times) == 0)
    if repeated.size:
        raise ValueError(
            f"add_positions trajectory has {repeated.size} repeated sample "
            f"time(s), first at t={sorted_times[repeated[0]]:g} s.\n"
            "  WHY: two positions at one instant leave the interpolated position "
            "undefined; which one wins would depend on sort order.\n"
            "  Fix: drop or average duplicate samples first, e.g. keep the first "
            "with np.unique(times, return_index=True)."
        )

    event_times = events[timestamp_column].to_numpy(dtype=np.float64)
    sorted_positions = positions[sort_idx]
    sorted_breaks = None
    if breaks is not None:
        # A break separates recordings in the input order. After sorting, a
        # break falls wherever neighbouring samples come from different ones.
        recording = recording_index(breaks)[sort_idx]
        sorted_breaks = np.r_[False, recording[1:] != recording[:-1]]

    # Interpolate positions at event times between valid samples, skipping
    # missing (NaN) rows as if deleted; the run bounds below keep it inside
    # the tracked span.
    valid = np.asarray(np.isfinite(sorted_positions).all(axis=1), dtype=bool)
    interpolated = np.full((len(event_times), n_dims), np.nan, dtype=np.float64)
    if valid.any():
        for dim in range(n_dims):
            interpolated[:, dim] = np.interp(
                event_times, sorted_times[valid], sorted_positions[valid, dim]
            )

    runs = run_time_bounds(
        sorted_times,
        observed_interval_mask(
            sorted_times,
            max_gap=max_gap,
            epochs=epochs,
            break_before=sorted_breaks,
            sample_valid=valid,
        ),
    )
    observed = np.zeros(len(event_times), dtype=bool)
    if len(runs):
        run_index = np.searchsorted(runs[:, 0], event_times, side="right") - 1
        observed = (run_index >= 0) & (event_times <= runs[np.maximum(run_index, 0), 1])
    interpolated[~observed, :] = np.nan

    # Create result DataFrame (copy to preserve original)
    result = events.copy()

    # Add coordinate columns
    column_names = ["x", "y", "z"][:n_dims]
    for i, col_name in enumerate(column_names):
        result[col_name] = interpolated[:, i]

    return result
