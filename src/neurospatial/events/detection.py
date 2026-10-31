"""
Spatial annotation of event tables.

- add_positions: Add x, y (and z) columns to events by interpolation
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurospatial._intervals import DEFAULT_MAX_GAP, as_intervals
from neurospatial._observation import sample_feature, validated_observations
from neurospatial._pynapple import adopts_time_support
from neurospatial.events._core import _event_times

if TYPE_CHECKING:
    import pandas as pd


@adopts_time_support
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
        When ``positions`` is a pynapple ``TsdFrame`` or ``Tsd``, ``None`` uses its
        ``time_support``.
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
        If timestamp_column is not in events, is not numeric, or holds a NaN
        or infinite time.
        If times is not 1-D, or positions has more than 2 dimensions or more
        than 3 columns.
        If times and positions have different lengths, or the trajectory is
        empty.
        If trajectory ``times`` are not strictly increasing or contain
        non-finite values.
        If ``break_before`` is not a Boolean array with one entry per sample.

    Notes
    -----
    Positions are sampled exactly as ``align_continuous_to_events`` samples a
    signal. An event exactly at a sample time gets that sample's position
    when every coordinate is finite there (and, with ``epochs``, the time
    lies in one half-open epoch). Any other event is linearly interpolated
    inside a usable interval: one whose gap is no longer than ``max_gap``,
    lies inside one epoch and crosses no ``break_before``. Missing positions
    (rows with a NaN coordinate) are skipped as if their rows were deleted,
    so an event in a bridged gap is interpolated between the valid samples
    on either side. Events inside a longer pause or outside the tracked span
    get NaN coordinates; positions are never extrapolated.

    This function only adds coordinate columns (x, y, z). It does not add
    derived columns like ``bin_index`` or ``region``; map the added coordinates
    to bins yourself with ``env.bin_at(events[["x", "y"]].to_numpy())`` for
    spatial analysis.

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
    call = "add_positions"
    event_times = _event_times(events, timestamp_column, name="events", call=call)
    normalized_epochs = as_intervals(epochs, name="epochs")
    t, xyz, support = validated_observations(
        times,
        positions,
        call=call,
        max_gap=max_gap,
        epochs=normalized_epochs,
        break_before=break_before,
        result="events between samples get NaN positions",
    )
    if not 1 <= xyz.shape[1] <= 3:
        raise ValueError(
            f"positions has {xyz.shape[1]} columns; add_positions writes x, y and "
            "z, so it takes 1 to 3 position dimensions.\n"
            "  Fix: pass positions of shape (n_samples, 1), (n_samples, 2) or "
            "(n_samples, 3); interpolate other per-sample signals with "
            "align_continuous_to_events."
        )

    result = events.copy()
    for name, coordinate in zip("xyz", xyz.T, strict=False):
        result[name], _ = sample_feature(
            t,
            coordinate,
            support,
            event_times,
            interpolation="linear",
            kind="linear",
            epochs=normalized_epochs,
        )
    return result
