"""
Event detection functions for neurospatial.

This module provides functions to detect events from trajectories and signals:
- extract_region_crossing_events: Detect region entry/exit events
- extract_threshold_crossing_events: Detect signal threshold crossings
- extract_movement_onset_events: Detect movement onset from position

Spatial utilities:
- add_positions: Add x, y columns to events by interpolation
- events_in_region: Filter events to those within a region
- spatial_event_rate: Compute spatial rate map of events
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def add_positions(
    events: pd.DataFrame,
    positions: NDArray[np.float64],
    times: NDArray[np.float64],
    *,
    timestamp_column: str = "timestamp",
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
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Position trajectory. Shape is (n_samples, 1) for 1D, (n_samples, 2)
        for 2D, or (n_samples, 3) for 3D trajectories.
    times : NDArray[np.float64], shape (n_samples,)
        Timestamps corresponding to each position sample.
    timestamp_column : str, default="timestamp"
        Name of the column in events containing timestamps.

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
        If timestamp_column is not in events.
        If times and positions have different lengths.

    Notes
    -----
    This function only adds coordinate columns (x, y, z). It does not add
    derived columns like ``bin_index`` or ``region``. Use ``events_in_region()``
    or ``spatial_event_rate()`` for spatial analysis after adding positions.

    Interpolation uses linear interpolation between trajectory samples.
    Events before or after the trajectory will be extrapolated.

    Events with NaN timestamps will have NaN positions.

    If trajectory times are unsorted, they will be sorted internally.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from neurospatial.events import add_positions

    Add positions to reward events:

    >>> rewards = pd.DataFrame({"timestamp": [1.5, 3.5], "size": [1, 2]})
    >>> times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    >>> positions = np.array([[0, 0], [2, 2], [4, 4], [6, 6], [8, 8]])
    >>> result = add_positions(rewards, positions, times)
    >>> result[["x", "y"]].values
    array([[3., 3.],
           [7., 7.]])
    """
    # Validate events is a DataFrame
    if not isinstance(events, pd.DataFrame):
        raise TypeError(
            f"events must be a pandas DataFrame, got {type(events).__name__}.\n"
            "  WHY: This function operates on DataFrames to preserve metadata.\n"
            "  HOW: Convert your data to a DataFrame before calling."
        )

    # Validate timestamp column exists
    if timestamp_column not in events.columns:
        raise ValueError(
            f"timestamp column '{timestamp_column}' not found in events.\n"
            f"  WHY: Events must have a timestamp column for interpolation.\n"
            f"  HOW: Use timestamp_column parameter to specify the correct name.\n"
            f"  Available columns: {list(events.columns)}"
        )

    # Convert to numpy arrays
    positions = np.asarray(positions, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)

    # Validate matching lengths
    if len(times) != len(positions):
        raise ValueError(
            f"times and positions must have same length, got {len(times)} and {len(positions)}.\n"
            "  WHY: Each position sample must have a corresponding timestamp.\n"
            "  HOW: Ensure positions.shape[0] == times.shape[0]."
        )

    # Ensure positions is 2D
    if positions.ndim == 1:
        positions = positions.reshape(-1, 1)

    n_dims = positions.shape[1]

    # Get event timestamps
    event_times = events[timestamp_column].values.astype(np.float64)

    # Handle empty events
    if len(events) == 0:
        result = events.copy()
        result["x"] = np.array([], dtype=np.float64)
        if n_dims >= 2:
            result["y"] = np.array([], dtype=np.float64)
        if n_dims >= 3:
            result["z"] = np.array([], dtype=np.float64)
        return result

    # Sort trajectory by time if needed
    sort_idx = np.argsort(times)
    sorted_times = times[sort_idx]
    sorted_positions = positions[sort_idx]

    # Interpolate positions at event times
    # Use scipy.interpolate for extrapolation support
    from scipy.interpolate import interp1d

    interpolated = np.empty((len(event_times), n_dims), dtype=np.float64)
    for dim in range(n_dims):
        interp_func = interp1d(
            sorted_times,
            sorted_positions[:, dim],
            kind="linear",
            fill_value="extrapolate",
        )
        interpolated[:, dim] = interp_func(event_times)

    # Handle NaN timestamps - propagate to positions
    nan_mask = np.isnan(event_times)
    interpolated[nan_mask, :] = np.nan

    # Create result DataFrame (copy to preserve original)
    result = events.copy()

    # Add coordinate columns
    column_names = ["x", "y", "z"][:n_dims]
    for i, col_name in enumerate(column_names):
        result[col_name] = interpolated[:, i]

    return result
