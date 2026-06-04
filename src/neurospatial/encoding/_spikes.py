"""Coerce spike input into the canonical spike-trains structure.

This module provides helpers to convert the various ways a user might pass
spike data into one canonical structure: a list of 1D NumPy arrays, one array
per neuron. It changes the *container shape* only -- the spike-time values are
never shifted, rescaled, or aligned.

The main function is `as_spike_trains()`, which accepts:

- 1D array (single neuron) → wrapped in list
- 2D array (n_neurons, max_spikes) with NaN padding → split, NaNs removed
- list/tuple of scalars (single neuron) → converted to 1D array, wrapped in list
- list/tuple of 1D arrays (canonical format) → each element converted to array

This coercion happens at the entry point of encoding functions, ensuring
consistent internal handling regardless of how the user provides spike data.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def as_spike_trains(
    spike_times: Any,
) -> list[NDArray[np.float64]]:
    """Coerce spike input into the canonical list-of-spike-trains structure.

    Returns a list of 1D arrays, one per neuron — the canonical structure the
    encoding/decoding functions consume. This standardizes the *container
    shape* only; the spike-time values are not shifted, rescaled, or aligned.

    Parameters
    ----------
    spike_times : array or sequence of arrays
        Spike times in one of these formats:

        - 1D array (single neuron) → wrapped in list
        - 2D array (n_neurons, max_spikes) → split along axis 0, NaN padding
          removed from each row
        - List/tuple of scalars (single neuron) → converted to 1D array,
          wrapped in list (e.g., ``[0.1, 0.5, 1.2]``)
        - List/tuple of 1D arrays (canonical format) → each element converted
          to float64 array

    Returns
    -------
    list[NDArray[np.float64]]
        List of 1D spike time arrays, one per neuron. Each array contains
        the spike times for that neuron in float64 dtype.

    Raises
    ------
    ValueError
        If input is a ragged object array, has more than 2 dimensions,
        or contains elements that are not 1D arrays.

    Examples
    --------
    Single neuron (1D array):

    >>> import numpy as np
    >>> from neurospatial.encoding import as_spike_trains
    >>> spikes = np.array([0.1, 0.5, 1.2])
    >>> normalized = as_spike_trains(spikes)
    >>> len(normalized)
    1
    >>> normalized[0]
    array([0.1, 0.5, 1.2])

    Single neuron (list of scalars - common user input):

    >>> spikes = [0.1, 0.5, 1.2]  # Plain list of floats
    >>> normalized = as_spike_trains(spikes)
    >>> len(normalized)
    1
    >>> normalized[0]
    array([0.1, 0.5, 1.2])

    Multiple neurons (list of arrays):

    >>> spikes = [np.array([0.1, 0.5]), np.array([0.2, 0.3, 0.8])]
    >>> normalized = as_spike_trains(spikes)
    >>> len(normalized)
    2

    NaN-padded 2D array:

    >>> spikes = np.array([[0.1, 0.5, np.nan], [0.2, 0.3, 0.8]])
    >>> normalized = as_spike_trains(spikes)
    >>> normalized[0]  # NaN removed
    array([0.1, 0.5])
    >>> normalized[1]  # No NaN to remove
    array([0.2, 0.3, 0.8])

    Empty list returns empty list:

    >>> as_spike_trains([])
    []
    """
    # Handle list/tuple explicitly (avoid mypy unreachable code warning)
    # Check for list/tuple before ndarray to handle sequences differently
    if isinstance(spike_times, (list, tuple)):
        # Empty sequence
        if len(spike_times) == 0:
            return []

        # Check if this is a list of scalars (single neuron's spike times)
        # This is a common user input pattern: [0.1, 0.5, 1.0]
        first_elem = spike_times[0]
        if isinstance(first_elem, (int, float, np.floating, np.integer)):
            # Treat as single neuron: convert entire list to 1D array
            arr = np.asarray(spike_times, dtype=np.float64)
            if arr.ndim != 1:
                raise ValueError(
                    f"Expected 1D array from list of scalars, got shape {arr.shape}"
                )
            return [arr]

        # Convert each element to 1D float64 array (list of arrays pattern)
        result: list[NDArray[np.float64]] = []
        for i, row in enumerate(spike_times):
            arr = np.asarray(row, dtype=np.float64)
            if arr.ndim != 1:
                raise ValueError(
                    f"Each spike train must be 1D, but element {i} has shape {arr.shape}"
                )
            result.append(arr)
        return result

    # Convert to array for shape inspection
    arr = np.asarray(spike_times)

    # Reject object arrays (ragged input passed as array)
    if arr.dtype == object:
        raise ValueError(
            "Received ragged array (dtype=object). Pass spike times as a list of "
            "1D arrays instead, e.g., [np.array([0.1, 0.2]), np.array([0.3])]"
        )

    # 1D array: single neuron
    if arr.ndim == 1:
        return [arr.astype(np.float64)]

    # 2D array: split along axis 0, remove NaN padding
    if arr.ndim == 2:
        return [row[~np.isnan(row)].astype(np.float64) for row in arr]

    raise ValueError(
        f"spike_times must be 1D array, 2D array, or sequence of arrays, "
        f"got shape {arr.shape}"
    )
