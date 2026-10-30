"""Checks of the inputs replay functions share."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error


def _checked_intervals(
    intervals: Any,
    operation: str,
    *,
    name: str = "intervals",
    allow_empty: bool = False,
    example: str | None = None,
) -> NDArray[np.float64]:
    """Event windows as finite ``(start, stop)`` rows with ``start < stop``.

    Parameters
    ----------
    intervals : array_like, shape (n_events, 2)
        The windows, in seconds.
    operation, name : str
        The calling function and argument, named in the error.
    allow_empty : bool, default=False
        Return no rows, shape (0, 2), for an empty input instead of refusing it.
    example : str, optional
        Where such an array comes from, offered in the error's fix.

    Returns
    -------
    NDArray[np.float64], shape (n_events, 2)
        The windows, in input order.
    """
    windows = np.asarray(intervals, dtype=np.float64)
    if allow_empty and windows.size == 0:
        return windows.reshape(0, 2)
    if (
        windows.ndim != 2
        or windows.shape[1] != 2
        or len(windows) == 0
        or not np.all(np.isfinite(windows))
        or np.any(windows[:, 1] <= windows[:, 0])
    ):
        raise ValueError(
            _format_error(
                f"{operation}: {name} must be finite (start, stop) rows in seconds "
                f"with start < stop, got shape {windows.shape}.",
                why="each row is one event, and spikes at start <= t < stop "
                "belong to it",
                fix="pass an array of shape (n_events, 2)"
                + (f", such as {example}" if example else " in seconds"),
            )
        )
    return windows
