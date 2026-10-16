"""Pure array helpers shared by stored geometry records and native builders."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def get_centers(bin_edges: NDArray[np.float64]) -> NDArray[np.float64]:
    """Calculate the center of each bin given its edges.

    Parameters
    ----------
    bin_edges : NDArray[np.float64], shape (n_edges,)
        A 1D array of sorted coordinates representing the edges that define
        a sequence of bins. For `N` bins, there will be `N+1` edges.

    Returns
    -------
    NDArray[np.float64], shape (n_edges - 1,)
        A 1D array containing the center coordinate of each bin.

    """
    return bin_edges[:-1] + np.diff(bin_edges) / 2


def _mark_read_only(value: object) -> None:
    """Mark an array, or the arrays in a (named) tuple, read-only in place."""
    if isinstance(value, np.ndarray):
        value.setflags(write=False)
    elif isinstance(value, tuple):
        for item in value:
            _mark_read_only(item)


class ReadOnlyState:
    """Restore read-only arrays when a frozen record is unpickled or deep-copied.

    Pickle and ``copy.deepcopy`` rebuild NumPy arrays as writable copies. A
    frozen dataclass that owns read-only arrays inherits this ``__setstate__``,
    which sets each attribute and marks the arrays in it, directly or inside
    tuples, read-only again.
    """

    def __setstate__(self, state: dict[str, Any]) -> None:
        for name, value in state.items():
            _mark_read_only(value)
            object.__setattr__(self, name, value)
