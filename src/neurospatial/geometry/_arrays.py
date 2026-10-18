"""Pure array helpers shared by stored geometry records and native builders."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


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


def boolean_selection(keep: ArrayLike, n_items: int) -> NDArray[np.bool_]:
    """Return ``keep`` as a boolean mask with one entry per item.

    Raises
    ------
    ValueError
        If ``keep`` is not boolean or does not have shape ``(n_items,)``.
    """
    mask = np.asarray(keep)
    if mask.dtype.kind != "b" or mask.shape != (n_items,):
        raise ValueError("keep must be a boolean mask with one entry per cell.")
    return mask


def per_dimension(
    dims: NDArray[np.intp], value: Callable[[int], float]
) -> NDArray[np.float64]:
    """``value(d)`` for each entry of ``dims``, computed once per distinct ``d``.

    Scalar arithmetic per dimension keeps results identical to evaluating
    ``value`` on a single shared dimension.
    """
    result = np.empty(len(dims), dtype=np.float64)
    for dim in np.unique(dims):
        result[dims == dim] = value(int(dim))
    return result


def axis_ranges(
    low: NDArray[np.float64], high: NDArray[np.float64]
) -> tuple[tuple[float, float], ...]:
    """Pair per-axis lower and upper bounds as ``(low, high)`` floats."""
    return tuple((float(a), float(b)) for a, b in zip(low, high, strict=True))


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
