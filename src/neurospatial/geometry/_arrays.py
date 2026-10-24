"""Pure array helpers shared by stored geometry records and native builders."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurospatial._exceptions import _format_error


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


def owned_grid(
    edges: Sequence[ArrayLike],
    shape: Sequence[Any] | None,
    active_mask: ArrayLike,
    *,
    name: str,
) -> tuple[tuple[NDArray[np.float64], ...], tuple[int, ...], NDArray[np.bool_]]:
    """Validate and own a rectangular grid's edges, cell counts and active mask.

    Every record that stores rectangular cells (grids, grid locators, grid
    frames, mask arenas and track linear axes) checks them here.

    Parameters
    ----------
    edges : sequence of array-like
        Cell edges along each axis. ``edges[d]`` must be one-dimensional,
        finite and strictly increasing, with shape ``(shape[d] + 1,)``.
    shape : sequence of int, or None
        Positive number of cells along each axis, one per edge array. None
        reads it from ``edges``.
    active_mask : array-like of bool, shape ``shape``
        Which cells are bins.
    name : str
        The record being built, named in error messages.

    Returns
    -------
    edges : tuple of ndarray of float64
        Read-only copies of the edges.
    shape : tuple of int
        The cell counts as Python integers.
    active_mask : ndarray of bool, shape ``shape``
        A read-only copy of the mask.

    Raises
    ------
    ValueError
        If there is no axis, an edge array is not real and one-dimensional,
        an axis has no cells, edges are not finite and strictly increasing,
        the mask is not boolean, or the edges, cell counts and mask disagree
        in shape.
    """
    fix = (
        "give each axis finite, strictly increasing edges with one more entry "
        "than it has cells, and a boolean active_mask with one entry per cell"
    )
    raw_edges = tuple(np.asarray(edge) for edge in edges)
    if not raw_edges or any(
        edge.ndim != 1 or edge.dtype.kind not in "iuf" for edge in raw_edges
    ):
        raise ValueError(
            _format_error(
                f"{name} needs one 1-D real array of edges per axis, got "
                f"{[edge.shape for edge in raw_edges]}.",
                fix=fix,
            )
        )
    counts = (
        tuple(len(edge) - 1 for edge in raw_edges) if shape is None else tuple(shape)
    )
    if len(counts) != len(raw_edges) or any(
        not isinstance(count, (int, np.integer))
        or isinstance(count, (bool, np.bool_))
        or count < 1
        for count in counts
    ):
        raise ValueError(
            _format_error(
                f"{name} shape {counts!r} must give a positive cell count for "
                f"each of its {len(raw_edges)} edge axes.",
                fix=fix,
            )
        )
    owned = tuple(np.array(edge, dtype=np.float64, copy=True) for edge in raw_edges)
    for axis, (edge, count) in enumerate(zip(owned, counts, strict=True)):
        if (
            edge.shape != (count + 1,)
            or not np.all(np.isfinite(edge))
            or np.any(np.diff(edge) <= 0)
        ):
            raise ValueError(
                _format_error(
                    f"{name} edges[{axis}] must be finite and strictly "
                    f"increasing with shape ({count + 1},), got shape {edge.shape}.",
                    fix=fix,
                )
            )
    raw_mask = np.asarray(active_mask)
    cells = tuple(int(count) for count in counts)
    if raw_mask.dtype.kind != "b" or raw_mask.shape != cells:
        raise ValueError(
            _format_error(
                f"{name} active_mask must be boolean with shape {cells}, got "
                f"{raw_mask.dtype} with shape {raw_mask.shape}.",
                fix=fix,
            )
        )
    mask = np.array(raw_mask, dtype=np.bool_, copy=True)
    for array in (*owned, mask):
        array.setflags(write=False)
    return owned, cells, mask


def owned_active_map(
    mapping: ArrayLike, size: int | None, *, name: str
) -> NDArray[np.intp]:
    """Validate and own a map from full-lattice cells to active-bin indices.

    Parameters
    ----------
    mapping : array-like of int, shape (size,)
        Active-bin index of each cell, or -1 for an inactive cell.
    size : int or None
        Number of cells in the full lattice; None accepts any length.
    name : str
        The field being checked, named in error messages.

    Returns
    -------
    ndarray of intp, shape (size,)
        A read-only copy.

    Raises
    ------
    ValueError
        If the map is not a one-dimensional integer array of ``size``
        entries, holds an index below -1, or does not number the active bins
        exactly once from zero.
    """
    raw = np.asarray(mapping)
    expected = (len(raw) if raw.ndim == 1 else -1,) if size is None else (size,)
    if (
        raw.shape != expected
        or (raw.size and raw.dtype.kind not in "iu")
        or np.any(raw < -1)
    ):
        cells = "cell" if size is None else f"of the {size} cells"
        raise ValueError(
            f"{name} must hold one integer index >= -1 for each {cells}, got "
            f"{raw.dtype} with shape {raw.shape}."
        )
    active = raw[raw >= 0]
    if not np.array_equal(np.sort(active), np.arange(len(active))):
        raise ValueError(f"{name} must number the active bins exactly once from zero.")
    owned = np.array(raw, dtype=np.intp, copy=True)
    owned.setflags(write=False)
    return owned


def mask_active_map(mask: NDArray[np.bool_]) -> NDArray[np.intp]:
    """Number a mask's active cells in row-major order.

    Parameters
    ----------
    mask : ndarray of bool, any shape
        Active cells of a full lattice.

    Returns
    -------
    ndarray of intp, shape (mask.size,)
        Read-only active-bin index of each flattened cell, or -1 for an
        inactive cell.
    """
    flat = np.asarray(mask, dtype=np.bool_).ravel()
    mapping = np.full(flat.size, -1, dtype=np.intp)
    mapping[flat] = np.arange(np.count_nonzero(flat), dtype=np.intp)
    mapping.setflags(write=False)
    return mapping


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
