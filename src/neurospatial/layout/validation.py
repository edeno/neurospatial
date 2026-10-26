"""Validation of layout build arguments."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error


def validate_cell_size(
    value: float | Sequence[float] | NDArray[np.floating],
    *,
    name: str = "bin_size",
    n_dims: int | None = None,
    scalar_only: bool = False,
) -> NDArray[np.float64]:
    """Check a cell-size argument and return one size per dimension.

    Every layout engine checks its cell size here (``bin_size``,
    ``hexagon_width``, ``point_spacing``, ``pixel_size``), so a bad value
    gets the same refusal whichever layout is built.

    Parameters
    ----------
    value : float or sequence of float
        One size for every dimension, or one per dimension. A NumPy scalar
        counts as one size.
    name : str, default "bin_size"
        Argument name shown in errors.
    n_dims : int, optional
        Number of dimensions: a single size is repeated to this length, and
        a sequence must have it.
    scalar_only : bool, default False
        Refuse a sequence, for layouts whose cells have one size.

    Returns
    -------
    ndarray of float64, shape (n_dims,)
        Positive, finite sizes; shape ``(1,)`` for a single size without
        ``n_dims``.

    Raises
    ------
    TypeError
        If ``value`` is not numeric.
    ValueError
        If a size is not finite and positive, or the number of sizes is
        wrong.

    Examples
    --------
    >>> validate_cell_size(2.0, n_dims=2)
    array([2., 2.])
    >>> validate_cell_size(np.float32(2.5))
    array([2.5])
    >>> validate_cell_size([2.0, 3.0], n_dims=2)
    array([2., 3.])
    """
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise TypeError(
            _format_error(
                f"[E1002] {name} must be a number or a sequence of numbers, got "
                f"{type(value).__name__} {value!r}.",
                why="Why: it is the cell width in the units of the positions.",
                fix=f"pass e.g. {name}=2.0",
            )
        )
    if raw.ndim == 0:
        sizes = np.full(1 if n_dims is None else n_dims, float(raw))
    elif scalar_only or raw.ndim != 1 or (n_dims is not None and len(raw) != n_dims):
        expected = (
            "a single number"
            if scalar_only
            else f"one number per dimension ({n_dims})"
            if n_dims is not None
            else "a number or a 1-D sequence of numbers"
        )
        raise ValueError(
            _format_error(
                f"[E1002] {name} must be {expected}, got shape {raw.shape}: {value!r}.",
                why="Why: each cell needs one width per spatial dimension.",
                fix=f"pass e.g. {name}=2.0",
            )
        )
    else:
        sizes = raw.astype(np.float64)
    if not np.all(np.isfinite(sizes) & (sizes > 0)):
        raise ValueError(
            _format_error(
                f"[E1002] {name} must be positive and finite, got {value!r}.",
                why="Why: cell widths must be finite and positive in every "
                "spatial dimension.",
                fix=f"pass e.g. {name}=2.0, in the same units as the positions",
            )
        )
    return sizes


def validate_count_threshold(value: int) -> int:
    """Check ``bin_count_threshold``: a whole number of samples, at least 0.

    Parameters
    ----------
    value : int
        Fewest position samples a bin must hold to be active.

    Returns
    -------
    int

    Raises
    ------
    ValueError
        If ``value`` is negative, fractional or not a number.

    Examples
    --------
    >>> validate_count_threshold(3)
    3
    """
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not np.isfinite(value)
        or value < 0
        or value != int(value)
    ):
        raise ValueError(
            _format_error(
                f"bin_count_threshold must be a non-negative whole number of "
                f"samples, got {value!r}.",
                why="Why: a bin is active when it holds at least this many "
                "position samples.",
                fix="pass e.g. bin_count_threshold=0 to keep every visited bin",
            )
        )
    return int(value)


def validate_dimension_ranges(
    dimension_ranges: Sequence[tuple[float, float]],
    *,
    n_dims: int | None = None,
    name: str = "dimension_ranges",
) -> list[tuple[float, float]]:
    """Check explicit ranges and return one ``(minimum, maximum)`` per axis.

    Every layout that takes ranges checks them here, so a bad pair gets the
    same refusal from ``create_layout``, the engines and ``get_n_bins``. A
    zero-span pair is kept: the grid widens it to one cell.

    Parameters
    ----------
    dimension_ranges : sequence of (float, float)
        The ``(minimum, maximum)`` extent of each axis, in the units of the
        positions.
    n_dims : int, optional
        Number of axes the ranges must describe; any number when None.
    name : str, default="dimension_ranges"
        Argument name used in errors.

    Returns
    -------
    list of (float, float)

    Raises
    ------
    ValueError
        If the number of pairs is not ``n_dims``, or a pair is not two
        values, not finite, or has its minimum above its maximum.
    TypeError
        If a pair holds a value that is not a number.

    Examples
    --------
    >>> validate_dimension_ranges([(0, 10), (0, 5)])
    [(0.0, 10.0), (0.0, 5.0)]
    """
    pairs = list(dimension_ranges)
    if n_dims is not None and len(pairs) != n_dims:
        raise ValueError(
            f"`{name}` length ({len(pairs)}) must match n_dims ({n_dims}): one "
            "(min, max) pair per dimension.",
        )
    ranges = []
    for i, pair in enumerate(pairs):
        try:
            values = [float(value) for value in pair]
        except (TypeError, ValueError) as e:
            raise TypeError(
                f"{name} must contain numeric tuples of (min, max). "
                f"Error at dimension {i}: got {pair!r}"
            ) from e
        if len(values) != 2:
            raise ValueError(
                f"{name}[{i}] is {pair!r}; each axis needs one (min, max) pair."
            )
        lo, hi = values
        if not (np.isfinite(lo) and np.isfinite(hi)):
            raise ValueError(
                _format_error(
                    f"{name}[{i}] is {tuple(pair)!r}, which is not finite.",
                    why="Why: cells are laid across each range, so both ends "
                    "must be finite numbers.",
                    fix="pass finite (min, max) pairs in the units of "
                    "positions, e.g. dimension_ranges=[(0, 100), (0, 100)]",
                )
            )
        if lo > hi:
            raise ValueError(
                _format_error(
                    f"{name}[{i}] is {tuple(pair)!r}, whose minimum is above "
                    "its maximum.",
                    why="Why: each pair is an axis's (minimum, maximum) "
                    "extent, minimum first; a reversed pair usually means "
                    "swapped values or a pair given for the wrong axis.",
                    fix=f"pass ({pair[1]!r}, {pair[0]!r}) if that is the "
                    "extent, and check that the pairs follow the columns of "
                    "positions",
                )
            )
        ranges.append((lo, hi))
    return ranges
