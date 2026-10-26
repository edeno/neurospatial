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


def validate_edge_spacing(
    edge_spacing: float | Sequence[float], n_edges: int
) -> NDArray[np.float64]:
    """Check the gaps between consecutive track edges and return one per gap.

    Parameters
    ----------
    edge_spacing : float or sequence of float
        One gap for every pair of consecutive edges of ``edge_order``, or one
        value per gap, in the units of the node positions.
    n_edges : int
        Number of edges in ``edge_order``.

    Returns
    -------
    ndarray of float64, shape (max(n_edges - 1, 0),)

    Raises
    ------
    ValueError
        If ``edge_spacing`` is not numbers, has the wrong number of values,
        or holds a gap that is negative or not finite.

    Examples
    --------
    >>> validate_edge_spacing(5.0, n_edges=3)
    array([5., 5.])
    """
    n_gaps = max(n_edges - 1, 0)
    scalar = np.ndim(edge_spacing) == 0
    try:
        gaps = np.asarray(edge_spacing, dtype=np.float64)
        numeric = not isinstance(edge_spacing, (str, bytes, bool, np.bool_))
    except (TypeError, ValueError):
        numeric = False
    if numeric and scalar:
        gaps = np.full(n_gaps, float(gaps))
    if numeric and gaps.ndim == 1 and gaps.shape != (n_gaps,):
        problem = (
            f"edge_spacing has length {gaps.shape[0]}, but edge_order has "
            f"{n_gaps} gap(s) between consecutive edges"
        )
    elif (
        not numeric
        or gaps.shape != (n_gaps,)
        or not np.all(np.isfinite(gaps) & (gaps >= 0))
    ):
        problem = f"edge_spacing={edge_spacing!r} is not a finite, non-negative gap"
    else:
        return gaps
    raise ValueError(
        _format_error(
            f"{problem}.",
            why="Why: edge_spacing is the linear distance inserted between "
            "consecutive track edges: one finite, non-negative number for "
            "every gap, or one per gap.",
            fix=f"pass one number, such as edge_spacing=0.0, or {n_gaps} "
            "non-negative values",
        )
    )


def validate_uniform_edges(
    grid_edges: Sequence[NDArray[np.floating]], *, label: str = "grid_edges[{axis}]"
) -> tuple[NDArray[np.float64], ...]:
    """Check that each axis's edges lay out cells of one width.

    Cell measures and face areas use each axis's first width, so the widths
    must agree up to float rounding of the edges, and that rounding must stay
    within 1 part in 10^4 of a cell. Edges far from the origin pass when they
    are exact, such as integers at 1e12.

    Parameters
    ----------
    grid_edges : sequence of ndarray
        Cell edges of each axis.
    label : str, default="grid_edges[{axis}]"
        Name of one axis's edges in errors, formatted with ``axis``.

    Returns
    -------
    tuple of ndarray of float64
        The edges as float arrays.

    Raises
    ------
    ValueError
        If an axis's edges are not finite, 1-D, at least two and strictly
        increasing; if its widths differ by more than rounding explains; or
        if rounding at its coordinates makes them differ by more than 1 part
        in 10^4 of a cell.

    Examples
    --------
    >>> validate_uniform_edges([np.linspace(0, 10, 6)])
    (array([ 0.,  2.,  4.,  6.,  8., 10.]),)
    """
    checked = []
    for axis, raw in enumerate(grid_edges):
        name = label.format(axis=axis)
        edges = np.asarray(raw, dtype=np.float64)
        if edges.ndim != 1 or edges.size < 2 or not np.all(np.isfinite(edges)):
            raise ValueError(
                f"{name} must be a finite 1-D array with >= 2 edges, "
                f"got shape {edges.shape}.\n"
                "Fix: pass e.g. np.linspace(start, stop, n_bins + 1)."
            )
        widths = np.diff(edges)
        if np.any(widths <= 0):
            raise ValueError(
                f"{name} must be strictly increasing; smallest width "
                f"is {widths.min():.6g}.\n"
                "Fix: sort the edges and drop duplicates."
            )
        w0 = widths[0]
        # Float spacing at the largest coordinate bounds each edge's
        # representation error.
        largest = float(np.max(np.abs(edges)))
        ulp = np.spacing(largest)
        deviation = float(np.max(np.abs(widths - w0)))
        allowance = 1e-7 * w0 + 4 * ulp
        # Where rounding can move a width by more than 1 part in 10^4, any
        # spread that large is either that rounding or cannot be told from it.
        coarse = 4 * ulp > 1e-4 * w0
        if coarse and deviation > 1e-4 * w0:
            raise ValueError(
                f"{name} reaches {largest:.6g}, where float64 resolves only "
                f"{ulp:.3g}; its bin widths differ from the first "
                f"({w0:.6g}) by up to {deviation:.3g}, more than 1 part in "
                "10^4, so bin widths and volumes would be wrong.\n"
                "Fix: subtract an origin offset before building the environment, "
                "e.g. positions - positions.min(axis=0)."
            )
        if deviation > allowance:
            raise ValueError(
                f"{name} must be uniformly spaced: widths differ from "
                f"the first ({w0:.10g}) by up to {deviation:.3g}, more than the "
                f"{allowance:.3g} rounding allowance; cell volumes and diffusion "
                "face measures assume one width per axis.\n"
                "Fix: use np.linspace(start, stop, n_bins + 1) for each axis."
            )
        checked.append(edges)
    return tuple(checked)
