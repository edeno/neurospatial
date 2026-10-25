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
