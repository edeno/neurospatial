"""Validation of layout build arguments."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def validate_bin_size(
    bin_size: float | NDArray[np.float64],
) -> NDArray[np.float64]:
    """Validate and normalize bin_size parameter.

    This function ensures bin_size is positive, finite, and properly formatted.
    Scalar values are converted to 1-element arrays for uniform processing.

    Parameters
    ----------
    bin_size : float | NDArray[np.float64]
        Bin size specification. Can be:
        - Scalar: Same size for all dimensions
        - Array: Per-dimension sizes

    Returns
    -------
    NDArray[np.float64]
        Validated bin_size as an array.

    Raises
    ------
    TypeError
        If bin_size is not numeric.
    ValueError
        If bin_size is not positive, finite, or contains invalid values.

    Examples
    --------
    >>> validate_bin_size(2.0)
    array([2.])

    >>> validate_bin_size(np.array([2.0, 3.0]))
    array([2., 3.])

    >>> validate_bin_size(-1.0)  # doctest: +SKIP
    ValueError: bin_size must be positive (got -1.0).

    >>> validate_bin_size(np.inf)  # doctest: +SKIP
    ValueError: bin_size contains infinite values...

    Notes
    -----
    This validator is used by layout engines to ensure consistent error
    messages and prevent common user errors like negative or zero bin sizes.

    See Also
    --------
    validate_dimension_ranges : Validates dimension_ranges parameter
    """
    # Convert to array for uniform processing
    bin_size_arr = np.atleast_1d(np.asarray(bin_size, dtype=np.float64))

    # Check for NaN
    if np.any(np.isnan(bin_size_arr)):
        raise ValueError(
            f"bin_size contains NaN values (got {bin_size}). "
            "bin_size must be finite numeric values."
        )

    # Check for infinity
    if np.any(np.isinf(bin_size_arr)):
        raise ValueError(
            f"bin_size contains infinite values (got {bin_size}). "
            "bin_size must be finite numeric values."
        )

    # Check positivity
    if np.any(bin_size_arr <= 0.0):
        raise ValueError(f"bin_size must be positive (got {bin_size}).")

    return bin_size_arr


def validate_dimension_ranges(
    dimension_ranges: list[tuple[float, float]] | tuple[tuple[float, float], ...],
    *,
    n_dims: int | None = None,
) -> list[tuple[float, float]]:
    """Validate and normalize dimension_ranges parameter.

    Ensures dimension_ranges has correct structure, valid numeric values,
    and consistent dimensionality if n_dims is provided.

    Parameters
    ----------
    dimension_ranges : list[tuple[float, float]] | tuple[tuple[float, float], ...]
        Dimension ranges as sequence of (min, max) tuples.
    n_dims : int | None, optional
        Expected number of dimensions. If provided, validates length matches.

    Returns
    -------
    list[tuple[float, float]]
        Validated dimension_ranges as list of tuples.

    Raises
    ------
    TypeError
        If dimension_ranges is not a sequence or contains non-tuples.
    ValueError
        If ranges are invalid (min >= max, non-finite values, wrong length).

    Examples
    --------
    >>> validate_dimension_ranges([(0, 100), (0, 200)])
    [(0.0, 100.0), (0.0, 200.0)]

    >>> validate_dimension_ranges([(0, 100), (0, 200)], n_dims=2)
    [(0.0, 100.0), (0.0, 200.0)]

    >>> validate_dimension_ranges([(100, 0)])  # doctest: +SKIP
    ValueError: dimension_ranges[0] has min >= max: (100, 0)

    >>> validate_dimension_ranges([(0, 100)], n_dims=2)  # doctest: +SKIP
    ValueError: dimension_ranges has 1 dimensions, expected 2

    Notes
    -----
    This validator is used by layout engines to ensure consistent error
    messages and prevent common user errors like inverted ranges or
    mismatched dimensionality.

    See Also
    --------
    validate_bin_size : Validates bin_size parameter
    """
    # Check type
    if not isinstance(dimension_ranges, (list, tuple)):
        raise TypeError(
            f"dimension_ranges must be a list or tuple, "
            f"got {type(dimension_ranges).__name__}"
        )

    # Check not empty
    if len(dimension_ranges) == 0:
        raise ValueError("dimension_ranges cannot be empty")

    # Check dimensionality if specified
    if n_dims is not None and len(dimension_ranges) != n_dims:
        raise ValueError(
            f"dimension_ranges has {len(dimension_ranges)} dimensions, "
            f"expected {n_dims}"
        )

    # Validate each range
    validated_ranges = []
    for i, range_tuple in enumerate(dimension_ranges):
        # Check tuple structure
        if not isinstance(range_tuple, (tuple, list)):
            raise TypeError(
                f"dimension_ranges[{i}] must be a tuple or list, "
                f"got {type(range_tuple).__name__}"
            )

        if len(range_tuple) != 2:
            raise ValueError(
                f"dimension_ranges[{i}] must have exactly 2 elements (min, max), "
                f"got {len(range_tuple)}: {range_tuple}"
            )

        min_val, max_val = range_tuple

        # Check numeric
        try:
            min_float = float(min_val)
            max_float = float(max_val)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"dimension_ranges[{i}] contains non-numeric values: {range_tuple}. "
                f"Error: {e}"
            ) from e

        # Check finite
        if not np.isfinite(min_float) or not np.isfinite(max_float):
            raise ValueError(
                f"dimension_ranges[{i}] contains non-finite values: "
                f"({min_float}, {max_float})"
            )

        # Check min < max
        if min_float >= max_float:
            raise ValueError(
                f"dimension_ranges[{i}] has min >= max: ({min_float}, {max_float}). "
                f"Ranges must have min < max."
            )

        validated_ranges.append((min_float, max_float))

    return validated_ranges
