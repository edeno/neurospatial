"""Shared input-validation helpers for neurospatial.

These helpers provide consistent, informative error messages when user
inputs contain non-finite values or arrays whose lengths disagree. They
are intentionally strict: they never silently coerce, drop, or broadcast
values, so that numerical errors surface loudly rather than propagating
into downstream computations.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def validate_finite(
    a: ArrayLike, *, name: str, allow_nan: bool = False
) -> NDArray[np.float64]:
    """Return ``a`` as float64, raising ValueError on non-finite values.

    Parameters
    ----------
    a : array-like
        Values to check. Converted to a float64 array.
    name : str
        Argument name, used in the error message.
    allow_nan : bool, optional
        If True, NaN is permitted (but Inf is not). Default False.

    Returns
    -------
    numpy.ndarray
        ``a`` converted to a float64 array. The values are returned
        unchanged; no coercion or dropping of non-finite values occurs.

    Raises
    ------
    ValueError
        If ``a`` contains any infinite value, or any NaN value when
        ``allow_nan`` is False. The message names the argument, the
        number of offending values, and the index and value of the
        first one.

    Examples
    --------
    >>> import numpy as np
    >>> validate_finite([1.0, 2.0, 3.0], name="x")
    array([1., 2., 3.])
    """
    arr = np.asarray(a, dtype=np.float64)
    bad = ~np.isfinite(arr)
    if allow_nan:
        bad &= ~np.isnan(arr)
    if bad.any():
        n = int(bad.sum())
        first = int(np.argmax(bad))
        raise ValueError(
            f"{name} contains {n} non-finite value(s) "
            f"(first at index {first}: {arr.flat[first]!r}). "
            f"Remove or mask them before calling."
        )
    return arr


def validate_lengths(name_to_array: dict[str, NDArray]) -> None:
    """Raise ValueError if the named 1-D arrays do not share a length.

    Lengths are compared exactly: arrays are neither reshaped nor
    broadcast. A length-1 array among longer arrays is treated as a
    mismatch, not as a broadcastable convenience.

    Parameters
    ----------
    name_to_array : dict of str to array-like
        Mapping from argument name to array. The length (``len``) of
        each array is compared.

    Returns
    -------
    None
        Returns nothing when all lengths agree.

    Raises
    ------
    ValueError
        If the arrays do not all share the same length. The message
        lists each name and its length.

    Examples
    --------
    >>> import numpy as np
    >>> s = np.array([0.1, 0.2])
    >>> t = np.array([0.0, 1.0])
    >>> p = np.array([[0.0, 0.0], [1.0, 1.0]])
    >>> validate_lengths({"spike_times": s, "times": t, "positions": p})
    """
    lengths = {k: len(np.asarray(v)) for k, v in name_to_array.items()}
    if len(set(lengths.values())) > 1:
        pairs = ", ".join(f"{k}={n}" for k, n in lengths.items())
        raise ValueError(f"Length mismatch: {pairs}. These must agree.")
