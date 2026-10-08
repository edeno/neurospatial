"""Shared input-validation helpers for neurospatial.

These helpers provide consistent, informative error messages when user
inputs contain non-finite values or arrays whose lengths disagree. They
are intentionally strict: they never silently coerce, drop, or broadcast
values, so that numerical errors surface loudly rather than propagating
into downstream computations.
"""

from __future__ import annotations

from typing import Literal

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


def times_positions_problems(
    t: NDArray[np.float64] | None, p: NDArray[np.float64] | None
) -> tuple[list[str], bool]:
    """Collect pair problems, skipping an array that failed numeric conversion."""
    problems: list[str] = []
    if t is not None:
        if t.ndim != 1:
            problems.append(f"times must be 1-D (n_samples,), got shape {t.shape}.")
        else:
            # Finiteness applies to every 1-D timestamp array, including a single
            # sample; only monotonicity needs two or more samples.
            finite = np.isfinite(t)
            if not finite.all():
                problems.append(
                    f"times has {int((~finite).sum())} non-finite value(s), "
                    f"first at index {int(np.argmin(finite))}."
                )
            elif t.size > 1:
                down = np.flatnonzero(np.diff(t) < 0)
                if down.size:
                    k = int(down[0])
                    problems.append(
                        f"times must be monotonically non-decreasing; it decreases at "
                        f"{down.size} place(s), first {float(t[k])!r} -> {float(t[k + 1])!r} "
                        f"at index {k}."
                    )
    if p is not None and p.ndim not in (1, 2):
        problems.append(f"positions must be (n_samples, n_dims), got shape {p.shape}.")
    if (
        t is not None
        and p is not None
        and t.ndim >= 1
        and p.ndim >= 1
        and len(t) != len(p)
    ):
        problems.append(
            f"times and positions must have the same length; times has "
            f"{len(t)} samples, positions has {len(p)}."
        )
    looks_swapped = (t is not None and t.ndim == 2) or (
        p is not None and p.ndim == 1 and p.size > 1 and bool(np.all(np.diff(p) >= 0))
    )
    return problems, looks_swapped


def validate_times_positions(
    times: ArrayLike,
    positions: ArrayLike,
    *,
    call: str,
    order: Literal["times, positions", "positions, times"] = "times, positions",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate a ``(times, positions)`` pair and name an argument swap.

    Parameters
    ----------
    times : array-like, shape (n_samples,)
        Sample timestamps in seconds; must be 1-D, finite and non-decreasing.
    positions : array-like, shape (n_samples, n_dims) or (n_samples,)
        One position row per timestamp.
    call : str
        Public function name, used in the message.
    order : {"times, positions", "positions, times"}, default="times, positions"
        The order in which ``call`` takes the two arguments, so the message
        names the swap the caller actually made.

    Returns
    -------
    times, positions : ndarray
        float64 arrays; shapes are not changed.

    Raises
    ------
    ValueError
        Listing every problem, with a ``Fix:`` line that names the swap when the
        arguments look swapped.
    """
    conversion_problems = []
    arrays: list[NDArray[np.float64] | None] = []
    for name, value in (("times", times), ("positions", positions)):
        try:
            arrays.append(np.asarray(value, dtype=np.float64))
        except (TypeError, ValueError):
            arrays.append(None)
            conversion_problems.append(
                f"{name} must contain numeric values convertible to float64, got {value!r}."
            )
    t, p = arrays
    problems, looks_swapped = times_positions_problems(t, p)
    problems = conversion_problems + problems
    if problems:
        raise ValueError(
            format_times_positions_error(
                problems, looks_swapped, call=call, order=order
            )
        )
    assert t is not None and p is not None
    return t, p


def format_times_positions_error(
    problems: list[str], looks_swapped: bool, *, call: str, order: str
) -> str:
    first, second = order.split(", ")
    fix = (
        f"Fix: did you pass {second} before {first}? Call {call}(..., {order}, ...)."
        if looks_swapped
        else "Fix: pass times as a sorted 1-D array of numeric timestamps in seconds with one "
        "numeric positions row per timestamp."
    )
    return (
        f"Invalid times/positions passed to {call}():\n- "
        + "\n- ".join(problems)
        + "\nWhy: each interval [times[k], times[k+1]) is weighted by its duration, "
        "so mis-shaped or unsorted timestamps give wrong numbers.\n" + fix
    )
