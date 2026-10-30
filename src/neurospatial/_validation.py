"""Shared input-validation helpers for neurospatial.

These helpers provide consistent, informative error messages when user
inputs contain non-finite values or arrays whose lengths disagree. They
are intentionally strict: they never silently coerce, drop, or broadcast
values, so that numerical errors surface loudly rather than propagating
into downstream computations.
"""

from __future__ import annotations

import numbers
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurospatial._exceptions import _format_error


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
        if allow_nan:
            raise ValueError(
                f"{name} contains {n} infinite value(s) "
                f"(first at index {first}: {arr.flat[first]!r}). "
                "NaN marks a missing value; replace infinities with NaN."
            )
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
                        f"at index {k}. To sort, index times and every per-sample "
                        'array with order = np.argsort(times, kind="stable").'
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
            format_times_positions_error(problems, looks_swapped, call=call)
        )
    assert t is not None and p is not None
    return t, p


def validate_strictly_increasing(times: NDArray[np.float64], *, call: str) -> None:
    """Raise ValueError when ``times`` repeat a timestamp.

    Parameters
    ----------
    times : ndarray, shape (n_samples,)
        Timestamps in seconds, already validated as finite and non-decreasing
        (``validate_times_positions`` rejects decreasing clocks).
    call : str
        Public function name, used in the message.

    Raises
    ------
    ValueError
        Naming the first repeated timestamp and how to resolve it.
    """
    repeated = np.flatnonzero(np.diff(times) <= 0)
    if repeated.size == 0:
        return
    k = int(repeated[0])
    raise ValueError(
        _format_error(
            f"{call}() needs strictly increasing times; times[k] == times[k+1] "
            f"== {times[k]:g} at index {k} ({repeated.size} repeated timestamps).",
            why=(
                "interpolation and derivatives need positive elapsed time "
                "between samples."
            ),
            fix=(
                "drop repeated rows (keep = np.r_[True, np.diff(times) > 0]); "
                "if the recordings restart their clocks, analyze each recording "
                "in its own call or map them to one clock."
            ),
        )
    )


def format_times_positions_error(
    problems: list[str], looks_swapped: bool, *, call: str, positions: bool = True
) -> str:
    """Format timestamp problems; ``positions=False`` for a call without positions."""
    rows = " with one numeric positions row per timestamp" if positions else ""
    fix = (
        "Fix: did you pass positions before times? "
        f"Call {call}(..., times, positions, ...)."
        if looks_swapped
        else "Fix: pass times as a sorted 1-D array of numeric timestamps in seconds"
        f"{rows}."
    )
    return (
        f"Invalid times/positions passed to {call}():\n- "
        + "\n- ".join(problems)
        + "\nWhy: each interval [times[k], times[k+1]) is weighted by its duration, "
        "so mis-shaped or unsorted timestamps give wrong numbers.\n" + fix
    )


def is_integer_at_least(value: object, minimum: int) -> bool:
    """Whether ``value`` is an integer of at least ``minimum``, booleans excluded.

    Python and NumPy integers both count; ``True`` is not read as 1.
    """
    return (
        not isinstance(value, (bool, np.bool_))
        and isinstance(value, numbers.Integral)
        and int(value) >= minimum
    )


def validate_integer_at_least(
    value: object, minimum: int, *, name: str, owner: str, hint: str = ""
) -> None:
    """Require an integer of at least ``minimum``, booleans excluded.

    Parameters
    ----------
    value : object
        The value to check.
    minimum : int
        The smallest allowed value.
    name : str
        Argument or field name, used in the error message.
    owner : str
        What received the value, as the message names it, such as
        ``"shuffle_events()"`` or ``"ReplayMethod 'x'"``.
    hint : str, default=""
        Added to the message after the value, such as the value a paper
        used (``"Davidson et al. used 1500"``) or why the minimum holds.

    Raises
    ------
    ValueError
        If ``value`` is not an integer of at least ``minimum``.
    """
    if not is_integer_at_least(value, minimum):
        bound = (
            "a positive integer"
            if minimum == 1
            else f"an integer of at least {minimum}"
        )
        raise ValueError(
            f"{owner}: {name} must be {bound}, got {value!r}"
            + (f"; {hint}." if hint else ".")
        )


def validate_alpha(alpha: object, *, owner: str) -> None:
    """Require a significance level strictly between 0 and 1.

    Booleans and non-numeric values are rejected rather than coerced.

    Parameters
    ----------
    alpha : object
        The significance level to check.
    owner : str
        What received the value, as the message names it, such as
        ``"ReplayMethod 'x'"``.

    Raises
    ------
    ValueError
        If ``alpha`` is not a number in (0, 1).
    """
    if (
        isinstance(alpha, (bool, np.bool_))
        or not isinstance(alpha, (int, float, np.integer, np.floating))
        or not 0 < alpha < 1
    ):
        raise ValueError(f"{owner}: alpha must be in (0, 1), got {alpha!r}.")


def validate_float_dtype(
    dtype: Any, *, stored: str
) -> type[np.float32] | type[np.float64]:
    """Resolve a working precision to ``np.float32`` or ``np.float64``.

    Parameters
    ----------
    dtype : Any
        Anything :class:`numpy.dtype` accepts: a scalar type, a dtype or a
        name such as ``"float32"``.
    stored : str
        What the dtype is stored in, named in the error message, e.g.
        ``"rate maps"``.

    Returns
    -------
    type
        ``np.float32`` or ``np.float64``.

    Raises
    ------
    ValueError
        If ``dtype`` is not single or double precision, or not a dtype.
    """
    message = (
        f"dtype must be np.float32 or np.float64, got {dtype!r}. Only single- "
        f"and double-precision {stored} are supported (float32 halves their "
        "memory)."
    )
    # An unparseable name such as "bogus" raises this message, not NumPy's
    # "data type 'bogus' not understood".
    try:
        resolved = np.dtype(dtype)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if resolved == np.float32:
        return np.float32
    if resolved == np.float64:
        return np.float64
    raise ValueError(message)


def validate_finite_scalar(
    value: Any, *, name: str, positive: bool, none_allowed: bool = False
) -> None:
    """Require one finite real number, nonnegative or strictly positive.

    Booleans and non-numeric values are rejected rather than coerced.

    Parameters
    ----------
    value : Any
        The value to check.
    name : str
        Name used in the error message.
    positive : bool
        Require ``value > 0``; otherwise ``value >= 0``.
    none_allowed : bool, default=False
        Accept ``None`` as "not set".

    Raises
    ------
    ValueError
        If ``value`` is not an allowed finite real number.
    """
    if none_allowed and value is None:
        return
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not np.isfinite(value)
        or (value <= 0 if positive else value < 0)
    ):
        raise ValueError(
            f"{name} must be finite and {'positive' if positive else 'nonnegative'}"
            f"{', or None' if none_allowed else ''}; got {value!r}."
        )


def validate_choice(
    value: object, choices: Sequence[object], *, name: str, owner: str
) -> None:
    """Require ``value`` to be one of the documented ``choices``.

    Parameters
    ----------
    value : object
        The value to check.
    choices : sequence
        Allowed values, usually strings and possibly ``None``.
    name : str
        Argument or field name, used in the error message.
    owner : str
        What received the value, as the message names it: a function with
        parentheses (``"compare_spatial_rates()"``) or a record (``"Run"``).

    Raises
    ------
    ValueError
        If ``value`` is not one of ``choices``.

    Examples
    --------
    >>> validate_choice("exit", ("entry", "exit"), name="direction", owner="f()")
    """
    if value not in choices:
        options = ", ".join(repr(choice) for choice in choices)
        raise ValueError(
            _format_error(
                f"{owner} got {name}={value!r}; {name} must be one of {options}.",
                why=f"each {name} option has one documented meaning.",
                fix=f"pass {name}= one of {options}",
            )
        )


def check_writable(
    paths: Sequence[Path],
    *,
    overwrite: bool,
    what: str,
    argument: str,
    alternative: str = "",
) -> None:
    """Refuse existing files/nonempty directories unless overwrite is explicit.

    Parameters
    ----------
    paths : sequence of pathlib.Path
        All targets an export will write.
    overwrite : bool
        Whether existing outputs may be replaced.
    what : str
        Description of the outputs for the diagnostic.
    argument : str
        Path argument name used in corrected-call guidance.
    alternative : str, optional
        Another fix to offer, as a clause such as ``"or resume it with ..."``.

    Raises
    ------
    FileExistsError
        If a file or nonempty directory exists and overwrite is False.
    """
    if overwrite:
        return
    existing = [p for p in paths if (p.is_dir() and any(p.iterdir())) or p.is_file()]
    if existing:
        raise FileExistsError(
            _format_error(
                f"Refusing to overwrite existing {what}: {', '.join(map(str, existing))}.",
                fix=f"pass overwrite=True to replace {'it' if len(existing) == 1 else 'them'}, "
                f"or choose a different {argument}"
                + (f", {alternative}" if alternative else "")
                + ".",
            )
        )
