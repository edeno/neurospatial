"""Shared input-validation helpers for neurospatial.

These helpers provide consistent, informative error messages when user
inputs contain non-finite values or arrays whose lengths disagree. They
are intentionally strict: they never silently coerce, drop, or broadcast
values, so that numerical errors surface loudly rather than propagating
into downstream computations.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TypeGuard

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurospatial._exceptions import _format_error
from neurospatial._intervals import _is_pynapple_object


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
        number of offending values, and the value of the first one with
        its index (the flat, C-order index for an array of more than one
        dimension).

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
        index = "flat index" if arr.ndim > 1 else "index"
        if allow_nan:
            raise ValueError(
                f"{name} contains {n} infinite value(s) "
                f"(first at {index} {first}: {float(arr.flat[first])}). "
                "NaN marks a missing value; replace infinities with NaN."
            )
        raise ValueError(
            f"{name} contains {n} non-finite value(s) "
            f"(first at {index} {first}: {float(arr.flat[first])}). "
            f"Remove or mask them before calling."
        )
    return arr


def timestamps_of(values: Any, *, name: str) -> Any:
    """Return the timestamps a times input carries.

    A pynapple ``Ts``, ``Tsd`` or ``TsdFrame`` keeps its timestamps in ``.t``
    and its data in ``.d``. ``np.asarray`` returns the data, which for a
    ``Tsd`` of reward sizes or peak amplitudes is not the times, so read
    ``.t`` from a pynapple object. Other objects pass through unchanged, even
    with a ``.t`` attribute: an xarray ``DataArray`` with a ``t`` dimension
    exposes its index there, not its values.

    Parameters
    ----------
    values : Any
        Times input: an array of times in seconds or a pynapple ``Ts``,
        ``Tsd`` or ``TsdFrame``.
    name : str
        Argument name, used in the error message.

    Returns
    -------
    Any
        ``values.t`` for a pynapple object, else ``values``.

    Raises
    ------
    ValueError
        If ``values`` is a spike group (pynapple ``TsGroup`` or a dict of
        trains), which holds one train per unit, not one set of times.
    """
    if isinstance(values, Mapping):
        raise ValueError(
            _format_error(
                f"{name} is a spike group ({type(values).__name__}), not one set "
                "of times.",
                why=f"Why: a TsGroup or dict holds one spike train per unit; "
                f"{name} takes one 1-D array of times in seconds.",
                fix="pass one unit's train, e.g. group[unit_id], or event times "
                "as an array or pynapple Ts; pass a whole spike group as "
                "spike_times to a population function.",
            )
        )
    if _is_pynapple_object(values):
        t = getattr(values, "t", None)
        if t is not None and np.ndim(t) == 1:
            return t
    return values


def validate_lengths(name_to_array: dict[str, NDArray]) -> None:
    """Raise ValueError if the named arrays do not share a first-axis length.

    Lengths along the first axis are compared exactly, so a 2-D
    ``(n_samples, n_dims)`` array is compared by its ``n_samples``. Arrays
    are neither reshaped nor broadcast: a length-1 array among longer arrays
    is a mismatch, not a broadcastable convenience.

    Parameters
    ----------
    name_to_array : dict of str to array-like
        Mapping from argument name to array. The length (``len``, the first
        axis) of each array is compared.

    Returns
    -------
    None
        Returns nothing when all lengths agree.

    Raises
    ------
    ValueError
        If the arrays do not all share the same first-axis length. The
        message lists each name and its length.

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
        raise ValueError(
            _format_error(
                f"Length mismatch: {pairs}.",
                why="Why: these arrays pair up entry by entry along their first "
                "axis; they are never broadcast or truncated, so different "
                "lengths mean they do not describe the same samples.",
                fix="pass arrays with one entry per sample along the first axis; "
                "check that none was filtered, resampled or trimmed without the "
                "others.",
            )
        )


def refuse_infinite(values: NDArray[np.float64], *, name: str, call: str) -> None:
    """Refuse infinite entries; NaN is the only marker of a missing sample.

    Parameters
    ----------
    values : ndarray of float64, shape (n_samples,) or (n_samples, n_columns)
        Per-sample values such as positions, headings or signals.
    name : str
        Argument name, used in the message.
    call : str
        Public function name, used in the message.

    Raises
    ------
    ValueError
        If any entry is infinite, naming how many and the first one.

    Examples
    --------
    >>> import numpy as np
    >>> refuse_infinite(np.array([[0.0, np.nan]]), name="positions", call="f")
    >>> refuse_infinite(np.array([[0.0, np.inf]]), name="positions", call="f")
    Traceback (most recent call last):
    ...
    ValueError: f() got 1 infinite value(s) in positions, first at row 0, column 1 (inf).
    Why: NaN marks a missing sample, but an infinite value is not a measurement; it usually comes from a division by zero in a calibration or unit conversion, so it is not treated as missing.
    Fix: correct the step that produced it, or mark those samples missing with positions[np.isinf(positions)] = np.nan.
    >>> refuse_infinite(np.array([0.0, -np.inf]), name="headings", call="f")
    Traceback (most recent call last):
    ...
    ValueError: f() got 1 infinite value(s) in headings, first at index 1 (-inf).
    Why: NaN marks a missing sample, but an infinite value is not a measurement; it usually comes from a division by zero in a calibration or unit conversion, so it is not treated as missing.
    Fix: correct the step that produced it, or mark those samples missing with headings[np.isinf(headings)] = np.nan.
    """
    infinite = np.isinf(values)
    if not infinite.any():
        return
    first = tuple(int(i) for i in np.argwhere(infinite)[0])
    where = (
        f"row {first[0]}, column {first[1]}"
        if len(first) == 2
        else f"index {', '.join(map(str, first))}"
    )
    raise ValueError(
        _format_error(
            f"{call}() got {int(infinite.sum())} infinite value(s) in {name}, "
            f"first at {where} ({float(values[first])}).",
            why="NaN marks a missing sample, but an infinite value is not a "
            "measurement; it usually comes from a division by zero in a "
            "calibration or unit conversion, so it is not treated as missing.",
            fix="correct the step that produced it, or mark those samples "
            f"missing with {name}[np.isinf({name})] = np.nan.",
        )
    )


def times_positions_problems(
    t: NDArray[np.float64] | None, p: NDArray[np.float64] | None
) -> tuple[list[str], bool]:
    """Collect pair problems, skipping an array that failed numeric conversion."""
    problems: list[str] = []
    # Whether the recorded times go backwards, evidence of a swap below.
    times_go_back = False
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
                # A NaN in times says nothing about order.
                recorded = t[finite]
                times_go_back = bool(np.any(recorded[1:] < recorded[:-1]))
            elif t.size > 1:
                down = np.flatnonzero(t[1:] < t[:-1])
                times_go_back = bool(down.size)
                if down.size:
                    k = int(down[0])
                    problems.append(
                        f"times must be strictly increasing; it decreases at "
                        f"{down.size} place(s), first {float(t[k])!r} -> {float(t[k + 1])!r} "
                        f"at index {k}. To sort, index times and every per-sample "
                        'array with order = np.argsort(times, kind="stable").'
                    )
                repeated = np.flatnonzero(t[1:] == t[:-1])
                if repeated.size:
                    k = int(repeated[0])
                    problems.append(
                        f"times must be strictly increasing; {repeated.size} "
                        f"timestamp(s) repeat the one before, first {float(t[k])!r} "
                        f"at indices {k} and {k + 1}. A repeated timestamp gives one "
                        "instant two samples and the interval between them no "
                        "duration. To keep the first row of each timestamp, index "
                        "times and every per-sample array with "
                        "keep = np.unique(times, return_index=True)[1]; if the "
                        "recordings restart their clocks, analyze each recording "
                        "in its own call or map them to one clock."
                    )
    if p is not None and (p.ndim not in (1, 2) or (p.ndim == 2 and p.shape[1] == 0)):
        problems.append(
            "positions must be (n_samples,) or (n_samples, n_dims) with "
            f"n_dims >= 1, got shape {p.shape}."
        )
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
    # Suggest a swap only on evidence: 2-D times are a trajectory, or times
    # whose recorded values go backwards (as positions do) while positions
    # would be valid times. Positions that never decrease alone are normal on
    # a linear track.
    looks_swapped = (t is not None and t.ndim == 2) or (
        times_go_back
        and p is not None
        and p.ndim == 1
        and p.size > 1
        and bool(np.isfinite(p).all())
        and bool(np.all(p[1:] >= p[:-1]))
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
        Sample timestamps in seconds; must be 1-D, finite and strictly
        increasing.
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
        arguments look swapped; or when ``positions`` hold an infinite value
        (:func:`refuse_infinite`), since NaN is the only missing-sample marker.
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
    refuse_infinite(p, name="positions", call=call)
    return t, p


def format_times_positions_error(
    problems: list[str], looks_swapped: bool, *, call: str, positions: bool = True
) -> str:
    """Format timestamp problems; ``positions=False`` for a call without positions."""
    rows = ", with one numeric positions row per timestamp" if positions else ""
    fix = (
        "Fix: did you pass positions before times? "
        f"Call {call}(..., times, positions, ...)."
        if looks_swapped
        else "Fix: pass times as a sorted 1-D array of numeric timestamps in seconds, "
        f"without repeats{rows}."
    )
    return (
        f"Invalid times/positions passed to {call}():\n- "
        + "\n- ".join(problems)
        + "\nWhy: each interval [times[k], times[k+1]) is weighted by its duration, "
        "so mis-shaped, unsorted or repeated timestamps give wrong numbers.\n" + fix
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


def validate_draw_count(value: object, *, name: str, owner: str) -> None:
    """Require a positive number of shuffles or surrogates to draw.

    Raises
    ------
    ValueError
        If ``value`` is not an integer of at least 1, which would yield an
        empty null distribution.
    """
    validate_integer_at_least(
        value,
        1,
        name=name,
        owner=owner,
        hint="a null distribution needs at least one draw, e.g. 1000",
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
    if not (is_finite_real(alpha, positive=True) and alpha < 1):
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


def is_finite_real(value: object, *, positive: bool) -> TypeGuard[float]:
    """Whether ``value`` is one finite real number, positive or nonnegative.

    Python and NumPy numbers both count; booleans are not read as 0 or 1.

    Parameters
    ----------
    value : object
        The value to check.
    positive : bool
        Require ``value > 0``; otherwise ``value >= 0``.

    Examples
    --------
    >>> is_finite_real(0.0, positive=False), is_finite_real(0.0, positive=True)
    (True, False)
    >>> (
    ...     is_finite_real(True, positive=True),
    ...     is_finite_real(float("nan"), positive=False),
    ... )
    (False, False)
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        return False
    number = float(value)
    return math.isfinite(number) and (number > 0 if positive else number >= 0)


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
    if not is_finite_real(value, positive=positive):
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
