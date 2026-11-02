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
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, TypeGuard, TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurospatial._exceptions import InputError, _format_error
from neurospatial._intervals import _is_pynapple_object

_T = TypeVar("_T")


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
            msg = (
                f"{name} contains {n} infinite value(s) "
                f"(first at {index} {first}: {float(arr.flat[first])}). "
                "NaN marks a missing value; replace infinities with NaN."
            )
            raise ValueError(msg)
        msg = (
            f"{name} contains {n} non-finite value(s) "
            f"(first at {index} {first}: {float(arr.flat[first])}). "
            f"Remove or mask them before calling."
        )
        raise ValueError(msg)
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


def validate_lengths(name_to_array: dict[str, NDArray[Any]]) -> None:
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
    InputError
        Listing every problem, with a ``Fix:`` line that names the swap when the
        arguments look swapped; or when ``positions`` hold an infinite value
        (:func:`refuse_infinite`), since NaN is the only missing-sample marker.
    """
    conversion_problems = []
    arrays: list[NDArray[np.float64] | None] = []
    for name, value in (("times", times), ("positions", positions)):
        try:
            arrays.append(np.asarray(value, dtype=np.float64))
        except (TypeError, ValueError):  # noqa: PERF203  # each value reports its own conversion failure
            arrays.append(None)
            conversion_problems.append(
                f"{name} must contain numeric values convertible to float64, got {value!r}."
            )
    t, p = arrays
    problems, looks_swapped = times_positions_problems(t, p)
    problems = conversion_problems + problems
    if problems:
        raise InputError(
            format_times_positions_error(problems, looks_swapped, call=call)
        )
    assert t is not None
    assert p is not None
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


def is_integer(value: object) -> TypeGuard[numbers.Integral]:
    """Whether ``value`` is a Python or NumPy integer; booleans are not.

    Examples
    --------
    >>> is_integer(np.int64(3)), is_integer(True), is_integer(3.0)
    (True, False, False)
    """
    return not isinstance(value, (bool, np.bool_)) and isinstance(
        value, numbers.Integral
    )


def is_real_number(value: object) -> bool:
    """Whether ``value`` is one real number; booleans, arrays and None are not.

    Examples
    --------
    >>> is_real_number(np.float32(-1.5)), is_real_number(True), is_real_number("1")
    (True, False, False)
    """
    return isinstance(value, numbers.Real) and not isinstance(value, (bool, np.bool_))


def is_integer_at_least(value: object, minimum: int) -> bool:
    """Whether ``value`` is an integer of at least ``minimum``, booleans excluded.

    Python and NumPy integers both count; ``True`` is not read as 1.
    """
    return is_integer(value) and int(value) >= minimum


def validate_integer_at_least(
    value: object,
    minimum: int,
    *,
    name: str,
    owner: str,
    hint: str = "",
) -> int:
    """Require an integer of at least ``minimum``, booleans excluded; return it.

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

    Returns
    -------
    int
        ``value`` as a Python int.

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
    assert isinstance(value, numbers.Integral)  # checked above
    return int(value)


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
        msg = f"{owner}: alpha must be in (0, 1), got {alpha!r}."
        raise ValueError(msg)


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
    value: object, *, name: str, positive: bool, none_allowed: bool = False
) -> float | None:
    """Require one finite real number, nonnegative or strictly positive; return it.

    Booleans and non-numeric values are rejected rather than coerced.

    Parameters
    ----------
    value : object
        The value to check.
    name : str
        Name used in the error message.
    positive : bool
        Require ``value > 0``; otherwise ``value >= 0``.
    none_allowed : bool, default=False
        Accept ``None`` as "not set".

    Returns
    -------
    float or None
        ``value`` as a float, or None when ``none_allowed`` and it is None.

    Raises
    ------
    ValueError
        If ``value`` is not an allowed finite real number.
    """
    if none_allowed and value is None:
        return None
    if not is_finite_real(value, positive=positive):
        msg = (
            f"{name} must be finite and {'positive' if positive else 'nonnegative'}"
            f"{', or None' if none_allowed else ''}; got {value!r}."
        )
        raise ValueError(msg)
    return float(value)


def validate_choice(
    value: object, choices: Sequence[_T], *, name: str, owner: str
) -> _T:
    """Require ``value`` to be one of the documented ``choices``; return it.

    The return value has the choices' type, so ``method =
    validate_choice(method, ("a", "b"), ...)`` narrows ``method`` for the type
    checker as well as at run time.

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

    Returns
    -------
    object
        The matching element of ``choices``.

    Raises
    ------
    ValueError
        If ``value`` is not one of ``choices``. A string close to a string
        choice (a typo or a shortened name) gets "Did you mean ...?".

    Examples
    --------
    >>> validate_choice("exit", ("entry", "exit"), name="direction", owner="f()")
    'exit'
    >>> validate_choice("ext", ("entry", "exit"), name="direction", owner="f()")
    Traceback (most recent call last):
        ...
    ValueError: f() got direction='ext'; direction must be one of 'entry', 'exit'. Did you mean 'exit'?
    ...
    """
    from difflib import get_close_matches

    for choice in choices:
        if value is choice or value == choice:
            return choice
    options = ", ".join(repr(choice) for choice in choices)
    what = f"{owner} got {name}={value!r}; {name} must be one of {options}."
    fix = f"pass {name}= one of {options}"
    if isinstance(value, str):
        names = [choice for choice in choices if isinstance(choice, str)]
        # A shortened name ("gaussian" for "gaussian_kde") is a prefix, which
        # difflib may rank below the cutoff.
        matches: list[str] = [name for name in names if name.startswith(value)]
        matches = matches or get_close_matches(value, names, n=1)
        if len(matches) == 1:
            what += f" Did you mean {matches[0]!r}?"
            fix = f"pass {name}={matches[0]!r}"
    raise ValueError(
        _format_error(
            what,
            why=f"each {name} option has one documented meaning.",
            fix=fix,
        )
    )


def validate_bool(value: object, *, name: str, owner: str) -> bool:
    """Require ``True`` or ``False``; return it as a Python bool.

    NumPy booleans count; ``0``, ``1`` and other truthy values do not.

    Raises
    ------
    TypeError
        If ``value`` is not a boolean.

    Examples
    --------
    >>> validate_bool(np.True_, name="show", owner="f()")
    True
    """
    if not isinstance(value, (bool, np.bool_)):
        msg = f"{owner}: {name} must be True or False, got {value!r}."
        raise TypeError(msg)
    return bool(value)


def validate_instance(
    value: object,
    cls: type[_T],
    *,
    name: str,
    owner: str,
    hint: str = "",
) -> _T:
    """Require an instance of ``cls``; return it, typed as one.

    Parameters
    ----------
    value : object
        The value to check.
    cls : type
        The required class.
    name : str
        Argument or field name, used in the error message.
    owner : str
        What received the value, as the message names it.
    hint : str, default=""
        How to fix it, added to the message.

    Raises
    ------
    TypeError
        If ``value`` is not an instance of ``cls``.

    Examples
    --------
    >>> validate_instance("cm", str, name="units", owner="f()")
    'cm'
    """
    if not isinstance(value, cls):
        msg = (
            f"{owner}: {name} must be a {cls.__name__}, got {type(value).__name__}"
            + (f"; {hint}." if hint else ".")
        )
        raise TypeError(msg)
    return value


def validate_mapping(
    value: object,
    *,
    name: str,
    owner: str,
    hint: str = "",
) -> Mapping[Any, Any]:
    """Require a mapping; return it, with keys and values of unknown type.

    Raises
    ------
    TypeError
        If ``value`` is not a :class:`~collections.abc.Mapping`.

    Examples
    --------
    >>> validate_mapping({"a": 1}, name="weights", owner="f()")
    {'a': 1}
    """
    if not isinstance(value, Mapping):
        msg = f"{owner}: {name} must be a mapping, got {type(value).__name__}" + (
            f"; {hint}." if hint else "."
        )
        raise TypeError(msg)
    return value


def validate_callable(
    value: object, *, name: str, owner: str, hint: str = ""
) -> Callable[..., Any]:
    """Require a callable; return it, typed as one.

    Raises
    ------
    TypeError
        If ``value`` is not callable.
    """
    if not callable(value):
        msg = f"{owner}: {name} must be callable, got {type(value).__name__}" + (
            f"; {hint}." if hint else "."
        )
        raise TypeError(msg)
    return value


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


def _train_count(value: object) -> int | None:
    """Number of spike trains in a list, tuple or 2-D array, else None.

    A list, tuple or 1-D object array holds trains when every element is a
    1-D sequence (a list of scalars is one train; a pynapple ``Ts`` is 1-D).
    A 2-D array with several rows holds one train per row, except an
    ``(n_spikes, 2)`` array with more than two rows, which is read as a table
    of spike time and unit. A ``(1, n)`` row vector (as from
    ``scipy.io.loadmat``) is one unit stored as a row, not a population.
    """
    if isinstance(value, (list, tuple)) or (
        isinstance(value, np.ndarray) and value.ndim == 1 and value.dtype == object
    ):
        # A pynapple ``Ts`` contributes its timestamps, as when trains are read.
        if len(value) and all(
            np.ndim(timestamps_of(item, name="spike_times")) == 1 for item in value
        ):
            return len(value)
        return None
    if isinstance(value, np.ndarray) and value.ndim == 2:
        n_rows, n_columns = value.shape
        if n_rows > 1 and n_columns > 1 and not (n_columns == 2 and n_rows > 2):
            return int(n_rows)
    return None


def _zero_padded_trains(value: NDArray[Any]) -> bool:
    """Whether every row of a 2-D array is increasing, then only zeros."""
    for row in value:
        nonzero = np.flatnonzero(row)
        end = nonzero[-1] + 1 if nonzero.size else 0
        if np.any(row[end:] != 0) or np.any(np.diff(row[:end]) <= 0):
            return False
    return True


def _is_result(value: object) -> bool:
    """Whether ``value`` is a neurospatial analysis result."""
    from neurospatial._results import ResultMixin

    return isinstance(value, ResultMixin)


def _result_fix(value: object, *, caller: str, expected: str) -> str:
    """What to pass instead of the result object ``value``."""
    kind = type(value).__name__
    if expected == "rates" and kind == "SpatialRateResult":
        return (
            "pass the population result from compute_spatial_rates(env, "
            "spike_times, times, positions); decoding compares the rate maps "
            "of every unit"
        )
    method = caller.rsplit(".", 1)[-1]
    if callable(getattr(value, method, None)):
        return f"call r.{method}() on the result r instead"
    for attribute in ("firing_rates", "firing_rate"):
        if hasattr(value, attribute):
            return f"pass the result's array, r.{attribute}"
    return f"pass the array {caller}() documents, not a {kind}"


def _refuse_wrong_kind(
    value: object,
    *,
    name: str,
    caller: str,
    expected: Literal["spike_train", "spike_counts", "spike_times", "rates", "array"],
    population: str | None = None,
) -> None:
    """Refuse an input of the wrong kind with the call that takes it.

    NumPy would otherwise fail later with a message about shapes
    ("inhomogeneous shape", "got shape ()") that does not say what was
    passed. Inputs of the expected kind pass silently.

    Parameters
    ----------
    value : object
        The argument as the caller passed it.
    name : str
        The argument's name, e.g. ``"spike_times"``.
    caller : str
        The public function, without parentheses, e.g.
        ``"compute_spatial_rate"`` or ``"BayesianDecoder.fit"``.
    expected : {"spike_train", "spike_counts", "spike_times", "rates", "array"}
        What ``name`` takes:

        - ``"spike_train"``: one unit's spike times. A list or tuple of
          trains, or a 2-D array of two or more equal-length trains, is
          refused and ``population`` named.
        - ``"spike_counts"``: a ``(n_time_bins, n_neurons)`` count matrix. A
          list or tuple of spike trains (lengths that differ, or every row in
          time order) is refused with the binning call, and so
          is the ``(counts, bin_edges)`` tuple ``bin_spikes_in_time``
          returns.
        - ``"spike_times"``: spike times of several units. A 2-D array of
          non-negative whole numbers, some above 1, with a value repeated in
          a row is a count matrix, not spike times (one unit cannot fire
          twice at the same instant), and is refused. Only 0s and 1s is left
          to the spike-raster refusal.
        - ``"rates"``: population rate maps; a one-unit rate result is
          refused with the population call.
        - ``"array"``: any array.

        For every kind, a neurospatial result object is refused with the
        method or attribute that gives what was meant.
    population : str, optional
        The population function, for ``"spike_train"``; None when there is
        none, and the fix says to call ``caller`` once per unit.

    Raises
    ------
    TypeError
        If ``value`` is of the wrong kind.

    Examples
    --------
    >>> import numpy as np
    >>> trains = [np.array([0.1, 0.5]), np.array([0.2])]
    >>> _refuse_wrong_kind(
    ...     trains,
    ...     name="spike_times",
    ...     caller="compute_spatial_rate",
    ...     expected="spike_train",
    ...     population="compute_spatial_rates",
    ... )
    Traceback (most recent call last):
        ...
    TypeError: compute_spatial_rate() takes one unit's spike times; got 2 trains.
    ...
    """
    if _is_result(value):
        raise TypeError(
            _format_error(
                f"{caller}() got a {type(value).__name__} for {name}, which "
                "takes an array.",
                fix=_result_fix(value, caller=caller, expected=expected),
            )
        )
    if expected == "spike_train":
        n_trains = _train_count(value)
        if n_trains is not None:
            plural = "train" if n_trains == 1 else "trains"
            raise TypeError(
                _format_error(
                    f"{caller}() takes one unit's spike times; got {n_trains} "
                    f"{plural}.",
                    why="a list of trains or a 2-D array holds several units' "
                    f"spike times, and {caller}() gives one unit's result.",
                    fix=(
                        f"call {caller}() once per unit"
                        if population is None
                        else f"use {population}(...) for a population"
                    )
                    + f", or pass {name}[i] for unit i.",
                )
            )
    elif expected == "spike_counts" and isinstance(value, (list, tuple)):
        if isinstance(value, tuple) and len(value) == 2 and np.ndim(value[0]) == 2:
            raise TypeError(
                _format_error(
                    f"{caller}() got a tuple for {name}: the (counts, bin_edges) "
                    "pair bin_spikes_in_time returns.",
                    fix="unpack it and pass the counts: counts, bin_edges = "
                    "bin_spikes_in_time(spike_times, dt=...).",
                )
            )
        n_trains = _train_count(value)
        if n_trains is not None:
            rows = [
                np.asarray(timestamps_of(item, name=name), dtype=np.float64)
                for item in value
            ]
            ragged = len({row.size for row in rows}) > 1
            # Spike trains are in time order; rows of counts are not (counts
            # repeat), so fractional counts get the fractional-count refusal.
            increasing = all(np.all(np.diff(row) > 0) for row in rows)
            if ragged or increasing:
                raise TypeError(
                    _format_error(
                        f"{caller}() takes a spike count matrix of shape "
                        f"(n_time_bins, n_neurons) for {name}; got {n_trains} "
                        "spike trains.",
                        why="decoding compares the number of spikes of each "
                        "neuron in each time bin with the rate maps.",
                        fix="bin them first with bin_spikes_in_time(spike_times, "
                        "dt=...), which returns (counts, bin_edges), and pass "
                        "the counts.",
                    )
                )
    elif expected == "spike_times" and isinstance(value, np.ndarray):
        if (
            value.ndim == 2
            and value.size
            and value.dtype.kind in "biuf"
            and np.isfinite(value).all()
            and value.min() >= 0
            # Only 0s and 1s is a spike raster, refused more precisely when
            # the trains are read.
            and value.max() > 1
            and np.all(value == np.round(value))
            and np.any(np.diff(np.sort(value, axis=1), axis=1) == 0)
            # Rows in time order followed by zeros are trains padded with
            # zeros, refused with their own fix when the trains are read.
            and not _zero_padded_trains(value)
        ):
            raise TypeError(
                _format_error(
                    f"{caller}() takes spike times, one array per unit; this "
                    f"{value.shape} array of non-negative whole numbers looks "
                    "like a spike count matrix.",
                    why="a value repeats within a row, and one unit cannot fire "
                    "twice at the same instant, so the rows are not spike times.",
                    fix="decode a count matrix of shape (n_time_bins, n_neurons) "
                    "with decode_position(env, counts, rates, dt=dt), with rates "
                    f"from compute_spatial_rates; or pass spike times to {caller}().",
                )
            )
