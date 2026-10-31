"""Read pynapple objects passed straight to analysis functions.

Analysis functions accept pynapple objects by duck typing and never import
pynapple. This module holds what those inputs carry beyond plain arrays: a
``TsGroup``'s per-unit metadata, a ``TsdFrame``'s column names and an
``IntervalSet``'s inclusive ends. :mod:`neurospatial.io.pynapple` applies the
same rules to explicit conversions.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

import numpy as np
from numpy.typing import NDArray

from neurospatial._intervals import _is_pynapple_object, pynapple_end_to_stop

if TYPE_CHECKING:
    import pandas as pd

_SPATIAL_AXES = ("x", "y", "z")
_P = ParamSpec("_P")
_R = TypeVar("_R")


def is_pynapple_type(value: Any, *names: str) -> bool:
    """Return whether ``value`` is a pynapple object of one of the named types."""
    return _is_pynapple_object(value) and type(value).__name__ in names


def refuse_permuted_spatial_columns(columns: Any, *, fix: str) -> None:
    """Refuse a TsdFrame whose columns are x/y(/z) in another order.

    Parameters
    ----------
    columns : sequence
        The frame's column labels.
    fix : str
        The corrected call, shown after ``Fix:``; ``{frame}`` is replaced by
        the reordered frame, ``tsdframe.loc[[...]]``.

    Raises
    ------
    ValueError
        If the labels are distinct spatial axis names not in x, y, z order.
    """
    labels = [str(c) for c in columns]
    lowered = [label.lower() for label in labels]
    spatial = set(lowered) <= set(_SPATIAL_AXES) and len(set(lowered)) == len(lowered)
    if not spatial or lowered == sorted(lowered):
        return
    canonical = sorted(lowered)
    frame = f"tsdframe.loc[{sorted(labels, key=str.lower)}]"
    raise ValueError(
        f"TsdFrame columns {labels} are spatial axes not in x, y, z order.\n"
        "Why: neurospatial reads positions by column position, so these values "
        f"would be used as ({', '.join(lowered)}) where every analysis expects "
        f"({', '.join(canonical)}), silently swapping the axes.\n"
        "Fix: reorder the columns first, " + fix.format(frame=frame) + "."
    )


def interval_rows(intervals: Any) -> NDArray[np.float64]:
    """Return a pynapple ``IntervalSet`` as half-open ``(n, 2)`` rows.

    Rows keep their order and are not merged. Each inclusive pynapple ``end``
    becomes the next float above it, so a row holds the same timestamps.

    Parameters
    ----------
    intervals : pynapple.IntervalSet

    Returns
    -------
    ndarray of float64, shape (n_intervals, 2)
        ``[start, stop)`` rows in seconds, an owned copy.
    """
    return np.column_stack(
        [
            np.array(intervals.start, dtype=np.float64),
            pynapple_end_to_stop(np.asarray(intervals.end, dtype=np.float64)),
        ]
    )


def tsgroup_unit_table(group: Any) -> pd.DataFrame | None:
    """Return a pynapple ``TsGroup``'s metadata as a unit table, or None.

    Rows are indexed by unit id (index named ``unit_id``). pynapple's ``rate``
    column is dropped: pynapple computes it as spike count over the group's
    time support, which is not the time an analysis observed, so it is not a
    property of the unit. A group with no other metadata gives None.

    Parameters
    ----------
    group : pynapple.TsGroup

    Returns
    -------
    pandas.DataFrame or None
        One row per unit, in group order.
    """
    metadata = getattr(group, "metadata", None)
    if metadata is None:
        return None
    table = metadata.drop(columns=["rate"], errors="ignore")
    if table.shape[1] == 0:
        return None
    table = table.copy()
    table.index = table.index.rename("unit_id")
    return table


#: Sample-clock timestamps: a pynapple object here is read as its ``.t``.
_TIME_ARGUMENTS = ("times", "frame_times", "true_times")
#: Sampled values: a pynapple object here is read as its ``.values``.
_VALUE_ARGUMENTS = ("positions", "true_positions", "headings")
#: Samples whose ``time_support`` becomes ``epochs`` when none are given.
_SUPPORT_ARGUMENTS = ("positions", "headings")


def _plain_sample(name: str, value: Any) -> Any:
    """Return a pynapple sample argument as the plain array it stands for."""
    if name in _TIME_ARGUMENTS:
        # np.asarray(tsd) gives the VALUES, not the timestamps.
        t = getattr(value, "t", None)
        return value if t is None else np.asarray(t, dtype=np.float64)
    if "positions" in name and hasattr(value, "columns"):
        refuse_permuted_spatial_columns(value.columns, fix=f"{name}={{frame}}")
    values = getattr(value, "values", None)
    return value if values is None else np.asarray(values)


def reads_pynapple_samples(func: Callable[_P, _R]) -> Callable[_P, _R]:
    """Read pynapple timestamps and samples as the arrays they stand for.

    Decorates a public function or method taking a sample-clock argument
    (``times``, ``frame_times``, ``true_times``) or sampled values
    (``positions``, ``true_positions``, ``headings``). When one of those is a
    pynapple object:

    - a time argument becomes the object's timestamps, ``.t``. ``np.asarray``
      of a ``Tsd`` gives its values, which would silently be read as times;
    - a values argument becomes ``.values``; a ``TsdFrame`` of positions
      whose columns are x/y/z in another order is refused, as by
      :func:`neurospatial.io.from_pynapple`;
    - if the function takes ``epochs`` and it is ``None``, the
      ``time_support`` of pynapple ``positions`` / ``headings`` (converted to
      half-open rows) becomes ``epochs``, so a recording gap shorter than
      ``max_gap`` is not counted as observed time. An explicit ``epochs=``
      wins; two samples with different supports are refused.

    Every other call runs the function unchanged. The signature, docstring
    and ``__wrapped__`` are the function's own.
    """
    signature = inspect.signature(func)
    names = list(signature.parameters)
    samples = [
        (name, names.index(name))
        for name in _TIME_ARGUMENTS + _VALUE_ARGUMENTS
        if name in names
    ]
    if not samples:
        raise TypeError(
            f"reads_pynapple_samples needs a times, positions or headings "
            f"parameter; {func.__qualname__} has {names}."
        )
    takes_epochs = "epochs" in names

    def given(name: str, position: int, args: Any, kwargs: Any) -> Any:
        return args[position] if position < len(args) else kwargs.get(name)

    @functools.wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        if not any(
            _is_pynapple_object(given(name, position, args, kwargs))
            for name, position in samples
        ):
            return func(*args, **kwargs)
        bound = signature.bind(*args, **kwargs)
        supports: dict[str, NDArray[np.float64]] = {}
        for name, _ in samples:
            value: Any = bound.arguments.get(name)
            if not _is_pynapple_object(value):
                continue
            support = getattr(value, "time_support", None)
            if name in _SUPPORT_ARGUMENTS and support is not None:
                supports[name] = interval_rows(support)
            bound.arguments[name] = _plain_sample(name, value)
        if takes_epochs and bound.arguments.get("epochs") is None and supports:
            bound.arguments["epochs"] = _shared_support(supports)
        return func(*bound.args, **bound.kwargs)

    wrapper.__reads_pynapple_samples__ = True  # type: ignore[attr-defined]
    return wrapper


def _shared_support(supports: dict[str, NDArray[np.float64]]) -> Any:
    """The one time support all pynapple samples share, else a refusal."""
    (first, rows), *others = supports.items()
    for name, other in others:
        if not np.array_equal(rows, other):
            raise ValueError(
                f"{first} and {name} have different pynapple time supports, so "
                "the epochs to analyze are ambiguous.\n"
                "Why: with epochs=None each pynapple sample's time_support is "
                "used as the epochs, and these disagree.\n"
                f"Fix: pass epochs= explicitly, e.g. "
                f"epochs={first}.time_support.intersect({name}.time_support)."
            )
    return rows
