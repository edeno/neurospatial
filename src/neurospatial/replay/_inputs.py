"""Checks of the inputs replay functions share.

Event windows, a decoded event's posterior rows and the times of those rows.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error

# Why callers need the time of every posterior row.
_FIT_WHY = "a fit along time needs the time of every posterior row"
_CORRELATION_WHY = "the weighted correlation pairs each row's time with its positions"
_STEP_WHY = (
    "jumps and the duration compare consecutive time bins, so the rows must "
    "be in time order"
)


def _checked_intervals(
    intervals: Any,
    operation: str,
    *,
    name: str = "intervals",
    allow_empty: bool = False,
    example: str | None = None,
) -> NDArray[np.float64]:
    """Event windows as finite ``(start, stop)`` rows with ``start < stop``.

    Parameters
    ----------
    intervals : array_like, shape (n_events, 2)
        The windows, in seconds.
    operation, name : str
        The calling function and argument, named in the error.
    allow_empty : bool, default=False
        Return no rows, shape (0, 2), for an empty input instead of refusing it.
    example : str, optional
        Where such an array comes from, offered in the error's fix.

    Returns
    -------
    NDArray[np.float64], shape (n_events, 2)
        The windows, in input order.
    """
    windows = np.asarray(intervals, dtype=np.float64)
    if allow_empty and windows.size == 0:
        return windows.reshape(0, 2)
    if (
        windows.ndim != 2
        or windows.shape[1] != 2
        or len(windows) == 0
        or not np.all(np.isfinite(windows))
        or np.any(windows[:, 1] <= windows[:, 0])
    ):
        raise ValueError(
            _format_error(
                f"{operation}: {name} must be finite (start, stop) rows in seconds "
                f"with start < stop, got shape {windows.shape}.",
                why="each row is one event, and spikes at start <= t < stop "
                "belong to it",
                fix="pass an array of shape (n_events, 2)"
                + (f", such as {example}" if example else " in seconds"),
            )
        )
    return windows


def _posterior_times(
    times: Any,
    n_time_bins: int,
    operation: str,
    *,
    why: str,
    distinct: bool = False,
    increasing: bool = False,
) -> NDArray[np.float64]:
    """One finite time per posterior row, optionally ordered.

    Parameters
    ----------
    times : array_like, shape (n_time_bins,)
        Time of each posterior row, in seconds.
    n_time_bins : int
        Number of posterior rows.
    operation : str
        The calling function, for the refusal message.
    why : str
        Why the caller needs the time of every row in order, for the
        refusal of non-finite or unordered times.
    distinct : bool, default=False
        Require at least two distinct times, for a fit along time.
    increasing : bool, default=False
        Require strictly increasing times.

    Returns
    -------
    NDArray[np.float64], shape (n_time_bins,)
        ``times`` as float64.

    Raises
    ------
    ValueError
        If ``times`` does not hold one entry per row, holds NaN or Inf, or
        breaks the ``distinct`` or ``increasing`` rule.
    """
    values: NDArray[np.float64] = np.asarray(times, dtype=np.float64)
    if values.shape != (n_time_bins,):
        raise ValueError(
            _format_error(
                f"{operation}: times has {values.size} entries in shape "
                f"{values.shape}, but posterior has {n_time_bins} rows.",
                why="each posterior row is the decode at one time",
                fix="pass one finite time per posterior row, in seconds, with "
                "shape (n_time_bins,); drop the times of any rows you dropped",
            )
        )
    finite = np.isfinite(values)
    if not finite.all():
        raise ValueError(
            _format_error(
                f"{operation}: times contains NaN or Inf at "
                f"{int(np.sum(~finite))} of {n_time_bins} rows.",
                why=why,
                fix="drop those rows from both posterior and times",
            )
        )
    if distinct and np.ptp(values) == 0:
        raise ValueError(
            _format_error(
                f"{operation} needs at least two distinct times, but every "
                "time bin has the same time.",
                why="a line through identical times has no slope",
                fix="pass the center time of each posterior row, in seconds",
            )
        )
    if increasing and np.any(np.diff(values) <= 0):
        raise ValueError(
            _format_error(
                f"{operation}: times must be strictly increasing from one "
                "posterior row to the next.",
                why=why,
                fix="sort the posterior rows and their times together, order "
                "= np.argsort(times); posterior, times = posterior[order], "
                "times[order], and drop rows with duplicate times",
            )
        )
    return values


def _probability_rows(
    posterior: Any,
    operation: str,
    *,
    sums: Literal["one", "one_or_zero", "at_most_one"],
) -> NDArray[np.float64]:
    """Refuse rows that are not a probability mass over the bins.

    Parameters
    ----------
    posterior : array_like, shape (n_time_bins, n_bins)
        Posterior mass per bin.
    operation : str
        The calling function, for the refusal message.
    sums : {"one", "one_or_zero", "at_most_one"}
        The row-sum rule: 1; 1 or exactly 0 (the time bins without spikes
        that ``empty_time_bins="zero"`` zeroes); or in (0, 1], for a
        template's share of a joint decode. Sums are compared with 1 within
        the tolerance a ``DecodingResult`` accepts (``1e-6``, which covers
        float32 rounding).

    Returns
    -------
    NDArray[np.float64], shape (n_time_bins, n_bins)
        The posterior as float64.

    Raises
    ------
    ValueError
        If a row holds NaN or Inf, a negative entry, or breaks ``sums``.
    """
    from neurospatial.decoding import _ROW_SUM_TOLERANCE

    values = np.asarray(posterior, dtype=np.float64)
    finite = np.isfinite(values).all(axis=1)
    negative = finite & (values < 0).any(axis=1)
    checked = finite & ~negative
    row_sums = values.sum(axis=1)
    problems = []
    if sums == "at_most_one":
        empty = checked & (row_sums <= 0)
        excess = checked & (row_sums > 1.0 + _ROW_SUM_TOLERANCE)
        bad = ~finite | negative | empty | excess
    else:
        unnormalized = checked & (np.abs(row_sums - 1.0) > _ROW_SUM_TOLERANCE)
        if sums == "one_or_zero":
            unnormalized &= row_sums != 0
        bad = ~finite | negative | unnormalized
    if not np.any(bad):
        return values
    if np.any(~finite):
        problems.append(f"{int(np.sum(~finite))} contain NaN or Inf")
    if np.any(negative):
        problems.append(f"{int(np.sum(negative))} have negative entries")
    if sums == "at_most_one":
        if np.any(empty):
            problems.append(f"{int(np.sum(empty))} have no mass")
        if np.any(excess):
            problems.append(
                f"{int(np.sum(excess))} sum to more than 1 (largest "
                f"{float(np.max(row_sums[excess])):.6g})"
            )
        share = ", or one template's share of one,"
        why = (
            "a row without mass, or with NaN, has no decoded position, and a "
            "row summing to more than 1 is not a probability"
        )
        fix = (
            "drop undecodable time bins together with their times, keep = "
            "np.isfinite(posterior).all(axis=1) & (posterior.sum(axis=1) > 0); "
            "posterior, times = posterior[keep], times[keep]"
        )
    else:
        if np.any(unnormalized):
            worst = row_sums[unnormalized][
                np.argmax(np.abs(row_sums[unnormalized] - 1.0))
            ]
            problems.append(
                f"{int(np.sum(unnormalized))} do not sum to 1 (worst row sums "
                f"to {worst:.6g}, |row sum - 1| = {abs(worst - 1.0):.2g})"
            )
        share = ","
        why = (
            "a row with NaN has no decoded position but would be read as the "
            "track start (bin 0), and a row that does not sum to 1 weighs its "
            "time bin more or less than the others (a sampled fit draws the "
            "missing mass at the track start)"
        )
        fix = (
            "drop undecodable time bins together with their times, keep = "
            "np.isfinite(posterior).all(axis=1); posterior, times = "
            "posterior[keep], times[keep]; renormalize rows that only lack "
            "mass with posterior / posterior.sum(axis=1, keepdims=True)"
        )
    raise ValueError(
        _format_error(
            f"{operation} needs each posterior row to be a probability mass "
            f"over the bins{share} but {int(np.sum(bad))} of {len(values)} "
            f"rows are not: {'; '.join(problems)}.",
            why=why,
            fix=fix,
        )
    )
