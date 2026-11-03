"""The automatic checks every dataset session passes before it is returned.

Two checks refuse, because a failure means the arrays would be wrong:

- **clock**: spikes, events and epochs are on the tracking clock;
- **units**: the position span and sampling rate match what the dataset
  states.

A refusal raises :class:`~neurospatial.datasets.DatasetCheckError` unless the
caller accepts it with ``accept_failed_checks``; an accepted failure is kept
in ``rec.failed_checks`` and heads ``rec.watch_outs``. The other checks only
describe: samples at exactly the origin, held and fast tracking samples, and
units silent in the analysed epochs go into ``rec.watch_outs`` with their
numbers. No check changes an array.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import InputError, _format_error
from neurospatial.datasets._errors import REFUSING_CHECKS, DatasetCheckError
from neurospatial.datasets._recording import Recording

__all__ = ["Stated", "run_checks"]

#: How far, as a share of a span's length, a time may lie outside the span
#: and still count as within it. The errors the clock check guards against
#: (milliseconds read as seconds, a second clock with another zero) put
#: times off by about the session's length or more; acquisition edges
#: (ephys running a few seconds past the camera, an epoch starting at 0 s
#: before the first spike) differ by a small share of it.
CLOCK_SLACK = 0.01
#: Largest relative difference between the measured and stated sampling rate.
RATE_TOLERANCE = 0.01
#: Share of samples exactly at the origin above which a watch-out is given.
ZERO_SHARE = 0.001
#: Repeats of the sample before from which a run of samples counts as held:
#: 10 repeats, so 11 identical samples.
MIN_HELD_REPEATS = 10


@dataclass(frozen=True)
class Stated:
    """What a dataset states about its sessions, to check the arrays against.

    Parameters
    ----------
    sampling_rate : float
        The stated position sampling rate in Hz.
    position_span : tuple of float or None, default None
        ``(low, high)``: the range, in ``position_units``, that the largest
        per-axis span of the tracked positions must lie in, from the stated
        apparatus. Required for a recording with positions.
    speed_limit : float or None, default None
        The fastest believable speed, in ``position_units`` per second,
        where a source states one; None counts no fast steps.
    speed_limit_source : str, default ""
        Where ``speed_limit`` is stated. Required with ``speed_limit``.

    Raises
    ------
    InputError
        If a number is not finite and positive, ``position_span`` is not a
        ``(low, high)`` pair with ``low <= high``, or ``speed_limit`` has no
        source.
    """

    sampling_rate: float
    position_span: tuple[float, float] | None = None
    speed_limit: float | None = None
    speed_limit_source: str = ""

    def __post_init__(self) -> None:
        problems = []
        if not _positive(self.sampling_rate):
            problems.append(f"sampling_rate {self.sampling_rate!r} is not positive")
        if self.position_span is not None:
            span = tuple(self.position_span)
            if not (
                len(span) == 2
                and all(_positive(value, zero=True) for value in span)
                and span[0] <= span[1]
            ):
                problems.append(
                    f"position_span {self.position_span!r} is not (low, high) "
                    "with 0 <= low <= high"
                )
            else:
                object.__setattr__(
                    self, "position_span", (float(span[0]), float(span[1]))
                )
        if self.speed_limit is not None:
            if not _positive(self.speed_limit):
                problems.append(f"speed_limit {self.speed_limit!r} is not positive")
            if not self.speed_limit_source.strip():
                problems.append("speed_limit has no speed_limit_source")
        if problems:
            raise InputError(
                _format_error(
                    f"Stated facts are invalid: {'; '.join(problems)}.",
                    why="the clock and units checks compare each session with "
                    "these numbers, and a speed limit is used only where a "
                    "source states it.",
                    fix="give the sampling rate in Hz, the expected span as "
                    "(low, high) in the positions' unit, and a speed limit only "
                    "with the source that states it.",
                )
            )


def _positive(value: Any, *, zero: bool = False) -> bool:
    """Whether ``value`` is a finite real number above 0 (or at least 0)."""
    if isinstance(value, bool) or not isinstance(value, int | float | np.number):
        return False
    return bool(np.isfinite(value) and (value >= 0 if zero else value > 0))


def run_checks(
    rec: Recording,
    *,
    stated: Stated,
    accept_failed_checks: tuple[str, ...] = (),
) -> Recording:
    """Check a session as its loader built it; refuse or describe what fails.

    Parameters
    ----------
    rec : Recording
        The session, with the dataset's own ``watch_outs`` and no
        ``failed_checks``.
    stated : Stated
        What the dataset states about the session.
    accept_failed_checks : tuple of str, default ()
        Refusing checks (``"clock"``, ``"units"``) to load through. A named
        check that passes changes nothing.

    Returns
    -------
    Recording
        ``rec`` with ``failed_checks`` set to the accepted failures and
        ``watch_outs`` extended: accepted failures first, then the dataset's
        own entries, then the warning checks' findings. The arrays are
        unchanged.

    Raises
    ------
    DatasetCheckError
        If a refusing check fails and is not in ``accept_failed_checks``.
    InputError
        If ``accept_failed_checks`` names an unknown check, or ``stated``
        has no ``position_span`` for a recording with positions.
    """
    accepted = _accepted(accept_failed_checks)
    if rec.positions is not None and stated.position_span is None:
        raise InputError(
            _format_error(
                f"{rec.dataset}/{rec.session} has positions, but the stated facts "
                "give no position_span.",
                why="the units check compares the tracked span with the stated "
                "apparatus to catch a unit or scale error.",
                fix="state the range the largest per-axis span must lie in, "
                "for example Stated(..., position_span=(80.0, 120.0)) for a "
                "1 m box in cm.",
            )
        )
    failures = {
        name: finding
        for name, finding in (
            ("clock", _clock_finding(rec)),
            ("units", _units_finding(rec, stated)),
        )
        if finding is not None
    }
    if refused := {
        name: text for name, text in failures.items() if name not in accepted
    }:
        raise DatasetCheckError(rec.dataset, rec.session, refused)
    flagged = [
        f"failed the {name} check, loaded with accept_failed_checks: {text}"
        for name, text in failures.items()
    ]
    warnings = [
        finding
        for finding in (
            _origin_finding(rec),
            _held_and_fast_finding(rec, stated),
            _silent_units_finding(rec),
        )
        if finding is not None
    ]
    return dataclasses.replace(
        rec,
        watch_outs=(*flagged, *rec.watch_outs, *warnings),
        failed_checks=tuple(failures),
    )


def _accepted(accept_failed_checks: Any) -> set[str]:
    """The accepted check names, refusing a lone string or an unknown name."""
    names = (
        [accept_failed_checks]
        if isinstance(accept_failed_checks, str)
        else list(accept_failed_checks)
    )
    unknown = [name for name in names if name not in REFUSING_CHECKS]
    if isinstance(accept_failed_checks, str) or unknown:
        raise InputError(
            _format_error(
                f"accept_failed_checks={accept_failed_checks!r} names "
                f"{unknown or names}, not a tuple of refusing checks.",
                why=f"only the {' and '.join(map(repr, REFUSING_CHECKS))} checks "
                "refuse a load; the others only add watch-outs.",
                fix="pass a tuple of check names, for example "
                "accept_failed_checks=('clock',).",
            )
        )
    return set(names)


def _clock_finding(rec: Recording) -> str | None:
    """What disagrees with the tracking clock, with numbers; None if nothing."""
    times = rec.times
    first, last = float(times[0]), float(times[-1])
    problems = []
    rows = np.concatenate(list(rec.epochs.values())) if rec.epochs else np.empty((0, 2))
    samples_in = np.searchsorted(times, rows[:, 1]) - np.searchsorted(times, rows[:, 0])
    untracked = rows[samples_in == 0]
    if len(rows) and not np.any(samples_in > 0):
        problems.append(
            f"no epoch holds a tracking sample: epochs run from {rows.min():.6g} to "
            f"{rows.max():.6g} s, tracking from {first:.6g} to {last:.6g} s"
        )
    trains = [train for train in rec.spike_times if train.size]
    spikes = np.concatenate(trains) if trains else np.empty(0)
    slack = CLOCK_SLACK * (last - first)
    near = (spikes >= first - slack) & (spikes <= last + slack)
    outside = ~(near | _within_any(spikes, untracked))
    if n_outside := int(outside.sum()):
        problems.append(
            f"{n_outside} of {spikes.size} spikes ({n_outside / spikes.size:.2%}) lie "
            f"outside the tracked span {first:.6g} to {last:.6g} s (give or take "
            f"{slack:.3g} s) and outside every untracked epoch; spikes run from "
            f"{spikes.min():.6g} to {spikes.max():.6g} s"
        )
    start = min(first, float(spikes.min())) if spikes.size else first
    stop = max(last, float(spikes.max())) if spikes.size else last
    margin = CLOCK_SLACK * (stop - start)
    span = f"the recording ({start:.6g} to {stop:.6g} s, give or take {margin:.3g} s)"
    for name, table in rec.epochs.items():
        beyond = (table[:, 0] < start - margin) | (table[:, 1] > stop + margin)
        if n_beyond := int(beyond.sum()):
            row = table[np.argmax(beyond)]
            problems.append(
                f"epochs[{name!r}] has {_count(n_beyond, 'row')} of {len(table)} "
                f"outside {span}, the first [{row[0]:.6g}, {row[1]:.6g}] s"
            )
        ordered = table[np.argsort(table[:, 0], kind="stable")]
        overlaps = ordered[1:, 0] < ordered[:-1, 1]
        if n_overlaps := int(overlaps.sum()):
            at = int(np.argmax(overlaps))
            problems.append(
                f"epochs[{name!r}] has {_count(n_overlaps, 'row')} overlapping the "
                f"row before, the first [{ordered[at + 1, 0]:.6g}, "
                f"{ordered[at + 1, 1]:.6g}] s inside [{ordered[at, 0]:.6g}, "
                f"{ordered[at, 1]:.6g}] s"
            )
    for name, values in rec.events.items():
        outer = (values < start - margin) | (values > stop + margin)
        if n_beyond := int(outer.reshape(len(values), -1).any(axis=1).sum()):
            problems.append(
                f"events[{name!r}] has {_count(n_beyond, 'event')} of {len(values)} "
                f"outside {span}; its times run from {values.min():.6g} to "
                f"{values.max():.6g} s"
            )
    if not problems:
        return None
    return (
        "; ".join(problems)
        + ". Spike, event and epoch times must be seconds on the clock of times: "
        "check each one's unit (milliseconds or samples read as seconds) and its "
        "zero (a second clock)."
    )


def _within_any(
    values: NDArray[np.float64], rows: NDArray[np.float64]
) -> NDArray[np.bool_]:
    """Whether each value lies in any closed ``[start, stop]`` row, shape (n,)."""
    if len(rows) == 0 or values.size == 0:
        return np.zeros(values.size, dtype=bool)
    rows = rows[np.argsort(rows[:, 0], kind="stable")]
    reach = np.maximum.accumulate(rows[:, 1])
    first = np.flatnonzero(np.r_[True, rows[1:, 0] > reach[:-1]])
    starts = rows[first, 0]
    stops = np.maximum.reduceat(rows[:, 1], first)
    index = np.searchsorted(starts, values, side="right") - 1
    inside: NDArray[np.bool_] = (index >= 0) & (values <= stops[np.maximum(index, 0)])
    return inside


def _units_finding(rec: Recording, stated: Stated) -> str | None:
    """How the span or sampling rate disagrees with the stated ones; None if not."""
    problems = []
    rate = rec.sampling_rate
    if abs(rate / stated.sampling_rate - 1) > RATE_TOLERANCE:
        problems.append(
            f"the sampling rate is {rate:.6g} Hz (one over the median interval), "
            f"not the stated {stated.sampling_rate:g} Hz within "
            f"{RATE_TOLERANCE:.0%}"
        )
    if rec.positions is not None and stated.position_span is not None:
        low, high = stated.position_span
        units = rec.position_units
        tracked = rec.positions[np.all(np.isfinite(rec.positions), axis=1)]
        if len(tracked) == 0:
            problems.append("no position sample is tracked, so the span is unknown")
        else:
            spans = tracked.max(axis=0) - tracked.min(axis=0)
            axis = int(np.argmax(spans))
            if not low <= spans[axis] <= high:
                problems.append(
                    f"the largest position span is {spans[axis]:.4g} {units} (axis "
                    f"{axis}), outside the stated {low:g} to {high:g} {units}"
                )
    if not problems:
        return None
    return (
        "; ".join(problems)
        + ". Check the position unit and scale (cm, m or pixels, and any "
        "calibration) and the sampling rate against the file and the paper."
    )


def _origin_finding(rec: Recording) -> str | None:
    """Samples exactly at the origin, when more than ``ZERO_SHARE`` of all."""
    if rec.positions is None:
        return None
    at_origin = np.all(rec.positions == 0, axis=1)
    share = float(at_origin.mean())
    if share <= ZERO_SHARE:
        return None
    origin = "(" + ", ".join(["0"] * rec.n_dims) + ")"
    return (
        f"{int(at_origin.sum())} position samples ({share:.2%}) are exactly at "
        f"{origin}: likely lost tracking stored as zeros. They are kept as "
        "stored; set them to NaN before analysis if the animal cannot be at "
        "the origin."
    )


def _held_and_fast_finding(rec: Recording, stated: Stated) -> str | None:
    """Held runs and, with a stated speed limit, fast steps; None if neither."""
    if rec.positions is None:
        return None
    held, runs, seconds, longest = _held_samples(rec.times, rec.positions)
    fast = (
        None
        if stated.speed_limit is None
        else _fast_samples(rec.times, rec.positions, held, stated.speed_limit)
    )
    n_fast = 0 if fast is None else int(fast.sum())
    if runs == 0 and n_fast == 0:
        return None
    units = rec.position_units
    found = []
    if runs:
        found.append(
            f"{_count(runs, 'run')} of {MIN_HELD_REPEATS + 1} or more identical "
            f"position samples ({MIN_HELD_REPEATS} or more repeats of the sample "
            f"before; {_count(int(held.sum()), 'repeat')}, {held.mean():.2%} of "
            f"samples, {seconds:.1f} s in all, the longest {longest:.1f} s)"
        )
    if n_fast:
        found.append(
            f"{_count(n_fast, 'sample')} farther from the last believable sample "
            f"before them than {stated.speed_limit:g} {units}/s allows "
            f"({stated.speed_limit_source})"
        )
    text = "; ".join(found) + ". Positions are kept as stored."
    if runs:
        text += (
            " A held run is lost tracking if the tracker repeated its last fix, or "
            "the animal holding still; check before analysing stops (wells, "
            "ripples)."
        )
    if n_fast:
        text += (
            " A fast sample is a tracking glitch or a jump the tracker bridged "
            "after losing the animal."
        )
    if stated.speed_limit is None:
        text += (
            " No speed limit is stated for this dataset, so fast steps are not counted."
        )
    return text


def _held_samples(
    times: NDArray[np.float64], positions: NDArray[np.float64]
) -> tuple[NDArray[np.bool_], int, float, float]:
    """Samples repeating the one before, in runs of ``MIN_HELD_REPEATS`` or more.

    Returns
    -------
    held : ndarray of bool, shape (n_time,)
        True at each repeat in a long run; the run's first sample, the last
        fix before it, is not held.
    runs : int
        The number of long runs.
    seconds, longest : float
        Their total and longest duration, from the run's first sample to its
        last repeat.
    """
    repeat = np.r_[False, np.all(np.diff(positions, axis=0) == 0, axis=1)]
    edges = np.diff(np.r_[0, repeat.astype(np.int8), 0])
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    long = stops - starts >= MIN_HELD_REPEATS
    starts, stops = starts[long], stops[long]
    counts = np.zeros(len(positions) + 1, dtype=np.int64)
    np.add.at(counts, starts, 1)
    np.add.at(counts, stops, -1)
    held = np.cumsum(counts[:-1]) > 0
    durations = times[stops - 1] - times[starts - 1]
    longest = float(durations.max()) if durations.size else 0.0
    return held, int(long.sum()), float(durations.sum()), longest


def _fast_samples(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    held: NDArray[np.bool_],
    speed_limit: float,
) -> NDArray[np.bool_]:
    """Samples farther from the last believable sample than ``speed_limit`` allows.

    One sequential pass over the finite samples that are not held. Two
    samples agree when their distance over the time between them is at most
    ``speed_limit``. The pass starts at the first sample that agrees with
    each of the next two, which agree with each other: a bad first sample
    would otherwise anchor the pass and count every later sample until the
    animal came back near it, and three samples because a glitch held for
    two frames agrees with itself. Samples before the start are fast when
    they do not agree with it; with no three agreeing, the pass starts at the
    first finite sample. Each later sample that agrees with the anchor
    becomes the anchor; one that does not is fast. NaN and held samples do
    not move the anchor, so a one-frame glitch counts once (not its return
    too) and every sample of an excursion counts. The pass is sequential
    because each test depends on the last sample kept.

    Returns
    -------
    ndarray of bool, shape (n_time,)
    """
    fast = np.zeros(len(positions), dtype=bool)
    order = np.flatnonzero(np.all(np.isfinite(positions), axis=1) & ~held).tolist()
    if not order:
        return fast
    points, clock = positions.tolist(), times.tolist()

    def agree(a: int, b: int) -> bool:
        reach = speed_limit * abs(clock[b] - clock[a])
        return bool(math.dist(points[a], points[b]) <= reach)

    first = next(
        (
            i
            for i in range(len(order) - 2)
            if agree(order[i], order[i + 1])
            and agree(order[i], order[i + 2])
            and agree(order[i + 1], order[i + 2])
        ),
        0,
    )
    anchor = order[first]
    for k in order[:first]:
        fast[k] = not agree(k, anchor)
    for k in order[first + 1 :]:
        if agree(anchor, k):
            anchor = k
        else:
            fast[k] = True
    return fast


def _count(n: int, noun: str) -> str:
    """``"1 run"``, ``"2 runs"``: a count and its noun."""
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _silent_units_finding(rec: Recording) -> str | None:
    """Units with no spikes in the analysed epochs; None if every unit fires."""
    fires = rec.units["fires_in_epochs"].to_numpy()
    n_silent = int((~fires).sum())
    if n_silent == 0:
        return None
    where = (
        "params['analysis_epochs']"
        if "analysis_epochs" in rec.params
        else "the tracked span"
    )
    return (
        f"{n_silent} of {rec.n_units} units have no spikes in the analysed epochs "
        f"({where}); they are kept, as the file lists them, and are "
        "units['fires_in_epochs'] False."
    )
