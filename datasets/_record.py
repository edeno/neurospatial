"""The plain-array record every loader returns, its validation and expectations.

``Recording`` holds a session in the units and conventions neurospatial expects:
seconds, radians with 0 = East and +pi/2 = North, NaN for missing samples. It
is internal to the survey harness, not a proposal for a neurospatial object.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

POSITION_UNITS = {"cm", "m", "pixel", "arbitrary"}


@dataclass(frozen=True)
class Recording:
    """One session as plain arrays.

    Attributes
    ----------
    dataset, session, source, license : str
        Registry key, session identifier, DOI or URL of the exact files loaded,
        and the data license ("unstated" if none).
    times : ndarray, shape (n_time,), optional
        Tracking sample times in seconds, strictly increasing.
    positions : ndarray, shape (n_time, n_dims), optional
        Tracked position in ``position_units``; NaN where tracking was lost.
    position_units : str, optional
        "cm", "m", "pixel" or "arbitrary".
    headings : ndarray, shape (n_time,), optional
        Allocentric head direction in radians, wrapped to [-pi, pi].
    spike_times : tuple of ndarray, optional
        One sorted array of spike times (s) per unit.
    unit_ids : ndarray, shape (n_units,), optional
        Unique unit identifiers, one per spike train.
    activity, activity_times, activity_kind : optional
        Imaging activity (n_frames, n_cells), its frame times (s) and kind.
    epochs : mapping of str to ndarray, shape (n, 2)
        Named [start, stop) intervals in seconds.
    events : mapping of str to ndarray, shape (n,) or (n, 2)
        Named event times or intervals in seconds.
    geometry : mapping
        Arena facts stated by the source; inferred values are prefixed ``inferred_``.
    params : mapping
        Analysis choices the driver uses (``bin_size`` in ``position_units``,
        ``dt`` in s, and optional ``min_speed``, ``analysis_epochs``,
        ``train_epochs``, ``test_epochs``, ``replay_dt``, ``nwb_source``).
    conversions : tuple of str
        Every transformation the loader applied, one line each.
    watch_outs : tuple of str
        The pitfalls ``neurospatial.datasets`` reports for a session it
        loaded (``from_public``); empty for the survey's own loaders.
    """

    dataset: str
    session: str
    source: str
    license: str
    times: NDArray[np.float64] | None = None
    positions: NDArray[np.float64] | None = None
    position_units: str | None = None
    headings: NDArray[np.float64] | None = None
    spike_times: tuple[NDArray[np.float64], ...] | None = None
    unit_ids: NDArray[Any] | None = None
    activity: NDArray[np.float64] | None = None
    activity_times: NDArray[np.float64] | None = None
    activity_kind: str | None = None
    epochs: Mapping[str, NDArray[np.float64]] = field(default_factory=dict)
    events: Mapping[str, NDArray[np.float64]] = field(default_factory=dict)
    geometry: Mapping[str, Any] = field(default_factory=dict)
    params: Mapping[str, Any] = field(default_factory=dict)
    conversions: tuple[str, ...] = ()
    watch_outs: tuple[str, ...] = ()


@dataclass(frozen=True)
class SurveyOnly:
    """A field the survey adds to a public dataset's session, and why.

    ``from_public`` records each one in ``conversions`` as a line starting
    ``survey-only``, so a result can be traced to what the public loader
    gives and what the survey chose on top of it.
    """

    value: Any
    why: str


def from_public(
    rec: Any,
    *,
    key: str,
    params: Mapping[str, SurveyOnly] | None = None,
    events: Mapping[str, SurveyOnly] | None = None,
    geometry: Mapping[str, SurveyOnly] | None = None,
    activity_kind: str | None = None,
) -> Recording:
    """The survey's ``Recording`` of a session ``neurospatial.datasets.load`` returned.

    Where the two records name or shape a field differently, it is mapped
    here, in one place:

    - ``dataset`` is the survey ``key`` (the results file's name);
      ``source`` is the file's URL (``rec.provenance.url``);
    - ``unit_ids`` is the index of ``rec.units``;
    - ``params["linearized"]`` is ``rec.linearized``, where the file has one;
    - ``geometry["environment"]`` returns ``rec.env`` when it is declared
      from the stated apparatus; an environment inferred from the samples is
      left out, so the driver builds its own from the samples.
      ``geometry["tracked_point"]`` is ``rec.tracking``;
    - ``activity_kind``, which the public record does not hold, is given here
      for imaging activity (``"dff"``, ``"binned_rate"``);
    - each ``Conversion`` becomes one line, ``"<text> [<basis>: <evidence>]"``;
    - ``watch_outs`` are kept as they are.

    Arrays are copied, writable. ``params``, ``events`` and ``geometry`` add
    the survey's own fields to the public ones, each with a ``survey-only``
    line in ``conversions`` saying why, and whether it replaces a public
    value.

    Parameters
    ----------
    rec : neurospatial.datasets.Recording
    key : str
        The survey's dataset key, for example ``"j16_23020757"``.
    params, events, geometry : mapping of str to SurveyOnly, optional
    activity_kind : str, optional
        What ``rec.activity`` holds; required when it is not None.

    Returns
    -------
    Recording

    Raises
    ------
    ValueError
        If the environment is declared and ``geometry`` gives no
        ``declared_from`` (the environment fidelity study needs it to tell an
        independent reference from the geometry the track was declared
        from), or ``rec.activity`` is given without ``activity_kind``.
    """

    def copied(value: Any) -> Any:
        return np.array(value) if isinstance(value, np.ndarray) else value

    def merged(
        field_name: str,
        public: Mapping[str, Any],
        extra: Mapping[str, SurveyOnly] | None,
    ) -> tuple[dict[str, Any], list[str]]:
        out = {name: copied(value) for name, value in public.items()}
        lines = []
        for name, added in (extra or {}).items():
            replaced = " (replaces the public loader's)" if name in out else ""
            out[name] = added.value
            lines.append(f"survey-only {field_name}[{name!r}]{replaced}: {added.why}")
        return out, lines

    declared = rec.env is not None and rec.env_kind == "declared"
    if declared and "declared_from" not in (geometry or {}):
        msg = (
            f"{key}: the environment is declared, so geometry needs"
            " 'declared_from' (one of _tracks.PLACED_BY), which the environment"
            " fidelity study judges the reference's independence by"
        )
        raise ValueError(msg)
    if rec.activity is not None and activity_kind is None:
        msg = f"{key}: rec.activity is given, so activity_kind is needed"
        raise ValueError(msg)
    public_params = dict(rec.params)
    if rec.linearized is not None:
        public_params["linearized"] = rec.linearized
    # A public Recording always says what was tracked (it refuses an empty
    # ``tracking``), so ``tracked_point`` is always a stated string.
    public_geometry: dict[str, Any] = {"tracked_point": rec.tracking}
    if declared:
        env = rec.env
        public_geometry["environment"] = lambda: env
    all_params, param_lines = merged("params", public_params, params)
    all_events, event_lines = merged("events", rec.events, events)
    all_geometry, geometry_lines = merged("geometry", public_geometry, geometry)
    return Recording(
        dataset=key,
        session=rec.session,
        source=rec.provenance.url,
        license=rec.license,
        times=copied(rec.times),
        positions=copied(rec.positions),
        position_units=rec.position_units,
        headings=copied(rec.headings),
        spike_times=tuple(np.array(train) for train in rec.spike_times),
        unit_ids=rec.units.index.to_numpy(),
        activity=copied(rec.activity),
        activity_times=copied(rec.activity_times),
        activity_kind=activity_kind,
        epochs={name: np.array(table) for name, table in rec.epochs.items()},
        events=all_events,
        geometry=all_geometry,
        params=all_params,
        conversions=(
            *(f"{c.text} [{c.basis}: {c.evidence}]" for c in rec.conversions),
            *param_lines,
            *event_lines,
            *geometry_lines,
        ),
        watch_outs=tuple(rec.watch_outs),
    )


def public_file(
    dataset: str, session: str, *, cache: Path | None = None, stream: bool = False
) -> str:
    """The file ``neurospatial.datasets.load`` reads for a session, for ``nwb_source``.

    Parameters
    ----------
    dataset, session : str
        The public dataset id and the archive's session id.
    cache : Path, optional
        The cache directory, as for ``neurospatial.datasets.load``.
    stream : bool, default False
        Give the URL the session is read from by byte range (a DANDI asset)
        instead of the cached file.

    Returns
    -------
    str
        The cached, checksum-verified path, or the byte-range URL after
        DANDI's published SHA-256 is checked.
    """
    from neurospatial.datasets._load import _session_entry, _stream_url, fetch_session

    entry = _session_entry(dataset, session)
    return _stream_url(entry) if stream else str(fetch_session(entry, cache=cache))


def _is_interval_table(table: NDArray[np.float64]) -> bool:
    """Whether ``table`` is finite (n, 2) [start, stop) rows with stop >= start."""
    return bool(
        table.ndim == 2
        and table.shape[1] == 2
        and np.isfinite(table).all()
        and np.all(table[:, 1] >= table[:, 0])
    )


def validate_recording(rec: Recording) -> None:
    """Raise ValueError listing every contract violation in ``rec``."""
    problems: list[str] = []
    tracked = rec.positions is not None or rec.headings is not None
    if (rec.times is None) == tracked:
        problems.append("times must be set exactly when positions or headings are")
    if rec.times is not None and (
        rec.times.ndim != 1 or not np.all(np.diff(rec.times) > 0)
    ):
        problems.append("times must be 1-D and strictly increasing")
    if rec.positions is not None:
        n_time = None if rec.times is None else rec.times.shape[0]
        if rec.positions.ndim != 2 or rec.positions.shape[0] != n_time:
            problems.append(
                f"positions shape {rec.positions.shape} does not match {n_time} times"
            )
        if rec.position_units not in POSITION_UNITS:
            problems.append(
                f"position_units {rec.position_units!r} not in {sorted(POSITION_UNITS)}"
            )
        required = {"bin_size"} | ({"dt"} if rec.spike_times is not None else set())
        if missing := sorted(required - set(rec.params)):
            problems.append(f"params missing {missing}")
    if rec.headings is not None:
        if rec.times is None or rec.headings.shape != rec.times.shape:
            problems.append("headings must have one value per time")
        finite = rec.headings[np.isfinite(rec.headings)]
        if finite.size and (finite.min() < -np.pi or finite.max() > np.pi):
            problems.append("headings must be radians in [-pi, pi]")
    if rec.spike_times is not None:
        if rec.unit_ids is None or len(rec.unit_ids) != len(rec.spike_times):
            problems.append("unit_ids must have one id per spike train")
        elif len(np.unique(rec.unit_ids)) != len(rec.unit_ids):
            problems.append("unit_ids must be unique")
        for index, train in enumerate(rec.spike_times):
            if not np.all(np.isfinite(train)) or np.any(np.diff(train) < 0):
                problems.append(f"spike train {index} must be finite and sorted")
    if (rec.activity is None) != (rec.activity_times is None):
        problems.append("activity and activity_times must be set together")
    elif (
        rec.activity is not None
        and rec.activity_times is not None
        and rec.activity.shape[0] != rec.activity_times.shape[0]
    ):
        problems.append("activity must have one row per activity time")
    for name, table in rec.epochs.items():
        if not _is_interval_table(table):
            problems.append(
                f"epochs[{name!r}] must be finite (n, 2) rows with stop >= start"
            )
    tables = {}
    for name in ("train_epochs", "test_epochs", "analysis_epochs"):
        if name not in rec.params:
            continue
        try:
            table = np.asarray(rec.params[name], dtype=float)
        except (TypeError, ValueError):
            table = np.empty(0)
        if not _is_interval_table(table):
            problems.append(
                f"params[{name!r}] must be finite (n, 2) rows with stop >= start"
            )
        elif name != "analysis_epochs":
            tables[name] = table
            if np.any(np.diff(table[:, 0]) < 0) or np.any(table[1:, 0] < table[:-1, 1]):
                problems.append(f"params[{name!r}] must be sorted, without overlaps")
    if len(tables) == 2:
        both = intersect_intervals(tables["train_epochs"], tables["test_epochs"])
        if len(both):
            problems.append(
                f"params['train_epochs'] and params['test_epochs'] overlap"
                f" ({len(both)} intersections, from {both[0, 0]:g} s)"
            )
    for name, table in rec.events.items():
        table = np.asarray(table, dtype=float)
        if not (
            (table.ndim == 1 and np.isfinite(table).all()) or _is_interval_table(table)
        ):
            problems.append(
                f"events[{name!r}] must be finite times (n,) or (n, 2) intervals"
                " with stop >= start"
            )
    if problems:
        msg = (
            f"{rec.dataset}/{rec.session} breaks the Recording contract:\n- "
            + "\n- ".join(problems)
        )
        raise ValueError(msg)


def _optional(attr: str, measure: Callable[[Any], Any]) -> Callable[[Recording], Any]:
    def run(rec: Recording) -> Any:
        value = getattr(rec, attr)
        return None if value is None else measure(value)

    return run


MEASURES: dict[str, Callable[[Recording], Any]] = {
    "n_units": _optional("spike_times", len),
    "n_cells": _optional("activity", lambda a: a.shape[1]),
    "n_dims": _optional("positions", lambda p: p.shape[1]),
    "sampling_rate_hz": _optional(
        "times", lambda t: float(1.0 / np.median(np.diff(t)))
    ),
    "duration_s": _optional("times", lambda t: float(t[-1] - t[0])),
    "position_min": _optional("positions", lambda p: np.nanmin(p, axis=0)),
    "position_max": _optional("positions", lambda p: np.nanmax(p, axis=0)),
    "position_nan_fraction": _optional(
        "positions", lambda p: float(np.isnan(p).any(axis=1).mean())
    ),
    "epoch_names": lambda r: sorted(r.epochs),
    "event_names": lambda r: sorted(r.events),
}


@dataclass(frozen=True)
class Range:
    """Inclusive expectation range, compared elementwise for array measures."""

    low: Any
    high: Any

    def __post_init__(self) -> None:
        if np.any(
            np.asarray(self.low, dtype=float) > np.asarray(self.high, dtype=float)
        ):
            msg = f"Range low {self.low!r} is above high {self.high!r}"
            raise ValueError(msg)

    def contains(self, actual: Any) -> bool:
        values = np.asarray(actual, dtype=float)
        low = np.asarray(self.low, dtype=float)
        high = np.asarray(self.high, dtype=float)
        return bool(np.all((values >= low) & (values <= high)))


def _check(rec: Recording, key: str, expected: Any) -> tuple[bool, str]:
    if callable(expected):
        return bool(expected(rec)), "predicate"
    actual = MEASURES[key](rec)
    detail = f"expected {expected!r}, got {actual!r}"
    if actual is None:
        return False, detail
    if isinstance(expected, Range):
        return expected.contains(actual), detail
    return bool(np.array_equal(np.asarray(actual), np.asarray(expected))), detail


def check_expectations(
    rec: Recording, expectations: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Evaluate pre-registered expectations; a failed or erroring check never raises.

    Each value is an exact value, a ``Range``, or a predicate on the
    ``Recording``. An unknown measure name is a mistake in the expectations and
    raises ``KeyError``.
    """
    unknown = [
        k for k, v in expectations.items() if not callable(v) and k not in MEASURES
    ]
    if unknown:
        msg = f"Unknown expectation {unknown[0]!r}; use one of {sorted(MEASURES)} or a predicate."
        raise KeyError(msg)
    rows = []
    for key, expected in expectations.items():
        try:
            passed, detail = _check(rec, key, expected)
        except Exception as exc:  # a check that cannot run has failed
            passed, detail = False, f"check raised {type(exc).__name__}: {exc}"
        rows.append({"key": key, "passed": passed, "detail": detail})
    return rows


def alternating_blocks(
    start: float, stop: float, length: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Split [start, stop) into alternating train and test blocks of ``length`` s.

    The last block ends at ``stop`` and is shorter than ``length`` unless the
    span is a whole number of blocks; an edge within a millionth of a block
    of ``stop`` (float rounding) is not a block of its own.

    Returns
    -------
    train, test : ndarray, shape (n, 2)
        Even-numbered and odd-numbered blocks as [start, stop) rows.
    """
    n_blocks = max(int(np.ceil((stop - start) / length - 1e-6)), 1)
    edges = np.append(start + length * np.arange(n_blocks), stop)
    blocks = np.column_stack([edges[:-1], edges[1:]])
    return blocks[::2], blocks[1::2]


def alternate_within(labels: NDArray[Any]) -> NDArray[np.bool_]:
    """Trials alternating between train and test within each label, in order.

    On an alternation task consecutive trials run different arms or
    directions, so even and odd trials would split them by arm; alternating
    within each label (arm, direction, start and end well) puts every label
    in both halves.

    Parameters
    ----------
    labels : ndarray, shape (n_trials,)
        Any hashable per-trial label, in trial order.

    Returns
    -------
    ndarray of bool, shape (n_trials,)
        True for the train trials (the 1st, 3rd, ... of each label), False
        for the test trials.
    """
    labels = np.asarray(labels)
    train = np.zeros(len(labels), dtype=bool)
    for label in np.unique(labels):
        mine = np.flatnonzero(labels == label)
        train[mine[::2]] = True
    return train


def heading_vs_movement(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
    *,
    min_speed: float,
    lag: int = 1,
) -> tuple[float, float]:
    """Compare a heading series with the direction of movement while running.

    A heading in the same allocentric frame as the positions points along the
    path while the animal runs, so the circular mean of heading minus movement
    direction is near 0 and the spread is tens of degrees. A mean near ±π/2 or
    π says the frame is rotated or mirrored; a spread near 90° says the
    series is not a heading of this path.

    Parameters
    ----------
    times : ndarray, shape (n_time,)
    positions : ndarray, shape (n_time, 2)
        Same length unit as ``min_speed``.
    headings : ndarray, shape (n_time,)
        Radians.
    min_speed : float
        Steps faster than this (length per second) count as running.
    lag : int, default 1
        The direction of movement at sample i is that of the step to sample
        i + ``lag``. On positions in whole pixels, one-sample steps of a few
        pixels take only a few directions (multiples of 45° for one-pixel
        steps), which widens the spread and biases the mean; a longer lag
        averages that out.

    Returns
    -------
    mean, spread : float
        Circular mean of heading minus movement direction, and the median
        absolute deviation from that mean, in radians.
    """
    if lag < 1:
        msg = f"lag must be a positive number of samples, got {lag}"
        raise ValueError(msg)
    step = positions[lag:] - positions[:-lag]
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.linalg.norm(step, axis=1) / (times[lag:] - times[:-lag])
    moving = np.arctan2(step[:, 1], step[:, 0])
    start = headings[:-lag]
    running = (speed > min_speed) & np.isfinite(start)
    diff = np.angle(np.exp(1j * (start[running] - moving[running])))
    mean = float(np.angle(np.mean(np.exp(1j * diff))))
    spread = float(np.median(np.abs(np.angle(np.exp(1j * (diff - mean))))))
    return mean, spread


def checked_span(
    positions: NDArray[np.float64], low: float, high: float, *, units: str = "cm"
) -> str:
    """Refuse positions whose largest per-axis span is outside ``[low, high]``.

    A loader states the span it expects of the apparatus in ``units``; a span
    outside it means a unit or scale error. Returns the line for
    ``conversions``.
    """
    tracked = positions[np.isfinite(positions).all(axis=1)]
    lo, hi = tracked.min(axis=0), tracked.max(axis=0)
    span = hi - lo
    if not low <= span.max() <= high:
        msg = (
            f"largest position span {span.max():.1f} {units} is outside the expected"
            f" {low:g}-{high:g} {units}; recheck the unit and scale"
        )
        raise ValueError(msg)
    axes = " by ".join(f"{a:.1f} to {b:.1f}" for a, b in zip(lo, hi, strict=True))
    return f"range checked: {axes} {units} (largest span {span.max():.1f}, expected {low:g}-{high:g})"


def intersect_intervals(
    a: NDArray[np.float64], b: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Pairwise intersections of two sets of [start, stop) rows, shape (n, 2)."""
    start = np.maximum(a[:, None, 0], b[None, :, 0])
    stop = np.minimum(a[:, None, 1], b[None, :, 1])
    keep = stop > start
    return np.column_stack([start[keep], stop[keep]])


def find_suspect_tracking(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    min_frozen: int,
    max_speed: float,
    units: str = "cm",
) -> tuple[dict[str, Any], str]:
    """Find held and fast samples that may be lost tracking; change nothing.

    A tracker that loses the animal often keeps reporting the last fix
    (identical samples) and then joins it to the next fix with a jump; an
    animal still at a well can also give identical samples when positions
    are coarse. So the samples are reported for the reader to judge, not
    removed:

    - held: a sample identical in every coordinate to the one before it, in
      a run of at least ``min_frozen`` such repeats, so ``min_frozen + 1``
      identical samples (the run's first sample, the last fix before it, is
      not counted);
    - fast: one sequential pass over the finite samples that are not held.
      Two samples agree when their distance over the time between them is
      at most ``max_speed`` (``units`` per second). The pass starts at the
      first sample that agrees with each of the next two (which agree with
      each other): a bad first sample would otherwise anchor the pass and
      count every later sample until the animal came back near it. Three
      samples, because a glitch held for two frames agrees with itself.
      Samples before the start are counted as fast when they do not agree
      with it, and the pass starts at the first finite sample when no three
      agree. The anchor is then the last sample passed; a later sample is
      passed, and becomes the anchor, when it agrees with the anchor, and is
      counted as fast otherwise. NaN and held samples are skipped and do not
      move the anchor, so the step after a held run is measured from the
      run's first sample, a one-frame glitch counts once (not its return
      too), and every sample of an excursion counts. After a real jump of
      distance ``d`` the pass re-anchors once ``d / max_speed`` seconds have
      passed.

    Parameters
    ----------
    times : ndarray, shape (n_time,)
    positions : ndarray, shape (n_time, n_dims)
    min_frozen : int
        Repeats of one value, in frames, from which a run counts as held.
    max_speed : float
        Fastest step counted as believable, ``units`` per second.
    units : str
        Length unit of ``positions``, for the message.

    Returns
    -------
    summary : dict
        ``held_runs``, ``held_samples``, ``held_share`` (of all samples),
        ``held_seconds`` (total), ``longest_held_seconds``, ``fast_samples``
        and the index arrays ``held`` and ``fast`` (boolean, shape
        (n_time,)).
    message : str
        One sentence for a warning (``load_warnings``).
    """
    positions = np.asarray(positions, dtype=float)
    times = np.asarray(times, dtype=float)
    repeat = np.r_[False, np.all(np.diff(positions, axis=0) == 0, axis=1)]
    edges = np.diff(np.r_[0, repeat.astype(np.int8), 0])
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    long_runs = stops - starts >= min_frozen
    held = np.zeros(len(positions), dtype=bool)
    for start, stop in zip(starts[long_runs], stops[long_runs], strict=True):
        held[start:stop] = True
    durations = times[stops[long_runs] - 1] - times[starts[long_runs] - 1]
    fast = np.zeros(len(positions), dtype=bool)
    finite = np.flatnonzero(np.isfinite(positions).all(axis=1) & ~held)
    points, clock = positions.tolist(), times.tolist()

    def agree(a: int, b: int) -> bool:
        return math.dist(points[a], points[b]) <= max_speed * abs(clock[b] - clock[a])

    order = finite.tolist()
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
    if order:
        anchor = order[first]
        for k in order[:first]:
            fast[k] = not agree(k, anchor)
        for k in order[first + 1 :]:  # sequential: each test needs the last kept
            if agree(anchor, k):
                anchor = k
            else:
                fast[k] = True
    summary = {
        "held_runs": int(long_runs.sum()),
        "held_samples": int(held.sum()),
        "held_share": float(held.mean()) if len(held) else 0.0,
        "held_seconds": float(durations.sum()),
        "longest_held_seconds": float(durations.max()) if len(durations) else 0.0,
        "fast_samples": int(fast.sum()),
        "held": held,
        "fast": fast,
    }
    message = (
        f"suspect tracking, positions kept as stored: {summary['held_runs']} runs of"
        f" {min_frozen + 1}+ identical samples ({summary['held_samples']} repeats,"
        f" {summary['held_share']:.1%}; {summary['held_seconds']:.1f} s in all,"
        f" longest {summary['longest_held_seconds']:.1f} s) and"
        f" {summary['fast_samples']} samples farther than {max_speed:g} {units}/s"
        " from the last sample before them. A held run is lost tracking if the"
        " tracker repeated its last fix, or stillness; check before analysing"
        " stops (wells, ripples)"
    )
    return summary, message
