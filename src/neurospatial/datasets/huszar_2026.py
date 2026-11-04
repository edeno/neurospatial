"""Huszár et al. (2026): mouse CA1 and CA3 on a cheeseboard maze (DANDI 001880).

What was recorded: units in CA1 and CA3 of mice whose pyramidal cells of one
embryonic birthdate express channelrhodopsin (opto-tagged), recorded while
the mice learned two new reward wells each day on a circular cheeseboard
maze. The paper's Methods describe the day: Probe 1 (about 30 minutes on the
maze), Rest 1 (about 1 hour in the home cage), Learning (about 1 hour of
trials), Rest 2, Probe 2; the files name the three maze sub-sessions
``PRE``, ``Learning`` and ``POST``. On each trial the mouse leaves a holding
box through a drawbridge, finds the two hidden water rewards, and returns;
the maze is wiped and turned 45 degrees clockwise between trials and the
same two wells "with respect to room cues" are baited again. The maze is "a
flat circular platform (90cm diameter) made of white acrylic with 177
equally spaced wells"; position is "a red LED on the head of the animal"
filmed by an overhead camera "at a frame rate of 30 Hz", whose "field of
view covered the platform but excluded the holding box itself" (the paper's
STAR Methods).

Sessions are DANDI 001880's 42 files (version 0.261007.1849, CC0-1.0), one
per recording day, 66 to 230 MB. The default session,
``sub-iuvE14-cagCre-ef1aDIOchr2-3_ses-iuvE14-cagCre-ef1aDIOchr2-3-221208``,
is the smallest (65.7 MB), the one the survey of public datasets loaded: 49
units, 45 learning trials, 2,118 ripples.

Sources read: DANDI 001880's metadata, the files, and the paper (Huszár,
Kirsanov, Henze, Huilgol, Valero, Huang & Buzsáki 2026, Neuron; the author
manuscript on PubMed Central, PMC13573635), whose Data availability names
this dandiset.

What the loader converts:

- positions: the file's ``PositionSeries`` as stored, in cm (its unit). The
  file's ``SubjectPosition`` also holds a ``VelocitySeries``, which is not
  loaded.
- epochs: ``on_maze`` (``onMaze``, when the LED was on the maze),
  ``open_box`` (``openBox``, when the box was open), ``pre``, ``learning``
  and ``post`` (``subSessions`` by label), and ``awake``, ``nrem`` and
  ``rem`` (``SleepStates`` by label).
- trials: the file's ``LearningTrials`` table, each row with its
  ``onMaze_start_time``, ``rewardTime`` and ``rewardOrder``.
- events: ``ripples``, the ``Ripples`` table's [start, stop) rows;
  ``ripple_peaks``, its ``peaks``; ``rewards``, every ``rewardTime`` of the
  learning trials, the water consumptions the experimenter recorded.
- reward wells: ``rewardCoordinates`` rows 1 and 2 as the point regions
  ``reward_1`` and ``reward_2`` of ``rec.env``; a trial's ``rewardOrder``
  names them in the order they were found (checked: the tracked position at
  each reward time lies a median of 2 to 3 cm from the well it names).
- ``units``: ``brain_region`` from the file's ``location``, ``label`` from
  its ``cell_type``; the file's other columns (``CA1_optoTag``,
  ``CA3_optoTag``, ``unit_name``, ``clu_id``, ``group_id``) are kept.
- headings: none. One LED is tracked.

The declared environment is the stated 90 cm disc, centred midway between
the 0.5th and 99.5th percentiles of the samples tracked during ``on_maze``
(the file states no origin: its reference frame is "Arbitrary, camera"), in
2.5 cm bins, with the reward wells as regions. The camera frame's
handedness (whether y grows up or down the room) is not stated either.

Analysis choices (``params``):

- ``bin_size`` 2.5 cm, the loader's choice: the paper's rate maps use 1 cm
  bins and a 4 cm (FWHM) Gaussian, which gives about 6,400 bins in the disc;
- ``min_speed`` 5 cm/s, the paper's running threshold ("speed was >5cm/s");
- ``analysis_epochs``: ``on_maze``, so the holding box is left out;
- ``dt`` 0.25 s, the decoding bin, and ``train_epochs`` / ``test_epochs``,
  the even and odd rows of ``on_maze``.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from neurospatial.datasets._recording import (
    Conversion,
    Recording,
    interval_rows,
    nwb_unit_table,
    tracked_mask,
)

if TYPE_CHECKING:
    from neurospatial import Environment
    from neurospatial.datasets._catalog import SessionEntry

__all__ = [
    "EXPECTED_WATCH_OUTS",
    "REQUIRES",
    "WATCH_OUTS",
    "WORKED_RESULT",
    "load_session",
]

#: The paper's finding this dataset reproduces, or why there is none yet.
WORKED_RESULT: dict[str, Any] = {
    "status": "not possible yet",
    "level": None,
    "finding": None,
    "quote": None,
    "source": None,
    "paper": None,
    "ours": None,
    "criterion": None,
    "parameters": None,
    "unmatched": (),
    "attempted": None,
    "why": (
        "The numbers the paper's text gives are pooled over its 42 sessions "
        '(for example "34.8 ± 2.5 trials (mean ± SEM)" per learning '
        "session, and 123.5 CA1 and 88.3 CA3 neurons per session), and its "
        "per-session results (memory retrieval, Figure 2D; subspace "
        "expression, Figure 4) are shown as plotted points without their "
        "values or rest on the authors' contrastive-PCA pipeline. One "
        "session's number cannot be checked against them."
    ),
}

#: Modules of the ``datasets`` extra the loader imports.
REQUIRES = ("pynwb",)

#: The stated platform's diameter, cm.
DIAMETER = 90.0
BIN_SIZE = 2.5
#: The paper's running threshold, cm/s.
MIN_SPEED = 5.0
DT = 0.25
#: The camera's stated frame rate, Hz (STAR Methods: "a frame rate of 30 Hz").
FRAME_RATE = 30.0
#: Above this share of off-maze tracked samples repeating the one before,
#: most of them are the tracker's held last fix.
HELD_OFF_MAZE = 0.5
#: The disc is centred between these percentiles of the on-maze samples, so
#: a few far samples do not move it (as Shipley's star centre).
CENTRE_PERCENTILES = (0.5, 99.5)

_BEHAVIOR = "processing/behavior"
_POSITION = f"{_BEHAVIOR}/SubjectPosition/PositionSeries"
_STATES = {"Awake": "awake", "Non-REM": "nrem", "REM": "rem"}
_SUB_SESSIONS = {"PRE": "pre", "Learning": "learning", "POST": "post"}
_PAPER = "the STAR Methods of Huszár et al. 2026, PMC13573635"

#: Findings computed from the data (by the checks or this loader) that each
#: verified session must report, as patterns searched for in its watch_outs.
EXPECTED_WATCH_OUTS: dict[str, tuple[str, ...]] = {
    "sub-iuvE14-cagCre-ef1aDIOchr2-3_ses-iuvE14-cagCre-ef1aDIOchr2-3-221208": (
        r"runs of 11 or more identical position samples",
        r"lie outside the declared environment",
        r"of the tracked samples outside epochs\['on_maze'\] repeat",
        r"\d+ of the \d+ learning trials with both rewards stop more than one "
        r"frame",
    ),
}

#: Pitfalls of each session checked by hand, with their evidence.
WATCH_OUTS: dict[str, tuple[str, ...]] = {
    "sub-iuvE14-cagCre-ef1aDIOchr2-3_ses-iuvE14-cagCre-ef1aDIOchr2-3-221208": (
        "This session holds 49 units (26 CA1, 23 CA3), fewer than the paper's "
        'per-session mean ("nCA1=123.5±9.7; nCA3=88.3±10 neurons per '
        f'session", {_PAPER}): it is the smallest file of the dandiset.',
    ),
}

_UNLOADED = (
    "The file also holds the VelocitySeries (vx, vy, speed in cm/s), the "
    "ripples' metrics and waveforms, and the opto-tagging light pulses "
    "(acquisition/pulses), which are not loaded; read them with pynwb."
)


def load_session(
    entry: SessionEntry, path_or_stream: str | os.PathLike[str]
) -> Recording:
    """Read one DANDI 001880 session into a :class:`Recording`.

    Parameters
    ----------
    entry : SessionEntry
        The session's registry entry.
    path_or_stream : str or path
        The cached NWB file, or its URL to read by byte range.

    Returns
    -------
    Recording
        Unchecked; :func:`neurospatial.datasets.load` runs the checks.

    Raises
    ------
    ValueError
        If the position series is not in cm, or the file has no tracked
        sample during ``onMaze``.
    """
    from neurospatial.datasets._fetch import open_nwb
    from neurospatial.io.nwb import read_intervals, read_position, read_units

    empty: dict[str, int] = {}

    def rows(name: str) -> Any:
        """The table's rows, those of zero length (start == stop) left out."""
        table = read_intervals(
            nwb, interval_name=f"{_BEHAVIOR}/{name}"
        ).intervals_table()
        zero = (table["start_time"] == table["stop_time"]).to_numpy()
        if zero.any():
            empty[name] = int(zero.sum())
        return table[~zero]

    with open_nwb(os.fspath(path_or_stream)) as nwb:
        position = read_position(nwb, position_name=_POSITION)
        series = nwb.processing["behavior"]["SubjectPosition"]["PositionSeries"]
        stated_unit = f"unit {series.unit!r}, conversion {series.conversion:g}"
        description, frame = series.description, series.reference_frame
        units = read_units(nwb)
        on_maze = interval_rows(rows("onMaze"))
        open_box = interval_rows(rows("openBox"))
        sub_sessions = rows("subSessions")
        states = rows("SleepStates")
        trials = read_intervals(nwb, interval_name=f"{_BEHAVIOR}/LearningTrials")
        ripples = nwb.processing["ecephys"]["Ripples"]
        ripple_rows = np.column_stack(
            [ripples["start_time"][:], ripples["stop_time"][:]]
        ).astype(np.float64)
        ripple_peaks = np.asarray(ripples["peaks"][:], dtype=np.float64)
        wells = (
            nwb.processing["behavior"]["rewardCoordinates"]
            .to_dataframe()[["x", "y"]]
            .to_numpy(dtype=np.float64)
        )
        pulses = nwb.acquisition.get("pulses")
        pulse_span = (
            None
            if pulses is None or not len(pulses)
            else (
                float(np.min(pulses["start_time"][:])),
                float(np.max(pulses["stop_time"][:])),
            )
        )
        subject = nwb.subject.subject_id
        started = nwb.session_start_time
    if position.units != "cm":
        msg = f"{entry.session}: {_POSITION} is in {position.units!r}, not cm"
        raise ValueError(msg)
    times = np.asarray(position.times, dtype=np.float64)
    positions = np.asarray(position.positions, dtype=np.float64)
    maze = _within(times, on_maze) & tracked_mask(positions)
    if not maze.any():
        msg = f"{entry.session}: no position sample is tracked during onMaze"
        raise ValueError(msg)
    env, centre, placement = _declared_disc(
        positions[maze], wells, entry.session, frame
    )
    recording_stop = _last_stop(
        interval_rows(states), interval_rows(sub_sessions), pulse_span
    )
    table = trials.intervals_table()
    found = _rewards(table)
    reward_times = np.sort([when for rewards in found for when, _ in rewards])
    order = ripple_rows[:, 0].argsort(kind="stable")
    unit_table, regions = nwb_unit_table(
        units, region_column="location", label_column="cell_type", keep_columns=True
    )
    epochs = {
        "on_maze": on_maze,
        "open_box": open_box,
        **{
            key: interval_rows(sub_sessions[sub_sessions["label"] == label])
            for label, key in _SUB_SESSIONS.items()
        },
        **{
            key: interval_rows(states[states["label"] == label])
            for label, key in _STATES.items()
            if (states["label"] == label).any()
        },
        **_off_camera(times, interval_rows(sub_sessions), recording_stop, pulse_span),
    }
    return Recording(
        **entry.identity(path_or_stream),
        animal=f"mouse {subject}",
        apparatus=entry.info.apparatus,
        brain_regions=regions or entry.info.brain_regions,
        clock=(
            "the file's clock, seconds from its session start time "
            f"({started:%Y-%m-%d} {started:%H:%M}, the recording's date with no "
            "time of day)"
        ),
        times=times,
        spike_times=tuple(units.spike_times),
        units=unit_table,
        env=env,
        env_kind="declared",
        env_source=(
            f"the stated {DIAMETER:g} cm diameter circular platform with 177 "
            f"wells ({_PAPER}), {BIN_SIZE:g} cm bins; the reward wells "
            "(rewardCoordinates) as regions reward_1 and reward_2"
        ),
        positions=positions,
        position_units="cm",
        tracking=(
            "a red LED on the head, filmed from above at 30 frames/s; the "
            f"camera does not see the holding box ({_PAPER})"
        ),
        epochs=epochs,
        events={
            "ripples": ripple_rows[order],
            "ripple_peaks": np.sort(ripple_peaks),
            "rewards": reward_times,
        },
        trials=trials,
        params={
            "bin_size": BIN_SIZE,
            "min_speed": MIN_SPEED,
            "dt": DT,
            "analysis_epochs": on_maze,
            "train_epochs": on_maze[::2],
            "test_epochs": on_maze[1::2],
        },
        conversions=(
            Conversion(
                f"positions: {_POSITION} as stored, cm",
                "stated",
                f"the file's series: {stated_unit} (read_position applies the "
                f"conversion); {description!r}; the camera frame's handedness is "
                "not stated (whether y grows up or down the room), so the "
                "positions keep the file's axes and may be the room's mirror "
                "image, in which case the maze's stated clockwise turn between "
                "trials appears counter-clockwise in them",
            ),
            Conversion(
                "epochs: onMaze as 'on_maze', openBox as 'open_box', "
                "subSessions PRE, Learning and POST as 'pre', 'learning' and "
                "'post' (the paper's Probe 1, Learning and Probe 2), "
                "SleepStates Awake, Non-REM and REM as 'awake', 'nrem' and 'rem'",
                "stated",
                "the file's interval tables in processing/behavior and their "
                "descriptions ('Intevals when LED was detected on a maze', "
                "'Intevals when the box was open', 'subSessions (PRE vs "
                "Learning vs POST)', 'Sleep state of the animal.')",
            ),
            Conversion(
                "epochs: 'rest', the stretches between consecutive "
                "sub-sessions; 'before_tracking', from 0 s to the first "
                "position sample; 'after_tracking', from the last position "
                "sample to the last stop of the SleepStates, subSessions and "
                "light pulses; 'opto_tagging', from the first to the last light "
                "pulse (acquisition/pulses)",
                "inferred",
                "the paper's day runs Probe 1, Rest 1 (~1h, home cage), "
                "Learning, Rest 2 (~1h), Probe 2, and light-responsive neurons "
                f"were identified 'in the homecage following Probe 2' ({_PAPER}); "
                "the file's interval tables run to "
                f"{recording_stop:.6g} s, its tracking from {times[0]:.6g} to "
                f"{times[-1]:.6g} s",
            ),
            Conversion(
                "trials: the LearningTrials table (start, stop, "
                "onMaze_start_time, rewardTime, rewardOrder); "
                "events['rewards']: every rewardTime",
                "stated",
                "the file's table, 'Learning trials and rewards information'; "
                "the paper: 'The experimenter manually recorded water "
                f"consumption time points using a footswitch' ({_PAPER})",
            ),
            Conversion(
                f"events['ripples']: the {len(ripple_rows)} Ripples rows as "
                "[start, stop) in s; events['ripple_peaks']: their peaks",
                "stated",
                "the file's processing/ecephys/Ripples table, 'Ripples and "
                f"their metrics'; detection as described in {_PAPER} (130-200 "
                "Hz band, 5 SD peak, 2 SD edges, 20-200 ms)",
            ),
            _reward_conversion(times, positions, found, wells),
            placement,
        ),
        watch_outs=(
            *WATCH_OUTS.get(entry.session, ()),
            *_empty_rows(empty),
            *_held_off_maze(times, positions, on_maze, centre),
            *_trial_stops(table, found, on_maze),
            *_short_trials(table, found),
            "Positions outside epochs['on_maze'] are not the animal's: the "
            "camera does not see the holding box, so use "
            "params['analysis_epochs'] (on_maze) for rate maps and occupancy. "
            "The positions are kept as stored.",
            "The maze was wiped and turned 45 degrees clockwise between "
            "learning trials, with the same wells baited 'with respect to "
            f"room cues' ({_PAPER}): positions, the environment and the reward "
            "wells are in the camera's (room) frame, not the turning maze's.",
            _UNLOADED,
        ),
    )


def _within(times: NDArray[np.float64], rows: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Whether each sorted time lies in a [start, stop) row, shape (n_time,)."""
    starts = np.searchsorted(times, rows[:, 0], side="left")
    stops = np.searchsorted(times, rows[:, 1], side="left")
    counts = np.zeros(times.size + 1, dtype=np.int64)
    np.add.at(counts, starts, 1)
    np.add.at(counts, stops, -1)
    inside: NDArray[np.bool_] = np.cumsum(counts[:-1]) > 0
    return inside


def _empty_rows(empty: dict[str, int]) -> tuple[str, ...]:
    """The interval rows of zero length the epochs leave out, by table."""
    if not empty:
        return ()
    counts = ", ".join(f"{n} in {name}" for name, n in empty.items())
    return (
        f"Rows of zero length (start equal to stop) are left out of the epochs "
        f"({counts}): an epoch row must start before it stops, and such a row "
        "holds no time.",
    )


def _last_stop(
    states: NDArray[np.float64],
    sub_sessions: NDArray[np.float64],
    pulse_span: tuple[float, float] | None,
) -> float:
    """The last stop of the file's sleep states, sub-sessions and light pulses, s."""
    stops = [float(states[:, 1].max()), float(sub_sessions[:, 1].max())]
    return max(stops if pulse_span is None else [*stops, pulse_span[1]])


def _off_camera(
    times: NDArray[np.float64],
    sub_sessions: NDArray[np.float64],
    recording_stop: float,
    pulse_span: tuple[float, float] | None,
) -> dict[str, NDArray[np.float64]]:
    """The recording's stretches without the maze: rest, before and after it.

    Parameters
    ----------
    times : ndarray, shape (n_time,)
    sub_sessions : ndarray, shape (n, 2)
        The ``subSessions`` rows, sorted.
    recording_stop : float
        The last stop of the file's interval tables (:func:`_last_stop`).
    pulse_span : (float, float) or None
        The first light pulse's start and the last one's stop.

    Returns
    -------
    dict of str to ndarray, shape (n, 2)
        ``rest`` (between consecutive sub-sessions), ``before_tracking``
        (from 0 s to the first sample), ``after_tracking`` (from the last
        sample to ``recording_stop``) and ``opto_tagging``, each where
        it is not empty.
    """
    found: dict[str, NDArray[np.float64]] = {}
    gaps = np.column_stack([sub_sessions[:-1, 1], sub_sessions[1:, 0]])
    if len(rest := gaps[gaps[:, 1] > gaps[:, 0]]):
        found["rest"] = rest
    if times[0] > 0:
        found["before_tracking"] = np.array([[0.0, times[0]]])
    if recording_stop > times[-1]:
        found["after_tracking"] = np.array([[times[-1], recording_stop]])
    if pulse_span is not None:
        found["opto_tagging"] = np.array([pulse_span])
    return found


def _declared_disc(
    on_maze: NDArray[np.float64],
    wells: NDArray[np.float64],
    session: str,
    frame: str | None,
) -> tuple[Environment, NDArray[np.float64], Conversion]:
    """The stated disc at the on-maze samples' percentile midrange, with the wells as regions.

    Parameters
    ----------
    on_maze : ndarray, shape (n_tracked, 2)
        The positions tracked during ``onMaze``, cm.
    wells : ndarray, shape (2, 2)
        The ``rewardCoordinates`` rows, cm.

    Returns
    -------
    env : Environment
    centre : ndarray, shape (2,)
        The disc's centre, cm.
    placement : Conversion
    """
    import shapely

    from neurospatial import Environment
    from neurospatial.regions import Region

    low, high = on_maze.min(axis=0), on_maze.max(axis=0)
    inner_low, inner_high = np.percentile(on_maze, CENTRE_PERCENTILES, axis=0)
    centre = (inner_low + inner_high) / 2
    radius = DIAMETER / 2
    beyond = float(np.mean(np.linalg.norm(on_maze - centre, axis=1) > radius))
    residuals = ", ".join(
        f"{axis} {span:.4g} cm, {span / DIAMETER - 1:+.1%}"
        for axis, span in zip("xy", high - low, strict=True)
    )
    regions = {
        f"reward_{k + 1}": Region(name=f"reward_{k + 1}", kind="point", data=well)
        for k, well in enumerate(wells)
    }
    disc = shapely.Point(*centre).buffer(radius, quad_segs=64)
    env = Environment.from_polygon(
        disc,
        bin_size=BIN_SIZE,
        units="cm",
        name=f"huszar_2026 {session}",
        regions=regions,
    )
    placement = Conversion(
        f"environment: the stated {DIAMETER:g} cm disc centred at "
        f"({centre[0]:.4g}, {centre[1]:.4g}) cm; the on-maze samples' range "
        f"against the stated diameter: {residuals}; {beyond:.1%} of them lie "
        f"beyond its {radius:g} cm radius",
        "inferred",
        f"the file states no origin (reference frame {frame!r}); the centre is "
        f"midway between the {CENTRE_PERCENTILES[0]:g}th and "
        f"{CENTRE_PERCENTILES[1]:g}th percentiles of the samples tracked during "
        f"onMaze, x {inner_low[0]:.4g} to {inner_high[0]:.4g} cm and y "
        f"{inner_low[1]:.4g} to {inner_high[1]:.4g} cm, so a few far samples do "
        f"not move it; their full range is x {low[0]:.4g} to {high[0]:.4g} cm and "
        f"y {low[1]:.4g} to {high[1]:.4g} cm, whose midrange lies "
        f"{np.linalg.norm((low + high) / 2 - centre):.2g} cm away",
    )
    return env, centre, placement


def _rewards(table: Any) -> list[list[tuple[float, int]]]:
    """Each trial's rewards found, as (time, rewardOrder) pairs.

    A reward not found has a missing ``rewardTime`` (NaN in the file, None as
    read) and ``rewardOrder`` 0; it is left out.
    """
    found = []
    for times, orders in zip(table["rewardTime"], table["rewardOrder"], strict=True):
        found.append(
            [
                (float(when), int(k))
                for when, k in zip(times, orders, strict=True)
                if when is not None and np.isfinite(when) and k in (1, 2)
            ]
        )
    return found


def _reward_conversion(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    found: list[list[tuple[float, int]]],
    wells: NDArray[np.float64],
) -> Conversion:
    """Why ``rewardOrder`` k names ``rewardCoordinates`` row k (1-based)."""
    named, other = [], []
    for rewards in found:
        for when, k in rewards:
            index = min(int(np.searchsorted(times, when)), times.size - 1)
            here = positions[index]
            named.append(np.linalg.norm(here - wells[k - 1]))
            other.append(np.linalg.norm(here - wells[2 - k]))
    return Conversion(
        "regions: rewardCoordinates rows 1 and 2 as 'reward_1' and 'reward_2'; "
        "a trial's rewardOrder k names region reward_k",
        "inferred",
        "the file says only 'Order of rewards'; the tracked position at each "
        f"of the {len(named)} reward times lies a median "
        f"{np.nanmedian(named):.3g} cm from the well its rewardOrder names and "
        f"{np.nanmedian(other):.3g} cm from the other",
    )


def _held_off_maze(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    on_maze: NDArray[np.float64],
    centre: NDArray[np.float64],
) -> tuple[str, ...]:
    """Off-maze samples that repeat the one before: the tracker's last fix.

    Only where more than ``HELD_OFF_MAZE`` of the tracked off-maze samples
    repeat the one before.
    """
    off = ~_within(times, on_maze) & tracked_mask(positions)
    if not off.any():
        return ()
    repeat = np.r_[False, np.all(np.diff(positions, axis=0) == 0, axis=1)]
    share = float(repeat[off].mean())
    if share <= HELD_OFF_MAZE:
        return ()
    on = _within(times, on_maze) & tracked_mask(positions)
    distance = np.linalg.norm(positions[off] - centre, axis=1)
    moving = off & ~repeat
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    path = float(np.nansum(steps[moving[1:] & off[:-1]]))
    return (
        f"{share:.0%} of the tracked samples outside epochs['on_maze'] repeat the "
        f"sample before ({int(off.sum())} samples, a median "
        f"{np.median(distance):.3g} cm from the platform's centre), against "
        f"{repeat[on].mean():.0%} on the maze: with more than "
        f"{HELD_OFF_MAZE:.0%} held, most likely the tracker holding its last fix "
        "at the platform's edge while the mouse is in the holding box, which "
        "the camera does not see."
        + (
            f" The other {int(moving.sum())} move, {path:.0f} cm of path in all, so "
            "the mouse is sometimes tracked on the maze outside an onMaze row."
            if path > 0
            else ""
        ),
    )


def _trial_stops(
    table: Any, found: list[list[tuple[float, int]]], on_maze: NDArray[np.float64]
) -> tuple[str, ...]:
    """How the file's trial stops fall after the second reward, with numbers.

    Only where a trial with both rewards stops more than one camera frame
    (1 / ``FRAME_RATE``) after its second reward.
    """
    stops = table["stop_time"].to_numpy(dtype=np.float64)
    both = np.array([len(rewards) == 2 for rewards in found], dtype=bool)
    if not both.any():
        return ()
    last = np.array(
        [max(when for when, _ in rewards) for rewards in found if len(rewards) == 2]
    )
    after = stops[both] - last
    late = after > 1.0 / FRAME_RATE
    if not late.any():
        return ()
    n_maze = int(np.isin(stops, on_maze[:, 1]).sum())
    return (
        f"{int(late.sum())} of the {int(both.sum())} learning trials with both "
        f"rewards stop more than one frame (1/{FRAME_RATE:g} s) after the second; "
        f"the trials with both rewards stop {after.min():.1f} to "
        f"{after.max():.1f} s after it (median {np.median(after):.1f} s), though "
        "the paper defines a trial's end as 'the time point of the second water "
        f"consumption' ({_PAPER}); {n_maze} of the {len(table)} trials stop at "
        "the stop of an onMaze row, when the LED leaves the maze. A trial's "
        "start is when the box opens; its onMaze_start_time is when the LED is "
        "first on the maze.",
    )


def _short_trials(table: Any, found: list[list[tuple[float, int]]]) -> tuple[str, ...]:
    """A watch-out for trials in which fewer than two rewards were found."""
    short = sum(len(rewards) < 2 for rewards in found)
    if not short:
        return ()
    return (
        f"In {short} of the {len(table)} learning trials fewer than two rewards "
        "were found (rewardTime missing, rewardOrder 0); the paper says a trial "
        "was aborted after 4 minutes of exploring.",
    )
