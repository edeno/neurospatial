"""Peyrache lab: mouse retrosplenial cortex in open arenas, with head direction (DANDI 000987).

What was recorded: units in the retrosplenial cortex (RSC) of three mice on
silicon probes, over sessions that alternate sleep in the home cage with
exploration of a square or a circular arena, the head tracked in 3-D by
motion capture. DANDI 000987 (version 0.261003.2342, CC-BY-4.0) holds 14
sessions, 20 to 156 MB, each with its units, the position and head
orientation of the tracked head, sleep and wake epochs, slow-wave sleep and
UP and DOWN states. DANDI's description says the data were used in
"Hyperpolarization-activated currents drive neuronal activation sequences in
sleep" (Mehrotra, Levenstein et al. 2024, Current Biology), Figure 5F. The
study's bioRxiv preprint (10.1101/2023.09.12.557442, v1, Methods) says "For
the RSC dataset, animals explored a circular arena (diameter 70 cm) with a
white cue card" "in addition to the square arena", that position and head
orientation were tracked "in 3D using 7 infrared cameras (Flex 13,
Optitrack)" and "sampled at 100 Hz (120 Hz for the RSC dataset)". That
120 Hz is not applied as a stated rate, which would refuse a session: 11 of
the 14 files sample at 119.76 Hz, but three (the default among them) at
100 Hz, so the rate is inferred and those three load with a watch-out.

The default session, ``sub-A7803_ses-20240509T213129``, is the survey's
file, the smallest with units and position (20.2 MB): mouse A7803, 37 units,
sleep (0 to 3,878 s) then the square arena (3,878 to 5,085 s), tracked from
3,880 to 5,083 s.

Sources read: DANDI 000987's metadata (description, related resources), the
files, the Current Biology paper's Crossref record (title, authors, volume,
pages) and the bioRxiv preprint's full text. The Current Biology version was
not read, so every number quoted here is the preprint's.

What the loader converts:

- positions: each file stores ``Position/x``, ``y`` and ``z`` as separate
  series with a blank unit and a blank reference frame. y is vertical (its
  span is a few cm, against 50 to 80 cm for x and z), the floor is (x, z),
  and the values are metres: read as metres, the circular epochs span 64 to
  70 cm, the preprint's 70 cm circle. The loader returns (x, -z) x 100 cm.
  Motion capture with y up is right-handed, so seen from above x points East
  and -z North (an inferred ``Conversion``). The heading's movement test
  cannot check that handedness: mirroring the positions and the heading
  together leaves heading minus movement unchanged.
- headings: ``CompassDirection/ry``, the rotation about the vertical axis,
  wrapped from [0, 2π) to [-π, π]. The frame is not verified: heading minus
  the direction of movement, per tracked epoch, has a mean that is not near
  0 in most sessions and changes between epochs of one session (a
  watch-out with the numbers), so the stored values are kept with an
  inferred ``Conversion``.
- epochs: ``sleep`` and ``wake`` from the epochs table's tags; ``sws``,
  ``up`` and ``down`` from their tables; ``tracking`` from
  ``position_time_support``; and each tracked epoch named by its arena,
  ``square``, ``circle`` or ``partial_circle``, in the order the file's
  session_description lists the arenas.

The environment is inferred from the samples (``Environment.open_field``,
2.5 cm bins): the preprint gives the circle's diameter but not the square's
side, and a session that visits two arenas has them in one coordinate frame,
so one environment covers both (a watch-out); build one per arena from its
epoch's samples.

Analysis choices (``params``):

- ``bin_size`` 2.5 cm; ``analysis_epochs`` the tracked epochs
  (``position_time_support``); ``min_speed`` None, since no source read
  states a running threshold;
- ``dt`` 0.25 s and alternating 60 s ``train_epochs`` / ``test_epochs``.
"""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from neurospatial.datasets._params import alternating_blocks
from neurospatial.datasets._recording import (
    Conversion,
    Recording,
    heading_vs_movement,
    interval_rows,
    nwb_unit_table,
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
        "DANDI's description says the files were used for Figure 5F of the "
        "Current Biology paper, which was not read; the bioRxiv preprint, "
        "which was, analyses the RSC units' sleep dynamics (UP and DOWN "
        "states, activation sequences) pooled over sessions and gives no "
        "spatial or head-direction number for one session to check."
    ),
}

#: Modules of the ``datasets`` extra the loader imports.
REQUIRES = ("pynwb",)

BIN_SIZE = 2.5
DT = 0.25
BLOCK = 60.0
#: Read as metres, a mouse arena's floor spans this many m on each axis.
FLOOR_M = (0.3, 1.5)
#: The vertical axis spans less than this share of the smaller floor span.
VERTICAL_SHARE = 0.5
#: Heading is compared with movement at about this rate, Hz, faster than
#: ``MOVING`` cm/s: at 100 or 120 Hz a motion-capture step is under 2 mm and
#: its direction is mostly noise.
COMPARE_RATE = 10.0
MOVING = 10.0
#: The frame counts as verified by movement when the circular mean of
#: heading minus movement is within ``FRAME_MEAN`` rad of 0 and the spread
#: is under ``FRAME_SPREAD`` rad (the survey of public datasets' rule).
FRAME_MEAN = 0.2
FRAME_SPREAD = np.radians(45.0)
_PREPRINT = (
    "the Methods of Mehrotra, Levenstein et al.'s preprint, bioRxiv 2023.09.12.557442"
)
#: The arenas a session_description names, in the order they are visited.
_ARENA = re.compile(r"(partial circular|circular|square) arena")
_ARENA_NAMES = {
    "partial circular": "partial_circle",
    "circular": "circle",
    "square": "square",
}

#: Findings computed from the data (by the checks or this loader) that each
#: verified session must report, as patterns searched for in its watch_outs.
EXPECTED_WATCH_OUTS: dict[str, tuple[str, ...]] = {
    "sub-A7803_ses-20240509T213129": (
        r"The heading frame is not verified by movement",
        r"the sampling rate is 100 Hz",
        r"runs of 11 or more identical position samples",
    ),
}

#: Pitfalls of each session checked by hand, with their evidence.
WATCH_OUTS: dict[str, tuple[str, ...]] = {
    "sub-A7803_ses-20240509T213129": (
        "The square arena's tracked span is 52.7 x 53.2 cm. The preprint says "
        'the arena was "a plastic platform with removable walls (width: 80 '
        'cm, height: 50 cm)" but not the side of the square they formed, so '
        "the environment is inferred from the samples, not declared.",
    ),
}

_UNLOADED = (
    "The file's CompassDirection also holds rx and rz, the head's other two "
    "rotations, which are not loaded; read them with pynwb."
)


def load_session(
    entry: SessionEntry, path_or_stream: str | os.PathLike[str]
) -> Recording:
    """Read one DANDI 000987 session into a :class:`Recording`.

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
        If the position or head-orientation series are not on one clock,
        their units are not blank and radian, or the value ranges no longer
        read as metres with y vertical.
    """
    from neurospatial.datasets._fetch import open_nwb
    from neurospatial.io.nwb import read_intervals, read_units

    def rows(name: str) -> Any:
        return interval_rows(read_intervals(nwb, interval_name=name).intervals_table())

    with open_nwb(os.fspath(path_or_stream)) as nwb:
        position = nwb.acquisition["Position"]
        compass = nwb.acquisition["CompassDirection"]
        axes = {key: position[key] for key in "xyz"}
        labels = {key: series.unit for key, series in axes.items()}
        frames = {series.reference_frame for series in axes.values()}
        xyz = {key: np.asarray(s.data[:], dtype=np.float64) for key, s in axes.items()}
        times = np.asarray(axes["x"].timestamps[:], dtype=np.float64)
        same_clock = all(
            np.array_equal(np.asarray(series.timestamps[:]), times)
            for series in (*axes.values(), *compass.spatial_series.values())
        )
        yaw = np.asarray(compass["ry"].data[:], dtype=np.float64)
        yaw_unit = compass["ry"].unit
        units = read_units(nwb)
        tagged = read_intervals(nwb, interval_name="epochs").intervals_table()
        epochs = {
            tag: interval_rows(tagged[[tag in tags for tags in tagged["tags"]]])
            for tag in ("sleep", "wake")
        }
        for name in ("SWS", "UP", "DOWN"):
            epochs[name.lower()] = rows(name)
        tracked = rows("position_time_support")
        description = nwb.session_description
        identifier = nwb.identifier
        subject = nwb.subject.subject_id
        started = nwb.session_start_time
    floor = float(np.ptp(xyz["x"])), float(np.ptp(xyz["z"]))
    height = float(np.ptp(xyz["y"]))
    if (
        set(labels.values()) != {""}
        or yaw_unit != "radian"
        or not same_clock
        or not all(FLOOR_M[0] <= span <= FLOOR_M[1] for span in floor)
        or height >= VERTICAL_SHARE * min(floor)
    ):
        msg = (
            f"{entry.session}: position units {labels}, yaw unit {yaw_unit!r}, "
            f"one clock {same_clock}, spans x {floor[0]:.3g}, y {height:.3g}, "
            f"z {floor[1]:.3g}; the blank-unit metres-with-y-vertical reading no "
            "longer holds"
        )
        raise ValueError(msg)
    from neurospatial._angles import wrap_angle

    positions = np.column_stack([xyz["x"], -xyz["z"]]) * 100.0
    headings = np.asarray(wrap_angle(yaw), dtype=np.float64)
    arenas, arena_line = _arena_epochs(description, tracked)
    epochs.update(arenas)
    epochs["tracking"] = tracked
    comparisons = _heading_by_epoch(times, positions, headings, tracked)
    negated = [
        spread
        for _, _, spread in _heading_by_epoch(times, positions, -headings, tracked)
    ]
    env = _inferred_environment(positions, entry.session)
    table, regions = nwb_unit_table(units, region_column="location")
    train, test = alternating_blocks(times[0], times[-1], BLOCK)
    recorded = _identifier_date(identifier)
    return Recording(
        **entry.identity(path_or_stream),
        animal=f"mouse {subject}",
        apparatus=entry.info.apparatus,
        brain_regions=regions or entry.info.brain_regions,
        clock=(
            "the file's clock, seconds from its session start time "
            f"({started:%Y-%m-%d %H:%M}, the day all 14 files of the dandiset "
            "start, minutes apart, so likely when they were written"
            + (f"; the identifier {identifier} reads as {recorded}" if recorded else "")
            + ")"
        ),
        times=times,
        spike_times=tuple(units.spike_times),
        units=table,
        env=env,
        env_kind="inferred",
        env_source=(
            "the tracked samples (Environment.open_field, "
            f"{BIN_SIZE:g} cm bins); the preprint states a 70 cm circle and "
            "a square of removable walls but not the square's side "
            f"({_PREPRINT})"
        ),
        positions=positions,
        position_units="cm",
        tracking=(
            "a rigid body of markers on the head, tracked in 3-D by an "
            f"Optitrack motion-capture system ({_PREPRINT}); its floor position"
        ),
        headings=headings,
        heading_source=(
            "the file's CompassDirection/ry, the rigid body's rotation about the "
            "vertical axis; frame not verified by movement"
        ),
        epochs=epochs,
        params={
            "bin_size": BIN_SIZE,
            "dt": DT,
            "min_speed": None,
            "analysis_epochs": tracked,
            "train_epochs": train,
            "test_epochs": test,
        },
        conversions=(
            Conversion(
                "positions: cm = (Position/x, -Position/z) x 100; Position/y is "
                "vertical and left out",
                "inferred",
                "the series' unit and reference frame are blank "
                f"({sorted(frames)!r}); spans x {floor[0]:.3g}, y {height:.3g}, "
                f"z {floor[1]:.3g}: y is the head's height, x and z the floor, "
                "and read as metres the dandiset's circular epochs span 64 to 70 "
                f"cm, the 70 cm circle of {_PREPRINT}. Motion capture with y up "
                "is right-handed, so seen from above x points East and -z North; "
                "the heading's movement test cannot check this: mirroring the "
                "positions and the heading together leaves heading minus "
                "movement unchanged",
            ),
            Conversion(
                "headings: CompassDirection/ry (radian, [0, 2π)) wrapped to "
                "[-π, π] (neurospatial.stats.wrap_angle), in the (x, -z) frame "
                "of the positions; not rotated or mirrored",
                "inferred",
                "ry is the rotation about the vertical y axis, the yaw. Its sign: "
                "against the same movement, ry negated spreads "
                f"{_degrees(negated)} degrees per tracked epoch and ry as stored "
                f"{_degrees([spread for _, _, spread in comparisons])}, so ry "
                "turns the way the (x, -z) frame does. Its zero is not verified "
                "(see the watch-out)",
            ),
            Conversion(
                "epochs: 'sleep' and 'wake' from the tags of the epochs table; "
                "'sws', 'up' and 'down' from the SWS, UP and DOWN tables; "
                "'tracking' from position_time_support",
                "stated",
                "the file's intervals: epochs (tagged sleep or wake), SWS, UP, "
                "DOWN and position_time_support",
            ),
            *arena_line,
        ),
        watch_outs=(
            *WATCH_OUTS.get(entry.session, ()),
            _frame_watch_out(comparisons),
            *_two_arenas(arenas),
            _UNLOADED,
        ),
    )


def _arena_epochs(
    description: str, tracked: NDArray[np.float64]
) -> tuple[dict[str, NDArray[np.float64]], tuple[Conversion, ...]]:
    """The tracked epochs named by the arena the session_description lists.

    The description names the arenas in the order they were visited, for
    example "sleep, square arena, circular arena, sleep"; each is matched
    with one row of ``position_time_support``, in order. With a different
    number of arenas and rows, no epoch is named.

    Returns
    -------
    epochs : dict of str to ndarray, shape (n, 2)
        ``square``, ``circle`` and ``partial_circle`` where named.
    conversions : tuple of Conversion
        The naming, empty when none is made.
    """
    named = [_ARENA_NAMES[match] for match in _ARENA.findall(description)]
    if len(named) != len(tracked):
        return {}, ()
    epochs: dict[str, NDArray[np.float64]] = {}
    for name in dict.fromkeys(named):
        epochs[name] = tracked[[found == name for found in named]]
    return epochs, (
        Conversion(
            "epochs: the rows of position_time_support named, in order, by the "
            f"arenas the session_description lists ({', '.join(named)})",
            "stated",
            f"the file's session_description: {description!r}",
        ),
    )


def _heading_by_epoch(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
    tracked: NDArray[np.float64],
) -> list[tuple[str, float, float]]:
    """Heading minus the direction of movement in each tracked epoch.

    Samples are thinned to about ``COMPARE_RATE`` Hz and compared above
    ``MOVING`` cm/s, with
    :func:`~neurospatial.datasets._recording.heading_vs_movement`.

    Returns
    -------
    list of (text, mean, spread)
        One per tracked epoch: its description, and the circular mean and
        spread in radians.
    """
    step = max(round(1.0 / (COMPARE_RATE * float(np.median(np.diff(times))))), 1)
    found = []
    for start, stop in tracked:
        inside = (times >= start) & (times < stop)
        mean, spread = heading_vs_movement(
            times[inside][::step],
            positions[inside][::step],
            headings[inside][::step],
            min_speed=MOVING,
        )
        found.append(
            (
                f"{start:.0f} to {stop:.0f} s, mean {mean:+.2f} rad, spread "
                f"{np.degrees(spread):.0f} degrees",
                mean,
                spread,
            )
        )
    return found


def _frame_watch_out(comparisons: list[tuple[str, float, float]]) -> str:
    """Whether heading follows movement in every tracked epoch, with numbers."""
    verified = all(
        abs(mean) <= FRAME_MEAN and spread < FRAME_SPREAD
        for _, mean, spread in comparisons
    )
    numbers = "; ".join(text for text, _, _ in comparisons)
    rule = (
        f"heading minus the direction of movement above {MOVING:g} cm/s, at "
        f"{COMPARE_RATE:g} Hz, per tracked epoch: {numbers}. The rule for a "
        f"verified frame is |mean| <= {FRAME_MEAN:g} rad and a spread under "
        f"{np.degrees(FRAME_SPREAD):.0f} degrees in every epoch."
    )
    if verified:
        return (
            f"The heading frame follows movement: {rule} The headings are "
            "kept as stored."
        )
    return (
        f"The heading frame is not verified by movement: {rule} A constant "
        "mean is an offset of the rigid body's zero or of the head's posture "
        "while running, which the file cannot tell apart; a mean that changes "
        "between epochs means the zero moved. The headings are kept as "
        "stored: tuning-curve widths and mean vector lengths do not depend on "
        "the frame, preferred directions and comparisons across epochs do."
    )


def _two_arenas(arenas: dict[str, NDArray[np.float64]]) -> tuple[str, ...]:
    """A watch-out when one environment covers more than one arena."""
    if len(arenas) < 2:
        return ()
    return (
        f"This session visits {len(arenas)} arenas ({', '.join(arenas)}), "
        "tracked in one frame, so rec.env, inferred from every sample, covers "
        "all of them. For one arena's maps, build its environment from its "
        "epoch's samples (Environment.open_field on the positions inside "
        "rec.epochs[name]) and pass epochs=rec.epochs[name].",
    )


def _inferred_environment(positions: NDArray[np.float64], session: str) -> Environment:
    """The open field the tracked samples cover, in ``BIN_SIZE`` cm bins."""
    from neurospatial import Environment

    return Environment.open_field(
        positions, bin_size=BIN_SIZE, units="cm", name=f"peyrache_rsc {session}"
    )


def _degrees(spreads: list[float]) -> str:
    """``"26 to 52"``: the range of spreads in radians, as whole degrees."""
    low, high = np.degrees(min(spreads)), np.degrees(max(spreads))
    return f"{low:.0f}" if round(low) == round(high) else f"{low:.0f} to {high:.0f}"


def _identifier_date(identifier: str) -> str:
    """``"2021-07-26"`` from an identifier like ``A7803-210726``; empty otherwise."""
    match = re.fullmatch(r"[A-Za-z0-9]+-(\d{2})(\d{2})(\d{2})", identifier)
    if match is None:
        return ""
    year, month, day = match.groups()
    return f"20{year}-{month}-{day}"
