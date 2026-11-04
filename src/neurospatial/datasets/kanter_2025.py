"""Kanter et al. (2025): rat LEC, MEC and hippocampus in open fields, with novel objects (EBRAINS).

What was recorded: Neuropixels 2.0 units in lateral entorhinal cortex (LEC),
medial entorhinal cortex (MEC) and the hippocampus (CA1 or CA3) of nine
Long Evans rats, while they foraged for scattered food in a square open
field of 1, 1.5 or 2 m with 50 cm walls (the data descriptor). In the novel
object task the rat foraged 7.5 min in the empty arena, then a novel object
(about 10 x 10 cm, 15 cm tall) was placed at a pseudorandom location for
7.5 min, then a second and a third, each at a different location: the
sequence Empty, OBJ1, OBJ2, OBJ3. Head position and head direction come
from five retroreflective markers on the implant, tracked by OptiTrack
motion capture and projected onto the horizontal plane.

Where it lives: EBRAINS dataset 10.25493/GTVB-FQR, CC-BY-SA-4.0, a public
bucket on the EBRAINS data proxy (no login). One NWB file per session; the
bucket's ``docs/index.json`` lists each file's tasks. The loader reads the
31 files whose epochs are all open-field foraging or novel objects (the 30
the index lists, and ``27963_2022-Feb-11_14-58-41``, which it leaves out);
the figure-eight, odor-sequence and sleep files are not read. The default,
``29629_2024-Oct-29_15-05-13`` (29.6 MB), is a novel-object session of rat
29629 with 156 LEC and 224 CA1 units.

Sources read: the data descriptor (``data-descriptor_eebc19898ded.pdf`` in
the bucket), ``docs/index.json``, ``scripts/nwb.py`` and the files. The
associated article was not read.

What the loader converts:

- positions: ``processing/behavior/position/animal_position`` columns x and
  y, in cm. The file states ``unit="meters"`` with ``conversion=0.01``, so
  the stored values are cm; ``read_position`` returns metres, multiplied by
  100. The third column (the head's height) is not used: the data
  descriptor says the markers were projected onto the horizontal plane.
- headings: ``direction/head_direction`` (radians, 0 to 2π, frame "(x=1,
  y=0) unit vector") wrapped to [-π, π]. Whether the frame agrees with the
  positions' is checked against the direction of movement above 20 cm/s at
  10 Hz; where it does not (an offset beyond 0.2 rad), the values are kept
  and the watch-outs give the offset, its spread and how it drifts over the
  session.
- unit ids: the units table repeats ids across the probes, so
  :func:`neurospatial.io.nwb.read_units` refuses it. The loader reads the
  table with pynwb and names each unit ``<area>-<id>`` (``LEC-12``,
  ``CA1-12``); each probe records one area, so these are unique.
- epochs: the epochs table's rows. A session the index lists as a novel
  object task has four rows, named ``empty``, ``object_1``, ``object_2``
  and ``object_3`` in time order (the data descriptor's sequence); every
  other session's rows are ``foraging``. Rows that run past the recording
  (its first to last sample or spike) are clipped to it
  (``clip_epochs_to_recording``, an inferred ``Conversion`` and a
  watch-out): in four sessions an epoch ends 14.7 to 31.9 s after the data.

Object positions: not in the archive. The files hold no object table or
location, the data descriptor says only that each object was placed "at a
pseudorandom location", ``docs/index.json`` lists tasks, and the dataset
publishes no video. The object-vector analyses, which take
``object_positions``, cannot run on these sessions.

The declared environment is the session's stated arena, the epochs table's
``arena_dimensions`` (a square, in cm), with 2.5 cm bins. The stated
reference frame ("Bottom left of the arena") does not hold: positions run
negative. So the square keeps the file's axes and is placed where it holds
the most tracked samples (an inferred ``Conversion``).

Analysis choices (``params``):

- ``bin_size`` 2.5 cm, ``min_speed`` None (foraging, as for the other open
  fields), ``dt`` 0.25 s;
- ``train_epochs`` and ``test_epochs``: alternating 60 s blocks.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from neurospatial.datasets._params import alternating_blocks
from neurospatial.datasets._recording import (
    Conversion,
    Recording,
    clip_epochs_to_recording,
    heading_vs_movement,
    interval_rows,
    tracked_mask,
)

if TYPE_CHECKING:
    import pandas as pd

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
        "The data descriptor reports no result, and the associated article "
        "(on event structure in the entorhinal cortex and hippocampus) was "
        "not read, so no published number can be reproduced from a source "
        "that was read. The object-related results the novel-object sessions "
        "invite need the objects' positions, which the archive does not hold."
    ),
}

#: Modules of the ``datasets`` extra the loader imports.
REQUIRES = ("pynwb",)

BIN_SIZE = 2.5
DT = 0.25
BLOCK = 60.0
#: Heading is compared with the direction of movement above this speed, cm/s.
HEADING_SPEED = 20.0
#: ... at about this rate, Hz: at 120 Hz a step's direction is mostly jitter.
HEADING_RATE = 10.0
#: Beyond this offset (rad) from the direction of movement, the heading frame
#: is not taken as the positions' frame.
FRAME_TOLERANCE = 0.2
#: Windows (s) over which the offset's drift is measured, and the running
#: samples each needs.
DRIFT_WINDOW = 60.0
DRIFT_MIN_SAMPLES = 10
#: Placements of the stated square are searched on this grid, cm.
PLACEMENT_STEP = 0.5
#: A sample's coordinates below this (cm) are negative beyond float rounding.
_ROUNDING = 1e-6
#: The data descriptor's object epochs, in order.
OBJECT_EPOCHS = ("empty", "object_1", "object_2", "object_3")
#: Short area names for unit ids, from the electrode groups' locations.
AREAS = {"LEC": "LEC", "MEC - Layer 2": "MEC", "HP - CA1": "CA1", "HP - CA3": "CA3"}
_DESCRIPTOR = "the data descriptor (data-descriptor_eebc19898ded.pdf)"
_INDEX = "the bucket's docs/index.json"

#: The sessions ``docs/index.json`` lists with the novel object task.
NOVEL_OBJECT_SESSIONS = frozenset(
    {
        "27963_2022-Feb-10_11-06-56",
        "27963_2022-Feb-10_14-07-47",
        "28003_2022-Mar-08_15-04-04",
        "28003_2022-Mar-10_10-53-58",
        "29630_2024-Sep-17_09-22-08",
        "29630_2024-Sep-18_10-18-21",
        "29630_2024-Sep-18_15-19-16",
        "29630_2024-Sep-20_09-49-05",
        "29629_2024-Oct-29_15-05-13",
        "29629_2024-Oct-30_15-13-05",
    }
)

#: Findings computed from the data (by the checks or this loader) that each
#: verified session must report, as patterns searched for in its watch_outs.
EXPECTED_WATCH_OUTS: dict[str, tuple[str, ...]] = {
    "29629_2024-Oct-29_15-05-13": (
        r"The units table repeats unit ids across its probes \(380 rows, 235 "
        r"distinct ids\)",
        r"The heading frame is not verified by movement",
        r"The file's reference frame says 'Bottom left of the arena'",
    ),
}

#: Pitfalls of each session checked by hand, with their evidence.
WATCH_OUTS: dict[str, tuple[str, ...]] = {
    "29629_2024-Oct-29_15-05-13": (
        "The empty-arena epoch runs 605 s (43 to 648 s), not the data "
        "descriptor's 7.5 min; the three object epochs run 450, 523 and 452 s. "
        "The epochs table's rows are used as stored.",
    ),
}

_NO_OBJECTS = (
    "No object positions: the files hold no object location, the data "
    "descriptor (data-descriptor_eebc19898ded.pdf) says only that each object "
    "was placed 'at a pseudorandom location', and the dataset publishes no "
    "video, so "
    "object-vector analyses (object_positions) cannot run. The epochs say when "
    "each object was in the arena, not where."
)


def load_session(
    entry: SessionEntry, path_or_stream: str | os.PathLike[str]
) -> Recording:
    """Read one session's NWB file into a :class:`Recording`.

    Parameters
    ----------
    entry : SessionEntry
        The session's registry entry.
    path_or_stream : str or path
        The cached NWB file.

    Returns
    -------
    Recording
        Unchecked; :func:`neurospatial.datasets.load` runs the checks.

    Raises
    ------
    ValueError
        If the position series is not in metres after the file's conversion,
        head direction and position stop sharing timestamps, the epochs
        table names more than one arena or a non-square one, or a
        novel-object session does not have four epochs.
    """
    from neurospatial.datasets._fetch import open_nwb
    from neurospatial.io.nwb import read_head_direction, read_position
    from neurospatial.stats import wrap_angle

    with open_nwb(os.fspath(path_or_stream)) as nwb:
        position = read_position(nwb, columns=[0, 1])
        stored = nwb.processing["behavior"]["position"]["animal_position"]
        stated_unit = f"unit {stored.unit!r}, conversion {stored.conversion:g}"
        frame = str(stored.reference_frame)
        heading = read_head_direction(nwb)
        heading_frame = str(
            nwb.processing["behavior"]["direction"]["head_direction"].reference_frame
        )
        file_units = nwb.units.to_dataframe()
        locations = [group.location for group in file_units["electrode_group"]]
        epoch_table = nwb.epochs.to_dataframe()
        subject = str(nwb.subject.subject_id)
        started = nwb.session_start_time
    if position.units != "m":
        msg = (
            f"{entry.session}: read_position gives {position.units!r}, not metres "
            f"(the file's {stated_unit}); recheck the cm reading"
        )
        raise ValueError(msg)
    times = np.asarray(position.times, dtype=np.float64)
    positions = np.asarray(position.positions, dtype=np.float64) * 100.0
    if not np.array_equal(np.asarray(heading.times), times):
        msg = f"{entry.session}: head direction and position no longer share timestamps"
        raise ValueError(msg)
    raw = np.asarray(heading.headings, dtype=np.float64)
    headings = np.asarray(wrap_angle(raw), dtype=np.float64)
    side, arena = _stated_arena(epoch_table, entry.session)
    epochs, epoch_source = _epochs(epoch_table, entry.session)
    spike_times = tuple(
        np.sort(np.asarray(train, dtype=np.float64))
        for train in file_units["spike_times"]
    )
    clipped = clip_epochs_to_recording(
        epochs, times, spike_times, source="the file's epochs table"
    )
    env, placement, span_watch_out = _declared_square(positions, side, entry.session)
    table, regions = unit_table(file_units, locations)
    frame_conversion, frame_watch_outs = _heading_frame(times, positions, headings)
    train, test = alternating_blocks(times[0], times[-1], BLOCK)
    is_object_session = entry.session in NOVEL_OBJECT_SESSIONS
    return Recording(
        **entry.identity(path_or_stream),
        animal=f"rat {subject}",
        apparatus=(
            f"{side:g} cm square open field ({arena}), "
            + ("novel objects" if is_object_session else "foraging")
        ),
        brain_regions=regions or entry.info.brain_regions,
        clock=(
            "the file's clock, seconds from its session start time "
            f"({started:%Y-%m-%d %H:%M})"
        ),
        times=times,
        spike_times=spike_times,
        units=table,
        env=env,
        env_kind="declared",
        env_source=(
            f"the stated {side:g} x {side:g} cm square arena of the file's "
            f"epochs table (arena {arena!r}, arena_dimensions in cm; "
            f"{_DESCRIPTOR}: square arenas of 1, 1.5 or 2 m with 50 cm walls), "
            f"{BIN_SIZE:g} cm bins"
        ),
        positions=positions,
        position_units="cm",
        tracking=(
            "the head: five retroreflective markers on the implant, tracked by "
            "OptiTrack motion capture and projected onto the horizontal plane "
            f"({_DESCRIPTOR})"
        ),
        headings=headings,
        heading_source=(
            "the file's head direction (direction/head_direction), from the "
            "motion-capture markers"
        ),
        epochs=clipped.epochs,
        params={
            "bin_size": BIN_SIZE,
            "dt": DT,
            "min_speed": None,
            "train_epochs": train,
            "test_epochs": test,
        },
        conversions=(
            Conversion(
                "positions: cm = animal_position columns x and y in m "
                "(read_position applies the file's conversion) x 100; the third "
                "column, the head's height, is not used",
                "stated",
                f"the file's animal_position: {stated_unit}; {_DESCRIPTOR}: the "
                "3D marker positions 'were projected onto the horizontal plane to "
                "yield the rat's 2D position and head direction'",
            ),
            Conversion(
                "headings: head_direction wrapped from [0, 2π) to [-π, π] "
                "(neurospatial.stats.wrap_angle)",
                "stated",
                f"the file's head_direction: unit 'radians', reference frame "
                f"{heading_frame!r}, values {raw.min():.3g} to {raw.max():.3g}",
            ),
            frame_conversion,
            Conversion(
                "units: ids <area>-<id> from the file's ids and each unit's "
                "electrode group location",
                "stated",
                f"the file's units table: {len(file_units)} rows, "
                f"{file_units.index.nunique()} distinct ids; each electrode group "
                "(probe) has one location",
            ),
            epoch_source,
            *(() if clipped.conversion is None else (clipped.conversion,)),
            placement,
        ),
        watch_outs=(
            *WATCH_OUTS.get(entry.session, ()),
            *(() if clipped.watch_out is None else (clipped.watch_out,)),
            *_repeated_ids(file_units),
            *frame_watch_outs,
            *_frame_claim(positions, frame),
            *span_watch_out,
            *_task_disagreement(epoch_table, entry.session),
            *((_NO_OBJECTS,) if is_object_session else ()),
            _unit_quality(table),
        ),
    )


def unit_table(
    file_units: pd.DataFrame, locations: list[str]
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """The units table, ids ``<area>-<id>``, and the units' brain regions.

    Parameters
    ----------
    file_units : pandas.DataFrame
        The file's units table (``nwb.units.to_dataframe()``), indexed by its
        own ids, which may repeat across probes.
    locations : list of str
        Each unit's electrode group location, in table order.

    Returns
    -------
    table : pandas.DataFrame
        Indexed by ``unit_id``; ``brain_region`` is the file's location,
        ``label`` None, and ``file_unit_id`` and ``cluster_quality`` are the
        file's.
    regions : tuple of str
        The locations, in order of first appearance.

    Raises
    ------
    ValueError
        If a location has no short area name, or two units get one id.
    """
    import pandas as pd

    if unknown := sorted(set(locations) - set(AREAS)):
        msg = f"electrode group locations {unknown} have no short area name"
        raise ValueError(msg)
    ids = [f"{AREAS[a]}-{i}" for a, i in zip(locations, file_units.index, strict=True)]
    if len(set(ids)) != len(ids):
        msg = "two units share an area and an id; <area>-<id> would not be unique"
        raise ValueError(msg)
    table = pd.DataFrame(
        {
            "brain_region": locations,
            "label": None,
            "file_unit_id": file_units.index.to_numpy(),
            "cluster_quality": file_units["cluster_quality"].to_numpy(),
        },
        index=pd.Index(ids, name="unit_id"),
    )
    return table, tuple(dict.fromkeys(locations))


def _stated_arena(epoch_table: pd.DataFrame, session: str) -> tuple[float, str]:
    """The side (cm) and name of the one square arena the epochs table states."""
    arenas = set(epoch_table["arena"])
    sides = {
        tuple(np.asarray(d, dtype=float)[:2]) for d in epoch_table["arena_dimensions"]
    }
    shapes = set(epoch_table["arena_geometry"])
    if len(arenas) != 1 or len(sides) != 1 or shapes != {"square"}:
        msg = (
            f"{session}: the epochs table states arenas {sorted(arenas)}, shapes "
            f"{sorted(shapes)} and sizes {sorted(sides)}; the loader declares one "
            "square arena per session"
        )
        raise ValueError(msg)
    ((width, height),) = sides
    if width != height:
        msg = f"{session}: the 'square' arena is {width:g} x {height:g} cm"
        raise ValueError(msg)
    return float(width), str(next(iter(arenas)))


def _epochs(
    epoch_table: pd.DataFrame, session: str
) -> tuple[dict[str, NDArray[np.float64]], Conversion]:
    """The named epochs and how they were named."""
    rows = interval_rows(epoch_table)
    tasks = sorted(set(epoch_table["task"]))
    if session not in NOVEL_OBJECT_SESSIONS:
        return {"foraging": rows}, Conversion(
            f"epochs['foraging']: the epochs table's {len(rows)} row(s)",
            "stated",
            f"the file's epochs table, tasks {tasks}; {_INDEX} lists no novel "
            "object task for this file",
        )
    if len(rows) != len(OBJECT_EPOCHS):
        msg = (
            f"{session}: a novel-object session with {len(rows)} epochs, not the "
            f"{len(OBJECT_EPOCHS)} of the sequence Empty, OBJ1, OBJ2, OBJ3"
        )
        raise ValueError(msg)
    epochs = {name: rows[k : k + 1] for k, name in enumerate(OBJECT_EPOCHS)}
    return epochs, Conversion(
        "epochs: the epochs table's four rows in time order named "
        f"{', '.join(OBJECT_EPOCHS)}",
        "stated",
        f"{_DESCRIPTOR}: 'the following sequence: Empty, OBJ1, OBJ2, OBJ3', each "
        f"about 7.5 min; {_INDEX} lists this file's rows as the novel object "
        f"task; the file's tasks {tasks}",
    )


def place_square(
    positions: NDArray[np.float64], side: float
) -> tuple[NDArray[np.float64], int]:
    """Where a ``side`` square with the samples' axes holds the most tracked samples.

    Placements are searched on a ``PLACEMENT_STEP`` grid of lower-left
    corners. Where several hold as many samples (a square larger than the
    samples' range), the centre of that range of placements is taken, so a
    square that holds every sample is centred on their range.

    Parameters
    ----------
    positions : ndarray, shape (n_time, 2)
    side : float
        The square's side, in the positions' unit.

    Returns
    -------
    centre : ndarray, shape (2,)
    n_inside : int
        Tracked samples inside the square at the best placement.
    """
    tracked = positions[tracked_mask(positions)]
    step = PLACEMENT_STEP
    k = round(side / step)
    low = np.floor(tracked.min(axis=0) / step) * step - side
    n_cells = np.ceil((tracked.max(axis=0) + side - low) / step).astype(int) + 1
    cells = np.floor((tracked - low) / step).astype(int)
    counts = np.zeros(n_cells + 1, dtype=np.int64)
    np.add.at(counts, (cells[:, 0] + 1, cells[:, 1] + 1), 1)
    summed = counts.cumsum(axis=0).cumsum(axis=1)
    # Samples in cells [i, i + k) x [j, j + k), for every lower-left cell (i, j).
    inside = summed[k:, k:] - summed[:-k, k:] - summed[k:, :-k] + summed[:-k, :-k]
    best = int(inside.max())
    corners = np.argwhere(inside == best)
    centre = low + step * corners.mean(axis=0) + side / 2
    return centre, best


def _declared_square(
    positions: NDArray[np.float64], side: float, session: str
) -> tuple[Environment, Conversion, tuple[str, ...]]:
    """The stated square placed on the samples, how, and a span watch-out."""
    import shapely

    from neurospatial import Environment

    centre, n_inside = place_square(positions, side)
    tracked = positions[tracked_mask(positions)]
    half = side / 2
    env = Environment.from_polygon(
        shapely.box(*(centre - half), *(centre + half)),
        bin_size=BIN_SIZE,
        units="cm",
        name=f"kanter_2025 {session}",
    )
    low, high = np.percentile(tracked, [0.5, 99.5], axis=0)
    spans = high - low
    turn = rectangle_turn(tracked)
    placement = Conversion(
        f"environment: the stated {side:g} x {side:g} cm square centred at "
        f"({centre[0]:.3g}, {centre[1]:.3g}) cm, sides along the file's x and y",
        "inferred",
        f"the file's reference frame says the origin is the arena's bottom left, "
        "which the positions contradict (see watch-outs), so the square is placed "
        f"where it holds the most tracked samples ({n_inside} of {len(tracked)}, "
        "centred among equally good placements); the samples' 0.5th to 99.5th "
        f"percentile range is {spans[0]:.4g} x {spans[1]:.4g} cm against the "
        f"stated {side:g} cm; their minimum rotated rectangle is turned "
        f"{turn:+.1f} degrees from the file's axes, so the axes are kept",
    )
    watch: tuple[str, ...] = ()
    over = spans - side
    if np.any(over > BIN_SIZE) or np.any(spans < 0.8 * side):
        watch = (
            "The tracked samples' 0.5th to 99.5th percentile range, "
            f"{spans[0]:.4g} x {spans[1]:.4g} cm, does not match the {side:g} cm "
            "square arena the epochs table states (more than one bin past it, or "
            "more than 20% short of it): the arena, the session's apparatus or the "
            "tracking differs from what the file states. The environment is the "
            "stated square, placed to hold the most samples; the positions are "
            "kept as stored.",
        )
    return env, placement, watch


def rectangle_turn(points: NDArray[np.float64]) -> float:
    """How far the points' minimum-area bounding rectangle is turned from the axes.

    The minimum-area rectangle has a side along an edge of the convex hull,
    so each hull edge's direction is tried.

    Parameters
    ----------
    points : ndarray, shape (n_points, 2)
        Finite points, not all on one line.

    Returns
    -------
    float
        Degrees, in [-45, 45).
    """
    from scipy.spatial import ConvexHull

    hull = points[ConvexHull(points).vertices]
    edges = np.roll(hull, -1, axis=0) - hull
    angles = np.unique(np.mod(np.arctan2(edges[:, 1], edges[:, 0]), np.pi / 2))
    cos, sin = np.cos(angles), np.sin(angles)
    # Hull points in each candidate frame: shape (n_angles, n_hull).
    u = hull[:, 0] * cos[:, None] + hull[:, 1] * sin[:, None]
    v = -hull[:, 0] * sin[:, None] + hull[:, 1] * cos[:, None]
    area = np.ptp(u, axis=1) * np.ptp(v, axis=1)
    turn = float(np.degrees(angles[np.argmin(area)]))
    return turn - 90.0 if turn >= 45.0 else turn


def _heading_frame(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
) -> tuple[Conversion, tuple[str, ...]]:
    """Heading against the direction of movement: the frame's conversion and watch-out."""
    every = max(round(1.0 / (HEADING_RATE * float(np.median(np.diff(times))))), 1)
    t, xy, h = times[::every], positions[::every], headings[::every]
    speed = np.linalg.norm(np.diff(xy, axis=0), axis=1) / np.diff(t)
    if (n_running := int(np.sum(speed > HEADING_SPEED))) < DRIFT_MIN_SAMPLES:
        return Conversion(
            "headings: allocentric in the positions' frame, 0 = +x and "
            "counter-clockwise; not rotated or mirrored",
            "inferred",
            "the file's head_direction frame is '(x=1, y=0) unit vector'; with "
            f"{n_running} samples faster than {HEADING_SPEED:g} cm/s the frame "
            "could not be compared with the direction of movement",
        ), (
            "The heading frame is not verified by movement: the rat ran faster "
            f"than {HEADING_SPEED:g} cm/s in only {n_running} samples at about "
            f"{HEADING_RATE:g} Hz, too few to compare. The stored head direction "
            "is kept.",
        )
    mean, spread = heading_vs_movement(t, xy, h, min_speed=HEADING_SPEED)
    _, negated = heading_vs_movement(t, xy, -h, min_speed=HEADING_SPEED)
    windows = _drift(t, xy, h)
    measured = (
        f"heading minus the direction of movement above {HEADING_SPEED:g} cm/s "
        f"(every {every}th sample, about {HEADING_RATE:g} Hz): circular mean "
        f"{mean:+.2f} rad, median spread {np.degrees(spread):.0f} degrees; the "
        f"heading negated gives a spread of {np.degrees(negated):.0f} degrees"
    )
    if windows is not None:
        n, lowest, highest, sd = windows
        measured += (
            f"; over {n} windows of {DRIFT_WINDOW:g} s with at least "
            f"{DRIFT_MIN_SAMPLES} running samples the window means run from "
            f"{lowest:+.2f} to {highest:+.2f} rad (circular SD {sd:.0f} degrees)"
        )
    verified = abs(mean) <= FRAME_TOLERANCE and spread < negated
    conversion = Conversion(
        "headings: allocentric in the positions' frame, 0 = +x and "
        "counter-clockwise; not rotated or mirrored",
        "inferred",
        f"the file's head_direction frame is '(x=1, y=0) unit vector'; {measured}"
        + (
            ""
            if verified
            else f"; the offset is beyond {FRAME_TOLERANCE:g} rad, so the frame is "
            "not verified"
        ),
    )
    if verified:
        return conversion, ()
    drift = (
        ""
        if windows is None
        else " The offset is not constant over the session, so no single "
        "correction follows from movement."
        if windows[3] > np.degrees(FRAME_TOLERANCE)
        else ""
    )
    return conversion, (
        f"The heading frame is not verified by movement: {measured}.{drift} "
        "The stored head direction is kept; tuning-curve shapes and widths do not "
        "depend on the frame, while absolute preferred directions and conjunctive "
        "position-direction comparisons do.",
    )


def _drift(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
) -> tuple[int, float, float, float] | None:
    """The heading offset per ``DRIFT_WINDOW`` window: count, range, circular SD (deg)."""
    from neurospatial._angles import wrap_angle

    step = np.diff(positions, axis=0)
    speed = np.linalg.norm(step, axis=1) / np.diff(times)
    running = (speed > HEADING_SPEED) & np.isfinite(headings[:-1])
    offset = wrap_angle(headings[:-1] - np.arctan2(step[:, 1], step[:, 0]))
    window = ((times[:-1] - times[0]) // DRIFT_WINDOW).astype(int)
    means = []
    for k in np.unique(window[running]):
        mine = running & (window == k)
        if mine.sum() >= DRIFT_MIN_SAMPLES:
            means.append(float(np.angle(np.mean(np.exp(1j * offset[mine])))))
    if len(means) < 2:
        return None
    length = float(np.abs(np.mean(np.exp(1j * np.asarray(means)))))
    sd = float(np.degrees(np.sqrt(-2.0 * np.log(max(length, 1e-12)))))
    return len(means), min(means), max(means), sd


def _repeated_ids(file_units: pd.DataFrame) -> tuple[str, ...]:
    """A watch-out when the file's unit ids repeat."""
    n_rows, n_ids = len(file_units), int(file_units.index.nunique())
    if n_ids == n_rows:
        return ()
    return (
        f"The units table repeats unit ids across its probes ({n_rows} rows, "
        f"{n_ids} distinct ids), so a unit is not named by its id alone: "
        "neurospatial.io.nwb.read_units refuses the table, and keying units by "
        "the file's id would merge units of different brain areas. The loader "
        "names units <area>-<id>; units['file_unit_id'] keeps the file's id.",
    )


def _frame_claim(positions: NDArray[np.float64], frame: str) -> tuple[str, ...]:
    """A watch-out when a bottom-left origin is stated but positions run negative."""
    if "bottom left" not in frame.lower():
        return ()
    tracked = positions[tracked_mask(positions)]
    negative = np.any(tracked < -_ROUNDING, axis=1)
    if not negative.any():
        return ()
    low, high = tracked.min(axis=0), tracked.max(axis=0)
    return (
        f"The file's reference frame says {frame!r}, but {negative.mean():.1%} of "
        f"tracked samples have a negative coordinate (x {low[0]:.1f} to "
        f"{high[0]:.1f} cm, y {low[1]:.1f} to {high[1]:.1f} cm), so the origin is "
        "not the arena's corner and the walls' positions are not stated. The "
        "environment is placed on the samples (see its conversion).",
    )


def _task_disagreement(epoch_table: pd.DataFrame, session: str) -> tuple[str, ...]:
    """A watch-out when the file's task names disagree with the index."""
    tasks = list(epoch_table["task"])
    if session in NOVEL_OBJECT_SESSIONS and "Novel object task" not in tasks:
        return (
            f"The file's epochs table names its tasks {sorted(set(tasks))}, while "
            f"{_INDEX} lists this file as a novel object task; the epochs are "
            f"named {', '.join(OBJECT_EPOCHS)} from the index.",
        )
    return ()


def _unit_quality(table: pd.DataFrame) -> str:
    """How many units the sorting labels 'good', the rest kept."""
    quality = table["cluster_quality"].astype(str)
    counts = quality.value_counts()
    listed = ", ".join(f"{n} {name!r}" for name, n in counts.items())
    return (
        f"Unit quality from the spike sorting: {listed} of {len(table)} units; "
        "every unit is kept, 'mua' included, and units['cluster_quality'] holds "
        "the label."
    )
