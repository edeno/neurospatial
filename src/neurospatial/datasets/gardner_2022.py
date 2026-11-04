"""Gardner et al. (2022): rat MEC grid modules in a 1.5 m open field (Figshare 16764508).

What was recorded: Neuropixels units in layers II and III of the medial
entorhinal cortex-parasubiculum region of three Long Evans rats (Q, R and S;
R on two days), four recordings in all. Each recording holds an open-field
session in a 1.5 x 1.5 m box, and wagon-wheel track or sleep-box sessions,
with the rat connected throughout. Head position and azimuth (head
direction) come from 3-D motion capture (OptiTrack, five markers on the
implant) projected onto the horizontal plane, at 100 Hz. The archive keeps
only the units the paper assigned to grid modules: 163 (rat Q, modules 1-2),
483 (rat R day 1, modules 1-3), 544 (rat R day 2, modules 1-3) and 140
(rat S, module 1), the paper's "from 140 to 544 grid cells per session".

Where it lives: Figshare 16764508 ("Toroidal topology of population activity
in grid cells", version 6), CC0, one 320 MB zip. A session is one recording:
two members read by HTTP byte range, ``<rat>_grid_modules_<m>.npz`` and
``<rat>_sessions.txt``.

Sources read: the paper's Methods (Gardner et al. 2022, Nature 602, 123,
open access, PMC8810387), the Figshare record and the files.

Reading the arrays safely: ``x``, ``y``, ``z``, ``t`` and ``azimuth`` are plain
arrays. ``spikes_mod<m>`` are 0-d object arrays, which ``np.load`` only reads
with ``allow_pickle=True``, letting the file run code. Their pickles import
only ``numpy.core.multiarray._reconstruct``, ``numpy.ndarray`` and
``numpy.dtype`` (dtypes ``O8`` and ``f8``) and hold a dict of unit index to
spike times (s). They are read by ``_pickled.read_npz``, which lists each
pickle's imports before running it and resolves only those names.

What the loader converts:

- only the recording's open-field session is loaded: the sessions text
  gives its start and end (``open_field_1``), and the samples and spikes
  between them are kept. The other sessions are on other apparatus (the
  wagon-wheel track, the 40 x 40 cm sleep box; "each in a different arena
  within the same room", the Methods), and are named under "Watch out".
- ``x``, ``y`` in metres, times 100 to cm. No source read states the unit;
  it is metres because the open-field samples span 1.43 to 1.48 m (rat R
  day 2: 1.56 m) against the stated 1.5 m box (an inferred ``Conversion``).
- ``azimuth`` is radians in [0, 2π), wrapped to [-π, π]. Its frame (0 = East,
  counter-clockwise) is not stated; minus the direction of movement while
  running it has a circular mean within 5 degrees of 0 in every recording,
  so it is taken to be in the positions' frame (an inferred ``Conversion``
  with the measured offset).
- ``epochs["open_field"]``: the session's ``valid_times`` where the sessions
  text gives them (rat R day 1), else its start and end;
  ``params["analysis_epochs"]`` is the same.

The declared environment is the stated "square open-field (OF) arena of
size 1.5 x 1.5 m" with 50 cm walls (the Methods), 3 cm bins. The files state
no origin or rotation, so the square is centred and rotated on the samples'
minimum rotated rectangle; the samples' own sides against 150 cm are in its
``Conversion``.

Analysis choices (``params``), from the Methods' "Spatial position and
direction tuning":

- ``bin_size`` 3 cm ("a square grid of 3 x 3-cm bins");
- ``min_speed`` 2.5 cm/s ("only time epochs in which the rat was moving at a
  speed above 2.5 cm s-1 were used for spatial or toroidal analyses");
- ``dt`` 0.25 s and ``train_epochs`` / ``test_epochs``, alternating 60 s
  blocks inside the analysis epochs, for decoding;
- ``module``: each unit's grid module, also ``units["module"]``.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

from neurospatial.datasets._arenas import PLACEMENT_TRIM, fitted_rectangle
from neurospatial.datasets._params import alternating_blocks
from neurospatial.datasets._recording import (
    Conversion,
    Recording,
    heading_vs_movement,
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
        "The paper's result is the toroidal topology of each module's "
        "population activity, found by persistent cohomology, which "
        "neurospatial does not compute. The one grid score the text read "
        "states (mean 0.677 +/- 0.017 over 618 cells, Extended Data Fig. 2f) "
        "is of open-field maps masked to the bins visited on the wagon-wheel "
        "track, pooled over all four recordings; the masking and the paper's "
        "rate-map interpolation are not neurospatial steps, and two of the "
        "recordings have no wagon-wheel session in their file."
    ),
}

#: Modules of the ``datasets`` extra the loader imports.
REQUIRES = ("fsspec",)

#: The side of the stated open-field box, in cm.
SIDE = 150.0
#: The Methods' rate-map bin, cm.
BIN_SIZE = 3.0
#: The Methods' speed threshold for spatial analyses, cm/s.
MIN_SPEED = 2.5
DT = 0.25
BLOCK = 60.0
#: Steps faster than this (cm/s) count as running in the heading check.
RUNNING = 10.0
#: Heading minus the direction of movement, beyond which the heading's
#: frame is not taken as verified: the circular mean (rad) and median spread.
MAX_HEADING_OFFSET = 0.2
MAX_HEADING_SPREAD = np.radians(45.0)
_PAPER = "the Methods of Gardner et al. 2022"

#: Findings computed from the data (by the checks or this loader) that each
#: verified session must report, as patterns searched for in its watch_outs.
EXPECTED_WATCH_OUTS: dict[str, tuple[str, ...]] = {
    "rat_r_day1": (
        r"leave out 14778 to 14890 s \(\d+ samples\) of open_field_1",
        r"also holds wagon_wheel_1 \(16925 to 20704 s\)",
        r"1 spike repeats the time of the spike before it",
    ),
}

#: Pitfalls of each session checked by hand, with their evidence.
WATCH_OUTS: dict[str, tuple[str, ...]] = {
    "rat_r_day1": (
        "Module 1 is the paper's R1, which its clustering found as two "
        "clusters, R1a and R1b, of similar spacing and orientation; the paper "
        "treats them as one module and the file does not say which unit is "
        f"in which ({_PAPER}, Extended Data Fig. 2).",
    ),
}

_ONLY_GRID = (
    "The archive holds only the units the paper assigned to grid modules "
    "(pure and conjunctive grid cells together, the paper's Extended Data "
    "Fig. 2g), not every unit recorded, so population measures over all MEC "
    "units cannot be computed from it."
)


class _Session(NamedTuple):
    """One line of a ``<rat>_sessions.txt`` file."""

    name: str
    start: float
    end: float
    valid: NDArray[np.float64]


def load_session(
    entry: SessionEntry, path_or_stream: str | os.PathLike[str]
) -> Recording:
    """Read one recording's open-field session into a :class:`Recording`.

    Parameters
    ----------
    entry : SessionEntry
        The session's registry entry.
    path_or_stream : str or path
        The directory holding the fetched zip members.

    Returns
    -------
    Recording
        Unchecked; :func:`neurospatial.datasets.load` runs the checks.

    Raises
    ------
    ValueError
        If the sessions text lists no open-field session or more than one,
        or ``azimuth`` is not radians in [0, 2π].
    pickle.UnpicklingError
        If a spike array's pickle names anything beyond NumPy's array
        reconstruction.
    """
    import pandas as pd

    from neurospatial._angles import wrap_angle
    from neurospatial._intervals import intersect_intervals
    from neurospatial.datasets._pickled import read_npz

    folder = Path(path_or_stream)
    names = [Path(member.path).name for member in entry.members]
    arrays = read_npz(folder / _named(names, ".npz"))
    listed = parse_sessions((folder / _named(names, "_sessions.txt")).read_text())
    session = _open_field(listed, entry.session)
    t = np.asarray(arrays["t"], dtype=np.float64)
    keep = (t >= session.start) & (t <= session.end)
    times = t[keep]
    positions = (
        np.column_stack([arrays["x"], arrays["y"]]).astype(np.float64)[keep] * 100.0
    )
    azimuth = np.asarray(arrays["azimuth"], dtype=np.float64)[keep]
    finite = azimuth[np.isfinite(azimuth)]
    if finite.size and (finite.min() < 0.0 or finite.max() > 2 * np.pi + 1e-6):
        msg = (
            f"{entry.session}: azimuth runs from {finite.min():.3g} to "
            f"{finite.max():.3g}, not radians in [0, 2pi); recheck its unit"
        )
        raise ValueError(msg)
    offset, spread = heading_vs_movement(times, positions, azimuth, min_speed=RUNNING)
    trains, unit_ids, modules = [], [], []
    for key in sorted(k for k in arrays if k.startswith("spikes_mod")):
        module = int(key.removeprefix("spikes_mod"))
        spikes = arrays[key][()]
        for index in sorted(spikes):
            train = np.asarray(spikes[index], dtype=np.float64)
            trains.append(train[(train >= session.start) & (train <= session.end)])
            unit_ids.append(f"mod{module}_{index}")
            modules.append(module)
    table = pd.DataFrame(
        {
            "brain_region": "MEC-parasubiculum",
            "label": [f"grid module {m}" for m in modules],
            "module": modules,
        },
        index=pd.Index(unit_ids, name="unit_id"),
    )
    env, placement = _declared_square(positions, entry.session)
    valid = session.valid
    train, test = (
        intersect_intervals(block, valid)
        for block in alternating_blocks(float(valid[0, 0]), float(valid[-1, 1]), BLOCK)
    )
    span = np.ptp(positions[tracked_mask(positions)], axis=0)
    heading = _heading_conversion(offset, spread)
    return Recording(
        **entry.identity(path_or_stream),
        animal=f"rat {_rat(entry.session)}",
        apparatus=entry.info.apparatus,
        brain_regions=entry.info.brain_regions,
        clock=(
            "the recording's clock, seconds from the start of acquisition "
            f"(the sessions text's times), from {times[0]:.2f} s"
        ),
        times=times,
        spike_times=tuple(trains),
        units=table,
        env=env,
        env_kind="declared",
        env_source=(
            f"the stated {SIDE:g} x {SIDE:g} cm open-field box ({_PAPER}: 'a "
            "square open-field (OF) arena of size 1.5 x 1.5 m', 50 cm walls), "
            f"{BIN_SIZE:g} cm bins"
        ),
        positions=positions,
        position_units="cm",
        tracking=(
            "the head: five retroreflective markers on the implant, tracked in "
            f"3-D by OptiTrack and projected onto the horizontal plane ({_PAPER})"
        ),
        headings=np.asarray(wrap_angle(azimuth), dtype=np.float64),
        heading_source=(
            f"the file's azimuth, the head direction of the motion capture ({_PAPER})"
        ),
        epochs={"open_field": valid},
        params={
            "bin_size": BIN_SIZE,
            "min_speed": MIN_SPEED,
            "dt": DT,
            "analysis_epochs": valid,
            "train_epochs": train,
            "test_epochs": test,
            "module": np.asarray(modules),
        },
        conversions=(
            Conversion(
                "spike_times: spikes_mod<m> read with a restricted unpickler "
                "(numpy _reconstruct, ndarray and the O8/f8 dtypes only); a "
                "dict of unit index to spike times in s; units named "
                "mod<m>_<index>",
                "stated",
                "the pickles' imports, listed from their opcodes before they "
                "run, are those names only",
            ),
            Conversion(
                f"times, positions, headings and spikes: those within "
                f"{session.name}, {session.start:g} to {session.end:g} s of the "
                f"sessions text; the file's {len(t)} samples run from "
                f"{t[0]:.2f} to {t[-1]:.2f} s",
                "stated",
                f"the file's {_named(names, '_sessions.txt')}: "
                + "; ".join(f"{s.name} {s.start:g} to {s.end:g} s" for s in listed),
            ),
            Conversion(
                "positions: x, y in m, times 100 to cm",
                "inferred",
                "no source read states the unit; the open-field samples span "
                f"{span[0]:.4g} x {span[1]:.4g} cm against the stated "
                f"{SIDE / 100:g} x {SIDE / 100:g} m box ({_PAPER})",
            ),
            heading,
            Conversion(
                f"epochs['open_field'] and params['analysis_epochs']: "
                f"{valid.tolist()} s",
                "stated",
                f"{session.name}'s "
                + (
                    "valid_times in the sessions text"
                    if len(valid) > 1 or valid[0, 0] != session.start
                    else "start and end in the sessions text"
                ),
            ),
            placement,
        ),
        watch_outs=(
            *WATCH_OUTS.get(entry.session, ()),
            *_heading_watch_out(offset, spread),
            *_invalid_stretches(times, session),
            _other_sessions(listed, session),
            *_repeated_spikes(trains),
            _ONLY_GRID,
        ),
    )


def parse_sessions(text: str) -> list[_Session]:
    """Every session of a ``<rat>_sessions.txt`` file, in its order.

    Each line reads ``#1,  open_field_1,  start=7457,  end=16045`` and may
    end with ``valid_times=[7457,14778;14890,16045]``.

    Parameters
    ----------
    text : str
        The file's contents.

    Returns
    -------
    list of _Session
        Name, start and end (s), and the valid ``(n, 2)`` ranges: the
        ``valid_times`` where given, else ``[[start, end]]``.

    Raises
    ------
    ValueError
        If a line that is not blank does not read as above.

    Examples
    --------
    >>> [
    ...     s.name
    ...     for s in parse_sessions("#1, a_1, start=0, end=5\\n#2, b_1, start=6, end=9")
    ... ]
    ['a_1', 'b_1']
    >>> parse_sessions("#1, a_1, start=0, end=9, valid_times=[0,4;5,9]")[
    ...     0
    ... ].valid.tolist()
    [[0.0, 4.0], [5.0, 9.0]]
    """
    pattern = re.compile(
        r"#\d+,\s*(?P<name>\w+),\s*start=(?P<start>[\d.]+),\s*end=(?P<end>[\d.]+)"
        r"(?:,\s*valid_times=\[(?P<valid>[^\]]+)\])?\s*$"
    )
    sessions = []
    for line in text.splitlines():
        if not line.strip():
            continue
        match = pattern.match(line.strip())
        if match is None:
            msg = (
                f"sessions text line {line!r} does not read as '#n, name, start=, end='"
            )
            raise ValueError(msg)
        start, end = float(match["start"]), float(match["end"])
        if match["valid"]:
            pairs = [pair.split(",") for pair in match["valid"].split(";")]
            valid = np.array([[float(a), float(b)] for a, b in pairs])
        else:
            valid = np.array([[start, end]])
        sessions.append(_Session(match["name"], start, end, valid))
    return sessions


def _open_field(listed: list[_Session], session: str) -> _Session:
    """The one open-field session of a recording."""
    found = [s for s in listed if s.name.startswith("open_field")]
    if len(found) != 1:
        msg = (
            f"{session}: the sessions text lists {len(found)} open-field "
            f"sessions ({[s.name for s in found]}), not one"
        )
        raise ValueError(msg)
    return found[0]


def _named(names: list[str], suffix: str) -> str:
    """The one member name ending with ``suffix``."""
    (name,) = [name for name in names if name.endswith(suffix)]
    return name


def _rat(session: str) -> str:
    """``"R"`` for ``"rat_r_day1"``: the rat's letter, as the paper names it."""
    return session.split("_")[1].upper()


def _heading_conversion(offset: float, spread: float) -> Conversion:
    """How the azimuth became ``headings``, with its check against movement."""
    verified = abs(offset) <= MAX_HEADING_OFFSET and spread <= MAX_HEADING_SPREAD
    return Conversion(
        "headings: azimuth in [0, 2pi) rad, wrapped to [-pi, pi]; taken as "
        "allocentric in the positions' frame (0 = East, counter-clockwise)",
        "inferred",
        "no source read states the azimuth's zero or sense; minus the "
        f"direction of movement above {RUNNING:g} cm/s it has a circular mean "
        f"of {np.degrees(offset):+.1f} deg and a median spread of "
        f"{np.degrees(spread):.1f} deg"
        + (
            ", so it points along the path"
            if verified
            else ", so its frame is not verified (see Watch out)"
        ),
    )


def _heading_watch_out(offset: float, spread: float) -> tuple[str, ...]:
    """A watch-out when the azimuth does not follow the direction of movement."""
    if abs(offset) <= MAX_HEADING_OFFSET and spread <= MAX_HEADING_SPREAD:
        return ()
    return (
        "The heading frame is not verified by movement: azimuth minus the "
        f"direction of movement above {RUNNING:g} cm/s has a circular mean of "
        f"{np.degrees(offset):+.1f} deg and a median spread of "
        f"{np.degrees(spread):.1f} deg (a frame in the positions' axes gives "
        f"a mean within {np.degrees(MAX_HEADING_OFFSET):.0f} deg of 0 and a "
        f"spread under {np.degrees(MAX_HEADING_SPREAD):.0f} deg). The values "
        "are kept as stored, wrapped to [-pi, pi].",
    )


def _invalid_stretches(
    times: NDArray[np.float64], session: _Session
) -> tuple[str, ...]:
    """The stretches of the session its ``valid_times`` leave out, with samples."""
    valid = session.valid
    edges = np.concatenate([[session.start], valid.ravel(), [session.end]])
    gaps = edges.reshape(-1, 2)
    gaps = gaps[gaps[:, 1] > gaps[:, 0]]
    if not len(gaps):
        return ()
    counts = [int(np.sum((times > a) & (times < b))) for a, b in gaps]
    stretches = ", ".join(
        f"{a:g} to {b:g} s ({n} samples)"
        for (a, b), n in zip(gaps, counts, strict=True)
    )
    return (
        f"The sessions text's valid_times leave out {stretches} of "
        f"{session.name}, without saying why ({_PAPER} say the task was paused "
        "to untwist the tether when needed). Those samples and spikes are kept; "
        "they lie outside epochs['open_field'] and params['analysis_epochs'].",
    )


def _other_sessions(listed: list[_Session], session: _Session) -> str:
    """The recording's other sessions, which this session does not load."""
    others = ", ".join(
        f"{s.name} ({s.start:g} to {s.end:g} s)" for s in listed if s != session
    )
    return (
        f"The file also holds {others}, on other apparatus; only "
        f"{session.name} is loaded, so spikes and samples outside "
        f"{session.start:g} to {session.end:g} s are not in this recording."
    )


def _repeated_spikes(trains: list[NDArray[np.float64]]) -> tuple[str, ...]:
    """A watch-out for spikes at exactly the time of the spike before them."""
    repeats = [int(np.sum(np.diff(train) == 0)) for train in trains]
    total = sum(repeats)
    if total == 0:
        return ()
    n_units = sum(1 for n in repeats if n)
    return (
        f"{total} {'spike repeats' if total == 1 else 'spikes repeat'} the time "
        f"of the spike before {'it' if total == 1 else 'them'} in the same unit "
        f"({n_units} {'unit' if n_units == 1 else 'units'}); they are kept, so "
        "each counts once in a rate.",
    )


def _declared_square(
    positions: NDArray[np.float64], session: str
) -> tuple[Environment, Conversion]:
    """The stated square on the samples' minimum rotated rectangle, and how it was placed."""
    from shapely import affinity
    from shapely.geometry import box

    from neurospatial import Environment

    centre, angle, lengths = fitted_rectangle(positions)
    half = SIDE / 2
    square = affinity.translate(
        affinity.rotate(box(-half, -half, half, half), angle, use_radians=True),
        *centre,
    )
    env = Environment.from_polygon(
        square, bin_size=BIN_SIZE, units="cm", name=f"gardner_2022 {session}"
    )
    residuals = ", ".join(f"{side:.4g} cm, {side / SIDE - 1:+.1%}" for side in lengths)
    placement = Conversion(
        f"environment: the stated {SIDE:g} x {SIDE:g} cm square centred at "
        f"({centre[0]:.3g}, {centre[1]:.3g}) cm, rotated "
        f"{np.degrees(angle):+.2f} deg from the file's axes; the samples' own "
        f"sides against the stated side: {residuals}",
        "inferred",
        "the files state no origin or rotation; the centre and rotation are "
        "those of the minimum rotated rectangle around the tracked samples, "
        f"less the {4 * PLACEMENT_TRIM:.1%} farthest from their median",
    )
    return env, placement
