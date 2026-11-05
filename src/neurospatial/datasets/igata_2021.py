"""Igata, Ikegaya & Sasaki (2021): rats learn a new route and CA1 replays it (Mendeley 4xk5w69yr5).

What was recorded: tetrode units and LFP in dorsal CA1 while rats ran from a
start area through a 1 m open field to a goal area and back along a walled
return path. The field is a 5 x 5 lattice of 20 cm squares with a short pole at
each grid point, so a rat chooses its own route between the poles. A trial
pays a small reward at a checkpoint lattice and a large one in the goal area.
On a recording day the checkpoint is first C1 (the prelearning phase, 11 to 14
trials), then moves to C2, and the rat learns the shorter route through it
(learning and postlearning phases). Position is three red LEDs on the electrode
assembly, tracked at 24.5 Hz by a ceiling camera.

Where it lives: Mendeley Data 10.17632/4xk5w69yr5, version 2 (CC-BY-4.0; the
paper cites version 1), one uncompressed ``.npz`` per trial and per return run.

- ``Learning_data.zip`` holds the paper's five rats' learning days: rat 1
  180912 (114 units), rat 2 180824 (72), rat 3 180208 (68), rat 4 180303 (79)
  and rat 5 180107 (22), the unit counts the SI states. Its members are read by
  byte range. The readme does not mention this zip: its folders ``rat1`` to
  ``rat5`` and those counts identify it.
- The 1,510 files beside it hold other days: 180913, the only one with many
  units (111), is the day after rat 1's, and the SI's day-2 condition has the
  same design; the rest have up to 7 units or none, and hold the closed-loop
  stimulation of the paper's Fig. 5 where a ``trial_stim`` field says so. The
  archive does not say which rat or which stimulation group each of those days
  is. Seventeen of 180209's files are published twice with one checksum; they
  are read once.

Sources read: the archive's ``readme.txt`` and ``virtual_maze_field.py`` (read
as data, never run), the paper (PNAS 118:e2011266118), its SI Appendix (Europe
PMC, PMC7817193), and ``ripple_detection``'s note on the paper at the pinned
commit.

What the loader converts:

- sessions: one per recording, ``<yymmdd>_detour<G|S><nn>``, the files sharing
  that prefix stitched on their one clock. ``S`` and ``G`` name the checkpoint
  lattice, inferred: the arrive-check-point flag rises in lattice S on every
  trial of the ``S`` sessions and lattice G on every trial of the ``G``
  sessions, which are C1 and C2 by the SI's lattice definitions. So an ``S``
  session is the prelearning phase and the same day's ``G01`` is what follows.
  Each starts its own clock, so they are separate sessions.
- times: ``log`` column 0, ms to s (the readme). A return run begins on the frame
  its trial ended on, the two rows being identical; one is dropped so times
  increase. Files follow one another by one frame, or by a longer gap where
  files are missing, and the gap stays a gap.
- positions: ``log`` columns 1 and 2, mm to cm (the readme). The frame is the
  file's, with y growing away from the paper's top edge, so a plot with y up is
  the paper's figure flipped top to bottom.
- spike times: ``spike``, ms from the file's first frame (inferred; see the
  conversion) plus that frame's time. ``spike`` is a pickle written by Python 2.
  :func:`_read_spikes` lists its imports first, and allows only those of
  :class:`~neurospatial.datasets._pickled.SpikeUnpickler` and the two
  latin-1 helpers such a pickle needs. It matches ``spike_hist`` exactly, while
  keeping the 30 kHz times that ``spike_hist`` bins to 1 ms.
- epochs and events: runs of the ``log`` flags the readme names.
- ``trials``: one row per trial file, with the file's ``locus_string_list``, the
  lattices the rat passed (``virtual_maze_field.py``).
- ``params['learning_point']``: the paper's rule on those strings, on the five
  learning days only, where the checkpoint moved at the session's start.

The LFP is in the same files, so it is on disk once a session is loaded. It is
read on request: ``rec.lfp.fetch()`` gives the file's representative tetrode,
and ``rec.lfp.fetch(tuple(rec.lfp.channels))`` every tetrode and the reference.
"""

from __future__ import annotations

import functools
import io
import os
import pickle
import re
import zipfile
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

from neurospatial.datasets._catalog import SessionEntry
from neurospatial.datasets._lfp import LFPFile, LFPSource
from neurospatial.datasets._pickled import SpikeUnpickler, pickle_imports
from neurospatial.datasets._recording import Conversion, Recording

__all__ = [
    "CANDIDATE_BASELINE",
    "EXPECTED_WATCH_OUTS",
    "LEARNING_RESULT",
    "REQUIRES",
    "WATCH_OUTS",
    "WORKED_RESULT",
    "learning_point",
    "load_session",
    "path_track",
]

_PAPER = "Igata et al. 2021"
_SI = f"{_PAPER}, SI Appendix"

#: The paper's finding this dataset reproduces, or why there is none yet.
WORKED_RESULT: dict[str, Any] = {
    "status": "not possible yet",
    "level": None,
    "finding": (
        "trajectory events among the paper's synchronous events; no such share "
        "is stated in the text, only a figure, so this was never a number to match"
    ),
    "quote": (
        '"Trajectory sequences covering more than 4 time bins (60 ms) and a '
        "cumulative distance of more than 20 cm were included in the further "
        'analyses." and "Trajectory events were defined from the trajectory '
        'sequences where p < 0.05." (SI Appendix, "Trajectory events")'
    ),
    "source": f"{_SI}, read from Europe PMC (PMC7817193)",
    "paper": "no share stated in the text; Fig. 4G and Fig. S14B plot it",
    "ours": None,
    "criterion": None,
    "parameters": None,
    "unmatched": (
        "the Gaussian-mixture split after the candidates: ripple_detection's "
        "igata_2021 stops at the candidates, and the SI's split is under-specified",
        "the paper's maps and MAP are per path and per phase; ours are one map "
        "of one session, one MAP over the joined paths",
    ),
    "attempted": {
        "value": 0.004,
        "criterion": (
            "the candidates stand out from the no-ripple baseline fixed in "
            "writing before any detection (CANDIDATE_BASELINE), and the judged "
            "share tops the largest of three null draws; the first was not met"
        ),
        "parameters": (
            "default session; candidates from ripple_detection's igata_2021, "
            "wholly inside stops (speed < 5 cm/s): 1,688; every unit, session "
            "speed. Maps: compute_spatial_rates on path_track(), 2 cm bins, "
            "speed > 5 cm/s, 5 cm bandwidth. Decoded in 20 ms bins stepped by 10 "
            "ms, every row kept to the event's end, judged by IGATA_2021 unmodified, "
            "1000 surrogates: 6 of 1,688"
        ),
        "differs": (
            "this is not a replay rate. The 6 events are the shortest candidates "
            "(median 68 ms against 130 ms) and none of the 263 that survive the "
            "baseline is among them: the surrogates also hold a trajectory, so "
            "p rises with duration and no survivor reaches 0.05 (smallest 0.08)"
        ),
    },
    "planted_recall": {
        "judged": 2,
        "planted": 300,
        "reading": (
            "replay planted into 150 survivors and 150 other immobile events, "
            "keeping every spike's time and drawing its cell from these rate maps "
            "along a real running path played 15 times faster, is judged replay "
            "0.7% of the time. So this test passes almost no known replay here, "
            "and 0 of 596 is not evidence that the events hold none. This is a "
            "limit of what the number can say: planted events come from the model "
            "the decoder assumes, and compression 15 is a choice"
        ),
    },
    "why": (
        "the replay test cannot be read here. On the default session the "
        "candidates do not stand out by count, since shifted spikes give more "
        "of them; two thirds of them do carry ripple power, and the 263 that "
        "pass both tests look like sharp-wave ripples, yet IGATA_2021 judges "
        "none of those. Rats 2 and 3 pass the fixed rule (11 judged against 6, "
        "13 against 12) and rat 4 fails it; across the four rats' 596 survivors, "
        "judged on their own, none is judged. So no number here is presented as "
        "replay"
    ),
}

#: The candidate detector against a no-ripple baseline, fixed in writing before
#: it was run on any real session. ``igata_2021`` in ``ripple_detection`` is the
#: candidate stage only: it has no Gaussian mixture and no speed rule on events.
CANDIDATE_BASELINE: dict[str, Any] = {
    "what": (
        "how ripple_detection's igata_2021 candidates compare with the same "
        "detector on spikes shifted per unit, and with random stopped windows"
    ),
    "seed": 2026,
    "n_draws": 20,
    "bandwidth": 5.0,
    "rule": (
        "Immobile candidates lie wholly inside a run of speed < 5 cm/s. They "
        "stand out if their count tops the 95th percentile of the 20 null "
        "counts (each unit circularly shifted by its own offset, at least 10 s "
        "from either end) and the share of them above the 95th percentile of "
        "100 random stopped windows of their duration, in mean overall ripple "
        "power, tops 5% at p < 0.01. An event survives if it passes that power "
        "test and its peak z-scored rate tops the 95th percentile of all null "
        "immobile candidates' peaks"
    ),
    "stand_out": False,  # the default session; each session's is in "sessions"
    "why_the_count_test_fails_here": (
        "shifting moves running spikes into stops: the stopped population rate "
        "is 0.355 Hz per unit (SD 0.408) in the file and 0.444 (SD 0.338) in "
        "draw 0, so the threshold falls from 1.171 to 1.119 Hz. The null is "
        "thus generous to chance events, and the count test is a hard one"
    ),
    "mixture_reading": (
        "one reading of the SI's under-specified split, a diagnostic only: peak "
        "z-scored rate and peak overall ripple power, a 1-D two-component "
        "mixture on each, keeping the higher component of either. It keeps "
        "1,120 of the default session's 1,688 (0.52 Hz of stop) and all 263 "
        "survivors"
    ),
    "sessions": {
        "180912_detourG01": {
            "candidates": 4222,
            "immobile": 1688,
            "stop_minutes": 36.13,
            "null_immobile": (
                1922,
                1896,
                1904,
                1922,
                1906,
                1957,
                1921,
                1943,
                1940,
                1936,
                1904,
                1987,
                1952,
                1965,
                1964,
                1935,
                1966,
                1966,
                1983,
                1986,
            ),
            "ripple_passed": 1116,
            "survivors": 263,
            "stand_out": False,
            "judged": 6,
            "null_judged": ((4, 1922), (4, 1896), (3, 1904)),
            "survivors_judged": 0,
            "survivors_with_segment": 214,
            "first_100_real": (1, 56),
            "first_100_null_0": (0, 37),
        },
        "180912_detourS01": {
            "candidates": 594,
            "immobile": 250,
            "stop_minutes": 5.91,
            "null_p95": 252.5,
            "ripple_passed": 161,
            "survivors": 35,
            "stand_out": False,
        },
        "180824_detourG01": {
            "candidates": 2005,
            "immobile": 1041,
            "stop_minutes": 24.25,
            "null_p95": 1010.45,
            "ripple_passed": 672,
            "survivors": 93,
            "stand_out": True,
            "judged": 11,
            "null_judged": ((1, 978), (2, 958), (6, 1038)),
            "survivors_judged": 0,
            "survivors_with_segment": 50,
        },
        "180208_detourG01": {
            "candidates": 2790,
            "immobile": 1743,
            "stop_minutes": 27.46,
            "null_p95": 1695.25,
            "ripple_passed": 328,
            "survivors": 18,
            "stand_out": True,
            "judged": 13,
            "null_judged": ((12, 1663), (8, 1695), (5, 1643)),
            "survivors_judged": 0,
            "survivors_with_segment": 14,
        },
        "180303_detourG01": {
            "candidates": 3390,
            "immobile": 2090,
            "stop_minutes": 36.26,
            "null_p95": 2041.65,
            "ripple_passed": 1172,
            "survivors": 222,
            "stand_out": True,
            "judged": 6,
            "null_judged": ((11, 1984), (2, 1931), (4, 1990)),
            "survivors_judged": 0,
            "survivors_with_segment": 129,
        },
    },
}

#: A published number the files reproduce: behaviour, not replay.
LEARNING_RESULT: dict[str, Any] = {
    "quote": (
        '"The rats required 26 \u00b1 9 trials (\u223c30 min) to progress from '
        'reward replacement to the learning point (n = 5 rats)." (Results)'
    ),
    "points": (46, 9, 7, 49, 18),
    "paper": (26, 9),
    "ours": (25.8, 9.1),
    "note": (
        "data are mean and SEM, as the SI gives them. The SI does not align "
        "its 5-trial window; a centred one is used, and a trailing one gives "
        "27.4 and 8.8, which rounds to 27 and 9"
    ),
}

#: Modules of the ``datasets`` extra the loader imports.
REQUIRES = ("fsspec",)

#: Stated by the SI Appendix, in cm. The grid and the areas it names.
LATTICE = 20.0
FIELD = 100.0
AREA_WIDTH = 20.0
START_AREA = 40.0
GOAL_AREA = 30.0
#: Grid for ``rec.env``; our choice, wide enough for rate maps on 80 minutes.
BIN_SIZE = 5.0
#: The paper's linear bin width, and its smoothing of rate maps.
TRACK_BIN_SIZE = 2.0
#: How far from a path line a sample still counts, cm: one lattice either side.
TRACK_WIDTH = 40.0
#: The paper's speed threshold for stops and place fields, cm/s.
MIN_SPEED = 5.0
#: Time bin for analyses of the session, s: the paper's 12 video frames.
DT = 0.49
#: The paper's LFP rate, Hz.
LFP_HZ = 2000.0
#: Files with these names are the five rats' learning days.
LEARNING_RATS = {"180912": 1, "180824": 2, "180208": 3, "180303": 4, "180107": 5}
#: A gap between files beyond this many seconds is a file missing from the
#: archive, not the one frame between a return run and the next trial.
MISSING_FILE_GAP = 1.0
#: The paper's moving-average window on the optimized trajectory, trials.
LEARNING_WINDOW = 5
#: An optimized trajectory has fewer field lattices than this (SI).
OPTIMIZED_LENGTH = 12

#: ``log`` columns, as the readme numbers them, where it agrees with the file.
_TIME, _X, _Y, _LED, _TRIAL, _POKE, _DELAY = 0, 1, 2, 3, 4, 5, 6
_CHECKPOINT, _TIMEOUT, _REWARD_PORT, _READY = 8, 9, 10, 12
#: The paper's lattices in the file's letters: S is U, G is E, C1 is S, C2 is G.
_S, _C1, _C2, _G = "U", "S", "G", "E"
_FIELD_LETTERS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXY")
_FILE = re.compile(r"(?P<session>.+)_(?P<kind>trial|return)(?P<number>\d+)\.npz")
_README = "the archive's readme.txt"
_FIELDS = "virtual_maze_field.py"

#: Findings computed from the data (by the checks or this loader) that each
#: verified session must report, as patterns searched for in its watch_outs.
EXPECTED_WATCH_OUTS: dict[str, tuple[str, ...]] = {
    "180912_detourG01": (
        r"lie outside the declared environment",
        r"flagged the LED undetected",
        r"longer than 1 s between files",
    ),
}

#: Pitfalls of each session checked by hand, with their evidence.
WATCH_OUTS: dict[str, tuple[str, ...]] = {
    "180912_detourG01": (
        "Return run 44 is not in the archive, so the 197 s after trial 44 hold no "
        "spikes or LFP, and the last trial, 63, has no return file: the session "
        "has 63 trial files and 61 return files.",
    ),
}

_GENERAL = (
    "The readme numbers 14 log columns, and rat 5's files have those 14. The "
    "other rats' files have 16: columns 0 to 12 mean the same (the checkpoint "
    "flag rises in the checkpoint lattice in both), column 15 is always 1, as "
    "the readme's sync column 13 is in rat 5, and columns 13 and 14 are not "
    "documented. Neither is read.",
    "Rest-box recordings ('tens of minutes before and after', the SI) are not "
    "in the archive, so there is no sleep or rest epoch; replay here is awake.",
    "The SI's apparatus has automatic doors between the areas, and no file says "
    "when each was open. rec.env leaves them open, so a path through it may "
    "cross a door the rat found closed.",
    "LFP values step by 0.25, and no source read gives their unit, so "
    "rec.lfp.unit is 'arbitrary'.",
    "ripple_detection's igata_2021 finds candidates only: the SI's Gaussian-"
    "mixture split after them is under-specified, and has no code to follow. "
    "Candidates are not synchronous events: on the four rats checked, spikes "
    "shifted per unit give about as many, and IGATA_2021 judges none of the "
    "events that pass a no-ripple baseline (CANDIDATE_BASELINE).",
)


class _File(NamedTuple):
    """What one ``.npz`` holds, on the session's clock."""

    kind: str
    number: int
    log: NDArray[np.float64]
    spikes: list[NDArray[np.float64]]
    string: str
    stimulation: NDArray[np.float64]


def _latin1(text: str, encoding: str = "latin1") -> bytes:
    """``_codecs.encode`` as a Python 2 NumPy pickle calls it, and no other way."""
    if encoding != "latin1" or type(text) is not str:
        msg = f"refusing _codecs.encode({text!r:.40}, {encoding!r})"
        raise pickle.UnpicklingError(msg)
    return text.encode("latin1")


def _bytes(value: Any = b"") -> bytes:
    """``bytes`` of bytes or text, as such a pickle calls it, and no other way."""
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("latin1")
    msg = f"refusing bytes({value!r:.40})"
    raise pickle.UnpicklingError(msg)


class _Python2SpikeUnpickler(SpikeUnpickler):
    """:class:`SpikeUnpickler`, plus the two helpers Python 2 pickles of arrays use.

    Python 2 stored raw bytes as text, so reading them back in Python 3 calls
    ``_codecs.encode`` and ``bytes``; each resolves to a function that only
    turns text into bytes.
    """

    ALLOWED = {  # noqa: RUF012
        **SpikeUnpickler.ALLOWED,
        ("_codecs", "encode"): _latin1,
        ("__builtin__", "bytes"): _bytes,
    }


def _read_spikes(path: Path) -> list[NDArray[np.float64]]:
    """Each unit's ``spike`` array of one file, ms from the file's first frame.

    Parameters
    ----------
    path : Path
        One ``.npz``.

    Returns
    -------
    list of ndarray, each shape (n_spikes,)
        One array per unit; empty for a file without units, whose ``spike`` is a
        plain empty array rather than a pickle.

    Raises
    ------
    pickle.UnpicklingError
        If the pickle imports anything outside the allow-list.
    """
    from numpy.lib import format as npy

    with np.load(path, allow_pickle=False) as data:
        if data["spike_hist"].shape[0] == 0:
            return []
    with zipfile.ZipFile(path) as archive, archive.open("spike.npy") as stream:
        version = npy.read_magic(stream)  # type: ignore[no-untyped-call]
        read_header = (
            npy.read_array_header_1_0
            if version == (1, 0)
            else npy.read_array_header_2_0
        )
        read_header(stream)  # type: ignore[no-untyped-call]
        blob = stream.read()
    allowed = {".".join(key) for key in _Python2SpikeUnpickler.ALLOWED}
    if unexpected := pickle_imports(blob) - allowed:
        msg = f"refusing {path.name}: its pickle imports {sorted(unexpected)}"
        raise pickle.UnpicklingError(msg)
    trains = _Python2SpikeUnpickler(io.BytesIO(blob), encoding="latin1").load()
    return [np.asarray(train, dtype=np.float64).ravel() for train in trains]


def _read_file(path: Path) -> _File:
    """One file's arrays; times stay in ms until :func:`_stitch` joins them."""
    match = _FILE.fullmatch(path.name)
    if match is None:
        msg = f"{path.name}: not a <session>_<trial|return><n>.npz file"
        raise ValueError(msg)
    with np.load(path, allow_pickle=False) as data:
        log = np.asarray(data["log"], dtype=np.float64)
        string = "".join(str(letter) for letter in data["locus_string_list"])
        stimulation = (
            np.ravel(data["trial_stim"]).astype(np.float64)
            if "trial_stim" in data.files
            else np.zeros(0)
        )
    return _File(
        match["kind"],
        int(match["number"]),
        log,
        _read_spikes(path),
        string,
        stimulation,
    )


class _Stitched(NamedTuple):
    """One session's files on one clock, in seconds and cm."""

    times: NDArray[np.float64]
    log: NDArray[np.float64]
    spikes: tuple[NDArray[np.float64], ...]
    stimulation: NDArray[np.float64]
    n_shared_frames: int
    gaps: NDArray[np.float64]


def _stitch(files: list[_File], session: str) -> _Stitched:
    """Join files that follow one another into one clock, dropping shared frames.

    Parameters
    ----------
    files : list of _File
        In the order of their first frames.
    session : str

    Returns
    -------
    _Stitched

    Raises
    ------
    ValueError
        If two files overlap, if a frame shared by two files differs, or if the
        files hold different numbers of units.
    """
    n_units = {len(file.spikes) for file in files}
    if len(n_units) != 1:
        msg = f"{session}: files hold {sorted(n_units)} units, so ids would not line up"
        raise ValueError(msg)
    logs = []
    shared = 0
    gaps = []
    for previous, file in zip([None, *files], files, strict=False):
        log = file.log
        if previous is not None:
            step = log[0, _TIME] - previous.log[-1, _TIME]
            if step == 0.0:
                if not np.array_equal(log[0], previous.log[-1]):
                    msg = (
                        f"{session}: {file.kind} {file.number} starts on the frame "
                        "its predecessor ended on, with different values"
                    )
                    raise ValueError(msg)
                log = log[1:]
                shared += 1
            elif step < 0.0:
                msg = (
                    f"{session}: {file.kind} {file.number} starts {-step:g} ms "
                    "before its predecessor ends"
                )
                raise ValueError(msg)
            else:
                gaps.append(step / 1e3)
        logs.append(log)
    joined = np.concatenate(logs)
    spikes = tuple(
        np.sort(
            np.concatenate(
                [(file.spikes[unit] + file.log[0, _TIME]) / 1e3 for file in files]
            )
        )
        for unit in range(n_units.pop())
    )
    stimulation = np.sort(
        np.concatenate(
            [(file.stimulation + file.log[0, _TIME]) / 1e3 for file in files]
        )
    )
    return _Stitched(
        joined[:, _TIME] / 1e3,
        joined,
        spikes,
        stimulation,
        shared,
        np.array(gaps),
    )


def _flag_rows(
    times: NDArray[np.float64],
    flag: NDArray[np.bool_],
    *,
    max_gap: float = MISSING_FILE_GAP,
) -> NDArray[np.float64]:
    """[first frame on, next frame) of each run of a per-frame flag, seconds.

    A run ends at a frame the flag is off, or at a gap between frames longer
    than ``max_gap``, whose last frame then closes it.

    Parameters
    ----------
    times : ndarray, shape (n_time,)
    flag : ndarray of bool, shape (n_time,)
    max_gap : float
        Seconds.

    Returns
    -------
    ndarray, shape (n_runs, 2)

    Examples
    --------
    >>> import numpy as np
    >>> flag = np.array([False, True, True, False, True, True])
    >>> _flag_rows(np.arange(6.0), flag).tolist()
    [[1.0, 3.0], [4.0, 5.0]]
    """
    broken = np.diff(times) > max_gap
    starts = flag & ~np.r_[False, flag[:-1] & ~broken]
    stops = flag & ~np.r_[flag[1:] & ~broken, False]
    first = np.flatnonzero(starts)
    last = np.flatnonzero(stops)
    # A run ends at the next frame, unless a gap or the session's end is there.
    follows = last + 1 < times.size
    follows[follows] = ~broken[last[follows]]
    ends = times[np.minimum(last + 1, times.size - 1)]
    ends = np.where(follows, ends, times[last])
    rows = np.column_stack([times[first], ends])
    return rows[rows[:, 1] > rows[:, 0]]


def _optimized(string: str) -> bool:
    """Whether a trial took the paper's optimized path, S-C2-G, under 12 lattices.

    Examples
    --------
    >>> _optimized("lUVQLGHIDEa")
    True
    >>> _optimized("lUVWXSTOJEa")  # by C1
    False
    >>> _optimized("lUVQLGFKFABCDEa")  # by C2, the long way round: 13 lattices
    False
    """
    field = "".join(letter for letter in string if letter in _FIELD_LETTERS)
    if len(field) >= OPTIMIZED_LENGTH or _C1 in field:
        return False
    if _S not in field or _C2 not in field or _G not in field:
        return False
    return field.index(_S) < field.index(_C2) < field.rindex(_G)


def learning_point(strings: list[str]) -> int | None:
    """Trials from the reward's move to the paper's learning point.

    The SI's rule: a moving average over 5 trials of whether each trial took
    the optimized path; the learning point is one trial before that average
    first exceeds 50%, or the next trial if that one was not optimized. The SI
    does not say how the window is aligned; it is centred here, shortened at
    the ends.

    Parameters
    ----------
    strings : list of str
        Each trial's lattice string, in order, from the first trial after the
        reward moved.

    Returns
    -------
    int or None
        The learning point's zero-based index, which is the number of trials
        before it; None if the average never exceeds 50%.

    Examples
    --------
    >>> learning_point(["lUVWXSTOJEa"] * 6 + ["lUVQLGHIDEa"] * 5)
    6
    """
    took = np.array([_optimized(string) for string in strings], dtype=np.float64)
    half = LEARNING_WINDOW // 2
    average = np.array(
        [took[max(k - half, 0) : k + half + 1].mean() for k in range(took.size)]
    )
    over = np.flatnonzero(average > 0.5)
    if not over.size:
        return None
    point = int(over[0]) - 1
    return point + 1 if not took[point] else point


def _rectangle(left: float, bottom: float, right: float, top: float) -> Any:
    """A rectangle's polygon, cm."""
    import shapely

    return shapely.box(left, bottom, right, top)


def _lattice_square(letter: str) -> Any:
    """The file's lattice ``letter`` (``A`` to ``Y``) as a polygon, cm."""
    index = ord(letter) - ord("A")
    column, row = index % 5, index // 5
    return _rectangle(
        column * LATTICE,
        row * LATTICE,
        (column + 1) * LATTICE,
        (row + 1) * LATTICE,
    )


def _maze(session: str) -> Any:
    """The apparatus: the field, start and goal areas and the L-shaped return path.

    The SI gives the sizes; ``virtual_maze_field.py`` where they sit. Its
    polygons cover the plane around the areas (a 30 cm wide goal box), so the
    widths here are the SI's. The doors are open, and the field's two inner
    sides are a wall.

    Returns
    -------
    Environment
    """
    from shapely.geometry import LineString
    from shapely.ops import unary_union

    from neurospatial import Environment
    from neurospatial.arena import OpenArena
    from neurospatial.regions import Region

    field = _rectangle(0.0, 0.0, FIELD, FIELD)
    start = _rectangle(-AREA_WIDTH, FIELD - START_AREA, 0.0, FIELD)
    goal = _rectangle(FIELD, 0.0, FIELD + AREA_WIDTH, GOAL_AREA)
    alley = unary_union(
        [
            _rectangle(FIELD, GOAL_AREA, FIELD + AREA_WIDTH, FIELD + AREA_WIDTH),
            _rectangle(-AREA_WIDTH, FIELD, FIELD, FIELD + AREA_WIDTH),
        ]
    )
    outline = unary_union([field, start, goal, alley]).simplify(0.0)
    arena = OpenArena(
        outline,
        walls={
            "field_to_return_path": LineString(
                [(0.0, FIELD), (FIELD, FIELD), (FIELD, GOAL_AREA)]
            )
        },
    )
    shapes = {
        "start_area": start,
        "goal_area": goal,
        "return_path": alley,
        "open_field": field,
        "S": _lattice_square(_S),
        "C1": _lattice_square(_C1),
        "C2": _lattice_square(_C2),
        "G": _lattice_square(_G),
    }
    return Environment.from_polygon(
        outline,
        bin_size=BIN_SIZE,
        units="cm",
        name=f"igata_2021 {session}",
        arena=arena,
        regions={
            name: Region(name=name, kind="polygon", data=shape)
            for name, shape in shapes.items()
        },
    )


def path_track(bin_size: float = TRACK_BIN_SIZE) -> Any:
    """The paper's paths as a 1-D environment, for replay.

    The paper projects each path's samples onto "a straight line connecting
    the centers of the two lattices at both ends" (SI Fig. S1I) and decodes
    along those lines. Here the lines S-C1, C1-G, G-C2 and S-C2 are the edges
    of one graph, with lines through the start area, the goal area and the
    return path to close it: an ordering of the paper's paths on one axis, not
    its per-path decoding. Samples more than 20 cm from every line, on routes
    far from the paths, fall in no bin.

    Parameters
    ----------
    bin_size : float, default 2.0
        Bin length, cm; the paper's.

    Returns
    -------
    Environment
        1-D, in cm. The nodes are named ``start``, ``S``, ``C1``, ``G``, ``C2``,
        ``goal``, ``corner`` and ``far_corner``.

    Examples
    --------
    >>> track = path_track()
    >>> track.units
    'cm'
    """
    import networkx as nx

    from neurospatial import Environment

    def centre(letter: str) -> tuple[float, float]:
        point = _lattice_square(letter).centroid
        return float(point.x), float(point.y)

    half = AREA_WIDTH / 2
    nodes = {
        "start": (-half, FIELD - START_AREA / 2),
        "S": centre(_S),
        "C1": centre(_C1),
        "G": centre(_G),
        "C2": centre(_C2),
        "goal": (FIELD + half, GOAL_AREA / 2),
        "corner": (FIELD + half, FIELD + half),
        "far_corner": (-half, FIELD + half),
    }
    order = [
        ("start", "S"),
        ("S", "C1"),
        ("C1", "G"),
        ("G", "C2"),
        ("C2", "S"),
        ("G", "goal"),
        ("goal", "corner"),
        ("corner", "far_corner"),
        ("far_corner", "start"),
    ]
    graph = nx.Graph()
    for name, position in nodes.items():
        graph.add_node(name, pos=position)
    for first, second in order:
        distance = float(np.hypot(*np.subtract(nodes[first], nodes[second])))
        graph.add_edge(first, second, distance=distance)
    return Environment.from_graph(
        graph,
        order,
        bin_size=bin_size,
        units="cm",
        name="igata_2021 paths",
        track_width=TRACK_WIDTH,
    )


def read_lfp(
    sources: tuple[str | Path, ...],
    channels: tuple[str, ...],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The LFP of a session's files, on the session's clock.

    Parameters
    ----------
    sources : tuple of path
        The session's files, in any order; they are put in time order.
    channels : tuple of str
        Names from ``tet_names``, or ``"ref"`` for the reference, in the order
        of the columns.

    Returns
    -------
    samples : ndarray, shape (n_time, n_channels)
        As stored.
    times : ndarray, shape (n_time,)
        Seconds. A file's ``n`` samples are spread evenly from its first frame
        up to, but not reaching, its last, since a file's length is within 1 ms
        of its span at 2 kHz but a step past the next file's first frame would
        break the clock's order.

    Raises
    ------
    ValueError
        If a file does not hold a channel.
    """
    columns, clocks = [], []
    for path in _in_time_order(sources):
        with np.load(path, allow_pickle=False) as data:
            names = [str(name) for name in data["tet_names"]]
            frames = data["log"][:, _TIME]
            stored = data["lfps"]
            whole = stored.shape[0] == len(names)
            rows = []
            for channel in channels:
                if channel == "ref":
                    rows.append(data["ref"][0])
                elif whole and channel in names:
                    rows.append(stored[names.index(channel)])
                elif channel == str(data["lfpch"]):
                    rows.append(data["lfp"])
                else:
                    msg = f"{Path(path).name} holds no channel {channel!r}"
                    raise ValueError(msg)
        columns.append(np.stack(rows, axis=1).astype(np.float64))
        clocks.append(
            np.linspace(frames[0], frames[-1], columns[-1].shape[0], endpoint=False)
            / 1e3
        )
    return np.concatenate(columns), np.concatenate(clocks)


def _in_time_order(sources: Any) -> list[Path]:
    """``sources`` in the order of their first frames."""

    def first_frame(path: Path) -> float:
        with np.load(path, allow_pickle=False) as data:
            return float(data["log"][0, _TIME])

    return sorted((Path(source) for source in sources), key=first_frame)


def _representative(folder_file: Path) -> tuple[list[str], str]:
    """The tetrodes whose LFP a file holds, and the one ``lfp`` copies.

    ``lfpch`` can drop the name's suffix (``tet012`` for ``tet012_1``), so the
    representative is the row of ``lfps`` equal to ``lfp``. Where ``lfps`` has
    fewer rows than ``tet_names`` has names, nothing says which tetrode a row
    is, so only ``lfp``, which ``lfpch`` names, is offered.
    """
    with np.load(folder_file, allow_pickle=False) as data:
        names = [str(name) for name in data["tet_names"]]
        if data["lfps"].shape[0] != len(names):
            return [str(data["lfpch"])], str(data["lfpch"])
        equal = [
            name
            for name, row in zip(names, data["lfps"], strict=True)
            if np.array_equal(row, data["lfp"])
        ]
    if len(equal) != 1:
        msg = f"{folder_file.name}: 'lfp' equals {len(equal)} rows of 'lfps', not one"
        raise ValueError(msg)
    return names, equal[0]


def _lfp_source(
    entry: SessionEntry, folder: Path, names: list[str], representative: str
) -> LFPSource:
    """The session's LFP: every tetrode and the reference, in the session's files."""
    files = tuple(
        LFPFile(
            path=f"{entry.session}/{Path(member.path).name}",
            sha256=member.sha256,
            size_bytes=member.size_bytes,
            url=member.url or entry.url,
            member=None if member.url else member.path,
        )
        for member in entry.members
    )
    channels = {
        name: f"tetrode {name}, dorsal CA1"
        + (" (the file's representative electrode)" if name == representative else "")
        for name in names
    }
    channels["ref"] = "the file's reference LFP"
    return LFPSource(
        dataset=entry.info.dataset,
        files=files,
        sampling_rate=LFP_HZ,
        channels=channels,
        unit="arbitrary",
        unit_basis=Conversion(
            "LFP: as stored; no scale is applied",
            "inferred",
            f"no source read gives a unit; values step by 0.25 and {_README} "
            "calls them only LFP",
        ),
        times_basis=Conversion(
            "LFP times: each file's samples spread evenly over its span, from "
            "its first frame, at the 2 kHz of the SI and the readme",
            "inferred",
            "each file's sample count over 2 kHz equals the span of its frames "
            "within 1 ms, on every file of the five learning days",
        ),
        reader=functools.partial(read_lfp),
        ripple_channels=(representative,),
        cache=folder.parents[1],
    )


def _no_units(session: str) -> tuple[str, ...]:
    return (
        f"{session} has no sorted units: its files hold behavior and LFP only, "
        "so nothing here can be decoded or judged.",
    )


def _stimulation_watch_out(n_pulses: int, times: NDArray[np.float64]) -> str:
    return (
        f"{n_pulses} electrical pulses to the ventral hippocampal commissure in "
        f"{float(times[-1] - times[0]) / 60:.0f} min are in events['stimulation'] "
        "(the files' trial_stim, ms from each file's first frame): the SI's "
        "closed-loop experiment. The archive does not say whether this animal's "
        "pulses followed each ripple at once, were delayed 250 ms, or were the "
        "nonlearning control, and they mark the LFP."
    )


def _led_watch_out(flag: NDArray[np.float64], log: NDArray[np.float64]) -> str | None:
    undetected = flag == 0.0
    if not undetected[1:].any():
        return None
    moved = np.linalg.norm(np.diff(log[:, [_X, _Y]], axis=0), axis=1)
    return (
        f"The log flagged the LED undetected on {undetected.mean():.1%} of frames "
        "(column 3), yet positions keep moving on them "
        f"({np.mean(moved[undetected[1:]] > 0):.1%} of those frames step), so "
        "the flag does not mark held positions, and no source read says what "
        "the tracker reports there. They are kept as stored."
    )


def _gap_watch_out(gaps: NDArray[np.float64]) -> str | None:
    long = gaps[gaps > MISSING_FILE_GAP]
    if not long.size:
        return None
    return (
        f"{long.size} {'gap' if long.size == 1 else 'gaps'} longer than 1 s "
        f"between files, up to {long.max():.1f} s: trial or return files are "
        "not in the archive there, so neither spikes nor LFP exist for that "
        "time, and the gaps stay gaps."
    )


def load_session(
    entry: SessionEntry, path_or_stream: str | os.PathLike[str]
) -> Recording:
    """Stitch one session's files into a :class:`Recording`.

    Parameters
    ----------
    entry : SessionEntry
        The session's registry entry.
    path_or_stream : str or path
        The directory holding the fetched files.

    Returns
    -------
    Recording
        Unchecked; :func:`neurospatial.datasets.load` runs the checks.

    Raises
    ------
    ValueError
        If the files overlap, disagree on the number of units or on a shared
        frame, or the representative LFP cannot be identified.
    pickle.UnpicklingError
        If a ``spike`` pickle imports anything outside the allow-list.
    """
    import pandas as pd

    from neurospatial import Intervals

    folder = Path(path_or_stream)
    names = {Path(member.path).name for member in entry.members}
    ordered = _in_time_order(
        path for path in folder.glob("*.npz") if path.name in names
    )
    files = [_read_file(path) for path in ordered]
    stitched = _stitch(files, entry.session)
    log = stitched.log
    times = stitched.times
    positions = log[:, [_X, _Y]] / 10.0
    tetrodes, representative = _representative(ordered[0])
    env = _maze(entry.session)
    day = entry.session[:6]
    checkpoint = entry.session[len(day) + len("_detour")]
    rat = LEARNING_RATS.get(day)
    n_units = len(stitched.spikes)
    table = pd.DataFrame(
        {
            "brain_region": ["CA1"] * n_units,
            "label": [None] * n_units,
        },
        index=pd.Index(
            [f"unit{unit:03d}" for unit in range(n_units)], name="unit_id", dtype=object
        ),
    )
    trial_files = [file for file in files if file.kind == "trial"]
    trial_rows = np.array(
        [[file.log[0, _TIME], file.log[-1, _TIME]] for file in trial_files]
    )
    trial_rows = trial_rows / 1e3
    return_rows = np.array(
        [
            [file.log[0, _TIME] / 1e3, file.log[-1, _TIME] / 1e3]
            for file in files
            if file.kind == "return"
        ]
    ).reshape(-1, 2)
    strings = [file.string for file in trial_files]
    events: dict[str, NDArray[np.float64]] = {
        "nose_poke": _flag_rows(times, log[:, _POKE] > 0),
        "reward_port": _flag_rows(times, log[:, _REWARD_PORT] > 0),
        "checkpoint": _flag_rows(times, log[:, _CHECKPOINT] > 0)[:, 0],
    }
    timeout = _flag_rows(times, log[:, _TIMEOUT] > 0)
    if timeout.size:
        events["timeout"] = timeout[:, 0]
    watch_outs = [*WATCH_OUTS.get(entry.session, ()), *_GENERAL]
    findings = (_led_watch_out(log[:, _LED], log), _gap_watch_out(stitched.gaps))
    watch_outs.extend(finding for finding in findings if finding is not None)
    if stitched.stimulation.size:
        events["stimulation"] = stitched.stimulation
        watch_outs.append(
            _stimulation_watch_out(stitched.stimulation.size, stitched.stimulation)
        )
    if not n_units:
        watch_outs.extend(_no_units(entry.session))
    params: dict[str, Any] = {
        "bin_size": BIN_SIZE,
        "track_bin_size": TRACK_BIN_SIZE,
        "min_speed": MIN_SPEED,
        "dt": DT,
        "train_epochs": trial_rows[::2],
        "test_epochs": trial_rows[1::2],
    }
    epochs = {
        "trial": _flag_rows(times, log[:, _TRIAL] > 0),
        "delay": _flag_rows(times, log[:, _DELAY] > 0),
        "trial_ready": _flag_rows(times, log[:, _READY] > 0),
        "return": return_rows,
    }
    if rat is not None and checkpoint == "G":
        point = learning_point(strings)
        params["learning_point"] = point
        if point is not None:
            epochs["learning"] = trial_rows[: point + 1]
            epochs["postlearning"] = trial_rows[point + 1 :]
    return Recording(
        **entry.identity(path_or_stream),
        animal=f"rat {rat}" if rat is not None else "a rat the archive does not name",
        apparatus=entry.info.apparatus,
        brain_regions=entry.info.brain_regions,
        clock=(
            "the video's, as log column 0 gives it, from an arbitrary start: each "
            "session has its own, so the same day's S and G sessions cannot be "
            "compared in time"
        ),
        times=times,
        spike_times=stitched.spikes,
        units=table,
        env=env,
        env_kind="declared",
        env_source=(
            f"{_SI}: 1 m square field in 20 cm lattices, 20 x 40 cm start area, "
            f"20 x 30 cm goal area, 20 cm wide L-shaped return path; placed by the "
            f"archive's {_FIELDS}, with doors open"
        ),
        positions=positions,
        position_units="cm",
        tracking=(
            "three red LEDs on the electrode assembly, ceiling camera at 24.5 Hz "
            f"({_SI})"
        ),
        epochs=epochs,
        events=events,
        trials=Intervals.from_arrays(
            trial_rows[:, 0],
            trial_rows[:, 1],
            trial=np.array([file.number for file in trial_files]),
            lattices=np.array(strings, dtype=object),
            optimized=np.array([_optimized(string) for string in strings]),
        ),
        params=params,
        conversions=(
            Conversion(
                "times: log column 0, ms / 1000",
                "stated",
                f"{_README}: 'video timestamp (ms)'",
            ),
            Conversion(
                "positions: log columns 1 and 2, mm / 10",
                "stated",
                f"{_README}: 'animal pos x (mm)'; the tracked span fits the "
                f"apparatus of the {_SI}",
            ),
            Conversion(
                "spike_times: spike ms / 1000 + the file's first frame time",
                "inferred",
                f"{_README} says only 'spike timestamps (ms)'. The values run "
                "from 0 to the file's span, are whole ticks of 30 kHz (the SI's "
                "rate), bin exactly into the 1 ms spike_hist from the file's "
                "start, and on that clock the file's per-frame counts ('cell') "
                "are the spikes within half a frame of each frame",
            ),
            Conversion(
                "epochs: runs of log columns 4 (trial), 6 (delay) and 12 (trial "
                "ready), and the spans of the return files",
                "stated",
                f"{_README} names these columns",
            ),
            Conversion(
                "events: runs of log columns 5 (nose_poke), 10 (reward_port); "
                "rising edges of 8 (checkpoint) and 9 (timeout)",
                "stated",
                f"{_README} names these columns",
            ),
            Conversion(
                f"checkpoint: lattice {checkpoint} from the session's name",
                "inferred",
                "column 8 rises inside that lattice on every trial of the five "
                "learning days' S and G sessions",
            ),
            Conversion(
                f"regions S, C1, C2, G: the file's lattices {_S}, {_C1}, {_C2}, {_G}",
                "stated",
                f"{_SI} labels u, s, g and e as S, C1, C2 and G, and "
                f"{_FIELDS} draws them in capitals",
            ),
        ),
        watch_outs=tuple(watch_outs),
        lfp=_lfp_source(entry, folder, tetrodes, representative),
    )
