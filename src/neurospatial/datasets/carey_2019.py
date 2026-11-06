"""Carey, Tanaka & van der Meer (2019): rat CA1 on a food-or-water T-maze (DataLad).

What was recorded: 16-tetrode recordings in dorsal CA1 of four male Long-Evans
rats, six daily sessions each. On each day a rat rested on a flower pot, ran
an elevated T-maze for food at the end of the left arm or water at the end of
the right, resting on a pedestal beside the arm between trials, then rested
again. Before each session it was restricted of food or water overnight, so
the two outcomes changed value from day to day. The paper asks what
hippocampal replay depicts after that change.

Where it lives: DataLad ``labs/mvdm/MotivationalT`` (ODC-PDDL-1.0), one folder
of loose files per session, each pinned by the SHA-256 in its git-annex key.
A load fetches the spike files, the video tracker's file, the events, the
metadata and ``keys.m``, about 274 MB, one file at a time; the LFP's ``.ncs``
files, about 20 MB a channel, come with ``rec.lfp.fetch``.

Sources read: the paper and its supplement (Nature Neuroscience 22, 1450-1459),
the archive's ``README.txt`` and ``datapackage.json``, the files, the paper's
code (github.com/vandermeerlab/papers at 3aebc8e, read, not run) and the
lab's code it pins (github.com/vandermeerlab/vandermeerlab at ad0bbd4, read,
not run): ``LoadSpikes.m``, ``LoadPos.m``, ``LoadCSC.m``, ``getEvents_Tmaze.m``,
``GenCandidateEvents.m``, ``precand.m``, ``amSWR.m``, ``amMUA.m``,
``GetMatchedTrials.m``, ``Generate_DecSeq.m``.

What the loader converts:

- spike times: each ``.t`` or ``._t`` file holds one unit's big-endian
  ``uint32`` times in tenths of a millisecond (its header says so), on the
  clock of the ``.nvt`` and ``.ncs`` files. Both extensions are loaded, and
  ``units["quality"]`` says which: ``._t`` is the lab's "questionable" rating,
  and the paper's code loads them (``load_questionable_cells = 1``, "load 5's").
  Its "at least 40 cells" keeps five R042 sessions only if both are counted.
- positions: the tracker's pixels / ``keys.m`` ``convFact``, in cm, on the
  camera's axes: ``y`` runs down the image, so a plot with ``y`` up is
  mirrored. A frame with ``x`` or ``y`` 0 is the format's mark for no
  detection and is NaN (``LoadPos.m``, ``removeZeros``, drops them).
- LFP: stored values x ``ADBitVolts``, microvolts, 2 kHz, each record's
  timestamp plus the sample's place in it. Channels ``CSC*`` are tetrode
  channels and ``HS*R*`` the reference channels.
- headings: none; the tracker's angle field does not follow movement.

The declared environment is the T-maze as a graph of the authors' two drawn
trajectories (``coordL_cm`` and ``coordR_cm``, joined at ``chp_cm``), 3 cm bins.
The rest pedestals and the flower pot lie off the track, so samples there fall
in no bin. :func:`trajectory_environment` gives the left or right trajectory
alone, as the paper decodes them.

Analysis choices (``params``):

- ``bin_size`` 3 cm and ``min_speed`` 4 cm/s, the paper's text; its code gates
  runs at 5 cm/s and detection at 10 pixels/s, 3.4 to 4.2 cm/s by axis and rat;
- ``analysis_epochs``: the trials; ``train_epochs`` / ``test_epochs``: their
  even and odd rows, for held-out decoding at ``dt`` 0.25 s;
- ``restriction``, what was withheld, ``layout`` and ``session_type``, from
  ``keys.m``; ``in_paper``, whether the paper analysed the session;
- ``left_trajectory``, ``right_trajectory``, ``choice_point`` and
  ``track_width``, cm;
- ``theta_channel``, the session's ``goodTheta`` first channel.

``rec.lfp.ripple_channels`` is the session's ``goodSWR`` with the channel the
paper's candidate file names first: in 3 of the 24 sessions, this one among
them, that is not the first ``goodSWR``.
"""

from __future__ import annotations

import os
import re
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from neurospatial.datasets._lfp import LFPSource
from neurospatial.datasets._neuralynx import (
    _spike_words,
    is_64_bit,
    read_events,
    read_lfp,
    read_spike_times,
    read_video_track,
)
from neurospatial.datasets._recording import (
    Conversion,
    Recording,
    heading_vs_movement,
    tracked_mask,
)
from neurospatial.datasets._tracks import nearest_segment, placement_tolerance

if TYPE_CHECKING:
    import networkx as nx

    from neurospatial import Environment
    from neurospatial.datasets._catalog import SessionEntry

__all__ = [
    "ACROSS_SESSIONS",
    "EXPECTED_WATCH_OUTS",
    "REQUIRES",
    "WATCH_OUTS",
    "WORKED_RESULT",
    "load_session",
    "matched_trials",
    "parse_keys",
    "read_session_lfp",
    "track_graph",
    "trajectory_environment",
]

#: Modules of the ``datasets`` extra the loader imports.
REQUIRES: tuple[str, ...] = ()

#: How far the track's corners may let it stray from the drawn trajectory, cm.
PATH_TOLERANCE = 0.5
#: The paper's linearized bins: "approximately 3 cm per bin", cm.
BIN_SIZE = 3.0
#: The share of trial samples the placement tolerance leaves outside the track.
TOLERANCE_QUANTILE = 0.99
#: The rate of the sessions' ``.ncs`` files, Hz (each header's SamplingFrequency).
LFP_RATE = 2000.0
#: The speed threshold of the paper's text, for detection and run epochs, cm/s.
MIN_SPEED = 4.0
#: Above this speed, cm/s, the tracker's angle is compared with movement.
HEADING_MIN_SPEED = 10.0
#: The decoding bin of ``params["dt"]``, s.
DT = 0.25
#: Events of the ``.nev`` file, as ``getEvents_Tmaze.m`` names them.
EVENT_TEXTS = {
    "center_photobeam": "TTL Input on AcqSystem1_0 board 0 port 1 value (0x0020).",
    "left_photobeam": "TTL Input on AcqSystem1_0 board 0 port 1 value (0x0080).",
    "right_photobeam": "TTL Input on AcqSystem1_0 board 0 port 1 value (0x0040).",
    "food_dispensed": "TTL Output on AcqSystem1_0 board 0 port 0 value (0x0004).",
    "water_dispensed": "TTL Output on AcqSystem1_0 board 0 port 0 value (0x0040).",
}
#: The sessions the paper analysed: ``fd([2:6 10:11 13:24])`` of the 24 sorted
#: session folders in ``getTmazeDataPath.m`` at the commit the paper pins.
PAPER_SESSIONS = frozenset(
    (
        "R042-2013-08-17",
        "R042-2013-08-18",
        "R042-2013-08-19",
        "R042-2013-08-20",
        "R042-2013-08-21",
        "R044-2013-12-21",
        "R044-2013-12-22",
        "R050-2014-03-29",
        "R050-2014-03-30",
        "R050-2014-03-31",
        "R050-2014-04-01",
        "R050-2014-04-02",
        "R050-2014-04-03",
        "R064-2015-04-18",
        "R064-2015-04-19",
        "R064-2015-04-20",
        "R064-2015-04-21",
        "R064-2015-04-22",
        "R064-2015-04-23",
    )
)
#: The channel each session's candidate file names (``parameters.amSWR.csc`` of
#: ``SWRcandidates/<session>-candidates.zip`` in the paper's code, read from
#: all 24): the one its detection ran on, which ``keys.m`` lists as a
#: ``goodSWR`` but not always first.
CANDIDATE_CHANNELS = {
    "R042-2013-08-16": "CSC14a",
    "R042-2013-08-17": "CSC05a",
    "R042-2013-08-18": "CSC11a",
    "R042-2013-08-19": "CSC04a",
    "R042-2013-08-20": "CSC11a",
    "R042-2013-08-21": "CSC15a",
    "R044-2013-12-18": "CSC14a",
    "R044-2013-12-19": "CSC14a",
    "R044-2013-12-20": "CSC14a",
    "R044-2013-12-21": "CSC14a",
    "R044-2013-12-22": "CSC15a",
    "R044-2013-12-23": "CSC15a",
    "R050-2014-03-29": "CSC03a",
    "R050-2014-03-30": "CSC07a",
    "R050-2014-03-31": "CSC07a",
    "R050-2014-04-01": "CSC06a",
    "R050-2014-04-02": "CSC07a",
    "R050-2014-04-03": "CSC07a",
    "R064-2015-04-18": "CSC01a",
    "R064-2015-04-19": "CSC07a",
    "R064-2015-04-20": "CSC07a",
    "R064-2015-04-21": "CSC07a",
    "R064-2015-04-22": "CSC07a",
    "R064-2015-04-23": "CSC07a",
}

#: The worked result: sequences in the paper's candidate events, judged by
#: CAREY_2019, against nulls fixed in writing before any decode.
WORKED_RESULT: dict[str, Any] = {
    "status": "sanity property",
    "level": "B",
    "finding": (
        "a sanity property of the detection and decoder, not a number of the "
        "paper: candidate events hold left or right sequences more often than "
        "when the rate maps or the spikes are shifted"
    ),
    "quote": (
        '"3,780 total sequences were detected across all animals and sessions" '
        '(Results); "Individual sequence numbers are 1257 sequences for R042 (5 '
        "sessions), 95 for R044 (2 sessions), 429 for R050 (6 sessions) and 1999 "
        'for R064 (6 sessions)." (Supplementary Fig. 6)'
    ),
    "source": (
        "Carey, Tanaka & van der Meer 2019, Nature Neuroscience 22:1450-1459, "
        "Results, Methods and the Supplementary Fig. 6 legend"
    ),
    "paper": (
        "no number for one session; 429 sequences over R050's 6 sessions, with "
        "no figure per session, from a pipeline with filters CAREY_2019 leaves out"
    ),
    "ours": 0.049,
    "criterion": (
        "65 of the 1,328 candidates (4.9%) hold a sequence on the left or right "
        "trajectory, which exceeds the 95th percentile of each null (20 draws, "
        "seed 2026): rate maps fitted on spikes with each unit shifted by its "
        "own offset within the task, 22 to 65 events, 95th percentile 52.65; "
        "decoded spikes all shifted by one offset over the recording, 2 to 14 "
        "events, 95th percentile 10.2. Fixed in writing before any decode. The "
        "margin over the first null is thin: its median is 37 events, and one "
        "draw equals the observed count"
    ),
    "parameters": (
        "literature_methods.carey_2019 at ripple_detection 0eddae64 as packaged, "
        "on CSC03a, with the file's 50 hand-labelled SWRs. Rate maps per side "
        "on trajectory_environment (3 cm bins): compute_spatial_rates binned, "
        "bandwidth 3 cm, min_speed 4 cm/s, on the matched trials (6 of each), "
        "each from 1 s before its next centre photobeam, the track's first 20 "
        "and last 15 cm left out, units with at least 25 such spikes (37 left, "
        "44 right). Decoding: counts in 5 ms bins summed over 25 ms, rows 5 ms "
        "apart, dt 0.025, the default prior (uniform over space), each candidate "
        "plus 0.25 s on each side. Judged by CAREY_2019: at least 4 active units "
        "in a window, MAP steps under 40 cm, at least 50 ms; a candidate counts "
        "when such a sequence overlaps it, the paper's rule"
    ),
    "unmatched": (
        "the paper's identity-shuffle filter (500 shuffles over the whole "
        "session) is not applied, and no claim is made about what it does",
        "the paper counts sequences left or right but not both, 50 events here "
        "(15 left only, 35 right only, against 34 in the first null's median), "
        "and it scans the whole session; here only windows around candidates",
        "spike density by a 2 ms Gaussian before the 25 ms boxcar is not applied",
        "the code's run speed is 5 cm/s and its detection gate 10 pixels/s, 3.4 "
        "to 4.2 cm/s by axis and rat; the paper's text says about 4 cm/s, used "
        "here",
        "the code's Poisson exposure is the 5 ms step for 25 ms counts "
        "(DecodeZ.m); here dt 0.025",
        "the detection differs: see the page's validation; on the paper's own "
        "1,654 candidates the same analysis finds 64 events, above both nulls "
        "(95th percentiles 61.8 and 12)",
    ),
    "attempted": None,
    "why": None,
}

#: Exploratory, apart from ``WORKED_RESULT``: the same analysis on the paper's
#: own candidates in its 19 sessions. Written down before those sessions ran; the
#: label permutation and the null were fixed after the counts were seen. No
#: criterion attaches. It does not reproduce the paper's shift: the left share
#: differs by restriction just as much with shifted rate maps, and follows the
#: share of units kept on each side (r = 0.83). This analysis cannot tell that
#: shift from a bias. All of this is measured on this analysis; it does not
#: assess the paper's own finding.
ACROSS_SESSIONS: dict[str, Any] = {
    "what": (
        "events with a CAREY_2019 sequence on the left or right trajectory only, "
        "in the paper's 19 sessions, its own candidates, its identity-shuffle "
        "filter not applied; events, where the paper counts sequences"
    ),
    "one_side_per_rat": {"R042": 1127, "R044": 70, "R050": 366, "R064": 1584},
    "paper_per_rat": {"R042": 1257, "R044": 95, "R050": 429, "R064": 1999},
    "left_share_food": 0.294,
    "left_share_water": 0.520,
    "paper_left_share": (0.35, 0.57),
    "label_permutation": {"z": -2.65, "p": 0.0047, "n": 10000, "seed": 2026},
    "r_with_left_trials": -0.69,
    "paper_r_with_behavior": -0.54,
    "left_share_follows_kept_units_r": 0.83,
    "null_own_offsets": {
        "draws": 20,
        "seed": 2026,
        "difference_observed": -0.227,
        "difference_median": -0.223,
        "difference_range": (-0.416, -0.091),
        "one_side_total_observed": 3147,
        "one_side_total_range": (2317, 2687),
        "sessions_above_95th_percentile": 5,
    },
}

#: Findings computed from the data that each verified session must report, as
#: patterns searched for in its watch_outs.
EXPECTED_WATCH_OUTS: dict[str, tuple[str, ...]] = {
    "R050-2014-03-29": (
        r"video frames \(3\.7%\) have no tracked point",
        r"tracked samples lie off the track",
        r"runs of 11 or more identical position samples",
        r"does not follow the head",
        r"The recording stops 2 times",
    ),
}

#: Pitfalls of each session checked by hand, with their evidence.
WATCH_OUTS: dict[str, tuple[str, ...]] = {
    "R050-2014-03-29": (
        "The LFP lags the spikes by about 2 ms: the .ncs headers state a "
        "DspFilterDelay of 1984 us with delay compensation disabled, and neither "
        "the lab's code nor this loader removes it.",
    ),
}

_NUMBER = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
_ALIAS = re.compile(r"ExpKeys\.(\w+)\s*;")
_ASSIGN = re.compile(r"^\s*ExpKeys\.(\w+)\s*=\s*(.*)$")
_UNIT = re.compile(r"-(TT\d+)r?_(\d+)\.(_?t)$")


def _key_value(text: str) -> Any:
    """One value of a ``keys.m`` assignment: text, a cell of text or numbers."""
    text = text.strip()
    if text.startswith("'"):
        return text[1 : text.index("'", 1)]
    if text.startswith("{"):
        return tuple(re.findall(r"'([^']*)'", text[: text.index("}")]))
    if text.startswith("["):
        body = text[1 : text.index("]")].replace(",", " ")
        rows = [row.split() for row in body.split(";") if row.strip()]
        return np.array(rows, dtype=np.float64).squeeze() if rows else np.empty(0)
    number = text.split(";")[0].split("%")[0]
    return float(number) if _NUMBER.fullmatch(number.strip()) else None


def parse_keys(text: str) -> dict[str, Any]:
    """The ``ExpKeys.<name> = <value>;`` lines of a ``keys.m`` file.

    The file is MATLAB code, so it is read as text and never run.

    Parameters
    ----------
    text : str
        The file's text.

    Returns
    -------
    dict of str to object
        Each name and its value: a str for a quoted string, a tuple of str for
        a cell array, a float or an ndarray for numbers, an earlier key's value
        for ``ExpKeys.a = ExpKeys.b;``, None for anything else (an expression,
        which is not evaluated).

    Examples
    --------
    >>> keys = parse_keys("ExpKeys.Layout = 'foodLeft'; % or 'foodRight'")
    >>> keys["Layout"]
    'foodLeft'
    """
    keys: dict[str, Any] = {}
    for line in text.splitlines():
        if assignment := _ASSIGN.match(line):
            same_as = _ALIAS.match(assignment[2])
            keys[assignment[1]] = (
                keys[same_as[1]] if same_as else _key_value(assignment[2])
            )
    return keys


def _one(folder: Path, pattern: str) -> Path:
    """The session's one file matching ``pattern``."""
    found = sorted(folder.glob(pattern))
    if len(found) != 1:
        msg = f"{folder.name}: {len(found)} files match {pattern!r}, not one"
        raise ValueError(msg)
    return found[0]


def _read_units(folder: Path) -> tuple[tuple[NDArray[np.float64], ...], Any, int]:
    """Every sorted unit by tetrode and cluster, its table, and its 64-bit files."""
    import pandas as pd

    named = sorted(
        (match[1], int(match[2]), match[3], path)
        for path in folder.iterdir()
        if (match := _UNIT.search(path.name)) is not None
    )
    trains = tuple(np.sort(read_spike_times(row[3])) for row in named)
    n_wide = sum(is_64_bit(_spike_words(row[3])) for row in named)
    table = pd.DataFrame(
        {
            "brain_region": "dorsal CA1",
            "label": None,
            "tetrode": [row[0] for row in named],
            "quality": ["good" if row[2] == "t" else "questionable" for row in named],
        },
        index=pd.Index([f"{row[0]}_{row[1]}" for row in named], name="unit_id"),
    )
    return trains, table, n_wide


def read_session_lfp(
    sources: tuple[str | Path, ...], channels: tuple[str, ...]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Read channels from their ``.ncs`` files, in microvolts.

    Parameters
    ----------
    sources : tuple of path
        The channels' files, ``<session>-<channel>.ncs``.
    channels : tuple of str
        The channels, in the order of the returned columns.

    Returns
    -------
    samples : ndarray, shape (n_time, n_channels)
        Microvolts: stored values times the header's ``ADBitVolts`` times 1e6.
    times : ndarray, shape (n_time,)
        Seconds on the acquisition clock.

    Raises
    ------
    ValueError
        If two channels' samples do not share their times.
    """
    files = {_channel_of(Path(source).name): Path(source) for source in sources}
    columns, times = [], None
    for channel in channels:
        stored, clock, header = read_lfp(files[channel])
        if times is not None and not np.array_equal(times, clock):
            msg = f"{channel} does not share the times of {channels[0]}"
            raise ValueError(msg)
        times = clock
        columns.append(stored * float(header["ADBitVolts"]) * 1e6)
    assert times is not None  # fetch asks for at least one channel
    return np.column_stack(columns), times


def _channel_of(file_name: str) -> str:
    """``CSC03a`` from ``R050-2014-03-29-CSC03a.ncs``."""
    return file_name.removesuffix(".ncs").rsplit("-", 1)[-1]


def _polyline_nodes(path: NDArray[np.float64]) -> NDArray[np.float64]:
    """The corners of a drawn path, within ``PATH_TOLERANCE`` of it, shape (n, 2)."""
    from shapely import LineString

    corners = LineString(path).simplify(PATH_TOLERANCE)
    return np.asarray(corners.coords, dtype=np.float64)


def _add_chain(graph: nx.Graph, names: list[Any], points: NDArray[np.float64]) -> None:
    """Add a chain of nodes at ``points``, each edge's ``distance`` its length."""
    for name, point in zip(names, points, strict=True):
        graph.add_node(name, pos=(float(point[0]), float(point[1])))
    for a, b in pairwise(names):
        step = np.subtract(graph.nodes[a]["pos"], graph.nodes[b]["pos"])
        graph.add_edge(a, b, distance=float(np.linalg.norm(step)))


def track_graph(
    left: NDArray[np.float64], right: NDArray[np.float64], choice: NDArray[np.float64]
) -> nx.Graph:
    """The T-maze as a graph, from the authors' two drawn trajectories.

    Parameters
    ----------
    left, right : ndarray, shape (n_points, 2)
        The drawn trajectory from the start of the stem to the end of each
        arm, cm.
    choice : ndarray, shape (2,)
        The choice point, cm.

    Returns
    -------
    networkx.Graph
        Nodes ``stem_<k>`` from the start, ``choice``, and ``left_<k>`` and
        ``right_<k>`` out along each arm, with ``pos`` in cm; each edge's
        ``distance`` is its length, cm. The stem is the left trajectory's, up
        to the choice point.
    """
    import networkx as nx

    def split(path: NDArray[np.float64]) -> int:
        return int(np.argmin(np.linalg.norm(path - choice, axis=1)))

    stem = _polyline_nodes(np.vstack([left[: split(left)], choice]))
    graph = nx.Graph()
    names: list[Any] = [f"stem_{k}" for k in range(len(stem) - 1)] + ["choice"]
    _add_chain(graph, names, stem)
    for side, path in (("left", left), ("right", right)):
        arm = _polyline_nodes(np.vstack([choice, path[split(path) + 1 :]]))
        names = ["choice"] + [f"{side}_{k}" for k in range(len(arm) - 1)]
        _add_chain(graph, names, arm)
    return graph


def trajectory_environment(
    path: NDArray[np.float64], *, track_width: float, name: str = ""
) -> Environment:
    """One drawn trajectory as a linear track, as the paper decodes each side.

    Parameters
    ----------
    path : ndarray, shape (n_points, 2)
        ``rec.params["left_trajectory"]`` or ``["right_trajectory"]``, cm.
    track_width : float
        ``rec.params["track_width"]``, cm.
    name : str, optional
        The environment's name.

    Returns
    -------
    Environment
        A track with ``BIN_SIZE`` bins, its first bin at the start of the stem.
    """
    import networkx as nx

    from neurospatial import Environment

    graph = nx.Graph()
    corners = _polyline_nodes(path)
    names = list(range(len(corners)))
    _add_chain(graph, names, corners)
    return Environment.from_graph(
        graph,
        list(pairwise(names)),
        bin_size=BIN_SIZE,
        units="cm",
        name=name,
        track_width=track_width,
    )


def matched_trials(
    arms: NDArray[Any], bad: NDArray[np.intp]
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Equal numbers of left and right trials, nearest in order, as the paper's.

    ``GetMatchedTrials.m``: bad trials are left out; each trial of the rarer
    side, in order, takes the nearest unused trial of the other by trial
    number, the later one when two are equally near (``nearest_idx3``). The
    paper: "(1) the number of left and right trials was equal and (2) the
    time difference between the left and right trials was minimized".

    Parameters
    ----------
    arms : ndarray of str, shape (n_trials,)
        ``"L"`` or ``"R"``.
    bad : ndarray of int, shape (n_bad,)
        1-based trial numbers to leave out, as ``keys.m`` lists them.

    Returns
    -------
    left, right : ndarray of int, shape (n_matched,)
        0-based rows of the matched trials, each sorted.

    Examples
    --------
    >>> left, right = matched_trials(np.array(list("LRRRL")), np.array([2]))
    >>> left.tolist(), right.tolist()
    ([0, 4], [2, 3])
    """
    usable = np.ones(arms.size, dtype=bool)
    usable[np.asarray(bad, dtype=np.intp) - 1] = False
    left = np.flatnonzero((arms == "L") & usable)
    right = np.flatnonzero((arms == "R") & usable)
    if left.size == right.size:
        return left, right
    few, many = (left, right) if left.size < right.size else (right, left)
    pool = many.tolist()
    chosen = []
    for trial in few:
        distance = np.abs(np.array(pool) - trial)
        nearest = int(np.flatnonzero(distance == distance.min())[-1])
        chosen.append(pool.pop(nearest))
    matched = np.sort(np.array(chosen, dtype=np.intp))
    return (few, matched) if left.size < right.size else (matched, few)


def _intervals(table: Any) -> NDArray[np.float64]:
    """An ``iv`` struct of the metadata as (n, 2) rows, seconds."""
    return np.column_stack(
        [np.atleast_1d(table.tstart), np.atleast_1d(table.tend)]
    ).astype(np.float64)


def _positions_cm(
    x: NDArray[np.int32], y: NDArray[np.int32], factor: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Tracked pixels / the session's cm factors; NaN where the tracker found nothing.

    Parameters
    ----------
    x, y : ndarray of int, shape (n_frames,)
        Pixels, 0 where nothing was tracked.
    factor : ndarray, shape (2,)
        ``convFact``: pixels per cm along x and y.

    Returns
    -------
    ndarray, shape (n_frames, 2)
    """
    positions = np.column_stack([x / factor[0], y / factor[1]])
    positions[(x == 0) | (y == 0)] = np.nan
    return positions


def _inside(times: NDArray[np.float64], rows: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Whether each time lies in one of sorted, disjoint [start, stop) ``rows``."""
    row = np.searchsorted(rows[:, 0], times, side="right") - 1
    held = row >= 0
    held[held] = times[held] < rows[row[held], 1]
    return held


def _declared_track(
    graph: nx.Graph,
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    trials: NDArray[np.float64],
    session: str,
) -> tuple[Environment, float, Conversion, NDArray[np.float64]]:
    """The T-maze as a track, its width sized from the trials' samples.

    Returns
    -------
    env : Environment
    width : float
        ``track_width``, cm.
    placement : Conversion
        The width used and its evidence.
    distance : ndarray, shape (n_tracked,)
        Each tracked sample's distance from the track, cm.
    """
    from neurospatial import Environment

    order = list(graph.edges)
    pos = {node: np.asarray(graph.nodes[node]["pos"]) for node in graph.nodes}
    starts = np.array([pos[a] for a, _ in order])
    ends = np.array([pos[b] for _, b in order])
    tracked = tracked_mask(positions)
    distance = nearest_segment(positions[tracked], starts, ends).distance
    width, _, tolerance = placement_tolerance(
        distance[_inside(times[tracked], trials)],
        TOLERANCE_QUANTILE,
        floor=BIN_SIZE,
        measured="the tracked point's distance from the track during trials",
        reason=(
            "No source read states the track's width. Only trial samples size "
            "it: between trials the rat rests off the track."
        ),
    )
    env = Environment.from_graph(
        graph,
        order,
        bin_size=BIN_SIZE,
        units="cm",
        name=f"carey_2019 {session}",
        track_width=width,
    )
    placement = Conversion(
        f"environment: track_width {width:.3g} cm, a placement tolerance",
        "inferred",
        tolerance,
    )
    return env, width, placement, distance


def _lfp_source(entry: SessionEntry, keys: dict[str, Any], folder: Path) -> LFPSource:
    """The session's LFP: one ``.ncs`` file per channel, nothing fetched."""
    held = [name for file in entry.lfp_files for name in file.channels]
    good = [_channel_of(name) for name in keys["goodSWR"]]
    theta = [_channel_of(name) for name in keys["goodTheta"]]
    paper = CANDIDATE_CHANNELS[entry.session]
    ripple = (paper, *(name for name in good if name != paper and name in held))

    def described(channel: str) -> str:
        roles = [
            "tetrode channel" if channel.startswith("CSC") else "reference channel",
            *(["the paper's ripple channel"] if channel == paper else []),
            *(["a goodSWR channel"] if channel in good and channel != paper else []),
            *(["the first goodTheta channel"] if channel == theta[0] else []),
        ]
        return ", ".join(roles)

    return LFPSource(
        dataset=entry.info.dataset,
        files=entry.lfp_files,
        sampling_rate=LFP_RATE,
        channels={name: described(name) for name in held},
        unit="uV",
        unit_basis=Conversion(
            "LFP: stored int16 x the file's ADBitVolts x 1e6, in microvolts",
            "stated",
            "each .ncs header's ADBitVolts, volts per stored unit; LoadCSC.m at "
            "the commit the paper pins scales the same way",
        ),
        times_basis=Conversion(
            "LFP times: each record's timestamp in microseconds, plus k / 2000 s "
            "for its k-th sample",
            "stated",
            "the Neuralynx record layout, as LoadCSC.m builds the times; on the "
            "default session they equal the tvec of the paper's candidate file",
        ),
        reader=read_session_lfp,
        ripple_channels=ripple,
        cache=folder.parents[1],
    )


def _cut_out(folder: Path, name: str) -> NDArray[np.float64] | None:
    """A file's kept ``[t_start, t_end]`` rows in seconds; None without the file."""
    from scipy.io import loadmat

    found = sorted(folder.glob(name))
    if not found:
        return None
    table = loadmat(found[0], squeeze_me=True)
    rows = np.column_stack(
        [np.atleast_1d(table["t_start"]), np.atleast_1d(table["t_end"])]
    )
    return rows[np.argsort(rows[:, 0])].astype(np.float64) * 1e-6


def _within(
    rows: NDArray[np.float64], times: NDArray[np.float64]
) -> NDArray[np.float64]:
    """``rows`` cut to the recording, empty ones dropped: some run 0 to infinity."""
    clipped = np.clip(rows, times[0], times[-1])
    return clipped[clipped[:, 0] < clipped[:, 1]]


def _epochs(
    keys: dict[str, Any],
    metadata: Any,
    folder: Path,
    matched: tuple[Any, Any],
    times: NDArray[np.float64],
) -> tuple[dict[str, NDArray[np.float64]], tuple[str, ...]]:
    """The session's named epochs, and the watch-outs its optional files give."""
    task = metadata.taskvars
    trials = _intervals(task.trial_iv)
    epochs = {
        "prerecord": np.atleast_2d(keys["prerecord"]),
        "task": np.atleast_2d(keys["task"]),
        "postrecord": np.atleast_2d(keys["postrecord"]),
        "trials_left": _intervals(task.trial_iv_L),
        "trials_right": _intervals(task.trial_iv_R),
        "matched_left": trials[matched[0]],
        "matched_right": trials[matched[1]],
        "rest": _intervals(task.rest_iv),
        "rest_left_pedestal": _intervals(task.rest_iv_pedL),
        "rest_right_pedestal": _intervals(task.rest_iv_pedR),
    }
    notes = []
    if (kept := _cut_out(folder, "*-times.mat")) is not None:
        kept = _within(kept, times)
        epochs["spikes_kept"] = kept
        notes.append(
            f"Spikes were cut out while the rat ate: {len(kept)} rows of "
            "*times.mat are the times kept, epochs['spikes_kept'], so a silent "
            "stretch outside them is not silence (the archive's README: 'times "
            "when the animal was consuming rewards were cut out from the spike "
            "data'). The LFP and positions run on. R042 was recorded referenced "
            "to ground."
        )
    if "detachIV" in metadata._fieldnames:
        attached = _intervals(metadata.detachIV)
        epochs["headstage_attached"] = attached
        notes.append(
            f"The headstage came off: metadata.detachIV holds {len(attached)} rows "
            "of attached time, epochs['headstage_attached'] (the archive's README: "
            "'a few times where the headstage detached'). The paper's detection "
            "keeps only events inside them."
        )
    if (kept := _cut_out(folder, "*HS_detach_times.mat")) is not None:
        epochs["headstage_attached_by_hand"] = _within(kept, times)
    return epochs, tuple(notes)


def _left_is_right_of_travel(graph: nx.Graph) -> bool:
    """Whether the left arm lies to the right of travel up the stem, as stored."""
    start = np.asarray(graph.nodes["stem_0"]["pos"])
    choice = np.asarray(graph.nodes["choice"]["pos"])
    end = np.asarray(
        graph.nodes[
            max(
                (n for n in graph.nodes if str(n).startswith("left_")),
                key=lambda n: int(str(n).split("_")[1]),
            )
        ]["pos"]
    )
    along, turn = choice - start, end - choice
    return bool(along[0] * turn[1] - along[1] * turn[0] < 0)


def _heading_note(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    angle: NDArray[np.int32],
) -> str:
    """Why the tracker's angle field is not ``headings``, with the numbers."""
    if not np.any(angle != 0):
        return "No headings: the tracker's angle field is empty."
    radians = np.where(angle == 0, np.nan, np.radians(angle.astype(np.float64)))
    offset, spread = heading_vs_movement(
        times, positions, np.angle(np.exp(1j * radians)), min_speed=HEADING_MIN_SPEED
    )
    return (
        "No headings: the tracker's angle field minus the direction of movement "
        f"above {HEADING_MIN_SPEED:g} cm/s has a circular mean of "
        f"{np.degrees(offset):+.0f} degrees and a median spread of "
        f"{np.degrees(spread):.0f} degrees, so it does not follow the head."
    )


def _watch_outs(
    entry: SessionEntry,
    keys: dict[str, Any],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    angle: NDArray[np.int32],
    off_track: NDArray[np.bool_],
    units: Any,
    graph: nx.Graph,
) -> tuple[str, ...]:
    """The findings this session's files give, each with its numbers."""
    n_lost = int(np.sum(~tracked_mask(positions)))
    off = float(np.mean(off_track))
    n_questionable = int(np.sum(units["quality"] == "questionable"))
    steps = np.diff(times)
    gaps = steps[steps > 1.0]
    good = [_channel_of(name) for name in keys["goodSWR"]]
    paper = CANDIDATE_CHANNELS[entry.session]
    bad = np.atleast_1d(keys["badTrials"]).astype(int).tolist()
    notes = [
        f"{n_lost} of {times.size} video frames ({n_lost / times.size:.1%}) have no "
        "tracked point (x or y 0, the file's mark for no detection, which the "
        "lab's LoadPos.m drops): positions are NaN there.",
        f"{off:.1%} of tracked samples lie off the track: the rat rests on a "
        "flower pot before and after the task and on a pedestal beside the arms "
        "between trials, so most of a session falls in no bin of rec.env. "
        "Epochs say where it was.",
        f"{n_questionable} of {len(units)} units are the lab's 'questionable' ._t "
        "clusters (units['quality']). The paper's code loads them, and its count "
        "of cells includes them; filter them if you want well-isolated units.",
        _heading_note(times, positions, angle),
    ]
    if gaps.size:
        notes.append(
            f"The recording stops {gaps.size} times, for "
            f"{', '.join(f'{gap:.1f} s' for gap in gaps)}, between the session's "
            "recording blocks: samples, spikes and LFP all have gaps there."
        )
    if _left_is_right_of_travel(graph):
        notes.append(
            "Positions are on the camera's axes, whose y runs down the image: "
            "with y up, a plot is mirrored and the left (food) arm is drawn on "
            "the rat's right as it runs up the stem."
        )
    if good[0] != paper:
        notes.append(
            f"keys.m lists {good[0]} first among goodSWR, but the paper's candidate "
            f"file names {paper}, so rec.lfp.ripple_channels starts with {paper}."
        )
    if bad:
        notes.append(
            f"Trials {bad} are keys.m's badTrials, where the rat was interfered "
            "with; rec.trials.bad marks them, and epochs['matched_left'] and "
            "['matched_right'] leave them out, as the paper's code does."
        )
    if entry.session not in PAPER_SESSIONS:
        notes.append(
            "The paper did not analyse this session (getTmazeDataPath.m lists its "
            "19 sessions). Its stated rule of at least 40 cells does not select "
            "those 19 alone: R044-2013-12-20 (43 units) and R044-2013-12-23 (41) "
            "reach it and are out, with no stated reason."
        )
    return tuple(notes)


def load_session(
    entry: SessionEntry, path_or_stream: str | os.PathLike[str]
) -> Recording:
    """Read one session's folder into a :class:`Recording`.

    Parameters
    ----------
    entry : SessionEntry
        The session's registry entry.
    path_or_stream : str or path
        The directory holding the session's fetched files.

    Returns
    -------
    Recording
        Unchecked; :func:`neurospatial.datasets.load` runs the checks.

    Raises
    ------
    ValueError
        If a file the session needs is missing or repeated, or the authors'
        trajectories in pixels and cm disagree with the keys' cm factors.
    """
    from scipy.io import loadmat

    from neurospatial import Intervals

    folder = Path(path_or_stream)
    keys = parse_keys(_one(folder, "*keys.m").read_text(encoding="latin-1"))
    metadata = loadmat(
        _one(folder, "*-metadata.mat"), squeeze_me=True, struct_as_record=False
    )["metadata"]
    track = read_video_track(_one(folder, "*.nvt"))
    factor = np.asarray(keys["convFact"], dtype=np.float64)
    coord = metadata.coord
    drawn = np.ptp(coord.coordL, axis=1) / np.ptp(coord.coordL_cm, axis=1)
    if not np.allclose(drawn, factor, rtol=1e-3):
        msg = (
            f"{entry.session}: the drawn trajectory's pixels per cm are {drawn}, "
            f"not keys.m's convFact {factor}"
        )
        raise ValueError(msg)
    positions = _positions_cm(track.x, track.y, factor)
    spike_times, units, n_wide = _read_units(folder)
    left, right = coord.coordL_cm.T, coord.coordR_cm.T
    choice = np.asarray(coord.chp_cm, dtype=np.float64)
    graph = track_graph(left, right, choice)
    task = metadata.taskvars
    trials = _intervals(task.trial_iv)
    env, width, placement, distance = _declared_track(
        graph, track.times, positions, trials[np.argsort(trials[:, 0])], entry.session
    )
    arms = np.array([str(arm) for arm in np.atleast_1d(task.sequence)], dtype=object)
    bad = np.atleast_1d(keys["badTrials"]).astype(np.intp)
    matched = matched_trials(arms, bad)
    epochs, notes = _epochs(keys, metadata, folder, matched, track.times)
    if n_wide:
        notes = (
            *notes,
            f"{n_wide} of {len(units)} spike files hold 64-bit times, though their "
            "headers do not say so: every other 32-bit word is 0. The loader reads "
            "the times; the lab's LoadSpikes.m reads 32 bits, which adds a spike "
            "at 0 s before each.",
        )
    event_times, texts = read_events(_one(folder, "*-Events.nev"))
    words = np.array(texts, dtype=object)
    events: dict[str, NDArray[np.float64]] = {
        name: event_times[words == text] for name, text in EVENT_TEXTS.items()
    }
    events["hand_labelled_swr"] = _intervals(metadata.SWRtimes)
    is_bad = np.zeros(arms.size, dtype=bool)
    is_bad[bad - 1] = True
    n_units = len(units)
    return Recording(
        **entry.identity(path_or_stream),
        animal=f"rat {entry.session.split('-')[0]}",
        apparatus=entry.info.apparatus,
        brain_regions=entry.info.brain_regions,
        clock=(
            "the Neuralynx acquisition clock, as stored in the video, LFP, event "
            "and spike files"
        ),
        times=track.times,
        spike_times=spike_times,
        units=units,
        env=env,
        env_kind="declared",
        env_source=(
            "the authors' drawn left and right trajectories (metadata coord), "
            f"{BIN_SIZE:g} cm bins; keys.m states {keys['pathlength']:g} cm from "
            f"start to reward, {keys['patharms']:g} cm from reward to reward"
        ),
        positions=positions,
        position_units="cm",
        tracking=(
            "the head: the point the Neuralynx video tracker finds, from the "
            "headstage's LEDs"
        ),
        epochs=epochs,
        events=events,
        trials=Intervals.from_arrays(trials[:, 0], trials[:, 1], arm=arms, bad=is_bad),
        params={
            "bin_size": BIN_SIZE,
            "min_speed": MIN_SPEED,
            "dt": DT,
            "analysis_epochs": trials,
            "train_epochs": trials[::2],
            "test_epochs": trials[1::2],
            "restriction": keys["RestrictionType"],
            "session_type": keys["Session"],
            "layout": keys["Layout"],
            "in_paper": entry.session in PAPER_SESSIONS,
            "left_trajectory": left,
            "right_trajectory": right,
            "choice_point": choice,
            "track_width": width,
            "theta_channel": _channel_of(keys["goodTheta"][0]),
        },
        conversions=(
            Conversion(
                f"spike_times: the {n_units} .t and ._t files, big-endian uint32 "
                "tenths of a millisecond / 10,000, in seconds",
                "stated",
                "each file's header: 'Time of spiking stored in timestamps (tenths "
                "of msecs) as unsigned integer'",
            ),
            Conversion(
                "positions: the tracker's pixels / keys.m convFact "
                f"({factor[0]:g}, {factor[1]:g} pixels per cm along x, y)",
                "stated",
                "keys.m: 'x conversion factor and y conversion factor for loading "
                "position data in cm'; the authors' trajectories in pixels and in "
                f"cm differ by the same factors ({drawn[0]:g}, {drawn[1]:g})",
            ),
            Conversion(
                f"positions: the {int(np.sum(~tracked_mask(positions)))} frames with "
                "x or y 0 are NaN, the file's own mark for no detection",
                "stated",
                "LoadPos.m at the commit the paper pins: keep_idx = find(X ~= 0 & "
                "Y ~= 0), with removeZeros on by default",
            ),
            placement,
        ),
        watch_outs=(
            *WATCH_OUTS.get(entry.session, ()),
            *notes,
            *_watch_outs(
                entry,
                keys,
                track.times,
                positions,
                track.angle,
                distance > width / 2,
                units,
                graph,
            ),
        ),
        lfp=_lfp_source(entry, keys, folder),
    )
