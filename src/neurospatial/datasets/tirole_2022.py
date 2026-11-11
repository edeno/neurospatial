"""Tirole, Huelin Gorriz et al. (2022): rat CA1 on two novel linear tracks (Zenodo 7114733).

What was recorded: five male Lister-Hooded rats with 24-tetrode microdrives
in dorsal CA1 (two of them with 8 of the tetrodes in left V1 instead). Each
session started with 1 h in a 20 cm rest pot, then the rat ran back and forth
for about 15 min on each of two novel 2 m linear tracks (30 min on the second
in Rat 4 session 3), with 10 min in the rest pot between them (none in Rat 2
session 2), and ended with 2 h in the rest pot. In sessions of the paper's
second protocol (Rat 1 session 1, Rat 4 sessions 1 and 2, Rat 5) the rat ran
a third track first, which the paper removed, and the deposit does not hold.
The paper asks whether the change of a place cell's firing rate between the
two tracks comes back during replay.

Where it lives: Zenodo 7114733 (CC0), the mirror of Dryad
10.5061/dryad.ksn02v76h, whose download API needs a token. Per session,
``RATn_SESSm_extracted_clusters.mat`` (sorted single units),
``..._position.mat`` (tracking, the authors' linearization) and
``..._CSC.mat`` (four LFP channels at 1 kHz, 1.2 to 1.7 GB, fetched only with
``rec.lfp.fetch``). The CSC files of every session but RAT2_SESS1 are
MATLAB v7.3 (HDF5), read with ``h5py``.

Sources read: the paper (eLife 11:e79031, Methods and Table 1), the deposit's
README.txt, the files, and the authors' code
(github.com/bendor-lab/Elife_Tirole_Huelin_Gorriz_2022 at 44ecf42, CC0):
``extract_replay_events.m``, ``calculate_place_fields.m``, ``spike_count.m``,
``bayesian_decoding.m``, ``replay_decoding.m``, ``replay_scoring.m``,
``weighted_correlation.m``, ``run_shuffles.m``, ``replay_significance.m``,
``number_of_significant_replays.m``, ``list_of_parameters.m``.

**The deposit's labels are crossed.** Sessions are offered under their Table 1
identity, built from the file pair whose clocks match, which Table 1's lap
counts confirm (the authors' ``extract_laps`` counts one-way runs, half of
them match Table 1 in every session):

- ``rat1_session1``: spikes and position from ``RAT1_SESS2`` (byte-identical
  to ``RAT4_SESS1``'s), LFP from ``RAT1_SESS1``;
- ``rat4_session3``: spikes and position from ``RAT1_SESS1``, LFP from
  ``RAT4_SESS3``;
- the other seven sessions under their own labels.

Rat 4 session 1 is not offered: its spike and position files in the deposit
are the copies of Rat 1 session 1's, so only its LFP (``RAT4_SESS1_CSC``) is
its own, and nothing else of it can be loaded.

What the loader converts:

- positions: ``position.x`` and ``position.y`` (camera pixels, as stored, on the
  regular 25 Hz grid ``position.t``) times ``position.cm_per_pixel``, the
  README's "average cm/pixel value across tracks". The axes are the camera's:
  ``y`` runs down the image, so a plot with ``y`` up is mirrored. Each track's
  own scale, 200 cm over the length of the line the authors drew along it,
  differs from that average by up to 9%, so lengths on a track measured in
  ``positions`` are off by that much (a watch-out gives each track's).
- linearized: the authors' ``position.linear(k).linear``, 0 to 200 cm along
  track ``k`` at that track's own scale, NaN off the track; the tracks do not
  overlap in time, so one array holds all of them and the epochs say which.
- spike_times: ``clusters.spike_times`` of each ``spike_id``, seconds, on the
  tracking clock. ``units`` keeps the authors' ids (1 to N) as ``unit_id``, with
  the tetrode and cluster from ``id_conversion``.
- LFP: the four channels of the CSC file as stored, at 1 kHz on ``CSCtime``.
  No unit is stated (the README's "in Volts ?" does not fit values in the
  thousands), so the unit is arbitrary.

Epochs: ``pre`` from the start of the file to the first track, ``track_1`` and
``track_2`` (the first and last tracked time on each track, the authors'
``time_window``), ``between_tracks`` and ``post`` (to the end of the file, or
in Rat 2 session 1 to the re-exposure); ``rest_pot``, the runs of samples
the file marks in the rest pot (``position.sleepbox``), as stored. Rat 2
session 1 also ran both tracks again after its POST rest
(``track_1_reexposure``, ``track_2_reexposure``, ``post_reexposure``), which
Table 1 does not list. The paper's PRE and POST are its rest; the file does
not mark sleep, which the authors scored by hand-set thresholds not in the
deposit, so there are no sleep epochs.

The declared environment is the tracks as a graph: each track the line the
authors drew along it (``track_mask_x``, ``track_mask_y``, simplified to its
corners), in cm at the average scale, 2 cm bins. :func:`track_environment`
gives one track alone as the 200 cm line the authors decoded on.

Analysis choices (``params``):

- ``bin_size`` 2 cm, the paper's rate maps for display, and
  ``decoding_bin_size`` 10 cm, its rate maps for decoding;
- ``min_speed`` 5 cm/s and ``max_speed`` 50 cm/s, the speeds of the rate maps in
  the authors' code (``speed_threshold_laps``, ``speed_threshold_max``); the
  paper's text says 4 to 50 cm/s;
- ``dt`` 0.02 s and ``run_dt`` 0.25 s, the paper's decoding bins in replay and
  in running;
- ``tracks``, each track's 1-D environment from :func:`track_environment`;
  ``track_epochs``, the epoch name of each.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

from neurospatial.datasets._lfp import LFPSource
from neurospatial.datasets._recording import Conversion, Recording
from neurospatial.datasets._tracks import nearest_segment, placement_tolerance

if TYPE_CHECKING:
    from neurospatial import Environment
    from neurospatial.datasets._catalog import SessionEntry

__all__ = [
    "EXPECTED_WATCH_OUTS",
    "REQUIRES",
    "SESSIONS",
    "WATCH_OUTS",
    "WORKED_RESULT",
    "load_session",
    "read_lfp",
    "track_environment",
]

#: Modules of the ``datasets`` extra the loader imports (for the v7.3 LFP).
REQUIRES: tuple[str, ...] = ("h5py",)

TRACK_LENGTH = 200.0
BIN_SIZE = 2.0
DECODING_BIN_SIZE = 10.0
MIN_SPEED = 5.0
MAX_SPEED = 50.0
DT = 0.02
RUN_DT = 0.25
LFP_RATE = 1000.0
#: Pixels a track line is simplified by to its corners.
_SIMPLIFY = 1.5
#: The share of track samples the placement tolerance leaves outside the track.
TOLERANCE_QUANTILE = 0.99
#: The LFP channels of every CSC file, by the label the file gives each.
CHANNELS = {
    "best_ripple": "the channel the authors chose for the most ripple power "
    "(125-300 Hz, pwelch)",
    "best_theta": "the channel the authors chose for the most theta power (4-12 Hz)",
    "best_delta": "the channel the authors chose for the most delta power (1-4 Hz)",
    "best_theta_low_ripple": "the channel with the largest difference of "
    "normalized theta and ripple power, the authors' theta reference",
}


class SessionInfo(NamedTuple):
    """One Table 1 session: its deposit files and what Table 1 states.

    Attributes
    ----------
    spikes_file : str
        The prefix of the clusters and position files.
    lfp_file : str
        The prefix of the CSC file.
    laps : tuple of int
        Table 1's laps on track 1 and track 2 (round trips).
    rest_minutes : tuple of int
        Table 1's PRE and POST rest, minutes.
    stable_cells : int
        Table 1's stable cells in the regression analysis.
    removed_first_track : bool
        The paper's second protocol: a first track, removed from the study.
    """

    spikes_file: str
    lfp_file: str
    laps: tuple[int, int]
    rest_minutes: tuple[int, int]
    stable_cells: int
    removed_first_track: bool


#: The nine sessions the deposit holds, under their Table 1 identity.
SESSIONS: dict[str, SessionInfo] = {
    "rat1_session1": SessionInfo(
        "RAT1_SESS2", "RAT1_SESS1", (10, 12), (83, 155), 9, True
    ),
    "rat2_session1": SessionInfo(
        "RAT2_SESS1", "RAT2_SESS1", (15, 59), (59, 125), 29, False
    ),
    "rat2_session2": SessionInfo(
        "RAT2_SESS2", "RAT2_SESS2", (22, 33), (54, 104), 17, False
    ),
    "rat3_session1": SessionInfo(
        "RAT3_SESS1", "RAT3_SESS1", (34, 27), (49, 123), 25, False
    ),
    "rat3_session2": SessionInfo(
        "RAT3_SESS2", "RAT3_SESS2", (36, 38), (69, 137), 33, False
    ),
    "rat4_session2": SessionInfo(
        "RAT4_SESS2", "RAT4_SESS2", (13, 11), (98, 125), 35, True
    ),
    "rat4_session3": SessionInfo(
        "RAT1_SESS1", "RAT4_SESS3", (30, 20), (72, 76), 20, False
    ),
    "rat5_session1": SessionInfo(
        "RAT5_SESS1", "RAT5_SESS1", (13, 11), (91, 76), 20, True
    ),
    "rat5_session2": SessionInfo(
        "RAT5_SESS2", "RAT5_SESS2", (13, 19), (72, 86), 32, True
    ),
}

_PAPER = "Tirole et al. 2022"
_README = "the deposit's README.txt"

#: Not yet: the replay checks against the authors' own output are pinned in
#: tests/datasets/test_tirole_2022.py.
WORKED_RESULT: dict[str, Any] = {
    "status": "not yet",
    "level": None,
    "finding": (
        "the difference of a place cell's peak in-field rate between the tracks "
        "predicts the difference of its peak rate in local RUN replay"
    ),
    "quote": (
        '"contextual rate modulation was expressed across local awake replay '
        'events (Figure 3A; RUN: B = 0.05, F(1,268) = 186, p<0.001, R2 = 0.41)"'
    ),
    "source": f"{_PAPER}, eLife 11:e79031, Results and Figure 3A",
    "paper": 0.05,
    "ours": None,
    "criterion": (
        "fixed before the run: over the nine sessions, the least-squares slope "
        "of the replay difference on the place-field difference is within 0.01 "
        "of the paper's B = 0.05, R^2 within 0.1 of its 0.41, and p < 0.001"
    ),
    "parameters": None,
    "unmatched": (),
}

#: Findings computed from the data that each verified session must report.
EXPECTED_WATCH_OUTS: dict[str, tuple[str, ...]] = {
    "rat3_session2": (
        r"Each track's own scale",
        r"lie outside the declared environment",
    ),
}

#: Pitfalls of each session checked by hand, with their evidence.
WATCH_OUTS: dict[str, tuple[str, ...]] = {
    "rat3_session2": (
        "Checked by hand: spikes (436982.7 to 452736.0 s), tracking (436982.7 "
        "to 452735.9 s) and LFP share one clock; the authors' extract_laps, run "
        "on this file, finds 73 and 77 one-way runs, 36.5 and 38.5 laps against "
        "Table 1's 36 and 38; and PRE and POST last 69.3 and 137.3 min against "
        "Table 1's 69 and 137.",
    ),
}

_CROSSED = (
    "The deposit's file labels are crossed: this session's files were paired "
    "by their clocks, and Table 1's laps and rest durations confirm the "
    "pairing. Spikes and position come from {spikes}, the LFP from {lfp}."
)
_NO_MUA = (
    "The deposit holds sorted single units only: the paper's candidate events "
    "came from multi-unit activity that included the clusters it classified as "
    "multi-unit, which are not deposited. A multi-unit rate built from these "
    "units has fewer spikes than the paper's."
)
_NO_WAVEFORMS = (
    "No waveforms are deposited, so the paper's pyramidal-cell filter (half "
    "width at half maximum over 500 us) cannot be applied; the authors' code "
    "falls back to the mean-rate rule alone (under 5 Hz) without them."
)
_V1 = (
    "Two of the five rats had 8 of their 24 tetrodes in left primary visual "
    "cortex; neither the paper nor the files say which rats or tetrodes, so "
    "units['brain_region'] says dorsal CA1 for every unit."
)
_NO_SLEEP = (
    "No sleep epochs: the authors staged sleep with thresholds set by hand that "
    "the deposit does not record. epochs['rest_pot'] holds the samples the file "
    "marks in the rest pot, as stored, in runs broken by tracking gaps."
)
_OFF_TRACK = (
    "The environment is the tracks alone: the rest pot, where the rat spends "
    "most of the session, and the third track of the second protocol lie off "
    "it, so most samples fall in no bin by design. The placement tolerance is "
    "set from the samples on the tracks (epochs track_1 and track_2)."
)
_LFP_UNIT = (
    "The README calls the LFP 'in Volts ?'; the values run to the thousands, so "
    "no unit is known and rec.lfp's unit is arbitrary."
)


def _read_mat(path: Path, name: str) -> Any:
    """One variable of a MATLAB v5/v7 file: a struct as a dict, a struct array as
    a list of dicts; nothing is executed."""
    from scipy.io import loadmat

    return loadmat(path, variable_names=[name], simplify_cells=True)[name]


def _tracks(position: dict[str, Any]) -> list[dict[str, Any]]:
    """``position.linear``: one dict per track, a list even for one track."""
    linear = position["linear"]
    return list(linear) if isinstance(linear, (list, tuple, np.ndarray)) else [linear]


def _track_names(n_tracks: int) -> list[str]:
    """Epoch names of the tracks in file order; tracks 3 and 4 are re-exposures."""
    names = ["track_1", "track_2", "track_1_reexposure", "track_2_reexposure"]
    if n_tracks not in (2, 4):
        msg = f"the position file holds {n_tracks} tracks, not 2 or 4"
        raise ValueError(msg)
    return names[:n_tracks]


def _runs(mask: NDArray[np.bool_], times: NDArray[np.float64]) -> NDArray[np.float64]:
    """Each run of True samples, from its first sample to the next sample's time.

    A sample stands for its frame, up to the next sample, so a run of one sample
    lasts one frame; a run at the end of the file ends one median frame after
    its last sample. Shape (n_runs, 2).
    """
    edges = np.diff(np.r_[0, mask.astype(np.int8), 0])
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    after = np.r_[times, times[-1] + np.median(np.diff(times))]
    return np.column_stack([times[starts], after[stops]])


def _epochs(
    times: NDArray[np.float64],
    windows: list[tuple[float, float]],
    in_pot: NDArray[np.bool_],
) -> dict[str, NDArray[np.float64]]:
    """PRE, the tracks, between them, POST and the rest-pot runs."""
    names = _track_names(len(windows))
    epochs = {
        name: np.array([window]) for name, window in zip(names, windows, strict=True)
    }
    start, end = float(times[0]), float(times[-1])
    epochs["pre"] = np.array([[start, windows[0][0]]])
    epochs["between_tracks"] = np.array([[windows[0][1], windows[1][0]]])
    post_end = windows[2][0] if len(windows) == 4 else end
    epochs["post"] = np.array([[windows[1][1], post_end]])
    if len(windows) == 4:
        epochs["post_reexposure"] = np.array([[windows[3][1], end]])
    epochs["rest_pot"] = _runs(in_pot, times)
    return epochs


def _track_lines(
    tracks: list[dict[str, Any]], scale: float
) -> list[NDArray[np.float64]]:
    """Each track's drawn line simplified to its corners, cm, shape (n_points, 2)."""
    import shapely

    lines = []
    for track in tracks:
        points = np.column_stack([track["track_mask_x"], track["track_mask_y"]]).astype(
            np.float64
        )
        corners = shapely.simplify(shapely.LineString(points), _SIMPLIFY)
        lines.append(np.asarray(corners.coords, dtype=np.float64) * scale)
    return lines


def _declared_tracks(
    lines: list[NDArray[np.float64]],
    positions: NDArray[np.float64],
    on_track: NDArray[np.bool_],
    session: str,
) -> tuple[Environment, Conversion, float | None]:
    """The tracks of the first exposure as one graph, placed by the drawn lines."""
    import networkx as nx

    from neurospatial import Environment

    graph = nx.Graph()
    order = []
    node = 0
    starts, ends = [], []
    for line in lines[:2]:
        first = node
        for point in line:
            graph.add_node(node, pos=tuple(point.tolist()))
            node += 1
        for a in range(first, node - 1):
            length = float(np.linalg.norm(line[a - first + 1] - line[a - first]))
            graph.add_edge(a, a + 1, distance=length)
            order.append((a, a + 1))
            starts.append(line[a - first])
            ends.append(line[a - first + 1])
    distance = nearest_segment(
        positions[on_track], np.array(starts), np.array(ends)
    ).distance
    width, quantile, evidence = placement_tolerance(
        distance,
        TOLERANCE_QUANTILE,
        floor=BIN_SIZE,
        measured="the tracked point's distance from the drawn lines while the rat "
        "was on a track",
        reason="No track width is stated; the head swings at the ends, where the "
        "rat drinks, so the tolerance is taken from the samples.",
    )
    env = Environment.from_graph(
        graph,
        order,
        bin_size=BIN_SIZE,
        units="cm",
        name=f"tirole_2022 {session}",
        track_width=width,
    )
    placement = Conversion(
        f"environment: track_width {width:.3g} cm, a placement tolerance",
        "inferred",
        evidence,
    )
    return env, placement, quantile


def track_environment(*, bin_size: float = DECODING_BIN_SIZE) -> Environment:
    """One track as the authors decoded it: a 200 cm line of ``bin_size`` bins.

    Positions on it are ``(rec.linearized, 0)`` while the rat is on that track
    (``rec.params["track_epochs"]``). The authors' code bins the linearized
    position from 0 to 200 cm (``calculate_place_fields.m``).

    Parameters
    ----------
    bin_size : float, default 10.0
        Bin length, cm; the paper decodes on 10 cm bins.

    Returns
    -------
    Environment

    Examples
    --------
    >>> from neurospatial.datasets.tirole_2022 import track_environment
    >>> track_environment().n_bins
    20
    """
    from neurospatial import Environment

    return Environment.linear_track(
        endpoints=[[0.0, 0.0], [TRACK_LENGTH, 0.0]],
        bin_size=bin_size,
        units="cm",
        track_width=bin_size,
        name="tirole_2022 track",
    )


def read_lfp(
    sources: tuple[str | Path, ...], channels: tuple[str, ...]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Read channels of a CSC file by their labels.

    A MATLAB v7.3 file (HDF5) is read with ``h5py``: ``CSC`` is a struct array
    whose fields are datasets of references, each to an array stored
    transposed, with text as ``uint16`` characters. A v7 file (RAT2_SESS1) is
    read with ``scipy.io.loadmat``.

    Parameters
    ----------
    sources : tuple of path
        The CSC file.
    channels : tuple of str
        Labels from :data:`CHANNELS`.

    Returns
    -------
    samples : ndarray, shape (n_time, n_channels)
    times : ndarray, shape (n_time,)
        ``CSCtime`` of the first channel asked for, seconds.

    Raises
    ------
    ValueError
        If a label is not in the file, or the channels' times differ.
    """
    import h5py

    path = Path(sources[0])
    columns: dict[str, NDArray[np.float64]] = {}
    clocks: dict[str, NDArray[np.float64]] = {}
    if h5py.is_hdf5(path):
        with h5py.File(path, "r") as file:
            struct = file["CSC"]

            def field(name: str, i: int) -> Any:
                return file[struct[name][()].ravel(order="F")[i]]

            n_channels = struct["channel_label"].size
            for i in range(n_channels):
                raw = np.asarray(field("channel_label", i)[()]).ravel(order="F")
                label = "".join(map(chr, raw))
                if label in channels:
                    columns[label] = np.asarray(field("CSCraw", i)[()], float).ravel()
                    clocks[label] = np.asarray(field("CSCtime", i)[()], float).ravel()
    else:
        struct = _read_mat(path, "CSC")
        for item in [struct] if isinstance(struct, dict) else struct:
            if item["channel_label"] in channels:
                columns[item["channel_label"]] = np.asarray(
                    item["CSCraw"], float
                ).ravel()
                clocks[item["channel_label"]] = np.asarray(
                    item["CSCtime"], float
                ).ravel()
    if missing := [name for name in channels if name not in columns]:
        msg = f"{path.name} holds no channel labelled {missing}"
        raise ValueError(msg)
    times = clocks[channels[0]]
    if any(not np.array_equal(clocks[name], times) for name in channels[1:]):
        msg = f"{path.name}: the channels {channels} have different times"
        raise ValueError(msg)
    return np.column_stack([columns[name] for name in channels]), times


def _lfp_source(entry: SessionEntry, folder: Path) -> LFPSource:
    """The session's CSC file: four channels, the authors' ripple channel first."""
    return LFPSource(
        dataset=entry.info.dataset,
        files=entry.lfp_files,
        sampling_rate=LFP_RATE,
        channels=CHANNELS,
        unit="arbitrary",
        unit_basis=Conversion(
            "LFP: CSCraw as stored; no unit is known",
            "inferred",
            f"{_README} says 'downsampled signal values (in Volts ?) - without any "
            "further filtering', and the values run to the thousands",
        ),
        times_basis=Conversion(
            "LFP times: CSCtime as stored, s, the clock of the spikes",
            "stated",
            f"{_README}: 'timestamps (in seconds) of each point in CSCraw'; the "
            "1 kHz rate is the file's SR",
        ),
        reader=functools.partial(read_lfp),
        ripple_channels=("best_ripple",),
        cache=folder.parents[1],
    )


def _scale_watch_out(tracks: list[dict[str, Any]], scale: float) -> str:
    """Each track's own cm per pixel against the average the positions use."""
    names = _track_names(len(tracks))
    listed = ", ".join(
        f"{name} {track['cm_per_pixel']:.4f} ({track['cm_per_pixel'] / scale - 1:+.1%})"
        for name, track in zip(names, tracks, strict=True)
    )
    return (
        f"Each track's own scale, 200 cm over the length of the line the authors "
        f"drew along it, differs from the average {scale:.4f} cm per pixel that "
        f"positions use: {listed}. The authors set each from its line and the "
        "2 m length, so a length measured in positions on a track is off by that "
        "share; rec.linearized is at each track's own scale, as the paper's rate "
        "maps are."
    )


def _rest_watch_out(
    epochs: dict[str, NDArray[np.float64]], info: SessionInfo
) -> tuple[str, ...]:
    """A watch-out where PRE or POST differs from Table 1 by more than a minute."""
    pre, post = (float(np.diff(epochs[name])[0, 0]) / 60 for name in ("pre", "post"))
    stated = info.rest_minutes
    if abs(pre - stated[0]) <= 1 and abs(post - stated[1]) <= 1:
        return ()
    swapped = abs(pre - stated[1]) <= 1 and abs(post - stated[0]) <= 1
    return (
        f"PRE and POST last {pre:.1f} and {post:.1f} min here, against Table 1's "
        f"{stated[0]} and {stated[1]} min"
        + (": Table 1 gives them the other way round." if swapped else "."),
    )


def load_session(
    entry: SessionEntry, path_or_stream: str | os.PathLike[str]
) -> Recording:
    """Read one session's clusters and position files into a :class:`Recording`.

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
    """
    import pandas as pd

    folder = Path(path_or_stream)
    info = SESSIONS[entry.session]
    position = _read_mat(
        folder / f"{info.spikes_file}_extracted_position.mat", "position"
    )
    clusters = _read_mat(
        folder / f"{info.spikes_file}_extracted_clusters.mat", "clusters"
    )
    times = np.asarray(position["t"], dtype=np.float64)
    scale = float(position["cm_per_pixel"])
    pixels = np.column_stack([position["x"], position["y"]]).astype(np.float64)
    positions = pixels * scale
    tracks = _tracks(position)
    names = _track_names(len(tracks))
    linear = np.stack([np.asarray(t["linear"], dtype=np.float64) for t in tracks])
    on_track = np.isfinite(linear)
    if np.any(on_track.sum(axis=0) > 1):
        msg = f"{entry.session}: two tracks hold a linear position at one time"
        raise ValueError(msg)
    linearized = np.where(
        on_track.any(axis=0), np.nansum(np.where(on_track, linear, 0.0), axis=0), np.nan
    )
    windows = [(float(times[row].min()), float(times[row].max())) for row in on_track]
    in_pot = np.isfinite(np.asarray(position["sleepbox"], dtype=np.float64))
    epochs = _epochs(times, windows, in_pot)
    ids = np.asarray(clusters["spike_id"]).astype(np.int64)
    spikes = np.asarray(clusters["spike_times"], dtype=np.float64)
    conversion = np.atleast_2d(np.asarray(clusters["id_conversion"])).astype(np.int64)
    order = np.argsort(ids, kind="stable")
    bounds = (
        np.searchsorted(ids[order], conversion[:, 0], side="left"),
        np.searchsorted(ids[order], conversion[:, 0], side="right"),
    )
    spike_times = tuple(
        np.sort(spikes[order][a:b]) for a, b in zip(*bounds, strict=True)
    )
    units = pd.DataFrame(
        {
            "brain_region": "dorsal CA1",
            "label": None,
            "tetrode": conversion[:, 1] // 1000,
            "cluster": conversion[:, 1] % 1000,
        },
        index=pd.Index(conversion[:, 0], name="unit_id"),
    )
    lines = _track_lines(tracks, scale)
    env, placement, quantile = _declared_tracks(
        lines, positions, np.any(on_track[:2], axis=0), entry.session
    )
    lfp = _lfp_source(entry, folder)
    rat = entry.session.removeprefix("rat").split("_session")[0]
    crossed = (
        (_CROSSED.format(spikes=info.spikes_file, lfp=info.lfp_file),)
        if info.spikes_file != info.lfp_file
        else ()
    )
    reexposure = (
        (
            "This session's file goes on after its POST rest: the rat ran both "
            "tracks again (track_1_reexposure, track_2_reexposure), which Table 1 "
            "does not list; epochs['post'] ends where it starts. The LFP file "
            "ends before it.",
        )
        if len(tracks) == 4
        else ()
    )
    removed = (
        (
            "The rat ran a third track first in this session, which the paper "
            "removed; the deposit does not hold it. epochs['pre'] (Table 1's PRE) "
            "spans it, so the rat is not in the rest pot all of PRE: "
            "epochs['rest_pot'] says where it is.",
        )
        if info.removed_first_track
        else ()
    )
    return Recording(
        **entry.identity(path_or_stream),
        animal=f"rat {rat} (male Lister-Hooded)",
        apparatus=entry.info.apparatus,
        brain_regions=entry.info.brain_regions,
        clock=(
            "the acquisition system's clock, seconds, as stored; tracking starts "
            f"at {times[0]:.1f} s"
        ),
        times=times,
        spike_times=spike_times,
        units=units,
        env=env,
        env_kind="declared",
        track_width_quantile=quantile,
        env_source=(
            "the two 2 m linear tracks, each the line the authors drew along it "
            "(track_mask_x, track_mask_y) simplified to its corners, at the "
            f"average scale; {BIN_SIZE:g} cm bins"
        ),
        positions=positions,
        position_units="cm",
        tracking="the head, by an overhead camera at 25 Hz",
        epochs=epochs,
        linearized=linearized,
        params={
            "bin_size": BIN_SIZE,
            "decoding_bin_size": DECODING_BIN_SIZE,
            "min_speed": MIN_SPEED,
            "max_speed": MAX_SPEED,
            "dt": DT,
            "run_dt": RUN_DT,
            "tracks": {name: track_environment() for name in names},
            "track_epochs": tuple(names),
        },
        conversions=(
            Conversion(
                f"positions: (x, y) in camera pixels x {scale:.6g}, the file's "
                "position.cm_per_pixel, cm, on the camera's axes",
                "stated",
                f"{_README}: x and y 'in pixels', cm_per_pixel the 'average "
                "cm/pixel value across tracks'",
            ),
            Conversion(
                "linearized: each track's position.linear(k).linear, cm, NaN off "
                "the tracks",
                "stated",
                f"{_README}: track length 2 m and cm_per_pixel 'based on track "
                "length and track mask'; the values run from 0 to 200",
            ),
            Conversion(
                f"spike_times: clusters.spike_times by spike_id, s, sorted, "
                f"{len(spike_times)} units",
                "stated",
                f"{_README}: 'spike timestamps' with 'single unit id' 1 to N, and "
                "id_conversion's '1000 * tet number + single unit id'",
            ),
            placement,
            lfp.unit_basis,
            lfp.times_basis,
        ),
        watch_outs=(
            *WATCH_OUTS.get(entry.session, ()),
            *crossed,
            *_rest_watch_out(epochs, info),
            _scale_watch_out(tracks, scale),
            _OFF_TRACK,
            *reexposure,
            *removed,
            _NO_MUA,
            _NO_WAVEFORMS,
            _V1,
            _NO_SLEEP,
            _LFP_UNIT,
        ),
        lfp=lfp,
    )
