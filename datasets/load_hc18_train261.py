"""Drieu, Todorova & Zugaro rat CA1 units while carried on a model train (CRCNS hc-18, session Train-261-20140617).

What was recorded: rat 261, tetrodes and octrodes in dorsal CA1 of both
hemispheres, over 5.5 h: sleep, three short travel sessions sitting on a
treadmill carried by a model train with the treadmill off (passive), sleep,
two with the treadmill on (the rat walks: "pseudo-active"), sleep, two
passive, sleep. One head LED tracked by video at 39.06 Hz in a 768 x 576
image.

Where it lives: CRCNS hc-18 (doi:10.6080/K0Z899MM), which needs a free
crcns.org account (``CRCNS_USERNAME``/``CRCNS_PASSWORD``). Each session is
one ``data/<session>.tar.gz`` in Neuroscope format (this one 6.5 GB, almost
all LFP and waveforms); the loader streams it once and keeps ``.res.N``,
``.clu.N``, ``.pos``, ``.cat.evt`` and ``.xml`` (134.8 MB).

What the loader converts:

- ``.res.N`` holds spike times in samples of a 32,552.083 Hz clock (the
  ``.xml``), not the 20 kHz of hc-2/hc-3; divided by it, they are seconds.
  Every cluster from 2 up is a unit (0 artefact, 1 unsortable).
- ``.pos`` row k is the video frame at k / 39.0625 s (the ``.xml`` video
  rate); ``-1`` marks an untracked frame and becomes NaN. Positions stay in
  video pixels: the data set states no scale.
- ``.cat.evt`` gives each subsession's start and end in ms; they become
  epochs ``travel`` (the seven travel sessions), ``passive``,
  ``pseudo_active`` and ``sleep``.

Analysis choices (``params``): maps, decoding and laps use the travel
epochs only (``analysis_epochs``); no speed threshold, because the train,
not the rat, moves the position; ``bin_size`` 5 px; ``dt`` 0.25 s with
alternating 60 s train/test blocks inside the travel epochs.

Declared environment (``geometry["environment"]``): the train's circuit as
a closed loop (a cyclic graph, every edge kept) through the travel samples'
median position in each of 72 angular sectors; the data set states no shape
or size, so the loop is the samples' own and the width is measured from
them. ``geometry["lap_start_region"]`` cuts the loop at its westmost point,
an arbitrary cut: a lap count does not depend on where a loop is cut, up to
one lap.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from _formats import neuroscope_rates, read_res_clu, read_whl
from _record import (
    Range,
    Recording,
    alternating_blocks,
    check_expectations,
    checked_span,
    intersect_intervals,
    validate_recording,
)
from _tracks import (
    cut_region,
    declared_geometry,
    loop_centreline,
    polyline_graph,
    with_declared,
)
from load_hc3_ec013_669 import fetch_session

DATASET = {
    "key": "hc18_train261",
    "title": "Drieu, Todorova & Zugaro (2018), rat 261, CA1 units on a treadmill carried by a model train (CRCNS hc-18)",
    "source": "https://doi.org/10.6080/K0Z899MM",
    "license": "CRCNS terms of use (account required; cite Drieu et al. 2018 and the data set)",
    "download_size": "134.8 MB (members of a 6.5 GB archive)",
}

# Pre-registered from the hc-18 description (three travel sessions, passive,
# active, passive, 2 x ~15 laps each, interleaved with ~90 min of sleep;
# Neuroscope format) and the session's .xml and .cat.evt (video 768 x 576 at
# 39.0625 Hz; 12 subsessions), before the first run of this loader. The
# .pos values are expected to be video pixels of one LED.
EXPECTATIONS: dict[str, Any] = {
    "n_dims": 2,
    "n_units": Range(20, 300),
    "sampling_rate_hz": Range(39.0, 39.1),
    "positions_inside_the_video": lambda rec: bool(
        np.nanmin(rec.positions) >= 0
        and np.nanmax(rec.positions[:, 0]) <= 768
        and np.nanmax(rec.positions[:, 1]) <= 576
    ),
    "travel_epochs": lambda rec: len(rec.epochs.get("travel", ())) == 7,
}


SUBDIR = "crcns_hc18"
ARCHIVE = "hc-18/data/Train-261-20140617.tar.gz"
BASE = "Train-261-20140617"
N_SHANKS = 8
_LOOP_POINTS = 72
_SHA256 = {
    "clu.1": "a6def5c7f5ccc6edf44069e6d3e95776fa1ace0001f2fc66340611a485c22509",
    "clu.2": "664fbaa3f28c52e5de4a1e9237881d2808476241918374b348a167194b57aaca",
    "clu.3": "5e4a1d6eda87336a8bf1828b06007c248be72c1bbe0ac7d52e2b2f3fa85e2b69",
    "clu.4": "d38546deb426ddf85e196b186cadbb5d94dc0c1445f20058c21660a41324ccf2",
    "clu.5": "3097fd23e3986982ca2067c1905a9adabb279c5acc4a06de7093ea6937e97cb5",
    "clu.6": "035a20bd9911b7bf1d1667b973529bdacc9f34a89c834e217d8e660d263a9031",
    "clu.7": "1557b6982bbe2ab43a82814835f40222599859516ca8816284a3b8afd5a871a2",
    "clu.8": "9ec8bac799be3e9b8f504d950269eda62aadc33a23a4d46d09ef916550544ba3",
    "res.1": "94b094a8756a5d4df1e14fc9a8bb12d26d91b2a9a7b989bc869cb11dea1e6d57",
    "res.2": "ad1911e6630a4817b277212f288f1acdcb62f78ab4cbe263cfb86cc12d258569",
    "res.3": "935642d45d07d8f5ada4e15ee7dfe776cd0ca6e9f3c33567230419b0a019fe7e",
    "res.4": "5b6da9540ca4a01fe56af44d00e716ce12de6a834d5b51a3d869b1991ae1f02c",
    "res.5": "08e6a0dc68f6d135015f6518d2a7cea9be98efa418a84ccc32bd2121ba73a033",
    "res.6": "d2627768f54014e98e92c438f6f6150a9c0a2faef723d3ce71f855f90371a0a0",
    "res.7": "de6ac867a6d3b77c946544ce7cc47bd6b38ffa511b52aa326c903d4a22aae27f",
    "res.8": "3d9be6a52991ecaf665d13792dd6df632d93a46495f6ecd48402d6eb783639e1",
    "pos": "ec6fae9c307ba3039106523095e8cf7548e943595f817f36a4bc0811ce996740",
    "cat.evt": "113bb3f84b05583aba289f050ce2c95264c4eb9b6703686739b50383fc3a6ddd",
    "xml": "de4e8a8fe4190212bd2dbef37a2f17d34a2a384dd5e2fe9f931f9f34d67aa277",
}


def _by_start(table: np.ndarray) -> np.ndarray:
    return table[np.argsort(table[:, 0])]


def read_subsessions(path: Path) -> dict[str, np.ndarray]:
    """``.cat.evt`` "beginning of"/"end of" pairs (ms) as [start, stop) s per kind."""
    starts: dict[str, float] = {}
    kinds: dict[str, list[tuple[float, float]]] = {}
    for line in path.read_text().splitlines():
        time_ms, text = line.split(maxsplit=1)
        edge, name = text.split(" of ", 1)
        if edge == "beginning":
            starts[name] = float(time_ms) / 1000.0
        else:
            kind = name.rsplit("-", 1)[1]
            kinds.setdefault(kind, []).append(
                (starts.pop(name), float(time_ms) / 1000.0)
            )
    return {kind: np.array(rows) for kind, rows in kinds.items()}


def load(cache: Path | None = None) -> Recording:
    """Fetch the session; one-LED positions in pixels, every sorted cluster, subsessions."""
    files = fetch_session(ARCHIVE, BASE, BASE, _SHA256, subdir=SUBDIR, cache=cache)
    rates = neuroscope_rates(files["xml"])
    wideband, video = rates["wideband_hz"], rates["video_hz"]
    frames = read_whl(files["pos"])
    times = np.arange(len(frames)) / video
    positions = frames[:, :2]
    span = checked_span(positions, 400.0, 768.0, units="pixel")
    kinds = read_subsessions(files["cat.evt"])
    travel_kinds = ("PassiveForward", "PseudoActiveForward")
    travel = _by_start(np.vstack([kinds[k] for k in travel_kinds]))
    sleep = _by_start(np.vstack([v for k, v in kinds.items() if k.startswith("Sleep")]))
    end = max(float(times[-1]), float(np.max(sleep)))
    trains, ids, found_last = [], [], 0.0
    for shank in range(1, N_SHANKS + 1):
        samples, clusters = read_res_clu(files[f"res.{shank}"], files[f"clu.{shank}"])
        found_last = max(found_last, float(samples.max()) / wideband)
        for cluster in np.unique(clusters[clusters >= 2]):
            trains.append(np.sort(samples[clusters == cluster] / wideband))
            ids.append(f"{shank}.{cluster}")
    if not 0.99 * end <= found_last <= 1.01 * end:
        msg = f"last spike at {found_last:.0f} s but the session ends at {end:.0f} s"
        raise ValueError(msg)
    blocks = np.vstack(alternating_blocks(travel[0, 0], travel[-1, 1], 60.0))
    blocks = blocks[np.argsort(blocks[:, 0])]
    in_travel = intersect_intervals(blocks, travel)
    block_index = np.searchsorted(blocks[:, 0], in_travel[:, 0], side="right") - 1
    rec = Recording(
        dataset=DATASET["key"],
        session=BASE,
        source=f"{DATASET['source']} {ARCHIVE}",
        license=DATASET["license"],
        times=times,
        positions=positions,
        position_units="pixel",
        spike_times=tuple(trains),
        unit_ids=np.asarray(ids),
        epochs={
            "travel": travel,
            "passive": kinds["PassiveForward"],
            "pseudo_active": kinds["PseudoActiveForward"],
            "sleep": sleep,
        },
        geometry={"apparatus": "model-train circuit", "video_px": (768, 576)},
        params={
            "bin_size": 5.0,
            "min_speed": None,
            "dt": 0.25,
            "analysis_epochs": travel,
            "train_epochs": in_travel[block_index % 2 == 0],
            "test_epochs": in_travel[block_index % 2 == 1],
        },
        conversions=(
            f"times: .pos row k at k / {video:g} s (video rate in the .xml);"
            f" {len(frames)} frames, {times[-1]:.0f} s",
            f"positions: one LED, video pixels as stored, -1 -> NaN"
            f" ({np.isnan(positions).any(axis=1).mean():.2%}); {span}",
            f"spikes: .res samples / {wideband:g} Hz (the .xml clock); {len(trains)}"
            f" clusters >= 2 on {N_SHANKS} shanks; last spike {found_last:.0f} s",
            "epochs: .cat.evt ms / 1000; travel = PassiveForward + PseudoActiveForward"
            f" ({len(travel)}, {np.sum(np.diff(travel, axis=1)):.0f} s)",
        ),
    )
    in_travel_samples = np.zeros(len(times), dtype=bool)
    for start, stop in travel:
        in_travel_samples |= (times >= start) & (times < stop)
    travel_xy = np.where(in_travel_samples[:, None], positions, np.nan)
    centreline = loop_centreline(travel_xy, n_points=_LOOP_POINTS)
    graph = polyline_graph(centreline, closed=True)
    west = int(np.argmin(centreline[:, 0]))
    region = cut_region(
        "west_cut",
        centreline[west],
        centreline[(west + 1) % _LOOP_POINTS] - centreline[west - 1],
        half_length=5.0,
        half_width=40.0,
    )
    entries, line = declared_geometry(
        graph,
        travel_xy,
        kind=f"closed circuit as a {_LOOP_POINTS}-segment loop",
        source="hc-18: a model train on a closed circuit (no shape or size stated)",
        bin_size=5.0,
        track_width=None,
        units="pixel",
        name=DATASET["key"],
        placement="the travel samples' median radius in each of"
        f" {_LOOP_POINTS} equal angular sectors around their centre; lap cut"
        " (geometry['lap_start_region']) at the loop's westmost point, a stated"
        " cut, not a start the source names",
    )
    return with_declared(rec, {**entries, "lap_start_region": region}, line)


def clock_assumed_20khz(rec: Recording) -> Recording:
    # A reader using hc-2/hc-3's 20 kHz for .res samples of a 32,552 Hz clock.
    assert rec.spike_times is not None
    scale = 32552.083 / 20000.0
    return replace(rec, spike_times=tuple(t * scale for t in rec.spike_times))


def pos_minus_one_kept(rec: Recording) -> Recording:
    # A reader that keeps the .pos -1 marker as a coordinate.
    assert rec.positions is not None
    return replace(rec, positions=np.nan_to_num(rec.positions, nan=-1.0))


HAZARDS = {
    "clock_assumed_20khz": clock_assumed_20khz,
    "pos_minus_one_kept": pos_minus_one_kept,
}


if __name__ == "__main__":
    rec = load()
    validate_recording(rec)
    print(
        f"{rec.dataset}: {len(rec.spike_times or ())} units, {rec.times[-1] - rec.times[0]:.0f} s"
    )
    for row in check_expectations(rec, EXPECTATIONS):
        print(f"  {'pass' if row['passed'] else 'FAIL'} {row['key']}: {row['detail']}")
    for line in rec.conversions:
        print(f"  conversion: {line}")
