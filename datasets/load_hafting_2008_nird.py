"""Hafting et al. 2008 rat MEC grid cells on a linear track (NIRD 10.11582/2017.00020).

What was recorded: rat 11265, two 10-minute sessions (11 and 12) on
2006-03-16 joined into one 20-minute recording, on a linear track (the readme:
"mostly 320 cm"), with three medial entorhinal units on tetrode 4. One
tracking LED at 50 Hz, smoothed by the authors with a moving mean.

Where it lives: NIRD Research Data Archive, DOI 10.11582/2017.00020
("Grid cell data; Hafting et al 2008"), CC BY 4.0 per the archive's
metadata, file ``hafting_et_al_2008.zip`` (20.9 MB) served from NIRD's S3
store; four members are read by HTTP byte range: the session's ``_POS.mat``
and the units ``_t4c1``, ``_t4c2``, ``_t4c4``. Each is MATLAB v5 with plain
arrays (``posx``, ``posy``, ``post``; ``ts``).

What the loader converts:

- ``posx``/``posy`` have no stated unit. ``posx`` spans exactly −160 to 160
  and ``posy`` exactly ±18.1 (both symmetric about 0): the authors centred
  the track and scaled it to its 320 cm length, so the values are cm. The
  span is range checked against 300-340 cm.
- ``post`` and ``ts`` are seconds from the start of the recording; kept.
- The sessions are joined with no gap on the clock: between 599.98 and
  600.00 s the LED moves 282 cm, from where session 11 ended to where 12
  began. Each session is an epoch, and ``analysis_epochs`` are the two
  sessions without the joining interval.

Analysis choices (``params``): ``bin_size`` 4 cm; ``min_speed`` 5 cm/s
(running); ``dt`` 0.25 s and alternating 60 s train/test blocks. The track
is also given as a one-edge graph from (−160, 0) to (160, 0) cm; its width
is not stated. That graph is the declared environment
(``geometry["environment"]``): the archive's ``readme_hafting2008.txt``
(read from the zip) states "a linear track (mostly 320 cm)", and the line is
drawn in the file's own frame, not given by the authors; the fit is in
``conversions``. Nothing independent of the samples checks it: the samples
span the stated 320 cm because the authors scaled the frame to it, and the
unit is read from that span.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from _record import (
    Range,
    Recording,
    alternating_blocks,
    check_expectations,
    checked_span,
    validate_recording,
)
from _tracks import declared_geometry

from neurospatial.datasets._fetch import fetch_zip_member

DATASET = {
    "key": "hafting_2008_nird",
    "title": "Hafting et al. (2008), rat 11265, sessions 16030611+12, MEC units on a 320 cm linear track",
    "source": "https://doi.org/10.11582/2017.00020",
    "license": "CC-BY-4.0",
    "download_size": "1.0 MB of a 20.9 MB zip (byte-range reads)",
}

# Pre-registered from the readme (mostly 320 cm linear tracks, 10 min
# sessions joined with "+") and the zip listing (three cells t4c1, t4c2,
# t4c4) before the first run of this loader.
EXPECTATIONS: dict[str, Any] = {
    "n_units": 3,
    "n_dims": 2,
    "sampling_rate_hz": Range(49.0, 51.0),
    "duration_s": Range(1190.0, 1210.0),
    # No extent expectation: the loader refuses a span outside 300-340 cm
    # (checked_span), so one here could never fail; the span is in
    # ``conversions``.
}

_ZIP = (
    "https://data.archive.sigma2.no/dataset/0c686c74-416d-470d-a9cd-265840d86ebc"
    "/download/hafting_et_al_2008.zip"
)
_SESSION = "11265-16030611+12"
_FILES = {
    "POS": "49d93996578e40ff0cdb9dbc2c40d45513f68fd69a585db40c5eec56bb75682a",
    "t4c1": "42e6e6df9456f44bf299c1fd8849fee385c310f2b7d85cf6a37c1235c13bc479",
    "t4c2": "72fd5fdb133b2323b6b1d31dbc714f96ae0026e86b00339ed5b31443ca27b21c",
    "t4c4": "19b538298de1363453afa3f997e053acd06fa2198bcea0482c55b356a45ac6ba",
}
_SPAN_CM = (300.0, 340.0)
_HALF_LENGTH_CM = 160.0
# A step longer than this between consecutive 50 Hz samples is not movement.
_JOIN_MIN_CM = 100.0


def _read(kind: str, cache: Path | None) -> dict[str, Any]:
    from scipy.io import loadmat

    path = fetch_zip_member(
        _ZIP,
        f"{_SESSION}_{kind}.mat",
        sha256=_FILES[kind],
        subdir=DATASET["key"],
        cache=cache,
    )
    return loadmat(path, squeeze_me=True)


def track_graph() -> Any:
    import networkx as nx

    graph = nx.Graph()
    graph.add_node("west", pos=(-_HALF_LENGTH_CM, 0.0))
    graph.add_node("east", pos=(_HALF_LENGTH_CM, 0.0))
    graph.add_edge("west", "east", distance=2 * _HALF_LENGTH_CM)
    return graph


def load(cache: Path | None = None) -> Recording:
    """Fetch the session's position and three units; positions in cm."""
    pos = _read("POS", cache)
    times = np.asarray(pos["post"], dtype=float)
    positions = np.column_stack([pos["posx"], pos["posy"]]).astype(float)
    span = checked_span(positions, *_SPAN_CM)
    # The two sessions are joined with no gap on the clock: the one step
    # between them is the 2.8 m jump from where session 11 ended to where 12
    # began. Each session is its own epoch, and the joining interval is in
    # neither.
    step = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    join = int(np.argmax(step))
    if step[join] < _JOIN_MIN_CM or join + 1 != len(times) // 2:
        msg = (
            f"largest step {step[join]:.0f} cm after sample {join} of {len(times)};"
            " expected one jump between two equal sessions, recheck"
        )
        raise ValueError(msg)
    sessions = np.array([[times[0], times[join]], [times[join + 1], times[-1]]])
    cells = [name for name in _FILES if name != "POS"]
    spikes = tuple(
        np.sort(np.asarray(_read(c, cache)["ts"], float).ravel()) for c in cells
    )
    train, test = alternating_blocks(times[0], times[-1], 60.0)
    graph = track_graph()
    declared, declared_line = declared_geometry(
        graph,
        positions,
        times=times,
        kind=f"straight track {2 * _HALF_LENGTH_CM:g} cm",
        source="readme_hafting2008.txt: 'a linear track (mostly 320 cm)'; width"
        " not stated",
        bin_size=4.0,
        physical_width=None,
        units="cm",
        name=DATASET["key"],
        placement="the file's own frame (posx spans -160 to 160, posy is centred on 0)",
        placed_by="file's coordinates",
        # No stated length is compared: the samples span the stated 320 cm
        # because the authors scaled the frame to it, and the unit is read
        # from that span, so the length cannot check the track.
        tracked_point="one tracking LED (readme_hafting2008.txt: 'the tracking LED')",
        speed_thresholds=(5.0, 30.0),
    )
    return Recording(
        dataset=DATASET["key"],
        session=_SESSION,
        source=f"{DATASET['source']} hafting_et_al_2008.zip",
        license=DATASET["license"],
        times=times,
        positions=positions,
        position_units="cm",
        spike_times=spikes,
        unit_ids=np.asarray(cells),
        epochs={"session_11": sessions[:1], "session_12": sessions[1:]},
        geometry={
            "graph": graph,
            "graph_root": "west",
            **declared,
            "position_span_cm": np.ptp(positions, axis=0),
        },
        params={
            "bin_size": 4.0,
            "min_speed": 5.0,
            "dt": 0.25,
            "analysis_epochs": sessions,
            "train_epochs": train,
            "test_epochs": test,
        },
        conversions=(
            "positions: posx, posy as stored; cm inferred (posx spans exactly"
            " -160 to 160 on the 320 cm track)",
            span,
            "times, spike_times: post and ts as stored (s)",
            f"epochs: session_11 {sessions[0, 0]:.2f}-{sessions[0, 1]:.2f} s and"
            f" session_12 {sessions[1, 0]:.2f}-{sessions[1, 1]:.2f} s, split at the"
            f" {step[join]:.0f} cm step between them (no gap on the clock); the"
            " analysis epochs leave the joining interval out",
            "unit_ids: tetrode and cluster from the file names",
            declared_line,
        ),
    )


if __name__ == "__main__":
    rec = load()
    validate_recording(rec)
    duration = rec.times[-1] - rec.times[0]
    print(f"{rec.dataset}: {len(rec.spike_times or ())} units, {duration:.0f} s")
    for row in check_expectations(rec, EXPECTATIONS):
        print(f"  {'pass' if row['passed'] else 'FAIL'} {row['key']}: {row['detail']}")
    for line in rec.conversions:
        print(f"  conversion: {line}")
