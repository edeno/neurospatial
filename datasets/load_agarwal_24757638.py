"""Agarwal, Pastalkova et al. rat CA1 and mPFC on a 3-arm maze with a wheel delay (Figshare 24757638).

What was recorded: one rat alternating between the arms of a 3-arm maze
(left, centre, right, centre, ...), running in a wheel before each arm
visit, while silicon probes recorded CA1 (shanks 1-32) and mPFC (higher
shanks): 439 sorted clusters, 95 trials, 63 minutes.

Where it lives: Figshare 24757638 v4 (CC BY 4.0), session ``rec02``:
``rec02_BehavElectrDataLFP.mat`` (226 MB, MATLAB v5), whose ``Track``,
``Spike``, ``Clu`` and ``Laps`` structs the record's description documents.
``data02.mat`` holds only 25 Hz spike counts and is not used.

What the loader converts:

- Spike times: ``Spike.res20kHz`` / 20 kHz (``xml.SampleRate``); the
  record's ``Spike.res`` (1250 Hz samples) agrees to 0.4 ms. Units are
  ``Spike.totclu`` (= ``Clu.totClu``), sorted per unit.
- Tracking: ``Track`` is sampled at the 1250 Hz LFP rate; sample i (1-based)
  is at i / 1250 s, the clock ``Spike.res`` uses. Every 25th sample is
  kept (50 Hz); ``xMM`` changes value every 34 samples on average
  (about 37 Hz), so 50 Hz keeps every camera frame.
- Positions: ``xMM``/``yMM`` (mm, as documented; also 3.77 x the
  ``xPix``/``yPix`` columns) / 10 -> cm; range checked. Samples with
  ``xPix`` = ``yPix`` = 0 (untracked; ``xMM`` = ``yMM`` = 0 there) are NaN.
- ``headDirDeg`` is not used: 78 % of tracked samples are exactly 0 and the
  rest sit on 45° steps (106 distinct values), and heading minus movement
  direction has a median spread of 89°, which is what an unrelated angle
  gives.
- Trials: ``Laps.startLfpInd``/``endLfpInd`` / 1250 Hz.

Analysis choices (``params``): ``bin_size`` 4 cm; ``min_speed`` 5 cm/s;
``dt`` 0.25 s, even and odd trials train and test the decoder;
``region`` CA1 for shanks 1-32, else mPFC; ``is_interneuron`` from
``Clu.isIntern``.

Declared environment (``geometry["environment"]``): the 3-arm maze as three
parallel arms joined by a base, the layout the samples show, its node
positions fitted to them; the wheel beside the base is not part of the
graph. Agarwal et al. 2026 (Methods) call it "the 2.13m-long maze" without
saying what that spans, and no source read here states a width, so the
track is built with a placement tolerance measured from the samples
(``_tracks.declared_geometry``).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from _fetch import fetch
from _record import (
    Range,
    Recording,
    check_expectations,
    checked_span,
    heading_vs_movement,
    validate_recording,
)
from _tracks import (
    arms_text,
    declared_geometry,
    edge_lengths,
    fitted_maze,
    parallel_arm_candidates,
    with_declared,
)

DATASET = {
    "key": "agarwal_24757638",
    "title": "Rat 3-arm alternation with wheel running, session rec02, CA1 and mPFC units",
    "source": "https://doi.org/10.6084/m9.figshare.24757638.v4",
    "license": "CC-BY-4.0",
    "download_size": "226 MB",
}

# Pre-registered from the Figshare description (spikes as 1250 Hz sample
# indices, xMM/yMM in mm, laps) and the file's variable list (439 clusters,
# 95 trials, tracking at 1250 Hz kept every 25th sample) before the first
# run of this loader.
EXPECTATIONS: dict[str, Any] = {
    "n_units": 439,
    "n_dims": 2,
    "sampling_rate_hz": Range(49.9, 50.1),
    "maze_extent_cm": lambda rec: Range(80.0, 220.0).contains(
        float(
            np.nanmax(np.ptp(rec.positions[np.isfinite(rec.positions).all(1)], axis=0))
        )
    ),
    "95_trials": lambda rec: len(rec.epochs["trials"]) == 95,
}

_NAME = "rec02_BehavElectrDataLFP.mat"
_URL = "https://ndownloader.figshare.com/files/43496781"
_SHA256 = "069a22de30a5c35c71dcde19d76e2480118aad2c3f59ed810dc56701cf187ce4"
_LFP_HZ = 1250.0
_KEEP_EVERY = 25
_SPAN_CM = (150.0, 260.0)
_FIT_CAP_CM = 10.0
_CA1_SHANKS = 32
RUNNING_CM_S = 10.0


def load(cache: Path | None = None) -> Recording:
    """Fetch session rec02; positions in cm at 50 Hz, spikes in s, trials."""
    from scipy.io import loadmat

    path = fetch(_URL, _NAME, sha256=_SHA256, subdir=DATASET["key"], cache=cache)
    mat = loadmat(
        path,
        squeeze_me=True,
        struct_as_record=False,
        variable_names=["Track", "Spike", "Clu", "Laps", "xml"],
    )
    track, spike, clu, laps = mat["Track"], mat["Spike"], mat["Clu"], mat["Laps"]
    spike_hz = float(mat["xml"].SampleRate)
    seconds = np.asarray(spike.res20kHz, dtype=float) / spike_hz
    check = np.abs(seconds - np.asarray(spike.res, dtype=float) / _LFP_HZ).max()
    if check > 1.0 / _LFP_HZ:
        msg = f"res20kHz and res disagree by {check * 1e3:.1f} ms"
        raise ValueError(msg)
    units = np.asarray(clu.totClu, dtype=int)
    owner = np.asarray(spike.totclu, dtype=int)
    order = np.argsort(seconds, kind="stable")
    seconds, owner = seconds[order], owner[order]
    spikes = tuple(seconds[owner == u] for u in units)

    keep = np.arange(0, len(track.xMM), _KEEP_EVERY)
    times = (keep + 1) / _LFP_HZ
    untracked = (track.xPix[keep] == 0) & (track.yPix[keep] == 0)
    positions = np.column_stack([track.xMM[keep], track.yMM[keep]]) / 10.0
    positions[untracked] = np.nan
    span = checked_span(positions, *_SPAN_CM)
    heading = np.radians(np.asarray(track.headDirDeg, dtype=float)[keep])
    heading[untracked] = np.nan
    offset, spread = heading_vs_movement(
        times, positions, heading, min_speed=RUNNING_CM_S
    )
    exactly_zero = float(np.mean(heading[~untracked] == 0.0))
    trials = np.column_stack(
        [
            np.asarray(laps.startLfpInd, dtype=float) / _LFP_HZ,
            np.asarray(laps.endLfpInd, dtype=float) / _LFP_HZ,
        ]
    )
    shank = np.asarray(clu.shank, dtype=int)
    rec = Recording(
        dataset=DATASET["key"],
        session="rec02",
        source=_URL,
        license=DATASET["license"],
        times=times,
        positions=positions,
        position_units="cm",
        spike_times=spikes,
        unit_ids=units,
        epochs={"trials": trials},
        params={
            "bin_size": 4.0,
            "min_speed": 5.0,
            "dt": 0.25,
            "train_epochs": trials[::2],
            "test_epochs": trials[1::2],
            "region": np.where(shank <= _CA1_SHANKS, "CA1", "mPFC"),
            "is_interneuron": np.asarray(clu.isIntern, dtype=bool),
        },
        conversions=(
            f"spike_times: Spike.res20kHz / {spike_hz:g} Hz (res / 1250 agrees to"
            f" {check * 1e3:.1f} ms); split by totclu",
            f"times: Track sample i (1-based) at i / {_LFP_HZ:g} s; every"
            f" {_KEEP_EVERY}th sample kept ({_LFP_HZ / _KEEP_EVERY:g} Hz)",
            "positions: xMM, yMM / 10 -> cm",
            span,
            f"positions: {untracked.mean():.2%} of kept samples with xPix = yPix = 0"
            " -> NaN",
            f"headings: headDirDeg not used: {exactly_zero:.0%} of tracked samples are"
            f" exactly 0; minus movement direction above {RUNNING_CM_S:g} cm/s: mean"
            f" {np.degrees(offset):.1f} deg, median spread {np.degrees(spread):.1f} deg",
            "epochs['trials']: Laps.startLfpInd, endLfpInd / 1250",
            f"params region: CA1 for shanks 1-{_CA1_SHANKS}"
            f" ({int(np.sum(shank <= _CA1_SHANKS))} units), mPFC above",
        ),
    )
    graph = fitted_maze(parallel_arm_candidates(positions), positions, cap=_FIT_CAP_CM)
    entries, line = declared_geometry(
        graph,
        positions,
        times=times,
        kind="3-arm maze: three parallel arms joined by a base",
        source="Figshare 24757638 description (3-arm maze and a wheel); Agarwal et"
        " al. 2026 Methods: 'The 2.13m-long maze' (what the length spans is not"
        " stated); width not stated",
        bin_size=4.0,
        physical_width=None,
        units="cm",
        name=DATASET["key"],
        placed_by="fitted to the samples",
        placement=f"node positions fitted to the samples ({arms_text(graph, 'cm')});"
        f" edges (cm): {edge_lengths(graph)}",
        tracked_point="not stated (the file's Track xMM, yMM)",
        speed_thresholds=(5.0, 30.0),
    )
    return with_declared(rec, entries, line)


def untracked_as_origin(rec: Recording) -> Recording:
    # The file stores untracked samples as xMM = yMM = 0; a reader that
    # kept them would put them at the camera's corner.
    assert rec.positions is not None
    return replace(rec, positions=np.nan_to_num(rec.positions, nan=0.0))


HAZARDS = {"untracked_as_origin": untracked_as_origin}


if __name__ == "__main__":
    rec = load()
    validate_recording(rec)
    duration = rec.times[-1] - rec.times[0]
    print(f"{rec.dataset}: {len(rec.spike_times or ())} units, {duration:.0f} s")
    for row in check_expectations(rec, EXPECTATIONS):
        print(f"  {'pass' if row['passed'] else 'FAIL'} {row['key']}: {row['detail']}")
    for line in rec.conversions:
        print(f"  conversion: {line}")
