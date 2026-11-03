"""Tirole, Huelin Gorriz, Takigawa, Kukovska & Bendor rat CA1 on two linear tracks (Zenodo 7114733).

What was recorded: rat 5, session 1, dorsal CA1 tetrodes (Neuralynx), in a
3.6 h session: rest in a sleep pot, 15 min on each of two novel 2 m linear
tracks (a third track run first was removed by the authors, the record's
README says, for consistency between protocols), then rest. One camera at
25 Hz.

Where it lives: Zenodo 7114733 (CC0), ``RAT5_SESS1_extracted_clusters.mat``
(7.2 MB) and ``RAT5_SESS1_extracted_position.mat`` (12.8 MB), MATLAB v5
structs described in the record's ``README.txt``. Note: the record's
``RAT1_SESS2`` and ``RAT4_SESS1`` files are byte-identical (same MD5 for both
the clusters and the position file), so one of those two sessions is not
what its name says; rat 5 is not affected.

What the loader converts:

- Positions: ``position.clean.x``/``y`` (pixels, cleaned by the authors and
  interpolated across removed jumps) at ``position.clean.t`` (s, the
  camera's own timestamps, 0.028-0.053 s apart). Only samples on a track
  (``position.linear(k).cropped_indices``) are kept; the rest of the session
  (rest pot, transfers) is NaN.
- Pixels to cm with ``position.cm_per_pixel`` (0.426), the average of the
  two tracks' own calibrations (0.388 and 0.443, from each track's 2 m
  length and mask). One factor keeps the two tracks in one consistent
  frame; with it, distances on track 1 read about 10 % long and on track 2
  about 4 % short. The span is range checked.
- Spike times (s, the same Neuralynx clock) are split by ``spike_id``;
  ``unit_ids`` are ``id_conversion``'s 1000 x tetrode + cluster.

Analysis choices (``params``): ``bin_size`` 4 cm; ``min_speed`` 5 cm/s;
``analysis_epochs`` the two track exposures; ``dt`` 0.25 s, with alternating
60 s train/test blocks inside the track exposures.
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
    alternating_blocks,
    check_expectations,
    checked_span,
    intersect_intervals,
    validate_recording,
)

DATASET = {
    "key": "takigawa_7114733",
    "title": "Tirole et al. (2022), rat 5 session 1, CA1 units on two novel 2 m linear tracks",
    "source": "https://doi.org/10.5281/zenodo.7114733",
    "license": "CC0",
    "download_size": "20.0 MB",
}

# Pre-registered from the record's README (25 Hz camera, two 2 m tracks,
# rest before, between and after) and the files' headers (109 clusters,
# track lengths 2 m) before the first run of this loader.
EXPECTATIONS: dict[str, Any] = {
    "n_units": 109,
    "n_dims": 2,
    "sampling_rate_hz": Range(24.9, 25.1),
    "two_tracks": lambda rec: len(rec.epochs["track_1"]) == 1
    and len(rec.epochs["track_2"]) == 1,
    "track_extent_cm": lambda rec: Range(180.0, 240.0).contains(
        float(
            np.nanmax(np.ptp(rec.positions[np.isfinite(rec.positions).all(1)], axis=0))
        )
    ),
}

_BASE = "https://zenodo.org/api/records/7114733/files"
_FILES = {
    "RAT5_SESS1_extracted_clusters.mat": "c8b486790c26a871a1ef34377805f8672c48e4ff8c41c44616f1f1c2a5291b36",
    "RAT5_SESS1_extracted_position.mat": "85472dfc7b3c9c560e1c951e9f81759c7da221c948b01ece7d1437eb50bdcb46",
}
_SPAN_CM = (180.0, 300.0)
_TRACK_M = 2.0


def _read(name: str, cache: Path | None) -> Any:
    from scipy.io import loadmat

    path = fetch(
        f"{_BASE}/{name}/content",
        name,
        sha256=_FILES[name],
        subdir=DATASET["key"],
        cache=cache,
    )
    return loadmat(path, squeeze_me=True, struct_as_record=False)


def load(cache: Path | None = None) -> Recording:
    """Fetch rat 5 session 1; on-track positions in cm, spikes in s."""
    position = _read("RAT5_SESS1_extracted_position.mat", cache)["position"]
    clusters = _read("RAT5_SESS1_extracted_clusters.mat", cache)["clusters"]
    clean = position.clean
    times = np.asarray(clean.t, dtype=float)
    pixels = np.column_stack([clean.x, clean.y]).astype(float)
    cm_per_pixel = float(position.cm_per_pixel)
    positions = np.full_like(pixels, np.nan)
    epochs, track_factors = {}, []
    for k, track in enumerate(position.linear, start=1):
        if float(track.length) != _TRACK_M:
            msg = f"track {k} length {track.length} m, not the README's 2 m"
            raise ValueError(msg)
        on = np.asarray(track.cropped_indices, dtype=int) - 1
        positions[on] = pixels[on] * cm_per_pixel
        epochs[f"track_{k}"] = np.array([[times[on[0]], times[on[-1]]]])
        track_factors.append(float(track.cm_per_pixel))
    span = checked_span(positions, *_SPAN_CM)
    interpolated = np.zeros(len(times), dtype=bool)
    interpolated[np.asarray(position.interpolated_indices, dtype=int) - 1] = True
    on_track = np.isfinite(positions).all(axis=1)
    ids = np.asarray(clusters.spike_id, dtype=int)
    spike_times = np.asarray(clusters.spike_times, dtype=float)
    conversion = np.asarray(clusters.id_conversion, dtype=int)
    order = np.argsort(spike_times, kind="stable")
    ids, spike_times = ids[order], spike_times[order]
    spikes = tuple(spike_times[ids == unit] for unit in conversion[:, 0])
    tracks = np.vstack([epochs["track_1"], epochs["track_2"]])
    blocks = alternating_blocks(times[0], times[-1], 60.0)
    return Recording(
        dataset=DATASET["key"],
        session="RAT5_SESS1",
        source=f"{DATASET['source']} RAT5_SESS1_extracted_*.mat",
        license=DATASET["license"],
        times=times,
        positions=positions,
        position_units="cm",
        spike_times=spikes,
        unit_ids=conversion[:, 1],
        epochs=epochs,
        params={
            "bin_size": 4.0,
            "min_speed": 5.0,
            "dt": 0.25,
            "analysis_epochs": tracks,
            "train_epochs": intersect_intervals(blocks[0], tracks),
            "test_epochs": intersect_intervals(blocks[1], tracks),
        },
        conversions=(
            f"positions: clean.x, clean.y (pixels) x cm_per_pixel {cm_per_pixel:.4f}"
            f" (tracks' own: {', '.join(f'{f:.4f}' for f in track_factors)})",
            span,
            "positions: NaN outside each track's cropped_indices"
            f" ({on_track.mean():.1%} of samples on a track)",
            f"positions: {np.mean(interpolated[on_track]):.1%} of on-track samples"
            " were interpolated by the authors (interpolated_indices), kept",
            "times: clean.t as stored (s)",
            f"spike_times: split by spike_id into {len(spikes)} units",
            "unit_ids: id_conversion column 2 (1000 x tetrode + cluster)",
        ),
    )


def pixels_labelled_cm(rec: Recording) -> Recording:
    # The stored x, y are pixels; a reader that missed cm_per_pixel would
    # keep them and call them cm (2.35 x too large).
    assert rec.positions is not None
    return replace(rec, positions=rec.positions / 0.42599001942669384)


HAZARDS = {"pixels_labelled_cm": pixels_labelled_cm}


if __name__ == "__main__":
    rec = load()
    validate_recording(rec)
    duration = rec.times[-1] - rec.times[0]
    print(f"{rec.dataset}: {len(rec.spike_times or ())} units, {duration:.0f} s")
    for row in check_expectations(rec, EXPECTATIONS):
        print(f"  {'pass' if row['passed'] else 'FAIL'} {row['key']}: {row['detail']}")
    for line in rec.conversions:
        print(f"  conversion: {line}")
