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
- Pixels to cm with each track's own ``position.linear(k).cm_per_pixel``
  (0.388 and 0.443, from the track's 2 m length and mask), not the
  session's ``cm_per_pixel`` (0.426, their average), which makes track 1
  read about 10 % long and track 2 about 4 % short. The two tracks are
  then in one frame only approximately; they stay at least 32 cm apart, so
  no bin holds both. Each track's extent along its principal axis is range
  checked (170-205 cm; the tracks are not straight, so it is less than the
  2 m path length).
- Each track's ``track_mask_x``/``track_mask_y`` is not a mask outline but
  its centreline, 1 pixel apart and 2 m long once scaled (not an independent
  check: ``cm_per_pixel`` comes from the 2 m length): kept, in cm with the
  same factor, as ``geometry["track_centrelines_cm"]``. The loader checks the
  length and that the track's samples lie a median of at most 10 cm from it.
- Spike times (s, the same Neuralynx clock) are split by ``spike_id``;
  ``unit_ids`` are ``id_conversion``'s 1000 x tetrode + cluster.

Analysis choices (``params``): ``bin_size`` 4 cm; ``min_speed`` 5 cm/s;
``analysis_epochs`` the two track exposures; ``dt`` 0.25 s, with alternating
60 s train/test blocks inside the track exposures.

Declared environment (``geometry["environment"]``): the two tracks as one
graph of two components, each the authors' centreline (simplified to within
0.5 cm, which keeps its length) as a path of segments, every edge kept.
Tirole et al. 2022 state no track width, so the track is built with a
placement tolerance measured from the samples.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from _record import (
    Range,
    Recording,
    alternating_blocks,
    check_expectations,
    intersect_intervals,
    validate_recording,
)
from _tracks import (
    declared_geometry,
    polyline_graph,
    simplify_polyline,
    with_declared,
)

from neurospatial.datasets._fetch import fetch

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
_TRACK_EXTENT_CM = (170.0, 205.0)
_SIMPLIFY_CM = 0.5
_CENTRELINE_MAX_MEDIAN_CM = 10.0
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
    positions = np.full_like(pixels, np.nan)
    epochs, factors, extents, centrelines = {}, [], [], []
    for k, track in enumerate(position.linear, start=1):
        if float(track.length) != _TRACK_M:
            msg = f"track {k} length {track.length} m, not the README's 2 m"
            raise ValueError(msg)
        on = np.asarray(track.cropped_indices, dtype=int) - 1
        factor = float(track.cm_per_pixel)
        positions[on] = pixels[on] * factor
        extent = principal_extent(positions[on])
        if not _TRACK_EXTENT_CM[0] <= extent <= _TRACK_EXTENT_CM[1]:
            msg = (
                f"track {k} spans {extent:.1f} cm along its principal axis, outside"
                f" {_TRACK_EXTENT_CM}; recheck cm_per_pixel"
            )
            raise ValueError(msg)
        epochs[f"track_{k}"] = np.array([[times[on[0]], times[on[-1]]]])
        factors.append(factor)
        extents.append(extent)
        centreline = (
            np.column_stack([track.track_mask_x, track.track_mask_y]).astype(float)
            * factor
        )
        check_centreline(centreline, positions[on], k)
        centrelines.append(centreline)
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
    rec = Recording(
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
        geometry={
            "cm_per_pixel_per_track": factors,
            "track_centrelines_cm": centrelines,
        },
        params={
            "bin_size": 4.0,
            "min_speed": 5.0,
            "dt": 0.25,
            "analysis_epochs": tracks,
            "train_epochs": intersect_intervals(blocks[0], tracks),
            "test_epochs": intersect_intervals(blocks[1], tracks),
        },
        conversions=(
            "positions: clean.x, clean.y (pixels) x each track's own cm_per_pixel"
            f" ({', '.join(f'{f:.4f}' for f in factors)}; session average"
            f" {float(position.cm_per_pixel):.4f} not used); the source does not say"
            " whether pixel y runs up or down the image, so y is kept as stored",
            "range checked: principal-axis extent per track"
            f" {', '.join(f'{e:.1f}' for e in extents)} cm (expected"
            f" {_TRACK_EXTENT_CM[0]:g}-{_TRACK_EXTENT_CM[1]:g})",
            "positions: NaN outside each track's cropped_indices"
            f" ({on_track.mean():.1%} of samples on a track)",
            f"positions: {np.mean(interpolated[on_track]):.1%} of on-track samples"
            " were interpolated by the authors (interpolated_indices), kept",
            "geometry['track_centrelines_cm']: track_mask_x, track_mask_y x the"
            " track's cm_per_pixel; lengths "
            + ", ".join(
                f"{np.linalg.norm(np.diff(c, axis=0), axis=1).sum():.1f}"
                for c in centrelines
            )
            + " cm",
            "times: clean.t as stored (s)",
            f"spike_times: split by spike_id into {len(spikes)} units",
            "unit_ids: id_conversion column 2 (1000 x tetrode + cluster)",
        ),
    )
    import networkx as nx

    graph = nx.Graph()
    for k, centreline in enumerate(centrelines, start=1):
        points = simplify_polyline(centreline, _SIMPLIFY_CM)
        names = [f"track{k}_{i}" for i in range(len(points))]
        graph = nx.compose(graph, polyline_graph(points, closed=False, names=names))
    entries, line = declared_geometry(
        graph,
        positions,
        times=times,
        kind="two tracks, each the authors' centreline as a path (two components)",
        source="file: track_mask_x, track_mask_y (each track's centreline); Tirole"
        " et al. 2022: 'two novel linear tracks (2 m)'; width not stated",
        bin_size=4.0,
        physical_width=None,
        units="cm",
        name=DATASET["key"],
        placement=f"the centrelines in cm, simplified to within"
        f" {_SIMPLIFY_CM:g} cm ({graph.number_of_edges()} segments)",
        placed_by="authors' centrelines",
        tracked_point="not stated",
        speed_thresholds=(5.0, 30.0),
    )
    return with_declared(rec, entries, line)


def check_centreline(centreline: np.ndarray, on_track: np.ndarray, k: int) -> None:
    """Refuse a centreline that is not about 2 m long or not under the samples.

    The length is not an independent check of the scale: the authors derived
    ``cm_per_pixel`` from the 2 m length. The position is: the track's samples
    must lie a median of at most ``_CENTRELINE_MAX_MEDIAN_CM`` from it.
    """
    from scipy.spatial import cKDTree

    length = float(np.linalg.norm(np.diff(centreline, axis=0), axis=1).sum())
    tracked = on_track[np.isfinite(on_track).all(axis=1)]
    offset = float(np.median(cKDTree(centreline).query(tracked)[0]))
    if not 0.95 * _TRACK_M * 100 <= length <= 1.05 * _TRACK_M * 100:
        msg = f"track {k} centreline is {length:.1f} cm long, not about 2 m"
        raise ValueError(msg)
    if offset > _CENTRELINE_MAX_MEDIAN_CM:
        msg = f"track {k} samples lie a median {offset:.1f} cm from its centreline"
        raise ValueError(msg)


def principal_extent(xy: np.ndarray) -> float:
    """Extent of 2-D samples along their first principal axis."""
    centred = xy - xy.mean(axis=0)
    axis = np.linalg.svd(centred, full_matrices=False)[2][0]
    return float(np.ptp(centred @ axis))


def pixels_labelled_cm(rec: Recording) -> Recording:
    # The stored x, y are pixels; a reader that missed cm_per_pixel would
    # keep them and call them cm (2.26 and 2.58 x too large).
    assert rec.positions is not None
    assert rec.times is not None
    positions = rec.positions.copy()
    for k, factor in enumerate(rec.geometry["cm_per_pixel_per_track"], start=1):
        ((start, stop),) = rec.epochs[f"track_{k}"]
        inside = (rec.times >= start) & (rec.times <= stop)
        positions[inside] /= factor
    return replace(rec, positions=positions)


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
