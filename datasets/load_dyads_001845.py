"""Jadhav lab rat dyads cooperating on paired W-mazes, pose keypoints, behavior only (DANDI 001845).

What was recorded: rat FXM110 on the left of two W-mazes, paired with FXM101
on the right, session 15 under a 100 % reward contingency: the pair is
rewarded when both poke matching wells. DeepLabCut tracked six keypoints
(nose, head, neck, body, tail base, tail end) in the shared overhead video;
the file holds the raw keypoints (pixels, with confidence), processed
positions in cm, well pokes and reward events. No neural data. The partner
rat is in its own file.

Where it lives: DANDI 001845, version 0.260526.1807, CC0-1.0; the smallest
behavior file (1.4 MB). Its keypoints cover only the first 123 s
(873.8-996.9 s) of a 1,577 s video and session; the dandiset's typical files
(about 9 MB) cover whole sessions. It is kept because it is the smallest
file with pose, and its short pose is itself a property of the published file.

What the loader converts: positions are ``processed_position/head``
(cm, "Maze coordinates in centimeters"): DeepLabCut coordinates with
confidence below 0.6 removed, gaps up to 0.5 s interpolated, smoothed
(sigma 0.15 s) and converted to cm by the authors. NaN where tracking was
not usable: 38 % of head samples (body 32 %, nose 45 %). The head is chosen
because the authors' own reward-well visits are defined from it; the body
has fewer NaN samples. The raw ``PoseEstimation`` keypoints have no NaN (low
confidence is a separate array) and their first nine timestamps are equal
(873.812 s), so ``read_pose`` refuses them; the processed series drop those
eight repeats. ``processed_position/tail_end`` is labelled centimeters but
spans 13-522 by 179-908, the raw pixel range: it was not converted.

Analysis choices (``params``): ``bin_size`` 2.5 cm; ``min_speed`` 4 cm/s.

Declared environment (``geometry["environment"]``): the left rat's W-maze
(Shukla et al. 2026, the paper the dandiset names: 80 x 80 cm with about
7 cm wide sections) as three parallel arms joined by a base, its node
positions fitted to the head samples with the stated 7 cm width. With 123 s
of pose the base is thinly sampled; the fit is in ``conversions``.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from _fetch import dandi_asset_url, fetch, open_nwb
from _record import Range, Recording, check_expectations, validate_recording
from _tracks import (
    arms_text,
    declared_geometry,
    edge_lengths,
    fitted_maze,
    parallel_arm_candidates,
    parallel_arm_lengths,
    with_declared,
)

DATASET = {
    "key": "dyads_001845",
    "title": "Shukla, Jadhav et al., rat FXM110 (with FXM101), cooperative W-maze, pose keypoints",
    "source": "https://dandiarchive.org/dandiset/001845/0.260526.1807",
    "license": "CC0-1.0",
    "download_size": "1.4 MB",
}

# Pre-registered from the dataset catalog (pose keypoints, 28-37 % NaN,
# behavior only) and the file's metadata (DeepLabCut PoseEstimation of six
# keypoints in pixels; processed positions in centimeters with NaN where
# confidence was low; video at about 30 Hz) before the first run of this
# loader.
EXPECTATIONS: dict[str, Any] = {
    "behavior_only": lambda rec: rec.spike_times is None,
    "n_dims": 2,
    "sampling_rate_hz": Range(29.0, 31.0),
    "position_nan_fraction": Range(0.28, 0.40),
    "centimetres": lambda rec: rec.position_units == "cm",
}


_PATH = "sub-FXM110/sub-FXM110_ses-15-rc100_behavior.nwb"
_ASSET_ID = "65a0aa61-cda8-4353-886c-ee950558d930"
_SHA256 = "498cdb5334d2e932604a93ca09ec44e510e643ef2a57178ac09006b04c9a119d"
_KEYPOINT = "head"
_PROCESSED = "processing/behavior/processed_position"
# A rat W-maze in cm: the tracked span along each axis must lie here.
_MAZE_CM = (50.0, 200.0)
_TRACK_WIDTH_CM = 7.0
_FIT_CAP_CM = 10.0


def load(cache: Path | None = None) -> Recording:
    """Fetch the session; head position in cm with NaN, well pokes and rewards."""
    from neurospatial.io.nwb import read_position

    path = fetch(
        dandi_asset_url(_ASSET_ID),
        Path(_PATH).name,
        sha256=_SHA256,
        subdir=DATASET["key"],
        cache=cache,
    )
    with open_nwb(str(path)) as nwbfile:
        position = read_position(nwbfile, position_name=f"{_PROCESSED}/{_KEYPOINT}")
        tail = read_position(nwbfile, position_name=f"{_PROCESSED}/tail_end")
        events = nwbfile.processing["behavior"]["behavioral_events"]
        pokes = {
            name: np.asarray(series.timestamps[:], dtype=float)
            for name, series in events.time_series.items()
        }
        visits = nwbfile.processing["behavior"]["reward_well_visits"].to_dataframe()

    positions = np.asarray(position.positions, dtype=float)
    span = np.nanmax(positions, axis=0) - np.nanmin(positions, axis=0)
    tail_span = np.nanmax(np.asarray(tail.positions), axis=0)
    if (
        position.units != "cm"
        or not all(_MAZE_CM[0] <= v <= _MAZE_CM[1] for v in span)
        or tail_span.max() < 4 * span.max()
    ):
        msg = (
            f"{_KEYPOINT}: unit {position.units!r}, span {span.round(1)} cm; tail_end"
            f" max {tail_span.round(0)}; the processed positions changed, recheck"
        )
        raise ValueError(msg)
    nan_share = float(np.isnan(positions).any(axis=1).mean())
    rec = Recording(
        dataset=DATASET["key"],
        session="FXM110-ses-15-rc100",
        source=f"{DATASET['source']} {_PATH} (asset {_ASSET_ID})",
        license=DATASET["license"],
        times=np.asarray(position.times, dtype=float),
        positions=positions,
        position_units="cm",
        epochs={
            "well_visits": visits[["start_time", "stop_time"]].to_numpy(dtype=float)
        },
        events=pokes,
        params={
            "bin_size": 2.5,
            "min_speed": 4.0,
            "nwb_source": str(path),
            "readers_used": ("position",),
            "pose_keypoint": "PoseEstimationSeriesHead",
        },
        conversions=(
            f"positions: read_position(position_name='{_PROCESSED}/{_KEYPOINT}'), cm;"
            f" range checked: {np.nanmin(positions[:, 0]):.1f}-"
            f"{np.nanmax(positions[:, 0]):.1f} by {np.nanmin(positions[:, 1]):.1f}-"
            f"{np.nanmax(positions[:, 1]):.1f} cm; {nan_share:.1%} NaN, kept",
            f"{_PROCESSED}/tail_end labelled cm reaches {tail_span.round(0)}: pixels,"
            " not used",
            "events: behavioral_events timestamps by name; epochs: reward_well_visits",
        ),
    )
    graph = fitted_maze(parallel_arm_candidates(positions), positions, cap=_FIT_CAP_CM)
    entries, line = declared_geometry(
        graph,
        positions,
        times=rec.times,
        kind="W-maze: three parallel arms joined by a base",
        source="Shukla et al. 2026 Methods: 'elevated W-shaped mazes (80 x 80 cm)"
        " with ~7-cm-wide track sections'",
        bin_size=2.5,
        physical_width=_TRACK_WIDTH_CM,
        units="cm",
        name=DATASET["key"],
        placed_by="fitted to the samples",
        stated_lengths=dict(
            zip(
                ("base", "arms"),
                ((x, 80.0) for x in parallel_arm_lengths(graph)),
                strict=True,
            )
        ),
        placement=f"node positions fitted to the samples ({arms_text(graph, 'cm')};"
        f" stated 80 x 80 cm); edges (cm): {edge_lengths(graph)}",
        tracked_point="the DeepLabCut 'head' keypoint as processed by the authors (processed_position/head)",
        speed_thresholds=(4.0, 30.0),
    )
    return with_declared(rec, entries, line)


def _tail_end_keypoint(rec: Recording) -> Recording:
    # A user choosing the tail_end keypoint: pixels under a cm label.
    with open_nwb(rec.params["nwb_source"]) as nwbfile:
        series = nwbfile.processing["behavior"]["processed_position"]["tail_end"]
        positions = np.asarray(series.data[:], dtype=float)
    return replace(rec, positions=positions)


HAZARDS = {"unconverted_keypoint": _tail_end_keypoint}


if __name__ == "__main__":
    rec = load()
    validate_recording(rec)
    print(f"{rec.dataset}: behavior only, {rec.times[-1] - rec.times[0]:.0f} s")
    for row in check_expectations(rec, EXPECTATIONS):
        print(f"  {'pass' if row['passed'] else 'FAIL'} {row['key']}: {row['detail']}")
    for line in rec.conversions:
        print(f"  conversion: {line}")
