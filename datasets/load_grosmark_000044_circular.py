"""Grosmark & Buzsaki 2016 rat CA1 on a circular track with sleep before and after (DANDI 000044).

What was recorded: rat Achilles, 11 Jan 2013, 104 units from bilateral CA1
silicon probes over about 10 h: sleep in the home cage (PRE, 5.4 h), running
on a 1 m circular track (MAZE, 45 min) and sleep again (POST, 4.1 h).
Position (one LED) is tracked at 39.06 Hz during the maze epoch only.

Where it lives: DANDI 000044 version 0.261005.2101, CC-BY-4.0. The file is
9.2 GB, almost all LFP, so it is not downloaded: the loader reads the few
datasets it needs by HTTP byte range (``_fetch.open_nwb``), after checking the
SHA-256 DANDI publishes for the asset.

What the loader converts:

- ``CircularMazePosition`` (x, y) is metres; converted to cm. Untracked
  samples are NaN in the file and stay NaN.
- ``CircularMazeLinearizedPosition`` (metres, 0 to about 2.9 m around the
  loop) is kept in ``params["linearized"]`` in cm. Its two ends are the same
  place on the track, and a share of its samples is exactly 0.
- Epochs ``PREEpoch``/``MazeEpoch``/``POSTEpoch`` become ``pre``/``maze``/
  ``post``; the ``states`` table becomes ``awake``, ``non_rem`` and ``rem``
  epochs, as for the figure-8.
- The file has no ripple table. Replay candidates are population bursts in
  POST non-REM sleep found by a stated rule (``population_bursts``) and stored
  as ``events["candidate_events"]``, not as ripples.

Analysis choices (``params``): ``bin_size`` 2.5 cm; ``min_speed`` 4 cm/s, as
for the figure-8; ``dt`` 0.25 s with alternating 60 s train/test blocks inside
the maze epoch; ``replay_dt`` 0.02 s; the first ``N_CANDIDATES`` POST non-REM
bursts are decoded, to bound the run time.

Declared environment (``geometry["environment"]``, ``declare_circle``): the
stated 1 m diameter circle (Grosmark & Buzsaki 2016) as a closed loop of
``N_VERTICES`` segments, centred by a robust circle fit to the samples on the
track; the width is not stated. ``geometry["lap_start_region"]`` cuts the
loop where the file's linearized coordinate starts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from _fetch import dandi_stream_url, open_nwb
from _record import (
    Range,
    Recording,
    alternating_blocks,
    check_expectations,
    checked_span,
    validate_recording,
)
from _tracks import (
    circle_points,
    cut_region,
    declared_geometry,
    fit_circle_centre,
    polyline_graph,
    with_declared,
)

DATASET = {
    "key": "grosmark_000044_circular",
    "title": "Grosmark & Buzsaki (2016), Achilles 11012013, rat CA1, 1 m circular track",
    "source": "https://dandiarchive.org/dandiset/000044/0.261005.2101",
    "license": "CC-BY-4.0",
    "download_size": "9.19 GB (streamed)",
}

# Pre-registered from the dataset catalog (104 units, 39.06 Hz, 1 m circular track, PRE/MAZE/POST epochs) before the first run of this loader.
EXPECTATIONS: dict[str, Any] = {
    "n_units": 104,
    "n_dims": 2,
    "sampling_rate_hz": Range(38.0, 40.0),
    "position_extent_cm": lambda rec: Range(90, 130).contains(
        float(
            np.nanmax(np.ptp(rec.positions[np.isfinite(rec.positions).all(1)], axis=0))
        )
    ),
    "epoch_names": ["maze", "post", "pre"],
}


_ASSET_ID = "080a847a-5211-4101-bc05-fd45b7e77dbd"
_SHA256 = "aed03b46d70d5a0cbf334ea034b3de6b16a6d2bbd5219530b95c97477356b157"
_PATH = "sub-Achilles/sub-Achilles_ses-Achilles-11012013_behavior+ecephys.nwb"
N_CANDIDATES = 200
_BURST_SIGMA_S = 0.015
_BURST_SD = 3.0
_BURST_DURATION_S = (0.05, 0.5)


def population_bursts(
    spike_times: tuple[np.ndarray, ...], windows: np.ndarray
) -> np.ndarray:
    """Population bursts inside ``windows``, as (start, stop) rows in s.

    Rule: the multiunit rate (every unit, 1 ms bins) smoothed with a 15 ms
    Gaussian; an event is a stretch whose peak exceeds the mean by 3 SD
    (mean and SD over ``windows``), bounded by the mean crossings around the
    peak, kept if it lasts 50-500 ms.
    """
    from scipy.ndimage import gaussian_filter1d

    step = 0.001
    events = []
    rows = []
    for start, stop in windows:
        edges = np.arange(start, stop + step, step)
        counts = sum(np.histogram(train, edges)[0] for train in spike_times)
        rows.append(
            (edges, gaussian_filter1d(counts.astype(float), _BURST_SIGMA_S / step))
        )
    rate = np.concatenate([r for _, r in rows])
    mean, sd = rate.mean(), rate.std()
    for edges, smooth in rows:
        above = smooth > mean
        change = np.diff(above.astype(int))
        starts = np.flatnonzero(change == 1) + 1
        stops = np.flatnonzero(change == -1) + 1
        if above[0]:
            starts = np.insert(starts, 0, 0)
        if above[-1]:
            stops = np.append(stops, len(above))
        for a, b in zip(starts, stops, strict=True):
            duration = (b - a) * step
            if smooth[a:b].max() > mean + _BURST_SD * sd and (
                _BURST_DURATION_S[0] <= duration <= _BURST_DURATION_S[1]
            ):
                events.append((edges[a], edges[b]))
    return np.asarray(events, dtype=float).reshape(-1, 2)


def load_session(
    *,
    dataset: dict[str, str],
    asset_id: str,
    sha256: str,
    path: str,
    track: str,
    span_cm: tuple[float, float],
    track_cm: tuple[float, float],
) -> Recording:
    """One 000044 session read by byte range; positions in cm."""
    from neurospatial.io.nwb import read_intervals, read_position, read_units

    source = dandi_stream_url(asset_id, sha256=sha256)
    with open_nwb(source) as nwbfile:
        # The container name alone is refused (G-012); the series path works.
        position = read_position(
            nwbfile,
            position_name=f"processing/behavior/{track}Position/{track}SpatialSeries",
        )
        linear = nwbfile.processing["behavior"][f"{track}LinearizedPosition"]
        linear_series = next(iter(linear.spatial_series.values()))
        linearized = np.asarray(linear_series.data[:], dtype=float).ravel()
        linear_unit = linear_series.unit
        units = read_units(nwbfile)
        session_epochs = read_intervals(nwbfile, interval_name="epochs")
        states = nwbfile.processing["behavior"]["states"].to_dataframe()

    if position.units != "m" or linear_unit != "meters":
        msg = f"expected metres, got {position.units!r} and {linear_unit!r}"
        raise ValueError(msg)
    times = np.asarray(position.times, dtype=float)
    positions = np.asarray(position.positions, dtype=float) * 100.0
    span = checked_span(positions, *span_cm)
    on_track = np.isfinite(linearized) & np.isfinite(positions).all(axis=1)
    track_length = float(np.nanmax(linearized) * 100.0)
    if not track_cm[0] <= track_length <= track_cm[1]:
        msg = (
            f"linearized track length {track_length:.0f} cm outside {track_cm}; recheck"
        )
        raise ValueError(msg)
    windows = np.column_stack(
        [np.asarray(session_epochs.start), np.asarray(session_epochs.stop)]
    )
    labels = [str(x).removesuffix("Epoch").lower() for x in session_epochs.label]
    epochs: dict[str, np.ndarray] = {
        label: windows[i : i + 1] for i, label in enumerate(labels)
    }
    for label in sorted(states["label"].unique()):
        key = str(label).lower().replace("-", "_")
        epochs[key] = states.loc[
            states["label"] == label, ["start_time", "stop_time"]
        ].to_numpy(dtype=float)
    maze = epochs["maze"]
    spike_times = tuple(np.asarray(s, dtype=float) for s in units.spike_times)
    post_nrem = epochs["non_rem"][
        (epochs["non_rem"][:, 0] >= epochs["post"][0, 0])
        & (epochs["non_rem"][:, 1] <= epochs["post"][0, 1])
    ]
    bursts = population_bursts(spike_times, post_nrem)
    train, test = alternating_blocks(maze[0, 0], maze[0, 1], 60.0)
    finite = np.isfinite(linearized)
    return Recording(
        dataset=dataset["key"],
        session=str(nwbfile.session_id) if nwbfile.session_id else path,
        source=f"{dataset['source']} {path} (asset {asset_id})",
        license=dataset["license"],
        times=times,
        positions=positions,
        position_units="cm",
        spike_times=spike_times,
        unit_ids=np.asarray(units.unit_ids),
        epochs=epochs,
        events={"candidate_events": bursts[:N_CANDIDATES], "all_bursts": bursts},
        params={
            "bin_size": 2.5,
            "min_speed": 4.0,
            "dt": 0.25,
            "replay_dt": 0.02,
            "analysis_epochs": maze,
            "train_epochs": train,
            "test_epochs": test,
            "linearized": linearized * 100.0,
            "nwb_source": source,
            "readers_used": ("position", "units"),
        },
        conversions=(
            f"positions: read_position {track}Position/{track}SpatialSeries, m times 100"
            f" -> cm; {span}; {on_track.mean() / np.isfinite(positions).all(axis=1).mean():.0%}"
            " of tracked samples are on the track (linearized coordinate defined)",
            f"params['linearized']: {track}LinearizedPosition, m times 100 -> cm,"
            f" 0 to {track_length:.0f} cm;"
            f" {np.mean(linearized[finite] == 0):.2%} of finite samples are exactly 0",
            "epochs: intervals/epochs labels without 'Epoch', lower case; sleep"
            " states from processing/behavior/states",
            f"events['candidate_events']: first {N_CANDIDATES} of {len(bursts)}"
            " population bursts in POST non-REM (rule in population_bursts); the"
            " file has no ripple table",
        ),
    )


DIAMETER_CM = 100.0
N_VERTICES = 72


def declare_circle(rec: Recording) -> Recording:
    """``rec`` with the stated 1 m circle declared, centred on the samples.

    The centre is a robust circle fit to the samples where the file's
    linearized coordinate is defined; the radius is the stated one. The lap
    start region is drawn across the track where that coordinate starts.
    """
    linear = np.asarray(rec.params["linearized"], dtype=float)
    on_track = np.isfinite(linear) & np.isfinite(rec.positions).all(axis=1)
    centre, radius = fit_circle_centre(rec.positions[on_track], scale=3.0)
    graph = polyline_graph(
        circle_points(centre, DIAMETER_CM / 2, N_VERTICES), closed=True
    )
    start = rec.positions[on_track & (linear > 0) & (linear < 5.0)].mean(axis=0)
    angle = np.arctan2(*(start - centre)[::-1])
    point = centre + DIAMETER_CM / 2 * np.array([np.cos(angle), np.sin(angle)])
    region = cut_region(
        "linearized_zero",
        point,
        (-np.sin(angle), np.cos(angle)),
        half_length=3.0,
        half_width=15.0,
    )
    entries, line = declared_geometry(
        graph,
        rec.positions,
        kind=f"circle {DIAMETER_CM:g} cm in diameter as a {N_VERTICES}-gon loop",
        source="Grosmark & Buzsaki 2016 Supplementary Materials: 'a 1 m in"
        " diameter wooden circular maze'; width not stated",
        bin_size=rec.params["bin_size"],
        track_width=None,
        units="cm",
        name=DATASET["key"],
        placed_by="fitted to the samples",
        stated_lengths={"diameter": (2 * radius, DIAMETER_CM)},
        placement=f"centre ({centre[0]:.1f}, {centre[1]:.1f}) cm from a robust"
        f" circle fit to the on-track samples (their own radius {radius:.1f} cm);"
        f" lap start region across the track at the file's linearized zero,"
        f" {np.degrees(angle):.0f} deg from the centre",
    )
    return with_declared(rec, {**entries, "lap_start_region": region}, line)


def load(cache: Path | None = None) -> Recording:
    """Stream the session; positions in cm, spikes, epochs and candidate events."""
    del cache  # read by byte range: nothing is cached
    rec = load_session(
        dataset=DATASET,
        asset_id=_ASSET_ID,
        sha256=_SHA256,
        path=_PATH,
        track="CircularMaze",
        span_cm=(100.0, 160.0),
        track_cm=(250.0, 330.0),
    )
    return declare_circle(rec)


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
