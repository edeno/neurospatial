"""Aery Jones et al. mouse hippocampal multi-unit activity on a linear track after CNO (DANDI 000165).

What was recorded: mouse Sumac, day 6, a 32-channel probe across CA1, CA3
and dentate gyrus layers, on a linear track one hour after CNO (hM4D in
dentate and CA3 interneurons). Position for 1,504 s; the ``units`` table is a
"MUA table": one multi-unit spike train per channel (32), not sorted units.

Where it lives: DANDI 000165, version 0.211118.1526, CC-BY-4.0; the
linear-track file of 23.3 MB.

What the loader converts:

- Times: the 45,114 position samples (1,504 s at 30 frames/s) have only
  32,340 distinct timestamps on a 0.0518 s grid; 12,774 timestamps hold two
  samples. The loader keeps one sample per timestamp, the mean of the samples
  that share it.
- Spikes: every non-empty channel's train starts with a spike at exactly
  0.0 s, and spikes end at 462 s while tracking runs to 1,504 s (the LFP is
  275 s long). Read as seconds, the spikes are spatially tuned (mean spatial
  information 0.78 bits/spike), and stretched to the tracking span they are
  less so (0.54 at x 3.26), so they are seconds on the tracking clock and
  cover only its first 462 s (checked once while writing this loader). The
  spike at 0.0 is dropped from each train.
- Length unit: positions are labelled centimeters, but the tracked span is
  35 by 6 (1st-99th percentile) and the 90th percentile speed 6.6 per s:
  short and slow for a mouse linear track in cm, and the paper is not at
  hand to state the track length, so the unit is not confirmed and the
  loader returns ``position_units="arbitrary"``.

Analysis choices (``params``):

- ``bin_size`` 1 (35 bins along the track); no ``min_speed`` (no speed is
  defensible without a length unit).
- ``analysis_epochs``: 0 to the last spike (462 s), the span the spikes
  cover; rates over the whole tracking would divide the spikes of 462 s by
  1,504 s of occupancy.
- ``dt`` 0.25 s, alternating 60 s ``train_epochs``/``test_epochs`` inside
  that span.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from _fetch import dandi_asset_url, fetch, open_nwb
from _record import (
    Range,
    Recording,
    alternating_blocks,
    check_expectations,
    validate_recording,
)

DATASET = {
    "key": "aeryjones_000165",
    "title": "Aery Jones et al., mouse Sumac day 6, hippocampal multi-unit activity on a linear track",
    "source": "https://dandiarchive.org/dandiset/000165/0.211118.1526",
    "license": "CC-BY-4.0",
    "download_size": "23.3 MB",
}

# Pre-registered from the dataset catalog (linear track) and the file's
# metadata (a 32-row "MUA table", position labelled centimeters) before the
# first run of this loader. The last two are consistency requirements of any
# session: tracking at a video rate, and spikes over the tracked session.
EXPECTATIONS: dict[str, Any] = {
    "n_units": 32,
    "n_dims": 2,
    "sampling_rate_hz": Range(25.0, 35.0),
    "spikes_cover_tracking": lambda rec: bool(
        max(float(np.max(t)) for t in rec.spike_times if t.size)
        >= rec.times[0] + 0.9 * (rec.times[-1] - rec.times[0])
    ),
}


_PATH = "sub-Sumac/sub-Sumac_ses-Sumac06-LinearTrack_behavior+ecephys.nwb"
_ASSET_ID = "fd219159-b8ee-4bb3-933b-581245af0317"
_SHA256 = "58d883df3f383bacd3db293f59a8ac61f322d148a516ec5e34799ce78657efa3"


def load(cache: Path | None = None) -> Recording:
    """Fetch the session; one position per timestamp, multi-unit trains, spike span."""
    from neurospatial.io.nwb import read_position, read_units

    path = fetch(
        dandi_asset_url(_ASSET_ID),
        Path(_PATH).name,
        sha256=_SHA256,
        subdir=DATASET["key"],
        cache=cache,
    )
    with open_nwb(str(path)) as nwbfile:
        position = read_position(nwbfile)
        units = read_units(nwbfile)

    raw_times = np.asarray(position.times, dtype=float)
    raw = np.asarray(position.positions, dtype=float)
    times, index, counts = np.unique(raw_times, return_inverse=True, return_counts=True)
    positions = np.column_stack(
        [np.bincount(index, weights=column) / counts for column in raw.T]
    )
    trains = [np.asarray(s, dtype=float) for s in units.spike_times]
    starts_at_zero = sum(bool(t.size and t[0] == 0.0) for t in trains)
    trains = [t[t > 0.0] for t in trains]
    last_spike = max(float(t[-1]) for t in trains if t.size)
    span = np.percentile(positions, 99, axis=0) - np.percentile(positions, 1, axis=0)
    if (
        np.any(np.diff(raw_times) < 0)
        or counts.max() > 2
        or not 300.0 <= last_spike <= 0.5 * times[-1]
        or not 20.0 <= span.max() <= 60.0
    ):
        msg = (
            f"the file no longer matches its quirks (max samples per timestamp"
            f" {counts.max()}, last spike {last_spike:.0f} s of {times[-1]:.0f} s"
            f" tracked, span {span.round(1)}); recheck"
        )
        raise ValueError(msg)
    covered = np.array([[times[0], last_spike]])
    train, test = alternating_blocks(times[0], last_spike, 60.0)
    return Recording(
        dataset=DATASET["key"],
        session="Sumac06_LinearTrack",
        source=f"{DATASET['source']} {_PATH} (asset {_ASSET_ID})",
        license=DATASET["license"],
        times=times,
        positions=positions,
        position_units="arbitrary",
        spike_times=tuple(trains),
        unit_ids=np.asarray(units.unit_ids),
        epochs={"spikes_recorded": covered},
        params={
            "bin_size": 1.0,
            "dt": 0.25,
            "analysis_epochs": covered,
            "train_epochs": train,
            "test_epochs": test,
            "nwb_source": str(path),
            "readers_used": ("position", "units"),
        },
        conversions=(
            f"times: {len(raw_times)} samples on {len(times)} distinct timestamps;"
            " samples sharing a timestamp averaged",
            "positions: read_position, labelled cm; range checked (1st-99th"
            f" percentile span {span[0]:.1f} by {span[1]:.1f}): too short for a"
            " mouse linear track in cm to confirm the label, kept as arbitrary units",
            f"spike_times: read_units (32 channel multi-unit trains); the spike at"
            f" exactly 0.0 s dropped from {starts_at_zero} trains; spikes end at"
            f" {last_spike:.1f} s of {times[-1]:.0f} s tracked",
            "params['analysis_epochs']: 0 to the last spike",
        ),
    )


def _duplicate_timestamps_kept(rec: Recording) -> Recording:
    # The file's own arrays: every sample, timestamps repeated.
    with open_nwb(rec.params["nwb_source"]) as nwbfile:
        series = nwbfile.processing["behavior"]["Position"]["SpatialSeries"]
        times = np.asarray(series.timestamps[:], dtype=float)
        positions = np.asarray(series.data[:], dtype=float)
    return replace(rec, times=times, positions=positions)


def _tracking_beyond_spikes(rec: Recording) -> Recording:
    # A user analysing the whole tracked session: 1,504 s of occupancy for
    # spikes recorded during 462 s.
    params = {k: v for k, v in rec.params.items() if k != "analysis_epochs"}
    return replace(rec, params=params)


HAZARDS = {
    "duplicate_timestamps_kept": _duplicate_timestamps_kept,
    "tracking_beyond_spikes": _tracking_beyond_spikes,
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
