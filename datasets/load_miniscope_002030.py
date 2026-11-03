"""Mouse CA1 one-photon calcium imaging in a circular open field (DANDI 002030).

What was recorded: mouse F53, day 1 of 3, exploring a 96 cm circular arena
with wall landmarks for about 12 minutes while a Miniscope V4.4 imaged dorsal
CA1 (GCaMP6s). The file holds ΔF/F traces of 26 CNMF-E components and the
body-centre position (DeepLabCut), both at 30 Hz on the imaging time base,
plus head and body direction and behavioral state flags. No spikes or
deconvolved events.

Where it lives: DANDI 002030, version 0.261016.1321, one 4.9 MB NWB file,
CC-BY-4.0.

What the loader converts: head direction is the ``headdirection`` series (a
bare ``SpatialSeries`` in ``processing/behavior``, not a ``CompassDirection``),
read with pynwb and wrapped to [−π, π]: the file says "a small share of
samples lies outside [−pi, pi]" (0.7 %). Its frame is stated: 0 along +x,
π/2 along +y. Positions are cm with the origin at a corner of the camera
frame.

Analysis choices (``params``): ``bin_size`` 2.5 cm (about 38 bins across the
96 cm arena).

Environment (``geometry["environment"]``): the 96 cm circular arena the
DANDI description states, built with ``Environment.from_polygon`` at
``bin_size``; its centre is fitted to the samples
(``_geometry.declared_circle``), and the residual of the samples' own fitted
diameter against 96 cm is in ``conversions``.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from _fetch import dandi_asset_url, fetch, open_nwb
from _geometry import declared_circle, with_environment
from _record import (
    Range,
    Recording,
    check_expectations,
    checked_span,
    validate_recording,
)

DATASET = {
    "key": "miniscope_002030",
    "title": "Mouse dorsal CA1 miniscope imaging, 96 cm circular open field",
    "source": "https://dandiarchive.org/dandiset/002030/0.261016.1321",
    "license": "CC-BY-4.0",
    "download_size": "4.9 MB",
}

# Pre-registered from the survey appendix before the first run of this loader.
EXPECTATIONS: dict[str, Any] = {
    "n_dims": 2,
    "sampling_rate_hz": Range(29.0, 31.0),
    "position_extent_at_most_100_cm": lambda rec: bool(
        np.all(np.nanmax(rec.positions, 0) - np.nanmin(rec.positions, 0) <= 100.0)
    ),
    "activity_is_dff": lambda rec: rec.activity_kind == "dff",
}

_PATH = "sub-F53/sub-F53_ses-FOF-F53-1D_behavior+ophys.nwb"
_ASSET_ID = "db375e64-d492-4bf9-963a-7935ebafcc1e"
_SHA256 = "90bf0d6998f961182c9733d9e015a290d8c0d55a879464bfdf41e291084cc060"


def load(cache: Path | None = None) -> Recording:
    """Fetch the session and return positions (cm), head direction and ΔF/F."""
    import pynwb

    from neurospatial.io.nwb import read_position
    from neurospatial.stats import wrap_angle

    path = fetch(
        dandi_asset_url(_ASSET_ID),
        Path(_PATH).name,
        sha256=_SHA256,
        subdir=DATASET["key"],
        cache=cache,
    )
    with pynwb.NWBHDF5IO(path, "r", load_namespaces=True) as io:
        nwbfile = io.read()
        position = read_position(nwbfile)
        heading_series = nwbfile.processing["behavior"]["headdirection"]
        headings = np.asarray(heading_series.data[:], dtype=float)
        calcium = nwbfile.processing["ophys"]["Fluorescence"]["Calcium"]
        dff = np.asarray(calcium.data[:], dtype=float)
        dff_times = calcium.starting_time + np.arange(len(dff)) / calcium.rate

    if position.units != "cm":
        msg = f"expected cm positions, got {position.units!r}"
        raise ValueError(msg)
    times = np.asarray(position.times, dtype=float)
    positions = np.asarray(position.positions, dtype=float)
    # The stated 96 cm circle: the tracked span must be within 10 % of it.
    span = checked_span(positions, 86.0, 106.0)
    rec = Recording(
        dataset=DATASET["key"],
        session="FOF_F53_1D",
        source=f"{DATASET['source']} {_PATH}",
        license=DATASET["license"],
        times=times,
        positions=positions,
        position_units="cm",
        headings=np.asarray(wrap_angle(headings)),
        activity=dff,
        activity_times=np.asarray(dff_times, dtype=float),
        activity_kind="dff",
        geometry={"shape": "circle", "diameter_cm": 96.0},
        params={
            "bin_size": 2.5,
            "nwb_source": str(path),
            "readers_used": ("position",),
        },
        conversions=(
            f"positions: read_position, cm; {span} (the 96 cm arena)",
            "times: rate 30 Hz; 21,427 frames = 714 s, the stated ~12 min session",
            "headings: processing/behavior/headdirection (radians), wrapped to [-pi, pi];"
            " 0.7 % of stored samples lie outside it",
            "activity: processing/ophys/Fluorescence/Calcium (dF/F), frame times from rate",
        ),
    )
    return with_environment(
        rec,
        *declared_circle(
            rec.positions,
            96.0,
            bin_size=rec.params["bin_size"],
            units="cm",
            name=DATASET["key"],
            source="DANDI 002030 description: 'circular open-field arena (96 cm in diameter)'",
        ),
        tracked_point="the body centre, DeepLabCut (DANDI 002030's body-centre position)",
    )


def _headings_as_stored(rec: Recording) -> Recording:
    # Undoes the wrap: the stored head direction, partly outside [-pi, pi].
    with open_nwb(rec.params["nwb_source"]) as nwbfile:
        stored = np.asarray(nwbfile.processing["behavior"]["headdirection"].data[:])
    return replace(rec, headings=stored)


HAZARDS = {"headings_as_stored": _headings_as_stored}


if __name__ == "__main__":
    rec = load()
    validate_recording(rec)
    print(
        f"{rec.dataset}: {rec.activity.shape[1]} cells, {rec.times[-1] - rec.times[0]:.0f} s"
    )
    for row in check_expectations(rec, EXPECTATIONS):
        print(f"  {'pass' if row['passed'] else 'FAIL'} {row['key']}: {row['detail']}")
    for line in rec.conversions:
        print(f"  conversion: {line}")
