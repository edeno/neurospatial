"""Kanter et al. rat LEC and CA1 Neuropixels units, open field with novel objects (EBRAINS 10.25493/GTVB-FQR).

What was recorded: rat 29629 ("Karri"), 29 Oct 2024, Neuropixels 2.0 probes in
lateral entorhinal cortex (156 units) and CA1 (224 units; 125 of the 380 are
"good", the rest "mua"), foraging in a 1 m square arena: 605 s empty, then
three novel objects placed one at a time for 450, 523 and 452 s (the data
descriptor says about 7.5 min each; the file's epochs). Head position and
direction from OptiTrack motion capture at 120 Hz, projected onto the floor.

Where it lives: EBRAINS dataset 10.25493/GTVB-FQR, CC-BY-SA-4.0, public
bucket ``d-885b4936-9345-43bd-880e-eebc19898ded`` on the EBRAINS data proxy
(no login). The novel-object session of 29.6 MB; the dataset's smallest files
are foraging-only sessions. The data descriptor lists position, head
direction, speed and spikes, and the objects' locations are not in the files
("a novel object was placed at a pseudorandom location"), so the
object-vector operation cannot run on it.

What the loader converts: ``position/animal_position`` is labelled
``"meters"`` with ``conversion=0.01``: the stored values are cm, and
``read_position`` returns metres (spans 0.98 by 0.98 m, the 1 m arena); the
loader keeps x and y and converts to cm. The reference frame says "Bottom left
of the arena", but x runs -41 to 58 cm and y -74 to 24 cm, so it is not. Head
direction (``direction/head_direction``) is radians in 0 to 2 pi, frame "(x=1,
y=0) unit vector": ``read_head_direction`` reads it; the loader keeps its
values wrapped to [-pi, pi]. Against movement faster than 20 cm/s
(``_record.heading_vs_movement``) the offset is about -0.3 rad with a spread of
about 17 degrees, larger than the survey's 0.2 rad rule, so the frame is
recorded as not verified. The offset is not constant: over the 17 windows of
60 s with at least 10 running samples (median 36), the window means range
from -0.62 to +2.33 rad (circular SD 31 degrees), so no single correction
follows from movement. The sign is tested: the heading
negated gives a spread of 81 degrees against 17. Mean
vector lengths do not depend on it. The units table repeats ids across the two
probes (380 rows, 235 distinct ids), so ``read_units`` refuses it; the loader
reads the table with pynwb and names each unit ``<area>-<id>``.

Analysis choices (``params``):

- ``bin_size`` 2.5 cm; ``min_speed`` None (foraging, as for the other open
  fields).
- ``dt`` 0.25 s, alternating 60 s ``train_epochs``/``test_epochs``.
- every unit is kept, "mua" included, as for the other datasets' unit tables.

Environment (``geometry["environment"]``): the 1 m square arena (50 cm
walls) the file's epochs table states, built with
``Environment.from_polygon`` at ``bin_size``. The stated frame ("Bottom left of
the arena") does not hold, so centre and rotation are fitted to the samples
(``_geometry.declared_rectangle``); the residual of their own fitted sides
against 100 cm is in ``conversions``.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from _fetch import fetch, open_nwb
from _geometry import declared_rectangle, with_environment
from _record import (
    Range,
    Recording,
    alternating_blocks,
    check_expectations,
    heading_vs_movement,
    validate_recording,
)

DATASET = {
    "key": "kanter_2025",
    "title": "Kanter et al. (2025), rat 29629, LEC and CA1 units, 1 m open field with novel objects",
    "source": "https://doi.org/10.25493/GTVB-FQR",
    "license": "CC-BY-SA-4.0",
    "download_size": "29.6 MB",
}

# Pre-registered from the data descriptor (OptiTrack head tracking projected on
# the floor; foraging in an empty 1 m arena, then three novel objects for
# 7.5 min each; no object positions listed among the data) and the file's
# metadata (380 unit rows, position and head direction at 120 Hz, four epochs)
# before the first run of this loader.
EXPECTATIONS: dict[str, Any] = {
    "n_units": 380,
    "n_dims": 2,
    "sampling_rate_hz": Range(119.0, 121.0),
    # No extent expectation: the loader refuses a span outside 90-105 cm
    # (_ARENA_CM), so one here could never fail; the span is in
    # ``conversions``.
    "epoch_names": ["empty", "object_1", "object_2", "object_3"],
    "headings_span_the_circle": lambda rec: bool(
        np.nanmin(rec.headings) < -3.0 and np.nanmax(rec.headings) > 3.0
    ),
    "headings_follow_movement": lambda rec: bool(
        abs(
            heading_vs_movement(rec.times, rec.positions, rec.headings, min_speed=5.0)[
                0
            ]
        )
        < np.pi / 4
    ),
}


_BUCKET = (
    "https://data-proxy.ebrains.eu/api/v1/buckets/"
    "d-885b4936-9345-43bd-880e-eebc19898ded"
)
_PATH = "data/29629/29629_2024-Oct-29_15-05-13.nwb"
_SHA256 = "3b2dd16b67303e8d632ca0960cf3f97ed037bc0c17a82e69402b2521ef47e8a2"
# The arena is 1 m square: the tracked span must lie here (cm).
_ARENA_CM = (90.0, 105.0)
_AREAS = {"LEC": "LEC", "HP - CA1": "CA1"}
_EPOCHS = ("empty", "object_1", "object_2", "object_3")


def load(cache: Path | None = None) -> Recording:
    """Fetch the session; positions in cm, head direction, spikes, object epochs."""
    from neurospatial.io.nwb import read_head_direction, read_position

    path = fetch(
        f"{_BUCKET}/{_PATH}?redirect=true",
        Path(_PATH).name,
        sha256=_SHA256,
        subdir=DATASET["key"],
        cache=cache,
    )
    with open_nwb(str(path)) as nwbfile:
        position = read_position(nwbfile, columns=[0, 1])
        heading = read_head_direction(nwbfile)
        table = nwbfile.units.to_dataframe()
        epochs = nwbfile.epochs.to_dataframe()

    positions = np.asarray(position.positions, dtype=float) * 100.0
    times = np.asarray(position.times, dtype=float)
    span = np.ptp(positions, axis=0)
    step = np.linalg.norm(np.diff(positions[::12], axis=0), axis=1)
    speed_90 = float(np.percentile(step / np.diff(times[::12]), 90))
    if (
        position.units != "m"
        or not all(_ARENA_CM[0] <= v <= _ARENA_CM[1] for v in span)
        or not 5.0 <= speed_90 <= 100.0
    ):
        msg = (
            f"position unit {position.units!r}, span {span.round(1)} cm, 90th"
            f" percentile speed {speed_90:.1f} cm/s; recheck the metres reading"
        )
        raise ValueError(msg)
    headings = np.angle(np.exp(1j * np.asarray(heading.headings, dtype=float)))
    if not np.array_equal(np.asarray(heading.times), times):
        msg = "head direction and position no longer share timestamps"
        raise ValueError(msg)
    mean, spread = heading_vs_movement(
        times[::12], positions[::12], headings[::12], min_speed=20.0
    )
    if spread > np.radians(45):
        msg = f"head direction no longer follows movement (spread {np.degrees(spread):.0f} deg)"
        raise ValueError(msg)
    areas = table["electrode_group"].map(lambda g: _AREAS[g.location]).to_numpy()
    unit_ids = np.array([f"{a}-{i}" for a, i in zip(areas, table.index, strict=True)])
    windows = epochs[["start_time", "stop_time"]].to_numpy(dtype=float)
    if len(windows) != len(_EPOCHS) or set(epochs["task"]) != {"Novel object task"}:
        msg = f"expected four novel-object epochs, got {epochs['task'].tolist()}"
        raise ValueError(msg)
    train, test = alternating_blocks(times[0], times[-1], 60.0)
    rec = Recording(
        dataset=DATASET["key"],
        session="29629_2024-Oct-29_15-05-13",
        source=f"{DATASET['source']} {_PATH}",
        license=DATASET["license"],
        times=times,
        positions=positions,
        position_units="cm",
        headings=headings,
        spike_times=tuple(
            np.sort(np.asarray(s, dtype=float)) for s in table["spike_times"]
        ),
        unit_ids=unit_ids,
        epochs={name: windows[i : i + 1] for i, name in enumerate(_EPOCHS)},
        geometry={"arena": "100 cm square, 50 cm walls", "object_positions": None},
        params={
            "bin_size": 2.5,
            "dt": 0.25,
            "train_epochs": train,
            "test_epochs": test,
            "nwb_source": str(path),
            "readers_used": ("position", "head_direction"),
            "unit_quality": table["cluster_quality"].to_numpy(),
        },
        conversions=(
            "positions: read_position(columns=[0, 1]) in m (stored cm x conversion"
            f" 0.01) x 100, cm; range checked: {positions[:, 0].min():.1f}-"
            f"{positions[:, 0].max():.1f} by {positions[:, 1].min():.1f}-"
            f"{positions[:, 1].max():.1f} cm; 90th percentile speed {speed_90:.1f} cm/s",
            "headings: read_head_direction (radians, 0-2 pi) wrapped to [-pi, pi];"
            f" frame not verified: heading_vs_movement mean {mean:+.2f} rad, spread"
            f" {np.degrees(spread):.0f} deg (10 Hz, > 20 cm/s)",
            "unit_ids: <area>-<id>; the table repeats ids across the LEC and CA1"
            f" probes ({len(table)} rows, {table.index.nunique()} distinct ids)",
            "epochs: the four novel-object epochs in order: empty, object 1-3"
            " (data descriptor); object positions are not in the file",
        ),
    )
    return with_environment(
        rec,
        *declared_rectangle(
            rec.positions,
            100.0,
            100.0,
            bin_size=rec.params["bin_size"],
            units="cm",
            name=DATASET["key"],
            source="NWB epochs table: arena_dimensions [100, 100, 50] cm, square",
        ),
        tracked_point="the head, OptiTrack motion capture projected onto the floor (per the loader, from the data descriptor)",
    )


def _units_merged_by_id(rec: Recording) -> Recording:
    # A user keying units by the table's id: the two probes' units that share
    # an id become one train.
    assert rec.spike_times is not None
    assert rec.unit_ids is not None
    plain = np.array([u.split("-", 1)[1] for u in rec.unit_ids])
    merged = {
        uid: np.sort(
            np.concatenate(
                [t for t, p in zip(rec.spike_times, plain, strict=True) if p == uid]
            )
        )
        for uid in dict.fromkeys(plain)
    }
    return replace(
        rec, spike_times=tuple(merged.values()), unit_ids=np.array(list(merged))
    )


HAZARDS = {"units_merged_by_id": _units_merged_by_id}


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
