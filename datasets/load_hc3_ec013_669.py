"""Mizuseki et al. rat ec013 CA1 and entorhinal units on a 250 cm linear track (CRCNS hc-3, session ec013.669).

What was recorded: rat ec013, an 8-shank silicon probe in dorsal CA1 (shanks
5-8) and entorhinal cortex layers 2-5 (shanks 1-4), running back and forth on
a 250 cm linear track for 13 min. Two head-mounted LEDs tracked by video at
39.06 Hz.

Where it lives: CRCNS hc-3, which needs a free crcns.org account
(``CRCNS_USERNAME``/``CRCNS_PASSWORD``). Each session is one
``<topdir>/<session>.tar.gz`` in Neuroscope format; this one is
``ec013.38/ec013.669.tar.gz`` (233.8 MB, almost all LFP and waveforms). The
loader streams it once and keeps ``.res.N``, ``.clu.N``, ``.whl`` and
``.xml`` (5 MB). Units are spike-sorted per top-level directory (all
sessions of a day together), and the data set's cell table
(``docs/crcns-hc3-metadata-tables.zip``, ``hc3-cell.csv``) lists the
sorted cells that passed the authors' quality check, with region and type.

What the loader converts:

- ``.res.N`` holds spike times in samples of the wideband clock the
  ``.xml`` states (20 kHz); divided by it, they are seconds.
- Units are the cell table's (electrode, cluster) pairs for topdir
  ``ec013.38``; clusters 0 (artefact) and 1 (unsortable) are never units.
- ``.whl`` row k is the video frame at k * 512 / 20000 s; ``-1`` marks an
  untracked LED and becomes NaN. Position is the midpoint of the two LEDs
  where both are tracked, NaN elsewhere, kept in video pixels: the data set
  states no scale. The track is diagonal in the image.
- Head direction is the angle from the back LED to the front LED. The data
  set does not say which LED is in front; the order whose heading follows
  the direction of movement while running is used (``heading_vs_movement``
  must give a mean within 0.2 rad and a spread under 45 deg). Frames where
  both LEDs have identical coordinates (one blob given to both) have no
  direction and are NaN; ``arctan2(0, 0)`` would call them 0 (east).
- Headings are in the frame of the pixel coordinates as stored. Whether the
  .whl y axis points down the video image, which would mirror the room, is
  not stated, so left and right turns are not interpreted.

Analysis choices (``params``): the pixel scale is inferred from the stated
apparatus (250 cm over the tracked span along the track's long axis) and is
used only to choose ``bin_size`` (about 2.5 cm) and ``min_speed`` (4 cm/s)
in pixels; ``dt`` 0.25 s with alternating 60 s train/test blocks.

Declared environment (``geometry["environment"]``): the 250 x 7 cm track
(Mizuseki et al. 2012, Behavioral Testing) as a one-edge graph in pixels at
the inferred scale, along the samples' principal axis and centred on them.
Its length is not an independent check: the scale comes from the same 250 cm.
"""

from __future__ import annotations

import csv
import io
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from _fetch import fetch_crcns, fetch_crcns_tar_members
from _formats import (
    neuroscope_rates,
    read_res_clu,
    read_whl,
    two_led_heading,
    whl_times,
)
from _record import (
    Range,
    Recording,
    alternating_blocks,
    check_expectations,
    checked_span,
    heading_vs_movement,
    validate_recording,
)
from _tracks import (
    declared_geometry,
    straight_placement,
    straight_track,
    with_declared,
)

DATASET = {
    "key": "hc3_ec013_669",
    "title": "Mizuseki et al., rat ec013 session ec013.669, CA1 and EC units on a 250 cm linear track (CRCNS hc-3)",
    "source": "https://crcns.org/data-sets/hc/hc-3",
    "license": "CRCNS terms of use (account required; cite the data set)",
    "download_size": "5.5 MB of a 233.8 MB archive, plus the 1.1 MB cell table",
}

# Pre-registered from the hc-3 metadata tables (topdir ec013.38: 125 sorted
# cells; session ec013.669, behavior "linear", 798.3 s), the original docs'
# README ("linear: linear maze, 250 cm") and the hc-2 description of the .whl
# file (39.06 Hz, two LEDs, -1 where untracked, no unit stated), before the
# first run of this loader. The .whl values are expected to be video pixels.
EXPECTATIONS: dict[str, Any] = {
    "n_units": Range(100, 125),
    "n_dims": 2,
    "sampling_rate_hz": Range(38.5, 39.5),
    "duration_s": Range(780.0, 800.0),
    # No extent expectation: the loader refuses a span outside 150-400 px
    # (checked_span), inside the 100-720 px once expected here, so that
    # expectation could never fail; the span is in ``conversions``.
    "has_headings": lambda rec: rec.headings is not None
    and bool(np.mean(np.isfinite(rec.headings)) > 0.5),
}


SUBDIR = "crcns_hc3"
ARCHIVE = "hc-3/ec013.38/ec013.669.tar.gz"
MEMBER_DIR = "ec013.38/ec013.669"
BASE = "ec013.669"
TOPDIR = "ec013.38"
N_SHANKS = 8
TRACK_CM = 250.0
TRACK_WIDTH_CM = 7.0
RUNNING_CM_S = 10.0
_SHA256 = {
    "clu.1": "ca742885578da8600369e0f38ef160a91409e38cb6d24c700ea9278c2904ebcd",
    "clu.2": "74494191a1672230069d0046549df07be36034889171e315b1ee445df88c241b",
    "clu.3": "73c2b078325f29afd4f6b1cad6ae2230221d76747cda0609d42752901dec9d16",
    "clu.4": "101e5a29fac41ea95827485674033996f440d0dece09aac1e561f5c20278675e",
    "clu.5": "717632b9badcfc16be8a62d434e4d8ec7f600ad20426fadd58ef113ece54197d",
    "clu.6": "52b3dd20654586cb2ac6e878708ef5152e52dcb60bf59fc56b0ab00ffdf29b4f",
    "clu.7": "ca742ae46dd15b35de6a2f94a1535d04e445b82374f94c9ce50bf6d8442e8f5c",
    "clu.8": "a50551794ea92d314fca2f852543969a23b6b6da1c7836869f77d0417bd82d63",
    "res.1": "2bb3dfd5d7b8d736d2fdfef07918b67f27d26cc737bf733d062358d31cedcbf3",
    "res.2": "ab62306c5a47867b9a0988774eb50cc938a99906742b559c6b85de051dc0c826",
    "res.3": "20eaba43420ffe9eb0c9f830eff1d81e269e49b5027c8746afff9c25dcb2ea38",
    "res.4": "43d3d1177daabefb4f82bd34fe71c0915f2063a769331ab5f0cf2fb952ad3cb7",
    "res.5": "dfefad73ea895112d191c1acc5874336d064a93a4d4c93a67463d308cbd9fe39",
    "res.6": "bfd658619f1c70d84955a8badbf6f2d81fefa6af8eef02e00f4a3d4e3f3e3fee",
    "res.7": "0050ebc63ff25a87819b4dcf16cd0f5dbdf6cb7daf80b7887dbe05c5fccc28fd",
    "res.8": "97ec239f21dbd9e756ceb3d4795930ff811e92308978ed5b240359c93a8852f7",
    "whl": "52afdd3178ea125ffbe308025ad2d088de286dff3a5b14bf866fb3d4cdbebaeb",
    "xml": "fb341581df54ad44d5a54e4128318e7dcb71121b22d1ec701f6b7e0c8ab1246b",
}
CELL_TABLE = "hc-3/docs/crcns-hc3-metadata-tables.zip"
_CELL_TABLE_SHA256 = "ce26a636a24f35589dc3ade47722e06943b1b73fb6abd44aab111f7cbf75be1f"
_CELL_COLUMNS = ("id", "topdir", "animal", "ele", "clu", "region")


def fetch_session(
    archive: str,
    member_dir: str,
    base: str,
    pins: dict[str, str],
    *,
    subdir: str,
    cache: Path | None,
) -> dict[str, Path]:
    """The session's Neuroscope files, keyed by extension (``"res.1"``, ``"whl"``)."""
    members = {f"{member_dir}/{base}.{ext}": sha for ext, sha in pins.items()}
    paths = fetch_crcns_tar_members(archive, members, subdir=subdir, cache=cache)
    return {ext: paths[f"{member_dir}/{base}.{ext}"] for ext in pins}


def cell_table(topdir: str, cache: Path | None = None) -> list[dict[str, str]]:
    """Rows of hc-3's ``hc3-cell.csv`` (no header) for one top-level directory."""
    path = fetch_crcns(
        CELL_TABLE, sha256=_CELL_TABLE_SHA256, subdir=SUBDIR, cache=cache
    )
    with zipfile.ZipFile(path) as archive:
        text = archive.read("hc3-metadata-tables/hc3-cell.csv").decode()
    rows = [
        dict(zip((*_CELL_COLUMNS, "cell_type"), (*row[:6], row[-1]), strict=True))
        for row in csv.reader(io.StringIO(text))
    ]
    return [row for row in rows if row["topdir"] == topdir]


def led_track(
    frames: np.ndarray, times: np.ndarray, *, min_speed: float
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Head position and direction from the two LEDs, with the LED order checked.

    Position is the LEDs' midpoint where both are tracked. The order whose
    front-minus-back angle follows the direction of movement above
    ``min_speed`` is used, and refused unless it passes the frame check.
    Frames with both LEDs at the same coordinates have NaN direction
    (``two_led_heading``).
    """
    first, second = frames[:, :2], frames[:, 2:4]
    positions = (first + second) / 2.0
    coincident = np.all(first == second, axis=1)
    checks = {}
    for order, (front, back) in {
        "first": (first, second),
        "second": (second, first),
    }.items():
        headings = two_led_heading(front, back)
        offset, spread = heading_vs_movement(
            times, positions, headings, min_speed=min_speed
        )
        checks[order] = (headings, offset, spread)
    order = min(checks, key=lambda k: abs(checks[k][1]))
    headings, offset, spread = checks[order]
    if abs(offset) > 0.2 or spread > np.radians(45):
        msg = (
            f"LED heading minus the direction of movement: mean {offset:.2f} rad,"
            f" spread {spread:.2f} rad with the {order} LED in front; recheck the"
            " LED columns and their frame"
        )
        raise ValueError(msg)
    tracked = np.isfinite(positions).all(axis=1)
    info = {
        "front": order,
        "offset": offset,
        "spread": spread,
        "other_offset": checks["second" if order == "first" else "first"][1],
        "coincident_share": float(coincident[tracked].mean()),
    }
    return positions, headings, info


def long_axis_span(positions: np.ndarray) -> float:
    """Span of the tracked positions along their first principal axis."""
    tracked = positions[np.isfinite(positions).all(axis=1)]
    centred = tracked - tracked.mean(axis=0)
    axis = np.linalg.svd(centred, full_matrices=False)[2][0]
    return float(np.ptp(centred @ axis))


def neuroscope_recording(
    files: dict[str, Path],
    *,
    dataset: dict[str, str],
    session: str,
    source: str,
    n_shanks: int,
    units: dict[tuple[int, int], dict[str, str]] | None,
    apparatus_cm: float,
    apparatus_span_px: Any,
    span_px: tuple[float, float],
    geometry: dict[str, Any],
) -> Recording:
    """Recording from a Neuroscope session tracked with two LEDs, in pixels.

    ``units`` maps (shank, cluster) to the source's cell-table row; None keeps
    every cluster from 2 up. ``apparatus_span_px`` measures the span in
    pixels that ``apparatus_cm`` corresponds to; it sets the inferred scale.
    """
    rates = neuroscope_rates(files["xml"])
    wideband = rates["wideband_hz"]
    if wideband != 20000.0:
        msg = f"the .xml states a {wideband} Hz wideband clock, not 20 kHz"
        raise ValueError(msg)
    frames = read_whl(files["whl"])
    times = whl_times(len(frames), wideband)
    midpoint = (frames[:, :2] + frames[:, 2:4]) / 2.0
    cm_per_px = apparatus_cm / apparatus_span_px(midpoint)
    positions, headings, leds = led_track(
        frames, times, min_speed=RUNNING_CM_S / cm_per_px
    )
    span = checked_span(positions, *span_px, units="pixel")
    trains, ids, kept, found = [], [], {}, 0
    for shank in range(1, n_shanks + 1):
        samples, clusters = read_res_clu(files[f"res.{shank}"], files[f"clu.{shank}"])
        for cluster in np.unique(clusters[clusters >= 2]):
            found += 1
            key = (shank, int(cluster))
            if units is not None and key not in units:
                continue
            trains.append(np.sort(samples[clusters == cluster] / wideband))
            ids.append(f"{shank}.{cluster}")
            kept[key] = (units or {}).get(key, {})
    missing = sorted(set(units or {}) - set(kept))
    train, test = alternating_blocks(times[0], times[-1], 60.0)
    tracked = np.isfinite(positions).all(axis=1)
    return Recording(
        dataset=dataset["key"],
        session=session,
        source=source,
        license=dataset["license"],
        times=times,
        positions=positions,
        position_units="pixel",
        headings=headings,
        spike_times=tuple(trains),
        unit_ids=np.asarray(ids),
        epochs={"session": np.array([[times[0], times[-1]]])},
        geometry={**geometry, "inferred_cm_per_pixel": cm_per_px},
        params={
            "bin_size": round(2.5 / cm_per_px, 2),
            "min_speed": round(4.0 / cm_per_px, 2),
            "dt": 0.25,
            "whl_source": str(files["whl"]),
            "train_epochs": train,
            "test_epochs": test,
            "region": np.array([row.get("region", "") for row in kept.values()]),
            "cell_type": np.array([row.get("cell_type", "") for row in kept.values()]),
        },
        conversions=(
            f"times: .whl row k at k * 512 / {wideband:g} s ({len(frames)} frames)",
            f"positions: midpoint of the two LEDs where both are tracked ({tracked.mean():.1%}"
            f" of frames), -1 -> NaN; video pixels as stored; {span}",
            f"scale: {cm_per_px:.3f} cm per pixel inferred from the stated {apparatus_cm:g} cm"
            f" apparatus; used only for bin_size {2.5 / cm_per_px:.2f} px and min_speed"
            f" {4.0 / cm_per_px:.2f} px/s",
            f"headings: {leds['front']} LED in front (back-to-front angle minus movement"
            f" direction above {RUNNING_CM_S:g} cm/s: circular mean"
            f" {np.degrees(leds['offset']):.1f} deg, median spread"
            f" {np.degrees(leds['spread']):.1f} deg; the other order gives"
            f" {np.degrees(leds['other_offset']):.1f} deg); {leds['coincident_share']:.1%}"
            " of tracked frames have both LEDs at one point -> NaN; frame of the"
            " stored pixel axes (y direction unstated)",
            f"spikes: .res samples / {wideband:g} Hz; {len(trains)} units kept of {found}"
            f" clusters >= 2"
            + (
                f" (the cell table's units; {len(missing)} listed units have no spikes"
                " in this session)"
                if units is not None
                else ""
            ),
        ),
    )


def load(cache: Path | None = None) -> Recording:
    """Fetch the session; positions in pixels, two-LED headings, the cell table's units."""
    files = fetch_session(
        ARCHIVE, MEMBER_DIR, BASE, _SHA256, subdir=SUBDIR, cache=cache
    )
    units = {
        (int(row["ele"]), int(row["clu"])): row for row in cell_table(TOPDIR, cache)
    }
    rec = neuroscope_recording(
        files,
        dataset=DATASET,
        session=f"{TOPDIR}/{BASE}",
        source=f"{DATASET['source']} {ARCHIVE}",
        n_shanks=N_SHANKS,
        units=units,
        apparatus_cm=TRACK_CM,
        apparatus_span_px=long_axis_span,
        span_px=(150.0, 400.0),
        geometry={"shape": "linear track", "length_cm": TRACK_CM},
    )
    cm_per_px = rec.geometry["inferred_cm_per_pixel"]
    graph, fit = straight_track(rec.positions, TRACK_CM / cm_per_px)
    entries, line = declared_geometry(
        graph,
        rec.positions,
        times=rec.times,
        kind=f"straight track {TRACK_CM:g} x {TRACK_WIDTH_CM:g} cm",
        source="Mizuseki et al. 2012 Behavioral Testing: 'the linear track (250 x 7"
        " cm2)'; in pixels at the inferred scale, which comes from the same 250 cm",
        bin_size=rec.params["bin_size"],
        physical_width=TRACK_WIDTH_CM / cm_per_px,
        units="pixel",
        name=DATASET["key"],
        placement=straight_placement(fit, "pixel"),
        # The pixel scale comes from the same 250 cm, so no length is compared.
        placed_by="samples' principal axis",
        tracked_point="the midpoint of two head-mounted LEDs (hc-3 .whl: two LEDs per frame)",
        speed_thresholds=(4.0 / cm_per_px, 30.0 / cm_per_px),
    )
    return with_declared(rec, entries, line)


def coincident_leds_as_east(rec: Recording) -> Recording:
    # arctan2(0, 0) == 0: a reader that keeps frames with both LEDs at one point.
    assert rec.headings is not None
    assert rec.positions is not None
    tracked = np.isfinite(rec.positions).all(axis=1)
    return replace(
        rec, headings=np.where(tracked & np.isnan(rec.headings), 0.0, rec.headings)
    )


def leds_swapped(rec: Recording) -> Recording:
    # The other LED taken as the front one: every heading turned by pi.
    assert rec.headings is not None
    return replace(rec, headings=np.angle(np.exp(1j * (rec.headings + np.pi))))


def whl_minus_one_kept(rec: Recording) -> Recording:
    """A reader that keeps the .whl's -1 marker and averages the two LEDs.

    Where both LEDs are lost the midpoint is (-1, -1); where one is lost it
    is halfway between the other LED and (-1, -1), a plausible-looking
    position inside the image.
    """
    raw = np.loadtxt(rec.params["whl_source"], dtype=float, ndmin=2)
    return replace(rec, positions=(raw[:, :2] + raw[:, 2:4]) / 2.0)


HAZARDS = {
    "coincident_leds_as_east": coincident_leds_as_east,
    "leds_swapped": leds_swapped,
    "whl_minus_one_kept": whl_minus_one_kept,
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
