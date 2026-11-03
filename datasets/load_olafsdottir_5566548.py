"""Olafsdottir, Carpenter & Barry 2016 rat hippocampus and deep MEC on a Z-track (Zenodo 5566548).

What was recorded: rat R2142 on 2014-08-06, 16 tetrodes (Axona DACQ):
tetrodes 1-8 in hippocampus and 9-16 in deep medial entorhinal cortex (the
record's README: R2142 is the reverse of the other rats), running a Z-shaped
track for 38 minutes. One tracking LED at 50 Hz.

Where it lives: Zenodo 5566548 (CC BY 4.0), ``Olafsdottir2016.zip``
(9.9 GB, members stored uncompressed); 34 members of
``r2142/2014-08-06/20140806_R2142_track1`` are read by HTTP byte range: the
``.pos`` and ``.set`` files, the 16 tetrode files ``.1``-``.16`` and their
``_N.cut`` files (Tint's cluster assignments; the README says to use these,
not the ``.clu`` files).

Reading the files: the Axona formats are parsed here with NumPy, following
Axona's "dacqUSB Data File Formats" (Axona Ltd; archived copy of
space-memory-navigation.org/DacqUSBFileFormats.pdf, 2018-09-03):

- every data file is an ASCII header of ``key value`` lines ending in
  ``data_start``, then binary data, then ``\\r\\ndata_end\\r\\n``;
- ``.pos``: 20-byte records, a 4-byte frame counter then eight big-endian
  16-bit words (two-spot mode: big spot x, y, little spot x, y, pixel counts,
  total, unused); 1023 marks an untracked spot. The frame counter is not a
  timestamp: sample X is at X / 50 s from the start (the document's note);
- tetrode files: 216-byte spikes, four channels of a 4-byte big-endian
  timestamp and 50 int8 samples; timestamps / ``timebase`` (96 kHz) are
  seconds;
- ``.cut`` (Tint, not in that document): text whose ``Exact_cut_for: ...
  spikes: N`` line is followed by N cluster numbers, one per spike in file
  order; cluster 0 is unassigned.

What the loader converts:

- Positions: big-spot x, y (pixels) / ``pixels_per_metre`` (390) x 100 -> cm;
  1023 -> NaN. The stored coordinates are relative to the tracking window
  (y spans 4-529 pixels while the header's window runs 27-562), and image y
  points down; y is flipped within the window (y = (window_max_y -
  window_min_y) - y) so the frame is right-handed with +y north on the
  image. ``pos_format`` must read ``t,x1,y1,x2,y2,numpix1,numpix2``, the
  order the parser assumes. Range checked.
- Times: sample index / 50 Hz (``sample_rate`` in the header).
- Units: one per tetrode and cluster >= 1; spike times are the first
  channel's timestamps / 96 kHz.

Analysis choices (``params``): ``bin_size`` 4 cm; ``min_speed`` 5 cm/s;
``dt`` 0.25 s and alternating 60 s train/test blocks; ``region`` per unit.

Declared environment (``geometry["environment"]``): the Z-track (Olafsdottir
et al. 2016, Online Methods: 10 cm wide runways, two parallel 190 cm
sections joined by a 220 cm diagonal) as a path of three segments whose four
nodes are fitted to the samples, with the stated 10 cm width. The fitted
sections are shorter than stated (``conversions``), so lengths come from the
samples, not the paper.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from _fetch import fetch_zip_member
from _record import (
    Range,
    Recording,
    alternating_blocks,
    check_expectations,
    checked_span,
    validate_recording,
)
from _tracks import (
    declared_geometry,
    edge_lengths,
    fitted_maze,
    with_declared,
    z_track_candidates,
)

DATASET = {
    "key": "olafsdottir_5566548",
    "title": "Olafsdottir et al. (2016), rat R2142, 2014-08-06, Z-track, 16 tetrodes",
    "source": "https://doi.org/10.5281/zenodo.5566548",
    "license": "CC-BY-4.0",
    "download_size": "105 MB of a 9.9 GB zip (byte-range reads)",
}

# Pre-registered from the README (.pos at 50 Hz, .cut per tetrode) and the
# .pos/.1 headers (duration 2276 s, 390 pixels per metre, window 717 x 535
# pixels, one tracked LED) before the first run of this loader.
EXPECTATIONS: dict[str, Any] = {
    "n_dims": 2,
    "n_units": Range(10, 150),
    "sampling_rate_hz": Range(49.9, 50.1),
    "duration_s": Range(2270.0, 2280.0),
    "inside_the_tracking_window_cm": lambda rec: Range(0.0, 190.0).contains(
        rec.geometry["position_span_cm"]
    ),
    "one_led_no_heading": lambda rec: rec.headings is None,
}

_ZIP = "https://zenodo.org/api/records/5566548/files/Olafsdottir2016.zip/content"
_FOLDER = "r2142/2014-08-06"
_TRIAL = "20140806_R2142_track1"
_FILES = {
    "20140806_R2142_track1.pos": "e8e98c9b794804d18a780bce99cb3fde054e7dfa682b3fd12fc672481e36697f",
    "20140806_R2142_track1.set": "c944fe566444841fd085c3aaa2a003912aa4ccbd9816c54c4874e736062e030d",
    "20140806_R2142_track1.1": "480f7b25d3d68d8d2333d550afea4163f832429fbc806d35d10792c63d54128f",
    "20140806_R2142_track1.2": "78602c7e9acfbec6097483b5d4fa0c4b8c0f5ced44809514556b2d129708986d",
    "20140806_R2142_track1.3": "3ffef734707a1230b52be3d1a15c0de8ef13a02c8ef61c56b5636b59252f339d",
    "20140806_R2142_track1.4": "f0703b4bcc3f30de4d14e4e4532508c03c89d705055e1c0cf2f40b11fad1534d",
    "20140806_R2142_track1.5": "0b5513ed3a04e98dced41d4798134968ea242cd08060ad5d70b3f0593dd4e910",
    "20140806_R2142_track1.6": "99ca3b98aeb542dd4ce73a161e84f43262eb76cced3df4eae9ff5737d24aeaab",
    "20140806_R2142_track1.7": "25076a1d95958f0119b13fcf4cb6af8fbc487ebde4a78f88ac5d8650436710ac",
    "20140806_R2142_track1.8": "decb51a585392aeeb4f3c0158174defe164d865055ab77bc24a31e518cd96553",
    "20140806_R2142_track1.9": "d50990d7f2886159ce50898b6ef97427c7321911434d9bc7a87967b6a11fdc35",
    "20140806_R2142_track1.10": "85b079c5ddeb298102d56c986e61c37c8eb5518520e46a088224437e85c77131",
    "20140806_R2142_track1.11": "eca261684c6804b38d610d6ee46ec25e94cda205046e0f52365b53f96059502a",
    "20140806_R2142_track1.12": "33db585885cf221be00aa3adf75eed236ff3803f0fbc86908e72a8d81d7f8603",
    "20140806_R2142_track1.13": "4ea1356c9881e4471f1b6f33f7ece2fdc68e0496442c5be0c4f9d8572a059d7f",
    "20140806_R2142_track1.14": "23996e022560c1f837da6b522b9fdffeb9b3a5520bb29bcfed32f5f03a90e7eb",
    "20140806_R2142_track1.15": "96e37133a37457648027ad9ea4a26513a670fdcc0efebc748f8df58eebe2d6e9",
    "20140806_R2142_track1.16": "dae673e9a6425c84449e21d2ab92547a85e92d3e68b79adda705a2a2f057db92",
    "20140806_R2142_track1_1.cut": "484b1f6ee5fae973b2f1afb0cd2b613b964e544c56244e13f5fcc0bae145089e",
    "20140806_R2142_track1_2.cut": "0af02ff8ec4f661f7952d03b3d0ac66a052490722e7c92da206e9f9ae23587d9",
    "20140806_R2142_track1_3.cut": "840b429d8414d22a24eb435ef91f1ad4aca7d7f7e0fb1b81e3dd62faf72c4536",
    "20140806_R2142_track1_4.cut": "e1827991df647af51062d603eb61d97eeb6b3262d3961fde8a227b4d5ef252f5",
    "20140806_R2142_track1_5.cut": "3f9e2062850499ef4973185650eb2d73ec5481347100fd67f8c38d52b906fac3",
    "20140806_R2142_track1_6.cut": "95bbaa062b603604039a9ca2e3877ab7e686e26302fc36493bef6a00b05dbbb8",
    "20140806_R2142_track1_7.cut": "fc9aa991abdd6b244e9d16c989c9962c0c935f4b1e98749c8d629cb3e9996841",
    "20140806_R2142_track1_8.cut": "d10ac3cda2de817fd422d1cdaeedb002d7bc3a3d0a9d5aebebe55956c9aecb0f",
    "20140806_R2142_track1_9.cut": "c1f28192d8e76e6f986e7960873f0a99eeca4778c7277255971cbfcc05e7903c",
    "20140806_R2142_track1_10.cut": "0e6214e5a530632586851f09e400aed3b24fcbf3f80acfda96a344c553c83da1",
    "20140806_R2142_track1_11.cut": "53f14c12cb071e479ad0cf666a853c606fb67a1b230bdb4d71e04ff6bc4b625c",
    "20140806_R2142_track1_12.cut": "2c9c30b913cc3b49a6ee5595bb481874e195e335b967d886a8b6260f9d58da9d",
    "20140806_R2142_track1_13.cut": "a07819001bb2811081c68a59a4342c92243803c13d120be7dd7b079e76adb6e1",
    "20140806_R2142_track1_14.cut": "ca852243e69939315f433401b310971372e384756f91978d219e275994808a45",
    "20140806_R2142_track1_15.cut": "1e91b0125a8235c1efff3ebdc615fe84f1fd79a1fe58eb396633fb8fbea389ff",
    "20140806_R2142_track1_16.cut": "2a159dd01fd5b22fb85c8434f846dd17cc37b8461be81194b29fda021c76addd",
}
_N_TETRODES = 16
_HIPPOCAMPAL_TETRODES = range(1, 9)
_SPAN_CM = (100.0, 190.0)
_RUNWAY_WIDTH_CM = 10.0
_FIT_CAP_CM = 10.0
_UNTRACKED = 1023
_DATA_START = b"data_start"
_DATA_END = b"\r\ndata_end\r\n"
POS_RECORD_BYTES = 20
SPIKE_RECORD_BYTES = 216


def read_header(blob: bytes) -> tuple[dict[str, str], bytes]:
    """Split an Axona data file into its header and its binary data.

    Returns
    -------
    header : dict
        ``key value`` lines before ``data_start`` (values as text, stripped).
    data : bytes
        The bytes between ``data_start`` and the ``\\r\\ndata_end\\r\\n``
        marker (or the end of the file if the marker is missing).
    """
    start = blob.index(_DATA_START)
    header = {}
    for line in blob[:start].decode("latin-1").splitlines():
        key, _, value = line.strip().partition(" ")
        if key:
            header[key] = value.strip()
    data = blob[start + len(_DATA_START) :]
    data = data.removesuffix(_DATA_END)
    return header, data


def _rate(text: str) -> float:
    return float(text.split()[0])


def read_pos(blob: bytes) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    """Positions of an Axona ``.pos`` file.

    Returns
    -------
    times : ndarray, shape (n,)
        Sample index / ``sample_rate`` (s).
    spots : ndarray, shape (n, 4)
        Big-spot x, y and little-spot x, y in pixels, NaN where untracked
        (1023).
    header : dict
    """
    header, data = read_header(blob)
    n = int(header["num_pos_samples"])
    if int(header["bytes_per_timestamp"]) != 4 or int(header["bytes_per_coord"]) != 2:
        msg = f"unexpected .pos layout: {header['bytes_per_timestamp']}-byte stamps"
        raise ValueError(msg)
    if len(data) != n * POS_RECORD_BYTES:
        msg = f".pos holds {len(data)} bytes, not {n} x {POS_RECORD_BYTES}"
        raise ValueError(msg)
    record = np.dtype([("frame", ">u4"), ("words", ">u2", 8)])
    words = np.frombuffer(data, dtype=record, count=n)["words"]
    spots = words[:, :4].astype(float)
    spots[spots == _UNTRACKED] = np.nan
    times = np.arange(n) / _rate(header["sample_rate"])
    return times, spots, header


def read_spike_times(blob: bytes) -> np.ndarray:
    """Spike times (s) of an Axona tetrode file: first channel's stamps / timebase."""
    header, data = read_header(blob)
    n = int(header["num_spikes"])
    if len(data) != n * SPIKE_RECORD_BYTES:
        msg = f"tetrode file holds {len(data)} bytes, not {n} x {SPIKE_RECORD_BYTES}"
        raise ValueError(msg)
    channel = np.dtype([("stamp", ">u4"), ("samples", "i1", (50,))])
    channels = np.frombuffer(data, dtype=channel, count=4 * n).reshape(n, 4)
    return channels[:, 0]["stamp"].astype(float) / _rate(header["timebase"])


def read_cut(text: str) -> np.ndarray:
    """Cluster number of every spike in a Tint ``.cut`` file, in file order."""
    _, _, body = text.partition("Exact_cut_for:")
    n = int(body.split("spikes:")[1].split()[0])
    clusters = np.array(body.split("spikes:")[1].split()[1:], dtype=int)
    if clusters.size != n:
        msg = f".cut lists {clusters.size} clusters for {n} spikes"
        raise ValueError(msg)
    return clusters


def _member(name: str, cache: Path | None) -> bytes:
    path = fetch_zip_member(
        _ZIP,
        f"{_FOLDER}/{name}",
        sha256=_FILES[name],
        subdir=DATASET["key"],
        cache=cache,
    )
    return path.read_bytes()


def load(cache: Path | None = None) -> Recording:
    """Fetch the Z-track session; positions in cm, units per tetrode cluster."""
    times, spots, header = read_pos(_member(f"{_TRIAL}.pos", cache))
    ppm = float(header["pixels_per_metre"])
    if header.get("pos_format") != "t,x1,y1,x2,y2,numpix1,numpix2":
        msg = f"unexpected pos_format {header.get('pos_format')!r}"
        raise ValueError(msg)
    window_height = float(header["window_max_y"]) - float(header["window_min_y"])
    pixels = spots[:, :2].copy()
    untracked = np.isnan(pixels).any(axis=1)
    if np.nanmax(pixels[:, 1]) > window_height:
        msg = "y exceeds the tracking window's height; recheck the window frame"
        raise ValueError(msg)
    pixels[:, 1] = window_height - pixels[:, 1]
    positions = pixels / ppm * 100.0
    span = checked_span(positions, *_SPAN_CM)
    spikes, ids, regions, n_unassigned = [], [], [], 0
    for tetrode in range(1, _N_TETRODES + 1):
        stamps = read_spike_times(_member(f"{_TRIAL}.{tetrode}", cache))
        clusters = read_cut(_member(f"{_TRIAL}_{tetrode}.cut", cache).decode("latin-1"))
        if clusters.size != stamps.size:
            msg = f"tetrode {tetrode}: {stamps.size} spikes but {clusters.size} cut labels"
            raise ValueError(msg)
        n_unassigned += int(np.sum(clusters == 0))
        for cluster in np.unique(clusters[clusters > 0]):
            spikes.append(np.sort(stamps[clusters == cluster]))
            ids.append(f"t{tetrode}c{cluster}")
            regions.append("HPC" if tetrode in _HIPPOCAMPAL_TETRODES else "MEC")
    train, test = alternating_blocks(times[0], times[-1], 60.0)
    rec = Recording(
        dataset=DATASET["key"],
        session=_TRIAL,
        source=f"{DATASET['source']} {_FOLDER}/{_TRIAL}",
        license=DATASET["license"],
        times=times,
        positions=positions,
        position_units="cm",
        spike_times=tuple(spikes),
        unit_ids=np.asarray(ids),
        geometry={
            "position_span_cm": np.nanmax(positions, 0) - np.nanmin(positions, 0),
            "tracking_window_px": [
                float(header["window_min_x"]),
                float(header["window_max_x"]),
                float(header["window_min_y"]),
                float(header["window_max_y"]),
            ],
        },
        params={
            "bin_size": 4.0,
            "min_speed": 5.0,
            "dt": 0.25,
            "train_epochs": train,
            "test_epochs": test,
            "region": np.asarray(regions),
        },
        conversions=(
            f"positions: big spot x, y (pixels) / {ppm:g} pixels per metre x 100 -> cm",
            f"positions: y flipped within the window (y = {window_height:g} - y)"
            " to point up",
            f"positions: {untracked.mean():.1%} of samples untracked (1023) -> NaN",
            span,
            f"times: sample index / {header['sample_rate']} (.pos header)",
            "spike_times: first channel's 4-byte timestamps / 96000 (timebase)",
            f"units: tetrode x .cut cluster >= 1; {n_unassigned} spikes in cluster 0"
            " (unassigned) left out",
        ),
    )
    graph = fitted_maze(z_track_candidates(positions), positions, cap=_FIT_CAP_CM)
    entries, line = declared_geometry(
        graph,
        positions,
        times=times,
        kind="Z-track: two parallel sections joined by a diagonal",
        source="Olafsdottir et al. 2016 Online Methods: '10 cm wide runways', 'two"
        " parallel sections of the Z (190 cm each) were connected by a diagonal"
        " section (220 cm)'",
        bin_size=4.0,
        physical_width=_RUNWAY_WIDTH_CM,
        units="cm",
        name=DATASET["key"],
        placed_by="fitted to the samples",
        stated_lengths={
            name: (graph.edges[u, v]["distance"], stated)
            for name, (u, v), stated in (
                ("first section", ("end_0", "turn_0"), 190.0),
                ("diagonal", ("turn_0", "turn_1"), 220.0),
                ("last section", ("turn_1", "end_1"), 190.0),
            )
        },
        placement="node positions fitted to the samples; sections (cm):"
        f" {edge_lengths(graph)} (stated 190, 220, 190)",
        tracked_point="the big LED spot of Axona two-spot tracking (the .pos header)",
        speed_thresholds=(5.0, 30.0),
    )
    return with_declared(rec, entries, line)


def pixels_labelled_cm(rec: Recording) -> Recording:
    # Positions left in pixels and called cm: 3.9 x too large.
    assert rec.positions is not None
    return replace(rec, positions=rec.positions * 3.9)


def image_y_down(rec: Recording) -> Recording:
    # The y flip undone: the image frame, mirrored.
    assert rec.positions is not None
    positions = rec.positions.copy()
    positions[:, 1] = 535.0 / 3.9 - positions[:, 1]
    return replace(rec, positions=positions)


HAZARDS = {"pixels_labelled_cm": pixels_labelled_cm, "image_y_down": image_y_down}


if __name__ == "__main__":
    rec = load()
    validate_recording(rec)
    duration = rec.times[-1] - rec.times[0]
    print(f"{rec.dataset}: {len(rec.spike_times or ())} units, {duration:.0f} s")
    for row in check_expectations(rec, EXPECTATIONS):
        print(f"  {'pass' if row['passed'] else 'FAIL'} {row['key']}: {row['detail']}")
    for line in rec.conversions:
        print(f"  conversion: {line}")
