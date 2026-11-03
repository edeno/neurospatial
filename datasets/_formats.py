"""Readers for file formats several loaders share, and two-LED head direction.

Neuroscope (Buzsaki lab, CRCNS hc-2, hc-3, hc-11, hc-18) stores each session
as ``<base>.res.<shank>`` (spike times in samples), ``<base>.clu.<shank>``
(cluster of each spike; first line the number of clusters; 0 is artefact
and 1 unsortable), ``<base>.whl`` (one tracking frame per line) and
``<base>.xml`` (the sampling rates). The ``.whl`` frame rate is the wideband
rate / 512, 39.0625 Hz at 20 kHz, so frame k is at k * 512 / 20000 s.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

WHL_SAMPLES_PER_FRAME = 512


def read_whl(path: Path) -> NDArray[np.float64]:
    """Tracking frames of a ``.whl`` file, shape (n_frames, n_columns).

    Each LED is an (x, y) pair of columns; ``-1`` in either coordinate means
    the LED was not tracked in that frame, and both of its coordinates become
    NaN. Values are kept in the file's unit (video pixels unless the source
    says otherwise).
    """
    frames = np.loadtxt(path, dtype=float, ndmin=2)
    if frames.shape[1] % 2:
        msg = f"{path} has {frames.shape[1]} columns; expected (x, y) pairs"
        raise ValueError(msg)
    for led in range(frames.shape[1] // 2):
        pair = frames[:, 2 * led : 2 * led + 2]
        pair[(pair == -1).any(axis=1)] = np.nan
    return frames


def whl_times(n_frames: int, wideband_hz: float) -> NDArray[np.float64]:
    """Time (s) of each ``.whl`` frame: frame k at k * 512 / ``wideband_hz``."""
    return np.arange(n_frames) * (WHL_SAMPLES_PER_FRAME / wideband_hz)


def read_res_clu(
    res_path: Path, clu_path: Path
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Spike samples and cluster ids of one shank, one entry per detected spike."""
    samples = np.loadtxt(res_path, dtype=np.int64, ndmin=1)
    clu = np.loadtxt(clu_path, dtype=np.int64, ndmin=1)
    n_clusters, clusters = int(clu[0]), clu[1:]
    if clusters.shape != samples.shape:
        msg = f"{clu_path.name} has {clusters.size} spikes, {res_path.name} {samples.size}"
        raise ValueError(msg)
    if np.unique(clusters).size > n_clusters:
        msg = f"{clu_path.name} declares {n_clusters} clusters but uses more"
        raise ValueError(msg)
    return samples, clusters


def neuroscope_rates(xml_path: Path) -> dict[str, float]:
    """Wideband, LFP and video sampling rates (Hz) stated in a Neuroscope ``.xml``."""
    root = ET.parse(xml_path).getroot()
    acquisition = root.find("acquisitionSystem")
    lfp = root.find("fieldPotentials/lfpSamplingRate")
    if acquisition is None:
        msg = f"{xml_path.name} has no acquisitionSystem"
        raise ValueError(msg)
    rates = {"wideband_hz": float(acquisition.findtext("samplingRate", "nan"))}
    if lfp is not None and lfp.text:
        rates["lfp_hz"] = float(lfp.text)
    video = root.findtext("video/samplingRate")
    if video:
        rates["video_hz"] = float(video)
    return rates


def two_led_heading(
    front: NDArray[np.float64], back: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Head direction (rad) from two head-mounted LEDs, shape (n_time,).

    The angle of the vector from the back LED to the front LED, in the frame
    of the coordinates given: 0 along +x, +pi/2 along +y. With y pointing up
    (north) that is the package's allocentric convention; coordinates with y
    pointing down a video image give the mirror image. NaN where either LED is
    missing, and where both LEDs have the same coordinates (a tracker that
    gave one blob to both): there is no direction there, and ``arctan2(0, 0)``
    would call it 0, east.

    Parameters
    ----------
    front, back : ndarray, shape (n_time, 2)
        (x, y) of the front and back LED.
    """
    step = front - back
    coincident = np.all(step == 0, axis=1)
    return np.where(coincident, np.nan, np.arctan2(step[:, 1], step[:, 0]))
