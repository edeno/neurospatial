"""Readers for the NeuroSuite files of a silicon-probe session (CRCNS hc-2, hc-3).

A session is a set of files sharing a base name: ``<base>.res.<shank>``, one
spike time per line in samples of the wideband clock; ``<base>.clu.<shank>``,
the number of clusters on its first line, then one cluster per spike of the
``.res`` file (cluster 0 is artefact, 1 unsortable spikes); ``<base>.xml``,
the acquisition parameters (wideband and LFP rates, channel count, voltage
range and amplification); ``<base>.eeg``, the LFP, little-endian int16
samples interleaved over every channel; and, where the session was tracked,
``<base>.whl``, one video frame per line. Each function reads one file and
returns what it stores: choosing units, scales or rules for missing samples
is the loader's.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

__all__ = ["Acquisition", "read_acquisition", "read_eeg", "read_res_clu", "read_whl"]


class Acquisition(NamedTuple):
    """The acquisition parameters a NeuroSuite ``.xml`` states.

    Attributes
    ----------
    wideband_hz : float
        The rate of the ``.res`` spike samples, Hz.
    lfp_hz : float
        The rate of the ``.eeg`` samples, Hz.
    n_channels : int
        The channels interleaved in the ``.eeg`` file.
    n_bits : int
    voltage_range : float
        The converter's input range, volts.
    amplification : float
        The gain before the converter.
    channel_groups : tuple of tuple of int
        The channels of each shank (anatomical group), in file order.
    """

    wideband_hz: float
    lfp_hz: float
    n_channels: int
    n_bits: int
    voltage_range: float
    amplification: float
    channel_groups: tuple[tuple[int, ...], ...]

    @property
    def microvolts_per_bit(self) -> float:
        """The voltage of one integer step at the electrode, µV."""
        return self.voltage_range / 2**self.n_bits / self.amplification * 1e6


def read_acquisition(path: str | os.PathLike[str]) -> Acquisition:
    """The acquisition parameters of a NeuroSuite ``.xml`` file.

    Raises
    ------
    ValueError
        If a parameter is missing or not a number.
    """
    root = ET.parse(path).getroot()  # a file pinned by SHA-256
    fields = {
        "wideband_hz": "acquisitionSystem/samplingRate",
        "n_channels": "acquisitionSystem/nChannels",
        "n_bits": "acquisitionSystem/nBits",
        "voltage_range": "acquisitionSystem/voltageRange",
        "amplification": "acquisitionSystem/amplification",
        "lfp_hz": "fieldPotentials/lfpSamplingRate",
    }
    values: dict[str, float] = {}
    for name, tag in fields.items():
        text = root.findtext(tag)
        try:
            values[name] = float(text)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            msg = f"{Path(path).name}: {tag} is {text!r}, not a number"
            raise ValueError(msg) from exc
    groups = tuple(
        tuple(int(str(channel.text)) for channel in group.findall("channel"))
        for group in root.findall("anatomicalDescription/channelGroups/group")
    )
    return Acquisition(
        wideband_hz=values["wideband_hz"],
        lfp_hz=values["lfp_hz"],
        n_channels=int(values["n_channels"]),
        n_bits=int(values["n_bits"]),
        voltage_range=values["voltage_range"],
        amplification=values["amplification"],
        channel_groups=groups,
    )


def read_res_clu(
    res_path: str | os.PathLike[str], clu_path: str | os.PathLike[str]
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Spike samples and cluster ids of one shank, one entry per detected spike.

    Returns
    -------
    samples : ndarray of int64, shape (n_spikes,)
        Wideband sample of each detected spike, as stored.
    clusters : ndarray of int64, shape (n_spikes,)
        Its cluster.

    Raises
    ------
    ValueError
        If the two files hold different numbers of spikes, or the ``.clu``
        file uses a cluster at or above the count on its first line.
    """
    samples = np.loadtxt(res_path, dtype=np.int64, ndmin=1)
    clu = np.loadtxt(clu_path, dtype=np.int64, ndmin=1)
    n_clusters, clusters = int(clu[0]), clu[1:]
    if clusters.shape != samples.shape:
        msg = (
            f"{Path(clu_path).name} has {clusters.size} spikes, "
            f"{Path(res_path).name} {samples.size}"
        )
        raise ValueError(msg)
    if clusters.size and (clusters.min() < 0 or clusters.max() >= n_clusters):
        msg = (
            f"{Path(clu_path).name} declares {n_clusters} clusters but uses "
            f"{int(clusters.min())} to {int(clusters.max())}"
        )
        raise ValueError(msg)
    return samples, clusters


def read_eeg(
    path: str | os.PathLike[str], n_channels: int, channels: tuple[int, ...]
) -> NDArray[np.int16]:
    """Chosen channels of a ``.eeg`` file, as stored, shape (n_time, n_chosen).

    Raises
    ------
    ValueError
        If the file's size is not a whole number of ``n_channels`` int16
        frames, or a channel is out of range.
    """
    size = Path(path).stat().st_size
    if size % (2 * n_channels):
        msg = f"{Path(path).name}: {size} bytes is not a whole number of {n_channels}-channel int16 frames"
        raise ValueError(msg)
    if bad := [c for c in channels if not 0 <= c < n_channels]:
        msg = f"{Path(path).name}: channels {bad} are outside 0 to {n_channels - 1}"
        raise ValueError(msg)
    frames = np.memmap(path, dtype="<i2", mode="r").reshape(-1, n_channels)
    return np.array(frames[:, list(channels)])


def read_whl(path: str | os.PathLike[str]) -> NDArray[np.float64]:
    """The frames of a ``.whl`` file as stored, shape (n_frames, n_columns).

    Raises
    ------
    ValueError
        If the file is empty or its columns are not (x, y) pairs.
    """
    frames = np.loadtxt(path, dtype=np.float64, ndmin=2)
    if frames.size == 0 or frames.shape[1] % 2:
        msg = f"{Path(path).name}: {frames.shape[1]} columns, not (x, y) pairs"
        raise ValueError(msg)
    return frames
