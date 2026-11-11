"""Readers for the Neuralynx and MClust files of a tetrode recording.

Fixed-layout binaries, so no Neuralynx library is needed. ``.ncs``, ``.nvt`` and
``.nev`` files open with a 16 kB ASCII header, then little-endian records whose
timestamps are in microseconds. MClust's ``.t`` and ``._t`` files open with a
text header, then big-endian timestamps in units of 100 microseconds, 32-bit
as a rule and 64-bit in some files.
Each function reads one file and returns what it stores, in seconds and in the
file's own values: choosing a scale or a rule for missing samples is the
loader's.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "VideoTrack",
    "read_events",
    "read_lfp",
    "read_spike_times",
    "read_video_track",
]

#: The bytes of a Neuralynx file's ASCII header.
HEADER_BYTES = 16 * 1024
#: Samples in one record of a ``.ncs`` file.
SAMPLES_PER_RECORD = 512

_NCS_RECORD = np.dtype(
    [
        ("timestamp", "<u8"),
        ("channel", "<u4"),
        ("rate", "<u4"),
        ("n_valid", "<u4"),
        ("samples", "<i2", (SAMPLES_PER_RECORD,)),
    ]
)
_NVT_RECORD = np.dtype(
    [
        ("start", "<u2"),
        ("id", "<u2"),
        ("size", "<u2"),
        ("timestamp", "<u8"),
        ("points", "<u4", (400,)),
        ("crc", "<i2"),
        ("x", "<i4"),
        ("y", "<i4"),
        ("angle", "<i4"),
        ("targets", "<i4", (50,)),
    ]
)
_NEV_RECORD = np.dtype(
    [
        ("start", "<i2"),
        ("id", "<i2"),
        ("size", "<i2"),
        ("timestamp", "<u8"),
        ("event_id", "<i2"),
        ("ttl", "<i2"),
        ("crc", "<i2"),
        ("dummy", "<i2", (2,)),
        ("extra", "<i4", (8,)),
        ("text", "S128"),
    ]
)
_US = 1e-6
#: MClust's timestamps per second. Seconds are ticks / 10,000, the correctly
#: rounded value and what MClust's and the vandermeerlab ``LoadSpikes.m`` compute;
#: multiplying by 1e-4 (not exactly representable) is off by one bit for many
#: ticks, which moves spikes that sit on a time-bin edge.
_TICKS_PER_SECOND = 10_000.0


class VideoTrack(NamedTuple):
    """What a ``.nvt`` file stores for each video frame.

    Attributes
    ----------
    times : ndarray, shape (n_frames,)
        Seconds on the acquisition clock.
    x, y : ndarray of int, shape (n_frames,)
        The tracked point in pixels; 0 where the tracker found nothing.
    angle : ndarray of int, shape (n_frames,)
        The tracker's angle, in degrees; 0 where it computed none.
    """

    times: NDArray[np.float64]
    x: NDArray[np.int32]
    y: NDArray[np.int32]
    angle: NDArray[np.int32]


def read_header(path: str | os.PathLike[str]) -> dict[str, str]:
    """The ``-Name value`` lines of a Neuralynx file's header.

    Parameters
    ----------
    path : str or path

    Returns
    -------
    dict of str to str
        Each name, without its dash, and the rest of its line.
    """
    with Path(path).open("rb") as stream:
        text = stream.read(HEADER_BYTES).split(b"\0", 1)[0].decode("latin-1")
    fields = {}
    for line in text.splitlines():
        if line.startswith("-"):
            name, _, value = line[1:].partition(" ")
            fields[name] = value.strip()
    return fields


def read_lfp(
    path: str | os.PathLike[str],
) -> tuple[NDArray[np.int16], NDArray[np.float64], dict[str, str]]:
    """One channel of a ``.ncs`` file.

    A record's timestamp is its first sample's; the later samples follow at the
    record's rate. Samples past a record's ``n_valid`` are not samples.

    Parameters
    ----------
    path : str or path

    Returns
    -------
    samples : ndarray of int16, shape (n_time,)
        The stored values; ``ADBitVolts`` in the header scales them to volts.
    times : ndarray, shape (n_time,)
        Seconds on the acquisition clock.
    header : dict of str to str

    Raises
    ------
    ValueError
        If the file's records hold more than one sampling rate.
    """
    records = np.fromfile(path, dtype=_NCS_RECORD, offset=HEADER_BYTES)
    rates = np.unique(records["rate"])
    if rates.size != 1:
        msg = f"{Path(path).name}: its records hold sampling rates {rates.tolist()}"
        raise ValueError(msg)
    offsets = np.arange(SAMPLES_PER_RECORD) / float(rates[0])
    starts = records["timestamp"].astype(np.float64) * _US
    valid = np.arange(SAMPLES_PER_RECORD) < records["n_valid"][:, None]
    times = (starts[:, None] + offsets)[valid]
    return records["samples"][valid], times, read_header(path)


def read_video_track(path: str | os.PathLike[str]) -> VideoTrack:
    """The tracked point of each frame of a ``.nvt`` file.

    Parameters
    ----------
    path : str or path

    Returns
    -------
    VideoTrack
    """
    records = np.fromfile(path, dtype=_NVT_RECORD, offset=HEADER_BYTES)
    return VideoTrack(
        times=records["timestamp"].astype(np.float64) * _US,
        x=records["x"].copy(),
        y=records["y"].copy(),
        angle=records["angle"].copy(),
    )


def read_events(
    path: str | os.PathLike[str],
) -> tuple[NDArray[np.float64], tuple[str, ...]]:
    """The events of a ``.nev`` file.

    Parameters
    ----------
    path : str or path

    Returns
    -------
    times : ndarray, shape (n_events,)
        Seconds on the acquisition clock.
    texts : tuple of str
        Each event's text.
    """
    records = np.fromfile(path, dtype=_NEV_RECORD, offset=HEADER_BYTES)
    times = records["timestamp"].astype(np.float64) * _US
    return times, tuple(text.decode("latin-1").strip() for text in records["text"])


def _spike_words(path: str | os.PathLike[str]) -> NDArray[np.unsignedinteger[Any]]:
    """The big-endian 32-bit words after an MClust file's header."""
    data = Path(path).read_bytes()
    marker = data.find(b"%%ENDHEADER")
    end = data.find(b"\n", marker)
    if marker < 0 or end < 0:
        msg = f"{Path(path).name}: no MClust header (%%ENDHEADER not found)"
        raise ValueError(msg)
    return np.frombuffer(data, dtype=">u4", offset=end + 1)


def is_64_bit(words: NDArray[np.unsignedinteger[Any]]) -> bool:
    """Whether MClust words are 64-bit times, every high word zero.

    The header does not say. The times are in tenths of a millisecond, so their
    high words are zero; 32-bit times are not zero every other word. Two words
    alone cannot show it, and are read as 32-bit.

    Parameters
    ----------
    words : ndarray, shape (n_words,)
        The file's big-endian 32-bit words.
    """
    return bool(words.size >= 4 and words.size % 2 == 0 and not np.any(words[::2]))


def read_spike_times(path: str | os.PathLike[str]) -> NDArray[np.float64]:
    """The spike times of an MClust ``.t`` or ``._t`` file, in stored order.

    Parameters
    ----------
    path : str or path

    Returns
    -------
    ndarray, shape (n_spikes,)
        Seconds on the acquisition clock, to the file's 100 microseconds.

    Raises
    ------
    ValueError
        If the file has no MClust header.
    """
    words = _spike_words(path)
    ticks = words[1::2] if is_64_bit(words) else words
    return ticks.astype(np.float64) / _TICKS_PER_SECOND
