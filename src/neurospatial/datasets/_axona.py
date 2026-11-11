"""Readers for the files of an Axona dacqUSB recording and its Tint cluster cuts.

Fixed-layout binaries, so no Axona library is needed. Each data file opens with
ASCII ``key value`` lines and ends its header with ``data_start``; the binary
data follow at once and are closed by ``\\r\\ndata_end\\r\\n``. The ``.set``
file is ASCII only. Each function reads one file and returns what it stores,
in seconds and in the file's own values, with the header that describes them;
choosing a scale, a frame or a rule for missing samples is the loader's
(:func:`microvolts_per_unit` gives the scale the ``.set`` file states).

Sources read for the layouts: Axona's "dacqUSB Data File Formats" document
(the tetrode, ``.eeg``/``.egf`` and ``.pos`` sections), and for what that
document leaves out, three open readers, read and not run: ephysiopy's
``axonaIO.py`` (MIT, github.com/rhayman/ephysiopy at 06a057a: tetrode
waveforms as signed bytes scaled by ``ADC_fullscale_mv / 1000 / gain / 128``,
with the gains of channels ``4 (tetrode - 1)`` to ``4 (tetrode - 1) + 3``),
the Barry lab's ``barrylab.io.axona`` (MIT, github.com/Barry-lab/PythonSpkAnalysis
at d51c15a: ``.egf`` little-endian ``int16``, ``.eeg`` signed bytes, the gain of
EEG slot ``n`` keyed by ``EEG_ch_n`` minus 1, cluster ids after ``spikes:`` in a
``.cut`` file) and Neo's ``axonarawio.py`` (BSD-3, 0.14.1: microvolts per stored
unit ``1000 ADC_fullscale_mv / (gain 2^(8 bytes - 1))``).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "AxonaSpikes",
    "PositionTrack",
    "microvolts_per_unit",
    "read_clu",
    "read_continuous",
    "read_cut",
    "read_header",
    "read_position",
    "read_set",
    "read_tetrode",
    "tetrode_channels",
]

#: The marker that ends every data file's header; the data follow at once.
DATA_START = b"data_start"
#: The marker after every data file's binary data.
DATA_END = b"\r\ndata_end\r\n"
#: Channels per tetrode, and the 4-byte timestamp each channel's block opens with.
CHANNELS = 4
TIMESTAMP_BYTES = 4
#: A position word holding this marks a spot the tracker did not find.
UNTRACKED = 1023
#: Words per position record after its 4-byte frame counter.
POSITION_WORDS = 8

_SPIKE_FORMAT = "t,ch1,t,ch2,t,ch3,t,ch4"


def _fields(text: str) -> dict[str, str]:
    """The ``key value`` lines of a header: each first word and the rest of its line."""
    fields = {}
    for line in text.splitlines():
        key, _, value = line.strip().partition(" ")
        if key:
            fields[key] = value.strip()
    return fields


def read_header(path: str | os.PathLike[str]) -> tuple[dict[str, str], int]:
    """A data file's header and where its binary data start.

    Parameters
    ----------
    path : str or path
        A tetrode (``.1`` to ``.32``), ``.pos``, ``.eeg`` or ``.egf`` file.

    Returns
    -------
    header : dict of str to str
        Each header key and the rest of its line, stripped.
    offset : int
        The byte offset of the first data byte, just past ``data_start``.

    Raises
    ------
    ValueError
        If the file has no ``data_start`` marker.
    """
    with Path(path).open("rb") as stream:
        head = stream.read(1 << 16)
    at = head.find(DATA_START)
    if at < 0:
        msg = f"{Path(path).name}: no 'data_start' in its first 64 kB; not an Axona data file"
        raise ValueError(msg)
    return _fields(head[:at].decode("latin-1")), at + len(DATA_START)


def read_set(path: str | os.PathLike[str]) -> dict[str, str]:
    """The ``key value`` lines of a ``.set`` file, the trial's hardware settings.

    Parameters
    ----------
    path : str or path

    Returns
    -------
    dict of str to str
        For example ``"gain_ch_32" -> "10000"``, ``"ADC_fullscale_mv" ->
        "1500"``, ``"trial_time" -> "09:30:49"``.
    """
    return _fields(Path(path).read_text(encoding="latin-1"))


def _number(header: dict[str, str], key: str, path: str | os.PathLike[str]) -> float:
    """The number a header ``key`` opens with: ``"96000 hz"`` gives 96000.0."""
    try:
        return float(header[key].split()[0])
    except (KeyError, IndexError, ValueError) as exc:
        msg = f"{Path(path).name}: header has no number for {key!r}"
        raise ValueError(msg) from exc


def _data(path: str | os.PathLike[str], offset: int, n_bytes: int) -> bytes:
    """``n_bytes`` of data from ``offset``, checked to end with ``data_end``."""
    with Path(path).open("rb") as stream:
        stream.seek(offset)
        data = stream.read(n_bytes + len(DATA_END))
    if len(data) < n_bytes or data[n_bytes:] != DATA_END:
        msg = (
            f"{Path(path).name}: the header's counts give {n_bytes} data bytes, but "
            f"the file holds {len(data) - len(DATA_END)} before its end marker"
        )
        raise ValueError(msg)
    return data[:n_bytes]


class AxonaSpikes(NamedTuple):
    """What a tetrode file stores for each detected spike.

    Attributes
    ----------
    times : ndarray, shape (n_spikes,)
        Seconds from the start of the trial: the timestamp / ``timebase``.
    waveforms : ndarray of int8, shape (n_spikes, 4, n_samples)
        Each channel's stored samples, in the file's own units; see
        :func:`microvolts_per_unit`.
    sample_rate : float
        The waveforms' sampling rate, Hz.
    header : dict of str to str
    """

    times: NDArray[np.float64]
    waveforms: NDArray[np.int8]
    sample_rate: float
    header: dict[str, str]


def read_tetrode(path: str | os.PathLike[str]) -> AxonaSpikes:
    """Read a tetrode file (``.1`` to ``.32``) recorded in unit mode.

    Each spike is four blocks, one per channel in order, of a 4-byte
    big-endian timestamp and ``samples_per_spike`` signed bytes (the dacqUSB
    document: "216 bytes per spike, structured as 54 bytes per channel").

    Parameters
    ----------
    path : str or path

    Returns
    -------
    AxonaSpikes

    Raises
    ------
    ValueError
        If the header does not describe that layout (4 channels, 4-byte
        timestamps, 1-byte samples, ``spike_format t,ch1,t,ch2,t,ch3,t,ch4``),
        the data do not hold ``num_spikes`` spikes before ``data_end``, or the
        four channels of a spike carry different timestamps.
    """
    header, offset = read_header(path)
    layout = {
        "num_chans": _number(header, "num_chans", path) == CHANNELS,
        "bytes_per_timestamp": _number(header, "bytes_per_timestamp", path)
        == TIMESTAMP_BYTES,
        "bytes_per_sample": _number(header, "bytes_per_sample", path) == 1,
        "spike_format": header.get("spike_format") == _SPIKE_FORMAT,
    }
    if not all(layout.values()):
        wrong = [key for key, ok in layout.items() if not ok]
        msg = (
            f"{Path(path).name}: header fields {wrong} do not describe four "
            "channels of 4-byte timestamps and 1-byte samples"
        )
        raise ValueError(msg)
    n_spikes = int(_number(header, "num_spikes", path))
    n_samples = int(_number(header, "samples_per_spike", path))
    block = np.dtype([("t", ">i4"), ("samples", "i1", (n_samples,))])
    data = _data(path, offset, n_spikes * CHANNELS * block.itemsize)
    records = np.frombuffer(data, dtype=block).reshape(n_spikes, CHANNELS)
    stamps = records["t"]
    if not np.array_equal(stamps, np.repeat(stamps[:, :1], CHANNELS, axis=1)):
        n_bad = int(np.sum(np.any(stamps != stamps[:, :1], axis=1)))
        msg = (
            f"{Path(path).name}: {n_bad} spikes carry different timestamps per channel"
        )
        raise ValueError(msg)
    timebase = _number(header, "timebase", path)
    return AxonaSpikes(
        times=stamps[:, 0].astype(np.float64) / timebase,
        waveforms=np.ascontiguousarray(records["samples"]),
        sample_rate=_number(header, "sample_rate", path),
        header=header,
    )


class PositionTrack(NamedTuple):
    """What a ``.pos`` file stores for each video frame.

    Attributes
    ----------
    times : ndarray, shape (n_frames,)
        Seconds from the start of the trial: frame ``k``, counted from 1, at
        ``k / sample_rate`` (the dacqUSB document: "the Xth sample in the file
        was taken at time X/50 seconds"; the frame counter "should be
        ignored").
    words : ndarray of uint16, shape (n_frames, 8)
        The eight big-endian words after each frame counter. In two-spot mode
        (``pos_format t,x1,y1,x2,y2,numpix1,numpix2``): big spot x, y, little
        spot x, y, the pixels of each spot, the total tracked pixels, unused.
        :data:`UNTRACKED` (1023) in x and y marks a spot not found.
    frame_counter : ndarray of uint32, shape (n_frames,)
        The camera's frame counter, kept for checks only.
    sample_rate : float
        Frames per second, Hz.
    pixels_per_metre : float
        The header's ``pixels_per_metre``.
    header : dict of str to str
    """

    times: NDArray[np.float64]
    words: NDArray[np.uint16]
    frame_counter: NDArray[np.uint32]
    sample_rate: float
    pixels_per_metre: float
    header: dict[str, str]


def read_position(path: str | os.PathLike[str]) -> PositionTrack:
    """Read a ``.pos`` file: one 20-byte record per video frame.

    Parameters
    ----------
    path : str or path

    Returns
    -------
    PositionTrack

    Raises
    ------
    ValueError
        If the header does not describe 4-byte frame counters and 2-byte
        words, or the data do not hold ``num_pos_samples`` records before
        ``data_end``.
    """
    header, offset = read_header(path)
    if (
        _number(header, "bytes_per_timestamp", path) != TIMESTAMP_BYTES
        or _number(header, "bytes_per_coord", path) != 2
    ):
        msg = f"{Path(path).name}: not 4-byte frame counters and 2-byte words"
        raise ValueError(msg)
    n_frames = int(_number(header, "num_pos_samples", path))
    record = np.dtype([("counter", ">u4"), ("words", ">u2", (POSITION_WORDS,))])
    records = np.frombuffer(
        _data(path, offset, n_frames * record.itemsize), dtype=record
    )
    rate = _number(header, "sample_rate", path)
    return PositionTrack(
        times=np.arange(1, n_frames + 1, dtype=np.float64) / rate,
        words=records["words"].astype(np.uint16),
        frame_counter=records["counter"].astype(np.uint32),
        sample_rate=rate,
        pixels_per_metre=_number(header, "pixels_per_metre", path),
        header=header,
    )


def read_continuous(
    path: str | os.PathLike[str],
) -> tuple[NDArray[np.int16], float, dict[str, str]]:
    """Read an ``.eeg`` or ``.egf`` file (``.eegN``, ``.egfN``): one channel of LFP.

    ``bytes_per_sample`` 1 is signed bytes (``.eeg``, 250 Hz), 2 is
    little-endian ``int16`` (``.egf``, 4.8 kHz); the dacqUSB document gives
    the rates and sizes, and the Barry lab's reader the byte order.

    Parameters
    ----------
    path : str or path

    Returns
    -------
    samples : ndarray of int16, shape (n_samples,)
        The stored values, in the file's own units; see
        :func:`microvolts_per_unit`.
    sample_rate : float
        Hz.
    header : dict of str to str

    Raises
    ------
    ValueError
        If ``bytes_per_sample`` is neither 1 nor 2, or the data do not hold the
        header's sample count before ``data_end``.
    """
    header, offset = read_header(path)
    width = int(_number(header, "bytes_per_sample", path))
    if width not in (1, 2):
        msg = f"{Path(path).name}: bytes_per_sample {width}, not 1 or 2"
        raise ValueError(msg)
    key = "num_EEG_samples" if "num_EEG_samples" in header else "num_EGF_samples"
    n_samples = int(_number(header, key, path))
    dtype = np.dtype("i1") if width == 1 else np.dtype("<i2")
    data = _data(path, offset, n_samples * width)
    samples = np.frombuffer(data, dtype=dtype).astype(np.int16)
    return samples, _number(header, "sample_rate", path), header


def read_cut(path: str | os.PathLike[str]) -> tuple[NDArray[np.int64], str]:
    """Read a Tint ``.cut`` file: the cluster of each spike of one tetrode file.

    After the cluster boundaries, a line ``Exact_cut_for: <trial> spikes: <n>``
    is followed by ``n`` cluster ids, one per spike in the tetrode file's
    order. Cluster 0 holds the spikes assigned to no cell.

    Parameters
    ----------
    path : str or path

    Returns
    -------
    clusters : ndarray of int, shape (n_spikes,)
    trial : str
        The trial (file stem) the cut was made for.

    Raises
    ------
    ValueError
        If the file has no ``Exact_cut_for:`` line or holds a different number
        of ids from the count it states.
    """
    tokens = Path(path).read_text(encoding="latin-1").split()
    try:
        at = tokens.index("Exact_cut_for:")
        trial, label, count = tokens[at + 1 : at + 4]
    except ValueError as exc:
        msg = f"{Path(path).name}: no 'Exact_cut_for:' line; not a Tint cut file"
        raise ValueError(msg) from exc
    if label != "spikes:":
        msg = (
            f"{Path(path).name}: 'Exact_cut_for: {trial}' is not followed by 'spikes:'"
        )
        raise ValueError(msg)
    clusters = np.array(tokens[at + 4 :], dtype=np.int64)
    if clusters.size != int(count):
        msg = (
            f"{Path(path).name}: states {count} spikes but holds {clusters.size} "
            "cluster ids"
        )
        raise ValueError(msg)
    return clusters, trial


def read_clu(path: str | os.PathLike[str]) -> NDArray[np.int64]:
    """Read a KlustaKwik ``.clu.N`` file: its cluster count, then one id per spike.

    Parameters
    ----------
    path : str or path

    Returns
    -------
    ndarray of int, shape (n_spikes,)
        The ids as stored. KlustaKwik numbers from 1, with 1 its noise
        cluster; a Tint ``.cut`` file of the same spikes numbers from 0.

    Raises
    ------
    ValueError
        If any id exceeds the count on the first line.
    """
    values = np.array(Path(path).read_text(encoding="latin-1").split(), dtype=np.int64)
    n_clusters, clusters = int(values[0]), values[1:]
    if clusters.size and (clusters.min() < 0 or clusters.max() > n_clusters):
        msg = f"{Path(path).name}: ids run to {clusters.max()}, past its {n_clusters}"
        raise ValueError(msg)
    return clusters


def tetrode_channels(tetrode: int) -> tuple[int, int, int, int]:
    """The four recording channels of a tetrode, numbered from 0 as ``gain_ch_N``.

    Tetrode 1 is channels 0 to 3, tetrode 2 channels 4 to 7, and so on.

    Examples
    --------
    >>> tetrode_channels(10)
    (36, 37, 38, 39)
    """
    first = CHANNELS * (int(tetrode) - 1)
    return (first, first + 1, first + 2, first + 3)


def microvolts_per_unit(
    settings: dict[str, str], channel: int, bytes_per_sample: int
) -> float:
    """Microvolts per stored unit of one recording channel, from the ``.set`` file.

    ``1000 * ADC_fullscale_mv / (gain_ch_<channel> * 2 ** (8 * bytes_per_sample
    - 1))``: the stored value's full scale is the converter's full scale over
    the channel's amplifier gain.

    Parameters
    ----------
    settings : dict of str to str
        The trial's ``.set`` file (:func:`read_set`).
    channel : int
        The recording channel, numbered from 0 (:func:`tetrode_channels`; for
        EEG slot ``n``, the ``.set`` file's ``EEG_ch_n`` minus 1).
    bytes_per_sample : int
        1 for tetrode waveforms and ``.eeg``, 2 for ``.egf``.

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If the ``.set`` file lacks ``ADC_fullscale_mv`` or the channel's gain,
        or the gain is not positive.

    Examples
    --------
    >>> settings = {"ADC_fullscale_mv": "1500", "gain_ch_36": "10000"}
    >>> round(microvolts_per_unit(settings, 36, 1), 4)
    1.1719
    """
    try:
        full_scale = float(settings["ADC_fullscale_mv"])
        gain = float(settings[f"gain_ch_{channel}"])
    except (KeyError, ValueError) as exc:
        msg = f"the .set file gives no ADC_fullscale_mv or gain_ch_{channel}"
        raise ValueError(msg) from exc
    if gain <= 0:
        msg = f"gain_ch_{channel} is {gain:g}; a gain must be positive"
        raise ValueError(msg)
    return 1000.0 * full_scale / (gain * 2.0 ** (8 * bytes_per_sample - 1))
