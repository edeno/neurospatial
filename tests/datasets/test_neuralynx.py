"""The Neuralynx and MClust readers, on files written from their record layouts."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from neurospatial.datasets._neuralynx import (
    HEADER_BYTES,
    SAMPLES_PER_RECORD,
    read_events,
    read_header,
    read_lfp,
    read_spike_times,
    read_video_track,
)


def _header(*lines: str) -> bytes:
    text = "######## Neuralynx Data File Header\n" + "\n".join(lines) + "\n"
    return text.encode("latin-1").ljust(HEADER_BYTES, b"\0")


def _ncs_record(timestamp: int, rate: int, n_valid: int, value: int) -> bytes:
    samples = np.arange(SAMPLES_PER_RECORD, dtype="<i2") + value
    return struct.pack("<QIII", timestamp, 3, rate, n_valid) + samples.tobytes()


def test_header_fields_are_read_as_text(tmp_path):
    path = tmp_path / "a.ncs"
    path.write_bytes(_header("-ADBitVolts 0.5", "## a comment", "-InputInverted True"))
    assert read_header(path) == {"ADBitVolts": "0.5", "InputInverted": "True"}


def test_lfp_times_each_sample_from_its_records_first_and_drops_invalid_samples(
    tmp_path,
):
    path = tmp_path / "a.ncs"
    # A 2 kHz record of 512 valid samples, one of 3 valid, then a gap.
    path.write_bytes(
        _header("-ADBitVolts 1e-07")
        + _ncs_record(1_000_000, 2000, SAMPLES_PER_RECORD, 0)
        + _ncs_record(1_256_000, 2000, 3, 1000)
        + _ncs_record(9_000_000, 2000, SAMPLES_PER_RECORD, 2000)
    )
    samples, times, header = read_lfp(path)
    assert samples.shape == times.shape == (2 * SAMPLES_PER_RECORD + 3,)
    assert samples[[0, 511, 512, 514, 515]].tolist() == [0, 511, 1000, 1002, 2000]
    assert times[1] - times[0] == pytest.approx(0.0005)
    assert times[[0, 512, 515]] == pytest.approx([1.0, 1.256, 9.0])
    assert times[514] == pytest.approx(1.257)
    assert header["ADBitVolts"] == "1e-07"


def test_lfp_with_two_rates_is_refused(tmp_path):
    path = tmp_path / "a.ncs"
    path.write_bytes(
        _header()
        + _ncs_record(0, 2000, SAMPLES_PER_RECORD, 0)
        + _ncs_record(1, 32000, SAMPLES_PER_RECORD, 0)
    )
    with pytest.raises(ValueError, match=r"sampling rates \[2000, 32000\]"):
        read_lfp(path)


def test_video_track_reads_each_frames_extracted_point(tmp_path):
    path = tmp_path / "VT1.nvt"
    frames = [(2_000_000, 100, 200, 90), (2_033_367, 0, 0, 0)]
    body = b"".join(
        struct.pack("<HHHQ", 0x800, 0x1000, 1800, timestamp)
        + bytes(400 * 4)
        + struct.pack("<hiii", 0, x, y, angle)
        + bytes(50 * 4)
        for timestamp, x, y, angle in frames
    )
    path.write_bytes(_header("-SamplingFrequency 29.97") + body)
    track = read_video_track(path)
    assert track.times == pytest.approx([2.0, 2.033367])
    assert track.x.tolist() == [100, 0]
    assert track.y.tolist() == [200, 0]
    assert track.angle.tolist() == [90, 0]


def test_events_read_times_and_text(tmp_path):
    path = tmp_path / "Events.nev"
    records = b"".join(
        struct.pack("<hhhQhhhhh", 0, 0, 0, timestamp, 11, 0, 0, 0, 0)
        + bytes(8 * 4)
        + text.encode().ljust(128, b"\0")
        for timestamp, text in [
            (5_000_000, "Starting Recording"),
            (6_500_000, "pellet"),
        ]
    )
    path.write_bytes(_header() + records)
    times, texts = read_events(path)
    assert times == pytest.approx([5.0, 6.5])
    assert texts == ("Starting Recording", "pellet")


@pytest.mark.parametrize("line_end", [b"\n", b"\r\n"])
def test_mclust_times_are_big_endian_tenths_of_a_millisecond(tmp_path, line_end):
    path = tmp_path / "TT04_1.t"
    header = line_end.join([b"%%BEGINHEADER", b"% T-file", b"%%ENDHEADER", b""])
    path.write_bytes(header + struct.pack(">II", 15_000_000, 15_000_123))
    assert read_spike_times(path) == pytest.approx([1500.0, 1500.0123])


def test_a_file_without_an_mclust_header_is_refused(tmp_path):
    path = tmp_path / "TT04_1.t"
    path.write_bytes(b"\x00\x01\x02\x03")
    with pytest.raises(ValueError, match="no MClust header"):
        read_spike_times(path)
