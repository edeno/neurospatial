"""Hand-written readers for formats neurospatial does not read: Axona DACQ and Deflate64 zips."""

from __future__ import annotations

import hashlib
import struct
import zipfile
import zlib

import numpy as np
import pytest
from load_olafsdottir_5566548 import read_cut, read_pos, read_spike_times

_END = b"\r\ndata_end\r\n"


def _header(**fields: str) -> bytes:
    return "".join(f"{k} {v}\r\n" for k, v in fields.items()).encode() + b"data_start"


def test_axona_pos_parser_round_trip():
    # Two-spot records per the dacqUSB document: 4-byte frame counter, eight
    # big-endian 16-bit words; 1023 marks an untracked spot.
    rng = np.random.default_rng(0)
    n = 250
    big = rng.integers(0, 700, (n, 2))
    little = rng.integers(0, 700, (n, 2))
    big[[3, 40]] = 1023  # untracked samples
    words = np.zeros((n, 8), dtype=">u2")
    words[:, 0:2], words[:, 2:4] = big, little
    words[:, 4:7] = rng.integers(0, 200, (n, 3))
    frames = np.arange(n, dtype=">u4") * 2  # not a timestamp; must be ignored
    record = np.dtype([("frame", ">u4"), ("words", ">u2", 8)])
    body = np.empty(n, dtype=record)
    body["frame"], body["words"] = frames, words
    blob = (
        _header(
            sample_rate="50.0 hz",
            bytes_per_timestamp="4",
            bytes_per_coord="2",
            pixels_per_metre="390",
            num_pos_samples=f"{n}      ",
        )
        + body.tobytes()
        + _END
    )
    times, spots, header = read_pos(blob)
    np.testing.assert_allclose(times, np.arange(n) / 50.0)
    expected = np.column_stack([big, little]).astype(float)
    expected[expected == 1023] = np.nan
    np.testing.assert_array_equal(spots, expected)
    assert header["pixels_per_metre"] == "390"


def test_axona_pos_parser_refuses_a_short_file():
    blob = _header(
        sample_rate="50.0 hz",
        bytes_per_timestamp="4",
        bytes_per_coord="2",
        num_pos_samples="10",
    ) + bytes(20 * 9)
    with pytest.raises(ValueError, match="not 10 x 20"):
        read_pos(blob)


def test_axona_tetrode_and_cut_round_trip():
    rng = np.random.default_rng(1)
    stamps = np.sort(rng.integers(0, 96000 * 600, 30)).astype(">u4")
    channel = np.dtype([("stamp", ">u4"), ("samples", "i1", (50,))])
    body = np.zeros((30, 4), dtype=channel)
    body["stamp"] = stamps[:, None]
    body["samples"] = rng.integers(-128, 128, (30, 4, 50))
    blob = _header(timebase="96000 hz", num_spikes="30") + body.tobytes() + _END
    np.testing.assert_allclose(read_spike_times(blob), stamps / 96000.0)
    clusters = rng.integers(0, 4, 30)
    cut = "n_clusters: 3\n\nExact_cut_for: trial spikes: 30\n" + " ".join(
        f"{c:3d}" for c in clusters
    )
    np.testing.assert_array_equal(read_cut(cut), clusters)


def test_deflate64_member_is_inflated_and_crc_checked(tmp_path):
    inflate64 = pytest.importorskip("inflate64")
    from neurospatial.datasets import _fetch

    payload = np.random.default_rng(2).bytes(200_000)
    deflater = inflate64.Deflater()
    compressed = deflater.deflate(payload) + deflater.flush()
    name = b"member.mat"
    crc = zlib.crc32(payload)
    local = (
        struct.pack(
            "<4sHHHHHIIIHH",
            b"PK\x03\x04",
            21,
            0,
            9,
            0,
            0,
            crc,
            len(compressed),
            len(payload),
            len(name),
            0,
        )
        + name
    )
    central = (
        struct.pack(
            "<4sHHHHHHIIIHHHHHII",
            b"PK\x01\x02",
            21,
            21,
            0,
            9,
            0,
            0,
            crc,
            len(compressed),
            len(payload),
            len(name),
            0,
            0,
            0,
            0,
            0,
            0,
        )
        + name
    )
    offset = len(local) + len(compressed)
    end = struct.pack("<4sHHHHIIH", b"PK\x05\x06", 0, 0, 1, 1, len(central), offset, 0)
    archive = tmp_path / "deflate64.zip"
    archive.write_bytes(local + compressed + central + end)
    with zipfile.ZipFile(archive) as z:
        assert z.getinfo("member.mat").compress_type == 9
    digest = hashlib.sha256(payload).hexdigest()
    got = _fetch.fetch_zip_member(
        str(archive), "member.mat", sha256=digest, subdir="d64", cache=tmp_path / "c"
    )
    assert got.read_bytes() == payload

    # A corrupted payload fails the CRC check before the digest is compared.
    broken = bytearray(local + compressed + central + end)
    broken[len(local) + 10] ^= 0xFF
    archive.write_bytes(bytes(broken))
    with pytest.raises(zipfile.BadZipFile, match="CRC-32"):
        _fetch.fetch_zip_member(
            str(archive),
            "member.mat",
            sha256=digest,
            subdir="d64b",
            cache=tmp_path / "c",
        )


def test_deflate64_member_with_a_different_local_name_is_refused(tmp_path):
    inflate64 = pytest.importorskip("inflate64")
    from neurospatial.datasets import _fetch

    payload = b"lattice " * 1000
    deflater = inflate64.Deflater()
    compressed = deflater.deflate(payload) + deflater.flush()
    crc = zlib.crc32(payload)

    def entry(signature: bytes, name: bytes, *, central: bool) -> bytes:
        sizes = (crc, len(compressed), len(payload), len(name))
        if central:
            return (
                struct.pack(
                    "<4sHHHHHHIIIHHHHHII",
                    signature,
                    21,
                    21,
                    0,
                    9,
                    0,
                    0,
                    *sizes,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                )
                + name
            )
        return struct.pack("<4sHHHHHIIIHH", signature, 21, 0, 9, 0, 0, *sizes, 0) + name

    local = entry(b"PK\x03\x04", b"evil.mat", central=False)
    central = entry(b"PK\x01\x02", b"good.mat", central=True)
    end = struct.pack(
        "<4sHHHHIIH",
        b"PK\x05\x06",
        0,
        0,
        1,
        1,
        len(central),
        len(local) + len(compressed),
        0,
    )
    archive = tmp_path / "renamed.zip"
    archive.write_bytes(local + compressed + central + end)
    with pytest.raises(zipfile.BadZipFile, match="local header names"):
        _fetch.fetch_zip_member(
            str(archive), "good.mat", sha256="0" * 64, subdir="r", cache=tmp_path / "c"
        )
