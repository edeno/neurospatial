"""A session's LFP: nothing fetched at load, verified files, the reader's contract."""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import pickle
import warnings
import zipfile

import numpy as np
import pytest

from neurospatial._exceptions import InputError
from neurospatial.datasets import Conversion, DownloadWarning, LFPSource
from neurospatial.datasets._lfp import LFPFile
from neurospatial.datasets._summary import summary_text

RATE = 1000.0
STATED = Conversion("LFP: int16 * 0.5 uV", "stated", "the readme")
ON_CLOCK = Conversion("LFP times: the file's own", "stated", "the readme")


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_columns(sources, channels, *, rate=RATE):
    """One text file per channel: one value per line, at ``rate`` from time 0."""
    columns = {}
    for source in sources:
        name = source.name.removesuffix(".txt")
        columns[name] = np.loadtxt(source)
    samples = np.column_stack([columns[name] for name in channels])
    return samples, np.arange(len(samples)) / rate


def _source(folder, **changes) -> LFPSource:
    """Two channels, one file each, served from ``folder`` by ``file://`` URLs."""
    files = []
    for name, value in (("a", 1.0), ("b", 2.0)):
        data = "\n".join([str(value)] * 10).encode()
        (folder / f"{name}.txt").write_bytes(data)
        files.append(
            LFPFile(
                path=f"s1/{name}.txt",
                sha256=_digest(data),
                size_bytes=len(data),
                url=(folder / f"{name}.txt").as_uri(),
                channels=(name,),
            )
        )
    fields = {
        "dataset": "demo",
        "files": tuple(files),
        "sampling_rate": RATE,
        "channels": {"a": "tetrode 1", "b": "tetrode 2, the ripple channel"},
        "unit": "uV",
        "unit_basis": STATED,
        "times_basis": ON_CLOCK,
        "reader": read_columns,
        "ripple_channels": ("b",),
    }
    return LFPSource(**(fields | changes))


def test_fetches_only_the_files_of_the_chosen_channels(tmp_path):
    """Asking for one channel downloads its file and leaves the other alone."""
    source = _source(tmp_path)
    cache = tmp_path / "cache"
    with pytest.warns(DownloadWarning, match="demo LFP"):
        samples, times = source.fetch(source.ripple_channels, cache=cache)
    assert samples.shape == (10, 1)
    assert np.all(samples == 2.0)
    assert times[1] - times[0] == pytest.approx(1 / RATE)
    assert (cache / "demo" / "s1" / "b.txt").exists()
    assert not (cache / "demo" / "s1" / "a.txt").exists()


def test_columns_follow_the_order_asked_for(tmp_path):
    source = _source(tmp_path)
    with pytest.warns(DownloadWarning):
        samples, _ = source.fetch(["b", "a"], cache=tmp_path / "cache")
    assert samples[0].tolist() == [2.0, 1.0]


def test_second_fetch_is_silent_and_default_reads_every_channel(tmp_path):
    source = _source(tmp_path, cache=tmp_path / "cache")
    with pytest.warns(DownloadWarning):
        source.fetch()
    with warnings.catch_warnings():
        warnings.simplefilter("error", DownloadWarning)
        samples, _ = source.fetch()
    assert samples[0].tolist() == [1.0, 2.0]


def test_a_changed_file_is_refused(tmp_path):
    source = _source(tmp_path)
    (tmp_path / "a.txt").write_bytes(b"9\n9\n")
    with (
        pytest.warns(DownloadWarning),
        pytest.raises(ValueError, match="does not match the pinned"),
    ):
        source.fetch(["a"], cache=tmp_path / "cache")


def test_zip_member_is_read_by_its_own_checksum(tmp_path):
    data = b"3\n3\n3\n"
    archive = tmp_path / "all.zip"
    with zipfile.ZipFile(archive, "w") as out:
        out.writestr("deep/a.txt", data)
        out.writestr("deep/other.bin", b"not asked for")
    source = LFPSource(
        dataset="demo",
        files=(
            LFPFile(
                path="s1/a.txt",
                sha256=_digest(data),
                size_bytes=len(data),
                url=archive.as_uri(),
                member="deep/a.txt",
            ),
        ),
        sampling_rate=RATE,
        channels={"a": "tetrode 1"},
        unit="uV",
        unit_basis=STATED,
        times_basis=ON_CLOCK,
        reader=read_columns,
    )
    with pytest.warns(DownloadWarning):
        samples, _ = source.fetch(cache=tmp_path / "cache")
    assert samples.ravel().tolist() == [3.0, 3.0, 3.0]


def test_a_file_naming_no_channels_is_always_fetched(tmp_path):
    """A file listing no channels (one per trial) is needed for any channel."""
    source = _source(tmp_path)
    shared = [dataclasses.replace(file, channels=()) for file in source.files]
    source = dataclasses.replace(source, files=tuple(shared), reader=read_columns)
    with pytest.warns(DownloadWarning, match="in 2 file"):
        source.fetch(["a"], cache=tmp_path / "cache")


@pytest.mark.parametrize(
    ("reader", "pattern"),
    [
        (lambda s, c: (np.zeros(9), np.arange(9) / RATE), "samples: shape"),
        (lambda s, c: (np.zeros((9, 1)), np.arange(8) / RATE), "one time per sample"),
        (lambda s, c: (np.zeros((9, 1)), np.arange(9)[::-1] / RATE), "strictly incr"),
        (lambda s, c: (np.zeros((9, 1)), np.arange(9) / 900.0), "against the stated"),
    ],
    ids=["one_dimension", "lengths_differ", "decreasing", "wrong_rate"],
)
def test_reader_breaking_the_contract_is_refused(tmp_path, reader, pattern):
    source = _source(tmp_path, reader=reader)
    with pytest.warns(DownloadWarning), pytest.raises(ValueError, match=pattern):
        source.fetch(["a"], cache=tmp_path / "cache")


def test_a_gap_between_trials_is_kept(tmp_path):
    def with_gap(sources, channels):
        times = np.r_[np.arange(50), np.arange(50) + 500] / RATE
        return np.zeros((100, 1)), times

    source = _source(tmp_path, reader=with_gap)
    with pytest.warns(DownloadWarning):
        _, times = source.fetch(["a"], cache=tmp_path / "cache")
    assert np.diff(times).max() == pytest.approx(0.451)


def test_unknown_channel_names_the_channels_there_are(tmp_path):
    source = _source(tmp_path)
    with pytest.raises(InputError, match=r"(?s)no channels \['z'\].*\['a', 'b'\]"):
        source.fetch(["z"])


def test_empty_ripple_channels_say_why(tmp_path):
    source = _source(tmp_path, ripple_channels=())
    with pytest.raises(InputError, match="no ripple channel"):
        source.fetch(source.ripple_channels)


def test_streaming_is_for_dandi_files(tmp_path):
    source = _source(tmp_path)
    with pytest.raises(InputError, match="not a DANDI asset"):
        source.fetch(stream=True)


def test_streaming_hands_the_reader_the_checked_url(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "neurospatial.datasets._fetch.dandi_stream_url",
        lambda asset_id, *, sha256: f"https://stream/{asset_id}/{sha256[:4]}",
    )
    seen = []

    def reader(sources, channels):
        seen.extend(sources)
        return np.zeros((9, 1)), np.arange(9) / RATE

    source = _source(
        tmp_path,
        files=(LFPFile(path="s.nwb", sha256="ab" * 32, size_bytes=1, asset_id="x"),),
        channels={"a": "tetrode 1"},
        ripple_channels=(),
        reader=reader,
    )
    source.fetch(stream=True)
    assert seen == ["https://stream/x/abab"]


@pytest.mark.parametrize(
    ("changes", "pattern"),
    [
        ({"files": ()}, "files: empty"),
        ({"sampling_rate": 0.0}, "sampling_rate"),
        ({"channels": {}}, "channels: empty"),
        ({"channels": {"a": " ", "b": "x"}}, r"what \['a'\] are"),
        ({"ripple_channels": ("z",)}, "ripple_channels"),
        ({"unit": ""}, "unit: empty"),
        ({"unit_basis": "a string"}, "unit_basis"),
        ({"channels": {"a": "x"}}, "not channels"),
        ({"channels": {"a": "x", "b": "y", "c": "z"}}, r"holds channels \['c'\]"),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_inconsistent_sources_are_refused(tmp_path, changes, pattern):
    with pytest.raises(InputError, match=pattern):
        _source(tmp_path, **changes)


def test_every_file_names_its_channels_or_none_does(tmp_path):
    source = _source(tmp_path)
    mixed = (source.files[0], dataclasses.replace(source.files[1], channels=()))
    with pytest.raises(InputError, match="every file, or of none"):
        dataclasses.replace(source, files=mixed)


@pytest.mark.parametrize(
    ("fields", "pattern"),
    [
        ({}, "exactly one of url and asset_id"),
        ({"url": "u", "asset_id": "a"}, "exactly one of url and asset_id"),
        ({"asset_id": "a", "member": "m"}, "needs the zip's url"),
        ({"url": "u", "member": "z/other.mat"}, "cached under its own name"),
        ({"url": "u", "sha256": "xyz"}, "not 64 hexadecimal"),
        ({"url": "u", "size_bytes": -1}, "negative"),
    ],
)
def test_inconsistent_files_are_refused(fields, pattern):
    base = {"path": "s/a.mat", "sha256": "a" * 64, "size_bytes": 1}
    with pytest.raises(InputError, match=pattern):
        LFPFile(**(base | fields))


def test_source_pickles_with_its_reader_and_channels(tmp_path):
    reader = functools.partial(read_columns, rate=RATE)
    source = _source(tmp_path, reader=reader)
    copy = pickle.loads(pickle.dumps(source))
    assert dict(copy.channels) == dict(source.channels)
    assert copy.files == source.files
    assert copy.reader.func is read_columns


def test_channels_cannot_be_changed(tmp_path):
    with pytest.raises(TypeError):
        _source(tmp_path).channels["a"] = "other"


def test_recording_carries_it_and_the_summary_says_so(tmp_path, square_recording):
    assert square_recording.lfp is None
    assert "LFP" not in summary_text(square_recording)
    rec = dataclasses.replace(square_recording, lfp=_source(tmp_path))
    text = " ".join(summary_text(rec).split())
    assert "LFP 2 channels at 1000 Hz" in text
    assert "fetched on request -> rec.lfp.fetch()" in text
    assert dict(pickle.loads(pickle.dumps(rec)).lfp.channels) == dict(rec.lfp.channels)
