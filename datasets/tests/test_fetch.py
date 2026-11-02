from __future__ import annotations

import hashlib
import urllib.request

import _fetch
import pytest
from _fetch import fetch


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "source" / "data.bin"
    path.parent.mkdir()
    payload = b"spatial data " * 1000
    path.write_bytes(payload)
    return path.as_uri(), hashlib.sha256(payload).hexdigest()


def test_fetch_verifies_and_caches(source, tmp_path, monkeypatch):
    url, digest = source
    cache = tmp_path / "cache"
    first = fetch(url, "data.bin", sha256=digest, subdir="demo", cache=cache)
    assert first == cache / "demo" / "data.bin"

    def offline(*args, **kwargs):
        msg = "the cached file must be reused without the network"
        raise AssertionError(msg)

    monkeypatch.setattr(urllib.request, "urlopen", offline)
    assert fetch(url, "data.bin", sha256=digest, subdir="demo", cache=cache) == first


def test_fetch_refuses_mismatched_existing_file(source, tmp_path):
    url, digest = source
    existing = tmp_path / "cache" / "demo" / "data.bin"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"something else")
    with pytest.raises(FileExistsError, match="different SHA-256"):
        fetch(url, "data.bin", sha256=digest, subdir="demo", cache=tmp_path / "cache")
    assert existing.read_bytes() == b"something else"


def test_bad_download_leaves_no_temp_file(source, tmp_path):
    url, _ = source
    with pytest.raises(ValueError, match="does not match the pinned"):
        fetch(url, "data.bin", sha256="0" * 64, subdir="demo", cache=tmp_path / "cache")
    assert list((tmp_path / "cache" / "demo").iterdir()) == []


def test_cache_root_reads_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("NEUROSPATIAL_DATA_DIR", str(tmp_path / "elsewhere"))
    assert _fetch.cache_root() == tmp_path / "elsewhere"
