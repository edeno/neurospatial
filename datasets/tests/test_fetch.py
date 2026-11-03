from __future__ import annotations

import datetime
import functools
import hashlib
import http.server
import io
import json
import threading
import urllib.request
from pathlib import Path

import _fetch
import numpy as np
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


def test_dandi_stream_url_refuses_changed_digest(monkeypatch):
    published = {"digest": {"dandi:sha2-256": "b" * 64}}

    def api(url):
        return io.BytesIO(json.dumps(published).encode())

    monkeypatch.setattr(urllib.request, "urlopen", api)
    with pytest.raises(ValueError, match=f"{'b' * 64}, not the pinned {'a' * 64}"):
        _fetch.dandi_stream_url("asset-1", sha256="a" * 64)
    assert _fetch.dandi_stream_url("asset-1", sha256="b" * 64).endswith(
        "/assets/asset-1/download/"
    )


class _RangeHandler(http.server.SimpleHTTPRequestHandler):
    """Static file server that honours single ``Range: bytes=a-b`` requests."""

    def log_message(self, *args):
        pass

    def send_head(self):
        spec = self.headers.get("Range")
        path = Path(self.translate_path(self.path))
        if spec is None or not path.is_file():
            return super().send_head()
        size = path.stat().st_size
        start, _, end = spec.removeprefix("bytes=").partition("-")
        start, end = int(start), min(int(end or size - 1), size - 1)
        with path.open("rb") as handle:
            handle.seek(start)
            body = handle.read(end - start + 1)
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        return io.BytesIO(body)


def test_open_nwb_reads_local_and_http(tmp_path):
    import pynwb
    from pynwb.behavior import Position

    nwbfile = pynwb.NWBFile("demo", "id", datetime.datetime.now(datetime.UTC))
    behavior = nwbfile.create_processing_module("behavior", "position")
    xy = np.column_stack([np.linspace(0, 50, 200), np.linspace(10, 20, 200)])
    behavior.add(
        Position(
            spatial_series=pynwb.behavior.SpatialSeries(
                name="xy",
                data=xy,
                timestamps=np.arange(200) / 30.0,
                reference_frame="corner",
                unit="cm",
            )
        )
    )
    path = tmp_path / "demo.nwb"
    with pynwb.NWBHDF5IO(path, "w") as io_:
        io_.write(nwbfile)

    def positions(source):
        with _fetch.open_nwb(source) as f:
            return f.processing["behavior"]["Position"]["xy"].data[:]

    handler = functools.partial(_RangeHandler, directory=str(tmp_path))
    with http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            remote = positions(f"http://127.0.0.1:{server.server_port}/demo.nwb")
        finally:
            server.shutdown()
    np.testing.assert_array_equal(positions(str(path)), xy)
    np.testing.assert_array_equal(remote, xy)
