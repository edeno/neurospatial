from __future__ import annotations

import datetime
import email.message
import functools
import hashlib
import http.client
import http.server
import io
import json
import logging
import tarfile
import threading
import traceback
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from types import SimpleNamespace

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


_USER, _PASSWORD = "dummy-user-7f3a", "dummy-secret-91c2"


class _PortalResponse(io.BytesIO):
    """A portal response: a body and a Content-Type header."""

    def __init__(self, body: bytes, content_type: str) -> None:
        super().__init__(body)
        self.headers = email.message.Message()
        self.headers["Content-Type"] = content_type


def _tar_gz(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


@pytest.fixture
def portal(monkeypatch):
    """Dummy credentials and a stubbed portal; ``portal.reply`` sets its answer."""
    monkeypatch.setenv("CRCNS_USERNAME", _USER)
    monkeypatch.setenv("CRCNS_PASSWORD", _PASSWORD)
    state = SimpleNamespace(reply=None, forms=[])

    class Opener:
        def open(self, url, data=None, timeout=None):
            state.forms.append(urllib.parse.parse_qs(data.decode()))
            if isinstance(state.reply, Exception):
                raise state.reply
            return state.reply

    monkeypatch.setattr(urllib.request, "build_opener", lambda *handlers: Opener())
    return state


def _echoing_http_error():
    # A server error whose status text and body repeat the login.
    return urllib.error.HTTPError(
        _fetch.CRCNS_PORTAL,
        403,
        f"Forbidden for {_USER}:{_PASSWORD}",
        email.message.Message(),
        io.BytesIO(f"user {_USER} password {_PASSWORD}".encode()),
    )


def _everything_shown(exc: BaseException) -> str:
    """Every exception in the chain as a traceback that also shows local variables."""
    text, seen = [], set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        text.append(
            "".join(
                traceback.TracebackException.from_exception(
                    exc, capture_locals=True
                ).format()
            )
        )
        exc = exc.__cause__ or exc.__context__
    return "".join(text)


_MEMBER = "s/resSave.mat"


def _raised(fetch, *args):
    """The exception ``fetch(*args)`` raises, caught in a frame holding no secret.

    Its traceback starts here, so the frames shown are this one and the
    library's, not the test's (whose fixtures hold the dummy values).
    """
    try:
        fetch(*args)
    except Exception as exc:  # the test checks the type
        return exc
    msg = "expected an error"
    raise AssertionError(msg)


def _fetch_file(cache):
    return _fetch.fetch_crcns(
        "hc-31/data/resSave.mat", sha256="0" * 64, subdir="crcns", cache=cache
    )


def _fetch_member(cache, member=_MEMBER):
    return _fetch.fetch_crcns_tar_members(
        "hc-31/data/resSave.mat", {member: "0" * 64}, subdir="crcns", cache=cache
    )


@pytest.mark.parametrize("fetch", [_fetch_file, _fetch_member], ids=["file", "member"])
@pytest.mark.parametrize(
    ("reply", "error"),
    [
        (
            lambda: _PortalResponse(
                f"<html>Logged in as {_USER}. Wrong password.</html>".encode(),
                "text/html; charset=UTF-8",
            ),
            PermissionError,
        ),
        (_echoing_http_error, OSError),
        (lambda: urllib.error.URLError(OSError(f"refused {_USER}")), OSError),
        # http.client errors are not URLError: the server hangs up, or
        # reading the headers times out.
        (lambda: http.client.RemoteDisconnected(f"closed for {_USER}"), OSError),
        (lambda: http.client.BadStatusLine(f"{_USER}:{_PASSWORD}"), OSError),
        (lambda: TimeoutError(f"timed out for {_USER}"), OSError),
        (
            lambda: _PortalResponse(
                _tar_gz({_MEMBER: b"file bytes"}), "application/octet-stream"
            ),
            ValueError,
        ),
    ],
    ids=[
        "login-page",
        "http-error",
        "network-error",
        "remote-disconnected",
        "bad-status-line",
        "header-timeout",
        "checksum-mismatch",
    ],
)
def test_crcns_fetch_never_logs_credentials(
    portal, fetch, reply, error, tmp_path, capfd, caplog
):
    caplog.set_level(logging.DEBUG)
    portal.reply = reply()
    raised = _raised(fetch, tmp_path)
    assert isinstance(raised, error)
    # The credentials were sent, by name from the environment ...
    assert portal.forms[0]["username"] == [_USER]
    assert portal.forms[0]["password"] == [_PASSWORD]
    assert portal.forms[0]["fn"] == ["hc-31/data/resSave.mat"]
    # ... and appear nowhere a user or a log would see them, not even in a
    # traceback that shows local variables (pytest --showlocals, IPython).
    out, err = capfd.readouterr()
    shown = _everything_shown(raised) + out + err + caplog.text
    assert "hc-31/data/resSave.mat" in shown
    for secret in (_USER, _PASSWORD):
        assert secret not in shown
    assert [f for f in tmp_path.rglob("*") if f.is_file()] == []


def test_crcns_missing_member_error_shows_no_credentials(portal, tmp_path):
    portal.reply = _PortalResponse(
        _tar_gz({_MEMBER: b"file bytes"}), "application/octet-stream"
    )
    raised = _raised(_fetch_member, tmp_path, "s/absent")
    assert isinstance(raised, FileNotFoundError)
    assert "no member" in str(raised)
    for secret in (_USER, _PASSWORD):
        assert secret not in _everything_shown(raised)


@pytest.mark.parametrize(
    ("present", "absent"),
    [
        ({"CRCNS_USERNAME": _USER}, "CRCNS_PASSWORD"),
        ({"CRCNS_PASSWORD": _PASSWORD}, "CRCNS_USERNAME"),
        ({}, "CRCNS_USERNAME and CRCNS_PASSWORD"),
    ],
    ids=["no-password", "no-username", "neither"],
)
def test_crcns_fetch_requires_the_account_variables(
    monkeypatch, tmp_path, present, absent
):
    monkeypatch.delenv("CRCNS_USERNAME", raising=False)
    monkeypatch.delenv("CRCNS_PASSWORD", raising=False)
    for name, value in present.items():
        monkeypatch.setenv(name, value)
    raised = _raised(_fetch_file, tmp_path)
    assert isinstance(raised, KeyError)
    assert f"{absent} not set" in str(raised)
    shown = _everything_shown(raised)
    for secret in present.values():
        assert secret not in shown


def test_crcns_tar_members_refuses_members_sharing_a_base_name(portal, tmp_path):
    with pytest.raises(ValueError, match="share a base name"):
        _fetch.fetch_crcns_tar_members(
            "hc-3/top/s.tar.gz",
            {"a/s.whl": "0" * 64, "b/s.whl": "1" * 64},
            subdir="crcns",
            cache=tmp_path,
        )
    assert portal.forms == []


def test_crcns_tar_members_extracts_only_the_pinned_members(portal, tmp_path):
    members = {"s/s.whl": b"1 2 3 4\n", "s/s.eeg": b"\0" * 100, "s/s.res.1": b"20\n"}
    portal.reply = _PortalResponse(_tar_gz(members), "application/octet-stream")
    pins = {
        name: hashlib.sha256(members[name]).hexdigest()
        for name in ("s/s.whl", "s/s.res.1")
    }
    paths = _fetch.fetch_crcns_tar_members(
        "hc-3/top/s.tar.gz", pins, subdir="crcns", cache=tmp_path
    )
    assert {name: p.read_bytes() for name, p in paths.items()} == {
        name: members[name] for name in pins
    }
    assert sorted(p.name for p in (tmp_path / "crcns").iterdir()) == [
        "s.res.1",
        "s.whl",
    ]
    portal.reply = urllib.error.URLError("offline")  # cached copies need no network
    assert (
        _fetch.fetch_crcns_tar_members(
            "hc-3/top/s.tar.gz", pins, subdir="crcns", cache=tmp_path
        )
        == paths
    )
    assert len(portal.forms) == 1
    portal.reply = _PortalResponse(_tar_gz(members), "application/octet-stream")
    with pytest.raises(FileNotFoundError, match=r"no member\(s\) \['s/absent'\]"):
        _fetch.fetch_crcns_tar_members(
            "hc-3/top/s.tar.gz", {"s/absent": "0" * 64}, subdir="crcns", cache=tmp_path
        )
