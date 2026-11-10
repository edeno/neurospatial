from __future__ import annotations

import contextlib
import datetime
import email.message
import functools
import hashlib
import http.client
import http.server
import io
import json
import logging
import os
import subprocess
import sys
import tarfile
import threading
import traceback
import urllib.error
import urllib.parse
import urllib.request
import uuid
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from neurospatial.datasets import _fetch
from neurospatial.datasets._fetch import fetch


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


def test_download_is_hashed_as_it_is_written(source, tmp_path, monkeypatch):
    """A download is never read back to hash it; a wrong digest is still refused."""
    url, digest = source

    def read_back(path):
        msg = f"{path} was read back to hash it"
        raise AssertionError(msg)

    monkeypatch.setattr(_fetch, "file_sha256", read_back)
    cache = tmp_path / "cache"
    path = fetch(url, "data.bin", sha256=digest, subdir="demo", cache=cache)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    with pytest.raises(ValueError, match="does not match the pinned"):
        fetch(url, "other.bin", sha256="0" * 64, subdir="demo", cache=cache)
    assert not (cache / "demo" / "other.bin").exists()


@pytest.fixture
def hashes(monkeypatch):
    """The files hashed by reading them, in order."""
    hashed = []
    file_sha256 = _fetch.file_sha256

    def counted(path):
        hashed.append(path)
        return file_sha256(path)

    monkeypatch.setattr(_fetch, "file_sha256", counted)
    return hashed


def _record(cache, subdir, name):
    return json.loads((cache / ".verified" / subdir / f"{name}.verified").read_text())


def test_a_recorded_file_is_trusted_without_hashing(source, tmp_path, hashes):
    """A download records its digest, size and time; an unchanged file is not hashed."""
    url, digest = source
    cache = tmp_path / "cache"
    path = fetch(url, "data.bin", sha256=digest, subdir="demo", cache=cache)
    stat = path.stat()
    assert _record(cache, "demo", "data.bin") == {
        "sha256": digest,
        "st_size": stat.st_size,
        "st_mtime_ns": stat.st_mtime_ns,
    }
    assert fetch(url, "data.bin", sha256=digest, subdir="demo", cache=cache) == path
    assert hashes == []
    # The record sits apart from the session's files.
    assert sorted(p.name for p in (cache / "demo").iterdir()) == ["data.bin"]


def test_a_file_whose_time_changed_is_hashed_and_recorded_again(
    source, tmp_path, hashes
):
    url, digest = source
    cache = tmp_path / "cache"
    path = fetch(url, "data.bin", sha256=digest, subdir="demo", cache=cache)
    touched = path.stat().st_mtime_ns + 5_000_000_000
    os.utime(path, ns=(touched, touched))
    assert fetch(url, "data.bin", sha256=digest, subdir="demo", cache=cache) == path
    assert hashes == [path]
    assert _record(cache, "demo", "data.bin")["st_mtime_ns"] == touched
    fetch(url, "data.bin", sha256=digest, subdir="demo", cache=cache)
    assert hashes == [path]


def test_an_edited_file_is_hashed_and_refused(source, tmp_path, hashes):
    """A changed size disagrees with the record, so the file is hashed: a mismatch."""
    url, digest = source
    cache = tmp_path / "cache"
    path = fetch(url, "data.bin", sha256=digest, subdir="demo", cache=cache)
    with path.open("ab") as stream:
        stream.write(b"more")
    with pytest.raises(FileExistsError, match="different SHA-256"):
        fetch(url, "data.bin", sha256=digest, subdir="demo", cache=cache)
    assert hashes == [path]


def test_a_cached_file_without_a_record_is_hashed_once(source, tmp_path, hashes):
    """A file cached before records were kept is hashed, then recorded."""
    url, digest = source
    cache = tmp_path / "cache"
    path = fetch(url, "data.bin", sha256=digest, subdir="demo", cache=cache)
    (cache / ".verified" / "demo" / "data.bin.verified").unlink()
    fetch(url, "data.bin", sha256=digest, subdir="demo", cache=cache)
    assert hashes == [path]
    assert _record(cache, "demo", "data.bin")["sha256"] == digest
    fetch(url, "data.bin", sha256=digest, subdir="demo", cache=cache)
    assert hashes == [path]


def test_a_record_of_another_digest_is_not_trusted(source, tmp_path, hashes):
    """A pin that changed is checked against the file, not against the record."""
    url, digest = source
    cache = tmp_path / "cache"
    path = fetch(url, "data.bin", sha256=digest, subdir="demo", cache=cache)
    with pytest.raises(FileExistsError, match="different SHA-256"):
        fetch(url, "data.bin", sha256="0" * 64, subdir="demo", cache=cache)
    assert hashes == [path]


def test_zip_members_are_recorded_outside_their_directory(tmp_path, hashes):
    import zipfile

    archive = tmp_path / "archive.zip"
    payload = b"member bytes " * 100
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("folder/member.mat", payload)
    digest = hashlib.sha256(payload).hexdigest()
    cache = tmp_path / "cache"

    def member():
        return _fetch.fetch_zip_member(
            archive.as_uri(),
            "folder/member.mat",
            sha256=digest,
            subdir="d/s",
            cache=cache,
        )

    path = member()
    assert member() == path
    assert hashes == []
    assert sorted(p.name for p in path.parent.iterdir()) == ["member.mat"]
    assert _record(cache, "d/s", "member.mat")["sha256"] == digest


def test_fetch_gives_up_on_a_stalled_server(source, tmp_path, monkeypatch):
    """The download's urlopen has a timeout, so a stalled server cannot hang it."""
    url, digest = source
    opened = urllib.request.urlopen
    timeouts = []

    def recording(url, *args, **kwargs):
        timeouts.append(kwargs.get("timeout"))
        return opened(url, *args, **kwargs)

    monkeypatch.setattr(urllib.request, "urlopen", recording)
    fetch(url, "data.bin", sha256=digest, subdir="demo", cache=tmp_path / "cache")
    assert timeouts == [_fetch.TIMEOUT]


def test_fetch_names_itself_to_the_server(source, tmp_path, monkeypatch):
    """Mendeley Data answers urllib's default User-Agent with HTTP 403."""
    url, digest = source
    opened = urllib.request.urlopen
    agents = []

    def recording(request, *args, **kwargs):
        agents.append(request.get_header("User-agent"))
        return opened(request, *args, **kwargs)

    monkeypatch.setattr(urllib.request, "urlopen", recording)
    fetch(url, "data.bin", sha256=digest, subdir="demo", cache=tmp_path / "cache")
    assert agents == ["neurospatial"]


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
    with pytest.raises(ValueError, match="does not match the pinned") as refused:
        fetch(url, "data.bin", sha256="0" * 64, subdir="demo", cache=tmp_path / "cache")
    assert list((tmp_path / "cache" / "demo").iterdir()) == []
    assert "\nWhy: " in str(refused.value)
    assert "\nFix: load again to retry" in str(refused.value)


def test_cache_root_reads_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("NEUROSPATIAL_DATA_DIR", str(tmp_path / "elsewhere"))
    assert _fetch.cache_root() == tmp_path / "elsewhere"


def test_cache_root_precedence(tmp_path, monkeypatch):
    monkeypatch.setenv("NEUROSPATIAL_DATA_DIR", str(tmp_path / "env"))
    assert _fetch.cache_root(tmp_path / "arg") == tmp_path / "arg"
    assert (tmp_path / "arg").is_dir()
    assert not (tmp_path / "env").exists()


@pytest.mark.parametrize(
    ("platform", "expected"),
    [
        ("linux", Path("/home/u/.cache/neurospatial/datasets")),
        ("darwin", Path("/home/u/.cache/neurospatial/datasets")),
        ("win32", Path("/local/neurospatial/datasets")),
    ],
)
def test_cache_root_per_user_default(tmp_path, monkeypatch, platform, expected):
    monkeypatch.delenv("NEUROSPATIAL_DATA_DIR", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", "/local")
    monkeypatch.setattr(_fetch.sys, "platform", platform)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: Path("/home/u")))
    monkeypatch.setattr(Path, "mkdir", lambda *args, **kwargs: None)
    assert _fetch.cache_root() == expected


@pytest.mark.parametrize("module", ["pynwb", "fsspec", "inflate64"])
def test_missing_extra_names_the_install(tmp_path, module, block_import):
    block_import(module)
    calls = {
        "pynwb": lambda: _fetch.open_nwb(str(tmp_path / "x.nwb")).__enter__(),
        "fsspec": lambda: _fetch.fetch_zip_member(
            "http://x/y.zip", "m", sha256="0" * 64, subdir="d", cache=tmp_path
        ),
        "inflate64": lambda: _fetch._copy_deflate64(None, None, None),
    }
    with pytest.raises(ImportError, match=r"neurospatial\[datasets\]"):
        calls[module]()


def test_dandi_stream_url_refuses_changed_digest(monkeypatch):
    published = {"digest": {"dandi:sha2-256": "b" * 64}}

    def api(url, *, timeout):
        assert timeout == _fetch.TIMEOUT  # a stalled server cannot hang the load
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


@contextlib.contextmanager
def _served(directory):
    """Serve ``directory`` over HTTP on this machine; yield the server's URL."""
    handler = functools.partial(_RangeHandler, directory=str(directory))
    with http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            yield f"http://127.0.0.1:{server.server_port}"
        finally:
            server.shutdown()


def test_open_nwb_reads_local_and_http(tmp_path):
    pynwb = pytest.importorskip("pynwb")
    pytest.importorskip("fsspec")
    pytest.importorskip("h5py")
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

    with _served(tmp_path) as url:
        remote = positions(f"{url}/demo.nwb")
    np.testing.assert_array_equal(positions(str(path)), xy)
    np.testing.assert_array_equal(remote, xy)


#: A stand-in for ndx-events 0.2.0. Its ``LabeledEvents`` and
#: ``AnnotatedEventsTable`` types are in no later version of the extension, so
#: only the file's own copy can describe them.
_OLD_EVENTS_NAMESPACE = {
    "namespaces": [
        {
            "name": "ndx-events",
            "version": "0.2.0",
            "doc": "events",
            "author": ["author"],
            "contact": ["contact"],
            "schema": [
                {
                    "namespace": "core",
                    "neurodata_types": [
                        "NWBDataInterface",
                        "DynamicTable",
                        "VectorData",
                    ],
                },
                {"source": "ndx-events.extensions"},
            ],
        }
    ]
}
_OLD_EVENTS_TYPES = {
    "groups": [
        {
            "neurodata_type_def": "LabeledEvents",
            "neurodata_type_inc": "NWBDataInterface",
            "doc": "event times, each with a label",
            "datasets": [
                {
                    "name": "timestamps",
                    "dtype": "float64",
                    "dims": ["num_events"],
                    "shape": [None],
                    "doc": "event times",
                },
                {
                    "name": "data",
                    "dtype": "uint8",
                    "dims": ["num_events"],
                    "shape": [None],
                    "doc": "index of each event's label",
                    "attributes": [
                        {
                            "name": "labels",
                            "dtype": "text",
                            "dims": ["num_labels"],
                            "shape": [None],
                            "doc": "the labels",
                        }
                    ],
                },
            ],
        },
        {
            "neurodata_type_def": "AnnotatedEventsTable",
            "neurodata_type_inc": "DynamicTable",
            "doc": "a table with one column the type requires",
            "datasets": [
                {
                    "name": "event_times",
                    "neurodata_type_inc": "VectorData",
                    "dtype": "float64",
                    "dims": ["num_events"],
                    "shape": [None],
                    "doc": "event times",
                }
            ],
        },
    ]
}
_REWARDS = [[1.0, 2.5, 4.0], [0, 1, 0], ["left_reward", "right_reward"]]
_ANNOTATED = {"event_times": [0.5, 3.0], "duration": [0.1, 0.2]}
#: pynwb's warning that it ignored a namespace a file caches, as hdmf worded
#: it before and from 4.1.0.
_IGNORED_NAMESPACE = r"Ignoring (the following )?cached namespace"


@pytest.fixture
def old_events_file(tmp_path):
    """An NWB file with objects of both 0.2.0 types and the namespace defining them."""
    pynwb = pytest.importorskip("pynwb")
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "old_events.nwb"
    nwbfile = pynwb.NWBFile("demo", "id", datetime.datetime.now(datetime.timezone.utc))
    nwbfile.create_processing_module("behavior", "events")
    with pynwb.NWBHDF5IO(path, "w") as io_:
        io_.write(nwbfile)
    text = h5py.string_dtype()

    def typed(obj, namespace, neurodata_type, **attrs):
        obj.attrs.update(
            namespace=namespace,
            neurodata_type=neurodata_type,
            object_id=str(uuid.uuid4()),
            **attrs,
        )
        return obj

    with h5py.File(path, "a") as h5:
        cached = h5["specifications"]
        if "ndx-events" in cached:  # pynwb cached the version this process imported
            del cached["ndx-events"]
        spec = cached.create_group("ndx-events/0.2.0")
        spec.create_dataset(
            "namespace", data=json.dumps(_OLD_EVENTS_NAMESPACE), dtype=text
        )
        spec.create_dataset(
            "ndx-events.extensions", data=json.dumps(_OLD_EVENTS_TYPES), dtype=text
        )
        behavior = h5["processing/behavior"]

        times, codes, labels = _REWARDS
        rewards = typed(behavior.create_group("rewards"), "ndx-events", "LabeledEvents")
        rewards.create_dataset("timestamps", data=np.array(times))
        data = rewards.create_dataset("data", data=np.array(codes, dtype=np.uint8))
        data.attrs.create("labels", labels, dtype=text)

        table = typed(
            behavior.create_group("annotated"),
            "ndx-events",
            "AnnotatedEventsTable",
            description="events",
        )
        table.attrs.create("colnames", list(_ANNOTATED), dtype=text)
        n_rows = len(_ANNOTATED["event_times"])
        ids = table.create_dataset("id", data=np.arange(n_rows))
        typed(ids, "hdmf-common", "ElementIdentifiers")
        for name, values in _ANNOTATED.items():
            column = table.create_dataset(name, data=np.array(values))
            typed(column, "hdmf-common", "VectorData", description=name)
    return path


def _read_old_events(source):
    """The reward table as ``huszar_2022`` reads its own, and the annotated table."""
    with _fetch.open_nwb(source) as nwbfile:
        behavior = nwbfile.processing["behavior"]
        rewards = behavior["rewards"]
        return (
            [
                rewards.timestamps[:].tolist(),
                rewards.data[:].tolist(),
                list(rewards.fields["data__labels"]),
            ],
            behavior["annotated"].to_dataframe().to_dict("list"),
        )


def _loaded_version(namespace):
    """The version of ``namespace`` in pynwb's process-wide type map."""
    import pynwb

    return pynwb.get_type_map().namespace_catalog.get_namespace(namespace)["version"]


@pytest.mark.parametrize("where", ["path", "http"])
def test_open_nwb_reads_types_only_the_files_extension_version_has(
    old_events_file, where
):
    """Importing ndx-events 0.4.0 does not hide objects of 0.2.0 types."""
    pytest.importorskip("ndx_events")  # loads its own version into pynwb, for good
    assert _loaded_version("ndx-events") != "0.2.0"
    with contextlib.ExitStack() as stack:
        source = str(old_events_file)
        if where == "http":
            pytest.importorskip("fsspec")
            pytest.importorskip("aiohttp")
            url = stack.enter_context(_served(old_events_file.parent))
            source = f"{url}/{old_events_file.name}"
        stack.enter_context(warnings.catch_warnings())
        warnings.filterwarnings("error", message=_IGNORED_NAMESPACE)
        rewards, annotated = _read_old_events(source)
    assert rewards == _REWARDS
    assert annotated == _ANNOTATED


def test_open_nwb_reads_those_types_when_the_extension_is_not_imported(
    old_events_file,
):
    """In a new process, where nothing has loaded another ndx-events version."""
    script = (
        "import json, sys, warnings\n"
        "import pynwb\n"
        "from neurospatial.datasets import _fetch\n"
        "warnings.simplefilter('error')\n"
        "with _fetch.open_nwb(sys.argv[1]) as nwbfile:\n"
        "    behavior = nwbfile.processing['behavior']\n"
        "    rewards = behavior['rewards']\n"
        "    read = (\n"
        "        [\n"
        "            rewards.timestamps[:].tolist(),\n"
        "            rewards.data[:].tolist(),\n"
        "            list(rewards.fields['data__labels']),\n"
        "        ],\n"
        "        behavior['annotated'].to_dataframe().to_dict('list'),\n"
        "    )\n"
        "assert 'ndx_events' not in sys.modules\n"
        "print(json.dumps(read))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(old_events_file)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == [_REWARDS, _ANNOTATED]


def test_open_nwb_keeps_an_extension_loaded_at_the_files_version(tmp_path):
    """The file caches the imported ndx-events: the package's own class reads the table."""
    pynwb = pytest.importorskip("pynwb")
    ndx_events = pytest.importorskip("ndx_events")
    import pandas as pd

    from neurospatial.io.nwb import read_events, write_events

    events = pd.DataFrame({"timestamp": [1.0, 2.5, 4.0], "label": ["a", "b", "a"]})
    nwbfile = pynwb.NWBFile("demo", "id", datetime.datetime.now(datetime.timezone.utc))
    write_events(nwbfile, events, name="rewards")
    path = tmp_path / "events.nwb"
    with pynwb.NWBHDF5IO(path, "w") as io_:
        io_.write(nwbfile)
    cached = pynwb.NWBHDF5IO.get_namespaces(path=str(path))
    assert cached["ndx-events"] == _loaded_version("ndx-events")

    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=_IGNORED_NAMESPACE)
        with _fetch.open_nwb(str(path)) as read:
            table = read.processing["behavior"]["rewards"]
            assert isinstance(table, ndx_events.EventsTable)
            read_back = read_events(read, table_name="rewards")
    np.testing.assert_array_equal(read_back["timestamp"], events["timestamp"])
    assert list(read_back["label"]) == list(events["label"])


def test_open_nwb_says_what_to_do_when_hdmf_cannot_set_the_loaded_version_aside(
    old_events_file, monkeypatch
):
    """hdmf before 4.1.0 has no ``NamespaceCatalog.core_namespaces``."""
    pytest.importorskip("ndx_events")
    import hdmf.spec

    monkeypatch.delattr(hdmf.spec.NamespaceCatalog, "core_namespaces", raising=False)
    with pytest.raises(RuntimeError) as excinfo:
        _read_old_events(str(old_events_file))
    message = str(excinfo.value)
    assert "caches ndx-events 0.2.0" in message
    assert f"has loaded ndx-events {_loaded_version('ndx-events')}" in message
    assert "hdmf>=4.1.0" in message
    assert "has not imported" in message
    assert isinstance(excinfo.value.__cause__, AttributeError)


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
            if isinstance(state.reply, BaseException):
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
    except BaseException as exc:  # the test checks the type, Ctrl-C's too
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


@pytest.mark.parametrize("stop", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("fetch", [_fetch_file, _fetch_member], ids=["file", "member"])
def test_crcns_interrupt_stops_without_the_request_frames(
    portal, fetch, stop, tmp_path
):
    portal.reply = stop(f"{_USER}:{_PASSWORD}")
    raised = _raised(fetch, tmp_path)
    assert type(raised) is stop
    shown = _everything_shown(raised)
    for secret in (_USER, _PASSWORD):
        assert secret not in shown
    assert [f for f in tmp_path.rglob("*") if f.is_file()] == []


@pytest.mark.parametrize(("code", "status"), [(3, 3), (0, 0), (None, 1), ("x", 1)])
def test_crcns_system_exit_keeps_an_integer_exit_code(portal, code, status, tmp_path):
    # A message is not kept (it could hold the form), so it exits with 1.
    portal.reply = SystemExit(f"{_USER}:{_PASSWORD}" if code == "x" else code)
    raised = _raised(_fetch_file, tmp_path)
    assert type(raised) is SystemExit
    assert raised.code == status


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


def test_crcns_tar_members_extracts_only_the_pinned_members(
    portal, tmp_path, monkeypatch
):
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
    hashed = []
    with monkeypatch.context() as patched:
        patched.setattr(_fetch, "file_sha256", hashed.append)
        assert (
            _fetch.fetch_crcns_tar_members(
                "hc-3/top/s.tar.gz", pins, subdir="crcns", cache=tmp_path
            )
            == paths
        )
    assert hashed == []  # each member's record vouches for it
    assert len(portal.forms) == 1
    portal.reply = _PortalResponse(_tar_gz(members), "application/octet-stream")
    with pytest.raises(FileNotFoundError, match=r"no member\(s\) \['s/absent'\]"):
        _fetch.fetch_crcns_tar_members(
            "hc-3/top/s.tar.gz", {"s/absent": "0" * 64}, subdir="crcns", cache=tmp_path
        )
