"""load() and fetch_all() on a synthetic dataset: registry, fetch, loader, checks."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

import pytest

from neurospatial import InputError
from neurospatial.datasets import (
    DatasetCheckError,
    _catalog,
    _fetch,
    _load,
    fetch_all,
    load,
)
from neurospatial.datasets._checks import run_checks
from neurospatial.datasets._summary import summary_text

DATASET = "synthetic_2026"
#: Sessions of the synthetic dataset: the second has spikes off the clock.
SESSIONS = ("s1", "clock_off")


def _registry(files: dict[str, Path]) -> dict:
    return {
        "dataset": DATASET,
        "paper": "Synthetic et al. (2026)",
        "paper_citation": "Synthetic A (2026) A simulated session.",
        "species": "simulated rat",
        "brain_regions": ["CA1"],
        "apparatus": "1 m square box",
        "license": "CC0-1.0",
        "archive": {"name": "Synthetic 0", "citation": "Synthetic 0 archive."},
        "stated": {
            "sampling_rate": {
                "value": 100.0,
                "basis": "stated",
                "source": "the simulation",
            },
            "position_span": {
                "value": [80.0, 120.0],
                "basis": "stated",
                "source": "the simulated 1 m box",
            },
        },
        "sessions": [
            {
                "session": session,
                "archive": "Synthetic 0",
                "url": path.as_uri(),
                "file_name": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size_bytes": path.stat().st_size,
                "verified": session == "s1",
                "default": session == "s1",
                "n_units": 6,
                "verification": "checked against the simulation" * (session == "s1"),
            }
            for session, path in files.items()
        ],
    }


def _no_download(*args, **kwargs):
    msg = "a file was downloaded"
    raise AssertionError(msg)


@pytest.fixture
def synthetic(monkeypatch, tmp_path, square_recording):
    """A registered synthetic dataset whose files are local and whose loader is fake.

    The loader returns the conftest square session with the entry's identity;
    for ``clock_off`` its spikes are shifted 1000 s past the tracking.
    """
    archive = tmp_path / "archive"
    archive.mkdir()
    files = {}
    for session in SESSIONS:
        files[session] = archive / f"{session}.bin"
        files[session].write_bytes(session.encode() * 100)
    registry = tmp_path / "registry"
    registry.mkdir()
    (registry / f"{DATASET}.json").write_text(json.dumps(_registry(files)))
    monkeypatch.setattr(_catalog, "REGISTRY", registry)

    calls = []

    def load_session(entry, path_or_stream):
        calls.append((entry.session, Path(path_or_stream)))
        shift = 1000.0 if entry.session == "clock_off" else 0.0
        return dataclasses.replace(
            square_recording,
            **entry.identity(path_or_stream),
            spike_times=tuple(t + shift for t in square_recording.spike_times),
            watch_outs=("the dataset's own pitfall",),
        )

    module = types.ModuleType(f"neurospatial.datasets.{DATASET}")
    module.REQUIRES = ()
    module.WATCH_OUTS = {"s1": ("the dataset's own pitfall",)}
    module.load_session = load_session
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return types.SimpleNamespace(cache=tmp_path / "cache", calls=calls, files=files)


def test_load_gives_the_default_session_checked(synthetic, square_stated):
    rec = load(DATASET, cache=synthetic.cache)
    assert rec.session == "s1"
    assert rec.verified
    cached = synthetic.cache / DATASET / "s1.bin"
    assert synthetic.calls == [("s1", cached)]
    assert cached.read_bytes() == synthetic.files["s1"].read_bytes()
    assert rec.provenance.url == synthetic.files["s1"].as_uri()
    fetched = datetime.fromisoformat(rec.provenance.fetched_at)
    mtime = datetime.fromtimestamp(cached.stat().st_mtime, timezone.utc)
    assert abs((fetched - mtime).total_seconds()) < 1
    # load() is the loader's recording checked once with the registry's facts.
    unchecked = dataclasses.replace(rec, watch_outs=("the dataset's own pitfall",))
    assert rec.watch_outs == run_checks(unchecked, stated=square_stated).watch_outs


def test_accepted_failure_travels(synthetic):
    with pytest.raises(DatasetCheckError) as caught:
        load(DATASET, "clock_off", cache=synthetic.cache)
    assert caught.value.checks == ("clock",)
    assert "accept_failed_checks=('clock',)" in str(caught.value)

    rec = load(
        DATASET, "clock_off", cache=synthetic.cache, accept_failed_checks=("clock",)
    )
    assert rec.failed_checks == ("clock",)
    assert rec.watch_outs[0].startswith("failed the clock check, loaded with")
    assert rec.watch_outs[1] == "the dataset's own pitfall"
    watch_out = summary_text(rec).split("\nWatch out", 1)[1].splitlines()
    assert watch_out[1].startswith("  ! failed the clock check")
    # Accepting a check that does not fail changes nothing.
    plain = load(DATASET, "s1", cache=synthetic.cache)
    accepted = load(
        DATASET, "s1", cache=synthetic.cache, accept_failed_checks=("clock", "units")
    )
    assert accepted.failed_checks == plain.failed_checks == ()
    assert accepted.watch_outs == plain.watch_outs


def test_unknown_names_are_refused_before_any_download(synthetic, monkeypatch):
    monkeypatch.setattr(_fetch, "fetch", _no_download)
    with pytest.raises(InputError, match=r"(?s)No dataset 'nope'.*synthetic_2026"):
        load("nope")
    with pytest.raises(InputError, match=r"(?s)no session 'sX'.*s1, clock_off"):
        load(DATASET, "sX")
    with pytest.raises(InputError, match="accept_failed_checks"):
        load(DATASET, accept_failed_checks=("speed",))


def test_stream_needs_a_dandi_asset(synthetic):
    with pytest.raises(InputError, match="cannot be streamed"):
        load(DATASET, stream=True)


def test_missing_extra_names_the_install(synthetic, monkeypatch, without_pynwb):
    """Without the datasets extra, load says how to install it, before downloading."""
    monkeypatch.setattr(
        sys.modules[f"neurospatial.datasets.{DATASET}"], "REQUIRES", ("pynwb",)
    )
    monkeypatch.setattr(_fetch, "fetch", _no_download)
    with pytest.raises(ImportError, match=r"pip install 'neurospatial\[datasets\]'"):
        load(DATASET, cache=synthetic.cache)


def test_fetch_all_downloads_every_session(synthetic, caplog):
    with caplog.at_level(logging.INFO, logger="neurospatial.datasets"):
        paths = fetch_all(DATASET, cache=synthetic.cache)
    assert [path.name for path in paths] == ["s1.bin", "clock_off.bin"]
    assert all(
        path.read_bytes() == synthetic.files[path.stem].read_bytes() for path in paths
    )
    assert "2 sessions, 1.1 kB in all; 2 to download (1.1 kB)" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="neurospatial.datasets"):
        assert fetch_all(DATASET, cache=synthetic.cache) == paths
    assert "0 to download (0 B)" in caplog.text


def test_fetch_all_refuses_without_room(synthetic, monkeypatch):
    usage = types.SimpleNamespace(total=10_000, used=9_000, free=1_000)
    monkeypatch.setattr(_load.shutil, "disk_usage", lambda path: usage)
    with pytest.raises(OSError, match=r"needs 1.1 kB for 2 sessions.*1 kB free"):
        fetch_all(DATASET, cache=synthetic.cache)
    assert not (synthetic.cache / DATASET).exists()
