"""Synthetic recordings for the dataset tests, built with neurospatial.simulation.

Three seeded sessions: a 1 m square box with positions and headings, a
T-maze track graph, and a heading-only session with no positions (as for a
dataset that publishes head direction alone). Each passes every refusing
check against its ``Stated`` facts.
"""

from __future__ import annotations

import builtins
import dataclasses
import sys

import numpy as np
import pandas as pd
import pytest

from neurospatial.behavior import heading_from_velocity
from neurospatial.datasets import Provenance, Recording
from neurospatial.datasets._checks import Stated
from neurospatial.simulation import open_field_session, tmaze_alternation_session


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Let a realdata test download into a cold cache: its DownloadWarning is shown.

    Everywhere else a DownloadWarning is an error (the suite's warnings filter),
    so a synthetic test that downloads must expect it.
    """
    for item in items:
        if item.get_closest_marker("realdata"):
            item.add_marker(
                pytest.mark.filterwarnings(
                    "default::neurospatial.datasets.DownloadWarning"
                )
            )


PROVENANCE = Provenance(
    archive="Synthetic 0",
    url="file:///synthetic/session.npz",
    sha256="0" * 64,
    size_bytes=1024,
    neurospatial_version="0.0",
    fetched_at="2026-11-03T00:00:00+00:00",
)


def unit_table(n_units: int) -> pd.DataFrame:
    """CA1 units labelled by the authors as pyramidal, ids 0 to n_units - 1."""
    return pd.DataFrame(
        {"brain_region": ["CA1"] * n_units, "label": ["pyramidal"] * n_units},
        index=pd.Index(np.arange(n_units), name="unit_id"),
    )


def _identity(session: str) -> dict:
    return {
        "dataset": "synthetic_2026",
        "session": session,
        "paper": "Synthetic et al. (2026)",
        "animal": "simulated rat s1",
        "apparatus": "simulated apparatus",
        "brain_regions": ("CA1",),
        "license": "CC0-1.0",
        "citation": "Synthetic A, Simulated B (2026) A simulated session. "
        "neurospatial test data.",
        "provenance": PROVENANCE,
        "clock": "the simulation's clock, from 0 s",
    }


@pytest.fixture(scope="session")
def open_field():
    """60 s in a 1 m box at 100 Hz, six place cells."""
    return open_field_session(duration=60.0, n_place_cells=6, rng=0)


@pytest.fixture(scope="session")
def tmaze():
    """60 s of T-maze alternation at 500 Hz, six place cells."""
    return tmaze_alternation_session(duration=60.0, n_trials=4, n_place_cells=6, rng=0)


@pytest.fixture
def square_recording(open_field) -> Recording:
    """The 1 m box with positions and headings (the direction of movement)."""
    return Recording(
        **_identity("square"),
        times=open_field.times,
        spike_times=tuple(open_field.spike_times),
        units=unit_table(len(open_field.spike_times)),
        env=open_field.env,
        env_kind="declared",
        env_source="1 m square box (simulation)",
        positions=open_field.positions,
        position_units="cm",
        tracking="simulated head position",
        headings=heading_from_velocity(open_field.times, open_field.positions),
        heading_source="direction of movement of the simulated path",
        epochs={"run": np.array([[0.0, 60.0]])},
    )


@pytest.fixture
def long_box_recording(square_recording) -> Recording:
    """2000 s at 100 Hz on a smooth closed path in the box, no headings.

    Long enough that a few tracking jumps, each spread over the 0.1 s of the
    speed's smoothing, stay within the fastest 0.1 % of samples.
    """
    times = np.arange(200_000) / 100.0
    positions = np.column_stack(
        [
            50.0 + 45.0 * np.sin(2 * np.pi * times / 17.0),
            50.0 + 45.0 * np.sin(2 * np.pi * times / 23.0),
        ]
    )
    return dataclasses.replace(
        square_recording,
        times=times,
        positions=positions,
        headings=None,
        heading_source="",
    )


@pytest.fixture
def jumpy_box_recording(long_box_recording) -> Recording:
    """``long_box_recording`` with three tracking jumps: 20 cm east for 0.2 s.

    A tracker that latches onto a reflection; one-sample glitches are mostly
    smoothed away by the speed's 0.1 s smoothing.
    """
    positions = np.array(long_box_recording.positions)
    for start in (40_000, 90_000, 150_000):
        positions[start : start + 20, 0] += 20.0
    return dataclasses.replace(long_box_recording, positions=positions)


@pytest.fixture
def track_recording(tmaze) -> Recording:
    """The T-maze graph; the simulated animal holds still at each arm end."""
    return Recording(
        **_identity("tmaze"),
        times=tmaze.times,
        spike_times=tuple(tmaze.spike_times),
        units=unit_table(len(tmaze.spike_times)),
        env=tmaze.env,
        env_kind="declared",
        env_source="T-maze track graph (simulation)",
        positions=tmaze.positions,
        position_units="cm",
        tracking="simulated head position",
    )


@pytest.fixture
def heading_recording(open_field) -> Recording:
    """The box session with headings and no positions or environment."""
    return Recording(
        **_identity("heading"),
        times=open_field.times,
        spike_times=tuple(open_field.spike_times),
        units=unit_table(len(open_field.spike_times)),
        tracking="simulated head direction",
        headings=heading_from_velocity(open_field.times, open_field.positions),
        heading_source="direction of movement of the simulated path",
    )


@pytest.fixture
def square_stated() -> Stated:
    """The box: 100 Hz, a largest span between 80 and 120 cm."""
    return Stated(sampling_rate=100.0, position_span=(80.0, 120.0))


@pytest.fixture
def track_stated() -> Stated:
    """The T-maze: 500 Hz, a largest span between 80 and 120 cm."""
    return Stated(sampling_rate=500.0, position_span=(80.0, 120.0))


@pytest.fixture
def heading_stated() -> Stated:
    """The heading-only session: 100 Hz."""
    return Stated(sampling_rate=100.0)


@pytest.fixture
def block_import(monkeypatch):
    """``block_import(name)``: imports of ``name`` fail as if it were not installed."""

    def block(module: str) -> None:
        real_import = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name == module:
                msg = f"No module named {module!r}"
                raise ModuleNotFoundError(msg, name=module)
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked)
        monkeypatch.delitem(sys.modules, module, raising=False)

    return block


@pytest.fixture
def without_pynwb(block_import):
    """Imports of pynwb fail as they do without the ``datasets`` extra."""
    block_import("pynwb")


@pytest.fixture
def no_download(monkeypatch):
    """Any download, of a file or of a zip member, fails the test."""
    from neurospatial.datasets import _fetch

    def refuse(*args, **kwargs):
        msg = "a file was downloaded"
        raise AssertionError(msg)

    monkeypatch.setattr(_fetch, "fetch", refuse)
    monkeypatch.setattr(_fetch, "fetch_zip_member", refuse)
    monkeypatch.setattr(_fetch, "fetch_crcns_tar_members", refuse)


@pytest.fixture
def session_entry():
    """``session_entry(dataset, session=None)``: a registry entry, the default's by default."""
    from neurospatial.datasets._load import _session_entry

    def entry(dataset: str, session: str | None = None):
        return _session_entry(dataset, session)

    return entry


@pytest.fixture
def assert_default_session():
    """``assert_default_session(rec, entry)``: what every real default session shows.

    The registry's default session, verified by hand, passing every refusing
    check, with the registry's unit count and checksum.
    """

    def check(rec: Recording, entry) -> None:
        assert entry.default
        assert rec.session == entry.session
        assert rec.verified
        assert rec.failed_checks == ()
        assert rec.n_units == entry.n_units
        assert rec.provenance.sha256 == entry.sha256

    return check
