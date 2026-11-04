"""Synthetic recordings for the dataset tests, built with neurospatial.simulation.

Three seeded sessions: a 1 m square box with positions and headings, a
T-maze track graph, and a heading-only session with no positions (as for a
dataset that publishes head direction alone). Each passes every refusing
check against its ``Stated`` facts.
"""

from __future__ import annotations

import builtins
import sys

import numpy as np
import pandas as pd
import pytest

from neurospatial.behavior import heading_from_velocity
from neurospatial.datasets import Provenance, Recording
from neurospatial.datasets._checks import Stated
from neurospatial.simulation import open_field_session, tmaze_alternation_session

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
def without_pynwb(monkeypatch):
    """Imports of pynwb fail as they do without the ``datasets`` extra."""
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "pynwb":
            msg = "No module named 'pynwb'"
            raise ModuleNotFoundError(msg, name="pynwb")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    monkeypatch.delitem(sys.modules, "pynwb", raising=False)
