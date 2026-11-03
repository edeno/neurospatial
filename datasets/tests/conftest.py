"""Shared fixtures: a simulated open-field recording with known place fields."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _record import Recording, alternating_blocks


@pytest.fixture(scope="session")
def simulated_recording() -> Recording:
    """600 s of 50 place cells in a 100 cm box (seeded), as a loader would return it."""
    from neurospatial.simulation import open_field_session

    session = open_field_session(duration=600.0, n_place_cells=50, rng=0)
    train, test = alternating_blocks(session.times[0], session.times[-1], 60.0)
    return Recording(
        dataset="simulated_open_field",
        session="seed0",
        source="neurospatial.simulation.open_field_session",
        license="n/a",
        times=session.times,
        positions=session.positions,
        position_units="cm",
        spike_times=tuple(np.sort(train) for train in session.spike_times),
        unit_ids=np.asarray(session.unit_ids),
        params={
            "bin_size": 2.5,
            "dt": 0.25,
            "train_epochs": train,
            "test_epochs": test,
        },
    )


@pytest.fixture
def loader_module() -> SimpleNamespace:
    """A stand-in loader module with no expectations or dataset hazards."""
    return SimpleNamespace(EXPECTATIONS={}, HAZARDS={})


IMAGING_FIELD = np.array([30.0, 70.0])  # cm


@pytest.fixture(scope="session")
def imaging_recording() -> Recording:
    """A 100 cm box covered by a seeded random walk; one cell with a field at IMAGING_FIELD."""
    rng = np.random.default_rng(4)
    times = np.arange(30_000) / 30.0
    steps = rng.normal(0, 1.5, (times.size, 2))
    positions = np.abs(
        (np.cumsum(steps, axis=0) + 50.0) % 200.0 - 100.0
    )  # reflect into [0, 100]
    distance2 = np.sum((positions - IMAGING_FIELD) ** 2, axis=1)
    rate = 0.2 + 15.0 * np.exp(-distance2 / (2 * 8.0**2))
    return Recording(
        dataset="imaging_sim",
        session="0",
        source="simulated",
        license="n/a",
        times=times,
        positions=positions,
        position_units="cm",
        activity=rate[:, None],
        activity_times=times.copy(),
        activity_kind="binned_rate",
        params={"bin_size": 5.0},
    )
