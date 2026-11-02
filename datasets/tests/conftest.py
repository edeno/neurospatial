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

    session = open_field_session(duration=600.0, n_place_cells=50, seed=0)
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
