"""Simulation holder attributes compose directly with array analyses."""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from neurospatial import compute_spatial_rates
from neurospatial.simulation import open_field_session


@pytest.fixture(scope="module")
def sim():
    return open_field_session(duration=60, seed=0)


def test_attributes_feed_analysis(sim):
    rates = compute_spatial_rates(
        sim.env, sim.spike_times, sim.times, sim.positions, unit_ids=sim.unit_ids
    )
    np.testing.assert_array_equal(rates.unit_ids, np.arange(len(sim.spike_times)))
    assert set(sim.ground_truth) == set(sim.unit_ids)


@pytest.mark.parametrize("field", ["unit_ids", "models"])
def test_mismatched_lengths_raise(sim, field):
    with pytest.raises(ValueError, match=r"must match one-to-one.*\nFix:"):
        replace(sim, **{field: getattr(sim, field)[:-1]})


def test_holder_is_frozen(sim):
    with pytest.raises(FrozenInstanceError):
        sim.times = np.array([0.0])
