"""The documented workflow on clean simulated data emits no warnings.

Warnings guard against silently discarded or mis-scaled data. A warning that
fires on this path, where nothing is wrong with the input, is noise that
trains users to ignore the real ones, so it is a bug in the warning.
"""

from __future__ import annotations

import warnings

import numpy as np
import shapely

from neurospatial import Environment
from neurospatial.behavior import compute_speed
from neurospatial.decoding import decode_session
from neurospatial.encoding import compute_spatial_rates
from neurospatial.events import peri_event_histogram
from neurospatial.simulation import open_field_session


def test_documented_workflow_emits_no_warnings() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        session = open_field_session(duration=120.0, n_place_cells=8, seed=0)
        times, positions, spike_times = (
            session.times,
            session.positions,
            session.spike_times,
        )
        env = Environment.from_polygon(
            shapely.box(0.0, 0.0, 100.0, 100.0), bin_size=4.0, units="cm"
        )
        rates = compute_spatial_rates(env, spike_times, times, positions)
        table = rates.summary_table()
        decoded = decode_session(env, spike_times, times, positions, dt=0.025)
        event_times = np.arange(10.0, 110.0, 10.0)
        psth = peri_event_histogram(spike_times[0], event_times, (-0.5, 0.5))
        speed = compute_speed(times, positions)

    assert len(table) == len(spike_times)
    assert np.isfinite(rates.firing_rates).any()
    assert decoded.posterior.shape == (decoded.times.size, env.n_bins)
    assert psth.n_events == event_times.size
    assert speed.shape == times.shape
