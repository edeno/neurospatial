"""decode_position accepts a SpatialRatesResult directly (Task 1.4f).

Passing a population rate result object (anything exposing ``firing_rates``)
should be equivalent to passing its ``firing_rates`` array, removing the
``np.stack([r.firing_rate ...])`` glue between encode and decode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.decoding.posterior import decode_position
from neurospatial.encoding.spatial import compute_spatial_rates


def _setup() -> tuple[Environment, np.ndarray, object, float]:
    rng = np.random.default_rng(0)
    positions = rng.uniform(0, 100, (1000, 2))
    env = Environment.from_samples(positions, bin_size=10.0)
    times = np.linspace(0, 100, 1000)
    spike_times = [np.sort(rng.uniform(0, 100, n)) for n in (40, 50, 30)]
    rates_result = compute_spatial_rates(
        env, spike_times, times, positions, bandwidth=10.0, fill_value=0.0
    )

    n_time_bins, n_neurons = 50, len(spike_times)
    spike_counts = rng.poisson(2, (n_time_bins, n_neurons)).astype(np.int64)
    return env, spike_counts, rates_result, 0.025


def test_decode_position_accepts_result_object() -> None:
    env, spike_counts, rates_result, dt = _setup()

    from_result = decode_position(env, spike_counts, rates_result, dt)
    from_array = decode_position(
        env, spike_counts, np.asarray(rates_result.firing_rates), dt
    )

    np.testing.assert_allclose(
        from_result.posterior, from_array.posterior, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_array_equal(from_result.map_estimate, from_array.map_estimate)


@dataclass
class _BadRates:
    firing_rates: Any


def test_decode_position_none_firing_rates_clear_error() -> None:
    """A result whose .firing_rates is None raises a clear ValueError (I2)."""
    env, spike_counts, _rates_result, dt = _setup()
    with pytest.raises(ValueError, match=r"\.firing_rates.*2-D.*None"):
        decode_position(env, spike_counts, _BadRates(firing_rates=None), dt)


def test_decode_position_dict_firing_rates_clear_error() -> None:
    """A result whose .firing_rates is a dict raises a clear ValueError (I2)."""
    env, spike_counts, _rates_result, dt = _setup()
    bad = _BadRates(firing_rates={"north": np.zeros(5), "south": np.zeros(5)})
    with pytest.raises(ValueError, match=r"\.firing_rates.*2-D"):
        decode_position(env, spike_counts, bad, dt)
