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


def test_decode_position_1d_firing_rates_clear_error() -> None:
    """A result whose .firing_rates is 1-D raises a clear ValueError."""
    env, spike_counts, _rates_result, dt = _setup()
    bad = _BadRates(firing_rates=np.zeros(5))
    with pytest.raises(ValueError, match=r"\.firing_rates.*2-D"):
        decode_position(env, spike_counts, bad, dt)


def _capture_model_dtype(env, spike_counts, encoding_models, dt):
    """Decode while spying on the dtype of the encoding models that reach
    the likelihood inside ``_prepare_decode_inputs``."""
    import neurospatial.decoding.posterior as posterior_mod

    captured: dict[str, Any] = {}
    real = posterior_mod._prepare_decode_inputs

    def spy(env_, counts_, models_, *args, **kwargs):
        sc, em, mask = real(env_, counts_, models_, *args, **kwargs)
        captured["dtype"] = em.dtype
        return sc, em, mask

    posterior_mod._prepare_decode_inputs = spy  # type: ignore[assignment]
    try:
        decode_position(env, spike_counts, encoding_models, dt)
    finally:
        posterior_mod._prepare_decode_inputs = real  # type: ignore[assignment]
    return captured["dtype"]


def test_decode_position_result_object_preserves_float32() -> None:
    """A float32 rate-result object yields a float32 model -- same as the raw
    array path (Fix A: object path must not promote to float64)."""
    rng = np.random.default_rng(0)
    positions = rng.uniform(0, 100, (1000, 2))
    env = Environment.from_samples(positions, bin_size=10.0)
    times = np.linspace(0, 100, 1000)
    spike_times = [np.sort(rng.uniform(0, 100, n)) for n in (40, 50, 30)]
    rates_result = compute_spatial_rates(
        env,
        spike_times,
        times,
        positions,
        bandwidth=10.0,
        fill_value=0.0,
        dtype=np.float32,
    )
    assert rates_result.firing_rates.dtype == np.float32

    n_time_bins, n_neurons = 50, len(spike_times)
    spike_counts = rng.poisson(2, (n_time_bins, n_neurons)).astype(np.int64)

    dtype_from_object = _capture_model_dtype(env, spike_counts, rates_result, 0.025)
    dtype_from_array = _capture_model_dtype(
        env, spike_counts, np.asarray(rates_result.firing_rates), 0.025
    )

    assert dtype_from_object == np.dtype(np.float32)
    # Object path must match raw-array path exactly.
    assert dtype_from_object == dtype_from_array


def test_decode_position_result_object_preserves_float64() -> None:
    """A float64 rate-result object yields a float64 model (unchanged)."""
    env, spike_counts, rates_result, dt = _setup()
    assert rates_result.firing_rates.dtype == np.float64
    assert _capture_model_dtype(env, spike_counts, rates_result, dt) == np.dtype(
        np.float64
    )


def test_decode_position_integer_firing_rates_cast_to_float64() -> None:
    """An integer .firing_rates array still works, cast to float64."""
    env, spike_counts, _rates_result, dt = _setup()
    n_bins = env.n_bins
    int_rates = np.ones((3, n_bins), dtype=np.int64)
    bad = _BadRates(firing_rates=int_rates)
    assert _capture_model_dtype(env, spike_counts, bad, dt) == np.dtype(np.float64)
