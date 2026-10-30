"""decode_position accepts a SpatialRatesResult directly.

Passing the rate result decodes its ``firing_rates`` with its
``spatial_support``; other result objects are refused.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from environment_helpers import publish_binning, publish_environment

from neurospatial import Environment, IncompatibleEnvironmentError
from neurospatial.decoding.posterior import decode_position, decode_position_summary
from neurospatial.encoding.spatial import SpatialRatesResult, compute_spatial_rates
from neurospatial.environment._fields import face_geometry


@pytest.fixture
def small_rates() -> SpatialRatesResult:
    env = Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool), grid_edges=(np.arange(3.0) * 10,) * 2
    )
    env = env.with_metadata(units="cm")
    env = env.with_metadata(frame="world")
    return SpatialRatesResult(
        firing_rates=np.array([[1.0, 2.0, 3.0, 4.0]]),
        occupancy=np.ones(4),
        env=env,
        method="binned",
        bandwidth=5.0,
    )


@pytest.mark.parametrize("decode", [decode_position, decode_position_summary])
@pytest.mark.parametrize("validate", [True, False])
@pytest.mark.parametrize(
    "mismatch", ["coordinates", "units", "frame", "topology", "measures"]
)
def test_rate_result_rejects_different_environment(
    small_rates, decode, validate, mismatch, monkeypatch
) -> None:
    """Equal bin counts cannot establish spatial or coordinate compatibility."""
    env = small_rates.env
    if mismatch == "coordinates":
        env = Environment.from_grid_mask(
            np.ones((2, 2), dtype=bool),
            grid_edges=(1000 + np.arange(3.0) * 10,) * 2,
        )
        env = env.with_metadata(units="cm", frame="world")
    elif mismatch == "topology":
        env = Environment.from_grid_mask(
            np.ones((2, 2), dtype=bool),
            grid_edges=(np.arange(3.0) * 10,) * 2,
            connect_diagonal_neighbors=False,
        )
        env = env.with_metadata(units="cm", frame="world")
    elif mismatch == "measures":
        from dataclasses import replace

        from neurospatial.geometry._binning import KnownFaces
        from neurospatial.geometry._locators import NearestBinLocator

        bins = replace(env._binning.bins, sizes=env.bin_sizes * 2)
        faces = face_geometry(env._cache, env._binning)._replace(volumes=bins.sizes)
        binning = replace(
            env._binning,
            bins=bins,
            locator=NearestBinLocator(bins.centers),
            faces=KnownFaces(faces),
            grid=None,
        )
        layout = publish_binning(
            binning,
            dimension_ranges=env.dimension_ranges,
            layout_type="UnequalCells",
            build_params={},
        )
        env = publish_environment(
            layout=layout,
            layout_type_used="UnequalCells",
            arena=env.arena,
            units="cm",
            frame="world",
        )
    else:
        env = env.with_metadata(**{mismatch: "m" if mismatch == "units" else "camera"})

    with pytest.raises(
        IncompatibleEnvironmentError, match=r"encoding result.*environment"
    ):
        decode(env, np.array([[0], [1]]), small_rates, dt=0.1, validate=validate)


@pytest.mark.parametrize("decode", [decode_position, decode_position_summary])
def test_rate_result_accepts_equivalent_copied_environment(small_rates, decode) -> None:
    counts = np.array([[0], [1]])
    reference = decode(small_rates.env, counts, small_rates, dt=0.1)
    copied = decode(small_rates.env.with_metadata(), counts, small_rates, dt=0.1)
    np.testing.assert_array_equal(copied.map_position, reference.map_position)
    np.testing.assert_array_equal(copied.posterior_entropy, reference.posterior_entropy)


def _setup() -> tuple[Environment, np.ndarray, object, float]:
    rng = np.random.default_rng(0)
    positions = rng.uniform(0, 100, (1000, 2))
    env = Environment.from_samples(positions, bin_size=10.0)
    times = np.linspace(0, 100, 1000)
    spike_times = [np.sort(rng.uniform(0, 100, n)) for n in (40, 50, 30)]
    rates_result = compute_spatial_rates(
        env, spike_times, times, positions, bandwidth=10.0
    )

    n_time_bins, n_neurons = 50, len(spike_times)
    spike_counts = rng.poisson(2, (n_time_bins, n_neurons)).astype(np.int64)
    return env, spike_counts, rates_result, 0.025


def test_decode_position_accepts_result_object() -> None:
    env, spike_counts, rates_result, dt = _setup()

    from_result = decode_position(env, spike_counts, rates_result, dt=dt)
    from_array = decode_position(
        env, spike_counts, np.asarray(rates_result.firing_rates), dt=dt
    )

    np.testing.assert_allclose(
        from_result.posterior, from_array.posterior, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_array_equal(from_result.map_bin, from_array.map_bin)


@dataclass
class _OtherRates:
    firing_rates: Any


@pytest.mark.parametrize("decode", [decode_position, decode_position_summary])
@pytest.mark.parametrize(
    "firing_rates", [None, {"north": np.zeros(5)}, np.zeros(5), np.ones((3, 1))]
)
def test_other_rate_objects_raise_type_error(decode, firing_rates) -> None:
    """Only SpatialRatesResult carries the support and env a decode needs."""
    env, spike_counts, _rates_result, dt = _setup()
    with pytest.raises(TypeError, match="SpatialRatesResult") as err:
        decode(env, spike_counts, _OtherRates(firing_rates=firing_rates), dt=dt)
    assert "Fix:" in str(err.value)


def _capture_model_dtype(env, spike_counts, encoding_models, dt):
    """Decode while spying on the dtype of the encoding models that reach
    the likelihood."""
    from neurospatial.decoding.likelihood import _PoissonRates

    captured: dict[str, Any] = {}
    real = _PoissonRates.from_rates.__func__

    def spy(cls, models, dt_, **kwargs):
        captured["dtype"] = np.asarray(models).dtype
        return real(cls, models, dt_, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_PoissonRates, "from_rates", classmethod(spy))
        decode_position(env, spike_counts, encoding_models, dt=dt)
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


def test_decode_position_integer_rate_array_matches_float() -> None:
    """An integer rate array decodes exactly like the same rates as floats."""
    env, spike_counts, _rates_result, dt = _setup()
    int_rates = np.arange(3 * env.n_bins).reshape(3, env.n_bins) % 7 + 1
    as_int = decode_position(env, spike_counts, int_rates, dt=dt)
    as_float = decode_position(env, spike_counts, int_rates.astype(float), dt=dt)
    np.testing.assert_array_equal(as_int.posterior, as_float.posterior)
