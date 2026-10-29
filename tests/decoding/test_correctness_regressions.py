"""Scientific identity, missing-rate likelihood and decode ownership regressions."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import poisson

from neurospatial import Environment
from neurospatial._exceptions import IncompatibleEnvironmentError
from neurospatial.decoding import (
    BayesianDecoder,
    DecodingResult,
    DecodingSummary,
    decode_position,
    decode_position_summary,
)
from neurospatial.encoding import SpatialRatesResult, compute_spatial_rates


@pytest.fixture
def tiny_env():
    positions = np.array([[0.0, 0.0], [0.0, 10.0], [10.0, 0.0], [10.0, 10.0]])
    return Environment.from_samples(positions, bin_size=5.0)


@pytest.mark.parametrize("decode", [decode_position, decode_position_summary])
@pytest.mark.parametrize("validate", [True, False])
@pytest.mark.parametrize("mismatch", ["coordinates", "units", "frame"])
def test_rate_result_environment_mismatch_raises(tiny_env, decode, validate, mismatch):
    """A matching bin count cannot establish coordinate or physical identity."""
    env = tiny_env
    if mismatch == "coordinates":
        other = Environment.from_samples(env.bin_centers + 100, bin_size=5.0)
    else:
        other = env.with_metadata(
            **{mismatch: "cm" if mismatch == "units" else "camera"}
        )
    assert other.n_bins == env.n_bins
    rates = SpatialRatesResult(
        np.ones((1, env.n_bins)), np.ones(env.n_bins), env, "binned", 5.0
    )
    with pytest.raises(IncompatibleEnvironmentError, match="environment") as caught:
        decode(other, np.array([[1]]), rates, dt=0.1, validate=validate)
    assert "Fix:" in str(caught.value)


@pytest.mark.parametrize("decode", [decode_position, decode_position_summary])
def test_rate_result_equivalent_environment_copy_decodes(tiny_env, decode):
    """Copied environments preserve the meaning and order of the rate bins."""
    env = tiny_env
    rates = SpatialRatesResult(
        np.arange(1.0, env.n_bins + 1)[None], np.ones(env.n_bins), env, "binned", 5.0
    )
    counts = np.array([[1], [0]])
    actual = decode(env.with_metadata(), counts, rates, dt=0.1)
    expected = decode(env, counts, rates, dt=0.1)
    np.testing.assert_array_equal(actual.map_position, expected.map_position)
    np.testing.assert_allclose(actual.posterior_entropy, expected.posterior_entropy)


@pytest.mark.parametrize("decode", [decode_position, decode_position_summary])
@pytest.mark.parametrize("missing", [np.nan, np.inf], ids=["nan", "inf"])
def test_partial_missing_rate_bins_are_excluded(decode, missing):
    """Bins missing any neuron's rate get zero mass; the rest follow Poisson.

    Ten neurons with flat rates carry no position information, so the
    posterior must be uniform over the fully observed bins whatever the
    counts. Dropping only the masked neurons' terms instead made the bin
    where nine of ten neurons were masked win with posterior 0.9.
    """
    env = Environment.from_samples(np.linspace(0, 100, 500)[:, None], bin_size=10.0)
    n_bins = env.n_bins
    rng = np.random.default_rng(3)
    models = rng.uniform(2.0, 20.0, (10, n_bins))
    models[:, 2:] = 5.0  # flat on the fully observed bins 2..n
    models[:9, 0] = missing  # nine of ten neurons masked at bin 0
    models[4, 1] = missing  # one neuron masked at bin 1
    counts = np.array([np.zeros(10), np.full(10, 2), np.arange(10) % 3], dtype=int)

    observed = np.isfinite(models).all(axis=0)
    log_pmf = poisson.logpmf(counts[:, :, None], np.where(observed, models, 1.0)[None])
    log_likelihood = np.where(observed, log_pmf.sum(axis=1), -np.inf)
    expected = np.exp(log_likelihood - log_likelihood.max(axis=1, keepdims=True))
    expected /= expected.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(expected[:, observed], 1.0 / observed.sum())

    with pytest.warns(UserWarning, match="2 of 11 bins"):
        result = decode(
            env,
            counts,
            models,
            dt=1.0,
            validate=not np.isinf(missing),
            time_chunk=1,
        )
    if isinstance(result, DecodingResult):
        np.testing.assert_allclose(result.posterior, expected, rtol=1e-12)
    else:
        np.testing.assert_allclose(result.peak_probability, expected.max(axis=1))
        assert np.all(result.map_bin >= 2)


@pytest.mark.parametrize("training", ["short", "gapped", "epochs"])
def test_fit_short_training_runs_predicts_later_recording(tiny_env, training):
    """Prediction dt does not constrain valid observed training intervals."""
    if training == "gapped":
        times = np.array([0.0, 0.1, 0.2, 1.0, 1.1, 1.2])
    elif training == "epochs":
        times = np.arange(20) / 10
    else:
        times = np.arange(4) / 10
    epochs = [(0.0, 0.3), (1.0, 1.3)] if training == "epochs" else None
    positions = tiny_env.bin_centers[np.arange(len(times)) % tiny_env.n_bins]
    spikes = [times[:-1] + 0.01]
    decoder = BayesianDecoder(
        tiny_env, dt=1.0, method="binned", warn_on_drop=False
    ).fit(spikes, times, positions, epochs=epochs)
    rates = compute_spatial_rates(
        tiny_env,
        spikes,
        times,
        positions,
        method="binned",
        epochs=epochs,
        warn_on_drop=False,
    )
    np.testing.assert_array_equal(decoder.encoding_models, rates.firing_rates)
    np.testing.assert_array_equal(decoder.spatial_support, rates.spatial_support)

    prediction_times = 10 + np.arange(11) / 5
    result = decoder.predict([np.array([10.1, 11.1])], prediction_times)
    assert result.n_time_bins == 2
    np.testing.assert_allclose(result.posterior.sum(axis=1), 1.0)
    np.testing.assert_array_equal(result.posterior[:, ~decoder.spatial_support], 0.0)


@pytest.mark.parametrize("summary", [False, True])
def test_xarray_exports_are_independent_writable_arrays(tiny_env, summary):
    """Editing estimates, coordinates or window metadata cannot alter results."""
    pytest.importorskip("xarray")
    map_bin = np.array([0, 1])
    times = np.array([0.0, 0.25])
    window = np.array([[0.0, 1.0]])
    if summary:
        result = DecodingSummary(
            times=times,
            mean_position=tiny_env.bin_centers[map_bin].copy(),
            posterior_entropy=np.zeros(2),
            peak_probability=np.ones(2),
            env=tiny_env,
            map_bin=map_bin,
            spike_window=window,
        )
        original_arrays = [
            result.map_bin,
            result.map_position,
            result.mean_position,
            result.posterior_entropy,
            result.peak_probability,
        ]
    else:
        posterior = np.zeros((2, tiny_env.n_bins))
        posterior[np.arange(2), map_bin] = 1.0
        result = DecodingResult(posterior, tiny_env, times, spike_window=window)
        original_arrays = [result.posterior, result.map_bin]
    original_arrays.extend([result.times, result.spike_window, tiny_env.bin_centers])
    snapshots = [array.copy() for array in original_arrays]

    dataset = result.to_xarray()
    for name in [*dataset.data_vars, *dataset.coords]:
        exported = dataset[name].values
        assert exported.flags.writeable, name
        assert all(
            not np.shares_memory(exported, array) for array in original_arrays
        ), name
        exported.flat[0] += 1
    exported_window = dataset.attrs["spike_window"]
    assert exported_window.flags.writeable
    assert not np.shares_memory(exported_window, result.spike_window)
    exported_window[0] += 1

    for original, snapshot in zip(original_arrays, snapshots, strict=True):
        np.testing.assert_array_equal(original, snapshot)
