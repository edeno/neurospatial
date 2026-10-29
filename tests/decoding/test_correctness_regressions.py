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
def test_partial_missing_rate_likelihood_matches_poisson(tiny_env, decode, missing):
    """Masked units drop their entire Poisson term, including the factorial."""
    models = np.full((2, tiny_env.n_bins), missing)
    models[:, :2] = [[missing, 4.0], [1.0, 1.0]]
    counts = np.array([[4, 0], [0, 2], [3, 1]])
    observed = np.isfinite(models)
    # Evaluate complete Poisson probabilities independently, with each missing
    # observation contributing the multiplicative identity. All-missing bins
    # remain excluded rather than being assigned neutral likelihood.
    probabilities = poisson.pmf(
        counts[:, :, None], np.where(observed, models, 0.0)[None]
    )
    likelihood = np.prod(np.where(observed[None], probabilities, 1.0), axis=1)
    likelihood[:, ~observed.any(axis=0)] = 0.0
    expected = likelihood / likelihood.sum(axis=1, keepdims=True)

    with pytest.warns(UserWarning, match="non-finite"):
        result = decode(
            tiny_env,
            counts,
            models,
            dt=1.0,
            validate=not np.isinf(missing),
            time_chunk=1,
        )
    np.testing.assert_array_equal(result.map_bin, expected.argmax(axis=1))
    # Four spikes at the finite 4 Hz model previously made bin 1 win, because
    # its omitted 4! factor was only present at that bin.
    assert result.map_bin[0] == 0
    if isinstance(result, DecodingResult):
        np.testing.assert_allclose(result.posterior, expected)
    else:
        np.testing.assert_allclose(result.peak_probability, expected.max(axis=1))


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
