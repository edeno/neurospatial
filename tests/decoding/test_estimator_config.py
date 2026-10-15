"""Fitted encoding configuration stays attached to the models it describes."""

from __future__ import annotations

import copy
import pickle
from dataclasses import replace

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.decoding import BayesianDecoder
from neurospatial.encoding import compute_spatial_rates


@pytest.fixture(scope="module")
def recording():
    rng = np.random.default_rng(17)
    times = np.arange(100) / 10
    positions = rng.uniform(0, 10, (100, 2))
    env = Environment.from_samples(positions, bin_size=5)
    return env, [times[::5], times[::7]], times, positions


@pytest.fixture(scope="module")
def fitted(recording):
    env, spikes, times, positions = recording
    return {
        method: BayesianDecoder(env, method=method, **options).fit(
            spikes, times, positions
        )
        for method, options in [
            ("binned", {"bandwidth": 1.0, "min_occupancy": 0.0}),
            ("glm", {"penalty": 1.0, "rank": 3}),
        ]
    }


@pytest.mark.parametrize(
    "method, name, value",
    [
        ("binned", "method", "gaussian_kde"),
        ("binned", "bandwidth", 2.0),
        ("binned", "min_occupancy", 0.1),
        ("glm", "penalty", 2.0),
        ("glm", "rank", 4),
    ],
)
def test_replace_rejects_changed_fitted_encoding(fitted, method, name, value):
    with pytest.raises(ValueError, match=rf"{name}.*fitted"):
        replace(fitted[method], **{name: value})


@pytest.mark.parametrize("method", ["binned", "glm"])
def test_replace_accepts_prediction_settings(fitted, method):
    fitted = fitted[method]
    updated = replace(
        fitted, dt=0.1, dtype=np.float32, max_gap=None, warn_on_drop=False
    )
    assert updated.encoding_models is fitted.encoding_models
    assert updated.analysis_settings == fitted.analysis_settings
    assert replace(fitted).encoding_models is fitted.encoding_models


@pytest.mark.parametrize("how", ["pickle", "deepcopy", "legacy_pickle"])
@pytest.mark.parametrize("method", ["binned", "glm"])
def test_restored_decoder_keeps_encoding_guard(fitted, how, method):
    fitted = fitted[method]
    if how == "deepcopy":
        restored = copy.deepcopy(fitted)
    else:
        source = copy.deepcopy(fitted)
        if how == "legacy_pickle":
            vars(source._fit).pop("encoding_config", None)
        restored = pickle.loads(pickle.dumps(source))
    assert replace(restored, dt=0.1).encoding_models is restored.encoding_models
    name = "bandwidth" if restored.method == "binned" else "penalty"
    with pytest.raises(ValueError, match=rf"{name}.*fitted"):
        replace(restored, **{name: 2.0})


@pytest.mark.parametrize("method", ["binned", "glm"])
def test_from_rates_uses_recorded_encoding_configuration(recording, method):
    env, spikes, times, positions = recording
    options = (
        {"bandwidth": 1.0, "min_occupancy": 0.1}
        if method == "binned"
        else {"penalty": 1.0, "rank": 3}
    )
    rates = compute_spatial_rates(
        env, spikes, times, positions, method=method, **options
    )
    decoder = BayesianDecoder.from_rates(rates)
    assert decoder.method == method
    for name, value in options.items():
        assert getattr(decoder, name) == value
    name = "bandwidth" if method == "binned" else "penalty"
    with pytest.raises(ValueError, match=rf"{name}.*fitted"):
        replace(decoder, **{name: 2.0})


def test_unfitted_encoding_remains_replaceable(recording):
    decoder = BayesianDecoder(recording[0], bandwidth=1.0)
    assert replace(decoder, bandwidth=2.0).bandwidth == 2.0


def test_fit_keeps_requested_defaults_when_encoder_resolves_them(recording):
    env, spikes, times, positions = recording
    decoder = BayesianDecoder(env).fit(spikes, times, positions)
    assert decoder.bandwidth is None
    assert decoder.analysis_settings["bandwidth"] > 0
    assert replace(decoder, dt=0.1).encoding_models is decoder.encoding_models
    with pytest.raises(ValueError, match=r"bandwidth.*fitted"):
        replace(decoder, bandwidth=decoder.analysis_settings["bandwidth"])


@pytest.mark.parametrize("pooled", [True, False])
def test_from_rates_keeps_automatic_penalty_choice(recording, pooled):
    env, spikes, times, positions = recording
    rates = compute_spatial_rates(
        env, spikes, times, positions, method="glm", rank=3, pooled=pooled
    )
    decoder = BayesianDecoder.from_rates(rates)
    assert decoder.method == "glm"
    assert decoder.penalty is None
    np.testing.assert_array_equal(decoder.encoding_models, rates.firing_rates)
    assert replace(decoder, dt=0.1).encoding_models is decoder.encoding_models


def test_hand_built_models_keep_configuration_guard(recording):
    env = recording[0]
    decoder = BayesianDecoder.from_models(env, np.ones((2, env.n_bins)))
    with pytest.raises(ValueError, match=r"bandwidth.*fitted"):
        replace(decoder, bandwidth=2.0)
