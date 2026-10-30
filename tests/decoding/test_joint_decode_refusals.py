"""decode_jointly refuses the inputs its docstring says it refuses."""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.decoding import decode_jointly

DT = 0.1


@pytest.fixture(scope="module")
def two_directions():
    """One 100 cm track with an outbound and an inbound set of maps for 8 cells."""
    env = Environment.linear_track(
        endpoints=[[0.0, 0.0], [100.0, 0.0]], bin_size=10.0, units="cm"
    )
    x = env.bin_centers[:, 0]
    outbound = 1.0 + 20.0 * np.exp(
        -0.5 * ((x - np.linspace(5.0, 95.0, 8)[:, None]) / 8.0) ** 2
    )
    counts = np.random.default_rng(0).poisson(1.0, (3, 8))
    return env, {"outbound": outbound, "inbound": outbound[::-1].copy()}, counts


def _flat(env):
    return np.ones(env.n_bins)


@pytest.mark.parametrize(
    ("make_prior", "match"),
    [
        (lambda env: np.zeros(env.n_bins), "zero total mass"),
        (
            lambda env: np.ones((3, env.n_bins)) * np.array([[1.0], [0.0], [1.0]]),
            "zero total mass in at least one time bin",
        ),
        (lambda env: -np.ones(env.n_bins), "negative or non-finite"),
        (lambda env: np.full(env.n_bins, np.nan), "negative or non-finite"),
    ],
    ids=["zero", "zero_row", "negative", "nan"],
)
def test_refuses_a_prior_without_valid_mass(two_directions, make_prior, match):
    env, models, counts = two_directions
    prior = {"outbound": make_prior(env), "inbound": _flat(env)}
    with pytest.raises(ValueError, match=match):
        decode_jointly(env, counts, models, dt=DT, prior=prior)


def test_refuses_a_prior_only_on_excluded_bins(two_directions):
    """Inbound has no rates in bins 0-4; its prior lies only there."""
    env, models, counts = two_directions
    inbound = models["inbound"].copy()
    inbound[:, :5] = np.nan
    prior = {"outbound": _flat(env), "inbound": np.r_[np.ones(5), np.zeros(5)]}
    with pytest.raises(ValueError, match="no mass in the bins where every"):
        decode_jointly(
            env,
            counts,
            {"outbound": models["outbound"], "inbound": inbound},
            dt=DT,
            prior=prior,
        )


def test_refuses_a_template_with_another_number_of_neurons(two_directions):
    env, models, counts = two_directions
    with pytest.raises(ValueError, match="Neuron-count mismatch"):
        decode_jointly(
            env,
            counts,
            {"outbound": models["outbound"], "inbound": models["inbound"][:7]},
            dt=DT,
        )


def test_refuses_times_of_another_length(two_directions):
    env, models, counts = two_directions
    with pytest.raises(ValueError, match=r"times has shape \(5,\), expected \(3,\)"):
        decode_jointly(env, counts, models, dt=DT, times=np.arange(5.0))
