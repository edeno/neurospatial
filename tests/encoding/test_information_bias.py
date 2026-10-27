"""Expected Skaggs information of an untuned cell, and the screens that subtract it."""

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.encoding import compute_spatial_rates
from neurospatial.simulation import generate_poisson_spikes

N_SAMPLES = 36000  # 20 minutes at 30 Hz


@pytest.fixture(scope="module")
def box():
    """A 1 m box sampled uniformly at 30 Hz, so every bin is well occupied."""
    rng = np.random.default_rng(0)
    positions = rng.uniform(0, 100, (N_SAMPLES, 2))
    env = Environment.from_samples(positions, bin_size=5.0, units="cm")
    return env, np.arange(N_SAMPLES) / 30.0, positions


def _untuned(times, n_spikes, n_units, seed):
    rate = np.full(len(times), n_spikes / times[-1])
    return [generate_poisson_spikes(rate, times, seed=seed + i) for i in range(n_units)]


def _spikes_in_map(rates):
    """Spikes the maps were estimated from: rate times dwell, summed."""
    occupancy = np.asarray(rates.occupancy)
    return np.nansum(np.asarray(rates.firing_rates) * occupancy, axis=1)


def test_unsmoothed_bias_is_the_miller_madow_form(box):
    env, times, positions = box
    rates = compute_spatial_rates(
        env,
        _untuned(times, 2000, 3, seed=10),
        times,
        positions,
        method="binned",
        bandwidth=0.0,
    )
    n_bins = int(rates.spatial_support.sum())
    expected = (n_bins - 1) / (2 * np.log(2) * _spikes_in_map(rates))
    np.testing.assert_allclose(rates.spatial_information_bias(), expected, rtol=1e-10)


@pytest.mark.parametrize("method", ["diffusion_kde", "gaussian_kde", "binned"])
def test_bias_is_the_mean_information_of_untuned_cells(box, method):
    """Smoothing shares spikes between bins; the bias follows the estimator."""
    env, times, positions = box
    rates = compute_spatial_rates(
        env,
        _untuned(times, 200, 40, seed=100),
        times,
        positions,
        method=method,
        bandwidth=5.0,
    )
    information = rates.spatial_information()
    bias = rates.spatial_information_bias()
    assert np.isfinite(bias).all()
    # 441 bins but far fewer independent ones: K = bias * 2 ln2 * N.
    effective_bins = np.median(bias * 2 * np.log(2) * _spikes_in_map(rates))
    assert 30 < effective_bins < 80
    assert np.mean(information) == pytest.approx(np.mean(bias), rel=0.15)


def test_screen_rejects_untuned_cells_that_pass_uncorrected(box):
    env, times, positions = box
    rates = compute_spatial_rates(
        env, _untuned(times, 60, 40, seed=200), times, positions, bandwidth=5.0
    )
    assert (rates.spatial_information() >= 0.3).mean() > 0.5
    assert not rates.classify(min_info=0.3).any()


def test_place_cell_passes_after_correction(box):
    env, times, positions = box
    field = 2.0 * np.exp(-np.sum((positions - 50.0) ** 2, axis=1) / (2 * 10.0**2))
    spikes = generate_poisson_spikes(field + 0.05, times, seed=7)
    rates = compute_spatial_rates(env, [spikes], times, positions, bandwidth=5.0)
    corrected = rates.spatial_information() - rates.spatial_information_bias()
    assert corrected[0] > 1.0
    assert rates.classify().tolist() == [True]


def test_fewer_spikes_than_independent_bins_is_refused(box):
    env, times, positions = box
    trains = [*_untuned(times, 15, 2, seed=300), np.array([])]
    rates = compute_spatial_rates(env, trains, times, positions, bandwidth=5.0)
    with pytest.warns(UserWarning, match=r"2 of 3 units .* independent bins"):
        bias = rates.spatial_information_bias()
    assert np.isnan(bias).all()
    with pytest.warns(UserWarning, match="independent bins"):
        assert not rates.classify(min_info=0.0).any()


def test_unpenalized_glm_bias_counts_the_basis_modes(box):
    """Without a penalty the GAM projects onto its basis: K = rank - 1."""
    env, times, positions = box
    rates = compute_spatial_rates(
        env,
        _untuned(times, 2000, 2, seed=400),
        times,
        positions,
        method="glm",
        penalty=0.0,
        rank=20,
    )
    occupancy = np.asarray(rates.occupancy)
    support = rates.spatial_support
    mean_rate = (
        rates.firing_rates[:, support] @ occupancy[support] / occupancy[support].sum()
    )
    expected = (rates.rank - 1) / (2 * np.log(2) * mean_rate * occupancy.sum())
    np.testing.assert_allclose(rates.spatial_information_bias(), expected, rtol=1e-6)
