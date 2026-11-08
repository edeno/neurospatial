"""Expected Skaggs information of an untuned cell, and the screens that subtract it."""

import warnings

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
    return [generate_poisson_spikes(rate, times, rng=seed + i) for i in range(n_units)]


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


@pytest.fixture(scope="module")
def ou_sessions():
    """Two 10-minute OU explorations of a 1 m box (5 cm bins, 30 Hz)."""
    from shapely.geometry import box as rectangle

    from neurospatial.simulation import simulate_trajectory_ou

    env = Environment.from_polygon(rectangle(0, 0, 100, 100), bin_size=5.0, units="cm")
    sessions = []
    for seed in (40, 41):
        positions, times = simulate_trajectory_ou(
            env,
            duration=600.0,
            speed_units="cm",
            dt=1 / 30,
            speed_mean=10.0,
            speed_std=5.2,
            rng=seed,
        )
        sessions.append((times, positions))
    return env, sessions


def _bias_against_untuned_information(env, sessions, n_spikes, n_units, **options):
    """Mean predicted bias, mean untuned information and its standard error."""
    information, bias = [], []
    for k, (times, positions) in enumerate(sessions):
        rates = compute_spatial_rates(
            env,
            _untuned(times, n_spikes, n_units, seed=10_000 * (k + 1) + n_spikes),
            times,
            positions,
            **options,
        )
        information.append(np.asarray(rates.spatial_information()))
        bias.append(np.asarray(rates.spatial_information_bias()))
    return np.concatenate(bias), np.concatenate(information)


@pytest.mark.parametrize("n_spikes", [100, 1000])
@pytest.mark.parametrize("bandwidth", [2.5, 5.0, 10.0])
@pytest.mark.parametrize("method", ["diffusion_kde", "gaussian_kde"])
def test_bias_is_the_mean_information_of_untuned_cells(
    ou_sessions, method, bandwidth, n_spikes
):
    """400 untuned Poisson cells on two trajectories, per estimator.

    The predicted bias averages to the untuned cells' mean information
    within 3 standard errors of that mean. At 100 spikes a 5 cm kernel
    leaves about 46 independent bins, half the spike count, and the
    second-order expansion behind the bias is off by up to 4% (measured),
    so 5% of the mean is allowed on top. A 2.5 cm kernel leaves more
    independent bins than 100 spikes, so there the bias is undetermined
    (NaN).
    """
    env, sessions = ou_sessions
    bias, information = _bias_against_untuned_information(
        env, sessions, n_spikes, 200, method=method, bandwidth=bandwidth
    )
    if bandwidth == 2.5 and n_spikes == 100:
        assert np.isnan(bias).all()
        return
    assert np.isfinite(bias).all()
    standard_error = information.std(ddof=1) / np.sqrt(information.size)
    expansion = 0.05 * information.mean() if n_spikes == 100 else 0.0
    assert abs(bias.mean() - information.mean()) <= 3 * standard_error + expansion


@pytest.mark.slow
@pytest.mark.parametrize("penalty", [1.0])
def test_penalized_glm_bias_is_the_mean_information_of_untuned_cells(
    ou_sessions, penalty
):
    """60 untuned cells of 1000 spikes, a GAM at a fixed penalty.

    REML picks a nearly flat fit for untuned cells (bias and information
    both near 1e-4 bits/spike), so a fixed penalty is what exercises the
    GAM's one-spike response.
    """
    env, sessions = ou_sessions
    bias, information = _bias_against_untuned_information(
        env, sessions, 1000, 30, method="glm", penalty=penalty
    )
    assert np.isfinite(bias).all()
    standard_error = information.std(ddof=1) / np.sqrt(information.size)
    assert abs(bias.mean() - information.mean()) <= 3 * standard_error


def test_screen_rejects_untuned_cells_that_pass_uncorrected(box):
    env, times, positions = box
    rates = compute_spatial_rates(
        env, _untuned(times, 60, 40, seed=200), times, positions, bandwidth=5.0
    )
    assert (rates.spatial_information() >= 0.3).mean() > 0.5
    verdict = rates.classify(min_info=0.3)
    assert not verdict.any()


def test_place_cell_passes_after_correction(box):
    env, times, positions = box
    field = 2.0 * np.exp(-np.sum((positions - 50.0) ** 2, axis=1) / (2 * 10.0**2))
    spikes = generate_poisson_spikes(field + 0.05, times, rng=7)
    rates = compute_spatial_rates(env, [spikes], times, positions, bandwidth=5.0)
    corrected = rates.spatial_information() - rates.spatial_information_bias()
    assert corrected[0] > 1.0
    assert rates.classify().tolist() == [True]


def test_fewer_spikes_than_independent_bins_is_refused(box):
    env, times, positions = box
    trains = [*_untuned(times, 15, 2, seed=300), np.array([])]
    rates = compute_spatial_rates(env, trains, times, positions, bandwidth=5.0)
    bias = rates.spatial_information_bias()
    assert np.isnan(bias).all()
    assert not rates.classify(min_info=0.0).any()


def test_too_few_spikes_is_undetermined_without_a_warning(box):
    """Units whose bias cannot be estimated read as undetermined, silently."""
    env, times, positions = box
    trains = [*_untuned(times, 15, 2, seed=300), np.array([])]
    rates = compute_spatial_rates(env, trains, times, positions, bandwidth=5.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        labels = rates.label_cell_types(boundary_source="analysis")
        table = rates.summary_table(boundary_source="analysis")
    # The silent unit has no spikes to be uncertain about: unclassified.
    assert labels.tolist() == ["undetermined", "undetermined", "unclassified"]
    assert table["cell_type"].tolist() == labels.tolist()
    assert table["passes_spatial_information"].isna().tolist() == [True, True, False]


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


@pytest.mark.parametrize("n_visited", [1, 3])
def test_unpenalized_glm_bias_counts_only_the_visited_modes(n_visited):
    """With fewer visited bins than basis modes, only visited modes count.

    The unvisited modes are not identified, but they do not change the rate
    at a visited bin, so K = (visited bins) - 1 instead of rank - 1; one
    visited bin carries no spatial information at all.
    """
    env = Environment.from_samples(
        np.arange(8.0)[:, None], bin_size=1, infer_active_bins=False, units="cm"
    )
    times = np.arange(1001) * 0.01
    positions = np.resize(env.bin_centers[:n_visited], (len(times), 1))
    spike_times = [np.arange(0.005, 10.0, 0.02)]
    with pytest.warns(UserWarning, match="not identifiable"):
        rates = compute_spatial_rates(
            env,
            spike_times,
            times,
            positions,
            method="glm",
            penalty=0.0,
            rank=8,
            min_speed=0,
        )
    occupancy = np.asarray(rates.occupancy)
    support = rates.spatial_support
    assert support.sum() == n_visited
    mean_rate = (
        rates.firing_rates[:, support] @ occupancy[support] / occupancy[support].sum()
    )
    expected = (n_visited - 1) / (2 * np.log(2) * mean_rate * occupancy.sum())
    np.testing.assert_allclose(
        rates.spatial_information_bias(), expected, rtol=1e-6, atol=1e-12
    )


def test_screens_share_one_probe_pass(box, monkeypatch):
    """classify, label_cell_types and summary_table reuse the probed bias."""
    import copy
    import pickle

    from neurospatial.encoding import _metrics

    env, times, positions = box
    rates = compute_spatial_rates(
        env, _untuned(times, 600, 3, seed=500), times, positions, bandwidth=5.0
    )
    calls = []
    probe = _metrics._untuned_rate_variance

    def counted(*args, **kwargs):
        calls.append(1)
        return probe(*args, **kwargs)

    monkeypatch.setattr(_metrics, "_untuned_rate_variance", counted)
    first = rates.spatial_information_bias()
    rates.classify()
    rates.label_cell_types(boundary_source="analysis")
    rates.summary_table(boundary_source="analysis")
    np.testing.assert_array_equal(rates.spatial_information_bias(), first)
    assert len(calls) == 1
    # Copies and pickles carry the cached value; it is never a writable view.
    for clone in (
        copy.copy(rates),
        copy.deepcopy(rates),
        pickle.loads(pickle.dumps(rates)),
    ):
        np.testing.assert_array_equal(clone.spatial_information_bias(), first)
    assert len(calls) == 1
    first[0] = -1.0  # the returned array belongs to the caller
    assert rates.spatial_information_bias()[0] > 0


def test_unit_results_share_the_population_probe(box, monkeypatch):
    """Iterating a population's units probes a linear estimator once."""
    from neurospatial.encoding import _metrics

    env, times, positions = box
    rates = compute_spatial_rates(
        env, _untuned(times, 600, 3, seed=500), times, positions, bandwidth=5.0
    )
    calls = []
    probe = _metrics._untuned_rate_variance

    def counted(*args, **kwargs):
        calls.append(1)
        return probe(*args, **kwargs)

    monkeypatch.setattr(_metrics, "_untuned_rate_variance", counted)
    unit_bias = [unit.spatial_information_bias() for unit in rates]
    np.testing.assert_allclose(unit_bias, rates.spatial_information_bias(), rtol=1e-12)
    assert len(calls) == 1
