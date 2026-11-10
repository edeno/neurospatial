"""Decoding error against the Cramér-Rao bound of a Poisson population.

On a 1-D track tiled by Gaussian place fields ``f_i(x)``, the Fisher
information about position in one time bin of length ``dt`` is

    J(x) = dt * sum_i f_i'(x)**2 / f_i(x).

An unbiased estimator has mean squared error at least ``1 / J(x)``, and the
posterior mean reaches it as the spike count per bin grows. True positions
are drawn uniformly over the middle of the track, far from its ends, so the
uniform prior adds no information there. The expected bound is the mean of
``1 / J(x)`` over those positions.
"""

from __future__ import annotations

import numpy as np

from neurospatial import Environment
from neurospatial.decoding import decode_position, mean_position

TRACK_LENGTH = 200.0
FIELD_CENTERS = np.arange(-30.0, TRACK_LENGTH + 31.0, 6.0)  # past both ends
FIELD_WIDTH = 8.0
BASELINE = 0.5
N_DRAWS = 20_000
DT = 0.5


def _rates(x, peak):
    """Rates in Hz and their derivatives, shape (n_cells, len(x))."""
    offset = x[None] - FIELD_CENTERS[:, None]
    bump = peak * np.exp(-0.5 * (offset / FIELD_WIDTH) ** 2)
    return BASELINE + bump, -offset / FIELD_WIDTH**2 * bump


def _error_and_bound(peak, seed=0):
    """Squared errors of the posterior mean and the bound 1/J at each draw."""
    env = Environment.linear_track(
        endpoints=[[0.0, 0.0], [TRACK_LENGTH, 0.0]], bin_size=1.0, units="cm"
    )
    rates, _ = _rates(env.bin_centers[:, 0], peak)
    true_x = np.random.default_rng(seed).uniform(50.0, 150.0, N_DRAWS)
    true_rates, slopes = _rates(true_x, peak)
    counts = np.random.default_rng(seed + 1).poisson(true_rates.T * DT)
    posterior = decode_position(env, counts, rates, dt=DT).posterior
    error = mean_position(env, posterior)[:, 0] - true_x
    fisher = DT * np.sum(slopes**2 / true_rates, axis=0)
    return error, 1.0 / fisher


def test_posterior_mean_reaches_the_bound_with_many_spikes():
    """About 110 spikes per bin: MSE / bound within Monte Carlo error of 1.

    The ratio cannot fall below 1 by more than Monte Carlo error. Above 1
    the allowance is 5% for the second-order excess, which measured 6.7% at
    36 spikes per bin and 3.4% at 61 (here the ratio is 0.997 +- 0.010).
    """
    error, bound = _error_and_bound(60.0)
    squared = error**2
    ratio = squared.mean() / bound.mean()
    se = squared.std() / np.sqrt(N_DRAWS) / bound.mean()
    assert 1.0 - 3 * se <= ratio <= 1.05, (ratio, se)
    assert abs(error.mean()) < 3 * error.std() / np.sqrt(N_DRAWS)


def test_excess_over_the_bound_shrinks_with_spike_count():
    """The bound is asymptotic: at 24 spikes per bin the MSE is 68% above it."""
    error_few, bound_few = _error_and_bound(7.5)
    error_many, bound_many = _error_and_bound(60.0)
    ratio_few = np.mean(error_few**2) / bound_few.mean()
    ratio_many = np.mean(error_many**2) / bound_many.mean()
    assert ratio_few > ratio_many + 0.1, (ratio_few, ratio_many)
