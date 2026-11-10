"""Null calibration of the phase-precession shuffle p-value.

Spikes with phases independent of position make the permutation null exact:
``p = (k + 1) / (n_shuffles + 1)`` is uniform on its grid, so with 199
shuffles ``p < 0.05`` occurs at rate 0.045, not 0.05.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from calibration_helpers import (
    assert_pvalues_uniform,
    assert_rejections_at_rate,
    monte_carlo_rejection_rate,
)

from neurospatial.encoding.phase_precession import compute_phase_precession

N_CELLS = 1500
N_SHUFFLES = 199


def _null_results(n_spikes: int, seed: int):
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(N_CELLS):
        positions = rng.uniform(0.0, 40.0, n_spikes)
        phases = rng.uniform(0.0, 2 * np.pi, n_spikes)
        with warnings.catch_warnings():
            # Untuned spikes often fit at a slope bound; the warning says so.
            warnings.filterwarnings(
                "ignore", message="The best-fitting phase-position slope lies"
            )
            results.append(
                compute_phase_precession(
                    positions,
                    phases,
                    position_range=(0.0, 40.0),
                    n_shuffles=N_SHUFFLES,
                    rng=rng,
                )
            )
    return results


@pytest.mark.slow
@pytest.mark.parametrize("n_spikes", [10, 100])
def test_pvalue_is_uniform_without_precession(n_spikes):
    results = _null_results(n_spikes, seed=n_spikes)
    p_values = np.array([r.p_value for r in results])
    assert_rejections_at_rate(
        p_values < 0.05, monte_carlo_rejection_rate(N_SHUFFLES, 0.05)
    )
    assert_pvalues_uniform(p_values)
    # has_phase_precession also needs a negative slope: about half the rate.
    precessing = (p_values < 0.05) & (np.array([r.slope for r in results]) < 0)
    assert_rejections_at_rate(
        precessing, monte_carlo_rejection_rate(N_SHUFFLES, 0.05) / 2
    )
