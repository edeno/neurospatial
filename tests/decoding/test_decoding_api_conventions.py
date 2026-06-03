"""Signature/naming-convention tests for decoding public functions.

These tests pin keyword-only separators, argument renames, and argument
ordering on the decoding public surface. They use ``inspect.signature`` for
structural assertions and verify that removed keyword arguments raise
``TypeError``.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from neurospatial.decoding.assemblies import detect_assemblies
from neurospatial.decoding.metrics import confusion_matrix
from neurospatial.decoding.trajectory import fit_isotonic_trajectory


@pytest.fixture
def assembly_spike_counts() -> np.ndarray:
    """Population spike-count matrix, shape (n_neurons, n_time_bins).

    Two distinct co-activating assemblies (neurons 0-2 and 5-7) so that
    detection finds multiple components and ICA initialization (and hence
    the seed) affects the recovered patterns.
    """
    rng = np.random.default_rng(7)
    n_neurons, n_time_bins = 10, 600
    counts = rng.poisson(2.0, (n_neurons, n_time_bins)).astype(np.float64)
    counts[:3] += rng.poisson(4.0, n_time_bins).astype(np.float64)
    counts[5:8] += rng.poisson(4.0, n_time_bins).astype(np.float64)
    return counts


def test_detect_assemblies_rng_keyword(assembly_spike_counts):
    """`detect_assemblies` accepts `rng`; `random_state` is removed."""
    sig = inspect.signature(detect_assemblies)
    assert "rng" in sig.parameters
    assert "random_state" not in sig.parameters
    assert sig.parameters["rng"].kind is inspect.Parameter.KEYWORD_ONLY

    # Reproducible across calls with the same int seed.
    r1 = detect_assemblies(assembly_spike_counts, rng=0)
    r2 = detect_assemblies(assembly_spike_counts, rng=0)
    assert len(r1.patterns) == len(r2.patterns)
    for p1, p2 in zip(r1.patterns, r2.patterns, strict=True):
        np.testing.assert_array_equal(p1.member_indices, p2.member_indices)

    # A different seed gives a different result (patterns differ in sign/order
    # under ICA initialization).
    r3 = detect_assemblies(assembly_spike_counts, rng=1)
    differs = any(
        not np.array_equal(p1.weights, p3.weights)
        for p1, p3 in zip(r1.patterns, r3.patterns, strict=True)
    )
    assert differs

    # A Generator is also accepted.
    detect_assemblies(assembly_spike_counts, rng=np.random.default_rng(0))

    # The removed keyword raises TypeError.
    with pytest.raises(TypeError):
        detect_assemblies(assembly_spike_counts, random_state=0)


def test_fit_isotonic_trajectory_env_optional():
    """`fit_isotonic_trajectory` runs without `env`; `env` is keyword-only."""
    sig = inspect.signature(fit_isotonic_trajectory)
    params = list(sig.parameters)
    assert params[0] == "posterior"
    assert params[1] == "times"
    assert sig.parameters["env"].kind is inspect.Parameter.KEYWORD_ONLY
    assert sig.parameters["env"].default is None

    n_time_bins, n_bins = 20, 50
    posterior = np.zeros((n_time_bins, n_bins))
    for t in range(n_time_bins):
        posterior[t, t * 2] = 1.0
    times = np.linspace(0, 1, n_time_bins)

    result_no_env = fit_isotonic_trajectory(posterior, times)
    result_with_env = fit_isotonic_trajectory(posterior, times, env=None)

    np.testing.assert_array_equal(
        result_no_env.fitted_positions, result_with_env.fitted_positions
    )
    assert result_no_env.r_squared == pytest.approx(result_with_env.r_squared)

    # Cannot pass env as the first positional argument any more.
    with pytest.raises(TypeError):
        fit_isotonic_trajectory(None, posterior, times)


def test_confusion_matrix_method_renamed(small_2d_env):
    """`confusion_matrix` uses `method`; `summary_method` is removed."""
    sig = inspect.signature(confusion_matrix)
    assert "method" in sig.parameters
    assert "summary_method" not in sig.parameters
    assert sig.parameters["method"].kind is inspect.Parameter.KEYWORD_ONLY

    n_bins = small_2d_env.n_bins
    rng = np.random.default_rng(0)
    posterior = rng.dirichlet(np.ones(n_bins), size=10)
    actual_bins = rng.integers(0, n_bins, size=10).astype(np.int64)

    cm = confusion_matrix(small_2d_env, posterior, actual_bins, method="map")
    assert cm.shape == (n_bins, n_bins)

    # The removed keyword raises TypeError.
    with pytest.raises(TypeError):
        confusion_matrix(small_2d_env, posterior, actual_bins, summary_method="map")
