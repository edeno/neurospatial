"""Shared fixtures for decoding tests.

Provides deterministic, small synthesized fixtures for the assembly
reactivation tests (template/match spike counts plus a clear assembly
pattern) and for the explained-variance partial-correlation tests
(a triplet of correlation vectors with known, distinct pairwise
correlations).
"""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial.decoding.assemblies import AssemblyPattern


@pytest.fixture
def assembly_pattern() -> AssemblyPattern:
    """A clear single-assembly pattern over 6 neurons.

    Neurons 0-2 are the assembly members (equal positive weight,
    unit-norm), neurons 3-5 have zero weight.
    """
    n_neurons = 6
    weights = np.zeros(n_neurons, dtype=np.float64)
    weights[:3] = 1.0 / np.sqrt(3.0)
    member_indices = np.array([0, 1, 2], dtype=np.int64)
    return AssemblyPattern(
        weights=weights,
        member_indices=member_indices,
        explained_variance_ratio=0.5,
    )


@pytest.fixture
def template_counts() -> np.ndarray:
    """Deterministic template-period spike counts, shape (6 neurons, 300 bins).

    The assembly members (neurons 0-2) carry structured co-activation so
    the projection onto the pattern has non-trivial magnitude.
    """
    rng = np.random.default_rng(0)
    n_neurons, n_bins = 6, 300
    counts = rng.poisson(2.0, (n_neurons, n_bins)).astype(np.float64)
    # Inject co-activation in the member neurons.
    drive = rng.poisson(3.0, n_bins).astype(np.float64)
    counts[:3] += drive
    return counts


@pytest.fixture
def corr_triplet() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Three correlation vectors (template, match, control).

    Constructed so that the pairwise correlations are known and distinct:

    - ``match`` is driven mostly by ``template`` and partly by ``control``,
    - ``control`` itself shares variance with ``template``.

    This makes the template->match relationship stronger than the
    control->match relationship (so EV > REV under role-swapping), while
    partialling out the control — which shares variance with both — strictly
    lowers EV below the raw template->match squared correlation.
    """
    rng = np.random.default_rng(0)
    n_pairs = 80
    template = rng.standard_normal(n_pairs)
    control = 0.6 * template + 0.8 * rng.standard_normal(n_pairs)
    match = 1.0 * template + 0.4 * control + 0.4 * rng.standard_normal(n_pairs)
    return template, match, control
