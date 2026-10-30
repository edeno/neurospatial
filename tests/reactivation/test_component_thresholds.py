"""Eigenvalue thresholds of detect_assemblies other than Marchenko-Pastur.

Lopes-dos-Santos, Ribeiro & Tort (2013) define surrogate thresholds as a
percentile of the largest eigenvalue of matrices whose neurons are shuffled
in time (bin shuffling) or rotated (circular shift); Peyrache et al. (2010)
place the finite-sample fluctuation of the largest eigenvalue at
N^(-2/3) above the Marchenko-Pastur edge.
"""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial.reactivation import detect_assemblies, marchenko_pastur_threshold


def _independent(seed: int, n_bins: int = 2000, n_neurons: int = 20) -> np.ndarray:
    return np.random.default_rng(seed).poisson(1.0, (n_bins, n_neurons)).astype(float)


def _slow_independent(seed: int, n_bins: int = 2000, n_neurons: int = 20) -> np.ndarray:
    """Independent neurons whose rates drift slowly, each on its own."""
    rng = np.random.default_rng(seed)
    drift = np.empty((n_bins, n_neurons))
    drift[0] = rng.standard_normal(n_neurons)
    for t in range(1, n_bins):
        drift[t] = 0.99 * drift[t - 1] + np.sqrt(1 - 0.99**2) * rng.standard_normal(
            n_neurons
        )
    return rng.poisson(np.exp(0.8 * drift)).astype(float)


def test_tracy_widom_adds_n_to_the_minus_two_thirds() -> None:
    counts = _independent(0)
    counts[:, 5] = 2.0  # constant: excluded, so n is the 19 varying neurons
    with pytest.warns(UserWarning, match="same count"):
        result = detect_assemblies(
            counts, method="pca", n_components=1, component_threshold="tracy_widom"
        )
    expected = marchenko_pastur_threshold(19, 2000) + 19 ** (-2 / 3)
    assert result.threshold == pytest.approx(expected, rel=1e-12)
    assert result.component_threshold == "tracy_widom"
    assert result.surrogate_max_eigenvalues.shape == (0,)
    assert result.n_significant == int(np.sum(result.eigenvalues > expected))


@pytest.mark.parametrize("kind", ["bin_shuffle", "circular_shift"])
def test_surrogate_threshold_is_the_percentile_of_the_maxima(kind: str) -> None:
    counts = _independent(1)
    result = detect_assemblies(
        counts,
        method="pca",
        n_components=1,
        component_threshold=kind,
        percentile=90.0,
        n_shuffles=50,
        rng=3,
    )
    maxima = result.surrogate_max_eigenvalues
    assert maxima.shape == (50,)
    assert result.threshold == pytest.approx(np.percentile(maxima, 90.0))
    # A surrogate's largest eigenvalue is at least the mean eigenvalue, 1.
    assert np.all(maxima > 1.0)
    again = detect_assemblies(
        counts,
        method="pca",
        n_components=1,
        component_threshold=kind,
        percentile=90.0,
        n_shuffles=50,
        rng=3,
    )
    np.testing.assert_array_equal(again.surrogate_max_eigenvalues, maxima)


def test_bin_shuffle_maximum_matches_a_hand_shuffled_matrix() -> None:
    counts = _independent(2, n_bins=300, n_neurons=6)
    result = detect_assemblies(
        counts,
        method="pca",
        n_components=1,
        component_threshold="bin_shuffle",
        percentile=50.0,
        n_shuffles=1,
        rng=np.random.default_rng(7),
    )
    # Replay the generator: one draw for the ICA seed, then the shuffle.
    generator = np.random.default_rng(7)
    generator.integers(0, 2**32 - 1)
    z = (counts - counts.mean(0)) / counts.std(0)
    shuffled = generator.permuted(z, axis=0)
    expected = np.linalg.eigvalsh(np.corrcoef(shuffled, rowvar=False)).max()
    assert result.surrogate_max_eigenvalues[0] == pytest.approx(expected, rel=1e-10)
    # Each neuron keeps its own counts, only their order changes.
    np.testing.assert_allclose(np.sort(shuffled, axis=0), np.sort(z, axis=0))


@pytest.mark.parametrize("kind", ["bin_shuffle", "circular_shift"])
def test_95th_percentile_is_close_to_marchenko_pastur_for_independent_neurons(
    kind: str,
) -> None:
    """Lopes-dos-Santos et al. (2013, Fig. 3C): the 95th percentiles of both
    surrogates are at or just below the analytical edge for Poisson neurons."""
    counts = _independent(4)
    result = detect_assemblies(
        counts,
        method="pca",
        n_components=1,
        component_threshold=kind,
        percentile=95.0,
        n_shuffles=200,
        rng=0,
    )
    edge = marchenko_pastur_threshold(20, 2000)
    assert result.threshold == pytest.approx(edge, rel=0.05)


def test_circular_shift_keeps_slow_drift_that_marchenko_pastur_calls_assemblies() -> (
    None
):
    """Independent but slowly drifting neurons: autocorrelation spreads the
    eigenvalues beyond the Marchenko-Pastur edge. Circular shifts keep each
    neuron's autocorrelation, so their threshold rises with it; bin shuffles
    destroy it and stay at the edge."""
    false_mp = false_circular = false_bin = 0
    for seed in range(5):
        counts = _slow_independent(seed)
        analytic = detect_assemblies(counts, method="pca", n_components=1)
        circular = detect_assemblies(
            counts,
            method="pca",
            n_components=1,
            component_threshold="circular_shift",
            percentile=95.0,
            n_shuffles=100,
            rng=seed,
        )
        shuffled = detect_assemblies(
            counts,
            method="pca",
            n_components=1,
            component_threshold="bin_shuffle",
            percentile=95.0,
            n_shuffles=100,
            rng=seed,
        )
        false_mp += analytic.n_significant > 0
        false_circular += circular.n_significant > 0
        false_bin += shuffled.n_significant > 0
        assert circular.threshold > analytic.threshold
    assert false_mp == 5
    assert false_bin == 5
    assert false_circular <= 1


@pytest.mark.parametrize(
    "kind", ["marchenko_pastur", "tracy_widom", "bin_shuffle", "circular_shift"]
)
def test_every_threshold_finds_a_planted_assembly(kind: str) -> None:
    rng = np.random.default_rng(5)
    counts = rng.poisson(1.0, (2000, 20)).astype(float)
    counts[:, [2, 9, 15]] += 3 * rng.poisson(0.3, (2000, 1))
    result = detect_assemblies(
        counts,
        method="pca",
        component_threshold=kind,
        percentile=None if kind in ("marchenko_pastur", "tracy_widom") else 99.0,
        n_shuffles=100,
        rng=0,
    )
    assert result.n_significant == 1
    assert result.patterns[0].member_indices.tolist() == [2, 9, 15]


def test_surrogate_threshold_needs_a_percentile() -> None:
    with pytest.raises(ValueError, match="needs percentile"):
        detect_assemblies(_independent(0), component_threshold="bin_shuffle")


def test_analytical_threshold_refuses_a_percentile() -> None:
    with pytest.raises(ValueError, match="applies only to the surrogate"):
        detect_assemblies(_independent(0), percentile=95.0)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"component_threshold": "shuffle"}, "component_threshold must be one of"),
        (
            {"component_threshold": "bin_shuffle", "percentile": 100.0},
            r"percentile must be in \(0, 100\)",
        ),
        (
            {"component_threshold": "bin_shuffle", "percentile": 95.0, "n_shuffles": 0},
            "n_shuffles must be a positive integer",
        ),
        (
            {
                "component_threshold": "bin_shuffle",
                "percentile": 95.0,
                "n_shuffles": 20.0,
            },
            "n_shuffles must be a positive integer",
        ),
        (
            {
                "component_threshold": "bin_shuffle",
                "percentile": 95.0,
                "n_shuffles": True,
            },
            "n_shuffles must be a positive integer",
        ),
    ],
)
def test_invalid_threshold_options(options: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        detect_assemblies(_independent(0), **options)
