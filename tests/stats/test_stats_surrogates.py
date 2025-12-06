"""Tests for neurospatial.stats.surrogates module.

This test file verifies that surrogate generation functions are correctly
importable from the new stats/surrogates.py location following the package
reorganization.

Functions tested:
- generate_poisson_surrogates
- generate_inhomogeneous_poisson_surrogates
- generate_jittered_spikes (NEW)
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_equal


class TestSurrogatesImports:
    """Test that all surrogate functions can be imported from stats.surrogates."""

    def test_import_generate_poisson_surrogates(self) -> None:
        """Should be importable from stats.surrogates."""
        from neurospatial.stats.surrogates import generate_poisson_surrogates

        assert callable(generate_poisson_surrogates)

    def test_import_generate_inhomogeneous_poisson_surrogates(self) -> None:
        """Should be importable from stats.surrogates."""
        from neurospatial.stats.surrogates import (
            generate_inhomogeneous_poisson_surrogates,
        )

        assert callable(generate_inhomogeneous_poisson_surrogates)

    def test_import_generate_jittered_spikes(self) -> None:
        """Should be importable from stats.surrogates (NEW function per PLAN.md)."""
        from neurospatial.stats.surrogates import generate_jittered_spikes

        assert callable(generate_jittered_spikes)

    def test_import_from_stats_package(self) -> None:
        """Should be importable from top-level stats package."""
        from neurospatial.stats import (
            generate_inhomogeneous_poisson_surrogates,
            generate_jittered_spikes,
            generate_poisson_surrogates,
        )

        assert callable(generate_poisson_surrogates)
        assert callable(generate_inhomogeneous_poisson_surrogates)
        assert callable(generate_jittered_spikes)


class TestGeneratePoissonSurrogatesNew:
    """Tests for generate_poisson_surrogates from new location."""

    @pytest.fixture
    def spike_counts(self) -> np.ndarray:
        """Sample spike counts: (10 time bins, 3 neurons)."""
        return np.array(
            [
                [1, 2, 0],
                [0, 1, 3],
                [2, 0, 1],
                [1, 1, 2],
                [0, 2, 1],
                [2, 1, 0],
                [1, 0, 2],
                [0, 3, 1],
                [2, 1, 1],
                [1, 0, 2],
            ],
            dtype=np.int64,
        )

    @pytest.fixture
    def dt(self) -> float:
        """Time bin width (25 ms)."""
        return 0.025

    def test_yields_correct_count(self, spike_counts: np.ndarray, dt: float) -> None:
        """Should yield exactly n_surrogates arrays."""
        from neurospatial.stats.surrogates import generate_poisson_surrogates

        n_surrogates = 10
        surrogates = list(
            generate_poisson_surrogates(
                spike_counts, dt, n_surrogates=n_surrogates, rng=42
            )
        )
        assert len(surrogates) == n_surrogates

    def test_yields_correct_shape(self, spike_counts: np.ndarray, dt: float) -> None:
        """Each surrogate should have same shape as input."""
        from neurospatial.stats.surrogates import generate_poisson_surrogates

        for surrogate in generate_poisson_surrogates(
            spike_counts, dt, n_surrogates=5, rng=42
        ):
            assert surrogate.shape == spike_counts.shape

    def test_yields_correct_dtype(self, spike_counts: np.ndarray, dt: float) -> None:
        """Surrogate arrays should be int64."""
        from neurospatial.stats.surrogates import generate_poisson_surrogates

        for surrogate in generate_poisson_surrogates(
            spike_counts, dt, n_surrogates=5, rng=42
        ):
            assert surrogate.dtype == np.int64

    def test_yields_non_negative_counts(
        self, spike_counts: np.ndarray, dt: float
    ) -> None:
        """Surrogate spike counts should be non-negative."""
        from neurospatial.stats.surrogates import generate_poisson_surrogates

        for surrogate in generate_poisson_surrogates(
            spike_counts, dt, n_surrogates=10, rng=42
        ):
            assert (surrogate >= 0).all()

    def test_reproducibility_with_seed(
        self, spike_counts: np.ndarray, dt: float
    ) -> None:
        """Same seed should produce same surrogates."""
        from neurospatial.stats.surrogates import generate_poisson_surrogates

        surrogates1 = list(
            generate_poisson_surrogates(spike_counts, dt, n_surrogates=5, rng=42)
        )
        surrogates2 = list(
            generate_poisson_surrogates(spike_counts, dt, n_surrogates=5, rng=42)
        )
        for s1, s2 in zip(surrogates1, surrogates2, strict=True):
            assert_array_equal(s1, s2)


class TestGenerateInhomogeneousPoissonSurrogatesNew:
    """Tests for generate_inhomogeneous_poisson_surrogates from new location."""

    @pytest.fixture
    def spike_counts(self) -> np.ndarray:
        """Sample spike counts with temporal structure: (10 time bins, 3 neurons)."""
        return np.array(
            [
                [1, 2, 0],
                [0, 1, 3],
                [2, 0, 1],
                [1, 1, 2],
                [0, 2, 1],
                [2, 1, 0],
                [1, 0, 2],
                [0, 3, 1],
                [2, 1, 1],
                [1, 0, 2],
            ],
            dtype=np.int64,
        )

    @pytest.fixture
    def dt(self) -> float:
        """Time bin width (25 ms)."""
        return 0.025

    def test_yields_correct_count(self, spike_counts: np.ndarray, dt: float) -> None:
        """Should yield exactly n_surrogates arrays."""
        from neurospatial.stats.surrogates import (
            generate_inhomogeneous_poisson_surrogates,
        )

        n_surrogates = 10
        surrogates = list(
            generate_inhomogeneous_poisson_surrogates(
                spike_counts, dt, n_surrogates=n_surrogates, rng=42
            )
        )
        assert len(surrogates) == n_surrogates

    def test_yields_correct_shape(self, spike_counts: np.ndarray, dt: float) -> None:
        """Each surrogate should have same shape as input."""
        from neurospatial.stats.surrogates import (
            generate_inhomogeneous_poisson_surrogates,
        )

        for surrogate in generate_inhomogeneous_poisson_surrogates(
            spike_counts, dt, n_surrogates=5, rng=42
        ):
            assert surrogate.shape == spike_counts.shape

    def test_yields_correct_dtype(self, spike_counts: np.ndarray, dt: float) -> None:
        """Surrogate arrays should be int64."""
        from neurospatial.stats.surrogates import (
            generate_inhomogeneous_poisson_surrogates,
        )

        for surrogate in generate_inhomogeneous_poisson_surrogates(
            spike_counts, dt, n_surrogates=5, rng=42
        ):
            assert surrogate.dtype == np.int64

    def test_reproducibility_with_seed(
        self, spike_counts: np.ndarray, dt: float
    ) -> None:
        """Same seed should produce same surrogates."""
        from neurospatial.stats.surrogates import (
            generate_inhomogeneous_poisson_surrogates,
        )

        surrogates1 = list(
            generate_inhomogeneous_poisson_surrogates(
                spike_counts, dt, n_surrogates=5, rng=42
            )
        )
        surrogates2 = list(
            generate_inhomogeneous_poisson_surrogates(
                spike_counts, dt, n_surrogates=5, rng=42
            )
        )
        for s1, s2 in zip(surrogates1, surrogates2, strict=True):
            assert_array_equal(s1, s2)


class TestGenerateJitteredSpikes:
    """Tests for generate_jittered_spikes (NEW function per PLAN.md)."""

    @pytest.fixture
    def spike_times(self) -> np.ndarray:
        """Sample spike times (sorted)."""
        return np.array([0.1, 0.15, 0.25, 0.4, 0.45, 0.7, 0.85, 1.0])

    def test_yields_correct_count(self, spike_times: np.ndarray) -> None:
        """Should yield exactly n_surrogates arrays."""
        from neurospatial.stats.surrogates import generate_jittered_spikes

        n_surrogates = 10
        surrogates = list(
            generate_jittered_spikes(
                spike_times, jitter_std=0.01, n_surrogates=n_surrogates, rng=42
            )
        )
        assert len(surrogates) == n_surrogates

    def test_yields_same_length(self, spike_times: np.ndarray) -> None:
        """Each surrogate should have same number of spikes as input."""
        from neurospatial.stats.surrogates import generate_jittered_spikes

        for surrogate in generate_jittered_spikes(
            spike_times, jitter_std=0.01, n_surrogates=5, rng=42
        ):
            assert len(surrogate) == len(spike_times)

    def test_spikes_are_sorted(self, spike_times: np.ndarray) -> None:
        """Output spike times should be sorted."""
        from neurospatial.stats.surrogates import generate_jittered_spikes

        for surrogate in generate_jittered_spikes(
            spike_times, jitter_std=0.01, n_surrogates=5, rng=42
        ):
            assert np.all(np.diff(surrogate) >= 0)

    def test_reproducibility_with_seed(self, spike_times: np.ndarray) -> None:
        """Same seed should produce same surrogates."""
        from neurospatial.stats.surrogates import generate_jittered_spikes

        surrogates1 = list(
            generate_jittered_spikes(
                spike_times, jitter_std=0.01, n_surrogates=5, rng=42
            )
        )
        surrogates2 = list(
            generate_jittered_spikes(
                spike_times, jitter_std=0.01, n_surrogates=5, rng=42
            )
        )
        for s1, s2 in zip(surrogates1, surrogates2, strict=True):
            assert_array_equal(s1, s2)

    def test_small_jitter_stays_close(self, spike_times: np.ndarray) -> None:
        """With small jitter, spikes should stay close to original times."""
        from neurospatial.stats.surrogates import generate_jittered_spikes

        jitter_std = 0.001  # 1 ms
        for surrogate in generate_jittered_spikes(
            spike_times, jitter_std=jitter_std, n_surrogates=5, rng=42
        ):
            # With 1 ms jitter, max deviation should be small (< 10 ms usually)
            max_deviation = np.max(np.abs(surrogate - spike_times))
            assert max_deviation < 0.05  # 50 ms max

    def test_empty_spikes(self) -> None:
        """Should handle empty spike array."""
        from neurospatial.stats.surrogates import generate_jittered_spikes

        empty_spikes = np.array([])
        surrogates = list(
            generate_jittered_spikes(
                empty_spikes, jitter_std=0.01, n_surrogates=3, rng=42
            )
        )
        assert len(surrogates) == 3
        for s in surrogates:
            assert len(s) == 0

    def test_single_spike(self) -> None:
        """Should handle single spike."""
        from neurospatial.stats.surrogates import generate_jittered_spikes

        single_spike = np.array([0.5])
        surrogates = list(
            generate_jittered_spikes(
                single_spike, jitter_std=0.01, n_surrogates=3, rng=42
            )
        )
        assert len(surrogates) == 3
        for s in surrogates:
            assert len(s) == 1

    def test_window_constraint(self, spike_times: np.ndarray) -> None:
        """Spikes should stay within window if specified."""
        from neurospatial.stats.surrogates import generate_jittered_spikes

        window = (0.0, 1.5)
        for surrogate in generate_jittered_spikes(
            spike_times, jitter_std=0.05, n_surrogates=5, rng=42, window=window
        ):
            assert np.all(surrogate >= window[0])
            assert np.all(surrogate <= window[1])
