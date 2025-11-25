"""
Tests for grid cell metrics.

Tests spatial_autocorrelation, grid_score, and periodicity_score functions.
"""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial import Environment


class TestSpatialAutocorrelation:
    """Tests for spatial_autocorrelation function."""

    def test_fft_method_on_regular_grid(self):
        """Test FFT method returns 2D autocorrelation on regular grid."""
        from neurospatial.metrics.grid_cells import spatial_autocorrelation

        # Create regular 2D grid environment using deterministic grid
        x = np.linspace(-20, 20, 21)
        xx, yy = np.meshgrid(x, x)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(positions, bin_size=4.0)

        # Create simple firing pattern (single peak)
        firing_rate = np.zeros(env.n_bins)
        center_bin = env.n_bins // 2
        firing_rate[center_bin] = 10.0

        # Compute autocorrelation with FFT method
        autocorr = spatial_autocorrelation(firing_rate, env, method="fft")

        # Should return 2D array
        assert autocorr.ndim == 2
        assert autocorr.shape == env.layout.grid_shape

        # Center should have highest autocorrelation (≈1.0)
        center_y, center_x = autocorr.shape[0] // 2, autocorr.shape[1] // 2
        assert autocorr[center_y, center_x] > 0.5

    def test_graph_method_returns_tuple(self):
        """Test graph method returns (distances, correlations) tuple."""
        from neurospatial.metrics.grid_cells import spatial_autocorrelation

        # Create environment using deterministic grid
        x = np.linspace(-20, 20, 21)
        xx, yy = np.meshgrid(x, x)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(positions, bin_size=4.0)

        # Random firing pattern - local RNG for isolation
        rng = np.random.default_rng(42)
        firing_rate = rng.random(env.n_bins) * 5.0

        # Compute autocorrelation with graph method
        result = spatial_autocorrelation(
            firing_rate, env, method="graph", n_distance_bins=30
        )

        # Should return tuple
        assert isinstance(result, tuple)
        assert len(result) == 2

        distances, correlations = result

        # Both should be 1D arrays with same length
        assert distances.ndim == 1
        assert correlations.ndim == 1
        assert distances.shape == correlations.shape
        assert len(distances) == 30

    def test_auto_method_works(self):
        """Test auto method can detect appropriate method."""
        from neurospatial.metrics.grid_cells import spatial_autocorrelation

        # Create environment using deterministic grid
        x = np.linspace(-20, 20, 21)
        xx, yy = np.meshgrid(x, x)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(positions, bin_size=4.0)

        rng = np.random.default_rng(42)
        firing_rate = rng.random(env.n_bins) * 5.0

        # Auto should work without error (returns either 2D array or tuple)
        result = spatial_autocorrelation(firing_rate, env, method="auto")

        # Should return something valid (either 2D array or tuple)
        assert result is not None
        # Either 2D array (FFT) or tuple (graph)
        assert isinstance(result, (np.ndarray, tuple))

    def test_raises_on_shape_mismatch(self):
        """Test raises ValueError if firing_rate shape doesn't match env.n_bins."""
        from neurospatial.metrics.grid_cells import spatial_autocorrelation

        # Deterministic grid
        x = np.linspace(-20, 20, 21)
        xx, yy = np.meshgrid(x, x)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(positions, bin_size=4.0)

        firing_rate = np.zeros(env.n_bins + 10)  # Wrong size

        with pytest.raises(ValueError, match=r"firing_rate\.shape must be"):
            spatial_autocorrelation(firing_rate, env)

    def test_raises_on_all_nan(self):
        """Test raises ValueError if all firing rates are NaN."""
        from neurospatial.metrics.grid_cells import spatial_autocorrelation

        # Deterministic grid
        x = np.linspace(-20, 20, 21)
        xx, yy = np.meshgrid(x, x)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(positions, bin_size=4.0)

        firing_rate = np.full(env.n_bins, np.nan)

        with pytest.raises(ValueError, match="All firing rates are NaN"):
            spatial_autocorrelation(firing_rate, env)

    def test_raises_on_constant_rates(self):
        """Test raises ValueError if all valid firing rates are constant."""
        from neurospatial.metrics.grid_cells import spatial_autocorrelation

        # Deterministic grid
        x = np.linspace(-20, 20, 21)
        xx, yy = np.meshgrid(x, x)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(positions, bin_size=4.0)

        firing_rate = np.full(env.n_bins, 5.0)  # All constant

        with pytest.raises(ValueError, match="All valid firing rates are constant"):
            spatial_autocorrelation(firing_rate, env)

    def test_raises_on_invalid_method(self):
        """Test raises ValueError on invalid method parameter."""
        from neurospatial.metrics.grid_cells import spatial_autocorrelation

        # Deterministic grid
        x = np.linspace(-20, 20, 21)
        xx, yy = np.meshgrid(x, x)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(positions, bin_size=4.0)

        rng = np.random.default_rng(42)
        firing_rate = rng.random(env.n_bins) * 5.0

        with pytest.raises(ValueError, match="method must be"):
            spatial_autocorrelation(firing_rate, env, method="invalid")

    def test_fft_autocorr_returns_finite_values(self):
        """Test FFT autocorrelation returns finite values."""
        from neurospatial.metrics.grid_cells import spatial_autocorrelation

        # Deterministic grid
        x = np.linspace(-20, 20, 21)
        xx, yy = np.meshgrid(x, x)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(positions, bin_size=4.0)

        # Create firing pattern with local RNG
        rng = np.random.default_rng(42)
        firing_rate = rng.random(env.n_bins) * 5.0

        autocorr = spatial_autocorrelation(firing_rate, env, method="fft")

        # Autocorrelation should be finite everywhere
        assert np.all(np.isfinite(autocorr))


class TestGridScore:
    """Tests for grid_score function."""

    def test_hexagonal_pattern_high_score(self):
        """Test hexagonal grid pattern produces high grid score."""
        from neurospatial.metrics.grid_cells import grid_score

        # Create synthetic hexagonal autocorrelogram
        # (Simplified: create peaks at 60° and 120° positions)
        size = 50
        autocorr = np.zeros((size, size))

        # Central peak
        center = size // 2
        autocorr[center, center] = 1.0

        # Add hexagonal peaks (6 peaks around center at 60° intervals)
        # This is a simplified approximation
        radius = size // 5
        for angle_deg in [0, 60, 120, 180, 240, 300]:
            angle_rad = np.radians(angle_deg)
            dy = int(radius * np.sin(angle_rad))
            dx = int(radius * np.cos(angle_rad))
            y, x = center + dy, center + dx
            if 0 <= y < size and 0 <= x < size:
                autocorr[y, x] = 0.6

        score = grid_score(autocorr)

        # Hexagonal pattern should produce positive score
        # (Not necessarily > 0.4 for this simplified pattern, but should be > 0)
        assert score > 0.0

    def test_place_cell_pattern_low_score(self):
        """Test place cell (single peak) produces low grid score."""
        from neurospatial.metrics.grid_cells import grid_score

        # Create autocorrelogram with single central peak (place cell)
        size = 50
        autocorr = np.zeros((size, size))

        # Gaussian peak at center
        center = size // 2
        y, x = np.ogrid[:size, :size]
        distance = np.sqrt((y - center) ** 2 + (x - center) ** 2)
        autocorr = np.exp(-(distance**2) / (2 * (size / 6) ** 2))

        score = grid_score(autocorr)

        # Single peak should produce near-zero or negative score
        # (No hexagonal symmetry)
        assert score < 0.3

    def test_random_noise_near_zero_score(self):
        """Test random noise produces near-zero grid score."""
        from neurospatial.metrics.grid_cells import grid_score

        # Random autocorrelogram with local RNG
        rng = np.random.default_rng(42)
        autocorr = rng.standard_normal((50, 50)) * 0.1

        score = grid_score(autocorr)

        # Random noise should produce score near 0 (no structure)
        assert -0.5 < score < 0.5

    def test_raises_on_1d_input(self):
        """Test raises ValueError on 1D input."""
        from neurospatial.metrics.grid_cells import grid_score

        rng = np.random.default_rng(42)
        autocorr = rng.standard_normal(50)

        with pytest.raises(ValueError, match="autocorr_2d must be 2D"):
            grid_score(autocorr)

    def test_raises_on_invalid_radii(self):
        """Test raises ValueError on invalid radius parameters."""
        from neurospatial.metrics.grid_cells import grid_score

        rng = np.random.default_rng(42)
        autocorr = rng.standard_normal((50, 50))

        # inner_radius_fraction not in (0, 1)
        with pytest.raises(ValueError, match="inner_radius_fraction must be in"):
            grid_score(autocorr, inner_radius_fraction=1.5)

        # outer <= inner
        with pytest.raises(ValueError, match="outer_radius_fraction must be in"):
            grid_score(autocorr, inner_radius_fraction=0.5, outer_radius_fraction=0.3)

    def test_returns_nan_on_all_nan_input(self):
        """Test returns NaN if autocorrelation contains NaN."""
        from neurospatial.metrics.grid_cells import grid_score

        autocorr = np.full((50, 50), np.nan)

        score = grid_score(autocorr)

        assert np.isnan(score)

    def test_score_in_valid_range(self):
        """Test grid score is in expected range [-2, 2]."""
        from neurospatial.metrics.grid_cells import grid_score

        # Random autocorrelogram with local RNG
        rng = np.random.default_rng(42)
        autocorr = rng.standard_normal((50, 50))

        score = grid_score(autocorr)

        # Score should be in valid range
        assert -2.0 <= score <= 2.0


class TestPeriodicityScore:
    """Tests for periodicity_score function."""

    def test_regular_peaks_high_score(self):
        """Test regular periodic peaks produce high periodicity score."""
        from neurospatial.metrics.grid_cells import periodicity_score

        # Create regularly spaced peaks
        distances = np.linspace(0, 100, 100)
        correlations = np.zeros(100)

        # Add peaks at regular intervals (every 20 units)
        peak_indices = [10, 30, 50, 70, 90]
        correlations[peak_indices] = [0.8, 0.7, 0.6, 0.5, 0.4]

        score = periodicity_score(distances, correlations, min_peaks=3)

        # Regular spacing should produce high score
        assert score > 0.6

    def test_irregular_peaks_low_score(self):
        """Test irregular peaks produce low periodicity score."""
        from neurospatial.metrics.grid_cells import periodicity_score

        # Create irregularly spaced peaks
        distances = np.linspace(0, 100, 100)
        correlations = np.zeros(100)

        # Add peaks at irregular intervals
        peak_indices = [10, 15, 40, 90]
        correlations[peak_indices] = [0.8, 0.7, 0.6, 0.5]

        score = periodicity_score(distances, correlations, min_peaks=3)

        # Irregular spacing should produce lower score
        assert score < 0.7

    def test_no_peaks_returns_nan(self):
        """Test returns NaN if no peaks detected."""
        from neurospatial.metrics.grid_cells import periodicity_score

        # Flat correlation profile (no peaks)
        distances = np.linspace(0, 100, 100)
        correlations = np.ones(100) * 0.5

        score = periodicity_score(distances, correlations)

        assert np.isnan(score)

    def test_raises_on_shape_mismatch(self):
        """Test raises ValueError if distances and correlations have different shapes."""
        from neurospatial.metrics.grid_cells import periodicity_score

        distances = np.linspace(0, 100, 100)
        correlations = np.zeros(50)  # Different size

        with pytest.raises(ValueError, match="must have same shape"):
            periodicity_score(distances, correlations)

    def test_raises_on_empty_input(self):
        """Test raises ValueError on empty input."""
        from neurospatial.metrics.grid_cells import periodicity_score

        distances = np.array([])
        correlations = np.array([])

        with pytest.raises(ValueError, match="cannot be empty"):
            periodicity_score(distances, correlations)

    def test_raises_on_invalid_min_peaks(self):
        """Test raises ValueError on invalid min_peaks parameter."""
        from neurospatial.metrics.grid_cells import periodicity_score

        distances = np.linspace(0, 100, 100)
        correlations = np.zeros(100)

        with pytest.raises(ValueError, match="min_peaks must be"):
            periodicity_score(distances, correlations, min_peaks=1)

    def test_returns_nan_on_all_nan(self):
        """Test returns NaN if all correlations are NaN."""
        from neurospatial.metrics.grid_cells import periodicity_score

        distances = np.linspace(0, 100, 100)
        correlations = np.full(100, np.nan)

        score = periodicity_score(distances, correlations)

        assert np.isnan(score)

    def test_score_in_valid_range(self):
        """Test periodicity score is in range [0, 1]."""
        from neurospatial.metrics.grid_cells import periodicity_score

        # Create some peaks
        distances = np.linspace(0, 100, 100)
        correlations = np.zeros(100)
        correlations[[10, 30, 50, 70]] = [0.8, 0.7, 0.6, 0.5]

        score = periodicity_score(distances, correlations)

        # Score should be in [0, 1]
        assert 0.0 <= score <= 1.0
