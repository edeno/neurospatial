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

    def test_grid_score_anchor_perfect_hexagonal(self):
        """Anchor test: Synthetic perfect hexagonal pattern should score > 0.3.

        Mathematical reasoning:
        - Perfect hexagonal symmetry has correlation at 60°, 120° rotations
        - Grid score = min(r60, r120) - max(r30, r90, r150)
        - For perfect hex: r60 ≈ r120 ≈ 1.0, r30 ≈ r90 ≈ r150 ≈ low
        - Expected score > 0.3 (literature threshold for grid cells)

        Reference: Langston et al. (2010) Science
        """
        from neurospatial.metrics.grid_cells import grid_score

        # Create synthetic autocorrelogram with perfect hexagonal peaks
        size = 100
        autocorr = np.zeros((size, size))
        center = size // 2

        # Central peak
        y_grid, x_grid = np.ogrid[:size, :size]
        dist_from_center = np.sqrt((y_grid - center) ** 2 + (x_grid - center) ** 2)
        autocorr = np.exp(-(dist_from_center**2) / (2 * 5**2))  # Central Gaussian

        # Add 6 peaks at 60° intervals (hexagonal pattern)
        radius = 20  # Distance from center to peaks
        for angle_deg in [0, 60, 120, 180, 240, 300]:
            angle_rad = np.radians(angle_deg)
            peak_y = center + int(radius * np.sin(angle_rad))
            peak_x = center + int(radius * np.cos(angle_rad))
            peak_dist = np.sqrt((y_grid - peak_y) ** 2 + (x_grid - peak_x) ** 2)
            autocorr += 0.8 * np.exp(-(peak_dist**2) / (2 * 5**2))

        # Normalize
        autocorr = autocorr / autocorr.max()

        score = grid_score(autocorr)

        # Perfect hexagonal pattern should have score > 0.3 (conservative threshold)
        assert score > 0.3, (
            f"Expected grid score > 0.3 for hexagonal pattern, got {score}"
        )

    def test_grid_score_anchor_isotropic_noise(self):
        """Anchor test: Isotropic random noise should have score near 0.

        Mathematical reasoning:
        - Random noise has no systematic angular structure
        - Correlations at all rotation angles are approximately equal
        - Grid score = min(r60, r120) - max(r30, r90, r150) ≈ 0
        """
        from neurospatial.metrics.grid_cells import grid_score

        # Create random autocorrelogram (seeded for reproducibility)
        rng = np.random.default_rng(12345)
        size = 50
        autocorr = rng.random((size, size)) * 0.5 + 0.5

        score = grid_score(autocorr)

        # Random noise should have score near 0 (within ±0.3)
        assert abs(score) < 0.3, f"Expected grid score near 0 for noise, got {score}"

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

    def test_periodicity_score_anchor_perfect_sine(self):
        """Anchor test: Perfect sinusoidal correlation should have high periodicity.

        Mathematical reasoning:
        - Sinusoidal pattern has evenly-spaced peaks
        - Peak spacing is perfectly regular (variance = 0)
        - Periodicity score = 1 - CV(peak_spacings) where CV = std/mean
        - For perfect periodicity: CV ≈ 0, score ≈ 1.0
        """
        from neurospatial.metrics.grid_cells import periodicity_score

        # Create perfect sinusoidal correlation profile
        distances = np.linspace(0, 100, 200)
        period = 20.0  # Peaks every 20 units
        correlations = 0.5 * np.sin(2 * np.pi * distances / period) + 0.5

        score = periodicity_score(distances, correlations, min_peaks=2)

        # Perfect sine should have high periodicity score (> 0.8)
        assert score > 0.8, f"Expected periodicity score > 0.8 for sine, got {score}"

    def test_periodicity_score_anchor_jittered_peaks(self):
        """Anchor test: Highly jittered peaks should have low periodicity.

        Mathematical reasoning:
        - Random peak spacing has high variance
        - CV = std/mean will be high when spacing varies
        - Periodicity score = 1 - CV will be low
        """
        from neurospatial.metrics.grid_cells import periodicity_score

        # Create correlation with irregular peaks (jittered positions)
        distances = np.linspace(0, 100, 200)
        correlations = np.zeros(200)

        # Add peaks at irregular intervals (10, 25, 55, 90) instead of regular
        for peak_pos in [10, 25, 55, 90]:
            peak_idx = np.argmin(np.abs(distances - peak_pos))
            # Add Gaussian-shaped peak
            peak_width = 3
            for i in range(max(0, peak_idx - 10), min(200, peak_idx + 10)):
                dist_from_peak = abs(i - peak_idx)
                correlations[i] += np.exp(-(dist_from_peak**2) / (2 * peak_width**2))

        correlations = correlations / correlations.max()

        score = periodicity_score(distances, correlations, min_peaks=3)

        # Irregular spacing should have lower periodicity score (< 0.7)
        assert score < 0.7, (
            f"Expected periodicity score < 0.7 for jittered, got {score}"
        )

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


def _create_hexagonal_autocorr(size: int = 100, radius: float = 20.0) -> np.ndarray:
    """Create synthetic hexagonal autocorrelogram for testing."""
    autocorr = np.zeros((size, size))
    center = size // 2

    # Coordinate grids
    y_grid, x_grid = np.ogrid[:size, :size]

    # Central peak
    dist_from_center = np.sqrt((y_grid - center) ** 2 + (x_grid - center) ** 2)
    autocorr = np.exp(-(dist_from_center**2) / (2 * 5**2))

    # Add 6 peaks at 60° intervals (hexagonal pattern)
    for angle_deg in [0, 60, 120, 180, 240, 300]:
        angle_rad = np.radians(angle_deg)
        peak_y = center + int(radius * np.sin(angle_rad))
        peak_x = center + int(radius * np.cos(angle_rad))
        peak_dist = np.sqrt((y_grid - peak_y) ** 2 + (x_grid - peak_x) ** 2)
        autocorr += 0.8 * np.exp(-(peak_dist**2) / (2 * 5**2))

    return autocorr / autocorr.max()


class TestGridScale:
    """Tests for grid_scale function."""

    def test_grid_scale_anchor_hexagonal_pattern(self):
        """Anchor test: Hexagonal pattern with known radius should return correct scale.

        Mathematical reasoning:
        - Peaks are placed at radius=20 pixels from center
        - With bin_size=2.0, expected scale = 20 * 2.0 = 40.0 units
        """
        from neurospatial.metrics.grid_cells import grid_scale

        # Create hexagonal pattern with known radius
        radius_pixels = 20.0
        bin_size = 2.0
        autocorr = _create_hexagonal_autocorr(size=100, radius=radius_pixels)

        scale = grid_scale(autocorr, bin_size=bin_size)

        # Should recover approximately the correct scale
        expected_scale = radius_pixels * bin_size
        assert abs(scale - expected_scale) < 10.0, (
            f"Expected scale ~{expected_scale}, got {scale}"
        )

    def test_grid_scale_returns_nan_on_flat_input(self):
        """Test returns NaN when no clear peaks detected (flat data)."""
        from neurospatial.metrics.grid_cells import grid_scale

        # Truly flat data (no local maxima possible)
        autocorr = np.ones((50, 50)) * 0.5

        scale = grid_scale(autocorr, bin_size=2.0)

        # Should return NaN (no peaks in flat data)
        assert np.isnan(scale)

    def test_grid_scale_raises_on_1d_input(self):
        """Test raises ValueError on 1D input."""
        from neurospatial.metrics.grid_cells import grid_scale

        autocorr = np.zeros(100)

        with pytest.raises(ValueError, match="autocorr_2d must be 2D"):
            grid_scale(autocorr, bin_size=2.0)

    def test_grid_scale_raises_on_invalid_bin_size(self):
        """Test raises ValueError on non-positive bin_size."""
        from neurospatial.metrics.grid_cells import grid_scale

        autocorr = _create_hexagonal_autocorr()

        with pytest.raises(ValueError, match="bin_size must be positive"):
            grid_scale(autocorr, bin_size=0.0)

        with pytest.raises(ValueError, match="bin_size must be positive"):
            grid_scale(autocorr, bin_size=-1.0)

    def test_grid_scale_scales_with_bin_size(self):
        """Test scale output is proportional to bin_size."""
        from neurospatial.metrics.grid_cells import grid_scale

        autocorr = _create_hexagonal_autocorr()

        scale1 = grid_scale(autocorr, bin_size=1.0)
        scale2 = grid_scale(autocorr, bin_size=2.0)

        # scale2 should be approximately 2x scale1
        assert abs(scale2 / scale1 - 2.0) < 0.1


class TestGridOrientation:
    """Tests for grid_orientation function."""

    def test_grid_orientation_anchor_horizontal_grid(self):
        """Anchor test: Grid aligned with horizontal axis should have orientation ~0°.

        Mathematical reasoning:
        - Peaks at 0°, 60°, 120°, 180°, 240°, 300° relative to horizontal
        - First peak at 0° (horizontal)
        - Orientation should be close to 0° (or equivalently ~60°)
        """
        from neurospatial.metrics.grid_cells import grid_orientation

        autocorr = _create_hexagonal_autocorr()

        orientation, orientation_std = grid_orientation(autocorr)

        # Orientation should be near 0° or 60° (equivalent due to symmetry)
        assert orientation >= 0.0
        assert orientation < 60.0
        # Std should be relatively low for clean pattern
        assert orientation_std < 15.0

    def test_grid_orientation_anchor_rotated_grid(self):
        """Anchor test: Grid rotated by 30° should have orientation ~30°."""
        from neurospatial.metrics.grid_cells import grid_orientation

        # Create hexagonal pattern rotated by 30°
        size = 100
        autocorr = np.zeros((size, size))
        center = size // 2
        radius = 20.0

        y_grid, x_grid = np.ogrid[:size, :size]

        # Central peak
        dist_from_center = np.sqrt((y_grid - center) ** 2 + (x_grid - center) ** 2)
        autocorr = np.exp(-(dist_from_center**2) / (2 * 5**2))

        # Add 6 peaks at 60° intervals, starting from 30° (not 0°)
        for angle_deg in [30, 90, 150, 210, 270, 330]:
            angle_rad = np.radians(angle_deg)
            peak_y = center + int(radius * np.sin(angle_rad))
            peak_x = center + int(radius * np.cos(angle_rad))
            peak_dist = np.sqrt((y_grid - peak_y) ** 2 + (x_grid - peak_x) ** 2)
            autocorr += 0.8 * np.exp(-(peak_dist**2) / (2 * 5**2))

        autocorr = autocorr / autocorr.max()

        orientation, _orientation_std = grid_orientation(autocorr)

        # Orientation should be near 30°
        assert 20.0 < orientation < 40.0, (
            f"Expected orientation ~30°, got {orientation}"
        )

    def test_grid_orientation_returns_nan_on_flat_input(self):
        """Test returns NaN when no clear peaks detected (flat data)."""
        from neurospatial.metrics.grid_cells import grid_orientation

        # Truly flat data (no local maxima possible)
        autocorr = np.ones((50, 50)) * 0.5

        orientation, orientation_std = grid_orientation(autocorr)

        # Should return NaN (no peaks in flat data)
        assert np.isnan(orientation)
        assert np.isnan(orientation_std)

    def test_grid_orientation_in_valid_range(self):
        """Test orientation is in range [0, 60)."""
        from neurospatial.metrics.grid_cells import grid_orientation

        autocorr = _create_hexagonal_autocorr()

        orientation, _ = grid_orientation(autocorr)

        assert 0.0 <= orientation < 60.0

    def test_grid_orientation_raises_on_1d_input(self):
        """Test raises ValueError on 1D input."""
        from neurospatial.metrics.grid_cells import grid_orientation

        autocorr = np.zeros(100)

        with pytest.raises(ValueError, match="autocorr_2d must be 2D"):
            grid_orientation(autocorr)


class TestGridProperties:
    """Tests for grid_properties function and GridProperties dataclass."""

    def test_grid_properties_returns_dataclass(self):
        """Test grid_properties returns GridProperties dataclass."""
        from neurospatial.metrics.grid_cells import GridProperties, grid_properties

        autocorr = _create_hexagonal_autocorr()

        props = grid_properties(autocorr, bin_size=2.0)

        assert isinstance(props, GridProperties)
        assert hasattr(props, "score")
        assert hasattr(props, "scale")
        assert hasattr(props, "orientation")
        assert hasattr(props, "orientation_std")
        assert hasattr(props, "peak_coords")
        assert hasattr(props, "n_peaks")

    def test_grid_properties_anchor_hexagonal(self):
        """Anchor test: Hexagonal pattern should have consistent properties."""
        from neurospatial.metrics.grid_cells import grid_properties

        autocorr = _create_hexagonal_autocorr(size=100, radius=20.0)
        bin_size = 2.0

        props = grid_properties(autocorr, bin_size=bin_size)

        # Score should be positive (hexagonal pattern)
        assert props.score > 0.3

        # Scale should be approximately 20 * 2 = 40
        assert 30.0 < props.scale < 50.0

        # Orientation should be in valid range
        assert 0.0 <= props.orientation < 60.0

        # Should detect 6 peaks
        assert props.n_peaks >= 5  # Allow some tolerance

        # Peak coords should be 2D array
        assert props.peak_coords.ndim == 2
        assert props.peak_coords.shape[1] == 2

    def test_grid_properties_matches_individual_functions(self):
        """Test grid_properties matches individual function outputs."""
        from neurospatial.metrics.grid_cells import (
            grid_orientation,
            grid_properties,
            grid_scale,
            grid_score,
        )

        autocorr = _create_hexagonal_autocorr()
        bin_size = 2.0

        # Get combined properties
        props = grid_properties(autocorr, bin_size=bin_size)

        # Get individual function outputs
        score_individual = grid_score(autocorr)
        scale_individual = grid_scale(autocorr, bin_size=bin_size)
        orientation_individual, _std_individual = grid_orientation(autocorr)

        # Score should match exactly
        assert props.score == score_individual

        # Scale should match (within tolerance due to peak detection variance)
        assert abs(props.scale - scale_individual) < 5.0

        # Orientation should be close (within tolerance)
        assert abs(props.orientation - orientation_individual) < 5.0

    def test_grid_properties_raises_on_invalid_input(self):
        """Test raises ValueError on invalid input."""
        from neurospatial.metrics.grid_cells import grid_properties

        # 1D input
        with pytest.raises(ValueError, match="autocorr_2d must be 2D"):
            grid_properties(np.zeros(100), bin_size=2.0)

        # Invalid bin_size
        with pytest.raises(ValueError, match="bin_size must be positive"):
            grid_properties(np.zeros((50, 50)), bin_size=0.0)

    def test_grid_properties_handles_no_peaks(self):
        """Test handles case with no detectable peaks (flat data)."""
        from neurospatial.metrics.grid_cells import grid_properties

        # Truly flat data (no local maxima possible)
        autocorr = np.ones((50, 50)) * 0.5

        props = grid_properties(autocorr, bin_size=2.0)

        # Score is computed independently (may be non-NaN for flat data)
        # Scale and orientation should be NaN
        assert np.isnan(props.scale)
        assert np.isnan(props.orientation)
        assert np.isnan(props.orientation_std)
        # n_peaks should be 0 for flat data
        assert props.n_peaks == 0


class TestFindAutocorrPeaks:
    """Tests for _find_autocorr_peaks helper function."""

    def test_finds_hexagonal_peaks(self):
        """Test finds 6 peaks in hexagonal pattern."""
        from neurospatial.metrics.grid_cells import _find_autocorr_peaks

        autocorr = _create_hexagonal_autocorr()

        peaks = _find_autocorr_peaks(autocorr)

        # Should find 6 peaks (excluding central peak)
        assert 5 <= len(peaks) <= 7

    def test_excludes_central_peak(self):
        """Test excludes central peak from results."""
        from neurospatial.metrics.grid_cells import _find_autocorr_peaks

        autocorr = _create_hexagonal_autocorr()

        peaks = _find_autocorr_peaks(autocorr)

        # All peaks should be non-zero distance from center
        distances = np.sqrt(peaks[:, 0] ** 2 + peaks[:, 1] ** 2)
        assert np.all(distances > 5)

    def test_returns_center_relative_coords(self):
        """Test returns coordinates relative to center."""
        from neurospatial.metrics.grid_cells import _find_autocorr_peaks

        autocorr = _create_hexagonal_autocorr(size=100)

        peaks = _find_autocorr_peaks(autocorr)

        # Peaks should be centered around (0, 0)
        # Mean should be near zero for symmetric pattern
        assert abs(np.mean(peaks[:, 0])) < 5
        assert abs(np.mean(peaks[:, 1])) < 5

    def test_sorted_by_distance(self):
        """Test peaks are sorted by distance from center."""
        from neurospatial.metrics.grid_cells import _find_autocorr_peaks

        autocorr = _create_hexagonal_autocorr()

        peaks = _find_autocorr_peaks(autocorr)

        if len(peaks) > 1:
            distances = np.sqrt(peaks[:, 0] ** 2 + peaks[:, 1] ** 2)
            # Should be sorted in ascending order
            assert np.all(np.diff(distances) >= 0)

    def test_returns_empty_on_flat_input(self):
        """Test returns empty array on flat input (no peaks)."""
        from neurospatial.metrics.grid_cells import _find_autocorr_peaks

        autocorr = np.ones((50, 50)) * 0.5

        peaks = _find_autocorr_peaks(autocorr)

        assert len(peaks) == 0
        assert peaks.shape == (0, 2)
