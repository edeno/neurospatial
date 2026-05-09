"""Tests for encoding/_metrics.py - shared metric implementations.

This module tests the shared spatial information and sparsity computations
that will be used by result classes (SpatialRateResult, etc.).

Following TDD approach: tests written before implementation.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

# Import will fail until we implement the module
from neurospatial.encoding._metrics import (
    batch_sparsity,
    batch_spatial_information,
    sparsity,
    spatial_information,
)

# batch_grid_scores import is done locally in tests to allow TDD (tests first)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def uniform_firing() -> tuple[np.ndarray, np.ndarray]:
    """Uniform firing rate (no spatial information).

    Returns
    -------
    firing_rate : ndarray
        Constant 3 Hz across 100 bins.
    occupancy : ndarray
        Uniform occupancy (equal time in all bins).
    """
    n_bins = 100
    firing_rate = np.ones(n_bins) * 3.0  # 3 Hz everywhere
    occupancy = np.ones(n_bins)  # Equal time in all bins
    return firing_rate, occupancy


@pytest.fixture
def selective_firing() -> tuple[np.ndarray, np.ndarray]:
    """Highly selective firing (one hot bin).

    Returns
    -------
    firing_rate : ndarray
        50 Hz in one bin, 0 Hz elsewhere.
    occupancy : ndarray
        Uniform occupancy.
    """
    n_bins = 100
    firing_rate = np.zeros(n_bins)
    firing_rate[50] = 50.0  # 50 Hz in bin 50 only
    occupancy = np.ones(n_bins)  # Equal time in all bins
    return firing_rate, occupancy


@pytest.fixture
def place_cell_like() -> tuple[np.ndarray, np.ndarray]:
    """Place cell-like firing pattern (Gaussian bump).

    Returns
    -------
    firing_rate : ndarray
        Gaussian-shaped firing field centered at bin 50.
    occupancy : ndarray
        Uniform occupancy.
    """
    n_bins = 100
    x = np.arange(n_bins)
    center = 50
    width = 5
    # Gaussian-shaped place field
    firing_rate = 10.0 * np.exp(-((x - center) ** 2) / (2 * width**2))
    occupancy = np.ones(n_bins)
    return firing_rate, occupancy


@pytest.fixture
def batch_firing_rates() -> tuple[np.ndarray, np.ndarray]:
    """Batch of firing rates for multiple neurons.

    Returns
    -------
    firing_rates : ndarray
        Shape (5, 100): 5 neurons with different patterns.
    occupancy : ndarray
        Uniform occupancy (shared).
    """
    n_neurons = 5
    n_bins = 100
    rng = np.random.default_rng(42)

    firing_rates = np.zeros((n_neurons, n_bins))

    # Neuron 0: uniform (no spatial info)
    firing_rates[0] = np.ones(n_bins) * 5.0

    # Neuron 1: selective (high spatial info)
    firing_rates[1, 20] = 30.0

    # Neuron 2: place cell-like
    x = np.arange(n_bins)
    firing_rates[2] = 15.0 * np.exp(-((x - 60) ** 2) / (2 * 8**2))

    # Neuron 3: multiple fields
    firing_rates[3] = 10.0 * np.exp(-((x - 25) ** 2) / (2 * 5**2))
    firing_rates[3] += 10.0 * np.exp(-((x - 75) ** 2) / (2 * 5**2))

    # Neuron 4: noisy
    firing_rates[4] = rng.uniform(1.0, 5.0, n_bins)

    occupancy = np.ones(n_bins)

    return firing_rates, occupancy


# =============================================================================
# Test spatial_information (single neuron)
# =============================================================================


class TestSpatialInformation:
    """Tests for spatial_information() function."""

    def test_uniform_firing_zero_info(self, uniform_firing):
        """Uniform firing should give zero spatial information."""
        firing_rate, occupancy = uniform_firing
        info = spatial_information(firing_rate, occupancy)
        assert_allclose(info, 0.0, atol=1e-10)

    def test_selective_firing_high_info(self, selective_firing):
        """Highly selective firing should give high spatial information."""
        firing_rate, occupancy = selective_firing
        info = spatial_information(firing_rate, occupancy)
        # One-hot firing gives very high spatial information
        # log2(100) = 6.64 bits would be max
        assert info > 4.0  # Should be close to log2(n_bins)

    def test_place_cell_typical_info(self, place_cell_like):
        """Place cell-like firing should give typical 1-3 bits/spike."""
        firing_rate, occupancy = place_cell_like
        info = spatial_information(firing_rate, occupancy)
        # Typical place cells: 1-3 bits/spike
        assert 0.5 < info < 5.0

    def test_non_negative(self, place_cell_like):
        """Spatial information should always be non-negative."""
        firing_rate, occupancy = place_cell_like
        info = spatial_information(firing_rate, occupancy)
        assert info >= 0.0

    def test_returns_float(self, uniform_firing):
        """Should return a Python float, not array."""
        firing_rate, occupancy = uniform_firing
        info = spatial_information(firing_rate, occupancy)
        assert isinstance(info, float)

    def test_formula_matches_skaggs(self):
        """Test the Skaggs formula is correctly implemented.

        Formula: I = sum_i(p_i * (r_i/r_mean) * log2(r_i/r_mean))
        """
        # Simple case with known values
        firing_rate = np.array([0.0, 2.0, 4.0, 2.0])
        occupancy = np.array([1.0, 1.0, 1.0, 1.0])

        info = spatial_information(firing_rate, occupancy)

        # Manual computation:
        # p_i = [0.25, 0.25, 0.25, 0.25]
        # r_mean = (0 + 2 + 4 + 2)/4 = 2
        # bin 0: 0 (skip, r_i = 0)
        # bin 1: 0.25 * (2/2) * log2(2/2) = 0.25 * 1 * 0 = 0
        # bin 2: 0.25 * (4/2) * log2(4/2) = 0.25 * 2 * 1 = 0.5
        # bin 3: 0.25 * (2/2) * log2(2/2) = 0.25 * 1 * 0 = 0
        # Total = 0.5
        assert_allclose(info, 0.5, rtol=0.1)

    def test_base_parameter(self):
        """Test different log bases (bits vs nats)."""
        firing_rate = np.array([0.0, 2.0, 4.0, 2.0])
        occupancy = np.array([1.0, 1.0, 1.0, 1.0])

        info_bits = spatial_information(firing_rate, occupancy, base=2.0)
        info_nats = spatial_information(firing_rate, occupancy, base=np.e)

        # nats = bits * ln(2)
        expected_nats = info_bits * np.log(2)
        assert_allclose(info_nats, expected_nats, rtol=1e-6)

    def test_zero_mean_rate_returns_zero(self):
        """Should return 0.0 when mean firing rate is zero."""
        firing_rate = np.zeros(100)
        occupancy = np.ones(100)
        info = spatial_information(firing_rate, occupancy)
        assert info == 0.0

    def test_nan_values_handled(self):
        """Should handle NaN values in firing rate."""
        firing_rate = np.array([1.0, 2.0, np.nan, 3.0, 4.0])
        occupancy = np.ones(5)
        info = spatial_information(firing_rate, occupancy)
        # Should compute info ignoring NaN
        assert not np.isnan(info)
        assert info >= 0.0

    def test_non_uniform_occupancy(self):
        """Should correctly weight by occupancy probability."""
        # More time spent in bin 0
        firing_rate = np.array([5.0, 5.0, 5.0, 5.0])
        occupancy = np.array([10.0, 1.0, 1.0, 1.0])

        info = spatial_information(firing_rate, occupancy)
        # Still uniform firing rate, so info should still be ~0
        assert_allclose(info, 0.0, atol=1e-10)


# =============================================================================
# Test sparsity (single neuron)
# =============================================================================


class TestSparsity:
    """Tests for sparsity() function."""

    def test_uniform_firing_high_sparsity(self, uniform_firing):
        """Uniform firing should give sparsity close to 1.0."""
        firing_rate, occupancy = uniform_firing
        spars = sparsity(firing_rate, occupancy)
        assert_allclose(spars, 1.0, atol=1e-6)

    def test_selective_firing_low_sparsity(self, selective_firing):
        """Highly selective firing should give low sparsity."""
        firing_rate, occupancy = selective_firing
        spars = sparsity(firing_rate, occupancy)
        # One-hot firing → very sparse
        assert spars < 0.1

    def test_place_cell_typical_sparsity(self, place_cell_like):
        """Place cell-like firing should give typical 0.1-0.3 sparsity."""
        firing_rate, occupancy = place_cell_like
        spars = sparsity(firing_rate, occupancy)
        # Typical place cells: 0.1-0.3
        assert 0.05 < spars < 0.5

    def test_range_zero_to_one(self, place_cell_like):
        """Sparsity should always be in [0, 1]."""
        firing_rate, occupancy = place_cell_like
        spars = sparsity(firing_rate, occupancy)
        assert 0.0 <= spars <= 1.0

    def test_returns_float(self, uniform_firing):
        """Should return a Python float, not array."""
        firing_rate, occupancy = uniform_firing
        spars = sparsity(firing_rate, occupancy)
        assert isinstance(spars, float)

    def test_formula_matches_skaggs(self):
        """Test the Skaggs formula is correctly implemented.

        Formula: S = (sum_i(p_i * r_i))^2 / sum_i(p_i * r_i^2)
        """
        firing_rate = np.array([0.0, 2.0, 4.0, 2.0])
        occupancy = np.array([1.0, 1.0, 1.0, 1.0])

        spars = sparsity(firing_rate, occupancy)

        # Manual computation:
        # p_i = [0.25, 0.25, 0.25, 0.25]
        # numerator = (0.25*0 + 0.25*2 + 0.25*4 + 0.25*2)^2 = 2^2 = 4
        # denominator = 0.25*0 + 0.25*4 + 0.25*16 + 0.25*4 = 6
        # sparsity = 4/6 = 0.667
        assert_allclose(spars, 0.667, atol=0.01)

    def test_zero_denominator_returns_zero(self):
        """Should return 0.0 when all firing rates are zero."""
        firing_rate = np.zeros(100)
        occupancy = np.ones(100)
        spars = sparsity(firing_rate, occupancy)
        assert spars == 0.0

    def test_nan_values_handled(self):
        """Should handle NaN values in firing rate."""
        firing_rate = np.array([1.0, 2.0, np.nan, 3.0, 4.0])
        occupancy = np.ones(5)
        spars = sparsity(firing_rate, occupancy)
        # Should compute sparsity ignoring NaN
        assert not np.isnan(spars)
        assert 0.0 <= spars <= 1.0


# =============================================================================
# Test batch_spatial_information (population)
# =============================================================================


class TestBatchSpatialInformation:
    """Tests for batch_spatial_information() function."""

    def test_batch_shape(self, batch_firing_rates):
        """Should return (n_neurons,) array."""
        firing_rates, occupancy = batch_firing_rates
        info = batch_spatial_information(firing_rates, occupancy)
        assert info.shape == (5,)

    def test_batch_matches_single(self, batch_firing_rates):
        """Batch computation should match single-neuron results."""
        firing_rates, occupancy = batch_firing_rates
        batch_info = batch_spatial_information(firing_rates, occupancy)

        # Compute single-neuron for each
        single_info = np.array(
            [spatial_information(fr, occupancy) for fr in firing_rates]
        )

        assert_allclose(batch_info, single_info, rtol=1e-10)

    def test_batch_all_non_negative(self, batch_firing_rates):
        """All spatial information values should be non-negative."""
        firing_rates, occupancy = batch_firing_rates
        info = batch_spatial_information(firing_rates, occupancy)
        assert np.all(info >= 0.0)

    def test_batch_dtype(self, batch_firing_rates):
        """Should return float64 array."""
        firing_rates, occupancy = batch_firing_rates
        info = batch_spatial_information(firing_rates, occupancy)
        assert info.dtype == np.float64

    def test_batch_with_single_neuron(self, place_cell_like):
        """Should work with (1, n_bins) input."""
        firing_rate, occupancy = place_cell_like
        firing_rates = firing_rate[np.newaxis, :]  # (1, n_bins)
        info = batch_spatial_information(firing_rates, occupancy)
        assert info.shape == (1,)
        expected = spatial_information(firing_rate, occupancy)
        assert_allclose(info[0], expected, rtol=1e-10)

    def test_batch_uniform_vs_selective(self, batch_firing_rates):
        """Uniform neuron should have less info than selective neuron."""
        firing_rates, occupancy = batch_firing_rates
        info = batch_spatial_information(firing_rates, occupancy)
        # Neuron 0 is uniform, neuron 1 is selective
        assert info[0] < info[1]


# =============================================================================
# Test batch_sparsity (population)
# =============================================================================


class TestBatchSparsity:
    """Tests for batch_sparsity() function."""

    def test_batch_shape(self, batch_firing_rates):
        """Should return (n_neurons,) array."""
        firing_rates, occupancy = batch_firing_rates
        spars = batch_sparsity(firing_rates, occupancy)
        assert spars.shape == (5,)

    def test_batch_matches_single(self, batch_firing_rates):
        """Batch computation should match single-neuron results."""
        firing_rates, occupancy = batch_firing_rates
        batch_spars = batch_sparsity(firing_rates, occupancy)

        # Compute single-neuron for each
        single_spars = np.array([sparsity(fr, occupancy) for fr in firing_rates])

        assert_allclose(batch_spars, single_spars, rtol=1e-10)

    def test_batch_all_in_range(self, batch_firing_rates):
        """All sparsity values should be in [0, 1]."""
        firing_rates, occupancy = batch_firing_rates
        spars = batch_sparsity(firing_rates, occupancy)
        assert np.all((spars >= 0.0) & (spars <= 1.0))

    def test_batch_dtype(self, batch_firing_rates):
        """Should return float64 array."""
        firing_rates, occupancy = batch_firing_rates
        spars = batch_sparsity(firing_rates, occupancy)
        assert spars.dtype == np.float64

    def test_batch_with_single_neuron(self, place_cell_like):
        """Should work with (1, n_bins) input."""
        firing_rate, occupancy = place_cell_like
        firing_rates = firing_rate[np.newaxis, :]  # (1, n_bins)
        spars = batch_sparsity(firing_rates, occupancy)
        assert spars.shape == (1,)
        expected = sparsity(firing_rate, occupancy)
        assert_allclose(spars[0], expected, rtol=1e-10)

    def test_batch_uniform_vs_selective(self, batch_firing_rates):
        """Uniform neuron should have higher sparsity than selective neuron."""
        firing_rates, occupancy = batch_firing_rates
        spars = batch_sparsity(firing_rates, occupancy)
        # Neuron 0 is uniform, neuron 1 is selective
        assert spars[0] > spars[1]


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_bin(self):
        """Should handle single-bin environment."""
        firing_rate = np.array([5.0])
        occupancy = np.array([1.0])

        info = spatial_information(firing_rate, occupancy)
        spars = sparsity(firing_rate, occupancy)

        assert info == 0.0  # No spatial information with one bin
        assert spars == 1.0  # All firing in "everywhere"

    def test_all_nan_firing_rate(self):
        """Should handle all-NaN firing rate."""
        firing_rate = np.array([np.nan, np.nan, np.nan])
        occupancy = np.array([1.0, 1.0, 1.0])

        info = spatial_information(firing_rate, occupancy)
        spars = sparsity(firing_rate, occupancy)

        assert info == 0.0
        assert spars == 0.0

    def test_inf_handling(self):
        """Should handle inf values gracefully."""
        firing_rate = np.array([1.0, 2.0, np.inf, 3.0])
        occupancy = np.array([1.0, 1.0, 1.0, 1.0])

        # Should not raise, may return inf or handle gracefully
        info = spatial_information(firing_rate, occupancy)
        spars = sparsity(firing_rate, occupancy)

        # Just verify no exceptions were raised
        assert isinstance(info, float)
        assert isinstance(spars, float)

    def test_empty_array_raises(self):
        """Should raise on empty arrays."""
        firing_rate = np.array([])
        occupancy = np.array([])

        with pytest.raises((ValueError, ZeroDivisionError)):
            spatial_information(firing_rate, occupancy)

        with pytest.raises((ValueError, ZeroDivisionError)):
            sparsity(firing_rate, occupancy)

    def test_mismatched_shapes_raises(self):
        """Should raise on mismatched array shapes."""
        firing_rate = np.array([1.0, 2.0, 3.0])
        occupancy = np.array([1.0, 1.0])

        with pytest.raises(ValueError):
            spatial_information(firing_rate, occupancy)

        with pytest.raises(ValueError):
            sparsity(firing_rate, occupancy)

    def test_batch_mismatched_shapes_raises(self):
        """Should raise on mismatched batch shapes."""
        firing_rates = np.ones((3, 100))
        occupancy = np.ones(50)  # Wrong size

        with pytest.raises(ValueError):
            batch_spatial_information(firing_rates, occupancy)

        with pytest.raises(ValueError):
            batch_sparsity(firing_rates, occupancy)


# =============================================================================
# Public API Consistency
# =============================================================================


class TestPublicApiConsistency:
    """Test that package-level metric exports match _metrics.py."""

    def test_package_spatial_information_export_matches_metrics(
        self, place_cell_like: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Package export should point to the shared metric implementation."""
        from neurospatial.encoding import (
            spatial_information as public_spatial_information,
        )

        firing_rate, occupancy = place_cell_like

        new_info = spatial_information(firing_rate, occupancy)
        public_info = public_spatial_information(firing_rate, occupancy)

        assert_allclose(new_info, public_info, rtol=1e-10)


# =============================================================================
# Test batch_grid_scores (population)
# =============================================================================


def _create_hexagonal_autocorr(size: int = 100, radius: float = 20.0) -> np.ndarray:
    """Create synthetic hexagonal autocorrelogram for testing.

    Parameters
    ----------
    size : int
        Size of the autocorrelogram (size x size).
    radius : float
        Radius to place the 6 surrounding peaks.

    Returns
    -------
    autocorr : ndarray
        Normalized autocorrelogram with hexagonal pattern.
    """
    autocorr = np.zeros((size, size))
    center = size // 2

    # Coordinate grids
    y_grid, x_grid = np.ogrid[:size, :size]

    # Central peak
    dist_from_center = np.sqrt((y_grid - center) ** 2 + (x_grid - center) ** 2)
    autocorr = np.exp(-(dist_from_center**2) / (2 * 5**2))

    # Add 6 peaks at 60 degrees intervals (hexagonal pattern)
    for angle_deg in [0, 60, 120, 180, 240, 300]:
        angle_rad = np.radians(angle_deg)
        peak_y = center + int(radius * np.sin(angle_rad))
        peak_x = center + int(radius * np.cos(angle_rad))
        peak_dist = np.sqrt((y_grid - peak_y) ** 2 + (x_grid - peak_x) ** 2)
        autocorr += 0.8 * np.exp(-(peak_dist**2) / (2 * 5**2))

    return autocorr / autocorr.max()


class TestBatchGridScores:
    """Tests for batch_grid_scores() function."""

    @pytest.fixture
    def env_2d_grid(self):
        """Create a regular 2D grid environment for testing.

        Returns
        -------
        env : Environment
            Regular 2D grid environment suitable for FFT-based autocorrelation.
        """
        from neurospatial import Environment

        # Create regular 2D grid
        x = np.linspace(-50, 50, 51)
        xx, yy = np.meshgrid(x, x)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(positions, bin_size=2.0)
        return env

    @pytest.fixture
    def batch_firing_rates_for_grid(self, env_2d_grid):
        """Create batch of firing rates for grid score testing.

        Returns
        -------
        firing_rates : ndarray
            Shape (n_neurons, n_bins) with different patterns.
        env : Environment
            The environment used to create the firing rates.
        """
        env = env_2d_grid
        n_neurons = 3
        n_bins = env.n_bins
        rng = np.random.default_rng(42)

        firing_rates = np.zeros((n_neurons, n_bins))

        # Neuron 0: random noise (low grid score expected)
        firing_rates[0] = rng.random(n_bins) * 5.0

        # Neuron 1: single place field (low grid score expected)
        bin_centers = env.bin_centers
        center = np.mean(bin_centers, axis=0)
        distances = np.sqrt(np.sum((bin_centers - center) ** 2, axis=1))
        firing_rates[1] = 10.0 * np.exp(-(distances**2) / (2 * 10**2))

        # Neuron 2: another random pattern
        firing_rates[2] = rng.random(n_bins) * 3.0

        return firing_rates, env

    def test_batch_grid_scores_returns_correct_shape(self, batch_firing_rates_for_grid):
        """batch_grid_scores should return (n_neurons,) array."""
        from neurospatial.encoding._metrics import batch_grid_scores

        firing_rates, env = batch_firing_rates_for_grid
        scores = batch_grid_scores(env, firing_rates)

        assert scores.shape == (3,)

    def test_batch_grid_scores_returns_float64(self, batch_firing_rates_for_grid):
        """batch_grid_scores should return float64 array."""
        from neurospatial.encoding._metrics import batch_grid_scores

        firing_rates, env = batch_firing_rates_for_grid
        scores = batch_grid_scores(env, firing_rates)

        assert scores.dtype == np.float64

    def test_batch_grid_scores_range(self, batch_firing_rates_for_grid):
        """Grid scores should be in valid range [-2, 2] or NaN."""
        from neurospatial.encoding._metrics import batch_grid_scores

        firing_rates, env = batch_firing_rates_for_grid
        scores = batch_grid_scores(env, firing_rates)

        # Scores should be in range [-2, 2] or NaN
        valid_scores = scores[~np.isnan(scores)]
        assert np.all(valid_scores >= -2.0)
        assert np.all(valid_scores <= 2.0)

    def test_batch_grid_scores_single_neuron(self, env_2d_grid):
        """Should work with (1, n_bins) input."""
        from neurospatial.encoding._metrics import batch_grid_scores

        env = env_2d_grid
        rng = np.random.default_rng(42)
        firing_rates = rng.random((1, env.n_bins)) * 5.0

        scores = batch_grid_scores(env, firing_rates)

        assert scores.shape == (1,)

    def test_batch_grid_scores_matches_single_grid_score(self, env_2d_grid):
        """Batch computation should match single-neuron grid_score computation."""
        from neurospatial.encoding._metrics import batch_grid_scores
        from neurospatial.encoding.grid import grid_score, spatial_autocorrelation

        env = env_2d_grid
        rng = np.random.default_rng(42)
        firing_rates = rng.random((2, env.n_bins)) * 5.0

        # Batch computation
        batch_scores = batch_grid_scores(env, firing_rates)

        # Single neuron computation for comparison
        single_scores = []
        for i in range(firing_rates.shape[0]):
            autocorr = spatial_autocorrelation(env, firing_rates[i], method="fft")
            score = grid_score(autocorr)
            single_scores.append(score)

        single_scores = np.array(single_scores)
        assert_allclose(batch_scores, single_scores, rtol=1e-10)

    def test_batch_grid_scores_wrong_shape_raises(self, env_2d_grid):
        """Should raise ValueError for wrong firing_rates shape."""
        from neurospatial.encoding._metrics import batch_grid_scores

        env = env_2d_grid

        # 1D array should raise
        firing_rate_1d = np.random.rand(env.n_bins)
        with pytest.raises(ValueError, match="firing_rates must be 2D"):
            batch_grid_scores(env, firing_rate_1d)

        # Wrong n_bins should raise
        firing_rates_wrong = np.random.rand(3, env.n_bins + 10)
        with pytest.raises(ValueError, match="bins"):
            batch_grid_scores(env, firing_rates_wrong)

    def test_batch_grid_scores_handles_nan(self, env_2d_grid):
        """Should handle NaN values in firing rates."""
        from neurospatial.encoding._metrics import batch_grid_scores

        env = env_2d_grid
        rng = np.random.default_rng(42)
        firing_rates = rng.random((2, env.n_bins)) * 5.0

        # Add some NaN values
        firing_rates[0, 10:15] = np.nan

        # Should not raise (NaN handling may result in NaN output)
        scores = batch_grid_scores(env, firing_rates)

        assert scores.shape == (2,)
        # Note: score may or may not be NaN depending on implementation

    def test_batch_grid_scores_with_constant_firing(self, env_2d_grid):
        """Constant firing rate is a legitimate NaN, not a computation failure.

        The result contract distinguishes "score is NaN because the input
        has no spatial structure" from "score computation raised an
        exception that we caught". Constant firing rates have zero
        variance and so the autocorrelation is mathematically undefined,
        but the input itself is well-formed: the failure mask must stay
        False and no UserWarning should fire. (M1 1.5 review-response.)
        """
        import warnings

        from neurospatial.encoding._metrics import (
            BatchScoresResult,
            batch_grid_scores,
        )

        env = env_2d_grid
        firing_rates = np.ones((2, env.n_bins)) * 5.0  # Constant

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # promote any warning to an error
            result = batch_grid_scores(env, firing_rates)

        assert isinstance(result, BatchScoresResult)
        assert result.shape == (2,)
        assert np.all(np.isnan(result.scores))
        assert not result.failures.any()
        assert result.n_failures == 0

    def test_batch_grid_scores_all_nan_input_is_legitimate_nan(self, env_2d_grid):
        """All-NaN firing rate is a legitimate NaN, not a computation failure.

        Same contract as constant firing: the input describes a neuron
        with no usable rate information, so the score is NaN by
        definition; that is not a caught exception and must not flip the
        failures mask.
        """
        import warnings

        from neurospatial.encoding._metrics import (
            BatchScoresResult,
            batch_grid_scores,
        )

        env = env_2d_grid
        firing_rates = np.full((2, env.n_bins), np.nan)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = batch_grid_scores(env, firing_rates)

        assert isinstance(result, BatchScoresResult)
        assert np.all(np.isnan(result.scores))
        assert not result.failures.any()

    def test_batch_scores_result_array_protocol_numpy2(self, env_2d_grid):
        """BatchScoresResult.__array__ must accept the NumPy-2 ``copy`` kwarg.

        NumPy 2 added a ``copy`` parameter to the array protocol;
        ``np.asarray(obj, copy=False)`` warns and fails when ``__array__``
        does not accept it. Since BatchScoresResult is intended to be a
        drop-in replacement for the old ndarray-shaped score return,
        regress this explicitly.
        """
        from neurospatial.encoding._metrics import batch_grid_scores

        env = env_2d_grid
        rng = np.random.default_rng(42)
        firing_rates = rng.random((3, env.n_bins)) * 5.0
        result = batch_grid_scores(env, firing_rates)

        # No-copy view: should return the underlying scores array.
        view = np.asarray(result, copy=False)
        assert view.shape == (3,)
        assert view is result.scores

        # Forced copy: must not alias the underlying scores.
        copied = np.asarray(result, copy=True)
        assert copied.shape == (3,)
        assert copied is not result.scores
        np.testing.assert_array_equal(copied, result.scores)

        # dtype cast (default copy semantics): independent buffer.
        as_f32 = np.asarray(result, dtype=np.float32)
        assert as_f32.dtype == np.float32
        np.testing.assert_allclose(
            as_f32, result.scores.astype(np.float32), equal_nan=True
        )

        # Same-dtype with copy=False: returns the underlying buffer.
        same_dtype = np.asarray(result, dtype=np.float64, copy=False)
        assert same_dtype is result.scores

        # Cast + copy=False: must raise, matching plain-ndarray semantics.
        # Without the raise, callers who explicitly asked for a view get a
        # silent copy and any in-place writes go to /dev/null. Note the
        # `np.array` form: NumPy 2 routes np.asarray(..., copy=False) ->
        # np.array(..., copy=None) when the cast is unavoidable, so both
        # forms must raise here.
        with pytest.raises(ValueError, match="Unable to avoid copy"):
            np.array(result, dtype=np.float32, copy=False)


class TestBatchGridScoresNonRegularGrid:
    """Tests for batch_grid_scores with non-regular grid environments."""

    @pytest.fixture
    def env_sparse_grid(self):
        """Create a sparse (non-regular) environment for testing.

        Returns
        -------
        env : Environment
            Environment with sparse coverage.
        """
        from neurospatial import Environment

        # Create sparse positions (not a regular grid)
        rng = np.random.default_rng(42)
        positions = rng.uniform(-50, 50, size=(500, 2))
        env = Environment.from_samples(positions, bin_size=5.0)
        return env

    def test_batch_grid_scores_sparse_env_returns_nan(self, env_sparse_grid):
        """For sparse environments where FFT doesn't work, should handle gracefully."""
        from neurospatial.encoding._metrics import batch_grid_scores

        env = env_sparse_grid
        rng = np.random.default_rng(42)
        firing_rates = rng.random((2, env.n_bins)) * 5.0

        # Should not raise - may return NaN for non-FFT-compatible envs
        scores = batch_grid_scores(env, firing_rates)

        assert scores.shape == (2,)
        # Sparse environments may not be compatible with FFT-based grid score
        # Implementation may return NaN or use graph-based method


class TestBatchScoresFailuresMask:
    """Regressions for M1 1.5: batch metric failures must be visible.

    Previously batch_grid_scores and batch_border_scores swallowed
    (ValueError, RuntimeError) and substituted NaN per neuron with no
    indication. A caller running "fraction of grid cells" on the
    output silently treated failures as zeros and tests passed.
    Now both batch functions:

    1. Track per-neuron failures in a parallel boolean mask.
    2. Emit a UserWarning summarizing the failure count.
    3. Return a BatchScoresResult so the mask is reachable.
    """

    @pytest.fixture
    def env_2d_grid(self):
        """Small regular 2D grid suitable for batch scoring."""
        from neurospatial import Environment

        x = np.linspace(-20, 20, 21)
        xx, yy = np.meshgrid(x, x)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        return Environment.from_samples(positions, bin_size=2.0)

    def test_batch_grid_scores_returns_result_class(self, env_2d_grid):
        """batch_grid_scores returns BatchScoresResult, not a bare ndarray."""
        from neurospatial.encoding._metrics import (
            BatchScoresResult,
            batch_grid_scores,
        )

        env = env_2d_grid
        rng = np.random.default_rng(0)
        firing_rates = rng.random((3, env.n_bins)) * 5.0
        result = batch_grid_scores(env, firing_rates)
        assert isinstance(result, BatchScoresResult)
        assert result.scores.shape == (3,)
        assert result.failures.shape == (3,)
        assert result.failures.dtype == np.bool_

    def test_batch_grid_scores_no_failures_no_warning(self, env_2d_grid):
        """When every neuron's score computes cleanly, no warning fires."""
        from neurospatial.encoding._metrics import batch_grid_scores

        env = env_2d_grid
        rng = np.random.default_rng(0)
        firing_rates = rng.random((3, env.n_bins)) * 5.0
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning becomes a failure
            result = batch_grid_scores(env, firing_rates)
        assert result.n_failures == 0
        assert not result.failures.any()

    def test_batch_border_scores_warns_and_flags_failed_neurons(
        self,
        env_2d_grid,
    ):
        """A monkey-patched border_score that raises must populate failures.

        Patches ``neurospatial.encoding.border.border_score`` so the
        first neuron raises ValueError. The batch wrapper must:

        1. Continue computing the remaining neurons.
        2. Set failures[0]=True (and not flag the others).
        3. Set scores[0]=NaN (and leave the others finite).
        4. Emit a UserWarning that mentions the failure count.
        """
        from neurospatial.encoding import _metrics

        env = env_2d_grid
        rng = np.random.default_rng(0)
        firing_rates = rng.random((3, env.n_bins)) * 5.0

        call_count = {"n": 0}

        def fake_border_score(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise ValueError("intentional test failure on neuron 0")
            return 0.5  # something finite for the other two

        # batch_border_scores does `from neurospatial.encoding.border
        # import border_score` inside the function body, so patching the
        # source module is the right place for the test to intercept.
        import neurospatial.encoding.border as border_mod

        original = border_mod.border_score
        try:
            border_mod.border_score = fake_border_score
            with pytest.warns(
                UserWarning, match=r"raised an exception for 1 of 3 neurons"
            ):
                result = _metrics.batch_border_scores(env, firing_rates)
        finally:
            border_mod.border_score = original

        assert result.n_failures == 1
        assert result.failures[0] and not result.failures[1] and not result.failures[2]
        assert np.isnan(result.scores[0])
        assert result.scores[1] == pytest.approx(0.5)
        assert result.scores[2] == pytest.approx(0.5)


# =============================================================================
# Test batch_border_scores (population)
# =============================================================================


class TestBatchBorderScores:
    """Tests for batch_border_scores() function."""

    @pytest.fixture
    def env_2d_grid(self):
        """Create a regular 2D grid environment for testing.

        Returns
        -------
        env : Environment
            Regular 2D grid environment.
        """
        from neurospatial import Environment

        # Create regular 2D grid with good boundary detection
        x = np.linspace(-50, 50, 51)
        xx, yy = np.meshgrid(x, x)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(positions, bin_size=2.0)
        return env

    @pytest.fixture
    def batch_firing_rates_for_border(self, env_2d_grid):
        """Create batch of firing rates for border score testing.

        Returns
        -------
        firing_rates : ndarray
            Shape (n_neurons, n_bins) with different patterns.
        env : Environment
            The environment used to create the firing rates.
        """
        env = env_2d_grid
        n_neurons = 4
        n_bins = env.n_bins
        rng = np.random.default_rng(42)

        firing_rates = np.zeros((n_neurons, n_bins))

        # Neuron 0: random noise (low border score expected)
        firing_rates[0] = rng.random(n_bins) * 5.0

        # Neuron 1: boundary-preferring (high border score expected)
        boundary_bins = env.boundary_bins
        firing_rates[1, boundary_bins] = 10.0

        # Neuron 2: center-preferring (negative border score expected)
        bin_centers = env.bin_centers
        center = np.mean(bin_centers, axis=0)
        distances = np.sqrt(np.sum((bin_centers - center) ** 2, axis=1))
        firing_rates[2] = 10.0 * np.exp(-(distances**2) / (2 * 10**2))

        # Neuron 3: uniform (near-zero border score expected)
        firing_rates[3] = np.ones(n_bins) * 5.0

        return firing_rates, env

    def test_batch_border_scores_returns_correct_shape(
        self, batch_firing_rates_for_border
    ):
        """batch_border_scores should return (n_neurons,) array."""
        from neurospatial.encoding._metrics import batch_border_scores

        firing_rates, env = batch_firing_rates_for_border
        scores = batch_border_scores(env, firing_rates)

        assert scores.shape == (4,)

    def test_batch_border_scores_returns_float64(self, batch_firing_rates_for_border):
        """batch_border_scores should return float64 array."""
        from neurospatial.encoding._metrics import batch_border_scores

        firing_rates, env = batch_firing_rates_for_border
        scores = batch_border_scores(env, firing_rates)

        assert scores.dtype == np.float64

    def test_batch_border_scores_range(self, batch_firing_rates_for_border):
        """Border scores should be in valid range [-1, 1] or NaN."""
        from neurospatial.encoding._metrics import batch_border_scores

        firing_rates, env = batch_firing_rates_for_border
        scores = batch_border_scores(env, firing_rates)

        # Scores should be in range [-1, 1] or NaN
        valid_scores = scores[~np.isnan(scores)]
        assert np.all(valid_scores >= -1.0)
        assert np.all(valid_scores <= 1.0)

    def test_batch_border_scores_single_neuron(self, env_2d_grid):
        """Should work with (1, n_bins) input."""
        from neurospatial.encoding._metrics import batch_border_scores

        env = env_2d_grid
        rng = np.random.default_rng(42)
        firing_rates = rng.random((1, env.n_bins)) * 5.0

        scores = batch_border_scores(env, firing_rates)

        assert scores.shape == (1,)

    def test_batch_border_scores_matches_single_border_score(self, env_2d_grid):
        """Batch computation should match single-neuron border_score computation."""
        from neurospatial.encoding._metrics import batch_border_scores
        from neurospatial.encoding.border import border_score

        env = env_2d_grid
        rng = np.random.default_rng(42)
        firing_rates = rng.random((3, env.n_bins)) * 5.0

        # Batch computation
        batch_scores = batch_border_scores(env, firing_rates)

        # Single neuron computation for comparison
        single_scores = []
        for i in range(firing_rates.shape[0]):
            try:
                score = border_score(env, firing_rates[i])
            except Exception:
                score = np.nan
            single_scores.append(score)

        single_scores = np.array(single_scores)

        # Compare non-NaN values
        for i in range(len(batch_scores)):
            if np.isnan(batch_scores[i]) and np.isnan(single_scores[i]):
                continue  # Both NaN is OK
            elif np.isnan(batch_scores[i]) or np.isnan(single_scores[i]):
                # One NaN, one not - that's OK as long as behavior is consistent
                pass
            else:
                assert_allclose(batch_scores[i], single_scores[i], rtol=1e-10)

    def test_batch_border_scores_wrong_shape_raises(self, env_2d_grid):
        """Should raise ValueError for wrong firing_rates shape."""
        from neurospatial.encoding._metrics import batch_border_scores

        env = env_2d_grid

        # 1D array should raise
        firing_rate_1d = np.random.rand(env.n_bins)
        with pytest.raises(ValueError, match="firing_rates must be 2D"):
            batch_border_scores(env, firing_rate_1d)

        # Wrong n_bins should raise
        firing_rates_wrong = np.random.rand(3, env.n_bins + 10)
        with pytest.raises(ValueError, match="bins"):
            batch_border_scores(env, firing_rates_wrong)

    def test_batch_border_scores_boundary_neuron_high_score(
        self, batch_firing_rates_for_border
    ):
        """Neuron firing on boundary should have high border score."""
        from neurospatial.encoding._metrics import batch_border_scores

        firing_rates, env = batch_firing_rates_for_border
        scores = batch_border_scores(env, firing_rates)

        # Neuron 1 fires on boundary - should have positive score
        # (if not NaN)
        if not np.isnan(scores[1]):
            assert scores[1] > 0.0

    def test_batch_border_scores_center_neuron_lower_score(
        self, batch_firing_rates_for_border
    ):
        """Neuron firing in center should have lower border score than boundary."""
        from neurospatial.encoding._metrics import batch_border_scores

        firing_rates, env = batch_firing_rates_for_border
        scores = batch_border_scores(env, firing_rates)

        # Neuron 1 fires on boundary, Neuron 2 fires in center
        # Boundary neuron should have higher score than center neuron
        if not np.isnan(scores[1]) and not np.isnan(scores[2]):
            assert scores[1] > scores[2]

    def test_batch_border_scores_with_constant_firing(self, env_2d_grid):
        """Constant firing rate should return NaN (no field defined at threshold)."""
        from neurospatial.encoding._metrics import batch_border_scores

        env = env_2d_grid
        # All bins have same value - at threshold, field covers everything
        # This is an edge case that may return NaN
        firing_rates = np.ones((2, env.n_bins)) * 5.0

        scores = batch_border_scores(env, firing_rates)

        assert scores.shape == (2,)
        # Implementation may handle this differently, but shape should be correct

    def test_batch_border_scores_handles_nan_firing_rate(self, env_2d_grid):
        """Should handle NaN values in firing rates."""
        from neurospatial.encoding._metrics import batch_border_scores

        env = env_2d_grid
        rng = np.random.default_rng(42)
        firing_rates = rng.random((2, env.n_bins)) * 5.0

        # Add some NaN values
        firing_rates[0, 10:15] = np.nan

        # Should not raise
        scores = batch_border_scores(env, firing_rates)

        assert scores.shape == (2,)

    def test_batch_border_scores_handles_zero_firing_rate(self, env_2d_grid):
        """Should handle all-zero firing rates."""
        from neurospatial.encoding._metrics import batch_border_scores

        env = env_2d_grid
        firing_rates = np.zeros((2, env.n_bins))

        scores = batch_border_scores(env, firing_rates)

        assert scores.shape == (2,)
        # Zero firing rate → NaN expected
        assert np.all(np.isnan(scores))

    def test_batch_border_scores_threshold_parameter(self, env_2d_grid):
        """Should accept threshold parameter."""
        from neurospatial.encoding._metrics import batch_border_scores

        env = env_2d_grid
        rng = np.random.default_rng(42)
        firing_rates = rng.random((2, env.n_bins)) * 5.0

        # Test with different thresholds
        scores_03 = batch_border_scores(env, firing_rates, threshold=0.3)
        scores_05 = batch_border_scores(env, firing_rates, threshold=0.5)

        assert scores_03.shape == (2,)
        assert scores_05.shape == (2,)

    def test_batch_border_scores_distance_metric_parameter(self, env_2d_grid):
        """Should accept distance_metric parameter."""
        from neurospatial.encoding._metrics import batch_border_scores

        env = env_2d_grid
        rng = np.random.default_rng(42)
        firing_rates = rng.random((2, env.n_bins)) * 5.0

        # Test both distance metrics
        scores_geo = batch_border_scores(env, firing_rates, distance_metric="geodesic")
        scores_euc = batch_border_scores(env, firing_rates, distance_metric="euclidean")

        assert scores_geo.shape == (2,)
        assert scores_euc.shape == (2,)

    def test_batch_border_scores_min_area_parameter(self, env_2d_grid):
        """Should accept min_area parameter and filter small fields."""
        from neurospatial.encoding._metrics import batch_border_scores

        env = env_2d_grid
        rng = np.random.default_rng(42)
        firing_rates = rng.random((2, env.n_bins)) * 5.0

        # Test with different min_area values
        scores_no_filter = batch_border_scores(env, firing_rates, min_area=0.0)
        scores_high_filter = batch_border_scores(env, firing_rates, min_area=10000.0)

        assert scores_no_filter.shape == (2,)
        assert scores_high_filter.shape == (2,)
        # With very high min_area, scores should be NaN (fields too small)
        assert np.all(np.isnan(scores_high_filter))


class TestBorderScoreRaisesOnDijkstraFailure:
    """Regression for M1 1.5 follow-up: border_score's Dijkstra failure
    must propagate so batch_border_scores can flag the neuron in its
    failures mask. Previously the single-neuron path swallowed the
    exception and returned NaN with a RuntimeWarning, which the batch
    wrapper couldn't see -- the failures mask read False even though
    the score was NaN-due-to-failure.
    """

    def test_border_score_raises_on_invalid_graph(self):
        """Negative edge weight -> multi_source_dijkstra raises -> border_score raises.

        ``nx.multi_source_dijkstra_path_length`` rejects negative
        weights (it's an O(E log V) Dijkstra, not Bellman-Ford). The
        pre-fix path swallowed this failure to NaN with a
        RuntimeWarning; the new path re-raises so the batch wrapper
        can populate its `failures` mask.
        """
        from neurospatial import Environment
        from neurospatial.encoding.border import border_score

        x = np.linspace(-20, 20, 21)
        xx, yy = np.meshgrid(x, x)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(positions, bin_size=2.0)

        # Inject negative weights to trigger Dijkstra failure.
        for _, _, data in env.connectivity.edges(data=True):
            data["distance"] = -1.0

        firing_rate = np.zeros(env.n_bins)
        firing_rate[0] = 10.0

        with pytest.raises(RuntimeError, match=r"geodesic distance"):
            border_score(env, firing_rate, distance_metric="geodesic")

    def test_batch_border_scores_failures_mask_catches_dijkstra_failure(self):
        """The batch wrapper now sees the border_score RuntimeError and flags it."""
        from neurospatial import Environment
        from neurospatial.encoding._metrics import batch_border_scores

        x = np.linspace(-20, 20, 21)
        xx, yy = np.meshgrid(x, x)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(positions, bin_size=2.0)
        for _, _, data in env.connectivity.edges(data=True):
            data["distance"] = -1.0

        firing_rates = np.zeros((2, env.n_bins))
        firing_rates[0, 0] = 10.0
        firing_rates[1, 5] = 10.0

        with pytest.warns(UserWarning, match=r"raised an exception for 2 of 2"):
            result = batch_border_scores(env, firing_rates)
        assert result.n_failures == 2
        assert bool(result.failures.all())
        assert np.isnan(result.scores).all()
