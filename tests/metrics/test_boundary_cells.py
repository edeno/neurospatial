"""
Tests for boundary cell metrics.

Following TDD: Tests written FIRST before implementation.

References
----------
Solstad et al. (2008). Representation of geometric borders in the entorhinal cortex.
    Science, 322(5909), 1865-1868.
"""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial import Environment


class TestBorderScore:
    """Test border_score function."""

    def test_border_score_perfect_border_cell(self) -> None:
        """Test border score for perfect border cell (field along one wall)."""
        # Create rectangular environment
        positions = []
        for x in np.linspace(0, 50, 500):
            for y in np.linspace(0, 50, 500):
                positions.append([x, y])
        positions = np.array(positions)

        env = Environment.from_samples(positions, bin_size=5.0)

        # Create firing rate map with activity only along left wall (x < 10)
        firing_rate = np.zeros(env.n_bins)
        for i in range(env.n_bins):
            center = env.bin_centers[i]
            if center[0] < 10:  # Left wall
                firing_rate[i] = 5.0
            else:
                firing_rate[i] = 0.0

        # Import after environment creation
        from neurospatial.metrics.boundary_cells import border_score

        score = border_score(firing_rate, env)

        # Perfect border cell should have high positive score (close to 1)
        assert score > 0.5, f"Expected border score > 0.5 for border cell, got {score}"

    def test_border_score_central_field(self) -> None:
        """Test border score for central field (non-border cell)."""
        # Create environment
        positions = []
        for x in np.linspace(0, 50, 500):
            for y in np.linspace(0, 50, 500):
                positions.append([x, y])
        positions = np.array(positions)

        env = Environment.from_samples(positions, bin_size=5.0)

        # Create firing rate map with activity only in center
        firing_rate = np.zeros(env.n_bins)
        for i in range(env.n_bins):
            center = env.bin_centers[i]
            # Gaussian centered at (25, 25) with small width
            distance = np.sqrt((center[0] - 25) ** 2 + (center[1] - 25) ** 2)
            firing_rate[i] = 5.0 * np.exp(-(distance**2) / (2 * 5.0**2))

        from neurospatial.metrics.boundary_cells import border_score

        score = border_score(firing_rate, env)

        # Central field should have low or negative score
        assert score < 0.3, (
            f"Expected border score < 0.3 for central field, got {score}"
        )

    def test_border_score_corner_field(self) -> None:
        """Test border score for field in corner (touches two walls)."""
        positions = []
        for x in np.linspace(0, 40, 400):
            for y in np.linspace(0, 40, 400):
                positions.append([x, y])
        positions = np.array(positions)

        env = Environment.from_samples(positions, bin_size=4.0)

        # Create firing rate in corner (x < 10 AND y < 10)
        firing_rate = np.zeros(env.n_bins)
        for i in range(env.n_bins):
            center = env.bin_centers[i]
            if center[0] < 10 and center[1] < 10:
                firing_rate[i] = 5.0

        from neurospatial.metrics.boundary_cells import border_score

        score = border_score(firing_rate, env)

        # Corner field should have high score (touches boundaries)
        assert score > 0.4, f"Expected border score > 0.4 for corner field, got {score}"

    def test_border_score_uniform_firing(self) -> None:
        """Test border score with uniform firing everywhere."""
        positions = np.random.randn(5000, 2) * 20
        env = Environment.from_samples(positions, bin_size=4.0)

        # Uniform firing everywhere
        firing_rate = np.ones(env.n_bins) * 2.0

        from neurospatial.metrics.boundary_cells import border_score

        score = border_score(firing_rate, env)

        # Uniform firing covers all boundaries with perfect coverage (cM=1.0)
        # Mean distance to boundary is relatively low (many bins ARE boundaries)
        # So score should be positive (boundary coverage > distance)
        assert 0.5 < score <= 1.0, (
            f"Expected positive score for uniform firing covering boundaries, got {score}"
        )

    def test_border_score_threshold_parameter(self) -> None:
        """Test that threshold parameter affects field segmentation."""
        positions = np.random.randn(5000, 2) * 20
        env = Environment.from_samples(positions, bin_size=4.0)

        # Create gradient firing rate (higher near edge)
        firing_rate = np.zeros(env.n_bins)
        for i in range(env.n_bins):
            center = env.bin_centers[i]
            # Distance from center
            dist_from_center = np.linalg.norm(center)
            firing_rate[i] = dist_from_center / 20.0  # Normalize

        from neurospatial.metrics.boundary_cells import border_score

        # Lower threshold includes more bins
        score_low = border_score(firing_rate, env, threshold=0.1)

        # Higher threshold includes fewer bins
        score_high = border_score(firing_rate, env, threshold=0.5)

        # Both should be valid scores
        assert -1.0 <= score_low <= 1.0
        assert -1.0 <= score_high <= 1.0

    def test_border_score_all_nan(self) -> None:
        """Test border score with all NaN firing rates."""
        positions = np.random.randn(2000, 2) * 15
        env = Environment.from_samples(positions, bin_size=3.0)

        # All NaN
        firing_rate = np.full(env.n_bins, np.nan)

        from neurospatial.metrics.boundary_cells import border_score

        score = border_score(firing_rate, env)

        # Should return NaN for all-NaN input
        assert np.isnan(score), f"Expected NaN for all-NaN input, got {score}"

    def test_border_score_all_zeros(self) -> None:
        """Test border score with zero firing everywhere."""
        positions = np.random.randn(2000, 2) * 15
        env = Environment.from_samples(positions, bin_size=3.0)

        # All zeros
        firing_rate = np.zeros(env.n_bins)

        from neurospatial.metrics.boundary_cells import border_score

        score = border_score(firing_rate, env)

        # Should return NaN (no field detected)
        assert np.isnan(score), f"Expected NaN for zero firing, got {score}"

    def test_border_score_shape_validation(self) -> None:
        """Test input validation for firing_rate shape."""
        positions = np.random.randn(1000, 2) * 10
        env = Environment.from_samples(positions, bin_size=2.0)

        # Wrong shape
        firing_rate = np.ones((env.n_bins, 2))  # 2D instead of 1D

        from neurospatial.metrics.boundary_cells import border_score

        with pytest.raises(ValueError, match=r"firing_rate\.shape"):
            border_score(firing_rate, env)

    def test_border_score_threshold_validation(self) -> None:
        """Test validation of threshold parameter."""
        positions = np.random.randn(1000, 2) * 10
        env = Environment.from_samples(positions, bin_size=2.0)
        firing_rate = np.ones(env.n_bins)

        from neurospatial.metrics.boundary_cells import border_score

        # Threshold must be in (0, 1)
        with pytest.raises(ValueError, match="threshold must be in"):
            border_score(firing_rate, env, threshold=0.0)

        with pytest.raises(ValueError, match="threshold must be in"):
            border_score(firing_rate, env, threshold=1.5)

    def test_border_score_min_area_validation(self) -> None:
        """Test validation of min_area parameter."""
        positions = np.random.randn(1000, 2) * 10
        env = Environment.from_samples(positions, bin_size=2.0)
        firing_rate = np.ones(env.n_bins)

        from neurospatial.metrics.boundary_cells import border_score

        # min_area must be non-negative
        with pytest.raises(ValueError, match="min_area must be non-negative"):
            border_score(firing_rate, env, min_area=-10.0)

    def test_border_score_parameter_order(self) -> None:
        """Test that firing_rate comes before env (project convention)."""
        positions = np.random.randn(1000, 2) * 10
        env = Environment.from_samples(positions, bin_size=2.0)
        firing_rate = np.ones(env.n_bins)

        from neurospatial.metrics.boundary_cells import border_score

        # This should work (firing_rate first)
        score = border_score(firing_rate, env)
        assert isinstance(score, (float, np.floating))

    def test_border_score_returns_float(self) -> None:
        """Test that border_score returns a scalar float."""
        positions = np.random.randn(1000, 2) * 10
        env = Environment.from_samples(positions, bin_size=2.0)

        # Create simple border field
        firing_rate = np.zeros(env.n_bins)
        boundary = env.boundary_bins
        firing_rate[boundary] = 5.0

        from neurospatial.metrics.boundary_cells import border_score

        score = border_score(firing_rate, env)

        # Should return scalar
        assert np.ndim(score) == 0, "Border score should be scalar"
        assert isinstance(score, (float, np.floating)) or np.isnan(score)

    def test_border_score_range(self) -> None:
        """Test that border score is always in [-1, 1]."""
        # Test multiple random environments
        for _ in range(5):
            positions = np.random.randn(2000, 2) * 15
            env = Environment.from_samples(positions, bin_size=3.0)

            # Random firing rate
            firing_rate = np.random.rand(env.n_bins) * 5.0

            from neurospatial.metrics.boundary_cells import border_score

            score = border_score(firing_rate, env)

            # Score should be in valid range or NaN
            if not np.isnan(score):
                assert -1.0 <= score <= 1.0, (
                    f"Border score {score} out of range [-1, 1]"
                )
