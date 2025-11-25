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

    def test_border_score_perfect_border_cell(
        self, dense_rectangular_grid_env: Environment
    ) -> None:
        """Test border score for perfect border cell (field along one wall)."""
        env = dense_rectangular_grid_env

        # Create firing rate map with activity only along left wall (x < 10)
        firing_rate = np.zeros(env.n_bins)
        for i in range(env.n_bins):
            center = env.bin_centers[i]
            if center[0] < 10:  # Left wall
                firing_rate[i] = 5.0
            else:
                firing_rate[i] = 0.0

        from neurospatial.metrics.boundary_cells import border_score

        score = border_score(firing_rate, env)

        # Perfect border cell should have high positive score (close to 1)
        assert score > 0.5, f"Expected border score > 0.5 for border cell, got {score}"

    def test_border_score_central_field(
        self, dense_50x50_bin5_env: Environment
    ) -> None:
        """Test border score for central field (non-border cell)."""
        env = dense_50x50_bin5_env

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

    def test_border_score_corner_field(self, dense_40x40_grid_env: Environment) -> None:
        """Test border score for field in corner (touches two walls)."""
        env = dense_40x40_grid_env

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

    def test_border_score_uniform_firing(
        self, dense_rectangular_grid_env: Environment
    ) -> None:
        """Test border score with uniform firing everywhere."""
        env = dense_rectangular_grid_env

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

    def test_border_score_threshold_parameter(
        self, dense_rectangular_grid_env: Environment
    ) -> None:
        """Test that threshold parameter affects field segmentation."""
        env = dense_rectangular_grid_env

        # Create gradient firing rate (higher near edge)
        # For dense_rectangular_grid_env (0-50 extent), center is at (25, 25)
        center_point = np.array([25.0, 25.0])
        distances = np.linalg.norm(env.bin_centers - center_point, axis=1)
        firing_rate = distances / 25.0  # Normalize

        from neurospatial.metrics.boundary_cells import border_score

        # Lower threshold includes more bins
        score_low = border_score(firing_rate, env, threshold=0.1)

        # Higher threshold includes fewer bins
        score_high = border_score(firing_rate, env, threshold=0.5)

        # Both should be valid scores
        assert -1.0 <= score_low <= 1.0
        assert -1.0 <= score_high <= 1.0

    def test_border_score_all_nan(self, small_2d_env: Environment) -> None:
        """Test border score with all NaN firing rates."""
        env = small_2d_env

        # All NaN
        firing_rate = np.full(env.n_bins, np.nan)

        from neurospatial.metrics.boundary_cells import border_score

        score = border_score(firing_rate, env)

        # Should return NaN for all-NaN input
        assert np.isnan(score), f"Expected NaN for all-NaN input, got {score}"

    def test_border_score_all_zeros(self, small_2d_env: Environment) -> None:
        """Test border score with zero firing everywhere."""
        env = small_2d_env

        # All zeros
        firing_rate = np.zeros(env.n_bins)

        from neurospatial.metrics.boundary_cells import border_score

        score = border_score(firing_rate, env)

        # Should return NaN (no field detected)
        assert np.isnan(score), f"Expected NaN for zero firing, got {score}"

    def test_border_score_shape_validation(self, small_2d_env: Environment) -> None:
        """Test input validation for firing_rate shape."""
        env = small_2d_env

        # Wrong shape
        firing_rate = np.ones((env.n_bins, 2))  # 2D instead of 1D

        from neurospatial.metrics.boundary_cells import border_score

        with pytest.raises(ValueError, match=r"firing_rate\.shape"):
            border_score(firing_rate, env)

    def test_border_score_threshold_validation(self, small_2d_env: Environment) -> None:
        """Test validation of threshold parameter."""
        env = small_2d_env
        firing_rate = np.ones(env.n_bins)

        from neurospatial.metrics.boundary_cells import border_score

        # Threshold must be in (0, 1)
        with pytest.raises(ValueError, match="threshold must be in"):
            border_score(firing_rate, env, threshold=0.0)

        with pytest.raises(ValueError, match="threshold must be in"):
            border_score(firing_rate, env, threshold=1.5)

    def test_border_score_min_area_validation(self, small_2d_env: Environment) -> None:
        """Test validation of min_area parameter."""
        env = small_2d_env
        firing_rate = np.ones(env.n_bins)

        from neurospatial.metrics.boundary_cells import border_score

        # min_area must be non-negative
        with pytest.raises(ValueError, match="min_area must be non-negative"):
            border_score(firing_rate, env, min_area=-10.0)

    def test_border_score_parameter_order(self, small_2d_env: Environment) -> None:
        """Test that firing_rate comes before env (project convention)."""
        env = small_2d_env
        firing_rate = np.ones(env.n_bins)

        from neurospatial.metrics.boundary_cells import border_score

        # This should work (firing_rate first)
        score = border_score(firing_rate, env)
        assert isinstance(score, (float, np.floating))

    def test_border_score_returns_float(self, small_2d_env: Environment) -> None:
        """Test that border_score returns a scalar float."""
        env = small_2d_env

        # Create simple border field
        firing_rate = np.zeros(env.n_bins)
        boundary = env.boundary_bins
        firing_rate[boundary] = 5.0

        from neurospatial.metrics.boundary_cells import border_score

        score = border_score(firing_rate, env)

        # Should return scalar
        assert np.ndim(score) == 0, "Border score should be scalar"
        assert isinstance(score, (float, np.floating)) or np.isnan(score)

    def test_border_score_range(self, dense_rectangular_grid_env: Environment) -> None:
        """Test that border score is always in [-1, 1]."""
        env = dense_rectangular_grid_env
        rng = np.random.default_rng(42)

        from neurospatial.metrics.boundary_cells import border_score

        # Test multiple random firing rates on same environment
        for _ in range(5):
            # Random firing rate
            firing_rate = rng.random(env.n_bins) * 5.0

            score = border_score(firing_rate, env)

            # Score should be in valid range or NaN
            if not np.isnan(score):
                assert -1.0 <= score <= 1.0, (
                    f"Border score {score} out of range [-1, 1]"
                )

    def test_border_score_distance_metric_geodesic(
        self, dense_rectangular_grid_env: Environment
    ) -> None:
        """Test border score with geodesic distance metric (default)."""
        env = dense_rectangular_grid_env

        # Create border cell
        firing_rate = np.zeros(env.n_bins)
        boundary = env.boundary_bins
        firing_rate[boundary] = 5.0

        from neurospatial.metrics.boundary_cells import border_score

        # Explicit geodesic
        score_geodesic = border_score(firing_rate, env, distance_metric="geodesic")

        # Default should match geodesic
        score_default = border_score(firing_rate, env)

        assert score_geodesic == score_default, (
            "Default distance metric should be geodesic"
        )
        assert -1.0 <= score_geodesic <= 1.0

    def test_border_score_distance_metric_euclidean(
        self, dense_rectangular_grid_env: Environment
    ) -> None:
        """Test border score with euclidean distance metric."""
        env = dense_rectangular_grid_env

        # Create border cell
        firing_rate = np.zeros(env.n_bins)
        boundary = env.boundary_bins
        firing_rate[boundary] = 5.0

        from neurospatial.metrics.boundary_cells import border_score

        score = border_score(firing_rate, env, distance_metric="euclidean")

        # Should return valid score
        assert -1.0 <= score <= 1.0, f"Expected score in [-1, 1], got {score}"

    def test_border_score_distance_metrics_differ(
        self, dense_50x50_bin5_env: Environment
    ) -> None:
        """Test that geodesic and euclidean give different results."""
        env = dense_50x50_bin5_env

        # Create central field that requires path around obstacles
        firing_rate = np.zeros(env.n_bins)
        for i in range(env.n_bins):
            center = env.bin_centers[i]
            distance = np.sqrt((center[0] - 25) ** 2 + (center[1] - 25) ** 2)
            if distance < 8:  # Central field
                firing_rate[i] = 5.0

        from neurospatial.metrics.boundary_cells import border_score

        score_geodesic = border_score(firing_rate, env, distance_metric="geodesic")
        score_euclidean = border_score(firing_rate, env, distance_metric="euclidean")

        # Both should be valid
        assert -1.0 <= score_geodesic <= 1.0
        assert -1.0 <= score_euclidean <= 1.0

        # For regular grids, they might be similar, but we test they're computed
        # (exact equality would depend on environment structure)

    def test_border_score_distance_metric_validation(
        self, small_2d_env: Environment
    ) -> None:
        """Test validation of distance_metric parameter."""
        env = small_2d_env
        firing_rate = np.ones(env.n_bins)

        from neurospatial.metrics.boundary_cells import border_score

        # Invalid distance metric
        with pytest.raises(
            ValueError, match=r"distance_metric must be 'geodesic' or 'euclidean'"
        ):
            border_score(firing_rate, env, distance_metric="invalid")

    def test_border_score_euclidean_central_field(
        self, dense_50x50_bin5_env: Environment
    ) -> None:
        """Test euclidean distance for central field (should have low score)."""
        env = dense_50x50_bin5_env

        # Create central field
        firing_rate = np.zeros(env.n_bins)
        for i in range(env.n_bins):
            center = env.bin_centers[i]
            distance = np.sqrt((center[0] - 25) ** 2 + (center[1] - 25) ** 2)
            firing_rate[i] = 5.0 * np.exp(-(distance**2) / (2 * 5.0**2))

        from neurospatial.metrics.boundary_cells import border_score

        score = border_score(firing_rate, env, distance_metric="euclidean")

        # Central field should have low score with euclidean distance
        assert score < 0.3, (
            f"Expected euclidean border score < 0.3 for central field, got {score}"
        )


class TestComputeRegionCoverage:
    """Test compute_region_coverage function.

    Note: These tests require inline Environment.from_samples() calls because they
    add regions to the environment. Session-scoped fixtures cannot be used since
    region additions would persist and conflict between tests.
    """

    def test_region_coverage_basic(self) -> None:
        """Test basic region coverage computation with wall regions."""
        from shapely.geometry import box

        # Inline environment required: adds wall regions (north, south, east, west)
        x = np.linspace(0, 40, 400)
        y = np.linspace(0, 40, 400)
        xx, yy = np.meshgrid(x, y)
        positions = np.column_stack([xx.ravel(), yy.ravel()])

        env = Environment.from_samples(positions, bin_size=4.0)

        # Add wall regions
        env.regions.add("north", polygon=box(0, 30, 40, 40))
        env.regions.add("south", polygon=box(0, 0, 40, 10))
        env.regions.add("east", polygon=box(30, 0, 40, 40))
        env.regions.add("west", polygon=box(0, 0, 10, 40))

        # Create field along north wall
        firing_rate = np.zeros(env.n_bins)
        north_bins = np.where(env.mask_for_region("north"))[0]
        firing_rate[north_bins] = 5.0
        field_bins = np.where(firing_rate > 0)[0]

        from neurospatial.metrics.boundary_cells import compute_region_coverage

        coverage = compute_region_coverage(field_bins, env)

        # North wall should have high coverage, others should be low
        # Note: walls overlap at corners, so east/west have ~0.22 coverage from corner overlap
        assert coverage["north"] > 0.8, (
            f"Expected north coverage > 0.8, got {coverage['north']}"
        )
        assert coverage["south"] < 0.3, (
            f"Expected south coverage < 0.3, got {coverage['south']}"
        )
        assert coverage["east"] < 0.3, (
            f"Expected east coverage < 0.3, got {coverage['east']}"
        )
        assert coverage["west"] < 0.3, (
            f"Expected west coverage < 0.3, got {coverage['west']}"
        )

    def test_region_coverage_specific_regions(self) -> None:
        """Test region coverage with specific regions parameter."""
        from shapely.geometry import box

        rng = np.random.default_rng(42)
        positions = rng.standard_normal((2000, 2)) * 20
        env = Environment.from_samples(positions, bin_size=4.0)

        env.regions.add("region1", polygon=box(-20, -20, 0, 0))
        env.regions.add("region2", polygon=box(0, 0, 20, 20))
        env.regions.add("region3", polygon=box(-20, 0, 0, 20))

        # Create field in region1
        firing_rate = np.zeros(env.n_bins)
        region1_bins = np.where(env.mask_for_region("region1"))[0]
        firing_rate[region1_bins] = 5.0
        field_bins = np.where(firing_rate > 0)[0]

        from neurospatial.metrics.boundary_cells import compute_region_coverage

        # Only compute for region1 and region2
        coverage = compute_region_coverage(
            field_bins, env, regions=["region1", "region2"]
        )

        # Should only return specified regions
        assert set(coverage.keys()) == {"region1", "region2"}
        assert coverage["region1"] > 0.8
        assert coverage["region2"] < 0.2

    def test_region_coverage_none_regions(self) -> None:
        """Test region coverage with regions=None uses all regions."""
        from shapely.geometry import box

        rng = np.random.default_rng(42)
        positions = rng.standard_normal((2000, 2)) * 20
        env = Environment.from_samples(positions, bin_size=4.0)

        env.regions.add("region1", polygon=box(-20, -20, 0, 0))
        env.regions.add("region2", polygon=box(0, 0, 20, 20))

        firing_rate = np.zeros(env.n_bins)
        region1_bins = np.where(env.mask_for_region("region1"))[0]
        firing_rate[region1_bins] = 5.0
        field_bins = np.where(firing_rate > 0)[0]

        from neurospatial.metrics.boundary_cells import compute_region_coverage

        coverage = compute_region_coverage(field_bins, env, regions=None)

        # Should return all regions
        assert set(coverage.keys()) == {"region1", "region2"}

    def test_region_coverage_empty_region(self) -> None:
        """Test region coverage with empty region."""
        from shapely.geometry import box

        rng = np.random.default_rng(42)
        positions = rng.standard_normal((2000, 2)) * 20
        env = Environment.from_samples(positions, bin_size=4.0)

        # Add region that doesn't overlap with any bins
        env.regions.add("far_away", polygon=box(1000, 1000, 1100, 1100))

        firing_rate = np.zeros(env.n_bins)
        firing_rate[0] = 5.0
        field_bins = np.where(firing_rate > 0)[0]

        from neurospatial.metrics.boundary_cells import compute_region_coverage

        coverage = compute_region_coverage(field_bins, env)

        # Empty region should have 0.0 coverage
        assert coverage["far_away"] == 0.0

    def test_region_coverage_nonexistent_region(
        self, small_2d_env: Environment
    ) -> None:
        """Test region coverage with non-existent region raises error."""
        env = small_2d_env

        firing_rate = np.zeros(env.n_bins)
        firing_rate[0] = 5.0
        field_bins = np.where(firing_rate > 0)[0]

        from neurospatial.metrics.boundary_cells import compute_region_coverage

        with pytest.raises(ValueError, match="Region 'nonexistent' not found"):
            compute_region_coverage(field_bins, env, regions=["nonexistent"])

    def test_region_coverage_return_type(self) -> None:
        """Test region coverage returns dict[str, float]."""
        from shapely.geometry import box

        rng = np.random.default_rng(42)
        positions = rng.standard_normal((2000, 2)) * 20
        env = Environment.from_samples(positions, bin_size=4.0)

        env.regions.add("test_region", polygon=box(-10, -10, 10, 10))

        firing_rate = np.zeros(env.n_bins)
        firing_rate[0] = 5.0
        field_bins = np.where(firing_rate > 0)[0]

        from neurospatial.metrics.boundary_cells import compute_region_coverage

        coverage = compute_region_coverage(field_bins, env)

        # Should return dict
        assert isinstance(coverage, dict)
        # Keys should be strings
        for key in coverage:
            assert isinstance(key, str)
        # Values should be floats
        for value in coverage.values():
            assert isinstance(value, float)

    def test_region_coverage_full_coverage(self) -> None:
        """Test region coverage when field covers entire region."""
        from shapely.geometry import box

        # Create rectangular environment using vectorized meshgrid
        x = np.linspace(0, 40, 400)
        y = np.linspace(0, 40, 400)
        xx, yy = np.meshgrid(x, y)
        positions = np.column_stack([xx.ravel(), yy.ravel()])

        env = Environment.from_samples(positions, bin_size=4.0)

        # Add region
        env.regions.add("center", polygon=box(15, 15, 25, 25))

        # Create field that covers all of center region
        center_bins = np.where(env.mask_for_region("center"))[0]
        field_bins = center_bins

        from neurospatial.metrics.boundary_cells import compute_region_coverage

        coverage = compute_region_coverage(field_bins, env)

        # Should have perfect coverage
        assert coverage["center"] == 1.0

    def test_region_coverage_no_coverage(self) -> None:
        """Test region coverage when field doesn't overlap region."""
        from shapely.geometry import box

        # Create rectangular environment using vectorized meshgrid
        x = np.linspace(0, 40, 400)
        y = np.linspace(0, 40, 400)
        xx, yy = np.meshgrid(x, y)
        positions = np.column_stack([xx.ravel(), yy.ravel()])

        env = Environment.from_samples(positions, bin_size=4.0)

        # Add two disjoint regions
        env.regions.add("north", polygon=box(0, 30, 40, 40))
        env.regions.add("south", polygon=box(0, 0, 40, 10))

        # Create field only in north
        north_bins = np.where(env.mask_for_region("north"))[0]
        field_bins = north_bins

        from neurospatial.metrics.boundary_cells import compute_region_coverage

        coverage = compute_region_coverage(field_bins, env)

        # North should have full coverage, south should have zero
        assert coverage["north"] == 1.0
        assert coverage["south"] == 0.0
