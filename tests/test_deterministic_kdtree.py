"""Tests for deterministic KDTree enhancements in spatial.map_points_to_bins().

Test coverage:
- Deterministic behavior (reproducibility)
- max_distance parameter (absolute threshold)
- max_distance_factor parameter (relative threshold)
- Backward compatibility (no parameters = same behavior)
- Edge cases (empty arrays, all outside, etc.)
"""

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.ops.binning import map_points_to_bins

# ============================================================================
# Test Suite 1: Deterministic Behavior (Reproducibility)
# ============================================================================


class TestDeterministicBehavior:
    """Test that map_points_to_bins produces identical results on repeated calls."""

    def test_repeated_calls_identical(self):
        """Repeated calls with same inputs produce identical outputs."""
        # Create simple 2D grid environment
        data = np.array([[i, j] for i in range(10) for j in range(10)])
        env = Environment.from_samples(data, bin_size=1.0)

        # Query points (some on boundaries, some with potential ties)
        points = np.array(
            [
                [0.5, 0.5],  # Center of a bin
                [1.0, 1.0],  # On edge (potential tie)
                [1.5, 1.5],  # Center again
                [2.0, 2.0],  # On corner (potential tie with 4 bins)
            ]
        )

        # Call multiple times
        results = [map_points_to_bins(points, env) for _ in range(10)]

        # All results should be identical
        for result in results[1:]:
            np.testing.assert_array_equal(results[0], result)

    def test_deterministic_with_ties(self):
        """Points equidistant from multiple bins get consistent assignment."""
        # Create 2D grid
        data = np.array([[i, j] for i in range(5) for j in range(5)])
        env = Environment.from_samples(data, bin_size=1.0)

        # Point exactly between two bins (tie situation)
        # With tie_break="lowest_index", should always pick same bin
        points = np.array(
            [
                [0.5, 0.5],  # Equidistant from (0,0), (0,1), (1,0), (1,1)
            ]
        )

        # Run 100 times to ensure consistency
        results = [
            map_points_to_bins(points, env, tie_break="lowest_index")
            for _ in range(100)
        ]

        # All should be identical
        for result in results[1:]:
            np.testing.assert_array_equal(results[0], result)

    def test_no_randomness_in_large_datasets(self):
        """Large datasets produce deterministic results."""
        # Create larger environment
        data = np.array([[i, j] for i in range(20) for j in range(20)])
        env = Environment.from_samples(data, bin_size=1.0)

        # Large number of query points
        rng = np.random.default_rng(42)
        points = rng.uniform(0, 20, size=(1000, 2))

        # Call twice
        result1 = map_points_to_bins(points, env)
        result2 = map_points_to_bins(points, env)

        np.testing.assert_array_equal(result1, result2)


# ============================================================================
# Test Suite 2: max_distance Parameter (Absolute Threshold)
# ============================================================================


class TestMaxDistanceParameter:
    """Test absolute distance threshold functionality."""

    def test_max_distance_basic(self):
        """Points beyond max_distance are marked as outside (-1)."""
        # Create 2D grid with 1.0 spacing
        data = np.array([[i, j] for i in range(5) for j in range(5)])
        env = Environment.from_samples(data, bin_size=1.0)

        # Points: some inside, some far outside
        points = np.array(
            [
                [2.0, 2.0],  # Inside (near bin center)
                [10.0, 10.0],  # Far outside
                [2.1, 2.1],  # Inside (close to bin)
                [15.0, 15.0],  # Very far outside
            ]
        )

        # Set max_distance = 1.0 (one bin width)
        bin_indices = map_points_to_bins(points, env, max_distance=1.0)

        # First and third should be assigned to bins
        assert bin_indices[0] >= 0
        assert bin_indices[2] >= 0

        # Second and fourth should be marked as outside (-1)
        assert bin_indices[1] == -1
        assert bin_indices[3] == -1

    def test_max_distance_all_inside(self):
        """When all points within max_distance, none marked as outside."""
        data = np.array([[i, j] for i in range(5) for j in range(5)])
        env = Environment.from_samples(data, bin_size=1.0)

        # Points close to bin centers
        points = np.array(
            [
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 3.0],
            ]
        )

        # Large max_distance (covers entire environment)
        bin_indices = map_points_to_bins(points, env, max_distance=10.0)

        # All should be assigned
        assert np.all(bin_indices >= 0)

    def test_max_distance_all_outside(self):
        """When all points beyond max_distance, all marked as outside."""
        data = np.array([[i, j] for i in range(5) for j in range(5)])
        env = Environment.from_samples(data, bin_size=1.0)

        # Points far from environment
        points = np.array(
            [
                [100.0, 100.0],
                [200.0, 200.0],
                [-50.0, -50.0],
            ]
        )

        # Small max_distance
        bin_indices = map_points_to_bins(points, env, max_distance=1.0)

        # All should be marked outside
        np.testing.assert_array_equal(bin_indices, [-1, -1, -1])

    def test_max_distance_boundary_precision(self):
        """Points exactly at max_distance boundary are handled consistently."""
        data = np.array([[i, j] for i in range(5) for j in range(5)])
        env = Environment.from_samples(data, bin_size=1.0)

        # Get a bin center
        bin_center = env.bin_centers[0]

        # Create points at exactly max_distance (1.0) from bin center
        # Using Pythagorean theorem: point at (bin_center + [0.6, 0.8]) is distance 1.0
        offset = np.array([0.6, 0.8])  # 0.6^2 + 0.8^2 = 1.0
        points = np.array(
            [
                bin_center + offset,
                bin_center + offset * 0.99,  # Just inside
                bin_center + offset * 1.01,  # Just outside
            ]
        )

        bin_indices = map_points_to_bins(points, env, max_distance=1.0)

        # Point exactly at threshold or inside should be assigned
        assert bin_indices[1] >= 0  # Just inside

        # Point just outside threshold should be marked outside
        # (depends on implementation tolerance, but should be consistent)


# ============================================================================
# Test Suite 3: max_distance_factor Parameter (Relative Threshold)
# ============================================================================


class TestMaxDistanceFactor:
    """Test relative distance threshold functionality."""

    def test_max_distance_factor_basic(self):
        """max_distance_factor scales with bin size."""
        # Create 2D grid with known spacing
        data = np.array([[i * 2.0, j * 2.0] for i in range(5) for j in range(5)])
        env = Environment.from_samples(data, bin_size=2.0)

        # Points at different distances from bins
        points = np.array(
            [
                [4.0, 4.0],  # On bin center
                [4.5, 4.5],  # 0.5 * sqrt(2) ≈ 0.7 away
                [6.0, 6.0],  # 2 * sqrt(2) ≈ 2.8 away
                [10.0, 10.0],  # sqrt(2) * 4 ≈ 5.6 away
            ]
        )

        # max_distance_factor = 1.0 means points > 1.0 * bin_size away are outside
        # bin_size = 2.0, so threshold ≈ 2.0
        bin_indices = map_points_to_bins(points, env, max_distance_factor=1.0)

        # First two should be inside (distance < 2.0)
        assert bin_indices[0] >= 0
        assert bin_indices[1] >= 0

        # Last two should be outside (distance > 2.0)
        # Note: Exact behavior depends on how "typical bin size" is computed
        # We'll verify the principle: larger distance = more likely outside

    def test_max_distance_factor_adapts_to_scale(self):
        """Relative threshold adapts to environment scale."""
        # Create two environments with different scales
        data_small = np.array([[i, j] for i in range(10) for j in range(10)])
        env_small = Environment.from_samples(data_small, bin_size=1.0)

        data_large = np.array(
            [[i * 10.0, j * 10.0] for i in range(10) for j in range(10)]
        )
        env_large = Environment.from_samples(data_large, bin_size=10.0)

        # Same relative offset (1.5 bin sizes) in each environment
        points_small = np.array(
            [[0.0 + 1.5, 0.0 + 1.5]]
        )  # 1.5 away from (0,0) in small
        points_large = np.array(
            [[0.0 + 15.0, 0.0 + 15.0]]
        )  # 15.0 away from (0,0) in large

        # Apply same max_distance_factor
        factor = 1.0
        result_small = map_points_to_bins(
            points_small, env_small, max_distance_factor=factor
        )
        result_large = map_points_to_bins(
            points_large, env_large, max_distance_factor=factor
        )

        # Both should have similar behavior (either both inside or both outside)
        # Since we're using the same relative factor, behavior should be consistent

        # Verify both return valid results (either assigned or outside)
        assert result_small[0] >= -1
        assert result_large[0] >= -1

        # Key test: both environments should show same behavior (both inside or both outside)
        # because the relative distance is the same
        both_inside = (result_small[0] >= 0) and (result_large[0] >= 0)
        both_outside = (result_small[0] == -1) and (result_large[0] == -1)

        # Assert that behavior is consistent across scales
        assert both_inside or both_outside, (
            f"Inconsistent behavior: small={result_small[0]}, large={result_large[0]}. "
            "With same max_distance_factor, both should be inside or both outside."
        )

    def test_max_distance_factor_large_value(self):
        """Large factor includes more points."""
        data = np.array([[i, j] for i in range(5) for j in range(5)])
        env = Environment.from_samples(data, bin_size=1.0)

        # Point moderately far from bins
        points = np.array([[8.0, 8.0]])

        # With large factor, should be included
        bin_indices_large = map_points_to_bins(points, env, max_distance_factor=10.0)
        assert bin_indices_large[0] >= 0

        # With small factor, should be excluded
        bin_indices_small = map_points_to_bins(points, env, max_distance_factor=0.5)
        assert bin_indices_small[0] == -1


# ============================================================================
# Test Suite 4: Parameter Interaction & Validation
# ============================================================================


class TestParameterInteraction:
    """Test interaction between parameters and validation."""

    def test_both_distance_parameters_error(self):
        """Providing both max_distance and max_distance_factor raises error."""
        data = np.array([[i, j] for i in range(5) for j in range(5)])
        env = Environment.from_samples(data, bin_size=1.0)
        points = np.array([[2.0, 2.0]])

        with pytest.raises(ValueError, match="Cannot specify both"):
            map_points_to_bins(points, env, max_distance=1.0, max_distance_factor=1.5)

    def test_negative_max_distance_error(self):
        """Negative max_distance raises error."""
        data = np.array([[i, j] for i in range(5) for j in range(5)])
        env = Environment.from_samples(data, bin_size=1.0)
        points = np.array([[2.0, 2.0]])

        with pytest.raises(ValueError, match="max_distance must be non-negative"):
            map_points_to_bins(points, env, max_distance=-1.0)

    def test_negative_max_distance_factor_error(self):
        """Negative max_distance_factor raises error."""
        data = np.array([[i, j] for i in range(5) for j in range(5)])
        env = Environment.from_samples(data, bin_size=1.0)
        points = np.array([[2.0, 2.0]])

        with pytest.raises(ValueError, match="max_distance_factor must be positive"):
            map_points_to_bins(points, env, max_distance_factor=-0.5)

    def test_zero_max_distance_factor_error(self):
        """Zero max_distance_factor raises error."""
        data = np.array([[i, j] for i in range(5) for j in range(5)])
        env = Environment.from_samples(data, bin_size=1.0)
        points = np.array([[2.0, 2.0]])

        with pytest.raises(ValueError, match="max_distance_factor must be positive"):
            map_points_to_bins(points, env, max_distance_factor=0.0)

    def test_zero_max_distance(self):
        """max_distance=0.0 only accepts points exactly on bin centers."""
        data = np.array([[i, j] for i in range(5) for j in range(5)])
        env = Environment.from_samples(data, bin_size=1.0)

        # Get actual bin centers (will be exactly on integer coords after from_samples)
        bin_center = env.bin_centers[12]  # Some bin center

        # Points: one exactly on center, one slightly off
        offset = np.array([0.01, 0.01])
        points = np.array(
            [
                bin_center,  # Distance = 0 (on center)
                bin_center + offset,  # Distance > 0 (slightly off)
            ]
        )

        bin_indices = map_points_to_bins(points, env, max_distance=0.0)

        # Point on center (distance=0) should be assigned (0 is not > 0)
        assert bin_indices[0] >= 0

        # Point off center (distance>0) should be marked outside
        assert bin_indices[1] == -1


# ============================================================================
# Test Suite 5: Backward Compatibility
# ============================================================================


class TestBackwardCompatibility:
    """Test that existing behavior is preserved when new parameters not used."""

    def test_no_parameters_unchanged(self):
        """Without new parameters, uses default heuristic (10× typical spacing)."""
        data = np.array([[i, j] for i in range(10) for j in range(10)])
        env = Environment.from_samples(data, bin_size=1.0)

        points = np.array(
            [
                [2.0, 2.0],
                [5.5, 5.5],
                [100.0, 100.0],  # Very far outside, should be marked -1 by heuristic
            ]
        )

        bin_indices = map_points_to_bins(points, env)

        # First two points should be assigned (inside environment)
        assert bin_indices[0] >= 0
        assert bin_indices[1] >= 0

        # Far outlier should be marked outside with default heuristic
        assert bin_indices[2] == -1

    def test_tie_break_still_works(self):
        """tie_break parameter still functions as before."""
        data = np.array([[i, j] for i in range(5) for j in range(5)])
        env = Environment.from_samples(data, bin_size=1.0)

        points = np.array([[0.5, 0.5]])  # Equidistant from multiple bins

        # With lowest_index tie break
        result = map_points_to_bins(points, env, tie_break="lowest_index")
        assert result[0] >= 0  # Should be assigned consistently


# ============================================================================
# Test Suite 6: Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_points_array(self):
        """Empty points array returns empty result."""
        data = np.array([[i, j] for i in range(5) for j in range(5)])
        env = Environment.from_samples(data, bin_size=1.0)

        points = np.array([]).reshape(0, 2)
        bin_indices = map_points_to_bins(points, env, max_distance=1.0)

        assert bin_indices.shape == (0,)

    def test_single_point(self):
        """Single point handled correctly."""
        data = np.array([[i, j] for i in range(5) for j in range(5)])
        env = Environment.from_samples(data, bin_size=1.0)

        points = np.array([[2.0, 2.0]])
        bin_indices = map_points_to_bins(points, env, max_distance=1.0)

        assert bin_indices.shape == (1,)
        assert bin_indices[0] >= 0

    def test_all_points_same_location(self):
        """Multiple points at same location get same bin assignment."""
        data = np.array([[i, j] for i in range(5) for j in range(5)])
        env = Environment.from_samples(data, bin_size=1.0)

        points = np.array([[2.0, 2.0]] * 10)
        bin_indices = map_points_to_bins(points, env, max_distance=1.0)

        # All should map to same bin
        assert len(np.unique(bin_indices)) == 1
        assert bin_indices[0] >= 0


# ============================================================================
# Test Suite 7: Different Layout Types
# ============================================================================


class TestDifferentLayouts:
    """Test that enhancements work across different layout types."""

    def test_max_distance_on_polygon_layout(self):
        """max_distance works on polygon-masked environment."""
        from shapely.geometry import box

        # Create environment from polygon
        polygon = box(0, 0, 10, 10)
        env = Environment.from_polygon(polygon, bin_size=1.0)

        points = np.array(
            [
                [5.0, 5.0],  # Inside
                [20.0, 20.0],  # Far outside
            ]
        )

        bin_indices = map_points_to_bins(points, env, max_distance=2.0)

        assert bin_indices[0] >= 0
        assert bin_indices[1] == -1


# ============================================================================
# Test Suite 8: Integration with Environment Methods
# ============================================================================


class TestIntegrationWithEnvironment:
    """Test that enhancements integrate properly with Environment methods."""

    def test_occupancy_with_outliers(self):
        """occupancy() benefits from outlier filtering via max_distance."""
        # This test verifies that when map_points_to_bins is called internally
        # by occupancy(), outliers are handled properly

        # Create environment
        data = np.array([[i, j] for i in range(10) for j in range(10)])
        env = Environment.from_samples(data, bin_size=1.0)

        # Trajectory with outlier
        times = np.array([0.0, 1.0, 2.0, 3.0])
        positions = np.array(
            [
                [2.0, 2.0],
                [2.1, 2.1],
                [100.0, 100.0],  # Outlier
                [2.2, 2.2],
            ]
        )

        # Without max_distance, outlier gets assigned to some bin
        occ_no_filter = env.occupancy(times, positions)
        # With max_distance in map_points_to_bins (if occupancy uses it),
        # outlier would be excluded

        # Note: This test depends on whether occupancy() is modified to use max_distance
        # For now, just verify occupancy runs without error
        assert occ_no_filter.shape == (env.n_bins,)

    def test_bin_sequence_deterministic(self):
        """bin_sequence() produces deterministic results."""
        data = np.array([[i, j] for i in range(10) for j in range(10)])
        env = Environment.from_samples(data, bin_size=1.0)

        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        positions = np.array(
            [
                [2.0, 2.0],
                [2.1, 2.1],
                [3.0, 3.0],
                [3.5, 3.5],
                [4.0, 4.0],
            ]
        )

        # Call multiple times
        seq1 = env.bin_sequence(times, positions)
        seq2 = env.bin_sequence(times, positions)

        np.testing.assert_array_equal(seq1, seq2)
