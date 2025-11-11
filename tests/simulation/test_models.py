"""Tests for neural models."""

import numpy as np

from neurospatial.simulation.models import NeuralModel, PlaceCellModel
from neurospatial.simulation.trajectory import simulate_trajectory_ou


class TestNeuralModelProtocol:
    """Tests for NeuralModel protocol."""

    def test_placecell_implements_protocol(self, simple_2d_env):
        """Test that PlaceCellModel implements NeuralModel protocol."""
        pc = PlaceCellModel(simple_2d_env, center=[50.0, 50.0])
        assert isinstance(pc, NeuralModel)

    def test_protocol_requires_firing_rate_method(self):
        """Test that protocol requires firing_rate method."""

        class BadModel:
            @property
            def ground_truth(self):
                return {}

        bad = BadModel()
        assert not isinstance(bad, NeuralModel)

    def test_protocol_requires_ground_truth_property(self):
        """Test that protocol requires ground_truth property."""

        class BadModel:
            def firing_rate(self, positions, times=None):
                return np.zeros(len(positions))

        bad = BadModel()
        assert not isinstance(bad, NeuralModel)


class TestPlaceCellModel:
    """Tests for PlaceCellModel."""

    def test_basic_initialization(self, simple_2d_env):
        """Test basic place cell initialization."""
        pc = PlaceCellModel(
            simple_2d_env,
            center=[50.0, 50.0],
            width=10.0,
            max_rate=20.0,
            baseline_rate=0.1,
        )

        assert pc.center[0] == 50.0
        assert pc.center[1] == 50.0
        assert pc.width == 10.0
        assert pc.max_rate == 20.0
        assert pc.baseline_rate == 0.1

    def test_random_center_selection(self, simple_2d_env):
        """Test that center is randomly chosen from bin centers if not provided."""
        pc = PlaceCellModel(simple_2d_env, seed=42)

        # Center should be one of the bin centers
        distances = np.linalg.norm(simple_2d_env.bin_centers - pc.center, axis=1)
        min_dist = np.min(distances)
        assert min_dist < 1e-10  # Essentially zero

    def test_default_width(self, simple_2d_env):
        """Test that width defaults to 3 * bin_size."""
        pc = PlaceCellModel(simple_2d_env, center=[50.0, 50.0])

        # Default width should be 3 * mean(bin_sizes)
        expected_width = 3.0 * np.mean(simple_2d_env.bin_sizes)  # Property, not method
        assert abs(pc.width - expected_width) < 1e-10

    def test_peak_firing_at_center(self, simple_2d_env):
        """Test that firing rate peaks at field center."""
        center = np.array([50.0, 50.0])
        pc = PlaceCellModel(
            simple_2d_env,
            center=center,
            width=10.0,
            max_rate=20.0,
            baseline_rate=0.0,
        )

        # Compute firing rate at center
        positions = center.reshape(1, -1)
        times = np.array([0.0])
        rate = pc.firing_rate(positions, times)

        # Should be max_rate
        assert abs(rate[0] - 20.0) < 1e-6

    def test_gaussian_falloff(self, simple_2d_env):
        """Test Gaussian falloff with distance."""
        center = np.array([50.0, 50.0])
        width = 10.0
        pc = PlaceCellModel(
            simple_2d_env,
            center=center,
            width=width,
            max_rate=20.0,
            baseline_rate=0.0,
        )

        # Test at 1 sigma away
        positions = np.array([[50.0 + width, 50.0]])  # 1σ in x direction
        times = np.array([0.0])
        rate = pc.firing_rate(positions, times)

        # Expected rate at 1σ: max_rate * exp(-0.5) ≈ 0.606 * max_rate
        expected = 20.0 * np.exp(-0.5)
        assert abs(rate[0] - expected) < 0.1

    def test_euclidean_distance_metric(self, simple_2d_env):
        """Test euclidean distance metric (default)."""
        pc = PlaceCellModel(
            simple_2d_env,
            center=[50.0, 50.0],
            width=10.0,
            distance_metric="euclidean",
        )

        positions, times = simulate_trajectory_ou(simple_2d_env, duration=1.0, seed=42)
        rates = pc.firing_rate(positions, times)

        assert len(rates) == len(positions)
        assert np.all(rates >= 0)  # All rates non-negative

    def test_geodesic_distance_metric(self, simple_2d_env):
        """Test geodesic distance metric."""
        pc = PlaceCellModel(
            simple_2d_env,
            center=[50.0, 50.0],
            width=10.0,
            distance_metric="geodesic",
        )

        positions, times = simulate_trajectory_ou(simple_2d_env, duration=1.0, seed=42)
        rates = pc.firing_rate(positions, times)

        assert len(rates) == len(positions)
        assert np.all(rates >= 0)  # All rates non-negative

    def test_condition_function_gates_firing(self, simple_2d_env):
        """Test that condition function gates firing correctly."""

        # Condition: only fire in right half of arena
        def right_half_only(positions, times):
            return positions[:, 0] > 50.0

        pc = PlaceCellModel(
            simple_2d_env,
            center=[60.0, 50.0],  # Center in right half
            width=10.0,
            condition=right_half_only,
        )

        # Test positions in left half (should not fire much)
        left_positions = np.array([[40.0, 50.0], [30.0, 50.0]])
        left_times = np.array([0.0, 1.0])
        left_rates = pc.firing_rate(left_positions, left_times)

        # Rates should be zero (gated by condition)
        assert np.all(left_rates < 0.01)

        # Test positions in right half (should fire)
        right_positions = np.array([[60.0, 50.0], [70.0, 50.0]])
        right_times = np.array([0.0, 1.0])
        right_rates = pc.firing_rate(right_positions, right_times)

        # First position (at center) should have high rate
        assert right_rates[0] > 10.0

    def test_ground_truth_property(self, simple_2d_env):
        """Test ground_truth property returns correct parameters."""
        center = np.array([50.0, 50.0])
        width = 10.0
        max_rate = 20.0
        baseline_rate = 0.5

        pc = PlaceCellModel(
            simple_2d_env,
            center=center,
            width=width,
            max_rate=max_rate,
            baseline_rate=baseline_rate,
        )

        gt = pc.ground_truth

        assert "center" in gt
        assert "width" in gt
        assert "max_rate" in gt
        assert "baseline_rate" in gt

        np.testing.assert_array_equal(gt["center"], center)
        assert gt["width"] == width
        assert gt["max_rate"] == max_rate
        assert gt["baseline_rate"] == baseline_rate

    def test_reproducibility_with_seed(self, simple_2d_env):
        """Test that same seed produces same random center."""
        pc1 = PlaceCellModel(simple_2d_env, seed=42)
        pc2 = PlaceCellModel(simple_2d_env, seed=42)

        np.testing.assert_array_equal(pc1.center, pc2.center)

    def test_numerical_stability_far_from_center(self, simple_2d_env):
        """Test numerical stability for positions far from center."""
        pc = PlaceCellModel(
            simple_2d_env,
            center=[50.0, 50.0],
            width=5.0,
            max_rate=20.0,
            baseline_rate=0.1,
        )

        # Position very far from center (> 5σ)
        far_position = np.array([[200.0, 200.0]])  # Outside environment
        times = np.array([0.0])

        # Should not raise error, should return baseline rate
        # Note: position is outside environment, so geodesic metric may fail
        # Using euclidean (default)
        rate = pc.firing_rate(far_position, times)

        # Rate should be close to baseline (Gaussian decayed to ~0)
        # Allow tolerance for small contribution from tail
        assert abs(rate[0] - pc.baseline_rate) < 0.001
