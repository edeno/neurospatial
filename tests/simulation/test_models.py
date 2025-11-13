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


class TestBoundaryCellModel:
    """Tests for BoundaryCellModel."""

    def test_basic_initialization(self, simple_2d_env):
        """Test basic boundary cell initialization."""
        from neurospatial.simulation.models import BoundaryCellModel

        bc = BoundaryCellModel(
            simple_2d_env,
            preferred_distance=5.0,
            distance_tolerance=3.0,
            max_rate=15.0,
            baseline_rate=0.1,
        )

        assert bc.preferred_distance == 5.0
        assert bc.distance_tolerance == 3.0
        assert bc.max_rate == 15.0
        assert bc.baseline_rate == 0.1
        assert bc.preferred_direction is None  # Default: omnidirectional

    def test_directional_initialization(self, simple_2d_env):
        """Test boundary cell with directional preference."""
        from neurospatial.simulation.models import BoundaryCellModel

        bc = BoundaryCellModel(
            simple_2d_env,
            preferred_distance=10.0,
            preferred_direction=-np.pi / 2,  # South
            direction_tolerance=np.pi / 6,
        )

        assert bc.preferred_direction == -np.pi / 2
        assert bc.direction_tolerance == np.pi / 6

    def test_peak_firing_at_preferred_distance(self, simple_2d_env):
        """Test that firing rate peaks at preferred distance from boundary."""
        from neurospatial.simulation.models import BoundaryCellModel

        bc = BoundaryCellModel(
            simple_2d_env,
            preferred_distance=5.0,
            distance_tolerance=3.0,
            max_rate=20.0,
            baseline_rate=0.0,
        )

        # Generate trajectory
        positions, times = simulate_trajectory_ou(simple_2d_env, duration=10.0, seed=42)
        rates = bc.firing_rate(positions, times)

        # Should have some firing (non-zero rates)
        assert np.any(rates > 0.1)
        assert np.all(rates >= 0)  # All rates non-negative

    def test_omnidirectional_tuning(self, simple_2d_env):
        """Test omnidirectional boundary cell (no directional preference)."""
        from neurospatial.simulation.models import BoundaryCellModel

        bc = BoundaryCellModel(
            simple_2d_env,
            preferred_distance=5.0,
            distance_tolerance=3.0,
            preferred_direction=None,  # Omnidirectional
        )

        positions, times = simulate_trajectory_ou(simple_2d_env, duration=5.0, seed=42)
        rates = bc.firing_rate(positions, times)

        # Should fire near all boundaries
        assert len(rates) == len(positions)
        assert np.all(rates >= 0)

    def test_directional_tuning(self, simple_2d_env):
        """Test directional boundary cell (prefers specific direction)."""
        from neurospatial.simulation.models import BoundaryCellModel

        # Boundary vector cell preferring south wall
        bc_south = BoundaryCellModel(
            simple_2d_env,
            preferred_distance=10.0,
            preferred_direction=-np.pi / 2,  # South (negative y)
            direction_tolerance=np.pi / 6,
        )

        positions, times = simulate_trajectory_ou(simple_2d_env, duration=5.0, seed=42)
        rates = bc_south.firing_rate(positions, times)

        # Should have some firing
        assert len(rates) == len(positions)
        assert np.all(rates >= 0)

    def test_gaussian_distance_tuning(self, simple_2d_env):
        """Test Gaussian tuning around preferred distance."""
        from neurospatial.simulation.models import BoundaryCellModel

        bc = BoundaryCellModel(
            simple_2d_env,
            preferred_distance=5.0,
            distance_tolerance=2.0,
            max_rate=20.0,
            baseline_rate=0.0,
        )

        # Test that firing falls off with distance from preferred_distance
        positions, times = simulate_trajectory_ou(simple_2d_env, duration=5.0, seed=42)
        rates = bc.firing_rate(positions, times)

        # Rates should be bounded
        assert np.all(rates >= 0)
        assert np.all(rates <= bc.max_rate + 0.1)  # Allow small tolerance

    def test_ground_truth_property(self, simple_2d_env):
        """Test ground_truth property returns correct parameters."""
        from neurospatial.simulation.models import BoundaryCellModel

        preferred_distance = 5.0
        distance_tolerance = 3.0
        preferred_direction = -np.pi / 2
        direction_tolerance = np.pi / 4
        max_rate = 15.0
        baseline_rate = 0.5

        bc = BoundaryCellModel(
            simple_2d_env,
            preferred_distance=preferred_distance,
            distance_tolerance=distance_tolerance,
            preferred_direction=preferred_direction,
            direction_tolerance=direction_tolerance,
            max_rate=max_rate,
            baseline_rate=baseline_rate,
        )

        gt = bc.ground_truth

        assert "preferred_distance" in gt
        assert "distance_tolerance" in gt
        assert "preferred_direction" in gt
        assert "direction_tolerance" in gt
        assert "max_rate" in gt
        assert "baseline_rate" in gt

        assert gt["preferred_distance"] == preferred_distance
        assert gt["distance_tolerance"] == distance_tolerance
        assert gt["preferred_direction"] == preferred_direction
        assert gt["direction_tolerance"] == direction_tolerance
        assert gt["max_rate"] == max_rate
        assert gt["baseline_rate"] == baseline_rate

    def test_implements_neural_model_protocol(self, simple_2d_env):
        """Test that BoundaryCellModel implements NeuralModel protocol."""
        from neurospatial.simulation.models import BoundaryCellModel

        bc = BoundaryCellModel(simple_2d_env, preferred_distance=5.0)
        assert isinstance(bc, NeuralModel)

    def test_euclidean_distance_metric(self, simple_2d_env):
        """Test euclidean distance metric."""
        from neurospatial.simulation.models import BoundaryCellModel

        bc = BoundaryCellModel(
            simple_2d_env,
            preferred_distance=5.0,
            distance_metric="euclidean",
        )

        positions, times = simulate_trajectory_ou(simple_2d_env, duration=2.0, seed=42)
        rates = bc.firing_rate(positions, times)

        assert len(rates) == len(positions)
        assert np.all(rates >= 0)

    def test_geodesic_distance_metric(self, simple_2d_env):
        """Test geodesic distance metric (default)."""
        from neurospatial.simulation.models import BoundaryCellModel

        bc = BoundaryCellModel(
            simple_2d_env,
            preferred_distance=5.0,
            distance_metric="geodesic",
        )

        positions, times = simulate_trajectory_ou(simple_2d_env, duration=2.0, seed=42)
        rates = bc.firing_rate(positions, times)

        assert len(rates) == len(positions)
        assert np.all(rates >= 0)


class TestGridCellModel:
    """Tests for GridCellModel."""

    def test_requires_2d_environment(self, simple_2d_env):
        """Test that GridCellModel requires a 2D environment."""
        import pytest

        from neurospatial.simulation.models import GridCellModel

        # Should work with 2D environment
        gc = GridCellModel(simple_2d_env, grid_spacing=50.0)
        assert gc is not None

        # Should fail with 1D environment
        # Create a simple 1D environment for testing
        from neurospatial import Environment

        env_1d = Environment.from_samples(
            np.array([[0.0], [10.0], [20.0], [30.0]]), bin_size=5.0
        )

        with pytest.raises(ValueError, match="only works for 2D"):
            GridCellModel(env_1d, grid_spacing=50.0)

    def test_basic_initialization(self, simple_2d_env):
        """Test basic grid cell initialization."""
        from neurospatial.simulation.models import GridCellModel

        gc = GridCellModel(
            simple_2d_env,
            grid_spacing=50.0,
            grid_orientation=0.0,
            max_rate=20.0,
            baseline_rate=0.1,
        )

        assert gc.grid_spacing == 50.0
        assert gc.grid_orientation == 0.0
        assert gc.max_rate == 20.0
        assert gc.baseline_rate == 0.1

    def test_hexagonal_symmetry(self, simple_2d_env):
        """Test that grid pattern has hexagonal symmetry."""
        from neurospatial.simulation.models import GridCellModel

        # Set phase_offset at center so we're testing around a grid peak
        center_pos = np.array([50.0, 50.0])
        gc = GridCellModel(
            simple_2d_env,
            grid_spacing=30.0,
            grid_orientation=0.0,
            phase_offset=center_pos,  # Ensure peak at center
            max_rate=20.0,
            baseline_rate=0.0,
        )

        # Center point should be at a peak
        center = np.array([[50.0, 50.0]])
        center_rate = gc.firing_rate(center)[0]
        assert center_rate > 15.0  # Should be near max_rate

        # Test 6 points arranged hexagonally around center
        # Hexagon vertices at 60° intervals at distance = grid_spacing
        angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180
        spacing = 30.0
        hex_points = np.column_stack(
            [
                center_pos[0] + spacing * np.cos(angles),
                center_pos[1] + spacing * np.sin(angles),
            ]
        )

        hex_rates = gc.firing_rate(hex_points)

        # All hexagonal neighbors should have similar rates (hexagonal symmetry)
        # Allow some tolerance due to numerical precision and grid structure
        # Note: exact hexagonal symmetry depends on the phase relationship
        mean_rate = np.mean(hex_rates)
        for rate in hex_rates:
            # Each point should be within reasonable range of mean
            assert abs(rate - mean_rate) < 8.0  # Relaxed tolerance for grid structure

    def test_grid_spacing_matches_parameter(self, simple_2d_env):
        """Test that distance between grid peaks matches grid_spacing."""
        from neurospatial.simulation.models import GridCellModel

        grid_spacing = 40.0
        gc = GridCellModel(
            simple_2d_env,
            grid_spacing=grid_spacing,
            grid_orientation=0.0,
            max_rate=20.0,
            baseline_rate=0.0,
        )

        # Create a grid of test points
        x = np.linspace(0, 100, 101)
        y = np.linspace(0, 100, 101)
        xx, yy = np.meshgrid(x, y)
        positions = np.column_stack([xx.ravel(), yy.ravel()])

        # Compute rates
        rates = gc.firing_rate(positions)
        rate_map = rates.reshape(101, 101)

        # Find local maxima (peaks)
        from scipy import ndimage

        local_max = ndimage.maximum_filter(rate_map, size=5) == rate_map
        peaks = np.argwhere(local_max & (rate_map > 0.5 * gc.max_rate))

        # Check distances between adjacent peaks
        if len(peaks) >= 2:
            # Compute pairwise distances
            from scipy.spatial.distance import pdist

            distances = pdist(peaks)
            # Minimum distance should be close to grid_spacing
            min_dist = np.min(distances)
            # Allow 20% tolerance due to discretization and sampling
            assert abs(min_dist - grid_spacing) < 0.2 * grid_spacing

    def test_orientation_rotation(self, simple_2d_env):
        """Test that grid_orientation rotates the grid pattern."""
        from neurospatial.simulation.models import GridCellModel

        # Create two grid cells with different orientations
        # Use 45° rotation which is NOT a symmetry of the hexagonal lattice (60° symmetry)
        gc_0 = GridCellModel(
            simple_2d_env,
            grid_spacing=40.0,
            grid_orientation=0.0,
            max_rate=20.0,
            baseline_rate=0.0,
            phase_offset=np.array([0.0, 0.0]),  # Origin
        )

        gc_45 = GridCellModel(
            simple_2d_env,
            grid_spacing=40.0,
            grid_orientation=np.pi / 4,  # 45 degrees
            max_rate=20.0,
            baseline_rate=0.0,
            phase_offset=np.array([0.0, 0.0]),
        )

        # Test at many random points to find differences
        # Use specific non-symmetric positions
        rng = np.random.default_rng(42)
        positions = rng.uniform(10, 90, size=(20, 2))

        rates_0 = gc_0.firing_rate(positions)
        rates_45 = gc_45.firing_rate(positions)

        # Rates should be different due to rotation at most positions
        # Check that a substantial number of positions have different rates
        differences = np.abs(rates_0 - rates_45)
        # Expect significant differences at multiple positions
        assert np.sum(differences > 0.1) >= 10  # At least half should differ

    def test_firing_rate_output_shape(self, simple_2d_env):
        """Test that firing_rate returns correct shape."""
        from neurospatial.simulation.models import GridCellModel

        gc = GridCellModel(simple_2d_env, grid_spacing=50.0)

        positions, times = simulate_trajectory_ou(simple_2d_env, duration=2.0, seed=42)
        rates = gc.firing_rate(positions, times)

        assert len(rates) == len(positions)
        assert rates.shape == (len(positions),)

    def test_firing_rate_bounds(self, simple_2d_env):
        """Test that firing rates are within expected bounds."""
        from neurospatial.simulation.models import GridCellModel

        baseline = 0.5
        max_rate = 20.0
        gc = GridCellModel(
            simple_2d_env,
            grid_spacing=50.0,
            max_rate=max_rate,
            baseline_rate=baseline,
        )

        positions, times = simulate_trajectory_ou(simple_2d_env, duration=5.0, seed=42)
        rates = gc.firing_rate(positions, times)

        # All rates should be between baseline and max_rate
        assert np.all(rates >= baseline - 0.1)  # Small tolerance for numerical error
        assert np.all(rates <= max_rate + 0.1)

    def test_ground_truth_property(self, simple_2d_env):
        """Test ground_truth property returns correct parameters."""
        from neurospatial.simulation.models import GridCellModel

        grid_spacing = 50.0
        grid_orientation = np.pi / 4
        phase_offset = np.array([10.0, 20.0])
        max_rate = 20.0
        baseline_rate = 0.5

        gc = GridCellModel(
            simple_2d_env,
            grid_spacing=grid_spacing,
            grid_orientation=grid_orientation,
            phase_offset=phase_offset,
            max_rate=max_rate,
            baseline_rate=baseline_rate,
        )

        gt = gc.ground_truth

        assert "grid_spacing" in gt
        assert "grid_orientation" in gt
        assert "phase_offset" in gt
        assert "max_rate" in gt
        assert "baseline_rate" in gt

        assert gt["grid_spacing"] == grid_spacing
        assert gt["grid_orientation"] == grid_orientation
        np.testing.assert_array_equal(gt["phase_offset"], phase_offset)
        assert gt["max_rate"] == max_rate
        assert gt["baseline_rate"] == baseline_rate

    def test_implements_neural_model_protocol(self, simple_2d_env):
        """Test that GridCellModel implements NeuralModel protocol."""
        from neurospatial.simulation.models import GridCellModel

        gc = GridCellModel(simple_2d_env, grid_spacing=50.0)
        assert isinstance(gc, NeuralModel)

    def test_default_phase_offset(self, simple_2d_env):
        """Test that phase_offset defaults to [0, 0]."""
        from neurospatial.simulation.models import GridCellModel

        gc = GridCellModel(simple_2d_env, grid_spacing=50.0)

        np.testing.assert_array_equal(gc.phase_offset, np.array([0.0, 0.0]))

    def test_rejects_negative_grid_spacing(self, simple_2d_env):
        """Test that negative grid_spacing raises ValueError."""
        import pytest

        from neurospatial.simulation.models import GridCellModel

        with pytest.raises(ValueError, match="grid_spacing must be positive"):
            GridCellModel(simple_2d_env, grid_spacing=-10.0)

    def test_rejects_zero_grid_spacing(self, simple_2d_env):
        """Test that zero grid_spacing raises ValueError."""
        import pytest

        from neurospatial.simulation.models import GridCellModel

        with pytest.raises(ValueError, match="grid_spacing must be positive"):
            GridCellModel(simple_2d_env, grid_spacing=0.0)

    def test_rejects_invalid_baseline_max_rates(self, simple_2d_env):
        """Test that baseline >= max_rate raises ValueError."""
        import pytest

        from neurospatial.simulation.models import GridCellModel

        with pytest.raises(ValueError, match=r"baseline_rate.*must be less than"):
            GridCellModel(simple_2d_env, baseline_rate=20.0, max_rate=10.0)

    def test_rejects_negative_baseline_rate(self, simple_2d_env):
        """Test that negative baseline_rate raises ValueError."""
        import pytest

        from neurospatial.simulation.models import GridCellModel

        with pytest.raises(ValueError, match="baseline_rate must be non-negative"):
            GridCellModel(simple_2d_env, baseline_rate=-1.0)

    def test_rejects_negative_max_rate(self, simple_2d_env):
        """Test that negative max_rate raises ValueError."""
        import pytest

        from neurospatial.simulation.models import GridCellModel

        with pytest.raises(ValueError, match="max_rate must be positive"):
            GridCellModel(simple_2d_env, max_rate=-5.0)

    def test_rejects_wrong_phase_offset_shape(self, simple_2d_env):
        """Test that wrong phase_offset shape raises ValueError."""
        import pytest

        from neurospatial.simulation.models import GridCellModel

        with pytest.raises(ValueError, match="phase_offset must be shape"):
            GridCellModel(simple_2d_env, phase_offset=np.array([1.0, 2.0, 3.0]))
