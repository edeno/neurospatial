"""Tests for neural models."""

import numpy as np
import pytest

from neurospatial import Environment
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

    @pytest.mark.parametrize("bin_size", [1.0, 2.0, 5.0])
    def test_default_width_is_physical_not_a_bin_multiple(self, bin_size):
        """The default field is 11.1 cm whatever the analysis bins."""
        from shapely.geometry import box

        env = Environment.from_polygon(
            box(0, 0, 100, 100), bin_size=bin_size, units="cm"
        )
        pc = PlaceCellModel(env, center=[50.0, 50.0])
        assert pc.width == pytest.approx(11.1)
        assert pc.max_rate == 12.4

    def test_default_width_follows_the_length_unit(self):
        from shapely.geometry import box

        env = Environment.from_polygon(box(0, 0, 1, 1), bin_size=0.02, units="m")
        assert PlaceCellModel(env, center=[0.5, 0.5]).width == pytest.approx(0.111)

    def test_default_width_needs_known_length_units(self):
        """Without metric units the cm default has no value; say how to fix it."""
        from shapely.geometry import box

        env = Environment.from_polygon(box(0, 0, 100, 100), bin_size=2.0)
        with pytest.raises(ValueError, match="needs width") as raised:
            PlaceCellModel(env, center=[50.0, 50.0])
        assert "Fix: pass width=" in str(raised.value)
        assert PlaceCellModel(env, center=[50.0, 50.0], width=8.0).width == 8.0

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

    def test_anisotropic_one_sigma_rate(self, simple_2d_env):
        """Anisotropic 1σ offset in x yields max_rate * exp(-0.5)."""
        center = np.array([50.0, 50.0])
        pc = PlaceCellModel(
            simple_2d_env,
            center=center,
            width=[10.0, 5.0],
            max_rate=20.0,
            baseline_rate=0.0,
        )

        # Offset by exactly the x-width (10.0) along x only -> 1σ in x.
        positions = np.array([[60.0, 50.0]])
        rate = pc.firing_rate(positions, np.array([0.0]))

        expected = 20.0 * np.exp(-0.5)
        assert abs(rate[0] - expected) < 1e-6

    def test_anisotropic_one_sigma_rate_y_axis(self, simple_2d_env):
        """Anisotropic 1σ offset in y yields max_rate * exp(-0.5)."""
        center = np.array([50.0, 50.0])
        pc = PlaceCellModel(
            simple_2d_env,
            center=center,
            width=[10.0, 5.0],
            max_rate=20.0,
            baseline_rate=0.0,
        )

        # Offset by exactly the y-width (5.0) along y only -> 1σ in y.
        positions = np.array([[50.0, 55.0]])
        rate = pc.firing_rate(positions, np.array([0.0]))

        expected = 20.0 * np.exp(-0.5)
        assert abs(rate[0] - expected) < 1e-6

    def test_isotropic_one_sigma_unchanged(self, simple_2d_env):
        """Isotropic 1σ rate is unchanged: max_rate * exp(-0.5)."""
        center = np.array([50.0, 50.0])
        pc = PlaceCellModel(
            simple_2d_env,
            center=center,
            width=10.0,
            max_rate=20.0,
            baseline_rate=0.0,
        )

        positions = np.array([[60.0, 50.0]])  # 1σ in x
        rate = pc.firing_rate(positions, np.array([0.0]))

        expected = 20.0 * np.exp(-0.5)
        assert abs(rate[0] - expected) < 1e-6

    def test_width_zero_raises(self, simple_2d_env):
        """width=0 raises ValueError mentioning 'positive'."""
        with pytest.raises(ValueError, match="positive"):
            PlaceCellModel(simple_2d_env, center=[50.0, 50.0], width=0.0)

    def test_width_negative_raises(self, simple_2d_env):
        """Negative width raises ValueError."""
        with pytest.raises(ValueError):
            PlaceCellModel(simple_2d_env, center=[50.0, 50.0], width=-5.0)

    def test_anisotropic_width_zero_component_raises(self, simple_2d_env):
        """Anisotropic width with a zero component raises ValueError."""
        with pytest.raises(ValueError):
            PlaceCellModel(simple_2d_env, center=[50.0, 50.0], width=[10.0, 0.0])

    def test_width_nan_raises(self, simple_2d_env):
        """NaN width raises ValueError mentioning 'finite'."""
        with pytest.raises(ValueError, match="finite"):
            PlaceCellModel(simple_2d_env, center=[50.0, 50.0], width=float("nan"))

    def test_euclidean_metric(self, simple_2d_env):
        """Test euclidean distance metric (default)."""
        pc = PlaceCellModel(
            simple_2d_env,
            center=[50.0, 50.0],
            width=10.0,
            metric="euclidean",
        )

        positions, times = simulate_trajectory_ou(
            simple_2d_env, duration=1.0, seed=42, speed_units="cm"
        )
        rates = pc.firing_rate(positions, times)

        assert len(rates) == len(positions)
        assert np.all(rates >= 0)  # All rates non-negative

    def test_geodesic_metric(self, simple_2d_env):
        """Test geodesic distance metric."""
        pc = PlaceCellModel(
            simple_2d_env,
            center=[50.0, 50.0],
            width=10.0,
            metric="geodesic",
        )

        positions, times = simulate_trajectory_ou(
            simple_2d_env, duration=1.0, seed=42, speed_units="cm"
        )
        rates = pc.firing_rate(positions, times)

        assert len(rates) == len(positions)
        assert np.all(rates >= 0)  # All rates non-negative

    def test_geodesic_missing_position_rate_is_nan(self, simple_2d_env):
        """A NaN position has no rate; it is not the rate at bin 0."""
        pc = PlaceCellModel(
            simple_2d_env, center=[5.0, 5.0], width=10.0, metric="geodesic"
        )

        rates = pc.firing_rate(np.array([[np.nan, np.nan], [5.0, 5.0]]))

        assert np.isnan(rates[0])
        assert rates[1] == pytest.approx(pc.max_rate)

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

    @pytest.mark.parametrize("width", [5.0, [5.0, 2.0]])
    def test_field_is_gaussian_in_its_tail(self, simple_2d_env, width):
        """The field keeps falling as a Gaussian beyond 5 standard deviations."""
        pc = PlaceCellModel(
            simple_2d_env,
            center=[50.0, 50.0],
            width=width,
            max_rate=20.0,
            baseline_rate=0.0,
        )
        x_width = np.atleast_1d(width)[0]
        sigmas = np.array([4.0, 6.0, 8.0])
        positions = np.column_stack([50.0 + sigmas * x_width, np.full(3, 50.0)])
        np.testing.assert_allclose(
            pc.firing_rate(positions), 20.0 * np.exp(-0.5 * sigmas**2), rtol=1e-12
        )


@pytest.fixture(scope="module")
def walled_box():
    """A walled 100 x 100 cm box with 2 cm bins."""
    from shapely.geometry import box

    return Environment.from_polygon(box(0, 0, 100, 100), bin_size=2.0, units="cm")


class TestBorderCellModel:
    """Firing from the exact distance to the nearest wall."""

    def test_default_tolerance_is_physical(self, walled_box):
        from shapely.geometry import box

        from neurospatial.simulation.models import BorderCellModel

        assert BorderCellModel(walled_box).distance_tolerance == pytest.approx(9.8)
        metres = Environment.from_polygon(box(0, 0, 1, 1), bin_size=0.02, units="m")
        assert BorderCellModel(metres).distance_tolerance == pytest.approx(0.098)

    def test_rate_follows_the_exact_distance_to_the_wall(self, walled_box):
        """No half-bin offset: the distance is measured to the wall itself."""
        from neurospatial.simulation.models import BorderCellModel

        cell = BorderCellModel(
            walled_box, distance_tolerance=3.0, max_rate=20.0, baseline_rate=0.5
        )
        x = np.array([0.25, 0.75, 1.0, 3.0, 7.0])
        positions = np.column_stack([x, np.full_like(x, 50.0)])
        expected = 0.5 + 19.5 * np.exp(-0.5 * (x / 3.0) ** 2)
        np.testing.assert_allclose(cell.firing_rate(positions), expected, rtol=1e-12)

    def test_preferred_distance_shifts_the_peak(self, walled_box):
        from neurospatial.simulation.models import BorderCellModel

        cell = BorderCellModel(
            walled_box,
            distance_tolerance=2.0,
            preferred_distance=10.0,
            baseline_rate=0.0,
        )
        x = np.linspace(0.0, 30.0, 301)
        rates = cell.firing_rate(np.column_stack([x, np.full_like(x, 50.0)]))
        assert x[np.argmax(rates)] == pytest.approx(10.0)
        assert rates.max() == pytest.approx(cell.max_rate)

    def test_true_map_is_a_border_map(self, walled_box):
        from neurospatial.encoding import border_score
        from neurospatial.simulation.models import BorderCellModel

        cell = BorderCellModel(walled_box, distance_tolerance=3.0, baseline_rate=0.0)
        assert border_score(walled_box, cell.firing_rate(walled_box.bin_centers)) > 0.5

    def test_refuses_an_environment_without_walls(self, simple_2d_env):
        """Bins inferred from positions declare no physical wall."""
        from neurospatial.simulation.models import BorderCellModel

        with pytest.raises(
            NotImplementedError, match="Physical boundaries are unknown"
        ):
            BorderCellModel(simple_2d_env, distance_tolerance=3.0)

    def test_rate_is_nan_outside_the_walls(self, walled_box):
        from neurospatial.simulation.models import BorderCellModel

        cell = BorderCellModel(walled_box, distance_tolerance=3.0)
        assert np.isnan(cell.firing_rate(np.array([[150.0, 50.0]]))[0])

    def test_ground_truth_and_protocol(self, walled_box):
        from neurospatial.simulation.models import BorderCellModel

        cell = BorderCellModel(
            walled_box,
            distance_tolerance=3.0,
            preferred_distance=1.0,
            max_rate=12.0,
            baseline_rate=0.2,
        )
        assert isinstance(cell, NeuralModel)
        assert cell.ground_truth == {
            "preferred_distance": 1.0,
            "distance_tolerance": 3.0,
            "max_rate": 12.0,
            "baseline_rate": 0.2,
        }

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"distance_tolerance": 0.0},
            {"distance_tolerance": 3.0, "preferred_distance": -1.0},
            {"distance_tolerance": 3.0, "baseline_rate": 20.0},
        ],
    )
    def test_invalid_parameters_raise(self, walled_box, kwargs):
        from neurospatial.simulation.models import BorderCellModel

        with pytest.raises(ValueError):
            BorderCellModel(walled_box, **kwargs)


class TestBoundaryVectorCellModel:
    """Every visible wall adds firing by its distance and direction."""

    def test_defaults(self, walled_box):
        """17.5 cm, a tolerance growing with distance, and a seeded direction."""
        from neurospatial.simulation.models import BoundaryVectorCellModel

        cell = BoundaryVectorCellModel(walled_box, seed=3)
        assert cell.preferred_distance == pytest.approx(17.5)
        assert cell.distance_tolerance == pytest.approx(17.5 / 12 + 8.0)
        assert cell.direction_tolerance == pytest.approx(np.pi / 6)
        again = BoundaryVectorCellModel(walled_box, seed=3)
        assert again.preferred_direction == cell.preferred_direction
        near = BoundaryVectorCellModel(walled_box, preferred_distance=6.0, seed=3)
        assert near.distance_tolerance == pytest.approx(6.0 / 12 + 8.0)

    @staticmethod
    def _west_cell(env, **kwargs):
        from neurospatial.simulation.models import BoundaryVectorCellModel

        params = {
            "preferred_distance": 10.0,
            "preferred_direction": np.pi,  # West
            "distance_tolerance": 4.0,
            "direction_tolerance": np.pi / 6,
            "max_rate": 15.0,
            "baseline_rate": 0.0,
        }
        params.update(kwargs)
        return BoundaryVectorCellModel(env, **params)

    def test_straight_wall_at_the_preferred_vector_gives_max_rate(self, walled_box):
        cell = self._west_cell(walled_box)
        rate = cell.firing_rate(np.array([[10.0, 50.0]]))[0]
        assert rate == pytest.approx(15.0, rel=1e-3)

    def test_field_lies_at_the_preferred_distance_from_that_wall(self, walled_box):
        cell = self._west_cell(walled_box)
        rates = cell.firing_rate(walled_box.bin_centers)
        middle = np.abs(walled_box.bin_centers[:, 1] - 50.0) < 20.0
        peak = walled_box.bin_centers[middle][np.argmax(rates[middle])]
        assert abs(peak[0] - 10.0) <= 1.0  # half a 2 cm bin
        # Away from the north and south walls, the east side has no wall near
        # the preferred vector: the cell is silent there.
        east = middle & (walled_box.bin_centers[:, 0] > 80.0)
        assert rates[east].max() < 1e-3 * 15.0

    def test_wall_touching_bins_fire_for_a_zero_distance_cell(self, walled_box):
        """The direction to the wall is defined even on the wall's own bins."""
        cell = self._west_cell(walled_box, preferred_distance=0.0)
        assert cell.firing_rate(np.array([[0.5, 50.0]]))[0] > 0.9 * 15.0

    def test_true_map_of_a_wall_hugging_cell_is_a_border_map(self, walled_box):
        from neurospatial.encoding import border_score

        cell = self._west_cell(walled_box, preferred_distance=0.0)
        rates = cell.firing_rate(walled_box.bin_centers)
        assert border_score(walled_box, rates) > 0.5

    def test_a_barrier_adds_a_second_field_beside_it(self, walled_box):
        from shapely.geometry import LineString, box

        from neurospatial.geometry import OpenArena

        arena = box(0, 0, 100, 100)
        barrier = LineString([(50, 20), (50, 80)])  # on a 2 cm bin edge
        env = Environment.from_polygon(
            arena,
            bin_size=2.0,
            units="cm",
            arena=OpenArena(arena, walls={"barrier": barrier}),
        )
        beside = np.array([[60.0, 50.0]])  # 10 cm east of the barrier
        assert self._west_cell(env).firing_rate(beside)[0] == pytest.approx(
            15.0, rel=0.05
        )
        assert self._west_cell(walled_box).firing_rate(beside)[0] < 0.01

    def test_direction_selects_the_wall(self, walled_box):
        west = self._west_cell(walled_box)
        east = self._west_cell(walled_box, preferred_direction=0.0)
        position = np.array([[10.0, 50.0]])
        assert west.firing_rate(position)[0] > 100 * east.firing_rate(position)[0]

    def test_rate_is_nan_outside_the_walls(self, walled_box):
        cell = self._west_cell(walled_box)
        assert np.isnan(cell.firing_rate(np.array([[-5.0, 50.0]]))[0])

    def test_ground_truth_and_protocol(self, walled_box):
        cell = self._west_cell(walled_box, baseline_rate=0.1)
        assert isinstance(cell, NeuralModel)
        assert cell.ground_truth == {
            "preferred_distance": 10.0,
            "preferred_direction": pytest.approx(np.pi),
            "distance_tolerance": 4.0,
            "direction_tolerance": pytest.approx(np.pi / 6),
            "max_rate": 15.0,
            "baseline_rate": 0.1,
        }

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"distance_tolerance": 0.0},
            {"direction_tolerance": 0.0},
            {"direction_tolerance": 4.0},
            {"preferred_direction": np.nan},
            {"preferred_distance": -1.0},
        ],
    )
    def test_invalid_parameters_raise(self, walled_box, kwargs):
        with pytest.raises(ValueError):
            self._west_cell(walled_box, **kwargs)


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

        # grid_orientation=0 puts the field lattice along +x, so the six
        # neighbors one spacing away at 0, 60, ..., 300 degrees are fields.
        np.testing.assert_allclose(hex_rates, center_rate, rtol=1e-9)

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

        positions, times = simulate_trajectory_ou(
            simple_2d_env, duration=2.0, seed=42, speed_units="cm"
        )
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

        positions, times = simulate_trajectory_ou(
            simple_2d_env, duration=5.0, seed=42, speed_units="cm"
        )
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

    @pytest.mark.parametrize(("units", "expected"), [("cm", 41.0), ("m", 0.41)])
    def test_default_spacing_is_the_dorsal_module_in_physical_units(
        self, units, expected
    ):
        from shapely.geometry import box

        from neurospatial.simulation.models import GridCellModel

        side = 100.0 if units == "cm" else 1.0
        env = Environment.from_polygon(
            box(0, 0, side, side), bin_size=side / 50, units=units
        )
        assert GridCellModel(env).grid_spacing == pytest.approx(expected)
        unitless = Environment.from_polygon(box(0, 0, 100, 100), bin_size=2.0)
        with pytest.raises(ValueError, match="needs grid_spacing"):
            GridCellModel(unitless)

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

    def test_firing_rate_matches_reference_loop_implementation(self, simple_2d_env):
        """Vectorized grid pattern must match the explicit per-wave-vector loop.

        Equivalence guard for the loop->matmul refactor of ``firing_rate``:
        the matrix form ``cos(rel_pos @ wave_vectors.T).mean(axis=1)`` must be
        numerically identical to summing three cosine gratings in a Python loop.
        """
        from neurospatial.simulation.models import GridCellModel

        gc = GridCellModel(
            simple_2d_env,
            grid_spacing=37.0,
            grid_orientation=0.4,
            phase_offset=np.array([3.0, -2.0]),
            max_rate=22.0,
            baseline_rate=0.3,
        )

        rng = np.random.default_rng(123)
        positions = rng.uniform(0, 100, size=(500, 2))

        # Reference: original explicit-loop algorithm.
        rel_pos = positions - gc.phase_offset
        k_magnitude = (4.0 * np.pi) / (np.sqrt(3.0) * gc.grid_spacing)
        angles = (
            np.array([0.0, np.pi / 3.0, 2.0 * np.pi / 3.0])
            + gc.grid_orientation
            - np.pi / 6.0
        )
        wave_vectors = k_magnitude * np.column_stack([np.cos(angles), np.sin(angles)])
        grid_pattern = np.zeros(len(rel_pos))
        for k_vec in wave_vectors:
            grid_pattern += np.cos(np.dot(rel_pos, k_vec))
        grid_pattern /= 3.0
        grid_pattern = np.maximum(0.0, grid_pattern)
        expected = gc.baseline_rate + (gc.max_rate - gc.baseline_rate) * grid_pattern

        actual = gc.firing_rate(positions)

        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("orientation_deg", [0.0, 10.0, 25.0, 50.0])
def test_simulated_orientation_is_recovered_by_grid_properties(orientation_deg):
    """Simulating at orientation theta and analysing returns theta."""
    from neurospatial import Environment
    from neurospatial.encoding.grid import (
        autocorrelation_pixel_size,
        grid_properties,
        spatial_autocorrelation,
    )
    from neurospatial.simulation.models.grid_cells import GridCellModel

    positions = np.random.default_rng(0).uniform(0, 120, (30000, 2))
    env = Environment.from_samples(positions, bin_size=2.0)
    grid_orientation = np.radians(orientation_deg)
    model = GridCellModel(env, grid_spacing=30.0, grid_orientation=grid_orientation)
    autocorr = spatial_autocorrelation(env, model.firing_rate(env.bin_centers))

    measured = grid_properties(
        autocorr, pixel_size=autocorrelation_pixel_size(env)
    ).orientation

    error = abs(measured - grid_orientation) % (np.pi / 3)
    assert min(error, np.pi / 3 - error) < np.radians(2.0)


class TestPolarPlaceCellModel:
    """On an egocentric polar environment the field is a Gaussian of real distance."""

    @pytest.fixture
    def surround(self):
        from neurospatial import Environment

        return Environment.from_polar_egocentric(
            distance_range=(0.0, 40.0),
            angle_range=(-np.pi, np.pi),
            distance_bin_size=5.0,
            angle_bin_size=np.pi / 8,
            units="cm",
        )

    def test_rate_falls_off_with_the_distance_between_places(self, surround):
        pc = PlaceCellModel(
            surround,
            center=np.array([10.0, 0.0]),
            width=5.0,
            max_rate=20.0,
            baseline_rate=0.0,
        )
        rows = np.array([[10.0, 0.0], [10.0, 2 * np.pi], [10.0, np.pi], [5.0, 0.0]])
        # 10 cm ahead, the same place a turn later, 10 cm behind (20 cm away),
        # and 5 cm ahead (5 cm away).
        expected = 20.0 * np.exp(-0.5 * (np.array([0.0, 0.0, 20.0, 5.0]) / 5.0) ** 2)
        np.testing.assert_allclose(pc.firing_rate(rows), expected, atol=1e-9)

    def test_per_axis_width_is_refused_with_a_working_alternative(self, surround):
        import neurospatial
        from neurospatial import CartesianOnlyError

        center, width = np.array([10.0, 0.0]), np.array([5.0, 0.3])
        with pytest.raises(CartesianOnlyError) as refused:
            PlaceCellModel(surround, center=center, width=width)
        example = refused.value.example
        assert example is not None and example in str(refused.value)
        model = eval(
            example,
            {
                "neurospatial": neurospatial,
                "env": surround,
                "center": center,
                "width": width,
            },
        )
        assert model.width == 5.0
