"""Tests for HeadDirectionCellModel."""

import numpy as np
import pytest

from neurospatial.simulation.models import HeadDirectionCellModel, NeuralModel


class TestHeadDirectionCellModel:
    """Tests for HeadDirectionCellModel."""

    def test_implements_neural_model_protocol(self):
        """Test that HeadDirectionCellModel implements NeuralModel protocol."""
        hd = HeadDirectionCellModel(preferred_direction=0.0)
        assert isinstance(hd, NeuralModel)

    def test_basic_initialization(self):
        """Test basic initialization with explicit parameters."""
        hd = HeadDirectionCellModel(
            preferred_direction=np.pi / 2,
            concentration=3.0,
            max_rate=50.0,
            baseline_rate=2.0,
        )

        assert hd.preferred_direction == pytest.approx(np.pi / 2, abs=1e-6)
        assert hd.concentration == 3.0
        assert hd.max_rate == 50.0
        assert hd.baseline_rate == 2.0

    def test_random_direction_selection(self):
        """Test that direction is randomly chosen if not provided."""
        hd1 = HeadDirectionCellModel(seed=42)
        hd2 = HeadDirectionCellModel(seed=42)
        hd3 = HeadDirectionCellModel(seed=123)

        # Same seed should give same direction
        assert hd1.preferred_direction == hd2.preferred_direction

        # Different seeds should (almost certainly) give different directions
        assert hd1.preferred_direction != hd3.preferred_direction

        # Direction should be in valid range
        assert -np.pi <= hd1.preferred_direction < np.pi

    def test_direction_wrapping(self):
        """Test that preferred direction is wrapped to [-pi, pi)."""
        # Large positive angle
        hd1 = HeadDirectionCellModel(preferred_direction=3 * np.pi)
        assert -np.pi <= hd1.preferred_direction < np.pi
        assert hd1.preferred_direction == pytest.approx(np.pi, abs=1e-6)

        # Large negative angle
        hd2 = HeadDirectionCellModel(preferred_direction=-3 * np.pi)
        assert -np.pi <= hd2.preferred_direction < np.pi

    def test_peak_firing_at_preferred_direction(self):
        """Test that firing rate peaks at preferred direction."""
        preferred = np.pi / 4  # 45 degrees
        hd = HeadDirectionCellModel(
            preferred_direction=preferred,
            concentration=5.0,
            max_rate=40.0,
            baseline_rate=1.0,
        )

        # Test at preferred direction
        headings = np.array([preferred])
        rates = hd.firing_rate(headings=headings)
        assert rates[0] == pytest.approx(40.0, abs=1e-6)

    def test_von_mises_tuning_curve(self):
        """Test that tuning curve follows von Mises distribution."""
        preferred = 0.0
        concentration = 3.0
        max_rate = 40.0
        baseline_rate = 1.0

        hd = HeadDirectionCellModel(
            preferred_direction=preferred,
            concentration=concentration,
            max_rate=max_rate,
            baseline_rate=baseline_rate,
        )

        # Test at various angles
        headings = np.linspace(-np.pi, np.pi, 100)
        rates = hd.firing_rate(headings=headings)

        # Peak should be at preferred direction (index ~50)
        peak_idx = np.argmax(rates)
        assert abs(headings[peak_idx] - preferred) < 0.1

        # Rates should be symmetric around preferred direction
        # Compare left and right sides
        half = len(headings) // 2
        left_rates = rates[:half]
        right_rates = rates[half:][::-1]
        np.testing.assert_allclose(left_rates, right_rates, rtol=0.1)

        # All rates should be between baseline and max
        assert np.all(rates >= baseline_rate)
        assert np.all(rates <= max_rate)

    def test_concentration_affects_tuning_width(self):
        """Test that higher concentration gives sharper tuning."""
        headings = np.linspace(-np.pi, np.pi, 360)

        # Low concentration (broad)
        hd_broad = HeadDirectionCellModel(
            preferred_direction=0.0,
            concentration=1.0,
            max_rate=40.0,
            baseline_rate=0.0,
        )
        rates_broad = hd_broad.firing_rate(headings=headings)

        # High concentration (sharp)
        hd_sharp = HeadDirectionCellModel(
            preferred_direction=0.0,
            concentration=10.0,
            max_rate=40.0,
            baseline_rate=0.0,
        )
        rates_sharp = hd_sharp.firing_rate(headings=headings)

        # Sharp tuning should have fewer bins above half-max rate
        half_max = 20.0  # half of max_rate
        broad_above_half = np.sum(rates_broad > half_max)
        sharp_above_half = np.sum(rates_sharp > half_max)
        assert sharp_above_half < broad_above_half

        # Also verify tuning widths from ground_truth
        assert (
            hd_sharp.ground_truth["tuning_width"]
            < hd_broad.ground_truth["tuning_width"]
        )

    def test_ground_truth_properties(self):
        """Test that ground_truth returns correct parameters."""
        hd = HeadDirectionCellModel(
            preferred_direction=np.pi / 3,
            concentration=4.0,
            max_rate=60.0,
            baseline_rate=5.0,
        )

        gt = hd.ground_truth

        assert gt["preferred_direction"] == pytest.approx(np.pi / 3, abs=1e-6)
        assert gt["preferred_direction_deg"] == pytest.approx(60.0, abs=0.1)
        assert gt["concentration"] == 4.0
        assert gt["max_rate"] == 60.0
        assert gt["baseline_rate"] == 5.0
        assert "tuning_width" in gt
        assert "tuning_width_deg" in gt

    def test_tuning_width_formula(self):
        """Test that tuning width follows von Mises HWHM formula."""
        concentration = 2.0
        hd = HeadDirectionCellModel(
            preferred_direction=0.0,
            concentration=concentration,
        )

        # Theoretical HWHM: arccos(1 - ln(2)/kappa)
        expected_hwhm = np.arccos(1 - np.log(2) / concentration)
        assert hd.ground_truth["tuning_width"] == pytest.approx(expected_hwhm, abs=1e-6)

    def test_mean_vector_length_property(self):
        """Test mean vector length matches theoretical value."""
        from scipy.special import i0, i1

        concentration = 3.0
        hd = HeadDirectionCellModel(
            preferred_direction=0.0,
            concentration=concentration,
        )

        # Theoretical MVL: I1(kappa) / I0(kappa)
        expected_mvl = i1(concentration) / i0(concentration)
        assert hd.mean_vector_length == pytest.approx(expected_mvl, abs=1e-6)

    def test_invalid_concentration_raises(self):
        """Test that non-positive concentration raises error."""
        with pytest.raises(ValueError, match="concentration must be positive"):
            HeadDirectionCellModel(concentration=0.0)

        with pytest.raises(ValueError, match="concentration must be positive"):
            HeadDirectionCellModel(concentration=-1.0)

    def test_invalid_rates_raise(self):
        """Test that invalid rate values raise errors."""
        with pytest.raises(ValueError, match="max_rate must be non-negative"):
            HeadDirectionCellModel(max_rate=-10.0)

        with pytest.raises(ValueError, match="baseline_rate must be non-negative"):
            HeadDirectionCellModel(baseline_rate=-1.0)

        with pytest.raises(ValueError, match=r"baseline_rate.*cannot exceed max_rate"):
            HeadDirectionCellModel(max_rate=10.0, baseline_rate=20.0)

    def test_firing_rate_with_positions_and_times(self, sample_positions, sample_times):
        """Test computing headings from positions."""
        hd = HeadDirectionCellModel(preferred_direction=0.0)

        # Should compute headings from velocity
        rates = hd.firing_rate(positions=sample_positions, times=sample_times)

        assert len(rates) == len(sample_times)
        assert np.all(rates >= hd.baseline_rate)
        assert np.all(rates <= hd.max_rate)

    def test_firing_rate_requires_headings_or_positions(self):
        """Test that either headings or positions must be provided."""
        hd = HeadDirectionCellModel()

        with pytest.raises(ValueError, match="Either headings or positions"):
            hd.firing_rate()

    def test_firing_rate_requires_times_with_positions(self, sample_positions):
        """Test that times must be provided when using positions."""
        hd = HeadDirectionCellModel()

        with pytest.raises(ValueError, match="times must be provided"):
            hd.firing_rate(positions=sample_positions)

    def test_headings_preferred_over_positions(self, sample_times):
        """Test that explicit headings are used when provided."""
        hd = HeadDirectionCellModel(
            preferred_direction=0.0,
            concentration=5.0,
            max_rate=40.0,
            baseline_rate=0.0,
        )

        # All headings at preferred direction should give max rate
        headings = np.zeros(100)
        positions = np.random.rand(100, 2) * 100  # Random positions

        rates = hd.firing_rate(
            positions=positions, times=sample_times[:100], headings=headings
        )

        # Should use headings, not compute from positions
        np.testing.assert_allclose(rates, 40.0, atol=1e-6)

    def test_population_of_hd_cells(self):
        """Test creating a population of HD cells with uniform preferred directions."""
        n_cells = 8
        preferred_dirs = np.linspace(0, 2 * np.pi, n_cells, endpoint=False)

        hd_cells = [
            HeadDirectionCellModel(preferred_direction=d, concentration=2.0)
            for d in preferred_dirs
        ]

        # Each cell should have correct preferred direction (wrapped)
        for i, cell in enumerate(hd_cells):
            wrapped_dir = np.arctan2(
                np.sin(preferred_dirs[i]), np.cos(preferred_dirs[i])
            )
            assert cell.preferred_direction == pytest.approx(wrapped_dir, abs=1e-6)

        # Population response
        headings = np.linspace(-np.pi, np.pi, 360)
        population_rates = np.column_stack(
            [cell.firing_rate(headings=headings) for cell in hd_cells]
        )

        assert population_rates.shape == (360, n_cells)

        # Each heading should activate different cells
        # Check that the winning cell rotates as heading changes
        winning_cells = np.argmax(population_rates, axis=1)
        unique_winners = len(np.unique(winning_cells))
        assert unique_winners == n_cells  # All cells should win at some point


@pytest.fixture
def sample_positions():
    """Create sample trajectory positions for testing."""
    rng = np.random.default_rng(42)
    return rng.uniform(0, 100, size=(1000, 2))


@pytest.fixture
def sample_times():
    """Create sample time points for testing."""
    return np.linspace(0, 10, 1000)
