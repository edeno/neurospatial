"""Tests for HeadDirectionCellModel."""

import numpy as np
import pytest

from neurospatial.simulation.models import HeadDirectionCellModel, NeuralModel


def _rates(model, headings, positions=None):
    """The model's rate for each heading, sampled at 30 Hz."""
    headings = np.asarray(headings, dtype=np.float64)
    n = len(headings)
    positions = np.zeros((n, 2)) if positions is None else positions
    return model.firing_rate(np.arange(n) / 30.0, positions, headings=headings)


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

    def test_default_tuning_matches_a_90_degree_firing_range(self):
        """Half the rate 22.5 degrees off the peak, as a 90-degree triangle."""
        hd = HeadDirectionCellModel(preferred_direction=0.0, baseline_rate=0.0)
        assert hd.max_rate == 35.0
        assert hd.ground_truth["tuning_width_deg"] == pytest.approx(22.5, abs=0.05)
        rates = _rates(hd, np.radians([0.0, 22.5]))
        assert rates[1] / rates[0] == pytest.approx(0.5, abs=1e-3)

    def test_random_direction_selection(self):
        """Test that direction is randomly chosen if not provided."""
        hd1 = HeadDirectionCellModel(seed=42).ground_truth["preferred_direction"]
        hd2 = HeadDirectionCellModel(seed=42).ground_truth["preferred_direction"]
        hd3 = HeadDirectionCellModel(seed=123).ground_truth["preferred_direction"]

        # Same seed should give same direction
        assert hd1 == hd2

        # Different seeds should (almost certainly) give different directions
        assert hd1 != hd3

        # Direction should be in valid range
        assert -np.pi <= hd1 < np.pi

    def test_direction_wrapping(self):
        """The ground truth wraps the preferred direction to (-pi, pi]."""
        # Large positive angle
        hd1 = HeadDirectionCellModel(preferred_direction=3 * np.pi)
        assert hd1.preferred_direction == 3 * np.pi
        wrapped = hd1.ground_truth["preferred_direction"]
        assert -np.pi < wrapped <= np.pi
        assert wrapped == pytest.approx(np.pi, abs=1e-6)

        # Large negative angle
        hd2 = HeadDirectionCellModel(preferred_direction=-3 * np.pi)
        assert -np.pi < hd2.ground_truth["preferred_direction"] <= np.pi

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
        rates = _rates(hd, headings)
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
        rates = _rates(hd, headings)

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
        rates_broad = _rates(hd_broad, headings)

        # High concentration (sharp)
        hd_sharp = HeadDirectionCellModel(
            preferred_direction=0.0,
            concentration=10.0,
            max_rate=40.0,
            baseline_rate=0.0,
        )
        rates_sharp = _rates(hd_sharp, headings)

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

    @pytest.mark.parametrize(
        ("concentration", "half_width_deg"), [(1, 72), (2, 49), (5, 31), (10, 21)]
    )
    def test_tuning_width_table_for_a_silent_baseline(
        self, concentration, half_width_deg
    ):
        """The concentration table of the class docstring (zero baseline)."""
        hd = HeadDirectionCellModel(
            preferred_direction=0.0, concentration=concentration, baseline_rate=0.0
        )
        assert round(hd.ground_truth["tuning_width_deg"]) == half_width_deg

    @pytest.mark.parametrize("concentration", [1.0, 2.0, 5.0, 10.0])
    @pytest.mark.parametrize("baseline_rate", [0.0, 1.0, 10.0])
    def test_tuning_width_is_where_the_rate_halves(self, concentration, baseline_rate):
        """Half of max_rate, as DirectionalRateResult.tuning_width measures it."""
        hd = HeadDirectionCellModel(
            preferred_direction=0.0,
            concentration=concentration,
            max_rate=40.0,
            baseline_rate=baseline_rate,
        )
        width = hd.ground_truth["tuning_width"]
        np.testing.assert_allclose(_rates(hd, [width, -width]), 20.0, rtol=1e-12)
        assert hd.ground_truth["tuning_width_deg"] == pytest.approx(np.degrees(width))

    @pytest.mark.parametrize(
        ("concentration", "baseline_rate"),
        [(0.2, 0.0), (5.0, 20.0), (5.0, 30.0), (1.0, 40.0)],
    )
    def test_tuning_width_is_nan_when_the_rate_never_halves(
        self, concentration, baseline_rate
    ):
        """A broad curve, or a baseline at or above half the peak, never halves."""
        hd = HeadDirectionCellModel(
            preferred_direction=0.0,
            concentration=concentration,
            max_rate=40.0,
            baseline_rate=baseline_rate,
        )
        headings = np.linspace(-np.pi, np.pi, 3601)
        assert _rates(hd, headings).min() >= 20.0
        assert np.isnan(hd.ground_truth["tuning_width"])
        assert np.isnan(hd.ground_truth["tuning_width_deg"])

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
        """With a silent baseline, the von Mises HWHM formula."""
        concentration = 2.0
        hd = HeadDirectionCellModel(
            preferred_direction=0.0, concentration=concentration, baseline_rate=0.0
        )

        # Theoretical HWHM: arccos(1 - ln(2)/kappa)
        expected_hwhm = np.arccos(1 - np.log(2) / concentration)
        assert hd.ground_truth["tuning_width"] == pytest.approx(expected_hwhm, abs=1e-6)

    def test_mean_vector_length_of_a_silent_baseline_is_von_mises(self):
        from scipy.special import i0, i1

        hd = HeadDirectionCellModel(
            preferred_direction=0.0, concentration=3.0, baseline_rate=0.0
        )
        assert hd.mean_vector_length == pytest.approx(i1(3.0) / i0(3.0), rel=1e-12)

    @pytest.mark.parametrize("concentration", [0.5, 3.0, 9.1, 800.0])
    @pytest.mark.parametrize("baseline_rate", [0.0, 0.5, 5.0, 20.0])
    def test_mean_vector_length_matches_the_tuning_curve(
        self, concentration, baseline_rate
    ):
        """The baseline fires in every direction, so it lowers the length."""
        hd = HeadDirectionCellModel(
            preferred_direction=1.0,
            concentration=concentration,
            max_rate=40.0,
            baseline_rate=baseline_rate,
        )
        headings = np.linspace(-np.pi, np.pi, 200_000, endpoint=False)
        rates = _rates(hd, headings)
        numerical = np.abs(np.sum(rates * np.exp(1j * headings))) / np.sum(rates)
        assert hd.mean_vector_length == pytest.approx(numerical, rel=1e-6, abs=1e-12)

    @pytest.mark.parametrize("concentration", [0.0, -1.0, np.nan, np.inf])
    def test_invalid_concentration_raises(self, concentration):
        with pytest.raises(
            ValueError, match="concentration must be finite and positive"
        ):
            HeadDirectionCellModel(concentration=concentration)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"max_rate": -10.0}, "max_rate must be a finite positive"),
            ({"max_rate": np.nan}, "max_rate must be a finite positive"),
            ({"baseline_rate": -1.0}, "baseline_rate must be a finite non-negative"),
            ({"baseline_rate": np.nan}, "baseline_rate must be a finite non-negative"),
            ({"max_rate": 10.0, "baseline_rate": 20.0}, "is above max_rate"),
            ({"preferred_direction": np.nan}, "preferred_direction must be a finite"),
        ],
    )
    def test_invalid_parameters_raise(self, kwargs, match):
        """NaN would otherwise pass every comparison and give NaN rates."""
        with pytest.raises(ValueError, match=match):
            HeadDirectionCellModel(**kwargs)

    def test_headings_are_required(self):
        hd = HeadDirectionCellModel(preferred_direction=0.0)
        with pytest.raises(ValueError, match="needs headings") as caught:
            hd.firing_rate(np.zeros(3), np.zeros((3, 2)))
        assert "generate_population_spikes" in str(caught.value)

    def test_direction_vectors_are_refused(self):
        """(n, 2) headings would be flattened into 2n rates."""
        hd = HeadDirectionCellModel(preferred_direction=0.0)
        vectors = np.column_stack([np.ones(5), np.zeros(5)])
        with pytest.raises(ValueError, match=r"headings of shape \(5, 2\)") as caught:
            hd.firing_rate(np.zeros(5), np.zeros((5, 2)), headings=vectors)
        assert "np.arctan2" in str(caught.value)

    def test_one_position_per_time(self):
        hd = HeadDirectionCellModel(preferred_direction=0.0)
        with pytest.raises(ValueError, match="5 times but 4 positions"):
            hd.firing_rate(np.zeros(5), np.zeros((4, 2)), headings=np.zeros(5))

    def test_missing_heading_gives_nan(self):
        hd = HeadDirectionCellModel(preferred_direction=0.0)
        rates = _rates(hd, [0.0, np.nan])
        assert np.isfinite(rates[0])
        assert np.isnan(rates[1])

    def test_firing_rate_with_headings_array(self):
        """Test firing rate with array of headings."""
        hd = HeadDirectionCellModel(preferred_direction=0.0)

        # Generate headings
        headings = np.linspace(-np.pi, np.pi, 100)
        rates = _rates(hd, headings)

        assert len(rates) == len(headings)
        assert np.all(rates >= hd.baseline_rate)
        assert np.all(rates <= hd.max_rate)

    def test_rate_follows_headings_not_positions(self, sample_positions):
        """The path runs anywhere, the head points at the preferred direction."""
        hd = HeadDirectionCellModel(
            preferred_direction=0.0,
            concentration=5.0,
            max_rate=40.0,
            baseline_rate=0.0,
        )

        # Facing East throughout, wherever the animal is and wherever it goes.
        rates = _rates(hd, np.zeros(100), positions=sample_positions[:100])
        np.testing.assert_allclose(rates, 40.0, rtol=1e-12)

        # The same path facing West: the rate is near baseline.
        rates = _rates(hd, np.full(100, np.pi), positions=sample_positions[:100])
        np.testing.assert_allclose(rates, 40.0 * np.exp(-10.0), rtol=1e-12)

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
            assert cell.ground_truth["preferred_direction"] == pytest.approx(
                wrapped_dir, abs=1e-6
            )

        # Population response
        headings = np.linspace(-np.pi, np.pi, 360)
        population_rates = np.column_stack(
            [_rates(cell, headings) for cell in hd_cells]
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
