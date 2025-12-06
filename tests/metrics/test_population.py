"""Tests for population-level place field metrics."""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.encoding.population import (
    PopulationCoverageResult,
    count_place_cells,
    field_density_map,
    field_overlap,
    plot_population_coverage,
    population_coverage,
    population_vector_correlation,
)

matplotlib.use("Agg")  # Use non-interactive backend for testing


class TestPopulationCoverageResult:
    """Test PopulationCoverageResult dataclass."""

    def test_dataclass_frozen(self) -> None:
        """Test that PopulationCoverageResult is immutable."""
        result = PopulationCoverageResult(
            coverage_fraction=0.5,
            is_covered=np.array([True, False]),
            field_count=np.array([1, 0]),
            covered_bins=np.array([0]),
            uncovered_bins=np.array([1]),
            uncovered_positions=np.array([[1.0, 1.0]]),
            n_neurons=1,
            n_place_cells=1,
            n_fields=1,
            place_fields=[[np.array([0])]],
        )
        with pytest.raises(AttributeError):
            result.coverage_fraction = 0.9  # type: ignore[misc]

    def test_str_representation(self) -> None:
        """Test human-readable string representation."""
        result = PopulationCoverageResult(
            coverage_fraction=0.75,
            is_covered=np.array([True, True, True, False]),
            field_count=np.array([2, 1, 1, 0]),
            covered_bins=np.array([0, 1, 2]),
            uncovered_bins=np.array([3]),
            uncovered_positions=np.array([[10.0, 10.0]]),
            n_neurons=5,
            n_place_cells=3,
            n_fields=4,
            place_fields=[[], [], [], [], []],
        )
        result_str = str(result)
        assert "75.0%" in result_str
        assert "3/5" in result_str
        assert "60.0%" in result_str  # place_cell_fraction
        assert "1.3" in result_str  # fields_per_place_cell
        assert "1 bins" in result_str  # gaps

    def test_place_cell_fraction(self) -> None:
        """Test place_cell_fraction computed property."""
        result = PopulationCoverageResult(
            coverage_fraction=0.5,
            is_covered=np.array([True, False]),
            field_count=np.array([1, 0]),
            covered_bins=np.array([0]),
            uncovered_bins=np.array([1]),
            uncovered_positions=np.array([[1.0, 1.0]]),
            n_neurons=10,
            n_place_cells=4,
            n_fields=5,
            place_fields=[],
        )
        assert result.place_cell_fraction == pytest.approx(0.4)

    def test_place_cell_fraction_zero_neurons(self) -> None:
        """Test place_cell_fraction when n_neurons is zero."""
        result = PopulationCoverageResult(
            coverage_fraction=0.0,
            is_covered=np.array([False]),
            field_count=np.array([0]),
            covered_bins=np.array([], dtype=np.intp),
            uncovered_bins=np.array([0]),
            uncovered_positions=np.array([[1.0, 1.0]]),
            n_neurons=0,
            n_place_cells=0,
            n_fields=0,
            place_fields=[],
        )
        assert result.place_cell_fraction == 0.0

    def test_fields_per_place_cell(self) -> None:
        """Test fields_per_place_cell computed property."""
        result = PopulationCoverageResult(
            coverage_fraction=0.5,
            is_covered=np.array([True, False]),
            field_count=np.array([1, 0]),
            covered_bins=np.array([0]),
            uncovered_bins=np.array([1]),
            uncovered_positions=np.array([[1.0, 1.0]]),
            n_neurons=10,
            n_place_cells=4,
            n_fields=6,
            place_fields=[],
        )
        assert result.fields_per_place_cell == pytest.approx(1.5)

    def test_fields_per_place_cell_zero_place_cells(self) -> None:
        """Test fields_per_place_cell when no place cells detected."""
        result = PopulationCoverageResult(
            coverage_fraction=0.0,
            is_covered=np.array([False]),
            field_count=np.array([0]),
            covered_bins=np.array([], dtype=np.intp),
            uncovered_bins=np.array([0]),
            uncovered_positions=np.array([[1.0, 1.0]]),
            n_neurons=5,
            n_place_cells=0,
            n_fields=0,
            place_fields=[],
        )
        assert result.fields_per_place_cell == 0.0

    def test_mean_redundancy(self) -> None:
        """Test mean_redundancy computed property."""
        result = PopulationCoverageResult(
            coverage_fraction=0.75,
            is_covered=np.array([True, True, True, False]),
            field_count=np.array([3, 2, 1, 0]),  # Mean of [3, 2, 1] = 2.0
            covered_bins=np.array([0, 1, 2]),
            uncovered_bins=np.array([3]),
            uncovered_positions=np.array([[10.0, 10.0]]),
            n_neurons=5,
            n_place_cells=3,
            n_fields=6,
            place_fields=[],
        )
        assert result.mean_redundancy == pytest.approx(2.0)

    def test_mean_redundancy_no_coverage(self) -> None:
        """Test mean_redundancy when no bins are covered."""
        result = PopulationCoverageResult(
            coverage_fraction=0.0,
            is_covered=np.array([False, False]),
            field_count=np.array([0, 0]),
            covered_bins=np.array([], dtype=np.intp),
            uncovered_bins=np.array([0, 1]),
            uncovered_positions=np.array([[1.0, 1.0], [2.0, 2.0]]),
            n_neurons=5,
            n_place_cells=0,
            n_fields=0,
            place_fields=[],
        )
        assert result.mean_redundancy == 0.0


class TestPopulationCoverage:
    """Test population_coverage function."""

    @pytest.fixture
    def simple_env(self) -> Environment:
        """Create a simple 2D environment for testing."""
        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 100, (1000, 2))
        return Environment.from_samples(positions, bin_size=10.0)

    def test_full_coverage(self, simple_env: Environment) -> None:
        """Test when all bins are covered by place fields."""
        n_bins = simple_env.n_bins
        # Create firing rates with high peaks for each bin
        # This ensures place fields cover all bins
        firing_rates = np.zeros((n_bins, n_bins))
        for i in range(n_bins):
            firing_rates[i, i] = 10.0  # Peak at bin i
            # Add some neighbors
            for j in simple_env.neighbors(i):
                firing_rates[i, j] = 5.0

        result = population_coverage(firing_rates, simple_env, min_size=1)
        assert result.coverage_fraction == pytest.approx(1.0)
        assert len(result.uncovered_bins) == 0
        assert result.n_neurons == n_bins

    def test_partial_coverage(self, simple_env: Environment) -> None:
        """Test partial coverage with some gaps."""
        n_bins = simple_env.n_bins
        # Create firing rates for only half the environment
        n_neurons = n_bins // 2
        firing_rates = np.zeros((n_neurons, n_bins))
        for i in range(n_neurons):
            firing_rates[i, i] = 10.0
            for j in simple_env.neighbors(i):
                if j < n_bins:
                    firing_rates[i, j] = 5.0

        result = population_coverage(firing_rates, simple_env, min_size=1)
        assert 0.0 < result.coverage_fraction < 1.0
        assert len(result.uncovered_bins) > 0
        assert len(result.uncovered_positions) == len(result.uncovered_bins)

    def test_zero_coverage(self, simple_env: Environment) -> None:
        """Test when no place fields are detected (all neurons are putative interneurons)."""
        n_bins = simple_env.n_bins
        # Create high mean firing rate (above max_mean_rate threshold)
        # This will cause detect_place_fields to classify them as interneurons
        firing_rates = np.ones((5, n_bins)) * 15.0  # Above default max_mean_rate=10

        result = population_coverage(firing_rates, simple_env)
        assert result.coverage_fraction == 0.0
        assert len(result.uncovered_bins) == n_bins
        assert result.n_place_cells == 0
        assert result.n_fields == 0

    def test_single_neuron(self, simple_env: Environment) -> None:
        """Test with a single neuron."""
        n_bins = simple_env.n_bins
        # Single neuron with a place field
        firing_rates = np.zeros((1, n_bins))
        firing_rates[0, 0] = 10.0
        for j in simple_env.neighbors(0):
            firing_rates[0, j] = 5.0

        result = population_coverage(firing_rates, simple_env, min_size=1)
        assert result.n_neurons == 1
        if result.n_place_cells > 0:
            assert result.n_place_cells == 1
            assert len(result.place_fields[0]) > 0

    def test_field_count_matches_density_map(self, simple_env: Environment) -> None:
        """Verify field_count matches field_density_map output."""
        n_bins = simple_env.n_bins
        firing_rates = np.zeros((3, n_bins))
        # Create overlapping fields
        firing_rates[0, 0:5] = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
        firing_rates[1, 3:8] = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
        firing_rates[2, 6:11] = np.array([10.0, 8.0, 6.0, 4.0, 2.0])

        result = population_coverage(firing_rates, simple_env, min_size=1)
        expected_density = field_density_map(result.place_fields, n_bins)
        np.testing.assert_array_equal(result.field_count, expected_density)

    def test_uncovered_positions_match_bin_centers(
        self, simple_env: Environment
    ) -> None:
        """Verify uncovered_positions matches env.bin_centers[uncovered_bins]."""
        n_bins = simple_env.n_bins
        # Low uniform firing (no fields detected)
        firing_rates = np.ones((5, n_bins)) * 0.1

        result = population_coverage(firing_rates, simple_env)
        expected_positions = simple_env.bin_centers[result.uncovered_bins]
        np.testing.assert_array_equal(result.uncovered_positions, expected_positions)

    def test_n_place_cells_count(self, simple_env: Environment) -> None:
        """Verify n_place_cells counts neurons with at least one field."""
        n_bins = simple_env.n_bins
        firing_rates = np.zeros((5, n_bins))
        # Only first 3 neurons have place fields
        for i in range(3):
            firing_rates[i, i * 10 : (i + 1) * 10] = np.linspace(10.0, 1.0, 10)

        result = population_coverage(firing_rates, simple_env, min_size=1)
        neurons_with_fields = sum(
            1 for fields in result.place_fields if len(fields) > 0
        )
        assert result.n_place_cells == neurons_with_fields

    def test_place_fields_format(self, simple_env: Environment) -> None:
        """Verify place_fields structure matches detect_place_fields output."""
        n_bins = simple_env.n_bins
        firing_rates = np.zeros((2, n_bins))
        firing_rates[0, 0:5] = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
        firing_rates[1, 10:15] = np.array([10.0, 8.0, 6.0, 4.0, 2.0])

        result = population_coverage(firing_rates, simple_env, min_size=1)
        # Should have list of length n_neurons
        assert len(result.place_fields) == 2
        # Each element is a list of arrays
        for neuron_fields in result.place_fields:
            assert isinstance(neuron_fields, list)
            for field in neuron_fields:
                assert isinstance(field, np.ndarray)
                assert field.dtype == np.int64

    # Validation tests
    def test_environment_not_fitted_error(self) -> None:
        """Test error when environment is not fitted."""
        from unittest.mock import MagicMock

        env = MagicMock(spec=Environment)
        env._is_fitted = False
        firing_rates = np.random.rand(5, 10)
        with pytest.raises(RuntimeError, match="Environment must be fitted"):
            population_coverage(firing_rates, env)

    def test_shape_mismatch_error(self, simple_env: Environment) -> None:
        """Test error when firing_rates shape doesn't match n_bins."""
        wrong_n_bins = simple_env.n_bins + 10
        firing_rates = np.random.rand(5, wrong_n_bins)
        with pytest.raises(ValueError, match="firing_rates shape mismatch"):
            population_coverage(firing_rates, simple_env)

    def test_non_2d_input_error(self, simple_env: Environment) -> None:
        """Test error for non-2D firing_rates."""
        firing_rates_1d = np.random.rand(simple_env.n_bins)
        with pytest.raises(ValueError, match="firing_rates must be 2D"):
            population_coverage(firing_rates_1d, simple_env)  # type: ignore[arg-type]

        firing_rates_3d = np.random.rand(5, simple_env.n_bins, 3)
        with pytest.raises(ValueError, match="firing_rates must be 2D"):
            population_coverage(firing_rates_3d, simple_env)  # type: ignore[arg-type]

    def test_threshold_out_of_range_error(self, simple_env: Environment) -> None:
        """Test error for threshold outside (0, 1)."""
        n_bins = simple_env.n_bins
        firing_rates = np.random.rand(5, n_bins)

        with pytest.raises(ValueError, match="threshold must be in range"):
            population_coverage(firing_rates, simple_env, threshold=0.0)

        with pytest.raises(ValueError, match="threshold must be in range"):
            population_coverage(firing_rates, simple_env, threshold=1.0)

        with pytest.raises(ValueError, match="threshold must be in range"):
            population_coverage(firing_rates, simple_env, threshold=-0.1)

        with pytest.raises(ValueError, match="threshold must be in range"):
            population_coverage(firing_rates, simple_env, threshold=1.5)

    def test_max_mean_rate_non_positive_error(self, simple_env: Environment) -> None:
        """Test error for non-positive max_mean_rate."""
        n_bins = simple_env.n_bins
        firing_rates = np.random.rand(5, n_bins)

        with pytest.raises(ValueError, match="max_mean_rate must be positive"):
            population_coverage(firing_rates, simple_env, max_mean_rate=0.0)

        with pytest.raises(ValueError, match="max_mean_rate must be positive"):
            population_coverage(firing_rates, simple_env, max_mean_rate=-1.0)


class TestPlotPopulationCoverage:
    """Test plot_population_coverage function."""

    @pytest.fixture
    def env_and_result(self) -> tuple[Environment, PopulationCoverageResult]:
        """Create a 2D environment and coverage result for testing."""
        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 100, (1000, 2))
        env = Environment.from_samples(positions, bin_size=10.0)
        n_bins = env.n_bins

        # Create firing rates with partial coverage
        firing_rates = np.zeros((5, n_bins))
        for i in range(3):
            firing_rates[i, i * 10 : (i + 1) * 10] = np.linspace(10.0, 1.0, 10)

        result = population_coverage(firing_rates, env, min_size=1)
        return env, result

    @pytest.fixture
    def env_1d_and_result(self) -> tuple[Environment, PopulationCoverageResult]:
        """Create a 1D (graph-based) environment and coverage result for testing."""
        import networkx as nx

        # Create a simple linear track graph
        graph = nx.Graph()
        graph.add_node("A", pos=(0.0,))
        graph.add_node("B", pos=(100.0,))
        graph.add_edge("A", "B", distance=100.0)  # Must have distance attribute

        env = Environment.from_graph(
            graph=graph,
            edge_order=[("A", "B")],
            edge_spacing=0.0,
            bin_size=10.0,
        )
        n_bins = env.n_bins

        # Create firing rates with partial coverage
        firing_rates = np.zeros((3, n_bins))
        # Ensure fields are large enough
        firing_rates[0, 0:3] = np.array([10.0, 8.0, 6.0])
        firing_rates[1, 5:8] = np.array([10.0, 8.0, 6.0])

        result = population_coverage(firing_rates, env, min_size=1)
        return env, result

    def test_binary_plot(
        self, env_and_result: tuple[Environment, PopulationCoverageResult]
    ) -> None:
        """Test binary coverage plot (show_field_count=False)."""
        env, result = env_and_result
        ax = plot_population_coverage(env, result, show_field_count=False)
        assert isinstance(ax, plt.Axes)
        plt.close()

    def test_field_count_plot(
        self, env_and_result: tuple[Environment, PopulationCoverageResult]
    ) -> None:
        """Test field count plot (show_field_count=True)."""
        env, result = env_and_result
        ax = plot_population_coverage(env, result, show_field_count=True)
        assert isinstance(ax, plt.Axes)
        plt.close()

    def test_gap_highlighting_on(
        self, env_and_result: tuple[Environment, PopulationCoverageResult]
    ) -> None:
        """Test gap highlighting enabled."""
        env, result = env_and_result
        ax = plot_population_coverage(env, result, highlight_gaps=True)
        assert isinstance(ax, plt.Axes)
        plt.close()

    def test_gap_highlighting_off(
        self, env_and_result: tuple[Environment, PopulationCoverageResult]
    ) -> None:
        """Test gap highlighting disabled."""
        env, result = env_and_result
        ax = plot_population_coverage(env, result, highlight_gaps=False)
        assert isinstance(ax, plt.Axes)
        plt.close()

    def test_1d_environment_plotting(
        self, env_1d_and_result: tuple[Environment, PopulationCoverageResult]
    ) -> None:
        """Test plotting for 1D environments."""
        env, result = env_1d_and_result
        ax = plot_population_coverage(env, result)
        assert isinstance(ax, plt.Axes)
        plt.close()

    def test_1d_environment_field_count(
        self, env_1d_and_result: tuple[Environment, PopulationCoverageResult]
    ) -> None:
        """Test 1D plotting with field count."""
        env, result = env_1d_and_result
        ax = plot_population_coverage(env, result, show_field_count=True)
        assert isinstance(ax, plt.Axes)
        plt.close()

    def test_custom_axes(
        self, env_and_result: tuple[Environment, PopulationCoverageResult]
    ) -> None:
        """Test passing custom axes."""
        env, result = env_and_result
        _, custom_ax = plt.subplots()
        returned_ax = plot_population_coverage(env, result, ax=custom_ax)
        assert returned_ax is custom_ax
        plt.close()

    def test_environment_not_fitted_error(self) -> None:
        """Test error when environment is not fitted."""
        from unittest.mock import MagicMock

        env = MagicMock(spec=Environment)
        env._is_fitted = False
        # Create a minimal valid result
        result = PopulationCoverageResult(
            coverage_fraction=0.5,
            is_covered=np.array([True]),
            field_count=np.array([1]),
            covered_bins=np.array([0]),
            uncovered_bins=np.array([]),
            uncovered_positions=np.array([]).reshape(0, 2),
            n_neurons=1,
            n_place_cells=1,
            n_fields=1,
            place_fields=[[np.array([0])]],
        )
        with pytest.raises(RuntimeError, match="Environment must be fitted"):
            plot_population_coverage(env, result)

    def test_wrong_result_type_error(
        self, env_and_result: tuple[Environment, PopulationCoverageResult]
    ) -> None:
        """Test error when result is not PopulationCoverageResult."""
        env, _ = env_and_result
        with pytest.raises(TypeError, match="result must be PopulationCoverageResult"):
            plot_population_coverage(env, {"coverage_fraction": 0.5})  # type: ignore[arg-type]

    def test_title_contains_coverage_info(
        self, env_and_result: tuple[Environment, PopulationCoverageResult]
    ) -> None:
        """Test that title contains coverage information."""
        env, result = env_and_result
        ax = plot_population_coverage(env, result)
        title = ax.get_title()
        assert "%" in title  # Coverage percentage
        assert "/" in title  # Place cells count
        plt.close()


class TestFieldDensityMap:
    """Test field_density_map function."""

    def test_field_density_map_no_overlap(self) -> None:
        """Test density map with non-overlapping fields."""
        n_bins = 10
        all_place_fields = [
            [np.array([0, 1, 2])],  # Cell 1
            [np.array([5, 6, 7])],  # Cell 2
        ]
        density = field_density_map(all_place_fields, n_bins)
        assert density.shape == (n_bins,)
        # Each field bin should have count 1
        assert density[0] == 1
        assert density[1] == 1
        assert density[2] == 1
        assert density[3] == 0
        assert density[5] == 1
        # Non-field bins should be 0
        assert density[4] == 0
        assert density[8] == 0

    def test_field_density_map_overlap(self) -> None:
        """Test density map with overlapping fields."""
        n_bins = 10
        all_place_fields = [
            [np.array([2, 3, 4])],  # Cell 1
            [np.array([3, 4, 5])],  # Cell 2 (overlaps at 3, 4)
            [np.array([4, 5, 6])],  # Cell 3 (overlaps at 4, 5)
        ]
        density = field_density_map(all_place_fields, n_bins)
        assert density[2] == 1  # Only cell 1
        assert density[3] == 2  # Cells 1, 2
        assert density[4] == 3  # Cells 1, 2, 3
        assert density[5] == 2  # Cells 2, 3
        assert density[6] == 1  # Only cell 3
        assert density[0] == 0  # No fields

    def test_field_density_map_multiple_fields(self) -> None:
        """Test density map when cells have multiple fields."""
        n_bins = 20
        all_place_fields = [
            [np.array([0, 1]), np.array([10, 11])],  # Cell 1: 2 fields
            [np.array([1, 2])],  # Cell 2: overlaps first field
        ]
        density = field_density_map(all_place_fields, n_bins)
        assert density[0] == 1  # Cell 1 field 1
        assert density[1] == 2  # Both cells
        assert density[2] == 1  # Cell 2 only
        assert density[10] == 1  # Cell 1 field 2
        assert density[11] == 1  # Cell 1 field 2


class TestCountPlaceCells:
    """Test count_place_cells function."""

    def test_count_place_cells_above_threshold(self) -> None:
        """Test counting cells above information threshold."""
        spatial_information = np.array([0.2, 0.8, 1.5, 0.3, 1.2])
        count = count_place_cells(spatial_information, threshold=0.5)
        # 3 cells have information > 0.5
        assert count == 3

    def test_count_place_cells_default_threshold(self) -> None:
        """Test default threshold of 0.5 bits/spike."""
        spatial_information = np.array([0.3, 0.5, 0.6, 0.4, 1.0])
        count = count_place_cells(spatial_information)
        # Default threshold is 0.5, so count cells with > 0.5
        # Values: 0.6, 1.0 are above threshold
        assert count == 2

    def test_count_place_cells_none_qualify(self) -> None:
        """Test when no cells exceed threshold."""
        spatial_information = np.array([0.1, 0.2, 0.3])
        count = count_place_cells(spatial_information, threshold=0.5)
        assert count == 0

    def test_count_place_cells_all_qualify(self) -> None:
        """Test when all cells exceed threshold."""
        spatial_information = np.array([1.0, 1.5, 2.0, 0.8])
        count = count_place_cells(spatial_information, threshold=0.5)
        assert count == 4

    def test_count_place_cells_boundary(self) -> None:
        """Test behavior at threshold boundary."""
        spatial_information = np.array([0.5, 0.50001, 0.49999])
        count = count_place_cells(spatial_information, threshold=0.5)
        # Only values strictly greater than threshold
        assert count == 1  # Only 0.50001

    def test_count_place_cells_nan_handling(self) -> None:
        """Test handling of NaN values in spatial information."""
        spatial_information = np.array([0.8, np.nan, 1.2, 0.3, np.nan])
        count = count_place_cells(spatial_information, threshold=0.5)
        # NaN values should be excluded from count
        assert count == 2  # 0.8 and 1.2


class TestFieldOverlap:
    """Test field_overlap function."""

    def test_field_overlap_identical(self) -> None:
        """Test overlap of identical fields (Jaccard = 1.0)."""
        field1 = np.array([0, 1, 2, 3])
        field2 = np.array([0, 1, 2, 3])
        overlap = field_overlap(field1, field2)
        assert overlap == pytest.approx(1.0)

    def test_field_overlap_disjoint(self) -> None:
        """Test overlap of completely disjoint fields (Jaccard = 0.0)."""
        field1 = np.array([0, 1, 2])
        field2 = np.array([10, 11, 12])
        overlap = field_overlap(field1, field2)
        assert overlap == pytest.approx(0.0)

    def test_field_overlap_partial(self) -> None:
        """Test partial overlap between fields."""
        field1 = np.array([0, 1, 2, 3])
        field2 = np.array([2, 3, 4, 5])
        overlap = field_overlap(field1, field2)
        # Intersection: {2, 3} = 2 bins
        # Union: {0, 1, 2, 3, 4, 5} = 6 bins
        # Jaccard: 2/6 = 0.333...
        assert overlap == pytest.approx(2 / 6)

    def test_field_overlap_subset(self) -> None:
        """Test when one field is subset of another."""
        field1 = np.array([1, 2])
        field2 = np.array([0, 1, 2, 3, 4])
        overlap = field_overlap(field1, field2)
        # Intersection: {1, 2} = 2
        # Union: {0, 1, 2, 3, 4} = 5
        # Jaccard: 2/5 = 0.4
        assert overlap == pytest.approx(2 / 5)

    def test_field_overlap_empty_fields(self) -> None:
        """Test overlap when one or both fields are empty."""
        field1 = np.array([0, 1, 2])
        field2 = np.array([], dtype=np.int64)
        overlap = field_overlap(field1, field2)
        # Empty intersection and union = field1
        # By convention, Jaccard = 0 when one set is empty
        assert overlap == pytest.approx(0.0)

    def test_field_overlap_both_empty(self) -> None:
        """Test overlap when both fields are empty."""
        field1 = np.array([], dtype=np.int64)
        field2 = np.array([], dtype=np.int64)
        overlap = field_overlap(field1, field2)
        # By convention, Jaccard(∅, ∅) = 0 (though sometimes defined as 1)
        # We use 0 to avoid division by zero issues
        assert overlap == pytest.approx(0.0)


class TestPopulationVectorCorrelation:
    """Test population_vector_correlation function."""

    def test_population_vector_correlation_shape(self) -> None:
        """Test correlation matrix has correct shape."""
        # 3 cells, 50 bins
        rng = np.random.default_rng(42)
        population_matrix = rng.random((3, 50))
        corr_matrix = population_vector_correlation(population_matrix)
        # Should be 3x3 symmetric matrix
        assert corr_matrix.shape == (3, 3)
        # Check symmetry
        np.testing.assert_allclose(corr_matrix, corr_matrix.T, rtol=1e-10)

    def test_population_vector_correlation_diagonal(self) -> None:
        """Test diagonal elements are 1.0 (self-correlation)."""
        rng = np.random.default_rng(42)
        population_matrix = rng.random((5, 100))
        corr_matrix = population_vector_correlation(population_matrix)
        # Diagonal should be all 1.0
        np.testing.assert_allclose(np.diag(corr_matrix), 1.0, rtol=1e-10)

    def test_population_vector_correlation_identical_cells(self) -> None:
        """Test correlation of identical firing patterns."""
        # Two identical cells
        rng = np.random.default_rng(42)
        pattern = rng.random(50)
        population_matrix = np.vstack([pattern, pattern])
        corr_matrix = population_vector_correlation(population_matrix)
        # Off-diagonal should be 1.0 (perfect correlation)
        assert corr_matrix[0, 1] == pytest.approx(1.0, abs=1e-10)
        assert corr_matrix[1, 0] == pytest.approx(1.0, abs=1e-10)

    def test_population_vector_correlation_orthogonal(self) -> None:
        """Test correlation of uncorrelated cells."""
        # Create orthogonal patterns (correlation ~ 0)
        rng = np.random.default_rng(42)
        n_bins = 1000  # Large number for better approximation
        cell1 = rng.standard_normal(n_bins)
        cell2 = rng.standard_normal(n_bins)
        population_matrix = np.vstack([cell1, cell2])
        corr_matrix = population_vector_correlation(population_matrix)
        # Should be close to 0 (not exactly due to randomness)
        assert abs(corr_matrix[0, 1]) < 0.1

    def test_population_vector_correlation_anticorrelated(self) -> None:
        """Test negative correlation (anticorrelated patterns)."""
        pattern = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # Negatively correlated pattern
        anti_pattern = -pattern
        population_matrix = np.vstack([pattern, anti_pattern])
        corr_matrix = population_vector_correlation(population_matrix)
        # Should be -1.0 (perfect anticorrelation)
        assert corr_matrix[0, 1] == pytest.approx(-1.0, abs=1e-10)

    def test_population_vector_correlation_single_cell(self) -> None:
        """Test correlation matrix for single cell."""
        rng = np.random.default_rng(42)
        population_matrix = rng.random((1, 50))
        corr_matrix = population_vector_correlation(population_matrix)
        assert corr_matrix.shape == (1, 1)
        assert corr_matrix[0, 0] == pytest.approx(1.0)

    def test_population_vector_correlation_constant_cells(self) -> None:
        """Test handling of cells with constant firing (zero variance)."""
        # Cell with constant firing rate (correlation undefined)
        rng = np.random.default_rng(42)
        constant_cell = np.ones(50)
        varying_cell = rng.random(50)
        population_matrix = np.vstack([constant_cell, varying_cell])
        corr_matrix = population_vector_correlation(population_matrix)
        # Correlation with constant cell should be NaN
        assert np.isnan(corr_matrix[0, 1])
        assert np.isnan(corr_matrix[1, 0])
        # Self-correlation of constant cell is still 1.0 by definition
        assert corr_matrix[0, 0] == pytest.approx(1.0)


class TestPopulationMetricsIntegration:
    """Integration tests combining multiple population metrics."""

    def test_full_population_workflow(self) -> None:
        """Test complete workflow with population_coverage and other metrics."""
        # Create environment
        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 100, (1000, 2))
        env = Environment.from_samples(positions, bin_size=10.0)
        n_bins = env.n_bins

        # Create synthetic firing rates
        firing_rates = np.zeros((5, n_bins))
        # Create distinct place fields for some neurons
        firing_rates[0, 0:10] = np.linspace(10.0, 1.0, 10)
        firing_rates[1, 20:30] = np.linspace(10.0, 1.0, 10)
        firing_rates[2, 25:35] = np.linspace(10.0, 1.0, 10)  # Overlaps neuron 1
        # Neurons 3 and 4 have no place fields (uniform low firing)
        firing_rates[3, :] = 0.1
        firing_rates[4, :] = 0.1

        # Run population_coverage
        result = population_coverage(firing_rates, env, min_size=1)

        # Verify result structure
        assert isinstance(result, PopulationCoverageResult)
        assert 0.0 <= result.coverage_fraction <= 1.0
        assert result.n_neurons == 5
        assert result.n_place_cells <= 5

        # Verify field_count matches field_density_map
        expected_density = field_density_map(result.place_fields, n_bins)
        np.testing.assert_array_equal(result.field_count, expected_density)

        # Verify consistency
        assert len(result.covered_bins) + len(result.uncovered_bins) == n_bins
        assert (result.is_covered.sum() / n_bins) == pytest.approx(
            result.coverage_fraction
        )

        # Test plotting
        ax = plot_population_coverage(env, result)
        assert isinstance(ax, plt.Axes)
        plt.close()

        ax = plot_population_coverage(env, result, show_field_count=True)
        assert isinstance(ax, plt.Axes)
        plt.close()
