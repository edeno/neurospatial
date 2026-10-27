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
    rate_map_correlations,
)

matplotlib.use("Agg")  # Use non-interactive backend for testing


def _coverage_result(place_fields, is_observed):
    is_observed = np.asarray(is_observed, dtype=bool)
    bin_centers = np.column_stack(
        [np.arange(len(is_observed), dtype=float), np.zeros(len(is_observed))]
    )
    return PopulationCoverageResult(
        place_fields=place_fields,
        is_observed=is_observed,
        bin_centers=bin_centers,
        bin_sizes=np.ones(len(is_observed)),
    )


class TestPopulationCoverageResult:
    """Counts and bin sets are derived from the fields and observed bins."""

    def test_derived_counts_and_bins(self) -> None:
        result = _coverage_result(
            [
                [np.array([0, 1]), np.array([3])],
                [np.array([1, 2])],
                [],
                [],
            ],
            [True, True, True, True, True, False],
        )
        assert result.n_neurons == 4
        assert result.n_place_cells == 2
        assert result.n_fields == 3
        np.testing.assert_array_equal(result.field_count, [1, 2, 1, 1, 0, 0])
        np.testing.assert_array_equal(result.covered_bins, [0, 1, 2, 3])
        # Bin 5 was never observed, so it is not a gap.
        np.testing.assert_array_equal(result.uncovered_bins, [4])
        np.testing.assert_array_equal(result.uncovered_positions, [[4.0, 0.0]])
        assert result.coverage_fraction == pytest.approx(4 / 5)
        assert result.place_cell_fraction == pytest.approx(0.5)
        assert result.fields_per_place_cell == pytest.approx(1.5)
        assert result.mean_redundancy == pytest.approx(5 / 4)
        text = str(result)
        assert "80.0%" in text and "2/4" in text and "1 bins" in text

    def test_dataclass_frozen(self) -> None:
        result = _coverage_result([[np.array([0])]], [True, True])
        with pytest.raises(AttributeError):
            result.is_observed = np.ones(2, dtype=bool)  # type: ignore[misc]

    def test_undefined_ratios_are_nan(self) -> None:
        """0/0 ratios are undefined, not zero."""
        empty = _coverage_result([], [False, False])
        assert np.isnan(empty.place_cell_fraction)
        assert np.isnan(empty.coverage_fraction)
        no_fields = _coverage_result([[], []], [True, True])
        assert np.isnan(no_fields.fields_per_place_cell)
        assert np.isnan(no_fields.mean_redundancy)
        assert no_fields.coverage_fraction == 0.0

    def test_field_bins_must_index_the_bins(self) -> None:
        with pytest.raises(ValueError, match="only 2 bins"):
            _coverage_result([[np.array([0, 5])]], [True, True])


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
            for j in simple_env.neighbors(bins=i):
                firing_rates[i, j] = 5.0

        result = population_coverage(simple_env, firing_rates, bandwidth=0.0)
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
            for j in simple_env.neighbors(bins=i):
                if j < n_bins:
                    firing_rates[i, j] = 5.0

        result = population_coverage(simple_env, firing_rates, bandwidth=0.0)
        assert 0.0 < result.coverage_fraction < 1.0
        assert len(result.uncovered_bins) > 0
        assert len(result.uncovered_positions) == len(result.uncovered_bins)

    def test_zero_coverage_with_explicit_map_mean_filter(
        self, simple_env: Environment
    ) -> None:
        """An explicitly requested map-mean cutoff can exclude every map."""
        n_bins = simple_env.n_bins
        # Every map exceeds the explicitly requested spatial-map-mean cutoff.
        firing_rates = np.ones((5, n_bins)) * 15.0

        with pytest.warns(UserWarning, match="spatial-map-mean filter"):
            result = population_coverage(
                simple_env, firing_rates, bandwidth=0.0, max_mean_rate=10.0
            )
        assert result.coverage_fraction == 0.0
        assert len(result.uncovered_bins) == n_bins
        assert result.n_place_cells == 0
        assert result.n_fields == 0

    def test_default_coverage_does_not_exclude_high_rate_maps(self, simple_env):
        """A flat high-rate map covers space descriptively without implying cell identity."""
        rates = np.full((2, simple_env.n_bins), 25.0)
        result = population_coverage(simple_env, rates, bandwidth=0.0)
        assert result.coverage_fraction == pytest.approx(1.0)
        assert result.n_place_cells == 2  # Units with segmented regions.
        assert all(fields.excluded_reason is None for fields in result.place_fields)

    def test_single_neuron(self, simple_env: Environment) -> None:
        """Test with a single neuron."""
        n_bins = simple_env.n_bins
        # Single neuron with a place field
        firing_rates = np.zeros((1, n_bins))
        firing_rates[0, 0] = 10.0
        for j in simple_env.neighbors(bins=0):
            firing_rates[0, j] = 5.0

        result = population_coverage(simple_env, firing_rates, bandwidth=0.0)
        assert result.n_neurons == 1
        assert result.n_place_cells == 1
        assert len(result.place_fields[0]) == 1
        np.testing.assert_array_equal(
            result.covered_bins, np.sort([0, *simple_env.neighbors(bins=0)])
        )

    def test_unobserved_bins_are_not_gaps(self, simple_env: Environment) -> None:
        """Bins NaN for every neuron were not visited: neither covered nor gaps."""
        n_bins = simple_env.n_bins
        firing_rates = np.full((1, n_bins), np.nan)
        observed = np.array([0, *simple_env.neighbors(bins=0)])
        firing_rates[0, observed] = 5.0
        firing_rates[0, 0] = 10.0

        result = population_coverage(simple_env, firing_rates, bandwidth=0.0)
        assert result.coverage_fraction == 1.0
        assert len(result.uncovered_bins) == 0

    def test_field_count_matches_density_map(self, simple_env: Environment) -> None:
        """Verify field_count matches field_density_map output."""
        n_bins = simple_env.n_bins
        firing_rates = np.zeros((3, n_bins))
        # Create overlapping fields
        firing_rates[0, 0:5] = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
        firing_rates[1, 3:8] = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
        firing_rates[2, 6:11] = np.array([10.0, 8.0, 6.0, 4.0, 2.0])

        result = population_coverage(simple_env, firing_rates, bandwidth=0.0)
        expected_density = field_density_map(result.place_fields, n_bins)
        np.testing.assert_array_equal(result.field_count, expected_density)

    def test_uncovered_positions_match_bin_centers(
        self, simple_env: Environment
    ) -> None:
        """Verify uncovered_positions matches env.bin_centers[uncovered_bins]."""
        n_bins = simple_env.n_bins
        # Low uniform firing (no fields detected)
        firing_rates = np.ones((5, n_bins)) * 0.1

        result = population_coverage(simple_env, firing_rates, bandwidth=0.0)
        expected_positions = simple_env.bin_centers[result.uncovered_bins]
        np.testing.assert_array_equal(result.uncovered_positions, expected_positions)

    def test_n_place_cells_count(self, simple_env: Environment) -> None:
        """Verify n_place_cells counts neurons with at least one field."""
        n_bins = simple_env.n_bins
        firing_rates = np.zeros((5, n_bins))
        # Only first 3 neurons have place fields
        for i in range(3):
            firing_rates[i, i * 10 : (i + 1) * 10] = np.linspace(10.0, 1.0, 10)

        result = population_coverage(simple_env, firing_rates, bandwidth=0.0)
        assert result.n_place_cells == 3
        assert [len(fields) > 0 for fields in result.place_fields] == [True] * 3 + [
            False
        ] * 2

    def test_place_fields_format(self, simple_env: Environment) -> None:
        """Verify place_fields structure matches detect_place_fields output."""
        n_bins = simple_env.n_bins
        firing_rates = np.zeros((2, n_bins))
        firing_rates[0, 0:5] = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
        firing_rates[1, 10:15] = np.array([10.0, 8.0, 6.0, 4.0, 2.0])

        from neurospatial.encoding.spatial import PlaceFieldsResult

        result = population_coverage(simple_env, firing_rates, bandwidth=0.0)
        # Should have list of length n_neurons
        assert len(result.place_fields) == 2
        # Each element is a PlaceFieldsResult that iterates as list[NDArray]
        for neuron_fields in result.place_fields:
            assert isinstance(neuron_fields, PlaceFieldsResult)
            for field in neuron_fields:
                assert isinstance(field, np.ndarray)
                assert field.dtype == np.int64

    # Validation tests

    def test_shape_mismatch_error(self, simple_env: Environment) -> None:
        """Test error when firing_rates shape doesn't match n_bins."""
        rng = np.random.default_rng(42)
        wrong_n_bins = simple_env.n_bins + 10
        firing_rates = rng.random((5, wrong_n_bins))
        with pytest.raises(ValueError, match="firing_rates shape mismatch"):
            population_coverage(simple_env, firing_rates, bandwidth=0.0)

    def test_non_2d_input_error(self, simple_env: Environment) -> None:
        """Test error for non-2D firing_rates."""
        rng = np.random.default_rng(42)
        firing_rates_1d = rng.random(simple_env.n_bins)
        with pytest.raises(ValueError, match="firing_rates must be 2D"):
            population_coverage(simple_env, firing_rates_1d, bandwidth=0.0)  # type: ignore[arg-type]

        firing_rates_3d = rng.random((5, simple_env.n_bins, 3))
        with pytest.raises(ValueError, match="firing_rates must be 2D"):
            population_coverage(simple_env, firing_rates_3d, bandwidth=0.0)  # type: ignore[arg-type]

    def test_threshold_out_of_range_error(self, simple_env: Environment) -> None:
        """Test error for threshold outside (0, 1)."""
        rng = np.random.default_rng(42)
        n_bins = simple_env.n_bins
        firing_rates = rng.random((5, n_bins))

        with pytest.raises(ValueError, match="threshold must be in range"):
            population_coverage(simple_env, firing_rates, bandwidth=0.0, threshold=0.0)

        with pytest.raises(ValueError, match="threshold must be in range"):
            population_coverage(simple_env, firing_rates, bandwidth=0.0, threshold=1.0)

        with pytest.raises(ValueError, match="threshold must be in range"):
            population_coverage(simple_env, firing_rates, bandwidth=0.0, threshold=-0.1)

        with pytest.raises(ValueError, match="threshold must be in range"):
            population_coverage(simple_env, firing_rates, bandwidth=0.0, threshold=1.5)

    def test_max_mean_rate_non_positive_error(self, simple_env: Environment) -> None:
        """Test error for non-positive max_mean_rate."""
        rng = np.random.default_rng(42)
        n_bins = simple_env.n_bins
        firing_rates = rng.random((5, n_bins))

        with pytest.raises(ValueError, match="max_mean_rate must be positive"):
            population_coverage(
                simple_env, firing_rates, bandwidth=0.0, max_mean_rate=0.0
            )

        with pytest.raises(ValueError, match="max_mean_rate must be positive"):
            population_coverage(
                simple_env, firing_rates, bandwidth=0.0, max_mean_rate=-1.0
            )

    def test_n_jobs_parity(self, simple_env: Environment) -> None:
        """population_coverage(n_jobs=2) equals n_jobs=1 on the same input."""
        n_bins = simple_env.n_bins
        # A handful of neurons with distinct place fields.
        rng = np.random.default_rng(7)
        n_neurons = 6
        firing_rates = np.zeros((n_neurons, n_bins))
        for i in range(n_neurons):
            peak = (i * 7) % n_bins
            firing_rates[i, peak] = 10.0
            for j in simple_env.neighbors(bins=peak):
                firing_rates[i, j] = 5.0
        # Add a little noise so detection is non-trivial.
        firing_rates += rng.random((n_neurons, n_bins)) * 0.1

        seq = population_coverage(simple_env, firing_rates, bandwidth=0.0, n_jobs=1)
        par = population_coverage(simple_env, firing_rates, bandwidth=0.0, n_jobs=2)

        # Scalar stats identical.
        assert seq.coverage_fraction == par.coverage_fraction
        assert seq.n_neurons == par.n_neurons
        assert seq.n_place_cells == par.n_place_cells
        assert seq.n_fields == par.n_fields

        # Array stats byte-for-byte identical.
        np.testing.assert_array_equal(seq.field_count, par.field_count)
        np.testing.assert_array_equal(seq.is_covered, par.is_covered)
        np.testing.assert_array_equal(seq.covered_bins, par.covered_bins)
        np.testing.assert_array_equal(seq.uncovered_bins, par.uncovered_bins)
        np.testing.assert_array_equal(seq.uncovered_positions, par.uncovered_positions)

        # Per-neuron place fields identical and in order.
        assert len(seq.place_fields) == len(par.place_fields)
        for seq_fields, par_fields in zip(
            seq.place_fields, par.place_fields, strict=True
        ):
            assert len(seq_fields) == len(par_fields)
            for seq_field, par_field in zip(seq_fields, par_fields, strict=True):
                np.testing.assert_array_equal(seq_field, par_field)


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

        result = population_coverage(env, firing_rates, bandwidth=0.0)
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

        result = population_coverage(env, firing_rates, bandwidth=0.0)
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

    @staticmethod
    def _gap_line_positions(ax: matplotlib.axes.Axes) -> np.ndarray:
        """Sorted x positions of the dashed vertical gap lines."""
        return np.sort(
            [line.get_xdata()[0] for line in ax.lines if line.get_linestyle() == "--"]
        )

    def test_1d_grid_with_gaps(self) -> None:
        """A 1-D grid (not a track graph) draws one gap line per uncovered bin."""
        env = Environment.from_samples(
            np.linspace(0.0, 100.0, 101)[:, None], bin_size=10.0
        )
        assert env.n_dims == 1
        assert not env.is_linearized_track
        firing_rates = np.zeros((2, env.n_bins))
        firing_rates[0, 2:5] = [6.0, 10.0, 6.0]
        result = population_coverage(env, firing_rates, bandwidth=0.0)
        assert result.uncovered_positions.shape[1] == 1
        assert len(result.uncovered_bins) > 0

        for show_field_count in (False, True):
            ax = plot_population_coverage(
                env, result, show_field_count=show_field_count
            )
            np.testing.assert_allclose(
                self._gap_line_positions(ax),
                np.sort(result.uncovered_positions[:, 0]),
            )
            plt.close()

    def test_2d_track_gaps_in_linear_coordinates(self) -> None:
        """A track embedded in 2-D marks gaps at their linearized positions."""
        import networkx as nx

        graph = nx.Graph()
        graph.add_node(0, pos=(0.0, 0.0))
        graph.add_node(1, pos=(50.0, 0.0))
        graph.add_node(2, pos=(50.0, 50.0))
        graph.add_edge(0, 1, distance=50.0)
        graph.add_edge(1, 2, distance=50.0)
        # An L-shaped track with a 20 cm display gap between its two segments.
        env = Environment.from_graph(
            graph=graph,
            edge_order=[(0, 1), (1, 2)],
            edge_spacing=20.0,
            bin_size=10.0,
        )
        assert env.is_linearized_track
        assert env.n_dims == 2
        firing_rates = np.zeros((2, env.n_bins))
        firing_rates[0, 2:5] = [6.0, 10.0, 6.0]
        result = population_coverage(env, firing_rates, bandwidth=0.0)

        ax = plot_population_coverage(env, result)
        np.testing.assert_allclose(
            self._gap_line_positions(ax),
            np.sort(env.to_linear(result.uncovered_positions)),
        )
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
    """count_place_cells counts the bias-corrected place-cell screen."""

    @pytest.fixture(scope="class")
    def rates(self):
        from neurospatial import Environment
        from neurospatial.encoding import compute_spatial_rates
        from neurospatial.simulation import generate_poisson_spikes

        rng = np.random.default_rng(0)
        positions = rng.uniform(0, 100, (36000, 2))
        env = Environment.from_samples(positions, bin_size=5.0, units="cm")
        times = np.arange(36000) / 30.0
        distance2 = np.sum((positions - 50.0) ** 2, axis=1)
        field = 8.0 * np.exp(-distance2 / (2 * 10.0**2))
        trains = [
            generate_poisson_spikes(field, times, seed=1),  # place cell
            generate_poisson_spikes(np.full(36000, 0.5), times, seed=2),  # untuned
            np.array([]),  # silent
        ]
        return compute_spatial_rates(env, trains, times, positions)

    def test_counts_the_classify_screen(self, rates) -> None:
        assert count_place_cells(rates) == int(rates.classify().sum()) == 1
        assert rates.classify().tolist() == [True, False, False]

    def test_untuned_cell_passes_only_without_the_correction(self, rates) -> None:
        corrected = rates.spatial_information() - rates.spatial_information_bias()
        raw = rates.spatial_information()
        # 600 spikes of an untuned cell: raw information is all bias.
        assert raw[1] > 0.04
        assert abs(corrected[1]) < 0.02
        assert count_place_cells(rates, min_info=raw[1]) == 1

    def test_cutoff_is_inclusive(self, rates) -> None:
        corrected = rates.spatial_information() - rates.spatial_information_bias()
        assert count_place_cells(rates, min_info=corrected[0]) == 1
        assert count_place_cells(rates, min_info=np.nextafter(corrected[0], 9)) == 0

    def test_information_values_alone_are_refused(self) -> None:
        with pytest.raises(TypeError, match="SpatialRatesResult"):
            count_place_cells(np.array([0.8, 1.2]))  # type: ignore[arg-type]


# A 20 cm track in 1 cm bins.
TRACK_20 = Environment.from_grid_mask(
    np.ones(20, dtype=bool), grid_edges=(np.arange(21.0),), units="cm"
)


class TestFieldOverlap:
    """Test field_overlap function."""

    def test_field_overlap_identical(self) -> None:
        """Test overlap of identical fields (Jaccard = 1.0)."""
        field1 = np.array([0, 1, 2, 3])
        field2 = np.array([0, 1, 2, 3])
        overlap = field_overlap(TRACK_20, field1, field2)
        assert overlap == pytest.approx(1.0)

    def test_field_overlap_disjoint(self) -> None:
        """Test overlap of completely disjoint fields (Jaccard = 0.0)."""
        field1 = np.array([0, 1, 2])
        field2 = np.array([10, 11, 12])
        overlap = field_overlap(TRACK_20, field1, field2)
        assert overlap == pytest.approx(0.0)

    def test_field_overlap_partial(self) -> None:
        """Test partial overlap between fields."""
        field1 = np.array([0, 1, 2, 3])
        field2 = np.array([2, 3, 4, 5])
        overlap = field_overlap(TRACK_20, field1, field2)
        # Intersection: {2, 3} = 2 bins
        # Union: {0, 1, 2, 3, 4, 5} = 6 bins
        # Jaccard: 2/6 = 0.333...
        assert overlap == pytest.approx(2 / 6)

    def test_field_overlap_subset(self) -> None:
        """Test when one field is subset of another."""
        field1 = np.array([1, 2])
        field2 = np.array([0, 1, 2, 3, 4])
        overlap = field_overlap(TRACK_20, field1, field2)
        # Intersection: {1, 2} = 2
        # Union: {0, 1, 2, 3, 4} = 5
        # Jaccard: 2/5 = 0.4
        assert overlap == pytest.approx(2 / 5)

    def test_field_overlap_empty_fields(self) -> None:
        """Test overlap when one or both fields are empty."""
        field1 = np.array([0, 1, 2])
        field2 = np.array([], dtype=np.int64)
        overlap = field_overlap(TRACK_20, field1, field2)
        # Empty intersection and union = field1
        # By convention, Jaccard = 0 when one set is empty
        assert overlap == pytest.approx(0.0)

    def test_field_overlap_both_empty(self) -> None:
        """No field bins in either set: nothing to compare, so NaN, not 0."""
        field1 = np.array([], dtype=np.int64)
        field2 = np.array([], dtype=np.int64)
        overlap = field_overlap(TRACK_20, field1, field2)
        assert np.isnan(overlap)


class TestRateMapCorrelations:
    """Test rate_map_correlations function."""

    def test_rate_map_correlations_shape(self) -> None:
        """Test correlation matrix has correct shape."""
        # 3 cells, 50 bins
        rng = np.random.default_rng(42)
        firing_rates = rng.random((3, 50))
        corr_matrix = rate_map_correlations(firing_rates)
        # Should be 3x3 symmetric matrix
        assert corr_matrix.shape == (3, 3)
        # Check symmetry
        np.testing.assert_allclose(corr_matrix, corr_matrix.T, rtol=1e-10)

    def test_rate_map_correlations_diagonal(self) -> None:
        """Test diagonal elements are 1.0 (self-correlation)."""
        rng = np.random.default_rng(42)
        firing_rates = rng.random((5, 100))
        corr_matrix = rate_map_correlations(firing_rates)
        # Diagonal should be all 1.0
        np.testing.assert_allclose(np.diag(corr_matrix), 1.0, rtol=1e-10)

    def test_rate_map_correlations_identical_cells(self) -> None:
        """Test correlation of identical firing patterns."""
        # Two identical cells
        rng = np.random.default_rng(42)
        pattern = rng.random(50)
        firing_rates = np.vstack([pattern, pattern])
        corr_matrix = rate_map_correlations(firing_rates)
        # Off-diagonal should be 1.0 (perfect correlation)
        assert corr_matrix[0, 1] == pytest.approx(1.0, abs=1e-10)
        assert corr_matrix[1, 0] == pytest.approx(1.0, abs=1e-10)

    def test_rate_map_correlations_orthogonal(self) -> None:
        """Test correlation of uncorrelated cells."""
        # Create orthogonal patterns (correlation ~ 0)
        rng = np.random.default_rng(42)
        n_bins = 1000  # Large number for better approximation
        cell1 = rng.standard_normal(n_bins)
        cell2 = rng.standard_normal(n_bins)
        firing_rates = np.vstack([cell1, cell2])
        corr_matrix = rate_map_correlations(firing_rates)
        # Should be close to 0 (not exactly due to randomness)
        assert abs(corr_matrix[0, 1]) < 0.1

    def test_rate_map_correlations_anticorrelated(self) -> None:
        """Test negative correlation (anticorrelated patterns)."""
        pattern = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # Negatively correlated pattern
        anti_pattern = -pattern
        firing_rates = np.vstack([pattern, anti_pattern])
        corr_matrix = rate_map_correlations(firing_rates)
        # Should be -1.0 (perfect anticorrelation)
        assert corr_matrix[0, 1] == pytest.approx(-1.0, abs=1e-10)

    def test_rate_map_correlations_single_cell(self) -> None:
        """Test correlation matrix for single cell."""
        rng = np.random.default_rng(42)
        firing_rates = rng.random((1, 50))
        corr_matrix = rate_map_correlations(firing_rates)
        assert corr_matrix.shape == (1, 1)
        assert corr_matrix[0, 0] == pytest.approx(1.0)

    def test_rate_map_correlations_constant_cells(self) -> None:
        """Test handling of cells with constant firing (zero variance)."""
        # Cell with constant firing rate (correlation undefined)
        rng = np.random.default_rng(42)
        constant_cell = np.ones(50)
        varying_cell = rng.random(50)
        firing_rates = np.vstack([constant_cell, varying_cell])
        corr_matrix = rate_map_correlations(firing_rates)
        # Correlation with constant cell should be NaN
        assert np.isnan(corr_matrix[0, 1])
        assert np.isnan(corr_matrix[1, 0])
        # A constant map has no defined self-correlation either
        assert np.isnan(corr_matrix[0, 0])
        assert corr_matrix[1, 1] == pytest.approx(1.0)

    def test_rate_map_correlations_roundoff_constant_map_is_nan(self) -> None:
        """Equal rates whose std rounds to ~1e-17 still make a constant map."""
        assert np.full(50, 0.1).std() != 0.0  # the trap this test guards
        rng = np.random.default_rng(42)
        varying = rng.random((2, 50))
        firing_rates = np.vstack([varying[0], np.full(50, 0.1), varying[1]])
        corr_matrix = rate_map_correlations(firing_rates)
        expected_nan = np.zeros((3, 3), dtype=bool)
        expected_nan[1, :] = True
        expected_nan[:, 1] = True
        np.testing.assert_array_equal(np.isnan(corr_matrix), expected_nan)
        keep = [0, 2]
        np.testing.assert_array_equal(
            corr_matrix[np.ix_(keep, keep)], rate_map_correlations(varying)
        )

    def test_rate_map_correlations_nan_bin_propagates_per_row(self) -> None:
        """A NaN bin makes that map's row, column and diagonal NaN only."""
        rng = np.random.default_rng(42)
        firing_rates = rng.random((3, 30))
        firing_rates[1, 4] = np.nan
        corr_matrix = rate_map_correlations(firing_rates)
        expected_nan = np.zeros((3, 3), dtype=bool)
        expected_nan[1, :] = True
        expected_nan[:, 1] = True
        np.testing.assert_array_equal(np.isnan(corr_matrix), expected_nan)
        np.testing.assert_allclose(
            corr_matrix[0, 2], np.corrcoef(firing_rates[0], firing_rates[2])[0, 1]
        )

    def test_rate_map_correlations_no_bins_is_all_nan(self) -> None:
        """No shared bins: nothing to correlate, so every entry is NaN."""
        corr_matrix = rate_map_correlations(np.ones((3, 0)))
        assert corr_matrix.shape == (3, 3)
        assert np.isnan(corr_matrix).all()

    def test_rate_map_correlations_map_flat_up_to_rounding_is_nan(self) -> None:
        """Rates equal up to rounding, as smoothing leaves a flat map, are flat."""
        rng = np.random.default_rng(42)
        varying = rng.random((2, 50))
        flat = 5.0 * (1.0 + 1e-15 * rng.standard_normal(50))
        assert flat.max() > flat.min()  # the trap: an exact check passes it
        corr_matrix = rate_map_correlations(np.vstack([varying[0], flat, varying[1]]))
        assert np.isnan(corr_matrix[1]).all() and np.isnan(corr_matrix[:, 1]).all()
        np.testing.assert_array_equal(
            corr_matrix[np.ix_([0, 2], [0, 2])], rate_map_correlations(varying)
        )

    def test_rate_map_correlations_two_bins_is_all_nan(self) -> None:
        """Two bins always correlate at +-1: the correlation is undefined."""
        corr_matrix = rate_map_correlations(np.array([[1.0, 3.0], [2.0, 1.0]]))
        assert np.isnan(corr_matrix).all()

    def test_rate_map_correlations_two_conditions(self) -> None:
        """Stacked conditions: the off-diagonal block's diagonal is per-map r."""
        rng = np.random.default_rng(42)
        a = rng.random((4, 30))
        corr_matrix = rate_map_correlations(np.vstack([a, a]))
        assert corr_matrix.shape == (8, 8)
        np.testing.assert_allclose(np.diag(corr_matrix[:4, 4:]), 1.0, atol=1e-10)


class TestPopulationVectorCorrelation:
    """Per-location population-vector correlation between two conditions."""

    DEMO = np.array([[1.0, 2.0, 5.0], [3.0, 2.0, 1.0], [2.0, 2.0, 4.0]])

    @pytest.fixture
    def random_maps(self) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(7)
        return rng.random((12, 40)), rng.random((12, 40))

    def test_identical_maps_are_one_and_flat_location_is_nan(self) -> None:
        pv = population_vector_correlation(self.DEMO, self.DEMO)
        assert pv.shape == (3,)
        assert pv.dtype == np.float64
        np.testing.assert_allclose(pv[[0, 2]], 1.0, rtol=1e-12)
        assert np.isnan(pv[1])

    def test_negated_map_is_minus_one(self) -> None:
        pv = population_vector_correlation(self.DEMO, -self.DEMO)
        np.testing.assert_allclose(pv[[0, 2]], -1.0, rtol=1e-12)
        assert np.isnan(pv[1])

    def test_matches_corrcoef_at_every_location(self, random_maps) -> None:
        a, c = random_maps
        pv = population_vector_correlation(a, c)
        expected = [np.corrcoef(a[:, b], c[:, b])[0, 1] for b in range(a.shape[1])]
        np.testing.assert_allclose(pv, expected, rtol=1e-12)

    def test_uncorrelated_maps_are_near_zero(self) -> None:
        rng = np.random.default_rng(3)
        a = rng.random((500, 50))
        c = rng.random((500, 50))
        pv = population_vector_correlation(a, c)
        assert np.all(np.isfinite(pv))
        assert abs(np.median(pv)) < 0.05
        assert np.all(np.abs(pv) < 0.2)

    def test_scale_invariant(self, random_maps) -> None:
        a, c = random_maps
        np.testing.assert_allclose(
            population_vector_correlation(a, 3.0 * c),
            population_vector_correlation(a, c),
            rtol=1e-12,
        )

    def test_shared_unit_permutation_leaves_result_unchanged(self, random_maps) -> None:
        a, c = random_maps
        order = np.random.default_rng(1).permutation(a.shape[0])
        np.testing.assert_allclose(
            population_vector_correlation(a[order], c[order]),
            population_vector_correlation(a, c),
            rtol=1e-12,
        )

    def test_unit_identity_matters(self) -> None:
        """Shuffling units in one map only breaks a perfect correlation."""
        a = np.random.default_rng(2).random((12, 40))
        order = np.roll(np.arange(12), 1)
        pv = population_vector_correlation(a, a[order])
        # Independent random maps: correlations scatter around zero, not one.
        assert abs(np.median(pv)) < 0.2

    def test_nan_rate_makes_location_nan(self, random_maps) -> None:
        a, c = random_maps
        c = c.copy()
        c[3, 5] = np.nan  # one unit unsupported at one location
        a = a.copy()
        a[:, 9] = np.nan  # a whole unvisited location
        pv = population_vector_correlation(a, c)
        assert np.isnan(pv[5]) and np.isnan(pv[9])
        assert np.isfinite(np.delete(pv, [5, 9])).all()

    def test_location_outside_spatial_support_is_nan(self, random_maps) -> None:
        a, c = random_maps
        support = np.ones(a.shape[1], dtype=bool)
        support[[0, 17]] = False
        pv = population_vector_correlation(a, c, spatial_support=support)
        unrestricted = population_vector_correlation(a, c)
        assert np.isnan(pv[[0, 17]]).all()
        np.testing.assert_array_equal(pv[support], unrestricted[support])

    def test_zero_filled_unsupported_location_is_nan_not_zero(self) -> None:
        """Zero-filled maps with spatial_support: an unvisited bin is NaN."""
        a = self.DEMO.copy()
        a[:, 0] = 0.0
        support = np.array([False, True, True])
        pv = population_vector_correlation(a, a, spatial_support=support)
        assert np.isnan(pv[0])
        assert pv[2] == pytest.approx(1.0)

    def test_roundoff_flat_location_is_nan(self) -> None:
        """Equal rates whose std rounds to ~1e-17 still count as flat."""
        assert np.full(3, 0.1).std() != 0.0  # the trap this test guards
        a = self.DEMO.copy()
        a[:, 1] = 0.1
        c = self.DEMO + 1.0
        c[:, 1] = [1.0, 4.0, 2.0]  # the partner varies, so only a is flat
        pv = population_vector_correlation(a, c)
        assert np.isnan(pv[1])
        pv_other = population_vector_correlation(c, a)
        assert np.isnan(pv_other[1])

    def test_location_flat_up_to_rounding_is_nan(self) -> None:
        """Rates equal up to rounding, as smoothing leaves them, count as flat."""
        a = self.DEMO.copy()
        a[:, 1] = 2.0 * (1.0 + np.array([1e-15, -2e-15, 0.0]))
        assert a[:, 1].max() > a[:, 1].min()  # the trap: an exact check passes it
        c = self.DEMO + 1.0
        c[:, 1] = [1.0, 4.0, 2.0]
        assert np.isnan(population_vector_correlation(a, c)[1])
        assert np.isnan(population_vector_correlation(c, a)[1])
        assert np.isfinite(population_vector_correlation(a, c)[[0, 2]]).all()

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same shape"):
            population_vector_correlation(np.ones((3, 4)), np.ones((3, 5)))

    def test_one_dimensional_input_raises(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            population_vector_correlation(np.ones(4), np.ones(4))

    @pytest.mark.parametrize("n_units", [1, 2])
    def test_fewer_than_three_units_raises(self, n_units: int) -> None:
        """Two units always correlate at +-1, so the correlation is undefined."""
        rates = np.random.default_rng(0).random((n_units, 4))
        with pytest.raises(ValueError, match="at least three units"):
            population_vector_correlation(rates, rates[:, ::-1])

    @pytest.mark.parametrize(
        "support",
        [np.ones(4, dtype=bool), np.ones(3)],
        ids=["wrong-shape", "not-bool"],
    )
    def test_bad_spatial_support_raises(self, support) -> None:
        with pytest.raises(ValueError, match="spatial_support"):
            population_vector_correlation(self.DEMO, self.DEMO, spatial_support=support)


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
        result = population_coverage(env, firing_rates, bandwidth=0.0)

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
