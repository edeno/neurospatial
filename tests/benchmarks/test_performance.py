"""
Performance benchmarks for neurospatial.

These benchmarks track performance over time to prevent regressions.
Use pytest-benchmark to run and compare results across commits.

Usage
-----
Run benchmarks:
    uv run pytest tests/benchmarks/ -v

Save baseline:
    uv run pytest tests/benchmarks/ --benchmark-save=baseline

Compare against baseline:
    uv run pytest tests/benchmarks/ --benchmark-compare=baseline

Generate histogram:
    uv run pytest tests/benchmarks/ --benchmark-histogram
"""

import numpy as np
import pytest

from neurospatial import (
    Environment,
    compute_place_field,
    detect_place_fields,
    map_points_to_bins,
    pairwise_distances,
    skaggs_information,
    sparsity,
)
from neurospatial.kernels import compute_diffusion_kernels

# =============================================================================
# Fixtures for Benchmark Data
# =============================================================================


@pytest.fixture(scope="module")
def benchmark_data_small():
    """Small dataset for quick benchmarks (1000 samples, 2D)."""
    rng = np.random.default_rng(42)
    positions = rng.uniform(0, 50, (1000, 2))
    times = np.linspace(0, 100, 1000)
    spike_times = rng.uniform(0, 100, 50)
    return positions, times, spike_times


@pytest.fixture(scope="module")
def benchmark_data_medium():
    """Medium dataset for standard benchmarks (5000 samples, 2D)."""
    rng = np.random.default_rng(42)
    positions = rng.uniform(0, 100, (5000, 2))
    times = np.linspace(0, 500, 5000)
    spike_times = rng.uniform(0, 500, 200)
    return positions, times, spike_times


@pytest.fixture(scope="module")
def benchmark_data_large():
    """Large dataset for stress benchmarks (10000 samples, 2D)."""
    rng = np.random.default_rng(42)
    positions = rng.uniform(0, 200, (10000, 2))
    times = np.linspace(0, 1000, 10000)
    spike_times = rng.uniform(0, 1000, 500)
    return positions, times, spike_times


@pytest.fixture(scope="module")
def small_env(benchmark_data_small):
    """Pre-built small environment for benchmarks."""
    positions, _, _ = benchmark_data_small
    return Environment.from_samples(positions, bin_size=5.0)


@pytest.fixture(scope="module")
def medium_env(benchmark_data_medium):
    """Pre-built medium environment for benchmarks."""
    positions, _, _ = benchmark_data_medium
    return Environment.from_samples(positions, bin_size=5.0)


@pytest.fixture(scope="module")
def large_env(benchmark_data_large):
    """Pre-built large environment for benchmarks."""
    positions, _, _ = benchmark_data_large
    return Environment.from_samples(positions, bin_size=10.0)


# =============================================================================
# Environment Creation Benchmarks
# =============================================================================


class TestEnvironmentCreationPerformance:
    """Benchmark Environment creation from samples."""

    def test_environment_from_samples_small(self, benchmark, benchmark_data_small):
        """Benchmark Environment.from_samples() with small dataset (1000 samples)."""
        positions, _, _ = benchmark_data_small

        result = benchmark(Environment.from_samples, positions, bin_size=5.0)

        assert result.n_bins > 0
        assert result.n_dims == 2

    def test_environment_from_samples_medium(self, benchmark, benchmark_data_medium):
        """Benchmark Environment.from_samples() with medium dataset (5000 samples)."""
        positions, _, _ = benchmark_data_medium

        result = benchmark(Environment.from_samples, positions, bin_size=5.0)

        assert result.n_bins > 0
        assert result.n_dims == 2

    @pytest.mark.slow
    def test_environment_from_samples_large(self, benchmark, benchmark_data_large):
        """Benchmark Environment.from_samples() with large dataset (10000 samples).

        Marked slow: Creates large environment with 10000 samples.
        """
        positions, _, _ = benchmark_data_large

        result = benchmark(Environment.from_samples, positions, bin_size=10.0)

        assert result.n_bins > 0
        assert result.n_dims == 2


# =============================================================================
# Place Field Computation Benchmarks
# =============================================================================


class TestPlaceFieldComputationPerformance:
    """Benchmark place field computation methods."""

    def test_place_field_diffusion_kde_small(
        self, benchmark, small_env, benchmark_data_small
    ):
        """Benchmark diffusion_kde with small environment."""
        _, times, spike_times = benchmark_data_small
        positions, _, _ = benchmark_data_small

        result = benchmark(
            compute_place_field,
            small_env,
            spike_times,
            times,
            positions,
            method="diffusion_kde",
            bandwidth=5.0,
        )

        assert result.shape == (small_env.n_bins,)

    def test_place_field_diffusion_kde_medium(
        self, benchmark, medium_env, benchmark_data_medium
    ):
        """Benchmark diffusion_kde with medium environment."""
        _, times, spike_times = benchmark_data_medium
        positions, _, _ = benchmark_data_medium

        result = benchmark(
            compute_place_field,
            medium_env,
            spike_times,
            times,
            positions,
            method="diffusion_kde",
            bandwidth=5.0,
        )

        assert result.shape == (medium_env.n_bins,)

    def test_place_field_binned_small(self, benchmark, small_env, benchmark_data_small):
        """Benchmark binned method with small environment."""
        _, times, spike_times = benchmark_data_small
        positions, _, _ = benchmark_data_small

        result = benchmark(
            compute_place_field,
            small_env,
            spike_times,
            times,
            positions,
            method="binned",
        )

        assert result.shape == (small_env.n_bins,)


# =============================================================================
# Spatial Query Benchmarks
# =============================================================================


class TestSpatialQueryPerformance:
    """Benchmark spatial query operations."""

    def test_bin_at_single_point(self, benchmark, medium_env):
        """Benchmark bin_at() for single point lookup."""
        point = np.array([[50.0, 50.0]])

        result = benchmark(medium_env.bin_at, point)

        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)
        assert isinstance(result[0], (int, np.integer))

    def test_bin_at_batch(self, benchmark, medium_env, benchmark_data_medium):
        """Benchmark bin_at() for batch point lookup (5000 points)."""
        positions, _, _ = benchmark_data_medium

        result = benchmark(medium_env.bin_at, positions)

        assert len(result) == len(positions)

    def test_map_points_to_bins_batch(
        self, benchmark, medium_env, benchmark_data_medium
    ):
        """Benchmark map_points_to_bins() with KDTree caching (5000 points)."""
        positions, _, _ = benchmark_data_medium

        result = benchmark(
            map_points_to_bins, positions, medium_env, tie_break="lowest_index"
        )

        assert len(result) == len(positions)

    def test_path_between(self, benchmark, medium_env):
        """Benchmark path_between() computation."""
        # Get two random bins from environment
        rng = np.random.default_rng(42)
        start_bin = rng.integers(0, medium_env.n_bins)
        end_bin = rng.integers(0, medium_env.n_bins)

        result = benchmark(medium_env.path_between, start_bin, end_bin)

        assert isinstance(result, list)


# =============================================================================
# Graph Operation Benchmarks
# =============================================================================


class TestGraphOperationPerformance:
    """Benchmark graph-based operations."""

    def test_compute_diffusion_kernels(self, benchmark, medium_env):
        """Benchmark diffusion kernel computation."""
        result = benchmark(
            compute_diffusion_kernels,
            medium_env.connectivity,
            bandwidth_sigma=5.0,
            bin_sizes=medium_env.bin_sizes,
            mode="transition",
        )

        assert result.shape == (medium_env.n_bins, medium_env.n_bins)

    def test_pairwise_distances_subset(self, benchmark, medium_env):
        """Benchmark pairwise_distances() for subset of nodes."""
        # Get 50 random bins
        rng = np.random.default_rng(42)
        nodes = rng.choice(medium_env.n_bins, size=50, replace=False).tolist()

        result = benchmark(pairwise_distances, medium_env.connectivity, nodes)

        assert result.shape == (len(nodes), len(nodes))

    def test_neighbors_query(self, benchmark, medium_env):
        """Benchmark neighbors() query."""
        # Get random bin
        rng = np.random.default_rng(42)
        bin_id = rng.integers(0, medium_env.n_bins)

        result = benchmark(medium_env.neighbors, bin_id)

        assert isinstance(result, list)


# =============================================================================
# Neuroscience Metric Benchmarks
# =============================================================================


class TestMetricComputationPerformance:
    """Benchmark neuroscience metric computations."""

    @pytest.fixture
    def firing_rate_and_occupancy(self, medium_env):
        """Generate firing rate and occupancy for metrics benchmarks."""
        rng = np.random.default_rng(42)
        firing_rate = rng.uniform(0, 10, size=medium_env.n_bins)
        occupancy = rng.uniform(0.1, 1.0, size=medium_env.n_bins)
        return firing_rate, occupancy

    def test_skaggs_information_computation(self, benchmark, firing_rate_and_occupancy):
        """Benchmark Skaggs information computation."""
        firing_rate, occupancy = firing_rate_and_occupancy

        result = benchmark(skaggs_information, firing_rate, occupancy)

        assert isinstance(result, float)
        assert result >= 0.0

    def test_sparsity_computation(self, benchmark, firing_rate_and_occupancy):
        """Benchmark sparsity computation."""
        firing_rate, occupancy = firing_rate_and_occupancy

        result = benchmark(sparsity, firing_rate, occupancy)

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_detect_place_fields(
        self, benchmark, medium_env, firing_rate_and_occupancy
    ):
        """Benchmark place field detection."""
        firing_rate, _ = firing_rate_and_occupancy

        result = benchmark(detect_place_fields, firing_rate, medium_env)

        assert isinstance(result, list)


# =============================================================================
# Field Operation Benchmarks
# =============================================================================


class TestFieldOperationPerformance:
    """Benchmark spatial field operations."""

    @pytest.fixture
    def test_field(self, medium_env):
        """Generate test field for benchmarks."""
        rng = np.random.default_rng(42)
        return rng.uniform(0, 1, size=medium_env.n_bins)

    def test_smooth_field(self, benchmark, medium_env, test_field):
        """Benchmark field smoothing."""
        result = benchmark(medium_env.smooth, test_field, bandwidth=5.0)

        assert result.shape == test_field.shape

    def test_interpolate_field(self, benchmark, medium_env, test_field):
        """Benchmark field interpolation."""
        # Generate query points
        rng = np.random.default_rng(42)
        query_points = rng.uniform(0, 100, (100, 2))

        result = benchmark(medium_env.interpolate, test_field, query_points)

        assert len(result) == len(query_points)


# =============================================================================
# Trajectory Analysis Benchmarks
# =============================================================================


class TestTrajectoryAnalysisPerformance:
    """Benchmark trajectory analysis operations."""

    def test_occupancy_computation(self, benchmark, medium_env, benchmark_data_medium):
        """Benchmark occupancy computation from trajectory."""
        positions, times, _ = benchmark_data_medium

        result = benchmark(medium_env.occupancy, times, positions)

        assert result.shape == (medium_env.n_bins,)

    def test_bin_sequence_extraction(
        self, benchmark, medium_env, benchmark_data_medium
    ):
        """Benchmark bin sequence extraction from positions."""
        positions, times, _ = benchmark_data_medium

        result = benchmark(medium_env.bin_sequence, times, positions)

        assert isinstance(result, np.ndarray)
        assert result.ndim == 1
        assert len(result) <= len(positions)

    def test_transition_matrix(self, benchmark, medium_env, benchmark_data_medium):
        """Benchmark transition matrix computation."""
        positions, times, _ = benchmark_data_medium

        result = benchmark(
            medium_env.transitions,
            times=times,
            positions=positions,
        )

        assert result.shape == (medium_env.n_bins, medium_env.n_bins)
