"""Tests for encoding backend benchmark script.

This module validates that the benchmark script runs correctly and produces
expected output format.

TDD approach: Test first, then implement the benchmark script.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add benchmarks directory to path for bench_encoding_backends import
BENCHMARKS_DIR = Path(__file__).parent.parent.parent / "benchmarks"
if not BENCHMARKS_DIR.exists():
    raise RuntimeError(f"Benchmarks directory not found: {BENCHMARKS_DIR}")
sys.path.insert(0, str(BENCHMARKS_DIR))

from neurospatial.encoding._backend import is_jax_available  # noqa: E402


@pytest.fixture(autouse=True)
def enable_jax_x64():
    """Enable JAX x64 mode for fair comparison with NumPy."""
    if is_jax_available():
        import jax

        jax.config.update("jax_enable_x64", True)


class TestBenchmarkRunner:
    """Tests that benchmark script components work correctly."""

    def test_benchmark_imports(self) -> None:
        """Benchmark module should import without errors."""
        # This will fail until we create the benchmark module
        import bench_encoding_backends

        assert hasattr(bench_encoding_backends, "run_encoding_benchmark")
        assert hasattr(bench_encoding_backends, "EncodingBenchmarkResult")

    def test_create_benchmark_data(self) -> None:
        """Benchmark data creation should work for different population sizes."""
        from bench_encoding_backends import create_benchmark_data

        # Test small population
        data = create_benchmark_data(n_neurons=10, n_samples=500, seed=42)

        assert "env" in data
        assert "spike_times_list" in data
        assert "times" in data
        assert "positions" in data
        assert len(data["spike_times_list"]) == 10

    def test_benchmark_result_structure(self) -> None:
        """Benchmark result dataclass should have expected fields."""
        from bench_encoding_backends import EncodingBenchmarkResult

        result = EncodingBenchmarkResult(
            n_neurons=10,
            backend="numpy",
            function_name="compute_spatial_rates",
            elapsed_ms=100.0,
            memory_mb=50.0,
        )

        assert result.n_neurons == 10
        assert result.backend == "numpy"
        assert result.function_name == "compute_spatial_rates"
        assert result.elapsed_ms == 100.0
        assert result.memory_mb == 50.0

    def test_run_single_benchmark_numpy(self) -> None:
        """Benchmark runner should work with NumPy backend."""
        from bench_encoding_backends import (
            create_benchmark_data,
            run_single_benchmark,
        )

        data = create_benchmark_data(n_neurons=5, n_samples=100, seed=42)
        result = run_single_benchmark(
            data=data,
            function_name="compute_spatial_rates",
            backend="numpy",
            n_iterations=1,
        )

        assert result.backend == "numpy"
        assert result.function_name == "compute_spatial_rates"
        assert result.elapsed_ms > 0
        assert result.n_neurons == 5

    @pytest.mark.skipif(
        not is_jax_available(),
        reason="JAX is not available on this platform",
    )
    def test_run_single_benchmark_jax(self) -> None:
        """Benchmark runner should work with JAX backend."""
        from bench_encoding_backends import (
            create_benchmark_data,
            run_single_benchmark,
        )

        data = create_benchmark_data(n_neurons=5, n_samples=100, seed=42)
        result = run_single_benchmark(
            data=data,
            function_name="compute_spatial_rates",
            backend="jax",
            n_iterations=1,
        )

        assert result.backend == "jax"
        assert result.function_name == "compute_spatial_rates"
        assert result.elapsed_ms > 0
        assert result.n_neurons == 5


class TestBenchmarkPopulationSizes:
    """Tests that benchmarks work across different population sizes."""

    @pytest.mark.parametrize("n_neurons", [10, 50])
    def test_benchmark_with_different_populations(self, n_neurons: int) -> None:
        """Benchmark should work with varying population sizes."""
        from bench_encoding_backends import (
            create_benchmark_data,
            run_single_benchmark,
        )

        data = create_benchmark_data(n_neurons=n_neurons, n_samples=100, seed=42)
        result = run_single_benchmark(
            data=data,
            function_name="compute_spatial_rates",
            backend="numpy",
            n_iterations=1,
        )

        assert result.n_neurons == n_neurons
        assert result.elapsed_ms > 0


class TestBenchmarkOutputFormat:
    """Tests for benchmark output formatting."""

    def test_results_to_dataframe(self) -> None:
        """Benchmark results should convert to DataFrame correctly."""
        from bench_encoding_backends import (
            EncodingBenchmarkResult,
            results_to_dataframe,
        )

        results = [
            EncodingBenchmarkResult(
                n_neurons=10,
                backend="numpy",
                function_name="compute_spatial_rates",
                elapsed_ms=100.0,
                memory_mb=50.0,
            ),
            EncodingBenchmarkResult(
                n_neurons=10,
                backend="jax",
                function_name="compute_spatial_rates",
                elapsed_ms=80.0,
                memory_mb=55.0,
            ),
        ]

        df = results_to_dataframe(results)

        assert len(df) == 2
        assert "n_neurons" in df.columns
        assert "backend" in df.columns
        assert "elapsed_ms" in df.columns
        assert "speedup" in df.columns or "memory_mb" in df.columns

    def test_print_summary_table(self, capsys) -> None:
        """Print summary should produce formatted output."""
        from bench_encoding_backends import (
            EncodingBenchmarkResult,
            print_summary_table,
        )

        results = [
            EncodingBenchmarkResult(
                n_neurons=10,
                backend="numpy",
                function_name="compute_spatial_rates",
                elapsed_ms=100.0,
                memory_mb=50.0,
            ),
        ]

        print_summary_table(results)
        captured = capsys.readouterr()

        # Should contain headers and data
        assert "numpy" in captured.out or "NumPy" in captured.out
        assert "10" in captured.out  # n_neurons
