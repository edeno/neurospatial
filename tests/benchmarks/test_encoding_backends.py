"""Tests for encoding backend benchmark script.

This module validates that the benchmark script runs correctly and produces
expected output format.

TDD approach: Test first, then implement the benchmark script.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

BENCHMARKS_DIR = Path(__file__).parent.parent.parent / "benchmarks"
BENCHMARK_MODULE_PATH = BENCHMARKS_DIR / "bench_encoding_backends.py"
if not BENCHMARKS_DIR.exists():
    raise RuntimeError(f"Benchmarks directory not found: {BENCHMARKS_DIR}")
if not BENCHMARK_MODULE_PATH.exists():
    raise RuntimeError(f"Benchmark module not found: {BENCHMARK_MODULE_PATH}")

from neurospatial.encoding._backend import is_jax_available  # noqa: E402

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def benchmark_module() -> ModuleType:
    """Load the benchmark script without mutating sys.path globally."""
    module_name = "_neurospatial_test_bench_encoding_backends"
    spec = importlib.util.spec_from_file_location(module_name, BENCHMARK_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not load benchmark module from {BENCHMARK_MODULE_PATH}"
        )

    module = importlib.util.module_from_spec(spec)
    previous_module = sys.modules.get(module_name)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    yield module

    if previous_module is None:
        sys.modules.pop(module_name, None)
    else:
        sys.modules[module_name] = previous_module


@pytest.fixture(autouse=True)
def enable_jax_x64():
    """Enable JAX x64 mode for fair comparison with NumPy."""
    if not is_jax_available():
        yield
        return

    import jax

    previous_value = bool(jax.config.read("jax_enable_x64"))
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous_value)


class TestBenchmarkRunner:
    """Tests that benchmark script components work correctly."""

    def test_benchmark_imports(self, benchmark_module: ModuleType) -> None:
        """Benchmark module should import without errors."""
        assert hasattr(benchmark_module, "run_encoding_benchmark")
        assert hasattr(benchmark_module, "EncodingBenchmarkResult")

    def test_create_benchmark_data(self, benchmark_module: ModuleType) -> None:
        """Benchmark data creation should work for different population sizes."""
        create_benchmark_data = benchmark_module.create_benchmark_data

        # Test small population
        data = create_benchmark_data(n_neurons=10, n_samples=500, seed=42)

        assert "env" in data
        assert "spike_times_list" in data
        assert "times" in data
        assert "positions" in data
        assert len(data["spike_times_list"]) == 10

    def test_benchmark_result_structure(self, benchmark_module: ModuleType) -> None:
        """Benchmark result dataclass should have expected fields."""
        EncodingBenchmarkResult = benchmark_module.EncodingBenchmarkResult

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

    def test_run_single_benchmark_numpy(self, benchmark_module: ModuleType) -> None:
        """Benchmark runner should work with NumPy backend."""
        create_benchmark_data = benchmark_module.create_benchmark_data
        run_single_benchmark = benchmark_module.run_single_benchmark

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

    def test_run_single_benchmark_glm_numpy(self, benchmark_module: ModuleType) -> None:
        """The backend benchmark accepts fixed-penalty glm workloads."""
        data = benchmark_module.create_benchmark_data(
            n_neurons=3, n_samples=500, seed=43
        )
        result = benchmark_module.run_single_benchmark(
            data=data,
            function_name="compute_spatial_rates",
            backend="numpy",
            n_iterations=1,
            method="glm",
            penalty=1.0,
            rank=10,
        )

        assert result.backend == "numpy"
        assert result.elapsed_ms > 0
        assert result.n_neurons == 3

    def test_run_single_benchmark_glm_per_unit(
        self, benchmark_module: ModuleType
    ) -> None:
        """The backend benchmark forwards the per-unit REML workload
        (``pooled=False`` with automatic REML), so a regression cannot silently
        benchmark shared REML instead."""
        data = benchmark_module.create_benchmark_data(
            n_neurons=3, n_samples=500, seed=44
        )
        result = benchmark_module.run_single_benchmark(
            data=data,
            function_name="compute_spatial_rates",
            backend="numpy",
            n_iterations=1,
            method="glm",
            penalty=None,  # automatic REML -> pooled matters
            rank=10,
            pooled=False,
        )

        assert result.backend == "numpy"
        assert result.elapsed_ms > 0
        assert result.n_neurons == 3

    @pytest.mark.skipif(
        not is_jax_available(),
        reason="JAX is not available on this platform",
    )
    def test_run_single_benchmark_jax(self, benchmark_module: ModuleType) -> None:
        """Benchmark runner should work with JAX backend."""
        create_benchmark_data = benchmark_module.create_benchmark_data
        run_single_benchmark = benchmark_module.run_single_benchmark

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
    def test_benchmark_with_different_populations(
        self, benchmark_module: ModuleType, n_neurons: int
    ) -> None:
        """Benchmark should work with varying population sizes."""
        create_benchmark_data = benchmark_module.create_benchmark_data
        run_single_benchmark = benchmark_module.run_single_benchmark

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

    def test_results_to_dataframe(self, benchmark_module: ModuleType) -> None:
        """Benchmark results should convert to DataFrame correctly."""
        EncodingBenchmarkResult = benchmark_module.EncodingBenchmarkResult
        results_to_dataframe = benchmark_module.results_to_dataframe

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

    def test_print_summary_table(self, benchmark_module: ModuleType, capsys) -> None:
        """Print summary should produce formatted output."""
        EncodingBenchmarkResult = benchmark_module.EncodingBenchmarkResult
        print_summary_table = benchmark_module.print_summary_table

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
