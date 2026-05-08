#!/usr/bin/env python
"""Benchmark script for encoding backend performance.

Compares NumPy vs JAX backends for spatial rate computation
across different population sizes.

Measures:
- Total computation time
- Time per neuron
- Memory usage

Usage:
    uv run python benchmarks/bench_encoding_backends.py
    uv run python benchmarks/bench_encoding_backends.py --n-neurons 10 100 1000
    uv run python benchmarks/bench_encoding_backends.py --no-jax

Example output:
    ============================================================
    ENCODING BACKEND BENCHMARK
    ============================================================
    Machine: darwin
    Python: 3.13.0
    JAX available: True

    Running benchmarks...

    ## compute_spatial_rates

    | n_neurons | Backend | Time (ms) | ms/neuron | Memory (MB) | Speedup |
    |-----------|---------|-----------|-----------|-------------|---------|
    | 10        | numpy   | 45.2      | 4.52      | 120.5       | 1.00x   |
    | 10        | jax     | 32.1      | 3.21      | 135.2       | 1.41x   |
    | 100       | numpy   | 412.5     | 4.13      | 145.3       | 1.00x   |
    | 100       | jax     | 285.3     | 2.85      | 168.7       | 1.45x   |
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import psutil

from neurospatial import Environment
from neurospatial.encoding._backend import is_jax_available

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd
    from numpy.typing import NDArray

# Type aliases for Literal types
SmoothingMethod = Literal["diffusion_kde", "gaussian_kde", "binned"]
Backend = Literal["numpy", "jax", "auto"]


# =============================================================================
# Data Types
# =============================================================================


@dataclass
class EncodingBenchmarkResult:
    """Result from a single benchmark run.

    Parameters
    ----------
    n_neurons : int
        Number of neurons in the population.
    backend : str
        Backend used ("numpy" or "jax").
    function_name : str
        Name of the function benchmarked.
    elapsed_ms : float
        Total elapsed time in milliseconds.
    memory_mb : float
        Memory usage in megabytes after computation.
    n_iterations : int, optional
        Number of iterations averaged over. Default is 1.
    """

    n_neurons: int
    backend: str
    function_name: str
    elapsed_ms: float
    memory_mb: float
    n_iterations: int = 1

    @property
    def ms_per_neuron(self) -> float:
        """Time per neuron in milliseconds."""
        return self.elapsed_ms / self.n_neurons if self.n_neurons > 0 else 0.0


# =============================================================================
# Benchmark Data Creation
# =============================================================================


def create_benchmark_data(
    n_neurons: int = 100,
    n_samples: int = 5000,
    n_bins: int = 100,
    duration: float = 60.0,
    mean_firing_rate: float = 5.0,
    seed: int = 42,
) -> dict:
    """Create benchmark data for encoding tests.

    Parameters
    ----------
    n_neurons : int, default=100
        Number of neurons to simulate.
    n_samples : int, default=5000
        Number of position samples (trajectory length).
    n_bins : int, default=100
        Approximate number of spatial bins (determines env size).
    duration : float, default=60.0
        Duration of recording in seconds.
    mean_firing_rate : float, default=5.0
        Mean firing rate in Hz for simulated neurons.
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary with keys:
        - env: Environment
        - spike_times_list: list of spike time arrays
        - times: array of timestamps
        - positions: array of positions
    """
    rng = np.random.default_rng(seed)

    # Calculate environment size to get ~n_bins
    bin_size = 5.0
    env_size = np.sqrt(n_bins) * bin_size

    # Create trajectory
    positions = rng.uniform(0, env_size, size=(n_samples, 2))
    times = np.linspace(0, duration, n_samples)

    # Create environment
    env = Environment.from_samples(positions, bin_size=bin_size)

    # Create spike trains
    spike_times_list: list[NDArray[np.float64]] = []
    for _ in range(n_neurons):
        # Poisson process with variable rate
        neuron_rate = rng.uniform(0.5 * mean_firing_rate, 1.5 * mean_firing_rate)
        n_spikes = rng.poisson(neuron_rate * duration)
        spikes = np.sort(rng.uniform(0, duration, size=n_spikes))
        spike_times_list.append(spikes)

    return {
        "env": env,
        "spike_times_list": spike_times_list,
        "times": times,
        "positions": positions,
    }


# =============================================================================
# Benchmark Runner
# =============================================================================


def _get_memory_mb() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process()
    return float(process.memory_info().rss / (1024 * 1024))


def _force_gc() -> None:
    """Force garbage collection."""
    gc.collect()
    gc.collect()
    gc.collect()


def run_single_benchmark(
    data: dict,
    function_name: str,
    backend: Backend,
    n_iterations: int = 3,
    warmup: bool = True,
    smoothing_method: SmoothingMethod = "binned",
    bandwidth: float = 5.0,
) -> EncodingBenchmarkResult:
    """Run a single benchmark.

    Parameters
    ----------
    data : dict
        Benchmark data from create_benchmark_data().
    function_name : str
        Name of the function to benchmark.
    backend : str
        Backend to use ("numpy" or "jax").
    n_iterations : int, default=3
        Number of iterations to average over.
    warmup : bool, default=True
        Whether to run a warmup iteration first.
    smoothing_method : str, default="binned"
        Smoothing method to use ("binned", "diffusion_kde", "gaussian_kde").
    bandwidth : float, default=5.0
        Bandwidth for smoothing methods (not used for "binned").

    Returns
    -------
    EncodingBenchmarkResult
        Benchmark results.
    """
    from neurospatial.encoding.spatial import compute_spatial_rates

    env = data["env"]
    spike_times_list = data["spike_times_list"]
    times = data["times"]
    positions = data["positions"]
    n_neurons = len(spike_times_list)

    # Get the function to benchmark
    if function_name == "compute_spatial_rates":

        def benchmark_fn():
            return compute_spatial_rates(
                env,
                spike_times_list,
                times,
                positions,
                smoothing_method=smoothing_method,
                bandwidth=bandwidth,
                backend=backend,
            )

    else:
        raise ValueError(f"Unknown function: {function_name}")

    # Warmup run (excludes JIT compilation time for JAX)
    if warmup:
        _force_gc()
        _ = benchmark_fn()

    # Timed runs
    _force_gc()
    timings = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        _ = benchmark_fn()
        end = time.perf_counter()
        timings.append((end - start) * 1000)  # Convert to ms

    elapsed_ms = float(np.mean(timings))
    memory_mb = _get_memory_mb()

    return EncodingBenchmarkResult(
        n_neurons=n_neurons,
        backend=backend,
        function_name=function_name,
        elapsed_ms=elapsed_ms,
        memory_mb=memory_mb,
        n_iterations=n_iterations,
    )


def run_encoding_benchmark(
    n_neurons_list: Sequence[int] = (10, 100, 1000),
    backends: Sequence[Backend] = ("numpy", "jax"),
    functions: Sequence[str] = ("compute_spatial_rates",),
    n_iterations: int = 3,
    seed: int = 42,
    smoothing_method: SmoothingMethod = "binned",
    bandwidth: float = 5.0,
) -> list[EncodingBenchmarkResult]:
    """Run full encoding benchmark suite.

    Parameters
    ----------
    n_neurons_list : sequence of int, default=(10, 100, 1000)
        Population sizes to benchmark.
    backends : sequence of str, default=("numpy", "jax")
        Backends to compare.
    functions : sequence of str, default=("compute_spatial_rates",)
        Functions to benchmark.
    n_iterations : int, default=3
        Number of iterations per benchmark.
    seed : int, default=42
        Random seed for reproducibility.
    smoothing_method : str, default="binned"
        Smoothing method to use ("binned", "diffusion_kde", "gaussian_kde").
    bandwidth : float, default=5.0
        Bandwidth for smoothing methods (not used for "binned").

    Returns
    -------
    list of EncodingBenchmarkResult
        All benchmark results.
    """
    results = []

    for function_name in functions:
        print(f"\nBenchmarking {function_name} (smoothing={smoothing_method})...")

        for n_neurons in n_neurons_list:
            print(f"  n_neurons={n_neurons}")
            data = create_benchmark_data(n_neurons=n_neurons, seed=seed)

            for backend in backends:
                # Skip JAX if not available
                if backend == "jax" and not is_jax_available():
                    print(f"    Skipping {backend} (not available)")
                    continue

                print(f"    {backend}...", end=" ", flush=True)
                try:
                    result = run_single_benchmark(
                        data=data,
                        function_name=function_name,
                        backend=backend,
                        n_iterations=n_iterations,
                        smoothing_method=smoothing_method,
                        bandwidth=bandwidth,
                    )
                    results.append(result)
                    print(f"{result.elapsed_ms:.1f} ms")
                except Exception as e:
                    print(f"FAILED: {e}")

    return results


# =============================================================================
# Output Formatting
# =============================================================================


def results_to_dataframe(results: list[EncodingBenchmarkResult]) -> pd.DataFrame:
    """Convert benchmark results to a pandas DataFrame.

    Parameters
    ----------
    results : list of EncodingBenchmarkResult
        Benchmark results.

    Returns
    -------
    pd.DataFrame
        DataFrame with benchmark data.
    """
    import pandas as pd

    data = {
        "function_name": [r.function_name for r in results],
        "n_neurons": [r.n_neurons for r in results],
        "backend": [r.backend for r in results],
        "elapsed_ms": [r.elapsed_ms for r in results],
        "ms_per_neuron": [r.ms_per_neuron for r in results],
        "memory_mb": [r.memory_mb for r in results],
        "n_iterations": [r.n_iterations for r in results],
    }

    df = pd.DataFrame(data)

    # Calculate speedup relative to NumPy
    speedups = []
    for _, row in df.iterrows():
        if row["backend"] == "numpy":
            speedups.append(1.0)
        else:
            numpy_time = df[
                (df["function_name"] == row["function_name"])
                & (df["n_neurons"] == row["n_neurons"])
                & (df["backend"] == "numpy")
            ]["elapsed_ms"]
            if len(numpy_time) > 0:
                speedups.append(numpy_time.values[0] / row["elapsed_ms"])
            else:
                speedups.append(np.nan)

    df["speedup"] = speedups

    return df


def print_summary_table(results: list[EncodingBenchmarkResult]) -> None:
    """Print benchmark results as a formatted table.

    Parameters
    ----------
    results : list of EncodingBenchmarkResult
        Benchmark results.
    """
    if not results:
        print("No results to display.")
        return

    df = results_to_dataframe(results)

    # Group by function
    for function_name in df["function_name"].unique():
        print(f"\n## {function_name}\n")
        print("| n_neurons | Backend | Time (ms) | ms/neuron | Memory (MB) | Speedup |")
        print("|-----------|---------|-----------|-----------|-------------|---------|")

        func_df = df[df["function_name"] == function_name].sort_values(
            ["n_neurons", "backend"]
        )

        for _, row in func_df.iterrows():
            speedup_str = (
                f"{row['speedup']:.2f}x" if not np.isnan(row["speedup"]) else "N/A"
            )
            print(
                f"| {row['n_neurons']:>9} | {row['backend']:<7} | "
                f"{row['elapsed_ms']:>9.1f} | {row['ms_per_neuron']:>9.2f} | "
                f"{row['memory_mb']:>11.1f} | {speedup_str:>7} |"
            )


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> None:
    """Main entry point for benchmark script."""
    parser = argparse.ArgumentParser(
        description="Benchmark encoding backend performance (NumPy vs JAX)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run python benchmarks/bench_encoding_backends.py
    uv run python benchmarks/bench_encoding_backends.py --n-neurons 10 100 1000
    uv run python benchmarks/bench_encoding_backends.py --no-jax
    uv run python benchmarks/bench_encoding_backends.py --iterations 5
        """,
    )
    parser.add_argument(
        "--n-neurons",
        type=int,
        nargs="+",
        default=[10, 100, 1000],
        help="Population sizes to benchmark (default: 10 100 1000)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of iterations to average (default: 3)",
    )
    parser.add_argument(
        "--no-jax",
        action="store_true",
        help="Skip JAX backend benchmarks",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Save results to CSV file",
    )
    parser.add_argument(
        "--smoothing",
        type=str,
        choices=["binned", "diffusion_kde", "gaussian_kde"],
        default="binned",
        help="Smoothing method (default: binned)",
    )
    parser.add_argument(
        "--bandwidth",
        type=float,
        default=5.0,
        help="Smoothing bandwidth in cm (default: 5.0)",
    )

    args = parser.parse_args()

    # Print header
    print("=" * 60)
    print("ENCODING BACKEND BENCHMARK")
    print("=" * 60)
    print(f"Machine: {sys.platform}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"JAX available: {is_jax_available()}")
    print(f"Population sizes: {args.n_neurons}")
    print(f"Iterations per benchmark: {args.iterations}")
    print(f"Smoothing method: {args.smoothing}")
    if args.smoothing != "binned":
        print(f"Bandwidth: {args.bandwidth} cm")

    # Determine backends to test
    backends: list[Backend] = ["numpy"]
    if not args.no_jax and is_jax_available():
        # Enable float64 for fair comparison
        try:
            import jax

            jax.config.update("jax_enable_x64", True)
            print("JAX x64 mode: enabled")
        except ImportError:
            pass
        backends.append("jax")

    # Run benchmarks
    results = run_encoding_benchmark(
        n_neurons_list=args.n_neurons,
        backends=backends,
        n_iterations=args.iterations,
        seed=args.seed,
        smoothing_method=args.smoothing,
        bandwidth=args.bandwidth,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print_summary_table(results)

    # Save to CSV if requested
    if args.csv and results:
        df = results_to_dataframe(results)
        df.to_csv(args.csv, index=False)
        print(f"\nResults saved to {args.csv}")


if __name__ == "__main__":
    main()
