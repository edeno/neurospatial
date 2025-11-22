#!/usr/bin/env python
"""Benchmark script for napari animation backend.

Measures:
- Initialization time (viewer + layer setup)
- Random seek time (frame switching)
- Peak memory usage

Usage:
    uv run python benchmarks/bench_napari.py [--config small|medium|large]
    uv run python benchmarks/bench_napari.py --all
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Add benchmarks to path
sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    BenchmarkResult,
    create_test_data,
    force_gc,
    get_benchmark_configs,
    get_memory_mb,
    timer,
)


def check_napari_available() -> bool:
    """Check if napari is available."""
    try:
        import napari  # noqa: F401

        return True
    except ImportError:
        return False


def benchmark_napari_init(
    env, fields, overlays, config
) -> tuple[float, float, float, Any]:
    """Benchmark napari viewer initialization.

    Returns
    -------
    elapsed_ms : float
        Time to initialize viewer in milliseconds.
    memory_mb : float
        Memory usage after initialization.
    peak_memory_mb : float
        Peak memory during initialization.
    viewer : napari.Viewer
        The created viewer (for subsequent tests).
    """
    from neurospatial.animation.core import animate_fields

    force_gc()
    start_memory = get_memory_mb()
    start_time = time.perf_counter()

    # Clear cache to ensure pickleable (standard practice)
    env.clear_cache()

    viewer = animate_fields(
        env,
        fields,
        backend="napari",
        overlays=overlays,
        fps=30,
    )

    end_time = time.perf_counter()
    end_memory = get_memory_mb()

    elapsed_ms = (end_time - start_time) * 1000
    memory_mb = end_memory
    peak_memory_mb = max(start_memory, end_memory)

    return elapsed_ms, memory_mb, peak_memory_mb, viewer


def benchmark_random_seek(viewer, n_frames: int, n_seeks: int = 100) -> float:
    """Benchmark random frame seeking.

    Parameters
    ----------
    viewer : napari.Viewer
        Initialized viewer.
    n_frames : int
        Total number of frames.
    n_seeks : int, default=100
        Number of random seeks to perform.

    Returns
    -------
    avg_seek_ms : float
        Average seek time in milliseconds.
    """
    rng = np.random.default_rng(42)
    frame_indices = rng.integers(0, n_frames, size=n_seeks)

    # Warm up
    viewer.dims.set_point(0, 0)
    viewer.dims.set_point(0, 1)

    start_time = time.perf_counter()

    for frame_idx in frame_indices:
        viewer.dims.set_point(0, frame_idx)

    end_time = time.perf_counter()

    total_ms = (end_time - start_time) * 1000
    avg_seek_ms = total_ms / n_seeks

    return avg_seek_ms


def run_napari_benchmark(config_name: str) -> BenchmarkResult:
    """Run napari benchmark for a specific config.

    Parameters
    ----------
    config_name : str
        One of "small", "medium", "large".

    Returns
    -------
    BenchmarkResult
        Timing results for this configuration.
    """
    result = BenchmarkResult(config_name=config_name, backend="napari")

    print(f"\nRunning napari benchmark: {config_name}")
    print("-" * 40)

    # Create test data
    with timer("data_creation") as t:
        env, fields, overlays, config = create_test_data(config_name)
    result.add(t)
    print(f"  Data creation: {t.elapsed_ms:.2f} ms")

    # Initialize viewer
    print("  Initializing napari viewer...")
    elapsed_ms, memory_mb, peak_memory_mb, viewer = benchmark_napari_init(
        env, fields, overlays, config
    )
    from utils import TimingResult

    result.add(
        TimingResult(
            name="viewer_init",
            elapsed_ms=elapsed_ms,
            memory_mb=memory_mb,
            peak_memory_mb=peak_memory_mb,
        )
    )
    print(f"  Viewer init: {elapsed_ms:.2f} ms, memory: {memory_mb:.2f} MB")

    # Random seek benchmark
    n_frames = config.n_frames
    n_seeks = min(100, n_frames)  # Don't seek more than available frames
    print(f"  Running {n_seeks} random seeks...")
    avg_seek_ms = benchmark_random_seek(viewer, n_frames, n_seeks=n_seeks)
    result.add(
        TimingResult(
            name=f"random_seek_avg (n={n_seeks})",
            elapsed_ms=avg_seek_ms,
            memory_mb=get_memory_mb(),
            peak_memory_mb=0.0,
        )
    )
    print(f"  Average seek time: {avg_seek_ms:.4f} ms")

    # Cleanup
    viewer.close()
    force_gc()

    return result


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark napari animation backend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run python benchmarks/bench_napari.py --config small
    uv run python benchmarks/bench_napari.py --all
        """,
    )
    parser.add_argument(
        "--config",
        choices=get_benchmark_configs(),
        default="small",
        help="Benchmark configuration to run",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all benchmark configurations",
    )

    args = parser.parse_args()

    if not check_napari_available():
        print("ERROR: napari is not available. Install with: uv add napari")
        sys.exit(1)

    # Suppress Qt warnings during benchmarking
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)

    print("=" * 60)
    print("NAPARI BACKEND BENCHMARK")
    print("=" * 60)
    print(f"Machine: {sys.platform}")
    print(f"Python: {sys.version.split()[0]}")

    results: list[BenchmarkResult] = []

    configs_to_run = get_benchmark_configs() if args.all else [args.config]

    for config_name in configs_to_run:
        try:
            result = run_napari_benchmark(config_name)
            results.append(result)
        except Exception as e:
            print(f"\nERROR running {config_name}: {e}")
            import traceback

            traceback.print_exc()

    # Print summary tables
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for result in results:
        result.print_table()


if __name__ == "__main__":
    main()
