#!/usr/bin/env python
"""Benchmark script for LazyFieldRenderer performance.

Measures:
- Memory overhead during renderer creation
- Random access frame retrieval time
- Sequential access frame retrieval time
- Memmap performance characteristics

This benchmark characterizes performance of the LazyFieldRenderer used for
napari-based animations of large datasets.

Note: Dask renderer was evaluated and removed after benchmarks showed
LazyFieldRenderer significantly outperforms it (20-45,000x faster creation,
20-220x faster access). See git history for comparison benchmarks.

Usage:
    uv run python benchmarks/bench_lazy_renderers.py
    uv run python benchmarks/bench_lazy_renderers.py --config medium
    uv run python benchmarks/bench_lazy_renderers.py --all
    uv run python benchmarks/bench_lazy_renderers.py --memmap
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

# Add benchmarks to path
sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    BenchmarkResult,
    TimingResult,
    force_gc,
    get_memory_mb,
)


@dataclass
class RendererBenchmarkConfig:
    """Configuration for renderer benchmark.

    Parameters
    ----------
    name : str
        Human-readable name for this configuration.
    n_frames : int
        Number of animation frames.
    n_bins : int
        Number of spatial bins per frame.
    use_memmap : bool
        Whether to use memory-mapped array.
    """

    name: str
    n_frames: int
    n_bins: int
    use_memmap: bool = False


# Benchmark configurations for renderer comparison
RENDERER_CONFIGS = {
    "small": RendererBenchmarkConfig(
        name="small",
        n_frames=1_000,
        n_bins=500,
    ),
    "medium": RendererBenchmarkConfig(
        name="medium",
        n_frames=10_000,
        n_bins=500,
    ),
    "large": RendererBenchmarkConfig(
        name="large",
        n_frames=100_000,
        n_bins=500,
    ),
    "small_memmap": RendererBenchmarkConfig(
        name="small_memmap",
        n_frames=1_000,
        n_bins=500,
        use_memmap=True,
    ),
    "medium_memmap": RendererBenchmarkConfig(
        name="medium_memmap",
        n_frames=10_000,
        n_bins=500,
        use_memmap=True,
    ),
    "large_memmap": RendererBenchmarkConfig(
        name="large_memmap",
        n_frames=100_000,
        n_bins=500,
        use_memmap=True,
    ),
}


def check_napari_available() -> bool:
    """Check if napari is available.

    Returns
    -------
    has_napari : bool
        Whether napari is available.
    """
    try:
        import napari  # noqa: F401

        return True
    except ImportError:
        return False


def create_benchmark_env(n_bins: int, seed: int = 42):
    """Create a simple environment for benchmarking.

    Parameters
    ----------
    n_bins : int
        Target number of bins.
    seed : int
        Random seed.

    Returns
    -------
    Environment
        A fitted environment.
    """
    from neurospatial import Environment

    rng = np.random.default_rng(seed)

    # Estimate grid size to get approximately n_bins
    # For a square grid: n_bins ≈ grid_size^2, so grid_size ≈ sqrt(n_bins)
    grid_size = int(np.ceil(np.sqrt(n_bins)))

    # Create positions that span the grid
    n_samples = max(1000, grid_size * 10)
    positions = rng.uniform(0, grid_size, size=(n_samples, 2))

    env = Environment.from_samples(positions, bin_size=1.0)

    return env


def create_benchmark_fields(
    n_frames: int,
    n_bins: int,
    use_memmap: bool,
    tmpdir: Path | None,
    seed: int = 42,
) -> tuple[NDArray[np.float64], Path | None]:
    """Create benchmark field data.

    Parameters
    ----------
    n_frames : int
        Number of frames.
    n_bins : int
        Number of bins per frame.
    use_memmap : bool
        Whether to use memory-mapped array.
    tmpdir : Path or None
        Temporary directory for memmap files.
    seed : int
        Random seed.

    Returns
    -------
    fields : ndarray
        Field data of shape (n_frames, n_bins).
    memmap_path : Path or None
        Path to memmap file if created.
    """
    rng = np.random.default_rng(seed)

    if use_memmap:
        if tmpdir is None:
            raise ValueError("tmpdir required for memmap")
        memmap_path = tmpdir / "fields.dat"
        fields: NDArray[np.float64] = np.memmap(
            str(memmap_path),
            dtype=np.float64,
            mode="w+",
            shape=(n_frames, n_bins),
        )
        # Fill with random data in chunks to avoid memory spike
        chunk_size = 1000
        for start in range(0, n_frames, chunk_size):
            end = min(start + chunk_size, n_frames)
            fields[start:end] = rng.random((end - start, n_bins))
        fields.flush()
        return fields, memmap_path
    else:
        fields = rng.random((n_frames, n_bins))
        return fields, None


def create_colormap_lookup() -> NDArray[np.uint8]:
    """Create colormap lookup table.

    Returns
    -------
    cmap_lookup : ndarray of shape (256, 3)
        RGB lookup table.
    """
    cmap = plt.get_cmap("viridis")
    cmap_lookup: NDArray[np.uint8] = (cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(
        np.uint8
    )
    return cmap_lookup


def benchmark_renderer_creation(
    env: Any,
    fields: NDArray[np.float64],
    cmap_lookup: NDArray[np.uint8],
) -> tuple[TimingResult, Any]:
    """Benchmark renderer creation time and memory.

    Parameters
    ----------
    env : Environment
        The environment.
    fields : ndarray
        Field data.
    cmap_lookup : ndarray
        Colormap lookup table.

    Returns
    -------
    result : TimingResult
        Timing results.
    renderer : Any
        The created renderer.
    """
    from neurospatial.animation.backends.napari_backend import (
        _create_lazy_field_renderer,
    )

    force_gc()
    start_memory = get_memory_mb()
    start_time = time.perf_counter()

    renderer = _create_lazy_field_renderer(env, fields, cmap_lookup, vmin=0.0, vmax=1.0)

    end_time = time.perf_counter()
    end_memory = get_memory_mb()

    elapsed_ms = (end_time - start_time) * 1000
    memory_delta = end_memory - start_memory

    result = TimingResult(
        name="creation",
        elapsed_ms=elapsed_ms,
        memory_mb=memory_delta,
        peak_memory_mb=end_memory,
    )

    return result, renderer


def benchmark_random_access(
    renderer: Any,
    n_frames: int,
    n_accesses: int = 100,
) -> TimingResult:
    """Benchmark random frame access.

    Parameters
    ----------
    renderer : Any
        The LazyFieldRenderer.
    n_frames : int
        Total number of frames.
    n_accesses : int
        Number of random accesses to perform.

    Returns
    -------
    result : TimingResult
        Timing results with mean access time in elapsed_ms.
    """
    rng = np.random.default_rng(42)
    indices = rng.integers(0, n_frames, size=n_accesses)

    # Warm up with a few accesses
    for idx in indices[:3]:
        _ = renderer[idx]

    force_gc()
    times = []

    for idx in indices:
        start = time.perf_counter()
        _ = renderer[idx]
        times.append(time.perf_counter() - start)

    mean_ms = np.mean(times) * 1000

    return TimingResult(
        name=f"random_access_mean (n={n_accesses})",
        elapsed_ms=mean_ms,
        memory_mb=get_memory_mb(),
        peak_memory_mb=0.0,
    )


def benchmark_sequential_access(
    renderer: Any,
    n_frames: int,
    n_accesses: int = 100,
) -> TimingResult:
    """Benchmark sequential frame access.

    Parameters
    ----------
    renderer : Any
        The LazyFieldRenderer.
    n_frames : int
        Total number of frames.
    n_accesses : int
        Number of sequential accesses to perform.

    Returns
    -------
    result : TimingResult
        Timing results with mean access time in elapsed_ms.
    """
    force_gc()
    times = []

    for idx in range(min(n_accesses, n_frames)):
        start = time.perf_counter()
        _ = renderer[idx]
        times.append(time.perf_counter() - start)

    mean_ms = np.mean(times) * 1000
    return TimingResult(
        name=f"sequential_access_mean (n={len(times)})",
        elapsed_ms=mean_ms,
        memory_mb=get_memory_mb(),
        peak_memory_mb=0.0,
    )


def benchmark_scrubbing_simulation(
    renderer: Any,
    n_frames: int,
    n_seeks: int = 50,
) -> TimingResult:
    """Benchmark rapid scrubbing (simulating user scrolling through timeline).

    Parameters
    ----------
    renderer : Any
        The renderer.
    n_frames : int
        Total number of frames.
    n_seeks : int
        Number of seek operations.

    Returns
    -------
    result : TimingResult
        Timing results.
    """
    # Simulate scrubbing pattern: jump to different parts of the timeline
    rng = np.random.default_rng(42)

    # Create a "scrubbing" pattern - back and forth with some jumps
    positions = []
    current = 0
    for _ in range(n_seeks):
        # Random step size (simulating varying scrub speeds)
        step = rng.integers(-100, 100)
        current = max(0, min(n_frames - 1, current + step))
        positions.append(current)

    force_gc()
    times = []

    for idx in positions:
        start = time.perf_counter()
        _ = renderer[idx]
        times.append(time.perf_counter() - start)

    mean_ms = np.mean(times) * 1000

    return TimingResult(
        name=f"scrubbing_mean (n={n_seeks})",
        elapsed_ms=mean_ms,
        memory_mb=get_memory_mb(),
        peak_memory_mb=0.0,
    )


def run_renderer_benchmark(
    config: RendererBenchmarkConfig,
) -> BenchmarkResult:
    """Run benchmark for a specific configuration.

    Parameters
    ----------
    config : RendererBenchmarkConfig
        Benchmark configuration.

    Returns
    -------
    result : BenchmarkResult
        Benchmark results for LazyFieldRenderer.
    """
    print(f"\n{'=' * 60}")
    print(f"Benchmark: {config.name}")
    print(f"  n_frames={config.n_frames:,}, n_bins={config.n_bins:,}")
    print(f"  memmap={config.use_memmap}")
    print("=" * 60)

    result = BenchmarkResult(config_name=config.name, backend="lazy")

    # Setup temporary directory for memmap if needed
    tmpdir_context = tempfile.TemporaryDirectory() if config.use_memmap else None
    tmpdir = Path(tmpdir_context.name) if tmpdir_context else None

    try:
        # Create environment
        print("\nCreating environment...")
        env = create_benchmark_env(config.n_bins)
        actual_n_bins = env.n_bins
        print(f"  Created env with {actual_n_bins:,} bins")

        # Create fields
        print("Creating fields...")
        force_gc()
        mem_before = get_memory_mb()
        fields, memmap_path = create_benchmark_fields(
            config.n_frames,
            actual_n_bins,
            config.use_memmap,
            tmpdir,
        )
        mem_after = get_memory_mb()
        print(f"  Fields shape: {fields.shape}")
        print(f"  Memory delta: {mem_after - mem_before:.2f} MB")
        if config.use_memmap:
            print(f"  Memmap file: {memmap_path}")

        # Create colormap
        cmap_lookup = create_colormap_lookup()

        print("\n--- LazyFieldRenderer ---")

        # Creation benchmark
        print("  Creating renderer...")
        try:
            timing, renderer = benchmark_renderer_creation(env, fields, cmap_lookup)
            result.add(timing)
            print(
                f"  Creation: {timing.elapsed_ms:.2f} ms, "
                f"memory delta: {timing.memory_mb:.2f} MB"
            )
        except Exception as e:
            print(f"  ERROR creating renderer: {e}")
            raise

        # Random access benchmark
        print("  Testing random access...")
        timing = benchmark_random_access(renderer, config.n_frames)
        result.add(timing)
        print(f"  Random access mean: {timing.elapsed_ms:.4f} ms")

        # Sequential access benchmark
        print("  Testing sequential access...")
        timing = benchmark_sequential_access(renderer, config.n_frames)
        result.add(timing)
        print(f"  Sequential access mean: {timing.elapsed_ms:.4f} ms")

        # Scrubbing benchmark
        print("  Testing scrubbing simulation...")
        timing = benchmark_scrubbing_simulation(renderer, config.n_frames)
        result.add(timing)
        print(f"  Scrubbing mean: {timing.elapsed_ms:.4f} ms")

        # Cleanup
        del renderer
        force_gc()

    finally:
        # Cleanup temp directory
        if tmpdir_context:
            tmpdir_context.cleanup()

    return result


def print_summary_table(all_results: dict[str, BenchmarkResult]) -> None:
    """Print a summary table across all configurations.

    Parameters
    ----------
    all_results : dict
        Mapping from config name to BenchmarkResult.
    """
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\n## Performance Summary (ms)")
    print("| Config | Creation | Random Access | Sequential | Scrubbing |")
    print("|--------|----------|---------------|------------|-----------|")

    for config_name, result in all_results.items():
        creation = random_access = sequential = scrubbing = "N/A"

        for t in result.timings:
            if "creation" in t.name:
                creation = f"{t.elapsed_ms:.2f}"
            elif "random_access" in t.name:
                random_access = f"{t.elapsed_ms:.4f}"
            elif "sequential" in t.name:
                sequential = f"{t.elapsed_ms:.4f}"
            elif "scrubbing" in t.name:
                scrubbing = f"{t.elapsed_ms:.4f}"

        print(
            f"| {config_name} | {creation} | {random_access} | {sequential} | {scrubbing} |"
        )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark LazyFieldRenderer performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run python benchmarks/bench_lazy_renderers.py
    uv run python benchmarks/bench_lazy_renderers.py --config medium
    uv run python benchmarks/bench_lazy_renderers.py --all
    uv run python benchmarks/bench_lazy_renderers.py --memmap
        """,
    )
    parser.add_argument(
        "--config",
        choices=list(RENDERER_CONFIGS.keys()),
        default="medium",
        help="Benchmark configuration to run (default: medium)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all benchmark configurations (including memmap)",
    )
    parser.add_argument(
        "--memmap",
        action="store_true",
        help="Run only memmap configurations",
    )

    args = parser.parse_args()

    has_napari = check_napari_available()

    if not has_napari:
        print("WARNING: napari not available, some tests may fail")

    print("=" * 60)
    print("LAZY FIELD RENDERER BENCHMARK")
    print("=" * 60)
    print(f"Machine: {sys.platform}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"napari available: {has_napari}")

    # Determine which configs to run
    if args.all:
        configs_to_run = list(RENDERER_CONFIGS.keys())
    elif args.memmap:
        configs_to_run = [k for k in RENDERER_CONFIGS if "memmap" in k]
    else:
        configs_to_run = [args.config]

    all_results: dict[str, BenchmarkResult] = {}

    for config_name in configs_to_run:
        try:
            config = RENDERER_CONFIGS[config_name]
            result = run_renderer_benchmark(config)
            all_results[config_name] = result
        except Exception as e:
            print(f"\nERROR running {config_name}: {e}")
            import traceback

            traceback.print_exc()

    # Print summary table
    if len(all_results) > 1:
        print_summary_table(all_results)

    # Print individual tables
    print("\n" + "=" * 60)
    print("DETAILED RESULTS")
    print("=" * 60)
    for _config_name, result in all_results.items():
        result.print_table()

    # Print key insights
    print("\n" + "=" * 60)
    print("KEY INSIGHTS")
    print("=" * 60)
    print("""
LazyFieldRenderer characteristics:

1. Near-zero creation overhead (~0.01 ms for any size)
   - No graph construction or pre-computation
   - Just stores references to input data

2. Sub-millisecond frame access
   - Random access: ~0.02-0.2 ms per frame
   - Sequential access: ~0.02 ms per frame
   - Scrubbing: ~0.02-0.2 ms per frame

3. Memory-mapped arrays work seamlessly
   - Same performance characteristics as in-memory arrays
   - Enables working with datasets larger than RAM

4. LRU caching (default 100 frames) provides fast repeated access
   - Configurable via chunk_size parameter
""")


if __name__ == "__main__":
    main()
