#!/usr/bin/env python
"""Benchmark script for widget animation backend.

Measures:
- Initialization time (first frame render)
- Scrubbing responsiveness (frame update times)
- Peak memory usage

Note: Widget backend is designed for Jupyter notebooks. This script
benchmarks the underlying rendering functions directly.

Usage:
    uv run python benchmarks/bench_widget.py [--config small|medium|large]
    uv run python benchmarks/bench_widget.py --all
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Add benchmarks to path
sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    BenchmarkResult,
    TimingResult,
    create_test_data,
    force_gc,
    get_benchmark_configs,
    get_memory_mb,
    timer,
)


def benchmark_widget_render(
    env, fields, overlays, config
) -> tuple[float, float, list[float], float]:
    """Benchmark widget rendering.

    Returns
    -------
    first_render_ms : float
        Time for first frame render in milliseconds.
    avg_render_ms : float
        Average frame render time in milliseconds.
    render_times : list of float
        Individual render times for each frame.
    memory_mb : float
        Memory usage after rendering.
    """
    from neurospatial.animation.backends.widget_backend import (
        PersistentFigureRenderer,
    )
    from neurospatial.animation.overlays import _convert_overlays_to_data

    # Build frame times for overlay conversion
    n_frames = len(fields)
    frame_times = np.linspace(0, n_frames / 30.0, n_frames)

    # Convert overlays
    overlay_data = None
    if overlays:
        overlay_data = _convert_overlays_to_data(
            overlays=overlays, frame_times=frame_times, n_frames=n_frames, env=env
        )

    force_gc()
    get_memory_mb()

    # Create renderer (vmin=0, vmax=1 for normalized fields)
    renderer = PersistentFigureRenderer(
        env=env,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        dpi=72,
    )

    render_times: list[float] = []

    # Render frames and measure times
    n_render_frames = min(n_frames, 100)  # Limit to 100 frames for benchmark

    for frame_idx in range(n_render_frames):
        field = fields[frame_idx]
        start_time = time.perf_counter()

        # Use render method for PNG output
        _ = renderer.render(
            field,
            frame_idx=frame_idx,
            overlay_data=overlay_data,
            show_regions=False,
            region_alpha=0.3,
        )

        end_time = time.perf_counter()
        render_times.append((end_time - start_time) * 1000)

    renderer.close()

    first_render_ms = render_times[0] if render_times else 0.0
    avg_render_ms = float(np.mean(render_times)) if render_times else 0.0
    end_memory = get_memory_mb()

    return first_render_ms, avg_render_ms, render_times, end_memory


def benchmark_scrubbing(
    env, fields, overlays, config, n_scrubs: int = 50
) -> tuple[float, list[float]]:
    """Benchmark random scrubbing (simulating user interaction).

    Returns
    -------
    avg_scrub_ms : float
        Average scrub time in milliseconds.
    scrub_times : list of float
        Individual scrub times.
    """
    from neurospatial.animation.backends.widget_backend import (
        PersistentFigureRenderer,
    )
    from neurospatial.animation.overlays import _convert_overlays_to_data

    # Build frame times for overlay conversion
    n_frames = len(fields)
    frame_times = np.linspace(0, n_frames / 30.0, n_frames)

    # Convert overlays
    overlay_data = None
    if overlays:
        overlay_data = _convert_overlays_to_data(
            overlays=overlays, frame_times=frame_times, n_frames=n_frames, env=env
        )

    # Create renderer and warm up
    renderer = PersistentFigureRenderer(
        env=env,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        dpi=72,
    )

    # Warm up with first frame
    _ = renderer.render(
        fields[0],
        frame_idx=0,
        overlay_data=overlay_data,
        show_regions=False,
        region_alpha=0.3,
    )

    # Random scrubbing
    rng = np.random.default_rng(42)
    frame_indices = rng.integers(0, n_frames, size=n_scrubs)

    scrub_times: list[float] = []

    for frame_idx in frame_indices:
        start_time = time.perf_counter()
        _ = renderer.render(
            fields[frame_idx],
            frame_idx=int(frame_idx),
            overlay_data=overlay_data,
            show_regions=False,
            region_alpha=0.3,
        )
        end_time = time.perf_counter()
        scrub_times.append((end_time - start_time) * 1000)

    renderer.close()

    avg_scrub_ms = float(np.mean(scrub_times)) if scrub_times else 0.0

    return avg_scrub_ms, scrub_times


def run_widget_benchmark(config_name: str) -> BenchmarkResult:
    """Run widget benchmark for a specific config.

    Parameters
    ----------
    config_name : str
        One of "small", "medium", "large".

    Returns
    -------
    BenchmarkResult
        Timing results for this configuration.
    """
    result = BenchmarkResult(config_name=config_name, backend="widget")

    print(f"\nRunning widget benchmark: {config_name}")
    print("-" * 40)

    # Create test data
    with timer("data_creation") as t:
        env, fields, overlays, config = create_test_data(config_name)
    result.add(t)
    print(f"  Data creation: {t.elapsed_ms:.2f} ms")

    # For large configs, limit frames to keep benchmark time reasonable
    max_frames_for_benchmark = 500
    actual_n_frames = len(fields)
    if actual_n_frames > max_frames_for_benchmark:
        print(
            f"  Limiting frames from {actual_n_frames} to {max_frames_for_benchmark} for benchmark"
        )
        fields = fields[:max_frames_for_benchmark]
        # Skip overlays when truncating (overlay truncation is complex due to skeletons)
        print("  Skipping overlays for truncated benchmark")
        overlays = []

    # Render benchmark
    print("  Running render benchmark...")
    first_render_ms, avg_render_ms, render_times, memory_mb = benchmark_widget_render(
        env, fields, overlays, config
    )

    result.add(
        TimingResult(
            name="first_render",
            elapsed_ms=first_render_ms,
            memory_mb=memory_mb,
            peak_memory_mb=memory_mb,
        )
    )
    result.add(
        TimingResult(
            name=f"avg_render (n={len(render_times)})",
            elapsed_ms=avg_render_ms,
            memory_mb=0.0,
            peak_memory_mb=0.0,
        )
    )
    print(f"  First render: {first_render_ms:.2f} ms")
    print(f"  Average render: {avg_render_ms:.2f} ms")

    # Scrubbing benchmark
    n_scrubs = min(50, len(fields))
    print(f"  Running scrubbing benchmark (n={n_scrubs})...")
    avg_scrub_ms, scrub_times = benchmark_scrubbing(
        env, fields, overlays, config, n_scrubs=n_scrubs
    )

    result.add(
        TimingResult(
            name=f"avg_scrub (n={n_scrubs})",
            elapsed_ms=avg_scrub_ms,
            memory_mb=get_memory_mb(),
            peak_memory_mb=0.0,
        )
    )
    print(f"  Average scrub time: {avg_scrub_ms:.2f} ms")

    # Statistics on scrub times
    if scrub_times:
        p50 = np.percentile(scrub_times, 50)
        p95 = np.percentile(scrub_times, 95)
        print(f"  Scrub P50: {p50:.2f} ms, P95: {p95:.2f} ms")

    force_gc()

    return result


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark widget animation backend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run python benchmarks/bench_widget.py --config small
    uv run python benchmarks/bench_widget.py --all
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

    # Suppress matplotlib warnings
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)

    # Use non-interactive backend
    import matplotlib

    matplotlib.use("Agg")

    print("=" * 60)
    print("WIDGET BACKEND BENCHMARK")
    print("=" * 60)
    print(f"Machine: {sys.platform}")
    print(f"Python: {sys.version.split()[0]}")

    results: list[BenchmarkResult] = []

    configs_to_run = get_benchmark_configs() if args.all else [args.config]

    for config_name in configs_to_run:
        try:
            result = run_widget_benchmark(config_name)
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
