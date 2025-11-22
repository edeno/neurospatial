#!/usr/bin/env python
"""Benchmark script for video animation backend.

Measures:
- Total export time
- Time per frame
- Peak memory usage

Usage:
    uv run python benchmarks/bench_video.py [--config small|medium|large]
    uv run python benchmarks/bench_video.py --all
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

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


def check_ffmpeg_available() -> bool:
    """Check if ffmpeg is available."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def benchmark_video_export(
    env, fields, overlays, config, output_path: Path, n_workers: int = 1
) -> tuple[float, float, float, float]:
    """Benchmark video export.

    Returns
    -------
    total_ms : float
        Total export time in milliseconds.
    time_per_frame_ms : float
        Time per frame in milliseconds.
    memory_mb : float
        Memory usage after export.
    file_size_mb : float
        Output file size in MB.
    """
    from neurospatial.animation.core import animate_fields

    force_gc()
    get_memory_mb()
    start_time = time.perf_counter()

    # Clear cache for parallel rendering
    env.clear_cache()

    animate_fields(
        env,
        fields,
        backend="video",
        overlays=overlays,
        save_path=str(output_path),
        fps=30,
        n_workers=n_workers,
        dpi=72,  # Lower DPI for faster benchmarking
    )

    end_time = time.perf_counter()
    end_memory = get_memory_mb()

    total_ms = (end_time - start_time) * 1000
    n_frames = config.n_frames
    time_per_frame_ms = total_ms / n_frames if n_frames > 0 else 0.0
    file_size_mb = output_path.stat().st_size / (1024 * 1024)

    return total_ms, time_per_frame_ms, end_memory, file_size_mb


def run_video_benchmark(config_name: str) -> BenchmarkResult:
    """Run video benchmark for a specific config.

    Parameters
    ----------
    config_name : str
        One of "small", "medium", "large".

    Returns
    -------
    BenchmarkResult
        Timing results for this configuration.
    """
    result = BenchmarkResult(config_name=config_name, backend="video")

    print(f"\nRunning video benchmark: {config_name}")
    print("-" * 40)

    # Create test data
    with timer("data_creation") as t:
        env, fields, overlays, config = create_test_data(config_name)
    result.add(t)
    print(f"  Data creation: {t.elapsed_ms:.2f} ms")

    # For large configs, limit frames to keep benchmark time reasonable
    max_frames_for_benchmark = 500
    actual_frames = len(fields)
    if actual_frames > max_frames_for_benchmark:
        print(
            f"  Limiting frames from {actual_frames} to {max_frames_for_benchmark} for benchmark"
        )
        fields = fields[:max_frames_for_benchmark]
        # Skip overlays when truncating (overlay truncation is complex due to skeletons)
        print("  Skipping overlays for truncated benchmark")
        overlays = []
        actual_frames = max_frames_for_benchmark

    # Create temporary output directory
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "benchmark.mp4"

        # Serial export (n_workers=1)
        print(f"  Exporting {actual_frames} frames (serial)...")
        total_ms, time_per_frame_ms, memory_mb, file_size_mb = benchmark_video_export(
            env, fields, overlays, config, output_path, n_workers=1
        )
        result.add(
            TimingResult(
                name=f"export_serial ({actual_frames} frames)",
                elapsed_ms=total_ms,
                memory_mb=memory_mb,
                peak_memory_mb=memory_mb,
            )
        )
        result.add(
            TimingResult(
                name="time_per_frame_serial",
                elapsed_ms=time_per_frame_ms,
                memory_mb=0.0,
                peak_memory_mb=0.0,
            )
        )
        print(
            f"  Serial export: {total_ms:.2f} ms total, {time_per_frame_ms:.2f} ms/frame"
        )
        print(f"  File size: {file_size_mb:.2f} MB")

        # Parallel export (n_workers=4)
        output_path_parallel = Path(tmpdir) / "benchmark_parallel.mp4"
        print(f"  Exporting {actual_frames} frames (parallel, n_workers=4)...")
        (
            total_ms_parallel,
            time_per_frame_ms_parallel,
            memory_mb_parallel,
            _file_size_mb_parallel,
        ) = benchmark_video_export(
            env, fields, overlays, config, output_path_parallel, n_workers=4
        )
        result.add(
            TimingResult(
                name=f"export_parallel_4 ({actual_frames} frames)",
                elapsed_ms=total_ms_parallel,
                memory_mb=memory_mb_parallel,
                peak_memory_mb=memory_mb_parallel,
            )
        )
        result.add(
            TimingResult(
                name="time_per_frame_parallel_4",
                elapsed_ms=time_per_frame_ms_parallel,
                memory_mb=0.0,
                peak_memory_mb=0.0,
            )
        )
        print(
            f"  Parallel export: {total_ms_parallel:.2f} ms total, {time_per_frame_ms_parallel:.2f} ms/frame"
        )

        # Calculate speedup
        speedup = total_ms / total_ms_parallel if total_ms_parallel > 0 else 0
        print(f"  Parallel speedup: {speedup:.2f}x")

    force_gc()

    return result


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark video animation backend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run python benchmarks/bench_video.py --config small
    uv run python benchmarks/bench_video.py --all
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

    if not check_ffmpeg_available():
        print("ERROR: ffmpeg is not available. Install with: brew install ffmpeg")
        sys.exit(1)

    print("=" * 60)
    print("VIDEO BACKEND BENCHMARK")
    print("=" * 60)
    print(f"Machine: {sys.platform}")
    print(f"Python: {sys.version.split()[0]}")

    results: list[BenchmarkResult] = []

    configs_to_run = get_benchmark_configs() if args.all else [args.config]

    for config_name in configs_to_run:
        try:
            result = run_video_benchmark(config_name)
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
