#!/usr/bin/env python
"""Profile _build_skeleton_vectors performance.

This script measures the performance of the skeleton vector generation
function on medium and large datasets to establish baselines before
optimization.

Usage
-----
    uv run python benchmarks/bench_skeleton_vectors.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Add scripts directory to path for benchmark_datasets import
# This must happen before importing from those modules
scripts_dir = Path(__file__).parent.parent / "scripts"
benchmarks_dir = Path(__file__).parent
sys.path.insert(0, str(scripts_dir))
sys.path.insert(0, str(benchmarks_dir))

from benchmark_datasets import (  # noqa: E402
    LARGE_CONFIG,
    MEDIUM_CONFIG,
    create_benchmark_env,
    create_benchmark_overlays,
)
from utils import timer  # noqa: E402


def profile_skeleton_vectors(
    config_name: str, n_runs: int = 3
) -> dict[str, Any] | None:
    """Profile _build_skeleton_vectors on a benchmark configuration.

    Parameters
    ----------
    config_name : str
        One of "medium" or "large".
    n_runs : int
        Number of runs to average.

    Returns
    -------
    dict or None
        Dictionary with profiling results, or None if profiling failed.
    """
    # Import here to avoid circular imports
    import numpy as np

    from neurospatial.animation.backends.napari_backend import _build_skeleton_vectors
    from neurospatial.animation.overlays import (
        BodypartOverlay,
        _convert_overlays_to_data,
    )

    # Select config
    configs = {"medium": MEDIUM_CONFIG, "large": LARGE_CONFIG}
    config = configs[config_name]

    print(f"\n{'=' * 60}")
    print(f"Profiling _build_skeleton_vectors on {config_name.upper()} config")
    print(f"{'=' * 60}")
    print(f"  n_frames: {config.n_frames:,}")
    print(f"  n_bodyparts: {config.n_bodyparts}")
    print(f"  n_skeleton_edges: {config.n_bodyparts - 1}")
    print(f"  expected iterations: {config.n_frames * (config.n_bodyparts - 1):,}")
    print()

    # Create test data
    print("Creating test data...")
    env = create_benchmark_env(config, seed=42)
    overlays = create_benchmark_overlays(env, config, seed=42)

    # Find the bodypart overlay
    bodypart_overlay = None
    for overlay in overlays:
        if isinstance(overlay, BodypartOverlay):
            bodypart_overlay = overlay
            break

    if bodypart_overlay is None:
        print("ERROR: No BodypartOverlay found in test data")
        return None

    # Convert to internal format
    n_frames = config.n_frames
    frame_times = np.arange(n_frames, dtype=np.float64)
    overlay_data_list = _convert_overlays_to_data(
        overlays=overlays,
        frame_times=frame_times,
        n_frames=n_frames,
        env=env,
    )

    # Get bodypart data from the OverlayData container
    if not overlay_data_list.bodypart_sets:
        print("ERROR: No BodypartData found in overlay_data")
        return None

    bodypart_data = overlay_data_list.bodypart_sets[0]

    print("Bodypart data prepared:")
    print(f"  n_bodyparts: {len(bodypart_data.bodyparts)}")
    skeleton_edge_count = (
        len(bodypart_data.skeleton.edges) if bodypart_data.skeleton else 0
    )
    print(f"  skeleton edges: {skeleton_edge_count}")
    print()

    # Run profiling
    print(f"Running {n_runs} profiling iterations...")
    timings_ms = []

    for run_idx in range(n_runs):
        with timer(f"run_{run_idx + 1}") as t:
            vectors, _features = _build_skeleton_vectors(bodypart_data, env)
        timings_ms.append(t.elapsed_ms)
        print(f"  Run {run_idx + 1}: {t.elapsed_ms:.2f} ms")

    avg_ms = sum(timings_ms) / len(timings_ms)
    min_ms = min(timings_ms)
    max_ms = max(timings_ms)

    print()
    print(f"Results for {config_name.upper()}:")
    print(f"  Average: {avg_ms:.2f} ms")
    print(f"  Min: {min_ms:.2f} ms")
    print(f"  Max: {max_ms:.2f} ms")
    print(f"  Output shape: {vectors.shape}")
    print(f"  n_segments: {vectors.shape[0]:,}")

    # Estimate per-frame and per-iteration cost
    n_iterations = config.n_frames * (config.n_bodyparts - 1)
    per_frame_us = (avg_ms * 1000) / config.n_frames
    per_iteration_us = (avg_ms * 1000) / n_iterations

    print()
    print(f"  Per-frame: {per_frame_us:.3f} µs")
    print(f"  Per-iteration: {per_iteration_us:.3f} µs")

    return {
        "config": config_name,
        "avg_ms": avg_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "n_frames": config.n_frames,
        "n_edges": config.n_bodyparts - 1,
        "n_segments": vectors.shape[0],
    }


def main() -> None:
    """Run profiling on medium and large configs."""
    print("=" * 60)
    print("_build_skeleton_vectors Profiling")
    print("=" * 60)

    results = []

    # Profile medium config
    result = profile_skeleton_vectors("medium", n_runs=5)
    if result:
        results.append(result)

    # Profile large config
    result = profile_skeleton_vectors("large", n_runs=3)
    if result:
        results.append(result)

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print()
    print("| Config | n_frames | n_edges | Avg Time (ms) | n_segments |")
    print("|--------|----------|---------|---------------|------------|")
    for r in results:
        print(
            f"| {r['config']:6} | {r['n_frames']:8,} | {r['n_edges']:7} | "
            f"{r['avg_ms']:13.2f} | {r['n_segments']:10,} |"
        )
    print()


if __name__ == "__main__":
    main()
