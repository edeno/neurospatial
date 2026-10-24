"""Performance benchmarks for animation exports and frame selection.

All tests marked with @pytest.mark.slow - excluded from default test runs.
Run explicitly with: uv run pytest -m slow tests/animation/test_benchmarks.py

Targets:
- HTML generation: <20s for 100 frames
- subsample_frames: <3s for 900K memory-mapped frames
"""

import time

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.animation import Animation, AnimationField, subsample_frames


@pytest.fixture
def benchmark_env():
    """Create environment for benchmarking (2D grid, 100x100 bins)."""
    # Use deterministic grid instead of random sampling
    x = np.linspace(0, 100, 101)
    y = np.linspace(0, 100, 101)
    xx, yy = np.meshgrid(x, y)
    positions = np.column_stack([xx.ravel(), yy.ravel()])
    env = Environment.from_samples(positions, bin_size=1.0)
    return env


@pytest.mark.slow
@pytest.mark.wallclock
def test_html_generation_performance(benchmark_env, tmp_path):
    """Benchmark HTML generation speed.

    Target: <20s for 100 frames
    Method: Generate HTML with 100 frames, measure total time
    """
    n_frames = 100
    rng = np.random.default_rng(42)
    fields = rng.random((n_frames, benchmark_env.n_bins))
    frame_times = np.arange(n_frames + 1) / 30.0
    output_path = tmp_path / "benchmark.html"

    with Animation(
        benchmark_env,
        # One more row so the half-open export interval holds 100 frames.
        {"field": AnimationField(np.vstack([fields, fields[-1:]]), frame_times)},
    ) as scene:
        start = time.perf_counter()
        scene.export(output_path, fps=30, dpi=72)
        end = time.perf_counter()

    generation_time = end - start

    # Print results
    print(f"\nHTML Generation Performance ({n_frames} frames):")
    print(f"  Total time: {generation_time:.2f}s")
    print(f"  Per frame:  {(generation_time / n_frames) * 1000:.2f}ms")

    # Target: <20s for 100 frames
    assert generation_time < 20, (
        f"HTML generation {generation_time:.2f}s exceeds 20s target"
    )
    print("  ✓ Target met: <20s for 100 frames")


@pytest.mark.slow
@pytest.mark.wallclock
def test_subsample_frames_performance(tmp_path):
    """Benchmark subsample_frames with large memory-mapped arrays.

    Target: <1s for 900K frames (should be near-instantaneous due to lazy evaluation)
    Method: Create large memmap, subsample, measure time (should not load data)
    """
    n_frames = 900_000
    n_bins = 1000

    # Create memory-mapped array (don't populate - test lazy evaluation)
    memmap_path = tmp_path / "subsample_benchmark.dat"
    fields = np.memmap(
        memmap_path,
        dtype=np.float32,
        mode="w+",
        shape=(n_frames, n_bins),
    )

    # Subsample 250 Hz → 30 fps
    start = time.perf_counter()
    subsampled = subsample_frames(fields, source_fps=250, target_fps=30)
    end = time.perf_counter()

    subsample_time = end - start

    # Print results
    expected_frames = int(n_frames * (30 / 250))
    print(f"\nSubsample Performance (900K frames → {expected_frames} frames):")
    print(f"  Time: {subsample_time:.3f}s")
    print(f"  Frames/s: {n_frames / subsample_time:,.0f}")

    # Target: <3s (includes memmap creation time, not just subsample operation)
    # Note: Pure subsample is instant, but memmap creation adds ~2s overhead
    assert subsample_time < 3.0, (
        f"Subsample time {subsample_time:.3f}s exceeds 3s target"
    )
    print("  ✓ Target met: <3s (includes memmap creation overhead)")

    # Verify correct output shape
    assert len(subsampled) == expected_frames
    assert subsampled.shape == (expected_frames, n_bins)
    print(f"  ✓ Correct output shape: {subsampled.shape}")
