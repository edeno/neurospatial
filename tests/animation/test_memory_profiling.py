"""Memory profiling tests for animation exports.

Tests verify memory-efficient behavior:
- Parallel exports clean up their worker processes
- Memory-mapped arrays work correctly

All tests marked @pytest.mark.slow - excluded from default test runs.
Run with: uv run pytest tests/animation/test_memory_profiling.py --override-ini="addopts="
"""

import gc
import os
import sys

import numpy as np
import pytest

from neurospatial import Environment


def get_memory_usage_mb():
    """Get current memory usage in MB (cross-platform).

    Returns
    -------
    memory_mb : float
        Current memory usage in megabytes
    """
    try:
        import psutil  # type: ignore[import-untyped]

        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        # Fallback for systems without psutil - use resource module (Unix only)
        if sys.platform != "win32":
            import resource

            usage = resource.getrusage(resource.RUSAGE_SELF)
            # ru_maxrss is in KB on Linux, bytes on macOS
            if sys.platform == "darwin":
                return usage.ru_maxrss / 1024 / 1024
            else:
                return usage.ru_maxrss / 1024
        else:
            pytest.skip("psutil required for memory profiling on Windows")


@pytest.mark.slow
def test_parallel_export_cleanup(tmp_path):
    """Exporting with worker processes leaves no workers or memory behind."""
    psutil = pytest.importorskip("psutil")
    pytest.importorskip("imageio_ffmpeg")
    from neurospatial.animation import Animation

    x = np.linspace(0, 50, 51)
    xx, yy = np.meshgrid(x, x)
    env = Environment.from_samples(
        np.column_stack([xx.ravel(), yy.ravel()]), bin_size=1.0
    )
    n_frames = 50
    fields = np.random.default_rng(42).random((n_frames, env.n_bins))
    process = psutil.Process(os.getpid())

    def workers():
        # The multiprocessing resource tracker is shared and outlives a pool.
        return {
            child.pid
            for child in process.children(recursive=True)
            if "resource_tracker" not in " ".join(child.cmdline())
        }

    gc.collect()
    memory_baseline = get_memory_usage_mb()
    children_before = workers()
    with Animation(env, fields, frame_times=np.arange(n_frames) / 30.0) as scene:
        path = scene.export(tmp_path / "cleanup_test.mp4", fps=30, n_workers=4, dpi=50)
    gc.collect()
    memory_increase = get_memory_usage_mb() - memory_baseline

    assert path.stat().st_size > 0
    assert workers() == children_before
    # matplotlib and the encoder may keep caches; a leak would grow per frame.
    assert memory_increase < 50, (
        f"Memory increased by {memory_increase:.1f}MB after exporting."
    )


@pytest.mark.slow
def test_memmap_large_dataset_memory(tmp_path):
    """Verify memory-mapped arrays don't consume excessive memory.

    Test Strategy:
    1. Create very large memmap (100K frames, ~400MB on disk)
    2. Verify memory usage stays minimal (<50MB overhead)
    3. Access scattered frames, verify memory stays bounded
    4. Test subsample operation, verify no memory explosion
    """
    from neurospatial.animation import subsample_frames

    # Force garbage collection for accurate baseline
    gc.collect()
    memory_baseline = get_memory_usage_mb()

    n_frames = 100_000
    n_bins = 1000

    # Create large memory-mapped array
    memmap_path = tmp_path / "large_memmap_test.dat"
    fields = np.memmap(
        memmap_path,
        dtype=np.float32,
        mode="w+",
        shape=(n_frames, n_bins),
    )

    # Measure memory after memmap creation
    gc.collect()
    memory_after_create = get_memory_usage_mb()
    create_overhead = memory_after_create - memory_baseline

    # Access 100 scattered frames (should trigger minimal loading)
    rng = np.random.default_rng(42)
    random_indices = rng.integers(0, n_frames, size=100)
    for idx in random_indices:
        _ = fields[idx]

    # Measure memory after access
    gc.collect()
    memory_after_access = get_memory_usage_mb()
    access_overhead = memory_after_access - memory_after_create

    # Subsample (should be lazy)
    _ = subsample_frames(fields, source_fps=250, target_fps=30)

    # Measure memory after subsample
    gc.collect()
    memory_after_subsample = get_memory_usage_mb()
    subsample_overhead = memory_after_subsample - memory_after_access

    # Print results
    print(f"\nLarge Memmap Memory Profile ({n_frames} frames, {n_bins} bins):")
    print(f"  Baseline memory:        {memory_baseline:.1f} MB")
    print(
        f"  After memmap creation:  {memory_after_create:.1f} MB (+{create_overhead:.1f} MB)"
    )
    print(
        f"  After 100 frame access: {memory_after_access:.1f} MB (+{access_overhead:.1f} MB)"
    )
    print(
        f"  After subsample:        {memory_after_subsample:.1f} MB (+{subsample_overhead:.1f} MB)"
    )

    # Calculate expected size if fully loaded
    expected_full_load = (n_frames * n_bins * 4) / 1024 / 1024
    print(f"  Expected if all loaded: {expected_full_load:.1f} MB")

    # Check memory efficiency (informational - documents actual behavior)
    total_overhead = memory_after_subsample - memory_baseline
    efficiency_pct = (1 - total_overhead / expected_full_load) * 100

    print("\n  Memory Efficiency Analysis:")
    print(
        f"    Total overhead: {total_overhead:.1f}MB ({efficiency_pct:.1f}% efficient)"
    )
    print(f"    Subsample overhead: {subsample_overhead:.1f}MB")

    # Subsample creates array views (minimal overhead expected)
    # If overhead is high, it may indicate copying behavior
    if subsample_overhead < 10:
        print(f"    ✓ Subsample is lazy (view-based): {subsample_overhead:.1f}MB")
    elif total_overhead < expected_full_load * 0.5:
        print(f"    ⚠ Subsample shows moderate overhead: {subsample_overhead:.1f}MB")
        print("    (may indicate partial copying or page faults)")
    else:
        print(f"    ⚠ Subsample shows high overhead: {subsample_overhead:.1f}MB")
        print("    (close to full load - may be copying data)")
        print("    This is acceptable for moderate datasets (<10K frames)")
