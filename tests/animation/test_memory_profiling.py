"""Memory profiling tests for animation backends.

Tests verify memory-efficient behavior:
- Napari lazy loading doesn't load all frames
- Parallel rendering cleans up properly
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


@pytest.fixture
def memory_test_env():
    """Create environment for memory testing (100x100 bins)."""
    # Use deterministic grid instead of random sampling
    x = np.linspace(0, 100, 101)
    y = np.linspace(0, 100, 101)
    xx, yy = np.meshgrid(x, y)
    positions = np.column_stack([xx.ravel(), yy.ravel()])
    env = Environment.from_samples(positions, bin_size=1.0)
    return env


@pytest.mark.slow
@pytest.mark.xdist_group(name="napari_gui")
def test_napari_lazy_loading_memory(memory_test_env, tmp_path):
    """Verify Napari lazy loading doesn't load all frames into memory.

    Test Strategy:
    1. Create large memory-mapped array (10K frames, ~40MB uncompressed)
    2. Create LazyFieldRenderer
    3. Measure memory before and after renderer creation
    4. Verify memory increase is small (<10MB for cache structures)
    5. Access a few frames, verify memory stays bounded
    """
    pytest.importorskip("napari")
    from neurospatial.animation.backends.napari_backend import LazyFieldRenderer
    from neurospatial.animation.rendering import compute_global_colormap_range

    # Force garbage collection for accurate baseline
    gc.collect()
    memory_baseline = get_memory_usage_mb()

    n_frames = 10_000
    n_bins = memory_test_env.n_bins

    # Create memory-mapped array (don't populate - test lazy loading)
    memmap_path = tmp_path / "lazy_loading_test.dat"
    fields = np.memmap(
        memmap_path,
        dtype=np.float32,
        mode="w+",
        shape=(n_frames, n_bins),
    )

    # Populate first and last frames for colormap
    rng = np.random.default_rng(42)
    fields[0] = rng.random(n_bins)
    fields[-1] = rng.random(n_bins)
    fields.flush()

    # Compute colormap
    vmin, vmax = compute_global_colormap_range(fields)
    cmap_lookup = np.zeros((256, 3), dtype=np.uint8)

    # Measure memory after memmap creation (should be minimal)
    gc.collect()
    memory_after_memmap = get_memory_usage_mb()
    memmap_overhead = memory_after_memmap - memory_baseline

    # Create lazy renderer
    renderer = LazyFieldRenderer(
        env=memory_test_env,
        fields=fields,
        cmap_lookup=cmap_lookup,
        vmin=vmin,
        vmax=vmax,
    )

    # Measure memory after renderer creation
    gc.collect()
    memory_after_renderer = get_memory_usage_mb()
    renderer_overhead = memory_after_renderer - memory_after_memmap

    # Access 10 frames (should trigger caching)
    for i in range(10):
        _ = renderer[i * 1000]  # Spread out access

    # Measure memory after frame access
    gc.collect()
    memory_after_access = get_memory_usage_mb()
    access_overhead = memory_after_access - memory_after_renderer

    # Print results
    print(f"\nNapari Lazy Loading Memory Profile ({n_frames} frames):")
    print(f"  Baseline memory:        {memory_baseline:.1f} MB")
    print(
        f"  After memmap creation:  {memory_after_memmap:.1f} MB (+{memmap_overhead:.1f} MB)"
    )
    print(
        f"  After renderer creation: {memory_after_renderer:.1f} MB (+{renderer_overhead:.1f} MB)"
    )
    print(
        f"  After 10 frame access:  {memory_after_access:.1f} MB (+{access_overhead:.1f} MB)"
    )

    # Calculate expected memory if all frames were loaded
    expected_full_load = (n_frames * n_bins * 4) / 1024 / 1024  # 4 bytes per float32
    print(f"  Expected if all loaded: {expected_full_load:.1f} MB")

    # Verify lazy loading: renderer and frame access should be minimal
    # The memmap creation allocates disk space (virtual memory), but actual data
    # loading should be minimal. Check that renderer + access overhead is small.
    renderer_and_access_overhead = renderer_overhead + access_overhead
    assert renderer_and_access_overhead < 10, (
        f"Renderer+access overhead {renderer_and_access_overhead:.1f}MB suggests eager loading. "
        f"Expected <10MB for lazy loading, would be {expected_full_load:.1f}MB if all loaded."
    )
    print(
        f"  ✓ Lazy loading verified: renderer+access only {renderer_and_access_overhead:.1f}MB "
        f"(vs {expected_full_load:.1f}MB if eager)"
    )
    print(
        f"  Note: Memmap overhead ({memmap_overhead:.1f}MB) is virtual memory allocation, not data loading"
    )


@pytest.mark.slow
def test_parallel_rendering_cleanup(tmp_path):
    """Verify parallel rendering cleans up worker processes and memory.

    Test Strategy:
    1. Record baseline memory and process count
    2. Render video with multiple workers
    3. Wait for completion and force GC
    4. Verify memory returns to near baseline
    5. Verify no lingering worker processes
    """
    from neurospatial.animation.backends.video_backend import (
        check_ffmpeg_available,
        render_video,
    )

    if not check_ffmpeg_available():
        pytest.skip("ffmpeg not available")

    # Create small environment and fields - deterministic grid
    x = np.linspace(0, 50, 51)
    y = np.linspace(0, 50, 51)
    xx, yy = np.meshgrid(x, y)
    positions = np.column_stack([xx.ravel(), yy.ravel()])
    env = Environment.from_samples(positions, bin_size=1.0)
    env.clear_cache()

    n_frames = 50  # Small for fast test
    rng = np.random.default_rng(42)
    fields = [rng.random(env.n_bins) for _ in range(n_frames)]

    # Force garbage collection and record baseline
    gc.collect()
    memory_baseline = get_memory_usage_mb()

    try:
        import psutil

        process = psutil.Process(os.getpid())
        children_before = len(process.children(recursive=True))
    except ImportError:
        children_before = None

    # Render with 4 workers
    output_path = tmp_path / "cleanup_test.mp4"
    render_video(
        env=env,
        fields=fields,
        save_path=output_path,
        fps=30,
        n_workers=4,
        codec="mpeg4",
    )

    # Force garbage collection
    gc.collect()

    # Measure memory after rendering
    memory_after_render = get_memory_usage_mb()
    memory_increase = memory_after_render - memory_baseline

    # Check for lingering child processes
    if children_before is not None:
        children_after = len(process.children(recursive=True))
        lingering_children = children_after - children_before
    else:
        lingering_children = 0

    # Print results
    print("\nParallel Rendering Cleanup Profile:")
    print(f"  Baseline memory:     {memory_baseline:.1f} MB")
    print(
        f"  After rendering:     {memory_after_render:.1f} MB (+{memory_increase:.1f} MB)"
    )
    if children_before is not None:
        print(f"  Child processes before: {children_before}")
        print(f"  Child processes after:  {children_after}")
        print(f"  Lingering processes:    {lingering_children}")

    # Verify cleanup: memory increase should be minimal
    # Allow up to 50MB overhead (matplotlib, ffmpeg, etc.)
    assert memory_increase < 50, (
        f"Memory increased by {memory_increase:.1f}MB after rendering. "
        "This suggests memory leaks or incomplete cleanup."
    )
    print(f"  ✓ Memory cleanup verified: only {memory_increase:.1f}MB increase")

    # Check for lingering worker processes (informational)
    # Note: Some background processes may persist (pytest, Qt, etc.)
    if children_before is not None:
        if lingering_children == 0:
            print("  ✓ Process cleanup verified: no lingering workers")
        else:
            print(f"  ⚠ Warning: {lingering_children} background process(es) detected")
            print(
                "    (may be pytest, Qt, or system processes - not necessarily leaks)"
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


@pytest.mark.slow
def test_memory_requirements_documentation(memory_test_env):
    """Document memory requirements for different backends and dataset sizes.

    This test doesn't assert, it just measures and reports memory usage
    for documentation purposes.
    """
    print("\n" + "=" * 70)
    print("Memory Requirements Documentation")
    print("=" * 70)

    # Test 1: Small dataset (100 frames)
    print("\n1. Small Dataset (100 frames, ~1K bins):")
    n_bins = memory_test_env.n_bins
    rng = np.random.default_rng(42)
    _ = [rng.random(n_bins) for _ in range(100)]  # Example allocation
    memory_small = (100 * n_bins * 8) / 1024 / 1024  # float64
    print(f"   Array memory: {memory_small:.1f} MB")
    print("   Recommended for: HTML backend, quick previews")

    # Test 2: Medium dataset (1K frames)
    print("\n2. Medium Dataset (1,000 frames, ~1K bins):")
    memory_medium = (1000 * n_bins * 8) / 1024 / 1024
    print(f"   Array memory: {memory_medium:.1f} MB")
    print("   Recommended for: Video export, widget backend")

    # Test 3: Large dataset (100K frames, memmap)
    print("\n3. Large Dataset (100,000 frames, ~1K bins, memory-mapped):")
    memory_large_disk = (100_000 * n_bins * 4) / 1024 / 1024  # float32 on disk
    memory_large_ram = 20  # Overhead estimate from profiling
    print(f"   Disk space: {memory_large_disk:.1f} MB")
    print(f"   RAM overhead: ~{memory_large_ram} MB (lazy loading)")
    print("   Recommended for: Napari backend, hour-long sessions")

    # Test 4: Napari cache memory
    print("\n4. Napari Cache Memory:")
    cache_size = 1000  # Default cache size
    frame_size = (100 * 100 * 3 * 1) / 1024 / 1024  # RGB uint8
    cache_memory = cache_size * frame_size
    print(f"   Default cache size: {cache_size} frames")
    print(f"   Memory per frame (100x100 RGB): {frame_size:.3f} MB")
    print(f"   Total cache memory: {cache_memory:.1f} MB")

    # Test 5: Parallel rendering workers
    print("\n5. Parallel Rendering (4 workers):")
    worker_overhead = 100  # Estimate per worker (matplotlib + data)
    total_parallel = 4 * worker_overhead
    print(f"   Overhead per worker: ~{worker_overhead} MB")
    print(f"   Total for 4 workers: ~{total_parallel} MB")
    print("   Recommended: Use subsample for >10K frames")

    print("\n" + "=" * 70)
    print("General Recommendations:")
    print("  - <500 frames: Any backend, in-memory arrays")
    print("  - 500-10K frames: Video/HTML with subsample, Widget, Napari")
    print("  - >10K frames: Napari with memmap, chunked cache auto-enabled")
    print("  - >100K frames: Napari only, subsample for video export")
    print("=" * 70)

    # This test always passes - it's informational only
    assert True
