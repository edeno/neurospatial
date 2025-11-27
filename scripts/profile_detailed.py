#!/usr/bin/env python
"""Detailed profiling of napari animation setup phases.

Breaks down timing for each component of the animation setup.
"""

import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "data"))


def profile_compute_global_colormap_range():
    """Profile the compute_global_colormap_range function."""
    from neurospatial.animation.rendering import compute_global_colormap_range

    # Create test data similar to bandit task
    n_frames = 41000
    n_bins = 557
    print(
        f"\nProfiling compute_global_colormap_range with {n_frames} frames, {n_bins} bins"
    )

    # Test 1: List of arrays (current implementation)
    fields_list = [np.random.rand(n_bins) for _ in range(n_frames)]

    start = time.perf_counter()
    _vmin, _vmax = compute_global_colormap_range(fields_list)
    list_time = time.perf_counter() - start
    print(f"  List of arrays: {list_time * 1000:.2f}ms")

    # Test 2: Stacked array (potential optimization)
    fields_stacked = np.random.rand(n_frames, n_bins)

    start = time.perf_counter()
    float(np.nanmin(fields_stacked))
    float(np.nanmax(fields_stacked))
    stacked_time = time.perf_counter() - start
    print(f"  Stacked array: {stacked_time * 1000:.2f}ms")
    print(f"  Speedup: {list_time / stacked_time:.1f}x")


def profile_transform_coords():
    """Profile coordinate transformation."""
    from neurospatial import Environment
    from neurospatial.animation.transforms import EnvScale, transform_coords_for_napari

    positions = np.random.randn(100, 2) * 50
    env = Environment.from_samples(positions, bin_size=5.0)

    n_frames = 41000
    coords = np.random.randn(n_frames, 2) * 20
    print(f"\nProfiling transform_coords_for_napari with {n_frames} points")

    # Without pre-computed scale
    start = time.perf_counter()
    transform_coords_for_napari(coords, env)
    no_scale_time = time.perf_counter() - start
    print(f"  Without EnvScale: {no_scale_time * 1000:.2f}ms")

    # With pre-computed scale
    env_scale = EnvScale.from_env(env)
    start = time.perf_counter()
    transform_coords_for_napari(coords, env_scale)
    with_scale_time = time.perf_counter() - start
    print(f"  With EnvScale: {with_scale_time * 1000:.2f}ms")
    print(f"  Speedup: {no_scale_time / with_scale_time:.1f}x")


def profile_lazy_renderer_init():
    """Profile lazy renderer initialization."""
    import matplotlib.pyplot as plt

    from neurospatial import Environment
    from neurospatial.animation.backends.napari_backend import (
        _create_lazy_field_renderer,
    )

    positions = np.random.randn(100, 2) * 50
    env = Environment.from_samples(positions, bin_size=5.0)

    n_frames = 1000  # Smaller for this test
    fields = [np.random.rand(env.n_bins) for _ in range(n_frames)]

    cmap_obj = plt.get_cmap("viridis")
    cmap_lookup = (cmap_obj(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)

    print(f"\nProfiling LazyFieldRenderer init with {n_frames} frames")

    start = time.perf_counter()
    renderer = _create_lazy_field_renderer(
        env,
        fields,
        cmap_lookup,
        0.0,
        1.0,
        cache_size=1000,
        chunk_size=10,
        max_chunks=100,
    )
    init_time = time.perf_counter() - start
    print(f"  Renderer init: {init_time * 1000:.2f}ms")

    # Profile first frame render
    start = time.perf_counter()
    renderer[0]
    first_frame_time = time.perf_counter() - start
    print(f"  First frame render: {first_frame_time * 1000:.2f}ms")

    # Profile cached frame access
    start = time.perf_counter()
    renderer[0]
    cached_frame_time = time.perf_counter() - start
    print(f"  Cached frame access: {cached_frame_time * 1000:.4f}ms")


def profile_napari_viewer_creation():
    """Profile napari viewer creation (no data)."""
    try:
        import napari

        print("\nProfiling napari.Viewer creation")

        start = time.perf_counter()
        viewer = napari.Viewer(title="Profile Test", show=False)
        viewer_time = time.perf_counter() - start
        print(f"  Viewer creation: {viewer_time * 1000:.2f}ms")

        viewer.close()
    except ImportError:
        print("\nNapari not available, skipping viewer profiling")


def profile_full_animation_breakdown():
    """Profile the full animation setup with detailed breakdown."""
    import matplotlib.pyplot as plt
    from load_bandit_data import load_neural_recording_from_files

    from neurospatial import Environment, HeadDirectionOverlay, PositionOverlay
    from neurospatial.animation.rendering import compute_global_colormap_range

    data_path = PROJECT_ROOT / "data"
    print("\n" + "=" * 60)
    print("FULL ANIMATION SETUP BREAKDOWN")
    print("=" * 60)

    # Load data
    start = time.perf_counter()
    data = load_neural_recording_from_files(data_path, "j1620210710_02_r1")
    load_time = time.perf_counter() - start
    print(f"\n1. Load data: {load_time * 1000:.0f}ms")

    position_info = data["position_info"]
    positions_2d = position_info[["head_position_x", "head_position_y"]].values
    times_array = position_info.index.values

    # Subsample
    subsample_rate = 17
    positions_sub = positions_2d[::subsample_rate]
    times_array[::subsample_rate]
    n_frames = len(positions_sub)

    head_angles = None
    if "head_orientation" in position_info.columns:
        head_angles = position_info["head_orientation"].values[::subsample_rate]

    print(f"   Frames: {n_frames:,}")

    # Create environment
    start = time.perf_counter()
    env = Environment.from_samples(positions_2d, bin_size=4.0, name="Maze")
    env.units = "cm"
    env_time = time.perf_counter() - start
    print(f"\n2. Create environment: {env_time * 1000:.0f}ms")
    print(f"   Bins: {env.n_bins}")

    # Create fields (simple tile for profiling)
    start = time.perf_counter()
    field = np.random.rand(env.n_bins)
    fields = np.tile(field, (n_frames, 1))
    fields_time = time.perf_counter() - start
    print(f"\n3. Create fields array: {fields_time * 1000:.0f}ms")
    print(f"   Shape: {fields.shape}")
    print(f"   Memory: {fields.nbytes / 1024 / 1024:.1f} MB")

    # Convert to list (napari expects list)
    start = time.perf_counter()
    fields_list = [fields[i] for i in range(n_frames)]
    list_time = time.perf_counter() - start
    print(f"\n4. Convert to list: {list_time * 1000:.0f}ms")

    # Compute colormap range
    start = time.perf_counter()
    _vmin, _vmax = compute_global_colormap_range(fields_list)
    cmap_time = time.perf_counter() - start
    print(f"\n5. Compute colormap range: {cmap_time * 1000:.0f}ms")

    # Create colormap lookup
    start = time.perf_counter()
    cmap_obj = plt.get_cmap("hot")
    (cmap_obj(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    cmap_lookup_time = time.perf_counter() - start
    print(f"\n6. Create colormap lookup: {cmap_lookup_time * 1000:.0f}ms")

    # Create overlays
    start = time.perf_counter()
    PositionOverlay(data=positions_sub, color="cyan", size=15.0, trail_length=15)
    pos_overlay_time = time.perf_counter() - start
    print(f"\n7. Create PositionOverlay: {pos_overlay_time * 1000:.2f}ms")

    if head_angles is not None:
        start = time.perf_counter()
        HeadDirectionOverlay(data=head_angles, color="yellow", length=10.0, width=2.0)
        hd_overlay_time = time.perf_counter() - start
        print(f"\n8. Create HeadDirectionOverlay: {hd_overlay_time * 1000:.2f}ms")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total = (
        load_time
        + env_time
        + fields_time
        + list_time
        + cmap_time
        + cmap_lookup_time
        + pos_overlay_time
    )
    if head_angles is not None:
        total += hd_overlay_time
    print(f"Total pre-napari setup: {total * 1000:.0f}ms")
    print("\nNote: Napari viewer creation adds ~2-3 seconds")


def main():
    print("=" * 60)
    print("DETAILED PERFORMANCE PROFILING")
    print("=" * 60)

    profile_compute_global_colormap_range()
    profile_transform_coords()
    profile_lazy_renderer_init()
    profile_napari_viewer_creation()
    profile_full_animation_breakdown()


if __name__ == "__main__":
    main()
