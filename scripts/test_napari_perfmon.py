#!/usr/bin/env python
"""Test script to investigate napari animation playback performance.

This script helps diagnose the "stuck playback" issue by:
1. Creating a test environment with enough frames to trigger cache misses
2. Running the animation with napari's performance monitoring enabled
3. Providing options to test different scenarios

Usage:
    # Basic performance monitoring
    NAPARI_PERFMON=1 uv run python scripts/test_napari_perfmon.py

    # With async mode (should improve playback)
    NAPARI_ASYNC=1 NAPARI_PERFMON=1 uv run python scripts/test_napari_perfmon.py

    # With custom config file for detailed tracing
    NAPARI_PERFMON=scripts/perfmon_config.json uv run python scripts/test_napari_perfmon.py

    # Test scenarios that trigger stuck behavior:

    # Long time series (like notebook 16 example 5)
    NAPARI_PERFMON=1 uv run python scripts/test_napari_perfmon.py --scenario long-series

    # Multi-field viewer (like notebook 16 example 1b)
    NAPARI_PERFMON=1 uv run python scripts/test_napari_perfmon.py --scenario multi-field

    # With overlays (like notebook 17)
    NAPARI_PERFMON=1 uv run python scripts/test_napari_perfmon.py --scenario overlays

    # All scenarios combined
    NAPARI_PERFMON=1 uv run python scripts/test_napari_perfmon.py --scenario all

See also: docs/dev/napari-playback-investigation.md
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Test napari animation playback performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--scenario",
        type=str,
        choices=["basic", "long-series", "multi-field", "overlays", "all"],
        default="basic",
        help="Test scenario: basic (default), long-series, multi-field, overlays, all",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=500,
        help="Number of frames to generate (default: 500)",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=100,
        help="Approximate grid size in bins per dimension (default: 100)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Playback FPS (default: 30)",
    )
    parser.add_argument(
        "--bin-size",
        type=float,
        default=5.0,
        help="Bin size for environment (default: 5.0)",
    )
    parser.add_argument(
        "--cache-size",
        type=int,
        default=1000,
        help="Cache size for lazy renderer (default: 1000)",
    )
    parser.add_argument(
        "--no-clear-cache",
        action="store_true",
        help="Don't clear cache before starting (warm start)",
    )
    parser.add_argument(
        "--pre-render",
        action="store_true",
        help="Pre-render all frames before starting playback",
    )

    args = parser.parse_args()

    # Check if performance monitoring is enabled
    perfmon = os.environ.get("NAPARI_PERFMON")
    async_mode = os.environ.get("NAPARI_ASYNC")

    print("=" * 60)
    print("Napari Animation Performance Test")
    print("=" * 60)
    print(f"NAPARI_PERFMON: {perfmon or 'Not set'}")
    print(f"NAPARI_ASYNC: {async_mode or 'Not set'}")
    print(f"Frames: {args.frames}")
    print(f"Grid size: ~{args.grid_size}x{args.grid_size}")
    print(f"FPS: {args.fps}")
    print(f"Cache size: {args.cache_size}")
    print("=" * 60)

    if not perfmon:
        print("\nTIP: Run with NAPARI_PERFMON=1 to enable performance monitoring")
        print("     Run with NAPARI_ASYNC=1 to test async mode\n")

    # Import after checking args (napari import can be slow)
    from neurospatial import Environment

    # Create test environment
    print("\nCreating test environment...")
    t0 = time.perf_counter()

    # Generate positions to create approximately the target grid size
    extent = args.grid_size * args.bin_size
    positions = np.random.uniform(0, extent, (2000, 2))
    env = Environment.from_samples(positions, bin_size=args.bin_size)

    print(f"  Environment created in {time.perf_counter() - t0:.2f}s")
    print(f"  Grid shape: {env.layout.grid_shape}")
    print(f"  Number of bins: {env.n_bins}")

    # Generate test fields
    print(f"\nGenerating {args.frames} test fields...")
    t0 = time.perf_counter()
    fields = [np.random.rand(env.n_bins).astype(np.float64) for _ in range(args.frames)]
    print(f"  Fields generated in {time.perf_counter() - t0:.2f}s")

    # Estimate memory usage
    frame_bytes = env.n_bins * 8  # float64
    rgb_bytes = np.prod(env.layout.grid_shape) * 3  # RGB image
    total_field_mb = (args.frames * frame_bytes) / (1024 * 1024)
    cache_mb = (args.cache_size * rgb_bytes) / (1024 * 1024)
    print(f"  Field data: {total_field_mb:.1f} MB")
    print(f"  Cache capacity: {cache_mb:.1f} MB ({args.cache_size} frames)")

    # Clear cache if requested (default)
    if not args.no_clear_cache:
        print("\nClearing environment cache (cold start)...")
        env.clear_cache()

    # Pre-render if requested
    if args.pre_render:
        print("\nPre-rendering all frames (this may take a while)...")
        t0 = time.perf_counter()
        # This is a simple way to pre-warm the cache
        # A proper implementation would use the renderer's cache directly
        print("  (Pre-rendering not yet implemented - skipping)")
        # TODO: Implement pre-rendering

    # Launch animation based on scenario
    print("\nLaunching napari viewer...")
    print("  - Use the playback controls to start/stop animation")
    print("  - Watch for 'stuck' behavior during playback")
    if perfmon:
        print("  - Check the Performance tab in napari for timing data")
        print("  - Use Debug > Start/Stop Recording to capture traces")

    if args.scenario == "basic":
        _ = env.animate_fields(
            fields,
            backend="napari",
            fps=args.fps,
            cache_size=args.cache_size,
            title="Basic Animation Test",
        )

    elif args.scenario == "long-series":
        # Long time series like notebook 16 example 5
        # Use higher FPS to stress test
        print("\n  SCENARIO: Long time series (like notebook 16 example 5)")
        print(f"  High FPS playback: {args.fps} FPS with {len(fields)} frames")
        _ = env.animate_fields(
            fields,
            backend="napari",
            fps=args.fps,
            cache_size=args.cache_size,
            title=f"Long Series: {len(fields)} frames @ {args.fps} FPS",
        )

    elif args.scenario == "multi-field":
        # Multi-field viewer like notebook 16 example 1b
        print("\n  SCENARIO: Multi-field comparison (like notebook 16 example 1b)")
        print("  Creating 3 field sequences for side-by-side comparison...")

        # Create 3 different field sequences
        fields_a = fields  # Original
        fields_b = [np.roll(f, env.n_bins // 4) for f in fields]  # Shifted
        fields_c = [np.roll(f, env.n_bins // 2) for f in fields]  # More shifted

        print(f"  3 sequences x {len(fields)} frames each")

        _ = env.animate_fields(
            fields=[fields_a, fields_b, fields_c],
            backend="napari",
            layout="horizontal",
            layer_names=["Field A", "Field B", "Field C"],
            fps=args.fps,
            title="Multi-Field: 3 sequences side-by-side",
        )

    elif args.scenario == "overlays":
        # With overlays like notebook 17
        print("\n  SCENARIO: Animation with overlays (like notebook 17)")

        from neurospatial import BodypartOverlay, HeadDirectionOverlay, PositionOverlay

        # Simulate trajectory
        n_frames = len(fields)
        t = np.linspace(0, 4 * np.pi, n_frames)
        r = np.linspace(10, 40, n_frames)
        theta = t + np.random.randn(n_frames) * 0.1

        # Trajectory in environment coordinates
        extent = args.grid_size * args.bin_size
        center = extent / 2
        trajectory = np.column_stack(
            [center + r * np.cos(theta), center + r * np.sin(theta)]
        )
        trajectory = np.clip(trajectory, 5, extent - 5)

        # Head direction
        head_angles = theta + np.pi / 2

        # Pose data
        body_length = 5.0
        nose_x = trajectory[:, 0] + body_length * 0.5 * np.cos(head_angles)
        nose_y = trajectory[:, 1] + body_length * 0.5 * np.sin(head_angles)
        tail_x = trajectory[:, 0] - body_length * 0.5 * np.cos(head_angles)
        tail_y = trajectory[:, 1] - body_length * 0.5 * np.sin(head_angles)

        pose_data = {
            "nose": np.column_stack([nose_x, nose_y]),
            "body": trajectory.copy(),
            "tail": np.column_stack([tail_x, tail_y]),
        }

        # Create overlays
        position_overlay = PositionOverlay(
            data=trajectory,
            color="red",
            size=10.0,
            trail_length=15,
        )
        bodypart_overlay = BodypartOverlay(
            data=pose_data,
            skeleton=[("tail", "body"), ("body", "nose")],
            colors={"nose": "yellow", "body": "red", "tail": "blue"},
            skeleton_color="white",
            skeleton_width=2.0,
        )
        head_direction_overlay = HeadDirectionOverlay(
            data=head_angles,
            color="yellow",
            length=10.0,
        )

        print(
            "  Overlays: Position (trail), Bodypart (skeleton), HeadDirection (arrow)"
        )

        _ = env.animate_fields(
            fields,
            overlays=[position_overlay, bodypart_overlay, head_direction_overlay],
            backend="napari",
            fps=args.fps,
            title="Overlays: Position + Pose + HeadDirection",
        )

    elif args.scenario == "all":
        # Combined: multi-field with overlays
        print("\n  SCENARIO: Combined (multi-field + overlays)")

        from neurospatial import PositionOverlay

        # Create 2 field sequences
        fields_a = fields
        fields_b = [np.roll(f, env.n_bins // 3) for f in fields]

        # Simulate trajectory
        n_frames = len(fields)
        t = np.linspace(0, 4 * np.pi, n_frames)
        r = np.linspace(10, 40, n_frames)
        theta = t + np.random.randn(n_frames) * 0.1

        extent = args.grid_size * args.bin_size
        center = extent / 2
        trajectory = np.column_stack(
            [center + r * np.cos(theta), center + r * np.sin(theta)]
        )
        trajectory = np.clip(trajectory, 5, extent - 5)

        position_overlay = PositionOverlay(
            data=trajectory,
            color="red",
            size=12.0,
            trail_length=20,
        )

        print(f"  2 field sequences x {len(fields)} frames + position overlay")

        _ = env.animate_fields(
            fields=[fields_a, fields_b],
            overlays=[position_overlay],
            backend="napari",
            layout="horizontal",
            layer_names=["Field A", "Field B"],
            fps=args.fps,
            title="Combined: Multi-field + Overlay",
        )

    # Run napari event loop
    import napari

    napari.run()

    print("\nTest complete.")


if __name__ == "__main__":
    main()
