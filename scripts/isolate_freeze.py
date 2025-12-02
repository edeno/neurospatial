#!/usr/bin/env python
"""Isolate the source of napari playback freezes.

Runs a series of tests with different configurations to identify
which component causes the ~100ms freezes.

Usage:
    NAPARI_PERFMON=scripts/perfmon_config.json uv run python scripts/isolate_freeze.py --test baseline
    NAPARI_PERFMON=scripts/perfmon_config.json uv run python scripts/isolate_freeze.py --test position
    NAPARI_PERFMON=scripts/perfmon_config.json uv run python scripts/isolate_freeze.py --test events
    NAPARI_PERFMON=scripts/perfmon_config.json uv run python scripts/isolate_freeze.py --test prerender
    NAPARI_PERFMON=scripts/perfmon_config.json uv run python scripts/isolate_freeze.py --test static
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from neurospatial import Environment, PositionOverlay
from neurospatial.animation.overlays import EventOverlay


def create_test_data(n_frames: int = 500, grid_size: int = 50):
    """Create synthetic test data."""
    # Create environment
    np.random.seed(42)
    positions = np.random.rand(1000, 2) * grid_size
    env = Environment.from_samples(positions, bin_size=1.0)

    # Create fields - simple gradient that changes over time
    fields = np.zeros((n_frames, env.n_bins), dtype=np.float64)
    for i in range(n_frames):
        # Moving hotspot
        center = env.bin_centers[i % env.n_bins]
        distances = np.linalg.norm(env.bin_centers - center, axis=1)
        fields[i] = np.exp(-(distances**2) / 100)

    # Frame times at 30 Hz
    frame_times = np.arange(n_frames) / 30.0

    # Trajectory - circular motion
    t = np.linspace(0, 4 * np.pi, n_frames)
    trajectory = np.column_stack(
        [grid_size / 2 + 15 * np.cos(t), grid_size / 2 + 15 * np.sin(t)]
    )

    # Events - random spikes
    n_events = 100
    event_times = np.sort(np.random.uniform(0, frame_times[-1], n_events))

    return {
        "env": env,
        "fields": fields,
        "frame_times": frame_times,
        "trajectory": trajectory,
        "event_times": event_times,
    }


def test_baseline(data: dict, n_steps: int = 100):
    """Test with no overlays - just the field animation."""

    env = data["env"]
    fields = data["fields"]
    frame_times = data["frame_times"]

    print("\n=== TEST: Baseline (no overlays) ===")
    print(f"Fields shape: {fields.shape}")
    print(f"Environment: {env.n_bins} bins")

    viewer = env.animate_fields(
        fields,
        frame_times=frame_times,
        backend="napari",
        title="Baseline Test",
    )

    # Measure frame stepping
    measure_frame_timing(viewer, n_steps)

    viewer.close()


def test_position_overlay(data: dict, n_steps: int = 100):
    """Test with position overlay only."""

    env = data["env"]
    fields = data["fields"]
    frame_times = data["frame_times"]
    trajectory = data["trajectory"]

    print("\n=== TEST: Position Overlay ===")

    position_overlay = PositionOverlay(
        data=trajectory,
        times=frame_times,
        color="cyan",
        size=8.0,
        trail_length=15,
    )

    viewer = env.animate_fields(
        fields,
        frame_times=frame_times,
        backend="napari",
        overlays=[position_overlay],
        title="Position Overlay Test",
    )

    measure_frame_timing(viewer, n_steps)

    viewer.close()


def test_event_overlay(data: dict, n_steps: int = 100, decay_frames: int = 0):
    """Test with event overlay only."""

    env = data["env"]
    fields = data["fields"]
    frame_times = data["frame_times"]
    trajectory = data["trajectory"]
    event_times = data["event_times"]

    print(f"\n=== TEST: Event Overlay (decay={decay_frames}) ===")

    event_overlay = EventOverlay(
        event_times={"spikes": event_times},
        positions=trajectory,
        position_times=frame_times,
        size=1.5,
        decay_frames=decay_frames,
    )

    viewer = env.animate_fields(
        fields,
        frame_times=frame_times,
        backend="napari",
        overlays=[event_overlay],
        title="Event Overlay Test",
    )

    measure_frame_timing(viewer, n_steps)

    viewer.close()


def test_prerendered(data: dict, n_steps: int = 100):
    """Test with pre-rendered 4D RGB array instead of lazy renderer."""
    import napari
    from matplotlib import colormaps

    from neurospatial.animation.rendering import field_to_rgb_for_napari

    env = data["env"]
    fields = data["fields"]

    print("\n=== TEST: Pre-rendered 4D Array ===")

    # Pre-render all frames
    cmap = colormaps["hot"]
    cmap_lookup = (cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)

    vmin, vmax = np.nanmin(fields), np.nanmax(fields)

    print("Pre-rendering frames...")
    start = time.perf_counter()
    frame_list = []
    for i in range(len(fields)):
        rgb = field_to_rgb_for_napari(env, fields[i], cmap_lookup, vmin, vmax)
        frame_list.append(rgb)
    frames_4d = np.stack(frame_list, axis=0)
    print(f"Pre-render time: {time.perf_counter() - start:.2f}s")
    print(f"4D array shape: {frames_4d.shape}, dtype: {frames_4d.dtype}")
    print(f"Memory: {frames_4d.nbytes / 1e6:.1f} MB")

    # Create viewer with pre-rendered data
    viewer = napari.Viewer(title="Pre-rendered Test")
    viewer.add_image(
        frames_4d,
        name="Pre-rendered Fields",
        rgb=True,
    )

    measure_frame_timing(viewer, n_steps)

    viewer.close()


def test_static_image(data: dict, n_steps: int = 100):
    """Test with static image - no time dimension."""
    import napari
    from matplotlib import colormaps

    from neurospatial.animation.rendering import field_to_rgb_for_napari

    env = data["env"]
    fields = data["fields"]

    print("\n=== TEST: Static Image (no animation) ===")

    # Render single frame
    cmap = colormaps["hot"]
    cmap_lookup = (cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    vmin, vmax = np.nanmin(fields), np.nanmax(fields)

    static_rgb = field_to_rgb_for_napari(env, fields[0], cmap_lookup, vmin, vmax)
    print(f"Static image shape: {static_rgb.shape}")

    viewer = napari.Viewer(title="Static Test")
    viewer.add_image(static_rgb, name="Static Field", rgb=True)

    # Just measure paint times with no frame changes
    import napari.qt

    app = napari.qt.get_app()

    print("Measuring paint without frame changes...")
    time_list: list[float] = []
    for _ in range(n_steps):
        start = time.perf_counter()
        viewer.window._qt_viewer.canvas.update()
        app.processEvents()
        time_list.append((time.perf_counter() - start) * 1000)

    times = np.array(time_list)
    print(f"  Mean: {np.mean(times):.2f} ms")
    print(f"  P95:  {np.percentile(times, 95):.2f} ms")
    print(f"  Max:  {np.max(times):.2f} ms")

    viewer.close()


def measure_frame_timing(viewer, n_steps: int):
    """Measure frame stepping timing."""
    import napari.qt

    app = napari.qt.get_app()
    n_frames = viewer.dims.range[0][1]

    print(f"Stepping through {n_steps} frames (total: {int(n_frames)})...")

    time_list: list[float] = []
    for i in range(n_steps):
        frame_idx = i % int(n_frames)

        start = time.perf_counter()
        viewer.dims.set_current_step(0, frame_idx)
        app.processEvents()
        elapsed = (time.perf_counter() - start) * 1000
        time_list.append(elapsed)

    times = np.array(time_list)

    print("\nFrame timing results:")
    print(f"  Mean:   {np.mean(times):.2f} ms")
    print(f"  Median: {np.median(times):.2f} ms")
    print(f"  P95:    {np.percentile(times, 95):.2f} ms")
    print(f"  Max:    {np.max(times):.2f} ms")
    print(f"  Freezes (>50ms): {np.sum(times > 50)}")
    print(f"  Freezes (>100ms): {np.sum(times > 100)}")

    # Distribution
    print("\n  Distribution:")
    print(
        f"    <10ms:  {np.sum(times < 10):3d} ({100 * np.sum(times < 10) / len(times):.0f}%)"
    )
    print(
        f"    10-30ms:{np.sum((times >= 10) & (times < 30)):3d} ({100 * np.sum((times >= 10) & (times < 30)) / len(times):.0f}%)"
    )
    print(
        f"    30-50ms:{np.sum((times >= 30) & (times < 50)):3d} ({100 * np.sum((times >= 30) & (times < 50)) / len(times):.0f}%)"
    )
    print(
        f"    >50ms:  {np.sum(times >= 50):3d} ({100 * np.sum(times >= 50) / len(times):.0f}%)"
    )


def main():
    parser = argparse.ArgumentParser(description="Isolate napari freeze source")
    parser.add_argument(
        "--test",
        choices=["baseline", "position", "events", "prerender", "static", "all"],
        default="all",
        help="Which test to run",
    )
    parser.add_argument("--frames", type=int, default=500, help="Number of frames")
    parser.add_argument("--steps", type=int, default=100, help="Frames to step through")
    parser.add_argument("--decay", type=int, default=0, help="Event decay frames")

    args = parser.parse_args()

    print("Creating test data...")
    data = create_test_data(n_frames=args.frames)

    if args.test == "all":
        test_static_image(data, args.steps)
        test_baseline(data, args.steps)
        test_position_overlay(data, args.steps)
        test_event_overlay(data, args.steps, args.decay)
        test_prerendered(data, args.steps)
    elif args.test == "baseline":
        test_baseline(data, args.steps)
    elif args.test == "position":
        test_position_overlay(data, args.steps)
    elif args.test == "events":
        test_event_overlay(data, args.steps, args.decay)
    elif args.test == "prerender":
        test_prerendered(data, args.steps)
    elif args.test == "static":
        test_static_image(data, args.steps)


if __name__ == "__main__":
    main()
