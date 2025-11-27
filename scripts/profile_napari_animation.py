#!/usr/bin/env python
"""Profile napari animation performance with real-world data.

This script profiles the napari animation backend using real hippocampal
recording data from the bandit task example. It supports multiple profiling
modes and generates trace files for analysis.

Usage
-----
Basic napari perfmon trace:
    NAPARI_PERFMON=/tmp/napari_trace.json uv run python scripts/profile_napari_animation.py

With py-spy (sampling profiler, shows system calls):
    py-spy record -o profile.svg -- uv run python scripts/profile_napari_animation.py --mode=headless

With cProfile (deterministic profiler):
    uv run python scripts/profile_napari_animation.py --mode=cprofile

Memory profiling:
    uv run python scripts/profile_napari_animation.py --mode=memory

Quick test (fewer frames):
    uv run python scripts/profile_napari_animation.py --frames=1000

Viewing Traces
--------------
- Chrome: chrome://tracing (drag-drop JSON file)
- Speedscope: https://www.speedscope.app/ (upload JSON for flame graphs)
- py-spy SVG: Open in browser directly
"""

from __future__ import annotations

import argparse
import cProfile
import io
import os
import pstats
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "data"))

if TYPE_CHECKING:
    from neurospatial import Environment


@contextmanager
def timer(name: str):
    """Simple context manager for timing code blocks."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"[TIMER] {name}: {elapsed:.3f}s")


def load_data(data_path: Path, n_frames: int | None = None):
    """Load and preprocess the bandit task data.

    Parameters
    ----------
    data_path : Path
        Path to data directory
    n_frames : int | None
        Maximum number of frames to use (for quick testing)

    Returns
    -------
    dict
        Dictionary with all necessary data
    """
    from load_bandit_data import load_neural_recording_from_files

    with timer("Loading raw data"):
        data = load_neural_recording_from_files(data_path, "j1620210710_02_r1")

    position_info = data["position_info"]
    spike_times_all = data["spike_times"]

    # Extract arrays
    positions_2d = position_info[["head_position_x", "head_position_y"]].values
    times_array = position_info.index.values

    # Find a good unit for place field
    spike_counts = [len(s) for s in spike_times_all]
    active_units = [i for i, count in enumerate(spike_counts) if count >= 50]
    example_units = [i for i in active_units if 3000 <= spike_counts[i] <= 20000][:1]

    if not example_units:
        example_units = [active_units[0]] if active_units else [0]

    best_unit = example_units[0]

    # Subsample for animation (500 Hz -> ~30 fps)
    subsample_rate = 17

    if n_frames is not None:
        max_samples = n_frames * subsample_rate
        positions_2d = positions_2d[:max_samples]
        times_array = times_array[:max_samples]

    positions_subsampled = positions_2d[::subsample_rate]
    times_subsampled = times_array[::subsample_rate]

    # Get head direction if available
    head_angles = None
    if "head_orientation" in position_info.columns:
        head_angles_full = position_info["head_orientation"].values
        if n_frames is not None:
            head_angles_full = head_angles_full[:max_samples]
        head_angles = head_angles_full[::subsample_rate]

    return {
        "positions_2d": positions_2d,
        "positions_subsampled": positions_subsampled,
        "times_array": times_array,
        "times_subsampled": times_subsampled,
        "spike_times": spike_times_all[best_unit],
        "head_angles": head_angles,
        "best_unit": best_unit,
    }


def create_environment(positions_2d: np.ndarray) -> Environment:
    """Create 2D environment from position data."""
    from neurospatial import Environment

    with timer("Creating environment"):
        env = Environment.from_samples(
            positions_2d,
            bin_size=4.0,
            name="Maze_2D",
        )
        env.units = "cm"

    return env


def compute_place_field(
    env: Environment,
    spike_times: np.ndarray,
    times_array: np.ndarray,
    positions: np.ndarray,
) -> np.ndarray:
    """Compute place field for the best unit."""
    from neurospatial import compute_place_field

    with timer("Computing place field"):
        firing_rate: np.ndarray = compute_place_field(
            env,
            spike_times,
            times_array,
            positions,
            method="diffusion_kde",
            bandwidth=8.0,
            min_occupancy_seconds=0.5,
        )

    return firing_rate


def create_overlays(positions: np.ndarray, head_angles: np.ndarray | None):
    """Create animation overlays."""
    from neurospatial import HeadDirectionOverlay, PositionOverlay

    overlays: list[PositionOverlay | HeadDirectionOverlay] = []

    with timer("Creating position overlay"):
        position_overlay = PositionOverlay(
            data=positions,
            color="cyan",
            size=15.0,
            trail_length=15,
        )
        overlays.append(position_overlay)

    if head_angles is not None:
        with timer("Creating head direction overlay"):
            head_direction_overlay = HeadDirectionOverlay(
                data=head_angles,
                # Uses defaults: color="hsv", length=15.0, width=3.0
            )
            overlays.append(head_direction_overlay)

    return overlays


def run_napari_animation(
    env: Environment,
    fields: np.ndarray,
    overlays: list,
    headless: bool = False,
    playback_frames: int = 100,
):
    """Run the napari animation and profile it.

    Parameters
    ----------
    env : Environment
        Fitted environment
    fields : ndarray
        Fields to animate (n_frames, n_bins)
    overlays : list
        List of overlay objects
    headless : bool
        If True, don't actually show the viewer (for profiling setup only)
    playback_frames : int
        Number of frames to simulate playback for
    """
    import napari

    n_frames = len(fields)
    print(f"\n[INFO] Animation setup: {n_frames:,} frames, {len(overlays)} overlays")

    # Profile the animation setup
    with timer("Total animation setup"):
        viewer = env.animate_fields(
            fields,
            overlays=overlays,
            backend="napari",
            colormap="hot",
            title="Profiling Session",
        )

    if headless:
        print("[INFO] Headless mode - closing viewer without display")
        viewer.close()
        return

    # Profile frame stepping (simulates playback)
    print(f"\n[INFO] Simulating {playback_frames} frame steps...")

    with timer(f"Stepping through {playback_frames} frames"):
        for i in range(min(playback_frames, n_frames)):
            viewer.dims.set_current_step(0, i)
            # Process Qt events to measure real update time
            napari.qt.get_app().processEvents()

    print("\n[INFO] Viewer open - close window to finish profiling")
    napari.run()


def profile_with_cprofile(func, *args, **kwargs):
    """Run function with cProfile and print stats."""
    profiler = cProfile.Profile()
    profiler.enable()

    try:
        result = func(*args, **kwargs)
    finally:
        profiler.disable()

    # Print stats
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumulative")
    stats.print_stats(50)  # Top 50 functions
    print("\n" + "=" * 80)
    print("CPROFILE RESULTS (top 50 by cumulative time)")
    print("=" * 80)
    print(stream.getvalue())

    # Also save to file
    profiler.dump_stats("/tmp/napari_animation.prof")
    print("[INFO] Full profile saved to /tmp/napari_animation.prof")
    print("       View with: snakeviz /tmp/napari_animation.prof")

    return result


def profile_memory():
    """Profile memory usage during animation setup."""
    try:
        import tracemalloc
    except ImportError:
        print("[ERROR] tracemalloc not available")
        return

    tracemalloc.start()

    # Run the animation setup
    data_path = PROJECT_ROOT / "data"
    data = load_data(data_path, n_frames=5000)
    env = create_environment(data["positions_2d"])

    snapshot1 = tracemalloc.take_snapshot()

    place_field = compute_place_field(
        env, data["spike_times"], data["times_array"], data["positions_2d"]
    )
    n_frames = len(data["positions_subsampled"])
    np.tile(place_field, (n_frames, 1))

    snapshot2 = tracemalloc.take_snapshot()

    create_overlays(data["positions_subsampled"], data["head_angles"])

    snapshot3 = tracemalloc.take_snapshot()

    # Print memory differences
    print("\n" + "=" * 80)
    print("MEMORY ALLOCATION ANALYSIS")
    print("=" * 80)

    print("\n[After creating fields array]")
    top_stats = snapshot2.compare_to(snapshot1, "lineno")
    for stat in top_stats[:10]:
        print(stat)

    print("\n[After creating overlays]")
    top_stats = snapshot3.compare_to(snapshot2, "lineno")
    for stat in top_stats[:10]:
        print(stat)

    current, peak = tracemalloc.get_traced_memory()
    print(f"\nCurrent memory: {current / 1024 / 1024:.1f} MB")
    print(f"Peak memory: {peak / 1024 / 1024:.1f} MB")

    tracemalloc.stop()


def print_perfmon_instructions():
    """Print instructions for napari perfmon."""
    print("\n" + "=" * 80)
    print("NAPARI PERFMON INSTRUCTIONS")
    print("=" * 80)
    print("""
To generate a detailed trace file, run with NAPARI_PERFMON environment variable:

    NAPARI_PERFMON=/tmp/napari_trace.json uv run python scripts/profile_napari_animation.py

This creates a Chrome Trace format JSON file that you can view:

1. Chrome/Chromium:
   - Open chrome://tracing
   - Click "Load" and select /tmp/napari_trace.json
   - Navigate timeline with WASD keys

2. Speedscope (flame graphs):
   - Go to https://www.speedscope.app/
   - Drag and drop the JSON file
   - View as flame graph, sandwich, or timeline

Key things to look for in traces:
- Long synchronous operations blocking the main thread
- Repeated expensive operations during playback
- Memory allocation patterns
- Qt event processing time
""")


def main():
    parser = argparse.ArgumentParser(description="Profile napari animation performance")
    parser.add_argument(
        "--mode",
        choices=["interactive", "headless", "cprofile", "memory"],
        default="interactive",
        help="Profiling mode (default: interactive)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Max frames to use (default: all ~41k frames)",
    )
    parser.add_argument(
        "--playback-frames",
        type=int,
        default=100,
        help="Number of frames to step through for playback profiling (default: 100)",
    )
    args = parser.parse_args()

    # Check for perfmon
    perfmon_path = os.environ.get("NAPARI_PERFMON")
    if perfmon_path:
        print(f"[INFO] Napari perfmon enabled, output: {perfmon_path}")
    else:
        print_perfmon_instructions()

    if args.mode == "memory":
        profile_memory()
        return

    # Load data
    data_path = PROJECT_ROOT / "data"
    print(f"\n[INFO] Loading data from {data_path}")

    data = load_data(data_path, n_frames=args.frames)

    print(f"[INFO] Position samples: {len(data['positions_2d']):,}")
    print(f"[INFO] Animation frames: {len(data['positions_subsampled']):,}")
    print(
        f"[INFO] Head direction: {'Available' if data['head_angles'] is not None else 'Not available'}"
    )

    # Create environment
    env = create_environment(data["positions_2d"])
    print(f"[INFO] Environment: {env.n_bins} bins")

    # Compute place field
    place_field = compute_place_field(
        env, data["spike_times"], data["times_array"], data["positions_2d"]
    )

    # Create tiled fields for animation
    n_frames = len(data["positions_subsampled"])
    with timer("Tiling place field for animation"):
        fields = np.tile(place_field, (n_frames, 1))
    print(f"[INFO] Fields array shape: {fields.shape}")
    print(f"[INFO] Fields memory: {fields.nbytes / 1024 / 1024:.1f} MB")

    # Create overlays
    overlays = create_overlays(
        data["positions_subsampled"],
        data["head_angles"],
    )

    # Run animation
    if args.mode == "cprofile":
        profile_with_cprofile(
            run_napari_animation,
            env,
            fields,
            overlays,
            headless=True,
            playback_frames=args.playback_frames,
        )
    else:
        run_napari_animation(
            env,
            fields,
            overlays,
            headless=(args.mode == "headless"),
            playback_frames=args.playback_frames,
        )


if __name__ == "__main__":
    main()
