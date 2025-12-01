#!/usr/bin/env python
"""Benchmark napari playback performance with synthetic data.

This script benchmarks the napari animation backend using synthetic test data,
supporting per-overlay selection via command-line arguments.

Dependencies
------------
- scipy : Required for head_direction and timeseries overlays (gaussian filtering)
- napari : Required for viewer backend
- benchmark_datasets : Local module in scripts/ directory

Usage
-----
Run with specific overlays:
    uv run python scripts/benchmark_napari_playback.py --position --events

Run with all overlays:
    uv run python scripts/benchmark_napari_playback.py --all-overlays

Quick test (fewer frames):
    uv run python scripts/benchmark_napari_playback.py --frames 100 --playback-frames 50

With napari perfmon tracing:
    NAPARI_PERFMON=/tmp/napari_trace.json uv run python scripts/benchmark_napari_playback.py

Headless mode (no viewer display, for CI):
    uv run python scripts/benchmark_napari_playback.py --headless

Examples
--------
>>> # Run from command line:
>>> # uv run python scripts/benchmark_napari_playback.py --position --head-direction
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

# Add project to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

if TYPE_CHECKING:
    from neurospatial import Environment

# Default configuration
DEFAULT_FRAMES = 500
DEFAULT_PLAYBACK_FRAMES = 100
DEFAULT_GRID_SIZE = 50
DEFAULT_SEED = 42
DEFAULT_FPS = 30.0  # Sampling rate for synthetic frame times


@dataclass
class TimingMetrics:
    """Container for benchmark timing metrics.

    Attributes
    ----------
    setup_time_s : float
        Time to set up viewer and overlays in seconds.
    frame_times_ms : list[float]
        Per-frame render times in milliseconds.
    total_frames : int
        Total number of frames stepped through.
    overlays_enabled : list[str]
        List of overlay names that were enabled.
    """

    setup_time_s: float
    frame_times_ms: list[float]
    total_frames: int
    overlays_enabled: list[str]


@contextmanager
def timer() -> Iterator[dict[str, float]]:
    """Context manager for timing code blocks.

    Yields
    ------
    dict[str, float]
        Dictionary with 'elapsed' key set after context exits.
    """
    result: dict[str, float] = {"elapsed": 0.0}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["elapsed"] = time.perf_counter() - start


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser with overlay selection options.
    """
    parser = argparse.ArgumentParser(
        description="Benchmark napari playback performance with synthetic data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with position overlay only
  uv run python scripts/benchmark_napari_playback.py --position

  # Run with all overlays
  uv run python scripts/benchmark_napari_playback.py --all-overlays

  # Quick test with fewer frames
  uv run python scripts/benchmark_napari_playback.py --frames 100 --playback-frames 50

  # With napari perfmon tracing
  NAPARI_PERFMON=/tmp/trace.json uv run python scripts/benchmark_napari_playback.py
        """,
    )

    # Frame configuration
    parser.add_argument(
        "--frames",
        type=int,
        default=DEFAULT_FRAMES,
        help=f"Number of animation frames to generate (default: {DEFAULT_FRAMES})",
    )
    parser.add_argument(
        "--playback-frames",
        type=int,
        default=DEFAULT_PLAYBACK_FRAMES,
        help=f"Number of frames to step through for timing (default: {DEFAULT_PLAYBACK_FRAMES})",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=DEFAULT_GRID_SIZE,
        help=f"Grid size for environment (default: {DEFAULT_GRID_SIZE})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducibility (default: {DEFAULT_SEED})",
    )

    # Overlay selection
    overlay_group = parser.add_argument_group("Overlay selection")
    overlay_group.add_argument(
        "--position",
        action="store_true",
        help="Include position overlay with trail",
    )
    overlay_group.add_argument(
        "--bodyparts",
        action="store_true",
        help="Include bodypart overlay with skeleton",
    )
    overlay_group.add_argument(
        "--head-direction",
        action="store_true",
        help="Include head direction overlay",
    )
    overlay_group.add_argument(
        "--events",
        action="store_true",
        help="Include event overlay (spike-like events)",
    )
    overlay_group.add_argument(
        "--timeseries",
        action="store_true",
        help="Include time series dock widget",
    )
    overlay_group.add_argument(
        "--video",
        action="store_true",
        help="Include video overlay (synthetic noise video)",
    )
    overlay_group.add_argument(
        "--all-overlays",
        action="store_true",
        help="Enable all overlays (equivalent to --position --bodyparts "
        "--head-direction --events --timeseries --video)",
    )

    # Execution mode
    mode_group = parser.add_argument_group("Execution mode")
    mode_group.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode (close viewer after setup, for CI testing)",
    )
    mode_group.add_argument(
        "--no-playback",
        action="store_true",
        help="Skip playback timing (only measure setup time)",
    )
    mode_group.add_argument(
        "--auto-close",
        action="store_true",
        help="Automatically close viewer after timing (for automated benchmarks)",
    )

    return parser


def generate_benchmark_data(
    n_frames: int,
    grid_size: int,
    seed: int | None = None,
) -> dict[str, Any]:
    """Generate synthetic benchmark data.

    Parameters
    ----------
    n_frames : int
        Number of animation frames.
    grid_size : int
        Size of the square grid.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary containing:
        - env: Fitted Environment
        - fields: Array of shape (n_frames, n_bins)
        - frame_times: Array of timestamps
    """
    from benchmark_datasets import (
        BenchmarkConfig,
        create_benchmark_env,
        create_benchmark_fields,
    )

    config = BenchmarkConfig(
        name="benchmark",
        n_frames=n_frames,
        grid_size=grid_size,
    )

    env = create_benchmark_env(config, seed=seed)
    fields = create_benchmark_fields(env, config, seed=seed)

    # Generate frame times
    frame_times = np.arange(n_frames) / DEFAULT_FPS

    return {
        "env": env,
        "fields": fields,
        "frame_times": frame_times,
    }


def _generate_smooth_trajectory(
    n_frames: int,
    dim_ranges: list[tuple[float, float]],
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Generate a smooth random walk trajectory with boundary reflection.

    Uses a Gaussian step model with specular reflection at dimension boundaries.
    This produces a continuous trajectory that stays within the environment bounds
    without introducing discontinuities or bias toward boundaries.

    Parameters
    ----------
    n_frames : int
        Number of frames.
    dim_ranges : list of (min, max) tuples
        Bounds for each dimension (e.g., [(0.0, 100.0), (0.0, 100.0)] for 2D).
    rng : np.random.Generator
        Random number generator for reproducibility.

    Returns
    -------
    NDArray[np.float64]
        Trajectory of shape (n_frames, n_dims). Each row is a position vector
        in environment coordinates.

    Notes
    -----
    The boundary reflection algorithm uses specular reflection: when a step
    crosses a boundary, the excess distance is reflected back. This process
    repeats iteratively until the position is within bounds, with a safety
    limit to prevent infinite loops on numerical edge cases.
    """
    n_dims = len(dim_ranges)

    # Start at a random position
    trajectory = np.zeros((n_frames, n_dims))
    for dim in range(n_dims):
        dim_min, dim_max = dim_ranges[dim]
        trajectory[0, dim] = rng.uniform(dim_min, dim_max)

    # Generate smooth random walk
    step_size = 0.5
    for frame in range(1, n_frames):
        step = rng.normal(0, step_size, size=n_dims)
        trajectory[frame] = trajectory[frame - 1] + step

        # Reflect at boundaries (with max iterations to prevent edge case infinite loops)
        for dim in range(n_dims):
            dim_min, dim_max = dim_ranges[dim]
            pos = trajectory[frame, dim]
            max_reflections = 10  # Safety limit for numerical edge cases
            reflections = 0
            while (pos < dim_min or pos > dim_max) and reflections < max_reflections:
                if pos < dim_min:
                    pos = 2 * dim_min - pos
                if pos > dim_max:
                    pos = 2 * dim_max - pos
                reflections += 1
            # Clamp if still out of bounds after max reflections
            pos = np.clip(pos, dim_min, dim_max)
            trajectory[frame, dim] = pos

    return trajectory


def create_selected_overlays(
    env: Environment,
    n_frames: int,
    seed: int | None = None,
    position: bool = False,
    bodyparts: bool = False,
    head_direction: bool = False,
    events: bool = False,
    timeseries: bool = False,
    video: bool = False,
) -> list[Any]:
    """Create overlays based on selection flags.

    Parameters
    ----------
    env : Environment
        Fitted environment.
    n_frames : int
        Number of animation frames.
    seed : int, optional
        Random seed for reproducibility.
    position : bool, default=False
        Include position overlay.
    bodyparts : bool, default=False
        Include bodypart overlay with skeleton.
    head_direction : bool, default=False
        Include head direction overlay.
    events : bool, default=False
        Include event overlay.
    timeseries : bool, default=False
        Include time series overlay.
    video : bool, default=False
        Include video overlay (synthetic noise video).

    Returns
    -------
    list[Any]
        List of overlay objects. Possible types: PositionOverlay, BodypartOverlay,
        HeadDirectionOverlay, EventOverlay, TimeSeriesOverlay, VideoOverlay.
    """
    from neurospatial.animation.overlays import (
        BodypartOverlay,
        EventOverlay,
        HeadDirectionOverlay,
        PositionOverlay,
        TimeSeriesOverlay,
        VideoOverlay,
    )
    from neurospatial.animation.skeleton import Skeleton

    rng = np.random.default_rng(seed)
    overlays: list[Any] = []

    dim_ranges = env.dimension_ranges
    if dim_ranges is None:
        raise ValueError("Environment must have dimension_ranges set")
    dim_ranges_list: list[tuple[float, float]] = list(dim_ranges)

    # Generate trajectory for position-based overlays
    trajectory = _generate_smooth_trajectory(n_frames, dim_ranges_list, rng)

    # Position overlay
    if position:
        position_overlay = PositionOverlay(
            data=trajectory.copy(),
            color="red",
            size=12.0,
            trail_length=15,
        )
        overlays.append(position_overlay)

    # Bodypart overlay with skeleton
    if bodyparts:
        n_bodyparts = 5
        bodypart_names = [f"bp{i}" for i in range(n_bodyparts)]

        edges = [
            (bodypart_names[i], bodypart_names[i + 1]) for i in range(n_bodyparts - 1)
        ]

        skeleton = Skeleton(
            name="benchmark_skeleton",
            nodes=tuple(bodypart_names),
            edges=tuple(edges),
            node_colors=dict.fromkeys(bodypart_names, "white"),
            edge_color="gray",
            edge_width=2.0,
        )

        bodypart_data: dict[str, NDArray[np.float64]] = {}
        for _, bp_name in enumerate(bodypart_names):
            offset = rng.uniform(-2, 2, size=2)
            jitter = rng.normal(0, 0.5, size=(n_frames, 2))
            bp_positions = trajectory + offset + jitter
            for dim in range(2):
                dim_min, dim_max = dim_ranges_list[dim]
                bp_positions[:, dim] = np.clip(bp_positions[:, dim], dim_min, dim_max)
            bodypart_data[bp_name] = bp_positions

        bodypart_overlay = BodypartOverlay(
            data=bodypart_data,
            skeleton=skeleton,
        )
        overlays.append(bodypart_overlay)

    # Head direction overlay
    if head_direction:
        velocity = np.diff(trajectory, axis=0, prepend=trajectory[:1])
        head_angles = np.arctan2(velocity[:, 1], velocity[:, 0])
        head_angles += rng.normal(0, 0.1, size=n_frames)

        from scipy.ndimage import gaussian_filter1d

        head_angles = gaussian_filter1d(head_angles, sigma=5)
        head_angles = np.arctan2(np.sin(head_angles), np.cos(head_angles))

        head_direction_overlay = HeadDirectionOverlay(
            data=head_angles,
            color="yellow",
            length=3.0,
        )
        overlays.append(head_direction_overlay)

    # Event overlay (spike-like events)
    if events:
        # Generate random spike events
        n_events = n_frames * 2  # Average 2 events per frame
        event_frame_indices = rng.integers(0, n_frames, size=n_events)
        # Convert frame indices to times
        event_times_arr = event_frame_indices.astype(np.float64) / DEFAULT_FPS

        event_pos = np.zeros((n_events, 2))

        for i, frame_idx in enumerate(event_frame_indices):
            # Event position near trajectory
            base_pos = trajectory[frame_idx]
            offset = rng.normal(0, 2, size=2)
            pos = base_pos + offset
            for dim in range(2):
                dim_min, dim_max = dim_ranges_list[dim]
                pos[dim] = np.clip(pos[dim], dim_min, dim_max)
            event_pos[i] = pos

        event_overlay = EventOverlay(
            event_times={"spikes": event_times_arr},
            event_positions={"spikes": event_pos},
            colors={"spikes": "cyan"},
            size=5.0,
            decay_frames=10,  # Events decay over 10 frames
        )
        overlays.append(event_overlay)

    # Time series overlay
    if timeseries:
        # Generate synthetic time series (e.g., LFP-like signal)
        from scipy.signal import butter, filtfilt

        ts_data = rng.standard_normal(n_frames)
        # Low-pass filter to create smooth signal
        b, a = butter(3, 0.1)
        ts_data = filtfilt(b, a, ts_data)

        # Generate timestamps
        ts_times = np.arange(n_frames) / DEFAULT_FPS

        timeseries_overlay = TimeSeriesOverlay(
            data=ts_data,
            times=ts_times,
            label="LFP (mV)",
            color="cyan",
            window_seconds=2.0,
        )
        overlays.append(timeseries_overlay)

    # Video overlay (synthetic noise video)
    if video:
        # Create synthetic video matching environment dimensions
        # Use 100x100 pixel video for reasonable performance
        video_height, video_width = 100, 100

        # Generate noisy grayscale video, convert to RGB
        video_frames = rng.integers(
            50, 200, size=(n_frames, video_height, video_width), dtype=np.uint8
        )
        # Convert to RGB by stacking the grayscale channel
        video_rgb = np.stack([video_frames] * 3, axis=-1)

        # Create timestamps for video frames
        video_times = np.arange(n_frames) / DEFAULT_FPS

        video_overlay = VideoOverlay(
            source=video_rgb,
            times=video_times,
            alpha=0.3,  # Lower opacity so field is visible
            z_order="below",  # Video behind field
        )
        overlays.append(video_overlay)

    return overlays


def print_timing_metrics(metrics: dict[str, Any]) -> None:
    """Print timing metrics in a readable format.

    Parameters
    ----------
    metrics : dict
        Dictionary containing timing data with keys:
        - setup_time: Setup time in seconds
        - frame_times_ms: List of per-frame times in ms
        - total_frames: Number of frames
        - overlays_enabled: List of overlay names
    """
    print("\n" + "=" * 60)
    print("NAPARI PLAYBACK BENCHMARK RESULTS")
    print("=" * 60)

    # Overlays
    overlays = metrics.get("overlays_enabled", [])
    if overlays:
        print(f"\nOverlays enabled: {', '.join(overlays)}")
    else:
        print("\nOverlays enabled: none (field only)")

    # Setup time
    setup_time = metrics.get("setup_time", 0.0)
    print(f"\nSetup time: {setup_time:.3f}s")

    # Frame timing statistics
    frame_times = metrics.get("frame_times_ms", [])
    total_frames = metrics.get("total_frames", len(frame_times))

    if frame_times:
        mean_time = np.mean(frame_times)
        median_time = np.median(frame_times)
        p95_time = np.percentile(frame_times, 95)
        max_time = np.max(frame_times)
        min_time = np.min(frame_times)

        print(f"\nFrame timing ({total_frames} frames):")
        print(f"  Mean:   {mean_time:.2f} ms")
        print(f"  Median: {median_time:.2f} ms")
        print(f"  P95:    {p95_time:.2f} ms")
        print(f"  Min:    {min_time:.2f} ms")
        print(f"  Max:    {max_time:.2f} ms")

        # Performance assessment
        target_fps = 30
        target_frame_time = 1000 / target_fps  # ~33.3 ms

        print(
            f"\nPerformance assessment (target: {target_fps} fps = {target_frame_time:.1f} ms/frame):"
        )
        if mean_time < target_frame_time:
            achievable_fps = 1000 / mean_time
            print(f"  ✓ Target met: Mean frame time allows ~{achievable_fps:.1f} fps")
        else:
            achievable_fps = 1000 / mean_time
            print(
                f"  ✗ Target not met: Mean frame time allows only ~{achievable_fps:.1f} fps"
            )

        # Frames that exceeded target
        slow_frames = sum(1 for t in frame_times if t > target_frame_time)
        slow_pct = (slow_frames / len(frame_times)) * 100
        print(
            f"  Frames exceeding target: {slow_frames}/{len(frame_times)} ({slow_pct:.1f}%)"
        )

    print("\n" + "=" * 60)


def run_benchmark(
    env: Environment,
    fields: NDArray[np.float32],
    frame_times: NDArray[np.float64],
    overlays: list[Any],
    playback_frames: int,
    headless: bool = False,
    no_playback: bool = False,
    auto_close: bool = False,
) -> TimingMetrics:
    """Run the benchmark and collect timing metrics.

    Parameters
    ----------
    env : Environment
        Fitted environment.
    fields : ndarray
        Field data of shape (n_frames, n_bins).
    frame_times : ndarray
        Frame timestamps.
    overlays : list
        List of overlay objects.
    playback_frames : int
        Number of frames to step through.
    headless : bool
        If True, close viewer without display.
    no_playback : bool
        If True, skip playback timing.
    auto_close : bool
        If True, close viewer automatically after timing.

    Returns
    -------
    TimingMetrics
        Container with timing metrics.
    """
    import napari

    n_frames = len(fields)

    # Identify enabled overlays
    overlay_names = []
    for overlay in overlays:
        class_name = type(overlay).__name__
        if "Position" in class_name:
            overlay_names.append("position")
        elif "Bodypart" in class_name:
            overlay_names.append("bodyparts")
        elif "HeadDirection" in class_name:
            overlay_names.append("head_direction")
        elif "Event" in class_name:
            overlay_names.append("events")
        elif "TimeSeries" in class_name:
            overlay_names.append("timeseries")
        elif "Video" in class_name:
            overlay_names.append("video")

    # Time setup
    with timer() as setup_timer:
        viewer = env.animate_fields(
            fields,
            frame_times=frame_times,
            overlays=overlays if overlays else None,
            backend="napari",
            colormap="viridis",
            title="Benchmark Session",
        )

    setup_time = setup_timer["elapsed"]
    print(f"[INFO] Setup completed in {setup_time:.3f}s")

    if headless:
        print("[INFO] Headless mode - closing viewer")
        viewer.close()
        return TimingMetrics(
            setup_time_s=setup_time,
            frame_times_ms=[],
            total_frames=0,
            overlays_enabled=overlay_names,
        )

    if no_playback:
        print("[INFO] Skipping playback timing (--no-playback)")
        print("[INFO] Viewer open - close window to finish")
        napari.run()
        return TimingMetrics(
            setup_time_s=setup_time,
            frame_times_ms=[],
            total_frames=0,
            overlays_enabled=overlay_names,
        )

    # Time frame stepping
    print(f"\n[INFO] Timing {playback_frames} frame steps...")
    frame_times_ms: list[float] = []
    app = napari.qt.get_app()

    frames_to_step = min(playback_frames, n_frames)
    for i in range(frames_to_step):
        start = time.perf_counter()
        viewer.dims.set_current_step(0, i)
        app.processEvents()  # Process Qt events
        elapsed_ms = (time.perf_counter() - start) * 1000
        frame_times_ms.append(elapsed_ms)

    print("[INFO] Frame stepping completed")

    if auto_close:
        print("[INFO] Auto-closing viewer")
        viewer.close()
    else:
        print("[INFO] Viewer open - close window to finish")
        napari.run()

    return TimingMetrics(
        setup_time_s=setup_time,
        frame_times_ms=frame_times_ms,
        total_frames=frames_to_step,
        overlays_enabled=overlay_names,
    )


def main() -> None:
    """Main entry point for the benchmark script."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Check for napari perfmon
    perfmon_path = os.environ.get("NAPARI_PERFMON")
    if perfmon_path:
        print(f"[INFO] Napari perfmon enabled, output: {perfmon_path}")

    # Resolve overlay flags
    include_position = args.position or args.all_overlays
    include_bodyparts = args.bodyparts or args.all_overlays
    include_head_direction = args.head_direction or args.all_overlays
    include_events = args.events or args.all_overlays
    include_timeseries = args.timeseries or args.all_overlays
    include_video = args.video or args.all_overlays

    # If no overlays specified and not --all-overlays, default to position only
    if not any(
        [
            args.position,
            args.bodyparts,
            args.head_direction,
            args.events,
            args.timeseries,
            args.video,
            args.all_overlays,
        ]
    ):
        print("[INFO] No overlays specified, defaulting to position overlay")
        include_position = True

    # Print configuration
    print("\n[INFO] Benchmark Configuration:")
    print(f"  Frames: {args.frames}")
    print(f"  Playback frames: {args.playback_frames}")
    print(f"  Grid size: {args.grid_size}")
    print(f"  Seed: {args.seed}")
    print(
        f"  Overlays: position={include_position}, bodyparts={include_bodyparts}, "
        f"head_direction={include_head_direction}, events={include_events}, "
        f"timeseries={include_timeseries}, video={include_video}"
    )

    # Generate data
    print("\n[INFO] Generating benchmark data...")
    with timer() as gen_timer:
        data = generate_benchmark_data(
            n_frames=args.frames,
            grid_size=args.grid_size,
            seed=args.seed,
        )

    print(f"[INFO] Data generation: {gen_timer['elapsed']:.3f}s")
    print(f"[INFO] Environment: {data['env'].n_bins} bins")
    print(f"[INFO] Fields shape: {data['fields'].shape}")

    # Create overlays
    print("\n[INFO] Creating overlays...")
    with timer() as overlay_timer:
        overlays = create_selected_overlays(
            env=data["env"],
            n_frames=args.frames,
            seed=args.seed,
            position=include_position,
            bodyparts=include_bodyparts,
            head_direction=include_head_direction,
            events=include_events,
            timeseries=include_timeseries,
            video=include_video,
        )

    print(f"[INFO] Overlay creation: {overlay_timer['elapsed']:.3f}s")
    print(f"[INFO] Created {len(overlays)} overlay(s)")

    # Run benchmark
    print("\n[INFO] Starting napari benchmark...")
    metrics = run_benchmark(
        env=data["env"],
        fields=data["fields"],
        frame_times=data["frame_times"],
        overlays=overlays,
        playback_frames=args.playback_frames,
        headless=args.headless,
        no_playback=args.no_playback,
        auto_close=args.auto_close,
    )

    # Print results
    print_timing_metrics(
        {
            "setup_time": metrics.setup_time_s,
            "frame_times_ms": metrics.frame_times_ms,
            "total_frames": metrics.total_frames,
            "overlays_enabled": metrics.overlays_enabled,
        }
    )


if __name__ == "__main__":
    main()
