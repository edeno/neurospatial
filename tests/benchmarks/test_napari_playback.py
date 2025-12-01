"""Napari playback performance benchmarks with pytest-benchmark.

This module provides automated benchmarks to track napari animation performance
after optimization work. Tests use pytest-benchmark for reproducible measurements
and comparison against baselines.

Baseline targets (from docs/performance_baseline.md):
- Individual overlays: <33.3ms per frame (30 fps)
- All overlays combined: Target <40ms (25 fps acceptable post-optimization)

Usage
-----
Run benchmarks:
    uv run pytest tests/benchmarks/test_napari_playback.py -v

Save baseline:
    uv run pytest tests/benchmarks/test_napari_playback.py --benchmark-save=napari_baseline

Compare against baseline:
    uv run pytest tests/benchmarks/test_napari_playback.py --benchmark-compare=napari_baseline

Generate histogram:
    uv run pytest tests/benchmarks/test_napari_playback.py --benchmark-histogram

Notes
-----
All tests are marked with @pytest.mark.slow and @pytest.mark.xdist_group(name="napari_gui")
to prevent parallel execution (napari requires single Qt event loop) and to exclude
from default test runs.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from numpy.typing import NDArray

# Add scripts directory to path for benchmark_datasets import
SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"
if not SCRIPTS_DIR.exists():
    raise RuntimeError(f"Scripts directory not found: {SCRIPTS_DIR}")
sys.path.insert(0, str(SCRIPTS_DIR))

if TYPE_CHECKING:
    import napari

    from neurospatial import Environment


# =============================================================================
# Constants and Configuration
# =============================================================================

# Performance targets in milliseconds
TARGET_FPS = 30
TARGET_FRAME_TIME_MS = 1000 / TARGET_FPS  # 33.3 ms
ACCEPTABLE_FPS = 25
ACCEPTABLE_FRAME_TIME_MS = 1000 / ACCEPTABLE_FPS  # 40 ms

# Default benchmark parameters
DEFAULT_FRAMES = 100  # Reduced for faster CI runs
DEFAULT_PLAYBACK_FRAMES = 50  # Frames to step through for timing
DEFAULT_GRID_SIZE = 50  # 50x50 grid
DEFAULT_SEED = 42


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def benchmark_env() -> Environment:
    """Create benchmark environment (50x50 grid, ~845 bins).

    This fixture is shared across all module tests (module-scoped).
    Do NOT mutate the environment (e.g., via env.regions.update_region()).
    Tests should only use read-only methods.

    Returns
    -------
    Environment
        A fitted 2D environment with approximately 845 active bins.
    """
    from benchmark_datasets import BenchmarkConfig, create_benchmark_env

    config = BenchmarkConfig(
        name="napari_benchmark",
        n_frames=DEFAULT_FRAMES,
        grid_size=DEFAULT_GRID_SIZE,
    )
    return cast("Environment", create_benchmark_env(config, seed=DEFAULT_SEED))


@pytest.fixture(scope="module")
def benchmark_fields(benchmark_env: Environment) -> NDArray[np.float32]:
    """Create benchmark fields (DEFAULT_FRAMES frames)."""
    from benchmark_datasets import BenchmarkConfig, create_benchmark_fields

    config = BenchmarkConfig(
        name="napari_benchmark",
        n_frames=DEFAULT_FRAMES,
        grid_size=DEFAULT_GRID_SIZE,
    )
    return cast(
        "NDArray[np.float32]",
        create_benchmark_fields(benchmark_env, config, seed=DEFAULT_SEED),
    )


@pytest.fixture(scope="module")
def frame_times() -> NDArray[np.float64]:
    """Create frame timestamps at 30 fps."""
    return np.arange(DEFAULT_FRAMES) / TARGET_FPS


@pytest.fixture(scope="module")
def trajectory(benchmark_env: Environment) -> NDArray[np.float64]:
    """Generate smooth trajectory for position-based overlays.

    Uses a random walk with boundary reflection to ensure trajectory
    stays within environment bounds. This produces realistic movement
    patterns for benchmarking position-based overlays.

    The boundary reflection algorithm uses specular reflection: when a
    step crosses a boundary, the excess distance is reflected back.

    Parameters
    ----------
    benchmark_env : Environment
        The benchmark environment for dimension bounds.

    Returns
    -------
    NDArray[np.float64]
        Trajectory array of shape (DEFAULT_FRAMES, n_dims) in environment
        coordinate space.
    """
    rng = np.random.default_rng(DEFAULT_SEED)

    dim_ranges = benchmark_env.dimension_ranges
    if dim_ranges is None:
        raise ValueError("Environment must have dimension_ranges set")
    dim_ranges_list = list(dim_ranges)

    n_dims = len(dim_ranges_list)
    trajectory = np.zeros((DEFAULT_FRAMES, n_dims))

    # Start at random position
    for dim in range(n_dims):
        dim_min, dim_max = dim_ranges_list[dim]
        trajectory[0, dim] = rng.uniform(dim_min, dim_max)

    # Random walk with boundary reflection
    step_size = 0.5
    for frame in range(1, DEFAULT_FRAMES):
        step = rng.normal(0, step_size, size=n_dims)
        trajectory[frame] = trajectory[frame - 1] + step

        for dim in range(n_dims):
            dim_min, dim_max = dim_ranges_list[dim]
            pos = trajectory[frame, dim]
            max_reflections = 10
            reflections = 0
            while (pos < dim_min or pos > dim_max) and reflections < max_reflections:
                if pos < dim_min:
                    pos = 2 * dim_min - pos
                if pos > dim_max:
                    pos = 2 * dim_max - pos
                reflections += 1
            trajectory[frame, dim] = np.clip(pos, dim_min, dim_max)

    return trajectory


# =============================================================================
# Helper Functions
# =============================================================================


def _step_frames(viewer: napari.Viewer, n_frames: int) -> list[float]:
    """Step through frames and return timing in milliseconds.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance.
    n_frames : int
        Number of frames to step through.

    Returns
    -------
    list[float]
        Per-frame times in milliseconds.
    """
    import time

    from napari._qt.qt_main_window import get_qapp

    app = get_qapp()
    frame_times_ms: list[float] = []

    for i in range(n_frames):
        start = time.perf_counter()
        viewer.dims.set_current_step(0, i)
        app.processEvents()
        elapsed_ms = (time.perf_counter() - start) * 1000
        frame_times_ms.append(elapsed_ms)

    return frame_times_ms


def _compute_stats(frame_times_ms: list[float]) -> dict[str, float]:
    """Compute timing statistics.

    Parameters
    ----------
    frame_times_ms : list[float]
        Per-frame times in milliseconds.

    Returns
    -------
    dict[str, float]
        Dictionary with mean, median, p95, min, max statistics.
    """
    return {
        "mean": float(np.mean(frame_times_ms)),
        "median": float(np.median(frame_times_ms)),
        "p95": float(np.percentile(frame_times_ms, 95)),
        "min": float(np.min(frame_times_ms)),
        "max": float(np.max(frame_times_ms)),
    }


# =============================================================================
# Individual Overlay Benchmarks
# =============================================================================


@pytest.mark.slow
@pytest.mark.xdist_group(name="napari_gui")
class TestNapariPlaybackIndividualOverlays:
    """Benchmark napari playback with individual overlays.

    Each test measures frame stepping performance with a single overlay type.
    Target: <33.3ms mean frame time (30 fps).
    """

    def test_field_only_playback(
        self,
        benchmark,
        benchmark_env: Environment,
        benchmark_fields: NDArray[np.float32],
        frame_times: NDArray[np.float64],
    ) -> None:
        """Benchmark playback with field only (no overlays).

        This establishes the baseline napari rendering overhead.

        Baseline: N/A (reference implementation)
        Target: <33.3ms (30 fps)
        """
        pytest.importorskip("napari")

        def setup_and_step():
            viewer = benchmark_env.animate_fields(
                benchmark_fields,
                frame_times=frame_times,
                backend="napari",
                colormap="viridis",
            )
            try:
                times = _step_frames(viewer, DEFAULT_PLAYBACK_FRAMES)
                return _compute_stats(times)
            finally:
                viewer.close()

        result = benchmark(setup_and_step)

        # Verify target met
        assert result["mean"] < TARGET_FRAME_TIME_MS, (
            f"Mean frame time {result['mean']:.2f}ms exceeds target {TARGET_FRAME_TIME_MS:.1f}ms"
        )

    def test_position_overlay_playback(
        self,
        benchmark,
        benchmark_env: Environment,
        benchmark_fields: NDArray[np.float32],
        frame_times: NDArray[np.float64],
        trajectory: NDArray[np.float64],
    ) -> None:
        """Benchmark playback with position overlay.

        Baseline: 21.87ms mean (~46 fps)
        Target: <33.3ms (30 fps)
        """
        pytest.importorskip("napari")
        from neurospatial.animation.overlays import PositionOverlay

        def setup_and_step():
            overlay = PositionOverlay(
                data=trajectory.copy(),
                color="red",
                size=12.0,
                trail_length=15,
            )
            viewer = benchmark_env.animate_fields(
                benchmark_fields,
                frame_times=frame_times,
                overlays=[overlay],
                backend="napari",
                colormap="viridis",
            )
            try:
                times = _step_frames(viewer, DEFAULT_PLAYBACK_FRAMES)
                return _compute_stats(times)
            finally:
                viewer.close()

        result = benchmark(setup_and_step)

        assert result["mean"] < TARGET_FRAME_TIME_MS, (
            f"Position overlay: {result['mean']:.2f}ms exceeds target {TARGET_FRAME_TIME_MS:.1f}ms"
        )

    def test_bodyparts_skeleton_playback(
        self,
        benchmark,
        benchmark_env: Environment,
        benchmark_fields: NDArray[np.float32],
        frame_times: NDArray[np.float64],
        trajectory: NDArray[np.float64],
    ) -> None:
        """Benchmark playback with bodyparts + skeleton overlay.

        Baseline: 26.30ms mean (~38 fps)
        Target: <33.3ms (30 fps)
        """
        pytest.importorskip("napari")
        from neurospatial.animation.overlays import BodypartOverlay
        from neurospatial.animation.skeleton import Skeleton

        rng = np.random.default_rng(DEFAULT_SEED + 1)
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

        dim_ranges = list(benchmark_env.dimension_ranges or [])

        def setup_and_step():
            bodypart_data: dict[str, NDArray[np.float64]] = {}
            for bp_name in bodypart_names:
                offset = rng.uniform(-2, 2, size=2)
                jitter = rng.normal(0, 0.5, size=(DEFAULT_FRAMES, 2))
                bp_positions = trajectory + offset + jitter
                for dim in range(2):
                    dim_min, dim_max = dim_ranges[dim]
                    bp_positions[:, dim] = np.clip(
                        bp_positions[:, dim], dim_min, dim_max
                    )
                bodypart_data[bp_name] = bp_positions.copy()

            overlay = BodypartOverlay(data=bodypart_data, skeleton=skeleton)
            viewer = benchmark_env.animate_fields(
                benchmark_fields,
                frame_times=frame_times,
                overlays=[overlay],
                backend="napari",
                colormap="viridis",
            )
            try:
                times = _step_frames(viewer, DEFAULT_PLAYBACK_FRAMES)
                return _compute_stats(times)
            finally:
                viewer.close()

        result = benchmark(setup_and_step)

        assert result["mean"] < TARGET_FRAME_TIME_MS, (
            f"Bodyparts overlay: {result['mean']:.2f}ms exceeds target {TARGET_FRAME_TIME_MS:.1f}ms"
        )

    def test_head_direction_playback(
        self,
        benchmark,
        benchmark_env: Environment,
        benchmark_fields: NDArray[np.float32],
        frame_times: NDArray[np.float64],
        trajectory: NDArray[np.float64],
    ) -> None:
        """Benchmark playback with head direction overlay.

        Baseline: 18.44ms mean (~54 fps)
        Target: <33.3ms (30 fps)
        """
        pytest.importorskip("napari")
        from scipy.ndimage import gaussian_filter1d

        from neurospatial.animation.overlays import HeadDirectionOverlay

        rng = np.random.default_rng(DEFAULT_SEED + 2)

        # Compute head angles from trajectory velocity
        velocity = np.diff(trajectory, axis=0, prepend=trajectory[:1])
        head_angles = np.arctan2(velocity[:, 1], velocity[:, 0])
        head_angles += rng.normal(0, 0.1, size=DEFAULT_FRAMES)
        head_angles = gaussian_filter1d(head_angles, sigma=5)
        head_angles = np.arctan2(np.sin(head_angles), np.cos(head_angles))

        def setup_and_step():
            overlay = HeadDirectionOverlay(
                data=head_angles.copy(),
                color="yellow",
                length=3.0,
            )
            viewer = benchmark_env.animate_fields(
                benchmark_fields,
                frame_times=frame_times,
                overlays=[overlay],
                backend="napari",
                colormap="viridis",
            )
            try:
                times = _step_frames(viewer, DEFAULT_PLAYBACK_FRAMES)
                return _compute_stats(times)
            finally:
                viewer.close()

        result = benchmark(setup_and_step)

        assert result["mean"] < TARGET_FRAME_TIME_MS, (
            f"Head direction overlay: {result['mean']:.2f}ms exceeds target {TARGET_FRAME_TIME_MS:.1f}ms"
        )

    def test_events_decay_playback(
        self,
        benchmark,
        benchmark_env: Environment,
        benchmark_fields: NDArray[np.float32],
        frame_times: NDArray[np.float64],
        trajectory: NDArray[np.float64],
    ) -> None:
        """Benchmark playback with event overlay (decay mode).

        Baseline: 19.20ms mean (~52 fps)
        Target: <33.3ms (30 fps)
        """
        pytest.importorskip("napari")
        from neurospatial.animation.overlays import EventOverlay

        rng = np.random.default_rng(DEFAULT_SEED + 3)
        dim_ranges = list(benchmark_env.dimension_ranges or [])

        # Generate random spike events
        n_events = DEFAULT_FRAMES * 2
        event_frame_indices = rng.integers(0, DEFAULT_FRAMES, size=n_events)
        event_times_arr = event_frame_indices.astype(np.float64) / TARGET_FPS

        event_pos = np.zeros((n_events, 2))
        for i, frame_idx in enumerate(event_frame_indices):
            base_pos = trajectory[frame_idx]
            offset = rng.normal(0, 2, size=2)
            pos = base_pos + offset
            for dim in range(2):
                dim_min, dim_max = dim_ranges[dim]
                pos[dim] = np.clip(pos[dim], dim_min, dim_max)
            event_pos[i] = pos

        def setup_and_step():
            overlay = EventOverlay(
                event_times={"spikes": event_times_arr.copy()},
                event_positions={"spikes": event_pos.copy()},
                colors={"spikes": "cyan"},
                size=5.0,
                decay_frames=10,
            )
            viewer = benchmark_env.animate_fields(
                benchmark_fields,
                frame_times=frame_times,
                overlays=[overlay],
                backend="napari",
                colormap="viridis",
            )
            try:
                times = _step_frames(viewer, DEFAULT_PLAYBACK_FRAMES)
                return _compute_stats(times)
            finally:
                viewer.close()

        result = benchmark(setup_and_step)

        assert result["mean"] < TARGET_FRAME_TIME_MS, (
            f"Events overlay: {result['mean']:.2f}ms exceeds target {TARGET_FRAME_TIME_MS:.1f}ms"
        )

    def test_timeseries_playback(
        self,
        benchmark,
        benchmark_env: Environment,
        benchmark_fields: NDArray[np.float32],
        frame_times: NDArray[np.float64],
    ) -> None:
        """Benchmark playback with time series dock widget.

        Baseline: 18.49ms mean (~54 fps)
        Target: <33.3ms (30 fps)
        """
        pytest.importorskip("napari")
        from scipy.signal import butter, filtfilt

        from neurospatial.animation.overlays import TimeSeriesOverlay

        rng = np.random.default_rng(DEFAULT_SEED + 4)

        # Generate synthetic time series
        ts_data = rng.standard_normal(DEFAULT_FRAMES)
        b, a = butter(3, 0.1)
        ts_data = filtfilt(b, a, ts_data)
        ts_times = np.arange(DEFAULT_FRAMES) / TARGET_FPS

        def setup_and_step():
            overlay = TimeSeriesOverlay(
                data=ts_data.copy(),
                times=ts_times.copy(),
                label="LFP (mV)",
                color="cyan",
                window_seconds=2.0,
            )
            viewer = benchmark_env.animate_fields(
                benchmark_fields,
                frame_times=frame_times,
                overlays=[overlay],
                backend="napari",
                colormap="viridis",
            )
            try:
                times = _step_frames(viewer, DEFAULT_PLAYBACK_FRAMES)
                return _compute_stats(times)
            finally:
                viewer.close()

        result = benchmark(setup_and_step)

        assert result["mean"] < TARGET_FRAME_TIME_MS, (
            f"Time series overlay: {result['mean']:.2f}ms exceeds target {TARGET_FRAME_TIME_MS:.1f}ms"
        )

    def test_video_overlay_playback(
        self,
        benchmark,
        benchmark_env: Environment,
        benchmark_fields: NDArray[np.float32],
        frame_times: NDArray[np.float64],
    ) -> None:
        """Benchmark playback with video overlay (in-memory).

        Baseline: 18.39ms mean (~54 fps)
        Target: <33.3ms (30 fps)
        """
        pytest.importorskip("napari")
        from neurospatial.animation.overlays import VideoOverlay

        rng = np.random.default_rng(DEFAULT_SEED + 5)

        # Create synthetic video (100x100 pixels)
        video_height, video_width = 100, 100
        video_frames = rng.integers(
            50, 200, size=(DEFAULT_FRAMES, video_height, video_width), dtype=np.uint8
        )
        video_rgb = np.stack([video_frames] * 3, axis=-1)
        video_times = np.arange(DEFAULT_FRAMES) / TARGET_FPS

        def setup_and_step():
            overlay = VideoOverlay(
                source=video_rgb.copy(),
                times=video_times.copy(),
                alpha=0.3,
                z_order="below",
            )
            viewer = benchmark_env.animate_fields(
                benchmark_fields,
                frame_times=frame_times,
                overlays=[overlay],
                backend="napari",
                colormap="viridis",
            )
            try:
                times = _step_frames(viewer, DEFAULT_PLAYBACK_FRAMES)
                return _compute_stats(times)
            finally:
                viewer.close()

        result = benchmark(setup_and_step)

        assert result["mean"] < TARGET_FRAME_TIME_MS, (
            f"Video overlay: {result['mean']:.2f}ms exceeds target {TARGET_FRAME_TIME_MS:.1f}ms"
        )


# =============================================================================
# Combined Overlay Benchmarks
# =============================================================================


@pytest.mark.slow
@pytest.mark.xdist_group(name="napari_gui")
class TestNapariPlaybackCombinedOverlays:
    """Benchmark napari playback with multiple overlays combined.

    Target: <40ms mean frame time (25 fps acceptable).
    """

    def test_all_overlays_combined(
        self,
        benchmark,
        benchmark_env: Environment,
        benchmark_fields: NDArray[np.float32],
        frame_times: NDArray[np.float64],
        trajectory: NDArray[np.float64],
    ) -> None:
        """Benchmark playback with all 6 overlay types combined.

        Baseline: 47.38ms mean (~21 fps) - pre-optimization
        Target: <40ms mean (~25 fps) - post-optimization
        """
        pytest.importorskip("napari")
        from scipy.ndimage import gaussian_filter1d
        from scipy.signal import butter, filtfilt

        from neurospatial.animation.overlays import (
            BodypartOverlay,
            EventOverlay,
            HeadDirectionOverlay,
            PositionOverlay,
            TimeSeriesOverlay,
            VideoOverlay,
        )
        from neurospatial.animation.skeleton import Skeleton

        rng = np.random.default_rng(DEFAULT_SEED + 10)
        dim_ranges = list(benchmark_env.dimension_ranges or [])

        # Pre-compute all overlay data
        # Position overlay
        position_data = trajectory.copy()

        # Bodyparts overlay
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
        for bp_name in bodypart_names:
            offset = rng.uniform(-2, 2, size=2)
            jitter = rng.normal(0, 0.5, size=(DEFAULT_FRAMES, 2))
            bp_positions = trajectory + offset + jitter
            for dim in range(2):
                dim_min, dim_max = dim_ranges[dim]
                bp_positions[:, dim] = np.clip(bp_positions[:, dim], dim_min, dim_max)
            bodypart_data[bp_name] = bp_positions.copy()

        # Head direction
        velocity = np.diff(trajectory, axis=0, prepend=trajectory[:1])
        head_angles = np.arctan2(velocity[:, 1], velocity[:, 0])
        head_angles += rng.normal(0, 0.1, size=DEFAULT_FRAMES)
        head_angles = gaussian_filter1d(head_angles, sigma=5)
        head_angles = np.arctan2(np.sin(head_angles), np.cos(head_angles))

        # Events
        n_events = DEFAULT_FRAMES * 2
        event_frame_indices = rng.integers(0, DEFAULT_FRAMES, size=n_events)
        event_times_arr = event_frame_indices.astype(np.float64) / TARGET_FPS
        event_pos = np.zeros((n_events, 2))
        for i, frame_idx in enumerate(event_frame_indices):
            base_pos = trajectory[frame_idx]
            offset = rng.normal(0, 2, size=2)
            pos = base_pos + offset
            for dim in range(2):
                dim_min, dim_max = dim_ranges[dim]
                pos[dim] = np.clip(pos[dim], dim_min, dim_max)
            event_pos[i] = pos

        # Time series
        ts_data = rng.standard_normal(DEFAULT_FRAMES)
        b, a = butter(3, 0.1)
        ts_data = filtfilt(b, a, ts_data)
        ts_times = np.arange(DEFAULT_FRAMES) / TARGET_FPS

        # Video
        video_height, video_width = 100, 100
        video_frames = rng.integers(
            50, 200, size=(DEFAULT_FRAMES, video_height, video_width), dtype=np.uint8
        )
        video_rgb = np.stack([video_frames] * 3, axis=-1)
        video_times = np.arange(DEFAULT_FRAMES) / TARGET_FPS

        def setup_and_step():
            overlays = [
                PositionOverlay(
                    data=position_data.copy(), color="red", size=12.0, trail_length=15
                ),
                BodypartOverlay(
                    data={k: v.copy() for k, v in bodypart_data.items()},
                    skeleton=skeleton,
                ),
                HeadDirectionOverlay(
                    data=head_angles.copy(), color="yellow", length=3.0
                ),
                EventOverlay(
                    event_times={"spikes": event_times_arr.copy()},
                    event_positions={"spikes": event_pos.copy()},
                    colors={"spikes": "cyan"},
                    size=5.0,
                    decay_frames=10,
                ),
                TimeSeriesOverlay(
                    data=ts_data.copy(),
                    times=ts_times.copy(),
                    label="LFP (mV)",
                    color="cyan",
                    window_seconds=2.0,
                ),
                VideoOverlay(
                    source=video_rgb.copy(),
                    times=video_times.copy(),
                    alpha=0.3,
                    z_order="below",
                ),
            ]

            viewer = benchmark_env.animate_fields(
                benchmark_fields,
                frame_times=frame_times,
                overlays=overlays,
                backend="napari",
                colormap="viridis",
            )
            try:
                times = _step_frames(viewer, DEFAULT_PLAYBACK_FRAMES)
                return _compute_stats(times)
            finally:
                viewer.close()

        result = benchmark(setup_and_step)

        # Use acceptable target for combined overlays
        assert result["mean"] < ACCEPTABLE_FRAME_TIME_MS, (
            f"All overlays combined: {result['mean']:.2f}ms exceeds acceptable {ACCEPTABLE_FRAME_TIME_MS:.1f}ms"
        )


# =============================================================================
# Field Size Variation Benchmarks
# =============================================================================


@pytest.mark.slow
@pytest.mark.xdist_group(name="napari_gui")
class TestNapariPlaybackFieldSizes:
    """Benchmark napari playback with different field sizes.

    Tests performance scaling with environment size.
    """

    @pytest.fixture(scope="class")
    def _env_100x100(self) -> Environment:
        """100x100 grid environment (~10,000 bins)."""
        from neurospatial import Environment

        x = np.linspace(0, 100, 101)
        y = np.linspace(0, 100, 101)
        xx, yy = np.meshgrid(x, y)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        return Environment.from_samples(positions, bin_size=1.0)

    @pytest.fixture(scope="class")
    def _fields_100x100(self, _env_100x100: Environment) -> NDArray[np.float32]:
        """Fields for 100x100 environment."""
        rng = np.random.default_rng(DEFAULT_SEED)
        return rng.random((DEFAULT_FRAMES, _env_100x100.n_bins)).astype(np.float32)

    def test_field_size_100x100(
        self,
        benchmark,
        _env_100x100: Environment,
        _fields_100x100: NDArray[np.float32],
        frame_times: NDArray[np.float64],
    ) -> None:
        """Benchmark playback with 100x100 grid (~10,000 bins).

        Tests larger field rendering performance.

        Baseline: N/A (scaling test)
        Target: <40ms (25 fps acceptable)
        """
        pytest.importorskip("napari")

        def setup_and_step():
            viewer = _env_100x100.animate_fields(
                _fields_100x100,
                frame_times=frame_times,
                backend="napari",
                colormap="viridis",
            )
            try:
                times = _step_frames(viewer, DEFAULT_PLAYBACK_FRAMES)
                return _compute_stats(times)
            finally:
                viewer.close()

        result = benchmark(setup_and_step)

        # Larger fields may be slower, use acceptable target
        assert result["mean"] < ACCEPTABLE_FRAME_TIME_MS, (
            f"100x100 field: {result['mean']:.2f}ms exceeds acceptable {ACCEPTABLE_FRAME_TIME_MS:.1f}ms"
        )


# =============================================================================
# Frame Count Variation Benchmarks
# =============================================================================


@pytest.mark.slow
@pytest.mark.xdist_group(name="napari_gui")
class TestNapariPlaybackFrameCounts:
    """Benchmark napari playback with different frame counts.

    Tests performance with varying animation lengths.
    """

    def test_frame_count_1000(
        self,
        benchmark,
        benchmark_env: Environment,
    ) -> None:
        """Benchmark playback with 1000 frames.

        Tests performance with longer animations.

        Baseline: N/A (scaling test)
        Target: <33.3ms (30 fps)
        """
        pytest.importorskip("napari")
        from benchmark_datasets import BenchmarkConfig, create_benchmark_fields

        n_frames = 1000
        playback_frames = 100  # Still only step through 100 for timing

        config = BenchmarkConfig(
            name="frame_count_1000",
            n_frames=n_frames,
            grid_size=DEFAULT_GRID_SIZE,
        )
        fields = create_benchmark_fields(benchmark_env, config, seed=DEFAULT_SEED)
        frame_times_local = np.arange(n_frames) / TARGET_FPS

        def setup_and_step():
            viewer = benchmark_env.animate_fields(
                fields,
                frame_times=frame_times_local,
                backend="napari",
                colormap="viridis",
            )
            try:
                times = _step_frames(viewer, playback_frames)
                return _compute_stats(times)
            finally:
                viewer.close()

        result = benchmark(setup_and_step)

        assert result["mean"] < TARGET_FRAME_TIME_MS, (
            f"1000 frames: {result['mean']:.2f}ms exceeds target {TARGET_FRAME_TIME_MS:.1f}ms"
        )
