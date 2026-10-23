#!/usr/bin/env python
"""Reproducible benchmarks of real Animation.show viewers (no mocks).

Run one backend per process, from the checkout you intend to measure::

    uv run python scripts/benchmark_animation_scenes.py --backend napari --smoke
    uv run python scripts/benchmark_animation_scenes.py --backend fastplotlib \
        --dimensions 3 --grid-size 32 --panels --output /tmp/animation.json

A display and the backend's optional dependencies are required. An offscreen
fastplotlib canvas can measure synchronous capture, but cannot measure GUI
playback. ``--playback-seconds 0`` explicitly disables playback. Capture timings
include source reads, preparation, rendering, GPU synchronization and RGB
readback; their throughput is NOT a GUI playback FPS measurement. GUI metrics
observe native draw callbacks and recording-time controls, not screen presents.
Run the same command at each revision to compare JSON artifacts. First imports,
viewer creation and first draw are included in setup, not warmed seek timings.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import ExitStack, suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


class CountingFieldSource:
    """Generate spatial rows on demand, counting actual source row reads.

    Memory scales with bins rather than frames x bins. Delay simulates an
    on-disk/network source. Counts include repeated reads and are thread-safe.
    """

    def __init__(
        self, centers: np.ndarray, frames: int, seed: int, delay_ms: float = 0
    ) -> None:
        self.shape = (frames, len(centers))
        self.dtype = np.dtype(np.float32)
        self.ndim = 2
        self.centers = np.asarray(centers, dtype=np.float32)
        self.extent = np.maximum(np.ptp(self.centers, axis=0), 1)
        self.origin = self.centers.min(axis=0)
        self.phase = np.random.default_rng(seed).uniform(0, 2 * np.pi)
        self.delay_s = delay_ms / 1000
        self._lock = threading.Lock()
        self._reads = 0

    def __len__(self) -> int:
        return self.shape[0]

    @property
    def reads(self) -> int:
        with self._lock:
            return self._reads

    def __getitem__(self, key: Any) -> np.ndarray:
        if not np.isscalar(key):
            indices = np.arange(len(self))[key]
            if not len(indices):
                return np.empty((0, self.shape[1]), dtype=self.dtype)
            return np.stack([self[int(index)] for index in indices])
        index = int(key)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self.delay_s:
            time.sleep(self.delay_s)
        angle = self.phase + 2 * np.pi * index / len(self)
        center = self.origin + self.extent * (
            0.5 + 0.25 * np.cos(angle + np.arange(self.centers.shape[1]))
        )
        row = np.exp(
            -np.square((self.centers - center) / (self.extent * 0.15)).sum(axis=1) / 2
        ).astype(np.float32)
        row[0] = np.nan  # Preserve unsupported observations in every workload.
        with self._lock:
            self._reads += 1
        return row


def timing_summary(samples_ms: list[float]) -> dict[str, Any]:
    """Summarize latencies without claiming a GUI playback frame rate."""
    if not samples_ms:
        return {"count": 0, "median_ms": None, "p95_ms": None, "p99_ms": None}
    values = np.asarray(samples_ms)
    return {
        "count": len(samples_ms),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
        "samples_ms": samples_ms,
    }


def rss_bytes() -> int | None:
    """Current process RSS, including native allocations, when psutil exists."""
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except ImportError:
        return None


def metadata() -> dict[str, Any]:
    """Describe the process, hardware, software and graphics selection inputs."""
    versions = {}
    for package in (
        "neurospatial",
        "numpy",
        "napari",
        "vispy",
        "fastplotlib",
        "pygfx",
        "wgpu",
        "rendercanvas",
        "PyQt6",
        "PySide6",
        "psutil",
    ):
        with suppress(importlib.metadata.PackageNotFoundError):
            versions[package] = importlib.metadata.version(package)
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    hardware: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
    }
    try:
        import psutil

        hardware["ram_bytes"] = int(psutil.virtual_memory().total)
    except ImportError:
        hardware["ram_bytes"] = None
    import neurospatial

    return {
        "source_path": str(Path(neurospatial.__file__).resolve()),
        "utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "executable": sys.executable,
        "revision": revision,
        "working_tree_dirty": dirty,
        "versions": versions,
        "hardware": hardware,
        "environment": {
            name: os.environ.get(name)
            for name in (
                "QT_QPA_PLATFORM",
                "RENDERCANVAS_FORCE_OFFSCREEN",
                "RENDERCANVAS_BACKEND",
                "WGPU_BACKEND_TYPE",
                "MPLBACKEND",
            )
        },
    }


def make_scene(args: argparse.Namespace) -> tuple[Any, CountingFieldSource]:
    """Construct the same lazy scientific workload for both native backends."""
    from neurospatial import Environment
    from neurospatial.animation import Animation, AnimationField, PositionOverlay
    from neurospatial.animation.overlays import SpikeRasterOverlay, TimeSeriesOverlay

    mask = np.ones((args.grid_size,) * args.dimensions, dtype=bool)
    mask.flat[-1] = False
    edges = tuple(np.arange(args.grid_size + 1, dtype=float) for _ in mask.shape)
    env = Environment.from_grid_mask(mask, edges, units="cm", frame="arena")
    times = 100 + np.arange(args.frames) / args.fps
    phase = np.linspace(0, 2 * np.pi, args.frames)
    positions = args.grid_size * (
        0.5 + 0.25 * np.cos(phase[:, None] + np.arange(args.dimensions))
    )
    overlays = [PositionOverlay(positions, times, size=14, trail_length=8)]
    if args.panels:
        overlays.extend(
            [
                TimeSeriesOverlay(np.sin(phase), times, label="Signal"),
                SpikeRasterOverlay([times[unit % 4 :: 4] for unit in range(16)]),
            ]
        )
    source = CountingFieldSource(env.bin_centers, args.frames, args.seed, args.delay_ms)
    field = AnimationField(source, times, vmin=0, vmax=1, label="Activity")
    return Animation(env, field, overlays=overlays, title="Animation benchmark"), source


class NativeDriver:
    """Keep backend-specific event processing outside the benchmark workload.

    Setup/seek/cleanup use AnimationView. Capture uses the single internal
    backend boundary because AnimationView currently exposes export, not a
    public in-memory capture session. Never replace native renderers with mocks.
    """

    def __init__(self, view: Any, args: argparse.Namespace) -> None:
        self.view, self.backend = view, args.backend
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._loop_task: Any = None
        if self.backend == "napari":
            from qtpy.QtWidgets import QApplication, QDoubleSpinBox, QPushButton

            self.app = QApplication.instance()
            view.native.window.resize(args.width, args.height)
            view.native.reset_view()
            controls = view.native.window._qt_window.findChildren(QPushButton)
            self.play = next(button for button in controls if button.text() == "Play")
            self.time_control = next(
                control
                for control in view.native.window._qt_window.findChildren(
                    QDoubleSpinBox
                )
                if control.suffix() == " s"
            )
            self.canvas = view.native.window._qt_viewer.canvas
            self.gui_available = True
        else:
            self.canvas = view.native.figure.canvas
            self.canvas.set_logical_size(args.width, args.height)
            self.gui_available = "offscreen" not in type(self.canvas).__module__

    def capture_session(self) -> Any:
        return self.view._backend.capture_session()

    def toggle_play(self) -> None:
        if self.backend == "napari":
            self.play.click()
        else:
            # Exercise the same public renderer key event as the space shortcut.
            from pygfx import KeyboardEvent

            self.view.native.figure.renderer.dispatch_event(
                KeyboardEvent("key_down", key="Space")
            )

    def request_seek(self, time: float) -> None:
        """Use the native GUI time control, including its asynchronous read path."""
        if self.backend == "napari":
            self.time_control.setValue(time)
        else:
            self.view.seek(time)

    def stop_play(self) -> None:
        if self.backend == "napari":
            if self.play.text() == "Pause":
                self.play.click()
        else:
            # NDWidget currently exposes playback inspection through the
            # version-checked compatibility boundary used by the backend.
            from neurospatial.environment._fastplotlib_compat import playback_state

            if playback_state(self.view.native.widget).get("time", False):
                self.toggle_play()

    def run_events(self, seconds: float, observe: Any = None) -> None:
        canvas_module = type(self.canvas).__module__
        if self.backend == "napari" or "qt" in canvas_module:
            from qtpy.QtCore import QEventLoop, QTimer

            loop = QEventLoop()
            timer = QTimer()
            if observe is not None:
                timer.timeout.connect(observe)
                timer.start(5)
            QTimer.singleShot(max(1, round(seconds * 1000)), loop.quit)
            try:
                loop.exec()
            finally:
                timer.stop()
        elif self.gui_available:
            from rendercanvas.auto import loop

            # Keep one asyncio loop across phases. rendercanvas.loop.stop()
            # closes every native canvas, so it cannot end a timed phase.
            if self._event_loop is None:
                self._event_loop = asyncio.new_event_loop()
                self._loop_task = self._event_loop.create_task(loop.run_async())

            async def observe_until() -> None:
                end = time.perf_counter() + seconds
                while time.perf_counter() < end:
                    if observe is not None:
                        observe()
                    await asyncio.sleep(0.005)

            self._event_loop.run_until_complete(observe_until())
        elif seconds:
            # An offscreen canvas has no GUI loop; do not simulate playback.
            raise RuntimeError("Offscreen canvas has no GUI event loop.")

    def close(self) -> None:
        try:
            self.view.close()
        finally:
            if self._event_loop is not None:
                pending = asyncio.all_tasks(self._event_loop)
                for task in pending:
                    task.cancel()
                if pending:
                    self._event_loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
                self._event_loop.run_until_complete(
                    self._event_loop.shutdown_asyncgens()
                )
                self._event_loop.close()

    def connect_draw(self, callback: Any) -> Any:
        if self.backend == "napari":
            self.canvas.events.draw.connect(callback)
            return lambda: self.canvas.events.draw.disconnect(callback)
        self.canvas.add_event_handler(callback, "before_draw")
        return lambda: self.canvas.remove_event_handler(callback, "before_draw")


def measure_playback(
    driver: NativeDriver, source: CountingFieldSource, seconds: float
) -> dict[str, Any]:
    """Observe actual native playback without synchronizing or reading pixels."""
    if not seconds or not driver.gui_available:
        return {
            "status": "disabled" if not seconds else "unavailable_offscreen",
            "note": "No GUI playback result is inferred from synchronous captures.",
        }
    draws: list[float] = []
    draw_recording_times: list[float] = []
    polls: list[float] = []
    recording_times: list[float] = []

    def observe() -> None:
        polls.append(time.perf_counter())
        recording_times.append(driver.view.time)

    def on_draw(event: Any) -> None:
        draws.append(time.perf_counter())
        draw_recording_times.append(driver.view.time)

    disconnect = driver.connect_draw(on_draw)
    reads = source.reads
    started = time.perf_counter()
    driver.toggle_play()
    try:
        driver.run_events(seconds, observe)
    finally:
        driver.stop_play()
        disconnect()
    elapsed = time.perf_counter() - started
    return {
        "status": "measured",
        "wall_seconds": elapsed,
        "draw_callback_count": len(draws),
        "recording_times_at_draw": draw_recording_times,
        "distinct_recording_times_at_draw": len(set(draw_recording_times)),
        "draw_callbacks_per_second": len(draws) / elapsed,
        "draw_interval": timing_summary((np.diff(draws) * 1000).tolist()),
        "event_loop_heartbeat_interval": timing_summary(
            (np.diff(polls) * 1000).tolist()
        ),
        "recording_seconds_advanced": (
            recording_times[-1] - recording_times[0] if recording_times else 0
        ),
        "source_row_reads": source.reads - reads,
        "note": "Native draw callbacks and control time; neither measures GPU presentation or guarantees a new spatial field per draw.",
    }


def measure_burst(
    driver: NativeDriver,
    scene: Any,
    source: CountingFieldSource,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Measure request responsiveness and source work during rapid lazy seeks."""
    if not args.burst_seeks or not driver.gui_available:
        return {
            "status": "disabled" if not args.burst_seeks else "unavailable_offscreen"
        }
    indices = np.random.default_rng(args.seed + 1).integers(
        0, args.frames, args.burst_seeks
    )
    before = source.reads
    latencies = []
    for index in indices:
        started = time.perf_counter()
        driver.request_seek(float(scene.frame_times[index]))
        latencies.append((time.perf_counter() - started) * 1000)
    enqueue_reads = source.reads - before
    driver.run_events(args.settle_seconds)
    return {
        "status": "measured",
        "requests": len(indices),
        "request_latency": timing_summary(latencies),
        "source_row_reads_during_enqueue": enqueue_reads,
        "source_row_reads_including_settle": source.reads - before,
        "settle_seconds": args.settle_seconds,
        "final_requested_time": float(scene.frame_times[indices[-1]]),
        "final_control_time": driver.view.time,
        "note": "Native GUI time-control requests; read counts expose coalescing work, while control time does not prove displayed-frame completion.",
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Return JSON-compatible metrics and always close scenes/native viewers."""
    result = {"schema_version": 1, "metadata": metadata(), "options": vars(args).copy()}
    result["options"]["output"] = str(args.output) if args.output else None
    result["rss_bytes"] = {"before_scene": rss_bytes()}
    started = time.perf_counter()
    scene, source = make_scene(args)
    result["scene_creation_s"] = time.perf_counter() - started
    with scene:
        result["scenario"] = {
            "dimensions": args.dimensions,
            "grid_shape": [args.grid_size] * args.dimensions,
            "bins": scene.env.n_bins,
            "frames": args.frames,
            "source": "generated_lazy_rows",
            "source_delay_ms": args.delay_ms,
            "panels": ["position", "signal", "spike_raster"]
            if args.panels
            else ["position"],
            "dtype": str(source.dtype),
            "fixed_color_limits": [0, 1],
            "requested_logical_resolution": [args.width, args.height],
        }
        options = {"fps": args.fps} if args.backend == "napari" else {}
        if args.backend == "fastplotlib" and args.field_rendering != "auto":
            options["field_rendering"] = args.field_rendering
        started = time.perf_counter()
        with (
            scene.show(backend=args.backend, **options) as view,
            ExitStack() as native_stack,
        ):
            driver = NativeDriver(view, args)
            native_stack.callback(driver.close)
            with driver.capture_session() as capture:
                first = capture(float(scene.frame_times[0]))
            result["setup_to_first_capture_s"] = time.perf_counter() - started
            result["setup_source_row_reads"] = source.reads
            if args.backend == "fastplotlib":
                result["graphics"] = {
                    "adapter": dict(view.native.figure.renderer.device.adapter.info),
                    "canvas_class": f"{type(driver.canvas).__module__}.{type(driver.canvas).__name__}",
                }
            else:
                from vispy.gloo import gl

                result["graphics"] = {
                    "renderer": str(gl.glGetParameter(gl.GL_RENDERER)),
                    "vendor": str(gl.glGetParameter(gl.GL_VENDOR)),
                    "version": str(gl.glGetParameter(gl.GL_VERSION)),
                }
            result["actual_rgb_resolution"] = [first.shape[1], first.shape[0]]
            result["rss_bytes"]["after_first_capture"] = rss_bytes()
            # GUI is measured before capture-session suspension. Warm its normal
            # event loop separately; no draw/readback is forced in this phase.
            if driver.gui_available:
                driver.run_events(0.1)
            view.seek(float(scene.frame_times[0]))
            result["gui_playback"] = measure_playback(
                driver, source, args.playback_seconds
            )
            result["rapid_seek"] = measure_burst(driver, scene, source, args)
            measured = {}
            for mode in ("sequential", "random"):
                indices = (
                    np.arange(args.samples) % args.frames
                    if mode == "sequential"
                    else np.random.default_rng(args.seed).integers(
                        0, args.frames, args.samples
                    )
                )
                with driver.capture_session() as capture:
                    for index in range(args.warmup):
                        capture(float(scene.frame_times[index % args.frames]))
                    before = source.reads
                    latencies = []
                    for index in indices:
                        started = time.perf_counter()
                        image = capture(float(scene.frame_times[index]))
                        latencies.append((time.perf_counter() - started) * 1000)
                    if (
                        image.dtype != np.uint8
                        or image.ndim != 3
                        or image.shape[2] != 3
                    ):
                        raise RuntimeError(
                            "Native capture did not return RGB uint8 pixels."
                        )
                    measured[mode] = {
                        **timing_summary(latencies),
                        "frame_indices": indices.tolist(),
                        "source_row_reads": source.reads - before,
                        "image_standard_deviation": float(image.std()),
                    }
            result["synchronous_seek_gpu_capture"] = measured
            result["rss_bytes"]["after_measurements"] = rss_bytes()
            result["total_source_row_reads"] = source.reads
    result["rss_bytes"]["after_close"] = rss_bytes()
    result["metric_notes"] = {
        "rss": "Current process resident memory, including native allocations; not GPU VRAM or peak RSS.",
        "capture": "Warmed synchronous source read + seek + GPU render + RGB readback; not GUI playback FPS.",
        "setup": "Animation.show through first native RGB capture, including imports and compilation; scene construction reported separately.",
    }
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    result.add_argument("--backend", choices=("napari", "fastplotlib"), required=True)
    result.add_argument("--dimensions", type=int, choices=(2, 3), default=2)
    result.add_argument(
        "--grid-size", type=int, default=32, help="Bins per axis; one corner excluded."
    )
    result.add_argument("--frames", type=int, default=240)
    result.add_argument("--samples", type=int, default=60)
    result.add_argument("--warmup", type=int, default=5)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--fps", type=float, default=30)
    result.add_argument("--panels", action="store_true")
    result.add_argument("--width", type=int, default=800)
    result.add_argument("--height", type=int, default=600)
    result.add_argument(
        "--field-rendering",
        choices=("auto", "points", "mip", "iso", "slice"),
        default="auto",
        help="Fastplotlib 3D field representation.",
    )
    result.add_argument(
        "--delay-ms", type=float, default=0, help="Delay each lazy source row read."
    )
    result.add_argument("--playback-seconds", type=float, default=2)
    result.add_argument(
        "--burst-seeks",
        type=int,
        default=0,
        help="Rapid public seek requests to inspect responsiveness/coalescing.",
    )
    result.add_argument("--settle-seconds", type=float, default=0.5)
    result.add_argument(
        "--output",
        type=Path,
        help="JSON artifact path; JSON is also printed to stdout.",
    )
    result.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny 4-bin-per-axis, 12-frame, four-capture workload; disables GUI playback.",
    )
    return result


def main() -> None:
    arg_parser = parser()
    args = arg_parser.parse_args()
    if args.smoke:
        args.grid_size, args.frames, args.samples, args.warmup = 4, 12, 4, 2
        args.playback_seconds = 0
    for name in ("grid_size", "frames", "samples", "width", "height"):
        if getattr(args, name) <= (1 if name in ("grid_size", "frames") else 0):
            arg_parser.error(
                f"--{name.replace('_', '-')} must be greater than {1 if name in ('grid_size', 'frames') else 0}."
            )
    if args.warmup < 0 or args.burst_seeks < 0:
        arg_parser.error("warmup and burst-seeks must be nonnegative.")
    for name in ("fps", "delay_ms", "playback_seconds", "settle_seconds"):
        value = getattr(args, name)
        if not np.isfinite(value) or value < 0 or (name == "fps" and value == 0):
            arg_parser.error(
                f"--{name.replace('_', '-')} must be finite and {'positive' if name == 'fps' else 'nonnegative'}."
            )
    if args.field_rendering != "auto" and (
        args.backend != "fastplotlib" or args.dimensions != 3
    ):
        arg_parser.error("--field-rendering requires fastplotlib and --dimensions 3.")
    # Keep napari's cache/config writes isolated and clean them after all views
    # close. This also makes the benchmark usable in restricted workspaces.
    with ExitStack() as stack:
        if args.backend == "napari":
            import napari.utils._platformdirs as napari_dirs

            cache = Path(
                stack.enter_context(
                    tempfile.TemporaryDirectory(prefix="animation-benchmark-")
                )
            )
            for name in (
                "user_cache_dir",
                "user_config_dir",
                "user_data_dir",
                "user_state_dir",
                "user_log_dir",
            ):
                directory = cache / name
                directory.mkdir()
                original = getattr(napari_dirs, name)
                setattr(napari_dirs, name, lambda directory=directory: str(directory))
                stack.callback(setattr, napari_dirs, name, original)
        data = run_benchmark(args)
    serialized = json.dumps(data, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")
    print(serialized)


if __name__ == "__main__":
    main()
