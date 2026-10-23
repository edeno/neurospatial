# Performance Profiling for Animation Views

This page describes how to measure and profile the interactive animation views
(`Animation.show` and `Environment.animate_fields` with `backend="napari"` or
`"fastplotlib"`) and file exports.

## Benchmark a viewer

`scripts/benchmark_animation_scenes.py` opens a real napari or fastplotlib
view on a reproducible lazy-source workload and writes a JSON record of
versions, workload, capture resolution, source reads, resident memory and
warmed median/p95/p99 seek latency:

```bash
# Small rendering check
uv run python scripts/benchmark_animation_scenes.py --backend napari --smoke

# A larger workload with time panels, saved for comparison between revisions
uv run python scripts/benchmark_animation_scenes.py --backend napari \
  --grid-size 64 --frames 2000 --samples 60 --panels --output /tmp/napari.json

# Slow sources and rapid scrubbing (needs a visible GUI)
uv run python scripts/benchmark_animation_scenes.py --backend fastplotlib \
  --delay-ms 80 --playback-seconds 1 --burst-seeks 12
```

Capture timing (synchronous seek plus GPU readback) is not playback FPS; GUI
playback and event-loop responsiveness are reported separately and are
omitted for offscreen runs. Compare runs only with matching workloads and
actual resolutions.

## Profile

### Napari perfmon

Napari writes Chrome Trace files for Qt events and the callables you list.
`NAPARI_PERFMON` takes `1` or the path of a config file, not an output path:

```bash
# Enable performance logging (no file)
NAPARI_PERFMON=1 uv run python scripts/benchmark_animation_scenes.py --backend napari --smoke

# Trace to the file named by "trace_file_on_start" in the config
NAPARI_PERFMON=/path/to/perfmon_config.json uv run python my_animation_script.py
```

List the functions to trace under `"callable_lists"` in the config, for
example `neurospatial.animation.backends._napari_scene.NapariScene._prepare`
(reading and preparing a frame on the worker thread) and
`neurospatial.animation.backends._napari_scene.NapariScene._apply` (updating
layers on the GUI thread). Open the trace in `chrome://tracing` or
[Speedscope](https://www.speedscope.app/) and look for long blocks on the main
thread, Qt event processing time and layer update durations.

### py-spy

A sampling profiler that shows the whole call stack, including native code,
with low overhead and no code changes:

```bash
pip install py-spy
py-spy record -o profile.svg -- uv run python scripts/benchmark_animation_scenes.py --backend napari --smoke
py-spy top -- uv run python my_animation_script.py
```

### cProfile

Deterministic function-level timing, for example of a file export:

```bash
uv run python -m cProfile -o /tmp/export.prof my_export_script.py
uv run python -c "import pstats; pstats.Stats('/tmp/export.prof').sort_stats('cumulative').print_stats(30)"
```

`snakeviz /tmp/export.prof` shows the same profile in a browser.

### Memory

```bash
pip install memory_profiler
mprof run uv run python my_animation_script.py
mprof plot
```

## Where time goes

- **Reading rows.** Views and exports read field rows lazily, so a
  memory-mapped posterior stays on disk; slow storage shows up as source-read
  time. Napari prepares rows on a worker thread and skips frames during
  playback when reads are slower than real time.
- **Drawing.** The notebook widget and file exports draw each frame with
  Matplotlib; time grows with the number of bins, panels and overlays, and
  with `dpi`. Napari and fastplotlib update GPU layers.
- **Overlays.** Long position trails, many pose keypoints and video decoding
  add work to every frame.

## Make it faster

- Export a window instead of the whole session:
  `Animation.export(path, time_range=(start, stop))`.
- Render long exports in several processes with `n_workers=None`; frames are
  identical to a serial export.
- Lower `dpi`, shorten `trail_length`, or drop overlays you do not need.
- Subsample a high-rate field for an overview:
  `subsample_frames(fields, source_fps=500, target_fps=30)`, with
  `frame_times` subsampled the same way.
- Use napari or fastplotlib rather than the notebook widget for long
  recordings.

## References

- [Napari Perfmon Documentation](https://napari.org/stable/howtos/perfmon.html)
- [py-spy Documentation](https://github.com/benfred/py-spy)
- [Speedscope](https://www.speedscope.app/)
